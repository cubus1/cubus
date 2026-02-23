//! CogRecord v3: hybrid binary + phase containers.
//!
//! Binary containers (META, BTREE) → fast rejection via POPCNT (hash-table behavior)
//! Phase containers (CAM, EMBED) → spatial navigation via Wasserstein/circular (space behavior)
//!
//! ## Container Layout
//!
//! | Container | Name  | Mode   | Bind   | Distance    | Sorted? |
//! |-----------|-------|--------|--------|-------------|---------|
//! | 0         | META  | Binary | XOR    | Hamming     | N/A     |
//! | 1         | CAM   | Phase  | ADD    | Wasserstein | YES     |
//! | 2         | BTREE | Binary | XOR    | Hamming     | N/A     |
//! | 3         | EMBED | Phase  | ADD    | Circular    | NO      |

use crate::phase::{
    circular_distance_i8, phase_bind_i8, phase_unbind_i8, sort_phase_vector, unsort_phase_vector,
    wasserstein_sorted_i8,
};
use numrus_rs::NumArrayU8;

/// Container size in bytes.
pub const CONTAINER_BYTES: usize = 2048;

/// Thresholds for the 4-channel hybrid sweep.
#[derive(Clone, Debug)]
pub struct HybridThresholds {
    /// Binary: max acceptable Hamming distance for META.
    pub meta_hamming: u64,
    /// Spatial: max acceptable Wasserstein-1 for CAM (sorted).
    pub cam_wasserstein: u64,
    /// Binary: max acceptable Hamming distance for BTREE.
    pub btree_hamming: u64,
    /// Phase: max acceptable circular distance for EMBED.
    pub embed_circular: u64,
}

/// Distances returned by successful sweep (all passed thresholds).
#[derive(Clone, Debug)]
pub struct HybridDistances {
    pub meta_hamming: u64,
    pub cam_wasserstein: u64,
    pub btree_hamming: u64,
    pub embed_circular: u64,
}

/// CogRecord v3: hybrid binary + phase containers.
///
/// ONE RECORD = ONE FACT (3-5 bundled items max).
#[derive(Clone)]
pub struct CogRecordV3 {
    /// Container 0: Codebook ID, content type, metadata flags.
    /// BINARY. XOR bind. VPOPCNTDQ distance.
    pub meta: NumArrayU8,

    /// Container 1: Content-addressable memory.
    /// PHASE INT8, PRE-SORTED. ADD bind. Wasserstein distance.
    pub cam: Vec<u8>,

    /// Container 2: Graph position, B-tree path.
    /// BINARY. XOR bind. VPOPCNTDQ distance.
    pub btree: NumArrayU8,

    /// Container 3: Dense embedding.
    /// PHASE INT8, UNSORTED. ADD bind. Circular distance.
    pub embed: Vec<u8>,

    /// Sort permutation for CAM (allows reversing the sort for unbinding).
    pub cam_perm: Option<Vec<u16>>,
}

impl CogRecordV3 {
    /// Construct from raw data. Sorts CAM container and stores permutation.
    pub fn new(meta: &[u8], cam: &[u8], btree: &[u8], embed: &[u8]) -> Self {
        assert_eq!(meta.len(), CONTAINER_BYTES);
        assert_eq!(cam.len(), CONTAINER_BYTES);
        assert_eq!(btree.len(), CONTAINER_BYTES);
        assert_eq!(embed.len(), CONTAINER_BYTES);

        let (sorted_cam, cam_perm) = sort_phase_vector(cam);

        Self {
            meta: NumArrayU8::new(meta.to_vec()),
            cam: sorted_cam,
            btree: NumArrayU8::new(btree.to_vec()),
            embed: embed.to_vec(),
            cam_perm: Some(cam_perm),
        }
    }

    /// Construct with pre-sorted CAM (e.g., from deserialization).
    pub fn from_sorted(
        meta: NumArrayU8,
        sorted_cam: Vec<u8>,
        btree: NumArrayU8,
        embed: Vec<u8>,
        cam_perm: Option<Vec<u16>>,
    ) -> Self {
        Self {
            meta,
            cam: sorted_cam,
            btree,
            embed,
            cam_perm,
        }
    }

    /// Create a zeroed CogRecordV3.
    pub fn zeros() -> Self {
        Self {
            meta: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
            cam: vec![0u8; CONTAINER_BYTES],
            btree: NumArrayU8::new(vec![0u8; CONTAINER_BYTES]),
            embed: vec![0u8; CONTAINER_BYTES],
            cam_perm: None,
        }
    }

    /// 4-channel hybrid sweep with short-circuit rejection.
    ///
    /// Order: META (cheapest) → BTREE → CAM → EMBED (most expensive).
    /// Returns `None` if any channel exceeds its threshold.
    pub fn hybrid_sweep(
        &self,
        other: &Self,
        thresholds: &HybridThresholds,
    ) -> Option<HybridDistances> {
        // Stage 1: META — binary Hamming
        let meta_dist =
            numrus_core::simd::hamming_distance(self.meta.data_slice(), other.meta.data_slice());
        if meta_dist > thresholds.meta_hamming {
            return None;
        }

        // Stage 2: BTREE — binary Hamming
        let btree_dist = numrus_core::simd::hamming_distance(
            self.btree.data_slice(),
            other.btree.data_slice(),
        );
        if btree_dist > thresholds.btree_hamming {
            return None;
        }

        // Stage 3: CAM — Wasserstein on sorted phase vectors
        let cam_dist = wasserstein_sorted_i8(&self.cam, &other.cam);
        if cam_dist > thresholds.cam_wasserstein {
            return None;
        }

        // Stage 4: EMBED — circular distance on unsorted phase vectors
        let embed_dist = circular_distance_i8(&self.embed, &other.embed);
        if embed_dist > thresholds.embed_circular {
            return None;
        }

        Some(HybridDistances {
            meta_hamming: meta_dist,
            cam_wasserstein: cam_dist,
            btree_hamming: btree_dist,
            embed_circular: embed_dist,
        })
    }

    /// Batch hybrid sweep against a database of CogRecordV3.
    pub fn hybrid_search(
        &self,
        database: &[Self],
        thresholds: &HybridThresholds,
    ) -> Vec<(usize, HybridDistances)> {
        database
            .iter()
            .enumerate()
            .filter_map(|(i, rec)| self.hybrid_sweep(rec, thresholds).map(|d| (i, d)))
            .collect()
    }

    /// Get the unsorted CAM for phase unbinding operations.
    /// Requires the sort permutation to be stored.
    pub fn unsorted_cam(&self) -> Option<Vec<u8>> {
        self.cam_perm
            .as_ref()
            .map(|perm| unsort_phase_vector(&self.cam, perm))
    }

    /// Phase-bind the CAM container with another phase vector.
    /// Unsorts, binds, re-sorts, returns new sorted CAM + updated permutation.
    pub fn phase_bind_cam(&self, other: &[u8]) -> Option<(Vec<u8>, Vec<u16>)> {
        let unsorted = self.unsorted_cam()?;
        let bound = phase_bind_i8(&unsorted, other);
        Some(sort_phase_vector(&bound))
    }

    /// Phase-bind the EMBED container (no sort needed since EMBED is unsorted).
    pub fn phase_bind_embed(&self, other: &[u8]) -> Vec<u8> {
        phase_bind_i8(&self.embed, other)
    }

    /// Phase-unbind the EMBED container.
    pub fn phase_unbind_embed(&self, key: &[u8]) -> Vec<u8> {
        phase_unbind_i8(&self.embed, key)
    }

    /// Serialize to bytes: META(2048) + CAM_sorted(2048) + BTREE(2048) + EMBED(2048) = 8192.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CONTAINER_BYTES * 4);
        out.extend_from_slice(self.meta.data_slice());
        out.extend_from_slice(&self.cam);
        out.extend_from_slice(self.btree.data_slice());
        out.extend_from_slice(&self.embed);
        out
    }

    /// Deserialize from 8192 bytes. CAM is assumed pre-sorted. No permutation.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), CONTAINER_BYTES * 4);
        Self {
            meta: NumArrayU8::new(data[0..CONTAINER_BYTES].to_vec()),
            cam: data[CONTAINER_BYTES..CONTAINER_BYTES * 2].to_vec(),
            btree: NumArrayU8::new(data[CONTAINER_BYTES * 2..CONTAINER_BYTES * 3].to_vec()),
            embed: data[CONTAINER_BYTES * 3..CONTAINER_BYTES * 4].to_vec(),
            cam_perm: None,
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(val: u8) -> CogRecordV3 {
        CogRecordV3::new(
            &vec![val; CONTAINER_BYTES],
            &vec![val.wrapping_add(10); CONTAINER_BYTES],
            &vec![val.wrapping_add(20); CONTAINER_BYTES],
            &vec![val.wrapping_add(30); CONTAINER_BYTES],
        )
    }

    #[test]
    fn test_cogrecord_v3_construction() {
        let rec = make_record(42);
        assert_eq!(rec.meta.len(), CONTAINER_BYTES);
        assert_eq!(rec.cam.len(), CONTAINER_BYTES);
        assert_eq!(rec.btree.len(), CONTAINER_BYTES);
        assert_eq!(rec.embed.len(), CONTAINER_BYTES);
        assert!(rec.cam_perm.is_some());
    }

    #[test]
    fn test_cogrecord_v3_cam_is_sorted() {
        let cam_data: Vec<u8> = vec![200, 50, 100, 0, 255, 128, 75, 10];
        let padding = vec![0u8; CONTAINER_BYTES - 8];
        let mut full_cam = cam_data.clone();
        full_cam.extend_from_slice(&padding);

        let rec = CogRecordV3::new(
            &vec![0u8; CONTAINER_BYTES],
            &full_cam,
            &vec![0u8; CONTAINER_BYTES],
            &vec![0u8; CONTAINER_BYTES],
        );

        // Verify sorted
        for i in 1..rec.cam.len() {
            assert!(rec.cam[i] >= rec.cam[i - 1]);
        }
    }

    #[test]
    fn test_cogrecord_v3_cam_unsort_round_trip() {
        let mut rng = super::super::phase::SplitMix64(42);
        let cam_data: Vec<u8> = (0..CONTAINER_BYTES)
            .map(|_| (rng.next() % 256) as u8)
            .collect();

        let rec = CogRecordV3::new(
            &vec![0u8; CONTAINER_BYTES],
            &cam_data,
            &vec![0u8; CONTAINER_BYTES],
            &vec![0u8; CONTAINER_BYTES],
        );

        let unsorted = rec.unsorted_cam().unwrap();
        assert_eq!(unsorted, cam_data);
    }

    #[test]
    fn test_hybrid_sweep_pass() {
        let rec1 = make_record(0);
        let rec2 = make_record(1);

        let thresholds = HybridThresholds {
            meta_hamming: 20000,
            cam_wasserstein: 100000,
            btree_hamming: 20000,
            embed_circular: 100000,
        };

        let result = rec1.hybrid_sweep(&rec2, &thresholds);
        assert!(result.is_some());
    }

    #[test]
    fn test_hybrid_sweep_reject_meta() {
        let rec1 = make_record(0);
        let rec2 = make_record(0xFF);

        let thresholds = HybridThresholds {
            meta_hamming: 100, // very tight — will reject
            cam_wasserstein: 100000,
            btree_hamming: 100000,
            embed_circular: 100000,
        };

        let result = rec1.hybrid_sweep(&rec2, &thresholds);
        assert!(result.is_none());
    }

    #[test]
    fn test_hybrid_sweep_self_match() {
        let rec = make_record(42);

        let thresholds = HybridThresholds {
            meta_hamming: 1,
            cam_wasserstein: 1,
            btree_hamming: 1,
            embed_circular: 1,
        };

        let result = rec.hybrid_sweep(&rec, &thresholds);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.meta_hamming, 0);
        assert_eq!(d.cam_wasserstein, 0);
        assert_eq!(d.btree_hamming, 0);
        assert_eq!(d.embed_circular, 0);
    }

    #[test]
    fn test_hybrid_search() {
        let query = make_record(0);
        let database: Vec<CogRecordV3> = (0..10).map(|i| make_record(i as u8)).collect();

        let thresholds = HybridThresholds {
            meta_hamming: 3000,
            cam_wasserstein: 100000,
            btree_hamming: 3000,
            embed_circular: 100000,
        };

        let results = query.hybrid_search(&database, &thresholds);
        // Record 0 should match (self)
        assert!(results.iter().any(|&(idx, _)| idx == 0));
    }

    #[test]
    fn test_serialization_round_trip() {
        let rec = make_record(42);
        let bytes = rec.to_bytes();
        assert_eq!(bytes.len(), CONTAINER_BYTES * 4);

        let rec2 = CogRecordV3::from_bytes(&bytes);
        assert_eq!(rec2.meta.data_slice(), rec.meta.data_slice());
        assert_eq!(rec2.cam, rec.cam);
        assert_eq!(rec2.btree.data_slice(), rec.btree.data_slice());
        assert_eq!(rec2.embed, rec.embed);
    }

    #[test]
    fn test_phase_bind_embed() {
        let rec = make_record(100);
        let key: Vec<u8> = (0..CONTAINER_BYTES).map(|i| (i % 256) as u8).collect();

        let bound = rec.phase_bind_embed(&key);
        let recovered = phase_unbind_i8(&bound, &key);
        assert_eq!(recovered, rec.embed);
    }

    #[test]
    fn test_zeros() {
        let rec = CogRecordV3::zeros();
        assert!(rec.meta.data_slice().iter().all(|&x| x == 0));
        assert!(rec.cam.iter().all(|&x| x == 0));
        assert!(rec.btree.data_slice().iter().all(|&x| x == 0));
        assert!(rec.embed.iter().all(|&x| x == 0));
    }
}
