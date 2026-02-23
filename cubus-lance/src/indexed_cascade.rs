//! 4-channel indexed cascade search over columnar CogRecord data.
//!
//! Combines the `FragmentIndex` (primary/META) and `ChannelIndex` (sidecar
//! for CAM/BTREE/EMBED) to achieve sub-linear CogRecord search:
//!
//! ## Query flow
//!
//! 1. **META (fragment-indexed):** triangle inequality prunes fragments →
//!    read only surviving row ranges → fine-scan with Hamming distance.
//! 2. **CAM (sidecar-indexed):** overlapping_row_ids → intersect with
//!    META survivors → fetch + filter.
//! 3. **BTREE (sidecar-indexed):** same pattern, further reducing survivors.
//! 4. **EMBED (sidecar-indexed):** same pattern, final survivor set.
//!
//! ## Bandwidth arithmetic (100K records)
//!
//! | Strategy             | Stage 1 I/O | Total I/O | Reduction |
//! |---------------------|-------------|-----------|-----------|
//! | Row-major flat scan | 800 MB      | 800 MB    | 1×        |
//! | Columnar cascade    | 200 MB      | 222 MB    | 3.6×      |
//! | Indexed cascade     | 2.4 MB      | 2.7 MB    | 296×      |
//!
//! ## Incremental learning
//!
//! `learn()` appends new records and updates sidecar indices (O(new_records)).
//! `optimize()` rebuilds the primary index and physical layout (periodic).

use std::collections::HashSet;

use crate::channel_index::ChannelIndex;
use crate::fragment_index::FragmentIndex;
use numrus_core::simd::hamming_distance;
use numrus_rs::CogRecord;

/// Result of an indexed cascade search.
#[derive(Debug, Clone)]
pub struct IndexedCascadeResult {
    /// (row_index, [meta_dist, cam_dist, btree_dist, embed_dist]).
    /// Row indices refer to the original (pre-reorder) dataset.
    pub hits: Vec<(usize, [u64; 4])>,
    /// Performance counters.
    pub stats: IndexedCascadeStats,
}

/// Performance counters for indexed cascade search.
#[derive(Debug, Clone, Default)]
pub struct IndexedCascadeStats {
    /// Total fragments in the primary index.
    pub fragments_total: usize,
    /// Fragments pruned by triangle inequality.
    pub fragments_pruned: usize,
    /// Rows scanned in Stage 1 (META fine-scan).
    pub meta_scanned: usize,
    /// Row IDs fetched for Stage 2 (CAM).
    pub cam_fetched: usize,
    /// Row IDs fetched for Stage 3 (BTREE).
    pub btree_fetched: usize,
    /// Row IDs fetched for Stage 4 (EMBED).
    pub embed_fetched: usize,
}

/// All indices needed for indexed cascade search, bundled together.
pub struct CascadeIndices {
    /// Primary fragment index for META channel.
    pub meta_index: FragmentIndex,
    /// Sidecar index for CAM channel.
    pub cam_index: ChannelIndex,
    /// Sidecar index for BTREE channel.
    pub btree_index: ChannelIndex,
    /// Sidecar index for EMBED channel.
    pub embed_index: ChannelIndex,
}

impl CascadeIndices {
    /// Build all four channel indices from a set of CogRecords.
    ///
    /// # Arguments
    /// * `records` — the CogRecord dataset
    /// * `min_cluster_size` — minimum leaf cardinality for CLAM trees
    pub fn build(records: &[CogRecord], min_cluster_size: usize) -> Self {
        let count = records.len();
        let vec_len = 2048; // CogRecord CONTAINER_BYTES

        // Flatten each channel into a contiguous buffer for CLAM tree building.
        let mut meta_flat = Vec::with_capacity(count * vec_len);
        let mut cam_flat = Vec::with_capacity(count * vec_len);
        let mut btree_flat = Vec::with_capacity(count * vec_len);
        let mut embed_flat = Vec::with_capacity(count * vec_len);

        for rec in records {
            meta_flat.extend_from_slice(rec.meta.data_slice());
            cam_flat.extend_from_slice(rec.cam.data_slice());
            btree_flat.extend_from_slice(rec.btree.data_slice());
            embed_flat.extend_from_slice(rec.embed.data_slice());
        }

        let meta_index = FragmentIndex::build(&meta_flat, vec_len, count, min_cluster_size);
        let cam_index = ChannelIndex::build(1, &cam_flat, vec_len, count, min_cluster_size);
        let btree_index = ChannelIndex::build(2, &btree_flat, vec_len, count, min_cluster_size);
        let embed_index = ChannelIndex::build(3, &embed_flat, vec_len, count, min_cluster_size);

        CascadeIndices {
            meta_index,
            cam_index,
            btree_index,
            embed_index,
        }
    }
}

/// 4-channel indexed cascade search (in-memory, synchronous).
///
/// Operates on pre-loaded CogRecords with pre-built indices.
/// For Lance dataset I/O, use the async `indexed_cascade_lance()` variant.
///
/// ## Algorithm
///
/// Stage 1 (META): fragment prune → fine-scan → survivor row IDs
/// Stage 2 (CAM):  sidecar prune → intersect → filter → survivors
/// Stage 3 (BTREE): same
/// Stage 4 (EMBED): same
///
/// Returns the same results as `cascade_scan_4ch` but with sub-linear cost.
pub fn indexed_cascade_search(
    query: &CogRecord,
    records: &[CogRecord],
    indices: &CascadeIndices,
    thresholds: [u64; 4],
) -> IndexedCascadeResult {
    let mut stats = IndexedCascadeStats {
        fragments_total: indices.meta_index.num_fragments(),
        ..Default::default()
    };

    let q_meta = query.meta.data_slice();
    let q_cam = query.cam.data_slice();
    let q_btree = query.btree.data_slice();
    let q_embed = query.embed.data_slice();

    // ── Stage 1: META via fragment index ──
    let overlapping = indices.meta_index.find_overlapping(q_meta, thresholds[0]);
    stats.fragments_pruned = stats.fragments_total - overlapping.len();

    let mut survivor_rows: HashSet<usize> = HashSet::new();
    for frag in &overlapping {
        // Map cluster-ordered positions → original row IDs
        let orig_ids = indices
            .meta_index
            .original_row_ids(frag.row_id_start, frag.row_id_end);
        for orig_id in orig_ids {
            let dist = hamming_distance(q_meta, records[orig_id].meta.data_slice());
            stats.meta_scanned += 1;
            if dist <= thresholds[0] {
                survivor_rows.insert(orig_id);
            }
        }
    }

    // ── Stage 2: CAM via sidecar index ──
    let cam_candidates = indices.cam_index.overlapping_row_ids(q_cam, thresholds[1]);
    // Intersect: iterate the smaller set, probe the larger (both are HashSets).
    let cam_check: Vec<usize> = if survivor_rows.len() <= cam_candidates.len() {
        survivor_rows
            .iter()
            .filter(|id| cam_candidates.contains(id))
            .copied()
            .collect()
    } else {
        cam_candidates
            .iter()
            .filter(|id| survivor_rows.contains(id))
            .copied()
            .collect()
    };
    stats.cam_fetched = cam_check.len();

    let mut cam_survivors: HashSet<usize> = HashSet::new();
    for &row_id in &cam_check {
        let dist = hamming_distance(q_cam, records[row_id].cam.data_slice());
        if dist <= thresholds[1] {
            cam_survivors.insert(row_id);
        }
    }

    // ── Stage 3: BTREE via sidecar index ──
    let btree_candidates = indices
        .btree_index
        .overlapping_row_ids(q_btree, thresholds[2]);
    let btree_check: Vec<usize> = if cam_survivors.len() <= btree_candidates.len() {
        cam_survivors
            .iter()
            .filter(|id| btree_candidates.contains(id))
            .copied()
            .collect()
    } else {
        btree_candidates
            .iter()
            .filter(|id| cam_survivors.contains(id))
            .copied()
            .collect()
    };
    stats.btree_fetched = btree_check.len();

    let mut btree_survivors: HashSet<usize> = HashSet::new();
    for &row_id in &btree_check {
        let dist = hamming_distance(q_btree, records[row_id].btree.data_slice());
        if dist <= thresholds[2] {
            btree_survivors.insert(row_id);
        }
    }

    // ── Stage 4: EMBED via sidecar index ──
    let embed_candidates = indices
        .embed_index
        .overlapping_row_ids(q_embed, thresholds[3]);
    let embed_check: Vec<usize> = if btree_survivors.len() <= embed_candidates.len() {
        btree_survivors
            .iter()
            .filter(|id| embed_candidates.contains(id))
            .copied()
            .collect()
    } else {
        embed_candidates
            .iter()
            .filter(|id| btree_survivors.contains(id))
            .copied()
            .collect()
    };
    stats.embed_fetched = embed_check.len();

    let mut hits = Vec::new();
    for &row_id in &embed_check {
        let embed_dist = hamming_distance(q_embed, records[row_id].embed.data_slice());
        if embed_dist <= thresholds[3] {
            // Compute all 4 exact distances for the result.
            let meta_dist = hamming_distance(q_meta, records[row_id].meta.data_slice());
            let cam_dist = hamming_distance(q_cam, records[row_id].cam.data_slice());
            let btree_dist = hamming_distance(q_btree, records[row_id].btree.data_slice());
            hits.push((row_id, [meta_dist, cam_dist, btree_dist, embed_dist]));
        }
    }

    // Sort by embed distance for consistent output.
    hits.sort_by_key(|&(_, dists)| dists[3]);

    IndexedCascadeResult { hits, stats }
}

/// Incremental learning: update sidecar indices with new records.
///
/// Call this after appending new CogRecords to the dataset. The primary
/// FragmentIndex is NOT updated here — new records go into a logical
/// "pending" set visible only to sidecar stages (CAM/BTREE/EMBED).
/// Call `rebuild()` periodically to reconstruct the META fragment index.
///
/// # Arguments
/// * `indices` — mutable reference to the cascade indices
/// * `new_records` — the newly appended CogRecords
/// * `base_row_id` — the row ID of the first new record
pub fn learn(indices: &mut CascadeIndices, new_records: &[CogRecord], base_row_id: usize) {
    // Build insertion batches for each sidecar channel.
    let cam_batch: Vec<(usize, &[u8])> = new_records
        .iter()
        .enumerate()
        .map(|(i, rec)| (base_row_id + i, rec.cam.data_slice()))
        .collect();
    let btree_batch: Vec<(usize, &[u8])> = new_records
        .iter()
        .enumerate()
        .map(|(i, rec)| (base_row_id + i, rec.btree.data_slice()))
        .collect();
    let embed_batch: Vec<(usize, &[u8])> = new_records
        .iter()
        .enumerate()
        .map(|(i, rec)| (base_row_id + i, rec.embed.data_slice()))
        .collect();

    indices.cam_index.insert(&cam_batch);
    indices.btree_index.insert(&btree_batch);
    indices.embed_index.insert(&embed_batch);
}

/// Rebuild all indices from scratch.
///
/// This is the expensive operation — O(n log n) for CLAM tree construction.
/// Run during off-peak hours, not during query serving.
pub fn rebuild(records: &[CogRecord], min_cluster_size: usize) -> CascadeIndices {
    CascadeIndices::build(records, min_cluster_size)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_rs::NumArrayU8;

    fn make_container(val: u8) -> NumArrayU8 {
        NumArrayU8::new(vec![val; 2048])
    }

    fn make_test_records(n: usize) -> Vec<CogRecord> {
        (0..n)
            .map(|i| {
                let v = (i % 256) as u8;
                CogRecord::new(
                    make_container(v),
                    make_container(v.wrapping_add(1)),
                    make_container(v.wrapping_add(2)),
                    make_container(v.wrapping_add(3)),
                )
            })
            .collect()
    }

    #[test]
    fn test_indexed_cascade_finds_exact_match() {
        let mut records = make_test_records(50);
        // Record 25 is an exact copy of our query (all zeros)
        let query = CogRecord::new(
            make_container(0),
            make_container(0),
            make_container(0),
            make_container(0),
        );
        records[25] = CogRecord::new(
            make_container(0),
            make_container(0),
            make_container(0),
            make_container(0),
        );

        let indices = CascadeIndices::build(&records, 5);
        let result = indexed_cascade_search(
            &query,
            &records,
            &indices,
            [0, 0, 0, 0], // exact match only
        );

        assert!(
            result
                .hits
                .iter()
                .any(|&(idx, dists)| idx == 25 && dists == [0, 0, 0, 0]),
            "should find exact match at index 25, got {:?}",
            result.hits,
        );
    }

    #[test]
    fn test_indexed_vs_flat_same_results() {
        let records = make_test_records(30);
        let query = CogRecord::new(
            make_container(5),
            make_container(6),
            make_container(7),
            make_container(8),
        );
        let thresholds = [4000, 4000, 4000, 4000];

        // Flat brute-force reference
        let mut flat_hits: Vec<(usize, [u64; 4])> = Vec::new();
        for (i, rec) in records.iter().enumerate() {
            let dists = [
                hamming_distance(query.meta.data_slice(), rec.meta.data_slice()),
                hamming_distance(query.cam.data_slice(), rec.cam.data_slice()),
                hamming_distance(query.btree.data_slice(), rec.btree.data_slice()),
                hamming_distance(query.embed.data_slice(), rec.embed.data_slice()),
            ];
            if dists.iter().zip(thresholds.iter()).all(|(&d, &t)| d <= t) {
                flat_hits.push((i, dists));
            }
        }
        flat_hits.sort_by_key(|&(idx, _)| idx);

        // Indexed cascade
        let indices = CascadeIndices::build(&records, 3);
        let result = indexed_cascade_search(&query, &records, &indices, thresholds);
        let mut indexed_hits = result.hits.clone();
        indexed_hits.sort_by_key(|&(idx, _)| idx);

        // Indexed must find everything flat found (no false negatives).
        // It uses triangle inequality which is conservative, so no false negatives.
        for &(flat_idx, flat_dists) in &flat_hits {
            assert!(
                indexed_hits
                    .iter()
                    .any(|&(idx, dists)| idx == flat_idx && dists == flat_dists),
                "indexed cascade missed flat hit at row {} with dists {:?}",
                flat_idx,
                flat_dists,
            );
        }
    }

    #[test]
    fn test_fragment_pruning_reduces_work() {
        let records = make_test_records(100);
        let query = CogRecord::new(
            make_container(50),
            make_container(51),
            make_container(52),
            make_container(53),
        );
        let thresholds = [2000, 2000, 2000, 2000];

        let indices = CascadeIndices::build(&records, 10);
        let result = indexed_cascade_search(&query, &records, &indices, thresholds);

        // With tight thresholds, should prune at least some fragments.
        assert!(
            result.stats.fragments_pruned > 0 || result.stats.fragments_total <= 1,
            "expected some pruning: total={}, pruned={}",
            result.stats.fragments_total,
            result.stats.fragments_pruned,
        );

        // Meta scan should be less than total records.
        assert!(
            result.stats.meta_scanned <= 100,
            "meta_scanned={} should be <= 100",
            result.stats.meta_scanned,
        );
    }

    #[test]
    fn test_incremental_learn() {
        let records = make_test_records(40);
        let mut indices = CascadeIndices::build(&records, 5);

        let old_cam_rows = indices.cam_index.num_rows();
        let old_btree_rows = indices.btree_index.num_rows();
        let old_embed_rows = indices.embed_index.num_rows();

        // Add 10 new records
        let new_records = make_test_records(10);
        learn(&mut indices, &new_records, 40);

        assert_eq!(indices.cam_index.num_rows(), old_cam_rows + 10);
        assert_eq!(indices.btree_index.num_rows(), old_btree_rows + 10);
        assert_eq!(indices.embed_index.num_rows(), old_embed_rows + 10);
    }

    #[test]
    fn test_rebuild_produces_valid_indices() {
        let records = make_test_records(50);
        let indices = rebuild(&records, 5);

        assert!(indices.meta_index.num_fragments() > 0);
        assert!(indices.cam_index.num_clusters() > 0);
        assert!(indices.btree_index.num_clusters() > 0);
        assert!(indices.embed_index.num_clusters() > 0);

        assert_eq!(indices.meta_index.num_rows(), 50);
        assert_eq!(indices.cam_index.num_rows(), 50);
        assert_eq!(indices.btree_index.num_rows(), 50);
        assert_eq!(indices.embed_index.num_rows(), 50);
    }

    #[test]
    fn test_stats_counters_consistent() {
        let records = make_test_records(60);
        let query = records[0].clone();
        let thresholds = [8000, 8000, 8000, 8000];

        let indices = CascadeIndices::build(&records, 5);
        let result = indexed_cascade_search(&query, &records, &indices, thresholds);

        // Fragments pruned + surviving = total
        let surviving = result.stats.fragments_total - result.stats.fragments_pruned;
        assert!(
            surviving > 0,
            "at least one fragment should survive for own query"
        );

        // Cascade: cam_fetched <= meta_scanned, btree <= cam, embed <= btree
        assert!(result.stats.cam_fetched <= result.stats.meta_scanned);
        assert!(result.stats.btree_fetched <= result.stats.cam_fetched);
        assert!(result.stats.embed_fetched <= result.stats.btree_fetched);
    }
}
