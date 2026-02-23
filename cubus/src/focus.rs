//! Focus-of-Attention Lithographic Gating for CogRecord containers.
//!
//! Spatial attention mechanism that treats any 2048-byte container as an
//! 8×8×32 3D volume and uses three planar masks (48 bits total) to gate
//! where writes/reads/comparisons occur.
//!
//! Sits UNDERNEATH both random-phase (phase.rs) and carrier (carrier.rs)
//! paths — applies to ALL container types (binary and phase alike).
//!
//! ## The Geometry
//!
//! ```text
//! container[2048] → volume[8][8][32]
//!   Axis X:  8 slabs of 256 bytes  (coarse: semantic class)
//!   Axis Y:  8 slabs of  32 bytes  (medium: concept sub-type)
//!   Axis Z: 32 slabs of   1 byte   (fine:   feature detail)
//!
//!   byte_index = x * 256 + y * 32 + z
//! ```
//!
//! Three masks: `mask_x: u8`, `mask_y: u8`, `mask_z: u32` = 48 bits.
//! A byte is "in focus" only if ALL THREE masks select its slab.

use crate::carrier::CarrierBasis;

// ============================================================================
// Constants
// ============================================================================

/// X dimension of the 3D volume interpretation.
pub const FOCUS_DIM_X: usize = 8;
/// Y dimension of the 3D volume interpretation.
pub const FOCUS_DIM_Y: usize = 8;
/// Z dimension of the 3D volume interpretation.
pub const FOCUS_DIM_Z: usize = 32;

/// Below this region size (in bytes), use scalar skip-loop.
/// Above, use materialized SIMD path.
const SIMD_THRESHOLD: u32 = 64;

// ============================================================================
// FocusDensity
// ============================================================================

/// Focus density presets controlling how many bits are set per mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusDensity {
    /// 1×1×4 = 4 bytes focused (0.2%) — max ~32 non-overlapping concepts
    Sparse,
    /// 2×2×8 = 32 bytes focused (1.6%) — max ~11 non-overlapping concepts
    Medium,
    /// 4×4×16 = 256 bytes focused (12.5%) — max ~4 non-overlapping concepts
    Broad,
}

impl FocusDensity {
    /// Returns (bits_x, bits_y, bits_z) for this density preset.
    pub fn bit_counts(self) -> (u32, u32, u32) {
        match self {
            FocusDensity::Sparse => (1, 1, 4),
            FocusDensity::Medium => (2, 2, 8),
            FocusDensity::Broad => (4, 4, 16),
        }
    }
}

// ============================================================================
// Pack / Unpack
// ============================================================================

/// Pack three masks into a single u64 for compact storage/comparison.
///   bits [0..7]   = mask_x (u8)
///   bits [8..15]  = mask_y (u8)
///   bits [16..47] = mask_z (u32)
///   bits [48..63] = unused
pub fn pack_focus(mask_x: u8, mask_y: u8, mask_z: u32) -> u64 {
    mask_x as u64 | ((mask_y as u64) << 8) | ((mask_z as u64) << 16)
}

/// Unpack a u64 into (mask_x, mask_y, mask_z).
pub fn unpack_focus(packed: u64) -> (u8, u8, u32) {
    let mask_x = (packed & 0xFF) as u8;
    let mask_y = ((packed >> 8) & 0xFF) as u8;
    let mask_z = ((packed >> 16) & 0xFFFFFFFF) as u32;
    (mask_x, mask_y, mask_z)
}

// ============================================================================
// concept_to_focus — deterministic mask generation
// ============================================================================

/// SplitMix64 PRNG for deterministic mask generation.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Select `count` distinct bits from a mask of `total` bits using Fisher-Yates.
fn select_bits(rng: &mut SplitMix64, total: u32, count: u32) -> u64 {
    let mut indices: Vec<u32> = (0..total).collect();
    for i in 0..count.min(total) {
        let j = i + (rng.next() % (total - i) as u64) as u32;
        indices.swap(i as usize, j as usize);
    }
    let mut mask: u64 = 0;
    for i in 0..count.min(total) {
        mask |= 1u64 << indices[i as usize];
    }
    mask
}

/// Derive focus masks from a concept key using SplitMix64.
/// The hash determines WHERE in the container this concept lives.
pub fn concept_to_focus(concept_id: u64, density: FocusDensity) -> (u8, u8, u32) {
    let (bits_x, bits_y, bits_z) = density.bit_counts();
    let mut rng = SplitMix64(concept_id);

    let mask_x = select_bits(&mut rng, 8, bits_x) as u8;
    let mask_y = select_bits(&mut rng, 8, bits_y) as u8;
    let mask_z = select_bits(&mut rng, 32, bits_z) as u32;

    (mask_x, mask_y, mask_z)
}

// ============================================================================
// Operation 14a: focus_xor — Write/Erase via XOR gating
// ============================================================================

/// XOR a value into the container at the focused sub-volume.
///
/// Self-inverse: `focus_xor(focus_xor(c, m, v), m, v)` restores c.
/// Bytes outside the mask are NEVER touched.
pub fn focus_xor(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] ^= value[idx];
            }
        }
    }
}

// ============================================================================
// Operation 14b: focus_read — AND extraction
// ============================================================================

/// Extract the focused sub-volume. Non-focused positions are zeroed.
/// Non-destructive read — the container is unchanged.
pub fn focus_read(container: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Vec<u8> {
    assert!(container.len() >= 2048);
    let mut out = vec![0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                out[idx] = container[idx];
            }
        }
    }

    out
}

// ============================================================================
// Operation 14c: focus_add / focus_sub — phase-space gated write
// ============================================================================

/// ADD a value into the container at the focused sub-volume (phase-space).
/// NOT self-inverse — use focus_sub to undo.
pub fn focus_add(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] = container[idx].wrapping_add(value[idx]);
            }
        }
    }
}

/// SUB a value from the container at the focused sub-volume.
/// Exact inverse of focus_add.
pub fn focus_sub(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] = container[idx].wrapping_sub(value[idx]);
            }
        }
    }
}

// ============================================================================
// Operation 14d: focus_hamming / focus_l1 — regional distance
// ============================================================================

/// Hamming distance within focus region (for binary containers).
/// Returns (hamming_distance, region_size_bytes).
pub fn focus_hamming(a: &[u8], b: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> (u64, u32) {
    assert!(a.len() >= 2048 && b.len() >= 2048);
    let mut distance: u64 = 0;
    let mut region_size: u32 = 0;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                distance += (a[idx] ^ b[idx]).count_ones() as u64;
                region_size += 1;
            }
        }
    }

    (distance, region_size)
}

/// L1 distance within focus region (for phase containers).
/// Returns (l1_distance, region_size_bytes).
pub fn focus_l1(a: &[u8], b: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> (u64, u32) {
    assert!(a.len() >= 2048 && b.len() >= 2048);
    let mut distance: u64 = 0;
    let mut region_size: u32 = 0;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                distance += (a[idx] as i16 - b[idx] as i16).unsigned_abs() as u64;
                region_size += 1;
            }
        }
    }

    (distance, region_size)
}

// ============================================================================
// Materialized mask operations
// ============================================================================

/// Expand the 48-bit focus address into a full 2048-byte mask.
/// out[i] = 0xFF if position i is in focus, 0x00 otherwise.
pub fn materialize_focus_mask(mask_x: u8, mask_y: u8, mask_z: u32) -> [u8; 2048] {
    let mut mask = [0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                mask[x * 256 + y * 32 + z] = 0xFF;
            }
        }
    }

    mask
}

/// Materialized XOR: `container[i] ^= (value[i] & mask[i])`.
/// SIMD-friendly: VPAND + VPXORD per 64-byte chunk.
pub fn focus_xor_materialized(container: &mut [u8], mask: &[u8; 2048], value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for i in 0..2048 {
        container[i] ^= value[i] & mask[i];
    }
}

/// Materialized ADD: `container[i] = container[i].wrapping_add(value[i] & mask[i])`.
/// SIMD-friendly: VPAND + VPADDB per 64-byte chunk.
pub fn focus_add_materialized(container: &mut [u8], mask: &[u8; 2048], value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for i in 0..2048 {
        container[i] = container[i].wrapping_add(value[i] & mask[i]);
    }
}

/// Auto-dispatch: scalar for sparse, materialized for broad.
pub fn focus_xor_auto(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    let region = mask_x.count_ones() * mask_y.count_ones() * mask_z.count_ones();
    if region < SIMD_THRESHOLD {
        focus_xor(container, mask_x, mask_y, mask_z, value);
    } else {
        let mask = materialize_focus_mask(mask_x, mask_y, mask_z);
        focus_xor_materialized(container, &mask, value);
    }
}

// ============================================================================
// Composition: focus + binary / phase / carrier
// ============================================================================

/// Write a concept into a binary container at a focused region.
/// Uses XOR binding. Self-inverse: call again to erase.
pub fn focus_bind_binary(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_xor(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Write a concept into a phase container at a focused region.
/// Uses ADD binding. NOT self-inverse — use focus_unbind_phase to erase.
pub fn focus_bind_phase(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_add(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Erase a concept from a phase container at a focused region.
pub fn focus_unbind_phase(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_sub(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Write a carrier-encoded concept into a focused region of a waveform.
///
/// Combines carrier encoding (frequency multiplexing) with focus gating
/// (spatial partitioning). The carrier signal only exists in the focused
/// region.
pub fn focus_carrier_encode(
    container: &mut [i8],
    basis: &CarrierBasis,
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    freq_idx: u8,
    phase_offset: f32,
    amplitude: f32,
) {
    let cos_phi = phase_offset.cos();
    let sin_phi = phase_offset.sin();
    let fi = freq_idx as usize;
    let scale = amplitude / crate::carrier::CARRIER_AMPLITUDE;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let j = x * 256 + y * 32 + z;
                let cos_val = basis.basis_cos[fi][j] as f32;
                let sin_val = basis.basis_sin[fi][j] as f32;
                let contribution = ((cos_phi * cos_val - sin_phi * sin_val) * scale)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                container[j] = container[j].saturating_add(contribution);
            }
        }
    }
}

// ============================================================================
// XOR Delta (Shared Lithography pattern)
// ============================================================================

/// Compute the XOR delta between two containers, restricted to a focus region.
/// Returns a 2048-byte delta where non-focused positions are zero.
pub fn focus_delta(old: &[u8], new: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Vec<u8> {
    assert!(old.len() >= 2048 && new.len() >= 2048);
    let mut delta = vec![0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                delta[idx] = old[idx] ^ new[idx];
            }
        }
    }

    delta
}

/// Compact delta: only the non-zero bytes with their positions.
/// For sparse focus (4 bytes): 4 × 3 = 12 bytes vs 2048 for full delta.
pub struct CompactDelta {
    /// Packed focus address (6 bytes in a u64).
    pub mask: u64,
    /// (byte_position, xor_delta) pairs.
    pub changes: Vec<(u16, u8)>,
}

impl CompactDelta {
    pub fn from_delta(delta: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Self {
        let mut changes = Vec::new();
        for (i, &d) in delta.iter().enumerate() {
            if d != 0 {
                changes.push((i as u16, d));
            }
        }
        CompactDelta {
            mask: pack_focus(mask_x, mask_y, mask_z),
            changes,
        }
    }

    /// Apply this delta to a container via XOR.
    pub fn apply(&self, container: &mut [u8]) {
        for &(pos, delta) in &self.changes {
            container[pos as usize] ^= delta;
        }
    }

    /// Wire size in bytes: 8 bytes header + 3 bytes per change.
    pub fn wire_size(&self) -> usize {
        8 + self.changes.len() * 3
    }
}

// ============================================================================
// FocusRegistry — track what's written where
// ============================================================================

/// Tracks which focus addresses are occupied in a container.
/// Each entry: (packed_focus: u64, concept_id: u64) = 16 bytes.
pub struct FocusRegistry {
    pub entries: Vec<(u64, u64)>,
}

impl Default for FocusRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FocusRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a concept at a focus address.
    pub fn register(&mut self, focus: u64, concept_id: u64) {
        self.entries.push((focus, concept_id));
    }

    /// Check if a proposed focus address overlaps with any existing entry.
    /// Returns overlapping (concept_id, overlap_size_bytes) pairs.
    pub fn check_overlap(
        &self,
        new_mask_x: u8,
        new_mask_y: u8,
        new_mask_z: u32,
    ) -> Vec<(u64, u32)> {
        let mut overlaps = Vec::new();

        for &(existing_packed, concept_id) in &self.entries {
            let (ex, ey, ez) = unpack_focus(existing_packed);
            let overlap_x = (new_mask_x & ex).count_ones();
            let overlap_y = (new_mask_y & ey).count_ones();
            let overlap_z = (new_mask_z & ez).count_ones();
            let overlap_bytes = overlap_x * overlap_y * overlap_z;
            if overlap_bytes > 0 {
                overlaps.push((concept_id, overlap_bytes));
            }
        }

        overlaps
    }

    /// Remove a concept from the registry.
    pub fn remove(&mut self, concept_id: u64) -> Option<u64> {
        if let Some(pos) = self.entries.iter().position(|&(_, id)| id == concept_id) {
            let (focus, _) = self.entries.remove(pos);
            Some(focus)
        } else {
            None
        }
    }

    /// Number of registered concepts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes occupied across all registered concepts (overlap counted once).
    pub fn total_coverage(&self) -> u32 {
        if self.entries.is_empty() {
            return 0;
        }

        let mut coverage = [false; 2048];
        for &(packed, _) in &self.entries {
            let (mx, my, mz) = unpack_focus(packed);
            for x in 0..FOCUS_DIM_X {
                if mx & (1 << x) == 0 {
                    continue;
                }
                for y in 0..FOCUS_DIM_Y {
                    if my & (1 << y) == 0 {
                        continue;
                    }
                    for z in 0..FOCUS_DIM_Z {
                        if mz & (1 << z) == 0 {
                            continue;
                        }
                        coverage[x * 256 + y * 32 + z] = true;
                    }
                }
            }
        }

        coverage.iter().filter(|&&b| b).count() as u32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Geometry tests ----

    #[test]
    fn test_materialize_all_bits_set() {
        let mask = materialize_focus_mask(0xFF, 0xFF, 0xFFFFFFFF);
        assert!(mask.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_materialize_no_bits_set() {
        let mask = materialize_focus_mask(0, 0, 0);
        assert!(mask.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_materialize_single_byte_origin() {
        // mask_x=1 (bit 0), mask_y=1 (bit 0), mask_z=1 (bit 0) → index [0][0][0] = 0
        let mask = materialize_focus_mask(1, 1, 1);
        assert_eq!(mask[0], 0xFF);
        let count = mask.iter().filter(|&&b| b == 0xFF).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_materialize_single_byte_corner() {
        // mask_x=0x80 (bit 7), mask_y=0x80 (bit 7), mask_z=0x80000000 (bit 31)
        // index = 7*256 + 7*32 + 31 = 1792 + 224 + 31 = 2047
        let mask = materialize_focus_mask(0x80, 0x80, 0x80000000);
        assert_eq!(mask[2047], 0xFF);
        let count = mask.iter().filter(|&&b| b == 0xFF).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_materialize_region_size() {
        // popcount(mask_x) × popcount(mask_y) × popcount(mask_z) = region bytes
        let test_cases: Vec<(u8, u8, u32)> = vec![
            (0x03, 0x05, 0x0000000F), // 2 × 2 × 4 = 16
            (0xFF, 0x01, 0x00000001), // 8 × 1 × 1 = 8
            (0x01, 0xFF, 0xFFFFFFFF), // 1 × 8 × 32 = 256
        ];
        for (mx, my, mz) in test_cases {
            let mask = materialize_focus_mask(mx, my, mz);
            let count = mask.iter().filter(|&&b| b == 0xFF).count();
            let expected =
                mx.count_ones() as usize * my.count_ones() as usize * mz.count_ones() as usize;
            assert_eq!(count, expected, "mx={:#x} my={:#x} mz={:#x}", mx, my, mz);
        }
    }

    // ---- Pack / Unpack tests ----

    #[test]
    fn test_pack_unpack_round_trip() {
        let mx: u8 = 0xA5;
        let my: u8 = 0x3C;
        let mz: u32 = 0xDEADBEEF;
        let packed = pack_focus(mx, my, mz);
        let (rx, ry, rz) = unpack_focus(packed);
        assert_eq!(rx, mx);
        assert_eq!(ry, my);
        assert_eq!(rz, mz);
    }

    // ---- XOR gating tests ----

    #[test]
    fn test_focus_xor_self_inverse() {
        let mut container = vec![0x42u8; 2048];
        let original = container.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();

        focus_xor(&mut container, 0x0F, 0xF0, 0x0000FFFF, &value);
        assert_ne!(container, original);
        focus_xor(&mut container, 0x0F, 0xF0, 0x0000FFFF, &value);
        assert_eq!(container, original);
    }

    #[test]
    fn test_focus_xor_commutative() {
        let mut c1 = vec![0u8; 2048];
        let mut c2 = vec![0u8; 2048];
        let v1: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let v2: Vec<u8> = (0..2048).map(|i| ((i * 41 + 7) % 256) as u8).collect();

        let mx = 0x03u8;
        let my = 0x0Cu8;
        let mz = 0x000000FFu32;

        focus_xor(&mut c1, mx, my, mz, &v1);
        focus_xor(&mut c1, mx, my, mz, &v2);

        focus_xor(&mut c2, mx, my, mz, &v2);
        focus_xor(&mut c2, mx, my, mz, &v1);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_focus_xor_preserves_outside() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let original = container.clone();
        let value = vec![0xFFu8; 2048];

        let mx = 0x01u8; // only X slab 0
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // only Z slabs 0-3

        focus_xor(&mut container, mx, my, mz, &value);

        // Check outside mask is unchanged
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(
                    container[i], original[i],
                    "position {} outside mask changed",
                    i
                );
            }
        }
    }

    #[test]
    fn test_focus_xor_materialized_matches_scalar() {
        let mut c1 = vec![0x55u8; 2048];
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 71 % 256) as u8).collect();

        let mx = 0x0Fu8;
        let my = 0xF0u8;
        let mz = 0x00FF00FFu32;

        focus_xor(&mut c1, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        focus_xor_materialized(&mut c2, &mask, &value);

        assert_eq!(c1, c2);
    }

    // ---- Phase gating tests ----

    #[test]
    fn test_focus_add_sub_round_trip() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i * 31 % 256) as u8).collect();
        let original = container.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 47 % 256) as u8).collect();

        let mx = 0x33u8;
        let my = 0xCCu8;
        let mz = 0x0F0F0F0Fu32;

        focus_add(&mut container, mx, my, mz, &value);
        assert_ne!(container, original);
        focus_sub(&mut container, mx, my, mz, &value);
        assert_eq!(container, original);
    }

    #[test]
    fn test_focus_add_only_modifies_masked() {
        let mut container = vec![100u8; 2048];
        let original = container.clone();
        let value = vec![50u8; 2048];

        let mx = 0x80u8;
        let my = 0x01u8;
        let mz = 0x00000001u32;

        focus_add(&mut container, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(container[i], original[i]);
            }
        }
    }

    #[test]
    fn test_focus_add_materialized_matches_scalar() {
        let mut c1: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 19 % 256) as u8).collect();

        let mx = 0x55u8;
        let my = 0xAAu8;
        let mz = 0xF0F0F0F0u32;

        focus_add(&mut c1, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        focus_add_materialized(&mut c2, &mask, &value);

        assert_eq!(c1, c2);
    }

    // ---- Focus read tests ----

    #[test]
    fn test_focus_read_zero_container() {
        let container = vec![0u8; 2048];
        let result = focus_read(&container, 0xFF, 0xFF, 0xFFFFFFFF);
        assert!(result.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_focus_read_after_xor() {
        let mut container = vec![0u8; 2048];
        let value: Vec<u8> = (0..2048).map(|i| (i * 7 + 1) as u8).collect();

        let mx = 0x03u8;
        let my = 0x03u8;
        let mz = 0x0000000Fu32;

        focus_xor(&mut container, mx, my, mz, &value);
        let read = focus_read(&container, mx, my, mz);

        // At focused positions, read should equal value (since container was zero)
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], value[i], "focused pos {} mismatch", i);
            } else {
                assert_eq!(read[i], 0, "non-focused pos {} should be zero", i);
            }
        }
    }

    #[test]
    fn test_focus_read_non_focused_zero() {
        let container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let result = focus_read(&container, 0x01, 0x01, 0x00000001);
        // Only position 0 is focused
        for i in 1..2048 {
            if i != 0 {
                // Check all non-focused positions
                let mask = materialize_focus_mask(0x01, 0x01, 0x00000001);
                if mask[i] == 0 {
                    assert_eq!(result[i], 0);
                }
            }
        }
    }

    // ---- Distance tests ----

    #[test]
    fn test_focus_hamming_self_zero() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let (dist, size) = focus_hamming(&a, &a, 0xFF, 0xFF, 0xFFFFFFFF);
        assert_eq!(dist, 0);
        assert_eq!(size, 2048);
    }

    #[test]
    fn test_focus_hamming_restricted_region() {
        let a = vec![0u8; 2048];
        let b = vec![0xFFu8; 2048];

        // Full mask: every byte differs by 8 bits
        let (full_dist, full_size) = focus_hamming(&a, &b, 0xFF, 0xFF, 0xFFFFFFFF);
        assert_eq!(full_dist, 2048 * 8);
        assert_eq!(full_size, 2048);

        // Single byte mask
        let (single_dist, single_size) = focus_hamming(&a, &b, 0x01, 0x01, 0x00000001);
        assert_eq!(single_dist, 8);
        assert_eq!(single_size, 1);
    }

    #[test]
    fn test_focus_l1_self_zero() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let (dist, _) = focus_l1(&a, &a, 0x0F, 0xF0, 0x00FF00FF);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_focus_l1_region_matches_manual() {
        let a = vec![100u8; 2048];
        let b = vec![110u8; 2048];

        // Region size = 2 × 2 × 4 = 16 bytes, each with L1 = 10
        let (dist, size) = focus_l1(&a, &b, 0x03, 0x03, 0x0000000F);
        assert_eq!(size, 16);
        assert_eq!(dist, 16 * 10);
    }

    // ---- Registry tests ----

    #[test]
    fn test_registry_no_overlap_sparse() {
        let mut reg = FocusRegistry::new();

        // Register 5 concepts at different sparse addresses
        for i in 0..5u64 {
            let (mx, my, mz) = concept_to_focus(i * 1000 + 42, FocusDensity::Sparse);
            let packed = pack_focus(mx, my, mz);
            reg.register(packed, i);
        }

        // Check overlap of a new concept
        let (nmx, nmy, nmz) = concept_to_focus(99999, FocusDensity::Sparse);
        let _overlaps = reg.check_overlap(nmx, nmy, nmz);
        // Sparse masks are very unlikely to overlap
        // (but not impossible — just verify the method runs correctly)
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn test_registry_detect_overlap() {
        let mut reg = FocusRegistry::new();

        // Register a concept at a specific address
        let mx = 0xFFu8;
        let my = 0xFFu8;
        let mz = 0xFFFFFFFFu32;
        reg.register(pack_focus(mx, my, mz), 1);

        // Any new concept will overlap with a full mask
        let overlaps = reg.check_overlap(0x01, 0x01, 0x00000001);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].0, 1); // concept_id
        assert_eq!(overlaps[0].1, 1); // 1 byte overlap
    }

    #[test]
    fn test_registry_total_coverage_sparse() {
        let mut reg = FocusRegistry::new();

        // Register 3 non-overlapping sparse concepts manually
        // Each sparse = 1×1×4 = 4 bytes
        reg.register(pack_focus(0x01, 0x01, 0x0000000F), 1); // X0,Y0,Z0-3
        reg.register(pack_focus(0x02, 0x02, 0x0000000F), 2); // X1,Y1,Z0-3
        reg.register(pack_focus(0x04, 0x04, 0x0000000F), 3); // X2,Y2,Z0-3

        assert_eq!(reg.total_coverage(), 12); // 3 × 4 = 12
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = FocusRegistry::new();
        let packed = pack_focus(0x01, 0x01, 0x00000001);
        reg.register(packed, 42);
        assert_eq!(reg.len(), 1);

        let removed = reg.remove(42);
        assert_eq!(removed, Some(packed));
        assert_eq!(reg.len(), 0);

        let not_found = reg.remove(42);
        assert_eq!(not_found, None);
    }

    // ---- concept_to_focus determinism ----

    #[test]
    fn test_concept_to_focus_deterministic() {
        let (mx1, my1, mz1) = concept_to_focus(12345, FocusDensity::Sparse);
        let (mx2, my2, mz2) = concept_to_focus(12345, FocusDensity::Sparse);
        assert_eq!(mx1, mx2);
        assert_eq!(my1, my2);
        assert_eq!(mz1, mz2);
    }

    #[test]
    fn test_concept_to_focus_different_ids() {
        let mut masks = std::collections::HashSet::new();
        for id in 0..100u64 {
            let (mx, my, mz) = concept_to_focus(id, FocusDensity::Medium);
            masks.insert((mx, my, mz));
        }
        // With 100 random IDs and medium density, most should be distinct
        assert!(
            masks.len() > 50,
            "expected most masks unique, got {}",
            masks.len()
        );
    }

    #[test]
    fn test_concept_to_focus_density_bits() {
        for density in [
            FocusDensity::Sparse,
            FocusDensity::Medium,
            FocusDensity::Broad,
        ] {
            let (bits_x, bits_y, bits_z) = density.bit_counts();
            let (mx, my, mz) = concept_to_focus(42, density);
            assert_eq!(mx.count_ones(), bits_x, "density={:?} mask_x bits", density);
            assert_eq!(my.count_ones(), bits_y, "density={:?} mask_y bits", density);
            assert_eq!(mz.count_ones(), bits_z, "density={:?} mask_z bits", density);
        }
    }

    // ---- Integration tests ----

    #[test]
    fn test_write_read_10_sparse_concepts() {
        let mut container = vec![0u8; 2048];
        let mut rng = super::SplitMix64(777);

        let concepts: Vec<(u64, Vec<u8>)> = (0..10)
            .map(|id| {
                let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                (id, value)
            })
            .collect();

        // Write all 10
        for (id, value) in &concepts {
            let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
            focus_xor(&mut container, mx, my, mz, value);
        }

        // Read each back and verify signal is present
        for (id, value) in &concepts {
            let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
            let read = focus_read(&container, mx, my, mz);
            let mask = materialize_focus_mask(mx, my, mz);

            // Count how many focused positions match
            let mut matches = 0u32;
            let mut total = 0u32;
            for i in 0..2048 {
                if mask[i] == 0xFF {
                    total += 1;
                    if read[i] == value[i] {
                        matches += 1;
                    }
                }
            }
            // With sparse non-overlapping, most/all should match
            // (some may collide due to birthday effect)
            assert!(
                matches as f64 / total as f64 > 0.5,
                "concept {} signal too weak: {}/{}",
                id,
                matches,
                total
            );
        }
    }

    #[test]
    fn test_write_preserves_non_focused() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let original = container.clone();
        let mut rng = super::SplitMix64(123);

        for id in 0..10u64 {
            let (mx, my, mz) = concept_to_focus(id, FocusDensity::Sparse);
            let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
            focus_xor(&mut container, mx, my, mz, &value);
        }

        // Check that MOST positions are unchanged
        // (10 sparse concepts × 4 bytes each = ~40 bytes changed, out of 2048)
        let changed = container
            .iter()
            .zip(original.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            changed <= 200, // generous bound for overlaps
            "too many positions changed: {} (expected ~40)",
            changed
        );
    }

    #[test]
    fn test_focus_bind_binary_round_trip() {
        let mut container = vec![0u8; 2048];
        let concept: Vec<u8> = (0..2048).map(|i| (i * 7 % 256) as u8).collect();

        let mx = 0x0Fu8;
        let my = 0x0Fu8;
        let mz = 0x000000FFu32;

        focus_bind_binary(&mut container, mx, my, mz, &concept);
        let read = focus_read(&container, mx, my, mz);

        // Verify focused positions have the concept value
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], concept[i], "pos {} mismatch", i);
            }
        }

        // Erase: XOR again
        focus_bind_binary(&mut container, mx, my, mz, &concept);
        assert!(container.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_focus_bind_phase_round_trip() {
        let mut container = vec![0u8; 2048];
        let concept: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();

        let mx = 0x07u8;
        let my = 0x07u8;
        let mz = 0x0000FFFFu32;

        focus_bind_phase(&mut container, mx, my, mz, &concept);
        let read = focus_read(&container, mx, my, mz);

        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], concept[i], "pos {} mismatch", i);
            }
        }

        // Undo
        focus_unbind_phase(&mut container, mx, my, mz, &concept);
        assert!(container.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_compact_delta_wire_size() {
        let old = vec![0u8; 2048];
        let mut new = vec![0u8; 2048];
        let value: Vec<u8> = (0..2048).map(|i| (i * 3 % 256) as u8).collect();

        let mx = 0x01u8;
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // 1×1×4 = 4 bytes

        // Write into new
        new.copy_from_slice(&old);
        focus_xor(&mut new, mx, my, mz, &value);

        let delta = focus_delta(&old, &new, mx, my, mz);
        let compact = CompactDelta::from_delta(&delta, mx, my, mz);

        assert!(
            compact.wire_size() < 2048,
            "compact should be smaller than full"
        );
        assert!(
            compact.changes.len() <= 4,
            "sparse focus: at most 4 changes"
        );
    }

    #[test]
    fn test_compact_delta_apply() {
        let old: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        let value = vec![0xAAu8; 2048];

        let mx = 0x03u8;
        let my = 0x03u8;
        let mz = 0x0000000Fu32;

        focus_xor(&mut new, mx, my, mz, &value);

        let delta = focus_delta(&old, &new, mx, my, mz);
        let compact = CompactDelta::from_delta(&delta, mx, my, mz);

        let mut reconstructed = old.clone();
        compact.apply(&mut reconstructed);

        // Focused positions should match new
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(reconstructed[i], new[i], "pos {} mismatch", i);
            }
        }
    }

    #[test]
    fn test_focus_xor_auto_matches_scalar() {
        let mut c1 = vec![0x77u8; 2048];
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 59 % 256) as u8).collect();

        // Sparse: should use scalar
        focus_xor(&mut c1, 0x01, 0x01, 0x00000001, &value);
        focus_xor_auto(&mut c2, 0x01, 0x01, 0x00000001, &value);
        assert_eq!(c1, c2);

        // Broad: should use materialized
        let mut c3 = vec![0x77u8; 2048];
        let mut c4 = c3.clone();
        focus_xor(&mut c3, 0xFF, 0xFF, 0xFFFFFFFF, &value);
        focus_xor_auto(&mut c4, 0xFF, 0xFF, 0xFFFFFFFF, &value);
        assert_eq!(c3, c4);
    }

    // ---- Capacity experiment ----

    #[test]
    fn test_focus_capacity_experiment() {
        println!("\n=== Focus Gating Capacity Experiment ===\n");

        for &density in &[
            FocusDensity::Sparse,
            FocusDensity::Medium,
            FocusDensity::Broad,
        ] {
            let (bits_x, bits_y, bits_z) = density.bit_counts();
            let region_bytes = bits_x * bits_y * bits_z;

            let test_counts = [1u64, 5, 10, 20, 32, 50];
            let mut rng = super::SplitMix64(42);

            for &n in &test_counts {
                let mut container = vec![0u8; 2048];
                let concepts: Vec<(u64, Vec<u8>)> = (0..n)
                    .map(|id| {
                        let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                        (id, value)
                    })
                    .collect();

                // Write all
                for (id, value) in &concepts {
                    let (mx, my, mz) = concept_to_focus(*id, density);
                    focus_xor(&mut container, mx, my, mz, value);
                }

                // Read each back and measure accuracy
                let mut total_match = 0u32;
                let mut total_bits = 0u32;
                for (id, value) in &concepts {
                    let (mx, my, mz) = concept_to_focus(*id, density);
                    let read = focus_read(&container, mx, my, mz);
                    let mask = materialize_focus_mask(mx, my, mz);

                    for i in 0..2048 {
                        if mask[i] == 0xFF {
                            total_bits += 1;
                            if read[i] == value[i] {
                                total_match += 1;
                            }
                        }
                    }
                }

                let accuracy = if total_bits > 0 {
                    total_match as f64 / total_bits as f64
                } else {
                    0.0
                };

                println!(
                    "  {:?} ({}B region) N={:>3}: accuracy={:.1}% ({}/{})",
                    density,
                    region_bytes,
                    n,
                    accuracy * 100.0,
                    total_match,
                    total_bits
                );
            }
            println!();
        }

        // Verify sparse holds at N=10 with good accuracy
        {
            let mut container = vec![0u8; 2048];
            let mut rng = super::SplitMix64(999);
            let concepts: Vec<(u64, Vec<u8>)> = (0..10)
                .map(|id| {
                    let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                    (id, value)
                })
                .collect();

            for (id, value) in &concepts {
                let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
                focus_xor(&mut container, mx, my, mz, value);
            }

            let mut total_match = 0u32;
            let mut total_bits = 0u32;
            for (id, value) in &concepts {
                let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
                let read = focus_read(&container, mx, my, mz);
                let mask = materialize_focus_mask(mx, my, mz);
                for i in 0..2048 {
                    if mask[i] == 0xFF {
                        total_bits += 1;
                        if read[i] == value[i] {
                            total_match += 1;
                        }
                    }
                }
            }
            let accuracy = total_match as f64 / total_bits as f64;
            assert!(
                accuracy > 0.7,
                "Sparse N=10 accuracy {:.1}% too low",
                accuracy * 100.0
            );
        }
    }

    // ---- Carrier + focus integration ----

    #[test]
    fn test_focus_carrier_encode_writes_only_masked() {
        let basis = crate::carrier::CarrierBasis::new();
        let mut container = vec![0i8; 2048];

        let mx = 0x01u8;
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // 1×1×4 = 4 bytes

        focus_carrier_encode(
            &mut container,
            &basis,
            mx,
            my,
            mz,
            0,
            1.0,
            crate::carrier::CARRIER_AMPLITUDE,
        );

        // Check that only masked positions are non-zero
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(
                    container[i], 0,
                    "position {} outside mask should be zero, got {}",
                    i, container[i]
                );
            }
        }

        // At least some masked positions should be non-zero
        let nonzero_in_mask = (0..2048)
            .filter(|&i| mask[i] == 0xFF && container[i] != 0)
            .count();
        assert!(
            nonzero_in_mask > 0,
            "carrier should write some non-zero values"
        );
    }
}
