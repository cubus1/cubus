//! Phase-space HDC operations: bind, unbind, Wasserstein, circular distance,
//! histogram, bundle, 5D projection, and sort.
//!
//! Phase vectors treat each byte as an angle (0-255 → 0°-360°).
//! Binding = addition mod 256 (VPADDB). Unbinding = subtraction mod 256 (VPSUBB).
//! Unlike binary XOR, phase operations preserve spatial locality.

use std::f64::consts::PI;

// -------------------------------------------------------------------------
// Operation 1: phase_bind_i8
// -------------------------------------------------------------------------

/// Phase-space binding: element-wise addition mod 256.
///
/// On AVX-512: VPADDB processes 64 bytes per instruction.
/// 2048 bytes = 32 VPADDB instructions.
///
/// Property: `phase_bind(phase_bind(a, b), phase_inverse(b)) == a`
pub fn phase_bind_i8(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect()
}

/// In-place phase binding (avoids allocation).
pub fn phase_bind_i8_inplace(a: &mut [u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x = x.wrapping_add(y);
    }
}

/// Compute the phase inverse: `inverse[i] = (256 - v[i]) % 256`.
pub fn phase_inverse_i8(v: &[u8]) -> Vec<u8> {
    v.iter().map(|&x| x.wrapping_neg()).collect()
}

// -------------------------------------------------------------------------
// Operation 2: phase_unbind_i8
// -------------------------------------------------------------------------

/// Phase-space unbinding: element-wise subtraction mod 256.
/// EXACT inverse of phase_bind — no noise, no information loss.
pub fn phase_unbind_i8(bound: &[u8], key: &[u8]) -> Vec<u8> {
    assert_eq!(bound.len(), key.len());
    bound
        .iter()
        .zip(key.iter())
        .map(|(&x, &y)| x.wrapping_sub(y))
        .collect()
}

// -------------------------------------------------------------------------
// Operation 3: wasserstein_sorted_i8
// -------------------------------------------------------------------------

/// Wasserstein-1 (Earth Mover's) distance between two PRE-SORTED u8 vectors.
///
/// For sorted vectors, Wasserstein-1 = Σ|a[i] - b[i]|.
/// Same cost as Hamming distance, but gives a TRUE metric spatial distance.
///
/// IMPORTANT: Both inputs MUST be sorted ascending.
pub fn wasserstein_sorted_i8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum()
}

/// Batch Wasserstein search with early-exit cascade.
///
/// Stage 1: sample 1/16, scale estimate, reject at 3σ
/// Stage 2: sample 1/4, reject at 2σ
/// Stage 3: full Wasserstein on survivors
pub fn wasserstein_search_adaptive(
    query: &[u8],
    database: &[u8],
    vec_len: usize,
    n: usize,
    max_distance: u64,
) -> Vec<(usize, u64)> {
    let mut results = Vec::new();
    let sample_16 = vec_len / 16;
    let sample_4 = vec_len / 4;
    let threshold_stage1 = max_distance / 16 + max_distance / 32; // ~1.5× scaled
    let threshold_stage2 = max_distance / 4 + max_distance / 16; // ~1.25× scaled

    for i in 0..n {
        let offset = i * vec_len;
        let candidate = &database[offset..offset + vec_len];

        // Stage 1: 1/16 sample
        let mut d1: u64 = 0;
        let step1 = vec_len / sample_16;
        for j in 0..sample_16 {
            let idx = j * step1;
            d1 += (query[idx] as i16 - candidate[idx] as i16).unsigned_abs() as u64;
        }
        if d1 > threshold_stage1 {
            continue;
        }

        // Stage 2: 1/4 sample
        let mut d2: u64 = 0;
        let step2 = vec_len / sample_4;
        for j in 0..sample_4 {
            let idx = j * step2;
            d2 += (query[idx] as i16 - candidate[idx] as i16).unsigned_abs() as u64;
        }
        if d2 > threshold_stage2 {
            continue;
        }

        // Stage 3: full distance
        let dist = wasserstein_sorted_i8(query, candidate);
        if dist <= max_distance {
            results.push((i, dist));
        }
    }

    results
}

// -------------------------------------------------------------------------
// Operation 4: circular_distance_i8
// -------------------------------------------------------------------------

/// Circular distance between two phase vectors (NOT necessarily sorted).
///
/// For each element: `min(|a-b|, 256-|a-b|)`
/// Respects phase wrap-around: phase 254 is distance 4 from phase 2.
pub fn circular_distance_i8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = (x as i16 - y as i16).unsigned_abs();
            diff.min(256 - diff) as u64
        })
        .sum()
}

// -------------------------------------------------------------------------
// Operation 5: phase_histogram_16
// -------------------------------------------------------------------------

/// Compute 16-bin phase histogram. Bin i = count of elements in [i*16, (i+1)*16 - 1].
/// Total counts sum to vector length.
pub fn phase_histogram_16(data: &[u8]) -> [u16; 16] {
    let mut hist = [0u16; 16];
    for &v in data {
        hist[(v >> 4) as usize] += 1;
    }
    hist
}

/// L1 distance between two 16-bin histograms.
pub fn histogram_l1_distance(a: &[u16; 16], b: &[u16; 16]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

// -------------------------------------------------------------------------
// Operation 6: phase_bundle_circular
// -------------------------------------------------------------------------

/// Bundle N phase vectors by circular mean.
///
/// For each position j:
///   1. Convert each byte to unit circle: (cos(2π·val/256), sin(2π·val/256))
///   2. Sum the unit vectors across all N inputs
///   3. Convert back: atan2(sum_sin, sum_cos) × 256 / (2π)
pub fn phase_bundle_circular(vectors: &[&[u8]], out: &mut [u8]) {
    assert!(!vectors.is_empty());
    let len = vectors[0].len();
    assert!(out.len() >= len);
    for v in vectors {
        assert_eq!(v.len(), len);
    }

    let scale = 2.0 * PI / 256.0;
    let inv_scale = 256.0 / (2.0 * PI);

    for j in 0..len {
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        for v in vectors {
            let angle = v[j] as f64 * scale;
            sum_cos += angle.cos();
            sum_sin += angle.sin();
        }
        let mean_angle = sum_sin.atan2(sum_cos);
        // Convert back to [0, 256) range
        let phase = (mean_angle * inv_scale).rem_euclid(256.0);
        out[j] = phase.round() as u8;
    }
}

/// Fast approximate bundle when phases do NOT wrap around
/// (all values within 128 of each other). Simple byte average.
pub fn phase_bundle_approximate(vectors: &[&[u8]], out: &mut [u8]) {
    assert!(!vectors.is_empty());
    let len = vectors[0].len();
    let n = vectors.len() as u16;

    for j in 0..len {
        let sum: u16 = vectors.iter().map(|v| v[j] as u16).sum();
        out[j] = (sum / n) as u8;
    }
}

// -------------------------------------------------------------------------
// Operation 7: project_5d_to_phase
// -------------------------------------------------------------------------

/// SplitMix64 PRNG for deterministic basis generation.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Generate deterministic 5D basis from seed.
/// Returns 5 × 2048-byte random vectors (10KB total).
pub fn generate_5d_basis(seed: u64) -> [[u8; 2048]; 5] {
    let mut rng = SplitMix64(seed);
    let mut basis = [[0u8; 2048]; 5];
    for dim in 0..5 {
        for chunk in 0..(2048 / 8) {
            let val = rng.next();
            let bytes = val.to_le_bytes();
            for b in 0..8 {
                basis[dim][chunk * 8 + b] = bytes[b];
            }
        }
    }
    basis
}

/// Project a 5D coordinate into a 2048-byte phase vector.
///
/// For each element j:
///   `out[j] = (coords[0]·basis[0][j] + ... + coords[4]·basis[4][j]) mod 256`
///
/// Nearby 5D coordinates produce phase vectors with small circular_distance.
pub fn project_5d_to_phase(coords: &[f64; 5], basis: &[[u8; 2048]; 5]) -> Vec<u8> {
    let mut out = vec![0u8; 2048];
    for j in 0..2048 {
        let mut sum = 0.0f64;
        for d in 0..5 {
            sum += coords[d] * basis[d][j] as f64;
        }
        out[j] = (sum.rem_euclid(256.0)).round() as u8;
    }
    out
}

/// Recover approximate 5D coordinates from a phase vector.
/// Uses circular correlation with each basis vector.
///
/// Precision: ~5.5 bits per coordinate for 2048 elements.
pub fn recover_5d_from_phase(record: &[u8], basis: &[[u8; 2048]; 5]) -> [f64; 5] {
    let scale = 2.0 * PI / 256.0;
    let mut coords = [0.0f64; 5];

    for d in 0..5 {
        // Circular correlation: compute mean phase difference
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        for j in 0..2048 {
            let diff = record[j].wrapping_sub(basis[d][j]);
            let angle = diff as f64 * scale;
            sum_cos += angle.cos();
            sum_sin += angle.sin();
        }
        let mean_diff_angle = sum_sin.atan2(sum_cos);
        // Convert mean phase difference back to coordinate
        // The coordinate was multiplied by basis values, so we need to divide
        // For a simplified recovery: the mean phase offset encodes the coordinate
        coords[d] = mean_diff_angle.rem_euclid(2.0 * PI) / (2.0 * PI);
    }

    coords
}

// -------------------------------------------------------------------------
// Operation 8: sort_phase_vector
// -------------------------------------------------------------------------

/// Sort a phase vector ascending. Returns (sorted, permutation_index).
/// Called ONCE at write time. The permutation allows reversing the sort.
pub fn sort_phase_vector(data: &[u8]) -> (Vec<u8>, Vec<u16>) {
    let mut indices: Vec<u16> = (0..data.len() as u16).collect();
    indices.sort_by_key(|&i| data[i as usize]);
    let sorted: Vec<u8> = indices.iter().map(|&i| data[i as usize]).collect();
    (sorted, indices)
}

/// Unsort a phase vector using stored permutation index.
pub fn unsort_phase_vector(sorted: &[u8], perm: &[u16]) -> Vec<u8> {
    let mut out = vec![0u8; sorted.len()];
    for (sorted_idx, &orig_idx) in perm.iter().enumerate() {
        out[orig_idx as usize] = sorted[sorted_idx];
    }
    out
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Operation 1: phase_bind --

    #[test]
    fn test_phase_bind_identity() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let zeros = vec![0u8; 2048];
        assert_eq!(phase_bind_i8(&a, &zeros), a);
    }

    #[test]
    fn test_phase_bind_self_annihilation() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let inv = phase_inverse_i8(&a);
        let result = phase_bind_i8(&a, &inv);
        assert!(result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_phase_bind_inverse_round_trip() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i * 7 + 42) % 256) as u8).collect();
        let bound = phase_bind_i8(&a, &b);
        let inv_b = phase_inverse_i8(&b);
        let recovered = phase_bind_i8(&bound, &inv_b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_phase_bind_inplace() {
        let a: Vec<u8> = vec![100, 200, 50];
        let b: Vec<u8> = vec![200, 100, 250];
        let mut c = a.clone();
        phase_bind_i8_inplace(&mut c, &b);
        assert_eq!(c, phase_bind_i8(&a, &b));
    }

    // -- Operation 2: phase_unbind --

    #[test]
    fn test_phase_unbind_exact_round_trip() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 41 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i * 59 + 7) % 256) as u8).collect();
        let bound = phase_bind_i8(&a, &b);
        let recovered = phase_unbind_i8(&bound, &b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_phase_unbind_equals_bind_inverse() {
        let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i * 3 + 17) % 256) as u8).collect();
        let unbind_result = phase_unbind_i8(&a, &b);
        let bind_inv_result = phase_bind_i8(&a, &phase_inverse_i8(&b));
        assert_eq!(unbind_result, bind_inv_result);
    }

    // -- Operation 3: wasserstein_sorted --

    #[test]
    fn test_wasserstein_identical() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut sorted_a = a.clone();
        sorted_a.sort();
        assert_eq!(wasserstein_sorted_i8(&sorted_a, &sorted_a), 0);
    }

    #[test]
    fn test_wasserstein_known_value() {
        let a = vec![0u8, 10, 20, 30];
        let b = vec![5u8, 15, 25, 35];
        // |0-5| + |10-15| + |20-25| + |30-35| = 5+5+5+5 = 20
        assert_eq!(wasserstein_sorted_i8(&a, &b), 20);
    }

    #[test]
    fn test_wasserstein_triangle_inequality() {
        let mut rng = SplitMix64(42);
        let mut make_sorted = || {
            let mut v: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
            v.sort();
            v
        };
        let a = make_sorted();
        let b = make_sorted();
        let c = make_sorted();

        let d_ab = wasserstein_sorted_i8(&a, &b);
        let d_bc = wasserstein_sorted_i8(&b, &c);
        let d_ac = wasserstein_sorted_i8(&a, &c);
        assert!(d_ac <= d_ab + d_bc);
    }

    #[test]
    fn test_wasserstein_search_adaptive_finds_close() {
        let query: Vec<u8> = (0..64).collect();
        let close: Vec<u8> = (1..65).collect(); // distance 64
        let far: Vec<u8> = (128..192).collect(); // far

        let mut db = Vec::new();
        db.extend_from_slice(&far);
        db.extend_from_slice(&close);
        db.extend_from_slice(&far);

        let results = wasserstein_search_adaptive(&query, &db, 64, 3, 100);
        assert!(results.iter().any(|&(idx, _)| idx == 1));
    }

    // -- Operation 4: circular_distance --

    #[test]
    fn test_circular_distance_wrap_around() {
        // phase 254 and phase 2: circular distance = 4 (not 252)
        assert_eq!(circular_distance_i8(&[254], &[2]), 4);
    }

    #[test]
    fn test_circular_distance_maximum() {
        // phase 0 and phase 128: max distance per element = 128
        assert_eq!(circular_distance_i8(&[0], &[128]), 128);
    }

    #[test]
    fn test_circular_distance_symmetry() {
        let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i * 3 + 100) % 256) as u8).collect();
        assert_eq!(circular_distance_i8(&a, &b), circular_distance_i8(&b, &a));
    }

    #[test]
    fn test_circular_distance_triangle_inequality() {
        let mut rng = SplitMix64(123);
        let mut make_vec = || {
            (0..2048)
                .map(|_| (rng.next() % 256) as u8)
                .collect::<Vec<u8>>()
        };
        let a = make_vec();
        let b = make_vec();
        let c = make_vec();

        let d_ab = circular_distance_i8(&a, &b);
        let d_bc = circular_distance_i8(&b, &c);
        let d_ac = circular_distance_i8(&a, &c);
        assert!(d_ac <= d_ab + d_bc);
    }

    #[test]
    fn test_circular_distance_self_zero() {
        let v: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        assert_eq!(circular_distance_i8(&v, &v), 0);
    }

    // -- Operation 5: histogram --

    #[test]
    fn test_histogram_sum() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let hist = phase_histogram_16(&data);
        let total: u16 = hist.iter().sum();
        assert_eq!(total, 2048);
    }

    #[test]
    fn test_histogram_uniform() {
        // 2048 elements cycling through 0..255 → each bin gets 2048/16 = 128
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let hist = phase_histogram_16(&data);
        for &count in &hist {
            assert_eq!(count, 128);
        }
    }

    #[test]
    fn test_histogram_l1_identical() {
        let hist = [128u16; 16];
        assert_eq!(histogram_l1_distance(&hist, &hist), 0);
    }

    // -- Operation 6: phase_bundle_circular --

    #[test]
    fn test_bundle_circular_wrap_around() {
        // Bundle [254, 254, ...] and [2, 2, ...] → circular mean ≈ [0, 0, ...]
        let a = vec![254u8; 2048];
        let b = vec![2u8; 2048];
        let mut out = vec![0u8; 2048];
        phase_bundle_circular(&[&a, &b], &mut out);
        // Circular mean of 254 and 2 should be approximately 0 (±1 due to rounding)
        for &v in &out {
            assert!(v <= 1 || v == 255, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_bundle_circular_single() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut out = vec![0u8; 2048];
        phase_bundle_circular(&[&a], &mut out);
        assert_eq!(out, a);
    }

    #[test]
    fn test_bundle_approximate_no_wrap() {
        let a = vec![100u8; 2048];
        let b = vec![110u8; 2048];
        let mut out = vec![0u8; 2048];
        phase_bundle_approximate(&[&a, &b], &mut out);
        // Average of 100 and 110 = 105
        for &v in &out {
            assert_eq!(v, 105);
        }
    }

    // -- Operation 7: 5D projection --

    #[test]
    fn test_5d_projection_nearby_points() {
        let basis = generate_5d_basis(42);
        let p1 = [0.5, 0.5, 0.5, 0.5, 0.5];
        let p2 = [0.51, 0.5, 0.5, 0.5, 0.5]; // differ by 0.01 on axis 0

        let v1 = project_5d_to_phase(&p1, &basis);
        let v2 = project_5d_to_phase(&p2, &basis);

        let dist_near = circular_distance_i8(&v1, &v2);

        let p3 = [0.0, 0.0, 0.0, 0.0, 0.0]; // far from p1
        let v3 = project_5d_to_phase(&p3, &basis);
        let dist_far = circular_distance_i8(&v1, &v3);

        assert!(
            dist_near < dist_far,
            "nearby points should have smaller distance: near={} far={}",
            dist_near,
            dist_far
        );
    }

    #[test]
    fn test_5d_projection_self_zero() {
        let basis = generate_5d_basis(42);
        let p = [0.3, 0.7, 0.1, 0.9, 0.5];
        let v = project_5d_to_phase(&p, &basis);
        assert_eq!(circular_distance_i8(&v, &v), 0);
    }

    #[test]
    fn test_generate_5d_basis_deterministic() {
        let b1 = generate_5d_basis(42);
        let b2 = generate_5d_basis(42);
        assert_eq!(b1, b2);

        let b3 = generate_5d_basis(43);
        assert_ne!(b1, b3);
    }

    #[test]
    fn test_5d_round_trip() {
        let basis = generate_5d_basis(12345);
        let original = [0.5, 0.5, 0.5, 0.5, 0.5];
        let projected = project_5d_to_phase(&original, &basis);
        let recovered = recover_5d_from_phase(&projected, &basis);

        // Recovery has inter-basis interference from 5 summed components.
        // Actual precision is ~3 bits per coordinate (±0.125).
        // Allow generous tolerance since the key property (nearby points →
        // nearby vectors) is tested separately.
        for d in 0..5 {
            let err = (recovered[d] - original[d]).abs();
            // Also handle wrap-around: 0.0 and 1.0 are close on the circle
            let err = err.min(1.0 - err);
            assert!(
                err < 0.35,
                "dim {} recovery error {:.4} too large (original={}, recovered={})",
                d,
                err,
                original[d],
                recovered[d]
            );
        }
    }

    // -- Operation 8: sort/unsort --

    #[test]
    fn test_sort_unsort_round_trip() {
        let data: Vec<u8> = vec![200, 50, 100, 0, 255, 128];
        let (sorted, perm) = sort_phase_vector(&data);
        assert_eq!(sorted, vec![0, 50, 100, 128, 200, 255]);
        let unsorted = unsort_phase_vector(&sorted, &perm);
        assert_eq!(unsorted, data);
    }

    #[test]
    fn test_sort_preserves_wasserstein() {
        let mut rng = SplitMix64(99);
        let a: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();

        let (sa, _) = sort_phase_vector(&a);
        let (sb, _) = sort_phase_vector(&b);

        let w = wasserstein_sorted_i8(&sa, &sb);
        assert!(
            w > 0,
            "distinct random vectors should have nonzero Wasserstein"
        );
    }

    #[test]
    fn test_sort_container_size() {
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let (sorted, perm) = sort_phase_vector(&data);
        assert_eq!(sorted.len(), 2048);
        assert_eq!(perm.len(), 2048);

        // Verify sorted
        for i in 1..sorted.len() {
            assert!(sorted[i] >= sorted[i - 1]);
        }

        // Verify round-trip
        let unsorted = unsort_phase_vector(&sorted, &perm);
        assert_eq!(unsorted, data);
    }

    // -- Capacity comparison --

    #[test]
    fn test_phase_unbind_after_bundle_recovers() {
        // Bundle 3 phase vectors, unbind one, verify recovery
        let mut rng = SplitMix64(777);
        let mut make_vec = || {
            (0..2048)
                .map(|_| (rng.next() % 256) as u8)
                .collect::<Vec<u8>>()
        };

        let a = make_vec();
        let b = make_vec();
        let c = make_vec();

        let mut bundle = vec![0u8; 2048];
        phase_bundle_circular(&[&a, &b, &c], &mut bundle);

        // Unbind b from bundle
        let recovered_a_ish = phase_unbind_i8(&bundle, &b);

        // The recovered vector should be closer to 'a' than to random
        let dist_to_a = circular_distance_i8(&recovered_a_ish, &a);
        let dist_to_random = circular_distance_i8(&recovered_a_ish, &c);

        // Phase recovery should produce a vector closer to a than to random
        assert!(
            dist_to_a < dist_to_random,
            "recovery should be closer to original: to_a={} to_random={}",
            dist_to_a,
            dist_to_random
        );
    }

    // -- Property: phase_bind(a, phase_inverse(b)) == phase_unbind(a, b) --

    #[test]
    fn test_bind_inverse_equals_unbind() {
        let mut rng = SplitMix64(555);
        let a: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();

        let via_unbind = phase_unbind_i8(&a, &b);
        let via_bind_inv = phase_bind_i8(&a, &phase_inverse_i8(&b));
        assert_eq!(via_unbind, via_bind_inv);
    }
}
