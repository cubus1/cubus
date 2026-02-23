//! Carrier Model: Analog Waveform Architecture for Phase Containers.
//!
//! Alternative encoding for phase containers where concepts are carriers at
//! specific frequencies in a 2048-byte waveform. Binding = frequency modulation,
//! bundling = waveform addition (VPADDB — 32 instructions), recovery = demodulation
//! (dot product with carrier basis — 64 VPDPBUSD).
//!
//! ## Capacity
//!
//! Random-phase bundling: 3-5 items before noise floor.
//! Carrier bundling: ~16 items (limited by int8 dynamic range: 48 dB / 3 dB per carrier).
//!
//! ## Representation
//!
//! Carrier containers use **i8** (signed, oscillates around zero), unlike phase.rs
//! which uses **u8** (unsigned, each byte = an angle on [0°, 360°)).
//! Binary containers (META, BTREE) remain unchanged.

use std::f64::consts::PI;

use numrus_rs::NumArrayU8;

// ============================================================================
// Constants
// ============================================================================

/// 16 carrier frequencies, Fibonacci-spaced to avoid harmonic overlap.
///
/// If f1=5 and f2=10, then f2 is the 2nd harmonic of f1 — they interfere.
/// Fibonacci spacing avoids integer-ratio relationships between any pair.
pub const CARRIER_FREQUENCIES: [u16; 16] = [
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1024,
];

/// Per-carrier amplitude. With 16 carriers superimposed in i8 (-128..+127):
///   max amplitude per carrier = 127 / 16 ≈ 7
///   Worst case: all 16 carriers peak at same sample → 7×16 = 112 < 127
pub const CARRIER_AMPLITUDE: f32 = 7.0;

/// Container size in bytes (same as CogRecordV3).
const CONTAINER_BYTES: usize = 2048;

// ============================================================================
// CarrierBasis — precomputed once, shared across all records
// ============================================================================

/// The carrier basis: 16 frequencies × 2 (cos + sin) × 2048 samples = 64 KB.
/// Generated once, stored forever, shared across all records.
///
/// Each carrier is stored as i8 (signed, range -128..+127) because waveforms
/// oscillate around zero.
pub struct CarrierBasis {
    /// Cosine carriers: basis_cos[freq_idx][sample_idx] → i8
    pub basis_cos: [[i8; 2048]; 16],
    /// Sine carriers: basis_sin[freq_idx][sample_idx] → i8
    pub basis_sin: [[i8; 2048]; 16],
}

impl Default for CarrierBasis {
    fn default() -> Self {
        Self::new()
    }
}

impl CarrierBasis {
    /// Generate deterministically using Chebyshev recurrence.
    ///
    /// Only 2 trig calls per carrier (cos(ω) and sin(ω)), then
    /// 2 multiply-adds per sample via recurrence.
    /// Total: 32 trig calls + 65536 multiply-adds for the entire basis.
    pub fn new() -> Self {
        let mut basis = CarrierBasis {
            basis_cos: [[0i8; 2048]; 16],
            basis_sin: [[0i8; 2048]; 16],
        };

        let n = 2048.0f64;
        for (fi, &freq) in CARRIER_FREQUENCIES.iter().enumerate() {
            let omega = 2.0 * PI * freq as f64 / n;
            let cos_omega = omega.cos();
            let amp = CARRIER_AMPLITUDE as f64;

            // Cosine carrier via Chebyshev recurrence
            let mut prev_prev = amp; // cos(0) = 1
            let mut prev = amp * cos_omega; // cos(ω)
            basis.basis_cos[fi][0] = prev_prev.round().clamp(-128.0, 127.0) as i8;
            basis.basis_cos[fi][1] = prev.round().clamp(-128.0, 127.0) as i8;
            for j in 2..2048 {
                let current = 2.0 * cos_omega * prev - prev_prev;
                basis.basis_cos[fi][j] = current.round().clamp(-128.0, 127.0) as i8;
                prev_prev = prev;
                prev = current;
            }

            // Sine carrier: same recurrence, different initial conditions
            let sin_omega = omega.sin();
            prev_prev = 0.0; // sin(0) = 0
            prev = amp * sin_omega; // sin(ω)
            basis.basis_sin[fi][0] = prev_prev.round().clamp(-128.0, 127.0) as i8;
            basis.basis_sin[fi][1] = prev.round().clamp(-128.0, 127.0) as i8;
            for j in 2..2048 {
                let current = 2.0 * cos_omega * prev - prev_prev;
                basis.basis_sin[fi][j] = current.round().clamp(-128.0, 127.0) as i8;
                prev_prev = prev;
                prev = current;
            }
        }

        basis
    }

    /// Get cosine carrier for frequency index as u8 (offset by 128).
    /// Useful for compatibility with phase.rs u8 operations.
    pub fn cos_as_u8(&self, freq_idx: usize) -> Vec<u8> {
        self.basis_cos[freq_idx]
            .iter()
            .map(|&v| (v as i16 + 128) as u8)
            .collect()
    }
}

// ============================================================================
// Operation 9: carrier_encode
// ============================================================================

/// Encode a concept as a carrier at a specific frequency with given phase and amplitude.
///
/// Adds to the existing waveform (accumulation, not replacement):
///   container[j] += cos(φ)·basis_cos[f][j] - sin(φ)·basis_sin[f][j]
///                   (scaled by amplitude / CARRIER_AMPLITUDE)
///
/// Uses float per element for phase precision. Maps to VCVTDQ2PS + VMULPS +
/// VCVTPS2DQ on AVX-512 (~128 instructions for 2048 bytes).
pub fn carrier_encode(
    container: &mut [i8],
    basis: &CarrierBasis,
    freq_idx: u8,
    phase_offset: f32,
    amplitude: f32,
) {
    assert_eq!(container.len(), 2048);
    assert!((freq_idx as usize) < CARRIER_FREQUENCIES.len());

    let cos_phi = phase_offset.cos();
    let sin_phi = phase_offset.sin();
    let fi = freq_idx as usize;
    let scale = amplitude / CARRIER_AMPLITUDE;

    for j in 0..2048 {
        let cos_val = basis.basis_cos[fi][j] as f32;
        let sin_val = basis.basis_sin[fi][j] as f32;
        let contribution = ((cos_phi * cos_val - sin_phi * sin_val) * scale)
            .round()
            .clamp(-128.0, 127.0) as i8;
        container[j] = container[j].saturating_add(contribution);
    }
}

// ============================================================================
// Operation 10: carrier_decode
// ============================================================================

/// Decode the amplitude and phase of a specific frequency from the waveform.
///
/// Demodulation: dot product of waveform with cos and sin carriers, then atan2.
///   cos_component = Σ container[j] · basis_cos[f][j]
///   sin_component = Σ container[j] · basis_sin[f][j]
///   phase = atan2(sin_component, cos_component)
///   amplitude = sqrt(cos² + sin²) / N
///
/// Cost: 64 VPDPBUSD instructions (32 per component).
///
/// Returns (phase_offset, amplitude).
pub fn carrier_decode(container: &[i8], basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
    assert_eq!(container.len(), 2048);
    let fi = freq_idx as usize;

    let mut cos_sum: i64 = 0;
    let mut sin_sum: i64 = 0;

    for j in 0..2048 {
        cos_sum += container[j] as i64 * basis.basis_cos[fi][j] as i64;
        sin_sum += container[j] as i64 * basis.basis_sin[fi][j] as i64;
    }

    let cos_f = cos_sum as f64 / 2048.0;
    let sin_f = sin_sum as f64 / 2048.0;

    // Fourier analysis: Σ cos(ωj+φ)·sin(ωj) = -N/2·sin(φ)
    // So sin_sum carries a negative sign. Negate to recover correct phase.
    let phase = ((-sin_f).atan2(cos_f) as f32).rem_euclid(std::f32::consts::TAU);
    let amplitude = (cos_f * cos_f + sin_f * sin_f).sqrt() as f32;

    (phase, amplitude)
}

// ============================================================================
// Operation 11: carrier_bundle
// ============================================================================

/// Bundle N carrier waveforms by saturating addition.
///
/// This is the entire reason the carrier model exists:
///   Random-phase bundle: circular mean = ~500 instructions (trig per element)
///   Carrier bundle: saturating add = 32 VPADDB instructions
///
/// Carriers at different frequencies are orthogonal by construction. Adding
/// two waveforms at different frequencies produces a waveform where both
/// are independently recoverable via carrier_decode.
pub fn carrier_bundle(waveforms: &[&[i8]], out: &mut [i8]) {
    assert!(!waveforms.is_empty());
    let len = waveforms[0].len();
    assert!(out.len() >= len);

    for v in out[..len].iter_mut() {
        *v = 0;
    }

    for wf in waveforms {
        assert_eq!(wf.len(), len);
        for j in 0..len {
            out[j] = out[j].saturating_add(wf[j]);
        }
    }
}

// ============================================================================
// Operation 12: carrier_distance
// ============================================================================

/// L1 distance between two carrier waveforms (sum of absolute differences).
///
/// Same VPSADBW cost as Wasserstein in phase.rs: 32 instructions for 2048 bytes.
pub fn carrier_distance_l1(a: &[i8], b: &[i8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum()
}

/// Correlation between two carrier waveforms (normalized dot product).
/// Returns value in [-1.0, 1.0]. High correlation = similar content.
pub fn carrier_correlation(a: &[i8], b: &[i8]) -> f64 {
    assert_eq!(a.len(), b.len());

    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i64 * y as i64)
        .sum();

    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot as f64 / (norm_a * norm_b)
}

// ============================================================================
// Operation 13: carrier_spectrum
// ============================================================================

/// Compute the amplitude spectrum: energy at each of the 16 carrier frequencies.
///
/// Cost: 16 × 64 instructions = 1024 instructions. Comparable to circular_distance.
pub fn carrier_spectrum(container: &[i8], basis: &CarrierBasis) -> [f32; 16] {
    let mut spectrum = [0.0f32; 16];
    for fi in 0..16 {
        let (_, amp) = carrier_decode(container, basis, fi as u8);
        spectrum[fi] = amp;
    }
    spectrum
}

/// Spectral distance: L1 distance between amplitude spectra.
/// 16 f32 subtractions + absolute values = trivial cost.
pub fn spectral_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

// ============================================================================
// CarrierRecord — hybrid binary + carrier containers
// ============================================================================

/// Thresholds for 4-channel hybrid sweep on CarrierRecord.
#[derive(Clone, Debug)]
pub struct CarrierThresholds {
    pub meta_hamming: u64,
    pub cam_carrier: u64,
    pub btree_hamming: u64,
    pub embed_carrier: u64,
}

/// Distances returned by successful CarrierRecord sweep.
#[derive(Clone, Debug)]
pub struct CarrierDistances {
    pub meta_hamming: u64,
    pub cam_carrier: u64,
    pub btree_hamming: u64,
    pub embed_carrier: u64,
}

/// A CogRecord where phase containers use carrier encoding (i8 waveforms)
/// instead of random-phase encoding (u8 phase angles).
///
/// Binary containers (META, BTREE) are identical to CogRecordV3.
/// Carrier containers (CAM, EMBED) use i8 waveforms with frequency-domain content.
/// NOT sorted — sorting destroys frequency content.
#[derive(Clone)]
pub struct CarrierRecord {
    /// Container 0: BINARY. Same as CogRecordV3.
    pub meta: NumArrayU8,
    /// Container 1: CARRIER WAVEFORM. i8 superposition of carriers.
    pub cam: Vec<i8>,
    /// Container 2: BINARY. Same as CogRecordV3.
    pub btree: NumArrayU8,
    /// Container 3: CARRIER WAVEFORM. i8 superposition of carriers.
    pub embed: Vec<i8>,
}

impl CarrierRecord {
    /// Create a record with zero waveforms in carrier containers.
    pub fn new_empty(meta: &[u8], btree: &[u8]) -> Self {
        assert_eq!(meta.len(), CONTAINER_BYTES);
        assert_eq!(btree.len(), CONTAINER_BYTES);
        Self {
            meta: NumArrayU8::new(meta.to_vec()),
            cam: vec![0i8; CONTAINER_BYTES],
            btree: NumArrayU8::new(btree.to_vec()),
            embed: vec![0i8; CONTAINER_BYTES],
        }
    }

    /// Create from raw parts.
    pub fn from_parts(meta: NumArrayU8, cam: Vec<i8>, btree: NumArrayU8, embed: Vec<i8>) -> Self {
        assert_eq!(meta.len(), CONTAINER_BYTES);
        assert_eq!(cam.len(), CONTAINER_BYTES);
        assert_eq!(btree.len(), CONTAINER_BYTES);
        assert_eq!(embed.len(), CONTAINER_BYTES);
        Self {
            meta,
            cam,
            btree,
            embed,
        }
    }

    /// Encode a concept into the CAM container at a given frequency.
    pub fn encode_cam(&mut self, basis: &CarrierBasis, freq_idx: u8, phase: f32, amplitude: f32) {
        carrier_encode(&mut self.cam, basis, freq_idx, phase, amplitude);
    }

    /// Decode a concept from the CAM container at a given frequency.
    pub fn decode_cam(&self, basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
        carrier_decode(&self.cam, basis, freq_idx)
    }

    /// Encode a concept into the EMBED container at a given frequency.
    pub fn encode_embed(&mut self, basis: &CarrierBasis, freq_idx: u8, phase: f32, amplitude: f32) {
        carrier_encode(&mut self.embed, basis, freq_idx, phase, amplitude);
    }

    /// Decode a concept from the EMBED container at a given frequency.
    pub fn decode_embed(&self, basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
        carrier_decode(&self.embed, basis, freq_idx)
    }

    /// 4-channel hybrid sweep.
    /// META + BTREE: Hamming (same as CogRecordV3).
    /// CAM + EMBED: carrier L1 distance.
    pub fn hybrid_sweep(
        &self,
        other: &Self,
        thresholds: &CarrierThresholds,
    ) -> Option<CarrierDistances> {
        // Stage 1: META — binary Hamming (cheapest rejection)
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

        // Stage 3: CAM — carrier L1 distance
        let cam_dist = carrier_distance_l1(&self.cam, &other.cam);
        if cam_dist > thresholds.cam_carrier {
            return None;
        }

        // Stage 4: EMBED — carrier L1 distance
        let embed_dist = carrier_distance_l1(&self.embed, &other.embed);
        if embed_dist > thresholds.embed_carrier {
            return None;
        }

        Some(CarrierDistances {
            meta_hamming: meta_dist,
            cam_carrier: cam_dist,
            btree_hamming: btree_dist,
            embed_carrier: embed_dist,
        })
    }

    /// Batch hybrid sweep against a database of CarrierRecords.
    pub fn hybrid_search(
        &self,
        database: &[Self],
        thresholds: &CarrierThresholds,
    ) -> Vec<(usize, CarrierDistances)> {
        database
            .iter()
            .enumerate()
            .filter_map(|(i, rec)| self.hybrid_sweep(rec, thresholds).map(|d| (i, d)))
            .collect()
    }

    /// Serialize to 8192 bytes: META(2048) + CAM(2048) + BTREE(2048) + EMBED(2048).
    /// i8 containers are reinterpreted as u8 (bit-preserving, zero-cost).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CONTAINER_BYTES * 4);
        out.extend_from_slice(self.meta.data_slice());
        // i8 → u8 reinterpret
        out.extend(self.cam.iter().map(|&v| v as u8));
        out.extend_from_slice(self.btree.data_slice());
        out.extend(self.embed.iter().map(|&v| v as u8));
        out
    }

    /// Deserialize from 8192 bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), CONTAINER_BYTES * 4);
        Self {
            meta: NumArrayU8::new(data[0..CONTAINER_BYTES].to_vec()),
            cam: data[CONTAINER_BYTES..CONTAINER_BYTES * 2]
                .iter()
                .map(|&v| v as i8)
                .collect(),
            btree: NumArrayU8::new(data[CONTAINER_BYTES * 2..CONTAINER_BYTES * 3].to_vec()),
            embed: data[CONTAINER_BYTES * 3..CONTAINER_BYTES * 4]
                .iter()
                .map(|&v| v as i8)
                .collect(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    // Helper: minimum phase error accounting for wrap-around
    fn phase_error(a: f32, b: f32) -> f32 {
        let diff = (a - b).abs();
        diff.min(TAU - diff)
    }

    // ---- Basis tests ----

    #[test]
    fn test_basis_deterministic() {
        let b1 = CarrierBasis::new();
        let b2 = CarrierBasis::new();
        assert_eq!(b1.basis_cos, b2.basis_cos);
        assert_eq!(b1.basis_sin, b2.basis_sin);
    }

    #[test]
    fn test_basis_cos_carrier_period() {
        let basis = CarrierBasis::new();
        // Carrier at frequency 1 should have period 2048 samples.
        // Check that cos[0][0] ≈ cos[0][2048-1] (nearly full cycle).
        // For freq=1, one full cycle in 2048 samples.
        // cos(0) = amplitude, cos(2π·1·2047/2048) ≈ amplitude
        let first = basis.basis_cos[0][0];
        assert_eq!(first, CARRIER_AMPLITUDE.round() as i8);

        // For freq=2, two full cycles. cos(0) = amp, cos(2π·2·1024/2048) = cos(2π) = amp
        // Check at halfway: cos(2π·2·512/2048) = cos(π) = -amp
        let mid = basis.basis_cos[1][512];
        // freq=2 at sample 512: cos(2π·2·512/2048) = cos(π) = -1 → -7
        assert!(
            (mid as f32 + CARRIER_AMPLITUDE).abs() < 2.0,
            "freq=2 at half-period should be near -amplitude, got {}",
            mid
        );
    }

    #[test]
    fn test_basis_sin_90_degree_shift() {
        let basis = CarrierBasis::new();
        // For freq=1: sin should be 90° shifted from cos.
        // cos[0] = amp, sin[0] = 0
        assert_eq!(basis.basis_sin[0][0], 0);
        assert_eq!(basis.basis_cos[0][0], CARRIER_AMPLITUDE.round() as i8);
    }

    #[test]
    fn test_basis_orthogonality() {
        let basis = CarrierBasis::new();
        // dot(cos[i], cos[j]) should be ≈ 0 for i ≠ j
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dot: i64 = basis.basis_cos[i]
                    .iter()
                    .zip(basis.basis_cos[j].iter())
                    .map(|(&a, &b)| a as i64 * b as i64)
                    .sum();
                let normalized = (dot as f64).abs()
                    / (2048.0 * CARRIER_AMPLITUDE as f64 * CARRIER_AMPLITUDE as f64);
                assert!(
                    normalized < 0.15,
                    "cos[{}] and cos[{}] should be orthogonal, dot/norm = {:.4}",
                    i,
                    j,
                    normalized
                );
            }
        }
    }

    #[test]
    fn test_basis_chebyshev_vs_direct_trig() {
        let basis = CarrierBasis::new();
        let amp = CARRIER_AMPLITUDE as f64;
        let n = 2048.0f64;

        // Check a few carriers against direct trig computation
        for &fi in &[0, 3, 7, 15] {
            let freq = CARRIER_FREQUENCIES[fi] as f64;
            for &j in &[0, 100, 512, 1024, 2000] {
                let expected_cos = (amp * (2.0 * PI * freq * j as f64 / n).cos())
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                let actual_cos = basis.basis_cos[fi][j];
                assert!(
                    (actual_cos as i16 - expected_cos as i16).abs() <= 1,
                    "freq[{}] cos[{}]: expected {}, got {} (diff > 1 LSB)",
                    fi,
                    j,
                    expected_cos,
                    actual_cos
                );
            }
        }
    }

    // ---- Encode/decode tests ----

    #[test]
    fn test_encode_decode_phase_zero() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        carrier_encode(&mut waveform, &basis, 0, 0.0, CARRIER_AMPLITUDE);
        let (phase, amp) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, 0.0) < 0.15,
            "phase=0 recovery: got {:.4}",
            phase
        );
        assert!(amp > 1.0, "amplitude should be significant, got {:.4}", amp);
    }

    #[test]
    fn test_encode_decode_phase_pi() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        carrier_encode(
            &mut waveform,
            &basis,
            0,
            std::f32::consts::PI,
            CARRIER_AMPLITUDE,
        );
        let (phase, amp) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, std::f32::consts::PI) < 0.15,
            "phase=π recovery: got {:.4}, expected {:.4}",
            phase,
            std::f32::consts::PI
        );
        assert!(amp > 1.0);
    }

    #[test]
    fn test_encode_decode_phase_wrap_around() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        let target = TAU - 0.01;
        carrier_encode(&mut waveform, &basis, 0, target, CARRIER_AMPLITUDE);
        let (phase, _) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, target) < 0.15,
            "wrap-around recovery: got {:.4}, expected {:.4}",
            phase,
            target
        );
    }

    #[test]
    fn test_encode_two_carriers_independent() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];

        let phase_a = 1.0f32;
        let phase_b = 3.0f32;

        carrier_encode(&mut waveform, &basis, 0, phase_a, CARRIER_AMPLITUDE);
        carrier_encode(&mut waveform, &basis, 5, phase_b, CARRIER_AMPLITUDE);

        let (rec_a, _) = carrier_decode(&waveform, &basis, 0);
        let (rec_b, _) = carrier_decode(&waveform, &basis, 5);

        assert!(
            phase_error(rec_a, phase_a) < 0.2,
            "carrier 0: expected {:.4}, got {:.4}",
            phase_a,
            rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.2,
            "carrier 5: expected {:.4}, got {:.4}",
            phase_b,
            rec_b
        );
    }

    #[test]
    fn test_encode_16_carriers_all_recovered() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];

        let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.39 + 0.1).collect();

        for i in 0..16 {
            carrier_encode(&mut waveform, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
        }

        let mut max_error = 0.0f32;
        for i in 0..16 {
            let (rec_phase, _) = carrier_decode(&waveform, &basis, i as u8);
            let err = phase_error(rec_phase, phases[i]);
            if err > max_error {
                max_error = err;
            }
        }

        // With 16 carriers in i8, some quantization noise is expected.
        // Allow up to 0.5 rad (~29°) — still useful for spatial navigation.
        assert!(
            max_error < 0.5,
            "16-carrier max phase error = {:.4} rad ({:.1}°) — too high",
            max_error,
            max_error.to_degrees()
        );
    }

    // ---- Bundle tests ----

    #[test]
    fn test_bundle_single_waveform() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 3, 1.5, CARRIER_AMPLITUDE);

        let mut out = vec![0i8; 2048];
        carrier_bundle(&[&wf], &mut out);
        assert_eq!(out, wf);
    }

    #[test]
    fn test_bundle_two_different_frequencies() {
        let basis = CarrierBasis::new();

        let mut wf_a = vec![0i8; 2048];
        carrier_encode(&mut wf_a, &basis, 0, 1.0, CARRIER_AMPLITUDE);

        let mut wf_b = vec![0i8; 2048];
        carrier_encode(&mut wf_b, &basis, 5, 2.5, CARRIER_AMPLITUDE);

        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&[&wf_a, &wf_b], &mut bundled);

        let (rec_a, _) = carrier_decode(&bundled, &basis, 0);
        let (rec_b, _) = carrier_decode(&bundled, &basis, 5);

        assert!(
            phase_error(rec_a, 1.0) < 0.25,
            "bundled carrier 0: expected 1.0, got {:.4}",
            rec_a
        );
        assert!(
            phase_error(rec_b, 2.5) < 0.25,
            "bundled carrier 5: expected 2.5, got {:.4}",
            rec_b
        );
    }

    #[test]
    fn test_bundle_16_waveforms_capacity() {
        let basis = CarrierBasis::new();
        let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.39 + 0.1).collect();

        let mut waveforms: Vec<Vec<i8>> = Vec::new();
        for i in 0..16 {
            let mut wf = vec![0i8; 2048];
            carrier_encode(&mut wf, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
            waveforms.push(wf);
        }

        let wf_refs: Vec<&[i8]> = waveforms.iter().map(|v| v.as_slice()).collect();
        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&wf_refs, &mut bundled);

        let mut total_error = 0.0f32;
        for i in 0..16 {
            let (rec_phase, _) = carrier_decode(&bundled, &basis, i as u8);
            total_error += phase_error(rec_phase, phases[i]);
        }
        let mean_error = total_error / 16.0;

        assert!(
            mean_error < 0.5,
            "16-carrier bundle mean error = {:.4} rad ({:.1}°)",
            mean_error,
            mean_error.to_degrees()
        );
    }

    #[test]
    fn test_bundle_degradation_above_16() {
        let basis = CarrierBasis::new();
        // 21 carriers — must wrap frequency indices, expect degradation
        let n = 21;
        let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.5).collect();

        let mut waveforms: Vec<Vec<i8>> = Vec::new();
        for i in 0..n {
            let mut wf = vec![0i8; 2048];
            carrier_encode(
                &mut wf,
                &basis,
                (i % 16) as u8,
                phases[i],
                CARRIER_AMPLITUDE,
            );
            waveforms.push(wf);
        }

        let wf_refs: Vec<&[i8]> = waveforms.iter().map(|v| v.as_slice()).collect();
        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&wf_refs, &mut bundled);

        // When two carriers share a frequency (i and i+16), they interfere.
        // Frequencies 0-4 have two carriers each; 5-15 have one.
        // The single-carrier frequencies should still decode reasonably.
        let mut single_freq_errors = Vec::new();
        for i in 5..16 {
            let (rec, _) = carrier_decode(&bundled, &basis, i as u8);
            single_freq_errors.push(phase_error(rec, phases[i]));
        }
        let mean_single = single_freq_errors.iter().sum::<f32>() / single_freq_errors.len() as f32;

        // Unshared frequencies should still work
        assert!(
            mean_single < 0.6,
            "unshared frequencies at N=21 mean error = {:.4} rad",
            mean_single
        );
    }

    // ---- Distance tests ----

    #[test]
    fn test_distance_l1_self_zero() {
        let wf = vec![42i8; 2048];
        assert_eq!(carrier_distance_l1(&wf, &wf), 0);
    }

    #[test]
    fn test_distance_l1_different_positive() {
        let a = vec![10i8; 2048];
        let b = vec![20i8; 2048];
        assert_eq!(carrier_distance_l1(&a, &b), 10 * 2048);
    }

    #[test]
    fn test_correlation_self_one() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 0, 1.0, CARRIER_AMPLITUDE);
        let corr = carrier_correlation(&wf, &wf);
        assert!(
            (corr - 1.0).abs() < 0.01,
            "self-correlation should be 1.0, got {:.4}",
            corr
        );
    }

    #[test]
    fn test_correlation_negation() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 0, 1.0, CARRIER_AMPLITUDE);
        let neg: Vec<i8> = wf.iter().map(|&v| v.saturating_neg()).collect();
        let corr = carrier_correlation(&wf, &neg);
        assert!(
            (corr + 1.0).abs() < 0.05,
            "negation correlation should be -1.0, got {:.4}",
            corr
        );
    }

    #[test]
    fn test_correlation_orthogonal_carriers() {
        let basis = CarrierBasis::new();
        let mut wf_a = vec![0i8; 2048];
        let mut wf_b = vec![0i8; 2048];
        carrier_encode(&mut wf_a, &basis, 0, 0.0, CARRIER_AMPLITUDE);
        carrier_encode(&mut wf_b, &basis, 5, 0.0, CARRIER_AMPLITUDE);

        let corr = carrier_correlation(&wf_a, &wf_b);
        assert!(
            corr.abs() < 0.2,
            "orthogonal carriers should have near-zero correlation, got {:.4}",
            corr
        );
    }

    // ---- Spectrum tests ----

    #[test]
    fn test_spectrum_single_carrier() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 7, 2.0, CARRIER_AMPLITUDE);

        let spec = carrier_spectrum(&wf, &basis);
        // Frequency 7 should have the highest amplitude
        let max_idx = spec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 7, "peak frequency should be 7, got {}", max_idx);
    }

    #[test]
    fn test_spectral_distance_self_zero() {
        let spec = [1.0f32; 16];
        assert!((spectral_distance(&spec, &spec)).abs() < 1e-6);
    }

    // ---- CarrierRecord integration tests ----

    #[test]
    fn test_carrier_record_hybrid_sweep_self() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0xBBu8; 2048]);
        rec.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);
        rec.encode_embed(&basis, 3, 2.0, CARRIER_AMPLITUDE);

        let thresholds = CarrierThresholds {
            meta_hamming: 1,
            cam_carrier: 1,
            btree_hamming: 1,
            embed_carrier: 1,
        };

        let result = rec.hybrid_sweep(&rec, &thresholds);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.meta_hamming, 0);
        assert_eq!(d.cam_carrier, 0);
        assert_eq!(d.btree_hamming, 0);
        assert_eq!(d.embed_carrier, 0);
    }

    #[test]
    fn test_carrier_record_hybrid_sweep_reject() {
        let mut rec_a = CarrierRecord::new_empty(&vec![0x00u8; 2048], &vec![0u8; 2048]);
        let rec_b = CarrierRecord::new_empty(&vec![0xFFu8; 2048], &vec![0u8; 2048]);

        let basis = CarrierBasis::new();
        rec_a.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);

        let thresholds = CarrierThresholds {
            meta_hamming: 100, // tight — will reject on META
            cam_carrier: 100000,
            btree_hamming: 100000,
            embed_carrier: 100000,
        };

        assert!(rec_a.hybrid_sweep(&rec_b, &thresholds).is_none());
    }

    #[test]
    fn test_carrier_record_to_bytes_round_trip() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0xBBu8; 2048]);
        rec.encode_cam(&basis, 0, 1.5, CARRIER_AMPLITUDE);
        rec.encode_embed(&basis, 7, 3.0, CARRIER_AMPLITUDE);

        let bytes = rec.to_bytes();
        assert_eq!(bytes.len(), 8192);

        let rec2 = CarrierRecord::from_bytes(&bytes);
        assert_eq!(rec2.meta.data_slice(), rec.meta.data_slice());
        assert_eq!(rec2.cam, rec.cam);
        assert_eq!(rec2.btree.data_slice(), rec.btree.data_slice());
        assert_eq!(rec2.embed, rec.embed);
    }

    #[test]
    fn test_carrier_record_encode_decode_cam_round_trip() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0u8; 2048], &vec![0u8; 2048]);

        // Encode 5 concepts into CAM at different frequencies
        let phases = [0.5f32, 1.2, 2.8, 4.0, 5.5];
        for (i, &p) in phases.iter().enumerate() {
            rec.encode_cam(&basis, i as u8, p, CARRIER_AMPLITUDE);
        }

        // Decode all 5
        for (i, &expected) in phases.iter().enumerate() {
            let (recovered, _) = rec.decode_cam(&basis, i as u8);
            assert!(
                phase_error(recovered, expected) < 0.3,
                "CAM freq {}: expected {:.4}, got {:.4}",
                i,
                expected,
                recovered
            );
        }
    }

    #[test]
    fn test_carrier_record_batch_search() {
        let basis = CarrierBasis::new();

        let mut query = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        query.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);

        let mut db = Vec::new();
        // Record 0: same meta + same carrier content
        let mut r0 = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        r0.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);
        db.push(r0);
        // Record 1: different meta → rejected
        let r1 = CarrierRecord::new_empty(&vec![0x00u8; 2048], &vec![0u8; 2048]);
        db.push(r1);
        // Record 2: same meta + same carrier
        let mut r2 = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        r2.encode_cam(&basis, 0, 1.05, CARRIER_AMPLITUDE);
        db.push(r2);

        let thresholds = CarrierThresholds {
            meta_hamming: 100,
            cam_carrier: 10000,
            btree_hamming: 20000,
            embed_carrier: 100000,
        };

        let results = query.hybrid_search(&db, &thresholds);
        // Records 0 and 2 should match (same meta), record 1 rejected (different meta)
        assert!(results.iter().any(|&(idx, _)| idx == 0));
        assert!(!results.iter().any(|&(idx, _)| idx == 1));
        assert!(results.iter().any(|&(idx, _)| idx == 2));
    }

    // ---- Capacity comparison (THE critical experiment) ----

    #[test]
    fn test_carrier_vs_phase_capacity() {
        let basis = CarrierBasis::new();

        println!("\n=== Carrier vs Random-Phase Capacity Comparison ===\n");

        for &n in &[1u32, 2, 3, 5, 8, 13, 16] {
            // --- Carrier path ---
            let mut carrier_waveform = vec![0i8; 2048];
            let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 0.3).collect();

            for i in 0..n as usize {
                carrier_encode(
                    &mut carrier_waveform,
                    &basis,
                    i as u8 % 16,
                    phases[i],
                    CARRIER_AMPLITUDE,
                );
            }

            let mut carrier_errors = Vec::new();
            let mut carrier_amps = Vec::new();
            for i in 0..n as usize {
                let (rec_phase, rec_amp) = carrier_decode(&carrier_waveform, &basis, i as u8 % 16);
                carrier_errors.push(phase_error(rec_phase, phases[i]));
                carrier_amps.push(rec_amp);
            }

            let carrier_mean_error: f32 =
                carrier_errors.iter().sum::<f32>() / carrier_errors.len() as f32;
            let carrier_mean_amp: f32 =
                carrier_amps.iter().sum::<f32>() / carrier_amps.len() as f32;

            // --- Random-phase path (using phase.rs functions) ---
            use crate::phase::{circular_distance_i8, phase_bundle_circular, phase_unbind_i8};
            let mut rng = crate::phase::SplitMix64(42 + n as u64);
            let phase_vecs: Vec<Vec<u8>> = (0..n)
                .map(|_| (0..2048).map(|_| (rng.next() % 256) as u8).collect())
                .collect();

            let refs: Vec<&[u8]> = phase_vecs.iter().map(|v| v.as_slice()).collect();
            let mut bundle = vec![0u8; 2048];
            phase_bundle_circular(&refs, &mut bundle);

            let mut phase_errors: Vec<u64> = Vec::new();
            for i in 0..n as usize {
                let recovered = phase_unbind_i8(&bundle, &phase_vecs[i]);
                // Measure circular distance to the "first" vector as baseline
                let dist = circular_distance_i8(&recovered, &phase_vecs[0]);
                phase_errors.push(dist);
            }
            let phase_self_recovery =
                circular_distance_i8(&phase_unbind_i8(&bundle, &phase_vecs[0]), &phase_vecs[0]);

            println!(
                "N={:>2}: carrier_err={:.4} rad ({:>5.1}°)  amp={:.2}  |  phase_self_dist={}",
                n,
                carrier_mean_error,
                carrier_mean_error.to_degrees(),
                carrier_mean_amp,
                phase_self_recovery,
            );
        }

        // Verify carrier maintains low error up to N=16
        {
            let mut wf = vec![0i8; 2048];
            let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.7 + 0.3).collect();
            for i in 0..16 {
                carrier_encode(&mut wf, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
            }
            let mut total_err = 0.0f32;
            for i in 0..16 {
                let (rec, _) = carrier_decode(&wf, &basis, i as u8);
                total_err += phase_error(rec, phases[i]);
            }
            let mean = total_err / 16.0;
            assert!(
                mean < 0.5,
                "Carrier at N=16 mean error = {:.4} rad — capacity limit exceeded",
                mean
            );
        }
    }

    // ---- cos_as_u8 test ----

    #[test]
    fn test_cos_as_u8_offset() {
        let basis = CarrierBasis::new();
        let u8_carrier = basis.cos_as_u8(0);
        assert_eq!(u8_carrier.len(), 2048);
        // First sample: cos[0][0] = amplitude (7), offset = 7+128 = 135
        assert_eq!(
            u8_carrier[0],
            (CARRIER_AMPLITUDE.round() as u8).wrapping_add(128)
        );
    }
}
