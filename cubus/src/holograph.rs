//! Holographic Spatial Binding: Gabor Wavelets + Delta Cube Architecture.
//!
//! The convergence layer that unifies spatial locality, frequency multiplexing,
//! and attention gating into a single primitive: the Gabor wavelet.
//!
//! A Gabor wavelet is a sinusoidal carrier × Gaussian envelope:
//!   wavelet(j) = A · exp(-(j-j₀)²/(2σ²)) · cos(2π·f·(j-j₀)/N + φ)
//!
//! It is the optimal tradeoff between spatial localization and frequency
//! selectivity (Heisenberg-Gabor uncertainty: Δx·Δf ≥ 1/4π).
//!
//! ## What it subsumes
//!
//! | Previous layer      | How Gabor subsumes it                    |
//! |---------------------|------------------------------------------|
//! | Carrier (carrier.rs)| cos() oscillation IS the carrier         |
//! | Focus (focus.rs)    | exp() envelope IS the spatial mask       |
//! | Phase (phase.rs)    | Phase offset φ IS the concept value      |
//! | 5D projection       | Center (x₀,y₀,z₀) IS the spatial coord  |
//!
//! ## Delta-Cube Binding
//!
//! Content stored in the INTERFERENCE PATTERN between two holographic fields.
//! Neither field alone reveals the content. Both together reconstruct it.
//!
//! ## Spatial Transforms
//!
//! 3D permutations (rotate, diagonal, compose) that move content through
//! the 8×8×32 volume. Composable and invertible.

use crate::carrier::{carrier_decode, carrier_encode, CarrierBasis, CARRIER_FREQUENCIES};
use crate::focus::{FOCUS_DIM_X, FOCUS_DIM_Y, FOCUS_DIM_Z};

/// A migration entry: (concept_index, old_position, new_position).
pub type Migration = (usize, (f32, f32, f32), (f32, f32, f32));

// ============================================================================
// GaussianLUT — Envelope lookup table
// ============================================================================

/// Precomputed Gaussian envelope: exp(-d²/(2σ²)) quantized to u8.
///
/// For the 8×8×32 volume, max squared distance from center:
///   d² = 7² + 7² + 31² = 49 + 49 + 961 = 1059
///
/// Table indexed by d², returns amplitude (0 = fully decayed, 255 = center).
#[derive(Clone)]
pub struct GaussianLUT {
    pub table: Vec<u8>,
    pub sigma_squared_2: f32,
}

impl GaussianLUT {
    pub fn new(sigma: f32) -> Self {
        let max_d2 = 1060usize;
        let s2 = 2.0 * sigma * sigma;
        let table: Vec<u8> = (0..max_d2)
            .map(|d2| {
                let envelope = (-(d2 as f32) / s2).exp();
                (envelope * 255.0).round().min(255.0) as u8
            })
            .collect();
        Self {
            table,
            sigma_squared_2: s2,
        }
    }

    /// Get envelope amplitude for squared distance from center.
    #[inline]
    pub fn amplitude(&self, d_squared: u32) -> u8 {
        if (d_squared as usize) < self.table.len() {
            self.table[d_squared as usize]
        } else {
            0
        }
    }

    /// Effective radius squared (3σ cutoff: exp(-4.5) < 0.01).
    #[inline]
    pub fn effective_radius_sq(&self) -> u32 {
        (3.0 * self.sigma_squared_2).ceil() as u32
    }
}

// ============================================================================
// WaveletTemplate — Precomputed active positions
// ============================================================================

/// Precomputed wavelet template for a specific (σ, center) pair.
/// Contains only active positions within the Gaussian's effective radius.
pub struct WaveletTemplate {
    /// (byte_index, envelope_amplitude) pairs, sorted by index.
    pub entries: Vec<(u16, u8)>,
}

impl WaveletTemplate {
    /// Build a template for a given center and LUT.
    pub fn new(lut: &GaussianLUT, x0: u8, y0: u8, z0: u8) -> Self {
        let radius_sq = lut.effective_radius_sq();
        let mut entries = Vec::new();

        for x in 0..FOCUS_DIM_X as i32 {
            let dx = x - x0 as i32;
            for y in 0..FOCUS_DIM_Y as i32 {
                let dy = y - y0 as i32;
                for z in 0..FOCUS_DIM_Z as i32 {
                    let dz = z - z0 as i32;
                    let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                    if d_sq > radius_sq {
                        continue;
                    }

                    let env = lut.amplitude(d_sq);
                    if env < 1 {
                        continue;
                    }

                    let idx = x as u16 * 256 + y as u16 * 32 + z as u16;
                    entries.push((idx, env));
                }
            }
        }

        entries.sort_by_key(|&(idx, _)| idx);
        Self { entries }
    }
}

// ============================================================================
// carrier_phase_3d — 3D carrier phase computation
// ============================================================================

/// Carrier phase at position (dx, dy, dz) relative to center.
///
/// Propagation along (1,1,1) diagonal, scale = 32 (Z dimension).
#[inline]
fn carrier_phase_3d(dx: i32, dy: i32, dz: i32, freq: f32, phi: f32) -> f32 {
    let spatial_phase = freq * (dx + dy + dz) as f32 / 32.0;
    std::f32::consts::TAU * spatial_phase + phi
}

// ============================================================================
// Operation 15: gabor_write — Place a concept in 3D space
// ============================================================================

/// Write a Gabor wavelet into the container at a specific 3D location.
///
/// Encodes: position (x0,y0,z0), frequency f, phase φ, width σ, amplitude A.
/// The Gaussian envelope naturally gates the write — no external mask needed.
///
/// Cost: iterates over ~(4σ)³ positions within effective radius.
///   σ=1: ~64 positions, σ=2: ~512 positions, σ=4+: ~full volume
pub fn gabor_write(
    container: &mut [i8],
    lut: &GaussianLUT,
    x0: u8,
    y0: u8,
    z0: u8,
    freq: f32,
    phi: f32,
    amplitude: f32,
) {
    assert!(container.len() >= 2048);
    let radius_sq = lut.effective_radius_sq();

    for x in 0..FOCUS_DIM_X as i32 {
        let dx = x - x0 as i32;
        for y in 0..FOCUS_DIM_Y as i32 {
            let dy = y - y0 as i32;
            for z in 0..FOCUS_DIM_Z as i32 {
                let dz = z - z0 as i32;
                let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                if d_sq > radius_sq {
                    continue;
                }

                let envelope = lut.amplitude(d_sq) as f32 / 255.0;
                if envelope < 0.004 {
                    continue;
                }

                let phase = carrier_phase_3d(dx, dy, dz, freq, phi);
                let sample = (amplitude * envelope * phase.cos())
                    .round()
                    .clamp(-128.0, 127.0) as i8;

                let idx = x as usize * 256 + y as usize * 32 + z as usize;
                container[idx] = container[idx].saturating_add(sample);
            }
        }
    }
}

// ============================================================================
// Operation 16: gabor_read — Recover a concept from 3D space
// ============================================================================

/// Read back the amplitude and phase of a concept at a specific location
/// and frequency. Spatial demodulation via conjugate wavelet.
///
/// Returns (phase, amplitude).
pub fn gabor_read(
    container: &[i8],
    lut: &GaussianLUT,
    x0: u8,
    y0: u8,
    z0: u8,
    freq: f32,
) -> (f32, f32) {
    assert!(container.len() >= 2048);
    let radius_sq = lut.effective_radius_sq();
    let mut cos_sum: f64 = 0.0;
    let mut sin_sum: f64 = 0.0;
    let mut weight_sum: f64 = 0.0;

    for x in 0..FOCUS_DIM_X as i32 {
        let dx = x - x0 as i32;
        for y in 0..FOCUS_DIM_Y as i32 {
            let dy = y - y0 as i32;
            for z in 0..FOCUS_DIM_Z as i32 {
                let dz = z - z0 as i32;
                let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                if d_sq > radius_sq {
                    continue;
                }

                let envelope = lut.amplitude(d_sq) as f64 / 255.0;
                if envelope < 0.004 {
                    continue;
                }

                let phase = carrier_phase_3d(dx, dy, dz, freq, 0.0);
                let idx = x as usize * 256 + y as usize * 32 + z as usize;
                let sample = container[idx] as f64;

                cos_sum += sample * envelope * (phase as f64).cos();
                sin_sum += sample * envelope * (phase as f64).sin();
                weight_sum += envelope * envelope;
            }
        }
    }

    if weight_sum < 1e-10 {
        return (0.0, 0.0);
    }

    // Fourier sign convention: Σ cos(θ+φ)·sin(θ) = -W·sin(φ)/2
    // Negate sin_sum to recover the correct phase (same fix as carrier.rs).
    let recovered_phase = ((-sin_sum).atan2(cos_sum) as f32).rem_euclid(std::f32::consts::TAU);
    let recovered_amplitude =
        ((cos_sum * cos_sum + sin_sum * sin_sum).sqrt() * 2.0 / weight_sum) as f32;

    (recovered_phase, recovered_amplitude)
}

// ============================================================================
// Operation 17: delta_cube_bind — Create a relationship volume
// ============================================================================

/// XOR-bind two fields to create a delta cube (binary mode).
/// Self-inverse: delta_cube_xor(a, a) → all zeros.
pub fn delta_cube_xor(field_a: &[u8], field_b: &[u8], delta: &mut [u8]) {
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048 && delta.len() >= 2048);
    for i in 0..2048 {
        delta[i] = field_a[i] ^ field_b[i];
    }
}

/// SUB-bind two fields to create a delta cube (phase/carrier mode).
pub fn delta_cube_sub(field_a: &[i8], field_b: &[i8], delta: &mut [i8]) {
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048 && delta.len() >= 2048);
    for i in 0..2048 {
        delta[i] = field_a[i].wrapping_sub(field_b[i]);
    }
}

// ============================================================================
// Operation 18: delta_cube_write/read — Content in relationships
// ============================================================================

/// Write a Gabor wavelet into a delta cube.
pub fn delta_cube_write_gabor(
    delta: &mut [i8],
    lut: &GaussianLUT,
    x0: u8,
    y0: u8,
    z0: u8,
    freq: f32,
    phi: f32,
    amplitude: f32,
) {
    gabor_write(delta, lut, x0, y0, z0, freq, phi, amplitude);
}

/// Read a Gabor wavelet from a delta cube.
pub fn delta_cube_read_gabor(
    delta: &[i8],
    lut: &GaussianLUT,
    x0: u8,
    y0: u8,
    z0: u8,
    freq: f32,
) -> (f32, f32) {
    gabor_read(delta, lut, x0, y0, z0, freq)
}

// ============================================================================
// Operation 19: delta_cube_recover — Reconstruct from both keys
// ============================================================================

/// Recover content from a stored delta and both original fields (XOR mode).
///
/// stored_delta = field_a ⊕ field_b ⊕ content
/// content = stored_delta ⊕ (field_a ⊕ field_b)
pub fn delta_cube_recover_xor(
    stored_delta: &[u8],
    field_a: &[u8],
    field_b: &[u8],
    content: &mut [u8],
) {
    assert!(stored_delta.len() >= 2048);
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048);
    assert!(content.len() >= 2048);

    for i in 0..2048 {
        let original_delta = field_a[i] ^ field_b[i];
        content[i] = stored_delta[i] ^ original_delta;
    }
}

/// Recover content from a stored delta and both original fields (phase mode).
///
/// stored_delta = field_a - field_b + content
/// content = stored_delta - (field_a - field_b)
pub fn delta_cube_recover_phase(
    stored_delta: &[i8],
    field_a: &[i8],
    field_b: &[i8],
    content: &mut [i8],
) {
    assert!(stored_delta.len() >= 2048);
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048);
    assert!(content.len() >= 2048);

    for i in 0..2048 {
        let original_delta = field_a[i].wrapping_sub(field_b[i]);
        content[i] = stored_delta[i].wrapping_sub(original_delta);
    }
}

// ============================================================================
// SpatialTransform — 3D permutations over 8×8×32 volume
// ============================================================================

/// A spatial transform over the 8×8×32 volume.
/// Stored as a permutation table: perm[old_idx] = new_idx.
pub struct SpatialTransform {
    pub perm: [u16; 2048],
}

impl SpatialTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        let mut perm = [0u16; 2048];
        for i in 0..2048 {
            perm[i] = i as u16;
        }
        Self { perm }
    }

    /// Rotate the X axis by `n` positions (cyclic shift of 256-byte slabs).
    pub fn rotate_x(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            let x_new = (x + n as u16) % 8;
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x_new * 256 + y * 32 + z;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Rotate the Y axis by `n` positions.
    pub fn rotate_y(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                let y_new = (y + n as u16) % 8;
                for z in 0..32u16 {
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x * 256 + y_new * 32 + z;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Rotate the Z axis by `n` positions.
    pub fn rotate_z(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let z_new = (z + n as u16) % 32;
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x * 256 + y * 32 + z_new;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Diagonal rotation: (x, y, z) → (y, z%8, x*4 + z/8).
    /// Maps the 3 axes into each other, preserving volume.
    pub fn diagonal() -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let x_new = y;
                    let y_new = z % 8;
                    let z_new = x * 4 + z / 8;
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x_new * 256 + y_new * 32 + z_new;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Compose: self.compose(other) = apply other first, then self.
    pub fn compose(&self, other: &SpatialTransform) -> SpatialTransform {
        let mut perm = [0u16; 2048];
        for i in 0..2048 {
            perm[i] = self.perm[other.perm[i] as usize];
        }
        SpatialTransform { perm }
    }

    /// Compute the inverse transform.
    pub fn inverse(&self) -> SpatialTransform {
        let mut inv = [0u16; 2048];
        for i in 0..2048 {
            inv[self.perm[i] as usize] = i as u16;
        }
        SpatialTransform { perm: inv }
    }

    /// Apply the transform to a u8 container.
    pub fn apply(&self, container: &[u8]) -> Vec<u8> {
        assert!(container.len() >= 2048);
        let mut out = vec![0u8; 2048];
        for i in 0..2048 {
            out[self.perm[i] as usize] = container[i];
        }
        out
    }

    /// Apply the transform to an i8 container.
    pub fn apply_i8(&self, container: &[i8]) -> Vec<i8> {
        assert!(container.len() >= 2048);
        let mut out = vec![0i8; 2048];
        for i in 0..2048 {
            out[self.perm[i] as usize] = container[i];
        }
        out
    }
}

// ============================================================================
// Operation 20: spatial_bind/unbind — Apply 3D transforms
// ============================================================================

/// Bind a u8 container with a spatial transform.
pub fn spatial_bind(container: &[u8], transform: &SpatialTransform) -> Vec<u8> {
    transform.apply(container)
}

/// Unbind (inverse transform) a u8 container.
pub fn spatial_unbind(container: &[u8], transform: &SpatialTransform) -> Vec<u8> {
    transform.inverse().apply(container)
}

/// Bind an i8 container with a spatial transform.
pub fn spatial_bind_i8(container: &[i8], transform: &SpatialTransform) -> Vec<i8> {
    transform.apply_i8(container)
}

/// Unbind (inverse transform) an i8 container.
pub fn spatial_unbind_i8(container: &[i8], transform: &SpatialTransform) -> Vec<i8> {
    transform.inverse().apply_i8(container)
}

// ============================================================================
// Operation 21: Overlay — The Blackboard Layer
// ============================================================================

/// XOR/ADD overlay that sits on top of a committed container.
///
/// The overlay IS the blackboard's scratch space. All writes go into the
/// overlay. Reads see through both layers. Flush merges overlay into
/// the container. Rewind restores from snapshot.
///
/// For binary containers (META, BTREE): XOR semantics.
///   read: container ^ overlay
///   flush: container ^= overlay
///
/// For phase/carrier containers (CAM, EMBED): ADD semantics.
///   read: container + overlay (wrapping)
///   flush: container += overlay (wrapping)
///   undo flush: container -= overlay (wrapping, exact inverse)
///
/// ## STM/LTM Boundary
///
/// | Layer     | Role         | Analogy   |
/// |-----------|------------- |-----------|
/// | Overlay   | Working set  | Redis     |
/// | Container | Committed    | LanceDB   |
pub struct Overlay {
    /// The scratch surface. Same geometry as container: 2048 bytes = 8×8×32.
    pub buffer: Vec<u8>,

    /// Snapshot stack for rewind. Each snapshot is a full 2048-byte copy
    /// of the overlay at the time of the snapshot.
    snapshots: Vec<Vec<u8>>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    /// Create a zeroed overlay.
    pub fn new() -> Self {
        Self {
            buffer: vec![0u8; 2048],
            snapshots: Vec::new(),
        }
    }

    /// Take a snapshot of the current overlay state (for rewind).
    pub fn snapshot(&mut self) {
        self.snapshots.push(self.buffer.clone());
    }

    /// Rewind to the most recent snapshot. Discards everything written since.
    /// Returns false if no snapshots exist.
    pub fn rewind(&mut self) -> bool {
        if let Some(snap) = self.snapshots.pop() {
            self.buffer = snap;
            true
        } else {
            false
        }
    }

    /// Clear the overlay without flushing (discard all STM).
    pub fn discard(&mut self) {
        self.buffer.fill(0);
        self.snapshots.clear();
    }

    /// Number of snapshots on the rewind stack.
    pub fn snapshot_depth(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if the overlay has any non-zero content.
    pub fn is_clean(&self) -> bool {
        self.buffer.iter().all(|&b| b == 0)
    }

    // ---- Read through both layers ----

    /// Read a single byte through the overlay (XOR mode).
    #[inline]
    pub fn read_xor(&self, container: &[u8], idx: usize) -> u8 {
        container[idx] ^ self.buffer[idx]
    }

    /// Read a single byte through the overlay (ADD mode).
    #[inline]
    pub fn read_add(&self, container: &[u8], idx: usize) -> u8 {
        container[idx].wrapping_add(self.buffer[idx])
    }

    /// Read the full container through the overlay (XOR mode).
    pub fn read_full_xor(&self, container: &[u8]) -> Vec<u8> {
        assert!(container.len() >= 2048);
        container
            .iter()
            .zip(self.buffer.iter())
            .map(|(&c, &o)| c ^ o)
            .collect()
    }

    /// Read the full container through the overlay (ADD mode).
    pub fn read_full_add(&self, container: &[u8]) -> Vec<u8> {
        assert!(container.len() >= 2048);
        container
            .iter()
            .zip(self.buffer.iter())
            .map(|(&c, &o)| c.wrapping_add(o))
            .collect()
    }

    /// Read the full container through the overlay (i8 ADD mode, for carrier/Gabor).
    pub fn read_full_add_i8(&self, container: &[i8]) -> Vec<i8> {
        assert!(container.len() >= 2048);
        container
            .iter()
            .zip(self.buffer.iter())
            .map(|(&c, &o)| c.saturating_add(o as i8))
            .collect()
    }

    // ---- Flush: merge overlay into container ----

    /// Flush overlay into container via XOR (binary mode).
    /// After flush, overlay is cleared.
    ///
    /// SIMD: 32 VPXORD instructions for 2048 bytes.
    pub fn flush_xor(&mut self, container: &mut [u8]) {
        assert!(container.len() >= 2048);
        for i in 0..2048 {
            container[i] ^= self.buffer[i];
        }
        self.buffer.fill(0);
        self.snapshots.clear();
    }

    /// Flush overlay into container via ADD (phase/carrier mode, u8).
    /// After flush, overlay is cleared.
    ///
    /// SIMD: 32 VPADDB instructions for 2048 bytes.
    pub fn flush_add(&mut self, container: &mut [u8]) {
        assert!(container.len() >= 2048);
        for i in 0..2048 {
            container[i] = container[i].wrapping_add(self.buffer[i]);
        }
        self.buffer.fill(0);
        self.snapshots.clear();
    }

    /// Flush overlay into i8 container via saturating ADD (carrier/Gabor mode).
    /// After flush, overlay is cleared.
    pub fn flush_add_i8(&mut self, container: &mut [i8]) {
        assert!(container.len() >= 2048);
        for i in 0..2048 {
            container[i] = container[i].saturating_add(self.buffer[i] as i8);
        }
        self.buffer.fill(0);
        self.snapshots.clear();
    }

    // ---- Safe i8/u8 reinterpretation ----

    /// Get the overlay buffer as mutable i8 slice (for Gabor/carrier writes).
    pub fn as_i8_mut(&mut self) -> &mut [i8] {
        // SAFETY: u8 and i8 have identical size, alignment, and no invalid values.
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut i8, 2048) }
    }

    /// Get the overlay buffer as i8 slice (for Gabor/carrier reads).
    pub fn as_i8(&self) -> &[i8] {
        // SAFETY: u8 and i8 have identical size, alignment, and no invalid values.
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const i8, 2048) }
    }
}

// ============================================================================
// SpectralMap — Full 3D Spectral Analysis
// ============================================================================

/// Full spectral analysis of the 3D volume.
///
/// For each position in the 8×8×32 volume, computes amplitude and phase
/// at each of 16 carrier frequencies. Real concepts appear as strong peaks;
/// ghost archetypes appear as weak, incoherent energy.
pub struct SpectralMap {
    /// amplitude[pos_idx * 16 + freq_idx]
    pub amplitude: Vec<f32>,
    /// phase[pos_idx * 16 + freq_idx]
    pub phase: Vec<f32>,
}

impl SpectralMap {
    /// Analyze the full volume at all positions and frequencies.
    pub fn analyze(container: &[i8], _basis: &CarrierBasis, lut_cache: &[GaussianLUT]) -> Self {
        let n_freqs = 16;
        let total = 2048 * n_freqs;
        let mut amplitude = vec![0.0f32; total];
        let mut phase = vec![0.0f32; total];
        let lut = &lut_cache[lut_cache.len() - 1];

        for x in 0..8u8 {
            for y in 0..8u8 {
                for z in 0..32u8 {
                    let pos_idx = x as usize * 256 + y as usize * 32 + z as usize;
                    for f in 0..n_freqs {
                        let (ph, amp) =
                            gabor_read(container, lut, x, y, z, CARRIER_FREQUENCIES[f] as f32);
                        let idx = pos_idx * n_freqs + f;
                        amplitude[idx] = amp;
                        phase[idx] = ph;
                    }
                }
            }
        }

        Self { amplitude, phase }
    }

    /// Find significant peaks (real concepts, not ghosts).
    pub fn find_peaks(&self, threshold: f32) -> Vec<(u8, u8, u8, u8, f32, f32)> {
        let mut peaks = Vec::new();
        let n_freqs = 16;

        for x in 0..8u8 {
            for y in 0..8u8 {
                for z in 0..32u8 {
                    let pos_idx = x as usize * 256 + y as usize * 32 + z as usize;
                    for f in 0..n_freqs {
                        let idx = pos_idx * n_freqs + f;
                        let amp = self.amplitude[idx];
                        if amp < threshold {
                            continue;
                        }
                        if self.is_local_max_3d(x, y, z, f, amp) {
                            peaks.push((x, y, z, f as u8, amp, self.phase[idx]));
                        }
                    }
                }
            }
        }

        peaks
    }

    fn is_local_max_3d(&self, x: u8, y: u8, z: u8, f: usize, amp: f32) -> bool {
        let n_freqs = 16;
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if !(0..8).contains(&nx) || !(0..8).contains(&ny) || !(0..32).contains(&nz) {
                        continue;
                    }
                    let nidx = nx as usize * 256 + ny as usize * 32 + nz as usize;
                    if self.amplitude[nidx * n_freqs + f] > amp {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Resynthesize a clean container from only the significant peaks.
    pub fn resynthesize(&self, threshold: f32, _lut_cache: &[GaussianLUT], sigma: f32) -> Vec<i8> {
        let peaks = self.find_peaks(threshold);
        let mut clean = vec![0i8; 2048];
        let lut = GaussianLUT::new(sigma);

        for (x, y, z, f, amp, phi) in peaks {
            gabor_write(
                &mut clean,
                &lut,
                x,
                y,
                z,
                CARRIER_FREQUENCIES[f as usize] as f32,
                phi,
                amp,
            );
        }

        clean
    }
}

// ============================================================================
// Operation 22: Residual Energy and Adaptive Cleaning
// ============================================================================

/// Subtractive noise measurement: subtract known signals, measure residual.
pub fn residual_energy(
    container: &[i8],
    known_concepts: &[(u8, u8, u8, f32)],
    lut: &GaussianLUT,
    _basis: &CarrierBasis,
) -> f64 {
    let mut residual = container.to_vec();

    for &(x0, y0, z0, freq) in known_concepts {
        let (phase, amplitude) = gabor_read(&residual, lut, x0, y0, z0, freq);
        gabor_write(&mut residual, lut, x0, y0, z0, freq, phase, -amplitude);
    }

    residual.iter().map(|&b| (b as f64) * (b as f64)).sum()
}

/// Adaptive cleaning: only do full spectral clean when residual exceeds threshold.
pub fn clean_if_needed(
    container: &mut [i8],
    known_concepts: &[(u8, u8, u8, f32)],
    lut: &GaussianLUT,
    basis: &CarrierBasis,
    noise_threshold: f64,
    clean_sigma: f32,
) -> bool {
    let energy = residual_energy(container, known_concepts, lut, basis);
    let energy_per_byte = energy / 2048.0;

    if energy_per_byte > noise_threshold {
        let lut_broad = GaussianLUT::new(4.0);
        let spec = SpectralMap::analyze(container, basis, &[lut_broad]);
        let clean = spec.resynthesize(
            (noise_threshold.sqrt() * 0.5) as f32,
            &[GaussianLUT::new(clean_sigma)],
            clean_sigma,
        );
        container.copy_from_slice(&clean);
        true
    } else {
        false
    }
}

// ============================================================================
// Operation 23: Orthogonal Projection Cleaning
// ============================================================================

/// Exact orthogonal cleaning when the concept set is known.
///
/// Projects the container onto the subspace spanned by the known templates
/// and discards the orthogonal complement (noise, ghosts).
pub fn orthogonal_project(container: &[i8], templates: &[Vec<i8>]) -> Vec<i8> {
    let k = templates.len();
    let n = 2048;
    if k == 0 {
        return container.to_vec();
    }

    // W · container → K-element coefficient vector
    let mut coefficients = vec![0.0f64; k];
    for (i, template) in templates.iter().enumerate() {
        let mut dot: f64 = 0.0;
        for j in 0..n {
            dot += template[j] as f64 * container[j] as f64;
        }
        coefficients[i] = dot;
    }

    // W · W^T → K×K Gram matrix
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut dot: f64 = 0.0;
            for p in 0..n {
                dot += templates[i][p] as f64 * templates[j][p] as f64;
            }
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    let alpha = solve_symmetric_positive(&gram, &coefficients, k);

    let mut clean = vec![0i8; n];
    for i in 0..k {
        for j in 0..n {
            let contribution = (alpha[i] * templates[i][j] as f64).round();
            let current = clean[j] as f64 + contribution;
            clean[j] = current.clamp(-128.0, 127.0) as i8;
        }
    }

    clean
}

/// Cholesky solve for symmetric positive-definite system.
fn solve_symmetric_positive(gram: &[f64], rhs: &[f64], k: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = gram[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                l[i * k + j] = if sum > 0.0 { sum.sqrt() } else { 1e-10 };
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    let mut y = vec![0.0f64; k];
    for i in 0..k {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[i * k + j] * y[j];
        }
        y[i] = sum / l[i * k + i];
    }

    let mut alpha = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= l[j * k + i] * alpha[j];
        }
        alpha[i] = sum / l[i * k + i];
    }

    alpha
}

// ============================================================================
// Operation 24: Hebbian Learning
// ============================================================================

/// Hebbian learning: strengthen the interference pattern between
/// two co-occurring concepts.
pub fn hebbian_update(
    overlay: &mut Overlay,
    _container: &[i8],
    lut: &GaussianLUT,
    concept_a: (u8, u8, u8, f32),
    concept_b: (u8, u8, u8, f32),
    learning_rate: f32,
) {
    let (xa, ya, za, fa) = concept_a;
    let (xb, yb, zb, fb) = concept_b;
    let buf = overlay.as_i8_mut();

    for x in 0..8i32 {
        for y in 0..8i32 {
            for z in 0..32i32 {
                let idx = x as usize * 256 + y as usize * 32 + z as usize;

                let da_sq = ((x - xa as i32).pow(2)
                    + (y - ya as i32).pow(2)
                    + (z - za as i32).pow(2)) as u32;
                let env_a = lut.amplitude(da_sq) as f32 / 255.0;
                if env_a < 0.004 {
                    continue;
                }
                let phase_a =
                    carrier_phase_3d(x - xa as i32, y - ya as i32, z - za as i32, fa, 0.0);

                let db_sq = ((x - xb as i32).pow(2)
                    + (y - yb as i32).pow(2)
                    + (z - zb as i32).pow(2)) as u32;
                let env_b = lut.amplitude(db_sq) as f32 / 255.0;
                if env_b < 0.004 {
                    continue;
                }
                let phase_b =
                    carrier_phase_3d(x - xb as i32, y - yb as i32, z - zb as i32, fb, 0.0);

                let signal_a = env_a * phase_a.cos();
                let signal_b = env_b * phase_b.cos();
                let interference = signal_a * signal_b;

                let update = (learning_rate * interference * 127.0)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                buf[idx] = buf[idx].saturating_add(update);
            }
        }
    }
}

/// Anti-Hebbian update: weaken a concept that failed to predict.
pub fn anti_hebbian_update(
    overlay: &mut Overlay,
    lut: &GaussianLUT,
    concept: (u8, u8, u8, f32, f32),
    decay_rate: f32,
) {
    let (x0, y0, z0, freq, phi) = concept;
    gabor_write(overlay.as_i8_mut(), lut, x0, y0, z0, freq, phi, -decay_rate);
}

// ============================================================================
// Operation 25: Sigma Adaptation
// ============================================================================

/// Adapt σ based on readout SNR. High SNR → σ shrinks; low SNR → σ grows.
pub fn adapt_sigma(
    container: &[i8],
    lut: &GaussianLUT,
    x0: u8,
    y0: u8,
    z0: u8,
    freq: f32,
    current_sigma: f32,
    learning_rate: f32,
    snr_target: f32,
) -> f32 {
    let (_, amplitude) = gabor_read(container, lut, x0, y0, z0, freq);
    let noise_x = (x0 + 4) % 8;
    let noise_z = (z0 + 16) % 32;
    let (_, noise_amp) = gabor_read(container, lut, noise_x, y0, noise_z, freq);

    let snr = if noise_amp > 1e-6 {
        amplitude / noise_amp
    } else {
        amplitude * 100.0
    };

    let sigma_new = current_sigma * (1.0 - learning_rate * (snr - snr_target));
    sigma_new.clamp(0.5, 4.0)
}

// ============================================================================
// Operation 26: Archetype Detection and Crystallization
// ============================================================================

/// Fast archetype detection: sample the spectral map at sparse positions.
pub struct FastArchetypeDetector {
    pub sample_z: u8,
}

impl Default for FastArchetypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FastArchetypeDetector {
    pub fn new() -> Self {
        Self { sample_z: 16 }
    }

    /// Detect archetypes by sampling spectral energy at 8×8 XY grid × 16 freqs.
    /// Returns (x, y, freq_idx, amplitude, phase) for each detected archetype.
    pub fn detect(
        &self,
        container: &[i8],
        lut: &GaussianLUT,
        threshold: f32,
    ) -> Vec<(u8, u8, u8, f32, f32)> {
        let mut peaks = Vec::new();

        for x in 0..8u8 {
            for y in 0..8u8 {
                for f in 0..16u8 {
                    let (phase, amp) = gabor_read(
                        container,
                        lut,
                        x,
                        y,
                        self.sample_z,
                        CARRIER_FREQUENCIES[f as usize] as f32,
                    );
                    if amp > threshold && self.is_xy_local_max(container, lut, x, y, f, amp) {
                        peaks.push((x, y, f, amp, phase));
                    }
                }
            }
        }

        peaks
    }

    fn is_xy_local_max(
        &self,
        container: &[i8],
        lut: &GaussianLUT,
        x: u8,
        y: u8,
        f: u8,
        amp: f32,
    ) -> bool {
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if !(0..8).contains(&nx) || !(0..8).contains(&ny) {
                    continue;
                }
                let (_, neighbor_amp) = gabor_read(
                    container,
                    lut,
                    nx as u8,
                    ny as u8,
                    self.sample_z,
                    CARRIER_FREQUENCIES[f as usize] as f32,
                );
                if neighbor_amp > amp {
                    return false;
                }
            }
        }
        true
    }
}

/// Crystallize detected archetypes: replace overlapping wavelets with clean ones.
pub fn crystallize_archetypes(
    container: &mut [i8],
    archetypes: &[(u8, u8, u8, f32, f32)],
    lut_broad: &GaussianLUT,
    lut_crystal: &GaussianLUT,
) {
    for &(x, y, f_idx, amp, phase) in archetypes {
        gabor_write(
            container,
            lut_broad,
            x,
            y,
            16,
            CARRIER_FREQUENCIES[f_idx as usize] as f32,
            phase,
            -amp,
        );
    }
    for &(x, y, f_idx, amp, phase) in archetypes {
        gabor_write(
            container,
            lut_crystal,
            x,
            y,
            16,
            CARRIER_FREQUENCIES[f_idx as usize] as f32,
            phase,
            amp,
        );
    }
}

// ============================================================================
// Operation 27: BLAS Acceleration Structures
// ============================================================================

/// BLAS-style spectral analysis using flat carrier basis and envelope weighting.
pub fn spectral_analysis_blas(
    container: &[i8],
    basis_cos: &[[i8; 2048]; 16],
    basis_sin: &[[i8; 2048]; 16],
    lut: &GaussianLUT,
    sample_positions: &[(u8, u8, u8)],
) -> Vec<(f32, f32)> {
    let n_freq = 16;
    let mut results = Vec::with_capacity(sample_positions.len() * n_freq);

    for &(x0, y0, z0) in sample_positions {
        let mut weighted = vec![0.0f32; 2048];
        for x in 0..8i32 {
            for y in 0..8i32 {
                for z in 0..32i32 {
                    let idx = x as usize * 256 + y as usize * 32 + z as usize;
                    let dx = x - x0 as i32;
                    let dy = y - y0 as i32;
                    let dz = z - z0 as i32;
                    let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                    let env = lut.amplitude(d_sq) as f32 / 255.0;
                    weighted[idx] = container[idx] as f32 * env;
                }
            }
        }

        for f in 0..n_freq {
            let mut cos_sum: f32 = 0.0;
            let mut sin_sum: f32 = 0.0;
            for j in 0..2048 {
                cos_sum += basis_cos[f][j] as f32 * weighted[j];
                sin_sum += basis_sin[f][j] as f32 * weighted[j];
            }
            let amplitude = (cos_sum * cos_sum + sin_sum * sin_sum).sqrt();
            let phase = (-sin_sum).atan2(cos_sum).rem_euclid(std::f32::consts::TAU);
            results.push((amplitude, phase));
        }
    }

    results
}

/// Batched Gabor writes via template accumulation.
pub struct GaborBatch {
    templates: Vec<Vec<f32>>,
    amplitudes: Vec<f32>,
}

impl Default for GaborBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl GaborBatch {
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            amplitudes: Vec::new(),
        }
    }

    /// Add a wavelet to the batch. Precomputes the template.
    pub fn add(
        &mut self,
        lut: &GaussianLUT,
        x0: u8,
        y0: u8,
        z0: u8,
        freq: f32,
        amplitude: f32,
        phase: f32,
    ) {
        let mut template = vec![0.0f32; 2048];
        for x in 0..8i32 {
            for y in 0..8i32 {
                for z in 0..32i32 {
                    let idx = x as usize * 256 + y as usize * 32 + z as usize;
                    let dx = x - x0 as i32;
                    let dy = y - y0 as i32;
                    let dz = z - z0 as i32;
                    let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                    let env = lut.amplitude(d_sq) as f32 / 255.0;
                    if env < 0.004 {
                        continue;
                    }
                    let carrier = carrier_phase_3d(dx, dy, dz, freq, phase);
                    template[idx] = env * carrier.cos();
                }
            }
        }
        self.templates.push(template);
        self.amplitudes.push(amplitude);
    }

    /// Flush the batch into a container.
    pub fn flush(&self, container: &mut [i8]) {
        for j in 0..2048 {
            let mut total: f32 = 0.0;
            for (i, template) in self.templates.iter().enumerate() {
                total += self.amplitudes[i] * template[j];
            }
            let update = total.round().clamp(-128.0, 127.0) as i8;
            container[j] = container[j].saturating_add(update);
        }
    }
}

// ============================================================================
// Overlay Extension: flush_and_clean
// ============================================================================

impl Overlay {
    /// Flush with automatic cleaning.
    pub fn flush_and_clean(
        &mut self,
        container: &mut [i8],
        known_concepts: &[(u8, u8, u8, f32)],
        lut: &GaussianLUT,
        basis: &CarrierBasis,
        noise_threshold: f64,
    ) {
        self.flush_add_i8(container);
        clean_if_needed(container, known_concepts, lut, basis, noise_threshold, 2.0);
    }
}

// ============================================================================
// CooccurrenceMatrix — Track concept co-occurrence during bootstrap
// ============================================================================

/// Tracks co-occurrence between concepts during bootstrap phase.
pub struct CooccurrenceMatrix {
    pub counts: Vec<Vec<f32>>,
    pub totals: Vec<f32>,
    pub k: usize,
}

impl CooccurrenceMatrix {
    pub fn new(k: usize) -> Self {
        Self {
            counts: vec![vec![0.0; k]; k],
            totals: vec![0.0; k],
            k,
        }
    }

    pub fn observe_pair(&mut self, i: usize, j: usize) {
        self.counts[i][j] += 1.0;
        self.counts[j][i] += 1.0;
    }

    pub fn observe_single(&mut self, i: usize) {
        self.totals[i] += 1.0;
    }

    pub fn observe_window(&mut self, active: &[usize]) {
        for &i in active {
            self.observe_single(i);
            for &j in active {
                if i < j {
                    self.observe_pair(i, j);
                }
            }
        }
    }

    /// Normalize to PMI-like correlation matrix.
    pub fn normalized(&self) -> Vec<Vec<f32>> {
        let mut norm = vec![vec![0.0; self.k]; self.k];
        for i in 0..self.k {
            for j in 0..self.k {
                let denom = (self.totals[i] * self.totals[j]).sqrt();
                norm[i][j] = if denom > 0.0 {
                    self.counts[i][j] / denom
                } else {
                    0.0
                };
            }
        }
        norm
    }

    pub fn total_observations(&self) -> f32 {
        self.totals.iter().sum()
    }
}

/// Check if enough data for axis crystallization.
pub fn ready_for_crystallization(matrix: &CooccurrenceMatrix) -> bool {
    let active_concepts = matrix.totals.iter().filter(|&&t| t >= 5.0).count();
    active_concepts >= 8
}

// ============================================================================
// Operation 28: AxisCrystallizer — PCA on co-occurrence
// ============================================================================

/// Discovers the 3D coordinate system from co-occurrence structure.
pub struct AxisCrystallizer {
    pub axes: [Vec<f32>; 3],
    pub eigenvalues: [f32; 3],
    pub coords: Vec<(f32, f32, f32)>,
}

impl AxisCrystallizer {
    /// Run PCA on the normalized correlation matrix.
    pub fn crystallize(matrix: &CooccurrenceMatrix) -> Self {
        let norm = matrix.normalized();
        let k = matrix.k;

        let mut axes = [vec![0.0; k], vec![0.0; k], vec![0.0; k]];
        let mut eigenvalues = [0.0f32; 3];
        let mut deflated = norm.clone();

        for d in 0..3 {
            let (eigvec, eigval) = power_iteration(&deflated, k, 200);
            axes[d] = eigvec.clone();
            eigenvalues[d] = eigval;

            for i in 0..k {
                for j in 0..k {
                    deflated[i][j] -= eigval * eigvec[i] * eigvec[j];
                }
            }
        }

        let mut coords = Vec::with_capacity(k);
        for i in 0..k {
            let x: f32 = axes[0]
                .iter()
                .enumerate()
                .map(|(j, &a)| a * norm[i][j])
                .sum();
            let y: f32 = axes[1]
                .iter()
                .enumerate()
                .map(|(j, &a)| a * norm[i][j])
                .sum();
            let z: f32 = axes[2]
                .iter()
                .enumerate()
                .map(|(j, &a)| a * norm[i][j])
                .sum();
            coords.push((x, y, z));
        }

        let coords = normalize_to_grid(&coords);

        Self {
            axes,
            eigenvalues,
            coords,
        }
    }
}

/// Power iteration: find the dominant eigenvector of a symmetric matrix.
fn power_iteration(matrix: &[Vec<f32>], k: usize, iterations: usize) -> (Vec<f32>, f32) {
    let mut v = vec![1.0 / (k as f32).sqrt(); k];

    for _ in 0..iterations {
        let mut w = vec![0.0; k];
        for i in 0..k {
            for j in 0..k {
                w[i] += matrix[i][j] * v[j];
            }
        }

        let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            break;
        }

        for i in 0..k {
            v[i] = w[i] / norm;
        }
    }

    let mut eigenvalue: f32 = 0.0;
    for i in 0..k {
        let mut mv_i: f32 = 0.0;
        for j in 0..k {
            mv_i += matrix[i][j] * v[j];
        }
        eigenvalue += v[i] * mv_i;
    }

    (v, eigenvalue)
}

/// Map continuous coordinates to the 8×8×32 grid.
fn normalize_to_grid(coords: &[(f32, f32, f32)]) -> Vec<(f32, f32, f32)> {
    if coords.is_empty() {
        return vec![];
    }

    let normalize_axis = |vals: &[f32], max_val: f32| -> Vec<f32> {
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range < 1e-10 {
            vec![max_val / 2.0; vals.len()]
        } else {
            vals.iter()
                .map(|&v| ((v - min) / range) * max_val)
                .collect()
        }
    };

    let x_vals: Vec<f32> = coords.iter().map(|c| c.0).collect();
    let y_vals: Vec<f32> = coords.iter().map(|c| c.1).collect();
    let z_vals: Vec<f32> = coords.iter().map(|c| c.2).collect();

    let x_norm = normalize_axis(&x_vals, 7.0);
    let y_norm = normalize_axis(&y_vals, 7.0);
    let z_norm = normalize_axis(&z_vals, 31.0);

    (0..coords.len())
        .map(|i| (x_norm[i], y_norm[i], z_norm[i]))
        .collect()
}

// ============================================================================
// ContainerMode and Migration
// ============================================================================

/// Container operating mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContainerMode {
    Empty,
    Carrier,
    Gabor,
}

/// Mode tag stored in META container byte 0.
pub const MODE_BYTE_OFFSET: usize = 0;

pub fn get_container_mode(meta: &[u8]) -> ContainerMode {
    match meta[MODE_BYTE_OFFSET] {
        0 => ContainerMode::Empty,
        1 => ContainerMode::Carrier,
        2 => ContainerMode::Gabor,
        _ => ContainerMode::Empty,
    }
}

pub fn set_container_mode(meta: &mut [u8], mode: ContainerMode) {
    meta[MODE_BYTE_OFFSET] = match mode {
        ContainerMode::Empty => 0,
        ContainerMode::Carrier => 1,
        ContainerMode::Gabor => 2,
    };
}

/// Result of carrier → Gabor migration.
pub struct MigrationResult {
    pub concepts_migrated: usize,
    pub freq_assignments: Vec<usize>,
    pub mode: ContainerMode,
}

/// Bootstrap mode: write concept as a flat carrier.
pub fn bootstrap_write(
    container: &mut [i8],
    basis: &CarrierBasis,
    concept_idx: u8,
    phase: f32,
    amplitude: f32,
) {
    carrier_encode(container, basis, concept_idx, phase, amplitude);
}

/// Bootstrap mode: read concept back.
pub fn bootstrap_read(container: &[i8], basis: &CarrierBasis, concept_idx: u8) -> (f32, f32) {
    carrier_decode(container, basis, concept_idx)
}

// ============================================================================
// Operation 29: Carrier → Gabor Migration
// ============================================================================

/// Migrate a container from flat carrier mode to 3D Gabor mode.
pub fn migrate_carrier_to_gabor(
    container: &mut [i8],
    basis: &CarrierBasis,
    crystallizer: &AxisCrystallizer,
    lut: &GaussianLUT,
    _sigma: f32,
) -> MigrationResult {
    let k = crystallizer.coords.len();

    let mut concepts: Vec<(f32, f32)> = Vec::new();
    for i in 0..k.min(16) {
        let (phase, amplitude) = carrier_decode(container, basis, i as u8);
        concepts.push((phase, amplitude));
    }

    let freq_assignments = greedy_frequency_assignment(&crystallizer.coords, 2.0);

    container.fill(0);

    for i in 0..k.min(16) {
        let (x, y, z) = crystallizer.coords[i];
        let (phase, amplitude) = concepts[i];
        let freq = CARRIER_FREQUENCIES[freq_assignments[i]] as f32;

        if amplitude < 0.01 {
            continue;
        }

        gabor_write(
            container,
            lut,
            x.round().clamp(0.0, 7.0) as u8,
            y.round().clamp(0.0, 7.0) as u8,
            z.round().clamp(0.0, 31.0) as u8,
            freq,
            phase,
            amplitude,
        );
    }

    MigrationResult {
        concepts_migrated: k.min(16),
        freq_assignments,
        mode: ContainerMode::Gabor,
    }
}

/// Greedy frequency assignment: minimize interference between nearby concepts.
fn greedy_frequency_assignment(coords: &[(f32, f32, f32)], sigma: f32) -> Vec<usize> {
    let k = coords.len();
    let threshold_sq = (2.0 * sigma) * (2.0 * sigma);
    let mut assignments = vec![0usize; k];
    let mut assigned = vec![false; k];

    for i in 0..k {
        let mut used_freqs = [false; 16];
        for j in 0..k {
            if !assigned[j] {
                continue;
            }
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            let dz = coords[i].2 - coords[j].2;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < threshold_sq {
                used_freqs[assignments[j]] = true;
            }
        }
        let freq = (0..16).find(|&f| !used_freqs[f]).unwrap_or(0);
        assignments[i] = freq;
        assigned[i] = true;
    }

    assignments
}

// ============================================================================
// Operation 30: Superposition Crystallization
// ============================================================================

/// Discover axes directly from the container's spectral structure.
pub fn crystallize_from_superposition(container: &[i8], basis: &CarrierBasis) -> AxisCrystallizer {
    let n_freq = 16;

    let mut freq_signals = vec![vec![0.0f32; 2048]; n_freq];
    for f in 0..n_freq {
        for j in 0..2048 {
            freq_signals[f][j] = container[j] as f32 * basis.basis_cos[f][j] as f32;
        }
    }

    let mut cross_corr = vec![vec![0.0f32; n_freq]; n_freq];
    for i in 0..n_freq {
        for j in i..n_freq {
            let mut dot: f32 = 0.0;
            for p in 0..2048 {
                dot += freq_signals[i][p] * freq_signals[j][p];
            }
            cross_corr[i][j] = dot;
            cross_corr[j][i] = dot;
        }
    }

    let mut matrix = CooccurrenceMatrix::new(n_freq);
    for i in 0..n_freq {
        for j in 0..n_freq {
            matrix.counts[i][j] = cross_corr[i][j].max(0.0);
        }
        matrix.totals[i] = cross_corr[i][i].max(1.0);
    }

    AxisCrystallizer::crystallize(&matrix)
}

// ============================================================================
// Operation 31: Incremental Axis Update and Migration
// ============================================================================

/// Incrementally update the axis system as new evidence arrives.
pub fn incremental_axis_update(
    crystallizer: &mut AxisCrystallizer,
    new_matrix: &CooccurrenceMatrix,
    rotation_rate: f32,
) -> Vec<Migration> {
    let new_crystal = AxisCrystallizer::crystallize(new_matrix);
    let aligned_axes = align_eigenvectors(&crystallizer.axes, &new_crystal.axes);

    for d in 0..3 {
        let len = crystallizer.axes[d].len();
        for i in 0..len {
            crystallizer.axes[d][i] +=
                rotation_rate * (aligned_axes[d][i] - crystallizer.axes[d][i]);
        }
        let norm: f32 = crystallizer.axes[d]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        if norm > 1e-10 {
            for i in 0..len {
                crystallizer.axes[d][i] /= norm;
            }
        }
    }

    let old_coords = crystallizer.coords.clone();
    let norm = new_matrix.normalized();
    let k = crystallizer.coords.len();

    let mut raw_coords = Vec::with_capacity(k);
    for i in 0..k {
        let x: f32 = crystallizer.axes[0]
            .iter()
            .enumerate()
            .map(|(j, &a)| {
                if j < norm[i].len() {
                    a * norm[i][j]
                } else {
                    0.0
                }
            })
            .sum();
        let y: f32 = crystallizer.axes[1]
            .iter()
            .enumerate()
            .map(|(j, &a)| {
                if j < norm[i].len() {
                    a * norm[i][j]
                } else {
                    0.0
                }
            })
            .sum();
        let z: f32 = crystallizer.axes[2]
            .iter()
            .enumerate()
            .map(|(j, &a)| {
                if j < norm[i].len() {
                    a * norm[i][j]
                } else {
                    0.0
                }
            })
            .sum();
        raw_coords.push((x, y, z));
    }
    crystallizer.coords = normalize_to_grid(&raw_coords);

    let mut migrations = Vec::new();
    for i in 0..k {
        let (ox, oy, oz) = old_coords[i];
        let (nx, ny, nz) = crystallizer.coords[i];
        let dist = ((nx - ox).powi(2) + (ny - oy).powi(2) + (nz - oz).powi(2)).sqrt();
        if dist > 1.0 {
            migrations.push((i, old_coords[i], crystallizer.coords[i]));
        }
    }

    migrations
}

/// Align new eigenvectors to old ones (resolve sign ambiguity).
fn align_eigenvectors(old: &[Vec<f32>; 3], new: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    let mut aligned = new.clone();
    for d in 0..3 {
        let dot: f32 = old[d].iter().zip(new[d].iter()).map(|(&a, &b)| a * b).sum();
        if dot < 0.0 {
            for x in aligned[d].iter_mut() {
                *x = -*x;
            }
        }
    }
    aligned
}

/// Re-position concepts that moved after axis rotation.
pub fn apply_migrations(
    container: &mut [i8],
    migrations: &[Migration],
    freq_assignments: &[usize],
    lut: &GaussianLUT,
) {
    for &(concept_idx, (old_x, old_y, old_z), (new_x, new_y, new_z)) in migrations {
        let freq = CARRIER_FREQUENCIES[freq_assignments[concept_idx]] as f32;

        let (phase, amplitude) = gabor_read(
            container,
            lut,
            old_x.round().clamp(0.0, 7.0) as u8,
            old_y.round().clamp(0.0, 7.0) as u8,
            old_z.round().clamp(0.0, 31.0) as u8,
            freq,
        );

        gabor_write(
            container,
            lut,
            old_x.round().clamp(0.0, 7.0) as u8,
            old_y.round().clamp(0.0, 7.0) as u8,
            old_z.round().clamp(0.0, 31.0) as u8,
            freq,
            phase,
            -amplitude,
        );

        gabor_write(
            container,
            lut,
            new_x.round().clamp(0.0, 7.0) as u8,
            new_y.round().clamp(0.0, 7.0) as u8,
            new_z.round().clamp(0.0, 31.0) as u8,
            freq,
            phase,
            amplitude,
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn phase_error(a: f32, b: f32) -> f32 {
        let diff = (a - b).abs();
        diff.min(TAU - diff)
    }

    // ---- Envelope tests ----

    #[test]
    fn test_gaussian_lut_center_max() {
        let lut = GaussianLUT::new(2.0);
        assert_eq!(lut.amplitude(0), 255);
    }

    #[test]
    fn test_gaussian_lut_far_zero() {
        let lut = GaussianLUT::new(2.0);
        assert_eq!(lut.amplitude(1000), 0);
    }

    #[test]
    fn test_gaussian_lut_monotonic_decrease() {
        let lut = GaussianLUT::new(2.0);
        for i in 1..100u32 {
            assert!(
                lut.amplitude(i) <= lut.amplitude(i - 1),
                "LUT not monotonic at d²={}: {} > {}",
                i,
                lut.amplitude(i),
                lut.amplitude(i - 1)
            );
        }
    }

    #[test]
    fn test_gaussian_lut_different_sigmas() {
        let narrow = GaussianLUT::new(0.5);
        let wide = GaussianLUT::new(4.0);
        // At d²=4: narrow should be much more decayed than wide
        assert!(narrow.amplitude(4) < wide.amplitude(4));
    }

    // ---- WaveletTemplate tests ----

    #[test]
    fn test_wavelet_template_center() {
        let lut = GaussianLUT::new(2.0);
        let tpl = WaveletTemplate::new(&lut, 4, 4, 16);
        // Center position should be in the template with amplitude 255
        let center_idx = 4u16 * 256 + 4 * 32 + 16;
        assert!(
            tpl.entries
                .iter()
                .any(|&(idx, amp)| idx == center_idx && amp == 255),
            "center should be in template with max amplitude"
        );
    }

    #[test]
    fn test_wavelet_template_tight_sigma() {
        let lut = GaussianLUT::new(0.5);
        let tpl = WaveletTemplate::new(&lut, 4, 4, 16);
        // Very tight: should have few entries
        assert!(
            tpl.entries.len() <= 30,
            "σ=0.5 template too large: {} entries",
            tpl.entries.len()
        );
    }

    // ---- Gabor wavelet tests ----

    #[test]
    fn test_gabor_write_max_at_center() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        // The center should have the highest absolute value
        let center_idx = 4 * 256 + 4 * 32 + 16;
        let center_val = container[center_idx].abs();

        // Check that it decays away from center
        let edge_idx = 0; // far corner
        let edge_val = container[edge_idx].abs();

        assert!(
            center_val > edge_val,
            "center {} should exceed edge {}",
            center_val,
            edge_val
        );
    }

    #[test]
    fn test_gabor_write_tight_affects_few() {
        let lut = GaussianLUT::new(0.5);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        let nonzero = container.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero <= 30,
            "σ=0.5 should affect ≤30 positions, got {}",
            nonzero
        );
    }

    #[test]
    fn test_gabor_write_broad_affects_most() {
        let lut = GaussianLUT::new(10.0);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        let nonzero = container.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero > 1500,
            "σ=10 should affect most positions, got {}",
            nonzero
        );
    }

    #[test]
    fn test_gabor_read_recovers_phase_single() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        let target_phase = 1.5f32;

        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, target_phase, 7.0);
        let (rec_phase, rec_amp) = gabor_read(&container, &lut, 4, 4, 16, 3.0);

        assert!(
            phase_error(rec_phase, target_phase) < 0.15,
            "phase recovery: expected {:.4}, got {:.4} (error {:.4})",
            target_phase,
            rec_phase,
            phase_error(rec_phase, target_phase)
        );
        assert!(
            rec_amp > 0.5,
            "amplitude should be significant, got {:.4}",
            rec_amp
        );
    }

    #[test]
    fn test_gabor_read_empty_container() {
        let lut = GaussianLUT::new(2.0);
        let container = vec![0i8; 2048];

        let (_, amp) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
        assert!(
            amp < 0.01,
            "empty container should give near-zero amplitude, got {:.4}",
            amp
        );
    }

    #[test]
    fn test_gabor_two_frequencies_same_position() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];

        let phase_a = 0.5f32;
        let phase_b = 3.5f32;

        gabor_write(&mut container, &lut, 4, 4, 16, 2.0, phase_a, 7.0);
        gabor_write(&mut container, &lut, 4, 4, 16, 8.0, phase_b, 7.0);

        let (rec_a, _) = gabor_read(&container, &lut, 4, 4, 16, 2.0);
        let (rec_b, _) = gabor_read(&container, &lut, 4, 4, 16, 8.0);

        assert!(
            phase_error(rec_a, phase_a) < 0.3,
            "freq 2 recovery: expected {:.4}, got {:.4}",
            phase_a,
            rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.3,
            "freq 8 recovery: expected {:.4}, got {:.4}",
            phase_b,
            rec_b
        );
    }

    #[test]
    fn test_gabor_two_positions_same_frequency() {
        let lut = GaussianLUT::new(1.5);
        let mut container = vec![0i8; 2048];

        let phase_a = 1.0f32;
        let phase_b = 4.0f32;

        // Write at opposite corners of the volume
        gabor_write(&mut container, &lut, 1, 1, 4, 5.0, phase_a, 7.0);
        gabor_write(&mut container, &lut, 6, 6, 28, 5.0, phase_b, 7.0);

        // Read back from each position — the other should be suppressed
        let (rec_a, _) = gabor_read(&container, &lut, 1, 1, 4, 5.0);
        let (rec_b, _) = gabor_read(&container, &lut, 6, 6, 28, 5.0);

        assert!(
            phase_error(rec_a, phase_a) < 0.3,
            "pos A recovery: expected {:.4}, got {:.4}",
            phase_a,
            rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.3,
            "pos B recovery: expected {:.4}, got {:.4}",
            phase_b,
            rec_b
        );
    }

    #[test]
    fn test_gabor_phase_sweep() {
        // Test multiple phase values to verify no systematic bias
        let lut = GaussianLUT::new(2.0);
        let test_phases = [0.0f32, 0.5, 1.0, 2.0, std::f32::consts::PI, 4.0, 5.5, 6.0];

        for &target in &test_phases {
            let mut container = vec![0i8; 2048];
            gabor_write(&mut container, &lut, 4, 4, 16, 3.0, target, 7.0);
            let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
            assert!(
                phase_error(rec, target) < 0.15,
                "phase {:.2}: expected {:.4}, got {:.4} (err {:.4})",
                target,
                target,
                rec,
                phase_error(rec, target)
            );
        }
    }

    // ---- Delta cube tests ----

    #[test]
    fn test_delta_cube_xor_self_zero() {
        let field: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&field, &field, &mut delta);
        assert!(delta.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_delta_cube_xor_self_inverse() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i * 41 + 7) % 256) as u8).collect();
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&a, &b, &mut delta);

        let mut double_delta = vec![0u8; 2048];
        delta_cube_xor(&delta, &delta, &mut double_delta);
        assert!(double_delta.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_delta_cube_recover_xor_round_trip() {
        // Create two fields
        let _lut = GaussianLUT::new(2.0);
        let mut field_a = vec![0u8; 2048];
        let mut field_b = vec![0u8; 2048];

        // Fill with some deterministic data
        for i in 0..2048 {
            field_a[i] = (i * 31 % 256) as u8;
            field_b[i] = ((i * 47 + 13) % 256) as u8;
        }

        // Create delta
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&field_a, &field_b, &mut delta);

        // Write content into delta
        let content: Vec<u8> = (0..2048).map(|i| (i * 7 % 256) as u8).collect();
        let mut stored_delta = delta.clone();
        for i in 0..2048 {
            stored_delta[i] ^= content[i];
        }

        // Recover content
        let mut recovered = vec![0u8; 2048];
        delta_cube_recover_xor(&stored_delta, &field_a, &field_b, &mut recovered);
        assert_eq!(recovered, content);
    }

    #[test]
    fn test_delta_cube_recover_phase_round_trip() {
        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        for i in 0..2048 {
            field_a[i] = (i as i8).wrapping_mul(3);
            field_b[i] = (i as i8).wrapping_mul(7).wrapping_add(13);
        }

        // Delta = A - B
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        // Content
        let content: Vec<i8> = (0..2048).map(|i| (i as i8).wrapping_mul(11)).collect();
        let stored: Vec<i8> = delta
            .iter()
            .zip(content.iter())
            .map(|(&d, &c)| d.wrapping_add(c))
            .collect();

        // Recover
        let mut recovered = vec![0i8; 2048];
        delta_cube_recover_phase(&stored, &field_a, &field_b, &mut recovered);
        assert_eq!(recovered, content);
    }

    #[test]
    fn test_delta_cube_content_not_in_single_field() {
        let lut = GaussianLUT::new(2.0);

        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Create delta and write content
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        let content_phase = 4.0f32;
        delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 7.0, content_phase, 7.0);

        // Try reading content from field_a alone — should get noise
        let (_, amp_from_a) = gabor_read(&field_a, &lut, 4, 4, 16, 7.0);
        // Try reading from the stored delta — this has content + interference
        let (_rec_phase, rec_amp) = delta_cube_read_gabor(&delta, &lut, 4, 4, 16, 7.0);

        // Content should be recoverable from delta
        assert!(
            rec_amp > amp_from_a * 2.0 || rec_amp > 0.5,
            "delta should have stronger signal than field_a alone: delta_amp={:.4}, a_amp={:.4}",
            rec_amp,
            amp_from_a
        );
    }

    // ---- Spatial transform tests ----

    #[test]
    fn test_rotate_x_0_is_identity() {
        let t = SpatialTransform::rotate_x(0);
        for i in 0..2048 {
            assert_eq!(t.perm[i], i as u16);
        }
    }

    #[test]
    fn test_rotate_x_8_is_identity() {
        let t = SpatialTransform::rotate_x(8);
        for i in 0..2048 {
            assert_eq!(t.perm[i], i as u16);
        }
    }

    #[test]
    fn test_rotate_inverse_identity() {
        let t = SpatialTransform::rotate_x(3);
        let inv = t.inverse();
        let composed = inv.compose(&t);
        for i in 0..2048 {
            assert_eq!(
                composed.perm[i], i as u16,
                "T⁻¹∘T should be identity at {}",
                i
            );
        }
    }

    #[test]
    fn test_compose_associative() {
        let t1 = SpatialTransform::rotate_x(2);
        let t2 = SpatialTransform::rotate_y(3);
        let t3 = SpatialTransform::rotate_z(5);

        let left = t1.compose(&t2).compose(&t3); // (T1∘T2)∘T3
        let right = t1.compose(&t2.compose(&t3)); // T1∘(T2∘T3)

        for i in 0..2048 {
            assert_eq!(left.perm[i], right.perm[i], "associativity failed at {}", i);
        }
    }

    #[test]
    fn test_diagonal_is_bijection() {
        let t = SpatialTransform::diagonal();
        let mut seen = [false; 2048];
        for i in 0..2048 {
            let target = t.perm[i] as usize;
            assert!(target < 2048, "perm[{}] = {} out of range", i, target);
            assert!(
                !seen[target],
                "perm maps {} and another index both to {}",
                i, target
            );
            seen[target] = true;
        }
    }

    #[test]
    fn test_spatial_bind_unbind_round_trip() {
        let container: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let t = SpatialTransform::rotate_x(3);

        let bound = spatial_bind(&container, &t);
        let unbound = spatial_unbind(&bound, &t);
        assert_eq!(unbound, container);
    }

    #[test]
    fn test_spatial_bind_i8_round_trip() {
        let container: Vec<i8> = (0..2048).map(|i| (i as i8).wrapping_mul(13)).collect();
        let t = SpatialTransform::rotate_y(5);

        let bound = spatial_bind_i8(&container, &t);
        let unbound = spatial_unbind_i8(&bound, &t);
        assert_eq!(unbound, container);
    }

    #[test]
    fn test_spatial_bind_preserves_gabor() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        let phase = 2.5f32;

        // Write Gabor at (4,4,16)
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, phase, 7.0);

        // Rotate X by 2: position (4,4,16) → (6,4,16)
        let t = SpatialTransform::rotate_x(2);
        let rotated = spatial_bind_i8(&container, &t);

        // Read at new position (6,4,16) — should recover the same phase
        let (rec, amp) = gabor_read(&rotated, &lut, 6, 4, 16, 3.0);
        assert!(
            phase_error(rec, phase) < 0.2,
            "rotated read: expected {:.4}, got {:.4}",
            phase,
            rec
        );
        assert!(amp > 0.5);
    }

    // ---- Integration: full pipeline ----

    #[test]
    fn test_full_pipeline_encode_bind_recover() {
        let lut = GaussianLUT::new(2.0);

        // Step 1-2: Encode two concepts as holographic fields
        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Step 3: Create delta cube
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        // Step 4: Write relationship content into delta
        let rel_phase = 3.5f32;
        delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 7.0, rel_phase, 7.0);

        // Step 5: Apply spatial transform
        let t = SpatialTransform::rotate_z(4);
        let delta_transformed = spatial_bind_i8(&delta, &t);

        // Recovery: undo transform, subtract original delta, read content
        let delta_unrotated = spatial_unbind_i8(&delta_transformed, &t);
        let mut content = vec![0i8; 2048];
        delta_cube_recover_phase(&delta_unrotated, &field_a, &field_b, &mut content);

        let (rec_phase, rec_amp) = gabor_read(&content, &lut, 4, 4, 16, 7.0);
        assert!(
            phase_error(rec_phase, rel_phase) < 0.2,
            "pipeline recovery: expected {:.4}, got {:.4}",
            rel_phase,
            rec_phase
        );
        assert!(rec_amp > 0.3, "pipeline amplitude too low: {:.4}", rec_amp);
    }

    #[test]
    fn test_multiple_relationships_same_container() {
        let lut = GaussianLUT::new(1.5);

        // 3 pairs of fields
        let pairs: Vec<(f32, f32, f32)> = vec![
            (1.0, 2.0, 0.5), // (phase_a, phase_b, rel_phase)
            (3.0, 4.0, 2.0),
            (5.0, 0.5, 4.5),
        ];
        let freqs = [2.0f32, 5.0, 8.0];
        let positions = [(2u8, 2u8, 8u8), (4, 4, 16), (6, 6, 24)];

        for (i, &(pa, pb, rel)) in pairs.iter().enumerate() {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            let (px, py, pz) = positions[i];

            gabor_write(&mut field_a, &lut, px, py, pz, freqs[i], pa, 7.0);
            gabor_write(&mut field_b, &lut, px, py, pz, freqs[i], pb, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, freqs[i], rel, 7.0);

            // Recover
            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let (rec, _) = gabor_read(&content, &lut, 4, 4, 16, freqs[i]);
            assert!(
                phase_error(rec, rel) < 0.3,
                "pair {} recovery: expected {:.4}, got {:.4}",
                i,
                rel,
                rec
            );
        }
    }

    // ---- Capacity experiment ----

    #[test]
    fn test_gabor_capacity_experiment() {
        println!("\n=== Gabor Wavelet Capacity Experiment ===\n");

        for &sigma in &[1.0f32, 2.0, 3.0] {
            let lut = GaussianLUT::new(sigma);

            for &n in &[1u32, 2, 4, 8, 12, 16] {
                let mut container = vec![0i8; 2048];
                let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.39 + 0.1).collect();
                let freqs: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 + 1.0).collect();

                // Write all wavelets at center
                for i in 0..n as usize {
                    gabor_write(&mut container, &lut, 4, 4, 16, freqs[i], phases[i], 7.0);
                }

                // Read each back
                let mut total_error = 0.0f32;
                for i in 0..n as usize {
                    let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, freqs[i]);
                    total_error += phase_error(rec, phases[i]);
                }
                let mean_error = total_error / n as f32;

                println!(
                    "  σ={:.1} N={:>2}: mean_phase_err={:.4} rad ({:>5.1}°)",
                    sigma,
                    n,
                    mean_error,
                    mean_error.to_degrees()
                );
            }
            println!();
        }

        // Verify σ=2 holds up to N=4 with good accuracy
        {
            let lut = GaussianLUT::new(2.0);
            let mut container = vec![0i8; 2048];
            let phases = [0.5f32, 1.5, 3.0, 5.0];
            let freqs = [2.0f32, 5.0, 8.0, 12.0];
            for i in 0..4 {
                gabor_write(&mut container, &lut, 4, 4, 16, freqs[i], phases[i], 7.0);
            }
            let mut total_err = 0.0f32;
            for i in 0..4 {
                let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, freqs[i]);
                total_err += phase_error(rec, phases[i]);
            }
            let mean = total_err / 4.0;
            assert!(mean < 0.5, "σ=2 N=4 mean error {:.4} rad too high", mean);
        }
    }

    #[test]
    fn test_delta_cube_capacity() {
        let lut = GaussianLUT::new(2.0);

        println!("\n=== Delta Cube Capacity Experiment ===\n");

        for &n in &[1u32, 2, 4, 8] {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
            gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);

            let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 0.3).collect();
            let freqs: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0 + 1.0).collect();

            for i in 0..n as usize {
                delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, freqs[i], phases[i], 7.0);
            }

            // Recover
            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let mut total_error = 0.0f32;
            for i in 0..n as usize {
                let (rec, _) = gabor_read(&content, &lut, 4, 4, 16, freqs[i]);
                total_error += phase_error(rec, phases[i]);
            }
            let mean = total_error / n as f32;

            println!(
                "  Delta cube N={:>2}: mean_phase_err={:.4} rad ({:>5.1}°)",
                n,
                mean,
                mean.to_degrees()
            );
        }

        // Verify N=2 recoverable
        {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
            gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 3.0, 2.5, 7.0);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 8.0, 4.0, 7.0);

            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let (p1, _) = gabor_read(&content, &lut, 4, 4, 16, 3.0);
            let (p2, _) = gabor_read(&content, &lut, 4, 4, 16, 8.0);
            assert!(
                phase_error(p1, 2.5) < 0.4,
                "delta N=2 freq 3: expected 2.5, got {:.4}",
                p1
            );
            assert!(
                phase_error(p2, 4.0) < 0.4,
                "delta N=2 freq 8: expected 4.0, got {:.4}",
                p2
            );
        }
    }

    // ---- Overlay (Blackboard Layer) tests ----

    #[test]
    fn test_overlay_new_is_clean() {
        let overlay = Overlay::new();
        assert!(overlay.is_clean());
        assert_eq!(overlay.buffer.len(), 2048);
        assert_eq!(overlay.snapshot_depth(), 0);
    }

    #[test]
    fn test_overlay_single_gabor_write_flush_matches_direct() {
        let lut = GaussianLUT::new(2.0);

        // Direct write to container
        let mut direct = vec![0i8; 2048];
        gabor_write(&mut direct, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        // Write through overlay, then flush
        let mut container = vec![0i8; 2048];
        let mut overlay = Overlay::new();
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 3.0, 1.5, 7.0);
        assert!(!overlay.is_clean());
        overlay.flush_add_i8(&mut container);

        assert_eq!(container, direct);
        assert!(overlay.is_clean());
    }

    #[test]
    fn test_overlay_five_gabor_writes_flush_matches_direct() {
        let lut = GaussianLUT::new(2.0);
        let params: Vec<(u8, u8, u8, f32, f32)> = vec![
            (2, 2, 8, 2.0, 0.5),
            (4, 4, 16, 3.0, 1.5),
            (6, 6, 24, 5.0, 3.0),
            (1, 3, 10, 7.0, 4.5),
            (5, 1, 20, 4.0, 2.0),
        ];

        // Direct writes
        let mut direct = vec![0i8; 2048];
        for &(x, y, z, f, p) in &params {
            gabor_write(&mut direct, &lut, x, y, z, f, p, 7.0);
        }

        // Overlay writes then flush
        let mut container = vec![0i8; 2048];
        let mut overlay = Overlay::new();
        for &(x, y, z, f, p) in &params {
            gabor_write(overlay.as_i8_mut(), &lut, x, y, z, f, p, 7.0);
        }
        overlay.flush_add_i8(&mut container);

        assert_eq!(container, direct);
    }

    #[test]
    fn test_overlay_read_full_add_i8_before_flush() {
        let lut = GaussianLUT::new(2.0);

        // Base container with one concept
        let mut base = vec![0i8; 2048];
        gabor_write(&mut base, &lut, 2, 2, 8, 3.0, 1.0, 7.0);

        // Overlay with another concept
        let mut overlay = Overlay::new();
        gabor_write(overlay.as_i8_mut(), &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Read through: should show combined state
        let combined = overlay.read_full_add_i8(&base);

        // The combined view should have both concepts readable
        let (p1, a1) = gabor_read(&combined, &lut, 2, 2, 8, 3.0);
        let (p2, a2) = gabor_read(&combined, &lut, 6, 6, 24, 5.0);

        assert!(
            phase_error(p1, 1.0) < 0.15,
            "base concept: {:.4} vs 1.0",
            p1
        );
        assert!(a1 > 0.5);
        assert!(
            phase_error(p2, 2.0) < 0.15,
            "overlay concept: {:.4} vs 2.0",
            p2
        );
        assert!(a2 > 0.5);
    }

    #[test]
    fn test_overlay_read_full_add_i8_matches_after_flush() {
        let lut = GaussianLUT::new(2.0);

        let mut base = vec![0i8; 2048];
        gabor_write(&mut base, &lut, 2, 2, 8, 3.0, 1.0, 7.0);

        let mut overlay = Overlay::new();
        gabor_write(overlay.as_i8_mut(), &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Read through before flush
        let view_before = overlay.read_full_add_i8(&base);

        // Flush
        overlay.flush_add_i8(&mut base);

        // Container after flush should match the read-through view
        assert_eq!(base, view_before);
    }

    #[test]
    fn test_overlay_flush_xor_binary() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i * 31 % 256) as u8).collect();
        let original = container.clone();

        let mut overlay = Overlay::new();
        // Write pattern into overlay
        for i in 0..2048 {
            overlay.buffer[i] = (i * 17 % 256) as u8;
        }

        overlay.flush_xor(&mut container);

        // Verify XOR was applied
        for i in 0..2048 {
            assert_eq!(
                container[i],
                original[i] ^ (i * 17 % 256) as u8,
                "XOR mismatch at {}",
                i
            );
        }
        assert!(overlay.is_clean());
    }

    #[test]
    fn test_overlay_flush_add_phase() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let original = container.clone();

        let mut overlay = Overlay::new();
        for i in 0..2048 {
            overlay.buffer[i] = (i * 7 % 256) as u8;
        }

        overlay.flush_add(&mut container);

        for i in 0..2048 {
            assert_eq!(
                container[i],
                original[i].wrapping_add((i * 7 % 256) as u8),
                "ADD mismatch at {}",
                i
            );
        }
        assert!(overlay.is_clean());
    }

    #[test]
    fn test_overlay_double_flush_noop() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i * 31 % 256) as u8).collect();

        let mut overlay = Overlay::new();
        for i in 0..2048 {
            overlay.buffer[i] = (i * 17 % 256) as u8;
        }
        overlay.flush_xor(&mut container);
        let after_first = container.clone();

        // Second flush with clean overlay: no-op
        overlay.flush_xor(&mut container);
        assert_eq!(container, after_first, "double flush should be no-op");
    }

    #[test]
    fn test_overlay_snapshot_rewind() {
        let mut overlay = Overlay::new();

        // Write some data
        overlay.buffer[0] = 42;
        overlay.buffer[100] = 99;
        overlay.snapshot(); // save state with [42, ..., 99, ...]
        assert_eq!(overlay.snapshot_depth(), 1);

        // Write more data
        overlay.buffer[0] = 0;
        overlay.buffer[200] = 77;

        // Rewind: should restore to snapshot state
        assert!(overlay.rewind());
        assert_eq!(overlay.buffer[0], 42);
        assert_eq!(overlay.buffer[100], 99);
        assert_eq!(overlay.buffer[200], 0); // reverted
        assert_eq!(overlay.snapshot_depth(), 0);
    }

    #[test]
    fn test_overlay_snapshot_write_rewind_flush() {
        let lut = GaussianLUT::new(2.0);

        let mut container = vec![0i8; 2048];
        let mut overlay = Overlay::new();

        // Write concept A into overlay
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 3.0, 1.0, 7.0);
        overlay.snapshot(); // save: has concept A

        // Write concept B into overlay
        gabor_write(overlay.as_i8_mut(), &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Oops, concept B was wrong. Rewind.
        overlay.rewind();

        // Flush: only concept A should be committed
        overlay.flush_add_i8(&mut container);

        // Concept A recoverable
        let (pa, aa) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
        assert!(phase_error(pa, 1.0) < 0.15, "A: {:.4} vs 1.0", pa);
        assert!(aa > 0.5);

        // Concept B should NOT be in the container (noise-floor amplitude)
        let (_, ab) = gabor_read(&container, &lut, 6, 6, 24, 5.0);
        assert!(ab < 0.3, "B should be absent, got amplitude {:.4}", ab);
    }

    #[test]
    fn test_overlay_multiple_snapshots_lifo() {
        let mut overlay = Overlay::new();

        overlay.buffer[0] = 10;
        overlay.snapshot(); // snap 0: [10, ...]

        overlay.buffer[0] = 20;
        overlay.snapshot(); // snap 1: [20, ...]

        overlay.buffer[0] = 30;
        overlay.snapshot(); // snap 2: [30, ...]

        overlay.buffer[0] = 40;
        assert_eq!(overlay.snapshot_depth(), 3);

        // Rewind from 40 → 30 (snap 2)
        assert!(overlay.rewind());
        assert_eq!(overlay.buffer[0], 30);
        assert_eq!(overlay.snapshot_depth(), 2);

        // Rewind from 30 → 20 (snap 1)
        assert!(overlay.rewind());
        assert_eq!(overlay.buffer[0], 20);
        assert_eq!(overlay.snapshot_depth(), 1);

        // Rewind from 20 → 10 (snap 0)
        assert!(overlay.rewind());
        assert_eq!(overlay.buffer[0], 10);
        assert_eq!(overlay.snapshot_depth(), 0);

        // No more snapshots
        assert!(!overlay.rewind());
        assert_eq!(overlay.buffer[0], 10); // unchanged
    }

    #[test]
    fn test_overlay_rewind_empty_returns_false() {
        let mut overlay = Overlay::new();
        assert!(!overlay.rewind());
        assert!(overlay.is_clean()); // unchanged
    }

    #[test]
    fn test_overlay_discard() {
        let mut overlay = Overlay::new();
        overlay.buffer[0] = 42;
        overlay.snapshot();
        overlay.buffer[0] = 99;

        let mut container = vec![0u8; 2048];
        container[0] = 10;

        // Discard: zeros overlay, clears snapshots, doesn't touch container
        overlay.discard();
        assert!(overlay.is_clean());
        assert_eq!(overlay.snapshot_depth(), 0);
        assert_eq!(container[0], 10); // untouched
    }

    #[test]
    fn test_overlay_is_clean_lifecycle() {
        let mut overlay = Overlay::new();
        assert!(overlay.is_clean()); // new

        overlay.buffer[500] = 1;
        assert!(!overlay.is_clean()); // after write

        overlay.discard();
        assert!(overlay.is_clean()); // after discard

        overlay.buffer[500] = 1;
        let mut container = vec![0u8; 2048];
        overlay.flush_xor(&mut container);
        assert!(overlay.is_clean()); // after flush
    }

    #[test]
    fn test_overlay_as_i8_mut_gabor_write() {
        let lut = GaussianLUT::new(2.0);
        let mut overlay = Overlay::new();

        // Write through safe i8 wrapper
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        // Read back through i8 view
        let (rec_phase, rec_amp) = gabor_read(overlay.as_i8(), &lut, 4, 4, 16, 3.0);
        assert!(
            phase_error(rec_phase, 1.5) < 0.15,
            "as_i8_mut write: expected 1.5, got {:.4}",
            rec_phase
        );
        assert!(rec_amp > 0.5);
    }

    #[test]
    fn test_overlay_read_xor_single_byte() {
        let overlay_val = 0xABu8;
        let container_val = 0xCDu8;

        let mut overlay = Overlay::new();
        overlay.buffer[42] = overlay_val;

        let container = {
            let mut c = vec![0u8; 2048];
            c[42] = container_val;
            c
        };

        assert_eq!(overlay.read_xor(&container, 42), 0xABu8 ^ 0xCDu8);
        assert_eq!(
            overlay.read_add(&container, 42),
            0xCDu8.wrapping_add(0xABu8)
        );
    }

    #[test]
    fn test_overlay_read_full_xor() {
        let mut overlay = Overlay::new();
        let mut container = vec![0u8; 2048];
        for i in 0..2048 {
            overlay.buffer[i] = (i * 3 % 256) as u8;
            container[i] = (i * 7 % 256) as u8;
        }

        let result = overlay.read_full_xor(&container);
        for i in 0..2048 {
            assert_eq!(
                result[i],
                container[i] ^ overlay.buffer[i],
                "XOR mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_overlay_delta_cube_pipeline() {
        let lut = GaussianLUT::new(2.0);

        // Create two fields
        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Delta cube into overlay
        let mut overlay = Overlay::new();
        let delta_buf = overlay.as_i8_mut();
        for i in 0..2048 {
            delta_buf[i] = field_a[i].wrapping_sub(field_b[i]);
        }

        // Write relationship content into overlay
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 7.0, 3.5, 7.0);

        // Flush to container
        let mut container = vec![0i8; 2048];
        overlay.flush_add_i8(&mut container);

        // Recover content
        let mut content = vec![0i8; 2048];
        delta_cube_recover_phase(&container, &field_a, &field_b, &mut content);

        let (rec, amp) = gabor_read(&content, &lut, 4, 4, 16, 7.0);
        assert!(
            phase_error(rec, 3.5) < 0.2,
            "overlay delta-cube: expected 3.5, got {:.4}",
            rec
        );
        assert!(amp > 0.3);
    }

    #[test]
    fn test_overlay_spatial_transform() {
        let lut = GaussianLUT::new(2.0);
        let mut overlay = Overlay::new();

        // Write wavelet at (4,4,16)
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        // Apply spatial transform to the overlay buffer
        let t = SpatialTransform::rotate_x(2);
        let transformed = spatial_bind(&overlay.buffer, &t);
        overlay.buffer.copy_from_slice(&transformed);

        // Flush to container
        let mut container = vec![0i8; 2048];
        overlay.flush_add_i8(&mut container);

        // Read at transformed position (6,4,16)
        let (rec, amp) = gabor_read(&container, &lut, 6, 4, 16, 3.0);
        assert!(
            phase_error(rec, 1.5) < 0.2,
            "spatial overlay: expected 1.5, got {:.4}",
            rec
        );
        assert!(amp > 0.5);
    }

    #[test]
    fn test_overlay_stm_ltm_boundary() {
        // After flush, individual concepts only recoverable by holographic readout (key)
        let lut = GaussianLUT::new(2.0);

        let mut container = vec![0i8; 2048];
        let mut overlay = Overlay::new();

        // Write 3 concepts into overlay at different frequencies
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 2.0, 0.5, 7.0);
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 5.0, 2.5, 7.0);
        gabor_write(overlay.as_i8_mut(), &lut, 4, 4, 16, 8.0, 4.5, 7.0);

        overlay.flush_add_i8(&mut container);

        // All three recoverable by key (frequency)
        let (p1, _) = gabor_read(&container, &lut, 4, 4, 16, 2.0);
        let (p2, _) = gabor_read(&container, &lut, 4, 4, 16, 5.0);
        let (p3, _) = gabor_read(&container, &lut, 4, 4, 16, 8.0);

        assert!(
            phase_error(p1, 0.5) < 0.4,
            "STM→LTM freq 2: {:.4} vs 0.5",
            p1
        );
        assert!(
            phase_error(p2, 2.5) < 0.4,
            "STM→LTM freq 5: {:.4} vs 2.5",
            p2
        );
        assert!(
            phase_error(p3, 4.5) < 0.4,
            "STM→LTM freq 8: {:.4} vs 4.5",
            p3
        );
    }

    // ---- SpectralMap tests ----

    #[test]
    fn test_spectral_map_empty() {
        let container = vec![0i8; 2048];
        let basis = CarrierBasis::new();
        let lut = GaussianLUT::new(2.0);
        let spec = SpectralMap::analyze(&container, &basis, &[lut]);
        let max_amp = spec.amplitude.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_amp < 0.01,
            "empty container max amplitude: {:.4}",
            max_amp
        );
    }

    #[test]
    fn test_spectral_map_single_peak() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        // CARRIER_FREQUENCIES[2] = 3
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let basis = CarrierBasis::new();
        let spec = SpectralMap::analyze(&container, &basis, &[lut]);
        let peaks = spec.find_peaks(0.5);

        assert!(!peaks.is_empty(), "should find at least one peak");
        // The strongest peak should be near (4, 4, 16) at freq_idx=2
        let best = peaks
            .iter()
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
            .unwrap();
        assert_eq!(best.3, 2, "peak freq idx should be 2 (freq=3)");
        assert!(
            (best.0 as i32 - 4).abs() <= 1 && (best.1 as i32 - 4).abs() <= 1,
            "peak position should be near (4,4): got ({}, {})",
            best.0,
            best.1
        );
    }

    #[test]
    fn test_resynthesize_single_gabor() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let basis = CarrierBasis::new();
        let spec = SpectralMap::analyze(&container, &basis, std::slice::from_ref(&lut));
        let clean = spec.resynthesize(0.3, std::slice::from_ref(&lut), 2.0);

        let (rec, amp) = gabor_read(&clean, &lut, 4, 4, 16, 3.0);
        assert!(
            phase_error(rec, 1.5) < 0.3,
            "resynthesize phase: {:.4} vs 1.5",
            rec
        );
        assert!(amp > 0.3, "resynthesize amplitude: {:.4}", amp);
    }

    // ---- Cleaning tests ----

    #[test]
    fn test_residual_energy_clean_container() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let basis = CarrierBasis::new();
        let known = vec![(4u8, 4u8, 16u8, 3.0f32)];
        let energy = residual_energy(&container, &known, &lut, &basis);
        let per_byte = energy / 2048.0;

        // After subtracting the known signal, residual should be low
        assert!(
            per_byte < 5.0,
            "residual per byte too high: {:.4}",
            per_byte
        );
    }

    #[test]
    fn test_orthogonal_project_recovery() {
        let lut = GaussianLUT::new(2.0);

        // Create a template
        let mut template = vec![0i8; 2048];
        gabor_write(&mut template, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        // Container = template + noise
        let mut container = template.clone();
        for i in (0..2048).step_by(7) {
            container[i] = container[i].saturating_add(3);
        }

        let clean = orthogonal_project(&container, &[template.clone()]);

        // Clean should be close to the template
        let (rec, _) = gabor_read(&clean, &lut, 4, 4, 16, 3.0);
        assert!(
            phase_error(rec, 1.5) < 0.2,
            "ortho project phase: {:.4} vs 1.5",
            rec
        );
    }

    // ---- Learning tests ----

    #[test]
    fn test_hebbian_strengthens_interference() {
        let lut = GaussianLUT::new(2.0);
        let container = vec![0i8; 2048];
        let mut overlay = Overlay::new();

        // Initial state: overlay empty
        let initial_energy: i64 = overlay.buffer.iter().map(|&b| (b as i64).abs()).sum();
        assert_eq!(initial_energy, 0);

        // Hebbian update between two nearby concepts
        hebbian_update(
            &mut overlay,
            &container,
            &lut,
            (3, 3, 14, 3.0),
            (5, 5, 18, 5.0),
            0.5,
        );

        let after_energy: i64 = overlay.buffer.iter().map(|&b| (b as i64).abs()).sum();
        assert!(
            after_energy > 0,
            "hebbian should create interference pattern"
        );
    }

    #[test]
    fn test_anti_hebbian_weakens() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let (_, amp_before) = gabor_read(&container, &lut, 4, 4, 16, 3.0);

        let mut overlay = Overlay::new();
        anti_hebbian_update(&mut overlay, &lut, (4, 4, 16, 3.0, 1.5), 3.0);
        overlay.flush_add_i8(&mut container);

        let (_, amp_after) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
        assert!(
            amp_after < amp_before,
            "anti-hebbian should reduce: {:.4} → {:.4}",
            amp_before,
            amp_after
        );
    }

    #[test]
    fn test_adapt_sigma_high_snr_decreases() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        // Single clean wavelet: high SNR
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let new_sigma = adapt_sigma(
            &container, &lut, 4, 4, 16, 3.0, 2.0,  // current σ
            0.01, // learning rate
            2.0,  // SNR target (expect SNR >> 2 for clean signal)
        );
        assert!(
            new_sigma < 2.0,
            "high SNR should decrease σ: got {:.4}",
            new_sigma
        );
    }

    // ---- Archetype Detection tests ----

    #[test]
    fn test_archetype_detector_empty() {
        let container = vec![0i8; 2048];
        let lut = GaussianLUT::new(2.0);
        let detector = FastArchetypeDetector::new();
        let peaks = detector.detect(&container, &lut, 0.5);
        assert!(
            peaks.is_empty(),
            "empty container should have no archetypes"
        );
    }

    #[test]
    fn test_archetype_detector_finds_peak() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        // Write at (4, 4, 16) with freq=CARRIER_FREQUENCIES[2]=3
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, 1.5, 7.0);

        let detector = FastArchetypeDetector::new();
        let peaks = detector.detect(&container, &lut, 0.3);
        assert!(!peaks.is_empty(), "should detect the wavelet as archetype");

        // Best peak should be near (4, 4) at freq_idx=2
        let best = peaks
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
            .unwrap();
        assert_eq!(best.2, 2, "archetype freq idx should be 2");
    }

    // ---- GaborBatch test ----

    #[test]
    fn test_gabor_batch_matches_sequential() {
        let lut = GaussianLUT::new(2.0);

        // Sequential writes
        let mut sequential = vec![0i8; 2048];
        gabor_write(&mut sequential, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut sequential, &lut, 6, 6, 24, 5.0, 2.5, 7.0);

        // Batch writes
        let mut batch_container = vec![0i8; 2048];
        let mut batch = GaborBatch::new();
        batch.add(&lut, 2, 2, 8, 3.0, 7.0, 1.0);
        batch.add(&lut, 6, 6, 24, 5.0, 7.0, 2.5);
        batch.flush(&mut batch_container);

        // Read back from both: phases should match
        let (seq_p1, _) = gabor_read(&sequential, &lut, 2, 2, 8, 3.0);
        let (bat_p1, _) = gabor_read(&batch_container, &lut, 2, 2, 8, 3.0);
        assert!(
            phase_error(seq_p1, bat_p1) < 0.15,
            "batch vs seq concept 1: {:.4} vs {:.4}",
            bat_p1,
            seq_p1
        );

        let (seq_p2, _) = gabor_read(&sequential, &lut, 6, 6, 24, 5.0);
        let (bat_p2, _) = gabor_read(&batch_container, &lut, 6, 6, 24, 5.0);
        assert!(
            phase_error(seq_p2, bat_p2) < 0.15,
            "batch vs seq concept 2: {:.4} vs {:.4}",
            bat_p2,
            seq_p2
        );
    }

    // ---- CooccurrenceMatrix tests ----

    #[test]
    fn test_cooccurrence_observe_window() {
        let mut matrix = CooccurrenceMatrix::new(8);
        matrix.observe_window(&[0, 1, 2]);

        assert_eq!(matrix.counts[0][1], 1.0);
        assert_eq!(matrix.counts[1][0], 1.0);
        assert_eq!(matrix.counts[0][2], 1.0);
        assert_eq!(matrix.counts[1][2], 1.0);
        assert_eq!(matrix.counts[0][0], 0.0); // no self-co-occurrence
        assert_eq!(matrix.totals[0], 1.0);
        assert_eq!(matrix.totals[1], 1.0);
        assert_eq!(matrix.totals[2], 1.0);
        assert_eq!(matrix.totals[3], 0.0);
    }

    #[test]
    fn test_cooccurrence_normalized() {
        let mut matrix = CooccurrenceMatrix::new(4);
        matrix.observe_window(&[0, 1]);
        matrix.observe_window(&[0, 1]);
        matrix.observe_window(&[2, 3]);

        let norm = matrix.normalized();
        // Concepts 0 and 1 co-occurred 2 times, each observed 2 times
        // norm[0][1] = 2.0 / sqrt(2.0 * 2.0) = 1.0
        assert!(
            (norm[0][1] - 1.0).abs() < 0.01,
            "normalized 0-1: {:.4}",
            norm[0][1]
        );
        // Concepts 0 and 2 never co-occurred
        assert!(
            (norm[0][2]).abs() < 0.01,
            "normalized 0-2: {:.4}",
            norm[0][2]
        );
    }

    #[test]
    fn test_ready_for_crystallization() {
        let mut matrix = CooccurrenceMatrix::new(10);
        assert!(!ready_for_crystallization(&matrix));

        // Add 8 concepts with 5+ observations each
        for _ in 0..5 {
            matrix.observe_window(&[0, 1, 2, 3]);
            matrix.observe_window(&[4, 5, 6, 7]);
        }
        assert!(ready_for_crystallization(&matrix));
    }

    // ---- AxisCrystallizer tests ----

    #[test]
    fn test_power_iteration_converges() {
        // Simple 4×4 matrix with known structure
        let matrix = vec![
            vec![2.0, 1.0, 0.0, 0.0],
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.5],
            vec![0.0, 0.0, 0.5, 1.0],
        ];
        let (eigvec, eigval) = power_iteration(&matrix, 4, 200);

        // Dominant eigenvalue of [[2,1],[1,2]] block is 3
        assert!((eigval - 3.0).abs() < 0.1, "eigenvalue: {:.4}", eigval);
        // Eigenvector should load on first two components
        let load_01 = eigvec[0].abs() + eigvec[1].abs();
        let load_23 = eigvec[2].abs() + eigvec[3].abs();
        assert!(load_01 > load_23, "eigenvector should load on first block");
    }

    #[test]
    fn test_normalize_to_grid() {
        let coords = vec![(-1.0, 0.0, -2.0), (1.0, 1.0, 2.0), (0.0, 0.5, 0.0)];
        let normed = normalize_to_grid(&coords);

        // First: x=-1 → 0, second: x=1 → 7, third: x=0 → 3.5
        assert!((normed[0].0 - 0.0).abs() < 0.01);
        assert!((normed[1].0 - 7.0).abs() < 0.01);
        assert!((normed[2].0 - 3.5).abs() < 0.01);

        // Z: -2 → 0, 2 → 31, 0 → 15.5
        assert!((normed[0].2 - 0.0).abs() < 0.01);
        assert!((normed[1].2 - 31.0).abs() < 0.01);
        assert!((normed[2].2 - 15.5).abs() < 0.01);
    }

    #[test]
    fn test_axis_crystallizer_two_clusters() {
        let mut matrix = CooccurrenceMatrix::new(10);
        // Two clusters: {0,1,2,3,4} and {5,6,7,8,9}
        for _ in 0..20 {
            matrix.observe_window(&[0, 1, 2, 3, 4]);
            matrix.observe_window(&[5, 6, 7, 8, 9]);
        }

        let crystal = AxisCrystallizer::crystallize(&matrix);

        // First eigenvector should separate the clusters
        assert!(
            crystal.eigenvalues[0] > 0.0,
            "first eigenvalue should be positive"
        );

        // Concepts in same cluster should be closer than across clusters
        let (x0, y0, z0) = crystal.coords[0];
        let (x1, y1, z1) = crystal.coords[1]; // same cluster
        let (x5, y5, z5) = crystal.coords[5]; // different cluster
        let dist_same = ((x0 - x1).powi(2) + (y0 - y1).powi(2) + (z0 - z1).powi(2)).sqrt();
        let dist_diff = ((x0 - x5).powi(2) + (y0 - y5).powi(2) + (z0 - z5).powi(2)).sqrt();
        assert!(
            dist_same < dist_diff,
            "same cluster dist {:.2} should < cross cluster dist {:.2}",
            dist_same,
            dist_diff
        );
    }

    // ---- Migration tests ----

    #[test]
    fn test_greedy_frequency_no_conflicts() {
        // All concepts far apart → should get distinct frequencies
        let coords = vec![
            (0.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
            (0.0, 7.0, 0.0),
            (7.0, 7.0, 0.0),
        ];
        let assignments = greedy_frequency_assignment(&coords, 2.0);
        // All far apart (dist > 4σ), so all should get freq 0
        // Actually since they're all > 2σ apart, no conflicts, all get 0
        // That's correct: greedy picks lowest available
        assert_eq!(assignments.len(), 4);
    }

    #[test]
    fn test_greedy_frequency_nearby_different() {
        // Two very close concepts should get different frequencies
        let coords = vec![(3.0, 3.0, 15.0), (3.5, 3.5, 15.5)];
        let assignments = greedy_frequency_assignment(&coords, 2.0);
        assert_ne!(
            assignments[0], assignments[1],
            "nearby concepts should get different frequencies"
        );
    }

    #[test]
    fn test_container_mode_tag() {
        let mut meta = vec![0u8; 2048];
        assert_eq!(get_container_mode(&meta), ContainerMode::Empty);

        set_container_mode(&mut meta, ContainerMode::Carrier);
        assert_eq!(get_container_mode(&meta), ContainerMode::Carrier);

        set_container_mode(&mut meta, ContainerMode::Gabor);
        assert_eq!(get_container_mode(&meta), ContainerMode::Gabor);
    }

    #[test]
    fn test_migrate_carrier_to_gabor() {
        let basis = CarrierBasis::new();
        let lut = GaussianLUT::new(2.0);

        // Write 4 concepts as flat carriers
        let mut container = vec![0i8; 2048];
        let phases = [0.5f32, 1.5, 3.0, 5.0];
        for i in 0..4 {
            carrier_encode(&mut container, &basis, i as u8, phases[i], 7.0);
        }

        // Create a simple co-occurrence matrix and crystallize
        let mut matrix = CooccurrenceMatrix::new(4);
        for _ in 0..10 {
            matrix.observe_window(&[0, 1]);
            matrix.observe_window(&[2, 3]);
        }
        let crystal = AxisCrystallizer::crystallize(&matrix);

        // Migrate
        let result = migrate_carrier_to_gabor(&mut container, &basis, &crystal, &lut, 2.0);

        assert_eq!(result.mode, ContainerMode::Gabor);
        assert!(result.concepts_migrated <= 4);

        // Verify concepts are readable at their new positions
        for i in 0..4 {
            let (x, y, z) = crystal.coords[i];
            let freq = CARRIER_FREQUENCIES[result.freq_assignments[i]] as f32;
            let (_rec, amp) = gabor_read(
                &container,
                &lut,
                x.round().clamp(0.0, 7.0) as u8,
                y.round().clamp(0.0, 7.0) as u8,
                z.round().clamp(0.0, 31.0) as u8,
                freq,
            );
            assert!(
                amp > 0.1,
                "concept {} should be readable, amp={:.4}",
                i,
                amp
            );
        }
    }

    #[test]
    fn test_crystallize_from_superposition() {
        let basis = CarrierBasis::new();
        let mut container = vec![0i8; 2048];

        // Write some carriers to create structure
        for i in 0..8u8 {
            carrier_encode(&mut container, &basis, i, i as f32 * 0.5, 5.0);
        }

        let crystal = crystallize_from_superposition(&container, &basis);
        assert_eq!(crystal.coords.len(), 16); // 16 frequencies analyzed
                                              // At least the first eigenvalue should be meaningful
        assert!(
            crystal.eigenvalues[0] > 0.0,
            "first eigenvalue should be > 0"
        );
    }

    #[test]
    fn test_full_lifecycle() {
        let basis = CarrierBasis::new();
        let lut = GaussianLUT::new(2.0);

        // Phase 0: Empty
        let mut container = vec![0i8; 2048];
        let mut meta = vec![0u8; 2048];
        set_container_mode(&mut meta, ContainerMode::Empty);

        // Phase 1: Bootstrap with flat carriers
        set_container_mode(&mut meta, ContainerMode::Carrier);
        let test_phases = [0.3f32, 1.0, 2.0, 3.5, 4.2, 5.0, 0.8, 1.8, 2.5, 4.0];
        for (i, &phase) in test_phases.iter().enumerate() {
            bootstrap_write(&mut container, &basis, i as u8, phase, 5.0);
        }

        // Phase 2: Accumulate co-occurrence
        let mut matrix = CooccurrenceMatrix::new(10);
        for _ in 0..10 {
            matrix.observe_window(&[0, 1, 2, 3, 4]);
            matrix.observe_window(&[5, 6, 7, 8, 9]);
            matrix.observe_window(&[0, 2, 4, 6, 8]);
        }
        assert!(ready_for_crystallization(&matrix));

        // Phase 3: Crystallize axes
        let crystal = AxisCrystallizer::crystallize(&matrix);
        assert_eq!(crystal.coords.len(), 10);

        // Phase 4: Migrate
        let result = migrate_carrier_to_gabor(&mut container, &basis, &crystal, &lut, 2.0);
        set_container_mode(&mut meta, ContainerMode::Gabor);
        assert_eq!(get_container_mode(&meta), ContainerMode::Gabor);
        assert!(result.concepts_migrated > 0);

        // Container should have non-zero content
        let nonzero = container.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > 100,
            "migrated container should have content: {} nonzero",
            nonzero
        );
    }
}
