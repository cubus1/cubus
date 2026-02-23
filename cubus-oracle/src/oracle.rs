//! Three-temperature holographic oracle.
//!
//! One oracle per entity with hot/warm/cold tiers, overexposure-triggered flush,
//! and coefficient-as-canonical-storage.

use crate::linalg::{cholesky_solve, upsample_to_f32};
use crate::sweep::{generate_template, Base};
use rand::Rng;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// One holographic oracle per entity.
///
/// The oracle exists at three temperatures:
///   HOT:  materialized in float32, full thinking resolution
///   WARM: materialized in base-B signed, working memory
///   COLD: coefficients only, canonical storage form
///
/// Only ONE temperature is materialized at a time.
/// The coefficients persist across all materializations.
pub struct Oracle {
    /// The concept coefficients: how much of each concept this entity holds.
    /// This IS the canonical storage. Everything else materializes from this.
    pub coefficients: Vec<f32>,

    /// Concept IDs: which templates from the shared library are active.
    /// Parallel to coefficients: concept_ids[i] indexes into the library.
    pub concept_ids: Vec<u32>,

    /// Current temperature / materialization state.
    pub temperature: Temperature,

    /// Overexposure score from last check (0.0 = clean, 1.0+ = flush needed).
    pub overexposure: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Temperature {
    /// Coefficients only, nothing materialized. Compact storage form.
    Cold,
    /// Working memory materialized in signed base-B at D dimensions per axis.
    Warm { d: usize, axes: usize, base: Base },
    /// Full thinking resolution in float32 at D dimensions per axis.
    Hot { d: usize, axes: usize },
}

impl Default for Oracle {
    fn default() -> Self {
        Self::new()
    }
}

impl Oracle {
    /// Create a new empty oracle.
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            concept_ids: Vec::new(),
            temperature: Temperature::Cold,
            overexposure: 0.0,
        }
    }

    /// Storage size in bytes (cold form, coefficients + IDs only).
    pub fn cold_size(&self) -> usize {
        self.coefficients.len() * 4 + self.concept_ids.len() * 4
    }

    /// Number of active concepts.
    pub fn k(&self) -> usize {
        self.coefficients.len()
    }

    /// Add a concept to the oracle (cold operation — just store coefficient + id).
    pub fn add_concept(&mut self, concept_id: u32, coefficient: f32) {
        self.concept_ids.push(concept_id);
        self.coefficients.push(coefficient);
    }

    /// Materialize from cold to warm.
    ///
    /// Reconstruct: warm[axis][j] = Σ coefficients[i] × library.warm[concept_ids[i]][axis][j]
    /// Then quantize to base.
    pub fn materialize_warm(&self, library: &TemplateLibrary) -> MaterializedHolograph {
        let k = self.k();
        let n_axes = library.axes;
        let d = library.d_warm;
        let base = library.base_warm;

        let mut accum = vec![vec![0.0f32; d]; n_axes];

        for i in 0..k {
            let cid = self.concept_ids[i] as usize;
            let coeff = self.coefficients[i];
            for a in 0..n_axes {
                for j in 0..d {
                    accum[a][j] += coeff * library.warm[cid][a][j] as f32;
                }
            }
        }

        // Quantize
        let min = base.min_val() as f32;
        let max = base.max_val() as f32;
        let mut axes_buf = vec![vec![0i8; d]; n_axes];
        for a in 0..n_axes {
            for j in 0..d {
                axes_buf[a][j] = accum[a][j].round().clamp(min, max) as i8;
            }
        }

        MaterializedHolograph::Warm {
            axes: axes_buf,
            d,
            base,
        }
    }

    /// Materialize from cold to hot.
    ///
    /// Reconstruct: hot[axis][j] = Σ coefficients[i] × library.hot[concept_ids[i]][axis][j]
    /// No quantization — full float32 precision.
    pub fn materialize_hot(&self, library: &TemplateLibrary) -> MaterializedHolograph {
        let k = self.k();
        let n_axes = library.axes;
        let d = library.d_hot;

        let mut axes_buf = vec![vec![0.0f32; d]; n_axes];

        for i in 0..k {
            let cid = self.concept_ids[i] as usize;
            let coeff = self.coefficients[i];
            for a in 0..n_axes {
                for j in 0..d {
                    axes_buf[a][j] += coeff * library.hot[cid][a][j];
                }
            }
        }

        MaterializedHolograph::Hot { axes: axes_buf, d }
    }

    /// Surgical cool: extract updated coefficients from a hot materialization.
    ///
    /// This is the consolidation step after thinking:
    ///   1. Orthogonal project hot buffer against hot templates
    ///   2. Extract K exact coefficients
    ///   3. Update oracle coefficients (the canonical form)
    ///   4. Discard the hot buffer
    pub fn surgical_cool(&mut self, hot: &MaterializedHolograph, library: &TemplateLibrary) {
        let MaterializedHolograph::Hot { axes, d } = hot else {
            panic!("surgical_cool requires hot materialization");
        };

        let k = self.k();
        if k == 0 {
            return;
        }
        let n_axes = library.axes;

        // Per-axis coefficient extraction, then average across axes
        let mut coeff_sum = vec![0.0f64; k];

        for a in 0..n_axes {
            // W · b: dot products of templates against hot buffer
            let mut wb = vec![0.0f64; k];
            for i in 0..k {
                let cid = self.concept_ids[i] as usize;
                for j in 0..*d {
                    wb[i] += library.hot[cid][a][j] as f64 * axes[a][j] as f64;
                }
            }

            // Gram matrix
            let gram = self.compute_gram_hot(library, a);

            // Solve
            let recovered = cholesky_solve(&gram, &wb, k);

            for i in 0..k {
                coeff_sum[i] += recovered[i];
            }
        }

        // Average across axes
        for i in 0..k {
            self.coefficients[i] = (coeff_sum[i] / n_axes as f64) as f32;
        }

        self.temperature = Temperature::Cold;
    }

    /// Compute the Gram matrix for one axis at hot resolution.
    fn compute_gram_hot(&self, library: &TemplateLibrary, axis: usize) -> Vec<f64> {
        let k = self.k();
        let d = library.d_hot;
        let mut gram = vec![0.0f64; k * k];

        for i in 0..k {
            let cid_i = self.concept_ids[i] as usize;
            for j in i..k {
                let cid_j = self.concept_ids[j] as usize;
                let mut dot = 0.0f64;
                for p in 0..d {
                    dot += library.hot[cid_i][axis][p] as f64 * library.hot[cid_j][axis][p] as f64;
                }
                gram[i * k + j] = dot;
                gram[j * k + i] = dot;
            }
        }

        gram
    }

    /// Check overexposure level of a warm materialization.
    ///
    /// Returns 0.0 (clean) to 1.0+ (needs flush).
    ///
    /// Three signals:
    ///   1. Saturation: dimensions hitting clamp boundary
    ///   2. Energy overflow: total energy vs expected for K concepts
    ///   3. Zero deficit: loss of Auslöschung zeros (signed only)
    pub fn check_overexposure(&mut self, warm: &MaterializedHolograph) -> f32 {
        let MaterializedHolograph::Warm { axes, d, base } = warm else {
            return 0.0;
        };

        let max_val = base.max_val();
        let k = self.k().max(1) as f32;
        let d_f = *d as f32;

        let mut max_score = 0.0f32;

        for axis_buf in axes {
            // Signal 1: saturation ratio
            let saturated = axis_buf.iter().filter(|&&v| v.abs() >= max_val).count();
            let sat_ratio = saturated as f32 / d_f;

            // Signal 2: energy density
            let energy: f64 = axis_buf.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let expected = k as f64 * d_f as f64 * 0.5;
            let energy_ratio = (energy / expected.max(1.0)) as f32 - 1.0;

            // Signal 3: zero deficit (signed only)
            let zero_deficit = if base.has_cancellation() {
                let zeros = axis_buf.iter().filter(|&&v| v == 0).count();
                let expected_zeros = d_f * (1.0 / base.cardinality() as f32);
                1.0 - (zeros as f32 / expected_zeros.max(1.0))
            } else {
                0.0
            };

            let axis_score = sat_ratio.max(energy_ratio).max(zero_deficit).max(0.0);

            max_score = max_score.max(axis_score);
        }

        self.overexposure = max_score;
        max_score
    }

    /// The flush decision logic.
    pub fn flush_decision(&self) -> FlushAction {
        match self.overexposure {
            x if x < 0.5 => FlushAction::None,
            x if x < 0.8 => FlushAction::SoftFlush,
            x if x < 1.0 => FlushAction::HardFlush,
            _ => FlushAction::Emergency,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlushAction {
    /// Overexposure < 0.5: clean, keep working.
    None,
    /// Overexposure 0.5-0.8: snapshot coefficients as insurance.
    SoftFlush,
    /// Overexposure 0.8-1.0: full surgical cool.
    HardFlush,
    /// Overexposure > 1.0: warm is corrupted.
    Emergency,
}

// ---------------------------------------------------------------------------
// Template Library
// ---------------------------------------------------------------------------

/// Shared template library: the vocabulary of concepts.
///
/// Generated once, used by all oracles. Each concept has templates
/// at multiple resolutions for the three-temperature materializations.
pub struct TemplateLibrary {
    /// Number of concepts in the library.
    pub size: usize,

    /// Dimensionality at warm temperature.
    pub d_warm: usize,
    /// Dimensionality at hot temperature.
    pub d_hot: usize,

    /// Base for warm templates.
    pub base_warm: Base,

    /// Number of axes.
    pub axes: usize,

    /// Warm templates: [concept_id][axis] → Vec<i8> of length d_warm
    pub warm: Vec<Vec<Vec<i8>>>,

    /// Hot templates: [concept_id][axis] → Vec<f32> of length d_hot
    /// Generated by upsampling warm templates.
    pub hot: Vec<Vec<Vec<f32>>>,
}

impl TemplateLibrary {
    /// Generate a new library with random templates.
    pub fn generate(
        size: usize,
        d_warm: usize,
        d_hot: usize,
        base_warm: Base,
        axes: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut warm = Vec::with_capacity(size);
        let mut hot = Vec::with_capacity(size);

        for _ in 0..size {
            let mut w_axes = Vec::with_capacity(axes);
            let mut h_axes = Vec::with_capacity(axes);
            for _ in 0..axes {
                // Generate warm template
                let w = generate_template(d_warm, base_warm, rng);
                // Upsample to hot (coherent)
                let h = upsample_to_f32(&w, d_hot);
                w_axes.push(w);
                h_axes.push(h);
            }
            warm.push(w_axes);
            hot.push(h_axes);
        }

        Self {
            size,
            d_warm,
            d_hot,
            base_warm,
            axes,
            warm,
            hot,
        }
    }
}

// ---------------------------------------------------------------------------
// Materialized Holograph
// ---------------------------------------------------------------------------

/// Materialized holograph at a specific temperature.
///
/// This is the WORKING BUFFER, not the oracle itself.
/// Created on demand from oracle coefficients + library templates.
/// Discarded after thinking is done and coefficients are updated.
pub enum MaterializedHolograph {
    Warm {
        /// Per-axis buffers: axes × d_warm signed integers
        axes: Vec<Vec<i8>>,
        d: usize,
        base: Base,
    },
    Hot {
        /// Per-axis buffers: axes × d_hot floats
        axes: Vec<Vec<f32>>,
        d: usize,
    },
}

impl MaterializedHolograph {
    /// Add a concept to the materialized holograph.
    pub fn add_concept(&mut self, concept_id: usize, coefficient: f32, library: &TemplateLibrary) {
        match self {
            MaterializedHolograph::Warm { axes, d, base } => {
                let max = base.max_val();
                let min = base.min_val();
                for a in 0..axes.len() {
                    for j in 0..*d {
                        let current = axes[a][j] as f32;
                        let update = coefficient * library.warm[concept_id][a][j] as f32;
                        axes[a][j] = (current + update).round().clamp(min as f32, max as f32) as i8;
                    }
                }
            }
            MaterializedHolograph::Hot { axes, d } => {
                for a in 0..axes.len() {
                    for j in 0..*d {
                        axes[a][j] += coefficient * library.hot[concept_id][a][j];
                    }
                }
            }
        }
    }

    /// Hebbian update: strengthen association between two concepts.
    pub fn hebbian_update(
        &mut self,
        concept_a: usize,
        concept_b: usize,
        learning_rate: f32,
        library: &TemplateLibrary,
    ) {
        match self {
            MaterializedHolograph::Warm { axes, d, base } => {
                let max = base.max_val();
                let min = base.min_val();
                for a in 0..axes.len() {
                    for j in 0..*d {
                        let ta = library.warm[concept_a][a][j] as f32;
                        let tb = library.warm[concept_b][a][j] as f32;
                        let interference = ta * tb * learning_rate;
                        let current = axes[a][j] as f32;
                        axes[a][j] = (current + interference)
                            .round()
                            .clamp(min as f32, max as f32)
                            as i8;
                    }
                }
            }
            MaterializedHolograph::Hot { axes, d } => {
                for a in 0..axes.len() {
                    for j in 0..*d {
                        let ta = library.hot[concept_a][a][j];
                        let tb = library.hot[concept_b][a][j];
                        axes[a][j] += ta * tb * learning_rate;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::downsample_to_base;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // Use small dimensions for fast tests
    const TEST_D_WARM: usize = 256;
    const TEST_D_HOT: usize = 512;
    const TEST_AXES: usize = 2;
    const TEST_BASE: Base = Base::Signed(5);
    const TEST_LIB_SIZE: usize = 20;

    fn test_library(rng: &mut StdRng) -> TemplateLibrary {
        TemplateLibrary::generate(
            TEST_LIB_SIZE,
            TEST_D_WARM,
            TEST_D_HOT,
            TEST_BASE,
            TEST_AXES,
            rng,
        )
    }

    // -- Oracle creation --

    #[test]
    fn test_new_oracle_cold_zero() {
        let oracle = Oracle::new();
        assert_eq!(oracle.temperature, Temperature::Cold);
        assert_eq!(oracle.k(), 0);
        assert_eq!(oracle.cold_size(), 0);
        assert_eq!(oracle.overexposure, 0.0);
    }

    // -- Add concept --

    #[test]
    fn test_add_concept() {
        let mut oracle = Oracle::new();
        oracle.add_concept(3, 0.75);
        oracle.add_concept(7, -0.5);
        assert_eq!(oracle.k(), 2);
        assert_eq!(oracle.concept_ids, vec![3, 7]);
        assert!((oracle.coefficients[0] - 0.75).abs() < 1e-6);
        assert!((oracle.coefficients[1] - (-0.5)).abs() < 1e-6);
        assert_eq!(oracle.cold_size(), 2 * 4 + 2 * 4); // 2 f32 + 2 u32
    }

    // -- Materialize warm --

    #[test]
    fn test_materialize_warm_reconstruction() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 1.0);
        oracle.add_concept(1, -0.5);

        let warm = oracle.materialize_warm(&lib);
        match &warm {
            MaterializedHolograph::Warm { axes, d, base: _ } => {
                assert_eq!(*d, TEST_D_WARM);
                assert_eq!(axes.len(), TEST_AXES);
                // Check that the buffer is non-trivial
                let nonzero = axes[0].iter().filter(|&&v| v != 0).count();
                assert!(nonzero > 0, "warm buffer should have nonzero elements");
            }
            _ => panic!("expected Warm"),
        }
    }

    // -- Materialize hot --

    #[test]
    fn test_materialize_hot_reconstruction() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 1.0);

        let hot = oracle.materialize_hot(&lib);
        match &hot {
            MaterializedHolograph::Hot { axes, d } => {
                assert_eq!(*d, TEST_D_HOT);
                assert_eq!(axes.len(), TEST_AXES);
                let nonzero = axes[0].iter().filter(|&&v| v.abs() > 1e-10).count();
                assert!(nonzero > 0, "hot buffer should have nonzero elements");
            }
            _ => panic!("expected Hot"),
        }
    }

    // -- Surgical cool --

    #[test]
    fn test_surgical_cool_recovers_coefficients() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.8);
        oracle.add_concept(1, -0.3);
        oracle.add_concept(2, 0.5);

        let original_coeffs = oracle.coefficients.clone();

        // Hot materialization preserves exact coefficients (no quantization)
        let hot = oracle.materialize_hot(&lib);
        oracle.surgical_cool(&hot, &lib);

        assert_eq!(oracle.temperature, Temperature::Cold);
        for i in 0..oracle.k() {
            let err = (oracle.coefficients[i] - original_coeffs[i]).abs();
            assert!(
                err < 0.01,
                "coefficient[{}]: original={}, recovered={}, err={}",
                i,
                original_coeffs[i],
                oracle.coefficients[i],
                err
            );
        }
    }

    // -- Round trip: cold → hot → surgical cool → cold --

    #[test]
    fn test_round_trip_hot_exact() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        for i in 0..5 {
            oracle.add_concept(i as u32, rng.gen_range(-1.0f32..1.0f32));
        }
        let original_coeffs = oracle.coefficients.clone();

        let hot = oracle.materialize_hot(&lib);
        oracle.surgical_cool(&hot, &lib);

        for i in 0..oracle.k() {
            let err = (oracle.coefficients[i] - original_coeffs[i]).abs();
            assert!(err < 0.01, "hot round trip coefficient[{}]: err={}", i, err);
        }
    }

    // -- Round trip: cold → warm → add concepts → surgical cool --

    #[test]
    fn test_round_trip_warm_with_add() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.5);
        oracle.add_concept(1, 0.3);

        // Materialize to hot, add a concept, then cool
        let mut hot = oracle.materialize_hot(&lib);
        hot.add_concept(2, 0.7, &lib);

        // Update oracle to know about concept 2
        oracle.add_concept(2, 0.0); // placeholder

        oracle.surgical_cool(&hot, &lib);

        // Concept 2 should now have a nonzero coefficient
        assert!(
            oracle.coefficients[2].abs() > 0.01,
            "added concept should have nonzero coefficient after cool, got {}",
            oracle.coefficients[2]
        );
    }

    // -- Overexposure --

    #[test]
    fn test_overexposure_empty() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);
        let mut oracle = Oracle::new();
        let warm = oracle.materialize_warm(&lib);
        let score = oracle.check_overexposure(&warm);
        // Empty oracle: no energy, no saturation
        // But the energy ratio = (0 / expected) - 1 = -1, clamped to 0
        assert!(score >= 0.0, "overexposure should be >= 0, got {}", score);
    }

    #[test]
    fn test_overexposure_few_concepts_low() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        for i in 0..5 {
            oracle.add_concept(i as u32, 0.3);
        }

        let warm = oracle.materialize_warm(&lib);
        let score = oracle.check_overexposure(&warm);
        assert!(
            score < 0.8,
            "5 concepts at D={} should be comfortable, got {}",
            TEST_D_WARM,
            score
        );
    }

    #[test]
    fn test_overexposure_many_concepts_high() {
        let mut rng = seeded_rng();
        // Use a very small D to force overexposure
        let lib = TemplateLibrary::generate(20, 32, 64, Base::Signed(3), 1, &mut rng);

        let mut oracle = Oracle::new();
        for i in 0..18 {
            oracle.add_concept(i as u32, 1.0); // big coefficients, small D
        }

        let warm = oracle.materialize_warm(&lib);
        let score = oracle.check_overexposure(&warm);
        assert!(
            score > 0.3,
            "18 concepts at D=32 should be cramped, got {}",
            score
        );
    }

    // -- Signed cancellation reduces overexposure --

    #[test]
    fn test_signed_cancellation_reduces_overexposure() {
        let mut rng = seeded_rng();
        let lib = TemplateLibrary::generate(10, 64, 128, Base::Signed(5), 1, &mut rng);

        // Positive coefficients only
        let mut oracle_pos = Oracle::new();
        for i in 0..8 {
            oracle_pos.add_concept(i as u32, 1.0);
        }
        let warm_pos = oracle_pos.materialize_warm(&lib);
        let score_pos = oracle_pos.check_overexposure(&warm_pos);

        // Mix of positive and negative (cancellation should help)
        let mut oracle_mix = Oracle::new();
        for i in 0..8 {
            let coeff = if i % 2 == 0 { 1.0 } else { -1.0 };
            oracle_mix.add_concept(i as u32, coeff);
        }
        let warm_mix = oracle_mix.materialize_warm(&lib);
        let score_mix = oracle_mix.check_overexposure(&warm_mix);

        // Mixed should have lower or equal overexposure due to cancellation
        assert!(
            score_mix <= score_pos + 0.1,
            "mixed should have ≤ overexposure: pos={}, mix={}",
            score_pos,
            score_mix
        );
    }

    // -- Flush decision transitions --

    #[test]
    fn test_flush_decision_transitions() {
        let mut oracle = Oracle::new();

        oracle.overexposure = 0.0;
        assert_eq!(oracle.flush_decision(), FlushAction::None);

        oracle.overexposure = 0.3;
        assert_eq!(oracle.flush_decision(), FlushAction::None);

        oracle.overexposure = 0.6;
        assert_eq!(oracle.flush_decision(), FlushAction::SoftFlush);

        oracle.overexposure = 0.9;
        assert_eq!(oracle.flush_decision(), FlushAction::HardFlush);

        oracle.overexposure = 1.5;
        assert_eq!(oracle.flush_decision(), FlushAction::Emergency);
    }

    // -- Template library --

    #[test]
    fn test_library_warm_hot_coherent() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        assert_eq!(lib.size, TEST_LIB_SIZE);
        assert_eq!(lib.warm.len(), TEST_LIB_SIZE);
        assert_eq!(lib.hot.len(), TEST_LIB_SIZE);

        // Each concept should have TEST_AXES axes
        assert_eq!(lib.warm[0].len(), TEST_AXES);
        assert_eq!(lib.hot[0].len(), TEST_AXES);

        // Warm template dimensionality
        assert_eq!(lib.warm[0][0].len(), TEST_D_WARM);
        // Hot template dimensionality
        assert_eq!(lib.hot[0][0].len(), TEST_D_HOT);
    }

    #[test]
    fn test_warm_hot_upsample_coherence() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        // Downsample hot back to warm and compare
        let warm_original = &lib.warm[0][0];
        let hot = &lib.hot[0][0];
        let warm_reconstructed = downsample_to_base(hot, TEST_D_WARM, TEST_BASE);

        let mut max_err = 0i8;
        for i in 0..TEST_D_WARM {
            let err = (warm_original[i] - warm_reconstructed[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err <= 1,
            "warm → hot → warm should be within quantization error, got max_err={}",
            max_err
        );
    }

    #[test]
    fn test_library_memory_footprint() {
        let mut rng = seeded_rng();
        let lib = TemplateLibrary::generate(200, 16384, 65536, Base::Signed(5), 3, &mut rng);
        // Warm: 200 × 3 × 16384 = 9,830,400 bytes
        let warm_bytes = 200 * 3 * 16384;
        assert_eq!(
            lib.warm.len() * lib.warm[0].len() * lib.warm[0][0].len(),
            warm_bytes
        );
        // Hot: 200 × 3 × 65536 × 4 = 157,286,400 bytes
        let hot_elements = 200 * 3 * 65536;
        assert_eq!(
            lib.hot.len() * lib.hot[0].len() * lib.hot[0][0].len(),
            hot_elements
        );
    }

    // -- Hebbian learning --

    #[test]
    fn test_hebbian_update_warm() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.5);
        oracle.add_concept(1, 0.5);

        let mut warm = oracle.materialize_warm(&lib);

        // Take a snapshot of energy before
        let energy_before: f64 = match &warm {
            MaterializedHolograph::Warm { axes, .. } => {
                axes[0].iter().map(|&v| (v as f64) * (v as f64)).sum()
            }
            _ => 0.0,
        };

        // Hebbian update
        warm.hebbian_update(0, 1, 0.5, &lib);

        let energy_after: f64 = match &warm {
            MaterializedHolograph::Warm { axes, .. } => {
                axes[0].iter().map(|&v| (v as f64) * (v as f64)).sum()
            }
            _ => 0.0,
        };

        // Energy should change after Hebbian update
        assert!(
            (energy_after - energy_before).abs() > 0.1,
            "Hebbian should change energy: before={}, after={}",
            energy_before,
            energy_after
        );
    }

    #[test]
    fn test_hebbian_update_hot() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.5);
        oracle.add_concept(1, 0.5);

        let mut hot = oracle.materialize_hot(&lib);

        // Hebbian update
        hot.hebbian_update(0, 1, 0.5, &lib);

        // Surgical cool and check coefficient changed
        oracle.surgical_cool(&hot, &lib);

        // Coefficients should have changed from the Hebbian update
        assert!(
            oracle.coefficients.iter().any(|&c| (c - 0.5).abs() > 0.01),
            "Hebbian + cool should change coefficients"
        );
    }

    // -- Full lifecycle --

    #[test]
    fn test_full_lifecycle() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        // 1. Create oracle
        let mut oracle = Oracle::new();
        assert_eq!(oracle.temperature, Temperature::Cold);

        // 2. Add concepts
        for i in 0..10 {
            oracle.add_concept(i as u32, rng.gen_range(-0.5f32..0.5f32));
        }
        let original = oracle.coefficients.clone();

        // 3. Materialize warm, check overexposure
        let warm = oracle.materialize_warm(&lib);
        let score = oracle.check_overexposure(&warm);
        let _action = oracle.flush_decision();

        // 4. Should be comfortable at D=256 with 10 concepts
        assert!(score < 1.0, "10 concepts should not trigger emergency");

        // 5. Materialize hot for thinking
        let hot = oracle.materialize_hot(&lib);

        // 6. Surgical cool: extract coefficients
        oracle.surgical_cool(&hot, &lib);
        assert_eq!(oracle.temperature, Temperature::Cold);

        // 7. Verify coefficients survived
        for i in 0..oracle.k() {
            let err = (oracle.coefficients[i] - original[i]).abs();
            assert!(
                err < 0.05,
                "lifecycle coefficient[{}]: original={}, recovered={}, err={}",
                i,
                original[i],
                oracle.coefficients[i],
                err
            );
        }
    }

    // -- Learning round trip --

    #[test]
    fn test_learning_round_trip() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.5);
        oracle.add_concept(1, 0.5);
        oracle.add_concept(2, -0.3);

        // Hot → Hebbian → cool
        let mut hot = oracle.materialize_hot(&lib);
        hot.hebbian_update(0, 1, 0.3, &lib);
        oracle.surgical_cool(&hot, &lib);

        // Re-materialize to verify the learning took effect
        let warm = oracle.materialize_warm(&lib);

        // The buffer should be non-trivial
        match &warm {
            MaterializedHolograph::Warm { axes, .. } => {
                let nonzero = axes[0].iter().filter(|&&v| v != 0).count();
                assert!(nonzero > 0, "warm after learning should be non-trivial");
            }
            _ => panic!("expected Warm"),
        }
    }

    // -- Anti-Hebbian --

    #[test]
    fn test_anti_hebbian_weakens() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.8);
        oracle.add_concept(1, 0.6);

        // Hot → anti-Hebbian (negative learning rate) → cool
        let mut hot = oracle.materialize_hot(&lib);
        hot.hebbian_update(0, 1, -0.5, &lib); // negative = anti-Hebbian
        oracle.surgical_cool(&hot, &lib);

        // At least one coefficient should have changed significantly
        let changed = oracle
            .coefficients
            .iter()
            .zip([0.8f32, 0.6].iter())
            .any(|(&c, &orig)| (c - orig).abs() > 0.01);
        assert!(
            changed,
            "anti-Hebbian should change coefficients: {:?}",
            oracle.coefficients
        );
    }

    // -- Mixed operations --

    #[test]
    fn test_mixed_add_hebbian_anti_hebbian() {
        let mut rng = seeded_rng();
        let lib = test_library(&mut rng);

        let mut oracle = Oracle::new();
        oracle.add_concept(0, 0.5);
        oracle.add_concept(1, 0.5);
        oracle.add_concept(2, 0.0); // placeholder for new concept

        let mut hot = oracle.materialize_hot(&lib);

        // Add concept 2 with coefficient 0.6
        hot.add_concept(2, 0.6, &lib);
        // Hebbian between 0 and 1
        hot.hebbian_update(0, 1, 0.3, &lib);
        // Anti-Hebbian between 1 and 2
        hot.hebbian_update(1, 2, -0.2, &lib);

        oracle.surgical_cool(&hot, &lib);

        // Concept 2 should now have a significant coefficient
        assert!(
            oracle.coefficients[2].abs() > 0.05,
            "concept 2 should have been added, got {}",
            oracle.coefficients[2]
        );
    }
}
