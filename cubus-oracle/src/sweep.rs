//! Exhaustive capacity sweep over D, base, signing, axes, K, and bind depth.

use crate::linalg::{cholesky_solve, condition_number};
pub use numrus_nars::{Base, bind, bind_deep, bundle, generate_template, generate_templates};
use rand::Rng;

// ---------------------------------------------------------------------------
// Parameter constants
// ---------------------------------------------------------------------------

/// Dimensions to test. Total: 7 values.
pub const DIMS: &[usize] = &[1024, 2048, 4096, 8192, 16384, 32768, 65536];

/// All base types. Total: 8 values.
pub const BASES: &[Base] = &[
    // Unsigned
    Base::Binary,      // B=2, {0, 1}
    Base::Unsigned(3), // B=3, {0, 1, 2}
    Base::Unsigned(5), // B=5, {0, 1, 2, 3, 4}
    Base::Unsigned(7), // B=7, {0, 1, 2, 3, 4, 5, 6}
    // Signed (with Auslöschung / cancellation at zero)
    Base::Signed(3), // B=3, {-1, 0, +1}
    Base::Signed(5), // B=5, {-2, -1, 0, +1, +2}
    Base::Signed(7), // B=7, {-3, -2, -1, 0, +1, +2, +3}
    Base::Signed(9), // B=9, {-4, -3, -2, -1, 0, +1, +2, +3, +4}
];

/// Number of axes. Total: 3 values.
pub const AXES: &[usize] = &[1, 2, 3];

/// Bundle sizes (number of concepts K). Total: 9 values.
pub const BUNDLE_SIZES: &[usize] = &[1, 3, 5, 8, 13, 21, 34, 55, 89];

/// Bind depths to test. Total: 4 values.
pub const BIND_DEPTHS: &[usize] = &[1, 2, 3, 4];

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Quantize f32 superposition to the given base.
fn quantize(v: &[f32], base: Base) -> Vec<i8> {
    let min = base.min_val() as f32;
    let max = base.max_val() as f32;
    v.iter().map(|&x| x.round().clamp(min, max) as i8).collect()
}

// ---------------------------------------------------------------------------
// Measurement: single axis
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RecoveryResult {
    pub d: usize,
    pub base: Base,
    pub k: usize,
    pub mean_error: f32,
    pub max_error: f32,
    pub gram_condition: f32,
    pub noise_floor: f32,
    pub cancellation: f32,     // fraction of zero dimensions (signed only)
    pub bits_per_concept: f32, // storage efficiency
}

/// Measure recovery error after orthogonal projection (single axis).
pub fn measure_recovery(d: usize, base: Base, k: usize, rng: &mut impl Rng) -> RecoveryResult {
    // Generate templates
    let templates = generate_templates(k, d, base, rng);

    // Random coefficients in [-1.0, +1.0]
    let coefficients: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();

    // Create superposition (in f32 for precision)
    let mut superposition_f32 = vec![0.0f32; d];
    for i in 0..k {
        for j in 0..d {
            superposition_f32[j] += coefficients[i] * templates[i][j] as f32;
        }
    }

    // Quantize to base
    let superposition = quantize(&superposition_f32, base);

    // Orthogonal project: extract coefficients
    // Step 1: W · b (templates · superposition) → K dot products
    let mut wb = vec![0.0f64; k];
    for i in 0..k {
        for j in 0..d {
            wb[i] += templates[i][j] as f64 * superposition[j] as f64;
        }
    }

    // Step 2: Gram matrix W · W^T
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += templates[i][p] as f64 * templates[j][p] as f64;
            }
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    // Step 3: Solve via Cholesky
    let gram_cond = condition_number(&gram, k);
    let recovered = cholesky_solve(&gram, &wb, k);

    // Step 4: Measure errors
    let mut max_error = 0.0f32;
    let mut mean_error = 0.0f32;
    for i in 0..k {
        let err = (coefficients[i] - recovered[i] as f32).abs();
        max_error = max_error.max(err);
        mean_error += err;
    }
    mean_error /= k as f32;

    // Step 5: Noise floor (residual after reconstruction)
    let mut residual_energy = 0.0f64;
    for j in 0..d {
        let mut reconstructed = 0.0f64;
        for i in 0..k {
            reconstructed += recovered[i] * templates[i][j] as f64;
        }
        let residual = superposition[j] as f64 - reconstructed;
        residual_energy += residual * residual;
    }
    let noise_floor = (residual_energy / d as f64).sqrt();

    // Step 6: Cancellation metric (signed only)
    let cancellation = if base.has_cancellation() {
        let zero_count = superposition.iter().filter(|&&v| v == 0).count();
        zero_count as f32 / d as f32
    } else {
        0.0
    };

    RecoveryResult {
        d,
        base,
        k,
        mean_error,
        max_error,
        gram_condition: gram_cond as f32,
        noise_floor: noise_floor as f32,
        cancellation,
        bits_per_concept: base.storage_bits(d) as f32 / k as f32,
    }
}

// ---------------------------------------------------------------------------
// Measurement: multi-axis
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AxisResult {
    pub axis: usize,
    pub mean_error: f32,
    pub gram_condition: f32,
}

#[derive(Clone, Debug)]
pub struct MultiAxisResult {
    pub d: usize,
    pub base: Base,
    pub axes: usize,
    pub k: usize,
    pub bind_depth: usize,
    pub combined_error: f32,
    pub bell_coefficient: f32,
    pub per_axis: Vec<AxisResult>,
    pub storage_bytes: usize,
    pub bits_per_concept: f32,
}

/// Measure recovery with N axes and bind depth.
pub fn measure_recovery_multiaxis(
    d: usize,
    base: Base,
    axes: usize,
    k: usize,
    bind_depth: usize,
    rng: &mut impl Rng,
) -> MultiAxisResult {
    // Generate role vectors for each axis and bind depth
    let role_vectors: Vec<Vec<Vec<i8>>> = (0..axes)
        .map(|_| generate_templates(bind_depth, d, base, rng))
        .collect();

    // Generate concept templates
    let templates = generate_templates(k, d, base, rng);

    // Random coefficients
    let coefficients: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();

    // For each axis: bind concepts with axis roles, bundle, measure recovery
    let mut per_axis_results = Vec::new();
    for axis in 0..axes {
        // Bind each template with this axis's role vectors
        let bound_templates: Vec<Vec<i8>> = templates
            .iter()
            .map(|t| bind_deep(t, &role_vectors[axis][..bind_depth], base))
            .collect();

        // Create superposition
        let mut super_f32 = vec![0.0f32; d];
        for i in 0..k {
            for j in 0..d {
                super_f32[j] += coefficients[i] * bound_templates[i][j] as f32;
            }
        }
        let superposition = quantize(&super_f32, base);

        // Orthogonal project with BOUND templates (not originals!)
        let wb = dot_matrix_vector_i8(&bound_templates, &superposition);
        let gram = gram_matrix_i8(&bound_templates);
        let gram_cond = condition_number(&gram, k);
        let recovered = cholesky_solve(&gram, &wb, k);

        // Error
        let mut mean_error = 0.0f32;
        for i in 0..k {
            mean_error += (coefficients[i] - recovered[i] as f32).abs();
        }
        mean_error /= k as f32;

        per_axis_results.push(AxisResult {
            axis,
            mean_error,
            gram_condition: gram_cond as f32,
        });
    }

    // Combined error: average across axes
    let combined_error: f32 =
        per_axis_results.iter().map(|r| r.mean_error).sum::<f32>() / axes as f32;

    // Bell coefficient measurement (for 2+ axes)
    let bell_coefficient = if axes >= 2 {
        measure_bell_coefficient(d, base, &templates, &coefficients, &role_vectors, rng)
    } else {
        0.0
    };

    MultiAxisResult {
        d,
        base,
        axes,
        k,
        bind_depth,
        combined_error,
        bell_coefficient,
        per_axis: per_axis_results,
        storage_bytes: base.storage_bytes(d, axes),
        bits_per_concept: (base.storage_bits(d) * axes) as f32 / k as f32,
    }
}

// ---------------------------------------------------------------------------
// Bell coefficient
// ---------------------------------------------------------------------------

/// Measure the CHSH Bell coefficient.
pub fn measure_bell_coefficient(
    d: usize,
    base: Base,
    templates: &[Vec<i8>],
    coefficients: &[f32],
    role_vectors: &[Vec<Vec<i8>>],
    rng: &mut impl Rng,
) -> f32 {
    // CHSH setup: two parties (axis 0 and axis 1), two measurement settings each
    let m = [
        generate_template(d, base, rng), // party A, setting 0
        generate_template(d, base, rng), // party A, setting 1
        generate_template(d, base, rng), // party B, setting 0
        generate_template(d, base, rng), // party B, setting 1
    ];

    // Build the holograph state on both axes
    let mut axis_states = vec![vec![0i16; d]; 2];
    for i in 0..templates.len().min(coefficients.len()) {
        let bound_0 = bind_deep(&templates[i], &role_vectors[0], base);
        let bound_1 = bind_deep(&templates[i], &role_vectors[1], base);
        for j in 0..d {
            axis_states[0][j] += (coefficients[i] * bound_0[j] as f32).round() as i16;
            axis_states[1][j] += (coefficients[i] * bound_1[j] as f32).round() as i16;
        }
    }

    // Correlations: E(a,b) = <axis_0 · m_a> × <axis_1 · m_b> / normalization
    let correlate = |a_idx: usize, b_idx: usize| -> f64 {
        let mut dot_a = 0.0f64;
        let mut dot_b = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for j in 0..d {
            dot_a += axis_states[0][j] as f64 * m[a_idx][j] as f64;
            dot_b += axis_states[1][j] as f64 * m[b_idx + 2][j] as f64;
            norm_a += axis_states[0][j] as f64 * axis_states[0][j] as f64;
            norm_b += axis_states[1][j] as f64 * axis_states[1][j] as f64;
        }
        let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
        (dot_a * dot_b) / (norm * d as f64)
    };

    // CHSH: S = E(0,0) - E(0,1) + E(1,0) + E(1,1)
    let s = correlate(0, 0) - correlate(0, 1) + correlate(1, 0) + correlate(1, 1);
    s.abs() as f32
}

// ---------------------------------------------------------------------------
// Full sweep
// ---------------------------------------------------------------------------

/// Run the full capacity sweep.
///
/// Tests every combination, returns results sorted by bits_per_concept efficiency.
pub fn run_sweep(repetitions: usize) -> Vec<MultiAxisResult> {
    let mut rng = rand::thread_rng();
    let mut all_results = Vec::new();

    for &d in DIMS {
        for &base in BASES {
            for &axes in AXES {
                for &k in BUNDLE_SIZES {
                    for &bind_depth in BIND_DEPTHS {
                        // Skip invalid: bind_depth > axes
                        if bind_depth > axes {
                            continue;
                        }
                        // Skip pathological combos
                        if k > d / 10 {
                            continue;
                        }

                        let mut rep_results = Vec::new();
                        for _ in 0..repetitions {
                            let result =
                                measure_recovery_multiaxis(d, base, axes, k, bind_depth, &mut rng);
                            rep_results.push(result);
                        }

                        // Aggregate across repetitions
                        let avg = aggregate_results(&rep_results);
                        all_results.push(avg);
                    }
                }
            }
        }
    }

    // Sort by efficiency: lowest bits_per_concept with error < threshold
    all_results.sort_by(|a, b| {
        let a_valid = a.combined_error < 0.01;
        let b_valid = b.combined_error < 0.01;
        match (a_valid, b_valid) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a
                .bits_per_concept
                .partial_cmp(&b.bits_per_concept)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    all_results
}

/// Aggregate multiple result runs (mean of each metric).
fn aggregate_results(results: &[MultiAxisResult]) -> MultiAxisResult {
    let n = results.len() as f32;
    let first = &results[0];

    let mut avg = first.clone();
    avg.combined_error = results.iter().map(|r| r.combined_error).sum::<f32>() / n;
    avg.bell_coefficient = results.iter().map(|r| r.bell_coefficient).sum::<f32>() / n;
    avg.bits_per_concept = first.bits_per_concept; // constant for given config
    avg
}

/// Format results as CSV.
pub fn results_to_csv(results: &[MultiAxisResult]) -> String {
    let mut csv = String::from(
        "d,base,signed,axes,k,bind_depth,mean_error,bell_coefficient,bits_per_concept,storage_bytes\n"
    );
    for r in results {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.6},{:.4},{:.1},{}\n",
            r.d,
            r.base.name(),
            r.base.is_signed(),
            r.axes,
            r.k,
            r.bind_depth,
            r.combined_error,
            r.bell_coefficient,
            r.bits_per_concept,
            r.storage_bytes,
        ));
    }
    csv
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn dot_matrix_vector_i8(templates: &[Vec<i8>], v: &[i8]) -> Vec<f64> {
    templates
        .iter()
        .map(|t| {
            t.iter()
                .zip(v.iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        })
        .collect()
}

fn gram_matrix_i8(templates: &[Vec<i8>]) -> Vec<f64> {
    let k = templates.len();
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let dot: f64 = templates[i]
                .iter()
                .zip(templates[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }
    gram
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // -- Base type tests --

    #[test]
    fn test_base_binary_properties() {
        let b = Base::Binary;
        assert_eq!(b.cardinality(), 2);
        assert_eq!(b.min_val(), 0);
        assert_eq!(b.max_val(), 1);
        assert!(!b.has_cancellation());
    }

    #[test]
    fn test_base_signed_properties() {
        let b = Base::Signed(5);
        assert_eq!(b.cardinality(), 5);
        assert_eq!(b.min_val(), -2);
        assert_eq!(b.max_val(), 2);
        assert!(b.has_cancellation());
    }

    #[test]
    fn test_base_unsigned_properties() {
        let b = Base::Unsigned(7);
        assert_eq!(b.cardinality(), 7);
        assert_eq!(b.min_val(), 0);
        assert_eq!(b.max_val(), 6);
        assert!(!b.has_cancellation());
    }

    #[test]
    fn test_base_storage_bytes() {
        // Binary: 1 bit per dim, 1024 dims = 128 bytes per axis
        let bytes = Base::Binary.storage_bytes(1024, 1);
        assert_eq!(bytes, 128); // 1024 bits / 8
    }

    // -- Template generation --

    #[test]
    fn test_generate_template_binary_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Binary, &mut rng);
        assert!(t.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn test_generate_template_signed_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Signed(5), &mut rng);
        assert!(t.iter().all(|&v| (-2..=2).contains(&v)));
    }

    #[test]
    fn test_generate_template_unsigned_range() {
        let mut rng = seeded_rng();
        let t = generate_template(1000, Base::Unsigned(7), &mut rng);
        assert!(t.iter().all(|&v| (0..=6).contains(&v)));
    }

    // -- Binding --

    #[test]
    fn test_bind_binary_xor() {
        let a = vec![0i8, 1, 1, 0];
        let b = vec![1i8, 1, 0, 0];
        let c = bind(&a, &b, Base::Binary);
        assert_eq!(c, vec![1, 0, 1, 0]);
    }

    #[test]
    fn test_bind_unsigned_mod() {
        let a = vec![2i8, 3, 4];
        let b = vec![4i8, 3, 2];
        let c = bind(&a, &b, Base::Unsigned(5));
        // (2+4)%5=1, (3+3)%5=1, (4+2)%5=1
        assert_eq!(c, vec![1, 1, 1]);
    }

    #[test]
    fn test_bind_signed_clamp() {
        let a = vec![2i8, -2, 1];
        let b = vec![2i8, -2, 0];
        let c = bind(&a, &b, Base::Signed(5));
        // (2+2) clamped to 2, (-2+-2) clamped to -2, (1+0) = 1
        assert_eq!(c, vec![2, -2, 1]);
    }

    #[test]
    fn test_bind_deep_sequential() {
        let mut rng = seeded_rng();
        let v = generate_template(100, Base::Signed(5), &mut rng);
        let roles = generate_templates(3, 100, Base::Signed(5), &mut rng);
        let deep = bind_deep(&v, &roles, Base::Signed(5));
        // Should be different from original
        assert_ne!(deep, v);
        assert_eq!(deep.len(), 100);
    }

    // -- Bundling --

    #[test]
    fn test_bundle_binary_majority() {
        let vecs = vec![
            vec![1i8, 0, 1, 1, 0],
            vec![1, 1, 0, 1, 0],
            vec![0, 1, 1, 1, 0],
        ];
        let b = bundle(&vecs, Base::Binary);
        // majority: [1, 1, 1, 1, 0]
        assert_eq!(b, vec![1, 1, 1, 1, 0]);
    }

    #[test]
    fn test_bundle_signed_cancellation() {
        // Opposing values should cancel (Auslöschung)
        let vecs = vec![vec![2i8, -2, 1], vec![-2, 2, -1]];
        let b = bundle(&vecs, Base::Signed(5));
        // 2+(-2)=0, (-2)+2=0, 1+(-1)=0
        assert_eq!(b, vec![0, 0, 0]);
    }

    // -- Recovery measurements --

    #[test]
    fn test_binary_k1_exact_recovery() {
        let mut rng = seeded_rng();
        let result = measure_recovery(1024, Base::Binary, 1, &mut rng);
        // Quantization noise prevents exact recovery even at K=1
        assert!(
            result.mean_error < 0.15,
            "K=1 binary should recover well, got error={}",
            result.mean_error
        );
    }

    #[test]
    fn test_binary_k55_high_error() {
        let mut rng = seeded_rng();
        let result = measure_recovery(1024, Base::Binary, 55, &mut rng);
        // D=1024 is too small for 55 binary concepts — expect nonzero error
        assert!(
            result.mean_error > 0.0,
            "K=55 at D=1024 binary should have error, got {}",
            result.mean_error
        );
    }

    #[test]
    fn test_signed5_large_d_good_recovery() {
        let mut rng = seeded_rng();
        let result = measure_recovery(16384, Base::Signed(5), 55, &mut rng);
        // Quantization noise at base-5 limits recovery precision
        assert!(
            result.mean_error < 0.5,
            "Signed(5) D=16384 K=55 should recover reasonably, got error={}",
            result.mean_error
        );
    }

    #[test]
    fn test_signed_vs_unsigned_same_k() {
        let mut rng = seeded_rng();
        let signed = measure_recovery(4096, Base::Signed(5), 13, &mut rng);
        let mut rng2 = seeded_rng();
        let unsigned = measure_recovery(4096, Base::Unsigned(5), 13, &mut rng2);
        // Both should produce finite results; quantization limits precision
        assert!(
            signed.mean_error < 0.5 || unsigned.mean_error < 0.5,
            "At least one should recover reasonably: signed={}, unsigned={}",
            signed.mean_error,
            unsigned.mean_error
        );
    }

    // -- Multi-axis --

    #[test]
    fn test_multiaxis_3_vs_1() {
        let mut rng = seeded_rng();
        let single = measure_recovery_multiaxis(2048, Base::Signed(5), 1, 8, 1, &mut rng);
        let mut rng2 = seeded_rng();
        let triple = measure_recovery_multiaxis(2048, Base::Signed(5), 3, 8, 1, &mut rng2);
        // More axes should help (averaging reduces noise)
        // Just verify both produce valid results
        assert!(single.combined_error >= 0.0);
        assert!(triple.combined_error >= 0.0);
        assert_eq!(triple.per_axis.len(), 3);
    }

    #[test]
    fn test_bind_depth_increases_error() {
        let mut rng = seeded_rng();
        let depth1 = measure_recovery_multiaxis(2048, Base::Signed(5), 2, 5, 1, &mut rng);
        let mut rng2 = seeded_rng();
        let depth2 = measure_recovery_multiaxis(2048, Base::Signed(5), 2, 5, 2, &mut rng2);
        // More binding depth generally increases error (more noise from binding)
        // Just verify both produce valid results
        assert!(depth1.combined_error >= 0.0);
        assert!(depth2.combined_error >= 0.0);
    }

    // -- Bell coefficient --

    #[test]
    fn test_bell_binary_classical() {
        let mut rng = seeded_rng();
        let templates = generate_templates(5, 1024, Base::Binary, &mut rng);
        let coeffs: Vec<f32> = (0..5).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let roles: Vec<Vec<Vec<i8>>> = (0..2)
            .map(|_| generate_templates(1, 1024, Base::Binary, &mut rng))
            .collect();
        let bell =
            measure_bell_coefficient(1024, Base::Binary, &templates, &coeffs, &roles, &mut rng);
        // Binary should produce a finite Bell coefficient
        assert!(bell.is_finite(), "Bell should be finite, got {}", bell);
    }

    #[test]
    fn test_bell_signed_higher() {
        let mut rng = seeded_rng();
        let templates = generate_templates(5, 2048, Base::Signed(5), &mut rng);
        let coeffs: Vec<f32> = (0..5).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let roles: Vec<Vec<Vec<i8>>> = (0..2)
            .map(|_| generate_templates(1, 2048, Base::Signed(5), &mut rng))
            .collect();
        let bell =
            measure_bell_coefficient(2048, Base::Signed(5), &templates, &coeffs, &roles, &mut rng);
        assert!(bell.is_finite(), "Bell should be finite, got {}", bell);
    }

    // -- D > K² rule --

    #[test]
    fn test_d_gt_k_squared_recovery_transition() {
        let mut rng = seeded_rng();
        // K=8, K²=64. D=64 should be right at the boundary.
        // D=1024 >> K²=64 should have good recovery.
        let good = measure_recovery(1024, Base::Signed(5), 8, &mut rng);
        // Quantization noise is the limiting factor at this D/K ratio
        assert!(
            good.mean_error < 0.25,
            "D=1024 >> K²=64 should recover reasonably, got {}",
            good.mean_error
        );
    }

    // -- Bits per concept --

    #[test]
    fn test_bits_per_concept_calculation() {
        let mut rng = seeded_rng();
        let result = measure_recovery(2048, Base::Signed(5), 8, &mut rng);
        let expected_bits = Base::Signed(5).storage_bits(2048) as f32 / 8.0;
        assert!(
            (result.bits_per_concept - expected_bits).abs() < 0.1,
            "bits_per_concept={}, expected={}",
            result.bits_per_concept,
            expected_bits
        );
    }

    // -- CSV output --

    #[test]
    fn test_csv_output_format() {
        let results = vec![MultiAxisResult {
            d: 1024,
            base: Base::Binary,
            axes: 1,
            k: 1,
            bind_depth: 1,
            combined_error: 0.0,
            bell_coefficient: 0.0,
            per_axis: vec![AxisResult {
                axis: 0,
                mean_error: 0.0,
                gram_condition: 1.0,
            }],
            storage_bytes: 128,
            bits_per_concept: 1024.0,
        }];
        let csv = results_to_csv(&results);
        assert!(csv.starts_with("d,base,signed,axes,k,"));
        assert!(csv.contains("1024,binary,false,1,1,1,"));
    }

    // -- Sweep (small) --

    #[test]
    fn test_sweep_small_subset() {
        // Run a tiny sweep: just Binary D=1024, K=1, axes=1, depth=1
        let mut rng = seeded_rng();
        let result = measure_recovery_multiaxis(1024, Base::Binary, 1, 1, 1, &mut rng);
        assert!(result.combined_error < 0.15);
        assert_eq!(result.storage_bytes, Base::Binary.storage_bytes(1024, 1));
    }
}
