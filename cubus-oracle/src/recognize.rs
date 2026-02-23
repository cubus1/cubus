//! Recognition as Projection — 64K-bit LSH + Gram-Schmidt readout.
//!
//! Key insight: the Gram-Schmidt projections computed during OrganicWAL's
//! write phase ARE class scores. Recognition is just reading those projections
//! without committing a write.
//!
//! Three recognition modes:
//!   1. **Hamming brute** — project query to 64K bits, scan all class fingerprints
//!   2. **Projection readout** — compute Gram-Schmidt projections against WAL
//!   3. **Two-stage** — Hamming shortlist (top-8) → projection re-rank
//!
//! Novelty detection: if max projection < threshold OR residual energy > threshold,
//! the query is novel (doesn't match any known class).

use numrus_substrate::{OrganicWAL, PlasticityTracker, XTransPattern};
use crate::sweep::Base;
use numrus_core::Blackboard;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of random hyperplanes for LSH projection.
const NUM_HYPERPLANES: usize = 65536;

/// Novelty threshold: max projection below this → novel.
const NOVELTY_PROJECTION_THRESHOLD: f32 = 0.15;

/// Novelty threshold: residual energy fraction above this → novel.
const NOVELTY_RESIDUAL_THRESHOLD: f32 = 0.85;

/// Default shortlist size for two-stage recognition.
const SHORTLIST_K: usize = 8;

use numrus_core::SplitMix64;

// ---------------------------------------------------------------------------
// Helper: perturbation
// ---------------------------------------------------------------------------

/// Perturb a template by adding Gaussian noise scaled by `noise_level`,
/// then quantize back to i8 range.
fn perturb(template: &[i8], noise_level: f32, rng: &mut SplitMix64) -> Vec<i8> {
    template
        .iter()
        .map(|&v| {
            let noisy = v as f32 + noise_level * rng.next_gaussian() as f32;
            noisy.round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Convert i8 template to f32 vector.
fn template_to_f32(template: &[i8]) -> Vec<f32> {
    template.iter().map(|&v| v as f32).collect()
}

// ---------------------------------------------------------------------------
// Part 1: Projector64K — 64K-bit Random Hyperplane LSH
// ---------------------------------------------------------------------------

/// Random hyperplane LSH projector (configurable size, default 64K bits).
///
/// Each hyperplane is a random unit vector. The projection of a float vector
/// onto each hyperplane gives a sign bit. The concatenation of all sign bits
/// is the fingerprint.
///
/// SNR scales as sqrt(num_hyperplanes): 64K gives ~2.5x the SNR of 8K.
///
/// Performance: hyperplanes stored in a single contiguous buffer
/// `[num_planes * d]` for cache-friendly sequential access during projection.
pub struct Projector64K {
    /// Flat hyperplane buffer: hyperplanes[plane * d .. (plane+1) * d].
    hyperplanes_flat: Vec<f32>,
    /// Dimensionality of input vectors.
    d: usize,
    /// Number of hyperplanes (bits in fingerprint). Must be multiple of 8.
    num_planes: usize,
}

impl Projector64K {
    /// Create a new projector with `d`-dimensional random hyperplanes.
    /// Uses the full 64K hyperplanes.
    ///
    /// Seeded deterministically for reproducibility.
    pub fn new(d: usize, seed: u64) -> Self {
        Self::with_planes(d, NUM_HYPERPLANES, seed)
    }

    /// Create a projector with a custom number of hyperplanes.
    /// `num_planes` must be a multiple of 8.
    pub fn with_planes(d: usize, num_planes: usize, seed: u64) -> Self {
        assert!(num_planes > 0 && num_planes.is_multiple_of(8));
        let mut rng = SplitMix64::new(seed);
        // Single contiguous allocation for cache-friendly access.
        // Layout: hyperplanes_flat[plane * d + dim]
        let mut hyperplanes_flat = vec![0.0f32; num_planes * d];
        for plane in 0..num_planes {
            let offset = plane * d;
            let mut norm_sq = 0.0f32;
            for dim in 0..d {
                let v = rng.next_gaussian() as f32;
                hyperplanes_flat[offset + dim] = v;
                norm_sq += v * v;
            }
            let inv_norm = 1.0 / norm_sq.sqrt().max(1e-10);
            for dim in 0..d {
                hyperplanes_flat[offset + dim] *= inv_norm;
            }
        }

        Self {
            hyperplanes_flat,
            d,
            num_planes,
        }
    }

    /// Project a float vector to a binary fingerprint packed as bytes.
    ///
    /// Each bit = sign(dot(vector, hyperplane_i)).
    /// Hot path: SIMD dot product via `dot_f32` (AVX-512 FMA when available).
    pub fn project(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.d);
        let num_bytes = self.num_planes / 8;
        let mut bytes = vec![0u8; num_bytes];
        let d = self.d;

        for i in 0..self.num_planes {
            let hp = &self.hyperplanes_flat[i * d..(i + 1) * d];
            let dot = numrus_core::simd::dot_f32(vector, hp);
            if dot >= 0.0 {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        bytes
    }

    /// Project an i8 template to a binary fingerprint.
    ///
    /// Converts template to f32 once, then uses SIMD `dot_f32` for all planes.
    pub fn project_signed(&self, template: &[i8]) -> Vec<u8> {
        let fv: Vec<f32> = template.iter().map(|&v| v as f32).collect();
        self.project(&fv)
    }

    /// Project a batch of i8 templates, returning their fingerprints.
    pub fn project_batch(&self, templates: &[Vec<i8>]) -> Vec<Vec<u8>> {
        templates.iter().map(|t| self.project_signed(t)).collect()
    }

    /// Dimensionality.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Number of hyperplanes (bits in fingerprint).
    pub fn num_planes(&self) -> usize {
        self.num_planes
    }

    /// Populate a Blackboard buffer with this projector's hyperplane data.
    ///
    /// Writes the flat hyperplane matrix into buffer `name` on the blackboard.
    /// The buffer is allocated (or reallocated) to the correct size.
    /// After this call, `from_blackboard()` can reconstruct the projector
    /// without regenerating random numbers (copy from arena, not zero-copy).
    pub fn write_to_blackboard(&self, bb: &mut Blackboard, name: &str) {
        bb.alloc_f32(name, self.hyperplanes_flat.len());
        let buf = bb
            .get_f32_mut(name)
            .expect("write_to_blackboard: buffer just allocated, should exist");
        buf.copy_from_slice(&self.hyperplanes_flat);
    }

    /// Reconstruct a projector from a Blackboard buffer (copy, no RNG).
    ///
    /// The hyperplane data is copied from the blackboard into the projector's
    /// owned buffer — avoids regenerating random numbers, not zero-copy.
    /// Use this when the same hyperplanes need to be reused across multiple
    /// experiments at the same dimensionality.
    ///
    /// Returns `None` if the buffer doesn't exist, isn't f32, or has wrong length.
    pub fn from_blackboard(
        bb: &Blackboard,
        name: &str,
        d: usize,
        num_planes: usize,
    ) -> Option<Self> {
        assert!(num_planes > 0 && num_planes.is_multiple_of(8));
        let buf = bb.get_f32(name)?;
        if buf.len() != num_planes * d {
            return None;
        }
        Some(Self {
            hyperplanes_flat: buf.to_vec(),
            d,
            num_planes,
        })
    }
}

// ---------------------------------------------------------------------------
// Part 2: Hamming Distance on 64K-bit Fingerprints
// ---------------------------------------------------------------------------

/// Hamming distance between two binary fingerprints stored as `u8` slices.
///
/// Delegates to `numrus_core::simd::hamming_distance` which uses
/// VPOPCNTDQ (AVX-512) or AVX2 when available.
pub fn hamming_64k(a: &[u8], b: &[u8]) -> u32 {
    numrus_core::simd::hamming_distance(a, b) as u32
}

/// Hamming similarity: 1.0 - (hamming_distance / num_bits).
///
/// 1.0 = identical, 0.5 = random/uncorrelated, 0.0 = anti-correlated.
pub fn hamming_similarity_64k(a: &[u8], b: &[u8]) -> f32 {
    let total_bits = a.len() * 8;
    1.0 - hamming_64k(a, b) as f32 / total_bits as f32
}

// ---------------------------------------------------------------------------
// Part 3: Recognition Result
// ---------------------------------------------------------------------------

/// Result of a recognition query.
#[derive(Clone, Debug)]
pub struct RecognitionResult {
    /// Best matching class index (into recognizer's class list).
    pub best_class: usize,
    /// Best matching class ID.
    pub best_class_id: u32,
    /// Score of the best match (higher = more confident).
    pub best_score: f32,
    /// All class scores, indexed by class position.
    pub scores: Vec<f32>,
    /// Whether the query is novel (doesn't match any known class).
    pub is_novel: bool,
    /// Residual energy fraction: how much of the query is unexplained.
    pub residual_energy: f32,
    /// Recognition method used.
    pub method: RecognitionMethod,
}

/// Recognition method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecognitionMethod {
    HammingBrute,
    ProjectionReadout,
    TwoStage,
}

// ---------------------------------------------------------------------------
// Part 4: Recognizer
// ---------------------------------------------------------------------------

/// Recognition engine using OrganicWAL's Gram-Schmidt projections.
///
/// The Recognizer wraps an OrganicWAL and adds:
///   - 64K-bit fingerprints for fast Hamming search
///   - Projection readout using Gram-Schmidt (the WAL projections ARE scores)
///   - Two-stage: Hamming shortlist → projection re-rank
///   - Novelty detection via residual energy
pub struct Recognizer {
    /// The organic WAL holding class templates and coefficients.
    pub wal: OrganicWAL,
    /// Plasticity tracker for learning.
    pub plasticity: PlasticityTracker,
    /// The holographic container (owned by recognizer, not WAL).
    pub container: Vec<i8>,
    /// 64K-bit projector.
    pub projector: Projector64K,
    /// Fingerprints for each registered class.
    pub fingerprints: Vec<Vec<u8>>,
    /// Running average templates (f32) for incremental learning.
    class_averages: Vec<Vec<f32>>,
    /// Number of examples seen per class.
    class_counts: Vec<u32>,
}

impl Recognizer {
    /// Create a new recognizer with full 64K-bit projection.
    ///
    /// - `d`: dimensionality of templates
    /// - `channels`: number of X-Trans channels (should be >= expected classes)
    /// - `projector_seed`: seed for deterministic hyperplane generation
    pub fn new(d: usize, channels: usize, projector_seed: u64) -> Self {
        Self::with_planes(d, channels, NUM_HYPERPLANES, projector_seed)
    }

    /// Create a recognizer with a custom number of hyperplanes.
    ///
    /// Use fewer planes for testing (e.g., 1024) to reduce memory.
    pub fn with_planes(d: usize, channels: usize, num_planes: usize, projector_seed: u64) -> Self {
        let projector = Projector64K::with_planes(d, num_planes, projector_seed);
        Self::with_projector(d, channels, projector)
    }

    /// Create a recognizer sharing a pre-built projector.
    ///
    /// Use this when running multiple experiments at the same dimensionality
    /// to avoid regenerating 64K hyperplanes per experiment.
    pub fn with_projector(d: usize, channels: usize, projector: Projector64K) -> Self {
        assert_eq!(projector.d(), d);
        let pattern = XTransPattern::new(d, channels);
        let wal = OrganicWAL::new(pattern);
        let plasticity = PlasticityTracker::new(0, 50);
        let container = vec![0i8; d];

        Self {
            wal,
            plasticity,
            container,
            projector,
            fingerprints: Vec::new(),
            class_averages: Vec::new(),
            class_counts: Vec::new(),
        }
    }

    /// Take the projector out for reuse in another Recognizer.
    pub fn take_projector(self) -> Projector64K {
        self.projector
    }

    /// Number of registered classes.
    pub fn num_classes(&self) -> usize {
        self.wal.k()
    }

    /// Register a new class with its initial template.
    ///
    /// Returns the class index.
    pub fn register_class(&mut self, class_id: u32, template: Vec<i8>) -> usize {
        let fp = self.projector.project_signed(&template);
        let avg = template_to_f32(&template);

        self.wal.register_concept(class_id, template);
        self.plasticity.add_concept();
        self.fingerprints.push(fp);
        self.class_averages.push(avg);
        self.class_counts.push(1);

        self.num_classes() - 1
    }

    /// Learn: update class template with a new example and write to container.
    ///
    /// The running average is updated incrementally:
    ///   avg = (count * avg + new) / (count + 1)
    ///
    /// Then the averaged template is quantized and written via the WAL.
    pub fn learn(&mut self, class_idx: usize, example: &[i8], amplitude: f32, learning_rate: f32) {
        let d = self.projector.d();
        assert_eq!(example.len(), d);

        // Update running average
        let count = self.class_counts[class_idx] as f32;
        for j in 0..d {
            self.class_averages[class_idx][j] =
                (count * self.class_averages[class_idx][j] + example[j] as f32) / (count + 1.0);
        }
        self.class_counts[class_idx] += 1;

        // Quantize averaged template back to i8 for the WAL
        let quantized: Vec<i8> = self.class_averages[class_idx]
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
            .collect();

        // Update the WAL's template and norm
        self.wal.update_template(class_idx, &quantized);

        // Update fingerprint
        self.fingerprints[class_idx] = self.projector.project_signed(&quantized);

        // Write to container via WAL with plasticity
        self.wal.write_plastic(
            &mut self.container,
            class_idx,
            amplitude,
            learning_rate,
            &mut self.plasticity,
        );
    }

    /// Recognize via Hamming brute-force: project query, find nearest fingerprint.
    pub fn recognize_hamming(&self, query: &[i8]) -> RecognitionResult {
        let query_fp = self.projector.project_signed(query);
        let k = self.num_classes();

        let mut scores = vec![0.0f32; k];
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..k {
            let sim = hamming_similarity_64k(&query_fp, &self.fingerprints[i]);
            // Map from [0.5, 1.0] range to [0.0, 1.0] for scoring
            scores[i] = (sim - 0.5) * 2.0;
            if scores[i] > best_score {
                best_score = scores[i];
                best_idx = i;
            }
        }

        let residual = self.compute_residual_energy(query);
        let is_novel =
            best_score < NOVELTY_PROJECTION_THRESHOLD || residual > NOVELTY_RESIDUAL_THRESHOLD;

        RecognitionResult {
            best_class: best_idx,
            best_class_id: self.wal.concept_ids[best_idx],
            best_score,
            scores,
            is_novel,
            residual_energy: residual,
            method: RecognitionMethod::HammingBrute,
        }
    }

    /// Recognize via projection readout: compute Gram-Schmidt projections.
    ///
    /// This is the key insight — the projections during Gram-Schmidt
    /// orthogonalization ARE class scores. We compute them without writing.
    pub fn recognize_orthogonal(&self, query: &[i8]) -> RecognitionResult {
        let d = self.projector.d();
        let k = self.num_classes();

        // Convert query to f32
        let mut residual = vec![0.0f32; d];
        for j in 0..d {
            residual[j] = query[j] as f32;
        }

        let query_energy: f32 = residual.iter().map(|x| x * x).sum();

        // Project onto each class template (Gram-Schmidt style)
        let mut scores = vec![0.0f32; k];

        for i in 0..k {
            let norm = self.wal.template_norm(i);
            if norm < 1e-10 {
                continue;
            }

            let known = self.wal.template(i);
            let mut dot = 0.0f64;
            for j in 0..d {
                dot += residual[j] as f64 * known[j] as f64;
            }

            let proj_coeff = (dot / norm) as f32;
            scores[i] = proj_coeff;

            // Subtract projection from residual
            for j in 0..d {
                residual[j] -= proj_coeff * known[j] as f32;
            }
        }

        let residual_energy_val: f32 = residual.iter().map(|x| x * x).sum();
        let residual_frac = if query_energy > 1e-10 {
            residual_energy_val / query_energy
        } else {
            1.0
        };

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..k {
            if scores[i].abs() > best_score {
                best_score = scores[i].abs();
                best_idx = i;
            }
        }
        // Use the actual signed score for the best match
        best_score = scores[best_idx];

        let is_novel = best_score.abs() < NOVELTY_PROJECTION_THRESHOLD
            || residual_frac > NOVELTY_RESIDUAL_THRESHOLD;

        RecognitionResult {
            best_class: best_idx,
            best_class_id: self.wal.concept_ids[best_idx],
            best_score,
            scores,
            is_novel,
            residual_energy: residual_frac,
            method: RecognitionMethod::ProjectionReadout,
        }
    }

    /// Recognize via two-stage: Hamming shortlist → projection re-rank.
    ///
    /// Stage 1: Hamming distance finds top-SHORTLIST_K candidates.
    /// Stage 2: Full projection readout on shortlist only.
    pub fn recognize_two_stage(&self, query: &[i8]) -> RecognitionResult {
        let k = self.num_classes();
        if k <= SHORTLIST_K {
            // Small enough for full projection readout
            return self.recognize_orthogonal(query);
        }

        // Stage 1: Hamming shortlist
        let query_fp = self.projector.project_signed(query);
        let mut hamming_scores: Vec<(usize, f32)> = (0..k)
            .map(|i| {
                let sim = hamming_similarity_64k(&query_fp, &self.fingerprints[i]);
                (i, sim)
            })
            .collect();
        hamming_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let shortlist: Vec<usize> = hamming_scores[..SHORTLIST_K]
            .iter()
            .map(|&(idx, _)| idx)
            .collect();

        // Stage 2: Projection readout on shortlist
        let d = self.projector.d();
        let mut residual = template_to_f32(query);
        let query_energy: f32 = residual.iter().map(|x| x * x).sum();

        let mut scores = vec![0.0f32; k];

        for &i in &shortlist {
            let norm = self.wal.template_norm(i);
            if norm < 1e-10 {
                continue;
            }

            let known = self.wal.template(i);
            let mut dot = 0.0f64;
            for j in 0..d {
                dot += residual[j] as f64 * known[j] as f64;
            }

            let proj_coeff = (dot / norm) as f32;
            scores[i] = proj_coeff;

            for j in 0..d {
                residual[j] -= proj_coeff * known[j] as f32;
            }
        }

        let residual_energy_val: f32 = residual.iter().map(|x| x * x).sum();
        let residual_frac = if query_energy > 1e-10 {
            residual_energy_val / query_energy
        } else {
            1.0
        };

        let mut best_idx = shortlist[0];
        let mut best_score = f32::NEG_INFINITY;
        for &i in &shortlist {
            if scores[i].abs() > best_score {
                best_score = scores[i].abs();
                best_idx = i;
            }
        }
        best_score = scores[best_idx];

        let is_novel = best_score.abs() < NOVELTY_PROJECTION_THRESHOLD
            || residual_frac > NOVELTY_RESIDUAL_THRESHOLD;

        RecognitionResult {
            best_class: best_idx,
            best_class_id: self.wal.concept_ids[best_idx],
            best_score,
            scores,
            is_novel,
            residual_energy: residual_frac,
            method: RecognitionMethod::TwoStage,
        }
    }

    /// Compute residual energy fraction for a query against all classes.
    fn compute_residual_energy(&self, query: &[i8]) -> f32 {
        let d = self.projector.d();
        let mut residual = template_to_f32(query);
        let query_energy: f32 = residual.iter().map(|x| x * x).sum();

        for i in 0..self.num_classes() {
            let norm = self.wal.template_norm(i);
            if norm < 1e-10 {
                continue;
            }
            let known = self.wal.template(i);
            let mut dot = 0.0f64;
            for j in 0..d {
                dot += residual[j] as f64 * known[j] as f64;
            }
            let proj_coeff = (dot / norm) as f32;
            for j in 0..d {
                residual[j] -= proj_coeff * known[j] as f32;
            }
        }

        let residual_energy_val: f32 = residual.iter().map(|x| x * x).sum();
        if query_energy > 1e-10 {
            residual_energy_val / query_energy
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Part 5: Experiment — Recognition Accuracy Sweep
// ---------------------------------------------------------------------------

/// Result of a recognition experiment at one parameter point.
#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub d: usize,
    pub base: Base,
    pub channels: usize,
    pub num_classes: usize,
    pub num_examples_per_class: usize,
    pub noise_level: f32,
    pub hamming_accuracy: f32,
    pub projection_accuracy: f32,
    pub two_stage_accuracy: f32,
    pub hamming_mean_score: f32,
    pub projection_mean_score: f32,
    pub novelty_detection_rate: f32,
    pub mean_residual_energy: f32,
}

/// Run a single recognition experiment (full 64K-bit projection).
pub fn run_recognition_experiment(
    d: usize,
    base: Base,
    channels: usize,
    num_classes: usize,
    examples_per_class: usize,
    noise_level: f32,
    seed: u64,
) -> ExperimentResult {
    run_recognition_experiment_inner(
        d,
        base,
        channels,
        num_classes,
        examples_per_class,
        noise_level,
        seed,
        NUM_HYPERPLANES,
    )
}

/// Inner experiment function with configurable plane count.
fn run_recognition_experiment_inner(
    d: usize,
    base: Base,
    channels: usize,
    num_classes: usize,
    examples_per_class: usize,
    noise_level: f32,
    seed: u64,
    num_planes: usize,
) -> ExperimentResult {
    let projector = Projector64K::with_planes(d, num_planes, seed ^ 0xCAFE);
    let (result, _projector) = run_recognition_experiment_with_projector(
        d,
        base,
        channels,
        num_classes,
        examples_per_class,
        noise_level,
        seed,
        projector,
    );
    result
}

/// Experiment with a pre-built projector (avoids regenerating hyperplanes).
///
/// Returns the projector alongside the result for reuse (blackboard pattern).
fn run_recognition_experiment_with_projector(
    d: usize,
    base: Base,
    channels: usize,
    num_classes: usize,
    examples_per_class: usize,
    noise_level: f32,
    seed: u64,
    projector: Projector64K,
) -> (ExperimentResult, Projector64K) {
    let mut rng = SplitMix64::new(seed);

    // Generate class templates
    let half = match base {
        Base::Binary => 1i8,
        Base::Unsigned(b) => (b - 1) as i8,
        Base::Signed(b) => (b / 2) as i8,
    };
    let min_val = match base {
        Base::Binary => 0i8,
        Base::Unsigned(_) => 0i8,
        Base::Signed(b) => -((b / 2) as i8),
    };

    let templates: Vec<Vec<i8>> = (0..num_classes)
        .map(|_| (0..d).map(|_| rng.gen_range_i8(min_val, half)).collect())
        .collect();

    // Create recognizer with shared projector
    let mut recognizer = Recognizer::with_projector(d, channels, projector);

    // Register classes
    for (i, t) in templates.iter().enumerate() {
        recognizer.register_class(i as u32, t.clone());
    }

    // Train: write each class multiple times with noisy examples
    for _example in 0..examples_per_class {
        for class_idx in 0..num_classes {
            let noisy = perturb(&templates[class_idx], noise_level, &mut rng);
            recognizer.learn(class_idx, &noisy, 0.5, 0.1);
        }
    }

    // Test: generate fresh noisy queries and measure accuracy
    let test_per_class = 5;
    let total_tests = num_classes * test_per_class;

    let mut hamming_correct = 0u32;
    let mut projection_correct = 0u32;
    let mut two_stage_correct = 0u32;
    let mut hamming_score_sum = 0.0f32;
    let mut projection_score_sum = 0.0f32;
    let mut residual_sum = 0.0f32;

    for class_idx in 0..num_classes {
        for _ in 0..test_per_class {
            let query = perturb(&templates[class_idx], noise_level, &mut rng);

            let hr = recognizer.recognize_hamming(&query);
            if hr.best_class == class_idx {
                hamming_correct += 1;
            }
            hamming_score_sum += hr.best_score;

            let pr = recognizer.recognize_orthogonal(&query);
            if pr.best_class == class_idx {
                projection_correct += 1;
            }
            projection_score_sum += pr.best_score.abs();
            residual_sum += pr.residual_energy;

            let tr = recognizer.recognize_two_stage(&query);
            if tr.best_class == class_idx {
                two_stage_correct += 1;
            }
        }
    }

    // Novelty detection test: generate random vectors (no class)
    let novelty_tests = 20;
    let mut novelty_detected = 0u32;
    for _ in 0..novelty_tests {
        let novel: Vec<i8> = (0..d).map(|_| rng.gen_range_i8(min_val, half)).collect();
        let pr = recognizer.recognize_orthogonal(&novel);
        if pr.is_novel {
            novelty_detected += 1;
        }
    }

    let result = ExperimentResult {
        d,
        base,
        channels,
        num_classes,
        num_examples_per_class: examples_per_class,
        noise_level,
        hamming_accuracy: hamming_correct as f32 / total_tests as f32,
        projection_accuracy: projection_correct as f32 / total_tests as f32,
        two_stage_accuracy: two_stage_correct as f32 / total_tests as f32,
        hamming_mean_score: hamming_score_sum / total_tests as f32,
        projection_mean_score: projection_score_sum / total_tests as f32,
        novelty_detection_rate: novelty_detected as f32 / novelty_tests as f32,
        mean_residual_energy: residual_sum / total_tests as f32,
    };

    // Return projector for reuse (blackboard pattern)
    let projector = recognizer.take_projector();
    (result, projector)
}

/// Run a sweep across parameter space (full 64K planes — slow but precise).
pub fn run_recognition_sweep() -> Vec<ExperimentResult> {
    run_recognition_sweep_with_planes(NUM_HYPERPLANES)
}

/// Run a fast sweep with reduced plane count (4096 — ~16× faster).
///
/// Quality numbers will be slightly lower than 64K but the relative
/// ranking between methods is preserved. Use for development iteration.
pub fn run_recognition_sweep_fast() -> Vec<ExperimentResult> {
    run_recognition_sweep_with_planes(4096)
}

/// Inner sweep with configurable plane count.
///
/// Uses the Blackboard pattern from numrus-core: hyperplane matrices are
/// generated once per dimensionality into a 64-byte-aligned arena buffer,
/// then each experiment reconstructs its projector from that buffer.
/// This avoids regenerating random hyperplanes per experiment (the dominant cost).
fn run_recognition_sweep_with_planes(num_planes: usize) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    let dims = [512, 1024, 2048];
    let bases = [Base::Signed(5), Base::Signed(7)];
    let class_counts = [4, 8, 16, 32];
    let noise_levels = [0.3, 0.5, 1.0];

    let mut bb = Blackboard::new();
    let projector_seed = 42u64 ^ 0xCAFE;

    for &d in &dims {
        // Build the projector ONCE per dimensionality, write to blackboard.
        let buf_name = format!("hp_{}", d);
        let master = Projector64K::with_planes(d, num_planes, projector_seed);
        master.write_to_blackboard(&mut bb, &buf_name);

        for &base in &bases {
            for &nc in &class_counts {
                let channels = (nc * 2).max(16).min(d / 4);
                for &noise in &noise_levels {
                    // Reconstruct projector from blackboard (copy, no RNG)
                    let projector = Projector64K::from_blackboard(&bb, &buf_name, d, num_planes)
                        .expect("projector buffer should exist in blackboard");
                    let (result, _) = run_recognition_experiment_with_projector(
                        d,
                        base,
                        channels,
                        nc,
                        3,     // examples_per_class
                        noise, // noise_level
                        42 + d as u64 + nc as u64,
                        projector,
                    );
                    results.push(result);
                }
            }
        }

        // Free the buffer for this dimensionality (next d will allocate its own)
        bb.free(&buf_name);
    }

    results
}

/// Print experiment results as a formatted table.
pub fn print_recognition_results(results: &[ExperimentResult]) {
    println!("{}", "=".repeat(120));
    println!("  RECOGNITION AS PROJECTION — Accuracy Sweep");
    println!("{}\n", "=".repeat(120));

    println!(
        "{:>6} {:>10} {:>4} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "D",
        "Base",
        "Ch",
        "K",
        "Noise",
        "Hamming",
        "Proj",
        "2-Stage",
        "H.Score",
        "P.Score",
        "Novelty",
        "Residual"
    );
    println!("{}", "-".repeat(120));

    for r in results {
        println!(
            "{:>6} {:>10} {:>4} {:>6} {:>6.2} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.3} {:>8.3} {:>8.1}% {:>8.3}",
            r.d,
            r.base.name(),
            r.channels,
            r.num_classes,
            r.noise_level,
            r.hamming_accuracy * 100.0,
            r.projection_accuracy * 100.0,
            r.two_stage_accuracy * 100.0,
            r.hamming_mean_score,
            r.projection_mean_score,
            r.novelty_detection_rate * 100.0,
            r.mean_residual_energy,
        );
    }

    // Summary statistics
    println!("\n{}", "=".repeat(120));
    println!("  SUMMARY\n");

    let avg_hamming: f32 =
        results.iter().map(|r| r.hamming_accuracy).sum::<f32>() / results.len() as f32;
    let avg_proj: f32 =
        results.iter().map(|r| r.projection_accuracy).sum::<f32>() / results.len() as f32;
    let avg_two: f32 =
        results.iter().map(|r| r.two_stage_accuracy).sum::<f32>() / results.len() as f32;
    let avg_novelty: f32 = results
        .iter()
        .map(|r| r.novelty_detection_rate)
        .sum::<f32>()
        / results.len() as f32;

    println!("  Mean Hamming accuracy:     {:>6.1}%", avg_hamming * 100.0);
    println!("  Mean Projection accuracy:  {:>6.1}%", avg_proj * 100.0);
    println!("  Mean Two-Stage accuracy:   {:>6.1}%", avg_two * 100.0);
    println!("  Mean Novelty detection:    {:>6.1}%", avg_novelty * 100.0);

    // Find best config
    if let Some(best) = results
        .iter()
        .max_by(|a, b| a.projection_accuracy.total_cmp(&b.projection_accuracy))
    {
        println!(
            "\n  Best projection config: D={}, {}, K={}, noise={:.2} → {:.1}%",
            best.d,
            best.base.name(),
            best.num_classes,
            best.noise_level,
            best.projection_accuracy * 100.0
        );
    }
}

/// Main entry point for the recognition experiment.
pub fn run_recognition() {
    println!("\n{}", "=".repeat(120));
    println!("  RECOGNITION AS PROJECTION");
    println!("  64K-bit LSH + Gram-Schmidt Readout");
    println!("{}\n", "=".repeat(120));

    // --- Timing benchmarks ---
    println!("--- Timing benchmarks ---\n");
    run_timing_benchmarks();
    println!();

    // --- Quick single experiment (4K planes for speed) ---
    println!("--- Quick single experiment: D=1024, Signed(7), K=8, noise=0.5 (4K planes) ---\n");
    let quick = run_recognition_experiment_inner(
        1024,
        Base::Signed(7),
        32,  // channels
        8,   // num_classes
        3,   // examples
        0.5, // noise
        42,
        4096,
    );

    println!(
        "  Hamming accuracy:    {:.1}%",
        quick.hamming_accuracy * 100.0
    );
    println!(
        "  Projection accuracy: {:.1}%",
        quick.projection_accuracy * 100.0
    );
    println!(
        "  Two-stage accuracy:  {:.1}%",
        quick.two_stage_accuracy * 100.0
    );
    println!(
        "  Novelty detection:   {:.1}%",
        quick.novelty_detection_rate * 100.0
    );
    println!("  Mean residual:       {:.3}", quick.mean_residual_energy);
    println!();

    // --- Full sweep (4K planes for feasible runtime) ---
    println!("--- Parameter sweep (4096 planes) ---\n");
    let results = run_recognition_sweep_fast();
    print_recognition_results(&results);

    // --- Plane count comparison: 1K vs 4K vs 16K vs 64K ---
    println!("\n--- Plane count comparison: D=1024, Signed(7), K=8, noise=0.5 ---\n");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Planes", "Hamming", "Proj", "2-Stage", "Novelty", "Residual"
    );
    println!("{}", "-".repeat(68));
    for &planes in &[1024, 4096, 16384, 65536] {
        let t0 = std::time::Instant::now();
        let r = run_recognition_experiment_inner(1024, Base::Signed(7), 32, 8, 3, 0.5, 42, planes);
        let elapsed = t0.elapsed();
        println!(
            "{:>8} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.3}  ({:.1}s)",
            planes,
            r.hamming_accuracy * 100.0,
            r.projection_accuracy * 100.0,
            r.two_stage_accuracy * 100.0,
            r.novelty_detection_rate * 100.0,
            r.mean_residual_energy,
            elapsed.as_secs_f32()
        );
    }
}

/// Run timing benchmarks for individual operations.
fn run_timing_benchmarks() {
    use std::time::Instant;

    let d = 1024;
    let num_classes = 20;
    let mut rng = SplitMix64::new(42);

    // 1. Projector creation (64K)
    let t0 = Instant::now();
    let proj = Projector64K::new(d, 99);
    let proj_create_ms = t0.elapsed().as_millis();
    println!(
        "  Projector64K::new(D={}, 64K planes):  {}ms",
        d, proj_create_ms
    );

    // 2. Single projection (64K bits)
    let template: Vec<i8> = (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect();
    let fv: Vec<f32> = template.iter().map(|&v| v as f32).collect();
    let t0 = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        std::hint::black_box(proj.project(std::hint::black_box(&fv)));
    }
    let proj_us = t0.elapsed().as_micros() as f64 / iters as f64;
    println!(
        "  Single project (64K bits):             {:.1}us ({:.2}ms)",
        proj_us,
        proj_us / 1000.0
    );

    // 3. Batch projection (100 templates)
    let templates: Vec<Vec<i8>> = (0..100)
        .map(|_| (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect())
        .collect();
    let t0 = Instant::now();
    let _fps = proj.project_batch(&templates);
    let batch_ms = t0.elapsed().as_millis();
    println!("  Batch project (100 × 64K):             {}ms", batch_ms);

    // 4. Hamming distance (64K bits)
    let fp1 = proj.project_signed(&templates[0]);
    let fp2 = proj.project_signed(&templates[1]);
    let t0 = Instant::now();
    let ham_iters = 100_000;
    for _ in 0..ham_iters {
        std::hint::black_box(hamming_64k(
            std::hint::black_box(&fp1),
            std::hint::black_box(&fp2),
        ));
    }
    let ham_ns = t0.elapsed().as_nanos() as f64 / ham_iters as f64;
    println!("  Hamming 64K (1024 u64s):               {:.0}ns", ham_ns);

    // 5. Recognizer creation + class registration
    let t0 = Instant::now();
    let mut recognizer = Recognizer::with_planes(d, num_classes * 2, 4096, 99);
    let class_templates: Vec<Vec<i8>> = (0..num_classes)
        .map(|_| (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect())
        .collect();
    for (i, t) in class_templates.iter().enumerate() {
        recognizer.register_class(i as u32, t.clone());
    }
    let reg_ms = t0.elapsed().as_millis();
    println!(
        "  Recognizer setup ({} classes, 4K):     {}ms",
        num_classes, reg_ms
    );

    // 6. Single recognize (projection readout, 20 classes)
    let query = &class_templates[0];
    let t0 = Instant::now();
    let rec_iters = 1000;
    for _ in 0..rec_iters {
        std::hint::black_box(recognizer.recognize_orthogonal(std::hint::black_box(query)));
    }
    let rec_us = t0.elapsed().as_micros() as f64 / rec_iters as f64;
    println!(
        "  Single recognize ({} classes):         {:.1}us ({:.3}ms)",
        num_classes,
        rec_us,
        rec_us / 1000.0
    );

    // 7. Single recognize_hamming (20 classes)
    let t0 = Instant::now();
    for _ in 0..rec_iters {
        std::hint::black_box(recognizer.recognize_hamming(std::hint::black_box(query)));
    }
    let ham_rec_us = t0.elapsed().as_micros() as f64 / rec_iters as f64;
    println!(
        "  Single hamming recognize ({} cl):     {:.1}us ({:.3}ms)",
        num_classes,
        ham_rec_us,
        ham_rec_us / 1000.0
    );

    // 8. Single two-stage recognize (20 classes)
    let t0 = Instant::now();
    for _ in 0..rec_iters {
        std::hint::black_box(recognizer.recognize_two_stage(std::hint::black_box(query)));
    }
    let two_rec_us = t0.elapsed().as_micros() as f64 / rec_iters as f64;
    println!(
        "  Single two-stage ({} classes):        {:.1}us ({:.3}ms)",
        num_classes,
        two_rec_us,
        two_rec_us / 1000.0
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random unit vector of dimension `d` using the provided RNG.
    fn random_unit_vector(d: usize, rng: &mut SplitMix64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for x in v.iter_mut() {
            *x /= norm;
        }
        v
    }

    // --- SplitMix64 tests ---

    #[test]
    fn test_rng_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_rng_gaussian_distribution() {
        let mut rng = SplitMix64::new(12345);
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        // Mean should be near 0, variance near 1
        assert!(mean.abs() < 0.1, "Gaussian mean = {}, expected ~0.0", mean);
        assert!(
            (variance - 1.0).abs() < 0.2,
            "Gaussian variance = {}, expected ~1.0",
            variance
        );
    }

    // --- Projector64K tests ---
    // Use 1024 planes for tests to keep memory reasonable in parallel runs.

    const TEST_PLANES: usize = 1024;

    #[test]
    fn test_projector_fingerprint_size() {
        let proj = Projector64K::with_planes(128, TEST_PLANES, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let fp = proj.project(&v);
        assert_eq!(fp.len(), TEST_PLANES / 8);
    }

    #[test]
    fn test_projector_deterministic() {
        let proj = Projector64K::with_planes(128, TEST_PLANES, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let fp1 = proj.project(&v);
        let fp2 = proj.project(&v);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_projector_self_similarity() {
        let proj = Projector64K::with_planes(128, TEST_PLANES, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let fp = proj.project(&v);
        assert_eq!(hamming_64k(&fp, &fp), 0);
        assert!((hamming_similarity_64k(&fp, &fp) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_projector_similar_vectors_close() {
        let proj = Projector64K::with_planes(256, TEST_PLANES, 99);
        let mut rng = SplitMix64::new(77);

        let v: Vec<f32> = (0..256).map(|_| rng.next_gaussian() as f32).collect();

        // Small perturbation
        let v_noisy: Vec<f32> = v
            .iter()
            .map(|&x| x + 0.1 * rng.next_gaussian() as f32)
            .collect();

        let fp_v = proj.project(&v);
        let fp_n = proj.project(&v_noisy);

        let sim = hamming_similarity_64k(&fp_v, &fp_n);
        assert!(
            sim > 0.7,
            "Similar vectors should have high Hamming similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_projector_orthogonal_vectors_half() {
        let proj = Projector64K::with_planes(256, TEST_PLANES, 55);
        let mut v1 = vec![0.0f32; 256];
        let mut v2 = vec![0.0f32; 256];

        // Two orthogonal vectors
        for i in 0..128 {
            v1[i] = 1.0;
        }
        for i in 128..256 {
            v2[i] = 1.0;
        }

        let fp1 = proj.project(&v1);
        let fp2 = proj.project(&v2);
        let sim = hamming_similarity_64k(&fp1, &fp2);

        // Orthogonal vectors should have ~0.5 similarity
        assert!(
            (sim - 0.5).abs() < 0.1,
            "Orthogonal vectors should have ~0.5 Hamming similarity, got {}",
            sim
        );
    }

    // --- Hamming tests ---

    #[test]
    fn test_hamming_zero_distance() {
        let a = vec![0xFFu8; 128];
        assert_eq!(hamming_64k(&a, &a), 0);
    }

    #[test]
    fn test_hamming_max_distance() {
        let a = vec![0u8; 128];
        let b = vec![0xFFu8; 128];
        assert_eq!(hamming_64k(&a, &b), (128 * 8) as u32);
    }

    #[test]
    fn test_hamming_single_bit() {
        let a = vec![0u8; 128];
        let mut b = vec![0u8; 128];
        b[0] = 1; // one bit different
        assert_eq!(hamming_64k(&a, &b), 1);
    }

    // --- Recognizer tests ---
    // Use TEST_PLANES for all recognizer tests to keep memory low.

    fn make_test_recognizer(d: usize, num_classes: usize) -> (Recognizer, Vec<Vec<i8>>) {
        let mut rng = SplitMix64::new(42);
        let channels = (num_classes * 2).max(8);
        let mut recognizer = Recognizer::with_planes(d, channels, TEST_PLANES, 99);

        let templates: Vec<Vec<i8>> = (0..num_classes)
            .map(|_| (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect())
            .collect();

        for (i, t) in templates.iter().enumerate() {
            recognizer.register_class(i as u32, t.clone());
        }

        (recognizer, templates)
    }

    #[test]
    fn test_recognizer_register_classes() {
        let (recognizer, _) = make_test_recognizer(256, 4);
        assert_eq!(recognizer.num_classes(), 4);
        assert_eq!(recognizer.fingerprints.len(), 4);
    }

    #[test]
    fn test_recognizer_self_recognition_hamming() {
        let (recognizer, templates) = make_test_recognizer(512, 4);

        for (i, t) in templates.iter().enumerate() {
            let result = recognizer.recognize_hamming(t);
            assert_eq!(
                result.best_class, i,
                "Hamming failed to recognize class {} (got {})",
                i, result.best_class
            );
        }
    }

    #[test]
    fn test_recognizer_self_recognition_projection() {
        let (recognizer, templates) = make_test_recognizer(512, 4);

        for (i, t) in templates.iter().enumerate() {
            let result = recognizer.recognize_orthogonal(t);
            assert_eq!(
                result.best_class, i,
                "Projection failed to recognize class {} (got {}), scores: {:?}",
                i, result.best_class, result.scores
            );
        }
    }

    #[test]
    fn test_recognizer_self_recognition_two_stage() {
        let (recognizer, templates) = make_test_recognizer(512, 4);

        for (i, t) in templates.iter().enumerate() {
            let result = recognizer.recognize_two_stage(t);
            assert_eq!(
                result.best_class, i,
                "Two-stage failed to recognize class {} (got {})",
                i, result.best_class
            );
        }
    }

    #[test]
    fn test_recognizer_noisy_recognition() {
        let (recognizer, templates) = make_test_recognizer(512, 4);
        let mut rng = SplitMix64::new(999);

        let mut correct = 0;
        let trials = 20;
        for _ in 0..trials {
            for (i, t) in templates.iter().enumerate() {
                let noisy = perturb(t, 0.5, &mut rng);
                let result = recognizer.recognize_orthogonal(&noisy);
                if result.best_class == i {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f32 / (trials * templates.len()) as f32;
        assert!(
            accuracy > 0.5,
            "Noisy recognition accuracy = {:.1}%, expected > 50%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_recognizer_learn_improves() {
        let d = 512;
        let num_classes = 4;
        let mut rng = SplitMix64::new(42);
        let channels = num_classes * 2;
        let mut recognizer = Recognizer::with_planes(d, channels, TEST_PLANES, 99);

        let templates: Vec<Vec<i8>> = (0..num_classes)
            .map(|_| (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect())
            .collect();

        for (i, t) in templates.iter().enumerate() {
            recognizer.register_class(i as u32, t.clone());
        }

        // Measure accuracy BEFORE learning
        let mut before_correct = 0u32;
        let mut rng_before = SplitMix64::new(999);
        for (i, t) in templates.iter().enumerate() {
            let noisy = perturb(t, 1.0, &mut rng_before);
            let result = recognizer.recognize_orthogonal(&noisy);
            if result.best_class == i {
                before_correct += 1;
            }
        }
        let before_accuracy = before_correct as f32 / num_classes as f32;

        // Learn with multiple noisy examples
        for _ in 0..5 {
            for (i, t) in templates.iter().enumerate() {
                let noisy = perturb(t, 0.5, &mut rng);
                recognizer.learn(i, &noisy, 0.5, 0.1);
            }
        }

        // Measure accuracy AFTER learning
        let mut rng_after = SplitMix64::new(999);
        let mut post_correct = 0u32;
        for (i, t) in templates.iter().enumerate() {
            let noisy = perturb(t, 1.0, &mut rng_after);
            let result = recognizer.recognize_orthogonal(&noisy);
            if result.best_class == i {
                post_correct += 1;
            }
        }
        let after_accuracy = post_correct as f32 / num_classes as f32;

        assert!(
            after_accuracy >= before_accuracy,
            "Learning should not decrease accuracy: before={:.1}%, after={:.1}%",
            before_accuracy * 100.0,
            after_accuracy * 100.0
        );
        assert!(
            after_accuracy > 0.5,
            "Post-learning accuracy {:.1}% should exceed 50%",
            after_accuracy * 100.0
        );
    }

    #[test]
    fn test_novelty_detection_random_query() {
        let (recognizer, _) = make_test_recognizer(512, 8);
        let mut rng = SplitMix64::new(777);

        // Random vector should be detected as novel
        let novel: Vec<i8> = (0..512).map(|_| rng.gen_range_i8(-3, 3)).collect();
        let result = recognizer.recognize_orthogonal(&novel);

        // With 8 classes in 512-D, a random vector should have high residual
        assert!(
            result.residual_energy > 0.5,
            "Random query residual = {}, expected > 0.5",
            result.residual_energy
        );
    }

    #[test]
    fn test_residual_energy_known_class() {
        let (recognizer, templates) = make_test_recognizer(512, 4);

        // Known template should have low residual
        let result = recognizer.recognize_orthogonal(&templates[0]);
        assert!(
            result.residual_energy < 0.99,
            "Known class residual = {}, expected < 0.99",
            result.residual_energy
        );
    }

    #[test]
    fn test_projection_scores_structure() {
        let (recognizer, templates) = make_test_recognizer(256, 4);

        let result = recognizer.recognize_orthogonal(&templates[0]);
        assert_eq!(result.scores.len(), 4);

        // The matched class should have the highest absolute score
        let best_abs = result.scores[result.best_class].abs();
        for (i, &s) in result.scores.iter().enumerate() {
            if i != result.best_class {
                assert!(
                    best_abs >= s.abs() - 1e-6,
                    "Best class {} score {} < class {} score {}",
                    result.best_class,
                    best_abs,
                    i,
                    s.abs()
                );
            }
        }
    }

    #[test]
    fn test_hamming_similarity_range() {
        let proj = Projector64K::with_planes(128, TEST_PLANES, 42);
        let mut rng = SplitMix64::new(11);

        let v1: Vec<f32> = (0..128).map(|_| rng.next_gaussian() as f32).collect();
        let v2: Vec<f32> = (0..128).map(|_| rng.next_gaussian() as f32).collect();

        let fp1 = proj.project(&v1);
        let fp2 = proj.project(&v2);
        let sim = hamming_similarity_64k(&fp1, &fp2);

        assert!(
            (0.0..=1.0).contains(&sim),
            "Similarity {} out of [0,1]",
            sim
        );
    }

    #[test]
    fn test_two_stage_correctness() {
        // With many classes, two-stage should still find the right answer
        let (recognizer, templates) = make_test_recognizer(512, 16);

        for (i, t) in templates.iter().enumerate() {
            let result = recognizer.recognize_two_stage(t);
            assert_eq!(
                result.best_class, i,
                "Two-stage failed for class {} (got {})",
                i, result.best_class
            );
            assert_eq!(result.method, RecognitionMethod::TwoStage);
        }
    }

    #[test]
    fn test_perturb_preserves_range() {
        let mut rng = SplitMix64::new(42);
        let template: Vec<i8> = (0..256).map(|_| rng.gen_range_i8(-3, 3)).collect();
        let noisy = perturb(&template, 2.0, &mut rng);
        assert_eq!(noisy.len(), 256);
    }

    #[test]
    fn test_random_unit_vector_norm() {
        let mut rng = SplitMix64::new(42);
        let v = random_unit_vector(256, &mut rng);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Unit vector norm = {}, expected 1.0",
            norm
        );
    }

    #[test]
    fn test_experiment_runs() {
        let result = run_recognition_experiment_inner(
            256,
            Base::Signed(7),
            16, // channels
            4,  // classes
            2,  // examples
            0.5,
            42,
            TEST_PLANES,
        );

        assert!(result.hamming_accuracy >= 0.0 && result.hamming_accuracy <= 1.0);
        assert!(result.projection_accuracy >= 0.0 && result.projection_accuracy <= 1.0);
        assert!(result.two_stage_accuracy >= 0.0 && result.two_stage_accuracy <= 1.0);
        assert!(result.novelty_detection_rate >= 0.0);
        assert!(result.mean_residual_energy >= 0.0);
    }

    #[test]
    fn test_recognizer_many_classes() {
        // Stress test with 32 classes
        let d = 512;
        let num_classes = 32;
        let mut rng = SplitMix64::new(42);
        let channels = 64;
        let mut recognizer = Recognizer::with_planes(d, channels, TEST_PLANES, 99);

        let templates: Vec<Vec<i8>> = (0..num_classes)
            .map(|_| (0..d).map(|_| rng.gen_range_i8(-3, 3)).collect())
            .collect();

        for (i, t) in templates.iter().enumerate() {
            recognizer.register_class(i as u32, t.clone());
        }

        // Self-recognition should work even with many classes
        let mut correct = 0;
        for (i, t) in templates.iter().enumerate() {
            let result = recognizer.recognize_orthogonal(t);
            if result.best_class == i {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / num_classes as f32;
        assert!(
            accuracy > 0.8,
            "32-class self-recognition = {:.1}%, expected > 80%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_projection_batch() {
        let proj = Projector64K::with_planes(128, TEST_PLANES, 42);
        let templates: Vec<Vec<i8>> = (0..4)
            .map(|i| {
                (0..128)
                    .map(|j| ((i * 37 + j * 13) % 7 - 3) as i8)
                    .collect()
            })
            .collect();

        let fps = proj.project_batch(&templates);
        assert_eq!(fps.len(), 4);

        // Each fingerprint should match individual projection
        for (i, t) in templates.iter().enumerate() {
            let fp_single = proj.project_signed(t);
            assert_eq!(fps[i], fp_single);
        }
    }
}
