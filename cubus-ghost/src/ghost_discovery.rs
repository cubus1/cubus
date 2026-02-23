//! Ghost Discovery — What the Signed Organic Holograph Sees
//!
//! Store 52 signal-processing concepts in a signed holograph using organic
//! templates (3-layer: domain + tau-proximity + individual). Read back
//! everything. Concepts that read back without being stored — those are the
//! ghosts. They emerge from cross-talk between correlated templates.
//!
//! The tau-proximity layer creates a smooth manifold in template space:
//! concepts with nearby τ addresses share template structure regardless
//! of domain. This creates semantic bridges:
//!
//!   motor (0x40-0x47) ↔ ctrl (0x50-0x57) ↔ nav (0x60-0x66) ↔
//!   plan (0x70-0x75) ↔ mode (0x81-0x88) ↔ proc (0x92-0x97) ↔
//!   comm (0xA0-0xA5) ... gap ... sens (0xE0-0xE8)
//!
//! Sens is isolated in τ-space. The experiment reveals whether the holograph
//! respects this topology.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use numrus_nars::Base;

// ---------------------------------------------------------------------------
// 52 Signal-Processing Concepts
// ---------------------------------------------------------------------------

/// Named domain range constants for indexing into [`CONCEPTS`].
///
/// Each constant is the `Range<usize>` of indices in the `CONCEPTS` slice
/// that belong to that domain.  `ALL_DOMAINS` collects them all for
/// programmatic iteration and consistency checks.
pub mod domain {
    use std::ops::Range;

    pub const MODE: Range<usize> = 0..4;
    pub const NAV: Range<usize> = 4..10;
    pub const PLAN: Range<usize> = 10..16;
    pub const SENS: Range<usize> = 16..24;
    pub const CTRL: Range<usize> = 24..32;
    pub const MOTOR: Range<usize> = 32..40;
    pub const PROC: Range<usize> = 40..46;
    pub const COMM: Range<usize> = 46..52;

    pub const ALL_DOMAINS: &[(&str, Range<usize>)] = &[
        ("mode", MODE),
        ("nav", NAV),
        ("plan", PLAN),
        ("sens", SENS),
        ("ctrl", CTRL),
        ("motor", MOTOR),
        ("proc", PROC),
        ("comm", COMM),
    ];
}

/// A single concept from the signal-processing ontology.
#[derive(Clone, Debug)]
pub struct Concept {
    pub id: &'static str,
    pub tau: u8,
    pub name: &'static str,
    pub domain: &'static str,
}

/// 52 concepts across 8 signal-processing domains.
pub const CONCEPTS: &[Concept] = &[
    // mode domain (4) — Operating modes [0x80-0x8F]
    Concept {
        id: "mode.standby",
        tau: 0x87,
        name: "Standby",
        domain: "mode",
    },
    Concept {
        id: "mode.active",
        tau: 0x83,
        name: "Active",
        domain: "mode",
    },
    Concept {
        id: "mode.learn",
        tau: 0x81,
        name: "Learning",
        domain: "mode",
    },
    Concept {
        id: "mode.deploy",
        tau: 0x88,
        name: "Deploy",
        domain: "mode",
    },
    // nav domain (6) — Navigation [0x60-0x6F]
    Concept {
        id: "nav.waypoint",
        tau: 0x66,
        name: "Waypoint",
        domain: "nav",
    },
    Concept {
        id: "nav.heading",
        tau: 0x61,
        name: "Heading",
        domain: "nav",
    },
    Concept {
        id: "nav.altitude",
        tau: 0x62,
        name: "Altitude",
        domain: "nav",
    },
    Concept {
        id: "nav.velocity",
        tau: 0x63,
        name: "Velocity",
        domain: "nav",
    },
    Concept {
        id: "nav.bearing",
        tau: 0x64,
        name: "Bearing",
        domain: "nav",
    },
    Concept {
        id: "nav.terrain",
        tau: 0x65,
        name: "Terrain",
        domain: "nav",
    },
    // plan domain (6) — Planning / scheduling [0x70-0x7F]
    Concept {
        id: "plan.mission",
        tau: 0x70,
        name: "Mission",
        domain: "plan",
    },
    Concept {
        id: "plan.objective",
        tau: 0x71,
        name: "Objective",
        domain: "plan",
    },
    Concept {
        id: "plan.priority",
        tau: 0x72,
        name: "Priority",
        domain: "plan",
    },
    Concept {
        id: "plan.deadline",
        tau: 0x73,
        name: "Deadline",
        domain: "plan",
    },
    Concept {
        id: "plan.resource",
        tau: 0x74,
        name: "Resource",
        domain: "plan",
    },
    Concept {
        id: "plan.constraint",
        tau: 0x75,
        name: "Constraint",
        domain: "plan",
    },
    // sens domain (8) — Sensorimotor / perception [0xE0-0xEF]
    Concept {
        id: "sens.lidar",
        tau: 0xE0,
        name: "Lidar",
        domain: "sens",
    },
    Concept {
        id: "sens.radar",
        tau: 0xE8,
        name: "Radar",
        domain: "sens",
    },
    Concept {
        id: "sens.camera",
        tau: 0xE1,
        name: "Camera",
        domain: "sens",
    },
    Concept {
        id: "sens.imu",
        tau: 0xE2,
        name: "IMU",
        domain: "sens",
    },
    Concept {
        id: "sens.gps",
        tau: 0xE3,
        name: "GPS",
        domain: "sens",
    },
    Concept {
        id: "sens.sonar",
        tau: 0xE4,
        name: "Sonar",
        domain: "sens",
    },
    Concept {
        id: "sens.thermal",
        tau: 0xE5,
        name: "Thermal",
        domain: "sens",
    },
    Concept {
        id: "sens.magnetic",
        tau: 0xE7,
        name: "Magnetic",
        domain: "sens",
    },
    // ctrl domain (8) — Control signals [0x50-0x5F]
    Concept {
        id: "ctrl.throttle",
        tau: 0x50,
        name: "Throttle",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.steering",
        tau: 0x51,
        name: "Steering",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.brake",
        tau: 0x52,
        name: "Brake",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.pitch",
        tau: 0x53,
        name: "Pitch",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.roll",
        tau: 0x54,
        name: "Roll",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.yaw",
        tau: 0x55,
        name: "Yaw",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.gain",
        tau: 0x56,
        name: "Gain",
        domain: "ctrl",
    },
    Concept {
        id: "ctrl.damping",
        tau: 0x57,
        name: "Damping",
        domain: "ctrl",
    },
    // motor domain (8) — Actuator / motor commands [0x40-0x4F]
    Concept {
        id: "motor.torque",
        tau: 0x40,
        name: "Torque",
        domain: "motor",
    },
    Concept {
        id: "motor.rpm",
        tau: 0x41,
        name: "RPM",
        domain: "motor",
    },
    Concept {
        id: "motor.current",
        tau: 0x42,
        name: "Current",
        domain: "motor",
    },
    Concept {
        id: "motor.pwm",
        tau: 0x43,
        name: "PWM",
        domain: "motor",
    },
    Concept {
        id: "motor.position",
        tau: 0x44,
        name: "Position",
        domain: "motor",
    },
    Concept {
        id: "motor.force",
        tau: 0x45,
        name: "Force",
        domain: "motor",
    },
    Concept {
        id: "motor.encoder",
        tau: 0x46,
        name: "Encoder",
        domain: "motor",
    },
    Concept {
        id: "motor.servo",
        tau: 0x47,
        name: "Servo",
        domain: "motor",
    },
    // proc domain (6) — Signal processing [0x90-0x9F]
    Concept {
        id: "proc.fft",
        tau: 0x92,
        name: "FFT",
        domain: "proc",
    },
    Concept {
        id: "proc.filter",
        tau: 0x93,
        name: "Filter",
        domain: "proc",
    },
    Concept {
        id: "proc.kalman",
        tau: 0x94,
        name: "Kalman",
        domain: "proc",
    },
    Concept {
        id: "proc.fusion",
        tau: 0x95,
        name: "Fusion",
        domain: "proc",
    },
    Concept {
        id: "proc.detect",
        tau: 0x96,
        name: "Detection",
        domain: "proc",
    },
    Concept {
        id: "proc.classify",
        tau: 0x97,
        name: "Classify",
        domain: "proc",
    },
    // comm domain (6) — Communications [0xA0-0xAF]
    Concept {
        id: "comm.radio",
        tau: 0xA0,
        name: "Radio",
        domain: "comm",
    },
    Concept {
        id: "comm.protocol",
        tau: 0xA1,
        name: "Protocol",
        domain: "comm",
    },
    Concept {
        id: "comm.telemetry",
        tau: 0xA2,
        name: "Telemetry",
        domain: "comm",
    },
    Concept {
        id: "comm.mesh",
        tau: 0xA3,
        name: "Mesh",
        domain: "comm",
    },
    Concept {
        id: "comm.encrypt",
        tau: 0xA4,
        name: "Encrypt",
        domain: "comm",
    },
    Concept {
        id: "comm.sync",
        tau: 0xA5,
        name: "Sync",
        domain: "comm",
    },
];

/// Total concept count.
pub const K_TOTAL: usize = 52;

/// Alias derived from the slice length (compile-time equivalent of `CONCEPTS.len()`).
pub const CONCEPT_COUNT: usize = K_TOTAL;

/// Domain boundaries for grouping: (name, start_index, end_index).
pub const DOMAINS: &[(&str, usize, usize)] = &[
    ("mode", 0, 4),
    ("nav", 4, 10),
    ("plan", 10, 16),
    ("sens", 16, 24),
    ("ctrl", 24, 32),
    ("motor", 32, 40),
    ("proc", 40, 46),
    ("comm", 46, 52),
];

use numrus_core::SplitMix64;

// ---------------------------------------------------------------------------
// Organic template generation (3-layer)
// ---------------------------------------------------------------------------

/// Simple deterministic hash for seeding.
fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Generate a random vector from an RNG at the given base.
fn generate_from_rng(d: usize, base: Base, rng: &mut SplitMix64) -> Vec<i8> {
    let min = base.min_val();
    let max = base.max_val();
    (0..d).map(|_| rng.gen_range_i8(min, max)).collect()
}

/// Number of coarse τ bins for the proximity manifold.
/// 8 bins of width 32: motor+ctrl share bin 2, nav+plan share bin 3,
/// mode+proc share bin 4, comm in bin 5, sens isolated in bin 7.
const TAU_BINS: usize = 8;
/// Tau bin width: 256 / 8 = 32 tau values per bin.
const TAU_BIN_WIDTH: f64 = 256.0 / TAU_BINS as f64;

/// Generate a tau-proximity basis vector by interpolating between coarse bins.
///
/// The tau space (0x00-0xFF) is divided into 8 bins. Each bin has a
/// deterministic random vector. A concept's tau address produces a
/// linear blend of its enclosing bin and the next, creating a smooth
/// manifold where nearby tau values get similar vectors.
///
/// This creates cross-domain bridges:
///   motor (0x40) ↔ ctrl (0x50): distance 1 bin → high correlation
///   ctrl  (0x50) ↔ nav  (0x60): distance 1 bin → high correlation
///   comm  (0xA0) ↔ sens (0xE0): distance 4 bins → low correlation
fn generate_tau_basis(tau: u8, d: usize, base: Base) -> Vec<i8> {
    let tau_f = tau as f64;
    let bin_lo = (tau_f / TAU_BIN_WIDTH).floor() as usize;
    let bin_hi = (bin_lo + 1) % TAU_BINS;
    let frac = (tau_f - bin_lo as f64 * TAU_BIN_WIDTH) / TAU_BIN_WIDTH;

    // Each bin has a deterministic random vector seeded by bin index
    let seed_lo = 0xDEAD_BEEF_u64.wrapping_mul(bin_lo as u64 + 1);
    let seed_hi = 0xDEAD_BEEF_u64.wrapping_mul(bin_hi as u64 + 1);
    let mut rng_lo = SplitMix64::new(seed_lo);
    let mut rng_hi = SplitMix64::new(seed_hi);

    let min_val = base.min_val() as f64;
    let max_val = base.max_val() as f64;

    (0..d)
        .map(|_| {
            let lo = rng_lo.gen_range_i8(base.min_val(), base.max_val()) as f64;
            let hi = rng_hi.gen_range_i8(base.min_val(), base.max_val()) as f64;
            let blended = lo * (1.0 - frac) + hi * frac;
            blended.round().clamp(min_val, max_val) as i8
        })
        .collect()
}

/// Generate an organic 3-layer template for one concept.
///
/// Layer 1 — Domain basis (weight: domain_w):
///   Shared across all concepts in the same domain.
///   Deterministic from domain name hash.
///
/// Layer 2 — Tau proximity (weight: tau_w):
///   Smooth manifold interpolated from coarse τ bins.
///   Creates cross-domain bridges between nearby τ addresses.
///   motor(0x40) and ctrl(0x50) share structure.
///   sens(0xE0) is isolated from everything else.
///
/// Layer 3 — Individual noise (weight: 1 - domain_w - tau_w):
///   Unique to each concept. Provides orthogonality for recovery.
pub fn generate_organic_template(
    concept: &Concept,
    d: usize,
    base: Base,
    domain_w: f32,
    tau_w: f32,
) -> Vec<i8> {
    let indiv_w = 1.0 - domain_w - tau_w;

    // Layer 1: Domain basis
    let domain_seed = hash_string(concept.domain);
    let mut domain_rng = SplitMix64::new(domain_seed);
    let domain_basis = generate_from_rng(d, base, &mut domain_rng);

    // Layer 2: Tau proximity manifold
    let tau_basis = generate_tau_basis(concept.tau, d, base);

    // Layer 3: Individual component
    let concept_seed = hash_string(concept.id);
    let mut concept_rng = SplitMix64::new(concept_seed);
    let individual = generate_from_rng(d, base, &mut concept_rng);

    // Blend
    let min_val = base.min_val() as f32;
    let max_val = base.max_val() as f32;

    (0..d)
        .map(|j| {
            let blended = domain_w * domain_basis[j] as f32
                + tau_w * tau_basis[j] as f32
                + indiv_w * individual[j] as f32;
            blended.round().clamp(min_val, max_val) as i8
        })
        .collect()
}

/// Generate all 52 organic templates.
pub fn generate_all_organic_templates(
    d: usize,
    base: Base,
    domain_w: f32,
    tau_w: f32,
) -> Vec<Vec<i8>> {
    CONCEPTS
        .iter()
        .map(|c| generate_organic_template(c, d, base, domain_w, tau_w))
        .collect()
}

/// Default weights: 35% domain, 35% tau, 30% individual.
pub const DEFAULT_DOMAIN_W: f32 = 0.35;
pub const DEFAULT_TAU_W: f32 = 0.35;

// ---------------------------------------------------------------------------
// Readback result
// ---------------------------------------------------------------------------

/// Readback result for one concept.
#[derive(Clone, Debug)]
pub struct ConceptReadback {
    pub index: usize,
    pub id: &'static str,
    pub name: &'static str,
    pub domain: &'static str,
    pub tau: u8,
    pub was_stored: bool,
    pub original_amplitude: f32,
    /// Signed readback coefficient (dot product / template norm^2).
    pub readback: f32,
}

// ---------------------------------------------------------------------------
// Ghost Holograph (signed only)
// ---------------------------------------------------------------------------

/// Signed organic holograph for ghost discovery.
///
/// Uses Signed(7) base with organic 3-layer templates.
/// Ghost = readback of unstored concept from signed cross-talk.
/// Positive readback: constructive interference (correlated templates).
/// Negative readback: destructive interference (anti-correlated).
/// Near zero: orthogonal (no ghost).
pub struct GhostHolograph {
    pub d: usize,
    pub base: Base,
    pub templates: Vec<Vec<i8>>,
    pub container: Vec<i8>,
    pub amplitudes: Vec<f32>,
    pub stored_indices: Vec<usize>,
}

impl GhostHolograph {
    /// Create with the given dimensionality.
    pub fn new(d: usize) -> Self {
        let base = Base::Signed(7);
        let templates = generate_all_organic_templates(d, base, DEFAULT_DOMAIN_W, DEFAULT_TAU_W);

        Self {
            d,
            base,
            templates,
            container: vec![0i8; d],
            amplitudes: vec![0.0; K_TOTAL],
            stored_indices: Vec::new(),
        }
    }

    /// Store a subset of concepts at given amplitudes.
    pub fn store(&mut self, indices: &[usize], amplitudes: &[f32]) {
        assert_eq!(indices.len(), amplitudes.len());
        let half = (7_i8 / 2) as f32; // 3.0

        for (&idx, &amp) in indices.iter().zip(amplitudes.iter()) {
            self.amplitudes[idx] = amp;
            self.stored_indices.push(idx);

            for j in 0..self.d {
                let write = amp * self.templates[idx][j] as f32;
                let new_val = self.container[j] as f32 + write;
                self.container[j] = new_val.round().clamp(-half, half) as i8;
            }
        }
    }

    /// Read back ALL 52 concepts.
    pub fn read_all(&self) -> Vec<ConceptReadback> {
        (0..K_TOTAL)
            .map(|idx| {
                let concept = &CONCEPTS[idx];
                let was_stored = self.stored_indices.contains(&idx);
                let readback = self.read_coefficient(&self.templates[idx]);

                ConceptReadback {
                    index: idx,
                    id: concept.id,
                    name: concept.name,
                    domain: concept.domain,
                    tau: concept.tau,
                    was_stored,
                    original_amplitude: self.amplitudes[idx],
                    readback,
                }
            })
            .collect()
    }

    /// Read a single coefficient: dot(container, template) / ||template||^2.
    fn read_coefficient(&self, template: &[i8]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm = 0.0f64;
        for j in 0..self.d {
            dot += self.container[j] as f64 * template[j] as f64;
            norm += template[j] as f64 * template[j] as f64;
        }
        if norm > 1e-10 {
            (dot / norm) as f32
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario 1: Sensor suite only.
///
/// Store all 8 sens concepts at amplitude 1.0.
/// Sens is isolated in τ-space (0xE0-0xE8, far from all other domains).
/// Question: does the holograph confirm this isolation, or leak anyway?
pub fn scenario_sens_only(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices: Vec<usize> = (16..24).collect();
    let amplitudes = vec![1.0f32; 8];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 2: Autonomous navigation state.
///
/// Store: mode.active (0.9), motor.rpm (1.0), sens.imu (0.8),
///        nav.altitude (0.9), motor.encoder (0.7)
///
/// Cross-domain experience. The question: does the signed holograph
/// discover the GESTALT these concepts imply?
pub fn scenario_navigation(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        1,  // mode.active
        33, // motor.rpm
        19, // sens.imu
        6,  // nav.altitude
        38, // motor.encoder
    ];
    let amplitudes = vec![0.9, 1.0, 0.8, 0.9, 0.7];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 3: Conflicting control signals.
///
/// Store: mode.learn (1.0), sens.camera (0.8),
///        plan.constraint (0.9), sens.gps (0.7),
///        proc.fft (1.0), proc.kalman (0.8)
///
/// Opposing control demands. Signed holograph: Auslöschung (cancellation).
/// What survives? What ghosts emerge from the interference pattern?
pub fn scenario_conflict(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        2,  // mode.learn
        18, // sens.camera
        15, // plan.constraint
        20, // sens.gps
        40, // proc.fft
        42, // proc.kalman
    ];
    let amplitudes = vec![1.0, 0.8, 0.9, 0.7, 1.0, 0.8];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 4: Adaptation / reconfiguration state.
///
/// Store: motor.torque (1.0), ctrl.pitch (0.9),
///        proc.fusion (0.8), proc.detect (0.7),
///        plan.deadline (0.6)
///
/// Open, exploratory state. What does the holograph see as the
/// destination? Where does the τ-topology lead?
pub fn scenario_adaptation(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        32, // motor.torque
        27, // ctrl.pitch
        43, // proc.fusion
        44, // proc.detect
        13, // plan.deadline
    ];
    let amplitudes = vec![1.0, 0.9, 0.8, 0.7, 0.6];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 5: Everything at once.
///
/// Store all 52 concepts at amplitude 1.0.
/// Maximum interference. Which concepts get amplified? Suppressed?
pub fn scenario_full_load(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices: Vec<usize> = (0..K_TOTAL).collect();
    let amplitudes = vec![1.0f32; K_TOTAL];
    h.store(&indices, &amplitudes);
    h.read_all()
}

// ---------------------------------------------------------------------------
// Output — Human Readable
// ---------------------------------------------------------------------------

/// Print the ghost discovery table for a scenario.
///
/// Ghosts: unstored concepts with |readback| > threshold.
///   GHOST+  = unstored, positive readback (constructive cross-talk)
///   GHOST-  = unstored, negative readback (destructive cross-talk)
///   recov   = stored, readback near original amplitude
///   ampl    = stored, readback > original (boosted by neighbors)
///   supp    = stored, readback < original (suppressed by interference)
pub fn print_ghost_table(scenario_name: &str, results: &[ConceptReadback], ghost_threshold: f32) {
    println!("\n{}", "=".repeat(72));
    println!("  GHOST DISCOVERY: {}", scenario_name);
    println!("{}", "=".repeat(72));
    println!(
        "  {:20} {:6} {:7} {:>4} {:>8} {:>8} {:>8}",
        "Concept", "Domain", "Stored", "tau", "Ampl", "Read", "Verdict"
    );
    println!("{}", "-".repeat(72));

    // Sort: ghosts first (by |readback| desc for unstored), then stored (by readback desc)
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| {
        let a_ghost = !a.was_stored && a.readback.abs() > ghost_threshold;
        let b_ghost = !b.was_stored && b.readback.abs() > ghost_threshold;
        match (a_ghost, b_ghost) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b
                .readback
                .abs()
                .partial_cmp(&a.readback.abs())
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    for r in &sorted {
        let stored_str = if r.was_stored { "YES" } else { "no" };
        let ampl_str = if r.was_stored {
            format!("{:>+8.3}", r.original_amplitude)
        } else {
            "       -".to_string()
        };

        let verdict = if !r.was_stored && r.readback.abs() > ghost_threshold {
            if r.readback > 0.0 {
                "GHOST+"
            } else {
                "GHOST-"
            }
        } else if r.was_stored {
            let ratio = r.readback / r.original_amplitude.max(0.001);
            if ratio > 1.1 {
                "ampl"
            } else if ratio < 0.5 {
                "supp"
            } else {
                "recov"
            }
        } else {
            ""
        };

        println!(
            "  {:20} {:6} {:7} 0x{:02X} {} {:>+8.3} {:>8}",
            r.name, r.domain, stored_str, r.tau, ampl_str, r.readback, verdict
        );
    }

    println!("{}", "-".repeat(72));

    // Summary
    let ghosts: Vec<&ConceptReadback> = results
        .iter()
        .filter(|r| !r.was_stored && r.readback.abs() > ghost_threshold)
        .collect();

    if ghosts.is_empty() {
        println!("  No ghosts above threshold {:.3}.", ghost_threshold);
    } else {
        println!(
            "  {} ghost(s) above |{:.3}|:",
            ghosts.len(),
            ghost_threshold
        );
        for &(domain_name, _, _) in DOMAINS {
            let domain_ghosts: Vec<&&ConceptReadback> =
                ghosts.iter().filter(|g| g.domain == domain_name).collect();
            if !domain_ghosts.is_empty() {
                let signals: Vec<String> = domain_ghosts
                    .iter()
                    .map(|g| format!("{} ({:+.3})", g.name, g.readback))
                    .collect();
                println!("    {}: {}", domain_name, signals.join(", "));
            }
        }
    }
    println!();
}

/// Print a domain-to-domain ghost matrix.
///
/// Each cell: average readback of unstored concepts in the column domain
/// when only concepts from the row domain are stored. Signed, so positive
/// means constructive cross-talk, negative means destructive.
pub fn ghost_matrix(d: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0f32; 8]; 8];

    for (src_idx, &(_src_name, src_start, src_end)) in DOMAINS.iter().enumerate() {
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (src_start..src_end).collect();
        let amplitudes = vec![1.0f32; indices.len()];
        h.store(&indices, &amplitudes);

        let results = h.read_all();

        for (dst_idx, &(_dst_name, dst_start, dst_end)) in DOMAINS.iter().enumerate() {
            if src_idx == dst_idx {
                continue;
            }
            let avg: f32 = results[dst_start..dst_end]
                .iter()
                .map(|r| r.readback)
                .sum::<f32>()
                / (dst_end - dst_start) as f32;
            matrix[src_idx][dst_idx] = avg;
        }
    }

    // Print with signed values
    println!("\n{}", "=".repeat(72));
    println!("  GHOST MATRIX: avg signed readback (row stored -> col ghost)");
    println!("  Positive = constructive cross-talk. Negative = destructive.");
    println!("{}", "=".repeat(72));
    print!("  {:8}", "stored>");
    for &(name, _, _) in DOMAINS {
        print!(" {:>7}", name);
    }
    println!();
    println!("{}", "-".repeat(72));

    for (src_idx, &(src_name, _, _)) in DOMAINS.iter().enumerate() {
        print!("  {:8}", src_name);
        for dst_idx in 0..8 {
            if src_idx == dst_idx {
                print!("    ---");
            } else {
                print!(" {:>+7.4}", matrix[src_idx][dst_idx]);
            }
        }
        println!();
    }
    println!("{}", "=".repeat(72));

    matrix
}

/// Run all scenarios at multiple dimensionalities.
///
/// At low D: ghosts are strong (limited orthogonality = more cross-talk).
/// At high D: ghosts fade (templates become more orthogonal).
/// The transition reveals the "noise floor" of semantic inference.
pub fn ghost_dimensionality_sweep() {
    let dims = [256, 512, 1024, 2048, 4096, 8192, 16384];
    let threshold = 0.02;

    println!("\n{}", "=".repeat(72));
    println!(
        "  GHOST COUNT vs DIMENSIONALITY (threshold = {:.3})",
        threshold
    );
    println!("{}", "=".repeat(72));
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "D", "Sens", "NavSt", "Conflict", "Adapt", "Full"
    );
    println!("{}", "-".repeat(72));

    for &d in &dims {
        let count_ghosts = |results: &[ConceptReadback]| -> usize {
            results
                .iter()
                .filter(|r| !r.was_stored && r.readback.abs() > threshold)
                .count()
        };

        let sens = scenario_sens_only(d);
        let nav = scenario_navigation(d);
        let conflict = scenario_conflict(d);
        let adapt = scenario_adaptation(d);
        let full = scenario_full_load(d);

        println!(
            "  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            d,
            count_ghosts(&sens),
            count_ghosts(&nav),
            count_ghosts(&conflict),
            count_ghosts(&adapt),
            count_ghosts(&full)
        );
    }

    println!("{}", "=".repeat(72));

    // Also show max |ghost| signal at each D
    println!("\n{}", "=".repeat(72));
    println!("  MAX |GHOST SIGNAL| vs DIMENSIONALITY");
    println!("{}", "=".repeat(72));
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "D", "Sens", "NavSt", "Conflict", "Adapt", "Full"
    );
    println!("{}", "-".repeat(72));

    for &d in &dims {
        let max_ghost = |results: &[ConceptReadback]| -> f32 {
            results
                .iter()
                .filter(|r| !r.was_stored)
                .map(|r| r.readback.abs())
                .fold(0.0f32, f32::max)
        };

        let sens = scenario_sens_only(d);
        let nav = scenario_navigation(d);
        let conflict = scenario_conflict(d);
        let adapt = scenario_adaptation(d);
        let full = scenario_full_load(d);

        println!(
            "  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            d,
            max_ghost(&sens),
            max_ghost(&nav),
            max_ghost(&conflict),
            max_ghost(&adapt),
            max_ghost(&full)
        );
    }
    println!("{}", "=".repeat(72));
}

/// τ-topology analysis: show template correlations along the τ chain.
///
/// Measures pairwise correlation between all 52 concepts, grouped by
/// τ distance. Reveals whether the organic templates actually encode
/// the intended proximity structure.
pub fn tau_topology_analysis(d: usize) {
    let base = Base::Signed(7);
    let templates = generate_all_organic_templates(d, base, DEFAULT_DOMAIN_W, DEFAULT_TAU_W);

    println!("\n{}", "=".repeat(72));
    println!(
        "  TAU TOPOLOGY: pairwise correlation vs tau distance (D={})",
        d
    );
    println!("{}", "=".repeat(72));

    // Bucket correlations by τ distance
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); 16];

    for i in 0..K_TOTAL {
        for j in (i + 1)..K_TOTAL {
            let tau_dist = (CONCEPTS[i].tau as i16 - CONCEPTS[j].tau as i16).unsigned_abs();
            let bin_dist = (tau_dist as usize) / TAU_BINS;
            let bin_dist = bin_dist.min(15);

            let corr = pearson_correlation(&templates[i], &templates[j]);
            buckets[bin_dist].push(corr);
        }
    }

    println!(
        "  {:>10}  {:>8}  {:>8}  {:>6}  {:>10}",
        "tau_dist", "avg_corr", "max_corr", "count", "interpretation"
    );
    println!("{}", "-".repeat(72));

    for (dist, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let avg: f32 = bucket.iter().sum::<f32>() / bucket.len() as f32;
        let max: f32 = bucket.iter().cloned().fold(f32::MIN, f32::max);
        let interp = if avg > 0.3 {
            "strong link"
        } else if avg > 0.1 {
            "weak link"
        } else if avg > 0.02 {
            "faint"
        } else {
            "orthogonal"
        };
        println!(
            "  {:>10}  {:>+8.4}  {:>+8.4}  {:>6}  {:>10}",
            format!("{}..{}", dist * TAU_BINS, (dist + 1) * TAU_BINS - 1),
            avg,
            max,
            bucket.len(),
            interp
        );
    }
    println!("{}", "=".repeat(72));

    // Cross-domain pairs of interest
    println!("\n  Named cross-domain correlations:");
    let pairs = [
        (32, 24, "motor.torque ↔ ctrl.throttle"), // tau 0x40 ↔ 0x50
        (33, 25, "motor.rpm ↔ ctrl.steering"),    // tau 0x41 ↔ 0x51
        (24, 5, "ctrl.throttle ↔ nav.heading"),   // tau 0x50 ↔ 0x61
        (10, 1, "plan.mission ↔ mode.active"),    // tau 0x70 ↔ 0x83
        (40, 46, "proc.fft ↔ comm.radio"),        // tau 0x92 ↔ 0xA0
        (16, 32, "sens.lidar ↔ motor.torque"),    // tau 0xE0 ↔ 0x40 (far!)
        (17, 40, "sens.radar ↔ proc.fft"),        // tau 0xE8 ↔ 0x92 (far!)
        (33, 6, "motor.rpm ↔ nav.altitude"),      // tau 0x41 ↔ 0x62
    ];

    for (i, j, label) in pairs {
        let corr = pearson_correlation(&templates[i], &templates[j]);
        let tau_dist = (CONCEPTS[i].tau as i16 - CONCEPTS[j].tau as i16).unsigned_abs();
        println!("    {:>+6.3}  tau_dist={:>3}  {}", corr, tau_dist, label);
    }
    println!();
}

/// The complete ghost discovery experiment.
pub fn run_ghost_discovery() {
    let d = 4096;
    let threshold = 0.02;

    println!("\n{}", "=".repeat(72));
    println!("  GHOST DISCOVERY EXPERIMENT");
    println!("  52 signal-processing concepts");
    println!("  Signed(7), organic 3-layer templates at D={}", d);
    println!(
        "  Weights: domain={:.0}%, tau={:.0}%, individual={:.0}%",
        DEFAULT_DOMAIN_W * 100.0,
        DEFAULT_TAU_W * 100.0,
        (1.0 - DEFAULT_DOMAIN_W - DEFAULT_TAU_W) * 100.0
    );
    println!("{}", "=".repeat(72));

    // τ-topology analysis first — verify the template structure
    tau_topology_analysis(d);

    // Scenario tables
    let results = scenario_sens_only(d);
    print_ghost_table("Sensor Suite (isolated in tau-space)", &results, threshold);

    let results = scenario_navigation(d);
    print_ghost_table("Autonomous Navigation", &results, threshold);

    let results = scenario_conflict(d);
    print_ghost_table("Conflicting Controls", &results, threshold);

    let results = scenario_adaptation(d);
    print_ghost_table("Adaptation State", &results, threshold);

    let results = scenario_full_load(d);
    print_ghost_table("Full Load (all 52)", &results, threshold);

    // Domain matrices
    let _ = ghost_matrix(d);

    // Dimensionality sweep
    ghost_dimensionality_sweep();

    println!("\n{}", "=".repeat(72));
    println!("  END OF EXPERIMENT");
    println!("  Name the findings. Don't theorize. Describe what you see.");
    println!("{}", "=".repeat(72));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pearson correlation between two i8 vectors.
fn pearson_correlation(a: &[i8], b: &[i8]) -> f32 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom > 1e-10 {
        (cov / denom) as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_determinism() {
        let c = &CONCEPTS[0];
        let t1 = generate_organic_template(c, 1024, Base::Signed(7), 0.35, 0.35);
        let t2 = generate_organic_template(c, 1024, Base::Signed(7), 0.35, 0.35);
        assert_eq!(t1, t2, "Templates must be deterministic");
    }

    #[test]
    fn test_same_domain_correlation() {
        let d = 4096;
        let base = Base::Signed(7);
        // sens domain: indices 16..24
        let t_lidar = generate_organic_template(&CONCEPTS[16], d, base, 0.35, 0.35);
        let t_camera = generate_organic_template(&CONCEPTS[18], d, base, 0.35, 0.35);
        let corr = pearson_correlation(&t_lidar, &t_camera);
        assert!(
            corr > 0.15,
            "Same-domain correlation should be > 0.15, got {}",
            corr
        );
    }

    #[test]
    fn test_tau_proximity_correlation() {
        let d = 4096;
        let base = Base::Signed(7);
        // motor.torque (tau=0x40) and ctrl.throttle (tau=0x50) — nearby tau
        let t_motor = generate_organic_template(&CONCEPTS[32], d, base, 0.35, 0.35);
        let t_ctrl = generate_organic_template(&CONCEPTS[24], d, base, 0.35, 0.35);
        let corr_near = pearson_correlation(&t_motor, &t_ctrl);

        // motor.torque (tau=0x40) and sens.lidar (tau=0xE0) — distant tau
        let t_sens = generate_organic_template(&CONCEPTS[16], d, base, 0.35, 0.35);
        let corr_far = pearson_correlation(&t_motor, &t_sens);

        assert!(
            corr_near > corr_far,
            "Nearby tau should correlate more: near={}, far={}",
            corr_near,
            corr_far
        );
    }

    #[test]
    fn test_sens_isolation() {
        let d = 4096;
        let base = Base::Signed(7);
        // sens.camera (tau=0xE1) vs proc.fft (tau=0x92) — far tau, different domain
        let t_sens = generate_organic_template(&CONCEPTS[18], d, base, 0.35, 0.35);
        let t_proc = generate_organic_template(&CONCEPTS[40], d, base, 0.35, 0.35);
        let corr = pearson_correlation(&t_sens, &t_proc).abs();
        assert!(
            corr < 0.10,
            "Sens-Proc cross-domain should be near zero, got {}",
            corr
        );
    }

    #[test]
    fn test_all_52_templates_generated() {
        let templates = generate_all_organic_templates(1024, Base::Signed(7), 0.35, 0.35);
        assert_eq!(templates.len(), K_TOTAL);
        for t in &templates {
            assert_eq!(t.len(), 1024);
        }
    }

    #[test]
    fn test_single_concept_recovery() {
        let d = 8192;
        let mut h = GhostHolograph::new(d);
        h.store(&[0], &[1.0]);
        let results = h.read_all();
        let r = &results[0];
        assert!(r.was_stored);
        assert!(
            (r.readback - 1.0).abs() < 0.2,
            "Recovery should be near 1.0, got {}",
            r.readback
        );
    }

    #[test]
    fn test_store_8_recovery() {
        let d = 8192;
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (16..24).collect();
        let amplitudes = vec![1.0f32; 8];
        h.store(&indices, &amplitudes);
        let results = h.read_all();

        for &idx in &indices {
            let r = &results[idx];
            assert!(
                r.readback > 0.3,
                "Recovery for {} should be > 0.3, got {}",
                r.name,
                r.readback
            );
        }
    }

    #[test]
    fn test_container_values_in_range() {
        let d = 4096;
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (0..K_TOTAL).collect();
        let amplitudes = vec![1.0f32; K_TOTAL];
        h.store(&indices, &amplitudes);

        for &v in &h.container {
            assert!((-3..=3).contains(&v), "Container out of range: {}", v);
        }
    }

    #[test]
    fn test_scenario_sens_counts() {
        let results = scenario_sens_only(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 8);
        assert_eq!(results.len() - stored, 44);
    }

    #[test]
    fn test_scenario_navigation_counts() {
        let results = scenario_navigation(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 5);
    }

    #[test]
    fn test_scenario_conflict_counts() {
        let results = scenario_conflict(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 6);
    }

    #[test]
    fn test_scenario_adaptation_counts() {
        let results = scenario_adaptation(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 5);
    }

    #[test]
    fn test_scenario_full_load_counts() {
        let results = scenario_full_load(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 52);
    }

    #[test]
    fn test_stored_concepts_positive_readback() {
        let results = scenario_sens_only(4096);
        for r in &results {
            if r.was_stored {
                assert!(
                    r.readback > 0.0,
                    "Stored {} should have positive readback, got {}",
                    r.name,
                    r.readback
                );
            }
        }
    }

    #[test]
    fn test_unstored_near_zero_at_high_d() {
        let d = 8192;
        let results = scenario_sens_only(d);
        // Check a concept far in tau-space: motor.torque (tau=0x40, far from sens 0xE0)
        let torque = &results[32];
        assert!(!torque.was_stored);
        assert!(
            torque.readback.abs() < 0.15,
            "Far unstored should be near 0 at D={}, got {}",
            d,
            torque.readback
        );
    }

    #[test]
    fn test_ghost_count_decreases_with_d() {
        let threshold = 0.05;
        let count = |d: usize| -> usize {
            scenario_sens_only(d)
                .iter()
                .filter(|r| !r.was_stored && r.readback.abs() > threshold)
                .count()
        };
        let low = count(256);
        let high = count(8192);
        assert!(
            low >= high,
            "Ghost count should decrease with D: low_d={}, high_d={}",
            low,
            high
        );
    }

    #[test]
    fn test_ghost_matrix_dimensions() {
        let matrix = ghost_matrix(512);
        assert_eq!(matrix.len(), 8);
        for row in &matrix {
            assert_eq!(row.len(), 8);
        }
        for i in 0..8 {
            assert_eq!(matrix[i][i], 0.0, "Diagonal should be zero");
        }
    }

    #[test]
    fn test_print_ghost_table_no_panic() {
        let results = scenario_sens_only(512);
        print_ghost_table("Test", &results, 0.02);
    }

    #[test]
    fn test_concept_count() {
        assert_eq!(CONCEPTS.len(), K_TOTAL);
    }

    #[test]
    fn test_domain_boundaries_cover_all() {
        let mut covered = 0;
        for &(_, start, end) in DOMAINS {
            assert!(start <= end);
            assert_eq!(start, covered);
            covered = end;
        }
        assert_eq!(covered, K_TOTAL);
    }

    #[test]
    fn test_domain_ranges_consistent() {
        for (name, range) in domain::ALL_DOMAINS {
            for idx in range.clone() {
                assert_eq!(
                    CONCEPTS[idx].domain, *name,
                    "CONCEPTS[{}] domain '{}' doesn't match expected '{}'",
                    idx, CONCEPTS[idx].domain, name
                );
            }
        }
        let total: usize = domain::ALL_DOMAINS.iter().map(|(_, r)| r.len()).sum();
        assert_eq!(total, CONCEPT_COUNT);
    }
}
