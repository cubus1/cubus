// Oracle/sweep functions carry many configuration parameters by design.
// Numeric loops on projections/amplitudes/scores use index access for clarity.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

//! # cubus-oracle
//!
//! Three-temperature holographic oracle with exhaustive capacity sweep.
//!
//! This crate implements two things:
//!
//! 1. **Capacity sweep** — test every combination of D, base, signing, axes, K,
//!    and bind depth to find the sweet spot for holographic storage.
//!
//! 2. **Three-temperature oracle** — one oracle per entity with hot/warm/cold tiers,
//!    overexposure-triggered flush, and coefficient-as-canonical-storage.
//!
//! The oracle is the thinking/storage tier. The 8KB CogRecordV3 in `cubus`
//! remains as the fast-rejection search tier.

pub mod linalg;
pub mod oracle;
pub mod recognize;
pub mod sweep;

pub use linalg::{cholesky_solve, condition_number, downsample_to_base, upsample_to_f32};

pub use sweep::{
    bind, bind_deep, bundle, generate_template, generate_templates, measure_bell_coefficient,
    measure_recovery, measure_recovery_multiaxis, run_sweep, AxisResult, Base, MultiAxisResult,
    RecoveryResult, AXES, BASES, BIND_DEPTHS, BUNDLE_SIZES, DIMS,
};

pub use oracle::{FlushAction, MaterializedHolograph, Oracle, Temperature, TemplateLibrary};

// Re-export from numrus-nars
pub use numrus_nars as nars;
pub use numrus_nars::{
    bf16_granger_causal_map,
    bf16_granger_causal_scan,
    bf16_reverse_trace,
    classify_learning_event,
    find_similar_pairs,
    forward_bind,
    granger_scan,
    granger_signal,
    reverse_trace,
    reverse_unbind,
    unbind,
    BF16CausalTrace,
    BF16Entity,
    BF16LearningEvent,
    BF16TraceStep,
    CausalFeatureMap,
    CausalTrace,
    Entity as NarsEntity,
    LearningInterpretation,
    Role as NarsRole,
    SimilarPair,
    TraceStep,
};

// Re-export from numrus-substrate
pub use numrus_substrate as substrate;
pub use numrus_substrate::{
    organic_flush, organic_read, organic_write, organic_write_f32, receptivity, AbsorptionTracker,
    FlushResult, MultiResPattern, OrganicWAL, PlasticityTracker, WriteResult, XTransPattern,
};
pub use numrus_substrate::FlushAction as OrganicFlushAction;

pub use recognize::{
    hamming_64k, hamming_similarity_64k, print_recognition_results, run_recognition,
    run_recognition_experiment, run_recognition_sweep, run_recognition_sweep_fast,
    ExperimentResult, Projector64K, RecognitionMethod, RecognitionResult, Recognizer,
};
