#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! # cubus
//!
//! Phase-space holographic operations for CogRecord v3.
//!
//! This crate implements the phase-space layer that sits on top of the
//! binary CogRecord engine in `numrus-rs`. While binary containers
//! (XOR bind, Hamming distance) provide hash-table-like exact matching,
//! phase containers (ADD bind, Wasserstein/circular distance) provide
//! genuine spatial navigation.
//!
//! ## The Hybrid Architecture
//!
//! | Container | Mode   | Bind     | Distance    | Use                  |
//! |-----------|--------|----------|-------------|----------------------|
//! | META      | Binary | XOR      | Hamming     | Fast rejection       |
//! | CAM       | Phase  | ADD      | Wasserstein | Spatial navigation   |
//! | BTREE     | Binary | XOR      | Hamming     | Graph structure      |
//! | EMBED     | Phase  | ADD      | Circular    | Dense similarity     |
//!
//! ## Operations
//!
//! - `phase_bind_i8` / `phase_unbind_i8` — reversible phase-space binding
//! - `wasserstein_sorted_i8` — O(N) Earth Mover's distance on sorted vectors
//! - `circular_distance_i8` — wrap-around distance for unsorted phase vectors
//! - `phase_histogram_16` — 16-bin compact spatial address
//! - `phase_bundle_circular` — correct circular mean bundling
//! - `project_5d_to_phase` / `recover_5d_from_phase` — spatial coordinate encoding
//! - `sort_phase_vector` / `unsort_phase_vector` — write-time preparation
//!
//! ## Carrier Model (alternative encoding)
//!
//! - `carrier_encode` / `carrier_decode` — frequency-domain concept encoding
//! - `carrier_bundle` — waveform addition (32 VPADDB vs ~500 trig instructions)
//! - `carrier_distance_l1` / `carrier_correlation` — waveform similarity
//! - `carrier_spectrum` / `spectral_distance` — frequency fingerprinting
//! - `CarrierRecord` — hybrid binary + carrier containers

pub mod carrier;
pub mod cogrecord_v3;
pub mod delta_layer;
pub mod focus;
pub mod holograph;
pub mod phase;

pub use phase::{
    circular_distance_i8, generate_5d_basis, histogram_l1_distance, phase_bind_i8,
    phase_bind_i8_inplace, phase_bundle_approximate, phase_bundle_circular, phase_histogram_16,
    phase_inverse_i8, phase_unbind_i8, project_5d_to_phase, recover_5d_from_phase,
    sort_phase_vector, unsort_phase_vector, wasserstein_search_adaptive, wasserstein_sorted_i8,
};

pub use cogrecord_v3::{CogRecordV3, HybridDistances, HybridThresholds, CONTAINER_BYTES};

pub use carrier::{
    carrier_bundle, carrier_correlation, carrier_decode, carrier_distance_l1, carrier_encode,
    carrier_spectrum, spectral_distance, CarrierBasis, CarrierDistances, CarrierRecord,
    CarrierThresholds, CARRIER_AMPLITUDE, CARRIER_FREQUENCIES,
};

pub use focus::{
    concept_to_focus, focus_add, focus_add_materialized, focus_bind_binary, focus_bind_phase,
    focus_carrier_encode, focus_delta, focus_hamming, focus_l1, focus_read, focus_sub,
    focus_unbind_phase, focus_xor, focus_xor_auto, focus_xor_materialized, materialize_focus_mask,
    pack_focus, unpack_focus, CompactDelta, FocusDensity, FocusRegistry, FOCUS_DIM_X, FOCUS_DIM_Y,
    FOCUS_DIM_Z,
};

pub use delta_layer::{DeltaLayer, LayerStack};

pub use holograph::{
    adapt_sigma,
    anti_hebbian_update,
    apply_migrations,
    bootstrap_read,
    bootstrap_write,
    clean_if_needed,
    crystallize_archetypes,
    crystallize_from_superposition,
    delta_cube_read_gabor,
    delta_cube_recover_phase,
    delta_cube_recover_xor,
    delta_cube_sub,
    delta_cube_write_gabor,
    // Delta cube operations
    delta_cube_xor,
    gabor_read,
    gabor_write,
    get_container_mode,
    // Learning
    hebbian_update,
    incremental_axis_update,
    migrate_carrier_to_gabor,
    orthogonal_project,
    ready_for_crystallization,
    residual_energy,
    set_container_mode,
    spatial_bind,
    spatial_bind_i8,
    spatial_unbind,
    spatial_unbind_i8,
    // BLAS acceleration
    spectral_analysis_blas,
    AxisCrystallizer,
    // Container lifecycle and migration
    ContainerMode,
    // Co-occurrence and axis discovery
    CooccurrenceMatrix,
    // Archetype detection
    FastArchetypeDetector,
    GaborBatch,
    // Core Gabor wavelet types and operations
    GaussianLUT,
    MigrationResult,
    // Overlay / blackboard layer
    Overlay,
    // Spatial transforms
    SpatialTransform,
    // Spectral analysis and cleaning
    SpectralMap,
    WaveletTemplate,
    MODE_BYTE_OFFSET,
};
