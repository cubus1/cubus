#![allow(clippy::needless_range_loop)]

//! # cubus-lance
//!
//! Optional Arrow/Lance/DataFusion bridge for the cubus ecosystem.
//!
//! This crate is **not** a dependency of any other cubus crate.
//! Enable it only if you need Arrow interop, Lance dataset I/O,
//! or DataFusion cascade scanning.
//!
//! ## Features
//!
//! - `arrow` (default) — Zero-copy NumArray <-> Arrow array conversions + DataFusion cascade scan
//! - `datafusion` — DataFusion 51 (implies `arrow`)
//! - `lance` — CogRecord <-> Lance dataset read/write (implies `arrow`)
//!
//! ## Example
//!
//! ```rust,ignore
//! use cubus_lance::IntoArrow;
//! use numrus_rs::NumArrayF32;
//!
//! let arr = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
//! let arrow_arr = arr.into_arrow();
//! ```

#[cfg(feature = "arrow")]
pub mod arrow_bridge;

#[cfg(feature = "arrow")]
pub mod datafusion_bridge;

#[cfg(feature = "lance")]
pub mod lance_io;

#[cfg(feature = "arrow")]
pub mod fragment_index;

#[cfg(feature = "arrow")]
pub mod channel_index;

#[cfg(feature = "arrow")]
pub mod indexed_cascade;

// Re-exports for convenience
#[cfg(feature = "arrow")]
pub use arrow_bridge::{
    cogrecord_schema, cogrecords_to_record_batch, record_batch_to_cogrecords, FromArrow, IntoArrow,
};

#[cfg(feature = "arrow")]
pub use datafusion_bridge::{arrow_to_flat_bytes, cascade_scan_4ch, hamming_scan_column};

#[cfg(feature = "lance")]
pub use lance_io::{append_cogrecords, read_cogrecords, write_cogrecords};

#[cfg(feature = "arrow")]
pub use fragment_index::{FragmentIndex, FragmentMeta};

#[cfg(feature = "arrow")]
pub use channel_index::{ChannelIndex, ClusterMeta};

#[cfg(feature = "arrow")]
pub use indexed_cascade::{
    indexed_cascade_search, learn, rebuild, CascadeIndices, IndexedCascadeResult,
    IndexedCascadeStats,
};
