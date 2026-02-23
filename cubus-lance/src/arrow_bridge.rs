//! Zero-copy conversions between numrus types and Arrow arrays.
//!
//! `NumArray::into_arrow()` consumes the array and transfers the backing `Vec<T>`
//! directly into an Arrow `Buffer` — no memcpy for primitive types.
//!
//! `NumArray::from_arrow()` copies from the Arrow buffer into a new `Vec<T>`
//! because Arrow owns its buffer with a different allocator.

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeBinaryBuilder, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, UInt8Array,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use numrus_rs::{CogRecord, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Convert a numrus type into an Arrow array (consumes self, zero-copy).
pub trait IntoArrow {
    type ArrowArray;
    fn into_arrow(self) -> Self::ArrowArray;
}

/// Construct a numrus type from an Arrow array (copies data).
pub trait FromArrow<A> {
    fn from_arrow(arr: &A) -> Self;
}

// ---------------------------------------------------------------------------
// Macro to implement for all primitive types
// ---------------------------------------------------------------------------

macro_rules! impl_arrow_bridge {
    ($NumTy:ty, $ArrowTy:ty, $RustTy:ty) => {
        impl IntoArrow for $NumTy {
            type ArrowArray = $ArrowTy;
            fn into_arrow(self) -> $ArrowTy {
                let len = self.len();
                let buf = Buffer::from_vec(self.into_data());
                <$ArrowTy>::new(ScalarBuffer::<$RustTy>::new(buf, 0, len), None)
            }
        }

        impl FromArrow<$ArrowTy> for $NumTy {
            fn from_arrow(arr: &$ArrowTy) -> Self {
                Self::new(arr.values().to_vec())
            }
        }
    };
}

impl_arrow_bridge!(NumArrayF32, Float32Array, f32);
impl_arrow_bridge!(NumArrayF64, Float64Array, f64);
impl_arrow_bridge!(NumArrayI32, Int32Array, i32);
impl_arrow_bridge!(NumArrayI64, Int64Array, i64);
impl_arrow_bridge!(NumArrayU8, UInt8Array, u8);

// ---------------------------------------------------------------------------
// CogRecord <-> Arrow RecordBatch
// ---------------------------------------------------------------------------

/// Container size in bytes (each container is 2048 bytes = 16384 bits).
const CONTAINER_BYTES: i32 = 2048;

/// Arrow schema for a batch of CogRecords.
/// Each container is a `FixedSizeBinary(2048)`.
pub fn cogrecord_schema() -> Schema {
    Schema::new(vec![
        Field::new("meta", DataType::FixedSizeBinary(CONTAINER_BYTES), false),
        Field::new("cam", DataType::FixedSizeBinary(CONTAINER_BYTES), false),
        Field::new("btree", DataType::FixedSizeBinary(CONTAINER_BYTES), false),
        Field::new("embed", DataType::FixedSizeBinary(CONTAINER_BYTES), false),
    ])
}

/// Convert a slice of CogRecords into an Arrow RecordBatch.
pub fn cogrecords_to_record_batch(
    records: &[CogRecord],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = Arc::new(cogrecord_schema());
    let n = records.len();

    let mut meta_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES);
    let mut cam_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES);
    let mut btree_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES);
    let mut embed_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES);

    for rec in records {
        meta_builder.append_value(rec.meta.data_slice())?;
        cam_builder.append_value(rec.cam.data_slice())?;
        btree_builder.append_value(rec.btree.data_slice())?;
        embed_builder.append_value(rec.embed.data_slice())?;
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(meta_builder.finish()) as ArrayRef,
            Arc::new(cam_builder.finish()) as ArrayRef,
            Arc::new(btree_builder.finish()) as ArrayRef,
            Arc::new(embed_builder.finish()) as ArrayRef,
        ],
    )
}

/// Convert an Arrow RecordBatch (matching `cogrecord_schema()`) back to CogRecords.
pub fn record_batch_to_cogrecords(batch: &RecordBatch) -> Vec<CogRecord> {
    let n = batch.num_rows();
    let meta_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("meta column");
    let cam_col = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("cam column");
    let btree_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("btree column");
    let embed_col = batch
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("embed column");

    (0..n)
        .map(|i| {
            CogRecord::new(
                NumArrayU8::new(meta_col.value(i).to_vec()),
                NumArrayU8::new(cam_col.value(i).to_vec()),
                NumArrayU8::new(btree_col.value(i).to_vec()),
                NumArrayU8::new(embed_col.value(i).to_vec()),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_round_trip() {
        let data = vec![1.0f32, 2.0, 3.0, -4.5, 0.0];
        let arr = NumArrayF32::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 5);
        let back = NumArrayF32::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_f64_round_trip() {
        let data = vec![1.0f64, -2.5, std::f64::consts::PI, 0.0];
        let arr = NumArrayF64::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 4);
        let back = NumArrayF64::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_i32_round_trip() {
        let data = vec![10i32, -20, 300, 0, i32::MAX, i32::MIN];
        let arr = NumArrayI32::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayI32::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_i64_round_trip() {
        let data = vec![100i64, -200, i64::MAX];
        let arr = NumArrayI64::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayI64::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_u8_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        let arr = NumArrayU8::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 256);
        let back = NumArrayU8::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_cogrecord_record_batch_round_trip() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; 2048]);
        let records: Vec<CogRecord> = (0..10)
            .map(|i| {
                CogRecord::new(
                    make_container(i),
                    make_container(i + 10),
                    make_container(i + 20),
                    make_container(i + 30),
                )
            })
            .collect();

        let batch = cogrecords_to_record_batch(&records).unwrap();
        assert_eq!(batch.num_rows(), 10);
        assert_eq!(batch.num_columns(), 4);

        let back = record_batch_to_cogrecords(&batch);
        assert_eq!(back.len(), 10);

        for (i, rec) in back.iter().enumerate() {
            assert_eq!(rec.meta.data_slice()[0], i as u8);
            assert_eq!(rec.cam.data_slice()[0], (i + 10) as u8);
            assert_eq!(rec.btree.data_slice()[0], (i + 20) as u8);
            assert_eq!(rec.embed.data_slice()[0], (i + 30) as u8);
        }
    }

    #[test]
    fn test_cogrecord_schema_fields() {
        let schema = cogrecord_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "meta");
        assert_eq!(schema.field(1).name(), "cam");
        assert_eq!(schema.field(2).name(), "btree");
        assert_eq!(schema.field(3).name(), "embed");
        for f in schema.fields() {
            assert_eq!(*f.data_type(), DataType::FixedSizeBinary(2048));
            assert!(!f.is_nullable());
        }
    }

    #[test]
    fn test_empty_array() {
        let arr = NumArrayF32::new(vec![]);
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 0);
        let back = NumArrayF32::from_arrow(&arrow);
        assert_eq!(back.len(), 0);
    }

    #[test]
    fn test_large_u8_container() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let arr = NumArrayU8::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayU8::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }
}
