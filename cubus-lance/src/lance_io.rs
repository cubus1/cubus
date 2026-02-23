//! CogRecord ↔ Lance dataset I/O.
//!
//! Write batches of CogRecords to a Lance dataset on disk (or object store),
//! and read them back. Lance stores each container as a `FixedSizeBinary(2048)`
//! column, enabling IVF indexing on any container.
//!
//! All functions are async (Lance uses tokio internally).

use crate::arrow_bridge::{
    cogrecord_schema, cogrecords_to_record_batch, record_batch_to_cogrecords,
};
use arrow::array::RecordBatchIterator;
use futures::StreamExt;
use lance::dataset::write::{WriteMode, WriteParams};
use lance::dataset::Dataset;
use numrus_rs::CogRecord;
use std::sync::Arc;

/// Write CogRecords to a new Lance dataset at `uri`.
///
/// Creates the dataset if it doesn't exist. Overwrites if it does.
pub async fn write_cogrecords(uri: &str, records: &[CogRecord]) -> Result<Dataset, lance::Error> {
    let batch =
        cogrecords_to_record_batch(records).expect("CogRecord data must be 2048 bytes per channel");
    let schema = Arc::new(cogrecord_schema());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut params = WriteParams::default();
    params.mode = WriteMode::Create;
    Dataset::write(reader, uri, Some(params)).await
}

/// Append CogRecords to an existing Lance dataset.
pub async fn append_cogrecords(uri: &str, records: &[CogRecord]) -> Result<Dataset, lance::Error> {
    let batch =
        cogrecords_to_record_batch(records).expect("CogRecord data must be 2048 bytes per channel");
    let schema = Arc::new(cogrecord_schema());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut params = WriteParams::default();
    params.mode = WriteMode::Append;
    Dataset::write(reader, uri, Some(params)).await
}

/// Read all CogRecords from a Lance dataset.
pub async fn read_cogrecords(uri: &str) -> Result<Vec<CogRecord>, lance::Error> {
    let dataset = Dataset::open(uri).await?;
    let mut records = Vec::new();

    let mut stream = dataset.scan().try_into_stream().await?;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        records.extend(record_batch_to_cogrecords(&batch));
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_rs::NumArrayU8;

    fn make_test_records(n: usize) -> Vec<CogRecord> {
        (0..n)
            .map(|i| {
                let val = (i % 256) as u8;
                CogRecord::new(
                    NumArrayU8::new(vec![val; 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(1); 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(2); 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(3); 2048]),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_write_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("test.lance");
        let uri_str = uri.to_str().unwrap();

        let records = make_test_records(100);
        write_cogrecords(uri_str, &records).await.unwrap();

        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 100);

        for (i, rec) in back.iter().enumerate() {
            let val = (i % 256) as u8;
            assert_eq!(rec.meta.data_slice()[0], val);
            assert_eq!(rec.cam.data_slice()[0], val.wrapping_add(1));
            assert_eq!(rec.btree.data_slice()[0], val.wrapping_add(2));
            assert_eq!(rec.embed.data_slice()[0], val.wrapping_add(3));
        }
    }

    #[tokio::test]
    async fn test_append() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("append.lance");
        let uri_str = uri.to_str().unwrap();

        write_cogrecords(uri_str, &make_test_records(50))
            .await
            .unwrap();
        append_cogrecords(uri_str, &make_test_records(50))
            .await
            .unwrap();

        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 100);
    }

    #[tokio::test]
    async fn test_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("empty.lance");
        let uri_str = uri.to_str().unwrap();

        write_cogrecords(uri_str, &[]).await.unwrap();
        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 0);
    }
}
