//! DataFusion bridge: zero-copy cascade scan over Arrow `FixedSizeBinaryArray` columns.
//!
//! When DataFusion scans a Lance dataset, it returns `RecordBatch` with
//! `FixedSizeBinaryArray` columns for each CogRecord container. This module
//! provides zero-copy access to those columns for SIMD-accelerated Hamming
//! search — no 8KB-per-record allocation.

use arrow::array::{Array, FixedSizeBinaryArray};
use numrus_core::simd::hamming_distance;
use numrus_rs::CogRecord;

/// Zero-copy slice into an Arrow `FixedSizeBinaryArray`'s backing buffer.
///
/// Arrow's `FixedSizeBinaryArray` stores values contiguously in memory,
/// which is the exact layout `hamming_search_adaptive` expects for a
/// packed database. This function returns the raw byte slice — no copy.
///
/// # Safety
///
/// The returned slice borrows from the `FixedSizeBinaryArray`. The caller
/// must ensure the array outlives the slice.
pub fn arrow_to_flat_bytes(col: &FixedSizeBinaryArray) -> &[u8] {
    col.value_data()
}

/// Scan an Arrow `FixedSizeBinaryArray` column with Hamming distance,
/// returning indices and distances of candidates below `max_distance`.
///
/// This is the zero-copy path: the Arrow column's backing buffer is used
/// directly as the packed database for linear Hamming scan. No allocation
/// for the database itself.
pub fn hamming_scan_column(
    query: &[u8],
    column: &FixedSizeBinaryArray,
    max_distance: u64,
) -> Vec<(usize, u64)> {
    let n = column.len();
    let vec_len = column.value_length() as usize;
    let flat = arrow_to_flat_bytes(column);
    let mut results = Vec::new();

    for i in 0..n {
        let offset = i * vec_len;
        let candidate = &flat[offset..offset + vec_len];

        // Scalar Hamming via u64 popcount
        let dist = hamming_distance(query, candidate);
        if dist <= max_distance {
            results.push((i, dist));
        }
    }

    results
}

/// 4-channel cascade scan over a RecordBatch of CogRecords.
///
/// Scans all 4 container columns with per-channel thresholds.
/// Uses compound early exit: if META exceeds threshold, skip CAM/BTREE/EMBED.
///
/// Zero-copy: operates directly on Arrow column buffers.
pub fn cascade_scan_4ch(
    query: &CogRecord,
    meta_col: &FixedSizeBinaryArray,
    cam_col: &FixedSizeBinaryArray,
    btree_col: &FixedSizeBinaryArray,
    embed_col: &FixedSizeBinaryArray,
    thresholds: [u64; 4],
) -> Vec<(usize, [u64; 4])> {
    let n = meta_col.len();
    assert_eq!(
        cam_col.len(),
        n,
        "cascade_scan_4ch: cam_col length {} != meta_col length {}",
        cam_col.len(),
        n
    );
    assert_eq!(
        btree_col.len(),
        n,
        "cascade_scan_4ch: btree_col length {} != meta_col length {}",
        btree_col.len(),
        n
    );
    assert_eq!(
        embed_col.len(),
        n,
        "cascade_scan_4ch: embed_col length {} != meta_col length {}",
        embed_col.len(),
        n
    );
    let vec_len = meta_col.value_length() as usize;
    let meta_flat = arrow_to_flat_bytes(meta_col);
    let cam_flat = arrow_to_flat_bytes(cam_col);
    let btree_flat = arrow_to_flat_bytes(btree_col);
    let embed_flat = arrow_to_flat_bytes(embed_col);

    let q_meta = query.meta.data_slice();
    let q_cam = query.cam.data_slice();
    let q_btree = query.btree.data_slice();
    let q_embed = query.embed.data_slice();

    let mut results = Vec::new();

    for i in 0..n {
        let offset = i * vec_len;

        // Stage 1: META (cheapest rejection)
        let meta_dist = hamming_distance(q_meta, &meta_flat[offset..offset + vec_len]);
        if meta_dist > thresholds[0] {
            continue;
        }

        // Stage 2: CAM
        let cam_dist = hamming_distance(q_cam, &cam_flat[offset..offset + vec_len]);
        if cam_dist > thresholds[1] {
            continue;
        }

        // Stage 3: BTREE
        let btree_dist = hamming_distance(q_btree, &btree_flat[offset..offset + vec_len]);
        if btree_dist > thresholds[2] {
            continue;
        }

        // Stage 4: EMBED
        let embed_dist = hamming_distance(q_embed, &embed_flat[offset..offset + vec_len]);
        if embed_dist > thresholds[3] {
            continue;
        }

        results.push((i, [meta_dist, cam_dist, btree_dist, embed_dist]));
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::FixedSizeBinaryBuilder;
    use numrus_rs::NumArrayU8;

    fn make_column(data: &[&[u8]], element_size: i32) -> FixedSizeBinaryArray {
        let mut builder = FixedSizeBinaryBuilder::with_capacity(data.len(), element_size);
        for row in data {
            builder.append_value(row).unwrap();
        }
        builder.finish()
    }

    #[test]
    fn test_arrow_to_flat_bytes_contiguous() {
        let row0 = vec![0u8; 2048];
        let row1 = vec![1u8; 2048];
        let col = make_column(&[&row0, &row1], 2048);

        let flat = arrow_to_flat_bytes(&col);
        assert_eq!(flat.len(), 4096);
        assert_eq!(flat[0], 0);
        assert_eq!(flat[2048], 1);
    }

    #[test]
    fn test_hamming_scan_column_exact_match() {
        let query = vec![0xAAu8; 2048];
        let rows: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                if i == 5 {
                    vec![0xAAu8; 2048] // exact match
                } else {
                    vec![i as u8; 2048]
                }
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let col = make_column(&refs, 2048);

        let results = hamming_scan_column(&query, &col, 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 5);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_hamming_scan_column_threshold() {
        let query = vec![0u8; 2048];
        let close = vec![1u8; 2048]; // Hamming distance = 2048 * 1 bit = 2048
        let far = vec![0xFFu8; 2048]; // Hamming distance = 2048 * 8 = 16384
        let col = make_column(&[&close, &far], 2048);

        let results = hamming_scan_column(&query, &col, 3000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_cascade_scan_4ch() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; 2048]);
        let query = CogRecord::new(
            make_container(0),
            make_container(0),
            make_container(0),
            make_container(0),
        );

        // Create 5 records: record 2 is close to query, others are far
        let records: Vec<CogRecord> = (0..5)
            .map(|i| {
                if i == 2 {
                    CogRecord::new(
                        make_container(1), // close in META
                        make_container(1), // close in CAM
                        make_container(1), // close in BTREE
                        make_container(1), // close in EMBED
                    )
                } else {
                    CogRecord::new(
                        make_container(0xFF),
                        make_container(0xFF),
                        make_container(0xFF),
                        make_container(0xFF),
                    )
                }
            })
            .collect();

        let meta_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.meta.data_slice().to_vec())
            .collect();
        let cam_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.cam.data_slice().to_vec())
            .collect();
        let btree_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.btree.data_slice().to_vec())
            .collect();
        let embed_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.embed.data_slice().to_vec())
            .collect();

        let m_refs: Vec<&[u8]> = meta_rows.iter().map(|r| r.as_slice()).collect();
        let c_refs: Vec<&[u8]> = cam_rows.iter().map(|r| r.as_slice()).collect();
        let b_refs: Vec<&[u8]> = btree_rows.iter().map(|r| r.as_slice()).collect();
        let e_refs: Vec<&[u8]> = embed_rows.iter().map(|r| r.as_slice()).collect();

        let meta_col = make_column(&m_refs, 2048);
        let cam_col = make_column(&c_refs, 2048);
        let btree_col = make_column(&b_refs, 2048);
        let embed_col = make_column(&e_refs, 2048);

        // Threshold: 3000 bits per channel — only record 2 should pass all 4
        let results = cascade_scan_4ch(
            &query,
            &meta_col,
            &cam_col,
            &btree_col,
            &embed_col,
            [3000, 3000, 3000, 3000],
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_hamming_distance_correctness() {
        let a = vec![0u8; 64];
        let b = vec![0xFFu8; 64];
        assert_eq!(hamming_distance(&a, &b), 64 * 8);

        let c = vec![0u8; 64];
        assert_eq!(hamming_distance(&a, &c), 0);
    }
}
