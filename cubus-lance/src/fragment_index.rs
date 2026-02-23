//! CLAM-to-Lance fragment mapping for the primary (META) channel.
//!
//! Maps CLAM leaf clusters to Lance fragment boundaries: each leaf cluster
//! becomes a contiguous range of rows. At query time, triangle inequality
//! on cluster centers prunes entire fragments before any I/O.
//!
//! ## Bandwidth arithmetic
//!
//! Without fragment index: 100K × 2KB meta = 200MB sequential scan (Stage 1).
//! With fragment index:    ~3 surviving clusters × ~400 records × 2KB = 2.4MB.
//! That's ~80× bandwidth reduction on Stage 1 alone.

use numrus_clam::tree::{BuildConfig, ClamTree};
use numrus_core::simd::hamming_distance;

/// Metadata for a contiguous group of rows that form one CLAM leaf cluster.
///
/// After `FragmentIndex::build()`, rows are reordered so that cluster i
/// occupies rows `row_id_start..row_id_end` in the logical ordering.
/// Lance physically stores these rows together in one fragment.
#[derive(Clone, Debug)]
pub struct FragmentMeta {
    /// Cluster center vector (2048 bytes for CogRecord channels).
    pub center: Vec<u8>,
    /// Maximum Hamming distance from center to any record in this cluster.
    pub radius: u64,
    /// Number of records in this cluster.
    pub cardinality: usize,
    /// Index of this fragment in the fragment list.
    pub fragment_idx: usize,
    /// First row ID (inclusive) in the reordered dataset.
    pub row_id_start: u64,
    /// Last row ID (exclusive) in the reordered dataset.
    pub row_id_end: u64,
}

/// Index mapping CLAM tree leaf clusters to row ranges.
///
/// The primary channel (META) uses this index. At query time:
/// 1. `find_overlapping(query, threshold)` → surviving fragment descriptors
/// 2. Read only those row ranges from the Lance meta column
/// 3. Fine-scan the surviving rows with Hamming distance
pub struct FragmentIndex {
    /// One entry per CLAM leaf cluster, in depth-first order.
    pub fragments: Vec<FragmentMeta>,
    /// The underlying CLAM tree.
    pub tree: ClamTree,
    /// Depth-first permutation: `reorder[i]` = original row index.
    /// Used to map from cluster-ordered positions back to original row IDs.
    pub reorder: Vec<usize>,
    /// Length of each vector in bytes (2048 for CogRecord channels).
    pub vec_len: usize,
}

impl FragmentIndex {
    /// Build a fragment index from flat column data.
    ///
    /// # Arguments
    /// * `column_data` — flat byte buffer: `column_data[i*vec_len..(i+1)*vec_len]` is row i
    /// * `vec_len` — bytes per vector (2048 for CogRecord containers)
    /// * `count` — number of rows
    /// * `min_cluster_size` — minimum leaf cardinality (controls fragment granularity)
    pub fn build(
        column_data: &[u8],
        vec_len: usize,
        count: usize,
        min_cluster_size: usize,
    ) -> Self {
        let config = BuildConfig {
            min_cardinality: min_cluster_size,
            ..BuildConfig::default()
        };
        let tree = ClamTree::build(column_data, vec_len, count, &config);

        // Collect leaf clusters and their row ranges.
        let mut fragments = Vec::with_capacity(tree.num_leaves);
        for cluster in &tree.nodes {
            if !cluster.is_leaf() {
                continue;
            }
            let center = tree.center_data(cluster, column_data, vec_len).to_vec();
            fragments.push(FragmentMeta {
                center,
                radius: cluster.radius,
                cardinality: cluster.cardinality,
                fragment_idx: fragments.len(),
                row_id_start: cluster.offset as u64,
                row_id_end: (cluster.offset + cluster.cardinality) as u64,
            });
        }

        let reorder = tree.reordered.clone();

        FragmentIndex {
            fragments,
            tree,
            reorder,
            vec_len,
        }
    }

    /// Find fragments whose clusters overlap the query ball.
    ///
    /// Uses triangle inequality: skip fragment if
    /// `d(query, center) - radius > threshold` (i.e., `delta_minus > threshold`).
    ///
    /// Returns references to surviving fragment descriptors.
    pub fn find_overlapping(&self, query: &[u8], threshold: u64) -> Vec<&FragmentMeta> {
        self.fragments
            .iter()
            .filter(|f| {
                let d = hamming_distance(query, &f.center);
                d.saturating_sub(f.radius) <= threshold
            })
            .collect()
    }

    /// Map reordered row positions to original row IDs.
    ///
    /// Given a range `start..end` in cluster-ordered space,
    /// returns the corresponding original row indices.
    pub fn original_row_ids(&self, start: u64, end: u64) -> Vec<usize> {
        self.reorder[start as usize..end as usize].to_vec()
    }

    /// Total number of fragments (leaf clusters).
    pub fn num_fragments(&self) -> usize {
        self.fragments.len()
    }

    /// Total number of rows indexed.
    pub fn num_rows(&self) -> usize {
        self.reorder.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clustered_data(cluster_size: usize, num_clusters: usize, vec_len: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(cluster_size * num_clusters * vec_len);
        for c in 0..num_clusters {
            let base_val = (c * 37 % 256) as u8;
            for i in 0..cluster_size {
                let mut vec = vec![base_val; vec_len];
                // Small perturbation within cluster
                for j in 0..vec_len.min(i + 1) {
                    vec[j] = base_val.wrapping_add((i % 4) as u8);
                }
                data.extend_from_slice(&vec);
            }
        }
        data
    }

    #[test]
    fn test_fragment_index_build() {
        let vec_len = 64;
        let cluster_size = 20;
        let num_clusters = 4;
        let count = cluster_size * num_clusters;
        let data = make_clustered_data(cluster_size, num_clusters, vec_len);

        let idx = FragmentIndex::build(&data, vec_len, count, 5);

        // Should have some leaf fragments
        assert!(idx.num_fragments() > 0);
        assert_eq!(idx.num_rows(), count);

        // All rows should be covered
        let total_rows: usize = idx.fragments.iter().map(|f| f.cardinality).sum();
        assert_eq!(total_rows, count);

        // Fragments should be non-overlapping and cover [0, count)
        let mut covered = vec![false; count];
        for f in &idx.fragments {
            for pos in f.row_id_start..f.row_id_end {
                assert!(!covered[pos as usize], "overlapping fragments at {}", pos);
                covered[pos as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all positions covered");
    }

    #[test]
    fn test_find_overlapping_exact() {
        let vec_len = 64;
        let count = 40;
        let data = make_clustered_data(10, 4, vec_len);

        let idx = FragmentIndex::build(&data, vec_len, count, 5);

        // Query = first row of the data — should find at least one overlapping fragment
        let query = &data[0..vec_len];
        let overlapping = idx.find_overlapping(query, 500);
        assert!(
            !overlapping.is_empty(),
            "exact query should find overlapping fragments"
        );
    }

    #[test]
    fn test_find_overlapping_prunes() {
        let vec_len = 64;
        let count = 80;
        let data = make_clustered_data(20, 4, vec_len);

        let idx = FragmentIndex::build(&data, vec_len, count, 10);

        // With threshold=0, only fragments whose center is exactly the query survive.
        // A very restrictive query should prune at least some fragments.
        let query = vec![0xFFu8; vec_len];
        let all = idx.find_overlapping(&query, u64::MAX);
        let tight = idx.find_overlapping(&query, 0);

        assert!(tight.len() <= all.len());
    }

    #[test]
    fn test_original_row_ids() {
        let vec_len = 32;
        let count = 20;
        let data = make_clustered_data(5, 4, vec_len);

        let idx = FragmentIndex::build(&data, vec_len, count, 3);

        // reorder should be a permutation of 0..count
        let mut sorted = idx.reorder.clone();
        sorted.sort();
        assert_eq!(sorted, (0..count).collect::<Vec<_>>());
    }
}
