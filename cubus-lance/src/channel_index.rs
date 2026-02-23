//! Per-channel sidecar index for non-primary (CAM, BTREE, EMBED) channels.
//!
//! The primary channel (META) is fragment-indexed — rows are physically
//! ordered by META cluster. Secondary channels can't share that physical
//! ordering, so they use a sidecar index: a CLAM tree over the channel data
//! with a mapping from cluster_id → original row IDs.
//!
//! At query time, triangle inequality prunes clusters, then the union of
//! surviving clusters' row IDs is intersected with the previous stage's
//! survivor set. This intersection is small (10%/1%/0.1% of N), so random
//! access via Lance `take()` is acceptable.

use numrus_clam::tree::{BuildConfig, ClamTree};
use numrus_core::simd::hamming_distance;
use std::collections::HashSet;

/// Metadata for one cluster in a secondary channel's CLAM tree.
#[derive(Clone, Debug)]
pub struct ClusterMeta {
    /// Cluster center vector (2048 bytes for CogRecord channels).
    pub center: Vec<u8>,
    /// Maximum Hamming distance from center to any record in cluster.
    pub radius: u64,
    /// Original row IDs of records in this cluster.
    pub row_ids: Vec<usize>,
}

/// Sidecar CLAM index for a non-primary CogRecord channel.
///
/// Unlike the primary FragmentIndex which controls physical layout,
/// this index is purely logical — it maps clusters to row ID lists
/// for targeted lookup after the previous cascade stage.
pub struct ChannelIndex {
    /// Which channel this indexes (1=cam, 2=btree, 3=embed).
    pub channel: usize,
    /// Per-leaf-cluster metadata.
    pub clusters: Vec<ClusterMeta>,
    /// The underlying CLAM tree.
    pub tree: ClamTree,
    /// Bytes per vector.
    pub vec_len: usize,
}

impl ChannelIndex {
    /// Build a sidecar index from flat column data.
    ///
    /// # Arguments
    /// * `channel` — channel number (1=cam, 2=btree, 3=embed)
    /// * `column_data` — flat byte buffer: `column_data[i*vec_len..(i+1)*vec_len]` is row i
    /// * `vec_len` — bytes per vector (2048 for CogRecord containers)
    /// * `count` — number of rows
    /// * `min_cluster_size` — minimum leaf cardinality
    pub fn build(
        channel: usize,
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

        // Collect leaf clusters with their original row IDs.
        let mut clusters = Vec::with_capacity(tree.num_leaves);
        for cluster in &tree.nodes {
            if !cluster.is_leaf() {
                continue;
            }
            let center = tree.center_data(cluster, column_data, vec_len).to_vec();
            let row_ids: Vec<usize> = tree
                .cluster_points(cluster, column_data, vec_len)
                .map(|(orig_idx, _)| orig_idx)
                .collect();
            clusters.push(ClusterMeta {
                center,
                radius: cluster.radius,
                row_ids,
            });
        }

        ChannelIndex {
            channel,
            clusters,
            tree,
            vec_len,
        }
    }

    /// Find original row IDs in clusters that overlap the query ball.
    ///
    /// Uses triangle inequality: skip cluster if `d(query, center) - radius > threshold`.
    /// Returns the union of row IDs from all overlapping clusters.
    pub fn overlapping_row_ids(&self, query: &[u8], threshold: u64) -> HashSet<usize> {
        let mut result = HashSet::new();
        for cluster in &self.clusters {
            let d = hamming_distance(query, &cluster.center);
            if d.saturating_sub(cluster.radius) <= threshold {
                result.extend(&cluster.row_ids);
            }
        }
        result
    }

    /// Incremental insert: assign new records to their nearest existing cluster.
    ///
    /// For each `(row_id, data)`, finds the cluster whose center is closest
    /// and appends the row_id to that cluster's list. This is O(new_records × clusters)
    /// — cheap for small batches against a moderate number of clusters.
    ///
    /// Does NOT update cluster centers or radii. Periodic `rebuild()` is needed
    /// when the data distribution shifts.
    pub fn insert(&mut self, new_data: &[(usize, &[u8])]) {
        for &(row_id, data) in new_data {
            let mut best_cluster = 0;
            let mut best_dist = u64::MAX;
            for (i, cluster) in self.clusters.iter().enumerate() {
                let d = hamming_distance(data, &cluster.center);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = i;
                }
            }
            self.clusters[best_cluster].row_ids.push(row_id);
            // Update radius if the new point extends the cluster.
            if best_dist > self.clusters[best_cluster].radius {
                self.clusters[best_cluster].radius = best_dist;
            }
        }
    }

    /// Total number of clusters (leaf nodes).
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Total number of indexed row IDs across all clusters.
    pub fn num_rows(&self) -> usize {
        self.clusters.iter().map(|c| c.row_ids.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_channel_data(n: usize, vec_len: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * vec_len);
        for i in 0..n {
            let val = (i * 31 % 256) as u8;
            data.extend(std::iter::repeat_n(val, vec_len));
        }
        data
    }

    #[test]
    fn test_channel_index_build() {
        let vec_len = 64;
        let count = 40;
        let data = make_channel_data(count, vec_len);

        let idx = ChannelIndex::build(1, &data, vec_len, count, 5);

        assert!(idx.num_clusters() > 0);
        assert_eq!(idx.num_rows(), count);

        // All original row IDs should appear exactly once.
        let mut seen = HashSet::new();
        for cluster in &idx.clusters {
            for &rid in &cluster.row_ids {
                assert!(seen.insert(rid), "duplicate row_id {}", rid);
            }
        }
        assert_eq!(seen.len(), count);
    }

    #[test]
    fn test_overlapping_row_ids() {
        let vec_len = 64;
        let count = 40;
        let data = make_channel_data(count, vec_len);

        let idx = ChannelIndex::build(1, &data, vec_len, count, 5);

        // Query matching first row — should find at least one overlapping cluster
        let query = &data[0..vec_len];
        let hits = idx.overlapping_row_ids(query, 500);
        assert!(!hits.is_empty());

        // With max threshold, all rows should appear
        let all = idx.overlapping_row_ids(query, u64::MAX);
        assert_eq!(all.len(), count);
    }

    #[test]
    fn test_insert_new_records() {
        let vec_len = 64;
        let count = 20;
        let data = make_channel_data(count, vec_len);

        let mut idx = ChannelIndex::build(1, &data, vec_len, count, 5);
        let before = idx.num_rows();

        // Insert 5 new records
        let new_vecs: Vec<Vec<u8>> = (0..5)
            .map(|i| vec![(i * 50 % 256) as u8; vec_len])
            .collect();
        let new_data: Vec<(usize, &[u8])> = new_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (count + i, v.as_slice()))
            .collect();

        idx.insert(&new_data);

        assert_eq!(idx.num_rows(), before + 5);
    }

    #[test]
    fn test_pruning_reduces_candidates() {
        let vec_len = 64;
        let count = 80;
        let data = make_channel_data(count, vec_len);

        let idx = ChannelIndex::build(1, &data, vec_len, count, 10);

        let query = vec![0xFFu8; vec_len];
        let loose = idx.overlapping_row_ids(&query, u64::MAX);
        let tight = idx.overlapping_row_ids(&query, 0);

        assert!(tight.len() <= loose.len());
    }
}
