//! XOR Delta Layer: sound borrow-free holographic storage.
//!
//! The core algebraic insight: XOR is its own inverse. A ground truth
//! fingerprint is `&self` forever — readers never need `&mut`. Writers
//! own a private `DeltaLayer` containing their XOR delta from ground truth.
//!
//! ```text
//! effective_value = ground_truth ^ delta
//!
//! To write new_value:  delta ^= effective_value ^ new_value
//!                    = delta ^ ground ^ delta ^ new_value
//!                    = ground ^ new_value
//! ```
//!
//! This eliminates the `&self → &mut` unsoundness entirely for binary vectors.
//! There is no `RefCell`, no `UnsafeCell`, no runtime borrow checking.
//!
//! ## Decision rule
//!
//! - `&mut [f32]` data → `split_at_mut` (see simd_ops)
//! - `Fingerprint<N>` data → XOR Delta Layers (this module)
//! - No overlap at the same level.

use numrus_core::fingerprint::Fingerprint;

/// A single XOR delta overlay on an immutable ground truth.
///
/// The ground truth is always borrowed as `&self` — no mutation needed.
/// The DeltaLayer owns its delta as a plain field — `&mut self` is just
/// regular Rust, no interior mutability tricks required.
pub struct DeltaLayer<const N: usize> {
    delta: Fingerprint<N>,
}

impl<const N: usize> DeltaLayer<N> {
    /// Create a clean (zero-delta) layer.
    #[inline]
    pub fn new() -> Self {
        Self {
            delta: Fingerprint::zero(),
        }
    }

    /// Read the effective value: `ground ^ delta`.
    #[inline]
    pub fn read(&self, ground: &Fingerprint<N>) -> Fingerprint<N> {
        ground ^ &self.delta
    }

    /// Write a new effective value.
    ///
    /// After this call, `self.read(ground)` will return `new_value`.
    ///
    /// Algebraically: `delta' = ground ^ new_value`, because
    /// `ground ^ delta' = ground ^ ground ^ new_value = new_value`.
    #[inline]
    pub fn write(&mut self, ground: &Fingerprint<N>, new_value: &Fingerprint<N>) {
        self.delta = ground ^ new_value;
    }

    /// Apply a targeted XOR patch to the effective value.
    ///
    /// `effective' = effective ^ patch`, implemented as `delta ^= patch`.
    #[inline]
    pub fn xor_patch(&mut self, patch: &Fingerprint<N>) {
        self.delta ^= patch;
    }

    /// Returns true if this layer has no changes (delta is zero).
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.delta.is_zero()
    }

    /// Number of bits that differ from ground truth.
    #[inline]
    pub fn changed_bits(&self) -> u32 {
        self.delta.popcount()
    }

    /// Borrow the raw delta (for serialization, wire transfer, etc.).
    #[inline]
    pub fn delta(&self) -> &Fingerprint<N> {
        &self.delta
    }

    /// Collapse this layer into the ground truth, producing a new ground truth.
    ///
    /// Consumes the layer — after collapse, create a fresh `DeltaLayer::new()`.
    #[inline]
    pub fn collapse(self, ground: &Fingerprint<N>) -> Fingerprint<N> {
        ground ^ &self.delta
    }
}

impl<const N: usize> Default for DeltaLayer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// A stack of XOR delta layers over an immutable ground truth.
///
/// Each layer represents a version. Reading through the stack composes
/// all deltas via XOR (which is associative and commutative).
///
/// ```text
/// effective = ground ^ layer[0].delta ^ layer[1].delta ^ ...
/// ```
///
/// Layers are ordered newest-first (push adds to front, pop removes from front).
pub struct LayerStack<const N: usize> {
    ground: Fingerprint<N>,
    layers: Vec<DeltaLayer<N>>,
}

impl<const N: usize> LayerStack<N> {
    /// Create a new stack with the given ground truth and no layers.
    pub fn new(ground: Fingerprint<N>) -> Self {
        Self {
            ground,
            layers: Vec::new(),
        }
    }

    /// Borrow the ground truth (always immutable).
    #[inline]
    pub fn ground(&self) -> &Fingerprint<N> {
        &self.ground
    }

    /// Number of active layers.
    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Push a new clean layer onto the stack. Returns its index.
    pub fn push_layer(&mut self) -> usize {
        let idx = self.layers.len();
        self.layers.push(DeltaLayer::new());
        idx
    }

    /// Get mutable access to a layer by index (for writing).
    #[inline]
    pub fn layer_mut(&mut self, idx: usize) -> &mut DeltaLayer<N> {
        &mut self.layers[idx]
    }

    /// Get immutable access to a layer by index (for inspection).
    #[inline]
    pub fn layer(&self, idx: usize) -> &DeltaLayer<N> {
        &self.layers[idx]
    }

    /// Read the effective value through all layers.
    ///
    /// `effective = ground ^ delta[0] ^ delta[1] ^ ...`
    pub fn read_all(&self) -> Fingerprint<N> {
        let mut result = self.ground.clone();
        for layer in &self.layers {
            result ^= layer.delta();
        }
        result
    }

    /// Read the effective value through layers 0..=idx.
    pub fn read_through(&self, idx: usize) -> Fingerprint<N> {
        let mut result = self.ground.clone();
        for layer in &self.layers[..=idx] {
            result ^= layer.delta();
        }
        result
    }

    /// Collapse all layers into a new ground truth, clearing the stack.
    ///
    /// This is the "flush" operation — turns accumulated deltas into
    /// the new canonical value. Returns the old ground truth.
    pub fn collapse_all(&mut self) -> Fingerprint<N> {
        let old_ground = self.ground.clone();
        self.ground = self.read_all();
        self.layers.clear();
        old_ground
    }

    /// Collapse a single layer into the ground truth by index.
    ///
    /// Removes the layer and updates the ground truth.
    pub fn collapse_layer(&mut self, idx: usize) -> &Fingerprint<N> {
        let layer = self.layers.remove(idx);
        self.ground = layer.collapse(&self.ground);
        &self.ground
    }

    /// Total number of changed bits across all layers (union of diffs).
    pub fn total_changed_bits(&self) -> u32 {
        let effective = self.read_all();
        self.ground.hamming_distance(&effective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fp() -> Fingerprint<4> {
        Fingerprint {
            words: [
                0xDEAD_BEEF_CAFE_BABE,
                0x1234_5678_9ABC_DEF0,
                0xFEDC_BA98_7654_3210,
                0x0123_4567_89AB_CDEF,
            ],
        }
    }

    fn other_fp() -> Fingerprint<4> {
        Fingerprint {
            words: [
                0x1111_1111_1111_1111,
                0x2222_2222_2222_2222,
                0x3333_3333_3333_3333,
                0x4444_4444_4444_4444,
            ],
        }
    }

    // --- DeltaLayer tests ---

    #[test]
    fn test_clean_layer_reads_ground() {
        let ground = sample_fp();
        let layer = DeltaLayer::<4>::new();
        assert_eq!(layer.read(&ground), ground);
        assert!(layer.is_clean());
    }

    #[test]
    fn test_write_then_read() {
        let ground = sample_fp();
        let new_val = other_fp();
        let mut layer = DeltaLayer::<4>::new();
        layer.write(&ground, &new_val);
        assert_eq!(layer.read(&ground), new_val);
        assert!(!layer.is_clean());
    }

    #[test]
    fn test_write_ground_is_clean() {
        let ground = sample_fp();
        let mut layer = DeltaLayer::<4>::new();
        layer.write(&ground, &ground); // write same value as ground
        assert!(layer.is_clean());
    }

    #[test]
    fn test_double_write() {
        let ground = sample_fp();
        let val1 = other_fp();
        let val2 = Fingerprint {
            words: [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD],
        };
        let mut layer = DeltaLayer::<4>::new();
        layer.write(&ground, &val1);
        assert_eq!(layer.read(&ground), val1);
        layer.write(&ground, &val2);
        assert_eq!(layer.read(&ground), val2);
    }

    #[test]
    fn test_xor_patch() {
        let ground = Fingerprint::<4>::zero();
        let mut layer = DeltaLayer::<4>::new();
        // Start clean: effective = 0
        let patch = Fingerprint {
            words: [0xFF, 0, 0, 0],
        };
        layer.xor_patch(&patch);
        // effective = 0 ^ 0xFF = 0xFF in word[0]
        let result = layer.read(&ground);
        assert_eq!(result.words[0], 0xFF);
        // Patch again with same value: self-inverse
        layer.xor_patch(&patch);
        assert!(layer.is_clean());
    }

    #[test]
    fn test_collapse() {
        let ground = sample_fp();
        let new_val = other_fp();
        let mut layer = DeltaLayer::<4>::new();
        layer.write(&ground, &new_val);
        let collapsed = layer.collapse(&ground);
        assert_eq!(collapsed, new_val);
    }

    #[test]
    fn test_changed_bits() {
        let ground = Fingerprint::<2> {
            words: [0x00, 0x00],
        };
        let mut layer = DeltaLayer::<2>::new();
        let new_val = Fingerprint::<2> {
            words: [0xFF, 0x00],
        }; // 8 bits changed
        layer.write(&ground, &new_val);
        assert_eq!(layer.changed_bits(), 8);
    }

    // --- LayerStack tests ---

    #[test]
    fn test_stack_no_layers() {
        let ground = sample_fp();
        let stack = LayerStack::new(ground.clone());
        assert_eq!(stack.depth(), 0);
        assert_eq!(stack.read_all(), ground);
    }

    #[test]
    fn test_stack_single_layer() {
        let ground = sample_fp();
        let new_val = other_fp();
        let mut stack = LayerStack::new(ground.clone());
        let idx = stack.push_layer();
        stack.layer_mut(idx).write(&ground, &new_val);
        assert_eq!(stack.read_all(), new_val);
    }

    #[test]
    fn test_stack_two_layers_compose() {
        // Two patches applied in sequence via XOR
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground.clone());

        let l0 = stack.push_layer();
        let patch0 = Fingerprint::<2> {
            words: [0xFF00, 0x0000],
        };
        stack.layer_mut(l0).xor_patch(&patch0);

        let l1 = stack.push_layer();
        let patch1 = Fingerprint::<2> {
            words: [0x00FF, 0x0000],
        };
        stack.layer_mut(l1).xor_patch(&patch1);

        // effective = 0 ^ 0xFF00 ^ 0x00FF = 0xFFFF
        let result = stack.read_all();
        assert_eq!(result.words[0], 0xFFFF);
        assert_eq!(result.words[1], 0x0000);
    }

    #[test]
    fn test_stack_read_through() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground.clone());

        let l0 = stack.push_layer();
        stack
            .layer_mut(l0)
            .xor_patch(&Fingerprint { words: [0xFF, 0] });

        let l1 = stack.push_layer();
        stack
            .layer_mut(l1)
            .xor_patch(&Fingerprint { words: [0xFF00, 0] });

        // Through layer 0 only: 0 ^ 0xFF = 0xFF
        assert_eq!(stack.read_through(0).words[0], 0xFF);
        // Through both: 0 ^ 0xFF ^ 0xFF00 = 0xFFFF
        assert_eq!(stack.read_through(1).words[0], 0xFFFF);
    }

    #[test]
    fn test_stack_collapse_all() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground);

        let l0 = stack.push_layer();
        stack.layer_mut(l0).xor_patch(&Fingerprint {
            words: [0xAA, 0xBB],
        });

        let l1 = stack.push_layer();
        stack.layer_mut(l1).xor_patch(&Fingerprint {
            words: [0x55, 0x44],
        });

        let expected = stack.read_all();
        let _old = stack.collapse_all();
        assert_eq!(stack.depth(), 0);
        assert_eq!(*stack.ground(), expected);
        assert_eq!(stack.read_all(), expected);
    }

    #[test]
    fn test_stack_collapse_single_layer() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground);

        let l0 = stack.push_layer();
        stack
            .layer_mut(l0)
            .xor_patch(&Fingerprint { words: [0xAA, 0] });

        let l1 = stack.push_layer();
        stack
            .layer_mut(l1)
            .xor_patch(&Fingerprint { words: [0x55, 0] });

        // Collapse layer 0 into ground
        stack.collapse_layer(0);
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.ground().words[0], 0xAA);
        // Remaining layer (was l1, now index 0) still adds 0x55
        assert_eq!(stack.read_all().words[0], 0xAA ^ 0x55);
    }

    #[test]
    fn test_total_changed_bits() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground);

        let l0 = stack.push_layer();
        // Set 8 bits in word 0
        stack
            .layer_mut(l0)
            .xor_patch(&Fingerprint { words: [0xFF, 0] });

        let l1 = stack.push_layer();
        // Set 4 bits in word 1
        stack
            .layer_mut(l1)
            .xor_patch(&Fingerprint { words: [0, 0x0F] });

        assert_eq!(stack.total_changed_bits(), 12);
    }

    #[test]
    fn test_ground_never_mutated_by_writes() {
        let ground = sample_fp();
        let ground_copy = ground.clone();
        let mut stack = LayerStack::new(ground);

        // Write through multiple layers
        for _ in 0..5 {
            let idx = stack.push_layer();
            stack.layer_mut(idx).xor_patch(&other_fp());
        }

        // Ground truth is untouched
        assert_eq!(*stack.ground(), ground_copy);
    }
}
