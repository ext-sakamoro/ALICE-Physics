//! Linear Bounding Volume Hierarchy (LBVH) - Stackless Edition
//!
//! Optimized spatial acceleration structure for broad-phase collision detection.
//!
//! # Features
//!
//! - Morton code-based construction (deterministic)
//! - Flat array storage (cache-friendly, 32 bytes per node)
//! - **Stackless traversal** using escape pointers (zero heap allocation during query)
//! - SIMD-friendly AABB intersection tests

use crate::collider::AABB;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Morton Codes (Z-order curve)
// ============================================================================

/// Expand 21-bit integer to 63 bits for 3D Morton code
#[inline]
const fn expand_bits(mut v: u64) -> u64 {
    // Spread bits: each bit is followed by two zero bits
    v = (v | (v << 32)) & 0x001F00000000FFFF;
    v = (v | (v << 16)) & 0x001F0000FF0000FF;
    v = (v | (v << 8)) & 0x100F00F00F00F00F;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3;
    v = (v | (v << 2)) & 0x1249249249249249;
    v
}

/// Compute 63-bit Morton code from 3D coordinates
///
/// Coordinates should be normalized to [0, 2^21) range (crate-internal helper).
#[inline]
#[must_use]
pub(crate) fn morton_code(x: u64, y: u64, z: u64) -> u64 {
    let x = x.min((1 << 21) - 1);
    let y = y.min((1 << 21) - 1);
    let z = z.min((1 << 21) - 1);

    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Compute the Morton code of a point within a bounding box.
///
/// Public since 1.2.0: ALICE-TRT's GPU Morton kernel asserts byte-exact
/// parity against this function (the 1.0 API freeze had made it
/// `pub(crate)`, which broke that parity test against a path dependency).
/// The isotropic normalisation and the 21-bit-per-axis quantisation are the
/// determinism contract; changing either re-pins the BVH goldens.
#[must_use]
pub fn point_to_morton(point: Vec3Fix, bounds: &AABB) -> u64 {
    let size = bounds.max - bounds.min;
    // Isotropic normalisation (1.2.0): every axis is scaled by the *largest*
    // extent so the 21-bit cells are cubes. Per-axis [0, 1] scaling made the
    // high Morton bits of a short axis outrank a long axis (a 2003 × 3 × 3
    // world interleaved y/z jitter above x clusters), which mixed distant
    // clusters into the same leaves and inflated the `find_pairs` superset.
    let mut scale = size.x;
    if size.y > scale {
        scale = size.y;
    }
    if size.z > scale {
        scale = size.z;
    }
    if scale.is_zero() || scale.is_negative() {
        return 0;
    }
    let quantise = |coord: Fix128, min: Fix128| -> u64 {
        let t = (coord - min) / scale;
        if t.is_negative() {
            0
        } else if t.hi >= 1 {
            0x1FFFFF
        } else {
            (t.lo >> 43) & 0x1FFFFF
        }
    };
    let nx = quantise(point.x, bounds.min.x);
    let ny = quantise(point.y, bounds.min.y);
    let nz = quantise(point.z, bounds.min.z);
    morton_code(nx, ny, nz)
}

// ============================================================================
// BVH Node (Stackless-Ready)
// ============================================================================

/// Sentinel value for "no escape" (end of traversal, crate-internal).
pub(crate) const ESCAPE_NONE: u32 = u32::MAX;

/// Placeholder escape target for a LEFT subtree's descendants whose
/// correct escape (= the right sibling's index) is not yet known at
/// push time. Value `0` is safe because index 0 is always the tree
/// root and escape pointers strictly move forward — no legitimate
/// escape target is ever `0`. `build_recursive` sweeps this
/// placeholder into the real right-sibling index in a single linear
/// pass over the left subtree once its size is known.
const LEFT_ESCAPE_PLACEHOLDER: u32 = 0;

/// BVH node (32 bytes, cache-line friendly)
///
/// Layout optimized for stackless traversal:
/// - `escape_idx`: Next node to visit if AABB test fails (skip entire subtree)
/// - For leaves: `escape_idx` points to next sibling or parent's escape
/// - For internal nodes: `escape_idx` points to next subtree after both children
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct BvhNode {
    /// Bounding box minimum (compressed to i32)
    pub aabb_min: [i32; 3],
    /// First child index (internal) or primitive start (leaf)
    pub first_child_or_prim: u32,
    /// Bounding box maximum (compressed to i32)
    pub aabb_max: [i32; 3],
    /// Packed: upper 8 bits = primitive count (0 = internal), lower 24 bits = escape index
    pub prim_count_escape: u32,
}

impl BvhNode {
    /// Maximum primitives per leaf (fits in 8 bits, crate-internal cap).
    pub(crate) const MAX_PRIMS_PER_LEAF: u32 = 255;

    /// Create internal node
    #[inline]
    #[must_use]
    pub fn internal(aabb: &AABB, first_child: u32, escape_idx: u32) -> Self {
        Self {
            aabb_min: aabb_to_i32_min(aabb),
            first_child_or_prim: first_child,
            aabb_max: aabb_to_i32_max(aabb),
            prim_count_escape: escape_idx & 0x00FFFFFF, // prim_count = 0 (internal)
        }
    }

    /// Create leaf node.
    ///
    /// `count` is saturated to 255 (crate-internal `MAX_PRIMS_PER_LEAF` cap) to prevent
    /// 8-bit overflow that would make the leaf appear as an internal node.
    #[inline]
    #[must_use]
    pub fn leaf(aabb: &AABB, first_prim: u32, count: u32, escape_idx: u32) -> Self {
        let clamped = count.min(Self::MAX_PRIMS_PER_LEAF);
        Self {
            aabb_min: aabb_to_i32_min(aabb),
            first_child_or_prim: first_prim,
            aabb_max: aabb_to_i32_max(aabb),
            prim_count_escape: ((clamped & 0xFF) << 24) | (escape_idx & 0x00FFFFFF),
        }
    }

    /// Check if this is a leaf node
    #[inline]
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        (self.prim_count_escape >> 24) > 0
    }

    /// Get primitive count (0 for internal nodes)
    #[inline]
    #[must_use]
    pub const fn prim_count(&self) -> u32 {
        self.prim_count_escape >> 24
    }

    /// Get escape index (next node to visit on AABB miss)
    #[inline]
    #[must_use]
    pub const fn escape_idx(&self) -> u32 {
        self.prim_count_escape & 0x00FFFFFF
    }

    /// Get AABB (reconstructed from compressed i32)
    #[inline]
    #[must_use]
    pub const fn get_aabb(&self) -> AABB {
        AABB {
            min: Vec3Fix::new(
                Fix128::from_int(self.aabb_min[0] as i64),
                Fix128::from_int(self.aabb_min[1] as i64),
                Fix128::from_int(self.aabb_min[2] as i64),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(self.aabb_max[0] as i64),
                Fix128::from_int(self.aabb_max[1] as i64),
                Fix128::from_int(self.aabb_max[2] as i64),
            ),
        }
    }

    /// Fast AABB intersection test (integer-only, no Fix128 reconstruction)
    #[inline]
    #[must_use]
    pub const fn intersects_i32(&self, query_min: &[i32; 3], query_max: &[i32; 3]) -> bool {
        self.aabb_min[0] <= query_max[0]
            && self.aabb_max[0] >= query_min[0]
            && self.aabb_min[1] <= query_max[1]
            && self.aabb_max[1] >= query_min[1]
            && self.aabb_min[2] <= query_max[2]
            && self.aabb_max[2] >= query_min[2]
    }
}

/// Floor a Fix128 to i32, clamped to i32 range.
/// For min bounds: floor ensures the AABB fully contains the original.
#[inline]
fn fix128_floor_i32(v: Fix128) -> i32 {
    // For negative values with fractional part, hi is already floor (two's complement)
    v.hi.max(i32::MIN as i64).min(i32::MAX as i64) as i32
}

/// Ceil a Fix128 to i32, clamped to i32 range.
/// For max bounds: ceil ensures the AABB fully contains the original.
#[inline]
fn fix128_ceil_i32(v: Fix128) -> i32 {
    let ceil = if v.lo > 0 { v.hi + 1 } else { v.hi };
    ceil.max(i32::MIN as i64).min(i32::MAX as i64) as i32
}

#[inline]
fn aabb_to_i32_min(aabb: &AABB) -> [i32; 3] {
    [
        fix128_floor_i32(aabb.min.x),
        fix128_floor_i32(aabb.min.y),
        fix128_floor_i32(aabb.min.z),
    ]
}

#[inline]
fn aabb_to_i32_max(aabb: &AABB) -> [i32; 3] {
    [
        fix128_ceil_i32(aabb.max.x),
        fix128_ceil_i32(aabb.max.y),
        fix128_ceil_i32(aabb.max.z),
    ]
}

// ============================================================================
// Linear BVH with Stackless Traversal
// ============================================================================

/// Primitive entry for BVH construction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BvhPrimitive {
    /// AABB of the primitive
    pub aabb: AABB,
    /// Original index (e.g., body index)
    pub index: u32,
    /// Morton code (computed during build)
    pub morton: u64,
}

/// Linear BVH (flat array storage with stackless traversal)
pub struct LinearBvh {
    /// Flat array of nodes (depth-first order with escape pointers)
    pub nodes: Vec<BvhNode>,
    /// Sorted primitive indices
    pub primitives: Vec<u32>,
    /// World bounds
    pub bounds: AABB,
}

impl LinearBvh {
    /// Build BVH from primitives
    #[must_use]
    pub fn build(mut primitives: Vec<BvhPrimitive>) -> Self {
        if primitives.is_empty() {
            return Self {
                nodes: Vec::new(),
                primitives: Vec::new(),
                bounds: AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO),
            };
        }

        // Compute world bounds
        let mut bounds = primitives[0].aabb;
        for prim in &primitives[1..] {
            bounds = bounds.union(&prim.aabb);
        }

        // Compute Morton codes
        for prim in &mut primitives {
            let center = Vec3Fix::new(
                (prim.aabb.min.x + prim.aabb.max.x).half(),
                (prim.aabb.min.y + prim.aabb.max.y).half(),
                (prim.aabb.min.z + prim.aabb.max.z).half(),
            );
            prim.morton = point_to_morton(center, &bounds);
        }

        // Sort by Morton code (stable sort for determinism)
        primitives.sort_by_key(|p| p.morton);

        // Build tree with escape pointers
        let mut nodes = Vec::new();
        let prim_indices: Vec<u32> = primitives.iter().map(|p| p.index).collect();

        Self::build_recursive(&mut nodes, &primitives, 0, primitives.len(), ESCAPE_NONE);
        debug_assert!(Self::debug_verify_escape_forward(&nodes));

        Self {
            nodes,
            primitives: prim_indices,
            bounds,
        }
    }

    /// Debug-only invariant: every escape pointer must be either
    /// [`ESCAPE_NONE`] or a strictly-greater node index.
    /// Backward escape pointers form cycles in stackless traversal
    /// (see `find_pairs`) and drive unbounded push into the pair Vec
    /// until OOM — this was the root cause of the 5 GB / n=50 crash
    /// before the placeholder=0 rewrite of `build_recursive`.
    #[cfg(debug_assertions)]
    fn debug_verify_escape_forward(nodes: &[BvhNode]) -> bool {
        for (i, node) in nodes.iter().enumerate() {
            let esc = node.escape_idx();
            if esc == ESCAPE_NONE {
                continue;
            }
            if (esc as usize) <= i {
                return false;
            }
        }
        true
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    fn debug_verify_escape_forward(_nodes: &[BvhNode]) -> bool {
        true
    }

    /// Refit-only path (position update, tree structure preserved).
    ///
    /// # Preconditions
    /// - The number and identity of primitives is unchanged from the
    ///   most recent [`Self::build`] call (no primitive add / remove).
    /// - `new_aabbs_by_prim_index` maps each original primitive index
    ///   (the `index` field on [`BvhPrimitive`]) to its new AABB.
    /// - `new_aabbs_by_prim_index.len()` covers every primitive index
    ///   currently referenced by the tree.
    ///
    /// # Effect
    /// Refreshes each leaf node's AABB from the input mapping and
    /// propagates the union upwards to the root, keeping the tree
    /// structure and Morton ordering intact. Should be paired with a
    /// periodic full rebuild ([`Self::build`]) when the geometry has
    /// deformed significantly (large primitive AABB churn degrades
    /// SAH efficiency of the retained tree).
    ///
    /// # Determinism
    /// The refit is a pure function of `new_aabbs_by_prim_index` and
    /// the retained tree structure; nodes are visited in flat-array
    /// index order (bottom-up), matching the discipline required by
    /// `deterministic-physics-lockstep-discipline` skill §1 経路 5.
    ///
    /// # Status
    /// Skeleton API committed as part of Turn D next-step
    /// (Fix128 broad-phase 維持 + BVH refit + hash grid ハイブリッド).
    /// The bottom-up propagation body is scheduled for the follow-up
    /// commit that also wires the refit path into the adaptive
    /// sub-stepping loop; the current signature is stable so
    /// downstream integration can begin.
    pub fn refit_leaves(&mut self, new_aabbs_by_prim_index: &[AABB]) {
        // Step 1 — Leaf refit: aggregate each leaf's primitive AABBs
        // from `new_aabbs_by_prim_index`, requantise, and write back.
        for node in &mut self.nodes {
            let prim_count = ((node.prim_count_escape >> 24) & 0xFF) as usize;
            if prim_count == 0 {
                continue; // Internal node — handled in Step 2.
            }
            let prim_start = node.first_child_or_prim as usize;
            let first_prim_idx = self.primitives[prim_start] as usize;
            debug_assert!(
                first_prim_idx < new_aabbs_by_prim_index.len(),
                "primitive index out of range for refit input"
            );
            let mut leaf_aabb = new_aabbs_by_prim_index[first_prim_idx];
            for k in 1..prim_count {
                let prim_idx = self.primitives[prim_start + k] as usize;
                debug_assert!(prim_idx < new_aabbs_by_prim_index.len());
                leaf_aabb = leaf_aabb.union(&new_aabbs_by_prim_index[prim_idx]);
            }
            node.aabb_min = aabb_to_i32_min(&leaf_aabb);
            node.aabb_max = aabb_to_i32_max(&leaf_aabb);
        }

        // Step 2 — Bottom-up internal-node union: walk the flat node
        // array in reverse (DFS pre-order is written left-to-right, so
        // reverse walk guarantees children are refit before parents).
        //
        // Left child index  = `first_child_or_prim` of the internal.
        // Right child index = `escape_idx` of the left child (points
        //                     just past the left subtree, i.e. at the
        //                     next sibling). When the escape jumps out
        //                     of bounds or back onto the parent, there
        //                     is no right sibling to fold in.
        //
        // The union is computed directly in the i32 quantised domain
        // so no `i32 → Fix128 → i32` inverse-quantisation is required,
        // keeping the refit bit-exact and division-free (skill §1
        // 経路 2 — no CORDIC / rounding involved).
        let node_count = self.nodes.len();
        for i in (0..node_count).rev() {
            let prim_count = (self.nodes[i].prim_count_escape >> 24) & 0xFF;
            if prim_count > 0 {
                continue; // Leaf — already refit in Step 1.
            }
            let left_idx = self.nodes[i].first_child_or_prim as usize;
            debug_assert!(left_idx < node_count, "left child index out of range");

            let left_escape = (self.nodes[left_idx].prim_count_escape & 0x00FF_FFFF) as usize;
            let has_right = left_escape < node_count && left_escape != i;

            let left_min = self.nodes[left_idx].aabb_min;
            let left_max = self.nodes[left_idx].aabb_max;

            let (new_min, new_max) = if has_right {
                let right_min = self.nodes[left_escape].aabb_min;
                let right_max = self.nodes[left_escape].aabb_max;
                (
                    [
                        left_min[0].min(right_min[0]),
                        left_min[1].min(right_min[1]),
                        left_min[2].min(right_min[2]),
                    ],
                    [
                        left_max[0].max(right_max[0]),
                        left_max[1].max(right_max[1]),
                        left_max[2].max(right_max[2]),
                    ],
                )
            } else {
                (left_min, left_max)
            };

            self.nodes[i].aabb_min = new_min;
            self.nodes[i].aabb_max = new_max;
        }
    }

    /// Recursive build with escape pointer assignment
    /// Recursively build a subtree covering `primitives[start..end]`.
    ///
    /// Returns the subtree size (number of nodes pushed by this call
    /// including nested descendants). The caller uses this to compute
    /// the right sibling's starting index without a second pass.
    ///
    /// Escape pointer convention: the LEFT child of an internal node
    /// cannot know its correct escape (= right sibling's index) at
    /// push time because the right subtree has not been built yet. We
    /// push it with the reserved placeholder `LEFT_ESCAPE_PLACEHOLDER`
    /// (= 0, which is never a legal escape target because index 0 is
    /// the tree root and escapes always move forward), and after the
    /// left subtree is fully built we scan `nodes[left_idx..right_idx]`
    /// in a single pass, replacing every placeholder with the real
    /// right sibling index.
    ///
    /// The previous implementation walked only the leftmost spine via
    /// a recursive `update_escape` and matched on `old_escape == root+1`,
    /// which unintentionally overwrote nested placeholders at deeper
    /// levels with the outer right-sibling index — producing backward
    /// escape pointers that formed cycles in stackless traversal and
    /// drove `find_pairs` into OOM push loops (5 GB at n=50 pile).
    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        primitives: &[BvhPrimitive],
        start: usize,
        end: usize,
        escape_idx: u32,
    ) -> usize {
        let node_idx = nodes.len();
        let count = end - start;

        // Compute bounds for this range
        let mut aabb = primitives[start].aabb;
        for prim in &primitives[start + 1..end] {
            aabb = aabb.union(&prim.aabb);
        }

        if count <= 4 {
            // Leaf node
            nodes.push(BvhNode::leaf(&aabb, start as u32, count as u32, escape_idx));
            return 1;
        }

        // Internal node — its own escape is already the caller-supplied value
        // (either parent's escape for a right child, or the placeholder set by
        // the outer level for a left child; the outer level will fix that).
        nodes.push(BvhNode::internal(&aabb, 0, escape_idx));

        // Find split point using Morton codes
        let mid = Self::find_split(primitives, start, end);

        // Build LEFT subtree with placeholder escape — no descendant can know
        // the right sibling's index yet.
        let left_idx = nodes.len();
        let left_size =
            Self::build_recursive(nodes, primitives, start, mid, LEFT_ESCAPE_PLACEHOLDER);
        let right_idx = (left_idx + left_size) as u32;

        // Build RIGHT subtree — its rightmost descendants inherit our escape.
        let right_size = Self::build_recursive(nodes, primitives, mid, end, escape_idx);

        // Update this node's first_child pointer
        nodes[node_idx].first_child_or_prim = left_idx as u32;

        // Fix every placeholder in the left subtree in a single pass.
        // Only nodes whose escape currently equals LEFT_ESCAPE_PLACEHOLDER
        // are touched — this is position-independent and safe under
        // arbitrary recursion depth.
        let end_of_left = left_idx + left_size;
        for slot in &mut nodes[left_idx..end_of_left] {
            if slot.escape_idx() == LEFT_ESCAPE_PLACEHOLDER {
                let prim_count = slot.prim_count();
                slot.prim_count_escape = (prim_count << 24) | (right_idx & 0x00FFFFFF);
            }
        }

        1 + left_size + right_size
    }

    /// Find split point based on highest differing bit in Morton codes
    fn find_split(primitives: &[BvhPrimitive], start: usize, end: usize) -> usize {
        let first_code = primitives[start].morton;
        let last_code = primitives[end - 1].morton;

        if first_code == last_code {
            return (start + end) / 2;
        }

        // Find highest differing bit
        let diff = first_code ^ last_code;
        let highest_bit = 63 - diff.leading_zeros() as usize;

        // Binary search for split point
        let mut lo = start;
        let mut hi = end - 1;

        while lo < hi {
            let mid = (lo + hi) / 2;
            let mid_code = primitives[mid].morton;
            let split_bit = (mid_code >> highest_bit) & 1;
            let first_bit = (first_code >> highest_bit) & 1;

            if split_bit == first_bit {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        lo.max(start + 1).min(end - 1)
    }

    /// Stackless query: find all primitives intersecting the given AABB
    ///
    /// **Zero heap allocation** during traversal - uses only a single index variable.
    /// This is the "黒焦げ" (crispy) optimization.
    #[inline]
    #[must_use]
    pub fn query(&self, aabb: &AABB) -> Vec<u32> {
        let mut result = Vec::new();

        if self.nodes.is_empty() {
            return result;
        }

        // Compress query AABB to i32 for fast comparison
        let query_min = aabb_to_i32_min(aabb);
        let query_max = aabb_to_i32_max(aabb);

        self.query_stackless(&query_min, &query_max, &mut result);
        result
    }

    /// Stackless traversal core - single register index, no stack
    #[inline]
    fn query_stackless(&self, query_min: &[i32; 3], query_max: &[i32; 3], result: &mut Vec<u32>) {
        let mut idx = 0u32;

        while idx != ESCAPE_NONE && (idx as usize) < self.nodes.len() {
            let node = &self.nodes[idx as usize];

            if node.intersects_i32(query_min, query_max) {
                // AABB hit
                if node.is_leaf() {
                    // Collect primitives
                    let start = node.first_child_or_prim as usize;
                    let count = node.prim_count() as usize;
                    for i in start..start + count {
                        if i < self.primitives.len() {
                            result.push(self.primitives[i]);
                        }
                    }
                    // Move to escape (next sibling or up)
                    idx = node.escape_idx();
                } else {
                    // Descend to first child
                    idx = node.first_child_or_prim;
                }
            } else {
                // AABB miss - skip entire subtree via escape pointer
                idx = node.escape_idx();
            }
        }
    }

    /// Query with callback (even more allocation-free)
    #[inline]
    pub fn query_callback<F>(&self, aabb: &AABB, mut callback: F)
    where
        F: FnMut(u32),
    {
        if self.nodes.is_empty() {
            return;
        }

        let query_min = aabb_to_i32_min(aabb);
        let query_max = aabb_to_i32_max(aabb);

        let mut idx = 0u32;

        while idx != ESCAPE_NONE && (idx as usize) < self.nodes.len() {
            let node = &self.nodes[idx as usize];

            if node.intersects_i32(&query_min, &query_max) {
                if node.is_leaf() {
                    let start = node.first_child_or_prim as usize;
                    let count = node.prim_count() as usize;
                    for i in start..start + count {
                        if i < self.primitives.len() {
                            callback(self.primitives[i]);
                        }
                    }
                    idx = node.escape_idx();
                } else {
                    idx = node.first_child_or_prim;
                }
            } else {
                idx = node.escape_idx();
            }
        }
    }

    /// Broad-phase candidate pairs `(i, j)` with `i < j` (primitive indices).
    ///
    /// Every pair whose AABBs overlap is reported. The result is a
    /// **superset**: each leaf is queried with its own (i32-quantised) AABB,
    /// which is the union of the (at most four, see `build`) primitives it
    /// holds, so a pair can be reported when only their leaves overlap. Callers run a
    /// narrow phase on the result (see `PhysicsWorld::detect_collisions`).
    ///
    /// Before 1.1.1 every primitive was queried with the BVH's *world*
    /// bounds, so this returned all `n·(n-1)/2` pairs regardless of overlap
    /// and the broad phase was effectively disabled (O(n²) narrow phase).
    #[must_use]
    pub fn find_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        if self.nodes.is_empty() || self.primitives.is_empty() {
            return pairs;
        }
        let mut candidates: Vec<u32> = Vec::new();
        for node in &self.nodes {
            if !node.is_leaf() {
                continue;
            }
            let start = node.first_child_or_prim as usize;
            let count = node.prim_count() as usize;
            candidates.clear();
            self.query_stackless(&node.aabb_min, &node.aabb_max, &mut candidates);
            for i in start..start + count {
                let Some(&prim_i) = self.primitives.get(i) else {
                    continue;
                };
                for &prim_j in &candidates {
                    if prim_i < prim_j {
                        pairs.push((prim_i, prim_j));
                    }
                }
            }
        }
        // Remove duplicates (deterministic)
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Node / leaf / primitive counts for diagnostics.
    #[must_use]
    pub fn stats(&self) -> BvhStats {
        let mut stats = BvhStats {
            node_count: self.nodes.len(),
            primitive_count: self.primitives.len(),
            ..BvhStats::default()
        };

        for node in &self.nodes {
            if node.is_leaf() {
                stats.leaf_count += 1;
                stats.max_leaf_prims = stats.max_leaf_prims.max(node.prim_count() as usize);
            } else {
                stats.internal_count += 1;
            }
        }

        stats
    }
}

/// BVH statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct BvhStats {
    /// Total number of BVH nodes
    pub node_count: usize,
    /// Number of leaf nodes
    pub leaf_count: usize,
    /// Number of internal nodes
    pub internal_count: usize,
    /// Total number of primitives stored
    pub primitive_count: usize,
    /// Maximum primitives in any leaf
    pub max_leaf_prims: usize,
}

// ---------------------------------------------------------------------------
// Hybrid broadphase: static BVH + sparse hash grid + large-body BVH
// ---------------------------------------------------------------------------

/// Raw signed 128-bit value of a [`Fix128`] (integer part in the high 64 bits).
#[inline]
const fn fix_raw(v: Fix128) -> i128 {
    ((v.hi as i128) << 64) | (v.lo as i128)
}

/// Grid coordinate of `v` for cells of raw size `2^shift`, clamped to `i64`.
///
/// Clamping is monotone, so two boxes whose cell ranges overlap before the
/// clamp still overlap after it: a far-away body can only share a cell with
/// more bodies (a larger candidate set), never miss one.
#[inline]
fn cell_coord(v: Fix128, shift: u32) -> i64 {
    let c = fix_raw(v) >> shift;
    if c > i64::MAX as i128 {
        i64::MAX
    } else if c < i64::MIN as i128 {
        i64::MIN
    } else {
        c as i64
    }
}

/// Largest axis extent of `aabb` in raw units (saturating, 0 for an inverted box).
#[inline]
fn raw_extent(aabb: &AABB) -> u128 {
    let axis = |lo: Fix128, hi: Fix128| -> u128 {
        match fix_raw(hi).checked_sub(fix_raw(lo)) {
            Some(d) if d > 0 => d as u128,
            Some(_) => 0,
            None => u128::MAX,
        }
    };
    axis(aabb.min.x, aabb.max.x)
        .max(axis(aabb.min.y, aabb.max.y))
        .max(axis(aabb.min.z, aabb.max.z))
}

/// Largest number of grid cells a body may occupy along one axis; bodies
/// wider than this go to the large-body layer instead of the grid.
const HYBRID_MAX_CELLS_PER_AXIS: u32 = 3;

/// Broad-phase that splits the bodies of a frame into three layers:
///
/// - **static** bodies go into a [`LinearBvh`] that is rebuilt only when the
///   static set changes (a body added, removed or moved, detected by comparing
///   the staged set with the one the BVH was built from);
/// - **small dynamic** bodies go into a sparse hash grid (cell coordinates
///   sorted, no dense array), rebuilt every frame in `O(N log N)`; the cell
///   size is chosen every frame as the smallest power of two at or above the
///   median extent of the dynamic bodies;
/// - **large dynamic** bodies (wider than [`HYBRID_MAX_CELLS_PER_AXIS`]
///   cells) go into a per-frame [`LinearBvh`], so a few big bodies neither
///   inflate the cell size for everyone nor occupy hundreds of cells.
///
/// [`Self::query_pairs`] reports every pair of staged bodies whose boxes
/// overlap (inclusive, [`AABB::intersects`]) and that are not both static,
/// as `(smaller id, larger id)` sorted ascending. The set is exact: every
/// candidate from any layer is checked against the exact boxes before it is
/// reported, so the result does not depend on the cell size or on BVH
/// quantisation, only the work does.
///
/// # Per-frame flow
/// 1. [`Self::clear_dynamic`]
/// 2. [`Self::insert_dynamic`] for every body with a collision box
///    (`is_static` routes it to the static layer)
/// 3. [`Self::build_dynamic`]
/// 4. [`Self::query_pairs`]
///
/// # Determinism
/// Cell coordinates are integer shifts of the raw fixed-point value, grid
/// entries are sorted by `(cell, slot)`, both BVHs walk a flat node array in
/// a fixed order, and the output is sorted; the result is a pure function of
/// the staged bodies and does not depend on their insertion order.
///
/// Not yet wired into `PhysicsWorld` (the world broadphase enum gains a
/// variant for it in a later change); only the unit tests below drive it.
// ALLOW-DEAD: world wiring of the hybrid broadphase lands in a later change
#[allow(dead_code)]
pub(crate) struct BroadphaseHybrid {
    /// Static bodies staged this frame, in insertion order.
    staged_static: Vec<(u32, AABB)>,
    /// Dynamic bodies staged this frame, in insertion order.
    staged_dynamic: Vec<(u32, AABB)>,
    /// The static set `static_bvh` was built from (BVH primitive index = position).
    static_set: Vec<(u32, AABB)>,
    /// BVH over `static_set`.
    static_bvh: LinearBvh,
    /// How many times the static BVH has been rebuilt.
    static_rebuilds: u64,
    /// Raw cell size is `2^cell_shift` (raw units: 2^64 = one world unit).
    cell_shift: u32,
    /// Small dynamic bodies: id, box, cell of the min corner.
    small: Vec<(u32, AABB, [i64; 3])>,
    /// Grid entries `(cell, index into small)`, sorted.
    entries: Vec<([i64; 3], u32)>,
    /// Large dynamic bodies: id, box (BVH primitive index = position).
    large: Vec<(u32, AABB)>,
    /// BVH over `large`, rebuilt every frame.
    large_bvh: LinearBvh,
}

// ALLOW-DEAD: world wiring of the hybrid broadphase lands in a later change
#[allow(dead_code)]
impl BroadphaseHybrid {
    /// An empty broad-phase; the cell size is chosen per frame from the bodies.
    #[must_use]
    pub fn new() -> Self {
        Self {
            staged_static: Vec::new(),
            staged_dynamic: Vec::new(),
            static_set: Vec::new(),
            static_bvh: LinearBvh::build(Vec::new()),
            static_rebuilds: 0,
            cell_shift: 64,
            small: Vec::new(),
            entries: Vec::new(),
            large: Vec::new(),
            large_bvh: LinearBvh::build(Vec::new()),
        }
    }

    /// Stage one body for this frame. `is_static` bodies form the static
    /// layer; call after [`Self::clear_dynamic`], once per body.
    pub fn insert_dynamic(&mut self, body_id: u32, aabb: AABB, is_static: bool) {
        if is_static {
            self.staged_static.push((body_id, aabb));
        } else {
            self.staged_dynamic.push((body_id, aabb));
        }
    }

    /// Forget the bodies staged for the previous frame. The static BVH is
    /// kept; it is rebuilt by [`Self::build_dynamic`] only if the static set
    /// staged next differs from the one it was built from.
    pub fn clear_dynamic(&mut self) {
        self.staged_static.clear();
        self.staged_dynamic.clear();
    }

    /// Finish the frame: rebuild the static BVH if the static set changed,
    /// pick the cell size, split the dynamic bodies into grid and large
    /// layers, and sort the grid entries.
    pub fn build_dynamic(&mut self) {
        if self.staged_static != self.static_set {
            self.static_set.clone_from(&self.staged_static);
            self.static_bvh = Self::bvh_over(&self.static_set);
            self.static_rebuilds += 1;
        }

        self.cell_shift = Self::auto_cell_shift(&self.staged_dynamic);
        let shift = self.cell_shift;
        // A body wider than this spans more than HYBRID_MAX_CELLS_PER_AXIS cells.
        let large_above: u128 = if shift + 2 >= 127 {
            u128::MAX
        } else {
            u128::from(HYBRID_MAX_CELLS_PER_AXIS - 1) << shift
        };

        self.small.clear();
        self.entries.clear();
        self.large.clear();
        for &(id, aabb) in &self.staged_dynamic {
            if raw_extent(&aabb) > large_above {
                self.large.push((id, aabb));
                continue;
            }
            let lo = [
                cell_coord(aabb.min.x, shift),
                cell_coord(aabb.min.y, shift),
                cell_coord(aabb.min.z, shift),
            ];
            let hi = [
                cell_coord(aabb.max.x, shift).max(lo[0]),
                cell_coord(aabb.max.y, shift).max(lo[1]),
                cell_coord(aabb.max.z, shift).max(lo[2]),
            ];
            let slot = self.small.len() as u32;
            self.small.push((id, aabb, lo));
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        self.entries.push(([x, y, z], slot));
                    }
                }
            }
        }
        self.entries.sort_unstable();
        self.large_bvh = Self::bvh_over(&self.large);
    }

    /// Every pair of staged bodies whose boxes overlap and that are not both
    /// static, written to `out` as `(smaller id, larger id)` in ascending
    /// order. Returns the number of candidate pairs whose exact boxes were
    /// compared (the broad-phase work, `>= out.len()`).
    pub fn query_pairs(&self, out: &mut Vec<(u32, u32)>) -> u64 {
        out.clear();
        let mut tested: u64 = 0;
        let push = |a: u32, b: u32, out: &mut Vec<(u32, u32)>| {
            if a != b {
                out.push(if a < b { (a, b) } else { (b, a) });
            }
        };

        // 1. Small × small: pairs sharing a cell, reported only in the cell
        //    holding the min corner of the two boxes' intersection, so each
        //    pair is checked once without a dedup pass.
        let entries = &self.entries;
        let mut start = 0;
        while start < entries.len() {
            let cell = entries[start].0;
            let mut end = start + 1;
            while end < entries.len() && entries[end].0 == cell {
                end += 1;
            }
            for i in start..end {
                let (id_a, aabb_a, lo_a) = &self.small[entries[i].1 as usize];
                for e in &entries[i + 1..end] {
                    let (id_b, aabb_b, lo_b) = &self.small[e.1 as usize];
                    let owner = [
                        lo_a[0].max(lo_b[0]),
                        lo_a[1].max(lo_b[1]),
                        lo_a[2].max(lo_b[2]),
                    ];
                    if owner != cell {
                        continue;
                    }
                    tested += 1;
                    if aabb_a.intersects(aabb_b) {
                        push(*id_a, *id_b, out);
                    }
                }
            }
            start = end;
        }

        // 2. Small × large.
        for (id_a, aabb_a, _) in &self.small {
            self.large_bvh.query_callback(aabb_a, |p| {
                let (id_b, aabb_b) = &self.large[p as usize];
                tested += 1;
                if aabb_a.intersects(aabb_b) {
                    push(*id_a, *id_b, out);
                }
            });
        }

        // 3. Large × large (each unordered pair once: higher BVH index only).
        for (k, (id_a, aabb_a)) in self.large.iter().enumerate() {
            self.large_bvh.query_callback(aabb_a, |p| {
                if (p as usize) <= k {
                    return;
                }
                let (id_b, aabb_b) = &self.large[p as usize];
                tested += 1;
                if aabb_a.intersects(aabb_b) {
                    push(*id_a, *id_b, out);
                }
            });
        }

        // 4. Dynamic × static (static × static pairs are never reported).
        if !self.static_set.is_empty() {
            let small = self.small.iter().map(|(id, aabb, _)| (id, aabb));
            let large = self.large.iter().map(|(id, aabb)| (id, aabb));
            for (id_a, aabb_a) in small.chain(large) {
                self.static_bvh.query_callback(aabb_a, |p| {
                    let (id_b, aabb_b) = &self.static_set[p as usize];
                    tested += 1;
                    if aabb_a.intersects(aabb_b) {
                        push(*id_a, *id_b, out);
                    }
                });
            }
        }

        out.sort_unstable();
        // A body staged twice (same id) would otherwise appear twice.
        out.dedup();
        tested
    }

    /// A BVH over `set`, primitive index = position in `set`.
    fn bvh_over(set: &[(u32, AABB)]) -> LinearBvh {
        LinearBvh::build(
            set.iter()
                .enumerate()
                .map(|(k, &(_, aabb))| BvhPrimitive {
                    aabb,
                    index: k as u32,
                    morton: 0,
                })
                .collect(),
        )
    }

    /// `shift` such that the raw cell size `2^shift` is the smallest power of
    /// two at or above the median extent of `bodies` (the largest extent when
    /// the median is zero, one world unit when every extent is zero).
    fn auto_cell_shift(bodies: &[(u32, AABB)]) -> u32 {
        let mut extents: Vec<u128> = bodies.iter().map(|(_, aabb)| raw_extent(aabb)).collect();
        if extents.is_empty() {
            return 64;
        }
        let mid = extents.len() / 2;
        let (_, &mut median, _) = extents.select_nth_unstable(mid);
        let size = if median > 0 {
            median
        } else {
            extents.iter().copied().max().unwrap_or(0)
        };
        if size == 0 {
            return 64;
        }
        // smallest s with 2^s >= size, kept below the i128 sign bit
        (128 - (size - 1).leading_zeros()).min(125)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_morton_code() {
        let code1 = morton_code(0, 0, 0);
        assert_eq!(code1, 0);

        let code2 = morton_code(1, 0, 0);
        let code3 = morton_code(0, 1, 0);
        let code4 = morton_code(0, 0, 1);

        assert!(code2 != code3);
        assert!(code3 != code4);
    }

    #[test]
    fn test_morton_ordering() {
        let code1 = morton_code(100, 100, 100);
        let code2 = morton_code(101, 100, 100);
        let code3 = morton_code(200, 200, 200);

        let diff12 = (code1 as i64 - code2 as i64).abs();
        let diff13 = (code1 as i64 - code3 as i64).abs();

        assert!(diff12 < diff13);
    }

    #[test]
    fn test_bvh_build() {
        let primitives = vec![
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 1, 1)),
                index: 0,
                morton: 0,
            },
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(2, 2, 2), Vec3Fix::from_int(3, 3, 3)),
                index: 1,
                morton: 0,
            },
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(5, 5, 5), Vec3Fix::from_int(6, 6, 6)),
                index: 2,
                morton: 0,
            },
        ];

        let bvh = LinearBvh::build(primitives);

        assert!(!bvh.nodes.is_empty());
        assert_eq!(bvh.primitives.len(), 3);

        // Check stats
        let stats = bvh.stats();
        assert!(stats.leaf_count > 0);
    }

    #[test]
    fn test_bvh_query() {
        let primitives = vec![
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 1, 1)),
                index: 0,
                morton: 0,
            },
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(10, 10, 10), Vec3Fix::from_int(11, 11, 11)),
                index: 1,
                morton: 0,
            },
        ];

        let bvh = LinearBvh::build(primitives);

        // Query near first primitive
        let query_aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(2, 2, 2));
        let results = bvh.query(&query_aabb);

        assert!(results.contains(&0), "Should find primitive 0");
    }

    #[test]
    fn test_stackless_query_callback() {
        let primitives = vec![
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 1, 1)),
                index: 10,
                morton: 0,
            },
            BvhPrimitive {
                aabb: AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(2, 2, 2)),
                index: 20,
                morton: 0,
            },
        ];

        let bvh = LinearBvh::build(primitives);
        let mut found = Vec::new();

        bvh.query_callback(
            &AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 1, 1)),
            |idx| found.push(idx),
        );

        assert!(!found.is_empty(), "Should find overlapping primitives");
    }

    #[test]
    fn test_expand_bits() {
        assert_eq!(expand_bits(0), 0);
        assert_eq!(expand_bits(1), 1);
        assert_eq!(expand_bits(0b11), 0b1001);
        assert_eq!(expand_bits(0b111), 0b1001001);
    }

    /// `refit_leaves` must update leaf AABBs when the underlying
    /// primitive positions have moved. Internal nodes are allowed to
    /// keep their build-time AABB in this leaf-only pass; broad-phase
    /// queries remain correct because false positives get filtered by
    /// the narrow-phase collision stage.
    #[test]
    fn refit_leaves_updates_leaf_aabbs() {
        let prims: Vec<BvhPrimitive> = (0..4)
            .map(|i| BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::from_int(i * 2, 0, 0),
                    Vec3Fix::from_int(i * 2 + 1, 1, 1),
                ),
                index: i as u32,
                morton: 0,
            })
            .collect();
        let mut bvh = LinearBvh::build(prims);
        assert!(!bvh.nodes.is_empty(), "BVH must be non-empty");

        // Move every primitive up by 10 world units on the Y axis.
        let new_aabbs: Vec<AABB> = (0..4)
            .map(|i| {
                AABB::new(
                    Vec3Fix::from_int(i * 2, 10, 0),
                    Vec3Fix::from_int(i * 2 + 1, 11, 1),
                )
            })
            .collect();

        bvh.refit_leaves(&new_aabbs);

        // Every leaf's decoded AABB must now sit inside the shifted
        // Y range. We check that at least one leaf reflects the shift.
        let mut leaf_shifted = false;
        for node in &bvh.nodes {
            let prim_count = (node.prim_count_escape >> 24) & 0xFF;
            if prim_count > 0 {
                // Leaf: y_min quantised, but the sign should now be
                // strictly greater than the pre-refit y_min (0).
                if node.aabb_min[1] > 0 {
                    leaf_shifted = true;
                    break;
                }
            }
        }
        assert!(
            leaf_shifted,
            "at least one leaf must reflect the upward primitive shift"
        );
    }

    /// `refit_leaves` must propagate the leaf refit into internal
    /// nodes via bottom-up union. The root AABB is the union of all
    /// leaf AABBs, so shifting every primitive on a single axis must
    /// grow the root AABB on that axis.
    #[test]
    fn refit_leaves_bottom_up_updates_internal_aabbs() {
        let prims: Vec<BvhPrimitive> = (0..8)
            .map(|i| BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::from_int(i * 2, 0, 0),
                    Vec3Fix::from_int(i * 2 + 1, 1, 1),
                ),
                index: i as u32,
                morton: 0,
            })
            .collect();
        let mut bvh = LinearBvh::build(prims);
        assert!(!bvh.nodes.is_empty(), "BVH must be non-empty");

        let root_aabb_max_before = bvh.nodes[0].aabb_max;

        // Shift every primitive upward by 100 world units on Y.
        let new_aabbs: Vec<AABB> = (0..8)
            .map(|i| {
                AABB::new(
                    Vec3Fix::from_int(i * 2, 100, 0),
                    Vec3Fix::from_int(i * 2 + 1, 101, 1),
                )
            })
            .collect();

        bvh.refit_leaves(&new_aabbs);

        let root_aabb_max_after = bvh.nodes[0].aabb_max;
        assert!(
            root_aabb_max_after[1] > root_aabb_max_before[1],
            "root aabb_max[y] must grow after bottom-up refit (before={}, after={})",
            root_aabb_max_before[1],
            root_aabb_max_after[1]
        );
    }

    // ---------------------------------------------------------------------
    // BroadphaseHybrid oracles
    // ---------------------------------------------------------------------

    /// One staged body: id, box, static flag.
    type Body = (u32, AABB, bool);

    /// Deterministic 64-bit LCG (Knuth MMIX constants).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        /// Uniform in `[0, side)` with a resolution of 1/1000.
        fn coord(&mut self, side: i64) -> Fix128 {
            Fix128::from_ratio((self.next() % (side as u64 * 1000)) as i64, 1000)
        }
        fn point(&mut self, side: i64) -> Vec3Fix {
            Vec3Fix::new(self.coord(side), self.coord(side), self.coord(side))
        }
    }

    fn ball(center: Vec3Fix, half_milli: i64) -> AABB {
        let h = Fix128::from_ratio(half_milli, 1000);
        AABB::from_center_half(center, Vec3Fix::new(h, h, h))
    }

    /// Side of a cube holding `n` bodies about `spacing_milli / 1000` units apart.
    fn side_for(n: usize, spacing_milli: i64) -> i64 {
        let mut k = 1i64;
        while k * k * k < n as i64 {
            k += 1;
        }
        ((k * spacing_milli + 999) / 1000).max(1)
    }

    /// `n` dynamic boxes of half-size 0.5, spread out (a few overlaps).
    fn scene_uniform(n: usize, seed: u64) -> Vec<Body> {
        let mut r = Lcg(seed);
        let side = side_for(n, 1600);
        (0..n as u32)
            .map(|i| (i, ball(r.point(side), 500), false))
            .collect()
    }

    /// `n` dynamic boxes packed densely over a wide static floor.
    fn scene_pile(n: usize, seed: u64) -> Vec<Body> {
        let mut r = Lcg(seed);
        let side = side_for(n, 900);
        let mut v: Vec<Body> = (1..=n as u32)
            .map(|i| (i, ball(r.point(side), 500), false))
            .collect();
        let floor = AABB::new(
            Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-1),
                Fix128::from_int(-5),
            ),
            Vec3Fix::new(
                Fix128::from_int(side + 5),
                Fix128::from_ratio(1, 10),
                Fix128::from_int(side + 5),
            ),
        );
        v.push((0, floor, true));
        v
    }

    /// As `scene_uniform`, but every 50th body has half-size 3 (2 % large).
    fn scene_mixed(n: usize, seed: u64) -> Vec<Body> {
        let mut r = Lcg(seed);
        let side = side_for(n, 1600);
        (0..n as u32)
            .map(|i| {
                let half = if i % 50 == 7 { 3000 } else { 500 };
                (i, ball(r.point(side), half), false)
            })
            .collect()
    }

    /// 95 % static boxes, 5 % dynamic, densely packed.
    fn scene_mostly_static(n: usize, seed: u64) -> Vec<Body> {
        let mut r = Lcg(seed);
        let side = side_for(n, 1100);
        (0..n as u32)
            .map(|i| (i, ball(r.point(side), 500), i % 20 != 3))
            .collect()
    }

    /// `scene_uniform` with two bodies thrown 10^12 units away (one pair of
    /// them overlapping each other far from everything else).
    fn scene_far_outlier(n: usize, seed: u64) -> Vec<Body> {
        let mut v = scene_uniform(n, seed);
        let far = Fix128::from_int(1_000_000_000_000);
        v[0].1 = ball(Vec3Fix::new(far, -far, far), 500);
        v[1].1 = ball(Vec3Fix::new(far + Fix128::from_ratio(1, 2), -far, far), 500);
        v
    }

    /// Exact reference: overlapping pairs not both static, sorted.
    fn brute_force(bodies: &[Body]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for (k, a) in bodies.iter().enumerate() {
            for b in &bodies[k + 1..] {
                if (a.2 && b.2) || a.0 == b.0 || !a.1.intersects(&b.1) {
                    continue;
                }
                out.push((a.0.min(b.0), a.0.max(b.0)));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn stage(h: &mut BroadphaseHybrid, bodies: &[Body]) {
        h.clear_dynamic();
        for &(id, aabb, st) in bodies {
            h.insert_dynamic(id, aabb, st);
        }
        h.build_dynamic();
    }

    /// Stage `bodies`, query, and return `(pairs, tested)`.
    fn run(h: &mut BroadphaseHybrid, bodies: &[Body]) -> (Vec<(u32, u32)>, u64) {
        stage(h, bodies);
        let mut out = Vec::new();
        let tested = h.query_pairs(&mut out);
        (out, tested)
    }

    /// Structural bounds after `build_dynamic`: no body occupies more than
    /// 27 cells (3 per axis), and the grid holds no more than 27 entries per
    /// small body, whatever the coordinates (sparse grid, no dense array).
    fn assert_grid_bounds(h: &BroadphaseHybrid, label: &str) {
        let mut per_slot = vec![0usize; h.small.len()];
        for &(_, s) in &h.entries {
            per_slot[s as usize] += 1;
        }
        let worst = per_slot.iter().copied().max().unwrap_or(0);
        assert!(worst <= 27, "{label}: a body occupies {worst} cells (> 27)");
        assert!(
            h.entries.len() <= 27 * h.small.len(),
            "{label}: {} grid entries for {} small bodies",
            h.entries.len(),
            h.small.len()
        );
    }

    type SceneFn = fn(usize, u64) -> Vec<Body>;
    const SCENES: [(&str, SceneFn); 5] = [
        ("uniform", scene_uniform),
        ("pile", scene_pile),
        ("mixed", scene_mixed),
        ("mostly_static", scene_mostly_static),
        ("far_outlier", scene_far_outlier),
    ];

    /// The reported pairs equal the brute-force exact overlap set (no missed
    /// pair, no extra pair) on every scene at 100 and 1000 bodies, and the
    /// work stays within a small multiple of the output: candidate pairs
    /// compared `<= 6 * (pairs + bodies)`. With a cell size taken from the
    /// largest body, or with the large bodies left in the grid, the mixed
    /// scene compares tens of times more candidates and fails the bound.
    #[test]
    fn hybrid_pairs_equal_brute_force_and_work_is_bounded() {
        for (name, scene) in SCENES {
            for (n, seed) in [(100usize, 0x5eed_0001u64), (1000, 0x5eed_0002)] {
                let bodies = scene(n, seed);
                let mut h = BroadphaseHybrid::new();
                let (pairs, tested) = run(&mut h, &bodies);
                let expected = brute_force(&bodies);
                let label = format!("{name} n={n}");
                assert!(!expected.is_empty(), "{label}: scene has no overlaps");
                let missed: Vec<_> = expected
                    .iter()
                    .filter(|p| pairs.binary_search(p).is_err())
                    .collect();
                assert!(
                    missed.is_empty(),
                    "{label}: missed {} pairs, e.g. {:?}",
                    missed.len(),
                    &missed[..missed.len().min(4)]
                );
                assert_eq!(
                    pairs, expected,
                    "{label}: pair set differs from brute force"
                );
                assert!(
                    tested >= pairs.len() as u64,
                    "{label}: tested {tested} < pairs"
                );
                let bound = 6 * (pairs.len() as u64 + n as u64);
                assert!(
                    tested <= bound,
                    "{label}: compared {tested} candidates (bound {bound}, {} pairs)",
                    pairs.len()
                );
                assert_grid_bounds(&h, &label);
            }
        }
    }

    /// The mixed scene routes exactly its 2 % large bodies to the large-body
    /// layer and keeps the cell size at the small bodies' scale.
    #[test]
    fn hybrid_routes_large_bodies_out_of_the_grid() {
        let bodies = scene_mixed(1000, 0x5eed_0003);
        let mut h = BroadphaseHybrid::new();
        stage(&mut h, &bodies);
        let mut large: Vec<u32> = h.large.iter().map(|&(id, _)| id).collect();
        large.sort_unstable();
        let expected: Vec<u32> = (0..1000u32).filter(|i| i % 50 == 7).collect();
        assert_eq!(large, expected);
        assert_eq!(h.small.len(), 1000 - expected.len());
        // small extent is exactly 1 unit -> cell 2^64 raw = 1 unit
        assert_eq!(h.cell_shift, 64);
    }

    /// The cell size is the smallest power of two at or above the median
    /// body extent, for several body scales (so it follows the bodies, not a
    /// constant and not the largest body).
    #[test]
    fn hybrid_cell_size_follows_the_median_extent() {
        // (half-size in thousandths, expected cell shift)
        for (half_milli, shift) in [(500i64, 64u32), (250, 63), (1000, 65), (3000, 67), (40, 61)] {
            let mut r = Lcg(0x5eed_0004);
            let mut bodies: Vec<Body> = (0..200u32)
                .map(|i| (i, ball(r.point(20), half_milli), false))
                .collect();
            // a few much larger bodies must not move the cell size
            for b in bodies.iter_mut().take(5) {
                b.1 = ball(r.point(20), half_milli * 10);
            }
            let mut h = BroadphaseHybrid::new();
            stage(&mut h, &bodies);
            assert_eq!(h.cell_shift, shift, "half {half_milli}/1000");
            let cell = 1u128 << h.cell_shift;
            let median = 2 * (u128::from(half_milli as u64) << 64) / 1000;
            assert!(
                cell >= median && cell < 2 * median + 2,
                "cell {cell} median {median}"
            );
        }
    }

    /// Adding, removing or moving a static body is detected: the static BVH
    /// is rebuilt (once per change) and the pairs stay exact; an unchanged
    /// static set is not rebuilt.
    #[test]
    fn hybrid_detects_static_set_changes() {
        let mut bodies = scene_mostly_static(1000, 0x5eed_0005);
        let mut h = BroadphaseHybrid::new();
        let check = |h: &mut BroadphaseHybrid, bodies: &[Body], label: &str| {
            let (pairs, _) = run(h, bodies);
            assert_eq!(pairs, brute_force(bodies), "{label}");
        };
        check(&mut h, &bodies, "initial");
        assert_eq!(h.static_rebuilds, 1);
        check(&mut h, &bodies, "unchanged");
        assert_eq!(
            h.static_rebuilds, 1,
            "unchanged static set must not rebuild"
        );

        // move a static body onto a dynamic one
        let dyn_center = bodies[3].1.min;
        let st = bodies
            .iter()
            .position(|b| b.2 && !b.1.intersects(&bodies[3].1))
            .unwrap();
        bodies[st].1 = AABB::new(dyn_center, bodies[3].1.max);
        check(&mut h, &bodies, "moved static");
        assert_eq!(h.static_rebuilds, 2);
        let a = bodies[st].0.min(3);
        let b = bodies[st].0.max(3);
        assert!(
            run(&mut h, &bodies).0.contains(&(a, b)),
            "moved static must touch body 3"
        );

        // add a static body overlapping dynamic body 23
        bodies.push((5000, bodies[23].1, true));
        check(&mut h, &bodies, "added static");
        assert_eq!(h.static_rebuilds, 3);

        // remove it again
        bodies.pop();
        check(&mut h, &bodies, "removed static");
        assert_eq!(h.static_rebuilds, 4);

        // a dynamic body turning static is a static-set change too
        bodies[43].2 = true;
        check(&mut h, &bodies, "dynamic became static");
        assert_eq!(h.static_rebuilds, 5);
    }

    /// Identical input gives an identical pair sequence, from a fresh or a
    /// reused instance, and the insertion order does not matter.
    #[test]
    fn hybrid_is_deterministic_and_order_independent() {
        for (name, scene) in SCENES {
            let bodies = scene(1000, 0x5eed_0006);
            let (a, ta) = run(&mut BroadphaseHybrid::new(), &bodies);
            let mut reused = BroadphaseHybrid::new();
            let _ = run(&mut reused, &scene_uniform(300, 1));
            let (b, tb) = run(&mut reused, &bodies);
            assert_eq!(a, b, "{name}: reused instance");
            assert_eq!(ta, tb, "{name}: reused instance work");
            let mut rev = bodies.clone();
            rev.reverse();
            let (c, _) = run(&mut BroadphaseHybrid::new(), &rev);
            assert_eq!(a, c, "{name}: reversed insertion order");
        }
    }

    /// Degenerate inputs: empty, one body, all bodies at one point (with and
    /// without extent), and boxes at ±4·10^18 units (near the Fix128 range).
    #[test]
    fn hybrid_degenerate_inputs() {
        let mut h = BroadphaseHybrid::new();
        let (p, t) = run(&mut h, &[]);
        assert!(p.is_empty() && t == 0);

        let one = [(9u32, ball(Vec3Fix::from_int(1, 2, 3), 500), false)];
        let (p, _) = run(&mut h, &one);
        assert!(p.is_empty());

        for half in [0i64, 500] {
            let at: Vec<Body> = (0..100u32)
                .map(|i| (i, ball(Vec3Fix::from_int(7, -7, 7), half), i % 10 == 0))
                .collect();
            let (p, _) = run(&mut h, &at);
            assert_eq!(p, brute_force(&at), "all at one point, half {half}");
            assert!(!p.is_empty());
            assert_grid_bounds(&h, "one point");
        }

        let big = 4_000_000_000_000_000_000i64;
        let mut huge: Vec<Body> = Vec::new();
        let mut r = Lcg(0x5eed_0007);
        for i in 0..60u32 {
            let base = match i % 3 {
                0 => Vec3Fix::from_int(big, big, big),
                1 => Vec3Fix::from_int(-big, big, -big),
                _ => Vec3Fix::from_int(0, 0, 0),
            };
            let c = base + r.point(4);
            huge.push((i, ball(c, 500), i % 7 == 0));
        }
        let (p, _) = run(&mut h, &huge);
        assert_eq!(p, brute_force(&huge), "huge coordinates");
        assert!(!p.is_empty());
        assert_grid_bounds(&h, "huge coordinates");
    }

    /// `refit_leaves` must preserve the flat tree structure (node
    /// count + primitive order) so downstream queries keep walking
    /// the same skeleton with only updated AABBs.
    #[test]
    fn refit_leaves_preserves_tree_structure() {
        let prims: Vec<BvhPrimitive> = (0..4)
            .map(|i| BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::from_int(i * 3, 0, 0),
                    Vec3Fix::from_int(i * 3 + 1, 1, 1),
                ),
                index: i as u32,
                morton: 0,
            })
            .collect();
        let mut bvh = LinearBvh::build(prims);
        let node_count_before = bvh.nodes.len();
        let primitives_before = bvh.primitives.clone();

        let new_aabbs: Vec<AABB> = (0..4)
            .map(|i| {
                AABB::new(
                    Vec3Fix::from_int(i * 3, 5, 0),
                    Vec3Fix::from_int(i * 3 + 1, 6, 1),
                )
            })
            .collect();

        bvh.refit_leaves(&new_aabbs);

        assert_eq!(
            bvh.nodes.len(),
            node_count_before,
            "node count must not change"
        );
        assert_eq!(
            bvh.primitives, primitives_before,
            "primitive ordering must not change"
        );
    }

    #[test]
    fn test_node_packing() {
        // Test that prim_count and escape_idx pack/unpack correctly
        let aabb = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));

        let leaf = BvhNode::leaf(&aabb, 100, 5, 0x123456);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.prim_count(), 5);
        assert_eq!(leaf.escape_idx(), 0x123456);
        assert_eq!(leaf.first_child_or_prim, 100);

        let internal = BvhNode::internal(&aabb, 50, 0xABCDEF);
        assert!(!internal.is_leaf());
        assert_eq!(internal.prim_count(), 0);
        assert_eq!(internal.escape_idx(), 0xABCDEF);
        assert_eq!(internal.first_child_or_prim, 50);
    }

    // ---- find_pairs (1.1.1: leaf AABB で query、全 pair 返却 bug の修正) ----

    fn unit_box(x: i64, y: i64, z: i64) -> AABB {
        AABB::new(
            Vec3Fix::from_int(x, y, z),
            Vec3Fix::from_int(x + 1, y + 1, z + 1),
        )
    }

    fn build_from(boxes: &[AABB]) -> LinearBvh {
        LinearBvh::build(
            boxes
                .iter()
                .enumerate()
                .map(|(i, b)| BvhPrimitive {
                    aabb: *b,
                    index: i as u32,
                    morton: 0,
                })
                .collect(),
        )
    }

    #[test]
    fn find_pairs_reports_no_pairs_between_far_clusters() {
        // 3 cluster × 4 箱 (cluster 内は互いに重なる、cluster 間は対角線上に 1000 離れる)
        // 1.1.0 以前はこれが 66 pair 全部返っていた
        // 1.2.0: `point_to_morton` は最大軸で等方正規化するので x 軸一列の異方配置でも
        // cluster が混ざらない (下の anisotropic test)、ここは対角配置
        let mut boxes = Vec::new();
        for c in 0..3i64 {
            let base = c * 1000;
            boxes.push(AABB::new(
                Vec3Fix::from_int(base, base, base),
                Vec3Fix::from_int(base + 2, base + 2, base + 2),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base + 1, base, base),
                Vec3Fix::from_int(base + 3, base + 2, base + 2),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base, base + 1, base),
                Vec3Fix::from_int(base + 2, base + 3, base + 2),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base + 1, base + 1, base + 1),
                Vec3Fix::from_int(base + 3, base + 3, base + 3),
            ));
        }
        let pairs = build_from(&boxes).find_pairs();
        assert_eq!(pairs.len(), 18, "cluster 内 6 pair × 3 のみ: {pairs:?}");
        for (i, j) in &pairs {
            assert_eq!(i / 4, j / 4, "cluster を跨ぐ pair {i}-{j}");
            assert!(i < j);
        }
    }

    #[test]
    fn find_pairs_anisotropic_clusters_along_x_do_not_mix() {
        // 3 cluster × 4 箱を x 軸一列に 1000 間隔で並べる (world 2003 × 3 × 3)
        // 1.1.0 の軸別正規化では y/z の上位 bit が x cluster より優先され leaf が world 全体に
        // 伸びて 66 pair 全部が候補になった; 等方正規化後は cluster 内 6 × 3 = 18 のみ
        let mut boxes = Vec::new();
        for c in 0..3i64 {
            let base = c * 1000;
            boxes.push(AABB::new(
                Vec3Fix::from_int(base, 0, 0),
                Vec3Fix::from_int(base + 2, 2, 2),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base + 1, 1, 0),
                Vec3Fix::from_int(base + 3, 3, 2),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base, 1, 1),
                Vec3Fix::from_int(base + 2, 3, 3),
            ));
            boxes.push(AABB::new(
                Vec3Fix::from_int(base + 1, 0, 1),
                Vec3Fix::from_int(base + 3, 2, 3),
            ));
        }
        let prims: Vec<BvhPrimitive> = boxes
            .iter()
            .enumerate()
            .map(|(i, aabb)| BvhPrimitive {
                aabb: *aabb,
                index: i as u32,
                morton: 0,
            })
            .collect();
        let bvh = LinearBvh::build(prims);
        let pairs = bvh.find_pairs();
        for &(i, j) in &pairs {
            assert_eq!(i / 4, j / 4, "cluster を跨ぐ pair {i}-{j}");
        }
        assert_eq!(pairs.len(), 18, "cluster 内 6 pair × 3 のみ: {pairs:?}");
    }

    #[test]
    fn find_pairs_is_a_superset_of_true_overlaps_and_strictly_smaller_than_all_pairs() {
        // 8×8 の格子に 1×1 箱を間隔 3 で置く (重なりなし) + 各箱に少し重なる相棒を追加
        let mut boxes = Vec::new();
        for gx in 0..8i64 {
            for gz in 0..8i64 {
                boxes.push(unit_box(gx * 3, 0, gz * 3));
            }
        }
        let n_base = boxes.len();
        for k in 0..n_base {
            let b = boxes[k];
            let half = Fix128::from_ratio(1, 2);
            boxes.push(AABB::new(
                Vec3Fix::new(b.min.x + half, b.min.y, b.min.z),
                Vec3Fix::new(b.max.x + half, b.max.y, b.max.z),
            ));
        }
        let n = boxes.len();
        let pairs = build_from(&boxes).find_pairs();
        // (1) 真に重なる pair (各箱とその相棒 = n_base 組) は必ず含む
        for k in 0..n_base {
            let want = (k as u32, (k + n_base) as u32);
            assert!(pairs.contains(&want), "missing overlapping pair {want:?}");
        }
        // (2) 全 pair より真に小さい (broad-phase が機能している)
        let all = n * (n - 1) / 2;
        assert!(
            pairs.len() < all / 4,
            "{} of {all} pairs = broad-phase が効いていない",
            pairs.len()
        );
        // (3) 報告 pair は近傍のみ (leaf 4 個の和より遠い箱は絶対に組にならない: 格子間隔 3 × 8 で 21 以上離れる箱は不可)
        for (i, j) in &pairs {
            let a = boxes[*i as usize];
            let b = boxes[*j as usize];
            let dx = (a.min.x - b.min.x).abs();
            let dz = (a.min.z - b.min.z).abs();
            assert!(
                dx < Fix128::from_int(21) && dz < Fix128::from_int(21),
                "far pair {i}-{j}"
            );
        }
        // (4) 決定論: 同じ入力で同じ結果、昇順・重複なし
        let again = build_from(&boxes).find_pairs();
        assert_eq!(pairs, again);
        for w in pairs.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn find_pairs_touching_and_empty_cases() {
        // 接触ちょうど (max == min) は i32 量子化で overlap 扱い (superset に含まれる)
        // 4 個以下は単一 leaf になるので全 pair が候補 = superset 契約の範囲内
        let touching =
            build_from(&[unit_box(0, 0, 0), unit_box(1, 0, 0), unit_box(50, 50, 50)]).find_pairs();
        assert!(touching.contains(&(0, 1)));
        // leaf が分かれる規模 (2 cluster × 5) では遠い cluster 間の pair は出ない
        let mut two = Vec::new();
        for k in 0..5i64 {
            two.push(unit_box(k, 0, 0));
        }
        for k in 0..5i64 {
            two.push(unit_box(500 + k, 500, 500));
        }
        let pairs = build_from(&two).find_pairs();
        assert!(pairs.contains(&(0, 1)) && pairs.contains(&(5, 6)));
        assert!(
            pairs.iter().all(|(i, j)| (*i < 5) == (*j < 5)),
            "cluster 間 pair: {pairs:?}"
        );
        assert!(build_from(&[]).find_pairs().is_empty());
        assert!(build_from(&[unit_box(0, 0, 0)]).find_pairs().is_empty());
    }

    // ---- morton / quantisation / split helpers -------------------------

    #[test]
    fn expand_bits_and_morton_code_interleave_exactly() {
        assert_eq!(expand_bits(0), 0);
        assert_eq!(expand_bits(1), 1);
        assert_eq!(expand_bits(0b11), 0b1001);
        assert_eq!(expand_bits(0b111), 0b1001001);
        assert_eq!(expand_bits(0b1000), 1 << 9);
        assert_eq!(expand_bits(1 << 20), 1 << 60);
        // x → bit 0、y → bit 1、z → bit 2 の順で interleave
        assert_eq!(morton_code(1, 0, 0), 0b001);
        assert_eq!(morton_code(0, 1, 0), 0b010);
        assert_eq!(morton_code(0, 0, 1), 0b100);
        assert_eq!(morton_code(2, 2, 2), 0b111000);
        assert_eq!(morton_code(0b11, 0b01, 0b10), 0b001_011 | 0b100_000);
        // 21 bit で clamp
        assert_eq!(morton_code(1 << 30, 0, 0), morton_code((1 << 21) - 1, 0, 0));
        // 単調性: 同じ y,z なら x が大きいほど code が大きい
        assert!(morton_code(5, 3, 3) < morton_code(6, 3, 3));
        assert!(morton_code(3, 5, 3) < morton_code(3, 6, 3));
    }

    #[test]
    fn point_to_morton_normalises_isotropically_and_clamps() {
        let bounds = AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(8, 8, 8));
        assert_eq!(point_to_morton(Vec3Fix::from_int(0, 0, 0), &bounds), 0);
        // 中点 → 各軸の top bit (bit 20) だけ立つ → x: bit 60、y: bit 61、z: bit 62
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(4, 0, 0), &bounds),
            1 << 60
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(0, 4, 0), &bounds),
            1 << 61
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(0, 0, 4), &bounds),
            1 << 62
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(4, 4, 4), &bounds),
            0b111 << 60
        );
        // 1/4 → bit 19 → x なら bit 57
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(2, 0, 0), &bounds),
            1 << 57
        );
        // 範囲外は clamp: 負 → 0、max 以上 → 全 bit
        assert_eq!(point_to_morton(Vec3Fix::from_int(-5, -5, -5), &bounds), 0);
        let all = morton_code(0x1FFFFF, 0x1FFFFF, 0x1FFFFF);
        assert_eq!(point_to_morton(Vec3Fix::from_int(8, 8, 8), &bounds), all);
        assert_eq!(point_to_morton(Vec3Fix::from_int(99, 99, 99), &bounds), all);
        // 等方正規化 (1.2.0): 退化 (size 0) 軸も最大軸 8 で割る → y = 3 は 3/8 = 0b011 (bit 19, 18)
        let flat = AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(8, 0, 8));
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(4, 3, 4), &flat),
            morton_code(1 << 20, 0b11 << 18, 1 << 20)
        );
        // 全軸退化は 0
        let point = AABB::new(Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(1, 1, 1));
        assert_eq!(point_to_morton(Vec3Fix::from_int(1, 1, 1), &point), 0);
        // 異方 world (2003 × 3 × 3): x が支配的、y/z の jitter は下位 bit にしか入らない
        let aniso = AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(2003, 3, 3));
        let near_a = point_to_morton(Vec3Fix::from_int(1000, 0, 0), &aniso);
        let near_b = point_to_morton(Vec3Fix::from_int(1001, 3, 3), &aniso);
        let far = point_to_morton(Vec3Fix::from_int(2000, 0, 0), &aniso);
        assert!(
            near_a.abs_diff(near_b) < near_a.abs_diff(far),
            "x cluster が y/z jitter より優先されない"
        );
        // 単調性 (x 方向)
        let a = point_to_morton(Vec3Fix::from_int(1, 1, 1), &bounds);
        let b = point_to_morton(Vec3Fix::from_int(2, 1, 1), &bounds);
        assert!(a < b);
    }

    #[test]
    fn fix128_to_i32_floor_and_ceil_semantics() {
        assert_eq!(fix128_floor_i32(Fix128::from_ratio(5, 2)), 2);
        assert_eq!(fix128_floor_i32(Fix128::from_int(2)), 2);
        assert_eq!(fix128_floor_i32(Fix128::from_ratio(-5, 2)), -3);
        assert_eq!(fix128_ceil_i32(Fix128::from_ratio(5, 2)), 3);
        assert_eq!(fix128_ceil_i32(Fix128::from_int(2)), 2);
        assert_eq!(fix128_ceil_i32(Fix128::from_ratio(-5, 2)), -2);
        assert_eq!(fix128_ceil_i32(Fix128 { hi: 7, lo: 1 }), 8);
        // clamp
        assert_eq!(fix128_floor_i32(Fix128::from_int(1 << 40)), i32::MAX);
        assert_eq!(fix128_ceil_i32(Fix128::from_int(-(1 << 40))), i32::MIN);
        let aabb = AABB::new(
            Vec3Fix::new(
                Fix128::from_ratio(-1, 2),
                Fix128::from_ratio(3, 2),
                Fix128::ZERO,
            ),
            Vec3Fix::new(
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(5, 2),
                Fix128::from_int(3),
            ),
        );
        assert_eq!(aabb_to_i32_min(&aabb), [-1, 1, 0]);
        assert_eq!(aabb_to_i32_max(&aabb), [1, 3, 3]);
    }

    #[test]
    fn find_split_splits_at_highest_differing_morton_bit() {
        let mk = |codes: &[u64]| -> Vec<BvhPrimitive> {
            codes
                .iter()
                .enumerate()
                .map(|(i, &m)| BvhPrimitive {
                    aabb: unit_box(0, 0, 0),
                    index: i as u32,
                    morton: m,
                })
                .collect()
        };
        // 最上位差分 bit (bit 3) が 0 の group [0b0001, 0b0011, 0b0101] と 1 の group [0b1000, 0b1001] → split = 3
        let prims = mk(&[0b0001, 0b0011, 0b0101, 0b1000, 0b1001]);
        assert_eq!(LinearBvh::find_split(&prims, 0, 5), 3);
        // 先頭 1 個だけ 0 group → 1
        let prims2 = mk(&[0b0001, 0b1000, 0b1001, 0b1100]);
        assert_eq!(LinearBvh::find_split(&prims2, 0, 4), 1);
        // 全部同じ code → 中央
        let same = mk(&[7, 7, 7, 7, 7, 7]);
        assert_eq!(LinearBvh::find_split(&same, 0, 6), 3);
        // 部分区間 [2, 6): codes [0b10, 0b10, 0b11, 0b11] は bit 0 で分かれる → 4
        let prims3 = mk(&[0, 0, 0b10, 0b10, 0b11, 0b11]);
        assert_eq!(LinearBvh::find_split(&prims3, 2, 6), 4);
        // 結果は必ず (start, end) の内側
        let prims4 = mk(&[0, 0b1111]);
        assert_eq!(LinearBvh::find_split(&prims4, 0, 2), 1);
    }

    #[test]
    fn build_and_stats_reflect_leaf_partitioning() {
        let boxes: Vec<AABB> = (0..9i64).map(|i| unit_box(i * 3, i * 3, i * 3)).collect();
        let bvh = build_from(&boxes);
        let st = bvh.stats();
        assert_eq!(st.primitive_count, 9);
        assert_eq!(st.node_count, bvh.nodes.len());
        assert_eq!(st.leaf_count + st.internal_count, st.node_count);
        assert!(
            st.leaf_count >= 3,
            "9 prims / 4 per leaf → 3 leaf 以上: {st:?}"
        );
        assert!(st.max_leaf_prims <= 4 && st.max_leaf_prims >= 1);
        // 全 primitive index が 1 回ずつ現れる
        let mut idx: Vec<u32> = bvh.primitives.clone();
        idx.sort_unstable();
        assert_eq!(idx, (0..9).collect::<Vec<u32>>());
        // 空
        let empty = LinearBvh::build(Vec::new());
        assert_eq!(empty.stats().node_count, 0);
        assert!(empty.query(&unit_box(0, 0, 0)).is_empty());
    }

    #[test]
    fn get_aabb_reconstructs_integer_boxes_exactly_and_conservatively_rounds_fractions() {
        // 整数 AABB は圧縮 (i32) → 復元で bit 一致 (負値含む)
        let boxes = [
            unit_box(0, 0, 0),
            AABB::new(Vec3Fix::from_int(-7, 3, -12), Vec3Fix::from_int(5, 9, -2)),
            AABB::new(
                Vec3Fix::from_int(-1000, -1000, -1000),
                Vec3Fix::from_int(1000, 1000, 1000),
            ),
        ];
        for (i, b) in boxes.iter().enumerate() {
            let leaf = BvhNode::leaf(b, 4, 2, 9);
            assert_eq!(leaf.get_aabb(), *b, "leaf {i}");
            let internal = BvhNode::internal(b, 1, ESCAPE_NONE);
            assert_eq!(internal.get_aabb(), *b, "internal {i}");
            // 復元した箱は自分自身の i32 テストと整合する
            assert!(leaf.intersects_i32(&leaf.aabb_min, &leaf.aabb_max));
        }

        // 小数 AABB: min は floor、max は ceil で保守的に膨らむ (元の箱を必ず包む)
        let frac = AABB::new(
            Vec3Fix::new(
                Fix128::from_ratio(-5, 2), // -2.5 → -3
                Fix128::from_ratio(1, 4),  //  0.25 → 0
                Fix128::from_int(3),       //  3 → 3 (整数はそのまま)
            ),
            Vec3Fix::new(
                Fix128::from_ratio(7, 2),  //  3.5 → 4
                Fix128::from_ratio(-1, 4), // -0.25 → 0
                Fix128::from_int(6),       //  6 → 6
            ),
        );
        let node = BvhNode::leaf(&frac, 0, 1, ESCAPE_NONE);
        let got = node.get_aabb();
        assert_eq!(got.min, Vec3Fix::from_int(-3, 0, 3));
        assert_eq!(got.max, Vec3Fix::from_int(4, 0, 6));
        assert_eq!(
            got.union(&frac),
            got,
            "reconstructed box must enclose original"
        );

        // build 後の root node の get_aabb は bvh.bounds を包み、整数入力なら一致
        let bvh = build_from(&boxes);
        let root = bvh.nodes[0].get_aabb();
        assert_eq!(root, bvh.bounds);
        assert_eq!(root.min, Vec3Fix::from_int(-1000, -1000, -1000));
        assert_eq!(root.max, Vec3Fix::from_int(1000, 1000, 1000));
        // 全 leaf の get_aabb は root に含まれる
        for n in &bvh.nodes {
            let a = n.get_aabb();
            assert_eq!(root.union(&a), root);
        }
    }

    #[test]
    fn point_to_morton_with_non_zero_bounds_origin_and_build_orders_by_center() {
        // bounds [10,18]³: 点 14 は中点 → top bit、点 10 は 0、点 12 は 1/4
        let bounds = AABB::new(Vec3Fix::from_int(10, 10, 10), Vec3Fix::from_int(18, 18, 18));
        assert_eq!(point_to_morton(Vec3Fix::from_int(10, 10, 10), &bounds), 0);
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(14, 10, 10), &bounds),
            1 << 60
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(10, 12, 10), &bounds),
            1 << 58
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(10, 10, 16), &bounds),
            (1 << 62) | (1 << 59)
        );
        // build は AABB の中心で並べる: サイズが増える箱を x 昇順に置くと primitives は [0,1,2]
        // (中心を min-max で計算する変異なら [2,1,0] になる)
        let boxes = [
            AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(1, 1, 1)),
            AABB::new(Vec3Fix::from_int(10, 0, 0), Vec3Fix::from_int(12, 2, 2)),
            AABB::new(Vec3Fix::from_int(20, 0, 0), Vec3Fix::from_int(23, 3, 3)),
            AABB::new(Vec3Fix::from_int(30, 0, 0), Vec3Fix::from_int(34, 4, 4)),
            AABB::new(Vec3Fix::from_int(40, 0, 0), Vec3Fix::from_int(45, 5, 5)),
        ];
        let bvh = build_from(&boxes);
        assert_eq!(bvh.primitives, vec![0, 1, 2, 3, 4]);
        assert_eq!(bvh.bounds.min, Vec3Fix::ZERO);
        assert_eq!(bvh.bounds.max, Vec3Fix::from_int(45, 5, 5));
        // 逆順に与えても中心順に並ぶ
        let rev: Vec<AABB> = boxes.iter().rev().copied().collect();
        assert_eq!(build_from(&rev).primitives, vec![4, 3, 2, 1, 0]);
    }

    // ---- mutation-kill tests (cargo-mutants missed list, 2026-09-15) ----

    /// `expand_bits` line 27 `<<` → `>>`: input bits 8..=20 only reach their
    /// final slot through the `<< 16` stage (after stage 1 they sit at 8..=15
    /// and 32..=36, and the mask of stage 2 keeps 24..=31 / 48..=52), so a
    /// `>> 16` drops every one of them. Exact expectations: 0x100 → bit 24,
    /// 0x1FFFFF → every third bit 0..=60 = 0x1249249249249249, 0b1011 → bits
    /// 0 / 3 / 9 = 0x209, alternating 0x155555 → 0x1041041041041041.
    #[test]
    fn expand_bits_exact_values_for_high_input_bits() {
        assert_eq!(expand_bits(0x100), 1 << 24);
        assert_eq!(expand_bits(0x1FFFFF), 0x1249_2492_4924_9249);
        assert_eq!(expand_bits(0b1011), 0x209);
        assert_eq!(expand_bits(0x15_5555), 0x1041_0410_4104_1041);
        assert_eq!(expand_bits(0xA_AAAA), 0x0208_2082_0820_8208);
    }

    /// `morton_code` lines 41 / 42 `(1 << 21) - 1` → `/ 1` / `+ 1` in the y / z
    /// clamp: 2^21 must clamp to 2^21 - 1 (all 21 bits set → every third bit of
    /// the code); the mutants let 2^21 through and `expand_bits` masks bit 21
    /// away, giving 0 for that axis. All three axes saturated = 2^63 - 1.
    #[test]
    fn morton_code_clamps_y_and_z_to_21_bits() {
        let all = 0x1249_2492_4924_9249u64;
        assert_eq!(morton_code(0, 1 << 21, 0), all << 1);
        assert_eq!(morton_code(0, 0, 1 << 21), all << 2);
        assert_eq!(morton_code(0, 1 << 30, 0), morton_code(0, (1 << 21) - 1, 0));
        assert_eq!(morton_code(0, 0, 1 << 30), morton_code(0, 0, (1 << 21) - 1));
        assert_eq!(
            morton_code(1 << 21, 1 << 21, 1 << 21),
            0x7FFF_FFFF_FFFF_FFFF
        );
        assert_eq!(morton_code(1 << 21, 1 << 21, 1 << 21), u64::MAX >> 1);
    }

    /// `point_to_morton` lines 63 / 66 `>` → `==`: the y (z) extent takes over
    /// the isotropic scale only when it is strictly larger than the running
    /// maximum. In a 2 × 8 × 2 box the point y = 4 is 4/8 → bit 20 of the y
    /// axis → Morton bit 61; with the scale stuck at x = 2 it would be 4/2 = 2
    /// → saturated. x = 1 must likewise be scaled by 8 (1/8 → bit 18 → bit 54)
    /// and not by its own extent 2 (1/2 → bit 60). Max corner → all 21 bits on
    /// every axis, min corner → 0.
    #[test]
    fn point_to_morton_scale_picks_strictly_larger_y_and_z_extent() {
        let tall = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 8, 2));
        assert_eq!(point_to_morton(Vec3Fix::from_int(0, 4, 0), &tall), 1 << 61);
        assert_eq!(point_to_morton(Vec3Fix::from_int(1, 0, 0), &tall), 1 << 54);
        let deep = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 2, 8));
        assert_eq!(point_to_morton(Vec3Fix::from_int(0, 0, 4), &deep), 1 << 62);
        assert_eq!(point_to_morton(Vec3Fix::from_int(0, 1, 0), &deep), 1 << 55);
        // max corner: the long axis saturates, the short axes are 2/8 = 1/4 → bit 19
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(2, 8, 2), &tall),
            morton_code(1 << 19, 0x1F_FFFF, 1 << 19)
        );
        assert_eq!(
            point_to_morton(Vec3Fix::from_int(2, 2, 8), &deep),
            morton_code(1 << 19, 1 << 19, 0x1F_FFFF)
        );
        assert_eq!(point_to_morton(Vec3Fix::ZERO, &tall), 0);
        assert_eq!(point_to_morton(Vec3Fix::ZERO, &deep), 0);
    }

    /// `point_to_morton` line 69 `||` → `&&`: inverted bounds (max < min) have
    /// a negative scale and must map to 0. With `&&` the quantiser runs and
    /// (2 - 4) / -4 = 1/2 sets bit 20 on every axis (0b111 << 60).
    #[test]
    fn point_to_morton_inverted_bounds_map_to_zero() {
        let inverted = AABB {
            min: Vec3Fix::from_int(4, 4, 4),
            max: Vec3Fix::ZERO,
        };
        assert_eq!(point_to_morton(Vec3Fix::from_int(2, 2, 2), &inverted), 0);
        assert_eq!(point_to_morton(Vec3Fix::from_int(9, -9, 0), &inverted), 0);
    }

    /// `LinearBvh::build` lines 289 / 290 `+` → `-` / `*` in the y / z centre:
    /// boxes y ∈ [1, 3] (centre 2) and y ∈ [-3, -1] (centre -2), identical on
    /// x / z, given in the order [+, -] → the negative box sorts first
    /// (`primitives == [1, 0]`). `(min - max) / 2` is -1 for both and
    /// `(min * max) / 2` is 3/2 for both, so the stable sort would keep [0, 1].
    #[test]
    fn build_sorts_by_y_and_z_centre_not_by_difference_or_product() {
        let y_pos = AABB::new(Vec3Fix::from_int(0, 1, 0), Vec3Fix::from_int(2, 3, 2));
        let y_neg = AABB::new(Vec3Fix::from_int(0, -3, 0), Vec3Fix::from_int(2, -1, 2));
        assert_eq!(build_from(&[y_pos, y_neg]).primitives, vec![1, 0]);
        assert_eq!(build_from(&[y_neg, y_pos]).primitives, vec![0, 1]);
        let z_pos = AABB::new(Vec3Fix::from_int(0, 0, 1), Vec3Fix::from_int(2, 2, 3));
        let z_neg = AABB::new(Vec3Fix::from_int(0, 0, -3), Vec3Fix::from_int(2, 2, -1));
        assert_eq!(build_from(&[z_pos, z_neg]).primitives, vec![1, 0]);
        assert_eq!(build_from(&[z_neg, z_pos]).primitives, vec![0, 1]);
    }

    /// `debug_verify_escape_forward` line 320 `→ true` and line 322 `==` →
    /// `!=`: a backward escape pointer (node 2 → 1) and a self-loop (node 0 →
    /// 0) must be reported. Only the debug-profile body is under test here;
    /// the release stub always returns `true`.
    #[cfg(debug_assertions)]
    #[test]
    fn debug_verify_escape_forward_rejects_backward_pointer() {
        let aabb = unit_box(0, 0, 0);
        let backward = [
            BvhNode::internal(&aabb, 1, ESCAPE_NONE),
            BvhNode::leaf(&aabb, 0, 1, 2),
            BvhNode::leaf(&aabb, 1, 1, 1),
        ];
        assert!(!LinearBvh::debug_verify_escape_forward(&backward));
        let self_loop = [
            BvhNode::internal(&aabb, 1, 0),
            BvhNode::leaf(&aabb, 0, 1, 2),
        ];
        assert!(!LinearBvh::debug_verify_escape_forward(&self_loop));
    }

    /// Forward pointers and `ESCAPE_NONE` are accepted in both profiles
    /// (`debug_verify_escape_forward` line 335 release stub → `false`).
    #[test]
    fn debug_verify_escape_forward_accepts_forward_and_none() {
        let aabb = unit_box(0, 0, 0);
        let good = [
            BvhNode::internal(&aabb, 1, ESCAPE_NONE),
            BvhNode::leaf(&aabb, 0, 1, 2),
            BvhNode::leaf(&aabb, 1, 1, ESCAPE_NONE),
        ];
        assert!(LinearBvh::debug_verify_escape_forward(&good));
        assert!(LinearBvh::debug_verify_escape_forward(&[]));
        let built = build_from(&(0..9i64).map(|i| unit_box(i * 3, 0, 0)).collect::<Vec<_>>());
        assert!(LinearBvh::debug_verify_escape_forward(&built.nodes));
    }

    /// `refit_leaves` line 417 `&` → `^` / `|` and line 418 `<` → `==` / `>`,
    /// `!=` → `==`: the right child must be folded into its parent. 8 boxes
    /// along x → the root is internal over two subtrees and the last
    /// primitive is in the right one; only that primitive grows to y = 50, so
    /// the root's y max must become 50 (the left child alone would give 1).
    /// Symmetrically, growing primitive 0 (left subtree) to y = -40 must reach
    /// the root's min.
    #[test]
    fn refit_leaves_folds_right_child_into_parent() {
        let boxes: Vec<AABB> = (0..8i64).map(|i| unit_box(i * 3, 0, 0)).collect();
        let mut bvh = build_from(&boxes);
        assert!(!bvh.nodes[0].is_leaf(), "8 prims must not fit one leaf");
        let mut grown = boxes.clone();
        grown[7] = AABB::new(Vec3Fix::from_int(21, 0, 0), Vec3Fix::from_int(22, 50, 1));
        bvh.refit_leaves(&grown);
        assert_eq!(bvh.nodes[0].aabb_max, [22, 50, 1]);
        assert_eq!(bvh.nodes[0].aabb_min, [0, 0, 0]);

        let mut bvh2 = build_from(&boxes);
        let mut low = boxes.clone();
        low[0] = AABB::new(Vec3Fix::from_int(0, -40, 0), Vec3Fix::from_int(1, 1, 1));
        bvh2.refit_leaves(&low);
        assert_eq!(bvh2.nodes[0].aabb_min, [0, -40, 0]);
        assert_eq!(bvh2.nodes[0].aabb_max, [22, 1, 1]);
    }

    /// `refit_leaves` line 418 `<` → `<=` and `&&` → `||`: a left child whose
    /// escape points exactly one past the end (`== node_count`) has no right
    /// sibling, so the parent takes the left AABB and no node is indexed out
    /// of range (the mutants read `nodes[node_count]` and panic).
    #[test]
    fn refit_leaves_left_escape_at_node_count_means_no_right_sibling() {
        let leaf_box = unit_box(0, 0, 0);
        let mut bvh = LinearBvh {
            nodes: vec![
                BvhNode::internal(&unit_box(-5, -5, -5), 1, ESCAPE_NONE),
                BvhNode::leaf(&leaf_box, 0, 1, 2),
            ],
            primitives: vec![0],
            bounds: leaf_box,
        };
        bvh.refit_leaves(&[unit_box(3, 4, 5)]);
        assert_eq!(bvh.nodes[1].aabb_min, [3, 4, 5]);
        assert_eq!(bvh.nodes[1].aabb_max, [4, 5, 6]);
        assert_eq!(bvh.nodes[0].aabb_min, [3, 4, 5]);
        assert_eq!(bvh.nodes[0].aabb_max, [4, 5, 6]);
    }
}
