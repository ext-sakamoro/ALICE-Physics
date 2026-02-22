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
fn expand_bits(mut v: u64) -> u64 {
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
/// Coordinates should be normalized to [0, 2^21) range
#[inline]
pub fn morton_code(x: u64, y: u64, z: u64) -> u64 {
    let x = x.min((1 << 21) - 1);
    let y = y.min((1 << 21) - 1);
    let z = z.min((1 << 21) - 1);

    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Compute Morton code from a point within a bounding box
pub fn point_to_morton(point: Vec3Fix, bounds: &AABB) -> u64 {
    let size = bounds.max - bounds.min;

    // Compute normalized coordinates [0, 1] and clamp for negative/out-of-range
    let nx = if size.x.is_zero() {
        0u64
    } else {
        let t = (point.x - bounds.min.x) / size.x;
        if t.is_negative() { 0 }
        else if t.hi >= 1 { 0x1FFFFF }
        else { (t.lo >> 43) & 0x1FFFFF }
    };

    let ny = if size.y.is_zero() {
        0u64
    } else {
        let t = (point.y - bounds.min.y) / size.y;
        if t.is_negative() { 0 }
        else if t.hi >= 1 { 0x1FFFFF }
        else { (t.lo >> 43) & 0x1FFFFF }
    };

    let nz = if size.z.is_zero() {
        0u64
    } else {
        let t = (point.z - bounds.min.z) / size.z;
        if t.is_negative() { 0 }
        else if t.hi >= 1 { 0x1FFFFF }
        else { (t.lo >> 43) & 0x1FFFFF }
    };

    morton_code(nx, ny, nz)
}

// ============================================================================
// BVH Node (Stackless-Ready)
// ============================================================================

/// Sentinel value for "no escape" (end of traversal)
pub const ESCAPE_NONE: u32 = u32::MAX;

/// BVH node (32 bytes, cache-line friendly)
///
/// Layout optimized for stackless traversal:
/// - `escape_idx`: Next node to visit if AABB test fails (skip entire subtree)
/// - For leaves: escape_idx points to next sibling or parent's escape
/// - For internal nodes: escape_idx points to next subtree after both children
#[derive(Clone, Copy, Debug)]
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
    /// Maximum primitives per leaf (fits in 8 bits)
    pub const MAX_PRIMS_PER_LEAF: u32 = 255;

    /// Create internal node
    #[inline]
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
    /// `count` is saturated to [`Self::MAX_PRIMS_PER_LEAF`] (255) to prevent
    /// 8-bit overflow that would make the leaf appear as an internal node.
    #[inline]
    pub fn leaf(aabb: &AABB, first_prim: u32, count: u32, escape_idx: u32) -> Self {
        let clamped = count.min(Self::MAX_PRIMS_PER_LEAF);
        debug_assert!(count <= Self::MAX_PRIMS_PER_LEAF);
        Self {
            aabb_min: aabb_to_i32_min(aabb),
            first_child_or_prim: first_prim,
            aabb_max: aabb_to_i32_max(aabb),
            prim_count_escape: ((clamped & 0xFF) << 24) | (escape_idx & 0x00FFFFFF),
        }
    }

    /// Check if this is a leaf node
    #[inline]
    pub fn is_leaf(&self) -> bool {
        (self.prim_count_escape >> 24) > 0
    }

    /// Get primitive count (0 for internal nodes)
    #[inline]
    pub fn prim_count(&self) -> u32 {
        self.prim_count_escape >> 24
    }

    /// Get escape index (next node to visit on AABB miss)
    #[inline]
    pub fn escape_idx(&self) -> u32 {
        self.prim_count_escape & 0x00FFFFFF
    }

    /// Get AABB (reconstructed from compressed i32)
    #[inline]
    pub fn get_aabb(&self) -> AABB {
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
    pub fn intersects_i32(&self, query_min: &[i32; 3], query_max: &[i32; 3]) -> bool {
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
#[derive(Clone, Copy, Debug)]
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

        Self {
            nodes,
            primitives: prim_indices,
            bounds,
        }
    }

    /// Recursive build with escape pointer assignment
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
        } else {
            // Internal node - placeholder, will be updated after children
            nodes.push(BvhNode::internal(&aabb, 0, escape_idx));

            // Find split point using Morton codes
            let mid = Self::find_split(primitives, start, end);

            // Build left child (escape to right child)
            let left_idx = nodes.len();
            let _right_escape = node_idx + 1; // Temporary, will point to right child
            Self::build_recursive(nodes, primitives, start, mid, left_idx as u32 + 1);

            // Build right child (escape to parent's escape)
            let right_idx = nodes.len();
            Self::build_recursive(nodes, primitives, mid, end, escape_idx);

            // Update this node's first_child pointer
            nodes[node_idx].first_child_or_prim = left_idx as u32;

            // Update left subtree's escape pointers to point to right subtree
            Self::update_escape(nodes, left_idx, right_idx as u32);
        }

        node_idx
    }

    /// Update escape pointers in a subtree
    fn update_escape(nodes: &mut [BvhNode], root: usize, new_escape: u32) {
        let old_escape = nodes[root].escape_idx();

        // Only update if this node's escape pointed to the placeholder
        if old_escape == root as u32 + 1 {
            let prim_count = nodes[root].prim_count();
            nodes[root].prim_count_escape = (prim_count << 24) | (new_escape & 0x00FFFFFF);
        }

        // Recursively update children if internal node
        if !nodes[root].is_leaf() {
            let first_child = nodes[root].first_child_or_prim as usize;
            if first_child < nodes.len() {
                Self::update_escape(nodes, first_child, new_escape);
                if first_child + 1 < nodes.len() {
                    // Right child's escape is already correct (parent's escape)
                }
            }
        }
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

    /// Find all potentially colliding pairs (broad phase)
    /// Uses stackless traversal internally.
    pub fn find_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();

        if self.nodes.is_empty() || self.primitives.is_empty() {
            return pairs;
        }

        // For each primitive, query overlapping primitives
        // This is O(n * log n) average case with good spatial locality
        for &prim_i in &self.primitives {
            self.query_callback(&self.bounds, |prim_j| {
                if prim_i < prim_j {
                    pairs.push((prim_i, prim_j));
                }
            });
        }

        // Remove duplicates (deterministic)
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Get statistics about the BVH
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

#[cfg(test)]
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
}
