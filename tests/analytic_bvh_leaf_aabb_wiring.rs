//! Oracles for the production entry point of `alice_physics::bvh::BvhNode`
//! driven by `examples/bvh_leaf_aabb_roundtrip.rs`: `get_aabb`.
//!
//! # What this file is and is not
//!
//! `src/bvh.rs`'s own `#[cfg(test)]` block already covers: integer AABBs
//! (including negative ones and a `[-1000,1000]` cube) round-tripping
//! exactly through both `BvhNode::leaf` and `BvhNode::internal`, one
//! fractional box (`[-2.5,0.25,3]..[3.5,-0.25,6]`) floor/ceil-rounding
//! conservatively, and a built tree's root / every leaf enclosing a
//! hand-picked `bvh.bounds`. None of that is repeated here. What this
//! file adds:
//!
//! * fractional rounding at the i32 boundary itself -- a vanishingly
//!   small fraction on either side of an integer (not an exact half, not
//!   the module test's `0.25`/`-0.25`), confirming any nonzero fraction
//!   moves the bound by a full unit rather than only "large enough"
//!   fractions,
//! * the i32 representable range's own extremes (`i32::MIN` /
//!   `i32::MAX`) reconstructing exactly, with no clamping artefact on
//!   the reconstruction side (clamping on the *construction* side is
//!   already covered by `fix128_to_i32_floor_and_ceil_semantics` in the
//!   module's own tests),
//! * a single-primitive tree, where the root is itself the only leaf and
//!   `get_aabb` must reproduce that one primitive's box exactly with no
//!   intervening union,
//! * per-leaf tightness on a *built* multi-leaf tree: each leaf's
//!   `get_aabb` must equal the union of exactly the primitives
//!   `LinearBvh::find_split` assigned to that leaf -- computed here by
//!   reading `bvh.primitives[start..start+count]` and unioning the
//!   original input boxes by hand, not by calling any BVH method a
//!   second time,
//! * the empty-tree boundary case, where there is no node at all to call
//!   `get_aabb` on, and
//! * `get_aabb` is `#[must_use]`; this file does not re-test that
//!   (`-D warnings` already enforces it at compile time for every
//!   call site above).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::bvh::{BvhNode, BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

fn unit_box(x: i64, y: i64, z: i64) -> AABB {
    AABB::new(
        Vec3Fix::from_int(x, y, z),
        Vec3Fix::from_int(x + 1, y + 1, z + 1),
    )
}

// ============================================================================
// Fractional rounding -- vanishing fractions and the i32 extremes
// ============================================================================

/// A fraction of `1/1_000_000` on either side of an integer is still a
/// nonzero fraction: `floor` must drop the min by a full unit and `ceil`
/// must raise the max by a full unit, exactly like a fraction of `1/2`
/// would -- the rounding rule has no magnitude threshold.
#[test]
fn get_aabb_rounds_any_nonzero_fraction_by_a_full_unit() {
    let tiny = AABB::new(
        Vec3Fix::new(
            Fix128::from_int(2) - Fix128::from_ratio(1, 1_000_000), // 1.999999 -> floor -> 1
            Fix128::from_int(-2) - Fix128::from_ratio(1, 1_000_000), // -2.000001 -> floor -> -3
            Fix128::ZERO,
        ),
        Vec3Fix::new(
            Fix128::from_int(2) + Fix128::from_ratio(1, 1_000_000), // 2.000001 -> ceil -> 3
            Fix128::from_int(-2) + Fix128::from_ratio(1, 1_000_000), // -1.999999 -> ceil -> -1
            Fix128::ZERO,
        ),
    );
    let node = BvhNode::leaf(&tiny, 0, 1, 0);
    let got = node.get_aabb();
    assert_eq!(got.min, Vec3Fix::from_int(1, -3, 0));
    assert_eq!(got.max, Vec3Fix::from_int(3, -1, 0));
    // The reconstructed box must still enclose the original (conservative).
    assert_eq!(got.union(&tiny), got);
}

/// `i32::MIN` / `i32::MAX` are the exact edges of the compressed storage
/// range. `get_aabb` must reconstruct them bit-exactly via
/// `Fix128::from_int`, independent of whatever clamped them into that
/// range on the construction side.
#[test]
fn get_aabb_reconstructs_i32_extremes_exactly() {
    let extreme = AABB::new(
        Vec3Fix::from_int(i64::from(i32::MIN), 0, 0),
        Vec3Fix::from_int(i64::from(i32::MAX), 0, 0),
    );
    let leaf = BvhNode::leaf(&extreme, 0, 1, 0);
    let got_leaf = leaf.get_aabb();
    assert_eq!(got_leaf.min.x, Fix128::from_int(i64::from(i32::MIN)));
    assert_eq!(got_leaf.max.x, Fix128::from_int(i64::from(i32::MAX)));

    let internal = BvhNode::internal(&extreme, 0, 0);
    let got_internal = internal.get_aabb();
    assert_eq!(got_internal.min.x, Fix128::from_int(i64::from(i32::MIN)));
    assert_eq!(got_internal.max.x, Fix128::from_int(i64::from(i32::MAX)));
}

// ============================================================================
// Single-leaf tree and empty tree -- boundary cases
// ============================================================================

/// A one-primitive tree's root is itself the sole leaf (the `count <= 4`
/// base case fires immediately), so `get_aabb` on the root must
/// reproduce that single primitive's box with no union at all.
#[test]
fn get_aabb_on_single_primitive_tree_root_matches_the_one_primitive_exactly() {
    let only_box = AABB::new(Vec3Fix::from_int(7, -3, 11), Vec3Fix::from_int(9, 1, 14));
    let bvh = LinearBvh::build(vec![BvhPrimitive {
        aabb: only_box,
        index: 0,
        morton: 0,
    }]);
    assert_eq!(bvh.nodes.len(), 1, "single primitive must build one node");
    assert!(bvh.nodes[0].is_leaf());
    let got = bvh.nodes[0].get_aabb();
    assert_eq!(got.min, only_box.min);
    assert_eq!(got.max, only_box.max);
}

/// An empty tree has zero nodes, so there is no node to call `get_aabb`
/// on at all -- the boundary is the absence of any call site, not a
/// special-cased return value. `nodes` and `primitives` must both be
/// empty and `bounds` must be the degenerate zero box `LinearBvh::build`
/// documents for this case.
#[test]
fn empty_tree_has_no_node_to_call_get_aabb_on() {
    let bvh = LinearBvh::build(Vec::new());
    assert!(bvh.nodes.is_empty());
    assert!(bvh.primitives.is_empty());
    assert_eq!(bvh.bounds.min, Vec3Fix::ZERO);
    assert_eq!(bvh.bounds.max, Vec3Fix::ZERO);
}

// ============================================================================
// Per-leaf tightness on a built multi-leaf tree
// ============================================================================

/// Build a tree from two widely separated clusters of 4 unit boxes each
/// (far enough apart that `find_split`'s highest-differing-Morton-bit
/// rule cannot merge them into one leaf -- each cluster alone is exactly
/// at the `count <= 4` leaf threshold). For every leaf in the resulting
/// tree, `get_aabb` must equal the union of *exactly* the input boxes
/// at the primitive indices that leaf covers (`bvh.primitives[start..
/// start+count]`), computed here by hand from the original `boxes` array
/// -- not by calling `AABB::union` on the node's own stored bounds, and
/// not by calling `get_aabb` a second time.
#[test]
fn get_aabb_on_each_leaf_equals_the_hand_union_of_its_own_primitives() {
    let mut boxes = Vec::new();
    for c in 0..2i64 {
        let base = c * 500;
        for k in 0..4i64 {
            boxes.push(unit_box(base + k, base, base));
        }
    }
    let prims: Vec<BvhPrimitive> = boxes
        .iter()
        .enumerate()
        .map(|(i, &aabb)| BvhPrimitive {
            aabb,
            index: i as u32,
            morton: 0,
        })
        .collect();
    let bvh = LinearBvh::build(prims);

    let mut leaves_checked = 0usize;
    for node in &bvh.nodes {
        if !node.is_leaf() {
            continue;
        }
        let start = node.first_child_or_prim as usize;
        let count = node.prim_count() as usize;
        let member_indices = &bvh.primitives[start..start + count];

        // Hand union: literal per-axis min/max over exactly this leaf's
        // member primitives, read directly from `boxes` by their
        // original index -- no BVH method involved.
        let mut want = boxes[member_indices[0] as usize];
        for &idx in &member_indices[1..] {
            let b = boxes[idx as usize];
            want.min = Vec3Fix::new(
                want.min.x.min(b.min.x),
                want.min.y.min(b.min.y),
                want.min.z.min(b.min.z),
            );
            want.max = Vec3Fix::new(
                want.max.x.max(b.max.x),
                want.max.y.max(b.max.y),
                want.max.z.max(b.max.z),
            );
        }

        let got = node.get_aabb();
        assert_eq!(got.min, want.min, "leaf members {member_indices:?} min");
        assert_eq!(got.max, want.max, "leaf members {member_indices:?} max");
        leaves_checked += 1;
    }
    assert_eq!(
        leaves_checked, 2,
        "two disjoint 4-box clusters must build exactly two leaves"
    );
}

/// `BvhNode` round-trips through its public constructors unchanged
/// beyond the AABB itself: `is_leaf`, `prim_count`, `escape_idx`, and
/// `first_child_or_prim` must all still hold their constructor inputs
/// after a `get_aabb` call (the method must be side-effect-free / take
/// `&self`).
#[test]
fn get_aabb_does_not_disturb_the_rest_of_the_node() {
    let b = unit_box(4, 4, 4);
    let leaf = BvhNode::leaf(&b, 12, 3, 99);
    let _ = leaf.get_aabb();
    assert!(leaf.is_leaf());
    assert_eq!(leaf.prim_count(), 3);
    assert_eq!(leaf.escape_idx(), 99);
    assert_eq!(leaf.first_child_or_prim, 12);
}
