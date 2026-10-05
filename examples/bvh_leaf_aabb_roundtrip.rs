//! Production entry point for `alice_physics::bvh::BvhNode::get_aabb`.
//!
//! # Wiring status (read before extending this file)
//!
//! `scripts/wiring-baseline.txt` lists five `src/bvh.rs` items as
//! `unwired`: `build_dynamic`, `clear_dynamic`, `get_aabb`,
//! `insert_dynamic`, `query_pairs`. Only **`get_aabb`** (a `const fn` on
//! the public, crate-root-re-exported `BvhNode`) is wired by this file.
//!
//! The other four are all methods on `BroadphaseHybrid`
//! (`src/bvh.rs:760`), which is declared `pub(crate)`:
//!
//! ```text
//! pub(crate) struct BroadphaseHybrid { .. }
//! impl BroadphaseHybrid { pub fn new(..); pub fn insert_dynamic(..); .. }
//! ```
//!
//! `pub(crate)` caps visibility at the defining crate's boundary.
//! `examples/` (like `tests/` and `src/bin/`) compile as a *separate*
//! crate that merely depends on `alice-physics`'s public API, so they
//! sit outside that boundary. Confirmed empirically: a one-line probe
//! (`use alice_physics::bvh::BroadphaseHybrid;`) fails to compile with
//!
//! ```text
//! error[E0603]: struct `BroadphaseHybrid` is private
//!   --> src/bvh.rs:760:1
//!    | pub(crate) struct BroadphaseHybrid {
//! ```
//!
//! so none of `insert_dynamic` / `clear_dynamic` / `build_dynamic` /
//! `query_pairs` can be called from an example without first widening
//! `BroadphaseHybrid`'s (and its `impl` block's) visibility to `pub` and
//! re-exporting it from `src/lib.rs` -- a semver-relevant API-surface
//! decision (this repo tracks exactly that kind of change in
//! `docs/PUBLIC_API_SNAPSHOT.txt`), not a purely-additive wiring change.
//! The repo has an existing precedent for declining to do this silently
//! (`CHANGELOG.md`, `eulerian_grid` cross-process primitives: "新規 pub
//! facade は semver 判断 ... 方針: 新規 pub は追加しない、
//! ALLOW-UNWIRED debt marker で現状維持"). This file follows the same
//! precedent and leaves those four as unresolved wiring debt pending
//! that decision, rather than promoting the type's visibility itself.
//!
//! `BroadphaseHybrid`'s own doc comment independently confirms the
//! `pub(crate)` is intentional, not an oversight: "Skeleton API
//! committed ... to freeze the **crate-internal** surface".
//!
//! `tests/analytic_bvh_leaf_aabb_wiring.rs` holds the closed-form
//! oracles for `get_aabb` (leaf / internal reconstruction, fractional
//! floor/ceil rounding, single-leaf and empty-tree boundary cases) that
//! this file's diagnostic prints do not repeat.
//!
//! ```bash
//! cargo run --example bvh_leaf_aabb_roundtrip --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::bvh::{BvhNode, BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

fn unit_box(x: i64, y: i64, z: i64, size: i64) -> AABB {
    AABB::new(
        Vec3Fix::from_int(x, y, z),
        Vec3Fix::from_int(x + size, y + size, z + size),
    )
}

/// A box with an explicit per-axis extent (unlike [`unit_box`], whose
/// `size` applies uniformly to all three axes).
fn flat_box(min: (i64, i64, i64), max: (i64, i64, i64)) -> AABB {
    AABB::new(
        Vec3Fix::from_int(min.0, min.1, min.2),
        Vec3Fix::from_int(max.0, max.1, max.2),
    )
}

fn main() {
    // ------------------------------------------------------------------
    // 1. Build a real `LinearBvh` from five hand-chosen scene objects
    //    (a floor platform, a crate resting on it, two stacked pillar
    //    segments far away, and an isolated piece of debris) and check
    //    that *every* node's `get_aabb()` -- leaf and internal alike --
    //    reconstructs exactly the bounds the build computed, by
    //    comparing against `Vec3Fix::from_int` applied directly to the
    //    node's raw `aabb_min` / `aabb_max` i32 fields (not by calling
    //    `get_aabb()` a second time).
    // ------------------------------------------------------------------
    let boxes = [
        flat_box((0, 0, 0), (4, 1, 4)), // 0: floor platform, 4x1x4 footprint in x/z
        unit_box(1, 1, 1, 1),           // 1: crate on the floor (touches at y=1)
        unit_box(20, 0, 0, 1),          // 2: pillar segment, isolated far away
        unit_box(20, 1, 0, 1),          // 3: pillar segment stacked on 2 (touches at y=1)
        unit_box(-10, -10, -10, 1),     // 4: isolated debris, far negative
    ];
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
    assert!(
        !bvh.nodes.is_empty(),
        "[bvh] MISMATCH LinearBvh::build(5 scene objects): expected a non-empty tree"
    );

    for (i, node) in bvh.nodes.iter().enumerate() {
        let want = AABB::new(
            Vec3Fix::from_int(
                i64::from(node.aabb_min[0]),
                i64::from(node.aabb_min[1]),
                i64::from(node.aabb_min[2]),
            ),
            Vec3Fix::from_int(
                i64::from(node.aabb_max[0]),
                i64::from(node.aabb_max[1]),
                i64::from(node.aabb_max[2]),
            ),
        );
        let got = node.get_aabb();
        assert_eq!(
            got, want,
            "[bvh] MISMATCH node {i} get_aabb(): got {got:?}, want {want:?} \
             (raw fields min={:?} max={:?})",
            node.aabb_min, node.aabb_max
        );
    }
    println!(
        "[bvh] ok get_aabb() round-trips exactly on all {} built nodes (integer AABBs)",
        bvh.nodes.len()
    );

    // ------------------------------------------------------------------
    // 2. The root node's get_aabb() must equal the hand-computed union
    //    of all five input boxes -- literal per-axis min/max read off
    //    the box list above, not a call to `AABB::union` or any other
    //    BVH method:
    //      min.x = min(0,1,20,20,-10)  = -10   max.x = max(4,2,21,21,-9) = 21
    //      min.y = min(0,1,0,1,-10)    = -10   max.y = max(1,2,1,2,-9)   =  2
    //      min.z = min(0,1,0,0,-10)    = -10   max.z = max(4,2,1,1,-9)   =  4
    // ------------------------------------------------------------------
    let hand_union_min = Vec3Fix::from_int(-10, -10, -10);
    let hand_union_max = Vec3Fix::from_int(21, 2, 4);
    let root_aabb = bvh.nodes[0].get_aabb();
    assert_eq!(
        root_aabb.min, hand_union_min,
        "[bvh] MISMATCH root get_aabb() min: got {:?}, want {hand_union_min:?}",
        root_aabb.min
    );
    assert_eq!(
        root_aabb.max, hand_union_max,
        "[bvh] MISMATCH root get_aabb() max: got {:?}, want {hand_union_max:?}",
        root_aabb.max
    );
    println!(
        "[bvh] ok root get_aabb() == hand-derived union: min=({},{},{}) max=({},{},{})",
        root_aabb.min.x.to_f64(),
        root_aabb.min.y.to_f64(),
        root_aabb.min.z.to_f64(),
        root_aabb.max.x.to_f64(),
        root_aabb.max.y.to_f64(),
        root_aabb.max.z.to_f64()
    );

    // ------------------------------------------------------------------
    // 3. Per-leaf tightness: for every leaf the build produced, read off
    //    which of the five original objects it holds
    //    (`bvh.primitives[start..start+count]`) and union exactly those
    //    objects' original boxes from the `boxes` array in step 1, via
    //    `AABB::union`'s plain per-axis min/max -- not by reusing the
    //    node's own stored bounds, and not by calling `get_aabb` a
    //    second time. That hand-combined union must equal the leaf's
    //    `get_aabb()`.
    // ------------------------------------------------------------------
    let mut leaves_checked = 0usize;
    for node in &bvh.nodes {
        if !node.is_leaf() {
            continue;
        }
        let start = node.first_child_or_prim as usize;
        let count = node.prim_count() as usize;
        let members = &bvh.primitives[start..start + count];

        let mut want = boxes[members[0] as usize];
        for &idx in &members[1..] {
            want = want.union(&boxes[idx as usize]);
        }
        let got = node.get_aabb();
        assert_eq!(
            got, want,
            "[bvh] MISMATCH leaf (members {members:?}) get_aabb(): got {got:?}, want {want:?}"
        );
        println!("[bvh] ok leaf members {members:?} get_aabb() == hand union of those objects");
        leaves_checked += 1;
    }
    assert_eq!(
        leaves_checked, 2,
        "[bvh] MISMATCH: 5 primitives at <=4 per leaf must build exactly 2 leaves, got {leaves_checked}"
    );

    // ------------------------------------------------------------------
    // 4. get_aabb() on directly-constructed leaf / internal nodes with
    //    fractional bounds -- independent of the build/query path above.
    //    min coordinates floor, max coordinates ceil (conservative
    //    outward rounding so the reconstructed box always encloses the
    //    original). These values are new relative to the module's own
    //    `#[cfg(test)]` oracle in `src/bvh.rs`: an exact-half negative
    //    minimum together with an exact-half positive maximum, and an
    //    axis that is already an exact integer on both ends (must not
    //    move at all).
    // ------------------------------------------------------------------
    let frac = AABB::new(
        Vec3Fix::new(
            Fix128::from_ratio(-1, 2),  // -0.5 -> floor -> -1
            Fix128::from_ratio(-17, 2), // -8.5 -> floor -> -9
            Fix128::from_int(5),        //  5   -> floor -> 5 (exact, unmoved)
        ),
        Vec3Fix::new(
            Fix128::from_ratio(1, 2),  //  0.5 -> ceil -> 1
            Fix128::from_ratio(17, 2), //  8.5 -> ceil -> 9
            Fix128::from_int(5),       //  5   -> ceil -> 5 (exact, unmoved)
        ),
    );
    let leaf = BvhNode::leaf(&frac, 0, 1, 0x00FF_FFFF);
    let leaf_got = leaf.get_aabb();
    let want_min = Vec3Fix::from_int(-1, -9, 5);
    let want_max = Vec3Fix::from_int(1, 9, 5);
    assert_eq!(
        leaf_got.min, want_min,
        "[bvh] MISMATCH fractional leaf get_aabb() min: got {:?}, want {want_min:?}",
        leaf_got.min
    );
    assert_eq!(
        leaf_got.max, want_max,
        "[bvh] MISMATCH fractional leaf get_aabb() max: got {:?}, want {want_max:?}",
        leaf_got.max
    );
    assert_eq!(
        leaf_got.union(&frac),
        leaf_got,
        "[bvh] MISMATCH fractional leaf get_aabb() must conservatively enclose the original box"
    );
    println!(
        "[bvh] ok get_aabb() on a directly-constructed fractional leaf: \
         min=({},{},{}) max=({},{},{}) encloses the original",
        leaf_got.min.x.to_f64(),
        leaf_got.min.y.to_f64(),
        leaf_got.min.z.to_f64(),
        leaf_got.max.x.to_f64(),
        leaf_got.max.y.to_f64(),
        leaf_got.max.z.to_f64()
    );

    println!("[bvh] all checks passed");
}
