//! Lib tests of `src/bvh.rs` for two behaviours the other lib tests leave
//! open: the traversal guards against an out-of-range index, and the count of
//! tested pairs `BroadphaseHybrid::query_pairs` returns.
//!
//! Included from `src/bvh.rs` as a `#[cfg(test)]` module, so they run with
//! `cargo test --lib`, the test set the mutation run uses.

use super::*;
#[cfg(not(feature = "std"))]
use alloc::vec;

/// A cube with integer centre and half size.
fn cube(x: i64, y: i64, z: i64, half: i64) -> AABB {
    let h = Fix128::from_int(half);
    AABB::from_center_half(Vec3Fix::from_int(x, y, z), Vec3Fix::new(h, h, h))
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }
}

/// `n` cubes with centres in [0, 60)³ and half sizes 1 to 3.
fn scene(n: usize, seed: u64) -> Vec<AABB> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let c = |r: &mut Lcg| r.next(60) as i64;
            let (x, y, z) = (c(&mut r), c(&mut r), c(&mut r));
            cube(x, y, z, 1 + r.next(3) as i64)
        })
        .collect()
}

fn tree(boxes: &[AABB]) -> LinearBvh {
    LinearBvh::build(
        boxes
            .iter()
            .enumerate()
            .map(|(i, &aabb)| BvhPrimitive {
                aabb,
                index: i as u32,
                morton: 0,
            })
            .collect(),
    )
}

/// An escape pointer equal to the node count ends the traversal like
/// `ESCAPE_NONE`, and a leaf whose primitive range runs past the primitive
/// list returns only the primitives that exist. Neither indexes out of range.
#[test]
fn traversal_stops_at_an_out_of_range_index() {
    let boxes = scene(20, 21);
    let mut bvh = tree(&boxes);
    let everything = cube(30, 30, 30, 100);
    let expected = bvh.query(&everything);

    // the last node in pre-order escapes to ESCAPE_NONE; point it one past
    let last = bvh.nodes.len() - 1;
    let count = bvh.nodes[last].prim_count();
    bvh.nodes[last].prim_count_escape = (count << 24) | bvh.nodes.len() as u32;
    assert_eq!(bvh.query(&everything), expected);
    let mut via_callback = Vec::new();
    bvh.query_callback(&everything, |p| via_callback.push(p));
    assert_eq!(via_callback, expected);

    // a leaf whose range ends past the primitive list
    let n = bvh.primitives.len() as u32;
    let (leaf, start) = bvh
        .nodes
        .iter()
        .enumerate()
        .find(|(_, node)| node.is_leaf())
        .map(|(i, node)| (i, node.first_child_or_prim))
        .expect("a leaf");
    bvh.nodes[leaf].first_child_or_prim = n - 1;
    let got = bvh.query(&everything);
    assert!(got.iter().all(|&p| p < n));
    let mut via_callback = Vec::new();
    bvh.query_callback(&everything, |p| via_callback.push(p));
    assert_eq!(via_callback, got);
    assert!(start < n);
}

/// `k` equal small boxes share one cell: every pair is tested once and
/// reported, so the returned work is exactly k (k - 1) / 2.
/// oracle: the count of unordered pairs.
#[test]
fn query_pairs_tests_each_pair_of_one_cell_once() {
    for k in [2u32, 3, 5, 8] {
        let mut h = BroadphaseHybrid::new();
        for id in 0..k {
            h.insert_dynamic(id, cube(10, 10, 10, 1), false);
        }
        h.build_dynamic();
        let mut out = Vec::new();
        let tested = h.query_pairs(&mut out);
        assert_eq!(tested, u64::from(k * (k - 1) / 2), "k = {k}");
        assert_eq!(out.len() as u32, k * (k - 1) / 2, "k = {k}");
    }
}

/// `LinearBvh`'s fields are public, so a caller can hand `find_pairs` a tree
/// it did not build. With no primitives there is nothing to pair, and it
/// returns at once, even when a node's escape pointer leads back to itself
/// (a traversal of that tree would not end).
#[test]
fn find_pairs_without_primitives_returns_at_once() {
    let looping = LinearBvh {
        nodes: vec![BvhNode::leaf(&cube(0, 0, 0, 1), 0, 1, 0)],
        primitives: Vec::new(),
        bounds: cube(0, 0, 0, 1),
    };
    assert!(looping.find_pairs().is_empty());
}
