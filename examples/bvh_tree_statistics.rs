//! BVH Tree Statistics Example
//!
//! Production entry point for the diagnostics of the two bounding volume
//! hierarchies: `LinearBvh::stats` / `BvhStats` (`src/bvh.rs`) and
//! `DynamicAabbTree::node_count` / `DynamicAabbTree::query`
//! (`src/dynamic_bvh.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - `LinearBvh` splits a range at the highest differing Morton bit and stops
//!   at four primitives per leaf. `2^k` unit boxes in a row along `x` have
//!   centres `(i + 1/2) / 2^k` of the bounds, so every split halves the range:
//!   `2^k / 4` leaves of four, one fewer internal node, and the primitive
//!   count is the input count; a row with a gap splits at the gap, so the
//!   fullest leaf is the larger side
//! - a binary tree of `n` leaves always has `2n − 1` nodes, so
//!   `DynamicAabbTree::node_count` is `2n − 1` after `n` inserts and drops
//!   by two per removal
//! - with a zero fattening margin `DynamicAabbTree::query` returns exactly
//!   the proxies whose box overlaps the query box, which a brute-force
//!   interval test reproduces
//!
//! Run with: `cargo run --example bvh_tree_statistics`

use alice_physics::bvh::{BvhPrimitive, BvhStats, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::dynamic_bvh::DynamicAabbTree;
use alice_physics::math::{Fix128, Vec3Fix};

/// The unit box `[x, x + 1] × [0, 1] × [0, 1]`.
fn unit_box(x: i64) -> AABB {
    AABB::new(Vec3Fix::from_int(x, 0, 0), Vec3Fix::from_int(x + 1, 1, 1))
}

fn row_of_boxes(n: u32) -> Vec<BvhPrimitive> {
    (0..n)
        .map(|i| BvhPrimitive {
            aabb: unit_box(i64::from(i)),
            index: i,
            morton: 0,
        })
        .collect()
}

fn linear_bvh_stats() {
    for n in [1u32, 4, 8, 16, 64] {
        let bvh = LinearBvh::build(row_of_boxes(n));
        let stats: BvhStats = bvh.stats();
        let n = n as usize;
        // Up to four primitives fit one leaf; above that every split halves.
        let leaves = n.div_ceil(4);
        assert_eq!(stats.primitive_count, n, "n = {n}: primitive count");
        assert_eq!(stats.leaf_count, leaves, "n = {n}: leaf count");
        assert_eq!(stats.internal_count, leaves - 1, "n = {n}: internal count");
        assert_eq!(stats.node_count, 2 * leaves - 1, "n = {n}: node count");
        assert_eq!(stats.max_leaf_prims, n.min(4), "n = {n}: fullest leaf");
        println!(
            "LinearBvh n = {n:>2}: nodes {}, leaves {}, internal {}, max per leaf {}",
            stats.node_count, stats.leaf_count, stats.internal_count, stats.max_leaf_prims
        );
    }

    // Four boxes at x = 0..4 and two at x = 10..12: the bounds are [0, 12],
    // the highest Morton bit splits at x = 6, so the leaves hold 4 and 2.
    let uneven: Vec<BvhPrimitive> = [0i64, 1, 2, 3, 10, 11]
        .into_iter()
        .zip(0u32..)
        .map(|(x, index)| BvhPrimitive {
            aabb: unit_box(x),
            index,
            morton: 0,
        })
        .collect();
    let stats: BvhStats = LinearBvh::build(uneven).stats();
    assert_eq!(
        (stats.node_count, stats.leaf_count, stats.internal_count),
        (3, 2, 1),
        "a 4 + 2 split is one internal node over two leaves"
    );
    assert_eq!(stats.max_leaf_prims, 4, "the fuller leaf holds 4");
    println!(
        "LinearBvh 4 + 2: nodes {}, max per leaf {}",
        stats.node_count, stats.max_leaf_prims
    );

    let empty: BvhStats = LinearBvh::build(Vec::new()).stats();
    assert_eq!(
        (empty.node_count, empty.leaf_count, empty.primitive_count),
        (0, 0, 0),
        "an empty BVH has no nodes"
    );
}

fn dynamic_tree() {
    let mut tree = DynamicAabbTree::new();
    tree.margin = Fix128::ZERO;
    assert_eq!(tree.node_count(), 0, "an empty tree has no nodes");

    // Boxes [2i, 2i + 1] along x: neighbours are one unit apart.
    let n = 10u32;
    let mut proxies = Vec::new();
    for i in 0..n {
        let b = unit_box(2 * i64::from(i));
        proxies.push(tree.insert(b, i));
        let leaves = (i + 1) as usize;
        assert_eq!(
            tree.node_count(),
            2 * leaves - 1,
            "2n - 1 after {leaves} inserts"
        );
    }
    for (removed, &proxy) in proxies.iter().take(3).enumerate() {
        tree.remove(proxy);
        let leaves = n as usize - removed - 1;
        assert_eq!(tree.node_count(), 2 * leaves - 1, "2n - 1 after a removal");
    }
    // Proxies 0, 1 and 2 are gone; 3..9 remain at x in [6, 19].
    let remaining: Vec<u32> = (3..n).collect();

    let queries = [
        (Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(5, 1, 1)),
        (Vec3Fix::from_int(7, 0, 0), Vec3Fix::from_int(8, 1, 1)),
        (Vec3Fix::from_int(9, 0, 0), Vec3Fix::from_int(14, 1, 1)),
        (Vec3Fix::from_int(0, 2, 0), Vec3Fix::from_int(30, 3, 1)),
        (Vec3Fix::from_int(-5, -5, -5), Vec3Fix::from_int(50, 5, 5)),
    ];
    for (lo, hi) in queries {
        let mut got = tree.query(&AABB::new(lo, hi));
        got.sort_unstable();
        // Closed intervals overlap when lo <= max and min <= hi on every axis.
        let want: Vec<u32> = remaining
            .iter()
            .copied()
            .filter(|&i| {
                let (bx0, bx1) = (2 * i64::from(i), 2 * i64::from(i) + 1);
                let x = Fix128::from_int(bx0) <= hi.x && lo.x <= Fix128::from_int(bx1);
                let y = Fix128::ZERO <= hi.y && lo.y <= Fix128::ONE;
                let z = Fix128::ZERO <= hi.z && lo.z <= Fix128::ONE;
                x && y && z
            })
            .collect();
        assert_eq!(
            got,
            want,
            "query x in [{}, {}], y in [{}, {}]",
            lo.x.to_f64(),
            hi.x.to_f64(),
            lo.y.to_f64(),
            hi.y.to_f64()
        );
        println!(
            "DynamicAabbTree query x [{:>3}, {:>3}] y [{:>2}, {:>2}] -> {got:?}",
            lo.x.to_f64(),
            hi.x.to_f64(),
            lo.y.to_f64(),
            hi.y.to_f64()
        );
    }
    println!(
        "DynamicAabbTree: {} nodes for {} proxies",
        tree.node_count(),
        remaining.len()
    );
}

fn main() {
    linear_bvh_stats();
    dynamic_tree();
    println!("all closed forms hold");
}
