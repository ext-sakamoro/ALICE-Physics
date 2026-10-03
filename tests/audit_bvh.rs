//! Audit oracles for `bvh` (LBVH with Morton ordering and stackless escape
//! traversal). The closed form for a broad phase is brute force: every pair /
//! query result must contain what an O(n^2) overlap scan finds (no false
//! negatives), the tree shape must satisfy the binary-tree and escape-pointer
//! invariants derived here from the node array alone, and Morton codes must
//! equal the bit interleave of the quantised coordinates computed
//! independently.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// the Morton cell index closed form is evaluated in f64
#![allow(clippy::disallowed_methods)]

use alice_physics::bvh::{point_to_morton, BvhNode, BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// value in `[-range, range)` with 1/16 resolution
    fn coord(&mut self, range: i64) -> Fix128 {
        let n = (self.next() % (2 * range as u64 * 16)) as i64 - range * 16;
        Fix128::from_ratio(n, 16)
    }

    fn extent(&mut self, max16: u64) -> Fix128 {
        Fix128::from_ratio((self.next() % max16) as i64, 16)
    }
}

fn random_boxes(seed: u64, n: usize, range: i64, max_extent16: u64) -> Vec<AABB> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let min = Vec3Fix::new(r.coord(range), r.coord(range), r.coord(range));
            let max = Vec3Fix::new(
                min.x + r.extent(max_extent16),
                min.y + r.extent(max_extent16),
                min.z + r.extent(max_extent16),
            );
            AABB::new(min, max)
        })
        .collect()
}

fn build(boxes: &[AABB]) -> LinearBvh {
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

fn overlaps(a: &AABB, b: &AABB) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

fn floor_i(v: Fix128) -> i64 {
    v.hi
}

fn ceil_i(v: Fix128) -> i64 {
    if v.lo > 0 {
        v.hi + 1
    } else {
        v.hi
    }
}

fn q_min(a: &AABB) -> [i64; 3] {
    [floor_i(a.min.x), floor_i(a.min.y), floor_i(a.min.z)]
}

fn q_max(a: &AABB) -> [i64; 3] {
    [ceil_i(a.max.x), ceil_i(a.max.y), ceil_i(a.max.z)]
}

fn node_min(n: &BvhNode) -> [i64; 3] {
    [
        n.aabb_min[0] as i64,
        n.aabb_min[1] as i64,
        n.aabb_min[2] as i64,
    ]
}

fn node_max(n: &BvhNode) -> [i64; 3] {
    [
        n.aabb_max[0] as i64,
        n.aabb_max[1] as i64,
        n.aabb_max[2] as i64,
    ]
}

/// Subtree size by walking `first_child_or_prim` (left) and the implicit right
/// child at `left + size(left)`; independent of the escape pointers.
fn subtree_size(nodes: &[BvhNode], i: usize) -> usize {
    if nodes[i].is_leaf() {
        1
    } else {
        let left = nodes[i].first_child_or_prim as usize;
        let ls = subtree_size(nodes, left);
        let rs = subtree_size(nodes, left + ls);
        1 + ls + rs
    }
}

/// Expected escape of every node: the index just past the nearest enclosing
/// left subtree, `None` when the node lies on the right spine of the root.
fn expected_escapes(nodes: &[BvhNode]) -> Vec<Option<usize>> {
    let mut out = vec![None; nodes.len()];
    fn walk(nodes: &[BvhNode], i: usize, esc: Option<usize>, out: &mut Vec<Option<usize>>) {
        out[i] = esc;
        if !nodes[i].is_leaf() {
            let left = nodes[i].first_child_or_prim as usize;
            let ls = subtree_size(nodes, left);
            let right = left + ls;
            walk(nodes, left, Some(right), out);
            walk(nodes, right, esc, out);
        }
    }
    walk(nodes, 0, None, &mut out);
    out
}

// ---------------------------------------------------------------------------
// structure
// ---------------------------------------------------------------------------

#[test]
fn tree_structure_satisfies_binary_tree_leaf_partition_and_escape_invariants() {
    for (seed, n) in [
        (1u64, 1usize),
        (2, 2),
        (3, 4),
        (4, 5),
        (5, 9),
        (6, 64),
        (7, 333),
        (8, 1000),
    ] {
        let boxes = random_boxes(seed, n, 40, 48);
        let bvh = build(&boxes);
        let nodes = &bvh.nodes;
        // primitives is a permutation of 0..n
        let mut seen = bvh.primitives.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n as u32).collect::<Vec<_>>(), "n={n}");
        // binary tree: nodes = 2 leaves - 1, leaf sizes 1..=4, ranges tile 0..n in DFS order
        let leaves: Vec<&BvhNode> = nodes.iter().filter(|x| x.is_leaf()).collect();
        assert_eq!(nodes.len(), 2 * leaves.len() - 1, "n={n}");
        let mut next = 0usize;
        for l in &leaves {
            assert!((1..=4).contains(&(l.prim_count() as usize)), "n={n}");
            assert_eq!(l.first_child_or_prim as usize, next, "n={n}");
            next += l.prim_count() as usize;
        }
        assert_eq!(next, n);
        assert_eq!(subtree_size(nodes, 0), nodes.len());
        // escape pointers
        let want = expected_escapes(nodes);
        for (i, node) in nodes.iter().enumerate() {
            match want[i] {
                Some(w) => assert_eq!(node.escape_idx() as usize, w, "n={n} node {i}"),
                None => assert!(node.escape_idx() as usize >= nodes.len(), "n={n} node {i}"),
            }
        }
        // bounds and conservative quantised boxes
        let mut world = boxes[0];
        for b in &boxes[1..] {
            world = world.union(b);
        }
        assert_eq!(bvh.bounds, world, "n={n}");
        assert_eq!(node_min(&nodes[0]), q_min(&world), "n={n} root min");
        assert_eq!(node_max(&nodes[0]), q_max(&world), "n={n} root max");
        for node in nodes {
            if node.is_leaf() {
                let s = node.first_child_or_prim as usize;
                for k in 0..node.prim_count() as usize {
                    let b = &boxes[bvh.primitives[s + k] as usize];
                    for a in 0..3 {
                        assert!(node_min(node)[a] <= q_min(b)[a], "n={n} leaf min");
                        assert!(node_max(node)[a] >= q_max(b)[a], "n={n} leaf max");
                    }
                }
            } else {
                let l = &nodes[node.first_child_or_prim as usize];
                let r = &nodes[node.first_child_or_prim as usize
                    + subtree_size(nodes, node.first_child_or_prim as usize)];
                for a in 0..3 {
                    assert_eq!(
                        node_min(node)[a],
                        node_min(l)[a].min(node_min(r)[a]),
                        "n={n} union min"
                    );
                    assert_eq!(
                        node_max(node)[a],
                        node_max(l)[a].max(node_max(r)[a]),
                        "n={n} union max"
                    );
                }
            }
        }
    }
}

#[test]
fn many_coincident_primitives_still_split_into_leaves_of_at_most_four() {
    let b = AABB::new(Vec3Fix::from_int(3, 3, 3), Vec3Fix::from_int(4, 4, 4));
    let boxes = vec![b; 300];
    let bvh = build(&boxes);
    let s = bvh.stats();
    assert_eq!(s.primitive_count, 300);
    assert!(s.max_leaf_prims <= 4);
    assert_eq!(bvh.query(&b).len(), 300);
    assert_eq!(bvh.find_pairs().len(), 300 * 299 / 2);
}

#[test]
fn empty_and_single_primitive_trees() {
    let empty = LinearBvh::build(Vec::new());
    assert!(empty.nodes.is_empty());
    let any = AABB::new(Vec3Fix::from_int(-5, -5, -5), Vec3Fix::from_int(5, 5, 5));
    assert!(empty.query(&any).is_empty());
    assert!(empty.find_pairs().is_empty());
    let one = build(&[any]);
    assert_eq!(one.nodes.len(), 1);
    assert!(one.nodes[0].is_leaf());
    assert_eq!(one.query(&any), vec![0]);
    assert!(one.find_pairs().is_empty());
    assert_eq!(one.stats().leaf_count, 1);
}

#[test]
fn stats_agree_with_a_direct_count_of_the_node_array() {
    let bvh = build(&random_boxes(11, 200, 30, 32));
    let s = bvh.stats();
    let leaves = bvh.nodes.iter().filter(|n| n.is_leaf()).count();
    assert_eq!(s.node_count, bvh.nodes.len());
    assert_eq!(s.leaf_count, leaves);
    assert_eq!(s.internal_count, bvh.nodes.len() - leaves);
    assert_eq!(s.primitive_count, 200);
    assert_eq!(
        s.max_leaf_prims,
        bvh.nodes
            .iter()
            .map(|n| n.prim_count() as usize)
            .max()
            .unwrap()
    );
}

// ---------------------------------------------------------------------------
// completeness against brute force
// ---------------------------------------------------------------------------

#[test]
fn query_never_misses_a_true_overlap_and_never_repeats_a_primitive() {
    for (seed, n, range) in [(21u64, 50usize, 20i64), (22, 400, 60), (23, 400, 2)] {
        let boxes = random_boxes(seed, n, range, 40);
        let bvh = build(&boxes);
        let queries = random_boxes(seed + 100, 60, range, 60);
        for q in &queries {
            let got = bvh.query(q);
            let mut sorted = got.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(before, sorted.len(), "duplicate primitive in query result");
            for (i, b) in boxes.iter().enumerate() {
                if overlaps(q, b) {
                    assert!(
                        got.contains(&(i as u32)),
                        "seed={seed}: primitive {i} overlaps but was missed"
                    );
                }
            }
            let mut via_cb = Vec::new();
            bvh.query_callback(q, |p| via_cb.push(p));
            assert_eq!(via_cb, got, "query_callback order differs from query");
        }
    }
}

#[test]
fn find_pairs_contains_every_true_overlap_and_is_sorted_unique_with_i_less_than_j() {
    for (seed, n, range, ext) in [
        (31u64, 120usize, 25i64, 40u64),
        (32, 300, 80, 24),
        (33, 200, 3, 16),
    ] {
        let boxes = random_boxes(seed, n, range, ext);
        let bvh = build(&boxes);
        let pairs = bvh.find_pairs();
        for w in pairs.windows(2) {
            assert!(
                w[0] < w[1],
                "pairs not strictly sorted: {:?} {:?}",
                w[0],
                w[1]
            );
        }
        for &(i, j) in &pairs {
            assert!(i < j && (j as usize) < n);
        }
        let set: std::collections::HashSet<(u32, u32)> = pairs.iter().copied().collect();
        for i in 0..n {
            for j in i + 1..n {
                if overlaps(&boxes[i], &boxes[j]) {
                    assert!(
                        set.contains(&(i as u32, j as u32)),
                        "seed={seed}: overlapping pair ({i},{j}) missing"
                    );
                }
            }
        }
    }
}

/// The exact set `find_pairs` is documented to return: every pair of primitives
/// whose *leaf* boxes (quantised to whole units, a leaf holds at most four
/// primitives) overlap, including two primitives of the same leaf.
fn leaf_box_pairs(bvh: &LinearBvh) -> Vec<(u32, u32)> {
    let leaves: Vec<&BvhNode> = bvh.nodes.iter().filter(|n| n.is_leaf()).collect();
    let mut out = Vec::new();
    for l in &leaves {
        for m in &leaves {
            let hit =
                (0..3).all(|a| l.aabb_min[a] <= m.aabb_max[a] && l.aabb_max[a] >= m.aabb_min[a]);
            if !hit {
                continue;
            }
            for i in 0..l.prim_count() as usize {
                for j in 0..m.prim_count() as usize {
                    let pi = bvh.primitives[l.first_child_or_prim as usize + i];
                    let pj = bvh.primitives[m.first_child_or_prim as usize + j];
                    if pi < pj {
                        out.push((pi, pj));
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[test]
fn find_pairs_is_exactly_the_set_of_primitive_pairs_whose_leaf_boxes_overlap() {
    for (seed, n, range, ext) in [
        (71u64, 3usize, 20i64, 16u64),
        (72, 64, 40, 24),
        (73, 500, 100, 30),
        (74, 150, 5, 16),
    ] {
        let boxes = random_boxes(seed, n, range, ext);
        let bvh = build(&boxes);
        assert_eq!(bvh.find_pairs(), leaf_box_pairs(&bvh), "seed={seed} n={n}");
    }
}

#[test]
fn separated_clusters_never_pair_across_clusters() {
    // 8 clusters of 20 coincident unit boxes, clusters 100 units apart on a ring
    let mut boxes = Vec::new();
    for c in 0..8i64 {
        let o = Vec3Fix::from_int((c % 2) * 100, ((c / 2) % 2) * 100, (c / 4) * 100);
        for _ in 0..20 {
            boxes.push(AABB::new(o, o + Vec3Fix::from_int(1, 1, 1)));
        }
    }
    let pairs = build(&boxes).find_pairs();
    for &(i, j) in &pairs {
        assert_eq!(i / 20, j / 20, "pair ({i},{j}) spans two clusters");
    }
    assert_eq!(pairs.len(), 8 * (20 * 19 / 2));
}

#[test]
fn coordinates_beyond_the_i32_range_are_still_found() {
    let mut boxes = random_boxes(41, 40, 10, 16);
    let far = AABB::new(
        Vec3Fix::from_int(5_000_000_000, -7_000_000_000, 3_000_000_000),
        Vec3Fix::from_int(5_000_000_001, -6_999_999_999, 3_000_000_001),
    );
    boxes.push(far);
    let bvh = build(&boxes);
    let got = bvh.query(&far);
    assert!(got.contains(&(boxes.len() as u32 - 1)), "{got:?}");
}

#[test]
fn input_order_does_not_change_the_tree_when_morton_codes_are_distinct() {
    // boxes on a lattice with unit spacing 8: centres (and hence codes) distinct
    let mut boxes = Vec::new();
    for i in 0..6i64 {
        for j in 0..5i64 {
            for k in 0..4i64 {
                let o = Vec3Fix::from_int(i * 8, j * 8, k * 8);
                boxes.push(AABB::new(o, o + Vec3Fix::from_int(2, 3, 1)));
            }
        }
    }
    let prims: Vec<BvhPrimitive> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| BvhPrimitive {
            aabb: *b,
            index: i as u32,
            morton: 0,
        })
        .collect();
    let mut shuffled = prims.clone();
    let mut r = Lcg(77);
    for i in (1..shuffled.len()).rev() {
        let j = (r.next() as usize) % (i + 1);
        shuffled.swap(i, j);
    }
    let a = LinearBvh::build(prims);
    let b = LinearBvh::build(shuffled);
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.primitives, b.primitives);
    assert_eq!(a.bounds, b.bounds);
}

// ---------------------------------------------------------------------------
// Morton codes
// ---------------------------------------------------------------------------

/// independent bit interleave: bit i of x -> bit 3i, y -> 3i+1, z -> 3i+2
fn interleave(x: u64, y: u64, z: u64) -> u64 {
    let mut out = 0u64;
    for i in 0..21 {
        out |= ((x >> i) & 1) << (3 * i);
        out |= ((y >> i) & 1) << (3 * i + 1);
        out |= ((z >> i) & 1) << (3 * i + 2);
    }
    out
}

fn cube_bounds() -> AABB {
    // 2^21 per side: one cell per unit, cell index = floor(coordinate)
    AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1 << 21, 1 << 21, 1 << 21))
}

#[test]
fn point_to_morton_equals_the_independent_bit_interleave_of_the_cell_index() {
    let b = cube_bounds();
    let samples: [(i64, i64, i64); 9] = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (12345, 54321, 99999),
        (0x1FFFFF, 0, 0x155555),
        (0xAAAAA, 0x55555, 0x1FFFFF),
        (1, 1, 1),
        (0x1FFFFF, 0x1FFFFF, 0x1FFFFF),
    ];
    for (x, y, z) in samples {
        let got = point_to_morton(Vec3Fix::from_int(x, y, z), &b);
        assert_eq!(
            got,
            interleave(x as u64, y as u64, z as u64),
            "({x},{y},{z})"
        );
        // a fractional offset inside the cell does not change the cell
        let frac = Vec3Fix::new(
            Fix128::from_int(x) + Fix128::from_ratio(3, 4),
            Fix128::from_int(y) + Fix128::from_ratio(1, 2),
            Fix128::from_int(z),
        );
        assert_eq!(
            point_to_morton(frac, &b),
            got,
            "fraction inside cell ({x},{y},{z})"
        );
    }
}

#[test]
fn morton_code_of_the_eight_cell_corners_is_x_plus_2y_plus_4z() {
    let b = cube_bounds();
    for z in 0..2i64 {
        for y in 0..2i64 {
            for x in 0..2i64 {
                let c = point_to_morton(Vec3Fix::from_int(x, y, z), &b);
                assert_eq!(c, (x + 2 * y + 4 * z) as u64);
            }
        }
    }
}

#[test]
fn morton_code_is_monotone_along_each_axis_and_clamps_outside_the_box() {
    let b = cube_bounds();
    let mut prev = 0u64;
    for x in (0..(1i64 << 21)).step_by(4099) {
        let c = point_to_morton(Vec3Fix::from_int(x, 17, 5), &b);
        assert!(c >= prev, "x-sweep not monotone at {x}");
        prev = c;
    }
    let mut prev = 0u64;
    for z in (0..(1i64 << 21)).step_by(4099) {
        let c = point_to_morton(Vec3Fix::from_int(17, 5, z), &b);
        assert!(c >= prev, "z-sweep not monotone at {z}");
        prev = c;
    }
    assert_eq!(point_to_morton(Vec3Fix::from_int(-5, -9, -1), &b), 0);
    assert_eq!(
        point_to_morton(Vec3Fix::from_int(1 << 22, 1 << 22, 1 << 22), &b),
        (1u64 << 63) - 1
    );
    assert_eq!(point_to_morton(b.max, &b), (1u64 << 63) - 1);
}

#[test]
fn morton_normalisation_uses_the_largest_extent_for_every_axis() {
    // bounds 1000 x 10 x 10: a point at y = 5 is half way up the *shortest* axis,
    // but with isotropic scaling its cell is 5/1000 of the way, not 1/2.
    let b = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1000, 10, 10));
    let c = point_to_morton(Vec3Fix::from_int(0, 5, 0), &b);
    let want_cell = (5.0_f64 / 1000.0 * (1u64 << 21) as f64).floor() as u64;
    assert!(want_cell < 0x1F_FFFF / 100);
    // the cell index is within 1 of the closed form (division rounding)
    let y_cell = (0..21).fold(0u64, |acc, i| acc | (((c >> (3 * i + 1)) & 1) << i));
    assert!(
        y_cell.abs_diff(want_cell) <= 1,
        "y cell {y_cell} vs {want_cell}"
    );
}

// ---------------------------------------------------------------------------
// node packing and refit
// ---------------------------------------------------------------------------

#[test]
fn node_packing_round_trips_count_and_escape_and_reconstructs_conservative_aabbs() {
    let aabb = AABB::new(
        Vec3Fix::new(
            Fix128::from_ratio(-5, 2),
            Fix128::from_ratio(1, 4),
            Fix128::from_int(7),
        ),
        Vec3Fix::new(
            Fix128::from_ratio(7, 2),
            Fix128::from_ratio(9, 4),
            Fix128::from_int(7),
        ),
    );
    let leaf = BvhNode::leaf(&aabb, 12, 4, 0x00ABCDEF);
    assert!(leaf.is_leaf());
    assert_eq!(leaf.prim_count(), 4);
    assert_eq!(leaf.escape_idx(), 0x00ABCDEF);
    assert_eq!(leaf.first_child_or_prim, 12);
    let internal = BvhNode::internal(&aabb, 33, 0x00123456);
    assert!(!internal.is_leaf());
    assert_eq!(internal.prim_count(), 0);
    assert_eq!(internal.escape_idx(), 0x00123456);
    // floor / ceil quantisation: [-2.5, 3.5] x [0.25, 2.25] x [7, 7] -> [-3, 4] x [0, 3] x [7, 7]
    assert_eq!(leaf.aabb_min, [-3, 0, 7]);
    assert_eq!(leaf.aabb_max, [4, 3, 7]);
    let back = leaf.get_aabb();
    assert_eq!(back.min, Vec3Fix::from_int(-3, 0, 7));
    assert_eq!(back.max, Vec3Fix::from_int(4, 3, 7));
    // intersects_i32 is inclusive on all six faces
    assert!(leaf.intersects_i32(&[4, 3, 7], &[9, 9, 9]));
    assert!(leaf.intersects_i32(&[-9, -9, -9], &[-3, 0, 7]));
    assert!(!leaf.intersects_i32(&[5, 0, 7], &[9, 9, 9]));
    assert!(!leaf.intersects_i32(&[-9, -9, -9], &[-4, 3, 7]));
    assert!(!leaf.intersects_i32(&[-3, 4, 7], &[4, 9, 7]));
    assert!(!leaf.intersects_i32(&[-3, 0, 8], &[4, 3, 9]));
}

fn moved(boxes: &[AABB], seed: u64) -> Vec<AABB> {
    let mut r = Lcg(seed);
    boxes
        .iter()
        .map(|b| {
            let d = Vec3Fix::new(r.coord(4), r.coord(4), r.coord(4));
            AABB::new(b.min + d, b.max + d)
        })
        .collect()
}

#[test]
fn refit_updates_every_leaf_and_internal_box_to_the_quantised_union_of_the_new_boxes() {
    let boxes = random_boxes(51, 150, 30, 24);
    let mut bvh = build(&boxes);
    let before = bvh.nodes.clone();
    let new_boxes = moved(&boxes, 52);
    bvh.refit_leaves(&new_boxes);
    // structure preserved
    for (a, b) in before.iter().zip(bvh.nodes.iter()) {
        assert_eq!(a.first_child_or_prim, b.first_child_or_prim);
        assert_eq!(a.prim_count_escape, b.prim_count_escape);
    }
    for node in &bvh.nodes {
        if node.is_leaf() {
            let s = node.first_child_or_prim as usize;
            let mut lo = [i64::MAX; 3];
            let mut hi = [i64::MIN; 3];
            for k in 0..node.prim_count() as usize {
                let b = &new_boxes[bvh.primitives[s + k] as usize];
                for a in 0..3 {
                    lo[a] = lo[a].min(q_min(b)[a]);
                    hi[a] = hi[a].max(q_max(b)[a]);
                }
            }
            assert_eq!(node_min(node), lo);
            assert_eq!(node_max(node), hi);
        } else {
            let li = node.first_child_or_prim as usize;
            let ri = li + subtree_size(&bvh.nodes, li);
            for a in 0..3 {
                assert_eq!(
                    node_min(node)[a],
                    node_min(&bvh.nodes[li])[a].min(node_min(&bvh.nodes[ri])[a])
                );
                assert_eq!(
                    node_max(node)[a],
                    node_max(&bvh.nodes[li])[a].max(node_max(&bvh.nodes[ri])[a])
                );
            }
        }
    }
    // and queries on the refit tree are complete for the new positions
    for q in random_boxes(53, 40, 30, 40) {
        let got = bvh.query(&q);
        for (i, b) in new_boxes.iter().enumerate() {
            if overlaps(&q, b) {
                assert!(got.contains(&(i as u32)), "refit tree misses primitive {i}");
            }
        }
    }
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-009: LinearBvh::refit_leaves refreshes every node box but leaves the public `bounds` field (documented 'World bounds') at the pre-refit value, so it no longer contains the primitives after they move"]
fn refit_keeps_the_world_bounds_field_in_sync_with_the_primitives() {
    let boxes = random_boxes(61, 40, 10, 8);
    let mut bvh = build(&boxes);
    let shifted: Vec<AABB> = boxes
        .iter()
        .map(|b| {
            AABB::new(
                b.min + Vec3Fix::from_int(1000, 0, 0),
                b.max + Vec3Fix::from_int(1000, 0, 0),
            )
        })
        .collect();
    bvh.refit_leaves(&shifted);
    let mut world = shifted[0];
    for b in &shifted[1..] {
        world = world.union(b);
    }
    assert_eq!(bvh.bounds, world);
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-010: BvhNode::leaf documents that count is saturated to 255 (it clamps), but a debug_assert!(count <= 255) fires first, so in a debug build the documented saturation is a panic and only a release build saturates (profile-dependent behaviour); build() never produces a leaf above 4, so only direct callers of the pub constructor see it"]
fn leaf_constructor_saturates_the_primitive_count_at_255_as_documented() {
    let aabb = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
    let n = BvhNode::leaf(&aabb, 0, 300, 7);
    assert_eq!(n.prim_count(), 255);
    assert_eq!(n.escape_idx(), 7);
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-011: the escape index is stored in 24 bits and ESCAPE_NONE (u32::MAX) is masked to 0x00FFFFFF, so escape_idx() never returns ESCAPE_NONE and an index >= 2^24 silently wraps (BvhNode::internal(.., 1 << 24) reads back as 0); no assert in build() for trees with more than 2^24 nodes"]
fn escape_index_above_the_24_bit_field_is_not_silently_truncated() {
    let aabb = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
    let n = BvhNode::internal(&aabb, 1, (1 << 24) + 5);
    assert_eq!(n.escape_idx(), (1 << 24) + 5);
}

#[test]
fn node_is_32_bytes_and_32_aligned_as_documented() {
    assert_eq!(core::mem::size_of::<BvhNode>(), 32);
    assert_eq!(core::mem::align_of::<BvhNode>(), 32);
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-013: node boxes are quantised to whole world units (floor/ceil to i32), so a scene whose objects are smaller than one unit loses the broad phase: 400 boxes of side 0.2 scattered in a 6-unit cube have 35 true overlapping pairs but find_pairs reports 18697 of the 79800 possible (the superset bound of the doc holds, the pruning does not; the near-all-pairs behaviour that 1.1.1 removed returns below one unit)"]
fn sub_unit_objects_keep_the_broad_phase_selective() {
    let n = 400usize;
    let mut r = Lcg(2024);
    let boxes: Vec<AABB> = (0..n)
        .map(|_| {
            // coordinates in [0, 6) at 1/64 resolution, side 0.2 (13/64)
            let c = |r: &mut Lcg| Fix128::from_ratio((r.next() % (6 * 64)) as i64, 64);
            let min = Vec3Fix::new(c(&mut r), c(&mut r), c(&mut r));
            let side = Fix128::from_ratio(13, 64);
            AABB::new(min, Vec3Fix::new(min.x + side, min.y + side, min.z + side))
        })
        .collect();
    let mut truth = 0usize;
    for i in 0..n {
        for j in i + 1..n {
            if overlaps(&boxes[i], &boxes[j]) {
                truth += 1;
            }
        }
    }
    let reported = build(&boxes).find_pairs().len();
    assert!(
        reported <= 8 * truth.max(1),
        "reported {reported} pairs for {truth} true overlaps"
    );
}
