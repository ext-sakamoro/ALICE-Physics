//! Oracle: the linear BVH keeps every overlapping pair at the ends of the `Fix128` range.
//!
//! # What is measured
//!
//! `LinearBvh` quantises each leaf box to `i32` (floor of the lower corner, ceiling of
//! the upper corner, clamped to the `i32` range), so its candidate pairs are a superset
//! of the pairs whose boxes overlap. The ceiling of a value in the top integer word
//! (`hi = i64::MAX` with a non-zero fraction) is past `i64`; it must clamp to `i32::MAX`
//! like every other value above the `i32` range. Adding 1 to `hi` overflowed there: a
//! panic in a debug build, and in a release build a wrap that put the box's upper bound
//! below its lower bound, so a body reaching the top of the range could vanish from the
//! candidates.
//!
//! The reference is the brute-force list of overlapping pairs (closed boxes, all axes),
//! computed from the boxes alone: the BVH's pairs must contain every one of them.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::bvh::{BvhPrimitive, LinearBvh};
use alice_physics_core::collider::AABB;
use alice_physics_core::math::{Fix128, Vec3Fix};

const TOP: Fix128 = Fix128 {
    hi: i64::MAX,
    lo: u64::MAX,
};
const BOTTOM: Fix128 = Fix128 {
    hi: i64::MIN,
    lo: 0,
};

fn on_x(lo: Fix128, hi: Fix128) -> AABB {
    let one = Fix128::ONE;
    AABB {
        min: Vec3Fix::new(lo, -one, -one),
        max: Vec3Fix::new(hi, one, one),
    }
}

fn overlap(a: &AABB, b: &AABB) -> bool {
    a.min.x <= b.max.x
        && b.min.x <= a.max.x
        && a.min.y <= b.max.y
        && b.min.y <= a.max.y
        && a.min.z <= b.max.z
        && b.min.z <= a.max.z
}

/// The BVH's candidates over `boxes` contain every pair whose boxes overlap; returns
/// how many overlapping pairs were checked.
fn assert_superset(boxes: &[AABB]) -> usize {
    let prims: Vec<BvhPrimitive> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| BvhPrimitive {
            aabb: *b,
            index: i as u32,
            morton: 0,
        })
        .collect();
    let pairs = LinearBvh::build(prims).find_pairs();
    let mut checked = 0;
    for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            if overlap(&boxes[i], &boxes[j]) {
                checked += 1;
                assert!(
                    pairs.contains(&(i as u32, j as u32)),
                    "overlapping pair ({i}, {j}) missing from the BVH candidates {pairs:?}"
                );
            }
        }
    }
    checked
}

#[test]
fn a_box_reaching_the_top_of_the_range_stays_a_candidate() {
    let near_top = Fix128 {
        hi: i64::MAX,
        lo: 1,
    };
    let boxes = [
        // reaches the very top: upper corner hi = i64::MAX with a fraction
        on_x(Fix128::from_int(1 << 40), TOP),
        on_x(Fix128::from_int(1 << 50), near_top),
        // an ordinary box that overlaps neither
        on_x(Fix128::ZERO, Fix128::ONE),
    ];
    assert!(
        assert_superset(&boxes) >= 1,
        "the scene has no overlapping pair"
    );
}

#[test]
fn boxes_at_both_ends_of_the_range_keep_their_pairs() {
    let boxes = [
        on_x(BOTTOM, Fix128::from_int(-(1 << 40))),
        on_x(Fix128::from_int(-(1 << 41)), Fix128::from_int(-(1 << 30))),
        on_x(Fix128::from_int(1 << 30), TOP),
        on_x(Fix128::from_int(1 << 62), TOP),
        on_x(BOTTOM, TOP),
    ];
    assert!(
        assert_superset(&boxes) >= 4,
        "the scene has too few overlapping pairs"
    );
}
