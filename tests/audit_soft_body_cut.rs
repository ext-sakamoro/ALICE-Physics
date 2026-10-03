//! Audit oracles for `alice_physics::soft_body_cut`.
//!
//! Expected values are hand-computed edge/plane intersections (linear
//! interpolation of the signed distances), independent of the
//! implementation's arithmetic order.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::soft_body_cut::{cut_cloth, cut_deformable, CutPlane};

fn fx(f: f64) -> Fix128 {
    Fix128::from_f64(f)
}
fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn vclose(a: Vec3Fix, x: f64, y: f64, z: f64) -> bool {
    (a.x.to_f64() - x).abs() < 1.0e-9
        && (a.y.to_f64() - y).abs() < 1.0e-9
        && (a.z.to_f64() - z).abs() < 1.0e-9
}
fn plane_y(y: f64) -> CutPlane {
    CutPlane {
        point: v(0.0, y, 0.0),
        normal: Vec3Fix::UNIT_Y,
    }
}

/// Edge from y = +1 to y = -3 crosses y = 0 at one quarter of the way:
/// t = d_i / (d_i - d_j) = 1 / 4, intersection (0.5, 0, 0) for x 0 -> 2.
#[test]
fn intersection_is_distance_weighted_not_midpoint() {
    let p = [v(0.0, 1.0, 0.0), v(2.0, -3.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane_y(0.0));
    assert_eq!(r.new_particles.len(), 1);
    assert!(vclose(r.new_particles[0], 0.5, 0.0, 0.0));
}

/// Edge order does not change the intersection point: (1, 0) gives the same
/// location as (0, 1); the removed constraint keeps the given order.
#[test]
fn reversed_edge_gives_same_intersection_point() {
    let p = [v(0.0, 1.0, 0.0), v(2.0, -3.0, 0.0)];
    let r = cut_deformable(&p, &[(1, 0)], &plane_y(0.0));
    assert_eq!(r.removed_constraints, vec![(1, 0)]);
    assert!(vclose(r.new_particles[0], 0.5, 0.0, 0.0));
}

/// Oblique plane x + y = 2 (unnormalised normal (1,1,0)): the intersection
/// lies on the plane. Edge (0,0,0) -> (4,4,0): d = -2 and 6 (times |n|^2
/// scaling cancels in t): t = 2/8 = 1/4... with signed distances relative
/// to the unnormalised normal: d0 = -2, d1 = +6, t = -2/-8 = 0.25 so the
/// point is (1, 1, 0), which satisfies x + y = 2.
#[test]
fn oblique_unnormalised_plane_intersection_lies_on_plane() {
    let plane = CutPlane {
        point: v(2.0, 0.0, 0.0),
        normal: v(1.0, 1.0, 0.0),
    };
    let p = [v(0.0, 0.0, 0.0), v(4.0, 4.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane);
    assert_eq!(r.side_a_particles, vec![1]);
    assert_eq!(r.side_b_particles, vec![0]);
    assert!(vclose(r.new_particles[0], 1.0, 1.0, 0.0));
}

/// Flipping the normal swaps the two sides and leaves the cut geometry
/// (removed edges and intersection points) unchanged.
#[test]
fn flipping_normal_swaps_sides_only() {
    let p = [
        v(0.0, 3.0, 0.0),
        v(0.0, -1.0, 0.0),
        v(1.0, 2.0, 0.0),
        v(1.0, -5.0, 0.0),
    ];
    let e = [(0, 1), (2, 3), (0, 2)];
    let up = plane_y(0.0);
    let down = CutPlane {
        point: up.point,
        normal: -up.normal,
    };
    let a = cut_deformable(&p, &e, &up);
    let b = cut_deformable(&p, &e, &down);
    assert_eq!(a.side_a_particles, vec![0, 2]);
    assert_eq!(a.side_b_particles, vec![1, 3]);
    assert_eq!(b.side_a_particles, a.side_b_particles);
    assert_eq!(b.side_b_particles, a.side_a_particles);
    assert_eq!(a.removed_constraints, b.removed_constraints);
    assert_eq!(a.new_particles.len(), b.new_particles.len());
    for (x, y) in a.new_particles.iter().zip(&b.new_particles) {
        assert!(vclose(*x, y.x.to_f64(), y.y.to_f64(), y.z.to_f64()));
    }
}

/// The plane anchor point offsets the plane: y = 2 splits at height 2.
#[test]
fn plane_point_offset_is_respected() {
    let p = [v(0.0, 1.0, 0.0), v(0.0, 4.0, 0.0), v(0.0, 2.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane_y(2.0));
    assert_eq!(r.side_b_particles, vec![0]);
    assert_eq!(r.side_a_particles, vec![1, 2]);
    assert!(vclose(r.new_particles[0], 0.0, 2.0, 0.0));
}

/// A particle exactly on the plane is side A (documented `>= 0`), and an
/// edge from that particle to a side-B particle is cut at the particle.
#[test]
fn on_plane_particle_belongs_to_side_a() {
    let p = [v(7.0, 0.0, 1.0), v(7.0, -2.0, 1.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane_y(0.0));
    assert_eq!(r.side_a_particles, vec![0]);
    assert_eq!(r.side_b_particles, vec![1]);
    assert_eq!(r.removed_constraints, vec![(0, 1)]);
    assert!(vclose(r.new_particles[0], 7.0, 0.0, 1.0));
}

/// Both endpoints on the plane: not a crossing edge.
#[test]
fn edge_lying_in_the_plane_is_not_cut() {
    let p = [v(0.0, 0.0, 0.0), v(3.0, 0.0, 5.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane_y(0.0));
    assert!(r.removed_constraints.is_empty());
    assert!(r.new_particles.is_empty());
}

/// Self-edge `(i, i)` never crosses.
#[test]
fn self_edge_never_crosses() {
    let p = [v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 0), (1, 1)], &plane_y(0.0));
    assert!(r.removed_constraints.is_empty());
}

/// Duplicate edges are each reported (the function does not deduplicate):
/// one new particle per listed crossing constraint.
#[test]
fn duplicate_edges_each_produce_an_intersection() {
    let p = [v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 1), (0, 1), (1, 0)], &plane_y(0.0));
    assert_eq!(r.removed_constraints.len(), 3);
    assert_eq!(r.new_particles.len(), 3);
}

/// Each side's index list is ascending and together they partition
/// `0..n` exactly once.
#[test]
fn sides_partition_all_particles_in_ascending_order() {
    let p: Vec<Vec3Fix> = (0..20)
        .map(|i| v(i as f64, ((i * 7) % 11) as f64 - 5.0, 0.0))
        .collect();
    let r = cut_deformable(&p, &[], &plane_y(0.0));
    let mut all: Vec<usize> = r.side_a_particles.clone();
    all.extend(&r.side_b_particles);
    all.sort_unstable();
    assert_eq!(all, (0..20).collect::<Vec<_>>());
    assert!(r.side_a_particles.windows(2).all(|w| w[0] < w[1]));
    assert!(r.side_b_particles.windows(2).all(|w| w[0] < w[1]));
    for &i in &r.side_a_particles {
        assert!(p[i].y.to_f64() >= 0.0);
    }
    for &i in &r.side_b_particles {
        assert!(p[i].y.to_f64() < 0.0);
    }
}

/// Edges with one out-of-range index are skipped without panic, while valid
/// edges in the same list are still processed in order.
#[test]
fn out_of_range_edge_is_skipped_and_valid_edges_continue() {
    let p = [v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0)];
    let r = cut_deformable(
        &p,
        &[(0, 5), (9, 1), (0, 1), (usize::MAX, 0)],
        &plane_y(0.0),
    );
    assert_eq!(r.removed_constraints, vec![(0, 1)]);
    assert_eq!(r.new_particles.len(), 1);
}

/// Zero normal: every signed distance is 0, so everything is side A and
/// nothing is cut.
#[test]
fn zero_normal_puts_everything_on_side_a() {
    let plane = CutPlane {
        point: Vec3Fix::ZERO,
        normal: Vec3Fix::ZERO,
    };
    let p = [v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 1)], &plane);
    assert_eq!(r.side_a_particles, vec![0, 1]);
    assert!(r.removed_constraints.is_empty());
}

/// `cut_cloth` returns exactly the same topology as `cut_deformable`.
#[test]
fn cut_cloth_equals_cut_deformable_on_a_grid() {
    let mut p = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            p.push(v(i as f64, j as f64 - 1.5, 0.25 * i as f64));
        }
    }
    let mut e = Vec::new();
    for j in 0..4usize {
        for i in 0..4usize {
            if i + 1 < 4 {
                e.push((j * 4 + i, j * 4 + i + 1));
            }
            if j + 1 < 4 {
                e.push((j * 4 + i, (j + 1) * 4 + i));
            }
        }
    }
    let plane = plane_y(0.0);
    let a = cut_deformable(&p, &e, &plane);
    let b = cut_cloth(&p, &e, &plane);
    assert_eq!(a.side_a_particles, b.side_a_particles);
    assert_eq!(a.side_b_particles, b.side_b_particles);
    assert_eq!(a.removed_constraints, b.removed_constraints);
    assert_eq!(a.new_particles, b.new_particles);
    // 4 vertical edges (columns) cross y = 0: rows y = -1.5,-0.5 vs 0.5,1.5
    assert_eq!(a.removed_constraints.len(), 4);
}

/// An index exactly equal to the particle count is out of range (the last
/// valid index is `len - 1`) and is skipped, for either endpoint.
#[test]
fn index_equal_to_particle_count_is_out_of_range() {
    let p = [v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0)];
    let r = cut_deformable(&p, &[(0, 2), (2, 1), (2, 2), (0, 1)], &plane_y(0.0));
    assert_eq!(r.removed_constraints, vec![(0, 1)]);
    assert_eq!(r.new_particles.len(), 1);
}
