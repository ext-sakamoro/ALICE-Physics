//! Audit oracles for `plane_collider::PlaneCollider`.
//!
//! Closed forms: Hessian form `n.p = d`; signed distance `n.p - d`; sphere
//! penetration `r - |dist|`; box penetration = deepest of the 8 corners (brute
//! force over corners, independent of the p/n-vertex shortcut in the code).

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() <= tol,
            "{what}: axis {k} got {} want {}",
            a[k],
            b[k]
        );
    }
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `new` normalises the normal and leaves the offset as the distance along the
/// *unit* normal; a zero normal falls back to Y-up (documented for `new`, which
/// had no direct oracle: existing ones exercised `from_point_normal`).
#[test]
fn new_normalises_the_normal_and_keeps_the_offset() {
    let p = PlaneCollider::new(v3(3.0, 0.0, 4.0), fx(2.0));
    close(arr(p.normal), [0.6, 0.0, 0.8], 1e-12, "normal");
    assert!((p.offset.to_f64() - 2.0).abs() < 1e-12);
    // A point 2 along the unit normal lies on the plane.
    assert!(p.distance_to_point(v3(1.2, 0.0, 1.6)).to_f64().abs() < 1e-12);
    let z = PlaneCollider::new(Vec3Fix::ZERO, fx(5.0));
    close(arr(z.normal), [0.0, 1.0, 0.0], 0.0, "fallback");
    assert_eq!(z.offset.to_f64(), 5.0);
}

/// Sphere contact data: depth r-|d|, normal from the plane toward the centre,
/// `point_a` on the sphere surface (c - n r), `point_b` the centre's projection.
#[test]
fn sphere_contact_data_matches_the_closed_form_on_both_sides() {
    // plane y = 1, centre at y = 1.3, r = 0.5: d = 0.3, depth = 0.2.
    let p = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(1.0));
    let r = p.intersect_sphere(v3(2.0, 1.3, -1.0), fx(0.5));
    assert!(r.colliding);
    assert!((r.depth.to_f64() - 0.2).abs() < 1e-12);
    close(arr(r.normal), [0.0, 1.0, 0.0], 1e-12, "normal front");
    close(arr(r.point_a), [2.0, 0.8, -1.0], 1e-12, "point_a front");
    close(arr(r.point_b), [2.0, 1.0, -1.0], 1e-12, "point_b front");
    // behind: centre at y = 0.7 -> d = -0.3, normal flips to -y.
    let r = p.intersect_sphere(v3(2.0, 0.7, -1.0), fx(0.5));
    assert!((r.depth.to_f64() - 0.2).abs() < 1e-12);
    close(arr(r.normal), [0.0, -1.0, 0.0], 1e-12, "normal back");
    close(arr(r.point_a), [2.0, 1.2, -1.0], 1e-12, "point_a back");
    close(arr(r.point_b), [2.0, 1.0, -1.0], 1e-12, "point_b back");
    // centre exactly on the plane: dist = 0 takes the +normal branch.
    let r = p.intersect_sphere(v3(0.0, 1.0, 0.0), fx(0.5));
    assert!((r.depth.to_f64() - 0.5).abs() < 1e-12);
    close(arr(r.normal), [0.0, 1.0, 0.0], 1e-12, "normal on plane");
}

/// Touching (|d| == r) is not a collision.
#[test]
fn a_sphere_exactly_touching_the_plane_does_not_collide() {
    let p = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    assert!(!p.intersect_sphere(v3(0.0, 0.5, 0.0), fx(0.5)).colliding);
    assert!(!p.intersect_sphere(v3(0.0, -0.5, 0.0), fx(0.5)).colliding);
    assert!(p.intersect_sphere(v3(0.0, 0.499, 0.0), fx(0.5)).colliding);
}

/// Oblique plane + offset: contact data along n = (1,2,2)/3.
#[test]
fn oblique_plane_sphere_contact() {
    let p = PlaneCollider::new(v3(1.0, 2.0, 2.0), fx(1.0));
    let n = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let c = [0.5, 1.0, 0.25];
    let d = n[0] * c[0] + n[1] * c[1] + n[2] * c[2] - 1.0;
    let r = 0.9;
    let hit = p.intersect_sphere(v3(c[0], c[1], c[2]), fx(r));
    assert!(hit.colliding);
    assert!((hit.depth.to_f64() - (r - d.abs())).abs() < 1e-12);
    let s = d.signum();
    close(arr(hit.normal), [s * n[0], s * n[1], s * n[2]], 1e-12, "n");
    close(
        arr(hit.point_b),
        [c[0] - n[0] * d, c[1] - n[1] * d, c[2] - n[2] * d],
        1e-12,
        "foot",
    );
}

/// Box vs plane against brute force over the 8 corners, for random oblique planes
/// (all sign combinations of the normal, which select different p/n vertices).
#[test]
fn box_penetration_is_the_deepest_corner_for_any_normal_orientation() {
    let mut s = 31u64;
    let mut hits = 0;
    for _ in 0..400 {
        let nrm = [lcg(&mut s) - 0.5, lcg(&mut s) - 0.5, lcg(&mut s) - 0.5];
        let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        if len < 0.2 {
            continue;
        }
        let u = [nrm[0] / len, nrm[1] / len, nrm[2] / len];
        let off = (lcg(&mut s) - 0.5) * 2.0;
        let lo = [
            (lcg(&mut s) - 0.5) * 3.0,
            (lcg(&mut s) - 0.5) * 3.0,
            (lcg(&mut s) - 0.5) * 3.0,
        ];
        let hi = [
            lo[0] + 0.2 + lcg(&mut s),
            lo[1] + 0.2 + lcg(&mut s),
            lo[2] + 0.2 + lcg(&mut s),
        ];
        let plane = PlaneCollider::new(v3(nrm[0], nrm[1], nrm[2]), fx(off));
        // brute force over corners
        let mut min_d = f64::MAX;
        let mut deepest = [0.0; 3];
        for m in 0..8 {
            let c = [
                if m & 1 == 0 { lo[0] } else { hi[0] },
                if m & 2 == 0 { lo[1] } else { hi[1] },
                if m & 4 == 0 { lo[2] } else { hi[2] },
            ];
            let d = u[0] * c[0] + u[1] * c[1] + u[2] * c[2] - off;
            if d < min_d {
                min_d = d;
                deepest = c;
            }
        }
        let aabb = AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));
        let hit = plane.intersect_aabb(&aabb);
        if min_d.abs() < 1e-9 {
            continue;
        }
        assert_eq!(
            hit.colliding,
            min_d < 0.0,
            "collide flag, min corner dist {min_d}"
        );
        if min_d < 0.0 {
            hits += 1;
            assert!(
                (hit.depth.to_f64() + min_d).abs() < 1e-9,
                "depth {} vs {}",
                hit.depth.to_f64(),
                -min_d
            );
            close(arr(hit.normal), u, 1e-9, "normal");
            close(arr(hit.point_a), deepest, 1e-9, "deepest corner");
        }
    }
    assert!(hits > 50, "scene did not exercise collisions ({hits})");
}

/// A box that only touches (deepest corner on the plane) does not collide, in line with
/// `is_front` counting the surface as front.
#[test]
fn a_box_resting_on_the_plane_is_not_a_hit() {
    let p = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(1.0));
    let touching = AABB::new(v3(-1.0, 1.0, -1.0), v3(1.0, 3.0, 1.0));
    assert!(!p.intersect_aabb(&touching).colliding);
    assert!(p.is_front(v3(0.0, 1.0, 0.0)));
}

/// Contact points of every colliding box result satisfy the same invariant as the sphere
/// branch and the partial-overlap branch: `point_b - point_a = depth * normal` (the plane
/// point sits `depth` along the normal from the penetrating point). The
/// fully-behind case used to project the box centre instead of the deepest
/// corner (AUD-A-S3W1-002).
#[test]
fn box_contact_points_are_depth_apart_along_the_normal_in_every_branch() {
    let p = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    // Fully behind, off-centre corner: box x in [0,2], y in [-3,-1].
    let aabb = AABB::new(v3(0.0, -3.0, 0.0), v3(2.0, -1.0, 2.0));
    let h = p.intersect_aabb(&aabb);
    assert!(h.colliding);
    let gap = [
        h.point_b.x.to_f64() - h.point_a.x.to_f64(),
        h.point_b.y.to_f64() - h.point_a.y.to_f64(),
        h.point_b.z.to_f64() - h.point_a.z.to_f64(),
    ];
    let want = [0.0, 3.0, 0.0];
    close(gap, want, 1e-12, "point_b - point_a");
}

/// Same invariant in the partial overlap branch (must hold today: green).
#[test]
fn partial_overlap_contact_points_are_depth_apart_along_the_normal() {
    let p = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    let aabb = AABB::new(v3(0.0, -0.5, 0.0), v3(2.0, 1.0, 2.0));
    let h = p.intersect_aabb(&aabb);
    assert!(h.colliding);
    close(arr(h.point_a), [0.0, -0.5, 0.0], 1e-12, "point_a");
    close(arr(h.point_b), [0.0, 0.0, 0.0], 1e-12, "point_b");
}

/// `project_point` is idempotent and moves along the normal by the signed distance.
#[test]
fn projection_lands_on_the_plane_and_is_idempotent() {
    let p = PlaneCollider::from_point_normal(v3(1.0, 1.0, 1.0), v3(2.0, -1.0, 2.0));
    for pt in [v3(3.0, 0.0, -2.0), v3(-1.0, 4.0, 2.5), v3(1.0, 1.0, 1.0)] {
        let q = p.project_point(pt);
        assert!(p.distance_to_point(q).to_f64().abs() < 1e-12);
        let r = p.project_point(q);
        close(arr(r), arr(q), 1e-12, "idempotent");
        let d = p.distance_to_point(pt).to_f64();
        let n = arr(p.normal);
        let a = arr(pt);
        close(
            arr(q),
            [a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2]],
            1e-12,
            "foot",
        );
    }
}

/// `flip` is an involution and swaps `is_front` away from the surface.
#[test]
fn flip_swaps_sides_and_is_an_involution() {
    let p = PlaneCollider::new(v3(1.0, 1.0, 0.0), fx(0.5));
    let f = p.flip();
    let ff = f.flip();
    assert_eq!(ff, p);
    for pt in [v3(3.0, 3.0, 0.0), v3(-3.0, -3.0, 1.0)] {
        assert_ne!(p.is_front(pt), f.is_front(pt));
        assert!(
            (p.distance_to_point(pt).to_f64() + f.distance_to_point(pt).to_f64()).abs() < 1e-12
        );
    }
}

/// `from_point_normal` with a non-unit normal: offset is `n_unit . point`, and the zero
/// normal falls back to Y-up with `offset = point.y`.
#[test]
fn from_point_normal_uses_the_unit_normal_for_the_offset() {
    let p = PlaneCollider::from_point_normal(v3(5.0, 0.0, 10.0), v3(3.0, 0.0, 4.0));
    close(arr(p.normal), [0.6, 0.0, 0.8], 1e-12, "normal");
    assert!(
        (p.offset.to_f64() - 11.0).abs() < 1e-12,
        "0.6*5 + 0.8*10 = 11"
    );
    let z = PlaneCollider::from_point_normal(v3(7.0, 2.5, 9.0), Vec3Fix::ZERO);
    close(arr(z.normal), [0.0, 1.0, 0.0], 0.0, "fallback");
    assert!((z.offset.to_f64() - 2.5).abs() < 1e-12);
}

/// Fully-behind box: depth is the deepest corner, the normal is the plane normal and
/// `point_a` is the deepest corner (`point_b` is checked by the test above).
#[test]
fn fully_behind_box_reports_depth_normal_and_deepest_corner() {
    let p = PlaneCollider::new(v3(1.0, 1.0, 0.0), fx(0.0));
    let n = 1.0 / 2f64.sqrt();
    // box x in [-3,-1], y in [-4,-2]: deepest corner (-3,-4,z), dist = -7n
    let aabb = AABB::new(v3(-3.0, -4.0, -1.0), v3(-1.0, -2.0, 1.0));
    let h = p.intersect_aabb(&aabb);
    assert!(h.colliding);
    assert!((h.depth.to_f64() - 7.0 * n).abs() < 1e-9);
    close(arr(h.normal), [n, n, 0.0], 1e-9, "normal");
    // deepest corner: x = -3 (min), y = -4 (min); z has zero normal component
    assert!((h.point_a.x.to_f64() + 3.0).abs() < 1e-12);
    assert!((h.point_a.y.to_f64() + 4.0).abs() < 1e-12);
    // negative-normal plane: the deepest corner flips to the max side
    let q = PlaneCollider::new(v3(-1.0, -1.0, 0.0), fx(0.0));
    let aabb = AABB::new(v3(1.0, 2.0, -1.0), v3(3.0, 4.0, 1.0));
    let h = q.intersect_aabb(&aabb);
    assert!(h.colliding);
    assert!((h.depth.to_f64() - 7.0 * n).abs() < 1e-9);
    assert!((h.point_a.x.to_f64() - 3.0).abs() < 1e-12);
    assert!((h.point_a.y.to_f64() - 4.0).abs() < 1e-12);
}
