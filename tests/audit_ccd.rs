//! Audit oracles for `alice_physics::ccd` (S2-2 audit).
//!
//! Time-of-impact answers are checked against brute-force time sampling of the
//! trajectory (independent of the closed forms in `src/ccd.rs`) and against
//! first-principles geometry. Scenes come from a fixed LCG so runs are
//! reproducible.
#![allow(clippy::disallowed_methods)]

use alice_physics::ccd::{
    aabb_plane_toi, adaptive_toi_substeps, capsule_plane_toi, conservative_advancement, needs_ccd,
    speculative_contact, sphere_capsule_toi, sphere_plane_toi, sphere_sphere_toi, swept_aabb,
    CcdConfig,
};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// multiple of 1/8 in [-lo, hi]
    fn q(&mut self, span: i64) -> f64 {
        ((self.next() % (16 * span as u64 + 1)) as i64 - 8 * span) as f64 / 8.0
    }
}

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn f(x: Fix128) -> f64 {
    x.to_f64()
}
fn len3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Galilean invariance and symmetry of the sphere-sphere TOI, and the geometric
/// definition of the answer: at t the centres are exactly r_a + r_b apart and
/// the contact point lies on A's surface along the A->B normal.
#[test]
fn sphere_sphere_toi_satisfies_the_touching_condition_on_random_scenes() {
    let mut rng = Lcg(12345);
    let mut hits = 0;
    for _ in 0..600 {
        let pa = [rng.q(3), rng.q(3), rng.q(3)];
        let pb = [rng.q(3), rng.q(3), rng.q(3)];
        let va = [rng.q(12), rng.q(12), rng.q(12)];
        let vb = [rng.q(12), rng.q(12), rng.q(12)];
        let (ra, rb) = (
            0.5 + (rng.next() % 8) as f64 / 8.0,
            0.5 + (rng.next() % 8) as f64 / 8.0,
        );
        let d0 = len3([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]);
        if d0 < ra + rb {
            continue; // starts overlapping: separate branch
        }
        let toi = sphere_sphere_toi(
            v3(pa[0], pa[1], pa[2]),
            fx(ra),
            v3(va[0], va[1], va[2]),
            v3(pb[0], pb[1], pb[2]),
            fx(rb),
            v3(vb[0], vb[1], vb[2]),
        );
        // brute force: first sampled t with the spheres overlapping
        let dist_at = |t: f64| {
            len3([
                pb[0] + vb[0] * t - pa[0] - va[0] * t,
                pb[1] + vb[1] * t - pa[1] - va[1] * t,
                pb[2] + vb[2] * t - pa[2] - va[2] * t,
            ])
        };
        let n = 4096;
        let mut min_d = f64::MAX;
        for s in 0..=n {
            min_d = min_d.min(dist_at(s as f64 / n as f64));
        }
        match toi {
            Some(h) => {
                hits += 1;
                let t = f(h.t);
                assert!((0.0..=1.0).contains(&t));
                assert!(
                    (dist_at(t) - (ra + rb)).abs() < 1e-6,
                    "not touching at t={t}: {}",
                    dist_at(t)
                );
                let nrm = [f(h.normal.x), f(h.normal.y), f(h.normal.z)];
                assert!((len3(nrm) - 1.0).abs() < 1e-6);
                let pa_t = [pa[0] + va[0] * t, pa[1] + va[1] * t, pa[2] + va[2] * t];
                let pt = [
                    f(h.point.x) - pa_t[0],
                    f(h.point.y) - pa_t[1],
                    f(h.point.z) - pa_t[2],
                ];
                assert!(
                    (len3(pt) - ra).abs() < 1e-6,
                    "contact point not on A's surface"
                );
            }
            None => assert!(
                min_d > ra + rb - 0.02,
                "missed a collision: min distance {min_d} < {}",
                ra + rb
            ),
        }
    }
    assert!(hits > 10, "scene set too weak: {hits} hits");
}

/// A common velocity added to both spheres cannot change when they touch.
#[test]
fn sphere_sphere_toi_is_galilean_invariant() {
    let a = (v3(-5.0, 0.0, 0.0), fx(1.0), v3(8.0, 1.0, 0.0));
    let b = (v3(6.0, 0.5, 0.0), fx(1.5), v3(-4.0, 0.0, 0.0));
    let t0 = sphere_sphere_toi(a.0, a.1, a.2, b.0, b.1, b.2).unwrap().t;
    let shift = v3(3.0, -2.0, 1.0);
    let t1 = sphere_sphere_toi(a.0, a.1, a.2 + shift, b.0, b.1, b.2 + shift)
        .unwrap()
        .t;
    assert!((f(t0) - f(t1)).abs() < 1e-9, "{} vs {}", f(t0), f(t1));
    // symmetry: swapping A and B keeps t and flips the normal
    let sw = sphere_sphere_toi(b.0, b.1, b.2, a.0, a.1, a.2).unwrap();
    assert!((f(sw.t) - f(t0)).abs() < 1e-9);
    let orig = sphere_sphere_toi(a.0, a.1, a.2, b.0, b.1, b.2).unwrap();
    assert!((f(sw.normal.x) + f(orig.normal.x)).abs() < 1e-6);
}

/// Two-sided sphere-plane TOI against brute-force sampling, front and back.
#[test]
fn sphere_plane_toi_matches_brute_force_on_both_sides() {
    let mut rng = Lcg(99);
    let (mut front, mut back) = (0, 0);
    for _ in 0..300 {
        let n = [0.0, 1.0, 0.0];
        let off = rng.q(2);
        let c = [rng.q(4), rng.q(8), rng.q(4)];
        let r = 0.5 + (rng.next() % 4) as f64 / 4.0;
        let vel = [rng.q(6), rng.q(10), rng.q(6)];
        let toi = sphere_plane_toi(
            v3(c[0], c[1], c[2]),
            fx(r),
            v3(vel[0], vel[1], vel[2]),
            v3(n[0], n[1], n[2]),
            fx(off),
        );
        let dist_at = |t: f64| (c[1] + vel[1] * t) - off;
        let d0 = dist_at(0.0);
        if d0.abs() <= r {
            assert_eq!(f(toi.expect("already touching").t), 0.0);
            continue;
        }
        let steps = 8192;
        let first = (0..=steps)
            .map(|s| s as f64 / steps as f64)
            .find(|&t| dist_at(t).abs() <= r);
        match (toi, first) {
            (Some(h), Some(tb)) => {
                if d0 > 0.0 {
                    front += 1
                } else {
                    back += 1
                }
                assert!(
                    (f(h.t) - tb).abs() < 2.0 / steps as f64 + 1e-6,
                    "t {} vs brute {tb}",
                    f(h.t)
                );
                let side = if d0 > 0.0 { 1.0 } else { -1.0 };
                assert!(
                    (f(h.normal.y) - side).abs() < 1e-9,
                    "normal not on the sphere's side"
                );
            }
            (None, None) => {}
            (a, b) => panic!(
                "mismatch toi={:?} brute={:?} (d0 {d0}, r {r}, v {vel:?})",
                a.map(|h| f(h.t)),
                b
            ),
        }
    }
    assert!(front > 5 && back > 5, "front {front} back {back}");
}

/// Swept AABB against brute-force overlap sampling (scenes with a 0.02 margin so
/// grazing contacts never decide the verdict).
#[test]
fn swept_aabb_matches_brute_force_overlap_sampling() {
    let mut rng = Lcg(2024);
    let (mut hit, mut miss) = (0, 0);
    for _ in 0..400 {
        let mk = |rng: &mut Lcg| {
            let c = [rng.q(5), rng.q(5), rng.q(5)];
            let h = [
                0.5 + (rng.next() % 4) as f64 / 4.0,
                0.5 + (rng.next() % 4) as f64 / 4.0,
                0.5 + (rng.next() % 4) as f64 / 4.0,
            ];
            (c, h)
        };
        let (mc, mh) = mk(&mut rng);
        let (tc, th) = mk(&mut rng);
        let vel = [rng.q(8), rng.q(8), rng.q(8)];
        let overlap = |t: f64, grow: f64| {
            (0..3).all(|k| (mc[k] + vel[k] * t - tc[k]).abs() <= mh[k] + th[k] + grow)
        };
        let steps = 4096;
        let any = |grow: f64| (0..=steps).any(|s| overlap(s as f64 / steps as f64, grow));
        let moving = AABB::new(
            v3(mc[0] - mh[0], mc[1] - mh[1], mc[2] - mh[2]),
            v3(mc[0] + mh[0], mc[1] + mh[1], mc[2] + mh[2]),
        );
        let target = AABB::new(
            v3(tc[0] - th[0], tc[1] - th[1], tc[2] - th[2]),
            v3(tc[0] + th[0], tc[1] + th[1], tc[2] + th[2]),
        );
        let got = swept_aabb(&moving, v3(vel[0], vel[1], vel[2]), &target);
        if any(-0.02) {
            let t = f(got.expect("margin-hit must be found"));
            hit += 1;
            let first = (0..=steps)
                .map(|s| s as f64 / steps as f64)
                .find(|&t| overlap(t, 0.0))
                .unwrap();
            assert!(
                (t - first).abs() < 2.0 / steps as f64 + 1e-6,
                "t {t} vs brute {first}"
            );
        } else if !any(0.02) {
            miss += 1;
            assert!(got.is_none(), "reported {:?} for a clear miss", got.map(f));
        }
    }
    assert!(hit > 20 && miss > 20, "hit {hit} miss {miss}");
}

/// needs_ccd uses the Euclidean speed: (3,4,12) has speed 13, so with dt = 1 the
/// displacement test is 13 > r/2 (strict) and the speed test 13 > threshold.
#[test]
fn needs_ccd_uses_euclidean_speed_with_strict_thresholds() {
    let cfg = CcdConfig::default();
    let vel = v3(3.0, 4.0, 12.0);
    assert!(needs_ccd(vel, fx(25.0), Fix128::ONE, &cfg)); // 13 > 12.5
    assert!(!needs_ccd(vel, fx(26.0), Fix128::ONE, &cfg)); // 13 > 13 is false
                                                           // speed 5 is not > threshold 5
    assert!(!needs_ccd(v3(3.0, 4.0, 0.0), fx(0.5), Fix128::ONE, &cfg));
}

/// speculative_contact must flag every pair that sphere_sphere_toi says touches
/// within the step (it is a conservative superset by construction).
#[test]
fn speculative_contact_covers_every_sphere_sphere_impact() {
    let mut rng = Lcg(777);
    let mut checked = 0;
    for _ in 0..900 {
        let pa = [rng.q(3), rng.q(3), rng.q(3)];
        let pb = [rng.q(3), rng.q(3), rng.q(3)];
        let va = [rng.q(12), rng.q(12), rng.q(12)];
        let vb = [rng.q(12), rng.q(12), rng.q(12)];
        let (ra, rb) = (0.5, 0.75);
        if len3([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]) < 1e-6 {
            continue;
        }
        let toi = sphere_sphere_toi(
            v3(pa[0], pa[1], pa[2]),
            fx(ra),
            v3(va[0], va[1], va[2]),
            v3(pb[0], pb[1], pb[2]),
            fx(rb),
            v3(vb[0], vb[1], vb[2]),
        );
        if toi.is_some() {
            checked += 1;
            let spec = speculative_contact(
                v3(pa[0], pa[1], pa[2]),
                v3(va[0], va[1], va[2]),
                fx(ra),
                v3(pb[0], pb[1], pb[2]),
                v3(vb[0], vb[1], vb[2]),
                fx(rb),
                Fix128::ONE,
            );
            assert!(
                spec.is_some(),
                "TOI hit but no speculative contact: pa {pa:?} pb {pb:?} va {va:?} vb {vb:?}"
            );
        }
    }
    assert!(checked > 20, "{checked}");
}

/// collider::Contact documents (and the solver and EPA use) a B->A normal:
/// translating A by depth * normal must separate the pair. speculative_contact
/// must report the same B->A normal (an A->B normal would push A deeper into B).
#[test]
fn speculative_contact_normal_follows_the_b_to_a_contract() {
    let c = speculative_contact(
        v3(0.0, 0.0, 0.0),
        Vec3Fix::ZERO,
        fx(1.0),
        v3(1.0, 0.0, 0.0),
        Vec3Fix::ZERO,
        fx(1.0),
        Fix128::ONE,
    )
    .unwrap();
    assert!((f(c.depth) - 1.0).abs() < 1e-9);
    let new_a = [
        f(c.normal.x) * f(c.depth),
        f(c.normal.y) * f(c.depth),
        f(c.normal.z) * f(c.depth),
    ];
    let centre_gap = len3([1.0 - new_a[0], 0.0 - new_a[1], 0.0 - new_a[2]]);
    assert!(
        centre_gap >= 2.0 - 1e-9,
        "after translating A by depth*normal the centres are {centre_gap} apart (< 2)"
    );
}

/// Two coincident, overlapping spheres are the deepest overlap there is; the
/// sphere-sphere TOI reports t = 0 for them but speculative_contact returns None.
#[test]
#[ignore = "known defect: AUD-A-S2W2-009: speculative_contact returns None for coincident centres (dist == 0 guard) although the spheres overlap by r_a + r_b; sphere_sphere_toi reports Some(t = 0) for the same pair"]
fn speculative_contact_reports_coincident_overlapping_spheres() {
    let p = v3(1.0, 2.0, 3.0);
    assert!(sphere_sphere_toi(p, fx(1.0), Vec3Fix::ZERO, p, fx(1.0), Vec3Fix::ZERO).is_some());
    let c = speculative_contact(
        p,
        Vec3Fix::ZERO,
        fx(1.0),
        p,
        Vec3Fix::ZERO,
        fx(1.0),
        Fix128::ONE,
    );
    assert!(
        c.is_some(),
        "coincident overlapping spheres produced no contact"
    );
}

/// Capsule against a plane, front side, tilted capsule: the lower endpoint
/// touches first. a=(0,6,0) b=(4,10,0), r=1, plane y=0 moving (0,-8,0):
/// lower endpoint a, y=6 -> touches at 6 - 8t = 1 -> t = 5/8.
#[test]
fn capsule_plane_toi_front_side_uses_the_lower_endpoint() {
    let h = capsule_plane_toi(
        v3(0.0, 6.0, 0.0),
        v3(4.0, 10.0, 0.0),
        fx(1.0),
        v3(0.0, -8.0, 0.0),
        v3(0.0, 1.0, 0.0),
        Fix128::ZERO,
    )
    .unwrap();
    assert!((f(h.t) - 0.625).abs() < 1e-9);
}

/// The same capsule mirrored behind the plane and moving toward it: the endpoint
/// nearest the plane (the larger signed distance) must touch first. a=(0,-10,0)
/// b=(5,-6,0), r=1, v=(0,10,0): b touches at -6 + 10t = -1 -> t = 0.5.
#[test]
#[ignore = "known defect: AUD-A-S2W2-010: capsule_plane_toi always takes the endpoint with the smaller signed distance, correct only in front of the plane; behind the plane it picks the far endpoint: expected t = 0.5, returns 0.9"]
fn capsule_plane_toi_behind_the_plane_uses_the_nearer_endpoint() {
    let h = capsule_plane_toi(
        v3(0.0, -10.0, 0.0),
        v3(5.0, -6.0, 0.0),
        fx(1.0),
        v3(0.0, 10.0, 0.0),
        v3(0.0, 1.0, 0.0),
        Fix128::ZERO,
    )
    .unwrap();
    assert!((f(h.t) - 0.5).abs() < 1e-9, "t = {}", f(h.t));
}

/// A capsule that crosses the plane (one endpoint on each side, both farther than
/// the radius) intersects it at t = 0 even at rest.
#[test]
#[ignore = "known defect: AUD-A-S2W2-010: capsule_plane_toi judges only one endpoint sphere, so a capsule a=(0,-5,0) b=(0,5,0) r=1 straddling the plane y=0 at rest returns None instead of an impact at t = 0"]
fn capsule_plane_toi_straddling_capsule_is_already_touching() {
    let h = capsule_plane_toi(
        v3(0.0, -5.0, 0.0),
        v3(0.0, 5.0, 0.0),
        fx(1.0),
        Vec3Fix::ZERO,
        v3(0.0, 1.0, 0.0),
        Fix128::ZERO,
    );
    assert_eq!(h.map(|x| f(x.t)), Some(0.0));
}

/// Sphere vs static capsule along the axis direction: the sphere slides along the
/// capsule while closing in, so the closest axis point moves. a=(-10,0,0)
/// b=(10,0,0), cap r=1, sphere r=1 from (0,10,0) with v=(10,-10,0): distance to
/// the axis is 10 - 10t (x stays inside the segment) -> contact at t = 0.8.
#[test]
#[ignore = "known defect: AUD-A-S2W2-011: sphere_capsule_toi freezes the closest axis point at the start position (treats it as a static sphere), so a sphere that slides along the capsule while closing in is missed: expected t = 0.8, returns None"]
fn sphere_capsule_toi_sliding_approach_hits_at_the_right_time() {
    let h = sphere_capsule_toi(
        v3(0.0, 10.0, 0.0),
        fx(1.0),
        v3(10.0, -10.0, 0.0),
        v3(-10.0, 0.0, 0.0),
        v3(10.0, 0.0, 0.0),
        fx(1.0),
    );
    let h = h.expect("the sphere reaches the capsule at t = 0.8");
    assert!((f(h.t) - 0.8).abs() < 1e-6, "t = {}", f(h.t));
}

/// Perpendicular approach (closest axis point does not move): exact.
#[test]
fn sphere_capsule_toi_perpendicular_approach_matches_closed_form() {
    let h = sphere_capsule_toi(
        v3(2.0, 10.0, 0.0),
        fx(1.0),
        v3(0.0, -10.0, 0.0),
        v3(-10.0, 0.0, 0.0),
        v3(10.0, 0.0, 0.0),
        fx(1.5),
    )
    .unwrap();
    // distance 10 - 10t = 2.5 -> t = 0.75
    assert!((f(h.t) - 0.75).abs() < 1e-9);
}

/// Half-space vs two-sided plane: the sphere version is two-sided (a sphere
/// behind the plane and moving away is a miss). A point-sized AABB in the same
/// situation is reported as an impact at t = 0.
#[test]
#[ignore = "known defect: AUD-A-S2W2-012: aabb_plane_toi treats the back side of the plane as solid (support distance <= 0 means impact at t = 0) while sphere_plane_toi is two-sided: a point box at y=-5 moving away from the plane y=0 returns Some(t=0), the radius-0 sphere returns None"]
fn aabb_plane_toi_is_consistent_with_the_two_sided_sphere_version() {
    let p = v3(0.0, -5.0, 0.0);
    let n = v3(0.0, 1.0, 0.0);
    let away = v3(0.0, -1.0, 0.0);
    let sphere = sphere_plane_toi(p, Fix128::ZERO, away, n, Fix128::ZERO);
    let aabb = aabb_plane_toi(&AABB::new(p, p), away, n, Fix128::ZERO);
    assert_eq!(
        sphere.is_some(),
        aabb.is_some(),
        "sphere {:?} vs aabb {:?}",
        sphere.map(|h| f(h.t)),
        aabb.map(|h| f(h.t))
    );
}

/// Front-side AABB: t = distance of the lowest vertex / closing speed.
#[test]
fn aabb_plane_toi_front_side_closed_form() {
    let b = AABB::new(v3(-1.0, 4.0, -1.0), v3(1.0, 6.0, 1.0));
    let h = aabb_plane_toi(&b, v3(0.0, -8.0, 0.0), v3(0.0, 1.0, 0.0), Fix128::ONE).unwrap();
    assert!((f(h.t) - 3.0 / 8.0).abs() < 1e-9); // (4 - 1) / 8
    assert!((f(h.point.y) - 1.0).abs() < 1e-9);
}

/// Conservative advancement on a plane whose approach is shallow (mostly sideways)
/// takes ~log(gap/tol)/log(1/(1-cos)) iterations; the 32-iteration cap runs out
/// before convergence and the function answers `None` ("no collision") although the
/// sphere does hit. Exact TOI from sphere_plane_toi for the same motion is 6/7.
#[test]
#[ignore = "known defect: AUD-A-S2W2-013: conservative_advancement returns None after max_iterations without converging (it advances by gap/|v| not gap/(v.n)), so a sphere closing on a plane at a shallow angle (v = (100,-10.5,0), gap 9) is reported as no collision although it hits at t = 0.857 (exact sphere_plane_toi)"]
fn conservative_advancement_does_not_give_up_on_a_shallow_approach() {
    let start = v3(0.0, 10.0, 0.0);
    let disp = v3(100.0, -10.5, 0.0);
    let exact = sphere_plane_toi(start, fx(1.0), disp, v3(0.0, 1.0, 0.0), Fix128::ZERO).unwrap();
    assert!((f(exact.t) - 9.0 / 10.5).abs() < 1e-9);
    let got = conservative_advancement(
        start,
        disp,
        fx(1.0),
        |p| (p.y, Vec3Fix::UNIT_Y),
        &CcdConfig::default(),
    );
    let got = got.expect("conservative advancement missed a collision");
    assert!((f(got.t) - f(exact.t)).abs() < 0.01, "t = {}", f(got.t));
}

/// Head-on approach converges within the tolerance of the exact answer from below.
#[test]
fn conservative_advancement_head_on_stops_within_tolerance_before_contact() {
    let cfg = CcdConfig::default();
    let got = conservative_advancement(
        v3(0.0, 10.0, 0.0),
        v3(0.0, -10.0, 0.0),
        fx(1.0),
        |p| (p.y, Vec3Fix::UNIT_Y),
        &cfg,
    )
    .unwrap();
    // exact contact t = 0.9; the advancement may stop up to tolerance/speed earlier
    assert!(
        f(got.t) <= 0.9 + 1e-9 && f(got.t) >= 0.9 - 1e-3 / 10.0 - 1e-9,
        "t = {}",
        f(got.t)
    );
}

/// The doc says no extra sub-stepping beyond what keeps a body from advancing
/// more than half the smaller radius per sub-step. A body moving along a space
/// diagonal travels sqrt(3) times its largest component, but the count is
/// derived from the largest component (L-infinity norm) only.
#[test]
#[ignore = "known defect: AUD-A-S2W2-014: adaptive_toi_substeps sizes sub-steps from the L-infinity speed, so a body moving along (1,1,1) advances sqrt(3) x the documented cap per sub-step (v=(6,6,6) r=0.1 dt=1/60: 2 sub-steps give 0.0866 per step vs cap 0.05, 4 are needed)"]
fn adaptive_toi_substeps_bounds_euclidean_travel_per_substep() {
    let dt = Fix128::from_ratio(1, 60);
    let n = adaptive_toi_substeps(
        v3(0.0, 0.0, 0.0),
        v3(6.0, 6.0, 6.0),
        fx(0.1),
        v3(0.2, 0.2, 0.2),
        Vec3Fix::ZERO,
        fx(0.1),
        dt,
        100,
    );
    let travel = 6.0 * 3.0f64.sqrt() / 60.0;
    let per_substep = travel / n as f64;
    assert!(
        per_substep <= 0.05 + 1e-9,
        "{n} sub-steps give {per_substep} per step (cap 0.05)"
    );
}

/// More speed never needs fewer sub-steps, and the count always lies in [1, max].
#[test]
fn adaptive_toi_substeps_is_monotone_in_speed_and_clamped() {
    let dt = Fix128::from_ratio(1, 60);
    let mut prev = 0u32;
    for s in [10, 20, 40, 80, 160, 320] {
        let n = adaptive_toi_substeps(
            v3(0.0, 0.0, 0.0),
            v3(s as f64, 0.0, 0.0),
            fx(0.1),
            v3(0.5, 0.0, 0.0),
            Vec3Fix::ZERO,
            fx(0.1),
            dt,
            16,
        );
        assert!((1..=16).contains(&n), "n {n}");
        assert!(n >= prev, "speed {s}: {n} < {prev}");
        prev = n;
    }
    assert_eq!(prev, 16, "the fastest case must saturate at max_substeps");
}

/// Touching at the start counts as overlapping: t = 0 whatever the velocity
/// (pins the `c <= 0` and `gap <= 0` comparisons at exact equality).
#[test]
fn touching_pairs_report_an_immediate_contact_even_when_separating() {
    let t = sphere_sphere_toi(
        v3(0.0, 0.0, 0.0),
        fx(1.0),
        v3(-5.0, 0.0, 0.0),
        v3(2.0, 0.0, 0.0),
        fx(1.0),
        v3(5.0, 0.0, 0.0),
    );
    assert_eq!(t.map(|h| f(h.t)), Some(0.0));
    let c = speculative_contact(
        v3(0.0, 0.0, 0.0),
        v3(-5.0, 0.0, 0.0),
        fx(1.0),
        v3(2.0, 0.0, 0.0),
        v3(5.0, 0.0, 0.0),
        fx(1.0),
        Fix128::ONE,
    )
    .expect("touching spheres are in contact");
    assert_eq!(f(c.depth), 0.0);
}

/// Resting contact with a plane (|dist| == r, no normal velocity) is an impact at
/// t = 0; a sphere resting *behind* the plane out of reach with zero normal
/// velocity never hits; a resting box on the plane is an impact at t = 0.
#[test]
fn resting_contact_with_a_plane_is_an_immediate_impact() {
    let n = v3(0.0, 1.0, 0.0);
    let tangential = v3(3.0, 0.0, 0.0);
    let h = sphere_plane_toi(v3(0.0, 1.0, 0.0), fx(1.0), tangential, n, Fix128::ZERO);
    assert_eq!(h.map(|x| f(x.t)), Some(0.0));
    assert!(sphere_plane_toi(v3(0.0, -5.0, 0.0), fx(1.0), tangential, n, Fix128::ZERO).is_none());
    assert!(sphere_plane_toi(v3(0.0, 5.0, 0.0), fx(1.0), tangential, n, Fix128::ZERO).is_none());
    let b = AABB::new(v3(-1.0, 0.0, -1.0), v3(1.0, 2.0, 1.0));
    let h = aabb_plane_toi(&b, tangential, n, Fix128::ZERO);
    assert_eq!(h.map(|x| f(x.t)), Some(0.0));
}

/// Exact sub-step counts when the CCD cap (half the smaller radius) dominates and
/// nothing clamps: n = ceil(L-inf travel / cap). Negative components count by
/// magnitude, and the smaller of two unequal radii sets the cap.
#[test]
fn adaptive_toi_substeps_counts_ceil_travel_over_half_the_smaller_radius() {
    let dt = Fix128::from_ratio(1, 60);
    let n = |va: Vec3Fix, ra: f64, rb: f64| {
        adaptive_toi_substeps(
            v3(0.0, 0.0, 0.0),
            va,
            fx(ra),
            v3(0.05, 0.0, 0.0),
            Vec3Fix::ZERO,
            fx(rb),
            dt,
            1000,
        )
    };
    // travel 6/60 = 0.1, cap 0.05 -> 2 ; travel 0.15 -> 3
    assert_eq!(n(v3(6.0, 0.0, 0.0), 0.1, 0.1), 2);
    assert_eq!(n(v3(9.0, 0.0, 0.0), 0.1, 0.1), 3);
    assert_eq!(n(v3(-6.0, 0.0, 0.0), 0.1, 0.1), 2);
    assert_eq!(n(v3(0.0, -9.0, 0.0), 0.1, 0.1), 3);
    assert_eq!(n(v3(0.0, 0.0, -9.0), 0.1, 0.1), 3);
    // unequal radii: the smaller (0.1) sets the cap whichever body carries it
    assert_eq!(n(v3(6.0, 0.0, 0.0), 0.1, 1.0), 2);
    assert_eq!(n(v3(6.0, 0.0, 0.0), 1.0, 0.1), 2);
}

/// Boxes that touch at t = 0 (face contact) report t = 0 both when closing in and
/// when moving apart (the overlap window is the single instant t = 0).
#[test]
fn swept_aabb_face_contact_at_the_start_is_an_immediate_hit() {
    let moving = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let toward = AABB::new(v3(1.0, 0.0, 0.0), v3(2.0, 1.0, 1.0));
    let apart = AABB::new(v3(-1.0, 0.0, 0.0), v3(0.0, 1.0, 1.0));
    let vel = v3(1.0, 0.0, 0.0);
    assert_eq!(swept_aabb(&moving, vel, &toward).map(f), Some(0.0));
    assert_eq!(swept_aabb(&moving, vel, &apart).map(f), Some(0.0));
}
