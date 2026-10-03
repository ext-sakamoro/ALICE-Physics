//! Oracles for `deformable::DeformableBody::{new_cube, center_of_mass, resolve_rigid_body_collisions}`
//! (`examples/deformable_cube_impact.rs`).
//!
//! * `new_cube(c, h, m)`: 8 corners `c +- h`, mass `m / 8` each, 5 tetrahedra of total volume `(2h)^3`
//!   (4 corner tets `(2h)^3 / 6` + 1 central tet `(2h)^3 / 3`), 12 surface triangles of total area `6 (2h)^2`
//!   covering every cube edge/diagonal exactly twice (closed surface)
//! * `center_of_mass` is the mean of the particle positions: `c` for a fresh cube, and it follows the
//!   rigid-body kinematics `y0 + g dt_s^2 n (n+1) / 2` in free fall (no damping, constraints are translation invariant)
//! * `resolve_rigid_body_collisions` for one particle (weight `wp = 1/mp`) and one sphere of weight `wr`:
//!   penetration `p = R - d`; particle moves out `p wp/(wp+wr)`, body moves `-p wr/(wp+wr)` (relative shift = `p`);
//!   an approach speed `u` along the normal becomes `0` (inelastic), conserving momentum; a static body
//!   (`wr = 0`) takes none of it, so the particle moves the full `p` and ends with zero normal speed
#![allow(clippy::disallowed_methods)]

use alice_physics::deformable::DeformableBody;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use std::collections::BTreeMap;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}
fn tup(v: Vec3Fix) -> (f64, f64, f64) {
    (v.x.to_f64(), v.y.to_f64(), v.z.to_f64())
}
fn near3(a: Vec3Fix, want: (f64, f64, f64), tol: f64) {
    let g = tup(a);
    assert!(
        (g.0 - want.0).abs() < tol && (g.1 - want.1).abs() < tol && (g.2 - want.2).abs() < tol,
        "{g:?} vs {want:?}"
    );
}
fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}
fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}
fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[test]
fn cube_has_eight_corners_at_center_plus_minus_h_with_split_mass() {
    let c = (1.0, -2.0, 3.0);
    let h = 0.5;
    let body = DeformableBody::new_cube(v3(c.0, c.1, c.2), f(h), Fix128::from_int(10));
    assert_eq!(body.particle_count(), 8);
    // all 8 sign combinations of (+-h, +-h, +-h) occur exactly once
    let mut seen = std::collections::BTreeSet::new();
    for p in &body.positions {
        let d = sub(tup(*p), c);
        assert!(d.0.abs() == h && d.1.abs() == h && d.2.abs() == h, "{d:?}");
        seen.insert((d.0 > 0.0, d.1 > 0.0, d.2 > 0.0));
    }
    assert_eq!(seen.len(), 8);
    // mass 10 / 8 per particle -> inverse mass 0.8
    for w in &body.inv_masses {
        assert!((w.to_f64() - 0.8).abs() < 1e-12, "{}", w.to_f64());
    }
    // zero mass -> pinned particles (zero inverse mass)
    let pinned = DeformableBody::new_cube(Vec3Fix::ZERO, Fix128::ONE, Fix128::ZERO);
    assert!(pinned.inv_masses.iter().all(|w| w.is_zero()));
}

#[test]
fn cube_tetrahedra_fill_the_cube_exactly() {
    for h in [0.25f64, 1.0, 2.5] {
        let body = DeformableBody::new_cube(v3(0.5, 0.5, 0.5), f(h), Fix128::ONE);
        assert_eq!(body.tetrahedra.len(), 5);
        let side3 = (2.0 * h).powi(3);
        let mut vols: Vec<f64> = body
            .tetrahedra
            .iter()
            .map(|t| {
                let p: Vec<_> = t.iter().map(|&i| tup(body.positions[i])).collect();
                (dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0]))) / 6.0).abs()
            })
            .collect();
        vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total: f64 = vols.iter().sum();
        assert!((total - side3).abs() < 1e-9 * side3, "h={h}: total {total}");
        for v in &vols[..4] {
            assert!((v - side3 / 6.0).abs() < 1e-9 * side3, "corner tet {v}");
        }
        assert!(
            (vols[4] - side3 / 3.0).abs() < 1e-9 * side3,
            "central tet {}",
            vols[4]
        );
    }
}

#[test]
fn cube_surface_is_a_closed_triangulation_of_area_six_sides_squared() {
    let h = 0.75;
    let body = DeformableBody::new_cube(Vec3Fix::ZERO, f(h), Fix128::ONE);
    assert_eq!(body.surface_triangles.len(), 12);
    let mut area = 0.0;
    let mut edges: BTreeMap<(usize, usize), (u32, u32)> = BTreeMap::new();
    let mut volume6 = 0.0; // sum p0 . (p1 x p2) = 6 * signed enclosed volume
    for tri in &body.surface_triangles {
        let p: Vec<_> = tri.iter().map(|&i| tup(body.positions[i])).collect();
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        area += 0.5 * dot(n, n).sqrt();
        volume6 += dot(p[0], cross(p[1], p[2]));
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let e = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
            if a < b {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
    let side = 2.0 * h;
    assert!((area - 6.0 * side * side).abs() < 1e-9, "area {area}");
    // closed, consistently oriented surface: every edge is used once in each direction
    assert_eq!(edges.len(), 18, "12 cube edges + 6 face diagonals");
    assert!(
        edges.values().all(|&(fwd, back)| fwd == 1 && back == 1),
        "{edges:?}"
    );
    // the enclosed volume has magnitude (2h)^3
    assert!(
        (volume6.abs() / 6.0 - side.powi(3)).abs() < 1e-9,
        "volume {}",
        volume6 / 6.0
    );
    // every triangle uses three distinct corners of the same cube face (one coordinate shared)
    for tri in &body.surface_triangles {
        let p: Vec<_> = tri.iter().map(|&i| tup(body.positions[i])).collect();
        let shared = (p[0].0 == p[1].0 && p[1].0 == p[2].0)
            || (p[0].1 == p[1].1 && p[1].1 == p[2].1)
            || (p[0].2 == p[1].2 && p[1].2 == p[2].2);
        assert!(shared, "triangle {tri:?} is not on a cube face");
    }
}

#[test]
fn center_of_mass_is_the_position_mean() {
    let c = (3.0, 4.0, -5.0);
    let mut body = DeformableBody::new_cube(v3(c.0, c.1, c.2), f(1.0), Fix128::from_int(8));
    near3(body.center_of_mass(), c, 1e-12);
    // move one particle: the mean shifts by delta / 8
    body.positions[2] = body.positions[2] + v3(8.0, -16.0, 24.0);
    near3(
        body.center_of_mass(),
        (c.0 + 1.0, c.1 - 2.0, c.2 + 3.0),
        1e-12,
    );
    // an arbitrary cloud: mean of the coordinates
    let pts = vec![
        v3(0.0, 0.0, 0.0),
        v3(4.0, 0.0, 0.0),
        v3(0.0, 6.0, 0.0),
        v3(0.0, 0.0, 8.0),
        v3(-5.0, 2.0, 1.0),
    ];
    let cloud = DeformableBody::new(pts, &[], Vec::new(), Fix128::ONE);
    near3(cloud.center_of_mass(), (-0.2, 1.6, 1.8), 1e-12);
    // empty body: zero
    let empty = DeformableBody::new(Vec::new(), &[], Vec::new(), Fix128::ONE);
    assert_eq!(empty.center_of_mass(), Vec3Fix::ZERO);
}

#[test]
fn free_falling_cube_follows_the_kinematic_center_of_mass() {
    let mut body = DeformableBody::new_cube(v3(0.0, 10.0, 0.0), f(0.5), Fix128::from_int(4));
    body.config.damping = Fix128::ONE;
    body.config.gravity = v3(0.0, -10.0, 0.0);
    let substeps = body.config.substeps as f64;
    let dt = 1.0 / 60.0;
    let hs = dt / substeps;
    let frames = 6;
    for _ in 0..frames {
        body.step(Fix128::from_ratio(1, 60));
    }
    // symplectic Euler: y_n = y0 + g hs^2 n (n + 1) / 2 with n = frames * substeps
    let n = frames as f64 * substeps;
    let want_y = 10.0 - 10.0 * hs * hs * n * (n + 1.0) / 2.0;
    let com = tup(body.center_of_mass());
    assert!((com.1 - want_y).abs() < 1e-9, "y {} vs {want_y}", com.1);
    assert!(com.0.abs() < 1e-12 && com.2.abs() < 1e-12);
    // the cube stays rigid: all pairwise corner distances (side 1 or sqrt2 or sqrt3) are preserved
    let c = body.positions.clone();
    for i in 0..8 {
        for j in (i + 1)..8 {
            let d = tup(c[i] - c[j]);
            let dist = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
            assert!(
                [1.0, 2f64.sqrt(), 3f64.sqrt()]
                    .iter()
                    .any(|s| (dist - s).abs() < 1e-6),
                "pair {i},{j} distance {dist}"
            );
        }
    }
}

fn single_particle(pos: (f64, f64, f64), vel: (f64, f64, f64), mass: i64) -> DeformableBody {
    let mut b = DeformableBody::new(
        vec![v3(pos.0, pos.1, pos.2)],
        &[],
        Vec::new(),
        Fix128::from_int(mass),
    );
    b.velocities[0] = v3(vel.0, vel.1, vel.2);
    b
}

#[test]
fn dynamic_sphere_shares_the_correction_by_inverse_mass_and_the_collision_is_inelastic() {
    let dt = Fix128::from_ratio(1, 60);
    for (mp, mr) in [(1i64, 1i64), (1, 3), (4, 1), (2, 6)] {
        let mut p = single_particle((0.5, 0.0, 0.0), (-2.0, 0.0, 0.0), mp);
        let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(mr))];
        p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
        let (mp, mr) = (mp as f64, mr as f64);
        let m = mp + mr;
        let pen = 0.5;
        // positions: particle out by pen * mr / m, body in by pen * mp / m; relative shift = pen
        near3(p.positions[0], (0.5 + pen * mr / m, 0.0, 0.0), 1e-9);
        near3(rb[0].position, (-pen * mp / m, 0.0, 0.0), 1e-9);
        let gap = p.positions[0].x.to_f64() - rb[0].position.x.to_f64();
        assert!(
            (gap - 1.0).abs() < 1e-9,
            "particle ends on the sphere surface, gap {gap}"
        );
        // velocities: common normal velocity -2 mp / m, momentum conserved
        let vp = p.velocities[0].x.to_f64();
        let vr = rb[0].velocity.x.to_f64();
        assert!((vp - vr).abs() < 1e-9, "inelastic: {vp} vs {vr}");
        assert!((vp - (-2.0 * mp / m)).abs() < 1e-9);
        assert!((mp * vp + mr * vr - mp * -2.0).abs() < 1e-9, "momentum");
        // tangential components untouched (none here)
        assert!(p.velocities[0].y.is_zero() && rb[0].velocity.y.is_zero());
    }
}

#[test]
fn static_body_is_exact_the_particle_takes_the_whole_correction() {
    let dt = Fix128::from_ratio(1, 60);
    for mp in [1i64, 7, 1000] {
        let mut p = single_particle((0.0, 0.5, 0.0), (0.0, -3.0, 0.0), mp);
        let mut wall = [RigidBody::new_static(Vec3Fix::ZERO)];
        p.resolve_rigid_body_collisions(&mut wall, &[Fix128::ONE], dt);
        // exactly on the surface (pen 0.5), no residual penetration, zero approach speed
        assert!(
            (p.positions[0].y.to_f64() - 1.0).abs() < 1e-12,
            "mp={mp}: y = {}",
            p.positions[0].y.to_f64()
        );
        assert!(
            p.velocities[0].y.to_f64().abs() < 1e-12,
            "mp={mp}: vy = {}",
            p.velocities[0].y.to_f64()
        );
        assert_eq!(wall[0].position, Vec3Fix::ZERO);
        assert_eq!(wall[0].velocity, Vec3Fix::ZERO);
    }
}

#[test]
fn separating_overlapping_and_tangential_cases() {
    let dt = Fix128::from_ratio(1, 60);
    // moving away: position still corrected, velocity untouched
    let mut p = single_particle((0.5, 0.0, 0.0), (2.0, 0.0, 0.0), 1);
    let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert!(p.positions[0].x.to_f64() > 0.5);
    assert_eq!(p.velocities[0], v3(2.0, 0.0, 0.0));
    assert_eq!(rb[0].velocity, Vec3Fix::ZERO);
    // tangential speed is not changed (only the normal component is removed)
    let mut p = single_particle((0.5, 0.0, 0.0), (-1.0, 4.0, -6.0), 2);
    let mut wall = [RigidBody::new_static(Vec3Fix::ZERO)];
    p.resolve_rigid_body_collisions(&mut wall, &[Fix128::ONE], dt);
    near3(p.velocities[0], (0.0, 4.0, -6.0), 1e-9);
    // no overlap: nothing happens
    let mut p = single_particle((1.5, 0.0, 0.0), (-2.0, 0.0, 0.0), 1);
    let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert_eq!(
        (p.positions[0], p.velocities[0], rb[0].position),
        (v3(1.5, 0.0, 0.0), v3(-2.0, 0.0, 0.0), Vec3Fix::ZERO)
    );
    // exactly on the surface (distance == radius): not penetrating
    let mut p = single_particle((1.0, 0.0, 0.0), (-2.0, 0.0, 0.0), 1);
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert_eq!(p.positions[0], v3(1.0, 0.0, 0.0));
    // centre of the sphere: no direction, untouched
    let mut p = single_particle((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), 1);
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert_eq!(p.positions[0], Vec3Fix::ZERO);
}

#[test]
fn radius_table_dt_and_pinned_particles() {
    let dt = Fix128::from_ratio(1, 60);
    // per-body radius: radius 2 -> pen 1.5 at distance 0.5, against a static wall particle ends at x = 2
    let mut p = single_particle((0.5, 0.0, 0.0), Vec3Fix::ZERO.x.to_f64().into_triple(), 1);
    let mut wall = [RigidBody::new_static(Vec3Fix::ZERO)];
    p.resolve_rigid_body_collisions(&mut wall, &[Fix128::from_int(2)], dt);
    assert!((p.positions[0].x.to_f64() - 2.0).abs() < 1e-12);
    // a dynamic body missing from the radius table defaults to radius 1; a static one is skipped
    let mut p = single_particle((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), 1);
    let mut dynamic = [RigidBody::new_dynamic(
        Vec3Fix::ZERO,
        Fix128::from_int(1_000_000_000),
    )];
    p.resolve_rigid_body_collisions(&mut dynamic, &[], dt);
    assert!(
        (p.positions[0].x.to_f64() - 1.0).abs() < 1e-6,
        "{}",
        p.positions[0].x.to_f64()
    );
    let mut p = single_particle((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), 1);
    let mut wall = [RigidBody::new_static(Vec3Fix::ZERO)];
    p.resolve_rigid_body_collisions(&mut wall, &[], dt);
    assert_eq!(
        p.positions[0],
        v3(0.5, 0.0, 0.0),
        "static body without a radius is skipped"
    );
    // dt == 0 is a no-op; any other dt gives the same result
    let mut a = single_particle((0.5, 0.0, 0.0), (-2.0, 0.0, 0.0), 1);
    let mut b = a_clone(&a);
    let mut r1 = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut r2 = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    a.resolve_rigid_body_collisions(&mut r1, &[Fix128::ONE], Fix128::ZERO);
    assert_eq!(
        (a.positions[0], r1[0].position),
        (v3(0.5, 0.0, 0.0), Vec3Fix::ZERO)
    );
    a.resolve_rigid_body_collisions(&mut r1, &[Fix128::ONE], dt);
    b.resolve_rigid_body_collisions(&mut r2, &[Fix128::ONE], Fix128::from_int(5));
    assert_eq!(
        (a.positions[0], a.velocities[0], r1[0].position),
        (b.positions[0], b.velocities[0], r2[0].position)
    );
    // pinned particles (zero inverse mass) are skipped
    let mut pin = DeformableBody::new(vec![v3(0.5, 0.0, 0.0)], &[], Vec::new(), Fix128::ZERO);
    let mut r = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    pin.resolve_rigid_body_collisions(&mut r, &[Fix128::ONE], dt);
    assert_eq!(
        (pin.positions[0], r[0].position),
        (v3(0.5, 0.0, 0.0), Vec3Fix::ZERO)
    );
}

fn a_clone(a: &DeformableBody) -> DeformableBody {
    let mut b = DeformableBody::new(a.positions.clone(), &[], Vec::new(), Fix128::ONE);
    b.velocities = a.velocities.clone();
    b
}

trait IntoTriple {
    fn into_triple(self) -> (f64, f64, f64);
}
impl IntoTriple for f64 {
    fn into_triple(self) -> (f64, f64, f64) {
        (self, 0.0, 0.0)
    }
}

#[test]
fn several_particles_accumulate_on_a_dynamic_body() {
    // two particles on opposite sides are resolved one after the other (Gauss-Seidel): the second sees the
    // body already moved by the first, so the body ends at +0.125 rather than 0; what holds exactly is momentum
    let dt = Fix128::from_ratio(1, 60);
    let mut b = DeformableBody::new(
        vec![v3(0.5, 0.0, 0.0), v3(-0.5, 0.0, 0.0)],
        &[],
        Vec::new(),
        Fix128::ONE,
    );
    let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    b.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    // hand sequence: p0 at 0.5 -> 0.75, body -> -0.25; p1 at -0.5 (distance 0.25, pen 0.75) -> -0.25 - 0.375 ... body -> +0.125
    near3(rb[0].position, (0.125, 0.0, 0.0), 1e-9);
    let total_p = b.velocities[0] + b.velocities[1] + rb[0].velocity;
    near3(total_p, (0.0, 0.0, 0.0), 1e-9);
}

fn det3(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> f64 {
    dot(a, cross(b, c))
}

#[test]
fn the_five_tetrahedra_tile_the_cube_without_overlap() {
    let body = DeformableBody::new_cube(v3(1.0, 2.0, 3.0), f(0.5), Fix128::ONE);
    let pts: Vec<_> = body.positions.iter().map(|p| tup(*p)).collect();
    let xs = [-0.87, -0.43, 0.11, 0.59];
    let ys = [-0.77, -0.29, 0.23, 0.71];
    let zs = [-0.91, -0.37, 0.17, 0.67];
    let mut checked = 0;
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                let q = (1.0 + 0.5 * x, 2.0 + 0.5 * y, 3.0 + 0.5 * z);
                let mut hit = 0;
                for t in &body.tetrahedra {
                    let (a, b, c, d) = (pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
                    let v = det3(sub(b, a), sub(c, a), sub(d, a));
                    // barycentric coordinates by Cramer's rule
                    let l1 = det3(sub(q, a), sub(c, a), sub(d, a)) / v;
                    let l2 = det3(sub(b, a), sub(q, a), sub(d, a)) / v;
                    let l3 = det3(sub(b, a), sub(c, a), sub(q, a)) / v;
                    let l0 = 1.0 - l1 - l2 - l3;
                    if l0 > 0.0 && l1 > 0.0 && l2 > 0.0 && l3 > 0.0 {
                        hit += 1;
                    }
                }
                assert_eq!(hit, 1, "point {q:?} lies in {hit} tetrahedra");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 64);
}

#[test]
fn contact_exactly_on_the_surface_changes_nothing_and_moving_bodies_use_relative_velocity() {
    let dt = Fix128::from_ratio(1, 60);
    // distance == radius: no penetration, so the approach velocity is untouched too
    let mut p = single_particle((1.0, 0.0, 0.0), (-2.0, 0.0, 0.0), 1);
    let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert_eq!(
        (p.positions[0], p.velocities[0], rb[0].velocity),
        (v3(1.0, 0.0, 0.0), v3(-2.0, 0.0, 0.0), Vec3Fix::ZERO)
    );
    // body moving at +1 toward... away from: particle -2, body +1 along the normal: approach speed 3,
    // both end at the momentum-weighted common velocity (m_p v_p + m_r v_r) / (m_p + m_r)
    for (mp, mr) in [(1i64, 1i64), (1, 3), (5, 2)] {
        let mut p = single_particle((0.5, 0.0, 0.0), (-2.0, 0.0, 0.0), mp);
        let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(mr))];
        rb[0].velocity = v3(1.0, 0.0, 0.0);
        p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
        let (mp, mr) = (mp as f64, mr as f64);
        let common = (mp * -2.0 + mr * 1.0) / (mp + mr);
        assert!(
            (p.velocities[0].x.to_f64() - common).abs() < 1e-9,
            "{} vs {common}",
            p.velocities[0].x.to_f64()
        );
        assert!((rb[0].velocity.x.to_f64() - common).abs() < 1e-9);
    }
    // a body already moving away faster than the particle approaches: relative velocity is separating, no impulse
    let mut p = single_particle((0.5, 0.0, 0.0), (-1.0, 0.0, 0.0), 1);
    let mut rb = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    rb[0].velocity = v3(-5.0, 0.0, 0.0);
    p.resolve_rigid_body_collisions(&mut rb, &[Fix128::ONE], dt);
    assert_eq!(p.velocities[0], v3(-1.0, 0.0, 0.0));
    assert_eq!(rb[0].velocity, v3(-5.0, 0.0, 0.0));
}
