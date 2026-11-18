//! Contact between a resident cloth ([`ClothParticipant`]) and the world's bodies.
//!
//! A body is a sphere of the collision radius the world uses for it; the cloth keeps
//! its faces at least `R = r + thickness` from the sphere's centre. The contact is
//! solved inside the cloth's iterations, both sides moved by their inverse masses,
//! and the body receives what the cloth moved it by as a force. Oracles:
//!
//! - a ball dropped on a pinned sheet comes to rest on it: over the last second no
//!   face is inside `R` by more than 1 % and the ball does not drift (steady
//!   penetration 0, rest without drift);
//! - a ball smaller than the grid's cells is held by the faces, not only the particles;
//! - a ball at the stated coupling limit (`v · W = 2 R`, `W` the width of one cloth
//!   step) is caught whatever its start phase;
//! - momentum balance: in zero gravity with no pins and no damping, the ball's and the
//!   cloth's momentum together do not change while they touch, and the ball loses a
//!   real part of its own (the exchange is not vacuous).

use alice_physics_core::cloth::{Cloth, ClothConfig};
use alice_physics_core::cloth_participant::ClothParticipant;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

const RES: usize = 11;
const PARTICLE_MASS: (i64, i64) = (1, 100);
const BALL_MASS: (i64, i64) = (1, 10);
const BALL_RADIUS: (i64, i64) = (8, 100);
const THICKNESS: (i64, i64) = (1, 100);

fn reach_of(radius: (i64, i64)) -> f64 {
    radius.0 as f64 / radius.1 as f64 + THICKNESS.0 as f64 / THICKNESS.1 as f64
}

fn gravity() -> Vec3Fix {
    Vec3Fix::new(Fix128::ZERO, r(-981, 100), Fix128::ZERO)
}

/// A 1 m sheet in the `x–z` plane at `y = 0`, centred on the origin, `RES × RES`
/// particles of 10 g (cells of 0.1 m), stepping `m` times per frame. `pinned` pins
/// its boundary; `damping` is the cloth's retention per 1/60 s.
fn sheet(m: usize, g: Vec3Fix, pinned: bool, damping: Fix128) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::new(r(-1, 2), Fix128::ZERO, r(-1, 2)),
        Fix128::ONE,
        Fix128::ONE,
        RES,
        RES,
        r(PARTICLE_MASS.0, PARTICLE_MASS.1),
    );
    c.config = ClothConfig {
        iterations: 4,
        substeps: m,
        gravity: g,
        damping,
        thickness: r(THICKNESS.0, THICKNESS.1),
        ..c.config
    };
    if pinned {
        for i in 0..RES {
            for j in 0..RES {
                if i == 0 || j == 0 || i == RES - 1 || j == RES - 1 {
                    c.pin(i * RES + j);
                }
            }
        }
    }
    c.set_sleep(false);
    c
}

struct Scene {
    world: PhysicsWorld,
    ball: usize,
}

/// A world of `n` substeps holding the sheet (stepping `m` times a frame) and a ball
/// of `BALL_MASS` and `radius` at `(x, y0, z)` moving at `vy`.
#[allow(clippy::too_many_arguments)]
fn scene(
    n: usize,
    m: usize,
    at: (Fix128, Fix128, Fix128),
    vy: Fix128,
    radius: (i64, i64),
    g: Vec3Fix,
    cloth: Cloth,
) -> Scene {
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: g,
        substeps: n,
        ..SolverConfig::default()
    });
    let mut body =
        RigidBody::new_dynamic(Vec3Fix::new(at.0, at.1, at.2), r(BALL_MASS.0, BALL_MASS.1));
    body.velocity = Vec3Fix::new(Fix128::ZERO, vy, Fix128::ZERO);
    let ball = world.add_body(body);
    world.set_body_collision_radius(ball, r(radius.0, radius.1));
    let _ = m;
    world
        .add_participant(Box::new(ClothParticipant::new(cloth)))
        .expect("register");
    Scene { world, ball }
}

fn fix_at(b: &[u8], at: usize) -> Fix128 {
    let hi = i64::from_le_bytes(b[at..at + 8].try_into().expect("8"));
    let lo = u64::from_le_bytes(b[at + 8..at + 16].try_into().expect("8"));
    Fix128::from_raw(hi, lo)
}

fn vec_at(b: &[u8], at: usize) -> Vec3Fix {
    Vec3Fix::new(fix_at(b, at), fix_at(b, at + 16), fix_at(b, at + 32))
}

/// Per particle `(position, velocity)` read from the resident's payload.
fn particles(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    let b = w.participant_state(0).expect("payload");
    let n = u32::from_le_bytes(b[4..8].try_into().expect("4")) as usize;
    (0..n)
        .map(|i| {
            let at = 16 + i * (3 * 48 + 16);
            (vec_at(&b, at), vec_at(&b, at + 96))
        })
        .collect()
}

/// The closest distance from `c` to any face of the sheet (its triangles).
fn closest_face(w: &PhysicsWorld, c: Vec3Fix) -> f64 {
    let p: Vec<[f64; 3]> = particles(w)
        .iter()
        .map(|(x, _)| [x.x.to_f64(), x.y.to_f64(), x.z.to_f64()])
        .collect();
    let c = [c.x.to_f64(), c.y.to_f64(), c.z.to_f64()];
    let mut best = f64::INFINITY;
    for i in 0..RES - 1 {
        for j in 0..RES - 1 {
            let (a, b, d, e) = (
                i * RES + j,
                i * RES + j + 1,
                (i + 1) * RES + j,
                (i + 1) * RES + j + 1,
            );
            for t in [[a, b, d], [b, e, d]] {
                best = best.min(point_triangle(c, p[t[0]], p[t[1]], p[t[2]]));
            }
        }
    }
    best
}

/// Distance from `p` to triangle `abc` (an independent f64 reference: Ericson §5.1.5).
fn point_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let at = |s: f64, t: f64| {
        let q = [
            a[0] + s * (b[0] - a[0]) + t * (c[0] - a[0]),
            a[1] + s * (b[1] - a[1]) + t * (c[1] - a[1]),
            a[2] + s * (b[2] - a[2]) + t * (c[2] - a[2]),
        ];
        dot(sub(p, q), sub(p, q)).sqrt()
    };
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return at(0.0, 0.0);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return at(1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return at(d1 / (d1 - d3), 0.0);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return at(0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return at(0.0, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return at(1.0 - w, w);
    }
    let denom = 1.0 / (va + vb + vc);
    at(vb * denom, vc * denom)
}

/// The sheet's height under the ball: the mean height of the particles within one
/// cell of the ball's `x–z` position.
fn sheet_height_under(w: &PhysicsWorld, c: Vec3Fix) -> f64 {
    let (cx, cz) = (c.x.to_f64(), c.z.to_f64());
    let near: Vec<f64> = particles(w)
        .iter()
        .filter(|(x, _)| (x.x.to_f64() - cx).abs() <= 0.1 && (x.z.to_f64() - cz).abs() <= 0.1)
        .map(|(x, _)| x.y.to_f64())
        .collect();
    near.iter().sum::<f64>() / near.len() as f64
}

/// A ball dropped on the pinned sheet under gravity comes to rest on it, at `m = 1`
/// and `m = 2` steps per frame and at world substeps that do and do not equal `m`:
/// over the last second of a 5 s run it is above the sheet, no face is inside `R` of
/// its centre by more than 1 % of `R`, its height moves by less than 0.1 mm and its
/// vertical speed stays below 1 mm/s (rest without drift).
#[test]
fn a_dropped_ball_comes_to_rest_on_the_sheet_without_sinking_or_drifting() {
    let reach = reach_of(BALL_RADIUS);
    for (n, m) in [(1usize, 1usize), (2, 2), (4, 2), (3, 1)] {
        let cloth = sheet(m, gravity(), true, r(99, 100));
        let mut s = scene(
            n,
            m,
            (Fix128::ZERO, r(3, 10), Fix128::ZERO),
            Fix128::ZERO,
            BALL_RADIUS,
            gravity(),
            cloth,
        );
        let (mut closest, mut lo, mut hi, mut fastest) =
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
        for frame in 0..300 {
            s.world.step(dt());
            if frame >= 240 {
                let c = s.world.bodies[s.ball].position;
                assert!(
                    c.y.to_f64() > sheet_height_under(&s.world, c),
                    "n {n} m {m} frame {frame}: the ball is above the sheet"
                );
                closest = closest.min(closest_face(&s.world, c));
                lo = lo.min(c.y.to_f64());
                hi = hi.max(c.y.to_f64());
                fastest = fastest.max(s.world.bodies[s.ball].velocity.y.to_f64().abs());
            }
        }
        println!(
            "[cloth contact] n {n} m {m}: closest face {closest:.5} (R {reach:.3}), height range {:.2e} m, |vy| max {fastest:.2e} m/s",
            hi - lo
        );
        assert!(
            closest >= reach * 0.99,
            "n {n} m {m}: penetration {:.5}",
            reach - closest
        );
        assert!(
            hi - lo < 1e-4,
            "n {n} m {m}: the ball drifts {:.2e} m in the last second",
            hi - lo
        );
        assert!(
            fastest < 1e-3,
            "n {n} m {m}: |vy| {fastest:.2e} m/s in the last second"
        );
    }
}

/// A ball whose `R` is below the grid's half-diagonal (`R = 4 cm < 0.1/√2 = 7.07 cm`),
/// dropped over the middle of a cell, is held by the cell's faces: particle-only
/// contact lets it fall between the four particles.
#[test]
fn a_ball_smaller_than_a_cell_is_held_by_its_faces() {
    let small = (3, 100);
    let reach = reach_of(small);
    let cloth = sheet(1, gravity(), true, r(99, 100));
    let mut s = scene(
        1,
        1,
        (r(5, 100), r(3, 10), r(5, 100)),
        Fix128::ZERO,
        small,
        gravity(),
        cloth,
    );
    for _ in 0..180 {
        s.world.step(dt());
    }
    let c = s.world.bodies[s.ball].position;
    assert!(
        c.y.to_f64() > sheet_height_under(&s.world, c),
        "the ball stays above the sheet (y {:.4})",
        c.y.to_f64()
    );
    assert!(
        closest_face(&s.world, c) >= reach * 0.99,
        "no face inside R"
    );
}

/// The stated coupling limit: a ball crossing the sheet's plane at `v` with
/// `v · W = 2 R` (`W = dt / m`, one cloth step) is caught at every start phase, at
/// `m = 1` and `m = 2`. (The cloth reads each body's centre at the start of a step and
/// predicts it over the step; the side a body starts a step on is the side it is kept
/// on.)
#[test]
fn a_ball_at_the_stated_coupling_limit_is_caught() {
    let reach = reach_of(BALL_RADIUS);
    for m in [1usize, 2] {
        let v = -2.0 * reach * 60.0 * m as f64;
        for y0 in [(50, 100), (47, 100), (44, 100), (41, 100)] {
            let cloth = sheet(m, Vec3Fix::ZERO, true, r(99, 100));
            let mut s = scene(
                m,
                m,
                (Fix128::ZERO, r(y0.0, y0.1), Fix128::ZERO),
                Fix128::from_f64(v),
                BALL_RADIUS,
                Vec3Fix::ZERO,
                cloth,
            );
            for _ in 0..90 {
                s.world.step(dt());
            }
            let c = s.world.bodies[s.ball].position;
            assert!(
                c.y.to_f64() > sheet_height_under(&s.world, c),
                "m {m} y0 {y0:?}: caught at the limit (ball y {:.3})",
                c.y.to_f64()
            );
        }
    }
}

/// Momentum balance: no gravity, no pins, no damping; a ball thrown down at a free
/// sheet at 2 m/s. Every frame the total momentum (ball + particles) equals the
/// initial `M · v` to 1e-12 kg·m/s, and by the end the ball has handed at least 10 %
/// of its momentum to the sheet.
#[test]
fn ball_and_cloth_momentum_together_is_conserved_through_the_contact() {
    let ball_mass = BALL_MASS.0 as f64 / BALL_MASS.1 as f64;
    let particle_mass = PARTICLE_MASS.0 as f64 / PARTICLE_MASS.1 as f64;
    for (n, m) in [(1usize, 1usize), (4, 2)] {
        let cloth = sheet(m, Vec3Fix::ZERO, false, Fix128::ONE);
        let mut s = scene(
            n,
            m,
            (Fix128::ZERO, r(3, 10), Fix128::ZERO),
            r(-2, 1),
            BALL_RADIUS,
            Vec3Fix::ZERO,
            cloth,
        );
        let p0 = ball_mass * -2.0;
        let mut worst = 0.0f64;
        for _ in 0..60 {
            s.world.step(dt());
            let cloth_py: f64 = particles(&s.world)
                .iter()
                .map(|(_, v)| v.y.to_f64() * particle_mass)
                .sum();
            let ball_py = s.world.bodies[s.ball].velocity.y.to_f64() * ball_mass;
            worst = worst.max((cloth_py + ball_py - p0).abs());
        }
        let ball_py = s.world.bodies[s.ball].velocity.y.to_f64() * ball_mass;
        println!("[cloth contact] n {n} m {m}: momentum error max {worst:.2e}, ball keeps {:.3} of its momentum", ball_py / p0);
        assert!(worst < 1e-12, "n {n} m {m}: momentum error {worst:.2e}");
        assert!(
            ball_py / p0 < 0.9,
            "n {n} m {m}: the ball handed momentum to the sheet ({:.3} kept)",
            ball_py / p0
        );
    }
}
