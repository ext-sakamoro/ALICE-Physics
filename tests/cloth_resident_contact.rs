//! Contact between a resident cloth ([`ClothParticipant`]) and the world's bodies.
//!
//! The bodies are spheres of their collision radius `r`; a particle is kept at least
//! `R = r + thickness` from a body's centre. Oracles: the resting contact leaves no
//! particle inside `R` (steady penetration 0), and the coupling limit of the MODEL
//! section: the cloth reads the bodies at the start of each of its `m` steps per
//! frame, so a body crossing at `v` is always caught while `v · dt / m < 2 R`, and
//! can pass through above it when its sampled positions straddle the band.

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
const BALL_RADIUS: (i64, i64) = (5, 100);
const THICKNESS: (i64, i64) = (1, 100);

/// `R = r + thickness` in metres.
fn reach() -> f64 {
    BALL_RADIUS.0 as f64 / BALL_RADIUS.1 as f64 + THICKNESS.0 as f64 / THICKNESS.1 as f64
}

/// A 1 m sheet in the `x–z` plane at `y = 0`, centred on the origin, `RES × RES`
/// particles of 10 g, its boundary pinned, stepping `m` times per frame.
fn sheet(m: usize, gravity: Vec3Fix) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::new(r(-1, 2), Fix128::ZERO, r(-1, 2)),
        Fix128::ONE,
        Fix128::ONE,
        RES,
        RES,
        r(1, 100),
    );
    c.config = ClothConfig {
        iterations: 4,
        substeps: m,
        gravity,
        damping: r(99, 100),
        thickness: r(THICKNESS.0, THICKNESS.1),
        ..c.config
    };
    for i in 0..RES {
        for j in 0..RES {
            if i == 0 || j == 0 || i == RES - 1 || j == RES - 1 {
                c.pin(i * RES + j);
            }
        }
    }
    c
}

/// A world of `m` substeps (so the cloth reads the bodies at every substep start)
/// holding the sheet and a 10 g ball of radius 5 cm at height `y0` moving at `vy`.
fn scene(m: usize, y0: Fix128, vy: Fix128, gravity: Vec3Fix) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity,
        substeps: m,
        ..SolverConfig::default()
    });
    let mut ball = RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, y0, Fix128::ZERO), r(1, 100));
    ball.velocity = Vec3Fix::new(Fix128::ZERO, vy, Fix128::ZERO);
    let b = w.add_body(ball);
    w.set_body_collision_radius(b, r(BALL_RADIUS.0, BALL_RADIUS.1));
    w.add_participant(Box::new(ClothParticipant::new(sheet(m, gravity))))
        .expect("register");
    (w, b)
}

fn fix_at(b: &[u8], at: usize) -> Fix128 {
    let hi = i64::from_le_bytes(b[at..at + 8].try_into().expect("8"));
    let lo = u64::from_le_bytes(b[at + 8..at + 16].try_into().expect("8"));
    Fix128::from_raw(hi, lo)
}

fn positions(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    let b = w.participant_state(0).expect("payload");
    let n = u32::from_le_bytes(b[4..8].try_into().expect("4")) as usize;
    (0..n)
        .map(|i| {
            let at = 16 + i * (3 * 48 + 16);
            Vec3Fix::new(fix_at(&b, at), fix_at(&b, at + 16), fix_at(&b, at + 32))
        })
        .collect()
}

/// Which side of the sheet the ball is on: above the sheet's centre particle.
fn ball_above_centre(w: &PhysicsWorld, b: usize) -> bool {
    let centre = positions(w)[(RES / 2) * RES + RES / 2];
    w.bodies[b].position.y > centre.y
}

/// A ball dropped on the pinned sheet under gravity comes to rest on it: after 5 s
/// it is still above the sheet, nearly still, and no particle is inside `R` of its
/// centre by more than 1 % of `R` (steady penetration 0).
#[test]
fn a_dropped_ball_rests_on_the_sheet_without_sinking_into_it() {
    let g = Vec3Fix::new(Fix128::ZERO, r(-981, 100), Fix128::ZERO);
    for m in [1usize, 2] {
        let (mut w, b) = scene(m, r(3, 10), Fix128::ZERO, g);
        for _ in 0..300 {
            w.step(dt());
        }
        assert!(ball_above_centre(&w, b), "m {m}: the ball stayed above the sheet");
        let c = w.bodies[b].position;
        let closest = positions(&w)
            .iter()
            .map(|p| (*p - c).length().to_f64())
            .fold(f64::INFINITY, f64::min);
        println!("[cloth contact] m {m}: ball y {:.4}, closest particle {closest:.5} (R {:.3})", c.y.to_f64(), reach());
        assert!(closest >= reach() * 0.99, "m {m}: penetration {}", reach() - closest);
        assert!(w.bodies[b].velocity.length().to_f64() < 0.05, "m {m}: the ball came to rest");
    }
}

/// Below the coupling limit the ball is caught whatever its start phase; above it,
/// a start whose sampled heights straddle the band `|y| < R` passes through, and
/// doubling `m` doubles the limit: the same speed passes at `m = 1` and is caught at
/// `m = 2`. (Oracle: the closed form `v · dt / m < 2 R`; heights are sampled at
/// `y0 − k · v · dt / m`.)
#[test]
fn the_coupling_limit_is_v_dt_over_m_below_twice_the_reach() {
    let limit = |m: usize| 2.0 * reach() * m as f64 * 60.0; // m/s
    let speed = |m: usize, ratio: f64| Fix128::from_f64(-limit(m) * ratio);
    // caught: three start phases at 0.75 of the limit, m = 1 and m = 2
    for m in [1usize, 2] {
        for y0 in [(50, 100), (47, 100), (44, 100)] {
            let (mut w, b) = scene(m, r(y0.0, y0.1), speed(m, 0.75), Vec3Fix::ZERO);
            for _ in 0..60 {
                w.step(dt());
            }
            assert!(ball_above_centre(&w, b), "m {m} y0 {y0:?}: caught at 0.75 of the limit");
        }
    }
    // passes: at 1.5 of the m = 1 limit (0.18 m per step, 2 R = 0.12 m), from y0 =
    // 0.45 the samples are 0.27, 0.09, -0.09: none inside |y| < 0.06
    let v = speed(1, 1.5);
    let (mut w, b) = scene(1, r(45, 100), v, Vec3Fix::ZERO);
    for _ in 0..60 {
        w.step(dt());
    }
    assert!(!ball_above_centre(&w, b), "m 1: passes through at 1.5 of its limit");
    // the same speed is 0.75 of the m = 2 limit: caught
    let (mut w, b) = scene(2, r(45, 100), v, Vec3Fix::ZERO);
    for _ in 0..60 {
        w.step(dt());
    }
    assert!(ball_above_centre(&w, b), "m 2: the same speed is caught");
}
