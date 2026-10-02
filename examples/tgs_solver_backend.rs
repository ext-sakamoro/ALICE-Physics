//! Demonstrates `PhysicsConfig::solver_backend`: the same three scenes
//! (free body, resting contact, one joint) run under both `SolverBackend::Xpbd`
//! (default) and `SolverBackend::Tgs`, printing the closed-form expectation
//! next to each backend's actual result.
//!
//! ```bash
//! cargo run --example tgs_solver_backend --features std
//! ```
//!
//! See `SolverBackend`'s doc for the documented gaps of the `Tgs` path
//! (joints unenforced, kinematic targets not advanced, SDF colliders
//! skipped) — the joint scene below exists specifically to make that first
//! gap visible side by side with XPBD, not to hide it.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn backend_name(b: SolverBackend) -> &'static str {
    match b {
        SolverBackend::Xpbd => "Xpbd",
        SolverBackend::Tgs => "Tgs",
        _ => "unknown",
    }
}

/// Scene 1: a single free body falling under gravity, no contacts, no
/// joints — damping disabled so the closed form is the exact semi-implicit
/// Euler recurrence both backends share for this case (see
/// `tests/analytic_tgs_wiring.rs::free_fall_matches_semi_implicit_euler_exactly_on_both_backends`).
fn scene_free_fall(backend: SolverBackend) {
    let g = Fix128::from_int(-10);
    let h = r(1, 60);
    let frames: u32 = 60;
    let x0 = Fix128::from_int(20);

    let mut v = Fix128::ZERO;
    let mut x = x0;
    for _ in 0..frames {
        v = v + g * h;
        x = x + v * h;
    }

    let cfg = PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, g, Fix128::ZERO),
        damping: Fix128::ONE,
        substeps: 1,
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, x0, Fix128::ZERO),
        Fix128::ONE,
    ));
    for _ in 0..frames {
        w.step(h);
    }

    println!(
        "[tgs_solver_backend free_fall] backend={:<4} closed_form_y={:.6} actual_y={:.6} closed_form_v={:.6} actual_v={:.6}",
        backend_name(backend),
        x.to_f64(),
        w.bodies[b].position.y.to_f64(),
        v.to_f64(),
        w.bodies[b].velocity.y.to_f64(),
    );
}

/// Scene 2: a dynamic sphere released exactly at the resting height on a
/// static sphere "ground", vertically aligned (zero torque on this scene
/// for either backend). Closed-form prediction: it stays within `slop` of
/// the resting height and does not fly off.
fn scene_resting_contact(backend: SolverBackend) {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let ground_r = r(10, 1);
    let ball_r = Fix128::ONE;
    let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.set_body_collision_radius(ground, ground_r);
    let resting_height = Fix128::from_int(11); // ground_r + ball_r
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, resting_height, Fix128::ZERO),
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, ball_r);

    let dt = r(1, 60);
    for _ in 0..120 {
        w.step(dt);
    }

    println!(
        "[tgs_solver_backend resting_contact] backend={:<4} closed_form_y≈{:.3} (within slop) actual_y={:.6} actual_vy={:.6}",
        backend_name(backend),
        resting_height.to_f64(),
        w.bodies[ball].position.y.to_f64(),
        w.bodies[ball].velocity.y.to_f64(),
    );
}

/// Scene 3: a dynamic body orbiting a fixed anchor via a rigid
/// `DistanceConstraint`, zero gravity. This is the scene that makes
/// `SolverBackend::Tgs`'s documented joint gap visible: XPBD enforces the
/// constraint (distance from the anchor stays near `target_distance`); TGS
/// only uses the joint for island grouping, so the orbiter follows an
/// *exact* free-inertial straight line instead — printed here as the
/// closed form for TGS specifically (it is not a valid prediction for
/// XPBD, which is why the two closed-form columns differ).
fn scene_joint(backend: SolverBackend) {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps: 1,
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut orbiter = RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    orbiter.velocity = Vec3Fix::from_int(0, 10, 0);
    let orbiter = w.add_body(orbiter);
    w.add_distance_constraint(DistanceConstraint::new(
        anchor,
        orbiter,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::from_int(5),
    ));

    let h = r(1, 60);
    let frames = 30;

    // TGS-only closed form: no force at all acts on the orbiter under this
    // backend (gravity=0, no contacts, joint unenforced), so position is
    // the exact free-inertial line, accumulated the same way production
    // accumulates it (one `+= v0 * h` per step, not a single `v0 * N*h`
    // multiply — see the comment in the integration test for why that
    // distinction matters at the bit level).
    let x0 = Vec3Fix::from_int(5, 0, 0);
    let v0 = Vec3Fix::from_int(0, 10, 0);
    let mut tgs_closed_form = x0;
    for _ in 0..frames {
        tgs_closed_form = tgs_closed_form + v0 * h;
    }

    for _ in 0..frames {
        w.step(h);
    }

    let actual = w.bodies[orbiter].position;
    let dist_from_anchor = (actual - w.bodies[anchor].position).length();
    match backend {
        SolverBackend::Tgs => println!(
            "[tgs_solver_backend joint] backend=Tgs  closed_form_pos=({:.4},{:.4},{:.4}) actual_pos=({:.4},{:.4},{:.4}) dist_from_anchor={:.4} (target_distance=5, unenforced under Tgs)",
            tgs_closed_form.x.to_f64(), tgs_closed_form.y.to_f64(), tgs_closed_form.z.to_f64(),
            actual.x.to_f64(), actual.y.to_f64(), actual.z.to_f64(),
            dist_from_anchor.to_f64(),
        ),
        _ => println!(
            "[tgs_solver_backend joint] backend=Xpbd actual_pos=({:.4},{:.4},{:.4}) dist_from_anchor={:.4} (target_distance=5, enforced by the joint)",
            actual.x.to_f64(), actual.y.to_f64(), actual.z.to_f64(),
            dist_from_anchor.to_f64(),
        ),
    }
}

fn main() {
    println!("ALICE-Physics SolverBackend::Tgs wiring demo");
    println!("=============================================");
    println!();
    println!("-- Scene 1: free fall (no contacts, no joints) --");
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        scene_free_fall(backend);
    }
    println!();
    println!("-- Scene 2: resting contact (vertically-aligned sphere-on-sphere) --");
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        scene_resting_contact(backend);
    }
    println!();
    println!("-- Scene 3: one joint (documents the Tgs backend's unenforced-joint gap) --");
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        scene_joint(backend);
    }
    println!();
    println!(
        "Done. See `SolverBackend`'s rustdoc for the Tgs path's full list of documented gaps."
    );
}
