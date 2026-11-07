//! The turn the position-level contact friction applies is composed in
//! `RigidBody::turn` and enters the angular velocity as `log(turn) / h`.
//!
//! Oracles:
//! - (a) a ball turned by two contacts in the same substep moves the same
//!   way whatever its initial orientation, bit for bit (the composition of
//!   several turns is orientation-free, as one turn is);
//! - (b) a kinematic body whose target rotates keeps the fallback: its
//!   angular velocity is the rotation change `log(q · q_prev⁻¹) / h` (within
//!   the rounding of `1/h`), while the ball it carries is turned by friction
//!   and rolls.
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::{
    PhysicsConfig, PhysicsMaterial, PhysicsWorld, RigidBody, SleepConfig, SolverBackend,
};

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping_rate: Fix128::ZERO,
        solver_backend: SolverBackend::Xpbd,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    w
}

fn motion(w: &PhysicsWorld) -> Vec<[Vec3Fix; 3]> {
    w.bodies
        .iter()
        .map(|b| [b.position, b.velocity, b.angular_velocity])
        .collect()
}

/// A ball of radius 1/2 at the origin resting in the V between two static
/// spheres of radius 50 whose centres lie 30° either side of straight down
/// at distance 50.5 (both contacts carry friction), sliding along the groove
/// (`z`); the groove's floor falls by `z²/101` over the run, so gravity keeps
/// the ball on both spheres.
fn v_groove(q: QuatFix) -> PhysicsWorld {
    let mut w = world();
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fr(1, 2), Fix128::ZERO));
    let d = Fix128::from_ratio(101, 2);
    let (sx, cy) = (d * fr(1, 2), d * (fr(3, 4).sqrt()));
    for x in [-1_i64, 1] {
        let s = w.add_body_with_radius(
            RigidBody::new_static(v3(sx * Fix128::from_int(x), -cy, Fix128::ZERO)),
            Fix128::from_int(50),
        );
        w.set_body_material(s, id);
    }
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(v3(
        Fix128::ZERO,
        Fix128::ZERO,
        fr(3, 1),
    ));
    b.inv_inertia = v3(fr(5, 2), fr(5, 2), fr(5, 2));
    b.set_rotation(q);
    let i = w.add_body_with_radius(b, fr(1, 2));
    w.set_body_material(i, id);
    w
}

fn orientations() -> Vec<QuatFix> {
    (0..12_i64)
        .map(|i| {
            let axis = v3(fr(1 + i, 3), fr(2 - i, 5), fr(3 + 2 * i, 7)).normalize();
            QuatFix::from_axis_angle(axis, fr(7 + i, 9)).normalize()
        })
        .collect()
}

#[test]
fn two_contacts_in_one_substep_keep_the_motion_independent_of_orientation() {
    let mut base = v_groove(QuatFix::IDENTITY);
    for _ in 0..60 {
        base.step(fr(1, 60));
    }
    let ball = 2;
    // still in the groove (both spheres support it; the floor fell by at most
    // z²/101 ≈ 0.09 at z = 3) and spinning: both contacts turned it
    let p = base.bodies[ball].position;
    assert!(
        p.x.abs().to_f64() < 0.01 && p.y.to_f64() > -0.15 && p.y.to_f64() <= 0.0,
        "the ball left the groove: {:?}",
        (p.x.to_f64(), p.y.to_f64(), p.z.to_f64())
    );
    let w = base.bodies[ball].angular_velocity;
    assert!(
        !(w.x.is_zero() && w.y.is_zero() && w.z.is_zero()),
        "the friction must spin the ball, or the scene tests nothing"
    );
    let m0 = motion(&base);
    for q in orientations() {
        let mut other = v_groove(q);
        for _ in 0..60 {
            other.step(fr(1, 60));
        }
        assert_eq!(
            m0,
            motion(&other),
            "the ball's motion depends on its initial orientation"
        );
    }
}

/// The derived angular velocity of a kinematic body is the exact logarithm of
/// its rotation change over the last substep (`angular_from_rotations`),
/// recomputed here from the public rotations with the same formula.
fn log_rate(q: QuatFix, prev: QuatFix, inv_h: Fix128) -> Vec3Fix {
    let dq = q.mul(prev.conjugate());
    let (v, w) = if dq.w < Fix128::ZERO {
        (v3(-dq.x, -dq.y, -dq.z), -dq.w)
    } else {
        (v3(dq.x, dq.y, dq.z), dq.w)
    };
    let s2 = v.dot(v);
    let factor = if s2 < Fix128::from_raw(0, 1 << 48) {
        let series = fr(1, 6) + s2 * (fr(3, 40) + s2 * fr(5, 112));
        (Fix128::ONE + s2 * series).double()
    } else {
        let s = s2.sqrt();
        Fix128::atan2(s, w).double() / s
    };
    v * (factor * inv_h)
}

#[test]
fn a_kinematic_turntable_keeps_the_rotation_change_rule_and_turns_the_ball_on_it() {
    let mut w = world();
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fr(1, 2), Fix128::ZERO));
    let table = w.add_body_with_radius(
        RigidBody::new_kinematic(v3(Fix128::ZERO, Fix128::from_int(-1000), Fix128::ZERO)),
        Fix128::from_int(1000),
    );
    w.set_body_material(table, id);
    let mut b = RigidBody::new_dynamic(v3(fr(2, 1), fr(1, 2), Fix128::ZERO), Fix128::ONE);
    b.inv_inertia = v3(fr(5, 2), fr(5, 2), fr(5, 2));
    let ball = w.add_body_with_radius(b, fr(1, 2));
    w.set_body_material(ball, id);
    let dt = fr(1, 60);
    let inv_h = Fix128::from_int(60 * 8);
    let mut angle = Fix128::ZERO;
    for _ in 0..30 {
        angle = angle + fr(1, 100);
        let target = QuatFix::from_axis_angle(v3(Fix128::ZERO, Fix128::ONE, Fix128::ZERO), angle);
        w.bodies[table].kinematic_target = Some((w.bodies[table].position, target));
        w.step(dt);
        let t = &w.bodies[table];
        // the same formula; the solver's `1/h` is its own rounding of 1/(dt/8),
        // so compare within 2⁻⁵⁰ (measured difference 154 ulp ≈ 8e-18)
        let d = t.angular_velocity - log_rate(t.rotation, t.prev_rotation, inv_h);
        let tol = Fix128::from_raw(0, 1 << 14);
        assert!(
            d.x.abs() <= tol && d.y.abs() <= tol && d.z.abs() <= tol,
            "a kinematic body's angular velocity is the logarithm of its rotation change: {d:?}"
        );
    }
    let wb = w.bodies[ball].angular_velocity;
    assert!(
        !(wb.x.is_zero() && wb.y.is_zero() && wb.z.is_zero()),
        "the turntable's friction must turn the ball"
    );
}
