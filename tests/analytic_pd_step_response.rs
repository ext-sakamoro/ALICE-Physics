//! Closed-form oracle for a PD-controlled mass step response (COV-SENSE-089).
//! `src/motor.rs::PdController::compute` is used to drive a body through
//! `RigidBody::add_force` and `PhysicsWorld::step`; the single-axis PD law
//! itself is covered by COV-RIGID-030, but nothing compared the resulting
//! step response to the standard second-order closed form before this test
//! (see docs/coverage/sense.toml COV-SENSE-089 evidence).
//!
//! For a mass m under `F = kp (target - x) - kd v` (position mode, zero
//! velocity target), `m x'' + kd x' + kp x = kp target` is the standard
//! second-order form `x'' + 2 zeta omega_n x' + omega_n^2 x = omega_n^2
//! target` with `omega_n = sqrt(kp / m)` and `zeta = kd / (2 sqrt(kp m))`
//! (Astrom-Murray ch.5.3). Starting at rest at x=0 with an underdamped
//! (zeta < 1) step to `target`, the response overshoots to
//! `target * (1 + exp(-zeta pi / sqrt(1 - zeta^2)))` at
//! `t_peak = pi / (omega_n sqrt(1 - zeta^2))`.

use alice_physics::det_math;
use alice_physics::sleeping::SleepConfig;
use alice_physics::{
    Fix128, MotorMode, PdController, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix,
};

fn ffx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

#[test]
fn pd_controlled_mass_overshoot_matches_second_order_closed_form() {
    let mass = 1.0_f64;
    let kp = 1.0_f64;
    let kd = 1.0_f64;
    let target = 1.0_f64;

    let omega_n = (kp / mass).sqrt();
    let zeta = kd / (2.0 * (kp * mass).sqrt());
    assert!(zeta < 1.0, "test setup must be underdamped: zeta={zeta}");

    let zeta_f32 = zeta as f32;
    let overshoot_fraction =
        det_math::exp(-zeta_f32 * std::f32::consts::PI / (1.0 - zeta_f32 * zeta_f32).sqrt());
    let omega_d = omega_n * (1.0 - zeta * zeta).sqrt();
    let t_peak = std::f64::consts::PI / omega_d;

    // damping is applied once per `step()` call, not scaled by dt, so
    // driving `step()` directly at a much smaller dt than ~1/60s (needed
    // here for accuracy at t_peak) would otherwise crush velocity almost
    // immediately (0.99^(many thousands) ~ 0); see the same pattern in
    // analytic_electromagnetic_world_motion.rs's SleepConfig finding.
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::ZERO,
        angular_threshold: Fix128::ZERO,
        frames_to_sleep: u32::MAX,
    });
    let body_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, ffx(mass)));

    let mut controller = PdController::new(ffx(kp), ffx(kd), ffx(1.0e9));
    controller.set_position_target(ffx(target));
    assert_eq!(controller.mode, MotorMode::Position);

    let dt_f64 = 1.0 / 4800.0;
    let dt = Fix128::from_ratio(1, 4800);
    let steps = (t_peak / dt_f64).round() as usize;
    let mut peak_x = 0.0_f64;
    for _ in 0..steps {
        let x = world.bodies[body_idx].position.x;
        let v = world.bodies[body_idx].velocity.x;
        let force = controller.compute(x, v);
        world.bodies[body_idx].add_force(Vec3Fix::new(force, Fix128::ZERO, Fix128::ZERO), dt);
        world.step(dt);
        let x_now = world.bodies[body_idx].position.x.to_f64();
        if x_now > peak_x {
            peak_x = x_now;
        }
    }

    let expected_peak = target * (1.0 + overshoot_fraction as f64);
    let rel_err = (peak_x - expected_peak).abs() / target;
    assert!(
        rel_err < 1e-2,
        "peak position: got {peak_x:.5} expected {expected_peak:.5} (overshoot {:.4}, rel err {rel_err:.4})",
        overshoot_fraction
    );
}
