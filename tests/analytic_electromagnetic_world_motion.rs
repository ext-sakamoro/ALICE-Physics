//! Closed-form oracles for charged-body motion driven through the real
//! `PhysicsWorld` time integration (COV-PLASMA-027, COV-PLASMA-130,
//! COV-PLASMA-131). The existing
//! `tests/analytic_electromagnetic_wiring.rs::cyclotron_orbit_has_the_gyro_radius_m_v_over_q_b`
//! integrates locally inside the test, not through the crate's own
//! `lorentz_force` + `RigidBody::add_force` + `PhysicsWorld::step` path (see
//! docs/coverage/plasma.toml COV-PLASMA-130 evidence before this test
//! landed); these tests drive the real path instead.
//!
//! For a charge q, mass m, uniform magnetic field B = (0, 0, Bz) and zero
//! electric field, m dv/dt = q v x B gives circular motion at angular
//! frequency omega = q Bz / m, period T = 2 pi / |omega|, radius
//! r = v_perp / |omega| = m v_perp / (|q| Bz):
//!   vx(t) = v0 cos(omega t), vy(t) = -v0 sin(omega t)
//!   x(t) = (v0/omega) sin(omega t), y(t) = (v0/omega) (cos(omega t) - 1)
//! starting from the origin with v(0) = (v0, 0, 0).
//!
//! With a uniform electric field added, the motion is the same cyclotron
//! orbit plus a constant guiding-centre drift at E x B / B^2 (Chen ch.2);
//! the velocity averaged over one full period isolates the drift because
//! the cyclotron component is periodic and averages to zero.
//!
//! `RigidBody::add_force` applies `F dt` to velocity once at the start of
//! each step before `PhysicsWorld::step` integrates position from that
//! same-step velocity (semi-implicit Euler), not a rotation-preserving
//! integrator (e.g. Boris) -- so matching the closed form to float/Fix128
//! precision is not expected; the tolerances below are sized from the
//! step count instead (see comments at each assertion).
//!
//! Sleep must be disabled for this to measure the physics rather than the
//! sleep system: `SleepConfig::frames_to_sleep` (default 60) counts steps,
//! not real time, and assumes a ~1/60 s step. Driving `PhysicsWorld::step`
//! directly at a much smaller dt (here T/20000, needed for integration
//! accuracy over a cyclotron period) makes that threshold correspond to a
//! tiny fraction of a real second, so a body whose instantaneous speed
//! legitimately dips near zero during gyration (as the E x B drift case
//! below does) falls asleep almost immediately and its velocity freezes --
//! found while developing this test: the drift average came out as
//! (0.909, -0.159) instead of (1.0, 0.0), stable regardless of step count,
//! and exactly (0.0, 0.0) after a few periods; disabling sleep alone fixed
//! it to (0.99984, 0.0), confirming the sleep system, not the Lorentz-force
//! integration, was the cause.

use alice_physics::electromagnetic::{lorentz_force, ChargedBody, EmSource};
use alice_physics::sleeping::SleepConfig;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn ffx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Runs `steps` frames of dt each, applying the Lorentz force from `source`
/// to the one charged body in `world` before every `step`, and returns the
/// position/velocity history (one entry per frame, including frame 0).
fn run_charged_body(
    world: &mut PhysicsWorld,
    body_idx: usize,
    charged: ChargedBody,
    source: &EmSource,
    dt: Fix128,
    steps: usize,
) -> Vec<(Vec3Fix, Vec3Fix)> {
    let mut history = vec![(
        world.bodies[body_idx].position,
        world.bodies[body_idx].velocity,
    )];
    for _ in 0..steps {
        let force = lorentz_force(charged, &world.bodies[body_idx], source);
        world.bodies[body_idx].add_force(force, dt);
        world.step(dt);
        history.push((
            world.bodies[body_idx].position,
            world.bodies[body_idx].velocity,
        ));
    }
    history
}

fn no_gravity_world() -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    // See the module doc: sleep's frames_to_sleep is a step count, not a
    // real-time threshold, and fires far too early at this test's dt.
    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::ZERO,
        angular_threshold: Fix128::ZERO,
        frames_to_sleep: u32::MAX,
    });
    world
}

#[test]
fn cyclotron_period_and_gyroradius_match_closed_form_through_world_step() {
    // q=1, m=1, Bz=1 -> omega=1 rad/s, T=2*pi s, v0=2 -> r=2 m.
    let q = 1.0_f64;
    let mass = 1.0_f64;
    let bz = 1.0_f64;
    let v0 = 2.0_f64;
    let omega = q * bz / mass;
    let t_period = 2.0 * std::f64::consts::PI / omega;

    let mut world = no_gravity_world();
    let body_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, ffx(mass)));
    world.bodies[body_idx].velocity = Vec3Fix::new(ffx(v0), Fix128::ZERO, Fix128::ZERO);
    let charged = ChargedBody::new(body_idx, ffx(q));
    let source = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, ffx(bz)),
    };

    let steps = 20_000; // dt = T / steps, small relative to the period
    let dt = ffx(t_period / steps as f64);
    let history = run_charged_body(&mut world, body_idx, charged, &source, dt, steps);

    // Quarter period: closed form x = v0/omega, y = -v0/omega (see module doc).
    let quarter_idx = steps / 4;
    let (pos_q, _) = history[quarter_idx];
    let r = v0 / omega; // gyroradius, 2.0 here
    let rel_err_x = ((pos_q.x.to_f64() - r) / r).abs();
    let rel_err_y = ((pos_q.y.to_f64() - (-r)) / r).abs();
    assert!(
        rel_err_x < 1e-2 && rel_err_y < 1e-2,
        "quarter-period position: got ({:.4}, {:.4}) expected ({:.4}, {:.4})",
        pos_q.x.to_f64(),
        pos_q.y.to_f64(),
        r,
        -r
    );

    // Full period: back near the start, within the same step-count tolerance.
    let (pos_t, vel_t) = history[steps];
    assert!(
        pos_t.x.to_f64().abs() < r * 1e-2 && pos_t.y.to_f64().abs() < r * 1e-2,
        "full-period position should return near the origin: got ({:.4}, {:.4})",
        pos_t.x.to_f64(),
        pos_t.y.to_f64()
    );
    let speed_err = ((vel_t.length().to_f64() - v0) / v0).abs();
    assert!(
        speed_err < 1e-2,
        "speed after one period: got {:.6} expected {v0} (rel err {speed_err})",
        vel_t.length().to_f64()
    );
}

#[test]
fn speed_is_not_exactly_conserved_by_the_semi_implicit_step() {
    // docs/coverage/plasma.toml COV-PLASMA-027 flags this as a plausible
    // consequence of applying a velocity-dependent force once per frame
    // rather than with a rotation-preserving integrator; this measures it
    // rather than asserting conservation. A non-zero but small drift here
    // is the expected, not-a-defect finding; a drift comparable to v0
    // itself would be.
    let q = 1.0_f64;
    let mass = 1.0_f64;
    let bz = 1.0_f64;
    let v0 = 2.0_f64;

    let mut world = no_gravity_world();
    let body_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, ffx(mass)));
    world.bodies[body_idx].velocity = Vec3Fix::new(ffx(v0), Fix128::ZERO, Fix128::ZERO);
    let charged = ChargedBody::new(body_idx, ffx(q));
    let source = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, ffx(bz)),
    };

    let steps = 20_000;
    let omega = q * bz / mass;
    let t_period = 2.0 * std::f64::consts::PI / omega;
    let dt = ffx(t_period / steps as f64);
    let history = run_charged_body(&mut world, body_idx, charged, &source, dt, steps);

    let speed_drift = (history[steps].1.length().to_f64() - v0) / v0;
    println!("speed drift over one period at {steps} steps/period: {speed_drift:+.6} (relative)");
    // Not asserting a specific sign or magnitude beyond "bounded, not runaway":
    // a semi-implicit Euler step on a rotating force typically gains a
    // small amount of energy per step, not loses an order-1 fraction.
    assert!(
        speed_drift.abs() < 0.1,
        "speed drift should be a small numerical effect, not order-1: {speed_drift}"
    );
}

#[test]
fn e_cross_b_drift_matches_closed_form_average_velocity_through_world_step() {
    // E = (0, 1, 0) V/m, B = (0, 0, 1) T -> drift = E x B / B^2 = (1, 0, 0) m/s,
    // independent of q and m (tested with two different charge/mass pairs).
    let bz = 1.0_f64;
    let ey = 1.0_f64;
    let expected_drift = Vec3Fix::new(ffx(ey * bz), Fix128::ZERO, Fix128::ZERO); // E x B / B^2 with these values

    for &(q, mass) in &[(1.0_f64, 1.0_f64), (2.0_f64, 0.5_f64)] {
        let omega = q * bz / mass;
        let t_period = 2.0 * std::f64::consts::PI / omega.abs();

        let mut world = no_gravity_world();
        let body_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, ffx(mass)));
        // Start at the drift velocity plus a perpendicular gyration component,
        // as a generic initial condition (not the drift velocity alone).
        world.bodies[body_idx].velocity =
            Vec3Fix::new(ffx(expected_drift.x.to_f64()), ffx(1.0), Fix128::ZERO);
        let charged = ChargedBody::new(body_idx, ffx(q));
        let source = EmSource::Uniform {
            electric: Vec3Fix::new(Fix128::ZERO, ffx(ey), Fix128::ZERO),
            magnetic: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, ffx(bz)),
        };

        let steps = 20_000;
        let dt = ffx(t_period / steps as f64);
        let mut vx_sum = 0.0_f64;
        let mut vy_sum = 0.0_f64;
        let mut history_len = 0usize;
        for _ in 0..steps {
            let force = lorentz_force(charged, &world.bodies[body_idx], &source);
            world.bodies[body_idx].add_force(force, dt);
            world.step(dt);
            vx_sum += world.bodies[body_idx].velocity.x.to_f64();
            vy_sum += world.bodies[body_idx].velocity.y.to_f64();
            history_len += 1;
        }
        let avg_vx = vx_sum / history_len as f64;
        let avg_vy = vy_sum / history_len as f64;

        let err_x = (avg_vx - expected_drift.x.to_f64()).abs();
        let err_y = avg_vy.abs();
        assert!(
            err_x < 0.005 && err_y < 0.005,
            "q={q} m={mass}: average velocity over one period ({avg_vx:.4}, {avg_vy:.4}) \
             expected drift ({:.4}, 0.0)",
            expected_drift.x.to_f64()
        );
    }
}
