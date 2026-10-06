//! Closed-form oracle for `WindZone::force_on` driven through the real
//! `PhysicsWorld` step loop (COV-ENV-115: "terminal velocity of a falling
//! body under quadratic air drag"). The existing wind_zone tests check the
//! instantaneous force law; none run a world to the terminal-velocity
//! asymptote (see docs/coverage/env.toml COV-ENV-115 evidence before this
//! test landed).
//!
//! Hoerner, *Fluid-Dynamic Drag*: a body falling under gravity reaches
//! `v_t = sqrt(2 m g / (rho C_d A))` when drag exactly balances weight.
//! With zero wind (so relative velocity is just the body's own velocity)
//! the closed-form approach to that limit is `v(t) = v_t * tanh(g t / v_t)`.

use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::wind_zone::WindZone;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

#[test]
fn falling_body_in_a_wind_zone_approaches_the_closed_form_terminal_velocity() {
    // mass 10, g 10 (solver default), rho 1, Cd 2, A 1 -> v_t = sqrt(2*10*10/(1*2*1)) = 10 m/s
    let mass = 10.0_f64;
    let g = 10.0_f64;
    let rho = 1.0_f64;
    let cd = 2.0_f64;
    let area = 1.0_f64;
    let v_t = (2.0 * mass * g / (rho * cd * area)).sqrt();

    // Frame-level damping (0.99 velocity retention by default) is an
    // unrelated decay term that would corrupt this closed form, so it is
    // disabled here; everything else is PhysicsConfig::default().
    let config = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let body_idx = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::ZERO,
        Fix128::from_f64(mass),
    ));

    let zone = WindZone {
        shape: ZoneShape::Aabb {
            min: Vec3Fix::new(fx(-1_000_000), fx(-1_000_000), fx(-1_000_000)),
            max: Vec3Fix::new(fx(1_000_000), fx(1_000_000), fx(1_000_000)),
        },
        air_density_kg_m3: Fix128::from_f64(rho),
        direction: Vec3Fix::ZERO,
        base_speed_m_s: Fix128::ZERO,
        turbulence_amplitude: Fix128::ZERO,
        gust_frequency_hz: Fix128::ZERO,
        drag_coefficient: Fix128::from_f64(cd),
        reference_area_m2: Fix128::from_f64(area),
    };

    let dt = Fix128::from_ratio(1, 240); // small frame step for integration accuracy
    let mut t = Fix128::ZERO;
    // v_t / g = 1 s time constant; run 8 s (8 time constants, tanh(8) ~ 1 - 1e-7)
    let frames = 240 * 8;
    for _ in 0..frames {
        let force = zone.force_on(&world.bodies[body_idx], t);
        world.bodies[body_idx].add_force(force, dt);
        world.step(dt);
        t = t + dt;
    }

    let got = -world.bodies[body_idx].velocity.y.to_f64(); // falling: velocity.y is negative
    let rel_err = (got - v_t).abs() / v_t;
    assert!(
        rel_err < 1e-3,
        "terminal velocity: got {got} expected {v_t} (rel err {rel_err})"
    );
}

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}
