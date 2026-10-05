//! A wing in the standard atmosphere: lift and drag laws, and a steady glide.
//!
//! 1. `atmosphere::Isa1976` gives the air density at the glide altitude
//!    (geometric 3 000 m, about 0.909 kg/m³).
//! 2. `lift_drag::LiftDragSurface` turns the angle of attack into `C_L`,
//!    `C_D` (attached flow, stall transition, flat plate) and the force on a
//!    `RigidBody`.
//! 3. A `PhysicsWorld` body carrying the wing is pitched so that the
//!    closed-form glide path puts the wing at `α* = 0.08 rad`; after the
//!    transient it flies the closed-form path and speed:
//!
//! ```text
//! tan γ = C_D / C_L,    V = √(2 m g / (ρ S √(C_L² + C_D²)))
//! ```
//!
//! The closed forms are evaluated here in `f64` and compared to the run.
//!
//! ```bash
//! cargo run --release --example aero_glider
//! ```

// f64 transcendentals compute the closed-form references, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::atmosphere::Isa1976;
use alice_physics::lift_drag::{LiftDragParams, LiftDragSurface, THIN_AIRFOIL_LIFT_SLOPE};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn main() {
    // --- 1. atmosphere --------------------------------------------------
    let z = Fix128::from_int(3_000);
    let air = Isa1976::at_geometric_altitude(z).expect("3 km is inside the model");
    let h = Isa1976::geopotential_altitude_m(z);
    let same = Isa1976::at_geopotential_altitude(h).expect("in range");
    assert_eq!(air, same);
    println!(
        "[aero_glider] ISA Z=3000 m (H={:.1} m, top {} m): T={:.3} K p={:.1} Pa rho={:.5} kg/m3 a={:.2} m/s",
        h.to_f64(),
        Isa1976::MAX_GEOPOTENTIAL_ALTITUDE_M,
        air.temperature_k.to_f64(),
        air.pressure_pa.to_f64(),
        air.density_kg_m3.to_f64(),
        air.speed_of_sound_m_s.to_f64()
    );
    let rho = air.density_kg_m3;

    // --- 2. wing --------------------------------------------------------
    let (area, ar, e, cd0) = (0.5_f64, 8.0_f64, 0.9_f64, 0.03_f64);
    let wing = LiftDragSurface::new(LiftDragParams {
        area_m2: Fix128::from_f64(area),
        aspect_ratio: Fix128::from_f64(ar),
        oswald_efficiency: Fix128::from_f64(e),
        section_lift_slope_per_rad: THIN_AIRFOIL_LIFT_SLOPE,
        zero_lift_angle_rad: Fix128::ZERO,
        stall_angle_rad: Fix128::from_ratio(1, 4),
        stall_transition_rad: Fix128::from_ratio(1, 10),
        zero_lift_drag_coefficient: Fix128::from_f64(cd0),
        chord_axis_local: Vec3Fix::UNIT_X,
        lift_axis_local: Vec3Fix::UNIT_Y,
        center_of_pressure_local: Vec3Fix::ZERO,
    })
    .expect("valid wing");
    let a = wing.lift_slope_per_rad().to_f64();
    println!(
        "[aero_glider] lift slope a={a:.4} /rad (2pi={:.4}), area {} m2",
        2.0 * std::f64::consts::PI,
        wing.params().area_m2.to_f64()
    );
    for deg in [0, 5, 10, 14, 17, 20, 45, 90] {
        let c = wing.coefficients(Fix128::from_f64(f64::from(deg).to_radians()));
        println!(
            "[aero_glider] alpha={deg:>2} deg  C_L={:+.4}  C_D={:.4}",
            c.lift.to_f64(),
            c.drag.to_f64()
        );
    }

    // --- 3. glide -------------------------------------------------------
    let alpha_star = 0.08_f64;
    let c = wing.coefficients(Fix128::from_f64(alpha_star));
    let (cl, cdr) = (c.lift.to_f64(), c.drag.to_f64());
    // Reference coefficients from the drag polar in f64.
    let cl_ref = a * alpha_star;
    let cd_ref = cd0 + cl_ref * cl_ref / (std::f64::consts::PI * e * ar);
    assert!((cl - cl_ref).abs() < 1e-9 && (cdr - cd_ref).abs() < 1e-9);
    let gamma = (cd_ref / cl_ref).atan();
    let mass = 2.0_f64;
    let config = SolverConfig {
        damping: Fix128::ONE,
        ..SolverConfig::default()
    };
    let g = -config.gravity.y.to_f64();
    let v_ss = (2.0 * mass * g / (rho.to_f64() * area * cl_ref.hypot(cd_ref))).sqrt();

    let mut world = PhysicsWorld::new(config);
    let mut body = RigidBody::new(Vec3Fix::from_int(0, 3_000, 0), Fix128::from_f64(mass));
    body.set_rotation(QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f64(alpha_star - gamma),
    ));
    body.velocity = Vec3Fix::new(
        Fix128::from_f64(0.9 * v_ss * gamma.cos()),
        Fix128::from_f64(-0.9 * v_ss * gamma.sin()),
        Fix128::ZERO,
    );
    let idx = world.add_body(body);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..(120 * 60) {
        let b = world.get_body_mut(idx).expect("body");
        wing.apply(b, Vec3Fix::ZERO, rho, dt).expect("rho >= 0");
        world.step(dt);
    }
    let b = world.get_body(idx).expect("body");
    let load = wing.load(b, Vec3Fix::ZERO, rho).expect("rho >= 0");
    let (vx, vy) = (b.velocity.x.to_f64(), b.velocity.y.to_f64());
    let speed = vx.hypot(vy);
    let path = (-vy / vx).atan();
    println!(
        "[aero_glider] steady glide: V={speed:.6} m/s (closed form {v_ss:.6}), gamma={path:.6} rad ({gamma:.6}), L/D={:.4}, alpha={:.6}",
        1.0 / path.tan(),
        load.angle_of_attack_rad.map_or(f64::NAN, Fix128::to_f64)
    );
    assert!(((speed - v_ss) / v_ss).abs() < 1e-5);
    assert!((path - gamma).abs() < 1e-5);
}
