//! A soft sheet springing back to its rest shape with `Cloth::set_rest_tether`.
//!
//! The sheet is poked out of shape and released; every particle is pulled toward its rest
//! position by an underdamped spring-damper (damping ratio 0.42), so it overshoots and
//! settles. The edge and bending constraints keep acting on top.
//!
//! Run: `cargo run --example cloth_rest_tether --features std`

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::Cloth;

fn main() {
    let (res, mass) = (7_usize, Fix128::from_ratio(1, 49));
    let mut sheet = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, res, res, mass);
    sheet.config.gravity = Vec3Fix::ZERO;
    sheet.config.damping = Fix128::ONE;

    // ω = 10 rad/s, ζ = 0.42 per particle: k = m ω², c = 2 ζ m ω
    let omega = Fix128::from_int(10);
    let zeta = Fix128::from_ratio(42, 100);
    let k = mass * omega * omega;
    let c = Fix128::from_int(2) * zeta * mass * omega;
    let rest: Vec<Vec3Fix> = sheet.positions.clone();
    sheet
        .set_rest_positions(&rest)
        .expect("one rest position per particle");
    sheet.set_rest_tether(k, c);
    let (k_set, c_set) = sheet.rest_tether().expect("enabled");
    println!(
        "[cloth_rest_tether] {} particles, k={:.4} N/m c={:.4} N*s/m",
        sheet.particle_count(),
        k_set.to_f64(),
        c_set.to_f64()
    );

    // poke the centre out of the plane
    let centre = res * res / 2;
    sheet.positions[centre] = sheet.positions[centre]
        + Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(3, 10), Fix128::ZERO);

    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..=120 {
        if frame % 10 == 0 {
            let lift = (sheet.positions[centre] - sheet.rest_positions()[centre]).y;
            println!(
                "[cloth_rest_tether] t={:.2}s centre lift={:+.5}",
                f64::from(frame) / 60.0,
                lift.to_f64()
            );
        }
        sheet.step(dt);
    }

    sheet.clear_rest_tether();
    println!(
        "[cloth_rest_tether] tether cleared: {:?}",
        sheet.rest_tether()
    );
}
