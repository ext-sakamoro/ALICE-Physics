//! One physics modifier on one SDF, without the chain's `Vec`
//!
//! Reaches `SingleModifiedSdf::new` and `SingleModifiedSdf::update`.
//!
//! A unit ball wears away at a constant rate: `Erosion { rate }` adds
//! `rate · t` to the distance, so after `n` updates of `dt` the surface is
//! the ball of radius `1 − rate · n · dt` and the distance at radius `ρ` is
//! `ρ − 1 + rate · n · dt`. With `rate = 1/4` and `dt = 1/8` every term is a
//! multiple of `1/32`, so the `f32` arithmetic is exact and the checks below
//! are `==`. The normal of the worn ball is still radial.
//!
//! Switching the modifier off (`is_active() == false`) leaves the original
//! distance, as `ModifiedSdf` does with an inactive member of its chain, while
//! `update` still advances it: switched back on, the wear accumulated in the
//! meantime shows at once.
//!
//! Run with: `cargo run --example single_modified_sdf`

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sim_modifier::{PhysicsModifier, SingleModifiedSdf};

/// Material removed at a constant rate while enabled.
struct Erosion {
    rate: f32,
    elapsed: f32,
    enabled: bool,
}

impl PhysicsModifier for Erosion {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d + self.rate * self.elapsed
    }
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;
    }
    fn name(&self) -> &str {
        "erosion"
    }
    fn is_active(&self) -> bool {
        self.enabled
    }
}

fn main() {
    let ball = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / l, y / l, z / l)
        },
    );
    let rate = 0.25_f32;
    let dt = 0.125_f32;
    let mut worn = SingleModifiedSdf::new(
        Box::new(ball),
        Erosion {
            rate,
            elapsed: 0.0,
            enabled: true,
        },
    );
    assert_eq!(worn.distance(2.0, 0.0, 0.0), 1.0, "no time has passed");

    for n in 1..=8_u8 {
        worn.update(dt);
        let want = 2.0 - 1.0 + rate * f32::from(n) * dt;
        assert_eq!(worn.distance(2.0, 0.0, 0.0), want, "frame {n}");
    }
    // Radius 1 − 1/4 · 1 = 3/4: a point at 3/4 is on the surface.
    assert_eq!(worn.distance(0.0, 0.75, 0.0), 0.0);
    let (nx, ny, nz) = worn.normal(0.0, 0.0, 0.5);
    assert!(nx.abs() < 1e-3 && ny.abs() < 1e-3 && (nz - 1.0).abs() < 1e-3);
    println!("[single_modified_sdf] worn radius after 1 s: 0.75");

    // Off: the original ball; updates still accumulate.
    worn.modifier.enabled = false;
    assert_eq!(
        worn.distance(2.0, 0.0, 0.0),
        1.0,
        "inactive: original distance"
    );
    for _ in 0..8 {
        worn.update(dt);
    }
    assert_eq!(worn.distance(2.0, 0.0, 0.0), 1.0);
    worn.modifier.enabled = true;
    // 2 s of wear in all: ρ − 1 + 1/4 · 2.
    assert_eq!(worn.distance(2.0, 0.0, 0.0), 1.5);
    println!("[single_modified_sdf] off for 1 s, on again: wear of 2 s shows");
}
