//! Driving every modifier of a `ModifiedSdf` once per frame
//!
//! Reaches `ModifiedSdf::update`, the per-frame call the `sim_modifier`
//! module doc describes (`update(dt)` advances each modifier's simulation,
//! then `modify_distance` alters the SDF in chain order).
//!
//! Two modifiers with a known linear law are chained over the plane `y = 0`:
//! - `Erosion { rate }` recedes the surface by `rate * t`
//!   (`d -> d + rate * t`)
//! - `Swelling { rate }` grows it by `rate * t` (`d -> d - rate * t`)
//!
//! After `n` frames of `dt` the closed form is
//! `d(y) = y + (r_erode - r_swell) n dt`, so with `r_erode = 3/4`,
//! `r_swell = 1/4`, `dt = 1/8` and `n = 16`, a point at `y = 2` reads `3`.
//!
//! Run with: `cargo run --example modified_sdf_frame_update`

use alice_physics::sdf_collider::SdfField;
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier};

/// The plane `y = 0`.
struct Ground;

impl SdfField for Ground {
    fn distance(&self, _x: f32, y: f32, _z: f32) -> f32 {
        y
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// Material removed at a constant rate.
struct Erosion {
    rate: f32,
    elapsed: f32,
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
}

/// Material added at a constant rate.
struct Swelling {
    rate: f32,
    elapsed: f32,
}

impl PhysicsModifier for Swelling {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d - self.rate * self.elapsed
    }
    fn update(&mut self, dt: f32) {
        self.elapsed += dt;
    }
    fn name(&self) -> &str {
        "swelling"
    }
}

fn main() {
    let r_erode = 0.75_f32;
    let r_swell = 0.25_f32;
    let dt = 0.125_f32;

    let mut sdf = ModifiedSdf::new(Box::new(Ground))
        .with_modifier(Box::new(Erosion {
            rate: r_erode,
            elapsed: 0.0,
        }))
        .with_modifier(Box::new(Swelling {
            rate: r_swell,
            elapsed: 0.0,
        }));
    assert_eq!(sdf.distance(0.0, 2.0, 0.0), 2.0, "no time has passed");

    for n in 1..=16_u8 {
        sdf.update(dt);
        let want = 2.0 + (r_erode - r_swell) * f32::from(n) * dt;
        let got = sdf.distance(0.0, 2.0, 0.0);
        // Every term is a multiple of 1/32, so the f32 arithmetic is exact.
        assert_eq!(got, want, "frame {n}");
        if n % 4 == 0 {
            println!("[modified_sdf] frame {n:>2}: d(y = 2) = {got} (closed form {want})");
        }
    }
    assert_eq!(sdf.distance(0.0, 2.0, 0.0), 3.0, "2 + (3/4 - 1/4) * 16 / 8");
    // The surface itself moved down by the same 1: y = -1 is now on it.
    assert_eq!(sdf.distance(5.0, -1.0, 5.0), 0.0);

    // Removing the swelling leaves only the erosion's 3/4 * 2 s = 3/2.
    sdf.clear_modifiers();
    sdf.add_modifier(Box::new(Erosion {
        rate: r_erode,
        elapsed: 2.0,
    }));
    sdf.update(0.0);
    assert_eq!(sdf.distance(0.0, 2.0, 0.0), 3.5, "2 + 3/4 * 2");

    println!("[modified_sdf] update advanced every modifier by dt, as the closed form requires");
}
