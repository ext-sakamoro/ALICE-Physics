//! Wind sheltered by obstacle geometry through `SdfWindField`
//!
//! Reaches `SdfWindField::new` and `SdfWindField::sample`.
//!
//! Closed form, from the module doc (`d = sdf.distance(x)`,
//! `shelter = clamp(d / decay_scale, 0, 1)`, `wind = dir * speed * shelter`,
//! default `decay_scale = 5 m`):
//! - over the ground plane `y = 0` with a 10 m/s wind along `+x`, the wind is
//!   `2 y` m/s for `0 <= y <= 5`, a full 10 m/s above, and 0 at or below the
//!   surface
//! - around a sphere of radius 2 at the origin (`d = |p| - 2`), the wind at
//!   distance `s` from its surface is `2 s` m/s up to `s = 5`
//!
//! Run with: `cargo run --example sdf_wind_shelter`

use alice_physics::sdf_collider::SdfField;
use alice_physics::sdf_wind_field::SdfWindField;

/// The ground plane `y = 0`.
struct Ground;

impl SdfField for Ground {
    fn distance(&self, _x: f32, y: f32, _z: f32) -> f32 {
        y
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// A sphere of radius 2 at the origin.
struct Boulder;

impl SdfField for Boulder {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        (x * x + y * y + z * z).sqrt() - 2.0
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let l = (x * x + y * y + z * z).sqrt();
        (x / l, y / l, z / l)
    }
}

fn close(got: [f32; 3], want: [f32; 3], what: &str) {
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() <= 1e-5,
            "{what}: got {got:?}, closed form {want:?}"
        );
    }
}

fn main() {
    let speed = 10.0_f32;
    let ground = Ground;
    let wind = SdfWindField::new(&ground, [1.0, 0.0, 0.0], speed);
    assert_eq!(wind.decay_scale_m, 5.0, "documented default ramp");

    for (y, want_x) in [
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 2.0),
        (2.5, 5.0),
        (5.0, 10.0),
        (20.0, 10.0),
    ] {
        let got = wind.sample([3.0, y, -7.0]);
        println!("[sdf_wind] y = {y:>5}: wind = {got:?} (closed form [{want_x}, 0, 0])");
        close(got, [want_x, 0.0, 0.0], "ground shelter");
    }

    // Direction is carried through unchanged: a diagonal wind keeps its axis.
    let s = core::f32::consts::FRAC_1_SQRT_2;
    let diagonal = SdfWindField::new(&ground, [s, 0.0, s], speed);
    close(
        diagonal.sample([0.0, 2.5, 0.0]),
        [5.0 * s, 0.0, 5.0 * s],
        "diagonal",
    );

    let boulder = Boulder;
    let around = SdfWindField::new(&boulder, [0.0, 0.0, 1.0], speed);
    close(
        around.sample([0.0, 3.0, 0.0]),
        [0.0, 0.0, 2.0],
        "1 m above the boulder",
    );
    close(
        around.sample([4.0, 0.0, 0.0]),
        [0.0, 0.0, 4.0],
        "2 m beside the boulder",
    );
    close(
        around.sample([0.0, 0.0, -9.0]),
        [0.0, 0.0, 10.0],
        "7 m upstream: full speed",
    );
    close(
        around.sample([0.5, 0.0, 0.0]),
        [0.0, 0.0, 0.0],
        "inside the boulder",
    );

    println!("[sdf_wind] shelter ramp matches clamp(d / 5, 0, 1) * 10 m/s");
}
