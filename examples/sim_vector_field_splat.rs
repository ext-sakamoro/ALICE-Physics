//! A gust deposited into a `VectorField3D`, sampled, decayed and cleared
//!
//! Reaches `VectorField3D::new`, `splat`, `sample`, `decay`, `clear` and
//! `ScalarField3D::contains`, `max_value`, `clear`.
//!
//! Closed forms, from the documented formulas (independent of the code):
//! - `splat` adds `value * w(d)` to every node within `radius`, with the
//!   smoothstep falloff `t = 1 - d / r`, `w = t^2 (3 - 2 t)`. With `r = 3/2`
//!   on a unit grid: centre `w = 1`, face neighbour (`d = 1`, `t = 1/3`)
//!   `w = 7/27`, edge neighbour (`d = √2`) `w = t^2 (3 - 2t)` with
//!   `t = 1 - √2 / 1.5`, corner neighbour (`d = √3 > r`) `w = 0`
//! - `sample` is trilinear, so halfway between the centre and a face
//!   neighbour it reads `(1 + 7/27) / 2` of the splatted vector
//! - `decay(rate, dt)` multiplies by `e^(-rate dt)`; `rate dt = 1` gives `1/e`
//!
//! Run with: `cargo run --example sim_vector_field_splat`

use alice_physics::{ScalarField3D, VectorField3D};

fn close3(got: (f32, f32, f32), want: (f64, f64, f64), what: &str) {
    for (g, w) in [(got.0, want.0), (got.1, want.1), (got.2, want.2)] {
        assert!(
            (f64::from(g) - w).abs() <= 1e-5 * (1.0 + w.abs()),
            "{what}: got {got:?}, closed form {want:?}"
        );
    }
}

fn main() {
    let v = (1.0_f64, -2.0_f64, 0.5_f64);
    let r = 1.5_f64;
    let smooth = |d: f64| {
        let t = 1.0 - d / r;
        if t <= 0.0 {
            0.0
        } else {
            t * t * (3.0 - 2.0 * t)
        }
    };
    let scaled = |w: f64| (v.0 * w, v.1 * w, v.2 * w);

    // 5 x 5 x 5 nodes over [0, 4]^3: unit spacing, centre node at (2, 2, 2).
    let mut wind = VectorField3D::new(5, 5, 5, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0));
    close3(
        wind.sample(2.0, 2.0, 2.0),
        (0.0, 0.0, 0.0),
        "starts at rest",
    );

    wind.splat(2.0, 2.0, 2.0, v.0 as f32, v.1 as f32, v.2 as f32, r as f32);

    assert!(
        (smooth(1.0) - 7.0 / 27.0).abs() < 1e-15,
        "hand value of w(1)"
    );
    close3(wind.sample(2.0, 2.0, 2.0), scaled(1.0), "centre");
    close3(
        wind.sample(3.0, 2.0, 2.0),
        scaled(7.0 / 27.0),
        "face neighbour",
    );
    close3(
        wind.sample(3.0, 3.0, 2.0),
        scaled(smooth(2.0_f64.sqrt())),
        "edge neighbour",
    );
    close3(
        wind.sample(3.0, 3.0, 3.0),
        scaled(0.0),
        "corner is outside r",
    );
    close3(
        wind.sample(4.0, 2.0, 2.0),
        scaled(0.0),
        "distance 2 is outside r",
    );
    close3(
        wind.sample(2.5, 2.0, 2.0),
        scaled((1.0 + 7.0 / 27.0) / 2.0),
        "trilinear midpoint",
    );
    println!(
        "[sim_field] gust at centre {:?}, face neighbour {:?}",
        wind.sample(2.0, 2.0, 2.0),
        wind.sample(3.0, 2.0, 2.0)
    );

    // The x component is a ScalarField3D: its peak is the centre value.
    assert!((f64::from(wind.x.max_value()) - v.0).abs() < 1e-6);
    assert!(wind.x.contains(0.0, 4.0, 2.0), "bounds are inclusive");
    assert!(!wind.x.contains(4.01, 2.0, 2.0), "outside +x");
    assert!(!wind.x.contains(2.0, -0.01, 2.0), "outside -y");

    wind.decay(1.0, 1.0);
    let inv_e = 1.0 / core::f64::consts::E;
    close3(wind.sample(2.0, 2.0, 2.0), scaled(inv_e), "decay by e^-1");

    wind.clear();
    close3(wind.sample(2.0, 2.0, 2.0), (0.0, 0.0, 0.0), "cleared");
    assert_eq!(wind.y.max_value(), 0.0, "every node is zero after clear");

    let mut heat = ScalarField3D::new_filled(3, 3, 3, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 5.0);
    assert_eq!(heat.max_value(), 5.0);
    heat.clear();
    assert_eq!(heat.max_value(), 0.0);

    println!("[sim_field] splat weights, trilinear sample, decay and clear match the closed forms");
}
