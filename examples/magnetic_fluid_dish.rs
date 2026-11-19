//! The surface of a ferrofluid in a dish under a magnet, checked against the closed form.
//!
//! Drives `electromagnetic::magnetic_fluid_surface_rise` (and through it
//! `magnetic_fluid_pressure`) from a production entry point. A dish 120 mm across holds a
//! liquid of susceptibility χ = 1, saturation 4e5 A/m and density 1200 kg/m³. A magnet of
//! moment 2 A·m² (pointing up) is held 20 mm over the centre of the surface. Along a line
//! through the centre the example prints the rise of the surface over its level with no
//! field, then shifts the whole profile so the liquid's volume (here, the mean height along
//! the line) is kept, as a closed dish does. On the axis `|B| = μ0/4π · 2m/d³ = 0.05 T`, so
//! the rise there is `χ|B|²/(2μ0 ρ g) ≈ 84.5 mm`.
//!
//! A second magnet 30 mm away adds its field before the square: midway between the two their
//! sideways fields cancel and the upward ones add, so the rise there is not the two rises added
//! (here it is less: the sideways part of each field is the larger at that point).
//!
//! ```bash
//! cargo run --release --example magnetic_fluid_dish --features std
//! ```

use alice_physics_core::electromagnetic::{
    dipole_sample, magnetic_fluid_surface_rise, MagneticResponse,
};
use alice_physics_core::math::{Fix128, Vec3Fix};

fn mm(v: i64) -> Fix128 {
    Fix128::from_ratio(v, 1000)
}

fn field_at(magnets: &[i64], x_mm: i64) -> Vec3Fix {
    magnets
        .iter()
        .map(|&mx| {
            dipole_sample(
                Vec3Fix::new(mm(mx), Fix128::ZERO, mm(20)),
                Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(2)),
                Vec3Fix::new(mm(x_mm), Fix128::ZERO, Fix128::ZERO),
            )
            .field
        })
        .fold(Vec3Fix::ZERO, |a, b| a + b)
}

fn main() {
    let liquid = MagneticResponse {
        chi_eff: Fix128::ONE,
        saturation_magnetization: Fix128::from_int(400_000),
    };
    let (rho, g) = (Fix128::from_int(1200), Fix128::from_ratio(981, 100));
    let rise = |magnets: &[i64], x: i64| {
        magnetic_fluid_surface_rise(field_at(magnets, x), &liquid, rho, g)
            .expect("a liquid with weight")
            .to_f64()
    };

    let xs: Vec<i64> = (-60..=60).step_by(5).collect();
    let free: Vec<f64> = xs.iter().map(|&x| rise(&[0], x)).collect();
    let mean = free.iter().sum::<f64>() / free.len() as f64;
    println!("one magnet 20 mm over the centre: rise over the level (mm), then in a closed dish");
    for (x, h) in xs.iter().zip(&free) {
        println!("  x {x:>4} mm  {:>8.3}  {:>8.3}", h * 1e3, (h - mean) * 1e3);
    }

    let mu0 = 4.0 * std::f64::consts::PI * 1e-7;
    let b = 1e-7 * 2.0 * 2.0 / (0.02 * 0.02 * 0.02);
    let want = b * b / (2.0 * mu0 * 1200.0 * 9.81);
    let got = rise(&[0], 0);
    println!(
        "centre: {:.6} mm, closed form {:.6} mm",
        got * 1e3,
        want * 1e3
    );
    assert!(
        (got - want).abs() <= 1e-10 * want,
        "the centre is the closed form"
    );

    let (a, b2, both) = (rise(&[-15], 0), rise(&[15], 0), rise(&[-15, 15], 0));
    println!(
        "two magnets 30 mm apart, midway: {:.3} mm (each alone {:.3} + {:.3} = {:.3} mm)",
        both * 1e3,
        a * 1e3,
        b2 * 1e3,
        (a + b2) * 1e3
    );
    let (fa, fb) = (field_at(&[-15], 0), field_at(&[15], 0));
    let sum = fa + fb;
    let want_both = sum.length_squared().to_f64() / (2.0 * mu0 * 1200.0 * 9.81);
    assert!(
        (both - want_both).abs() <= 1e-10 * want_both,
        "the rise of the summed field"
    );
    assert!(
        (both - (a + b2)).abs() > 0.1 * (a + b2),
        "not the rises added"
    );
}
