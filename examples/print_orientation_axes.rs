//! Load-aligned print orientation: `LoadDirection::{axis_x, axis_y}` and `optimize_grid`.
//!
//! The strength along a load at angle `theta` from the layer stack axis (Z) is
//! `sigma_z cos^2 + sigma_xy sin^2`. A load along X or Y already lies in the layer plane, so the
//! identity orientation is optimal and nothing can be gained (`improvement = 0`); a load along Z
//! pulls across layers, and the grid search finds a quarter-turn about X or Y that moves it into
//! the plane, gaining `sigma_xy - sigma_z`.
//!
//! ```bash
//! cargo run --release --example print_orientation_axes --features std
//! ```

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::print_orientation::{optimize_grid, LoadDirection};

fn main() {
    let pla = MaterialProperties::pla();
    let step = Fix128::PI / Fix128::from_int(12); // 15 degrees
    for (name, load) in [
        ("X", LoadDirection::axis_x()),
        ("Y", LoadDirection::axis_y()),
        ("Z", LoadDirection::axis_z()),
    ] {
        let r = optimize_grid(&load, &pla, step);
        println!(
            "load along {name}: identity {:.3} MPa -> best {:.3} MPa (gain {:.3}), best angle to Z {:.4} rad",
            r.identity_yield_mpa.to_f64(),
            r.effective_yield_mpa.to_f64(),
            r.improvement_mpa.to_f64(),
            r.angle_to_z_axis.to_f64()
        );
    }
    println!(
        "sigma_xy = {:.3}, sigma_z = {:.3}",
        pla.yield_strength_mpa.to_f64(),
        (pla.yield_strength_mpa * pla.anisotropy_z_ratio).to_f64()
    );
}
