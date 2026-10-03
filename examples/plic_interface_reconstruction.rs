//! PLIC (piecewise-linear interface calculation) reconstruction of a VOF ball.
//!
//! For every mixed cell of a volume-of-fluid field of a sphere the example
//! rebuilds the interface plane with `plic_normal` + `plic_plane_offset`, then
//! measures the fluid volume that plane really encloses with
//! `truncated_cube_volume`. The plane is right when that volume equals the cell's
//! own `f dx^3`, and the normal is right when it lines up with the radius.
//!
//! References: Youngs (1982) for the normal, Scardovelli & Zaleski (2000) for the
//! plane position.
//!
//! ```bash
//! cargo run --release --example plic_interface_reconstruction --features std
//! ```

use alice_physics::interface_capture::{plic_normal, plic_plane_offset, truncated_cube_volume};
use alice_physics::math::Fix128;
use alice_physics::multiphase::Grid3d;

fn main() {
    let n = 21usize;
    let dx = 0.25f64;
    let c = 10.0 * dx;
    let r = 6.0 * dx;
    let mut vof = Grid3d::new(n, n, n, Fix128::from_f64(dx), Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64 * dx - c, j as f64 * dx - c, k as f64 * dx - c);
                let phi = (x * x + y * y + z * z).sqrt() - r;
                vof.set(i, j, k, Fix128::from_f64((0.5 - phi / dx).clamp(0.0, 1.0)));
            }
        }
    }

    let dx_fix = vof.dx;
    let dx3 = dx * dx * dx;
    let (mut mixed, mut worst_volume, mut worst_cos) = (0usize, 0.0f64, 1.0f64);
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let f = vof.get(i, j, k);
                if f <= Fix128::ZERO || f >= Fix128::ONE {
                    continue;
                }
                let normal = plic_normal(&vof, i, j, k);
                let d = plic_plane_offset(normal, f, dx_fix);
                let enclosed = truncated_cube_volume(normal, d, dx_fix).to_f64();
                worst_volume = worst_volume.max((enclosed - f.to_f64() * dx3).abs() / dx3);
                let (x, y, z) = (i as f64 * dx - c, j as f64 * dx - c, k as f64 * dx - c);
                let l = (x * x + y * y + z * z).sqrt();
                let cos =
                    (normal.x.to_f64() * x + normal.y.to_f64() * y + normal.z.to_f64() * z) / l;
                worst_cos = worst_cos.min(cos);
                mixed += 1;
            }
        }
    }
    println!("mixed cells: {mixed}");
    println!("worst |enclosed - f dx^3| / dx^3 = {worst_volume:.3e}");
    println!("worst n . r_hat = {worst_cos:.4}");
    assert!(mixed > 100);
    assert!(
        worst_volume < 1e-6,
        "plane must hold the cell's fluid volume"
    );
    assert!(worst_cos > 0.9, "normal must follow the radius");
}
