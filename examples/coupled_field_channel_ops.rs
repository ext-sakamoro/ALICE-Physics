//! Point writes, diffusion, relaxation and bounds of a `CoupledField` channel
//!
//! Reaches `CoupledField::add`, `sum`, `min`, `max`, `diffuse`, `gradient`,
//! `decay_toward`, `decay` and `clamp`.
//!
//! Closed forms, each worked by hand from the documented formulas:
//! - `diffuse` is one explicit-Euler step of `dT/dt = rate * laplacian(T)`
//!   with a mirror ghost (`T[-1] = T[1]`). On five nodes of spacing 1 with
//!   `rate * dt = 1/4`, the pulse `[0, 0, 4, 0, 0]` becomes `[0, 1, 2, 1, 0]`
//!   and then `[1/2, 1, 3/2, 1, 1/2]`. The mirror ghost conserves the sum
//!   with the end nodes weighted `1/2` (4 at every step), not the plain sum
//!   (4, then 4.5)
//! - `gradient` of the affine field `T = 3 x - 2 y + 1` is `(3, -2, 0)`
//!   wherever the one-cell stencil stays inside the grid
//! - `decay_toward(target, rate, dt)` is `(v - target) e^(-rate dt) + target`;
//!   with `rate dt = 1`, `30` relaxes toward `10` to `10 + 20 / e`, and
//!   `decay` (target 0) takes `30` to `30 / e`
//! - `clamp(0, 1)` maps `-5 -> 0`, `3/10 -> 3/10`, `7 -> 1`
//!
//! Run with: `cargo run --example coupled_field_channel_ops`

use alice_physics::coupled_field::CoupledField;
use alice_physics::math::{Fix128, Vec3Fix};

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn row(f: &CoupledField) -> Vec<Fix128> {
    (0..f.nx()).map(|i| f.get(i, 0, 0)).collect()
}

fn main() {
    let zero = Fix128::ZERO;
    let half = Fix128::from_ratio(1, 2);

    // ---- bounds and point writes ----------------------------------------
    let mut rod = CoupledField::try_new(5, 1, 1, (zero, zero, zero), (fx(4), zero, zero))
        .expect("non-empty grid");
    assert_eq!(rod.min(), (zero, zero, zero));
    assert_eq!(rod.max(), (fx(4), zero, zero));
    assert_eq!(rod.cell_size().0, Fix128::ONE, "(4 - 0) / (5 - 1)");

    rod.add(2, 0, 0, fx(3));
    rod.add(2, 0, 0, fx(1));
    rod.add(9, 0, 0, fx(100)); // outside the grid: ignored
    assert_eq!(row(&rod), vec![zero, zero, fx(4), zero, zero]);
    assert_eq!(rod.sum(), fx(4), "3 + 1, the out-of-range write is dropped");

    // ---- explicit diffusion with the mirror ghost -----------------------
    let rate = Fix128::ONE;
    let dt = Fix128::from_ratio(1, 4);
    rod.diffuse(dt, rate);
    assert_eq!(row(&rod), vec![zero, fx(1), fx(2), fx(1), zero], "step 1");
    rod.diffuse(dt, rate);
    let step2 = vec![half, fx(1), Fix128::from_ratio(3, 2), fx(1), half];
    assert_eq!(row(&rod), step2, "step 2");
    let v = row(&rod);
    let weighted = (v[0] + v[4]) * half + v[1] + v[2] + v[3];
    assert_eq!(weighted, fx(4), "end-weighted sum is conserved");
    assert_eq!(rod.sum(), Fix128::from_ratio(9, 2), "plain sum is not");
    println!(
        "[coupled_field] diffusion: plain sum {} , end-weighted sum {}",
        rod.sum().to_f64(),
        weighted.to_f64()
    );

    // ---- gradient of an affine field ------------------------------------
    let mut plane = CoupledField::try_new(5, 3, 1, (zero, zero, zero), (fx(4), fx(2), zero))
        .expect("non-empty grid");
    for ix in 0..5 {
        for iy in 0..3 {
            plane.set(ix, iy, 0, fx(3 * ix as i64 - 2 * iy as i64 + 1));
        }
    }
    let g = plane.gradient(Vec3Fix::new(fx(2), fx(1), zero));
    assert_eq!(g, Vec3Fix::new(fx(3), fx(-2), zero), "∇(3x - 2y + 1)");

    // ---- exponential relaxation -----------------------------------------
    let inv_e = 1.0 / core::f64::consts::E;
    let mut hot =
        CoupledField::try_new_filled(2, 1, 1, (zero, zero, zero), (fx(1), zero, zero), fx(30))
            .expect("non-empty grid");
    hot.decay_toward(fx(10), Fix128::ONE, Fix128::ONE);
    let want = 10.0 + 20.0 * inv_e;
    let got = hot.get(0, 0, 0).to_f64();
    println!("[coupled_field] decay_toward: {got:.7} (closed form {want:.7})");
    assert!((got - want).abs() < 1e-5 * want, "decay_toward");

    hot.fill(fx(30));
    hot.decay(Fix128::ONE, Fix128::ONE);
    let want = 30.0 * inv_e;
    let got = hot.get(1, 0, 0).to_f64();
    assert!((got - want).abs() < 1e-5 * want, "decay: {got} vs {want}");

    // ---- clamp ------------------------------------------------------------
    let mut c = CoupledField::try_new(3, 1, 1, (zero, zero, zero), (fx(2), zero, zero))
        .expect("non-empty grid");
    c.set(0, 0, 0, fx(-5));
    c.set(1, 0, 0, Fix128::from_ratio(3, 10));
    c.set(2, 0, 0, fx(7));
    c.clamp(zero, Fix128::ONE);
    assert_eq!(row(&c), vec![zero, Fix128::from_ratio(3, 10), Fix128::ONE]);

    println!("[coupled_field] add / sum / bounds / diffuse / gradient / decay / clamp match the hand-worked values");
}
