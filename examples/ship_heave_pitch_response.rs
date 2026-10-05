//! Ship Heave and Pitch Response Example
//!
//! Production entry point for the 2-DOF ship state of `src/wave_ship.rs`:
//! `ShipResponse` and `ShipResponse::advance`, which `wave_ship_spectrum.rs`
//! does not use.
//!
//! `advance` is the semi-implicit Euler step of two decoupled oscillators
//! `m z̈ + c ż + k z = F` (heave) and `I θ̈ + c_θ θ̇ + k_θ θ = M` (pitch):
//! the velocity is updated first and the position with the new velocity.
//! Every value this example prints is checked against a closed form of that
//! discrete map, computed independently of the function under test:
//! - one step from `(z, v)`: `v' = v + dt (F − c v − k z) / m`, `z' = z + dt v'`
//! - undamped and unforced, the step conserves
//!   `Ĩ = v² + ω² z² − ω² dt z v` with `ω² = k / m` exactly (the modified
//!   energy of the symplectic Euler map), while the plain energy
//!   `v² + ω² z²` oscillates
//! - damped under a constant force the fixed point is `z = F / k`, `v = 0`,
//!   reached geometrically
//! - the two DOFs are decoupled: a pitch moment leaves heave at zero
//! - a non-positive mass or inertia leaves that DOF untouched
//!
//! Run with: `cargo run --example ship_heave_pitch_response`

use alice_physics::math::Fix128;
use alice_physics::wave_ship::ShipResponse;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol,
        "{what}: got {g:.15}, closed form {want:.15} (tolerance {tol:.1e})"
    );
}

/// Heave-only step with the pitch DOF switched off (zero inertia).
fn heave_step(s: &mut ShipResponse, force: f64, m: f64, k: f64, c: f64, dt: f64) {
    s.advance(
        fx(force),
        Fix128::ZERO,
        fx(m),
        Fix128::ZERO,
        fx(k),
        Fix128::ZERO,
        fx(c),
        Fix128::ZERO,
        fx(dt),
    );
}

fn one_step() {
    let (m, k, c, f, dt) = (4.0, 16.0, 8.0, 32.0, 1.0 / 16.0);
    let (z0, v0) = (0.5, -0.25);
    let mut s = ShipResponse {
        heave_m: fx(z0),
        heave_velocity_m_per_s: fx(v0),
        ..ShipResponse::default()
    };
    heave_step(&mut s, f, m, k, c, dt);
    let v1 = v0 + dt * (f - c * v0 - k * z0) / m;
    let z1 = z0 + dt * v1;
    close(s.heave_velocity_m_per_s, v1, 0.0, "one step: velocity");
    close(
        s.heave_m,
        z1,
        0.0,
        "one step: position with the new velocity",
    );
    assert_eq!(s.pitch_rad, Fix128::ZERO, "pitch switched off");
    println!("one step: z {z0} -> {z1}, v {v0} -> {v1}");
}

fn modified_energy() {
    let (m, k, dt) = (4.0, 16.0, 1.0 / 16.0);
    let w2 = k / m;
    let invariant = |s: &ShipResponse| {
        let (z, v) = (s.heave_m.to_f64(), s.heave_velocity_m_per_s.to_f64());
        v * v + w2 * z * z - w2 * dt * z * v
    };
    let mut s = ShipResponse {
        heave_m: Fix128::ONE,
        ..ShipResponse::default()
    };
    let start = invariant(&s);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for _ in 0..2000 {
        heave_step(&mut s, 0.0, m, k, 0.0, dt);
        let (z, v) = (s.heave_m.to_f64(), s.heave_velocity_m_per_s.to_f64());
        let plain = v * v + w2 * z * z;
        lo = lo.min(plain);
        hi = hi.max(plain);
        assert!(
            (invariant(&s) - start).abs() <= 1e-12,
            "the modified energy drifted to {} from {start}",
            invariant(&s)
        );
    }
    // The plain energy is not conserved by the discrete map: it swings by
    // about ω dt of its value, so this is not a vacuous check of a frozen state.
    assert!(
        hi - lo > 0.1 * start * (w2.sqrt() * dt),
        "the plain energy should oscillate, it stayed in [{lo}, {hi}]"
    );
    println!(
        "undamped: modified energy {start} conserved over 2000 steps, plain in [{lo:.6}, {hi:.6}]"
    );
}

fn damped_equilibrium() {
    let (m, k, c, f, dt) = (4.0, 16.0, 8.0, 32.0, 1.0 / 16.0);
    let mut s = ShipResponse::default();
    for _ in 0..2000 {
        heave_step(&mut s, f, m, k, c, dt);
    }
    close(s.heave_m, f / k, 1e-12, "damped heave settles at F / k");
    close(s.heave_velocity_m_per_s, 0.0, 1e-12, "and stops");
    println!(
        "damped: heave settles at {} (F/k = {})",
        s.heave_m.to_f64(),
        f / k
    );
}

fn pitch_and_guards() {
    let (inertia, k, c, moment, dt) = (2.0, 8.0, 4.0, 2.0, 1.0 / 16.0);
    let mut s = ShipResponse::default();
    for _ in 0..2000 {
        s.advance(
            Fix128::ZERO,
            fx(moment),
            Fix128::ONE,
            fx(inertia),
            Fix128::ONE,
            fx(k),
            Fix128::ONE,
            fx(c),
            fx(dt),
        );
    }
    close(s.pitch_rad, moment / k, 1e-12, "pitch settles at M / k_θ");
    close(s.pitch_velocity_rad_per_s, 0.0, 1e-12, "and stops");
    assert_eq!(s.heave_m, Fix128::ZERO, "no heave force, no heave");
    assert_eq!(s.heave_velocity_m_per_s, Fix128::ZERO);

    let before = ShipResponse {
        heave_m: Fix128::ONE,
        heave_velocity_m_per_s: Fix128::ONE,
        pitch_rad: Fix128::ONE,
        pitch_velocity_rad_per_s: Fix128::ONE,
    };
    let mut after = before;
    after.advance(
        fx(100.0),
        fx(100.0),
        Fix128::ZERO,
        fx(-1.0),
        fx(k),
        fx(k),
        fx(c),
        fx(c),
        fx(dt),
    );
    assert_eq!(
        after, before,
        "zero mass and negative inertia leave the state alone"
    );
    println!(
        "pitch: settles at {} (M/k = {})",
        s.pitch_rad.to_f64(),
        moment / k
    );
}

fn main() {
    one_step();
    modified_energy();
    damped_equilibrium();
    pitch_and_guards();
    println!("all closed forms hold");
}
