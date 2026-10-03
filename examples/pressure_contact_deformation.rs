//! Contact-force deformation production entry point for
//! `alice_physics::pressure`: `PressureModifier::{apply_pressure_at,
//! apply_impact, pressure_at, deformation_at}`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `src/pressure.rs`'s own `#[cfg(test)]` module already exercises
//! `apply_impact`/`deformation_at` and `apply_pressure_at` indirectly
//! through `update`, but tests do not count as production callers for the
//! wiring guard, and nothing in `src/` / `examples/` / `benches/` called
//! any of the four before this file existed. This example is that caller.
//!
//! # What this module actually computes
//!
//! `src/pressure.rs` is **not** the CFD pressure-projection solver
//! (`alice_physics::cfd_solver::CfdSolver::step_with_pressure_solver`,
//! wired by `examples/pressure_solvers.rs`) and not an ideal-gas /
//! hydrostatic pressure law. It is a contact-force surface-deformation
//! modifier: `apply_pressure_at` splats a transient "contact pressure"
//! value into a `ScalarField3D` (think: force spread over a contact
//! patch); `apply_impact` computes an *immediate* permanent dent
//! (`dent_depth = impulse * deformation_rate`, clamped to
//! `max_deformation`) and splats that into a separate `deformation`
//! field; `pressure_at`/`deformation_at` trilinearly sample those two
//! fields. The closed form this example checks is therefore not a
//! textbook physical law but the module's own documented splat/clamp
//! arithmetic (`src/pressure.rs` lines 92-103), evaluated independently
//! in plain `f32` below rather than by calling the functions under test.
//!
//! # Why every query point here lands exactly on a grid node
//!
//! `ScalarField3D::splat` deposits a smoothstep-weighted value into every
//! grid cell within `radius` of the splat center, and `sample` trilinearly
//! interpolates between the 8 cells surrounding the query point. Both
//! steps are easy to get bit-approximate but hard to get bit-exact. This
//! example sidesteps that by choosing a `5^3` grid over `[-2,2]^3`
//! (`cell_size == 1.0` on every axis) and always splatting/sampling at
//! world-space `(0,0,0)`, which is exactly grid node `(2,2,2)`:
//!
//! * `sample(0,0,0)` has `fx == fy == fz == 0.0`, so the trilinear
//!   formula's `mul_add(d[hi] - d[lo], f, d[lo])` collapses to exactly
//!   `d[lo]` (multiplying by `f == 0.0` is exact in IEEE 754) -- the
//!   returned value is the stored grid value, not an interpolated one.
//! * With `radius == 0.5 < cell_size == 1.0`, every grid node other than
//!   the center is at distance `>= 1.0 > radius`, so `splat`'s
//!   `dist_sq < radius_sq` test excludes every cell but the center.
//! * At the center itself `dist == 0.0`, so `splat`'s smoothstep weight
//!   `t * t * (3 - 2*t)` with `t = 1 - dist/radius = 1.0` evaluates to
//!   exactly `1.0 * 1.0 * (3.0 - 2.0) == 1.0`.
//!
//! So `pressure_at(0,0,0)` after one `apply_pressure_at(0,0,0,force,0.5)`
//! on a fresh field is exactly `force`, and `deformation_at(0,0,0)` after
//! one `apply_impact(0,0,0,impulse,0.5)` is exactly
//! `(impulse * deformation_rate).min(max_deformation)` -- both verified
//! below against independently written `f32` arithmetic, not by calling
//! the function under test.
//!
//! `tests/analytic_pressure_wiring.rs` holds the full closed-form oracle
//! suite (unclamped/clamped `apply_impact`, zero-radius and zero-magnitude
//! boundaries, and `f32`-overflow extreme magnitudes); this file is a
//! smaller illustrative run of the same two functions for `cargo run`.
//!
//! ```bash
//! cargo run --example pressure_contact_deformation --features std
//! ```

use alice_physics::pressure::{PressureConfig, PressureModifier};

/// Exact-match assertion (every value in this file is representable in
/// `f32` without rounding, per the grid-node argument in the module doc
/// comment above), with a tiny epsilon only for the one case that
/// involves a non-terminating decimal (`2.0 * 0.05`).
fn assert_close(got: f32, want: f32, tol: f32, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= tol,
        "[pressure] MISMATCH {what}: got {got}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
    println!("[pressure] ok {what}: got {got}, want {want} (abs err {err:.2e})");
}

/// A fresh 5^3 grid over `[-2,2]^3`: `cell_size == (4.0)/(5-1) == 1.0` on
/// every axis, so world-space `(0,0,0)` is exactly grid node `(2,2,2)`.
fn fresh_modifier(config: PressureConfig) -> PressureModifier {
    PressureModifier::new(config, 5, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
}

fn main() {
    // ------------------------------------------------------------------
    // 1. apply_pressure_at + pressure_at -- splat a contact-force value
    //    at a grid node and read it back. Weight at the center is exactly
    //    1.0 (derived in the module doc comment above), so the sampled
    //    pressure must equal the splatted force exactly.
    // ------------------------------------------------------------------
    let mut m1 = fresh_modifier(PressureConfig::default());
    let force = 37.5_f32;
    m1.apply_pressure_at(0.0, 0.0, 0.0, force, 0.5);
    let got_pressure = m1.pressure_at(0.0, 0.0, 0.0);
    assert_close(
        got_pressure,
        force,
        0.0,
        "pressure_at(splat center) == splatted force exactly",
    );

    // Off-center, still within bounds but far from the splat: must read
    // back exactly the field's initial zero (nothing leaked outside the
    // splat radius).
    let got_far = m1.pressure_at(2.0, 2.0, 2.0);
    assert_close(
        got_far,
        0.0,
        0.0,
        "pressure_at(far corner) == 0.0 (unsplatted)",
    );

    // ------------------------------------------------------------------
    // 2. apply_impact + deformation_at -- unclamped case. Default config:
    //    deformation_rate = 0.05, max_deformation = 1.0.
    //    dent_depth = impulse * deformation_rate = 2.0 * 0.05 = 0.1,
    //    which is below max_deformation, so clamped == dent_depth exactly
    //    (the `.min()` is a no-op here).
    // ------------------------------------------------------------------
    let mut m2 = fresh_modifier(PressureConfig::default());
    let impulse = 2.0_f32;
    let expected_dent = impulse * PressureConfig::default().deformation_rate;
    m2.apply_impact(0.0, 0.0, 0.0, impulse, 0.5);
    let got_deform = m2.deformation_at(0.0, 0.0, 0.0);
    assert_close(
        got_deform,
        expected_dent,
        1e-6,
        "deformation_at(unclamped impact) == impulse * deformation_rate",
    );

    // ------------------------------------------------------------------
    // 3. apply_impact -- clamped case. impulse = 1000.0 drives
    //    dent_depth = 50.0, far past max_deformation = 1.0, so
    //    `.min(max_deformation)` must clamp the splatted value to exactly
    //    1.0.
    // ------------------------------------------------------------------
    let mut m3 = fresh_modifier(PressureConfig::default());
    m3.apply_impact(0.0, 0.0, 0.0, 1000.0, 0.5);
    let got_clamped = m3.deformation_at(0.0, 0.0, 0.0);
    assert_close(
        got_clamped,
        1.0,
        0.0,
        "deformation_at(clamped impact) == max_deformation exactly",
    );

    println!(
        "[pressure] all 4 production entry points (apply_pressure_at, apply_impact, \
         pressure_at, deformation_at) verified against hand-derived splat/clamp arithmetic"
    );
}
