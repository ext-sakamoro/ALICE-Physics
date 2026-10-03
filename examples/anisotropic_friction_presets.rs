//! Production entry point for `alice_physics::anisotropic_friction`:
//! `AnisotropicFriction::{tyre_asphalt, ski_snow, skate_ice}` (the three
//! presets) and `AnisotropicFriction::friction_force`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `src/anisotropic_friction.rs`'s own `#[cfg(test)]` module already
//! exercises all four, but tests do not count as production callers for the
//! wiring guard, and nothing in `src/` / `examples/` / `benches/` called any
//! of them before this file existed. This example is that caller.
//!
//! # Scenario
//!
//! For each of the three presets this drives `friction_force` with a
//! relative velocity purely along the surface's longitudinal tangent axis,
//! then purely along the transverse axis, at a slip speed (10 m/s) above
//! every preset's documented `slip_threshold_m_s` (at most 0.05 m/s for
//! `tyre_asphalt`/`ski_snow`, 0.02 m/s for `skate_ice`) so the kinetic
//! coefficients apply in both cases.
//!
//! On a pure axis, the module doc's friction-ellipse formula
//!
//! ```text
//! F = -N * ( mu_long * v_long * t_long + mu_trans * v_trans * t_trans ) / |v_tan|
//! ```
//!
//! collapses exactly: with `v_trans = 0`, `|v_tan| = |v_long|` and
//! `v_long / |v_tan| = sign(v_long)`, so `F = -N * mu_long * sign(v_long) *
//! t_long` -- plain isotropic Coulomb friction along that one axis, with no
//! contribution from the other. The expected magnitudes below
//! (`N * mu_long_kinetic`, `N * mu_trans_kinetic`) are computed from each
//! preset's own documented coefficient fields, written out again in plain
//! f64 arithmetic -- never by calling `friction_force` itself.
//!
//! `tests/analytic_anisotropic_friction_wiring.rs` holds the off-axis
//! general-angle (45 deg) case, which needs the non-degenerate `sqrt`
//! branch of `friction_force`'s slip-speed computation, plus the
//! zero-velocity and extreme-magnitude-velocity degenerate inputs.
//!
//! ```bash
//! cargo run --example anisotropic_friction_presets --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::math::{Fix128, Vec3Fix};

/// Relative-error check against an independently hand-derived f64 closed
/// form (never computed by calling `friction_force` itself), falling back
/// to an absolute check when the expected value is exactly zero -- per this
/// repo's analytic-oracle-tests discipline
/// (`~/claude-config/rules/analytic-oracle-tests.md`).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[anisotropic_friction] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[anisotropic_friction] ok {what}: got {g:.6}, want {want:.6} (err {err:.2e})");
}

/// A preset under test, paired with its kinetic coefficients written out
/// again by hand from `src/anisotropic_friction.rs`'s doc comments -- the
/// independent reference, not a re-read of the live struct fields.
struct PresetCase {
    name: &'static str,
    friction: AnisotropicFriction,
    mu_long_kinetic: f64,
    mu_trans_kinetic: f64,
}

fn main() {
    let tol = 1e-9;
    let normal_force = Fix128::from_int(100);
    let normal_force_f64 = 100.0_f64;
    // Above every preset's slip_threshold_m_s (max 0.05 m/s), so all three
    // select their kinetic coefficients, not static.
    let speed = Fix128::from_int(10);
    let tangent_long = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let tangent_trans = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);

    let cases = [
        PresetCase {
            name: "tyre_asphalt",
            friction: AnisotropicFriction::tyre_asphalt(),
            mu_long_kinetic: 0.9,
            mu_trans_kinetic: 0.7,
        },
        PresetCase {
            name: "ski_snow",
            friction: AnisotropicFriction::ski_snow(),
            mu_long_kinetic: 0.04,
            mu_trans_kinetic: 0.75,
        },
        PresetCase {
            name: "skate_ice",
            friction: AnisotropicFriction::skate_ice(),
            mu_long_kinetic: 0.015,
            mu_trans_kinetic: 0.7,
        },
    ];

    for case in cases {
        let name = case.name;

        // ------------------------------------------------------------------
        // Pure longitudinal slip: F = -N * mu_long_kinetic * t_long, zero on
        // the transverse axis.
        // ------------------------------------------------------------------
        let v_long = tangent_long * speed;
        let f_long =
            case.friction
                .friction_force(normal_force, tangent_long, tangent_trans, v_long);
        let want_long = -normal_force_f64 * case.mu_long_kinetic;
        assert_rel(
            f_long.x,
            want_long,
            tol,
            &format!("{name} longitudinal slip Fx = -N*mu_long_kinetic"),
        );
        assert_rel(
            f_long.y,
            0.0,
            tol,
            &format!("{name} longitudinal slip Fy = 0"),
        );
        assert_rel(
            f_long.z,
            0.0,
            tol,
            &format!("{name} longitudinal slip Fz = 0 (no transverse leakage)"),
        );

        // ------------------------------------------------------------------
        // Pure transverse slip: F = -N * mu_trans_kinetic * t_trans, zero on
        // the longitudinal axis.
        // ------------------------------------------------------------------
        let v_trans = tangent_trans * speed;
        let f_trans =
            case.friction
                .friction_force(normal_force, tangent_long, tangent_trans, v_trans);
        let want_trans = -normal_force_f64 * case.mu_trans_kinetic;
        assert_rel(
            f_trans.x,
            0.0,
            tol,
            &format!("{name} transverse slip Fx = 0 (no longitudinal leakage)"),
        );
        assert_rel(
            f_trans.y,
            0.0,
            tol,
            &format!("{name} transverse slip Fy = 0"),
        );
        assert_rel(
            f_trans.z,
            want_trans,
            tol,
            &format!("{name} transverse slip Fz = -N*mu_trans_kinetic"),
        );

        println!(
            "[anisotropic_friction] {name} verified: longitudinal leg {want_long:.3} N, \
             transverse leg {want_trans:.3} N"
        );
    }

    println!(
        "[anisotropic_friction] all 3 presets (tyre_asphalt, ski_snow, skate_ice) and \
         friction_force verified against hand-derived friction-ellipse closed forms"
    );
}
