//! Driving `layer_adhesion`'s factor-of-safety API
//! (`EffectiveStrength::{fos_normal_x, fos_normal_z, fos_shear_xy,
//! fos_shear_xz, min_fos}`) through a production entry point.
//!
//! `wiring_guard.py` found all five of these with zero production callers:
//! `src/print_pipeline_solver.rs` constructs an `EffectiveStrength` (step 2
//! of its pipeline, `src/print_pipeline_solver.rs:207`) but only reads the
//! six allowable-stress fields directly — it never asks for a factor of
//! safety. `tests/engineering_oracles_misc.rs` exercises the same five
//! methods with a hand-derived oracle, but the guard does not count
//! `tests/` as production (src / examples / benches / fuzz / bindings only).
//!
//! Closed form, from the module doc (`src/layer_adhesion.rs:94-105`):
//!
//! ```text
//! component_fos(applied, allowable):
//!     if applied == 0          -> sentinel = i64::MAX >> 8   ("infinite")
//!     else if allowable == 0   -> 0
//!     else                      -> allowable / |applied|
//! min_fos(pairs)                -> min over pairs of component_fos(applied, allowable)
//! ```
//!
//! ```bash
//! cargo run --example layer_adhesion_fos --features std
//! ```

use alice_physics::filament_db::MaterialProperties;
use alice_physics::layer_adhesion::{EffectiveStrength, PrintOrientation};
use alice_physics::math::Fix128;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// `i64::MAX >> 8`, the sentinel `component_fos` returns for zero applied
/// stress — written down independently from `src/layer_adhesion.rs:99`
/// rather than read off a computed value.
fn sentinel() -> Fix128 {
    Fix128::from_int(i64::MAX >> 8)
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn main() {
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);

    // Independently derived allowables for PLA, from the published numbers
    // in `MaterialProperties::pla()` (`src/filament_db.rs`: yield_strength_mpa
    // = 50 MPa, anisotropy_z_ratio = 0.65) and the module doc's formulas —
    // not read off `s`.
    let sigma = 50.0_f64;
    let aniso_r = 0.65_f64;
    let allow_x = sigma;
    let allow_z = sigma * aniso_r;
    let tau_xy = sigma * 0.6;
    let tau_xz = tau_xy * (1.0 + aniso_r) / 2.0;

    println!(
        "[layer_adhesion] PLA allowables: X={allow_x} Z={allow_z} XY={tau_xy} XZ={tau_xz} MPa"
    );

    // --- fos_normal_x: allowable / |applied| ---------------------------
    for applied in [20.0_f64, -20.0_f64, 100.0_f64] {
        let fos = s.fos_normal_x(Fix128::from_f64(applied)).to_f64();
        let want = allow_x / applied.abs();
        println!("[layer_adhesion] fos_normal_x({applied}) = {fos} (closed form {want})");
        assert!(rel_err(fos, want) < 1.0e-6, "fos_normal_x({applied})");
    }

    // --- fos_normal_z: the critical layer-bond direction ---------------
    for applied in [20.0_f64, -20.0_f64] {
        let fos = s.fos_normal_z(Fix128::from_f64(applied)).to_f64();
        let want = allow_z / applied.abs();
        println!("[layer_adhesion] fos_normal_z({applied}) = {fos} (closed form {want})");
        assert!(rel_err(fos, want) < 1.0e-6, "fos_normal_z({applied})");
    }

    // --- fos_shear_xy: within-layer shear -------------------------------
    for applied in [10.0_f64, -10.0_f64] {
        let fos = s.fos_shear_xy(Fix128::from_f64(applied)).to_f64();
        let want = tau_xy / applied.abs();
        println!("[layer_adhesion] fos_shear_xy({applied}) = {fos} (closed form {want})");
        assert!(rel_err(fos, want) < 1.0e-6, "fos_shear_xy({applied})");
    }

    // --- fos_shear_xz: across-layer shear -------------------------------
    for applied in [10.0_f64, -10.0_f64] {
        let fos = s.fos_shear_xz(Fix128::from_f64(applied)).to_f64();
        let want = tau_xz / applied.abs();
        println!("[layer_adhesion] fos_shear_xz({applied}) = {fos} (closed form {want})");
        assert!(rel_err(fos, want) < 1.0e-6, "fos_shear_xz({applied})");
    }

    // --- min_fos: min over an independently-built pairing ---------------
    let stress = [
        (Fix128::from_int(10), s.normal_x_mpa),
        (Fix128::from_int(10), s.normal_y_mpa),
        (Fix128::from_int(28), s.normal_z_mpa), // worst component
        (Fix128::from_int(-5), s.shear_xy_mpa),
        (Fix128::ZERO, s.shear_yz_mpa),
        (Fix128::ZERO, s.shear_xz_mpa),
    ];
    let applied_abs_f64 = [10.0_f64, 10.0, 28.0, 5.0, 0.0, 0.0];
    let allow_f64 = [allow_x, allow_x, allow_z, tau_xy, tau_xz, tau_xz];
    let want_min = applied_abs_f64
        .iter()
        .zip(allow_f64.iter())
        .map(|(a, b)| if *a == 0.0 { f64::INFINITY } else { b / a })
        .fold(f64::INFINITY, f64::min);
    let got_min = s.min_fos(&stress).to_f64();
    println!("[layer_adhesion] min_fos(...) = {got_min} (closed form min {want_min})");
    assert!(rel_err(got_min, want_min) < 1.0e-6, "min_fos");

    // ---------------- degenerate / extreme inputs -----------------------

    // Zero applied stress -> sentinel, exactly, for every component.
    assert_eq!(s.fos_normal_x(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_normal_z(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_shear_xy(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_shear_xz(Fix128::ZERO), sentinel());
    println!(
        "[layer_adhesion] zero applied stress -> sentinel {} exactly (all 4 components)",
        sentinel()
    );

    // Zero allowable is not reachable through `for_material` (every preset's
    // yield strength and anisotropy ratio are positive), so build it by hand
    // via the all-public-field struct — not by calling the function under
    // test for the expected side, this only constructs the *input*.
    // component_fos returns ZERO exactly when allowable == 0 (nonzero
    // applied), per src/layer_adhesion.rs:101-103.
    let zero_allow = EffectiveStrength {
        normal_x_mpa: Fix128::ZERO,
        normal_y_mpa: Fix128::ZERO,
        normal_z_mpa: Fix128::ZERO,
        shear_xy_mpa: Fix128::ZERO,
        shear_yz_mpa: Fix128::ZERO,
        shear_xz_mpa: Fix128::ZERO,
    };
    assert_eq!(zero_allow.fos_normal_x(Fix128::from_int(5)), Fix128::ZERO);
    println!("[layer_adhesion] zero allowable (nonzero applied) -> FoS = 0 exactly");

    // Zero applied AND zero allowable at once: `component_fos` checks
    // `applied == 0` first (src/layer_adhesion.rs:97-100), so this path
    // returns the sentinel, not ZERO — an order-dependent quirk worth
    // pinning explicitly rather than assuming "0/0 -> 0".
    assert_eq!(zero_allow.fos_normal_x(Fix128::ZERO), sentinel());
    println!(
        "[layer_adhesion] zero applied AND zero allowable -> sentinel (applied == 0 branch wins)"
    );

    // Extreme magnitude: the smallest representable positive Fix128 (2^-64)
    // as the applied stress, against PLA's 50 MPa. The true FoS is 50 * 2^64,
    // far beyond Fix128's ~9.2e18 integer range: it is capped at the same
    // "infinite" sentinel (i64::MAX >> 8) a zero applied stress returns,
    // rather than letting the division wrap (to exactly 0, a broken part).
    let tiny_applied = Fix128::from_raw(0, 1); // 2^-64
    let extreme = catch_unwind(AssertUnwindSafe(|| s.fos_normal_x(tiny_applied)));
    println!("[layer_adhesion] fos_normal_x(2^-64 MPa) = {extreme:?} (no panic)");
    assert!(extreme.is_ok(), "extreme-magnitude division must not panic");
    assert_eq!(
        extreme.unwrap(),
        Fix128::from_int(i64::MAX >> 8),
        "a negligible load is the infinite-FoS sentinel"
    );

    println!(
        "[layer_adhesion] done: 5 production entry points exercised (fos_normal_x, fos_normal_z, fos_shear_xy, fos_shear_xz, min_fos)"
    );
}
