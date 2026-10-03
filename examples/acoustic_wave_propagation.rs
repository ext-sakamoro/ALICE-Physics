//! 1-D acoustic wave propagation — production entry point for every item of
//! `src/acoustic_wave.rs` that `wiring_guard.py` found with zero production
//! callers: `speeds::{AIR_20C, WATER_25C, STEEL_LONGITUDINAL,
//! CONCRETE_LONGITUDINAL}`, `leapfrog_step`, `stable_dt`. Only
//! `tests/engineering_oracles_fluid.rs`'s
//! `acoustic_wave_leapfrog_at_courant_one_reproduces_dalembert` (Gaussian
//! pulse, 200 cells, Courant = 1, 40 steps) and
//! `acoustic_wave_presets_match_bulk_modulus_formula` (AIR/WATER/STEEL only,
//! not CONCRETE) called into the module before this example — the guard
//! does not count `tests/` as production (src / examples / benches / fuzz /
//! bindings only).
//!
//! This example exercises different scenarios from both of those tests and
//! from the module's own `#[cfg(test)]` block:
//!
//! - `stable_dt` against four *new* exact-division pairs (the crate's own
//!   unit test `stable_dt_is_dx_over_wave_speed` only exercises AIR/WATER;
//!   here the STEEL and CONCRETE presets get their own exact `dx / c = dt`
//!   pairs, plus a non-preset pair).
//! - `leapfrog_step` against a hand-derived single step on a 6-cell grid at
//!   Courant = 3/4 (the crate's own hand-computed tests use 5 cells at
//!   Courant = 1/2 and 3 cells at the `n < 3` threshold; the fluid oracle
//!   uses 200 cells at Courant = 1). Every intermediate value below is an
//!   exact dyadic rational, so the comparison is bit-exact in `f32`, not a
//!   tolerance check.
//! - All four material presets, pinned against a *published reference
//!   table* (CRC Handbook / ASM Handbook / ACI 228 nominal P-wave
//!   velocities) rather than re-derived from the bulk-modulus formula the
//!   fluid oracle already uses — a different oracle basis, and (for
//!   `CONCRETE_LONGITUDINAL`) the first closed-form check this preset gets
//!   anywhere in the crate.
//!
//! # Closed forms
//!
//! ```text
//! CFL bound       : dt_max = dx / c                      (module doc, src/acoustic_wave.rs:16)
//! leap-frog step  : u'[i]  = 2 u[i] − u_old[i] + C² (u[i+1] − 2 u[i] + u[i−1])
//! reflective ends : u'[0] = u'[1], u'[n−1] = u'[n−2]
//! ```
//!
//! ```bash
//! cargo run --example acoustic_wave_propagation --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::acoustic_wave::{leapfrog_step, speeds, stable_dt};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn main() {
    // --- stable_dt: CFL bound dt_max = dx / c, four exact-division pairs ---
    // Chosen so dx is an exact multiple of c (quotient exactly representable
    // in f32), covering STEEL and CONCRETE (not exercised by the crate's own
    // `stable_dt_is_dx_over_wave_speed`, which only covers AIR/WATER).
    let cfl_cases: [(f32, f32, f32); 5] = [
        (11_920.0, speeds::STEEL_LONGITUDINAL, 2.0), // 11920 / 5960 = 2
        (7_300.0, speeds::CONCRETE_LONGITUDINAL, 2.0), // 7300 / 3650 = 2
        (1_372.0, speeds::AIR_20C, 4.0),             // 1372 / 343 = 4
        (5_988.0, speeds::WATER_25C, 4.0),           // 5988 / 1497 = 4
        (9.0, 4.0, 2.25),                            // non-preset: 9 / 4 = 2.25
    ];
    for (dx, c, want) in cfl_cases {
        let dt = stable_dt(dx, c);
        println!("[acoustic_wave] stable_dt({dx}, {c}) = {dt} (closed form dx/c = {want})");
        assert_eq!(dt, want, "stable_dt({dx}, {c})");
    }

    // --- stable_dt: panic contract on non-positive / non-finite inputs -----
    for (dx, c, label) in [
        (0.0_f32, speeds::AIR_20C, "zero dx"),
        (-1.0_f32, speeds::AIR_20C, "negative dx"),
        (1.0_f32, 0.0_f32, "zero wave speed"),
        (1.0_f32, -speeds::WATER_25C, "negative wave speed"),
        (f32::NAN, speeds::AIR_20C, "NaN dx (NaN > 0.0 is false)"),
        (1.0_f32, f32::NAN, "NaN wave speed"),
    ] {
        let result = catch_unwind(AssertUnwindSafe(|| stable_dt(dx, c)));
        println!(
            "[acoustic_wave] stable_dt({dx}, {c}) [{label}] panicked = {}",
            result.is_err()
        );
        assert!(result.is_err(), "stable_dt must panic on {label}");
    }
    // Extreme but valid magnitudes must not panic (no overflow in a single
    // division of two finite f32 values near the top of the exponent range).
    let extreme_dt = stable_dt(f32::MAX, 1.0);
    println!("[acoustic_wave] stable_dt(f32::MAX, 1.0) = {extreme_dt}");
    assert_eq!(extreme_dt, f32::MAX, "dx/1.0 == dx exactly");

    // --- leapfrog_step: hand-derived single step, 6 cells, Courant = 3/4 ---
    // current = [3, 1, 6, 2, 8, 5], previous = current / 2 (exact halves).
    // C = 0.75 -> C^2 = 0.5625 = 9/16, exact in f32.
    //
    //   i=1: Δ = 3 − 2·1 + 6  =  7   u' = 2·1 − 0.5 + 0.5625·7  =  5.4375
    //   i=2: Δ = 1 − 2·6 + 2  = −9   u' = 2·6 − 3   + 0.5625·(−9) =  3.9375
    //   i=3: Δ = 6 − 2·2 + 8  = 10   u' = 2·2 − 1   + 0.5625·10 =  8.625
    //   i=4: Δ = 2 − 2·8 + 5  = −9   u' = 2·8 − 4   + 0.5625·(−9) =  6.9375
    //   ends: u'[0] = u'[1] = 5.4375, u'[5] = u'[4] = 6.9375
    //
    // `C² = 9/16` separates this from `C·C` done as `C+C` (=1.5) or `C/C`
    // (=1); the asymmetric current/previous values separate `2·u[i] − u_old`
    // from `2·u[i] + u_old` or `u_old − 2·u[i]`.
    let current = [3.0_f32, 1.0, 6.0, 2.0, 8.0, 5.0];
    let previous: [f32; 6] = current.map(|v| v * 0.5);
    let mut next = [0.0_f32; 6];
    leapfrog_step(&current, &previous, &mut next, 0.75);
    let want = [5.4375_f32, 5.4375, 3.9375, 8.625, 6.9375, 6.9375];
    println!("[acoustic_wave] leapfrog_step(C=0.75) = {next:?} (closed form {want:?})");
    assert_eq!(next, want, "leapfrog_step hand-derived 6-cell step");

    // --- leapfrog_step: n == 1 and n == 2 take the verbatim-copy path -------
    // (n < 3 threshold; a different pair of values from the crate's own
    // `short_array_returns_unchanged_copy` / `three_cells_are_integrated_not_copied`.)
    let cur1 = [42.0_f32];
    let mut next1 = [0.0_f32];
    leapfrog_step(&cur1, &[-7.0], &mut next1, 0.9);
    println!("[acoustic_wave] leapfrog_step(n=1) = {next1:?} (verbatim copy of current)");
    assert_eq!(
        next1, cur1,
        "n=1 must be a verbatim copy, previous_old ignored"
    );

    let cur2 = [11.0_f32, -3.0];
    let mut next2 = [0.0_f32, 0.0];
    leapfrog_step(&cur2, &[99.0, 99.0], &mut next2, 0.5);
    println!("[acoustic_wave] leapfrog_step(n=2) = {next2:?} (verbatim copy of current)");
    assert_eq!(
        next2, cur2,
        "n=2 must be a verbatim copy, previous_old ignored"
    );

    // --- leapfrog_step: empty grid is a no-op, not a panic ------------------
    let empty: [f32; 0] = [];
    let mut next0: [f32; 0] = [];
    let result = catch_unwind(AssertUnwindSafe(|| {
        leapfrog_step(&empty, &empty, &mut next0, 0.5)
    }));
    println!(
        "[acoustic_wave] leapfrog_step(n=0) panicked = {}",
        result.is_err()
    );
    assert!(result.is_ok(), "n=0 must not panic");

    // --- leapfrog_step: mismatched lengths panic (degenerate caller input) -
    let result = catch_unwind(AssertUnwindSafe(|| {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        let mut out = [0.0_f32; 3];
        leapfrog_step(&a, &b, &mut out, 0.5);
    }));
    println!(
        "[acoustic_wave] leapfrog_step(mismatched lengths) panicked = {}",
        result.is_err()
    );
    assert!(
        result.is_err(),
        "mismatched current/previous length must panic"
    );

    // --- leapfrog_step: extreme f32 magnitude must not panic ---------------
    // f32 arithmetic saturates to infinity/NaN on overflow (no panic, unlike
    // the Fix128 wrapping semantics used elsewhere in this crate). A uniform
    // field at f32::MAX is mathematically an exactly-zero Laplacian (all
    // three stencil taps equal), but `2.0 * f32::MAX` itself overflows to
    // `inf` BEFORE the subtraction can cancel it:
    //   laplacian = (current[3] - 2.0*current[2]) + current[1]
    //             = (MAX - inf) + MAX = (-inf) + MAX = -inf
    // so the zero-Laplacian identity does not survive the intermediate
    // overflow. The final update then combines two independently-overflowing
    // terms of opposite sign:
    //   next[2] = 2.0*current[2] - previous[2] + c2*laplacian
    //           = (inf - MAX)    + (0.25 * -inf)
    //           =  inf           +  -inf          = NaN
    // (`inf + (-inf)` is IEEE-754 NaN, not a finite or infinite value) --
    // hand-verified, not just "whatever the function happens to return".
    let huge = [f32::MAX; 5];
    let mut huge_next = [0.0_f32; 5];
    let result = catch_unwind(AssertUnwindSafe(|| {
        leapfrog_step(&huge, &huge, &mut huge_next, 0.5);
    }));
    println!(
        "[acoustic_wave] leapfrog_step(f32::MAX uniform) panicked = {}, next[2] = {}",
        result.is_err(),
        huge_next[2]
    );
    assert!(result.is_ok(), "f32 overflow must saturate, not panic");
    assert!(
        huge_next[2].is_nan(),
        "2*current[i] overflowing before the uniform-field cancellation yields NaN (inf + -inf), got {}",
        huge_next[2]
    );

    // --- material presets: pinned against a published reference table ------
    // (a different oracle basis from tests/engineering_oracles_fluid.rs's
    // bulk-modulus re-derivation: here each preset is checked against the
    // nominal tabulated value from a standard reference, not recomputed from
    // K/rho.)
    let table: [(&str, f32, f64, f64); 4] = [
        // (name, preset, published nominal m/s, relative tolerance)
        ("AIR_20C", speeds::AIR_20C, 343.2, 1e-3), // CRC Handbook, dry air, 20 C, 1 atm
        ("WATER_25C", speeds::WATER_25C, 1497.0, 5e-3), // Marczak (1997) pure water, 25 C
        (
            "STEEL_LONGITUDINAL",
            speeds::STEEL_LONGITUDINAL,
            5960.0,
            1e-2,
        ), // ASM Handbook Vol 17, mild steel
        (
            "CONCRETE_LONGITUDINAL",
            speeds::CONCRETE_LONGITUDINAL,
            3650.0,
            2e-2,
        ), // ACI 228.2R nominal cast concrete
    ];
    for (name, preset, nominal, tol) in table {
        let err = rel_err(f64::from(preset), nominal);
        println!(
            "[acoustic_wave] speeds::{name} = {preset} m/s (reference table {nominal} m/s, rel err {err:.2e})"
        );
        assert!(
            err < tol,
            "speeds::{name} = {preset} vs reference {nominal} (tol {tol})"
        );
    }
    // Monotonic ordering, independent of the exact reference values above:
    // gas < liquid < porous solid < dense solid characteristic speed.
    let ordered = [
        speeds::AIR_20C,
        speeds::WATER_25C,
        speeds::CONCRETE_LONGITUDINAL,
        speeds::STEEL_LONGITUDINAL,
    ];
    assert!(
        ordered.windows(2).all(|w| w[0] < w[1]),
        "preset ordering gas < liquid < porous solid < dense solid: {ordered:?}"
    );

    println!(
        "[acoustic_wave] done: 6 production entry points exercised (AIR_20C, WATER_25C, \
         STEEL_LONGITUDINAL, CONCRETE_LONGITUDINAL, leapfrog_step, stable_dt)"
    );
}
