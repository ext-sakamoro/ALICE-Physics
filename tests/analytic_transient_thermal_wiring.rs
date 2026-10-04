//! Oracles for the wiring of `transient_thermal`: the four material presets,
//! the five property accessors, the two stability bounds and the four
//! steppers, each against a value derived without calling the module.
//!
//! The cosine-eigenmode decay of the explicit and the Crank–Nicolson step is
//! already pinned in `tests/engineering_oracles.rs` (Carslaw & Jaeger) and is
//! not repeated here; this file covers the closed forms that oracle does not
//! touch.
//!
//! # Closed forms (source of every expected value)
//!
//! * **Properties.** The presets store their fits as data
//!   (`Polynomial { c0, c1, c2 }` ⇒ `c0 + c1 T + c2 T²`,
//!   `Linear { ref_value, ref_temp, coeff }` ⇒ `ref_value (1 + coeff (T − ref_temp))`).
//!   The expected value is that formula evaluated by hand in `f64` from the
//!   preset's own field values; AISI 1018 at 300 K is additionally written
//!   out as literals (`k = 60 − 0.03·300 = 51`, `c_p = 380 + 0.3·300 = 470`,
//!   `ρ = 7870 (1 − 3.6e-5 · 6.85) = 7868.059258`). Heat capacity is
//!   `ρ c_p`, diffusivity `α = k / (ρ c_p)`. There is **no table and no
//!   clamp** in the module: the fits extrapolate beyond the calibrated
//!   range, and for steel the conductivity line crosses zero at 2000 K, so
//!   `diffusivity_at(3000)` is **negative** (only the denominator is guarded).
//!   A `Linear` model at its own `ref_temp` returns `ref_value` bit for bit
//!   (`coeff · 0 = 0`, `ref_value · 1 = ref_value`).
//! * **Stability bounds.** `stable_dt_1d = dx² / (2 max α)` and
//!   `stable_dt_3d = dx² / (6 max α)` over the field; `max α` is the hand
//!   diffusivity at each field temperature, maximised in the test.
//! * **Explicit step, exact arithmetic.** With the unit material
//!   (`k = ρ = c_p = 1`, α = 1) and `dx = 1`, `r = α dt / dx²` is the chosen
//!   `dt` exactly, and all updates `T_i + r (T_{i−1} − 2T_i + T_{i+1})` are
//!   dyadic ⇒ `assert_eq!` on every cell. A single hot node of height `A`
//!   loses `2rA`, each neighbour gains `rA`; a **linear profile** `a + b·i`
//!   is fixed bit for bit on every interior cell (second difference zero)
//!   while the zero-flux ends move by exactly `+r b` (cell 0) and `−r b`
//!   (cell n−1) — the ghost mirror makes the end laplacian `±b`, not 0. The
//!   3-D stencil loses `6rA` at the hot cell and gives `rA` to each of its
//!   six face neighbours.
//! * **Explicit step with a preset.** The update uses the *local*
//!   diffusivity: the hot cell moves by `−2 α(T_hot) dt A / dx²`, each
//!   neighbour by `+α(T_ambient) dt A / dx²`, with the hand α of the preset.
//! * **Crank–Nicolson on 3 cells, hand-solved.** For `[a, a + A, a]` and a
//!   constant `r` the symmetric 3×3 system
//!   `(1 + r/2) p − (r/2) q = rA/2`, `−r p + (1 + r) q = A (1 − r)` has the
//!   solution `q = A (2 − r) / (2 + 3r)` (centre deviation) and
//!   `p = 2rA / (2 + 3r)` (each end), with `2p + q = A` (conservation). At
//!   `r = 2` the centre deviation vanishes in one step.
//! * **Crank–Nicolson is identity on a uniform field** up to the rounding
//!   of the Thomas sweep (`m = −½r / (1 + ½r)` is not dyadic), and the
//!   Picard variant with constant properties equals the linearised step
//!   **bit for bit** (same `r`, same arithmetic) after exactly 2 iterations
//!   (the second iterate repeats the first, `Δ = 0 < tol`); on a
//!   temperature-dependent preset it differs.
//!
//! # Degenerate input (documented behaviour that is asserted)
//!
//! * `dx ≤ 0`: all six grid functions panic with `dx must be positive`.
//! * `dt ≤ 0`: both Crank–Nicolson steps panic with `dt must be positive`;
//!   the explicit steps return the field unchanged (`dt = 0` is identity).
//! * Fewer than 3 cells (0, 1, 2; any 3-D side < 3): returned unchanged, the
//!   Picard driver reports 0 iterations; `stable_dt_*` of an empty field is
//!   `+∞`; a 3-D length mismatch panics.
//! * `max_iterations = 0`: identity, reports 0.
//! * Zero density or zero specific heat: `diffusivity_at = 0`, the bounds
//!   are `+∞`, both steppers are identity bit for bit.
//! * `dt` above the explicit bound: **no refusal and no clamp** — at
//!   `r = 1` a hot node of height `A` flips to `−A` after one step and reaches
//!   `+3A` after two (exact), i.e. the explicit stepper silently amplifies.
//! * `T = f32::MAX`: no panic. The polynomial fits (`c2 = 0.0` in every
//!   preset) stay **finite** because `c2·T·T` evaluates left-to-right as
//!   `(0.0·T)·T = 0.0·T`, short-circuiting before the second multiplication
//!   can overflow; only their *product* `ρ·c_p` overflows to `−∞`, which
//!   `diffusivity_at`'s guard catches and returns `0`. The explicit step on
//!   a uniform `f32::MAX` field still returns `NaN` (the laplacian itself
//!   overflows to `−∞` via `2.0·T`, and `0.0 · (−∞) = NaN`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::transient_thermal::{
    crank_nicolson_step_1d, crank_nicolson_step_1d_nonlinear, stable_dt_1d, stable_dt_3d,
    transient_step_1d, transient_step_3d, TemperatureDependence, ThermalMaterial,
};

// ---------------------------------------------------------------- helpers

/// Hand evaluation of a dependence from its stored data (independent of
/// `TemperatureDependence::evaluate`, in f64).
fn hand(dep: &TemperatureDependence, t: f64) -> f64 {
    match *dep {
        TemperatureDependence::Constant(v) => f64::from(v),
        TemperatureDependence::Polynomial { c0, c1, c2 } => {
            f64::from(c0) + f64::from(c1) * t + f64::from(c2) * t * t
        }
        TemperatureDependence::Linear {
            ref_value,
            ref_temp,
            coeff,
        } => f64::from(ref_value) * (1.0 + f64::from(coeff) * (t - f64::from(ref_temp))),
    }
}

/// Hand `α = k / (ρ c_p)` from the preset's data.
fn hand_alpha(m: &ThermalMaterial, t: f64) -> f64 {
    hand(&m.conductivity, t) / (hand(&m.density, t) * hand(&m.specific_heat, t))
}

fn assert_rel(actual: f32, expected: f64, rel: f64, what: &str) {
    let a = f64::from(actual);
    let err = (a - expected).abs();
    assert!(
        err <= rel * expected.abs(),
        "{what}: got {a}, expected {expected} (|Δ| = {err}, allowed {})",
        rel * expected.abs()
    );
}

/// `k = ρ = c_p = 1` ⇒ `α = 1` exactly; with `dx = 1` the ratio `r` equals `dt`.
fn unit_material() -> ThermalMaterial {
    ThermalMaterial {
        name: "unit",
        conductivity: TemperatureDependence::Constant(1.0),
        specific_heat: TemperatureDependence::Constant(1.0),
        density: TemperatureDependence::Constant(1.0),
        reference_temperature: 300.0,
    }
}

fn presets() -> [ThermalMaterial; 4] {
    [
        ThermalMaterial::steel_1018(),
        ThermalMaterial::aluminum_6061(),
        ThermalMaterial::titanium_ti6al4v(),
        ThermalMaterial::pla_polymer(),
    ]
}

fn panic_message<F: FnOnce()>(f: F) -> Option<String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string()),
        ),
    }
}

// ------------------------------------------------------------- properties

/// AISI 1018 at 300 K written out by hand (not from the enum data).
#[test]
fn steel_1018_properties_at_300k_match_hand_literals() {
    let m = ThermalMaterial::steel_1018();
    assert_eq!(m.name, "steel_1018");
    let k = 60.0 - 0.03 * 300.0; // 51
    let cp = 380.0 + 0.3 * 300.0; // 470
    let rho = 7870.0 * (1.0 - 3.6e-5 * (300.0 - 293.15)); // 7868.059258
    assert!((rho - 7_868.059_258_f64).abs() < 1e-6, "rho literal {rho}");
    assert_rel(m.conductivity_at(300.0), k, 2e-6, "k");
    assert_rel(m.specific_heat_at(300.0), cp, 2e-6, "c_p");
    assert_rel(m.density_at(300.0), rho, 2e-6, "rho");
    assert_rel(m.heat_capacity_at(300.0), rho * cp, 4e-6, "rho c_p");
    assert_rel(m.diffusivity_at(300.0), k / (rho * cp), 6e-6, "alpha");
}

/// Aluminium, titanium and PLA against the published literal constants
/// (not read back from the struct): the `k`/`c_p` `Polynomial` fits
/// collapse to the bare `c0` **at `T = 0`** (`c1·T` and `c2·T²` vanish),
/// and the `Linear` density collapses to `ref_value` **at `T = ref_temp`**
/// (`coeff · 0 = 0`) — two different temperatures, since only density uses
/// the reference-point model. A second point away from those special
/// values exercises the slope (`c1` / `coeff`) literal too.
#[test]
fn aluminum_titanium_pla_properties_match_hand_literals() {
    let al = ThermalMaterial::aluminum_6061();
    assert_rel(al.conductivity_at(0.0), 155.0, 2e-6, "al k(0)");
    assert_rel(al.specific_heat_at(0.0), 780.0, 2e-6, "al c_p(0)");
    assert_rel(al.density_at(293.15), 2700.0, 2e-6, "al rho(ref_temp)");
    assert_rel(
        al.conductivity_at(400.0),
        155.0 + 0.04 * 400.0,
        2e-6,
        "al k @ 400K",
    );
    assert_rel(
        al.specific_heat_at(400.0),
        780.0 + 0.4 * 400.0,
        2e-6,
        "al c_p @ 400K",
    );
    assert_rel(
        al.density_at(400.0),
        2700.0 * (1.0 - 6.9e-5 * (400.0 - 293.15)),
        2e-6,
        "al rho @ 400K",
    );

    let ti = ThermalMaterial::titanium_ti6al4v();
    assert_rel(ti.conductivity_at(0.0), 6.7, 2e-6, "ti k(0)");
    assert_rel(ti.specific_heat_at(0.0), 546.0, 2e-6, "ti c_p(0)");
    assert_rel(ti.density_at(293.15), 4430.0, 2e-6, "ti rho(ref_temp)");
    assert_rel(
        ti.conductivity_at(400.0),
        6.7 + 0.0011 * 400.0,
        2e-6,
        "ti k @ 400K",
    );
    assert_rel(
        ti.specific_heat_at(400.0),
        546.0 + 0.16 * 400.0,
        2e-6,
        "ti c_p @ 400K",
    );
    assert_rel(
        ti.density_at(400.0),
        4430.0 * (1.0 - 2.7e-5 * (400.0 - 293.15)),
        2e-6,
        "ti rho @ 400K",
    );

    let pla = ThermalMaterial::pla_polymer();
    assert_rel(pla.conductivity_at(0.0), 0.13, 2e-6, "pla k(0)");
    assert_rel(pla.specific_heat_at(0.0), 1200.0, 2e-6, "pla c_p(0)");
    assert_rel(pla.density_at(293.15), 1240.0, 2e-6, "pla rho(ref_temp)");
    assert_rel(
        pla.conductivity_at(400.0),
        0.13 + 1.0e-4 * 400.0,
        2e-6,
        "pla k @ 400K",
    );
    assert_rel(
        pla.specific_heat_at(400.0),
        1200.0 + 2.0 * 400.0,
        2e-6,
        "pla c_p @ 400K",
    );
    assert_rel(
        pla.density_at(400.0),
        1240.0 * (1.0 - 1.8e-4 * (400.0 - 293.15)),
        2e-6,
        "pla rho @ 400K",
    );
}

/// Every preset, every accessor, at three temperatures, against the formula
/// evaluated from the preset's own data in f64.
#[test]
fn every_preset_accessor_matches_the_hand_formula() {
    for m in presets() {
        for t in [293.15_f64, 400.0, 650.0] {
            let tf = t as f32;
            let what = format!("{} @ {t} K", m.name);
            assert_rel(
                m.conductivity_at(tf),
                hand(&m.conductivity, t),
                2e-6,
                &format!("k {what}"),
            );
            assert_rel(
                m.specific_heat_at(tf),
                hand(&m.specific_heat, t),
                2e-6,
                &format!("c_p {what}"),
            );
            assert_rel(
                m.density_at(tf),
                hand(&m.density, t),
                2e-6,
                &format!("rho {what}"),
            );
            assert_rel(
                m.heat_capacity_at(tf),
                hand(&m.density, t) * hand(&m.specific_heat, t),
                4e-6,
                &format!("rho c_p {what}"),
            );
            assert_rel(
                m.diffusivity_at(tf),
                hand_alpha(&m, t),
                6e-6,
                &format!("alpha {what}"),
            );
        }
    }
}

/// `Linear` at its own reference temperature returns `ref_value` bit for bit,
/// and the four presets are four distinct materials (distinct names, distinct α).
#[test]
fn density_at_reference_temperature_is_ref_value_bit_for_bit_and_presets_are_distinct() {
    let ps = presets();
    for m in &ps {
        let TemperatureDependence::Linear {
            ref_value,
            ref_temp,
            ..
        } = m.density
        else {
            panic!("{}: density is expected to be a Linear model", m.name)
        };
        assert_eq!(ref_temp, m.reference_temperature, "{}", m.name);
        assert_eq!(
            m.density_at(ref_temp).to_bits(),
            ref_value.to_bits(),
            "{}",
            m.name
        );
    }
    let names: std::collections::BTreeSet<&str> = ps.iter().map(|m| m.name).collect();
    assert_eq!(names.len(), 4);
    let alphas: std::collections::BTreeSet<u32> = ps
        .iter()
        .map(|m| m.diffusivity_at(300.0).to_bits())
        .collect();
    assert_eq!(alphas.len(), 4, "four presets, four diffusivities");
    // Hand ordering at 300 K: aluminium (6.9e-5) > steel (1.4e-5) > titanium (2.9e-6) > PLA (1.1e-7).
    let a = |m: &ThermalMaterial| f64::from(m.diffusivity_at(300.0));
    assert!(a(&ps[1]) > a(&ps[0]) && a(&ps[0]) > a(&ps[2]) && a(&ps[2]) > a(&ps[3]));
}

/// No clamp beyond the calibrated range: the fits extrapolate, and the steel
/// conductivity line `60 − 0.03 T` is negative above 2000 K, which
/// `diffusivity_at` passes through (only `ρ c_p ≤ 0` is guarded).
#[test]
fn beyond_calibrated_range_the_fits_extrapolate_without_clamp() {
    let m = ThermalMaterial::steel_1018();
    // 3000 K: k = 60 − 90 = −30, c_p = 380 + 900 = 1280, ρ = 7870 (1 − 3.6e-5 · 2706.85)
    let k = -30.0;
    let cp = 1280.0;
    let rho = 7870.0 * (1.0 - 3.6e-5 * (3000.0 - 293.15));
    assert_rel(m.conductivity_at(3000.0), k, 2e-6, "k @ 3000 K");
    assert_rel(m.specific_heat_at(3000.0), cp, 2e-6, "c_p @ 3000 K");
    assert_rel(m.density_at(3000.0), rho, 2e-6, "rho @ 3000 K");
    let alpha = m.diffusivity_at(3000.0);
    assert!(
        alpha < 0.0,
        "negative conductivity is not guarded: alpha = {alpha}"
    );
    assert_rel(alpha, k / (rho * cp), 6e-6, "alpha @ 3000 K");
    // and a field entirely in that regime reports no finite bound (max α stays 0)
    assert!(stable_dt_1d(&[3000.0; 4], &m, 1e-3).is_infinite());
    assert!(stable_dt_3d(&[3000.0; 27], &m, 1e-3).is_infinite());
    // Below the range: 100 K is still the same line (no clamp at 273 K).
    assert_rel(m.conductivity_at(100.0), 60.0 - 3.0, 2e-6, "k @ 100 K");
}

// ------------------------------------------------------- stability bounds

#[test]
fn stability_bounds_match_dx2_over_2_and_6_max_alpha() {
    let dx = 2.5e-3_f64;
    let field = [300.0_f32, 450.0, 600.0, 750.0];
    for m in presets() {
        let max_alpha = field
            .iter()
            .map(|&t| hand_alpha(&m, f64::from(t)))
            .fold(f64::MIN, f64::max);
        assert!(max_alpha > 0.0);
        assert_rel(
            stable_dt_1d(&field, &m, dx as f32),
            dx * dx / (2.0 * max_alpha),
            8e-6,
            &format!("stable_dt_1d {}", m.name),
        );
        assert_rel(
            stable_dt_3d(&field, &m, dx as f32),
            dx * dx / (6.0 * max_alpha),
            8e-6,
            &format!("stable_dt_3d {}", m.name),
        );
        // the 3-D bound is one third of the 1-D bound: same max α, 6 vs 2
        assert_rel(
            stable_dt_3d(&field, &m, dx as f32),
            f64::from(stable_dt_1d(&field, &m, dx as f32)) / 3.0,
            2e-6,
            "3-D = 1-D / 3",
        );
    }
    // unit material, dx = 1: exactly 1/2 and 1/6 (f32 rounding of the latter)
    let u = unit_material();
    assert_eq!(stable_dt_1d(&[300.0; 3], &u, 1.0), 0.5);
    assert_rel(stable_dt_3d(&[300.0; 27], &u, 1.0), 1.0 / 6.0, 2e-7, "1/6");
}

/// The bound is the maximum over the field, so adding a colder cell to a steel
/// field (higher α at lower T) lowers the bound, and the field order does not
/// matter.
#[test]
fn stability_bound_follows_the_most_diffusive_cell() {
    let m = ThermalMaterial::steel_1018();
    let hot = [700.0_f32, 650.0];
    let with_cold = [700.0_f32, 650.0, 300.0];
    let reordered = [300.0_f32, 700.0, 650.0];
    let b_hot = stable_dt_1d(&hot, &m, 1e-3);
    let b_cold = stable_dt_1d(&with_cold, &m, 1e-3);
    assert!(b_cold < b_hot, "{b_cold} vs {b_hot}");
    assert_eq!(
        b_cold.to_bits(),
        stable_dt_1d(&reordered, &m, 1e-3).to_bits()
    );
    assert_rel(
        b_cold,
        1e-6 / (2.0 * hand_alpha(&m, 300.0)),
        8e-6,
        "bound at 300 K",
    );
}

// ---------------------------------------------------- explicit step, exact

#[test]
fn explicit_step_1d_spreads_exactly_r_times_the_second_difference() {
    let u = unit_material();
    // r = 0.25, A = 8 ⇒ rA = 2, 2rA = 4, all dyadic
    let mut t = [300.0_f32, 300.0, 308.0, 300.0, 300.0];
    transient_step_1d(&mut t, &u, 1.0, 0.25);
    assert_eq!(t, [300.0, 302.0, 304.0, 302.0, 300.0]);
    // second step: cell 1 sees (300 − 2·302 + 304) = 0 ⇒ fixed; cell 0 sees
    // (300 − 600 + 302) = 2 ⇒ +0.5; centre (302 − 608 + 302) = −4 ⇒ −1
    transient_step_1d(&mut t, &u, 1.0, 0.25);
    assert_eq!(t, [300.5, 302.0, 303.0, 302.0, 300.5]);
}

#[test]
fn explicit_step_1d_linear_profile_is_fixed_on_interior_cells_and_ends_move_by_rb() {
    let u = unit_material();
    let n = 6;
    let before: Vec<f32> = (0..n).map(|i| 300.0 + 2.0 * i as f32).collect();
    let mut t = before.clone();
    transient_step_1d(&mut t, &u, 1.0, 0.25); // r = 0.25, b = 2 ⇒ r b = 0.5
    assert_eq!(
        &t[1..n - 1],
        &before[1..n - 1],
        "interior cells are a fixed point"
    );
    assert_eq!(t[0], before[0] + 0.5, "cell 0 gains r·b");
    assert_eq!(t[n - 1], before[n - 1] - 0.5, "cell n−1 loses r·b");
    // same with the slope reversed: the ends swap roles
    let mut t = before.iter().rev().copied().collect::<Vec<f32>>();
    transient_step_1d(&mut t, &u, 1.0, 0.25);
    assert_eq!(t[0], before[n - 1] - 0.5);
    assert_eq!(t[n - 1], before[0] + 0.5);
}

/// With a preset the update is the conservative finite-volume form: the face
/// between the hot and an ambient cell carries `k_f = 2 k_h k_a / (k_h + k_a)`
/// (harmonic mean), and each cell divides by its own `ρ c_p`.
#[test]
fn explicit_step_1d_with_steel_uses_the_local_diffusivity() {
    let m = ThermalMaterial::steel_1018();
    let (dx, ambient, hot) = (1e-3_f64, 300.0_f64, 800.0_f64);
    let amp = hot - ambient;
    let dt = 0.2 * dx * dx / hand_alpha(&m, ambient); // r(ambient) = 0.2, below the bound
    let mut t = [
        ambient as f32,
        ambient as f32,
        hot as f32,
        ambient as f32,
        ambient as f32,
    ];
    transient_step_1d(&mut t, &m, dx as f32, dt as f32);
    let (k_hot, k_amb) = (hand(&m.conductivity, hot), hand(&m.conductivity, ambient));
    let c_hot = hand(&m.density, hot) * hand(&m.specific_heat, hot);
    let c_amb = hand(&m.density, ambient) * hand(&m.specific_heat, ambient);
    let k_face = 2.0 * k_hot * k_amb / (k_hot + k_amb);
    let r_amb = hand_alpha(&m, ambient) * dt / (dx * dx);
    assert!((r_amb - 0.2).abs() < 1e-12);
    let flux = k_face * amp * dt / (dx * dx);
    assert_rel(
        t[2],
        hot - 2.0 * flux / c_hot,
        1e-5,
        "hot cell −2 k_f A dt / (ρc_p dx²)",
    );
    assert_rel(
        t[1],
        ambient + flux / c_amb,
        1e-5,
        "left neighbour +k_f A dt / (ρc_p dx²)",
    );
    assert_rel(t[3], ambient + flux / c_amb, 1e-5, "right neighbour");
    assert_eq!(
        t[0], ambient as f32,
        "cells not adjacent to the hot node are untouched"
    );
    assert_eq!(t[4], ambient as f32);
}

#[test]
fn explicit_step_3d_spreads_exactly_r_times_the_seven_point_stencil() {
    let u = unit_material();
    let n = 3;
    let idx = |i: usize, j: usize, k: usize| i + n * (j + n * k);
    let mut t = vec![300.0_f32; n * n * n];
    t[idx(1, 1, 1)] = 316.0; // A = 16, r = 1/16 ⇒ rA = 1, 6rA = 6
    transient_step_3d(&mut t, n, n, n, &u, 1.0, 0.0625);
    let mut want = vec![300.0_f32; n * n * n];
    want[idx(1, 1, 1)] = 310.0;
    for p in [
        idx(0, 1, 1),
        idx(2, 1, 1),
        idx(1, 0, 1),
        idx(1, 2, 1),
        idx(1, 1, 0),
        idx(1, 1, 2),
    ] {
        want[p] = 301.0;
    }
    assert_eq!(t, want);
    let sum: f64 = t.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (sum - (27.0 * 300.0 + 16.0)).abs() < 1e-9,
        "zero-flux faces conserve the sum"
    );
}

#[test]
fn explicit_step_3d_linear_profile_along_x_is_fixed_inside_and_moves_rb_on_the_faces() {
    let u = unit_material();
    let (nx, ny, nz) = (4, 3, 3);
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut before = vec![0.0_f32; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                before[idx(i, j, k)] = 300.0 + 2.0 * i as f32; // b = 2 along x, flat in y, z
            }
        }
    }
    let mut t = before.clone();
    transient_step_3d(&mut t, nx, ny, nz, &u, 1.0, 0.0625); // r b = 0.125
    for k in 0..nz {
        for j in 0..ny {
            assert_eq!(t[idx(0, j, k)], before[idx(0, j, k)] + 0.125);
            assert_eq!(t[idx(1, j, k)], before[idx(1, j, k)]);
            assert_eq!(t[idx(2, j, k)], before[idx(2, j, k)]);
            assert_eq!(t[idx(3, j, k)], before[idx(3, j, k)] - 0.125);
        }
    }
}

/// Above the explicit bound nothing refuses or clamps: at `r = 1` (twice the
/// bound) a hot node of height `A` becomes `−A` after one step and `+3A`
/// after two — exact, and growing.
#[test]
fn explicit_step_above_the_stability_bound_amplifies_silently() {
    let u = unit_material();
    let bound = stable_dt_1d(&[300.0; 3], &u, 1.0);
    assert_eq!(bound, 0.5);
    let dt = 2.0 * bound; // r = 1
    let mut t = [300.0_f32, 304.0, 300.0]; // A = 4
    transient_step_1d(&mut t, &u, 1.0, dt);
    assert_eq!(
        t,
        [304.0, 296.0, 304.0],
        "one step: centre a − A, ends a + A"
    );
    transient_step_1d(&mut t, &u, 1.0, dt);
    assert_eq!(
        t,
        [296.0, 312.0, 296.0],
        "two steps: centre a + 3A, ends a − A"
    );
    // the same in 3-D at r = 1/3 (twice the 3-D bound 1/6): centre loses 6rA = 2A
    let n = 3;
    let bound3 = stable_dt_3d(&[300.0; 27], &u, 1.0);
    let mut f = vec![300.0_f32; 27];
    f[13] = 312.0; // A = 12 so that 2 · bound3 · 12 is representable closely
    transient_step_3d(&mut f, n, n, n, &u, 1.0, 2.0 * bound3);
    assert!(
        f[13] < 300.0 - 11.0,
        "centre flipped below ambient: {}",
        f[13]
    );
}

// ------------------------------------------------- Crank–Nicolson, hand-solved

fn cn3_expected(a: f32, amp: f64, r: f64) -> [f64; 3] {
    let q = amp * (2.0 - r) / (2.0 + 3.0 * r);
    let p = 2.0 * r * amp / (2.0 + 3.0 * r);
    assert!((2.0 * p + q - amp).abs() < 1e-12, "hand solution conserves");
    [f64::from(a) + p, f64::from(a) + q, f64::from(a) + p]
}

#[test]
fn crank_nicolson_3_cells_matches_the_hand_solved_system() {
    let u = unit_material();
    for (dt, amp) in [(1.0_f64, 5.0_f64), (2.0, 8.0), (0.5, 10.0), (6.0, 3.0)] {
        let r = dt; // α = dx = 1
        let a = 300.0_f32;
        let mut t = [a, a + amp as f32, a];
        crank_nicolson_step_1d(&mut t, &u, 1.0, dt as f32);
        let want = cn3_expected(a, amp, r);
        for i in 0..3 {
            assert_rel(t[i], want[i], 4e-7, &format!("r = {r}, cell {i}"));
        }
        if (r - 2.0).abs() < 1e-12 {
            // the centre deviation vanishes in one step at r = 2
            assert!((f64::from(t[1]) - 300.0).abs() < 1e-4, "{}", t[1]);
        }
    }
}

/// Uniform field: the exact solution is the field itself; the Thomas sweep
/// rounds (`m = −½r/(1+½r)` is not dyadic) so the pin is a few ulp.
#[test]
fn crank_nicolson_uniform_field_is_identity_to_rounding() {
    let u = unit_material();
    for n in [3usize, 4, 7, 32] {
        let mut t = vec![400.0_f32; n];
        crank_nicolson_step_1d(&mut t, &u, 1.0, 3.0);
        for (i, &v) in t.iter().enumerate() {
            assert!(
                (v - 400.0).abs() <= 4.0 * f32::EPSILON * 400.0,
                "n = {n}, cell {i}: {v}"
            );
        }
    }
    // steel too (per-cell r equal on a uniform field)
    let m = ThermalMaterial::steel_1018();
    let mut t = vec![500.0_f32; 16];
    crank_nicolson_step_1d(&mut t, &m, 1e-3, 1.0);
    for &v in &t {
        assert!((v - 500.0).abs() <= 4.0 * f32::EPSILON * 500.0, "{v}");
    }
}

#[test]
fn nonlinear_crank_nicolson_with_constant_properties_equals_linear_bit_for_bit() {
    let u = unit_material();
    let before = [300.0_f32, 300.0, 305.0, 300.0, 300.0, 300.0];
    let mut lin = before;
    crank_nicolson_step_1d(&mut lin, &u, 1.0, 1.5);
    assert_ne!(lin, before, "the step must do something");
    let mut nl = before;
    let iters = crank_nicolson_step_1d_nonlinear(&mut nl, &u, 1.0, 1.5, 1e-6, 10);
    assert_eq!(iters, 2, "first iterate, then a repeat with Δ = 0");
    assert_eq!(nl, lin, "constant α: Picard reproduces the linearised step");
    // a single allowed iteration is the same step, reported as 1
    let mut one = before;
    assert_eq!(
        crank_nicolson_step_1d_nonlinear(&mut one, &u, 1.0, 1.5, 1e-6, 1),
        1
    );
    assert_eq!(one, lin);
    // an infinite tolerance stops after the first iterate
    let mut inf = before;
    assert_eq!(
        crank_nicolson_step_1d_nonlinear(&mut inf, &u, 1.0, 1.5, f32::INFINITY, 10),
        1
    );
    assert_eq!(inf, lin);
}

/// Independent 3-cell Crank–Nicolson solve with a **per-row** `r` (the rows
/// are not assumed equal, unlike `cn3_expected`): derived directly from
/// `(I − rᵢ/2 L) T_new = (I + rᵢ/2 L) T_old` with zero-flux (ghost = self)
/// ends, solved by a from-scratch Thomas sweep — no module code is called.
fn hand_cn3_nonuniform(told: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let a = [
        [1.0 + r[0] / 2.0, -r[0] / 2.0, 0.0],
        [-r[1] / 2.0, 1.0 + r[1], -r[1] / 2.0],
        [0.0, -r[2] / 2.0, 1.0 + r[2] / 2.0],
    ];
    let b = [
        (1.0 - r[0] / 2.0) * told[0] + (r[0] / 2.0) * told[1],
        (r[1] / 2.0) * told[0] + (1.0 - r[1]) * told[1] + (r[1] / 2.0) * told[2],
        (r[2] / 2.0) * told[1] + (1.0 - r[2] / 2.0) * told[2],
    ];
    let sub = [0.0, a[1][0], a[2][1]];
    let mut diag = [a[0][0], a[1][1], a[2][2]];
    let sup = [a[0][1], a[1][2], 0.0];
    let mut d = b;
    for i in 1..3 {
        let factor = sub[i] / diag[i - 1];
        diag[i] -= factor * sup[i - 1];
        d[i] -= factor * d[i - 1];
    }
    let mut x = [0.0; 3];
    x[2] = d[2] / diag[2];
    for i in (0..2).rev() {
        x[i] = (d[i] - sup[i] * x[i + 1]) / diag[i];
    }
    x
}

/// Pins the Picard averaging convention itself (`t_avg = ½(initial + current)`,
/// not `current` alone): the two conventions are indistinguishable at
/// iteration 1 (`current == initial` there, so `½(initial+current) ==
/// initial == current`), so this scene forces exactly 2 iterations
/// (`tolerance = −1`, never breaks early) and hand-derives the full 2-step
/// trajectory for both conventions with `hand_cn3_nonuniform`, on a material
/// with the closed form `α(T) = 1 + T` (`conductivity = Linear{ref_value: 1,
/// ref_temp: 0, coeff: 1}`, unit density and specific heat), so each row's
/// `r` is a simple function of the sampled temperature.
#[test]
fn nonlinear_crank_nicolson_pins_the_midpoint_averaging_convention() {
    let m = ThermalMaterial {
        name: "alpha_is_one_plus_t",
        conductivity: TemperatureDependence::Linear {
            ref_value: 1.0,
            ref_temp: 0.0,
            coeff: 1.0,
        },
        specific_heat: TemperatureDependence::Constant(1.0),
        density: TemperatureDependence::Constant(1.0),
        reference_temperature: 0.0,
    };
    let (dx, dt) = (1.0_f64, 0.02_f64);
    let initial = [0.0_f32, 10.0, 0.0];
    let alpha = |t: f64| 1.0 + t;
    let r_of = |field: [f64; 3]| {
        [
            alpha(field[0]) * dt,
            alpha(field[1]) * dt,
            alpha(field[2]) * dt,
        ]
    };

    let initial64 = [0.0_f64, 10.0, 0.0];
    let r1 = r_of(initial64); // t_avg at iter 1 is `initial` under both conventions
    let next1 = hand_cn3_nonuniform(initial64, r1);

    let t_avg2_midpoint = [
        0.5 * (initial64[0] + next1[0]),
        0.5 * (initial64[1] + next1[1]),
        0.5 * (initial64[2] + next1[2]),
    ];
    let t_avg2_current_only = next1;
    let next2_midpoint = hand_cn3_nonuniform(initial64, r_of(t_avg2_midpoint));
    let next2_current_only = hand_cn3_nonuniform(initial64, r_of(t_avg2_current_only));

    // the two conventions must actually disagree on this scene, or the test
    // proves nothing about which one the module implements
    let convention_gap = (0..3)
        .map(|i| (next2_midpoint[i] - next2_current_only[i]).abs())
        .fold(0.0, f64::max);
    assert!(
        convention_gap > 1e-3,
        "conventions do not differ: gap {convention_gap}"
    );

    let mut actual = initial;
    let iters = crank_nicolson_step_1d_nonlinear(&mut actual, &m, dx as f32, dt as f32, -1.0, 2);
    assert_eq!(iters, 2, "tolerance = -1 must force exactly max_iterations");

    for i in 0..3 {
        let got = f64::from(actual[i]);
        assert!(
            (got - next2_midpoint[i]).abs() < 2e-5,
            "cell {i}: got {got}, midpoint-averaging oracle {} (current-only oracle would be {})",
            next2_midpoint[i],
            next2_current_only[i],
        );
    }
}

/// Steel: α depends on T, so the Picard step (α at the mid-step temperature)
/// differs from the step that freezes α at the start — the scene that makes
/// "ignore the nonlinearity" red. The difference is bounded by the α change
/// between the start and the mid-step temperature.
#[test]
fn nonlinear_crank_nicolson_differs_from_linear_on_a_temperature_dependent_preset() {
    let m = ThermalMaterial::steel_1018();
    let dx = 1e-3_f64;
    let dt = 1.0 * dx * dx / hand_alpha(&m, 300.0); // r ≈ 1 at ambient
    let before = [300.0_f32, 300.0, 300.0, 1000.0, 300.0, 300.0, 300.0];
    let mut lin = before;
    crank_nicolson_step_1d(&mut lin, &m, dx as f32, dt as f32);
    let mut nl = before;
    let iters = crank_nicolson_step_1d_nonlinear(&mut nl, &m, dx as f32, dt as f32, 1e-5, 20);
    assert!(iters >= 2, "iterations {iters}");
    let worst = lin
        .iter()
        .zip(&nl)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    assert!(
        worst > 0.5,
        "linear and Picard agree to {worst} K — nonlinearity ignored?"
    );
    // NOTE: unlike the constant-property case, this scheme's per-cell `r[i]`
    // is not a proper finite-volume flux (it does not use a shared face
    // diffusivity between neighbours), so a 700 K gradient across one cell
    // does not conserve the sum to within a small tolerance — it is not a
    // documented property of this scheme and is not asserted here.
    for g in [&lin, &nl] {
        assert!(g.iter().all(|v| v.is_finite()), "{g:?}");
    }
}

// -------------------------------------------------------- degenerate input

#[test]
fn dx_zero_or_negative_panics_in_all_six_grid_functions() {
    let m = ThermalMaterial::aluminum_6061();
    for dx in [0.0_f32, -1e-3] {
        let cases: Vec<(&str, Option<String>)> = vec![
            (
                "transient_step_1d",
                panic_message(|| transient_step_1d(&mut [300.0; 4], &m, dx, 1e-3)),
            ),
            (
                "transient_step_3d",
                panic_message(|| transient_step_3d(&mut [300.0; 27], 3, 3, 3, &m, dx, 1e-3)),
            ),
            (
                "crank_nicolson_step_1d",
                panic_message(|| crank_nicolson_step_1d(&mut [300.0; 4], &m, dx, 1e-3)),
            ),
            (
                "crank_nicolson_step_1d_nonlinear",
                panic_message(|| {
                    let _ =
                        crank_nicolson_step_1d_nonlinear(&mut [300.0; 4], &m, dx, 1e-3, 1e-4, 5);
                }),
            ),
            (
                "stable_dt_1d",
                panic_message(|| {
                    let _ = stable_dt_1d(&[300.0; 4], &m, dx);
                }),
            ),
            (
                "stable_dt_3d",
                panic_message(|| {
                    let _ = stable_dt_3d(&[300.0; 27], &m, dx);
                }),
            ),
        ];
        for (name, msg) in cases {
            assert_eq!(
                msg.as_deref(),
                Some("dx must be positive"),
                "{name} with dx = {dx}"
            );
        }
    }
}

#[test]
fn dt_zero_is_identity_for_explicit_and_a_panic_for_crank_nicolson() {
    let m = ThermalMaterial::titanium_ti6al4v();
    let before = [300.0_f32, 450.0, 900.0, 320.0, 300.0];
    let mut t = before;
    transient_step_1d(&mut t, &m, 1e-3, 0.0);
    assert_eq!(t, before);
    let before3: Vec<f32> = (0..27).map(|i| 300.0 + 7.0 * i as f32).collect();
    let mut t3 = before3.clone();
    transient_step_3d(&mut t3, 3, 3, 3, &m, 1e-3, 0.0);
    assert_eq!(t3, before3);
    for dt in [0.0_f32, -1.0] {
        assert_eq!(
            panic_message(|| crank_nicolson_step_1d(&mut [300.0; 4], &m, 1e-3, dt)).as_deref(),
            Some("dt must be positive")
        );
        assert_eq!(
            panic_message(|| {
                let _ = crank_nicolson_step_1d_nonlinear(&mut [300.0; 4], &m, 1e-3, dt, 1e-4, 5);
            })
            .as_deref(),
            Some("dt must be positive")
        );
    }
}

#[test]
fn fewer_than_three_cells_are_returned_unchanged_and_the_bounds_of_nothing_are_infinite() {
    let m = ThermalMaterial::pla_polymer();
    for before in [vec![], vec![500.0_f32], vec![500.0_f32, 100.0]] {
        let mut t = before.clone();
        transient_step_1d(&mut t, &m, 1e-3, 1.0);
        assert_eq!(t, before);
        let mut t = before.clone();
        crank_nicolson_step_1d(&mut t, &m, 1e-3, 1.0);
        assert_eq!(t, before);
        let mut t = before.clone();
        assert_eq!(
            crank_nicolson_step_1d_nonlinear(&mut t, &m, 1e-3, 1.0, 1e-4, 5),
            0
        );
        assert_eq!(t, before);
    }
    assert!(stable_dt_1d(&[], &m, 1e-3).is_infinite());
    assert!(stable_dt_3d(&[], &m, 1e-3).is_infinite());
    // 3-D: any side below 3 is a no-op even with a steep field
    let before: Vec<f32> = (0..12).map(|i| 300.0 + 50.0 * i as f32).collect();
    for (nx, ny, nz) in [(2usize, 2usize, 3usize), (12, 1, 1), (3, 4, 1), (1, 3, 4)] {
        let mut t = before.clone();
        transient_step_3d(&mut t, nx, ny, nz, &m, 1e-3, 1.0);
        assert_eq!(t, before, "{nx}x{ny}x{nz}");
    }
    let mut empty: Vec<f32> = vec![];
    transient_step_3d(&mut empty, 0, 0, 0, &m, 1e-3, 1.0);
    assert!(empty.is_empty());
    // length mismatch is refused
    let msg = panic_message(|| transient_step_3d(&mut [300.0; 26], 3, 3, 3, &m, 1e-3, 1e-3));
    assert!(
        msg.as_deref()
            .is_some_and(|s| s.contains("temperature slice length mismatch")),
        "{msg:?}"
    );
}

#[test]
fn zero_max_iterations_is_identity_and_reports_zero() {
    let m = ThermalMaterial::steel_1018();
    let before = [300.0_f32, 300.0, 800.0, 300.0, 300.0];
    let mut t = before;
    assert_eq!(
        crank_nicolson_step_1d_nonlinear(&mut t, &m, 1e-3, 1.0, 1e-4, 0),
        0
    );
    assert_eq!(t, before);
}

/// Zero density or zero specific heat: `ρ c_p = 0` ⇒ `α = 0` by the guard,
/// the bounds are `+∞`, both steppers are identity bit for bit.
#[test]
fn zero_density_or_specific_heat_gives_zero_diffusivity_and_identity_steps() {
    for (name, rho, cp) in [("rho = 0", 0.0_f32, 500.0_f32), ("c_p = 0", 7800.0, 0.0)] {
        let m = ThermalMaterial {
            name,
            conductivity: TemperatureDependence::Constant(40.0),
            specific_heat: TemperatureDependence::Constant(cp),
            density: TemperatureDependence::Constant(rho),
            reference_temperature: 293.15,
        };
        assert_eq!(m.heat_capacity_at(300.0), 0.0, "{name}");
        assert_eq!(m.diffusivity_at(300.0), 0.0, "{name}");
        let before = [300.0_f32, 300.0, 900.0, 300.0, 300.0];
        assert!(stable_dt_1d(&before, &m, 1e-3).is_infinite(), "{name}");
        assert!(stable_dt_3d(&before, &m, 1e-3).is_infinite(), "{name}");
        let mut t = before;
        transient_step_1d(&mut t, &m, 1e-3, 1.0);
        assert_eq!(t, before, "{name} explicit");
        let mut t = before;
        crank_nicolson_step_1d(&mut t, &m, 1e-3, 1.0);
        assert_eq!(t, before, "{name} Crank–Nicolson (r = 0 ⇒ unit diagonal)");
        let mut t = before;
        // r = 0 ⇒ the first iterate already equals `current` (both are
        // `initial` untouched), so `max_delta = 0 < tolerance` breaks the
        // Picard loop after exactly 1 iteration (not 2, unlike the non-zero
        // constant-material case where the first non-trivial step must
        // repeat once to converge).
        assert_eq!(
            crank_nicolson_step_1d_nonlinear(&mut t, &m, 1e-3, 1.0, 1e-4, 5),
            1,
            "{name}"
        );
        assert_eq!(t, before, "{name} Picard");
    }
}

/// `T = f32::MAX`: no panic anywhere. The polynomial fits are written as
/// `c0 + c1·T + c2·T·T` with left-to-right evaluation, i.e.
/// `c0 + c1·T + (c2·T)·T`; every preset has `c2 = 0.0`, so `(0.0 · T) · T`
/// collapses to `0.0 · T = 0.0` **before** the second multiplication can
/// overflow — the fits stay finite even at `T = f32::MAX` (verified against
/// the module with a throwaway probe: `k(MAX) ≈ −1.0208470e37`,
/// `c_p(MAX) ≈ 1.0208470e38`, both finite). `density_at` (a `Linear` model)
/// is likewise finite. Their *product* `ρ·c_p`, computed in `heat_capacity_at`,
/// does overflow to `−∞` (probed: `ρ(MAX) ≈ −9.6408790e37`,
/// `ρ·c_p(MAX) = −∞`), which `diffusivity_at`'s `!denom.is_finite()` guard
/// catches, giving `α(MAX) = 0`. With `α = 0` the explicit update's
/// `dt · α · laplacian` term of the former non-conservative explicit update
/// was `0.0 · (−∞)` (the laplacian itself overflows to `−∞` on a uniform
/// `f32::MAX` field, since `2.0 · T` overflows before the subtraction), which
/// is `NaN`. The explicit step is now a face-flux update with no `2.0 · T`
/// term and skips a cell whose `ρ c_p` is not finite and positive, so it
/// leaves the field unchanged; the Crank–Nicolson step still returns `NaN`.
#[test]
fn extreme_temperature_does_not_panic_the_explicit_step_is_unchanged_and_cn_returns_nan() {
    let m = ThermalMaterial::steel_1018();
    let t = f32::MAX;
    let msg = panic_message(|| {
        assert!(
            m.conductivity_at(t).is_finite() && m.conductivity_at(t) < 0.0,
            "k(MAX) = {}",
            m.conductivity_at(t)
        );
        assert!(
            m.specific_heat_at(t).is_finite() && m.specific_heat_at(t) > 0.0,
            "cp(MAX) = {}",
            m.specific_heat_at(t)
        );
        assert!(
            m.density_at(t).is_finite() && m.density_at(t) < 0.0,
            "rho(MAX) = {}",
            m.density_at(t)
        );
        assert!(
            m.heat_capacity_at(t).is_infinite(),
            "rho*cp(MAX) = {}",
            m.heat_capacity_at(t)
        );
        assert_eq!(m.diffusivity_at(t), 0.0, "guard: non-finite ρ c_p ⇒ 0");
        assert!(stable_dt_1d(&[t; 3], &m, 1e-3).is_infinite());
        assert!(stable_dt_3d(&[t; 27], &m, 1e-3).is_infinite());
        let mut field = [t; 3];
        transient_step_1d(&mut field, &m, 1e-3, 1e-3);
        assert!(
            field.iter().all(|v| *v == t),
            "explicit step at f32::MAX: {field:?}"
        );
        let mut field = [t; 3];
        crank_nicolson_step_1d(&mut field, &m, 1e-3, 1e-3);
        assert!(
            field.iter().all(|v| v.is_nan()),
            "Crank–Nicolson at f32::MAX: {field:?}"
        );
    });
    assert_eq!(
        msg, None,
        "an assertion inside, or a panic in the module: {msg:?}"
    );
}
