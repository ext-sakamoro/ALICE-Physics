//! Second, independent set of oracles for `pair_potential`.
//!
//! `tests/analytic_pair_potential.rs` compares the energies and forces with
//! the closed forms evaluated in `f64` (LJ `ε = 1.7, σ = 0.9`, Morse
//! `D, a, r_e` fixed, Yukawa, the shifted forms at `r_c = 2.5`). This file
//! does not re-evaluate those closed forms. It checks the module against
//! itself through relations that hold for any correct implementation and
//! fail for the typical mistakes (sign, factor, swapped exponents):
//!
//! - `F = −dU/dr` by a central finite difference of the module's own energy,
//!   for every potential and every shift mode (the energy and force code
//!   paths are separate, so a wrong factor or sign in one of them shows);
//! - `U_FS(r) = ∫_r^{r_c} F_FS(s) ds` by Simpson's rule on the module's
//!   shifted force;
//! - the LJ minimum located by bisection on the sign of the force, compared
//!   with `2^{1/6} σ`; the well depth `−ε`; the curvature by a second
//!   difference of the energy, compared with `k = 72 ε / (2^{1/3} σ²)`;
//! - exact power-of-two ratios: `U(2σ)/ε = 4(2⁻¹² − 2⁻⁶)` separates the 12-6
//!   exponents; the Morse energy at `r_e ± ln 2 / a`;
//! - Yukawa with a screening length far beyond the distances tested reduces
//!   to Coulomb.
//!
//! Parameters: LJ `ε = 0.45, σ = 1.3`; Morse `D = 2.2, a = 1.4, r_e = 0.85`;
//! reduced Coulomb `k = 1`, `q = (1.5, −0.8)`.

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::pair_potential::{
    lorentz_berthelot, Coulomb, LennardJones, Morse, PairPotential, PairPotentialError, ShiftMode,
    Truncated, Yukawa,
};

const EPS: f64 = 0.45;
const SIG: f64 = 1.3;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn f(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn lj() -> LennardJones {
    LennardJones::new(fx(EPS), fx(SIG)).unwrap()
}

fn morse() -> Morse {
    Morse::new(fx(2.2), fx(1.4), fx(0.85)).unwrap()
}

fn coulomb() -> Coulomb {
    Coulomb::with_constant(Fix128::ONE, fx(1.5), fx(-0.8)).unwrap()
}

fn yukawa(lambda: f64) -> Yukawa {
    Yukawa::with_constant(Fix128::ONE, fx(1.5), fx(-0.8), fx(lambda)).unwrap()
}

fn u<P: PairPotential>(p: &P, r: f64) -> f64 {
    f(p.energy(fx(r)).unwrap())
}

fn force<P: PairPotential>(p: &P, r: f64) -> f64 {
    f(p.force(fx(r)).unwrap())
}

/// `−dU/dr` by a fourth-order central difference
/// `(−U(r+2δ) + 8U(r+δ) − 8U(r−δ) + U(r−2δ)) / (12δ)` (truncation
/// `O(δ⁴ U⁽⁵⁾)`, rounding `≈ 2⁻⁶⁴·|U|/δ`).
fn fd_force<P: PairPotential>(p: &P, r: f64, delta: f64) -> f64 {
    -(-u(p, r + 2.0 * delta) + 8.0 * u(p, r + delta) - 8.0 * u(p, r - delta)
        + u(p, r - 2.0 * delta))
        / (12.0 * delta)
}

fn assert_force_is_minus_gradient<P: PairPotential>(p: &P, rs: &[f64], what: &str) {
    for &r in rs {
        let want = fd_force(p, r, 1e-4 * r);
        let got = force(p, r);
        // absolute floor 1e-10: `exp_fix` (Morse, Yukawa) is accurate to
        // ≈ 1e-15 relative, which the difference divides by δ ≈ 1e-4
        let scale = got.abs();
        assert!(
            (got - want).abs() < 1e-9 * scale + 1e-10,
            "{what} r={r}: F {got} vs −dU/dr {want} (rel {:e})",
            (got - want) / scale
        );
    }
}

// ---------------------------------------------------------------------------
// F = −dU/dr
// ---------------------------------------------------------------------------

#[test]
fn lj_force_is_minus_energy_gradient() {
    assert_force_is_minus_gradient(&lj(), &[1.05, 1.2, 1.3, 1.46, 1.6, 2.1, 3.3], "LJ");
}

#[test]
fn morse_force_is_minus_energy_gradient() {
    assert_force_is_minus_gradient(&morse(), &[0.4, 0.7, 0.85, 1.0, 1.6, 3.0], "Morse");
}

#[test]
fn coulomb_and_yukawa_force_is_minus_energy_gradient() {
    assert_force_is_minus_gradient(&coulomb(), &[0.3, 0.9, 2.0, 5.0], "Coulomb");
    assert_force_is_minus_gradient(&yukawa(0.7), &[0.3, 0.9, 2.0, 5.0], "Yukawa");
}

#[test]
fn shifted_forms_force_is_minus_energy_gradient() {
    let rc = 3.1;
    for mode in [
        ShiftMode::None,
        ShiftMode::EnergyShift,
        ShiftMode::ForceShift,
    ] {
        let t = Truncated::new(lj(), fx(rc), mode).unwrap();
        assert_force_is_minus_gradient(&t, &[1.1, 1.5, 2.2, 2.9], &format!("LJ {mode:?}"));
        let tm = Truncated::new(morse(), fx(rc), mode).unwrap();
        assert_force_is_minus_gradient(&tm, &[0.6, 1.2, 2.5], &format!("Morse {mode:?}"));
    }
}

/// oracle: for the shifted-force form, both `U` and `F` vanish at `r_c` and
/// `F = −dU/dr` inside, so `U_FS(r) = ∫_r^{r_c} F_FS(s) ds`. Composite
/// Simpson with 4000 panels on the module's force (error
/// `(r_c − r) h⁴ max|F⁽⁴⁾| / 180`; at `r = 1.1`, `F⁽⁴⁾ ≈ 4·10⁶` and
/// `h = 4·10⁻⁴`, so `≲ 1e-9`; 400 panels measured `4·10⁻⁷`). Also: `EnergyShift` and `ForceShift` energies differ by
/// exactly `(r − r_c) F(r_c)`, i.e. linearly in `r`.
#[test]
fn shifted_force_energy_is_integral_of_shifted_force() {
    let rc = 2.7;
    let fs = Truncated::new(lj(), fx(rc), ShiftMode::ForceShift).unwrap();
    let es = Truncated::new(lj(), fx(rc), ShiftMode::EnergyShift).unwrap();
    let f_rc = force(&lj(), rc);
    for &r in &[1.1, 1.4, 2.0, 2.6] {
        let n = 4000;
        let h = (rc - r) / n as f64;
        let mut s = force(&fs, r) + 0.0; // F_FS(r_c) = 0 at the far end
        for k in 1..n {
            let w = if k % 2 == 1 { 4.0 } else { 2.0 };
            s += w * force(&fs, r + h * k as f64);
        }
        let integral = s * h / 3.0;
        let got = u(&fs, r);
        assert!(
            (got - integral).abs() < 1e-8 * got.abs().max(1e-3),
            "r={r}: U_FS {got} vs ∫F_FS {integral}"
        );
        let diff = u(&fs, r) - u(&es, r);
        assert!(
            (diff - (r - rc) * f_rc).abs() < 1e-14,
            "r={r}: U_FS − U_ES = {diff} vs (r − r_c)F(r_c)"
        );
    }
}

// ---------------------------------------------------------------------------
// Lennard-Jones: minimum, depth, curvature, exponents
// ---------------------------------------------------------------------------

/// oracle: bisection on the sign of the module's force (repulsive inside,
/// attractive outside) finds the minimum at `2^{1/6} σ` (to `1e-12`); the
/// energy there is `−ε`; the curvature from the second difference of the
/// energy is `k = 72 ε / (2^{1/3} σ²)` (to `1e-6`; truncation
/// `δ²U⁗/12 ≈ 1e-9` relative at `δ = 1e-4`).
#[test]
fn lj_minimum_depth_and_curvature() {
    let p = lj();
    let (mut lo, mut hi) = (1.0 * SIG, 1.5 * SIG);
    assert!(force(&p, lo) > 0.0 && force(&p, hi) < 0.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if force(&p, mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r_min = 0.5 * (lo + hi);
    let want = 2f64.powf(1.0 / 6.0) * SIG;
    assert!((r_min - want).abs() < 1e-12, "r_min {r_min} vs {want}");
    assert!(
        (u(&p, want) + EPS).abs() < 1e-13,
        "U(r_min) {}",
        u(&p, want)
    );
    let d = 1e-4;
    let k_num = (u(&p, want + d) - 2.0 * u(&p, want) + u(&p, want - d)) / (d * d);
    let k = 72.0 * EPS / (2f64.powf(1.0 / 3.0) * SIG * SIG);
    assert!(((k_num - k) / k).abs() < 1e-6, "k {k_num} vs {k}");
}

/// oracle: at `r = σ` the energy crosses zero (`|U| ≤ 1e-15`, `σ = 1.3` not
/// dyadic) with force `24ε/σ` (`2·1 − 1 = 1`); at `r = 2σ` the energy is
/// `4ε(2⁻¹² − 2⁻⁶)` and the force `24ε(2·2⁻¹² − 2⁻⁶)/(2σ)` — a swap of the
/// exponents 12 ↔ 6 changes the sign of both.
#[test]
fn lj_exact_ratios_pin_the_exponents() {
    let p = lj();
    assert!(u(&p, SIG).abs() < 1e-15, "U(σ) = {}", u(&p, SIG));
    let f_sigma = force(&p, SIG);
    assert!(((f_sigma - 24.0 * EPS / SIG) / f_sigma).abs() < 1e-13);
    let s12 = 2f64.powi(-12);
    let s6 = 2f64.powi(-6);
    let u2 = u(&p, 2.0 * SIG);
    let f2 = force(&p, 2.0 * SIG);
    let want_u = 4.0 * EPS * (s12 - s6);
    let want_f = 24.0 * EPS * (2.0 * s12 - s6) / (2.0 * SIG);
    assert!(
        ((u2 - want_u) / want_u).abs() < 1e-12,
        "U(2σ) {u2} vs {want_u}"
    );
    assert!(
        ((f2 - want_f) / want_f).abs() < 1e-12,
        "F(2σ) {f2} vs {want_f}"
    );
    // scaling: U(λσ; σ) is independent of σ for fixed ε
    let other = LennardJones::new(fx(EPS), fx(0.37)).unwrap();
    for lam in [0.95, 1.2, 1.9] {
        let a = u(&p, lam * SIG);
        let b = u(&other, lam * 0.37);
        assert!(
            (a - b).abs() < 1e-12 * a.abs().max(1e-3),
            "λ={lam}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Morse
// ---------------------------------------------------------------------------

/// oracle: with `x = e^{−a(r − r_e)}`, `U = D(1 − x)² − D`. At
/// `r = r_e − ln2/a` (`x = 2`) `U = 0`; at `r = r_e + ln2/a` (`x = 1/2`)
/// `U = −3D/4` and `F = −2Da·(1/2)(1/2) = −Da/2`. The curvature at `r_e` by a
/// second difference matches `harmonic_force_constant()` and `2Da²`.
#[test]
fn morse_values_at_half_and_double_decay() {
    let p = morse();
    let (d, a, re) = (2.2, 1.4, 0.85);
    let r_in = re - 2f64.ln() / a;
    let r_out = re + 2f64.ln() / a;
    assert!(u(&p, r_in).abs() < 1e-13, "U(x=2) = {}", u(&p, r_in));
    assert!((u(&p, r_out) + 0.75 * d).abs() < 1e-13, "U(x=1/2)");
    assert!((force(&p, r_out) + d * a / 2.0).abs() < 1e-13, "F(x=1/2)");
    let h = 1e-4;
    let k_num = (u(&p, re + h) - 2.0 * u(&p, re) + u(&p, re - h)) / (h * h);
    let k = 2.0 * d * a * a;
    assert!(((k_num - k) / k).abs() < 1e-6, "k {k_num} vs {k}");
    assert!((f(p.harmonic_force_constant()) - k).abs() < 1e-14);
}

// ---------------------------------------------------------------------------
// Yukawa → Coulomb, Coulomb sign
// ---------------------------------------------------------------------------

/// oracle: `λ = 10⁶` at `r ≤ 5` gives `e^{−r/λ} ≥ 1 − 5·10⁻⁶`, so Yukawa is
/// Coulomb to that relative accuracy; Coulomb with opposite charges is
/// attractive (`F < 0`) and `U·r` is constant (`k q₁ q₂ = −1.2`).
#[test]
fn yukawa_reduces_to_coulomb_and_coulomb_scales_as_one_over_r() {
    let c = coulomb();
    let y = yukawa(1e6);
    for &r in &[0.25, 1.0, 2.5, 5.0] {
        let (uc, uy) = (u(&c, r), u(&y, r));
        assert!(((uy - uc) / uc).abs() < 6e-6, "r={r}: U {uy} vs {uc}");
        let (fc, fy) = (force(&c, r), force(&y, r));
        assert!(((fy - fc) / fc).abs() < 6e-6, "r={r}: F {fy} vs {fc}");
        assert!((uc * r + 1.2).abs() < 1e-13, "r={r}: U·r = {}", uc * r);
        assert!(fc < 0.0, "opposite charges must attract");
    }
}

// ---------------------------------------------------------------------------
// Vector form, mixing rule
// ---------------------------------------------------------------------------

/// oracle: `force_on_first(d)` is `F(|d|) d/|d|`: parallel to `d` for a
/// repulsive distance, antiparallel for an attractive one, magnitude
/// `|F(|d|)|`, and odd in `d`.
#[test]
fn force_vector_direction_and_oddness() {
    let p = lj();
    for (d, repulsive) in [
        ([0.6, -0.8, 0.5], true),  // |d| ≈ 1.118 < r_min
        ([1.2, 0.9, -0.7], false), // |d| ≈ 1.655 > r_min
    ] {
        let dv = Vec3Fix::new(fx(d[0]), fx(d[1]), fx(d[2]));
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let fv = p.force_on_first(dv).unwrap();
        let fm = p.force_on_first(-dv).unwrap();
        let mag = force(&p, r);
        let comps = [f(fv.x), f(fv.y), f(fv.z)];
        for c in 0..3 {
            let want = mag * d[c] / r;
            assert!((comps[c] - want).abs() < 1e-12 * mag.abs(), "axis {c}");
        }
        assert_eq!(mag > 0.0, repulsive);
        let back = [f(fm.x), f(fm.y), f(fm.z)];
        for c in 0..3 {
            assert!((back[c] + comps[c]).abs() < 1e-17, "odd in d, axis {c}");
        }
    }
}

/// oracle: mixing a species with itself returns its own parameters; mixing
/// `(ε, σ) = (0.45, 1.3)` with `(1.8, 0.7)` gives `σ = 1.0`, `ε = 0.9`.
#[test]
fn lorentz_berthelot_identity_and_values() {
    let a = lj();
    let b = LennardJones::new(fx(1.8), fx(0.7)).unwrap();
    let aa = lorentz_berthelot(&a, &a);
    assert!((f(aa.sigma()) - SIG).abs() < 1e-17);
    assert!((f(aa.epsilon()) - EPS).abs() < 1e-15);
    let ab = lorentz_berthelot(&a, &b);
    assert!((f(ab.sigma()) - 1.0).abs() < 1e-17);
    assert!((f(ab.epsilon()) - 0.9).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

/// `r = 0` and `r < 0` are `NonPositiveDistance` for every potential and
/// every shift mode; `r = 2⁻⁶³` is `Overflow` (`1/r` out of range) and
/// `r = 10⁻⁴σ` is `Overflow` for LJ (`(σ/r)¹² = 10⁴⁸`); `ε = 0` and zero
/// charges give exactly zero energy and force; at and beyond `r_c` the
/// truncated forms are exactly zero; a zero separation vector is an error.
#[test]
fn degenerate_inputs_have_explicit_outcomes() {
    let lj = lj();
    for r in [Fix128::ZERO, fx(-0.5)] {
        assert_eq!(lj.energy(r), Err(PairPotentialError::NonPositiveDistance));
        assert_eq!(lj.force(r), Err(PairPotentialError::NonPositiveDistance));
        assert_eq!(
            morse().energy(r),
            Err(PairPotentialError::NonPositiveDistance)
        );
        assert_eq!(
            coulomb().force(r),
            Err(PairPotentialError::NonPositiveDistance)
        );
        assert_eq!(
            yukawa(1.0).energy(r),
            Err(PairPotentialError::NonPositiveDistance)
        );
        let t = Truncated::new(lj, fx(2.5), ShiftMode::ForceShift).unwrap();
        assert_eq!(t.energy(r), Err(PairPotentialError::NonPositiveDistance));
        assert_eq!(t.force(r), Err(PairPotentialError::NonPositiveDistance));
    }
    assert_eq!(
        coulomb().energy(Fix128::from_raw(0, 2)),
        Err(PairPotentialError::Overflow)
    );
    assert_eq!(lj.energy(fx(1.3e-4)), Err(PairPotentialError::Overflow));
    assert_eq!(
        lj.force_on_first(Vec3Fix::ZERO),
        Err(PairPotentialError::NonPositiveDistance)
    );

    let flat = LennardJones::new(Fix128::ZERO, fx(SIG)).unwrap();
    let neutral = Coulomb::with_constant(Fix128::ONE, Fix128::ZERO, fx(3.0)).unwrap();
    for r in [0.5, 1.3, 4.0] {
        assert_eq!(flat.energy(fx(r)).unwrap(), Fix128::ZERO);
        assert_eq!(flat.force(fx(r)).unwrap(), Fix128::ZERO);
        assert_eq!(neutral.energy(fx(r)).unwrap(), Fix128::ZERO);
        assert_eq!(neutral.force(fx(r)).unwrap(), Fix128::ZERO);
    }

    for mode in [
        ShiftMode::None,
        ShiftMode::EnergyShift,
        ShiftMode::ForceShift,
    ] {
        let t = Truncated::new(lj, fx(2.5), mode).unwrap();
        for r in [2.5, 2.5 + 1e-12, 7.0] {
            assert_eq!(t.energy(fx(r)).unwrap(), Fix128::ZERO, "{mode:?} r={r}");
            assert_eq!(t.force(fx(r)).unwrap(), Fix128::ZERO, "{mode:?} r={r}");
        }
    }
    assert_eq!(
        Truncated::new(lj, Fix128::ZERO, ShiftMode::None).err(),
        Some(PairPotentialError::NonPositiveCutoff)
    );
    assert_eq!(
        LennardJones::new(fx(-0.1), fx(1.0)).err(),
        Some(PairPotentialError::NegativeEpsilon)
    );
    assert_eq!(
        Morse::new(fx(1.0), Fix128::ZERO, fx(1.0)).err(),
        Some(PairPotentialError::NonPositiveWidth)
    );
    assert_eq!(
        Yukawa::with_constant(Fix128::ONE, Fix128::ONE, Fix128::ONE, Fix128::ZERO).err(),
        Some(PairPotentialError::NonPositiveScreeningLength)
    );
}
