//! Oracles for `pair_potential`: Lennard-Jones 12-6, Morse, Coulomb,
//! screened Coulomb (Yukawa), cutoff / shift and the Lorentz–Berthelot rule.
//!
//! # Where the expected values come from
//!
//! - Lennard-Jones: `U = 4ε[(σ/r)¹² − (σ/r)⁶]`, `F = −dU/dr =
//!   24ε[2(σ/r)¹² − (σ/r)⁶]/r`; minimum at `r_min = 2^{1/6} σ` with `U = −ε`,
//!   `U(σ) = 0`, `U''(r_min) = 72 ε / r_min² = 36·2^{2/3} ε/σ² ≈ 57.146 ε/σ²`
//!   (Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §1.3).
//! - Morse: `U = D(1 − e^{−a(r−r_e)})² − D`, `F = −2Da e(1 − e)` with
//!   `e = e^{−a(r−r_e)}`; `U(r_e) = −D`, `U''(r_e) = 2Da²` (Morse 1929).
//! - Coulomb: `U = k q_i q_j / r`, `F = k q_i q_j / r²` (repulsive for like
//!   charges); the force on `i` equals `q_i E_j(x_i)` with `E_j` the field of
//!   `electromagnetic::EmSource::PointCharge` at `x_j`.
//! - Yukawa (Debye–Hückel screened Coulomb): `U = k q_i q_j e^{−r/λ}/r`,
//!   `F = k q_i q_j e^{−r/λ}(1/r² + 1/(λ r))`.
//! - Cutoff: energy shift `U(r) − U(r_c)`, shifted force
//!   `U(r) − U(r_c) − (r − r_c)U'(r_c)` and `F(r) − F(r_c)` (Allen & Tildesley
//!   §5.2.3).
//! - Lorentz–Berthelot: `σ_ij = (σ_i + σ_j)/2`, `ε_ij = √(ε_i ε_j)`.
//!
//! All expected values are evaluated in `f64` from the formulas above, never
//! from the crate. Tolerances: `1e-12` relative for rational expressions (a
//! dozen `Fix128` operations, each exact to 2⁻⁶⁴, against `f64` at 1.1e-16
//! relative), `1e-12` relative for `exp` through `math_util::exp_fix`
//! (12-term Taylor series after reduction to `|x| ≤ 0.5`, remainder
//! `0.5¹³/13! < 2e-14`, then squaring that doubles the relative error per
//! square: measured below 1e-14 on the values here).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `powi` / `exp` / `sqrt` compute closed-form references, not state.
#![allow(clippy::disallowed_methods)]

use alice_physics::electromagnetic::EmSource;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::pair_potential::{
    lorentz_berthelot, Coulomb, LennardJones, Morse, PairPotential, PairPotentialError, ShiftMode,
    Truncated, Yukawa, COULOMB_CONSTANT,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn assert_rel(actual: Fix128, expected: f64, tol: f64, what: &str) {
    let a = actual.to_f64();
    let scale = expected.abs().max(1e-300);
    assert!(
        (a - expected).abs() <= tol * scale,
        "{what}: actual {a:e}, expected {expected:e}, rel err {:e}",
        (a - expected).abs() / scale
    );
}

fn assert_abs(actual: Fix128, expected: f64, tol: f64, what: &str) {
    let a = actual.to_f64();
    assert!(
        (a - expected).abs() <= tol,
        "{what}: actual {a:e}, expected {expected:e}, abs err {:e}",
        (a - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// Lennard-Jones
// ---------------------------------------------------------------------------

const EPS: f64 = 1.7;
const SIG: f64 = 0.9;

fn lj() -> LennardJones {
    LennardJones::new(fx(EPS), fx(SIG)).expect("valid LJ")
}

fn lj_u(r: f64) -> f64 {
    let s6 = (SIG / r).powi(6);
    4.0 * EPS * (s6 * s6 - s6)
}

fn lj_f(r: f64) -> f64 {
    let s6 = (SIG / r).powi(6);
    24.0 * EPS * (2.0 * s6 * s6 - s6) / r
}

#[test]
fn lj_energy_and_force_match_closed_form() {
    let p = lj();
    for &r in &[0.8, 0.9, 1.0, 1.1, 1.3, 1.7, 2.2, 3.0] {
        assert_rel(p.energy(fx(r)).unwrap(), lj_u(r), 1e-12, "LJ U");
        assert_rel(p.force(fx(r)).unwrap(), lj_f(r), 1e-12, "LJ F");
    }
}

#[test]
fn lj_zero_crossing_at_sigma_is_exact() {
    // (σ/σ)¹² − (σ/σ)⁶ = 0 with no rounding: s² = σ²/σ² = 1 exactly.
    let p = LennardJones::new(fx(EPS), Fix128::from_int(2)).unwrap();
    assert_eq!(p.energy(Fix128::from_int(2)).unwrap(), Fix128::ZERO);
}

#[test]
fn lj_minimum_at_two_to_one_sixth_sigma() {
    let p = lj();
    let r_min = 2f64.powf(1.0 / 6.0) * SIG;
    assert_rel(p.energy(fx(r_min)).unwrap(), -EPS, 1e-12, "U(r_min) = -eps");
    // F(r_min) = 0; the input r_min is the f64 value (relative error up to
    // 2^-53 ≈ 1.1e-16) and |dF/dr| = U'' = 72 ε / r_min² ≈ 120, so
    // |F| ≤ 120 · 1.01 · 1.1e-16 ≈ 1.4e-14 plus Fix128 rounding
    assert_abs(p.force(fx(r_min)).unwrap(), 0.0, 3e-14, "F(r_min) = 0");
    // force sign: repulsive inside, attractive outside
    assert!(p.force(fx(r_min * 0.99)).unwrap() > Fix128::ZERO);
    assert!(p.force(fx(r_min * 1.01)).unwrap() < Fix128::ZERO);
}

#[test]
fn lj_curvature_at_minimum_is_57_eps_over_sigma_squared() {
    // k = U''(r_min) = 36·2^{2/3} ε/σ² ≈ 57.146 ε/σ². The centred difference
    // of the force has truncation error U''''·δ²/6 (δ = 1e-5 σ: ~1e-8
    // relative) and rounding 2⁻⁶⁴·|F|/δ (negligible), so 1e-6 is a wide margin.
    let p = lj();
    let r_min = 2f64.powf(1.0 / 6.0) * SIG;
    let d = 1e-5 * SIG;
    let k_num = -(p.force(fx(r_min + d)).unwrap().to_f64()
        - p.force(fx(r_min - d)).unwrap().to_f64())
        / (2.0 * d);
    let k = 36.0 * 2f64.powf(2.0 / 3.0) * EPS / (SIG * SIG);
    assert!((k - 57.146_437_5 * EPS / (SIG * SIG)).abs() < 1e-5);
    assert!(
        (k_num - k).abs() <= 1e-6 * k,
        "k = {k_num}, closed form {k}"
    );
}

#[test]
fn lj_force_vector_points_along_separation() {
    let p = lj();
    let d = Vec3Fix::new(fx(0.3), fx(-0.6), fx(0.8));
    let r = (0.09f64 + 0.36 + 0.64).sqrt();
    let f = p.force_on_first(d).unwrap();
    let fm = lj_f(r) / r;
    assert_rel(f.x, fm * 0.3, 1e-12, "Fx");
    assert_rel(f.y, fm * -0.6, 1e-12, "Fy");
    assert_rel(f.z, fm * 0.8, 1e-12, "Fz");
}

#[test]
fn lj_rejects_invalid_parameters_and_distances() {
    assert_eq!(
        LennardJones::new(fx(1.0), Fix128::ZERO),
        Err(PairPotentialError::NonPositiveSigma)
    );
    assert_eq!(
        LennardJones::new(fx(1.0), fx(-1.0)),
        Err(PairPotentialError::NonPositiveSigma)
    );
    assert_eq!(
        LennardJones::new(fx(-1.0), fx(1.0)),
        Err(PairPotentialError::NegativeEpsilon)
    );
    // ε = 0 is a valid (non-interacting) potential
    let zero = LennardJones::new(Fix128::ZERO, fx(1.0)).unwrap();
    assert_eq!(zero.energy(fx(1.3)).unwrap(), Fix128::ZERO);
    assert_eq!(zero.force(fx(1.3)).unwrap(), Fix128::ZERO);
    let p = lj();
    assert_eq!(
        p.energy(Fix128::ZERO),
        Err(PairPotentialError::NonPositiveDistance)
    );
    assert_eq!(
        p.force(fx(-1.0)),
        Err(PairPotentialError::NonPositiveDistance)
    );
    assert_eq!(
        p.force_on_first(Vec3Fix::ZERO),
        Err(PairPotentialError::NonPositiveDistance)
    );
    // (σ/r)¹² beyond the Fix128 range (|x| < 2^63) is an error, not a wrap:
    // σ/r = 100 gives 1e24.
    assert_eq!(p.energy(fx(SIG / 100.0)), Err(PairPotentialError::Overflow));
    assert_eq!(p.force(fx(SIG / 100.0)), Err(PairPotentialError::Overflow));
}

// ---------------------------------------------------------------------------
// Morse
// ---------------------------------------------------------------------------

const MD: f64 = 2.5;
const MA: f64 = 1.8;
const MRE: f64 = 1.2;

fn morse() -> Morse {
    Morse::new(fx(MD), fx(MA), fx(MRE)).unwrap()
}

fn morse_u(r: f64) -> f64 {
    let e = (-MA * (r - MRE)).exp();
    MD * (1.0 - e) * (1.0 - e) - MD
}

fn morse_f(r: f64) -> f64 {
    let e = (-MA * (r - MRE)).exp();
    -2.0 * MD * MA * e * (1.0 - e)
}

#[test]
fn morse_energy_and_force_match_closed_form() {
    let p = morse();
    for &r in &[0.6, 0.9, 1.1, 1.2, 1.4, 2.0, 3.5, 6.0] {
        // absolute floor 1e-13: near r_e the force passes through 0
        let u = p.energy(fx(r)).unwrap().to_f64();
        assert!(
            (u - morse_u(r)).abs() <= 1e-12 * morse_u(r).abs() + 1e-13,
            "U({r})"
        );
        let f = p.force(fx(r)).unwrap().to_f64();
        assert!(
            (f - morse_f(r)).abs() <= 1e-12 * morse_f(r).abs() + 1e-13,
            "F({r})"
        );
    }
}

#[test]
fn morse_minimum_is_minus_d_at_r_e() {
    let p = morse();
    // a(r - r_e) = 0 exactly, e^0 = 1 exactly
    assert_eq!(p.energy(fx(MRE)).unwrap(), -fx(MD));
    assert_eq!(p.force(fx(MRE)).unwrap(), Fix128::ZERO);
    // dissociation limit: U → 0 (e^{-a(r - r_e)} < 1e-17 at r = 25)
    assert_abs(p.energy(fx(25.0)).unwrap(), 0.0, 1e-15, "U(∞) = 0");
}

#[test]
fn morse_harmonic_force_constant_is_2da2() {
    // F(r_e + δ) = −2Da²δ (1 − (3/2) a δ + O(δ²)); δ = 1e-6 makes the
    // anharmonic part 2.7e-6 relative.
    let p = morse();
    let k = 2.0 * MD * MA * MA;
    assert_rel(p.harmonic_force_constant(), k, 1e-15, "2Da²");
    for &d in &[1e-6, -1e-6] {
        let f = p.force(fx(MRE + d)).unwrap().to_f64();
        let k_eff = -f / d;
        assert!(
            (k_eff - k).abs() <= 3e-6 * k + 1e-9,
            "k_eff {k_eff}, 2Da² {k}"
        );
    }
}

#[test]
fn morse_repulsive_wall_beyond_exp_fix_range() {
    // `math_util::exp_fix` saturates at x ≥ 20; the Morse wall needs
    // e^{a(r_e − r)} beyond that. x = 7 (3 − 0.1) = 20.3: e = 6.5e8,
    // (1 − e)² = 4.3e17 and |F| = 2Da e(e − 1) = 6e18 < 2^63, so the values
    // exist and must match f64 (squaring 6 times after the Taylor step
    // multiplies its 2e-14 relative error by 2^6: 1e-11 relative covers it).
    let p = Morse::new(fx(1.0), fx(7.0), fx(3.0)).unwrap();
    let r = 0.1;
    let e = (7.0f64 * (3.0 - r)).exp();
    assert_rel(
        p.energy(fx(r)).unwrap(),
        (1.0 - e) * (1.0 - e) - 1.0,
        1e-11,
        "wall U",
    );
    assert_rel(
        p.force(fx(r)).unwrap(),
        -2.0 * 7.0 * e * (1.0 - e),
        1e-11,
        "wall F",
    );
    // x = 10 (3 − 0.1) = 29: (1 − e)² = 1.5e25 > 2^63 ⇒ Err, not a wrap
    // x = 7 (3 − 0.001) = 20.993: U = 1.7e18 fits but |F| = 2.4e19 does not
    let p7 = Morse::new(fx(1.0), fx(7.0), fx(3.0)).unwrap();
    assert!(p7.energy(fx(0.001)).is_ok());
    assert_eq!(p7.force(fx(0.001)), Err(PairPotentialError::Overflow));
    let p = Morse::new(fx(1.0), fx(10.0), fx(3.0)).unwrap();
    assert_eq!(p.energy(fx(0.1)), Err(PairPotentialError::Overflow));
    assert_eq!(p.force(fx(0.1)), Err(PairPotentialError::Overflow));
    // x = 60: e itself (1.1e26) is beyond the range
    let p = Morse::new(fx(1.0), fx(20.0), fx(3.0)).unwrap();
    assert_eq!(p.energy(fx(0.001)), Err(PairPotentialError::Overflow));
}

#[test]
fn morse_rejects_invalid_parameters() {
    assert_eq!(
        Morse::new(fx(-1.0), fx(1.0), fx(1.0)),
        Err(PairPotentialError::NegativeWellDepth)
    );
    assert_eq!(
        Morse::new(fx(1.0), Fix128::ZERO, fx(1.0)),
        Err(PairPotentialError::NonPositiveWidth)
    );
    assert_eq!(
        Morse::new(fx(1.0), fx(1.0), Fix128::ZERO),
        Err(PairPotentialError::NonPositiveEquilibriumDistance)
    );
    assert_eq!(
        morse().energy(Fix128::ZERO),
        Err(PairPotentialError::NonPositiveDistance)
    );
}

// ---------------------------------------------------------------------------
// Coulomb and Yukawa
// ---------------------------------------------------------------------------

#[test]
fn coulomb_constant_is_the_electromagnetic_one() {
    // electromagnetic::EmSource::PointCharge uses k = 8.99e9 V·m/C
    assert_eq!(COULOMB_CONSTANT, Fix128::from_int(8_990_000_000));
}

#[test]
fn coulomb_inverse_square_law() {
    let (qi, qj) = (2e-6, -3e-6);
    let k = 8.99e9;
    let p = Coulomb::new(fx(qi), fx(qj));
    for &r in &[0.05, 0.1, 0.5, 2.0] {
        assert_rel(p.energy(fx(r)).unwrap(), k * qi * qj / r, 1e-9, "Coulomb U");
        assert_rel(
            p.force(fx(r)).unwrap(),
            k * qi * qj / (r * r),
            1e-9,
            "Coulomb F",
        );
    }
    // unlike charges attract: F < 0 means toward the other particle
    assert!(p.force(fx(1.0)).unwrap() < Fix128::ZERO);
    // F(2r) / F(r) = 1/4 (inverse-square, not inverse-cube)
    let ratio = p.force(fx(1.0)).unwrap().to_f64() / p.force(fx(0.5)).unwrap().to_f64();
    assert!((ratio - 0.25).abs() < 1e-9, "ratio {ratio}");
}

#[test]
fn coulomb_force_matches_electromagnetic_point_charge_field() {
    // Same configuration both ways: the force on charge i at x_i from charge
    // j at x_j equals q_i · E_j(x_i).
    let qi = fx(1.5e-6);
    let qj = fx(-0.75e-6);
    let xi = Vec3Fix::new(fx(0.4), fx(-0.2), fx(1.1));
    let xj = Vec3Fix::new(fx(-0.3), fx(0.5), fx(0.2));
    let source = EmSource::PointCharge {
        position: xj,
        charge_c: qj,
    };
    let (e, b) = source.sample(xi);
    assert_eq!(b, Vec3Fix::ZERO);
    let from_field = e * qi;
    let from_pair = Coulomb::new(qi, qj).force_on_first(xi - xj).unwrap();
    // two different rounding orders of k q_j / r³ · d · q_i; each Fix128 step
    // is exact to 2^-64 ≈ 5.4e-20 absolute, |F| ≈ 1e-2, so 1e-12 relative
    // has a large margin
    for (a, b, c) in [
        (from_pair.x, from_field.x, "x"),
        (from_pair.y, from_field.y, "y"),
        (from_pair.z, from_field.z, "z"),
    ] {
        let (a, b) = (a.to_f64(), b.to_f64());
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1e-30),
            "{c}: pair {a:e}, field {b:e}"
        );
    }
    // and the field force is attractive (toward x_j) for unlike charges
    assert!(from_pair.dot(xj - xi) > Fix128::ZERO);
}

#[test]
fn coulomb_with_constant_and_validation() {
    let p = Coulomb::with_constant(Fix128::ONE, fx(2.0), fx(3.0)).unwrap();
    assert_eq!(p.energy(Fix128::from_int(2)).unwrap(), Fix128::from_int(3));
    assert_eq!(
        Coulomb::with_constant(Fix128::ZERO, fx(1.0), fx(1.0)),
        Err(PairPotentialError::NonPositiveCoulombConstant)
    );
    assert_eq!(
        p.force(Fix128::ZERO),
        Err(PairPotentialError::NonPositiveDistance)
    );
}

#[test]
fn yukawa_matches_screened_coulomb_closed_form() {
    let (k, qi, qj, lam) = (1.0, 1.2, -0.8, 0.7);
    let p = Yukawa::with_constant(fx(k), fx(qi), fx(qj), fx(lam)).unwrap();
    for &r in &[0.2, 0.5, 1.0, 2.0, 4.0] {
        let e = (-r / lam).exp();
        let u = k * qi * qj * e / r;
        let f = k * qi * qj * e * (1.0 / (r * r) + 1.0 / (lam * r));
        assert_rel(p.energy(fx(r)).unwrap(), u, 1e-12, "Yukawa U");
        assert_rel(p.force(fx(r)).unwrap(), f, 1e-12, "Yukawa F");
    }
    // λ → ∞ recovers Coulomb (relative difference ≈ r/λ)
    let far = Yukawa::with_constant(fx(k), fx(qi), fx(qj), fx(1e6)).unwrap();
    let c = Coulomb::with_constant(fx(k), fx(qi), fx(qj)).unwrap();
    let (a, b) = (
        far.force(fx(1.0)).unwrap().to_f64(),
        c.force(fx(1.0)).unwrap().to_f64(),
    );
    assert!((a - b).abs() <= 2e-6 * b.abs());
    assert_eq!(
        Yukawa::new(fx(1.0), fx(1.0), Fix128::ZERO),
        Err(PairPotentialError::NonPositiveScreeningLength)
    );
    // SI constant path is the same k as Coulomb
    let y = Yukawa::new(fx(1e-6), fx(1e-6), fx(1e6)).unwrap();
    let c = Coulomb::new(fx(1e-6), fx(1e-6));
    let (a, b) = (
        y.energy(fx(1.0)).unwrap().to_f64(),
        c.energy(fx(1.0)).unwrap().to_f64(),
    );
    assert!((a - b).abs() <= 2e-6 * b.abs());
}

// ---------------------------------------------------------------------------
// Cutoff and shift
// ---------------------------------------------------------------------------

const RC: f64 = 2.5;

#[test]
fn energy_shift_vanishes_at_cutoff_and_keeps_force() {
    let t = Truncated::new(lj(), fx(RC), ShiftMode::EnergyShift).unwrap();
    assert_eq!(t.energy(fx(RC)).unwrap(), Fix128::ZERO);
    assert_eq!(t.cutoff(), fx(RC));
    assert_eq!(t.mode(), ShiftMode::EnergyShift);
    assert_eq!(*t.inner(), lj());
    for &r in &[0.95, 1.2, 2.0, 2.49] {
        assert_rel(
            t.energy(fx(r)).unwrap(),
            lj_u(r) - lj_u(RC),
            1e-12,
            "U - U(rc)",
        );
        assert_rel(t.force(fx(r)).unwrap(), lj_f(r), 1e-12, "F unchanged");
    }
    // beyond the cutoff: exactly zero
    assert_eq!(t.energy(fx(3.0)).unwrap(), Fix128::ZERO);
    assert_eq!(t.force(fx(3.0)).unwrap(), Fix128::ZERO);
}

#[test]
fn force_shift_vanishes_at_cutoff_in_energy_and_force() {
    let t = Truncated::new(lj(), fx(RC), ShiftMode::ForceShift).unwrap();
    assert_eq!(t.energy(fx(RC)).unwrap(), Fix128::ZERO);
    assert_eq!(t.force(fx(RC)).unwrap(), Fix128::ZERO);
    for &r in &[0.95, 1.2, 2.0, 2.49] {
        // U'(rc) = −F(rc)
        let u = lj_u(r) - lj_u(RC) + (r - RC) * lj_f(RC);
        assert_rel(t.energy(fx(r)).unwrap(), u, 1e-11, "shifted-force U");
        assert_rel(
            t.force(fx(r)).unwrap(),
            lj_f(r) - lj_f(RC),
            1e-12,
            "F - F(rc)",
        );
    }
    // continuity just inside the cutoff: |U|, |F| = O(r_c − r)
    let r = RC - 1e-6;
    assert!(t.energy(fx(r)).unwrap().abs().to_f64() < 1e-10);
    assert!(t.force(fx(r)).unwrap().abs().to_f64() < 1e-5);
}

#[test]
fn plain_truncation_keeps_the_jump() {
    let t = Truncated::new(lj(), fx(RC), ShiftMode::None).unwrap();
    assert_rel(t.energy(fx(2.0)).unwrap(), lj_u(2.0), 1e-12, "U unshifted");
    assert_rel(
        t.energy(fx(RC - 1e-9)).unwrap(),
        lj_u(RC),
        1e-7,
        "U just inside",
    );
    assert_eq!(t.energy(fx(RC)).unwrap(), Fix128::ZERO);
}

#[test]
fn truncated_rejects_non_positive_cutoff() {
    assert_eq!(
        Truncated::new(lj(), Fix128::ZERO, ShiftMode::EnergyShift).err(),
        Some(PairPotentialError::NonPositiveCutoff)
    );
    assert_eq!(
        Truncated::new(lj(), fx(-1.0), ShiftMode::ForceShift).err(),
        Some(PairPotentialError::NonPositiveCutoff)
    );
}

// ---------------------------------------------------------------------------
// Mixing rule
// ---------------------------------------------------------------------------

#[test]
fn lorentz_berthelot_closed_form() {
    let a = LennardJones::new(Fix128::ONE, Fix128::ONE).unwrap();
    let b = LennardJones::new(Fix128::from_int(4), Fix128::from_int(2)).unwrap();
    let ab = lorentz_berthelot(&a, &b);
    assert_eq!(ab.sigma(), fx(1.5));
    assert_eq!(ab.epsilon(), Fix128::from_int(2));
    let c = LennardJones::new(fx(0.3), fx(0.7)).unwrap();
    let d = LennardJones::new(fx(1.9), fx(1.1)).unwrap();
    let cd = lorentz_berthelot(&c, &d);
    assert_rel(cd.sigma(), 0.9, 1e-15, "σ_ij");
    assert_rel(cd.epsilon(), (0.3f64 * 1.9).sqrt(), 1e-15, "ε_ij");
    // symmetric and idempotent on equal species
    assert_eq!(lorentz_berthelot(&c, &d), lorentz_berthelot(&d, &c));
    assert_eq!(lorentz_berthelot(&a, &a), a);
}

#[test]
fn pair_potential_error_messages_are_distinct() {
    use std::collections::BTreeSet;
    let all = [
        PairPotentialError::NonPositiveDistance,
        PairPotentialError::NonPositiveSigma,
        PairPotentialError::NegativeEpsilon,
        PairPotentialError::NegativeWellDepth,
        PairPotentialError::NonPositiveWidth,
        PairPotentialError::NonPositiveEquilibriumDistance,
        PairPotentialError::NonPositiveCoulombConstant,
        PairPotentialError::NonPositiveScreeningLength,
        PairPotentialError::NonPositiveCutoff,
        PairPotentialError::Overflow,
    ];
    let msgs: BTreeSet<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(msgs.len(), all.len());
}
