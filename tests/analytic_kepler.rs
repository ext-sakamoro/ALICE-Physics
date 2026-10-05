//! Independent oracles for `kepler` (two-body problem).
//!
//! Every expected value comes from outside the module: published worked
//! examples (Vallado 4th ed. Examples 2-1 and 2-6, Meeus *Astronomical
//! Algorithms* Example 30.a), closed forms evaluated here in `f64`
//! (`T = 2π√(a³/μ)`, vis-viva, `ε = −μ/2a`, `h = √(μa(1−e²))`, circular speed
//! `√(μ/r)`, the J2 critical inclination `cos² i = 1/5`), or an `f64` Kepler
//! solver written in this file (plain Newton in a different arithmetic). No
//! expected value is produced by calling the function under test.
//!
//! Units: km, s, `μ = 398 600.4418 km³/s²` (a test-local value; the crate
//! carries no body constants).

#![allow(clippy::disallowed_methods)]

use alice_physics::kepler::{
    j2_arg_periapsis_rate, j2_raan_rate, mean_from_true_anomaly, mean_motion, orbital_period,
    solve_kepler, specific_angular_momentum, specific_orbital_energy, true_from_mean_anomaly,
    vis_viva_speed, KeplerError, OrbitalElements, StateVector, DEGENERACY_TOLERANCE,
};
use alice_physics::{Fix128, Vec3Fix};
use core::f64::consts::PI;

const MU: f64 = 398_600.441_8;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// `Fix128::to_f64` adds the fraction to a negative integer part
/// (`−1 + lo/2⁶⁴` for a small negative value), so it carries an absolute
/// error of `2⁻⁵³` instead of a relative one; convert the magnitude instead
/// where small negative values are compared relatively.
fn f64_of(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn v3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Smallest signed difference of two angles (rad).
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0 * PI);
    if d > PI {
        d - 2.0 * PI
    } else {
        d
    }
}

fn elements(a: f64, e: f64, i: f64, raan: f64, argp: f64, nu: f64) -> OrbitalElements {
    OrbitalElements::new(fx(a), fx(e), fx(i), fx(raan), fx(argp), fx(nu)).unwrap()
}

/// Reference: plain `f64` Newton on `E − e sin E = M` (no bracket, start at
/// the middle of the turn containing `M`).
fn ref_kepler(m: f64, e: f64) -> f64 {
    // Start at the middle of the turn that contains M (E0 = π for M in
    // [0, 2π)), which Newton converges from for every e < 1
    // (Charles & Tatum 1998).
    let mut ecc = m - m.rem_euclid(2.0 * PI) + PI;
    for _ in 0..200 {
        ecc -= (ecc - e * ecc.sin() - m) / (1.0 - e * ecc.cos());
    }
    ecc
}

// ---------------------------------------------------------------------------
// Kepler's equation
// ---------------------------------------------------------------------------

/// oracle: Vallado 4th ed. Example 2-1 — `M = 235.4°`, `e = 0.4`
/// → `E = 220.512 074 767 522°`.
///
/// Tolerance `1e-10°` (`1.7e-12 rad`): the printed value has 12 decimals
/// (rounding `5e-13°`) and the solver stops within `KEPLER_TOLERANCE`
/// (`5.7e-14 rad`) above the CORDIC floor (`≈ 2⁻⁴⁸ = 3.6e-15`).
#[test]
fn kepler_matches_vallado_example_2_1() {
    let e = solve_kepler(fx(235.4_f64.to_radians()), fx(0.4)).unwrap();
    let deg = e.to_f64().to_degrees();
    assert!((deg - 220.512_074_767_522).abs() < 1e-10, "E = {deg}°");
}

/// oracle: Meeus, *Astronomical Algorithms* 2nd ed., Example 30.a —
/// `M = 5°`, `e = 0.1` → `E = 5.554 589 253 872 320°`. Tolerance as above.
#[test]
fn kepler_matches_meeus_example_30a() {
    let e = solve_kepler(fx(5.0_f64.to_radians()), fx(0.1)).unwrap();
    let deg = e.to_f64().to_degrees();
    assert!((deg - 5.554_589_253_872_32).abs() < 1e-10, "E = {deg}°");
}

/// oracle: the defining identity `E − e·sin E = M`, evaluated in `f64`, for
/// `e` from 0 to `1 − 10⁻⁶` and `M` across several turns, both signs, and the
/// edges `0`, `±π`.
///
/// Tolerance `2e-13`: the solver stops on a step `≤ 5.7e-14` (or a bracket of
/// that width), the residual is `f′·ΔE ≤ (1+e)·5.7e-14 ≈ 1.2e-13`, plus the
/// CORDIC floor and the `f64` evaluation of `sin` at `|E| ≤ 25` (`≈ 1e-14`).
#[test]
fn kepler_residual_identity_over_grid() {
    let es = [
        0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 0.999, 0.999_999,
    ];
    let ms = [
        0.0, 1e-9, -1e-9, 0.1, 0.5, 1.0, 2.0, 3.0, PI, -PI, -0.1, -2.5, 4.0, 6.0, 7.5, -20.0, 20.0,
    ];
    for &e in &es {
        for &m in &ms {
            let ef = fx(e);
            let mf = fx(m);
            let ecc = solve_kepler(mf, ef).unwrap().to_f64();
            let resid = ecc - ef.to_f64() * ecc.sin() - mf.to_f64();
            assert!(
                resid.abs() < 2e-13,
                "e={e} M={m}: E={ecc}, residual {resid:e}"
            );
        }
    }
}

/// oracle: `E` agrees with an independent `f64` Newton solver
/// (`ref_kepler`, different start, no bracket, `f64`), tolerance `1e-12` rad
/// (solver tolerance + CORDIC floor + `f64` noise, with the `1/(1−e)` growth
/// of `dE/dM` capped at `e = 0.99`).
#[test]
fn kepler_agrees_with_independent_f64_solver() {
    for &e in &[0.0, 0.2, 0.6, 0.9, 0.99] {
        for k in 0..37 {
            let m = -9.0 + 0.5 * k as f64;
            let mf = fx(m);
            let got = solve_kepler(mf, fx(e)).unwrap().to_f64();
            let want = ref_kepler(mf.to_f64(), fx(e).to_f64());
            assert!((got - want).abs() < 1e-12, "e={e} M={m}: {got} vs {want}");
        }
    }
}

/// oracle: `e = 0` makes Kepler's equation `E = M`; the solver returns `M`
/// bit for bit (the first iterate has `f = 0`).
#[test]
fn kepler_circular_returns_mean_anomaly_exactly() {
    for m in [0.0, 0.3, -1.7, 3.0, 12.0, -40.0] {
        let mf = fx(m);
        assert_eq!(solve_kepler(mf, Fix128::ZERO).unwrap(), mf, "M = {m}");
    }
}

/// Degenerate inputs of `solve_kepler`: `e < 0`, `e = 1`, `e > 1` are
/// `Err(EccentricityOutOfRange)` (hyperbolic Kepler is not supported).
#[test]
fn kepler_rejects_eccentricity_outside_unit_interval() {
    for e in [-0.1, 1.0, 1.5] {
        assert_eq!(
            solve_kepler(fx(1.0), fx(e)),
            Err(KeplerError::EccentricityOutOfRange),
            "e = {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Elements ↔ state
// ---------------------------------------------------------------------------

/// oracle: Vallado 4th ed. Example 2-6 (COE2RV): `p = 11 067.790 km`,
/// `e = 0.832 85`, `i = 87.87°`, `Ω = 227.89°`, `ω = 53.38°`, `ν = 92.335°`
/// → `r = (6525.368, 6861.532, 6449.119) km`,
/// `v = (4.902 279, 5.533 140, −1.975 710) km/s`.
///
/// Tolerance: half the last printed digit plus `1e-9` relative arithmetic
/// noise (`5.1e-4 km`, `5.1e-7 km/s`).
#[test]
fn elements_to_state_matches_vallado_example_2_6() {
    let e = 0.832_85;
    let a = 11_067.790 / (1.0 - e * e);
    let el = elements(
        a,
        e,
        87.87_f64.to_radians(),
        227.89_f64.to_radians(),
        53.38_f64.to_radians(),
        92.335_f64.to_radians(),
    );
    let s = el.to_state(fx(MU)).unwrap();
    let r = v3(s.position);
    let v = v3(s.velocity);
    let r_want = [6525.368, 6861.532, 6449.119];
    let v_want = [4.902_279, 5.533_140, -1.975_710];
    for k in 0..3 {
        assert!((r[k] - r_want[k]).abs() < 5.1e-4, "r[{k}] = {}", r[k]);
        assert!((v[k] - v_want[k]).abs() < 5.1e-7, "v[{k}] = {}", v[k]);
    }
}

/// oracle: elements → state → elements is the identity on non-degenerate
/// orbits (`e ≥ 0.05`, `i` away from `0` and `π`).
///
/// Tolerance: `a` relative `1e-11`; angles `1e-10 rad`. Each CORDIC call
/// carries `≈ 2⁻⁴⁸`; `e` and `i` come back from vector norms of
/// `O(10⁴)`-sized quantities (relative noise `≈ 1e-14`), and `ω`, `ν` are
/// conditioned by `1/e ≤ 20` and `Ω` by `1/sin i ≤ 6`, so the round trip
/// error stays near `1e-12`; the bound keeps a factor 100 for the products
/// of up to six such terms.
#[test]
fn elements_state_round_trip() {
    let mu = fx(MU);
    for &(a, e) in &[
        (7000.0, 0.05),
        (12_000.0, 0.3),
        (26_560.0, 0.74),
        (42_164.0, 0.9),
    ] {
        for &i in &[0.2, 1.0, 1.9, 2.9] {
            for &raan in &[0.0, 1.3, 4.0] {
                for &argp in &[0.4, 3.5, 6.0] {
                    for &nu in &[0.0, 1.0, 3.0, 5.5] {
                        let el = elements(a, e, i, raan, argp, nu);
                        let back =
                            OrbitalElements::from_state(&el.to_state(mu).unwrap(), mu).unwrap();
                        let ctx = format!("a={a} e={e} i={i} Ω={raan} ω={argp} ν={nu}");
                        assert!(
                            ((back.semi_major_axis.to_f64() - a) / a).abs() < 1e-11,
                            "{ctx}: a {}",
                            back.semi_major_axis.to_f64()
                        );
                        assert!((back.eccentricity.to_f64() - e).abs() < 1e-10, "{ctx}: e");
                        assert!((back.inclination.to_f64() - i).abs() < 1e-10, "{ctx}: i");
                        for (name, got, want) in [
                            ("Ω", back.raan, raan),
                            ("ω", back.arg_periapsis, argp),
                            ("ν", back.true_anomaly, nu),
                        ] {
                            let d = angle_diff(got.to_f64(), want);
                            assert!(d.abs() < 1e-10, "{ctx}: {name} {} (Δ {d:e})", got.to_f64());
                        }
                    }
                }
            }
        }
    }
}

/// oracle (degenerate, circular inclined): `ω` is undefined, `from_state`
/// returns `ω = 0` and `ν` = argument of latitude `u = ω_in + ν_in`, and `Ω`,
/// `i` unchanged. The state is reproduced by `to_state` of the result.
#[test]
fn circular_inclined_orbit_uses_argument_of_latitude() {
    let mu = fx(MU);
    let el = elements(7000.0, 0.0, 0.9, 2.0, 1.1, 0.7);
    let s = el.to_state(mu).unwrap();
    let back = OrbitalElements::from_state(&s, mu).unwrap();
    assert!(back.eccentricity < DEGENERACY_TOLERANCE);
    assert_eq!(back.arg_periapsis, Fix128::ZERO);
    assert!(angle_diff(back.raan.to_f64(), 2.0).abs() < 1e-10);
    assert!(angle_diff(back.true_anomaly.to_f64(), 1.8).abs() < 1e-10);
    let s2 = back.to_state(mu).unwrap();
    assert!(norm(v3(s2.position - s.position)) < 1e-8);
    assert!(norm(v3(s2.velocity - s.velocity)) < 1e-11);
}

/// oracle (degenerate, prograde equatorial `i = 0`): `Ω` is undefined,
/// `from_state` returns `Ω = 0` and `ω` = longitude of periapsis `Ω_in + ω_in`.
#[test]
fn prograde_equatorial_orbit_uses_longitude_of_periapsis() {
    let mu = fx(MU);
    let el = elements(9000.0, 0.2, 0.0, 1.0, 0.5, 2.0);
    let back = OrbitalElements::from_state(&el.to_state(mu).unwrap(), mu).unwrap();
    assert_eq!(back.raan, Fix128::ZERO);
    assert!(back.inclination.to_f64().abs() < 1e-12);
    assert!(angle_diff(back.arg_periapsis.to_f64(), 1.5).abs() < 1e-10);
    assert!(angle_diff(back.true_anomaly.to_f64(), 2.0).abs() < 1e-10);
}

/// oracle (degenerate, retrograde equatorial `i = π`): the periapsis
/// direction is `(cos(Ω−ω), sin(Ω−ω), 0)` and angles are measured about
/// `ĥ = −ẑ`, so `from_state` returns `Ω = 0`, `ω = ω_in − Ω_in`.
#[test]
fn retrograde_equatorial_orbit_measures_about_negative_z() {
    let mu = fx(MU);
    let el = elements(9000.0, 0.2, PI, 1.0, 0.5, 2.0);
    let s = el.to_state(mu).unwrap();
    let back = OrbitalElements::from_state(&s, mu).unwrap();
    assert_eq!(back.raan, Fix128::ZERO);
    assert!((back.inclination.to_f64() - PI).abs() < 1e-12);
    assert!(angle_diff(back.arg_periapsis.to_f64(), 0.5 - 1.0).abs() < 1e-10);
    assert!(angle_diff(back.true_anomaly.to_f64(), 2.0).abs() < 1e-10);
    let s2 = back.to_state(mu).unwrap();
    assert!(norm(v3(s2.position - s.position)) < 1e-8);
}

/// oracle (degenerate, circular equatorial): `Ω = ω = 0`, `ν` = true
/// longitude `Ω_in + ω_in + ν_in`.
#[test]
fn circular_equatorial_orbit_uses_true_longitude() {
    let mu = fx(MU);
    let el = elements(7000.0, 0.0, 0.0, 0.3, 0.4, 0.5);
    let back = OrbitalElements::from_state(&el.to_state(mu).unwrap(), mu).unwrap();
    assert_eq!(back.raan, Fix128::ZERO);
    assert_eq!(back.arg_periapsis, Fix128::ZERO);
    assert!(angle_diff(back.true_anomaly.to_f64(), 1.2).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Closed-form laws
// ---------------------------------------------------------------------------

/// oracle: a circular orbit has `|r| = a` and `|v| = √(μ/a)` at every `ν`,
/// and vis-viva at `r = a` gives the same speed. Relative tolerance `1e-12`.
#[test]
fn circular_orbit_speed_is_sqrt_mu_over_r() {
    let mu = fx(MU);
    for a in [6678.137, 20_000.0, 42_164.0] {
        let want = (MU / a).sqrt();
        for nu in [0.0, 1.0, 2.5, 4.0] {
            let s = elements(a, 0.0, 0.5, 0.2, 0.0, nu).to_state(mu).unwrap();
            assert!(((norm(v3(s.position)) - a) / a).abs() < 1e-12);
            assert!(((norm(v3(s.velocity)) - want) / want).abs() < 1e-12);
        }
        let vv = vis_viva_speed(mu, fx(a), fx(a)).unwrap().to_f64();
        assert!(((vv - want) / want).abs() < 1e-12);
    }
}

/// oracle: Kepler's third law in `f64`, `T = 2π√(a³/μ)`, relative `1e-12`;
/// and the geostationary radius `a = 42 164.17 km` gives one sidereal day
/// `86 164.09 s` (tolerance 0.02 s: both published values carry 2 decimals).
#[test]
fn orbital_period_is_keplers_third_law() {
    let mu = fx(MU);
    for a in [6678.137, 7000.0, 26_560.0, 384_400.0] {
        let want = 2.0 * PI * (a * a * a / MU).sqrt();
        let got = orbital_period(mu, fx(a)).unwrap().to_f64();
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "a={a}: {got} vs {want}"
        );
        let n = mean_motion(mu, fx(a)).unwrap().to_f64();
        assert!(((n - 2.0 * PI / want) * want).abs() < 1e-11);
    }
    let geo = orbital_period(mu, fx(42_164.17)).unwrap().to_f64();
    assert!((geo - 86_164.09).abs() < 0.02, "GEO period {geo}");
}

/// oracle: along an eccentric orbit, the speed from `to_state` equals
/// vis-viva `√(μ(2/r − 1/a))` at the same `r`, the specific energy
/// `v²/2 − μ/r` equals `−μ/(2a)` and `|r × v|` equals `√(μa(1−e²))`,
/// all evaluated in `f64` from the state; and the module's
/// `specific_orbital_energy` / `specific_angular_momentum` agree with the
/// same closed forms. Relative tolerance `1e-12`.
#[test]
fn energy_angular_momentum_and_vis_viva_along_orbit() {
    let mu = fx(MU);
    let (a, e) = (15_000.0, 0.6);
    let energy = -MU / (2.0 * a);
    let h = (MU * a * (1.0 - e * e)).sqrt();
    assert!(
        ((specific_orbital_energy(mu, fx(a)).unwrap().to_f64() - energy) / energy).abs() < 1e-12
    );
    assert!(
        ((specific_angular_momentum(mu, fx(a), fx(e))
            .unwrap()
            .to_f64()
            - h)
            / h)
            .abs()
            < 1e-12
    );
    for k in 0..24 {
        let nu = k as f64 * PI / 12.0;
        let s = elements(a, e, 1.2, 0.7, 2.2, nu).to_state(mu).unwrap();
        let (r, v) = (v3(s.position), v3(s.velocity));
        let rn = norm(r);
        let vn = norm(v);
        let want_v = (MU * (2.0 / rn - 1.0 / a)).sqrt();
        assert!(((vn - want_v) / want_v).abs() < 1e-12, "ν={nu}: speed");
        let vv = vis_viva_speed(mu, fx(rn), fx(a)).unwrap().to_f64();
        assert!(
            ((vv - want_v) / want_v).abs() < 1e-12,
            "ν={nu}: vis_viva_speed"
        );
        assert!(
            ((0.5 * vn * vn - MU / rn - energy) / energy).abs() < 1e-12,
            "ν={nu}: energy"
        );
        assert!(((norm(cross(r, v)) - h) / h).abs() < 1e-12, "ν={nu}: h");
        // conic equation r = p / (1 + e cos ν)
        let r_want = a * (1.0 - e * e) / (1.0 + e * nu.cos());
        assert!(((rn - r_want) / r_want).abs() < 1e-12, "ν={nu}: r");
    }
}

// ---------------------------------------------------------------------------
// Propagation
// ---------------------------------------------------------------------------

/// oracle: after one period the state returns to the start.
///
/// Tolerance `1e-10·a` in position: `n·T` reproduces `2π` to a few units of
/// `2⁻⁶⁴·T`, Kepler's equation is solved to `5.7e-14 rad`, and
/// `|dr/dM| ≤ a(1+e)/(1−e)·…` stays below `20·a` for `e ≤ 0.8`, so the
/// position error is `≲ 1e-12·a`; factor 100 margin.
#[test]
fn propagation_over_one_period_returns_to_start() {
    let mu = fx(MU);
    for &(a, e) in &[(7000.0, 0.0), (12_000.0, 0.3), (26_560.0, 0.8)] {
        for nu in [0.0, 1.0, 3.0, 5.0] {
            let el = elements(a, e, 1.0, 0.5, 0.8, nu);
            let s0 = el.to_state(mu).unwrap();
            let t = orbital_period(mu, fx(a)).unwrap();
            let s1 = el.state_at(mu, t).unwrap();
            let d = norm(v3(s1.position - s0.position));
            assert!(d < 1e-10 * a, "a={a} e={e} ν={nu}: Δr = {d:e}");
        }
    }
}

/// oracle: propagation agrees with the `f64` reference — mean anomaly
/// `M(t) = M₀ + √(μ/a³)·t`, Kepler by `ref_kepler`, `ν` by the half-angle
/// formula — at arbitrary times including backward ones. Angle tolerance
/// `1e-9 rad` (`n·t` for `|t|` up to `3·10⁵ s` carries `|t|·ulp`-level error
/// in both arithmetics, `1/(1−e)²` amplification at `e = 0.7` is 11).
#[test]
fn propagation_matches_f64_reference() {
    let mu = fx(MU);
    let (a, e, nu0) = (20_000.0, 0.7, 0.4);
    let el = elements(a, e, 0.3, 1.0, 2.0, nu0);
    let e0 = 2.0 * (((1.0 - e) / (1.0 + e)).sqrt() * (nu0 / 2.0).tan()).atan();
    let m0 = e0 - e * e0.sin();
    let n = (MU / (a * a * a)).sqrt();
    for t in [-300_000.0, -1234.5, 0.0, 60.0, 5000.0, 99_999.0] {
        let got = el.propagate(mu, fx(t)).unwrap().true_anomaly.to_f64();
        let ecc = ref_kepler(m0 + n * t, e);
        let want = 2.0 * (((1.0 + e) / (1.0 - e)).sqrt() * (ecc / 2.0).tan()).atan();
        assert!(
            angle_diff(got, want).abs() < 1e-9,
            "t={t}: ν {got} vs {want}"
        );
    }
}

/// oracle: a circular orbit advances `ν` by `2π·t/T` (quarter period → `π/2`).
#[test]
fn circular_orbit_quarter_period_advances_quarter_turn() {
    let mu = fx(MU);
    let el = elements(8000.0, 0.0, 0.5, 0.0, 0.0, 0.25);
    let quarter = orbital_period(mu, fx(8000.0)).unwrap() / Fix128::from_int(4);
    let got = el.propagate(mu, quarter).unwrap().true_anomaly.to_f64();
    assert!(angle_diff(got, 0.25 + PI / 2.0).abs() < 1e-12, "ν = {got}");
}

/// oracle: `ν → M → ν` is the identity (anomaly conversions are inverse to
/// each other), tolerance `1e-11 rad` at `e ≤ 0.9`.
#[test]
fn anomaly_conversions_are_inverse() {
    for e in [0.0, 0.1, 0.5, 0.9] {
        for k in 0..16 {
            let nu = k as f64 * PI / 8.0 + 0.01;
            let m = mean_from_true_anomaly(fx(nu), fx(e)).unwrap();
            let back = true_from_mean_anomaly(m, fx(e)).unwrap().to_f64();
            assert!(angle_diff(back, nu).abs() < 1e-11, "e={e} ν={nu}: {back}");
        }
    }
}

// ---------------------------------------------------------------------------
// J2 secular rates
// ---------------------------------------------------------------------------

/// oracle: a sun-synchronous orbit. For a 700 km circular orbit
/// (`a = 7078.137 km`, `R = 6378.137 km`, `J2 = 1.082 63·10⁻³`, test-local
/// values) the published sun-synchronous inclination is `98.19°`, where the
/// node must advance one turn per tropical year, `2π / (365.2422·86400 s)
/// = 1.991 06·10⁻⁷ rad/s`.
///
/// Tolerance `1e-3` relative: the published inclination is rounded to
/// `0.005°` and `d(Ω̇)/di = (3/2)·n·J2·(R/p)²·sin i ≈ 1.38·10⁻⁶ rad/s/rad`,
/// which turns `0.005°` into `1.2·10⁻¹⁰ rad/s = 6·10⁻⁴` relative.
#[test]
fn j2_node_rate_of_sun_synchronous_orbit() {
    let (a, r_eq, j2) = (7078.137, 6378.137, 1.082_63e-3);
    let rate = f64_of(
        j2_raan_rate(
            fx(MU),
            fx(a),
            fx(0.0),
            fx(98.19_f64.to_radians()),
            fx(j2),
            fx(r_eq),
        )
        .unwrap(),
    );
    let want = 2.0 * PI / (365.2422 * 86_400.0);
    assert!(((rate - want) / want).abs() < 1e-3, "Ω̇ = {rate:e}");
}

/// oracle: signs and zeros of the J2 rates — the node regresses
/// (`Ω̇ < 0`) for prograde orbits, advances for retrograde ones and stands
/// still for a polar orbit; `ω̇ = 0` at the critical inclination
/// `cos² i = 1/5` and `ω̇ > 0` below it. Zero tolerance `1e-18 rad/s`
/// (CORDIC `cos` error `2⁻⁴⁸` times `|Ω̇|_max ≈ 1.4·10⁻⁶`).
#[test]
fn j2_rate_signs_and_critical_inclination() {
    let (a, e, r_eq, j2) = (fx(7500.0), fx(0.01), fx(6378.137), fx(1.082_63e-3));
    let mu = fx(MU);
    let node = |i: f64| f64_of(j2_raan_rate(mu, a, e, fx(i), j2, r_eq).unwrap());
    let apse = |i: f64| f64_of(j2_arg_periapsis_rate(mu, a, e, fx(i), j2, r_eq).unwrap());
    assert!(node(0.5) < 0.0);
    assert!(node(2.5) > 0.0);
    assert!(
        node(PI / 2.0).abs() < 1e-18,
        "polar Ω̇ = {:e}",
        node(PI / 2.0)
    );
    let crit = (1.0 / 5.0_f64.sqrt()).acos();
    assert!(apse(crit).abs() < 1e-18, "critical ω̇ = {:e}", apse(crit));
    assert!(apse(0.3) > 0.0);
    assert!(apse(1.3) < 0.0);
    // equatorial magnitude ratio |ω̇/Ω̇| = (3/4·4)/(3/2) = 2 at i = 0
    assert!((apse(0.0) / node(0.0) + 2.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Degenerate inputs (each one asserts the documented result)
// ---------------------------------------------------------------------------

/// `μ ≤ 0`, `a ≤ 0`, `e` outside `[0, 1)`, `i` outside `[0, π]`, `r ≤ 0`,
/// `r ≥ 2a`, `R ≤ 0` each return their documented `Err`.
#[test]
fn degenerate_parameters_are_rejected() {
    let mu = fx(MU);
    for bad in [0.0, -1.0] {
        assert_eq!(
            orbital_period(fx(bad), fx(7000.0)),
            Err(KeplerError::NonPositiveGravitationalParameter)
        );
        assert_eq!(
            orbital_period(mu, fx(bad)),
            Err(KeplerError::NonPositiveSemiMajorAxis)
        );
        assert_eq!(
            vis_viva_speed(mu, fx(bad), fx(7000.0)),
            Err(KeplerError::NonPositiveRadius)
        );
        assert_eq!(
            j2_raan_rate(mu, fx(7000.0), fx(0.0), fx(1.0), fx(1e-3), fx(bad)),
            Err(KeplerError::NonPositiveEquatorialRadius)
        );
        assert_eq!(
            OrbitalElements::new(fx(bad), fx(0.1), fx(1.0), fx(0.0), fx(0.0), fx(0.0)),
            Err(KeplerError::NonPositiveSemiMajorAxis)
        );
    }
    assert_eq!(
        vis_viva_speed(mu, fx(14_000.0), fx(7000.0)),
        Err(KeplerError::RadiusUnreachable)
    );
    assert_eq!(
        vis_viva_speed(mu, fx(20_000.0), fx(7000.0)),
        Err(KeplerError::RadiusUnreachable)
    );
    for e in [-0.01, 1.0, 2.0] {
        assert_eq!(
            OrbitalElements::new(fx(7000.0), fx(e), fx(1.0), fx(0.0), fx(0.0), fx(0.0)),
            Err(KeplerError::EccentricityOutOfRange),
            "e = {e}"
        );
        assert_eq!(
            specific_angular_momentum(mu, fx(7000.0), fx(e)),
            Err(KeplerError::EccentricityOutOfRange)
        );
    }
    for i in [-0.01, PI + 1e-6] {
        assert_eq!(
            OrbitalElements::new(fx(7000.0), fx(0.1), fx(i), fx(0.0), fx(0.0), fx(0.0)),
            Err(KeplerError::InclinationOutOfRange)
        );
    }
    let el = elements(7000.0, 0.1, 1.0, 0.0, 0.0, 0.0);
    assert_eq!(
        el.to_state(Fix128::ZERO),
        Err(KeplerError::NonPositiveGravitationalParameter)
    );
    assert_eq!(
        el.propagate(fx(-1.0), fx(10.0)),
        Err(KeplerError::NonPositiveGravitationalParameter)
    );
    // A field set out of range directly (bypassing `new`) is caught too.
    let mut broken = el;
    broken.eccentricity = Fix128::ONE;
    assert_eq!(
        broken.to_state(mu),
        Err(KeplerError::EccentricityOutOfRange)
    );
}

/// State vectors with no elliptic orbit: zero position, radial motion
/// (`r × v = 0`, also `v = 0`), escape speed and above, and `μ ≤ 0`.
#[test]
fn degenerate_states_are_rejected() {
    let mu = fx(MU);
    let r = Vec3Fix::new(fx(7000.0), Fix128::ZERO, Fix128::ZERO);
    let st = |p: Vec3Fix, v: Vec3Fix| StateVector {
        position: p,
        velocity: v,
    };
    assert_eq!(
        OrbitalElements::from_state(&st(Vec3Fix::ZERO, Vec3Fix::UNIT_Y), mu),
        Err(KeplerError::ZeroPosition)
    );
    assert_eq!(
        OrbitalElements::from_state(&st(r, Vec3Fix::ZERO), mu),
        Err(KeplerError::RadialTrajectory)
    );
    assert_eq!(
        OrbitalElements::from_state(
            &st(r, Vec3Fix::new(fx(3.0), Fix128::ZERO, Fix128::ZERO)),
            mu
        ),
        Err(KeplerError::RadialTrajectory)
    );
    let v_esc = (2.0 * MU / 7000.0).sqrt();
    for v in [v_esc * 1.000_001, v_esc * 2.0] {
        assert_eq!(
            OrbitalElements::from_state(
                &st(r, Vec3Fix::new(Fix128::ZERO, fx(v), Fix128::ZERO)),
                mu
            ),
            Err(KeplerError::NotElliptic),
            "v = {v}"
        );
    }
    let ok = st(r, Vec3Fix::new(Fix128::ZERO, fx(7.0), Fix128::ZERO));
    assert_eq!(
        OrbitalElements::from_state(&ok, fx(0.0)),
        Err(KeplerError::NonPositiveGravitationalParameter)
    );
    assert!(OrbitalElements::from_state(&ok, mu).is_ok());
}
