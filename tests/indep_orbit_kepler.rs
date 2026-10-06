//! Second, independent set of oracles for `kepler`.
//!
//! `tests/analytic_kepler.rs` uses km / s with the Earth's `μ`, Vallado and
//! Meeus worked examples, a plain `f64` Newton solver, and the COE2RV
//! perifocal-rotation formula. This file deliberately uses none of those:
//!
//! - units: canonical (`μ = 1`) and heliocentric AU / year (`μ = 4π²`, where
//!   Kepler's third law reads `T = a^{3/2}` exactly);
//! - Kepler's equation: bisection on `[0, 2π]` and the Bessel series
//!   `E = M + Σ (2/n) J_n(n e) sin(n M)` (Watson, *Bessel Functions*, §17.2);
//! - state vectors: built in `f64` from a Rodrigues axis-angle rotation of the
//!   perifocal frame, with the expected `i`, `Ω`, `ω`, `ν` read off the
//!   geometric definitions (`cos i = ĥ·ẑ`, node line `ẑ × h`, periapsis
//!   direction `R·x̂`) instead of an Euler-angle composition;
//! - propagation: the eccentric-anomaly form of the position
//!   `r = a(cos E − e, √(1−e²) sin E)` and velocity
//!   `v = √(μa)/r · (−sin E, √(1−e²) cos E)` (Danby §6.3), not the
//!   true-anomaly form the module uses.
//!
//! No expected value is produced by calling the function under test.

#![allow(clippy::disallowed_methods)]

use alice_physics::kepler::{
    j2_arg_periapsis_rate, j2_raan_rate, orbital_period, solve_kepler, vis_viva_speed, KeplerError,
    OrbitalElements, StateVector, DEGENERACY_TOLERANCE,
};
use alice_physics::{Fix128, Vec3Fix};
use core::f64::consts::PI;

type V = [f64; 3];

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// Sign-magnitude conversion (`to_f64` is only absolutely accurate for
/// negative values).
fn f(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn v3(v: Vec3Fix) -> V {
    [f(v.x), f(v.y), f(v.z)]
}

fn fv(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}

fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Signed smallest difference of two angles.
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0 * PI);
    if d > PI {
        d - 2.0 * PI
    } else {
        d
    }
}

/// Rodrigues rotation of `v` about the unit axis `k` by `theta`.
fn rotate(v: V, k: V, theta: f64) -> V {
    let (s, c) = theta.sin_cos();
    add(
        add(scale(v, c), scale(cross(k, v), s)),
        scale(k, dot(k, v) * (1.0 - c)),
    )
}

fn unit(a: V) -> V {
    scale(a, 1.0 / norm(a))
}

/// Kepler's equation by bisection on `[0, 2π]` for `M` in `[0, 2π)`
/// (`f(E) = E − e sin E − M` is increasing, `f(0) ≤ 0 ≤ f(2π)`).
fn kepler_bisect(m: f64, e: f64) -> f64 {
    let m = m.rem_euclid(2.0 * PI);
    let (mut lo, mut hi) = (0.0_f64, 2.0 * PI);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid - e * mid.sin() - m < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Bessel function of the first kind by its power series.
fn bessel_j(n: u32, x: f64) -> f64 {
    let half = x / 2.0;
    // first term (x/2)^n / n!
    let mut term = 1.0;
    for k in 1..=n {
        term *= half / k as f64;
    }
    let mut sum = term;
    for k in 1..80 {
        term *= -half * half / (k as f64 * (k + n) as f64);
        sum += term;
    }
    sum
}

/// Kepler's equation by the Bessel series (converges for `e < 0.6627`).
fn kepler_bessel(m: f64, e: f64) -> f64 {
    let mut ecc = m;
    for n in 1..=60_u32 {
        let nf = f64::from(n);
        ecc += 2.0 / nf * bessel_j(n, nf * e) * (nf * m).sin();
    }
    ecc
}

/// An orbit set up in `f64`: perifocal axes `P` (periapsis) and `Q` rotated
/// into the inertial frame by a Rodrigues rotation.
struct Orbit {
    mu: f64,
    a: f64,
    e: f64,
    p_axis: V,
    q_axis: V,
}

impl Orbit {
    fn new(mu: f64, a: f64, e: f64, axis: V, angle: f64) -> Self {
        let k = unit(axis);
        Self {
            mu,
            a,
            e,
            p_axis: rotate([1.0, 0.0, 0.0], k, angle),
            q_axis: rotate([0.0, 1.0, 0.0], k, angle),
        }
    }

    /// State at eccentric anomaly `E` (Danby §6.3 form).
    fn state_at_eccentric(&self, ecc: f64) -> (V, V) {
        let (s, c) = ecc.sin_cos();
        let b = (1.0 - self.e * self.e).sqrt();
        let x = self.a * (c - self.e);
        let y = self.a * b * s;
        let r = self.a * (1.0 - self.e * c);
        let k = (self.mu * self.a).sqrt() / r;
        let pos = add(scale(self.p_axis, x), scale(self.q_axis, y));
        let vel = add(scale(self.p_axis, -k * s), scale(self.q_axis, k * b * c));
        (pos, vel)
    }

    fn eccentric_from_true(&self, nu: f64) -> f64 {
        2.0 * (((1.0 - self.e) / (1.0 + self.e)).sqrt() * (nu / 2.0).tan()).atan()
    }

    fn state_vector(&self, ecc: f64) -> StateVector {
        let (p, v) = self.state_at_eccentric(ecc);
        StateVector {
            position: fv(p),
            velocity: fv(v),
        }
    }

    fn mean_motion(&self) -> f64 {
        (self.mu / (self.a * self.a * self.a)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Kepler's equation
// ---------------------------------------------------------------------------

/// oracle: `solve_kepler` agrees with a bisection on `[0, 2π]` (no Newton,
/// no bracket around `M`) for high eccentricities, including the hard corner
/// `e → 1`, `M → 0` where the solution is `E ≈ (6M)^{1/3}`.
///
/// Tolerance: the solver stops `≤ 5.7·10⁻¹⁴` from the root and its `sin` is
/// accurate to `≈ 2⁻⁴⁸`; an error `δ` in `e·sin E` moves the root by
/// `δ/(1 − e cos E) ≤ δ/(1 − e)`, so `1e-13 + 1e-14/(1 − e)`.
#[test]
fn kepler_matches_bisection_at_high_eccentricity() {
    for &e in &[0.75, 0.9, 0.95, 0.99, 0.999] {
        for &m in &[1e-4, 1e-3, 0.02, 0.3, 1.1, 2.9, 3.3, 4.7, 6.2] {
            let ef = fx(e);
            let mf = fx(m);
            let got = f(solve_kepler(mf, ef).unwrap());
            let want = kepler_bisect(f(mf), f(ef));
            let tol = 1e-13 + 1e-14 / (1.0 - e);
            assert!(
                angle_diff(got, want).abs() < tol,
                "e={e} M={m}: {got} vs bisection {want} (Δ {:e})",
                angle_diff(got, want)
            );
        }
    }
}

/// oracle: the Bessel series `E = M + Σ (2/n) J_n(ne) sin(nM)` (Lagrange /
/// Bessel 1824), a solution of Kepler's equation that involves no iteration
/// at all. Valid for `e` below the Laplace limit `0.6627`; truncated at
/// `n = 60` its tail is below `1e-15` for `e ≤ 0.5`. Tolerance `1e-12`.
#[test]
fn kepler_matches_bessel_series() {
    for &e in &[0.05, 0.2, 0.35, 0.5] {
        for k in 0..25 {
            let m = -3.0 + 0.27 * f64::from(k);
            let got = f(solve_kepler(fx(m), fx(e)).unwrap());
            let want = kepler_bessel(f(fx(m)), f(fx(e)));
            assert!(
                (got - want).abs() < 1e-12,
                "e={e} M={m}: {got} vs Bessel {want}"
            );
        }
    }
}

/// oracle: a mean anomaly many turns away gives the root of the same
/// equation on that turn: `E(M + 2πk) = E(M) + 2πk` for `k = ±3, +11`.
#[test]
fn kepler_is_two_pi_periodic_in_the_root() {
    let e = fx(0.87);
    for &m in &[0.4, 2.0, 5.1] {
        let base = f(solve_kepler(fx(m), e).unwrap());
        for k in [-3_i32, 3, 11] {
            let shifted = m + 2.0 * PI * f64::from(k);
            let got = f(solve_kepler(fx(shifted), e).unwrap());
            let want = base + 2.0 * PI * f64::from(k);
            assert!((got - want).abs() < 1e-11, "M={m} k={k}: {got} vs {want}");
        }
    }
}

// ---------------------------------------------------------------------------
// State vector → elements
// ---------------------------------------------------------------------------

/// oracle: elements recovered from an `f64` state built by a Rodrigues
/// rotation equal the geometric definitions: `a` and `e` as set,
/// `cos i = ĥ_z`, `Ω = atan2(n_y, n_x)` with `n = ẑ × ĥ`, `ω` the angle from
/// `n` to the periapsis direction about `ĥ`, `ν` as set.
#[test]
fn elements_from_rotated_state_match_geometric_definitions() {
    let cases: &[(f64, f64, V, f64)] = &[
        (1.0, 0.42, [0.3, -0.8, 0.5], 1.1),
        (3.7, 0.9, [-0.6, 0.2, 0.77], 2.6),
        (0.8, 0.99, [0.9, 0.4, -0.2], -2.2),
        (2.2, 0.15, [0.1, 1.0, 0.05], 0.7),
    ];
    for &(a, e, axis, angle) in cases {
        let orbit = Orbit::new(1.0, a, e, axis, angle);
        let h_hat = unit(cross(orbit.p_axis, orbit.q_axis));
        let node = cross([0.0, 0.0, 1.0], h_hat);
        let want_i = h_hat[2].acos();
        let want_raan = node[1].atan2(node[0]).rem_euclid(2.0 * PI);
        let want_argp = dot(h_hat, cross(node, orbit.p_axis))
            .atan2(dot(node, orbit.p_axis))
            .rem_euclid(2.0 * PI);
        for &nu in &[0.1, 2.2, 4.4] {
            let state = orbit.state_vector(orbit.eccentric_from_true(nu));
            let el = OrbitalElements::from_state(&state, fx(1.0)).unwrap();
            let ctx = format!("a={a} e={e} axis={axis:?} ν={nu}");
            assert!(
                ((f(el.semi_major_axis) - a) / a).abs() < 1e-11,
                "{ctx}: a = {}",
                f(el.semi_major_axis)
            );
            assert!((f(el.eccentricity) - e).abs() < 1e-11, "{ctx}: e");
            assert!((f(el.inclination) - want_i).abs() < 1e-11, "{ctx}: i");
            for (name, got, want) in [
                ("Ω", el.raan, want_raan),
                ("ω", el.arg_periapsis, want_argp),
                ("ν", el.true_anomaly, nu),
            ] {
                let d = angle_diff(f(got), want);
                assert!(d.abs() < 1e-10, "{ctx}: {name} {} vs {want}", f(got));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Propagation
// ---------------------------------------------------------------------------

/// oracle: `state_at(Δt)` equals the eccentric-anomaly form of the orbit at
/// `M₀ + nΔt`, `E` by bisection. Covers `e = 0.9` and `0.99` at times that
/// cross periapsis, and negative `Δt`.
///
/// Tolerance `1e-9·a`: `|dr/dM| ≤ a√((1+e)/(1−e))·…` reaches `≈ 14a` at
/// `e = 0.99`; with the Kepler solve at `≤ 1e-12` rad there and `n·Δt`
/// rounded at `2⁻⁶⁴·|Δt|`, the position error is `≲ 1e-11·a`.
#[test]
fn propagation_matches_eccentric_anomaly_closed_form() {
    for &(a, e) in &[(1.0, 0.0), (1.6, 0.3), (1.0, 0.9), (0.5, 0.99)] {
        let orbit = Orbit::new(1.0, a, e, [0.2, 0.7, -0.4], 0.9);
        let ecc0 = 2.5;
        let el = OrbitalElements::from_state(&orbit.state_vector(ecc0), fx(1.0)).unwrap();
        let m0 = ecc0 - e * ecc0.sin();
        let period = 2.0 * PI / orbit.mean_motion();
        for &frac in &[-0.73, -0.1, 0.05, 0.31, 0.5, 0.99, 2.4] {
            let dt = frac * period;
            let got = el.state_at(fx(1.0), fx(dt)).unwrap();
            let want = orbit.state_at_eccentric(kepler_bisect(m0 + orbit.mean_motion() * dt, e));
            let dr = norm(sub(v3(got.position), want.0));
            let dv = norm(sub(v3(got.velocity), want.1));
            let vscale = norm(want.1);
            assert!(dr < 1e-9 * a, "a={a} e={e} t/T={frac}: Δr {dr:e}");
            assert!(
                dv < 1e-9 * vscale,
                "a={a} e={e} t/T={frac}: Δv {dv:e} (|v| {vscale})"
            );
        }
    }
}

/// oracle: along a propagated `e = 0.99` orbit the specific energy
/// `v²/2 − μ/r` stays `−μ/(2a)`, `|r × v|` stays `√(μa(1−e²))`, and the speed
/// equals `vis_viva_speed` at the same radius (rel. `1e-10`; at periapsis
/// `v²/2` and `μ/r` are `≈ 100` and cancel to `−0.5`).
#[test]
fn energy_and_angular_momentum_conserved_along_eccentric_orbit() {
    let (mu, a, e) = (1.0, 1.0, 0.99);
    let orbit = Orbit::new(mu, a, e, [0.5, -0.5, 0.7], 1.3);
    let el = OrbitalElements::from_state(&orbit.state_vector(0.0), fx(mu)).unwrap();
    let energy = -mu / (2.0 * a);
    let h = (mu * a * (1.0 - e * e)).sqrt();
    let period = 2.0 * PI;
    let mut min_r = f64::MAX;
    for k in 0..97 {
        let t = period * f64::from(k) / 97.0;
        let s = el.state_at(fx(mu), fx(t)).unwrap();
        let (r, v) = (v3(s.position), v3(s.velocity));
        let (rn, vn) = (norm(r), norm(v));
        min_r = min_r.min(rn);
        let eps = 0.5 * vn * vn - mu / rn;
        assert!(((eps - energy) / energy).abs() < 1e-10, "k={k}: ε {eps}");
        assert!(((norm(cross(r, v)) - h) / h).abs() < 1e-10, "k={k}: h");
        let vv = f(vis_viva_speed(fx(mu), fx(rn), fx(a)).unwrap());
        assert!(
            ((vv - vn) / vn).abs() < 1e-10,
            "k={k}: vis-viva {vv} vs {vn}"
        );
    }
    // periapsis a(1−e) = 0.01 is sampled at t = 0
    assert!((min_r - 0.01).abs() < 1e-12, "min r {min_r}");
}

/// oracle: a circular orbit (`e = 0`) in an inclined plane advances by
/// exactly `2π t/T`: after `T/3` the position has turned by `120°` about `ĥ`
/// and its radius is unchanged.
#[test]
fn circular_orbit_turns_uniformly() {
    let orbit = Orbit::new(1.0, 2.5, 0.0, [1.0, 1.0, 1.0], 0.6);
    let s0 = orbit.state_vector(0.4);
    let el = OrbitalElements::from_state(&s0, fx(1.0)).unwrap();
    assert!(
        el.eccentricity < DEGENERACY_TOLERANCE,
        "e = {}",
        f(el.eccentricity)
    );
    let t = 2.0 * PI * 2.5_f64.powf(1.5) / 3.0;
    let s1 = el.state_at(fx(1.0), fx(t)).unwrap();
    let (r0, r1) = (v3(s0.position), v3(s1.position));
    assert!((norm(r1) - 2.5).abs() < 1e-11, "|r| {}", norm(r1));
    let h_hat = unit(cross(orbit.p_axis, orbit.q_axis));
    let turned = dot(h_hat, cross(r0, r1)).atan2(dot(r0, r1));
    assert!(
        (turned - 2.0 * PI / 3.0).abs() < 1e-11,
        "turned {turned} rad"
    );
}

/// oracle: propagation is invertible: `+Δt` then `−Δt` returns `ν`
/// (`e = 0.95`, `Δt` = 1.37 periods).
#[test]
fn forward_then_backward_propagation_is_identity() {
    let el = OrbitalElements::new(fx(1.3), fx(0.95), fx(0.4), fx(2.0), fx(5.0), fx(0.3)).unwrap();
    let mu = fx(1.0);
    let dt = fx(1.37 * 2.0 * PI * 1.3_f64.powf(1.5));
    let there = el.propagate(mu, dt).unwrap();
    let back = there.propagate(mu, -dt).unwrap();
    assert!(
        angle_diff(f(back.true_anomaly), 0.3).abs() < 1e-10,
        "ν {}",
        f(back.true_anomaly)
    );
    // zero time step: same orbit, ν reproduced (not necessarily bit for bit,
    // it goes through ν → M → E → ν)
    let same = el.propagate(mu, Fix128::ZERO).unwrap();
    assert_eq!(same.semi_major_axis, el.semi_major_axis);
    assert_eq!(same.eccentricity, el.eccentricity);
    assert!(angle_diff(f(same.true_anomaly), 0.3).abs() < 1e-11);
}

// ---------------------------------------------------------------------------
// Kepler's third law in heliocentric units
// ---------------------------------------------------------------------------

/// oracle: with `μ = 4π² AU³/yr²` Kepler's third law is `T = a^{3/2}` years
/// (Earth `a = 1 → T = 1`; Jupiter `a = 5.2026 → 11.867`; Mercury
/// `a = 0.387 1 → 0.240 8`). Relative `1e-12`.
#[test]
fn period_in_years_is_a_to_three_halves() {
    let mu = fx(4.0 * PI * PI);
    for &a in &[1.0, 5.2026, 0.3871, 30.07] {
        let got = f(orbital_period(mu, fx(a)).unwrap());
        let want = a.powf(1.5);
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "a={a}: {got} vs {want}"
        );
    }
}

/// oracle: the ratio of the two J2 secular rates depends only on `i`:
/// `ω̇/Ω̇ = −(5 cos² i − 1)/(2 cos i)` (from Vallado §9.6's two formulas,
/// `n`, `J2`, `R`, `p` cancel). Checked for inclinations not used elsewhere.
#[test]
fn j2_rate_ratio_depends_only_on_inclination() {
    for &i in &[0.2, 0.7, 1.3, 2.0, 2.8] {
        let args = (fx(1.0), fx(1.8), fx(0.25), fx(i), fx(1e-3), fx(1.0));
        let raan = f(j2_raan_rate(args.0, args.1, args.2, args.3, args.4, args.5).unwrap());
        let argp =
            f(j2_arg_periapsis_rate(args.0, args.1, args.2, args.3, args.4, args.5).unwrap());
        let c = i.cos();
        let want = -(5.0 * c * c - 1.0) / (2.0 * c);
        assert!(
            ((argp / raan) - want).abs() < 1e-9 * want.abs().max(1.0),
            "i={i}: {} vs {want}",
            argp / raan
        );
    }
}

// ---------------------------------------------------------------------------
// Degenerate input: explicit outcomes
// ---------------------------------------------------------------------------

/// Parabolic (`v² = 2μ/r` exactly, energy `0`) and hyperbolic states are
/// `NotElliptic`; `e = 1`, `e = 1.5` in Kepler's equation are
/// `EccentricityOutOfRange`; `r = 2a` exactly in vis-viva is
/// `RadiusUnreachable`.
#[test]
fn unbound_orbits_are_rejected() {
    let mu = fx(1.0);
    // r = 2, v = 1: v²/2 − μ/r = 0.5 − 0.5 = 0 exactly in fixed point
    let parabolic = StateVector {
        position: Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
        velocity: Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
    };
    assert_eq!(
        OrbitalElements::from_state(&parabolic, mu).err(),
        Some(KeplerError::NotElliptic)
    );
    let hyperbolic = StateVector {
        position: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        velocity: Vec3Fix::new(Fix128::ZERO, fx(1.6), fx(0.2)),
    };
    assert_eq!(
        OrbitalElements::from_state(&hyperbolic, mu).err(),
        Some(KeplerError::NotElliptic)
    );
    // just below escape speed is bound
    let bound = StateVector {
        position: Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
        velocity: Vec3Fix::new(Fix128::ZERO, fx(0.999_999), Fix128::ZERO),
    };
    let el = OrbitalElements::from_state(&bound, mu).unwrap();
    // a = −μ/(2ε), ε = v²/2 − 1/2
    let v: f64 = 0.999_999;
    let want_a = -1.0 / (v * v - 1.0);
    assert!(((f(el.semi_major_axis) - want_a) / want_a).abs() < 1e-9);

    for e in [Fix128::ONE, fx(1.5)] {
        assert_eq!(
            solve_kepler(fx(0.5), e),
            Err(KeplerError::EccentricityOutOfRange)
        );
    }
    assert_eq!(
        vis_viva_speed(mu, Fix128::from_int(4), Fix128::from_int(2)),
        Err(KeplerError::RadiusUnreachable)
    );
    // r just inside 2a: v² = μ(2/r − 1/a) > 0
    let v = f(vis_viva_speed(mu, fx(3.9), Fix128::from_int(2)).unwrap());
    let want = (2.0 / 3.9 - 0.5_f64).sqrt();
    assert!(((v - want) / want).abs() < 1e-12, "{v} vs {want}");
}

/// `r = 0`, a radial trajectory, `μ ≤ 0` and `a ≤ 0` each have their own
/// error; zero time propagation of an invalid element set is an error, not a
/// silent return.
#[test]
fn degenerate_states_and_parameters() {
    let mu = fx(1.0);
    let zero = StateVector {
        position: Vec3Fix::ZERO,
        velocity: Vec3Fix::UNIT_Y,
    };
    assert_eq!(
        OrbitalElements::from_state(&zero, mu).err(),
        Some(KeplerError::ZeroPosition)
    );
    let radial = StateVector {
        position: Vec3Fix::new(fx(0.0), fx(-3.0), fx(0.0)),
        velocity: Vec3Fix::new(fx(0.0), fx(0.2), fx(0.0)),
    };
    assert_eq!(
        OrbitalElements::from_state(&radial, mu).err(),
        Some(KeplerError::RadialTrajectory)
    );
    assert_eq!(
        orbital_period(Fix128::ZERO, Fix128::ONE),
        Err(KeplerError::NonPositiveGravitationalParameter)
    );
    assert_eq!(
        orbital_period(mu, fx(-1.0)),
        Err(KeplerError::NonPositiveSemiMajorAxis)
    );
    let mut bad =
        OrbitalElements::new(fx(1.0), fx(0.5), fx(0.1), fx(0.0), fx(0.0), fx(0.0)).unwrap();
    bad.eccentricity = fx(1.2);
    assert_eq!(
        bad.propagate(mu, Fix128::ZERO).err(),
        Some(KeplerError::EccentricityOutOfRange)
    );
    assert_eq!(
        bad.to_state(mu).err(),
        Some(KeplerError::EccentricityOutOfRange)
    );
}
