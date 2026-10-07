//! Oracles for `Cloth::set_rest_tether` (a soft sheet springing back to its rest shape).
//!
//! Law: every particle is pulled toward its rest position by `F = k (rest − x) − c v`.
//! With particle mass `m`, `ω = sqrt(k/m)` and `ζ = c / (2 sqrt(k m))`; for `ζ < 1` the
//! continuous return of the error `e = x − rest` is the damped oscillation
//! `e(t) = e^(−ζωt) (e0 cos ω_d t + (v0 + ζω e0)/ω_d · sin ω_d t)`, `ω_d = ω sqrt(1 − ζ²)`.
//!
//! Discrete law: XPBD with damping on `C = x − rest` (gradient I, inverse mass `w`), the
//! multiplier accumulated over the iterations of a substep. The constraint is linear, so
//! the first iteration solves the substep exactly; for an unconstrained particle the update
//! is backward Euler, `x' = (x (1 + 2ζa) + h v) / D`, `D = 1 + 2ζa + a²`, `a = ω h`,
//! `h = dt / substeps`.
//!
//! Error bound (ζ < 1, from rest): the roots of `D z² − (2 + 2ζa) z + 1 = 0` are
//! `z = ((1 + ζa) ± i a sqrt(1 − ζ²)) / D`, with `|z| = D^(−1/2)`, so per step
//! `ln|z| = −ζa − (1 − 2ζ²) a²/2 + O(a³)` and `arg z = a sqrt(1 − ζ²) − ζ sqrt(1 − ζ²) a² + O(a³)`
//! against the continuous `−ζa` and `a sqrt(1 − ζ²)`. Over `s = ωt` the relative amplitude
//! and phase errors are `(1 − 2ζ²) a s / 2` and `ζ sqrt(1 − ζ²) a s`, whose quadrature sum
//! is exactly `a s / 2`. The amplitude of the continuous solution from rest is
//! `|e0| / sqrt(1 − ζ²)`. The start contributes a constant offset: the first step gives
//! `x1 = e0 (1 − a² + O(a³))` against `e0 (1 − a²/2)`, which shifts the sine coefficient by
//! `a |e0| / (2 sqrt(1 − ζ²))`. Hence
//! `|e_n − e(t_n)| <= a |e0| / sqrt(1 − ζ²) · e^(−ζs) (1 + s) / 2 + O(a²)`, maximal at
//! `s = 1/ζ − 1` with value `a |e0| e^(ζ − 1) / (2 ζ sqrt(1 − ζ²))` = `0.734 a |e0|` at
//! ζ = 0.42. The assertions use that bound (+5% for the `O(a²)` rest) and require at least
//! a quarter of it (the triangle inequality makes the bound loose, but it has teeth).

use alice_physics::det_math::exp64;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::Cloth;

/// `n` independent particles (a single row has no triangles, hence no edge or bending
/// constraints), mass `mass`, no gravity, no frame damping.
fn row(n: usize, mass: i64) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(100 * (n.max(2) as i64 - 1)),
        Fix128::ONE,
        n,
        1,
        Fix128::from_int(mass),
    );
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    c
}

/// `k = m ω²`, `c = 2 ζ m ω` with `ζ = zn / zd`.
fn kc(mass: i64, omega: i64, zn: i64, zd: i64) -> (Fix128, Fix128) {
    let (m, w) = (Fix128::from_int(mass), Fix128::from_int(omega));
    (
        m * w * w,
        Fix128::from_int(2) * m * w * Fix128::from_ratio(zn, zd),
    )
}

/// `(sin x, cos x)` by Taylor series after reduction to `[−π, π]` (the platform `f64::sin`
/// is not used in this crate; error ≲ 1e-15 here).
fn sin_cos64(x: f64) -> (f64, f64) {
    let tau = 2.0 * core::f64::consts::PI;
    let mut r = x.rem_euclid(tau);
    if r > core::f64::consts::PI {
        r -= tau;
    }
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    let mut term = 1.0_f64; // r^n / n!
    for n in 0..40_u32 {
        match n % 4 {
            0 => c += term,
            1 => s += term,
            2 => c -= term,
            _ => s -= term,
        }
        term *= r / f64::from(n + 1);
    }
    (s, c)
}

fn underdamped(e0: f64, v0: f64, w: f64, zeta: f64, t: f64) -> f64 {
    let wd = w * (1.0 - zeta * zeta).sqrt();
    let (s, c) = sin_cos64(wd * t);
    exp64(-zeta * w * t) * (e0 * c + (v0 + zeta * w * e0) / wd * s)
}

fn comps(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

#[test]
fn underdamped_return_tracks_the_continuous_closed_form() {
    let zeta = 0.42_f64;
    let coef = exp64(zeta - 1.0) / (2.0 * zeta * (1.0 - zeta * zeta).sqrt());
    assert!((coef - 0.734).abs() < 1e-3);
    for (omega, den, substeps) in [
        (6_i64, 60_i64, 1_usize),
        (12, 60, 2),
        (12, 240, 1),
        (20, 60, 4),
        (4, 30, 1),
    ] {
        let w = omega as f64;
        let h = 1.0 / (den as f64 * substeps as f64);
        let mut cloth = row(1, 2);
        cloth.config.substeps = substeps;
        cloth.config.iterations = 1;
        let rest = cloth.positions[0];
        let (k, c) = kc(2, omega, 42, 100);
        cloth.set_rest_tether(k, c);
        let e0 = [0.5_f64, -0.25, 0.125];
        cloth.positions[0] = rest
            + Vec3Fix::new(
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(-1, 4),
                Fix128::from_ratio(1, 8),
            );
        let mut worst = [0.0_f64; 3];
        let frames = (8.0 / (zeta * w) * den as f64) as i64;
        for n in 1..=frames {
            cloth.step(Fix128::from_ratio(1, den));
            let t = n as f64 / den as f64;
            let e = comps(cloth.positions[0] - rest);
            for i in 0..3 {
                worst[i] = worst[i].max((e[i] - underdamped(e0[i], 0.0, w, zeta, t)).abs());
            }
        }
        for i in 0..3 {
            let bound = coef * w * h * e0[i].abs() * 1.05;
            assert!(
                worst[i] <= bound,
                "ω {omega}, dt 1/{den}, substeps {substeps}, axis {i}: {} > {bound}",
                worst[i]
            );
            assert!(
                worst[i] >= 0.25 * bound,
                "ω {omega}, axis {i}: error {} suspiciously small (bound {bound})",
                worst[i]
            );
        }
    }
}

/// Against an independent f64 implementation of the backward Euler recurrence, with an
/// initial velocity, for under-, critically and over-damped tethers.
#[test]
fn follows_the_discrete_recurrence_with_initial_velocity() {
    for (zn, zd) in [(42_i64, 100_i64), (1, 1), (3, 1)] {
        let zeta = zn as f64 / zd as f64;
        let (w, h) = (9.0_f64, 1.0 / 120.0);
        let mut cloth = row(1, 3);
        cloth.config.substeps = 2;
        cloth.config.iterations = 3;
        let rest = cloth.positions[0];
        let (k, c) = kc(3, 9, zn, zd);
        cloth.set_rest_tether(k, c);
        cloth.positions[0] = rest + Vec3Fix::from_int(1, 0, -1);
        cloth.velocities[0] = Vec3Fix::from_int(-4, 2, 0);
        let (mut x, mut v) = ([1.0_f64, 0.0, -1.0], [-4.0_f64, 2.0, 0.0]);
        let a = w * h;
        let d = 1.0 + 2.0 * zeta * a + a * a;
        for frame in 1..=180 {
            cloth.step(Fix128::from_ratio(1, 60));
            for _ in 0..2 {
                for i in 0..3 {
                    let xn = (x[i] * (1.0 + 2.0 * zeta * a) + h * v[i]) / d;
                    v[i] = (xn - x[i]) / h;
                    x[i] = xn;
                }
            }
            let e = comps(cloth.positions[0] - rest);
            let vel = comps(cloth.velocities[0]);
            for i in 0..3 {
                assert!(
                    (e[i] - x[i]).abs() < 1e-10,
                    "ζ {zeta}, frame {frame}, axis {i}: {} vs {}",
                    e[i],
                    x[i]
                );
                assert!(
                    (vel[i] - v[i]).abs() < 1e-8,
                    "ζ {zeta}, frame {frame}, axis {i}: v {} vs {}",
                    vel[i],
                    v[i]
                );
            }
        }
    }
}

/// ζ = 0.42 overshoots by about the continuous `exp(−πζ/sqrt(1 − ζ²)) = 0.234`; ζ >= 1 never
/// crosses the rest position from rest.
#[test]
fn damping_ratio_controls_overshoot() {
    let run = |zn: i64, zd: i64| {
        let mut cloth = row(1, 1);
        cloth.config.substeps = 4;
        let rest = cloth.positions[0];
        let (k, c) = kc(1, 10, zn, zd);
        cloth.set_rest_tether(k, c);
        cloth.positions[0] = rest + Vec3Fix::from_int(1, 0, 0);
        let mut min = f64::MAX;
        for _ in 0..240 {
            cloth.step(Fix128::from_ratio(1, 60));
            min = min.min((cloth.positions[0] - rest).x.to_f64());
        }
        min
    };
    let under = run(42, 100);
    assert!(under < -0.20 && under > -0.24, "ζ = 0.42 overshoot {under}");
    assert!(run(1, 1) >= 0.0);
    assert!(run(2, 1) >= 0.0);
}

#[test]
fn independent_of_iterations() {
    let run = |iterations: usize| {
        let mut cloth = row(1, 2);
        cloth.config.substeps = 2;
        cloth.config.iterations = iterations;
        let rest = cloth.positions[0];
        let (k, c) = kc(2, 7, 42, 100);
        cloth.set_rest_tether(k, c);
        cloth.positions[0] = rest + Vec3Fix::from_int(0, 2, 1);
        (0..120)
            .map(|_| {
                cloth.step(Fix128::from_ratio(1, 60));
                comps(cloth.positions[0] - rest)
            })
            .collect::<Vec<_>>()
    };
    let one = run(1);
    for iterations in [2, 8, 16] {
        for (n, (a, b)) in one.iter().zip(run(iterations)).enumerate() {
            for i in 0..3 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-15,
                    "iterations {iterations}, frame {n}: {a:?} vs {b:?}"
                );
            }
        }
    }
}

#[test]
fn converges_with_substeps() {
    let (omega, den, zeta) = (10_i64, 30_i64, 0.42_f64);
    let coef = exp64(zeta - 1.0) / (2.0 * zeta * (1.0 - zeta * zeta).sqrt());
    let mut prev = f64::MAX;
    for substeps in [1_usize, 2, 4, 8] {
        let h = 1.0 / (den as f64 * substeps as f64);
        let mut cloth = row(1, 1);
        cloth.config.substeps = substeps;
        let rest = cloth.positions[0];
        let (k, c) = kc(1, omega, 42, 100);
        cloth.set_rest_tether(k, c);
        cloth.positions[0] = rest + Vec3Fix::from_int(1, 0, 0);
        let mut worst = 0.0_f64;
        for n in 1..=60 {
            cloth.step(Fix128::from_ratio(1, den));
            let t = n as f64 / den as f64;
            worst = worst.max(
                ((cloth.positions[0] - rest).x.to_f64() - underdamped(1.0, 0.0, 10.0, zeta, t))
                    .abs(),
            );
        }
        assert!(
            worst <= coef * 10.0 * h * 1.05,
            "substeps {substeps}: {worst}"
        );
        assert!(
            worst < prev,
            "substeps {substeps}: error did not shrink ({worst} >= {prev})"
        );
        prev = worst;
    }
}

fn bits(c: &Cloth) -> Vec<(i64, u64)> {
    c.positions
        .iter()
        .chain(&c.velocities)
        .flat_map(|v| [v.x, v.y, v.z])
        .map(|f| (f.hi, f.lo))
        .collect()
}

/// Particles of different mass (hence different ω) in one cloth each follow exactly the run
/// they would have alone.
#[test]
fn tethers_of_different_particles_are_independent() {
    let masses = [1_i64, 2, 5, 9];
    let (k, c) = (Fix128::from_int(40), Fix128::from_int(3));
    let offsets = [
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(0, -2, 0),
        Vec3Fix::from_int(0, 0, 3),
        Vec3Fix::from_int(1, 1, 1),
    ];
    let mut together = row(4, 1);
    for (i, m) in masses.iter().enumerate() {
        together.inv_masses[i] = Fix128::ONE / Fix128::from_int(*m);
    }
    together.set_rest_tether(k, c);
    for (i, o) in offsets.iter().enumerate() {
        together.positions[i] = together.positions[i] + *o;
    }
    for _ in 0..120 {
        together.step(Fix128::from_ratio(1, 60));
    }
    for (i, m) in masses.iter().enumerate() {
        let mut alone = row(1, 1);
        alone.positions[0] = together.rest_positions()[i];
        alone.prev_positions[0] = alone.positions[0];
        alone.inv_masses[0] = Fix128::ONE / Fix128::from_int(*m);
        alone.set_rest_tether(k, c);
        alone.positions[0] = alone.positions[0] + offsets[i];
        for _ in 0..120 {
            alone.step(Fix128::from_ratio(1, 60));
        }
        assert_eq!(together.positions[i], alone.positions[0], "particle {i}");
        assert_eq!(together.velocities[i], alone.velocities[0], "particle {i}");
    }
}

fn sheet() -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        5,
        5,
        Fix128::from_ratio(1, 25),
    );
    c.config.stretch_compliance = Fix128::from_ratio(1, 1000);
    c.config.self_collision = true;
    c
}

/// A sheet poked out of shape (with gravity, edges, bending and self-collision on) springs
/// back to its rest shape; a pinned corner stays put; a replay is bit-identical.
#[test]
fn a_poked_sheet_springs_back_and_replays_bit_identically() {
    let run = || {
        let mut c = sheet();
        c.pin(0);
        let (k, d) = kc(1, 12, 42, 100);
        // per particle mass 1/25: k = m ω², c = 2 ζ m ω
        let m = Fix128::from_ratio(1, 25);
        c.set_rest_tether(k * m, d * m);
        c.config.gravity = Vec3Fix::ZERO;
        for i in [6_usize, 12, 18] {
            c.positions[i] = c.positions[i]
                + Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(3, 10), Fix128::ZERO);
        }
        let start_pin = c.positions[0];
        for _ in 0..300 {
            c.step(Fix128::from_ratio(1, 60));
        }
        assert_eq!(c.positions[0], start_pin);
        c
    };
    let a = run();
    let worst = a
        .positions
        .iter()
        .zip(a.rest_positions())
        .map(|(x, r)| (*x - *r).length().to_f64())
        .fold(0.0, f64::max);
    assert!(worst < 1e-4, "sheet did not return to rest: {worst}");
    assert_eq!(bits(&a), bits(&run()));
}

// ---- degenerate input ----

/// Mismatched rest positions are rejected without change; zero (or negative, clamped)
/// stiffness and damping leave the cloth bit-identical to one without a tether; clearing
/// the tether restores the plain step; a stiffness far beyond the substep lands on rest.
#[test]
fn degenerate_rest_tethers() {
    let mut c = sheet();
    assert_eq!(c.rest_tether(), None);
    assert!(c.rest_positions().is_empty());
    let err = c.set_rest_positions(&[Vec3Fix::ZERO; 3]).unwrap_err();
    assert_eq!((err.expected, err.got), (25, 3));
    assert!(c.rest_positions().is_empty());

    let mut plain = sheet();
    let mut zero = sheet();
    zero.set_rest_tether(Fix128::ZERO, Fix128::from_int(-5));
    assert_eq!(zero.rest_tether(), Some((Fix128::ZERO, Fix128::ZERO)));
    for _ in 0..30 {
        plain.step(Fix128::from_ratio(1, 60));
        zero.step(Fix128::from_ratio(1, 60));
    }
    assert_eq!(bits(&plain), bits(&zero));

    let mut cleared = sheet();
    cleared.set_rest_tether(Fix128::from_int(5), Fix128::ONE);
    cleared.clear_rest_tether();
    assert_eq!(cleared.rest_tether(), None);
    assert_eq!(cleared.rest_positions().len(), 25);
    let mut plain = sheet();
    for _ in 0..30 {
        plain.step(Fix128::from_ratio(1, 60));
        cleared.step(Fix128::from_ratio(1, 60));
    }
    assert_eq!(bits(&plain), bits(&cleared));

    // explicit rest positions, then a very stiff tether: one substep lands on rest
    let mut stiff = row(2, 1);
    let rest = [Vec3Fix::from_int(3, 3, 3), Vec3Fix::from_int(-3, 0, 1)];
    stiff.set_rest_positions(&rest).unwrap();
    stiff.set_rest_tether(Fix128::from_int(1_000_000_000_000), Fix128::ZERO);
    assert_eq!(stiff.rest_positions(), &rest);
    stiff.step(Fix128::from_ratio(1, 60));
    for (i, (x, r)) in stiff.positions.iter().zip(&rest).enumerate() {
        assert!((*x - *r).length().to_f64() < 1e-6, "particle {i}");
    }
}
