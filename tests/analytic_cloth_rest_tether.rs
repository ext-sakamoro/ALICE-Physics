//! Oracles for `Cloth::set_rest_tether` (a soft sheet springing back to its rest shape).
//!
//! Law: every particle is pulled toward its rest position by `F = k (rest − x) − c v`, on top
//! of gravity and wind. With particle mass `m`, `ω = sqrt(k/m)` and `ζ = c / (2 sqrt(k m))`;
//! the error `e = x − rest` obeys `ë = −ω² e − 2ζω ė + a`. For `ζ < 1` its continuous
//! solution is the damped oscillation about `e∞ = a / ω²`,
//! `e(t) = e∞ + e^(−ζωt) (u0 cos ω_d t + (v0 + ζω u0)/ω_d · sin ω_d t)`, `u0 = e0 − e∞`,
//! `ω_d = ω sqrt(1 − ζ²)`; for `ζ = 1` it is `e∞ + e^(−ωt) (u0 + (v0 + ω u0) t)` and for
//! `ζ > 1` the sum of the two decaying exponentials.
//!
//! Discrete law: every substep advances `(e, v)` by the exact solution over `h` (the 2×2
//! matrix exponential, evaluated in `Fix128`), then the sheet's own constraints act. A
//! particle without constraints therefore follows the continuous solution at every frame,
//! for any `substeps` and any `ω h`; the only error is the `Fix128` evaluation (a few units
//! of 2⁻⁶⁴ per substep). The assertions use 1e-9.
//!
//! Before this law the tether was an XPBD constraint (backward Euler per substep), whose
//! numerical damping cut the ζ = 0.42 overshoot to 0.204 at one substep against the
//! continuous 0.2337; those bounds are replaced by the exact ones here.
//!
//! The expected values below are the closed forms evaluated in `f64` (with a Taylor sine /
//! cosine and the crate's deterministic `exp64`); the implementation is never called to
//! produce them.

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

/// Continuous `(e(t), v(t))` of `ë = −ω² e − 2ζω ė + a` from `(e0, v0)`, all three regimes
/// (`ω = 0` is the pure damper `ë = −γ ė + a` with `γ = gamma`).
fn exact(e0: f64, v0: f64, w: f64, zeta: f64, gamma: f64, a: f64, t: f64) -> (f64, f64) {
    if w == 0.0 {
        let d = exp64(-gamma * t);
        let vinf = a / gamma;
        let e = e0 + vinf * t + (v0 - vinf) * (1.0 - d) / gamma;
        return (e, vinf + (v0 - vinf) * d);
    }
    let einf = a / (w * w);
    let u0 = e0 - einf;
    if (zeta - 1.0).abs() < 1e-12 {
        let d = exp64(-w * t);
        let b = v0 + w * u0;
        return (einf + d * (u0 + b * t), d * (b - w * (u0 + b * t)));
    }
    if zeta < 1.0 {
        let wd = w * (1.0 - zeta * zeta).sqrt();
        let (s, c) = sin_cos64(wd * t);
        let d = exp64(-zeta * w * t);
        let b = (v0 + zeta * w * u0) / wd;
        let e = einf + d * (u0 * c + b * s);
        let v = d * ((b * wd - zeta * w * u0) * c - (u0 * wd + zeta * w * b) * s);
        return (e, v);
    }
    let r = w * (zeta * zeta - 1.0).sqrt();
    let (l1, l2) = (zeta * w - r, zeta * w + r);
    // u = A e^(−l1 t) + B e^(−l2 t), A + B = u0, −l1 A − l2 B = v0
    let bb = -(v0 + l1 * u0) / (l2 - l1);
    let aa = u0 - bb;
    let (d1, d2) = (exp64(-l1 * t), exp64(-l2 * t));
    (einf + aa * d1 + bb * d2, -l1 * aa * d1 - l2 * bb * d2)
}

fn comps(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// One particle without constraints follows the continuous solution at every frame to 1e-9,
/// for substeps 1 / 2 / 4, `ω h` from 0.1 to 2, under-, critically and over-damped, with an
/// initial velocity and a constant acceleration (gravity).
#[test]
fn a_lone_particle_follows_the_continuous_solution_at_every_frame() {
    let mass = 2_i64;
    let mut cases = 0;
    for substeps in [1_usize, 2, 4] {
        // ω h = ω / (60 · substeps) ∈ {0.1, 0.5, 1, 2}
        for wh_tenths in [1_i64, 5, 10, 20] {
            let omega = 6 * wh_tenths * substeps as i64;
            for (zn, zd) in [(0_i64, 1_i64), (42, 100), (1, 1), (3, 1)] {
                let zeta = zn as f64 / zd as f64;
                let w = omega as f64;
                let mut cloth = row(1, mass);
                cloth.config.substeps = substeps;
                cloth.config.gravity = Vec3Fix::new(
                    Fix128::from_ratio(1, 2),
                    Fix128::from_ratio(-49, 5),
                    Fix128::ZERO,
                );
                let rest = cloth.positions[0];
                let (k, c) = kc(mass, omega, zn, zd);
                cloth.set_rest_tether(k, c);
                let e0 = [0.5_f64, -0.25, 0.125];
                let v0 = [-3.0_f64, 1.5, 0.0];
                let acc = [0.5_f64, -9.8, 0.0];
                cloth.positions[0] = rest
                    + Vec3Fix::new(
                        Fix128::from_ratio(1, 2),
                        Fix128::from_ratio(-1, 4),
                        Fix128::from_ratio(1, 8),
                    );
                cloth.velocities[0] =
                    Vec3Fix::new(Fix128::from_int(-3), Fix128::from_ratio(3, 2), Fix128::ZERO);
                for n in 1..=120 {
                    cloth.step(Fix128::from_ratio(1, 60));
                    let t = n as f64 / 60.0;
                    let e = comps(cloth.positions[0] - rest);
                    let v = comps(cloth.velocities[0]);
                    for i in 0..3 {
                        let (ee, ve) = exact(e0[i], v0[i], w, zeta, 2.0 * zeta * w, acc[i], t);
                        assert!(
                            (e[i] - ee).abs() < 1e-9,
                            "substeps {substeps}, ωh {}, ζ {zeta}, frame {n}, axis {i}: \
                             e {} vs {ee}",
                            wh_tenths as f64 / 10.0,
                            e[i]
                        );
                        assert!(
                            (v[i] - ve).abs() < 1e-9 * w.max(1.0),
                            "substeps {substeps}, ωh {}, ζ {zeta}, frame {n}, axis {i}: \
                             v {} vs {ve}",
                            wh_tenths as f64 / 10.0,
                            v[i]
                        );
                    }
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 48);
}

/// `k = 0`: a pure damper `ë = −γ ė + a` (the overdamped branch with one zero rate).
#[test]
fn a_pure_damper_follows_its_exponential() {
    let mut cloth = row(1, 4);
    cloth.config.substeps = 3;
    cloth.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let rest = cloth.positions[0];
    // γ = c / m = 6
    cloth.set_rest_tether(Fix128::ZERO, Fix128::from_int(24));
    cloth.velocities[0] = Vec3Fix::from_int(5, 0, 0);
    for n in 1..=90 {
        cloth.step(Fix128::from_ratio(1, 60));
        let t = n as f64 / 60.0;
        let e = comps(cloth.positions[0] - rest);
        let v = comps(cloth.velocities[0]);
        let (ex, vx) = exact(0.0, 5.0, 0.0, 0.0, 6.0, 0.0, t);
        let (ey, vy) = exact(0.0, 0.0, 0.0, 0.0, 6.0, -10.0, t);
        assert!(
            (e[0] - ex).abs() < 1e-9 && (v[0] - vx).abs() < 1e-9,
            "frame {n} x"
        );
        assert!(
            (e[1] - ey).abs() < 1e-9 && (v[1] - vy).abs() < 1e-9,
            "frame {n} y"
        );
    }
}

/// The site's 餅 spring (`T = 0.5 s`, `ω = 4π`, ζ = 0.42) released from rest overshoots by
/// the continuous `exp(−πζ / sqrt(1 − ζ²)) = 0.2337` at one substep as at four: the sampled
/// minimum matches the continuous one sampled at the same frames to 1e-9, and lies within
/// the 60 Hz sampling of the true peak.
#[test]
fn zeta_042_overshoots_by_the_continuous_amount_at_any_substep() {
    let zeta = 0.42_f64;
    let w = 4.0 * core::f64::consts::PI;
    let peak = exp64(-core::f64::consts::PI * zeta / (1.0 - zeta * zeta).sqrt());
    assert!((peak - 0.2337).abs() < 1e-4);
    let mut want = f64::MAX;
    for n in 1..=120 {
        want = want.min(exact(1.0, 0.0, w, zeta, 2.0 * zeta * w, 0.0, n as f64 / 60.0).0);
    }
    // the true peak falls between frames 16 and 17; sampling costs at most ~1e-3
    assert!((-want - peak).abs() < 2e-3, "{want} vs −{peak}");
    for substeps in [1_usize, 2, 4] {
        let mut cloth = row(1, 1);
        cloth.config.substeps = substeps;
        let rest = cloth.positions[0];
        let omega = Fix128::PI.double().double();
        cloth.set_rest_tether(omega * omega, Fix128::from_ratio(84, 100) * omega);
        cloth.positions[0] = rest + Vec3Fix::from_int(1, 0, 0);
        let mut min = f64::MAX;
        for _ in 0..120 {
            cloth.step(Fix128::from_ratio(1, 60));
            min = min.min((cloth.positions[0] - rest).x.to_f64());
        }
        assert!(
            (min - want).abs() < 1e-9,
            "substeps {substeps}: overshoot {min} vs {want}"
        );
    }
}

/// ζ = 1 and ζ = 3 never cross the rest position from rest (and come back monotonically).
#[test]
fn critical_and_overdamped_do_not_overshoot() {
    for (zn, zd) in [(1_i64, 1_i64), (3, 1)] {
        for substeps in [1_usize, 4] {
            let mut cloth = row(1, 1);
            cloth.config.substeps = substeps;
            let rest = cloth.positions[0];
            let (k, c) = kc(1, 10, zn, zd);
            cloth.set_rest_tether(k, c);
            cloth.positions[0] = rest + Vec3Fix::from_int(1, 0, 0);
            let mut prev = 1.0_f64;
            for n in 0..240 {
                cloth.step(Fix128::from_ratio(1, 60));
                let x = (cloth.positions[0] - rest).x.to_f64();
                assert!(x >= 0.0, "ζ {zn}/{zd}, substeps {substeps}, frame {n}: {x}");
                assert!(x <= prev, "ζ {zn}/{zd}, frame {n}: {x} > {prev}");
                prev = x;
            }
        }
    }
}

/// The tether is solved once per substep outside the constraint iterations, so a lone
/// particle is bit-identical for every `iterations`.
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
                cloth.positions[0]
            })
            .collect::<Vec<_>>()
    };
    let one = run(1);
    for iterations in [2, 8, 16] {
        assert_eq!(one, run(iterations), "iterations {iterations}");
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
/// they would have alone, and each follows its own continuous solution.
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
        // ω = sqrt(40 / m), ζ = 3 / (2 sqrt(40 m))
        let mf = *m as f64;
        let w = (40.0 / mf).sqrt();
        let zeta = 3.0 / (2.0 * (40.0 * mf).sqrt());
        let e = comps(together.positions[i] - together.rest_positions()[i]);
        let o = comps(offsets[i]);
        for ax in 0..3 {
            let (ee, _) = exact(o[ax], 0.0, w, zeta, 3.0 / mf, 0.0, 2.0);
            assert!(
                (e[ax] - ee).abs() < 1e-9,
                "particle {i}, axis {ax}: {} vs {ee}",
                e[ax]
            );
        }
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

/// A sheet poked out of shape (with edges, bending and self-collision on) springs back to its
/// rest shape at one substep as at four; a pinned corner stays put; a replay is
/// bit-identical.
#[test]
fn a_poked_sheet_springs_back_and_replays_bit_identically() {
    for substeps in [1_usize, 4] {
        let run = || {
            let mut c = sheet();
            c.config.substeps = substeps;
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
        assert!(
            worst < 1e-4,
            "substeps {substeps}: sheet did not return to rest: {worst}"
        );
        assert_eq!(bits(&a), bits(&run()));
    }
}

// ---- degenerate input ----

/// Mismatched rest positions are rejected without change; zero (or negative, clamped)
/// stiffness and damping leave the cloth bit-identical to one without a tether; clearing
/// the tether restores the plain step.
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
}

/// Extreme magnitudes stay on the law (no numerical damping, no blow-up): an undamped
/// tether far stiffer than the substep (`ω h ≈ 4000`) keeps oscillating as `e0 cos ωt`; the
/// same tether critically damped lands on rest within a frame; an enormous damper with no
/// spring barely moves a particle from rest (`e ≈ e0`, velocity ≈ 0).
#[test]
fn extreme_tethers_stay_on_the_law() {
    let rest = [Vec3Fix::from_int(3, 3, 3), Vec3Fix::from_int(-3, 0, 1)];
    let offset = Vec3Fix::new(Fix128::ONE, Fix128::from_ratio(-1, 2), Fix128::ZERO);
    let start = |k: i64, c: i64| {
        let mut s = row(2, 1);
        s.set_rest_positions(&rest).unwrap();
        s.set_rest_tether(Fix128::from_int(k), Fix128::from_int(c));
        assert_eq!(s.rest_positions(), &rest);
        for (x, r) in s.positions.iter_mut().zip(&rest) {
            *x = *r + offset;
        }
        s
    };

    // ω = 1e6: e(t) = e0 cos ωt
    let mut stiff = start(1_000_000_000_000, 0);
    for n in 1..=3 {
        stiff.step(Fix128::from_ratio(1, 60));
        let (_, cs) = sin_cos64(1e6 * n as f64 / 60.0);
        for (i, (x, r)) in stiff.positions.iter().zip(&rest).enumerate() {
            let e = comps(*x - *r);
            assert!(
                (e[0] - cs).abs() < 1e-6,
                "frame {n}, particle {i}: {} vs {cs}",
                e[0]
            );
            assert!((e[1] + 0.5 * cs).abs() < 1e-6, "frame {n}, particle {i}");
        }
    }

    // ζ = 1 (c = 2 sqrt(k m)): (1 + ωt) e^(−ωt) ≈ 0 after one frame
    let mut critical = start(1_000_000_000_000, 2_000_000);
    critical.step(Fix128::from_ratio(1, 60));
    for (i, (x, r)) in critical.positions.iter().zip(&rest).enumerate() {
        assert!((*x - *r).length().to_f64() < 1e-12, "particle {i}");
    }

    // γ = 1e12, k = 0, from rest: nothing moves the particle
    let mut damper = start(0, 1_000_000_000_000);
    damper.step(Fix128::from_ratio(1, 60));
    for (i, (x, r)) in damper.positions.iter().zip(&rest).enumerate() {
        assert!(
            ((*x - *r) - offset).length().to_f64() < 1e-15,
            "particle {i}"
        );
        assert!(
            damper.velocities[i].length().to_f64() < 1e-15,
            "particle {i}"
        );
    }
}
