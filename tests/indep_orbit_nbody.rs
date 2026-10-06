//! Second, independent set of oracles for `nbody`.
//!
//! `tests/analytic_nbody.rs` checks the circular equal-mass pair against the
//! leapfrog's discrete-circle closed form, the `e = 0.5` pair's energy bound,
//! and the figure-eight's return after one period. This file uses other
//! systems and other derivations:
//!
//! - an eccentric (`e = 0.3`) unequal-mass pair compared with the
//!   eccentric-anomaly closed form of the relative Kepler orbit
//!   (`μ = G(m₁ + m₂)`), at a fraction of a period, with the global error
//!   order measured from three step sizes;
//! - a massless test particle around a single mass (the zero-mass rule of the
//!   module), whose attractor must not move at all;
//! - time reversibility of the leapfrog (forward, negate velocities, forward);
//! - the figure-eight's choreography symmetry: after `T/3` the three bodies
//!   occupy each other's initial positions (a cyclic permutation), and its
//!   angular momentum is zero and stays zero;
//! - the acceleration as `−∇U/m` by a central finite difference of the
//!   module's potential energy, with Plummer softening.
//!
//! No expected value is produced by calling the code under test.

#![allow(clippy::disallowed_methods)]

use alice_physics::nbody::{kinetic_energy, total_momentum, DirectSum, NBodyError, VelocityVerlet};
use alice_physics::{Fix128, Vec3Fix};
use core::f64::consts::PI;

type V = [f64; 3];

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

fn v3(v: Vec3Fix) -> V {
    [f(v.x), f(v.y), f(v.z)]
}

fn fv(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}

fn dist(a: V, b: V) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn law(g: f64, eps: f64) -> DirectSum {
    DirectSum::new(fx(g), fx(eps)).unwrap()
}

fn kepler_bisect(m: f64, e: f64) -> f64 {
    let turns = (m / (2.0 * PI)).floor();
    let m0 = m - 2.0 * PI * turns;
    let (mut lo, mut hi) = (0.0_f64, 2.0 * PI);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid - e * mid.sin() - m0 < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi) + 2.0 * PI * turns
}

/// Relative orbit in the x–y plane at eccentric anomaly `E` (periapsis on
/// `+x`): `r = a(cos E − e, √(1−e²) sin E)`, `v = √(μa)/r (−sin E, √(1−e²) cos E)`.
fn relative_state(mu: f64, a: f64, e: f64, ecc: f64) -> (V, V) {
    let (s, c) = ecc.sin_cos();
    let b = (1.0 - e * e).sqrt();
    let r = a * (1.0 - e * c);
    let k = (mu * a).sqrt() / r;
    ([a * (c - e), a * b * s, 0.0], [-k * s, k * b * c, 0.0])
}

// ---------------------------------------------------------------------------
// Two bodies reduce to Kepler
// ---------------------------------------------------------------------------

/// Run an unequal-mass pair (`G = 0.5`, `m = (1, 3)`, `μ = 2`, `a = 1.2`,
/// `e = 0.3`, centre of mass at rest at the origin) from periapsis for
/// `0.37 T` in `steps` steps; return the error of the relative separation
/// against the Kepler closed form and the drift of the centre of mass.
fn eccentric_pair_error(steps: i64) -> (f64, f64) {
    let (g, m1, m2) = (0.5, 1.0, 3.0);
    let mu = g * (m1 + m2);
    let (a, e) = (1.2, 0.3);
    let (r, v) = relative_state(mu, a, e, 0.0);
    // body 2 − body 1 = relative; centre of mass at the origin, at rest
    let w1 = m2 / (m1 + m2);
    let w2 = m1 / (m1 + m2);
    let mut pos = vec![
        fv([-w1 * r[0], -w1 * r[1], 0.0]),
        fv([w2 * r[0], w2 * r[1], 0.0]),
    ];
    let mut vel = vec![
        fv([-w1 * v[0], -w1 * v[1], 0.0]),
        fv([w2 * v[0], w2 * v[1], 0.0]),
    ];
    let mass = vec![fx(m1), fx(m2)];
    let period = 2.0 * PI * (a * a * a / mu).sqrt();
    let t = 0.37 * period;
    let dt = fx(t) / Fix128::from_int(steps);
    let l = law(g, 0.0);
    let mut vv = VelocityVerlet::new();
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    let t_done = f(dt) * steps as f64;
    let n = (mu / (a * a * a)).sqrt();
    let want = relative_state(mu, a, e, kepler_bisect(n * t_done, e)).0;
    let got = v3(pos[1] - pos[0]);
    let com = [
        (m1 * f(pos[0].x) + m2 * f(pos[1].x)) / (m1 + m2),
        (m1 * f(pos[0].y) + m2 * f(pos[1].y)) / (m1 + m2),
        0.0,
    ];
    (dist(got, want), dist(com, [0.0; 3]))
}

/// oracle: the relative motion of two bodies is the Kepler orbit with
/// `μ = G(m₁ + m₂)`; the leapfrog's global error at a fixed time is second
/// order: `err(N)/err(2N)` and `err(2N)/err(4N)` in `[3.5, 4.5]`, and the
/// absolute error at `N = 800` is below `2e-4·a` (leading term
/// `≈ C·(ω dt)²`, `ω dt ≈ 0.003`). The centre of mass does not move
/// (`< 1e-15`).
#[test]
fn eccentric_pair_converges_to_kepler_at_second_order() {
    let (e1, c1) = eccentric_pair_error(800);
    let (e2, c2) = eccentric_pair_error(1600);
    let (e3, c3) = eccentric_pair_error(3200);
    assert!(e1 < 2e-4 * 1.2, "N=800 error {e1:e}");
    for (lo, hi) in [(e1, e2), (e2, e3)] {
        let ratio = lo / hi;
        assert!(
            (3.5..4.5).contains(&ratio),
            "error ratio {ratio} ({lo:e} / {hi:e})"
        );
    }
    for c in [c1, c2, c3] {
        assert!(c < 1e-15, "centre of mass drift {c:e}");
    }
}

/// oracle: a zero-mass test particle (allowed by the module) around a mass
/// `M = 2` (`G = 1.5`, `μ = 3`): the attractor feels `G·0·… = 0` and stays
/// bit-for-bit at rest at its start, and the particle follows the Kepler
/// orbit (`a = 0.9`, `e = 0.6`, one full period at 20 000 steps; measured
/// leapfrog error `2.1·10⁻⁷`, bound `2e-6·a`).
#[test]
fn test_particle_around_fixed_mass_follows_kepler() {
    let (g, big) = (1.5, 2.0);
    let mu = g * big;
    let (a, e) = (0.9, 0.6);
    let (r, v) = relative_state(mu, a, e, 1.0);
    let start = fv([0.25, -0.5, 0.125]);
    let mut pos = vec![start, start + fv(r)];
    let mut vel = vec![Vec3Fix::ZERO, fv(v)];
    let mass = vec![fx(big), Fix128::ZERO];
    let period = 2.0 * PI * (a * a * a / mu).sqrt();
    let steps = 20_000;
    let dt = fx(period) / Fix128::from_int(steps);
    let l = law(g, 0.0);
    let mut vv = VelocityVerlet::new();
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    assert_eq!(pos[0], start, "attractor moved");
    assert_eq!(vel[0], Vec3Fix::ZERO, "attractor accelerated");
    let t_done = f(dt) * steps as f64;
    let n = (mu / (a * a * a)).sqrt();
    let m0 = 1.0 - e * 1.0_f64.sin();
    let want = relative_state(mu, a, e, kepler_bisect(m0 + n * t_done, e)).0;
    let err = dist(v3(pos[1] - pos[0]), want);
    assert!(err < 2e-6 * a, "test particle error {err:e}");
}

// ---------------------------------------------------------------------------
// Time reversibility and momentum
// ---------------------------------------------------------------------------

/// oracle: the leapfrog is time reversible: integrate a 4-body system 3000
/// steps, negate every velocity, integrate 3000 more, and every body is back
/// at its start with negated initial velocity. In exact arithmetic the
/// return is exact; the bound `1e-11` leaves room for the fixed-point
/// rounding (`2⁻⁶⁴` per operation, amplified by the mild chaos of 4 bodies).
#[test]
fn leapfrog_is_time_reversible() {
    let p0 = vec![
        fv([1.0, 0.0, 0.1]),
        fv([-0.6, 0.9, 0.0]),
        fv([-0.4, -0.8, -0.2]),
        fv([0.1, 0.05, 1.3]),
    ];
    let v0 = vec![
        fv([0.0, 0.6, 0.0]),
        fv([-0.5, -0.2, 0.1]),
        fv([0.4, -0.3, 0.0]),
        fv([0.05, 0.0, -0.1]),
    ];
    let mass = vec![fx(1.0), fx(0.7), fx(1.3), fx(0.2)];
    let l = law(1.0, 0.05);
    let dt = fx(1e-3);
    let mut pos = p0.clone();
    let mut vel = v0.clone();
    let mut vv = VelocityVerlet::new();
    for _ in 0..3000 {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    let moved = dist(v3(pos[0]), v3(p0[0]));
    assert!(moved > 0.3, "bodies barely moved ({moved})");
    for v in &mut vel {
        *v = -*v;
    }
    for _ in 0..3000 {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    for k in 0..4 {
        let dp = dist(v3(pos[k]), v3(p0[k]));
        let dv = dist(v3(vel[k]), v3(-v0[k]));
        assert!(dp < 1e-11, "body {k}: Δx {dp:e}");
        assert!(dv < 1e-11, "body {k}: Δv {dv:e}");
    }
}

/// oracle: total momentum is conserved. With unequal masses the per-body
/// kicks round independently, so `Σ m v` drifts only at the `2⁻⁶⁴` level
/// (2000 steps, measured `9.6·10⁻¹⁶`, bound `1e-14`). With all masses `1`
/// the pair accelerations are `+p` and `−p` of the same value, but the kick
/// `a·dt/2` rounds toward `−∞` (floor), so `(−p)·h ≠ −(p·h)` by one unit and
/// the sum is not bit-exact either (measured `7.4·10⁻¹⁶`); same bound.
#[test]
fn total_momentum_is_conserved() {
    for masses in [[0.3, 2.1, 0.9, 1.7], [1.0, 1.0, 1.0, 1.0]] {
        let mut pos = vec![
            fv([0.0, 0.0, 0.0]),
            fv([1.1, 0.2, -0.3]),
            fv([-0.7, 0.8, 0.4]),
            fv([0.3, -1.2, 0.6]),
        ];
        let mut vel = vec![
            fv([0.2, 0.1, 0.0]),
            fv([-0.1, 0.4, 0.2]),
            fv([0.5, -0.3, -0.1]),
            fv([0.0, 0.2, 0.3]),
        ];
        let mass: Vec<Fix128> = masses.iter().map(|&m| fx(m)).collect();
        let p0 = v3(total_momentum(&vel, &mass).unwrap());
        let l = law(0.8, 0.0);
        let mut vv = VelocityVerlet::new();
        let mut worst = 0.0_f64;
        for _ in 0..2000 {
            vv.step(&l, &mut pos, &mut vel, &mass, fx(2e-3)).unwrap();
            worst = worst.max(dist(v3(total_momentum(&vel, &mass).unwrap()), p0));
        }
        assert!(worst < 1e-14, "masses {masses:?}: ΔP {worst:e}");
    }
}

// ---------------------------------------------------------------------------
// Figure-eight: choreography and zero angular momentum
// ---------------------------------------------------------------------------

/// oracle: the Chenciner–Montgomery figure-eight is a choreography — all
/// three bodies run along the same curve, `T/3` apart — so after `T/3` the
/// positions are a cyclic permutation (not the identity) of the initial
/// ones. Its total angular momentum is zero (Chenciner & Montgomery 2000,
/// Ann. Math. 152) and stays zero. Initial data Simó's, `T = 6.325 913 98`;
/// position tolerance `2e-6` (8-decimal data plus `O(dt²)`).
#[test]
fn figure_eight_is_a_choreography_with_zero_angular_momentum() {
    let (x1, y1): (f64, f64) = (0.970_004_36, -0.243_087_53);
    let (vx3, vy3): (f64, f64) = (-0.932_407_37, -0.864_731_46);
    let p0 = vec![fv([x1, y1, 0.0]), fv([-x1, -y1, 0.0]), Vec3Fix::ZERO];
    let mut pos = p0.clone();
    let mut vel = vec![
        fv([-vx3 / 2.0, -vy3 / 2.0, 0.0]),
        fv([-vx3 / 2.0, -vy3 / 2.0, 0.0]),
        fv([vx3, vy3, 0.0]),
    ];
    let mass = vec![Fix128::ONE; 3];
    let ang = |p: &[Vec3Fix], v: &[Vec3Fix]| -> f64 {
        p.iter()
            .zip(v)
            .map(|(p, v)| f(p.x) * f(v.y) - f(p.y) * f(v.x))
            .sum()
    };
    let l0 = ang(&pos, &vel);
    assert!(l0.abs() < 1e-8, "initial L = {l0:e}");
    let steps = 15_000;
    let dt = fx(6.325_913_98 / 3.0) / Fix128::from_int(steps);
    let l = law(1.0, 0.0);
    let mut vv = VelocityVerlet::new();
    let mut l_max = 0.0_f64;
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
        l_max = l_max.max((ang(&pos, &vel) - l0).abs());
    }
    assert!(l_max < 1e-13, "angular momentum drift {l_max:e}");
    // find the permutation: body k now sits where body perm[k] started
    let mut perm = [usize::MAX; 3];
    for k in 0..3 {
        for (j, start) in p0.iter().enumerate() {
            if dist(v3(pos[k]), v3(*start)) < 2e-6 {
                perm[k] = j;
            }
        }
        assert!(
            perm[k] != usize::MAX,
            "body {k} at {:?} is not at any initial position",
            v3(pos[k])
        );
    }
    let identity = perm == [0, 1, 2];
    let cyclic = perm == [1, 2, 0] || perm == [2, 0, 1];
    assert!(!identity && cyclic, "permutation {perm:?}");
}

/// oracle: the figure-eight's energy error over one period is second order:
/// `max|ΔE|` at `dt` and `dt/2` have ratio in `[3.5, 4.5]`; the initial
/// energy agrees with an `f64` evaluation of `Σ ½v² − Σ 1/r_ij`.
#[test]
fn figure_eight_energy_error_is_second_order() {
    let (x1, y1): (f64, f64) = (0.970_004_36, -0.243_087_53);
    let (vx3, vy3): (f64, f64) = (-0.932_407_37, -0.864_731_46);
    let p_init = [[x1, y1, 0.0], [-x1, -y1, 0.0], [0.0, 0.0, 0.0]];
    let v_init = [
        [-vx3 / 2.0, -vy3 / 2.0, 0.0],
        [-vx3 / 2.0, -vy3 / 2.0, 0.0],
        [vx3, vy3, 0.0],
    ];
    let mut e_ref = 0.0_f64;
    for k in 0..3 {
        e_ref += 0.5 * (v_init[k][0].powi(2) + v_init[k][1].powi(2));
        for j in (k + 1)..3 {
            e_ref -= 1.0 / dist(p_init[k], p_init[j]);
        }
    }
    let mass = vec![Fix128::ONE; 3];
    let l = law(1.0, 0.0);
    let energy = |p: &[Vec3Fix], v: &[Vec3Fix]| {
        f(kinetic_energy(v, &mass).unwrap()) + f(l.potential_energy(p, &mass).unwrap())
    };
    let run = |steps: i64| -> f64 {
        let mut pos: Vec<Vec3Fix> = p_init.iter().map(|&p| fv(p)).collect();
        let mut vel: Vec<Vec3Fix> = v_init.iter().map(|&v| fv(v)).collect();
        let e0 = energy(&pos, &vel);
        assert!((e0 - e_ref).abs() < 1e-14, "E₀ {e0} vs f64 {e_ref}");
        let dt = fx(6.325_913_98) / Fix128::from_int(steps);
        let mut vv = VelocityVerlet::new();
        let mut worst = 0.0_f64;
        for _ in 0..steps {
            vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
            worst = worst.max((energy(&pos, &vel) - e0).abs());
        }
        worst
    };
    let (a, b) = (run(1000), run(2000));
    let ratio = a / b;
    assert!((3.5..4.5).contains(&ratio), "ratio {ratio} ({a:e}, {b:e})");
}

// ---------------------------------------------------------------------------
// Force = −∇U / m
// ---------------------------------------------------------------------------

/// oracle: each acceleration equals `−(1/mᵢ) ∂U/∂xᵢ`, the gradient taken by
/// a central difference (`δ = 1e-6`) of the module's softened potential
/// energy (`ε = 0.3`, `G = 1.7`, 3 bodies). The difference has truncation
/// `O(δ²·U''') ≈ 1e-12` and rounding `2⁻⁶⁴/δ ≈ 1e-13`; tolerance `1e-9`
/// relative to the largest acceleration.
#[test]
fn acceleration_is_minus_gradient_of_potential_over_mass() {
    let l = law(1.7, 0.3);
    let pos = [[0.2, -0.1, 0.4], [-0.5, 0.6, 0.0], [0.35, 0.3, -0.45]];
    let mass = [fx(0.9), fx(2.3), fx(0.4)];
    let fpos: Vec<Vec3Fix> = pos.iter().map(|&p| fv(p)).collect();
    let mut acc = Vec::new();
    l.accelerations(&fpos, &mass, &mut acc).unwrap();
    let amax = acc
        .iter()
        .map(|a| dist(v3(*a), [0.0; 3]))
        .fold(0.0, f64::max);
    let delta = 1e-6;
    for i in 0..3 {
        for c in 0..3 {
            let mut plus = pos;
            let mut minus = pos;
            plus[i][c] += delta;
            minus[i][c] -= delta;
            let up: Vec<Vec3Fix> = plus.iter().map(|&p| fv(p)).collect();
            let um: Vec<Vec3Fix> = minus.iter().map(|&p| fv(p)).collect();
            let grad = (f(l.potential_energy(&up, &mass).unwrap())
                - f(l.potential_energy(&um, &mass).unwrap()))
                / (2.0 * delta);
            let want = -grad / f(mass[i]);
            let got = v3(acc[i])[c];
            assert!(
                (got - want).abs() < 1e-9 * amax,
                "body {i} axis {c}: {got} vs −∇U/m {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

/// `dt = 0` leaves positions and velocities bit for bit unchanged;
/// `dt < 0` integrates backward (undoes a forward step to rounding);
/// a negative mass is rejected before anything is written.
#[test]
fn zero_and_negative_time_steps() {
    let l = law(1.0, 0.0);
    let mass = vec![fx(1.0), fx(0.5)];
    let p0 = vec![fv([0.0, 0.0, 0.0]), fv([1.0, 0.0, 0.0])];
    let v0 = vec![fv([0.0, -0.3, 0.0]), fv([0.0, 0.6, 0.0])];
    let (mut pos, mut vel) = (p0.clone(), v0.clone());
    let mut vv = VelocityVerlet::new();
    vv.step(&l, &mut pos, &mut vel, &mass, Fix128::ZERO)
        .unwrap();
    assert_eq!(pos, p0);
    assert_eq!(vel, v0);
    vv.step(&l, &mut pos, &mut vel, &mass, fx(0.01)).unwrap();
    vv.step(&l, &mut pos, &mut vel, &mass, fx(-0.01)).unwrap();
    for k in 0..2 {
        assert!(dist(v3(pos[k]), v3(p0[k])) < 1e-17, "body {k} x");
        assert!(dist(v3(vel[k]), v3(v0[k])) < 1e-17, "body {k} v");
    }
    let mut pos2 = p0.clone();
    let mut vel2 = v0.clone();
    assert_eq!(
        vv.step(&l, &mut pos2, &mut vel2, &[fx(1.0), fx(-0.5)], fx(0.01)),
        Err(NBodyError::NegativeMass)
    );
    assert_eq!(pos2, p0);
    assert_eq!(vel2, v0);
}
