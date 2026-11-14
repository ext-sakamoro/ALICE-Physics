//! Closed-form oracles of the 2D material point method ([`alice_physics_core::mpm2d`]).
//!
//! 1. Free fall: with no boundary in reach the grid adds `g dt` to every node's velocity and the
//!    transfers conserve momentum, so the centre of mass follows symplectic Euler's discrete
//!    parabola `x_n = x_0 + n dt v_0 + g dt² n (n + 1) / 2` to the rounding.
//! 2. A spinning free blob (no gravity): the transfers and the internal stress conserve linear
//!    momentum and the APIC angular momentum `Σ m (x × v) + m D (C₁₀ − C₀₁)`, `D = cell²/4`.
//! 3. A column under its own weight on a frictionless ground, plane strain, free sides: the
//!    static settlement of the top is `ρ g H² (1 − ν²) / (2 E)` (the 1D bar with the plane
//!    strain modulus `E / (1 − ν²)`); measured as the time average over whole periods.
//! 4. The same column's first axial mode: a quarter-wave bar fixed at the ground, period
//!    `4 H / c`, `c = sqrt(E / ((1 − ν²) ρ))`.
//! 5. Energy: kinetic + elastic + potential never grows above its start by more than the
//!    rounding over 20 periods (the scheme may dissipate, it may not create energy).
//! 6. Determinism: two runs from the same inputs end with the same state digest.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{Mpm2d, Mpm2dConfig, Mpm2dMaterial};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.01;
const E: f64 = 1.0e5;
const NU: f64 = 0.3;
const RHO: f64 = 1000.0;
const G: f64 = 9.81;
/// the column: 4 cells wide, 10 cells tall (0.04 m × 0.1 m)
const W: f64 = 0.04;
const H: f64 = 0.1;

fn material() -> Mpm2dMaterial {
    Mpm2dMaterial {
        youngs: fx(E),
        poisson: fx(NU),
        density: fx(RHO),
    }
}

/// the plane strain bar speed `sqrt(E / ((1 − ν²) ρ))`
fn bar_speed() -> f64 {
    (E / ((1.0 - NU * NU) * RHO)).sqrt()
}

/// a step a third of the cell crossing time of the bar wave
fn dt() -> f64 {
    CELL / bar_speed() / 3.0
}

/// 2 × 2 particles per cell over the rectangle `[x0, x0 + w] × [y0, y0 + h]`, at rest
fn block(world: &mut Mpm2d, x0: f64, y0: f64, w: f64, h: f64, v: impl Fn(f64, f64) -> [f64; 2]) {
    let nx = (w / CELL).round() as usize * 2;
    let ny = (h / CELL).round() as usize * 2;
    let area = CELL * CELL / 4.0;
    for i in 0..nx {
        for j in 0..ny {
            let x = x0 + (i as f64 + 0.5) * CELL / 2.0;
            let y = y0 + (j as f64 + 0.5) * CELL / 2.0;
            let [vx, vy] = v(x, y);
            world.add_particle([fx(x), fx(y)], [fx(vx), fx(vy)], fx(area));
        }
    }
}

fn world(gravity: f64, ground: Option<f64>, nodes: [usize; 2]) -> Mpm2d {
    Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes,
            gravity: [Fix128::ZERO, fx(-gravity)],
            dt: fx(dt()),
            ground: ground.map(fx),
        },
        material(),
    )
    .expect("a valid world")
}

fn centre_of_mass(w: &Mpm2d) -> [f64; 2] {
    let m = w.mass().to_f64();
    let mut c = [0.0; 2];
    for p in w.particles() {
        c[0] += p.mass.to_f64() * p.x[0].to_f64();
        c[1] += p.mass.to_f64() * p.x[1].to_f64();
    }
    [c[0] / m, c[1] / m]
}

/// the mean height of the particles within half a cell of the column's top at rest
fn top_height(w: &Mpm2d, top: &[usize]) -> f64 {
    top.iter()
        .map(|&i| w.particles()[i].x[1].to_f64())
        .sum::<f64>()
        / top.len() as f64
}

#[test]
fn a_free_blob_falls_on_the_discrete_parabola() {
    let mut w = world(G, None, [64, 64]);
    block(&mut w, 0.2, 0.4, W, W, |_, _| [0.3, 0.5]);
    let c0 = centre_of_mass(&w);
    let h = dt();
    for n in 1..=200u32 {
        w.step().expect("the step succeeds");
        let n = f64::from(n);
        let want = [
            c0[0] + n * h * 0.3,
            c0[1] + n * h * 0.5 - G * h * h * n * (n + 1.0) / 2.0,
        ];
        let got = centre_of_mass(&w);
        for k in 0..2 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "step {n}: axis {k} {} vs {}",
                got[k],
                want[k]
            );
        }
    }
}

#[test]
fn a_spinning_free_blob_keeps_its_momenta() {
    let mut w = world(0.0, None, [64, 64]);
    // a rigid spin of 3 rad/s about the blob's centre (0.32, 0.32), plus a drift
    let (cx, cy, omega) = (0.32, 0.32, 3.0);
    block(&mut w, 0.3, 0.3, W, W, |x, y| {
        [0.1 - omega * (y - cy), omega * (x - cx)]
    });
    let p0 = w.momentum();
    let l0 = w.angular_momentum();
    for n in 1..=300 {
        w.step().expect("the step succeeds");
        let p = w.momentum();
        assert!(
            (p[0] - p0[0]).abs().to_f64() < 1e-15 && (p[1] - p0[1]).abs().to_f64() < 1e-15,
            "step {n}: momentum"
        );
        assert!(
            (w.angular_momentum() - l0).abs().to_f64() < 1e-15,
            "step {n}: angular momentum {} vs {}",
            w.angular_momentum().to_f64(),
            l0.to_f64()
        );
    }
}

/// (the column world, the indices of its top row of particles)
fn column() -> (Mpm2d, Vec<usize>) {
    let ground = 0.05;
    let mut w = world(G, Some(ground), [24, 40]);
    block(&mut w, 0.1, ground, W, H, |_, _| [0.0, 0.0]);
    let top_y = ground + H - CELL / 4.0;
    let top = (0..w.particles().len())
        .filter(|&i| (w.particles()[i].x[1].to_f64() - top_y).abs() < 1e-9)
        .collect();
    (w, top)
}

/// the top's height every step over `periods` periods of the first mode
fn column_history(periods: f64) -> (Vec<f64>, f64) {
    let (mut w, top) = column();
    let rest = top_height(&w, &top);
    let period = 4.0 * H / bar_speed();
    let steps = (periods * period / dt()).round() as usize;
    let mut hist = Vec::with_capacity(steps);
    for _ in 0..steps {
        w.step().expect("the step succeeds");
        hist.push(rest - top_height(&w, &top));
    }
    (hist, period)
}

#[test]
fn a_column_settles_by_the_plane_strain_bar_value() {
    let (hist, period) = column_history(10.0);
    let per = (period / dt()).round() as usize;
    // the time average over the last 8 whole periods
    let tail = &hist[hist.len() - 8 * per..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let want = RHO * G * H * H * (1.0 - NU * NU) / (2.0 * E);
    assert!(
        (mean - want).abs() < 0.05 * want,
        "settlement {mean:.4e} m vs closed form {want:.4e} m"
    );
}

#[test]
fn a_column_oscillates_with_the_quarter_wave_period() {
    let (hist, period) = column_history(10.0);
    let mean = hist.iter().sum::<f64>() / hist.len() as f64;
    // upward crossings of the mean, after the first period
    let per = (period / dt()).round() as usize;
    let ups: Vec<usize> = (per + 1..hist.len())
        .filter(|&k| hist[k - 1] < mean && hist[k] >= mean)
        .collect();
    assert!(ups.len() >= 5, "{} crossings", ups.len());
    let measured = (ups[ups.len() - 1] - ups[0]) as f64 * dt() / (ups.len() - 1) as f64;
    assert!(
        (measured - period).abs() < 0.05 * period,
        "period {measured:.5} s vs 4H/c = {period:.5} s"
    );
}

#[test]
fn the_energy_of_the_column_never_grows() {
    let (mut w, _) = column();
    let ground = 0.05;
    let potential = |w: &Mpm2d| {
        w.particles()
            .iter()
            .map(|p| p.mass.to_f64() * G * (p.x[1].to_f64() - ground))
            .sum::<f64>()
    };
    let total =
        |w: &Mpm2d| w.kinetic_energy().to_f64() + w.elastic_energy().to_f64() + potential(w);
    let e0 = total(&w);
    let steps = (20.0 * 4.0 * H / bar_speed() / dt()).round() as usize;
    for n in 1..=steps {
        w.step().expect("the step succeeds");
        let e = total(&w);
        assert!(
            e <= e0 + 1e-12 * e0.abs(),
            "step {n}: energy {e} above the start {e0}"
        );
    }
}

#[test]
fn two_runs_end_in_the_same_state() {
    let run = || {
        let (mut w, _) = column();
        for _ in 0..200 {
            w.step().expect("the step succeeds");
        }
        w.state_digest()
    };
    assert_eq!(run(), run());
}

#[test]
fn invalid_worlds_are_refused() {
    let ok = Mpm2dConfig {
        origin: [Fix128::ZERO; 2],
        cell: fx(CELL),
        nodes: [8, 8],
        gravity: [Fix128::ZERO; 2],
        dt: fx(dt()),
        ground: None,
    };
    assert!(Mpm2d::new(ok, material()).is_ok());
    assert!(Mpm2d::new(
        Mpm2dConfig {
            cell: Fix128::ZERO,
            ..ok
        },
        material()
    )
    .is_err());
    assert!(Mpm2d::new(
        Mpm2dConfig {
            dt: Fix128::ZERO,
            ..ok
        },
        material()
    )
    .is_err());
    assert!(Mpm2d::new(
        Mpm2dConfig {
            nodes: [2, 8],
            ..ok
        },
        material()
    )
    .is_err());
    let m = material();
    assert!(Mpm2d::new(
        ok,
        Mpm2dMaterial {
            poisson: fx(0.5),
            ..m
        }
    )
    .is_err());
    assert!(Mpm2d::new(
        ok,
        Mpm2dMaterial {
            youngs: Fix128::ZERO,
            ..m
        }
    )
    .is_err());
    assert!(Mpm2d::new(
        ok,
        Mpm2dMaterial {
            density: fx(-1.0),
            ..m
        }
    )
    .is_err());
}
