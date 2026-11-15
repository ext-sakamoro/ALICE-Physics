//! Closed-form oracles of the 2D material point method ([`alice_physics_core::mpm2d`]).
//!
//! 1. Free fall: with no boundary in reach the grid adds `g dt` to every node's velocity and the
//!    transfers conserve momentum, so the centre of mass follows symplectic Euler's discrete
//!    parabola `x_n = x_0 + n dt v_0 + g dt² n (n + 1) / 2` to the rounding.
//! 2. A spinning free blob (no gravity): the transfers and the internal stress conserve linear
//!    momentum and the APIC angular momentum `Σ m (x × v) + m D (C₁₀ − C₀₁)`, `D = cell²/4`, in
//!    exact arithmetic; in `Q64.64` each step's operations round (the grid's division of
//!    momentum by mass floors), so the drift is held to a rounding budget: per step, per
//!    particle, per stencil node, 16 roundings of `2⁻⁶⁴` scaled by the momentum's size.
//! 3. A column under its own weight on a frictionless ground, plane strain, free sides: the
//!    static settlement of the top is `ρ g H² (1 − ν²) / (2 E)` (the 1D bar with the plane
//!    strain modulus `E / (1 − ν²)`); measured as the time average over whole periods.
//! 4. The same column's first axial mode: a quarter-wave bar fixed at the ground, period
//!    `4 H / c`, `c = sqrt(E / ((1 − ν²) ρ))`.
//! 5. Energy: kinetic + elastic + potential does not grow: the largest value in each period of
//!    the first mode is at most the previous period's, over 20 periods. Symplectic Euler's
//!    energy oscillates within a period (measured: above the start by 3.6e-5 of it in the
//!    first period only), and APIC dissipates (measured: about 3e-5 of the start per period),
//!    so the bound is on the period maxima, not on every step.
//! 6. Determinism: two runs from the same inputs end with the same state digest.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{Maxwell, Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel};

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
    material_of(Mpm2dModel::FixedCorotated)
}

fn material_of(model: Mpm2dModel) -> Mpm2dMaterial {
    Mpm2dMaterial {
        youngs: fx(E),
        poisson: fx(NU),
        density: fx(RHO),
        model,
        maxwell: None,
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
    world_of(Mpm2dModel::FixedCorotated, gravity, ground, nodes)
}

const MODELS: [Mpm2dModel; 2] = [Mpm2dModel::FixedCorotated, Mpm2dModel::NeoHookean];

fn world_of(model: Mpm2dModel, gravity: f64, ground: Option<f64>, nodes: [usize; 2]) -> Mpm2d {
    Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes,
            gravity: [Fix128::ZERO, fx(-gravity)],
            dt: fx(dt()),
            ground: ground.map(fx),
        },
        material_of(model),
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
    let ulp = 2f64.powi(-64);
    let particles = w.particles().len() as f64;
    let scale = w.mass().to_f64() + p0[0].to_f64().abs() + p0[1].to_f64().abs() + 1.0;
    // the lever arm is at most the grid's extent (0.64 m)
    let lever = 1.0 + 0.64;
    for n in 1..=300u32 {
        w.step().expect("the step succeeds");
        let budget = f64::from(n) * particles * 9.0 * 16.0 * ulp * scale;
        let p = w.momentum();
        for k in 0..2 {
            let d = (p[k] - p0[k]).to_f64().abs();
            assert!(
                d <= budget,
                "step {n}: momentum axis {k} drifted {d:.3e} > {budget:.3e}"
            );
        }
        let d = (w.angular_momentum() - l0).to_f64().abs();
        assert!(
            d <= budget * lever,
            "step {n}: angular momentum drifted {d:.3e} > {:.3e}",
            budget * lever
        );
    }
    // the budget is not loose: a missing affine term breaks the angular momentum by far more
    assert!(l0.to_f64().abs() > 1e3 * 300.0 * particles * 9.0 * 16.0 * ulp * scale * lever);
}

/// (the column world, the indices of its top row of particles)
fn column(model: Mpm2dModel) -> (Mpm2d, Vec<usize>) {
    let ground = 0.05;
    let mut w = world_of(model, G, Some(ground), [24, 40]);
    block(&mut w, 0.1, ground, W, H, |_, _| [0.0, 0.0]);
    let top_y = ground + H - CELL / 4.0;
    let top = (0..w.particles().len())
        .filter(|&i| (w.particles()[i].x[1].to_f64() - top_y).abs() < 1e-9)
        .collect();
    (w, top)
}

/// the top's height every step over `periods` periods of the first mode
fn column_history(model: Mpm2dModel, periods: f64) -> (Vec<f64>, f64) {
    let (mut w, top) = column(model);
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
    for model in MODELS {
        let (hist, period) = column_history(model, 10.0);
        let per = (period / dt()).round() as usize;
        // the time average over the last 8 whole periods
        let tail = &hist[hist.len() - 8 * per..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let want = RHO * G * H * H * (1.0 - NU * NU) / (2.0 * E);
        assert!(
            (mean - want).abs() < 0.05 * want,
            "{model:?}: settlement {mean:.4e} m vs closed form {want:.4e} m"
        );
    }
}

#[test]
fn a_column_oscillates_with_the_quarter_wave_period() {
    for model in MODELS {
        let (hist, period) = column_history(model, 10.0);
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
            "{model:?}: period {measured:.5} s vs 4H/c = {period:.5} s"
        );
    }
}

#[test]
fn the_energy_of_the_column_never_grows() {
    for model in MODELS {
        let (mut w, _) = column(model);
        let ground = 0.05;
        let potential = |w: &Mpm2d| {
            w.particles()
                .iter()
                .map(|p| p.mass.to_f64() * G * (p.x[1].to_f64() - ground))
                .sum::<f64>()
        };
        let total =
            |w: &Mpm2d| w.kinetic_energy().to_f64() + w.elastic_energy().to_f64() + potential(w);
        let per = (4.0 * H / bar_speed() / dt()).round() as usize;
        let mut previous = f64::INFINITY;
        for k in 0..20 {
            let mut top = f64::NEG_INFINITY;
            for _ in 0..per {
                w.step().expect("the step succeeds");
                top = top.max(total(&w));
            }
            assert!(
                top <= previous,
                "{model:?} period {k}: largest energy {top} above the previous period's {previous}"
            );
            previous = top;
        }
    }
}

#[test]
fn two_runs_end_in_the_same_state() {
    let run = || {
        let (mut w, _) = column(Mpm2dModel::NeoHookean);
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

#[test]
#[ignore = "a measurement, not a check"]
fn measure_the_energy_per_period() {
    let (mut w, _) = column(Mpm2dModel::FixedCorotated);
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
    let per = (4.0 * H / bar_speed() / dt()).round() as usize;
    let mut drift = Vec::new();
    for k in 0..40 {
        let (mut hi, mut lo) = (f64::MIN, f64::MAX);
        for _ in 0..per {
            w.step().expect("step");
            let e = total(&w) - e0;
            hi = hi.max(e);
            lo = lo.min(e);
        }
        drift.push((k, hi / e0, lo / e0));
    }
    for d in drift {
        println!(
            "period {:2}: max {:+.3e} min {:+.3e} (relative to E0)",
            d.0, d.1, d.2
        );
    }
}

/// A refused step changes nothing: a particle whose stencil leaves the grid makes the step
/// return the error with every particle as it was, and so does an inversion.
#[test]
fn a_refused_step_changes_nothing() {
    let mut w = world(0.0, None, [16, 16]);
    block(&mut w, 0.05, 0.05, 0.02, 0.02, |_, _| [0.0, 0.0]);
    // one particle flying out of the grid within a step
    w.add_particle(
        [fx(0.135), fx(0.08)],
        [fx(30.0), fx(0.0)],
        fx(CELL * CELL / 4.0),
    );
    let before = w.particles().to_vec();
    let mut refused = None;
    for n in 0..20 {
        match w.step() {
            Ok(()) => {}
            Err(e) => {
                refused = Some((n, e));
                break;
            }
        }
    }
    let (n, e) = refused.expect("the particle leaves the grid");
    assert!(
        matches!(e, alice_physics_core::mpm2d::Mpm2dError::OutOfGrid { .. }),
        "{e:?}"
    );
    let _ = n;
    let snapshot = w.particles().to_vec();
    assert!(w.step().is_err());
    assert_eq!(
        w.particles(),
        &snapshot[..],
        "the refused step left the particles as they were"
    );
    assert_ne!(before, snapshot);
    // an inversion: a particle squeezed through itself in one step
    let mut w = world(0.0, None, [16, 16]);
    block(&mut w, 0.05, 0.05, 0.02, 0.02, |x, _| {
        [if x < 0.06 { 400.0 } else { -400.0 }, 0.0]
    });
    let start = w.particles().to_vec();
    let got = w.step();
    assert!(
        matches!(
            got,
            Err(alice_physics_core::mpm2d::Mpm2dError::Inverted { .. })
        ),
        "{got:?}"
    );
    assert_eq!(w.particles(), &start[..]);
}

/// Seeding a disk of radius `r` from its signed distance puts particles only inside it, at
/// rest, and their rest areas sum to `π r²` within the lattice's boundary error (half the
/// perimeter times a sample spacing).
#[test]
fn a_seeded_disk_has_the_disk_area() {
    let mut w = world(0.0, None, [32, 32]);
    let (cx, cy, r) = (0.16, 0.15, 0.093);
    let n = w.seed(
        |p| {
            let dx = p[0].to_f64() - cx;
            let dy = p[1].to_f64() - cy;
            fx((dx * dx + dy * dy).sqrt() - r)
        },
        2,
    );
    assert_eq!(n, w.particles().len());
    let area: f64 = w.particles().iter().map(|p| p.area0.to_f64()).sum();
    let want = std::f64::consts::PI * r * r;
    let spacing = CELL / 2.0;
    assert!(
        (area - want).abs() < 2.0 * std::f64::consts::PI * r * spacing / 2.0,
        "seeded area {area:.6e} vs π r² {want:.6e}"
    );
    for p in w.particles() {
        let d = ((p.x[0].to_f64() - cx).powi(2) + (p.x[1].to_f64() - cy).powi(2)).sqrt();
        assert!(d < r, "a particle outside the disk at {d}");
        assert_eq!(p.v, [Fix128::ZERO; 2]);
        assert!((p.mass.to_f64() - RHO * p.area0.to_f64()).abs() < 1e-12);
    }
    assert_eq!(
        w.seed(|_| fx(-1.0), 0),
        0,
        "no samples per axis seeds nothing"
    );
}

/// A snapshot taken mid-run restores a world that steps on bit for bit as the original, for
/// both materials; damaged or foreign bytes are refused.
#[test]
fn a_restored_snapshot_replays_bit_for_bit() {
    for model in MODELS {
        let (mut a, _) = column(model);
        for _ in 0..50 {
            a.step().expect("step");
        }
        let blob = a.snapshot();
        let mut b = Mpm2d::restore(&blob).expect("restores");
        assert_eq!(a, b);
        for _ in 0..50 {
            a.step().expect("step");
            b.step().expect("step");
        }
        assert_eq!(a.state_digest(), b.state_digest(), "{model:?}");
        assert_eq!(a, b);
        let mut bad = blob.clone();
        bad[0] ^= 1;
        assert!(Mpm2d::restore(&bad).is_none(), "magic");
        let mut bad = blob.clone();
        bad[4] ^= 1;
        assert!(Mpm2d::restore(&bad).is_none(), "version");
        assert!(Mpm2d::restore(&blob[..blob.len() - 1]).is_none(), "short");
        let mut long = blob.clone();
        long.push(0);
        assert!(Mpm2d::restore(&long).is_none(), "long");
        let mut bad = blob.clone();
        let at = blob.len() - 10;
        bad[at] = 7;
        assert!(Mpm2d::restore(&bad).is_none(), "model tag");
    }
}

/// The viscoelastic column (one Maxwell branch, relaxation time `T`, fraction `g`) under its
/// own weight creeps like the standard linear solid under a held load: the settlement is
/// `δ∞ (1 − g e^(−t/τc))`, `τc = T / (1 − g)`, `δ∞ = ρ g H² (1 − ν²) / (2 E (1 − g))`
/// (the relaxed modulus `E (1 − g)`); measured as the average over one period of the first
/// mode around `t`, within 5 % of `δ∞` at `t = τc / 2, τc, 2 τc, 3 τc`.
#[test]
fn a_viscoelastic_column_creeps_like_the_standard_linear_solid() {
    let (g, relax) = (0.5, 0.1);
    let (mut w, top) = column(Mpm2dModel::FixedCorotated);
    let mut m = *w.material();
    m.maxwell = Some(Maxwell {
        fraction: fx(g),
        relaxation_time: fx(relax),
    });
    let mut v = Mpm2d::new(*w.config(), m).expect("valid");
    for p in w.particles() {
        v.add_particle(p.x, p.v, p.area0);
    }
    w = v;
    let rest = top_height(&w, &top);
    let tc = relax / (1.0 - g);
    let inf = RHO * G * H * H * (1.0 - NU * NU) / (2.0 * E * (1.0 - g));
    let per = (4.0 * H / bar_speed() / dt()).round() as usize;
    let mut hist = Vec::new();
    let last = (3.0 * tc / dt()).round() as usize + per;
    for _ in 0..last {
        w.step().expect("the step succeeds");
        hist.push(rest - top_height(&w, &top));
    }
    for k in [0.5f64, 1.0, 2.0, 3.0] {
        let at = (k * tc / dt()).round() as usize;
        let window = &hist[at - per / 2..at + per / 2];
        let got = window.iter().sum::<f64>() / window.len() as f64;
        let want = inf * (1.0 - g * (-k).exp());
        assert!(
            (got - want).abs() < 0.05 * inf,
            "t = {k} τc: settlement {got:.4e} m vs {want:.4e} m (δ∞ {inf:.4e})"
        );
    }
}

/// A Maxwell branch of fraction 0 changes nothing: the run is bit for bit the elastic one.
#[test]
fn a_zero_fraction_branch_is_the_elastic_material() {
    let (mut a, _) = column(Mpm2dModel::NeoHookean);
    let mut m = *a.material();
    m.maxwell = Some(Maxwell {
        fraction: Fix128::ZERO,
        relaxation_time: fx(0.3),
    });
    let mut b = Mpm2d::new(*a.config(), m).expect("valid");
    for p in a.particles() {
        b.add_particle(p.x, p.v, p.area0);
    }
    for _ in 0..300 {
        a.step().expect("step");
        b.step().expect("step");
    }
    for (p, q) in a.particles().iter().zip(b.particles()) {
        assert_eq!((p.x, p.v, p.c, p.f), (q.x, q.v, q.c, q.f));
    }
}
