//! Oracles of the backward Euler step of the 2D material point method
//! ([`alice_physics_core::mpm2d::Mpm2d::step_implicit`]).
//!
//! 1. A free blob in rigid translation (no stress): the step's potential is minimised by the
//!    explicit grid velocity, so the centre of mass follows symplectic Euler's discrete
//!    parabola to 1e-12 m at 1/60 s steps.
//! 2. A column under its own weight at a step longer than its first period (1/60 s against
//!    0.038 s) settles to the plane strain bar value `ρ g H² (1 − ν²) / (2 E)` within 5 % (the
//!    step damps the mode it does not resolve, so the settlement is read at rest, not as a
//!    time average).
//! 3. Backward Euler does not create energy: kinetic + elastic + potential is non-increasing
//!    step to step, to the solve's tolerance (a step longer than the period and a short one).
//! 4. As the step shrinks the implicit run approaches the explicit one: the difference of the
//!    column's top after the same time halves with the step (first order).
//! 5. The viscoelastic column (one Maxwell branch) creeps like the standard linear solid at
//!    1/60 s steps: within 5 % of `δ∞ (1 − g e^(−t/τc))` at `t = τc / 2, τc, 2 τc, 3 τc`.
//! 6. A spinning free blob keeps its linear momentum within the budget the tolerance allows
//!    (each free node's residual force is at most `tol × m_i`, so `|ΔP| ≤ Δt⁻¹`… per step
//!    `Σ m_i tol`).
//! 7. Determinism: two runs give the same digest, and a snapshot taken mid-run replays bit
//!    for bit under implicit steps.
//! 8. A solve that cannot converge is refused and changes nothing.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{
    ImplicitSolve, Maxwell, Mpm2d, Mpm2dConfig, Mpm2dError, Mpm2dMaterial, Mpm2dModel,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.01;
const E: f64 = 1.0e5;
const NU: f64 = 0.3;
const RHO: f64 = 1000.0;
const G: f64 = 9.81;
const W: f64 = 0.04;
const H: f64 = 0.1;
const GROUND: f64 = 0.05;
const FRAME: f64 = 1.0 / 60.0;
const MODELS: [Mpm2dModel; 2] = [Mpm2dModel::FixedCorotated, Mpm2dModel::NeoHookean];

fn bar_speed() -> f64 {
    (E / ((1.0 - NU * NU) * RHO)).sqrt()
}

/// the explicit oracles' step: a third of the cell crossing time
fn explicit_dt() -> f64 {
    CELL / bar_speed() / 3.0
}

fn material(model: Mpm2dModel) -> Mpm2dMaterial {
    Mpm2dMaterial {
        youngs: fx(E),
        poisson: fx(NU),
        density: fx(RHO),
        model,
        maxwell: None,
    }
}

fn world(model: Mpm2dModel, gravity: f64, ground: Option<f64>, dt: f64) -> Mpm2d {
    Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes: [24, 40],
            gravity: [Fix128::ZERO, fx(-gravity)],
            dt: fx(dt),
            ground: ground.map(fx),
        },
        material(model),
    )
    .expect("a valid world")
}

/// 2 × 2 particles per cell over `[x0, x0 + w] × [y0, y0 + h]`, velocity `v(x, y)`
fn block(w: &mut Mpm2d, x0: f64, y0: f64, bw: f64, bh: f64, v: impl Fn(f64, f64) -> [f64; 2]) {
    let nx = (bw / CELL).round() as usize * 2;
    let ny = (bh / CELL).round() as usize * 2;
    let area = CELL * CELL / 4.0;
    for i in 0..nx {
        for j in 0..ny {
            let x = x0 + (i as f64 + 0.5) * CELL / 2.0;
            let y = y0 + (j as f64 + 0.5) * CELL / 2.0;
            let [vx, vy] = v(x, y);
            w.add_particle([fx(x), fx(y)], [fx(vx), fx(vy)], fx(area));
        }
    }
}

fn column(model: Mpm2dModel, dt: f64) -> (Mpm2d, Vec<usize>) {
    let mut w = world(model, G, Some(GROUND), dt);
    block(&mut w, 0.1, GROUND, W, H, |_, _| [0.0, 0.0]);
    let top_y = GROUND + H - CELL / 4.0;
    let top = (0..w.particles().len())
        .filter(|&i| (w.particles()[i].x[1].to_f64() - top_y).abs() < 1e-9)
        .collect();
    (w, top)
}

fn top_height(w: &Mpm2d, top: &[usize]) -> f64 {
    top.iter()
        .map(|&i| w.particles()[i].x[1].to_f64())
        .sum::<f64>()
        / top.len() as f64
}

fn total_energy(w: &Mpm2d) -> f64 {
    let potential: f64 = w
        .particles()
        .iter()
        .map(|p| p.mass.to_f64() * G * (p.x[1].to_f64() - GROUND))
        .sum();
    w.kinetic_energy().to_f64() + w.elastic_energy().to_f64() + potential
}

fn solve() -> ImplicitSolve {
    ImplicitSolve::default()
}

#[test]
fn a_free_blob_falls_on_the_discrete_parabola() {
    // no ground, gravity only, a rigid translation: the potential's minimum is the explicit
    // grid velocity, so the centre of mass follows `x_n = x_0 + n dt v_0 + g dt² n (n + 1) / 2`
    let dt = FRAME;
    let mut w = world(Mpm2dModel::FixedCorotated, G, None, dt);
    block(&mut w, 0.08, 0.3, 0.04, 0.04, |_, _| [0.3, 0.1]);
    let c0 = centre_of_mass(&w);
    for n in 1..=12 {
        let r = w.step_implicit(&solve()).expect("the step converges");
        assert!(r.newton <= 1, "step {n}: {} Newton iterations", r.newton);
        let c = centre_of_mass(&w);
        let k = n as f64;
        let want = [
            c0[0] + k * dt * 0.3,
            c0[1] + k * dt * 0.1 - G * dt * dt * k * (k + 1.0) / 2.0,
        ];
        for a in 0..2 {
            assert!(
                (c[a] - want[a]).abs() < 1e-12,
                "step {n} axis {a}: {:.15e} vs {:.15e}",
                c[a],
                want[a]
            );
        }
    }
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

#[test]
fn a_column_at_frame_steps_settles_by_the_plane_strain_bar_value() {
    let want = RHO * G * H * H * (1.0 - NU * NU) / (2.0 * E);
    for model in MODELS {
        let (mut w, top) = column(model, FRAME);
        let rest = top_height(&w, &top);
        let mut newton = 0;
        for _ in 0..60 {
            newton += w
                .step_implicit(&solve())
                .expect("the step converges")
                .newton;
        }
        let got = rest - top_height(&w, &top);
        println!("{model:?}: settlement {got:.4e} m (closed form {want:.4e}), {newton} Newton iterations in 60 steps");
        assert!(
            (got - want).abs() < 0.05 * want,
            "{model:?}: settlement {got:.4e} m vs {want:.4e} m"
        );
    }
}

#[test]
fn backward_euler_does_not_create_energy() {
    let tol = solve().tolerance.to_f64();
    for model in MODELS {
        for dt in [FRAME, explicit_dt()] {
            let (mut w, _) = column(model, dt);
            let steps = if dt == FRAME { 60 } else { 400 };
            let mut previous = total_energy(&w);
            let mass = w.mass().to_f64();
            for n in 0..steps {
                w.step_implicit(&solve()).expect("the step converges");
                let e = total_energy(&w);
                // the solve stops at a residual velocity of `tol` per node: its work is at
                // most of the order of M tol (|v| + tol), far below the step's dissipation
                let slack = mass * tol * (1.0 + tol);
                assert!(
                    e <= previous + slack,
                    "{model:?} dt {dt:.2e} step {n}: energy {e:.12e} above {previous:.12e}"
                );
                previous = e;
            }
        }
    }
}

#[test]
fn the_implicit_run_approaches_the_explicit_one_as_the_step_shrinks() {
    let t = 0.02;
    let mut diffs = Vec::new();
    for k in [1.0, 2.0, 4.0] {
        let dt = explicit_dt() / k;
        let steps = (t / dt).round() as usize;
        let (mut e, top) = column(Mpm2dModel::FixedCorotated, dt);
        let mut i = e.clone();
        for _ in 0..steps {
            e.step().expect("explicit");
            i.step_implicit(&solve()).expect("implicit");
        }
        diffs.push((top_height(&e, &top) - top_height(&i, &top)).abs());
    }
    println!("top height differences at dt, dt/2, dt/4: {diffs:?}");
    for w in diffs.windows(2) {
        let ratio = w[0] / w[1];
        assert!(
            (1.5..=2.8).contains(&ratio),
            "first order: the difference should halve with the step (ratios {diffs:?})"
        );
    }
}

#[test]
fn a_viscoelastic_column_at_frame_steps_creeps_like_the_standard_linear_solid() {
    let (g, relax) = (0.5, 0.1);
    let (base, top) = column(Mpm2dModel::FixedCorotated, FRAME);
    let mut m = *base.material();
    m.maxwell = Some(Maxwell {
        fraction: fx(g),
        relaxation_time: fx(relax),
    });
    let mut w = Mpm2d::new(*base.config(), m).expect("valid");
    for p in base.particles() {
        w.add_particle(p.x, p.v, p.area0);
    }
    let rest = top_height(&w, &top);
    let tc = relax / (1.0 - g);
    let inf = RHO * G * H * H * (1.0 - NU * NU) / (2.0 * E * (1.0 - g));
    let mut hist = Vec::new();
    let last = (3.0 * tc / FRAME).round() as usize + 1;
    for _ in 0..last {
        w.step_implicit(&solve()).expect("the step converges");
        hist.push(rest - top_height(&w, &top));
    }
    for k in [0.5f64, 1.0, 2.0, 3.0] {
        let at = (k * tc / FRAME).round() as usize - 1;
        let got = hist[at];
        let want = inf * (1.0 - g * (-k).exp());
        assert!(
            (got - want).abs() < 0.05 * inf,
            "t = {k} τc: settlement {got:.4e} m vs {want:.4e} m (δ∞ {inf:.4e})"
        );
    }
}

#[test]
fn a_spinning_free_blob_keeps_its_momentum_within_the_tolerance() {
    let mut w = world(Mpm2dModel::NeoHookean, 0.0, None, FRAME);
    let (cx, cy) = (0.12, 0.2);
    block(&mut w, cx - 0.02, cy - 0.02, 0.04, 0.04, |x, y| {
        [0.05 - 2.0 * (y - cy), 2.0 * (x - cx)]
    });
    let p0 = w.momentum();
    let tol = solve().tolerance.to_f64();
    let mass = w.mass().to_f64();
    for n in 1..=30 {
        w.step_implicit(&solve()).expect("the step converges");
        let p = w.momentum();
        let drift = ((p[0] - p0[0]).to_f64().powi(2) + (p[1] - p0[1]).to_f64().powi(2)).sqrt();
        // every free node's residual is at most tol × m_i, summed over the nodes and steps;
        // twice that for the rounding of the transfers
        let budget = 2.0 * n as f64 * mass * tol;
        assert!(
            drift <= budget,
            "step {n}: momentum drift {drift:.3e} above {budget:.3e}"
        );
    }
}

#[test]
fn implicit_runs_are_deterministic_and_replay_from_a_snapshot() {
    let (mut a, _) = column(Mpm2dModel::NeoHookean, FRAME);
    let (mut b, _) = column(Mpm2dModel::NeoHookean, FRAME);
    for _ in 0..10 {
        a.step_implicit(&solve()).expect("step");
        b.step_implicit(&solve()).expect("step");
    }
    assert_eq!(a.state_digest(), b.state_digest());
    let mut c = Mpm2d::restore(&a.snapshot()).expect("restores");
    for _ in 0..10 {
        a.step_implicit(&solve()).expect("step");
        c.step_implicit(&solve()).expect("step");
    }
    assert_eq!(a, c);
}

#[test]
fn a_solve_that_cannot_converge_is_refused_and_changes_nothing() {
    let (mut w, _) = column(Mpm2dModel::FixedCorotated, FRAME);
    w.step_implicit(&solve()).expect("a first step");
    let before = w.clone();
    let tight = ImplicitSolve {
        max_newton: 1,
        ..solve()
    };
    match w.step_implicit(&tight) {
        Err(Mpm2dError::NotConverged { newton }) => assert!(newton <= 1),
        other => panic!("expected NotConverged, got {other:?}"),
    }
    assert_eq!(w, before);
}

/// Measurement (not an oracle): the cost of one 1/60 s frame of a soft block the size of the
/// letter scene (0.827 m² ≈ 0.91 × 0.91 m, E 2e6 Pa, ρ 1000, ν 0.3, on the ground, under its
/// own weight, 2 × 2 particles per cell) at cells 0.02 and 0.04 m: explicit at a third of the
/// cell crossing time against implicit at 1 and 2 steps per frame (Jacobi preconditioner).
/// Prints the wall time per frame and the solve's iterations. Run with
/// `cargo test --release --test analytic_mpm2d_implicit -- --ignored --nocapture`.
#[test]
#[ignore = "measurement: prints the cost per frame of explicit and implicit steps"]
fn measure_the_cost_per_frame_of_a_letter_sized_block() {
    use std::time::Instant;
    let (e, rho, side): (f64, f64, f64) = (2.0e6, 1000.0, 0.91);
    for cell in [0.04f64, 0.02] {
        let c = (e / ((1.0 - NU * NU) * rho)).sqrt();
        let build = |dt: f64| {
            let nodes = ((side + 0.3) / cell).ceil() as usize + 4;
            let mut w = Mpm2d::new(
                Mpm2dConfig {
                    origin: [Fix128::ZERO, Fix128::ZERO],
                    cell: fx(cell),
                    nodes: [nodes, nodes],
                    gravity: [Fix128::ZERO, fx(-G)],
                    dt: fx(dt),
                    ground: Some(fx(0.1)),
                },
                Mpm2dMaterial {
                    youngs: fx(e),
                    poisson: fx(NU),
                    density: fx(rho),
                    model: Mpm2dModel::NeoHookean,
                    maxwell: None,
                },
            )
            .expect("valid");
            let n = (side / cell).round() as usize * 2;
            let area = cell * cell / 4.0;
            for i in 0..n {
                for j in 0..n {
                    let x = 0.15 + (i as f64 + 0.5) * cell / 2.0;
                    let y = 0.1 + (j as f64 + 0.5) * cell / 2.0;
                    w.add_particle([fx(x), fx(y)], [Fix128::ZERO; 2], fx(area));
                }
            }
            w
        };
        let dt_e = cell / c / 3.0;
        let per_frame = (FRAME / dt_e).ceil() as usize;
        let mut w = build(FRAME / per_frame as f64);
        let t = Instant::now();
        for _ in 0..per_frame {
            w.step().expect("explicit");
        }
        let explicit = t.elapsed().as_secs_f64();
        println!(
            "cell {cell}: {} particles, explicit {per_frame} steps/frame, {explicit:.3} s/frame",
            w.particles().len()
        );
        for steps in [1usize, 2] {
            let mut w = build(FRAME / steps as f64);
            for frame in 0..3 {
                let t = Instant::now();
                let (mut newton, mut cg) = (0, 0);
                for _ in 0..steps {
                    let r = w.step_implicit(&solve()).expect("implicit");
                    newton += r.newton;
                    cg += r.cg;
                }
                println!(
                    "  implicit {steps} step(s)/frame, frame {frame}: {:.3} s, {newton} Newton, {cg} CG",
                    t.elapsed().as_secs_f64()
                );
            }
        }
    }
}

/// Measurement (not an oracle): one letter "E" of the logo alone (strokes 0.085 m, 0.5 m
/// tall, arms 0.3 m long, area ≈ 0.097 m²), E 2e6 Pa, ρ 1000, ν 0.3, neo-Hookean, on the
/// ground, cell 0.04 m: the explicit step's cost per 1/60 s frame. Run with
/// `cargo test --release --test analytic_mpm2d_implicit -- --ignored --nocapture`.
#[test]
#[ignore = "measurement: prints the cost per frame of one letter"]
fn measure_the_cost_per_frame_of_one_letter() {
    use std::time::Instant;
    let (e, rho, cell): (f64, f64, f64) = (2.0e6, 1000.0, 0.04);
    let (stroke, tall, arm) = (0.085, 0.5, 0.3);
    let c = (e / ((1.0 - NU * NU) * rho)).sqrt();
    let per_frame = (FRAME / (cell / c / 3.0)).ceil() as usize;
    let nodes = ((tall + 0.4) / cell).ceil() as usize;
    let mut w = Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(cell),
            nodes: [nodes, nodes],
            gravity: [Fix128::ZERO, fx(-G)],
            dt: fx(FRAME / per_frame as f64),
            ground: Some(fx(0.1)),
        },
        Mpm2dMaterial {
            youngs: fx(e),
            poisson: fx(NU),
            density: fx(rho),
            model: Mpm2dModel::NeoHookean,
            maxwell: None,
        },
    )
    .expect("valid");
    let (x0, y0) = (0.2, 0.1);
    let inside = |x: f64, y: f64| {
        let (x, y) = (x - x0, y - y0);
        let bar = (0.0..stroke).contains(&x) && (0.0..tall).contains(&y);
        let arms = (0.0..arm).contains(&x)
            && [0.0, (tall - stroke) / 2.0, tall - stroke]
                .iter()
                .any(|&a| (a..a + stroke).contains(&y));
        bar || arms
    };
    let n = (((tall + 0.1) / cell).ceil() as usize) * 2;
    let h = cell / 2.0;
    for i in 0..n {
        for j in 0..n {
            let x = x0 + (i as f64 + 0.5) * h;
            let y = y0 + (j as f64 + 0.5) * h;
            if inside(x, y) {
                w.add_particle([fx(x), fx(y)], [Fix128::ZERO; 2], fx(h * h));
            }
        }
    }
    let area: f64 = w.particles().iter().map(|p| p.area0.to_f64()).sum();
    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        for _ in 0..per_frame {
            w.step().expect("explicit");
        }
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    println!(
        "letter E: {} particles, {area:.4} m², explicit {per_frame} steps/frame, median {:.4} s/frame (min {:.4}, max {:.4})",
        w.particles().len(),
        times[5],
        times[0],
        times[9]
    );
}
