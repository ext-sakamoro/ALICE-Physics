//! Measurements (not oracles) of the 2D material point method's cost per frame: the explicit
//! step against the backward Euler step on blocks the size of a letter and on the live letter
//! scene. Every test here is ignored; run them with
//! .
//! They print wall times and iteration counts and assert only that the steps succeed. The clock
//! reads each carry their own allow, so no allow here covers a whole function or the file.

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{ImplicitSolve, Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const NU: f64 = 0.3;
const G: f64 = 9.81;
const FRAME: f64 = 1.0 / 60.0;

/// the distance between two points, in f64 for reporting (not the code under test)
#[allow(
    clippy::disallowed_methods,
    reason = "measured quantity: a distance printed in f64, never compared against the engine"
)]
fn distance(p: [f64; 2], q: [f64; 2]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

fn solve() -> ImplicitSolve {
    ImplicitSolve::default()
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
        #[allow(
            clippy::disallowed_methods,
            reason = "timing harness: the wall-clock start of the measured frame, printed, never asserted"
        )]
        let t = Instant::now();
        for _ in 0..per_frame {
            w.step().expect("explicit");
        }
        #[allow(
            clippy::disallowed_methods,
            reason = "timing harness: the wall-clock cost of the measured frame, printed, never asserted"
        )]
        let explicit = t.elapsed().as_secs_f64();
        println!(
            "cell {cell}: {} particles, explicit {per_frame} steps/frame, {explicit:.3} s/frame",
            w.particles().len()
        );
        for steps in [1usize, 2] {
            let mut w = build(FRAME / steps as f64);
            for frame in 0..3 {
                #[allow(
                    clippy::disallowed_methods,
                    reason = "timing harness: the wall-clock start of the measured frame, printed, never asserted"
                )]
                let t = Instant::now();
                let (mut newton, mut cg) = (0, 0);
                for _ in 0..steps {
                    let r = w.step_implicit(&solve()).expect("implicit");
                    newton += r.newton;
                    cg += r.cg;
                }
                #[allow(
                    clippy::disallowed_methods,
                    reason = "timing harness: the wall-clock cost of the measured frame, printed, never asserted"
                )]
                let secs = t.elapsed().as_secs_f64();
                println!(
                    "  implicit {steps} step(s)/frame, frame {frame}: {secs:.3} s, {newton} Newton, {cg} CG"
                );
            }
        }
    }
}

/// Measurement (not an oracle): one letter "E" of the logo alone (strokes 0.085 m, 0.5 m
/// tall, arms 0.3 m long, area ≈ 0.097 m²), ρ 1000, ν 0.3, neo-Hookean, cell 0.04 m, the
/// explicit step's cost per 1/60 s frame, in two settings: E 2e6 Pa standing on the ground,
/// and E 2e5 Pa as the live step runs it: carried by an elastic backing (`foundation_loads`,
/// κ 2e5 N/m⁴, damping ratio 0.5, anchors set `ρ g / κ` above the rest positions so the backing
/// carries the weight) and pressed by a hand (`particles_within` 0.06 m of the top arm's
/// middle, `shared_loads` 20 N/m down), every load computed every step. Run with
/// `cargo test --release --test analytic_mpm2d_implicit -- --ignored --nocapture`.
#[test]
#[ignore = "measurement: prints the cost per frame of one letter"]
fn measure_the_cost_per_frame_of_one_letter() {
    use std::time::Instant;
    let (rho, cell, kappa): (f64, f64, f64) = (1000.0, 0.04, 2.0e5);
    let (stroke, tall, arm) = (0.085, 0.5, 0.3);
    for (e, backed) in [(2.0e6, false), (2.0e5, true)] {
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
                ground: (!backed).then(|| fx(0.1)),
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
        let lift = fx(rho * G / kappa);
        // damping ratio 0.5 of the backing, c = 2 ζ √(κ ρ); the hand at the top arm's middle
        let damping = 2.0 * 0.5 * (kappa * rho).sqrt();
        let hand = [fx(x0 + arm / 2.0), fx(y0 + tall - stroke / 2.0)];
        let anchors: Vec<[Fix128; 2]> = w
            .particles()
            .iter()
            .map(|p| [p.x[0], p.x[1] + lift])
            .collect();
        let area: f64 = w.particles().iter().map(|p| p.area0.to_f64()).sum();
        let mut times = Vec::new();
        for _ in 0..101 {
            #[allow(
                clippy::disallowed_methods,
                reason = "timing harness: the wall-clock start of the measured frame, printed, never asserted"
            )]
            let t = Instant::now();
            for _ in 0..per_frame {
                if backed {
                    // the live step: the damped backing and a hand pressing the top arm
                    let mut loads = w
                        .foundation_loads(&anchors, fx(kappa), fx(damping))
                        .expect("anchors");
                    let touched = w.particles_within(hand, fx(0.06));
                    if !touched.is_empty() {
                        loads.extend(
                            w.shared_loads(&touched, [Fix128::ZERO, fx(-20.0)])
                                .expect("a patch"),
                        );
                    }
                    w.step_with_loads(&loads).expect("explicit");
                } else {
                    w.step().expect("explicit");
                }
            }
            #[allow(
                clippy::disallowed_methods,
                reason = "timing harness: the wall-clock cost of the measured frame, printed, never asserted"
            )]
            let secs = t.elapsed().as_secs_f64();
            times.push(secs);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "letter E, E {e:.0e}{}: {} particles, {area:.4} m², explicit {per_frame} steps/frame, median {:.2} ms/frame (min {:.2}, max {:.2})",
            if backed { " on the backing" } else { " on the ground" },
            w.particles().len(),
            times[50] * 1e3,
            times[0] * 1e3,
            times[100] * 1e3
        );
    }
}

/// Measurement (not an oracle): the live letter (as `measure_the_cost_per_frame_of_one_letter`,
/// E 2e5 on the damped backing) pressed by the hand for 60 frames, then released for 60, by the
/// explicit step (a third of the cell crossing time) and by the implicit step at 1 and 2 steps
/// per frame. Prints each run's cost per frame, the press deflection at the hand, the largest
/// particle distance from the explicit run at the end of the press, and the hand point's
/// oscillation after the release (peak-to-peak over the first and the last 20 frames).
#[test]
#[ignore = "measurement: prints the implicit step's cost and its difference from explicit on the live letter"]
fn measure_the_implicit_step_on_the_live_letter() {
    use std::time::Instant;
    let (rho, cell, kappa, e): (f64, f64, f64, f64) = (1000.0, 0.04, 2.0e5, 2.0e5);
    let (stroke, tall, arm) = (0.085, 0.5, 0.3);
    let c = (e / ((1.0 - NU * NU) * rho)).sqrt();
    let explicit_steps = (FRAME / (cell / c / 3.0)).ceil() as usize;
    let (x0, y0) = (0.2, 0.1);
    let build = |steps: usize| {
        let nodes = ((tall + 0.4) / cell).ceil() as usize;
        let mut w = Mpm2d::new(
            Mpm2dConfig {
                origin: [Fix128::ZERO, Fix128::ZERO],
                cell: fx(cell),
                nodes: [nodes, nodes],
                gravity: [Fix128::ZERO, fx(-G)],
                dt: fx(FRAME / steps as f64),
                ground: None,
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
        w
    };
    let lift = fx(rho * G / kappa);
    let damping = 2.0 * 0.5 * (kappa * rho).sqrt();
    let hand = [fx(x0 + arm / 2.0), fx(y0 + tall - stroke / 2.0)];
    // (steps per frame, implicit?)
    let mut ends: Vec<Vec<[f64; 2]>> = Vec::new();
    for (steps, implicit) in [(explicit_steps, false), (1, true), (2, true)] {
        let mut w = build(steps);
        let anchors: Vec<[Fix128; 2]> = w
            .particles()
            .iter()
            .map(|p| [p.x[0], p.x[1] + lift])
            .collect();
        let touched = w.particles_within(hand, fx(0.06));
        let rest_y: f64 = touched
            .iter()
            .map(|&i| w.particles()[i].x[1].to_f64())
            .sum::<f64>()
            / touched.len() as f64;
        let hand_y = |w: &Mpm2d| {
            touched
                .iter()
                .map(|&i| w.particles()[i].x[1].to_f64())
                .sum::<f64>()
                / touched.len() as f64
        };
        let mut frame_ms = Vec::new();
        let mut after = Vec::new();
        let mut deflection = 0.0;
        for frame in 0..120 {
            let pressing = frame < 60;
            #[allow(
                clippy::disallowed_methods,
                reason = "timing harness: the wall-clock start of the measured frame, printed, never asserted"
            )]
            let t = Instant::now();
            for _ in 0..steps {
                let mut loads = w
                    .foundation_loads(&anchors, fx(kappa), fx(damping))
                    .expect("anchors");
                if pressing {
                    loads.extend(
                        w.shared_loads(&touched, [Fix128::ZERO, fx(-20.0)])
                            .expect("a patch"),
                    );
                }
                if implicit {
                    w.step_implicit_with_loads(&ImplicitSolve::default(), &loads)
                        .expect("implicit");
                } else {
                    w.step_with_loads(&loads).expect("explicit");
                }
            }
            #[allow(
                clippy::disallowed_methods,
                reason = "timing harness: the wall-clock cost of the measured frame, printed, never asserted"
            )]
            let secs = t.elapsed().as_secs_f64();
            frame_ms.push(secs * 1e3);
            if frame == 59 {
                deflection = rest_y - hand_y(&w);
                ends.push(
                    w.particles()
                        .iter()
                        .map(|p| [p.x[0].to_f64(), p.x[1].to_f64()])
                        .collect(),
                );
            }
            if !pressing {
                after.push(hand_y(&w));
            }
        }
        frame_ms.sort_by(f64::total_cmp);
        let p2p = |s: &[f64]| {
            s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - s.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let off = ends
            .last()
            .zip(ends.first())
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(p, q)| distance(*p, *q))
                    .fold(0.0, f64::max)
            })
            .unwrap_or(0.0);
        println!(
            "{} {steps} step(s)/frame: median {:.2} ms/frame; press deflection {:.3e} m; max distance from explicit {:.3e} m; release peak-to-peak first 20 frames {:.3e} m, last 20 {:.3e} m",
            if implicit { "implicit" } else { "explicit" },
            frame_ms[60],
            deflection,
            off,
            p2p(&after[..20]),
            p2p(&after[40..])
        );
    }
}
