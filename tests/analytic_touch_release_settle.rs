//! A soft sheet held to its rest shape, dragged by a rigid hand and released, must come back to rest and be put
//! exactly there, within the time the rest tether's decay allows; at rest every velocity is exactly zero.
//!
//! The playback is the production entry (`touch_layers::replay_with`): one sheet of 4×4 particles, mass 1/100 each,
//! the rest tether `k = m ω²`, `c = 2 ζ m ω` with ω = 12.566 rad/s (a 0.5 s period) and ζ = 0.42, frame damping 1
//! (no extra damping), 6 iterations, 1 substep, rigid edges, no gravity, no self-collision, steps of 1/60 s.
//!
//! Oracle (closed form, outside the playback): each particle on its own tether is a damped oscillator
//! `m x'' = -k x - c x'`. Underdamped (ζ < 1), its distance from rest is bounded by the envelope
//! `|x(t)| ≤ A e^{-ζ ω t}` with `A = sqrt(x₀² + ((v₀ + ζ ω x₀) / ω_d)²)`, `ω_d = ω sqrt(1 - ζ²)`, and its speed
//! by `ω A e^{-ζ ω t} / sqrt(1 - ζ²)`. Both fall under the rest tolerance ε = 2⁻¹⁶ (m, m/s) once
//! `t ≥ ln(ω A / (ε (1 - ζ²)^{1/2})) / (ζ ω)`. The edges are conservative and are satisfied at rest, so they only
//! add stiffness to a deformation and do not slow the decay; the bound is taken over every particle from the state
//! measured at the step of release, and allowed 1.5 times for the discrete solver (one projection pass per step).
//! For the fast scene below (release at 0.8 em from rest, 8.24 em/s) it gives about 2.4 s, so with the allowance the
//! sheet must be at rest and inactive within about 3.6 s after release, inside the 5 s playback.
//!
//! Measured before the fold branch (0da4bc23, 4 iterations): every drag up to 50 per cent of the scene settled; at 75
//! per cent two of four durations did not; from 100 per cent no duration did, and the sheet stayed folded (part of
//! its triangles turned over) and circulated at a constant energy. The cause was the bend constraint's correction
//! rounding to zero at and next to a fold of a sheet that never leaves its plane; with the fold branch and 6
//! iterations every case of the sweep settles inside the bound and the energy never rises after release (with 4
//! iterations the edge projection is not converged: a hard drag still turns triangles over while held and the energy
//! rises in the step of release).
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::InputLog;
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::touch_layers::{replay_with, PlaybackConfig, TouchBodies};
use alice_physics_core::{Fix128, Vec3Fix};

const STEPS: u64 = 300;
const RES: usize = 4;
const RATE: f64 = 60.0;
const OMEGA: f64 = 12.566;
const ZETA: f64 = 0.42;
/// Allowance on the closed-form time for the discrete solver
const ALLOWANCE: f64 = 1.5;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn rest_tolerance() -> Fix128 {
    Fix128::from_ratio(1, 65_536)
}

fn centre() -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    )
}

fn sheet() -> Cloth {
    sheet_at(0)
}

/// Sheet `g` of a row: 1 em square, the next one 1.2 em further along x
fn sheet_at(g: i64) -> Cloth {
    let mass = Fix128::from_ratio(1, 100);
    let omega = Fix128::from_ratio(12_566, 1000);
    let origin = Vec3Fix::new(Fix128::from_ratio(6 * g, 5), Fix128::ZERO, Fix128::ZERO);
    let mut c = Cloth::new_grid(origin, Fix128::ONE, Fix128::ONE, RES, RES, mass);
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    c.config.iterations = 6;
    c.config.substeps = 1;
    c.set_rest_tether(
        mass * omega * omega,
        Fix128::from_ratio(84, 100) * mass * omega,
    );
    c
}

fn config() -> PlaybackConfig {
    config_with(rest_tolerance())
}

fn config_with(tolerance: Fix128) -> PlaybackConfig {
    PlaybackConfig::new(STEPS, dt(), 5, tolerance).expect("config")
}

fn at(dx: i64, dz: i64) -> Vec3Fix {
    centre()
        + Vec3Fix::new(
            Fix128::from_ratio(dx, 1000),
            Fix128::ZERO,
            Fix128::from_ratio(dz, 1000),
        )
}

/// Grab at (-0.4, -0.4) em from the centre at step 5, drag `amp` per cent of (0.70, 0.35) em over `drag` steps,
/// release on the next step. Returns the log and the release step.
fn drag_log(amp: i64, drag: u64) -> (InputLog<Touch>, u64) {
    let mut log = InputLog::new();
    log.record(5, Touch::grab(0, at(-400, -400))).expect("grab");
    let d = i64::try_from(drag).expect("small");
    for s in 6..=5 + drag {
        let k = i64::try_from(s - 5).expect("small");
        log.record(
            s,
            Touch::drag(at(
                -400 + 700 * amp * k / (100 * d),
                -400 + 350 * amp * k / (100 * d),
            )),
        )
        .expect("drag");
    }
    let release = 6 + drag;
    log.record(release, Touch::Release).expect("release");
    (log, release)
}

#[derive(Debug)]
struct Run {
    release: u64,
    /// closed-form steps after release (with the allowance) for the state measured at release
    bound: u64,
    /// first step at which the sheet is inactive and exactly at rest, if any
    settled_at: Option<u64>,
    /// tether + kinetic energy at release and at the end
    energy_release: f64,
    energy_end: f64,
    /// any step where every particle was exactly at its rest place but a velocity was not exactly zero
    rest_with_speed: Option<u64>,
    /// at the step the playback put the sheet at rest: the largest distance on any axis between where the step left a
    /// particle and its rest place (the jump of the snap), measured by stepping a copy of the previous state
    snap_jump: Option<f64>,
    /// steps (without a snap) where stepping a copy of the previous state did not give the state the playback reached
    /// (must be 0, or the snap measurement does not measure the playback)
    copy_mismatch: u64,
}

/// The largest distance on any axis between a particle and its rest place
fn max_offset(c: &Cloth) -> f64 {
    c.positions
        .iter()
        .zip(c.rest_positions())
        .map(|(p, r)| {
            let d = *p - *r;
            d.x.abs()
                .to_f64()
                .max(d.y.abs().to_f64())
                .max(d.z.abs().to_f64())
        })
        .fold(0.0, f64::max)
}

fn energy(c: &Cloth) -> f64 {
    let m = 0.01;
    let k = m * OMEGA * OMEGA;
    c.positions
        .iter()
        .zip(c.rest_positions())
        .zip(&c.velocities)
        .map(|((p, r), v)| {
            let d = *p - *r;
            0.5 * k * (d.x * d.x + d.y * d.y + d.z * d.z).to_f64()
                + 0.5 * m * (v.x * v.x + v.y * v.y + v.z * v.z).to_f64()
        })
        .sum()
}

/// The envelope amplitude `A` (max over particles and axes) of the state at release
fn envelope_amplitude(c: &Cloth) -> f64 {
    let sq = (1.0 - ZETA * ZETA).sqrt();
    let wd = OMEGA * sq;
    let mut a_max: f64 = 0.0;
    for ((p, r), v) in c
        .positions
        .iter()
        .zip(c.rest_positions())
        .zip(&c.velocities)
    {
        for (x, u) in [((*p - *r).x, v.x), ((*p - *r).y, v.y), ((*p - *r).z, v.z)] {
            let (x0, v0) = (x.to_f64(), u.to_f64());
            let a = (x0 * x0 + ((v0 + ZETA * OMEGA * x0) / wd).powi(2)).sqrt();
            a_max = a_max.max(a);
        }
    }
    a_max
}

/// The closed-form steps after release from the state at release (max over particles), with the allowance
fn closed_form_steps(c: &Cloth) -> u64 {
    let eps = 1.0 / 65_536.0;
    let sq = (1.0 - ZETA * ZETA).sqrt();
    let a_max = envelope_amplitude(c);
    if a_max == 0.0 {
        return 0;
    }
    let t = (OMEGA * a_max / (eps * sq)).ln() / (ZETA * OMEGA);
    (t * RATE * ALLOWANCE).ceil() as u64
}

fn run(amp: i64, drag: u64) -> Run {
    run_with(amp, drag, rest_tolerance())
}

fn run_with(amp: i64, drag: u64, tolerance: Fix128) -> Run {
    let (log, release) = drag_log(amp, drag);
    let mut before: Option<Cloth> = None;
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let mut out = Run {
        release,
        bound: 0,
        settled_at: None,
        energy_release: 0.0,
        energy_end: 0.0,
        rest_with_speed: None,
        snap_jump: None,
        copy_mismatch: 0,
    };
    replay_with(
        &[sheet()],
        &hand,
        &[log],
        config_with(tolerance),
        |step, bodies: &TouchBodies| {
            let c = bodies.cloth(0).expect("one sheet");
            if step == release {
                out.bound = closed_form_steps(c);
                out.energy_release = energy(c);
            }
            let at_rest = c
                .positions
                .iter()
                .zip(c.rest_positions())
                .all(|(p, r)| p == r);
            if at_rest
                && c.velocities.iter().any(|v| *v != Vec3Fix::ZERO)
                && out.rest_with_speed.is_none()
            {
                out.rest_with_speed = Some(step);
            }
            if step > release && out.settled_at.is_none() && bodies.active().is_empty() && at_rest {
                out.settled_at = Some(step);
                // the released sheet is stepped alone (no hand pulls it), so stepping a copy of the previous state
                // gives the state the snap replaced
                if let Some(mut b) = before.take() {
                    b.step(dt());
                    out.snap_jump = Some(max_offset(&b));
                }
            }
            if let Some(b) = before.as_ref() {
                if step > release && !bodies.active().is_empty() {
                    let mut b = b.clone();
                    b.step(dt());
                    if b.positions != c.positions || b.velocities != c.velocities {
                        out.copy_mismatch += 1;
                    }
                }
            }
            before = (!bodies.active().is_empty()).then(|| c.clone());
            out.energy_end = energy(c);
        },
    )
    .expect("replay");
    out
}

/// Settled inside the closed-form bound, and put at rest by a snap no wider than the rest tolerance
fn settles_in_bound(r: &Run) -> bool {
    r.settled_at.is_some_and(|s| s - r.release <= r.bound)
        && r.snap_jump.is_some_and(|j| j <= rest_tolerance().to_f64())
}

/// Control (green): a slow, small drag settles well inside the closed-form bound, so the bound is not vacuous
#[test]
fn a_slow_small_drag_settles_inside_the_closed_form_bound() {
    let r = run(10, 70);
    assert!(
        r.bound > 0 && r.release + r.bound < STEPS,
        "bound fits the playback: {r:?}"
    );
    assert!(
        settles_in_bound(&r),
        "slow drag must settle within the bound: {r:?}"
    );
    assert!(
        r.energy_end <= r.energy_release,
        "energy may not grow after release: {r:?}"
    );
}

/// Invariant: a sheet exactly at its rest shape has every velocity exactly zero (checked at every step of the slow
/// scene and of the fast scene, whether or not it settles)
#[test]
fn at_rest_every_velocity_is_exactly_zero() {
    for (amp, drag) in [(10, 70), (100, 35)] {
        let r = run(amp, drag);
        assert_eq!(
            r.rest_with_speed, None,
            "amp {amp} drag {drag}: at rest with a nonzero velocity: {r:?}"
        );
    }
}

/// The scene: 0.7 / 0.35 em in 35 steps (0.58 s), released at about 8 em/s
#[test]
fn a_fast_drag_settles_inside_the_closed_form_bound() {
    let r = run(100, 35);
    assert!(
        r.release + r.bound < STEPS,
        "bound fits the playback: {r:?}"
    );
    assert!(settles_in_bound(&r), "{r:?}");
}

/// The sweep across displacement and duration: every case settles inside its own bound (the defect shows where it
/// does not; the fix must hold across the boundary, not at one scene)
#[test]
fn every_drag_of_the_sweep_settles_inside_its_bound() {
    let mut failing = Vec::new();
    let mut checked = 0;
    for amp in [10, 25, 50, 75, 100, 150] {
        for drag in [10u64, 35, 70, 140] {
            let r = run(amp, drag);
            checked += 1;
            if !settles_in_bound(&r) || r.energy_end > r.energy_release {
                failing.push(format!(
                    "amp {amp} drag {drag}: release {} bound {} settled {:?} energy {:.4}->{:.4}",
                    r.release, r.bound, r.settled_at, r.energy_release, r.energy_end
                ));
            }
        }
    }
    assert_eq!(checked, 24);
    assert!(
        failing.is_empty(),
        "{} of {checked} do not settle:\n{}",
        failing.len(),
        failing.join("\n")
    );
}

/// The released sheet's energy must fall: at the end it is below its value at release by the tether's decay
#[test]
fn the_energy_of_a_released_sheet_decays() {
    let r = run(100, 35);
    // over the 3.9 s after release the envelope falls by e^{-2 ζ ω t} in energy
    let t = (STEPS - 1 - r.release) as f64 / RATE;
    let ceiling = r.energy_release * (-2.0 * ZETA * OMEGA * t / ALLOWANCE).exp();
    assert!(
        r.energy_end <= ceiling.max(1e-12),
        "energy {:.6} at the end, ceiling {:.3e}: {r:?}",
        r.energy_end,
        ceiling
    );
}

/// The snap measurement has teeth: a playback that snaps from a wider tolerance (2^-6) puts the slow scene at rest
/// from further away, and the measured jump says so (and the bound check refuses it)
#[test]
fn a_wider_snap_is_seen_by_the_jump_measurement() {
    let r = run_with(10, 70, Fix128::from_ratio(1, 64));
    let jump = r.snap_jump.expect("the wide snap puts the sheet at rest");
    assert!(jump > rest_tolerance().to_f64(), "jump {jump:e}: {r:?}");
    assert!(!settles_in_bound(&r), "{r:?}");
    let exact = run(10, 70);
    assert!(
        exact
            .snap_jump
            .is_some_and(|j| j <= rest_tolerance().to_f64()),
        "{exact:?}"
    );
    assert_eq!(
        exact.copy_mismatch, 0,
        "stepping a copy must reproduce the playback: {exact:?}"
    );
}

/// A row of 6 sheets touched one after another within 5 s: each grabbed at (-0.4, -0.4) em from its centre, dragged
/// `amp` per cent of (0.70, 0.35) em over `drag` steps and released on the next step; the next grab follows at once.
/// With 5 bodies allowed to move, the sixth grab puts the earliest released body that still moves exactly at rest.
#[derive(Debug)]
struct Row {
    /// the step of the sixth grab
    sixth: u64,
    /// bodies moving (and not held) just before the sixth grab
    moving_before_sixth: usize,
    /// every eviction: (body, steps since its release, the most its decay envelope allows at that time (with the
    /// allowance), the jump in em)
    evictions: Vec<(usize, u64, f64, f64)>,
}

fn row(amp: i64, drag: u64) -> Row {
    let mut log = InputLog::new();
    let span = drag + 2;
    let mut releases = Vec::new();
    let mut grabs = Vec::new();
    for g in 0..6u64 {
        let start = 5 + g * span;
        grabs.push(start);
        let gi = i64::try_from(g).expect("small");
        let base = |dx: i64, dz: i64| {
            at(dx, dz) + Vec3Fix::new(Fix128::from_ratio(6 * gi, 5), Fix128::ZERO, Fix128::ZERO)
        };
        log.record(
            start,
            Touch::grab(u32::try_from(g).expect("small"), base(-400, -400)),
        )
        .expect("grab");
        let d = i64::try_from(drag).expect("small");
        for s in start + 1..=start + drag {
            let k = i64::try_from(s - start).expect("small");
            log.record(
                s,
                Touch::drag(base(
                    -400 + 700 * amp * k / (100 * d),
                    -400 + 350 * amp * k / (100 * d),
                )),
            )
            .expect("drag");
        }
        releases.push(start + drag + 1);
        log.record(start + drag + 1, Touch::Release)
            .expect("release");
    }
    let sixth = 5 + 5 * span;
    let cloths: Vec<Cloth> = (0..6).map(sheet_at).collect();
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let mut out = Row {
        sixth,
        moving_before_sixth: 0,
        evictions: Vec::new(),
    };
    let mut prev: Option<(Vec<usize>, Vec<Cloth>)> = None;
    let mut amplitude = vec![0.0f64; 6];
    replay_with(
        &cloths,
        &hand,
        &[log],
        config(),
        |step, bodies: &TouchBodies| {
            for (g, &r) in releases.iter().enumerate() {
                if step == r {
                    amplitude[g] = envelope_amplitude(bodies.cloth(g).expect("sheet"));
                }
            }
            if let Some((active, states)) = prev.as_ref() {
                if step == sixth {
                    out.moving_before_sixth = active.len();
                }
                for &b in active {
                    // left the list without a step: the grab put it at rest from where it was
                    let now = bodies.cloth(b).expect("sheet");
                    let left = !bodies.active().contains(&b);
                    let jump = max_offset(&states[b]);
                    if grabs.contains(&step)
                        && left
                        && jump > rest_tolerance().to_f64()
                        && now.positions == now.rest_positions()
                    {
                        let since = step.saturating_sub(releases[b]);
                        let t = since as f64 / RATE;
                        let allowed = amplitude[b] * (-ZETA * OMEGA * t / ALLOWANCE).exp()
                            + rest_tolerance().to_f64();
                        out.evictions.push((b, since, allowed, jump));
                    }
                }
            }
            prev = Some((
                bodies.active().to_vec(),
                (0..6)
                    .map(|g| bodies.cloth(g).expect("sheet").clone())
                    .collect(),
            ));
        },
    )
    .expect("replay");
    out
}

/// Reported (run with --nocapture): bodies still moving at the sixth grab and the eviction jumps, for an ordinary,
/// a quick and a large drag per sheet
#[test]
fn a_row_of_six_reports_the_moving_bodies_and_eviction_jumps() {
    for (amp, drag) in [(10, 35), (50, 10), (100, 35)] {
        let r = row(amp, drag);
        println!("row amp {amp} drag {drag}: {r:?}");
        assert!(r.sixth < STEPS && r.sixth <= 300, "{r:?}");
    }
}

/// An eviction may only move a body by what its decay envelope still allows at that time (`A e^{-ζ ω t / 1.5}` plus the
/// rest tolerance): a body that decays as the tether says is evicted from close to rest; how many evictions a row
/// forces is a budget rule (reported above), how far they jump is physics
#[test]
fn every_eviction_moves_a_body_no_further_than_its_decay_envelope() {
    for (amp, drag) in [(10, 35), (50, 10), (100, 35)] {
        let r = row(amp, drag);
        let late: Vec<_> = r.evictions.iter().filter(|e| e.3 > e.2).collect();
        assert!(late.is_empty(), "amp {amp} drag {drag}: {late:?} in {r:?}");
    }
}
