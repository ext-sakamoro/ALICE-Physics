//! A soft sheet held to its rest shape, dragged by a rigid hand and released, must come back to rest and be put
//! exactly there, within the time the rest tether's decay allows; at rest every velocity is exactly zero.
//!
//! The playback is the production entry (`touch_layers::replay_with`): one sheet of 4×4 particles, mass 1/100 each,
//! the rest tether `k = m ω²`, `c = 2 ζ m ω` with ω = 12.566 rad/s (a 0.5 s period) and ζ = 0.42, frame damping 1
//! (no extra damping), 4 iterations, 1 substep, rigid edges, no gravity, no self-collision, steps of 1/60 s.
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
//! Measured on 0da4bc23 (the sweep below; a two-sheet run of the same scene gives identical numbers on 0da4bc23 and
//! on 38cb908c): every drag up to 50 per cent of the scene settles; at 75 per cent two of four durations (10 and 140
//! steps) do not; from 100 per cent no duration does. A sheet that does not settle is left folded (part of its
//! triangles turned over) and circulating at a constant energy. The boundary follows the displacement, not the speed.
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
    let mass = Fix128::from_ratio(1, 100);
    let omega = Fix128::from_ratio(12_566, 1000);
    let mut c = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, RES, RES, mass);
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    c.config.iterations = 4;
    c.config.substeps = 1;
    c.set_rest_tether(
        mass * omega * omega,
        Fix128::from_ratio(84, 100) * mass * omega,
    );
    c
}

fn config() -> PlaybackConfig {
    PlaybackConfig::new(STEPS, dt(), 5, rest_tolerance()).expect("config")
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

/// The closed-form steps after release from the state at release (max over particles), with the allowance
fn closed_form_steps(c: &Cloth) -> u64 {
    let eps = 1.0 / 65_536.0;
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
    if a_max == 0.0 {
        return 0;
    }
    let t = (OMEGA * a_max / (eps * sq)).ln() / (ZETA * OMEGA);
    (t * RATE * ALLOWANCE).ceil() as u64
}

fn run(amp: i64, drag: u64) -> Run {
    let (log, release) = drag_log(amp, drag);
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let mut out = Run {
        release,
        bound: 0,
        settled_at: None,
        energy_release: 0.0,
        energy_end: 0.0,
        rest_with_speed: None,
    };
    replay_with(
        &[sheet()],
        &hand,
        &[log],
        config(),
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
            }
            out.energy_end = energy(c);
        },
    )
    .expect("replay");
    out
}

fn settles_in_bound(r: &Run) -> bool {
    r.settled_at.is_some_and(|s| s - r.release <= r.bound)
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
#[ignore = "known defect: touch-release-does-not-settle: after a fast drag the released sheet stays folded (5 of 18 triangles turned over, edge strain 0.215 with rigid edges) and circulates at a constant tether energy 1.34 (max deviation 0.4832, max speed 0.5129 from 1.6 s after release to the end) instead of settling; measured on 0da4bc23"]
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
#[ignore = "known defect: touch-release-does-not-settle: 10 of 24 do not settle (amp 75 at 10 and 140 steps, amp 100 and 150 at every duration); amp 10-50 at every duration and amp 75 at 35 and 70 steps settle; measured on 0da4bc23"]
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
#[ignore = "known defect: touch-release-does-not-settle: the energy stays at 1.34 (tether) from 1.6 s after release"]
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
