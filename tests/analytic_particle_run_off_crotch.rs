//! Oracles for `ParticleSystem::step_with_run_off` where two faces meet in a concave crotch (a valley, the inner
//! corner of a letter, a gutter). A drop that reaches the crotch is held by both faces: it may move only along the
//! line where they meet.
//!
//! - in a V valley the drop's mechanical energy `|v|²/2 + g y` never rises once it is on the surface (it falls by the
//!   drag), for three slopes
//! - a drop in a level crotch ends at rest in the crease and stays, within a bound from the drag and the impact loss
//!   per crossing; at 45° and 60° at once, at 30° after climbing the far face
//! - in a gutter whose crotch line falls at `s` (rise over run along it) the drop runs along the line at the closed
//!   form `g s / (√(1 + s²) drag)` and stays on it
//! - a drop on a single slope (no crotch) steps to the same bits as before the crotch was handled (pinned digest)
//!
//! The scenes use the world's rain parameters (drag 6 /s, drip cosine 1/2, reach 0.02 m, step 1/60 s): before the
//! crotch was handled the drop in a valley gained up to 0.048 J/kg a step, rising and falling across the crotch.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the closed forms are evaluated in f64 by the test, not by the engine"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::particle::{LandingTarget, Particle, ParticleSystem, RunOff};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::ClosureSdf;

const G: f64 = 9.81;
const DRAG: f64 = 6.0;
const STEPS: usize = 600;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn reach() -> Fix128 {
    Fix128::from_ratio(2, 100)
}

fn run_off() -> RunOff {
    RunOff {
        drag: Fix128::from_int(6),
        drip_cos: Fix128::from_ratio(1, 2),
    }
}

/// One drop falling at 6.5 m/s from `(x, y, z)`, no air damping
fn one_drop(x: f32, y: f32, z: f32) -> ParticleSystem {
    let mut s = ParticleSystem::new(1, Vec3Fix::from_f32(0.0, -G as f32, 0.0));
    s.damping = Fix128::ONE;
    s.particles.push(Particle::new(
        Vec3Fix::from_f32(x, y, z),
        Vec3Fix::from_f32(0.0, -6.5, 0.0),
        Fix128::from_int(1000),
        Fix128::ONE,
    ));
    s
}

/// Solid below a V of slope `k` whose crotch line falls by `s` per unit of z: `y < k |x| - s z`
fn valley(k: f32, s: f32) -> ClosureSdf {
    let n = (1.0 + k * k + s * s).sqrt();
    ClosureSdf::new(
        move |x, y, z| (y - k * x.abs() + s * z) / n,
        move |x, _, _| {
            let sx = if x >= 0.0 { 1.0 } else { -1.0 };
            (-k * sx / n, 1.0 / n, s / n)
        },
    )
}

/// Solid below one plane of slope `k`
fn slope(k: f32) -> ClosureSdf {
    let n = (1.0 + k * k).sqrt();
    ClosureSdf::new(
        move |x, y, _| (y - k * x) / n,
        move |_, _, _| (-k / n, 1.0 / n, 0.0),
    )
}

struct Track {
    on: bool,
    x: f64,
    y: f64,
    z: f64,
    v: [f64; 3],
}

fn track(sys: &mut ParticleSystem, field: &ClosureSdf) -> Vec<Track> {
    let mut rng = DeterministicRng::new(1);
    let targets = [LandingTarget::Field(field)];
    let mut out = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        sys.step_with_run_off(dt(), &mut rng, &targets, reach(), run_off());
        let p = &sys.particles[0];
        out.push(Track {
            on: sys.particle_on_surface(0).is_some(),
            x: p.position.x.to_f64(),
            y: p.position.y.to_f64(),
            z: p.position.z.to_f64(),
            v: [
                p.velocity.x.to_f64(),
                p.velocity.y.to_f64(),
                p.velocity.z.to_f64(),
            ],
        });
    }
    out
}

fn energy(t: &Track) -> f64 {
    (t.v[0] * t.v[0] + t.v[1] * t.v[1] + t.v[2] * t.v[2]) / 2.0 + G * t.y
}

fn speed(t: &Track) -> f64 {
    (t.v[0] * t.v[0] + t.v[1] * t.v[1] + t.v[2] * t.v[2]).sqrt()
}

/// oracle: in a V valley the drop's energy per unit mass never rises between two steps on the surface. The bound is
/// the field's f32: a position read through the field is off by at most a few f32 ulps of the 0.1 m scene, ~1e-8 m,
/// so `g` times that, 1e-7 J/kg; the defect this pins rose by up to 4.8e-2 J/kg a step
#[test]
fn in_a_valley_the_drop_never_gains_energy() {
    const BOUND: f64 = 1e-7;
    for k in [0.58_f32, 1.0, 1.73] {
        let mut sys = one_drop(0.03, 0.2, 0.0);
        let path = track(&mut sys, &valley(k, 0.0));
        let mut compared = 0;
        for w in path.windows(2) {
            if w[0].on && w[1].on {
                compared += 1;
                let rise = energy(&w[1]) - energy(&w[0]);
                assert!(
                    rise <= BOUND,
                    "slope {k}: energy rose by {rise:.3e} J/kg at ({:.4}, {:.4})",
                    w[1].x,
                    w[1].y
                );
            }
        }
        assert!(
            compared > 100,
            "slope {k}: only {compared} steps on the surface"
        );
    }
}

/// oracle: a drop in a level crotch (a V valley along z, face angle θ) ends at rest in the crease and stays, within
/// the bound of the MODEL: after it first reaches the crease at speed `v₀`, an impact leaves at most `r = cos 2θ` of
/// the speed up the far face, an excursion at speed `v` lasts at most `2 v / a + 1 / c` (`a = g sin θ`), and it stops
/// climbing after `N = ⌈ln(a dt / v₀) / ln r⌉` crossings, so it is at rest within `Σ_{n<N} (2 r^n v₀ / a + 1 / c) +
/// 2 dt`. At 45° and 60° (`r ≤ 0`) it rests at once (within 2 steps of arriving); at 30° it climbs the far face first
/// (more than a millimetre from the crease), as an inelastic impact leaves it moving away from the first face. The
/// drop starts at rest on one face, so its first crossing is the slow arrival the crease rule decides: holding it at
/// every crossing (always on the line) fails the 30° climb, never holding it fails the 60° rest
#[test]
fn a_drop_comes_to_rest_in_a_level_crotch_within_the_bound() {
    let dt = 1.0 / 60.0;
    for k in [0.58_f32, 1.0, 1.73] {
        // released at rest a millimetre above one face, 0.1 m from the crease: it lands at a crawl, slides down and
        // first reaches the crease at the slide's own speed (a drop falling onto the crease from height crosses it on
        // the landing's momentum and climbs the far face whatever the crease rule does)
        let mut sys = one_drop(0.1, 0.1 * k + 0.001, 0.0);
        sys.particles[0].velocity = Vec3Fix::from_f32(0.0, 0.0, 0.0);
        let path = track(&mut sys, &valley(k, 0.0));
        let theta = f64::from(k).atan();
        let a = G * theta.sin();
        // cos 2θ from the slope itself, (1 − k²)/(1 + k²): exactly 0 at 45° (the cosine of 2 atan 1 is 6e-17)
        let kd = f64::from(k);
        let r = (1.0 - kd * kd) / (1.0 + kd * kd);
        // it first reaches the crease where, on the surface, it is on it or has crossed it within the step
        let arrive = (1..STEPS)
            .find(|&i| {
                path[i].on
                    && (path[i].x.abs() <= 1e-6
                        || (path[i - 1].on && path[i].x * path[i - 1].x < 0.0))
            })
            .unwrap_or_else(|| panic!("slope {k}: the drop never reached the crotch"));
        let v0 = speed(&path[arrive - 1]).max(speed(&path[arrive]));
        // at rest the drop is put back on the crease from either face each step: the position may move by the
        // projection's tolerance, `max_travel / 2^24`, from each side
        let still = 2.0 * 0.02 / f64::from(1u32 << 24);
        let bound = if r <= 0.0 {
            2.0 * dt
        } else {
            let n = ((a * dt / v0).ln() / r.ln()).ceil().max(0.0) as i32;
            (0..n)
                .map(|i| 2.0 * r.powi(i) * v0 / a + 1.0 / DRAG)
                .sum::<f64>()
                + 2.0 * dt
        };
        // at rest from `rest` on: on the crease, not moving, not moved
        let rest = (arrive..STEPS)
            .find(|&i| {
                let at = &path[i];
                path[i..].iter().all(|t| {
                    t.on && t.x.abs() <= 1e-6
                        && speed(t) <= 1e-9
                        && (t.x - at.x).abs() + (t.y - at.y).abs() + (t.z - at.z).abs() <= still
                })
            })
            .unwrap_or_else(|| panic!("slope {k}: the drop never came to rest in the crotch"));
        let took = (rest - arrive) as f64 * dt;
        assert!(
            took <= bound,
            "slope {k}: at rest {took:.3} s after reaching the crotch, the bound is {bound:.3} s"
        );
        let climbed = path[arrive..rest]
            .iter()
            .map(|t| t.x.abs())
            .fold(0.0, f64::max);
        if r > 0.0 {
            assert!(
                climbed > 1e-3,
                "slope {k}: the drop did not climb the far face ({climbed:.2e} m)"
            );
        } else {
            assert!(
                rest - arrive <= 2,
                "slope {k}: at rest {} steps after arriving",
                rest - arrive
            );
        }
    }
}

/// oracle: in a gutter (a V valley of slope 1 whose crotch line falls by `s` per unit of z) the drop runs along the
/// crotch line at `g s / (√(1 + s²) drag)`, the incline's closed form for the line's own angle, and stays on it (x = 0,
/// y = -s z). Within the field's f32 (relative 1e-5 of the speed) after ten drag times
#[test]
fn in_a_gutter_the_drop_runs_along_the_crotch_at_the_closed_form() {
    for s in [0.2_f32, 0.5] {
        let mut sys = one_drop(0.03, 0.2, 0.0);
        let path = track(&mut sys, &valley(1.0, s));
        let sd = f64::from(s);
        let want = G * sd / ((1.0 + sd * sd).sqrt() * DRAG);
        let end = &path[STEPS - 1];
        assert!(end.on, "s {s}: the drop is not on the surface at the end");
        assert!(
            end.x.abs() <= 1e-6,
            "s {s}: the drop is {} off the crotch line",
            end.x
        );
        assert!(
            (end.y + sd * end.z).abs() <= 1e-6,
            "s {s}: the drop is off the crotch line (y + s z = {})",
            end.y + sd * end.z
        );
        assert!(
            (speed(end) - want).abs() <= 1e-5 * want,
            "s {s}: speed {:.6} m/s, the closed form gives {want:.6}",
            speed(end)
        );
    }
}

/// The digest of a drop run 600 steps down a single slope of 1 (no crotch), recorded before the crotch was handled
const SLOPE_DIGEST: &str = "6d081b9acbfc9e3146cc76bdc50b3d0399672e2f4e990fb36eefb224dd4fb4d3";

/// oracle: a drop on a single slope steps to the bits it stepped to before the crotch was handled: the crotch's path is
/// taken only where a move enters a second face
#[test]
fn on_a_single_slope_the_bits_are_unchanged() {
    let mut sys = one_drop(0.03, 0.2, 0.0);
    let field = slope(1.0);
    let mut rng = DeterministicRng::new(1);
    let targets = [LandingTarget::Field(&field)];
    for _ in 0..STEPS {
        sys.step_with_run_off(dt(), &mut rng, &targets, reach(), run_off());
    }
    let hex: String = sys
        .state_digest(&rng)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, SLOPE_DIGEST);
}
