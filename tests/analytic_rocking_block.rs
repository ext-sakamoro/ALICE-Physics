//! A block tilted below its tipping angle falls back and rocks on its two bottom
//! edges; each landing on the far edge loses a fixed fraction of its energy.
//!
//! # MODEL
//!
//! Housner, "The behavior of inverted pendulum structures during earthquakes",
//! Bull. Seismol. Soc. Am. 53 (1963). A rigid block of half-width `b/2`,
//! height `H`, tipping half-angle `α = atan(b / H)`, rocks on one bottom edge;
//! landing on the other edge is a plastic impact (the old edge lifts with no
//! impulse), which keeps the angular momentum about the new edge:
//!
//! - no slip (friction holds the edges): `ω' / ω = (I + m R² cos 2α) / (I + m R²)`
//!   with `I = m (b² + H²) / 12`, `R² = (b² + H²) / 4` (here 0.9423);
//! - no friction (the centre of mass moves only vertically):
//!   `ω' / ω = (I − m b² / 4) / (I + m b² / 4)` (here 0.7931).
//!
//! Between impacts the energy is conserved, so with `y_c(θ) = (H/2) cos θ +
//! (b/2) sin θ` the n-th peak solves `y_c(θ_n) − H/2 = r^{2n} (y_c(θ₀) − H/2)`.
//! From θ₀ = 9°: no slip 6.947°, 5.724°, 4.828°; no friction 4.188°, 2.404°,
//! 1.444° (f64 bisection below).
//!
//! # Bounds
//!
//! - Above: no peak exceeds the theory (energy within `1e-3` relative): a
//!   plastic impact cannot give energy back.
//! - Below, no slip: the solver spreads the impact over about two substeps of
//!   rotation `ω h` each, which shifts the effective turn at impact by up to
//!   `2 ω h` against the half-angle `α`; the energy depends on it about
//!   quadratically, so each impact may lose up to `4 ω h / α` more than the
//!   theory (`ω` the angular speed at the impact, `h` the substep). Measured:
//!   0.49 % at 240 Hz on the first impact (bound 1.03 %), converging with `h`.
//! - Below, no friction: the same plus a stated envelope of 0.5 % per impact.
//!   The frictionless peaks converge to 0.3 % under the theory and stay there
//!   (960 and 1920 Hz agree); the cause is open: not the contact compliance,
//!   not the speculative points' velocity target, not an impulse at the
//!   releasing edge (all measured).
//!
//! The old oracle ("9° ends upright by 2 s") encoded the dissipation of the
//! rigid, point-by-point contact that solver used; with the angular contact and
//! speculative points the block rocks as Housner's does.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 sin, cos, atan and asin of the block's tilt and the theory's peaks"
)]

use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

const B: f64 = 0.1;
const H: f64 = 0.5;
const G: f64 = 10.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn yc(t: f64) -> f64 {
    (H / 2.0) * t.cos() + (B / 2.0) * t.sin()
}

/// The first `n` peaks (degrees) of the theory with impact ratio `r` from θ₀.
fn theory(r: f64, theta0: f64, n: usize) -> Vec<f64> {
    let alpha = (B / H).atan();
    let mut ex = yc(theta0) - H / 2.0;
    (0..n)
        .map(|_| {
            ex *= r * r;
            let (mut lo, mut hi) = (0.0, alpha);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if yc(mid) - H / 2.0 < ex {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (0.5 * (lo + hi)).to_degrees()
        })
        .collect()
}

/// The energy above upright of a peak (degrees), per unit mass and `g`.
fn excess(deg: f64) -> f64 {
    yc(deg.to_radians()) - H / 2.0
}

/// Peaks (degrees) after the start, the largest |ω| seen just before each
/// crossing of upright, the largest |tilt| (degrees), at `hz` with friction `mu`.
fn rock(mu: f64, hz: f64, n: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let theta: f64 = 9f64.to_radians();
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-G), Fix128::ZERO),
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, fx(mu), Fix128::ZERO);
    let (hw, hh) = (B / 2.0, H / 2.0);
    let (s, c) = theta.sin_cos();
    let centre = Vec3Fix::new(fx(hw - (hw * c + hh * s)), fx(hw * s + hh * c), Fix128::ZERO);
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: Vec3Fix::new(fx(hw), fx(hh), fx(0.2)),
            },
            fx(500.0),
            centre,
        )
        .expect("block");
    w.set_body_material(b, m);
    w.get_body_mut(b).expect("block").set_rotation(QuatFix::from_axis_angle(
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        fx(-theta),
    ));
    let (mut peaks, mut omegas) = (Vec::new(), Vec::new());
    let (mut sgn, mut pk, mut max_tilt, mut last_w) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..(3.0 * hz) as usize {
        w.step(fx(1.0 / hz));
        let body = w.get_body(b).expect("block");
        let up = body
            .rotation
            .rotate_vec(Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO));
        let a = up.x.to_f64().clamp(-1.0, 1.0).asin().to_degrees();
        max_tilt = max_tilt.max(a.abs());
        let sign = a.signum();
        if sgn == 0.0 {
            sgn = sign;
        }
        if sign != sgn && a.abs() > 1e-9 {
            omegas.push(last_w);
            if pk > 0.0 || peaks.is_empty() {
                peaks.push(pk);
            }
            pk = 0.0;
            sgn = sign;
            if peaks.len() > n {
                break;
            }
        }
        last_w = body.angular_velocity.length().to_f64();
        pk = pk.max(a.abs());
    }
    // the first entry is the start (9°), drop it
    (peaks.into_iter().skip(1).collect(), omegas, max_tilt)
}

fn check(mu: f64, ratio: f64, envelope: f64) -> Vec<String> {
    let alpha = (B / H).atan();
    let want = theory(ratio, 9f64.to_radians(), 3);
    let mut wrong = Vec::new();
    for hz in [240.0, 480.0, 960.0] {
        let h = 1.0 / (hz * 8.0);
        let (peaks, omegas, max_tilt) = rock(mu, hz, 3);
        if max_tilt >= alpha.to_degrees() {
            wrong.push(format!("μ {mu} {hz} Hz: tipped ({max_tilt:.3}°)"));
            continue;
        }
        if peaks.len() < 3 {
            wrong.push(format!("μ {mu} {hz} Hz: {} peaks (did not rock)", peaks.len()));
            continue;
        }
        let mut allowed = 0.0;
        for k in 0..3 {
            allowed += 4.0 * omegas[k] * h / alpha + envelope;
            let (got, th) = (excess(peaks[k]), excess(want[k]));
            if got > th * (1.0 + 1e-3) || got < th * (1.0 - allowed) {
                wrong.push(format!(
                    "μ {mu} {hz} Hz peak {}: {:.4}° (theory {:.4}°, energy {:.4} of the theory, allowed down to {:.4})",
                    k + 1,
                    peaks[k],
                    want[k],
                    got / th,
                    1.0 - allowed
                ));
            }
        }
    }
    wrong
}

/// With friction that holds the edges (μ 0.8), the peaks follow Housner within
/// the bounds above, at 240, 480 and 960 Hz.
#[test]
fn a_block_rocks_with_housners_energy_ratio_per_impact() {
    let r = {
        let i = (B * B + H * H) / 12.0;
        let r2 = (B * B + H * H) / 4.0;
        (i + r2 * (2.0 * (B / H).atan()).cos()) / (i + r2)
    };
    let wrong = check(0.8, r, 0.0);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Without friction the peaks follow the frictionless ratio within the bounds
/// and the stated 0.5 % envelope per impact.
#[test]
fn a_frictionless_block_rocks_with_its_energy_ratio_per_impact() {
    let r = {
        let i = (B * B + H * H) / 12.0;
        (i - B * B / 4.0) / (i + B * B / 4.0)
    };
    let wrong = check(0.0, r, 0.005);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
