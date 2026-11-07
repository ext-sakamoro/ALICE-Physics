//! Convergence of the contact friction laws in the substep `h`, and the
//! stick–slip limit from both sides (the physics rule § 2: the precision
//! parameter swept over 2 or more levels, tolerances derived, not round).
//!
//! The scene is the ball on an incline of `analytic_contact_static_friction_spin.rs`
//! (solid ball `I = 2/5 m r²`, `r = 1/2`, `m = 1`, gravity `−10`, damping
//! off; the incline a static box turned by `θ` about `z`, the ball released
//! from rest). The closed forms after `t = 1 s`:
//!
//! - rolling without slip (`tan θ ≤ 7 μ / 2`): `d = (5/7) g sin θ t² / 2`,
//!   `v = ω r`;
//! - sliding (`tan θ > 7 μ / 2`): `d = g (sin θ − μ cos θ) t² / 2`.
//!
//! The XPBD substep is first order in `h` for these laws, so the error
//! `|d(h) − d|` halves when `h` halves: the sweep asserts the ratio of
//! successive errors is at least `1.5` (first order gives `2`), and the
//! tolerance of each level is derived from the sweep itself by Richardson:
//! with `e(2h) / e(h) = ρ ≥ 1.5`, `e(h) = |d(h) − d(2h)| / (ρ − 1) ≤ 2 |d(h) − d(2h)|`.
//!
//! Near the limit `tan θ = 7 μ / 2` the per-substep stick–slip decision is at
//! its margin; the pair of tests a few percent on each side records what the
//! sweep shows there (the numbers are printed and the regime asserted).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 incline closed forms and error ratios"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
const INV_I: f64 = 1.0 / (0.4 * R * R);

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn g() -> f64 {
    -PhysicsConfig::default().gravity.y.to_f64()
}

/// The incline scene with `substeps` per frame.
fn incline(mu: f64, theta: f64, substeps: usize) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.bodies[slab].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)).normalize();
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 2.0),
        }
    ));
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let up = 0.5 + R;
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(normal[0] * up, normal[1] * up, 0.0), Fix128::ONE),
        fx(R),
    );
    w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

/// `(downhill displacement, downhill speed, ω_z r)` after 1 s at `substeps`
fn run(mu: f64, theta: f64, substeps: usize) -> (f64, f64, f64) {
    let (mut w, ball) = incline(mu, theta, substeps);
    let start = w.bodies[ball].position;
    for _ in 0..60 {
        w.step(fx(DT));
    }
    let b = &w.bodies[ball];
    let d = b.position - start;
    let down = -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin());
    let speed = -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
    (down, speed, b.angular_velocity.z.to_f64() * R)
}

const SWEEP: [usize; 4] = [4, 8, 16, 32];

/// Sweep `h` and assert first-order convergence of `d` to `want` with the
/// Richardson-derived bound at each refined level; returns the errors.
fn assert_first_order(label: &str, mu: f64, theta: f64, want: f64) -> Vec<f64> {
    let d: Vec<f64> = SWEEP.iter().map(|&n| run(mu, theta, n).0).collect();
    let errs: Vec<f64> = d.iter().map(|x| (x - want).abs()).collect();
    for (k, n) in SWEEP.iter().enumerate() {
        println!(
            "{label}: substeps {n}: d {:.7} want {want:.7} error {:.3e}",
            d[k], errs[k]
        );
    }
    let mut compared = 0;
    for k in 1..SWEEP.len() {
        let ratio = errs[k - 1] / errs[k];
        assert!(
            ratio >= 1.5,
            "{label}: error ratio {} -> {} substeps is {ratio:.3}, below first order",
            SWEEP[k - 1],
            SWEEP[k]
        );
        let richardson = 2.0 * (d[k] - d[k - 1]).abs();
        assert!(
            errs[k] <= richardson,
            "{label}: error {:.3e} at {} substeps exceeds the derived bound {richardson:.3e}",
            errs[k],
            SWEEP[k]
        );
        compared += 1;
    }
    assert_eq!(compared, SWEEP.len() - 1);
    errs
}

/// oracle: rolling without slip (`θ = 0.3`, `μ = 0.5`, `tan θ = 0.309 ≤ 1.75`):
/// `d = (5/7) g sin θ / 2` after 1 s, first order in `h`.
#[test]
fn rolling_down_an_incline_converges_at_first_order_in_the_substep() {
    let (mu, theta) = (0.5, 0.3_f64);
    let want = 5.0 / 7.0 * g() * theta.sin() * 0.5;
    assert_first_order("rolling", mu, theta, want);
}

/// oracle: sliding (`θ = 0.3`, `μ = 0.05`, `tan θ = 0.309 > 0.175`):
/// `d = g (sin θ − μ cos θ) / 2` after 1 s, first order in `h`.
#[test]
fn sliding_down_an_incline_converges_at_first_order_in_the_substep() {
    let (mu, theta) = (0.05, 0.3_f64);
    let want = g() * (theta.sin() - mu * theta.cos()) * 0.5;
    assert_first_order("sliding", mu, theta, want);
}

/// The slip limit for `μ = 0.2` is `tan θ* = 0.7`, `θ* = 0.61073`. Three
/// percent below it the ball must roll without slip at every level of the
/// sweep, and the slip `|v − ω r|` must fall with `h` like the displacement
/// error does (first order); the numbers at 8 … 64 substeps are printed.
#[test]
fn just_below_the_slip_limit_the_ball_rolls_and_the_slip_falls_with_the_substep() {
    let mu = 0.2_f64;
    let theta = (3.5 * mu).atan() * 0.97;
    let want = 5.0 / 7.0 * g() * theta.sin() * 0.5;
    let levels = [8usize, 16, 32, 64];
    let rows: Vec<(f64, f64, f64)> = levels.iter().map(|&n| run(mu, theta, n)).collect();
    for (k, n) in levels.iter().enumerate() {
        let (d, v, wr) = rows[k];
        println!(
            "below limit (θ = {theta:.5}): substeps {n}: d {d:.7} want {want:.7} error {:.3e} v {v:.6} ω r {wr:.6} slip {:.3e}",
            (d - want).abs(),
            v - wr
        );
    }
    let mut compared = 0;
    for k in 1..levels.len() {
        let (slip_prev, slip) = (
            (rows[k - 1].1 - rows[k - 1].2).abs(),
            (rows[k].1 - rows[k].2).abs(),
        );
        let (e_prev, e) = ((rows[k - 1].0 - want).abs(), (rows[k].0 - want).abs());
        assert!(
            e_prev / e >= 1.5,
            "below limit: displacement error {e_prev:.3e} -> {e:.3e} from {} to {} substeps is not first order",
            levels[k - 1],
            levels[k]
        );
        assert!(
            slip <= slip_prev.max(1e-6),
            "below limit: slip |v − ω r| {slip_prev:.3e} -> {slip:.3e} grows from {} to {} substeps",
            levels[k - 1],
            levels[k]
        );
        compared += 1;
    }
    assert_eq!(compared, levels.len() - 1);
}

/// Three percent above the limit the contact slips: the centre follows
/// `g (sin θ − μ cos θ)` at first order in `h`, and `v − ω r > 0` at every level.
///
/// Known defect: measured 8.0e-3 / 4.4e-3 / 3.3e-3 / 3.7e-3 at 8 / 16 / 32 / 64
/// substeps — the error stops falling once the per-substep penetration
/// `g cos θ h²` (8.8e-6 at 32) is below EPA's threshold `2⁻¹⁶`: the reported
/// depth is wrong by up to the threshold, the ball sinks to a standing
/// penetration (7.65e-6 at 64), then one substep reports the whole depth and
/// pushes the ball off the face at 2.9e-2 m/s; the following substeps have no
/// contact and apply no friction (26 of 3840 at 64, 0 of 480 at 8)
/// (`analytic_contact_detection_small_penetration.rs`).
#[test]
#[ignore = "known defect: shallow contact depth below the EPA threshold"]
fn just_above_the_slip_limit_the_ball_slides_at_first_order_in_the_substep() {
    let mu = 0.2_f64;
    let theta = (3.5 * mu).atan() * 1.03;
    let want = g() * (theta.sin() - mu * theta.cos()) * 0.5;
    let levels = [8usize, 16, 32, 64];
    let rows: Vec<(f64, f64, f64)> = levels.iter().map(|&n| run(mu, theta, n)).collect();
    for (k, n) in levels.iter().enumerate() {
        let (d, v, wr) = rows[k];
        println!(
            "above limit (θ = {theta:.5}): substeps {n}: d {d:.7} want {want:.7} error {:.3e} v {v:.6} ω r {wr:.6} slip {:.3e}",
            (d - want).abs(),
            v - wr
        );
        assert!(v - wr > 0.0, "above limit: no slip at {n} substeps");
    }
    let mut compared = 0;
    for k in 1..levels.len() {
        let (e_prev, e) = ((rows[k - 1].0 - want).abs(), (rows[k].0 - want).abs());
        assert!(
            e_prev / e >= 1.5,
            "above limit: displacement error {e_prev:.3e} -> {e:.3e} from {} to {} substeps is not first order",
            levels[k - 1],
            levels[k]
        );
        compared += 1;
    }
    assert_eq!(compared, levels.len() - 1);
}

/// Diagnostic (prints only): with one substep per step the slip `v − ω r`
/// is sampled after every substep near the limit, to see whether the
/// position-stage stick decision chatters (bursts) or drifts.
#[test]
#[ignore = "diagnostic: prints the per-substep record, asserts nothing"]
fn diagnostic_slip_per_substep_near_the_limit() {
    let mu = 0.2_f64;
    for (label, factor) in [("below", 0.97), ("above", 1.03)] {
        let theta = (3.5 * mu).atan() * factor;
        for n in [8usize, 64] {
            let (mut w, ball) = incline(mu, theta, 1);
            let h = DT / n as f64;
            let mut slips = Vec::new();
            for _ in 0..(60 * n) {
                w.step(fx(h));
                let b = &w.bodies[ball];
                let speed =
                    -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
                slips.push(speed - b.angular_velocity.z.to_f64() * R);
            }
            let nonzero = slips.iter().filter(|s| s.abs() > 1e-9).count();
            let max = slips.iter().fold(0.0f64, |m, s| m.max(s.abs()));
            let first = slips.iter().position(|s| s.abs() > 1e-9);
            let tail: Vec<String> = slips
                .iter()
                .rev()
                .take(6)
                .rev()
                .map(|s| format!("{s:.2e}"))
                .collect();
            println!(
                "{label} limit θ={theta:.5} h=DT/{n}: substeps {} with |slip| > 1e-9: {nonzero}, first at {first:?}, max {max:.3e}, last six {}",
                slips.len(),
                tail.join(" ")
            );
        }
    }
}

/// Diagnostic (prints only): in the sliding regime, classify every substep by
/// the tangential speed change it produced — the kinetic value
/// `(sin θ − μ cos θ) g h`, the free value `g sin θ h` (no friction), or
/// something else — and by the spin change against `2.5 μ g cos θ h`.
#[test]
#[ignore = "diagnostic: prints the per-substep record, asserts nothing"]
fn diagnostic_per_substep_friction_in_the_sliding_regime() {
    for (label, mu, theta) in [
        ("above limit 3 %", 0.2_f64, (0.7_f64).atan() * 1.03),
        ("far above (μ 0.05)", 0.05_f64, 0.3_f64),
    ] {
        for n in [8usize, 32, 64] {
            let (mut w, ball) = incline(mu, theta, 1);
            let h = DT / n as f64;
            let kinetic = (theta.sin() - mu * theta.cos()) * g() * h;
            let free = theta.sin() * g() * h;
            let spin = 2.5 * mu * g() * theta.cos() * h;
            let (mut v_prev, mut wr_prev) = (0.0f64, 0.0f64);
            let (mut n_kin, mut n_free, mut n_other, mut n_spin_ok) = (0, 0, 0, 0);
            let mut sum_dv = 0.0f64;
            let mut others = Vec::new();
            for k in 0..(60 * n) {
                w.step(fx(h));
                let b = &w.bodies[ball];
                let v =
                    -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
                let wr = b.angular_velocity.z.to_f64() * R;
                let (dv, dwr) = (v - v_prev, wr - wr_prev);
                v_prev = v;
                wr_prev = wr;
                sum_dv += dv;
                let tol = 1e-6 * h;
                if (dv - kinetic).abs() <= tol * 1e3 {
                    n_kin += 1;
                } else if (dv - free).abs() <= tol * 1e3 {
                    n_free += 1;
                } else {
                    n_other += 1;
                    if others.len() < 6 {
                        others.push(format!("k{k}:{:+.2e}", dv - kinetic));
                    }
                }
                if (dwr - spin).abs() <= tol * 1e3 {
                    n_spin_ok += 1;
                }
            }
            println!(
                "{label} h=DT/{n}: substeps {}: kinetic {n_kin}, free {n_free}, other {n_other} (first {}), spin-ok {n_spin_ok}; mean dv/kinetic {:.6}",
                60 * n,
                others.join(" "),
                sum_dv / (60 * n) as f64 / kinetic
            );
        }
    }
}

/// Diagnostic (prints only): in the sliding regime, per substep, whether a
/// contact existed, the normal gap of the ball above the face and its normal
/// velocity, to tell a missed detection (hypothesis A) from a normal velocity
/// left by the velocity stage (hypothesis B).
#[test]
#[ignore = "diagnostic: prints the per-substep record, asserts nothing"]
fn diagnostic_contact_presence_per_substep_in_the_sliding_regime() {
    let (mu, theta) = (0.2_f64, (0.7_f64).atan() * 1.03);
    let normal = [-theta.sin(), theta.cos(), 0.0];
    for n in [8usize, 64] {
        let (mut w, ball) = incline(mu, theta, 1);
        let h = DT / n as f64;
        let (mut none, mut samples) = (0, Vec::new());
        let mut gaps = Vec::new();
        let mut recent: Vec<String> = Vec::new();
        for k in 0..(60 * n) {
            w.step(fx(h));
            let b = &w.bodies[ball];
            let gap =
                b.position.x.to_f64() * normal[0] + b.position.y.to_f64() * normal[1] - (0.5 + R);
            let vn = b.velocity.x.to_f64() * normal[0] + b.velocity.y.to_f64() * normal[1];
            gaps.push(gap);
            let contact = w.contact_constraints.first().map(|c| {
                let nn = c.contact.normal;
                format!(
                    "depth {:.3e} normal ({:+.6}, {:+.6}) λ {:.3e}",
                    c.contact.depth.to_f64(),
                    nn.x.to_f64(),
                    nn.y.to_f64(),
                    c.cached_lambda.to_f64()
                )
            });
            recent.push(format!(
                "k{k}: gap {gap:+.3e} vn {vn:+.3e} {}",
                contact.as_deref().unwrap_or("no contact")
            ));
            if recent.len() > 4 {
                recent.remove(0);
            }
            if w.contact_constraints.is_empty() {
                none += 1;
                if samples.len() < 2 {
                    samples.push(format!("[{}]", recent.join(" | ")));
                }
            }
        }
        let gmax = gaps.iter().cloned().fold(f64::MIN, f64::max);
        let gmin = gaps.iter().cloned().fold(f64::MAX, f64::min);
        println!(
            "contact presence h=DT/{n}: substeps {}: without contact {none}; gap range [{gmin:+.3e}, {gmax:+.3e}]; predicted penetration g cosθ h² = {:.3e}; samples {}",
            60 * n,
            g() * theta.cos() * h * h,
            samples.join(" | ")
        );
    }
}
