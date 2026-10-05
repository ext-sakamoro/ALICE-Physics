//! Oracles for `crowd_force::SocialForce` (Helbing–Molnár 1995 /
//! Helbing–Farkas–Vicsek 2000 social force model).
//!
//! # Where the expected values come from
//!
//! - Driving term `m (v0 ê − v)/τ` alone: `dv/dt = (v0 ê − v)/τ`, so
//!   `v(t) = v0 ê + (v(0) − v0 ê) e^{−t/τ}`. The semi-implicit Euler step of
//!   the module gives the discrete solution `v_n = v0 ê + (v(0) − v0 ê)(1 − h/τ)^n`
//!   exactly (one multiply-add per step).
//! - Pedestrian pair (Helbing, Farkas, Vicsek, *Nature* 407, 487 (2000), eq. 2):
//!   `f_ij = {A e^{(r_ij − d_ij)/B} + k g(r_ij − d_ij)} n_ij + κ g(r_ij − d_ij) Δv^t_ji t_ij`
//!   with `g(x) = max(x, 0)`, `n_ij = (x_i − x_j)/d_ij`, `t_ij = (−n_ij,y, n_ij,x)`,
//!   `Δv^t_ji = (v_j − v_i)·t_ij`.
//! - View-angle anisotropy on the social term (Helbing & Johansson,
//!   *Encyclopedia of Complexity and Systems Science* (2009); Johansson,
//!   Helbing, Shukla, *Adv. Complex Syst.* 10 (2007)):
//!   `w = λ + (1 − λ)(1 + cos φ_ij)/2`, `cos φ_ij = −n_ij·ê_i`.
//! - Wall (HFV 2000, eq. 3): `f_iW = {A_W e^{(r_i − d_iW)/B_W} + k g(r_i − d_iW)} n_iW
//!   − κ g(r_i − d_iW)(v_i·t_iW) t_iW`.
//! - Steady state against a wall without contact: `m v0/τ = A_W e^{(r − d*)/B_W}`
//!   ⇒ `d* = r + B_W ln(A_W τ/(m v0))`.
//!
//! Each expected value is evaluated here in `f64` from these formulas, not by
//! calling the crate.
//!
//! # Tolerances
//!
//! - Terms with an exponential: relative `2e-6`. `Fix128::exp` is bounded at
//!   relative `1e-6` for `|x| ≤ 40` by its own sweep (`src/math.rs`, doc of
//!   `Fix128::exp`); the factor 2 covers the handful of `Fix128` products after
//!   it (2⁻⁶⁴ each). A wrong sign in the exponent or `B` in the wrong place is
//!   off by a factor `e^{0.4/0.08}` or more.
//! - Terms without an exponential (body force, friction, driving): `1e-9`
//!   absolute against values of order 1e2–1e4 (only `Fix128` products and one
//!   `sqrt`).
//! - Bit identity (cell list vs direct sum, determinism, translation, Σ of
//!   internal forces): exact. `Fix128` addition is an exact integer addition,
//!   so the sum does not depend on order.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 transcendentals compute closed-form references, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::crowd_force::{
    CrowdForceError, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
};
use alice_physics::math::Fix128;
use alice_physics::physics2d::Vec2Fix;

// Representative values of HFV 2000 (Nature 407, 487, methods).
const A: f64 = 2000.0; // N
const B: f64 = 0.08; // m
const K: f64 = 1.2e5; // kg/s²
const KAPPA: f64 = 2.4e5; // kg/(m s)
const MASS: f64 = 80.0; // kg
const TAU: f64 = 0.5; // s
const LAMBDA: f64 = 0.1;

const REL_EXP: f64 = 2e-6;
const ABS: f64 = 1e-9;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v2(x: f64, y: f64) -> Vec2Fix {
    Vec2Fix::new(fx(x), fx(y))
}

fn to2(v: Vec2Fix) -> [f64; 2] {
    [v.x.to_f64(), v.y.to_f64()]
}

fn params() -> InteractionParams {
    InteractionParams {
        strength_n: fx(A),
        range_m: fx(B),
        body_stiffness: fx(K),
        sliding_friction: fx(KAPPA),
    }
}

fn model(lambda: f64, cutoff: f64) -> SocialForce {
    SocialForce::new(params(), params(), fx(lambda), fx(cutoff)).expect("valid params")
}

#[allow(clippy::too_many_arguments)] // one literal per field keeps the scenes readable
fn ped(x: f64, y: f64, vx: f64, vy: f64, r: f64, ex: f64, ey: f64, v0: f64) -> Pedestrian {
    Pedestrian {
        position: v2(x, y),
        velocity: v2(vx, vy),
        radius_m: fx(r),
        mass_kg: fx(MASS),
        desired_speed_m_s: fx(v0),
        desired_direction: v2(ex, ey),
        relaxation_time_s: fx(TAU),
    }
}

fn rel(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = ((actual - expected) / expected).abs();
    assert!(
        err <= tol,
        "{label}: actual {actual:e}, expected {expected:e}, rel err {err:e} > {tol:e}"
    );
}

fn abs_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{label}: actual {actual:e}, expected {expected:e}, abs err {err:e} > {tol:e}"
    );
}

fn view_weight(lambda: f64, cos_phi: f64) -> f64 {
    lambda + (1.0 - lambda) * (1.0 + cos_phi) / 2.0
}

// ---------------------------------------------------------------------------
// Driving term
// ---------------------------------------------------------------------------

#[test]
fn driving_force_matches_closed_form() {
    let m = model(LAMBDA, 3.0);
    let p = ped(0.0, 0.0, 0.3, -0.2, 0.3, 0.6, 0.8, 1.34);
    let f = to2(m.driving_force(&p).expect("valid"));
    let ex = MASS * (1.34 * 0.6 - 0.3) / TAU;
    let ey = MASS * (1.34 * 0.8 + 0.2) / TAU;
    abs_close("driving x", f[0], ex, ABS);
    abs_close("driving y", f[1], ey, ABS);
}

#[test]
fn driving_direction_is_normalised() {
    // ê given with length 5: the desired velocity is still v0 ê / |ê|.
    let m = model(LAMBDA, 3.0);
    let p = ped(0.0, 0.0, 0.0, 0.0, 0.3, 3.0, 4.0, 1.0);
    let f = to2(m.driving_force(&p).expect("valid"));
    abs_close("x", f[0], MASS * 0.6 / TAU, 1e-9);
    abs_close("y", f[1], MASS * 0.8 / TAU, 1e-9);
}

#[test]
fn speed_relaxes_exponentially_to_desired_speed() {
    let m = model(LAMBDA, 3.0);
    let (v0, vi, h) = (1.34_f64, 0.2_f64, 0.01_f64);
    let mut peds = vec![ped(0.0, 0.0, vi, 0.0, 0.3, 1.0, 0.0, v0)];
    let n_steps = 200; // t = 2 s = 4 τ
    for n in 1..=n_steps {
        m.step(&mut peds, &[], fx(h), NeighborSearch::Direct, None)
            .expect("step");
        let t = n as f64 * h;
        let v = peds[0].velocity.x.to_f64();
        // Discrete solution of the semi-implicit Euler step (exact up to 2⁻⁶⁴ per step).
        let x = h / TAU;
        let discrete = v0 + (vi - v0) * (1.0 - x).powi(n);
        abs_close("discrete relaxation", v, discrete, 1e-12);
        // Continuous v(t) = v0 + (vi − v0) e^{−t/τ}. (1−x)^n = e^{−t/τ} e^{−y} with
        // 0 ≤ y = n(x²/2 + x³/3 + …) ≤ n x² / (2 (1 − x)), and 1 − e^{−y} ≤ y.
        let cont = v0 + (vi - v0) * (-t / TAU).exp();
        let bound = (vi - v0).abs() * (-t / TAU).exp() * n as f64 * x * x / (2.0 * (1.0 - x));
        assert!(
            (v - cont).abs() <= bound + 1e-12,
            "t={t}: |v − v_exact| = {:e} > bound {bound:e}",
            (v - cont).abs()
        );
        assert_eq!(peds[0].velocity.y, Fix128::ZERO, "no lateral drift");
    }
    // After 4 τ the gap to v0 is e^{−4} of the initial gap.
    let v = peds[0].velocity.x.to_f64();
    rel(
        "gap after 4τ",
        v0 - v,
        (v0 - vi) * (1.0 - h / TAU).powi(200),
        1e-9,
    );
}

// ---------------------------------------------------------------------------
// Pedestrian pair
// ---------------------------------------------------------------------------

#[test]
fn pair_repulsion_magnitude_and_direction() {
    let m = model(LAMBDA, 3.0);
    // i at origin heading +x, j ahead at d = 0.8 (no contact, r_ij = 0.6).
    let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    let j = ped(0.8, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    let (fi, fj) = m.pair_forces(&i, &j);
    let mag = A * ((0.6 - 0.8) / B).exp();
    // j ahead of i: cos φ = 1, w_i = 1. n_ij = (−1, 0).
    rel("f_i x", -to2(fi)[0], mag, REL_EXP);
    abs_close("f_i y", to2(fi)[1], 0.0, 0.0);
    // i is behind j (j heads +x): cos φ = −1, w_j = λ. n_ji = (+1, 0).
    rel("f_j x", to2(fj)[0], LAMBDA * mag, REL_EXP);
    abs_close("f_j y", to2(fj)[1], 0.0, 0.0);
}

#[test]
fn pair_repulsion_in_oblique_direction() {
    let m = model(1.0, 3.0);
    let i = ped(1.0, 2.0, 0.0, 0.0, 0.25, 1.0, 0.0, 1.0);
    let j = ped(1.6, 2.8, 0.0, 0.0, 0.35, 1.0, 0.0, 1.0);
    let (fi, _) = m.pair_forces(&i, &j);
    let d = (0.6f64 * 0.6 + 0.8 * 0.8).sqrt(); // 1.0
    let mag = A * ((0.6 - d) / B).exp();
    rel("f_i x", to2(fi)[0], mag * -0.6, REL_EXP);
    rel("f_i y", to2(fi)[1], mag * -0.8, REL_EXP);
}

#[test]
fn view_angle_anisotropy() {
    let m = model(LAMBDA, 3.0);
    let j = ped(0.9, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 1.0);
    let mag = A * ((0.6 - 0.9) / B).exp();
    // (heading of i, cos φ = −n_ij·ê_i with n_ij = (−1, 0))
    for (ex, ey, cos_phi) in [
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (0.6, 0.8, 0.6),
        (-0.8, 0.6, -0.8),
    ] {
        let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, ex, ey, 1.0);
        let (fi, _) = m.pair_forces(&i, &j);
        let w = view_weight(LAMBDA, cos_phi);
        rel(&format!("ê=({ex},{ey})"), -to2(fi)[0], w * mag, REL_EXP);
        // The weight is also exposed on its own.
        let n_ij = v2(-1.0, 0.0);
        abs_close(
            &format!("weight ê=({ex},{ey})"),
            m.anisotropy_weight(&i, n_ij).to_f64(),
            w,
            1e-12,
        );
    }
    // Behind vs ahead: exactly λ.
    let ahead = m
        .pair_forces(&ped(0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0), &j)
        .0;
    let behind = m
        .pair_forces(&ped(0.0, 0.0, 0.0, 0.0, 0.3, -1.0, 0.0, 1.0), &j)
        .0;
    rel(
        "behind / ahead",
        behind.x.to_f64() / ahead.x.to_f64(),
        LAMBDA,
        1e-12,
    );
}

#[test]
fn view_angle_does_not_depend_on_velocity() {
    // The heading is the desired direction ê, so a pedestrian at rest still
    // has a defined view angle (velocity 0 is not a degenerate input).
    let m = model(LAMBDA, 3.0);
    let j = ped(0.9, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 1.0);
    let at_rest = ped(0.0, 0.0, 0.0, 0.0, 0.3, -1.0, 0.0, 1.0);
    let moving = ped(0.0, 0.0, 0.7, 0.3, 0.3, -1.0, 0.0, 1.0);
    let mag = A * ((0.6 - 0.9) / B).exp();
    rel(
        "at rest",
        -m.pair_forces(&at_rest, &j).0.x.to_f64(),
        LAMBDA * mag,
        REL_EXP,
    );
    assert_eq!(m.pair_forces(&at_rest, &j), m.pair_forces(&moving, &j));
}

#[test]
fn zero_desired_direction_is_isotropic() {
    // ê = 0 (no preferred direction): w = 1 on every side.
    let m = model(LAMBDA, 3.0);
    let j = ped(0.9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let mag = A * ((0.6 - 0.9) / B).exp();
    let (fi, fj) = m.pair_forces(&i, &j);
    rel("f_i", -fi.x.to_f64(), mag, REL_EXP);
    rel("f_j", fj.x.to_f64(), mag, REL_EXP);
    assert_eq!(m.anisotropy_weight(&i, v2(-1.0, 0.0)), Fix128::ONE);
}

#[test]
fn body_force_and_sliding_friction_in_contact() {
    let m = model(1.0, 3.0);
    // d = 0.5 < r_ij = 0.6: overlap 0.1. j moves +y relative to i.
    let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let j = ped(0.5, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0);
    let (fi, fj) = m.pair_forces(&i, &j);
    let overlap = 0.1;
    let social = A * (overlap / B).exp();
    // n_ij = (−1, 0); t_ij = (0, −1); Δv^t_ji = (v_j − v_i)·t_ij = −1
    // friction on i = κ g Δv t = κ · 0.1 · (−1) · (0, −1) = (0, +κ · 0.1)
    rel("f_i x", -fi.x.to_f64(), social + K * overlap, REL_EXP);
    rel(
        "f_i y (friction drags i along j)",
        fi.y.to_f64(),
        KAPPA * overlap,
        1e-12,
    );
    // Body force and friction are antisymmetric bit for bit.
    assert_eq!(fj.y, -fi.y, "friction antisymmetric");
    // Body force alone, without the exponential: take A = 0.
    let no_social = SocialForce::new(
        InteractionParams {
            strength_n: Fix128::ZERO,
            ..params()
        },
        params(),
        Fix128::ONE,
        fx(3.0),
    )
    .expect("valid");
    let (gi, gj) = no_social.pair_forces(&i, &j);
    abs_close("body only x", gi.x.to_f64(), -K * overlap, ABS);
    abs_close("body only y", gi.y.to_f64(), KAPPA * overlap, ABS);
    assert_eq!(gi, -gj, "isotropic pair forces antisymmetric");
}

#[test]
fn friction_uses_only_the_tangential_relative_velocity() {
    let m = SocialForce::new(
        InteractionParams {
            strength_n: Fix128::ZERO,
            body_stiffness: Fix128::ZERO,
            ..params()
        },
        params(),
        Fix128::ONE,
        fx(3.0),
    )
    .expect("valid");
    // Relative velocity along the normal only: no friction.
    let i = ped(0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0);
    let j = ped(0.4, 0.0, -0.5, 0.0, 0.3, 0.0, 0.0, 0.0);
    assert_eq!(m.pair_forces(&i, &j).0, Vec2Fix::ZERO);
    // Oblique: n_ij along (−0.6, −0.8), relative velocity (v_j − v_i) = (1, 0.5).
    let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let j = ped(0.3, 0.4, 1.0, 0.5, 0.3, 0.0, 0.0, 0.0);
    let (fi, _) = m.pair_forces(&i, &j);
    let (nx, ny) = (-0.6, -0.8);
    let (tx, ty) = (-ny, nx);
    let dvt = 1.0 * tx + 0.5 * ty;
    let g = 0.6 - 0.5;
    abs_close("x", fi.x.to_f64(), KAPPA * g * dvt * tx, 1e-9);
    abs_close("y", fi.y.to_f64(), KAPPA * g * dvt * ty, 1e-9);
}

#[test]
fn no_body_force_or_friction_without_contact() {
    let m = SocialForce::new(
        InteractionParams {
            strength_n: Fix128::ZERO,
            ..params()
        },
        params(),
        Fix128::ONE,
        fx(3.0),
    )
    .expect("valid");
    // d = 0.7 > r_ij = 0.6 with tangential relative velocity: g = 0.
    let i = ped(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let j = ped(0.7, 0.0, 0.0, 2.0, 0.3, 0.0, 0.0, 0.0);
    assert_eq!(m.pair_forces(&i, &j), (Vec2Fix::ZERO, Vec2Fix::ZERO));
}

#[test]
fn internal_forces_sum_to_zero_when_isotropic() {
    let m = model(1.0, 2.0);
    let peds = scattered(60, 4.0, 0.0);
    let mut f = Vec::new();
    m.total_forces(&peds, &[], NeighborSearch::Direct, &mut f)
        .expect("valid");
    // Subtract the driving term (exact: the total is driving + Σ pair terms).
    let mut sum = Vec2Fix::ZERO;
    for (p, fi) in peds.iter().zip(&f) {
        sum = sum + (*fi - m.driving_force(p).expect("valid"));
    }
    assert_eq!(sum, Vec2Fix::ZERO, "Σ internal forces = 0 exactly (λ = 1)");
    // Some pairs are in contact, otherwise the check is weak.
    let contacts = count_contacts(&peds);
    assert!(contacts >= 5, "only {contacts} contacts");
}

#[test]
fn anisotropic_internal_forces_do_not_sum_to_zero() {
    // With λ < 1 the social term is not reciprocal (w_i ≠ w_j); the body force and
    // friction still are. Σ equals Σ over pairs of (w_i − w_j) A e^{..} n_ij.
    let m = model(LAMBDA, 2.0);
    let peds = scattered(60, 4.0, 0.0);
    let mut f = Vec::new();
    m.total_forces(&peds, &[], NeighborSearch::Direct, &mut f)
        .expect("valid");
    let mut sum = [0.0f64; 2];
    for (p, fi) in peds.iter().zip(&f) {
        let d = to2(*fi - m.driving_force(p).expect("valid"));
        sum[0] += d[0];
        sum[1] += d[1];
    }
    let mut expect = [0.0f64; 2];
    for a in 0..peds.len() {
        for b in a + 1..peds.len() {
            let (pa, pb) = (to2(peds[a].position), to2(peds[b].position));
            let (dx, dy) = (pa[0] - pb[0], pa[1] - pb[1]);
            let d = (dx * dx + dy * dy).sqrt();
            if d > 2.0 {
                continue;
            }
            let n = [dx / d, dy / d];
            let ea = unit(to2(peds[a].desired_direction));
            let eb = unit(to2(peds[b].desired_direction));
            let wa = view_weight(LAMBDA, -(n[0] * ea[0] + n[1] * ea[1]));
            let wb = view_weight(LAMBDA, n[0] * eb[0] + n[1] * eb[1]);
            let rij = peds[a].radius_m.to_f64() + peds[b].radius_m.to_f64();
            let s = A * ((rij - d) / B).exp();
            expect[0] += (wa - wb) * s * n[0];
            expect[1] += (wa - wb) * s * n[1];
        }
    }
    let scale = expect[0].abs().max(expect[1].abs());
    assert!(
        scale > 1.0,
        "anisotropic residual too small to test: {scale}"
    );
    abs_close("Σ x", sum[0], expect[0], 1e-4 * scale);
    abs_close("Σ y", sum[1], expect[1], 1e-4 * scale);
}

// ---------------------------------------------------------------------------
// Walls
// ---------------------------------------------------------------------------

#[test]
fn wall_repulsion_closed_form() {
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    let p = ped(1.0, 0.5, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    let f = to2(m.wall_force(&p, &w));
    rel("y", f[1], A * ((0.3 - 0.5) / B).exp(), REL_EXP);
    // The nearest point s = 0.6 is not dyadic: x carries its rounding only.
    abs_close("x", f[0], 0.0, 1e-12);
    // Below the wall the force points to −y (the wall is two-sided).
    let p = ped(1.0, -0.5, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    rel(
        "y below",
        -to2(m.wall_force(&p, &w))[1],
        A * ((0.3 - 0.5) / B).exp(),
        REL_EXP,
    );
}

#[test]
fn wall_force_beyond_segment_end_points_away_from_end() {
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(0.0, 0.0),
    };
    // Nearest point is the end (0, 0); distance 0.5 along (0.6, 0.8).
    let p = ped(0.3, 0.4, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    let f = to2(m.wall_force(&p, &w));
    let mag = A * ((0.3 - 0.5) / B).exp();
    rel("x", f[0], 0.6 * mag, REL_EXP);
    rel("y", f[1], 0.8 * mag, REL_EXP);
}

#[test]
fn wall_contact_body_force_and_friction() {
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    // Overlap 0.1, sliding along +x at 1.2 m/s.
    let p = ped(0.0, 0.2, 1.2, 0.0, 0.3, 1.0, 0.0, 1.0);
    let f = to2(m.wall_force(&p, &w));
    rel("normal", f[1], A * (0.1 / B).exp() + K * 0.1, REL_EXP);
    // friction −κ g (v·t) t opposes the sliding velocity
    abs_close("friction", f[0], -KAPPA * 0.1 * 1.2, 1e-8);
}

#[test]
fn degenerate_wall_segment_is_a_point() {
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(2.0, 2.0),
        end: v2(2.0, 2.0),
    };
    let p = ped(2.0, 2.6, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    let f = to2(m.wall_force(&p, &w));
    rel("y", f[1], A * ((0.3 - 0.6) / B).exp(), REL_EXP);
    assert_eq!(f[0], 0.0);
}

#[test]
fn steady_distance_in_front_of_wall() {
    // Walking into a wall at x = 0 (ê = +x): the pedestrian stops where the
    // wall repulsion balances the driving force, d* = r + B ln(A τ / (m v0)).
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(0.0, -5.0),
        end: v2(0.0, 5.0),
    };
    let v0 = 0.8;
    let mut peds = vec![ped(-3.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, v0)];
    for _ in 0..4000 {
        m.step(&mut peds, &[w], fx(0.01), NeighborSearch::Direct, None)
            .expect("step");
    }
    let d = -peds[0].position.x.to_f64();
    let expected = 0.3 + B * (A * TAU / (MASS * v0)).ln();
    // exp is relative 1e-6 ⇒ d is off by ≤ B · 1e-6; the transient has decayed
    // as e^{−t/(2τ)} = e^{−40}.
    abs_close("d*", d, expected, B * 2e-6);
    abs_close("v at rest", peds[0].velocity.x.to_f64(), 0.0, 1e-9);
    assert!(expected > 0.3, "the steady state is not in contact");
}

// ---------------------------------------------------------------------------
// Neighbour search and determinism
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random crowd (64-bit LCG, dyadic coordinates).
fn scattered(n: usize, side: f64, v0: f64) -> Vec<Pedestrian> {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f64) / (1u64 << 24) as f64
    };
    (0..n)
        .map(|_| {
            let x = (next() * side * 1024.0).round() / 1024.0;
            let y = (next() * side * 1024.0).round() / 1024.0;
            let vx = next() - 0.5;
            let vy = next() - 0.5;
            let r = 0.25 + 0.1 * next();
            let a = next() * std::f64::consts::TAU;
            ped(x, y, vx, vy, r, a.cos(), a.sin(), v0)
        })
        .collect()
}

fn unit(v: [f64; 2]) -> [f64; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
    [v[0] / l, v[1] / l]
}

fn count_contacts(peds: &[Pedestrian]) -> usize {
    let mut c = 0;
    for a in 0..peds.len() {
        for b in a + 1..peds.len() {
            let (pa, pb) = (to2(peds[a].position), to2(peds[b].position));
            let d = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
            if d < peds[a].radius_m.to_f64() + peds[b].radius_m.to_f64() {
                c += 1;
            }
        }
    }
    c
}

#[test]
fn cell_list_matches_direct_sum_bit_for_bit() {
    for (n, side, cutoff) in [(200, 20.0, 2.0), (120, 6.0, 1.5), (50, 40.0, 3.0)] {
        let m = model(LAMBDA, cutoff);
        let peds = scattered(n, side, 1.2);
        let walls = [
            WallSegment {
                start: v2(-1.0, -1.0),
                end: v2(side + 1.0, -1.0),
            },
            WallSegment {
                start: v2(-1.0, side + 1.0),
                end: v2(side + 1.0, side + 1.0),
            },
        ];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        m.total_forces(&peds, &walls, NeighborSearch::Direct, &mut a)
            .expect("direct");
        m.total_forces(&peds, &walls, NeighborSearch::CellList, &mut b)
            .expect("cell list");
        assert_eq!(a, b, "n={n} side={side} cutoff={cutoff}");
    }
}

#[test]
fn cell_list_finds_pairs_at_exactly_the_cutoff_across_cells() {
    // A pair whose distance is exactly the cutoff, straddling cell boundaries
    // and the origin, interacts in both searches.
    let cutoff = 1.5;
    let m = model(1.0, cutoff);
    for (x0, y0) in [(-0.75, 0.0), (1.5, 3.0), (-3.0, -1.5), (2.75, -0.125)] {
        let peds = vec![
            ped(x0, y0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0),
            ped(x0 + cutoff, y0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0),
        ];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        m.total_forces(&peds, &[], NeighborSearch::Direct, &mut a)
            .expect("direct");
        m.total_forces(&peds, &[], NeighborSearch::CellList, &mut b)
            .expect("cell");
        assert_eq!(a, b, "x0={x0}");
        rel(
            "pair at cutoff",
            -a[0].x.to_f64(),
            A * ((0.6 - cutoff) / B).exp(),
            REL_EXP,
        );
    }
}

#[test]
fn pairs_beyond_the_cutoff_do_not_interact() {
    let m = model(1.0, 1.0);
    let peds = vec![
        ped(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0),
        ped(1.0 + 1.0 / 1024.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0),
    ];
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        let mut f = Vec::new();
        m.total_forces(&peds, &[], search, &mut f).expect("valid");
        assert_eq!(f, vec![Vec2Fix::ZERO, Vec2Fix::ZERO], "{search:?}");
    }
}

#[test]
fn forces_are_deterministic_and_translation_invariant() {
    let m = model(LAMBDA, 2.0);
    let peds = scattered(150, 12.0, 1.2);
    let mut moved = peds.clone();
    // A dyadic shift keeps every difference x_i − x_j bit-identical.
    for p in &mut moved {
        p.position = p.position + v2(1024.5, -333.25);
    }
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    m.total_forces(&peds, &[], NeighborSearch::CellList, &mut a)
        .expect("1");
    m.total_forces(&peds, &[], NeighborSearch::CellList, &mut b)
        .expect("2");
    m.total_forces(&moved, &[], NeighborSearch::CellList, &mut c)
        .expect("3");
    assert_eq!(a, b, "same input, same bits");
    assert_eq!(a, c, "translated crowd, same bits");

    let mut s1 = peds.clone();
    let mut s2 = peds;
    for _ in 0..20 {
        m.step(
            &mut s1,
            &[],
            fx(0.005),
            NeighborSearch::Direct,
            Some(fx(1.3)),
        )
        .expect("s1");
        m.step(
            &mut s2,
            &[],
            fx(0.005),
            NeighborSearch::CellList,
            Some(fx(1.3)),
        )
        .expect("s2");
    }
    assert_eq!(
        s1, s2,
        "trajectories bit-identical between the two searches"
    );
}

// ---------------------------------------------------------------------------
// Speed cap
// ---------------------------------------------------------------------------

#[test]
fn speed_cap_limits_to_ratio_times_desired_speed() {
    let m = model(LAMBDA, 3.0);
    // Pushed hard by a wall contact so that the uncapped speed exceeds 1.3 v0.
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    let mut peds = vec![ped(0.0, 0.1, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0)];
    m.step(
        &mut peds,
        &[w],
        fx(0.01),
        NeighborSearch::Direct,
        Some(fx(1.3)),
    )
    .expect("step");
    let v = to2(peds[0].velocity);
    abs_close(
        "|v| = 1.3 v0",
        (v[0] * v[0] + v[1] * v[1]).sqrt(),
        1.3,
        1e-12,
    );
    // Position advanced with the capped velocity.
    abs_close("x", peds[0].position.x.to_f64(), v[0] * 0.01, 1e-15);
    // Uncapped the speed is much larger.
    let mut free = vec![ped(0.0, 0.1, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0)];
    m.step(&mut free, &[w], fx(0.01), NeighborSearch::Direct, None)
        .expect("step");
    // v_y = h/m (A e^{0.2/B} + k 0.2) ≈ 6.05 m/s, v_x = h v0/τ = 0.02 m/s.
    rel(
        "uncapped v_y",
        free[0].velocity.y.to_f64(),
        0.01 / MASS * (A * (0.2 / B).exp() + K * 0.2),
        REL_EXP,
    );
    abs_close(
        "uncapped v_x",
        free[0].velocity.x.to_f64(),
        0.01 * 1.0 / TAU,
        1e-12,
    );
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn coincident_pedestrians_exert_no_force_on_each_other() {
    // n_ij is undefined at d = 0; the module takes n_ij = 0, so the pair force
    // is exactly 0 on both (no preferred direction, Σ = 0 preserved).
    let m = model(LAMBDA, 3.0);
    let i = ped(1.0, 1.0, 0.5, 0.0, 0.3, 1.0, 0.0, 1.0);
    let j = ped(1.0, 1.0, -0.5, 0.3, 0.3, -1.0, 0.0, 1.0);
    assert_eq!(m.pair_forces(&i, &j), (Vec2Fix::ZERO, Vec2Fix::ZERO));
    // In a crowd the pair is skipped the same way; the rest is unaffected.
    let k = ped(1.8, 1.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
    let mut f = Vec::new();
    m.total_forces(&[i, j, k], &[], NeighborSearch::CellList, &mut f)
        .expect("valid");
    let expect_i = m.driving_force(&i).expect("i") + m.pair_forces(&i, &k).0;
    assert_eq!(f[0], expect_i);
}

#[test]
fn pedestrian_on_the_wall_line_gets_no_wall_force() {
    let m = model(LAMBDA, 3.0);
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    let p = ped(1.0, 0.0, 1.0, 0.0, 0.3, 1.0, 0.0, 1.0);
    assert_eq!(m.wall_force(&p, &w), Vec2Fix::ZERO);
}

#[test]
fn invalid_parameters_are_rejected() {
    let bad_b = InteractionParams {
        range_m: Fix128::ZERO,
        ..params()
    };
    assert_eq!(
        SocialForce::new(bad_b, params(), fx(0.5), fx(2.0)).unwrap_err(),
        CrowdForceError::NonPositiveRange
    );
    let neg_b = InteractionParams {
        range_m: fx(-0.1),
        ..params()
    };
    assert_eq!(
        SocialForce::new(params(), neg_b, fx(0.5), fx(2.0)).unwrap_err(),
        CrowdForceError::NonPositiveRange
    );
    let neg_a = InteractionParams {
        strength_n: fx(-1.0),
        ..params()
    };
    assert_eq!(
        SocialForce::new(neg_a, params(), fx(0.5), fx(2.0)).unwrap_err(),
        CrowdForceError::NegativeCoefficient
    );
    assert_eq!(
        SocialForce::new(params(), params(), fx(1.5), fx(2.0)).unwrap_err(),
        CrowdForceError::AnisotropyOutOfRange
    );
    assert_eq!(
        SocialForce::new(params(), params(), fx(-0.1), fx(2.0)).unwrap_err(),
        CrowdForceError::AnisotropyOutOfRange
    );
    assert_eq!(
        SocialForce::new(params(), params(), fx(0.5), Fix128::ZERO).unwrap_err(),
        CrowdForceError::NonPositiveCutoff
    );
}

#[test]
fn invalid_pedestrians_are_rejected_with_index() {
    let m = model(LAMBDA, 2.0);
    let mut peds = vec![
        ped(0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0),
        ped(5.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0),
    ];
    peds[1].relaxation_time_s = Fix128::ZERO;
    let mut f = Vec::new();
    assert_eq!(
        m.total_forces(&peds, &[], NeighborSearch::Direct, &mut f),
        Err(CrowdForceError::NonPositiveRelaxationTime { index: 1 })
    );
    assert_eq!(
        m.driving_force(&peds[1]),
        Err(CrowdForceError::NonPositiveRelaxationTime { index: 0 })
    );
    peds[1].relaxation_time_s = fx(-0.5);
    let before = peds.clone();
    assert_eq!(
        m.step(&mut peds, &[], fx(0.01), NeighborSearch::Direct, None),
        Err(CrowdForceError::NonPositiveRelaxationTime { index: 1 })
    );
    assert_eq!(peds, before, "a rejected step leaves the crowd unchanged");
    peds[1].relaxation_time_s = fx(TAU);
    peds[0].mass_kg = Fix128::ZERO;
    assert_eq!(
        m.step(&mut peds, &[], fx(0.01), NeighborSearch::Direct, None),
        Err(CrowdForceError::NonPositiveMass { index: 0 })
    );
    peds[0].mass_kg = fx(MASS);
    peds[0].radius_m = fx(-0.1);
    assert_eq!(
        m.step(&mut peds, &[], fx(0.01), NeighborSearch::CellList, None),
        Err(CrowdForceError::NegativeRadius { index: 0 })
    );
    peds[0].radius_m = fx(0.3);
    peds[1].desired_speed_m_s = fx(-1.0);
    assert_eq!(
        m.step(&mut peds, &[], fx(0.01), NeighborSearch::CellList, None),
        Err(CrowdForceError::NegativeDesiredSpeed { index: 1 })
    );
    peds[1].desired_speed_m_s = fx(1.0);
    assert_eq!(
        m.step(&mut peds, &[], Fix128::ZERO, NeighborSearch::Direct, None),
        Err(CrowdForceError::NonPositiveTimeStep)
    );
    assert_eq!(
        m.step(
            &mut peds,
            &[],
            fx(0.01),
            NeighborSearch::Direct,
            Some(Fix128::ZERO)
        ),
        Err(CrowdForceError::NonPositiveSpeedCap)
    );
}

#[test]
fn empty_and_single_pedestrian_crowds() {
    let m = model(LAMBDA, 2.0);
    let mut f = vec![Vec2Fix::UNIT_X; 3];
    m.total_forces(&[], &[], NeighborSearch::CellList, &mut f)
        .expect("N = 0");
    assert!(f.is_empty(), "N = 0 clears the output");
    let mut none: Vec<Pedestrian> = Vec::new();
    m.step(&mut none, &[], fx(0.01), NeighborSearch::Direct, None)
        .expect("N = 0 step");

    // N = 1: only the driving force and the walls.
    let p = ped(0.0, 0.5, 0.2, 0.0, 0.3, 1.0, 0.0, 1.0);
    let w = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        m.total_forces(&[p], &[w], search, &mut f).expect("N = 1");
        assert_eq!(
            f,
            vec![m.driving_force(&p).expect("p") + m.wall_force(&p, &w)]
        );
    }
}

// ---------------------------------------------------------------------------
// Bottleneck flow (literature comparison)
// ---------------------------------------------------------------------------

/// Crowd of `n` leaving a 12 m × 12 m room through a door of width `width`
/// in the wall `x = 0`, heading for the door centre (HFV 2000 parameters,
/// isotropic `λ = 1` as in that paper, `v0 = 1.34 m/s`, cap `1.3 v0`,
/// `h = 4 ms`). Returns the crossing times of `x = 0` in order.
fn bottleneck_crossings(n: usize, width: f64, t_end: f64) -> Vec<f64> {
    let m = model(1.0, 1.2);
    let half = width / 2.0;
    let seg = |a: (f64, f64), b: (f64, f64)| WallSegment {
        start: v2(a.0, a.1),
        end: v2(b.0, b.1),
    };
    let walls = [
        seg((0.0, -6.0), (0.0, -half)),
        seg((0.0, half), (0.0, 6.0)),
        seg((-12.0, -6.0), (0.0, -6.0)),
        seg((-12.0, 6.0), (0.0, 6.0)),
        seg((-12.0, -6.0), (-12.0, 6.0)),
    ];
    let cols = 8;
    let mut peds: Vec<Pedestrian> = (0..n)
        .map(|k| {
            let (c, r) = ((k % cols) as f64, (k / cols) as f64);
            let x = -1.5 - 0.875 * r;
            let y = -4.375 + 1.25 * c + if (k / cols) % 2 == 0 { 0.0 } else { 0.5 };
            let radius = 0.25 + 0.1 * ((k * 7 % 11) as f64 / 10.0);
            ped(x, y, 0.0, 0.0, radius, 1.0, 0.0, 1.34)
        })
        .collect();
    let h = 0.004;
    let mut crossed = vec![false; n];
    let mut times = Vec::new();
    for s in 1..=(t_end / h).round() as usize {
        for p in &mut peds {
            p.desired_direction = if p.position.x < Fix128::ZERO {
                -p.position
            } else {
                Vec2Fix::UNIT_X
            };
        }
        m.step(
            &mut peds,
            &walls,
            fx(h),
            NeighborSearch::CellList,
            Some(fx(1.3)),
        )
        .expect("step");
        for (k, p) in peds.iter().enumerate() {
            if !crossed[k] && p.position.x >= Fix128::ZERO {
                crossed[k] = true;
                times.push(s as f64 * h);
            }
        }
        if times.len() == n {
            break;
        }
    }
    times
}

/// Specific flow `J / w` through a door, against experiments.
///
/// Laboratory bottleneck experiments with normal walking report specific
/// flows of about `1.9 (m s)⁻¹`, nearly independent of the width for
/// `0.8–1.2 m` (Seyfried, Passon, Steffen, Boltes, Rupprecht, Klingsch,
/// Transp. Sci. 43, 395 (2009), and the data compiled there). The social
/// force model with these parameters gives lower flows: measured here
/// 0.77 / 1.09 (m s)⁻¹ for 0.8 / 1.2 m with 40 pedestrians, and 0.93 / 1.04 /
/// 1.33 (m s)⁻¹ for 0.8 / 1.0 / 1.2 m with 64 (release, 2026-10-05). The assert is the order of magnitude
/// (within a factor `√10` of 1.9) and a flow that grows with the width; a
/// missing wall, a missing body force or a broken neighbour search changes
/// the flow by more than that or blocks it (0 crossings).
#[test]
#[ignore = "runtime: about 30 s in release (2 widths, 40 pedestrians, up to 15000 Fix128 steps each); run by run_ignored.py"]
fn bottleneck_specific_flow_matches_experiments_in_order_of_magnitude() {
    let lit = 1.9;
    let mut flows = Vec::new();
    for width in [0.8, 1.2] {
        let times = bottleneck_crossings(40, width, 60.0);
        assert!(
            times.len() >= 30,
            "width {width}: only {} crossed",
            times.len()
        );
        // Steady part: from the 10th to the 30th crossing.
        let flow = 20.0 / (times[29] - times[9]);
        let specific = flow / width;
        eprintln!("width {width} m: J = {flow:.3} 1/s, J/w = {specific:.3} 1/(m s)");
        assert!(
            specific >= lit / 10f64.sqrt() && specific <= lit * 10f64.sqrt(),
            "width {width}: J/w = {specific} outside [{}, {}]",
            lit / 10f64.sqrt(),
            lit * 10f64.sqrt()
        );
        flows.push(flow);
    }
    assert!(flows[1] > flows[0], "flow grows with the width: {flows:?}");
}
