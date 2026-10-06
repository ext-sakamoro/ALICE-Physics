//! Independent oracles for `crowd_force::SocialForce`.
//!
//! The existing file (`analytic_crowd_force.rs`) uses the HFV 2000 values
//! (A 2000 N, B 0.08 m, k 1.2e5, κ 2.4e5, m 80, τ 0.5) and checks the force
//! formulas term by term. Here another parameter set is used (A 900 N,
//! B 0.1 m, k 1.0e5, κ 1.5e5, m 70 kg, τ 0.8 s, radii 0.25 / 0.28 m) and the
//! checks go through properties instead of the formula:
//!
//! - Decay law: `ln f(d)` is linear in `d` with slope `−1/B` (three-point fit).
//! - Energy: for `λ = 1`, `κ = 0` the pair force is conservative with
//!   potential `U = A B e^{(r_ij − d)/B} + ½ k g(r_ij − d)²`; a head-on
//!   collision through contact integrated by `step` keeps `KE + U` and
//!   rebounds elastically, with total momentum exactly zero.
//! - Angular momentum: central pair forces have no moment about the pair.
//! - Relaxation of the position, not only the velocity: the discrete sum of
//!   the semi-implicit Euler step and the continuous
//!   `x(t) = v0 ê t + (v(0) − v0 ê) τ (1 − e^{−t/τ})`.
//! - Wall ≡ pedestrian of radius 0 standing at the nearest wall point, with
//!   equal parameters and `λ = 1`.
//! - Rotation by 90° and mirror equivariance of `total_forces` (to rounding:
//!   `Fix128` products round toward −∞, so exact bit symmetry is not promised).
//! - Boundary values `λ = 0` and `λ = 1`, a speed cap with `v0 = 0`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `ln` / `exp` compute references outside the crate.
#![allow(clippy::disallowed_methods)]

use alice_physics::crowd_force::{
    CrowdForceError, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
};
use alice_physics::math::Fix128;
use alice_physics::physics2d::Vec2Fix;

const A: f64 = 900.0;
const B: f64 = 0.1;
const K: f64 = 1.0e5;
const KAPPA: f64 = 1.5e5;
const MASS: f64 = 70.0;
const TAU: f64 = 0.8;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v2(x: f64, y: f64) -> Vec2Fix {
    Vec2Fix::new(fx(x), fx(y))
}

fn to2(v: Vec2Fix) -> [f64; 2] {
    [v.x.to_f64(), v.y.to_f64()]
}

fn params(kappa: f64) -> InteractionParams {
    InteractionParams {
        strength_n: fx(A),
        range_m: fx(B),
        body_stiffness: fx(K),
        sliding_friction: fx(kappa),
    }
}

fn model(lambda: f64, kappa: f64) -> SocialForce {
    SocialForce::new(params(kappa), params(kappa), fx(lambda), fx(3.0)).expect("valid")
}

fn ped(pos: [f64; 2], vel: [f64; 2], r: f64, dir: [f64; 2], v0: f64) -> Pedestrian {
    Pedestrian {
        position: v2(pos[0], pos[1]),
        velocity: v2(vel[0], vel[1]),
        radius_m: fx(r),
        mass_kg: fx(MASS),
        desired_speed_m_s: fx(v0),
        desired_direction: v2(dir[0], dir[1]),
        relaxation_time_s: fx(TAU),
    }
}

#[test]
fn repulsion_decays_exponentially_with_range_b() {
    // oracle: ln |f(d)| = ln A + (r_ij − d)/B ⇒ slope −1/B, zero curvature.
    // Fix128::exp relative error ≤ 1e-6 ⇒ ln error ≤ 1e-6 per point.
    let m = model(1.0, KAPPA);
    let i = ped([0.0, 0.0], [0.0, 0.0], 0.25, [0.0, 1.0], 1.2);
    let lnf = |d: f64| {
        let j = ped([d, 0.0], [0.0, 0.0], 0.28, [0.0, 1.0], 1.2);
        let (fi, _) = m.pair_forces(&i, &j);
        let f = to2(fi);
        assert!(f[0] < 0.0 && f[1] == 0.0, "pushed away from j along −x");
        (-f[0]).ln()
    };
    let (d1, d2, d3) = (0.9, 1.2, 1.5);
    let slope = (lnf(d3) - lnf(d1)) / (d3 - d1);
    assert!(
        (slope + 1.0 / B).abs() < 1e-4,
        "slope {slope} vs {}",
        -1.0 / B
    );
    let curvature = lnf(d1) - 2.0 * lnf(d2) + lnf(d3);
    assert!(
        curvature.abs() < 5e-6,
        "ln f not linear in d: {curvature:e}"
    );
    // Intercept: |f(r_ij)| = A at zero gap.
    let at_touch = lnf(0.53);
    assert!((at_touch - A.ln()).abs() < 2e-6, "|f| at d = r_ij is not A");
}

#[test]
fn central_pair_forces_have_no_moment_about_the_pair() {
    // κ = 0: social + body force act along n_ij, so (x_i − x_j) × f_ij = 0
    // and f_ij + f_ji = 0 (λ = 1), for gaps and overlaps alike.
    let m = model(1.0, 0.0);
    for (dx, dy) in [(0.31, 0.44), (-0.2, 0.35), (0.9, -1.3), (0.5, 0.01)] {
        let i = ped([1.0, -2.0], [0.4, 0.3], 0.25, [1.0, 0.0], 1.3);
        let j = ped([1.0 + dx, -2.0 + dy], [-0.7, 0.2], 0.28, [0.0, -1.0], 1.1);
        let (fi, fj) = m.pair_forces(&i, &j);
        assert_eq!(fi + fj, Vec2Fix::ZERO, "Newton 3 at ({dx}, {dy})");
        let f = to2(fi);
        let moment = (-dx) * f[1] - (-dy) * f[0];
        let mag = (f[0] * f[0] + f[1] * f[1]).sqrt();
        assert!(
            moment.abs() <= 1e-12 * (1.0 + mag),
            "moment {moment:e} at ({dx}, {dy})"
        );
    }
}

/// Pair energy `KE + A B e^{(r_ij − d)/B} + ½ k g²` of two pedestrians on the x axis.
fn pair_energy(p: &[Pedestrian]) -> f64 {
    let ke: f64 = p
        .iter()
        .map(|q| {
            let v = to2(q.velocity);
            0.5 * MASS * (v[0] * v[0] + v[1] * v[1])
        })
        .sum();
    let d = (p[0].position.x - p[1].position.x).abs().to_f64();
    let gap = 0.5 - d;
    ke + A * B * (gap / B).exp() + 0.5 * K * gap.max(0.0).powi(2)
}

#[test]
fn head_on_collision_through_contact_conserves_energy_and_momentum() {
    // λ = 1, κ = 0, ê = 0 and τ = 1e6 s (driving term m(0 − v)/τ, energy
    // loss 2t/τ = 2e-6). KE 157.5 J > U(contact) = A B = 90 J, so the bodies
    // overlap and the body force takes part. Semi-implicit Euler is
    // symplectic: the energy error stays bounded, O(h ω) during contact and
    // back to O(h²)-small after separation.
    let m = SocialForce::new(params(0.0), params(0.0), Fix128::ONE, fx(3.0)).unwrap();
    let mk = |x: f64, v: f64| Pedestrian {
        relaxation_time_s: fx(1e6),
        ..ped([x, 0.0], [v, 0.0], 0.25, [0.0, 0.0], 0.0)
    };
    let mut peds = vec![mk(-0.6, 1.5), mk(0.6, -1.5)];
    let e0 = pair_energy(&peds);
    let h = 5e-5;
    let mut min_d = f64::MAX;
    let mut max_dev = 0.0_f64;
    let mut max_mom = 0.0_f64;
    for _ in 0..20_000 {
        m.step(&mut peds, &[], fx(h), NeighborSearch::Direct, None)
            .unwrap();
        // Mirror-symmetric start: the total momentum stays zero up to the
        // Fix128 rounding (products round toward −∞, so a mirrored product
        // can differ by 2⁻⁶⁴; a few raw units per step at most).
        let p_sum = to2(peds[0].velocity + peds[1].velocity);
        let x_sum = to2(peds[0].position + peds[1].position);
        max_mom = max_mom.max(p_sum[0].abs()).max(x_sum[0].abs());
        assert_eq!(p_sum[1], 0.0);
        let d = (peds[1].position.x - peds[0].position.x).to_f64();
        min_d = min_d.min(d);
        max_dev = max_dev.max((pair_energy(&peds) - e0).abs() / e0);
    }
    assert!(min_d < 0.5, "no contact reached (min d {min_d})");
    assert!(max_mom < 1e-12, "momentum / centre drift {max_mom:e}");
    assert!(
        max_dev < 5e-3,
        "energy deviation {max_dev:e} during the run"
    );
    let e1 = pair_energy(&peds);
    assert!(((e1 - e0) / e0).abs() < 1e-4, "energy {e0} → {e1}");
    // Elastic rebound: KE_end = E₀ − U(d_end) = m v², so each pedestrian
    // separates at √((E₀ − U(d_end))/m). The start at d = 1.2 m stored
    // U = A B e^{−7} = 0.082 J, which the rebound releases (v slightly above
    // 1.5 m/s); a lossy or wrongly signed force would miss this by far more.
    let v = to2(peds[0].velocity);
    assert!(v[0] < 0.0, "pedestrian 0 bounced back");
    let d_end = (peds[1].position.x - peds[0].position.x).to_f64();
    let v_end = ((e0 - A * B * ((0.5 - d_end) / B).exp()) / MASS).sqrt();
    assert!(
        ((-v[0] - v_end) / v_end).abs() < 5e-5,
        "rebound speed {} vs {v_end}",
        -v[0]
    );
    assert!(-v[0] > 1.5, "the stored start potential is released");
}

#[test]
fn position_follows_the_relaxation_integral() {
    // Single pedestrian, desired (0.6, 0.8)·1.1, initial velocity (0.9, −0.5).
    // Discrete: v_k = u + (v_0 − u) q^k, q = 1 − h/τ, x_n = x_0 + h Σ_{k=1}^n v_k.
    // Continuous: x(t) = u t + (v_0 − u) τ (1 − e^{−t/τ}).
    let m = model(0.4, KAPPA);
    let (h, n) = (0.004_f64, 500usize); // t = 2 s = 2.5 τ
    let u = [0.66, 0.88];
    let vi = [0.9, -0.5];
    let x0 = [3.0, -1.0];
    let mut peds = vec![ped(x0, vi, 0.25, [0.6, 0.8], 1.1)];
    for _ in 0..n {
        m.step(&mut peds, &[], fx(h), NeighborSearch::CellList, None)
            .unwrap();
    }
    let q = 1.0 - h / TAU;
    let geom = q * (1.0 - q.powi(n as i32)) / (1.0 - q);
    let t = h * n as f64;
    let pos = to2(peds[0].position);
    for c in 0..2 {
        let discrete = x0[c] + h * (u[c] * n as f64 + (vi[c] - u[c]) * geom);
        assert!(
            (pos[c] - discrete).abs() < 1e-10,
            "discrete x[{c}] {} vs {discrete}",
            pos[c]
        );
        let cont = x0[c] + u[c] * t + (vi[c] - u[c]) * TAU * (1.0 - (-t / TAU).exp());
        // Euler offset ≤ h |v_0 − u| (one step of the transient).
        assert!(
            (pos[c] - cont).abs() <= h * (vi[c] - u[c]).abs() + 1e-10,
            "continuous x[{c}] {} vs {cont}",
            pos[c]
        );
    }
}

#[test]
fn wall_is_a_pedestrian_of_radius_zero_at_the_nearest_point() {
    // Equal pedestrian and wall parameters, λ = 1: f_iW must equal the pair
    // force from a resting radius-0 pedestrian at the foot of the wall.
    let m = model(1.0, KAPPA);
    let wall = WallSegment {
        start: v2(-5.0, 1.0),
        end: v2(5.0, 1.0),
    };
    for (y, vx) in [(1.6, 0.4), (1.2, -0.7), (0.85, 0.3)] {
        let p = ped([0.7, y], [vx, 0.2], 0.28, [1.0, 0.0], 1.3);
        let foot = ped([0.7, 1.0], [0.0, 0.0], 0.0, [0.0, 0.0], 0.0);
        let fw = to2(m.wall_force(&p, &wall));
        let (fp, _) = m.pair_forces(&p, &foot);
        let fp = to2(fp);
        for c in 0..2 {
            assert!(
                (fw[c] - fp[c]).abs() <= 1e-9 * (1.0 + fp[c].abs()),
                "y {y}: wall {fw:?} vs point {fp:?}"
            );
        }
    }
    // Two-sided: mirror positions across the wall get mirrored forces.
    let up = ped([0.0, 1.3], [0.0, 0.0], 0.28, [1.0, 0.0], 1.0);
    let down = ped([0.0, 0.7], [0.0, 0.0], 0.28, [1.0, 0.0], 1.0);
    let (fu, fd) = (m.wall_force(&up, &wall), m.wall_force(&down, &wall));
    assert!(fu.y > Fix128::ZERO && fd.y < Fix128::ZERO);
    assert!((fu.y + fd.y).abs().to_f64() < 1e-9);
}

/// Equal up to Fix128 rounding: products round toward −∞, so an exactly
/// rotated or mirrored input can give results a few raw units apart.
fn close2(label: &str, a: Vec2Fix, b: Vec2Fix) {
    let (a, b) = (to2(a), to2(b));
    for c in 0..2 {
        assert!(
            (a[c] - b[c]).abs() <= 1e-12 * (1.0 + a[c].abs()),
            "{label}: {a:?} vs {b:?}"
        );
    }
}

fn crowd() -> Vec<Pedestrian> {
    vec![
        ped([0.0, 0.0], [0.5, 0.1], 0.25, [1.0, 0.2], 1.3),
        ped([0.45, 0.1], [-0.3, 0.4], 0.28, [-1.0, 0.0], 1.1),
        ped([0.2, 0.6], [0.0, -0.2], 0.25, [0.3, -1.0], 1.2),
        ped([-0.9, 0.4], [0.7, 0.0], 0.27, [0.0, 0.0], 0.0),
        ped([1.7, -0.8], [0.2, 0.9], 0.26, [-0.5, 0.5], 1.4),
        ped([0.5, -0.45], [0.1, 0.1], 0.25, [0.6, 0.8], 1.0),
    ]
}

fn walls() -> Vec<WallSegment> {
    vec![
        WallSegment {
            start: v2(-2.0, -1.0),
            end: v2(2.5, -1.0),
        },
        WallSegment {
            start: v2(0.3, 1.0),
            end: v2(0.3, 1.0),
        },
    ]
}

#[test]
fn total_forces_are_equivariant_under_rotation_by_90_degrees_and_mirror() {
    // (x, y) → (−y, x) and (x, y) → (−x, y) are exact in fixed point; the
    // forces must transform the same way, up to rounding.
    let m = model(0.3, KAPPA);
    let rot = |v: Vec2Fix| Vec2Fix::new(-v.y, v.x);
    let mir = |v: Vec2Fix| Vec2Fix::new(-v.x, v.y);
    let map_all = |f: &dyn Fn(Vec2Fix) -> Vec2Fix| {
        let peds: Vec<Pedestrian> = crowd()
            .into_iter()
            .map(|p| Pedestrian {
                position: f(p.position),
                velocity: f(p.velocity),
                desired_direction: f(p.desired_direction),
                ..p
            })
            .collect();
        let ws: Vec<WallSegment> = walls()
            .into_iter()
            .map(|w| WallSegment {
                start: f(w.start),
                end: f(w.end),
            })
            .collect();
        (peds, ws)
    };
    let mut base = Vec::new();
    m.total_forces(&crowd(), &walls(), NeighborSearch::Direct, &mut base)
        .unwrap();
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        let (p, w) = map_all(&rot);
        let mut out = Vec::new();
        m.total_forces(&p, &w, search, &mut out).unwrap();
        for (i, (a, b)) in base.iter().zip(&out).enumerate() {
            close2(
                &format!("rotation, pedestrian {i}, {search:?}"),
                rot(*a),
                *b,
            );
        }
        // Friction carries the tangent t = (−n_y, n_x), which keeps its
        // handedness under rotation but not under a mirror; the mirror
        // check therefore uses friction-free parameters.
        let mf = model(0.3, 0.0);
        let mut b0 = Vec::new();
        mf.total_forces(&crowd(), &walls(), search, &mut b0)
            .unwrap();
        let (p, w) = map_all(&mir);
        let mut out = Vec::new();
        mf.total_forces(&p, &w, search, &mut out).unwrap();
        for (i, (a, b)) in b0.iter().zip(&out).enumerate() {
            close2(&format!("mirror, pedestrian {i}, {search:?}"), mir(*a), *b);
        }
    }
}

#[test]
fn mirror_with_friction_is_equivariant_within_rounding() {
    // With κ > 0 the mirror flips t and Δv^t together, so f is still the
    // mirror image (t Δv^t is even in the handedness of t).
    let m = model(0.3, KAPPA);
    let mir = |v: Vec2Fix| Vec2Fix::new(-v.x, v.y);
    let peds: Vec<Pedestrian> = crowd()
        .into_iter()
        .map(|p| Pedestrian {
            position: mir(p.position),
            velocity: mir(p.velocity),
            desired_direction: mir(p.desired_direction),
            ..p
        })
        .collect();
    let ws: Vec<WallSegment> = walls()
        .into_iter()
        .map(|w| WallSegment {
            start: mir(w.start),
            end: mir(w.end),
        })
        .collect();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    m.total_forces(&crowd(), &walls(), NeighborSearch::Direct, &mut a)
        .unwrap();
    m.total_forces(&peds, &ws, NeighborSearch::Direct, &mut b)
        .unwrap();
    for (i, (fa, fb)) in a.iter().zip(&b).enumerate() {
        let (x, y) = (to2(mir(*fa)), to2(*fb));
        for c in 0..2 {
            assert!(
                (x[c] - y[c]).abs() <= 1e-9 * (1.0 + x[c].abs()),
                "pedestrian {i}: {x:?} vs {y:?}"
            );
        }
    }
}

#[test]
fn zero_anisotropy_ignores_a_neighbour_straight_behind() {
    // λ = 0 (allowed, the closed end of [0, 1]): w = (1 + cos φ)/2, so a
    // neighbour straight behind (cos φ = −1) exerts no social force on i,
    // while i still pushes the neighbour (i is straight ahead of it).
    let m = model(0.0, KAPPA);
    let i = ped([0.0, 0.0], [0.0, 0.0], 0.25, [1.0, 0.0], 1.2);
    let behind = ped([-0.9, 0.0], [0.0, 0.0], 0.28, [1.0, 0.0], 1.2);
    let (fi, fj) = m.pair_forces(&i, &behind);
    // Zero up to rounding of n_ij = diff/d (|cos φ| = 1 − O(2⁻⁶⁴)).
    assert!(
        fi.x.abs().to_f64() < 1e-15 && fi.y == Fix128::ZERO,
        "{fi:?}"
    );
    let expected = A * ((0.53 - 0.9) / B).exp();
    assert!(((-fj.x.to_f64() - expected) / expected).abs() < 2e-6);
    // Boundary values accepted; one raw unit outside rejected.
    assert!(SocialForce::new(params(KAPPA), params(KAPPA), Fix128::ONE, fx(3.0)).is_ok());
    assert_eq!(
        SocialForce::new(
            params(KAPPA),
            params(KAPPA),
            Fix128::from_raw(1, 1),
            fx(3.0)
        )
        .unwrap_err(),
        CrowdForceError::AnisotropyOutOfRange
    );
}

#[test]
fn speed_cap_with_zero_desired_speed_holds_the_pedestrian_still() {
    // Documented cap |v| ≤ c v0 after the velocity update: with v0 = 0 a
    // pedestrian pushed by a neighbour keeps v = 0 and does not move.
    let m = model(1.0, KAPPA);
    let mut peds = vec![
        ped([0.0, 0.0], [0.0, 0.0], 0.25, [0.0, 0.0], 0.0),
        ped([0.4, 0.0], [0.0, 0.0], 0.28, [0.0, 0.0], 1.0),
    ];
    let start = peds[0].position;
    for _ in 0..10 {
        m.step(
            &mut peds,
            &[],
            fx(0.01),
            NeighborSearch::Direct,
            Some(fx(1.3)),
        )
        .unwrap();
    }
    assert_eq!(peds[0].velocity, Vec2Fix::ZERO);
    assert_eq!(peds[0].position, start);
    assert!(peds[1].position.x > fx(0.4), "the other one is pushed away");
}
