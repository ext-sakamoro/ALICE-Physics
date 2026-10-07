//! Analytic oracles for the gyroscopic term `ω × Iω` of free 3D rigid bodies.
//!
//! Euler's equations for a torque-free body in its principal frame,
//!
//! ```text
//! I₁ ω̇₁ = (I₂ − I₃) ω₂ ω₃
//! I₂ ω̇₂ = (I₃ − I₁) ω₃ ω₁
//! I₃ ω̇₃ = (I₁ − I₂) ω₁ ω₂
//! ```
//!
//! have closed-form consequences that do not depend on how the solver is
//! written, and every oracle below is one of them:
//!
//! - (a) the world angular momentum `L = R I_b Rᵀ ω` is constant;
//! - (b) for a symmetric top (`I₁ = I₂`) the body-frame `ω` precesses about the
//!   symmetry axis at `Ω = (I₃ − I₁) / I₁ · ω₃`;
//! - (c) rotation about the intermediate axis is unstable (the body flips),
//!   rotation about the major and the minor axis is stable;
//! - (d) for isotropic inertia `ω × Iω = 0`, so the term must change nothing,
//!   checked against a golden recorded before the term existed;
//! - (e) the rotational kinetic energy `½ ωᵀ I ω` is constant. Both
//!   backends integrate the free rotation by the second-order symplectic
//!   splitting of Dullweber, Leimkuhler and McLachlan (1997): its energy error
//!   is of order `h²` and does not drift (the bound is derived from the
//!   splitting's leading error term);
//! - (f) degenerate inputs (zero `ω`, zero / infinite inertia, a huge `ω`) do
//!   not panic and give the result stated on each test.
//!
//! Expected values are computed here from the formulas above, never by calling
//! the solver. Every scene uses the production entry `PhysicsWorld::step`
//! (and `step_parallel` under the `parallel` feature) with zero gravity and no
//! damping, on both `SolverBackend::Xpbd` and `SolverBackend::Tgs`.

#![cfg(feature = "std")]

use alice_physics::det_math::atan2_64;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

const FRAME_DT: f64 = 1.0 / 60.0;
const SUBSTEPS: usize = 8;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn to_f(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cfg(backend: SolverBackend) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps: SUBSTEPS,
        solver_backend: backend,
        ..PhysicsConfig::default()
    }
}

const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];

/// How a scene is advanced by one frame.
#[derive(Clone, Copy, Debug)]
enum Entry {
    Step,
    #[cfg(feature = "parallel")]
    StepParallel,
}

fn entries() -> Vec<(SolverBackend, Entry)> {
    #[allow(unused_mut)]
    let mut v: Vec<(SolverBackend, Entry)> = BACKENDS.iter().map(|&b| (b, Entry::Step)).collect();
    #[cfg(feature = "parallel")]
    v.push((SolverBackend::Xpbd, Entry::StepParallel));
    v
}

fn advance(w: &mut PhysicsWorld, e: Entry) {
    match e {
        Entry::Step => w.step(fx(FRAME_DT)),
        #[cfg(feature = "parallel")]
        Entry::StepParallel => w.step_parallel(fx(FRAME_DT)),
    }
}

/// A free body with principal moments `inertia` (body frame), world angular
/// velocity `omega` and no linear motion.
fn free_body(inertia: [f64; 3], omega: [f64; 3]) -> RigidBody {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = v3(1.0 / inertia[0], 1.0 / inertia[1], 1.0 / inertia[2]);
    b.angular_velocity = v3(omega[0], omega[1], omega[2]);
    b
}

fn one_body_world(backend: SolverBackend, body: RigidBody) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(cfg(backend));
    w.add_body(body);
    w
}

/// Body-frame angular velocity `Rᵀ ω`.
fn body_omega(b: &RigidBody) -> [f64; 3] {
    to_f(b.rotation.conjugate().rotate_vec(b.angular_velocity))
}

/// World angular momentum `R I_b Rᵀ ω`, evaluated in f64 from the state.
fn world_l(b: &RigidBody, inertia: [f64; 3]) -> [f64; 3] {
    let wb = body_omega(b);
    let lb = v3(inertia[0] * wb[0], inertia[1] * wb[1], inertia[2] * wb[2]);
    to_f(b.rotation.rotate_vec(lb))
}

/// Rotational kinetic energy `½ Σ I_i ω_i²` in the body frame.
fn energy(b: &RigidBody, inertia: [f64; 3]) -> f64 {
    let wb = body_omega(b);
    0.5 * (inertia[0] * wb[0] * wb[0] + inertia[1] * wb[1] * wb[1] + inertia[2] * wb[2] * wb[2])
}

// ============================================================================
// (a) conservation of world angular momentum
// ============================================================================

/// oracle: torque-free ⇒ `dL/dt = 0` exactly. Every sub-flow of the
/// splitting turns the body by `+θ` and `L_b` by `−θ` about the same axis, so
/// `R L_b` is unchanged up to rounding: the only error left is the f64
/// evaluation of `L` here (a few ulp, `~1e-15`) plus fixed-point rounding
/// (`2⁻⁶⁴` per operation, ~10⁵ operations); the bound used is `1e-12`, on
/// both backends (XPBD keeps the split's end velocity instead of
/// re-deriving it from the rotation change). A solver without
/// the term keeps `ω` constant in the world while `R` turns, so `L` follows
/// `R I_b Rᵀ` and changes by order one over the same 2 s.
#[test]
fn a_world_angular_momentum_is_conserved_for_asymmetric_body() {
    let inertia = [1.0, 2.0, 3.0];
    let omega = [1.0, 0.7, -0.4];
    let frames = 120;
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let mut w = one_body_world(backend, free_body(inertia, omega));
        let l0 = world_l(&w.bodies[0], inertia);
        let mut worst = 0.0f64;
        for _ in 0..frames {
            advance(&mut w, e);
            let l = world_l(&w.bodies[0], inertia);
            worst = worst.max(norm(sub(l, l0)) / norm(l0));
        }
        eprintln!("(a) {backend:?} {e:?}: max |L - L0| / |L0| = {worst:.3e}");
        if worst >= 1e-12 {
            bad.push(format!(
                "(a) {backend:?} {e:?}: world angular momentum drifted by {worst:.3e} of |L0| \
                 (closed form: constant)"
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ============================================================================
// (b) symmetric top precession rate
// ============================================================================

/// oracle: `I₁ = I₂ = 1`, `I₃ = 2`, `ω_b(0) = (0.2, 0, 1)`. Euler gives
/// `ω̇₁ = −Ω ω₂`, `ω̇₂ = Ω ω₁` with `Ω = (I₃ − I₁)/I₁ · ω₃ = 1 rad/s`, so the
/// body-frame `(ω₁, ω₂)` turns counter-clockwise by `Ω t = 2 rad` in 2 s while
/// `ω₃` stays 1, for every `substeps`. Phase error of a first-order step is
/// `O(h)` per unit time in the rate, `≤ T · h · Ω² ≈ 0.008 rad` at the
/// coarsest `h = 1/240`; the bound used is 0.03 rad. Without
/// the term `R` turns about the constant `ω` itself, so `Rᵀ ω` and the angle
/// stay at 0.
#[test]
fn b_symmetric_top_precesses_at_closed_form_rate() {
    let inertia = [1.0, 1.0, 2.0];
    let omega = [0.2, 0.0, 1.0];
    let big_omega = (inertia[2] - inertia[0]) / inertia[0] * omega[2];
    let frames = 120;
    let t = frames as f64 * FRAME_DT;
    let mut bad = Vec::new();
    // precision sweep: the closed form does not depend on `substeps`
    for (substeps, (backend, e)) in [4usize, 8, 16]
        .into_iter()
        .flat_map(|n| entries().into_iter().map(move |x| (n, x)))
    {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            substeps,
            ..cfg(backend)
        });
        w.add_body(free_body(inertia, omega));
        let mut angle = 0.0f64;
        let mut prev = body_omega(&w.bodies[0]);
        for _ in 0..frames {
            advance(&mut w, e);
            let wb = body_omega(&w.bodies[0]);
            // unwrap one frame's increment (|ΔΦ| ≪ π per frame)
            let d = atan2_64(wb[1], wb[0]) - atan2_64(prev[1], prev[0]);
            let d = (d + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            angle += d;
            prev = wb;
        }
        let expected = big_omega * t;
        let wb = body_omega(&w.bodies[0]);
        eprintln!(
            "(b) {backend:?} {e:?} substeps {substeps}: angle {angle:.6} expected {expected:.6}, ω3 {:.6}",
            wb[2]
        );
        if (angle - expected).abs() >= 0.03 {
            bad.push(format!(
                "(b) {backend:?} {e:?} substeps {substeps}: precession angle {angle:.6} rad, closed form {expected:.6} rad"
            ));
        }
        if (wb[2] - omega[2]).abs() >= 1e-3 {
            bad.push(format!(
                "(b) {backend:?} {e:?}: ω3 must be constant for I1 = I2, got {:.6}",
                wb[2]
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Precession angle error (rad) of the symmetric top of
/// `b_symmetric_top_precesses_at_closed_form_rate` after 2 s.
fn precession_error(backend: SolverBackend, e: Entry, substeps: usize) -> f64 {
    let inertia = [1.0, 1.0, 2.0];
    let omega = [0.2, 0.0, 1.0];
    let big_omega = (inertia[2] - inertia[0]) / inertia[0] * omega[2];
    let frames = 120;
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps,
        ..cfg(backend)
    });
    w.add_body(free_body(inertia, omega));
    let mut angle = 0.0f64;
    let mut prev = body_omega(&w.bodies[0]);
    for _ in 0..frames {
        advance(&mut w, e);
        let wb = body_omega(&w.bodies[0]);
        let d = atan2_64(wb[1], wb[0]) - atan2_64(prev[1], prev[0]);
        let d = (d + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        angle += d;
        prev = wb;
    }
    angle - big_omega * frames as f64 * FRAME_DT
}

/// oracle: the Strang composition is second order, so halving `h` divides
/// the precession error by 4 (a first-order composition such as
/// `R₁(h) R₂(h) R₃(h)` divides it by 2). Accepted ratio band `[3, 5]` for
/// `substeps` 4 → 8 → 16; the errors (`~1e-5 … 1e-6` rad) are far above the
/// rounding floor (`~1e-12` rad), so the ratio is not noise.
#[test]
fn b_precession_error_is_second_order() {
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let errs: Vec<f64> = [4usize, 8, 16]
            .iter()
            .map(|&n| precession_error(backend, e, n))
            .collect();
        eprintln!("(b) {backend:?} {e:?} precession error at substeps 4/8/16: {errs:?}");
        for k in 0..2 {
            let ratio = errs[k] / errs[k + 1];
            if !(3.0..=5.0).contains(&ratio) {
                bad.push(format!(
                    "(b) {backend:?} {e:?}: precession error ratio {ratio:.3} at substeps {} → {} \
                     (second order: 4), errors {errs:?}",
                    4 << k,
                    8 << k
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ============================================================================
// (c) Dzhanibekov effect
// ============================================================================

/// Run `frames` frames and return (min, max) of each body-frame component.
fn omega_range(
    backend: SolverBackend,
    e: Entry,
    inertia: [f64; 3],
    omega: [f64; 3],
    frames: usize,
) -> ([f64; 3], [f64; 3]) {
    let mut w = one_body_world(backend, free_body(inertia, omega));
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for _ in 0..frames {
        advance(&mut w, e);
        let wb = body_omega(&w.bodies[0]);
        for k in 0..3 {
            lo[k] = lo[k].min(wb[k]);
            hi[k] = hi[k].max(wb[k]);
        }
    }
    (lo, hi)
}

/// oracle: `I = (1, 2, 3)`. Linearising Euler about `ω = (0, s, 0)` gives
/// perturbations growing as `e^{λt}` with
/// `λ = s · √((I₂ − I₁)(I₃ − I₂) / (I₁ I₃)) = 2 / √3 ≈ 1.155 /s` for `s = 2`;
/// from `ε = 0.01` the perturbation reaches order `s` after
/// `ln(s/ε)/λ ≈ 4.6 s`, so within 8 s `ω₂` changes sign (the flip) and, `E`
/// and `|L|` being conserved, its magnitude returns close to `s`. Without
/// the term `Rᵀ ω` is constant and never flips.
#[test]
fn c_intermediate_axis_flips() {
    let inertia: [f64; 3] = [1.0, 2.0, 3.0];
    let s: f64 = 2.0;
    let frames = 480;
    let lambda = s
        * ((inertia[1] - inertia[0]) * (inertia[2] - inertia[1]) / (inertia[0] * inertia[2]))
            .sqrt();
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let (lo, _) = omega_range(backend, e, inertia, [0.01, s, 0.01], frames);
        eprintln!("(c) {backend:?} {e:?}: intermediate min ω2 = {:.4}", lo[1]);
        if lo[1] >= -0.9 * s {
            bad.push(format!(
                "(c) {backend:?} {e:?}: rotation about the intermediate axis did not flip \
                 within 8 s (min body ω2 = {:.4}, closed-form e-folding time {:.2} s)",
                lo[1],
                1.0 / lambda
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// oracle: about the minor (`I₁`) and the major (`I₃`) axis the linearised
/// equations are a harmonic oscillator, so a perturbation `ε = 0.01` stays of
/// order `ε` (amplitude ratio `≤ √(κ)` with `κ ≤ 3` between the two
/// transverse components): bounded by `10 ε` here, and the spin component
/// stays within 5 % of `s` over 8 s.
#[test]
fn c_major_and_minor_axes_are_stable() {
    let inertia = [1.0, 2.0, 3.0];
    let s = 2.0;
    let eps = 0.01;
    let frames = 480;
    for (backend, e) in entries() {
        for axis in [0usize, 2] {
            let mut omega = [eps; 3];
            omega[axis] = s;
            let (lo, hi) = omega_range(backend, e, inertia, omega, frames);
            eprintln!("(c) {backend:?} {e:?} axis {axis}: lo {lo:?} hi {hi:?}");
            assert!(
                lo[axis] > 0.95 * s,
                "(c) {backend:?} {e:?}: spin about stable axis {axis} fell to {:.4}",
                lo[axis]
            );
            for k in (0..3).filter(|&k| k != axis) {
                assert!(
                    lo[k].abs().max(hi[k].abs()) < 10.0 * eps,
                    "(c) {backend:?} {e:?}: perturbation on axis {k} grew to {:.4} \
                     while spinning about stable axis {axis}",
                    lo[k].abs().max(hi[k].abs())
                );
            }
        }
    }
}

// ============================================================================
// (d) isotropic inertia: bit-identical to the solver without the term
// ============================================================================

fn mix(h: &mut u64, f: Fix128) {
    for b in f.hi.to_le_bytes().iter().chain(f.lo.to_le_bytes().iter()) {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn hash_bodies(w: &PhysicsWorld) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in &w.bodies {
        for v in [b.position, b.velocity, b.angular_velocity] {
            mix(&mut h, v.x);
            mix(&mut h, v.y);
            mix(&mut h, v.z);
        }
        for f in [b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w] {
            mix(&mut h, f);
        }
    }
    h
}

/// A scene whose bodies all have `ω × Iω = 0` in exact arithmetic, or no
/// gyroscopic response at all: unit-sphere inertia, a scaled isotropic
/// inertia, colliding isotropic spheres under gravity, an infinite-inertia
/// dynamic body (`inv_inertia = 0`), a body with one infinite principal
/// moment, a static and a kinematic body.
fn isotropic_scene(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    let mut a = RigidBody::new(v3(0.0, 5.0, 0.0), Fix128::ONE);
    a.angular_velocity = v3(1.0, 2.0, 3.0);
    w.add_body(a);
    let mut b = RigidBody::new(v3(4.0, 5.0, 0.0), fx(2.0));
    b.inv_inertia = v3(0.25, 0.25, 0.25);
    b.angular_velocity = v3(-2.0, 1.0, 0.5);
    w.add_body(b);
    let mut c = RigidBody::new(v3(-4.0, 1.0, 0.0), Fix128::ONE);
    c.angular_velocity = v3(0.3, -0.7, 1.1);
    c.velocity = v3(3.0, 0.0, 0.0);
    w.add_body_with_radius(c, fx(0.5));
    let mut d = RigidBody::new(v3(-2.0, 1.0, 0.0), Fix128::ONE);
    d.angular_velocity = v3(-0.4, 0.2, 0.9);
    d.velocity = v3(-3.0, 0.0, 0.0);
    w.add_body_with_radius(d, fx(0.5));
    let mut e = RigidBody::new(v3(8.0, 5.0, 0.0), Fix128::ONE);
    e.inv_inertia = Vec3Fix::ZERO;
    e.angular_velocity = v3(1.0, -1.0, 0.5);
    w.add_body(e);
    let mut f = RigidBody::new(v3(12.0, 5.0, 0.0), Fix128::ONE);
    f.inv_inertia = v3(1.0, 0.0, 0.5);
    f.angular_velocity = v3(0.6, 0.8, -0.3);
    w.add_body(f);
    w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
    let mut k = RigidBody::new_kinematic(v3(20.0, 0.0, 0.0));
    k.angular_velocity = v3(0.5, 0.5, 0.5);
    w.add_body(k);
    w
}

/// Golden hashes of `isotropic_scene` after 60 frames, recorded with the
/// solver without the gyroscopic term (the XPBD value re-recorded after the
/// pre-solve restitution and static friction of the contact response, which
/// change the colliding pair, and again for the exact-logarithm velocity
/// re-derivation, which changes every spinning XPBD body; the term itself
/// leaves every body of this scene on its previous path).
const GOLDEN_ISOTROPIC_XPBD: u64 = 0xa78f_4abf_b26c_12be;
// The TGS value was re-recorded when the TGS orientation integrator became
// the exact exponential map, and again when an isotropic inverse inertia
// became the plain product `c·τ` (no rotation round trip), which changes the
// spinning colliding spheres of the TGS contact solve in the last bits.
const GOLDEN_ISOTROPIC_TGS: u64 = 0x645f_6a73_cb85_0486;

/// oracle: `I_b = s·1 ⇒ ω × (s ω) = 0`, so adding the term must leave every
/// bit of an isotropic scene unchanged (likewise bodies the term does not
/// apply to). Compared against goldens recorded before the term existed.
#[test]
fn d_isotropic_and_exempt_bodies_are_bit_identical_to_before() {
    for (backend, golden) in [
        (SolverBackend::Xpbd, GOLDEN_ISOTROPIC_XPBD),
        (SolverBackend::Tgs, GOLDEN_ISOTROPIC_TGS),
    ] {
        let mut w = isotropic_scene(backend);
        for _ in 0..60 {
            w.step(fx(FRAME_DT));
        }
        let got = hash_bodies(&w);
        eprintln!("(d) {backend:?}: hash {got:#018x}");
        assert_eq!(got, golden, "(d) {backend:?}: isotropic scene changed");
    }
}

#[cfg(feature = "parallel")]
const GOLDEN_ISOTROPIC_PARALLEL: u64 = 0xa78f_4abf_b26c_12be;

#[cfg(feature = "parallel")]
#[test]
fn d_isotropic_scene_is_bit_identical_under_step_parallel() {
    let mut w = isotropic_scene(SolverBackend::Xpbd);
    for _ in 0..60 {
        w.step_parallel(fx(FRAME_DT));
    }
    let got = hash_bodies(&w);
    eprintln!("(d) step_parallel: hash {got:#018x}");
    assert_eq!(got, GOLDEN_ISOTROPIC_PARALLEL);
}

// ============================================================================
// (e) kinetic energy never grows
// ============================================================================

/// Largest `|E − E0| / E0` over `frames` frames.
fn worst_energy_error(
    backend: SolverBackend,
    e: Entry,
    inertia: [f64; 3],
    omega: [f64; 3],
    substeps: usize,
    frames: usize,
) -> f64 {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps,
        ..cfg(backend)
    });
    w.add_body(free_body(inertia, omega));
    let e0 = energy(&w.bodies[0], inertia);
    let mut worst = 0.0f64;
    for _ in 0..frames {
        advance(&mut w, e);
        worst = worst.max((energy(&w.bodies[0], inertia) - e0).abs() / e0);
    }
    worst
}

/// oracle: a symplectic second-order method conserves a modified
/// energy `H̃ = H + h² K + O(h⁴)`, so `|E(t) − E0| ≤ 2 h² max|K|`: the error
/// is bounded (no drift) and divides by 4 when `h` halves. Asymmetric body
/// `I = (1, 2, 3)`, `ω = (6, −5, 4)`, 2 s: the ratio of the worst energy error
/// at `substeps` 4 → 8 → 16 must lie in `[3, 5]`. A dissipative or first-order
/// step drifts by `O(h)` and gives a ratio near 2.
#[test]
fn e_energy_error_is_second_order() {
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let errs: Vec<f64> = [4usize, 8, 16]
            .iter()
            .map(|&n| worst_energy_error(backend, e, [1.0, 2.0, 3.0], [6.0, -5.0, 4.0], n, 120))
            .collect();
        eprintln!("(e) {backend:?} {e:?} worst |ΔE|/E0 at substeps 4/8/16: {errs:?}");
        for k in 0..2 {
            let ratio = errs[k] / errs[k + 1];
            if !(3.0..=5.0).contains(&ratio) {
                bad.push(format!(
                    "(e) {backend:?} {e:?}: energy error ratio {ratio:.3} at substeps {} → {} \
                     (second order: 4), errors {errs:?}",
                    4 << k,
                    8 << k
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// oracle: thin rod `I = (0.02, 1, 1)`, `ω_b = (30, 40, 0)`
/// (`|ω| = 50`), 10 s at `h = 1/480`. The rod is a symmetric top
/// (`I₂ = I₃ = I⊥`): `H₁` commutes with `H₂ + H₃`, so the outer half-steps
/// `R₁(h/2)` add only `O(h⁴)`, and the leading energy error comes from the
/// inner composition `R₂(h/2) R₃(h) R₂(h/2)`. By the
/// Baker–Campbell–Hausdorff formula its modified Hamiltonian is
/// `H₂ + H₃ + h² (c₁ {H₂,{H₂,H₃}} + c₂ {H₃,{H₃,H₂}})` with
/// `|c₁| + |c₂| = 1/24 + 1/12 = 1/8`. With the Lie–Poisson bracket
/// `{H₂,{H₂,H₃}} = ±L₂² (L₁² − L₃²) / I⊥³` (and `2 ↔ 3`),
/// `|L₂² (L₁² − L₃²)| ≤ |L⊥|⁴/4 + L₁² |L⊥|²` where `L₁` and `|L⊥|` are
/// invariants of the symmetric top. Hence
///
/// ```text
/// |ΔE| / E0 ≤ (h²/4) (|L⊥|⁴/4 + L₁² |L⊥|²) / (I⊥³ E0) ≈ 8.6e-4
/// ```
///
/// The implicit one-Newton-step term (the previous scheme) loses about 99 %
/// of `E` on this scene.
#[test]
fn e_thin_rod_energy_stays_in_second_order_band() {
    let inertia: [f64; 3] = [0.02, 1.0, 1.0];
    let omega_b: [f64; 3] = [30.0, 40.0, 0.0];
    let h = FRAME_DT / SUBSTEPS as f64;
    let l1 = inertia[0] * omega_b[0];
    let (l2, l3) = (inertia[1] * omega_b[1], inertia[2] * omega_b[2]);
    let lp2 = l2 * l2 + l3 * l3;
    let e0 = 0.5
        * (inertia[0] * omega_b[0] * omega_b[0]
            + inertia[1] * omega_b[1] * omega_b[1]
            + inertia[2] * omega_b[2] * omega_b[2]);
    let i_perp: f64 = inertia[1];
    let bound = h * h / 4.0 * (lp2 * lp2 / 4.0 + l1 * l1 * lp2) / (i_perp * i_perp * i_perp * e0);
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let worst = worst_energy_error(backend, e, inertia, omega_b, SUBSTEPS, 600);
        eprintln!("(e) {backend:?} {e:?} thin rod: worst |ΔE|/E0 = {worst:.3e}, bound {bound:.3e}");
        if worst > bound {
            bad.push(format!(
                "(e) {backend:?} {e:?} thin rod: energy error {worst:.3e} exceeds the \
                 second-order bound {bound:.3e}"
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Rotation vector (axis · angle, world frame) of the unit quaternion `d`,
/// with the sign of `w` folded so the angle is at most π.
fn rotation_vector(d: QuatFix) -> [f64; 3] {
    let (x, y, z, w) = (d.x.to_f64(), d.y.to_f64(), d.z.to_f64(), d.w.to_f64());
    let (x, y, z, w) = if w < 0.0 {
        (-x, -y, -z, -w)
    } else {
        (x, y, z, w)
    };
    let s = (x * x + y * y + z * z).sqrt();
    if s == 0.0 {
        return [0.0; 3];
    }
    let angle = 2.0 * atan2_64(s, w);
    [x / s * angle, y / s * angle, z / s * angle]
}

/// oracle (contact): kinematics `q̇ = ½ ω q` holds whatever the forces,
/// so the orientation change over the run must equal `∫ ω dt`. An
/// anisotropic ball spinning about its (principal, vertical) axis is dropped
/// with a horizontal velocity onto a static sphere; friction changes `ω`
/// inside the solve, and that change must reach the orientation on top of
/// the split's free rotation. Compared: the summed per-frame rotation vectors
/// of `q_{n+1} q_n⁻¹` against the trapezoid sum of `ω dt`. Error budget: the
/// trapezoid rule is exact for `ω` linear in time; a jump `Δω` inside a frame
/// (contact onset) costs at most `½ dt |Δω|`, and composing non-parallel
/// rotations costs `O(dt² |ω|²)` per frame; with `|Δω| ≤ 4 rad/s` and
/// `|ω| ≤ 4 rad/s` the sum stays below `0.05 + 60 · dt² · 16 ≈ 0.08` rad.
/// An orientation that ignored the solve's velocity altogether would miss the
/// rolling about `z` (1.4 rad here). Dropping only the within-sub-step part of
/// the change costs `h |Δω_total|` (about `3e-3` rad here, below this
/// oracle's resolution); that part is pinned by the unit test
/// `advance_split_applies_the_solve_change_on_top_of_the_free_rotation`.
#[test]
fn f_tgs_contact_change_of_omega_reaches_the_orientation() {
    // XPBD's sphere contacts do not turn the body (measured: no rolling), so
    // the XPBD counterpart is the jointed scene below.
    for (backend, e) in entries()
        .into_iter()
        .filter(|(b, _)| *b == SolverBackend::Tgs)
    {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: v3(0.0, -10.0, 0.0),
            substeps: SUBSTEPS,
            solver_backend: backend,
            ..PhysicsConfig::default()
        });
        let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        w.set_body_collision_radius(ground, fx(10.0));
        let mut ball = free_body([0.4, 0.5, 0.6], [0.0, 1.0, 0.0]);
        ball.position = v3(0.0, 11.0, 0.0);
        ball.velocity = v3(3.0, 0.0, 0.0);
        let ball = w.add_body(ball);
        w.set_body_collision_radius(ball, Fix128::ONE);
        let mut from_q = [0.0f64; 3];
        let mut from_w = [0.0f64; 3];
        let mut q = w.bodies[ball].rotation;
        let mut om = to_f(w.bodies[ball].angular_velocity);
        for _ in 0..60 {
            advance(&mut w, e);
            let q2 = w.bodies[ball].rotation;
            let om2 = to_f(w.bodies[ball].angular_velocity);
            let rv = rotation_vector(q2.mul(q.conjugate()));
            for k in 0..3 {
                from_q[k] += rv[k];
                from_w[k] += 0.5 * (om[k] + om2[k]) * FRAME_DT;
            }
            q = q2;
            om = om2;
        }
        let err = norm(sub(from_q, from_w));
        eprintln!("(f) {backend:?} {e:?} contact: Σ rotation {from_q:?}, ∫ω dt {from_w:?}, |diff| {err:.3e}");
        assert!(
            from_w[2].abs() > 1.0,
            "(f) {backend:?} {e:?} contact: the scene must roll (∫ω_z dt = {:.3})",
            from_w[2]
        );
        assert!(
        err < 0.08,
        "(f) {backend:?} {e:?} contact: orientation change {from_q:?} differs from ∫ω dt {from_w:?} by {err:.3e}"
    );
    }
}

/// oracle (joint, XPBD): the same kinematic identity `Δq = ∫ ω dt` for a
/// pendulum whose position solve turns the body every substep. An
/// anisotropic bob (`I = (0.4, 0.5, 0.6)`), 2 m from a static anchor on a ball
/// joint, released horizontally with a spin of 1 rad/s about its arm. XPBD
/// keeps the split's end velocity and adds the rotation the joint applied
/// beyond the split's end orientation; dropping that addition leaves `ω`
/// without the swing while `q` swings. Error budget as in the contact oracle
/// (trapezoid exact for linear `ω`, `O(dt² |ω|²)` per frame from composing
/// non-parallel rotations): `|ω| ≤ 4 rad/s` over 60 frames gives
/// `60 · ½ dt² · 16 ≈ 0.13` rad; the swing itself is `> 1` rad.
#[test]
fn f_joint_change_of_omega_reaches_the_angular_velocity() {
    use alice_physics::joint::{BallJoint, Joint};
    for (backend, e) in entries()
        .into_iter()
        .filter(|(b, _)| *b == SolverBackend::Xpbd)
    {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: v3(0.0, -10.0, 0.0),
            substeps: SUBSTEPS,
            solver_backend: backend,
            ..PhysicsConfig::default()
        });
        let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mut bob = free_body([0.4, 0.5, 0.6], [1.0, 0.0, 0.0]);
        bob.position = v3(2.0, 0.0, 0.0);
        let bob = w.add_body(bob);
        w.add_joint(Joint::Ball(BallJoint::new(
            anchor,
            bob,
            Vec3Fix::ZERO,
            v3(-2.0, 0.0, 0.0),
        )));
        let mut from_q = [0.0f64; 3];
        let mut from_w = [0.0f64; 3];
        let mut q = w.bodies[bob].rotation;
        let mut om = to_f(w.bodies[bob].angular_velocity);
        for _ in 0..60 {
            advance(&mut w, e);
            let q2 = w.bodies[bob].rotation;
            let om2 = to_f(w.bodies[bob].angular_velocity);
            let rv = rotation_vector(q2.mul(q.conjugate()));
            for k in 0..3 {
                from_q[k] += rv[k];
                from_w[k] += 0.5 * (om[k] + om2[k]) * FRAME_DT;
            }
            q = q2;
            om = om2;
        }
        let err = norm(sub(from_q, from_w));
        eprintln!("(f) {backend:?} {e:?} joint: Σ rotation {from_q:?}, ∫ω dt {from_w:?}, |diff| {err:.3e}");
        assert!(
            from_q[2].abs() > 1.0,
            "(f) {backend:?} {e:?} joint: the bob must swing (Σ rotation_z = {:.3})",
            from_q[2]
        );
        assert!(
            err < 0.13,
            "(f) {backend:?} {e:?} joint: orientation change {from_q:?} differs from ∫ω dt {from_w:?} by {err:.3e}"
        );
    }
}

// ============================================================================
// (f) degenerate inputs
// ============================================================================

/// oracle: `ω = 0 ⇒ ω × Iω = 0`, so an asymmetric body at rest stays at
/// rest, like an isotropic one. Compared with the closed form to rounding,
/// not bit for bit: XPBD derives the velocity of a body without gyroscopic
/// response from `q·q_prev⁻¹` (scaled by `≈ 2/h`), which turns the rounding
/// of `q` into a residue of about `10³` ulp (`~5e-17 rad/s`) even for an
/// isotropic body at rest, and an asymmetric body then carries that residue
/// through the splitting instead. Bound: `|ω| ≤ 1e-12 rad/s` and every component of `q`
/// within `1e-12` of its initial value after 10 frames.
#[test]
fn f_zero_omega_stays_at_rest() {
    for (backend, e) in entries() {
        let q = QuatFix::from_axis_angle(v3(0.0, 0.6, 0.8), fx(0.7));
        for inertia in [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]] {
            let mut body = free_body(inertia, [0.0; 3]);
            body.rotation = q;
            let mut w = one_body_world(backend, body);
            for _ in 0..10 {
                advance(&mut w, e);
            }
            let b = &w.bodies[0];
            let om = norm(to_f(b.angular_velocity));
            let dq = [
                b.rotation.x.to_f64() - q.x.to_f64(),
                b.rotation.y.to_f64() - q.y.to_f64(),
                b.rotation.z.to_f64() - q.z.to_f64(),
                b.rotation.w.to_f64() - q.w.to_f64(),
            ];
            let worst = dq.iter().fold(0.0f64, |m, d| m.max(d.abs()));
            assert!(
                om <= 1e-12 && worst <= 1e-12,
                "{backend:?} {e:?} I={inertia:?}: |ω| = {om:.3e}, max |Δq| = {worst:.3e}"
            );
        }
    }
}

/// Expected: a static body (infinite mass and inertia) is never moved and a
/// body whose `inv_inertia` is zero keeps rotating at its world `ω` (an
/// infinite moment has no gyroscopic response); covered bit-for-bit by
/// `d_isotropic_and_exempt_bodies_are_bit_identical_to_before`. Here: a huge
/// `|ω| ≈ 2.7e6 rad/s` on an asymmetric body must not panic, must keep the
/// orientation a unit quaternion, and (splitting at `h|ω| ≈ 6e3`, far
/// outside the asymptotic regime) keep `|L|` constant to rounding and `E`
/// inside the interval that constant `|L|` allows.
#[test]
fn f_huge_omega_does_not_panic_and_keeps_l() {
    let inertia = [1.0, 2.0, 3.0];
    let omega = [1.0e6, 2.0e6, -1.5e6];
    for (backend, e) in entries() {
        let mut w = one_body_world(backend, free_body(inertia, omega));
        let e0 = energy(&w.bodies[0], inertia);
        let l0 = norm(world_l(&w.bodies[0], inertia));
        for _ in 0..3 {
            advance(&mut w, e);
        }
        let b = &w.bodies[0];
        let q = b.rotation;
        let qn = (q.x.to_f64() * q.x.to_f64()
            + q.y.to_f64() * q.y.to_f64()
            + q.z.to_f64() * q.z.to_f64()
            + q.w.to_f64() * q.w.to_f64())
        .sqrt();
        let en = energy(b, inertia);
        eprintln!("(f) {backend:?} {e:?}: |q| = {qn}, E/E0 = {}", en / e0);
        assert!((qn - 1.0).abs() < 1e-6, "{backend:?} {e:?}: |q| = {qn}");
        // the splitting conserves `|L_b|` to rounding, which confines `E`
        // to `[|L|²/(2 I_max), |L|²/(2 I_min)]`
        let l = norm(world_l(b, inertia));
        let rel = (l - l0).abs() / l0;
        let (lo, hi) = (l0 * l0 / (2.0 * 3.0), l0 * l0 / 2.0);
        assert!(rel < 1e-12, "{backend:?} {e:?}: |L| changed by {rel:.3e}");
        assert!(
            en >= lo * (1.0 - 1e-12) && en <= hi * (1.0 + 1e-12),
            "{backend:?} {e:?}: E = {en} outside [{lo}, {hi}]"
        );
    }
}
