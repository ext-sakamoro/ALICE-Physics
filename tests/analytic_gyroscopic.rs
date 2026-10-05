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
//! - (e) the rotational kinetic energy `½ ωᵀ I ω` is constant, and an implicit
//!   discretisation may only lose it, never gain it;
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

/// oracle: torque-free ⇒ `dL/dt = 0` exactly. A first-order method has a
/// global error `≤ T · h · |ω|² · κ` relative to `|L|` (`κ = I_max / I_min`
/// bounds the body-frame rate of `L_b`, `|dL_b/dt| = |ω × L_b|`), here
/// `2 · (1/480) · 1.3² · 3 ≈ 0.021`; the bound used is 0.03. A solver without
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
        if worst >= 0.03 {
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
/// solver as it was before the gyroscopic term was added.
const GOLDEN_ISOTROPIC_XPBD: u64 = 0x4155_588b_1de1_ab6d;
const GOLDEN_ISOTROPIC_TGS: u64 = 0x107a_2c02_798b_e9f5;

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
const GOLDEN_ISOTROPIC_PARALLEL: u64 = 0x4155_588b_1de1_ab6d;

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

/// oracle: torque-free ⇒ `E = ½ ωᵀ I ω` constant. The implicit step solves
/// `I(ω₂ − ω₁) = −h ω₂ × Iω₂`; dotting with `ω₂` gives
/// `ω₂ᵀ I ω₂ = ω₂ᵀ I ω₁ ≤ √(ω₂ᵀIω₂ · ω₁ᵀIω₁)`, i.e. `E₂ ≤ E₁`. A forward-Euler
/// step instead gains `½ h² |I^{-½}(ω × Iω)|²`-order energy every step. The
/// bound per frame is a relative `1e-9` for rounding and the single Newton
/// step's `O(h³)` residual (`(|ω| h)³ ≈ 1e-4` relative at `|ω| ≈ 10`, but the
/// residual's energy contribution is second order in it). Run at
/// `|ω| ≈ 10 rad/s` so `|ω| h ≈ 0.02`.
#[test]
fn e_rotational_energy_does_not_grow() {
    let inertia = [1.0, 2.0, 3.0];
    let omega = [6.0, -5.0, 4.0];
    let mut bad = Vec::new();
    for (backend, e) in entries() {
        let mut w = one_body_world(backend, free_body(inertia, omega));
        let e0 = energy(&w.bodies[0], inertia);
        let mut prev = e0;
        let mut worst_gain = f64::MIN;
        for _ in 0..120 {
            advance(&mut w, e);
            let en = energy(&w.bodies[0], inertia);
            worst_gain = worst_gain.max((en - prev) / e0);
            prev = en;
        }
        eprintln!(
            "(e) {backend:?} {e:?}: worst per-frame gain {worst_gain:.3e}, E end/start {:.6}",
            prev / e0
        );
        if worst_gain > 1e-9 {
            bad.push(format!(
                "(e) {backend:?} {e:?}: rotational energy grew by {worst_gain:.3e} of E0 in one frame"
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ============================================================================
// (f) degenerate inputs
// ============================================================================

/// oracle: `ω = 0 ⇒ ω × Iω = 0`, so an asymmetric body at rest must evolve
/// bit-identically to an isotropic twin at rest with the same orientation
/// (the twin has no gyroscopic response at all). Not compared against an
/// exact zero: XPBD derives `ω` from `q·q_prev⁻¹`, which already leaves a
/// 1-ulp residue for a body at rest before any of this was added.
#[test]
fn f_zero_omega_matches_isotropic_twin_bit_for_bit() {
    for (backend, e) in entries() {
        let q = QuatFix::from_axis_angle(v3(0.0, 0.6, 0.8), fx(0.7));
        let mut asym = free_body([1.0, 2.0, 3.0], [0.0; 3]);
        asym.rotation = q;
        let mut iso = free_body([2.0, 2.0, 2.0], [0.0; 3]);
        iso.rotation = q;
        let mut wa = one_body_world(backend, asym);
        let mut wi = one_body_world(backend, iso);
        for _ in 0..10 {
            advance(&mut wa, e);
            advance(&mut wi, e);
        }
        assert_eq!(
            wa.bodies[0].angular_velocity, wi.bodies[0].angular_velocity,
            "{backend:?} {e:?}"
        );
        assert_eq!(
            wa.bodies[0].rotation, wi.bodies[0].rotation,
            "{backend:?} {e:?}"
        );
    }
}

/// Expected: a static body (infinite mass and inertia) is never moved and a
/// body whose `inv_inertia` is zero keeps rotating at its world `ω` (an
/// infinite moment has no gyroscopic response); covered bit-for-bit by
/// `d_isotropic_and_exempt_bodies_are_bit_identical_to_before`. Here: a huge
/// `|ω| ≈ 2.7e6 rad/s` on an asymmetric body must not panic, must keep the
/// orientation a unit quaternion, and must not gain energy (the bound of (e)),
/// whether the term is applied or skipped because its products leave the
/// fixed-point range.
#[test]
fn f_huge_omega_does_not_panic_or_gain_energy() {
    let inertia = [1.0, 2.0, 3.0];
    let omega = [1.0e6, 2.0e6, -1.5e6];
    for (backend, e) in entries() {
        let mut w = one_body_world(backend, free_body(inertia, omega));
        let e0 = energy(&w.bodies[0], inertia);
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
        assert!(
            en <= e0 * (1.0 + 1e-9),
            "{backend:?} {e:?}: E/E0 = {}",
            en / e0
        );
    }
}
