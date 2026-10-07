//! Two-way momentum exchange between rigid bodies and a homogeneous medium
//! ([`DragMedium`]) inside the world's substep loop.
//!
//! # Oracles
//!
//! * **Closed form (one body).** With `F = c (u − v)` on the body and `−F` on
//!   the medium, both from the velocities at the start of the substep, the
//!   relative velocity `w = v − u` obeys
//!   `w_{k+1} = w_k − c h (1/m + 1/M) w_k`, so after `n` substeps
//!   `w_n = w_0 (1 − x)^n` with `x = c h (1/m + 1/M)` (forward Euler of
//!   `dw/dt = −c (1/m + 1/M) w`, solution `w_0 e^{−c (1/m + 1/M) t}`). For
//!   `0 ≤ x ≤ 1`, `|(1 − x)^n − e^{−n x}| ≤ n x² / 2` (each factor differs
//!   from `e^{−x}` by at most `x²/2` and all factors are in `[0, 1]`).
//! * **Momentum.** `Σ m_i v_i + P` with `P` the medium momentum. The
//!   reference sum is computed here as `v_i · m_i` (an integer `m_i`, so the
//!   product is exact), not with the participant's own helper.
//!
//! # Rounding bound on XPBD
//!
//! XPBD derives every body velocity from the position change of the substep,
//! `v' = ((x + v·h) − x) · (1/h)`. With `v·h` truncated by at most `ε < 2⁻⁶⁴`,
//! `1/h` rounded by `δ < 2⁻⁶⁴` and the product rounded by `ε' ≤ 2⁻⁶⁴`,
//! `|v' − v| ≤ ε/h + |v·h|·δ + ε' ≤ 2⁻⁶⁴ (1/h + |v| h + 2)` per component.
//! The medium's bookkeeping is exact, so this is the only momentum that goes
//! missing, and over `n` substeps
//!
//! ```text
//! |ΔP_total| ≤ n · Σ_i m_i · 2⁻⁶⁴ (1/h + V h + 2)        (|v| ≤ V)
//! ```
//!
//! per component. The scenes stay in the no-overshoot regime
//! (`c_i h / m_i ≤ 1`, `Σ c_i h / M ≤ 1`), where every velocity is a convex
//! combination of the previous ones, so `V` is the largest initial speed
//! component (plus one for slack). On TGS the velocity is kept as integrated
//! and the sum is the same bit pattern in every frame.
//!
//! XPBD and TGS differ by the XPBD loss above, propagated by the exchange.
//! The exchange map `I − h L` is self-adjoint in the mass-weighted inner
//! product `⟨a, b⟩ = Σ m_i a_i b_i + M a_u b_u` with eigenvalues in
//! `[−1, 1]` in this regime, so it does not grow a difference in that norm;
//! the difference after `n` substeps is then at most `n q` in that norm, with
//! `q = √(Σ m) · 2⁻⁶⁴ (1/h + V h + 10)` (the `+ 10` raw units cover the
//! medium's own roundings of `u`, `F`, `dv` in the two runs), i.e. at most
//! `n q / √m_min` on any one velocity component.

use alice_physics::coupling_medium::{
    DragMedium, DragMediumError, DRAG_MEDIUM_KIND, MEDIUM_OBS_MOMENTUM, MEDIUM_OBS_VELOCITY,
};
use alice_physics::det_math::exp64;
use alice_physics::sleeping::SleepConfig;
use alice_physics::world_participant::{
    Observed, Participant, ParticipantFault, StateError, StepError, WorldFault,
};
use alice_physics::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix, WorldSnapshotError,
};

const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];
const ULP: f64 = 1.0 / 18_446_744_073_709_551_616.0; // 2⁻⁶⁴

/// `b^n` by repeated multiplication (IEEE products, the same on every
/// platform; relative error at most about `n · 1.1e-16`).
fn powi(b: f64, n: i32) -> f64 {
    (0..n).fold(1.0, |acc, _| acc * b)
}

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn to_f(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// A world without gravity, damping or sleeping.
fn world(backend: SolverBackend, substeps: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    w
}

/// Free bodies of integer masses `masses`, 100 m apart (never in contact),
/// with velocities `vel`.
fn add_bodies(w: &mut PhysicsWorld, masses: &[i64], vel: &[[f64; 3]]) {
    for (k, (&m, v)) in masses.iter().zip(vel).enumerate() {
        let mut b = RigidBody::new_dynamic(v3(0.0, 100.0 * k as f64, 0.0), Fix128::from_int(m));
        b.velocity = v3(v[0], v[1], v[2]);
        w.add_body(b);
    }
}

fn medium(mass: f64, u: [f64; 3], couplings: &[(usize, f64)]) -> DragMedium {
    let mut m = DragMedium::new(fx(mass), v3(u[0], u[1], u[2])).expect("medium");
    for &(body, c) in couplings {
        m.couple(body, fx(c)).expect("coupling");
    }
    m
}

/// Observed medium (velocity, momentum) of participant `index`.
fn observed(w: &PhysicsWorld, index: usize) -> (Vec3Fix, Vec3Fix) {
    let Some(Observed::Exact(sink)) = w.observe_participant(index) else {
        panic!("medium observation undecided or missing");
    };
    let get = |ch: u32| {
        sink.values()
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, v)| *v)
            .expect("channel")
    };
    (
        Vec3Fix::new(
            get(MEDIUM_OBS_VELOCITY),
            get(MEDIUM_OBS_VELOCITY + 1),
            get(MEDIUM_OBS_VELOCITY + 2),
        ),
        Vec3Fix::new(
            get(MEDIUM_OBS_MOMENTUM),
            get(MEDIUM_OBS_MOMENTUM + 1),
            get(MEDIUM_OBS_MOMENTUM + 2),
        ),
    )
}

/// `P + Σ m_i v_i` over the first `masses.len()` bodies (exact: integer
/// masses).
fn total_momentum(w: &PhysicsWorld, masses: &[i64]) -> Vec3Fix {
    let (_, mut p) = observed(w, 0);
    for (b, &m) in w.bodies.iter().zip(masses) {
        p = p + b.velocity * Fix128::from_int(m);
    }
    p
}

fn max_abs(v: Vec3Fix) -> f64 {
    let [x, y, z] = to_f(v);
    x.abs().max(y.abs()).max(z.abs())
}

fn bodies_bits(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

// ---------------------------------------------------------------------------
// 1. Closed form
// ---------------------------------------------------------------------------

/// One body (`m = 2`, `v0 = (3, −1, 0.5)`) and the medium (`M = 6`,
/// `u0 = (−1, 0, 0)`), `c = 1.5`, `dt = 1/64`, 4 substeps: `h = 1/256`,
/// `x = c h (1/m + 1/M) = 1/256`. After 120 frames (`n = 480`) the relative
/// velocity matches `w_0 (1 − x)^n` to `1e-12` (the f64 evaluation of the
/// closed form, about `n · 1.1e-16 · |w_0|`, dominates the Fix128 rounding
/// of at most `2⁻⁶⁴ (1/h + 10)` per substep), and the continuous
/// `w_0 e^{−n x}` to within `|w_0| n x² / 2`.
#[test]
fn relative_velocity_follows_the_discrete_closed_form() {
    let (m, mm, c): (f64, f64, f64) = (2.0, 6.0, 1.5);
    let w0 = [4.0, -1.0, 0.5];
    for backend in BACKENDS {
        let mut w = world(backend, 4);
        add_bodies(&mut w, &[2], &[[3.0, -1.0, 0.5]]);
        w.add_participant(Box::new(medium(mm, [-1.0, 0.0, 0.0], &[(0, c)])))
            .expect("register");
        let dt = Fix128::from_ratio(1, 64);
        let h: f64 = 1.0 / 256.0;
        let x = c * h * (1.0 / m + 1.0 / mm);
        for frame in 1..=120 {
            w.try_step(dt).expect("step");
            let n = 4 * frame;
            let (u, _) = observed(&w, 0);
            let rel = to_f(w.bodies[0].velocity - u);
            for k in 0..3 {
                let discrete = w0[k] * powi(1.0 - x, n);
                let continuous = w0[k] * exp64(-(n as f64) * x);
                assert!(
                    (rel[k] - discrete).abs() <= 1e-12,
                    "{backend:?} frame {frame} axis {k}: {} vs discrete {discrete}",
                    rel[k]
                );
                let bound = w0[k].abs() * n as f64 * x * x / 2.0;
                assert!(
                    (rel[k] - continuous).abs() <= bound + 1e-12,
                    "{backend:?} frame {frame} axis {k}: {} vs continuous {continuous} (bound {bound})",
                    rel[k]
                );
            }
        }
        // The relaxation happened: e^{-480/256} ≈ 0.153.
        let (u, _) = observed(&w, 0);
        let rel = to_f(w.bodies[0].velocity - u);
        assert!((rel[0] - 4.0 * powi(1.0 - x, 480)).abs() < 1e-12 && rel[0] < 0.7);
    }
}

// ---------------------------------------------------------------------------
// 2. Momentum
// ---------------------------------------------------------------------------

const MASSES: [i64; 4] = [1, 2, 4, 8];
const VEL: [[f64; 3]; 4] = [
    [1.0, 0.0, 0.7],
    [2.0, -0.3, 0.7],
    [3.0, -0.6, -0.7],
    [-1.5, 0.25, 0.0],
];
const COEFF: [(usize, f64); 4] = [(0, 0.3), (1, 0.7), (2, 2.0), (3, 1.3)];
/// Largest initial speed component (bodies and medium) plus one.
const V_BOUND: f64 = 4.0;

fn momentum_scene(backend: SolverBackend, substeps: usize) -> PhysicsWorld {
    let mut w = world(backend, substeps);
    add_bodies(&mut w, &MASSES, &VEL);
    w.add_participant(Box::new(medium(3.0, [-2.0, 0.5, 0.0], &COEFF)))
        .expect("register");
    w
}

/// The frame widths and substep counts of the momentum and path tests:
/// dyadic and not, power-of-two substeps and not.
const CASES: [(i64, i64, usize); 3] = [(1, 64, 4), (1, 60, 8), (1, 64, 3)];

/// Exact momentum conservation holds when (all four are true here): every
/// coupled body has a power-of-two mass `≥ 1` (so `dv / inv_mass` and
/// `v · m` are exact), no other participant pushes those bodies, nothing else
/// changes their momentum (no gravity, damping, contact or sleep), and the
/// world keeps the integrated velocity (TGS). On TGS the sum is the same bit
/// pattern in every frame; on XPBD it stays within the bound of the module
/// documentation.
///
/// XPBD: this tightens to `assert_eq!` once XPBD keeps the predicted velocity
/// of bodies its constraints did not move.
#[test]
fn total_momentum_is_conserved_exactly_on_tgs_and_within_the_bound_on_xpbd() {
    let sum_m: f64 = MASSES.iter().map(|&m| m as f64).sum();
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let h = num as f64 / den as f64 / substeps as f64;
        for backend in BACKENDS {
            let mut w = momentum_scene(backend, substeps);
            let start = total_momentum(&w, &MASSES);
            let (_, p0) = observed(&w, 0);
            let mut worst = 0.0f64;
            for frame in 1..=240 {
                w.try_step(dt).expect("step");
                let now = total_momentum(&w, &MASSES);
                match backend {
                    SolverBackend::Tgs => {
                        assert_eq!(now, start, "TGS {num}/{den} s{substeps} frame {frame}");
                    }
                    _ => {
                        let n = (frame * substeps) as f64;
                        let bound = n * sum_m * ULP * (1.0 / h + V_BOUND * h + 2.0);
                        let d = max_abs(now - start);
                        worst = worst.max(d / bound);
                        assert!(
                            d <= bound,
                            "XPBD {num}/{den} s{substeps} frame {frame}: |ΔP| {d:e} > {bound:e}"
                        );
                    }
                }
            }
            // The exchange moved momentum between bodies and medium (not a
            // conservation by doing nothing).
            let (_, p1) = observed(&w, 0);
            assert!(
                max_abs(p1 - p0) > 1.0,
                "{backend:?}: medium momentum barely moved"
            );
            if backend == SolverBackend::Xpbd {
                println!("XPBD {num}/{den} s{substeps}: worst |ΔP| / bound = {worst:.3}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Determinism through a snapshot
// ---------------------------------------------------------------------------

/// A snapshot after 37 frames restored into a fresh world (same bodies, a new
/// medium of the same configuration) continues bit for bit like the
/// uninterrupted run, on both backends; a medium of another configuration
/// refuses the payload and leaves its world unchanged.
#[test]
fn snapshot_restore_continues_bit_for_bit() {
    let dt = Fix128::from_ratio(1, 60);
    for backend in BACKENDS {
        let mut a = momentum_scene(backend, 4);
        for _ in 0..37 {
            a.try_step(dt).expect("step");
        }
        let blob = a.snapshot_world();
        let mut b = momentum_scene(backend, 4);
        b.restore_world(&blob).expect("restore");
        assert_eq!(b.participant_state(0), a.participant_state(0));
        for frame in 0..50 {
            a.try_step(dt).expect("step");
            b.try_step(dt).expect("step");
            assert_eq!(
                bodies_bits(&a),
                bodies_bits(&b),
                "{backend:?} frame {frame}"
            );
            assert_eq!(a.participant_state(0), b.participant_state(0));
        }

        let mut other = world(backend, 4);
        add_bodies(&mut other, &MASSES, &VEL);
        let mut coeff = COEFF;
        coeff[1].1 = 0.75;
        other
            .add_participant(Box::new(medium(3.0, [-2.0, 0.5, 0.0], &coeff)))
            .expect("register");
        let before = other.snapshot_world();
        assert_eq!(
            other.restore_world(&blob),
            Err(WorldSnapshotError::ParticipantState {
                index: 0,
                error: StateError::InvalidValue
            })
        );
        assert_eq!(other.snapshot_world(), before);
    }
}

// ---------------------------------------------------------------------------
// 4. Paths
// ---------------------------------------------------------------------------

/// The same scene on XPBD and on TGS gives bit-identical body velocities and
/// medium momentum, frame by frame. XPBD keeps the predicted velocity of a
/// body its constraints did not move (`v = v_pred + Δx_corr / h`, and nothing
/// in this scene moves a body), so both backends apply the same staged force
/// updates to the same velocities. XPBD hands the participants
/// `h = dt / substeps` and TGS its own solve width `dt * (1 / substeps as f32)`;
/// the comparison runs only for the cases where the two widths are the same
/// value, and the other cases are checked to really have different widths.
#[test]
fn xpbd_and_tgs_agree_bit_for_bit() {
    let mut compared = 0;
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let xpbd_h = dt / Fix128::from_int(substeps as i64);
        let tgs_h = dt * Fix128::from_f32(1.0 / substeps as f32);
        if xpbd_h != tgs_h {
            // different widths: the two runs solve different discretisations
            assert!(substeps != 1 && !substeps.is_power_of_two());
            continue;
        }
        compared += 1;
        let mut x = momentum_scene(SolverBackend::Xpbd, substeps);
        let mut t = momentum_scene(SolverBackend::Tgs, substeps);
        for frame in 1..=240 {
            x.try_step(dt).expect("step");
            t.try_step(dt).expect("step");
            for (i, (bx, bt)) in x.bodies.iter().zip(&t.bodies).enumerate() {
                assert_eq!(
                    bx.velocity, bt.velocity,
                    "{num}/{den} s{substeps} frame {frame} body {i}"
                );
            }
            assert_eq!(
                observed(&x, 0),
                observed(&t, 0),
                "{num}/{den} s{substeps} frame {frame} medium"
            );
        }
    }
    assert!(compared >= 2, "too few cases with equal widths: {compared}");
}

// ---------------------------------------------------------------------------
// 5. Substep convergence
// ---------------------------------------------------------------------------

/// One body, `c (1/m + 1/M) = 1`, one second of `dt = 1/16`: the error of the
/// relative velocity against `w_0 e^{−t}` falls with every halving of `h`
/// (substeps 1, 2, 4, 8), by about the factor 2 of a first-order method
/// (`(1 − x)^n − e^{−nx} ≈ −n x² e^{−nx} / 2`, linear in `h` at fixed `t`).
#[test]
fn halving_the_substep_moves_toward_the_continuous_solution() {
    let mut errors = Vec::new();
    for substeps in [1usize, 2, 4, 8] {
        let mut w = world(SolverBackend::Tgs, substeps);
        add_bodies(&mut w, &[2], &[[3.0, 0.0, 0.0]]);
        w.add_participant(Box::new(medium(6.0, [-1.0, 0.0, 0.0], &[(0, 1.5)])))
            .expect("register");
        for _ in 0..16 {
            w.try_step(Fix128::from_ratio(1, 16)).expect("step");
        }
        let rel = (w.bodies[0].velocity.x - observed(&w, 0).0.x).to_f64();
        errors.push((rel - 4.0 * exp64(-1.0)).abs());
    }
    for pair in errors.windows(2) {
        assert!(pair[1] < pair[0], "error did not decrease: {errors:?}");
        let ratio = pair[0] / pair[1];
        assert!((1.8..=2.2).contains(&ratio), "not first order: {errors:?}");
    }
}

// ---------------------------------------------------------------------------
// 6. Gravity and a floor
// ---------------------------------------------------------------------------

/// A ball (`m = 1`, radius 0.5) resting on a frictionless floor (the top of a
/// static sphere of radius `R = 10⁶`) under gravity `−10`, in a medium of
/// `M = 2` moving at `u0 = 4` along `x`, `c = 1`. Along `x` only the drag acts
/// (frictionless contact, vertical gravity), so the ball reaches the common
/// velocity `V = M u0 / (m + M) = 8/3` with `|v − u| = 4 e^{−1.5 t}`
/// (discrete form), and the horizontal momentum `m v + P` stays `8`. The
/// floor is curved, so its normal tilts by `x / R` and adds a horizontal
/// force of at most `m g |x| / R`; over `T` seconds with `|x| ≤ X` that is at
/// most `m g X T / R` of momentum, the tolerance of both checks. The ball
/// stays on the floor.
#[test]
fn a_ball_on_a_frictionless_floor_reaches_the_common_velocity() {
    for backend in BACKENDS {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: v3(0.0, -10.0, 0.0),
            damping: Fix128::ONE,
            substeps: 4,
            solver_backend: backend,
            ..PhysicsConfig::default()
        });
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: u32::MAX,
            ..SleepConfig::default()
        });
        let r = 1.0e6;
        let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -r, 0.0)), fx(r));
        let ball = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
            fx(0.5),
        );
        w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
        let id = w
            .material_table
            .register(alice_physics::PhysicsMaterial::new(
                0,
                Fix128::ZERO,
                Fix128::ZERO,
            ));
        w.set_body_material(ground, id);
        w.set_body_material(ball, id);
        w.add_participant(Box::new(medium(2.0, [4.0, 0.0, 0.0], &[(ball, 1.0)])))
            .expect("register");
        let frames: i32 = 240; // T = 4 s at 60 Hz
        let mut x_max = 0.0f64;
        for _ in 0..frames {
            w.try_step(Fix128::from_ratio(1, 60)).expect("step");
            x_max = x_max.max(w.bodies[ball].position.x.to_f64().abs());
            let y = w.bodies[ball].position.y.to_f64();
            assert!(
                (y - 0.5).abs() < 0.01,
                "{backend:?}: ball left the floor, y = {y}"
            );
        }
        let t = frames as f64 / 60.0;
        let tilt = 10.0 * x_max * t / r;
        let (u, p) = observed(&w, 0);
        let v = w.bodies[ball].velocity.x.to_f64();
        let px = v + p.x.to_f64();
        assert!(
            (px - 8.0).abs() <= tilt,
            "{backend:?}: momentum {px} (tilt bound {tilt:e})"
        );
        let h: f64 = 1.0 / 240.0;
        let transient = 4.0 * powi(1.0 - 1.5 * h, 4 * frames);
        assert!(
            ((v - u.x.to_f64()).abs() - transient).abs() <= 2.0 * tilt,
            "{backend:?}: v {v} u {} transient {transient}",
            u.x.to_f64()
        );
        assert!(
            (v - 8.0 / 3.0).abs() <= transient + 2.0 * tilt,
            "{backend:?}: v {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Degenerate inputs
// ---------------------------------------------------------------------------

/// Construction refuses `M ≤ 0`, `c < 0`, a body coupled twice and a
/// momentum `M u` out of range, each with its own error.
#[test]
fn construction_refuses_invalid_media() {
    assert_eq!(
        DragMedium::new(Fix128::ZERO, Vec3Fix::ZERO).err(),
        Some(DragMediumError::NonPositiveMass)
    );
    assert_eq!(
        DragMedium::new(fx(-1.0), Vec3Fix::ZERO).err(),
        Some(DragMediumError::NonPositiveMass)
    );
    assert_eq!(
        DragMedium::new(Fix128::from_int(1 << 40), Vec3Fix::from_int(1 << 40, 0, 0)).err(),
        Some(DragMediumError::MomentumOutOfRange)
    );
    let mut m = DragMedium::new(Fix128::ONE, Vec3Fix::ZERO).expect("medium");
    assert_eq!(
        m.couple(0, fx(-0.5)),
        Err(DragMediumError::NegativeCoefficient { body: 0 })
    );
    m.couple(0, Fix128::ZERO).expect("c = 0 is allowed");
    assert_eq!(
        m.couple(0, Fix128::ONE),
        Err(DragMediumError::DuplicateBody { body: 0 })
    );
    assert_eq!(m.couplings().len(), 1);
}

/// A coupled index past the last body is a participant fault
/// ([`ParticipantFault::InvalidState`]): the step runs to the end without the
/// medium's forces (the bodies match a world without the participant), the
/// medium is unchanged and the next step is refused.
#[test]
fn a_missing_body_is_a_participant_fault_and_changes_nothing() {
    for backend in BACKENDS {
        let mut w = world(backend, 4);
        add_bodies(&mut w, &[1, 2], &VEL[..2]);
        let mut bare = world(backend, 4);
        add_bodies(&mut bare, &[1, 2], &VEL[..2]);
        w.add_participant(Box::new(medium(
            3.0,
            [1.0, 0.0, 0.0],
            &[(0, 1.0), (5, 1.0)],
        )))
        .expect("register");
        let state = w.participant_state(0);
        let dt = Fix128::from_ratio(1, 60);
        let fault = WorldFault::Participant {
            index: 0,
            kind: DRAG_MEDIUM_KIND,
            fault: ParticipantFault::InvalidState,
        };
        assert_eq!(w.try_step(dt), Err(StepError::FaultRaised(fault)));
        bare.step(dt);
        assert_eq!(bodies_bits(&w), bodies_bits(&bare));
        assert_eq!(w.participant_state(0), state);
        assert_eq!(w.try_step(dt), Err(StepError::Faulted(fault)));
        assert_eq!(w.observe_participant(0), Some(Observed::Undecided));
    }
}

/// A product out of range (`F = c (u − v)` with `v = 2⁶²`, `c = 4`) is
/// [`ParticipantFault::OutOfRange`] before anything is staged: no body gets a
/// force, the medium is unchanged.
#[test]
fn an_out_of_range_force_is_a_participant_fault() {
    let mut w = world(SolverBackend::Tgs, 2);
    let mut bare = world(SolverBackend::Tgs, 2);
    for x in [&mut w, &mut bare] {
        x.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut fast = RigidBody::new_dynamic(v3(0.0, 100.0, 0.0), Fix128::ONE);
        fast.velocity = Vec3Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO, Fix128::ZERO);
        x.add_body(fast);
    }
    w.add_participant(Box::new(medium(
        1.0,
        [0.0, 0.0, 0.0],
        &[(0, 1.0), (1, 4.0)],
    )))
    .expect("register");
    let state = w.participant_state(0);
    let dt = Fix128::from_ratio(1, 1024);
    assert_eq!(
        w.try_step(dt),
        Err(StepError::FaultRaised(WorldFault::Participant {
            index: 0,
            kind: DRAG_MEDIUM_KIND,
            fault: ParticipantFault::OutOfRange,
        }))
    );
    bare.step(dt);
    assert_eq!(bodies_bits(&w), bodies_bits(&bare));
    assert_eq!(w.participant_state(0), state);
}

/// A force whose product is in range but whose new velocity `v + dv` is not
/// (`v = 5·2⁶⁰`, `u = 6·2⁶⁰`, `c · inv_mass · h = 4`, so `dv = 4·2⁶⁰`) is
/// also refused by the medium itself ([`ParticipantFault::OutOfRange`]), not
/// left to the world (which would record [`WorldFault::ForceOutOfRange`]
/// after the medium had already taken the reaction).
#[test]
fn a_new_velocity_out_of_range_is_refused_by_the_medium() {
    let big = |k: i64| Fix128::from_raw(k << 60, 0);
    let build = || {
        let mut w = world(SolverBackend::Tgs, 1);
        let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_ratio(1, 4));
        b.velocity = Vec3Fix::new(big(5), Fix128::ZERO, Fix128::ZERO);
        w.add_body(b);
        w
    };
    let mut w = build();
    let mut bare = build();
    let mut m = DragMedium::new(
        Fix128::ONE,
        Vec3Fix::new(big(6), Fix128::ZERO, Fix128::ZERO),
    )
    .expect("medium");
    m.couple(0, Fix128::ONE).expect("couple");
    w.add_participant(Box::new(m)).expect("register");
    let state = w.participant_state(0);
    assert_eq!(
        w.try_step(Fix128::ONE),
        Err(StepError::FaultRaised(WorldFault::Participant {
            index: 0,
            kind: DRAG_MEDIUM_KIND,
            fault: ParticipantFault::OutOfRange,
        }))
    );
    bare.step(Fix128::ONE);
    assert_eq!(bodies_bits(&w), bodies_bits(&bare));
    assert_eq!(w.participant_state(0), state);
}

/// Static and kinematic bodies get no force and give no reaction (even with a
/// non-zero `inv_mass`): a medium
/// coupled to a static, a kinematic and a dynamic body evolves bit for bit
/// like one coupled to the dynamic body alone, and the static and kinematic
/// bodies move as without the medium.
#[test]
fn static_and_kinematic_bodies_are_skipped() {
    for backend in BACKENDS {
        let build = |couple_all: bool| {
            let mut w = world(backend, 4);
            // A non-zero `inv_mass` on both: the body type, not the mass,
            // decides that the world applies no force.
            let mut s = RigidBody::new_static(v3(0.0, -100.0, 0.0));
            s.inv_mass = Fix128::ONE;
            w.add_body(s);
            let mut k = RigidBody::new_kinematic(v3(0.0, 100.0, 0.0));
            k.velocity = v3(0.5, 0.0, 0.0);
            k.inv_mass = Fix128::ONE;
            w.add_body(k);
            add_bodies(&mut w, &[2], &[[3.0, 0.0, 0.0]]);
            let couplings: &[(usize, f64)] = if couple_all {
                &[(0, 1.0), (1, 1.0), (2, 1.0)]
            } else {
                &[(2, 1.0)]
            };
            w.add_participant(Box::new(medium(3.0, [-1.0, 0.0, 0.0], couplings)))
                .expect("register");
            w
        };
        let mut all = build(true);
        let mut one = build(false);
        for _ in 0..60 {
            all.try_step(Fix128::from_ratio(1, 60)).expect("step");
            one.try_step(Fix128::from_ratio(1, 60)).expect("step");
        }
        assert_eq!(bodies_bits(&all), bodies_bits(&one), "{backend:?}");
        assert_eq!(observed(&all, 0), observed(&one, 0), "{backend:?}");
    }
}

/// A medium coupled to no body keeps its momentum and leaves the world as it
/// is without it, bit for bit.
#[test]
fn a_medium_with_no_bodies_changes_nothing() {
    for backend in BACKENDS {
        let mut w = world(backend, 4);
        let mut bare = world(backend, 4);
        add_bodies(&mut w, &MASSES, &VEL);
        add_bodies(&mut bare, &MASSES, &VEL);
        w.add_participant(Box::new(medium(3.0, [-2.0, 0.5, 0.0], &[])))
            .expect("register");
        let (_, p0) = observed(&w, 0);
        for _ in 0..30 {
            w.try_step(Fix128::from_ratio(1, 60)).expect("step");
            bare.step(Fix128::from_ratio(1, 60));
        }
        assert_eq!(bodies_bits(&w), bodies_bits(&bare));
        assert_eq!(observed(&w, 0).1, p0);
    }
}

/// `check_state` refuses a payload of another length, version or
/// configuration and accepts its own; `read_state` restores the momentum.
#[test]
fn state_payload_is_checked() {
    let mut a = medium(3.0, [1.0, 2.0, 3.0], &[(0, 1.0)]);
    let mut bytes = Vec::new();
    a.write_state(&mut bytes);
    assert_eq!(bytes.len(), 60);
    assert_eq!(a.check_state(&bytes), Ok(()));
    assert_eq!(
        a.check_state(&bytes[..59]),
        Err(StateError::Length {
            expected: 60,
            found: 59
        })
    );
    let mut bad = bytes.clone();
    bad[0] = 2;
    assert_eq!(a.check_state(&bad), Err(StateError::InvalidValue));
    let other = medium(3.0, [1.0, 2.0, 3.0], &[(0, 2.0)]);
    assert_eq!(other.check_state(&bytes), Err(StateError::InvalidValue));
    let mut b = medium(3.0, [0.0, 0.0, 0.0], &[(0, 1.0)]);
    assert_eq!(b.check_state(&bytes), Ok(()));
    b.read_state(&bytes);
    assert_eq!(b.momentum(), a.momentum());
    a.couple(1, Fix128::ONE).expect("couple");
    assert_eq!(a.check_state(&bytes), Err(StateError::InvalidValue));
}
