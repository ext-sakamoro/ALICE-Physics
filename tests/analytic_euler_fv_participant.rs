//! `EulerFv1dParticipant` (`EUFV`): the 1-D Euler finite-volume solver
//! advancing inside the world's substep loop, splitting each substep into its
//! own CFL steps (`StepRule::Subcycle`).
//!
//! Oracles:
//! - **Sod shock tube (closed form)**: run through the world to `t = 1/4`, the
//!   density matches the exact Riemann solution (`exact_riemann`, itself
//!   checked against Toro's table 4.2 in `tests/analytic_euler_fv.rs`) with
//!   an L1 error that falls under refinement at the first-order Godunov rate
//!   (between `2^{1/2}` and `2` per halving of `Δx`).
//! - **World path = direct stepping**: the world's substeps give the cells,
//!   clock and step counts of direct `advance_to` calls at the same targets,
//!   bit for bit, under both solver backends.
//! - **Conservation**: with periodic boundaries `Σρ`, `Σρu`, `ΣE` stay bit for
//!   bit through the world.
//! - **Subcycle determinism**: snapshot, branch and resume gives the same cells
//!   and the same per-substep step counts as an unbroken run.
//! - **Payloads**: another length, version, digest, a negative clock or a cell
//!   the solver refuses is refused.

use alice_physics_core::euler_fv::{
    exact_riemann, Boundary, Conserved, EulerConfig, EulerFv1d, EulerFv1dParticipant, Primitive,
    EULER_FV_OBS_ENERGY, EULER_FV_OBS_LAST_STEPS, EULER_FV_OBS_MASS, EULER_FV_OBS_MOMENTUM,
    EULER_FV_OBS_STEPS, EULER_FV_OBS_TIME, EULER_FV_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{Observed, Participant, StateError, StepRule};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const HEADER: usize = 4 + 8 + 16 + 8 + 8;

fn q(num: i64, den: i64) -> Fix128 {
    Fix128::from_int(num) / Fix128::from_int(den)
}

fn gamma() -> Fix128 {
    q(7, 5)
}

fn prim(rho: Fix128, u: Fix128, p: Fix128) -> Primitive {
    Primitive { rho, u, p }
}

fn sod_left() -> Primitive {
    prim(Fix128::ONE, Fix128::ZERO, Fix128::ONE)
}

fn sod_right() -> Primitive {
    prim(q(1, 8), Fix128::ZERO, q(1, 10))
}

fn sod(n: usize) -> EulerFv1d {
    let init: Vec<Primitive> = (0..n)
        .map(|i| if 2 * i < n { sod_left() } else { sod_right() })
        .collect();
    EulerFv1d::new(EulerConfig::godunov(gamma()), q(1, n as i64), &init).expect("valid")
}

/// A smooth periodic density / pressure wave moving to the right.
fn periodic(n: usize) -> EulerFv1d {
    let mut config = EulerConfig::godunov(gamma());
    config.left = Boundary::Periodic;
    config.right = Boundary::Periodic;
    let init: Vec<Primitive> = (0..n)
        .map(|i| {
            let bump = if i < n / 4 { q(1, 2) } else { Fix128::ZERO };
            prim(Fix128::ONE + bump, q(1, 3), Fix128::ONE + bump)
        })
        .collect();
    EulerFv1d::new(config, q(1, n as i64), &init).expect("valid")
}

fn world(substeps: usize, backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    w.config.solver_backend = backend;
    w
}

fn register(w: &mut PhysicsWorld, s: EulerFv1d) -> usize {
    let p = EulerFv1dParticipant::new(s);
    assert_eq!(p.kind(), EULER_FV_PARTICIPANT_KIND);
    assert_eq!(p.step_rule(), StepRule::Subcycle);
    w.add_participant(Box::new(p)).expect("register")
}

fn observed(w: &PhysicsWorld, index: usize) -> Vec<(u32, Fix128)> {
    match w.observe_participant(index).expect("participant") {
        Observed::Exact(sink) => sink.values().to_vec(),
        Observed::Undecided => panic!("observation undecided"),
    }
}

fn channel(obs: &[(u32, Fix128)], c: u32) -> Fix128 {
    obs.iter().find(|(k, _)| *k == c).expect("channel").1
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn fix_at(b: &[u8], at: usize) -> Fix128 {
    Fix128 {
        hi: u64_at(b, at) as i64,
        lo: u64_at(b, at + 8),
    }
}

/// Cells decoded from a participant payload.
fn cells_of(state: &[u8]) -> Vec<Conserved> {
    state[HEADER..]
        .chunks_exact(48)
        .map(|c| Conserved {
            rho: fix_at(c, 0),
            mom: fix_at(c, 16),
            energy: fix_at(c, 32),
        })
        .collect()
}

fn f64_of(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

/// L1 density error against the exact Riemann solution at `t`, diaphragm at
/// `x = 1/2`.
fn l1_density(cells: &[Conserved], t: Fix128) -> f64 {
    let n = cells.len();
    let star = exact_riemann(gamma(), &sod_left(), &sod_right()).expect("no vacuum");
    let dx = 1.0 / n as f64;
    cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let x = q(2 * i as i64 + 1, 2 * n as i64) - q(1, 2);
            let e = star.sample(gamma(), &sod_left(), &sod_right(), x / t);
            (f64_of(c.rho) - f64_of(e.rho)).abs() * dx
        })
        .sum()
}

#[test]
fn sod_through_the_world_converges_to_the_exact_riemann_solution() {
    // 4 frames of 2 substeps of 1/32: t = 1/4 exactly.
    let t = q(1, 4);
    let mut errs = Vec::new();
    for n in [50usize, 100, 200] {
        let mut w = world(2, SolverBackend::Xpbd);
        let i = register(&mut w, sod(n));
        for _ in 0..4 {
            w.try_step(q(1, 16)).expect("step");
        }
        let obs = observed(&w, i);
        assert_eq!(channel(&obs, EULER_FV_OBS_TIME), t, "N={n}: clock");
        let steps = channel(&obs, EULER_FV_OBS_STEPS);
        assert!(
            steps > Fix128::from_int(8),
            "N={n}: at least one CFL step per substep"
        );
        let state = w.participant_state(i).expect("state");
        errs.push(l1_density(&cells_of(&state), t));
    }
    let r1 = errs[0] / errs[1];
    let r2 = errs[1] / errs[2];
    eprintln!("world sod L1 {errs:?}, ratios {r1:.3} {r2:.3}");
    assert!(errs[0] < 0.05, "L1 at N=50: {}", errs[0]);
    for ratio in [r1, r2] {
        assert!(
            (1.35..2.0).contains(&ratio),
            "L1 {errs:?}: expected the first-order Godunov rate between 2^(1/2) and 2"
        );
    }
}

#[test]
fn the_world_path_equals_direct_advance_to_calls_bit_for_bit() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut w = world(4, backend);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let i = register(&mut w, sod(64));
        let mut direct = sod(64);
        let mut counts = Vec::new();
        let h = q(1, 64);
        for frame in 0..3 {
            w.try_step(Fix128::from_int(4) * h).expect("step");
            for k in 0..4 {
                let target = Fix128::from_int(4 * frame + k + 1) * h;
                counts.push(direct.advance_to(target).expect("stable") as u64);
            }
            let obs = observed(&w, i);
            assert_eq!(
                channel(&obs, EULER_FV_OBS_LAST_STEPS),
                Fix128::from_int(*counts.last().expect("count") as i64),
                "{backend:?} frame {frame}: last subcycle count"
            );
        }
        let state = w.participant_state(i).expect("state");
        assert_eq!(
            cells_of(&state),
            direct.cells(),
            "{backend:?}: cells differ"
        );
        assert_eq!(fix_at(&state, 12), direct.time());
        assert_eq!(u64_at(&state, 36), counts.iter().sum::<u64>());
        // Not vacuous: the gas moved and more than one step ran in a substep.
        assert_ne!(cells_of(&state), sod(64).cells());
        assert!(
            counts.iter().any(|&c| c > 1),
            "{backend:?}: counts {counts:?}"
        );
    }
}

#[test]
fn periodic_totals_are_kept_bit_for_bit_through_the_world() {
    let mut w = world(3, SolverBackend::Xpbd);
    let start = periodic(48).totals();
    let i = register(&mut w, periodic(48));
    for _ in 0..10 {
        w.try_step(q(1, 20)).expect("step");
        let obs = observed(&w, i);
        assert_eq!(channel(&obs, EULER_FV_OBS_MASS), start.rho);
        assert_eq!(channel(&obs, EULER_FV_OBS_MOMENTUM), start.mom);
        assert_eq!(channel(&obs, EULER_FV_OBS_ENERGY), start.energy);
    }
    let state = w.participant_state(i).expect("state");
    assert_ne!(
        cells_of(&state),
        periodic(48).cells(),
        "the wave must have moved"
    );
}

#[test]
fn snapshot_branch_and_resume_repeats_the_subcycle_counts() {
    let dt = q(1, 32);
    let mut a = world(2, SolverBackend::Xpbd);
    let ia = register(&mut a, sod(80));
    for _ in 0..2 {
        a.try_step(dt).expect("step");
    }
    let snap = a.snapshot_world();
    let mut b = world(2, SolverBackend::Xpbd);
    let ib = register(&mut b, sod(80));
    b.restore_world(&snap).expect("restore");
    assert_eq!(a.participant_state(ia), b.participant_state(ib));
    for _ in 0..3 {
        a.try_step(dt).expect("step");
        b.try_step(dt).expect("step");
        assert_eq!(
            channel(&observed(&a, ia), EULER_FV_OBS_LAST_STEPS),
            channel(&observed(&b, ib), EULER_FV_OBS_LAST_STEPS)
        );
    }
    assert_eq!(a.participant_state(ia), b.participant_state(ib));
    assert_eq!(a.snapshot_world(), b.snapshot_world());

    // Another cell count is another participant.
    let mut other = world(2, SolverBackend::Xpbd);
    register(&mut other, sod(40));
    assert!(other.restore_world(&snap).is_err());
}

#[test]
fn check_state_refuses_foreign_or_damaged_payloads() {
    let mut w = world(1, SolverBackend::Xpbd);
    let i = register(&mut w, sod(16));
    w.try_step(q(1, 64)).expect("step");
    let good = w.participant_state(i).expect("state");
    let p = EulerFv1dParticipant::new(sod(16));
    assert_eq!(p.check_state(&good), Ok(()));

    assert!(matches!(
        p.check_state(&good[..good.len() - 1]),
        Err(StateError::Length { .. })
    ));
    let mut version = good.clone();
    version[0] ^= 1;
    assert_eq!(p.check_state(&version), Err(StateError::InvalidValue));
    let mut digest = good.clone();
    digest[4] ^= 1;
    assert_eq!(p.check_state(&digest), Err(StateError::InvalidValue));
    let mut clock = good.clone();
    clock[12..20].copy_from_slice(&(-1i64).to_le_bytes());
    assert_eq!(p.check_state(&clock), Err(StateError::InvalidValue));
    // A cell with zero density.
    let mut cell = good.clone();
    cell[HEADER..HEADER + 16].fill(0);
    assert_eq!(p.check_state(&cell), Err(StateError::InvalidValue));
    // Another configuration is another participant.
    let mut hllc = EulerConfig::godunov(gamma());
    hllc.solver = alice_physics_core::euler_fv::RiemannSolver::Hllc;
    let init: Vec<Primitive> = (0..16)
        .map(|k| if k < 8 { sod_left() } else { sod_right() })
        .collect();
    let other = EulerFv1dParticipant::new(EulerFv1d::new(hllc, q(1, 16), &init).expect("valid"));
    assert_eq!(other.check_state(&good), Err(StateError::InvalidValue));
    // The same cells on another spacing are another participant.
    let wide = EulerFv1dParticipant::new(
        EulerFv1d::new(EulerConfig::godunov(gamma()), q(1, 32), &init).expect("valid"),
    );
    assert_eq!(wide.check_state(&good), Err(StateError::InvalidValue));

    // read_state after an accepted check restores clock, counts and cells.
    let mut fresh = EulerFv1dParticipant::new(sod(16));
    fresh.read_state(&good);
    assert_eq!(fresh.solver().time(), q(1, 64));
    assert_eq!(fresh.solver().cells(), cells_of(&good).as_slice());
    assert_eq!(fresh.steps(), u64_at(&good, 36));
    assert_eq!(fresh.last_steps(), u64_at(&good, 28));
    assert!(fresh.steps() > 0);
}
