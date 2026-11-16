//! `YeeGridParticipant` (`YEEG`): the Yee lattice advancing inside the world's
//! substep loop on its own fixed step (`StepRule::Fixed`).
//!
//! Oracles:
//! - **Cavity resonance (closed form)**: the 4×4 PEC cavity driven at
//!   `m = n = 2` with `S = 1/2` has `2cos(ωΔt) = 2 − S²λ = 1`, so it returns to
//!   its initial lattice every 6 steps and is inverted after 3. Through the
//!   world this must hold with the period measured in world seconds, and the
//!   period ties to the continuum resonance `f = (c/2)√((m/a)² + (n/b)²)` as
//!   the exact rational `T²f² = 9/8` (`c/Δx = S/step`).
//! - **World path = direct stepping**: `k` frames of `s` substeps of width
//!   `n·step` give the lattice of `k·s·n` direct `YeeGrid::step` calls, bit for
//!   bit, on a lattice with a graded PML, a material block, currents and a
//!   charge, under both solver backends.
//! - **Refusal**: a substep that `step` does not divide is refused with
//!   `StepError::Rule` and the world (bodies, participant, snapshot bytes) is
//!   unchanged.
//! - **Snapshot**: snapshot, branch and resume equals an unbroken run;
//!   payloads of another version, digest, lattice shape or length are refused.

use alice_physics_core::maxwell_fdtd::{
    Absorber, Component, Material, MaterialMap, YeeGrid, YeeGridParticipant, YeeParticipantError,
    COURANT_3D, YEE_GRID_OBS_MAX_FIELD, YEE_GRID_OBS_STEPS, YEE_GRID_OBS_TIME,
    YEE_GRID_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{
    Observed, Participant, RegisterError, StateError, StepError, StepRule,
};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

/// Bytes of the participant header before the lattice state.
const HEADER: usize = 4 + 8 + 16 + 8;

fn q(num: i64, den: i64) -> Fix128 {
    Fix128::from_int(num) / Fix128::from_int(den)
}

/// World seconds of one lattice step in these tests (dyadic, exact).
fn step() -> Fix128 {
    q(1, 64)
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

/// Mode `m = 2` of a 4-cell PEC side: `sin(π·2·a/4)` at the 5 nodes.
const MODE_4_2: [i64; 5] = [0, 1, 0, -1, 0];

/// The `Ez` TM cavity of 4×4 cells driven at `m = n = 2`, Courant `1/2`.
fn cavity() -> YeeGrid {
    let mut g = YeeGrid::new(4, 4, 1, q(1, 2));
    for (a, sa) in MODE_4_2.iter().enumerate() {
        for (b, sb) in MODE_4_2.iter().enumerate() {
            g.set(Component::Ez, a, b, 0, Fix128::from_int(sa * sb));
        }
    }
    g
}

/// A lattice that exercises every part of the state: graded PML, a material
/// block, field samples, currents and a charge.
fn full_grid() -> YeeGrid {
    let mut map = MaterialMap::vacuum(12, 10, 8);
    map.fill(
        [5, 4, 3],
        [7, 6, 5],
        Material::new(q(9, 4), Fix128::ONE, q(1, 8)),
    );
    let mut g = YeeGrid::new_with_absorber(
        12,
        10,
        8,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 2, 2],
            sigma_max: Fix128::from_int(3),
        },
    )
    .with_materials(&map)
    .expect("materials");
    g.set(Component::Ez, 4, 5, 4, Fix128::ONE);
    g.set(Component::Hx, 6, 4, 4, q(-3, 7));
    g.set_current(Component::Ex, 4, 4, 4, q(1, 3));
    g.set_current(Component::Ey, 8, 5, 3, q(-2, 5));
    g.set_charge(3, 3, 3, q(5, 11));
    g
}

fn lattice_bytes(g: &YeeGrid) -> Vec<u8> {
    let mut out = Vec::new();
    g.write_state(&mut out);
    out
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// `(clock, steps, lattice bytes)` of participant `index`.
fn state_of(w: &PhysicsWorld, index: usize) -> (Fix128, u64, Vec<u8>) {
    let s = w.participant_state(index).expect("participant");
    let time = Fix128 {
        hi: u64_at(&s, 12) as i64,
        lo: u64_at(&s, 20),
    };
    (time, u64_at(&s, 28), s[HEADER..].to_vec())
}

fn observed(w: &PhysicsWorld, index: usize) -> Vec<(u32, Fix128)> {
    match w.observe_participant(index).expect("participant") {
        Observed::Exact(sink) => sink.values().to_vec(),
        Observed::Undecided => panic!("observation undecided"),
    }
}

fn register(w: &mut PhysicsWorld, g: YeeGrid) -> usize {
    let p = YeeGridParticipant::new(g, step()).expect("participant");
    assert_eq!(p.kind(), YEE_GRID_PARTICIPANT_KIND);
    assert_eq!(p.step_rule(), StepRule::Fixed(step()));
    w.add_participant(Box::new(p)).expect("register")
}

#[test]
fn the_cavity_returns_every_six_lattice_steps_through_the_world() {
    // 2 substeps of 3 lattice steps each: one frame is one period.
    let mut w = world(2, SolverBackend::Xpbd);
    let i = register(&mut w, cavity());
    let initial = lattice_bytes(&cavity());
    let period = Fix128::from_int(6) * step();
    for frame in 1..=24_i64 {
        w.try_step(period).expect("step");
        let (time, steps, lattice) = state_of(&w, i);
        assert_eq!(
            lattice, initial,
            "frame {frame}: the lattice must return bit for bit"
        );
        assert_eq!(steps, 6 * frame as u64);
        assert_eq!(time, Fix128::from_int(frame) * period);
    }
    // The return would be vacuous on a lattice that never moved or had died.
    let obs = observed(&w, i);
    assert!(
        obs.contains(&(YEE_GRID_OBS_MAX_FIELD, Fix128::ONE)),
        "{obs:?}"
    );
    assert!(
        obs.contains(&(YEE_GRID_OBS_STEPS, Fix128::from_int(144))),
        "{obs:?}"
    );

    // Continuum resonance in world units: c/Δx = S/step, so
    // f² = (1/4)((2/4)² + (2/4)²)·(S/step)² and T = 6·step gives T²f² = 9/8.
    let s = q(1, 2);
    let c_over_dx = s / step();
    let f_sq = q(1, 4) * (q(2, 4) * q(2, 4) + q(2, 4) * q(2, 4)) * c_over_dx * c_over_dx;
    assert_eq!(f_sq, Fix128::from_int(128));
    assert_eq!(period * period * f_sq, q(9, 8));
}

#[test]
fn half_a_period_through_the_world_inverts_the_mode() {
    // One substep of 3 lattice steps: half a period.
    let mut w = world(1, SolverBackend::Xpbd);
    let i = register(&mut w, cavity());
    w.try_step(Fix128::from_int(3) * step()).expect("step");
    let mut inverted = cavity();
    for (a, sa) in MODE_4_2.iter().enumerate() {
        for (b, sb) in MODE_4_2.iter().enumerate() {
            inverted.set(Component::Ez, a, b, 0, Fix128::from_int(-(sa * sb)));
        }
    }
    let (time, steps, lattice) = state_of(&w, i);
    assert_eq!(steps, 3);
    assert_eq!(time, Fix128::from_int(3) * step());
    // Hx / Hy carry the energy at the half period, so compare Ez only.
    let back = YeeGrid::read_state(&lattice).expect("lattice");
    for a in 0..5 {
        for b in 0..5 {
            assert_eq!(
                back.get(Component::Ez, a, b, 0),
                inverted.get(Component::Ez, a, b, 0),
                "Ez({a},{b})"
            );
        }
    }
    assert!(back.get(Component::Ez, 1, 1, 0) == -Fix128::ONE);
}

#[test]
fn the_world_path_equals_direct_stepping_bit_for_bit() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        // 4 substeps of 2 lattice steps, 5 frames: 40 lattice steps.
        let mut w = world(4, backend);
        // A body in the world shows the participant runs beside the solver.
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let i = register(&mut w, full_grid());
        let dt = Fix128::from_int(8) * step();
        for _ in 0..5 {
            w.try_step(dt).expect("step");
        }
        let mut direct = full_grid();
        for _ in 0..40 {
            direct.step();
        }
        let (time, steps, lattice) = state_of(&w, i);
        assert_eq!(steps, 40, "{backend:?}");
        assert_eq!(time, Fix128::from_int(5) * dt, "{backend:?}");
        assert_eq!(
            lattice,
            lattice_bytes(&direct),
            "{backend:?}: lattice differs"
        );
        // Not vacuous: the field moved away from its start.
        assert_ne!(lattice, lattice_bytes(&full_grid()), "{backend:?}");
        let obs = observed(&w, i);
        assert!(obs.contains(&(YEE_GRID_OBS_TIME, Fix128::from_int(5) * dt)));
        assert!(obs.contains(&(YEE_GRID_OBS_MAX_FIELD, direct.max_abs_field())));
    }
}

#[test]
fn a_substep_the_step_does_not_divide_is_refused_and_nothing_moves() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        ..PhysicsConfig::default()
    });
    // Gravity on: a step that ran would move this body.
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO),
        Fix128::ONE,
    ));
    let i = register(&mut w, full_grid());
    let before = w.snapshot_world();
    let state = w.participant_state(i);
    // 1.5 lattice steps per substep.
    let h = q(3, 128);
    match w.try_step(h) {
        Err(StepError::Rule {
            index,
            error: RegisterError::StepRuleMismatch { substep, step: s },
        }) => {
            assert_eq!(index, i);
            assert_eq!(substep, h);
            assert_eq!(s, step());
        }
        other => panic!("expected StepError::Rule, got {other:?}"),
    }
    assert_eq!(w.snapshot_world(), before, "the world must be unchanged");
    assert_eq!(w.participant_state(i), state);
    // The same world steps once the width divides.
    w.try_step(Fix128::from_int(2) * step()).expect("step");
    assert_eq!(state_of(&w, i).1, 2);
}

#[test]
fn a_non_positive_step_is_refused() {
    for s in [Fix128::ZERO, -step()] {
        assert_eq!(
            YeeGridParticipant::new(cavity(), s).err(),
            Some(YeeParticipantError::NonPositiveStep)
        );
    }
}

#[test]
fn snapshot_branch_and_resume_equals_an_unbroken_run() {
    let dt = Fix128::from_int(4) * step();
    let mut a = world(2, SolverBackend::Xpbd);
    let ia = register(&mut a, full_grid());
    for _ in 0..3 {
        a.try_step(dt).expect("step");
    }
    let snap = a.snapshot_world();
    let mut b = world(2, SolverBackend::Xpbd);
    let ib = register(&mut b, full_grid());
    b.restore_world(&snap).expect("restore");
    assert_eq!(a.participant_state(ia), b.participant_state(ib));
    for _ in 0..4 {
        a.try_step(dt).expect("step");
        b.try_step(dt).expect("step");
    }
    assert_eq!(a.participant_state(ia), b.participant_state(ib));
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    assert_eq!(state_of(&a, ia).1, 14 * 2);

    // A lattice of another shape is another participant: its payload is refused.
    let mut other = world(2, SolverBackend::Xpbd);
    register(&mut other, cavity());
    assert!(other.restore_world(&snap).is_err());
}

#[test]
fn check_state_refuses_foreign_or_damaged_payloads() {
    let p = YeeGridParticipant::new(full_grid(), step()).expect("participant");
    let mut good = Vec::new();
    p.write_state(&mut good);
    assert_eq!(p.check_state(&good), Ok(()));

    assert!(matches!(
        p.check_state(&good[..HEADER - 1]),
        Err(StateError::Length {
            expected: HEADER,
            ..
        })
    ));
    let mut version = good.clone();
    version[0] ^= 1;
    assert_eq!(p.check_state(&version), Err(StateError::InvalidValue));
    let mut digest = good.clone();
    digest[4] ^= 1;
    assert_eq!(p.check_state(&digest), Err(StateError::InvalidValue));
    // A lattice that does not read back.
    assert_eq!(
        p.check_state(&good[..good.len() - 1]),
        Err(StateError::InvalidValue)
    );
    // The same header over a lattice of another shape.
    let mut swapped = good[..HEADER].to_vec();
    swapped.extend(lattice_bytes(&cavity()));
    assert_eq!(p.check_state(&swapped), Err(StateError::InvalidValue));
    // Another step is another participant.
    let q2 = YeeGridParticipant::new(full_grid(), q(1, 32)).expect("participant");
    assert_eq!(q2.check_state(&good), Err(StateError::InvalidValue));

    // read_state after an accepted check restores clock, count and lattice.
    let mut w = world(1, SolverBackend::Xpbd);
    let i = register(&mut w, full_grid());
    w.try_step(Fix128::from_int(3) * step()).expect("step");
    let moved = w.participant_state(i).expect("state");
    let mut fresh = YeeGridParticipant::new(full_grid(), step()).expect("participant");
    assert_eq!(fresh.check_state(&moved), Ok(()));
    fresh.read_state(&moved);
    assert_eq!(fresh.steps(), 3);
    assert_eq!(fresh.time(), Fix128::from_int(3) * step());
    assert_eq!(fresh.step(), step());
    assert_eq!(lattice_bytes(fresh.grid()), moved[HEADER..].to_vec());
}
