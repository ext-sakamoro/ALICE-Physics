//! `TransientFemParticipant` (`FEMD`): a transient P1 FEM body (average
//! acceleration Newmark) advancing inside the world's substep loop on its own
//! fixed step (`StepRule::Fixed(dt)`).
//!
//! Oracles:
//! - **Step response (closed form)**: one tetrahedron with a single free
//!   degree of freedom under a suddenly applied load follows
//!   `u_s(1 − cos nθ)`, `tan(θ/2) = ω dt/2`, `ω = √(K/M)` from the element
//!   geometry, through the world; the peak is `2 u_s` (the dynamic
//!   amplification factor of a step load). `u_s` comes from the static solver.
//! - **World path = direct stepping**: `k` substeps of `n·dt` give the fields
//!   of `k·n` direct `TransientSolver::step` calls bit for bit, under both
//!   solver backends, on a 2×1×1 box with a fixed face and a load.
//! - **Refusal**: a substep `dt` does not divide is refused with
//!   `StepError::Rule` and the world is unchanged.
//! - **Snapshot**: snapshot, branch and resume equals an unbroken run;
//!   payloads of another length, version, digest, a negative clock or a
//!   prescribed degree of freedom off its value are refused.

use alice_physics_core::dynamic_fem::{
    DynamicsConfig, MassLumping, TransientFemParticipant, TransientSolver,
    TRANSIENT_FEM_OBS_ITERATIONS, TRANSIENT_FEM_OBS_STEPS, TRANSIENT_FEM_OBS_TIME,
    TRANSIENT_FEM_PARTICIPANT_KIND,
};
use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics_core::world_participant::{
    Observed, Participant, RegisterError, StateError, StepError, StepRule,
};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const HEADER: usize = 4 + 8 + 16 + 8 + 8;
const DT_SHIFT: u32 = 23;
const DENSITY_F64: f64 = 1.0 / 1_073_741_824.0;
const CORNER_TET_VOLUME: f64 = 0.5;
const CORNER_TET_GRAD3_Z: f64 = 1.0 / 3.0;

fn two_pow_neg(shift: u32) -> Fix128 {
    Fix128::from_raw(0, 1u64 << (64 - shift))
}

fn dt() -> Fix128 {
    two_pow_neg(DT_SHIFT)
}

fn corner_tet() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]);
    mesh.vertices.push([1.0, 0.0, 0.0]);
    mesh.vertices.push([0.0, 1.0, 0.0]);
    mesh.vertices.push([0.0, 0.0, 3.0]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh
}

/// Two unit cubes along x, six Kuhn tetrahedra each.
fn box_2x1x1() -> SdfTetMesh {
    let (nx, ny, nz) = (2usize, 1usize, 1usize);
    let idx = |i: usize, j: usize, k: usize| (i + j * (nx + 1) + k * (nx + 1) * (ny + 1)) as u32;
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for i in 0..nx {
        for path in PATHS {
            let mut s = [0usize; 3];
            let mut corners = [idx(i, 0, 0); 4];
            for (n, axis) in path.into_iter().enumerate() {
                s[axis] = 1;
                corners[n + 1] = idx(i + s[0], s[1], s[2]);
            }
            mesh.tets.push(Tetrahedron { vertices: corners });
        }
    }
    mesh
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(Fix128::from_int(3500), Fix128::ZERO).expect("E > 0, ν = 0")
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(20_000, Fix128::from_f64(1e-12)).expect("valid solver config")
}

fn dynamics() -> DynamicsConfig {
    DynamicsConfig::try_new(two_pow_neg(30), dt(), MassLumping::Lumped).expect("valid dynamics")
}

fn single_dof_boundary(load_n: f64) -> BoundaryConditions {
    let mut b = BoundaryConditions::new();
    for v in 0..3u32 {
        b.fix(v);
    }
    b.prescribe(3, Axis::Y, Fix128::ZERO);
    b.prescribe(3, Axis::Z, Fix128::ZERO);
    b.add_load(3, Axis::X, Fix128::from_f64(load_n));
    b
}

fn box_boundary() -> BoundaryConditions {
    let mut b = BoundaryConditions::new();
    // the x = 0 face held, a load on the far corner
    for v in [0u32, 3, 6, 9] {
        b.fix(v);
    }
    b.add_load(11, Axis::Y, Fix128::from_int(-2));
    b.add_load(5, Axis::X, Fix128::ONE);
    b
}

fn single_dof_solver() -> TransientSolver {
    TransientSolver::new(
        &corner_tet(),
        &material(),
        &single_dof_boundary(1.0),
        &dynamics(),
        &solver_config(),
    )
    .expect("well posed")
}

fn box_solver() -> TransientSolver {
    TransientSolver::new(
        &box_2x1x1(),
        &material(),
        &box_boundary(),
        &dynamics(),
        &solver_config(),
    )
    .expect("well posed")
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

fn register(w: &mut PhysicsWorld, s: TransientSolver) -> usize {
    let p = TransientFemParticipant::new(s);
    assert_eq!(p.kind(), TRANSIENT_FEM_PARTICIPANT_KIND);
    assert_eq!(p.step_rule(), StepRule::Fixed(dt()));
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

/// `(u, v, a)` decoded from a participant payload.
fn fields_of(state: &[u8]) -> (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>) {
    let ndof = (state.len() - HEADER) / 48;
    let f = |k: usize| {
        (0..ndof)
            .map(|d| fix_at(state, HEADER + (k * ndof + d) * 16))
            .collect()
    };
    (f(0), f(1), f(2))
}

/// `cos(n θ)` for `n = 1..=count`, `tan(θ/2) = Ω/2`, by the Chebyshev
/// recurrence from `cos θ = (1 − Ω²/4)/(1 + Ω²/4)`.
fn cos_multiples(omega_dt: f64, count: usize) -> Vec<f64> {
    let q = omega_dt * omega_dt / 4.0;
    let c1 = (1.0 - q) / (1.0 + q);
    let mut out = Vec::with_capacity(count);
    let (mut prev, mut cur) = (1.0_f64, c1);
    for _ in 0..count {
        out.push(cur);
        let next = 2.0 * c1 * cur - prev;
        prev = cur;
        cur = next;
    }
    out
}

#[test]
fn a_single_free_dof_follows_the_closed_form_step_response_through_the_world() {
    let e_mpa = 3500.0;
    let k = CORNER_TET_VOLUME * (e_mpa / 2.0) * CORNER_TET_GRAD3_Z * CORNER_TET_GRAD3_Z;
    let m = DENSITY_F64 * CORNER_TET_VOLUME / 4.0;
    let u_s = linear_elastic_fem::solve(
        &corner_tet(),
        &material(),
        &single_dof_boundary(1.0),
        &solver_config(),
    )
    .expect("static scene")
    .displacements[3][0]
        .to_f64();
    assert!(
        ((u_s - 1.0 / k) * k).abs() <= 1e-9,
        "static u_s {u_s} vs 1/K {}",
        1.0 / k
    );
    let dt_f = 1.0 / f64::from(1u32 << DT_SHIFT);
    let omega_dt = (k / m).sqrt() * dt_f;
    // 2 substeps of 2 Newmark steps per frame: 4 steps per frame.
    let frames = (6.0 * core::f64::consts::TAU / omega_dt / 4.0) as usize;
    let reference = cos_multiples(omega_dt, frames * 4);
    let mut w = world(2, SolverBackend::Xpbd);
    let i = register(&mut w, single_dof_solver());
    let (mut worst, mut peak) = (0.0_f64, 0.0_f64);
    for f in 1..=frames {
        w.try_step(Fix128::from_int(4) * dt()).expect("step");
        let state = w.participant_state(i).expect("state");
        let got = fields_of(&state).0[9].to_f64();
        let want = u_s * (1.0 - reference[4 * f - 1]);
        worst = worst.max((got - want).abs() / u_s);
        peak = peak.max(got / u_s);
    }
    eprintln!("FEMD step response: {frames} frames, worst {worst:.3e} of u_s, peak {peak:.6} u_s");
    assert!(worst <= 1e-6, "left u_s(1 − cos nθ) by {worst:.3e} of u_s");
    assert!(
        (peak - 2.0).abs() <= 1e-2,
        "dynamic amplification {peak:.6}, closed form 2"
    );
    let obs = observed(&w, i);
    assert_eq!(
        channel(&obs, TRANSIENT_FEM_OBS_STEPS),
        Fix128::from_int(4 * frames as i64)
    );
    assert_eq!(
        channel(&obs, TRANSIENT_FEM_OBS_TIME),
        Fix128::from_int(4 * frames as i64) * dt()
    );
    assert!(channel(&obs, TRANSIENT_FEM_OBS_ITERATIONS) > Fix128::ZERO);
}

#[test]
fn the_world_path_equals_direct_stepping_bit_for_bit() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        // 4 substeps (a power of two: the world forms `h` as `dt · (1/substeps)`,
        // which is exact only then) of 2 Newmark steps, 3 frames.
        let mut w = world(4, backend);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let i = register(&mut w, box_solver());
        for _ in 0..3 {
            w.try_step(Fix128::from_int(8) * dt()).expect("step");
        }
        let mut direct = box_solver();
        for _ in 0..24 {
            direct.step().expect("converges");
        }
        let state = w.participant_state(i).expect("state");
        let (u, v, a) = fields_of(&state);
        assert_eq!(u, direct.displacements(), "{backend:?}: u");
        assert_eq!(v, direct.velocities(), "{backend:?}: v");
        assert_eq!(a, direct.accelerations(), "{backend:?}: a");
        assert_eq!(u64_at(&state, 28), 24);
        assert_ne!(
            u,
            box_solver().displacements(),
            "{backend:?}: the body must have moved"
        );
    }
}

#[test]
fn a_substep_dt_does_not_divide_is_refused_and_nothing_moves() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO),
        Fix128::ONE,
    ));
    let i = register(&mut w, box_solver());
    let before = w.snapshot_world();
    let h = dt() + dt().half();
    match w.try_step(h) {
        Err(StepError::Rule {
            index,
            error: RegisterError::StepRuleMismatch { substep, step },
        }) => {
            assert_eq!((index, substep, step), (i, h, dt()));
        }
        other => panic!("expected StepError::Rule, got {other:?}"),
    }
    assert_eq!(w.snapshot_world(), before);
    w.try_step(Fix128::from_int(2) * dt()).expect("step");
    assert_eq!(u64_at(&w.participant_state(i).expect("state"), 28), 2);
}

#[test]
fn snapshot_branch_and_resume_equals_an_unbroken_run() {
    let h = Fix128::from_int(4) * dt();
    let mut a = world(2, SolverBackend::Xpbd);
    let ia = register(&mut a, box_solver());
    for _ in 0..3 {
        a.try_step(h).expect("step");
    }
    let snap = a.snapshot_world();
    let mut b = world(2, SolverBackend::Xpbd);
    let ib = register(&mut b, box_solver());
    b.restore_world(&snap).expect("restore");
    assert_eq!(a.participant_state(ia), b.participant_state(ib));
    for _ in 0..3 {
        a.try_step(h).expect("step");
        b.try_step(h).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());

    // Another body is another participant.
    let mut other = world(2, SolverBackend::Xpbd);
    register(&mut other, single_dof_solver());
    assert!(other.restore_world(&snap).is_err());
}

#[test]
fn check_state_refuses_foreign_or_damaged_payloads() {
    let mut w = world(1, SolverBackend::Xpbd);
    let i = register(&mut w, box_solver());
    w.try_step(Fix128::from_int(3) * dt()).expect("step");
    let good = w.participant_state(i).expect("state");
    let p = TransientFemParticipant::new(box_solver());
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
    // vertex 0 is fixed: its x displacement must stay 0
    let mut held = good.clone();
    held[HEADER..HEADER + 8].copy_from_slice(&1i64.to_le_bytes());
    assert_eq!(p.check_state(&held), Err(StateError::InvalidValue));
    // another load is another participant
    let mut b = box_boundary();
    b.add_load(5, Axis::Z, Fix128::ONE);
    let other = TransientFemParticipant::new(
        TransientSolver::new(&box_2x1x1(), &material(), &b, &dynamics(), &solver_config())
            .expect("well posed"),
    );
    assert_eq!(other.check_state(&good), Err(StateError::InvalidValue));

    let mut fresh = TransientFemParticipant::new(box_solver());
    fresh.read_state(&good);
    let (u, v, a) = fields_of(&good);
    assert_eq!(fresh.solver().displacements(), u.as_slice());
    assert_eq!(fresh.solver().velocities(), v.as_slice());
    assert_eq!(fresh.solver().accelerations(), a.as_slice());
    assert_eq!(fresh.steps(), 3);
    assert_eq!(fresh.time(), Fix128::from_int(3) * dt());
}
