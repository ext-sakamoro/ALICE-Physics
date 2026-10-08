//! Cross-platform determinism goldens for the stepping paths and scene
//! features the other goldens do not run.
//!
//! `tests/determinism_golden_coverage.rs` lists every combination of solver
//! backend × stepping entry point and every scene feature a world step has,
//! and checks which of them a golden pins. The goldens here pin the ones the
//! older files left out: the TGS backend on `step`, `step_n` and `try_step`,
//! the entry points other than `step` on XPBD, ball joints, sleeping,
//! continuous collision, the non-default broadphases and a registered
//! participant.
//!
//! Not pinned here: TGS on the parallel and GPU-bridge entry points. Those
//! currently give the XPBD result bit for bit (the backend setting is not
//! read on those paths), and whether that is the intended behaviour is not
//! decided, so a golden would fix one answer in advance. They stay listed as
//! gaps in `tests/determinism_golden_coverage.rs`.
//!
//! Each test first checks a closed-form or physical invariant of the result
//! (so a changed digest comes with a statement of what the physics did) and
//! then pins the SHA-256 of position, velocity, rotation and angular velocity
//! of every body, in the same byte layout as `tests/determinism_golden_contacts.rs`.
//!
//! Each entry point has its own small helper and the backend is passed in as
//! a value, so the code a test reaches names only the entry point and the
//! backend it pins (the coverage check reads that code).
//!
//! # Path scene
//!
//! `PhysicsConfig::default()` with the given backend, frame damping 1 and
//! sleeping disabled. A static ground sphere of radius `10^6` whose top is
//! `y = 0`, two balls of radius 1/2 and mass 1 stacked on it (contacts), and
//! a pendulum: a static pivot and a bob of mass 1 held at distance 1 by a
//! ball joint (constraint solving). `dt = 1/60 s`, 90 frames.

#![cfg(feature = "std")]

use alice_physics::coupling_medium::{DragMedium, MEDIUM_OBS_MOMENTUM};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{SleepConfig, SolverBackend, WorldCcdConfig};
use sha2::{Digest, Sha256};

const FRAMES: usize = 90;
const GROUND_R: i64 = 1_000_000;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn write_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

/// SHA-256 of position, velocity, rotation and angular velocity of every
/// body, as hex (the layout of `tests/determinism_golden_contacts.rs`).
fn hash_world(w: &PhysicsWorld) -> String {
    let mut bytes = Vec::with_capacity(w.bodies.len() * 13 * 16);
    for b in &w.bodies {
        for f in [
            b.position.x,
            b.position.y,
            b.position.z,
            b.velocity.x,
            b.velocity.y,
            b.velocity.z,
            b.rotation.x,
            b.rotation.y,
            b.rotation.z,
            b.rotation.w,
            b.angular_velocity.x,
            b.angular_velocity.y,
            b.angular_velocity.z,
        ] {
            write_fix(&mut bytes, f);
        }
    }
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn assert_golden(scenario: &str, actual: &str, expected: &str) {
    assert_eq!(
        actual, expected,
        "\npath golden '{scenario}'\n  expected: {expected}\n  actual:   {actual}\n\
         Update the constant only for a deliberate simulation change whose invariant \
         still holds; a mismatch on one platform only is a determinism bug."
    );
}

/// Bodies of the path scene: 0 ground, 1 lower ball, 2 upper ball, 3 pivot, 4 bob.
const LOWER: usize = 1;
const UPPER: usize = 2;
const PIVOT: usize = 3;
const BOB: usize = 4;

fn path_scene(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        solver_backend: backend,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    let r = Fix128::from_int(GROUND_R);
    w.add_body_with_radius(RigidBody::new_static(v3(Fix128::ZERO, -r, Fix128::ZERO)), r);
    let half = Fix128::from_ratio(1, 2);
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(Fix128::ZERO, half, Fix128::ZERO), Fix128::ONE),
        half,
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(
            v3(Fix128::ZERO, Fix128::from_ratio(8, 5), Fix128::ZERO),
            Fix128::ONE,
        ),
        half,
    );
    let pivot = w.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 4, 0)));
    let bob = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(6, 4, 0),
        Fix128::ONE,
    ));
    w.add_joint(Joint::Ball(BallJoint::new(
        pivot,
        bob,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, 0),
    )));
    w
}

/// The path scene after [`FRAMES`] frames still has its balls resting on the
/// ground and on each other and its pendulum at length 1 (a solver that lost
/// a contact or the joint fails here before the digest is compared).
fn check_path_scene(w: &PhysicsWorld, what: &str) {
    let y = |i: usize| w.bodies[i].position.y.to_f64();
    for b in &w.bodies {
        assert!(
            b.position.x.to_f64().is_finite() && b.position.y.to_f64().is_finite(),
            "{what}: non-finite state"
        );
    }
    assert!(
        (y(LOWER) - 0.5).abs() < 0.05,
        "{what}: lower ball at y = {}",
        y(LOWER)
    );
    assert!(
        (y(UPPER) - y(LOWER) - 1.0).abs() < 0.05,
        "{what}: upper ball {} above lower {}",
        y(UPPER),
        y(LOWER)
    );
    let d = w.bodies[BOB].position - w.bodies[PIVOT].position;
    let len = d.length().to_f64();
    assert!((len - 1.0).abs() < 0.05, "{what}: pendulum length {len}");
    assert!(
        w.bodies[BOB].position.y.to_f64() < 4.0,
        "{what}: the bob did not swing down"
    );
}

fn run_step(mut w: PhysicsWorld) -> PhysicsWorld {
    for _ in 0..FRAMES {
        w.step(dt());
    }
    w
}

fn run_step_n(mut w: PhysicsWorld) -> PhysicsWorld {
    w.step_n(FRAMES, dt());
    w
}

fn run_try_step(mut w: PhysicsWorld) -> PhysicsWorld {
    for _ in 0..FRAMES {
        w.try_step(dt()).expect("no participant can fail");
    }
    w
}

#[cfg(feature = "parallel")]
fn run_step_parallel(mut w: PhysicsWorld) -> PhysicsWorld {
    for _ in 0..FRAMES {
        w.step_parallel(dt());
    }
    w
}

#[cfg(feature = "parallel")]
fn run_try_step_parallel(mut w: PhysicsWorld) -> PhysicsWorld {
    for _ in 0..FRAMES {
        w.try_step_parallel(dt()).expect("no participant can fail");
    }
    w
}

/// A bridge that hands everything back as it was sent: islands, contact
/// constraints and body positions come back unchanged and its solve
/// iterations do nothing. `step_with_bridge` / `substep_with_bridge` then run
/// their CPU stages around a contact and joint solve that changes nothing; the
/// goldens pin that path, not a GPU solver.
#[cfg(feature = "gpu-solver-bridge")]
#[derive(Default)]
struct IdentityBridge {
    positions: Vec<[Fix128; 3]>,
    velocities: Vec<[Fix128; 3]>,
    constraints: Vec<alice_physics::solver::ContactConstraint>,
    body_positions: Vec<[Fix128; 3]>,
}

#[cfg(feature = "gpu-solver-bridge")]
impl alice_physics::gpu_bridge::GpuSolverBridge for IdentityBridge {
    fn send_island(&mut self, p: &[[Fix128; 3]], v: &[[Fix128; 3]]) {
        self.positions = p.to_vec();
        self.velocities = v.to_vec();
    }
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, p: &mut [[Fix128; 3]], v: &mut [[Fix128; 3]]) {
        p.copy_from_slice(&self.positions);
        v.copy_from_slice(&self.velocities);
    }
    fn assert_bit_exact_vs_cpu(
        &self,
        _fixture: &alice_physics::gpu_bridge::DiffFixture,
    ) -> Result<(), alice_physics::gpu_bridge::GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, c: &[alice_physics::solver::ContactConstraint]) {
        self.constraints = c.to_vec();
    }
    fn send_body_state(&mut self, positions: &[[Fix128; 3]], _inv_masses: &[Fix128]) {
        self.body_positions = positions.to_vec();
    }
    fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {}
    fn recv_contact_constraints(&self, c: &mut [alice_physics::solver::ContactConstraint]) {
        c.clone_from_slice(&self.constraints);
    }
    fn recv_body_positions(&self, positions: &mut [[Fix128; 3]]) {
        positions.copy_from_slice(&self.body_positions);
    }
    fn send_joints(&mut self, _joints: &[Joint]) {}
    fn send_body_rotations(&mut self, _rotations: &[[Fix128; 4]]) {}
    fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {}
}

#[cfg(feature = "gpu-solver-bridge")]
fn run_step_with_bridge(mut w: PhysicsWorld) -> PhysicsWorld {
    let mut bridge = IdentityBridge::default();
    for _ in 0..FRAMES {
        w.step_with_bridge(&mut bridge, dt());
    }
    w
}

/// One substep per call, at the substep width of the default config.
#[cfg(feature = "gpu-solver-bridge")]
fn run_substep_with_bridge(mut w: PhysicsWorld) -> PhysicsWorld {
    let mut bridge = IdentityBridge::default();
    let substeps = w.config.substeps;
    let h = dt() / Fix128::from_int(substeps as i64);
    for _ in 0..FRAMES * substeps {
        w.substep_with_bridge(&mut bridge, h);
    }
    w
}

// ── XPBD ────────────────────────────────────────────────────────────────────

const GOLDEN_XPBD_STEP: &str = "7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39";

#[test]
fn golden_path_xpbd_step() {
    let w = run_step(path_scene(SolverBackend::Xpbd));
    check_path_scene(&w, "xpbd step");
    assert_golden("xpbd step", &hash_world(&w), GOLDEN_XPBD_STEP);
}

/// `step_n(n, dt)` is `n` calls of `step(dt)`: same digest.
#[test]
fn golden_path_xpbd_step_n() {
    let w = run_step_n(path_scene(SolverBackend::Xpbd));
    check_path_scene(&w, "xpbd step_n");
    assert_golden("xpbd step_n", &hash_world(&w), GOLDEN_XPBD_STEP);
}

/// Without participants `try_step` is `step`: same digest.
#[test]
fn golden_path_xpbd_try_step() {
    let w = run_try_step(path_scene(SolverBackend::Xpbd));
    check_path_scene(&w, "xpbd try_step");
    assert_golden("xpbd try_step", &hash_world(&w), GOLDEN_XPBD_STEP);
}

/// The batched solve orders constraints by graph colour, not by index, so it
/// is its own golden (see `PhysicsWorld::step_parallel`).
#[cfg(feature = "parallel")]
const GOLDEN_XPBD_STEP_PARALLEL: &str =
    "7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39";

#[cfg(feature = "parallel")]
#[test]
fn golden_path_xpbd_step_parallel() {
    let w = run_step_parallel(path_scene(SolverBackend::Xpbd));
    check_path_scene(&w, "xpbd step_parallel");
    assert_golden(
        "xpbd step_parallel",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_PARALLEL,
    );
}

#[cfg(feature = "parallel")]
#[test]
fn golden_path_xpbd_try_step_parallel() {
    let w = run_try_step_parallel(path_scene(SolverBackend::Xpbd));
    check_path_scene(&w, "xpbd try_step_parallel");
    assert_golden(
        "xpbd try_step_parallel",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_PARALLEL,
    );
}

#[cfg(feature = "gpu-solver-bridge")]
const GOLDEN_XPBD_STEP_WITH_BRIDGE: &str =
    "d73751a75b8b95b567fc4c212638e7fd994e1506e0abbb5af49a662032c0d9fd";

#[cfg(feature = "gpu-solver-bridge")]
#[test]
fn golden_path_xpbd_step_with_bridge() {
    let w = run_step_with_bridge(path_scene(SolverBackend::Xpbd));
    assert!(w.bodies.iter().all(|b| b.position.y.to_f64().is_finite()));
    assert_golden(
        "xpbd step_with_bridge",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_WITH_BRIDGE,
    );
}

/// `step_with_bridge` is a loop of `substep_with_bridge` over the substeps
/// of each frame: same digest.
#[cfg(feature = "gpu-solver-bridge")]
#[test]
fn golden_path_xpbd_substep_with_bridge() {
    let w = run_substep_with_bridge(path_scene(SolverBackend::Xpbd));
    assert!(w.bodies.iter().all(|b| b.position.y.to_f64().is_finite()));
    assert_golden(
        "xpbd substep_with_bridge",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_WITH_BRIDGE,
    );
}

// ── TGS on the paths that do not consult the backend ────────────────────────
//
// `step_parallel`, `try_step_parallel`, `step_with_bridge` and
// `substep_with_bridge` run the XPBD substep loop whatever
// `config.solver_backend` says (see their documentation). These goldens pin
// that relation: a TGS world stepped through one of them gives the XPBD
// digest of the same path.

#[cfg(feature = "parallel")]
#[test]
fn golden_path_tgs_step_parallel_gives_the_xpbd_bits() {
    let w = run_step_parallel(path_scene(SolverBackend::Tgs));
    check_path_scene(&w, "tgs step_parallel");
    assert_golden(
        "tgs step_parallel",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_PARALLEL,
    );
}

#[cfg(feature = "parallel")]
#[test]
fn golden_path_tgs_try_step_parallel_gives_the_xpbd_bits() {
    let w = run_try_step_parallel(path_scene(SolverBackend::Tgs));
    check_path_scene(&w, "tgs try_step_parallel");
    assert_golden(
        "tgs try_step_parallel",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_PARALLEL,
    );
}

#[cfg(feature = "gpu-solver-bridge")]
#[test]
fn golden_path_tgs_step_with_bridge_gives_the_xpbd_bits() {
    let w = run_step_with_bridge(path_scene(SolverBackend::Tgs));
    assert!(w.bodies.iter().all(|b| b.position.y.to_f64().is_finite()));
    assert_golden(
        "tgs step_with_bridge",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_WITH_BRIDGE,
    );
}

#[cfg(feature = "gpu-solver-bridge")]
#[test]
fn golden_path_tgs_substep_with_bridge_gives_the_xpbd_bits() {
    let w = run_substep_with_bridge(path_scene(SolverBackend::Tgs));
    assert!(w.bodies.iter().all(|b| b.position.y.to_f64().is_finite()));
    assert_golden(
        "tgs substep_with_bridge",
        &hash_world(&w),
        GOLDEN_XPBD_STEP_WITH_BRIDGE,
    );
}

// ── TGS ─────────────────────────────────────────────────────────────────────

const GOLDEN_TGS_STEP: &str = "cfb9f3e814a4b0a55c96019f1df345eb4aaba233fa5321e021dbcb916ff730fe";

#[test]
fn golden_path_tgs_step() {
    let w = run_step(path_scene(SolverBackend::Tgs));
    check_path_scene(&w, "tgs step");
    assert_golden("tgs step", &hash_world(&w), GOLDEN_TGS_STEP);
}

#[test]
fn golden_path_tgs_step_n() {
    let w = run_step_n(path_scene(SolverBackend::Tgs));
    check_path_scene(&w, "tgs step_n");
    assert_golden("tgs step_n", &hash_world(&w), GOLDEN_TGS_STEP);
}

#[test]
fn golden_path_tgs_try_step() {
    let w = run_try_step(path_scene(SolverBackend::Tgs));
    check_path_scene(&w, "tgs try_step");
    assert_golden("tgs try_step", &hash_world(&w), GOLDEN_TGS_STEP);
}

// ── Scene features ──────────────────────────────────────────────────────────

/// Balls dropped onto the ground with sleeping on: after they settle every
/// dynamic body is asleep, and the digest pins the state the sleep decision left.
const GOLDEN_SLEEPING: &str = "2d2fbb89f587cd27872e85f4ed2d856c72bb3f0d62a3d0d5d2f5360260ac7a86";

#[test]
fn golden_sleeping() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping: Fix128::from_ratio(9, 10),
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: 30,
        ..SleepConfig::default()
    });
    let r = Fix128::from_int(GROUND_R);
    w.add_body_with_radius(RigidBody::new_static(v3(Fix128::ZERO, -r, Fix128::ZERO)), r);
    let half = Fix128::from_ratio(1, 2);
    for x in [-3_i64, 0, 3] {
        w.add_body_with_radius(
            RigidBody::new_dynamic(
                v3(Fix128::from_int(x), Fix128::ONE, Fix128::ZERO),
                Fix128::ONE,
            ),
            half,
        );
    }
    for _ in 0..600 {
        w.step(dt());
    }
    for i in 1..w.bodies.len() {
        assert!(w.is_sleeping(i), "body {i} is still awake after 600 frames");
        assert!((w.bodies[i].position.y.to_f64() - 0.5).abs() < 0.05);
    }
    assert_golden("sleeping", &hash_world(&w), GOLDEN_SLEEPING);
}

/// A sphere of radius 1/4 at 640 m/s toward a still plate 1/32 thick, one
/// substep of 1/64 s: with continuous collision on it stops on the near face,
/// `x = 5 − 1/64 − 1/4`, and leaves with half the speed (restitution 1/2).
const GOLDEN_CONTINUOUS_COLLISION: &str =
    "afb100a521cc3d308a842b963c1e97112ac2cd824040bb2a65fdb81c846b4d7a";

#[test]
fn golden_continuous_collision() {
    use alice_physics::material::PhysicsMaterial;
    use alice_physics::shape::Shape;
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    w.set_continuous_collision(WorldCcdConfig::on());
    let plate = w.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)));
    w.set_body_shape(
        plate,
        &Shape::Box {
            half_extents: v3(
                Fix128::from_ratio(1, 64),
                Fix128::from_int(2),
                Fix128::from_int(2),
            ),
        },
    );
    let mut ball = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    ball.velocity = Vec3Fix::from_int(640, 0, 0);
    let s = w.add_body_with_radius(ball, Fix128::from_ratio(1, 4));
    let id = w.material_table.register(PhysicsMaterial::new(
        1,
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    ));
    for i in 0..w.bodies.len() {
        w.set_body_material(i, id);
    }
    let h = Fix128::from_ratio(1, 64);
    w.step(h);
    let x = w.bodies[s].position.x.to_f64();
    let vx = w.bodies[s].velocity.x.to_f64();
    assert!(
        (x - (5.0 - 1.0 / 64.0 - 0.25)).abs() < 1e-6,
        "stopped at x = {x}"
    );
    assert!((vx + 320.0).abs() < 1e-6, "left at vx = {vx}");
    for _ in 0..4 {
        w.step(h);
    }
    assert!(
        w.bodies[s].position.x.to_f64() < 5.0,
        "the sphere tunnelled through the plate"
    );
    assert_golden(
        "continuous collision",
        &hash_world(&w),
        GOLDEN_CONTINUOUS_COLLISION,
    );
}

/// Every broadphase finds the same pairs, so the path scene steps to the
/// same bits under each (`Broadphase` documents this); the non-default kinds
/// are pinned to the default's digest.
#[test]
fn golden_broadphase_dynamic_tree() {
    let mut w = path_scene(SolverBackend::Xpbd);
    w.set_broadphase(Broadphase::DynamicTree);
    let w = run_step(w);
    check_path_scene(&w, "dynamic tree");
    assert_golden("broadphase dynamic tree", &hash_world(&w), GOLDEN_XPBD_STEP);
}

#[test]
fn golden_broadphase_hybrid() {
    let mut w = path_scene(SolverBackend::Xpbd);
    w.set_broadphase(Broadphase::Hybrid);
    let w = run_step(w);
    check_path_scene(&w, "hybrid");
    assert_golden("broadphase hybrid", &hash_world(&w), GOLDEN_XPBD_STEP);
}

/// Two bodies of masses 2 and 4 coupled to a drag medium of mass 8, no
/// gravity, no damping, no contact: the total momentum of bodies and medium
/// is conserved exactly (power-of-two masses), and the digest pins the
/// exchange.
const GOLDEN_PARTICIPANT: &str = "3de49332adc74a203ca1c416e3bc162bc432aa692ff5b61a58f5aa13c14bd684";

#[test]
fn golden_participant() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    let mut a = RigidBody::new_dynamic(Vec3Fix::from_int(-2, 0, 0), Fix128::from_int(2));
    a.velocity = Vec3Fix::from_int(3, -1, 0);
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(4));
    b.velocity = Vec3Fix::from_int(-1, 2, 1);
    let ia = w.add_body(a);
    let ib = w.add_body(b);
    let mut medium = DragMedium::new(Fix128::from_int(8), Vec3Fix::ZERO).expect("medium");
    medium
        .couple(ia, Fix128::from_ratio(1, 2))
        .expect("couple a");
    medium
        .couple(ib, Fix128::from_ratio(1, 4))
        .expect("couple b");
    let index = w.add_participant(Box::new(medium)).expect("register");
    let momentum = |w: &PhysicsWorld| {
        w.bodies[ia].velocity * Fix128::from_int(2) + w.bodies[ib].velocity * Fix128::from_int(4)
    };
    let before = momentum(&w);
    for _ in 0..FRAMES {
        w.try_step(dt()).expect("step");
    }
    let Some(alice_physics::world_participant::Observed::Exact(sink)) =
        w.observe_participant(index)
    else {
        panic!("medium observation missing");
    };
    let get = |ch: u32| {
        sink.values()
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, v)| *v)
            .expect("momentum channel")
    };
    let medium_momentum = Vec3Fix::new(
        get(MEDIUM_OBS_MOMENTUM),
        get(MEDIUM_OBS_MOMENTUM + 1),
        get(MEDIUM_OBS_MOMENTUM + 2),
    );
    let total = momentum(&w) + medium_momentum;
    assert_eq!(total, before, "body + medium momentum is not conserved");
    assert_golden("participant", &hash_world(&w), GOLDEN_PARTICIPANT);
}
