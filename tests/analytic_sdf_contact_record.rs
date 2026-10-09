//! The SDF contacts a step records (`PhysicsWorld::last_step_sdf_contacts`,
//! `SubstepCtx::sdf_contacts`) against closed-form values.
//!
//! Every expected value is derived here from the scene (positions, velocities,
//! gravity, substep width) by the integration the XPBD substep performs:
//! `v += g·h`, `x += v·h`, then the push-out `x += n·depth`, then
//! `v = (x − x_prev) / h`. Nothing is computed by calling the solver. Each check
//! counts the entries it compared and fails when that count is 0.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfContact};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use alice_physics::SolverBackend;

/// f32 field queries (the SDF boundary is `f32`) on dyadic inputs are exact;
/// on the non-dyadic tilted / rotated scenes they carry `f32` rounding.
const TOL: f64 = 1e-5;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// Plane through the origin with unit normal `n` (outside on the `n` side).
fn plane(n: [f32; 3]) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| n[0] * x + n[1] * y + n[2] * z,
        move |_, _, _| (n[0], n[1], n[2]),
    )
}

fn config(substeps: usize, gravity_y: i64) -> PhysicsConfig {
    PhysicsConfig {
        substeps,
        gravity: Vec3Fix::from_int(0, gravity_y, 0),
        ..PhysicsConfig::default()
    }
}

/// One expected record.
#[derive(Clone, Copy, Debug)]
struct Want {
    body: usize,
    collider: usize,
    point: [f64; 3],
    normal: [f64; 3],
    depth: f64,
    approach: f64,
    substep: usize,
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL
}

/// Compare the record with the expectation entry by entry; returns how many
/// entries were compared (the caller asserts it is not 0).
fn check(label: &str, got: &[SdfContact], want: &[Want]) -> usize {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {} records, expected {}: {got:?}",
        got.len(),
        want.len()
    );
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.body_index, w.body, "{label} #{k} body");
        assert_eq!(g.collider_index, w.collider, "{label} #{k} collider");
        assert_eq!(g.substep, w.substep, "{label} #{k} substep");
        let (p, n) = (f3(g.point), f3(g.normal));
        for i in 0..3 {
            assert!(
                close(p[i], w.point[i]),
                "{label} #{k} point {p:?} vs {:?}",
                w.point
            );
            assert!(
                close(n[i], w.normal[i]),
                "{label} #{k} normal {n:?} vs {:?}",
                w.normal
            );
        }
        assert!(
            close(g.depth.to_f64(), w.depth),
            "{label} #{k} depth {} vs {}",
            g.depth.to_f64(),
            w.depth
        );
        assert!(
            close(g.approach_speed.to_f64(), w.approach),
            "{label} #{k} approach_speed {} vs {}",
            g.approach_speed.to_f64(),
            w.approach
        );
    }
    want.len()
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

const FALL_X: f64 = 1.25;
const FALL_Z: f64 = -0.75;

/// A sphere of radius 1/2 falling at 4 m/s from 0.8 m onto the plane `y = 0`,
/// gravity −8, 4 substeps of `h = 1/32` (`dt = 1/8`).
fn falling_sphere() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(config(4, -8));
    w.set_sdf_collision_radius(r(1, 2));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(plane([0.0, 1.0, 0.0])),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    let mut b = RigidBody::new_dynamic(v3(FALL_X, 0.8, FALL_Z), Fix128::ONE);
    b.velocity = v3(0.0, -4.0, 0.0);
    w.add_body(b);
    w
}

/// Closed form of the falling sphere over the first step.
///
/// Before any contact, after substep `k` (0-based): `v = v0 + g·h·(k+1)`,
/// `y = y0 + Σ_{j=1..k+1} (v0 + g·h·j)·h`. With `y0 = 0.8, v0 = −4, g = −8,
/// h = 1/32`: `y = 0.6671875, 0.5265625, 0.378125` — the first overlap is in
/// substep 2, depth `0.5 − 0.378125`, approach `−(v0 + 3·g·h) = 4.75`. The
/// push-out sets `y = 0.5`, so the velocity leaving substep 2 is
/// `(0.5 − 0.5265625)/h = −0.85`; substep 3 gives `v = −0.85 + g·h = −1.1`,
/// `y = 0.5 − 1.1·h = 0.465625`: depth `0.034375`, approach `1.1`.
fn falling_sphere_want() -> Vec<Want> {
    let (y0, v0, g, h, rad) = (0.8f64, -4.0f64, -8.0f64, 1.0f64 / 32.0, 0.5f64);
    let mut y = y0;
    let mut v = v0;
    let mut out = Vec::new();
    for k in 0..4 {
        let prev = y;
        v += g * h;
        y += v * h;
        if y < rad {
            out.push(Want {
                body: 0,
                collider: 0,
                point: [FALL_X, 0.0, FALL_Z],
                normal: [0.0, 1.0, 0.0],
                depth: rad - y,
                approach: -v,
                substep: k,
            });
            y = rad;
        }
        v = (y - prev) / h;
    }
    // The derivation in the doc comment, as numbers.
    assert_eq!(out.len(), 2);
    assert!((out[0].depth - 0.121_875).abs() < 1e-12 && (out[0].approach - 4.75).abs() < 1e-12);
    assert!((out[1].depth - 0.034_375).abs() < 1e-12 && (out[1].approach - 1.1).abs() < 1e-12);
    out
}

/// Two colliders (floor `y = 0`, wall `x = 0`), one substep of `h = 1/8`, no
/// gravity. Body 0 reaches both, body 1 the floor only, body 2 is static and
/// body 3 a sensor (both overlap the floor and are never pushed).
fn two_colliders() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(config(1, 0));
    w.set_sdf_collision_radius(r(1, 2));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(plane([0.0, 1.0, 0.0])),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(plane([1.0, 0.0, 0.0])),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    let mut a = RigidBody::new_dynamic(v3(0.45, 0.6, 0.0), Fix128::ONE);
    a.velocity = v3(-0.8, -1.6, 0.0);
    w.add_body(a);
    let mut b = RigidBody::new_dynamic(v3(3.0, 0.55, 0.0), Fix128::ONE);
    b.velocity = v3(0.0, -0.8, 0.0);
    w.add_body(b);
    w.add_body(RigidBody::new_static(v3(-5.0, -0.2, 0.0)));
    w.add_body(RigidBody::new_sensor(v3(8.0, 0.2, 0.0)));
    w
}

/// After `h = 1/8`: body 0 at `(0.35, 0.4)`, floor depth `0.1` (approach 1.6,
/// point `(0.35, 0, 0)`), pushed to `y = 0.5`; then the wall, evaluated at the
/// pushed position: depth `0.15` (approach 0.8, point `(0, 0.5, 0)`). Body 1
/// at `y = 0.45`: floor depth `0.05` (approach 0.8). Order: body, then
/// collider.
fn two_colliders_want() -> Vec<Want> {
    let h = 1.0 / 8.0;
    let (ax, ay) = (0.45 - 0.8 * h, 0.6 - 1.6 * h);
    let by = 0.55 - 0.8 * h;
    vec![
        Want {
            body: 0,
            collider: 0,
            point: [ax, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            depth: 0.5 - ay,
            approach: 1.6,
            substep: 0,
        },
        Want {
            body: 0,
            collider: 1,
            point: [0.0, 0.5, 0.0],
            normal: [1.0, 0.0, 0.0],
            depth: 0.5 - ax,
            approach: 0.8,
            substep: 0,
        },
        Want {
            body: 1,
            collider: 0,
            point: [3.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            depth: 0.5 - by,
            approach: 0.8,
            substep: 0,
        },
    ]
}

/// A plane through the origin tilted to the normal `n = (3/5, 4/5, 0)`; a
/// sphere 0.7 from it moves at `−2·n + 0.5·t` (`t = (4/5, −3/5, 0)` along the
/// plane), one substep of `h = 1/4`, no gravity: distance `0.7 − 2·h = 0.2`,
/// depth `0.3`, approach `2`, point `c − n·0.2`.
fn tilted_plane() -> (PhysicsWorld, Vec<Want>) {
    let n = [0.6f64, 0.8, 0.0];
    let t = [0.8f64, -0.6, 0.0];
    let mut w = PhysicsWorld::new(config(1, 0));
    w.set_sdf_collision_radius(r(1, 2));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(plane([0.6, 0.8, 0.0])),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    let c0: Vec<f64> = (0..3).map(|i| n[i] * 0.7 + t[i] * 1.0).collect();
    let vel: Vec<f64> = (0..3).map(|i| -2.0 * n[i] + 0.5 * t[i]).collect();
    let mut b = RigidBody::new_dynamic(v3(c0[0], c0[1], c0[2]), Fix128::ONE);
    b.velocity = v3(vel[0], vel[1], vel[2]);
    w.add_body(b);
    let h = 0.25;
    let c: Vec<f64> = (0..3).map(|i| c0[i] + vel[i] * h).collect();
    let dist = 0.7 - 2.0 * h;
    let want = vec![Want {
        body: 0,
        collider: 0,
        point: [c[0] - n[0] * dist, c[1] - n[1] * dist, c[2] - n[2] * dist],
        normal: n,
        depth: 0.5 - dist,
        approach: 2.0,
        substep: 0,
    }];
    (w, want)
}

/// A plane attached to body 0 (`SdfCollider::new_dynamic`): local `y = 0`
/// with local normal `+y`, the body turned 90° about `z` so the world normal
/// is `(−1, 0, 0)`. Body 0 sits at `(5, 0, 0)` moving at `(1, 0, 0)`; body 1 at
/// `(4.4, 0, 10)` moves at `(3, 0, 0)`; one substep of `h = 1/10`, no gravity.
/// The collider follows its body to `x = 5.1`; body 1 reaches `x = 4.7`:
/// distance `0.4`, depth `0.1`, point `(5.1, 0, 10)`, approach `3`.
fn rotated_dynamic_collider() -> (PhysicsWorld, Vec<Want>) {
    let mut w = PhysicsWorld::new(config(1, 0));
    w.set_sdf_collision_radius(r(1, 2));
    let half = core::f64::consts::FRAC_1_SQRT_2;
    let mut carrier = RigidBody::new_dynamic(v3(5.0, 0.0, 0.0), Fix128::ONE);
    carrier.rotation = QuatFix::new(
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::from_f64(half),
        Fix128::from_f64(half),
    );
    carrier.velocity = v3(1.0, 0.0, 0.0);
    let carrier = w.add_body(carrier);
    let mut b = RigidBody::new_dynamic(v3(4.4, 0.0, 10.0), Fix128::ONE);
    b.velocity = v3(3.0, 0.0, 0.0);
    w.add_body(b);
    w.add_sdf_collider(SdfCollider::new_dynamic(
        Box::new(plane([0.0, 1.0, 0.0])),
        carrier,
    ));
    let h = 0.1;
    let wall = 5.0 + 1.0 * h;
    let x = 4.4 + 3.0 * h;
    let want = vec![Want {
        body: 1,
        collider: 0,
        point: [wall, 0.0, 10.0],
        normal: [-1.0, 0.0, 0.0],
        depth: 0.5 - (wall - x),
        approach: 3.0,
        substep: 0,
    }];
    (w, want)
}

// ---------------------------------------------------------------------------
// Step paths
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use alice_physics::joint::Joint;
    use alice_physics::math::Fix128;

    /// Hands everything back as it was sent (no GPU solve).
    #[derive(Default)]
    pub struct IdentityBridge {
        positions: Vec<[Fix128; 3]>,
        velocities: Vec<[Fix128; 3]>,
        constraints: Vec<alice_physics::solver::ContactConstraint>,
        body_positions: Vec<[Fix128; 3]>,
    }

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
}

/// One way of running a step.
type StepPath = (&'static str, fn(&mut PhysicsWorld, Fix128));

/// What a participant saw: `(substep index, records)` per substep.
type SeenLog = Arc<Mutex<Vec<(usize, Vec<SdfContact>)>>>;

/// The XPBD step paths compiled into this build, by name.
fn xpbd_paths() -> Vec<StepPath> {
    #[cfg_attr(
        not(any(feature = "parallel", feature = "gpu-solver-bridge")),
        allow(unused_mut)
    )]
    let mut paths: Vec<StepPath> = vec![("step", |w, dt| w.step(dt))];
    #[cfg(feature = "parallel")]
    paths.push(("step_parallel", |w, dt| w.step_parallel(dt)));
    #[cfg(feature = "gpu-solver-bridge")]
    paths.push(("step_with_bridge", |w, dt| {
        let mut b = bridge::IdentityBridge::default();
        w.step_with_bridge(&mut b, dt);
    }));
    paths
}

// ---------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------

#[test]
fn falling_sphere_records_each_push_out_with_closed_form_values_on_every_xpbd_path() {
    let want = falling_sphere_want();
    let mut compared = 0;
    for (name, step) in xpbd_paths() {
        let mut w = falling_sphere();
        step(&mut w, r(1, 8));
        compared += check(name, w.last_step_sdf_contacts(), &want);
    }
    assert!(compared > 0, "no record compared");
}

#[test]
fn two_bodies_two_colliders_are_recorded_by_body_then_collider_and_static_or_sensor_never() {
    let want = two_colliders_want();
    let mut compared = 0;
    for (name, step) in xpbd_paths() {
        let mut w = two_colliders();
        step(&mut w, r(1, 8));
        compared += check(name, w.last_step_sdf_contacts(), &want);
    }
    assert!(compared > 0, "no record compared");
}

#[test]
fn tilted_plane_contact_has_the_plane_normal_and_the_projected_point() {
    let mut compared = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, want) = tilted_plane();
        step(&mut w, r(1, 4));
        compared += check(name, w.last_step_sdf_contacts(), &want);
    }
    assert!(compared > 0, "no record compared");
}

#[test]
fn rotated_collider_attached_to_a_moving_body_is_recorded_at_the_body_pose() {
    let mut compared = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, want) = rotated_dynamic_collider();
        step(&mut w, r(1, 10));
        compared += check(name, w.last_step_sdf_contacts(), &want);
    }
    assert!(compared > 0, "no record compared");
}

#[test]
fn no_overlap_records_nothing_and_each_step_starts_empty() {
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        // Far above the plane: nothing all step.
        let mut w = falling_sphere();
        w.bodies[0].position = v3(0.0, 50.0, 0.0);
        step(&mut w, r(1, 8));
        assert_eq!(w.last_step_sdf_contacts(), &[], "{name}: far body");

        // A step with contacts, then the body is lifted away: the next step
        // records nothing (the record does not carry over).
        let mut w = falling_sphere();
        step(&mut w, r(1, 8));
        assert_eq!(w.last_step_sdf_contacts().len(), 2, "{name}: first step");
        w.bodies[0].position = v3(0.0, 50.0, 0.0);
        w.bodies[0].velocity = Vec3Fix::ZERO;
        step(&mut w, r(1, 8));
        assert_eq!(w.last_step_sdf_contacts(), &[], "{name}: second step");
        runs += 1;
    }
    assert!(runs > 0);
}

/// TGS resolves SDF overlap once per step, before its substeps and before
/// gravity acts: the record is the overlap at the head of the step, at
/// substep 0. A sphere at `y = 0.4` moving at `−3`: depth `0.1`, approach `3`.
#[test]
fn tgs_records_the_overlap_at_the_head_of_the_step() {
    let mut w = falling_sphere();
    w.config.solver_backend = SolverBackend::Tgs;
    w.bodies[0].position = v3(FALL_X, 0.4, FALL_Z);
    w.bodies[0].velocity = v3(0.0, -3.0, 0.0);
    w.step(r(1, 8));
    let want = [Want {
        body: 0,
        collider: 0,
        point: [FALL_X, 0.0, FALL_Z],
        normal: [0.0, 1.0, 0.0],
        depth: 0.1,
        approach: 3.0,
        substep: 0,
    }];
    assert_eq!(check("tgs", w.last_step_sdf_contacts(), &want), 1);
}

// ---------------------------------------------------------------------------
// Participants
// ---------------------------------------------------------------------------

/// Logs `(substep index, records seen)` for every substep it runs in.
struct Watcher {
    seen: SeenLog,
}

impl Participant for Watcher {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x5DFC)
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _h: Fix128) -> Result<(), ParticipantFault> {
        self.seen
            .lock()
            .map_err(|_| ParticipantFault::InvalidState)?
            .push((ctx.substep_index(), ctx.sdf_contacts().to_vec()));
        Ok(())
    }
    fn observe(&self, _out: &mut ObservationSink) {}
    fn write_state(&self, _out: &mut Vec<u8>) {}
    fn check_state(&self, _bytes: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _bytes: &[u8]) {}
}

fn watched(mut w: PhysicsWorld) -> (PhysicsWorld, SeenLog) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    w.add_participant(Box::new(Watcher {
        seen: Arc::clone(&seen),
    }))
    .expect("register");
    (w, seen)
}

/// Substep `i` of an XPBD step sees the contacts of substeps `0..i` of the
/// same step: the falling sphere's first contact is in substep 2, so
/// substeps 0..=2 see none and substep 3 sees that one; the next step's
/// substep 0 sees none again.
#[test]
fn participants_read_the_contacts_of_the_earlier_substeps_of_the_step() {
    let want = falling_sphere_want();
    let mut compared = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, seen) = watched(falling_sphere());
        step(&mut w, r(1, 8));
        step(&mut w, r(1, 8));
        let seen = seen.lock().expect("lock").clone();
        assert_eq!(seen.len(), 8, "{name}: 2 steps × 4 substeps");
        for (k, (index, contacts)) in seen.iter().enumerate() {
            assert_eq!(*index, k % 4, "{name}: substep index");
            let expected: &[Want] = match k {
                3 => &want[..1],
                _ if k < 4 => &[],
                4 => &[],
                _ => {
                    // Later substeps of step 2: only what step 2 recorded so far.
                    continue;
                }
            };
            compared += check(&format!("{name} ctx {k}"), contacts, expected);
        }
        // At the end of the step the world record holds the whole step.
        assert!(
            seen[7].1.len() <= w.last_step_sdf_contacts().len(),
            "{name}: ctx saw more than the step recorded"
        );
    }
    assert!(compared > 0, "no ctx record compared");
}

#[test]
fn tgs_participants_see_the_step_contacts_in_every_substep() {
    let mut w = falling_sphere();
    w.config.solver_backend = SolverBackend::Tgs;
    w.bodies[0].position = v3(FALL_X, 0.4, FALL_Z);
    w.bodies[0].velocity = v3(0.0, -3.0, 0.0);
    let (mut w, seen) = watched(w);
    w.step(r(1, 8));
    let seen = seen.lock().expect("lock").clone();
    assert!(!seen.is_empty());
    for (_, contacts) in &seen {
        assert_eq!(contacts.as_slice(), w.last_step_sdf_contacts());
        assert_eq!(contacts.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Observation only
// ---------------------------------------------------------------------------

fn bits(w: &PhysicsWorld) -> Vec<[i128; 13]> {
    w.bodies
        .iter()
        .map(|b| {
            let f = [
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
            ];
            f.map(|x| ((x.hi as i128) << 64) | x.lo as i128)
        })
        .collect()
}

/// A world whose record is read after every step (and handed to a
/// participant that stages nothing) ends bit for bit where an untouched one
/// ends, over 40 steps with the falling sphere, the two-collider scene and
/// the attached collider together.
#[test]
fn reading_the_record_does_not_change_the_simulation() {
    fn scene() -> PhysicsWorld {
        let mut w = falling_sphere();
        let mut b = RigidBody::new_dynamic(v3(-3.0, 1.5, 2.0), Fix128::ONE);
        b.velocity = v3(1.0, -2.0, 0.5);
        w.add_body(b);
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(plane([1.0, 0.0, 0.0])),
            v3(-4.0, 0.0, 0.0),
            QuatFix::IDENTITY,
        ));
        w
    }
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        let mut plain = scene();
        let (mut read, _seen) = watched(scene());
        let mut total = 0usize;
        for _ in 0..40 {
            step(&mut plain, r(1, 60));
            step(&mut read, r(1, 60));
            total += read.last_step_sdf_contacts().len();
        }
        assert!(total > 0, "{name}: the scene made no SDF contact");
        assert_eq!(bits(&plain), bits(&read), "{name}: state differs");
        runs += 1;
    }
    assert!(runs > 0);
}

/// Degenerate worlds: no body, no collider, or only static bodies. Each step
/// runs and records nothing (an empty record, not a panic).
#[test]
fn degenerate_worlds_record_nothing() {
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        let mut empty = PhysicsWorld::new(config(4, -8));
        step(&mut empty, r(1, 8));
        assert_eq!(empty.last_step_sdf_contacts(), &[], "{name}: empty world");

        let mut no_collider = falling_sphere();
        no_collider.sdf_colliders.clear();
        step(&mut no_collider, r(1, 8));
        assert_eq!(
            no_collider.last_step_sdf_contacts(),
            &[],
            "{name}: no collider"
        );

        let mut only_static = falling_sphere();
        only_static.bodies.clear();
        only_static.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
        step(&mut only_static, r(1, 8));
        assert_eq!(
            only_static.last_step_sdf_contacts(),
            &[],
            "{name}: static only"
        );
        runs += 1;
    }
    assert!(runs > 0);
    // A fresh world has an empty record before any step.
    assert_eq!(falling_sphere().last_step_sdf_contacts(), &[]);
}
