//! A body whose rotation is given as a non-unit quaternion (norm 2, 1/2, 1.3)
//! is stored by the world as the unit quaternion it stands for, whichever way
//! the rotation came in:
//!
//! - `RigidBody::set_rotation` on a body already in the world;
//! - `PhysicsWorld::add_body` / `add_body_with_radius` of a body built with the
//!   non-unit rotation (`RigidBody::with_rotation` stores it as given);
//! - a direct write to the public `rotation` / `prev_rotation` fields between
//!   steps, which the next step brings to unit length before reading it.
//!
//! `q v q*` scales `v` by `|q|^2`, so a stored non-unit quaternion resized every
//! rotation applied from it: the world inverse inertia (`|q|^4`), joint anchors,
//! the frame an attached SDF collider is evaluated in, the lidar mount and the
//! IMU body frame, and the axes drawn by the debug renderer. A step leaves the
//! rotation unnormalized when the angular velocity is zero and the inertia is
//! isotropic, so the scenes here use exactly that case.
//!
//! # Expected values
//!
//! Each scene is run once with the scaled rotation `s·q` and once with
//! `normalize(s·q)`, and the observations must agree bit for bit: the world
//! stores `normalize(s·q)` for the first, and a quaternion within `2^-32` of
//! unit length is stored unchanged, so both runs start from the same bits.
//! Against the original unit `q` (which differs from `normalize(s·q)` by the
//! rounding of the normalization) the observations agree to `1e-9`, and to
//! `1e-4` on the SDF path, which evaluates the field in `f32`.
//!
//! Observations made before any step (`set_rotation` and `add_body` entries)
//! check the entry itself; observations after one step check all three.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::debug_render::{debug_draw_world, DebugDrawData, DebugDrawFlags};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sensors::{Imu, Lidar};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

const NEAR: f64 = 1e-9;
const NEAR_F32: f64 = 1e-4;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scaled(q: QuatFix, s: Fix128) -> QuatFix {
    QuatFix::new(q.x * s, q.y * s, q.z * s, q.w * s)
}

/// A turn about a slanted axis, so every component of the quaternion is non-zero.
fn turn() -> QuatFix {
    QuatFix::from_axis_angle(v3(1.0, 2.0, 3.0).normalize(), fx(0.7))
}

const SCALES: [f64; 3] = [2.0, 0.5, 1.3];

/// How the rotation of the body under test reaches the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    SetRotation,
    AddBody,
    AddBodyWithRadius,
    Field,
}

const ENTRIES: [Entry; 4] = [
    Entry::SetRotation,
    Entry::AddBody,
    Entry::AddBodyWithRadius,
    Entry::Field,
];

/// Add `body` to `w` with rotation `q` through `entry`; returns its index.
fn enter(w: &mut PhysicsWorld, body: RigidBody, q: QuatFix, entry: Entry) -> usize {
    match entry {
        Entry::SetRotation => {
            let i = w.add_body(body);
            w.bodies[i].set_rotation(q);
            i
        }
        Entry::AddBody => w.add_body(body.with_rotation(q)),
        Entry::AddBodyWithRadius => w.add_body_with_radius(body.with_rotation(q), Fix128::ONE),
        Entry::Field => {
            let i = w.add_body(body);
            w.bodies[i].rotation = q;
            w.bodies[i].prev_rotation = q;
            i
        }
    }
}

/// Observations of one scene: the values the scene reads, in order, and the
/// tolerance against the unit-`q` run.
struct Obs {
    values: Vec<Fix128>,
    tol: f64,
}

impl Obs {
    fn new(tol: f64) -> Self {
        Self {
            values: Vec::new(),
            tol,
        }
    }

    fn v(&mut self, v: Vec3Fix) {
        self.values.extend([v.x, v.y, v.z]);
    }

    fn s(&mut self, s: Fix128) {
        self.values.push(s);
    }
}

/// Run `scene` for every entry and scale, and compare the observations of
/// `s·q` with those of `normalize(s·q)` (bit for bit) and of `q` (to the
/// scene's tolerance). The unit run must observe something non-zero, so the
/// comparison is not vacuous.
fn check(name: &str, scene: fn(QuatFix, Entry) -> Obs) {
    let q = turn();
    for entry in ENTRIES {
        let unit = scene(q, entry);
        assert!(
            unit.values.iter().any(|x| !x.is_zero()),
            "{name} {entry:?}: every observation is zero"
        );
        for s in SCALES {
            let qs = scaled(q, fx(s));
            let got = scene(qs, entry);
            let want = scene(qs.normalize(), entry);
            assert_eq!(
                got.values, want.values,
                "{name} {entry:?} |q| = {s}: differs from the normalized rotation"
            );
            assert_eq!(got.values.len(), unit.values.len());
            for (k, (a, b)) in got.values.iter().zip(&unit.values).enumerate() {
                let d = (*a - *b).abs().to_f64();
                assert!(
                    d < unit.tol,
                    "{name} {entry:?} |q| = {s} observation {k}: {} vs unit {} (|d| = {d:e})",
                    a.to_f64(),
                    b.to_f64()
                );
            }
        }
    }
}

fn zero_gravity() -> SolverConfig {
    SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    }
}

/// `apply_impulse_at` off the centre of mass: the angular response
/// `I⁻¹ (r × J)` goes through the stored rotation twice.
fn impulse_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(zero_gravity());
    let i = enter(
        &mut w,
        RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE),
        q,
        entry,
    );
    let mut o = Obs::new(NEAR);
    let kick = |w: &PhysicsWorld, o: &mut Obs| {
        let mut b = w.bodies[i];
        b.apply_impulse_at(v3(0.0, 0.0, 1.5), v3(1.0, 0.5, 0.0));
        o.v(b.angular_velocity);
        o.v(b.velocity);
    };
    if entry != Entry::Field {
        kick(&w, &mut o);
    }
    w.step(dt());
    kick(&w, &mut o);
    o.v(w.bodies[i].position);
    o
}

/// A ball joint between a static anchor and the rotated body, one step.
fn ball_joint_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let a = w.add_body(RigidBody::new_static(v3(0.0, 2.0, 0.0)));
    let b = enter(
        &mut w,
        RigidBody::new_dynamic(v3(0.5, 0.0, 0.0), Fix128::ONE),
        q,
        entry,
    );
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        v3(0.25, 1.0, 0.5),
    )));
    w.step(dt());
    let mut o = Obs::new(NEAR);
    o.v(w.bodies[b].position);
    o.v(w.bodies[b].velocity);
    o.v(w.bodies[b].angular_velocity);
    o
}

/// An off-centre ball SDF carried by the rotated (static) body, probed by a
/// dynamic sphere; reads `sdf_contacts` before and after a step.
fn sdf_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(zero_gravity());
    let carrier = enter(&mut w, RigidBody::new_static(Vec3Fix::ZERO), q, entry);
    // Inside the ball in the carrier's unit frame (local (1.1, 0.2, 0), 0.63
    // from the ball centre). The step pushes it out; it is put back before the
    // second read, which then reads the collider pose the step left.
    let at = turn().rotate_vec(v3(1.1, 0.2, 0.0));
    let probe = w.add_body(RigidBody::new_dynamic(at, Fix128::ONE));
    w.add_sdf_collider(SdfCollider::new_dynamic(
        Box::new(ClosureSdf::new(
            |x, y, z| ((x - 0.5) * (x - 0.5) + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = ((x - 0.5) * (x - 0.5) + y * y + z * z).sqrt().max(1e-6);
                ((x - 0.5) / l, y / l, z / l)
            },
        )),
        carrier,
    ));
    let mut o = Obs::new(NEAR_F32);
    let read = |w: &PhysicsWorld, o: &mut Obs| {
        let contacts = w.sdf_contacts();
        assert!(!contacts.is_empty(), "the probe touches the field");
        o.s(Fix128::from_int(contacts.len() as i64));
        for (_, c) in contacts {
            o.s(c.depth);
            o.v(c.normal);
        }
    };
    if entry != Entry::Field {
        read(&w, &mut o);
    }
    w.step(dt());
    o.v(w.bodies[probe].position);
    w.bodies[probe].set_position(at);
    read(&w, &mut o);
    o
}

/// A lidar mounted on the rotated body, looking at a wall.
fn lidar_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(zero_gravity());
    let b = enter(&mut w, RigidBody::new_static(Vec3Fix::ZERO), q, entry);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        -Vec3Fix::UNIT_Z,
        fx(6.0),
    )));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Z,
        fx(6.0),
    )));
    let mut lidar = Lidar::new((fx(-1.0), fx(1.0), 5), (fx(-0.5), fx(0.5), 3), fx(9.0)).with_pose(
        v3(0.5, 1.0, 0.25),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(0.3)),
    );
    let mut o = Obs::new(NEAR);
    let mut scan = |w: &PhysicsWorld, o: &mut Obs| {
        let s = lidar.scan_from_body(w, b).expect("body exists");
        let hits = s.ranges.iter().filter(|r| r.is_some()).count();
        assert!(
            hits > 0 && hits < s.ranges.len(),
            "some rays hit, some miss"
        );
        o.s(Fix128::from_int(hits as i64));
        for r in &s.ranges {
            o.s(r.unwrap_or(Fix128::from_int(-1)));
        }
    };
    if entry != Entry::Field {
        scan(&w, &mut o);
    }
    w.step(dt());
    scan(&w, &mut o);
    o
}

/// An IMU on the rotated body falling under gravity.
fn imu_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let b = enter(
        &mut w,
        RigidBody::new_dynamic(v3(0.0, 5.0, 0.0), Fix128::ONE),
        q,
        entry,
    );
    let mut imu = Imu::new(&w, b).expect("body exists");
    let mut o = Obs::new(NEAR);
    if entry != Entry::Field {
        let r = imu.sample(&w, dt()).expect("reading");
        o.v(r.specific_force);
        o.v(r.angular_velocity);
    }
    w.step(dt());
    let r = imu.sample(&w, dt()).expect("reading");
    o.v(r.specific_force);
    o.v(r.acceleration);
    o
}

/// The axes the debug renderer draws for the rotated body.
fn axes_scene(q: QuatFix, entry: Entry) -> Obs {
    let mut w = PhysicsWorld::new(zero_gravity());
    let b = enter(&mut w, RigidBody::new_static(v3(1.0, 2.0, 3.0)), q, entry);
    let flags = DebugDrawFlags {
        draw_aabbs: false,
        draw_centers: false,
        draw_contacts: false,
        draw_contact_normals: false,
        draw_joints: false,
        draw_axes: true,
        ..DebugDrawFlags::default()
    };
    let mut o = Obs::new(NEAR);
    let draw = |w: &PhysicsWorld, o: &mut Obs| {
        let mut data = DebugDrawData::new();
        debug_draw_world(w, &flags, &mut data);
        o.s(Fix128::from_int(data.lines.len() as i64));
        for l in &data.lines {
            o.v(l.end - l.start);
        }
    };
    if entry != Entry::Field {
        draw(&w, &mut o);
    }
    w.step(dt());
    draw(&w, &mut o);
    o.v(w.bodies[b].position);
    o
}

#[test]
fn nonunit_rotation_impulse_response_matches_unit() {
    check("impulse", impulse_scene);
}

#[test]
fn nonunit_rotation_ball_joint_step_matches_unit() {
    check("ball joint", ball_joint_scene);
}

#[test]
fn nonunit_rotation_sdf_contact_matches_unit() {
    check("sdf contact", sdf_scene);
}

#[test]
fn nonunit_rotation_lidar_on_body_matches_unit() {
    check("lidar", lidar_scene);
}

#[test]
fn nonunit_rotation_imu_matches_unit() {
    check("imu", imu_scene);
}

#[test]
fn nonunit_rotation_debug_axes_match_unit() {
    check("debug axes", axes_scene);
}

/// The stored rotation itself is of unit length after each entry (and after a
/// step for the field entry), and a unit rotation is stored bit for bit.
#[test]
fn stored_rotation_is_unit_after_every_entry() {
    let q = turn();
    for entry in ENTRIES {
        for s in SCALES {
            let qs = scaled(q, fx(s));
            let mut w = PhysicsWorld::new(zero_gravity());
            let i = enter(
                &mut w,
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
                qs,
                entry,
            );
            if entry != Entry::Field {
                for r in [w.bodies[i].rotation, w.bodies[i].prev_rotation] {
                    assert_eq!(r, qs.normalize(), "{entry:?} |q| = {s}, before a step");
                }
            }
            w.step(dt());
            for r in [w.bodies[i].rotation, w.bodies[i].prev_rotation] {
                assert_eq!(r, qs.normalize(), "{entry:?} |q| = {s}");
            }
        }
        let mut w = PhysicsWorld::new(zero_gravity());
        let i = enter(
            &mut w,
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            q,
            entry,
        );
        assert_eq!(w.bodies[i].rotation, q, "{entry:?}: unit rotation changed");
    }
}

/// A kinematic target given with a non-unit rotation, by
/// `set_kinematic_target` or by a direct write to the field, ends the step at
/// the unit rotation it stands for.
#[test]
fn kinematic_target_rotation_is_unit() {
    let q = turn();
    for s in SCALES {
        let qs = scaled(q, fx(s));
        for direct in [false, true] {
            let mut w = PhysicsWorld::new(zero_gravity());
            let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
            body.body_type = alice_physics::solver::BodyType::Kinematic;
            let i = w.add_body(body);
            if direct {
                w.bodies[i].kinematic_target = Some((v3(1.0, 0.0, 0.0), qs));
            } else {
                w.bodies[i].set_kinematic_target(v3(1.0, 0.0, 0.0), qs);
                assert_eq!(
                    w.bodies[i].kinematic_target,
                    Some((v3(1.0, 0.0, 0.0), qs.normalize())),
                    "|q| = {s}: stored target"
                );
            }
            w.step(dt());
            assert_eq!(
                w.bodies[i].rotation,
                qs.normalize(),
                "|q| = {s}, direct = {direct}"
            );
        }
    }
}

/// Every step path brings a rotation written directly to the field to unit
/// length: `step` with either backend, and `step_parallel`.
#[test]
fn every_step_path_makes_a_field_rotation_unit() {
    use alice_physics::solver::SolverBackend;
    let q = turn();
    for s in SCALES {
        let qs = scaled(q, fx(s));
        let run = |go: &dyn Fn(&mut PhysicsWorld)| {
            let mut w = PhysicsWorld::new(zero_gravity());
            let i = enter(
                &mut w,
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
                qs,
                Entry::Field,
            );
            go(&mut w);
            assert_eq!(w.bodies[i].rotation, qs.normalize(), "|q| = {s}");
        };
        run(&|w| w.step(dt()));
        run(&|w| {
            w.config.solver_backend = SolverBackend::Tgs;
            w.step(dt());
        });
        #[cfg(feature = "parallel")]
        run(&|w| w.step_parallel(dt()));
    }
}

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;

    pub(super) struct PassThrough;

    impl GpuSolverBridge for PassThrough {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {}
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    /// `step_with_bridge` and `substep_with_bridge` (callable on its own) bring
    /// a rotation written directly to the field to unit length.
    #[test]
    fn bridge_paths_make_a_field_rotation_unit() {
        let q = turn();
        for s in SCALES {
            let qs = scaled(q, fx(s));
            for whole_step in [true, false] {
                let mut w = PhysicsWorld::new(zero_gravity());
                let i = enter(
                    &mut w,
                    RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
                    qs,
                    Entry::Field,
                );
                let mut bridge = PassThrough;
                if whole_step {
                    w.step_with_bridge(&mut bridge, dt());
                } else {
                    w.substep_with_bridge(&mut bridge, dt());
                }
                assert_eq!(
                    w.bodies[i].rotation,
                    qs.normalize(),
                    "|q| = {s}, whole step = {whole_step}"
                );
            }
        }
    }
}

/// A participant that records the rotation of body 0 it is handed in each
/// substep.
struct Recorder {
    seen: std::sync::Arc<std::sync::Mutex<Vec<QuatFix>>>,
}

impl Participant for Recorder {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(7)
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        self.seen
            .lock()
            .expect("recorder lock")
            .push(ctx.bodies()[0].rotation);
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// A participant runs before the first substep, so it reads the rotation as
/// the step found it: on every step path (with either backend, the parallel
/// one and the bridge one) that is already the unit rotation.
#[test]
fn participants_see_the_unit_rotation_on_every_step_path() {
    use alice_physics::solver::SolverBackend;
    let q = turn();
    for s in SCALES {
        let qs = scaled(q, fx(s));
        let run = |name: &str, go: &dyn Fn(&mut PhysicsWorld)| {
            let mut w = PhysicsWorld::new(zero_gravity());
            enter(
                &mut w,
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
                qs,
                Entry::Field,
            );
            let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            w.add_participant(Box::new(Recorder { seen: seen.clone() }))
                .expect("register");
            go(&mut w);
            let seen = seen.lock().expect("recorder lock");
            assert!(!seen.is_empty(), "{name}: the participant ran");
            assert_eq!(seen[0], qs.normalize(), "{name} |q| = {s}");
        };
        run("step", &|w| w.step(dt()));
        run("tgs", &|w| {
            w.config.solver_backend = SolverBackend::Tgs;
            w.step(dt());
        });
        #[cfg(feature = "parallel")]
        run("parallel", &|w| w.step_parallel(dt()));
        #[cfg(feature = "gpu-solver-bridge")]
        run("bridge", &|w| {
            w.step_with_bridge(&mut bridge::PassThrough, dt())
        });
    }
}
