//! Sleep skip of `PhysicsWorld::step`: sleeping bodies at rest cost nothing
//! per stage, and skipping them changes no result.
//!
//! Three groups:
//!
//! 1. **Work counters** (`StageWork`): in a world where 99 % of the bodies sleep,
//!    every per-stage counter equals the one of a world with the same awake
//!    bodies and ten times fewer sleeping ones, and doubles with the awake
//!    count. The only counters that grow with the sleeping bodies are
//!    `sleep_scanned` (one check per body per step that a parked body was not
//!    edited between steps) and `parked_sleep_updates` (its `idle_frames + 1`),
//!    asserted here with their exact values.
//! 2. **Bit-identical results**: a twin world with the skip turned off is
//!    stepped alongside, and every body's full state, the sleep data, the
//!    snapshot blob and the contact / trigger events are compared after every
//!    step. Each scenario also asserts that bodies were parked and that the
//!    event it is about happened, so a scenario that never exercises the skip
//!    cannot pass by default.
//! 3. **Wake rules** (unchanged by the skip, asserted on both twins):
//!    a contact with an awake body wakes a sleeping one; a joint partner that
//!    moves drags a sleeping body awake; a force field or an impulse on a
//!    sleeping body does *not* wake it (its velocity is reset at the next step)
//!    unless `wake_body` is called; `wake_body` wakes it.

use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sleeping::{SleepConfig, SleepState};
use alice_physics::static_collider::StaticCollider;
use alice_physics::{
    BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, PlaneCollider, QuatFix, RigidBody,
    StageWork, Vec3Fix,
};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn half() -> Fix128 {
    Fix128::from_ratio(1, 2)
}

/// `n` bodies of radius 0.5 on a 2 m grid at `y = 0`; the first `awake` of them
/// are lifted to `y = 100` and left awake (they fall freely and never touch the
/// grid during a test), the rest are put to sleep at rest.
fn grid_world(n: usize, awake: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    // The awake bodies get their own 32-wide grid so their layout (and with it
    // the broad-phase's candidate pairs among them) does not depend on `n`.
    let side = (n as f64).sqrt().ceil() as i64;
    for i in 0..n {
        let (x, y, z) = if i < awake {
            ((i as i64 % 32) * 2, 100, (i as i64 / 32) * 2)
        } else {
            ((i as i64 % side) * 2, 0, (i as i64 / side) * 2)
        };
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE),
            half(),
        );
    }
    for i in awake..n {
        w.islands.sleep_data[i].state = SleepState::Sleeping;
        w.islands.sleep_data[i].idle_frames = 100;
    }
    w
}

/// Stats of the third step (the first one parks and fills the tree).
fn steady_stats(w: &mut PhysicsWorld) -> StageWork {
    w.step(dt());
    w.step(dt());
    w.step(dt());
    w.stage_work()
}

fn substeps() -> u64 {
    PhysicsConfig::default().substeps as u64
}

/// The counters a stage pays per body; all of them must not see a parked body.
fn per_stage(s: &StageWork) -> [(&'static str, u64); 10] {
    [
        ("force_field_bodies", s.force_field_bodies),
        ("integrated", s.integrated),
        ("broadphase_primitives", s.broadphase_primitives),
        ("broadphase_pairs", s.broadphase_pairs),
        ("resolution_bodies", s.resolution_bodies),
        ("velocity_bodies", s.velocity_bodies),
        ("damping_bodies", s.damping_bodies),
        ("sleep_evaluated", s.sleep_evaluated),
        ("tree_inserts", s.tree_inserts),
        ("tree_removes", s.tree_removes),
    ]
}

#[test]
fn stage_work_depends_on_awake_bodies_not_on_sleeping_ones() {
    let awake = 1_000;
    let big = steady_stats(&mut grid_world(100_000, awake));
    let small = steady_stats(&mut grid_world(10_000, awake));
    let double = steady_stats(&mut grid_world(100_000, 2 * awake));

    // Same awake bodies, 99 000 vs 9 000 sleeping ones: identical work.
    assert_eq!(
        per_stage(&big),
        per_stage(&small),
        "big {big:?}\nsmall {small:?}"
    );

    // Exact values: every awake body is integrated, boxed and has its velocity
    // derived once per substep, damped and evaluated for sleep once per step.
    let a = awake as u64;
    assert_eq!(big.integrated, a * substeps());
    assert_eq!(big.broadphase_primitives, a * substeps());
    assert_eq!(big.velocity_bodies, a * substeps());
    assert_eq!(big.damping_bodies, a);
    assert_eq!(big.sleep_evaluated, a);
    assert_eq!(big.tree_inserts, 0, "steady state re-inserts nothing");
    assert_eq!(big.tree_removes, 0);
    assert_eq!(big.unparked, 0);

    // Twice the awake bodies, twice the per-body work.
    assert_eq!(double.integrated, 2 * big.integrated);
    assert_eq!(double.velocity_bodies, 2 * big.velocity_bodies);
    assert_eq!(double.broadphase_primitives, 2 * big.broadphase_primitives);
    assert_eq!(double.sleep_evaluated, 2 * big.sleep_evaluated);

    // The light per-step pass is the one part that sees every body.
    assert_eq!(big.sleep_scanned, 100_000);
    assert_eq!(big.parked, 99_000);
    assert_eq!(big.parked_sleep_updates, 99_000);
    assert_eq!(small.sleep_scanned, 10_000);
    assert_eq!(small.parked_sleep_updates, 9_000);
}

#[test]
fn skip_off_visits_every_body() {
    let mut w = grid_world(2_000, 20);
    w.set_sleep_skip(false);
    assert!(!w.sleep_skip());
    let s = steady_stats(&mut w);
    assert_eq!(s.integrated, 2_000 * substeps());
    assert_eq!(s.broadphase_primitives, 2_000 * substeps());
    assert_eq!(s.sleep_evaluated, 2_000);
    assert_eq!(s.parked, 0);
    assert_eq!(s.sleep_scanned, 0);
}

// ── Bit-identical results ──────────────────────────────────────────────────

fn body_bits(b: &RigidBody) -> [Vec3Fix; 4] {
    [b.position, b.velocity, b.angular_velocity, b.prev_position]
}

fn assert_twins(on: &PhysicsWorld, off: &PhysicsWorld, ctx: &str) {
    assert_eq!(on.bodies.len(), off.bodies.len(), "{ctx}: body count");
    for (i, (a, b)) in on.bodies.iter().zip(&off.bodies).enumerate() {
        assert_eq!(body_bits(a), body_bits(b), "{ctx}: body {i} linear state");
        assert_eq!(
            (a.rotation, a.prev_rotation),
            (b.rotation, b.prev_rotation),
            "{ctx}: body {i} rotation"
        );
    }
    assert_eq!(
        on.islands.sleep_data, off.islands.sleep_data,
        "{ctx}: sleep data"
    );
    assert_eq!(
        on.serialize_state(),
        off.serialize_state(),
        "{ctx}: snapshot blob"
    );
    assert_eq!(
        on.events.contact_events(),
        off.events.contact_events(),
        "{ctx}: contact events"
    );
    assert_eq!(
        on.events.trigger_events(),
        off.events.trigger_events(),
        "{ctx}: trigger events"
    );
}

/// Two copies of a world, one with the sleep skip off.
struct Twins {
    on: PhysicsWorld,
    off: PhysicsWorld,
    parked: u64,
    unparked: u64,
}

impl Twins {
    fn new(build: impl Fn() -> PhysicsWorld) -> Self {
        let on = build();
        let mut off = build();
        off.set_sleep_skip(false);
        assert!(on.sleep_skip(), "skip is on by default");
        Self {
            on,
            off,
            parked: 0,
            unparked: 0,
        }
    }

    fn both(&mut self, f: impl Fn(&mut PhysicsWorld)) {
        f(&mut self.on);
        f(&mut self.off);
    }

    fn step(&mut self, frames: usize, ctx: &str) {
        for k in 0..frames {
            self.on.step(dt());
            self.off.step(dt());
            let s = self.on.stage_work();
            self.parked += s.parked;
            self.unparked += s.unparked;
            assert_twins(&self.on, &self.off, &format!("{ctx} frame {k}"));
        }
    }

    fn sleeping(&self, i: usize) -> bool {
        let a = self.on.is_sleeping(i);
        assert_eq!(a, self.off.is_sleeping(i), "sleep state of {i} differs");
        a
    }
}

fn quick_sleep() -> SleepConfig {
    SleepConfig {
        frames_to_sleep: 5,
        ..SleepConfig::default()
    }
}

/// Weightless row of resting spheres that fall asleep on their own, plus one
/// projectile (index 0) aimed at body `target`.
fn row_with_projectile(rows: i64, target: i64) -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_sleep_config(quick_sleep());
    let mut p = RigidBody::new_dynamic(Vec3Fix::from_int(target * 3, 0, -6), Fix128::ONE);
    p.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 3));
    p.prev_rotation = p.rotation;
    w.add_body_with_radius(p, half());
    for i in 0..rows {
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(i * 3, 0, 0), Fix128::ONE);
        // A non-identity rotation: the derived angular velocity of an unchanged
        // rotation is not exactly zero for these, which the cache must carry.
        b.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(i + 1, 7));
        b.prev_rotation = b.rotation;
        w.add_body_with_radius(b, half());
    }
    w
}

#[test]
fn contact_with_awake_body_wakes_a_parked_one_bit_identically() {
    let mut t = Twins::new(|| row_with_projectile(40, 10));
    // Let the row and the projectile fall asleep (all bodies at rest).
    t.step(8, "settle");
    assert!(t.sleeping(0) && t.sleeping(11), "everything asleep");
    assert!(t.parked > 0, "the skip parked bodies");
    // Wake and launch the projectile at body 11.
    t.both(|w| {
        w.wake_body(0);
        w.bodies[0].set_velocity(Vec3Fix::from_int(0, 0, 6));
    });
    t.step(90, "hit");
    assert!(!t.sleeping(11) || t.unparked > 0, "target was woken");
    assert!(t.unparked > 0, "a parked body was woken by a contact");
    let moved = t.on.bodies[11].position.z;
    assert!(moved > Fix128::ZERO, "target pushed along +z: {moved:?}");
    assert!(t.sleeping(30), "a body nobody touched stays asleep");
}

#[test]
fn force_field_contact_event_velocity_matches_while_parked() {
    let mut t = Twins::new(|| {
        let mut w = row_with_projectile(12, 4);
        w.add_force_field(ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_ratio(1, 1000),
        }));
        w
    });
    t.step(8, "settle");
    // A force field alone does not wake a sleeping body (rule pinned on both).
    assert!(t.sleeping(5), "force field does not wake");
    assert!(t.parked > 0);
    // Teleport the projectile so it overlaps body 5 in the first substep: the
    // contact event's relative velocity carries the field's kick of that body.
    t.both(|w| {
        w.wake_body(0);
        w.bodies[0].set_position(Vec3Fix::new(
            Fix128::from_int(12),
            Fix128::ZERO,
            Fix128::from_ratio(-9, 10),
        ));
    });
    t.on.step(dt());
    t.off.step(dt());
    assert_twins(&t.on, &t.off, "overlap step");
    assert!(
        t.on.events
            .contact_events()
            .iter()
            .any(|e| e.body_a == 0 || e.body_b == 0),
        "the overlap produced a contact event"
    );
    assert!(
        t.on.stage_work().unparked > 0,
        "the hit body was parked when hit"
    );
    t.step(30, "after");
}

#[test]
fn joint_partner_drags_sleeping_body_awake() {
    let mut t = Twins::new(|| {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.set_sleep_config(quick_sleep());
        let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 10, 0)));
        let a = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 8, 0), Fix128::ONE),
            half(),
        );
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 6, 0), Fix128::ONE),
            half(),
        );
        w.add_joint(Joint::Ball(BallJoint::new(
            anchor,
            a,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 2, 0),
        )));
        w.add_joint(Joint::Ball(BallJoint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 2, 0),
        )));
        // Far away loose bodies that do get parked.
        for i in 0..20 {
            let id = w.add_body_with_radius(
                RigidBody::new_dynamic(Vec3Fix::from_int(100 + 3 * i, 0, 0), Fix128::ONE),
                half(),
            );
            w.islands.sleep_data[id].state = SleepState::Sleeping;
        }
        w
    });
    t.step(200, "hang");
    assert!(t.sleeping(1) && t.sleeping(2), "the chain fell asleep");
    assert!(t.parked > 0);
    // Kick the lower link awake: the upper one is pulled along and wakes.
    t.both(|w| {
        w.wake_body(2);
        w.bodies[2].set_velocity(Vec3Fix::from_int(5, 0, 0));
    });
    t.step(3, "kick");
    assert!(!t.sleeping(1), "joint partner woke the upper link");
}

#[test]
fn impulse_on_sleeping_body_needs_wake_body() {
    let mut t = Twins::new(|| row_with_projectile(6, 2));
    t.step(8, "settle");
    assert!(t.sleeping(3));
    let start = t.on.bodies[3].position;
    // Without wake_body the impulse is dropped at the next step.
    t.both(|w| w.bodies[3].apply_impulse(Vec3Fix::from_int(0, 4, 0)));
    t.step(5, "dropped impulse");
    assert!(t.sleeping(3), "impulse alone does not wake");
    assert_eq!(t.on.bodies[3].position, start, "and does not move it");
    assert_eq!(t.on.bodies[3].velocity, Vec3Fix::ZERO, "velocity reset");
    // With wake_body it moves.
    t.both(|w| {
        w.wake_body(3);
        w.bodies[3].apply_impulse(Vec3Fix::from_int(0, 4, 0));
    });
    t.step(5, "woken impulse");
    assert!(!t.sleeping(3), "wake_body wakes it");
    assert!(t.on.bodies[3].position.y > start.y, "and it moves");
}

#[test]
fn edits_between_steps_are_seen() {
    let mut t = Twins::new(|| row_with_projectile(10, 3));
    t.step(8, "settle");
    assert!(t.parked > 0);
    // Teleport a sleeping body (`set_position` keeps `prev == position`, so only
    // the position itself tells) into the awake projectile: the contact must
    // be found at the new place and wake it.
    t.both(|w| {
        w.wake_body(0);
        w.bodies[5].set_position(Vec3Fix::new(
            Fix128::from_int(9),
            Fix128::ZERO,
            Fix128::from_ratio(-54, 10),
        ));
    });
    let unparked_before = t.unparked;
    t.step(3, "teleport");
    assert!(
        t.unparked > unparked_before,
        "the teleported body was woken by the contact"
    );
    // Spin a sleeping body through the setter without waking it: the next step
    // resets its angular velocity to the at-rest one.
    t.both(|w| w.bodies[8].set_angular_velocity(Vec3Fix::from_int(0, 3, 0)));
    t.step(2, "spin without wake");
    assert!(t.sleeping(8), "spin alone does not wake");
    // Write the position field directly (prev stays behind).
    t.both(|w| w.bodies[6].position = Vec3Fix::from_int(15, 1, 0));
    t.step(3, "direct write");
    // Rotate a sleeping body through the setter.
    t.both(|w| {
        w.bodies[7].set_rotation(QuatFix::from_axis_angle(
            Vec3Fix::UNIT_Z,
            Fix128::from_ratio(2, 5),
        ));
    });
    t.step(3, "rotate");
    // Change the radius and the sleep thresholds.
    t.both(|w| {
        w.set_body_collision_radius(8, Fix128::from_int(2));
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: 3,
            ..SleepConfig::default()
        });
    });
    t.step(5, "radius + config");
    // Remove a sleeping body: every body wakes (`remove_body` rebuilds the
    // island manager), with or without the skip.
    t.both(|w| {
        w.remove_body(4);
    });
    t.step(1, "remove");
    t.step(12, "after remove");
}

#[test]
fn static_and_sdf_colliders_under_parked_bodies() {
    let sdf = || ClosureSdf::new(|_x, y, _z| y + 5.0, |_x, _y, _z| (0.0, 1.0, 0.0));
    let mut t = Twins::new(|| {
        let mut w = row_with_projectile(10, 3);
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            Fix128::from_int(-3),
        )));
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(sdf()),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        w
    });
    t.step(8, "settle");
    assert!(t.parked > 0, "clear of both colliders: parked");
    // Raise the SDF floor through the sleeping row (public field edit).
    t.both(|w| w.sdf_colliders[0].position = Vec3Fix::from_int(0, 5, 0));
    t.step(4, "sdf moved");
    let y = t.on.bodies[5].position.y;
    assert!(
        y > Fix128::ZERO,
        "sleeping body pushed out of the moved SDF: {y:?}"
    );
    // A static plane added just below the centres of the row (the plane is
    // two-sided: a sphere is pushed to the side its centre is on): every body
    // of it now overlaps the plane and is pushed up.
    t.step(10, "resettle");
    let y0 = t.on.bodies[5].position.y;
    let offset = y0 - Fix128::from_ratio(1, 4);
    t.both(|w| {
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            offset,
        )));
    });
    t.step(4, "plane added");
    let y1 = t.on.bodies[5].position.y;
    assert!(y1 > y0, "pushed up by the new plane: {y0:?} -> {y1:?}");
}

#[test]
fn natural_pile_under_gravity_is_bit_identical() {
    let mut t = Twins::new(|| {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: 10,
            ..SleepConfig::default()
        });
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        )));
        for i in 0..30i64 {
            w.add_body_with_radius(
                RigidBody::new_dynamic(
                    Vec3Fix::new(
                        Fix128::from_ratio(i % 6 * 11, 10),
                        Fix128::from_int(1 + i / 6),
                        Fix128::from_ratio(i % 5, 10),
                    ),
                    Fix128::ONE,
                ),
                half(),
            );
        }
        w
    });
    t.step(400, "pile");
}

// ── Degenerate inputs ──────────────────────────────────────────────────────

/// 0 bodies: the step does nothing and counts nothing.
#[test]
fn empty_world_counts_nothing() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.step(dt());
    assert_eq!(w.stage_work(), StageWork::default());
}

/// 1 sleeping body: parked, nothing visits it, `idle_frames` still counts up
/// and it does not move.
#[test]
fn single_sleeping_body_is_parked_and_counts_idle_frames() {
    let mut w = grid_world(1, 0);
    let before = w.bodies[0];
    let s = steady_stats(&mut w);
    assert_eq!(s.parked, 1);
    assert_eq!(s.integrated, 0);
    assert_eq!(s.broadphase_primitives, 0);
    assert_eq!(s.broadphase_pairs, 0);
    assert_eq!(s.sleep_scanned, 1);
    assert_eq!(s.parked_sleep_updates, 1);
    assert_eq!(
        w.islands.sleep_data[0].idle_frames, 103,
        "100 + one per step"
    );
    assert_eq!(w.bodies[0].position, before.position);
    assert!(w.is_sleeping(0));
}

/// All bodies asleep: no stage runs, every body with a radius has a proxy
/// (inserted on the first step only), no pair is generated.
#[test]
fn all_sleeping_world_does_no_stage_work() {
    let mut w = grid_world(500, 0);
    w.step(dt());
    let first = w.stage_work();
    assert_eq!(first.tree_inserts, 500);
    let s = steady_stats(&mut w);
    assert_eq!(s.tree_inserts, 0);
    assert_eq!(s.parked, 500);
    for (name, v) in per_stage(&s) {
        assert_eq!(v, 0, "{name}");
    }
}

/// A sleeping body removed: the remaining ones are woken by `remove_body` and
/// leave the tree on the next step.
#[test]
fn removing_a_sleeping_body_unparks_the_rest() {
    let mut w = grid_world(50, 0);
    steady_stats(&mut w);
    assert!(w.remove_body(10).is_some());
    w.step(dt());
    let s = w.stage_work();
    assert_eq!(s.parked, 0, "remove_body woke every body");
    // remove_body invalidates every parked verdict (indices moved): all 50
    // proxies, the removed body's included, leave the tree.
    assert_eq!(s.tree_removes, 50, "their proxies left the tree");
    assert_eq!(s.integrated, 49 * substeps());
}

/// The bench world (`benches/world_scale.rs`): weightless, undamped, the awake
/// bodies drift upward above a sleeping grid. Same with and without the skip.
#[test]
fn bench_world_is_bit_identical() {
    let build = || {
        let config = PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            damping: Fix128::ONE,
            ..PhysicsConfig::default()
        };
        let mut w = PhysicsWorld::new(config);
        let n = 500usize;
        let awake = 5usize;
        let side = (n as f64).sqrt().ceil() as i64;
        for i in 0..n {
            let x = (i as i64 % side) * 2;
            let z = (i as i64 / side) * 2;
            let y = if i < awake { 100 } else { 0 };
            let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE);
            if i < awake {
                body.velocity = Vec3Fix::from_int(0, 1, 0);
            }
            w.add_body_with_radius(body, half());
        }
        for i in awake..n {
            w.islands.sleep_data[i].state = SleepState::Sleeping;
            w.islands.sleep_data[i].idle_frames = 100;
        }
        w
    };
    let mut t = Twins::new(build);
    t.step(120, "drift");
    assert!(t.parked > 0);
    let awake = (0..5).filter(|&i| !t.sleeping(i)).count();
    assert_eq!(awake, 5, "the drifting bodies stay awake");
}
