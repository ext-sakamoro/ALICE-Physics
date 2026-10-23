//! Cross-platform determinism golden for the overflow flag of pairs whose
//! collision radii sum past the `Fix128` range.
//!
//! The other goldens hash position, velocity, rotation and angular velocity;
//! the sticky overflow flag (`PhysicsWorld::overflow_detected`) is not part of
//! those bytes. In the scenes here nothing moves (no gravity, every body at
//! rest), so the only state a step can change is the flag and the sleep
//! bookkeeping. The golden hashes `serialize_state` (which carries the flag)
//! of every world, under every broadphase kind with the sleep skip on and off.
//!
//! # Scenes
//!
//! Every body has collision radius `2^62`: two of them sum to `2^63`, past the
//! range, while every sphere cube used here stays inside it. `dt = 1/60 s`,
//! three steps, for each of `Bvh`, `DynamicTree`, `Hybrid` × sleep skip off / on:
//!
//! * two dynamic bodies with unit ellipsoid colliders, centres `2 + gap`
//!   apart, `gap` in {-1/4, 0, 1/4, 3/4, 5/4, 3} (boxes overlap for `gap <= 0`), both
//!   awake, the first asleep, and both asleep;
//! * two awake sensor bodies without colliders at `x = 0` and `x = 2^40`;
//! * two static bodies without colliders, centres 2 apart;
//! * two sleeping dynamic bodies without colliders at `x = 0` and `x = 2^40`;
//! * two awake dynamic bodies at `x = 0` and `x = 2^40` in the same collision
//!   group (the filter rejects the pair);
//! * the sleeping pair above with the second radius the largest value, so its
//!   cube `2^40 + r` leaves the range;
//! * the sleeping pair above, then body 0 woken and one more step (the state
//!   is hashed before and after the wake step).
//!
//! The invariant checked before the digest: no body has moved.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepState;
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use sha2::{Digest, Sha256};

const STEPS: usize = 3;

const KINDS: [Broadphase; 3] = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];

/// SHA-256 of the concatenated `serialize_state` bytes of every scene world,
/// in the order [`all_states`] produces them.
const OVERFLOW_FLAG_GOLDEN: &str =
    "4a6eddf45bd7413e5ab423cc589433ac76ff0fbd4ae534936480702d4832a430";

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

fn huge() -> Fix128 {
    Fix128::from_int(1 << 62)
}

fn world(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    w
}

fn put_to_sleep(w: &mut PhysicsWorld, i: usize) {
    w.islands.sleep_data[i].state = SleepState::Sleeping;
    w.islands.sleep_data[i].idle_frames = 100;
}

/// `asleep` says which of the two bodies start asleep
fn collider_pair(kind: Broadphase, skip: bool, gap: Fix128, asleep: (bool, bool)) -> PhysicsWorld {
    let mut w = world(kind, skip);
    let unit = Shape::Ellipsoid {
        radii: Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE),
    };
    for (x, sleeps) in [Fix128::ZERO, Fix128::from_int(2) + gap]
        .into_iter()
        .zip([asleep.0, asleep.1])
    {
        let i = w.add_body_with_radius(RigidBody::new_dynamic(x_at(x), Fix128::ONE), huge());
        assert!(w.set_body_shape(i, &unit));
        w.set_body_collision_radius(i, huge());
        if sleeps {
            put_to_sleep(&mut w, i);
        }
    }
    w
}

fn sensor_pair(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = world(kind, skip);
    for x in [Fix128::ZERO, Fix128::from_int(1 << 40)] {
        let mut b = RigidBody::new_dynamic(x_at(x), Fix128::ONE);
        b.is_sensor = true;
        w.add_body_with_radius(b, huge());
    }
    w
}

fn static_pair(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = world(kind, skip);
    for x in [Fix128::ZERO, Fix128::from_int(2)] {
        w.add_body_with_radius(RigidBody::new_static(x_at(x)), huge());
    }
    w
}

fn sleeping_pair(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = world(kind, skip);
    for x in [Fix128::ZERO, Fix128::from_int(1 << 40)] {
        let i = w.add_body_with_radius(RigidBody::new_dynamic(x_at(x), Fix128::ONE), huge());
        put_to_sleep(&mut w, i);
    }
    w
}

/// The sleeping pair with the second body's radius the largest value: its cube
/// `2^40 + r` leaves the range.
fn cube_out_pair(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = world(kind, skip);
    let radii = [huge(), Fix128::from_raw(i64::MAX, u64::MAX)];
    for (x, r) in [Fix128::ZERO, Fix128::from_int(1 << 40)]
        .into_iter()
        .zip(radii)
    {
        let i = w.add_body_with_radius(RigidBody::new_dynamic(x_at(x), Fix128::ONE), r);
        put_to_sleep(&mut w, i);
    }
    w
}

fn filtered_pair(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = world(kind, skip);
    let same_group = CollisionFilter {
        group: 7,
        ..CollisionFilter::DEFAULT
    };
    for x in [Fix128::ZERO, Fix128::from_int(1 << 40)] {
        let i = w.add_body_with_radius(RigidBody::new_dynamic(x_at(x), Fix128::ONE), huge());
        w.set_body_filter(i, same_group);
    }
    w
}

/// Steps `w` and checks that no body moved.
fn stepped(mut w: PhysicsWorld, steps: usize, what: &str) -> PhysicsWorld {
    let before: Vec<Vec3Fix> = w.bodies.iter().map(|b| b.position).collect();
    for _ in 0..steps {
        w.step(dt());
    }
    for (i, b) in w.bodies.iter().enumerate() {
        assert_eq!(b.position, before[i], "{what}: body {i} moved");
    }
    w
}

/// The `serialize_state` of every scene world, with a label for each.
fn all_states() -> Vec<(String, Vec<u8>)> {
    let gaps = [
        Fix128::from_ratio(-1, 4),
        Fix128::ZERO,
        Fix128::from_ratio(1, 4),
        Fix128::from_ratio(3, 4),
        Fix128::from_ratio(5, 4),
        Fix128::from_int(3),
    ];
    let mut out = Vec::new();
    for kind in KINDS {
        for skip in [false, true] {
            for asleep in [(false, false), (true, false), (true, true)] {
                for gap in gaps {
                    let what = format!("{kind:?} skip={skip} collider {asleep:?} gap={gap:?}");
                    let w = stepped(collider_pair(kind, skip, gap, asleep), STEPS, &what);
                    out.push((
                        format!("{what}: flag {}", w.overflow_detected()),
                        w.serialize_state(),
                    ));
                }
            }
            let what = format!("{kind:?} skip={skip} sensor pair");
            let w = stepped(sensor_pair(kind, skip), STEPS, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));
            let what = format!("{kind:?} skip={skip} static pair");
            let w = stepped(static_pair(kind, skip), STEPS, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));

            let what = format!("{kind:?} skip={skip} sleeping pair");
            let mut w = stepped(sleeping_pair(kind, skip), STEPS, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));
            w.wake_body(0);
            let what = format!("{kind:?} skip={skip} sleeping pair, woken");
            let w = stepped(w, 1, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));

            let what = format!("{kind:?} skip={skip} sleeping pair, cube out of range");
            let w = stepped(cube_out_pair(kind, skip), STEPS, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));

            let what = format!("{kind:?} skip={skip} filtered pair");
            let w = stepped(filtered_pair(kind, skip), STEPS, &what);
            out.push((
                format!("{what}: flag {}", w.overflow_detected()),
                w.serialize_state(),
            ));
        }
    }
    out
}

#[test]
fn golden_out_of_range_radius_sum() {
    let states = all_states();
    assert_eq!(states.len(), 3 * 2 * 24, "24 worlds per kind and skip mode");
    let mut bytes = Vec::new();
    for (what, state) in &states {
        eprintln!("{what}");
        bytes.extend_from_slice(state);
    }
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let actual: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        actual, OVERFLOW_FLAG_GOLDEN,
        "\noverflow flag golden\n  expected: {OVERFLOW_FLAG_GOLDEN}\n  actual:   {actual}\n\
         Update the constant only for a deliberate change of when the flag is raised; \
         a mismatch on one platform only is a determinism bug."
    );
}
