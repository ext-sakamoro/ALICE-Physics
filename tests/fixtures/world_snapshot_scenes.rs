// Scenes of the frozen world snapshot fixtures (`world_snapshot_v{3,4,5}_*.bin`).
//
// This file is shared, unchanged, by the generator (`world_snapshot_gen.rs`,
// built against a release tag or the current tree) and by the tests that read
// the fixtures (`include!`d into `tests/world_snapshot_v5.rs`). It uses only
// public API present since 2.0.0, by full path, so it compiles in both places.
// See `README.md` in this directory for how each fixture was generated.

/// Steps after which a "later" fixture (`*_later.bin`) is written: the same
/// scene stepped this many more times by the writing release.
pub const FIXTURE_LATER_STEPS: usize = 20;

pub fn fixture_dt() -> alice_physics::Fix128 {
    alice_physics::Fix128::from_ratio(1, 60)
}

fn fixture_sleep(w: &mut alice_physics::PhysicsWorld) {
    w.set_sleep_config(alice_physics::SleepConfig {
        linear_threshold: alice_physics::Fix128::from_ratio(1, 100),
        angular_threshold: alice_physics::Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
}

/// A static anchor and two dynamic bodies of radius 1/4, the first held by
/// a ball joint, default broad phase, 3 steps.
pub fn fixture_joint_pair() -> alice_physics::PhysicsWorld {
    use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
    use alice_physics::{Fix128, Vec3Fix};
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 5, 0), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(2, 5, 1), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.add_joint(alice_physics::joint::Joint::Ball(
        alice_physics::joint::BallJoint::new(
            anchor,
            a,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-1, 0, 0),
        ),
    ));
    w.step_n(3, fixture_dt());
    w
}

/// `DynamicTree`, a static floor and two dynamic bodies of radius 1/2 joined
/// by a ball joint; a body of radius 1/4 far away is added, stepped once
/// (so it enters the tree) and removed, then 30 more steps. The tree keeps
/// the freed node and its free list.
pub fn fixture_dynamic_tree_freed() -> alice_physics::PhysicsWorld {
    use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
    use alice_physics::{Fix128, Vec3Fix};
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(Broadphase::DynamicTree);
    fixture_sleep(&mut w);
    let half = Fix128::from_ratio(1, 2);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 1, 0), Fix128::ONE),
        half,
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(-1, 1, 0), Fix128::ONE),
        half,
    );
    w.add_joint(alice_physics::joint::Joint::Ball(
        alice_physics::joint::BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO),
    ));
    let extra = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(50, 50, 50), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.step(fixture_dt());
    w.remove_body(extra);
    w.step_n(30, fixture_dt());
    w
}

/// A static floor and four dynamic bodies of radius 1/2 whose boxes overlap
/// (so the broad phase reports pairs), with `kind`, 20 steps.
pub fn fixture_overlapping(kind: alice_physics::solver::Broadphase) -> alice_physics::PhysicsWorld {
    use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
    use alice_physics::{Fix128, Vec3Fix};
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    fixture_sleep(&mut w);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)));
    let half = Fix128::from_ratio(1, 2);
    for (x, z) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(x, 2, z), Fix128::ONE),
            half,
        );
    }
    w.step_n(20, fixture_dt());
    w
}

/// Every fixture scene by the name used in its file names.
pub fn fixture_scenes() -> Vec<(&'static str, alice_physics::PhysicsWorld)> {
    use alice_physics::solver::Broadphase;
    vec![
        ("joint_pair", fixture_joint_pair()),
        ("dynamic_tree_freed", fixture_dynamic_tree_freed()),
        ("dynamic_tree", fixture_overlapping(Broadphase::DynamicTree)),
        ("bvh", fixture_overlapping(Broadphase::Bvh)),
        ("hybrid", fixture_overlapping(Broadphase::Hybrid)),
    ]
}
