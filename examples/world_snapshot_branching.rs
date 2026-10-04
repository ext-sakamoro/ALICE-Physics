//! Branching from a whole-world snapshot.
//!
//! A pendulum chain (ball + hinge joints), a wind field and a resting box are
//! stepped to a branch point. The whole world is saved with
//! `PhysicsWorld::snapshot_world`, three alternative pushes are tried on worlds
//! restored from the same blob with `PhysicsWorld::from_world_snapshot`, and
//! the original is checked to be unaffected and bit-identical to a fresh
//! restore run with the same input. A world holding an SDF collider is
//! restored with `restore_world`, which keeps the SDF field of the target.
//!
//! Run: `cargo run --release --example world_snapshot_branching`

use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::{
    Fix128, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, SleepConfig, Vec3Fix,
    WorldSnapshotError,
};

fn build() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 10,
    });
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 6, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 6, 0), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(2, 6, 0), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        a,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, 0),
    )));
    let z = Vec3Fix::from_int(0, 0, 1);
    w.add_joint(Joint::Hinge(HingeJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, 0),
        z,
        z,
    )));
    let mut idle = RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    idle.gravity_scale = Fix128::ZERO;
    w.add_body(idle);
    w.add_force_field(
        ForceFieldInstance::new(ForceField::Directional {
            direction: z,
            strength: Fix128::from_ratio(1, 2),
        })
        .with_affected_bodies(vec![b]),
    );
    w
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let mut world = build();
    world.step_n(30, dt);
    let blob = world.snapshot_world();
    println!(
        "[world_snapshot branching] branch point after 30 steps: {} bytes, idle body sleeping = {}",
        blob.len(),
        world.is_sleeping(3)
    );
    assert!(world.is_sleeping(3));

    // Three alternative pushes on the chain's tip, each from the same blob.
    let mut tips = Vec::new();
    for push in [-2i64, 0, 2] {
        let mut branch = PhysicsWorld::from_world_snapshot(&blob).expect("restore");
        branch.bodies[2].velocity = branch.bodies[2].velocity + Vec3Fix::from_int(push, 0, 0);
        branch.step_n(60, dt);
        let p = branch.bodies[2].position;
        println!(
            "[world_snapshot branching] push {push:+}: tip at ({:.4}, {:.4}, {:.4})",
            p.x.to_f64(),
            p.y.to_f64(),
            p.z.to_f64()
        );
        tips.push(p);
    }
    assert!(
        tips[0] != tips[1] && tips[1] != tips[2],
        "pushes change the outcome"
    );

    // The original was not touched by the branches, and continuing it is
    // bit-identical to the "no push" branch.
    assert_eq!(world.snapshot_world(), blob);
    world.step_n(60, dt);
    assert_eq!(world.bodies[2].position, tips[1]);
    println!("[world_snapshot branching] original continued == no-push branch: bit-identical");

    // A world with an SDF collider: the field is code, so it is restored into
    // a world that already holds one.
    let ball = || {
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt().max(1e-6);
                (x / l, y / l, z / l)
            },
        ))
    };
    let mut with_sdf = build();
    with_sdf.add_sdf_collider(SdfCollider::new_static(
        ball(),
        Vec3Fix::from_int(1, 3, 0),
        QuatFix::IDENTITY,
    ));
    with_sdf.step_n(10, dt);
    let sdf_blob = with_sdf.snapshot_world();
    assert_eq!(
        PhysicsWorld::from_world_snapshot(&sdf_blob).err(),
        Some(WorldSnapshotError::SdfFieldCountMismatch {
            snapshot: 1,
            world: 0
        })
    );
    let mut target = PhysicsWorld::new(PhysicsConfig::default());
    target.add_sdf_collider(SdfCollider::new_static(
        ball(),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    target
        .restore_world(&sdf_blob)
        .expect("restore with SDF field");
    with_sdf.step_n(20, dt);
    target.step_n(20, dt);
    assert_eq!(with_sdf.snapshot_world(), target.snapshot_world());
    println!("[world_snapshot branching] SDF world restored into a target holding the field: bit-identical after 20 steps");

    // A corrupted blob is refused with the reason.
    let mut bad = blob.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    let err = PhysicsWorld::from_world_snapshot(&bad).unwrap_err();
    println!("[world_snapshot branching] corrupted blob: {err}");
    assert!(matches!(err, WorldSnapshotError::ChecksumMismatch { .. }));
}
