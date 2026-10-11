#![no_main]
use alice_physics::world_participant::StepError;
use alice_physics::{
    BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix,
};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    /// Bodies in the world (capped at 6).
    body_count: u8,
    /// Joint index pairs, pushed straight into `PhysicsWorld::joints`: they may
    /// name a body past the last one, the same body twice, or a huge index.
    joints: Vec<(u16, u16)>,
    /// Map an index byte pair to usize::MAX instead (a far out-of-range index).
    huge: Vec<bool>,
    /// Use the TGS backend instead of XPBD.
    tgs: bool,
}

fn index(raw: u16, huge: bool) -> usize {
    if huge {
        usize::MAX
    } else {
        raw as usize % 12
    }
}

// The start-of-step joint check: whatever pairs are pushed into `joints`, the
// checked steps never panic; they return `StepError::Joint` naming the first
// joint that names a missing body or the same body twice, and leave the world
// unchanged; with only valid joints the unchecked `step` runs without panicking.
fuzz_target!(|input: Input| {
    let config = PhysicsConfig {
        substeps: 2,
        solver_backend: if input.tgs { SolverBackend::Tgs } else { SolverBackend::Xpbd },
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let body_count = (input.body_count as usize).clamp(1, 6);
    for i in 0..body_count {
        world.add_body(RigidBody::new(Vec3Fix::from_int(i as i64, 0, 0), Fix128::ONE));
    }
    for (k, &(a, b)) in input.joints.iter().take(8).enumerate() {
        let huge = input.huge.get(k).copied().unwrap_or(false);
        world.joints.push(Joint::Ball(BallJoint::new(
            index(a, huge),
            index(b, false),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
    }
    let first_invalid = world.joints.iter().enumerate().find_map(|(i, j)| {
        let (a, b) = j.bodies();
        (a >= body_count || b >= body_count || a == b).then_some((i, (a, b)))
    });
    let dt = Fix128::from_ratio(1, 60);
    let before: Vec<(Vec3Fix, Vec3Fix)> =
        world.bodies.iter().map(|b| (b.position, b.velocity)).collect();

    match first_invalid {
        Some((index, bodies)) => {
            let want = Err(StepError::Joint { index, bodies, body_count });
            assert_eq!(world.try_step(dt), want);
            #[cfg(feature = "parallel")]
            assert_eq!(world.try_step_parallel(dt), want);
            let after: Vec<(Vec3Fix, Vec3Fix)> =
                world.bodies.iter().map(|b| (b.position, b.velocity)).collect();
            assert_eq!(after, before, "a refused step moved a body");
        }
        None => {
            world.step(dt);
            #[cfg(feature = "parallel")]
            world.step_parallel(dt);
        }
    }
});
