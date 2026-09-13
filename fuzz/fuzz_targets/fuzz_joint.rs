#![no_main]
use alice_physics::solver::DistanceConstraint;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct JointInput {
    /// Number of bodies in the chain (capped at 8).
    body_count: u8,
    /// Per-body position offsets from the previous body (i8 to bound range).
    offsets: Vec<(i8, i8, i8)>,
    /// Per-constraint target distances (u8, mapped to 1..=32 to avoid 0).
    distances: Vec<u8>,
    /// Per-constraint compliance in 1/1000 units (0 = rigid, up to ~255/1000).
    compliances: Vec<u8>,
    /// Number of simulation steps (capped at 32).
    step_count: u8,
}

// Fuzz distance-constraint joint chains: build a chain of bodies coupled by
// distance constraints with arbitrary target distances and compliances, then
// step the solver. Must never panic regardless of input — the XPBD solver
// must remain robust under degenerate configurations (zero-length target,
// coincident anchors, unusual compliance ratios).
fuzz_target!(|input: JointInput| {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let body_count = (input.body_count as usize).clamp(2, 8);

    // Build a chain of bodies with cumulative offsets.
    let mut px: i64 = 0;
    let mut py: i64 = 10;
    let mut pz: i64 = 0;
    let mut body_ids = Vec::with_capacity(body_count);
    for i in 0..body_count {
        let (dx, dy, dz) = input.offsets.get(i).copied().unwrap_or((1, 0, 0));
        px = px.saturating_add(dx as i64);
        py = py.saturating_add(dy as i64);
        pz = pz.saturating_add(dz as i64);
        let body = RigidBody::new_dynamic(Vec3Fix::from_int(px, py, pz), Fix128::ONE);
        body_ids.push(world.add_body(body));
    }

    // Anchor the first body (infinite mass) so the chain has a fixed end.
    world.bodies[body_ids[0]].inv_mass = Fix128::ZERO;

    // Add distance constraints between consecutive bodies.
    for i in 0..(body_count - 1) {
        let dist_raw = input.distances.get(i).copied().unwrap_or(1);
        let dist = (dist_raw as i64).max(1); // avoid target_distance = 0
        let comp_raw = input.compliances.get(i).copied().unwrap_or(0);
        let compliance = Fix128::from_ratio(comp_raw as i64, 1000);

        let mut constraint = DistanceConstraint::new(
            body_ids[i],
            body_ids[i + 1],
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(dist),
        );
        constraint.compliance = compliance;
        world.add_distance_constraint(constraint);
    }

    let dt = Fix128::from_ratio(1, 60);
    let steps = (input.step_count as usize).min(32);
    for _ in 0..steps {
        world.step(dt);
    }
});
