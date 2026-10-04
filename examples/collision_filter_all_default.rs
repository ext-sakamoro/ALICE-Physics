//! Catch-all and default collision filters in a `PhysicsWorld` scene
//!
//! Reaches `CollisionFilter::ALL`, `CollisionFilter::DEFAULT`,
//! `layers::ALL` and `layers::DEFAULT` through `PhysicsWorld::set_body_filter`
//! and `PhysicsWorld::step`.
//!
//! Every outcome is derived by hand from the documented rule
//! `(a.layer & b.mask) != 0 && (b.layer & a.mask) != 0`, gated first by
//! "same non-zero group never collides" (`src/filter.rs`), and from the
//! documented constant values (`ALL` = every bit, `DEFAULT` = layer 1 / bit 0).
//! Each pair overlaps by `0.5` (two unit spheres `1.5` apart) and the pairs sit
//! `10` apart, so a pair reports a contact exactly when its filters allow it.
//!
//! Run with: `cargo run --example collision_filter_all_default`

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{layers, CollisionFilter};

/// Zero gravity and one substep, so `contact_events` reflects a single
/// detection pass (same set-up as `examples/collision_filter_categories.rs`).
fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..PhysicsConfig::default()
    }
}

/// One pair of overlapping unit spheres, the `k`-th pair sits at `x = 10 k`.
fn add_pair(
    world: &mut PhysicsWorld,
    k: i64,
    fa: CollisionFilter,
    fb: CollisionFilter,
) -> (usize, usize) {
    let x = Fix128::from_int(10 * k);
    let a = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO), Fix128::ONE),
        Fix128::ONE,
    );
    let b = world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(x + Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        Fix128::ONE,
    );
    world.set_body_filter(a, fa);
    world.set_body_filter(b, fb);
    (a, b)
}

fn main() {
    // ---- the constants themselves (documented values) -------------------
    assert_eq!(layers::ALL, u32::MAX, "layers::ALL is every bit");
    assert_eq!(layers::DEFAULT, 1, "layers::DEFAULT is bit 0");
    assert_eq!(
        (
            CollisionFilter::ALL.layer,
            CollisionFilter::ALL.mask,
            CollisionFilter::ALL.group
        ),
        (u32::MAX, u32::MAX, 0)
    );
    assert_eq!(
        CollisionFilter::DEFAULT,
        CollisionFilter::new(layers::DEFAULT, layers::ALL),
        "DEFAULT is layer 1 colliding with every layer"
    );
    assert_eq!(CollisionFilter::default(), CollisionFilter::DEFAULT);

    // ---- pairs, with the expected outcome derived from the rule ----------
    // (filter a, filter b, collides?, why)
    let only_vehicle = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE);
    let cases: [(CollisionFilter, CollisionFilter, bool, &str); 7] = [
        // ALL.layer & VEHICLE != 0 and VEHICLE & ALL.mask != 0.
        (
            CollisionFilter::ALL,
            only_vehicle,
            true,
            "ALL meets a narrow filter",
        ),
        // NONE has layer 0 and mask 0: both terms are 0.
        (
            CollisionFilter::ALL,
            CollisionFilter::NONE,
            false,
            "ALL cannot reach NONE",
        ),
        // Same non-zero group wins over the masks.
        (
            CollisionFilter::ALL.with_group(7),
            CollisionFilter::ALL.with_group(7),
            false,
            "ALL in a shared group",
        ),
        // DEFAULT (bit 0, mask every bit) against a mask of bit 0 only.
        (
            CollisionFilter::DEFAULT,
            CollisionFilter::new(1 << 4, 1 << 0),
            true,
            "DEFAULT layer is bit 0",
        ),
        // Same partner with mask bit 1: DEFAULT has no bit 1.
        (
            CollisionFilter::DEFAULT,
            CollisionFilter::new(1 << 4, 1 << 1),
            false,
            "DEFAULT layer is not bit 1",
        ),
        // layers::ALL as a mask accepts DEBRIS; DEBRIS-only mask accepts layer DEBRIS.
        (
            CollisionFilter::new(layers::DEBRIS, layers::ALL),
            CollisionFilter::new(layers::DEFAULT, layers::DEBRIS),
            true,
            "layers::ALL mask",
        ),
        // STATIC & DEBRIS == 0, so the other direction fails.
        (
            CollisionFilter::new(layers::STATIC, layers::ALL),
            CollisionFilter::new(layers::DEFAULT, layers::DEBRIS),
            false,
            "layers::ALL mask is one direction only",
        ),
    ];

    let mut world = PhysicsWorld::new(weightless());
    let mut ids = Vec::new();
    for (k, (fa, fb, _, _)) in cases.iter().enumerate() {
        ids.push(add_pair(&mut world, k as i64, *fa, *fb));
    }

    world.step(Fix128::from_ratio(1, 64));

    let pairs: Vec<(usize, usize)> = world
        .contact_events()
        .iter()
        .map(|e| (e.body_a.min(e.body_b), e.body_a.max(e.body_b)))
        .collect();

    for ((a, b), (_, _, want, why)) in ids.iter().zip(cases.iter()) {
        let got = pairs.contains(&(*a.min(b), *a.max(b)));
        println!("[filter] {why:<40} contact = {got} (expected {want})");
        assert_eq!(got, *want, "{why}");
    }
    let expected_contacts = cases.iter().filter(|c| c.2).count();
    assert_eq!(
        pairs.len(),
        expected_contacts,
        "no contact outside the listed pairs"
    );
    println!("[filter] {expected_contacts} contacts, all as derived from the layer/mask rule");
}
