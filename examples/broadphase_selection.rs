//! Choosing the broad-phase
//!
//! Two worlds with the same sixty bodies, one on the default BVH (rebuilt every
//! step) and one on the persistent dynamic tree. Both hand sorted candidate pairs
//! to the same exact narrow-phase, so after the same steps every body is in the
//! same place to the last bit. The tree keeps a fattened proxy per body and only
//! re-inserts a body that leaves it.
//!
//! ```bash
//! cargo run --example broadphase_selection --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// Sixty spheres of radius 0.5 on a jittered lattice, all heading for the middle.
fn crowd(kind: Broadphase) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    world.set_broadphase(kind);
    for i in 0..60 {
        let (x, y, z) = (f64::from(i % 5), f64::from((i / 5) % 4), f64::from(i / 20));
        let (px, py, pz) = (x * 1.3 - 2.6, y * 1.3 - 2.0, z * 1.3 - 1.3);
        let at = v3(px, py, pz);
        let body =
            world.add_body_with_radius(RigidBody::new(at, Fix128::ONE), Fix128::from_f64(0.5));
        world
            .get_body_mut(body)
            .expect("the body just added")
            // Every body heads for the middle, so the lattice squeezes together.
            .velocity = v3(-px * 0.5, -py * 0.5, -pz * 0.5);
    }
    world
}

fn main() {
    let mut bvh = crowd(Broadphase::Bvh);
    let mut tree = crowd(Broadphase::DynamicTree);
    let dt = Fix128::from_ratio(1, 60);
    let mut most_contacts = 0;
    for _ in 0..90 {
        bvh.step(dt);
        tree.step(dt);
        most_contacts = most_contacts.max(tree.contact_constraints.len());
    }
    let same = bvh
        .bodies
        .iter()
        .zip(&tree.bodies)
        .all(|(a, b)| a.position == b.position && a.velocity == b.velocity);
    println!("after 90 steps every body is bit-identical under both broad-phases: {same}");
    println!(
        "{:?} and {:?} resolved up to {most_contacts} contacts in one step",
        bvh.broadphase(),
        tree.broadphase()
    );

    let stats = tree.broadphase_stats();
    println!(
        "the tree holds {} proxies in {} levels (the BVH world keeps none: {})",
        stats.proxies,
        stats.height,
        bvh.broadphase_stats().proxies
    );
    let fat = tree.broadphase_proxy_aabb(0).expect("body 0 has a proxy");
    let at = tree.bodies[0].position;
    println!(
        "body 0 is at ({:.3}, {:.3}, {:.3}); its proxy box reaches x from {:.3} to {:.3} (radius 0.5 + margin 0.5 on each side, kept until it leaves)",
        at.x.to_f64(),
        at.y.to_f64(),
        at.z.to_f64(),
        fat.min.x.to_f64(),
        fat.max.x.to_f64()
    );
}
