//! A collision mesh from a signed distance field
//!
//! A ball given as an SDF becomes a closed triangle mesh (marching tetrahedra),
//! is simplified by edge collapse, and serves as a static collider a sphere body
//! comes to rest on. The printout checks the mesh against the ball's closed forms:
//! `V − E + F = 2` for a sphere, the enclosed volume against `4πR³/3`, and the
//! height the body is lifted to against `R + r`.
//!
//! ```bash
//! cargo run --example collision_mesh_from_sdf --features std
//! ```

use std::collections::BTreeSet;

use alice_physics::collision_mesh_gen::{
    compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
    CollisionMeshConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// `V − E + F` and the volume the triangles enclose (divergence theorem).
fn describe(mesh: &CollisionMesh) -> (i64, f64) {
    let edges: BTreeSet<(usize, usize)> = mesh
        .triangles
        .iter()
        .flat_map(|t| (0..3).map(move |k| (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]))))
        .collect();
    let euler = mesh.vertices.len() as i64 - edges.len() as i64 + mesh.triangles.len() as i64;
    let volume = mesh
        .triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (
                mesh.vertices[t[0]],
                mesh.vertices[t[1]],
                mesh.vertices[t[2]],
            );
            a.dot(b.cross(c)).to_f64() / 6.0
        })
        .sum();
    (euler, volume)
}

fn main() {
    let radius = 1.0;
    let ball = move |p: Vec3Fix| p.length() - Fix128::from_f64(radius);
    let config = CollisionMeshConfig {
        resolution: 30,
        bounds_min: v3(-1.5, -1.5, -1.5),
        bounds_max: v3(1.5, 1.5, 1.5),
    };

    let mesh = generate_collision_mesh(ball, &config);
    let (euler, volume) = describe(&mesh);
    let ideal = 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
    println!(
        "mesh: {} vertices, {} triangles, V - E + F = {euler} (2), volume {volume:.4} of {ideal:.4}",
        mesh.vertices.len(),
        mesh.triangles.len()
    );
    let aabb = compute_mesh_aabb(&mesh);
    println!(
        "bounding box x from {:.3} to {:.3} (the ball reaches ±{radius})",
        aabb.min.x.to_f64(),
        aabb.max.x.to_f64()
    );

    let lighter = simplify_collision_mesh(&mesh, mesh.triangles.len() / 4);
    let (euler, volume) = describe(&lighter);
    println!(
        "simplified to {} triangles: V - E + F = {euler} (2), volume {volume:.4}",
        lighter.triangles.len()
    );

    // A sphere body of radius 0.5 dropped 0.1 into the top of the mesh is lifted to
    // R + r = 1.5.
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    world.add_static_collider(mesh.to_static_collider());
    let body = world.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, radius + 0.5 - 0.1, 0.0), Fix128::ONE),
        Fix128::from_f64(0.5),
    );
    world.step(Fix128::from_ratio(1, 60));
    println!(
        "a body dropped 0.1 into the mesh rests at y = {:.4} (R + r = {:.4})",
        world
            .get_body(body)
            .expect("the body just added")
            .position
            .y
            .to_f64(),
        radius + 0.5
    );
}
