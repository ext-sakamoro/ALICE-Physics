//! Spheres resting on planes, height fields and triangle meshes
//!
//! Three immovable surfaces — a tilted plane, a ramp-shaped height field and a
//! one-sided triangle ground — and one body started inside each. One step of a
//! zero-gravity world corrects the body out of the surface along the surface
//! normal, so it ends one collision radius from the surface, measured *along the
//! normal*: on a slope that is `r / cosθ` above the surface vertically, not `r`.
//! (Bodies on a slope slide down it under gravity: this path applies the
//! position correction only, with no friction.)
//!
//! The example also calls the primitives' own queries (`PlaneCollider` signed
//! distance and `flip`, `HeightField` interpolation and slope normal, `TriMesh`
//! capsule and box tests) and prints them next to the world's result.
//!
//! ```bash
//! cargo run --example static_colliders --features std
//! ```

use alice_physics::collider::AABB;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::{TriMesh, Triangle};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// A zero-gravity world with one static surface and one body started at `at`; the
/// body's position after one step, which is the collision correction alone.
fn corrected(collider: StaticCollider, at: Vec3Fix, radius: Fix128) -> Vec3Fix {
    let config = SolverConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.add_static_collider(collider);
    let body = world.add_body_with_radius(RigidBody::new_dynamic(at, Fix128::ONE), radius);
    assert_eq!(world.static_collider_count(), 1);
    world.step(Fix128::from_ratio(1, 60));
    let at_rest = world.get_body(body).expect("body").position;
    // The surface can be taken away again; the world then holds none.
    let removed = world.remove_static_collider(0);
    assert!(removed.is_some() && world.static_collider_count() == 0);
    at_rest
}

fn show(name: &str, p: Vec3Fix) {
    println!(
        "{name:<6} body ends at ({:.4}, {:.4}, {:.4})",
        p.x.to_f64(),
        p.y.to_f64(),
        p.z.to_f64()
    );
}

fn main() {
    let radius = Fix128::from_f64(0.5);

    // 1. A plane through the origin tilted 45° about X: normal (0, 1, 1)/√2.
    let plane = PlaneCollider::from_point_normal(Vec3Fix::ZERO, v3(0.0, 1.0, 1.0));
    let above = v3(0.0, 2.0, 0.0);
    println!(
        "plane: distance of (0,2,0) = {:.4}, in front: {}, flipped distance = {:.4}",
        plane.distance_to_point(above).to_f64(),
        plane.is_front(above),
        plane.flip().distance_to_point(above).to_f64(),
    );
    let foot = plane.project_point(above);
    println!(
        "       its foot on the plane is ({:.3}, {:.3}, {:.3})",
        foot.x.to_f64(),
        foot.y.to_f64(),
        foot.z.to_f64()
    );
    let box_hit = plane.intersect_aabb(&AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0)));
    println!(
        "       a unit box at the origin sinks {:.4}",
        box_hit.depth.to_f64()
    );
    show(
        "plane",
        corrected(StaticCollider::Plane(plane), v3(0.0, 0.3, 0.1), radius),
    );

    // 2. A ramp y = x / 2 as a height field, 12 × 12 points one unit apart.
    let mut ramp = HeightField::flat(12, 12, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    for gz in 0..12u32 {
        for gx in 0..12u32 {
            ramp.set_height(gx, gz, Fix128::from_f64(0.5 * f64::from(gx)));
        }
    }
    let n = ramp.sample_normal(Fix128::from_int(5), Fix128::from_int(5));
    println!(
        "ramp:  height at x = 5.5 is {:.4}, normal ({:.4}, {:.4}, {:.4}), \
         signed distance of (5, 3, 5) = {:.4}, corner height {:.2}",
        ramp.sample_height(Fix128::from_f64(5.5), Fix128::from_int(5))
            .to_f64(),
        n.x.to_f64(),
        n.y.to_f64(),
        n.z.to_f64(),
        ramp.signed_distance(v3(5.0, 3.0, 5.0)).to_f64(),
        ramp.get_height(11, 11).to_f64(),
    );
    show(
        "ramp",
        corrected(StaticCollider::HeightField(ramp), v3(5.0, 2.8, 5.0), radius),
    );

    // 3. A ground of two triangles at y = -2.
    let (a, b, c, d) = (
        v3(-30.0, -2.0, -30.0),
        v3(30.0, -2.0, -30.0),
        v3(30.0, -2.0, 30.0),
        v3(-30.0, -2.0, 30.0),
    );
    let ground = TriMesh::from_triangles(vec![Triangle::new(a, b, c), Triangle::new(a, c, d)]);
    let capsule = ground.collide_capsule(v3(-1.0, -1.8, 0.0), v3(1.0, -1.8, 0.0), radius);
    let boxed = ground.collide_aabb(&AABB::new(v3(-1.0, -2.5, -1.0), v3(1.0, -0.5, 1.0)));
    println!(
        "mesh:  {} triangles, capsule depth {:.4}, box depth {:.4}",
        ground.triangle_count(),
        capsule.map_or(0.0, |h| h.depth.to_f64()),
        boxed.map_or(0.0, |h| h.depth.to_f64()),
    );
    show(
        "mesh",
        corrected(StaticCollider::TriMesh(ground), v3(0.0, -1.8, 0.0), radius),
    );
}
