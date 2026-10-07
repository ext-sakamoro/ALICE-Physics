//! A sphere cast that grazes a triangle mesh finds the hit: a cast that drifts
//! into the floor it rests on at a slope of 2^-31, and a cast along the floor
//! towards a ramp of 1e-8 rad.
//!
//! Both went through `ray_triangle`, whose parallel test used to drop every
//! ray within a sine of 2^-24 of the triangle's plane.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn quads(quads: &[[Vec3Fix; 4]]) -> TriMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for q in quads {
        let base = u32::try_from(vertices.len()).expect("small mesh");
        vertices.extend_from_slice(q);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    TriMesh::from_indexed(&vertices, &indices)
}

/// A sphere of radius 0.3 resting exactly on a flat floor and cast downward at
/// a slope of 2^-31 touches the floor at once: t = 0, for any cast length.
#[test]
fn a_sphere_resting_on_a_mesh_floor_and_cast_into_it_hits_at_zero() {
    let r = Fix128::from_ratio(3, 10);
    let s = Fix128::ONE / Fix128::from_int(1i64 << 31);
    for max_t in [100i64, 1000, 10000] {
        let l = Fix128::from_int(5 * max_t / 2);
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.add_static_collider(StaticCollider::TriMesh(quads(&[[
            v3(-l, Fix128::ZERO, -l),
            v3(-l, Fix128::ZERO, l),
            v3(l, Fix128::ZERO, l),
            v3(l, Fix128::ZERO, -l),
        ]])));
        let d = v3((Fix128::ONE - s * s).sqrt(), -s, Fix128::ZERO);
        let hit = w.cast_sphere(
            v3(Fix128::ZERO, r, Fix128::ZERO),
            r,
            d,
            Fix128::from_int(max_t),
            &RayFilter::default(),
        );
        assert_eq!(
            hit.map(|h| h.t),
            Some(Fix128::ZERO),
            "cast length {max_t}: the resting sphere must touch at t = 0"
        );
    }
}

/// Sliding along the floor towards a ramp of 1e-8 rad starting at x = 12, the
/// sphere (centre at x = 1) first touches the ramp where it rises by r tan(θ/2)
/// along the normal: t = 11 − r tan(θ/2) ≈ 10.9999999985.
#[test]
fn a_sphere_cast_along_a_mesh_floor_finds_a_shallow_ramp_ahead() {
    let r = Fix128::from_ratio(3, 10);
    // tan(1e-8) = 1e-8 to within 4e-25
    let k = Fix128::from_f64(1e-8);
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let p = |x: i64, y: Fix128, z: i64| v3(Fix128::from_int(x), y, Fix128::from_int(z));
    let z = Fix128::ZERO;
    let top = k * Fix128::from_int(28);
    w.add_static_collider(StaticCollider::TriMesh(quads(&[
        [p(-20, z, -20), p(-20, z, 20), p(12, z, 20), p(12, z, -20)],
        [p(12, z, -20), p(12, z, 20), p(40, top, 20), p(40, top, -20)],
    ])));
    let hit = w
        .cast_sphere(
            v3(Fix128::ONE, r, Fix128::ZERO),
            r,
            v3(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            Fix128::from_int(30),
            &RayFilter::default(),
        )
        .expect("the ramp is ahead of the sphere");
    // r tan(θ/2) = 0.3 · 0.5e-8 to within 1e-25
    let want = 11.0 - 0.3 * 0.5e-8;
    assert!(
        (hit.t.to_f64() - want).abs() < 1e-6,
        "first contact at t = {}, expected {want}",
        hit.t.to_f64()
    );
}
