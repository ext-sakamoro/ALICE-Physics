//! Oracles for bodies with a shape or a compound against SDF colliders.
//!
//! # What is measured
//!
//! A body used to meet an SDF as a sphere of one global radius whatever its shape.
//! Now a body with a shape (or a compound) meets it as that shape. Against a **flat**
//! SDF (a half-space `y < offset`) the deepest point of a convex solid has a closed
//! form from the geometry: the lowest corner of a turned box, `h − (|sin θ|·hx +
//! cos θ·hy)`; the lowest rim point of a tilted cylinder, `h − (hh·cos θ + r·sin θ)`;
//! the lowest end of a tilted capsule, `h − cos θ − r`. Against a **curved** SDF
//! (a ball obstacle) a dense brute-force sample of the box is the reference, and the
//! sampled contact is checked to be within a stated fraction of it and never deeper.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere};
use alice_physics::compound::CompoundShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn rot_z(angle: f64) -> QuatFix {
    QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(angle))
}

/// The half-space `y < offset` (negative below), as a static SDF collider.
fn plane(offset: f32) -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            move |_, y, _| y - offset,
            |_, _, _| (0.0, 1.0, 0.0),
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

/// A ball obstacle of radius `r` about the origin (negative inside).
fn ball(r: f32) -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            move |x, y, z| (x * x + y * y + z * z).sqrt() - r,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                if l < 1e-9 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / l, y / l, z / l)
                }
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

fn world_with(sdf: SdfCollider) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    w.add_sdf_collider(sdf);
    w
}

/// The one contact of the world's single body, as `(depth, normal)`; `None` when the
/// body is clear of the field.
fn contact_of(w: &PhysicsWorld) -> Option<(f64, [f64; 3])> {
    let contacts = w.sdf_contacts();
    assert!(
        contacts.len() <= 1,
        "one body, one field: {}",
        contacts.len()
    );
    contacts.first().map(|(_, c)| {
        (
            c.depth.to_f64(),
            [
                c.normal.x.to_f64(),
                c.normal.y.to_f64(),
                c.normal.z.to_f64(),
            ],
        )
    })
}

fn expect_depth(w: &PhysicsWorld, depth: f64, what: &str) {
    let (got, normal) =
        contact_of(w).unwrap_or_else(|| panic!("{what}: no contact, expected {depth}"));
    assert!(
        (got - depth).abs() < 2e-5,
        "{what}: depth {got} but the closed form is {depth}"
    );
    assert!(
        (normal[1] - 1.0).abs() < 1e-5 && normal[0].abs() < 1e-5 && normal[2].abs() < 1e-5,
        "{what}: the normal of a floor is +y, got {normal:?}"
    );
}

/// A turned box reaches into the floor by its lowest corner, for the axis-aligned
/// case (the AABB path) and for several turns (the corner path).
#[test]
fn a_turned_box_reaches_into_a_floor_by_its_lowest_corner() {
    let (hx, hy, hz) = (1.0, 0.5, 0.75);
    for &theta in &[0.0, 0.3, 0.9, 1.4] {
        let reach = f64::sin(theta) * hx + f64::cos(theta) * hy; // the box's half-height
        let mut w = world_with(plane(0.0));
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: v3(hx, hy, hz),
                },
                Fix128::from_int(1000),
                v3(0.0, reach - 0.2, 0.0),
            )
            .expect("a valid solid");
        w.get_body_mut(b).expect("body").rotation = rot_z(theta);
        expect_depth(&w, 0.2, &format!("box turned {theta}"));
        // The same box lifted 0.1 clear of the floor: no contact.
        w.get_body_mut(b).expect("body").position = v3(0.0, reach + 0.1, 0.0);
        assert!(
            contact_of(&w).is_none(),
            "box turned {theta}, above the floor"
        );
    }
}

/// A tilted cylinder reaches in by `hh·cos θ + r·sin θ` below its centre: the
/// generic support path, exact against a flat field.
#[test]
fn a_tilted_cylinder_reaches_into_a_floor_by_its_lowest_rim_point() {
    let (r, hh) = (0.6, 1.0);
    for &theta in &[0.0, 0.5, 1.1] {
        let reach = hh * f64::cos(theta) + r * f64::sin(theta);
        let mut w = world_with(plane(0.0));
        let b = w
            .add_shaped_body(
                &Shape::Cylinder {
                    radius: fx(r),
                    half_height: fx(hh),
                },
                Fix128::from_int(1000),
                v3(0.0, reach - 0.15, 0.0),
            )
            .expect("a valid solid");
        w.get_body_mut(b).expect("body").rotation = rot_z(theta);
        expect_depth(&w, 0.15, &format!("cylinder tilted {theta}"));
    }
}

/// An upright cone stands on its base: the base is `H/2` below its centre of mass,
/// so a centre of mass at `H/2 − 0.1` sinks the base by 0.1.
#[test]
fn an_upright_cone_stands_on_its_base() {
    let (r, hh) = (0.8, 1.2);
    let mut w = world_with(plane(0.0));
    w.add_shaped_body(
        &Shape::Cone {
            radius: fx(r),
            half_height: fx(hh),
        },
        Fix128::from_int(1000),
        v3(0.0, hh / 2.0 - 0.1, 0.0),
    )
    .expect("a valid solid");
    expect_depth(&w, 0.1, "cone base");
}

/// A sphere body (an ellipsoid of equal radii) meets a ball obstacle at the sum of
/// the radii.
#[test]
fn a_spherical_body_meets_a_ball_at_the_sum_of_the_radii() {
    let mut w = world_with(ball(2.0));
    w.add_shaped_body(
        &Shape::Ellipsoid {
            radii: v3(0.5, 0.5, 0.5),
        },
        Fix128::from_int(1000),
        v3(2.2, 0.0, 0.0),
    )
    .expect("a valid solid");
    // The centre is 2.2 from the ball's centre: 0.2 outside its surface; the body's
    // radius 0.5 reaches 0.3 in.
    let (depth, normal) = contact_of(&w).expect("touching");
    assert!((depth - 0.3).abs() < 2e-5, "depth {depth}");
    assert!(
        (normal[0] - 1.0).abs() < 1e-5,
        "the normal points away from the ball"
    );
}

/// A compound placed so that the lowest point of its children (given in the
/// authored frame, `authored_low`) is `sink` below the floor, and the depth the
/// world reports for it. The body's origin is the compound's centre of mass, so the
/// lowest point is `authored_low − com.y` above the body.
fn compound_depth(c: &CompoundShape, authored_low: f64, sink: f64) -> Option<f64> {
    let density = Fix128::from_int(1000);
    let com_y = c.mass_properties(density).center_of_mass.y.to_f64();
    let mut w = world_with(plane(0.0));
    w.add_compound_body(c, density, v3(0.0, -(authored_low - com_y) - sink, 0.0))
        .expect("the children have volume");
    contact_of(&w).map(|(d, _)| d)
}

fn capsule_child(theta: f64) -> (CompoundShape, f64) {
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0), fx(0.5)),
        v3(-6.0, 0.0, 0.0),
        rot_z(theta),
    );
    (c, -f64::cos(theta) - 0.5)
}

/// Each kind of child meets the floor by its own lowest point: a tilted capsule
/// (its lower end sphere), a box, a convex hull (its lowest vertex) and a sphere
/// child whose offset the child's turn moves.
#[test]
fn each_kind_of_child_reaches_in_by_its_lowest_point() {
    let sink = 0.1;
    let (capsule, capsule_low) = capsule_child(0.4);
    let mut boxed = CompoundShape::new();
    boxed.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), QuatFix::IDENTITY),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let mut hull = CompoundShape::new();
    hull.add_convex_hull(
        ConvexHull::new(vec![
            v3(0.0, -0.3, 0.0),
            v3(1.0, 0.7, 0.0),
            v3(-1.0, 0.7, 0.5),
            v3(0.0, 0.7, -1.0),
        ]),
        v3(6.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let mut ball_child = CompoundShape::new();
    // The sphere's own offset (1, 0, 0) is turned by the child's rotation 0.5 about z.
    ball_child.add_sphere(
        Sphere::new(v3(1.0, 0.0, 0.0), fx(0.4)),
        v3(0.0, 0.2, 0.0),
        rot_z(0.5),
    );
    let sphere_low = 0.2 + f64::sin(0.5) - 0.4;
    // The capsule turned over (by pi + 0.4): its other end is now the lower one.
    // (the lowest point is the same value: the capsule is symmetric end for end)
    let (flipped, _) = capsule_child(0.4 + std::f64::consts::PI);
    let flipped_low = capsule_low;
    // A box with a rotation of its own (0.3) inside a child turned 0.5, its centre
    // offset by (0.5, 0.2, 0): the offset is turned by the child, the box by both.
    let mut turned_box = CompoundShape::new();
    turned_box.add_box(
        OrientedBox::new(v3(0.5, 0.2, 0.0), v3(1.0, 0.5, 0.75), rot_z(0.3)),
        Vec3Fix::ZERO,
        rot_z(0.5),
    );
    let box_low =
        0.5 * f64::sin(0.5) + 0.2 * f64::cos(0.5) - (f64::sin(0.8) * 1.0 + f64::cos(0.8) * 0.5);
    // A tetrahedron in a child turned 0.7: the lowest of its turned vertices.
    let tetra = [
        (0.0, -0.3, 0.0),
        (1.0, 0.7, 0.0),
        (-1.0, 0.7, 0.5),
        (0.0, 0.7, -1.0),
    ];
    let mut turned_hull = CompoundShape::new();
    turned_hull.add_convex_hull(
        ConvexHull::new(tetra.iter().map(|&(x, y, z)| v3(x, y, z)).collect()),
        Vec3Fix::ZERO,
        rot_z(0.7),
    );
    let hull_low = tetra
        .iter()
        .map(|&(x, y, _)| x * f64::sin(0.7) + y * f64::cos(0.7))
        .fold(f64::MAX, f64::min);
    for (name, c, low) in [
        ("capsule", &capsule, capsule_low),
        ("capsule, upside down", &flipped, flipped_low),
        ("box", &boxed, -1.0),
        ("box turned twice with an offset", &turned_box, box_low),
        ("hull", &hull, -0.3),
        ("hull in a turned child", &turned_hull, hull_low),
        ("sphere", &ball_child, sphere_low),
    ] {
        let depth = compound_depth(c, low, sink).unwrap_or_else(|| panic!("{name}: no contact"));
        assert!(
            (depth - sink).abs() < 2e-5,
            "{name}: depth {depth}, the closed form is {sink}"
        );
        assert!(
            compound_depth(c, low, -0.1).is_none(),
            "{name}: 0.1 above the floor touches nothing"
        );
    }
}

/// The deepest child decides, wherever it is in the list: a box and a hull first
/// and the capsule last, 0.6 below the floor at its lowest, which puts the box 0.18
/// in (`0.6 − (cos 0.4 + 0.5 − 1)`); the hull stays clear.
#[test]
fn the_deepest_child_decides_the_depth_against_a_floor() {
    let theta = 0.4;
    let (capsule, capsule_low) = capsule_child(theta);
    let mut c = CompoundShape::new();
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), QuatFix::IDENTITY),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    c.add_convex_hull(
        ConvexHull::new(vec![
            v3(0.0, -0.3, 0.0),
            v3(1.0, 0.7, 0.0),
            v3(-1.0, 0.7, 0.5),
            v3(0.0, 0.7, -1.0),
        ]),
        v3(6.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    c.children.push(capsule.children[0].clone());
    let depth = compound_depth(&c, capsule_low, 0.6).expect("touching");
    assert!((depth - 0.6).abs() < 2e-5, "capsule lowest, last: {depth}");
    // Without the capsule the box (lowest at -1) is the deepest of the two left: the
    // same sink puts it 0.6 in, the hull (lowest -0.3) 0.6 - 0.7 < 0: clear.
    c.children.pop();
    let depth = compound_depth(&c, -1.0, 0.6).expect("touching");
    assert!((depth - 0.6).abs() < 2e-5, "box lowest: {depth}");
}

/// Against a ball an axis-aligned box lands flat: the face centre is the deepest
/// point, and the box's depth is `R − y_bottom` there (a corner is shallower).
#[test]
fn a_box_lands_flat_on_a_ball_by_its_face_centre() {
    let big_r = 3.0;
    let mut w = world_with(ball(big_r as f32));
    // Bottom face at y = 2.8, 0.2 below the ball's top (3.0): half-height 1, centre 3.8.
    w.add_shaped_body(
        &Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Fix128::from_int(1000),
        v3(0.0, 3.8, 0.0),
    )
    .expect("a valid solid");
    let (depth, normal) = contact_of(&w).expect("touching");
    assert!(
        (depth - 0.2).abs() < 2e-5,
        "face centre: depth {depth}, the closed form is 0.2"
    );
    assert!(
        (normal[1] - 1.0).abs() < 1e-5,
        "the face centre is straight above the ball's centre"
    );
}

/// A turned box on a ball: the deepest point can be an edge between the samples, so
/// the sampled depth is never deeper than the brute-force truth and within 15% of
/// it.
#[test]
fn a_turned_box_on_a_ball_is_within_the_sampling_error() {
    let (r, centre_y, theta) = (3.0f64, 3.9f64, 0.35f64);
    let (hx, hy, hz) = (1.0, 0.8, 0.9);
    let mut w = world_with(ball(r as f32));
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(hx, hy, hz),
            },
            Fix128::from_int(1000),
            v3(0.0, centre_y, 0.0),
        )
        .expect("a valid solid");
    w.get_body_mut(b).expect("body").rotation = rot_z(theta);
    let (depth, _) = contact_of(&w).expect("touching");

    // Brute force: the deepest point of the solid box, from a dense grid.
    let (s, c) = (theta.sin(), theta.cos());
    let n = 60;
    let mut truth = 0.0f64;
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let l = [
                    hx * (2.0 * i as f64 / n as f64 - 1.0),
                    hy * (2.0 * j as f64 / n as f64 - 1.0),
                    hz * (2.0 * k as f64 / n as f64 - 1.0),
                ];
                let p = [c * l[0] - s * l[1], centre_y + s * l[0] + c * l[1], l[2]];
                let d = r - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                truth = truth.max(d);
            }
        }
    }
    assert!(truth > 0.05, "the scene is in contact: {truth}");
    assert!(
        depth <= truth + 1e-5,
        "sampled {depth} is deeper than the truth {truth}"
    );
    assert!(
        depth >= 0.85 * truth,
        "sampled {depth} is more than 15% short of the truth {truth}"
    );
}

/// The correction moves a body along the field's normal by the depth: after a step
/// a turned box rests with its lowest corner exactly on the floor.
#[test]
fn a_step_leaves_a_turned_box_resting_on_the_floor() {
    let (hx, hy, hz, theta) = (1.0, 0.5, 0.75, 0.6);
    let reach = f64::sin(theta) * hx + f64::cos(theta) * hy;
    let mut w = world_with(plane(0.0));
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(hx, hy, hz),
            },
            Fix128::from_int(1000),
            v3(0.0, reach - 0.25, 0.0),
        )
        .expect("a valid solid");
    w.get_body_mut(b).expect("body").rotation = rot_z(theta);
    w.step(Fix128::from_ratio(1, 60));
    let y = w.get_body(b).expect("body").position.y.to_f64();
    assert!(
        (y - reach).abs() < 2e-5,
        "the centre rests at the half-height {reach}, not at {y}"
    );
    assert!(contact_of(&w).is_none() || contact_of(&w).is_some_and(|(d, _)| d < 2e-5));
}

/// An SDF scaled by `s` is the plane `y = s·offset`: closed form for the depth.
#[test]
fn a_scaled_sdf_moves_the_surface_by_the_scale() {
    let sdf = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |_, y, _| y - 1.0,
            |_, _, _| (0.0, 1.0, 0.0),
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
    .with_scale(fx(2.5));
    let mut w = world_with(sdf);
    // A sphere body of radius 0.5 centred at y = 2.3: the plane is at y = 2.5, so
    // the sphere reaches 0.5 + 0.2 = 0.7 into it... centre is 0.2 below the plane.
    w.add_shaped_body(
        &Shape::Ellipsoid {
            radii: v3(0.5, 0.5, 0.5),
        },
        Fix128::from_int(1000),
        v3(0.0, 2.3, 0.0),
    )
    .expect("a valid solid");
    let (depth, _) = contact_of(&w).expect("touching");
    assert!(
        (depth - 0.7).abs() < 2e-5,
        "depth {depth}, the closed form is 0.7"
    );
}

/// A body without a shape is still a sphere of the global radius; static bodies,
/// sensors and an SDF attached to the body itself are skipped.
#[test]
fn plain_bodies_static_bodies_and_sensors_keep_their_old_behaviour() {
    use alice_physics::solver::RigidBody;
    let mut w = world_with(plane(0.0));
    w.set_sdf_collision_radius(fx(0.5));
    // A plain body 0.3 above the floor: its sphere of radius 0.5 reaches 0.2 in.
    let plain = w.add_body(RigidBody::new(v3(0.0, 0.3, 0.0), Fix128::ONE));
    let contacts = w.sdf_contacts();
    assert_eq!(contacts.len(), 1);
    assert_eq!(contacts[0].0, plain);
    assert!((contacts[0].1.depth.to_f64() - 0.2).abs() < 2e-5);
    // A static body below the floor is never reported.
    w.add_body(RigidBody::new_static(v3(0.0, -5.0, 0.0)));
    assert_eq!(w.sdf_contacts().len(), 1);
    // A sensor is skipped.
    let mut sensor = RigidBody::new(v3(0.0, 0.1, 0.0), Fix128::ONE);
    sensor.is_sensor = true;
    w.add_body(sensor);
    assert_eq!(w.sdf_contacts().len(), 1);
}

/// An ellipsoid with unequal radii is not a sphere: it reaches into a floor by its
/// vertical half-extent `sqrt((a sin θ)² + (b cos θ)²)` when turned by θ about z.
#[test]
fn an_ellipsoid_body_reaches_into_a_floor_by_its_vertical_half_extent() {
    let (a, b, c) = (1.0, 0.4, 0.6);
    for &theta in &[0.3, 1.0] {
        let half = f64::hypot(a * f64::sin(theta), b * f64::cos(theta));
        let mut w = world_with(plane(0.0));
        let body = w
            .add_shaped_body(
                &Shape::Ellipsoid { radii: v3(a, b, c) },
                Fix128::from_int(1000),
                v3(0.0, half - 0.12, 0.0),
            )
            .expect("a valid solid");
        w.get_body_mut(body).expect("body").rotation = rot_z(theta);
        expect_depth(&w, 0.12, &format!("ellipsoid turned {theta}"));
    }
}

/// An eccentric ellipsoid beside a ball: its nearest point to the ball is found by
/// chasing the support point along the field's normal (the first support point, at
/// the centre's normal, is not yet the nearest). The reference is a dense sample of
/// the ellipsoid's surface.
#[test]
fn an_eccentric_ellipsoid_beside_a_ball_finds_its_nearest_point() {
    let (cx, cy, a, b, r) = (2.0f64, 1.6f64, 1.2f64, 0.35f64, 2.5f64);
    let mut w = world_with(ball(r as f32));
    w.add_shaped_body(
        &Shape::Ellipsoid { radii: v3(a, b, b) },
        Fix128::from_int(1000),
        v3(cx, cy, 0.0),
    )
    .expect("a valid solid");
    let (depth, _) = contact_of(&w).expect("touching");
    let n = 300;
    let mut truth = f64::MIN;
    for i in 0..=n {
        let th = std::f64::consts::PI * i as f64 / n as f64;
        for j in 0..2 * n {
            let ph = std::f64::consts::PI * j as f64 / n as f64;
            let p = [
                cx + a * th.cos(),
                cy + b * th.sin() * ph.cos(),
                b * th.sin() * ph.sin(),
            ];
            truth = truth.max(r - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
        }
    }
    assert!(truth > 0.5, "the scene is in contact: {truth}");
    assert!(
        depth <= truth + 1e-4,
        "sampled {depth} is deeper than the truth {truth}"
    );
    assert!(
        (truth - depth).abs() < 2e-3,
        "the nearest point is found to 2e-3: {depth} against {truth}"
    );
}

/// A box beside a ball reaches in by its nearest face centre, on an axis that is not
/// the box's shortest: the z extent is 0.75, not 0.5.
#[test]
fn a_box_beside_a_ball_reaches_in_by_its_nearest_face_centre() {
    let r = 2.0;
    let mut w = world_with(ball(r as f32));
    // The -z face is at z = d - 0.75; with d = 2.75 + (-0.15) it is 0.15 inside.
    w.add_shaped_body(
        &Shape::Box {
            half_extents: v3(1.0, 0.5, 0.75),
        },
        Fix128::from_int(1000),
        v3(0.0, 0.0, 2.6),
    )
    .expect("a valid solid");
    let (depth, normal) = contact_of(&w).expect("touching");
    assert!((depth - 0.15).abs() < 2e-5, "depth {depth}");
    assert!(
        (normal[2] - 1.0).abs() < 1e-5,
        "pushed along +z: {normal:?}"
    );
}

/// An SDF attached to a body does not collide with that body.
#[test]
fn an_sdf_attached_to_a_body_skips_that_body() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    let body = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(1.0, 1.0, 1.0),
            },
            Fix128::from_int(1000),
            v3(0.0, -0.5, 0.0),
        )
        .expect("a valid solid");
    // A floor attached to that very body: the body is deep inside it, and is skipped.
    w.add_sdf_collider(SdfCollider::new_dynamic(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        body,
    ));
    assert!(w.sdf_contacts().is_empty());
}
