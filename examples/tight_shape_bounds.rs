//! Tight world boxes and surface areas of the convex shapes, and the support of
//! a compound body.
//!
//! Each shape's `aabb` is the smallest world-axis box of the turned solid (a
//! cylinder along `â` reaches `|â_i| hh + r √(1 − â_i²)` on axis `i`, a cone is
//! the hull of its apex and base disc, an ellipsoid reaches `√(Σ_j R_ij² r_j²)`,
//! a torus `R √(1 − â_i²) + r`). A body's collider box in the world is the same
//! box, so the narrow phase and the ray caster skip pairs whose solids are far
//! apart even when their bounding spheres overlap.
//!
//! Every printed value is checked against its closed form.
//!
//! ```bash
//! cargo run --example tight_shape_bounds
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{gjk, Capsule, Sphere, Support};
use alice_physics::compound::{CompoundShape, ShapeRef, TransformedCompound};
use alice_physics::cone::Cone;
use alice_physics::cylinder::Cylinder;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsWorld, SolverConfig};
use alice_physics::torus::Torus;
use alice_physics::wedge::Wedge;

use std::f64::consts::PI;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn check(name: &str, got: Fix128, want: f64) {
    let g = got.to_f64();
    println!("{name:<40} {g:>14.9}  (closed form {want:.9})");
    assert!(
        (g - want).abs() <= 1e-9 * want.abs().max(1.0),
        "{name}: {g} vs {want}"
    );
}

fn surface_areas() {
    println!("-- surface areas");
    let b = OrientedBox::new(Vec3Fix::ZERO, v3(0.5, 1.0, 2.0), QuatFix::IDENTITY);
    // 2 (ab + bc + ca) for sides (1, 2, 4)
    check("box 1 x 2 x 4", b.surface_area(), 2.0 * (2.0 + 8.0 + 4.0));
    let c = Cone::new(Vec3Fix::ZERO, fx(3.0), fx(2.0));
    // pi r (r + sqrt(r^2 + h^2)) with r = 3, h = 4: slant 5
    check("cone r 3, h 4", c.surface_area(), PI * 3.0 * 8.0);
    let y = Cylinder::new(Vec3Fix::ZERO, fx(1.5), fx(0.5));
    // 2 pi r (r + h) with r = 0.5, h = 3
    check(
        "cylinder r 0.5, h 3",
        y.surface_area(),
        2.0 * PI * 0.5 * 3.5,
    );
    let t = Torus::new(Vec3Fix::ZERO, fx(2.0), fx(0.25));
    // 4 pi^2 R r
    check(
        "torus R 2, r 0.25",
        t.surface_area(),
        4.0 * PI * PI * 2.0 * 0.25,
    );
}

fn turned_boxes() {
    println!("-- world boxes of turned solids");
    // a quarter turn about Z lays the Y axis along -X
    let quarter = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI).normalize();
    let cone = Cone::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(2.0), quarter).aabb();
    // apex at -2 on X, base disc about +2 on X with radius 1 in Y and Z
    check("cone lying along -X: min x", cone.min.x, -2.0);
    check("cone lying along -X: max x", cone.max.x, 2.0);
    check("cone lying along -X: max y", cone.max.y, 1.0);
    let wedge = Wedge::with_rotation(Vec3Fix::ZERO, fx(2.0), fx(1.0), fx(4.0), quarter).aabb();
    // the apex edge (0, 1/2) turns to x = -1/2, the base corners (±1, -1/2) to x = 1/2
    check("wedge turned: min x", wedge.min.x, -0.5);
    check("wedge turned: max y", wedge.max.y, 1.0);
    check("wedge turned: max z", wedge.max.z, 2.0);
    let tilt = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(PI / 3.0)).normalize();
    let cyl = Cylinder::with_rotation(Vec3Fix::ZERO, fx(3.0), fx(0.5), tilt).aabb();
    // axis (-sin 60, cos 60, 0): x reaches 3 sin 60 + 0.5 cos 60
    let (s, c) = (3.0f64.sqrt() / 2.0, 0.5);
    check(
        "cylinder tilted 60 deg: max x",
        cyl.max.x,
        3.0 * s + 0.5 * c,
    );
    check("cylinder tilted 60 deg: max z", cyl.max.z, 0.5);
}

fn sample_compound() -> CompoundShape {
    let mut compound = CompoundShape::new();
    compound.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(0.5)),
        v3(2.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    compound.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 0.25, 0.25), QuatFix::IDENTITY),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    compound.add_capsule(
        Capsule::new(v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0), fx(0.25)),
        v3(-1.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    compound
}

fn compound_support() {
    println!("-- compound support");
    let compound = sample_compound();
    // each child's box in its own frame
    for child in &compound.children {
        let local = child.shape.aabb();
        let kind = match child.shape {
            ShapeRef::Sphere(_) => "sphere",
            ShapeRef::Capsule(_) => "capsule",
            ShapeRef::Box(_) => "box",
            ShapeRef::ConvexHull(_) => "hull",
        };
        println!(
            "{kind:<8} local box x [{:.3}, {:.3}] y [{:.3}, {:.3}]",
            local.min.x.to_f64(),
            local.max.x.to_f64(),
            local.min.y.to_f64(),
            local.max.y.to_f64()
        );
    }
    let at = v3(10.0, 0.0, 0.0);
    // +X: the sphere at x = 2 reaches 2.5; -X: the capsule at x = -1 reaches -1.25;
    // +Y: the capsule's top cap reaches 1.25
    let plus_x = compound.support_world(Vec3Fix::UNIT_X, at, QuatFix::IDENTITY);
    check("support +X", plus_x.x, 12.5);
    let minus_x = compound.support_world(-Vec3Fix::UNIT_X, at, QuatFix::IDENTITY);
    check("support -X", minus_x.x, 8.75);
    let posed = TransformedCompound {
        compound: &compound,
        position: at,
        rotation: QuatFix::IDENTITY,
    };
    check("support +Y", posed.support(Vec3Fix::UNIT_Y).y, 1.25);
    // a ball touching the sphere child is met by GJK on the posed compound
    let ball = Sphere::new(v3(12.9, 0.0, 0.0), fx(0.5));
    assert!(gjk(&posed, &ball).colliding);
    let far = Sphere::new(v3(13.1, 0.0, 0.0), fx(0.5));
    assert!(!gjk(&posed, &far).colliding);
    println!("ball at x = 12.9 meets the compound, at x = 13.1 it does not");
}

fn world_ray() {
    println!("-- ray caster over tight collider boxes");
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    // a long thin rod along X: its bounding sphere has radius ~5, its box is
    // 0.2 thick in Y
    let rod = Shape::Cylinder {
        radius: fx(0.1),
        half_height: fx(5.0),
    };
    let i = world
        .add_shaped_body(&rod, Fix128::ONE, Vec3Fix::ZERO)
        .expect("valid rod");
    world.bodies[i].rotation =
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI).normalize();
    let caster = world.ray_caster(RayFilter::new());
    let across = |y: f64| {
        !caster
            .candidates(v3(-20.0, y, 0.0), Vec3Fix::UNIT_X, fx(40.0))
            .is_empty()
    };
    assert!(across(0.05));
    assert!(!across(1.0));
    println!("a ray 1.0 above the rod skips it; one 0.05 above is tested");
}

fn main() {
    surface_areas();
    turned_boxes();
    compound_support();
    world_ray();
}
