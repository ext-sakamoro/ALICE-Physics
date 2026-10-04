//! Soft Bodies on SDF Example
//!
//! Production entry point for `Cloth::step_with_sdf`,
//! `DeformableBody::step_with_sdf` and `Fluid::step_with_sdf`, the SDF
//! collision step of the three particle bodies (`Rope::step_with_sdf` is
//! driven by `rope_pin_constraints`).
//!
//! Every scene uses gravity zero and one substep, and starts at rest with
//! every internal constraint already satisfied, so the substep moves no
//! particle and the only displacement is the SDF response. That response has
//! a closed form against a plane:
//! - cloth keeps particles `thickness` above the surface: a flat sheet at
//!   `y = -0.5` lands at `y = thickness` (1/64 here, exact in `f32`)
//! - a deformable body is pushed onto the surface: a cube entirely below the
//!   plane `y = 0` ends with every vertex at `y = 0`, `x` and `z` unchanged
//! - a fluid is contained by the field (negative inside): a particle above the
//!   lid `y = 1` comes back to `y = 1`; with one particle the density
//!   constraint has no neighbour gradient, so the substep leaves it in place
//!
//! Run with: `cargo run --example soft_bodies_on_sdf`

use alice_physics::cloth::Cloth;
use alice_physics::deformable::DeformableBody;
use alice_physics::fluid::{Fluid, FluidConfig};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

/// The plane `y = h`, normal `+y` (negative below the plane).
fn plane_at(h: f32) -> SdfCollider {
    let field = ClosureSdf::new(move |_x, y, _z| y - h, |_x, _y, _z| (0.0, 1.0, 0.0));
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// The region below the lid `y = h` (negative inside, outward normal `+y`).
fn below_lid(h: f32) -> SdfCollider {
    plane_at(h)
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);

    // ---- cloth: a flat 3x3 sheet below the ground --------------------------
    let mut cloth = Cloth::new_grid(
        Vec3Fix::from_f32(-1.0, -0.5, -1.0),
        Fix128::from_int(2),
        Fix128::from_int(2),
        3,
        3,
        Fix128::ONE,
    );
    cloth.config.gravity = Vec3Fix::ZERO;
    cloth.config.substeps = 1;
    cloth.config.thickness = Fix128::from_ratio(1, 64);
    let start: Vec<Vec3Fix> = cloth.positions.clone();
    assert!(start.iter().all(|p| p.y == Fix128::from_f32(-0.5)));
    cloth.step_with_sdf(dt, &[plane_at(0.0)]);
    for (i, (p, s)) in cloth.positions.iter().zip(&start).enumerate() {
        assert_eq!(
            p.y,
            Fix128::from_ratio(1, 64),
            "cloth particle {i} rests thickness above the plane"
        );
        assert_eq!(
            (p.x, p.z),
            (s.x, s.z),
            "cloth particle {i} moves only along the normal"
        );
    }
    println!(
        "[cloth] {} particles from y = -0.5 to y = {} (thickness 1/64)",
        cloth.particle_count(),
        cloth.positions[0].y.to_f32()
    );

    // ---- deformable: a cube entirely below the ground ----------------------
    let mut cube = DeformableBody::new_cube(
        Vec3Fix::from_f32(0.0, -1.0, 0.0),
        Fix128::from_ratio(1, 2),
        Fix128::ONE,
    );
    cube.config.gravity = Vec3Fix::ZERO;
    cube.config.substeps = 1;
    let start: Vec<Vec3Fix> = cube.positions.clone();
    assert!(
        start.iter().all(|p| p.y < Fix128::ZERO),
        "every vertex starts below the plane"
    );
    cube.step_with_sdf(dt, &[plane_at(0.0)]);
    for (i, (p, s)) in cube.positions.iter().zip(&start).enumerate() {
        assert_eq!(
            p.y,
            Fix128::ZERO,
            "cube vertex {i} is pushed onto the plane"
        );
        assert_eq!(
            (p.x, p.z),
            (s.x, s.z),
            "cube vertex {i} moves only along the normal"
        );
    }
    println!(
        "[deformable] {} vertices from y in [-1.5, -0.5] to y = 0",
        cube.particle_count()
    );

    // ---- fluid: one particle above the lid of its container ----------------
    let config = FluidConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..FluidConfig::default()
    };
    let mut fluid = Fluid::new(vec![Vec3Fix::from_f32(0.25, 1.5, -0.25)], config);
    fluid.step_with_sdf(dt, &[below_lid(1.0)]);
    let p = fluid.positions[0];
    assert_eq!(
        p.y,
        Fix128::ONE,
        "the escaped particle comes back to the lid"
    );
    assert_eq!(
        (p.x, p.z),
        (Fix128::from_f32(0.25), Fix128::from_f32(-0.25)),
        "only the normal component moves"
    );
    println!("[fluid] particle from y = 1.5 back to y = {}", p.y.to_f32());

    println!("all soft-body SDF checks passed");
}
