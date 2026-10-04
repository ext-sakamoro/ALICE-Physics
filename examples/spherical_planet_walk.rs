//! A walker on a small planet: radial height field ground, a water level
//! that can stand above it, constant gravity toward the centre and
//! great-circle locomotion.
//!
//! Checks each result against its closed form: the walker lands at
//! `R + h + radius + skin`, a quarter of the circumference takes
//! `(π/2)·r / speed` seconds, and over the sunken half it stands on the
//! water rather than on the ground below it.
//!
//! ```bash
//! cargo run --release --example spherical_planet_walk --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::{SdfField, SdfUnion};
use alice_physics::spherical_terrain::{central_gravity, SphericalHeightField, SurfaceHeight};

const R: f32 = 100.0;
const WATER: f32 = -1.0;

/// Ground 0.5 m up on the northern half, sunk 4 m below the reference
/// sphere on the southern half, joined by a smooth band.
struct Hemispheres;

impl SurfaceHeight for Hemispheres {
    fn height(&self, dir: [f32; 3]) -> f32 {
        let t = ((dir[1] + 0.2) / 0.4).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        -4.0 + 4.5 * s
    }
}

fn main() {
    let centre = [0.0_f32; 3];
    // |dh/dθ| <= 4.5 * 1.5 / 0.4 ≈ 16.9 m/rad, i.e. 0.17 m per metre of arc
    let ground = SphericalHeightField::new(centre, R, Hemispheres)
        .with_max_slope(0.17)
        .with_normal_step(1.0e-3);
    let water = SphericalHeightField::new(centre, R, |_dir: [f32; 3]| WATER);
    println!(
        "planet: centre {:?}, R = {} m, ground at the north pole {} m",
        ground.center(),
        ground.radius(),
        ground.surface_radius([0.0, 1.0, 0.0])
    );
    let world = SdfUnion::new(ground, water);

    let g = 9.8;
    let dt = 1.0 / 120.0;
    let mut walker = SdfCharacter::new([0.0, R + 5.0, 0.0], 0.3, 1.7);
    let a = central_gravity(centre, g, walker.position);
    println!("gravity at the drop point: {a:?}");

    // 1. drop onto the north pole
    for _ in 0..240 {
        walker.apply_central_gravity(centre, g, dt);
        walker.step_on_sphere(&world, centre, dt, [0.0; 3]);
    }
    let r = (walker.position[1].powi(2) + walker.position[0].powi(2) + walker.position[2].powi(2))
        .sqrt();
    // the slope bound divides the distance by k = sqrt(1 + s²), so the
    // clearance a push establishes is k times larger along the radial line
    let k = (1.0_f32 + 0.17 * 0.17).sqrt();
    let want = R + 0.5 + (walker.radius + walker.skin_width) * k;
    println!("landed at r = {r:.4} (closed form {want:.4})");
    assert!((r - want).abs() < 2e-3);
    assert!(walker.is_grounded(&world));

    // 2. walk south along the +x meridian at 5 m/s for a quarter turn
    let speed = 5.0;
    let steps = ((std::f32::consts::FRAC_PI_2 * want / speed) / dt).round() as usize;
    for _ in 0..steps {
        let up = walker.up;
        // heading = (y × x) × up, i.e. toward +x on the north side
        let heading = [up[1], -up[0], 0.0];
        walker.apply_central_gravity(centre, g, dt);
        walker.step_on_sphere(
            &world,
            centre,
            dt,
            [heading[0] * speed, heading[1] * speed, heading[2] * speed],
        );
    }
    let p = walker.position;
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let lat = p[1] / r;
    println!("after {steps} steps: lat {lat:.3}, r = {r:.4}");

    // 3. keep walking into the sunken half: the water is the higher surface
    for _ in 0..steps {
        let up = walker.up;
        let heading = [up[1], -up[0], 0.0];
        walker.apply_central_gravity(centre, g, dt);
        walker.step_on_sphere(
            &world,
            centre,
            dt,
            [heading[0] * speed, heading[1] * speed, heading[2] * speed],
        );
    }
    let p = walker.position;
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let on_water = R + WATER + walker.radius + walker.skin_width;
    println!(
        "south of the band: lat {:.3}, r = {r:.4} (water {on_water:.4}, sunken ground {:.4})",
        p[1] / r,
        R - 4.0 + walker.radius
    );
    assert!(p[1] / r < -0.2);
    assert!((r - on_water).abs() < 2e-3);
    assert!(world.distance(p[0], p[1], p[2]) > 0.0);

    // the unscaled radial value of the sunken ground under the walker
    let ground = SphericalHeightField::new(centre, R, Hemispheres);
    let below = ground.radial_height(p);
    println!("height above the sunken ground: {below:.4} m");
    assert!((below - (r - (R - 4.0))).abs() < 1e-3);
}
