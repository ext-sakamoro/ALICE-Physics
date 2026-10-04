//! SDF Character Terrain Example
//!
//! Production entry point for `src/sdf_character.rs`: a vertical-capsule
//! character driven over signed distance fields with `apply_gravity` +
//! `step` (free fall, landing, sliding along a wall) and the lower-level
//! `move_and_slide` / `MoveOutcome::resolved_position`.
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - free fall is semi-implicit Euler from rest, `y_n = y_0 + g dt^2 n(n+1)/2`
//! - landing on the plane `y = 0` stops at `y = radius + skin_width` (the push
//!   rounds up by `skin_width`)
//!   with the downward velocity removed exactly (the correction is pure `+y`)
//! - sliding along the wall `x = 0` keeps the tangential control step intact
//! - on a field whose normal points inward the push goes deeper, so the safe
//!   read is the best sample, not the last one
//!
//! Run with: `cargo run --example sdf_character_terrain`

use alice_physics::sdf_character::{MoveOutcome, SdfCharacter};
use alice_physics::sdf_collider::SdfField;

/// The ground plane `y = 0`: exact distance field, normal `+y`.
struct Ground;

impl SdfField for Ground {
    fn distance(&self, _x: f32, y: f32, _z: f32) -> f32 {
        y
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// The wall `x = 0`: exact distance field, normal `+x`.
struct Wall;

impl SdfField for Wall {
    fn distance(&self, x: f32, _y: f32, _z: f32) -> f32 {
        x
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (1.0, 0.0, 0.0)
    }
}

/// The plane `y = 0` with its normal reversed: an inconsistent field, so every
/// push moves the character further in.
struct InvertedGround;

impl SdfField for InvertedGround {
    fn distance(&self, _x: f32, y: f32, _z: f32) -> f32 {
        y
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, -1.0, 0.0)
    }
}

fn close(got: f32, want: f32, tol: f32, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, closed form {want} (tolerance {tol})"
    );
}

fn main() {
    let g = -9.81_f32;
    let dt = 1.0_f32 / 60.0;

    // ---- free fall, then landing on the ground plane --------------------
    let y0 = 2.0_f32;
    let mut c = SdfCharacter::new([0.0, y0, 0.0], 0.35, 1.8);
    assert!(!c.is_grounded(&Ground), "starts in the air");
    let mut landed_at = None;
    for n in 1..=120 {
        c.apply_gravity([0.0, g, 0.0], dt);
        let out = c.step(&Ground, dt, [0.0, 0.0, 0.0]);
        let free_y = y0 + g * dt * dt * (n * (n + 1)) as f32 / 2.0;
        if free_y > c.radius {
            // no contact yet: semi-implicit Euler from rest
            close(c.position[1], free_y, 1e-4, "free-fall height");
            close(c.velocity[1], g * dt * n as f32, 1e-4, "free-fall velocity");
            assert!(
                out.converged && out.iterations == 0,
                "no penetration in the air"
            );
        } else if landed_at.is_none() {
            landed_at = Some(n);
        }
    }
    let n_land = landed_at.expect("the character reaches the ground within 2 s");
    let (r, skin) = (c.radius, c.skin_width);
    // the push rounds up to `radius - sample + skin_width`, so the rest height is
    // `radius + skin_width` (a few f32 ulps of the 0.35 scale either way)
    close(
        c.position[1],
        r + skin,
        skin / 10.0,
        "rest height on the plane",
    );
    assert_eq!(
        c.velocity[1], 0.0,
        "the downward velocity is removed exactly"
    );
    assert!(c.is_grounded(&Ground), "grounded after landing");
    println!(
        "landing: frame {n_land}, rest height {:.6} (radius {r}), vy {}",
        c.position[1], c.velocity[1]
    );

    // ---- sliding along a wall ------------------------------------------
    let mut w = SdfCharacter::new([1.0, 0.0, 0.0], 0.35, 1.8);
    let control = [-0.3_f32, 0.0, 0.2];
    for n in 1..=10 {
        w.step(&Wall, dt, control);
        assert!(
            w.position[0] >= w.radius,
            "never inside the wall: x = {}",
            w.position[0]
        );
        close(
            w.position[2],
            0.2 * n as f32,
            1e-5,
            "tangential travel along the wall",
        );
    }
    close(
        w.position[0],
        w.radius + w.skin_width,
        w.skin_width / 10.0,
        "pressed against the wall",
    );
    println!(
        "wall slide: x {:.6} (radius {}), z {:.3} after 10 frames",
        w.position[0], w.radius, w.position[2]
    );

    // ---- move_and_slide on a field that pushes the wrong way ------------
    let mut deep = SdfCharacter::new([0.0, 0.1, 0.0], 0.35, 1.8);
    deep.max_iterations = 3;
    let out: MoveOutcome = deep.move_and_slide(&InvertedGround, [0.0, 0.0, 0.0]);
    assert!(
        !out.converged,
        "an inward normal never resolves the penetration"
    );
    assert!(
        out.position[1] < 0.1,
        "the last position is deeper than the start: {}",
        out.position[1]
    );
    assert_eq!(
        out.best_position,
        [0.0, 0.1, 0.0],
        "the best sample is the starting point"
    );
    assert_eq!(
        out.resolved_position(),
        out.best_position,
        "the safe read is the best sample"
    );
    close(
        out.best_distance,
        0.1,
        0.0,
        "best distance is the start's distance",
    );
    println!(
        "inverted field: last y {:.4}, resolved y {:.4}",
        out.position[1],
        out.resolved_position()[1]
    );

    // on the consistent field the same call converges and both reads agree
    let ok = deep.move_and_slide(&Ground, [0.0, 0.0, 0.0]);
    assert!(ok.converged);
    assert_eq!(ok.resolved_position(), ok.position);
    println!("all SDF character checks passed");
}
