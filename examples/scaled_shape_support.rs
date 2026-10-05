//! A shape scaled about the origin, as GJK sees it
//!
//! Reaches `collider::ScaledShape` and `ScaledShape::new`.
//!
//! `ScaledShape` scales any `Support` shape uniformly about the world
//! origin. A ball centred at `c` with radius `r` scaled by `s > 0` is the
//! ball centred at `s·c` with radius `s·r`, so its support in a unit
//! direction `d` is `s·(c + r·d)`, and GJK finds it touching a second ball
//! exactly when the centre distance is at most the sum of the radii. A
//! negative scale mirrors the shape through the origin: the box
//! `[1,2]×[1,3]×[1,4]` scaled by `-2` spans `[-4,-2]×[-6,-2]×[-8,-2]`.
//!
//! Every value below is a dyadic rational, so the support points are exact.
//!
//! Run with: `cargo run --example scaled_shape_support`

use alice_physics::collider::{gjk, ScaledShape, Sphere, Support, AABB};
use alice_physics::math::{Fix128, Vec3Fix};

fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn main() {
    // Unit ball at (1, 0, 0) scaled by 2: the ball at (2, 0, 0), radius 2.
    let scaled = ScaledShape::new(Sphere::new(v3(1, 0, 0), Fix128::ONE), Fix128::from_int(2));
    assert_eq!(scaled.support(v3(1, 0, 0)), v3(4, 0, 0));
    assert_eq!(scaled.support(v3(-1, 0, 0)), v3(0, 0, 0));
    assert_eq!(scaled.support(v3(0, 1, 0)), v3(2, 2, 0));
    println!(
        "[scaled_shape] ball (1,0,0) r 1 x 2 -> support +x {:?}",
        scaled.support(v3(1, 0, 0)).to_f32()
    );

    // Against a unit ball on the +x axis: they touch while its centre is at
    // most 2 + 2 + 1 = 5 from the origin (sum of radii 3 from (2, 0, 0)).
    let probe = |x: i64| Sphere::new(v3(x, 0, 0), Fix128::ONE);
    assert!(gjk(&scaled, &probe(4)).colliding, "centre distance 2 < 3");
    assert!(!gjk(&scaled, &probe(6)).colliding, "centre distance 4 > 3");
    // The unscaled ball reaches only to x = 2: the probe at 4 is clear of it.
    assert!(!gjk(&Sphere::new(v3(1, 0, 0), Fix128::ONE), &probe(4)).colliding);
    println!("[scaled_shape] gjk: probe at x = 4 hits the scaled ball, at x = 6 misses");

    // Negative scale: the mirrored, doubled box.
    let mirrored = ScaledShape::new(AABB::new(v3(1, 1, 1), v3(2, 3, 4)), Fix128::from_int(-2));
    assert_eq!(mirrored.support(v3(1, 1, 1)), v3(-2, -2, -2));
    assert_eq!(mirrored.support(v3(-1, -1, -1)), v3(-4, -6, -8));
    assert!(gjk(&mirrored, &Sphere::new(v3(-3, -4, -5), Fix128::ONE)).colliding);
    assert!(!gjk(&mirrored, &Sphere::new(v3(3, 4, 5), Fix128::ONE)).colliding);
    println!("[scaled_shape] box x -2 spans [-4,-2]x[-6,-2]x[-8,-2]");
}
