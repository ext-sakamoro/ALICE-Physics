//! Closed-form oracle for the four-bar linkage rocker angle across a full
//! crank revolution (COV-MBD-067). The existing
//! `tests/engineering_oracles_misc.rs::kinematic_loop_four_bar_preserves_link_lengths_under_solver`
//! only checks that link lengths are preserved while the mechanism settles
//! under no driving at all (see docs/coverage/mbd.toml COV-MBD-067 evidence
//! before this test landed); nothing drives the crank through a revolution
//! and compares the rocker angle to a closed form.
//!
//! The expected rocker angle for a given crank angle is computed here from
//! plane circle-circle intersection (independent of, and mathematically
//! equivalent to, the Freudenstein trigonometric equation Norton ch.4
//! derives for the same linkage): with the crank pin A placed at crank
//! angle theta2, the coupler pin B lies on both the circle of radius
//! r_coupler about A and the circle of radius r_rocker about the fixed
//! ground pin O4; picking the same "open configuration" branch the crate's
//! own `try_four_bar_linkage` uses at theta2 = 0 gives the Freudenstein
//! rocker angle theta4 = atan2(B - O4).

use alice_physics::det_math;
use alice_physics::kinematic_loop::four_bar_linkage;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn ffx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Intersection of the circle of radius `r_a` about `center_a` and the
/// circle of radius `r_b` about `center_b`, picking the branch on the
/// positive-rotation (counter-clockwise, +90 degree) side of the
/// `center_a -> center_b` direction -- the same "open configuration" side
/// `try_four_bar_linkage` picks at crank angle 0 (there, A is left of O4
/// along +x and B is placed at positive y).
///
/// Uses `alice_physics::det_math` rather than `f32::{sin,cos,atan2}` (not
/// `f64`: det_math only has deterministic transcendentals for `f32`, ample
/// precision for this oracle's 1e-2 angular tolerance).
fn circle_intersection_open_branch(
    center_a: (f32, f32),
    r_a: f32,
    center_b: (f32, f32),
    r_b: f32,
) -> (f32, f32) {
    let dx = center_b.0 - center_a.0;
    let dy = center_b.1 - center_a.1;
    let d = (dx * dx + dy * dy).sqrt();
    let a = (d * d + r_a * r_a - r_b * r_b) / (2.0 * d);
    let h_sq = (r_a * r_a - a * a).max(0.0);
    let h = h_sq.sqrt();
    // Unit vector along A->B axis, and its +90 degree rotation.
    let ux = dx / d;
    let uy = dy / d;
    let px = center_a.0 + a * ux;
    let py = center_a.1 + a * uy;
    // +90 degree rotation of (ux, uy) is (-uy, ux); this is the branch
    // matching try_four_bar_linkage's "positive y at crank angle 0" choice
    // when center_a = A = (r2, 0) and center_b = O4 = (r1, 0) (dx > 0, so
    // the rotation (-uy, ux) = (0, 1) points to +y, same as the source).
    (px - h * uy, py + h * ux)
}

/// Expected rocker angle (radians, O4 -> B) at crank angle `theta2`.
fn expected_rocker_angle(
    ground_length: f32,
    crank_length: f32,
    coupler_length: f32,
    rocker_length: f32,
    theta2: f32,
) -> f32 {
    let a = (
        crank_length * det_math::cos(theta2),
        crank_length * det_math::sin(theta2),
    );
    let o4 = (ground_length, 0.0);
    let b = circle_intersection_open_branch(a, coupler_length, o4, rocker_length);
    det_math::atan2(b.1 - o4.1, b.0 - o4.0)
}

/// Shortest signed angular difference `a - b`, wrapped to (-pi, pi].
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI {
        d -= 2.0 * std::f32::consts::PI;
    }
    while d <= -std::f32::consts::PI {
        d += 2.0 * std::f32::consts::PI;
    }
    d
}

#[test]
fn rocker_angle_matches_freudenstein_relation_across_a_crank_revolution() {
    // Grashof: shortest + longest < sum of the other two, strictly, or the
    // mechanism has a collinear "change point" configuration somewhere in
    // the crank revolution (1, 2, 1.5, 2.5 hits this exactly: 1+2.5 ==
    // 2+1.5, found as a single-angle failure near a toggle position before
    // this margin was added).
    let (l_crank, l_coupler, l_rocker, l_ground) = (1.0_f32, 2.0_f32, 1.5_f32, 2.3_f32);

    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let linkage = four_bar_linkage(
        &mut world,
        Vec3Fix::ZERO,
        ffx(l_crank as f64),
        ffx(l_coupler as f64),
        ffx(l_rocker as f64),
        ffx(l_ground as f64),
        Fix128::ONE,
    );

    // Sanity: theta2 = 0 should already match (this is the configuration
    // try_four_bar_linkage itself builds), confirming the branch choice
    // above agrees with the crate's own construction before driving anything.
    let o4 = (l_ground, 0.0_f32);
    let b0 = world.bodies[linkage.coupler].position;
    let theta4_at_build = det_math::atan2(b0.y.to_f64() as f32 - o4.1, b0.x.to_f64() as f32 - o4.0);
    let theta4_expected_at_0 = expected_rocker_angle(l_ground, l_crank, l_coupler, l_rocker, 0.0);
    assert!(
        angle_diff(theta4_at_build, theta4_expected_at_0).abs() < 1e-5,
        "branch choice sanity check at theta2=0: built {theta4_at_build:.6} expected {theta4_expected_at_0:.6}"
    );

    let dt = Fix128::from_ratio(1, 60);
    let steps_per_angle = 120; // let the coupler settle at each driven crank angle
    let n_angles = 24;
    for i in 0..=n_angles {
        let theta2 = 2.0 * std::f32::consts::PI * (i as f32) / (n_angles as f32);
        let crank_pos = Vec3Fix::new(
            ffx((l_crank * det_math::cos(theta2)) as f64),
            ffx((l_crank * det_math::sin(theta2)) as f64),
            Fix128::ZERO,
        );
        world.bodies[linkage.crank].position = crank_pos;
        world.bodies[linkage.crank].velocity = Vec3Fix::ZERO;
        for _ in 0..steps_per_angle {
            world.bodies[linkage.crank].position = crank_pos;
            world.bodies[linkage.crank].velocity = Vec3Fix::ZERO;
            world.step(dt);
            linkage.closure.apply(&mut world);
        }

        let b = world.bodies[linkage.coupler].position;
        let theta4_sim = det_math::atan2(b.y.to_f64() as f32 - o4.1, b.x.to_f64() as f32 - o4.0);
        let theta4_expected = expected_rocker_angle(l_ground, l_crank, l_coupler, l_rocker, theta2);

        let err = angle_diff(theta4_sim, theta4_expected).abs();
        assert!(
            err < 1e-2,
            "crank angle {:.3} rad: rocker angle got {theta4_sim:.4} expected {theta4_expected:.4} (err {err:.4})",
            theta2
        );
    }
}
