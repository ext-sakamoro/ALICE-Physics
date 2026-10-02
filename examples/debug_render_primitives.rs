//! Debug-render primitive wiring: colors, shapes, and a full-world draw pass
//!
//! Exercises every previously-unwired item in `debug_render`:
//! - the 9 `DebugColor` constants, each color-coding a different primitive
//!   kind (bodies / velocity / contacts / joints / colliders / bounds /
//!   highlights)
//! - `arrow`, `axes`, `point`, `sphere` called directly to build a known
//!   primitive list
//! - `primitive_count` checked against the exact count each call adds
//! - `debug_draw_world`, the production entry point, run over a small
//!   `PhysicsWorld` scene (one static body, one dynamic body, a distance
//!   constraint, and a contact constraint)
//!
//! ```bash
//! cargo run --example debug_render_primitives --features std
//! ```

use alice_physics::collider::Contact;
use alice_physics::debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{
    ContactConstraint, DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig,
};

fn main() {
    // --- Part 1: direct primitive construction -------------------------
    //
    // `point`/`arrow`/`sphere`/`axes` are plain data recorders: each call's
    // contribution to `primitive_count` is a closed form read from
    // `src/debug_render.rs` (not guessed):
    //   - `point`            -> +1 point
    //   - `arrow` (len != 0) -> +1 line, +1 point (arrow-tip marker)
    //   - `sphere`           -> +48 lines (3 rings x 16 segments), +0 points
    //   - `axes`             -> 3 `arrow` calls -> +3 lines, +3 points
    let mut manual = DebugDrawData::new();
    assert_eq!(manual.primitive_count(), 0, "empty draw data starts at 0");

    // Highlight marker: a single point, color-coded ORANGE.
    manual.point(Vec3Fix::ZERO, DebugColor::ORANGE, Fix128::from_ratio(1, 10));
    assert_eq!(manual.primitive_count(), 1);

    // Collider wireframe: a sphere, color-coded MAGENTA.
    manual.sphere(Vec3Fix::UNIT_X, Fix128::ONE, DebugColor::MAGENTA);
    assert_eq!(manual.primitive_count(), 1 + 48);

    // Local frame: 3 arrows (X/Y/Z), hardcoded RED/GREEN/BLUE by `axes` itself.
    manual.axes(Vec3Fix::ZERO, QuatFix::IDENTITY, Fix128::ONE);
    assert_eq!(manual.primitive_count(), 1 + 48 + 3 + 3);

    // A standalone arrow, color-coded WHITE, confirms the +1 line / +1 point
    // split independently of `axes`.
    manual.arrow(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, DebugColor::WHITE);
    let after_direct = manual.primitive_count();
    assert_eq!(after_direct, 1 + 48 + 3 + 3 + 1 + 1);
    println!(
        "[debug_render] direct construction: {after_direct} primitives (point=1 + sphere=48 + axes=6 + arrow=2)"
    );

    // --- Part 2: debug_draw_world over a physics world ------------------
    //
    // One static body (GRAY) and one dynamic body (GREEN), connected by a
    // distance constraint (joint line, CYAN) and a contact constraint
    // (point_a RED / point_b BLUE, + normal arrow RED). Velocity arrows
    // (YELLOW) are disabled so the count stays a closed form.
    let mut world = PhysicsWorld::new(SolverConfig::default());
    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let bob = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO),
        Fix128::ONE,
    ));

    world.add_distance_constraint(DistanceConstraint::new(
        anchor,
        bob,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::from_int(2),
    ));

    world.add_contact(ContactConstraint::new(
        anchor,
        bob,
        Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 10), Fix128::ZERO),
        },
    ));

    let flags = DebugDrawFlags {
        draw_aabbs: false,
        draw_centers: true,
        draw_velocities: false,
        draw_contacts: true,
        draw_contact_normals: true,
        draw_joints: true,
        draw_bvh: false,
        draw_axes: true,
    };

    let mut frame = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut frame);

    // Closed form (2 bodies, draw_centers + draw_axes; 1 contact with
    // normals; 1 joint):
    //   centers:        2 bodies x 1 point                   = 2 points
    //   axes:           2 bodies x 3 arrows (line + point)    = 6 lines, 6 points
    //   contact points: point_a + point_b                     = 2 points
    //   contact normal: 1 arrow (depth != 0 -> line + point)  = 1 line, 1 point
    //   joint:          1 distance constraint -> 1 line        = 1 line
    let expected_lines = 6 + 1 + 1;
    let expected_points = 2 + 6 + 2 + 1;
    assert_eq!(frame.lines.len(), expected_lines, "line count mismatch");
    assert_eq!(frame.points.len(), expected_points, "point count mismatch");
    assert_eq!(frame.primitive_count(), expected_lines + expected_points);
    println!(
        "[debug_render] debug_draw_world: {} primitives ({} lines, {} points) over {} bodies",
        frame.primitive_count(),
        frame.lines.len(),
        frame.points.len(),
        world.bodies.len()
    );

    // debug_draw_world clears the buffer unconditionally before drawing, so
    // calling it again on an unchanged scene reproduces the same count
    // rather than accumulating.
    debug_draw_world(&world, &flags, &mut frame);
    assert_eq!(frame.primitive_count(), expected_lines + expected_points);
    println!(
        "[debug_render] repeat draw stays at {} (clear() verified)",
        frame.primitive_count()
    );
}
