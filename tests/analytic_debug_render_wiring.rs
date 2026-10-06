//! Analytic-oracle tests for `debug_render`'s wiring pass.
//!
//! `debug_render` had 15 items (`BLUE`/`CYAN`/`GRAY`/`GREEN`/`MAGENTA`/
//! `ORANGE`/`RED`/`WHITE`/`YELLOW`/`arrow`/`axes`/`debug_draw_world`/`point`/
//! `primitive_count`/`sphere`) that production never called outside the
//! module's own `#[cfg(test)]` block. `examples/debug_render_primitives.rs`
//! is the production entry point; this file pins the closed forms
//! independently of that example and of the implementation under test.
//! Expected values are transcribed by hand from `src/debug_render.rs`
//! (line numbers noted per test), not produced by calling the functions
//! being checked.

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::debug_render::{
    debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags, DebugLine, DebugPoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{
    ContactConstraint, DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn scene_with_one_contact_and_one_joint() -> PhysicsWorld {
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
    world
}

fn all_flags_false() -> DebugDrawFlags {
    DebugDrawFlags {
        draw_aabbs: false,
        draw_centers: false,
        draw_velocities: false,
        draw_contacts: false,
        draw_contact_normals: false,
        draw_joints: false,
        draw_bvh: false,
        draw_axes: false,
    }
}

// --- Color constants: literal RGBA pin (src/debug_render.rs:39-55) --------

#[test]
fn debug_color_constants_pin_exact_rgba_bytes() {
    let cases: [(DebugColor, (u8, u8, u8, u8)); 9] = [
        (DebugColor::RED, (255, 50, 50, 255)),
        (DebugColor::GREEN, (50, 255, 50, 255)),
        (DebugColor::BLUE, (50, 50, 255, 255)),
        (DebugColor::YELLOW, (255, 255, 50, 255)),
        (DebugColor::CYAN, (50, 255, 255, 255)),
        (DebugColor::MAGENTA, (255, 50, 255, 255)),
        (DebugColor::WHITE, (255, 255, 255, 255)),
        (DebugColor::GRAY, (128, 128, 128, 255)),
        (DebugColor::ORANGE, (255, 165, 0, 255)),
    ];
    for (color, (r, g, b, a)) in cases {
        assert_eq!(
            (color.r, color.g, color.b, color.a),
            (r, g, b, a),
            "{color:?}"
        );
    }
    // Distinctness: a transposed-literal mutation (e.g. swapping two
    // constants' bytes) would still pass a per-constant check that compares
    // a constant against itself, so also pin that all 9 are pairwise
    // different colors.
    let all = [
        DebugColor::RED,
        DebugColor::GREEN,
        DebugColor::BLUE,
        DebugColor::YELLOW,
        DebugColor::CYAN,
        DebugColor::MAGENTA,
        DebugColor::WHITE,
        DebugColor::GRAY,
        DebugColor::ORANGE,
    ];
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(all[i], all[j], "constants {i} and {j} collide");
        }
    }
}

// --- point / arrow / sphere / axes: exact data-recorder round trip -------

#[test]
fn point_stores_position_color_size_verbatim() {
    let mut data = DebugDrawData::new();
    let pos = Vec3Fix::new(
        Fix128::from_ratio(7, 3),
        Fix128::from_ratio(-11, 4),
        Fix128::from_int(42),
    );
    let size = Fix128::from_ratio(9, 5);
    data.point(pos, DebugColor::CYAN, size);

    assert_eq!(data.points.len(), 1);
    assert_eq!(data.points[0], DebugPoint::new(pos, DebugColor::CYAN, size));
    assert_eq!(data.points[0].position, pos);
    assert_eq!(data.points[0].color, DebugColor::CYAN);
    assert_eq!(data.points[0].size, size);
}

#[test]
fn primitive_count_is_zero_before_any_add() {
    assert_eq!(DebugDrawData::new().primitive_count(), 0);
    assert_eq!(DebugDrawData::default().primitive_count(), 0);
}

/// `arrow` (src/debug_render.rs:273-286): records the line verbatim, then
/// for a nonzero direction adds a head-point marker at `end` whose `size` is
/// `0.2 * |end - start|`. Oracle direction here is a 3-4-5 triangle so the
/// length is exact in Fix128 (no sqrt rounding to reason about).
#[test]
fn arrow_records_line_verbatim_and_head_point_size_is_fifth_of_length() {
    let mut data = DebugDrawData::new();
    let start = Vec3Fix::from_int(1, 2, 3);
    let end = start + Vec3Fix::from_int(3, 4, 0); // |delta| = 5 exactly
    data.arrow(start, end, DebugColor::ORANGE);

    // shaft + two barbs (AUD-A-S4W3-008: the head used to be computed and dropped)
    assert_eq!(data.lines.len(), 3);
    assert_eq!(
        data.lines[0],
        DebugLine::new(start, end, DebugColor::ORANGE)
    );
    // barbs start at the tip and end head_len back along the shaft, offset
    // head_len / 2 to either side: the shaft is (3,4,0)/5, head_len = 1, so
    // head_point = end - (0.6, 0.8, 0) and the side is (3,4,0)/5 × Y = (0, 0, 0.6)/|..| = ±Z
    let head_point = end
        - Vec3Fix::new(
            Fix128::from_ratio(3, 5),
            Fix128::from_ratio(4, 5),
            Fix128::ZERO,
        );
    let half = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_ratio(1, 2));
    let ends = [data.lines[1].end, data.lines[2].end];
    for b in &data.lines[1..] {
        assert_eq!(b.start, end);
        assert_eq!(b.color, DebugColor::ORANGE);
    }
    let close = |a: Vec3Fix, b: Vec3Fix| (a - b).length().to_f64() < 1e-9;
    assert!(
        (close(ends[0], head_point + half) && close(ends[1], head_point - half))
            || (close(ends[0], head_point - half) && close(ends[1], head_point + half)),
        "barb ends {ends:?}"
    );

    assert_eq!(data.points.len(), 1);
    assert_eq!(data.points[0].position, end);
    assert_eq!(data.points[0].color, DebugColor::ORANGE);
    let expected_head_len = Fix128::from_int(5) * Fix128::from_ratio(2, 10);
    assert_eq!(data.points[0].size, expected_head_len);
}

#[test]
fn zero_length_arrow_draws_the_line_but_skips_the_head_point() {
    let mut data = DebugDrawData::new();
    data.arrow(Vec3Fix::ZERO, Vec3Fix::ZERO, DebugColor::GRAY);
    // src/debug_render.rs:278-280: `if len.is_zero() { return; }` — the line
    // is always recorded first, the head-point marker is skipped.
    assert_eq!(data.lines.len(), 1);
    assert_eq!(
        data.lines[0],
        DebugLine::new(Vec3Fix::ZERO, Vec3Fix::ZERO, DebugColor::GRAY)
    );
    assert_eq!(data.points.len(), 0);
}

/// `sphere` (src/debug_render.rs:223-249): 3 rings (XY, XZ, YZ) of 16
/// segments each = 48 lines, 0 points, all in the given color. The oracle
/// segment count (16) is read from the source, not derived from a formula —
/// it is a hardcoded constant in the implementation.
#[test]
fn sphere_emits_48_lines_in_3_rings_and_no_points() {
    let mut data = DebugDrawData::new();
    data.sphere(Vec3Fix::ZERO, Fix128::ONE, DebugColor::MAGENTA);
    assert_eq!(data.lines.len(), 48);
    assert_eq!(data.points.len(), 0);
    assert!(data.lines.iter().all(|l| l.color == DebugColor::MAGENTA));
}

#[test]
fn zero_radius_sphere_does_not_panic_and_still_emits_48_degenerate_lines() {
    let mut data = DebugDrawData::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        data.sphere(Vec3Fix::ZERO, Fix128::ZERO, DebugColor::WHITE);
    }));
    assert!(result.is_ok(), "zero-radius sphere panicked: {result:?}");
    // Segment count (16/ring) is independent of radius, so a degenerate
    // sphere still emits 48 lines, each collapsed to a single point.
    assert_eq!(data.lines.len(), 48);
    assert!(data
        .lines
        .iter()
        .all(|l| l.start == Vec3Fix::ZERO && l.end == Vec3Fix::ZERO));
}

/// `axes` (src/debug_render.rs:289-297) takes no color argument: X/Y/Z are
/// hardcoded RED/GREEN/BLUE, always in that order, regardless of rotation.
#[test]
fn axes_emits_three_arrows_hardcoded_rgb_in_xyz_order() {
    let mut data = DebugDrawData::new();
    data.axes(Vec3Fix::ZERO, QuatFix::IDENTITY, Fix128::ONE);

    // each arrow is a shaft followed by its two barbs
    assert_eq!(data.lines.len(), 9);
    assert_eq!(data.points.len(), 3);
    assert_eq!(data.lines[0].color, DebugColor::RED);
    assert_eq!(data.lines[3].color, DebugColor::GREEN);
    assert_eq!(data.lines[6].color, DebugColor::BLUE);
    assert_eq!(data.lines[0].end, Vec3Fix::UNIT_X);
    assert_eq!(data.lines[3].end, Vec3Fix::UNIT_Y);
    assert_eq!(data.lines[6].end, Vec3Fix::UNIT_Z);
}

// --- debug_draw_world: production entry point ------------------------------

/// Closed form for `scene_with_one_contact_and_one_joint()` with every flag
/// enabled except `draw_velocities` (hand count, also checked against the
/// implementation in `examples/debug_render_primitives.rs`):
///   centers:         2 bodies x 1 point                  = 2 points
///   axes:            2 bodies x 3 arrows (3 lines + point) = 18 lines, 6 points
///   contact points:  point_a + point_b                    = 2 points
///   contact normal:  1 arrow, depth != 0 -> 3 lines + point = 3 lines, 1 point
///   joint:           1 distance constraint -> 1 line      = 1 line
///   total:           22 lines, 11 points, 33 primitives
#[test]
fn debug_draw_world_primitive_count_matches_closed_form() {
    let world = scene_with_one_contact_and_one_joint();
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
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);

    assert_eq!(data.lines.len(), 22);
    assert_eq!(data.points.len(), 11);
    assert_eq!(data.primitive_count(), 33);
}

/// Both-directions complement of the test above: every flag that
/// contributes a primitive in the closed-form scene is turned off, so the
/// same non-empty scene (2 bodies, 1 contact, 1 joint) must draw nothing.
/// This is the gate that a "configuration ignored" mutation (e.g. a flag
/// check removed from `debug_draw_world`) would fail.
#[test]
fn all_draw_flags_false_yields_zero_primitives_on_a_non_empty_scene() {
    let world = scene_with_one_contact_and_one_joint();
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &all_flags_false(), &mut data);
    assert_eq!(data.primitive_count(), 0);
}

/// `draw_contact_normals` is a gate nested inside `draw_contacts`
/// (src/debug_render.rs:336-345): with contacts on but normals off, only
/// the 2 contact points are drawn, never the normal arrow. `all_flags_false`
/// cannot see a mutation that drops just this inner gate (it never reaches
/// the nested block at all), so this scene isolates it.
#[test]
fn draw_contact_normals_false_omits_the_normal_arrow_but_keeps_contact_points() {
    let world = scene_with_one_contact_and_one_joint();
    let mut flags = all_flags_false();
    flags.draw_contacts = true;
    flags.draw_contact_normals = false;

    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);

    assert_eq!(data.lines.len(), 0);
    assert_eq!(data.points.len(), 2);
    assert_eq!(data.points[0].color, DebugColor::RED);
    assert_eq!(data.points[1].color, DebugColor::BLUE);
}

/// `draw_velocities` is additionally gated by `!is_static`
/// (src/debug_render.rs:326-329): only the dynamic body may contribute a
/// velocity arrow, never the static one, independent of every other flag.
#[test]
fn draw_velocities_only_adds_an_arrow_for_the_non_static_body() {
    let mut world = PhysicsWorld::new(SolverConfig::default());
    world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut bob = RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE);
    bob.velocity = Vec3Fix::from_int(3, 4, 0); // 3-4-5 triangle: nonzero exact length
    world.add_body(bob);

    let mut flags = all_flags_false();
    flags.draw_velocities = true;

    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);

    // Exactly one velocity arrow (the dynamic body): shaft + two barbs + head
    // point, since |velocity| = 5 != 0.
    assert_eq!(data.lines.len(), 3);
    assert_eq!(data.points.len(), 1);
    assert_eq!(data.lines[0].color, DebugColor::YELLOW);
    assert_eq!(data.lines[0].start, Vec3Fix::from_int(0, 2, 0));
    assert_eq!(
        data.lines[0].end,
        Vec3Fix::from_int(0, 2, 0) + Vec3Fix::from_int(3, 4, 0)
    );
}

#[test]
fn debug_draw_world_on_empty_world_clears_prior_contents_to_zero() {
    let world = PhysicsWorld::new(SolverConfig::default());
    let flags = DebugDrawFlags::default();
    let mut data = DebugDrawData::new();
    // Pre-populate to prove `debug_draw_world` clears unconditionally
    // (src/debug_render.rs:312, `data.clear();` before anything else).
    data.point(Vec3Fix::ZERO, DebugColor::RED, Fix128::ONE);
    assert_eq!(data.primitive_count(), 1);

    debug_draw_world(&world, &flags, &mut data);
    assert_eq!(data.primitive_count(), 0);
}

#[test]
fn repeated_debug_draw_world_calls_do_not_accumulate() {
    let world = scene_with_one_contact_and_one_joint();
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
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);
    let first = data.primitive_count();
    debug_draw_world(&world, &flags, &mut data);
    assert_eq!(data.primitive_count(), first);
}

// --- Degenerate / extreme inputs -------------------------------------------

/// Fix128 arithmetic is unconditionally wrapping (src/math.rs:381-480: every
/// `Add`/`Sub`/`Mul` impl uses `wrapping_*`/`overflowing_*`, no panic even in
/// debug builds), so this is a round-trip check at the representable extreme,
/// not a search for a panic that cannot happen by construction.
#[test]
fn extreme_coordinates_round_trip_exactly_through_point_without_panicking() {
    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);
    let pos = Vec3Fix::new(extreme, extreme, extreme);
    let mut data = DebugDrawData::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        data.point(pos, DebugColor::RED, extreme)
    }));
    assert!(
        result.is_ok(),
        "point() panicked on extreme input: {result:?}"
    );
    assert_eq!(data.points[0].position, pos);
    assert_eq!(data.points[0].size, extreme);
}

#[test]
fn extreme_coordinates_do_not_panic_through_arrow_and_line_endpoints_are_the_raw_inputs() {
    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);
    let pos = Vec3Fix::new(extreme, extreme, extreme);
    let mut data = DebugDrawData::new();

    // `arrow` subtracts (`end - start`) and takes a `length()` (sqrt) to
    // decide whether to add the head point — both wrapping operations.
    let result = catch_unwind(AssertUnwindSafe(|| {
        data.arrow(Vec3Fix::ZERO, pos, DebugColor::BLUE);
    }));
    assert!(
        result.is_ok(),
        "arrow() panicked on extreme input: {result:?}"
    );

    // The stored line endpoints are the raw inputs, independent of whatever
    // `length()`/`normalize()` computed internally for the head marker.
    // the squared length wraps to zero at this magnitude, so no head is drawn
    assert_eq!(data.lines.len(), 1);
    assert_eq!(data.lines[0].start, Vec3Fix::ZERO);
    assert_eq!(data.lines[0].end, pos);
}

#[test]
fn extreme_radius_and_center_do_not_panic_through_sphere() {
    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);
    let pos = Vec3Fix::new(extreme, extreme, extreme);
    let mut data = DebugDrawData::new();

    // `sphere`'s ring helper multiplies `radius` by `cos`/`sin` of 16
    // angles per ring; at the representable extreme every multiply wraps.
    let result = catch_unwind(AssertUnwindSafe(|| {
        data.sphere(pos, extreme, DebugColor::GREEN);
    }));
    assert!(
        result.is_ok(),
        "sphere() panicked on extreme input: {result:?}"
    );
    assert_eq!(data.lines.len(), 48);
}

// --- draw_aabbs: the broad-phase box of each body ---------------------------
//
// `debug_draw_world` draws, under `draw_aabbs`, the box the broad-phase tests
// for a body: centre = body position, half-extent = collision radius on every
// axis. The 12 edges come from `DebugDrawData::aabb` in a fixed order (bottom
// face, top face, verticals). Expected endpoints below are written by hand
// from that closed form.

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// The 12 edges of the box `[lo, hi]` in the order `DebugDrawData::aabb`
/// emits them, written out corner by corner.
fn box_edges(lo: Vec3Fix, hi: Vec3Fix) -> [(Vec3Fix, Vec3Fix); 12] {
    let c = [
        v(lo.x, lo.y, lo.z),
        v(hi.x, lo.y, lo.z),
        v(hi.x, hi.y, lo.z),
        v(lo.x, hi.y, lo.z),
        v(lo.x, lo.y, hi.z),
        v(hi.x, lo.y, hi.z),
        v(hi.x, hi.y, hi.z),
        v(lo.x, hi.y, hi.z),
    ];
    [
        (c[0], c[1]),
        (c[1], c[2]),
        (c[2], c[3]),
        (c[3], c[0]),
        (c[4], c[5]),
        (c[5], c[6]),
        (c[6], c[7]),
        (c[7], c[4]),
        (c[0], c[4]),
        (c[1], c[5]),
        (c[2], c[6]),
        (c[3], c[7]),
    ]
}

/// Dynamic body at (1, 2, 3) with radius 1/2, static body at the origin with
/// radius 2, and a dynamic body with no collision radius.
fn scene_with_radii() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(SolverConfig::default());
    world.add_body_with_radius(
        RigidBody::new_dynamic(
            v(Fix128::ONE, Fix128::from_int(2), Fix128::from_int(3)),
            Fix128::ONE,
        ),
        Fix128::from_ratio(1, 2),
    );
    world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::from_int(2));
    world.add_body(RigidBody::new_dynamic(
        v(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));
    world
}

#[test]
fn draw_aabbs_emits_the_broadphase_box_of_every_body_with_a_radius() {
    let world = scene_with_radii();
    let flags = DebugDrawFlags {
        draw_aabbs: true,
        ..all_flags_false()
    };
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);

    // 2 bodies with a radius x 12 edges; the third body draws nothing.
    assert_eq!(data.lines.len(), 24);
    assert!(data.points.is_empty());

    let half = Fix128::from_ratio(1, 2);
    let dynamic = box_edges(
        v(half, Fix128::from_ratio(3, 2), Fix128::from_ratio(5, 2)),
        v(
            Fix128::from_ratio(3, 2),
            Fix128::from_ratio(5, 2),
            Fix128::from_ratio(7, 2),
        ),
    );
    let two = Fix128::from_int(2);
    let stat = box_edges(v(-two, -two, -two), v(two, two, two));
    for (k, (start, end)) in dynamic.iter().enumerate() {
        assert_eq!(data.lines[k].start, *start, "dynamic edge {k} start");
        assert_eq!(data.lines[k].end, *end, "dynamic edge {k} end");
        assert_eq!(data.lines[k].color, DebugColor::GREEN);
    }
    for (k, (start, end)) in stat.iter().enumerate() {
        assert_eq!(data.lines[12 + k].start, *start, "static edge {k} start");
        assert_eq!(data.lines[12 + k].end, *end, "static edge {k} end");
        assert_eq!(data.lines[12 + k].color, DebugColor::GRAY);
    }
}

#[test]
fn draw_aabbs_off_emits_no_box() {
    let world = scene_with_radii();
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &all_flags_false(), &mut data);
    assert_eq!(data.primitive_count(), 0);

    // Default flags draw the boxes (24 lines) plus one centre point per body;
    // with `draw_aabbs` turned off only the 3 centre points remain.
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &DebugDrawFlags::default(), &mut data);
    assert_eq!(data.lines.len(), 24);
    assert_eq!(data.points.len(), 3);
    let flags = DebugDrawFlags {
        draw_aabbs: false,
        ..DebugDrawFlags::default()
    };
    let mut data = DebugDrawData::new();
    debug_draw_world(&world, &flags, &mut data);
    assert!(data.lines.is_empty());
    assert_eq!(data.points.len(), 3);
}
