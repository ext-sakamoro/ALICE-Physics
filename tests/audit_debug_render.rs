//! Audit oracles for `alice_physics::debug_render`.
//!
//! Expected geometry is written from the shapes themselves (box corners and edge
//! lengths, circle vertices at `r (cos, sin)`, rotated axes), not read back from
//! the recorder under test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::collider::{Contact, AABB};
use alice_physics::debug_render::{
    debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags, DebugLine,
};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{
    ContactConstraint, DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn close3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}
fn none() -> DebugDrawFlags {
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

// ----------------------------------------------------------------- aabb wireframe

#[test]
fn aabb_wireframe_is_the_twelve_edges_of_the_box() {
    let (lo, hi) = ([-1.0, 2.0, 0.5], [3.0, 5.0, 9.5]); // dims 4 x 3 x 9
    let mut d = DebugDrawData::new();
    d.aabb(
        &AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2])),
        DebugColor::WHITE,
    );
    assert_eq!(d.lines.len(), 12);
    assert!(d.points.is_empty());
    let dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let mut per_axis = [0usize; 3];
    let mut corner_deg: Vec<([f64; 3], usize)> = Vec::new();
    for l in &d.lines {
        assert_eq!(l.color, DebugColor::WHITE);
        let (a, b) = (arr(l.start), arr(l.end));
        // every coordinate is a box bound
        for p in [a, b] {
            for i in 0..3 {
                assert!(p[i] == lo[i] || p[i] == hi[i], "{p:?} not a corner");
            }
        }
        // axis-aligned: exactly one coordinate differs, by the box dimension
        let diff: Vec<usize> = (0..3).filter(|&i| a[i] != b[i]).collect();
        assert_eq!(diff.len(), 1, "edge {a:?}-{b:?} is not axis-aligned");
        per_axis[diff[0]] += 1;
        assert!(((b[diff[0]] - a[diff[0]]).abs() - dims[diff[0]]).abs() < 1e-12);
        for p in [a, b] {
            match corner_deg.iter_mut().find(|(q, _)| *q == p) {
                Some((_, n)) => *n += 1,
                None => corner_deg.push((p, 1)),
            }
        }
    }
    assert_eq!(per_axis, [4, 4, 4]);
    assert_eq!(corner_deg.len(), 8, "8 distinct corners");
    assert!(
        corner_deg.iter().all(|(_, n)| *n == 3),
        "each corner joins 3 edges"
    );
}

// ----------------------------------------------------------------- sphere wireframe

#[test]
fn sphere_rings_lie_on_the_sphere_in_three_orthogonal_planes() {
    let (c, r) = ([1.0, -2.0, 3.0], 2.5);
    let mut d = DebugDrawData::new();
    d.sphere(v3(c[0], c[1], c[2]), fx(r), DebugColor::GREEN);
    assert_eq!(d.lines.len(), 48);
    // the 16 expected vertices of a ring, at 22.5 degree steps
    let ring: Vec<(f64, f64)> = (0..16)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::PI / 8.0;
            (r * a.cos(), r * a.sin())
        })
        .collect();
    // classify each line by the coordinate that stays at the centre
    let mut planes = [0usize; 3];
    for l in &d.lines {
        for p in [arr(l.start), arr(l.end)] {
            let off = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            let len = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
            assert!((len - r).abs() < 1e-8, "vertex off the sphere: {len}");
        }
        let (s, e) = (arr(l.start), arr(l.end));
        let fixed: Vec<usize> = (0..3)
            .filter(|&i| (s[i] - c[i]).abs() < 1e-9 && (e[i] - c[i]).abs() < 1e-9)
            .collect();
        assert_eq!(fixed.len(), 1, "line {s:?}-{e:?} is not in one axis plane");
        planes[fixed[0]] += 1;
        // both ends are ring vertices, one 22.5 degree step apart (chord = 2 r sin(pi/16))
        let chord = 2.0 * r * (std::f64::consts::PI / 16.0).sin();
        let dd = [e[0] - s[0], e[1] - s[1], e[2] - s[2]];
        let l = (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]).sqrt();
        assert!(
            (l - chord).abs() < 1e-8,
            "segment length {l} vs chord {chord}"
        );
        let (i, j) = match fixed[0] {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let on_ring = |p: [f64; 3]| {
            ring.iter()
                .any(|&(a, b)| (p[i] - c[i] - a).abs() < 1e-8 && (p[j] - c[j] - b).abs() < 1e-8)
        };
        assert!(on_ring(s) && on_ring(e), "vertex not at a 22.5 degree step");
    }
    assert_eq!(
        planes,
        [16, 16, 16],
        "16 segments in each of the three planes"
    );
}

#[test]
fn sphere_rings_are_closed_loops() {
    // every vertex of a ring is the end of one segment and the start of the next
    let mut d = DebugDrawData::new();
    d.sphere(Vec3Fix::ZERO, fx(1.0), DebugColor::RED);
    for ring in 0..3 {
        let seg = &d.lines[ring * 16..ring * 16 + 16];
        for k in 0..16 {
            let next = &seg[(k + 1) % 16];
            assert!(
                close3(arr(seg[k].end), arr(next.start), 1e-8),
                "ring {ring} segment {k} does not meet the next"
            );
        }
    }
}

// ----------------------------------------------------------------- arrow / axes

#[test]
#[ignore = "known defect: AUD-A-S4W3-008: arrow() documents 'line + arrowhead' but the computed head_point is discarded (`let _ = head_point`); only the shaft line and a point marker at the tip are recorded, no head geometry"]
fn arrow_records_head_geometry_besides_the_shaft() {
    let mut d = DebugDrawData::new();
    d.arrow(Vec3Fix::ZERO, v3(3.0, 4.0, 0.0), DebugColor::ORANGE);
    assert!(
        d.lines.len() >= 2,
        "only {} line(s): no arrowhead",
        d.lines.len()
    );
}

#[test]
fn axes_follow_the_rotation_scale_and_position() {
    // 90 degrees about +Z: x -> +y, y -> -x, z -> z
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::PI.half());
    let p = [1.0, 2.0, 3.0];
    let s = 2.5;
    let mut d = DebugDrawData::new();
    d.axes(v3(p[0], p[1], p[2]), q, fx(s));
    assert_eq!(d.lines.len(), 3);
    let want = [[0.0, s, 0.0], [-s, 0.0, 0.0], [0.0, 0.0, s]];
    let colors = [DebugColor::RED, DebugColor::GREEN, DebugColor::BLUE];
    for k in 0..3 {
        let l = &d.lines[k];
        assert_eq!(l.color, colors[k]);
        assert!(close3(arr(l.start), p, 1e-12), "axis {k} start");
        let e = arr(l.end);
        let w = [p[0] + want[k][0], p[1] + want[k][1], p[2] + want[k][2]];
        assert!(close3(e, w, 1e-8), "axis {k}: {e:?} vs {w:?}");
    }
    // tip markers sit at the axis ends, with size 0.2 * (scale)
    assert_eq!(d.points.len(), 3);
    for k in 0..3 {
        assert!((d.points[k].size.to_f64() - 0.2 * s).abs() < 1e-9);
        assert_eq!(d.points[k].position, d.lines[k].end);
    }
}

// ----------------------------------------------------------------- recorder

#[test]
fn clear_empties_both_lists_and_counts_add_up() {
    let mut d = DebugDrawData::new();
    for _ in 0..3 {
        d.line(Vec3Fix::ZERO, Vec3Fix::UNIT_X, DebugColor::RED);
    }
    for _ in 0..5 {
        d.point(Vec3Fix::ZERO, DebugColor::RED, Fix128::ONE);
    }
    assert_eq!(d.primitive_count(), 8);
    d.clear();
    assert!(d.lines.is_empty() && d.points.is_empty());
    assert_eq!(d.primitive_count(), 0);
}

#[test]
fn default_flags_match_the_documented_table() {
    let f = DebugDrawFlags::default();
    assert!(f.draw_aabbs);
    assert!(f.draw_centers);
    assert!(!f.draw_velocities);
    assert!(f.draw_contacts);
    assert!(f.draw_contact_normals);
    assert!(f.draw_joints);
    assert!(!f.draw_bvh);
    assert!(!f.draw_axes);
}

// ----------------------------------------------------------------- debug_draw_world

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let a = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    let mut dynb = RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), Fix128::ONE);
    dynb.velocity = v3(1.0, -2.0, 0.5);
    let b = w.add_body(dynb);
    w.add_distance_constraint(DistanceConstraint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        fx(2.0),
    ));
    w.add_contact(ContactConstraint::new(
        a,
        b,
        Contact {
            depth: fx(0.25),
            normal: v3(0.0, 1.0, 0.0),
            point_a: v3(0.5, 0.0, 0.0),
            point_b: v3(0.5, 0.25, 0.0),
        },
    ));
    w
}

#[test]
fn centers_are_marked_with_the_body_state_color_and_size() {
    let w = scene();
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_centers: true,
            ..none()
        },
        &mut d,
    );
    assert_eq!(d.points.len(), 2);
    assert!(d.lines.is_empty());
    assert_eq!(d.points[0].position, w.bodies[0].position);
    assert_eq!(d.points[0].color, DebugColor::GRAY, "static body");
    assert_eq!(d.points[1].position, w.bodies[1].position);
    assert_eq!(d.points[1].color, DebugColor::GREEN, "dynamic body");
    for p in &d.points {
        assert!((p.size.to_f64() - 0.1).abs() < 1e-12);
    }
}

#[test]
fn contact_points_and_normal_arrow_follow_the_contact() {
    let w = scene();
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_contacts: true,
            draw_contact_normals: true,
            ..none()
        },
        &mut d,
    );
    // points: A (red), B (blue), then the arrow tip marker
    assert_eq!(d.points.len(), 3);
    assert_eq!(arr(d.points[0].position), [0.5, 0.0, 0.0]);
    assert_eq!(d.points[0].color, DebugColor::RED);
    assert_eq!(arr(d.points[1].position), [0.5, 0.25, 0.0]);
    assert_eq!(d.points[1].color, DebugColor::BLUE);
    for p in &d.points[..2] {
        assert!((p.size.to_f64() - 0.05).abs() < 1e-12);
    }
    // normal arrow from point_a along normal * depth
    assert_eq!(d.lines.len(), 1);
    assert_eq!(
        d.lines[0],
        DebugLine::new(v3(0.5, 0.0, 0.0), v3(0.5, 0.25, 0.0), DebugColor::RED)
    );
}

#[test]
fn velocity_arrow_goes_from_the_body_by_its_velocity_for_dynamic_bodies_only() {
    let w = scene();
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_velocities: true,
            ..none()
        },
        &mut d,
    );
    assert_eq!(d.lines.len(), 1);
    assert_eq!(arr(d.lines[0].start), [0.0, 2.0, 0.0]);
    assert_eq!(arr(d.lines[0].end), [1.0, 0.0, 0.5]);
    assert_eq!(d.lines[0].color, DebugColor::YELLOW);
}

#[test]
fn distance_constraint_is_a_cyan_line_between_the_body_centers() {
    let w = scene();
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_joints: true,
            ..none()
        },
        &mut d,
    );
    assert_eq!(d.lines.len(), 1);
    assert_eq!(
        d.lines[0],
        DebugLine::new(w.bodies[0].position, w.bodies[1].position, DebugColor::CYAN)
    );
}

#[test]
fn axes_flag_draws_three_arrows_per_body_at_the_body_pose() {
    let mut w = scene();
    w.bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::PI.half());
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_axes: true,
            ..none()
        },
        &mut d,
    );
    assert_eq!(d.lines.len(), 6);
    // second body's x axis (line index 3) points along +y after the 90 degree rotation
    assert!(close3(arr(d.lines[3].start), [0.0, 2.0, 0.0], 1e-12));
    assert!(close3(arr(d.lines[3].end), [0.0, 3.0, 0.0], 1e-8));
    assert!(close3(arr(d.lines[0].end), [1.0, 0.0, 0.0], 1e-12));
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-009: draw_joints draws only distance_constraints; the world's joint list (ball / hinge / fixed / slider / spring / d6 / cone-twist) is never drawn although the flag is documented as 'joint connections'"]
fn draw_joints_also_draws_the_world_joint_list() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let a = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    let b = w.add_body(RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), Fix128::ONE));
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_joints: true,
            ..none()
        },
        &mut d,
    );
    assert_eq!(d.lines.len(), 1, "ball joint not drawn");
}

#[test]
#[ignore = "known defect: tracked as the unwired draw_aabbs / draw_bvh flags: debug_draw_world never reads draw_aabbs or draw_bvh, so enabling them draws nothing (AABB wireframe exists as DebugDrawData::aabb)"]
fn draw_aabbs_flag_draws_something() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let mut b = RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), Fix128::ONE);
    b.velocity = Vec3Fix::ZERO;
    w.add_body_with_radius(b, fx(0.5));
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_aabbs: true,
            ..none()
        },
        &mut d,
    );
    assert!(d.primitive_count() > 0, "draw_aabbs = true drew nothing");
    let mut d = DebugDrawData::new();
    debug_draw_world(
        &w,
        &DebugDrawFlags {
            draw_bvh: true,
            ..none()
        },
        &mut d,
    );
    assert!(d.primitive_count() > 0, "draw_bvh = true drew nothing");
}
