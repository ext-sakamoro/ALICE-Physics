//! Oracles for `sdf_ccd::sphere_trace_sdf_field`: sphere tracing against a
//! borrowed distance field (no `Box`, no `'static`, no `Send` / `Sync`).
//!
//! 1. Bit identity: the borrowed entry point and `sphere_trace_sdf` give the
//!    same `Option<TOI>` (`TOI: Eq`, compared with `assert_eq!`) for the same
//!    field and pose, over sphere / plane fields and hit / miss / start-inside
//!    cases, with a translated, rotated, scaled pose, and with a pose whose
//!    cache was left stale (position changed without `update_cache`).
//! 2. Closed form: a sphere of radius `r` fired at a unit sphere hits at
//!    `t* = (d - 1 - r) / |disp|`. Tracing stops once the gap is within
//!    `tolerance`, so `t` lies in `[t* - tolerance / |disp|, t*]`.
//! 3. A closure that captures non-`'static`, non-`Send` locals compiles and
//!    is called.
//! 4. Degenerate input: zero displacement, zero radius, zero iterations.
//! 5. The discrete tests `collide_point_sdf_field` / `collide_sphere_sdf_field`
//!    are bit identical to `collide_point_sdf` / `collide_sphere_sdf` and
//!    match the closed-form depth `r - (|c| - 1)` against a unit sphere.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::cell::Cell;
use std::rc::Rc;

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_ccd::{sphere_trace_sdf, sphere_trace_sdf_field, SdfCcdConfig};
use alice_physics::sdf_collider::{
    collide_point_sdf, collide_point_sdf_field, collide_sphere_sdf, collide_sphere_sdf_field,
    ClosureSdf, ClosureSdfQuery, SdfCollider, SdfFrame,
};

fn sphere_d(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt() - 1.0
}

fn sphere_n(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let l = (x * x + y * y + z * z).sqrt();
    if l < 1e-10 {
        (0.0, 1.0, 0.0)
    } else {
        (x / l, y / l, z / l)
    }
}

fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(sphere_d, sphere_n)
}

fn plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

/// (start, displacement, radius) cases: hit, miss, start inside.
fn cases() -> Vec<(Vec3Fix, Vec3Fix, Fix128)> {
    vec![
        // hit along -x side of the sphere / falling onto the plane
        (
            Vec3Fix::from_f32(-5.0, 0.0, 0.0),
            Vec3Fix::from_f32(10.0, 0.0, 0.0),
            Fix128::from_f32(0.5),
        ),
        (
            Vec3Fix::from_f32(0.3, 5.0, -0.2),
            Vec3Fix::from_f32(0.0, -10.0, 0.0),
            Fix128::from_f32(0.25),
        ),
        // miss: parallel, far above
        (
            Vec3Fix::from_f32(-5.0, 5.0, 0.0),
            Vec3Fix::from_f32(10.0, 0.0, 0.0),
            Fix128::from_f32(0.5),
        ),
        // start inside: the gap is already negative
        (
            Vec3Fix::from_f32(0.1, -0.2, 0.0),
            Vec3Fix::from_f32(3.0, 1.0, 0.0),
            Fix128::from_f32(0.5),
        ),
        // oblique hit
        (
            Vec3Fix::from_f32(-4.0, 3.0, 1.0),
            Vec3Fix::from_f32(8.0, -6.0, -1.5),
            Fix128::from_f32(0.4),
        ),
    ]
}

fn colliders(make: fn() -> ClosureSdf) -> Vec<SdfCollider> {
    let rot = QuatFix::from_axis_angle(
        Vec3Fix::from_f32(0.3, 1.0, -0.2).normalize(),
        Fix128::from_f32(0.7),
    );
    let mut stale = SdfCollider::new_static(Box::new(make()), Vec3Fix::ZERO, rot);
    // Change the pose without `update_cache`: the cached inverse rotation now
    // disagrees with `rotation`; both entry points must use the cache.
    stale.rotation = QuatFix::IDENTITY;
    stale.position = Vec3Fix::from_f32(0.2, -0.1, 0.05);
    vec![
        SdfCollider::new_static(Box::new(make()), Vec3Fix::ZERO, QuatFix::IDENTITY),
        SdfCollider::new_static(Box::new(make()), Vec3Fix::from_f32(0.5, -0.25, 0.75), rot)
            .with_scale(Fix128::from_f32(1.5)),
        stale,
    ]
}

#[test]
fn borrowed_field_is_bit_identical_to_collider_entry_point() {
    let config = SdfCcdConfig::default();
    let mut hits = 0;
    let mut misses = 0;
    let mut compared = 0;
    for make in [unit_sphere as fn() -> ClosureSdf, plane] {
        for col in colliders(make) {
            let frame = col.frame();
            // the same field object, borrowed instead of boxed
            let field = make();
            for (start, disp, radius) in cases() {
                let a = sphere_trace_sdf(start, disp, radius, &col, &config);
                let b = sphere_trace_sdf_field(start, disp, radius, &field, &frame, &config);
                // and through the collider's own `dyn SdfField`
                let c = sphere_trace_sdf_field(start, disp, radius, &*col.field, &frame, &config);
                assert_eq!(a, b, "boxed vs borrowed field");
                assert_eq!(a, c, "boxed vs dyn field");
                if a.is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 30);
    // both branches are actually compared
    assert!(hits >= 10 && misses >= 3, "hits {hits} misses {misses}");
}

#[test]
fn frame_new_matches_collider_frame() {
    let rot = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_f32(0.4));
    let pos = Vec3Fix::from_f32(1.0, 2.0, -3.0);
    let scale = Fix128::from_f32(2.5);
    let col = SdfCollider::new_static(Box::new(unit_sphere()), pos, rot).with_scale(scale);
    assert_eq!(SdfFrame::new(pos, rot, scale), col.frame());
    assert_eq!(
        SdfFrame::IDENTITY,
        SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY).frame()
    );
}

#[test]
fn closed_form_toi_against_unit_sphere() {
    let config = SdfCcdConfig::default();
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    // centre distance 5, contact at |x| = 1 + r
    for (r, speed) in [(0.5_f32, 10.0_f32), (0.25, 20.0), (0.0, 10.0)] {
        let start = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
        let disp = Vec3Fix::from_f32(speed, 0.0, 0.0);
        let toi = sphere_trace_sdf_field(
            start,
            disp,
            Fix128::from_f32(r),
            &field,
            &SdfFrame::IDENTITY,
            &config,
        )
        .expect("hit");
        let t = toi.t.to_f32();
        let want = (5.0 - 1.0 - r) / speed;
        let slack = config.tolerance / speed;
        assert!(t <= want + 1e-6, "r {r}: t {t} past t* {want}");
        assert!(
            want - t <= slack + 1e-6,
            "r {r}: t {t} short of t* {want} by more than {slack}"
        );
        let (nx, ny, nz) = toi.normal.to_f32();
        assert!((nx + 1.0).abs() < 1e-4 && ny.abs() < 1e-4 && nz.abs() < 1e-4);
        let (px, _, _) = toi.point.to_f32();
        assert!((px + 1.0).abs() < 2e-3, "contact point x {px}");
    }
}

#[test]
fn closed_form_toi_through_translated_scaled_frame() {
    // A unit sphere scaled by 2 and moved to (10, 0, 0): world radius 2.
    let config = SdfCcdConfig::default();
    let frame = SdfFrame::new(
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        QuatFix::IDENTITY,
        Fix128::from_int(2),
    );
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    let toi = sphere_trace_sdf_field(
        Vec3Fix::ZERO,
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &field,
        &frame,
        &config,
    )
    .expect("hit");
    let t = toi.t.to_f32();
    let want = (10.0 - 2.0 - 0.5) / 10.0;
    assert!(
        t <= want + 1e-6 && want - t <= config.tolerance / 10.0 + 1e-6,
        "t {t}"
    );
    // without the frame the trace sees the sphere at the origin: start inside
    let at_origin = sphere_trace_sdf_field(
        Vec3Fix::ZERO,
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &field,
        &SdfFrame::IDENTITY,
        &config,
    )
    .expect("start inside");
    assert_eq!(at_origin.t, Fix128::ZERO);
}

#[test]
fn borrowed_non_send_closure_compiles_and_is_called() {
    // Locals that are neither `'static` nor `Send`: a borrowed array of
    // centres and an `Rc<Cell>` evaluation counter.
    let centres = [(0.0_f32, 0.0_f32, 0.0_f32), (0.0, 10.0, 0.0)];
    let calls = Rc::new(Cell::new(0_u32));
    let counter = Rc::clone(&calls);
    let nearest = |x: f32, y: f32, z: f32| {
        centres
            .iter()
            .map(|&(cx, cy, cz)| sphere_d(x - cx, y - cy, z - cz))
            .fold(f32::INFINITY, f32::min)
    };
    let field = ClosureSdfQuery::new(
        |x, y, z| {
            counter.set(counter.get() + 1);
            nearest(x, y, z)
        },
        |x, y, z| {
            let (cx, cy, cz) = *centres
                .iter()
                .min_by(|a, b| {
                    sphere_d(x - a.0, y - a.1, z - a.2).total_cmp(&sphere_d(
                        x - b.0,
                        y - b.1,
                        z - b.2,
                    ))
                })
                .expect("non-empty");
            sphere_n(x - cx, y - cy, z - cz)
        },
    );
    let config = SdfCcdConfig::default();
    let toi = sphere_trace_sdf_field(
        Vec3Fix::from_f32(0.0, 5.0, 0.0),
        Vec3Fix::from_f32(0.0, -10.0, 0.0),
        Fix128::from_f32(0.5),
        &field,
        &SdfFrame::IDENTITY,
        &config,
    )
    .expect("hit the lower sphere");
    // centre 5, lower sphere top at 1, contact at y = 1.5: t* = 0.35
    let t = toi.t.to_f32();
    assert!(t <= 0.35 + 1e-6 && 0.35 - t <= 1e-4 + 1e-6, "t {t}");
    assert!(
        calls.get() >= 2,
        "distance closure called {} times",
        calls.get()
    );
}

#[test]
fn degenerate_inputs() {
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    let frame = SdfFrame::IDENTITY;
    let config = SdfCcdConfig::default();
    let start = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
    let disp = Vec3Fix::from_f32(10.0, 0.0, 0.0);
    let r = Fix128::from_f32(0.5);

    // zero displacement: None, even when starting inside
    assert_eq!(
        sphere_trace_sdf_field(start, Vec3Fix::ZERO, r, &field, &frame, &config),
        None
    );
    assert_eq!(
        sphere_trace_sdf_field(Vec3Fix::ZERO, Vec3Fix::ZERO, r, &field, &frame, &config),
        None
    );
    // zero radius: the centre itself reaches the surface, t* = 0.4
    let p =
        sphere_trace_sdf_field(start, disp, Fix128::ZERO, &field, &frame, &config).expect("hit");
    let t = p.t.to_f32();
    assert!(t <= 0.4 + 1e-6 && 0.4 - t <= 1e-4 + 1e-6, "t {t}");
    // zero iterations: None, even when starting inside
    let none = SdfCcdConfig {
        max_iterations: 0,
        ..config
    };
    assert_eq!(
        sphere_trace_sdf_field(start, disp, r, &field, &frame, &none),
        None
    );
    assert_eq!(
        sphere_trace_sdf_field(Vec3Fix::ZERO, disp, r, &field, &frame, &none),
        None
    );
}

/// Query points: inside, on the surface region, outside, for both fields.
fn points() -> Vec<Vec3Fix> {
    vec![
        Vec3Fix::from_f32(0.5, 0.0, 0.0),
        Vec3Fix::from_f32(0.1, -0.3, 0.2),
        Vec3Fix::from_f32(1.2, 0.3, -0.4),
        Vec3Fix::from_f32(-0.7, 0.9, 0.6),
        Vec3Fix::from_f32(2.5, -1.5, 0.5),
        Vec3Fix::from_f32(0.0, 3.0, 0.0),
    ]
}

#[test]
fn discrete_borrowed_tests_are_bit_identical_to_collider_entry_points() {
    let mut hits = 0;
    let mut misses = 0;
    let mut compared = 0;
    for make in [unit_sphere as fn() -> ClosureSdf, plane] {
        for col in colliders(make) {
            let frame = col.frame();
            let field = make();
            for p in points() {
                let a = collide_point_sdf(p, &col);
                assert_eq!(
                    a,
                    collide_point_sdf_field(p, &field, &frame),
                    "point, borrowed"
                );
                assert_eq!(
                    a,
                    collide_point_sdf_field(p, &*col.field, &frame),
                    "point, dyn"
                );
                for r in [Fix128::ZERO, Fix128::from_f32(0.3), Fix128::from_f32(0.8)] {
                    let b = collide_sphere_sdf(p, r, &col);
                    assert_eq!(
                        b,
                        collide_sphere_sdf_field(p, r, &field, &frame),
                        "sphere, borrowed"
                    );
                    assert_eq!(
                        b,
                        collide_sphere_sdf_field(p, r, &*col.field, &frame),
                        "sphere, dyn"
                    );
                    if b.is_some() {
                        hits += 1;
                    } else {
                        misses += 1;
                    }
                    compared += 1;
                }
                if a.is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 2 * 3 * 6 * 4);
    assert!(hits >= 30 && misses >= 30, "hits {hits} misses {misses}");
}

#[test]
fn discrete_closed_form_against_unit_sphere() {
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    let near = |a: Vec3Fix, b: (f32, f32, f32), tol: f32| {
        let (x, y, z) = a.to_f32();
        (x - b.0).abs() < tol && (y - b.1).abs() < tol && (z - b.2).abs() < tol
    };

    // point at x = 0.5 inside the unit sphere: depth 1 - 0.5, normal +x, surface x = 1
    let c = collide_point_sdf_field(
        Vec3Fix::from_f32(0.5, 0.0, 0.0),
        &field,
        &SdfFrame::IDENTITY,
    )
    .expect("inside");
    assert!((c.depth.to_f32() - 0.5).abs() < 1e-5);
    assert!(near(c.normal, (1.0, 0.0, 0.0), 1e-5));
    assert!(near(c.point_b, (1.0, 0.0, 0.0), 1e-5));
    // outside: None
    assert_eq!(
        collide_point_sdf_field(
            Vec3Fix::from_f32(1.5, 0.0, 0.0),
            &field,
            &SdfFrame::IDENTITY
        ),
        None
    );

    // sphere r = 0.5 centred at x = 1.2: depth 0.5 - 0.2 = 0.3, point_a x = 0.7, point_b x = 1
    let s = collide_sphere_sdf_field(
        Vec3Fix::from_f32(1.2, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &field,
        &SdfFrame::IDENTITY,
    )
    .expect("overlap");
    assert!(
        (s.depth.to_f32() - 0.3).abs() < 1e-5,
        "depth {}",
        s.depth.to_f32()
    );
    assert!(near(s.normal, (1.0, 0.0, 0.0), 1e-5));
    assert!(near(s.point_a, (0.7, 0.0, 0.0), 1e-5));
    assert!(near(s.point_b, (1.0, 0.0, 0.0), 1e-5));

    // the same through a frame: unit sphere scaled by 2 at (10, 0, 0), centre
    // at x = 12.3: world distance 0.3, depth 0.5 - 0.3 = 0.2, normal +x
    let frame = SdfFrame::new(
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        QuatFix::IDENTITY,
        Fix128::from_int(2),
    );
    let s = collide_sphere_sdf_field(
        Vec3Fix::from_f32(12.3, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &field,
        &frame,
    )
    .expect("overlap");
    assert!(
        (s.depth.to_f32() - 0.2).abs() < 1e-5,
        "depth {}",
        s.depth.to_f32()
    );
    assert!(near(s.point_b, (12.0, 0.0, 0.0), 1e-5));

    // rotated frame: the plane y = 0 turned 90 degrees about z becomes x = 0
    // with normal -x... rotate (0, 1, 0) by +90 deg about z gives (-1, 0, 0)
    let flat = ClosureSdfQuery::new(
        |_x: f32, y: f32, _z: f32| y,
        |_x: f32, _y: f32, _z: f32| (0.0, 1.0, 0.0),
    );
    let rot = QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f32(std::f32::consts::FRAC_PI_2),
    );
    let rf = SdfFrame::new(Vec3Fix::ZERO, rot, Fix128::ONE);
    // world point (0.25, 0, 0): local y = -0.25, inside by 0.25
    let c = collide_point_sdf_field(Vec3Fix::from_f32(0.25, 0.0, 0.0), &flat, &rf).expect("inside");
    assert!((c.depth.to_f32() - 0.25).abs() < 1e-4);
    assert!(near(c.normal, (-1.0, 0.0, 0.0), 1e-4));
}

#[test]
fn discrete_degenerate_inputs() {
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    let id = SdfFrame::IDENTITY;
    // zero radius sphere = point test: same hit set, same depth and normal bit
    // for bit; `point_b` is formed differently (`c - n * d` vs `p + n * depth`)
    // so it agrees to rounding only. Exactly on the surface both give None.
    let mut inside = 0;
    for p in points() {
        let s = collide_sphere_sdf_field(p, Fix128::ZERO, &field, &id);
        let q = collide_point_sdf_field(p, &field, &id);
        assert_eq!(
            s.map(|c| (c.depth, c.normal)),
            q.map(|c| (c.depth, c.normal))
        );
        if let (Some(s), Some(q)) = (s, q) {
            assert_eq!(s.point_a, p);
            let (dx, dy, dz) = (s.point_b - q.point_b).to_f32();
            assert!(dx.abs() < 1e-12 && dy.abs() < 1e-12 && dz.abs() < 1e-12);
            inside += 1;
        }
    }
    assert_eq!(inside, 2);
    let on = Vec3Fix::from_f32(1.0, 0.0, 0.0);
    assert_eq!(collide_point_sdf_field(on, &field, &id), None);
    assert_eq!(
        collide_sphere_sdf_field(on, Fix128::ZERO, &field, &id),
        None
    );
    // a sphere just touching (gap == radius) is not a contact
    assert_eq!(
        collide_sphere_sdf_field(
            Vec3Fix::from_f32(1.5, 0.0, 0.0),
            Fix128::from_f32(0.5),
            &field,
            &id
        ),
        None
    );
}

#[test]
fn collider_frame_uses_the_cache_until_update_cache() {
    // `SdfCollider` documents that `update_cache` must be called after its
    // pose fields are changed; until then queries use the cached inverse
    // rotation. Plane y = 0 turned +90 degrees about z: the world point
    // (0.25, 0, 0) is at local y = -0.25, inside by 0.25.
    let rot = QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f32(std::f32::consts::FRAC_PI_2),
    );
    let mut col = SdfCollider::new_static(Box::new(plane()), Vec3Fix::ZERO, rot);
    col.rotation = QuatFix::IDENTITY; // stale: inv_rotation still undoes `rot`
    let p = Vec3Fix::from_f32(0.25, 0.0, 0.0);
    let c = collide_point_sdf_field(p, &plane(), &col.frame()).expect("cached pose: inside");
    assert!((c.depth.to_f32() - 0.25).abs() < 1e-4);
    assert_eq!(collide_point_sdf(p, &col), Some(c));
    // after update_cache the plane is y = 0 again and the point is on it
    col.update_cache();
    assert_eq!(collide_point_sdf_field(p, &plane(), &col.frame()), None);
}

#[test]
fn discrete_closed_form_through_rotated_scaled_frame() {
    let field = ClosureSdfQuery::new(sphere_d, sphere_n);
    // unit sphere scaled by 3 at the origin: the point (2, 0, 0) is inside by 1
    let big = SdfFrame::new(Vec3Fix::ZERO, QuatFix::IDENTITY, Fix128::from_int(3));
    // (local distance -1/3, times scale 3)
    let c =
        collide_point_sdf_field(Vec3Fix::from_f32(2.0, 0.0, 0.0), &field, &big).expect("inside");
    assert!(
        (c.depth.to_f32() - 1.0).abs() < 1e-5,
        "depth {}",
        c.depth.to_f32()
    );
    // a point at 3.5 is outside the scaled sphere
    assert_eq!(
        collide_point_sdf_field(Vec3Fix::from_f32(3.5, 0.0, 0.0), &field, &big),
        None
    );

    // plane y = 0 turned +90 degrees about z: local +y is world -x. A sphere
    // of radius 0.5 centred at world x = -0.25 (local y = 0.25) penetrates by
    // 0.25 with world normal -x and surface point x = 0.
    let flat = ClosureSdfQuery::new(
        |_x: f32, y: f32, _z: f32| y,
        |_x: f32, _y: f32, _z: f32| (0.0, 1.0, 0.0),
    );
    let rot = QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f32(std::f32::consts::FRAC_PI_2),
    );
    let rf = SdfFrame::new(Vec3Fix::ZERO, rot, Fix128::ONE);
    let s = collide_sphere_sdf_field(
        Vec3Fix::from_f32(-0.25, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &flat,
        &rf,
    )
    .expect("overlap");
    assert!((s.depth.to_f32() - 0.25).abs() < 1e-4);
    let (nx, ny, nz) = s.normal.to_f32();
    assert!((nx + 1.0).abs() < 1e-4 && ny.abs() < 1e-4 && nz.abs() < 1e-4);
    let (ax, _, _) = s.point_a.to_f32();
    let (bx, _, _) = s.point_b.to_f32();
    assert!((ax - 0.25).abs() < 1e-4 && bx.abs() < 1e-4, "a {ax} b {bx}");
}
