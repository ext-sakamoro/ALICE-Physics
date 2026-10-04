//! Audit oracles for `sdf_destruction`.
//!
//! Closed forms (independent of the crate):
//!
//! * sharp subtraction: `d = max(d_orig, -d_shape)`, with the textbook SDFs of
//!   a sphere `|l| - r`, a box `|max(q,0)| + min(max(q),0)` with `q = |l| - h`
//!   and a Y-axis cylinder, evaluated at the local point `l = q^-1 (p - c)`;
//! * smooth subtraction with factor `k`: `max(a, b) + k h^2 / 4` where
//!   `b = -d_shape` and `h = max(1 - |a - b| / k, 0)`;
//! * a projectile bore along unit direction `u` from `e` with depth `L` is a
//!   cylinder whose end caps sit at `e` and `e + L u`.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_destruction::{
    destruction_from_explosion, destruction_from_impact, destruction_from_projectile,
    DestructibleSdf, DestructionShape, DestructionType,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}
fn deep() -> ClosureSdf {
    ClosureSdf::new(|_x, _y, _z| -100.0, |_x, _y, _z| (0.0, 1.0, 0.0))
}
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

fn quat_f64(axis: [f64; 3], ang: f64) -> [f64; 4] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let s = (ang / 2.0).sin() / n;
    [axis[0] * s, axis[1] * s, axis[2] * s, (ang / 2.0).cos()]
}
fn quat_fix(q: [f64; 4]) -> QuatFix {
    QuatFix::new(fx(q[0]), fx(q[1]), fx(q[2]), fx(q[3]))
}
fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let c = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = c(u, v);
    let t2 = c(u, t);
    [
        v[0] + 2.0 * (w * t[0] + t2[0]),
        v[1] + 2.0 * (w * t[1] + t2[1]),
        v[2] + 2.0 * (w * t[2] + t2[2]),
    ]
}
fn conj(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}
fn len3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn box_sdf(l: [f64; 3], h: [f64; 3]) -> f64 {
    let q = [l[0].abs() - h[0], l[1].abs() - h[1], l[2].abs() - h[2]];
    let out = len3([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]);
    out + q[0].max(q[1]).max(q[2]).min(0.0)
}
fn cyl_sdf(l: [f64; 3], r: f64, hh: f64) -> f64 {
    let dr = (l[0] * l[0] + l[2] * l[2]).sqrt() - r;
    let da = l[1].abs() - hh;
    let out = (dr.max(0.0).powi(2) + da.max(0.0).powi(2)).sqrt();
    out + dr.max(da).min(0.0)
}

const C: [f64; 3] = [0.5, -0.25, 1.0];

#[test]
fn sharp_subtraction_is_max_of_original_and_negated_shape_for_rotated_shapes() {
    let q = quat_f64([1.0, 2.0, 0.5], 0.9);
    let cases: Vec<(&str, DestructionShape, Box<dyn Fn([f64; 3]) -> f64>)> = vec![
        (
            "sphere",
            DestructionShape::sphere(v3(C[0], C[1], C[2]), 0.8).with_rotation(quat_fix(q)),
            Box::new(|l| len3(l) - 0.8),
        ),
        (
            "box",
            DestructionShape::cube(v3(C[0], C[1], C[2]), (0.6, 0.3, 0.9))
                .with_rotation(quat_fix(q)),
            Box::new(|l| box_sdf(l, [0.6, 0.3, 0.9])),
        ),
        (
            "cylinder",
            DestructionShape::cylinder(v3(C[0], C[1], C[2]), 0.5, 0.7).with_rotation(quat_fix(q)),
            Box::new(|l| cyl_sdf(l, 0.5, 0.7)),
        ),
    ];
    for (name, shape, local) in cases {
        let mut d = DestructibleSdf::new(Box::new(ground()));
        d.apply_destruction(shape);
        let mut s = 5u64;
        let mut carved = 0;
        for _ in 0..400 {
            let p = [
                C[0] + 2.0 * lcg(&mut s),
                C[1] + 2.0 * lcg(&mut s),
                C[2] + 2.0 * lcg(&mut s),
            ];
            let l = rot(conj(q), [p[0] - C[0], p[1] - C[1], p[2] - C[2]]);
            let want = p[1].max(-local(l));
            let got = f64::from(d.distance(p[0] as f32, p[1] as f32, p[2] as f32));
            if -local(l) > p[1] {
                carved += 1;
            }
            assert!(
                (got - want).abs() < 2e-4,
                "{name} at {p:?}: {got} vs {want}"
            );
        }
        assert!(
            carved > 20,
            "{name}: scene must exercise carving ({carved})"
        );
    }
}

#[test]
fn custom_shape_is_evaluated_in_the_local_frame() {
    let q = quat_f64([0.0, 0.0, 1.0], 0.7);
    let mut shape = DestructionShape::sphere(v3(C[0], C[1], C[2]), 1.0).with_rotation(quat_fix(q));
    shape.shape = DestructionType::Custom {
        sdf: Arc::new(|lx, ly, lz| 2.0 * lx + 0.5 * ly - 0.25 * lz + 0.1),
    };
    let mut d = DestructibleSdf::new(Box::new(deep()));
    d.apply_destruction(shape);
    for p in [[1.0, 0.0, 0.5], [-0.5, 1.5, 2.0], [0.2, -0.3, 0.9]] {
        let l = rot(conj(q), [p[0] - C[0], p[1] - C[1], p[2] - C[2]]);
        let want = -(2.0 * l[0] + 0.5 * l[1] - 0.25 * l[2] + 0.1);
        let got = f64::from(d.distance(p[0] as f32, p[1] as f32, p[2] as f32));
        assert!((got - want).abs() < 2e-4, "{p:?}: {got} vs {want}");
    }
}

fn smax_ref(a: f64, b: f64, k: f64) -> f64 {
    let h = (1.0 - (a - b).abs() / k).max(0.0);
    a.max(b) + k * h * h / 4.0
}

#[test]
fn smooth_subtraction_matches_the_quadratic_smooth_max() {
    for k in [0.05_f32, 0.4, 1.5] {
        let mut d = DestructibleSdf::new(Box::new(ground()));
        d.apply_destruction(DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0).with_smoothing(k));
        let mut blended = 0;
        for i in 0..400 {
            let t = f64::from(i) / 399.0;
            let p = [0.0, 0.2 + 1.5 * t, 0.0];
            let a = p[1];
            let b = -(p[1].abs() - 1.0);
            let want = smax_ref(a, b, f64::from(k));
            if (a - b).abs() < f64::from(k) {
                blended += 1;
            }
            let got = f64::from(d.distance(p[0] as f32, p[1] as f32, p[2] as f32));
            assert!(
                (got - want).abs() < 2e-5,
                "k={k} y={}: {got} vs {want}",
                p[1]
            );
            let sharp = a.max(b);
            assert!(got >= sharp - 1e-6 && got <= sharp + f64::from(k) / 4.0 + 1e-6);
        }
        assert!(
            blended > 0 || k < 0.1,
            "k={k}: blend region must be sampled"
        );
    }
}

#[test]
fn non_positive_smoothing_is_a_sharp_subtraction() {
    for k in [0.0_f32, -0.5] {
        let mut d = DestructibleSdf::new(Box::new(ground()));
        d.apply_destruction(DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0).with_smoothing(k));
        let got = d.distance(0.0, 0.5, 0.0);
        assert!((got - 0.5).abs() < 1e-6, "k={k}: {got}");
        let got = d.distance(0.0, 0.0, 0.0);
        assert!((got - 1.0).abs() < 1e-6, "k={k}: {got}");
    }
}

#[test]
fn sharp_destructions_commute() {
    let a = || DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0);
    let b = || DestructionShape::cube(v3(0.5, 0.2, 0.0), (0.4, 0.4, 0.4));
    let c = || DestructionShape::cylinder(v3(-0.5, 0.1, 0.3), 0.3, 0.6);
    let mk = |order: [usize; 3]| {
        let mut d = DestructibleSdf::new(Box::new(ground()));
        let shapes = [a(), b(), c()];
        for i in order {
            d.apply_destruction(shapes[i].clone());
        }
        d
    };
    let d1 = mk([0, 1, 2]);
    let d2 = mk([2, 0, 1]);
    let mut s = 9u64;
    for _ in 0..200 {
        let (x, y, z) = (
            lcg(&mut s) as f32 * 2.0,
            lcg(&mut s) as f32 * 2.0,
            lcg(&mut s) as f32 * 2.0,
        );
        assert_eq!(d1.distance(x, y, z), d2.distance(x, y, z));
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-009: optimize() is documented as removing shapes fully contained by newer ones, but drops the 8 oldest of 40 disjoint craters regardless of containment, so carved geometry is restored (crater centre reads 1.0 before optimize, 0.0 after)"]
fn optimize_does_not_restore_material_carved_by_disjoint_craters() {
    let mut d = DestructibleSdf::new(Box::new(ground()));
    for i in 0..40 {
        d.apply_destruction(DestructionShape::sphere(
            v3(f64::from(i) * 3.0, 0.0, 0.0),
            1.0,
        ));
    }
    let before: Vec<f32> = (0..40)
        .map(|i| d.distance(i as f32 * 3.0, 0.0, 0.0))
        .collect();
    d.optimize();
    for i in 0..40 {
        let after = d.distance(i as f32 * 3.0, 0.0, 0.0);
        assert_eq!(
            after, before[i],
            "crater {i} changed: {} -> {after}",
            before[i]
        );
    }
}

#[test]
fn optimize_leaves_the_field_unchanged_when_nothing_exceeds_the_cap() {
    let mut d = DestructibleSdf::new(Box::new(ground()));
    for i in 0..32 {
        d.apply_destruction(DestructionShape::sphere(
            v3(f64::from(i) * 3.0, 0.0, 0.0),
            1.0,
        ));
    }
    let before: Vec<f32> = (0..32)
        .map(|i| d.distance(i as f32 * 3.0, 0.3, 0.0))
        .collect();
    d.optimize();
    for i in 0..32 {
        assert_eq!(d.distance(i as f32 * 3.0, 0.3, 0.0), before[i]);
    }
}

#[test]
fn impact_crater_is_centred_on_point_b_and_uses_the_speed_magnitude() {
    let contact = Contact {
        depth: fx(0.1),
        normal: v3(0.0, 1.0, 0.0),
        point_a: v3(10.0, 10.0, 10.0),
        point_b: v3(1.0, 2.0, 3.0),
    };
    for v in [4.0, -4.0] {
        let s = destruction_from_impact(&contact, fx(v), 0.25, 0.1, 5.0);
        assert_eq!(s.center, v3(1.0, 2.0, 3.0));
        match s.shape {
            DestructionType::Sphere { radius } => {
                assert!((radius - 1.0).abs() < 1e-6, "v={v}: {radius}")
            }
            ref o => panic!("expected sphere, got {o:?}"),
        }
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-010: destruction_from_impact panics (f32::clamp assertion) when min_radius > max_radius, with no documented precondition (min 2.0, max 1.0)"]
fn impact_with_inverted_radius_range_does_not_panic() {
    let contact = Contact {
        depth: fx(0.1),
        normal: v3(0.0, 1.0, 0.0),
        point_a: v3(0.0, 0.0, 0.0),
        point_b: v3(0.0, 0.0, 0.0),
    };
    let r = catch_unwind(AssertUnwindSafe(|| {
        destruction_from_impact(&contact, fx(4.0), 0.25, 2.0, 1.0)
    }));
    assert!(r.is_ok(), "panicked on min_radius > max_radius");
}

#[test]
fn explosion_is_a_smooth_sphere_at_the_centre() {
    let s = destruction_from_explosion(v3(1.0, 2.0, 3.0), 2.5, 0.3);
    assert_eq!(s.center, v3(1.0, 2.0, 3.0));
    assert_eq!(s.smooth_factor, 0.3);
    assert!(matches!(s.shape, DestructionType::Sphere { radius } if radius == 2.5));
}

fn unit(d: [f64; 3]) -> [f64; 3] {
    let l = len3(d);
    [d[0] / l, d[1] / l, d[2] / l]
}

#[test]
fn projectile_rotation_maps_the_cylinder_axis_onto_the_direction_away_from_the_poles() {
    let mut s = 21u64;
    let mut n = 0;
    while n < 40 {
        let d = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        if len3(d) < 0.3 {
            continue;
        }
        let u = unit(d);
        if u[1].abs() > 0.99 {
            continue;
        }
        n += 1;
        let shape = destruction_from_projectile(v3(0.0, 0.0, 0.0), v3(d[0], d[1], d[2]), 0.5, 2.0);
        let r = shape.rotation.rotate_vec(Vec3Fix::UNIT_Y).to_f32();
        assert!((f64::from(r.0) - u[0]).abs() < 1e-5, "{u:?} -> {r:?}");
        assert!((f64::from(r.1) - u[1]).abs() < 1e-5, "{u:?} -> {r:?}");
        assert!((f64::from(r.2) - u[2]).abs() < 1e-5, "{u:?} -> {r:?}");
    }
}

#[test]
fn projectile_bore_end_caps_sit_at_entry_and_entry_plus_depth_for_a_unit_direction() {
    let e = [1.0, 2.0, -1.0];
    let u = unit([0.6, 0.3, -0.5]);
    let depth = 4.0_f32;
    let dd = f64::from(depth);
    let shape = destruction_from_projectile(v3(e[0], e[1], e[2]), v3(u[0], u[1], u[2]), 0.5, depth);
    let mut d = DestructibleSdf::new(Box::new(deep()));
    d.apply_destruction(shape);
    let at = |t: f64, off: [f64; 3]| {
        let p = [
            e[0] + u[0] * t + off[0],
            e[1] + u[1] * t + off[1],
            e[2] + u[2] * t + off[2],
        ];
        -f64::from(d.distance(p[0] as f32, p[1] as f32, p[2] as f32))
    };
    assert!(at(0.0, [0.0; 3]).abs() < 1e-4, "entry cap");
    assert!(at(dd, [0.0; 3]).abs() < 1e-4, "far cap");
    assert!(
        (at(dd / 2.0, [0.0; 3]) + 0.5).abs() < 1e-4,
        "axis centre is r inside"
    );
    assert!(
        (at(-0.5, [0.0; 3]) - 0.5).abs() < 1e-4,
        "0.5 before the entry cap"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-011: destruction_from_projectile places the centre at entry + direction * depth/2 without normalising, so a non-unit direction (0,0,2), depth 4 gives centre z = 4 instead of 2"]
fn projectile_centre_does_not_depend_on_the_length_of_the_direction() {
    let a = destruction_from_projectile(v3(0.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), 0.5, 4.0);
    let b = destruction_from_projectile(v3(0.0, 0.0, 0.0), v3(0.0, 0.0, 2.0), 0.5, 4.0);
    assert_eq!(a.center, b.center);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-012: a direction within about 2.6 degrees of +Y or -Y is snapped to the pole (dot > 0.999): direction (sin 2deg, cos 2deg, 0) gives identity rotation, so the bore axis is off by 2 degrees (0.035 per unit length)"]
fn projectile_direction_near_the_pole_is_not_snapped_to_it() {
    let a = 2.0_f64.to_radians();
    for sign in [1.0, -1.0] {
        let u = [a.sin(), sign * a.cos(), 0.0];
        let shape = destruction_from_projectile(v3(0.0, 0.0, 0.0), v3(u[0], u[1], u[2]), 0.5, 2.0);
        let r = shape.rotation.rotate_vec(Vec3Fix::UNIT_Y).to_f32();
        assert!(
            (f64::from(r.0) - u[0]).abs() < 1e-4 && (f64::from(r.1) - u[1]).abs() < 1e-4,
            "sign {sign}: {r:?} vs {u:?}"
        );
    }
}

#[test]
fn normal_points_toward_the_crater_centre_inside_a_carved_region() {
    let mut d = DestructibleSdf::new(Box::new(ground()));
    d.apply_destruction(DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0));
    for p in [[0.3, 0.2, 0.1], [-0.4, -0.2, 0.2], [0.1, -0.3, 0.6]] {
        // Inside the crater (|p| < 1): d = 1 - |p| > y there, gradient = -p/|p|.
        let l = len3(p);
        assert!(1.0 - l > p[1], "point must be carved");
        let n = d.normal(p[0] as f32, p[1] as f32, p[2] as f32);
        let want = [-p[0] / l, -p[1] / l, -p[2] / l];
        assert!((f64::from(n.0) - want[0]).abs() < 2e-3, "{p:?}");
        assert!((f64::from(n.1) - want[1]).abs() < 2e-3, "{p:?}");
        assert!((f64::from(n.2) - want[2]).abs() < 2e-3, "{p:?}");
    }
    // Outside the crater the ground normal is untouched.
    let n = d.normal(5.0, 1.0, 0.0);
    assert!(n.0.abs() < 1e-3 && (n.1 - 1.0).abs() < 1e-3 && n.2.abs() < 1e-3);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-013: DestructibleSdf::normal uses the absolute finite-difference step 0.001 in f32, so far from the origin the step is lost (unit sphere at (1e4,5e3,0): normal (1,0,0) instead of (0.894,0.447,0))"]
fn normal_is_accurate_far_from_the_origin() {
    let d = DestructibleSdf::new(Box::new(unit_sphere()));
    let n = d.normal(1.0e4, 5.0e3, 0.0);
    let s5 = 5.0_f32.sqrt();
    assert!(
        (n.0 - 2.0 / s5).abs() < 1e-2 && (n.1 - 1.0 / s5).abs() < 1e-2 && n.2.abs() < 1e-2,
        "{n:?}"
    );
}

#[test]
fn non_unit_rotation_does_not_resize_the_crater() {
    let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, fx(2.0));
    let mut d = DestructibleSdf::new(Box::new(deep()));
    d.apply_destruction(DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0).with_rotation(q));
    let got = d.distance(0.5, 0.0, 0.0);
    assert!(
        (got - 0.5).abs() < 1e-5,
        "distance {got}, expected 1 - 0.5 = 0.5"
    );
}

#[test]
fn crater_depth_reads_back_as_the_radius_at_the_centre_for_each_shape() {
    for shape in [
        DestructionShape::sphere(v3(0.0, 0.0, 0.0), 0.7),
        DestructionShape::cube(v3(0.0, 0.0, 0.0), (0.7, 0.9, 1.1)),
        DestructionShape::cylinder(v3(0.0, 0.0, 0.0), 0.7, 0.9),
    ] {
        let mut d = DestructibleSdf::new(Box::new(deep()));
        d.apply_destruction(shape);
        assert!((d.distance(0.0, 0.0, 0.0) - 0.7).abs() < 1e-5);
    }
}

#[test]
fn normal_finite_difference_step_is_small_enough_for_a_curved_crater() {
    // Crater field -d_shape = y + c x^3 (original field far below): exact gradient
    // (3 c x^2, 1, 0); a central difference of step e adds c e^2 to the x component.
    let c = 100.0_f32;
    let mut shape = DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0);
    shape.shape = DestructionType::Custom {
        sdf: Arc::new(move |lx, ly, _lz| -(c * lx * lx * lx + ly)),
    };
    let mut d = DestructibleSdf::new(Box::new(deep()));
    d.apply_destruction(shape);
    let x = 0.1_f64;
    let gx = 3.0 * f64::from(c) * x * x;
    let l = (gx * gx + 1.0).sqrt();
    let n = d.normal(x as f32, 0.0, 0.0);
    assert!(
        (f64::from(n.0) - gx / l).abs() < 2e-4,
        "nx {} vs {}",
        n.0,
        gx / l
    );
    assert!(
        (f64::from(n.1) - 1.0 / l).abs() < 2e-4,
        "ny {} vs {}",
        n.1,
        1.0 / l
    );
}

#[test]
fn normal_direction_does_not_depend_on_a_small_slope_magnitude() {
    let mut shape = DestructionShape::sphere(v3(0.0, 0.0, 0.0), 1.0);
    shape.shape = DestructionType::Custom {
        sdf: Arc::new(|lx, _ly, _lz| -0.1 * lx),
    };
    let mut d = DestructibleSdf::new(Box::new(deep()));
    d.apply_destruction(shape);
    let n = d.normal(0.3, 0.2, 0.1);
    assert!(
        (n.0 - 1.0).abs() < 1e-3 && n.1.abs() < 1e-3 && n.2.abs() < 1e-3,
        "{n:?}"
    );
}

#[test]
fn projectile_direction_a_few_degrees_off_either_pole_still_aligns() {
    for deg in [5.0_f64, 10.0, 20.0] {
        let a = deg.to_radians();
        for sign in [1.0, -1.0] {
            let u = [a.sin(), sign * a.cos(), 0.0];
            let shape =
                destruction_from_projectile(v3(0.0, 0.0, 0.0), v3(u[0], u[1], u[2]), 0.5, 2.0);
            let r = shape.rotation.rotate_vec(Vec3Fix::UNIT_Y).to_f32();
            assert!(
                (f64::from(r.0) - u[0]).abs() < 1e-5
                    && (f64::from(r.1) - u[1]).abs() < 1e-5
                    && f64::from(r.2).abs() < 1e-5,
                "{deg} deg, sign {sign}: {r:?} vs {u:?}"
            );
        }
    }
}
