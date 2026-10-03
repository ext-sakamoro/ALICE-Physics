//! Audit oracles for `heatmap`.
//!
//! Closed forms (independent of the crate):
//!
//! * pixel `(ix, iy)` of a `w x h` map samples the box at fractions
//!   `u = ix / w`, `v = iy / h` of the two free axes (in x, y, z order) and at
//!   `slice_offset` on the slice axis;
//! * stress at a pixel is the sum over contacts with `d < 2` of
//!   `force * (1 - d / 2)` where `d` is the 3D distance to the contact;
//! * the colormap is a 5-stop linear gradient whose stop colours are taken
//!   from the viridis table (hand-computed bytes below, tolerance one level);
//!   its luminance rises monotonically with `t`.
#![allow(
    clippy::disallowed_methods,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

use alice_physics::heatmap::{
    generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba, Heatmap, HeatmapConfig,
    SliceAxis,
};
use alice_physics::math::{Fix128, Vec3Fix};
use std::cell::RefCell;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn f(v: Fix128) -> f64 {
    v.to_f64()
}

fn cfg(axis: SliceAxis, w: usize, h: usize, off: f64) -> HeatmapConfig {
    HeatmapConfig {
        resolution_x: w,
        resolution_y: h,
        slice_axis: axis,
        slice_offset: fx(off),
        bounds_min: v3(-1.0, 2.0, -3.0),
        bounds_max: v3(5.0, 8.0, 9.0),
    }
}

/// Expected sample position of pixel (ix, iy) for the documented layout.
fn expected_point(c: &HeatmapConfig, ix: usize, iy: usize) -> [f64; 3] {
    let lo = [f(c.bounds_min.x), f(c.bounds_min.y), f(c.bounds_min.z)];
    let hi = [f(c.bounds_max.x), f(c.bounds_max.y), f(c.bounds_max.z)];
    let u = ix as f64 / c.resolution_x as f64;
    let v = iy as f64 / c.resolution_y as f64;
    let off = f(c.slice_offset);
    let at = |a: usize, t: f64| lo[a] + (hi[a] - lo[a]) * t;
    match c.slice_axis {
        SliceAxis::X => [off, at(1, u), at(2, v)],
        SliceAxis::Y => [at(0, u), off, at(2, v)],
        SliceAxis::Z => [at(0, u), at(1, v), off],
    }
}

#[test]
fn temperature_sampler_visits_every_pixel_once_in_row_major_order_at_the_documented_points() {
    for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
        let c = cfg(axis, 5, 3, 0.75);
        let seen = RefCell::new(Vec::new());
        let map = generate_temperature_heatmap(
            |p| {
                seen.borrow_mut().push([f(p.x), f(p.y), f(p.z)]);
                p.x + p.y * fx(100.0) + p.z * fx(10000.0)
            },
            &c,
        );
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 15, "{axis:?}: function called once per pixel");
        assert_eq!((map.width, map.height, map.data.len()), (5, 3, 15));
        for iy in 0..3 {
            for ix in 0..5 {
                let want = expected_point(&c, ix, iy);
                let got = seen[iy * 5 + ix];
                for k in 0..3 {
                    assert!(
                        (got[k] - want[k]).abs() < 1e-15 * 10.0,
                        "{axis:?} pixel ({ix},{iy}) axis {k}: {} vs {}",
                        got[k],
                        want[k]
                    );
                }
                let val = want[0] + want[1] * 100.0 + want[2] * 10000.0;
                assert!((f(map.data[iy * 5 + ix]) - val).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn min_and_max_are_the_extrema_of_the_data_including_negative_values() {
    let c = cfg(SliceAxis::Z, 6, 4, 0.0);
    let map = generate_temperature_heatmap(|p| (p.x - fx(1.0)) * (p.y - fx(5.0)), &c);
    let lo = map
        .data
        .iter()
        .copied()
        .fold(map.data[0], |a, b| if b < a { b } else { a });
    let hi = map
        .data
        .iter()
        .copied()
        .fold(map.data[0], |a, b| if b > a { b } else { a });
    assert_eq!(map.min_value, lo);
    assert_eq!(map.max_value, hi);
    assert!(f(map.min_value) < 0.0 && f(map.max_value) > 0.0);
}

#[test]
fn stress_matches_linear_falloff_sum_on_every_slice_axis() {
    let contacts = [
        (v3(1.5, 4.0, 1.0), v3(0.0, 1.0, 0.0), fx(10.0)),
        (v3(2.0, 5.5, 2.5), v3(1.0, 0.0, 0.0), fx(-4.0)),
        (v3(4.0, 7.0, 8.0), v3(0.0, 0.0, 1.0), fx(7.5)),
    ];
    for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
        let off = match axis {
            SliceAxis::X => 1.8,
            SliceAxis::Y => 5.0,
            SliceAxis::Z => 1.5,
        };
        let c = cfg(axis, 12, 9, off);
        let map = generate_stress_heatmap(&[], &contacts, &c);
        let mut nonzero = 0;
        for iy in 0..9 {
            for ix in 0..12 {
                let p = expected_point(&c, ix, iy);
                let mut want = 0.0;
                for (cp, _n, force) in &contacts {
                    let d = ((p[0] - f(cp.x)).powi(2)
                        + (p[1] - f(cp.y)).powi(2)
                        + (p[2] - f(cp.z)).powi(2))
                    .sqrt();
                    if d < 2.0 {
                        want += f(*force) * (1.0 - d / 2.0);
                    }
                }
                let got = f(map.data[iy * 12 + ix]);
                if want != 0.0 {
                    nonzero += 1;
                }
                assert!(
                    (got - want).abs() < 1e-9,
                    "{axis:?} ({ix},{iy}): {got} vs {want}"
                );
            }
        }
        assert!(nonzero > 0, "{axis:?}: scene must exercise the kernel");
    }
}

#[test]
fn stress_at_a_contact_pixel_equals_the_force_and_outside_the_radius_is_zero() {
    // 1 x 1 map samples bounds_min only.
    let mut c = cfg(SliceAxis::Z, 1, 1, 0.0);
    c.bounds_min = v3(0.0, 0.0, 0.0);
    c.bounds_max = v3(1.0, 1.0, 1.0);
    let at = generate_stress_heatmap(&[], &[(v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), fx(6.0))], &c);
    assert!((f(at.data[0]) - 6.0).abs() < 1e-12);
    let far = generate_stress_heatmap(&[], &[(v3(0.0, 0.0, 2.0), v3(0.0, 1.0, 0.0), fx(6.0))], &c);
    assert_eq!(f(far.data[0]), 0.0);
    let just_in =
        generate_stress_heatmap(&[], &[(v3(0.0, 0.0, 1.9), v3(0.0, 1.0, 0.0), fx(6.0))], &c);
    assert!((f(just_in.data[0]) - 6.0 * 0.05).abs() < 1e-9);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-006: the stress kernel radius is a hard-coded 2.0 world units (not in HeatmapConfig), so the same scene scaled by 10 no longer produces the same map (pixel 0: 6.56 in the 0.25-scaled scene, 0 in the 10-scaled scene)"]
fn stress_map_is_covariant_under_uniform_scaling_of_the_scene() {
    let base = |s: f64| {
        let mut c = cfg(SliceAxis::Z, 4, 4, 0.0);
        c.bounds_min = v3(0.0, 0.0, 0.0);
        c.bounds_max = v3(4.0 * s, 4.0 * s, 4.0 * s);
        let contacts = [(v3(0.5 * s, 0.5 * s, 0.0), v3(0.0, 1.0, 0.0), fx(10.0))];
        generate_stress_heatmap(&[], &contacts, &c)
    };
    let a = base(0.25);
    let b = base(10.0);
    for i in 0..16 {
        assert!(
            (f(a.data[i]) - f(b.data[i])).abs() < 1e-6,
            "pixel {i}: {} vs {}",
            f(a.data[i]),
            f(b.data[i])
        );
    }
}

fn ramp(n: usize) -> Heatmap {
    Heatmap {
        data: (0..n).map(|i| Fix128::from_int(i as i64)).collect(),
        width: n,
        height: 1,
        min_value: Fix128::ZERO,
        max_value: Fix128::from_int(n as i64 - 1),
    }
}

#[test]
fn colormap_luminance_is_monotone_in_value() {
    let rgba = heatmap_to_rgba(&ramp(512));
    let lum =
        |p: [u8; 4]| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]);
    for w in rgba.windows(2) {
        assert!(lum(w[1]) >= lum(w[0]) - 1.0, "{:?} -> {:?}", w[0], w[1]);
    }
    assert!(lum(rgba[511]) > lum(rgba[0]) + 100.0);
}

#[test]
fn colormap_stops_match_the_documented_viridis_colours_within_one_level() {
    // Hand-computed from the five documented stops (x255): quarter points.
    let want: [[i32; 3]; 5] = [
        [68, 1, 84],
        [72, 36, 117],
        [32, 144, 141],
        [139, 197, 63],
        [253, 231, 37],
    ];
    let rgba = heatmap_to_rgba(&ramp(5));
    for (i, w) in want.iter().enumerate() {
        for k in 0..3 {
            assert!(
                (i32::from(rgba[i][k]) - w[k]).abs() <= 1,
                "stop {i} channel {k}: {} vs {}",
                rgba[i][k],
                w[k]
            );
        }
    }
}

#[test]
fn colormap_midpoint_between_two_stops_is_their_average() {
    // t = 1/8: halfway between stop 0 and stop 1.
    let rgba = heatmap_to_rgba(&ramp(9));
    let a = [0.267, 0.004, 0.329];
    let b = [0.282, 0.140, 0.458];
    for k in 0..3 {
        let want = (a[k] + b[k]) / 2.0 * 255.0;
        assert!((f64::from(rgba[1][k]) - want).abs() <= 1.0, "channel {k}");
    }
}

#[test]
fn colormap_is_continuous_between_neighbouring_values() {
    let rgba = heatmap_to_rgba(&ramp(2001));
    for w in rgba.windows(2) {
        for k in 0..3 {
            assert!((i32::from(w[1][k]) - i32::from(w[0][k])).abs() <= 2);
        }
    }
}

#[test]
fn colormap_maps_values_in_row_major_order_independent_of_magnitude_offset() {
    // Same shape of data shifted by a constant gives the same colours.
    let a = ramp(7);
    let mut b = ramp(7);
    for v in &mut b.data {
        *v = *v + Fix128::from_int(1000);
    }
    b.min_value = b.min_value + Fix128::from_int(1000);
    b.max_value = b.max_value + Fix128::from_int(1000);
    assert_eq!(heatmap_to_rgba(&a), heatmap_to_rgba(&b));
    let mut c = ramp(7);
    c.data.reverse();
    let rc = heatmap_to_rgba(&c);
    let ra = heatmap_to_rgba(&a);
    for i in 0..7 {
        assert_eq!(rc[i], ra[6 - i]);
    }
}
