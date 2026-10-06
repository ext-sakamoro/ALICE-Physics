//! Audit oracles for `alice_physics::heightfield`.
//!
//! Expected values come from an independent `f64` bilinear reference, from planes
//! and the bilinear saddle `h = x·z` (whose central difference is exact), and from
//! the tangent-plane distance of a sphere to a plane.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

/// Build a field whose vertex (gx, gz) has height `f(world_x, world_z)`.
fn field_from(
    w: u32,
    d: u32,
    spacing: f64,
    origin: [f64; 3],
    f: impl Fn(f64, f64) -> f64,
) -> HeightField {
    let mut hs = Vec::new();
    for gz in 0..d {
        for gx in 0..w {
            hs.push(fx(f(
                origin[0] + spacing * f64::from(gx),
                origin[2] + spacing * f64::from(gz),
            )));
        }
    }
    HeightField::new(hs, w, d, fx(spacing), v3(origin[0], origin[1], origin[2]))
}

/// Independent bilinear reference with edge clamping (continuous clamp of the
/// grid coordinate to `[0, n-1]`).
fn reference(hs: &[f64], w: u32, d: u32, spacing: f64, origin: [f64; 3], x: f64, z: f64) -> f64 {
    let at = |gx: u32, gz: u32| hs[(gx + gz * w) as usize];
    let ux = ((x - origin[0]) / spacing).clamp(0.0, f64::from(w - 1));
    let uz = ((z - origin[2]) / spacing).clamp(0.0, f64::from(d - 1));
    let ix = (ux.floor() as u32).min(w.saturating_sub(2));
    let iz = (uz.floor() as u32).min(d.saturating_sub(2));
    let tx = ux - f64::from(ix);
    let tz = uz - f64::from(iz);
    let (ix1, iz1) = ((ix + 1).min(w - 1), (iz + 1).min(d - 1));
    let h0 = at(ix, iz) * (1.0 - tx) + at(ix1, iz) * tx;
    let h1 = at(ix, iz1) * (1.0 - tx) + at(ix1, iz1) * tx;
    h0 * (1.0 - tz) + h1 * tz
}

// ------------------------------------------------------------------ sample_height

#[test]
fn sample_height_is_bilinear_on_a_random_non_square_field_with_offset_origin() {
    let (w, d, sp) = (6u32, 4u32, 0.5);
    let origin = [-1.25, 7.0, 2.5];
    let mut s = 99u64;
    let hs: Vec<f64> = (0..w * d).map(|_| lcg(&mut s) * 5.0).collect();
    let field = HeightField::new(
        hs.iter().map(|&h| fx(h)).collect(),
        w,
        d,
        fx(sp),
        v3(origin[0], origin[1], origin[2]),
    );
    for _ in 0..400 {
        // inside the grid
        let x = origin[0] + lcg(&mut s).abs() * sp * f64::from(w - 1);
        let z = origin[2] + lcg(&mut s).abs() * sp * f64::from(d - 1);
        let got = field.sample_height(fx(x), fx(z)).to_f64();
        let want = reference(&hs, w, d, sp, origin, x, z);
        assert!((got - want).abs() < 1e-9, "({x}, {z}): {got} vs {want}");
    }
}

#[test]
fn sample_height_at_a_vertex_is_the_vertex_height() {
    let (w, d, sp) = (5u32, 5u32, 0.25);
    let mut s = 5u64;
    let hs: Vec<f64> = (0..w * d).map(|_| lcg(&mut s) * 3.0).collect();
    let field = HeightField::new(
        hs.iter().map(|&h| fx(h)).collect(),
        w,
        d,
        fx(sp),
        Vec3Fix::ZERO,
    );
    for gz in 0..d {
        for gx in 0..w {
            let got = field
                .sample_height(fx(sp * f64::from(gx)), fx(sp * f64::from(gz)))
                .to_f64();
            let want = hs[(gx + gz * w) as usize];
            assert!((got - want).abs() < 1e-9, "vertex ({gx},{gz})");
        }
    }
}

#[test]
fn sample_height_outside_the_grid_holds_the_nearest_edge_value() {
    let (w, d, sp) = (4u32, 3u32, 1.0);
    let mut s = 8u64;
    let hs: Vec<f64> = (0..w * d).map(|_| lcg(&mut s) * 3.0).collect();
    let field = HeightField::new(
        hs.iter().map(|&h| fx(h)).collect(),
        w,
        d,
        fx(sp),
        Vec3Fix::ZERO,
    );
    for &(x, z) in &[
        (-50.5, 0.5),
        (1000.25, 1.5),
        (1.5, -7.25),
        (2.5, 900.0),
        (-3.0, -3.0),
        (99.0, 99.0),
        (6.0e9, 1.5),
        (1.5, 7.0e9),
        (-6.0e9, 1.5),
    ] {
        let got = field.sample_height(fx(x), fx(z)).to_f64();
        let want = reference(&hs, w, d, sp, [0.0; 3], x, z);
        assert!((got - want).abs() < 1e-9, "({x}, {z}): {got} vs {want}");
    }
}

#[test]
fn single_column_and_single_row_fields_interpolate_along_the_long_axis() {
    // width 1, depth 3: heights along z = [1, 3, 5]
    let f = HeightField::new(
        vec![fx(1.0), fx(3.0), fx(5.0)],
        1,
        3,
        fx(1.0),
        Vec3Fix::ZERO,
    );
    for &x in &[-4.0, 0.0, 0.7, 12.0] {
        assert!((f.sample_height(fx(x), fx(0.5)).to_f64() - 2.0).abs() < 1e-9);
        assert!((f.sample_height(fx(x), fx(1.75)).to_f64() - 4.5).abs() < 1e-9);
    }
    // width 3, depth 1
    let g = HeightField::new(
        vec![fx(1.0), fx(3.0), fx(5.0)],
        3,
        1,
        fx(1.0),
        Vec3Fix::ZERO,
    );
    for &z in &[-4.0, 0.0, 9.0] {
        assert!((g.sample_height(fx(0.5), fx(z)).to_f64() - 2.0).abs() < 1e-9);
        assert!((g.sample_height(fx(1.75), fx(z)).to_f64() - 4.5).abs() < 1e-9);
    }
}

#[test]
fn signed_distance_is_the_vertical_offset_from_the_interpolated_surface() {
    let (w, d, sp) = (5u32, 5u32, 1.0);
    let field = field_from(w, d, sp, [0.0; 3], |x, z| 0.3 * x * z);
    for &(x, z, y) in &[(1.5, 2.5, 3.0), (3.25, 0.75, -1.0), (0.0, 0.0, 0.5)] {
        let surf = 0.3 * x * z;
        let got = field.signed_distance(v3(x, y, z)).to_f64();
        assert!((got - (y - surf)).abs() < 1e-9, "({x},{z}): {got}");
    }
}

// ------------------------------------------------------------------ sample_normal

fn plane_normal(a: f64, b: f64) -> [f64; 3] {
    let n = (1.0 + a * a + b * b).sqrt();
    [-a / n, 1.0 / n, -b / n]
}

#[test]
fn normal_of_a_tilted_plane_matches_the_closed_form_away_from_the_border() {
    for &(a, b, sp) in &[(0.5, 0.0, 1.0), (0.0, -0.75, 0.5), (0.3, 0.2, 0.25)] {
        let origin = [1.0, 0.0, -2.0];
        let f = field_from(10, 10, sp, origin, |x, z| a * x + b * z);
        let got = arr(f.sample_normal(fx(origin[0] + 4.3 * sp), fx(origin[2] + 3.7 * sp)));
        let want = plane_normal(a, b);
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-8,
                "slope ({a},{b}) axis {i}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn normal_of_a_bilinear_saddle_is_exact_in_the_interior() {
    // h = x*z is reproduced exactly by bilinear interpolation, and the central
    // difference of a bilinear function is exact: grad = (z, x).
    let f = field_from(8, 8, 1.0, [0.0; 3], |x, z| 0.2 * x * z);
    for &(x, z) in &[(2.3, 3.1), (4.5, 2.5), (3.9, 5.2)] {
        let got = arr(f.sample_normal(fx(x), fx(z)));
        let want = plane_normal(0.2 * z, 0.2 * x);
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-8,
                "({x},{z}) axis {i}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn normal_is_unit_length() {
    let f = field_from(8, 8, 0.5, [0.0; 3], |x, z| 0.7 * x - 0.4 * z);
    let n = arr(f.sample_normal(fx(1.1), fx(1.3)));
    assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) - 1.0).abs() < 1e-9);
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-003: sample_normal at a grid border point of a plane returns half the slope (clamped outside sample halves the central difference): ramp 0.5, expected nx=-0.4472 got about -0.2425"]
fn normal_on_the_border_of_a_plane_is_the_plane_normal() {
    let f = field_from(8, 8, 1.0, [0.0; 3], |x, _| 0.5 * x);
    let got = arr(f.sample_normal(fx(0.0), fx(3.0)));
    let want = plane_normal(0.5, 0.0);
    assert!((got[0] - want[0]).abs() < 1e-6, "{got:?} vs {want:?}");
    let got = arr(f.sample_normal(fx(7.0), fx(3.0)));
    assert!((got[0] - want[0]).abs() < 1e-6, "{got:?} vs {want:?}");
}

#[test]
fn normal_of_a_zero_spacing_field_does_not_panic_and_points_up() {
    let f = HeightField::flat(4, 4, Fix128::ZERO, Vec3Fix::ZERO, Fix128::ZERO);
    let r = catch_unwind(AssertUnwindSafe(|| arr(f.sample_normal(fx(1.0), fx(1.0)))));
    let n = r.expect("sample_normal panicked on zero spacing");
    assert!(n[1] > 0.99, "{n:?}");
}

// ------------------------------------------------------------------ collide_sphere

#[test]
fn sphere_depth_is_radius_minus_the_tangent_plane_distance() {
    for &s in &[0.0, 0.5, 1.0, -0.75] {
        let f = field_from(14, 14, 1.0, [0.0; 3], |x, _| s * x);
        let norm = (1.0 + s * s).sqrt();
        let n = [-s / norm, 1.0 / norm, 0.0];
        let (cx, cz, r) = (6.0, 5.0, 0.5);
        for &dy in &[0.1, 0.25, 0.4, -0.3, -1.0] {
            let c = v3(cx, s * cx + dy, cz);
            let dist = dy / norm; // tangent-plane distance
            let got = f.collide_sphere(c, fx(r));
            if dist < r - 1e-6 {
                let k = got.unwrap_or_else(|| panic!("s={s} dy={dy}: no contact"));
                assert!(
                    (k.depth.to_f64() - (r - dist)).abs() < 1e-7,
                    "s={s} dy={dy}: depth {}",
                    k.depth.to_f64()
                );
                let kn = arr(k.normal);
                for i in 0..3 {
                    assert!((kn[i] - n[i]).abs() < 1e-7, "normal {kn:?} vs {n:?}");
                }
                let pa = arr(k.point_a);
                let want_pa = [cx - n[0] * r, s * cx + dy - n[1] * r, cz - n[2] * r];
                for i in 0..3 {
                    assert!(
                        (pa[i] - want_pa[i]).abs() < 1e-7,
                        "point_a {pa:?} vs {want_pa:?}"
                    );
                }
                // point_b lies on the surface and (b - a)·n equals the depth
                let pb = arr(k.point_b);
                assert!(
                    (pb[1] - s * pb[0]).abs() < 1e-7,
                    "point_b off the plane: {pb:?}"
                );
                let ba = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let along = ba[0] * n[0] + ba[1] * n[1] + ba[2] * n[2];
                assert!((along - (r - dist)).abs() < 1e-7);
            } else if dist > r + 1e-6 {
                assert!(got.is_none(), "s={s} dy={dy}: unexpected contact");
            }
        }
    }
}

#[test]
fn sphere_exactly_one_radius_from_the_surface_is_clear() {
    // doc: "clear of the surface when its centre is at least `radius` away"
    let f = HeightField::flat(8, 8, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    assert!(f.collide_sphere(v3(3.0, 1.0, 3.0), Fix128::ONE).is_none());
    assert!(f
        .collide_sphere(v3(3.0, 0.9375, 3.0), Fix128::ONE)
        .is_some());
}

#[test]
fn sphere_contact_normal_points_from_the_surface_to_the_sphere() {
    // Contact normal convention: from B (terrain) to A (sphere) = up on a flat field.
    let f = HeightField::flat(8, 8, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    let k = f.collide_sphere(v3(3.0, 0.5, 3.0), Fix128::ONE).unwrap();
    assert!((k.normal.y.to_f64() - 1.0).abs() < 1e-9);
    assert!((k.depth.to_f64() - 0.5).abs() < 1e-9);
}

#[test]
fn sphere_far_outside_the_grid_does_not_collide() {
    let f = HeightField::flat(8, 8, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    for &(x, z) in &[(-30.0, 3.0), (40.0, 3.0), (3.0, -30.0), (3.0, 40.0)] {
        assert!(
            f.collide_sphere(v3(x, 0.0, z), Fix128::ONE).is_none(),
            "({x},{z})"
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-004: collide_sphere margin is 2 cells before the origin but 3 cells past the last vertex (bound uses width, not width-1): a sphere 2.5 cells past the far edge collides, 2.5 cells before the near edge does not"]
fn lateral_margin_is_the_same_on_both_sides() {
    let f = HeightField::flat(8, 8, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    let near = f.collide_sphere(v3(-2.5, 0.0, 3.0), Fix128::ONE).is_some();
    let far = f
        .collide_sphere(v3(7.0 + 2.5, 0.0, 3.0), Fix128::ONE)
        .is_some();
    assert_eq!(near, far, "near side {near}, far side {far}");
    let near = f.collide_sphere(v3(3.0, 0.0, -2.5), Fix128::ONE).is_some();
    let far = f
        .collide_sphere(v3(3.0, 0.0, 7.0 + 2.5), Fix128::ONE)
        .is_some();
    assert_eq!(near, far, "z: near side {near}, far side {far}");
}

#[test]
fn degenerate_spacing_or_size_gives_no_contact() {
    let zero = HeightField::flat(4, 4, Fix128::ZERO, Vec3Fix::ZERO, Fix128::ZERO);
    assert!(zero
        .collide_sphere(v3(1.0, -5.0, 1.0), Fix128::ONE)
        .is_none());
    let neg = HeightField::flat(4, 4, fx(-1.0), Vec3Fix::ZERO, Fix128::ZERO);
    assert!(neg
        .collide_sphere(v3(1.0, -5.0, 1.0), Fix128::ONE)
        .is_none());
    let empty = HeightField::new(Vec::new(), 0, 0, Fix128::ONE, Vec3Fix::ZERO);
    assert!(empty
        .collide_sphere(v3(1.0, -5.0, 1.0), Fix128::ONE)
        .is_none());
}

// ------------------------------------------------------------------ aabb

#[test]
fn aabb_spans_the_grid_and_the_height_range() {
    let (w, d, sp) = (5u32, 3u32, 0.5);
    let origin = [-1.0, 0.0, 4.0];
    let mut s = 21u64;
    let hs: Vec<f64> = (0..w * d).map(|_| lcg(&mut s) * 4.0).collect();
    let f = HeightField::new(
        hs.iter().map(|&h| fx(h)).collect(),
        w,
        d,
        fx(sp),
        v3(origin[0], origin[1], origin[2]),
    );
    let bb = f.aabb();
    let lo = hs.iter().cloned().fold(f64::MAX, f64::min);
    let hi = hs.iter().cloned().fold(f64::MIN, f64::max);
    assert!((bb.min.x.to_f64() - origin[0]).abs() < 1e-12);
    assert!((bb.min.z.to_f64() - origin[2]).abs() < 1e-12);
    assert!((bb.max.x.to_f64() - (origin[0] + sp * 4.0)).abs() < 1e-12);
    assert!((bb.max.z.to_f64() - (origin[2] + sp * 2.0)).abs() < 1e-12);
    assert!((bb.min.y.to_f64() - lo).abs() < 1e-12);
    assert!((bb.max.y.to_f64() - hi).abs() < 1e-12);
}

#[test]
fn aabb_of_an_empty_field_does_not_panic() {
    let f = HeightField::new(Vec::new(), 0, 0, Fix128::ONE, Vec3Fix::ZERO);
    let r = catch_unwind(AssertUnwindSafe(|| f.aabb()));
    assert!(r.is_ok(), "aabb panicked on an empty field");
}

#[test]
fn aabb_of_a_single_vertex_field_is_a_point() {
    let f = HeightField::new(vec![fx(2.5)], 1, 1, fx(1.0), v3(1.0, 0.0, 2.0));
    let bb = f.aabb();
    assert_eq!(arr(bb.min), [1.0, 2.5, 2.0]);
    assert_eq!(arr(bb.max), [1.0, 2.5, 2.0]);
}

// ------------------------------------------------------------------ origin.y / spacing validation

#[test]
#[ignore = "known defect: AUD-A-S4W3-006: origin.y is never read (sample_height / signed_distance / aabb use the stored heights only), although the field doc calls origin the world-space min corner; origin.y = 10 over a zero-height grid still gives surface 0"]
fn origin_y_offsets_the_surface() {
    let f = HeightField::flat(4, 4, Fix128::ONE, v3(0.0, 10.0, 0.0), Fix128::ZERO);
    let h = f.sample_height(fx(1.5), fx(1.5)).to_f64();
    assert!((h - 10.0).abs() < 1e-9, "surface at {h}, expected 10");
}

#[test]
fn aabb_of_a_negative_spacing_field_is_not_inverted() {
    // AUD-A-S4W3-007: a spacing <= 0 has no surface (collide_sphere returns
    // None), so the box is the degenerate point at the origin, like an empty field
    for spacing in [-1.0, 0.0] {
        let origin = v3(2.0, 0.0, -3.0);
        let f = HeightField::flat(4, 4, fx(spacing), origin, Fix128::ONE);
        let bb = f.aabb();
        assert!(bb.min.x <= bb.max.x && bb.min.y <= bb.max.y && bb.min.z <= bb.max.z, "spacing {spacing}");
        assert_eq!((bb.min, bb.max), (origin, origin), "spacing {spacing}");
        assert!(f.collide_sphere(v3(3.0, 1.0, -2.0), Fix128::ONE).is_none());
    }
    // a positive spacing is unchanged: x and z span (width-1) and (depth-1) spacings
    let f = HeightField::flat(4, 3, fx(2.0), Vec3Fix::ZERO, Fix128::ONE);
    let bb = f.aabb();
    assert_eq!((bb.max.x.to_f64(), bb.max.z.to_f64(), bb.max.y.to_f64()), (6.0, 4.0, 1.0));
}

// ------------------------------------------------------------------ indexing

#[test]
fn heights_are_stored_x_major_within_a_row_and_rows_are_z() {
    // width 5, depth 3: vertex (gx, gz) is heights[gx + gz * width]
    let hs: Vec<Fix128> = (0..15).map(|i| fx(f64::from(i))).collect();
    let f = HeightField::new(hs, 5, 3, fx(1.0), Vec3Fix::ZERO);
    assert_eq!(f.get_height(4, 0).to_f64(), 4.0);
    assert_eq!(f.get_height(0, 1).to_f64(), 5.0);
    assert_eq!(f.get_height(3, 2).to_f64(), 13.0);
    // sampling agrees with the grid orientation (x along width, z along depth)
    assert!((f.sample_height(fx(3.0), fx(2.0)).to_f64() - 13.0).abs() < 1e-9);
    assert!((f.sample_height(fx(0.0), fx(1.0)).to_f64() - 5.0).abs() < 1e-9);
}

#[test]
fn sphere_resting_on_the_last_row_and_column_collides() {
    // Vertices on the far border (x = width-1, z = depth-1) are inside the field.
    let f = HeightField::flat(8, 6, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    assert!(f.collide_sphere(v3(7.0, 0.5, 2.0), Fix128::ONE).is_some());
    assert!(f.collide_sphere(v3(3.0, 0.5, 5.0), Fix128::ONE).is_some());
    assert!(f.collide_sphere(v3(0.0, 0.5, 0.0), Fix128::ONE).is_some());
    assert!(f.collide_sphere(v3(7.0, 0.5, 5.0), Fix128::ONE).is_some());
}

#[test]
fn near_side_margin_is_two_cells() {
    // Pins the current (undocumented) lateral margin on the origin side: a sphere
    // within 2 cells before the origin still sees the clamped edge, beyond 2 it does not.
    let f = HeightField::flat(8, 8, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
    assert!(f.collide_sphere(v3(-1.5, 0.0, 3.0), Fix128::ONE).is_some());
    assert!(f.collide_sphere(v3(-2.5, 0.0, 3.0), Fix128::ONE).is_none());
    assert!(f.collide_sphere(v3(3.0, 0.0, -1.5), Fix128::ONE).is_some());
    assert!(f.collide_sphere(v3(3.0, 0.0, -2.5), Fix128::ONE).is_none());
}

#[test]
fn sphere_contact_respects_a_non_zero_origin() {
    let f = HeightField::flat(8, 8, Fix128::ONE, v3(10.0, 0.0, 20.0), Fix128::ZERO);
    assert!(f.collide_sphere(v3(13.0, 0.5, 23.0), Fix128::ONE).is_some());
    assert!(f
        .collide_sphere(v3(10.0 - 1.5, 0.5, 23.0), Fix128::ONE)
        .is_some());
    assert!(f
        .collide_sphere(v3(10.0 - 2.5, 0.5, 23.0), Fix128::ONE)
        .is_none());
    assert!(f
        .collide_sphere(v3(13.0, 0.5, 20.0 - 1.5), Fix128::ONE)
        .is_some());
    assert!(f
        .collide_sphere(v3(13.0, 0.5, 20.0 - 2.5), Fix128::ONE)
        .is_none());
    // far from the field in world terms, near it in raw coordinates
    assert!(f.collide_sphere(v3(3.0, 0.5, 3.0), Fix128::ONE).is_none());
}

#[test]
fn sample_height_holds_the_edge_value_beyond_two_to_the_32_cells() {
    // 4294967297 = 2^32 + 1 cells from the origin: the grid coordinate must not wrap.
    let hs: Vec<Fix128> = (0..12).map(|i| fx(f64::from(i))).collect();
    let f = HeightField::new(hs, 4, 3, fx(1.0), Vec3Fix::ZERO);
    let want = reference(
        &(0..12).map(f64::from).collect::<Vec<_>>(),
        4,
        3,
        1.0,
        [0.0; 3],
        3.0,
        1.0,
    );
    let got = f.sample_height(fx(4_294_967_297.0), fx(1.0)).to_f64();
    assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    let want = reference(
        &(0..12).map(f64::from).collect::<Vec<_>>(),
        4,
        3,
        1.0,
        [0.0; 3],
        1.0,
        2.0,
    );
    let got = f.sample_height(fx(1.0), fx(4_294_967_297.0)).to_f64();
    assert!((got - want).abs() < 1e-9, "{got} vs {want}");
}

#[test]
fn normal_across_a_kink_uses_the_half_spacing_central_difference() {
    // Pins the stencil (undocumented): step = spacing / 2 on each side. Heights
    // h(x) = max(0, x - 3) on spacing 1; at x = 3.25 the samples are h(3.75) = 0.75
    // and h(2.75) = 0, so dh/dx = 0.75 / (2 * 0.5) = 0.75.
    let f = field_from(8, 8, 1.0, [0.0; 3], |x, _| (x - 3.0).max(0.0));
    let got = arr(f.sample_normal(fx(3.25), fx(3.0)));
    let want = plane_normal(0.75, 0.0);
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() < 1e-8,
            "axis {i}: {got:?} vs {want:?}"
        );
    }
}

#[test]
fn a_field_with_one_zero_dimension_has_no_surface_for_every_query() {
    for (w, d) in [(0u32, 3u32), (3, 0), (0, 0)] {
        let f = HeightField::new(Vec::new(), w, d, Fix128::ONE, Vec3Fix::ZERO);
        assert_eq!(f.get_height(1, 1), Fix128::ZERO, "get_height {w}x{d}");
        assert_eq!(
            f.sample_height(fx(0.5), fx(0.5)),
            Fix128::ZERO,
            "sample_height {w}x{d}"
        );
        assert!(
            f.collide_sphere(v3(0.5, -3.0, 0.5), Fix128::ONE).is_none(),
            "collide_sphere {w}x{d}"
        );
    }
}

#[test]
fn a_zero_spacing_field_reports_height_zero_instead_of_dividing() {
    // Pins the current contract: spacing 0 gives no interpolation (0), not the first vertex.
    let f = HeightField::new(
        vec![fx(5.0), fx(6.0), fx(7.0), fx(8.0)],
        2,
        2,
        Fix128::ZERO,
        Vec3Fix::ZERO,
    );
    assert_eq!(f.sample_height(fx(0.5), fx(0.5)), Fix128::ZERO);
    assert_eq!(f.sample_height(fx(100.0), fx(-3.0)), Fix128::ZERO);
}
