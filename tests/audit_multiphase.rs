//! Audit oracles for `alice_physics::multiphase`.
//! Expected values: hand-derived closed forms (translation of a slab, affine reproduction by
//! trilinear interpolation, Laplacian of a signed-distance sphere = 2/rho).
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{
    advect_vof_rigid, curvature_at, initialize_level_set_sphere, trilinear_range, trilinear_sample,
    Grid3d, VofScheme,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

#[test]
fn grid_layout_is_x_fastest_and_out_of_range_is_zero_without_aliasing() {
    let (nx, ny, nz) = (3usize, 4usize, 5usize);
    let mut g = Grid3d::new(nx, ny, nz, Fix128::ONE, Fix128::ZERO);
    assert_eq!(g.total(), 60);
    assert_eq!(g.data.len(), 60);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                g.set(i, j, k, int((100 * k + 10 * j + i) as i64));
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let flat = i + nx * (j + ny * k);
                assert_eq!(g.idx(i, j, k), flat);
                assert_eq!(g.data[flat], int((100 * k + 10 * j + i) as i64));
                assert_eq!(g.get(i, j, k), g.data[flat]);
            }
        }
    }
    // i == nx would alias (0, j+1, k) in flat storage: must still read 0 and write nothing
    let before = g.data.clone();
    assert_eq!(g.get(nx, 0, 0), Fix128::ZERO);
    assert_eq!(g.get(0, ny, 0), Fix128::ZERO);
    assert_eq!(g.get(0, 0, nz), Fix128::ZERO);
    g.set(nx, 0, 0, int(777));
    g.set(0, ny, 0, int(777));
    g.set(0, 0, nz, int(777));
    g.set(usize::MAX, 0, 0, int(777));
    assert_eq!(g.data, before);
}

/// Trilinear interpolation reproduces any affine field exactly at interior points.
#[test]
fn trilinear_sample_reproduces_an_affine_field() {
    let (a, b, c, d) = (0.5, 2.0, -3.0, 0.75);
    let mut g = Grid3d::new(5, 6, 4, Fix128::ONE, Fix128::ZERO);
    for k in 0..4 {
        for j in 0..6 {
            for i in 0..5 {
                g.set(i, j, k, fx(a + b * i as f64 + c * j as f64 + d * k as f64));
            }
        }
    }
    for &(x, y, z) in &[
        (0.0, 0.0, 0.0),
        (1.25, 2.5, 0.75),
        (3.9375, 4.0625, 2.5),
        (4.0, 5.0, 3.0),
        (2.5, 0.125, 1.875),
    ] {
        let got = trilinear_sample(&g, fx(x), fx(y), fx(z)).to_f64();
        let want = a + b * x + c * y + d * z;
        assert!((got - want).abs() < 1e-12, "({x},{y},{z}): {got} vs {want}");
    }
}

/// "Coordinates outside the grid clamp to boundary values": negative and beyond-the-end
/// coordinates give the nearest boundary sample, per axis independently.
#[test]
fn trilinear_sample_clamps_outside_coordinates_per_axis() {
    let mut g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::ZERO);
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                g.set(i, j, k, int(1 + i as i64 + 10 * j as i64 + 100 * k as i64));
            }
        }
    }
    let at = |x: f64, y: f64, z: f64| trilinear_sample(&g, fx(x), fx(y), fx(z)).to_f64();
    assert_eq!(at(-5.0, 0.0, 0.0), 1.0);
    assert_eq!(at(0.0, -5.0, 0.0), 1.0);
    assert_eq!(at(0.0, 0.0, -5.0), 1.0);
    assert_eq!(at(9.0, 0.0, 0.0), 3.0);
    assert_eq!(at(0.0, 9.0, 0.0), 21.0);
    assert_eq!(at(0.0, 0.0, 9.0), 201.0);
    assert_eq!(at(2.5, 2.5, 2.5), 223.0);
    assert_eq!(at(-1.0, 9.0, 1.5), 21.0 + 150.0);
}

/// The 8-corner range brackets the sample, equals the min/max of the surrounding corners,
/// and clamps like the sampler.
#[test]
fn trilinear_range_brackets_the_sample_and_matches_the_corner_extrema() {
    let mut g = Grid3d::new(4, 4, 4, Fix128::ONE, Fix128::ZERO);
    let val = |i: usize, j: usize, k: usize| ((i * 7 + j * 13 + k * 29) % 11) as f64 - 4.0;
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..4 {
                g.set(i, j, k, fx(val(i, j, k)));
            }
        }
    }
    for &(x, y, z) in &[
        (0.5, 0.5, 0.5),
        (1.25, 2.75, 0.125),
        (2.9, 1.1, 2.2),
        (-3.0, 0.5, 0.5),
        (8.0, 8.0, 8.0),
    ] {
        let (lo, hi) = trilinear_range(&g, fx(x), fx(y), fx(z));
        let s = trilinear_sample(&g, fx(x), fx(y), fx(z));
        assert!(
            lo <= s && s <= hi,
            "({x},{y},{z}) {:?} not in [{:?}, {:?}]",
            s.to_f64(),
            lo.to_f64(),
            hi.to_f64()
        );
        let cl = |v: f64, n: usize| (v.max(0.0).floor() as usize).min(n - 1);
        let (i0, j0, k0) = (cl(x, 4), cl(y, 4), cl(z, 4));
        let mut lo_w = f64::INFINITY;
        let mut hi_w = f64::NEG_INFINITY;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let v = val((i0 + di).min(3), (j0 + dj).min(3), (k0 + dk).min(3));
                    lo_w = lo_w.min(v);
                    hi_w = hi_w.max(v);
                }
            }
        }
        assert_eq!((lo.to_f64(), hi.to_f64()), (lo_w, hi_w), "({x},{y},{z})");
    }
}

/// Zero-extent grids: the sibling routines document "no cells, returns zero" and are guarded;
/// the public samplers subtract 1 from the extent (usize) and panic in debug builds.
#[test]
fn samplers_on_a_zero_extent_grid_return_zero_instead_of_panicking() {
    let g = Grid3d::new(0, 3, 3, Fix128::ONE, Fix128::ONE);
    let a = catch_unwind(AssertUnwindSafe(|| {
        trilinear_sample(&g, Fix128::ONE, Fix128::ONE, Fix128::ONE)
    }));
    assert_eq!(a.expect("trilinear_sample panicked"), Fix128::ZERO);
    let b = catch_unwind(AssertUnwindSafe(|| {
        trilinear_range(&g, Fix128::ONE, Fix128::ONE, Fix128::ONE)
    }));
    assert_eq!(
        b.expect("trilinear_range panicked"),
        (Fix128::ZERO, Fix128::ZERO)
    );
}

fn line(axis: usize, vals: &[f64], dx: Fix128) -> Grid3d {
    let n = vals.len();
    let (nx, ny, nz) = match axis {
        0 => (n, 1, 1),
        1 => (1, n, 1),
        _ => (1, 1, n),
    };
    let mut g = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    for (m, v) in vals.iter().enumerate() {
        g.data[m] = fx(*v);
    }
    g
}
fn along(axis: usize, u: f64) -> Vec3Fix {
    let mut c = [Fix128::ZERO; 3];
    c[axis] = fx(u);
    Vec3Fix::new(c[0], c[1], c[2])
}
fn values(g: &Grid3d) -> Vec<f64> {
    g.data.iter().map(|v| v.to_f64()).collect()
}

/// Doc of `advect_vof_rigid`: "a slab carried by u for dt = k dx / u lands k cells over with its
/// profile intact under either scheme". Semi-Lagrangian: yes. Upwind: only k = 1 (c = 1); at k = 2
/// the explicit update `2 f_up - f` shifts the slab by one cell and the clamp hides the rest.
#[test]
fn slab_translates_k_cells_under_semi_lagrangian_on_every_axis_and_sign() {
    let slab = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for axis in 0..3 {
        // dx = 1/2, u = +/-2 m/s, dt = k dx / u -> k cells; here k = 3 (positive), k = 1 (negative)
        let mut g = line(axis, &slab, fx(0.5));
        let vol = advect_vof_rigid(
            &mut g,
            VofScheme::SemiLagrangian,
            along(axis, 2.0),
            fx(0.75),
        );
        assert_eq!(
            values(&g),
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "axis {axis} +k=3"
        );
        assert_eq!(vol.to_f64(), 3.0 * 0.125);
        let mut g = line(axis, &slab, fx(0.5));
        let _ = advect_vof_rigid(
            &mut g,
            VofScheme::SemiLagrangian,
            along(axis, -2.0),
            fx(0.25),
        );
        assert_eq!(
            values(&g),
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "axis {axis} -k=1"
        );
    }
}

#[test]
// AUD-A-S2W3-008
fn upwind_slab_translates_k_cells_for_k_greater_than_one() {
    let slab = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut g = line(0, &slab, Fix128::ONE);
    // u = 2 m/s, dx = 1, dt = 1: k = 2 cells
    let _ = advect_vof_rigid(&mut g, VofScheme::Upwind, along(0, 2.0), Fix128::ONE);
    assert_eq!(
        values(&g),
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
}

/// Courant 1/4 on an interior pulse: both schemes give `0.75 f_i + 0.25 f_{i-1}` (positive u) /
/// `0.75 f_i + 0.25 f_{i+1}` (negative u), volume conserved while the pulse stays inside.
#[test]
fn quarter_courant_blend_and_volume_conservation_for_both_schemes() {
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        for axis in 0..3 {
            let pulse = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            let mut g = line(axis, &pulse, Fix128::ONE);
            let vol = advect_vof_rigid(&mut g, scheme, along(axis, 1.0), Fix128::from_ratio(1, 4));
            assert_eq!(
                values(&g),
                [0.0, 0.0, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0],
                "{scheme:?} axis {axis}"
            );
            assert_eq!(vol.to_f64(), 1.0);
            let mut g = line(axis, &pulse, Fix128::ONE);
            let vol = advect_vof_rigid(&mut g, scheme, along(axis, -1.0), Fix128::from_ratio(1, 4));
            assert_eq!(
                values(&g),
                [0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
                "{scheme:?} axis {axis}"
            );
            assert_eq!(vol.to_f64(), 1.0);
        }
    }
}

/// Far above the CFL limit the upwind step carries the field off the grid: a
/// displacement of 100 cells on a 6-cell line leaves it empty, the exact answer
/// with an empty inflow (AUD-A-S2W3-008; this test used to pin the clamp's
/// sign rule of a single step with c = 100). A displacement of exactly the
/// grid length (6 cells, c = 6, sub-cycled at c = 1) empties it too
#[test]
fn upwind_far_beyond_the_courant_limit_carries_the_field_off_the_grid() {
    for u in [100.0, 6.0] {
        let mut g = line(0, &[0.25, 0.5, 0.5, 1.0, 0.0, 0.5], Fix128::ONE);
        let vol = advect_vof_rigid(&mut g, VofScheme::Upwind, along(0, u), Fix128::ONE);
        assert_eq!(values(&g), [0.0; 6], "u = {u}");
        assert_eq!(vol, Fix128::ZERO);
    }
    // c = 2.5 (not an integer) on a long line: sub-cycled at 2.5 / 3, bounded and
    // volume-conserving while inside
    let start = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut g = line(0, &start, Fix128::ONE);
    let vol = advect_vof_rigid(&mut g, VofScheme::Upwind, along(0, 2.5), Fix128::ONE);
    assert!((vol.to_f64() - 2.0).abs() < 1e-12, "{}", vol.to_f64());
    // oracle: ceil(2.5) = 3 independent f64 upwind sub-steps at c = 2.5 / 3
    // (empty inflow), each f_i <- f_i - c (f_i - f_{i-1})
    let mut want = start;
    for _ in 0..3 {
        let prev = want;
        for i in 0..want.len() {
            let up = if i == 0 { 0.0 } else { prev[i - 1] };
            want[i] = prev[i] - 2.5 / 3.0 * (prev[i] - up);
        }
    }
    for (got, want) in values(&g).iter().zip(want) {
        assert!((got - want).abs() < 1e-12, "{:?} vs {want:?}", values(&g));
    }
}

/// Rigid convection never creates mass: the returned volume is `sum(f) dx^3` and an advected
/// field keeps every cell inside [0, 1] (checked on a non-trivial 3D field, Courant 1/2 per axis
/// stays below 1/3 total? use 1/8 each).
#[test]
fn returned_volume_matches_sum_and_values_stay_bounded_in_3d() {
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        let mut g = Grid3d::new(6, 5, 4, fx(0.25), Fix128::ZERO);
        for (m, v) in g.data.iter_mut().enumerate() {
            *v = Fix128::from_ratio((m % 5) as i64, 4);
        }
        let vol = advect_vof_rigid(
            &mut g,
            scheme,
            Vec3Fix::new(fx(0.5), fx(-0.5), fx(0.5)),
            Fix128::ONE,
        );
        let sum: f64 = values(&g).iter().sum();
        assert_eq!(vol.to_f64(), sum * 0.25 * 0.25 * 0.25);
        for v in values(&g) {
            assert!((0.0..=1.0).contains(&v), "{scheme:?}: {v}");
        }
    }
}

/// `initialize_level_set_sphere`: node (i,j,k) holds |x - c| - r with x = (i,j,k) dx; check all
/// nodes against an f64 evaluation on a scaled grid with an off-grid centre, plus sign structure.
#[test]
fn sphere_level_set_is_the_signed_distance_at_every_node() {
    let dx = 0.5;
    let mut g = Grid3d::new(9, 8, 7, fx(dx), Fix128::ZERO);
    let (cx, cy, cz, r) = (2.1, 1.7, 1.3, 1.25);
    initialize_level_set_sphere(&mut g, fx(cx), fx(cy), fx(cz), fx(r));
    for k in 0..7 {
        for j in 0..8 {
            for i in 0..9 {
                let (x, y, z) = (i as f64 * dx - cx, j as f64 * dx - cy, k as f64 * dx - cz);
                let want = (x * x + y * y + z * z).sqrt() - r;
                let got = g.get(i, j, k).to_f64();
                assert!((got - want).abs() < 1e-9, "({i},{j},{k}): {got} vs {want}");
                assert_eq!(got < 0.0, want < 0.0);
            }
        }
    }
}

fn sphere_grid(n: usize, dx: f64, r: f64) -> Grid3d {
    let mut g = Grid3d::new(n, n, n, fx(dx), Fix128::ZERO);
    let c = (n - 1) as f64 / 2.0 * dx;
    initialize_level_set_sphere(&mut g, fx(c), fx(c), fx(c), fx(r));
    g
}

/// Laplacian of a signed distance to a sphere is 2 / rho (rho = distance from the centre):
/// at the surface kappa = 2 / r, positive for the convex fluid A, in 1/m (scales with 1/dx^2
/// in the stencil: same physical sphere at two resolutions).
#[test]
fn curvature_of_a_sphere_is_two_over_radius_at_two_resolutions() {
    // (n, dx, r): physical radius 6 m sphere, then radius 3 m on a 0.5 m grid
    for (n, dx, r) in [(21usize, 1.0, 6.0), (17usize, 0.5, 3.0)] {
        let g = sphere_grid(n, dx, r);
        let c = (n - 1) / 2;
        let steps = (r / dx).round() as usize;
        let want = 2.0 / r;
        for (i, j, k) in [(c + steps, c, c), (c, c + steps, c), (c, c, c - steps)] {
            let got = curvature_at(&g, i, j, k).to_f64();
            assert!(
                ((got - want) / want).abs() < 0.02,
                "n {n} ({i},{j},{k}): {got} vs {want}"
            );
            assert!(got > 0.0);
        }
    }
}

/// Grid boundary, 2-D-thin grids and a plane: zero (no interior stencil / no curvature).
#[test]
fn curvature_is_zero_on_the_boundary_and_for_planar_fields() {
    let g = sphere_grid(9, 1.0, 2.0);
    for &(i, j, k) in &[
        (0, 4, 4),
        (4, 0, 4),
        (4, 4, 0),
        (8, 4, 4),
        (4, 8, 4),
        (4, 4, 8),
    ] {
        assert_eq!(curvature_at(&g, i, j, k), Fix128::ZERO);
    }
    assert_eq!(curvature_at(&g, 9, 4, 4), Fix128::ZERO);
    let mut p = Grid3d::new(5, 5, 5, fx(0.5), Fix128::ZERO);
    for k in 0..5 {
        for j in 0..5 {
            for i in 0..5 {
                p.set(i, j, k, fx(0.5 * (i as f64) - 1.0));
            }
        }
    }
    assert_eq!(curvature_at(&p, 2, 2, 2), Fix128::ZERO);
}

/// Doc: kappa is the Laplacian "scaled by 1/|grad phi|". The level-set curvature div(n) does
/// not depend on the scale of phi: 2 * (signed distance) is the same interface and must give
/// the same kappa. The implementation returns the bare Laplacian (doubles).
#[test]
// AUD-A-S2W3-009
fn curvature_is_invariant_under_scaling_of_phi() {
    let g = sphere_grid(21, 1.0, 6.0);
    let mut g2 = g.clone();
    for v in &mut g2.data {
        *v = *v + *v;
    }
    let a = curvature_at(&g, 16, 10, 10).to_f64();
    let b = curvature_at(&g2, 16, 10, 10).to_f64();
    assert!((a - b).abs() < 0.02 * a.abs(), "{a} vs {b}");
}

/// Upwind with the slab touching the inflow / outflow boundary on every axis and sign: the
/// neighbour cell next to the boundary (index 1 or n-2) must read the real boundary cell, not
/// the "outside = 0" fallback. Slab [1,1,1,0,0,0] at Courant 1 -> [0,1,1,1,0,0] (positive),
/// slab [0,0,0,1,1,1] -> [0,0,1,1,1,0] (negative).
#[test]
fn upwind_reads_the_boundary_neighbour_on_every_axis() {
    for axis in 0..3 {
        let mut g = line(axis, &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], Fix128::ONE);
        let _ = advect_vof_rigid(&mut g, VofScheme::Upwind, along(axis, 1.0), Fix128::ONE);
        assert_eq!(values(&g), [0.0, 1.0, 1.0, 1.0, 0.0, 0.0], "axis {axis} +");
        let mut g = line(axis, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], Fix128::ONE);
        let _ = advect_vof_rigid(&mut g, VofScheme::Upwind, along(axis, -1.0), Fix128::ONE);
        assert_eq!(values(&g), [0.0, 0.0, 1.0, 1.0, 1.0, 0.0], "axis {axis} -");
    }
}

/// Doc: both schemes clamp the result to [0, 1]. Semi-Lagrangian on a field with out-of-range
/// values (the sampler interpolates them) stays inside [0, 1].
#[test]
fn semi_lagrangian_clamps_out_of_range_samples() {
    let mut g = line(0, &[5.0, -3.0, 0.5, 7.0, -1.0, 2.0], Fix128::ONE);
    let _ = advect_vof_rigid(
        &mut g,
        VofScheme::SemiLagrangian,
        along(0, 0.5),
        Fix128::ONE,
    );
    for v in values(&g) {
        assert!((0.0..=1.0).contains(&v), "{v}");
    }
    // zero displacement with out-of-range values: clamp still applies (5 -> 1, -3 -> 0)
    let mut g = line(0, &[5.0, -3.0, 0.5], Fix128::ONE);
    let _ = advect_vof_rigid(
        &mut g,
        VofScheme::SemiLagrangian,
        along(0, 0.0),
        Fix128::ONE,
    );
    assert_eq!(values(&g), [1.0, 0.0, 0.5]);
}

/// A level set that is not a signed distance: phi = rho^2 - R^2 (|grad phi| = 2 rho).
/// The curvature of its zero set is still 2 / R; the bare Laplacian would give 6
/// (AUD-A-S2W3-009). Evaluated where the grid point lies exactly on the
/// surface, rho = R = 6 on a unit grid
#[test]
fn curvature_of_a_non_distance_level_set_is_two_over_radius() {
    let n = 21usize;
    let mut g = Grid3d::new(n, n, n, Fix128::ONE, Fix128::ZERO);
    let c = 10.0;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                g.set(i, j, k, fx(x * x + y * y + z * z - 36.0));
            }
        }
    }
    let kappa = curvature_at(&g, 16, 10, 10).to_f64();
    assert!((kappa - 2.0 / 6.0).abs() < 1e-9, "kappa = {kappa}");
    // off the axes (4, 4, 2), also on rho = 6: the mixed second derivatives count
    let kappa = curvature_at(&g, 14, 14, 12).to_f64();
    assert!((kappa - 2.0 / 6.0).abs() < 1e-9, "kappa off-axis = {kappa}");
}

/// Mixed second derivatives: phi = x y - 6 (a hyperbolic cylinder) has
/// grad = (y, x, 0) and only phi_xy = 1, so kappa = -2 x y / (x^2 + y^2)^(3/2);
/// at (2, 3) that is -12 / 13^(3/2). Central differences are exact for a
/// quadratic, so the stencil must reproduce it to rounding
#[test]
fn curvature_uses_the_mixed_second_derivatives() {
    let n = 9usize;
    let mut g = Grid3d::new(n, n, n, Fix128::ONE, Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                g.set(i, j, k, fx(i as f64 * j as f64 - 6.0));
            }
        }
    }
    let want = -12.0 / 13.0f64.powf(1.5);
    let kappa = curvature_at(&g, 2, 3, 4).to_f64();
    assert!((kappa - want).abs() < 1e-9, "kappa = {kappa} vs {want}");
}
