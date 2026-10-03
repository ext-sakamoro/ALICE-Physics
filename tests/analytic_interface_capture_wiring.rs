//! Oracles for the PLIC helpers of `interface_capture` (`plic_normal`,
//! `plic_plane_offset`, `truncated_cube_volume`).
//!
//! The module's own claim: "place a plane `n.x = d` such that the volume of fluid on
//! the negative side matches `f dx^3`". The oracles are:
//!
//! * a closed-form unit-cube cut volume for axis-aligned and 2-D normals
//!   (`V = clamp(d/dx + 1/2, 0, 1) dx^3`, trapezoid areas),
//! * an independent f64 quadrature `V(beta) = int int clamp((beta - a1 u1 - a2 u2)/a3, 0, 1)`,
//! * the round trip `V(plane_offset(n, f)) = f dx^3`,
//! * `plic_normal` of a linear ramp = `-grad f / |grad f|` exactly, and of a
//!   sphere-shaped VOF field ~ the radial unit vector.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::interface_capture::{plic_normal, plic_plane_offset, truncated_cube_volume};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::Grid3d;

fn unit(x: f64, y: f64, z: f64) -> Vec3Fix {
    let l = (x * x + y * y + z * z).sqrt();
    Vec3Fix::new(
        Fix128::from_f64(x / l),
        Fix128::from_f64(y / l),
        Fix128::from_f64(z / l),
    )
}

fn normals() -> Vec<(&'static str, Vec3Fix)> {
    vec![
        ("+x", unit(1.0, 0.0, 0.0)),
        ("-y", unit(0.0, -1.0, 0.0)),
        ("xy", unit(1.0, 1.0, 0.0)),
        ("diag", unit(1.0, 1.0, 1.0)),
        ("oblique", unit(0.3, -0.5, 0.8)),
        ("steep", unit(0.05, 0.1, 1.0)),
        ("mixed", unit(-0.7, 0.2, 0.4)),
    ]
}

/// independent f64 quadrature of the cut volume fraction of the unit cube,
/// `{ sum a_i u_i <= beta }` with all `a_i > 0`
fn quad_fraction(a: [f64; 3], beta: f64) -> f64 {
    let n = 1500usize;
    let h = 1.0 / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let u1 = (i as f64 + 0.5) * h;
        for j in 0..n {
            let u2 = (j as f64 + 0.5) * h;
            sum += ((beta - a[0] * u1 - a[1] * u2) / a[2]).clamp(0.0, 1.0);
        }
    }
    sum * h * h
}

// -------------------------------------------------------- cut volume

#[test]
fn axis_aligned_cut_is_linear_in_the_offset() {
    let dx = Fix128::from_int(2);
    let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    // V = dx^3 clamp(d/dx + 1/2, 0, 1); dx = 2, dx^3 = 8
    for (d, frac) in [
        (-2.0, 0.0),
        (-1.0, 0.0),
        (-0.5, 0.25),
        (0.0, 0.5),
        (0.5, 0.75),
        (1.0, 1.0),
        (3.0, 1.0),
    ] {
        let v = truncated_cube_volume(n, Fix128::from_f64(d), dx).to_f64();
        assert!(
            (v - 8.0 * frac).abs() < 1e-9,
            "d = {d}: {v} vs {}",
            8.0 * frac
        );
    }
}

#[test]
fn diagonal_2d_cut_matches_the_trapezoid_area() {
    // n = (1,1,0)/sqrt2 on the unit cube: area of {u1 + u2 <= s}, s = d sqrt2 + 1
    //   s <= 1: s^2/2,  1 <= s <= 2: 1 - (2-s)^2/2
    let dx = Fix128::ONE;
    let n = unit(1.0, 1.0, 0.0);
    for k in -8..=8 {
        let d = f64::from(k) * 0.1;
        let s = d * 2f64.sqrt() + 1.0;
        let want = if s <= 0.0 {
            0.0
        } else if s <= 1.0 {
            0.5 * s * s
        } else if s <= 2.0 {
            1.0 - 0.5 * (2.0 - s) * (2.0 - s)
        } else {
            1.0
        };
        let v = truncated_cube_volume(n, Fix128::from_f64(d), dx).to_f64();
        assert!((v - want).abs() < 1e-9, "d = {d}: {v} vs {want}");
    }
}

#[test]
fn oblique_cut_matches_independent_quadrature() {
    let dx = Fix128::from_ratio(1, 2); // dx^3 = 1/8
    for (name, n) in [
        ("oblique", unit(0.3, 0.5, 0.8)),
        ("diag", unit(1.0, 1.0, 1.0)),
    ] {
        let a = [n.x.to_f64(), n.y.to_f64(), n.z.to_f64()];
        let half_sum = 0.5 * (a[0] + a[1] + a[2]);
        for k in -6..=6 {
            let d_over_dx = f64::from(k) * 0.12;
            let d = d_over_dx * 0.5;
            let want = quad_fraction(a, d_over_dx + half_sum) * 0.125;
            let got = truncated_cube_volume(n, Fix128::from_f64(d), dx).to_f64();
            assert!((got - want).abs() < 2e-5, "{name} k={k}: {got} vs {want}");
        }
    }
}

#[test]
fn cut_volume_is_monotone_in_d_and_symmetric_under_point_reflection() {
    let dx = Fix128::ONE;
    for (name, n) in normals() {
        let mut last = -1.0;
        for k in -20..=20 {
            let d = Fix128::from_ratio(k, 10);
            let v = truncated_cube_volume(n, d, dx).to_f64();
            assert!(v >= last - 1e-12, "{name}: not monotone at d = {k}/10");
            last = v;
            // V(d) + V(-d) = 1 (x -> -x maps the cube to itself)
            let w = truncated_cube_volume(n, Fix128::ZERO - d, dx).to_f64();
            assert!((v + w - 1.0).abs() < 1e-9, "{name} d = {k}/10: {v} + {w}");
        }
        assert!(
            last > 1.0 - 1e-12 && last <= 1.0 + 1e-12,
            "{name}: full cube at d = 2"
        );
    }
}

#[test]
fn zero_normal_cuts_all_or_nothing() {
    let z = Vec3Fix::ZERO;
    assert_eq!(
        truncated_cube_volume(z, Fix128::ONE, Fix128::ONE),
        Fix128::ONE
    );
    assert_eq!(
        truncated_cube_volume(z, Fix128::ZERO, Fix128::ONE),
        Fix128::ONE
    );
    assert_eq!(
        truncated_cube_volume(z, Fix128::NEG_ONE, Fix128::ONE),
        Fix128::ZERO
    );
}

// ----------------------------------------------------- offset round trip

#[test]
fn plane_offset_round_trips_to_the_requested_volume() {
    for dx in [Fix128::ONE, Fix128::from_ratio(1, 4)] {
        let dx3 = (dx * dx * dx).to_f64();
        for (name, n) in normals() {
            for f in [
                0.001, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 0.999,
            ] {
                let d = plic_plane_offset(n, Fix128::from_f64(f), dx);
                let v = truncated_cube_volume(n, d, dx).to_f64() / dx3;
                assert!(
                    (v - f).abs() < 1e-6,
                    "{name} dx = {} f = {f}: recovered volume fraction {v}",
                    dx.to_f64()
                );
            }
        }
    }
}

#[test]
fn plane_offset_is_antisymmetric_monotone_and_scales_with_dx() {
    for (name, n) in normals() {
        let mut last = f64::NEG_INFINITY;
        for k in 1..20 {
            let f = f64::from(k) / 20.0;
            let d = plic_plane_offset(n, Fix128::from_f64(f), Fix128::ONE).to_f64();
            let dm = plic_plane_offset(n, Fix128::from_f64(1.0 - f), Fix128::ONE).to_f64();
            assert!((d + dm).abs() < 1e-6, "{name} f = {f}: {d} vs {dm}");
            assert!(d >= last - 1e-9, "{name}: not monotone at f = {f}");
            last = d;
            // dimension: d is a length
            let d3 = plic_plane_offset(n, Fix128::from_f64(f), Fix128::from_int(3)).to_f64();
            assert!(
                (d3 - 3.0 * d).abs() < 1e-6,
                "{name} f = {f}: {d3} vs 3 * {d}"
            );
        }
    }
}

#[test]
fn half_full_cell_has_its_plane_through_the_centre() {
    for (name, n) in normals() {
        let d = plic_plane_offset(n, Fix128::from_ratio(1, 2), Fix128::ONE).to_f64();
        assert!(d.abs() < 1e-6, "{name}: {d}");
    }
}

#[test]
fn empty_and_full_cells_put_the_plane_outside_the_cube() {
    for (name, n) in normals() {
        let empty = plic_plane_offset(n, Fix128::ZERO, Fix128::ONE);
        let full = plic_plane_offset(n, Fix128::ONE, Fix128::ONE);
        assert_eq!(
            truncated_cube_volume(n, empty, Fix128::ONE),
            Fix128::ZERO,
            "{name}"
        );
        assert_eq!(
            truncated_cube_volume(n, full, Fix128::ONE),
            Fix128::ONE,
            "{name}"
        );
        // out-of-range fractions behave like the clamped ends
        assert_eq!(
            plic_plane_offset(n, Fix128::from_int(-3), Fix128::ONE),
            empty
        );
        assert_eq!(plic_plane_offset(n, Fix128::from_int(5), Fix128::ONE), full);
    }
}

#[test]
fn corner_tetrahedron_regime_matches_the_closed_form() {
    // n = (1,1,1)/sqrt3, small f: the plane cuts a corner tetrahedron of legs L:
    // V = L^3 / 6 (unit cube, legs along the axes), and the plane distance from the
    // corner is L / sqrt3, so d = -(sqrt3/2) + L/sqrt3 with L = cbrt(6 f).
    let n = unit(1.0, 1.0, 1.0);
    for f in [1e-4f64, 1e-3, 1e-2, 0.03] {
        let l = (6.0 * f).cbrt();
        let want = -(3f64.sqrt() / 2.0) + l / 3f64.sqrt();
        let d = plic_plane_offset(n, Fix128::from_f64(f), Fix128::ONE).to_f64();
        assert!((d - want).abs() < 1e-6, "f = {f}: {d} vs {want}");
    }
}

// ------------------------------------------------------------ normal

fn ramp(a: f64, b: f64, c: f64) -> Grid3d {
    let n = 7;
    let mut g = Grid3d::new(n, n, n, Fix128::from_ratio(1, 2), Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = i as f64 * 0.5;
                let y = j as f64 * 0.5;
                let z = k as f64 * 0.5;
                g.set(i, j, k, Fix128::from_f64(0.5 + a * x + b * y + c * z));
            }
        }
    }
    g
}

#[test]
fn normal_of_a_linear_ramp_is_minus_the_gradient_direction() {
    for (a, b, c) in [
        (0.1, 0.0, 0.0),
        (0.0, -0.05, 0.0),
        (0.03, 0.04, 0.0),
        (0.02, -0.04, 0.06),
    ] {
        let n = plic_normal(&ramp(a, b, c), 3, 3, 3);
        let l = (a * a + b * b + c * c).sqrt();
        assert!((n.x.to_f64() + a / l).abs() < 1e-9, "{a},{b},{c}");
        assert!((n.y.to_f64() + b / l).abs() < 1e-9);
        assert!((n.z.to_f64() + c / l).abs() < 1e-9);
        let len = n.length_squared().to_f64();
        assert!((len - 1.0).abs() < 1e-9, "unit length: {len}");
    }
}

#[test]
fn normal_degenerate_inputs_return_the_zero_vector() {
    let g = ramp(0.1, 0.1, 0.1);
    for (i, j, k) in [
        (0, 3, 3),
        (6, 3, 3),
        (3, 0, 3),
        (3, 6, 3),
        (3, 3, 0),
        (3, 3, 6),
    ] {
        assert_eq!(plic_normal(&g, i, j, k), Vec3Fix::default(), "{i},{j},{k}");
    }
    // the last interior cell has a stencil
    assert_ne!(plic_normal(&g, 5, 5, 5), Vec3Fix::default());
    assert_ne!(plic_normal(&g, 1, 1, 1), Vec3Fix::default());
    // uniform field
    let u = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::from_ratio(1, 2));
    assert_eq!(plic_normal(&u, 2, 2, 2), Vec3Fix::default());
    // zero spacing
    let z = Grid3d::new(5, 5, 5, Fix128::ZERO, Fix128::ZERO);
    assert_eq!(plic_normal(&z, 2, 2, 2), Vec3Fix::default());
}

#[test]
fn sphere_vof_normal_points_from_fluid_to_gas_along_the_radius() {
    // VOF of a ball: f = clamp(1/2 - phi/dx, 0, 1) with phi = |x - c| - R (fluid inside)
    let n = 21usize;
    let dx = 0.25f64;
    let c = 10.0 * dx;
    let r = 6.0 * dx;
    let mut g = Grid3d::new(n, n, n, Fix128::from_f64(dx), Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64 * dx - c, j as f64 * dx - c, k as f64 * dx - c);
                let phi = (x * x + y * y + z * z).sqrt() - r;
                g.set(i, j, k, Fix128::from_f64((0.5 - phi / dx).clamp(0.0, 1.0)));
            }
        }
    }
    for (i, j, k) in [
        (16, 10, 10),
        (4, 10, 10),
        (10, 16, 10),
        (10, 10, 4),
        (14, 14, 10),
        (13, 13, 13),
    ] {
        let nrm = plic_normal(&g, i, j, k);
        let (x, y, z) = (i as f64 * dx - c, j as f64 * dx - c, k as f64 * dx - c);
        let l = (x * x + y * y + z * z).sqrt();
        let dot = nrm.x.to_f64() * x / l + nrm.y.to_f64() * y / l + nrm.z.to_f64() * z / l;
        assert!(dot > 0.97, "cell {i},{j},{k}: n . r_hat = {dot}");
    }
}
