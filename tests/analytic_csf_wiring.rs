//! Closed-form oracles for the three previously-unwired surface-tension presets
//! (`SIGMA_PLA_AIR`, `SIGMA_MERCURY_AIR`, `SIGMA_STEEL_ARGON`) and for the CSF
//! machinery they feed (`csf_body_force`, `compute_csf_field`, `interface_normal`,
//! `smeared_delta`).
//!
//! Oracles (textbook, not read back from the implementation):
//!
//! ```text
//! Young-Laplace         dp = 2 sigma / R            (sphere)
//! CSF band integral     int f . n dn = sigma * kappa,   kappa = 2/R   (Brackbill 1992)
//! capillary length      l_c = sqrt(sigma / (rho g))     (Hg ~ 1.9 mm, water ~ 2.7 mm)
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{initialize_level_set_sphere, Grid3d};
use alice_physics::surface_tension_csf::{
    compute_csf_field, csf_body_force, interface_normal, smeared_delta, SIGMA_MERCURY_AIR,
    SIGMA_PLA_AIR, SIGMA_STEEL_ARGON, SIGMA_WATER_AIR,
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// sphere of radius 8 dx centred in a 24^3 grid, dx = 1/16 (R = 0.5 m)
fn sphere_grid() -> (Grid3d, Fix128, Fix128) {
    let n = 24usize;
    let dx = Fix128::from_ratio(1, 16);
    let mut phi = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    let c = Fix128::from_int(12) * dx;
    let radius = Fix128::from_int(8) * dx;
    initialize_level_set_sphere(&mut phi, c, c, c, radius);
    (phi, dx, radius)
}

/// signed band integral of `f . n_out` on the +x axis through the centre
fn band_integral(phi: &Grid3d, dx: Fix128, sigma: Fix128) -> f64 {
    let eps = dx.double();
    let mut jump = Fix128::ZERO;
    for i in 19..=21 {
        let f = csf_body_force(phi, i, 12, 12, sigma, eps);
        let n = interface_normal(phi, i, 12, 12);
        jump = jump + (f.x * n.x + f.y * n.y + f.z * n.z) * dx;
    }
    jump.to_f64()
}

// ---------------------------------------------------------------- presets

#[test]
fn presets_equal_their_documented_values() {
    // doc comments: water 0.072, PLA melt 0.030, mercury 0.485, molten steel 1.6 (N/m)
    assert!((SIGMA_WATER_AIR.to_f64() - 0.072).abs() < 1e-15);
    assert!((SIGMA_PLA_AIR.to_f64() - 0.030).abs() < 1e-15);
    assert!((SIGMA_MERCURY_AIR.to_f64() - 0.485).abs() < 1e-15);
    assert!((SIGMA_STEEL_ARGON.to_f64() - 1.6).abs() < 1e-15);
}

#[test]
fn preset_ordering_and_physical_magnitude() {
    // PLA melt < water < mercury < molten steel (CRC Handbook / Keene 1988 surveys)
    assert!(SIGMA_PLA_AIR < SIGMA_WATER_AIR);
    assert!(SIGMA_WATER_AIR < SIGMA_MERCURY_AIR);
    assert!(SIGMA_MERCURY_AIR < SIGMA_STEEL_ARGON);
    // capillary length sqrt(sigma / (rho g)): mercury 13534 kg/m^3 -> ~1.9 mm,
    // water 998 kg/m^3 -> ~2.7 mm (independent physical reference values)
    let g = 9.81;
    let lc_hg = (SIGMA_MERCURY_AIR.to_f64() / (13_534.0 * g)).sqrt();
    let lc_w = (SIGMA_WATER_AIR.to_f64() / (998.0 * g)).sqrt();
    assert!(rel(lc_hg, 1.9e-3) < 0.03, "Hg l_c = {lc_hg}");
    assert!(rel(lc_w, 2.7e-3) < 0.03, "water l_c = {lc_w}");
    // molten steel (~7000 kg/m^3): ~ 4.8 mm
    let lc_fe = (SIGMA_STEEL_ARGON.to_f64() / (7_000.0 * g)).sqrt();
    assert!(rel(lc_fe, 4.8e-3) < 0.03, "Fe l_c = {lc_fe}");
}

#[test]
fn young_laplace_band_integral_scales_linearly_with_each_preset() {
    let (phi, dx, radius) = sphere_grid();
    let r = radius.to_f64();
    let w = band_integral(&phi, dx, SIGMA_WATER_AIR);
    for (name, s) in [
        ("PLA", SIGMA_PLA_AIR),
        ("Hg", SIGMA_MERCURY_AIR),
        ("steel", SIGMA_STEEL_ARGON),
    ] {
        let j = band_integral(&phi, dx, s);
        // signed: surface tension pulls inward, so int f . n_out dn = -sigma kappa = -2 sigma / R
        // (2 % discretisation, see band weights 1.008)
        let want = -2.0 * s.to_f64() / r;
        assert!(rel(j, want) < 0.02, "{name}: {j} vs {want}");
        // linearity in sigma, exactly the preset ratio
        assert!(
            rel(j / w, s.to_f64() / SIGMA_WATER_AIR.to_f64()) < 1e-12,
            "{name}: ratio {}",
            j / w
        );
    }
}

#[test]
fn csf_force_is_radial_and_zero_outside_the_band() {
    let (phi, dx, _) = sphere_grid();
    let eps = dx.double();
    // oblique cell on the interface band: offset (6, 4, 2) cells from the centre
    let f = csf_body_force(&phi, 12 + 6, 12 + 4, 12 + 2, SIGMA_MERCURY_AIR, eps);
    let len = (f.x * f.x + f.y * f.y + f.z * f.z).sqrt().to_f64();
    assert!(len > 1.0, "the cell must be inside the band, |f| = {len}");
    let r = (6.0f64 * 6.0 + 4.0 * 4.0 + 2.0 * 2.0).sqrt();
    // the force is parallel to the radius vector (either sign), all three components
    for (c, o) in [(f.x, 6.0), (f.y, 4.0), (f.z, 2.0)] {
        let cosine = c.to_f64() / len;
        assert!(
            (cosine.abs() - o / r).abs() < 2e-2,
            "component {cosine} vs {}",
            o / r
        );
    }
    // far inside and far outside the 2 dx band
    assert_eq!(
        csf_body_force(&phi, 12, 12, 12, SIGMA_MERCURY_AIR, eps),
        Vec3Fix::ZERO
    );
    assert_eq!(
        csf_body_force(&phi, 22, 12, 12, SIGMA_MERCURY_AIR, eps),
        Vec3Fix::ZERO
    );
}

#[test]
fn presets_are_within_a_few_ulp_of_the_decimal_values() {
    let ulp4 = Fix128::from_raw(0, 4);
    for (s, num, den) in [
        (SIGMA_WATER_AIR, 72, 1000),
        (SIGMA_PLA_AIR, 30, 1000),
        (SIGMA_MERCURY_AIR, 485, 1000),
    ] {
        assert!(
            (s - Fix128::from_ratio(num, den)).abs() <= ulp4,
            "{}/{}",
            num,
            den
        );
    }
    // steel is quantised through an f64 literal (documented): within 1e-16
    assert!(
        (SIGMA_STEEL_ARGON - Fix128::from_ratio(16, 10)).abs()
            < Fix128::from_ratio(1, 10_000_000_000_000_000)
    );
}

#[test]
fn normal_stencil_reaches_the_last_interior_cell_on_every_axis() {
    // phi = x + 2y + 3z: every interior cell, including index n-2, has a stencil
    let n = 5usize;
    let mut phi = Grid3d::new(n, n, n, Fix128::ONE, Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                phi.set(i, j, k, Fix128::from_int((i + 2 * j + 3 * k) as i64));
            }
        }
    }
    let len = 14.0f64.sqrt();
    for (i, j, k) in [(3, 2, 2), (2, 3, 2), (2, 2, 3), (1, 1, 1), (3, 3, 3)] {
        let nrm = interface_normal(&phi, i, j, k);
        assert!(rel(nrm.x.to_f64(), 1.0 / len) < 1e-12, "{i},{j},{k}");
        assert!(rel(nrm.y.to_f64(), 2.0 / len) < 1e-12);
        assert!(rel(nrm.z.to_f64(), 3.0 / len) < 1e-12);
    }
}

/// Surface tension pulls toward the centre of curvature: on the +x side of a
/// droplet the force points to -x (and the mirror on the -x side).
#[test]
fn csf_force_points_toward_the_centre_of_curvature() {
    let (phi, dx, _) = sphere_grid();
    let eps = dx.double();
    let right = csf_body_force(&phi, 20, 12, 12, SIGMA_MERCURY_AIR, eps);
    let left = csf_body_force(&phi, 4, 12, 12, SIGMA_MERCURY_AIR, eps);
    assert!(right.x < Fix128::ZERO, "+x side: {}", right.x.to_f64());
    assert!(left.x > Fix128::ZERO, "-x side: {}", left.x.to_f64());
    // oblique cell: f . (cell - centre) < 0
    let f = csf_body_force(&phi, 18, 16, 14, SIGMA_MERCURY_AIR, eps);
    assert!(
        f.x * Fix128::from_int(6) + f.y * Fix128::from_int(4) + f.z * Fix128::from_int(2)
            < Fix128::ZERO
    );
}

/// Young-Laplace through a pressure solve: for a static droplet `grad p = f`, so
/// `lap p = div f` with `p_out = 0` far away; the solution must give
/// `p_in - p_out = 2 sigma / R` (de Gennes et al. 2004 s.1.1). The Poisson solve
/// (7-point SOR, Dirichlet 0 on the box faces) lives in this test in f64 and
/// shares no code with the crate; only `compute_csf_field` is the input.
#[test]
fn young_laplace_pressure_jump_from_a_poisson_solve_of_the_csf_force() {
    let (phi, dx, radius) = sphere_grid();
    let n = 24usize;
    let sigma = SIGMA_MERCURY_AIR;
    let (fx, fy, fz) = compute_csf_field(&phi, sigma, dx.double());
    let h = dx.to_f64();
    let idx = |i: usize, j: usize, k: usize| i + n * (j + n * k);
    let mut rhs = vec![0.0f64; n * n * n];
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                rhs[idx(i, j, k)] = (fx[idx(i + 1, j, k)].to_f64() - fx[idx(i - 1, j, k)].to_f64()
                    + fy[idx(i, j + 1, k)].to_f64()
                    - fy[idx(i, j - 1, k)].to_f64()
                    + fz[idx(i, j, k + 1)].to_f64()
                    - fz[idx(i, j, k - 1)].to_f64())
                    / (2.0 * h);
            }
        }
    }
    let mut p = vec![0.0f64; n * n * n];
    let omega = 1.85;
    for _ in 0..1500 {
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let nb = p[idx(i + 1, j, k)]
                        + p[idx(i - 1, j, k)]
                        + p[idx(i, j + 1, k)]
                        + p[idx(i, j - 1, k)]
                        + p[idx(i, j, k + 1)]
                        + p[idx(i, j, k - 1)];
                    let gs = (nb - h * h * rhs[idx(i, j, k)]) / 6.0;
                    let c = idx(i, j, k);
                    p[c] += omega * (gs - p[c]);
                }
            }
        }
    }
    let p_in = p[idx(12, 12, 12)];
    let p_out = p[idx(2, 2, 2)];
    let want = 2.0 * sigma.to_f64() / radius.to_f64();
    assert!(
        rel(p_in - p_out, want) < 0.08,
        "p_in - p_out = {} vs 2 sigma / R = {want}",
        p_in - p_out
    );
}

// ------------------------------------------------------------ field wrap

#[test]
fn compute_csf_field_matches_cellwise_force_on_a_non_cubic_grid() {
    // nx != ny != nz so a swapped stride shows up
    let (nx, ny, nz) = (14usize, 10usize, 12usize);
    let dx = Fix128::from_ratio(1, 8);
    let mut phi = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    initialize_level_set_sphere(
        &mut phi,
        Fix128::from_int(7) * dx,
        Fix128::from_int(5) * dx,
        Fix128::from_int(6) * dx,
        Fix128::from_int(3) * dx,
    );
    let eps = dx + dx;
    let (fx, fy, fz) = compute_csf_field(&phi, SIGMA_PLA_AIR, eps);
    assert_eq!(fx.len(), nx * ny * nz);
    assert_eq!(fy.len(), nx * ny * nz);
    assert_eq!(fz.len(), nx * ny * nz);
    let mut nonzero = 0;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let f = csf_body_force(&phi, i, j, k, SIGMA_PLA_AIR, eps);
                let idx = i + nx * (j + ny * k);
                assert_eq!((fx[idx], fy[idx], fz[idx]), (f.x, f.y, f.z), "{i},{j},{k}");
                if f != Vec3Fix::ZERO {
                    nonzero += 1;
                }
            }
        }
    }
    assert!(nonzero > 20, "the band must be populated, got {nonzero}");
}

#[test]
fn flat_interface_carries_no_force() {
    // phi = x - 8 dx: kappa = 0 everywhere, so f = 0 although the band is populated
    let (nx, ny, nz) = (16usize, 8usize, 8usize);
    let dx = Fix128::from_ratio(1, 4);
    let mut phi = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                phi.set(
                    i,
                    j,
                    k,
                    (Fix128::from_int(i as i64) - Fix128::from_int(8)) * dx,
                );
            }
        }
    }
    assert_ne!(smeared_delta(Fix128::ZERO, dx.double()), Fix128::ZERO);
    let (fx, fy, fz) = compute_csf_field(&phi, SIGMA_STEEL_ARGON, dx.double());
    assert!(fx
        .iter()
        .chain(fy.iter())
        .chain(fz.iter())
        .all(|v| v.is_zero()));
    // the normal of that plane is exactly +x
    assert_eq!(
        interface_normal(&phi, 8, 4, 4),
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
}

// ---------------------------------------------------------------- delta

#[test]
fn smeared_delta_hat_values_and_degenerate_epsilon() {
    let eps = Fix128::from_int(2);
    // (1 - |phi|/eps)/eps : phi = 0 -> 1/2; phi = 1 -> 1/4; phi = -1 -> 1/4; |phi| >= eps -> 0
    assert_eq!(smeared_delta(Fix128::ZERO, eps), Fix128::from_ratio(1, 2));
    assert_eq!(smeared_delta(Fix128::ONE, eps), Fix128::from_ratio(1, 4));
    assert_eq!(
        smeared_delta(Fix128::NEG_ONE, eps),
        Fix128::from_ratio(1, 4)
    );
    assert_eq!(smeared_delta(eps, eps), Fix128::ZERO);
    assert_eq!(smeared_delta(Fix128::from_int(-3), eps), Fix128::ZERO);
    // degenerate widths
    assert_eq!(smeared_delta(Fix128::ZERO, Fix128::ZERO), Fix128::ZERO);
    assert_eq!(
        smeared_delta(Fix128::ZERO, Fix128::from_int(-2)),
        Fix128::ZERO,
        "a negative width is treated as an empty band"
    );
}

#[test]
fn interface_normal_degenerate_inputs_return_the_zero_vector() {
    let phi = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::from_int(3));
    // uniform field: zero gradient
    assert_eq!(interface_normal(&phi, 2, 2, 2), Vec3Fix::default());
    // each of the six faces is "no stencil"
    for (i, j, k) in [
        (0, 2, 2),
        (4, 2, 2),
        (2, 0, 2),
        (2, 4, 2),
        (2, 2, 0),
        (2, 2, 4),
    ] {
        assert_eq!(
            interface_normal(&phi, i, j, k),
            Vec3Fix::default(),
            "{i},{j},{k}"
        );
    }
    // zero spacing
    let mut z = Grid3d::new(5, 5, 5, Fix128::ZERO, Fix128::ZERO);
    z.set(3, 2, 2, Fix128::ONE);
    assert_eq!(interface_normal(&z, 2, 2, 2), Vec3Fix::default());
}

#[test]
fn normal_of_a_tilted_plane_matches_the_analytic_unit_vector() {
    // phi = 3 x + 4 y (gradient (3,4,0), |grad| = 5) -> normal (0.6, 0.8, 0)
    let mut phi = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::ZERO);
    for k in 0..5 {
        for j in 0..5 {
            for i in 0..5 {
                phi.set(
                    i,
                    j,
                    k,
                    Fix128::from_int(3 * i as i64) + Fix128::from_int(4 * j as i64),
                );
            }
        }
    }
    let n = interface_normal(&phi, 2, 2, 2);
    assert!(rel(n.x.to_f64(), 0.6) < 1e-12);
    assert!(rel(n.y.to_f64(), 0.8) < 1e-12);
    assert_eq!(n.z, Fix128::ZERO);
}
