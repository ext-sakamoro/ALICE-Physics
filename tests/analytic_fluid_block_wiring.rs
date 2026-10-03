//! Oracles for `Fluid::new_block` (previously unwired): the lattice it generates.
//!
//! Closed form: a block `[min, max]` at spacing `s` holds `prod_axes (floor((max - min)/s) + 1)`
//! particles (a point on `max` is included), laid out x-major (x outer, z inner).
//! For dyadic `s` every coordinate is exactly representable, so the checks are exact.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::fluid::{Fluid, FluidConfig};
use alice_physics::math::{Fix128, Vec3Fix};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn block(min: Vec3Fix, max: Vec3Fix, s: Fix128) -> Fluid {
    Fluid::new_block(min, max, s, FluidConfig::default())
}

#[test]
fn particle_count_is_the_product_of_per_axis_counts() {
    // [0,1]^3 at 1/4: 5 per axis
    assert_eq!(
        block(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1), q(1, 4)).particle_count(),
        125
    );
    // unequal extents: x 5 points, y 3, z 2
    let f = block(
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::ONE, q(1, 2), q(1, 4)),
        q(1, 4),
    );
    assert_eq!(f.particle_count(), 5 * 3 * 2);
    // spacing larger than the extent: only the origin point
    assert_eq!(
        block(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(1, 1, 1),
            Fix128::from_int(2)
        )
        .particle_count(),
        1
    );
    // spacing that does not divide the extent: floor((1)/(0.375)) + 1 = 3 per axis
    assert_eq!(
        block(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1), q(3, 8)).particle_count(),
        27
    );
}

#[test]
fn degenerate_boxes_are_empty_or_a_single_particle() {
    let p = Vec3Fix::from_int(2, 3, 4);
    assert_eq!(block(p, p, q(1, 4)).particle_count(), 1, "min == max");
    // inverted on one axis: no particles
    let inv = block(
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::ONE, Fix128::NEG_ONE, Fix128::ONE),
        q(1, 4),
    );
    assert_eq!(inv.particle_count(), 0);
    assert!(inv.positions.is_empty() && inv.velocities.is_empty() && inv.densities.is_empty());
}

#[test]
fn lattice_is_x_major_and_exactly_on_the_grid() {
    let min = Vec3Fix::from_int(-1, 2, 0);
    let f = block(
        min,
        Vec3Fix::new(
            Fix128::from_int(-1) + q(1, 2),
            Fix128::from_int(2) + q(1, 2),
            q(1, 2),
        ),
        q(1, 4),
    );
    // 3 x 3 x 3
    assert_eq!(f.particle_count(), 27);
    for ix in 0..3i64 {
        for iy in 0..3i64 {
            for iz in 0..3i64 {
                let idx = (ix * 9 + iy * 3 + iz) as usize;
                let want = Vec3Fix::new(
                    Fix128::from_int(-1) + q(ix, 4),
                    Fix128::from_int(2) + q(iy, 4),
                    q(iz, 4),
                );
                assert_eq!(f.positions[idx], want, "({ix},{iy},{iz})");
            }
        }
    }
}

#[test]
fn block_is_at_rest_with_zeroed_state_and_keeps_its_config() {
    let cfg = FluidConfig {
        iterations: 7,
        substeps: 3,
        ..FluidConfig::default()
    };
    let f = Fluid::new_block(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1), q(1, 2), cfg);
    assert_eq!(f.config, cfg);
    assert_eq!(f.particle_count(), 27);
    assert!(f.velocities.iter().all(|v| *v == Vec3Fix::ZERO));
    assert!(f.densities.iter().all(|d| d.is_zero()));
    assert_eq!(f.velocities.len(), 27);
    assert_eq!(f.densities.len(), 27);
}

#[test]
fn non_positive_spacing_is_refused_instead_of_looping_forever() {
    for s in [Fix128::ZERO, Fix128::from_int(-1)] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let _ = block(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1), s);
        }));
        assert!(r.is_err(), "spacing {} must panic", s.to_f64());
    }
}

/// Poly6 normalisation check, independent of the crate: the density of an interior
/// particle of a lattice at spacing `s` must equal `m * sum_j W(r_ij)` with the standard
/// `W = 315 / (64 pi h^9) (h^2 - r^2)^3` (Muller et al. 2003; Macklin & Muller 2013 eq. 2).
/// The implementation's kernel has no `1/pi` ("simplified constant"), so its densities are
/// `pi` times larger and the rest-density constraint is satisfied by a lattice `pi^(1/3)`
/// times too sparse.
#[test]
#[ignore = "known defect: fluid poly6 / spiky kernels omit the 1/pi normalisation (Backlog ALICE-Physics fluid kernel normalisation)"]
fn interior_density_matches_the_normalised_poly6_sum() {
    let h = 0.2f64;
    let s = 0.05f64;
    let mass = 0.001f64;
    let cfg = FluidConfig {
        kernel_radius: Fix128::from_f64(h),
        particle_mass: Fix128::from_f64(mass),
        rest_density: Fix128::from_int(1000),
        iterations: 1,
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        viscosity: Fix128::ZERO,
        vorticity_strength: Fix128::ZERO,
        surface_tension: Fix128::ZERO,
        ..FluidConfig::default()
    };
    let n = 9usize; // 9^3 lattice, centre particle fully surrounded (4 spacings = 0.2 = h)
    let ext = s * (n as f64 - 1.0);
    let mut f = Fluid::new_block(
        Vec3Fix::ZERO,
        Vec3Fix::new(
            Fix128::from_f64(ext),
            Fix128::from_f64(ext),
            Fix128::from_f64(ext),
        ),
        Fix128::from_f64(s),
        cfg,
    );
    assert_eq!(f.particle_count(), n * n * n);
    f.step(Fix128::from_ratio(1, 60));
    let centre = (n / 2) * n * n + (n / 2) * n + n / 2;
    let mut want = 0.0;
    let c = (n / 2) as f64;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2))
                    * s
                    * s;
                if r2 < h * h {
                    want += mass * 315.0 / (64.0 * std::f64::consts::PI * h.powi(9))
                        * (h * h - r2).powi(3);
                }
            }
        }
    }
    let got = f.densities[centre].to_f64();
    assert!(
        (got - want).abs() / want < 1e-3,
        "density {got} vs normalised poly6 sum {want} (ratio {})",
        got / want
    );
}
