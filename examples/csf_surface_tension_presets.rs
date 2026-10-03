//! Young-Laplace pressure jump of a droplet from the CSF force, for every
//! surface-tension preset (`SIGMA_WATER_AIR`, `SIGMA_PLA_AIR`,
//! `SIGMA_MERCURY_AIR`, `SIGMA_STEEL_ARGON`).
//!
//! The continuum surface force `f = -sigma kappa n delta` pulls toward the centre
//! of curvature; integrating `f` across the band on the axis gives the pressure
//! jump `p_in - p_out = 2 sigma / R` (Brackbill, Kothe & Zemach 1992; de Gennes
//! et al. 2004). The reference value is computed here from the preset's own
//! `sigma`, never from the force field.
//!
//! ```bash
//! cargo run --release --example csf_surface_tension_presets --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::multiphase::{initialize_level_set_sphere, Grid3d};
use alice_physics::surface_tension_csf::{
    compute_csf_field, SIGMA_MERCURY_AIR, SIGMA_PLA_AIR, SIGMA_STEEL_ARGON, SIGMA_WATER_AIR,
};

fn main() {
    let n = 24usize;
    let dx = Fix128::from_ratio(1, 16);
    let c = Fix128::from_int(12) * dx;
    let radius = Fix128::from_int(8) * dx; // 0.5 m
    let mut phi = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    initialize_level_set_sphere(&mut phi, c, c, c, radius);

    for (name, sigma) in [
        ("water/air", SIGMA_WATER_AIR),
        ("PLA melt/air", SIGMA_PLA_AIR),
        ("mercury/air", SIGMA_MERCURY_AIR),
        ("molten steel/argon", SIGMA_STEEL_ARGON),
    ] {
        let (fx, _fy, _fz) = compute_csf_field(&phi, sigma, dx.double());
        // inward force on the +x side: p_in - p_out = - sum f_x dx across the band
        let mut jump = Fix128::ZERO;
        for i in 19..=21 {
            jump = jump - fx[i + n * (12 + n * 12)] * dx;
        }
        let want = 2.0 * sigma.to_f64() / radius.to_f64();
        println!(
            "{name:20} sigma = {:.3} N/m  dp = {:.5} Pa  2 sigma / R = {want:.5} Pa",
            sigma.to_f64(),
            jump.to_f64()
        );
        assert!((jump.to_f64() - want).abs() / want < 0.02, "{name}");
    }
}
