//! The fixed boundary particles of [`Fluid::box_boundary`] complete the kernel support of a
//! fluid particle next to a wall: with uniform particle volumes `m / rho_0` (the
//! lattice-calibrated mass), the boundary layers fill the kernel support, edges and corners
//! included, so a fluid lattice inside the box reads exactly `rho_0` at a flat wall, at the
//! floor, at a 90 degree edge and at a 3-wall corner. The number of layers needed is
//! `ceil(h / s) - 1` on the lattice: at the default `h = 2 s` one layer already reaches every
//! lattice point inside the support (the next one sits at exactly `h`, where `W = 0`); at
//! `h = 3 s` two layers are needed and one leaves part of the support empty.
//!
//! oracle: `sum_j m W_ij = rho_0` on the full rest lattice (the definition of
//! [`FluidConfig::lattice_mass`]); the box boundary is the rest lattice continued outside the
//! box, so the sum over fluid and boundary is the full-lattice sum.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

const NX: i64 = 4;
const NY: i64 = 8;
const NZ: i64 = 4;

/// The densities of a fluid lattice in the box, read from the first constraint iteration of
/// one substep without gravity and at rest (the predicted positions are the lattice)
fn lattice_densities(layers: usize, h_over_s: i64) -> (Fluid, Vec<f64>) {
    let base = FluidConfig::default();
    let s = base.rest_spacing;
    let h = s * Fix128::from_int(h_over_s);
    let config = FluidConfig {
        kernel_radius: h,
        particle_mass: FluidConfig::lattice_mass(base.rest_density, h, s),
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        iterations: 1,
        ..base
    };
    let half = s / Fix128::from_int(2);
    let fluid = Fluid::new_block(
        Vec3Fix::new(half, half, half),
        Vec3Fix::new(
            Fix128::from_int(NX) * s - half,
            Fix128::from_int(NY) * s - half,
            Fix128::from_int(NZ) * s - half,
        ),
        s,
        config,
    );
    let boundary = Fluid::box_boundary(
        Vec3Fix::ZERO,
        Vec3Fix::new(
            Fix128::from_int(NX) * s,
            Fix128::from_int(2 * NY) * s,
            Fix128::from_int(NZ) * s,
        ),
        s,
        layers,
    );
    let mut fluid = fluid.with_boundary(boundary);
    // the probes are found on the lattice: a short support moves the particles in the step
    let lattice = fluid.positions.clone();
    fluid.step(Fix128::from_ratio(1, 1_000_000));
    fluid.positions = lattice;
    let rho0 = config.rest_density.to_f64();
    let ratio = fluid.densities.iter().map(|d| d.to_f64() / rho0).collect();
    (fluid, ratio)
}

/// Index of the fluid particle at lattice cell `(i, j, k)`
fn at(fluid: &Fluid, i: i64, j: i64, k: i64) -> usize {
    let s = fluid.config.rest_spacing.to_f64();
    let c = |n: i64| (n as f64 + 0.5) * s;
    fluid
        .positions
        .iter()
        .position(|p| {
            (p.x.to_f64() - c(i)).abs() < 1e-9
                && (p.y.to_f64() - c(j)).abs() < 1e-9
                && (p.z.to_f64() - c(k)).abs() < 1e-9
        })
        .expect("lattice cell inside the block")
}

/// The four probe points: next to the x+ wall at mid height, on the floor away from the
/// walls, on the floor / x+ edge, and in the floor / x+ / z+ corner
fn probes(fluid: &Fluid) -> [(&'static str, usize); 4] {
    [
        ("flat wall", at(fluid, NX - 1, 3, 1)),
        ("floor", at(fluid, 1, 0, 1)),
        ("90 degree edge", at(fluid, NX - 1, 0, 1)),
        ("3-wall corner", at(fluid, NX - 1, 0, NZ - 1)),
    ]
}

fn assert_full_support(layers: usize, h_over_s: i64) {
    let (fluid, rho) = lattice_densities(layers, h_over_s);
    for (name, i) in probes(&fluid) {
        eprintln!(
            "{layers} layer(s), h = {h_over_s} s, {name}: rho / rho_0 = {}",
            rho[i]
        );
        assert!(
            (rho[i] - 1.0).abs() < 1e-9,
            "{name}: rho / rho_0 = {}",
            rho[i]
        );
    }
}

#[test]
fn the_default_box_boundary_completes_the_kernel_support_at_walls_edges_and_corners() {
    assert_full_support(2, 2);
}

#[test]
fn two_layers_complete_a_three_spacing_kernel() {
    assert_full_support(2, 3);
}

/// The property depends on the layer count: at `h = 3 s` one layer leaves the outer part of
/// the support empty, and every probe reads below `rho_0`, more where two or three walls meet
#[test]
fn one_layer_leaves_a_three_spacing_kernel_short() {
    let (fluid, rho) = lattice_densities(1, 3);
    let p = probes(&fluid);
    for (name, i) in p {
        eprintln!("1 layer, h = 3 s, {name}: rho / rho_0 = {}", rho[i]);
        assert!(rho[i] < 1.0 - 1e-3, "{name}: rho / rho_0 = {}", rho[i]);
    }
    let (flat, edge, corner) = (rho[p[0].1], rho[p[2].1], rho[p[3].1]);
    assert!(
        edge < flat && corner < edge,
        "flat {flat}, edge {edge}, corner {corner}"
    );
}
