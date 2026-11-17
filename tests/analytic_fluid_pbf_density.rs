//! Oracles for the PBF density constraint (`src/fluid.rs` MODEL section).
//!
//! Scene: a 4 x 8 x 4 lattice at the default rest spacing `s = h / 2`, in an open box
//! lined with two layers of boundary particles (`Fluid::box_boundary`), default
//! `FluidConfig` (lattice-calibrated mass, XSPH 0.3, 10 iterations, 2 substeps), frame
//! `dt = 1/60 s`.
//!
//! External references:
//! - hydrostatics: fluid at rest on a floor is carried by the integral of `rho g h` over
//!   the floor, which is its weight `M g`; the floor reaction is read from
//!   `Fluid::boundary_force` (fluid-fluid corrections cancel pairwise)
//! - incompressibility: the density stays within 2 % of `rho_0`
//! - a resting block with no gravity is an equilibrium: it must not move
//!
//! Pass lines (fixed before the implementation): density error <= 2 %, largest particle
//! speed over the last 60 frames < 0.05 m/s, the rest block keeps its height.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 weight M g and density ratios for the PBF hydrostatic and rest-block oracles"
)]

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

const NX: i64 = 4;
const NY: i64 = 8;
const NZ: i64 = 4;

fn column(config: FluidConfig) -> Fluid {
    let s = config.rest_spacing;
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
        2,
    );
    fluid.with_boundary(boundary)
}

struct Run {
    /// Highest particle centre over the rest height `NY s`
    top: f64,
    /// Mean |rho / rho_0 - 1| over the middle half of the sorted densities
    density_error: f64,
    /// Largest particle speed in every frame
    max_speed: Vec<f64>,
    /// Floor reaction over the weight `M |g|` (gravity runs only)
    support: f64,
}

fn run(config: FluidConfig, frames: usize) -> Run {
    let mut fluid = column(config);
    let dt = Fix128::from_ratio(1, 60);
    let mut max_speed = Vec::with_capacity(frames);
    for _ in 0..frames {
        fluid.step(dt);
        max_speed.push(
            fluid
                .velocities
                .iter()
                .map(|v| v.length().to_f64())
                .fold(0.0, f64::max),
        );
    }
    let s = fluid.config.rest_spacing.to_f64();
    let top = fluid
        .positions
        .iter()
        .map(|p| p.y.to_f64())
        .fold(f64::MIN, f64::max)
        / (NY as f64 * s);
    let rho0 = fluid.config.rest_density.to_f64();
    let mut d: Vec<f64> = fluid.densities.iter().map(|d| d.to_f64()).collect();
    d.sort_by(f64::total_cmp);
    let mid = &d[d.len() / 4..3 * d.len() / 4];
    let density_error = mid.iter().map(|x| (x / rho0 - 1.0).abs()).sum::<f64>() / mid.len() as f64;
    let g = fluid.config.gravity.y.to_f64().abs();
    let weight = fluid.particle_count() as f64 * fluid.config.particle_mass.to_f64() * g;
    let support = if weight > 0.0 {
        fluid.boundary_force().y.to_f64() / weight
    } else {
        0.0
    };
    Run {
        top,
        density_error,
        max_speed,
        support,
    }
}

fn tail_max(v: &[f64], from: usize, to: usize) -> f64 {
    v[from..to].iter().copied().fold(0.0, f64::max)
}

fn no_gravity() -> FluidConfig {
    FluidConfig {
        gravity: Vec3Fix::ZERO,
        ..FluidConfig::default()
    }
}

/// A resting block without gravity stays at rest: height (a perfect lattice gives
/// `(NY - 1/2) / NY = 0.9375`), density and speed. The settling transient is bounded
/// too: measured 0.227 m/s in frames 60-119 at the default scale (0.113 at `s = 0.05`).
#[test]
fn rest_block_without_gravity_stays_at_rest() {
    let r = run(no_gravity(), 300);
    let ideal = (NY as f64 - 0.5) / NY as f64;
    assert!(
        (r.top - ideal).abs() < 0.02,
        "top/rest {} vs lattice {ideal}",
        r.top
    );
    assert!(r.density_error <= 0.02, "density error {}", r.density_error);
    let last = tail_max(&r.max_speed, 240, 300);
    assert!(last < 0.05, "max speed over the last 60 frames {last} m/s");
    let transient = tail_max(&r.max_speed, 60, 120);
    assert!(
        transient < 0.3,
        "max speed in frames 60-119 {transient} m/s"
    );
}

/// A column under gravity settles (900 frames at the default scale, MODEL), stays
/// within 2 % of `rho_0`, and the floor carries its weight.
#[test]
fn column_under_gravity_settles_and_the_floor_carries_its_weight() {
    let r = run(FluidConfig::default(), 900);
    assert!(r.density_error <= 0.02, "density error {}", r.density_error);
    let last = tail_max(&r.max_speed, 840, 900);
    assert!(last < 0.05, "max speed over the last 60 frames {last} m/s");
    assert!(
        (r.support - 1.0).abs() < 0.01,
        "floor reaction / weight {}",
        r.support
    );
}

/// Without the boundary particles the column has nothing to stand on in the
/// constraint: the floor reaction must be zero (the oracle above reads the walls).
#[test]
fn floor_reaction_is_zero_without_boundary_particles() {
    let config = FluidConfig::default();
    let s = config.rest_spacing;
    let mut fluid = Fluid::new_block(Vec3Fix::ZERO, Vec3Fix::new(s, s, s), s, config);
    assert!(
        fluid.boundary_forces().is_empty(),
        "no readings before a step"
    );
    fluid.step(Fix128::from_ratio(1, 60));
    assert_eq!(fluid.boundary_force(), Vec3Fix::ZERO);
    assert!(
        fluid.boundary_forces().is_empty(),
        "one reading per boundary particle"
    );
}

/// The default mass makes the default rest lattice exactly `rho_0`:
/// `sum_k m W(|k| s) = rho_0` over the lattice points within `h` (closed form of
/// `lattice_mass`, recomputed here from the poly6 kernel in f64).
#[test]
fn default_mass_makes_the_rest_lattice_exactly_rho0() {
    let c = FluidConfig::default();
    let h = c.kernel_radius.to_f64();
    let s = c.rest_spacing.to_f64();
    let m = c.particle_mass.to_f64();
    let poly6 = |r2: f64| {
        if r2 >= h * h {
            0.0
        } else {
            315.0 / (64.0 * std::f64::consts::PI * h.powi(9)) * (h * h - r2).powi(3)
        }
    };
    let mut sum = 0.0;
    for a in -2i32..=2 {
        for b in -2i32..=2 {
            for k in -2i32..=2 {
                sum += poly6(f64::from(a * a + b * b + k * k) * s * s);
            }
        }
    }
    let rho = m * sum;
    let rho0 = c.rest_density.to_f64();
    assert!((rho / rho0 - 1.0).abs() < 1e-9, "lattice density {rho}");
}

/// `box_boundary` lines the floor and four walls with `layers` shells and leaves the
/// top open: count = (nx + 2L)(ny + L)(nz + 2L) - nx ny nz.
#[test]
fn box_boundary_counts_match_the_closed_form() {
    let s = Fix128::from_ratio(1, 4);
    let max = Vec3Fix::new(
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_ratio(3, 4),
    );
    let (nx, ny, nz) = (4usize, 8usize, 3usize);
    for l in 1..=3usize {
        let b = Fluid::box_boundary(Vec3Fix::ZERO, max, s, l);
        assert_eq!(
            b.len(),
            (nx + 2 * l) * (ny + l) * (nz + 2 * l) - nx * ny * nz,
            "layers {l}"
        );
        assert!(b.iter().all(|p| p.y < Fix128::from_int(2)), "no lid");
    }
    assert!(Fluid::box_boundary(Vec3Fix::ZERO, max, s, 0).is_empty());
    assert!(Fluid::box_boundary(Vec3Fix::ZERO, max, Fix128::ZERO, 2).is_empty());
    assert!(Fluid::box_boundary(max, Vec3Fix::ZERO, s, 2).is_empty());
}

/// A free block laid out on a lattice 15 % wider than the rest spacing (interior density
/// `rho_0 / 1.15^3 = 0.66 rho_0`, `C < 0`) has an interior under tension: the two-sided
/// interior constraint pulls it back to `rho_0` while the free surface stays one-sided.
/// With the paper's `C >= 0` clamp everywhere the block keeps its 0.66 density.
#[test]
fn stretched_block_contracts_to_the_rest_density() {
    let config = no_gravity();
    let s = config.rest_spacing * Fix128::from_ratio(115, 100);
    let n = 5i64;
    let mut fluid = Fluid::new_block(
        Vec3Fix::ZERO,
        Vec3Fix::new(
            Fix128::from_int(n - 1) * s,
            Fix128::from_int(n - 1) * s,
            Fix128::from_int(n - 1) * s,
        ),
        s,
        config,
    );
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..300 {
        fluid.step(dt);
    }
    let rho0 = config.rest_density.to_f64();
    let mut d: Vec<f64> = fluid.densities.iter().map(|d| d.to_f64()).collect();
    d.sort_by(f64::total_cmp);
    // the densest quarter is the interior
    let interior = &d[3 * d.len() / 4..];
    let err = interior.iter().map(|x| (x / rho0 - 1.0).abs()).sum::<f64>() / interior.len() as f64;
    assert!(err <= 0.02, "interior density error {err}");
}

/// The floor reaction carries the weight at any mass, not only near the default
/// `m ~ 1 kg` (a reaction that dropped the mass would read 1 at the default): at
/// `rho_0 = 2500` the lattice mass is 2.5x and PBF positions are unchanged (the
/// constraint is in `rho / rho_0`), so the column settles the same and
/// `boundary_force / (M g)` is again 1 within 1 %.
#[test]
fn floor_reaction_scales_with_the_particle_mass() {
    let base = FluidConfig::default();
    let rest_density = Fix128::from_int(2500);
    let config = FluidConfig {
        rest_density,
        particle_mass: FluidConfig::lattice_mass(
            rest_density,
            base.kernel_radius,
            base.rest_spacing,
        ),
        ..base
    };
    assert!(
        config.particle_mass.to_f64() > 2.0,
        "mass {}",
        config.particle_mass.to_f64()
    );
    let r = run(config, 900);
    assert!(
        (r.support - 1.0).abs() < 0.01,
        "floor reaction / weight {}",
        r.support
    );
}

/// Settled column after `frames`, for the wall readings below
fn settled(config: FluidConfig, frames: usize) -> Fluid {
    let mut fluid = column(config);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..frames {
        fluid.step(dt);
    }
    fluid
}

/// Hydrostatic pressure is linear in depth (F031: the pressure field of a resting tank
/// is linear with the right bottom pressure), so the force on a side wall is
/// `integral_0^H rho g (H - y) D dy = 1/2 rho_0 g H^2 D` with `H = M / (rho_0 W D)`
/// (W, D the fluid width and depth). This reads the pressure profile, which the floor
/// reaction (a momentum balance) cannot: any profile with the right integral carries
/// the weight. Pass line: within 5 % (fixed before the run). The per-particle readings
/// also add up to `boundary_force`.
#[test]
fn side_wall_carries_the_hydrostatic_force() {
    let fluid = settled(FluidConfig::default(), 900);
    let s = fluid.config.rest_spacing.to_f64();
    let rho0 = fluid.config.rest_density.to_f64();
    let g = fluid.config.gravity.y.to_f64().abs();
    let (w, d) = (NX as f64 * s, NZ as f64 * s);
    let mass = fluid.particle_count() as f64 * fluid.config.particle_mass.to_f64();
    let height = mass / (rho0 * w * d);
    let expected = 0.5 * rho0 * g * height * height * d;
    let forces = fluid.boundary_forces();
    assert_eq!(forces.len(), fluid.boundary.len());
    let wall = Fix128::from_int(NX) * fluid.config.rest_spacing;
    let mut side = 0.0;
    let mut on_wall = 0usize;
    for (p, f) in fluid.boundary.iter().zip(forces) {
        if p.x > wall && p.y > Fix128::ZERO {
            side += f.x.to_f64();
            on_wall += 1;
        }
    }
    assert!(on_wall > 0, "no x+ wall particles selected");
    // the wall pushes the fluid towards -x
    let ratio = -side / expected;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "side wall force / (1/2 rho g H^2 D) = {ratio} ({side} N vs {expected} N)"
    );
    let total: f64 = forces.iter().map(|f| f.y.to_f64()).sum();
    let reported = fluid.boundary_force().y.to_f64();
    assert!(
        (total - reported).abs() <= 1e-6 * reported.abs(),
        "per-particle sum {total} vs boundary_force {reported}"
    );
}

/// Substeps, not iterations, set the accuracy (S154); the settled answer must not
/// depend on them: at 4 substeps the column still carries its weight within 1 % and
/// stays within 2 % of `rho_0`.
#[test]
fn settled_column_does_not_depend_on_the_substep_count() {
    let config = FluidConfig {
        substeps: 4,
        ..FluidConfig::default()
    };
    let r = run(config, 900);
    assert!(r.density_error <= 0.02, "density error {}", r.density_error);
    assert!(
        (r.support - 1.0).abs() < 0.01,
        "floor reaction / weight {}",
        r.support
    );
}

/// Archimedes (F091): a fixed solid in a resting tank is pushed up by the weight of the
/// fluid it displaces, `rho_0 g V`. Scene at `s = 0.05` (settles within 300 frames,
/// MODEL): a 6 x 10 x 6 lattice in the open box with a 2 x 2 x 2-cell cube of boundary
/// particles at cells (2..4, 3..5, 2..4) in place of fluid, `V = 8 s^3` (the boundary
/// plane sits half a spacing outside the particle centres, as for the box walls). The
/// cube's readings sum to the force it exerts on the fluid, `-rho_0 g V` in y. Pass
/// line: within 5 % (fixed before the run, the same discretisation as the side wall).
#[test]
fn submerged_cube_feels_the_archimedes_force() {
    let s = Fix128::from_ratio(1, 20);
    let h = s * Fix128::from_int(2);
    let rest_density = Fix128::from_int(1000);
    let config = FluidConfig {
        rest_spacing: s,
        kernel_radius: h,
        particle_mass: FluidConfig::lattice_mass(rest_density, h, s),
        ..FluidConfig::default()
    };
    let (nx, ny, nz) = (6i64, 10i64, 6i64);
    let in_cube =
        |i: i64, j: i64, k: i64| (2..4).contains(&i) && (3..5).contains(&j) && (2..4).contains(&k);
    let centre = |i: i64| (Fix128::from_int(i) + Fix128::from_ratio(1, 2)) * s;
    let mut fluid_points = Vec::new();
    let mut cube = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = Vec3Fix::new(centre(i), centre(j), centre(k));
                if in_cube(i, j, k) {
                    cube.push(p);
                } else {
                    fluid_points.push(p);
                }
            }
        }
    }
    assert_eq!(cube.len(), 8);
    let mut boundary = Fluid::box_boundary(
        Vec3Fix::ZERO,
        Vec3Fix::new(
            Fix128::from_int(nx) * s,
            Fix128::from_int(2 * ny) * s,
            Fix128::from_int(nz) * s,
        ),
        s,
        2,
    );
    let first_cube = boundary.len();
    boundary.extend(cube);
    let mut fluid = Fluid::new(fluid_points, config).with_boundary(boundary);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..300 {
        fluid.step(dt);
    }
    let g = fluid.config.gravity.y.to_f64().abs();
    let sf = s.to_f64();
    let expected = rest_density.to_f64() * g * 8.0 * sf * sf * sf;
    let on_fluid: f64 = fluid.boundary_forces()[first_cube..]
        .iter()
        .map(|f| f.y.to_f64())
        .sum();
    let ratio = -on_fluid / expected;
    eprintln!("buoyancy / (rho_0 g V) = {ratio}");
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "buoyancy / (rho_0 g V) = {ratio} ({} N vs {expected} N)",
        -on_fluid
    );
}
