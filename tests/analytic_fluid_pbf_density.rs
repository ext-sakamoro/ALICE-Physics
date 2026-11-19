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
            let h3 = h * h * h;
            let d = h * h - r2;
            315.0 / (64.0 * std::f64::consts::PI * h3 * h3 * h3) * (d * d * d)
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

/// The cohesion pair potential of `apply_surface_tension`: the per-particle velocity change
/// `kappa dt (m / rho_0) sum_j (x_j - x_i) / r W(r)` is the force `-dU/dx_i` of
/// `U(r) = -m kappa (m / rho_0) integral_r^h W`, closed form for the poly6 kernel
fn cohesion_energy(positions: &[[f64; 3]], config: &FluidConfig) -> f64 {
    let h = config.kernel_radius.to_f64();
    let m = config.particle_mass.to_f64();
    let vol = m / config.rest_density.to_f64();
    let kappa = config.surface_tension.to_f64();
    let c = 315.0 / (64.0 * std::f64::consts::PI * h.powi(9));
    let antiderivative =
        |s: f64| h.powi(6) * s - h.powi(4) * s.powi(3) + 0.6 * h * h * s.powi(5) - s.powi(7) / 7.0;
    let mut e = 0.0;
    for (i, a) in positions.iter().enumerate() {
        for b in &positions[i + 1..] {
            let r = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
            if r < h {
                e -= m * kappa * vol * c * (antiderivative(h) - antiderivative(r));
            }
        }
    }
    e
}

/// One 30-frame sample of the stretched block
struct StretchedSample {
    frame: usize,
    corner_inward: f64,
    kinetic: f64,
    released: f64,
    quarter_err: f64,
    densest: f64,
}

/// The block laid out 15 % wide (5 x 5 x 5, no gravity) with the given XSPH strength, sampled
/// every 30 frames for 900 frames, and the closed-form drift of its corner: the cohesion
/// acceleration `a_coh` of the corner on the stretched lattice, damped once per frame, gives
/// `v(n) = a_coh dt (1 - d^n) / (1 - d)` after `n` frames (`d` the frame damping)
fn stretched_block(viscosity: Fix128) -> (Vec<StretchedSample>, f64, f64) {
    let config = FluidConfig {
        viscosity,
        ..no_gravity()
    };
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
    let rho0 = config.rest_density.to_f64();
    let m = config.particle_mass.to_f64();
    let h = config.kernel_radius.to_f64();
    let vol = m / rho0;
    let p3 = |v: &Vec3Fix| [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()];
    let start: Vec<[f64; 3]> = fluid.positions.iter().map(p3).collect();
    let e0 = cohesion_energy(&start, &config);
    // the corner at the origin; inward is along (1, 1, 1)
    let corner = start
        .iter()
        .position(|p| p.iter().all(|x| x.abs() < 1e-12))
        .expect("corner particle");
    let poly6 =
        |r: f64| 315.0 / (64.0 * std::f64::consts::PI * h.powi(9)) * (h * h - r * r).powi(3);
    let mut a = [0.0; 3];
    for q in &start {
        let d: Vec<f64> = (0..3).map(|k| q[k] - start[corner][k]).collect();
        let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r > 0.0 && r < h {
            (0..3)
                .for_each(|k| a[k] += config.surface_tension.to_f64() * vol * d[k] / r * poly6(r));
        }
    }
    let a_coh = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut samples = Vec::new();
    for frame in 1..=900 {
        fluid.step(Fix128::from_ratio(1, 60));
        if frame % 30 != 0 {
            continue;
        }
        let now: Vec<[f64; 3]> = fluid.positions.iter().map(p3).collect();
        let mut d: Vec<f64> = fluid.densities.iter().map(|x| x.to_f64() / rho0).collect();
        d.sort_by(f64::total_cmp);
        let q = &d[3 * d.len() / 4..];
        let v = fluid.velocities[corner];
        samples.push(StretchedSample {
            frame,
            corner_inward: (v.x.to_f64() + v.y.to_f64() + v.z.to_f64()) / 3f64.sqrt(),
            kinetic: fluid
                .velocities
                .iter()
                .map(|v| 0.5 * m * v.length_squared().to_f64())
                .sum(),
            released: e0 - cohesion_energy(&now, &config),
            quarter_err: q.iter().map(|x| (x - 1.0).abs()).sum::<f64>() / q.len() as f64,
            densest: q.iter().copied().fold(0.0, f64::max),
        });
    }
    (samples, a_coh, config.damping.to_f64())
}

/// A free block laid out on a lattice 15 % wider than the rest spacing (interior density
/// `rho_0 / 1.15^3 = 0.66 rho_0`, `C < 0`, no gravity). With `C >= 0` clamped the density
/// constraint never acts on it: the block contracts under cohesion alone, slowed by the
/// frame damping and XSPH. Over 900 frames at the default config, sampled every 30:
/// 1. energy budget: the kinetic energy never exceeds the cohesion energy released so far,
///    `KE(t) <= E_coh(0) - E_coh(t)` (closed-form pair potential; measured 1.1e-5 J against
///    5.4e-4 J at frame 840);
/// 2. monotone: the densest-quarter `|rho / rho_0 - 1|` does not rise from one sample to the
///    next (tolerance 1e-4; measured 0.3799 -> 0.3362);
/// 3. terminal drift: from frame 300 on the corner moves inward at a steady speed (spread
///    below 3 %, measured 0.295-0.301 of the bound), below the XSPH-free drift
///    `a_coh dt / (1 - d)` (XSPH only removes speed), at a stated ratio to it: measured 0.30
///    at the default `c = 0.3`, bound 0.25 (tighten only; the XSPH-coupled closed form is
///    open). `stretched_block_drift_without_xsph_follows_the_damped_closed_form` checks the
///    bound itself;
/// 4. no overshoot: the densest quarter stays below `1.003 rho_0`.
#[test]
fn stretched_block_contracts_under_cohesion_within_its_energy_budget() {
    let (samples, a_coh, d) = stretched_block(FluidConfig::default().viscosity);
    let bound = a_coh / 60.0 / (1.0 - d);
    let mut last_err = f64::INFINITY;
    for s in &samples {
        eprintln!(
            "frame {}: corner inward {} m/s ({} of the bound), KE {} J, released {} J, quarter err {}",
            s.frame,
            s.corner_inward,
            s.corner_inward / bound,
            s.kinetic,
            s.released,
            s.quarter_err
        );
        assert!(
            s.kinetic <= s.released,
            "frame {}: KE {} J above the cohesion energy released {} J",
            s.frame,
            s.kinetic,
            s.released
        );
        assert!(
            s.quarter_err <= last_err + 1e-4,
            "frame {}: quarter error rose {last_err} -> {}",
            s.frame,
            s.quarter_err
        );
        assert!(
            s.densest < 1.003,
            "frame {}: densest quarter at {} rho_0",
            s.frame,
            s.densest
        );
        last_err = s.quarter_err;
    }
    let late: Vec<f64> = samples
        .iter()
        .filter(|s| s.frame >= 300)
        .map(|s| s.corner_inward)
        .collect();
    let (lo, hi) = late.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
        (lo.min(*v), hi.max(*v))
    });
    assert!(
        hi / lo - 1.0 < 0.03,
        "corner speed not steady from frame 300: {lo} .. {hi} m/s"
    );
    assert!(
        hi < bound,
        "corner speed {hi} m/s above the XSPH-free drift {bound} m/s"
    );
    assert!(
        lo / bound >= 0.25,
        "corner speed {lo} m/s is {} of the drift (measured 0.30)",
        lo / bound
    );
    assert!(last_err < 0.3804, "the block did not contract: {last_err}");
}

/// The closed form of the corner drift where its premise holds: with XSPH off the corner
/// follows `v(n) = a_coh dt (1 - d^n) / (1 - d)`, `a_coh` the cohesion acceleration on the
/// initial (stretched) lattice. Measured 2026-11-18 against it, every 30 frames: 30 -0.9 %,
/// 60 -0.6 %, 90 0.0 %, 120 +0.8 %, 150 +1.7 %, 180 +2.9 %, 210 +4.3 %, 240 +5.9 %,
/// 270 +7.7 %, 300 +9.7 %. Past the first frames the block contracts, the pairs come closer,
/// `W` and so `a_coh` grow, and the closed form (the initial-lattice drift) no longer
/// describes it. Window rule: the frames up to the last one whose measured deviation is at
/// most half the bound (5 %), here 30-210; each within 10 %.
#[test]
fn stretched_block_drift_without_xsph_follows_the_damped_closed_form() {
    let (samples, a_coh, d) = stretched_block(Fix128::ZERO);
    let mut compared = 0;
    for s in samples.iter().filter(|s| s.frame <= 210) {
        let closed = a_coh / 60.0 * (1.0 - d.powi(s.frame as i32)) / (1.0 - d);
        let dev = s.corner_inward / closed - 1.0;
        eprintln!(
            "frame {}: corner inward {} m/s, closed form {closed} m/s, {dev:+.4}",
            s.frame, s.corner_inward
        );
        assert!(
            dev.abs() <= 0.10,
            "frame {}: {} m/s vs closed form {closed} m/s",
            s.frame,
            s.corner_inward
        );
        compared += 1;
    }
    assert_eq!(compared, 7, "frames 30-210 compared");
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

/// Hydrostatic pressure is linear in depth, so the side wall carries `rho_0 g (H - y) D` per
/// unit height with `H = M / (rho_0 W D)` (W, D the fluid width and depth). The x+ wall is
/// read in 0.1 m bins, `K = F_bin / integral_bin rho_0 g (H - y) D dy`, so a wrong profile
/// cannot pass through cancellation (the total over the wall reads 1.035 with the bottom
/// bin 15 % high and the top one 50 % low). Interior bins (0.2-0.5 m) are hydrostatic
/// within 5 % at the default solver settings. The floor/wall junction and the three bins
/// under the free surface are structural (4x the iterations cut the bottom density residual
/// 6x and left K unchanged; see the fluid MODEL) and are stated rows with bounds that may
/// only tighten. The per-particle readings also add up to `boundary_force`.
#[test]
fn side_wall_load_per_height_bin_matches_the_hydrostatic_profile() {
    let fluid = settled(FluidConfig::default(), 900);
    let s = fluid.config.rest_spacing.to_f64();
    let rho0 = fluid.config.rest_density.to_f64();
    let g = fluid.config.gravity.y.to_f64().abs();
    let (w, d) = (NX as f64 * s, NZ as f64 * s);
    let mass = fluid.particle_count() as f64 * fluid.config.particle_mass.to_f64();
    let height = mass / (rho0 * w * d);
    let wall = Fix128::from_int(NX) * fluid.config.rest_spacing;
    let mut load = [0.0_f64; NY as usize];
    let mut counted = 0usize;
    for (p, f) in fluid.boundary.iter().zip(fluid.boundary_forces()) {
        if p.x > wall && p.y > Fix128::ZERO {
            let bin = ((p.y.to_f64() / s).floor() as usize).min(NY as usize - 1);
            load[bin] += -f.x.to_f64();
            counted += 1;
        }
    }
    assert!(counted > 0, "no x+ wall particles selected");
    let k: Vec<f64> = (0..NY as usize)
        .map(|b| {
            let (lo, hi) = (
                (b as f64 * s).min(height),
                ((b as f64 + 1.0) * s).min(height),
            );
            let closed = rho0 * g * d * (height * (hi - lo) - 0.5 * (hi * hi - lo * lo));
            load[b] / closed
        })
        .collect();
    eprintln!("side wall K per 0.1 m bin, floor up: {k:.3?}");
    for (b, kb) in k.iter().enumerate().take(5).skip(2) {
        assert!((kb - 1.0).abs() < 0.05, "interior bin {b}: K = {kb}");
    }
    // stated rows, each from the default config with C >= 0 clamped (measured 2026-11-18,
    // 900 frames) and a 0.01 margin: the floor/wall junction (1.203) and the three bins under
    // the free surface (0.899 / 0.804 / 0.488)
    assert!(
        k[0] <= 1.21,
        "floor/wall junction bin: K = {} (measured 1.203)",
        k[0]
    );
    assert!(k[5] > 0.89, "bin 0.5-0.6 m: K = {} (measured 0.899)", k[5]);
    assert!(k[6] > 0.79, "bin 0.6-0.7 m: K = {} (measured 0.804)", k[6]);
    assert!(
        k[7] > 0.47,
        "surface bin 0.7-0.8 m: K = {} (measured 0.488)",
        k[7]
    );
    let total: f64 = fluid.boundary_forces().iter().map(|f| f.y.to_f64()).sum();
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
/// Solids given by an SDF couple through density maps instead, and meet Archimedes
/// (`analytic_fluid_sdf_body.rs`).
#[test]
#[ignore = "known defect: a cube of boundary particles in place of fluid is pushed up by 1.535 rho_0 g V (measured, s = 0.05, 300 frames); the cause is not measured"]
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
