//! Fluid / field-module oracles (group B of the 1.2.0 "deterministic ≠
//! correct" audit).
//!
//! Every test below compares a module against a textbook closed form or a
//! published worked example, with the tolerance of the discretisation
//! stated next to the assertion. Modules that are empirical heuristics
//! with no closed form get invariant tests only (monotonicity, symmetry,
//! limits, conservation) and are listed as `validation: none` in the
//! report. Nothing here touches `src/`; when a module disagrees with the
//! textbook the test pins the invariant that *is* true and the
//! discrepancy is reported instead of asserted.
//!
//! Companion of `tests/engineering_oracles.rs` (transient thermal +
//! fatigue), which is not duplicated here.

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::acoustic_wave::{leapfrog_step, speeds, stable_dt};
use alice_physics::aeroelasticity::{viv_step, VivParameters, VivState};
use alice_physics::buoyancy_zone::{BuoyancyZone, ZoneShape};
use alice_physics::cfd_solver::CfdSolver;
use alice_physics::cloth_fluid::{apply_fluid_forces_to_cloth, ClothFluidCoupling};
use alice_physics::compressible::{
    normal_shock_jump, riemann_invariants, stagnation_pressure_ratio, stagnation_temp_ratio,
    IdealGas,
};
use alice_physics::electromagnetic::{lorentz_force, lorentz_force_sum, ChargedBody, EmSource};
use alice_physics::erosion::{ErosionConfig, ErosionModifier, ErosionType};
use alice_physics::eulerian_grid::{project_pressure, MacGrid};
use alice_physics::fsi_advanced::{
    aggregate_forces, buoyancy_force, drag_force, react_back_pressure, SolidSample,
};
use alice_physics::interface_capture::fast_sweeping_reinit;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{
    curvature_at, initialize_level_set_sphere, trilinear_sample, Grid3d,
};
use alice_physics::non_newtonian::{Bingham, Carreau, HerschelBulkley, PowerLaw};
use alice_physics::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::piezoelectric::PiezoElement;
use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_sph::{poly6, spiky_grad, SphConfig, SphParticle, SphSolver};
use alice_physics::sdf_wind_field::SdfWindField;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::smoke_fire::{
    boussinesq_buoyancy_n_per_m3, heat_release_j_per_m3_s, reaction_rate_kg_per_m3_s,
    soot_generation_kg_per_m3_s, ArrheniusReaction,
};
use alice_physics::solver::RigidBody;
use alice_physics::surface_tension_csf::{
    csf_body_force, interface_normal, smeared_delta, SIGMA_WATER_AIR,
};
use alice_physics::thermal::{ThermalConfig, ThermalModifier};
use alice_physics::turbulence::{
    smagorinsky_eddy_viscosity, strain_rate_magnitude, SMAGORINSKY_CS,
};
use alice_physics::wave_ship::{
    free_surface_elevation, froude_krylov_vertical_n, Jonswap, ShipResponse, WaveComponent,
};
use alice_physics::wind_zone::WindZone;

use core::f64::consts::PI;

/// Relative error `|a − b| / |b|` (b ≠ 0).
fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// `|a − b| ≤ 2⁻⁴⁸ ≈ 3.6e-15`: the two values agree to within the
/// truncation noise of the I64F64 format (`Fix128` multiplication
/// truncates, so products of non-dyadic ratios such as 0.02 or 9.81
/// carry a few thousand last-place units depending on evaluation order).
fn fix_close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() <= Fix128::from_raw(0, 1 << 16)
}

// ============================================================================
// cfd_solver — viscous decay of a shear eigenmode (heat-equation analogue)
// ============================================================================

/// A unidirectional shear flow `u(y) x̂` has zero divergence and zero
/// advection, so the incompressible Navier–Stokes momentum equation
/// reduces to `∂u/∂t = ν ∂²u/∂y²` (Batchelor, *An Introduction to Fluid
/// Dynamics* §4.3 — the same PDE as Stokes' first problem). With zero-flux
/// walls (the crate's `down = center` mirror) the cosine mode
/// `u = A cos(π y / L)` decays as `exp(−ν π² t / L²)` exactly (Carslaw &
/// Jaeger 1959 §3.4). The u faces at `y_j = (j + ½) dx`, `L = ny·dx`.
///
/// The x-boundary u faces (`i = 0`, `i = nx`) are never diffused by the
/// solver, so they act as a fixed-velocity Dirichlet strip whose
/// divergence in the boundary cells drives a spurious pressure and a
/// secondary `v` flow (see report). Because the projection is elliptic
/// that error decays like `exp(−π x / L_y)` into the domain; the grid is
/// made wide (`nx = 64`) so the sampled centre column at `i = 32` is
/// `2 L_y` from the strip (`e^{−2π} ≈ 0.2 %` of a 27 % wall error).
#[test]
fn cfd_solver_shear_mode_decays_at_the_viscous_rate() {
    let (nx, ny, nz) = (64usize, 16usize, 2usize);
    let dx = Fix128::from_ratio(1, 128); // exact in binary
    let mut solver = CfdSolver::new(nx, ny, nz, dx);
    solver.gravity = Vec3Fix::ZERO;
    solver.density_kg_m3 = Fix128::from_int(1000);
    solver.dynamic_viscosity_pas = Fix128::ONE; // ν = 1e-3 m²/s
    let nu = 1e-3f64;
    let dxf = 1.0f64 / 128.0;
    let l = ny as f64 * dxf;
    let amp = 1e-3f64; // m/s — advection displacement over the run is 0.06 cell

    // u(i, j, k) = A cos(π (j + ½) / ny), uniform in x and z.
    for k in 0..nz {
        for j in 0..ny {
            let value = Fix128::from_f64(amp * (PI * (j as f64 + 0.5) / ny as f64).cos());
            for i in 0..=nx {
                let ix = i + (nx + 1) * (j + ny * k);
                solver.grid.u[ix] = value;
            }
        }
    }
    let dt = Fix128::from_ratio(1, 100); // r = ν dt / dx² = 0.164 < 1/6 (3-D explicit limit)
    let steps = 50;
    for _ in 0..steps {
        solver.step(dt);
    }
    let t = steps as f64 * 0.01;
    let decay = (-nu * PI * PI * t / (l * l)).exp(); // 0.680
    assert!(
        decay > 0.6 && decay < 0.75,
        "run is not in the useful decay range: {decay}"
    );

    // 1.2.0: the boundary faces diffuse too (zero-gradient mirror), so the
    // shear profile is uniform in x — check the wall column (i = 0) and the
    // centre column against the same closed form; before 1.2.0 the wall column
    // was 27 % off and drove a secondary v flow of ~10 % of A.
    let mut worst = 0.0f64;
    let mut mean = 0.0f64;
    for i in [0usize, nx / 2, nx] {
        for j in 0..ny {
            let got = solver.grid.u(i, j, 0).to_f64();
            let want = amp * (PI * (j as f64 + 0.5) / ny as f64).cos() * decay;
            worst = worst.max((got - want).abs());
            if i == nx / 2 {
                mean += got;
            }
        }
    }
    mean /= ny as f64;
    // discrete decay (1 − 4 r sin²(π dx / 2L))^n vs exp(−ν π² t / L²) differ by
    // 0.03 % here; 0.5 % of the mode amplitude covers it at every column
    assert!(
        worst < 0.005 * amp,
        "shear mode max error {worst:.3e} m/s (0.5 % of A = {:.1e})",
        0.005 * amp
    );
    // zero-flux walls conserve the momentum of the column (mode has zero mean)
    assert!(mean.abs() < 1e-4 * amp, "column mean drifted to {mean:.3e}");
    // a pure shear mode drives no secondary flow anywhere: |v| stays at the
    // level of the divergence-projection rounding
    let v_max = solver
        .grid
        .v
        .iter()
        .map(|v| v.to_f64().abs())
        .fold(0.0, f64::max);
    assert!(
        v_max < 1e-3 * amp,
        "secondary v {v_max:.3e} (should be rounding only)"
    );
    let w_max = solver
        .grid
        .w
        .iter()
        .map(|w| w.to_f64().abs())
        .fold(0.0, f64::max);
    assert!(
        w_max < 1e-9 * amp,
        "w should be rounding noise only: {w_max:.3e}"
    );
}

// ============================================================================
// eulerian_grid — Helmholtz–Hodge projection
// ============================================================================

/// `project_pressure` solves `∇²p = (ρ/dt) ∇·u*` and subtracts
/// `(dt/ρ) ∇p` (Chorin 1968; Bridson, *Fluid Simulation for Computer
/// Graphics* ch. 5). The oracle is the defining property: interior cells
/// are divergence-free afterwards, and an already divergence-free field
/// is a fixed point (its pressure is identically zero).
#[test]
fn eulerian_grid_projection_removes_interior_divergence() {
    let n = 6usize;
    let dx = Fix128::from_ratio(1, 16);
    let mut grid = MacGrid::new(n, n, n, dx);
    // u = a·x on the x faces → ∇·u = a everywhere (a = 1 /s)
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let ix = i + (n + 1) * (j + n * k);
                grid.u[ix] = Fix128::from_int(i as i64) * dx;
            }
        }
    }
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                assert_eq!(
                    grid.divergence(i, j, k),
                    Fix128::ONE,
                    "∇·(a x x̂) = a exactly"
                );
            }
        }
    }
    // Gauss–Seidel on a 6³ Dirichlet box: ρ_GS = cos²(π/7) ≈ 0.81 → 150
    // iterations leave ~1e-14 of the initial residual.
    project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 150);
    let mut worst = 0.0f64;
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                worst = worst.max(grid.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    assert!(
        worst < 1e-8,
        "interior divergence after projection {worst:.3e} (was 1.0)"
    );
    assert!(
        grid.pressure.iter().any(|p| !p.is_zero()),
        "a divergent field must produce a pressure field"
    );
}

#[test]
fn eulerian_grid_projection_is_identity_on_a_solenoidal_field() {
    let n = 5usize;
    let dx = Fix128::from_ratio(1, 8);
    let mut grid = MacGrid::new(n, n, n, dx);
    // uniform flow u = 3 x̂ is divergence-free
    for u in grid.u.iter_mut() {
        *u = Fix128::from_int(3);
    }
    let before = grid.clone();
    project_pressure(
        &mut grid,
        Fix128::from_ratio(1, 60),
        Fix128::from_int(1000),
        40,
    );
    assert!(grid.pressure.iter().all(|p| p.is_zero()), "p must stay 0");
    assert_eq!(
        grid.u, before.u,
        "solenoidal u must be untouched (bit-exact)"
    );
    assert_eq!(grid.v, before.v);
    assert_eq!(grid.w, before.w);
}

// ============================================================================
// sdf_sph — kernel normalisation + density-summation consistency
// ============================================================================

/// Müller, Charypar & Gross 2003 eq. (20): the Poly6 kernel integrates to
/// one over its support, `∫₀ʰ 4π r² W dr = 4π·(315/(64π h⁹))·(16/315) h⁹ = 1`.
/// Simpson's rule on a degree-8 polynomial with 2000 panels is exact to
/// f64 rounding; the module evaluates in f32, hence 1e-4.
#[test]
fn sdf_sph_poly6_integrates_to_one() {
    let h = 0.05f32;
    let panels = 2000usize;
    let dr = f64::from(h) / panels as f64;
    let mut sum = 0.0f64;
    for i in 0..=panels {
        let r = i as f64 * dr;
        let w = f64::from(poly6(r as f32, h));
        let weight = if i == 0 || i == panels {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * 4.0 * PI * r * r * w;
    }
    sum *= dr / 3.0;
    assert!((sum - 1.0).abs() < 1e-4, "∫ W_poly6 dV = {sum}");
    assert_eq!(poly6(h, h), 0.0, "compact support: W(h) = 0");
    assert!(poly6(0.0, h) > poly6(0.5 * h, h), "monotone decreasing");
}

/// Müller 2003 eq. (21): `W_spiky = 15/(π h⁶) (h − r)³`, so
/// `|∇W| = 45/(π h⁶) (h − r)²` and `W(0) = ∫₀ʰ |∇W| dr = 15/(π h³)`.
#[test]
fn sdf_sph_spiky_gradient_matches_closed_form_and_integrates_to_kernel_peak() {
    let h = 0.05f32;
    let hf = f64::from(h);
    for &frac in &[0.0f64, 0.1, 0.25, 0.5, 0.9] {
        let r = frac * hf;
        let want = 45.0 / (PI * hf.powi(6)) * (hf - r).powi(2);
        let got = f64::from(spiky_grad(r as f32, h));
        assert!(
            rel_err(got, want) < 1e-5,
            "|∇W_spiky|({frac} h) = {got} vs {want}"
        );
    }
    assert_eq!(spiky_grad(h, h), 0.0);
    // ∫₀ʰ |∇W| dr = W_spiky(0) = 15 / (π h³) (Simpson, 2000 panels)
    let panels = 2000usize;
    let dr = hf / panels as f64;
    let mut sum = 0.0f64;
    for i in 0..=panels {
        let r = i as f64 * dr;
        let weight = if i == 0 || i == panels {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * f64::from(spiky_grad(r as f32, h));
    }
    sum *= dr / 3.0;
    let peak = 15.0 / (PI * hf.powi(3));
    assert!(rel_err(sum, peak) < 1e-4, "W_spiky(0) = {sum} vs {peak}");
}

/// Density summation on a cubic lattice recovers the rest density
/// (Monaghan, *Rep. Prog. Phys.* 68 (2005) §2: with `m = ρ₀ s³` the sum
/// `Σ m W` is the lattice quadrature of `ρ₀ ∫ W dV = ρ₀`; for `h/s = 2.5`
/// the quadrature error of the Poly6 kernel is below 3 %). The SDF is a
/// far-away plane so the boundary term is inactive; `dt = 0` evaluates
/// density and pressure without moving anything.
#[test]
fn sdf_sph_lattice_density_recovers_rest_density() {
    let spacing = 0.01f32;
    let rho0 = 1000.0f32;
    let config = SphConfig {
        kernel_radius: 2.5 * spacing,
        particle_mass: rho0 * spacing * spacing * spacing,
        rest_density: rho0,
        gas_stiffness: 20.0,
        viscosity: 0.0,
        gravity: [0.0, 0.0, 0.0],
        boundary_strength: 0.0,
        repel_range: 0.0,
    };
    let n = 9i32;
    let mut particles = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                particles.push(SphParticle::at_rest([
                    x as f32 * spacing,
                    y as f32 * spacing,
                    z as f32 * spacing,
                ]));
            }
        }
    }
    let far_plane = ClosureSdf::new(|_x, _y, _z| 10.0, |_x, _y, _z| (0.0, 1.0, 0.0));
    let mut solver = SphSolver::new(particles, config, &far_plane);
    let before: Vec<[f32; 3]> = solver.particles.iter().map(|p| p.position).collect();
    solver.step(0.0);
    let centre = (n * n * n / 2) as usize; // (4, 4, 4)
    let rho = f64::from(solver.particles[centre].density);
    assert!(
        rel_err(rho, f64::from(rho0)) < 0.03,
        "lattice density {rho} vs ρ₀ = {rho0} (3 % quadrature tolerance)"
    );
    // p = k (ρ − ρ₀) clamped at 0: Müller eq. (12) with the crate's clamp
    let want_p = (20.0 * (solver.particles[centre].density - rho0)).max(0.0);
    assert_eq!(solver.particles[centre].pressure, want_p);
    // dt = 0 must not move anything
    for (p, b) in solver.particles.iter().zip(&before) {
        assert_eq!(p.position, *b);
    }
}

// ============================================================================
// pressure (SDF modifier) — first-order relaxation + yield accumulation
// ============================================================================

/// `decay` multiplies by `exp(−rate·dt)` each step, so the transient
/// pressure obeys `p(t) = p₀ exp(−rate t)` (first-order relaxation, same
/// closed form as Newton cooling). Diffusion is a zero-flux Laplacian and
/// therefore conserves `Σ p` (Carslaw & Jaeger §1.9 flux balance).
#[test]
fn pressure_modifier_transient_pressure_relaxes_exponentially_and_diffusion_conserves_sum() {
    let config = PressureConfig {
        diffusion_rate: 0.0,
        decay_rate: 0.5,
        yield_threshold: 1e9,
        deformation_rate: 0.05,
        max_deformation: 1.0,
        internal_pressure: 0.0,
        expansion_rate: 0.0,
    };
    let mut m = PressureModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.pressure.data.fill(100.0);
    for _ in 0..20 {
        m.update(0.1);
    }
    let want = 100.0 * (-0.5f64 * 2.0).exp(); // 36.79
    let got = f64::from(m.pressure_at(0.0, 0.0, 0.0));
    // 20 f32 multiplications by det_math::exp(−0.05): ≤ 1e-5 relative
    assert!(rel_err(got, want) < 1e-4, "p(2 s) = {got} vs {want}");
    assert!(
        m.deformation.data.iter().all(|&d| d == 0.0),
        "below yield: no permanent deformation"
    );

    // diffusion only: Σ p conserved
    let config = PressureConfig {
        diffusion_rate: 0.2,
        decay_rate: 0.0,
        ..config
    };
    let mut m = PressureModifier::new(config, 8, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.apply_pressure_at(0.1, -0.2, 0.05, 50.0, 0.6);
    let sum0: f64 = m.pressure.data.iter().map(|&v| f64::from(v)).sum();
    assert!(sum0 > 0.0);
    let peak0 = f64::from(m.pressure.max_value());
    for _ in 0..30 {
        m.update(0.01); // r = 0.2·0.01/(2/7)² = 0.0245 ≪ 1/6, stable
    }
    let sum1: f64 = m.pressure.data.iter().map(|&v| f64::from(v)).sum();
    assert!(
        rel_err(sum1, sum0) < 1e-4,
        "Σ p changed under zero-flux diffusion: {sum0} → {sum1}"
    );
    assert!(
        f64::from(m.pressure.max_value()) < peak0,
        "diffusion lowers the peak"
    );
}

/// Above the yield threshold the crate accumulates
/// `Δd = (p − p_y)·rate·dt`, i.e. a Bingham-type overstress law
/// `ḋ = rate·⟨p − p_y⟩` (Bingham 1917), capped at `max_deformation`.
#[test]
fn pressure_modifier_yield_deformation_grows_linearly_in_overstress_and_saturates() {
    let config = PressureConfig {
        diffusion_rate: 0.0,
        decay_rate: 0.0,
        yield_threshold: 10.0,
        deformation_rate: 0.05,
        max_deformation: 1.0,
        internal_pressure: 0.0,
        expansion_rate: 0.0,
    };
    let mut m = PressureModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.pressure.data.fill(30.0); // overstress 20 → ḋ = 1.0 /s
    for step in 1..=5 {
        m.update(0.1);
        let got = f64::from(m.deformation.data[0]);
        let want = 0.1 * step as f64;
        assert!(
            (got - want).abs() < 1e-5,
            "d after {step} steps = {got} vs {want}"
        );
    }
    for _ in 0..20 {
        m.update(0.1);
    }
    assert!(
        m.deformation.data.iter().all(|&d| d == 1.0),
        "deformation must saturate exactly at max_deformation"
    );
    // modify_distance adds the dent: surface recedes by exactly d
    assert_eq!(m.modify_distance(0.0, 0.0, 0.0, 0.25), 1.25);
}

// ============================================================================
// buoyancy_zone — Archimedes
// ============================================================================

/// Archimedes: a fully submerged sphere feels `F = ρ_w g (4/3) π r³` upward
/// (Batchelor §1.4). The crate's `4π/3 ≈ 4.1888` differs from `4.18879`
/// by 2.5e-6, hence the 1e-5 tolerance.
#[test]
fn buoyancy_zone_fully_submerged_sphere_matches_archimedes() {
    let pool = ZoneShape::Aabb {
        min: Vec3Fix::from_int(-10, -10, -10),
        max: Vec3Fix::from_int(10, 0, 10),
    };
    let zone = BuoyancyZone::water_pool(pool);
    let r = Fix128::ONE;
    let body = RigidBody::new(Vec3Fix::from_int(0, -5, 0), Fix128::from_int(10));
    let f = zone.force_on(&body, r);
    let want = 1000.0 * 9.81 * 4.0 / 3.0 * PI; // 41 092 N
    assert!(
        rel_err(f.y.to_f64(), want) < 1e-5,
        "F_b = {} vs {want}",
        f.y.to_f64()
    );
    assert_eq!(f.x, Fix128::ZERO);
    assert_eq!(f.z, Fix128::ZERO);
    // deeper submersion does not change the force (saturated fraction)
    let deeper = RigidBody::new(Vec3Fix::from_int(0, -8, 0), Fix128::from_int(10));
    assert_eq!(zone.force_on(&deeper, r), f);
    // centre-at-depth-r is the first fully-submerged position: fraction = 1
    assert_eq!(
        zone.submerged_fraction(Vec3Fix::from_int(0, -1, 0), r),
        Fix128::ONE
    );
    // 1.2.0: exact spherical cap h²(3r − h)/(4r³) — centre r above the surface
    // → 0, at r/2 above → h = r/2 → (1/4)(5/2)/4 = 5/32, at the surface → ½,
    // r/2 below → h = 3r/2 → (9/4)(3/2)/4 = 27/32; continuous through the surface
    for (y, want) in [(1.0f64, 0.0f64), (0.5, 5.0 / 32.0), (-0.5, 27.0 / 32.0)] {
        let got = zone
            .submerged_fraction(
                Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(y), Fix128::ZERO),
                r,
            )
            .to_f64();
        assert!(
            (got - want).abs() < 1e-9,
            "centre y = {y}: fraction {got}, cap {want}"
        );
    }
    let just_above = zone
        .submerged_fraction(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 1_000_000), Fix128::ZERO),
            r,
        )
        .to_f64();
    assert!(
        (just_above - 0.5).abs() < 1e-5,
        "no jump at the surface: {just_above}"
    );
    // below the pool floor there is no water
    assert_eq!(
        zone.submerged_fraction(Vec3Fix::from_int(0, -12, 0), r),
        Fix128::ZERO
    );
    // at half submersion the cap is exactly ½
    let half = zone.submerged_fraction(
        Vec3Fix::new(
            Fix128::ZERO,
            Fix128::ZERO - Fix128::from_ratio(1, 1_000_000),
            Fix128::ZERO,
        ),
        r,
    );
    assert!(
        (half.to_f64() - 0.5).abs() < 1e-5,
        "fraction just below the surface = {}",
        half.to_f64()
    );
    // monotone non-decreasing in depth
    let mut prev = Fix128::ZERO;
    for milli in 0..=2000i64 {
        let y = Fix128::ZERO - Fix128::from_ratio(milli, 1000);
        let frac = zone.submerged_fraction(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO), r);
        assert!(
            frac >= prev,
            "fraction decreased at depth {}",
            milli as f64 / 1000.0
        );
        prev = frac;
    }
}

// ============================================================================
// compressible — Anderson, Modern Compressible Flow
// ============================================================================

/// Anderson eq. (3.18): `a = √(γ R T)`; ISA sea level 288.15 K → 340.3 m/s
/// (Anderson App. A uses R = 287.05; with the crate's R = 287, 340.26).
/// ISA density 1.2250 kg/m³ at 101 325 Pa (ICAO Doc 7488).
#[test]
fn compressible_ideal_gas_speed_of_sound_and_isa_density() {
    let air = IdealGas::air();
    let t = Fix128::from_ratio(28_815, 100);
    let a = air.speed_of_sound(t).to_f64();
    let want = (1.4f64 * 287.0 * 288.15).sqrt();
    assert!(rel_err(a, want) < 1e-9, "a = {a} vs {want}");
    assert!(
        rel_err(a, 340.3) < 1e-3,
        "ISA sea-level a = {a} vs 340.3 m/s"
    );
    let rho = air.density(Fix128::from_int(101_325), t).to_f64();
    assert!(rel_err(rho, 1.2250) < 1e-3, "ISA ρ = {rho} vs 1.2250");
    // a = √(γ p / ρ) must agree with √(γ R T)
    let a_pd = air
        .speed_of_sound_from_pd(
            Fix128::from_int(101_325),
            air.density(Fix128::from_int(101_325), t),
        )
        .to_f64();
    assert!(rel_err(a_pd, a) < 1e-9);
    // helium: γ = 5/3, R = 2077 → a(293.15) = 1007 m/s (CRC Handbook: 1007)
    let he = IdealGas::helium()
        .speed_of_sound(Fix128::from_ratio(29_315, 100))
        .to_f64();
    assert!(rel_err(he, 1007.0) < 2e-3, "a_He = {he}");
    // Mach: u = a → M = 1 exactly up to the division rounding
    let m = air.mach_number(air.speed_of_sound(t), t).to_f64();
    assert!((m - 1.0).abs() < 1e-15, "M = {m}");
}

/// Anderson Table A.2 (normal shock, γ = 1.4) at M₁ = 2:
/// p₂/p₁ = 4.500, ρ₂/ρ₁ = 2.6667, T₂/T₁ = 1.6875, M₂ = 0.57735.
#[test]
fn compressible_normal_shock_matches_anderson_table_a2() {
    let air = IdealGas::air();
    let s = normal_shock_jump(&air, Fix128::from_int(2));
    assert!(rel_err(s.pressure_ratio.to_f64(), 4.5) < 1e-12);
    assert!(rel_err(s.density_ratio.to_f64(), 8.0 / 3.0) < 1e-12);
    assert!(rel_err(s.temperature_ratio.to_f64(), 1.6875) < 1e-12);
    assert!(rel_err(s.mach_downstream.to_f64(), (1.0f64 / 3.0).sqrt()) < 1e-12);
    // M₁ = 3: p₂/p₁ = 10.333, ρ₂/ρ₁ = 3.857, T₂/T₁ = 2.679, M₂ = 0.4752
    let s3 = normal_shock_jump(&air, Fix128::from_int(3));
    assert!(rel_err(s3.pressure_ratio.to_f64(), 31.0 / 3.0) < 1e-12);
    assert!(rel_err(s3.density_ratio.to_f64(), 27.0 / 7.0) < 1e-12);
    assert!(rel_err(s3.temperature_ratio.to_f64(), 2.6790) < 1e-4);
    assert!(rel_err(s3.mach_downstream.to_f64(), 0.4752) < 1e-4);
    // entropy condition: a shock always compresses (Anderson §3.6)
    assert!(s.pressure_ratio > Fix128::ONE && s.density_ratio > Fix128::ONE);
    assert!(s.mach_downstream < Fix128::ONE);
    // strong-shock limit ρ₂/ρ₁ → (γ+1)/(γ−1) = 6
    let s_inf = normal_shock_jump(&air, Fix128::from_int(1000));
    assert!((s_inf.density_ratio.to_f64() - 6.0).abs() < 1e-4);
}

/// Anderson eq. (3.28): `T₀/T = 1 + (γ−1)/2 M²` (Table A.1: 1.2 at M = 1,
/// 1.8 at M = 2). The pressure ratio `(T₀/T)^{γ/(γ−1)}` (eq. 3.30) is
/// implemented with the integer exponent 4 instead of 3.5 — see the
/// report; only the invariants that survive that substitution are pinned.
#[test]
fn compressible_isentropic_stagnation_relations() {
    let air = IdealGas::air();
    assert!(rel_err(stagnation_temp_ratio(&air, Fix128::ONE).to_f64(), 1.2) < 1e-12);
    assert!(
        rel_err(
            stagnation_temp_ratio(&air, Fix128::from_int(2)).to_f64(),
            1.8
        ) < 1e-12
    );
    assert_eq!(stagnation_temp_ratio(&air, Fix128::ZERO), Fix128::ONE);
    assert_eq!(stagnation_pressure_ratio(&air, Fix128::ZERO), Fix128::ONE);
    let p_half = stagnation_pressure_ratio(&air, Fix128::from_ratio(1, 2)).to_f64();
    let p_one = stagnation_pressure_ratio(&air, Fix128::ONE).to_f64();
    assert!(p_one > p_half && p_half > 1.0, "p₀/p must grow with M");
    // Anderson Table A.1: p₀/p = 1.1862 at M = 0.5, 1.8929 at M = 1, 7.824 at M = 2
    // (1.2.0: exact γ/(γ−1) exponent; before, (T₀/T)⁴ was 9.5 % high at M = 1)
    let exact_half = 1.05f64.powf(3.5);
    let exact_one = 1.2f64.powf(3.5);
    let exact_two = 1.8f64.powf(3.5);
    assert!(
        rel_err(p_half, exact_half) < 1e-6,
        "p₀/p(M=0.5) = {p_half} vs {exact_half}"
    );
    assert!(
        rel_err(p_one, exact_one) < 1e-6,
        "p₀/p(M=1) = {p_one} vs {exact_one}"
    );
    let p_two = stagnation_pressure_ratio(&air, Fix128::from_int(2)).to_f64();
    assert!(
        rel_err(p_two, exact_two) < 1e-6,
        "p₀/p(M=2) = {p_two} vs {exact_two}"
    );
    // monatomic gas: γ = 5/3 → exponent 2.5 (the old fixed 4 was 22 % high at M = 1)
    let helium = IdealGas {
        gas_constant: Fix128::from_int(2077),
        gamma: Fix128::from_ratio(5, 3),
    };
    let he_one = stagnation_pressure_ratio(&helium, Fix128::ONE).to_f64();
    let he_exact = (1.0 + (5.0 / 3.0 - 1.0) / 2.0f64).powf(2.5);
    assert!(
        rel_err(he_one, he_exact) < 1e-6,
        "He p₀/p(M=1) = {he_one} vs {he_exact}"
    );
    // Riemann invariants J± = u ± 2a/(γ−1) (Anderson eq. 7.66)
    let (jp, jm) = riemann_invariants(&air, Fix128::from_int(100), Fix128::from_int(340));
    assert!((jp.to_f64() - (100.0 + 2.0 * 340.0 / 0.4)).abs() < 1e-9);
    assert!((jm.to_f64() - (100.0 - 2.0 * 340.0 / 0.4)).abs() < 1e-9);
}

// ============================================================================
// non_newtonian — Chhabra & Richardson, Bird–Stewart–Lightfoot
// ============================================================================

/// Bingham 1917 / Chhabra & Richardson eq. (1.16): `τ = τ_y + μ_p γ̇`
/// above yield; Herschel–Bulkley `τ = τ_y + K γ̇ⁿ`; Ostwald–de Waele
/// `τ = K γ̇ⁿ` with apparent viscosity `η = K γ̇ⁿ⁻¹` (BSL Table 8.3-1).
/// Worked example: drilling mud τ_y = 10 Pa, μ_p = 0.02 Pa·s at γ̇ = 500 /s
/// → τ = 20 Pa (Chhabra & Richardson Example 1.1 form).
#[test]
fn non_newtonian_bingham_herschel_bulkley_and_power_law_closed_forms() {
    let mud = Bingham {
        yield_stress: Fix128::from_int(10),
        plastic_viscosity: Fix128::from_ratio(2, 100),
    };
    assert!(fix_close(
        mud.stress(Fix128::from_int(500)),
        Fix128::from_int(20)
    ));
    assert_eq!(
        mud.stress(Fix128::ZERO),
        Fix128::ZERO,
        "no flow below yield"
    );
    assert!(!mud.flows_under_stress(Fix128::from_int(10)));
    assert!(mud.flows_under_stress(Fix128::from_ratio(1001, 100)));
    // Bingham apparent viscosity τ/γ̇ = μ_p + τ_y/γ̇ decreases with γ̇
    let eta_1 = mud.stress(Fix128::from_int(1)).to_f64();
    let eta_100 = mud.stress(Fix128::from_int(100)).to_f64() / 100.0;
    assert!((eta_1 - 10.02).abs() < 1e-12 && (eta_100 - 0.12).abs() < 1e-12);

    let hb = HerschelBulkley {
        yield_stress: Fix128::from_int(5),
        k: Fix128::from_ratio(3, 10),
        n_int: 2,
    };
    // τ = 5 + 0.3·4² = 9.8
    assert!((hb.stress(Fix128::from_int(4)).to_f64() - 9.8).abs() < 1e-15);

    let dilatant = PowerLaw::shear_thickening(Fix128::from_ratio(1, 2), 2);
    // τ = 0.5·γ̇², η = 0.5·γ̇ (cornstarch-like, n = 2)
    assert_eq!(dilatant.stress(Fix128::from_int(6)), Fix128::from_int(18));
    assert_eq!(
        dilatant.apparent_viscosity(Fix128::from_int(6)),
        Fix128::from_int(3)
    );
    let newton = PowerLaw::newtonian(Fix128::from_ratio(1, 1000)); // water
    assert_eq!(
        newton.apparent_viscosity(Fix128::from_int(7)),
        Fix128::from_ratio(1, 1000)
    );
    // shear-thinning Ostwald–de Waele τ = K γ̇^(1/n) (Chhabra & Richardson eq. 1.4,
    // flow index 1/n < 1): K = 2, n = 2 → τ(4) = 4, τ(9) = 6, τ(16) = 8 — the stress
    // still GROWS with shear rate while the apparent viscosity K γ̇^(1/n − 1) falls
    // (1.2.0: the previous code returned K γ̇^(1−n), a decreasing flow curve)
    let thinning = PowerLaw::shear_thinning(Fix128::from_int(2), 2);
    for (g, want) in [(4i64, 4.0f64), (9, 6.0), (16, 8.0), (100, 20.0)] {
        let tau = thinning.stress(Fix128::from_int(g)).to_f64();
        assert!((tau - want).abs() < 1e-6, "τ({g}) = {tau}, want {want}");
    }
    let eta4 = thinning.apparent_viscosity(Fix128::from_int(4)).to_f64();
    let eta16 = thinning.apparent_viscosity(Fix128::from_int(16)).to_f64();
    assert!(
        (eta4 - 1.0).abs() < 1e-6 && (eta16 - 0.5).abs() < 1e-6,
        "η(4) = {eta4}, η(16) = {eta16}"
    );
    // flow curves are monotone increasing in γ̇ (Chhabra & Richardson §1.3)
    let mut prev = Fix128::ZERO;
    for g in 1..=50i64 {
        let gd = Fix128::from_int(g);
        let tau = mud.stress(gd)
            + hb.stress(gd)
            + dilatant.stress(gd)
            + newton.stress(gd)
            + thinning.stress(gd);
        assert!(tau > prev);
        prev = tau;
    }
}

/// Carreau (Bird, Armstrong & Hassager Vol. 1 eq. 4.1-9):
/// `η = η_∞ + (η₀ − η_∞)(1 + (λ γ̇)²)^{(n−1)/2}`. With the crate's integer
/// half-exponent −1 this is the `n = −1` member: `η₀` at rest, `η_∞` at
/// infinite shear, and exactly the midpoint at `λ γ̇ = 1`.
#[test]
fn non_newtonian_carreau_limits_and_midpoint() {
    let melt = Carreau {
        eta_zero: Fix128::from_int(1000),
        eta_inf: Fix128::from_int(10),
        lambda: Fix128::from_ratio(1, 10),
        half_exponent: -1,
    };
    assert_eq!(melt.viscosity(Fix128::ZERO), Fix128::from_int(1000));
    // λγ̇ = 1 → midpoint 505; λγ̇ = 3 → 10 + 990/10 = 109
    assert!(fix_close(
        melt.viscosity(Fix128::from_int(10)),
        Fix128::from_int(505)
    ));
    assert!(fix_close(
        melt.viscosity(Fix128::from_int(30)),
        Fix128::from_int(109)
    ));
    let high = melt.viscosity(Fix128::from_int(100_000)).to_f64();
    assert!((high - 10.0).abs() < 1e-5, "η(∞) → η_∞: {high}");
    // shear thinning: monotone non-increasing
    let mut prev = melt.viscosity(Fix128::ZERO);
    for g in 1..=40i64 {
        let eta = melt.viscosity(Fix128::from_int(g));
        assert!(eta <= prev);
        prev = eta;
    }
}

/// Fractional flow index through `viscosity_with_index`: the polymer-melt
/// `n = 0.4` (Bird et al. Table 4.1-2 order of magnitude) against the f64
/// closed form over four decades of shear rate, and consistency with the
/// integer path at `n = −1` / `n = 3`.
#[test]
fn non_newtonian_carreau_fractional_index_matches_closed_form() {
    let melt = Carreau {
        eta_zero: Fix128::from_int(1000),
        eta_inf: Fix128::from_int(10),
        lambda: Fix128::from_ratio(1, 10),
        half_exponent: -1,
    };
    let n = Fix128::from_ratio(2, 5); // 0.4
    for &gamma in &[0.0f64, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0] {
        let got = melt
            .viscosity_with_index(Fix128::from_f64(gamma), n)
            .to_f64();
        let want = 10.0 + 990.0 * (1.0 + (0.1 * gamma).powi(2)).powf((0.4 - 1.0) / 2.0);
        assert!(
            rel_err(got, want) < 1e-5,
            "Carreau n = 0.4 at γ̇ = {gamma}: {got} vs {want}"
        );
    }
    // n = 1 is Newtonian at every shear rate
    assert!(fix_close(
        melt.viscosity_with_index(Fix128::from_int(500), Fix128::ONE),
        Fix128::from_int(1000)
    ));
    // integer members agree with the repeated-multiplication path
    for &gamma in &[0i64, 3, 10, 30] {
        let g = Fix128::from_int(gamma);
        assert!(
            fix_close(
                melt.viscosity_with_index(g, Fix128::from_int(-1)),
                melt.viscosity(g)
            ),
            "n = −1 mismatch at γ̇ = {gamma}"
        );
        let thick = Carreau {
            half_exponent: 1,
            ..melt
        };
        assert!(
            fix_close(
                thick.viscosity_with_index(g, Fix128::from_int(3)),
                thick.viscosity(g)
            ),
            "n = 3 mismatch at γ̇ = {gamma}"
        );
    }
}

// ============================================================================
// surface_tension_csf — Young–Laplace through the CSF band
// ============================================================================

/// Brackbill, Kothe & Zemach 1992 eq. (9)–(10): the CSF body force
/// `f = σ κ n̂ δ(φ)` integrates across the interface band to the surface
/// pressure jump `σ κ`, which for a sphere is the Young–Laplace
/// `Δp = 2σ/R` (de Gennes et al. §1.1). On the x axis through the centre
/// the band cells at `φ = −dx, 0, +dx` carry `δ = ¼, ½, ¼ (1/dx)` for
/// `ε = 2dx` (so `Σ δ dx = 1` exactly) and curvatures `2/(R∓dx), 2/R`;
/// their weighted sum is `2/R · 1.008`, hence the 2 % tolerance.
#[test]
fn surface_tension_csf_band_integral_matches_young_laplace() {
    let n = 24usize;
    let dx = Fix128::from_ratio(1, 16);
    let mut phi = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    let c = Fix128::from_int(12) * dx;
    let radius = Fix128::from_int(8) * dx; // 0.5 m
    initialize_level_set_sphere(&mut phi, c, c, c, radius);
    assert_eq!(
        phi.get(20, 12, 12),
        Fix128::ZERO,
        "cell (20,12,12) is on the interface"
    );
    let sigma = Fix128::from_ratio(72, 1000);
    let eps = dx.double();
    let mut jump = Fix128::ZERO;
    for i in 19..=21 {
        let f = csf_body_force(&phi, i, 12, 12, sigma, eps);
        assert_eq!(f.y, Fix128::ZERO, "force is radial (x) on the x axis");
        assert_eq!(f.z, Fix128::ZERO);
        jump = jump + f.x.abs() * dx;
    }
    assert!(
        csf_body_force(&phi, 17, 12, 12, sigma, eps) == Vec3Fix::ZERO,
        "no force outside the smeared band"
    );
    let want = 2.0 * 0.072 / 0.5; // 0.288 Pa
    assert!(
        rel_err(jump.to_f64(), want) < 0.02,
        "∫ f·dn = {} vs Young–Laplace 2σ/R = {want}",
        jump.to_f64()
    );
    // the interface normal on the axis is exactly ±x̂
    let nrm = interface_normal(&phi, 20, 12, 12);
    assert_eq!(nrm, Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO));
    // the water/air preset is 72 mN/m (CRC Handbook: 72.75 mN/m at 20 °C)
    assert!(rel_err(SIGMA_WATER_AIR.to_f64(), 0.07275) < 0.011);
}

/// The smeared delta `δ = (1 − |φ|/ε)/ε` on `|φ| < ε` is a hat of unit
/// area: `∫ δ dφ = 1` (Brackbill 1992 eq. (10) requirement).
#[test]
fn surface_tension_csf_smeared_delta_has_unit_area() {
    let eps = Fix128::from_ratio(3, 2);
    let steps = 3000i64;
    let h = eps.double() / Fix128::from_int(steps);
    let mut area = Fix128::ZERO;
    for s in 0..steps {
        let phi = Fix128::ZERO - eps + h * (Fix128::from_int(s) + Fix128::from_ratio(1, 2));
        area = area + smeared_delta(phi, eps) * h;
    }
    // midpoint rule is exact on each linear piece of the hat
    assert!(
        (area.to_f64() - 1.0).abs() < 1e-12,
        "∫ δ = {}",
        area.to_f64()
    );
    assert_eq!(smeared_delta(eps, eps), Fix128::ZERO);
    assert_eq!(smeared_delta(Fix128::ZERO, eps), Fix128::ONE / eps);
}

// ============================================================================
// phase_change — enthalpy method (Stefan plateau + enthalpy conservation)
// ============================================================================

/// Enthalpy method (Voller & Cross 1981): with `c_p = 1` the cell enthalpy
/// is `H = T + latent`; heating a solid past `T_m` holds `T = T_m` while
/// the latent buffer fills to `L_f`, then the surplus reappears as
/// temperature. Step-by-step closed form for a single isolated cell heated
/// by `q` per step: `T = T_m`, `latent = k·q − (T_m − T_0)` on the plateau,
/// and `T = T_0 + k·q − L_f` once `latent = L_f` (Carslaw & Jaeger ch. XI,
/// lumped limit). The same run must not depend on the step size.
#[test]
fn phase_change_enthalpy_plateau_and_conservation() {
    let config = PhaseChangeConfig {
        melt_temperature: 200.0,
        boil_temperature: 500.0,
        latent_heat_fusion: 50.0,
        latent_heat_vaporization: 100.0,
        diffusion_rate: 0.0,
        ambient_temperature: 190.0,
        cooling_rate: 0.0,
        liquid_flow_speed: 0.0,
        gas_expansion_rate: 0.0,
        gas_dissipation_rate: 0.0,
        max_offset: 3.0,
    };
    for &dt in &[0.1f32, 0.025] {
        let mut m = PhaseChangeModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        // heat 5 K per step: 2 steps to reach T_m, 10 more on the plateau
        for step in 1..=20 {
            for v in &mut m.temperature.data {
                *v += 5.0;
            }
            m.update(dt);
            let (t, lh) = (m.temperature.data[0], m.latent_heat.data[0]);
            let supplied = 190.0 + 5.0 * step as f32;
            assert!(
                (t + lh - supplied).abs() < 1e-3,
                "dt {dt} step {step}: enthalpy T + L = {} vs {supplied}",
                t + lh
            );
            if supplied <= 200.0 {
                assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Solid);
                assert_eq!(lh, 0.0);
            } else if supplied < 250.0 {
                assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Solid, "step {step}");
                assert_eq!(t, 200.0, "plateau at T_m, step {step}");
                assert_eq!(lh, supplied - 200.0);
            } else {
                assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid, "step {step}");
                assert_eq!(lh, 50.0);
                assert_eq!(t, supplied - 50.0, "surplus returns to T, step {step}");
            }
        }
    }
    // a big superheat melts in one step and the surplus above L_f survives
    let mut m = PhaseChangeModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.temperature.data.fill(330.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid);
    assert_eq!(m.temperature.data[0], 280.0);
    // boil: 500 + 100 needed; 560 → plateau at 500 with 60 stored
    m.temperature.data.fill(560.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid);
    assert_eq!(m.temperature.data[0], 500.0);
    assert_eq!(m.latent_heat.data[0], 110.0);
    m.temperature.data.fill(540.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Gas, "buffer full → gas");
    assert_eq!(m.temperature.data[0], 500.0);
    assert_eq!(m.latent_heat.data[0], 150.0);
    m.temperature.data.fill(510.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Gas);
    assert_eq!(m.temperature.data[0], 510.0, "gas above T_b stores nothing");
    // cooling is the mirror image: a gas at 470 releases 30 of L_v and sits
    // at 500; a liquid at 170 releases L_f on the 200 plateau, then freezes
    m.temperature.data.fill(470.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Gas);
    assert_eq!(m.temperature.data[0], 500.0);
    assert_eq!(m.latent_heat.data[0], 120.0);
    m.temperature.data.fill(380.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid, "condensed");
    assert_eq!(m.temperature.data[0], 450.0);
    assert_eq!(m.latent_heat.data[0], 50.0);
    m.temperature.data.fill(170.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid, "freezing plateau");
    assert_eq!(m.temperature.data[0], 200.0);
    assert_eq!(m.latent_heat.data[0], 20.0);
    m.temperature.data.fill(170.0);
    m.update(0.1);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Solid, "frozen");
    assert_eq!(m.temperature.data[0], 190.0);
    assert_eq!(m.latent_heat.data[0], 0.0);
}

/// Liquid gravity flow moves SDF offset down a column without creating or
/// destroying it: `Σ offset` over the column is unchanged by the transfer
/// (only the gas / liquid source terms add material, both zero here).
#[test]
fn phase_change_liquid_flow_conserves_column_offset() {
    let config = PhaseChangeConfig {
        melt_temperature: 200.0,
        boil_temperature: 500.0,
        latent_heat_fusion: 0.0,
        latent_heat_vaporization: 0.0,
        diffusion_rate: 0.0,
        ambient_temperature: 300.0, // everything liquid
        cooling_rate: 0.0,
        liquid_flow_speed: 2.0,
        gas_expansion_rate: 0.0,
        gas_dissipation_rate: 0.0,
        max_offset: 3.0,
    };
    let mut m = PhaseChangeModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.update(0.0); // classify all cells as liquid without moving anything
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid);
    // column x = 0, z = 0: top cell carries 1.0 of offset
    let top = m.sdf_offset.index(0, 3, 0);
    m.sdf_offset.data[top] = 1.0;
    let column_sum = |m: &PhaseChangeModifier| -> f64 {
        (0..4)
            .map(|iy| f64::from(m.sdf_offset.data[m.sdf_offset.index(0, iy, 0)]))
            .sum()
    };
    // liquid softening adds 0.1 · flow · dt per liquid cell per step; the
    // transfer itself must add nothing on top of that
    let dt = 0.05f32;
    let softening_per_step = f64::from(0.1 * 2.0 * dt) * 4.0;
    for step in 1..=40 {
        m.update(dt);
        let want = 1.0 + softening_per_step * step as f64;
        let got = column_sum(&m);
        assert!(
            (got - want).abs() < 1e-4,
            "step {step}: column offset {got} vs {want}"
        );
    }
    // and the material actually moved down: the bottom cell holds more than
    // it could have gained from softening alone
    let bottom = m.sdf_offset.data[m.sdf_offset.index(0, 0, 0)];
    assert!(
        f64::from(bottom) > softening_per_step / 4.0 * 40.0 + 0.5,
        "bottom cell {bottom}"
    );
}

// ============================================================================
// thermal (SDF modifier) — Newton cooling + 1-D conduction eigenmode
// ============================================================================

/// Newton's law of cooling: `T − T_∞ = (T₀ − T_∞) exp(−h t)` (Incropera &
/// DeWitt §5.2 lumped capacitance, with the crate's `cooling_rate` as
/// `h A / (ρ c V)`). Uniform field so diffusion is inert; 100 steps of
/// `det_math::exp(−0.01)` in f32 → ≤ 1e-5 relative.
#[test]
fn thermal_modifier_uniform_field_obeys_newton_cooling() {
    let config = ThermalConfig {
        diffusion_rate: 0.5,
        ambient_temperature: 20.0,
        cooling_rate: 0.1,
        melt_temperature: 1e9,
        melt_rate: 0.0,
        droop_strength: 0.0,
        expansion_coefficient: 0.0,
        freeze_temperature: -1e9,
        freeze_rate: 0.0,
    };
    let mut m = ThermalModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    m.temperature.data.fill(120.0);
    for _ in 0..100 {
        m.update(0.1);
    }
    let want = 20.0 + 100.0 * (-1.0f64).exp(); // 56.79 K
    let got = f64::from(m.temperature_at(0.0, 0.0, 0.0));
    assert!((got - want).abs() < 1e-3, "T(10 s) = {got} vs {want}");
    assert!(
        m.melt_accumulator.data.iter().all(|&v| v == 0.0),
        "below melt: no recession"
    );
}

/// 1-D conduction eigenmode through the modifier's `diffuse` (zero-flux
/// mirror at the boundary cells, cell-centred finite volume with nodes at
/// `x_i = i·Δ`, domain `L = n·Δ`): `T = T_∞ + A cos(π (i+½)/n)` decays as
/// `exp(−α π² t / L²)` (Carslaw & Jaeger §3.4). Cooling is switched off so
/// only conduction acts; `r = α dt / Δ² = 0.2` keeps the explicit step
/// stable for a field uniform in y and z.
#[test]
fn thermal_modifier_cosine_mode_decays_at_the_conduction_rate() {
    let n = 16usize;
    let alpha = 0.5f64;
    let config = ThermalConfig {
        diffusion_rate: alpha as f32,
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        melt_temperature: 1e9,
        melt_rate: 0.0,
        droop_strength: 0.0,
        expansion_coefficient: 0.0,
        freeze_temperature: -1e9,
        freeze_rate: 0.0,
    };
    let mut m = ThermalModifier::new(config, n, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    let cell = 1.0f64 / (n as f64 - 1.0); // ScalarField3D: (max − min)/(n − 1)
    let l = n as f64 * cell;
    let amp = 50.0f64;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let t = 20.0 + amp * (PI * (ix as f64 + 0.5) / n as f64).cos();
                let idx = m.temperature.index(ix, iy, iz);
                m.temperature.data[idx] = t as f32;
            }
        }
    }
    let dt = 0.2 * cell * cell / alpha; // r = 0.2
    let steps = 50;
    for _ in 0..steps {
        m.update(dt as f32);
    }
    let t = steps as f64 * dt;
    let decay = (-alpha * PI * PI * t / (l * l)).exp(); // 0.68
    let mut worst = 0.0f64;
    let mut mean = 0.0f64;
    for ix in 0..n {
        let idx = m.temperature.index(ix, 7, 7);
        let got = f64::from(m.temperature.data[idx]);
        let want = 20.0 + amp * (PI * (ix as f64 + 0.5) / n as f64).cos() * decay;
        worst = worst.max((got - want).abs());
        mean += got;
    }
    mean /= n as f64;
    // discrete vs continuous decay rate differ by 0.4 % of the exponent → < 0.5 K on 50 K
    assert!(worst < 0.5, "conduction mode max error {worst} K");
    assert!(
        (mean - 20.0).abs() < 0.02,
        "zero-flux walls must conserve energy: {mean}"
    );
}

// ============================================================================
// wind_zone — quadratic drag law
// ============================================================================

/// `F = ½ ρ C_d A |v_rel| v_rel` (White, *Fluid Mechanics* eq. 7.60 with
/// the drag coefficient defined on the projected area). Light-breeze
/// preset at t = 0 (gust `sin 0 = 0`): 3 m/s, ρ = 1.225, C_d = 1.2,
/// A = 0.5 m² → 3.3075 N. Galilean invariance: a body moving with the
/// wind feels nothing.
#[test]
fn wind_zone_force_matches_quadratic_drag_law() {
    let box_shape = ZoneShape::Aabb {
        min: Vec3Fix::from_int(-10, -10, -10),
        max: Vec3Fix::from_int(10, 10, 10),
    };
    let zone = WindZone::light_breeze(box_shape);
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let f = zone.force_on(&body, Fix128::ZERO);
    let want = 0.5 * 1.225 * 9.0 * 1.2 * 0.5;
    assert!(
        rel_err(f.x.to_f64(), want) < 1e-12,
        "F = {} vs {want}",
        f.x.to_f64()
    );
    assert_eq!(f.y, Fix128::ZERO);
    assert_eq!(f.z, Fix128::ZERO);
    // quadratic: storm preset (20 m/s) → (20/3)² × breeze force
    let storm = WindZone::storm(box_shape).force_on(&body, Fix128::ZERO);
    assert!(rel_err(storm.x.to_f64(), want * (20.0f64 / 3.0).powi(2)) < 1e-12);
    // gust peak at t = 1/(4 f): 3 + 0.5 = 3.5 m/s (CORDIC sine ≈ 1e-9)
    let quarter = Fix128::ONE / (Fix128::from_int(4) * zone.gust_frequency_hz);
    let peak = zone.force_on(&body, quarter);
    assert!(rel_err(peak.x.to_f64(), 0.5 * 1.225 * 12.25 * 0.6) < 1e-7);
    // co-moving body: v_rel = 0 → F = 0 exactly
    let mut riding = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    riding.velocity = zone.instantaneous_wind_vector(Fix128::ZERO);
    assert_eq!(zone.force_on(&riding, Fix128::ZERO), Vec3Fix::ZERO);
    // headwind: body at +3 m/s into 3 m/s wind → v_rel = 6 → 4× and opposing
    let mut head = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    head.velocity = Vec3Fix::from_int(-3, 0, 0);
    let fh = zone.force_on(&head, Fix128::ZERO);
    assert!(rel_err(fh.x.to_f64(), 4.0 * want) < 1e-12);
    // outside the zone: nothing
    let far = RigidBody::new(Vec3Fix::from_int(50, 0, 0), Fix128::ONE);
    assert_eq!(zone.force_on(&far, Fix128::ZERO), Vec3Fix::ZERO);
}

// ============================================================================
// sdf_wind_field — empirical shelter ramp (validation: none)
// ============================================================================

/// No closed form: the shelter factor `clamp(d / decay_scale, 0, 1)` is
/// a heuristic (the atmospheric surface layer follows the log law
/// `u = (u*/κ) ln(z/z₀)`, Stull 1988 §9.7, which this ramp does not
/// reproduce). Pinned invariants: zero inside obstacles, linear ramp,
/// saturation at the free-stream speed, direction preserved.
#[test]
fn sdf_wind_field_shelter_ramp_invariants() {
    let ground = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let mut field = SdfWindField::new(&ground, [0.6, 0.0, 0.8], 10.0);
    field.decay_scale_m = 4.0;
    assert_eq!(field.sample([0.0, -1.0, 0.0]), [0.0, 0.0, 0.0]);
    assert_eq!(field.sample([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    let mut prev = 0.0f32;
    for k in 1..=16 {
        let y = k as f32 * 0.5;
        let v = field.sample([0.0, y, 0.0]);
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let want = 10.0 * (y / 4.0).min(1.0);
        assert!((speed - want).abs() < 1e-5, "|v|({y}) = {speed} vs {want}");
        assert!(speed >= prev, "shelter must be monotone in distance");
        assert!(
            v[1] == 0.0 && (v[0] * 0.8 - v[2] * 0.6).abs() < 1e-6,
            "direction preserved"
        );
        prev = speed;
    }
    assert!((prev - 10.0).abs() < 1e-6, "saturates at the base speed");
}

// ============================================================================
// aeroelasticity — Van der Pol limit cycle + Strouhal lock-in
// ============================================================================

/// Decoupled wake (`A = 0`): the Van der Pol oscillator
/// `q̈ + ε (q² − 1) q̇ + Ω² q = 0` has a limit cycle of amplitude 2
/// (Strogatz, *Nonlinear Dynamics and Chaos* §7.6; Facchinetti, de Langre
/// & Biolley 2004 §2.2 normalise `q = 2 C_L / C_L0` for exactly this
/// reason) and frequency `Ω_s = 2π St U / D` — the Strouhal law
/// `f = St U / D` (Blevins, *Flow-Induced Vibration* §3.2), 3.333 Hz here.
/// The body is then a linear oscillator driven at Ω_s whose steady
/// amplitude is `F₀ / √((ω_n² − Ω²)² + (2ζω_nΩ)²)` (Rao, *Mechanical
/// Vibrations* eq. 3.29). Forward Euler at `Ω dt = 4e-4` adds ~1.5 % to
/// the limit-cycle amplitude, hence the 3 % tolerances. 1.2.0: the damping is
/// Facchinetti's `ε Ω_f (q² − 1) q̇` (growth rate `ε Ω_f / 2 ≈ 3 /s` here), so the
/// cycle is established within a second; the run still measures over 35–40 s.
#[test]
fn aeroelasticity_van_der_pol_amplitude_and_strouhal_frequency() {
    let params = VivParameters {
        wake_coupling_a: 0.0,
        ..VivParameters::facchinetti_reference()
    };
    let dt = 2e-5f32;
    let total = 40.0f32;
    let steps = (total / dt) as usize;
    let mut state = VivState {
        wake_q: 1.0,
        ..VivState::seeded()
    };
    let mut q_peak = 0.0f32;
    let mut y_peak = 0.0f32;
    let mut crossings = 0usize;
    let mut prev_q = state.wake_q;
    let window_start = (35.0 / dt) as usize; // measure over the last 5 s
    let count_start = (30.0 / dt) as usize; // count zero crossings over 10 s
    for n in 0..steps {
        viv_step(&mut state, &params, dt);
        if n >= window_start {
            q_peak = q_peak.max(state.wake_q.abs());
            y_peak = y_peak.max(state.displacement_m.abs());
        }
        if n >= count_start && (state.wake_q > 0.0) != (prev_q > 0.0) {
            crossings += 1;
        }
        prev_q = state.wake_q;
    }
    assert!(
        (f64::from(q_peak) - 2.0).abs() < 0.06,
        "Van der Pol limit-cycle amplitude {q_peak} vs 2 (3 %)"
    );
    let f_measured = crossings as f64 / (2.0 * 10.0);
    let f_strouhal = 0.2 * 0.5 / 0.03;
    // zero-crossing count over 10 s resolves 0.05 Hz (1.5 %); the Van der Pol
    // frequency is Ω(1 − ε²/16) (Strogatz §7.6) = 0.6 % below Ω_f at ε = 0.3
    assert!(
        rel_err(f_measured, f_strouhal) < 0.025,
        "shedding frequency {f_measured} Hz vs St·U/D = {f_strouhal} Hz"
    );
    // driven body: amplitude ratio y/q equals the transfer function
    let omega = 2.0 * PI * f_strouhal;
    let wn = 24.0f64;
    let zeta = 0.02f64;
    let force_per_q = 0.3 * 1000.0 * 0.25 * 0.03 / (2.0 * 0.7);
    let transfer = force_per_q
        / ((wn * wn - omega * omega).powi(2) + (2.0 * zeta * wn * omega).powi(2)).sqrt();
    let ratio = f64::from(y_peak) / f64::from(q_peak);
    // at ε = 0.3 the Van der Pol cycle carries a ~3 % third harmonic; the body
    // (linear, near-resonant) responds to the fundamental, so y_peak / q_peak sits
    // a few % below the single-frequency transfer function — 5 % tolerance
    assert!(
        rel_err(ratio, transfer) < 0.05,
        "y/q = {ratio} vs |H(Ω)|·F/q = {transfer} (5 %)"
    );
}

// ============================================================================
// acoustic_wave — d'Alembert
// ============================================================================

/// d'Alembert (Strauss, *Partial Differential Equations* §2.1):
/// `u = ½[f(x − ct) + f(x + ct)]` for zero initial velocity. At Courant
/// number 1 the leap-frog scheme reproduces this translation exactly
/// (the "magic time step", Trefethen *Spectral Methods* / LeVeque
/// *FDM for ODEs and PDEs* §10.2), so the only error is f32 rounding.
/// The right-going half crosses `n` cells in `n` steps: travel time
/// `L / c` recovered with `dt = dx / c` from `stable_dt`.
#[test]
fn acoustic_wave_leapfrog_at_courant_one_reproduces_dalembert() {
    let n = 200usize;
    let dx = 0.01f32;
    let c = speeds::AIR_20C;
    let dt = stable_dt(dx, c);
    let courant = c * dt / dx;
    assert_eq!(courant, 1.0);
    let f = |i: i64| -> f32 {
        let x = (i - 100) as f32;
        (-(x * x) / 8.0).exp() // Gaussian, 3-cell half-width
    };
    let current: Vec<f32> = (0..n as i64).map(f).collect();
    // exact previous state at t = −dt for zero velocity: ½[f(x−c dt) + f(x+c dt)]
    let previous: Vec<f32> = (0..n as i64).map(|i| 0.5 * (f(i - 1) + f(i + 1))).collect();
    let mut prev = previous;
    let mut cur = current;
    let mut next = vec![0.0f32; n];
    let steps = 40i64;
    for _ in 0..steps {
        leapfrog_step(&cur, &prev, &mut next, courant);
        prev.clone_from(&cur);
        cur.clone_from(&next);
    }
    let mut worst = 0.0f32;
    for i in 0..n as i64 {
        let want = 0.5 * (f(i - steps) + f(i + steps));
        worst = worst.max((cur[i as usize] - want).abs());
    }
    assert!(
        worst < 1e-5,
        "d'Alembert max error {worst} after {steps} steps"
    );
    // the right-going crest is at x₀ + c t = 100 + 40 cells
    let (argmax, _) = cur
        .iter()
        .enumerate()
        .skip(101)
        .fold(
            (0usize, 0.0f32),
            |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc },
        );
    assert_eq!(argmax, 140, "crest travelled c·t = {} cells", steps);
    assert!(
        (cur[140] - 0.5).abs() < 1e-5,
        "each half carries ½ of the pulse"
    );
}

/// `c = √(K/ρ)` for a fluid, `√(γ p/ρ)` for an ideal gas (Kinsler & Frey,
/// *Fundamentals of Acoustics* §5.6 and Table A.10). Presets: air 343 m/s
/// at 20 °C (√(1.4·101325/1.204) = 343.2); water 1497 m/s at 25 °C
/// (K = 2.24 GPa, ρ = 997 → 1499).
#[test]
fn acoustic_wave_presets_match_bulk_modulus_formula() {
    let air = (1.4f64 * 101_325.0 / 1.204).sqrt();
    assert!(rel_err(f64::from(speeds::AIR_20C), air) < 1e-3, "air {air}");
    let water = (2.24e9f64 / 997.0).sqrt();
    assert!(
        rel_err(f64::from(speeds::WATER_25C), water) < 2e-3,
        "water {water}"
    );
    // steel longitudinal (bulk) wave √((K + 4G/3)/ρ) with K = 160, G = 80 GPa,
    // ρ = 7850 → 5794–5960 depending on alloy (Kinsler & Frey Table A.10: 5960)
    let steel = ((160e9f64 + 4.0 * 80e9 / 3.0) / 7850.0).sqrt();
    assert!(
        rel_err(f64::from(speeds::STEEL_LONGITUDINAL), steel) < 0.03,
        "steel {steel}"
    );
}

// ============================================================================
// electromagnetic — Coulomb / Lorentz / dipole (Griffiths)
// ============================================================================

/// Griffiths, *Introduction to Electrodynamics*: Coulomb eq. (2.1)
/// `F = k q₁q₂ / r²` (two 1 µC charges at 1 m → 8.99 mN); Lorentz
/// eq. (5.1) `F = q (E + v × B)` with `|v × B| = vB` for `v ⊥ B`; on-axis
/// dipole field eq. (5.89) `B = (μ₀/4π) 2m/r³`, equatorial `−(μ₀/4π) m/r³`.
#[test]
fn electromagnetic_coulomb_lorentz_and_dipole_magnitudes() {
    let micro = Fix128::from_ratio(1, 1_000_000);
    let source = EmSource::PointCharge {
        position: Vec3Fix::ZERO,
        charge_c: micro,
    };
    let mut body = RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let f = lorentz_force(ChargedBody::new(0, micro), &body, &source);
    let want = 8.99e9 * 1e-12;
    assert!(
        rel_err(f.x.to_f64(), want) < 1e-12,
        "Coulomb {} vs {want}",
        f.x.to_f64()
    );
    assert_eq!(f.y, Fix128::ZERO);
    // inverse square: at r = 2 the force is a quarter
    body.position = Vec3Fix::from_int(2, 0, 0);
    let f2 = lorentz_force(ChargedBody::new(0, micro), &body, &source);
    assert!(rel_err(f2.x.to_f64(), want / 4.0) < 1e-12);
    // like charges repel, opposite attract (sign)
    let f_neg = lorentz_force(ChargedBody::new(0, Fix128::ZERO - micro), &body, &source);
    assert!(fix_close(f_neg.x, Fix128::ZERO - f2.x));

    // Lorentz: q = 1 C, v = 3 x̂, B = 2 ẑ → F = q v × B = (0, −6, 0)
    let uniform = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::from_int(0, 0, 2),
    };
    let mut mover = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    mover.velocity = Vec3Fix::from_int(3, 0, 0);
    let fl = lorentz_force(ChargedBody::new(0, Fix128::ONE), &mover, &uniform);
    assert_eq!(fl, Vec3Fix::from_int(0, -6, 0));
    assert_eq!(
        fl.dot(mover.velocity),
        Fix128::ZERO,
        "magnetic force does no work"
    );
    // superposition of an E field: F = q(E + v × B)
    let e_and_b = EmSource::Uniform {
        electric: Vec3Fix::from_int(5, 0, 0),
        magnetic: Vec3Fix::from_int(0, 0, 2),
    };
    assert_eq!(
        lorentz_force(ChargedBody::new(0, Fix128::ONE), &mover, &e_and_b),
        Vec3Fix::from_int(5, -6, 0)
    );
    assert_eq!(
        lorentz_force_sum(
            ChargedBody::new(0, Fix128::ONE),
            &mover,
            &[uniform, e_and_b]
        ),
        Vec3Fix::from_int(5, -12, 0)
    );

    // dipole m = 1 A·m² ẑ at r = 0.1 m
    let dipole = EmSource::MagneticDipole {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::from_int(0, 0, 1),
    };
    let tenth = Fix128::from_ratio(1, 10);
    let (_, b_axis) = dipole.sample(Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, tenth));
    assert!(
        rel_err(b_axis.z.to_f64(), 2e-4) < 1e-12,
        "on-axis B = {}",
        b_axis.z.to_f64()
    );
    assert_eq!(b_axis.x, Fix128::ZERO);
    let (_, b_eq) = dipole.sample(Vec3Fix::new(tenth, Fix128::ZERO, Fix128::ZERO));
    assert!(
        rel_err(b_eq.z.to_f64(), -1e-4) < 1e-12,
        "equatorial B = {}",
        b_eq.z.to_f64()
    );
    assert_eq!(b_eq.x, Fix128::ZERO);
}

// ============================================================================
// piezoelectric — d33 constitutive relations
// ============================================================================

/// IEEE Std 176 / Uchino, *Ferroelectric Devices* §2: direct effect
/// `Q = d₃₃ F`, `V = Q/C = d₃₃ F t / (ε A)` (equivalently `V = g₃₃ σ t`
/// with `g₃₃ = d₃₃/ε`); converse blocked force `F_b = d₃₃ Y A V / t`.
/// PZT-5A (Berlincourt / Morgan data): d₃₃ = 374 pC/N, ε_r = 1700 →
/// g₃₃ = 24.8 mV·m/N, which the preset reproduces.
#[test]
fn piezoelectric_direct_and_converse_effects_match_d33_relations() {
    let eps0 = 8.854_188e-12f64;
    let (area, thick) = (1e-4f64, 2e-3f64);
    let e = PiezoElement::pzt_5a(area as f32, thick as f32);
    let d33 = 3.74e-10f64;
    let eps = 1700.0 * eps0;
    let g33 = d33 / eps;
    assert!(rel_err(g33, 24.8e-3) < 0.01, "g₃₃ = {g33}");
    let force = 10.0f64;
    let v = f64::from(e.voltage_from_force(force as f32));
    let want_v = d33 * force * thick / (eps * area); // 4.97 V
    assert!(rel_err(v, want_v) < 1e-5, "V = {v} vs {want_v}");
    assert!(
        rel_err(v, g33 * (force / area) * thick) < 1e-5,
        "V = g₃₃ σ t"
    );
    // charge form: Q = d F, C = ε A / t, V = Q / C
    let q = d33 * force;
    let cap = eps * area / thick;
    assert!(rel_err(v, q / cap) < 1e-5);
    // converse: 100 V → E = 50 kV/m → F_b = E d Y A
    let fb = f64::from(e.force_from_voltage(100.0));
    let want_f = 100.0 / thick * d33 * 61e9 * area;
    assert!(rel_err(fb, want_f) < 1e-5, "F_b = {fb} vs {want_f}");
    // round trip is the coupling factor k² = d² Y / ε < 1 (passivity)
    let k_sq = f64::from(e.force_from_voltage(e.voltage_from_force(1.0)));
    assert!(rel_err(k_sq, d33 * d33 * 61e9 / eps) < 1e-4);
    assert!(k_sq < 1.0, "k² = {k_sq} must be below 1");
    // linearity
    assert!(rel_err(f64::from(e.voltage_from_force(20.0)), 2.0 * v) < 1e-6);
    // Hooke: strain = σ / Y
    assert!(rel_err(f64::from(e.strain_under_stress(61e6)), 1e-3) < 1e-6);
    // quartz has the lowest d₃₃, PZT the highest (Uchino Table 2.2 ordering)
    let quartz = PiezoElement::quartz(area as f32, thick as f32);
    let pvdf = PiezoElement::pvdf(area as f32, thick as f32);
    assert!(quartz.d_coefficient < pvdf.d_coefficient && pvdf.d_coefficient < e.d_coefficient);
}

// ============================================================================
// turbulence — Smagorinsky for simple shear
// ============================================================================

/// Pope, *Turbulent Flows* eq. (13.127)–(13.128): `ν_t = (C_s Δ)² |S̄|`
/// with `|S̄| = (2 S̄_ij S̄_ij)^½`. Simple shear `u = γ̇ y` has
/// `S₁₂ = S₂₁ = γ̇/2` so `|S̄| = γ̇` and `ν_t = (C_s Δ)² γ̇`; isotropic
/// strain `S = diag(s, s, s)` gives `|S̄| = √6 s`.
#[test]
fn turbulence_smagorinsky_simple_shear_closed_form() {
    let gamma = Fix128::from_int(100);
    let s = strain_rate_magnitude(
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        gamma.half(),
        Fix128::ZERO,
        Fix128::ZERO,
    );
    assert!(
        (s.to_f64() - 100.0).abs() < 1e-12,
        "|S̄| = {} vs γ̇",
        s.to_f64()
    );
    let delta = Fix128::from_ratio(1, 100);
    let nu_t = smagorinsky_eddy_viscosity(delta, s).to_f64();
    let cs = SMAGORINSKY_CS.to_f64();
    assert!(rel_err(cs, 0.17) < 1e-6, "C_s = {cs}");
    let want = (cs * 0.01).powi(2) * 100.0; // 2.89e-4 m²/s
    assert!(rel_err(nu_t, want) < 1e-9, "ν_t = {nu_t} vs {want}");
    let iso = strain_rate_magnitude(
        Fix128::ONE,
        Fix128::ONE,
        Fix128::ONE,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
    );
    assert!((iso.to_f64() - 6.0f64.sqrt()).abs() < 1e-12);
    // ν_t ∝ Δ² and ∝ |S̄| (dimensional consistency of the closure)
    let nu_2d = smagorinsky_eddy_viscosity(delta.double(), s).to_f64();
    assert!(rel_err(nu_2d, 4.0 * nu_t) < 1e-12);
}

// ============================================================================
// smoke_fire — Arrhenius + Boussinesq
// ============================================================================

/// Turns, *An Introduction to Combustion* eq. (4.8): `k = A exp(−E_a/RT)`;
/// single-step rate `r = k [F][O]`. At 2000 K for the methane preset
/// `r = 1.3e9 · exp(−12.148) = 6887 kg/(m³ s)`, and the ratio between two
/// temperatures is `exp(E_a/R (1/T₁ − 1/T₂))`. Boussinesq (Incropera
/// §9.2): `f_b = ρ₀ β ΔT g` with `β = 1/T` for an ideal gas.
#[test]
fn smoke_fire_arrhenius_and_boussinesq_closed_forms() {
    let r = ArrheniusReaction::methane_air();
    let t1 = 1500.0f64;
    let t2 = 2000.0f64;
    let rate = |t: f64| {
        reaction_rate_kg_per_m3_s(&r, Fix128::from_int(t as i64), Fix128::ONE, Fix128::ONE).to_f64()
    };
    let want2 = 1.3e9 * (-202_000.0f64 / (8.314 * t2)).exp();
    assert!(
        rel_err(rate(t2), want2) < 1e-6,
        "r(2000 K) = {} vs {want2}",
        rate(t2)
    );
    let want1 = 1.3e9 * (-202_000.0f64 / (8.314 * t1)).exp();
    assert!(rel_err(rate(t1), want1) < 1e-6);
    let ratio_want = (202_000.0f64 / 8.314 * (1.0 / t1 - 1.0 / t2)).exp();
    assert!(
        rel_err(rate(t2) / rate(t1), ratio_want) < 1e-6,
        "Arrhenius ratio"
    );
    // bilinear in reactant densities
    let half = reaction_rate_kg_per_m3_s(
        &r,
        Fix128::from_int(2000),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(4),
    )
    .to_f64();
    assert!(rel_err(half, 2.0 * rate(t2)) < 1e-12);
    // heat release q̇ = r Δh_c and soot ṡ = Y_s r are exact products
    let rr = Fix128::from_int(10);
    assert_eq!(
        heat_release_j_per_m3_s(&r, rr),
        Fix128::from_int(500_000_000)
    );
    assert!(fix_close(
        soot_generation_kg_per_m3_s(&r, rr),
        Fix128::from_ratio(150, 1000)
    ));
    // Boussinesq: air ρ = 1.204, β = 1/300, ΔT = 100 K, g = 9.81 → 3.937 N/m³
    let f = boussinesq_buoyancy_n_per_m3(
        Fix128::from_ratio(1204, 1000),
        Fix128::from_ratio(1, 300),
        Fix128::from_int(100),
        Fix128::from_ratio(981, 100),
    )
    .to_f64();
    assert!(
        rel_err(f, 1.204 * 100.0 * 9.81 / 300.0) < 1e-12,
        "f_b = {f}"
    );
    // cold plume sinks: sign follows ΔT
    let cold = boussinesq_buoyancy_n_per_m3(
        Fix128::ONE,
        Fix128::from_ratio(1, 300),
        Fix128::from_int(-100),
        Fix128::from_ratio(981, 100),
    );
    assert!(cold < Fix128::ZERO);
}

// ============================================================================
// erosion — empirical rate law (validation: none)
// ============================================================================

/// No textbook closed form: erosion mass loss scales as `v^n` with
/// `n ≈ 2–3` in Finnie 1960 / Bitter 1963; the crate's `Wind` and `Water`
/// laws use `n = 1` and `Ablation` `n = 2`. Pinned: linear accumulation
/// `depth = rate (1 − hardness) v^n e t`, exact `v²` for ablation, the
/// 1.5× water factor, speed-independence of chemical attack, saturation.
#[test]
fn erosion_rate_law_invariants() {
    let run = |kind: ErosionType, speed: f32, steps: usize| -> f64 {
        let config = ErosionConfig {
            erosion_type: kind,
            rate: 0.01,
            hardness: 0.5,
            max_depth: 2.0,
            smoothing: 0.0,
            flow_direction: (1.0, 0.0, 0.0),
            flow_speed: speed,
        };
        let mut m = ErosionModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        for _ in 0..steps {
            // exposure decays with a hard-coded 5 /s each update and must be
            // re-supplied by the caller every frame (module doc)
            m.exposure.data.fill(1.0);
            m.update(0.1);
        }
        f64::from(m.erosion_at(0.0, 0.0, 0.0))
    };
    let wind = run(ErosionType::Wind, 2.0, 10);
    assert!(
        (wind - 0.01 * 0.5 * 2.0 * 1.0).abs() < 1e-6,
        "wind depth {wind}"
    );
    assert!(
        rel_err(run(ErosionType::Wind, 4.0, 10), 2.0 * wind) < 1e-5,
        "wind ∝ v"
    );
    let abl = run(ErosionType::Ablation, 2.0, 10);
    assert!(rel_err(abl, 2.0 * wind) < 1e-5, "ablation ∝ v²: {abl}");
    assert!(rel_err(run(ErosionType::Ablation, 4.0, 10), 4.0 * abl) < 1e-5);
    assert!(rel_err(run(ErosionType::Water, 2.0, 10), 1.5 * wind) < 1e-5);
    let chem = run(ErosionType::Chemical, 2.0, 10);
    assert!(
        (chem - run(ErosionType::Chemical, 9.0, 10)).abs() < 1e-9,
        "chemical ∝ v⁰"
    );
    assert!(
        rel_err(run(ErosionType::Wind, 2.0, 30), 3.0 * wind) < 1e-5,
        "linear in time"
    );
    assert!(
        (run(ErosionType::Ablation, 20.0, 100) - 2.0).abs() < 1e-6,
        "saturates at max_depth"
    );
    // harder material erodes less: (1 − h) factor
    let config = ErosionConfig {
        hardness: 0.9,
        smoothing: 0.0,
        flow_speed: 2.0,
        ..ErosionConfig::default()
    };
    let mut hard = ErosionModifier::new(config, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    for _ in 0..10 {
        hard.exposure.data.fill(1.0);
        hard.update(0.1);
    }
    assert!(rel_err(f64::from(hard.erosion_at(0.0, 0.0, 0.0)), 0.2 * wind) < 1e-4);
    // surface recedes by exactly the eroded depth
    assert!(
        (f64::from(hard.modify_distance(0.0, 0.0, 0.0, 1.0)) - (1.0 + 0.2 * wind)).abs() < 1e-6
    );
}

// ============================================================================
// wave_ship — deep-water dispersion, Froude–Krylov, SDOF response
// ============================================================================

/// Deep-water linear waves (Dean & Dalrymple, *Water Wave Mechanics*
/// eq. 3.30): `ω² = g k`, phase speed `c = g/ω`. A component built with
/// `k = ω²/g` must therefore satisfy `η(x + cΔt, t + Δt) = η(x, t)`.
/// Froude–Krylov on a vertical-walled box (Faltinsen §3.1): the
/// hydrostatic heave force is `ρ g A_wp (d + η)`.
#[test]
fn wave_ship_deep_water_dispersion_and_froude_krylov() {
    let g = 9.81f64;
    let period = 8.0f64;
    let omega = 2.0 * PI / period;
    let k = omega * omega / g;
    let comp = WaveComponent {
        amplitude_m: Fix128::from_ratio(3, 2),
        omega_rad_per_s: Fix128::from_f64(omega),
        wavenumber_rad_per_m: Fix128::from_f64(k),
        phase_rad: Fix128::from_ratio(1, 3),
    };
    let c = g / omega; // 12.49 m/s
    for (x, t) in [(0.0f64, 0.0f64), (12.5, 3.0), (-40.0, 7.25), (100.0, 11.0)] {
        let e0 = free_surface_elevation(&[comp], Fix128::from_f64(x), Fix128::from_f64(t)).to_f64();
        let e1 = free_surface_elevation(
            &[comp],
            Fix128::from_f64(x + c * 2.5),
            Fix128::from_f64(t + 2.5),
        )
        .to_f64();
        // Fix128 CORDIC cosine + from_f64 seeding ≈ 1e-9 on a 1.5 m amplitude
        assert!(
            (e0 - e1).abs() < 1e-8,
            "crest must travel at c: {e0} vs {e1}"
        );
        let want = 1.5 * (k * x - omega * t + 1.0 / 3.0).cos();
        assert!((e0 - want).abs() < 1e-8, "η({x},{t}) = {e0} vs {want}");
    }
    // superposition: two components add linearly
    let two = free_surface_elevation(&[comp, comp], Fix128::ZERO, Fix128::ONE).to_f64();
    let one = free_surface_elevation(&[comp], Fix128::ZERO, Fix128::ONE).to_f64();
    assert!((two - 2.0 * one).abs() < 1e-12);
    // wavelength λ = g T² / 2π = 99.9 m for T = 8 s (Dean & Dalrymple Table 3.1)
    assert!(rel_err(2.0 * PI / k, 99.92) < 1e-3);

    let fk = froude_krylov_vertical_n(
        Fix128::from_int(1025),
        Fix128::from_ratio(981, 100),
        Fix128::from_int(200),
        Fix128::from_int(4),
        Fix128::from_ratio(3, 2),
    )
    .to_f64();
    assert!(
        rel_err(fk, 1025.0 * 9.81 * 200.0 * 5.5) < 1e-12,
        "F_FK = {fk}"
    );
}

/// The heave equation `m z̈ + c ż + k z = F` with `c = F = 0` is a
/// harmonic oscillator of period `2π √(m/k)` (Rao §2.2); the crate
/// integrates it with symplectic Euler (velocity first), which is
/// energy-bounded, so the amplitude stays at `z₀` and the period error is
/// `O((ω dt)²) ≈ 4e-6`. With damping and a constant force the static
/// offset is `F/k`.
#[test]
fn wave_ship_heave_free_oscillation_period_and_static_offset() {
    let mass = Fix128::from_int(1_000_000);
    let stiffness = Fix128::from_int(4_000_000); // ω_n = 2 rad/s, T = π s
    let inertia = Fix128::from_int(1);
    let dt = Fix128::from_ratio(1, 1000);
    let mut ship = ShipResponse {
        heave_m: Fix128::ONE,
        ..ShipResponse::default()
    };
    let mut crossings = Vec::new();
    let mut prev = ship.heave_m;
    let mut peak = 0.0f64;
    let steps = 20_000; // 20 s ≈ 6.4 periods
    for n in 0..steps {
        ship.advance(
            Fix128::ZERO,
            Fix128::ZERO,
            mass,
            inertia,
            stiffness,
            Fix128::ONE,
            Fix128::ZERO,
            Fix128::ZERO,
            dt,
        );
        if (ship.heave_m > Fix128::ZERO) != (prev > Fix128::ZERO) {
            crossings.push(n as f64 * 1e-3);
        }
        prev = ship.heave_m;
        peak = peak.max(ship.heave_m.to_f64().abs());
    }
    assert!(
        crossings.len() >= 12,
        "expected ≥ 12 zero crossings, got {}",
        crossings.len()
    );
    let period =
        2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    assert!(
        rel_err(period, PI) < 1e-3,
        "heave period {period} vs 2π√(m/k) = π"
    );
    assert!(
        (peak - 1.0).abs() < 2e-3,
        "symplectic Euler keeps the amplitude: {peak}"
    );
    // pitch is decoupled and untouched
    assert_eq!(ship.pitch_rad, Fix128::ZERO);

    // static offset under constant force with damping: z → F / k
    let mut loaded = ShipResponse::default();
    for _ in 0..40_000 {
        loaded.advance(
            Fix128::from_int(2_000_000),
            Fix128::ZERO,
            mass,
            inertia,
            stiffness,
            Fix128::ONE,
            Fix128::from_int(4_000_000), // ζ = 1 (critical)
            Fix128::ZERO,
            dt,
        );
    }
    assert!(
        (loaded.heave_m.to_f64() - 0.5).abs() < 1e-6,
        "z_static = F/k = 0.5"
    );
}

/// Hasselmann et al. 1973 / Chakrabarti 1987 eq. (4.29) JONSWAP:
/// `S(ω) = 5/16 H_s² ω_p⁴ ω⁻⁵ exp(−5/4 (ω_p/ω)⁴) γ^r`. Checked against the
/// f64 evaluation of the same formula, its peak location, the ω⁻⁵ tail and
/// the Pierson–Moskowitz reduction at γ = 1 (1.2.0; before, the module
/// returned the tail factor only, monotone and unbounded as ω → 0).
#[test]
fn wave_ship_spectrum_is_jonswap() {
    let j = Jonswap::north_sea();
    let wp = j.peak_omega().to_f64();
    assert!(rel_err(wp, 2.0 * PI / 9.0) < 1e-12);
    let hs = j.significant_wave_height_m.to_f64();
    let gamma = j.gamma.to_f64();
    let reference = |w: f64| {
        let sigma = if w <= wp { 0.07 } else { 0.09 };
        let r = (-(w - wp).powi(2) / (2.0 * sigma * sigma * wp * wp)).exp();
        5.0 / 16.0 * hs * hs * wp.powi(4) / w.powi(5)
            * (-1.25 * (wp / w).powi(4)).exp()
            * gamma.powf(r)
    };
    for k in 1..=40 {
        let w = wp * f64::from(k) / 10.0; // 0.1 ω_p .. 4 ω_p
        let got = j.spectrum_density(Fix128::from_f64(w)).to_f64();
        let want = reference(w);
        if want < 1e-9 {
            assert!(got < 1e-8, "S({w}) = {got} should be ≈ 0 (want {want:e})");
            continue;
        }
        assert!(rel_err(got, want) < 1e-4, "S({w}) = {got} vs {want}");
    }
    // peak at ω_p: larger than either neighbour
    let s_p = j.spectrum_density(j.peak_omega()).to_f64();
    let s_lo = j.spectrum_density(Fix128::from_f64(wp * 0.9)).to_f64();
    let s_hi = j.spectrum_density(Fix128::from_f64(wp * 1.1)).to_f64();
    assert!(
        s_p > s_lo && s_p > s_hi,
        "no peak at ω_p: {s_lo} {s_p} {s_hi}"
    );
    // ω⁻⁵ tail far above the peak: S(4ω_p)/S(3ω_p) ≈ (3/4)⁵ · (cut-off ratio ≈ 1)
    let s3 = j.spectrum_density(Fix128::from_f64(3.0 * wp)).to_f64();
    let s4 = j.spectrum_density(Fix128::from_f64(4.0 * wp)).to_f64();
    assert!(
        rel_err(
            s4 / s3,
            (0.75f64).powi(5) * (-1.25 * (0.25f64.powi(4) - (1.0 / 3.0f64).powi(4))).exp()
        ) < 1e-3
    );
    // Pierson–Moskowitz reduction (γ = 1): m₀ = ∫S dω ≈ H_s²/16 (Chakrabarti eq. 4.20)
    let pm = Jonswap {
        gamma: Fix128::ONE,
        ..Jonswap::north_sea()
    };
    let (mut m0, dw) = (0.0f64, wp / 200.0);
    let mut w = dw;
    while w < 12.0 * wp {
        m0 += pm.spectrum_density(Fix128::from_f64(w)).to_f64() * dw;
        w += dw;
    }
    assert!(
        rel_err(m0, hs * hs / 16.0) < 0.02,
        "PM m₀ = {m0} vs H_s²/16 = {}",
        hs * hs / 16.0
    );
    assert_eq!(j.spectrum_density(Fix128::ZERO), Fix128::ZERO);
}

// ============================================================================
// fsi_advanced — drag, Archimedes, terminal velocity, Newton III
// ============================================================================

/// White, *Fluid Mechanics* eq. (7.60): `F_d = ½ ρ C_d A V²`; terminal
/// velocity of a falling sphere (White Ex. 7.8 form) `V_t = √(2 m g / (ρ C_d A))`
/// balances weight against drag. Buoyancy is Archimedes `ρ V g`;
/// aggregate torque is `r × F`; `react_back_pressure` deposits `−F`
/// (Newton's third law → momentum conservation of the coupled system).
#[test]
fn fsi_advanced_drag_buoyancy_terminal_velocity_and_reaction() {
    let rho = 1.2f64;
    let cd = 0.47f64;
    let r = 0.1f64;
    let area = PI * r * r;
    let mass = 1.0f64;
    let g = 9.81f64;
    let v_t = (2.0 * mass * g / (rho * cd * area)).sqrt(); // 33.3 m/s
    let falling = SolidSample {
        position: Vec3Fix::ZERO,
        velocity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-v_t), Fix128::ZERO),
        area_m2: Fix128::from_f64(area),
        volume_m3: Fix128::ZERO,
    };
    let f = drag_force(
        &falling,
        Vec3Fix::ZERO,
        Fix128::from_f64(rho),
        Fix128::from_f64(cd),
    );
    assert!(
        rel_err(f.y.to_f64(), mass * g) < 1e-9,
        "drag at V_t = weight: {}",
        f.y.to_f64()
    );
    assert_eq!(f.x, Fix128::ZERO);
    // magnitude law ½ρC_dAV² and direction opposing relative motion
    let mut sample = falling;
    sample.velocity = Vec3Fix::from_int(3, 4, 0); // |v| = 5
    let f5 = drag_force(&sample, Vec3Fix::ZERO, Fix128::from_int(1000), Fix128::ONE);
    let want = 0.5 * 1000.0 * area * 25.0;
    assert!(rel_err(f5.length().to_f64(), want) < 1e-9);
    assert!(
        rel_err(f5.x.to_f64() / f5.y.to_f64(), 0.75) < 1e-12,
        "anti-parallel to v"
    );
    assert!(f5.x < Fix128::ZERO && f5.y < Fix128::ZERO);
    // Galilean: same fluid velocity → no drag
    assert_eq!(
        drag_force(
            &sample,
            sample.velocity,
            Fix128::from_int(1000),
            Fix128::ONE
        ),
        Vec3Fix::ZERO
    );

    // Archimedes on a 1 L sample in water
    let litre = SolidSample {
        volume_m3: Fix128::from_ratio(1, 1000),
        ..falling
    };
    let fb = buoyancy_force(&litre, Fix128::from_int(1000), Fix128::from_ratio(981, 100));
    assert_eq!(fb.x, Fix128::ZERO);
    assert_eq!(fb.z, Fix128::ZERO);
    assert!(fix_close(fb.y, Fix128::from_ratio(981, 100)));

    // dumbbell: buoyant sample at +x, none at −x → torque about origin = r × F = 2 x̂ × 10 ŷ = 20 ẑ
    let at_rest = |x: i64, vol: i64| SolidSample {
        position: Vec3Fix::from_int(x, 0, 0),
        velocity: Vec3Fix::ZERO,
        area_m2: Fix128::ZERO,
        volume_m3: Fix128::from_ratio(vol, 1000),
    };
    let samples = [at_rest(2, 1), at_rest(-2, 0)];
    let (net, torque) = aggregate_forces(
        &samples,
        |_p| Vec3Fix::ZERO,
        Fix128::from_int(1000),
        Fix128::ONE,
        Fix128::from_int(10),
        Vec3Fix::ZERO,
    );
    assert_eq!(net.x, Fix128::ZERO);
    assert_eq!(net.z, Fix128::ZERO);
    assert!(
        fix_close(net.y, Fix128::from_int(10)),
        "net F_y = {}",
        net.y.to_f64()
    );
    assert_eq!(torque.x, Fix128::ZERO);
    assert_eq!(torque.y, Fix128::ZERO);
    assert!(
        fix_close(torque.z, Fix128::from_int(20)),
        "τ = r × F = 2 x̂ × 10 ŷ"
    );

    // Newton III: deposits sum to −Σ F exactly
    let forces = [Vec3Fix::from_int(1, -2, 3), Vec3Fix::from_int(-4, 5, 6)];
    let mut deposited = Vec3Fix::ZERO;
    react_back_pressure(&samples, &forces, |_pos, f| deposited = deposited + f);
    assert_eq!(deposited + forces[0] + forces[1], Vec3Fix::ZERO);
}

// ============================================================================
// multiphase — sphere curvature, Eikonal seed, trilinear exactness
// ============================================================================

/// Mean curvature of a sphere `κ = ∇·n̂ = 2/R` (Sethian, *Level Set
/// Methods* §1.2; Osher & Fedkiw §1.4). For the signed distance `|x| − R`
/// the crate's Laplacian estimate at an on-axis interface cell is
/// `2(√(R²+dx²) − R)·2/dx² = (2/R)(1 − dx²/4R² + …)`, i.e. 0.4 % low for
/// `R = 8 dx`; `|∇φ| = 1` holds exactly on the axis (Eikonal property of
/// a distance function).
#[test]
fn multiphase_sphere_level_set_curvature_and_eikonal_seed() {
    let n = 24usize;
    let dx = Fix128::from_ratio(1, 16);
    let mut phi = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    let c = Fix128::from_int(12) * dx;
    let radius = Fix128::from_int(8) * dx;
    initialize_level_set_sphere(&mut phi, c, c, c, radius);
    let kappa = curvature_at(&phi, 20, 12, 12).to_f64();
    assert!(rel_err(kappa, 2.0 / 0.5) < 0.01, "κ = {kappa} vs 2/R = 4");
    // curvature grows toward the centre: κ(R−dx) > κ(R) > κ(R+dx)
    let inner = curvature_at(&phi, 19, 12, 12).to_f64();
    let outer = curvature_at(&phi, 21, 12, 12).to_f64();
    assert!(inner > kappa && kappa > outer);
    assert!(rel_err(inner, 2.0 / (0.5 - 1.0 / 16.0)) < 0.01);
    assert!(rel_err(outer, 2.0 / (0.5 + 1.0 / 16.0)) < 0.01);
    // |∇φ| = 1 exactly on the axis, and never above 1 (φ is 1-Lipschitz)
    let grad_x = (phi.get(21, 12, 12) - phi.get(19, 12, 12)) / dx.double();
    assert_eq!(grad_x, Fix128::ONE);
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let gx = (phi.get(i + 1, j, k) - phi.get(i - 1, j, k)) / dx.double();
                let gy = (phi.get(i, j + 1, k) - phi.get(i, j - 1, k)) / dx.double();
                let gz = (phi.get(i, j, k + 1) - phi.get(i, j, k - 1)) / dx.double();
                let mag = (gx * gx + gy * gy + gz * gz).sqrt().to_f64();
                assert!(mag <= 1.0 + 1e-12, "|∇φ| = {mag} > 1 at ({i},{j},{k})");
            }
        }
    }
    // inside negative, outside positive, exact on the axis
    assert_eq!(phi.get(12, 12, 12), Fix128::ZERO - radius);
    assert_eq!(phi.get(23, 12, 12), Fix128::from_int(3) * dx);
}

/// Trilinear interpolation reproduces any affine field exactly (it is a
/// tensor-product linear Lagrange basis, partition of unity with linear
/// precision — Press et al. *Numerical Recipes* §3.6).
#[test]
fn multiphase_trilinear_sample_is_exact_on_affine_fields() {
    let n = 4usize;
    let mut g = Grid3d::new(n, n, n, Fix128::ONE, Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                // f = 1 + 2i + 3j + 4k
                g.set(
                    i,
                    j,
                    k,
                    Fix128::from_int(1 + 2 * i as i64 + 3 * j as i64 + 4 * k as i64),
                );
            }
        }
    }
    let x = Fix128::from_ratio(5, 4);
    let y = Fix128::from_ratio(5, 2);
    let z = Fix128::from_ratio(3, 4);
    let got = trilinear_sample(&g, x, y, z);
    let want =
        Fix128::ONE + Fix128::from_int(2) * x + Fix128::from_int(3) * y + Fix128::from_int(4) * z;
    assert_eq!(got, want, "affine field must be reproduced bit-exactly");
    assert_eq!(
        trilinear_sample(&g, Fix128::from_int(2), Fix128::ONE, Fix128::from_int(3)),
        g.get(2, 1, 3)
    );
}

// ============================================================================
// cloth_fluid — empirical coupling (validation: none)
// ============================================================================

/// No physical closed form: forces are applied as velocity increments
/// without a mass, the "surface tension" term is the mean fluid velocity
/// times a factor, and the coupling is one-way. Pinned invariants: no
/// neighbours → no change; equal velocities → no drag (Galilean); the
/// drag alone is a linear relaxation `v ← v (1 − C_d ρ N dt)` per step
/// (explicit Euler of `v̇ = −c v`, the Stokes-drag form); buoyancy is
/// vertical and proportional to the neighbour count.
#[test]
fn cloth_fluid_linear_drag_relaxation_and_symmetries() {
    let coupling = ClothFluidCoupling {
        drag_coefficient: Fix128::from_ratio(1, 2),
        buoyancy_factor: Fix128::ZERO,
        surface_tension: Fix128::ZERO,
    };
    let cloth_pos = [Vec3Fix::ZERO];
    let fluid_pos = [
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::new(Fix128::from_ratio(1, 4), Fix128::ZERO, Fix128::ZERO),
    ];
    let fluid_vel = [Vec3Fix::ZERO, Vec3Fix::ZERO];
    let dt = Fix128::from_ratio(1, 100);
    let rho = Fix128::from_int(10);
    let mut vel = [Vec3Fix::from_int(8, 0, 0)];
    // factor per step: 1 − C_d ρ N dt = 1 − 0.5·10·2·0.01 = 0.9
    let factor = Fix128::ONE - Fix128::from_ratio(1, 2) * rho * Fix128::from_int(2) * dt;
    let mut want = vel[0];
    for _ in 0..5 {
        apply_fluid_forces_to_cloth(
            &coupling, &cloth_pos, &mut vel, &fluid_pos, &fluid_vel, rho, dt,
        );
        want = want * factor;
        assert!(
            fix_close(vel[0].x, want.x) && vel[0].y.is_zero() && vel[0].z.is_zero(),
            "geometric relaxation: {:?} vs {:?}",
            vel[0],
            want
        );
    }
    // Galilean: cloth moving with the fluid feels no drag
    let moving = [Vec3Fix::from_int(2, 1, 0), Vec3Fix::from_int(2, 1, 0)];
    let mut vel = [Vec3Fix::from_int(2, 1, 0)];
    apply_fluid_forces_to_cloth(
        &coupling, &cloth_pos, &mut vel, &fluid_pos, &moving, rho, dt,
    );
    assert_eq!(vel[0], Vec3Fix::from_int(2, 1, 0));
    // no neighbours within 0.5 m: untouched
    let far = [Vec3Fix::from_int(5, 5, 5)];
    let mut vel = [Vec3Fix::from_int(1, 2, 3)];
    apply_fluid_forces_to_cloth(
        &coupling,
        &cloth_pos,
        &mut vel,
        &far,
        &[Vec3Fix::ZERO],
        rho,
        dt,
    );
    assert_eq!(vel[0], Vec3Fix::from_int(1, 2, 3));
    // buoyancy alone: Δv = b N dt ŷ, twice as much with twice the neighbours
    let buoyant = ClothFluidCoupling {
        drag_coefficient: Fix128::ZERO,
        buoyancy_factor: Fix128::from_int(3),
        surface_tension: Fix128::ZERO,
    };
    let mut vel = [Vec3Fix::ZERO];
    apply_fluid_forces_to_cloth(
        &buoyant, &cloth_pos, &mut vel, &fluid_pos, &fluid_vel, rho, dt,
    );
    assert_eq!(
        vel[0],
        Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_int(3) * Fix128::from_int(2) * dt,
            Fix128::ZERO
        )
    );
    let mut vel1 = [Vec3Fix::ZERO];
    apply_fluid_forces_to_cloth(
        &buoyant,
        &cloth_pos,
        &mut vel1,
        &fluid_pos[..1],
        &fluid_vel[..1],
        rho,
        dt,
    );
    assert_eq!(vel1[0].y.double(), vel[0].y);
}

// ============================================================================
// interface_capture — fast sweeping solves the Eikonal equation
// ============================================================================

/// The Eikonal equation `|∇φ| = 1` with `φ = 0` on a sphere has the
/// signed distance `|x − c| − R` as its viscosity solution (Sethian §8.4;
/// Zhao 2005). Seeding a 3-cell band with the exact distance and every
/// other cell with a 100× exaggerated value, the Godunov fast sweep must
/// rebuild the distance function. The scheme is first order: along the
/// grid axes outside the sphere the 1-neighbour update `a + h`
/// reproduces the distance bit-exactly, elsewhere the error grows to ≈ 0.06·|d| + 0.35 dx and
/// peaks at the sphere centre (the medial-axis kink of the viscosity
/// solution) at 0.70 dx on this `R = 4 dx` sphere — everything stays
/// sub-cell (`< 1 dx`). The sign is preserved and `|φ|` never grows
/// (Godunov causality), and every updated cell satisfies the discrete
/// Rouy–Tourin Eikonal equation to rounding.
#[test]
fn interface_capture_fast_sweeping_recovers_the_distance_function() {
    let n = 16usize;
    let dx = Fix128::ONE;
    let c = Fix128::from_int(8);
    let radius = Fix128::from_int(4);
    let mut exact = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    initialize_level_set_sphere(&mut exact, c, c, c, radius);
    let band = Fix128::from_ratio(3, 2);
    let mut phi = exact.clone();
    for v in phi.data.iter_mut() {
        if v.abs() > band {
            *v = *v * Fix128::from_int(100);
        }
    }
    let seeded = phi.clone();
    fast_sweeping_reinit(&mut phi, 2);
    let mut worst = 0.0f64;
    let mut sum_abs = 0.0f64;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let got = phi.get(i, j, k);
                let want = exact.get(i, j, k);
                assert_eq!(
                    got.is_negative(),
                    want.is_negative(),
                    "sign flipped at ({i},{j},{k})"
                );
                let err = (got - want).to_f64().abs();
                worst = worst.max(err);
                sum_abs += err;
                assert!(
                    got.abs() <= seeded.get(i, j, k).abs(),
                    "Godunov update may only shrink |φ|"
                );
                // grid-aligned characteristics: bit-exact distance
                let on_axis = u8::from(i == 8) + u8::from(j == 8) + u8::from(k == 8) >= 2;
                // (outside only: inside, the transverse neighbours are closer to
                // the interface and the 2-/3-neighbour solve takes over)
                if on_axis && want > band {
                    assert_eq!(got, want, "axis cell ({i},{j},{k}) must be exact");
                }
            }
        }
    }
    assert!(
        worst < 1.0,
        "distance-function max error {worst} dx must stay sub-cell (measured 0.70 at the centre)"
    );
    assert!(
        worst > 0.5,
        "the medial-axis kink error is a property of the scheme; if it vanished the tolerance should be tightened: {worst}"
    );
    let mean_abs = sum_abs / (n * n * n) as f64;
    assert!(mean_abs < 0.3, "mean |error| {mean_abs} dx");
    // discrete Eikonal residual: |∇φ| = 1 (upwind, Rouy–Tourin) away from the band
    let mut worst_res = 0.0f64;
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let p = phi.get(i, j, k).abs();
                if p <= band + dx {
                    continue;
                }
                let axis = |a: Fix128, b: Fix128| {
                    let m = if a.abs() < b.abs() { a.abs() } else { b.abs() };
                    let d = p - m;
                    if d.is_negative() {
                        Fix128::ZERO
                    } else {
                        d
                    }
                };
                let gx = axis(phi.get(i - 1, j, k), phi.get(i + 1, j, k));
                let gy = axis(phi.get(i, j - 1, k), phi.get(i, j + 1, k));
                let gz = axis(phi.get(i, j, k - 1), phi.get(i, j, k + 1));
                let mag = (gx * gx + gy * gy + gz * gz).sqrt().to_f64();
                worst_res = worst_res.max((mag - 1.0).abs());
            }
        }
    }
    assert!(
        worst_res < 1e-9,
        "upwind |∇φ| − 1 = {worst_res} (converged sweep is exact)"
    );
}
