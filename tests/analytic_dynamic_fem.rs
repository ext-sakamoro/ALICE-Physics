//! Closed-form oracles for the transient FEM solver
//! ([`alice_physics::dynamic_fem`]).
//!
//! Three layers, because a single one of them cannot fail for the right reason:
//!
//! 1. **The integrator on its own** lives in the module's own tests — the exact
//!    four-cycle at `ω·dt = 2` and the measured energy drift elsewhere.
//! 2. **The mass matrix on its own**, here: a free body under a given total
//!    force accelerates at `F/(ρV)`, which is the only statement that reads the
//!    *total* mass and nothing else.
//! 3. **Mass and stiffness together**, here: the step response of a
//!    single-degree-of-freedom scene against `u_s(1 − cos nθ)`, the axial mode
//!    of a bar against `√(E/ρ)`, and the exact `√(5/2)` frequency ratio between
//!    the two mass distributions.
//!
//! Layer 3 is where the inertia has to be load-bearing, so the scenes are
//! vibrating ones. A scene whose answer is the static solution would pass with
//! the time term deleted; `a_vibrating_scene_is_not_what_the_static_solver_says`
//! is the teeth that keeps that honest.
//!
//! # Units
//!
//! mm / N / MPa / s, so density is in tonne/mm³. `ρ = 2⁻³⁰ = 9.3e-10` is used
//! throughout: close to PLA's `1.24e-9` and a power of two, which keeps it off
//! the list of things that round.

#![cfg(feature = "std")]

use alice_physics::dynamic_fem::{DynamicsConfig, MassLumping, TransientSolver};
use alice_physics::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, FemError, Preconditioner, SolverConfig,
};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics::Fix128;

// ---------------------------------------------------------------------------
// scenes
// ---------------------------------------------------------------------------

/// `2⁻ˢʰⁱᶠᵗ` as a `Fix128`.
fn two_pow_neg(shift: u32) -> Fix128 {
    Fix128::from_raw(0, 1u64 << (64 - shift))
}

/// Density used by every scene here: `2⁻³⁰` tonne/mm³.
fn density() -> Fix128 {
    two_pow_neg(30)
}

const DENSITY_F64: f64 = 1.0 / 1_073_741_824.0;

/// A corner tetrahedron with `det J = 3`, hence `V = 1/2` exactly.
///
/// The height is three so that the volume is dyadic. `V = |det J|/6` and the
/// shape function gradients `J⁻¹` can never both be exact in a binary fixed
/// point format — `J⁻¹` dyadic forces `det J⁻¹` dyadic, so `det J` is one over a
/// dyadic, which is never `6·2ᵏ`. The volume is the half that matters for the
/// mass, so it is the half this scene makes exact.
fn corner_tet() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]);
    mesh.vertices.push([1.0, 0.0, 0.0]);
    mesh.vertices.push([0.0, 1.0, 0.0]);
    mesh.vertices.push([0.0, 0.0, 3.0]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh
}

const CORNER_TET_VOLUME: f64 = 0.5;
/// `∇N₃ = (0, 0, 1/3)` for [`corner_tet`], from `J⁻¹` with `J = diag(1, 1, 3)`.
const CORNER_TET_GRAD3_Z: f64 = 1.0 / 3.0;

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Axis-aligned box `[0,nx·h] × [0,ny·h] × [0,nz·h]` in Kuhn 6-tet cells, the
/// same construction `tests/analytic_linear_elastic_fem.rs` uses.
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// `ν = 0`, so `λ = 0` and the one-dimensional rod wave speed is exactly
/// `√(E/ρ)` with no lateral coupling to correct for.
fn uniaxial(e_mpa: i64) -> ElasticMaterial {
    ElasticMaterial::new(Fix128::from_int(e_mpa), Fix128::ZERO).expect("E > 0, ν in (−1, 1/2)")
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(20_000, Fix128::from_f64(1e-12)).expect("valid solver config")
}

fn dynamics(dt_shift: u32, lumping: MassLumping) -> DynamicsConfig {
    DynamicsConfig::try_new(density(), two_pow_neg(dt_shift), lumping).expect("valid dynamics")
}

/// The step number of the **first** local maximum of `trace`, one-based.
///
/// Not the largest value in the window. A step-loaded elastic body is a sum of
/// modes, and its envelope beats: the bar below reaches `1.98 u_s` at step 1581
/// while the wave first returns to the tip at step 65. Taking the maximum over
/// a window therefore measures where the window ended, which is how the first
/// draft of these tests read a transit time three times too long and a density
/// scaling of `0.86` where the closed form says `2`.
fn first_local_max(trace: &[f64]) -> usize {
    for i in 1..trace.len() - 1 {
        if trace[i] > trace[i - 1] && trace[i] >= trace[i + 1] {
            return i + 1;
        }
    }
    0
}

/// `cos(nθ)` for `n = 1..=count`, where `θ` is the trapezoidal rule's phase per
/// step at `Ω = ω·dt`.
///
/// No trigonometry anywhere, which is not a stylistic choice: `clippy.toml`
/// forbids `f64::cos` and `f64::atan` crate-wide — including in tests — because
/// the platform `libm` is not bit-exact across targets, and a reference value
/// that moves between machines is not a reference value.
///
/// It is also the better oracle. The amplification matrix of the trapezoidal
/// rule has eigenvalues `(1 − Ω²/4 ± iΩ)/(1 + Ω²/4)`, whose modulus is exactly
/// one and whose real and imaginary parts are **rational in `Ω`**:
///
/// ```text
/// cos θ = (1 − Ω²/4)/(1 + Ω²/4)        sin θ = Ω/(1 + Ω²/4)
/// ```
///
/// so `θ = 2·atan(Ω/2)` never has to be formed. `cos nθ` then comes from the
/// Chebyshev recurrence `c_{n+1} = 2 c₁ c_n − c_{n−1}`, four operations per
/// step, with no library call between the closed form and the assert.
fn cos_multiples(omega_dt: f64, count: usize) -> Vec<f64> {
    let q = omega_dt * omega_dt / 4.0;
    let c1 = (1.0 - q) / (1.0 + q);
    let mut out = Vec::with_capacity(count);
    let (mut prev, mut cur) = (1.0_f64, c1);
    for _ in 0..count {
        out.push(cur);
        let next = 2.0 * c1 * cur - prev;
        prev = cur;
        cur = next;
    }
    out
}

// ---------------------------------------------------------------------------
// layer 2: the mass matrix on its own
// ---------------------------------------------------------------------------

/// An unconstrained body under a total force `F` accelerates at `F/(ρV)`, and
/// Newmark reproduces a constant acceleration exactly, so the centre of mass
/// follows `u(t) = ½ a t²` with no discretisation error at all.
///
/// Three things make this the mass test and not a disguised stiffness test. The
/// body translates rigidly, so `K u` is a rigid mode and contributes nothing.
/// There are no constraints, which the static solver refuses outright and the
/// transient one accepts because `Ṁ` is positive definite on its own. And the
/// load is a fixed total force rather than one proportional to the density, so
/// the mass does not cancel: halving `ρ` must double the acceleration.
///
/// Both distributions are checked because a single tetrahedron's consistent and
/// lumped row sums are both `ρV/4`, so they must agree here exactly — and
/// disagree in `the_two_mass_distributions_differ_by_the_closed_form_ratio`.
#[test]
fn a_free_body_under_a_total_force_follows_the_closed_form_parabola() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let dt_shift = 20u32;
    let dt = f64::from(1u32) / 1_048_576.0;
    // F = 1 N split over the four nodes.
    let total_force = 1.0_f64;
    let mut boundary = BoundaryConditions::new();
    for v in 0..4u32 {
        boundary.add_load(v, Axis::X, Fix128::from_f64(total_force / 4.0));
    }
    let want_accel = total_force / (DENSITY_F64 * CORNER_TET_VOLUME);

    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let mut solver = TransientSolver::new(
            &mesh,
            &material,
            &boundary,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("an unconstrained transient solve is well posed");
        let mut worst = 0.0_f64;
        for n in 1..=100u32 {
            solver.step().expect("the effective system converges");
            let t = f64::from(n) * dt;
            let want = 0.5 * want_accel * t * t;
            for node in 0..4usize {
                let got = solver.displacements()[node * 3].to_f64();
                let err = ((got - want) / want).abs();
                if err > worst {
                    worst = err;
                }
            }
        }
        assert!(
            worst <= 1e-12,
            "{lumping:?}: free flight departed from ½at² by {worst:.3e} relative \
             (a = {want_accel:.6e} mm/s²)"
        );
    }
}

/// Halving the density doubles the acceleration of the same scene under the
/// same force, which is the statement `a = F/M` and nothing else.
///
/// Without it the test above would still pass if `ρ` were ignored and some other
/// quantity with the right units took its place, because that test reads the
/// expected acceleration from the same `ρ` it hands the solver.
#[test]
fn halving_the_density_doubles_the_free_flight_acceleration() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let mut boundary = BoundaryConditions::new();
    for v in 0..4u32 {
        boundary.add_load(v, Axis::X, Fix128::from_f64(0.25));
    }
    let mut displacement = [0.0_f64; 2];
    for (slot, shift) in displacement.iter_mut().zip([30u32, 31u32]) {
        let dynamics =
            DynamicsConfig::try_new(two_pow_neg(shift), two_pow_neg(20), MassLumping::Consistent)
                .expect("valid dynamics");
        let mut solver =
            TransientSolver::new(&mesh, &material, &boundary, &dynamics, &solver_config())
                .expect("well posed");
        for _ in 0..50 {
            solver.step().expect("converges");
        }
        *slot = solver.displacements()[0].to_f64();
    }
    let ratio = displacement[1] / displacement[0];
    assert!(
        (ratio - 2.0).abs() <= 1e-9,
        "halving ρ changed the displacement by ×{ratio:.12}, closed form 2"
    );
}

// ---------------------------------------------------------------------------
// layer 3: mass and stiffness together
// ---------------------------------------------------------------------------

/// Reduce [`corner_tet`] to one free degree of freedom: node 3 along x.
fn single_dof_boundary(load_n: f64) -> BoundaryConditions {
    let mut boundary = BoundaryConditions::new();
    for v in 0..3u32 {
        boundary.fix(v);
    }
    boundary.prescribe(3, Axis::Y, Fix128::ZERO);
    boundary.prescribe(3, Axis::Z, Fix128::ZERO);
    boundary.add_load(3, Axis::X, Fix128::from_f64(load_n));
    boundary
}

/// `K₃ₓ,₃ₓ = V[(λ+2μ)g_x² + μ(g_y² + g_z²)]` with `∇N₃ = (0, 0, 1/3)` and
/// `λ = 0`, so the whole entry is `V μ /9`.
fn single_dof_stiffness(e_mpa: f64) -> f64 {
    let mu = e_mpa / 2.0; // ν = 0
    CORNER_TET_VOLUME * mu * CORNER_TET_GRAD3_Z * CORNER_TET_GRAD3_Z
}

/// `Mᵢᵢ` for one tetrahedron: `ρV/10` consistent (`2·ρV/20`), `ρV/4` lumped.
fn single_dof_mass(lumping: MassLumping) -> f64 {
    let m = DENSITY_F64 * CORNER_TET_VOLUME;
    match lumping {
        MassLumping::Lumped => m / 4.0,
        _ => m / 10.0,
    }
}

/// The step response of a single-degree-of-freedom system is
/// `u(t) = u_s(1 − cos ωt)`, and the trapezoidal rule reproduces it exactly with
/// `ω` replaced by the algorithmic frequency `θ/dt`, `tan(θ/2) = ω dt / 2`.
///
/// Every factor in that sentence comes from somewhere independent:
///
/// - `u_s` from [`linear_elastic_fem::solve`] on the same scene, so the
///   stiffness this module assembles is pinned against the one the static
///   module assembles rather than against itself;
/// - `ω = √(K/M)` from the closed forms [`single_dof_stiffness`] and
///   [`single_dof_mass`], written out from the element geometry;
/// - `θ = 2·atan(ω dt/2)` from the amplification matrix of the trapezoidal rule,
///   whose eigenvalues `(1 − Ω²/4 ± iΩ)/(1 + Ω²/4)` have modulus exactly one.
///
/// The peak of `2 u_s` is the classical dynamic amplification factor for a
/// suddenly applied load, which is the part a reader can check without any of
/// the above.
#[test]
fn a_single_free_degree_of_freedom_tracks_the_closed_form_step_response() {
    let mesh = corner_tet();
    let e_mpa = 3500.0;
    let material = uniaxial(3500);
    let load = 1.0_f64;
    let boundary = single_dof_boundary(load);
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);

    let static_solution = linear_elastic_fem::solve(&mesh, &material, &boundary, &solver_config())
        .expect("the static scene is well constrained");
    let u_s = static_solution.displacements[3][0].to_f64();
    let closed_form_u_s = load / single_dof_stiffness(e_mpa);
    assert!(
        ((u_s - closed_form_u_s) / closed_form_u_s).abs() <= 1e-9,
        "the static solver and the closed form disagree on u_s: \
         {u_s:.12e} vs {closed_form_u_s:.12e}"
    );

    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let omega = (single_dof_stiffness(e_mpa) / single_dof_mass(lumping)).sqrt();
        let omega_dt = omega * dt;
        // Six periods. `θ ≤ Ω` always, so `6·2π/Ω` is a few steps short of six
        // and never over — enough cycles for a phase error to show, and the
        // exact count does not enter any assert.
        let steps = (6.0 * core::f64::consts::TAU / omega_dt) as usize;
        let reference = cos_multiples(omega_dt, steps);
        let mut solver = TransientSolver::new(
            &mesh,
            &material,
            &boundary,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        let mut worst = 0.0_f64;
        let mut peak = 0.0_f64;
        for &cos_n_theta in &reference {
            solver.step().expect("converges");
            let got = solver.displacements()[9].to_f64();
            let want = u_s * (1.0 - cos_n_theta);
            worst = worst.max((got - want).abs() / u_s);
            peak = peak.max(got / u_s);
        }
        assert!(
            worst <= 1e-6,
            "{lumping:?}: over {steps} steps the response left u_s(1 − cos nθ) by \
             {worst:.3e} of u_s (Ω = {omega_dt:.9})"
        );
        assert!(
            (peak - 2.0).abs() <= 1e-3,
            "{lumping:?}: the dynamic amplification factor came out {peak:.6}, closed form 2"
        );
    }
}

/// `ω ∝ 1/√M`, and this scene's two mass distributions are `ρV/10` and `ρV/4`,
/// so the frequency ratio is exactly `√(5/2)`.
///
/// This is the one statement that separates the two distributions. The free
/// flight test cannot: a rigid translation sees only the row sums, which are
/// `ρV/4` for both. Without it, `MassLumping::Consistent` could be implemented
/// as a second name for `Lumped` and every other test here would still pass.
#[test]
fn the_two_mass_distributions_differ_by_the_closed_form_ratio() {
    let ratio =
        (single_dof_mass(MassLumping::Lumped) / single_dof_mass(MassLumping::Consistent)).sqrt();
    assert!(
        (ratio - 2.5_f64.sqrt()).abs() < 1e-15,
        "the closed forms themselves disagree: {ratio}"
    );

    let mesh = corner_tet();
    let material = uniaxial(3500);
    let boundary = single_dof_boundary(1.0);
    // `2⁻²⁶` and not the `2⁻²³` the other scenes use. The turning point is only
    // located to the nearest whole step, and at `2⁻²³` the consistent half
    // period is 18 steps, so the quantisation alone is 5% — larger than the
    // 2% separation between √(5/2) and the value a mis-scaled mass would give.
    // At `2⁻²⁶` the half period is about 146 steps and the quantisation is 0.7%.
    let dt_shift = 26u32;

    let mut half_period = [0usize; 2];
    for (slot, lumping) in half_period
        .iter_mut()
        .zip([MassLumping::Consistent, MassLumping::Lumped])
    {
        let mut solver = TransientSolver::new(
            &mesh,
            &material,
            &boundary,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        // Half the first period is the first maximum of `1 − cos`.
        let mut trace = Vec::with_capacity(4000);
        for _ in 0..4000u32 {
            solver.step().expect("converges");
            trace.push(solver.displacements()[9].to_f64());
        }
        *slot = first_local_max(&trace);
        assert!(*slot > 10, "the peak was found at step {slot}, too coarse");
    }
    // The half period in steps is `π/θ`, and the algorithmic `θ` differs from
    // `Ω` by `O(Ω²/12)` — `3.9e-5` and `1.6e-5` for the two cases here, so the
    // ratio of step counts is the ratio of frequencies to five digits, well
    // inside the 0.7% the whole-step quantisation already costs. Measuring the
    // counts directly avoids recovering `θ` through a `tan`, which `clippy.toml`
    // forbids anyway.
    let got = half_period[1] as f64 / half_period[0] as f64;
    assert!(
        (got - 2.5_f64.sqrt()).abs() <= 2e-2,
        "consistent/lumped frequency ratio measured {got:.6} ({} then {} steps to the \
         turning point), closed form √(5/2) = {:.6}",
        half_period[0],
        half_period[1],
        2.5_f64.sqrt()
    );
}

/// A rod clamped at one end and loaded suddenly at the other carries a wave
/// that reaches the clamp and returns, so the tip turns around at `t = 2L/c`
/// with `c = √(E/ρ)` the one-dimensional wave speed, and the displacement there
/// is twice the static one.
///
/// `ν = 0` is what makes `c` exactly `√(E/ρ)`; with a Poisson ratio the rod and
/// the three-dimensional dilatational speeds differ and the closed form would
/// have to pick one.
///
/// This is the only test here whose expected value does not pass through the
/// element matrices at all — it is the continuum answer, so it is the one that
/// can tell that the assembled system is the discretisation of the right
/// equation and not merely self-consistent. The price is that it is also the
/// loosest: eight P1 tetrahedra along the span are stiff, and the measured
/// transit time comes out `6%` short of the continuum value (`0.9388` to
/// `0.9461` of `2L/c` over the density and step size variations tried below).
/// The bound is `10%`, which leaves room for that and not for a factor.
#[test]
fn the_axial_bar_reaches_its_peak_at_the_closed_form_wave_transit_time() {
    let (nx, h) = (8usize, 1.0_f32);
    let length = f64::from(nx as u32) * f64::from(h);
    let mesh = kuhn_box(nx, 1, 1, h);
    let e_mpa = 3500.0;
    let material = uniaxial(3500);
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);

    let mut boundary = BoundaryConditions::new();
    for j in 0..2usize {
        for k in 0..2usize {
            boundary.fix(node_index(nx, 1, 0, j, k));
        }
    }
    for j in 0..2usize {
        for k in 0..2usize {
            boundary.add_load(node_index(nx, 1, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }

    let wave_speed = (e_mpa / DENSITY_F64).sqrt();
    let want_peak_time = 2.0 * length / wave_speed;

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(dt_shift, MassLumping::Consistent),
        &solver_config(),
    )
    .expect("well posed");
    let tip = node_index(nx, 1, nx, 0, 0) as usize * 3;
    let static_tip = linear_elastic_fem::solve(&mesh, &material, &boundary, &solver_config())
        .expect("well constrained")
        .displacements[tip / 3][0]
        .to_f64();
    let budget = (4.0 * want_peak_time / dt) as usize;
    let mut trace = Vec::with_capacity(budget);
    for _ in 0..budget {
        solver.step().expect("converges");
        trace.push(solver.displacements()[tip].to_f64());
    }
    let best_n = first_local_max(&trace);
    let got_peak_time = best_n as f64 * dt;
    let error = (got_peak_time - want_peak_time).abs() / want_peak_time;
    assert!(
        error <= 0.10,
        "the tip turned around at {got_peak_time:.6e} s, closed form 2L/c = \
         {want_peak_time:.6e} s (relative error {error:.4}, c = {wave_speed:.4e} mm/s, \
         step {best_n} of {budget})"
    );
    // The turning point of a step-loaded rod is at twice the static deflection,
    // the same dynamic amplification factor the single-degree-of-freedom scene
    // shows. Measured 1.95 to 1.98 here, the shortfall being the same stiff
    // discretisation that moves the transit time.
    let amplification = trace[best_n - 1] / static_tip;
    assert!(
        (1.85..=2.05).contains(&amplification),
        "the turning point was at {amplification:.4}× the static deflection, closed form 2"
    );
}

/// Multiplying the density by sixteen quadruples the transit time, because
/// `c = √(E/ρ)`.
///
/// The discretisation error cancels in the ratio: every eigenvalue of the
/// discrete system scales as `1/√ρ` exactly, however badly eight elements
/// resolve the rod. That makes this a far tighter statement about the mass than
/// the absolute transit time above, and the two fail for different reasons — a
/// mass scaled by a constant moves the absolute value and leaves the ratio
/// alone, while a mis-assembled stiffness moves both.
///
/// Sixteen and not four, so that the quantisation of the turning point onto
/// whole steps (`±1` of `65`, or `1.5%`) is small against the thing being
/// measured. Measured `65 → 260` steps, ratio `4.000`.
#[test]
fn the_transit_time_scales_as_the_square_root_of_density() {
    let (nx, h) = (8usize, 1.0_f32);
    let mesh = kuhn_box(nx, 1, 1, h);
    let material = uniaxial(3500);
    let dt_shift = 23u32;

    let mut boundary = BoundaryConditions::new();
    for j in 0..2usize {
        for k in 0..2usize {
            boundary.fix(node_index(nx, 1, 0, j, k));
            boundary.add_load(node_index(nx, 1, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }

    let tip = node_index(nx, 1, nx, 0, 0) as usize * 3;
    let mut peak_step = [0usize; 2];
    for (slot, rho_shift) in peak_step.iter_mut().zip([30u32, 26u32]) {
        let dynamics = DynamicsConfig::try_new(
            two_pow_neg(rho_shift),
            two_pow_neg(dt_shift),
            MassLumping::Consistent,
        )
        .expect("valid dynamics");
        let mut solver =
            TransientSolver::new(&mesh, &material, &boundary, &dynamics, &solver_config())
                .expect("well posed");
        let mut trace = Vec::with_capacity(1000);
        for _ in 0..1000u32 {
            solver.step().expect("converges");
            trace.push(solver.displacements()[tip].to_f64());
        }
        *slot = first_local_max(&trace);
        assert!(
            *slot > 10,
            "the turning point was at step {slot}, too coarse"
        );
    }
    let ratio = peak_step[1] as f64 / peak_step[0] as f64;
    assert!(
        (ratio - 4.0).abs() <= 0.1,
        "a sixteenfold ρ changed the transit time by ×{ratio:.6} \
         ({} then {} steps), closed form 4",
        peak_step[0],
        peak_step[1]
    );
}

/// The same step response with `Preconditioner::JacobiScaled`, which is what
/// makes `diag(K + Ṁ)` load-bearing.
///
/// Added because a mutation that deleted the mass from the effective diagonal
/// survived every other test in this file. The default is
/// `Preconditioner::None`, and `build_preconditioner` then returns ones without
/// reading the diagonal at all, so the diagonal was assembled and never looked
/// at. The preconditioner cannot change the answer — that is the point of a
/// preconditioner — so the assert is still the closed form; what changes is
/// that a wrong diagonal now has to pass through the iteration, which it does
/// not survive.
#[test]
fn the_jacobi_preconditioned_path_reaches_the_same_closed_form() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let boundary = single_dof_boundary(1.0);
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    let config = SolverConfig::try_new(20_000, Fix128::from_f64(1e-12))
        .expect("valid solver config")
        .with_preconditioner(Preconditioner::JacobiScaled);

    let u_s = linear_elastic_fem::solve(&mesh, &material, &boundary, &config)
        .expect("well constrained")
        .displacements[3][0]
        .to_f64();
    let omega = (single_dof_stiffness(3500.0) / single_dof_mass(MassLumping::Consistent)).sqrt();
    let reference = cos_multiples(omega * dt, 400);

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(dt_shift, MassLumping::Consistent),
        &config,
    )
    .expect("well posed");
    let mut worst = 0.0_f64;
    for &cos_n_theta in &reference {
        solver.step().expect("converges");
        let got = solver.displacements()[9].to_f64();
        worst = worst.max((got - u_s * (1.0 - cos_n_theta)).abs() / u_s);
    }
    assert!(
        worst <= 1e-6,
        "with JacobiScaled the response left u_s(1 − cos nθ) by {worst:.3e} of u_s"
    );
}

/// A scene with a **non-zero** prescribed displacement, which is what makes the
/// `K u₀` term of the initial acceleration load-bearing.
///
/// Added because a mutation that computed the initial residual as `f − (K+Ṁ)u₀`
/// instead of `f − K u₀` survived every other test in this file: all of them
/// prescribe zero, and `u₀ = 0` makes the two expressions identical.
///
/// Holding node 3 at `x = d` and loading nothing is a static problem with a
/// known answer — the body is fully constrained, so it sits at the prescribed
/// field forever with zero velocity. The closed form is therefore exact and the
/// assert can be an `assert_eq!`: every free degree of freedom must stay at
/// zero for the whole run, and the prescribed one at `d`.
#[test]
fn a_non_zero_prescribed_displacement_is_held_exactly() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let displacement = Fix128::from_f64(0.0625); // 2⁻⁴, dyadic
    let mut boundary = BoundaryConditions::new();
    for v in 0..3u32 {
        boundary.fix(v);
    }
    boundary.prescribe(3, Axis::X, displacement);
    boundary.prescribe(3, Axis::Y, Fix128::ZERO);
    boundary.prescribe(3, Axis::Z, Fix128::ZERO);

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(23, MassLumping::Consistent),
        &solver_config(),
    )
    .expect("well posed");
    for n in 1..=200u32 {
        solver.step().expect("converges");
        assert_eq!(
            solver.displacements()[9],
            displacement,
            "the prescribed value moved at step {n}"
        );
        for (d, &value) in solver.velocities().iter().enumerate() {
            assert_eq!(value, Fix128::ZERO, "velocity at dof {d}, step {n}");
        }
    }
}

/// The same idea with one free degree of freedom left, so that `K u₀` actually
/// drives something.
///
/// Fully constrained is not enough on its own: with no free degree of freedom
/// the initial-acceleration solve has nothing to solve for.
///
/// **Which degree of freedom is prescribed matters, and not in an obvious way.**
/// The first version of this scene pulled node 3 along `x` and freed its `y`,
/// and the free coordinate never moved. The reason is in the geometry:
/// `∇N₃ = (0, 0, 1/3)` is purely axial, so node 3's row of `∫BᵀσdV` reads only
/// `σ_zz`, `σ_yz` and `σ_zx`, and a displacement of node 3 along `x` produces
/// none of those on a free row. Prescribing node **1** along **z** does:
/// `∇N₁ = (1, 0, 0)` puts `u₁z` into the engineering shear `γ_zx`, which with
/// `λ = 0` gives `σ_zx = μ u₁z` and a non-zero force on node 3's free `x`.
///
/// So the closed form is the dynamic amplification factor again, now driven by
/// a suddenly imposed *displacement* rather than a load: the free coordinate
/// starts at zero and swings to twice its static value, which
/// [`linear_elastic_fem::solve`] supplies for the identical scene.
#[test]
fn a_prescribed_displacement_drives_the_free_coordinate_from_the_right_acceleration() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let displacement = Fix128::from_f64(0.0625); // 2⁻⁴, dyadic
    let mut boundary = BoundaryConditions::new();
    boundary.fix(0);
    boundary.fix(2);
    boundary.prescribe(1, Axis::X, Fix128::ZERO);
    boundary.prescribe(1, Axis::Y, Fix128::ZERO);
    boundary.prescribe(1, Axis::Z, displacement);
    boundary.prescribe(3, Axis::Y, Fix128::ZERO);
    boundary.prescribe(3, Axis::Z, Fix128::ZERO);

    let static_x = linear_elastic_fem::solve(&mesh, &material, &boundary, &solver_config())
        .expect("well constrained")
        .displacements[3][0]
        .to_f64();
    assert!(
        static_x.abs() > 1e-6,
        "the prescribed displacement moves the free coordinate by {static_x:.3e}, \
         which is nothing — this scene would pass with K u₀ ignored"
    );

    // The sampled peak is below the true one, and the shortfall has a closed
    // form rather than a fitted tolerance. Writing `nθ = π + φ` for the step
    // nearest the turning point, the sampled value is `1 − cos(π+φ) = 1 + cos φ`
    // and the shortfall from 2 is `1 − cos φ ≈ φ²/2`. The step grid puts
    // `|φ| ≤ θ/2`, so
    //
    //     2 − peak ≤ θ²/8 ≤ Ω²/8        with Ω = ω·dt and θ = 2·atan(Ω/2) < Ω
    //
    // ⚠️ The shortfall is **not** proportional to `dt²`; only its envelope is.
    // `φ` depends on the fractional part of `π/θ`, which jumps around as `dt`
    // changes, so halving `dt` does not quarter the error. Measured:
    //
    // | dt    | peak step | 2 − peak  | Ω²/8     |
    // |-------|-----------|-----------|----------|
    // | 2⁻²³  | 18        | 1.187e-3  | 3.709e-3 |
    // | 2⁻²⁴  | 36        | 9.251e-4  | 9.272e-4 |  ← fractional part ≈ ½, the worst case
    // | 2⁻²⁵  | 73        | 1.086e-6  | 2.318e-4 |
    // | 2⁻²⁶  | 146       | 1.690e-6  | 5.795e-5 |
    // | 2⁻²⁷  | 292       | 1.865e-6  | 1.449e-5 |
    //
    // The bound holds at every step size and is met to within 0.2% at `2⁻²⁴`,
    // so it is tight and not decoration. Below `2⁻²⁵` the shortfall stops
    // falling and settles near `1.9e-6`: that is the conjugate gradient's
    // residual floor, not the sampling, and `SOLVER_FLOOR` covers it.
    const SOLVER_FLOOR: f64 = 1e-5;
    let dt_shift = 25u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    let omega = (single_dof_stiffness(3500.0) / single_dof_mass(MassLumping::Consistent)).sqrt();
    let omega_dt = omega * dt;
    let bound = omega_dt * omega_dt / 8.0 + SOLVER_FLOOR;

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(dt_shift, MassLumping::Consistent),
        &solver_config(),
    )
    .expect("well posed");
    // Normalised by the static value, so the swing runs 0 → 2 → 0 whichever way
    // the coupling points.
    let mut trace = Vec::with_capacity(400);
    for _ in 0..400u32 {
        solver.step().expect("converges");
        trace.push(solver.displacements()[9].to_f64() / static_x);
    }
    let peak_step = first_local_max(&trace);
    assert!(
        peak_step > 0,
        "the free coordinate never turned around in 400 steps, so nothing was measured"
    );
    let peak = trace[peak_step - 1];
    assert!(
        peak <= 2.0 + 1e-9,
        "the free coordinate reached {peak:.9}× its static value at step {peak_step}; \
         an undamped single-degree-of-freedom system cannot exceed 2"
    );
    assert!(
        2.0 - peak <= bound,
        "the free coordinate peaked at {peak:.9}× its static value {static_x:.6e} at step \
         {peak_step}, short of the closed form 2 by {:.3e} — past the sampling envelope \
         Ω²/8 = {:.3e} plus the solver floor {SOLVER_FLOOR:.0e}. An initial acceleration \
         that ignores K u₀ starts the swing from the wrong place",
        2.0 - peak,
        omega_dt * omega_dt / 8.0
    );
    assert!(
        peak > 1.5,
        "the free coordinate peaked at only {peak:.6}× its static value; a solve without \
         inertia approaches 1 and never overshoots"
    );
}

/// A prescribed displacement on the **same axis** as a free degree of freedom,
/// which is what makes the `K` in `M a₀ = f − K u₀` load-bearing rather than
/// `K + Ṁ`.
///
/// Added because a mutation that wrote the initial residual as `f − (K+Ṁ)u₀`
/// survived everything else here, including the two scenes above. The reason is
/// that `Ṁ` couples nodes only within an axis: `apply_mass` sums the four nodal
/// values of one axis and never mixes axes. The earlier scene prescribes node 1
/// along **z** and frees node 3 along **x**, so `Ṁ u₀` is zero on the free row
/// and the two expressions agree there after all.
///
/// Prescribing node 1 along **x** fixes that: now `Ṁ u₀` is non-zero on node 3's
/// free **x** row. And `K u₀` is exactly zero on it, structurally — node 3's row
/// of `∫BᵀσdV` reads only `σ_zz`, `σ_yz` and `σ_zx` because `∇N₃ = (0,0,1/3)`,
/// and `γ_zx = Σᵢ(∇Nᵢ_z·uᵢx + ∇Nᵢ_x·uᵢz)` picks up nothing from `u₁x` because
/// `∇N₁ = (1,0,0)` has no `z`. So the correct initial acceleration is zero, the
/// free coordinate never moves, and the assert can be exact: measured
/// `max |u| = 0` over 200 steps, as a raw `Fix128`, not a tolerance.
#[test]
fn a_same_axis_prescribed_displacement_leaves_an_uncoupled_free_dof_exactly_at_rest() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let displacement = Fix128::from_f64(0.0625); // 2⁻⁴, dyadic
    let mut boundary = BoundaryConditions::new();
    boundary.fix(0);
    boundary.fix(2);
    boundary.prescribe(1, Axis::X, displacement);
    boundary.prescribe(1, Axis::Y, Fix128::ZERO);
    boundary.prescribe(1, Axis::Z, Fix128::ZERO);
    boundary.prescribe(3, Axis::Y, Fix128::ZERO);
    boundary.prescribe(3, Axis::Z, Fix128::ZERO);

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(23, MassLumping::Consistent),
        &solver_config(),
    )
    .expect("well posed");
    for n in 1..=200u32 {
        solver.step().expect("converges");
        assert_eq!(
            solver.displacements()[9],
            Fix128::ZERO,
            "node 3 moved along x at step {n}; K u₀ is structurally zero on that row, \
             so only a residual that wrongly includes Ṁ u₀ can start it moving"
        );
        assert_eq!(
            solver.velocities()[9],
            Fix128::ZERO,
            "node 3 gained velocity along x at step {n}"
        );
    }
}

// ---------------------------------------------------------------------------
// degenerate input
// ---------------------------------------------------------------------------

/// Every degenerate input returns `Err`, and none of them panics, divides by
/// zero or overflows on the way.
///
/// The cases are the ones that reach arithmetic rather than a type: a zero or
/// negative step divides by zero two layers down, a step small enough that
/// `4/dt²` leaves [`Fix128`] silently wraps, and a zero or negative density
/// makes the effective operator indefinite. The assert is on the variant, not
/// merely on `is_err`, because `InvalidConfig` and `DegenerateElement` call for
/// different responses from a caller.
#[test]
fn degenerate_configurations_are_refused_rather_than_panicking() {
    let good_dt = two_pow_neg(23);
    let good_rho = density();
    let cases: [(&str, Fix128, Fix128); 7] = [
        ("dt = 0", good_rho, Fix128::ZERO),
        ("dt < 0", good_rho, Fix128::ZERO - good_dt),
        ("ρ = 0", Fix128::ZERO, good_dt),
        ("ρ < 0", Fix128::ZERO - good_rho, good_dt),
        // dt² underflows the 2⁻⁶⁴ floor entirely.
        ("dt = 2⁻⁶⁰", good_rho, Fix128::from_raw(0, 1 << 4)),
        // 4/dt² leaves the i64 integer part.
        ("dt = 2⁻³⁵", good_rho, Fix128::from_raw(0, 1 << 29)),
        ("dt = 2⁻⁴⁰", good_rho, Fix128::from_raw(0, 1 << 24)),
    ];
    for (label, rho, dt) in cases {
        let got = DynamicsConfig::try_new(rho, dt, MassLumping::Consistent);
        assert!(
            matches!(got, Err(FemError::InvalidConfig(_))),
            "{label}: expected InvalidConfig, got {got:?}"
        );
    }
    // And the good one is still accepted, so the guards are not refusing
    // everything.
    assert!(
        DynamicsConfig::try_new(good_rho, good_dt, MassLumping::Consistent).is_ok(),
        "the valid configuration was refused, so these cases prove nothing"
    );
}

/// Degenerate meshes and boundary data are refused by variant, not by panic.
///
/// A flat tetrahedron is the one that matters: its shape function gradients are
/// `cross / det` with `det = 0`, so without the guard it divides by zero before
/// anything else notices.
#[test]
fn degenerate_meshes_are_refused_rather_than_panicking() {
    let material = uniaxial(3500);
    let dynamics = dynamics(23, MassLumping::Consistent);
    let config = solver_config();
    let empty = BoundaryConditions::new();

    let no_vertices = SdfTetMesh::default();
    assert!(
        matches!(
            TransientSolver::new(&no_vertices, &material, &empty, &dynamics, &config),
            Err(FemError::EmptyMesh)
        ),
        "an empty mesh must be refused"
    );

    let mut no_tets = SdfTetMesh::default();
    no_tets.vertices.push([0.0, 0.0, 0.0]);
    assert!(
        matches!(
            TransientSolver::new(&no_tets, &material, &empty, &dynamics, &config),
            Err(FemError::EmptyMesh)
        ),
        "a mesh with vertices but no tetrahedra must be refused"
    );

    // Four coplanar points: det J = 0, so ∇N would divide by zero.
    let mut flat = SdfTetMesh::default();
    flat.vertices.push([0.0, 0.0, 0.0]);
    flat.vertices.push([1.0, 0.0, 0.0]);
    flat.vertices.push([0.0, 1.0, 0.0]);
    flat.vertices.push([1.0, 1.0, 0.0]);
    flat.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    assert!(
        matches!(
            TransientSolver::new(&flat, &material, &empty, &dynamics, &config),
            Err(FemError::DegenerateElement { tet: 0 })
        ),
        "a flat tetrahedron must be refused before ∇N divides by zero"
    );

    // A tetrahedron naming a vertex the mesh does not have.
    let mut dangling = corner_tet();
    dangling.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 9],
    });
    assert!(
        matches!(
            TransientSolver::new(&dangling, &material, &empty, &dynamics, &config),
            Err(FemError::VertexOutOfRange { vertex: 9, .. })
        ),
        "a tetrahedron naming a missing vertex must be refused"
    );

    // A boundary condition naming a vertex the mesh does not have.
    let mut bad_bc = BoundaryConditions::new();
    bad_bc.add_load(42, Axis::X, Fix128::ONE);
    assert!(
        matches!(
            TransientSolver::new(&corner_tet(), &material, &bad_bc, &dynamics, &config),
            Err(FemError::VertexOutOfRange { vertex: 42, .. })
        ),
        "a load on a missing vertex must be refused"
    );

    // The well-formed scene still builds, so the guards are not refusing
    // everything.
    assert!(
        TransientSolver::new(&corner_tet(), &material, &empty, &dynamics, &config).is_ok(),
        "the valid mesh was refused, so these cases prove nothing"
    );
}

// ---------------------------------------------------------------------------
// teeth
// ---------------------------------------------------------------------------

/// The same scene has one answer from the static solver and an oscillation from
/// this one, and the oscillation crosses the static answer repeatedly.
///
/// Without this, every test above could be satisfied by a solver whose inertia
/// term was quietly dropped on scenes where the static answer happens to be
/// close — and `tests/analytic_fem_convergence.rs` shows that a scene chosen for
/// a clean closed form is exactly the kind that degenerates that way. The count
/// is asserted, not the mere presence of a crossing, because one crossing is
/// what a monotone approach to the static solution also produces.
#[test]
fn a_vibrating_scene_is_not_what_the_static_solver_says() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let boundary = single_dof_boundary(1.0);
    let config = solver_config();

    let u_s = linear_elastic_fem::solve(&mesh, &material, &boundary, &config)
        .expect("well constrained")
        .displacements[3][0]
        .to_f64();

    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(23, MassLumping::Consistent),
        &config,
    )
    .expect("well posed");
    let mut crossings = 0u32;
    let mut above = false;
    let mut peak = 0.0_f64;
    for _ in 0..4000u32 {
        solver.step().expect("converges");
        let u = solver.displacements()[9].to_f64();
        peak = peak.max(u);
        let now_above = u > u_s;
        if now_above != above {
            crossings += 1;
            above = now_above;
        }
    }
    assert!(
        crossings >= 6,
        "the response crossed the static solution {crossings} times in 4000 steps; \
         a solver without inertia crosses it once and stays"
    );
    assert!(
        peak > 1.5 * u_s,
        "the response peaked at {peak:.6e}, only {:.2}× the static {u_s:.6e}; \
         an over-damped or inertia-free solve never overshoots",
        peak / u_s
    );
}

/// The static solver refuses an unconstrained body and the transient one does
/// not, which is a difference in the systems and not in the implementations:
/// `K` is singular on the rigid modes and `K + Ṁ` is not.
#[test]
fn the_transient_system_is_well_posed_where_the_static_one_is_singular() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let mut boundary = BoundaryConditions::new();
    boundary.add_load(3, Axis::X, Fix128::ONE);

    assert!(
        linear_elastic_fem::solve(&mesh, &material, &boundary, &solver_config()).is_err(),
        "the static solve must refuse a body with no constraints"
    );
    let mut solver = TransientSolver::new(
        &mesh,
        &material,
        &boundary,
        &dynamics(20, MassLumping::Consistent),
        &solver_config(),
    )
    .expect("the transient solve accepts it");
    solver.step().expect("and advances");
    assert!(
        solver.displacements()[9] > Fix128::ZERO,
        "the loaded node must have moved"
    );
}

/// A time step that is not a power of two costs nothing measurable — **because
/// of how the coefficient is formed**, and this test is what holds that in
/// place.
///
/// The first draft of the solver wrote `4/(dt·dt)`. `Fix128` has a fixed
/// absolute resolution of `2⁻⁶⁴`, so a quantity's relative precision is
/// `2⁻⁶⁴/value`, and `dt²` is the smallest number in the whole method: at
/// `dt = 1/6 000 000` it has 19 bits left and `4/dt²` inherits a relative error
/// of `1.1e-6`. That is a bias on the algorithmic frequency, so the phase it
/// costs **accumulates linearly in the step count** — and it gets worse as `dt`
/// is refined, which is the opposite of the way a time step usually behaves.
///
/// Writing the same constant as `(2/dt)·(2/dt)` never forms the small quantity.
/// Measured, worst departure from `u_s(1 − cos nθ)` over 500 steps:
///
/// | `dt` | rel. error of `4/(dt·dt)` | rel. error of `(2/dt)²` | departure, direct | departure, invert-first |
/// |---|---|---|---|---|
/// | `2⁻²³` | 0 | 0 | `6.2e-13` | `7.9e-13` |
/// | `2⁻²¹` | 0 | 0 | `1.2e-13` | `1.9e-14` |
/// | `1/6 000 000` | `1.088e-6` | **0** | `1.47e-5` | **`4.5e-13`** |
/// | `1/60` | `2.5e-16` | **0** | `9.7e-11` | **`9.7e-12`** |
///
/// Seven and a half orders of magnitude on the small non-dyadic step, and the
/// dyadic column does not move, because for a dyadic `dt` the two forms are
/// bit-identical (`dynamic_fem`'s own
/// `the_newmark_coefficients_are_exact_for_a_dyadic_step` asserts that
/// equality directly).
///
/// So the assert here is not "a non-dyadic step is worse" — it is no longer
/// true — but **"the direct form would have been worse, and the solver does not
/// use it"**. Both halves are checked, because only the second one fails if
/// someone simplifies `(2/dt)·(2/dt)` back to `4/(dt·dt)`: every other test in
/// this file passes with either form.
#[test]
fn forming_the_coefficient_by_inverting_first_is_what_makes_any_step_safe() {
    let mesh = corner_tet();
    let material = uniaxial(3500);
    let boundary = single_dof_boundary(1.0);
    let u_s = linear_elastic_fem::solve(&mesh, &material, &boundary, &solver_config())
        .expect("well constrained")
        .displacements[3][0]
        .to_f64();
    let omega = (single_dof_stiffness(3500.0) / single_dof_mass(MassLumping::Consistent)).sqrt();

    let departure = |dt: Fix128| -> f64 {
        let reference = cos_multiples(omega * dt.to_f64(), 500);
        let dynamics = DynamicsConfig::try_new(density(), dt, MassLumping::Consistent)
            .expect("any positive dt is valid");
        let mut solver =
            TransientSolver::new(&mesh, &material, &boundary, &dynamics, &solver_config())
                .expect("well posed");
        let mut worst = 0.0_f64;
        for &cos_n_theta in &reference {
            solver.step().expect("converges");
            let got = solver.displacements()[9].to_f64();
            let want = u_s * (1.0 - cos_n_theta);
            worst = worst.max((got - want).abs() / u_s);
        }
        worst
    };

    let small_non_dyadic = Fix128::from_ratio(1, 6_000_000);
    for (label, dt, bound) in [
        ("2⁻²³", two_pow_neg(23), 1e-11_f64),
        ("2⁻²¹", two_pow_neg(21), 1e-11),
        ("1/6e6", small_non_dyadic, 1e-11),
        ("1/60", Fix128::from_ratio(1, 60), 1e-10),
    ] {
        let got = departure(dt);
        assert!(
            got <= bound,
            "dt = {label}: the response departed from the closed form by {got:.3e} of u_s, \
             bound {bound:.0e} — a non-dyadic step is supposed to cost nothing now"
        );
    }

    // The half that keeps the reason alive: `4/(dt·dt)` really is worse, so the
    // way the solver forms the coefficient is load-bearing and not decoration.
    let exact = 4.0 / (small_non_dyadic.to_f64() * small_non_dyadic.to_f64());
    let direct = (Fix128::from_int(4) / (small_non_dyadic * small_non_dyadic)).to_f64();
    let two_over_dt = Fix128::from_int(2) / small_non_dyadic;
    let invert_first = (two_over_dt * two_over_dt).to_f64();
    let direct_error = (direct - exact).abs() / exact;
    let invert_first_error = (invert_first - exact).abs() / exact;
    assert!(
        direct_error >= 1e-7,
        "4/(dt·dt) at dt = 1/6e6 came out within {direct_error:.3e} of exact; \
         it was measured at 1.088e-6, so either Fix128 changed or this test no \
         longer measures what it describes"
    );
    assert!(
        invert_first_error <= direct_error / 1e3,
        "(2/dt)² is {invert_first_error:.3e} off and 4/(dt·dt) is {direct_error:.3e}; \
         the invert-first form is supposed to win by orders of magnitude"
    );
}
