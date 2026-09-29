//! Wall boundary conditions inside the pressure projection (`src/eulerian_grid.rs`).
//!
//! Until the face mask landed, `MacGrid` had no way to say "this face is a
//! wall": the projection removed the divergence *inside* the box while the
//! boundary faces kept pushing fluid out of it. The divergence read as
//! machine zero and the mass balance was still wrong, which is why a
//! lid-driven cavity run agreed with the reference near the lid and was
//! 3-4× too weak on the return flow.
//!
//! Every oracle below is a closed-form consequence of the no-through-flow
//! condition `u·n = 0`, or the published Ghia (1982) table. None of them is
//! derived from running the solver.
//!
//! # Provenance of the reference values
//!
//! `GHIA_RE100` is Table I of
//! **Ghia, U., Ghia, K. N., Shin, C. T. (1982), "High-Re solutions for
//! incompressible flow using the Navier-Stokes equations and a multigrid
//! method", J. Comput. Phys. 48(3) 387-411** — the `u` component along the
//! vertical line through the geometric centre of the cavity.
//!
//! The original PDF was not reachable from here, so this is **not a primary
//! verification**: three independent transcriptions of the Re=100 column
//! were compared and agree on all 17 rows (a public gist, the Table 2 of an
//! Uppsala thesis `diva2:1668016`, and a re-fetch of the gist in a later
//! session). The same gist has transcription errors in the Re=3200 and
//! Re=10000 columns, which are not used here.

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{project_pressure, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

/// Ghia (1982) Table I, Re = 100: `(y, u)` on the vertical centreline of a
/// unit cavity with a lid moving at `u = 1`. See the module header for how
/// far this was verified.
const GHIA_RE100: [(f64, f64); 17] = [
    (1.0000, 1.00000),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5000, -0.20581),
    (0.4531, -0.21090),
    (0.2813, -0.15662),
    (0.1719, -0.10150),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0000, 0.00000),
];

/// Peak return velocity in the reference profile (`y = 0.4531`).
const GHIA_RE100_PEAK_RETURN: f64 = -0.21090;

// ===========================================================================
// Helpers
// ===========================================================================

/// A divergent, wall-incompatible seed field: every face gets a value from a
/// fixed integer pattern so the initial `∇·u` is large and has no symmetry
/// that could make an oracle pass by accident.
fn seed_lumpy_field(grid: &mut MacGrid) {
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                let ix = i + (grid.nx + 1) * (j + grid.ny * k);
                grid.u[ix] = Fix128::from_ratio(((i * 5 + j * 3 + k * 7) % 11) as i64 - 5, 10);
            }
        }
    }
    for k in 0..grid.nz {
        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                let ix = i + grid.nx * (j + (grid.ny + 1) * k);
                grid.v[ix] = Fix128::from_ratio(((i * 2 + j * 7 + k * 3) % 9) as i64 - 4, 10);
            }
        }
    }
    for k in 0..=grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let ix = i + grid.nx * (j + grid.ny * k);
                grid.w[ix] = Fix128::from_ratio(((i * 3 + j * 5 + k * 2) % 7) as i64 - 3, 10);
            }
        }
    }
}

/// Largest `|∇·u|` over **every** cell, rim included.
fn max_abs_divergence(grid: &MacGrid) -> f64 {
    let mut worst = 0.0f64;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                worst = worst.max(grid.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    worst
}

// ===========================================================================
// Oracle 1 — a wall carries no flux
// ===========================================================================

/// Oracle: `u·n = 0` on a solid wall. A closed box therefore has exactly zero
/// normal velocity on all six boundary face layers *after* the projection —
/// not "small", zero, because the constraint is imposed, not converged to.
#[test]
fn walls_carry_no_flux_after_projection() {
    let n = 8usize;
    let dx = Fix128::from_ratio(1, 8);
    let mut grid = MacGrid::new(n, n, n, dx);
    seed_lumpy_field(&mut grid);
    grid.set_closed_box_walls();

    project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 200);

    let mut worst_face = 0.0f64;
    for k in 0..n {
        for j in 0..n {
            for &i in &[0usize, n] {
                worst_face = worst_face.max(grid.u(i, j, k).to_f64().abs());
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            for &j in &[0usize, n] {
                worst_face = worst_face.max(grid.v(i, j, k).to_f64().abs());
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            for &k in &[0usize, n] {
                worst_face = worst_face.max(grid.w(i, j, k).to_f64().abs());
            }
        }
    }
    assert_eq!(
        worst_face, 0.0,
        "a wall face must carry exactly zero normal velocity (worst {worst_face:.4e})"
    );
}

// ===========================================================================
// Oracle 2 — net flux through every cross-section of a closed box is zero
// ===========================================================================

/// Oracle: integrate `∇·u = 0` over the slab `x ∈ [0, i·dx]` of a closed box.
/// The surface integral collapses to `Σ_{j,k} u[i,j,k]` minus the flux through
/// the `x = 0` wall, which is zero; so the net flux through **every**
/// cross-section must vanish, for all three axes.
///
/// This is the quantity that read `+2.4842` on the vertical centreline while
/// the divergence read `3.47e-18`: the projection was removing divergence
/// inside the box and simultaneously pumping fluid out through the sides.
#[test]
fn closed_box_has_zero_net_flux_through_every_cross_section() {
    let n = 8usize;
    let dx = Fix128::from_ratio(1, 8);
    let mut grid = MacGrid::new(n, n, n, dx);
    seed_lumpy_field(&mut grid);
    grid.set_closed_box_walls();

    project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 400);

    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    for i in 0..=n {
        let mut flux = 0.0f64;
        for k in 0..n {
            for j in 0..n {
                flux += grid.u(i, j, k).to_f64();
            }
        }
        if flux.abs() > worst {
            worst = flux.abs();
            worst_label = format!("u cross-section i={i}");
        }
    }
    for j in 0..=n {
        let mut flux = 0.0f64;
        for k in 0..n {
            for i in 0..n {
                flux += grid.v(i, j, k).to_f64();
            }
        }
        if flux.abs() > worst {
            worst = flux.abs();
            worst_label = format!("v cross-section j={j}");
        }
    }
    for k in 0..=n {
        let mut flux = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                flux += grid.w(i, j, k).to_f64();
            }
        }
        if flux.abs() > worst {
            worst = flux.abs();
            worst_label = format!("w cross-section k={k}");
        }
    }
    assert!(
        worst < 1e-9,
        "net flux through a cross-section of a closed box must be 0, worst {worst:.4e} at {worst_label}"
    );
}

// ===========================================================================
// Oracle 3 — the projection has to reach the boundary layer
// ===========================================================================

/// Oracle: after a projection the discrete `∇·u` is zero in **every** cell.
/// There is nothing special about a cell that touches the boundary — the
/// correction subtracts the same gradient the Poisson operator solved for, so
/// the residual is uniform over the domain.
///
/// With the boundary face layer excluded from the correction sweep, the rim
/// cells had no degree of freedom left to fix their divergence and it grew
/// instead (`3.137 → 13.773` on `u = sin(πx)`), which no `1..n-1` test could
/// see.
#[test]
fn projection_reaches_the_boundary_layer_in_a_closed_box() {
    let n = 8usize;
    let dx = Fix128::from_ratio(1, 8);
    let mut grid = MacGrid::new(n, n, n, dx);
    seed_lumpy_field(&mut grid);
    grid.set_closed_box_walls();

    let before = max_abs_divergence(&grid);
    project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 600);
    let after = max_abs_divergence(&grid);

    assert!(
        after < 1e-6,
        "max |∇·u| over the whole domain after projection {after:.4e} (before {before:.4e})"
    );
}

// ===========================================================================
// Oracle 4 — a walled slab is a genuine 2-D problem
// ===========================================================================

/// Oracle: if the data has no `z` dependence and the `z` walls carry no flux,
/// the 3-D projection **is** the 2-D projection. The answer therefore cannot
/// depend on how many `z` layers the slab is cut into — `nz = 1`, `2` and `4`
/// must give the same `u` field, bit for bit, because `Fix128` arithmetic is
/// deterministic and the per-cell stencil is identical.
///
/// The fixed `1/6` divisor broke exactly this: at `nz = 1` both `z`
/// neighbours are missing and their zero contributions turn the operator into
/// a screened Poisson `∇²p − (2/dx²)p`, whose screening length is shorter
/// than one cell — the pressure collapses to `−rhs/6` and almost no
/// correction is applied (measured: 0.007 % of the divergence removed).
#[test]
fn walled_slab_projection_is_independent_of_the_layer_count() {
    let n = 6usize;
    let dx = Fix128::from_ratio(1, 8);

    let mut profiles = Vec::new();
    let mut divergences = Vec::new();
    for &nz in &[1usize, 2, 4] {
        let mut grid = MacGrid::new(n, n, nz, dx);
        // z-invariant seed: same pattern in every layer.
        for k in 0..nz {
            for j in 0..n {
                for i in 0..=n {
                    let ix = i + (n + 1) * (j + n * k);
                    grid.u[ix] = Fix128::from_ratio(((i * 5 + j * 3) % 11) as i64 - 5, 10);
                }
            }
        }
        for k in 0..nz {
            for j in 0..=n {
                for i in 0..n {
                    let ix = i + n * (j + (n + 1) * k);
                    grid.v[ix] = Fix128::from_ratio(((i * 2 + j * 7) % 9) as i64 - 4, 10);
                }
            }
        }
        grid.set_closed_box_walls();
        project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 400);

        divergences.push(max_abs_divergence(&grid));
        let mut layer0 = Vec::new();
        for j in 0..n {
            for i in 0..=n {
                layer0.push(grid.u(i, j, 0));
            }
        }
        profiles.push(layer0);
    }

    assert!(
        divergences.iter().all(|&d| d < 1e-6),
        "a walled slab must project at every layer count, max |∇·u| = {divergences:?}"
    );
    assert_eq!(
        profiles[0], profiles[1],
        "nz = 1 and nz = 2 must give the identical u field for z-invariant data"
    );
    assert_eq!(
        profiles[0], profiles[2],
        "nz = 1 and nz = 4 must give the identical u field for z-invariant data"
    );
}

// ===========================================================================
// Oracle 5 — lid-driven cavity against Ghia (1982)
// ===========================================================================

/// Grid resolution of the cavity oracle that runs in CI.
const CAVITY_N: usize = 16;
/// Steps taken to reach the steady state (`t = 30` at `dt = 0.05`).
const CAVITY_STEPS: u32 = 600;
/// Gauss-Seidel sweeps per projection.
const CAVITY_GS: u32 = 200;

/// Run a Re=100 lid-driven cavity and return the `u` profile sampled on the
/// vertical centreline as `(y, u)` pairs, plus the net `x` flux through that
/// line (which must be zero in a sealed cavity).
///
/// `walls_in_projection` selects **how** the no-through-flow condition is
/// applied, and is the only difference between the two runs:
///
/// - `false` — the boundary normal velocities are zeroed from the outside
///   before every step, which is all a caller could do without a face mask.
///   The projection does not know those faces are walls, so it pushes fluid
///   through them again during the same step.
/// - `true` — the same faces are additionally marked solid, so the
///   constraint is part of the Poisson problem.
///
/// The tangential no-slip walls and the moving lid are applied the same way
/// in both runs: the viscous term in `diffuse_velocity` mirrors the boundary
/// with a zero-gradient ghost value, and a wall with velocity `U` needs
/// `u_ghost = 2U − u_in` instead, so the difference `2(U − u_in)` is added
/// explicitly here. That is an operator split, not a solver change, and it is
/// identical on both sides of the comparison.
fn run_lid_driven_cavity(
    n: usize,
    steps: u32,
    gs_iterations: u32,
    walls_in_projection: bool,
) -> (Vec<(f64, f64)>, f64) {
    let dx = Fix128::from_ratio(1, n as i64);
    let lid_u = Fix128::ONE;
    // Re = U·L/ν = 100 with U = L = 1 and ρ = 1 → ν = μ = 1/100.
    let nu = Fix128::from_ratio(1, 100);
    let dt = Fix128::from_ratio(1, 20);

    let mut solver = CfdSolver::new(n, n, 1, dx);
    solver.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = nu;
    solver.jacobi_iterations = gs_iterations;
    solver.use_turbulence = false;
    if walls_in_projection {
        solver.grid.set_closed_box_walls();
    }

    let shear = nu * dt / (dx * dx);
    let two = Fix128::from_int(2);
    for _ in 0..steps {
        // Tangential no-slip on the four side walls + the moving lid.
        for i in 0..=n {
            let ix_bot = i;
            solver.grid.u[ix_bot] = solver.grid.u[ix_bot] - shear * two * solver.grid.u(i, 0, 0);
            let ix_top = i + (n + 1) * (n - 1);
            solver.grid.u[ix_top] =
                solver.grid.u[ix_top] + shear * two * (lid_u - solver.grid.u(i, n - 1, 0));
        }
        for j in 0..=n {
            let ix_left = n * j;
            solver.grid.v[ix_left] = solver.grid.v[ix_left] - shear * two * solver.grid.v(0, j, 0);
            let ix_right = (n - 1) + n * j;
            solver.grid.v[ix_right] =
                solver.grid.v[ix_right] - shear * two * solver.grid.v(n - 1, j, 0);
        }
        // No-through-flow imposed from outside the solver — everything a
        // caller can do without a face mask.
        for j in 0..n {
            solver.grid.u[(n + 1) * j] = Fix128::ZERO;
            solver.grid.u[n + (n + 1) * j] = Fix128::ZERO;
        }
        for i in 0..n {
            solver.grid.v[i] = Fix128::ZERO;
            solver.grid.v[i + n * n] = Fix128::ZERO;
        }

        solver.step(dt);
    }

    // x = 1/2 sits exactly on the u-face column i = n/2.
    let mid = n / 2;
    let mut profile = Vec::with_capacity(n);
    let mut net_flux = 0.0f64;
    for j in 0..n {
        let y = (j as f64 + 0.5) / n as f64;
        let u = solver.grid.u(mid, j, 0).to_f64();
        profile.push((y, u));
        net_flux += u;
    }
    (profile, net_flux)
}

/// Linear interpolation of a `(y, u)` profile, clamped at both ends
/// (`u = 0` on the floor, `u = lid` on the lid).
fn sample_profile(profile: &[(f64, f64)], y: f64) -> f64 {
    if y <= profile[0].0 {
        let t = y / profile[0].0;
        return t * profile[0].1;
    }
    let last = profile[profile.len() - 1];
    if y >= last.0 {
        let t = (y - last.0) / (1.0 - last.0);
        return last.1 + t * (1.0 - last.1);
    }
    for pair in profile.windows(2) {
        let (y0, u0) = pair[0];
        let (y1, u1) = pair[1];
        if y >= y0 && y <= y1 {
            let t = (y - y0) / (y1 - y0);
            return u0 + t * (u1 - u0);
        }
    }
    last.1
}

/// Largest deviation from the Ghia profile over the tabulated heights.
fn max_deviation_from_ghia(profile: &[(f64, f64)]) -> f64 {
    GHIA_RE100
        .iter()
        .map(|&(y, u_ref)| (sample_profile(profile, y) - u_ref).abs())
        .fold(0.0f64, f64::max)
}

/// Oracle: a sealed lid-driven cavity at Re=100 must reproduce the Ghia
/// (1982) return flow, and the net `x` flux through the centreline must be
/// zero because no fluid enters or leaves.
///
/// The tolerances are set by the discretisation, not by the reference:
/// `16 × 16` cells with semi-Lagrangian advection smear the primary vortex,
/// so the peak return velocity is expected to come in short of the reference
/// `−0.21090`. The gate asks for **half** of it — far below what the scheme
/// should manage and far above the `29 %` measured when the walls were only
/// imposed from outside the projection.
#[test]
fn lid_driven_cavity_return_flow_matches_ghia_1982() {
    let (with_mask, flux_with_mask) =
        run_lid_driven_cavity(CAVITY_N, CAVITY_STEPS, CAVITY_GS, true);
    let (outside_only, flux_outside_only) =
        run_lid_driven_cavity(CAVITY_N, CAVITY_STEPS, CAVITY_GS, false);

    let peak_with_mask = with_mask.iter().fold(0.0f64, |acc, &(_, u)| acc.min(u));
    let peak_outside_only = outside_only.iter().fold(0.0f64, |acc, &(_, u)| acc.min(u));
    let dev_with_mask = max_deviation_from_ghia(&with_mask);
    let dev_outside_only = max_deviation_from_ghia(&outside_only);

    println!("Ghia Re=100 peak return {GHIA_RE100_PEAK_RETURN:+.5}");
    println!(
        "  mask in projection : peak {peak_with_mask:+.5} \
         ({:.1} % of reference), max deviation {dev_with_mask:.5}, centreline net flux {flux_with_mask:+.5}",
        100.0 * peak_with_mask / GHIA_RE100_PEAK_RETURN
    );
    println!(
        "  outside only       : peak {peak_outside_only:+.5} \
         ({:.1} % of reference), max deviation {dev_outside_only:.5}, centreline net flux {flux_outside_only:+.5}",
        100.0 * peak_outside_only / GHIA_RE100_PEAK_RETURN
    );
    for &(y, u_ref) in &GHIA_RE100 {
        println!(
            "  y={y:.4}  ghia {u_ref:+.5}  mask {:+.5}  outside {:+.5}",
            sample_profile(&with_mask, y),
            sample_profile(&outside_only, y)
        );
    }

    assert!(
        flux_with_mask.abs() < 1e-6,
        "net x flux through the centreline of a sealed cavity must be 0, got {flux_with_mask:+.5e}"
    );
    assert!(
        peak_with_mask <= 0.5 * GHIA_RE100_PEAK_RETURN,
        "peak return velocity {peak_with_mask:+.5} is below half the reference \
         {GHIA_RE100_PEAK_RETURN:+.5}"
    );
    assert!(
        dev_with_mask < dev_outside_only,
        "putting the walls inside the projection must move the profile toward the \
         reference: max deviation {dev_with_mask:.5} with the mask vs \
         {dev_outside_only:.5} without it"
    );
    for &(y, u_ref) in &GHIA_RE100 {
        let u = sample_profile(&with_mask, y);
        if u_ref.abs() > 0.05 {
            assert!(
                u * u_ref > 0.0,
                "the flow direction at y={y:.4} disagrees with the reference \
                 (got {u:+.5}, reference {u_ref:+.5})"
            );
        }
    }
}
