//! Oracles for [`project_pressure_distributed`] — the slab-decomposed pressure
//! projection with every rank on its own thread, exchanging halo layers as byte
//! messages.
//!
//! # What is pinned, and where each expected value comes from
//!
//! 1. **Bit identity with the single-process projection.** For every rank count —
//!    1, counts that divide `nz`, counts that do not, and counts above `nz` — the
//!    decomposed answer equals, entry for entry, the answer of the public
//!    single-process solve ([`project_pressure`] for the two Gauss-Seidel
//!    decompositions, [`project_pressure_multigrid`] for the multigrid one). The
//!    reference is a different code path (one address space, one sweep over the
//!    whole grid); the decomposition argument for why the two must agree is that
//!    a red-black colour sweep reads only the opposite colour, so splitting it
//!    across slabs reorders no read, and `Fix128` addition is a group operation.
//!    Teeth: the reference with one more sweep / cycle is a different field, and
//!    so is the input.
//! 2. **The projected field is discretely divergence-free.** The divergence is
//!    computed here from the raw face arrays (`(u[i+1] − u[i] + v[j+1] − v[j] +
//!    w[k+1] − w[k]) / dx`, indexed by the documented layout), not through any
//!    accessor of the crate. The bound is a fraction of the seeded divergence.
//! 3. **A manufactured solution is recovered.** The input is
//!    `u = ∇_h q + curl_h ψ` on a closed box: `q` a cell potential, `ψ` a
//!    stream function on the `x-y` cell corners that vanishes on the boundary, so
//!    `curl_h ψ` has zero discrete divergence and zero normal velocity on every
//!    wall (both by the telescoping of the difference stencil). The discrete
//!    Helmholtz decomposition is then exact: the projection must return
//!    `curl_h ψ` as the velocity and a pressure whose differences are
//!    `ρ/dt · (q_i − q_{i−1})`.
//! 4. **Refusals.** Zero ranks, a zero count, a zero `dt` / density / spacing,
//!    multigrid off a power-of-two grid and a solver that is not a slab
//!    decomposition each return the named error and leave the grid untouched.

#![cfg(feature = "std")]

use alice_physics::cfd_solver::{PressureSolver, PressureSolverError};
use alice_physics::eulerian_grid::{
    project_pressure, project_pressure_distributed, project_pressure_multigrid, FaceBc, MacGrid,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 100)
}

fn rho() -> Fix128 {
    Fix128::from_int(1000)
}

/// A small integer hash, so a seeded field has no structure a halo bug could
/// hide behind.
fn h(i: usize, j: usize, k: usize, salt: usize) -> i64 {
    let mut x = (i as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((j as u64).wrapping_mul(0x85EB_CA6B))
        .wrapping_add((k as u64).wrapping_mul(0xC2B2_AE35))
        .wrapping_add(salt as u64);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    (x % 97) as i64 - 48
}

#[derive(Clone, Copy, Debug)]
enum Scene {
    /// No face conditions: the rim is plain fluid (exterior `p = 0`).
    Open,
    /// A closed box with an interior wall, an inflow on the low `x` side and
    /// outflow faces at the low end of `z` and the high end of `x`, some of
    /// which sit next to the inflow and the walls.
    Mixed,
}

fn seeded(nx: usize, ny: usize, nz: usize, scene: Scene) -> MacGrid {
    let mut g = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                g.u[i + (nx + 1) * (j + ny * k)] = Fix128::from_ratio(h(i, j, k, 1), 16);
            }
        }
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                g.v[i + nx * (j + (ny + 1) * k)] = Fix128::from_ratio(h(i, j, k, 2), 16);
            }
        }
    }
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                g.w[i + nx * (j + ny * k)] = Fix128::from_ratio(h(i, j, k, 3), 16);
            }
        }
    }
    // A non-zero starting pressure, so a halo that starts from zero instead of
    // from the field is a different answer.
    for (c, p) in g.pressure.iter_mut().enumerate() {
        *p = Fix128::from_ratio(h(c, 0, 0, 4), 64);
    }
    if let Scene::Mixed = scene {
        g.set_closed_box_walls();
        for k in 0..nz {
            for j in 0..ny {
                g.set_u_bc(
                    0,
                    j,
                    k,
                    FaceBc::Inflow {
                        normal_velocity: Fix128::from_ratio(1, 2),
                    },
                );
                if j % 2 == 0 {
                    g.set_u_bc(nx, j, k, FaceBc::Outflow);
                }
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                if (i + j) % 3 == 0 {
                    g.set_w_bc(i, j, 0, FaceBc::Outflow);
                }
            }
        }
        // The X-face next to the low-x inflow is an outflow on one row, so the
        // low-end ordering of the conditions is exercised; and an interior
        // wall face crosses a slab boundary.
        g.set_u_bc(1, 0, 0, FaceBc::Outflow);
        g.set_w_bc(
            1,
            1,
            nz / 2,
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
        );
    }
    g
}

fn bits_eq(a: &MacGrid, b: &MacGrid) -> bool {
    a.pressure == b.pressure && a.u == b.u && a.v == b.v && a.w == b.w
}

fn first_difference(a: &MacGrid, b: &MacGrid) -> String {
    for (name, x, y) in [
        ("pressure", &a.pressure, &b.pressure),
        ("u", &a.u, &b.u),
        ("v", &a.v, &b.v),
        ("w", &a.w, &b.w),
    ] {
        if let Some(i) = x.iter().zip(y.iter()).position(|(p, q)| p != q) {
            return format!("{name}[{i}]: {:?} vs {:?}", x[i], y[i]);
        }
    }
    "no difference".into()
}

fn distributed(base: &MacGrid, solver: PressureSolver) -> MacGrid {
    let mut g = base.clone();
    project_pressure_distributed(&mut g, dt(), rho(), solver).expect("a valid request");
    g
}

fn single_gs(base: &MacGrid, sweeps: u32) -> MacGrid {
    let mut g = base.clone();
    project_pressure(&mut g, dt(), rho(), sweeps);
    g
}

fn single_mg(base: &MacGrid, cycles: u32) -> MacGrid {
    let mut g = base.clone();
    project_pressure_multigrid(&mut g, dt(), rho(), cycles);
    g
}

/// Oracle 1, Gauss-Seidel: both decompositions reproduce `project_pressure` to
/// the bit for every rank count, on a grid whose `nz` (12) is divisible by some
/// of the counts and not by others, and with more ranks than layers.
#[test]
fn gauss_seidel_decompositions_are_bit_identical_to_the_single_process_solve() {
    const SWEEPS: u32 = 3;
    for scene in [Scene::Open, Scene::Mixed] {
        for (nx, ny, nz) in [(8, 8, 8), (6, 5, 12)] {
            let base = seeded(nx, ny, nz, scene);
            let want = single_gs(&base, SWEEPS);
            assert!(!bits_eq(&want, &base), "the reference must move the field");
            assert!(
                !bits_eq(&want, &single_gs(&base, SWEEPS + 1)),
                "one more sweep must be a different field, or equality proves nothing",
            );
            for ranks in [1, 2, 3, 4, 5, 8, nz + 7] {
                for solver in [
                    PressureSolver::DecomposedGs {
                        ranks,
                        sweeps: SWEEPS,
                    },
                    PressureSolver::BandedGs {
                        ranks,
                        sweeps: SWEEPS,
                    },
                ] {
                    let got = distributed(&base, solver);
                    assert!(
                        bits_eq(&got, &want),
                        "{nx}x{ny}x{nz} {scene:?} {solver:?}: {}",
                        first_difference(&got, &want),
                    );
                }
            }
        }
    }
}

/// Oracle 1, multigrid: the decomposed W-cycle reproduces
/// `project_pressure_multigrid` to the bit, including rank counts that leave
/// levels with fewer layers than ranks (agglomerated to rank 0) and counts above
/// `nz`.
#[test]
fn the_multigrid_decomposition_is_bit_identical_to_the_single_process_solve() {
    const CYCLES: u32 = 2;
    for scene in [Scene::Open, Scene::Mixed] {
        for ((nx, ny, nz), rank_counts) in [
            ((16, 16, 16), &[1usize, 2, 3, 4, 8, 20][..]),
            ((8, 4, 32), &[2, 5, 8][..]),
            ((32, 32, 32), &[8][..]),
        ] {
            let base = seeded(nx, ny, nz, scene);
            let want = single_mg(&base, CYCLES);
            assert!(!bits_eq(&want, &base));
            assert!(!bits_eq(&want, &single_mg(&base, CYCLES + 1)));
            for &ranks in rank_counts {
                let solver = PressureSolver::DecomposedMultigrid {
                    ranks,
                    cycles: CYCLES,
                };
                let got = distributed(&base, solver);
                assert!(
                    bits_eq(&got, &want),
                    "{nx}x{ny}x{nz} {scene:?} {solver:?}: {}",
                    first_difference(&got, &want),
                );
            }
        }
    }
}

/// `max |∇·u|` over every cell, from the raw face arrays.
fn max_divergence(g: &MacGrid) -> f64 {
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);
    let dx = g.dx.to_f64();
    let mut worst = 0f64;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let u = |i: usize| g.u[i + (nx + 1) * (j + ny * k)].to_f64();
                let v = |j: usize| g.v[i + nx * (j + (ny + 1) * k)].to_f64();
                let w = |k: usize| g.w[i + nx * (j + ny * k)].to_f64();
                let d = (u(i + 1) - u(i) + v(j + 1) - v(j) + w(k + 1) - w(k)) / dx;
                worst = worst.max(d.abs());
            }
        }
    }
    worst
}

/// Oracle 2: after the projection the field is divergence-free to a small
/// fraction of the seeded divergence, for every decomposed solver.
///
/// The bounds are the convergence each count buys on this grid (multigrid: a
/// rate per W-cycle that does not depend on the grid; Gauss-Seidel on 8³: many
/// sweeps), with orders of magnitude of margin (measured: about 3e-15 and 4e-13
/// of the seed); the claim being tested is the zero of the closed form, not the
/// number.
#[test]
fn the_projected_field_is_discretely_divergence_free() {
    let mut walled = seeded(16, 16, 16, Scene::Open);
    walled.set_closed_box_walls();
    walled.enforce_face_boundaries();
    let before = max_divergence(&walled);
    assert!(before > 10.0, "the seed must carry divergence: {before}");
    let mg = distributed(
        &walled,
        PressureSolver::DecomposedMultigrid {
            ranks: 4,
            cycles: 30,
        },
    );
    let after = max_divergence(&mg);
    assert!(
        after < before * 1e-10,
        "multigrid over 4 ranks left max|div| {after:e} of {before:e}",
    );

    let mut small = seeded(8, 8, 8, Scene::Open);
    small.set_closed_box_walls();
    small.enforce_face_boundaries();
    let before = max_divergence(&small);
    for solver in [
        PressureSolver::DecomposedGs {
            ranks: 3,
            sweeps: 400,
        },
        PressureSolver::BandedGs {
            ranks: 3,
            sweeps: 400,
        },
    ] {
        let after = max_divergence(&distributed(&small, solver));
        assert!(
            after < before * 1e-9,
            "{solver:?} left max|div| {after:e} of {before:e}",
        );
    }
}

/// Oracle 3: the discrete Helmholtz decomposition of a manufactured field.
#[test]
fn a_manufactured_gradient_plus_curl_field_is_split_exactly() {
    let n = 16;
    let dx = Fix128::from_ratio(1, 8);
    let q = |i: usize, j: usize, k: usize| Fix128::from_ratio(h(i, j, k, 7), 32);
    // Zero on the boundary corners (i or j at 0 or n), so `curl_h ψ` has no
    // normal component on the walls.
    let psi = |i: usize, j: usize| {
        if i == 0 || j == 0 || i == n || j == n {
            Fix128::ZERO
        } else {
            Fix128::from_ratio(h(i, j, 0, 9), 16)
        }
    };
    let mut g = MacGrid::new(n, n, n, dx);
    g.set_closed_box_walls();
    let mut curl_u = vec![Fix128::ZERO; g.u.len()];
    let mut curl_v = vec![Fix128::ZERO; g.v.len()];
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let ix = i + (n + 1) * (j + n * k);
                curl_u[ix] = (psi(i, j + 1) - psi(i, j)) / dx;
                let grad = if i == 0 || i == n {
                    Fix128::ZERO
                } else {
                    (q(i, j, k) - q(i - 1, j, k)) / dx
                };
                g.u[ix] = grad + curl_u[ix];
            }
        }
        for j in 0..=n {
            for i in 0..n {
                let ix = i + n * (j + (n + 1) * k);
                curl_v[ix] = -((psi(i + 1, j) - psi(i, j)) / dx);
                let grad = if j == 0 || j == n {
                    Fix128::ZERO
                } else {
                    (q(i, j, k) - q(i, j - 1, k)) / dx
                };
                g.v[ix] = grad + curl_v[ix];
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                g.w[i + n * (j + n * k)] = if k == 0 || k == n {
                    Fix128::ZERO
                } else {
                    (q(i, j, k) - q(i, j, k - 1)) / dx
                };
            }
        }
    }

    let got = distributed(
        &g,
        PressureSolver::DecomposedMultigrid {
            ranks: 4,
            cycles: 40,
        },
    );

    let scale = (rho() / dt()).to_f64();
    let mut vel_err = 0f64;
    let mut vel_max = 0f64;
    for (x, want) in got.u.iter().zip(&curl_u).chain(got.v.iter().zip(&curl_v)) {
        vel_err = vel_err.max((x.to_f64() - want.to_f64()).abs());
        vel_max = vel_max.max(want.to_f64().abs());
    }
    for x in &got.w {
        vel_err = vel_err.max(x.to_f64().abs());
    }
    assert!(vel_max > 1.0, "the curl part must be non-trivial");
    assert!(
        vel_err < 1e-10 * vel_max,
        "velocity is not the curl part: max error {vel_err:e} against {vel_max:e}",
    );

    let p = |i: usize, j: usize, k: usize| got.pressure[i + n * (j + n * k)].to_f64();
    let mut p_err = 0f64;
    let mut p_max = 0f64;
    for k in 0..n {
        for j in 0..n {
            for i in 1..n {
                let want = scale * (q(i, j, k) - q(i - 1, j, k)).to_f64();
                p_err = p_err.max((p(i, j, k) - p(i - 1, j, k) - want).abs());
                p_max = p_max.max(want.abs());
            }
        }
    }
    assert!(
        p_err < 1e-10 * p_max,
        "pressure differences are not ρ/dt · Δq: max error {p_err:e} against {p_max:e}",
    );
}

/// Oracle 4: every refusal names its reason and leaves the grid untouched.
#[test]
fn degenerate_requests_are_refused_and_leave_the_grid_untouched() {
    let base = seeded(8, 8, 8, Scene::Mixed);
    let gs = |ranks, sweeps| PressureSolver::BandedGs { ranks, sweeps };
    let cases: Vec<(MacGrid, Fix128, Fix128, PressureSolver, PressureSolverError)> = vec![
        (
            base.clone(),
            dt(),
            rho(),
            gs(0, 3),
            PressureSolverError::ZeroRanks,
        ),
        (
            base.clone(),
            dt(),
            rho(),
            PressureSolver::DecomposedGs {
                ranks: 0,
                sweeps: 3,
            },
            PressureSolverError::ZeroRanks,
        ),
        (
            base.clone(),
            dt(),
            rho(),
            PressureSolver::DecomposedMultigrid {
                ranks: 0,
                cycles: 2,
            },
            PressureSolverError::ZeroRanks,
        ),
        (
            base.clone(),
            dt(),
            rho(),
            gs(2, 0),
            PressureSolverError::ZeroIterations,
        ),
        (
            base.clone(),
            dt(),
            rho(),
            PressureSolver::DecomposedMultigrid {
                ranks: 2,
                cycles: 0,
            },
            PressureSolverError::ZeroIterations,
        ),
        (
            base.clone(),
            Fix128::ZERO,
            rho(),
            gs(2, 3),
            PressureSolverError::ZeroTimeStep,
        ),
        (
            base.clone(),
            dt(),
            Fix128::ZERO,
            gs(2, 3),
            PressureSolverError::ZeroDensity,
        ),
        (
            MacGrid::new(8, 8, 8, Fix128::ZERO),
            dt(),
            rho(),
            gs(2, 3),
            PressureSolverError::ZeroSpacing,
        ),
        (
            seeded(8, 8, 12, Scene::Open),
            dt(),
            rho(),
            PressureSolver::DecomposedMultigrid {
                ranks: 2,
                cycles: 2,
            },
            PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                extents: (8, 8, 12),
            },
        ),
        (
            base.clone(),
            dt(),
            rho(),
            PressureSolver::RedBlackGs { sweeps: 3 },
            PressureSolverError::NotDecomposed,
        ),
        (
            base.clone(),
            dt(),
            rho(),
            PressureSolver::Multigrid { cycles: 2 },
            PressureSolverError::NotDecomposed,
        ),
    ];
    for (grid, dt_s, density, solver, want) in cases {
        let mut g = grid.clone();
        let got = project_pressure_distributed(&mut g, dt_s, density, solver);
        assert_eq!(got, Err(want), "{solver:?}");
        assert!(bits_eq(&g, &grid), "{solver:?} touched the grid on refusal");
    }
}
