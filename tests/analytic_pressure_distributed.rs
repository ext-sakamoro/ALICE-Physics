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
//! 5. **Low-end outflows.** An outflow face at the low end of every axis, next
//!    to an inflow or a wall, through every solver and rank count: the ranks
//!    impose the face conditions on their own faces, and the order they do it in
//!    has to be the single-process one for oracle 1 to hold. Teeth: the
//!    reference's inward neighbours change under enforcement on every low-end
//!    face, so reading them before or after is a different field.
//! 6. **Working set.** [`project_pressure_distributed_with_report`] reports, per
//!    rank, the bytes the documented closed form gives from the rank's slab
//!    (`DistributedProjectionReport`'s table: 19 per held face, 16 per band cell,
//!    38 per owned cell), the total is their sum, and a rank of a many-rank run
//!    holds a fraction of what a single rank holds.

#![cfg(feature = "std")]

use alice_physics::cfd_solver::{PressureSolver, PressureSolverError};
use alice_physics::eulerian_grid::{
    project_pressure, project_pressure_distributed, project_pressure_distributed_with_report,
    project_pressure_multigrid, FaceBc, MacGrid,
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
    /// An outflow on the low face of every axis, and on the face one cell
    /// inward an inflow or a wall at rest (X), a moving wall or a symmetry plane
    /// (Y), an inflow or a wall at rest (Z); outflows on the high faces too.
    LowEnd,
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
    if let Scene::LowEnd = scene {
        let rest = FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        };
        for k in 0..nz {
            for j in 0..ny {
                g.set_u_bc(0, j, k, FaceBc::Outflow);
                let inward = if (j + k) % 2 == 0 {
                    FaceBc::Inflow {
                        normal_velocity: Fix128::from_ratio(3, 4),
                    }
                } else {
                    rest
                };
                g.set_u_bc(1, j, k, inward);
                g.set_u_bc(nx, j, k, FaceBc::Outflow);
            }
            for i in 0..nx {
                g.set_v_bc(i, 0, k, FaceBc::Outflow);
                let inward = if (i + k) % 2 == 0 {
                    FaceBc::Wall {
                        velocity: Vec3Fix::new(
                            Fix128::from_ratio(1, 2),
                            Fix128::ZERO,
                            Fix128::ZERO,
                        ),
                    }
                } else {
                    FaceBc::SlipWall
                };
                g.set_v_bc(i, 1, k, inward);
                g.set_v_bc(i, ny, k, FaceBc::Outflow);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                g.set_w_bc(i, j, 0, FaceBc::Outflow);
                let inward = if (i + j) % 2 == 0 {
                    FaceBc::Inflow {
                        normal_velocity: Fix128::from_ratio(-5, 8),
                    }
                } else {
                    rest
                };
                g.set_w_bc(i, j, 1, inward);
                g.set_w_bc(i, j, nz, FaceBc::Outflow);
            }
        }
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

/// Oracle 5: outflow faces at the low end of every axis, next to inflow and wall
/// faces, through all three decomposed solvers and every rank count — counts
/// that divide `nz`, counts that do not, and counts above it.
#[test]
fn low_end_outflows_next_to_inflow_or_walls_stay_bit_identical() {
    // Teeth: on every low-end face the inward neighbour's value changes when the
    // conditions are imposed, so an outflow that copied it before its own
    // condition was imposed would be a different number.
    let probe = seeded(8, 8, 8, Scene::LowEnd);
    let mut enforced = probe.clone();
    enforced.enforce_face_boundaries();
    let (nx, ny, nz) = (8usize, 8usize, 8usize);
    let mut changed = [0usize; 3];
    for k in 0..nz {
        for j in 0..ny {
            let ix = 1 + (nx + 1) * (j + ny * k);
            changed[0] += usize::from(probe.u[ix] != enforced.u[ix]);
        }
        for i in 0..nx {
            let ix = i + nx * (1 + (ny + 1) * k);
            changed[1] += usize::from(probe.v[ix] != enforced.v[ix]);
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let ix = i + nx * (j + ny);
            changed[2] += usize::from(probe.w[ix] != enforced.w[ix]);
        }
    }
    assert!(
        changed.iter().all(|&c| c > nx * ny / 2),
        "the inward neighbours of the low-end outflows must change under enforcement on \
         every axis: {changed:?}",
    );

    const SWEEPS: u32 = 3;
    for (nx, ny, nz) in [(8, 8, 8), (6, 5, 12)] {
        let base = seeded(nx, ny, nz, Scene::LowEnd);
        let want = single_gs(&base, SWEEPS);
        assert!(!bits_eq(&want, &single_gs(&base, SWEEPS + 1)));
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
                    "{nx}x{ny}x{nz} LowEnd {solver:?}: {}",
                    first_difference(&got, &want),
                );
            }
        }
    }

    const CYCLES: u32 = 2;
    for ((nx, ny, nz), rank_counts) in [
        ((16, 16, 16), &[1usize, 2, 3, 4, 8, 20][..]),
        ((8, 4, 32), &[2, 5, 8][..]),
    ] {
        let base = seeded(nx, ny, nz, Scene::LowEnd);
        let want = single_mg(&base, CYCLES);
        assert!(!bits_eq(&want, &single_mg(&base, CYCLES + 1)));
        for &ranks in rank_counts {
            let solver = PressureSolver::DecomposedMultigrid {
                ranks,
                cycles: CYCLES,
            };
            let got = distributed(&base, solver);
            assert!(
                bits_eq(&got, &want),
                "{nx}x{ny}x{nz} LowEnd {solver:?}: {}",
                first_difference(&got, &want),
            );
        }
    }
}

/// The closed form of one rank's working set, from the table on
/// `DistributedProjectionReport`: a rank owning layers `k0..k1` of an
/// `nx × ny × nz` grid holds the X- and Y-faces of those layers and the Z-faces
/// `k0..=k1` (16 + 3 bytes each), a pressure band of its layers widened by one
/// halo layer each side within the grid (16 bytes a cell), and per owned cell the
/// open-face mask (6), the inverse degree (16) and the right-hand side (16).
fn closed_form_rank_bytes(nx: usize, ny: usize, nz: usize, (k0, k1): (usize, usize)) -> usize {
    if k0 == k1 {
        return 0;
    }
    let layers = k1 - k0;
    let faces = layers * (nx + 1) * ny + layers * nx * (ny + 1) + (layers + 1) * nx * ny;
    let band_layers = (k1 + 1).min(nz) - k0.saturating_sub(1);
    let cells = layers * nx * ny;
    faces * (16 + 3) + band_layers * nx * ny * 16 + cells * (6 + 16 + 16)
}

/// Equal slabs, `rank · nz / ranks`: the Gauss-Seidel decomposition for every
/// count, and the multigrid one for a power-of-two count on a power-of-two grid
/// (where every level down to the last distributed one splits evenly).
fn even_bounds(nz: usize, ranks: usize, rank: usize) -> (usize, usize) {
    (rank * nz / ranks, (rank + 1) * nz / ranks)
}

fn report(
    base: &MacGrid,
    solver: PressureSolver,
) -> alice_physics::eulerian_grid::DistributedProjectionReport {
    let mut g = base.clone();
    let got = project_pressure_distributed_with_report(&mut g, dt(), rho(), solver)
        .expect("a valid request");
    let mut plain = base.clone();
    project_pressure_distributed(&mut plain, dt(), rho(), solver).expect("a valid request");
    assert!(
        bits_eq(&g, &plain),
        "{solver:?}: the reporting call must give the same answer as the plain one",
    );
    got
}

/// Oracle 6: the per-rank working set is the closed form of the rank's slab, the
/// total is the sum, and a rank of an 8-rank run holds a fraction of what the
/// single rank of a 1-rank run holds.
#[test]
fn each_rank_reports_the_working_set_of_its_slab() {
    for (nx, ny, nz) in [(6, 5, 12), (8, 8, 8)] {
        let base = seeded(nx, ny, nz, Scene::Mixed);
        for ranks in [1, 2, 3, 5, 8, nz + 7] {
            let got = report(&base, PressureSolver::BandedGs { ranks, sweeps: 2 });
            let want: Vec<usize> = (0..ranks)
                .map(|r| closed_form_rank_bytes(nx, ny, nz, even_bounds(nz, ranks, r)))
                .collect();
            assert_eq!(got.rank_bytes, want, "{nx}x{ny}x{nz} BandedGs over {ranks}");
            assert_eq!(got.total_bytes(), Some(want.iter().sum()));
            assert_eq!(got.max_rank_bytes(), want.iter().copied().max());
        }
    }
    for (nx, ny, nz) in [(16, 16, 16), (8, 4, 32)] {
        let base = seeded(nx, ny, nz, Scene::Mixed);
        for ranks in [1, 2, 4, 8] {
            let got = report(
                &base,
                PressureSolver::DecomposedMultigrid { ranks, cycles: 1 },
            );
            let want: Vec<usize> = (0..ranks)
                .map(|r| closed_form_rank_bytes(nx, ny, nz, even_bounds(nz, ranks, r)))
                .collect();
            assert_eq!(
                got.rank_bytes, want,
                "{nx}x{ny}x{nz} DecomposedMultigrid over {ranks}",
            );
            assert_eq!(got.total_bytes(), Some(want.iter().sum()));
        }
    }

    // A slab, not the grid: 8 ranks of a 32-layer grid each hold 4 layers plus
    // halo, under a quarter of the one rank that holds all 32.
    let base = seeded(16, 16, 32, Scene::Open);
    let gs = |ranks| PressureSolver::BandedGs { ranks, sweeps: 1 };
    let whole = report(&base, gs(1)).max_rank_bytes().expect("one rank");
    let banded = report(&base, gs(8)).max_rank_bytes().expect("eight ranks");
    assert!(
        banded * 4 < whole,
        "a rank of 8 holds {banded} bytes against {whole} for the whole grid",
    );

    // Ranks that each copy the whole grid hold no slab, and say so.
    let copies = report(
        &base,
        PressureSolver::DecomposedGs {
            ranks: 3,
            sweeps: 1,
        },
    );
    assert!(copies.rank_bytes.is_empty());
    assert_eq!(copies.max_rank_bytes(), None);
    assert_eq!(copies.total_bytes(), None);
}

/// The reporting entry refuses what the plain one refuses, leaving the grid as
/// it was.
#[test]
fn the_reporting_entry_refuses_what_the_plain_one_refuses() {
    let base = seeded(8, 8, 12, Scene::Mixed);
    for (solver, want) in [
        (
            PressureSolver::BandedGs {
                ranks: 0,
                sweeps: 1,
            },
            PressureSolverError::ZeroRanks,
        ),
        (
            PressureSolver::DecomposedMultigrid {
                ranks: 2,
                cycles: 1,
            },
            PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                extents: (8, 8, 12),
            },
        ),
        (
            PressureSolver::RedBlackGs { sweeps: 1 },
            PressureSolverError::NotDecomposed,
        ),
    ] {
        let mut g = base.clone();
        let got = project_pressure_distributed_with_report(&mut g, dt(), rho(), solver);
        assert_eq!(got, Err(want), "{solver:?}");
        assert!(bits_eq(&g, &base), "{solver:?} touched the grid on refusal");
    }
}
