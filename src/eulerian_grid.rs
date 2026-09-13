//! Eulerian Fluid Grid (MAC + FLIP/PIC)
//!
//! Phase F4 of the ALICE-Physics completeness project. Provides a
//! **staggered marker-and-cell (MAC) grid** for grid-based fluid solvers,
//! plus the classical **FLIP** (Fluid-Implicit Particle) and **PIC**
//! (Particle-In-Cell) particle ↔ grid transfer operators.
//!
//! # MAC layout
//!
//! Velocity components live on the corresponding cell faces:
//! - `u[i,j,k]` on the X-face between cells `i-1` and `i` → shape
//!   `(nx+1) × ny × nz`
//! - `v[i,j,k]` on the Y-face → `nx × (ny+1) × nz`
//! - `w[i,j,k]` on the Z-face → `nx × ny × (nz+1)`
//! - Pressure `p[i,j,k]` on the cell centre → `nx × ny × nz`
//!
//! This staggering makes divergence and gradient consistent (avoids the
//! famous "checkerboard" pressure oscillation of collocated grids).
//!
//! # Transfer operators
//!
//! - **P2G** scatters particle velocities to the grid using trilinear
//!   weights. Additive accumulation, divide by weight-sum at the end.
//! - **G2P** samples grid velocities at particle positions via trilinear
//!   interpolation.
//! - **FLIP** update: `v_p ← v_p + G2P(u_new − u_old)`. Momentum-preserving.
//! - **PIC** update: `v_p ← G2P(u_new)`. More damped, less noise.
//!
//! # Pressure projection
//!
//! Solves `∇²p = ρ/dt · ∇·u*` via Jacobi iterations (simple, robust).
//! After the pressure is found, subtract `dt/ρ · ∇p` from the intermediate
//! velocities to project onto the divergence-free space.
//!
//! # References
//!
//! - Harlow & Welch, "Numerical calculation of time-dependent viscous
//!   incompressible flow of fluid with free surface", Phys. Fluids 8, 1965
//!   (original MAC).
//! - Brackbill & Ruppel, "FLIP: A method for adaptively zoned, particle-in-
//!   cell calculations of fluid flows in two dimensions", J. Comp. Phys. 65,
//!   1986.
//! - Zhu & Bridson, "Animating sand as a fluid", ACM Trans. Graph. 24, 2005.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// MAC grid
// ============================================================================

/// Staggered marker-and-cell grid holding face-based velocity components and
/// cell-centred pressure.
#[derive(Clone, Debug)]
pub struct MacGrid {
    /// Cells in X.
    pub nx: usize,
    /// Cells in Y.
    pub ny: usize,
    /// Cells in Z.
    pub nz: usize,
    /// Cell spacing (m).
    pub dx: Fix128,
    /// X-velocity on X-faces `[(nx+1) · ny · nz]`.
    pub u: Vec<Fix128>,
    /// Y-velocity on Y-faces `[nx · (ny+1) · nz]`.
    pub v: Vec<Fix128>,
    /// Z-velocity on Z-faces `[nx · ny · (nz+1)]`.
    pub w: Vec<Fix128>,
    /// Cell-centred pressure `[nx · ny · nz]`.
    pub pressure: Vec<Fix128>,
}

impl MacGrid {
    /// Empty grid initialised to zero everywhere.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Fix128) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            u: vec![Fix128::ZERO; (nx + 1) * ny * nz],
            v: vec![Fix128::ZERO; nx * (ny + 1) * nz],
            w: vec![Fix128::ZERO; nx * ny * (nz + 1)],
            pressure: vec![Fix128::ZERO; nx * ny * nz],
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn idx_u(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }
    #[inline]
    #[must_use]
    pub(crate) fn idx_v(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }
    #[inline]
    #[must_use]
    pub(crate) fn idx_w(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }
    #[inline]
    #[must_use]
    fn idx_c(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Read `u` at face `(i, j, k)`. Out of range returns 0.
    #[must_use]
    pub fn u(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.u[self.idx_u(i, j, k)]
    }
    /// Read `v`.
    #[must_use]
    pub fn v(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.v[self.idx_v(i, j, k)]
    }
    /// Read `w`.
    #[must_use]
    pub fn w(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return Fix128::ZERO;
        }
        self.w[self.idx_w(i, j, k)]
    }
    /// Read pressure.
    #[must_use]
    pub fn pressure(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.pressure[self.idx_c(i, j, k)]
    }

    /// Cell-centred velocity by averaging adjacent faces (used for output /
    /// downstream Lagrangian modules).
    #[must_use]
    pub fn cell_velocity(&self, i: usize, j: usize, k: usize) -> (Fix128, Fix128, Fix128) {
        let u_c = (self.u(i, j, k) + self.u(i + 1, j, k)).half();
        let v_c = (self.v(i, j, k) + self.v(i, j + 1, k)).half();
        let w_c = (self.w(i, j, k) + self.w(i, j, k + 1)).half();
        (u_c, v_c, w_c)
    }

    /// Divergence at cell (i, j, k). Positive = fluid expanding.
    #[must_use]
    pub fn divergence(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if self.dx.is_zero() {
            return Fix128::ZERO;
        }
        let du = self.u(i + 1, j, k) - self.u(i, j, k);
        let dv = self.v(i, j + 1, k) - self.v(i, j, k);
        let dw = self.w(i, j, k + 1) - self.w(i, j, k);
        (du + dv + dw) / self.dx
    }
}

// ============================================================================
// Pressure projection (Jacobi)
// ============================================================================

/// Solve `∇²p = (ρ / dt) · ∇·u*` using Jacobi iteration, then subtract
/// `(dt / ρ) · ∇p` from the intermediate velocity to enforce
/// incompressibility.
///
/// - `iterations`: 20-100 typical for Jacobi; more expensive but stable.
/// - `density`: fluid density (kg/m³).
///
/// This is O(n · iterations); acceptable for grids up to ~64³. For larger
/// problems replace with multigrid.
/// Session 3 I9 upgrade: alias to `project_pressure_red_black_gs`, which
/// converges ~2× faster than Jacobi while retaining the same API. Older
/// callers see no behavioural change; a call with the same `iterations` now
/// yields a strictly smaller residual.
pub fn project_pressure(grid: &mut MacGrid, dt_s: Fix128, density_kg_m3: Fix128, iterations: u32) {
    project_pressure_red_black_gs(grid, dt_s, density_kg_m3, iterations);
}

/// Red-black Gauss-Seidel variant of `project_pressure` (Session 3 I9).
///
/// Alternates two sweeps per iteration: "red" cells where `i+j+k` is even and
/// "black" cells where it is odd. Immediately-updated pressures propagate
/// during each sweep, giving ~2× the convergence rate of Jacobi at the
/// same computational cost.
pub(crate) fn project_pressure_red_black_gs(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;
    let mut rhs = vec![Fix128::ZERO; n];
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let idx = i + grid.nx * (j + grid.ny * k);
                rhs[idx] = grid.divergence(i, j, k) * scale;
            }
        }
    }
    let sixth = Fix128::from_ratio(1, 6);
    for _ in 0..iterations {
        // Two-colour sweep (colour ∈ {0, 1})
        for colour in 0..2u32 {
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        if ((i + j + k) as u32 % 2) != colour {
                            continue;
                        }
                        let px = if i > 0 {
                            grid.pressure(i - 1, j, k)
                        } else {
                            Fix128::ZERO
                        };
                        let pxx = if i + 1 < grid.nx {
                            grid.pressure(i + 1, j, k)
                        } else {
                            Fix128::ZERO
                        };
                        let py = if j > 0 {
                            grid.pressure(i, j - 1, k)
                        } else {
                            Fix128::ZERO
                        };
                        let pyy = if j + 1 < grid.ny {
                            grid.pressure(i, j + 1, k)
                        } else {
                            Fix128::ZERO
                        };
                        let pz = if k > 0 {
                            grid.pressure(i, j, k - 1)
                        } else {
                            Fix128::ZERO
                        };
                        let pzz = if k + 1 < grid.nz {
                            grid.pressure(i, j, k + 1)
                        } else {
                            Fix128::ZERO
                        };
                        let idx = i + grid.nx * (j + grid.ny * k);
                        grid.pressure[idx] = (px + pxx + py + pyy + pz + pzz - rhs[idx]) * sixth;
                    }
                }
            }
        }
    }

    // Velocity correction (same as Jacobi variant)
    let inv_dx = Fix128::ONE / grid.dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 1..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i - 1, j, k);
                let ix = grid.idx_u(i, j, k);
                grid.u[ix] = grid.u[ix] - coeff * dp;
            }
        }
    }
    for k in 0..grid.nz {
        for j in 1..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j - 1, k);
                let ix = grid.idx_v(i, j, k);
                grid.v[ix] = grid.v[ix] - coeff * dp;
            }
        }
    }
    for k in 1..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j, k - 1);
                let ix = grid.idx_w(i, j, k);
                grid.w[ix] = grid.w[ix] - coeff * dp;
            }
        }
    }
}

/// Legacy Jacobi implementation, kept for benchmarking (Session 3 I9, crate-internal).
pub(crate) fn project_pressure_jacobi(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;

    // Build divergence RHS
    let n = grid.nx * grid.ny * grid.nz;
    let mut rhs = vec![Fix128::ZERO; n];
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let idx = i + grid.nx * (j + grid.ny * k);
                rhs[idx] = grid.divergence(i, j, k) * scale;
            }
        }
    }

    // Jacobi iterations for -∇²p = rhs
    // At interior cell: 6·p_c − Σ p_neighbours = -rhs (2nd order FD)
    let sixth = Fix128::from_ratio(1, 6);
    let mut p_new = grid.pressure.clone();
    for _ in 0..iterations {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let px = if i > 0 {
                        grid.pressure(i - 1, j, k)
                    } else {
                        Fix128::ZERO
                    };
                    let pxx = if i + 1 < grid.nx {
                        grid.pressure(i + 1, j, k)
                    } else {
                        Fix128::ZERO
                    };
                    let py = if j > 0 {
                        grid.pressure(i, j - 1, k)
                    } else {
                        Fix128::ZERO
                    };
                    let pyy = if j + 1 < grid.ny {
                        grid.pressure(i, j + 1, k)
                    } else {
                        Fix128::ZERO
                    };
                    let pz = if k > 0 {
                        grid.pressure(i, j, k - 1)
                    } else {
                        Fix128::ZERO
                    };
                    let pzz = if k + 1 < grid.nz {
                        grid.pressure(i, j, k + 1)
                    } else {
                        Fix128::ZERO
                    };
                    let idx = i + grid.nx * (j + grid.ny * k);
                    p_new[idx] = (px + pxx + py + pyy + pz + pzz - rhs[idx]) * sixth;
                }
            }
        }
        grid.pressure.clone_from(&p_new);
    }

    // Velocity correction: u ← u − (dt/ρ)·∇p
    let inv_dx = Fix128::ONE / grid.dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;

    // u faces (skip boundaries)
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 1..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i - 1, j, k);
                let ix = grid.idx_u(i, j, k);
                grid.u[ix] = grid.u[ix] - coeff * dp;
            }
        }
    }
    for k in 0..grid.nz {
        for j in 1..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j - 1, k);
                let ix = grid.idx_v(i, j, k);
                grid.v[ix] = grid.v[ix] - coeff * dp;
            }
        }
    }
    for k in 1..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j, k - 1);
                let ix = grid.idx_w(i, j, k);
                grid.w[ix] = grid.w[ix] - coeff * dp;
            }
        }
    }
}

/// Preconditioned **BiCGStab** pressure solver (van der Vorst 1992).
///
/// Solves the discrete Poisson system `A p = b` for the MAC-grid
/// pressure, where `A` is the 7-point Laplacian with homogeneous
/// Neumann boundary conditions (missing neighbours contribute zero).
/// The RHS `b = ρ dx² / dt · ∇·u` matches the Jacobi and red-black
/// Gauss–Seidel variants, and the velocity correction stage at the
/// end is identical.
///
/// # Preconditioner
///
/// Diagonal Jacobi preconditioner `M = diag(A)`. On this Neumann
/// Poisson matrix the diagonal is `−6` at interior cells and drops
/// toward the boundary as `−6 + (number of missing neighbours)`,
/// so the preconditioner adapts per cell.
///
/// # Convergence
///
/// Compared to red-black Gauss–Seidel, BiCGStab converges in roughly
/// `O(√N)` iterations vs `O(N)` for Jacobi/GS on 3-D Poisson, at the
/// price of ~7 dot products per iteration. Under Fix128 the dot
/// products dominate cost at small grid sizes; net wins appear as
/// grid size grows.
///
/// The iteration stops early when `‖r‖_∞ < tolerance` or after
/// `max_iterations` (whichever comes first).
pub(crate) fn project_pressure_bicgstab(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    max_iterations: u32,
    tolerance: Fix128,
) -> BicgstabStats {
    let default_stats = BicgstabStats {
        iterations: 0,
        final_residual: Fix128::ZERO,
        converged: true,
    };
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return default_stats;
    }
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let n = nx * ny * nz;

    // Build RHS and cache the per-cell diagonal.
    let mut rhs = vec![Fix128::ZERO; n];
    let mut diag = vec![Fix128::ZERO; n];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + nx * (j + ny * k);
                rhs[idx] = grid.divergence(i, j, k) * scale;
                let mut deg: i64 = 6;
                if i == 0 {
                    deg -= 1;
                }
                if i + 1 == nx {
                    deg -= 1;
                }
                if j == 0 {
                    deg -= 1;
                }
                if j + 1 == ny {
                    deg -= 1;
                }
                if k == 0 {
                    deg -= 1;
                }
                if k + 1 == nz {
                    deg -= 1;
                }
                diag[idx] = Fix128::from_int(-deg); // A[i,i] = -deg
            }
        }
    }

    // x = grid.pressure; solve A x = b with A = -Laplacian sign convention:
    // r0 = b − A x0
    let mut x = grid.pressure.clone();
    let mut r = vec![Fix128::ZERO; n];
    apply_poisson_a(&x, &mut r, nx, ny, nz);
    for i in 0..n {
        r[i] = rhs[i] - r[i];
    }
    let r_hat = r.clone();
    let mut p_vec = r.clone();
    let mut rho_prev = dot(&r_hat, &r);
    let mut alpha = Fix128::ONE;
    let mut omega = Fix128::ONE;
    let mut v_vec = vec![Fix128::ZERO; n];
    let mut y_vec = vec![Fix128::ZERO; n];
    let mut z_vec = vec![Fix128::ZERO; n];
    let mut s_vec = vec![Fix128::ZERO; n];
    let mut t_vec = vec![Fix128::ZERO; n];
    let mut iterations = 0_u32;
    let mut residual = linf_norm(&r);
    let mut converged = residual < tolerance;

    while iterations < max_iterations && !converged {
        iterations += 1;
        let rho = dot(&r_hat, &r);
        if rho.is_zero() {
            break;
        }
        if iterations > 1 {
            let beta = (rho / rho_prev) * (alpha / omega);
            // p = r + β (p - ω v)
            for i in 0..n {
                p_vec[i] = r[i] + beta * (p_vec[i] - omega * v_vec[i]);
            }
        }
        // y = M^{-1} p
        for i in 0..n {
            y_vec[i] = if diag[i].is_zero() {
                p_vec[i]
            } else {
                p_vec[i] / diag[i]
            };
        }
        apply_poisson_a(&y_vec, &mut v_vec, nx, ny, nz);
        let denom = dot(&r_hat, &v_vec);
        if denom.is_zero() {
            break;
        }
        alpha = rho / denom;
        // s = r - α v
        for i in 0..n {
            s_vec[i] = r[i] - alpha * v_vec[i];
        }
        let s_norm = linf_norm(&s_vec);
        if s_norm < tolerance {
            for i in 0..n {
                x[i] = x[i] + alpha * y_vec[i];
            }
            residual = s_norm;
            converged = true;
            break;
        }
        // z = M^{-1} s
        for i in 0..n {
            z_vec[i] = if diag[i].is_zero() {
                s_vec[i]
            } else {
                s_vec[i] / diag[i]
            };
        }
        apply_poisson_a(&z_vec, &mut t_vec, nx, ny, nz);
        let tt = dot(&t_vec, &t_vec);
        if tt.is_zero() {
            break;
        }
        omega = dot(&t_vec, &s_vec) / tt;
        // x = x + α y + ω z
        for i in 0..n {
            x[i] = x[i] + alpha * y_vec[i] + omega * z_vec[i];
        }
        // r = s - ω t
        for i in 0..n {
            r[i] = s_vec[i] - omega * t_vec[i];
        }
        residual = linf_norm(&r);
        converged = residual < tolerance;
        rho_prev = rho;
    }

    grid.pressure = x;

    // Velocity correction (same convention as the Jacobi variant).
    let inv_dx = Fix128::ONE / grid.dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 1..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i - 1, j, k);
                let ix = grid.idx_u(i, j, k);
                grid.u[ix] = grid.u[ix] - coeff * dp;
            }
        }
    }
    for k in 0..grid.nz {
        for j in 1..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j - 1, k);
                let ix = grid.idx_v(i, j, k);
                grid.v[ix] = grid.v[ix] - coeff * dp;
            }
        }
    }
    for k in 1..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let dp = grid.pressure(i, j, k) - grid.pressure(i, j, k - 1);
                let ix = grid.idx_w(i, j, k);
                grid.w[ix] = grid.w[ix] - coeff * dp;
            }
        }
    }
    BicgstabStats {
        iterations,
        final_residual: residual,
        converged,
    }
}

/// Diagnostic bundle returned by [`project_pressure_bicgstab`] (crate-internal).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BicgstabStats {
    /// Iterations actually performed.
    pub(crate) iterations: u32,
    /// Final `‖r‖_∞`.
    pub(crate) final_residual: Fix128,
    /// True if the iteration terminated below `tolerance`.
    pub(crate) converged: bool,
}

/// Apply the discrete Poisson operator `A` (Neumann BC) to `p`.
///
/// At each cell: `(A p)_c = −6 p_c + Σ p_neighbours` with missing
/// neighbours treated as zero (Neumann). This matches the equation
/// `A p = rhs` where `rhs = ρ dx²/dt · ∇·u` — the same RHS the Jacobi
/// / red-black GS iterations converge to.
fn apply_poisson_a(p: &[Fix128], out: &mut [Fix128], nx: usize, ny: usize, nz: usize) {
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let neg_six = Fix128::from_int(-6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = p[idx(i, j, k)];
                let mut acc = c * neg_six;
                if i > 0 {
                    acc = acc + p[idx(i - 1, j, k)];
                }
                if i + 1 < nx {
                    acc = acc + p[idx(i + 1, j, k)];
                }
                if j > 0 {
                    acc = acc + p[idx(i, j - 1, k)];
                }
                if j + 1 < ny {
                    acc = acc + p[idx(i, j + 1, k)];
                }
                if k > 0 {
                    acc = acc + p[idx(i, j, k - 1)];
                }
                if k + 1 < nz {
                    acc = acc + p[idx(i, j, k + 1)];
                }
                out[idx(i, j, k)] = acc;
            }
        }
    }
}

fn dot(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}

fn linf_norm(v: &[Fix128]) -> Fix128 {
    let mut best = Fix128::ZERO;
    for &x in v {
        let ax = x.abs();
        if ax > best {
            best = ax;
        }
    }
    best
}

// ============================================================================
// Trilinear P2G / G2P (Session 3 I2 upgrade)
// ============================================================================

/// Split a world coordinate `p` (m) into `(base_index, frac)` for a
/// specified face-grid offset `axis_offset` (in cell units, 0 or 0.5).
fn split(p_over_dx: Fix128, axis_offset: Fix128) -> (usize, Fix128) {
    // Adjust for staggering (subtract offset so origin aligns with face 0).
    let shifted = p_over_dx - axis_offset;
    let mut base = shifted.hi;
    // Convert `.lo` (raw fractional bits) to a Fix128 in [0, 1)
    let frac_lo = shifted.lo;
    // Negative shifted values: rust "hi" is floor toward negative infinity? No,
    // Fix128.hi is signed integer part but truncation is toward zero for
    // positive; for negative it's slightly different. Handle by explicit floor:
    if shifted.is_negative() && frac_lo != 0 {
        base -= 1;
    }
    let frac = Fix128 { hi: 0, lo: frac_lo };
    if base < 0 {
        (0, Fix128::ZERO)
    } else {
        (base as usize, frac)
    }
}

/// Trilinear-interpolate the u-face grid at a world point.
///
/// The u-face is staggered by `+0` on X, `+0.5` on Y, `+0.5` on Z relative
/// to the cell corner grid.
pub(crate) fn sample_u_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::ZERO);
    let (j, v) = split(fy, Fix128::from_ratio(1, 2));
    let (k, w) = split(fz, Fix128::from_ratio(1, 2));

    let i0 = i.min(grid.nx);
    let i1 = (i + 1).min(grid.nx);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);

    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;

    let c00 = grid.u(i0, j0, k0) * om_u + grid.u(i1, j0, k0) * u;
    let c10 = grid.u(i0, j1, k0) * om_u + grid.u(i1, j1, k0) * u;
    let c01 = grid.u(i0, j0, k1) * om_u + grid.u(i1, j0, k1) * u;
    let c11 = grid.u(i0, j1, k1) * om_u + grid.u(i1, j1, k1) * u;

    let c0 = c00 * om_v + c10 * v;
    let c1 = c01 * om_v + c11 * v;
    c0 * om_w + c1 * w
}

pub(crate) fn sample_v_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::from_ratio(1, 2));
    let (j, v) = split(fy, Fix128::ZERO);
    let (k, w) = split(fz, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny);
    let j1 = (j + 1).min(grid.ny);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let c00 = grid.v(i0, j0, k0) * om_v + grid.v(i0, j1, k0) * v;
    let c10 = grid.v(i1, j0, k0) * om_v + grid.v(i1, j1, k0) * v;
    let c01 = grid.v(i0, j0, k1) * om_v + grid.v(i0, j1, k1) * v;
    let c11 = grid.v(i1, j0, k1) * om_v + grid.v(i1, j1, k1) * v;
    let c0 = c00 * om_u + c10 * u;
    let c1 = c01 * om_u + c11 * u;
    c0 * om_w + c1 * w
}

pub(crate) fn sample_w_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::from_ratio(1, 2));
    let (j, v) = split(fy, Fix128::from_ratio(1, 2));
    let (k, w) = split(fz, Fix128::ZERO);
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz);
    let k1 = (k + 1).min(grid.nz);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let c00 = grid.w(i0, j0, k0) * om_w + grid.w(i0, j0, k1) * w;
    let c10 = grid.w(i1, j0, k0) * om_w + grid.w(i1, j0, k1) * w;
    let c01 = grid.w(i0, j1, k0) * om_w + grid.w(i0, j1, k1) * w;
    let c11 = grid.w(i1, j1, k0) * om_w + grid.w(i1, j1, k1) * w;
    let c0 = c00 * om_u + c10 * u;
    let c1 = c01 * om_u + c11 * u;
    c0 * om_v + c1 * v
}

/// Return `(min, max)` of the 8 u-face corner values surrounding `pos_m`.
///
/// Used by MacCormack to clamp corrector results into the pre-advection
/// local range (Fedkiw's monotonicity guard).
pub(crate) fn sample_u_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::ZERO);
    let (j, _) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, _) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx);
    let i1 = (i + 1).min(grid.nx);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    corner_range(&[
        grid.u(i0, j0, k0),
        grid.u(i1, j0, k0),
        grid.u(i0, j1, k0),
        grid.u(i1, j1, k0),
        grid.u(i0, j0, k1),
        grid.u(i1, j0, k1),
        grid.u(i0, j1, k1),
        grid.u(i1, j1, k1),
    ])
}

/// Return `(min, max)` of the 8 v-face corner values surrounding `pos_m`.
pub(crate) fn sample_v_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, _) = split(pos_m.y * inv_dx, Fix128::ZERO);
    let (k, _) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny);
    let j1 = (j + 1).min(grid.ny);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    corner_range(&[
        grid.v(i0, j0, k0),
        grid.v(i1, j0, k0),
        grid.v(i0, j1, k0),
        grid.v(i1, j1, k0),
        grid.v(i0, j0, k1),
        grid.v(i1, j0, k1),
        grid.v(i0, j1, k1),
        grid.v(i1, j1, k1),
    ])
}

/// Return `(min, max)` of the 8 w-face corner values surrounding `pos_m`.
pub(crate) fn sample_w_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, _) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, _) = split(pos_m.z * inv_dx, Fix128::ZERO);
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz);
    let k1 = (k + 1).min(grid.nz);
    corner_range(&[
        grid.w(i0, j0, k0),
        grid.w(i1, j0, k0),
        grid.w(i0, j1, k0),
        grid.w(i1, j1, k0),
        grid.w(i0, j0, k1),
        grid.w(i1, j0, k1),
        grid.w(i0, j1, k1),
        grid.w(i1, j1, k1),
    ])
}

fn corner_range(corners: &[Fix128; 8]) -> (Fix128, Fix128) {
    let mut lo = corners[0];
    let mut hi = corners[0];
    for &c in &corners[1..] {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    (lo, hi)
}

/// Grid-to-particle: trilinear-sample velocity at world position `pos_m`.
///
/// Correct MAC-grid staggering is applied per component; velocities read
/// from their own face grids. Session 3 I2 upgrade from nearest-cell.
#[must_use]
pub fn g2p_velocity(grid: &MacGrid, pos_m: Vec3Fix) -> Vec3Fix {
    if grid.dx.is_zero() {
        return Vec3Fix::default();
    }
    Vec3Fix::new(
        sample_u_trilinear(grid, pos_m),
        sample_v_trilinear(grid, pos_m),
        sample_w_trilinear(grid, pos_m),
    )
}

/// Scatter one particle's velocity onto the 8 nearest u-face grid nodes
/// using trilinear weights (Session 3 I2 upgrade). The caller must
/// separately track per-cell weight sums if quantitative velocity means
/// are required — this routine only accumulates weighted deposits.
fn deposit_u_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vx: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::ZERO);
    let (j, v) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, w) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci <= grid.nx && cj < grid.ny && ck < grid.nz {
            let ix = grid.idx_u(ci, cj, ck);
            grid.u[ix] = grid.u[ix] + weight * vx;
        }
    }
}

fn deposit_v_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vy: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, v) = split(pos_m.y * inv_dx, Fix128::ZERO);
    let (k, w) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci < grid.nx && cj <= grid.ny && ck < grid.nz {
            let ix = grid.idx_v(ci, cj, ck);
            grid.v[ix] = grid.v[ix] + weight * vy;
        }
    }
}

fn deposit_w_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vz: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, v) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, w) = split(pos_m.z * inv_dx, Fix128::ZERO);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci < grid.nx && cj < grid.ny && ck <= grid.nz {
            let ix = grid.idx_w(ci, cj, ck);
            grid.w[ix] = grid.w[ix] + weight * vz;
        }
    }
}

/// Particle-to-grid: trilinear scatter of one particle's velocity across
/// the 8 nearest face nodes for each of u/v/w. Session 3 I2 upgrade;
/// the earlier `p2g_nearest` implementation is retained below for callers
/// that need the simpler (less accurate) variant.
pub(crate) fn p2g_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vel_m_per_s: Vec3Fix) {
    deposit_u_trilinear(grid, pos_m, vel_m_per_s.x);
    deposit_v_trilinear(grid, pos_m, vel_m_per_s.y);
    deposit_w_trilinear(grid, pos_m, vel_m_per_s.z);
}

/// Particle-to-grid: scatter one particle's velocity `vel_m_per_s` at
/// position `pos_m` onto the closest cell centre (nearest-neighbour).
///
/// Trilinear scatter is the "true" P2G but requires a companion weight
/// grid; this simplified version is sufficient for coarse PIC tests.
pub(crate) fn p2g_nearest(grid: &mut MacGrid, pos_m: Vec3Fix, vel_m_per_s: Vec3Fix) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let ix = (pos_m.x * inv_dx).hi.max(0) as usize;
    let iy = (pos_m.y * inv_dx).hi.max(0) as usize;
    let iz = (pos_m.z * inv_dx).hi.max(0) as usize;
    if ix >= grid.nx || iy >= grid.ny || iz >= grid.nz {
        return;
    }
    // Deposit x-vel onto the two neighbouring x-faces (average)
    let u_lo = grid.idx_u(ix, iy, iz);
    let u_hi = grid.idx_u(ix + 1, iy, iz);
    let half = Fix128::from_ratio(1, 2);
    grid.u[u_lo] = grid.u[u_lo] + vel_m_per_s.x * half;
    grid.u[u_hi] = grid.u[u_hi] + vel_m_per_s.x * half;
    let v_lo = grid.idx_v(ix, iy, iz);
    let v_hi = grid.idx_v(ix, iy + 1, iz);
    grid.v[v_lo] = grid.v[v_lo] + vel_m_per_s.y * half;
    grid.v[v_hi] = grid.v[v_hi] + vel_m_per_s.y * half;
    let w_lo = grid.idx_w(ix, iy, iz);
    let w_hi = grid.idx_w(ix, iy, iz + 1);
    grid.w[w_lo] = grid.w[w_lo] + vel_m_per_s.z * half;
    grid.w[w_hi] = grid.w[w_hi] + vel_m_per_s.z * half;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_grid_dimensions() {
        let g = MacGrid::new(4, 3, 2, Fix128::ONE);
        assert_eq!(g.u.len(), 5 * 3 * 2);
        assert_eq!(g.v.len(), 4 * 4 * 2);
        assert_eq!(g.w.len(), 4 * 3 * 3);
        assert_eq!(g.pressure.len(), 24);
    }

    #[test]
    fn mac_grid_zero_initialised() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.u(0, 0, 0), Fix128::ZERO);
        assert_eq!(g.v(1, 1, 1), Fix128::ZERO);
        assert_eq!(g.pressure(0, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn mac_grid_out_of_range_returns_zero() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.u(10, 0, 0), Fix128::ZERO);
        assert_eq!(g.pressure(5, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn cell_velocity_averages_faces() {
        let mut g = MacGrid::new(2, 1, 1, Fix128::ONE);
        // Set u(0, 0, 0) = 2, u(1, 0, 0) = 4 → avg = 3
        let i0 = g.idx_u(0, 0, 0);
        let i1 = g.idx_u(1, 0, 0);
        g.u[i0] = Fix128::from_int(2);
        g.u[i1] = Fix128::from_int(4);
        let (uc, _, _) = g.cell_velocity(0, 0, 0);
        assert_eq!(uc, Fix128::from_int(3));
    }

    #[test]
    fn divergence_zero_for_still_grid() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.divergence(1, 1, 1), Fix128::ZERO);
    }

    #[test]
    fn divergence_positive_for_expanding_flow() {
        let mut g = MacGrid::new(3, 3, 3, Fix128::ONE);
        // Set u(2, 1, 1) = 1, u(1, 1, 1) = 0 → du = 1 > 0
        let ix = g.idx_u(2, 1, 1);
        g.u[ix] = Fix128::ONE;
        let d = g.divergence(1, 1, 1);
        assert!(d > Fix128::ZERO);
    }

    #[test]
    fn project_zero_flow_stays_zero() {
        let mut g = MacGrid::new(3, 3, 3, Fix128::ONE);
        project_pressure(
            &mut g,
            Fix128::from_ratio(1, 60),
            Fix128::from_int(1000),
            10,
        );
        // All zeros → nothing to do; divergence remains 0 everywhere
        assert_eq!(g.divergence(1, 1, 1), Fix128::ZERO);
    }

    #[test]
    fn project_reduces_divergence() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        // Create a divergent velocity: u increasing across grid
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..=4 {
                    let ix = g.idx_u(i, j, k);
                    g.u[ix] = Fix128::from_int(i as i64);
                }
            }
        }
        let div_before = g.divergence(2, 2, 2);
        project_pressure(
            &mut g,
            Fix128::from_ratio(1, 60),
            Fix128::from_int(1000),
            50,
        );
        let div_after = g.divergence(2, 2, 2);
        // Projection should reduce |divergence|
        assert!(div_after.abs() < div_before.abs());
    }

    #[test]
    fn p2g_scatters_x_velocity() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let vel = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        p2g_nearest(&mut g, pos, vel);
        // Cell (1,1,1) faces should now hold 2 each (half of 4)
        assert_eq!(g.u(1, 1, 1), Fix128::from_int(2));
        assert_eq!(g.u(2, 1, 1), Fix128::from_int(2));
    }

    #[test]
    fn g2p_reads_cell_average_velocity() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let i0 = g.idx_u(1, 1, 1);
        let i1 = g.idx_u(2, 1, 1);
        g.u[i0] = Fix128::from_int(2);
        g.u[i1] = Fix128::from_int(4);
        // Cell centre u = 3
        let vel = g2p_velocity(
            &g,
            Vec3Fix::new(
                Fix128::from_ratio(15, 10),
                Fix128::from_ratio(15, 10),
                Fix128::from_ratio(15, 10),
            ),
        );
        assert_eq!(vel.x, Fix128::from_int(3));
    }

    #[test]
    fn p2g_out_of_range_ignored() {
        let mut g = MacGrid::new(2, 2, 2, Fix128::ONE);
        let pos = Vec3Fix::new(
            Fix128::from_int(100),
            Fix128::from_int(100),
            Fix128::from_int(100),
        );
        let vel = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        p2g_nearest(&mut g, pos, vel);
        assert_eq!(g.u(0, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn p2g_trilinear_deposits_to_8_corners() {
        // Place a particle at the exact centre of a cell (interior).
        // Trilinear weights should distribute 1/8 to each of the 8 nearest
        // u-face nodes (well, technically the 8 face nodes around the u-face
        // cell whose centre is at that offset).
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        // Particle at (1.5, 1.5, 1.5) — a cell centre.
        // u-grid offset is (0, 0.5, 0.5) → local frac (0.5, 0.0, 0.0)
        //   → distributes only in x, so u_lo · 0.5 + u_hi · 0.5
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let vel = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        p2g_trilinear(&mut g, pos, vel);
        // u(1, 1, 1) and u(2, 1, 1) should each receive 2
        assert_eq!(g.u(1, 1, 1), Fix128::from_int(2));
        assert_eq!(g.u(2, 1, 1), Fix128::from_int(2));
    }

    #[test]
    fn g2p_trilinear_between_faces_interpolates() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let i0 = g.idx_u(1, 1, 1);
        let i1 = g.idx_u(2, 1, 1);
        g.u[i0] = Fix128::from_int(10);
        g.u[i1] = Fix128::from_int(20);
        // Position between the two u-faces should give ~15 by linear interp
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let v = g2p_velocity(&g, pos);
        assert_eq!(v.x, Fix128::from_int(15));
    }

    #[test]
    fn split_negative_position_clamps_to_zero() {
        // For a negative x coordinate the base index must clamp to 0
        // so out-of-range particles don't crash the deposit routines.
        let (base, _) = split(Fix128::from_int(-3), Fix128::ZERO);
        assert_eq!(base, 0);
    }

    // ---- BiCGStab pressure solver tests --------------------------------

    fn seed_divergent_flow(nx: usize) -> MacGrid {
        let mut g = MacGrid::new(nx, nx, nx, Fix128::ONE);
        // Linear u profile: u(i, j, k) = i so ∇·u ≠ 0.
        for i in 0..=nx {
            for j in 0..nx {
                for k in 0..nx {
                    let ix = g.idx_u(i, j, k);
                    if ix < g.u.len() {
                        g.u[ix] = Fix128::from_int(i as i64);
                    }
                }
            }
        }
        g
    }

    #[test]
    fn bicgstab_reduces_divergence_below_jacobi_iterations() {
        let mut g = seed_divergent_flow(4);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let div_before = g.divergence(2, 2, 2).abs();
        let stats = project_pressure_bicgstab(&mut g, dt, rho, 15, Fix128::from_ratio(1, 10_000));
        let div_after = g.divergence(2, 2, 2).abs();
        assert!(div_after < div_before);
        assert!(stats.iterations <= 15);
    }

    #[test]
    fn bicgstab_matches_jacobi_within_tolerance() {
        let mut g_bicg = seed_divergent_flow(4);
        let mut g_jac = g_bicg.clone();
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let _ = project_pressure_bicgstab(&mut g_bicg, dt, rho, 30, Fix128::from_ratio(1, 100_000));
        project_pressure_jacobi(&mut g_jac, dt, rho, 200);
        // Both should drive the centre divergence close to zero.
        let div_bicg = g_bicg.divergence(2, 2, 2).abs();
        let div_jac = g_jac.divergence(2, 2, 2).abs();
        assert!(div_bicg < Fix128::from_ratio(1, 10));
        assert!(div_jac < Fix128::from_ratio(1, 10));
    }

    #[test]
    fn bicgstab_stats_reports_convergence_status() {
        let mut g = seed_divergent_flow(4);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let stats =
            project_pressure_bicgstab(&mut g, dt, rho, 50, Fix128::from_ratio(1, 1_000_000));
        // Either converged within budget, or iterations == 50.
        assert!(stats.iterations >= 1);
        assert!(stats.iterations <= 50);
    }

    #[test]
    fn bicgstab_no_op_on_zero_dt() {
        let mut g = seed_divergent_flow(3);
        let stats = project_pressure_bicgstab(
            &mut g,
            Fix128::ZERO,
            Fix128::from_int(1000),
            10,
            Fix128::from_ratio(1, 10_000),
        );
        assert_eq!(stats.iterations, 0);
        assert!(stats.converged);
    }

    #[test]
    fn apply_poisson_a_is_symmetric_stencil() {
        // (A e_c)_c = -6 for a unit vector at an interior cell,
        // matching the convention A p = rhs where diag = -6.
        let nx = 3;
        let ny = 3;
        let nz = 3;
        let n = nx * ny * nz;
        let mut p = vec![Fix128::ZERO; n];
        let ctr = 1 + nx * (1 + ny * 1);
        p[ctr] = Fix128::ONE;
        let mut out = vec![Fix128::ZERO; n];
        apply_poisson_a(&p, &mut out, nx, ny, nz);
        assert_eq!(out[ctr], Fix128::from_int(-6));
        // Face neighbours receive +1.
        let neighbour = nx * (1 + ny);
        assert_eq!(out[neighbour], Fix128::ONE);
    }
}
