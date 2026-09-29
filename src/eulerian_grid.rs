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
//!
//! # Integration status
//!
//! Only the red-black Gauss-Seidel projection (`project_pressure`),
//! trilinear sampling helpers (`sample_u/v/w_range/trilinear`), and
//! `g2p_velocity` are currently wired into `cfd_solver.rs`. The
//! Jacobi + BiCGStab pressure variants and P2G scatter operators
//! are reserved crate-internal API awaiting downstream integration.
//!
//! # Face mask — walls inside the projection
//!
//! `u_solid` / `v_solid` / `w_solid` mark which faces are walls
//! ([`MacGrid::set_closed_box_walls`] does the six boundary layers of a
//! sealed box). The mask is part of the Poisson problem, not a post-pass:
//!
//! - a **solid** face carries no flux, so it leaves both the off-diagonal
//!   coupling and the diagonal count — the homogeneous Neumann condition
//!   `∂p/∂n = 0` — and its normal velocity is held at zero;
//! - a face that is **not** solid always counts toward the diagonal; on the
//!   domain boundary its neighbour is the exterior `p = 0` (open / free
//!   surface), which is the behaviour of a grid with no mask set.
//!
//! Two consequences that the old fixed `1/6` divisor got wrong:
//!
//! - the diagonal is the number of open faces, so a slab with walled `z`
//!   sides is a genuine 2-D Poisson problem at any `nz` instead of a screened
//!   one that barely moves the field;
//! - the velocity correction sweeps every face including the boundary layer,
//!   so the rim cells have the degree of freedom they need and their
//!   divergence is removed with the rest.

// Reserved algorithm variants (Jacobi / BiCGStab pressure, P2G scatter) are
// pub(crate) but currently unused outside their own unit tests — awaiting
// cfd_solver integration.
#![allow(dead_code)]

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
    /// Solid flag per X-face `[(nx+1) · ny · nz]`; see [`MacGrid::set_u_solid`].
    pub u_solid: Vec<bool>,
    /// Solid flag per Y-face `[nx · (ny+1) · nz]`; see [`MacGrid::set_v_solid`].
    pub v_solid: Vec<bool>,
    /// Solid flag per Z-face `[nx · ny · (nz+1)]`; see [`MacGrid::set_w_solid`].
    pub w_solid: Vec<bool>,
}

impl MacGrid {
    /// Empty grid initialised to zero everywhere, with every face fluid
    /// (no walls). Call [`MacGrid::set_closed_box_walls`] to turn the six
    /// domain-boundary face layers into no-through-flow walls.
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
            u_solid: vec![false; (nx + 1) * ny * nz],
            v_solid: vec![false; nx * (ny + 1) * nz],
            w_solid: vec![false; nx * ny * (nz + 1)],
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

    /// Is the X-face `(i, j, k)` a wall? Out of range returns `false`
    /// (there is no such face).
    #[must_use]
    pub fn is_u_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return false;
        }
        self.u_solid[self.idx_u(i, j, k)]
    }
    /// Is the Y-face `(i, j, k)` a wall?
    #[must_use]
    pub fn is_v_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return false;
        }
        self.v_solid[self.idx_v(i, j, k)]
    }
    /// Is the Z-face `(i, j, k)` a wall?
    #[must_use]
    pub fn is_w_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return false;
        }
        self.w_solid[self.idx_w(i, j, k)]
    }

    /// Mark the X-face `(i, j, k)` as a wall (`true`) or as fluid (`false`).
    ///
    /// A wall face carries no flux: the pressure projection holds its normal
    /// velocity at zero and drops it from the Poisson stencil, which is the
    /// homogeneous Neumann condition `∂p/∂n = 0` on that face. A face left
    /// `false` on the domain boundary keeps the open condition (exterior
    /// pressure `p = 0`). Out-of-range indices are ignored.
    pub fn set_u_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return;
        }
        let ix = self.idx_u(i, j, k);
        self.u_solid[ix] = solid;
    }
    /// Mark the Y-face `(i, j, k)` as a wall; see [`MacGrid::set_u_solid`].
    pub fn set_v_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return;
        }
        let ix = self.idx_v(i, j, k);
        self.v_solid[ix] = solid;
    }
    /// Mark the Z-face `(i, j, k)` as a wall; see [`MacGrid::set_u_solid`].
    pub fn set_w_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return;
        }
        let ix = self.idx_w(i, j, k);
        self.w_solid[ix] = solid;
    }

    /// Turn the six domain-boundary face layers into walls — the closed box
    /// used by the lid-driven cavity and by any sealed container.
    ///
    /// Marks `u` at `i = 0, nx`, `v` at `j = 0, ny` and `w` at `k = 0, nz`.
    /// Interior faces are left untouched, so this composes with obstacle
    /// masks set through [`MacGrid::set_u_solid`] and friends.
    pub fn set_closed_box_walls(&mut self) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                self.set_u_solid(0, j, k, true);
                self.set_u_solid(self.nx, j, k, true);
            }
        }
        for k in 0..self.nz {
            for i in 0..self.nx {
                self.set_v_solid(i, 0, k, true);
                self.set_v_solid(i, self.ny, k, true);
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                self.set_w_solid(i, j, 0, true);
                self.set_w_solid(i, j, self.nz, true);
            }
        }
    }

    /// Zero the normal velocity on every face marked solid.
    ///
    /// The projection calls this before it builds the divergence right-hand
    /// side, so whatever advection / body forces / diffusion left on a wall
    /// face never enters the Poisson problem.
    pub fn enforce_solid_faces(&mut self) {
        for (val, solid) in self.u.iter_mut().zip(self.u_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
        for (val, solid) in self.v.iter_mut().zip(self.v_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
        for (val, solid) in self.w.iter_mut().zip(self.w_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
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
// Poisson stencil under the face mask
// ============================================================================

/// Which faces of each cell take part in the pressure solve.
///
/// `open[c]` is ordered `[-x, +x, -y, +y, -z, +z]`. A solid face is closed:
/// it contributes neither an off-diagonal coupling nor a count to the
/// diagonal, which is the homogeneous Neumann condition. An open face always
/// counts toward the diagonal; when it sits on the domain boundary its
/// neighbour pressure is the exterior `p = 0`.
///
/// Owned rather than borrowed so the solvers can keep mutating the grid.
struct PoissonMask {
    nx: usize,
    ny: usize,
    nz: usize,
    open: Vec<[bool; 6]>,
}

impl PoissonMask {
    fn from_grid(grid: &MacGrid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut open = vec![[true; 6]; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    open[i + nx * (j + ny * k)] = [
                        !grid.is_u_solid(i, j, k),
                        !grid.is_u_solid(i + 1, j, k),
                        !grid.is_v_solid(i, j, k),
                        !grid.is_v_solid(i, j + 1, k),
                        !grid.is_w_solid(i, j, k),
                        !grid.is_w_solid(i, j, k + 1),
                    ];
                }
            }
        }
        Self { nx, ny, nz, open }
    }

    /// Number of open faces of cell `c`; `−degree` is the matrix diagonal.
    #[inline]
    fn degree(&self, c: usize) -> i64 {
        self.open[c].iter().filter(|&&o| o).count() as i64
    }

    /// Sum of the neighbour pressures reachable through the open faces of
    /// cell `(i, j, k)`. Faces open onto the exterior contribute `p = 0`.
    #[inline]
    fn neighbour_sum(&self, p: &[Fix128], i: usize, j: usize, k: usize) -> Fix128 {
        let c = i + self.nx * (j + self.ny * k);
        let o = self.open[c];
        let mut acc = Fix128::ZERO;
        if o[0] && i > 0 {
            acc = acc + p[c - 1];
        }
        if o[1] && i + 1 < self.nx {
            acc = acc + p[c + 1];
        }
        if o[2] && j > 0 {
            acc = acc + p[c - self.nx];
        }
        if o[3] && j + 1 < self.ny {
            acc = acc + p[c + self.nx];
        }
        if o[4] && k > 0 {
            acc = acc + p[c - self.nx * self.ny];
        }
        if o[5] && k + 1 < self.nz {
            acc = acc + p[c + self.nx * self.ny];
        }
        acc
    }

    /// `out = A p` with `(A p)_c = −degree(c)·p_c + Σ_open p_neighbour`.
    fn apply(&self, p: &[Fix128], out: &mut [Fix128]) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let c = i + self.nx * (j + self.ny * k);
                    out[c] =
                        p[c] * Fix128::from_int(-self.degree(c)) + self.neighbour_sum(p, i, j, k);
                }
            }
        }
    }
}

/// `1 / degree(c)` per cell, zero where a cell is sealed on all six faces.
///
/// A sealed cell has no flux across any face, so its divergence — and with it
/// its right-hand side — is identically zero and the relaxation drives its
/// pressure to zero. That is the correct answer: the cell is decoupled from
/// the rest of the field and its pressure has no gradient to produce.
fn inverse_degrees(mask: &PoissonMask, n: usize) -> Vec<Fix128> {
    let mut inv = vec![Fix128::ZERO; n];
    for (c, slot) in inv.iter_mut().enumerate() {
        let deg = mask.degree(c);
        if deg > 0 {
            *slot = Fix128::from_ratio(1, deg);
        }
    }
    inv
}

/// Subtract `coeff · ∇p` from the face velocities and hold the walls at zero.
///
/// The sweep covers **every** face, boundary layer included: the Poisson
/// operator counts a non-solid boundary face against the exterior `p = 0`, so
/// skipping it leaves the rim cells with no degree of freedom and their
/// divergence grows instead of vanishing.
fn subtract_pressure_gradient(grid: &mut MacGrid, coeff: Fix128) {
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                let ix = grid.idx_u(i, j, k);
                if grid.u_solid[ix] {
                    grid.u[ix] = Fix128::ZERO;
                    continue;
                }
                let hi = if i < grid.nx {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if i > 0 {
                    grid.pressure(i - 1, j, k)
                } else {
                    Fix128::ZERO
                };
                grid.u[ix] = grid.u[ix] - coeff * (hi - lo);
            }
        }
    }
    for k in 0..grid.nz {
        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                let ix = grid.idx_v(i, j, k);
                if grid.v_solid[ix] {
                    grid.v[ix] = Fix128::ZERO;
                    continue;
                }
                let hi = if j < grid.ny {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if j > 0 {
                    grid.pressure(i, j - 1, k)
                } else {
                    Fix128::ZERO
                };
                grid.v[ix] = grid.v[ix] - coeff * (hi - lo);
            }
        }
    }
    for k in 0..=grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let ix = grid.idx_w(i, j, k);
                if grid.w_solid[ix] {
                    grid.w[ix] = Fix128::ZERO;
                    continue;
                }
                let hi = if k < grid.nz {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if k > 0 {
                    grid.pressure(i, j, k - 1)
                } else {
                    Fix128::ZERO
                };
                grid.w[ix] = grid.w[ix] - coeff * (hi - lo);
            }
        }
    }
}

/// Build `rhs = ρ dx²/dt · ∇·u` after the walls have been enforced.
fn poisson_rhs(grid: &MacGrid, scale: Fix128) -> Vec<Fix128> {
    let mut rhs = vec![Fix128::ZERO; grid.nx * grid.ny * grid.nz];
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                rhs[i + grid.nx * (j + grid.ny * k)] = grid.divergence(i, j, k) * scale;
            }
        }
    }
    rhs
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
    grid.enforce_solid_faces();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);
    for _ in 0..iterations {
        // Two-colour sweep (colour ∈ {0, 1})
        for colour in 0..2u32 {
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        if ((i + j + k) as u32 % 2) != colour {
                            continue;
                        }
                        let idx = i + grid.nx * (j + grid.ny * k);
                        let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                        grid.pressure[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                    }
                }
            }
        }
    }

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
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
    grid.enforce_solid_faces();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);

    // Jacobi iterations for `A p = rhs`, `A` the masked 7-point Laplacian.
    let mut p_new = grid.pressure.clone();
    for _ in 0..iterations {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = i + grid.nx * (j + grid.ny * k);
                    let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                    p_new[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                }
            }
        }
        grid.pressure.clone_from(&p_new);
    }

    // Velocity correction: u ← u − (dt/ρ)·∇p
    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
}

/// Preconditioned **BiCGStab** pressure solver (van der Vorst 1992).
///
/// Solves the discrete Poisson system `A p = b` for the MAC-grid
/// pressure, where `A` is the 7-point Laplacian restricted by the face mask:
/// a solid face drops out entirely (homogeneous Neumann) and an open face on
/// the domain boundary couples to the exterior `p = 0`. The RHS
/// `b = ρ dx² / dt · ∇·u` matches the Jacobi and red-black Gauss–Seidel
/// variants, and the velocity correction stage at the end is identical.
///
/// # Preconditioner
///
/// Diagonal Jacobi preconditioner `M = diag(A)`, read from the **same**
/// stencil the operator uses: `−(number of open faces)`, which is `−6` in the
/// interior and drops by one per walled face. Before the face mask landed the
/// operator used a fixed `−6` while the preconditioner counted missing
/// neighbours, so the two disagreed at every boundary cell — harmless for the
/// answer, but it slowed the iteration and the doc described the
/// preconditioner as if it were the operator.
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
    grid.enforce_solid_faces();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;

    // Build RHS and cache the per-cell diagonal from the operator's own mask.
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let mut diag = vec![Fix128::ZERO; n];
    for (c, slot) in diag.iter_mut().enumerate() {
        *slot = Fix128::from_int(-mask.degree(c)); // A[c,c] = −degree(c)
    }

    // x = grid.pressure; solve A x = b with A = -Laplacian sign convention:
    // r0 = b − A x0
    let mut x = grid.pressure.clone();
    let mut r = vec![Fix128::ZERO; n];
    mask.apply(&x, &mut r);
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
        mask.apply(&y_vec, &mut v_vec);
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
        mask.apply(&z_vec, &mut t_vec);
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
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
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
    // Fix128 is two's-complement I64F64: `hi` is already floor(shifted) and
    // `lo` the non-negative fractional part in [0, 1), for negative values
    // too. Anything left of face 0 clamps to the first face.
    if shifted.hi < 0 {
        (0, Fix128::ZERO)
    } else {
        (
            shifted.hi as usize,
            Fix128 {
                hi: 0,
                lo: shifted.lo,
            },
        )
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
    fn poisson_operator_diagonal_matches_the_preconditioner() {
        // The preconditioner BiCGStab divides by must be the diagonal of the
        // operator it is preconditioning: `(A e_c)_c == −degree(c)`. Before
        // the face mask the operator used a fixed −6 while `diag` counted
        // missing neighbours, so the two disagreed on every boundary cell.
        let nx = 3;
        let ny = 3;
        let nz = 3;
        let mut grid = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
        grid.set_u_solid(1, 1, 1, true); // one interior wall face
        let mask = PoissonMask::from_grid(&grid);
        let n = nx * ny * nz;
        for c in 0..n {
            let mut e = vec![Fix128::ZERO; n];
            e[c] = Fix128::ONE;
            let mut out = vec![Fix128::ZERO; n];
            mask.apply(&e, &mut out);
            assert_eq!(
                out[c],
                Fix128::from_int(-mask.degree(c)),
                "diagonal of cell {c} disagrees with degree()"
            );
        }
    }

    #[test]
    fn poisson_operator_is_symmetric() {
        // `A` must be symmetric for BiCGStab's convergence theory to apply:
        // `(A e_a)_b == (A e_b)_a`. A one-sided face mask (marking a face
        // solid for one of its two cells only) would break it.
        let nx = 3;
        let ny = 3;
        let nz = 2;
        let mut grid = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
        grid.set_closed_box_walls();
        grid.set_v_solid(1, 1, 0, true);
        let mask = PoissonMask::from_grid(&grid);
        let n = nx * ny * nz;
        let mut columns = Vec::with_capacity(n);
        for c in 0..n {
            let mut e = vec![Fix128::ZERO; n];
            e[c] = Fix128::ONE;
            let mut out = vec![Fix128::ZERO; n];
            mask.apply(&e, &mut out);
            columns.push(out);
        }
        for (a, col_a) in columns.iter().enumerate() {
            for (b, col_b) in columns.iter().enumerate() {
                assert_eq!(col_a[b], col_b[a], "A[{b},{a}] != A[{a},{b}]");
            }
        }
    }

    #[test]
    fn closed_box_degree_drops_to_the_open_face_count() {
        // A sealed 1-cell-thick slab has its two z faces walled, so the
        // stencil is the 2-D one (degree 4), not the 3-D one with two zero
        // contributions (degree 6) the fixed 1/6 divisor assumed.
        let mut grid = MacGrid::new(2, 2, 1, Fix128::from_ratio(1, 8));
        grid.set_closed_box_walls();
        let mask = PoissonMask::from_grid(&grid);
        for c in 0..4 {
            assert_eq!(mask.degree(c), 2, "corner cell of a sealed 2x2x1 slab");
        }
        let mut open_slab = MacGrid::new(2, 2, 1, Fix128::from_ratio(1, 8));
        open_slab.set_w_solid(0, 0, 0, true);
        open_slab.set_w_solid(0, 0, 1, true);
        let open_mask = PoissonMask::from_grid(&open_slab);
        assert_eq!(
            open_mask.degree(0),
            4,
            "only the z faces are walled, the four lateral faces stay open"
        );
    }

    #[test]
    fn walls_are_held_at_zero_by_the_projection() {
        let mut grid = MacGrid::new(4, 4, 4, Fix128::from_ratio(1, 8));
        grid.u.fill(Fix128::ONE);
        grid.set_closed_box_walls();
        project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 50);
        for j in 0..4 {
            for k in 0..4 {
                assert_eq!(grid.u(0, j, k), Fix128::ZERO);
                assert_eq!(grid.u(4, j, k), Fix128::ZERO);
            }
        }
    }
}
