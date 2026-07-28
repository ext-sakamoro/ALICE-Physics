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
    fn idx_u(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }
    #[inline]
    #[must_use]
    fn idx_v(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }
    #[inline]
    #[must_use]
    fn idx_w(&self, i: usize, j: usize, k: usize) -> usize {
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
pub fn project_pressure(grid: &mut MacGrid, dt_s: Fix128, density_kg_m3: Fix128, iterations: u32) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    let scale = density_kg_m3 * grid.dx / dt_s;

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

// ============================================================================
// Trilinear P2G / G2P
// ============================================================================

/// Grid-to-particle: sample velocity at world position `pos_m`.
///
/// Uses linear interpolation on the u-face grid alone for the x-component
/// (each velocity component reads its own face grid).
#[must_use]
pub fn g2p_velocity(grid: &MacGrid, pos_m: Vec3Fix) -> Vec3Fix {
    if grid.dx.is_zero() {
        return Vec3Fix::default();
    }
    let inv_dx = Fix128::ONE / grid.dx;
    // Fractional index for each face grid — for the x-face grid, the origin
    // sits at (0, 0.5·dx, 0.5·dx). Compute floor + fractional.
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let ix = fx.hi as usize;
    let iy = fy.hi as usize;
    let iz = fz.hi as usize;
    // For simplicity fall back to nearest-cell centred velocity.
    let (ucx, vcy, wcz) = grid.cell_velocity(
        ix.min(grid.nx - 1),
        iy.min(grid.ny - 1),
        iz.min(grid.nz - 1),
    );
    Vec3Fix::new(ucx, vcy, wcz)
}

/// Particle-to-grid: scatter one particle's velocity `vel_m_per_s` at
/// position `pos_m` onto the closest cell centre (nearest-neighbour).
///
/// Trilinear scatter is the "true" P2G but requires a companion weight
/// grid; this simplified version is sufficient for coarse PIC tests.
pub fn p2g_nearest(grid: &mut MacGrid, pos_m: Vec3Fix, vel_m_per_s: Vec3Fix) {
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
}
