//! Spatial Hash Grid
//!
//! A hash-based spatial acceleration structure for neighbor queries.
//! Used by [`crate::fluid`] for SPH neighbor search and available
//! for cloth self-collision or any particle-based simulation.
//!
//! # How It Works
//!
//! The grid divides space into uniform cells. Each particle is inserted
//! into the cell corresponding to its position. Neighbor queries examine
//! the 3x3x3 neighborhood of cells around the query point.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Spatial hash grid for O(n) neighbor queries in particle simulations.
///
/// Divides space into a uniform 3D grid of `grid_dim^3` cells. Uses a
/// CSR (Compressed Sparse Row) flat-buffer layout: `indices` holds all
/// particle indices packed contiguously, and `cell_offsets[h]..cell_offsets[h+1]`
/// gives the slice for cell `h`. This eliminates per-cell heap allocations
/// and improves cache locality for neighbor queries.
///
/// # Build flow
///
/// 1. Call `clear()` to reset counts.
/// 2. Call `insert()` for each particle.
/// 3. Call `build()` to finalize the CSR layout.
/// 4. Call `query_neighbors_into()` for lookups.
pub struct SpatialGrid {
    inv_cell_size: Fix128,
    /// Raw cell size (units of 2^-64) when `1 / cell_size` is not
    /// representable (`|raw| <= 2`); the cell coordinate is then the exact
    /// floor quotient `x_raw / cell_raw`.
    tiny_cell_raw: Option<i128>,
    /// Flat particle index buffer (CSR values).
    indices: Vec<usize>,
    /// Cell start offsets in `indices`; length = total_cells + 1.
    cell_offsets: Vec<usize>,
    /// Per-cell counts used during the two-pass build.
    counts: Vec<usize>,
    grid_dim: usize,
    grid_half: i64,
    total_cells: usize,
}

impl SpatialGrid {
    /// Create a new spatial grid.
    ///
    /// - `cell_size`: Side length of each cubic cell (should match the
    ///   interaction radius for best performance).
    /// - `grid_dim`: Number of cells along each axis. Total cells = `grid_dim^3`.
    #[must_use]
    pub fn new(cell_size: Fix128, grid_dim: usize) -> Self {
        let inv_cell = if cell_size.is_zero() {
            Fix128::ONE
        } else {
            Fix128::ONE / cell_size
        };
        let cell_raw = fix_raw(cell_size);
        let tiny_cell_raw = (cell_raw != 0 && cell_raw.unsigned_abs() <= 2).then_some(cell_raw);
        let grid_half = (grid_dim as i64) / 2;
        let total_cells = grid_dim * grid_dim * grid_dim;
        Self {
            inv_cell_size: inv_cell,
            tiny_cell_raw,
            indices: Vec::new(),
            cell_offsets: vec![0; total_cells + 1],
            counts: vec![0; total_cells],
            grid_dim,
            grid_half,
            total_cells,
        }
    }

    /// Reset the grid for a fresh build pass.
    pub fn clear(&mut self) {
        self.counts.fill(0);
        self.cell_offsets.fill(0);
        self.indices.clear();
    }

    /// Cell coordinate along one axis:
    /// `clamp(floor(x / cell_size) + grid_dim / 2, 0, grid_dim - 1)`.
    ///
    /// Total over the whole `Fix128` range, identical in debug and release:
    ///
    /// - `|x / cell_size| < 2^62` (every position that fits in a grid one can
    ///   allocate): `floor` is taken from `x * (1 / cell_size)` exactly as
    ///   before, so these cells are bit-unchanged.
    /// - `|x / cell_size| >= 2^62`: the scaled coordinate (or the
    ///   `+ grid_dim / 2` offset) does not fit in `i64`; the cell is the
    ///   border cell on the side of the sign of `x / cell_size` (`0` or
    ///   `grid_dim - 1`), which is what the clamp gives for the exact value
    ///   because `grid_dim / 2 < 2^62` for any allocatable grid.
    /// - `|cell_size| <= 2^-63` (`1 / cell_size` not representable): the cell
    ///   is the exact floor quotient of the raw values.
    ///
    /// Requires `grid_dim > 0`.
    #[inline(always)]
    fn axis_cell(&self, x: Fix128) -> usize {
        let gd = self.grid_dim as i64;
        let half = self.grid_half;
        let x_raw = fix_raw(x);
        if let Some(c_raw) = self.tiny_cell_raw {
            // floor(x_raw / c_raw); `checked_div` fails only for i128::MIN / -1
            // whose quotient is +2^127 (beyond every cell)
            let q = match x_raw.checked_div(c_raw) {
                Some(q) if x_raw % c_raw != 0 && ((x_raw < 0) != (c_raw < 0)) => q - 1,
                Some(q) => q,
                None => return self.grid_dim - 1,
            };
            return q
                .saturating_add(i128::from(half))
                .clamp(0, i128::from(gd - 1)) as usize;
        }
        let inv_raw = fix_raw(self.inv_cell_size);
        // |x * inv| < 2^62  <=>  |x_raw| * |inv_raw| < 2^(62 + 128)
        if mul_u128_high(x_raw.unsigned_abs(), inv_raw.unsigned_abs()) < (1u128 << 62) {
            return ((x * self.inv_cell_size).hi.saturating_add(half)).clamp(0, gd - 1) as usize;
        }
        if (x_raw < 0) != (inv_raw < 0) {
            0
        } else {
            self.grid_dim - 1
        }
    }

    /// Compute the cell index for a given position.
    ///
    /// Each axis is `clamp(floor(x / cell_size) + grid_dim / 2, 0, grid_dim - 1)`
    /// for every `Fix128` input: positions far outside the grid (up to the
    /// ends of the `Fix128` range, or with a cell size down to `2^-64`) land
    /// in the border cell on their side, never panic and never wrap, in both
    /// debug and release builds.
    ///
    /// A grid built with `grid_dim = 0` has no cells; every position hashes
    /// to `0`, and callers already guard `h < total_cells` (which is also
    /// `0`) before using the result.
    #[inline(always)]
    #[must_use]
    pub fn hash(&self, pos: Vec3Fix) -> usize {
        if self.grid_dim == 0 {
            return 0;
        }
        let ix = self.axis_cell(pos.x);
        let iy = self.axis_cell(pos.y);
        let iz = self.axis_cell(pos.z);
        ix + iy * self.grid_dim + iz * self.grid_dim * self.grid_dim
    }

    /// Record a particle insertion (pass 1: count only).
    ///
    /// Call `build()` after all insertions to finalize the CSR layout.
    pub fn insert(&mut self, idx: usize, pos: Vec3Fix) {
        let h = self.hash(pos);
        if h < self.total_cells {
            self.counts[h] += 1;
            // Store (h, idx) temporarily in `indices` as interleaved pairs.
            self.indices.push(h);
            self.indices.push(idx);
        }
    }

    /// Finalize the CSR layout after all `insert()` calls.
    ///
    /// Converts the temporary (cell, particle) pairs into a proper
    /// prefix-sum offset table and sorted flat index buffer.
    pub fn build(&mut self) {
        // Prefix-sum counts → cell_offsets
        let mut running = 0usize;
        for h in 0..self.total_cells {
            self.cell_offsets[h] = running;
            running += self.counts[h];
            self.counts[h] = 0; // reuse as write cursor below
        }
        self.cell_offsets[self.total_cells] = running;

        // Scatter particle indices into a final sorted buffer
        let n_pairs = self.indices.len() / 2;
        let mut final_buf = vec![0usize; running];
        for k in 0..n_pairs {
            let h = self.indices[k * 2];
            let idx = self.indices[k * 2 + 1];
            let slot = self.cell_offsets[h] + self.counts[h];
            final_buf[slot] = idx;
            self.counts[h] += 1;
        }
        self.indices = final_buf;
    }

    /// Collect all particle indices in the 3x3x3 neighborhood of `pos`.
    ///
    /// Results are appended to `neighbors` (which is cleared first).
    // LIMITATION(COV-PART-052): `_radius_sq` is reserved for future distance filtering.
    /// `_radius_sq` is reserved for future distance filtering.
    pub fn query_neighbors_into(
        &self,
        pos: Vec3Fix,
        _radius_sq: Fix128,
        neighbors: &mut Vec<usize>,
    ) {
        neighbors.clear();
        if self.grid_dim == 0 {
            return;
        }
        let cx = self.axis_cell(pos.x) as i32;
        let cy = self.axis_cell(pos.y) as i32;
        let cz = self.axis_cell(pos.z) as i32;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    let nz = cz + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;
                    if nx >= self.grid_dim || ny >= self.grid_dim || nz >= self.grid_dim {
                        continue;
                    }

                    let h = nx + ny * self.grid_dim + nz * self.grid_dim * self.grid_dim;
                    let start = self.cell_offsets[h];
                    let end = self.cell_offsets[h + 1];
                    for &idx in &self.indices[start..end] {
                        neighbors.push(idx);
                    }
                }
            }
        }
    }
}

/// The signed 128-bit raw value of a `Fix128` (units of 2^-64).
#[inline(always)]
fn fix_raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

/// High 128 bits of the 256-bit product `a * b`.
#[inline(always)]
fn mul_u128_high(a: u128, b: u128) -> u128 {
    const M: u128 = u64::MAX as u128;
    let (a1, a0) = (a >> 64, a & M);
    let (b1, b0) = (b >> 64, b & M);
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let mid = (ll >> 64) + (lh & M) + (hl & M);
    hh + (lh >> 64) + (hl >> 64) + (mid >> 64)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_grid_basic() {
        let mut grid = SpatialGrid::new(Fix128::from_ratio(1, 5), 32);
        grid.insert(0, Vec3Fix::ZERO);
        grid.insert(1, Vec3Fix::from_f32(0.1, 0.0, 0.0));
        grid.insert(2, Vec3Fix::from_f32(10.0, 0.0, 0.0));
        grid.build();

        let h_sq = Fix128::from_ratio(1, 5) * Fix128::from_ratio(1, 5);
        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, h_sq, &mut neighbors);
        assert!(neighbors.contains(&0), "Should find self");
        assert!(neighbors.contains(&1), "Should find nearby particle");
    }

    #[test]
    fn test_spatial_grid_clear() {
        let mut grid = SpatialGrid::new(Fix128::ONE, 8);
        grid.insert(0, Vec3Fix::ZERO);
        grid.build();
        grid.clear();

        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, Fix128::ONE, &mut neighbors);
        assert!(neighbors.is_empty(), "Grid should be empty after clear");
    }

    #[test]
    fn test_spatial_grid_hash_determinism() {
        let grid = SpatialGrid::new(Fix128::from_ratio(1, 5), 32);
        let pos = Vec3Fix::from_f32(1.5, -2.3, 0.7);
        let h1 = grid.hash(pos);
        let h2 = grid.hash(pos);
        assert_eq!(h1, h2, "Hash must be deterministic");
    }
}
