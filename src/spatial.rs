//! Spatial Hash Grid
//!
//! A hash-based spatial acceleration structure for neighbor queries.
//! Used by [`fluid`](crate::fluid) for SPH neighbor search and available
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
        let grid_half = (grid_dim as i64) / 2;
        let total_cells = grid_dim * grid_dim * grid_dim;
        Self {
            inv_cell_size: inv_cell,
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
        for c in &mut self.counts {
            *c = 0;
        }
        for o in &mut self.cell_offsets {
            *o = 0;
        }
        self.indices.clear();
    }

    /// Compute the cell index for a given position.
    #[inline(always)]
    #[must_use]
    pub fn hash(&self, pos: Vec3Fix) -> usize {
        let gd = self.grid_dim as i64;
        let half = self.grid_half;
        let ix = ((pos.x * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
        let iy = ((pos.y * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
        let iz = ((pos.z * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
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
    /// `_radius_sq` is reserved for future distance filtering.
    pub fn query_neighbors_into(
        &self,
        pos: Vec3Fix,
        _radius_sq: Fix128,
        neighbors: &mut Vec<usize>,
    ) {
        let gd = self.grid_dim as i64;
        let half = self.grid_half;
        let cx = ((pos.x * self.inv_cell_size).hi + half).clamp(0, gd - 1) as i32;
        let cy = ((pos.y * self.inv_cell_size).hi + half).clamp(0, gd - 1) as i32;
        let cz = ((pos.z * self.inv_cell_size).hi + half).clamp(0, gd - 1) as i32;

        neighbors.clear();

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
