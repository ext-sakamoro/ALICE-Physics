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
/// Divides space into a uniform 3D grid of `grid_dim^3` cells. Particles
/// are inserted by hashing their position to a cell index. Neighbor queries
/// return all particles in the 3x3x3 block of cells surrounding the query
/// point, giving approximate range search in O(1) per query (amortized).
pub struct SpatialGrid {
    inv_cell_size: Fix128,
    cells: Vec<Vec<usize>>,
    grid_dim: usize,
    grid_half: i64,
}

impl SpatialGrid {
    /// Create a new spatial grid.
    ///
    /// - `cell_size`: Side length of each cubic cell (should match the
    ///   interaction radius for best performance).
    /// - `grid_dim`: Number of cells along each axis. Total cells = `grid_dim^3`.
    pub fn new(cell_size: Fix128, grid_dim: usize) -> Self {
        let inv_cell = if cell_size.is_zero() {
            Fix128::ONE
        } else {
            Fix128::ONE / cell_size
        };
        let grid_half = (grid_dim as i64) / 2;
        Self {
            inv_cell_size: inv_cell,
            cells: vec![Vec::new(); grid_dim * grid_dim * grid_dim],
            grid_dim,
            grid_half,
        }
    }

    /// Clear all cells (retains allocated memory for reuse).
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    /// Compute the cell index for a given position.
    #[inline(always)]
    pub fn hash(&self, pos: Vec3Fix) -> usize {
        let gd = self.grid_dim as i64;
        let half = self.grid_half;
        let ix = ((pos.x * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
        let iy = ((pos.y * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
        let iz = ((pos.z * self.inv_cell_size).hi + half).clamp(0, gd - 1) as usize;
        ix + iy * self.grid_dim + iz * self.grid_dim * self.grid_dim
    }

    /// Insert a particle index at the given position.
    pub fn insert(&mut self, idx: usize, pos: Vec3Fix) {
        let h = self.hash(pos);
        if h < self.cells.len() {
            self.cells[h].push(idx);
        }
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
                    for &idx in &self.cells[h] {
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
