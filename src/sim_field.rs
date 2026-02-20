//! 3D Simulation Field Infrastructure
//!
//! Uniform grid-based scalar and vector fields for physics-driven
//! SDF modifiers. Supports trilinear interpolation, diffusion (heat
//! equation), decay, and point-splatting.
//!
//! # Design
//!
//! Fields store f32 values on a regular grid and convert between
//! world-space (f32) and grid indices. All modifiers (thermal,
//! pressure, erosion, etc.) build on these fields.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// ScalarField3D
// ============================================================================

/// 3D uniform scalar field with trilinear interpolation
#[derive(Clone, Debug)]
pub struct ScalarField3D {
    /// Grid data in row-major order: data[iz * ny * nx + iy * nx + ix]
    pub data: Vec<f32>,
    /// Grid resolution
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// World-space bounds (min corner)
    pub min: (f32, f32, f32),
    /// World-space bounds (max corner)
    pub max: (f32, f32, f32),
    /// Precomputed cell size
    cell_size: (f32, f32, f32),
    /// Precomputed inverse cell size
    inv_cell_size: (f32, f32, f32),
    /// Reusable scratch buffer for diffuse() (avoids per-call allocation)
    scratch: Vec<f32>,
}

impl ScalarField3D {
    /// Create a zero-filled field with given resolution and bounds
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        let cx = if nx > 1 {
            (max.0 - min.0) / (nx - 1) as f32
        } else {
            1.0
        };
        let cy = if ny > 1 {
            (max.1 - min.1) / (ny - 1) as f32
        } else {
            1.0
        };
        let cz = if nz > 1 {
            (max.2 - min.2) / (nz - 1) as f32
        } else {
            1.0
        };

        let n = nx * ny * nz;
        Self {
            data: vec![0.0; n],
            nx,
            ny,
            nz,
            min,
            max,
            cell_size: (cx, cy, cz),
            inv_cell_size: (1.0 / cx, 1.0 / cy, 1.0 / cz),
            scratch: vec![0.0; n],
        }
    }

    /// Create a field filled with a constant value
    pub fn new_filled(
        nx: usize,
        ny: usize,
        nz: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
        value: f32,
    ) -> Self {
        let mut field = Self::new(nx, ny, nz, min, max);
        for v in &mut field.data {
            *v = value;
        }
        field
    }

    /// Total number of cells
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Linear index from grid coordinates
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.ny * self.nx + iy * self.nx + ix
    }

    /// Get value at grid coordinates (clamped)
    #[inline]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        let ix = ix.min(self.nx - 1);
        let iy = iy.min(self.ny - 1);
        let iz = iz.min(self.nz - 1);
        self.data[self.index(ix, iy, iz)]
    }

    /// Set value at grid coordinates
    #[inline]
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, value: f32) {
        if ix < self.nx && iy < self.ny && iz < self.nz {
            let idx = self.index(ix, iy, iz);
            self.data[idx] = value;
        }
    }

    /// Add value at grid coordinates
    #[inline]
    pub fn add(&mut self, ix: usize, iy: usize, iz: usize, delta: f32) {
        if ix < self.nx && iy < self.ny && iz < self.nz {
            let idx = self.index(ix, iy, iz);
            self.data[idx] += delta;
        }
    }

    /// World-space to fractional grid coordinates
    #[inline]
    fn world_to_grid(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let gx = (x - self.min.0) * self.inv_cell_size.0;
        let gy = (y - self.min.1) * self.inv_cell_size.1;
        let gz = (z - self.min.2) * self.inv_cell_size.2;
        (gx, gy, gz)
    }

    /// Trilinear interpolation at world-space position
    #[inline]
    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let (gx, gy, gz) = self.world_to_grid(x, y, z);

        // Clamp to valid grid range
        let max_x = (self.nx - 1) as f32;
        let max_y = (self.ny - 1) as f32;
        let max_z = (self.nz - 1) as f32;
        let gx = gx.max(0.0).min(max_x);
        let gy = gy.max(0.0).min(max_y);
        let gz = gz.max(0.0).min(max_z);

        let ix0 = gx as usize;
        let iy0 = gy as usize;
        let iz0 = gz as usize;
        let ix1 = (ix0 + 1).min(self.nx - 1);
        let iy1 = (iy0 + 1).min(self.ny - 1);
        let iz1 = (iz0 + 1).min(self.nz - 1);

        let fx = gx - ix0 as f32;
        let fy = gy - iy0 as f32;
        let fz = gz - iz0 as f32;

        // Direct data access (indices already clamped, skip redundant bounds check in get())
        let nx = self.nx;
        let ny_nx = self.ny * nx;
        let d = &self.data;
        let i000 = iz0 * ny_nx + iy0 * nx + ix0;
        let i100 = iz0 * ny_nx + iy0 * nx + ix1;
        let i010 = iz0 * ny_nx + iy1 * nx + ix0;
        let i110 = iz0 * ny_nx + iy1 * nx + ix1;
        let i001 = iz1 * ny_nx + iy0 * nx + ix0;
        let i101 = iz1 * ny_nx + iy0 * nx + ix1;
        let i011 = iz1 * ny_nx + iy1 * nx + ix0;
        let i111 = iz1 * ny_nx + iy1 * nx + ix1;

        let c00 = d[i000] + (d[i100] - d[i000]) * fx;
        let c10 = d[i010] + (d[i110] - d[i010]) * fx;
        let c01 = d[i001] + (d[i101] - d[i001]) * fx;
        let c11 = d[i011] + (d[i111] - d[i011]) * fx;

        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;

        c0 + (c1 - c0) * fz
    }

    /// Central-difference gradient at world-space position
    #[inline]
    pub fn gradient(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let eps_x = self.cell_size.0 * 0.5;
        let eps_y = self.cell_size.1 * 0.5;
        let eps_z = self.cell_size.2 * 0.5;

        // 2*eps = cell_size, so 1/(2*eps) = inv_cell_size (precomputed, no division)
        let inv_2ex = self.inv_cell_size.0;
        let inv_2ey = self.inv_cell_size.1;
        let inv_2ez = self.inv_cell_size.2;

        let dx = (self.sample(x + eps_x, y, z) - self.sample(x - eps_x, y, z)) * inv_2ex;
        let dy = (self.sample(x, y + eps_y, z) - self.sample(x, y - eps_y, z)) * inv_2ey;
        let dz = (self.sample(x, y, z + eps_z) - self.sample(x, y, z - eps_z)) * inv_2ez;

        (dx, dy, dz)
    }

    /// Splat value at world-space position with radius (additive)
    pub fn splat(&mut self, x: f32, y: f32, z: f32, value: f32, radius: f32) {
        let (gx, gy, gz) = self.world_to_grid(x, y, z);
        let gr = radius * self.inv_cell_size.0; // Approximate radius in grid units

        let ix_min = ((gx - gr).floor() as i32).max(0) as usize;
        let ix_max = ((gx + gr).ceil() as i32).min(self.nx as i32 - 1) as usize;
        let iy_min = ((gy - gr).floor() as i32).max(0) as usize;
        let iy_max = ((gy + gr).ceil() as i32).min(self.ny as i32 - 1) as usize;
        let iz_min = ((gz - gr).floor() as i32).max(0) as usize;
        let iz_max = ((gz + gr).ceil() as i32).min(self.nz as i32 - 1) as usize;

        let inv_r = if radius > 1e-10 { 1.0 / radius } else { 1.0 };
        let radius_sq = radius * radius;

        for iz in iz_min..=iz_max {
            for iy in iy_min..=iy_max {
                for ix in ix_min..=ix_max {
                    let wx = self.min.0 + ix as f32 * self.cell_size.0;
                    let wy = self.min.1 + iy as f32 * self.cell_size.1;
                    let wz = self.min.2 + iz as f32 * self.cell_size.2;

                    let dx = wx - x;
                    let dy = wy - y;
                    let dz = wz - z;
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    // Early reject with squared distance (no sqrt)
                    if dist_sq < radius_sq {
                        let dist = dist_sq.sqrt();
                        // Smooth falloff (cubic)
                        let t = 1.0 - dist * inv_r;
                        let weight = t * t * (3.0 - 2.0 * t); // smoothstep
                        let idx = self.index(ix, iy, iz);
                        self.data[idx] += value * weight;
                    }
                }
            }
        }
    }

    /// Diffuse field values (heat equation: dT/dt = k * laplacian(T))
    ///
    /// Uses explicit Euler with 7-point stencil.
    /// Scratch buffer is reused across calls (zero allocation after first call).
    pub fn diffuse(&mut self, dt: f32, rate: f32) {
        let inv_dx2 = self.inv_cell_size.0 * self.inv_cell_size.0;
        let inv_dy2 = self.inv_cell_size.1 * self.inv_cell_size.1;
        let inv_dz2 = self.inv_cell_size.2 * self.inv_cell_size.2;

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let ny_nx = ny * nx;
        let rate_dt = rate * dt;

        for iz in 0..nz {
            let iz_off = iz * ny_nx;
            for iy in 0..ny {
                let iy_off = iz_off + iy * nx;
                for ix in 0..nx {
                    let idx = iy_off + ix;
                    let c = self.data[idx];

                    let xm = if ix > 0 { self.data[idx - 1] } else { c };
                    let xp = if ix + 1 < nx { self.data[idx + 1] } else { c };
                    let ym = if iy > 0 { self.data[idx - nx] } else { c };
                    let yp = if iy + 1 < ny { self.data[idx + nx] } else { c };
                    let zm = if iz > 0 { self.data[idx - ny_nx] } else { c };
                    let zp = if iz + 1 < nz {
                        self.data[idx + ny_nx]
                    } else {
                        c
                    };

                    let laplacian = (xm + xp - 2.0 * c) * inv_dx2
                        + (ym + yp - 2.0 * c) * inv_dy2
                        + (zm + zp - 2.0 * c) * inv_dz2;

                    self.scratch[idx] = c + rate_dt * laplacian;
                }
            }
        }

        std::mem::swap(&mut self.data, &mut self.scratch);
    }

    /// Exponential decay toward zero: value *= exp(-rate * dt)
    pub fn decay(&mut self, rate: f32, dt: f32) {
        let factor = (-rate * dt).exp();
        for v in &mut self.data {
            *v *= factor;
        }
    }

    /// Decay toward a target value
    pub fn decay_toward(&mut self, target: f32, rate: f32, dt: f32) {
        let factor = (-rate * dt).exp();
        for v in &mut self.data {
            *v = target + (*v - target) * factor;
        }
    }

    /// Clamp all values to range
    pub fn clamp(&mut self, lo: f32, hi: f32) {
        for v in &mut self.data {
            if *v < lo {
                *v = lo;
            }
            if *v > hi {
                *v = hi;
            }
        }
    }

    /// Set all values to zero
    pub fn clear(&mut self) {
        for v in &mut self.data {
            *v = 0.0;
        }
    }

    /// Maximum value in the field
    pub fn max_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Check if a world-space point is inside the field bounds
    #[inline]
    pub fn contains(&self, x: f32, y: f32, z: f32) -> bool {
        x >= self.min.0
            && x <= self.max.0
            && y >= self.min.1
            && y <= self.max.1
            && z >= self.min.2
            && z <= self.max.2
    }
}

// ============================================================================
// VectorField3D
// ============================================================================

/// 3D uniform vector field (three scalar fields)
#[derive(Clone, Debug)]
pub struct VectorField3D {
    /// X component
    pub x: ScalarField3D,
    /// Y component
    pub y: ScalarField3D,
    /// Z component
    pub z: ScalarField3D,
}

impl VectorField3D {
    /// Create a zero vector field
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        Self {
            x: ScalarField3D::new(nx, ny, nz, min, max),
            y: ScalarField3D::new(nx, ny, nz, min, max),
            z: ScalarField3D::new(nx, ny, nz, min, max),
        }
    }

    /// Sample vector at world-space position
    pub fn sample(&self, wx: f32, wy: f32, wz: f32) -> (f32, f32, f32) {
        (
            self.x.sample(wx, wy, wz),
            self.y.sample(wx, wy, wz),
            self.z.sample(wx, wy, wz),
        )
    }

    /// Splat a vector at world-space position
    pub fn splat(&mut self, wx: f32, wy: f32, wz: f32, vx: f32, vy: f32, vz: f32, radius: f32) {
        self.x.splat(wx, wy, wz, vx, radius);
        self.y.splat(wx, wy, wz, vy, radius);
        self.z.splat(wx, wy, wz, vz, radius);
    }

    /// Decay all components
    pub fn decay(&mut self, rate: f32, dt: f32) {
        self.x.decay(rate, dt);
        self.y.decay(rate, dt);
        self.z.decay(rate, dt);
    }

    /// Clear all components
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field_get_set() {
        let mut field = ScalarField3D::new(4, 4, 4, (0.0, 0.0, 0.0), (3.0, 3.0, 3.0));
        field.set(1, 2, 3, 42.0);
        assert_eq!(field.get(1, 2, 3), 42.0);
        assert_eq!(field.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_trilinear_interpolation() {
        let mut field = ScalarField3D::new(2, 2, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        // Set corner values
        field.set(0, 0, 0, 0.0);
        field.set(1, 0, 0, 10.0);
        field.set(0, 1, 0, 0.0);
        field.set(1, 1, 0, 10.0);
        field.set(0, 0, 1, 0.0);
        field.set(1, 0, 1, 10.0);
        field.set(0, 1, 1, 0.0);
        field.set(1, 1, 1, 10.0);

        // Midpoint along X should be 5.0
        let mid = field.sample(0.5, 0.0, 0.0);
        assert!((mid - 5.0).abs() < 0.01, "Expected ~5.0, got {}", mid);

        // Corner should be exact
        let corner = field.sample(1.0, 0.0, 0.0);
        assert!(
            (corner - 10.0).abs() < 0.01,
            "Expected ~10.0, got {}",
            corner
        );
    }

    #[test]
    fn test_splat() {
        let mut field = ScalarField3D::new(8, 8, 8, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        field.splat(0.0, 0.0, 0.0, 1.0, 0.5);

        // Center should have high value
        let center = field.sample(0.0, 0.0, 0.0);
        assert!(
            center > 0.5,
            "Center should be high after splat, got {}",
            center
        );

        // Far corner should be zero
        let far = field.sample(1.0, 1.0, 1.0);
        assert!(far < 0.01, "Far corner should be near zero, got {}", far);
    }

    #[test]
    fn test_diffusion() {
        let mut field = ScalarField3D::new(8, 8, 8, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        // Hot spot at center
        field.set(4, 4, 4, 100.0);

        let initial_max = field.max_value();

        // Diffuse
        for _ in 0..10 {
            field.diffuse(0.01, 1.0);
        }

        // Max should decrease (heat spreads)
        let after_max = field.max_value();
        assert!(after_max < initial_max, "Diffusion should spread heat");

        // Neighbors should have gained heat
        let neighbor = field.get(3, 4, 4);
        assert!(neighbor > 0.0, "Neighbor should have received heat");
    }

    #[test]
    fn test_gradient() {
        let mut field = ScalarField3D::new(8, 8, 8, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
        // Linear ramp along X: value = x
        for ix in 0..8 {
            for iy in 0..8 {
                for iz in 0..8 {
                    let x = -1.0 + ix as f32 * (2.0 / 7.0);
                    field.set(ix, iy, iz, x);
                }
            }
        }

        let (gx, gy, gz) = field.gradient(0.0, 0.0, 0.0);
        assert!(
            (gx - 1.0).abs() < 0.2,
            "X gradient should be ~1.0, got {}",
            gx
        );
        assert!(gy.abs() < 0.1, "Y gradient should be ~0, got {}", gy);
        assert!(gz.abs() < 0.1, "Z gradient should be ~0, got {}", gz);
    }

    #[test]
    fn test_decay() {
        let mut field = ScalarField3D::new(4, 4, 4, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        field.set(1, 1, 1, 100.0);

        field.decay(1.0, 1.0); // e^(-1) ≈ 0.368
        let v = field.get(1, 1, 1);
        assert!(
            (v - 36.8).abs() < 1.0,
            "After decay, expected ~36.8, got {}",
            v
        );
    }
}
