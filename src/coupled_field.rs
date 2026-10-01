//! Deterministic coupling channel between physics subsystems.
//!
//! [`CoupledField`] is a uniform 3D scalar field stored in [`Fix128`], i.e. in
//! the same number system as the solver core. It exists so that two physics
//! that each own a field of the *same* quantity have somewhere to meet: they
//! publish into one channel, the channel produces one agreed field, and they
//! adopt it back.
//!
//! # Why a second field type
//!
//! `sim_field::ScalarField3D` is `f32` and lives in the SDF geometry
//! layer — it modulates a signed distance, which is the only thing the
//! `sim_modifier::PhysicsModifier` chain can carry between
//! implementors. That layer is kept as it is; this module runs beside it in
//! `Fix128` so a coupled quantity (temperature, pressure, damage) can be
//! exchanged with the integer core's arithmetic rather than through a scalar
//! distance.
//!
//! # Determinism
//!
//! Every operation here is integer arithmetic on [`Fix128`], so it is
//! bit-identical on every platform. Addition is exact (no rounding), which is
//! what makes [`reconcile_mean`] independent of the order of its participants.
//! `CoupledField::copy_from_f32` / `CoupledField::write_to_f32` convert at
//! the boundary with an `f32` owner; the conversion itself is a pure function,
//! but it obviously cannot add precision the `f32` side never had.
//!
//! # Boundary condition
//!
//! [`CoupledField::diffuse`] uses a **reflective (Neumann) ghost node**: the
//! value outside the grid is the mirror of the first interior node about the
//! boundary node. That makes `cos(k x)` with `k = m·π / L` an exact eigenmode
//! of the discrete operator, which is what
//! `tests/analytic_coupled_field.rs` compares against. `ScalarField3D::diffuse`
//! instead copies the centre value into the ghost, which halves the boundary
//! Laplacian and admits no closed-form mode; the two are therefore not
//! bit-comparable at the boundary by design.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
#[cfg(feature = "std")]
use crate::sim_field::ScalarField3D;
use core::fmt;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Errors
// ============================================================================

/// Why a coupling operation could not be carried out.
///
/// Every failure mode is a mismatch that would otherwise be papered over by
/// silently truncating, resampling or skipping cells; the channel refuses
/// instead so that a wrong pairing is visible at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoupledFieldError {
    /// A grid dimension was zero; a field with no cells has no samples.
    EmptyGrid {
        /// The rejected resolution `(nx, ny, nz)`.
        resolution: (usize, usize, usize),
    },
    /// Two grids that had to agree have different resolutions.
    ResolutionMismatch {
        /// Resolution of the channel.
        channel: (usize, usize, usize),
        /// Resolution of the other participant.
        other: (usize, usize, usize),
    },
    /// Two grids that had to agree cover different regions of space.
    BoundsMismatch,
    /// [`reconcile_mean`] was called with an empty participant list; there is
    /// no field to agree on.
    NoParticipants,
}

impl fmt::Display for CoupledFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGrid { resolution } => {
                let (nx, ny, nz) = resolution;
                write!(f, "empty coupled field grid ({nx}x{ny}x{nz})")
            }
            Self::ResolutionMismatch { channel, other } => {
                let (cx, cy, cz) = channel;
                let (ox, oy, oz) = other;
                write!(
                    f,
                    "coupled field resolution mismatch: channel {cx}x{cy}x{cz}, other {ox}x{oy}x{oz}"
                )
            }
            Self::BoundsMismatch => {
                write!(f, "coupled field bounds mismatch")
            }
            Self::NoParticipants => {
                write!(f, "no participants to reconcile")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoupledFieldError {}

// ============================================================================
// CoupledField
// ============================================================================

/// Uniform 3D scalar field in [`Fix128`], sampled by trilinear interpolation.
///
/// Storage is row-major, `data[iz * ny * nx + iy * nx + ix]`, with node `i`
/// along an axis sitting at `min + i * cell_size` — i.e. the outer nodes lie
/// exactly on `min` and `max`, matching `sim_field::ScalarField3D`.
#[derive(Clone, Debug)]
pub struct CoupledField {
    data: Vec<Fix128>,
    nx: usize,
    ny: usize,
    nz: usize,
    min: (Fix128, Fix128, Fix128),
    max: (Fix128, Fix128, Fix128),
    cell: (Fix128, Fix128, Fix128),
    scratch: Vec<Fix128>,
}

impl CoupledField {
    /// Create a zero-filled field.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::EmptyGrid`] if any resolution is zero.
    pub fn try_new(
        nx: usize,
        ny: usize,
        nz: usize,
        min: (Fix128, Fix128, Fix128),
        max: (Fix128, Fix128, Fix128),
    ) -> Result<Self, CoupledFieldError> {
        Self::try_new_filled(nx, ny, nz, min, max, Fix128::ZERO)
    }

    /// Create a field with every cell set to `value`.
    ///
    /// A degenerate axis (`n == 1`) is given a cell size of one so that the
    /// stored spacing is never zero; that axis contributes nothing to
    /// [`Self::diffuse`] and interpolates as a constant.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::EmptyGrid`] if any resolution is zero.
    pub fn try_new_filled(
        nx: usize,
        ny: usize,
        nz: usize,
        min: (Fix128, Fix128, Fix128),
        max: (Fix128, Fix128, Fix128),
        value: Fix128,
    ) -> Result<Self, CoupledFieldError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(CoupledFieldError::EmptyGrid {
                resolution: (nx, ny, nz),
            });
        }
        let span = |lo: Fix128, hi: Fix128, n: usize| -> Fix128 {
            if n > 1 {
                (hi - lo) / Fix128::from_int((n - 1) as i64)
            } else {
                Fix128::ONE
            }
        };
        let cell = (
            span(min.0, max.0, nx),
            span(min.1, max.1, ny),
            span(min.2, max.2, nz),
        );
        let n = nx * ny * nz;
        Ok(Self {
            data: vec![value; n],
            nx,
            ny,
            nz,
            min,
            max,
            cell,
            scratch: vec![Fix128::ZERO; n],
        })
    }

    /// Create a zero-filled channel covering the same grid as an `f32` field.
    ///
    /// This is the entry point for coupling a subsystem that owns a
    /// `ScalarField3D`: the channel built here is accepted by
    /// [`Self::copy_from_f32`] and [`Self::write_to_f32`] for that field.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::EmptyGrid`] if the source has a zero dimension.
    #[cfg(feature = "std")]
    pub fn try_matching(src: &ScalarField3D) -> Result<Self, CoupledFieldError> {
        Self::try_new(
            src.nx,
            src.ny,
            src.nz,
            (
                Fix128::from_f32(src.min.0),
                Fix128::from_f32(src.min.1),
                Fix128::from_f32(src.min.2),
            ),
            (
                Fix128::from_f32(src.max.0),
                Fix128::from_f32(src.max.1),
                Fix128::from_f32(src.max.2),
            ),
        )
    }

    /// Resolution along X.
    #[inline]
    #[must_use]
    pub const fn nx(&self) -> usize {
        self.nx
    }

    /// Resolution along Y.
    #[inline]
    #[must_use]
    pub const fn ny(&self) -> usize {
        self.ny
    }

    /// Resolution along Z.
    #[inline]
    #[must_use]
    pub const fn nz(&self) -> usize {
        self.nz
    }

    /// Total number of cells.
    #[inline]
    #[must_use]
    pub const fn cell_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// World-space minimum corner.
    #[inline]
    #[must_use]
    pub const fn min(&self) -> (Fix128, Fix128, Fix128) {
        self.min
    }

    /// World-space maximum corner.
    #[inline]
    #[must_use]
    pub const fn max(&self) -> (Fix128, Fix128, Fix128) {
        self.max
    }

    /// Node spacing along each axis.
    #[inline]
    #[must_use]
    pub const fn cell_size(&self) -> (Fix128, Fix128, Fix128) {
        self.cell
    }

    /// Cell values in storage order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Fix128] {
        &self.data
    }

    /// Cell values in storage order, mutable.
    ///
    /// The length cannot change through a slice, so the `nx * ny * nz`
    /// invariant survives any write made through it.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Fix128] {
        &mut self.data
    }

    /// Linear index of a grid node.
    #[inline]
    #[must_use]
    pub const fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.ny * self.nx + iy * self.nx + ix
    }

    /// Value at a grid node, indices clamped into range.
    #[inline]
    #[must_use]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> Fix128 {
        let ix = ix.min(self.nx - 1);
        let iy = iy.min(self.ny - 1);
        let iz = iz.min(self.nz - 1);
        self.data[self.index(ix, iy, iz)]
    }

    /// Write a grid node. Out-of-range indices are ignored.
    #[inline]
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, value: Fix128) {
        if ix < self.nx && iy < self.ny && iz < self.nz {
            let idx = self.index(ix, iy, iz);
            self.data[idx] = value;
        }
    }

    /// Add to a grid node. Out-of-range indices are ignored.
    #[inline]
    pub fn add(&mut self, ix: usize, iy: usize, iz: usize, delta: Fix128) {
        if ix < self.nx && iy < self.ny && iz < self.nz {
            let idx = self.index(ix, iy, iz);
            self.data[idx] = self.data[idx] + delta;
        }
    }

    /// Is a world-space point inside the field bounds?
    #[inline]
    #[must_use]
    pub fn contains(&self, p: Vec3Fix) -> bool {
        p.x >= self.min.0
            && p.x <= self.max.0
            && p.y >= self.min.1
            && p.y <= self.max.1
            && p.z >= self.min.2
            && p.z <= self.max.2
    }

    /// Fractional grid coordinates of a world-space point.
    #[inline]
    fn world_to_grid(&self, p: Vec3Fix) -> (Fix128, Fix128, Fix128) {
        (
            (p.x - self.min.0) / self.cell.0,
            (p.y - self.min.1) / self.cell.1,
            (p.z - self.min.2) / self.cell.2,
        )
    }

    /// Bracketing node indices and the interpolation fraction for one axis.
    ///
    /// The coordinate is clamped to `[0, n - 1]` first, so a point outside the
    /// grid samples (and splats onto) the nearest boundary node.
    #[inline]
    fn locate(g: Fix128, n: usize) -> (usize, usize, Fix128) {
        let top = Fix128::from_int((n - 1) as i64);
        let g = if g.is_negative() {
            Fix128::ZERO
        } else if g > top {
            top
        } else {
            g
        };
        let floor = g.floor();
        let i0 = (floor.hi as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, g - floor)
    }

    /// The eight bracketing node indices and the three fractions for a point.
    #[inline]
    fn cell_of(&self, p: Vec3Fix) -> ([usize; 8], Fix128, Fix128, Fix128) {
        let (gx, gy, gz) = self.world_to_grid(p);
        let (ix0, ix1, fx) = Self::locate(gx, self.nx);
        let (iy0, iy1, fy) = Self::locate(gy, self.ny);
        let (iz0, iz1, fz) = Self::locate(gz, self.nz);

        let nx = self.nx;
        let ny_nx = self.ny * nx;
        let corners = [
            iz0 * ny_nx + iy0 * nx + ix0, // 000
            iz0 * ny_nx + iy0 * nx + ix1, // 100
            iz0 * ny_nx + iy1 * nx + ix0, // 010
            iz0 * ny_nx + iy1 * nx + ix1, // 110
            iz1 * ny_nx + iy0 * nx + ix0, // 001
            iz1 * ny_nx + iy0 * nx + ix1, // 101
            iz1 * ny_nx + iy1 * nx + ix0, // 011
            iz1 * ny_nx + iy1 * nx + ix1, // 111
        ];
        (corners, fx, fy, fz)
    }

    /// Trilinear interpolation at a world-space point.
    ///
    /// Written as nested lerps `a + (b - a) * f`, which reproduces any
    /// function that is affine in `(x, y, z)` exactly whenever the
    /// intermediate products are exact — the property
    /// `trilinear_reproduces_a_linear_function_bit_exactly` pins.
    #[must_use]
    pub fn sample(&self, p: Vec3Fix) -> Fix128 {
        let (c, fx, fy, fz) = self.cell_of(p);
        let d = &self.data;

        let c00 = d[c[0]] + (d[c[1]] - d[c[0]]) * fx;
        let c10 = d[c[2]] + (d[c[3]] - d[c[2]]) * fx;
        let c01 = d[c[4]] + (d[c[5]] - d[c[4]]) * fx;
        let c11 = d[c[6]] + (d[c[7]] - d[c[6]]) * fx;

        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;

        c0 + (c1 - c0) * fz
    }

    /// Central-difference gradient at a world-space point.
    ///
    /// The stencil is one cell wide (`±cell/2`), so for an affine field the
    /// result is the exact coefficient vector as long as the whole stencil
    /// stays inside the grid; at the boundary the clamp in [`Self::sample`]
    /// shortens one arm and the estimate degrades to a one-sided difference
    /// scaled by the full cell.
    #[must_use]
    pub fn gradient(&self, p: Vec3Fix) -> Vec3Fix {
        let two = Fix128::from_int(2);
        let ex = self.cell.0 / two;
        let ey = self.cell.1 / two;
        let ez = self.cell.2 / two;

        let dx = (self.sample(Vec3Fix::new(p.x + ex, p.y, p.z))
            - self.sample(Vec3Fix::new(p.x - ex, p.y, p.z)))
            / self.cell.0;
        let dy = (self.sample(Vec3Fix::new(p.x, p.y + ey, p.z))
            - self.sample(Vec3Fix::new(p.x, p.y - ey, p.z)))
            / self.cell.1;
        let dz = (self.sample(Vec3Fix::new(p.x, p.y, p.z + ez))
            - self.sample(Vec3Fix::new(p.x, p.y, p.z - ez)))
            / self.cell.2;

        Vec3Fix::new(dx, dy, dz)
    }

    /// Deposit `value` onto the eight nodes around `p` with trilinear weights.
    ///
    /// This is the transpose of [`Self::sample`]: for any field `g`,
    /// `dot(splat_of_one_at(p), g) == sample(g, p)` up to rounding. The
    /// weights sum to one in exact arithmetic, so the deposited total is
    /// `value`; the rounding bound is derived in
    /// `splat_conserves_total_mass_within_the_derived_bound`.
    ///
    /// A point outside the grid, or on a face, has coincident corners — the
    /// contributions then land on the same node and the total is unaffected.
    ///
    /// Note that `sim_field::ScalarField3D::splat` means something
    /// different: it spreads a value over a *radius* with a smoothstep
    /// falloff, which is not mass-preserving and is not an adjoint of its
    /// `sample`.
    pub fn splat(&mut self, p: Vec3Fix, value: Fix128) {
        let (c, fx, fy, fz) = self.cell_of(p);
        let one = Fix128::ONE;
        let (gx, gy, gz) = (one - fx, one - fy, one - fz);

        let w = [
            gx * gy * gz, // 000
            fx * gy * gz, // 100
            gx * fy * gz, // 010
            fx * fy * gz, // 110
            gx * gy * fz, // 001
            fx * gy * fz, // 101
            gx * fy * fz, // 011
            fx * fy * fz, // 111
        ];

        for (idx, weight) in c.iter().zip(w.iter()) {
            self.data[*idx] = self.data[*idx] + value * *weight;
        }
    }

    /// One explicit-Euler step of `dT/dt = rate * laplacian(T)`.
    ///
    /// Seven-point stencil with a reflective ghost node on every face (see the
    /// module docs). Stability requires
    /// `rate * dt * (2/hx² + 2/hy² + 2/hz²) <= 1`; the method does not enforce
    /// it, exactly as the `f32` field does not.
    pub fn diffuse(&mut self, dt: Fix128, rate: Fix128) {
        let two = Fix128::from_int(2);
        let hx2 = self.cell.0 * self.cell.0;
        let hy2 = self.cell.1 * self.cell.1;
        let hz2 = self.cell.2 * self.cell.2;
        let rate_dt = rate * dt;

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let ny_nx = ny * nx;

        for iz in 0..nz {
            let iz_off = iz * ny_nx;
            for iy in 0..ny {
                let iy_off = iz_off + iy * nx;
                for ix in 0..nx {
                    let idx = iy_off + ix;
                    let c = self.data[idx];
                    let two_c = two * c;
                    let mut lap = Fix128::ZERO;

                    if nx > 1 {
                        let xm = if ix > 0 {
                            self.data[idx - 1]
                        } else {
                            self.data[idx + 1]
                        };
                        let xp = if ix + 1 < nx {
                            self.data[idx + 1]
                        } else {
                            self.data[idx - 1]
                        };
                        lap = lap + (xm + xp - two_c) / hx2;
                    }
                    if ny > 1 {
                        let ym = if iy > 0 {
                            self.data[idx - nx]
                        } else {
                            self.data[idx + nx]
                        };
                        let yp = if iy + 1 < ny {
                            self.data[idx + nx]
                        } else {
                            self.data[idx - nx]
                        };
                        lap = lap + (ym + yp - two_c) / hy2;
                    }
                    if nz > 1 {
                        let zm = if iz > 0 {
                            self.data[idx - ny_nx]
                        } else {
                            self.data[idx + ny_nx]
                        };
                        let zp = if iz + 1 < nz {
                            self.data[idx + ny_nx]
                        } else {
                            self.data[idx - ny_nx]
                        };
                        lap = lap + (zm + zp - two_c) / hz2;
                    }

                    self.scratch[idx] = c + rate_dt * lap;
                }
            }
        }

        core::mem::swap(&mut self.data, &mut self.scratch);
    }

    /// Relax every cell toward `target`: `v = (v - target) * e^(-rate*dt) + target`.
    ///
    /// The factor comes from [`Fix128::exp`], whose relative error is ≲ 1e-6;
    /// it is deterministic but not exact, so this is not a bit-exact
    /// reproduction of the continuous exponential.
    pub fn decay_toward(&mut self, target: Fix128, rate: Fix128, dt: Fix128) {
        let factor = (-(rate * dt)).exp();
        for v in &mut self.data {
            *v = (*v - target) * factor + target;
        }
    }

    /// Relax every cell toward zero.
    pub fn decay(&mut self, rate: Fix128, dt: Fix128) {
        self.decay_toward(Fix128::ZERO, rate, dt);
    }

    /// Clamp every cell into `[lo, hi]`.
    pub fn clamp(&mut self, lo: Fix128, hi: Fix128) {
        for v in &mut self.data {
            if *v < lo {
                *v = lo;
            } else if *v > hi {
                *v = hi;
            }
        }
    }

    /// Set every cell to `value`.
    pub fn fill(&mut self, value: Fix128) {
        self.data.fill(value);
    }

    /// Set every cell to zero.
    pub fn clear(&mut self) {
        self.fill(Fix128::ZERO);
    }

    /// Sum of every cell.
    ///
    /// Exact: [`Fix128`] addition does not round, so the result does not
    /// depend on the summation order.
    #[must_use]
    pub fn sum(&self) -> Fix128 {
        self.data.iter().fold(Fix128::ZERO, |acc, v| acc + *v)
    }

    /// Largest cell value, or [`Fix128::ZERO`] for an impossible empty grid.
    #[must_use]
    pub fn max_value(&self) -> Fix128 {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
            .unwrap_or(Fix128::ZERO)
    }

    /// Do two channels describe the same grid?
    #[must_use]
    pub fn same_grid_as(&self, other: &Self) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && self.nz == other.nz
            && self.min == other.min
            && self.max == other.max
    }

    /// Reject a channel that does not describe the same grid as `self`.
    fn require_same_grid(&self, other: &Self) -> Result<(), CoupledFieldError> {
        if self.nx != other.nx || self.ny != other.ny || self.nz != other.nz {
            return Err(CoupledFieldError::ResolutionMismatch {
                channel: (self.nx, self.ny, self.nz),
                other: (other.nx, other.ny, other.nz),
            });
        }
        if self.min != other.min || self.max != other.max {
            return Err(CoupledFieldError::BoundsMismatch);
        }
        Ok(())
    }

    /// Reject an `f32` field that does not describe the same grid as `self`.
    #[cfg(feature = "std")]
    fn require_same_grid_f32(&self, other: &ScalarField3D) -> Result<(), CoupledFieldError> {
        if self.nx != other.nx || self.ny != other.ny || self.nz != other.nz {
            return Err(CoupledFieldError::ResolutionMismatch {
                channel: (self.nx, self.ny, self.nz),
                other: (other.nx, other.ny, other.nz),
            });
        }
        let min = (
            Fix128::from_f32(other.min.0),
            Fix128::from_f32(other.min.1),
            Fix128::from_f32(other.min.2),
        );
        let max = (
            Fix128::from_f32(other.max.0),
            Fix128::from_f32(other.max.1),
            Fix128::from_f32(other.max.2),
        );
        if self.min != min || self.max != max {
            return Err(CoupledFieldError::BoundsMismatch);
        }
        Ok(())
    }

    /// Add another channel cell by cell.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::ResolutionMismatch`] or
    /// [`CoupledFieldError::BoundsMismatch`] if the grids differ.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), CoupledFieldError> {
        self.require_same_grid(other)?;
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            *dst = *dst + *src;
        }
        Ok(())
    }

    /// Divide every cell by `divisor`, or do nothing when it is zero.
    pub fn scale_div(&mut self, divisor: Fix128) {
        if divisor.is_zero() {
            return;
        }
        for v in &mut self.data {
            *v = *v / divisor;
        }
    }

    /// Blend another channel in: `self = self + (other - self) * weight`.
    ///
    /// `weight == 0` leaves `self` untouched, `weight == 1` replaces it.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::ResolutionMismatch`] or
    /// [`CoupledFieldError::BoundsMismatch`] if the grids differ.
    pub fn blend_from(&mut self, other: &Self, weight: Fix128) -> Result<(), CoupledFieldError> {
        self.require_same_grid(other)?;
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            *dst = *dst + (*src - *dst) * weight;
        }
        Ok(())
    }

    /// Load the channel from an `f32` field on the same grid.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::ResolutionMismatch`] or
    /// [`CoupledFieldError::BoundsMismatch`] if the grids differ.
    #[cfg(feature = "std")]
    pub fn copy_from_f32(&mut self, src: &ScalarField3D) -> Result<(), CoupledFieldError> {
        self.require_same_grid_f32(src)?;
        for (dst, v) in self.data.iter_mut().zip(src.data.iter()) {
            *dst = Fix128::from_f32(*v);
        }
        Ok(())
    }

    /// Store the channel into an `f32` field on the same grid.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::ResolutionMismatch`] or
    /// [`CoupledFieldError::BoundsMismatch`] if the grids differ.
    #[cfg(feature = "std")]
    pub fn write_to_f32(&self, dst: &mut ScalarField3D) -> Result<(), CoupledFieldError> {
        self.require_same_grid_f32(dst)?;
        for (v, cell) in dst.data.iter_mut().zip(self.data.iter()) {
            *v = cell.to_f32();
        }
        Ok(())
    }
}

// ============================================================================
// TemperatureRise
// ============================================================================

/// A field of temperature **rises** above a stress-free reference, in K.
///
/// # Why this is a separate type
///
/// A temperature `T` and a temperature rise `ΔT` carry the same unit and are
/// not the same kind of thing: `T` is a point of an affine space and `ΔT` is a
/// vector of the space that acts on it, exactly as `std::time::Instant` is to
/// `std::time::Duration`. Two points subtract to a vector, a point plus a
/// vector is a point, and **the sum of two points is not defined** — so the
/// arithmetic mean of two absolute temperatures is an absolute temperature,
/// which is why [`reconcile_mean`] resolves multiplicity without resolving the
/// unit.
///
/// `linear_elastic_fem::ThermalExpansion` is specified on the vector side: its
/// eigenstrain is `ε_th = α ΔT I`, so `ΔT = 0` everywhere has to mean "no
/// eigenstrain". Before this type existed, the rise and the absolute
/// temperature were both a [`CoupledField`] on the same grid under the same
/// [`CoupledScalar::coupled_name`] of `"temperature"`, and both temperature
/// owners (`thermal::ThermalModifier` and
/// `phase_change::PhaseChangeModifier`) fill theirs from
/// `ambient_temperature`, i.e. absolutely. Nothing in a `&CoupledField`
/// records which of the two it holds, so routing an owner's field straight
/// into the eigenstrain compiled and ran and loaded the reference temperature
/// itself as if it were a rise. For PLA (`E = 3500` MPa, `ν = 0.35`,
/// `α = 10⁻³` K⁻¹) clamped at a reference of 25 K that is a spurious
/// `E α T_ref / (1 − 2ν) = 291.666667` MPa of stress.
///
/// # Naming the reference is the whole point
///
/// [`Self::from_absolute`] is the only way to build one, so a caller has to
/// state the reference it is measuring from. A field that was *already* built
/// as a difference passes `Fix128::ZERO`, which says in the call that its
/// reference is zero rather than leaving the question unasked.
///
/// # Cost
///
/// The shifted field is owned, so `from_absolute` copies the grid once. The
/// shift is eager because the consumer samples the field directly and must see
/// rises at sample time; a lazily shifted view would have to be applied at
/// every sample instead.
///
/// `Debug` but deliberately not `Clone`: nothing in the crate needs to
/// duplicate a rise, and adding `Clone` later is a minor change while removing
/// it would be a breaking one.
#[derive(Debug)]
pub struct TemperatureRise {
    field: CoupledField,
}

impl TemperatureRise {
    /// Subtract a stress-free reference from an absolute temperature field.
    ///
    /// `rise(x) = absolute(x) − reference` cell by cell. [`Fix128`]
    /// subtraction is exact, so the result is the difference with no rounding
    /// and the grid is carried over unchanged.
    ///
    /// Pass `Fix128::ZERO` when `absolute` already holds rises; that is the
    /// identity, and writing it records in the call that the reference was
    /// considered.
    #[must_use]
    pub fn from_absolute(absolute: &CoupledField, reference: Fix128) -> Self {
        let mut field = absolute.clone();
        for cell in field.as_mut_slice() {
            *cell = *cell - reference;
        }
        Self { field }
    }

    /// The shifted field, for the consumer that samples it.
    pub(crate) const fn field(&self) -> &CoupledField {
        &self.field
    }
}

// ============================================================================
// Participants
// ============================================================================

/// A subsystem that owns one scalar field and agrees to share it.
///
/// This is the coupling channel the `sim_modifier::PhysicsModifier`
/// trait does not have: `update(&mut self, dt)` receives nothing but the time
/// step, and `modify_distance(&self, ..)` cannot write back, so a modifier has
/// no argument through which another modifier's field could reach it. An
/// implementor of this trait can both hand its field out and take one back.
pub trait CoupledScalar {
    /// Name of the quantity shared, for diagnostics (`"temperature"`, ...).
    fn coupled_name(&self) -> &'static str;

    /// A zero-filled channel on this subsystem's grid.
    ///
    /// # Errors
    ///
    /// [`CoupledFieldError::EmptyGrid`] if the subsystem's grid is degenerate.
    fn coupled_channel(&self) -> Result<CoupledField, CoupledFieldError>;

    /// Write this subsystem's field into `out`.
    ///
    /// # Errors
    ///
    /// A grid mismatch between the subsystem and `out`.
    fn publish(&self, out: &mut CoupledField) -> Result<(), CoupledFieldError>;

    /// Replace this subsystem's field with the channel's contents.
    ///
    /// # Errors
    ///
    /// A grid mismatch between the subsystem and `src`.
    fn adopt(&mut self, src: &CoupledField) -> Result<(), CoupledFieldError>;
}

/// Make every participant agree on the arithmetic mean of their fields.
///
/// `channel` receives the agreed field and each participant adopts it, so a
/// point that had one value per owner afterwards has one value in total. This
/// is the constraint the two temperature owners
/// (`thermal::ThermalModifier` and
/// `phase_change::PhaseChangeModifier`) otherwise lack: each advances
/// its own copy and nothing ever relates them.
///
/// The mean is computed as an exact [`Fix128`] sum followed by one division,
/// so the result does not depend on the order of `participants` — the property
/// `reconcile_is_independent_of_participant_order` pins.
///
/// The mean is a *channel* policy, not a conservation law: it is the right
/// thing when the participants model the same quantity on the same grid with
/// comparable confidence. A weighted merge is
/// [`CoupledField::blend_from`] plus [`CoupledScalar::adopt`].
///
/// # Errors
///
/// [`CoupledFieldError::NoParticipants`] for an empty slice, or a grid
/// mismatch between any participant and `channel`.
pub fn reconcile_mean(
    participants: &mut [&mut dyn CoupledScalar],
    channel: &mut CoupledField,
) -> Result<(), CoupledFieldError> {
    let Some(first) = participants.first() else {
        return Err(CoupledFieldError::NoParticipants);
    };
    let mut staging = first.coupled_channel()?;
    channel.require_same_grid(&staging)?;

    channel.clear();
    for p in participants.iter() {
        p.publish(&mut staging)?;
        channel.add_assign(&staging)?;
    }
    channel.scale_div(Fix128::from_int(participants.len() as i64));

    for p in participants.iter_mut() {
        p.adopt(channel)?;
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    fn unit_grid() -> CoupledField {
        CoupledField::try_new(
            5,
            5,
            5,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (
                Fix128::from_int(4),
                Fix128::from_int(4),
                Fix128::from_int(4),
            ),
        )
        .expect("non-degenerate grid")
    }

    #[test]
    fn empty_grid_is_rejected() {
        let err = CoupledField::try_new(
            0,
            2,
            2,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (Fix128::ONE, Fix128::ONE, Fix128::ONE),
        )
        .expect_err("a zero dimension must be refused");
        assert_eq!(
            err,
            CoupledFieldError::EmptyGrid {
                resolution: (0, 2, 2)
            }
        );
    }

    #[test]
    fn cell_size_is_the_node_spacing() {
        let f = unit_grid();
        assert_eq!(f.cell_size().0, Fix128::ONE);
        assert_eq!(f.cell_count(), 125);
    }

    #[test]
    fn get_and_set_round_trip() {
        let mut f = unit_grid();
        f.set(1, 2, 3, Fix128::from_int(7));
        assert_eq!(f.get(1, 2, 3), Fix128::from_int(7));
        assert_eq!(f.get(0, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn sample_at_a_node_returns_that_node() {
        let mut f = unit_grid();
        f.set(2, 1, 3, Fix128::from_int(9));
        assert_eq!(f.sample(v(2, 1, 3)), Fix128::from_int(9));
    }

    #[test]
    fn grid_mismatch_is_reported_not_silently_resampled() {
        let mut a = unit_grid();
        let b = CoupledField::try_new(
            4,
            5,
            5,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (
                Fix128::from_int(4),
                Fix128::from_int(4),
                Fix128::from_int(4),
            ),
        )
        .expect("non-degenerate grid");
        assert_eq!(
            a.add_assign(&b),
            Err(CoupledFieldError::ResolutionMismatch {
                channel: (5, 5, 5),
                other: (4, 5, 5),
            })
        );
    }

    #[test]
    fn reconcile_without_participants_is_an_error() {
        let mut channel = unit_grid();
        let mut none: [&mut dyn CoupledScalar; 0] = [];
        assert_eq!(
            reconcile_mean(&mut none, &mut channel),
            Err(CoupledFieldError::NoParticipants)
        );
    }

    #[test]
    fn sum_is_exact_over_many_cells() {
        let mut f = unit_grid();
        let q = Fix128::from_ratio(1, 4);
        for cell in f.as_mut_slice() {
            *cell = q;
        }
        // 125 cells of 1/4 is exactly 31.25, with no accumulated drift.
        assert_eq!(f.sum(), Fix128::from_ratio(125, 4));
    }
}
