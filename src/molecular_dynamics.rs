//! Molecular dynamics of point particles under a pair potential: velocity
//! Verlet integration, a periodic cell list and the NVE observables.
//!
//! # Integrator
//!
//! [`VelocityVerlet::step`] advances positions `x`, velocities `v` and
//! forces `F` by `h`:
//!
//! ```text
//! v ← v + (h / 2m) F(x)
//! x ← x + h v              (wrapped back into the periodic box)
//! v ← v + (h / 2m) F(x)
//! ```
//!
//! Velocity Verlet is symplectic and time reversible: in the microcanonical
//! (NVE) ensemble the total energy error is `O(h²)` and stays bounded instead
//! of drifting. A harmonic mode of frequency `ω` is integrated exactly
//! periodically with `cos(ω_d h) = 1 − (ωh)²/2` (stable for `ωh < 2`).
//!
//! The integrator is self-contained: it does not go through
//! `PhysicsWorld::step`, so the world's frame-head force impulse and its
//! rotational integration do not enter. Particles are points (no rotation).
//!
//! # Periodic boundary and minimum image
//!
// LIMITATION(COV-PART-055): [`PeriodicBox`] is an orthorhombic box `[0, L_x) × [0, L_y) × [0, L_z)`.
//! [`PeriodicBox`] is an orthorhombic box `[0, L_x) × [0, L_y) × [0, L_z)`.
//! Positions are wrapped into it after every drift and pair separations use
//! the minimum-image convention `d ← d − L round(d / L)` (range
//! `[−L/2, L/2)`). With a cutoff `r_c` this is exact only if every
//! `L ≥ 2 r_c` (then at most one image of a particle is within `r_c`, and a
//! particle never sees its own image); a smaller box is
//! [`MdError::BoxTooSmall`].
//!
//! # Neighbour search
//!
//! [`pair_forces_cell_list`] bins the particles into cells of side at least
//! `r_c` (slightly more, so that the rounding of the cell index cannot move
//! a neighbour two cells away) and examines, for every particle, the
//! particles of the 27 surrounding cells (fewer distinct cells when a box
//! side holds only 1 or 2 cells), with periodic wrap-around. For a fixed
//! density this is `O(N)` per evaluation. [`pair_forces_all_pairs`] is the
//! `O(N²)` reference over every pair.
//!
//! The crate's [`crate::spatial::SpatialGrid`] is not used here: it clamps
//! cell coordinates at the grid edge instead of wrapping them (no periodic
//! neighbours), its grid is centred on the origin rather than aligned with
//! `[0, L)`, and its query order is the cell order of the caller's position.
//!
//! # Determinism
//!
//! `Fix128` throughout (no `std` needed). Each pair `(i, j)`, `i < j`, is
//! evaluated once from `d = minimum_image(x_i − x_j)` and its force is added
//! as `+F` to `i` and `−F` to `j`. Fixed-point addition is exact, so the sums
//! do not depend on the order in which the pairs are found: the cell list
//! and the all-pairs sum give bit-identical forces and energies, and the
//! total pair force is exactly zero. The pair order itself is also fixed
//! (by particle index, then cell index).
//!
//! # Observables
//!
//! Kinetic energy `K = Σ ½ m v²`, potential energy `U = Σ_{pairs} U(r)`,
//! total `K + U`, momentum `Σ m v`, and the instantaneous temperature
//! `T = 2K / (k_B (3N − 3))` (three degrees of freedom per particle minus the
// LIMITATION(COV-PART-037): `k_B` is a parameter: the SI value `1.380649e-23 J/K` is below the `Fix128` resolution (`2⁻⁶⁴ ≈ 5.4e-20`), so MD in `Fix128` is run in reduced units (`k_B = 1`, energies in `ε`).
//! conserved total momentum). `k_B` is a parameter: the SI value
//! `1.380649e-23 J/K` is below the `Fix128` resolution (`2⁻⁶⁴ ≈ 5.4e-20`),
//! so MD in `Fix128` is run in reduced units (`k_B = 1`, energies in `ε`).
//!
//! # Degenerate input
//!
//! `N = 0` and `N = 1` are valid (no pairs, free flight). Coincident
//! particles (`r = 0`) are [`MdError::Potential`] with the pair's indices, and
//! a failed `step` leaves the state unchanged. Non-positive box lengths,
//! masses or time steps, mismatched array lengths, and `L < 2 r_c` are
//! errors.
//!
//! # Range
//!
//! [`VelocityVerlet::step`] uses the wrapping `Fix128` operators: a kick
//! `v + F h/(2m)`, a drift `x + h v`, a force or energy sum that leaves the
//! `Fix128` range wraps silently, and a pair closer than about `2⁻³²` (where
//! `|d|²` rounds to 0) is reported as coincident.
//! [`VelocityVerlet::try_step`] is the range-checked step: it evaluates the
//! same expressions in the same order with every product, quotient, sum and
//! difference checked, returns [`MdStepError::Overflow`] or
//! [`MdStepError::PairBelowResolution`] instead, and leaves the state
//! unchanged. Where `try_step` succeeds its result is bit-identical to
//! `step`'s. [`MdParticipant`] advances with `try_step`.

use crate::math::{Fix128, Vec3Fix};
use crate::pair_potential::{PairPotential, PairPotentialError, Truncated};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Why an MD system could not be built or advanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdError {
    /// A box side `L ≤ 0`.
    NonPositiveBoxLength,
    /// A box side `L < 2 r_c`: the minimum-image convention does not hold.
    BoxTooSmall,
    /// Positions, velocities and masses have different lengths.
    LengthMismatch,
    /// A particle mass `m ≤ 0`.
    NonPositiveMass,
    /// Time step `h ≤ 0`.
    NonPositiveTimestep,
    /// Boltzmann constant `k_B ≤ 0`.
    NonPositiveBoltzmannConstant,
    /// Temperature needs `N ≥ 2` (`3N − 3 > 0` degrees of freedom).
    TooFewParticles,
    /// The pair potential failed for particles `i < j`.
    Potential {
        /// First particle.
        i: usize,
        /// Second particle.
        j: usize,
        /// The potential's error.
        error: PairPotentialError,
    },
}

impl core::fmt::Display for MdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveBoxLength => f.write_str("periodic box lengths must be positive"),
            Self::BoxTooSmall => {
                f.write_str("periodic box side is smaller than twice the cutoff radius")
            }
            Self::LengthMismatch => {
                f.write_str("positions, velocities and masses must have the same length")
            }
            Self::NonPositiveMass => f.write_str("particle masses must be positive"),
            Self::NonPositiveTimestep => f.write_str("time step must be positive"),
            Self::NonPositiveBoltzmannConstant => {
                f.write_str("Boltzmann constant must be positive")
            }
            Self::TooFewParticles => f.write_str("temperature needs at least two particles"),
            Self::Potential { i, j, error } => write!(f, "pair ({i}, {j}): {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MdError {}

/// Why [`VelocityVerlet::try_step`] did not advance the system. The state is
/// unchanged in every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MdStepError {
    /// An error that [`VelocityVerlet::step`] reports as well.
    Md(MdError),
    /// A value of the step left the `Fix128` range (`|x| ≥ 2⁶³`): the kick
    /// factor `h/(2m)`, a kick, a drift, the wrap into the box, a pair
    /// separation or its square, a pair force `d F/r`, the force on a
    /// particle or the potential energy sum.
    Overflow,
    /// Particles `i < j` are apart (`d ≠ 0`) but closer than the length
    /// resolution: `|d|²` rounds to 0 (about `|d| < 2⁻³²`), so `|d|` and the
    /// force direction cannot be evaluated.
    PairBelowResolution {
        /// First particle.
        i: usize,
        /// Second particle.
        j: usize,
    },
}

impl From<MdError> for MdStepError {
    fn from(e: MdError) -> Self {
        Self::Md(e)
    }
}

impl core::fmt::Display for MdStepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Md(e) => core::fmt::Display::fmt(e, f),
            Self::Overflow => f.write_str("a value of the step is outside the fixed-point range"),
            Self::PairBelowResolution { i, j } => write!(
                f,
                "pair ({i}, {j}) is closer than the length resolution of the fixed-point range"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MdStepError {}

// ---------------------------------------------------------------------------
// Range-checked arithmetic (same values as the operators where `Some`)
// ---------------------------------------------------------------------------

/// `a − b` as `Sub`, `None` if `|a − b| ≥ 2⁶³`.
pub(crate) fn checked_sub(a: Fix128, b: Fix128) -> Option<Fix128> {
    let ra = ((a.hi as i128) << 64) | (a.lo as i128);
    let rb = ((b.hi as i128) << 64) | (b.lo as i128);
    let d = ra.checked_sub(rb)?;
    Some(Fix128::from_raw((d >> 64) as i64, d as u64))
}

/// `a / b` as `Div` (truncating), `None` if `b = 0` or the quotient has
/// `|a / b| ≥ 2⁶³` (where `Div` keeps the low 64 bits of the integer part).
pub(crate) fn checked_quotient(a: Fix128, b: Fix128) -> Option<Fix128> {
    if b.is_zero() {
        return None;
    }
    let mag = |f: Fix128| (((f.hi as i128) << 64) | (f.lo as i128)).unsigned_abs();
    // integer part of |a| / |b|, the `quot_hi` of `Div`
    if mag(a) / mag(b) >= 1u128 << 63 {
        return None;
    }
    Some(a / b)
}

fn checked_add3(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        a.x.checked_add(b.x)?,
        a.y.checked_add(b.y)?,
        a.z.checked_add(b.z)?,
    ))
}

fn checked_sub3(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        checked_sub(a.x, b.x)?,
        checked_sub(a.y, b.y)?,
        checked_sub(a.z, b.z)?,
    ))
}

// ---------------------------------------------------------------------------
// Periodic box
// ---------------------------------------------------------------------------

/// Orthorhombic periodic box `[0, L_x) × [0, L_y) × [0, L_z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicBox {
    lengths: Vec3Fix,
}

fn wrap_scalar(x: Fix128, l: Fix128) -> Fix128 {
    if x >= Fix128::ZERO && x < l {
        return x;
    }
    // x − k L with integer k: exact in fixed point; the loops correct the
    // truncation of x / L by at most one period
    let k = (x / l).floor();
    let mut y = x - k * l;
    while y < Fix128::ZERO {
        y = y + l;
    }
    while y >= l {
        y = y - l;
    }
    y
}

/// [`wrap_scalar`] with every operation range-checked.
fn wrap_scalar_checked(x: Fix128, l: Fix128) -> Option<Fix128> {
    if x >= Fix128::ZERO && x < l {
        return Some(x);
    }
    let k = checked_quotient(x, l)?.floor();
    let mut y = checked_sub(x, k.checked_mul(l)?)?;
    while y < Fix128::ZERO {
        y = y.checked_add(l)?;
    }
    while y >= l {
        y = checked_sub(y, l)?;
    }
    Some(y)
}

impl PeriodicBox {
    /// Box with side lengths `L_x, L_y, L_z > 0`.
    ///
    /// # Errors
    ///
    /// [`MdError::NonPositiveBoxLength`].
    pub fn new(lengths: Vec3Fix) -> Result<Self, MdError> {
        if lengths.x <= Fix128::ZERO || lengths.y <= Fix128::ZERO || lengths.z <= Fix128::ZERO {
            return Err(MdError::NonPositiveBoxLength);
        }
        Ok(Self { lengths })
    }

    /// Cubic box of side `L > 0`.
    ///
    /// # Errors
    ///
    /// [`MdError::NonPositiveBoxLength`].
    pub fn cubic(length: Fix128) -> Result<Self, MdError> {
        Self::new(Vec3Fix::new(length, length, length))
    }

    /// Side lengths.
    #[must_use]
    pub const fn lengths(&self) -> Vec3Fix {
        self.lengths
    }

    /// The image of `p` inside `[0, L)` on every axis.
    #[must_use]
    pub fn wrap(&self, p: Vec3Fix) -> Vec3Fix {
        Vec3Fix::new(
            wrap_scalar(p.x, self.lengths.x),
            wrap_scalar(p.y, self.lengths.y),
            wrap_scalar(p.z, self.lengths.z),
        )
    }

    /// Minimum-image separation: the image of `d` in `[−L/2, L/2)` on every
    /// axis (exact: `wrap(d + L/2) − L/2`).
    #[must_use]
    pub fn minimum_image(&self, d: Vec3Fix) -> Vec3Fix {
        let h = Vec3Fix::new(
            self.lengths.x.half(),
            self.lengths.y.half(),
            self.lengths.z.half(),
        );
        self.wrap(d + h) - h
    }

    fn wrap_checked(&self, p: Vec3Fix) -> Option<Vec3Fix> {
        Some(Vec3Fix::new(
            wrap_scalar_checked(p.x, self.lengths.x)?,
            wrap_scalar_checked(p.y, self.lengths.y)?,
            wrap_scalar_checked(p.z, self.lengths.z)?,
        ))
    }

    fn minimum_image_checked(&self, d: Vec3Fix) -> Option<Vec3Fix> {
        let h = Vec3Fix::new(
            self.lengths.x.half(),
            self.lengths.y.half(),
            self.lengths.z.half(),
        );
        checked_sub3(self.wrap_checked(checked_add3(d, h)?)?, h)
    }

    fn check_cutoff(&self, cutoff: Fix128) -> Result<(), MdError> {
        let two_rc = cutoff.double();
        if self.lengths.x < two_rc || self.lengths.y < two_rc || self.lengths.z < two_rc {
            return Err(MdError::BoxTooSmall);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pair forces
// ---------------------------------------------------------------------------

/// Forces on every particle and the total pair potential energy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairForces {
    /// Force on each particle (same order as the positions).
    pub forces: Vec<Vec3Fix>,
    /// `Σ_{pairs} U(r)`.
    pub potential_energy: Fix128,
}

/// Adds the pair `(i, j)`, `i < j`, if it is inside the cutoff.
fn accumulate_pair<P: PairPotential>(
    potential: &Truncated<P>,
    periodic_box: &PeriodicBox,
    positions: &[Vec3Fix],
    (i, j): (usize, usize),
    cutoff_sq: Fix128,
    out: &mut PairForces,
) -> Result<(), MdError> {
    let d = periodic_box.minimum_image(positions[i] - positions[j]);
    let r_sq = d.length_squared();
    if r_sq >= cutoff_sq {
        return Ok(());
    }
    let wrap_err = |error| MdError::Potential { i, j, error };
    let r = r_sq.sqrt();
    let f = potential.force(r).map_err(wrap_err)?;
    let u = potential.energy(r).map_err(wrap_err)?;
    let fv = d * (f / r);
    out.forces[i] = out.forces[i] + fv;
    out.forces[j] = out.forces[j] - fv;
    out.potential_energy = out.potential_energy + u;
    Ok(())
}

/// [`accumulate_pair`] with every operation range-checked, and a pair whose
/// `|d|²` rounds to 0 at `d ≠ 0` refused.
fn accumulate_pair_checked<P: PairPotential>(
    potential: &Truncated<P>,
    periodic_box: &PeriodicBox,
    positions: &[Vec3Fix],
    (i, j): (usize, usize),
    cutoff_sq: Fix128,
    out: &mut PairForces,
) -> Result<(), MdStepError> {
    let ovf = MdStepError::Overflow;
    let d = periodic_box
        .minimum_image_checked(checked_sub3(positions[i], positions[j]).ok_or(ovf)?)
        .ok_or(ovf)?;
    let r_sq = d.checked_length_squared().ok_or(ovf)?;
    if r_sq >= cutoff_sq {
        return Ok(());
    }
    if r_sq.is_zero() && d != Vec3Fix::ZERO {
        return Err(MdStepError::PairBelowResolution { i, j });
    }
    let wrap_err = |error| MdStepError::Md(MdError::Potential { i, j, error });
    let r = r_sq.sqrt();
    let f = potential.force(r).map_err(wrap_err)?;
    let u = potential.energy(r).map_err(wrap_err)?;
    let fv = d
        .checked_scale(checked_quotient(f, r).ok_or(ovf)?)
        .ok_or(ovf)?;
    out.forces[i] = checked_add3(out.forces[i], fv).ok_or(ovf)?;
    out.forces[j] = checked_sub3(out.forces[j], fv).ok_or(ovf)?;
    out.potential_energy = out.potential_energy.checked_add(u).ok_or(ovf)?;
    Ok(())
}

fn empty_forces(n: usize) -> PairForces {
    PairForces {
        forces: vec![Vec3Fix::ZERO; n],
        potential_energy: Fix128::ZERO,
    }
}

/// `O(N²)` reference: every pair `i < j` in index order.
///
/// # Errors
///
/// [`MdError::BoxTooSmall`] if a side is below `2 r_c`;
/// [`MdError::Potential`] for the first failing pair.
pub fn pair_forces_all_pairs<P: PairPotential>(
    potential: &Truncated<P>,
    periodic_box: &PeriodicBox,
    positions: &[Vec3Fix],
) -> Result<PairForces, MdError> {
    periodic_box.check_cutoff(potential.cutoff())?;
    let cutoff_sq = potential.cutoff() * potential.cutoff();
    let mut out = empty_forces(positions.len());
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            accumulate_pair(
                potential,
                periodic_box,
                positions,
                (i, j),
                cutoff_sq,
                &mut out,
            )?;
        }
    }
    Ok(out)
}

/// Number of cells along one axis: the largest `n ≥ 1` with
/// `n (r_c + r_c 2⁻²⁰) ≤ L`.
fn cells_along(length: Fix128, cutoff: Fix128) -> usize {
    let width = cutoff + cutoff.shr_bits(20);
    let n = (length / width).hi;
    if n < 1 {
        1
    } else {
        n as usize
    }
}

fn cell_coord(x: Fix128, n: usize, length: Fix128) -> usize {
    let c = (x * Fix128::from_int(n as i64) / length).hi;
    c.clamp(0, n as i64 - 1) as usize
}

/// Distinct cells `c − 1, c, c + 1` (mod `n`) in ascending order.
fn neighbour_cells(c: usize, n: usize) -> ([usize; 3], usize) {
    match n {
        1 => ([0, 0, 0], 1),
        2 => ([0, 1, 0], 2),
        _ => {
            let mut a = [(c + n - 1) % n, c, (c + 1) % n];
            a.sort_unstable();
            (a, 3)
        }
    }
}

/// `O(N)` (fixed density) cell-list evaluation with periodic neighbours.
/// Bit-identical to [`pair_forces_all_pairs`].
///
/// # Errors
///
/// [`MdError::BoxTooSmall`] if a side is below `2 r_c`;
/// [`MdError::Potential`] for the first failing pair found.
pub fn pair_forces_cell_list<P: PairPotential>(
    potential: &Truncated<P>,
    periodic_box: &PeriodicBox,
    positions: &[Vec3Fix],
) -> Result<PairForces, MdError> {
    let cutoff = potential.cutoff();
    periodic_box.check_cutoff(cutoff)?;
    let cutoff_sq = cutoff * cutoff;
    let l = periodic_box.lengths();
    let dims = [
        cells_along(l.x, cutoff),
        cells_along(l.y, cutoff),
        cells_along(l.z, cutoff),
    ];
    let coords: Vec<[usize; 3]> = positions
        .iter()
        .map(|p| {
            let p = periodic_box.wrap(*p);
            [
                cell_coord(p.x, dims[0], l.x),
                cell_coord(p.y, dims[1], l.y),
                cell_coord(p.z, dims[2], l.z),
            ]
        })
        .collect();
    let mut out = empty_forces(positions.len());
    for_each_cell_pair(dims, &coords, |pair| {
        accumulate_pair(
            potential,
            periodic_box,
            positions,
            pair,
            cutoff_sq,
            &mut out,
        )
    })?;
    Ok(out)
}

/// [`pair_forces_cell_list`] with every operation range-checked (the same
/// values where it succeeds).
fn pair_forces_cell_list_checked<P: PairPotential>(
    potential: &Truncated<P>,
    periodic_box: &PeriodicBox,
    positions: &[Vec3Fix],
) -> Result<PairForces, MdStepError> {
    let ovf = MdStepError::Overflow;
    let cutoff = potential.cutoff();
    periodic_box.check_cutoff(cutoff)?;
    let cutoff_sq = cutoff.checked_mul(cutoff).ok_or(ovf)?;
    let l = periodic_box.lengths();
    let along = |length: Fix128| -> Result<usize, MdStepError> {
        let width = cutoff.checked_add(cutoff.shr_bits(20)).ok_or(ovf)?;
        let n = checked_quotient(length, width).ok_or(ovf)?.hi;
        Ok(if n < 1 { 1 } else { n as usize })
    };
    let dims = [along(l.x)?, along(l.y)?, along(l.z)?];
    let coord = |x: Fix128, n: usize, length: Fix128| -> Result<usize, MdStepError> {
        let xn = x.checked_mul(Fix128::from_int(n as i64)).ok_or(ovf)?;
        let c = checked_quotient(xn, length).ok_or(ovf)?.hi;
        Ok(c.clamp(0, n as i64 - 1) as usize)
    };
    let mut coords: Vec<[usize; 3]> = Vec::with_capacity(positions.len());
    for p in positions {
        let p = periodic_box.wrap_checked(*p).ok_or(ovf)?;
        coords.push([
            coord(p.x, dims[0], l.x)?,
            coord(p.y, dims[1], l.y)?,
            coord(p.z, dims[2], l.z)?,
        ]);
    }
    let mut out = empty_forces(positions.len());
    for_each_cell_pair(dims, &coords, |pair| {
        accumulate_pair_checked(
            potential,
            periodic_box,
            positions,
            pair,
            cutoff_sq,
            &mut out,
        )
    })?;
    Ok(out)
}

/// Calls `visit((i, j))` for every pair `i < j` of particles in the same or
/// adjacent cells (periodic), in the order of `i`, then the neighbour cell
/// (`z`, `y`, `x` ascending), then `j` ascending. Stops at the first error.
fn for_each_cell_pair<E>(
    dims: [usize; 3],
    coords: &[[usize; 3]],
    mut visit: impl FnMut((usize, usize)) -> Result<(), E>,
) -> Result<(), E> {
    let n_cells = dims[0] * dims[1] * dims[2];
    let flat = |c: [usize; 3]| c[0] + dims[0] * (c[1] + dims[1] * c[2]);

    // CSR buckets, members in ascending particle index
    let mut starts = vec![0usize; n_cells + 1];
    for c in coords {
        starts[flat(*c) + 1] += 1;
    }
    for k in 0..n_cells {
        starts[k + 1] += starts[k];
    }
    let mut cursor = starts.clone();
    let mut members = vec![0usize; coords.len()];
    for (idx, c) in coords.iter().enumerate() {
        let cell = flat(*c);
        members[cursor[cell]] = idx;
        cursor[cell] += 1;
    }

    for (i, ci) in coords.iter().enumerate() {
        let (zs, nz) = neighbour_cells(ci[2], dims[2]);
        let (ys, ny) = neighbour_cells(ci[1], dims[1]);
        let (xs, nx) = neighbour_cells(ci[0], dims[0]);
        for &cz in &zs[..nz] {
            for &cy in &ys[..ny] {
                for &cx in &xs[..nx] {
                    let cell = flat([cx, cy, cz]);
                    for &j in &members[starts[cell]..starts[cell + 1]] {
                        if j > i {
                            visit((i, j))?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Velocity Verlet
// ---------------------------------------------------------------------------

/// A periodic system of point particles under a truncated pair potential,
/// advanced by velocity Verlet (NVE).
///
/// Not the same type as [`crate::nbody::VelocityVerlet`]: that one is an
/// integrator that does not own the system (the caller passes positions,
/// velocities and masses to every `step`), while this one is
/// `VelocityVerlet<P: PairPotential>` and owns the particles, the periodic
/// box and the potential, and steps them. Neither is re-exported at the
/// crate root; a re-export must use a distinct alias.
#[derive(Debug, Clone)]
pub struct VelocityVerlet<P> {
    potential: Truncated<P>,
    periodic_box: PeriodicBox,
    positions: Vec<Vec3Fix>,
    velocities: Vec<Vec3Fix>,
    masses: Vec<Fix128>,
    forces: Vec<Vec3Fix>,
    potential_energy: Fix128,
}

impl<P: PairPotential> VelocityVerlet<P> {
    /// Builds the system, wraps the positions into the box and evaluates the
    /// initial forces. The neighbour search radius is the potential's
    /// cutoff.
    ///
    /// # Errors
    ///
    /// [`MdError::LengthMismatch`], [`MdError::NonPositiveMass`],
    /// [`MdError::BoxTooSmall`], [`MdError::Potential`].
    pub fn new(
        potential: Truncated<P>,
        periodic_box: PeriodicBox,
        positions: Vec<Vec3Fix>,
        velocities: Vec<Vec3Fix>,
        masses: Vec<Fix128>,
    ) -> Result<Self, MdError> {
        if positions.len() != velocities.len() || positions.len() != masses.len() {
            return Err(MdError::LengthMismatch);
        }
        if masses.iter().any(|m| *m <= Fix128::ZERO) {
            return Err(MdError::NonPositiveMass);
        }
        periodic_box.check_cutoff(potential.cutoff())?;
        let positions: Vec<Vec3Fix> = positions.iter().map(|p| periodic_box.wrap(*p)).collect();
        let pf = pair_forces_cell_list(&potential, &periodic_box, &positions)?;
        Ok(Self {
            potential,
            periodic_box,
            positions,
            velocities,
            masses,
            forces: pf.forces,
            potential_energy: pf.potential_energy,
        })
    }

    /// One velocity Verlet step of length `dt`. On error the state is not
    /// changed.
    ///
    /// The arithmetic is not range-checked: a kick, drift, force or energy
    /// that leaves the `Fix128` range wraps, and a pair closer than the
    /// length resolution (`|d|²` rounds to 0) is [`MdError::Potential`] with
    /// [`PairPotentialError::NonPositiveDistance`]. [`Self::try_step`] checks
    /// the range and is bit-identical to this step where it succeeds.
    ///
    /// # Errors
    ///
    /// [`MdError::NonPositiveTimestep`], [`MdError::Potential`].
    pub fn step(&mut self, dt: Fix128) -> Result<(), MdError> {
        if dt <= Fix128::ZERO {
            return Err(MdError::NonPositiveTimestep);
        }
        let half = dt.half();
        let kicks: Vec<Fix128> = self.masses.iter().map(|m| half / *m).collect();
        let v_half: Vec<Vec3Fix> = self
            .velocities
            .iter()
            .zip(&self.forces)
            .zip(&kicks)
            .map(|((v, f), k)| *v + *f * *k)
            .collect();
        let x_new: Vec<Vec3Fix> = self
            .positions
            .iter()
            .zip(&v_half)
            .map(|(x, v)| self.periodic_box.wrap(*x + *v * dt))
            .collect();
        let pf = pair_forces_cell_list(&self.potential, &self.periodic_box, &x_new)?;
        self.velocities = v_half
            .iter()
            .zip(&pf.forces)
            .zip(&kicks)
            .map(|((v, f), k)| *v + *f * *k)
            .collect();
        self.positions = x_new;
        self.forces = pf.forces;
        self.potential_energy = pf.potential_energy;
        Ok(())
    }

    /// [`Self::step`] with the range checked: the same expressions in the
    /// same order, with every product, quotient, sum and difference of the
    /// step (the kick factor `h/(2m)`, both kicks, the drift and its wrap
    /// into the box, the pair separations and their squares, the pair forces
    /// and the force and energy sums) checked against the `Fix128` range.
    ///
    /// Where it returns `Ok` the new state is bit-identical to the one
    /// [`Self::step`] produces. On error the state is not changed.
    ///
    /// # Errors
    ///
    /// [`MdStepError::Md`] with the errors of [`Self::step`];
    /// [`MdStepError::Overflow`] for a value out of range;
    /// [`MdStepError::PairBelowResolution`] for a pair at `d ≠ 0` whose
    /// `|d|²` rounds to 0. A wrap into a box much smaller than the drifted
    /// coordinate (`|x / L| ≥ 2⁶³`) is [`MdStepError::Overflow`].
    pub fn try_step(&mut self, dt: Fix128) -> Result<(), MdStepError> {
        if dt <= Fix128::ZERO {
            return Err(MdError::NonPositiveTimestep.into());
        }
        let ovf = MdStepError::Overflow;
        let half = dt.half();
        let kicks = self
            .masses
            .iter()
            .map(|m| checked_quotient(half, *m))
            .collect::<Option<Vec<Fix128>>>()
            .ok_or(ovf)?;
        let kick = |v: &Vec3Fix, f: &Vec3Fix, k: &Fix128| checked_add3(*v, f.checked_scale(*k)?);
        let v_half = self
            .velocities
            .iter()
            .zip(&self.forces)
            .zip(&kicks)
            .map(|((v, f), k)| kick(v, f, k))
            .collect::<Option<Vec<Vec3Fix>>>()
            .ok_or(ovf)?;
        let x_new = self
            .positions
            .iter()
            .zip(&v_half)
            .map(|(x, v)| {
                self.periodic_box
                    .wrap_checked(checked_add3(*x, v.checked_scale(dt)?)?)
            })
            .collect::<Option<Vec<Vec3Fix>>>()
            .ok_or(ovf)?;
        let pf = pair_forces_cell_list_checked(&self.potential, &self.periodic_box, &x_new)?;
        let v_new = v_half
            .iter()
            .zip(&pf.forces)
            .zip(&kicks)
            .map(|((v, f), k)| kick(v, f, k))
            .collect::<Option<Vec<Vec3Fix>>>()
            .ok_or(ovf)?;
        self.velocities = v_new;
        self.positions = x_new;
        self.forces = pf.forces;
        self.potential_energy = pf.potential_energy;
        Ok(())
    }

    /// Positions, wrapped into the box.
    #[must_use]
    pub fn positions(&self) -> &[Vec3Fix] {
        &self.positions
    }

    /// Velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3Fix] {
        &self.velocities
    }

    /// Forces at the current positions.
    #[must_use]
    pub fn forces(&self) -> &[Vec3Fix] {
        &self.forces
    }

    /// Masses.
    #[must_use]
    pub fn masses(&self) -> &[Fix128] {
        &self.masses
    }

    /// The periodic box.
    #[must_use]
    pub const fn periodic_box(&self) -> &PeriodicBox {
        &self.periodic_box
    }

    /// The truncated pair potential.
    #[must_use]
    pub const fn potential(&self) -> &Truncated<P> {
        &self.potential
    }

    /// `Σ_{pairs} U(r)` at the current positions.
    #[must_use]
    pub const fn potential_energy(&self) -> Fix128 {
        self.potential_energy
    }

    /// `Σ ½ m v²`.
    #[must_use]
    pub fn kinetic_energy(&self) -> Fix128 {
        self.masses
            .iter()
            .zip(&self.velocities)
            .fold(Fix128::ZERO, |acc, (m, v)| {
                acc + (*m * v.length_squared()).half()
            })
    }

    /// Kinetic plus potential energy.
    #[must_use]
    pub fn total_energy(&self) -> Fix128 {
        self.kinetic_energy() + self.potential_energy
    }

    /// `Σ m v`.
    #[must_use]
    pub fn momentum(&self) -> Vec3Fix {
        self.masses
            .iter()
            .zip(&self.velocities)
            .fold(Vec3Fix::ZERO, |acc, (m, v)| acc + *v * *m)
    }

    /// Instantaneous temperature `2K / (k_B (3N − 3))`.
    ///
    /// # Errors
    ///
    /// [`MdError::NonPositiveBoltzmannConstant`], [`MdError::TooFewParticles`]
    /// (`N < 2`).
    pub fn instantaneous_temperature(&self, boltzmann_constant: Fix128) -> Result<Fix128, MdError> {
        if boltzmann_constant <= Fix128::ZERO {
            return Err(MdError::NonPositiveBoltzmannConstant);
        }
        let n = self.positions.len();
        if n < 2 {
            return Err(MdError::TooFewParticles);
        }
        let dof = Fix128::from_int(3 * n as i64 - 3);
        Ok(self.kinetic_energy().double() / (dof * boltzmann_constant))
    }
}

// ---------------------------------------------------------------------------
// World participant
// ---------------------------------------------------------------------------

/// Snapshot tag of [`MdParticipant`] (every pair potential): the ASCII code
/// `MDVV`, big endian.
pub const MD_PARTICIPANT_KIND: ParticipantKind = ParticipantKind::new(u32::from_be_bytes(*b"MDVV"));

/// Observation channel of [`MdParticipant`]: kinetic energy `K`.
pub const MD_OBS_KINETIC: u32 = 0;
/// Observation channel of [`MdParticipant`]: potential energy `U`.
pub const MD_OBS_POTENTIAL: u32 = 1;
/// Observation channel of [`MdParticipant`]: total energy `K + U`.
pub const MD_OBS_TOTAL: u32 = 2;
/// Observation channel of [`MdParticipant`]: instantaneous temperature
/// `2K / (k_B (3N − 3))`, reported only for `N ≥ 2`.
pub const MD_OBS_TEMPERATURE: u32 = 3;

/// Payload layout version written by [`MdParticipant`].
const MD_STATE_VERSION: u32 = 1;
/// `version: u32`, `digest: u64`, box lengths (3 `Fix128`), cutoff
/// (`Fix128`), `count: u64`.
const MD_HEADER_LEN: usize = 4 + 8 + 3 * 16 + 16 + 8;
/// Mass, position, velocity and force: ten `Fix128` per particle.
const MD_PARTICLE_LEN: usize = 10 * 16;

/// A [`VelocityVerlet`] system as a participant of the world's substep loop:
/// one [`VelocityVerlet::try_step`] of the world substep width `h` per substep
/// ([`StepRule::FollowSubstep`](crate::world_participant::StepRule::FollowSubstep)).
///
/// # Time step
///
/// Velocity Verlet is stable for `ωh < 2` on the fastest mode; the world's
/// substep width is that `h`, so the scene has to choose it. A non-positive
/// `h` never reaches the participant: the world checks the step rule of every
/// participant before any of them runs.
///
/// # Rigid bodies
///
/// The particles do not interact with the world's rigid bodies: the
/// participant neither reads them nor stages forces on them.
///
/// # Faults
///
/// [`VelocityVerlet::try_step`] leaves the state unchanged on error. A value
/// of the step that leaves the `Fix128` range ([`MdStepError::Overflow`], a
/// kick or drift among them), a pair closer than the length resolution
/// ([`MdStepError::PairBelowResolution`]) and a pair whose potential value
/// leaves the range ([`PairPotentialError::Overflow`]) are
/// [`ParticipantFault::OutOfRange`]; every other error (coincident particles,
/// an invalid potential argument) is [`ParticipantFault::InvalidState`]. Both
/// leave the participant unchanged.
///
/// # Snapshot payload
///
/// Little endian: `version: u32` (1), `digest: u64`, the box lengths and the
/// truncation cutoff (`Fix128` as `hi: i64`, `lo: u64`), `count: u64`, then
/// per particle mass, position, velocity and force, then the potential
/// energy. The digest is FNV-1a 64 over the box lengths, the cutoff and
/// `k_B`. [`Participant::check_state`] refuses a payload whose digest, box or
/// cutoff differs from this participant's, or with a non-positive mass
/// ([`StateError::InvalidValue`]), and one whose length does not match its
/// count ([`StateError::Length`]). The particle count is state.
///
/// The type and the parameters of the pair potential are not in the payload
/// and are not compared: restoring a payload written with another potential
/// (or other `ε`, `σ`) of the same cutoff is accepted and continues with
/// this participant's potential. The stored forces and potential energy
/// are then those of the other potential until the next step.
#[derive(Debug, Clone)]
pub struct MdParticipant<P> {
    system: VelocityVerlet<P>,
    boltzmann_constant: Fix128,
    digest: u64,
}

impl<P: PairPotential> MdParticipant<P> {
    /// A participant owning `system`; `boltzmann_constant` is the `k_B` of the
    /// temperature observation.
    ///
    /// # Errors
    ///
    /// [`MdError::NonPositiveBoltzmannConstant`].
    pub fn new(system: VelocityVerlet<P>, boltzmann_constant: Fix128) -> Result<Self, MdError> {
        if boltzmann_constant <= Fix128::ZERO {
            return Err(MdError::NonPositiveBoltzmannConstant);
        }
        let mut b = Vec::new();
        let l = system.periodic_box.lengths();
        for f in [l.x, l.y, l.z, system.potential.cutoff(), boltzmann_constant] {
            md_put_fix(&mut b, f);
        }
        let digest = md_fnv1a64(&b);
        Ok(Self {
            system,
            boltzmann_constant,
            digest,
        })
    }

    /// The system.
    #[must_use]
    pub const fn system(&self) -> &VelocityVerlet<P> {
        &self.system
    }

    /// The `k_B` of the temperature observation.
    #[must_use]
    pub const fn boltzmann_constant(&self) -> Fix128 {
        self.boltzmann_constant
    }
}

impl<P: PairPotential + Send> Participant for MdParticipant<P> {
    fn kind(&self) -> ParticipantKind {
        MD_PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        self.system.try_step(h).map_err(|e| match e {
            MdStepError::Overflow
            | MdStepError::PairBelowResolution { .. }
            | MdStepError::Md(MdError::Potential {
                error: PairPotentialError::Overflow,
                ..
            }) => ParticipantFault::OutOfRange,
            _ => ParticipantFault::InvalidState,
        })
    }

    fn observe(&self, out: &mut ObservationSink) {
        let k = self.system.kinetic_energy();
        let u = self.system.potential_energy();
        out.push(MD_OBS_KINETIC, k);
        out.push(MD_OBS_POTENTIAL, u);
        out.push(MD_OBS_TOTAL, k + u);
        if let Ok(t) = self
            .system
            .instantaneous_temperature(self.boltzmann_constant)
        {
            out.push(MD_OBS_TEMPERATURE, t);
        }
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let s = &self.system;
        out.extend_from_slice(&MD_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        let l = s.periodic_box.lengths();
        for f in [l.x, l.y, l.z, s.potential.cutoff()] {
            md_put_fix(out, f);
        }
        out.extend_from_slice(&(s.positions.len() as u64).to_le_bytes());
        for i in 0..s.positions.len() {
            md_put_fix(out, s.masses[i]);
            for v in [s.positions[i], s.velocities[i], s.forces[i]] {
                for f in [v.x, v.y, v.z] {
                    md_put_fix(out, f);
                }
            }
        }
        md_put_fix(out, s.potential_energy);
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() < MD_HEADER_LEN {
            return Err(StateError::Length {
                expected: MD_HEADER_LEN,
                found: bytes.len(),
            });
        }
        if md_read_u32(bytes, 0) != MD_STATE_VERSION || md_read_u64(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        let l = self.system.periodic_box.lengths();
        let own = [l.x, l.y, l.z, self.system.potential.cutoff()];
        if (0..4).any(|k| md_get_fix(bytes, 12 + 16 * k) != own[k]) {
            return Err(StateError::InvalidValue);
        }
        let count = md_read_u64(bytes, MD_HEADER_LEN - 8);
        let expected = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(MD_PARTICLE_LEN))
            .and_then(|b| b.checked_add(MD_HEADER_LEN + 16))
            .unwrap_or(usize::MAX);
        if bytes.len() != expected {
            return Err(StateError::Length {
                expected,
                found: bytes.len(),
            });
        }
        let masses_positive = bytes[MD_HEADER_LEN..bytes.len() - 16]
            .chunks_exact(MD_PARTICLE_LEN)
            .all(|c| md_get_fix(c, 0) > Fix128::ZERO);
        if !masses_positive {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let body = &bytes[MD_HEADER_LEN..bytes.len() - 16];
        let n = body.len() / MD_PARTICLE_LEN;
        let s = &mut self.system;
        s.masses = Vec::with_capacity(n);
        s.positions = Vec::with_capacity(n);
        s.velocities = Vec::with_capacity(n);
        s.forces = Vec::with_capacity(n);
        for c in body.chunks_exact(MD_PARTICLE_LEN) {
            let v = |k: usize| {
                Vec3Fix::new(
                    md_get_fix(c, 16 + 48 * k),
                    md_get_fix(c, 32 + 48 * k),
                    md_get_fix(c, 48 + 48 * k),
                )
            };
            s.masses.push(md_get_fix(c, 0));
            s.positions.push(v(0));
            s.velocities.push(v(1));
            s.forces.push(v(2));
        }
        s.potential_energy = md_get_fix(bytes, bytes.len() - 16);
    }
}

/// FNV-1a 64 (offset basis `0xcbf2_9ce4_8422_2325`, prime `0x100_0000_01b3`).
fn md_fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn md_put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn md_read_u32(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(a)
}

fn md_read_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn md_get_fix(b: &[u8], at: usize) -> Fix128 {
    Fix128::from_raw(md_read_u64(b, at) as i64, md_read_u64(b, at + 8))
}
