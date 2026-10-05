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

use crate::math::{Fix128, Vec3Fix};
use crate::pair_potential::{PairPotential, PairPotentialError, Truncated};

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
    let n_cells = dims[0] * dims[1] * dims[2];
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
    let flat = |c: [usize; 3]| c[0] + dims[0] * (c[1] + dims[1] * c[2]);

    // CSR buckets, members in ascending particle index
    let mut starts = vec![0usize; n_cells + 1];
    for c in &coords {
        starts[flat(*c) + 1] += 1;
    }
    for k in 0..n_cells {
        starts[k + 1] += starts[k];
    }
    let mut cursor = starts.clone();
    let mut members = vec![0usize; positions.len()];
    for (idx, c) in coords.iter().enumerate() {
        let cell = flat(*c);
        members[cursor[cell]] = idx;
        cursor[cell] += 1;
    }

    let mut out = empty_forces(positions.len());
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
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Velocity Verlet
// ---------------------------------------------------------------------------

/// A periodic system of point particles under a truncated pair potential,
/// advanced by velocity Verlet (NVE).
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
