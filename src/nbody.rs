//! N-body gravity: direct-sum mutual attraction and a symplectic integrator.
//!
//! Every body attracts every other with Newton's law, optionally Plummer
//! softened:
//!
//! ```text
//! aᵢ = Σ_{j≠i} G·mⱼ·(xⱼ − xᵢ) / (|xⱼ − xᵢ|² + ε²)^{3/2}
//! U  = −Σ_{i<j} G·mᵢ·mⱼ / √(|xⱼ − xᵢ|² + ε²)
//! ```
//!
//! [`DirectSum`] evaluates this over all `N(N−1)/2` pairs — **`O(N²)` per
//! evaluation**, no tree or multipole approximation, so it is exact up to
//! fixed-point rounding and meant for tens to a few thousand bodies.
//! [`VelocityVerlet`] integrates it with the kick–drift–kick leapfrog
//! (symplectic and time-reversible, second order: the energy error stays
//! bounded instead of drifting, and halving `dt` divides it by about 4).
//! [`DirectSum::step_world`] runs the same scheme on the bodies of a
//! [`PhysicsWorld`].
//!
//! This module is the many-body law. The two-body closed forms (Kepler's
//! equation, orbital elements, vis-viva) are [`crate::kepler`]; one fixed
//! attractor acting on world bodies is [`crate::force::ForceField::Point`].
//!
//! # Softening
//!
//! `ε > 0` replaces the point mass by a Plummer sphere: the force is finite
//! everywhere and zero at coincidence, and `U` above is its exact potential,
//! so the softened system conserves energy too. With `ε = 0` a pair of
//! bodies at exactly the same position has no direction; that pair is
//! skipped (contributes no acceleration and no potential) instead of
// LIMITATION(COV-ORBIT-041): Close encounters with `ε = 0` are otherwise exact and need a `dt` that resolves them.
//! producing an infinite value. Close encounters with `ε = 0` are otherwise
//! exact and need a `dt` that resolves them.
//!
//! # Determinism
//!
//! All arithmetic is [`Fix128`]. Pairs are visited in the fixed order
//! `(0,1), (0,2), …, (1,2), …`, each pair's force is computed once and added
//! to one body and subtracted from the other, and [`Fix128`] addition is
//! exact integer addition (associative), so the accumulated accelerations do
//! not depend on the summation order at all. The same inputs give the same
//! bits on every platform.
//!
//! Mirror and rotation symmetry hold only up to rounding: a [`Fix128`] product
//! rounds towards −∞, so `(−a)·b` and `−(a·b)` can differ by one unit of
//! `2⁻⁶⁴`, and a mirrored or rotated input can give results that differ in
//! the last bits.
//! For the same reason the total momentum `Σ m v` is conserved to rounding,
//! not bit for bit.
//!
//! # Range
//!
//! `1/(d² + ε²)^{3/2}` must stay below about `9·10¹⁸`: with `ε = 0` that is
//! a separation above `≈ 10⁻⁶` length units. Choose units (or `ε`) so that
//! close approaches stay above that.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::{BodyType, PhysicsWorld, RigidBody};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Why an N-body function refused its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NBodyError {
    /// The gravitational constant `G` is not strictly positive.
    NonPositiveGravitationalConstant,
    /// The softening length `ε` is negative.
    NegativeSoftening,
    /// A mass is negative (zero is allowed: a test particle that feels the
    /// others but pulls on nothing).
    NegativeMass,
    /// The position, velocity and mass slices do not have the same length.
    LengthMismatch,
}

impl core::fmt::Display for NBodyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::NonPositiveGravitationalConstant => "the gravitational constant must be positive",
            Self::NegativeSoftening => "the softening length must not be negative",
            Self::NegativeMass => "a mass must not be negative",
            Self::LengthMismatch => "positions, velocities and masses must have the same length",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NBodyError {}

fn check_masses(masses: &[Fix128]) -> Result<(), NBodyError> {
    if masses.iter().any(|m| m.is_negative()) {
        Err(NBodyError::NegativeMass)
    } else {
        Ok(())
    }
}

/// Direct-sum mutual gravity with gravitational constant `G` and Plummer
/// softening length `ε` (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectSum {
    gravitational_constant: Fix128,
    softening: Fix128,
}

impl DirectSum {
    /// The law with constant `G > 0` and softening `ε ≥ 0`, in the caller's
    /// units (this crate carries no value of `G`).
    ///
    /// # Errors
    ///
    /// [`NBodyError::NonPositiveGravitationalConstant`] for `G ≤ 0`,
    /// [`NBodyError::NegativeSoftening`] for `ε < 0`.
    pub fn new(gravitational_constant: Fix128, softening: Fix128) -> Result<Self, NBodyError> {
        if gravitational_constant <= Fix128::ZERO {
            return Err(NBodyError::NonPositiveGravitationalConstant);
        }
        if softening.is_negative() {
            return Err(NBodyError::NegativeSoftening);
        }
        Ok(Self {
            gravitational_constant,
            softening,
        })
    }

    /// The gravitational constant `G`.
    #[must_use]
    pub fn gravitational_constant(&self) -> Fix128 {
        self.gravitational_constant
    }

    /// The softening length `ε`.
    #[must_use]
    pub fn softening(&self) -> Fix128 {
        self.softening
    }

    /// Accelerations of all bodies, written to `out` (resized to `N`;
    /// previous contents are discarded). `O(N²)`.
    ///
    /// # Errors
    ///
    /// [`NBodyError::LengthMismatch`] if `positions` and `masses` differ in
    /// length, [`NBodyError::NegativeMass`] for a negative mass; `out` is not
    /// touched on error.
    pub fn accelerations(
        &self,
        positions: &[Vec3Fix],
        masses: &[Fix128],
        out: &mut Vec<Vec3Fix>,
    ) -> Result<(), NBodyError> {
        if positions.len() != masses.len() {
            return Err(NBodyError::LengthMismatch);
        }
        check_masses(masses)?;
        let n = positions.len();
        out.clear();
        out.resize(n, Vec3Fix::ZERO);
        let eps2 = self.softening * self.softening;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = positions[j] - positions[i];
                let r2 = d.length_squared() + eps2;
                if r2.is_zero() {
                    continue;
                }
                let inv_r = Fix128::ONE / r2.sqrt();
                let pull = d * (self.gravitational_constant * inv_r * inv_r * inv_r);
                out[i] = out[i] + pull * masses[j];
                out[j] = out[j] - pull * masses[i];
            }
        }
        Ok(())
    }

    /// Total (softened) potential energy `U = −Σ_{i<j} G·mᵢ·mⱼ/√(d² + ε²)`.
    /// `O(N²)`.
    ///
    /// # Errors
    ///
    /// As [`Self::accelerations`].
    pub fn potential_energy(
        &self,
        positions: &[Vec3Fix],
        masses: &[Fix128],
    ) -> Result<Fix128, NBodyError> {
        if positions.len() != masses.len() {
            return Err(NBodyError::LengthMismatch);
        }
        check_masses(masses)?;
        let eps2 = self.softening * self.softening;
        let mut total = Fix128::ZERO;
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let r2 = (positions[j] - positions[i]).length_squared() + eps2;
                if r2.is_zero() {
                    continue;
                }
                total = total - self.gravitational_constant * masses[i] * masses[j] / r2.sqrt();
            }
        }
        Ok(total)
    }

    /// Add `aᵢ·dt` to the velocity of every **dynamic** body of `bodies`
    // LIMITATION(COV-ORBIT-032): Static and kinematic bodies neither feel nor exert the attraction (they have no finite mass)
    /// (`inv_mass > 0`, mass `1/inv_mass`). Static and kinematic bodies
    /// neither feel nor exert the attraction (they have no finite mass); to
    /// pull bodies toward a fixed centre use
    /// [`crate::force::ForceField::Point`]. Positions are not changed.
    ///
    /// This is one kick of the integrator. Called once before
    /// [`PhysicsWorld::step`] with the whole `dt` (the way
    /// [`PhysicsWorld::force_fields`] enter the step, at the start of the
    /// frame) it gives symplectic Euler, which is only first order; use
    /// [`Self::step_world`] for the second-order scheme.
    pub fn kick_bodies(&self, bodies: &mut [RigidBody], dt: Fix128) {
        let _ = self.kick_dynamic(bodies, dt);
    }

    /// Kick the dynamic bodies; returns how many took part.
    fn kick_dynamic(&self, bodies: &mut [RigidBody], dt: Fix128) -> usize {
        let index: Vec<usize> = (0..bodies.len())
            .filter(|&i| {
                bodies[i].body_type == BodyType::Dynamic && bodies[i].inv_mass > Fix128::ZERO
            })
            .collect();
        if index.len() < 2 {
            return index.len();
        }
        let positions: Vec<Vec3Fix> = index.iter().map(|&i| bodies[i].position).collect();
        let masses: Vec<Fix128> = index
            .iter()
            .map(|&i| Fix128::ONE / bodies[i].inv_mass)
            .collect();
        let mut acc = Vec::new();
        // Lengths agree and every mass is positive by construction.
        if self.accelerations(&positions, &masses, &mut acc).is_err() {
            return 0;
        }
        for (k, &i) in index.iter().enumerate() {
            bodies[i].velocity = bodies[i].velocity + acc[k] * dt;
        }
        index.len()
    }

    /// One frame of the kick–drift–kick leapfrog on the world's bodies:
    /// a half kick of `dt/2` ([`Self::kick_bodies`]), [`PhysicsWorld::step`]
    /// as the drift, and a second half kick at the new positions. Every body
    /// that takes part is woken first, so a slow body is not left asleep
    /// while its velocity changes.
    ///
    /// The result is the same scheme as [`VelocityVerlet`] **when the step is
    /// a pure drift**: set `config.gravity` to zero (or keep it if the uniform
    /// field is wanted, it then enters at the frame head as a first-order
    /// term) and `config.damping` and each body's `linear_damping` to one
    /// (frame damping scales the velocity and breaks the symplectic map).
    /// Contacts, joints and [`PhysicsWorld::force_fields`] still act inside
    /// the step as usual. Two `O(N²)` evaluations per frame.
    ///
    /// Non-positive `dt` does nothing (as `step`).
    pub fn step_world(&self, world: &mut PhysicsWorld, dt: Fix128) {
        if dt <= Fix128::ZERO {
            return;
        }
        let half = dt.half();
        if self.kick_dynamic(&mut world.bodies, half) >= 2 {
            for i in 0..world.bodies.len() {
                if world.bodies[i].body_type == BodyType::Dynamic {
                    world.wake_body(i);
                }
            }
        }
        world.step(dt);
        let _ = self.kick_dynamic(&mut world.bodies, half);
    }
}

/// Total kinetic energy `Σ mᵢ·|vᵢ|²/2`.
///
/// # Errors
///
/// [`NBodyError::LengthMismatch`], [`NBodyError::NegativeMass`].
pub fn kinetic_energy(velocities: &[Vec3Fix], masses: &[Fix128]) -> Result<Fix128, NBodyError> {
    if velocities.len() != masses.len() {
        return Err(NBodyError::LengthMismatch);
    }
    check_masses(masses)?;
    Ok(velocities
        .iter()
        .zip(masses)
        .fold(Fix128::ZERO, |acc, (v, &m)| {
            acc + (m * v.length_squared()).half()
        }))
}

/// Total linear momentum `Σ mᵢ·vᵢ`.
///
/// # Errors
///
/// [`NBodyError::LengthMismatch`], [`NBodyError::NegativeMass`].
pub fn total_momentum(velocities: &[Vec3Fix], masses: &[Fix128]) -> Result<Vec3Fix, NBodyError> {
    if velocities.len() != masses.len() {
        return Err(NBodyError::LengthMismatch);
    }
    check_masses(masses)?;
    Ok(velocities
        .iter()
        .zip(masses)
        .fold(Vec3Fix::ZERO, |acc, (&v, &m)| acc + v * m))
}

/// Velocity-Verlet (kick–drift–kick leapfrog) integrator for a
/// [`DirectSum`] system:
///
/// ```text
/// v ← v + a(x)·dt/2;   x ← x + v·dt;   v ← v + a(x)·dt/2
/// ```
///
/// Symplectic, time-reversible and second order. The acceleration at the end
/// of a step is kept and reused at the start of the next one (one `O(N²)`
/// evaluation per step) as long as the positions, masses and law are those
/// it was computed for; anything else is detected and recomputed.
///
/// Not the same type as [`crate::molecular_dynamics::VelocityVerlet`]: this
/// one does not own the system (the caller passes positions, velocities and
/// masses to every `step`; it keeps only the acceleration cache), while that
/// one is `VelocityVerlet<P: PairPotential>` and owns a periodic particle
/// system that it steps. Neither is re-exported at the crate root; a
/// re-export must use a distinct alias.
#[derive(Clone, Debug, Default)]
pub struct VelocityVerlet {
    acc: Vec<Vec3Fix>,
    cached_positions: Vec<Vec3Fix>,
    cached_masses: Vec<Fix128>,
    cached_law: Option<DirectSum>,
}

impl VelocityVerlet {
    /// An integrator with an empty acceleration cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance `positions` and `velocities` by `dt` under `law`.
    ///
    /// # Errors
    ///
    /// [`NBodyError::LengthMismatch`] if the three slices differ in length,
    /// [`NBodyError::NegativeMass`]; nothing is modified on error.
    pub fn step(
        &mut self,
        law: &DirectSum,
        positions: &mut [Vec3Fix],
        velocities: &mut [Vec3Fix],
        masses: &[Fix128],
        dt: Fix128,
    ) -> Result<(), NBodyError> {
        if positions.len() != velocities.len() || positions.len() != masses.len() {
            return Err(NBodyError::LengthMismatch);
        }
        check_masses(masses)?;
        let fresh = self.cached_law == Some(*law)
            && self.cached_positions.as_slice() == &*positions
            && self.cached_masses.as_slice() == masses;
        if !fresh {
            law.accelerations(positions, masses, &mut self.acc)?;
        }
        let half = dt.half();
        for ((x, v), a) in positions
            .iter_mut()
            .zip(velocities.iter_mut())
            .zip(&self.acc)
        {
            *v = *v + *a * half;
            *x = *x + *v * dt;
        }
        law.accelerations(positions, masses, &mut self.acc)?;
        for (v, a) in velocities.iter_mut().zip(&self.acc) {
            *v = *v + *a * half;
        }
        self.cached_positions.clear();
        self.cached_positions.extend_from_slice(positions);
        self.cached_masses.clear();
        self.cached_masses.extend_from_slice(masses);
        self.cached_law = Some(*law);
        Ok(())
    }
}
