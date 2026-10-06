//! Rotor and propeller thrust, torque and ideal hover power.
//!
//! # Static thrust model
//!
//! ```text
//! T = C_T ρ n² D⁴        (thrust, N, along the rotor axis)
//! Q = C_Q ρ n² D⁵        (shaft torque, N·m)
//! ```
//!
//! with `n` the rotation speed in revolutions per second, `D` the diameter
//! and `C_T`, `C_Q` the (dimensionless) thrust and torque coefficients.
// LIMITATION(COV-MBD-115): The coefficients are constants: this is a **static thrust model**.
//! The coefficients are constants: this is a **static thrust model**. The
//! fall of thrust with axial inflow (advance ratio `J = V / (n D)`) and with
//! forward flight is not modelled, so the values are those of a rotor in
//! still air.
//!
//! # Direction of the torque on the body
//!
//! [`RotorSpin`] gives the sense of rotation about the thrust axis `â` by
//! the right-hand rule. The air resists the rotor with a torque `−s Q â`
//! and the motor turns the rotor with `+s Q â`; its reaction on the body
//! that carries the motor is
//!
//! ```text
//! τ_reaction = −s Q â,    s = +1 (RightHanded), −1 (LeftHanded)
//! ```
//!
//! so a body held up by a single right-handed rotor spins left-handed.
//! [`Rotor::load`] adds the moment of the thrust about the centre of mass,
//! `r × T â`, for a hub off the centre of mass.
//!
//! # Momentum theory (hover)
//!
//! ```text
//! v_i = √(T / (2 ρ A))     (induced velocity at the disk)
//! P   = T v_i              (ideal induced power)
//! ```
//!
//! [`hover_induced_velocity_m_s`] and [`ideal_hover_power_w`] take the
//! thrust and the disk area `A = π D² / 4` ([`Rotor::disk_area_m2`]).
//!
//! # Degenerate inputs
//!
//! `n = 0` or `ρ = 0` give zero thrust and torque. A negative `n`, `ρ` or
//! thrust, and a non-positive disk area, are errors ([`RotorError`]): the
//! direction of rotation is [`RotorSpin`], not the sign of `n`.
//!
//! # Use with `PhysicsWorld`
//!
//! `apply` adds the load as one impulse before `PhysicsWorld::step`, the same
//! form as [`RigidBody::add_force`] and the world's force fields, while the
//! world spreads gravity over its substeps. The velocity at frame boundaries
//! is exact, but within a frame the two do not overlap: a body held at
// LIMITATION(COV-MBD-132): exactly `T = m g` rises by `g dt² (s − 1) / (2 s)` per frame
//! exactly `T = m g` rises by `g dt² (s − 1) / (2 s)` per frame (`s`
//! substeps), about 0.73 m in 10 s at 60 Hz with 8 substeps.
//!
//! # Determinism
//!
//! `Fix128` throughout, available under `no_std`.
//!
//! Mirror and rotation symmetry hold only up to rounding: a `Fix128` product
//! rounds towards −∞, so `(−a)·b` and `−(a·b)` can differ by one unit of
//! `2⁻⁶⁴`, and a mirrored or rotated input can give results that differ in
//! the last bits.
//! For example the reaction torques of the two senses of rotation are
//! negatives of each other only to within one unit.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

/// Sense of rotation of a rotor about its thrust axis (right-hand rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorSpin {
    /// Angular velocity along `+â`.
    RightHanded,
    /// Angular velocity along `−â`.
    LeftHanded,
}

/// Parameters of a [`Rotor`]. Validated by [`Rotor::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotorParams {
    /// Rotor diameter `D` (m), `> 0`.
    pub diameter_m: Fix128,
    /// Thrust coefficient `C_T`, `≥ 0`.
    pub thrust_coefficient: Fix128,
    /// Torque coefficient `C_Q`, `≥ 0`.
    pub torque_coefficient: Fix128,
    /// Thrust axis `â` in body coordinates. Normalised by `new`.
    pub thrust_axis_local: Vec3Fix,
    /// Hub position in body coordinates, relative to the centre of mass (m).
    pub hub_position_local: Vec3Fix,
    /// Sense of rotation about `â`.
    pub spin: RotorSpin,
}

/// Why a rotor quantity could not be built or evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorError {
    /// `diameter_m ≤ 0`.
    NonPositiveDiameter,
    /// `thrust_coefficient < 0`.
    NegativeThrustCoefficient,
    /// `torque_coefficient < 0`.
    NegativeTorqueCoefficient,
    /// The thrust axis is the zero vector.
    DegenerateAxis,
    /// Rotation speed `n < 0` (the sense of rotation is [`RotorSpin`]).
    NegativeRotationSpeed,
    /// Air density `< 0`.
    NegativeAirDensity,
    /// Thrust `< 0` passed to a momentum-theory function.
    NegativeThrust,
    /// Disk area `≤ 0` passed to a momentum-theory function.
    NonPositiveDiskArea,
}

impl core::fmt::Display for RotorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::NonPositiveDiameter => "rotor diameter must be positive",
            Self::NegativeThrustCoefficient => "thrust coefficient must not be negative",
            Self::NegativeTorqueCoefficient => "torque coefficient must not be negative",
            Self::DegenerateAxis => "thrust axis must be non-zero",
            Self::NegativeRotationSpeed => "rotation speed must not be negative",
            Self::NegativeAirDensity => "air density must not be negative",
            Self::NegativeThrust => "thrust must not be negative",
            Self::NonPositiveDiskArea => "disk area must be positive",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RotorError {}

/// Load of a rotor on its body at one instant (world frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotorLoad {
    /// Thrust magnitude `T` (N).
    pub thrust_n: Fix128,
    /// Shaft torque magnitude `Q` (N·m).
    pub shaft_torque_nm: Fix128,
    /// Thrust force `T â` (N).
    pub force: Vec3Fix,
    /// Reaction torque on the body, `−s Q â` (N·m).
    pub reaction_torque: Vec3Fix,
    /// Total moment about the centre of mass, `r × T â + τ_reaction` (N·m).
    pub torque: Vec3Fix,
    /// World position of the hub, where `force` acts.
    pub application_point: Vec3Fix,
}

/// A validated rotor. See the [module documentation](self) for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rotor {
    params: RotorParams,
    axis: Vec3Fix,
}

impl Rotor {
    /// Validates `params` and builds the rotor.
    ///
    /// # Errors
    ///
    /// The [`RotorError`] variant naming the first invalid parameter.
    pub fn new(params: RotorParams) -> Result<Self, RotorError> {
        if params.diameter_m <= Fix128::ZERO {
            return Err(RotorError::NonPositiveDiameter);
        }
        if params.thrust_coefficient.is_negative() {
            return Err(RotorError::NegativeThrustCoefficient);
        }
        if params.torque_coefficient.is_negative() {
            return Err(RotorError::NegativeTorqueCoefficient);
        }
        let axis = params
            .thrust_axis_local
            .try_normalize()
            .ok_or(RotorError::DegenerateAxis)?;
        Ok(Self { params, axis })
    }

    /// `ρ n²` after the sign checks shared by thrust and torque.
    fn rho_n2(air_density_kg_m3: Fix128, rev_per_s: Fix128) -> Result<Fix128, RotorError> {
        if air_density_kg_m3.is_negative() {
            return Err(RotorError::NegativeAirDensity);
        }
        if rev_per_s.is_negative() {
            return Err(RotorError::NegativeRotationSpeed);
        }
        Ok(air_density_kg_m3 * rev_per_s * rev_per_s)
    }

    /// `D⁴`.
    fn d4(&self) -> Fix128 {
        let d2 = self.params.diameter_m * self.params.diameter_m;
        d2 * d2
    }

    /// The parameters the rotor was built from (axis as given, not normalised).
    #[must_use]
    pub fn params(&self) -> &RotorParams {
        &self.params
    }

    /// Disk area `A = π D² / 4` (m²).
    #[must_use]
    pub fn disk_area_m2(&self) -> Fix128 {
        let d = self.params.diameter_m;
        Fix128::PI * d * d / Fix128::from_int(4)
    }

    /// Thrust `T = C_T ρ n² D⁴` (N) at `n` revolutions per second.
    ///
    /// # Errors
    ///
    /// [`RotorError::NegativeAirDensity`] / [`RotorError::NegativeRotationSpeed`].
    pub fn thrust_n(
        &self,
        air_density_kg_m3: Fix128,
        rev_per_s: Fix128,
    ) -> Result<Fix128, RotorError> {
        Ok(
            self.params.thrust_coefficient
                * Self::rho_n2(air_density_kg_m3, rev_per_s)?
                * self.d4(),
        )
    }

    /// Shaft torque `Q = C_Q ρ n² D⁵` (N·m) at `n` revolutions per second.
    ///
    /// # Errors
    ///
    /// As [`Self::thrust_n`].
    pub fn shaft_torque_nm(
        &self,
        air_density_kg_m3: Fix128,
        rev_per_s: Fix128,
    ) -> Result<Fix128, RotorError> {
        Ok(self.params.torque_coefficient
            * Self::rho_n2(air_density_kg_m3, rev_per_s)?
            * self.d4()
            * self.params.diameter_m)
    }

    /// Rotation speed `n = √(T / (C_T ρ D⁴))` (rev/s) that gives `thrust_n`.
    /// `C_T = 0` or `ρ = 0` with a positive target returns `ZERO` (no speed
    /// produces thrust; the `Fix128` division convention).
    ///
    /// # Errors
    ///
    /// [`RotorError::NegativeAirDensity`] / [`RotorError::NegativeThrust`].
    pub fn speed_for_thrust(
        &self,
        thrust_n: Fix128,
        air_density_kg_m3: Fix128,
    ) -> Result<Fix128, RotorError> {
        if air_density_kg_m3.is_negative() {
            return Err(RotorError::NegativeAirDensity);
        }
        if thrust_n.is_negative() {
            return Err(RotorError::NegativeThrust);
        }
        Ok((thrust_n / (self.params.thrust_coefficient * air_density_kg_m3 * self.d4())).sqrt())
    }

    /// Load on `body` at `n` revolutions per second in air of density `ρ`.
    ///
    /// # Errors
    ///
    /// As [`Self::thrust_n`].
    pub fn load(
        &self,
        body: &RigidBody,
        air_density_kg_m3: Fix128,
        rev_per_s: Fix128,
    ) -> Result<RotorLoad, RotorError> {
        let thrust_n = self.thrust_n(air_density_kg_m3, rev_per_s)?;
        let shaft_torque_nm = self.shaft_torque_nm(air_density_kg_m3, rev_per_s)?;
        let axis = body.rotation.rotate_vec(self.axis);
        let r = body.rotation.rotate_vec(self.params.hub_position_local);
        let force = axis * thrust_n;
        let reaction_torque = match self.params.spin {
            RotorSpin::RightHanded => -axis * shaft_torque_nm,
            RotorSpin::LeftHanded => axis * shaft_torque_nm,
        };
        Ok(RotorLoad {
            thrust_n,
            shaft_torque_nm,
            force,
            reaction_torque,
            torque: r.cross(force) + reaction_torque,
            application_point: body.position + r,
        })
    }

    /// Computes [`Self::load`] and applies it for `dt`: the thrust as an
    /// impulse `T â dt` at the hub ([`RigidBody::apply_impulse_at`], which
    /// carries `r × T â`) and the reaction torque with
    /// [`RigidBody::add_torque`]. Returns the load that was applied.
    ///
    /// # Errors
    ///
    /// As [`Self::load`]; the body is unchanged on error.
    pub fn apply(
        &self,
        body: &mut RigidBody,
        air_density_kg_m3: Fix128,
        rev_per_s: Fix128,
        dt: Fix128,
    ) -> Result<RotorLoad, RotorError> {
        let load = self.load(body, air_density_kg_m3, rev_per_s)?;
        body.apply_impulse_at(load.force * dt, load.application_point);
        body.add_torque(load.reaction_torque, dt);
        Ok(load)
    }
}

/// Induced velocity at the disk in hover, `v_i = √(T / (2 ρ A))` (m/s).
/// `ρ = 0` with a positive thrust returns `ZERO` (the `Fix128` division
/// convention: the momentum-theory velocity is unbounded).
///
/// # Errors
///
/// [`RotorError::NegativeThrust`] / [`RotorError::NegativeAirDensity`] /
/// [`RotorError::NonPositiveDiskArea`].
pub fn hover_induced_velocity_m_s(
    thrust_n: Fix128,
    air_density_kg_m3: Fix128,
    disk_area_m2: Fix128,
) -> Result<Fix128, RotorError> {
    if thrust_n.is_negative() {
        return Err(RotorError::NegativeThrust);
    }
    if air_density_kg_m3.is_negative() {
        return Err(RotorError::NegativeAirDensity);
    }
    if disk_area_m2 <= Fix128::ZERO {
        return Err(RotorError::NonPositiveDiskArea);
    }
    Ok((thrust_n / (air_density_kg_m3 * disk_area_m2).double()).sqrt())
}

/// Ideal induced power in hover, `P = T v_i` (W).
///
/// # Errors
///
/// As [`hover_induced_velocity_m_s`].
pub fn ideal_hover_power_w(
    thrust_n: Fix128,
    air_density_kg_m3: Fix128,
    disk_area_m2: Fix128,
) -> Result<Fix128, RotorError> {
    Ok(thrust_n * hover_induced_velocity_m_s(thrust_n, air_density_kg_m3, disk_area_m2)?)
}
