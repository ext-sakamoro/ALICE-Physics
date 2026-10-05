//! Lift and drag of a wing surface attached to a rigid body.
//!
//! A flat lifting surface (a wing, a fin, a tail plane) is described by its
//! planform area `S`, aspect ratio `AR`, Oswald efficiency `e`, section lift
//! slope `a₀`, zero-lift angle `α₀`, stall angle `α_s`, zero-lift drag
//! coefficient `C_D0`, and where it sits on the body (chord and lift axes,
//! centre of pressure).
//!
//! # Relative air flow and angle of attack
//!
//! ```text
//! r      = R · cp_local                      (centre of pressure, world, from the COM)
//! v_cp   = v + ω × r                          (velocity of the surface point)
//! u      = v_cp − wind                        (surface velocity through the air)
//! u_p    = u − (u·ŝ) ŝ                        (span component dropped)
//! α      = atan2(−u_p·n̂, u_p·ĉ)              (angle of attack, in (−π, π])
//! ```
//!
//! with `ĉ` the chord axis (pointing to the leading edge, i.e. "forward"),
//! `n̂` the lift axis (the surface normal on the suction side, "up") and
//! `ŝ = ĉ × n̂` the span axis, all rotated into the world by the body
//! rotation `R`. The span component of the flow does not produce lift or
//! drag in this strip model, and the dynamic pressure uses `|u_p|`.
//!
//! # Lift and drag coefficients
//!
//! The finite-wing lift slope applies the lifting-line correction to the
//! section slope (`a₀ = 2π` for a thin aerofoil, [`THIN_AIRFOIL_LIFT_SLOPE`]):
//!
//! ```text
//! a = a₀ / (1 + a₀ / (π e AR))
//! ```
//!
//! With `αₑ = α − α₀` wrapped into `(−π, π]`, `x = |αₑ|`, `σ = sign(αₑ)`,
//! `α_t = α_s + Δ` (`Δ` = [`LiftDragParams::stall_transition_rad`]) and the
//! induced-drag factor `k = 1 / (π e AR)`:
//!
//! ```text
//! attached,    x ≤ α_s        C_L = a αₑ
//!                              C_D = C_D0 + k C_L²
//! transition,  α_s < x < α_t  t   = (x − α_s) / Δ
//!                              C_L = σ [(1 − t) a α_s + t sin 2α_t]
//!                              C_D = (1 − t)(C_D0 + k (a α_s)²) + t (C_D0 + 2 sin² α_t)
//! separated,   x ≥ α_t        C_L = 2 sin αₑ cos αₑ   (flat plate)
//!                              C_D = C_D0 + 2 sin² αₑ
//! ```
//!
//! `C_L` and `C_D` are continuous in `α` everywhere. [`LiftDragSurface::new`]
//! requires `sin 2α_t < a α_s`, so past the stall angle `C_L` falls from its
//! attached maximum `a α_s` down to the flat-plate value (rising again
//! towards 45° as a flat plate does). Reverse flow (`|αₑ|` near π) uses the
//! flat-plate branch.
//!
//! # Forces
//!
//! ```text
//! q = ½ ρ |u_p|²
//! L = q S C_L · (ŝ × û_p)          (perpendicular to the flow, in the chord–lift plane)
//! D = q S C_D · (−û_p)             (opposite to the surface velocity)
//! F = L + D,  applied at the centre of pressure, τ = r × F about the COM
//! ```
//!
//! [`LiftDragSurface::apply`] adds `F · dt` as an impulse at the centre of
//! pressure ([`RigidBody::apply_impulse_at`]), the same "force × dt" form as
//! [`RigidBody::add_force`]. The air density is an argument, so a caller can
//! pass a value from [`crate::atmosphere::Isa1976`].
//!
//! # Degenerate inputs
//!
//! - `|u_p| = 0` (no flow across the surface): the angle of attack is
//!   undefined, the load is zero and [`AeroLoad::angle_of_attack_rad`] is
//!   `None`.
//! - `ρ = 0`: zero force (the angle of attack is still reported);
//!   `ρ < 0` is [`LiftDragError::NegativeAirDensity`].
//! - `|u_p|` so small that its square is below the `Fix128` resolution
//!   (about `2e-10 m/s`) counts as no flow.
//! - Invalid surface parameters (`S ≤ 0`, `AR ≤ 0`, `e ∉ (0, 1]`, ...) are
//!   rejected by [`LiftDragSurface::new`]; a constructed surface always has
//!   valid parameters because its fields are private.
//!
//! # Use with `PhysicsWorld`
//!
//! `apply` adds the load as one impulse before `PhysicsWorld::step`, the same
//! form as [`RigidBody::add_force`] and the world's force fields, while the
//! world spreads gravity over its substeps. The velocity at frame boundaries
//! is exact (a steady glide settles on the closed-form speed and path), but
//! within a frame the two do not overlap, so the position carries an
//! offset of order `g dt² (s − 1) / (2 s)` per frame (`s` substeps).
//!
//! # Determinism
//!
//! `Fix128` throughout (CORDIC `sin_cos` / `atan2`), available under `no_std`.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

/// Section lift slope of a thin aerofoil, `2π` per radian.
pub const THIN_AIRFOIL_LIFT_SLOPE: Fix128 = Fix128::TWO_PI;

/// Parameters of a [`LiftDragSurface`]. Validated by [`LiftDragSurface::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiftDragParams {
    /// Planform area `S` (m²), `> 0`.
    pub area_m2: Fix128,
    /// Aspect ratio `AR = b² / S`, `> 0`.
    pub aspect_ratio: Fix128,
    /// Oswald span efficiency `e`, in `(0, 1]`.
    pub oswald_efficiency: Fix128,
    /// Section (2-D) lift slope `a₀` (per rad), `> 0`; [`THIN_AIRFOIL_LIFT_SLOPE`] for a thin aerofoil.
    pub section_lift_slope_per_rad: Fix128,
    /// Zero-lift angle of attack `α₀` (rad); `0` for a symmetric section, negative for positive camber.
    pub zero_lift_angle_rad: Fix128,
    /// Stall angle `α_s` (rad), measured from `α₀`; in `(0, π/2)`.
    pub stall_angle_rad: Fix128,
    /// Width `Δ` (rad) of the transition from attached to flat-plate flow; `> 0` and `α_s + Δ < π/2`.
    pub stall_transition_rad: Fix128,
    /// Zero-lift drag coefficient `C_D0`, `≥ 0`.
    pub zero_lift_drag_coefficient: Fix128,
    /// Chord axis `ĉ` in body coordinates, pointing to the leading edge. Normalised by `new`.
    pub chord_axis_local: Vec3Fix,
    /// Lift axis `n̂` in body coordinates (suction-side normal). Made orthogonal to `ĉ` and normalised by `new`.
    pub lift_axis_local: Vec3Fix,
    /// Centre of pressure in body coordinates, relative to the centre of mass (m).
    pub center_of_pressure_local: Vec3Fix,
}

/// Why a [`LiftDragSurface`] could not be built or evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiftDragError {
    /// `area_m2 ≤ 0`.
    NonPositiveArea,
    /// `aspect_ratio ≤ 0`.
    NonPositiveAspectRatio,
    /// `oswald_efficiency` outside `(0, 1]`.
    OswaldEfficiencyOutOfRange,
    /// `section_lift_slope_per_rad ≤ 0`.
    NonPositiveLiftSlope,
    /// `stall_angle_rad` outside `(0, π/2)`.
    StallAngleOutOfRange,
    /// `stall_transition_rad ≤ 0`, or `stall_angle_rad + stall_transition_rad ≥ π/2`.
    StallTransitionOutOfRange,
    /// `zero_lift_drag_coefficient < 0`.
    NegativeZeroLiftDrag,
    /// The chord axis is zero, or the lift axis is zero or parallel to it.
    DegenerateAxes,
    /// `sin 2(α_s + Δ) ≥ a α_s`: the lift would not fall past the stall angle.
    NoLiftLossAtStall,
    /// Air density `< 0`.
    NegativeAirDensity,
}

impl core::fmt::Display for LiftDragError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::NonPositiveArea => "area must be positive",
            Self::NonPositiveAspectRatio => "aspect ratio must be positive",
            Self::OswaldEfficiencyOutOfRange => "Oswald efficiency must be in (0, 1]",
            Self::NonPositiveLiftSlope => "section lift slope must be positive",
            Self::StallAngleOutOfRange => "stall angle must be in (0, pi/2)",
            Self::StallTransitionOutOfRange => {
                "stall transition must be positive and end before pi/2"
            }
            Self::NegativeZeroLiftDrag => "zero-lift drag coefficient must not be negative",
            Self::DegenerateAxes => "chord and lift axes must be non-zero and not parallel",
            Self::NoLiftLossAtStall => "lift must fall past the stall angle",
            Self::NegativeAirDensity => "air density must not be negative",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LiftDragError {}

/// Lift and drag coefficients at one angle of attack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AeroCoefficients {
    /// Lift coefficient `C_L`.
    pub lift: Fix128,
    /// Drag coefficient `C_D`.
    pub drag: Fix128,
}

/// Aerodynamic load of a surface on its body at one instant (world frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AeroLoad {
    /// Total force `L + D` (N).
    pub force: Vec3Fix,
    /// Lift component of `force` (N).
    pub lift: Vec3Fix,
    /// Drag component of `force` (N).
    pub drag: Vec3Fix,
    /// Moment of `force` about the centre of mass, `r × F` (N·m).
    pub torque: Vec3Fix,
    /// World position of the centre of pressure, where `force` acts.
    pub application_point: Vec3Fix,
    /// Angle of attack `α` (rad), `None` when there is no flow across the surface.
    pub angle_of_attack_rad: Option<Fix128>,
    /// Airspeed across the surface, `|u_p|` (m/s).
    pub airspeed_m_s: Fix128,
}

/// A validated lifting surface. See the [module documentation](self) for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiftDragSurface {
    params: LiftDragParams,
    /// Unit chord axis `ĉ` (body frame).
    chord: Vec3Fix,
    /// Unit lift axis `n̂`, orthogonal to `ĉ` (body frame).
    lift_axis: Vec3Fix,
    /// Finite-wing slope `a`.
    lift_slope: Fix128,
    /// `k = 1 / (π e AR)`.
    induced_drag_factor: Fix128,
    /// `α_t = α_s + Δ`.
    separated_angle: Fix128,
    /// `C_L`, `C_D` at `α_s` (attached branch).
    stall_coefficients: AeroCoefficients,
    /// `C_L`, `C_D` at `α_t` (flat-plate branch).
    separated_coefficients: AeroCoefficients,
}

/// `x` wrapped into `[−π, π)` (the flat-plate and attached branches are
/// both `π`-periodic or odd, so the closed/open end does not matter).
fn wrap_angle(x: Fix128) -> Fix128 {
    let turns = ((x + Fix128::PI) / Fix128::TWO_PI).floor();
    x - turns * Fix128::TWO_PI
}

/// Flat-plate coefficients `(2 sin α cos α, C_D0 + 2 sin² α)`.
fn flat_plate(alpha: Fix128, cd0: Fix128) -> AeroCoefficients {
    let (sin, cos) = alpha.sin_cos();
    AeroCoefficients {
        lift: (sin * cos).double(),
        drag: cd0 + (sin * sin).double(),
    }
}

impl LiftDragSurface {
    /// Validates `params` and builds the surface.
    ///
    /// # Errors
    ///
    /// The [`LiftDragError`] variant naming the first invalid parameter.
    pub fn new(params: LiftDragParams) -> Result<Self, LiftDragError> {
        let p = params;
        if p.area_m2 <= Fix128::ZERO {
            return Err(LiftDragError::NonPositiveArea);
        }
        if p.aspect_ratio <= Fix128::ZERO {
            return Err(LiftDragError::NonPositiveAspectRatio);
        }
        if p.oswald_efficiency <= Fix128::ZERO || p.oswald_efficiency > Fix128::ONE {
            return Err(LiftDragError::OswaldEfficiencyOutOfRange);
        }
        if p.section_lift_slope_per_rad <= Fix128::ZERO {
            return Err(LiftDragError::NonPositiveLiftSlope);
        }
        if p.stall_angle_rad <= Fix128::ZERO || p.stall_angle_rad >= Fix128::HALF_PI {
            return Err(LiftDragError::StallAngleOutOfRange);
        }
        let separated_angle = p.stall_angle_rad + p.stall_transition_rad;
        if p.stall_transition_rad <= Fix128::ZERO || separated_angle >= Fix128::HALF_PI {
            return Err(LiftDragError::StallTransitionOutOfRange);
        }
        if p.zero_lift_drag_coefficient.is_negative() {
            return Err(LiftDragError::NegativeZeroLiftDrag);
        }
        let chord = p
            .chord_axis_local
            .try_normalize()
            .ok_or(LiftDragError::DegenerateAxes)?;
        // Gram-Schmidt: keep the part of the lift axis orthogonal to the chord.
        // A remainder below 1e-6 of the given length counts as parallel.
        let given = p.lift_axis_local.length();
        let normal = p.lift_axis_local - chord * p.lift_axis_local.dot(chord);
        let normal_len = normal.length();
        if given.is_zero() || normal_len <= given * Fix128::from_ratio(1, 1_000_000) {
            return Err(LiftDragError::DegenerateAxes);
        }
        let lift_axis = normal / normal_len;

        let pi_e_ar = Fix128::PI * p.oswald_efficiency * p.aspect_ratio;
        let a0 = p.section_lift_slope_per_rad;
        let lift_slope = a0 / (Fix128::ONE + a0 / pi_e_ar);
        let induced_drag_factor = Fix128::ONE / pi_e_ar;
        let stall_lift = lift_slope * p.stall_angle_rad;
        let stall_coefficients = AeroCoefficients {
            lift: stall_lift,
            drag: p.zero_lift_drag_coefficient + induced_drag_factor * stall_lift * stall_lift,
        };
        let separated_coefficients = flat_plate(separated_angle, p.zero_lift_drag_coefficient);
        if separated_coefficients.lift >= stall_lift {
            return Err(LiftDragError::NoLiftLossAtStall);
        }
        Ok(Self {
            params,
            chord,
            lift_axis,
            lift_slope,
            induced_drag_factor,
            separated_angle,
            stall_coefficients,
            separated_coefficients,
        })
    }

    /// The parameters the surface was built from (axes as given, not normalised).
    #[must_use]
    pub fn params(&self) -> &LiftDragParams {
        &self.params
    }

    /// Finite-wing lift slope `a = a₀ / (1 + a₀ / (π e AR))` (per rad).
    #[must_use]
    pub fn lift_slope_per_rad(&self) -> Fix128 {
        self.lift_slope
    }

    /// `C_L` and `C_D` at angle of attack `α` (rad, any value; wrapped into `(−π, π]`).
    #[must_use]
    pub fn coefficients(&self, angle_of_attack_rad: Fix128) -> AeroCoefficients {
        let p = &self.params;
        let effective = wrap_angle(angle_of_attack_rad - p.zero_lift_angle_rad);
        let x = effective.abs();
        let negative = effective.is_negative();
        if x <= p.stall_angle_rad {
            let lift = self.lift_slope * effective;
            return AeroCoefficients {
                lift,
                drag: p.zero_lift_drag_coefficient + self.induced_drag_factor * lift * lift,
            };
        }
        if x < self.separated_angle {
            let t = (x - p.stall_angle_rad) / p.stall_transition_rad;
            let keep = Fix128::ONE - t;
            let (st, sep) = (self.stall_coefficients, self.separated_coefficients);
            let lift = keep * st.lift + t * sep.lift;
            return AeroCoefficients {
                lift: if negative { -lift } else { lift },
                drag: keep * st.drag + t * sep.drag,
            };
        }
        flat_plate(effective, p.zero_lift_drag_coefficient)
    }

    /// Load on `body` from a uniform `wind` (m/s, world) in air of density
    /// `air_density_kg_m3`.
    ///
    /// # Errors
    ///
    /// [`LiftDragError::NegativeAirDensity`] when the density is negative.
    pub fn load(
        &self,
        body: &RigidBody,
        wind: Vec3Fix,
        air_density_kg_m3: Fix128,
    ) -> Result<AeroLoad, LiftDragError> {
        if air_density_kg_m3.is_negative() {
            return Err(LiftDragError::NegativeAirDensity);
        }
        let rot = body.rotation;
        let r = rot.rotate_vec(self.params.center_of_pressure_local);
        let application_point = body.position + r;
        let chord = rot.rotate_vec(self.chord);
        let normal = rot.rotate_vec(self.lift_axis);
        let span = chord.cross(normal);
        let u = body.velocity + body.angular_velocity.cross(r) - wind;
        let u_p = u - span * u.dot(span);
        let (u_hat, airspeed_m_s) = u_p.normalize_with_length();
        if airspeed_m_s.is_zero() {
            return Ok(AeroLoad {
                force: Vec3Fix::ZERO,
                lift: Vec3Fix::ZERO,
                drag: Vec3Fix::ZERO,
                torque: Vec3Fix::ZERO,
                application_point,
                angle_of_attack_rad: None,
                airspeed_m_s,
            });
        }
        let alpha = Fix128::atan2(-u_p.dot(normal), u_p.dot(chord));
        let c = self.coefficients(alpha);
        let q_s = (air_density_kg_m3 * airspeed_m_s * airspeed_m_s).half() * self.params.area_m2;
        let lift = span.cross(u_hat) * (q_s * c.lift);
        let drag = -u_hat * (q_s * c.drag);
        let force = lift + drag;
        Ok(AeroLoad {
            force,
            lift,
            drag,
            torque: r.cross(force),
            application_point,
            angle_of_attack_rad: Some(alpha),
            airspeed_m_s,
        })
    }

    /// Computes [`Self::load`] and applies `force · dt` to `body` at the
    /// centre of pressure. Returns the load that was applied.
    ///
    /// # Errors
    ///
    /// As [`Self::load`]; the body is unchanged on error.
    pub fn apply(
        &self,
        body: &mut RigidBody,
        wind: Vec3Fix,
        air_density_kg_m3: Fix128,
        dt: Fix128,
    ) -> Result<AeroLoad, LiftDragError> {
        let load = self.load(body, wind, air_density_kg_m3)?;
        body.apply_impulse_at(load.force * dt, load.application_point);
        Ok(load)
    }
}
