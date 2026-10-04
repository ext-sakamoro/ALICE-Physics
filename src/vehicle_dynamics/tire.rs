//! Tyre force models: slip in, contact-patch force out.
//!
//! # Conventions (shared with [`super::DynamicVehicle`])
//!
//! - Wheel frame: `x` = wheel heading projected onto the ground plane,
//!   `y` = `normal × x` (points to the wheel's right when the normal is up).
//! - `v_x`, `v_y`: contact-point velocity components in that frame.
//! - Slip ratio `κ = (ω r − v_x) / max(|v_x|, v_floor)`: positive while
//!   driving (wheel surface faster than the road), `−1` for a locked wheel
//!   rolling forward.
//! - Lateral slip `tan α = v_y / max(|v_x|, v_floor)`: positive when the
//!   contact point slides to the right. The lateral force opposes it.
//! - Output force components are in the same frame: `longitudinal` along `x`,
//!   `lateral` along `y`, in newtons.
//!
//! Every model saturates on the friction ellipse with semi-axes
//! `μ_x F_z` and `μ_y F_z` taken from [`TireInput::grip`].

use crate::anisotropic_friction::AnisotropicFriction;
use crate::math::Fix128;

/// Slip state and load of one contact patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireInput {
    /// Slip ratio `κ` (dimensionless, see module conventions).
    pub slip_ratio: Fix128,
    /// Lateral slip `tan α` (dimensionless, see module conventions).
    pub slip_tan_alpha: Fix128,
    /// Normal load `F_z` (N). Non-positive load gives a zero force.
    pub normal_load: Fix128,
    /// Road grip at this contact (already scaled by surface and weather).
    /// `*_static` is the peak (adhesion) coefficient, `*_kinetic` the
    /// full-sliding coefficient.
    pub grip: AnisotropicFriction,
}

/// Contact-patch force in the wheel frame (N).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireForce {
    /// Force along the wheel heading.
    pub longitudinal: Fix128,
    /// Force along the wheel's lateral axis.
    pub lateral: Fix128,
}

/// Brush (Fiala-type) model with combined slip.
///
/// Contract:
/// - slope at zero slip: `∂F_x/∂κ = longitudinal_stiffness`,
///   `∂F_y/∂tan α = −cornering_stiffness` (both independent of `F_z`)
/// - full sliding (locked wheel, `κ = −1`, `tan α = 0`): `F_x = −μ_x,kinetic F_z`
///   exactly
/// - `|F|` never leaves the friction ellipse built from the static coefficients
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushTire {
    /// `C_κ` (N per unit slip ratio).
    pub longitudinal_stiffness: Fix128,
    /// `C_α` (N per unit `tan α`).
    pub cornering_stiffness: Fix128,
}

/// Pacejka Magic Formula (pure-slip curves, combined by the friction ellipse).
///
/// `F = D sin(C atan(B s − E (B s − atan(B s))))` with `D = μ_static F_z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MagicFormulaTire {
    /// Longitudinal stiffness factor `B_x`.
    pub b_x: Fix128,
    /// Longitudinal shape factor `C_x`.
    pub c_x: Fix128,
    /// Longitudinal curvature factor `E_x`.
    pub e_x: Fix128,
    /// Lateral stiffness factor `B_y` (input is `α` in radians).
    pub b_y: Fix128,
    /// Lateral shape factor `C_y`.
    pub c_y: Fix128,
    /// Lateral curvature factor `E_y`.
    pub e_y: Fix128,
}

/// Selectable tyre model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireModel {
    /// Brush / Fiala model.
    Brush(BrushTire),
    /// Pacejka Magic Formula.
    MagicFormula(MagicFormulaTire),
}

impl BrushTire {
    /// Passenger-car tyre: `C_κ = 80 kN`, `C_α = 60 kN`.
    #[must_use]
    pub fn passenger_car() -> Self {
        todo!("STUB: BrushTire::passenger_car")
    }
}

impl MagicFormulaTire {
    /// Commonly quoted passenger-car coefficients
    /// (`B_x 10, C_x 1.9, E_x 0.97, B_y 8, C_y 1.3, E_y −1`).
    #[must_use]
    pub fn passenger_car() -> Self {
        todo!("STUB: MagicFormulaTire::passenger_car")
    }
}

impl TireModel {
    /// Contact-patch force for one wheel.
    #[must_use]
    pub fn force(&self, input: &TireInput) -> TireForce {
        let _ = input;
        todo!("STUB: TireModel::force")
    }

    /// Small-slip longitudinal slope `∂F_x/∂κ` at load `normal_load` and grip
    /// `mu_peak` (used by the vehicle's implicit wheel-spin update).
    #[must_use]
    pub fn longitudinal_slope(&self, normal_load: Fix128, mu_peak: Fix128) -> Fix128 {
        let _ = (normal_load, mu_peak);
        todo!("STUB: TireModel::longitudinal_slope")
    }
}
