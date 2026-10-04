//! Engine torque curve, gearbox, engine braking and differential.

use crate::math::Fix128;
use crate::vehicle::EngineConfig;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Full-throttle torque curve: `(rpm, torque Nm)` points, linearly
/// interpolated, held flat below the first point, zero above `max_rpm`
/// (rev limiter).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorqueCurve {
    /// Points sorted by rpm.
    pub points: Vec<(Fix128, Fix128)>,
}

/// How the driven axle splits torque between its wheels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Differential {
    /// Equal torque to both wheels, speeds free.
    Open,
    /// Both wheels forced to the same speed (the vehicle couples their spin).
    Locked,
}

/// Engine + gearbox + final drive + differential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Powertrain {
    /// Full-throttle torque curve.
    pub curve: TorqueCurve,
    /// Idle speed (rpm); the engine speed never reads below it.
    pub idle_rpm: Fix128,
    /// Rev limit (rpm); no drive torque at or above it.
    pub max_rpm: Fix128,
    /// Engine-braking torque at the crank per rpm with the throttle closed
    /// (Nm/rpm), opposing rotation.
    pub engine_brake_per_rpm: Fix128,
    /// Gear ratios, first gear first.
    pub gear_ratios: Vec<Fix128>,
    /// Final-drive ratio.
    pub final_drive: Fix128,
    /// Current gear index.
    pub current_gear: usize,
    /// Differential type.
    pub differential: Differential,
}

impl TorqueCurve {
    /// Torque at `rpm` (Nm) per the struct contract.
    #[must_use]
    pub fn torque_at(&self, rpm: Fix128, max_rpm: Fix128) -> Fix128 {
        let _ = (rpm, max_rpm);
        todo!("STUB: TorqueCurve::torque_at")
    }
}

impl Powertrain {
    /// Build from the legacy [`EngineConfig`] + gear table so that its
    /// `max_rpm`, `engine_brake` and `num_gears` are actually read
    /// (flat curve at `max_torque`; `engine_brake` maps to
    /// `engine_brake_per_rpm = engine_brake · max_torque / max_rpm`;
    /// `num_gears` truncates the gear table).
    #[must_use]
    pub fn from_engine_config(engine: &EngineConfig, gear_ratios: &[Fix128]) -> Self {
        let _ = (engine, gear_ratios);
        todo!("STUB: Powertrain::from_engine_config")
    }

    /// Total ratio crank → wheel for the current gear (`gear · final_drive`).
    #[must_use]
    pub fn total_ratio(&self) -> Fix128 {
        todo!("STUB: Powertrain::total_ratio")
    }

    /// Engine speed (rpm) for a mean driven-wheel spin `wheel_omega` (rad/s):
    /// `|ω| · ratio · 60 / 2π`, floored at `idle_rpm`.
    #[must_use]
    pub fn engine_rpm(&self, wheel_omega: Fix128) -> Fix128 {
        let _ = wheel_omega;
        todo!("STUB: Powertrain::engine_rpm")
    }

    /// Total torque at the driven axle (Nm, positive = forward drive) for
    /// `throttle ∈ [0, 1]` and mean driven-wheel spin `wheel_omega`:
    /// `throttle · curve(rpm) · ratio` minus, with the throttle closed,
    /// engine braking `engine_brake_per_rpm · rpm · ratio` opposing spin.
    #[must_use]
    pub fn axle_torque(&self, throttle: Fix128, wheel_omega: Fix128) -> Fix128 {
        let _ = (throttle, wheel_omega);
        todo!("STUB: Powertrain::axle_torque")
    }

    /// Shift up one gear (clamped).
    pub fn shift_up(&mut self) {
        todo!("STUB: Powertrain::shift_up")
    }

    /// Shift down one gear (clamped).
    pub fn shift_down(&mut self) {
        todo!("STUB: Powertrain::shift_down")
    }
}
