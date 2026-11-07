//! Engine torque curve, gearbox, engine braking and differential.
//!
//! # MODEL — engine torque curve, gearbox and engine braking
//!
//! - **方程式 (equations)**: a quasi-static driveline; `Powertrain` holds
//!   parameters only (l. 99–117). Crank speed from the mean driven-wheel spin
//!   `ω` (rad/s): `rpm_c = |ω| |R| · 60 / 2π` (`coupled_rpm`, l. 218), floored
//!   at `idle_rpm` for the read-out (`engine_rpm`, l. 227–234).
//!   `R = gear_ratios[min(current_gear, len − 1)] · final_drive`, `0` for an
//!   empty table (`total_ratio`, l. 208–213). Full-throttle torque `T(rpm)`:
//!   `0` for `rpm ≥ max_rpm` or an empty table, the first point's torque
//!   below the first point, else linear from the last point with `r_i ≤ rpm`
//!   to the next one, the last point's torque when there is none
//!   (`torque_at`, l. 137–158). With `θ = clamp(throttle, 0, 1)` (l. 257–263):
//!   `τ = θ T(engine_rpm(ω)) R + (1 − θ) b`, `b = −sign(ω) k_b rpm_c |R|`,
//!   `k_b = engine_brake_per_rpm` used as stored (`axle_torque`, l. 264–276).
//!   This module only declares `Differential` (`Open` / `Locked`, l. 90–95);
//!   the torque split or shared spin is applied by the vehicle module.
//!   `from_engine_config` maps the legacy `EngineConfig` to the one-point curve
//!   `(0, max_torque)`, `final_drive = 1`, `Open`, gear `0`, the table cut at `num_gears`,
//!   `k_b = engine_brake · max_torque / max_rpm` (l. 177–201).
//! - **単位 (units)**: every quantity is `Fix128` (l. 83–117): torque Nm, crank
//!   speed rpm (1/min) from rad/s via `60 / 2π` (`coupled_rpm`, l. 218), ratios
//!   dimensionless, `k_b` Nm/rpm; `f64` only in the unit tests' oracles (l. 307).
//! - **保存則 (conservation)**: nothing is stored and there is no crank inertia
//!   (no state field, l. 99–117). The drive part of `τ` supplies the power
//!   `θ T R ω`, the braking part removes `(1 − θ) k_b rpm_c |R| |ω|`
//!   (`axle_torque`, l. 265–276). Drive torque is not limited by engine power:
//!   the idle-floored speed gives `T(idle_rpm) R` at `ω = 0` (l. 265–269).
//! - **適用範囲 (range)**: any `ω`; `rpm_c` uses `|ω|` and `|R|`, so a reverse
//!   spin or gear reads a positive rpm and the drive keeps the sign of `R`
//!   (l. 218, l. 265–269). `rpm_c ≥ 0` and the floor only raises it (l. 218,
//!   l. 229–233), so `max_rpm ≤ 0` gives no drive (`torque_at`, l. 138–140).
//!   Engine braking for `max_rpm ≤ 0` or a negative `k_b` depends on the
//!   path: `from_engine_config` stores `k_b = 0` in both cases (l. 179–188);
//!   a `Powertrain` built directly uses `engine_brake_per_rpm` as given, so a
//!   negative value drives the car with the throttle closed (`axle_torque`,
//!   l. 270–276). An empty gear table is neutral, `R = 0` (l. 210); a
//!   `current_gear` past the table reads as the top gear (l. 211); shifting
//!   clamps to `[0, top]` (`shift_up` / `shift_down`, l. 282–294). An unsorted
//!   table is evaluated on the last point with `r_i ≤ rpm`: a repeated rpm is
//!   a right-continuous step and the divisor is never zero (`torque_at`,
//!   l. 147–157). A throttle outside `[0, 1]` is clamped (l. 257–263). No path
//!   returns an error (l. 137–294).
//! - **近似 (approximations)**: no crank or flywheel inertia and no clutch: the
//!   drive part reads the idle-floored speed, the braking part the coupled one,
//!   so below idle the clutch is taken as slipping (l. 265–270). No torque
//!   converter, shift time, torque interruption, driveline compliance or loss:
//!   `τ` is the curve times `R` (l. 265–269). No limited-slip differential, only
//!   `Open` and `Locked` (l. 90–95). Engine braking is linear in `rpm_c`
//!   (l. 270); part throttle is a linear blend (l. 276).
//! - **検証 (validation)**:
//!   `tests/analytic_vehicle_dynamics.rs::gear_top_speed_converges_to_rev_limit`
//!   (rev limiter: the speed converges to v_n = max_rpm · 2π/60 · r / R_n in
//!   gears 1 and 2), `::launch_within_grip_follows_drive_torque` (axle
//!   torque θ · 300 · 3.5 split equally over the two driven wheels,
//!   a = ΣF / (M + Σ I_w / r²)), `::rolling_resistance_coast_down_is_linear`
//!   (throttle 0 with zero engine braking adds no torque);
//!   `tests/vehicle_dynamics_degenerate.rs::airborne_driven_wheels_spin_up_freely`
//!   (open differential: half the axle torque per driven wheel,
//!   ω = (T · 3.5 / 2) dt / I_w after one frame). The module's own unit
//!   tests (l. 354–781) compare curve, rpm and axle torque to f64 closed
//!   forms (interpolation, rev limit, idle floor, engine braking, throttle
//!   blend and clamp, legacy mapping, degenerate tables, reverse gear, shifting).
//!   validation missing: engine-braking coast-down through the vehicle (every
//!   vehicle-level run sets engine_brake_per_rpm to 0; unit oracles only);
//!   validation missing: a sloped torque curve through the vehicle (every
//!   vehicle-level run pins a flat 300 Nm curve);
//!   validation missing: the locked differential and a reverse gear through
//!   the vehicle (unit oracles only).
//! - **段 (tiers)**: L0 only.

use crate::math::Fix128;
use crate::vehicle::EngineConfig;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

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
    ///
    /// - `rpm ≥ max_rpm` ⇒ `0` (rev limiter; with `max_rpm ≤ 0` every
    ///   non-negative rpm is at the limit)
    /// - no points ⇒ `0` at every rpm
    /// - `rpm` below the first point ⇒ the first point's torque
    /// - `rpm` at or above the last point (and below `max_rpm`) ⇒ the last
    ///   point's torque
    /// - otherwise linear between the last point with `r_i ≤ rpm` and the
    ///   next one: `T_i + (rpm − r_i)(T_{i+1} − T_i)/(r_{i+1} − r_i)`; exact
    ///   on every point
    ///
    /// A repeated rpm is a step and the curve is right-continuous there (the
    /// later point's torque applies at that rpm). Because the segment is
    /// chosen as "last `r_i ≤ rpm`", `r_{i+1} > rpm ≥ r_i` always holds, so
    /// the divisor is never zero even for an unsorted table.
    #[must_use]
    pub fn torque_at(&self, rpm: Fix128, max_rpm: Fix128) -> Fix128 {
        if rpm >= max_rpm {
            return Fix128::ZERO;
        }
        let Some(&(r0, t0)) = self.points.first() else {
            return Fix128::ZERO;
        };
        if rpm < r0 {
            return t0;
        }
        let mut i = 0;
        for (k, &(r, _)) in self.points.iter().enumerate() {
            if r <= rpm {
                i = k;
            }
        }
        let (ri, ti) = self.points[i];
        match self.points.get(i + 1) {
            None => ti,
            Some(&(rn, tn)) => ti + (rpm - ri) * (tn - ti) / (rn - ri),
        }
    }
}

impl Powertrain {
    /// Build from the legacy [`EngineConfig`] + gear table so that its
    /// `max_rpm`, `engine_brake` and `num_gears` are actually read
    /// (flat curve at `max_torque`; `engine_brake` maps to
    /// `engine_brake_per_rpm = engine_brake · max_torque / max_rpm`;
    /// `num_gears` truncates the gear table).
    ///
    /// - `idle_rpm` and `max_rpm` are copied
    /// - `final_drive = 1` (the legacy gear table already includes it),
    ///   `differential = Open`, `current_gear = 0`
    /// - `num_gears` only truncates: a value above the table length keeps
    ///   the whole table, `0` gives an empty table (neutral)
    /// - `max_rpm ≤ 0` ⇒ `engine_brake_per_rpm = 0` (no division by zero, no
    ///   sign flip); a negative result is clamped to `0` so that a closed
    ///   throttle never drives the car
    #[must_use]
    pub fn from_engine_config(engine: &EngineConfig, gear_ratios: &[Fix128]) -> Self {
        let n = engine.num_gears.min(gear_ratios.len());
        let engine_brake_per_rpm = if engine.max_rpm > Fix128::ZERO {
            let k = engine.engine_brake * engine.max_torque / engine.max_rpm;
            if k.is_negative() {
                Fix128::ZERO
            } else {
                k
            }
        } else {
            Fix128::ZERO
        };
        Self {
            curve: TorqueCurve {
                points: vec![(Fix128::ZERO, engine.max_torque)],
            },
            idle_rpm: engine.idle_rpm,
            max_rpm: engine.max_rpm,
            engine_brake_per_rpm,
            gear_ratios: gear_ratios[..n].to_vec(),
            final_drive: Fix128::ONE,
            current_gear: 0,
            differential: Differential::Open,
        }
    }

    /// Total ratio crank → wheel for the current gear (`gear · final_drive`).
    ///
    /// Empty gear table ⇒ `0` (neutral). A `current_gear` past the table
    /// reads as the top gear. A negative gear ratio is a reverse gear.
    #[must_use]
    pub fn total_ratio(&self) -> Fix128 {
        match self.gear_ratios.len() {
            0 => Fix128::ZERO,
            len => self.gear_ratios[self.current_gear.min(len - 1)] * self.final_drive,
        }
    }

    /// Crank speed (rpm) coupled to the wheels, without the idle floor:
    /// `|ω| · |ratio| · 60 / 2π`.
    fn coupled_rpm(&self, wheel_omega: Fix128) -> Fix128 {
        wheel_omega.abs() * self.total_ratio().abs() * Fix128::from_int(60) / Fix128::TWO_PI
    }

    /// Engine speed (rpm) for a mean driven-wheel spin `wheel_omega` (rad/s):
    /// `|ω| · ratio · 60 / 2π`, floored at `idle_rpm`.
    ///
    /// `|ratio|` is used, so a reverse gear (negative ratio) reads a positive
    /// rpm; in neutral (empty table) the engine reads `idle_rpm`.
    #[must_use]
    pub fn engine_rpm(&self, wheel_omega: Fix128) -> Fix128 {
        let rpm = self.coupled_rpm(wheel_omega);
        if rpm < self.idle_rpm {
            self.idle_rpm
        } else {
            rpm
        }
    }

    /// Total torque at the driven axle (Nm, positive = forward drive) for
    /// `throttle ∈ [0, 1]` and mean driven-wheel spin `wheel_omega`:
    /// `throttle · curve(rpm) · ratio` minus, with the throttle closed,
    /// engine braking `engine_brake_per_rpm · rpm · ratio` opposing spin.
    ///
    /// With `θ = clamp(throttle, 0, 1)` and `R = total_ratio()`:
    ///
    /// `τ = θ · curve(engine_rpm(ω), max_rpm) · R
    ///      − (1 − θ) · sign(ω) · engine_brake_per_rpm · rpm_c · |R|`
    ///
    /// - part throttle blends the two linearly: full throttle has no engine
    ///   braking, closed throttle has no drive
    /// - the drive part reads the idle-floored engine speed, so full torque
    ///   is available from rest; it carries the sign of `R` (reverse gear
    ///   drives backwards) and is `0` at or above `max_rpm`
    /// - the braking part uses the wheel-coupled crank speed
    ///   `rpm_c = |ω| · |R| · 60 / 2π` without the idle floor (below idle the
    ///   clutch is taken as slipping), so it is proportional to `ω`, opposes
    ///   the spin and is `0` at `ω = 0`
    #[must_use]
    pub fn axle_torque(&self, throttle: Fix128, wheel_omega: Fix128) -> Fix128 {
        let th = if throttle < Fix128::ZERO {
            Fix128::ZERO
        } else if throttle > Fix128::ONE {
            Fix128::ONE
        } else {
            throttle
        };
        let ratio = self.total_ratio();
        let drive = th
            * self
                .curve
                .torque_at(self.engine_rpm(wheel_omega), self.max_rpm)
            * ratio;
        let brake_mag = self.engine_brake_per_rpm * self.coupled_rpm(wheel_omega) * ratio.abs();
        let brake = if wheel_omega.is_negative() {
            brake_mag
        } else {
            -brake_mag
        };
        drive + (Fix128::ONE - th) * brake
    }

    /// Shift up one gear (clamped).
    ///
    /// Stops at the top gear; an empty table keeps gear `0`.
    pub fn shift_up(&mut self) {
        let top = self.gear_ratios.len().saturating_sub(1);
        self.current_gear = self.current_gear.saturating_add(1).min(top);
    }

    /// Shift down one gear (clamped).
    ///
    /// Stops at gear `0`; a `current_gear` past the table steps down from the
    /// top gear.
    pub fn shift_down(&mut self) {
        let top = self.gear_ratios.len().saturating_sub(1);
        self.current_gear = self.current_gear.min(top).saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    //! Oracles: every expected value is computed here in `f64` from the
    //! closed-form expression quoted next to it, never by calling the
    //! function under test.
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    fn close(got: Fix128, want: f64, tol: f64) -> bool {
        (got.to_f64() - want).abs() <= tol
    }

    /// `(1000, 200) (3000, 300) (5000, 250)`, rev limit 6000.
    fn curve3() -> TorqueCurve {
        TorqueCurve {
            points: vec![
                (Fix128::from_int(1000), Fix128::from_int(200)),
                (Fix128::from_int(3000), Fix128::from_int(300)),
                (Fix128::from_int(5000), Fix128::from_int(250)),
            ],
        }
    }

    /// Hand-built powertrain: flat 200 Nm curve, idle 800, limit 6000,
    /// brake 0.02 Nm/rpm, gears 3.0 / 2.0 / 1.0, final drive 4.0.
    fn pt() -> Powertrain {
        Powertrain {
            curve: TorqueCurve {
                points: vec![(Fix128::ZERO, Fix128::from_int(200))],
            },
            idle_rpm: Fix128::from_int(800),
            max_rpm: Fix128::from_int(6000),
            engine_brake_per_rpm: Fix128::from_ratio(2, 100),
            gear_ratios: vec![
                Fix128::from_int(3),
                Fix128::from_int(2),
                Fix128::from_int(1),
            ],
            final_drive: Fix128::from_int(4),
            current_gear: 0,
            differential: Differential::Open,
        }
    }

    /// oracle: kinematic crank speed `|ω|·R·60/(2π)` in f64.
    fn rpm_f64(omega: f64, ratio: f64) -> f64 {
        omega.abs() * ratio.abs() * 60.0 / (2.0 * core::f64::consts::PI)
    }

    // ---------- 1. interpolation ----------

    #[test]
    fn torque_curve_exact_on_points() {
        let c = curve3();
        let m = Fix128::from_int(6000);
        for (r, t) in [(1000, 200), (3000, 300), (5000, 250)] {
            assert_eq!(c.torque_at(Fix128::from_int(r), m), Fix128::from_int(t));
        }
    }

    #[test]
    fn torque_curve_linear_between_points() {
        let c = curve3();
        let m = Fix128::from_int(6000);
        // oracle: T = T_i + (rpm - r_i)(T_{i+1} - T_i)/(r_{i+1} - r_i)
        for (rpm, want) in [
            (2000.0, 200.0 + 1000.0 * 100.0 / 2000.0),
            (1500.0, 200.0 + 500.0 * 100.0 / 2000.0),
            (4000.0, 300.0 + 1000.0 * (-50.0) / 2000.0),
            (4321.5, 300.0 + 1321.5 * (-50.0) / 2000.0),
        ] {
            let got = c.torque_at(fx(rpm), m);
            assert!(close(got, want, 1e-9), "rpm {rpm}: {got} vs {want}");
        }
    }

    #[test]
    fn torque_curve_flat_outside_points_below_limit() {
        let c = curve3();
        let m = Fix128::from_int(6000);
        // below the first point: held at T_0
        assert_eq!(c.torque_at(Fix128::ZERO, m), Fix128::from_int(200));
        assert_eq!(c.torque_at(Fix128::from_int(-50), m), Fix128::from_int(200));
        assert_eq!(c.torque_at(Fix128::from_int(999), m), Fix128::from_int(200));
        // above the last point but below the limit: held at T_last
        assert_eq!(
            c.torque_at(Fix128::from_int(5999), m),
            Fix128::from_int(250)
        );
    }

    #[test]
    fn torque_curve_zero_at_and_above_rev_limit() {
        let c = curve3();
        let m = Fix128::from_int(6000);
        assert_eq!(c.torque_at(Fix128::from_int(6000), m), Fix128::ZERO);
        assert_eq!(c.torque_at(Fix128::from_int(9000), m), Fix128::ZERO);
        // limit inside the table cuts the table too
        let m2 = Fix128::from_int(2500);
        assert_eq!(c.torque_at(Fix128::from_int(3000), m2), Fix128::ZERO);
        assert!(close(c.torque_at(Fix128::from_int(2000), m2), 250.0, 1e-9));
    }

    #[test]
    fn torque_curve_empty_is_zero() {
        // contract: no points = no torque at any rpm
        let c = TorqueCurve { points: vec![] };
        let m = Fix128::from_int(6000);
        for r in [-100, 0, 1000, 5999, 6000] {
            assert_eq!(c.torque_at(Fix128::from_int(r), m), Fix128::ZERO);
        }
    }

    #[test]
    fn torque_curve_duplicate_rpm_is_right_continuous() {
        // contract: a repeated rpm is a step; at that rpm the later point wins
        let c = TorqueCurve {
            points: vec![
                (Fix128::from_int(1000), Fix128::from_int(100)),
                (Fix128::from_int(1000), Fix128::from_int(200)),
                (Fix128::from_int(2000), Fix128::from_int(300)),
            ],
        };
        let m = Fix128::from_int(6000);
        assert_eq!(c.torque_at(Fix128::from_int(999), m), Fix128::from_int(100));
        assert_eq!(
            c.torque_at(Fix128::from_int(1000), m),
            Fix128::from_int(200)
        );
        assert!(close(c.torque_at(Fix128::from_int(1500), m), 250.0, 1e-9));
    }

    // ---------- 2. rev limiter ⇒ top speed per gear ----------

    #[test]
    fn rev_limiter_caps_speed_per_gear() {
        let r = 0.3_f64; // wheel radius (m), test constant
        let mut p = pt();
        for gear in 0..3 {
            p.current_gear = gear;
            let ratio = [3.0, 2.0, 1.0][gear] * 4.0;
            // oracle: ω_max = max_rpm·2π/60 / R,  v_max = ω_max·r
            let omega_max = 6000.0 * 2.0 * core::f64::consts::PI / 60.0 / ratio;
            let v_max = omega_max * r;
            let v_closed = 6000.0 * 2.0 * core::f64::consts::PI / 60.0 * r / ratio;
            assert!((v_max - v_closed).abs() < 1e-12);

            let below = p.axle_torque(Fix128::ONE, fx(omega_max * (1.0 - 1e-6)));
            let above = p.axle_torque(Fix128::ONE, fx(omega_max * (1.0 + 1e-6)));
            // oracle: below the limit full throttle = 200 · R
            assert!(close(below, 200.0 * ratio, 1e-9), "gear {gear}: {below}");
            assert_eq!(above, Fix128::ZERO, "gear {gear}");
            let rpm_above = p.engine_rpm(fx(omega_max * (1.0 + 1e-6)));
            assert!(rpm_above >= p.max_rpm);
        }
    }

    // ---------- engine speed / ratio ----------

    #[test]
    fn total_ratio_is_gear_times_final_drive() {
        let mut p = pt();
        for (g, want) in [(0, 12), (1, 8), (2, 4)] {
            p.current_gear = g;
            assert_eq!(p.total_ratio(), Fix128::from_int(want));
        }
    }

    #[test]
    fn engine_rpm_kinematic_with_idle_floor() {
        let p = pt(); // R = 12
        for omega in [10.0, 25.0, 40.0] {
            let want = rpm_f64(omega, 12.0).max(800.0);
            assert!(close(p.engine_rpm(fx(omega)), want, 1e-9), "ω {omega}");
        }
        // below idle: floored
        assert_eq!(p.engine_rpm(fx(1.0)), Fix128::from_int(800));
        assert_eq!(p.engine_rpm(Fix128::ZERO), Fix128::from_int(800));
    }

    // ---------- 3. from_engine_config reads every field ----------

    fn legacy_gears() -> Vec<Fix128> {
        vec![
            Fix128::from_ratio(35, 10),
            Fix128::from_ratio(22, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(11, 10),
            Fix128::from_ratio(8, 10),
        ]
    }

    #[test]
    fn from_engine_config_maps_fields() {
        let e = EngineConfig::default(); // 300 Nm, 7000, idle 800, brake 0.5, 5 gears
        let p = Powertrain::from_engine_config(&e, &legacy_gears());
        assert_eq!(p.max_rpm, Fix128::from_int(7000));
        assert_eq!(p.idle_rpm, Fix128::from_int(800));
        assert_eq!(p.final_drive, Fix128::ONE);
        assert_eq!(p.differential, Differential::Open);
        assert_eq!(p.current_gear, 0);
        assert_eq!(p.gear_ratios, legacy_gears());
        // oracle: per_rpm = 0.5 · 300 / 7000
        assert!(close(p.engine_brake_per_rpm, 0.5 * 300.0 / 7000.0, 1e-15));
        // flat curve at max_torque below the limit
        for r in [0, 800, 3500, 6999] {
            assert_eq!(
                p.curve.torque_at(Fix128::from_int(r), p.max_rpm),
                Fix128::from_int(300)
            );
        }
    }

    #[test]
    fn from_engine_config_reads_max_rpm() {
        let a = EngineConfig::default();
        let b = EngineConfig {
            max_rpm: Fix128::from_int(5000),
            ..a
        };
        let pa = Powertrain::from_engine_config(&a, &legacy_gears());
        let pb = Powertrain::from_engine_config(&b, &legacy_gears());
        // first gear R = 3.5: ω at 6000 rpm = 6000·2π/60/3.5
        let omega = 6000.0 * 2.0 * core::f64::consts::PI / 60.0 / 3.5;
        assert!(close(
            pa.axle_torque(Fix128::ONE, fx(omega)),
            300.0 * 3.5,
            1e-9
        ));
        assert_eq!(pb.axle_torque(Fix128::ONE, fx(omega)), Fix128::ZERO);
        // and the engine-brake mapping divides by it: 0.5·300/5000
        assert!(close(pb.engine_brake_per_rpm, 0.5 * 300.0 / 5000.0, 1e-15));
    }

    #[test]
    fn from_engine_config_reads_engine_brake() {
        let a = EngineConfig::default();
        let b = EngineConfig {
            engine_brake: Fix128::from_ratio(2, 10),
            ..a
        };
        let pa = Powertrain::from_engine_config(&a, &legacy_gears());
        let pb = Powertrain::from_engine_config(&b, &legacy_gears());
        let omega = 200.0; // rpm = 200·3.5·60/2π ≈ 6685 > idle
        let rpm = rpm_f64(omega, 3.5);
        // oracle: −(eb·T/max)·rpm·R
        let wa = -(0.5 * 300.0 / 7000.0) * rpm * 3.5;
        let wb = -(0.2 * 300.0 / 7000.0) * rpm * 3.5;
        assert!(close(pa.axle_torque(Fix128::ZERO, fx(omega)), wa, 1e-9));
        assert!(close(pb.axle_torque(Fix128::ZERO, fx(omega)), wb, 1e-9));
    }

    #[test]
    fn from_engine_config_reads_num_gears() {
        let a = EngineConfig::default();
        let b = EngineConfig { num_gears: 3, ..a };
        let pa = Powertrain::from_engine_config(&a, &legacy_gears());
        let mut pb = Powertrain::from_engine_config(&b, &legacy_gears());
        assert_eq!(pa.gear_ratios.len(), 5);
        assert_eq!(pb.gear_ratios, legacy_gears()[..3].to_vec());
        for _ in 0..10 {
            pb.shift_up();
        }
        assert_eq!(pb.current_gear, 2);
        assert_eq!(pb.total_ratio(), Fix128::from_ratio(15, 10));
    }

    #[test]
    fn from_engine_config_reads_idle_rpm() {
        let a = EngineConfig::default();
        let b = EngineConfig {
            idle_rpm: Fix128::from_int(1200),
            ..a
        };
        let pb = Powertrain::from_engine_config(&b, &legacy_gears());
        assert_eq!(pb.engine_rpm(Fix128::ZERO), Fix128::from_int(1200));
    }

    // ---------- 4. engine brake ----------

    #[test]
    fn engine_brake_opposes_spin_and_scales_with_rpm() {
        let p = pt(); // R = 12, k = 0.02
        for omega in [20.0, 40.0, 80.0] {
            let rpm = rpm_f64(omega, 12.0);
            assert!(rpm > 800.0);
            // oracle: −k·rpm·R·sign(ω)
            let want = -0.02 * rpm * 12.0;
            let fwd = p.axle_torque(Fix128::ZERO, fx(omega));
            let back = p.axle_torque(Fix128::ZERO, fx(-omega));
            assert!(close(fwd, want, 1e-9), "ω {omega}: {fwd} vs {want}");
            assert!(close(back, -want, 1e-9), "ω -{omega}: {back}");
            assert!(fwd.is_negative() && !back.is_negative());
        }
        // proportional: doubling ω doubles the brake torque
        let t1 = p.axle_torque(Fix128::ZERO, fx(30.0)).to_f64();
        let t2 = p.axle_torque(Fix128::ZERO, fx(60.0)).to_f64();
        assert!((t2 - 2.0 * t1).abs() < 1e-9);
    }

    #[test]
    fn engine_brake_below_idle_scales_to_zero_at_standstill() {
        // contract: the brake uses the wheel-coupled crank speed, not the
        // idle-floored one, so it vanishes at ω = 0 (no torque from rest)
        let p = pt();
        assert_eq!(p.axle_torque(Fix128::ZERO, Fix128::ZERO), Fix128::ZERO);
        let omega = 2.0; // rpm ≈ 229 < idle 800
        let want = -0.02 * rpm_f64(omega, 12.0) * 12.0;
        assert!(close(p.axle_torque(Fix128::ZERO, fx(omega)), want, 1e-9));
    }

    #[test]
    fn partial_throttle_blends_drive_and_brake() {
        let p = pt();
        let omega = 40.0;
        let rpm = rpm_f64(omega, 12.0);
        for th in [0.25, 0.5, 0.75] {
            // oracle: th·T·R − (1 − th)·k·rpm·R
            let want = th * 200.0 * 12.0 - (1.0 - th) * 0.02 * rpm * 12.0;
            assert!(
                close(p.axle_torque(fx(th), fx(omega)), want, 1e-9),
                "th {th}"
            );
        }
        // full throttle has no brake part
        assert!(close(p.axle_torque(Fix128::ONE, fx(omega)), 2400.0, 1e-9));
    }

    // ---------- 5. degenerate inputs ----------

    #[test]
    fn empty_gear_table_is_neutral() {
        // contract: no gears ⇒ ratio 0, engine at idle, no axle torque,
        // shifting keeps gear 0
        let mut p = pt();
        p.gear_ratios.clear();
        assert_eq!(p.total_ratio(), Fix128::ZERO);
        assert_eq!(p.engine_rpm(fx(50.0)), Fix128::from_int(800));
        for th in [0.0, 0.5, 1.0] {
            assert_eq!(p.axle_torque(fx(th), fx(50.0)), Fix128::ZERO);
        }
        p.shift_up();
        assert_eq!(p.current_gear, 0);
        p.shift_down();
        assert_eq!(p.current_gear, 0);
    }

    #[test]
    fn num_gears_zero_gives_empty_table() {
        let e = EngineConfig {
            num_gears: 0,
            ..EngineConfig::default()
        };
        let p = Powertrain::from_engine_config(&e, &legacy_gears());
        assert!(p.gear_ratios.is_empty());
        assert_eq!(p.total_ratio(), Fix128::ZERO);
        assert_eq!(p.axle_torque(Fix128::ONE, fx(10.0)), Fix128::ZERO);
    }

    #[test]
    fn num_gears_beyond_table_keeps_table() {
        // contract: num_gears only truncates; it never pads
        let e = EngineConfig {
            num_gears: 9,
            ..EngineConfig::default()
        };
        let p = Powertrain::from_engine_config(&e, &legacy_gears()[..2]);
        assert_eq!(p.gear_ratios, legacy_gears()[..2].to_vec());
    }

    #[test]
    fn max_rpm_zero_means_no_drive_and_no_brake() {
        // contract: max_rpm ≤ 0 ⇒ every rpm is at the limit (no drive) and
        // the brake mapping divides by zero ⇒ engine_brake_per_rpm = 0
        let e = EngineConfig {
            max_rpm: Fix128::ZERO,
            ..EngineConfig::default()
        };
        let p = Powertrain::from_engine_config(&e, &legacy_gears());
        assert_eq!(p.engine_brake_per_rpm, Fix128::ZERO);
        for omega in [0.0, 5.0, 100.0] {
            assert_eq!(p.axle_torque(Fix128::ONE, fx(omega)), Fix128::ZERO);
            assert_eq!(p.axle_torque(Fix128::ZERO, fx(omega)), Fix128::ZERO);
        }
        let neg = EngineConfig {
            max_rpm: Fix128::from_int(-7000),
            ..EngineConfig::default()
        };
        let pn = Powertrain::from_engine_config(&neg, &legacy_gears());
        assert_eq!(pn.engine_brake_per_rpm, Fix128::ZERO);
    }

    #[test]
    fn negative_engine_brake_is_clamped_to_zero() {
        // contract: a negative coefficient would drive the car with the
        // throttle closed; from_engine_config clamps it to 0
        let e = EngineConfig {
            engine_brake: Fix128::from_int(-1),
            ..EngineConfig::default()
        };
        let p = Powertrain::from_engine_config(&e, &legacy_gears());
        assert_eq!(p.engine_brake_per_rpm, Fix128::ZERO);
    }

    #[test]
    fn negative_omega_rpm_is_magnitude_and_drive_stays_forward() {
        let p = pt();
        let omega = 40.0;
        assert_eq!(p.engine_rpm(fx(-omega)), p.engine_rpm(fx(omega)));
        // oracle: forward gear, rolling backward, full throttle: +T·R
        assert!(close(p.axle_torque(Fix128::ONE, fx(-omega)), 2400.0, 1e-9));
    }

    #[test]
    fn throttle_out_of_range_is_clamped() {
        let p = pt();
        let omega = fx(40.0);
        assert_eq!(
            p.axle_torque(Fix128::NEG_ONE, omega),
            p.axle_torque(Fix128::ZERO, omega)
        );
        assert_eq!(
            p.axle_torque(Fix128::from_int(2), omega),
            p.axle_torque(Fix128::ONE, omega)
        );
        // and the clamped values are the closed forms
        let rpm = rpm_f64(40.0, 12.0);
        assert!(close(
            p.axle_torque(Fix128::NEG_ONE, omega),
            -0.02 * rpm * 12.0,
            1e-9
        ));
        assert!(close(
            p.axle_torque(Fix128::from_int(2), omega),
            2400.0,
            1e-9
        ));
    }

    #[test]
    fn reverse_gear_negative_ratio() {
        // contract: a negative ratio is a reverse gear: rpm uses |R|, drive
        // torque carries R's sign
        let mut p = pt();
        p.gear_ratios = vec![Fix128::from_int(-3)];
        let omega = 10.0;
        let want_rpm = rpm_f64(omega, 12.0).max(800.0);
        assert!(close(p.engine_rpm(fx(-omega)), want_rpm, 1e-9));
        assert!(close(p.axle_torque(Fix128::ONE, fx(-omega)), -2400.0, 1e-9));
    }

    #[test]
    fn shift_clamps_at_both_ends() {
        let mut p = pt();
        p.shift_down();
        assert_eq!(p.current_gear, 0);
        p.shift_up();
        assert_eq!(p.current_gear, 1);
        p.shift_up();
        p.shift_up();
        p.shift_up();
        assert_eq!(p.current_gear, 2);
        p.shift_down();
        assert_eq!(p.current_gear, 1);
    }

    #[test]
    fn out_of_range_current_gear_reads_as_top_gear() {
        // contract: a current_gear past the table (pub field) acts as the
        // top gear; shift_down steps from the top gear
        let mut p = pt();
        p.current_gear = 10;
        assert_eq!(p.total_ratio(), Fix128::from_int(4));
        p.shift_down();
        assert_eq!(p.current_gear, 1);
        p.current_gear = 10;
        p.shift_up();
        assert_eq!(p.current_gear, 2);
    }
}
