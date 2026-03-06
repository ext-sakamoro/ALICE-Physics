//! Vehicle Physics
//!
//! Wheel + suspension + engine model for ground vehicles.
//! Supports heightfield terrain and SDF surface driving.
//!
//! # Components
//!
//! - **Wheel**: Raycast-based ground contact, slip model
//! - **Suspension**: Spring-damper for each wheel
//! - **Engine**: Torque curve, gear ratios
//! - **Steering**: Ackermann geometry
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Vehicle Configuration
// ============================================================================

/// Wheel configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelConfig {
    /// Suspension attachment point (local to chassis)
    pub local_position: Vec3Fix,
    /// Suspension rest length
    pub suspension_rest: Fix128,
    /// Suspension spring stiffness
    pub spring_stiffness: Fix128,
    /// Suspension damping coefficient
    pub damping: Fix128,
    /// Wheel radius
    pub radius: Fix128,
    /// Maximum steering angle (radians, 0 for rear wheels)
    pub max_steer_angle: Fix128,
    /// Whether this wheel is driven (receives engine torque)
    pub driven: bool,
    /// Whether this wheel has brakes
    pub has_brake: bool,
    /// Progressive spring rate coefficient (0 = linear, >0 = stiffness increases with compression)
    pub progressive_rate: Fix128,
    /// Bump damping coefficient (compression direction).
    /// 0 = use `damping` for both directions (symmetric).
    pub bump_damping: Fix128,
    /// Rebound damping coefficient (extension direction).
    /// 0 = use `damping` for both directions (symmetric).
    pub rebound_damping: Fix128,
    /// Bump stop stiffness (N/m). Very stiff spring activated when compression exceeds `bump_stop_threshold`.
    /// 0 = no bump stop.
    pub bump_stop_stiffness: Fix128,
    /// Bump stop activation threshold (0..1, fraction of full compression).
    /// Default 0.9 = bump stop activates at 90% compression.
    pub bump_stop_threshold: Fix128,
}

impl Default for WheelConfig {
    fn default() -> Self {
        Self {
            local_position: Vec3Fix::ZERO,
            suspension_rest: Fix128::from_ratio(3, 10),
            spring_stiffness: Fix128::from_int(50000),
            damping: Fix128::from_int(4500),
            radius: Fix128::from_ratio(3, 10),
            max_steer_angle: Fix128::from_ratio(5, 10), // ~28.6 degrees
            driven: false,
            has_brake: true,
            progressive_rate: Fix128::ZERO,
            bump_damping: Fix128::ZERO,
            rebound_damping: Fix128::ZERO,
            bump_stop_stiffness: Fix128::ZERO,
            bump_stop_threshold: Fix128::from_ratio(9, 10), // 0.9
        }
    }
}

/// Engine configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    /// Maximum engine torque (Nm)
    pub max_torque: Fix128,
    /// Maximum engine RPM
    pub max_rpm: Fix128,
    /// Idle RPM
    pub idle_rpm: Fix128,
    /// Engine braking coefficient
    pub engine_brake: Fix128,
    /// Number of gears
    pub num_gears: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_torque: Fix128::from_int(300),
            max_rpm: Fix128::from_int(7000),
            idle_rpm: Fix128::from_int(800),
            engine_brake: Fix128::from_ratio(5, 10),
            num_gears: 5,
        }
    }
}

/// Vehicle configuration
#[derive(Clone, Debug)]
pub struct VehicleConfig {
    /// Wheel configurations
    pub wheels: Vec<WheelConfig>,
    /// Engine configuration
    pub engine: EngineConfig,
    /// Gear ratios (including final drive)
    pub gear_ratios: Vec<Fix128>,
    /// Brake force (N)
    pub brake_force: Fix128,
    /// Aerodynamic drag coefficient
    pub aero_drag: Fix128,
    /// Downforce coefficient
    pub downforce: Fix128,
    /// Ground plane height (Y coordinate)
    pub ground_height: Fix128,
    /// Anti-roll bar stiffness (N/m)。左右ホイール圧縮差に比例するスタビライザー力。
    /// 0 = 無効。
    pub anti_roll_stiffness: Fix128,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            wheels: vec![
                WheelConfig {
                    local_position: Vec3Fix::from_f32(-0.8, -0.2, 1.2),
                    driven: false,
                    max_steer_angle: Fix128::from_ratio(5, 10),
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(0.8, -0.2, 1.2),
                    driven: false,
                    max_steer_angle: Fix128::from_ratio(5, 10),
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(-0.8, -0.2, -1.2),
                    driven: true,
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(0.8, -0.2, -1.2),
                    driven: true,
                    ..Default::default()
                },
            ],
            engine: EngineConfig::default(),
            gear_ratios: vec![
                Fix128::from_ratio(35, 10), // 1st
                Fix128::from_ratio(22, 10), // 2nd
                Fix128::from_ratio(15, 10), // 3rd
                Fix128::from_ratio(11, 10), // 4th
                Fix128::from_ratio(8, 10),  // 5th
            ],
            brake_force: Fix128::from_int(10000),
            aero_drag: Fix128::from_ratio(4, 10),
            downforce: Fix128::from_ratio(1, 10),
            ground_height: Fix128::ZERO,
            anti_roll_stiffness: Fix128::from_int(5000),
        }
    }
}

// ============================================================================
// Wheel State
// ============================================================================

/// Runtime wheel state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelState {
    /// Current suspension compression (0 = fully extended, 1 = bottomed out)
    pub compression: Fix128,
    /// Suspension force applied this frame
    pub suspension_force: Fix128,
    /// Whether wheel is touching ground
    pub grounded: bool,
    /// Ground contact point (world space)
    pub contact_point: Vec3Fix,
    /// Ground normal at contact
    pub ground_normal: Vec3Fix,
    /// Current wheel rotation angle (radians)
    pub spin_angle: Fix128,
    /// Wheel angular velocity (rad/s)
    pub spin_speed: Fix128,
    /// Lateral slip ratio
    pub lateral_slip: Fix128,
    /// Longitudinal slip ratio
    pub longitudinal_slip: Fix128,
}

impl Default for WheelState {
    fn default() -> Self {
        Self {
            compression: Fix128::ZERO,
            suspension_force: Fix128::ZERO,
            grounded: false,
            contact_point: Vec3Fix::ZERO,
            ground_normal: Vec3Fix::UNIT_Y,
            spin_angle: Fix128::ZERO,
            spin_speed: Fix128::ZERO,
            lateral_slip: Fix128::ZERO,
            longitudinal_slip: Fix128::ZERO,
        }
    }
}

// ============================================================================
// Vehicle
// ============================================================================

/// Vehicle physics simulation
pub struct Vehicle {
    /// Vehicle configuration
    pub config: VehicleConfig,
    /// Per-wheel runtime state
    pub wheel_states: Vec<WheelState>,
    /// Current gear (0 = first gear)
    pub current_gear: usize,
    /// Current engine RPM
    pub engine_rpm: Fix128,
    /// Throttle input (0..1)
    pub throttle: Fix128,
    /// Brake input (0..1)
    pub brake: Fix128,
    /// Steering input (-1..1)
    pub steering: Fix128,
    /// Forward speed (km/h)
    pub speed_kmh: Fix128,
}

impl Vehicle {
    /// Create a new vehicle
    #[must_use]
    pub fn new(config: VehicleConfig) -> Self {
        let n = config.wheels.len();
        Self {
            wheel_states: vec![WheelState::default(); n],
            current_gear: 0,
            engine_rpm: config.engine.idle_rpm,
            throttle: Fix128::ZERO,
            brake: Fix128::ZERO,
            steering: Fix128::ZERO,
            speed_kmh: Fix128::ZERO,
            config,
        }
    }

    /// Create default 4-wheel vehicle
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(VehicleConfig::default())
    }

    /// Update vehicle physics
    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, chassis: &mut RigidBody, dt: Fix128) {
        if chassis.is_static() {
            return;
        }

        let forward = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Z);
        let right = chassis.rotation.rotate_vec(Vec3Fix::UNIT_X);
        let up = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Y);

        // Compute forward speed
        self.speed_kmh = chassis.velocity.dot(forward) * Fix128::from_ratio(36, 10);

        // Engine torque
        let gear_ratio = if self.current_gear < self.config.gear_ratios.len() {
            self.config.gear_ratios[self.current_gear]
        } else {
            Fix128::ONE
        };
        let engine_torque = self.config.engine.max_torque * self.throttle;
        let wheel_torque = engine_torque * gear_ratio;

        // Process each wheel
        let num_driven = self.config.wheels.iter().filter(|w| w.driven).count() as i64;
        let torque_per_wheel = if num_driven > 0 {
            wheel_torque / Fix128::from_int(num_driven)
        } else {
            Fix128::ZERO
        };

        let mut total_force = Vec3Fix::ZERO;

        for i in 0..self.config.wheels.len() {
            let wheel_cfg = self.config.wheels[i];

            // World-space wheel position
            let wheel_world =
                chassis.position + chassis.rotation.rotate_vec(wheel_cfg.local_position);

            // Raycast down for ground contact
            let ray_start = wheel_world;
            let max_dist = wheel_cfg.suspension_rest + wheel_cfg.radius;

            // Ground plane check (configurable height)
            let ground_y = self.config.ground_height;
            let dist_to_ground = ray_start.y - ground_y;

            if dist_to_ground < max_dist && dist_to_ground > Fix128::ZERO {
                self.wheel_states[i].grounded = true;
                self.wheel_states[i].ground_normal = Vec3Fix::UNIT_Y;
                self.wheel_states[i].contact_point =
                    Vec3Fix::new(wheel_world.x, ground_y, wheel_world.z);

                // Suspension
                let compression =
                    Fix128::ONE - (dist_to_ground - wheel_cfg.radius) / wheel_cfg.suspension_rest;
                let compression = if compression < Fix128::ZERO {
                    Fix128::ZERO
                } else if compression > Fix128::ONE {
                    Fix128::ONE
                } else {
                    compression
                };

                // Progressive spring: stiffness increases with compression
                let effective_stiffness = wheel_cfg.spring_stiffness
                    * (Fix128::ONE + wheel_cfg.progressive_rate * compression);
                let mut spring_force = effective_stiffness * compression;

                // Bump stop: very stiff spring at high compression
                if !wheel_cfg.bump_stop_stiffness.is_zero()
                    && compression > wheel_cfg.bump_stop_threshold
                {
                    let overshoot = compression - wheel_cfg.bump_stop_threshold;
                    spring_force = spring_force + wheel_cfg.bump_stop_stiffness * overshoot;
                }

                // Asymmetric damping: bump (compression) vs rebound (extension)
                let vel_y = chassis.velocity.dot(up);
                let damp_coeff = if vel_y < Fix128::ZERO {
                    // Moving downward = compressing suspension = bump
                    if wheel_cfg.bump_damping.is_zero() {
                        wheel_cfg.damping
                    } else {
                        wheel_cfg.bump_damping
                    }
                } else {
                    // Moving upward = extending suspension = rebound
                    if wheel_cfg.rebound_damping.is_zero() {
                        wheel_cfg.damping
                    } else {
                        wheel_cfg.rebound_damping
                    }
                };
                let damp_force = damp_coeff * vel_y;
                let susp_force = spring_force - damp_force;

                self.wheel_states[i].compression = compression;
                self.wheel_states[i].suspension_force = susp_force;

                total_force = total_force + up * susp_force;

                // Driving force
                if wheel_cfg.driven {
                    let drive_force = forward * (torque_per_wheel / wheel_cfg.radius);
                    total_force = total_force + drive_force;
                }

                // Braking
                if wheel_cfg.has_brake && self.brake > Fix128::ZERO {
                    let speed = chassis.velocity.dot(forward);
                    if speed.abs() > Fix128::from_ratio(1, 10) {
                        let brake_dir = if speed > Fix128::ZERO {
                            -forward
                        } else {
                            forward
                        };
                        let brake_mag = self.config.brake_force * self.brake;
                        total_force = total_force + brake_dir * brake_mag;
                    }
                }

                // Lateral friction (resist sliding)
                let lateral_vel = chassis.velocity.dot(right);
                let lateral_friction = -right * (lateral_vel * Fix128::from_int(5000));
                total_force = total_force + lateral_friction;

                // Steering
                if wheel_cfg.max_steer_angle > Fix128::ZERO {
                    let steer_angle = self.steering * wheel_cfg.max_steer_angle;
                    let (sin_a, cos_a) = steer_angle.sin_cos();
                    let steer_force = (right * sin_a + forward * cos_a) - forward;
                    let speed_factor = chassis.velocity.dot(forward).abs();
                    total_force =
                        total_force + steer_force * (speed_factor * Fix128::from_int(1000));
                }
            } else {
                self.wheel_states[i].grounded = false;
                self.wheel_states[i].compression = Fix128::ZERO;
            }

            // Update wheel spin
            if self.wheel_states[i].grounded {
                let ground_speed = chassis.velocity.dot(forward);
                self.wheel_states[i].spin_speed = ground_speed / wheel_cfg.radius;
            }
            self.wheel_states[i].spin_angle =
                self.wheel_states[i].spin_angle + self.wheel_states[i].spin_speed * dt;
        }

        // Anti-roll bar: 左右ホイール圧縮差に比例する復元力。
        // ホイールをペアで処理（0-1がフロント、2-3がリア等）。
        if !self.config.anti_roll_stiffness.is_zero() && self.config.wheels.len() >= 2 {
            let num_pairs = self.config.wheels.len() / 2;
            for pair in 0..num_pairs {
                let left = pair * 2;
                let right = pair * 2 + 1;
                let diff =
                    self.wheel_states[left].compression - self.wheel_states[right].compression;
                let anti_roll_force = self.config.anti_roll_stiffness * diff;
                // 左を押し下げ、右を押し上げ（差が正→左が沈んでいる）
                if self.wheel_states[left].grounded {
                    total_force = total_force - up * anti_roll_force;
                }
                if self.wheel_states[right].grounded {
                    total_force = total_force + up * anti_roll_force;
                }
            }
        }

        // Aerodynamic drag
        let speed_sq = chassis.velocity.length_squared();
        if !speed_sq.is_zero() {
            let speed = speed_sq.sqrt();
            let drag_dir = chassis.velocity / speed;
            total_force = total_force - drag_dir * (self.config.aero_drag * speed_sq);
            // Downforce
            total_force = total_force - up * (self.config.downforce * speed_sq);
        }

        // Apply forces (impulse = force * dt)
        chassis.apply_impulse(total_force * dt);

        // Update engine RPM
        let wheel_rpm = self.speed_kmh.abs() * Fix128::from_ratio(100, 36) * gear_ratio;
        self.engine_rpm = wheel_rpm.abs();
        if self.engine_rpm < self.config.engine.idle_rpm {
            self.engine_rpm = self.config.engine.idle_rpm;
        }
    }

    /// Shift gear up
    pub fn shift_up(&mut self) {
        if self.current_gear + 1 < self.config.gear_ratios.len() {
            self.current_gear += 1;
        }
    }

    /// Shift gear down
    pub fn shift_down(&mut self) {
        if self.current_gear > 0 {
            self.current_gear -= 1;
        }
    }

    /// Number of grounded wheels
    #[must_use]
    pub fn grounded_wheels(&self) -> usize {
        self.wheel_states.iter().filter(|w| w.grounded).count()
    }
}

impl core::fmt::Debug for Vehicle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vehicle")
            .field("config", &self.config)
            .field(
                "wheel_states",
                &format_args!("[{} items]", self.wheel_states.len()),
            )
            .field("current_gear", &self.current_gear)
            .field("engine_rpm", &self.engine_rpm)
            .field("throttle", &self.throttle)
            .field("brake", &self.brake)
            .field("steering", &self.steering)
            .field("speed_kmh", &self.speed_kmh)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_creation() {
        let vehicle = Vehicle::new_default();
        assert_eq!(vehicle.config.wheels.len(), 4);
        assert_eq!(vehicle.wheel_states.len(), 4);
    }

    #[test]
    fn test_vehicle_suspension() {
        let mut vehicle = Vehicle::new_default();
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // Some wheels should be grounded
        assert!(vehicle.grounded_wheels() > 0, "Wheels should touch ground");
    }

    #[test]
    fn test_vehicle_acceleration() {
        let mut vehicle = Vehicle::new_default();
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        vehicle.throttle = Fix128::ONE; // Full throttle

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            vehicle.update(&mut chassis, dt);
        }

        // Vehicle should have gained forward velocity
        let speed = vehicle.speed_kmh.to_f32();
        assert!(
            speed.abs() > 0.01,
            "Vehicle should accelerate, speed={speed}"
        );
    }

    #[test]
    fn test_gear_shifting() {
        let mut vehicle = Vehicle::new_default();
        assert_eq!(vehicle.current_gear, 0);

        vehicle.shift_up();
        assert_eq!(vehicle.current_gear, 1);

        vehicle.shift_up();
        vehicle.shift_up();
        vehicle.shift_up();
        vehicle.shift_up(); // Should cap at max gear
        assert_eq!(vehicle.current_gear, 4);

        vehicle.shift_down();
        assert_eq!(vehicle.current_gear, 3);
    }

    #[test]
    fn test_progressive_spring() {
        // progressive_rate > 0 → 圧縮が深いほどばねが硬くなる
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.progressive_rate = Fix128::ONE; // stiffness doubles at full compression
        }
        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // 圧縮が0以上のホイールがあれば、プログレッシブで suspension_force が増加しているはず
        let grounded: Vec<_> = vehicle.wheel_states.iter().filter(|w| w.grounded).collect();
        assert!(!grounded.is_empty(), "Should have grounded wheels");
        for ws in &grounded {
            assert!(
                ws.suspension_force > Fix128::ZERO,
                "Suspension force should be positive"
            );
        }
    }

    #[test]
    fn test_anti_roll_bar() {
        let config = VehicleConfig::default();
        assert!(
            !config.anti_roll_stiffness.is_zero(),
            "Default anti-roll stiffness should be non-zero"
        );

        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // 左右圧縮差が小さければ anti-roll bar は微小な影響のみ
        // 基本的なシミュレーションが動作していることを確認
        assert!(vehicle.grounded_wheels() > 0);
    }

    #[test]
    fn test_asymmetric_damping_defaults() {
        let w = WheelConfig::default();
        // デフォルトではバンプ/リバウンド個別設定はゼロ（対称ダンピング使用）
        assert!(w.bump_damping.is_zero());
        assert!(w.rebound_damping.is_zero());
    }

    #[test]
    fn test_asymmetric_damping_simulation() {
        // バンプ減衰 > リバウンド減衰 の Rally 向け設定
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_damping = Fix128::from_int(6000); // 強いバンプ減衰
            w.rebound_damping = Fix128::from_int(3000); // 弱いリバウンド減衰
        }
        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..30 {
            vehicle.update(&mut chassis, dt);
        }

        assert!(vehicle.grounded_wheels() > 0, "Should have grounded wheels");
        // サスペンション力が正（支持している）ことを確認
        for ws in &vehicle.wheel_states {
            if ws.grounded {
                assert!(
                    ws.suspension_force > Fix128::ZERO,
                    "Suspension force should be positive with asymmetric damping"
                );
            }
        }
    }

    #[test]
    fn test_bump_stop_defaults() {
        let w = WheelConfig::default();
        assert!(w.bump_stop_stiffness.is_zero()); // デフォルトはバンプストップ無効
        let expected = Fix128::from_ratio(9, 10);
        assert_eq!(w.bump_stop_threshold, expected); // 閾値 0.9
    }

    #[test]
    fn test_bump_stop_activation() {
        // バンプストップ有効: 高圧縮時に追加スプリング力
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_stop_stiffness = Fix128::from_int(100_000); // 非常に硬い
            w.bump_stop_threshold = Fix128::from_ratio(5, 10); // 50% 圧縮で発動
        }

        let mut vehicle_with_stop = Vehicle::new(config.clone());
        let mut vehicle_without_stop = Vehicle::new(VehicleConfig::default());

        // 低い位置から開始（高い圧縮を生じさせる）
        let mut chassis_with =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.35, 0.0), Fix128::from_int(1500));
        let mut chassis_without =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.35, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle_with_stop.update(&mut chassis_with, dt);
        vehicle_without_stop.update(&mut chassis_without, dt);

        // バンプストップありの方がサスペンション力が大きい
        let force_with: Fix128 = vehicle_with_stop
            .wheel_states
            .iter()
            .filter(|w| w.grounded)
            .map(|w| w.suspension_force)
            .fold(Fix128::ZERO, |a, b| a + b);
        let force_without: Fix128 = vehicle_without_stop
            .wheel_states
            .iter()
            .filter(|w| w.grounded)
            .map(|w| w.suspension_force)
            .fold(Fix128::ZERO, |a, b| a + b);

        assert!(
            force_with >= force_without,
            "Bump stop should increase suspension force: with={}, without={}",
            force_with.to_f32(),
            force_without.to_f32()
        );
    }

    #[test]
    fn test_bump_stop_no_effect_low_compression() {
        // 圧縮が閾値以下ならバンプストップは効かない
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_stop_stiffness = Fix128::from_int(100_000);
            w.bump_stop_threshold = Fix128::from_ratio(95, 100); // 95%
        }

        let mut vehicle_with = Vehicle::new(config);
        let mut vehicle_without = Vehicle::new(VehicleConfig::default());

        // 高い位置 = 低い圧縮
        let mut chassis_with =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.55, 0.0), Fix128::from_int(1500));
        let mut chassis_without =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.55, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle_with.update(&mut chassis_with, dt);
        vehicle_without.update(&mut chassis_without, dt);

        // 低圧縮では両者の力は同じ
        for (ws_with, ws_without) in vehicle_with
            .wheel_states
            .iter()
            .zip(vehicle_without.wheel_states.iter())
        {
            if ws_with.grounded && ws_without.grounded {
                let diff = (ws_with.suspension_force - ws_without.suspension_force)
                    .to_f32()
                    .abs();
                assert!(
                    diff < 0.01,
                    "Low compression: bump stop should have no effect, diff={diff}"
                );
            }
        }
    }
}
