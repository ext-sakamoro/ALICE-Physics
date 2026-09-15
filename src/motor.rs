//! Motors and PD Controllers
//!
//! Drive joints to target positions/velocities using proportional-derivative control.
//! Deterministic: all computations use Fix128.

use crate::joint::Joint;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::RigidBody;

/// Motor mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotorMode {
    /// Disabled (no motor force)
    Off,
    /// Drive to target position (PD control)
    Position,
    /// Drive to target velocity (P control)
    Velocity,
}

/// PD controller for a single joint axis
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdController {
    /// Proportional gain (stiffness)
    pub kp: Fix128,
    /// Derivative gain (damping)
    pub kd: Fix128,
    /// Maximum output force/torque
    pub max_force: Fix128,
    /// Target position (for Position mode)
    pub target_position: Fix128,
    /// Target velocity (for Velocity mode)
    pub target_velocity: Fix128,
    /// Motor mode
    pub mode: MotorMode,
}

impl PdController {
    /// Create a new PD controller
    #[inline]
    #[must_use]
    pub const fn new(kp: Fix128, kd: Fix128, max_force: Fix128) -> Self {
        Self {
            kp,
            kd,
            max_force,
            target_position: Fix128::ZERO,
            target_velocity: Fix128::ZERO,
            mode: MotorMode::Off,
        }
    }

    /// Set position target
    pub fn set_position_target(&mut self, target: Fix128) {
        self.target_position = target;
        self.mode = MotorMode::Position;
    }

    /// Set velocity target
    pub fn set_velocity_target(&mut self, target: Fix128) {
        self.target_velocity = target;
        self.mode = MotorMode::Velocity;
    }

    /// Disable the motor
    pub fn disable(&mut self) {
        self.mode = MotorMode::Off;
    }

    /// Compute control output
    ///
    /// Returns clamped force/torque value.
    #[must_use]
    pub fn compute(&self, current_position: Fix128, current_velocity: Fix128) -> Fix128 {
        match self.mode {
            MotorMode::Off => Fix128::ZERO,
            MotorMode::Position => {
                let error = self.target_position - current_position;
                let vel_error = self.target_velocity - current_velocity;
                let output = self.kp * error + self.kd * vel_error;
                clamp(output, -self.max_force, self.max_force)
            }
            MotorMode::Velocity => {
                let vel_error = self.target_velocity - current_velocity;
                let output = self.kp * vel_error;
                clamp(output, -self.max_force, self.max_force)
            }
        }
    }
}

impl Default for PdController {
    fn default() -> Self {
        Self::new(
            Fix128::from_int(100),  // kp = 100
            Fix128::from_int(10),   // kd = 10
            Fix128::from_int(1000), // max_force = 1000
        )
    }
}

/// Motor attached to a joint
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointMotor {
    /// Index into joints array
    pub joint_index: usize,
    /// PD controller for the primary axis
    pub controller: PdController,
}

impl JointMotor {
    /// Create a new motor for the given joint
    #[must_use]
    pub const fn new(joint_index: usize, controller: PdController) -> Self {
        Self {
            joint_index,
            controller,
        }
    }
}

/// 3-axis PD controller for ball joints / free rotation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdController3D {
    /// Per-axis proportional gains
    pub kp: Vec3Fix,
    /// Per-axis derivative gains
    pub kd: Vec3Fix,
    /// Maximum torque per axis
    pub max_torque: Fix128,
    /// Target orientation
    pub target_rotation: QuatFix,
    /// Target angular velocity
    pub target_angular_velocity: Vec3Fix,
    /// Motor mode
    pub mode: MotorMode,
}

impl PdController3D {
    /// Create a new 3-axis PD controller
    #[inline]
    #[must_use]
    pub const fn new(kp: Vec3Fix, kd: Vec3Fix, max_torque: Fix128) -> Self {
        Self {
            kp,
            kd,
            max_torque,
            target_rotation: QuatFix::IDENTITY,
            target_angular_velocity: Vec3Fix::ZERO,
            mode: MotorMode::Off,
        }
    }

    /// Set rotation target
    pub fn set_rotation_target(&mut self, target: QuatFix) {
        self.target_rotation = target;
        self.mode = MotorMode::Position;
    }

    /// Compute 3D torque output
    #[must_use]
    pub fn compute_torque(
        &self,
        current_rotation: QuatFix,
        current_angular_velocity: Vec3Fix,
    ) -> Vec3Fix {
        match self.mode {
            MotorMode::Off => Vec3Fix::ZERO,
            MotorMode::Position => {
                // Rotation error as axis-angle
                let error_quat = self.target_rotation.mul(current_rotation.conjugate());
                let error_axis = Vec3Fix::new(error_quat.x, error_quat.y, error_quat.z);
                let two = Fix128::from_int(2);
                let error_scaled =
                    Vec3Fix::new(error_axis.x * two, error_axis.y * two, error_axis.z * two);

                // PD control per axis
                let vel_error = self.target_angular_velocity - current_angular_velocity;
                let torque = Vec3Fix::new(
                    self.kp.x * error_scaled.x + self.kd.x * vel_error.x,
                    self.kp.y * error_scaled.y + self.kd.y * vel_error.y,
                    self.kp.z * error_scaled.z + self.kd.z * vel_error.z,
                );

                // Clamp torque magnitude
                let mag = torque.length();
                if mag > self.max_torque && !mag.is_zero() {
                    torque * (self.max_torque / mag)
                } else {
                    torque
                }
            }
            MotorMode::Velocity => {
                let vel_error = self.target_angular_velocity - current_angular_velocity;
                let torque = Vec3Fix::new(
                    self.kp.x * vel_error.x,
                    self.kp.y * vel_error.y,
                    self.kp.z * vel_error.z,
                );

                let mag = torque.length();
                if mag > self.max_torque && !mag.is_zero() {
                    torque * (self.max_torque / mag)
                } else {
                    torque
                }
            }
        }
    }
}

/// Apply all joint motors for one timestep
pub fn apply_motors(motors: &[JointMotor], joints: &[Joint], bodies: &mut [RigidBody], dt: Fix128) {
    for motor in motors {
        if motor.controller.mode == MotorMode::Off {
            continue;
        }

        if motor.joint_index >= joints.len() {
            continue;
        }

        let (body_a_idx, body_b_idx) = joints[motor.joint_index].bodies();

        if body_a_idx >= bodies.len() || body_b_idx >= bodies.len() {
            continue;
        }

        // Compute current state along joint axis
        let body_a = bodies[body_a_idx];
        let body_b = bodies[body_b_idx];

        let delta = body_b.position - body_a.position;
        let current_pos = delta.length();
        let rel_vel = body_b.velocity - body_a.velocity;
        let current_vel = if current_pos.is_zero() {
            Fix128::ZERO
        } else {
            rel_vel.dot(delta / current_pos)
        };

        let force = motor.controller.compute(current_pos, current_vel);

        if force.is_zero() || current_pos.is_zero() {
            continue;
        }

        let direction = delta / current_pos;
        let impulse = direction * (force * dt);

        if !body_a.inv_mass.is_zero() {
            bodies[body_a_idx].velocity = bodies[body_a_idx].velocity - impulse * body_a.inv_mass;
        }
        if !body_b.inv_mass.is_zero() {
            bodies[body_b_idx].velocity = bodies[body_b_idx].velocity + impulse * body_b.inv_mass;
        }
    }
}

#[inline]
fn clamp(v: Fix128, min: Fix128, max: Fix128) -> Fix128 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::BallJoint;

    #[test]
    fn test_pd_position_control() {
        let mut pd = PdController::new(
            Fix128::from_int(10),
            Fix128::from_int(1),
            Fix128::from_int(100),
        );
        pd.set_position_target(Fix128::from_int(5));

        let output = pd.compute(Fix128::ZERO, Fix128::ZERO);
        // error = 5, output = 10*5 + 1*0 = 50
        assert_eq!(output.hi, 50);
    }

    #[test]
    fn test_pd_clamp() {
        let mut pd = PdController::new(
            Fix128::from_int(1000),
            Fix128::ZERO,
            Fix128::from_int(10), // max = 10
        );
        pd.set_position_target(Fix128::from_int(100));

        let output = pd.compute(Fix128::ZERO, Fix128::ZERO);
        assert_eq!(output.hi, 10, "Should be clamped to max_force");
    }

    #[test]
    fn test_pd_off() {
        let pd = PdController::new(
            Fix128::from_int(10),
            Fix128::from_int(1),
            Fix128::from_int(100),
        );
        // Mode is Off by default
        let output = pd.compute(Fix128::ZERO, Fix128::ZERO);
        assert!(output.is_zero());
    }

    #[test]
    fn test_velocity_mode() {
        let mut pd = PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(1000));
        pd.set_velocity_target(Fix128::from_int(5));

        let output = pd.compute(Fix128::ZERO, Fix128::ZERO);
        // vel_error = 5, output = 10*5 = 50
        assert_eq!(output.hi, 50);
    }

    #[test]
    fn test_3d_controller() {
        let mut pd = PdController3D::new(
            Vec3Fix::from_int(100, 100, 100),
            Vec3Fix::from_int(10, 10, 10),
            Fix128::from_int(1000),
        );
        pd.set_rotation_target(QuatFix::IDENTITY);
        pd.mode = MotorMode::Position;

        let torque = pd.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO);
        // No error => zero torque
        let mag = torque.length();
        assert!(
            mag < Fix128::from_ratio(1, 10),
            "Zero error should give zero torque"
        );
    }

    /// A (inv_mass ia) at origin、B (inv_mass ib) at (4, 0, 0)、Ball joint 0-1
    fn motor_scene(ia: i64, ib: i64) -> (Vec<RigidBody>, Vec<Joint>) {
        let mut a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(4, 0, 0), Fix128::ONE);
        a.inv_mass = Fix128::from_int(ia);
        b.inv_mass = Fix128::from_int(ib);
        let joints = vec![Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))];
        (vec![a, b], joints)
    }

    #[test]
    fn apply_motors_position_mode_pushes_along_joint_axis_by_force_dt_inv_mass() {
        // kp 10 / kd 0 / max 100、target 距離 6、現在 4 → force = 10 * 2 = 20
        // dt 1/4 → impulse 5 → B (inv 1) velocity += (5, 0, 0)、A static は不動
        let (mut bodies, joints) = motor_scene(0, 1);
        let mut pd = PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(100));
        pd.set_position_target(Fix128::from_int(6));
        let motors = [JointMotor::new(0, pd)];
        apply_motors(&motors, &joints, &mut bodies, Fix128::from_ratio(1, 4));
        assert_eq!(bodies[1].velocity, Vec3Fix::from_int(5, 0, 0));
        assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);

        // 目標が現在より近い (2) → force = -20 → B は A 側へ (-5, 0, 0)
        let (mut closer, joints) = motor_scene(0, 1);
        let mut pd_in = pd;
        pd_in.set_position_target(Fix128::from_int(2));
        apply_motors(
            &[JointMotor::new(0, pd_in)],
            &joints,
            &mut closer,
            Fix128::from_ratio(1, 4),
        );
        assert_eq!(closer[1].velocity, Vec3Fix::from_int(-5, 0, 0));

        // 両 dynamic (inv 1 / inv 2): impulse 5 → A -= 5、B += 10 (運動量 = 質量比で分配)
        let (mut both, joints) = motor_scene(1, 2);
        apply_motors(
            &[JointMotor::new(0, pd)],
            &joints,
            &mut both,
            Fix128::from_ratio(1, 4),
        );
        assert_eq!(both[0].velocity, Vec3Fix::from_int(-5, 0, 0));
        assert_eq!(both[1].velocity, Vec3Fix::from_int(10, 0, 0));

        // max_force 8 で clamp → impulse 2
        let (mut clamped, joints) = motor_scene(0, 1);
        let mut pd_clamp =
            PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(8));
        pd_clamp.set_position_target(Fix128::from_int(6));
        apply_motors(
            &[JointMotor::new(0, pd_clamp)],
            &joints,
            &mut clamped,
            Fix128::from_ratio(1, 4),
        );
        assert_eq!(clamped[1].velocity, Vec3Fix::from_int(2, 0, 0));
    }

    #[test]
    fn apply_motors_velocity_mode_uses_relative_velocity_along_axis() {
        // kp 10、target vel 3、B が +x に 1 → vel_error 2 → force 20 → impulse 5 → B (6, 0, 0)
        // 軸直交成分 (y) は current_vel に寄与しない
        let (mut bodies, joints) = motor_scene(0, 1);
        bodies[1].velocity = Vec3Fix::from_int(1, 7, 0);
        let mut pd = PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(100));
        pd.set_velocity_target(Fix128::from_int(3));
        apply_motors(
            &[JointMotor::new(0, pd)],
            &joints,
            &mut bodies,
            Fix128::from_ratio(1, 4),
        );
        assert_eq!(bodies[1].velocity, Vec3Fix::from_int(6, 7, 0));
    }

    #[test]
    fn apply_motors_skips_off_out_of_range_zero_force_and_coincident_bodies() {
        let (mut bodies, joints) = motor_scene(0, 1);
        let mut pd = PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(100));
        pd.set_position_target(Fix128::from_int(6));
        let dt = Fix128::from_ratio(1, 4);

        // Off
        let mut off = pd;
        off.disable();
        apply_motors(&[JointMotor::new(0, off)], &joints, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
        // joint index 範囲外
        apply_motors(&[JointMotor::new(1, pd)], &joints, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
        // body index 範囲外 (joint が body 5 を参照)
        let far = [Joint::Ball(BallJoint::new(
            0,
            5,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))];
        apply_motors(&[JointMotor::new(0, pd)], &far, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
        // 目標 = 現在距離 4 → force 0 → 不変
        let mut at_target = pd;
        at_target.set_position_target(Fix128::from_int(4));
        apply_motors(&[JointMotor::new(0, at_target)], &joints, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
        // 同一点 (current_pos 0) → 方向が定義できず skip
        bodies[1].position = Vec3Fix::ZERO;
        apply_motors(&[JointMotor::new(0, pd)], &joints, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
        // 通常 path は動く (上の skip が「何もしない」だけでない事の対照)
        bodies[1].position = Vec3Fix::from_int(4, 0, 0);
        apply_motors(&[JointMotor::new(0, pd)], &joints, &mut bodies, dt);
        assert_eq!(bodies[1].velocity, Vec3Fix::from_int(5, 0, 0));
    }
}
