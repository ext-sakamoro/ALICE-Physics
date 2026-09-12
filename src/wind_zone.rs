//! `WindZone` — bounded aerodynamic force field applied to rigid bodies
//! and cloth / hair particles.
//!
//! Companion of [`crate::buoyancy_zone`]: reuses the same
//! [`crate::buoyancy_zone::ZoneShape`] volume primitive but replaces
//! the buoyancy + drag model with a directional wind + turbulent gust
//! model tailored for cloth simulation, hair strand advection, and
//! kinematic character wind effects.
//!
//! Aerodynamic force acting on a particle / body inside the zone
//!
//! ```text
//! F = ½ · ρ_air · C_d · A · |v_rel| · v_rel
//! ```
//!
//! where `v_rel = wind_velocity(t) − body_velocity` and
//! `wind_velocity(t) = base_direction · base_speed
//!                     + turbulence · sin(2π · gust_frequency · t)`.
//!
//! The MVP evaluates the aerodynamic force in the body's centre and
//! ignores surface-orientation dependence; full projected-area
//! integration and streamwise-turbulence spectral models are future
//! work.

use crate::buoyancy_zone::ZoneShape;
use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

/// Directional wind + turbulent gust volume.
#[derive(Debug, Clone, Copy)]
pub struct WindZone {
    /// Geometry of the wind pocket.
    pub shape: ZoneShape,
    /// Air density (kg/m³). Sea-level ISA = 1.225.
    pub air_density_kg_m3: Fix128,
    /// Nominal wind direction (unit vector). The builder normalises
    /// on construction; the field can be non-unit if the caller wants
    /// to bake wind magnitude in.
    pub direction: Vec3Fix,
    /// Base wind speed (m/s) applied along `direction`.
    pub base_speed_m_s: Fix128,
    /// Turbulence amplitude scaled by `sin(2π · gust_frequency · t)`
    /// and added to the base wind vector.
    pub turbulence_amplitude: Fix128,
    /// Gust temporal frequency in Hz.
    pub gust_frequency_hz: Fix128,
    /// Drag coefficient `C_d`. Cloth panels are often ~1.2; unit
    /// spheres are 0.47.
    pub drag_coefficient: Fix128,
    /// Reference cross-sectional area `A` (m²). Callers who want
    /// particle-sized cloth can pass the mesh triangle area; for
    /// bodies the projected area of the enclosing sphere is a
    /// reasonable proxy.
    pub reference_area_m2: Fix128,
}

impl WindZone {
    /// Preset for a light outdoor breeze (3 m/s base + 0.5 m/s gust,
    /// C_d = 1.2, A = 0.5 m²).
    #[must_use]
    pub fn light_breeze(shape: ZoneShape) -> Self {
        Self {
            shape,
            air_density_kg_m3: Fix128::from_ratio(1225, 1000),
            direction: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            base_speed_m_s: Fix128::from_int(3),
            turbulence_amplitude: Fix128::from_ratio(1, 2),
            gust_frequency_hz: Fix128::from_ratio(1, 2),
            drag_coefficient: Fix128::from_ratio(12, 10),
            reference_area_m2: Fix128::from_ratio(1, 2),
        }
    }

    /// Preset for a storm-force wind (20 m/s base + 5 m/s gust).
    #[must_use]
    pub fn storm(shape: ZoneShape) -> Self {
        Self {
            shape,
            air_density_kg_m3: Fix128::from_ratio(1225, 1000),
            direction: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            base_speed_m_s: Fix128::from_int(20),
            turbulence_amplitude: Fix128::from_int(5),
            gust_frequency_hz: Fix128::from_ratio(3, 2),
            drag_coefficient: Fix128::from_ratio(12, 10),
            reference_area_m2: Fix128::from_ratio(1, 2),
        }
    }

    /// Instantaneous wind velocity vector at simulation time `t_s`.
    ///
    /// The base direction is scaled by `base_speed_m_s`, then the
    /// turbulence gust `turbulence · sin(2π · f · t)` is added along
    /// the same direction. Callers who need cross-wind gusts can
    /// override this via [`WindZone::instantaneous_wind_vector`] after
    /// construction.
    #[must_use]
    pub fn instantaneous_wind_vector(&self, t_s: Fix128) -> Vec3Fix {
        let base = self.direction * self.base_speed_m_s;
        // sin(2π · f · t). We can call the crate's fixed-point sine
        // because it is CORDIC-based and cheap.
        let two_pi = Fix128::PI * Fix128::from_int(2);
        let phase = two_pi * self.gust_frequency_hz * t_s;
        let gust_scalar = self.turbulence_amplitude * phase.sin();
        base + self.direction * gust_scalar
    }

    /// Aerodynamic force on `body` at simulation time `t_s`. Returns
    /// zero if the body centre is outside the zone.
    #[must_use]
    pub fn force_on(&self, body: &RigidBody, t_s: Fix128) -> Vec3Fix {
        if !self.shape.contains(body.position) {
            return Vec3Fix::ZERO;
        }
        let wind = self.instantaneous_wind_vector(t_s);
        self.aerodynamic_force(wind, body.velocity)
    }

    /// Aerodynamic force on a cloth / hair particle at simulation time
    /// `t_s`. `position` is checked against the zone; `velocity` is
    /// the particle's current velocity.
    #[must_use]
    pub fn force_on_particle(&self, position: Vec3Fix, velocity: Vec3Fix, t_s: Fix128) -> Vec3Fix {
        if !self.shape.contains(position) {
            return Vec3Fix::ZERO;
        }
        let wind = self.instantaneous_wind_vector(t_s);
        self.aerodynamic_force(wind, velocity)
    }

    fn aerodynamic_force(&self, wind_velocity: Vec3Fix, body_velocity: Vec3Fix) -> Vec3Fix {
        let relative = wind_velocity - body_velocity;
        let speed_sq = relative.length_squared();
        if speed_sq.is_zero() {
            return Vec3Fix::ZERO;
        }
        let speed = speed_sq.sqrt();
        let half = Fix128::from_ratio(1, 2);
        let magnitude = half
            * self.air_density_kg_m3
            * self.drag_coefficient
            * self.reference_area_m2
            * speed
            * speed;
        // Direction of the aerodynamic force is aligned with the
        // relative velocity.
        relative * (magnitude / speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buoyancy_zone::ZoneShape;
    use crate::math::QuatFix;
    use crate::solver::{BodyType, RigidBody};

    fn wide_aabb() -> ZoneShape {
        ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-10),
                Fix128::from_int(-10),
                Fix128::from_int(-10),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(10),
                Fix128::from_int(10),
                Fix128::from_int(10),
            ),
        }
    }

    fn resting_body_at(x: i64, y: i64, z: i64) -> RigidBody {
        RigidBody {
            position: Vec3Fix::new(
                Fix128::from_int(x),
                Fix128::from_int(y),
                Fix128::from_int(z),
            ),
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ONE,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: Vec3Fix::ZERO,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ONE,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        }
    }

    #[test]
    fn light_breeze_preset_configured() {
        let zone = WindZone::light_breeze(wide_aabb());
        assert_eq!(zone.base_speed_m_s, Fix128::from_int(3));
        assert!(zone.direction.x > Fix128::ZERO);
    }

    #[test]
    fn storm_preset_stronger_than_breeze() {
        let breeze = WindZone::light_breeze(wide_aabb());
        let storm = WindZone::storm(wide_aabb());
        assert!(storm.base_speed_m_s > breeze.base_speed_m_s);
        assert!(storm.turbulence_amplitude > breeze.turbulence_amplitude);
    }

    #[test]
    fn wind_vector_at_zero_time_matches_base() {
        let zone = WindZone::light_breeze(wide_aabb());
        let v = zone.instantaneous_wind_vector(Fix128::ZERO);
        // sin(0) is close to 0 under CORDIC, so only base term remains
        // to within one ULP of the CORDIC accuracy.
        let expected = zone.direction * zone.base_speed_m_s;
        let dx = v.x - expected.x;
        let dy = v.y - expected.y;
        let dz = v.z - expected.z;
        let tolerance = Fix128::from_ratio(1, 100);
        assert!(dx.abs() < tolerance);
        assert!(dy.abs() < tolerance);
        assert!(dz.abs() < tolerance);
    }

    #[test]
    fn force_zero_outside_zone() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(100, 0, 0); // outside AABB.
        let f = zone.force_on(&body, Fix128::ZERO);
        assert_eq!(f, Vec3Fix::ZERO);
    }

    #[test]
    fn force_direction_aligned_with_wind() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(0, 0, 0);
        let f = zone.force_on(&body, Fix128::ZERO);
        // Wind is +x, body is resting, so force must be +x.
        assert!(f.x > Fix128::ZERO);
        assert_eq!(f.y, Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn force_reverses_when_body_moves_faster_than_wind() {
        let zone = WindZone::light_breeze(wide_aabb());
        let mut body = resting_body_at(0, 0, 0);
        body.velocity = Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        let f = zone.force_on(&body, Fix128::ZERO);
        // Body outruns the wind → drag pulls back (−x).
        assert!(f.x < Fix128::ZERO);
    }

    #[test]
    fn particle_force_matches_body_force_for_same_state() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(0, 0, 0);
        let f_body = zone.force_on(&body, Fix128::ZERO);
        let f_particle = zone.force_on_particle(body.position, body.velocity, Fix128::ZERO);
        assert_eq!(f_body, f_particle);
    }

    #[test]
    fn zero_force_when_wind_matches_body_velocity() {
        let zone = WindZone::light_breeze(wide_aabb());
        let mut body = resting_body_at(0, 0, 0);
        body.velocity = zone.direction * zone.base_speed_m_s;
        let f = zone.force_on(&body, Fix128::ZERO);
        assert_eq!(f, Vec3Fix::ZERO);
    }
}
