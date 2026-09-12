//! `BuoyancyZone` — 3-D fluid volume that applies buoyancy + drag to
//! immersed rigid bodies.
//!
//! Complements the existing [`crate::force::ForceField::Buoyancy`]
//! variant (infinite half-space with a horizontal water surface) by
//! modelling a **bounded** fluid volume with an explicit shape. Two
//! shapes are supported today:
//!
//! - [`ZoneShape::Aabb`] — axis-aligned box, e.g. a pool or tank.
//! - [`ZoneShape::Sphere`] — spherical pocket, e.g. a droplet or
//!   underwater cave.
//!
//! For each dynamic body inside the zone the resulting force is
//!
//! ```text
//! F = ρ_water · V_displaced · |g|  · ŷ_up
//!   − c_lin  · v
//!   − c_quad · v · |v|
//! ```
//!
//! where `V_displaced` is estimated from a spherical body proxy of
//! radius `body_radius`; the exact submerged volume of an arbitrary
//! mesh is out of scope for this module and remains future work.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

/// Shape of the buoyancy zone volume.
#[derive(Debug, Clone, Copy)]
pub enum ZoneShape {
    /// Axis-aligned box defined by `min` and `max` corners.
    Aabb {
        /// Lower corner (inclusive).
        min: Vec3Fix,
        /// Upper corner (inclusive).
        max: Vec3Fix,
    },
    /// Sphere with a centre and radius.
    Sphere {
        /// Centre position.
        centre: Vec3Fix,
        /// Radius.
        radius: Fix128,
    },
}

impl ZoneShape {
    /// True when `point` lies strictly inside the shape.
    #[must_use]
    pub fn contains(&self, point: Vec3Fix) -> bool {
        match self {
            Self::Aabb { min, max } => {
                point.x >= min.x
                    && point.x <= max.x
                    && point.y >= min.y
                    && point.y <= max.y
                    && point.z >= min.z
                    && point.z <= max.z
            }
            Self::Sphere { centre, radius } => {
                let delta = point - *centre;
                delta.length_squared() <= (*radius) * (*radius)
            }
        }
    }

    /// Depth of `point` below the zone's upper surface (positive when
    /// submerged, zero when at or above the surface).
    #[must_use]
    pub fn depth_below_surface(&self, point: Vec3Fix) -> Fix128 {
        match self {
            Self::Aabb { min, max } => {
                if point.x < min.x
                    || point.x > max.x
                    || point.z < min.z
                    || point.z > max.z
                    || point.y > max.y
                {
                    return Fix128::ZERO;
                }
                let clamped_y = if point.y < min.y { min.y } else { point.y };
                max.y - clamped_y
            }
            Self::Sphere { centre, radius } => {
                let delta = point - *centre;
                let dist_squared = delta.length_squared();
                let r_squared = (*radius) * (*radius);
                if dist_squared >= r_squared {
                    return Fix128::ZERO;
                }
                let surface_y = centre.y + *radius;
                if point.y > surface_y {
                    Fix128::ZERO
                } else {
                    surface_y - point.y
                }
            }
        }
    }
}

/// Bounded fluid volume applying buoyancy + drag.
#[derive(Debug, Clone, Copy)]
pub struct BuoyancyZone {
    /// Geometry of the fluid pocket.
    pub shape: ZoneShape,
    /// Fluid density (kg/m³). Water at 20 °C is ≈ 998.
    pub density_kg_m3: Fix128,
    /// Gravity magnitude (m/s²). Positive; direction is fixed to
    /// `+y` for buoyancy (world coordinates assume `+y = up`).
    pub gravity: Fix128,
    /// Linear drag coefficient `c_lin`. Applied as `-c_lin · v`.
    pub drag_linear: Fix128,
    /// Quadratic drag coefficient `c_quad`. Applied as `-c_quad · v · |v|`.
    pub drag_quadratic: Fix128,
}

impl BuoyancyZone {
    /// Construct a zone with reasonable defaults for a water pool
    /// (ρ = 1000 kg/m³, |g| = 9.81 m/s², linear drag 0.5, quadratic drag 1.0).
    #[must_use]
    pub fn water_pool(shape: ZoneShape) -> Self {
        Self {
            shape,
            density_kg_m3: Fix128::from_int(1000),
            gravity: Fix128::from_ratio(981, 100),
            drag_linear: Fix128::from_ratio(1, 2),
            drag_quadratic: Fix128::ONE,
        }
    }

    /// Compute the buoyancy + drag force acting on `body`. `body_radius`
    /// approximates the immersed volume as a sphere; callers can pass
    /// the enclosing-sphere radius of a compound collider.
    #[must_use]
    pub fn force_on(&self, body: &RigidBody, body_radius: Fix128) -> Vec3Fix {
        let submerged_fraction = self.submerged_fraction(body.position, body_radius);
        if submerged_fraction.is_zero() {
            return Vec3Fix::ZERO;
        }

        // V_displaced = fraction · (4/3) π r³. We approximate 4π/3 ≈ 4.18879
        // via Fix128::from_ratio(41888, 10000).
        let four_pi_third = Fix128::from_ratio(41888, 10000);
        let radius_cubed = body_radius * body_radius * body_radius;
        let full_volume = four_pi_third * radius_cubed;
        let displaced = full_volume * submerged_fraction;
        let buoy_magnitude = self.density_kg_m3 * displaced * self.gravity;
        let buoyancy = Vec3Fix::new(Fix128::ZERO, buoy_magnitude, Fix128::ZERO);

        // Linear drag: -c_lin · v.
        let linear_drag = if body.velocity.length_squared().is_zero() {
            Vec3Fix::ZERO
        } else {
            body.velocity * (-self.drag_linear)
        };

        // Quadratic drag: -c_quad · v · |v|.
        let speed_sq = body.velocity.length_squared();
        let quadratic_drag = if speed_sq.is_zero() {
            Vec3Fix::ZERO
        } else {
            let speed = speed_sq.sqrt();
            body.velocity * (-self.drag_quadratic * speed)
        };

        buoyancy + linear_drag + quadratic_drag
    }

    /// Approximate submerged fraction (`0..=1`) of a sphere of `radius`
    /// centred at `position`.
    ///
    /// The estimate is linear in depth of the sphere centre relative to
    /// the zone's upper surface, capped at `1.0`. For AABB shapes the
    /// centre must additionally lie inside the horizontal footprint;
    /// for spheres the centre must lie inside the ball.
    #[must_use]
    pub fn submerged_fraction(&self, position: Vec3Fix, radius: Fix128) -> Fix128 {
        if radius.is_zero() {
            return Fix128::ZERO;
        }
        let diameter = radius + radius;
        let depth = self.shape.depth_below_surface(position);
        if depth.is_zero() {
            return Fix128::ZERO;
        }
        // fraction = clamp( (depth + radius) / diameter , 0, 1 )
        let scaled = (depth + radius) / diameter;
        if scaled > Fix128::ONE {
            Fix128::ONE
        } else {
            scaled
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{BodyType, RigidBody};

    fn floating_body(y: i64) -> RigidBody {
        let mut body = RigidBody {
            position: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(y), Fix128::ZERO),
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ONE,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: Vec3Fix::ZERO,
            rotation: crate::math::QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: crate::math::QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ONE,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        };
        body.prev_position = body.position;
        body
    }

    #[test]
    fn aabb_contains_interior_point() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        assert!(shape.contains(Vec3Fix::ZERO));
        assert!(!shape.contains(Vec3Fix::new(
            Fix128::from_int(2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
    }

    #[test]
    fn sphere_contains_interior_point() {
        let shape = ZoneShape::Sphere {
            centre: Vec3Fix::ZERO,
            radius: Fix128::from_int(1),
        };
        assert!(shape.contains(Vec3Fix::new(
            Fix128::from_ratio(1, 2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
        assert!(!shape.contains(Vec3Fix::new(
            Fix128::from_int(2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
    }

    #[test]
    fn aabb_depth_zero_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        let above = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO);
        assert_eq!(shape.depth_below_surface(above), Fix128::ZERO);
    }

    #[test]
    fn aabb_depth_matches_expected_at_centre() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-2),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(2),
                Fix128::from_int(1),
            ),
        };
        // Point at (0, 0, 0). Zone top at y = 2. Depth should be 2.
        let depth = shape.depth_below_surface(Vec3Fix::ZERO);
        assert_eq!(depth, Fix128::from_int(2));
    }

    #[test]
    fn buoyancy_produces_upward_force_when_submerged() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let body = floating_body(-1);
        let force = zone.force_on(&body, Fix128::ONE);
        // Upward buoyancy => positive y component.
        assert!(force.y > Fix128::ZERO);
        // No horizontal force in resting state.
        assert_eq!(force.x, Fix128::ZERO);
        assert_eq!(force.z, Fix128::ZERO);
    }

    #[test]
    fn buoyancy_zero_when_body_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        // Body at y = 10, zone max y = 5 → outside.
        let body = floating_body(10);
        let force = zone.force_on(&body, Fix128::ONE);
        assert_eq!(force, Vec3Fix::ZERO);
    }

    #[test]
    fn drag_opposes_velocity_when_submerged() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let mut body = floating_body(0);
        body.velocity = Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO);
        let force = zone.force_on(&body, Fix128::ONE);
        // Drag should be negative x (opposing velocity).
        assert!(force.x < Fix128::ZERO);
    }

    #[test]
    fn submerged_fraction_capped_at_one() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let position = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-4), Fix128::ZERO);
        // Deep inside — fraction should be 1.
        let fraction = zone.submerged_fraction(position, Fix128::from_ratio(1, 2));
        assert_eq!(fraction, Fix128::ONE);
    }

    #[test]
    fn submerged_fraction_zero_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let position = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO);
        let fraction = zone.submerged_fraction(position, Fix128::ONE);
        assert_eq!(fraction, Fix128::ZERO);
    }

    #[test]
    fn zero_radius_has_zero_fraction() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let fraction = zone.submerged_fraction(Vec3Fix::ZERO, Fix128::ZERO);
        assert_eq!(fraction, Fix128::ZERO);
    }
}
