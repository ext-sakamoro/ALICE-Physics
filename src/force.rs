//! Custom Force Fields and Gravity
//!
//! Apply custom forces, directional gravity, point gravity, wind, drag,
//! and other force fields to rigid bodies.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Force field type
#[derive(Clone, Copy, Debug)]
pub enum ForceField {
    /// Constant directional force (e.g., wind)
    Directional {
        direction: Vec3Fix,
        strength: Fix128,
    },

    /// Point attractor/repulsor (gravity well)
    Point {
        center: Vec3Fix,
        strength: Fix128,
        /// If true, force pushes away from center (explosion)
        repulsive: bool,
        /// Maximum force (prevents singularity near center)
        max_force: Fix128,
    },

    /// Linear drag (velocity-proportional resistance)
    Drag {
        coefficient: Fix128,
    },

    /// Buoyancy (upward force below a surface level)
    Buoyancy {
        surface_y: Fix128,
        density: Fix128,
        drag: Fix128,
    },

    /// Vortex (rotational force around an axis)
    Vortex {
        center: Vec3Fix,
        axis: Vec3Fix,
        strength: Fix128,
        falloff_radius: Fix128,
    },
}

/// A force field with optional body filter
#[derive(Clone, Debug)]
pub struct ForceFieldInstance {
    pub field: ForceField,
    /// If Some, only affects bodies in this list. If None, affects all bodies.
    pub affected_bodies: Option<Vec<usize>>,
    /// Whether this field is active
    pub enabled: bool,
}

impl ForceFieldInstance {
    #[inline]
    pub fn new(field: ForceField) -> Self {
        Self {
            field,
            affected_bodies: None,
            enabled: true,
        }
    }

    pub fn with_affected_bodies(mut self, bodies: Vec<usize>) -> Self {
        self.affected_bodies = Some(bodies);
        self
    }

    /// Check if this field affects a given body
    #[inline]
    fn affects(&self, body_index: usize) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.affected_bodies {
            None => true,
            Some(list) => list.contains(&body_index),
        }
    }
}

/// Compute force from a force field on a body at a given position
pub fn compute_force(field: &ForceField, body: &RigidBody) -> Vec3Fix {
    match field {
        ForceField::Directional { direction, strength } => {
            *direction * *strength
        }

        ForceField::Point { center, strength, repulsive, max_force } => {
            let delta = *center - body.position;
            let dist_sq = delta.length_squared();

            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }

            let dist = dist_sq.sqrt();
            let direction = delta / dist;

            // Inverse-square law: F = strength / r^2
            let force_mag = *strength / dist_sq;
            let clamped = if force_mag > *max_force { *max_force } else { force_mag };

            if *repulsive {
                -direction * clamped
            } else {
                direction * clamped
            }
        }

        ForceField::Drag { coefficient } => {
            let speed_sq = body.velocity.length_squared();
            if speed_sq.is_zero() {
                return Vec3Fix::ZERO;
            }
            let speed = speed_sq.sqrt();
            let drag_dir = body.velocity / speed;
            -drag_dir * (*coefficient * speed)
        }

        ForceField::Buoyancy { surface_y, density, drag } => {
            let depth = *surface_y - body.position.y;
            if depth <= Fix128::ZERO {
                return Vec3Fix::ZERO;
            }

            // Buoyancy force proportional to submerged depth
            let buoyancy = Vec3Fix::new(Fix128::ZERO, *density * depth, Fix128::ZERO);

            // Water drag
            let water_drag = if body.velocity.length_squared().is_zero() {
                Vec3Fix::ZERO
            } else {
                body.velocity * (-*drag)
            };

            buoyancy + water_drag
        }

        ForceField::Vortex { center, axis, strength, falloff_radius } => {
            let delta = body.position - *center;

            // Project delta onto plane perpendicular to axis
            let axis_norm = axis.normalize();
            let along_axis = axis_norm * delta.dot(axis_norm);
            let radial = delta - along_axis;
            let dist = radial.length();

            if dist.is_zero() || falloff_radius.is_zero() {
                return Vec3Fix::ZERO;
            }

            // Tangent direction (cross product of axis and radial)
            let tangent = axis_norm.cross(radial.normalize());

            // Falloff: linear decrease beyond falloff_radius
            let falloff = if dist < *falloff_radius {
                Fix128::ONE
            } else {
                *falloff_radius / dist
            };

            tangent * (*strength * falloff)
        }
    }
}

/// Apply all force fields to all bodies for one timestep
pub fn apply_force_fields(
    fields: &[ForceFieldInstance],
    bodies: &mut [RigidBody],
    dt: Fix128,
) {
    for (body_idx, body) in bodies.iter_mut().enumerate() {
        if body.is_static() {
            continue;
        }

        let mut total_force = Vec3Fix::ZERO;

        for field_inst in fields {
            if !field_inst.affects(body_idx) {
                continue;
            }
            total_force = total_force + compute_force(&field_inst.field, body);
        }

        // F = ma, a = F * inv_mass, v += a * dt
        let acceleration = total_force * body.inv_mass;
        body.velocity = body.velocity + acceleration * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directional_force() {
        let field = ForceField::Directional {
            direction: Vec3Fix::UNIT_Y,
            strength: Fix128::from_int(10),
        };
        let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        let force = compute_force(&field, &body);
        assert_eq!(force.y.hi, 10);
    }

    #[test]
    fn test_drag_force() {
        let field = ForceField::Drag {
            coefficient: Fix128::from_int(1),
        };
        let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        body.velocity = Vec3Fix::from_int(10, 0, 0);

        let force = compute_force(&field, &body);
        // Drag opposes velocity
        assert!(force.x < Fix128::ZERO, "Drag should oppose motion");
    }

    #[test]
    fn test_point_gravity() {
        let field = ForceField::Point {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            repulsive: false,
            max_force: Fix128::from_int(1000),
        };
        let body = RigidBody::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        // Should pull toward center
        assert!(force.x < Fix128::ZERO, "Point gravity should pull toward center");
    }

    #[test]
    fn test_buoyancy() {
        let field = ForceField::Buoyancy {
            surface_y: Fix128::from_int(5),
            density: Fix128::from_int(10),
            drag: Fix128::ONE,
        };

        // Body below surface
        let body = RigidBody::new(Vec3Fix::from_int(0, 2, 0), Fix128::ONE);
        let force = compute_force(&field, &body);
        assert!(force.y > Fix128::ZERO, "Buoyancy should push up");

        // Body above surface
        let body_above = RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        let force_above = compute_force(&field, &body_above);
        assert!(force_above.y.is_zero() && force_above.x.is_zero(), "No buoyancy above surface");
    }

    #[test]
    fn test_apply_force_fields() {
        let fields = vec![
            ForceFieldInstance::new(ForceField::Directional {
                direction: Vec3Fix::UNIT_X,
                strength: Fix128::from_int(10),
            }),
        ];

        let mut bodies = vec![
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        ];

        let dt = Fix128::from_ratio(1, 60);
        apply_force_fields(&fields, &mut bodies, dt);

        // Velocity should have increased in X
        assert!(bodies[0].velocity.x > Fix128::ZERO, "Force should accelerate body");
    }

    #[test]
    fn test_affected_bodies_filter() {
        let field = ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(100),
        }).with_affected_bodies(vec![0]); // Only affects body 0

        assert!(field.affects(0));
        assert!(!field.affects(1));
    }

    #[test]
    fn test_vortex() {
        let field = ForceField::Vortex {
            center: Vec3Fix::ZERO,
            axis: Vec3Fix::UNIT_Y,
            strength: Fix128::from_int(10),
            falloff_radius: Fix128::from_int(100),
        };

        let body = RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        // Vortex should produce tangential force (Z direction for body at +X)
        assert!(force.z.abs() > Fix128::ZERO, "Vortex should produce tangential force");
    }
}
