//! Custom Force Fields and Gravity
//!
//! Apply custom forces, directional gravity, point gravity, wind, drag,
//! and other force fields to rigid bodies.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Force field type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForceField {
    /// Constant directional force (e.g., wind)
    Directional {
        /// Force direction (normalized)
        direction: Vec3Fix,
        /// Force magnitude
        strength: Fix128,
    },

    /// Point attractor/repulsor (gravity well)
    Point {
        /// Attractor center position
        center: Vec3Fix,
        /// Attraction strength
        strength: Fix128,
        /// If true, force pushes away from center (explosion)
        repulsive: bool,
        /// Maximum force (prevents singularity near center)
        max_force: Fix128,
    },

    /// Linear drag (velocity-proportional resistance)
    Drag {
        /// Drag coefficient
        coefficient: Fix128,
    },

    /// Buoyancy (upward force below a surface level)
    Buoyancy {
        /// Water surface Y coordinate
        surface_y: Fix128,
        /// Fluid density
        density: Fix128,
        /// Fluid drag coefficient
        drag: Fix128,
    },

    /// Vortex (rotational force around an axis)
    Vortex {
        /// Vortex center position
        center: Vec3Fix,
        /// Vortex rotation axis
        axis: Vec3Fix,
        /// Rotational strength
        strength: Fix128,
        /// Radius beyond which force falls off
        falloff_radius: Fix128,
    },

    /// Explosion (radial impulse with distance falloff)
    ///
    /// Force: `strength * (1 - dist/radius)^falloff_power` directed away
    /// from `center`. Zero force beyond `radius`.
    Explosion {
        /// Explosion center position
        center: Vec3Fix,
        /// Peak force strength at center
        strength: Fix128,
        /// Maximum blast radius (zero force beyond this distance)
        radius: Fix128,
        /// Falloff exponent (1 = linear, 2 = quadratic, etc.)
        falloff_power: Fix128,
    },

    /// Magnetic dipole field approximation
    ///
    /// Simplified dipole model: force magnitude proportional to `strength / r^3`
    /// directed along the dipole axis (`moment`). The `moment` vector encodes
    /// both the dipole direction and relative magnitude.
    Magnetic {
        /// Dipole position in world space
        position: Vec3Fix,
        /// Dipole moment direction (normalized direction of the magnetic field)
        moment: Vec3Fix,
        /// Overall field strength multiplier
        strength: Fix128,
    },
}

/// A force field with optional body filter
#[derive(Clone, Debug)]
pub struct ForceFieldInstance {
    /// The force field definition
    pub field: ForceField,
    /// If Some, only affects bodies in this list. If None, affects all bodies.
    pub affected_bodies: Option<Vec<usize>>,
    /// Whether this field is active
    pub enabled: bool,
}

impl ForceFieldInstance {
    /// Create a new force field instance affecting all bodies
    #[inline]
    #[must_use]
    pub fn new(field: ForceField) -> Self {
        Self {
            field,
            affected_bodies: None,
            enabled: true,
        }
    }

    /// Restrict this field to only affect specific bodies
    #[must_use]
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
#[must_use]
pub fn compute_force(field: &ForceField, body: &RigidBody) -> Vec3Fix {
    match field {
        ForceField::Directional {
            direction,
            strength,
        } => *direction * *strength,

        ForceField::Point {
            center,
            strength,
            repulsive,
            max_force,
        } => {
            let delta = *center - body.position;
            let dist_sq = delta.length_squared();

            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }

            let dist = dist_sq.sqrt();
            let direction = delta / dist;

            // Inverse-square law: F = strength / r^2
            let force_mag = *strength / dist_sq;
            let clamped = if force_mag > *max_force {
                *max_force
            } else {
                force_mag
            };

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

        ForceField::Buoyancy {
            surface_y,
            density,
            drag,
        } => {
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

        ForceField::Vortex {
            center,
            axis,
            strength,
            falloff_radius,
        } => {
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

        ForceField::Explosion {
            center,
            strength,
            radius,
            falloff_power,
        } => {
            let delta = body.position - *center;
            let dist_sq = delta.length_squared();

            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }

            let dist = dist_sq.sqrt();

            // Beyond radius: zero force
            if dist >= *radius {
                return Vec3Fix::ZERO;
            }

            let direction = delta / dist;

            // Falloff: (1 - dist/radius)^falloff_power
            let ratio = Fix128::ONE - dist / *radius;

            // Integer-power approximation for falloff_power via repeated multiplication.
            // For non-integer powers this is a truncated approximation; sufficient
            // for game-quality explosion curves.
            let power_int = falloff_power.hi.max(0) as u32;
            let mut falloff = Fix128::ONE;
            let mut i = 0u32;
            while i < power_int {
                falloff = falloff * ratio;
                i += 1;
            }
            // If power is 0, falloff stays ONE (constant force within radius).

            direction * (*strength * falloff)
        }

        ForceField::Magnetic {
            position,
            moment,
            strength,
        } => {
            let delta = body.position - *position;
            let dist_sq = delta.length_squared();

            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }

            let dist = dist_sq.sqrt();

            // Force magnitude: strength / r^3
            let r_cubed = dist_sq * dist;

            if r_cubed.is_zero() {
                return Vec3Fix::ZERO;
            }

            let force_mag = *strength / r_cubed;

            // Force direction along dipole moment axis
            let moment_dir = moment.normalize();

            // Simplified dipole: force along moment direction, magnitude ~ 1/r^3
            // In a full dipole model the force depends on angle; here we project
            // the displacement onto the dipole axis for a directional bias.
            let alignment = delta.dot(moment_dir);

            // If body is along the dipole axis, it is attracted; perpendicular = weaker.
            // Simplified: force = strength / r^3 * dot(r_hat, m_hat) * m_hat
            // This gives attraction along the axis and zero force in the equatorial plane.
            let signed_mag = force_mag * alignment / dist;

            moment_dir * signed_mag
        }
    }
}

/// Apply all force fields to all bodies for one timestep
pub fn apply_force_fields(fields: &[ForceFieldInstance], bodies: &mut [RigidBody], dt: Fix128) {
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

#[cfg(all(test, feature = "std"))]
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
        assert!(
            force.x < Fix128::ZERO,
            "Point gravity should pull toward center"
        );
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
        assert!(
            force_above.y.is_zero() && force_above.x.is_zero(),
            "No buoyancy above surface"
        );
    }

    #[test]
    fn test_apply_force_fields() {
        let fields = vec![ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(10),
        })];

        let mut bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];

        let dt = Fix128::from_ratio(1, 60);
        apply_force_fields(&fields, &mut bodies, dt);

        // Velocity should have increased in X
        assert!(
            bodies[0].velocity.x > Fix128::ZERO,
            "Force should accelerate body"
        );
    }

    #[test]
    fn test_affected_bodies_filter() {
        let field = ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(100),
        })
        .with_affected_bodies(vec![0]); // Only affects body 0

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
        assert!(
            force.z.abs() > Fix128::ZERO,
            "Vortex should produce tangential force"
        );
    }

    // --- Explosion tests ---

    #[test]
    fn test_explosion_pushes_outward() {
        let field = ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            radius: Fix128::from_int(20),
            falloff_power: Fix128::ONE,
        };
        let body = RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        // Force should push away from center (positive X)
        assert!(
            force.x > Fix128::ZERO,
            "Explosion should push body away from center"
        );
    }

    #[test]
    fn test_explosion_zero_beyond_radius() {
        let field = ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            radius: Fix128::from_int(10),
            falloff_power: Fix128::ONE,
        };
        // Body at distance 15, radius is 10 => zero force
        let body = RigidBody::new(Vec3Fix::from_int(15, 0, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        assert!(
            force.x.is_zero() && force.y.is_zero() && force.z.is_zero(),
            "No explosion force beyond radius"
        );
    }

    #[test]
    fn test_explosion_at_center() {
        let field = ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            radius: Fix128::from_int(10),
            falloff_power: Fix128::ONE,
        };
        let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        let force = compute_force(&field, &body);
        assert!(
            force.x.is_zero() && force.y.is_zero() && force.z.is_zero(),
            "No force at explosion center (zero distance)"
        );
    }

    #[test]
    fn test_explosion_falloff() {
        let field_linear = ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            radius: Fix128::from_int(20),
            falloff_power: Fix128::ONE,
        };
        let field_quadratic = ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(100),
            radius: Fix128::from_int(20),
            falloff_power: Fix128::from_int(2),
        };

        let body = RigidBody::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let force_linear = compute_force(&field_linear, &body);
        let force_quadratic = compute_force(&field_quadratic, &body);

        // At dist=10, radius=20: ratio = 0.5
        // linear: 0.5^1 = 0.5, quadratic: 0.5^2 = 0.25
        // So quadratic force should be smaller
        assert!(
            force_quadratic.x < force_linear.x,
            "Quadratic falloff should produce weaker force than linear at same distance"
        );
    }

    #[test]
    fn test_explosion_apply_to_body() {
        let fields = vec![ForceFieldInstance::new(ForceField::Explosion {
            center: Vec3Fix::ZERO,
            strength: Fix128::from_int(1000),
            radius: Fix128::from_int(50),
            falloff_power: Fix128::ONE,
        })];

        let mut bodies = vec![RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE)];
        let dt = Fix128::from_ratio(1, 60);
        apply_force_fields(&fields, &mut bodies, dt);

        assert!(
            bodies[0].velocity.x > Fix128::ZERO,
            "Explosion should accelerate body away"
        );
    }

    // --- Magnetic tests ---

    #[test]
    fn test_magnetic_along_dipole_axis() {
        let field = ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(1000),
        };
        // Body along the dipole axis (+X)
        let body = RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        // Should produce force along X axis (dipole moment direction)
        assert!(
            force.x.abs() > Fix128::ZERO,
            "Magnetic force should exist along dipole axis"
        );
    }

    #[test]
    fn test_magnetic_perpendicular_to_dipole() {
        let field = ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(1000),
        };
        // Body perpendicular to dipole axis (along Y)
        let body = RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
        let force = compute_force(&field, &body);

        // dot(delta, moment) = 0, so force should be zero
        assert!(
            force.x.is_zero() && force.y.is_zero() && force.z.is_zero(),
            "No magnetic force perpendicular to dipole axis"
        );
    }

    #[test]
    fn test_magnetic_at_dipole_position() {
        let field = ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_Z,
            strength: Fix128::from_int(100),
        };
        let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        let force = compute_force(&field, &body);
        assert!(
            force.x.is_zero() && force.y.is_zero() && force.z.is_zero(),
            "No force at the dipole location"
        );
    }

    #[test]
    fn test_magnetic_inverse_cube_falloff() {
        let field = ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(10000),
        };

        let body_near = RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE);
        let body_far = RigidBody::new(Vec3Fix::from_int(4, 0, 0), Fix128::ONE);

        let force_near = compute_force(&field, &body_near);
        let force_far = compute_force(&field, &body_far);

        // 1/r^3: at r=2 vs r=4, force ratio should be (4/2)^3 = 8
        // force_near should be stronger than force_far
        assert!(
            force_near.x.abs() > force_far.x.abs(),
            "Magnetic force should be stronger at closer distance (1/r^3 falloff)"
        );
    }

    #[test]
    fn test_magnetic_apply_to_body() {
        let fields = vec![ForceFieldInstance::new(ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(10000),
        })];

        let mut bodies = vec![RigidBody::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE)];
        let dt = Fix128::from_ratio(1, 60);
        apply_force_fields(&fields, &mut bodies, dt);

        assert!(
            bodies[0].velocity.x.abs() > Fix128::ZERO,
            "Magnetic field should accelerate body"
        );
    }
}
