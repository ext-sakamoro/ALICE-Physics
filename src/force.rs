//! Custom Force Fields and Gravity
//!
//! Apply custom forces, directional gravity, point gravity, wind, drag,
//! and other force fields to rigid bodies.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Force field type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub const fn new(field: ForceField) -> Self {
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
        self.affected_bodies
            .as_ref()
            .is_none_or(|list| list.contains(&body_index))
    }
}

/// `ForceField::Point` の `|delta|² ≥ 2⁶³` (距離 ≥ 2³¹·⁵ ≈ 3.04e9) の経路
///
/// `r²` を作らずに `(strength / r) / r` で大きさを、2 乗を経ない正規化で向きを
/// 求める 範囲内の式と bit 一致はしないが、範囲内でこの経路は通らない
///
/// 距離そのものが表せない (`r ≥ 2⁶³`) 時は半分の長さで計算して 1/4 にする
/// (`|F| < 2⁻⁶²` なので結果は 0 か数 ulp)
fn point_force_beyond_square_range(
    delta: Vec3Fix,
    strength: Fix128,
    repulsive: bool,
    max_force: Fix128,
) -> Vec3Fix {
    let Some(direction) = delta.try_normalize_scaled() else {
        return Vec3Fix::ZERO;
    };
    let force_mag = match delta.checked_length_scaled() {
        Some(r) => (strength / r) / r,
        None => {
            let two = Fix128::from_int(2);
            let half = Vec3Fix::new(delta.x / two, delta.y / two, delta.z / two);
            match half.checked_length_scaled() {
                Some(h) => ((strength / h) / h) / Fix128::from_int(4),
                // |delta| ≤ √3·2⁶³ なので |delta / 2| < 2⁶³ で、ここには来ない
                None => Fix128::ZERO,
            }
        }
    };
    // 範囲外では r² ≥ 2⁶³ > strength / max_force なので、近距離側の下限
    // (`dist_sq_for_force`) は掛からない 上限の比較は範囲内と同じ
    let clamped = if force_mag > max_force {
        max_force
    } else {
        force_mag
    };
    if repulsive {
        -direction * clamped
    } else {
        direction * clamped
    }
}

/// Compute force from a force field on a body at a given position
#[must_use]
pub fn compute_force(field: &ForceField, body: &RigidBody) -> Vec3Fix {
    match field {
        ForceField::Directional {
            direction,
            strength,
        } => {
            // `direction` is documented as normalized; a non-unit one is
            // normalized here rather than scaling the force by its length
            if direction.length_squared() == Fix128::ONE {
                *direction * *strength
            } else {
                direction.normalize() * *strength
            }
        }

        ForceField::Point {
            center,
            strength,
            repulsive,
            max_force,
        } => {
            let delta = *center - body.position;

            if delta == Vec3Fix::ZERO {
                return Vec3Fix::ZERO;
            }

            // |delta| ≥ 2^31.5 では `length_squared` が wrap し、力が 0 か上限値に
            // 飛んでいた 範囲内は従来の式のまま (決定論の固定値と bit 互換)、
            // 2 乗が範囲外の時だけ 2 乗を経ない式に切り替える
            let Some(dist_sq) = delta.checked_length_squared() else {
                return point_force_beyond_square_range(delta, *strength, *repulsive, *max_force);
            };

            // Squaring underflows `dist_sq` to exactly zero once
            // |delta| < ~2.33e-10, even though `delta` itself is not the
            // zero vector — `delta / dist_sq.sqrt()` would then silently
            // give the zero direction instead of pointing toward the cap.
            let direction = if dist_sq < Fix128::ONE {
                // Below 1.0, `dist_sq` has progressively fewer representable
                // bits (exactly zero once |delta| < ~2.33e-10, barely a bit
                // or two of precision just above that), which would hand
                // `sqrt` a result good to only a digit or so. Doubling delta
                // (an exact bit shift, no rounding) until its squared length
                // is comfortably large restores full precision; the scale
                // factor cancels out exactly in the division below.
                let mut scaled = delta;
                let mut scaled_dist_sq = dist_sq;
                for _ in 0..128 {
                    if scaled_dist_sq >= Fix128::ONE {
                        break;
                    }
                    scaled = scaled + scaled;
                    scaled_dist_sq = scaled.length_squared();
                }
                scaled / scaled_dist_sq.sqrt()
            } else {
                delta / dist_sq.sqrt()
            };

            // Floor dist_sq at the value where strength / dist_sq == max_force
            // *before* dividing, so the division itself cannot overflow
            // Fix128 near the singularity. Computing the unclamped quotient
            // first and comparing afterwards lets it wrap (e.g. d = 3.2e-9
            // gives +8.4e18 — the wrong sign and far past the cap — instead
            // of the cap).
            let dist_sq_for_force = if max_force.is_zero() {
                dist_sq
            } else {
                let floor = (*strength / *max_force).abs();
                if dist_sq < floor {
                    floor
                } else {
                    dist_sq
                }
            };

            // Inverse-square law: F = strength / r^2
            let force_mag = *strength / dist_sq_for_force;
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

            // `ratio^power` by `Fix128::powf_pos`: a whole power up to 64 is
            // left-to-right multiplication (the same steps as before), a
            // fractional part follows the documented formula, and a large
            // power takes O(log n) squarings (the loop used to truncate 0.5 to
            // 0 and run 2e9 times for 2e9). Power 0 is a constant force within
            // the radius.
            let power = if falloff_power.is_negative() {
                Fix128::ZERO
            } else {
                *falloff_power
            };
            let falloff = ratio.powf_pos(power);

            direction * (*strength * falloff)
        }

        ForceField::Magnetic {
            position,
            moment,
            strength,
        } => {
            let delta = body.position - *position;
            let dist_sq = delta.length_squared();

            if delta == Vec3Fix::ZERO {
                return Vec3Fix::ZERO;
            }

            let dist = dist_sq.sqrt();

            // Force magnitude: strength / r^3, floored at the r^3 that would
            // make the quotient reach a saturation cap well inside Fix128's
            // range. Computing r^3 directly as `dist_sq * dist` underflows to
            // exactly zero once d^3 drops below the 2^-64 resolution floor
            // (unlike the Point field's `strength / r^2`, which has an
            // explicit `max_force`, Magnetic has no cap at all) — dividing by
            // that zero then collapsed the force straight back down to zero
            // right where it should instead saturate near its maximum.
            //
            // The cap is the true representable maximum divided down by a
            // generous margin (2^10), not the maximum itself: `r_cubed_floor`
            // below is computed by one Fix128 division (which truncates) and
            // `force_mag` by another, so round-tripping through the exact
            // maximum can overshoot it by a rounding unit and wrap — turning
            // a saturated-but-finite force into a huge negative one. The
            // margin absorbs that without changing the "effectively capped"
            // behavior the oracle checks for.
            let r_cubed = dist_sq * dist;
            let saturation_cap = Fix128::from_raw(i64::MAX, u64::MAX) / Fix128::from_int(1024);
            let r_cubed_floor = strength.abs() / saturation_cap;
            let r_cubed = if r_cubed < r_cubed_floor {
                r_cubed_floor
            } else {
                r_cubed
            };

            let force_mag = *strength / r_cubed;

            // Force direction along dipole moment axis
            let moment_dir = moment.normalize();

            // cos(theta) between delta and the dipole axis (the "alignment"
            // below, divided by the true distance). `dist_sq.sqrt()` loses
            // relative precision once `dist_sq` is close to the 2^-64
            // resolution floor, and multiplying that imprecision into a
            // `force_mag` already near Fix128's maximum magnitude turns a
            // few-ppm ratio error into an absolute swing large enough to
            // break monotonicity as d shrinks (two capped distances that
            // should give the identical saturated force instead differ by
            // ~1e10 out of ~9.2e18). Doubling delta (exact bit shifts, no
            // rounding) until its squared length is comfortably large
            // restores full precision before forming the ratio; the scale
            // factor cancels out exactly between the dot product and the
            // distance it is divided by.
            let (delta_p, dist_sq_p) = if dist_sq < Fix128::ONE {
                let mut scaled = delta;
                let mut scaled_dist_sq = dist_sq;
                for _ in 0..128 {
                    if scaled_dist_sq >= Fix128::ONE {
                        break;
                    }
                    scaled = scaled + scaled;
                    scaled_dist_sq = scaled.length_squared();
                }
                (scaled, scaled_dist_sq)
            } else {
                (delta, dist_sq)
            };
            let cos_theta = delta_p.dot(moment_dir) / dist_sq_p.sqrt();

            // Simplified dipole: force along moment direction, magnitude ~ 1/r^3
            // In a full dipole model the force depends on angle; here we project
            // the displacement onto the dipole axis for a directional bias.
            // If body is along the dipole axis, it is attracted; perpendicular = weaker.
            // Simplified: force = strength / r^3 * dot(r_hat, m_hat) * m_hat
            // This gives attraction along the axis and zero force in the equatorial plane.
            let signed_mag = force_mag * cos_theta;

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

        body.velocity = force_field_velocity(fields, body_idx, body, dt);
    }
}

/// The velocity [`apply_force_fields`] gives body `body_idx` (`body` itself is
/// not changed): `v + (Σ F · inv_mass) · dt` over the fields that affect it.
#[must_use]
pub(crate) fn force_field_velocity(
    fields: &[ForceFieldInstance],
    body_idx: usize,
    body: &RigidBody,
    dt: Fix128,
) -> Vec3Fix {
    let mut total_force = Vec3Fix::ZERO;

    for field_inst in fields {
        if !field_inst.affects(body_idx) {
            continue;
        }
        total_force = total_force + compute_force(&field_inst.field, body);
    }

    // F = ma, a = F * inv_mass, v += a * dt
    let acceleration = total_force * body.inv_mass;
    body.velocity + acceleration * dt
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

    fn pow2(e: i32) -> Fix128 {
        if e >= 0 {
            Fix128::from_int(1_i64 << e)
        } else {
            Fix128::ONE / Fix128::from_int(1_i64 << -e)
        }
    }

    fn body_at(x: Fix128) -> RigidBody {
        RigidBody::new(Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO), Fix128::ONE)
    }

    fn point(strength: Fix128, repulsive: bool, max_force: Fix128) -> ForceField {
        ForceField::Point {
            center: Vec3Fix::ZERO,
            strength,
            repulsive,
            max_force,
        }
    }

    /// oracle: past the square range (`r = 2³²`, `r² = 2⁶⁴`) the point force is
    /// still `s/r²` toward the centre: `2⁶²/2⁶⁴ = 1/4` along `−x` for a body at
    /// `+x` (`+x` when repulsive), capped at `max_force = 1/8`.
    #[test]
    fn point_force_beyond_the_square_range() {
        let b = body_at(pow2(32));
        let s = pow2(62);
        let quarter = Fix128::from_ratio(1, 4);
        assert_eq!(
            compute_force(&point(s, false, pow2(10)), &b),
            Vec3Fix::new(-quarter, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            compute_force(&point(s, true, pow2(10)), &b),
            Vec3Fix::new(quarter, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            compute_force(&point(s, false, Fix128::from_ratio(1, 8)), &b),
            Vec3Fix::new(-Fix128::from_ratio(1, 8), Fix128::ZERO, Fix128::ZERO)
        );
    }

    /// oracle: when even the distance does not fit (`|δ| = 3·2⁶¹·√3 > 2⁶³`)
    /// the force `s/r² < 2⁶²/2¹²⁴` is below `2⁻⁶⁰` and points toward the
    /// centre (no component away from it).
    #[test]
    fn point_force_when_the_distance_does_not_fit() {
        let c = Fix128::from_int(3_i64 << 61);
        let field = ForceField::Point {
            center: Vec3Fix::new(c, c, c),
            strength: pow2(62),
            repulsive: false,
            max_force: pow2(10),
        };
        let f = compute_force(&field, &body_at(Fix128::ZERO));
        for v in [f.x, f.y, f.z] {
            assert!(v >= Fix128::ZERO && v < pow2(-60), "{f:?}");
        }
    }

    /// oracle: near the centre (`r = 2⁻²⁰`, `r² = 2⁻⁴⁰ < 1`) the direction is
    /// still exactly `−x` and the force `s/r² = 2⁴⁰` for `s = 1` under a cap
    /// of `2⁵⁰`; with a cap of `2¹⁰` the floor `s/cap = 2⁻¹⁰ > r²` makes it
    /// exactly the cap; a cap of 0 gives no force; at the centre there is
    /// none either.
    #[test]
    fn point_force_near_the_centre() {
        let b = body_at(pow2(-20));
        let on_x = |v: Fix128| Vec3Fix::new(v, Fix128::ZERO, Fix128::ZERO);
        assert_eq!(
            compute_force(&point(Fix128::ONE, false, pow2(50)), &b),
            on_x(-pow2(40))
        );
        assert_eq!(
            compute_force(&point(Fix128::ONE, false, pow2(10)), &b),
            on_x(-pow2(10))
        );
        assert_eq!(
            compute_force(&point(Fix128::ONE, false, Fix128::ZERO), &b),
            Vec3Fix::ZERO
        );
        assert_eq!(
            compute_force(&point(Fix128::ONE, false, pow2(10)), &body_at(Fix128::ZERO)),
            Vec3Fix::ZERO
        );
    }

    /// oracle: below the surface `y = 0` at depth 2 with density 3 the
    /// buoyancy is `(0, 6, 0)`; moving at `(1, 0, 2)` with drag 1/2 adds
    /// `−v/2`. A body on the vortex axis feels nothing; neither does one at
    /// the centre of a magnet.
    #[test]
    fn buoyancy_drag_vortex_axis_and_magnet_centre() {
        let mut b = RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::ONE);
        b.velocity = Vec3Fix::from_int(1, 0, 2);
        let water = ForceField::Buoyancy {
            surface_y: Fix128::ZERO,
            density: Fix128::from_int(3),
            drag: Fix128::from_ratio(1, 2),
        };
        assert_eq!(
            compute_force(&water, &b),
            Vec3Fix::new(-Fix128::from_ratio(1, 2), Fix128::from_int(6), -Fix128::ONE)
        );
        let vortex = ForceField::Vortex {
            center: Vec3Fix::ZERO,
            axis: Vec3Fix::UNIT_Y,
            strength: Fix128::ONE,
            falloff_radius: Fix128::ONE,
        };
        assert_eq!(compute_force(&vortex, &b), Vec3Fix::ZERO);
        let magnet = ForceField::Magnetic {
            position: b.position,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::ONE,
        };
        assert_eq!(compute_force(&magnet, &b), Vec3Fix::ZERO);
    }

    /// oracle: on the dipole axis at `r = 2⁻¹⁰` (`r² < 1`) the magnetic force
    /// is `s/r³·cos θ` along the moment with `cos θ = 1`: `2³⁰` for `s = 1`.
    #[test]
    fn magnetic_force_near_the_dipole() {
        let magnet = ForceField::Magnetic {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::UNIT_X,
            strength: Fix128::ONE,
        };
        let f = compute_force(&magnet, &body_at(pow2(-10)));
        assert!(
            (f.x.to_f64() / 1_073_741_824.0 - 1.0).abs() < 1e-12,
            "{f:?}"
        );
        assert!(f.y.is_zero() && f.z.is_zero());
    }
}
