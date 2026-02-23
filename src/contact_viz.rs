//! Contact Force Visualization
//!
//! Generates visual primitives for contact forces and friction cones.
//! These can be rendered by any graphics backend to display contact
//! interaction data.
//!
//! - **Contact Arrows**: Arrows at contact points showing normal/tangent forces
//! - **Friction Cones**: Cone geometry showing the friction limit surface
//!
//! All computations use deterministic 128-bit fixed-point arithmetic.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Contact Visualization Types
// ============================================================================

/// An arrow representing a contact force.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactArrow {
    /// Contact point position
    pub position: Vec3Fix,
    /// Force direction (normalized)
    pub normal: Vec3Fix,
    /// Force magnitude
    pub force_magnitude: Fix128,
    /// True if this arrow represents a friction (tangential) force
    pub is_friction: bool,
}

/// A friction cone at a contact point.
///
/// The cone axis is aligned with the contact normal, and the half-angle
/// is determined by `arctan(friction_coefficient)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrictionCone {
    /// Contact point position (cone apex)
    pub position: Vec3Fix,
    /// Cone axis direction (contact normal)
    pub normal: Vec3Fix,
    /// Half-angle of the friction cone
    pub half_angle: Fix128,
    /// Height of the cone (proportional to normal force)
    pub height: Fix128,
}

// ============================================================================
// Arrow Generation
// ============================================================================

/// Generate contact force arrows from contact data.
///
/// Each input tuple `(position, normal, force_magnitude)` produces one
/// contact arrow. All arrows are marked as normal-force arrows.
///
/// # Arguments
///
/// - `contacts`: Slice of `(contact_point, contact_normal, force_magnitude)` tuples
#[must_use]
pub fn generate_contact_arrows(contacts: &[(Vec3Fix, Vec3Fix, Fix128)]) -> Vec<ContactArrow> {
    contacts
        .iter()
        .map(|&(position, normal, force_magnitude)| {
            let dir = normal.normalize();
            ContactArrow {
                position,
                normal: dir,
                force_magnitude,
                is_friction: false,
            }
        })
        .collect()
}

/// Generate friction force arrows from contact data.
///
/// For each contact, computes two tangential directions perpendicular to
/// the contact normal and creates arrows representing potential friction forces.
/// The friction force magnitude is bounded by `mu * F_n`.
///
/// # Arguments
///
/// - `contacts`: Slice of `(contact_point, contact_normal, normal_force)` tuples
/// - `friction_coefficient`: Coulomb friction coefficient (mu)
#[must_use]
pub fn generate_friction_arrows(
    contacts: &[(Vec3Fix, Vec3Fix, Fix128)],
    friction_coefficient: Fix128,
) -> Vec<ContactArrow> {
    let mut arrows = Vec::with_capacity(contacts.len() * 2);

    for &(position, normal, normal_force) in contacts {
        let n = normal.normalize();
        let friction_force = normal_force * friction_coefficient;

        // Compute tangent basis
        let (t1, t2) = tangent_basis(n);

        arrows.push(ContactArrow {
            position,
            normal: t1,
            force_magnitude: friction_force,
            is_friction: true,
        });

        arrows.push(ContactArrow {
            position,
            normal: t2,
            force_magnitude: friction_force,
            is_friction: true,
        });
    }

    arrows
}

// ============================================================================
// Friction Cone Generation
// ============================================================================

/// Generate friction cones from contact data.
///
/// Each cone is centered at the contact point, with its axis along the
/// contact normal. The half-angle corresponds to `arctan(mu)` and the
/// height is proportional to the normal force.
///
/// # Arguments
///
/// - `contacts`: Slice of `(contact_point, contact_normal, normal_force)` tuples
/// - `friction_coefficient`: Coulomb friction coefficient (mu)
#[must_use]
pub fn generate_friction_cones(
    contacts: &[(Vec3Fix, Vec3Fix, Fix128)],
    friction_coefficient: Fix128,
) -> Vec<FrictionCone> {
    // half_angle = atan(mu)
    let half_angle = friction_coefficient.atan();

    contacts
        .iter()
        .map(|&(position, normal, normal_force)| FrictionCone {
            position,
            normal: normal.normalize(),
            half_angle,
            height: normal_force,
        })
        .collect()
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute an orthonormal tangent basis for a given normal vector.
///
/// Returns two vectors `(t1, t2)` perpendicular to `n` and to each other.
fn tangent_basis(n: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
    // Use the axis least aligned with n for the cross product
    let abs_x = n.x.abs();
    let abs_y = n.y.abs();
    let abs_z = n.z.abs();

    let helper = if abs_x < abs_y && abs_x < abs_z {
        Vec3Fix::UNIT_X
    } else if abs_y < abs_z {
        Vec3Fix::UNIT_Y
    } else {
        Vec3Fix::UNIT_Z
    };

    let t1 = n.cross(helper).normalize();
    let t2 = n.cross(t1).normalize();
    (t1, t2)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn make_contacts() -> Vec<(Vec3Fix, Vec3Fix, Fix128)> {
        vec![
            (Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(10)),
            (
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::UNIT_Y,
                Fix128::from_int(5),
            ),
        ]
    }

    #[test]
    fn test_generate_contact_arrows_count() {
        let contacts = make_contacts();
        let arrows = generate_contact_arrows(&contacts);
        assert_eq!(arrows.len(), 2);
    }

    #[test]
    fn test_contact_arrows_not_friction() {
        let contacts = make_contacts();
        let arrows = generate_contact_arrows(&contacts);
        for arrow in &arrows {
            assert!(!arrow.is_friction);
        }
    }

    #[test]
    fn test_contact_arrows_force_magnitude() {
        let contacts = make_contacts();
        let arrows = generate_contact_arrows(&contacts);
        assert_eq!(arrows[0].force_magnitude.hi, 10);
        assert_eq!(arrows[1].force_magnitude.hi, 5);
    }

    #[test]
    fn test_contact_arrows_direction_normalized() {
        let contacts = vec![(Vec3Fix::ZERO, Vec3Fix::from_int(0, 3, 4), Fix128::ONE)];
        let arrows = generate_contact_arrows(&contacts);
        let len = arrows[0].normal.length();
        let eps = Fix128::from_ratio(1, 100);
        assert!((len - Fix128::ONE).abs() < eps);
    }

    #[test]
    fn test_generate_contact_arrows_empty() {
        let arrows = generate_contact_arrows(&[]);
        assert!(arrows.is_empty());
    }

    #[test]
    fn test_friction_cones_count() {
        let contacts = make_contacts();
        let cones = generate_friction_cones(&contacts, Fix128::from_ratio(5, 10));
        assert_eq!(cones.len(), 2);
    }

    #[test]
    fn test_friction_cones_half_angle() {
        let contacts = make_contacts();
        // Use mu=1.0 so atan(1) = PI/4 ~ 0.785, well within CORDIC precision
        let mu = Fix128::ONE;
        let cones = generate_friction_cones(&contacts, mu);
        let pi_quarter = Fix128::PI / Fix128::from_int(4);
        let eps = Fix128::from_ratio(1, 10);
        for cone in &cones {
            // half_angle = atan(1.0) should be approximately PI/4
            assert!(
                !cone.half_angle.is_negative(),
                "half_angle should be non-negative"
            );
            assert!(
                (cone.half_angle - pi_quarter).abs() < eps,
                "half_angle should be approximately PI/4"
            );
        }
    }

    #[test]
    fn test_friction_cones_height() {
        let contacts = make_contacts();
        let cones = generate_friction_cones(&contacts, Fix128::ONE);
        assert_eq!(cones[0].height.hi, 10);
        assert_eq!(cones[1].height.hi, 5);
    }

    #[test]
    fn test_friction_arrows_count() {
        let contacts = make_contacts();
        let arrows = generate_friction_arrows(&contacts, Fix128::from_ratio(3, 10));
        // 2 tangent directions per contact
        assert_eq!(arrows.len(), 4);
    }

    #[test]
    fn test_friction_arrows_are_friction() {
        let contacts = make_contacts();
        let arrows = generate_friction_arrows(&contacts, Fix128::ONE);
        for arrow in &arrows {
            assert!(arrow.is_friction);
        }
    }

    #[test]
    fn test_tangent_basis_perpendicular() {
        let n = Vec3Fix::UNIT_Y;
        let (t1, t2) = tangent_basis(n);
        // t1 should be perpendicular to n
        let dot_n_t1 = n.dot(t1);
        let eps = Fix128::from_ratio(1, 1000);
        assert!(dot_n_t1.abs() < eps);
        // t2 should be perpendicular to n
        let dot_n_t2 = n.dot(t2);
        assert!(dot_n_t2.abs() < eps);
        // t1 should be perpendicular to t2
        let dot_t1_t2 = t1.dot(t2);
        assert!(dot_t1_t2.abs() < eps);
    }
}
