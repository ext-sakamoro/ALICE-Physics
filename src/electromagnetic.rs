//! Lorentz-force electromagnetics for charged rigid bodies.
//!
//! Complements [`crate::force::ForceField::Vortex`] and gravity-well
//! fields with the canonical electromagnetic force
//!
//! ```text
//! F = q · (E + v × B)
//! ```
//!
//! and provides uniform-field / dipole-field / single-charge Coulomb
//! sources sufficient for gameplay-scale magnetics, mag-lev
//! demonstrations, and induction-motor toy models. Full Maxwell
//! equations, magnetic-hysteresis, and Faraday-induced EMF closures
//! are out of scope.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

/// Charged rigid body decorator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChargedBody {
    /// Underlying rigid-body index in `PhysicsWorld`.
    pub body: usize,
    /// Charge in coulombs.
    pub charge_c: Fix128,
}

impl ChargedBody {
    /// Construct with a positive or negative charge.
    #[must_use]
    pub const fn new(body: usize, charge_c: Fix128) -> Self {
        Self { body, charge_c }
    }
}

/// A source of electric / magnetic field.
#[derive(Debug, Clone, Copy)]
pub enum EmSource {
    /// Uniform vector electric or magnetic field.
    Uniform {
        /// Uniform electric field `E` (V/m).
        electric: Vec3Fix,
        /// Uniform magnetic field `B` (T).
        magnetic: Vec3Fix,
    },
    /// Point charge at `position` with strength `charge_c`, producing
    /// an inverse-square Coulomb field `E = k · q / r² · r̂` with
    /// `k = 8.99e9 V·m/C`.
    PointCharge {
        /// World-space position of the source.
        position: Vec3Fix,
        /// Charge in coulombs.
        charge_c: Fix128,
    },
    /// Ideal magnetic dipole with moment `m` at `position`. Produces
    /// `B = μ₀ / (4π) · (3(m·r̂)r̂ − m) / r³` with
    /// `μ₀ / (4π) ≈ 1e-7 T·m/A`.
    MagneticDipole {
        /// World-space position of the dipole.
        position: Vec3Fix,
        /// Dipole moment vector `m` (A·m²).
        moment: Vec3Fix,
    },
}

impl EmSource {
    /// Sample `(E, B)` at a world-space point.
    #[must_use]
    pub fn sample(&self, point: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
        match *self {
            Self::Uniform { electric, magnetic } => (electric, magnetic),
            Self::PointCharge { position, charge_c } => {
                let r = point - position;
                let r_sq = r.length_squared();
                if r_sq <= Fix128::ZERO {
                    return (Vec3Fix::ZERO, Vec3Fix::ZERO);
                }
                // Coulomb constant k ≈ 8.99e9 V·m/C.
                let k = Fix128::from_int(8_990_000_000);
                let r_mag = r_sq.sqrt();
                let scale = k * charge_c / (r_sq * r_mag);
                (r * scale, Vec3Fix::ZERO)
            }
            Self::MagneticDipole { position, moment } => {
                let r = point - position;
                let r_sq = r.length_squared();
                if r_sq <= Fix128::ZERO {
                    return (Vec3Fix::ZERO, Vec3Fix::ZERO);
                }
                let r_mag = r_sq.sqrt();
                let r_hat = r * (Fix128::ONE / r_mag);
                let m_dot_r = moment.dot(r_hat);
                let inv_r3 = Fix128::ONE / (r_sq * r_mag);
                // μ₀ / (4π) = 1e-7 T·m/A.
                let mu0_over_4pi = Fix128::from_ratio(1, 10_000_000);
                let three = Fix128::from_int(3);
                let b = (r_hat * (three * m_dot_r) - moment) * (mu0_over_4pi * inv_r3);
                (Vec3Fix::ZERO, b)
            }
        }
    }
}

/// Lorentz force `q(E + v × B)` on a charged body from a single source.
#[must_use]
pub fn lorentz_force(charged: ChargedBody, body: &RigidBody, source: &EmSource) -> Vec3Fix {
    let (e_field, b_field) = source.sample(body.position);
    let v_cross_b = body.velocity.cross(b_field);
    (e_field + v_cross_b) * charged.charge_c
}

/// Aggregate Lorentz force from a slice of sources.
#[must_use]
pub fn lorentz_force_sum(charged: ChargedBody, body: &RigidBody, sources: &[EmSource]) -> Vec3Fix {
    sources.iter().fold(Vec3Fix::ZERO, |acc, s| {
        acc + lorentz_force(charged, body, s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::solver::{BodyType, RigidBody};

    fn body_at(x: i64, y: i64, z: i64, vx: i64, vy: i64, vz: i64) -> RigidBody {
        RigidBody {
            position: Vec3Fix::new(
                Fix128::from_int(x),
                Fix128::from_int(y),
                Fix128::from_int(z),
            ),
            velocity: Vec3Fix::new(
                Fix128::from_int(vx),
                Fix128::from_int(vy),
                Fix128::from_int(vz),
            ),
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
    fn uniform_electric_field_pushes_positive_charge() {
        let source = EmSource::Uniform {
            electric: Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
            magnetic: Vec3Fix::ZERO,
        };
        let charged = ChargedBody::new(0, Fix128::ONE);
        let body = body_at(0, 0, 0, 0, 0, 0);
        let f = lorentz_force(charged, &body, &source);
        assert!(f.x > Fix128::ZERO);
        assert_eq!(f.y, Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn uniform_electric_field_pulls_negative_charge() {
        let source = EmSource::Uniform {
            electric: Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
            magnetic: Vec3Fix::ZERO,
        };
        let charged = ChargedBody::new(0, -Fix128::ONE);
        let body = body_at(0, 0, 0, 0, 0, 0);
        let f = lorentz_force(charged, &body, &source);
        assert!(f.x < Fix128::ZERO);
    }

    #[test]
    fn magnetic_field_produces_perpendicular_force() {
        // B = +z, v = +x → v×B = +x × +z = -y.
        let source = EmSource::Uniform {
            electric: Vec3Fix::ZERO,
            magnetic: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        };
        let charged = ChargedBody::new(0, Fix128::ONE);
        let body = body_at(0, 0, 0, 1, 0, 0);
        let f = lorentz_force(charged, &body, &source);
        assert!(f.y < Fix128::ZERO);
        assert_eq!(f.x, Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn stationary_body_feels_no_magnetic_force() {
        let source = EmSource::Uniform {
            electric: Vec3Fix::ZERO,
            magnetic: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        };
        let charged = ChargedBody::new(0, Fix128::ONE);
        let body = body_at(0, 0, 0, 0, 0, 0);
        let f = lorentz_force(charged, &body, &source);
        assert_eq!(f, Vec3Fix::ZERO);
    }

    #[test]
    fn point_charge_produces_radial_field() {
        let source = EmSource::PointCharge {
            position: Vec3Fix::ZERO,
            charge_c: Fix128::from_ratio(1, 1000),
        };
        let (e, b) = source.sample(Vec3Fix::new(
            Fix128::from_int(1),
            Fix128::ZERO,
            Fix128::ZERO,
        ));
        assert!(e.x > Fix128::ZERO);
        assert_eq!(e.y, Fix128::ZERO);
        assert_eq!(b, Vec3Fix::ZERO);
    }

    #[test]
    fn magnetic_dipole_axis_alignment() {
        let source = EmSource::MagneticDipole {
            position: Vec3Fix::ZERO,
            moment: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        };
        // Along the dipole axis (+z), B is aligned with the moment.
        let (_e, b) = source.sample(Vec3Fix::new(
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::from_int(1),
        ));
        assert!(b.z > Fix128::ZERO);
    }

    #[test]
    fn lorentz_force_sum_accumulates_multiple_sources() {
        let s1 = EmSource::Uniform {
            electric: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
            magnetic: Vec3Fix::ZERO,
        };
        let s2 = EmSource::Uniform {
            electric: Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            magnetic: Vec3Fix::ZERO,
        };
        let charged = ChargedBody::new(0, Fix128::ONE);
        let body = body_at(0, 0, 0, 0, 0, 0);
        let f = lorentz_force_sum(charged, &body, &[s1, s2]);
        assert_eq!(f.x, Fix128::from_int(3));
    }
}
