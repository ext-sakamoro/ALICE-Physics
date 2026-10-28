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

// ---------------------------------------------------------------------------
// Magnetisable bodies in the field of permanent magnets
// ---------------------------------------------------------------------------

/// `μ0 / (4π)` in T·m/A (exactly `1e-7` by the SI definition before 2019, the value
/// [`EmSource::MagneticDipole`] uses)
fn mu0_over_4pi() -> Fix128 {
    Fix128::from_ratio(1, 10_000_000)
}

/// `μ0 = 4π·1e-7` T·m/A, to ten significant digits
fn mu0() -> Fix128 {
    Fix128::from_ratio(12_566_370_614, 10_000_000_000_000_000)
}

/// The field of the magnets at a point and its spatial derivative: `field` is `B` (T) and
/// `jacobian[j]` is `∂B/∂x_j` (T/m). The field of a point dipole is curl-free and
/// divergence-free there, so the Jacobian is symmetric and traceless. Samples of several magnets
/// add (`a + b`): the fields superpose, and the response of a body is taken of the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DipoleSample {
    /// The magnetic field `B` (T)
    pub field: Vec3Fix,
    /// `jacobian[j] = ∂B/∂x_j` (T/m)
    pub jacobian: [Vec3Fix; 3],
}

impl DipoleSample {
    /// No field
    pub const ZERO: Self = Self {
        field: Vec3Fix::ZERO,
        jacobian: [Vec3Fix::ZERO; 3],
    };
}

impl core::ops::Add for DipoleSample {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            field: self.field + rhs.field,
            jacobian: [
                self.jacobian[0] + rhs.jacobian[0],
                self.jacobian[1] + rhs.jacobian[1],
                self.jacobian[2] + rhs.jacobian[2],
            ],
        }
    }
}

/// The field of an ideal magnetic dipole of moment `moment` (A·m²) at `position`, and its
/// Jacobian, at `point`: `B = μ0/4π · (3(m·r̂)r̂ − m)/r³` and
/// `∂B_i/∂x_j = μ0/4π · (3(m_j r̂_i + m_i r̂_j + (m·r̂)δ_ij) − 15(m·r̂) r̂_i r̂_j)/r⁴`.
///
/// Computed with the unit vector `r̂` and the powers of `1/r`, so the terms stay of order one at
/// millimetre distances in fixed point. At the dipole itself (`r = 0`) there is no field.
#[must_use]
pub fn dipole_sample(position: Vec3Fix, moment: Vec3Fix, point: Vec3Fix) -> DipoleSample {
    let (r_hat, r) = (point - position).normalize_with_length();
    if r.is_zero() {
        return DipoleSample::ZERO;
    }
    let inv_r = Fix128::ONE / r;
    let k_r3 = mu0_over_4pi() * inv_r * inv_r * inv_r;
    let k_r4 = k_r3 * inv_r;
    let three = Fix128::from_int(3);
    let fifteen = Fix128::from_int(15);
    let m_dot_r = moment.dot(r_hat);
    let field = (r_hat * (three * m_dot_r) - moment) * k_r3;
    let r_c = [r_hat.x, r_hat.y, r_hat.z];
    let m_c = [moment.x, moment.y, moment.z];
    let column = |j: usize| -> Vec3Fix {
        let entry = |i: usize| -> Fix128 {
            let delta = if i == j { m_dot_r } else { Fix128::ZERO };
            (three * (m_c[j] * r_c[i] + m_c[i] * r_c[j] + delta)
                - fifteen * m_dot_r * r_c[i] * r_c[j])
                * k_r4
        };
        Vec3Fix::new(entry(0), entry(1), entry(2))
    };
    DipoleSample {
        field,
        jacobian: [column(0), column(1), column(2)],
    }
}

/// The magnetic response of a soft ferromagnetic body: in a field `B` it takes the moment
/// `min(chi_eff·V·|B|/μ0, saturation_magnetization·V)·B̂` (`V` its volume)
///
/// `chi_eff` is the susceptibility of the body's shape (a sphere of a material with a large
/// susceptibility χ has `3χ/(χ+3) ≈ 3`, steel 3), `saturation_magnetization` is `M_s` (A/m, iron
/// and steel about 1.7e6, which with `chi_eff = 3` saturates at `B_s ≈ 0.71 T`). A body that does
/// not magnetise (rubber, glass) has both zero.
///
/// The body is a point dipole: valid while the field changes over lengths much larger than the
/// body, approximate within about two or three body radii of a magnet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MagneticResponse {
    /// Effective (shape) susceptibility, dimensionless
    pub chi_eff: Fix128,
    /// Saturation magnetisation `M_s` (A/m)
    pub saturation_magnetization: Fix128,
}

impl MagneticResponse {
    /// The field magnitude above which the body is saturated: `B_s = μ0·M_s/χ_eff` (`None` for a
    /// body that does not magnetise)
    fn saturation_field(&self) -> Option<Fix128> {
        if self.chi_eff <= Fix128::ZERO || self.saturation_magnetization <= Fix128::ZERO {
            None
        } else {
            Some(mu0() * self.saturation_magnetization / self.chi_eff)
        }
    }
}

/// The energy (J) of a magnetisable body of volume `volume` (m³) in the field `field`:
/// `U = −χ_eff·V·|B|²/(2μ0)` below the saturation field `B_s`, `U = −M_s·V·|B| + M_s·V·B_s/2`
/// above it (equal at `B_s`). Zero for a body that does not magnetise.
#[must_use]
pub fn induced_dipole_energy(
    field: Vec3Fix,
    response: &MagneticResponse,
    volume: Fix128,
) -> Fix128 {
    let Some(b_s) = response.saturation_field() else {
        return Fix128::ZERO;
    };
    let b_sq = field.length_squared();
    let two = Fix128::from_int(2);
    if b_sq <= b_s * b_s {
        // χ_eff/(2μ0) first (about 1e6), then |B|² and V: a product of the small factors first
        // (χ_eff·V·|B|² ≈ 1e-17) would keep only a few digits in fixed point
        -(response.chi_eff / (two * mu0()) * b_sq * volume)
    } else {
        let m_sat = response.saturation_magnetization * volume;
        -(m_sat * b_sq.sqrt()) + m_sat * b_s / two
    }
}

/// The force (N) on a magnetisable body of volume `volume` (m³) where the magnets' field and its
/// Jacobian are `sample`: `F = −∇U` of [`induced_dipole_energy`], that is
/// `(χ_eff·V/μ0)·JᵀB` below the saturation field and `M_s·V·JᵀB/|B|` above it (`J` the Jacobian,
/// `JᵀB = ∇(|B|²/2)`). Zero for a body that does not magnetise.
#[must_use]
pub fn induced_dipole_force(
    sample: &DipoleSample,
    response: &MagneticResponse,
    volume: Fix128,
) -> Vec3Fix {
    let Some(b_s) = response.saturation_field() else {
        return Vec3Fix::ZERO;
    };
    let b = sample.field;
    let grad_half_b_sq = Vec3Fix::new(
        sample.jacobian[0].dot(b),
        sample.jacobian[1].dot(b),
        sample.jacobian[2].dot(b),
    );
    let b_sq = b.length_squared();
    if b_sq <= b_s * b_s {
        grad_half_b_sq * (response.chi_eff * volume / mu0())
    } else {
        grad_half_b_sq * (response.saturation_magnetization * volume / b_sq.sqrt())
    }
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
