//! Lorentz-force electromagnetics for charged rigid bodies, the field of permanent magnets, the
//! force on magnetisable bodies in it, and the surface of a magnetisable liquid at rest in it.
//!
//! # MODEL
//!
//! - **Equations**: the Lorentz force `F = q(E + v × B)` on a charged body (`lorentz_force`,
//!   `lorentz_force_sum`, sources from `EmSource::sample`); the field of a point magnetic dipole
//!   `B = μ0/4π · (3(m·r̂)r̂ − m)/r³` and its Jacobian (`dipole_sample`); a soft magnetisable body
//!   of moment `min(χ_eff·V·|B|/μ0, M_s·V)·B̂`, energy `U = −χ_eff·V·|B|²/(2μ0)` below
//!   `B_s = μ0·M_s/χ_eff` and `−M_s·V·|B| + M_s·V·B_s/2` above (`induced_dipole_energy`), force
//!   `F = −∇U` (`induced_dipole_force`); a magnetisable liquid at rest: the ferrohydrostatic
//!   Bernoulli condition at its free surface (Rosensweig), `ρ g h = μ0·M̄·H + μ0·M_n²/2` with `H`
//!   inside the liquid from the boundary conditions of a flat layer (`liquid_field`: tangential `H`
//!   and normal `B` continuous, `M = min(χ|H|, M_s)·Ĥ`), so below saturation
//!   `h = χ/(2μ0 ρ g)·(B_t² + B_n²/(1+χ))` (`magnetic_fluid_traction`,
//!   `magnetic_fluid_surface_rise`).
//! - **Units**: SI. q C, E V/m, B T, v m/s, F N, m A·m², r m, V m³, M_s A/m, U J, p* Pa, ρ kg/m³,
//!   g m/s², h m; `μ0/4π = 1e-7` T·m/A (`mu0_over_4pi`), `μ0` to ten digits (`mu0`).
//! - **Conservation**: the fields are prescribed sources with no state (`EmSource`,
//!   `dipole_sample`); the magnetic force on a magnetisable body is conservative (`−∇U`,
//!   `induced_dipole_force`); the Lorentz force of a static `B` does no work
//!   (`lorentz_force`). The liquid's rise is a static balance (`magnetic_fluid_surface_rise`):
//!   no flow, nothing is conserved or dissipated over time, the volume is the caller's
//!   (`magnetic_fluid_traction`).
//! - **Range**: point dipoles, valid at distances much larger than the magnet and the body
//!   (`dipole_sample`, no field at `r = 0`); quasi-static fields only (`lorentz_force`); the
//!   liquid's rise holds for any `χ` and saturation below the Rosensweig threshold, and is `None` for
//!   a non-positive density or gravity (`magnetic_fluid_surface_rise`).
//! - **Approximations**: no full Maxwell equations, no induction (Faraday), no hysteresis
//!   (`EmSource::sample`); a magnetisable body is a point dipole with a linear-then-saturating
//!   magnetisation (`MagneticResponse`); the liquid is a wide flat layer: the field just above its
//!   surface is the magnets' applied field at the undisturbed level (its own magnetisation does not
//!   change the field outside it; a narrow dish or a tall mound breaks this), the surface normal
//!   points out of the liquid into the air (`liquid_field`), its surface has no tension and raises
//!   no spikes (the Rosensweig instability is not modelled), and the field is not recomputed on the
//!   raised surface (`magnetic_fluid_surface_rise`).
//! - **Validation**: `tests/analytic_magnet_induced_dipole.rs` (the dipole field, the energy and
//!   force in both regimes, `F = −∇U` by central difference),
//!   `tests/analytic_magnetic_fluid_surface.rs` (the flat layer's closed forms at χ = 1 for a
//!   normal, a tangential and an oblique field, the χ → 0 limit, the saturated normal-field closed
//!   form, continuity at the saturation field, the saturated solver with a huge `M_s` against the
//!   linear closed form, two magnets superpose their fields before squaring, a liquid that does not
//!   magnetise or has no weight), `the_dipole_field_at_three_points_of_known_geometry`,
//!   `the_jacobian_is_the_derivative_of_the_field_symmetric_and_traceless`,
//!   `uniform_electric_field_pushes_positive_charge`, `magnetic_field_produces_perpendicular_force`,
//!   `the_magnetic_liquid_traction_and_rise`.
//! - **Tiers**: L0 only.
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

// ---------------------------------------------------------------------------
// A magnetisable liquid at rest in the field of permanent magnets
// ---------------------------------------------------------------------------

/// The field inside a magnetisable liquid just under its surface, and its magnetisation there, where the magnets'
/// field just above the surface is `field` (`B₀`, T) and the surface's unit normal is `normal` (out of the liquid).
/// The liquid fills the half space under a flat surface: across it the tangential `H` and the normal `B` are
/// continuous, so `H_t = B₀_t/μ0` and `H_n + M_n = B₀_n/μ0`, with `M = min(χ·|H|, M_s)·Ĥ` (`χ` is
/// `response.chi_eff`, the liquid's susceptibility; `M_s` its saturation magnetisation). Below saturation
/// `H_n = B₀_n/(μ0(1+χ))`; above it the normal condition `x + M_s·x/√(H_t² + x²) = B₀_n/μ0` is solved for `x = H_n`
/// by bisection (strictly increasing in `x`, one root between 0 and `B₀_n/μ0`). Returns `(|H|, M_n)` in A/m.
fn liquid_field(field: Vec3Fix, normal: Vec3Fix, response: &MagneticResponse) -> (Fix128, Fix128) {
    let h0 = field * (Fix128::ONE / mu0());
    let h0n = h0.dot(normal);
    let t = h0.length_squared() - h0n * h0n;
    let h0t_sq = if t.is_negative() { Fix128::ZERO } else { t };
    let chi = response.chi_eff;
    if chi <= Fix128::ZERO || response.saturation_magnetization <= Fix128::ZERO {
        return (h0.length(), Fix128::ZERO);
    }
    let m_s = response.saturation_magnetization;
    let h_s = m_s / chi;
    // linear: self-consistent while |H| stays at or below the saturation field
    let hn = h0n / (Fix128::ONE + chi);
    let h_lin = (h0t_sq + hn * hn).sqrt();
    if h_lin <= h_s {
        return (h_lin, chi * hn);
    }
    // saturated: |M| = M_s along H; solve the normal condition for |H_n| (H_n has the sign of B₀_n)
    let target = h0n.abs();
    let g = |x: Fix128| -> Fix128 {
        let len = (h0t_sq + x * x).sqrt();
        let m_n = if len.is_zero() {
            Fix128::ZERO
        } else {
            m_s * x / len
        };
        x + m_n - target
    };
    let (mut lo, mut hi) = (Fix128::ZERO, target);
    for _ in 0..96 {
        let mid = (lo + hi) * Fix128::from_ratio(1, 2);
        if g(mid) < Fix128::ZERO {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let x = (lo + hi) * Fix128::from_ratio(1, 2);
    let len = (h0t_sq + x * x).sqrt();
    let m_n = if len.is_zero() {
        Fix128::ZERO
    } else {
        m_s * x / len
    };
    let m_n = if h0n.is_negative() { -m_n } else { m_n };
    (len, m_n)
}

/// The magnetic traction (Pa) that lifts the free surface of a magnetisable liquid at rest, where the magnets' field
/// just above the surface is `field` (T) and the surface's unit normal is `normal` (out of the liquid): the
/// ferrohydrostatic Bernoulli condition at the interface (Rosensweig) gives the fluid magnetic pressure
/// `μ0·M̄·H = μ0∫₀^|H| M dH'` (with `H` inside the liquid, `M̄` the field-averaged magnetisation) plus the magnetic
/// normal traction `μ0·M_n²/2`. Below saturation this is `χ/(2μ0)·(B₀_t² + B₀_n²/(1+χ))`; above the saturation field
/// `H_s = M_s/χ` the pressure is `μ0(M_s·|H| − M_s·H_s/2)`. Zero for a liquid that does not magnetise. The liquid is a
/// flat layer (see [`liquid_field`]): a surface that bends strongly over the field's length scale is outside it.
#[must_use]
pub fn magnetic_fluid_traction(
    field: Vec3Fix,
    normal: Vec3Fix,
    response: &MagneticResponse,
) -> Fix128 {
    if response.chi_eff <= Fix128::ZERO || response.saturation_magnetization <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    let (h, m_n) = liquid_field(field, normal, response);
    let m_s = response.saturation_magnetization;
    let h_s = m_s / response.chi_eff;
    let two = Fix128::from_int(2);
    let mu = mu0();
    // μ0∫₀^H M dH': χH²/2 below H_s, M_s·H − M_s·H_s/2 above (equal at H_s)
    let pressure = if h <= h_s {
        mu * response.chi_eff * h * h / two
    } else {
        mu * (m_s * h - m_s * h_s / two)
    };
    pressure + mu * m_n * m_n / two
}

/// How far (m) the free surface of a magnetisable liquid at rest stands above its level where there is no field, at
/// a point of the surface where the magnets' field is `field` and the surface's unit normal is `normal`: the
/// ferrohydrostatic Bernoulli equation `ρ·g·h = (magnetic traction)` along the surface, with the traction of
/// [`magnetic_fluid_traction`]. Several magnets: add their fields (or their [`DipoleSample`]s) first, then take the
/// rise of the sum; rises do not add.
///
/// `density` is ρ (kg/m³), `gravity` is `|g|` (m/s²). `None` when either is not positive (no liquid at rest to
/// stand). The level where there is no field is the caller's: in a closed dish the liquid drawn up under a magnet
/// comes from the rest, so the caller shifts the whole surface to keep the volume.
///
/// Valid below the Rosensweig threshold (a smooth mound, no spikes): the surface's own tension and the instability
/// that raises spikes above the critical magnetisation are not in this equation. The field is the magnets' field at
/// the point the caller gives (usually the undisturbed surface), not recomputed on the raised surface.
#[must_use]
pub fn magnetic_fluid_surface_rise(
    field: Vec3Fix,
    normal: Vec3Fix,
    response: &MagneticResponse,
    density: Fix128,
    gravity: Fix128,
) -> Option<Fix128> {
    if density <= Fix128::ZERO || gravity <= Fix128::ZERO {
        return None;
    }
    Some(magnetic_fluid_traction(field, normal, response) / (density * gravity))
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
            linear_damping_rate: Fix128::ZERO,
            angular_damping_rate: Fix128::ZERO,
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

/// Values the unit tests of the magnet law share (this module and [`crate::force`])
#[cfg(test)]
pub(crate) mod magnet_test_support {
    use super::MagneticResponse;
    use crate::math::{Fix128, Vec3Fix};

    /// Soft steel: `chi_eff = 3`, `M_s = 1.7e6` A/m
    pub(crate) fn steel() -> MagneticResponse {
        MagneticResponse {
            chi_eff: Fix128::from_int(3),
            saturation_magnetization: Fix128::from_int(1_700_000),
        }
    }

    /// `(x, y, z) / denom` as a vector
    pub(crate) fn v(x: i64, y: i64, z: i64, denom: i64) -> Vec3Fix {
        Vec3Fix::new(
            Fix128::from_ratio(x, denom),
            Fix128::from_ratio(y, denom),
            Fix128::from_ratio(z, denom),
        )
    }

    /// `got` within `rel` of `want` (relative to `scale`, the size of the quantity compared)
    pub(crate) fn close(got: Fix128, want: f64, scale: f64, rel: f64) -> bool {
        (got.to_f64() - want).abs() <= rel * scale
    }
}

#[cfg(test)]
mod magnet_tests {
    use super::magnet_test_support::{close, steel, v};
    use super::*;

    /// `μ0` in f64, as the code takes it
    const MU0: f64 = 1.256_637_061_4e-6;

    /// The point-dipole field at three points of known geometry (`m = ẑ`, `r = 0.1` m, so
    /// `μ0/4π · |m|/r³ = 1e-4` T): on the axis `2e-4 ẑ`, on the equator `−1e-4 ẑ`, and at
    /// `r̂ = (0.6, 0, 0.8)`, where `m·r̂ = 0.8`, `1e-4·(3·0.8·r̂ − ẑ) = (1.44e-4, 0, 0.92e-4)`.
    #[test]
    fn the_dipole_field_at_three_points_of_known_geometry() {
        let m = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        for (point, want) in [
            (v(0, 0, 1, 10), [0.0, 0.0, 2e-4]),
            (v(1, 0, 0, 10), [0.0, 0.0, -1e-4]),
            (v(6, 0, 8, 100), [1.44e-4, 0.0, 0.92e-4]),
        ] {
            let b = dipole_sample(Vec3Fix::ZERO, m, point).field;
            for (got, want) in [b.x, b.y, b.z].into_iter().zip(want) {
                assert!(close(got, want, 2e-4, 1e-12), "{point:?}: {b:?}");
            }
        }
        // shifting the magnet and the point together leaves the field unchanged
        let shift = v(3, -7, 2, 1);
        let moved = dipole_sample(shift, m, shift + v(6, 0, 8, 100)).field;
        assert!(close(moved.x, 1.44e-4, 2e-4, 1e-12) && close(moved.z, 0.92e-4, 2e-4, 1e-12));
    }

    /// `jacobian[j]` is `∂B/∂x_j`: it matches the central difference of the field, and is
    /// symmetric and traceless (a curl- and divergence-free field)
    #[test]
    fn the_jacobian_is_the_derivative_of_the_field_symmetric_and_traceless() {
        let (pos, m) = (v(1, 2, -1, 100), v(3, 5, -2, 10));
        let p = v(4, -2, 5, 100);
        let s = dipole_sample(pos, m, p);
        let h = Fix128::from_ratio(1, 1_000_000);
        let axes = [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z];
        let comp = |u: Vec3Fix, i: usize| [u.x, u.y, u.z][i].to_f64();
        let scale = (0..3)
            .flat_map(|j| (0..3).map(move |i| (i, j)))
            .map(|(i, j)| comp(s.jacobian[j], i).abs())
            .fold(0.0, f64::max);
        assert!(scale > 0.0);
        for (j, axis) in axes.iter().enumerate() {
            let plus = dipole_sample(pos, m, p + *axis * h).field;
            let minus = dipole_sample(pos, m, p - *axis * h).field;
            for i in 0..3 {
                let fd = (comp(plus, i) - comp(minus, i)) / 2e-6;
                let got = comp(s.jacobian[j], i);
                assert!(
                    (got - fd).abs() <= 1e-6 * scale,
                    "dB_{i}/dx_{j}: {got} vs {fd}"
                );
                let mirror = comp(s.jacobian[i], j);
                assert!(
                    (got - mirror).abs() <= 1e-15 * scale,
                    "not symmetric at ({i}, {j})"
                );
            }
        }
        // the three diagonal entries are each of the size of `scale` and cancel, so the sum keeps
        // the rounding of each (measured 3.7e-15 of `scale` here); 1e-13 still fails any error in a
        // term of the closed form, which moves the trace by a fraction of `scale`
        let trace = comp(s.jacobian[0], 0) + comp(s.jacobian[1], 1) + comp(s.jacobian[2], 2);
        assert!(
            trace.abs() <= 1e-13 * scale,
            "trace {trace} of scale {scale}"
        );
    }

    /// `J = Jᵀ` at points all around a dipole of a tilted moment (the field is curl-free): each
    /// pair of mirrored entries agrees to the rounding of the closed form, 1e-15 of the largest entry
    #[test]
    fn the_jacobian_is_symmetric_all_around_the_dipole() {
        let m = v(-2, 7, 3, 10);
        let mut compared = 0;
        for point in [
            v(5, 0, 0, 100),
            v(0, -5, 0, 100),
            v(0, 0, 5, 100),
            v(3, 4, -12, 100),
            v(-7, 1, 2, 1000),
            v(9, -9, 9, 10),
        ] {
            let s = dipole_sample(v(1, 1, 1, 1000), m, v(1, 1, 1, 1000) + point);
            let e = |i: usize, j: usize| {
                [s.jacobian[j].x, s.jacobian[j].y, s.jacobian[j].z][i].to_f64()
            };
            let scale = (0..9).map(|k| e(k / 3, k % 3).abs()).fold(0.0, f64::max);
            assert!(scale > 0.0, "{point:?}");
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                assert!(
                    (e(i, j) - e(j, i)).abs() <= 1e-15 * scale,
                    "{point:?}: ({i}, {j})"
                );
                compared += 1;
            }
        }
        assert_eq!(compared, 18);
    }

    /// At the dipole itself there is no field, and samples add field by field and column by column
    #[test]
    fn no_field_at_the_dipole_and_samples_add_componentwise() {
        let m = v(1, 2, 3, 1);
        assert_eq!(
            dipole_sample(v(1, 1, 1, 1), m, v(1, 1, 1, 1)),
            DipoleSample::ZERO
        );
        let a = dipole_sample(Vec3Fix::ZERO, m, v(1, 0, 0, 10));
        let b = dipole_sample(v(0, 1, 0, 10), m, v(1, 0, 0, 10));
        let sum = a + b;
        assert_eq!(sum.field, a.field + b.field);
        for j in 0..3 {
            assert_eq!(sum.jacobian[j], a.jacobian[j] + b.jacobian[j]);
        }
        assert_eq!(a + DipoleSample::ZERO, a);
    }

    /// `B_s = μ0·M_s/χ_eff` (about 0.712 T for steel); a body with no susceptibility or no
    /// saturation magnetisation (zero or negative) does not magnetise
    #[test]
    fn the_saturation_field_and_the_bodies_that_do_not_magnetise() {
        let b_s = steel().saturation_field().unwrap();
        assert!(close(b_s, MU0 * 1.7e6 / 3.0, 1.0, 1e-12), "{b_s:?}");
        let field = v(1, 0, 0, 1);
        let sample = DipoleSample {
            field,
            jacobian: [v(1, 0, 0, 1); 3],
        };
        for (chi, ms) in [(0, 1_700_000), (3, 0), (-3, 1_700_000), (3, -1)] {
            let r = MagneticResponse {
                chi_eff: Fix128::from_int(chi),
                saturation_magnetization: Fix128::from_int(ms),
            };
            assert_eq!(r.saturation_field(), None, "({chi}, {ms})");
            assert_eq!(induced_dipole_energy(field, &r, Fix128::ONE), Fix128::ZERO);
            assert_eq!(
                induced_dipole_force(&sample, &r, Fix128::ONE),
                Vec3Fix::ZERO
            );
        }
    }

    /// The energy below saturation is `−χ_eff·V·|B|²/(2μ0)` and above it
    /// `−M_s·V·|B| + M_s·V·B_s/2`; at `|B| = B_s` the two agree (the linear branch is taken)
    #[test]
    fn the_energy_in_the_linear_and_the_saturated_regime() {
        let volume = Fix128::from_ratio(1, 1_000_000);
        let b_s = MU0 * 1.7e6 / 3.0;
        // |B| = 0.1 T along an oblique direction: |(0.06, 0, 0.08)| = 0.1
        let linear = induced_dipole_energy(v(6, 0, 8, 100), &steel(), volume);
        let want = -3.0 * 1e-6 * 0.01 / (2.0 * MU0);
        assert!(
            close(linear, want, want.abs(), 1e-9),
            "{linear:?} vs {want}"
        );
        // |B| = 1 T
        let saturated = induced_dipole_energy(v(0, 1, 0, 1), &steel(), volume);
        let want = -1.7e6 * 1e-6 + 1.7e6 * 1e-6 * b_s / 2.0;
        assert!(
            close(saturated, want, want.abs(), 1e-9),
            "{saturated:?} vs {want}"
        );
        // exactly at the saturation field both formulas give −M_s·V·B_s/2
        let at = induced_dipole_energy(
            Vec3Fix::new(
                steel().saturation_field().unwrap(),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            &steel(),
            volume,
        );
        let want = -1.7e6 * 1e-6 * b_s / 2.0;
        assert!(close(at, want, want.abs(), 1e-9), "{at:?} vs {want}");
    }

    /// The force is `(χ_eff·V/μ0)·JᵀB` below saturation and `M_s·V·JᵀB/|B|` above it, for a
    /// sample written by hand (`J = diag(2, −1, −1)` T/m, traceless)
    #[test]
    fn the_force_in_the_linear_and_the_saturated_regime() {
        let volume = Fix128::from_ratio(1, 1_000_000);
        let jacobian = [v(2, 0, 0, 1), v(0, -1, 0, 1), v(0, 0, -1, 1)];
        // B = (0.1, 0.05, 0) T: JᵀB = (0.2, −0.05, 0)
        let linear = induced_dipole_force(
            &DipoleSample {
                field: v(10, 5, 0, 100),
                jacobian,
            },
            &steel(),
            volume,
        );
        let k = 3.0 * 1e-6 / MU0;
        assert!(close(linear.x, k * 0.2, k * 0.2, 1e-9), "{linear:?}");
        assert!(close(linear.y, -k * 0.05, k * 0.2, 1e-9), "{linear:?}");
        assert!(linear.z.is_zero());
        // B = (1.2, 1.6, 0) T, |B| = 2: JᵀB = (2.4, −1.6, 0), scaled by M_s·V/|B| = 0.85 (|B| ≠ 1
        // so that the division by |B| shows)
        let saturated = induced_dipole_force(
            &DipoleSample {
                field: v(12, 16, 0, 10),
                jacobian,
            },
            &steel(),
            volume,
        );
        assert!(close(saturated.x, 0.85 * 2.4, 2.04, 1e-9), "{saturated:?}");
        assert!(close(saturated.y, -0.85 * 1.6, 2.04, 1e-9), "{saturated:?}");
        assert!(saturated.z.is_zero());
    }

    /// Unit oracle (the lib's own coverage): the flat layer's magnetic traction against Rosensweig's normal-field
    /// closed form `χB²/(2μ0(1+χ))` and the tangential one `χB²/(2μ0)` at χ = 1, the saturated normal field
    /// `μ0(M_s(H₀ − M_s) − M_s·H_s/2) + μ0 M_s²/2`, and no rise without weight or magnetisation
    #[test]
    fn the_magnetic_liquid_traction_and_rise() {
        let liquid = MagneticResponse {
            chi_eff: Fix128::ONE,
            saturation_magnetization: Fix128::from_int(400_000),
        };
        let up = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let normal_field = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(5, 100), Fix128::ZERO);
        let side_field = Vec3Fix::new(Fix128::from_ratio(5, 100), Fix128::ZERO, Fix128::ZERO);
        let mu0 = 4.0 * core::f64::consts::PI * 1e-7;
        let b2 = 0.05 * 0.05;
        let tn = magnetic_fluid_traction(normal_field, up, &liquid);
        assert!(close(tn, b2 / (2.0 * mu0 * 2.0), 1000.0, 1e-9), "{tn:?}");
        let tt = magnetic_fluid_traction(side_field, up, &liquid);
        assert!(close(tt, b2 / (2.0 * mu0), 1000.0, 1e-9), "{tt:?}");
        let h = magnetic_fluid_surface_rise(
            normal_field,
            up,
            &liquid,
            Fix128::from_int(1200),
            Fix128::from_ratio(981, 100),
        )
        .expect("a liquid with weight");
        assert!(
            close(h, b2 / (4.0 * mu0 * 1200.0 * 9.81), 0.1, 1e-9),
            "{h:?}"
        );
        // saturated normal field: χ = 1, M_s = H_s = 10000 A/m, H₀ = 0.05/μ0 ≈ 39789 A/m, the linear H₀/2 > H_s, and
        // the normal B condition gives H = H₀ − M_s
        let sat = MagneticResponse {
            chi_eff: Fix128::ONE,
            saturation_magnetization: Fix128::from_int(10_000),
        };
        let h0 = 0.05 / mu0;
        let (m_s, h_s) = (10_000.0, 10_000.0);
        let want = mu0 * (m_s * (h0 - m_s) - m_s * h_s / 2.0) + mu0 * m_s * m_s / 2.0;
        let ts = magnetic_fluid_traction(normal_field, up, &sat);
        assert!(close(ts, want, 1000.0, 1e-9), "{ts:?} vs {want}");
        assert_eq!(
            magnetic_fluid_surface_rise(normal_field, up, &liquid, Fix128::ZERO, Fix128::ONE),
            None
        );
        assert_eq!(
            magnetic_fluid_surface_rise(normal_field, up, &liquid, Fix128::ONE, -Fix128::ONE),
            None
        );
        let none = MagneticResponse {
            chi_eff: Fix128::ZERO,
            saturation_magnetization: Fix128::ZERO,
        };
        assert_eq!(
            magnetic_fluid_surface_rise(normal_field, up, &none, Fix128::ONE, Fix128::ONE),
            Some(Fix128::ZERO)
        );
    }
}
