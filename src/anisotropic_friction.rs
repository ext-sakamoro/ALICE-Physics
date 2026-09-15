//! Direction-dependent friction for tyres, skis, and ice-blade contacts.
//!
//! Complements [`crate::material::PhysicsMaterial`] (isotropic Coulomb friction)
//! with an orthotropic friction model — separate longitudinal and
//! transverse coefficients plus a kinematic multiplier that kicks in
//! once the tangential slip magnitude exceeds a threshold.
//!
//! For a contact whose surface tangent frame is `(t_long, t_trans)`
//! and relative tangential velocity `v_tan`, the friction force is
//!
//! ```text
//! v_long  = v_tan · t_long
//! v_trans = v_tan · t_trans
//! |v_tan| = sqrt(v_long² + v_trans²)
//! F = -N · ( μ_long · v_long · t_long + μ_trans · v_trans · t_trans ) / |v_tan|
//! ```
//!
//! — the orthotropic *friction ellipse* (Zmitrowicz 1981; the same law as
//! Bullet / PhysX anisotropic friction): along a principal axis it is Coulomb
//! `−μ N t̂`, for oblique slip the magnitude is `N sqrt(μ_long² cos²θ +
//! μ_trans² sin²θ)` and the direction is the ellipse normal, and with
//! `μ_long = μ_trans` it reduces to isotropic Coulomb `−μ N v̂` for every
//! slip direction. (Before 1.2.0 the module summed `sign(v_i) μ_i t̂_i` per
//! axis — a box law that was `√2 μ N` at 45° in the isotropic case.)
//! `N` is the contact normal force; the coefficients switch from
//! `μ_static` to `μ_kinetic` once the tangential slip speed `|v_tan|`
//! exceeds the threshold (inclusive: at the threshold static applies).

use crate::math::{Fix128, Vec3Fix};

/// Orthotropic friction parameters aligned with a surface tangent frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnisotropicFriction {
    /// Static friction coefficient along the longitudinal axis.
    pub longitudinal_static: Fix128,
    /// Kinetic friction coefficient along the longitudinal axis.
    pub longitudinal_kinetic: Fix128,
    /// Static friction coefficient along the transverse axis.
    pub transverse_static: Fix128,
    /// Kinetic friction coefficient along the transverse axis.
    pub transverse_kinetic: Fix128,
    /// Tangential slip speed (m/s) below which the static coefficient
    /// applies. Typical: 0.05.
    pub slip_threshold_m_s: Fix128,
}

impl AnisotropicFriction {
    /// Tyre on asphalt — high longitudinal grip, moderate transverse.
    #[must_use]
    pub fn tyre_asphalt() -> Self {
        Self {
            longitudinal_static: Fix128::from_ratio(11, 10),
            longitudinal_kinetic: Fix128::from_ratio(9, 10),
            transverse_static: Fix128::from_ratio(9, 10),
            transverse_kinetic: Fix128::from_ratio(7, 10),
            slip_threshold_m_s: Fix128::from_ratio(5, 100),
        }
    }

    /// Ski on packed snow — very low longitudinal, high transverse.
    #[must_use]
    pub fn ski_snow() -> Self {
        Self {
            longitudinal_static: Fix128::from_ratio(6, 100),
            longitudinal_kinetic: Fix128::from_ratio(4, 100),
            transverse_static: Fix128::from_ratio(9, 10),
            transverse_kinetic: Fix128::from_ratio(75, 100),
            slip_threshold_m_s: Fix128::from_ratio(5, 100),
        }
    }

    /// Ice blade — near-zero longitudinal friction.
    #[must_use]
    pub fn skate_ice() -> Self {
        Self {
            longitudinal_static: Fix128::from_ratio(2, 100),
            longitudinal_kinetic: Fix128::from_ratio(15, 1000),
            transverse_static: Fix128::from_ratio(85, 100),
            transverse_kinetic: Fix128::from_ratio(7, 10),
            slip_threshold_m_s: Fix128::from_ratio(2, 100),
        }
    }

    /// Compute the tangential friction force acting on the sliding
    /// body.
    ///
    /// Inputs are expressed in world coordinates:
    ///
    /// - `normal_force`: contact normal load (N). Non-positive returns zero.
    /// - `tangent_long`, `tangent_trans`: orthonormal tangent axes on
    ///   the contact surface.
    /// - `relative_velocity`: sliding velocity in world space.
    #[must_use]
    pub fn friction_force(
        &self,
        normal_force: Fix128,
        tangent_long: Vec3Fix,
        tangent_trans: Vec3Fix,
        relative_velocity: Vec3Fix,
    ) -> Vec3Fix {
        if normal_force <= Fix128::ZERO {
            return Vec3Fix::ZERO;
        }
        let v_long = relative_velocity.dot(tangent_long);
        let v_trans = relative_velocity.dot(tangent_trans);
        // |v_tan|; exact on the principal axes (no square/root round trip)
        // so the Coulomb limit there is bit-exact.
        let slip = if v_long.is_zero() {
            v_trans.abs()
        } else if v_trans.is_zero() {
            v_long.abs()
        } else {
            (v_long * v_long + v_trans * v_trans).sqrt()
        };
        if slip.is_zero() {
            return Vec3Fix::ZERO;
        }
        let (mu_long, mu_trans) = self.select_coefficients(slip);
        // Friction ellipse: each axis contributes in proportion to its share
        // of the slip velocity, so the total is −N · M v̂ with M = diag(μ).
        let long_force = tangent_long * (-normal_force * mu_long * v_long / slip);
        let trans_force = tangent_trans * (-normal_force * mu_trans * v_trans / slip);
        long_force + trans_force
    }

    /// `(μ_long, μ_trans)` for the given tangential slip speed: static at
    /// or below the threshold, kinetic above it.
    fn select_coefficients(&self, slip_speed: Fix128) -> (Fix128, Fix128) {
        if slip_speed <= self.slip_threshold_m_s {
            (self.longitudinal_static, self.transverse_static)
        } else {
            (self.longitudinal_kinetic, self.transverse_kinetic)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> (Vec3Fix, Vec3Fix) {
        (
            Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        )
    }

    #[test]
    fn preset_tyre_high_longitudinal_grip() {
        let m = AnisotropicFriction::tyre_asphalt();
        assert!(m.longitudinal_static > m.transverse_static);
    }

    #[test]
    fn preset_ski_low_longitudinal_grip() {
        let m = AnisotropicFriction::ski_snow();
        assert!(m.longitudinal_static < m.transverse_static);
    }

    #[test]
    fn preset_ice_extreme_anisotropy() {
        let m = AnisotropicFriction::skate_ice();
        assert!(m.longitudinal_kinetic < Fix128::from_ratio(5, 100));
        assert!(m.transverse_static > Fix128::from_ratio(5, 10));
    }

    #[test]
    fn zero_normal_load_yields_zero_force() {
        let m = AnisotropicFriction::tyre_asphalt();
        let (tl, tt) = axes();
        let f = m.friction_force(
            Fix128::ZERO,
            tl,
            tt,
            Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
        );
        assert_eq!(f, Vec3Fix::ZERO);
    }

    #[test]
    fn friction_opposes_pure_longitudinal_slip() {
        let m = AnisotropicFriction::tyre_asphalt();
        let (tl, tt) = axes();
        let velocity = Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO);
        let f = m.friction_force(Fix128::from_int(100), tl, tt, velocity);
        assert!(f.x < Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn friction_opposes_pure_transverse_slip() {
        let m = AnisotropicFriction::tyre_asphalt();
        let (tl, tt) = axes();
        let velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(5));
        let f = m.friction_force(Fix128::from_int(100), tl, tt, velocity);
        assert_eq!(f.x, Fix128::ZERO);
        assert!(f.z < Fix128::ZERO);
    }

    #[test]
    fn slow_slip_selects_static_coefficient() {
        let mut m = AnisotropicFriction::tyre_asphalt();
        m.longitudinal_static = Fix128::from_int(2);
        m.longitudinal_kinetic = Fix128::from_int(1);
        let (tl, tt) = axes();
        // Below threshold — static.
        let slow = Vec3Fix::new(Fix128::from_ratio(1, 100), Fix128::ZERO, Fix128::ZERO);
        let f_slow = m.friction_force(Fix128::from_int(10), tl, tt, slow);
        // Above threshold — kinetic.
        let fast = Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO);
        let f_fast = m.friction_force(Fix128::from_int(10), tl, tt, fast);
        // The static magnitude should be greater than the kinetic magnitude.
        assert!(f_slow.x.abs() > f_fast.x.abs());
    }
}
