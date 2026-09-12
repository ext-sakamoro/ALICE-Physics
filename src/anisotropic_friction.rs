//! Direction-dependent friction for tyres, skis, and ice-blade contacts.
//!
//! Complements [`crate::material::Material`] (isotropic Coulomb friction)
//! with an orthotropic friction model — separate longitudinal and
//! transverse coefficients plus a kinematic multiplier that kicks in
//! once the tangential slip magnitude exceeds a threshold.
//!
//! For a contact whose surface tangent frame is `(t_long, t_trans)`
//! and relative tangential velocity `v_tan`, the friction force is
//!
//! ```text
//! v_long = v_tan · t_long
//! v_trans = v_tan · t_trans
//! F = -N · ( μ_long  · sign(v_long)  · t_long
//!          + μ_trans · sign(v_trans) · t_trans )
//! ```
//!
//! where `N` is the contact normal force and each coefficient becomes
//! `μ_static` below the slip threshold and `μ_kinetic` above it.

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
        let mu_long = self.select_coefficient(v_long, true);
        let mu_trans = self.select_coefficient(v_trans, false);
        let long_force = tangent_long * (-normal_force * mu_long * sign(v_long));
        let trans_force = tangent_trans * (-normal_force * mu_trans * sign(v_trans));
        long_force + trans_force
    }

    fn select_coefficient(&self, speed: Fix128, longitudinal: bool) -> Fix128 {
        let mag = speed.abs();
        let (static_mu, kinetic_mu) = if longitudinal {
            (self.longitudinal_static, self.longitudinal_kinetic)
        } else {
            (self.transverse_static, self.transverse_kinetic)
        };
        if mag <= self.slip_threshold_m_s {
            static_mu
        } else {
            kinetic_mu
        }
    }
}

fn sign(x: Fix128) -> Fix128 {
    if x > Fix128::ZERO {
        Fix128::ONE
    } else if x < Fix128::ZERO {
        -Fix128::ONE
    } else {
        Fix128::ZERO
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
