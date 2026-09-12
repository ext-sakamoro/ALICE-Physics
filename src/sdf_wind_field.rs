//! SDF-driven wind field: wind vector is a function of the local SDF
//! distance to obstacles.
//!
//! Complements [`crate::wind_zone`] (uniform bounded wind) by making
//! the wind field aware of the surrounding geometry. Typical use case:
//! wind flowing around a building — velocity accelerates over the roof
//! and decelerates in the wake shadow, so anything downstream sees a
//! naturally-varying vector without a pre-baked flow-field lookup.
//!
//! The MVP model:
//!
//! ```text
//! d      = sdf.distance(x)
//! shelter = clamp(d / decay_scale, 0, 1)
//! wind    = base_direction · base_speed · shelter
//! ```
//!
//! Points deep inside the wake shadow (small `d`) receive attenuated
//! wind; points far from any obstacle receive the full base wind.
//! Callers who need lift over the roof / turbulence spectra should
//! compose the field with [`crate::wind_zone`] or upgrade to a full
//! LES post-process.

use crate::sdf_collider::SdfField;

/// SDF-driven wind sampler.
pub struct SdfWindField<'a, F: SdfField + ?Sized> {
    /// SDF describing obstacle geometry. Positive distance = outside.
    pub sdf: &'a F,
    /// Nominal wind direction (should be a unit vector).
    pub base_direction: [f32; 3],
    /// Nominal wind speed at distances `>= decay_scale`.
    pub base_speed_m_s: f32,
    /// Distance scale over which the shelter attenuation ramps from
    /// 0 → 1. Typical: 5 m for building-scale geometry.
    pub decay_scale_m: f32,
}

impl<'a, F: SdfField + ?Sized> SdfWindField<'a, F> {
    /// Construct a sampler with sensible defaults for outdoor scenes.
    pub fn new(sdf: &'a F, direction: [f32; 3], speed: f32) -> Self {
        Self {
            sdf,
            base_direction: direction,
            base_speed_m_s: speed,
            decay_scale_m: 5.0,
        }
    }

    /// Sample the wind velocity at `position` (m).
    #[must_use]
    pub fn sample(&self, position: [f32; 3]) -> [f32; 3] {
        let d = self
            .sdf
            .distance(position[0], position[1], position[2])
            .max(0.0);
        let shelter = if self.decay_scale_m <= 0.0 {
            1.0
        } else {
            (d / self.decay_scale_m).clamp(0.0, 1.0)
        };
        let scale = self.base_speed_m_s * shelter;
        [
            self.base_direction[0] * scale,
            self.base_direction[1] * scale,
            self.base_direction[2] * scale,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    fn box_wall() -> ClosureSdf {
        // Half-space x < 0 is inside the wall, x > 0 outside.
        ClosureSdf::new(|x, _y, _z| x, |_x, _y, _z| (1.0, 0.0, 0.0))
    }

    #[test]
    fn wind_zero_inside_obstacle() {
        let sdf = ground_plane();
        let field = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 10.0);
        // Point below the plane (y < 0) → inside → distance clamped to 0.
        let v = field.sample([0.0, -1.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn wind_at_full_strength_far_from_obstacle() {
        let sdf = ground_plane();
        let field = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 10.0);
        let v = field.sample([0.0, 100.0, 0.0]);
        assert!((v[0] - 10.0).abs() < 1.0e-3);
    }

    #[test]
    fn wind_attenuates_within_decay_scale() {
        let sdf = box_wall();
        let mut field = SdfWindField::new(&sdf, [0.0, 0.0, 1.0], 5.0);
        field.decay_scale_m = 10.0;
        // Sample at x = 2 → d = 2, shelter = 0.2 → v_z = 1.0.
        let v = field.sample([2.0, 0.0, 0.0]);
        assert!((v[2] - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn zero_decay_scale_yields_full_wind_everywhere() {
        let sdf = ground_plane();
        let mut field = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 3.0);
        field.decay_scale_m = 0.0;
        let v = field.sample([0.0, 1.0, 0.0]);
        assert!((v[0] - 3.0).abs() < 1.0e-3);
    }

    #[test]
    fn direction_preserved_in_sample() {
        let sdf = ground_plane();
        let field = SdfWindField::new(&sdf, [0.0, 0.0, 1.0], 8.0);
        let v = field.sample([0.0, 50.0, 0.0]);
        assert!(v[0].abs() < 1.0e-3);
        assert!(v[1].abs() < 1.0e-3);
        assert!(v[2] > 0.0);
    }
}
