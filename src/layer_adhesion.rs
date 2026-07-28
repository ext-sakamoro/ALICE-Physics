//! FDM Layer Adhesion Analysis (Print-Direction × Load-Direction Matrix)
//!
//! Phase C1 of the ALICE-Physics completeness project. Practical extension of
//! `anisotropic.rs`: rather than modelling full orthotropic elasticity, this
//! module provides a **quick 6×6 lookup** that returns the effective strength
//! for a given loading direction relative to the print orientation. Useful
//! for early feasibility checks before running a full FEA.
//!
//! # Concept
//!
//! A printed part is laid down in the XY plane, growing along Z one layer at
//! a time. The strength envelope has three regimes:
//!
//! - **Within-layer (XY normal / XY shear)** — full material strength, since
//!   forces travel through a continuous extrusion path.
//! - **Across-layer normal (Z tension)** — reduced to the layer-bond strength
//!   `σ_z = σ_iso × anisotropy_z_ratio` (0.6–0.9 for common FDM materials).
//! - **Across-layer shear (Z-XY plane)** — intermediate; empirical factor
//!   `~ (1 + anisotropy_z_ratio) / 2`.
//!
//! # Output
//!
//! `EffectiveStrength { normal_x, normal_y, normal_z, shear_xy, shear_yz, shear_xz }`
//! all in MPa. Use these to convert a load-case bending stress into a factor
//! of safety without invoking the full Hill or Tsai-Wu criteria.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Print orientation
// ============================================================================

/// How the part is laid down on the build plate.
///
/// Only the "standard flat" orientation (layers along XY, growing +Z) is
/// currently supported; other orientations should be obtained by applying
/// a rotation before calling this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrintOrientation {
    /// Print bed = XY plane, layer stacking direction = +Z. Standard.
    #[default]
    XYFlat,
}

// ============================================================================
// Effective strength tensor (diagonal, 6 components)
// ============================================================================

/// Effective allowable stresses (MPa) for each of the 6 principal load
/// components at a printed part's layer geometry.
///
/// Sign convention: values are magnitudes (positive). Use with `.abs()` of
/// the applied stress when checking safety.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectiveStrength {
    /// Normal stress limit along X (MPa).
    pub normal_x_mpa: Fix128,
    /// Normal stress limit along Y (MPa).
    pub normal_y_mpa: Fix128,
    /// Normal stress limit along Z (MPa) — the layer-bond direction.
    pub normal_z_mpa: Fix128,
    /// Shear stress limit in the XY plane (MPa) — within-layer.
    pub shear_xy_mpa: Fix128,
    /// Shear stress limit in the YZ plane (MPa) — across layers.
    pub shear_yz_mpa: Fix128,
    /// Shear stress limit in the XZ plane (MPa) — across layers.
    pub shear_xz_mpa: Fix128,
}

impl EffectiveStrength {
    /// Compute for the standard XY-flat orientation, given the material and
    /// an in-plane shear factor (typical polymer: 0.6·σ_y).
    #[must_use]
    pub fn for_material(m: &MaterialProperties, orientation: PrintOrientation) -> Self {
        let sigma = m.yield_strength_mpa;
        let sigma_z = sigma * m.anisotropy_z_ratio;
        let tau_xy = sigma * Fix128::from_ratio(6, 10);
        // Across-layer shear: linear interpolation between within-layer τ and Z tension
        let inter = (Fix128::ONE + m.anisotropy_z_ratio) * Fix128::from_ratio(1, 2);
        let tau_z_across = tau_xy * inter;
        match orientation {
            PrintOrientation::XYFlat => Self {
                normal_x_mpa: sigma,
                normal_y_mpa: sigma,
                normal_z_mpa: sigma_z,
                shear_xy_mpa: tau_xy,
                shear_yz_mpa: tau_z_across,
                shear_xz_mpa: tau_z_across,
            },
        }
    }

    /// Factor of safety for a given applied stress (MPa) at a specified
    /// component. Returns positive infinity sentinel for zero applied stress.
    fn component_fos(&self, applied: Fix128, allowable: Fix128) -> Fix128 {
        let mag = applied.abs();
        if mag.is_zero() {
            return Fix128::from_int(i64::MAX >> 8);
        }
        if allowable.is_zero() {
            return Fix128::ZERO;
        }
        allowable / mag
    }

    /// FoS for a normal-X stress state.
    #[must_use]
    pub fn fos_normal_x(&self, applied: Fix128) -> Fix128 {
        self.component_fos(applied, self.normal_x_mpa)
    }

    /// FoS for a normal-Z stress state (the critical layer-bond direction).
    #[must_use]
    pub fn fos_normal_z(&self, applied: Fix128) -> Fix128 {
        self.component_fos(applied, self.normal_z_mpa)
    }

    /// FoS for a XY shear stress state.
    #[must_use]
    pub fn fos_shear_xy(&self, applied: Fix128) -> Fix128 {
        self.component_fos(applied, self.shear_xy_mpa)
    }

    /// FoS for a XZ (across-layer) shear stress state.
    #[must_use]
    pub fn fos_shear_xz(&self, applied: Fix128) -> Fix128 {
        self.component_fos(applied, self.shear_xz_mpa)
    }

    /// Minimum FoS across all 6 components.
    #[must_use]
    pub fn min_fos(&self, stress: &[(Fix128, Fix128); 6]) -> Fix128 {
        // stress = [(σ_x, S_x), (σ_y, S_y), ...] pre-paired for convenience
        let mut worst = Fix128::from_int(i64::MAX >> 8);
        for (applied, allow) in stress {
            let fos = self.component_fos(*applied, *allow);
            if fos < worst {
                worst = fos;
            }
        }
        worst
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_z_below_normal_x() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        // PLA anisotropy = 0.65 → normal_z = 50 × 0.65 = 32.5 MPa < normal_x = 50 MPa
        assert!(s.normal_z_mpa < s.normal_x_mpa);
    }

    #[test]
    fn normal_x_equals_normal_y() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        assert_eq!(s.normal_x_mpa, s.normal_y_mpa);
    }

    #[test]
    fn shear_xy_higher_than_across_layer_shear() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        assert!(s.shear_xy_mpa > s.shear_yz_mpa);
    }

    #[test]
    fn tpu_has_higher_z_ratio_than_pla() {
        let s_tpu =
            EffectiveStrength::for_material(&MaterialProperties::tpu(), PrintOrientation::XYFlat);
        let s_pla =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        let r_tpu = s_tpu.normal_z_mpa / s_tpu.normal_x_mpa;
        let r_pla = s_pla.normal_z_mpa / s_pla.normal_x_mpa;
        assert!(r_tpu > r_pla);
    }

    #[test]
    fn cf_nylon_most_z_penalized() {
        let s_cfn = EffectiveStrength::for_material(
            &MaterialProperties::cf_nylon(),
            PrintOrientation::XYFlat,
        );
        let s_pla =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        let r_cfn = s_cfn.normal_z_mpa / s_cfn.normal_x_mpa;
        let r_pla = s_pla.normal_z_mpa / s_pla.normal_x_mpa;
        assert!(r_cfn < r_pla);
    }

    #[test]
    fn sheet_metal_isotropic() {
        // SUS304 anisotropy = 1.0 → all normal directions equal
        let s = EffectiveStrength::for_material(
            &MaterialProperties::sus304(),
            PrintOrientation::XYFlat,
        );
        assert_eq!(s.normal_x_mpa, s.normal_z_mpa);
    }

    #[test]
    fn fos_zero_stress_is_huge_sentinel() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        let fos = s.fos_normal_x(Fix128::ZERO);
        assert!(fos > Fix128::from_int(1_000_000));
    }

    #[test]
    fn fos_normal_x_correct() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        // 25 MPa applied vs 50 MPa allowable → FoS = 2
        let fos = s.fos_normal_x(Fix128::from_int(25));
        assert_eq!(fos, Fix128::from_int(2));
    }

    #[test]
    fn fos_normal_z_lower_for_pla() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        // Same applied stress → Z FoS < X FoS
        let fos_x = s.fos_normal_x(Fix128::from_int(20));
        let fos_z = s.fos_normal_z(Fix128::from_int(20));
        assert!(fos_z < fos_x);
    }

    #[test]
    fn min_fos_picks_worst_component() {
        let s =
            EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
        let stress = [
            (Fix128::from_int(10), s.normal_x_mpa),
            (Fix128::from_int(10), s.normal_y_mpa),
            (Fix128::from_int(30), s.normal_z_mpa), // worst: high stress + weak direction
            (Fix128::ZERO, s.shear_xy_mpa),
            (Fix128::ZERO, s.shear_yz_mpa),
            (Fix128::ZERO, s.shear_xz_mpa),
        ];
        let fos_min = s.min_fos(&stress);
        // FoS for Z should be s.normal_z / 30 = 32.5/30 ≈ 1.08
        let fos_z = s.fos_normal_z(Fix128::from_int(30));
        assert_eq!(fos_min, fos_z);
    }
}
