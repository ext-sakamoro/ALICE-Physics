//! Elastoplastic Deformation with von Mises Yield & Creep Prediction
//!
//! Phase B2 of the ALICE-Physics completeness project. Extends the elastic
//! stress models (`beam_stress`, `anisotropic`) with **permanent (plastic)
//! deformation** and **long-term creep** — the two mechanisms responsible for
//! FDM part failures that "elastic-only" analysis misses:
//!
//! - PLA shelves that sag over 6 months (creep under gravity).
//! - Metal brackets that yield locally at bolt holes (plastic slip).
//! - SKADIS pegs that permanently splay after repeated loading.
//!
//! # Models provided
//!
//! - **von Mises equivalent stress** for a full 3D stress tensor.
//! - **Isotropic / kinematic / combined hardening** (`HardeningType`).
//! - **Radial return** integration algorithm for one time step under prescribed
//!   trial stress (classical J2 plasticity, small-strain).
//! - **Norton power-law creep** ε̇_c = A · σⁿ with time integration.
//! - Preset `PlasticModel::from_fdm_material()` derives reasonable defaults
//!   from the isotropic `MaterialProperties`.
//!
//! # Unit convention
//!
//! Same as `beam_stress` / `anisotropic`: MPa, mm, N, s.
//!
//! # References
//!
//! - Simo & Hughes, *Computational Inelasticity*, Springer 1998. Ch 2 (radial
//!   return), Ch 3 (kinematic hardening).
//! - Chaboche, "A review of some plasticity and viscoplasticity constitutive
//!   theories", Int. J. Plasticity 24 (2008).
//! - Norton, "The creep of steel at high temperatures", McGraw-Hill 1929
//!   (original power-law).
//! - Bellehumeur et al., "Modeling of Bond Formation Between Polymer Filaments
//!   in the FDM Process", J. Manuf. Processes 2004 (PLA creep parameters).
//!
//! # Integration status
//!
//! `PlasticModel`, `PlasticState`, `NortonCreep`, and `radial_return_1d`
//! are wired into `structural_solver.rs`. `PlasticStep`,
//! `current_yield_mpa`, and `strain_rate_per_s` are crate-internal and
//! reached through `radial_return_1d` / `integrate`.
//! `PlasticModel::with_hardening` and `NortonCreep::petg_room_temp` have no
//! consumer (the solver builds its model with `from_fdm_material` and
//! hard-codes the PLA creep preset) and carry `ALLOW-UNWIRED` debt markers
//! with closed-form oracles in this module's unit tests. `StressTensor` is
//! a crate-internal duplicate of the public
//! `linear_elastic_fem::StressTensor` with no caller; it is kept under a
//! per-item `allow(dead_code)` until the two are consolidated.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Stress tensor
// ============================================================================

/// Symmetric 3D Cauchy stress tensor (6 independent components, in MPa).
///
/// Positive normal stresses are tensile.
// ALLOW-DEAD: duplicate of the public linear_elastic_fem::StressTensor (xx..zx, von_mises, hydrostatic); no crate caller, consolidation is an API decision
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct StressTensor {
    /// Normal stress σ_xx.
    pub(crate) sxx: Fix128,
    /// Normal stress σ_yy.
    pub(crate) syy: Fix128,
    /// Normal stress σ_zz.
    pub(crate) szz: Fix128,
    /// Shear stress τ_xy.
    pub(crate) sxy: Fix128,
    /// Shear stress τ_xz.
    pub(crate) sxz: Fix128,
    /// Shear stress τ_yz.
    pub(crate) syz: Fix128,
}

// ALLOW-DEAD: uniaxial_x / hydrostatic / von_mises belong to the duplicate StressTensor above; same consolidation decision
#[allow(dead_code)]
impl StressTensor {
    /// Uniaxial stress along X (all other components zero).
    #[must_use]
    pub(crate) const fn uniaxial_x(sigma: Fix128) -> Self {
        Self {
            sxx: sigma,
            syy: Fix128::ZERO,
            szz: Fix128::ZERO,
            sxy: Fix128::ZERO,
            sxz: Fix128::ZERO,
            syz: Fix128::ZERO,
        }
    }

    /// Hydrostatic (spherical) part σ_H = ⅓·tr(σ).
    #[must_use]
    pub(crate) fn hydrostatic(&self) -> Fix128 {
        (self.sxx + self.syy + self.szz) * Fix128::from_ratio(1, 3)
    }

    /// von Mises equivalent stress:
    /// `σ_eq = √( ½ · ((σ_xx − σ_yy)² + (σ_yy − σ_zz)² + (σ_zz − σ_xx)²)
    ///           + 3·(τ_xy² + τ_xz² + τ_yz²) )`
    #[must_use]
    pub(crate) fn von_mises(&self) -> Fix128 {
        let d1 = self.sxx - self.syy;
        let d2 = self.syy - self.szz;
        let d3 = self.szz - self.sxx;
        let normal_part = (d1 * d1 + d2 * d2 + d3 * d3) * Fix128::from_ratio(1, 2);
        let shear_part =
            (self.sxy * self.sxy + self.sxz * self.sxz + self.syz * self.syz) * Fix128::from_int(3);
        (normal_part + shear_part).sqrt()
    }
}

// ============================================================================
// Hardening
// ============================================================================

/// Plastic hardening law selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HardeningType {
    /// Yield surface expands uniformly. Simple, but no Bauschinger effect.
    #[default]
    Isotropic,
    /// Yield surface translates (back-stress evolves). Reproduces Bauschinger
    /// under load reversals — required for cyclic / fatigue analysis.
    Kinematic,
    /// Combined isotropic + kinematic (equal split).
    Combined,
}

// ============================================================================
// PlasticModel
// ============================================================================

/// Elastoplastic constitutive model parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlasticModel {
    /// Initial yield stress σ_y0 (MPa).
    pub yield_strength_mpa: Fix128,
    /// Linear hardening modulus H (MPa). Slope of the σ vs ε_p curve.
    ///
    /// Typical polymer values: PLA H ≈ 200 MPa, ABS H ≈ 150 MPa.
    /// Metals: mild steel H ≈ 1500 MPa, aluminium H ≈ 500 MPa.
    pub hardening_modulus_mpa: Fix128,
    /// Hardening law.
    pub hardening_type: HardeningType,
    /// Young's modulus (MPa) for elastic strain accumulation.
    pub youngs_modulus_mpa: Fix128,
}

impl PlasticModel {
    /// Construct from an isotropic material. Defaults to isotropic hardening
    /// with `H = 0.05 · E` (typical elastic-plastic ratio for polymers).
    #[must_use]
    pub fn from_fdm_material(m: &MaterialProperties) -> Self {
        let e_mpa = m.youngs_modulus_gpa * Fix128::from_int(1000);
        let h_mpa = e_mpa * Fix128::from_ratio(5, 100);
        Self {
            yield_strength_mpa: m.yield_strength_mpa,
            hardening_modulus_mpa: h_mpa,
            hardening_type: HardeningType::Isotropic,
            youngs_modulus_mpa: e_mpa,
        }
    }

    /// Override the hardening law, leaving every other parameter unchanged.
    // ALLOW-DEAD: pub(crate) with no crate caller, same debt as the ALLOW-UNWIRED marker below
    // ALLOW-UNWIRED: wiring debt Backlog structural-pub-crate-residue (hardening_type is a pub field, the solver sets nothing but the from_fdm_material default), oracle src/plastic.rs tests::with_hardening_back_stress_matches_radial_return_closed_form
    #[allow(dead_code)]
    #[must_use]
    pub(crate) const fn with_hardening(mut self, ht: HardeningType) -> Self {
        self.hardening_type = ht;
        self
    }
}

// ============================================================================
// PlasticState
// ============================================================================

/// Internal state variables that evolve during plastic deformation.
///
/// One instance is carried per integration point (per mesh element or per
/// beam cross-section). Must be persisted across time steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlasticState {
    /// Accumulated equivalent plastic strain (dimensionless).
    pub equivalent_plastic_strain: Fix128,
    /// Back-stress (MPa) — center of the yield surface (kinematic hardening).
    pub back_stress_mpa: Fix128,
    /// Accumulated creep strain (dimensionless). Independent of plasticity.
    pub creep_strain: Fix128,
}

// ============================================================================
// Yield check & radial return
// ============================================================================

/// Current yield stress accounting for hardening.
///
/// - Isotropic: `σ_y0 + H · ε_p`
/// - Kinematic: `σ_y0` (radius fixed, back-stress captures translation)
/// - Combined: `σ_y0 + ½ · H · ε_p` (half growth, half translation)
#[must_use]
pub(crate) fn current_yield_mpa(model: &PlasticModel, state: &PlasticState) -> Fix128 {
    match model.hardening_type {
        HardeningType::Isotropic => {
            model.yield_strength_mpa + model.hardening_modulus_mpa * state.equivalent_plastic_strain
        }
        HardeningType::Kinematic => model.yield_strength_mpa,
        HardeningType::Combined => {
            model.yield_strength_mpa
                + model.hardening_modulus_mpa
                    * state.equivalent_plastic_strain
                    * Fix128::from_ratio(1, 2)
        }
    }
}

/// Result of an incremental plasticity update (crate-internal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlasticStep {
    /// Actual stress magnitude after return-mapping (MPa).
    pub(crate) stress_mpa: Fix128,
    /// Plastic strain increment this step (dimensionless).
    pub(crate) plastic_strain_increment: Fix128,
    /// True iff yielding occurred (plastic strain > 0).
    pub(crate) yielded: bool,
}

/// One-dimensional radial-return update for uniaxial loading.
///
/// Given the elastic-trial stress `trial_stress_mpa` computed as
/// `σ_trial = σ_prev + E · Δε` (predictor step), this function checks the
/// yield condition against the current hardened yield surface. If yielding,
/// it computes the plastic multiplier that returns the trial stress to the
/// yield surface (radial return, Simo & Hughes eq. 2.13) and updates
/// `state.equivalent_plastic_strain` (isotropic) or `state.back_stress_mpa`
/// (kinematic).
///
/// This is the standard J2 return-mapping specialised to 1-D — sufficient
/// for beam-like analyses. Full 3-D versions use the same structure with
/// a deviatoric stress tensor.
pub(crate) fn radial_return_1d(
    trial_stress_mpa: Fix128,
    model: &PlasticModel,
    state: &mut PlasticState,
) -> PlasticStep {
    let yield_now = current_yield_mpa(model, state);

    // Effective stress relative to back-stress (kinematic shift)
    let effective = trial_stress_mpa - state.back_stress_mpa;
    let abs_eff = effective.abs();

    if abs_eff <= yield_now {
        // Elastic: trial stress is within yield surface
        return PlasticStep {
            stress_mpa: trial_stress_mpa,
            plastic_strain_increment: Fix128::ZERO,
            yielded: false,
        };
    }

    // Plastic loading: solve for plastic multiplier Δλ
    // f = |σ − α| − (σ_y + H·ε_p) = 0
    // With E as the elastic modulus, the return gives
    // Δλ = (|σ_trial − α| − σ_y_current) / (E + H)
    let e_plus_h = model.youngs_modulus_mpa + model.hardening_modulus_mpa;
    if e_plus_h.is_zero() {
        return PlasticStep {
            stress_mpa: trial_stress_mpa,
            plastic_strain_increment: Fix128::ZERO,
            yielded: false,
        };
    }
    let dlambda = (abs_eff - yield_now) / e_plus_h;

    // Direction: sign of the effective stress
    let sign = if effective.is_negative() {
        Fix128::NEG_ONE
    } else {
        Fix128::ONE
    };

    // Update state
    state.equivalent_plastic_strain = state.equivalent_plastic_strain + dlambda;
    match model.hardening_type {
        HardeningType::Kinematic => {
            // Back-stress evolves in the flow direction
            state.back_stress_mpa =
                state.back_stress_mpa + sign * model.hardening_modulus_mpa * dlambda;
        }
        HardeningType::Combined => {
            // Half of H goes to back-stress
            state.back_stress_mpa = state.back_stress_mpa
                + sign * model.hardening_modulus_mpa * dlambda * Fix128::from_ratio(1, 2);
        }
        HardeningType::Isotropic => {}
    }

    // Corrected stress = trial minus elastic overshoot
    let stress = trial_stress_mpa - sign * model.youngs_modulus_mpa * dlambda;

    PlasticStep {
        stress_mpa: stress,
        plastic_strain_increment: dlambda,
        yielded: true,
    }
}

// ============================================================================
// Norton creep
// ============================================================================

/// Norton power-law creep model: ε̇_c = A · σⁿ
///
/// Coefficients are strongly temperature-dependent. Presets in
/// `NortonCreep::pla_room_temp()` etc. reproduce the typical creep response
/// documented in Bellehumeur (2004) and Prusa's technical papers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NortonCreep {
    /// Material constant A (creep coefficient) — units chosen such that with
    /// σ in MPa and integer exponent n, ε̇ is per second.
    pub a: Fix128,
    /// Stress exponent n (dimensionless, typically 3-8 for polymers, 1-3
    /// for metals near room temperature). Stored as integer for numerical
    /// stability (Fix128 lacks `.powf`).
    pub n: u32,
}

impl NortonCreep {
    /// PLA at 25°C — significant creep even at moderate stress.
    /// Calibrated to Bellehumeur (2004): 1 % strain after 6 months at 10 MPa,
    /// i.e. `A = 0.01 / (10³ MPa³ · 1.5768e7 s) ≈ 6.34e-13 /(MPa³·s)`, `n = 3`.
    /// Before 1.2.0 `A = 3e-10` was 470× the documented calibration (466 %
    /// strain in 6 months; `tests/engineering_oracles_solid.rs`).
    #[must_use]
    pub fn pla_room_temp() -> Self {
        Self {
            a: Fix128::from_ratio(634, 1_000_000_000_000_000_i64),
            n: 3,
        }
    }

    /// PETG at 25°C — lower creep than PLA (higher Tg):
    /// `A = 1e-13 /(MPa³·s)`, `n = 3`, about 6.3× below the PLA calibration.
    // ALLOW-DEAD: pub(crate) with no crate caller, same debt as the ALLOW-UNWIRED marker below
    // ALLOW-UNWIRED: wiring debt Backlog structural-pub-crate-residue (structural_solver::new hard-codes the PLA creep preset), oracle src/plastic.rs tests::petg_norton_preset_integrates_a_sigma_cubed_dt
    #[allow(dead_code)]
    #[must_use]
    pub(crate) fn petg_room_temp() -> Self {
        // ≈ 6× lower than the PLA calibration (1.2.0: 1e-11 was 16× *above*
        // the corrected PLA value)
        Self {
            a: Fix128::from_ratio(1, 10_000_000_000_000_i64),
            n: 3,
        }
    }

    /// Instantaneous creep strain rate (per second) at stress `sigma_mpa` (crate-internal).
    #[must_use]
    pub(crate) fn strain_rate_per_s(&self, sigma_mpa: Fix128) -> Fix128 {
        let mut sn = Fix128::ONE;
        for _ in 0..self.n {
            sn = sn * sigma_mpa;
        }
        self.a * sn
    }

    /// Advance creep state by `dt` seconds under constant stress.
    ///
    /// Uses forward-Euler integration: `ε_c_new = ε_c + ε̇·dt`. Adequate for
    /// engineering time-scales (hours to years) with a suitably small dt.
    pub fn integrate(&self, sigma_mpa: Fix128, dt_seconds: Fix128, state: &mut PlasticState) {
        let rate = self.strain_rate_per_s(sigma_mpa);
        state.creep_strain = state.creep_strain + rate * dt_seconds;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn von_mises_uniaxial_equals_sigma() {
        let s = StressTensor::uniaxial_x(Fix128::from_int(100));
        let vm = s.von_mises();
        assert!(approx_eq(
            vm,
            Fix128::from_int(100),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn von_mises_pure_shear() {
        let s = StressTensor {
            sxy: Fix128::from_int(50),
            ..Default::default()
        };
        // Pure shear τ → σ_eq = √3 · τ
        let vm = s.von_mises();
        let expected = Fix128::from_int(50) * Fix128::from_int(3).sqrt();
        assert!(
            approx_eq(vm, expected, Fix128::from_ratio(1, 100)),
            "got {}, expected {}",
            vm.to_f32(),
            expected.to_f32()
        );
    }

    #[test]
    fn von_mises_hydrostatic_is_zero() {
        let s = StressTensor {
            sxx: Fix128::from_int(100),
            syy: Fix128::from_int(100),
            szz: Fix128::from_int(100),
            ..Default::default()
        };
        // Pure hydrostatic → no distortion → σ_eq = 0
        let vm = s.von_mises();
        assert!(vm < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn hydrostatic_average_of_normals() {
        let s = StressTensor {
            sxx: Fix128::from_int(30),
            syy: Fix128::from_int(60),
            szz: Fix128::from_int(90),
            ..Default::default()
        };
        // (30+60+90)/3 = 60 but 1/3 is non-terminating in binary; allow ULP.
        assert!(approx_eq(
            s.hydrostatic(),
            Fix128::from_int(60),
            Fix128::from_ratio(1, 1000)
        ));
    }

    #[test]
    fn elastic_stress_below_yield() {
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        // PLA σ_y = 50 MPa, apply 30 → elastic
        let step = radial_return_1d(Fix128::from_int(30), &model, &mut state);
        assert!(!step.yielded);
        assert_eq!(step.stress_mpa, Fix128::from_int(30));
        assert_eq!(state.equivalent_plastic_strain, Fix128::ZERO);
    }

    #[test]
    fn plastic_loading_isotropic() {
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        // Trial stress 60 MPa > yield 50 MPa
        let step = radial_return_1d(Fix128::from_int(60), &model, &mut state);
        assert!(step.yielded);
        assert!(step.plastic_strain_increment > Fix128::ZERO);
        // Stress should be pulled back near yield surface
        assert!(step.stress_mpa < Fix128::from_int(60));
        assert!(step.stress_mpa > Fix128::from_int(40));
    }

    #[test]
    fn isotropic_hardening_raises_next_yield() {
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        // First yield
        let _ = radial_return_1d(Fix128::from_int(60), &model, &mut state);
        let y1 = current_yield_mpa(&model, &state);
        // Second overload
        let _ = radial_return_1d(Fix128::from_int(70), &model, &mut state);
        let y2 = current_yield_mpa(&model, &state);
        assert!(y2 > y1, "isotropic hardening should grow σ_y");
    }

    #[test]
    fn kinematic_hardening_moves_back_stress() {
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla())
            .with_hardening(HardeningType::Kinematic);
        let mut state = PlasticState::default();
        // Tensile loading beyond yield
        let _ = radial_return_1d(Fix128::from_int(60), &model, &mut state);
        assert!(state.back_stress_mpa > Fix128::ZERO);
    }

    #[test]
    fn kinematic_hardening_bauschinger_reversal() {
        // After heavy tension the back-stress is positive. A subsequent
        // negative loading past yield must push the back-stress in the
        // negative direction (Bauschinger effect).
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla())
            .with_hardening(HardeningType::Kinematic);
        let mut state = PlasticState::default();
        // Heavy tension: apply trial stress way above yield to force plastic flow
        let _ = radial_return_1d(Fix128::from_int(500), &model, &mut state);
        let alpha_after_tension = state.back_stress_mpa;
        assert!(alpha_after_tension > Fix128::ZERO);
        // Heavy compression
        let _ = radial_return_1d(Fix128::from_int(-500), &model, &mut state);
        // Back-stress should have decreased (moved toward zero or negative)
        assert!(state.back_stress_mpa < alpha_after_tension);
    }

    #[test]
    fn combined_hardening_between_iso_and_kin() {
        let mat = MaterialProperties::pla();
        let model_iso = PlasticModel::from_fdm_material(&mat);
        let model_comb =
            PlasticModel::from_fdm_material(&mat).with_hardening(HardeningType::Combined);

        let mut s_iso = PlasticState::default();
        let mut s_comb = PlasticState::default();
        let _ = radial_return_1d(Fix128::from_int(60), &model_iso, &mut s_iso);
        let _ = radial_return_1d(Fix128::from_int(60), &model_comb, &mut s_comb);

        let y_iso = current_yield_mpa(&model_iso, &s_iso);
        let y_comb = current_yield_mpa(&model_comb, &s_comb);
        // Combined has half isotropic growth so yield should be intermediate
        assert!(y_comb > mat.yield_strength_mpa);
        assert!(y_comb < y_iso);
    }

    #[test]
    fn plastic_strain_accumulates_across_steps() {
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        for _ in 0..5 {
            let _ = radial_return_1d(Fix128::from_int(100), &model, &mut state);
        }
        // With H > 0, subsequent yields decrease Δλ but strain still grows
        assert!(state.equivalent_plastic_strain > Fix128::ZERO);
    }

    #[test]
    fn norton_creep_rate_scales_with_stress_cubed() {
        let creep = NortonCreep::pla_room_temp();
        let r1 = creep.strain_rate_per_s(Fix128::from_int(10));
        let r2 = creep.strain_rate_per_s(Fix128::from_int(20));
        // n=3, so 2× stress → 8× rate
        let ratio = r2 / r1;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(8),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn norton_creep_integration_accumulates() {
        let creep = NortonCreep::pla_room_temp();
        let mut state = PlasticState::default();
        // 10 MPa constant for 100 seconds
        creep.integrate(Fix128::from_int(10), Fix128::from_int(100), &mut state);
        assert!(state.creep_strain > Fix128::ZERO);
    }

    #[test]
    fn petg_creeps_slower_than_pla() {
        let pla = NortonCreep::pla_room_temp();
        let petg = NortonCreep::petg_room_temp();
        let sigma = Fix128::from_int(20);
        let r_pla = pla.strain_rate_per_s(sigma);
        let r_petg = petg.strain_rate_per_s(sigma);
        assert!(r_petg < r_pla);
    }

    #[test]
    fn creep_strain_independent_of_plastic_strain() {
        let creep = NortonCreep::pla_room_temp();
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        // Plastic loading first
        let _ = radial_return_1d(Fix128::from_int(70), &model, &mut state);
        let ep_before = state.equivalent_plastic_strain;
        // Creep on top
        creep.integrate(Fix128::from_int(30), Fix128::from_int(1000), &mut state);
        // Plastic strain unchanged, creep strain grew
        assert_eq!(state.equivalent_plastic_strain, ep_before);
        assert!(state.creep_strain > Fix128::ZERO);
    }

    /// Oracle: 1-D radial return by hand (Simo & Hughes eq. 2.13) for the PLA
    /// model (`σ_y = 50`, `E = 3500`, `H = 0.05·E = 175`) and a trial of
    /// `60 MPa`: `Δλ = (60 − 50) / (E + H) = 10/3675 = 2.7210884e-3`,
    /// corrected stress `60 − E·Δλ = 50.476190`. The hardening law chosen
    /// through `with_hardening` decides where `H·Δλ = 0.476190` goes:
    /// kinematic → all into the back-stress, combined → half
    /// (`0.238095`), isotropic → none. A second kinematic step with the same
    /// trial sees `|60 − α₁| = 59.523810` against the unchanged radius 50:
    /// `Δλ₂ = 9.523810/3675`, `α₂ = α₁ + 175·Δλ₂ = 0.929705`.
    #[test]
    fn with_hardening_back_stress_matches_radial_return_closed_form() {
        let base = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let (e, h) = (3500.0_f64, 175.0_f64);
        let dl1 = 10.0 / (e + h);
        let cases = [
            (HardeningType::Kinematic, h * dl1),
            (HardeningType::Combined, 0.5 * h * dl1),
            (HardeningType::Isotropic, 0.0),
        ];
        for (law, want_alpha) in cases {
            let model = base.with_hardening(law);
            assert_eq!(model.hardening_type, law);
            assert_eq!(model.yield_strength_mpa, base.yield_strength_mpa);
            assert_eq!(model.hardening_modulus_mpa, base.hardening_modulus_mpa);
            assert_eq!(model.youngs_modulus_mpa, base.youngs_modulus_mpa);
            let mut state = PlasticState::default();
            let step = radial_return_1d(Fix128::from_int(60), &model, &mut state);
            assert!(step.yielded);
            let got_alpha = state.back_stress_mpa.to_f64();
            assert!(
                (got_alpha - want_alpha).abs() < 1e-9,
                "{law:?}: α = {got_alpha}, want {want_alpha}"
            );
            let got_ep = state.equivalent_plastic_strain.to_f64();
            assert!((got_ep - dl1).abs() < 1e-9, "{law:?}: ε_p = {got_ep}");
            let got_s = step.stress_mpa.to_f64();
            assert!(
                (got_s - (60.0 - e * dl1)).abs() < 1e-9,
                "{law:?}: σ = {got_s}"
            );
        }
        // second kinematic step: Bauschinger shift of the surface centre
        let model = base.with_hardening(HardeningType::Kinematic);
        let mut state = PlasticState::default();
        let _ = radial_return_1d(Fix128::from_int(60), &model, &mut state);
        let _ = radial_return_1d(Fix128::from_int(60), &model, &mut state);
        let alpha1 = h * dl1;
        let dl2 = (60.0 - alpha1 - 50.0) / (e + h);
        let want_alpha2 = alpha1 + h * dl2;
        let got_alpha2 = state.back_stress_mpa.to_f64();
        assert!(
            (got_alpha2 - want_alpha2).abs() < 1e-9,
            "α₂ = {got_alpha2}, want {want_alpha2}"
        );
    }

    /// Degenerate use of the builder: re-applying the model's own law is the
    /// identity, applying a law twice equals applying it once, and
    /// overriding back restores the original (only `hardening_type` moves).
    #[test]
    fn with_hardening_is_idempotent_and_reversible() {
        let base = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        assert_eq!(base.with_hardening(HardeningType::Isotropic), base);
        assert_eq!(
            base.with_hardening(HardeningType::Kinematic)
                .with_hardening(HardeningType::Kinematic),
            base.with_hardening(HardeningType::Kinematic)
        );
        assert_eq!(
            base.with_hardening(HardeningType::Combined)
                .with_hardening(HardeningType::Isotropic),
            base
        );
    }

    /// Oracle: Norton `ε_c = A·σⁿ·Δt` by hand with the PETG preset
    /// (`A = 1e-13`, `n = 3`): `σ = 20 MPa`, `Δt = 10⁶ s` →
    /// `1e-13 · 8000 · 10⁶ = 8e-4`. The exponent is odd, so a compressive
    /// `−20 MPa` gives `−8e-4` (signed creep). Tolerance: `A = 1e-13` is
    /// `1 844 674.4` Fix128 ulp stored truncated (relative `2.2e-7`, i.e.
    /// `1.8e-10` absolute here); `1e-9` leaves no room for a wrong constant.
    #[test]
    fn petg_norton_preset_integrates_a_sigma_cubed_dt() {
        let creep = NortonCreep::petg_room_temp();
        let dt = Fix128::from_int(1_000_000);
        let mut state = PlasticState::default();
        creep.integrate(Fix128::from_int(20), dt, &mut state);
        let got = state.creep_strain.to_f64();
        assert!((got - 8e-4).abs() < 1e-9, "got {got}");
        let mut neg = PlasticState::default();
        creep.integrate(Fix128::from_int(-20), dt, &mut neg);
        let got_neg = neg.creep_strain.to_f64();
        assert!((got_neg + 8e-4).abs() < 1e-9, "got {got_neg}");
    }

    /// Degenerate inputs for the PETG preset: zero stress and zero time step
    /// leave the creep strain exactly unchanged (`ε̇ = 0` / `Δt = 0`).
    #[test]
    fn petg_norton_preset_degenerate_stress_or_dt_leaves_state() {
        let creep = NortonCreep::petg_room_temp();
        let mut state = PlasticState {
            creep_strain: Fix128::from_ratio(1, 1000),
            ..Default::default()
        };
        creep.integrate(Fix128::ZERO, Fix128::from_int(1_000_000), &mut state);
        assert_eq!(state.creep_strain, Fix128::from_ratio(1, 1000));
        creep.integrate(Fix128::from_int(20), Fix128::ZERO, &mut state);
        assert_eq!(state.creep_strain, Fix128::from_ratio(1, 1000));
    }

    #[test]
    fn uniaxial_von_mises_matches_beam_stress() {
        // Sanity check: a uniaxial stress state via StressTensor gives the same
        // magnitude as feeding it directly to a plasticity check.
        let stress = StressTensor::uniaxial_x(Fix128::from_int(45));
        let vm = stress.von_mises();
        let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
        let mut state = PlasticState::default();
        // 45 MPa < 50 MPa yield → elastic
        let step = radial_return_1d(vm, &model, &mut state);
        assert!(!step.yielded);
    }
}
