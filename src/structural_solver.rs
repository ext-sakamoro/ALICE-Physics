//! Structural (Solid Mechanics) Time-Stepping Solver (Session 3 S2)
//!
//! Composes Session 1-2 strength / plasticity / creep / fatigue / buckling
//! models into a single time-stepping loop that follows a component through
//! its service life:
//!
//! 1. Apply the current load case → nominal stress via `beam_stress`.
//! 2. Yield check via `plastic::radial_return_1d` (permanent strain
//!    accumulates each step).
//! 3. Creep integration via `plastic::NortonCreep` and the long-term
//!    Findley model from `creep_longterm` (temperature-shifted).
//! 4. Fatigue damage accumulation via Miner's rule (`fatigue`).
//! 5. Buckling check via `buckling::analyze_column` — instability trip.
//!
//! Output: `StructuralReport` per step and a final `StructuralHistory`
//! with lifetime totals.

use crate::beam_stress::{BeamAnalysis, ColumnEndCondition, CrossSection, LoadCase};
use crate::buckling::{analyze_column, BucklingRegime, ColumnBucklingReport};
use crate::creep_longterm::{predict_strain, FindleyParameters};
use crate::fatigue::{miner_damage, SnCurve, SpectrumEntry};
use crate::filament_db::MaterialProperties;
use crate::math::Fix128;
use crate::plastic::{radial_return_1d, NortonCreep, PlasticModel, PlasticState};

/// Per-step diagnostic snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructuralReport {
    /// Elapsed simulation time (hours).
    pub elapsed_hours: Fix128,
    /// Applied bending stress (MPa).
    pub bending_stress_mpa: Fix128,
    /// Column buckling factor of safety (P_cr / applied axial).
    pub buckling_fos: Fix128,
    /// Current plastic strain (dimensionless).
    pub plastic_strain: Fix128,
    /// Cumulative creep strain (dimensionless).
    pub creep_strain: Fix128,
    /// Cumulative fatigue damage (0 to > 1, where 1 = failure).
    pub fatigue_damage: Fix128,
    /// True iff current step tripped a yield, buckling, or fatigue failure.
    pub failed_this_step: bool,
    /// Aggregate `is_safe` (no failures throughout the run so far).
    pub is_safe: bool,
}

/// Lifetime accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StructuralHistory {
    /// Total elapsed time (hours).
    pub elapsed_hours: Fix128,
    /// First step index at which any failure was detected, if any.
    pub failure_step: Option<u64>,
    /// Final plastic state.
    pub plastic_state: PlasticState,
    /// Final Miner damage `D`.
    pub fatigue_damage: Fix128,
    /// Number of steps advanced.
    pub steps: u64,
}

/// Structural solver configuration bundle.
#[derive(Clone, Copy, Debug)]
pub struct StructuralSolver {
    /// Beam geometry (used for bending stress and axial buckling).
    pub section: CrossSection,
    /// Bending load applied every step.
    pub load: LoadCase,
    /// Axial column load (N) applied every step (0 = pure bending).
    pub axial_load_n: Fix128,
    /// Column length for buckling check (mm).
    pub column_length_mm: Fix128,
    /// End condition for buckling check.
    pub end_condition: ColumnEndCondition,
    /// Filament material.
    pub material: MaterialProperties,
    /// Plastic constitutive model.
    pub plastic_model: PlasticModel,
    /// Fatigue S-N curve.
    pub sn_curve: SnCurve,
    /// Long-term creep parameters.
    pub creep_params: FindleyParameters,
    /// Norton creep (short-term); used for the plastic-state creep field.
    pub norton_creep: NortonCreep,
    /// Operating temperature (°C) used for time-temperature shift.
    pub operating_temp_c: Fix128,
    /// Time-step size (seconds).
    pub dt_s: Fix128,
    /// State carried across steps.
    pub state: PlasticState,
    /// Total elapsed simulation time (hours).
    pub elapsed_hours: Fix128,
    /// Accumulated fatigue damage (Miner).
    pub fatigue_damage: Fix128,
    /// Failure step marker (once set, remains set).
    pub failure_step: Option<u64>,
    /// Step counter.
    pub step_count: u64,
}

impl StructuralSolver {
    /// Construct with `from_fdm_material` defaults for the given material and
    /// bending load case. Axial load defaults to zero (pure bending).
    #[must_use]
    pub fn new(section: CrossSection, load: LoadCase, material: MaterialProperties) -> Self {
        Self {
            section,
            load,
            axial_load_n: Fix128::ZERO,
            column_length_mm: load.length_mm(),
            end_condition: ColumnEndCondition::PinPin,
            material,
            plastic_model: PlasticModel::from_fdm_material(&material),
            sn_curve: SnCurve::from_fdm_material(&material),
            creep_params: FindleyParameters::pla_25c_moderate(),
            norton_creep: NortonCreep::pla_room_temp(),
            operating_temp_c: Fix128::from_int(25),
            dt_s: Fix128::from_ratio(3600, 1), // default 1 hour steps
            state: PlasticState::default(),
            elapsed_hours: Fix128::ZERO,
            fatigue_damage: Fix128::ZERO,
            failure_step: None,
            step_count: 0,
        }
    }

    /// Advance one step and return the current diagnostics.
    pub fn step(&mut self) -> StructuralReport {
        // 1. Bending stress from current beam / load
        let beam = BeamAnalysis::new(self.section, self.load, self.material);
        let beam_report = beam.analyze();
        let sigma = beam_report.max_bending_stress_mpa;

        // 2. Plasticity update (radial return; simplified as if this stress
        //    is applied as a trial each step). Steady-state cyclic loading
        //    can be modelled by a stress spectrum via `fatigue`.
        let step = radial_return_1d(sigma, &self.plastic_model, &mut self.state);
        let yielded_this_step = step.yielded;

        // 3. Creep — accumulate short-term Norton over one dt window.
        self.norton_creep
            .integrate(sigma, self.dt_s, &mut self.state);
        // Also compute the long-term Findley projection at the operating
        // temperature for reporting (does not feed back into radial return).
        // `state.creep_strain` stays the pure Norton accumulation; the report
        // carries max(Norton, Findley). Before 1.2.0 the max was written back
        // into the state, so the next Norton increment landed on top of the
        // Findley value and the reported creep was neither model.
        let long_term_creep_strain = predict_strain(
            &self.creep_params,
            &self.material,
            self.elapsed_hours,
            self.operating_temp_c,
        );
        let reported_creep = if long_term_creep_strain > self.state.creep_strain {
            long_term_creep_strain
        } else {
            self.state.creep_strain
        };

        // 4. Fatigue: apply the current stress as one cycle.
        let spectrum: [SpectrumEntry; 1] = [(sigma, 1)];
        let dd = miner_damage(&spectrum, &self.sn_curve);
        self.fatigue_damage = self.fatigue_damage + dd;

        // 5. Buckling: compute critical load and FoS.
        let bp: ColumnBucklingReport = analyze_column(
            &self.section,
            self.column_length_mm,
            self.end_condition,
            &self.material,
        );
        let buckling_fos = if self.axial_load_n.is_zero() {
            Fix128::from_int(i64::MAX >> 8)
        } else {
            bp.critical_load_n / self.axial_load_n
        };
        let buckled = buckling_fos < Fix128::ONE && bp.regime != BucklingRegime::Yielding;

        // Failure conditions
        let fatigue_failed = self.fatigue_damage >= Fix128::ONE;
        let overloaded = !beam_report.is_safe;
        let failed_now = yielded_this_step || fatigue_failed || overloaded || buckled;
        if failed_now && self.failure_step.is_none() {
            self.failure_step = Some(self.step_count);
        }

        // Advance clock (dt_s to hours)
        self.elapsed_hours = self.elapsed_hours + self.dt_s / Fix128::from_int(3600);
        self.step_count += 1;

        StructuralReport {
            elapsed_hours: self.elapsed_hours,
            bending_stress_mpa: sigma,
            buckling_fos,
            plastic_strain: self.state.equivalent_plastic_strain,
            creep_strain: reported_creep,
            fatigue_damage: self.fatigue_damage,
            failed_this_step: failed_now,
            is_safe: self.failure_step.is_none(),
        }
    }

    /// Run for `n` steps and return a snapshot of the final state.
    pub fn run(&mut self, n_steps: u64) -> StructuralHistory {
        for _ in 0..n_steps {
            let _ = self.step();
        }
        StructuralHistory {
            elapsed_hours: self.elapsed_hours,
            failure_step: self.failure_step,
            plastic_state: self.state,
            fatigue_damage: self.fatigue_damage,
            steps: self.step_count,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_new_defaults() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(5),
            length_mm: Fix128::from_int(200),
        };
        let solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        assert_eq!(solver.step_count, 0);
        assert_eq!(solver.elapsed_hours, Fix128::ZERO);
        assert!(solver.failure_step.is_none());
    }

    #[test]
    fn step_advances_time() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(5),
            length_mm: Fix128::from_int(200),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        let r = solver.step();
        assert!(r.elapsed_hours > Fix128::ZERO);
        assert_eq!(solver.step_count, 1);
    }

    #[test]
    fn safe_light_load_stays_safe() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(2),
            length_mm: Fix128::from_int(200),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        let history = solver.run(10);
        assert!(history.failure_step.is_none());
        assert!(history.fatigue_damage < Fix128::ONE);
    }

    #[test]
    fn overloaded_beam_flags_failure() {
        // Section that yields at 5 N load already
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(5),
            height_mm: Fix128::from_int(5),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(100),
            length_mm: Fix128::from_int(300),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        let r = solver.step();
        assert!(r.failed_this_step);
        assert_eq!(solver.failure_step, Some(0));
    }

    #[test]
    fn buckling_flags_when_axial_load_exceeds_cr() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(5),
            height_mm: Fix128::from_int(5),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(1),
            length_mm: Fix128::from_int(500),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        solver.axial_load_n = Fix128::from_int(10_000); // huge axial
        let r = solver.step();
        assert!(r.buckling_fos < Fix128::ONE);
    }

    #[test]
    fn run_produces_history_snapshot() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(5),
            length_mm: Fix128::from_int(200),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        let history = solver.run(5);
        assert_eq!(history.steps, 5);
        assert!(history.elapsed_hours > Fix128::ZERO);
    }

    #[test]
    fn creep_strain_grows_over_time() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(3),
            length_mm: Fix128::from_int(200),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        solver.operating_temp_c = Fix128::from_int(55); // close to PLA Tg
        let r1 = solver.step();
        let r_later = solver.run(10);
        // the Norton accumulation in the state grows on its own (1.2.0: it is
        // no longer overwritten by the Findley projection), and the reported
        // creep of a later step is monotone
        let r11 = solver.step();
        assert!(r_later.plastic_state.creep_strain > Fix128::ZERO);
        assert!(r11.creep_strain >= r1.creep_strain);
    }

    #[test]
    fn fatigue_damage_accumulates_each_step() {
        let section = CrossSection::Rectangular {
            width_mm: Fix128::from_int(5),
            height_mm: Fix128::from_int(5),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(20),
            length_mm: Fix128::from_int(200),
        };
        let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
        let r1 = solver.step();
        let r2 = solver.step();
        assert!(r2.fatigue_damage >= r1.fatigue_damage);
    }
}
