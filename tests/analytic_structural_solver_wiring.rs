//! Independent oracles for `structural_solver::{StructuralSolver::run,
//! StructuralHistory}` and the loop they summarise: bending stress, yield
//! (radial return), Norton + Findley creep, Miner fatigue and the column
//! buckling trip, one step per `dt_s`.
//!
//! Closed forms (PLA: E = 3500 MPa, yield 50 MPa, UTS 60 MPa):
//!
//! ```text
//! sigma          = M / Z                       Z = b h^2 / 6
//! yield          e_{n+1} = e_n E/(E+H) + (sigma - sy)/(E+H),  H = 0.05 E
//!                -> e_n = (sigma - sy)/H * (1 - (E/(E+H))^n)
//! Norton         eps_c(n) = n * A sigma^3 dt          A = 6.34e-13, n = 3
//! Findley        eps(t) = e0 + m t^3,  e0 = 0.003, m = 8.3e-14 (t in hours)
//! fatigue        N = N_e (S_e/S)^5,  S_e = 0.3 UTS = 18,  N_e = 1e6;  D += 1/N
//! Euler column   sigma_cr = pi^2 E / (K L / r)^2,  r = sqrt(I/A)
//! Johnson        sigma_cr = sy (1 - sy lambda^2 / (4 pi^2 E))
//! ```
//!
//! Nothing here touches `src/`.

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{ColumnEndCondition, CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::{StructuralHistory, StructuralSolver};
use core::f64::consts::PI;

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn rect(w: i64, h: i64) -> CrossSection {
    CrossSection::Rectangular {
        width_mm: int(w),
        height_mm: int(h),
    }
}

fn cantilever(load: i64, len: i64) -> LoadCase {
    LoadCase::CantileverEndPoint {
        load_n: int(load),
        length_mm: int(len),
    }
}

fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let e = ((g - want) / want).abs();
    assert!(e <= tol, "{what}: got {g}, want {want} (rel {e:.3e})");
}

/// 15x20 section: `Z = 15*400/6 = 1000 mm^3`, so a 100 N tip load on 200 mm is
/// `sigma = 20000/1000 = 20 MPa` (FoS 2.5, elastic, above the 18 MPa fatigue
/// endurance of PLA).
fn solver_at_20_mpa() -> StructuralSolver {
    StructuralSolver::new(
        rect(15, 20),
        cantilever(100, 200),
        MaterialProperties::pla(),
    )
}

/// Per-step Miner damage at 20 MPa: `N = 1e6 * (18/20)^5 = 590490` cycles
/// (Fix128 floors the cycle count, so 590489 or 590490).
const D20: f64 = 1.0 / 590_490.0;

// ---------------------------------------------------------------------------
// run / StructuralHistory
// ---------------------------------------------------------------------------

/// `run(n)` is `n` calls of `step` and returns the final state as a
/// `StructuralHistory`: steps, elapsed hours (`n * dt / 3600`), fatigue damage
/// (`n` times the per-step Miner term), Norton creep (`n * A sigma^3 dt`), no
/// plastic strain, no failure.
#[test]
fn run_history_matches_closed_forms_for_an_elastic_beam() {
    let mut s = solver_at_20_mpa();
    s.dt_s = int(7200); // 2 h per step
    let h = s.run(10);
    assert_eq!(h.steps, 10);
    assert_eq!(h.elapsed_hours, int(20));
    assert_eq!(h.failure_step, None);
    assert_rel(
        h.fatigue_damage,
        10.0 * D20,
        2e-5,
        "Miner damage after 10 steps",
    );
    assert_eq!(h.plastic_state.equivalent_plastic_strain, Fix128::ZERO);
    assert_eq!(h.plastic_state.back_stress_mpa, Fix128::ZERO);
    let norton = 10.0 * 6.34e-13 * 20.0f64.powi(3) * 7200.0;
    assert_rel(
        h.plastic_state.creep_strain,
        norton,
        2e-7,
        "Norton creep (A is quantised to 2^-64)",
    );
    // history mirrors the solver's own counters
    assert_eq!(h.elapsed_hours, s.elapsed_hours);
    assert_eq!(h.fatigue_damage, s.fatigue_damage);
    assert_eq!(h.plastic_state, s.state);
    assert_eq!(h.steps, s.step_count);
}

/// `run` accumulates across calls (it does not reset): 3 + 2 steps equals 5
/// steps, bit for bit, and `run(0)` is a pure snapshot.
#[test]
fn run_is_cumulative_and_zero_steps_is_a_snapshot() {
    let mut a = solver_at_20_mpa();
    let _ = a.run(3);
    let snap = a.run(0);
    assert_eq!(snap.steps, 3);
    let h_a = a.run(2);

    let mut b = solver_at_20_mpa();
    let h_b = b.run(5);
    assert_eq!(h_a, h_b);
    assert_eq!(h_a.steps, 5);

    let fresh = solver_at_20_mpa().run(0);
    assert_eq!(fresh, StructuralHistory::default());
    assert_eq!(fresh.failure_step, None);
}

/// `run` and a hand loop of `step` agree exactly (the same code path).
#[test]
fn run_equals_repeated_step() {
    let mut a = solver_at_20_mpa();
    let h = a.run(7);
    let mut b = solver_at_20_mpa();
    let mut last = b.step();
    for _ in 1..7 {
        last = b.step();
    }
    assert_eq!(h.fatigue_damage, last.fatigue_damage);
    assert_eq!(h.elapsed_hours, last.elapsed_hours);
    assert_eq!(
        h.plastic_state.equivalent_plastic_strain,
        last.plastic_strain
    );
    assert_eq!(h.failure_step.is_none(), last.is_safe);
}

// ---------------------------------------------------------------------------
// the loop: stress, yield, creep, fatigue, failure marker
// ---------------------------------------------------------------------------

/// Step report at 20 MPa: stress `M/Z`, clock `(k+1) dt/3600` hours, no yield,
/// no plastic strain, buckling FoS is the "infinite" sentinel for zero axial
/// load, and the first step's creep is the Findley instantaneous strain
/// `e0 = 0.003` (the Norton term is 1.8e-5 and smaller).
#[test]
fn first_step_report() {
    let mut s = solver_at_20_mpa();
    let r = s.step();
    assert_rel(r.bending_stress_mpa, 20.0, 1e-12, "sigma = M/Z");
    assert_eq!(r.elapsed_hours, int(1));
    assert_eq!(r.plastic_strain, Fix128::ZERO);
    assert_eq!(r.buckling_fos, int(i64::MAX >> 8));
    assert!(!r.failed_this_step && r.is_safe);
    assert_rel(r.creep_strain, 0.003, 1e-12, "Findley e0");
    assert_rel(r.fatigue_damage, D20, 2e-5, "one cycle at 20 MPa");
}

/// Below the 18 MPa endurance (0.3 UTS) a cycle does no fatigue damage: 10 MPa
/// (`Z = 1000`, `M = 10000`) accumulates exactly zero.
#[test]
fn stress_below_the_endurance_limit_does_no_fatigue_damage() {
    let mut s = StructuralSolver::new(rect(15, 20), cantilever(50, 200), MaterialProperties::pla());
    let h = s.run(50);
    assert_eq!(h.fatigue_damage, Fix128::ZERO);
    assert_eq!(h.failure_step, None);
}

/// Yielding: at 60 MPa (`M = 60000`, `Z = 1000`) against yield 50 and
/// `H = 0.05 E = 175`, the plastic strain after `n` steps is
/// `(sigma - sy)/H * (1 - (E/(E+H))^n)`. The first step alone is
/// `10/3675 = 2.7211e-3`; the step flags the failure and fixes the marker.
#[test]
fn yielding_accumulates_plastic_strain_by_the_radial_return_recurrence() {
    let mut s = StructuralSolver::new(
        rect(15, 20),
        cantilever(300, 200),
        MaterialProperties::pla(),
    );
    let r0 = s.step();
    assert_rel(
        r0.plastic_strain,
        10.0 / 3675.0,
        1e-12,
        "first radial return",
    );
    assert!(r0.failed_this_step);
    assert_eq!(s.failure_step, Some(0));
    let h = s.run(19); // 20 steps in total
    let ratio = 3500.0 / 3675.0;
    let want = (10.0 / 175.0) * (1.0 - f64::powi(ratio, 20));
    assert_rel(
        h.plastic_state.equivalent_plastic_strain,
        want,
        1e-9,
        "e_20",
    );
    assert_eq!(
        h.failure_step,
        Some(0),
        "marker stays at the first failing step"
    );
    assert_eq!(h.steps, 20);
}

/// 30 MPa is elastic (below yield 50) but the beam FoS is `50/30 = 1.67 < 2`,
/// so the step is flagged without any plastic strain.
#[test]
fn low_beam_factor_of_safety_trips_the_step_without_yielding() {
    let mut s = StructuralSolver::new(
        rect(15, 20),
        cantilever(150, 200),
        MaterialProperties::pla(),
    );
    let r = s.step();
    assert_rel(r.bending_stress_mpa, 30.0, 1e-12, "sigma");
    assert_eq!(r.plastic_strain, Fix128::ZERO);
    assert!(r.failed_this_step);
    assert!(!r.is_safe);
    assert_eq!(s.failure_step, Some(0));
}

/// Miner failure: with the damage pre-set so that it crosses 1 on the fourth
/// step, steps 0-2 are clean, step 3 trips, and `failure_step` stays 3 while
/// later steps keep reporting the failure.
#[test]
fn fatigue_failure_marks_the_crossing_step_and_stays() {
    let mut s = solver_at_20_mpa();
    s.fatigue_damage = Fix128::from_f64(1.0 - 3.5 * D20);
    for k in 0..3 {
        let r = s.step();
        assert!(!r.failed_this_step, "step {k}");
        assert!(r.is_safe, "step {k}");
    }
    assert_eq!(s.failure_step, None);
    let r3 = s.step();
    assert!(r3.failed_this_step);
    assert!(!r3.is_safe);
    assert!(r3.fatigue_damage >= Fix128::ONE);
    assert_eq!(s.failure_step, Some(3));
    let r4 = s.step();
    assert!(r4.failed_this_step);
    assert_eq!(s.failure_step, Some(3), "set once, never overwritten");
    let h = s.run(0);
    assert_eq!(h.failure_step, Some(3));
}

// ---------------------------------------------------------------------------
// creep: Norton vs Findley, temperature
// ---------------------------------------------------------------------------

/// Reported creep is `max(Norton state, Findley(t))`. At 1 h steps the Findley
/// strain `0.003 + 8.3e-14 t^3` dominates; at 1000 h steps the accumulated
/// Norton term `(k+1) * A sigma^3 dt = 0.01826 (k+1)` overtakes it.
#[test]
fn reported_creep_is_the_larger_of_norton_and_findley() {
    // Findley dominates (1 h steps, step k is evaluated at t = k hours)
    let mut s = solver_at_20_mpa();
    for k in 0..5 {
        let r = s.step();
        let t = f64::from(k);
        assert_rel(
            r.creep_strain,
            0.003 + 8.3e-14 * t * t * t,
            1e-12,
            "Findley regime",
        );
    }
    // Norton dominates (1000 h steps)
    let mut s = solver_at_20_mpa();
    s.dt_s = int(3_600_000);
    for k in 1..=3 {
        let r = s.step();
        let norton = f64::from(k) * 6.34e-13 * 8000.0 * 3.6e6;
        assert_rel(
            r.creep_strain,
            norton,
            2e-7,
            "Norton regime (A quantised to 2^-64)",
        );
    }
}

/// Above `T_g` (PLA 60 C) the Findley clock runs on the WLF-shifted time
/// `t / a_T`, `log10 a_T = -C1 dT/(C2 + dT)` with the universal `C1 = 17.44`,
/// `C2 = 51.6`. At 80 C (`dT = 20`) and `t = 3 h`:
/// `eps = e0 + m (t/a_T)^3`.
#[test]
fn creep_above_glass_transition_uses_the_wlf_shifted_time() {
    let mut s = solver_at_20_mpa();
    s.operating_temp_c = int(80);
    for _ in 0..3 {
        let _ = s.step();
    }
    let r = s.step(); // evaluated at t = 3 h
    let dt = 20.0f64;
    let log_at = -17.44 * dt / (51.6 + dt);
    let a_t = 10.0f64.powf(log_at);
    let t_eff = 3.0 / a_t;
    let want = 0.003 + 8.3e-14 * t_eff * t_eff * t_eff;
    assert_rel(r.creep_strain, want, 1e-6, "WLF-shifted Findley");
    // far above the room-temperature value
    assert!(r.creep_strain.to_f64() > 100.0 * 0.003);
}

// ---------------------------------------------------------------------------
// buckling trip and end condition
// ---------------------------------------------------------------------------

/// 10x10 column of 500 mm (`r = 2.8868`, `lambda = 173.2` pin-pin, Euler):
/// `P_cr = pi^2 E I / L^2 = 115.15 N`. At half that axial load the FoS is 2
/// (no trip); at twice, 0.5 (trip). The `end_condition` field sets `K`:
/// cantilever (`K = 2`) cuts `P_cr` to a quarter, so the same half-load now
/// trips.
#[test]
fn axial_load_trips_buckling_and_end_condition_sets_the_critical_load() {
    let i = 10.0 * 1000.0 / 12.0;
    let p_cr = PI * PI * 3500.0 * i / (500.0 * 500.0);
    let make = |ec: ColumnEndCondition, axial: f64| {
        let mut s =
            StructuralSolver::new(rect(10, 10), cantilever(1, 100), MaterialProperties::pla());
        s.column_length_mm = int(500);
        s.end_condition = ec;
        s.axial_load_n = Fix128::from_f64(axial);
        s
    };
    let r = make(ColumnEndCondition::PinPin, p_cr / 2.0).step();
    assert_rel(r.buckling_fos, 2.0, 1e-9, "pin-pin FoS at half load");
    assert!(!r.failed_this_step);

    let r = make(ColumnEndCondition::PinPin, 2.0 * p_cr).step();
    assert_rel(r.buckling_fos, 0.5, 1e-9, "pin-pin FoS at double load");
    assert!(r.failed_this_step && !r.is_safe);

    let r = make(ColumnEndCondition::Cantilever, p_cr / 2.0).step();
    assert_rel(r.buckling_fos, 0.5, 1e-9, "cantilever column, K = 2");
    assert!(r.failed_this_step);

    let r = make(ColumnEndCondition::FixedFixed, p_cr / 2.0).step();
    assert_rel(r.buckling_fos, 8.0, 1e-9, "fixed-fixed column, K = 1/2");
    assert!(!r.failed_this_step);
}

/// A stocky column (L = 20 mm, `lambda = 6.93` below the transition 37.2) is
/// in the Johnson regime: `sigma_cr = sy (1 - sy lambda^2 / (4 pi^2 E))` =
/// 49.13 MPa, `P_cr = 4913 N`; 6000 N trips it.
#[test]
fn stocky_column_uses_the_johnson_critical_load() {
    let lambda2 = 20.0f64 * 20.0 / (100.0 / 12.0);
    let sigma_cr = 50.0 * (1.0 - 50.0 * lambda2 / (4.0 * PI * PI * 3500.0));
    let p_cr = sigma_cr * 100.0;
    let mut s = StructuralSolver::new(rect(10, 10), cantilever(1, 100), MaterialProperties::pla());
    s.column_length_mm = int(20);
    s.axial_load_n = int(6000);
    let r = s.step();
    assert_rel(r.buckling_fos, p_cr / 6000.0, 1e-9, "Johnson FoS");
    assert!(r.failed_this_step);
}

// ---------------------------------------------------------------------------
// construction defaults
// ---------------------------------------------------------------------------

/// `new` documents `from_fdm_material` defaults: zero axial load, pin-pin
/// column the length of the load span, 25 C, one-hour steps, clean state.
#[test]
fn new_sets_the_documented_defaults() {
    let s = StructuralSolver::new(rect(10, 20), cantilever(5, 200), MaterialProperties::pla());
    assert_eq!(s.axial_load_n, Fix128::ZERO);
    assert_eq!(s.column_length_mm, int(200));
    assert_eq!(s.end_condition, ColumnEndCondition::PinPin);
    assert_eq!(s.operating_temp_c, int(25));
    assert_eq!(s.dt_s, int(3600));
    assert_eq!(s.step_count, 0);
    assert_eq!(s.elapsed_hours, Fix128::ZERO);
    assert_eq!(s.fatigue_damage, Fix128::ZERO);
    assert_eq!(s.failure_step, None);
    // material-derived: H = 0.05 E, S_e = 0.3 UTS, N_e = 1e6, m = 5
    assert_rel(s.plastic_model.hardening_modulus_mpa, 175.0, 1e-12, "H");
    assert_rel(s.plastic_model.youngs_modulus_mpa, 3500.0, 1e-12, "E");
    assert_rel(s.sn_curve.endurance_stress_mpa, 18.0, 1e-12, "S_e");
    assert_eq!(s.sn_curve.endurance_cycles, 1_000_000);
    assert_eq!(s.sn_curve.fatigue_exponent_m, 5);
}

// ---------------------------------------------------------------------------
// boundaries
// ---------------------------------------------------------------------------

/// Failure is `D >= 1`: a step that lands exactly on `D = 1` fails, one ulp
/// below does not.
#[test]
fn fatigue_failure_threshold_is_inclusive() {
    let mut probe = solver_at_20_mpa();
    let d = probe.step().fatigue_damage; // exact per-step damage in Fix128

    let mut s = solver_at_20_mpa();
    s.fatigue_damage = Fix128::ONE - d;
    let r = s.step();
    assert_eq!(r.fatigue_damage, Fix128::ONE);
    assert!(r.failed_this_step, "D == 1 is a failure");

    let mut s = solver_at_20_mpa();
    s.fatigue_damage = Fix128::ONE - d - Fix128::from_raw(0, 1);
    let r = s.step();
    assert!(r.fatigue_damage < Fix128::ONE);
    assert!(!r.failed_this_step, "D just below 1 is not");
}

/// Buckling trips on `FoS < 1`: an axial load exactly equal to `P_cr` gives
/// `FoS = 1` and does not trip; one ulp more does.
#[test]
fn buckling_trip_is_strict() {
    use alice_physics::buckling::analyze_column;
    let section = rect(10, 10);
    let p_cr = analyze_column(
        &section,
        int(500),
        ColumnEndCondition::PinPin,
        &MaterialProperties::pla(),
    )
    .critical_load_n;
    let make = |axial: Fix128| {
        let mut s = StructuralSolver::new(section, cantilever(1, 100), MaterialProperties::pla());
        s.column_length_mm = int(500);
        s.axial_load_n = axial;
        s
    };
    let r = make(p_cr).step();
    assert_eq!(r.buckling_fos, Fix128::ONE);
    assert!(!r.failed_this_step);
    let r = make(p_cr + Fix128::from_raw(0, 1 << 40)).step();
    assert!(r.buckling_fos < Fix128::ONE);
    assert!(r.failed_this_step);
}

/// `is_safe` in the report is the run-so-far verdict, not this step's: after a
/// failure, a later step with a harmless load still reports `is_safe == false`
/// while `failed_this_step` is false.
#[test]
fn is_safe_is_sticky_after_a_failure() {
    let mut s = StructuralSolver::new(
        rect(15, 20),
        cantilever(300, 200),
        MaterialProperties::pla(),
    );
    assert!(s.step().failed_this_step);
    s.load = cantilever(1, 200); // 0.2 MPa: nothing fails now
    s.state = Default::default(); // and no residual plasticity
    let r = s.step();
    assert!(!r.failed_this_step);
    assert!(!r.is_safe, "an earlier failure keeps the run unsafe");
    assert_eq!(s.failure_step, Some(0));
}
