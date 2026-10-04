//! Audit oracles for cfd_solver: the refusal messages carry the measured
//! numbers they document, `step_count` advances by exactly one per step on
//! every entry point and not at all on a refusal, and a resting fluid with no
//! boundary conditions falls freely at `v = g t` (AUD-C-S1W4-013).
//!
//! The existing checks were `!to_string().is_empty()`, a bare `step_count`
//! after one call, and "some v face is negative" after a step under gravity.
//! Each is replaced here by a value fixed outside the implementation.

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, PressureSolverError, RansState, StepError, StepOptions,
    TurbulenceModel,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::turbulence::WallFunctionError;

fn q(a: i64, b: i64) -> Fix128 {
    Fix128::from_ratio(a, b)
}

fn opts() -> StepOptions {
    StepOptions::new(PressureSolver::RedBlackGs { sweeps: 4 })
}

// ------------------------------------------------------------- Display

#[test]
fn step_error_pressure_display_is_the_prefix_followed_by_the_inner_message() {
    // The wrapper documents itself as "the pressure solver request was
    // refused": the inner message must survive verbatim behind one prefix.
    let inner = [
        PressureSolverError::ZeroTimeStep,
        PressureSolverError::ZeroDensity,
        PressureSolverError::ZeroSpacing,
        PressureSolverError::ZeroIterations,
        PressureSolverError::MultigridNeedsPowerOfTwoExtents { extents: (3, 5, 6) },
        PressureSolverError::NonPositiveTolerance,
        PressureSolverError::ZeroRanks,
    ];
    for e in inner {
        let wrapped = StepError::Pressure(e).to_string();
        let own = e.to_string();
        assert!(!own.is_empty(), "{e:?} has an empty message");
        assert_eq!(
            wrapped,
            format!("pressure solver: {own}"),
            "{e:?}: the wrapper must be one prefix plus the inner text"
        );
        // From<PressureSolverError> is the same wrapper
        assert_eq!(StepError::from(e).to_string(), wrapped);
    }
}

#[test]
fn multigrid_extent_refusal_prints_the_three_extents_in_axis_order() {
    // The extents are the measured input; a message that reorders or drops
    // them would point the caller at the wrong axis.
    for (nx, ny, nz) in [(3usize, 5usize, 6usize), (8, 7, 16), (1, 2, 9)] {
        let text = PressureSolverError::MultigridNeedsPowerOfTwoExtents {
            extents: (nx, ny, nz),
        }
        .to_string();
        let want = format!("{nx}x{ny}x{nz}");
        assert!(
            text.ends_with(&want),
            "message {text:?} must end with the extents {want}"
        );
    }
}

#[test]
fn diffusion_refusal_prints_the_measured_number_and_the_one_sixth_limit() {
    // Dyadic numbers have an exact decimal form, so the printed number is
    // fixed: 1/4 -> 0.25, 3/8 -> 0.375, 5/16 -> 0.3125.
    for (num, den, printed) in [(1, 4, "0.25"), (3, 8, "0.375"), (5, 16, "0.3125")] {
        let text = StepError::DiffusionUnstable {
            diffusion_number: q(num, den),
        }
        .to_string();
        assert!(
            text.contains(printed),
            "message {text:?} must contain the measured diffusion number {printed}"
        );
        assert!(text.contains("1/6"), "message {text:?} must name the limit");
    }
}

#[test]
fn every_refusal_message_is_distinct() {
    let texts: Vec<String> = vec![
        StepError::WallModelNeedsViscosity.to_string(),
        StepError::TurbulenceFieldShape.to_string(),
        StepError::NegativeEddyViscosity.to_string(),
        StepError::ZeroReinitCount.to_string(),
        StepError::DiffusionUnstable {
            diffusion_number: q(1, 4),
        }
        .to_string(),
        PressureSolverError::ZeroTimeStep.to_string(),
        PressureSolverError::ZeroDensity.to_string(),
        PressureSolverError::ZeroSpacing.to_string(),
        PressureSolverError::ZeroIterations.to_string(),
        PressureSolverError::NonPositiveTolerance.to_string(),
        PressureSolverError::ZeroRanks.to_string(),
        WallFunctionError::NonPositiveWallDistance.to_string(),
        WallFunctionError::NonPositiveViscosity.to_string(),
        WallFunctionError::NegativeSpeed.to_string(),
    ];
    for (a, ta) in texts.iter().enumerate() {
        assert!(!ta.is_empty());
        for tb in texts.iter().skip(a + 1) {
            assert_ne!(ta, tb, "two refusals share one message");
        }
    }
}

// ------------------------------------------------------------- step_count

fn resting(n: usize) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    s.gravity = Vec3Fix::ZERO;
    s
}

#[test]
fn step_count_advances_by_exactly_one_per_step_on_every_entry_point() {
    let n = 4;
    let dt = q(1, 100);
    let mut s = resting(n);
    let mut expected = 0u64;
    for round in 0..3 {
        s.step(dt);
        expected += 1;
        assert_eq!(s.step_count, expected, "step, round {round}");
        s.step_multigrid(dt, 2);
        expected += 1;
        assert_eq!(s.step_count, expected, "step_multigrid, round {round}");
        s.step_with_options(dt, &opts()).expect("steps");
        expected += 1;
        assert_eq!(s.step_count, expected, "step_with_options, round {round}");
        let mut state = RansState::new(n, n, n, TurbulenceModel::Smagorinsky);
        s.step_rans(dt, &opts(), &mut state).expect("steps");
        expected += 1;
        assert_eq!(s.step_count, expected, "step_rans, round {round}");
    }
    assert_eq!(s.step_count, 12);
}

#[test]
fn a_refused_step_leaves_step_count_where_it_was() {
    let n = 4;
    let dt = q(1, 100);
    let mut s = resting(n);
    s.step(dt);
    s.step(dt);
    assert_eq!(s.step_count, 2);
    assert_eq!(
        s.step_with_options(Fix128::ZERO, &opts()),
        Err(StepError::Pressure(PressureSolverError::ZeroTimeStep))
    );
    assert_eq!(s.step_count, 2);
    let mut wrong = RansState::new(n + 1, n, n, TurbulenceModel::Smagorinsky);
    assert!(matches!(
        s.step_rans(dt, &opts(), &mut wrong),
        Err(StepError::TurbulenceFieldShape)
    ));
    assert_eq!(s.step_count, 2);
    s.step(dt);
    assert_eq!(s.step_count, 3);
}

// ------------------------------------------------------------- free fall

#[test]
fn resting_fluid_without_boundaries_falls_at_g_times_elapsed_time() {
    // No face boundary condition is set (the default), so a uniform velocity
    // has zero divergence in every cell: advection of a uniform field is the
    // identity, the viscous term of a uniform field is zero and the
    // projection has nothing to remove. Gravity alone then gives
    // v = g_y * n * dt on every v face, u = w = 0.
    let n = 4;
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    let g_y = -9.81f64;
    assert_eq!(s.gravity.y, q(-981, 100));
    let dt = 1.0 / 128.0; // dyadic, so n * dt is exact
    for steps in 1..=4u32 {
        s.step(Fix128::from_ratio(1, 128));
        let want = g_y * f64::from(steps) * dt;
        for (ix, v) in s.grid.v.iter().enumerate() {
            let got = v.to_f64();
            assert!(
                (got - want).abs() < 1e-12,
                "after {steps} steps v[{ix}] = {got}, free fall gives {want}"
            );
        }
        assert!(s.grid.u.iter().all(|u| u.is_zero()), "u must stay zero");
        assert!(s.grid.w.iter().all(|w| w.is_zero()), "w must stay zero");
    }
}

#[test]
fn free_fall_follows_the_gravity_vector_along_every_axis() {
    // The same argument with a gravity vector that has all three components:
    // each face family integrates only its own component.
    let n = 4;
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    s.gravity = Vec3Fix::new(q(3, 2), q(-5, 4), q(7, 8));
    let dt = q(1, 64);
    s.step(dt);
    s.step(dt);
    let t = 2.0 / 64.0;
    for (name, field, g) in [
        ("u", &s.grid.u, 1.5f64),
        ("v", &s.grid.v, -1.25),
        ("w", &s.grid.w, 0.875),
    ] {
        for (ix, c) in field.iter().enumerate() {
            let got = c.to_f64();
            assert!(
                (got - g * t).abs() < 1e-12,
                "{name}[{ix}] = {got}, free fall gives {}",
                g * t
            );
        }
    }
}
