//! Closed-form oracles and degenerate-input tests for the thirteen
//! `linear_elastic_fem` items that were implemented but unwired:
//! `DiagonalStats`, `alpha_per_k`, `default_poissons_ratio`, `from_filament`,
//! `hydrostatic`, `load_count`, `poissons_ratio`, `prescribed_count`,
//! `stiffness_diagonal_stats`, `with_poisson`, `with_preconditioner`,
//! `with_stagnation_fraction`, `youngs_modulus_mpa`.
//!
//! `tests/analytic_linear_elastic_fem.rs` already carries closed-form or
//! round-trip coverage for `with_poisson`, `with_stagnation_fraction` /
//! `stagnation_window_fraction`, `prescribed_count`, `from_filament` (on the
//! `Fdm` category only), `default_poissons_ratio` (ordering only),
//! `poissons_ratio` / `youngs_modulus_mpa` (via `ElasticMaterial::new`
//! error paths), and `stiffness_diagonal_stats` / `DiagonalStats`
//! (positivity / ordering only). This file does not repeat those; it adds
//! what was missing: `alpha_per_k` and `load_count` had zero coverage
//! anywhere in the crate; `with_preconditioner` is exercised behaviourally
//! (iteration count, not the answer) but nothing ever asserts
//! `preconditioner()` returns what `with_preconditioner` was given; and
//! three items (`default_poissons_ratio`, `from_filament`,
//! `stiffness_diagonal_stats`) only had an ordering or positivity
//! invariant, not a value pinned against an independent closed form, and
//! not a non-default scene — see
//! `default_poissons_ratio_ignored_by_from_filament_would_be_invisible_on_fdm`
//! for why that last gap matters
//! (`default-valued-argument-makes-its-wiring-mutation-identity`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::panic;

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::linear_elastic_fem::{
    stiffness_diagonal_stats, Axis, BoundaryConditions, ElasticMaterial, FemError, Preconditioner,
    SolverConfig, StressTensor, ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol,
        "{what}: got {g:.15e}, closed form {want:.15e}, difference {:.3e} > tol {tol:.3e}",
        (g - want).abs()
    );
}

/// A single tetrahedron with vertices at the origin and the three axis unit
/// points: `p0=(0,0,0)`, `p1=(1,0,0)`, `p2=(0,1,0)`, `p3=(0,0,1)`.
fn unit_right_tet() -> SdfTetMesh {
    SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        tets: vec![Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    }
}

// ---------------------------------------------------------------------------
// `ThermalExpansion::alpha_per_k` — zero existing coverage anywhere
// ---------------------------------------------------------------------------

/// Pairing with [`TemperatureRise::from_absolute`] is pure storage: no
/// arithmetic happens between the coefficient going in and
/// [`ThermalExpansion::alpha_per_k`] reading it back, so the round trip is
/// exact at every value the doc says is legal, including negative
/// coefficients (real materials, per the doc) and both ends of the `Fix128`
/// range (the doc performs no validation on `alpha_per_k` at all).
#[test]
fn alpha_per_k_round_trips_exactly_at_every_value_the_doc_permits() {
    let grid = CoupledField::try_new_filled(
        1,
        1,
        1,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (Fix128::ONE, Fix128::ONE, Fix128::ONE),
        Fix128::from_int(10),
    )
    .expect("1^3 grid is valid");
    let rise = TemperatureRise::from_absolute(&grid, Fix128::from_int(5));

    for alpha in [
        Fix128::ZERO,
        Fix128::from_ratio(1, 1000),
        -Fix128::from_ratio(1, 1000), // negative expansion coefficients are real materials
        Fix128::from_raw(i64::MAX, u64::MAX), // extreme positive
        Fix128::from_raw(i64::MIN, 0), // extreme negative
    ] {
        let thermal = ThermalExpansion::from_rise(&rise, alpha);
        assert_eq!(
            thermal.alpha_per_k(),
            alpha,
            "alpha_per_k must echo exactly what from_rise was given, no rounding and no clamp"
        );
    }
}

// ---------------------------------------------------------------------------
// `BoundaryConditions::load_count` — zero existing coverage anywhere
// ---------------------------------------------------------------------------

/// `load_count` counts distinct `(vertex, axis)` degrees of freedom, not the
/// number of `add_load` calls: repeated calls at the same DOF accumulate
/// into one entry (per `BoundaryConditions::add_load`'s own doc), so the
/// count must stay the same while the stored force changes. This is the
/// "two directions in one test" form the oracle rule asks for: three
/// distinct DOFs must read back as three, and adding to one of them again
/// must not become four.
#[test]
fn load_count_counts_distinct_degrees_of_freedom_not_calls() {
    let mut bc = BoundaryConditions::new();
    assert_eq!(bc.load_count(), 0, "degenerate: no loads at all");

    bc.add_load(0, Axis::X, fx(1.0));
    bc.add_load(1, Axis::Y, fx(2.0));
    bc.add_load(2, Axis::Z, fx(3.0));
    assert_eq!(bc.load_count(), 3, "three distinct (vertex, axis) pairs");

    // Same (vertex, axis) as the first call: accumulates into the same slot.
    bc.add_load(0, Axis::X, fx(4.0));
    assert_eq!(
        bc.load_count(),
        3,
        "a repeated (vertex, axis) must accumulate, not add a fourth entry"
    );
    assert_eq!(
        bc.loads()[0].2,
        fx(5.0),
        "and the accumulated force must be the sum (1.0 + 4.0)"
    );

    // A different axis on an already-loaded vertex is a new entry.
    bc.add_load(0, Axis::Y, fx(9.0));
    assert_eq!(
        bc.load_count(),
        4,
        "same vertex, different axis, is a distinct DOF"
    );
}

// ---------------------------------------------------------------------------
// `SolverConfig::with_preconditioner` / `preconditioner` — round trip
// ---------------------------------------------------------------------------

/// `tests/analytic_linear_elastic_fem.rs` exercises `with_preconditioner`
/// behaviourally (iteration count differs, the converged answer does not),
/// but nothing anywhere asserts that `preconditioner()` reads back what
/// `with_preconditioner` was given. No arithmetic happens between the two —
/// it is `Copy` storage — so the round trip is exact.
#[test]
fn with_preconditioner_round_trips_exactly() {
    let base = SolverConfig::default();
    assert_eq!(
        base.preconditioner(),
        Preconditioner::None,
        "the documented default"
    );

    let tuned = base.with_preconditioner(Preconditioner::JacobiScaled);
    assert_eq!(tuned.preconditioner(), Preconditioner::JacobiScaled);

    let reverted = tuned.with_preconditioner(Preconditioner::None);
    assert_eq!(reverted.preconditioner(), Preconditioner::None);
}

// ---------------------------------------------------------------------------
// `ElasticMaterial::default_poissons_ratio` — exact table pin
// ---------------------------------------------------------------------------

/// Closed form: the table in `ElasticMaterial::from_filament`'s doc, read
/// verbatim (0.35 / 0.30 / 0.35 / 0.40). `from_filament` is not called here;
/// this test is independent of it.
#[test]
fn default_poissons_ratio_matches_the_documented_table_exactly() {
    // `from_ratio`, not `fx` (which round-trips through `f64`): 0.35 has no
    // exact binary representation, so comparing `Fix128::from_f64(0.35)`
    // against the table's own `Fix128::from_ratio(35, 100)` would fail on a
    // rounding difference that has nothing to do with the table being
    // right or wrong. The doc states the table as ratios, so the rational
    // constructor is the closed form here.
    assert_eq!(
        ElasticMaterial::default_poissons_ratio(MaterialCategory::Fdm),
        Fix128::from_ratio(35, 100)
    );
    assert_eq!(
        ElasticMaterial::default_poissons_ratio(MaterialCategory::Sla),
        Fix128::from_ratio(35, 100)
    );
    assert_eq!(
        ElasticMaterial::default_poissons_ratio(MaterialCategory::SheetMetal),
        Fix128::from_ratio(30, 100)
    );
    assert_eq!(
        ElasticMaterial::default_poissons_ratio(MaterialCategory::Powder),
        Fix128::from_ratio(40, 100)
    );
}

// ---------------------------------------------------------------------------
// `ElasticMaterial::from_filament` — non-default category + degenerate E
// ---------------------------------------------------------------------------

/// Every existing `from_filament` scene in the crate (including the solver's
/// own `tests/analytic_linear_elastic_fem.rs`) uses a `Fdm` material, which
/// is also the category table's own `Fdm` row. A wiring mutation that made
/// `from_filament` ignore `material.category` and always read the `Fdm` row
/// would therefore be invisible on every existing scene. `SheetMetal`'s
/// default (0.30) is different from `Fdm`'s (0.35), so it is the scene that
/// catches that mutation — see
/// `default-valued-argument-makes-its-wiring-mutation-identity` in the
/// pitfall catalog for the general shape of this gap.
#[test]
fn default_poissons_ratio_ignored_by_from_filament_would_be_invisible_on_fdm() {
    let sheet = MaterialProperties::sus304();
    assert_eq!(sheet.category, MaterialCategory::SheetMetal);
    let material = ElasticMaterial::from_filament(&sheet).expect("SUS304 is valid");
    assert_eq!(
        material.poissons_ratio(),
        ElasticMaterial::default_poissons_ratio(MaterialCategory::SheetMetal),
        "from_filament on a SheetMetal entry must read the SheetMetal row"
    );
    assert_ne!(
        material.poissons_ratio(),
        ElasticMaterial::default_poissons_ratio(MaterialCategory::Fdm),
        "and that row must differ from Fdm's, or this scene cannot tell the two apart"
    );

    // GPa -> MPa conversion, exact (x1000, both operands integers in Fix128).
    assert_eq!(material.youngs_modulus_mpa(), fx(200_000.0));
}

/// `from_filament` must propagate `ElasticMaterial::new`'s validation rather
/// than constructing an invalid material directly — a degenerate database
/// entry (non-positive Young's modulus) is the only way `from_filament`
/// itself can fail, and no existing scene exercises it.
#[test]
fn from_filament_rejects_a_non_positive_youngs_modulus() {
    let mut zero_e = MaterialProperties::sus304();
    zero_e.youngs_modulus_gpa = Fix128::ZERO;
    assert!(
        matches!(
            ElasticMaterial::from_filament(&zero_e),
            Err(FemError::InvalidMaterial(_))
        ),
        "E = 0 from the database must be rejected, not silently accepted"
    );

    let mut negative_e = MaterialProperties::sus304();
    negative_e.youngs_modulus_gpa = fx(-1.0);
    assert!(
        matches!(
            ElasticMaterial::from_filament(&negative_e),
            Err(FemError::InvalidMaterial(_))
        ),
        "E < 0 from the database must be rejected"
    );
}

// ---------------------------------------------------------------------------
// `StressTensor::hydrostatic` — exact closed form + overflow
// ---------------------------------------------------------------------------

/// `hydrostatic = (xx + yy + zz) / 3`. A non-hydrostatic, dyadic scene
/// (`xx = 6`, the rest zero) distinguishes dividing by 3 from dividing by 2
/// (2.0 vs 3.0) and from averaging the wrong subset of components (0.0 if
/// any permutation omitting `xx` were used by mistake).
#[test]
fn hydrostatic_is_the_trace_over_three_exactly() {
    let s = StressTensor {
        xx: fx(6.0),
        yy: Fix128::ZERO,
        zz: Fix128::ZERO,
        xy: fx(100.0),
        yz: fx(100.0),
        zx: fx(100.0),
    };
    assert_eq!(
        s.hydrostatic(),
        fx(2.0),
        "6/3 = 2, and shear must not leak into a normal-stress-only quantity"
    );

    for p in [fx(1.0), fx(-7.5), Fix128::ZERO] {
        let hydro = StressTensor {
            xx: p,
            yy: p,
            zz: p,
            ..StressTensor::default()
        };
        assert_eq!(
            hydro.hydrostatic(),
            p,
            "a fully hydrostatic state's mean normal stress is the state itself"
        );
    }
}

/// `Fix128` addition wraps (no panic, ever, even in debug builds — see
/// `src/math.rs`'s `Add` impl), so "does not panic" is not a value
/// assertion. This test independently reconstructs the 128-bit two's
/// complement wraparound of `xx + yy + zz` from the raw `(hi, lo)` words
/// using `i128` wrapping arithmetic (not `Fix128::add`) and checks the sum
/// `hydrostatic` divides by three is exactly the value that arithmetic
/// defines — not merely "some value that didn't crash".
#[test]
fn hydrostatic_does_not_panic_under_extreme_magnitude_and_the_sum_is_the_defined_wraparound() {
    /// Reconstruct the two's complement 128-bit integer `value * 2^64` from
    /// a `Fix128`'s raw words, the same way `Fix128::Add` does when it
    /// splits the addition into a `u64` low half (with carry) and an `i64`
    /// high half (`wrapping_add` plus carry) — that split is exactly a
    /// 128-bit two's complement addition, so reassembling it here with
    /// `i128` and comparing is an independent check, not a call into
    /// `Fix128`'s own arithmetic.
    fn raw128(f: Fix128) -> i128 {
        (i128::from(f.hi) << 64) | i128::from(f.lo)
    }

    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);
    let s = StressTensor {
        xx: extreme,
        yy: extreme,
        zz: extreme,
        ..StressTensor::default()
    };

    let result = panic::catch_unwind(|| s.hydrostatic());
    assert!(
        result.is_ok(),
        "Fix128 arithmetic is wrapping; hydrostatic must not panic on extreme magnitude"
    );

    let trace = s.xx + s.yy + s.zz;
    let expected_trace_raw = raw128(extreme)
        .wrapping_add(raw128(extreme))
        .wrapping_add(raw128(extreme));
    assert_eq!(
        raw128(trace),
        expected_trace_raw,
        "the trace computed by Fix128::Add must match the independent i128 wraparound"
    );
}

// ---------------------------------------------------------------------------
// `stiffness_diagonal_stats` / `DiagonalStats` — hand-derived single element
// ---------------------------------------------------------------------------

/// Closed form, derived independently of `stiffness_diagonal`: for a P1
/// tetrahedron the diagonal entry of the global stiffness at node `n`, axis
/// `a`, is `|V| · [(λ+2μ) g_a² + μ(g_b² + g_c²)]` with `g` that node's shape
/// function gradient (standard isotropic P1 FEM, e.g. Zienkiewicz & Taylor;
/// this is the same formula `src/linear_elastic_fem.rs` documents on
/// `stiffness_diagonal`, re-derived here from the gradients below rather
/// than taken on trust).
///
/// For [`unit_right_tet`] (`p0=(0,0,0)`, `p1=(1,0,0)`, `p2=(0,1,0)`,
/// `p3=(0,0,1)`): `J = I`, so `g1=(1,0,0)`, `g2=(0,1,0)`, `g3=(0,0,1)`,
/// `g0 = -(g1+g2+g3) = (-1,-1,-1)`, and `|V| = 1/6`. Node 0's three axes each
/// see `(λ+2μ)·1 + μ·(1+1) = λ+4μ` (every component of `g0` is ±1), while
/// nodes 1-3 each see `λ+2μ` on their own axis and `μ` on the other two.
/// That makes `μ/6` the minimum, `(λ+4μ)/6` the maximum, and the mean of all
/// twelve entries `(λ+4μ)/12` (sum = `3(λ+4μ) + 3(λ+4μ) = 6λ+24μ`, scaled by
/// the shared `1/6` then divided by 12 dofs).
///
/// Two material scenes are used because `ν=0` alone makes `λ=0`, under which
/// a mutation that dropped the `λ` term from the diagonal formula entirely
/// would be invisible.
#[test]
fn stiffness_diagonal_stats_matches_the_hand_derived_single_element() {
    let mesh = unit_right_tet();
    let bc = BoundaryConditions::new();

    // Scene A: nu = 0 => lambda = 0, mu = E/2. E = 2 => mu = 1.
    let a = ElasticMaterial::new(fx(2.0), fx(0.0)).expect("valid");
    let stats_a = stiffness_diagonal_stats(&mesh, &a, &bc).expect("well posed");
    assert_eq!(stats_a.free_dofs, 12, "no boundary conditions: all 12 free");
    close(stats_a.min, 1.0 / 6.0, 1e-15, "scene A min = mu/6");
    close(
        stats_a.max,
        4.0 / 6.0,
        1e-15,
        "scene A max = (lambda+4 mu)/6",
    );
    close(
        stats_a.mean,
        4.0 / 12.0,
        1e-15,
        "scene A mean = (lambda+4 mu)/12",
    );

    // Scene B: nu = 1/4 => lambda = mu = 2E/5 for any E; E = 5 => lambda = mu = 2.
    let b = ElasticMaterial::new(fx(5.0), fx(0.25)).expect("valid");
    let stats_b = stiffness_diagonal_stats(&mesh, &b, &bc).expect("well posed");
    assert_eq!(stats_b.free_dofs, 12);
    close(stats_b.min, 2.0 / 6.0, 1e-15, "scene B min = mu/6 = 2/6");
    close(
        stats_b.max,
        10.0 / 6.0,
        1e-15,
        "scene B max = (lambda+4 mu)/6 = 10/6",
    );
    close(
        stats_b.mean,
        10.0 / 12.0,
        1e-15,
        "scene B mean = (lambda+4 mu)/12 = 10/12",
    );
}

/// Degenerate inputs: an empty mesh has no stiffness to characterise, and a
/// mesh with every degree of freedom prescribed has no free diagonal entries
/// to report — both are reported as errors, not as an empty or zeroed
/// `DiagonalStats`.
#[test]
fn stiffness_diagonal_stats_rejects_empty_and_fully_constrained_inputs() {
    let empty = SdfTetMesh::default();
    let bc = BoundaryConditions::new();
    let material = ElasticMaterial::new(fx(1.0), fx(0.3)).expect("valid");
    assert_eq!(
        stiffness_diagonal_stats(&empty, &material, &bc),
        Err(FemError::EmptyMesh),
        "degenerate: a mesh with no vertices and no tets has no diagonal"
    );

    let mesh = unit_right_tet();
    let mut fully_fixed = BoundaryConditions::new();
    for v in 0..4 {
        fully_fixed.fix(v);
    }
    assert_eq!(
        stiffness_diagonal_stats(&mesh, &material, &fully_fixed),
        Err(FemError::UnderConstrained),
        "degenerate: every DOF prescribed leaves zero free diagonal entries"
    );

    let mut out_of_range = BoundaryConditions::new();
    out_of_range.fix(9_999);
    assert!(
        matches!(
            stiffness_diagonal_stats(&mesh, &material, &out_of_range),
            Err(FemError::VertexOutOfRange { vertex: 9_999, .. })
        ),
        "degenerate: a boundary condition on a vertex the mesh does not have"
    );
}
