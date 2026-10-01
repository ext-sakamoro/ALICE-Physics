//! The per-increment entry point reproduces `solve_elastoplastic` to the bit.
//!
//! # What this file is, and is not
//!
//! The folds below are a **change detector**, not a closed form: they were read
//! off the implementation before the load loop was extracted into
//! [`alice_physics::linear_elastic_fem::ElastoplasticProblem`], and their only
//! claim is that moving the loop changed nothing. The closed-form oracles for
//! the physics live in `tests/analytic_elastoplastic_fem.rs` (19) and
//! `tests/analytic_plastic_dissipation.rs` (25); this file sits beside them to
//! catch a refactor that keeps every one of those green while quietly moving a
//! number none of them reads.
//!
//! # Why four folds and not one
//!
//! `ε̄_p` and `W_p` do not carry the same information. The equivalent plastic
//! strain is a state variable: for a fixed total strain it is the same whether
//! the load arrives in one increment or five hundred. The plastic work is a
//! **path integral**, and its discrete form
//!
//! ```text
//! W_p = σ_y·ε̄_p + (H/2)(ε̄_p² + Σ_k Δε̄_k²)
//! ```
//!
//! carries `Σ_k Δε̄_k²`, which shrinks as the increments do. So a refactor that
//! changes **when** the plastic state is committed can leave displacements,
//! stresses and `ε̄_p` bit-identical and still move `W_p` — which is exactly
//! what the increment API does, since it hands the commit to the caller. One
//! fold over everything would also catch that, but would not say which of the
//! four moved; four folds name the field.
//!
//! `the_equivalent_strain_is_step_count_free_while_the_work_is_not` is the
//! vacuity guard: it asserts the asymmetry the paragraph above describes, so a
//! future change that made `W_p` step-count independent (by integrating the
//! continuous form instead of the path) would fail here rather than silently
//! make the four folds redundant.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::linear_elastic_fem::{
    solve_elastoplastic, Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig,
    ElastoplasticIncrementRequest, ElastoplasticProblem, ElastoplasticSolution, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// Folds
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit basis.
const FOLD_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn fold_word(acc: u64, word: u64) -> u64 {
    let mut a = acc;
    for byte in word.to_le_bytes() {
        a ^= u64::from(byte);
        a = a.wrapping_mul(0x0100_0000_01b3);
    }
    a
}

/// Fold the raw `Q64.64` words, not `to_f64()`: an `f64` carries 53 of the 128
/// bits, so a fold built on it would agree across a difference in the low
/// fractional bits — which is the size of difference a commit moved by one
/// iteration produces.
fn fold_fix(acc: u64, value: Fix128) -> u64 {
    fold_word(fold_word(acc, value.hi as u64), value.lo)
}

/// `[displacements, element stress, ε̄_p, W_p]`, each folded on its own.
fn folds(s: &ElastoplasticSolution) -> [u64; 4] {
    let mut displacements = FOLD_BASIS;
    for node in &s.field.displacements {
        for component in node {
            displacements = fold_fix(displacements, *component);
        }
    }
    let mut stress = FOLD_BASIS;
    for t in &s.field.element_stress {
        for component in [t.xx, t.yy, t.zz, t.xy, t.yz, t.zx] {
            stress = fold_fix(stress, component);
        }
    }
    let mut equivalent = FOLD_BASIS;
    for v in &s.equivalent_plastic_strain {
        equivalent = fold_fix(equivalent, *v);
    }
    let mut work = FOLD_BASIS;
    for v in &s.dissipation {
        work = fold_fix(work, *v);
    }
    [displacements, stress, equivalent, work]
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node(i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * 3 + k * 6).expect("the 3x2x2 lattice fits u32")
}

/// Kuhn 6-tet subdivision of `[0,4] × [0,2] × [0,2]`, 12 tetrahedra.
fn bar_mesh() -> SdfTetMesh {
    let (nx, ny, nz, h) = (2usize, 1usize, 1usize, 2.0f32);
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node(i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// `u_x = 0` on `x = 0`, `u_x = 0.005 · L` on `x = L`, lateral rigid modes
/// removed. `ε = 0.005` against a yield strain of `σ_y / E = 2 / 1024`, so
/// every element yields.
fn bar_bc() -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.prescribe(node(0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node(2, j, k), Axis::X, fx(0.005) * Fix128::from_int(4));
        }
    }
    bc.prescribe(node(0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node(0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node(0, 1, 0), Axis::Z, Fix128::ZERO);
    bc
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(1024.0), fx(0.25)).expect("E > 0 and nu in (-1, 0.5)")
}

fn config() -> ElastoplasticConfig {
    ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24),
        fx(2.0),
        fx(1024.0),
    )
    .expect("a valid elastoplastic config")
}

/// `n` equal increments reaching load factor one.
fn uniform_path(n: usize) -> Vec<Fix128> {
    (1..=n)
        .map(|i| Fix128::from_ratio(i as i64, n as i64))
        .collect()
}

fn solve(n: usize) -> ElastoplasticSolution {
    solve_elastoplastic(
        &bar_mesh(),
        &material(),
        &bar_bc(),
        &config(),
        &uniform_path(n),
    )
    .expect("the bar solves")
}

// ---------------------------------------------------------------------------
// Pinned folds (read off the implementation before the extraction)
// ---------------------------------------------------------------------------

/// `[displacements, stress, ε̄_p, W_p]` for a single increment.
const GOLDEN_ONE_STEP: [u64; 4] = [
    0xbfe0_513d_00e3_fd27,
    0xc71f_1b6c_472b_7d4c,
    0xb530_8fe0_760a_9b6b,
    0x9340_5269_5b20_24e7,
];

/// The same for eight equal increments.
const GOLDEN_EIGHT_STEPS: [u64; 4] = [
    0x88f1_d1d0_eb54_1f08,
    0xc537_ab23_cad8_f4e3,
    0xd390_fb9a_ca57_5d52,
    0xf9bc_2797_6a60_63fc,
];

/// Every output of `solve_elastoplastic` is unchanged by the extraction.
///
/// Four folds, so a failure names the field that moved. The Newton count and
/// the step count are pinned too: the extraction must not change how many
/// linear solves the path costs.
#[test]
fn the_single_increment_path_reproduces_the_recorded_solution() {
    let one = solve(1);
    assert_eq!(
        folds(&one),
        GOLDEN_ONE_STEP,
        "one increment: [displacements, stress, eqps, work] moved"
    );
    assert_eq!(one.newton_iterations, 4);
    assert_eq!(one.steps, 1);
}

/// As above for eight increments, which is the case that reads `Σ Δε̄_k²`.
#[test]
fn the_eight_increment_path_reproduces_the_recorded_solution() {
    let eight = solve(8);
    assert_eq!(
        folds(&eight),
        GOLDEN_EIGHT_STEPS,
        "eight increments: [displacements, stress, eqps, work] moved"
    );
    assert_eq!(eight.newton_iterations, 13);
    assert_eq!(eight.steps, 8);
}

/// How closely `ε̄_p` is allowed to differ between the two paths.
///
/// Path independence of the equivalent plastic strain is exact in exact
/// arithmetic — the radial return accumulates the same total for a monotonic
/// path whatever the increments are — so the only difference is what the
/// solver's own stopping rule leaves behind. The Newton tolerance here is
/// `2⁻⁴⁰` relative with an absolute floor, and the strain inherits that; `1e-9`
/// is two orders above the measured `9.6e-11` and still nine orders below the
/// path dependence it has to be distinguished from.
const EQPS_AGREEMENT: f64 = 1.0e-9;

/// How much further apart the plastic work has to be.
///
/// A millionfold: the measurement is a ratio of `1.8e9`, so this threshold has
/// three orders of margin while still refusing anything that could be read as
/// rounding.
const WORK_SEPARATION: f64 = 1.0e6;

/// The asymmetry the four folds exist for: `ε̄_p` does not depend on how the
/// load was split, `W_p` does.
///
/// Without this, the two tests above could both pass on an implementation
/// where `W_p` had stopped being a path integral — the folds would still
/// differ between the two step counts, because `ε̄_p` differs in its last bits,
/// but the field that is supposed to carry the path would no longer carry it.
///
/// The two sides are asserted against one threshold each rather than against
/// each other element by element, because the quantities are separated by nine
/// orders of magnitude: a test that only said "they differ" would pass on a
/// difference of one ulp.
#[test]
fn the_equivalent_strain_is_step_count_free_while_the_work_is_not() {
    let one = solve(1);
    let eight = solve(8);
    assert!(
        !one.dissipation.is_empty(),
        "no elements, so nothing below is measured"
    );

    let mut worst_strain = 0.0_f64;
    let mut closest_work = f64::MAX;
    let mut yielded = 0usize;
    for e in 0..one.equivalent_plastic_strain.len() {
        let (s1, s8) = (
            one.equivalent_plastic_strain[e].to_f64(),
            eight.equivalent_plastic_strain[e].to_f64(),
        );
        let (w1, w8) = (one.dissipation[e].to_f64(), eight.dissipation[e].to_f64());
        if s1 == 0.0 || w1 == 0.0 {
            continue;
        }
        yielded += 1;
        worst_strain = worst_strain.max(((s1 - s8) / s1).abs());
        closest_work = closest_work.min(((w1 - w8) / w1).abs());
        assert!(
            w1 > w8,
            "element {e}: the plastic work did not fall when the increments \
             shrank ({w1:.12e} against {w8:.12e}). The discrete excess over the \
             continuous form is (H/2)·Σ Δε̄_k², which is first order in the \
             increment, so one coarse increment must overshoot eight fine ones."
        );
    }

    assert!(
        yielded > 0,
        "no element yielded, so neither threshold below is measured"
    );
    assert!(
        worst_strain <= EQPS_AGREEMENT,
        "the equivalent plastic strain moved by {worst_strain:.3e} between a \
         one-increment and an eight-increment path, over the {EQPS_AGREEMENT:.0e} \
         the solver's stopping rule accounts for. It is a state variable: if it \
         now depends on the path, the return mapping is accumulating something \
         it should not."
    );
    assert!(
        closest_work >= WORK_SEPARATION * EQPS_AGREEMENT,
        "the plastic work moved by only {closest_work:.3e} between the two \
         paths, under the {:.0e} that separates it from the strain's rounding. \
         W_p is a path integral and must carry Σ Δε̄_k²; if it no longer does, \
         the four folds above stop distinguishing a commit moved by one \
         iteration.",
        WORK_SEPARATION * EQPS_AGREEMENT
    );
}

// ---------------------------------------------------------------------------
// The increment API, driven from outside the crate
// ---------------------------------------------------------------------------
//
// An integration test is a separate crate, so these also check that the new
// public items can actually be built and called from downstream — the failure
// mode `#[non_exhaustive]` without a constructor produces, which crate-internal
// tests cannot see.

/// Stepping the path by hand reproduces `solve_elastoplastic` exactly.
///
/// This is the claim the extraction rests on: the single call is implemented on
/// top of the increment API, so if the two ever disagree, one of them has grown
/// a second Newton loop.
#[test]
fn stepping_by_hand_reproduces_the_single_call() {
    for n in [1usize, 3, 8] {
        let mesh = bar_mesh();
        let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(), &config())
            .expect("the bar prepares");
        let mut state = problem.virgin_state();
        let mut increments = 0u32;
        for factor in uniform_path(n) {
            let increment = problem
                .step(&state, &ElastoplasticIncrementRequest::new(factor))
                .expect("the increment solves");
            increments += 1;
            increment.commit(&mut state);
        }
        assert_eq!(increments as usize, n);

        let whole = solve(n);
        assert_eq!(
            state.equivalent_plastic_strain(),
            whole.equivalent_plastic_strain,
            "{n} increments: ε̄_p differs between the stepped and the single call"
        );
        assert_eq!(
            state.dissipation(),
            whole.dissipation,
            "{n} increments: W_p differs between the stepped and the single call. \
             This is the field a moved commit changes without touching the others."
        );
        assert_eq!(
            state.displacements(),
            whole.field.displacements,
            "{n} increments: displacements differ"
        );
        assert_eq!(
            state.newton_iterations(),
            whole.newton_iterations,
            "{n} increments: the Newton count differs, so the stepped path is not \
             taking the same iterations"
        );
    }
}

/// `plastic_work_increment` sums to the committed total, and no single
/// increment carries all of it.
///
/// The first half is the contract a thermal solve relies on: it deposits the
/// increment, not the total, so the increments have to add up. The second half
/// is the vacuity guard — on a one-increment path the sum would be trivially
/// the total, and the test would say nothing about `ΔW_p` being per-increment.
#[test]
fn the_work_increments_sum_to_the_committed_total() {
    let mesh = bar_mesh();
    let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(), &config())
        .expect("the bar prepares");
    let mut state = problem.virgin_state();
    let mut summed = vec![Fix128::ZERO; mesh.tets.len()];
    let mut per_increment: Vec<Vec<Fix128>> = Vec::new();

    for factor in uniform_path(4) {
        let increment = problem
            .step(&state, &ElastoplasticIncrementRequest::new(factor))
            .expect("the increment solves");
        for (total, add) in summed
            .iter_mut()
            .zip(increment.plastic_work_increment.iter())
        {
            assert!(
                !add.is_negative(),
                "an increment reported negative plastic work ({}), which the \
                 monotonic equivalent strain forbids",
                add.to_f64()
            );
            *total = *total + *add;
        }
        per_increment.push(increment.plastic_work_increment.clone());
        increment.commit(&mut state);
    }

    assert_eq!(
        summed,
        state.dissipation(),
        "the per-increment plastic work did not add up to the committed total, so \
         a thermal solve depositing the increments would not deposit the whole \
         dissipation"
    );

    let yielding = (0..mesh.tets.len())
        .filter(|&e| !state.dissipation()[e].is_zero())
        .count();
    assert!(
        yielding > 0,
        "no element yielded, so nothing above is measured"
    );
    for e in 0..mesh.tets.len() {
        if state.dissipation()[e].is_zero() {
            continue;
        }
        let carried = per_increment.iter().filter(|inc| !inc[e].is_zero()).count();
        assert!(
            carried > 1,
            "element {e} took all of its plastic work in one of the four \
             increments, so this scene cannot tell a per-increment quantity from \
             a total"
        );
    }
}

/// Re-running the same increment from the same committed state gives the same
/// answer, and committing is what moves the state.
///
/// This is the property the coupled iteration needs: a sweep that solves the
/// increment, looks at it and throws it away must leave nothing behind.
#[test]
fn an_uncommitted_increment_leaves_the_state_untouched() {
    let mesh = bar_mesh();
    let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(), &config())
        .expect("the bar prepares");
    let mut state = problem.virgin_state();
    let request = ElastoplasticIncrementRequest::new(Fix128::ONE);
    assert_eq!(request.factor(), Fix128::ONE);

    let first = problem.step(&state, &request).expect("solves");
    let second = problem.step(&state, &request).expect("solves again");
    assert_eq!(
        first, second,
        "two solves of the same increment from the same state disagreed, so the \
         step is carrying something across calls"
    );

    let before = state.clone();
    let third = problem.step(&state, &request).expect("solves a third time");
    assert_eq!(state, before, "stepping changed the state without a commit");
    assert!(
        third.plastic_work_increment.iter().any(|w| !w.is_zero()),
        "the increment did no plastic work, so the commit below moves nothing"
    );
    third.commit(&mut state);
    assert_ne!(
        state, before,
        "committing did not change the state, so the uncommitted-equality above \
         is vacuous"
    );
}

/// An increment that is already converged costs no Newton iteration and keeps
/// the previous solve's report.
///
/// Repeating a load factor is the only way to reach the branch where the Newton
/// loop breaks before it has run once, and it is the branch that decides what
/// `iterations`, `relative_residual` and `effective_relative_tolerance` mean at
/// the end of a path: they describe the **last linear solve**, not the last
/// increment, so a no-op increment must carry them through rather than report
/// zero. `solve_elastoplastic` had this behaviour before the load loop was
/// extracted — those three were function-scope variables that no increment
/// reset — and nothing else in the suite reaches the branch, because every
/// other scene's last increment runs at least one solve.
#[test]
fn a_repeated_load_factor_is_a_free_increment_that_keeps_the_last_report() {
    let mesh = bar_mesh();
    let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(), &config())
        .expect("the bar prepares");
    let mut state = problem.virgin_state();

    let first = problem
        .step(&state, &ElastoplasticIncrementRequest::new(Fix128::ONE))
        .expect("the first increment solves");
    assert!(
        first.newton_iterations() > 0,
        "the first increment converged without a solve, so the comparison below \
         has nothing to carry through"
    );
    assert!(
        first.field.iterations > 0,
        "the first increment ran no conjugate gradient iteration"
    );
    let first_work = first.plastic_work_increment.clone();
    let report = (
        first.field.iterations,
        first.field.relative_residual,
        first.field.effective_relative_tolerance,
    );
    first.commit(&mut state);

    let again = problem
        .step(&state, &ElastoplasticIncrementRequest::new(Fix128::ONE))
        .expect("the repeated increment solves");
    assert_eq!(
        again.newton_iterations(),
        0,
        "the same load factor applied twice took a Newton iteration, so the \
         state it started from was not the converged one"
    );
    // Not exactly zero: the return map lands on the yield surface to within
    // the truncation of `Fix128`, so re-mapping a committed state recovers a
    // few ulps of further plastic strain. `Δε̄` at the floor times a yield
    // radius of order `σ_y` bounds the work by a handful of ulps; 64 is that
    // handful with room, and the measurement is 14.
    let floor = Fix128::from_raw(0, 64);
    let worst = again
        .plastic_work_increment
        .iter()
        .copied()
        .fold(Fix128::ZERO, |a, w| if w.abs() > a { w.abs() } else { a });
    assert!(
        worst <= floor,
        "a free increment dissipated {} of plastic work, over the {} the \
         truncation floor accounts for. The increment is already converged, so \
         anything above the floor means the return map is advancing the state.",
        worst.to_f64(),
        floor.to_f64()
    );
    let smallest_first = first_work
        .iter()
        .copied()
        .fold(Fix128::from_int(1 << 20), |a, w| if w < a { w } else { a });
    assert!(
        smallest_first > floor * Fix128::from_int(1 << 30),
        "the first increment's smallest plastic work ({}) is not far enough above \
         the floor ({}) for the bound above to mean anything",
        smallest_first.to_f64(),
        floor.to_f64()
    );
    assert_eq!(
        (
            again.field.iterations,
            again.field.relative_residual,
            again.field.effective_relative_tolerance
        ),
        report,
        "the free increment reset the solver report instead of carrying the \
         previous solve's. These three describe the last linear solve over the \
         whole path, which is what solve_elastoplastic reported before the load \
         loop was extracted."
    );

    again.commit(&mut state);
    assert_eq!(
        state.newton_iterations(),
        4,
        "the free increment changed the Newton total"
    );
}
