//! Oracles for error-driven adaptive refinement on P1 tetrahedra.
//!
//! The mesh side of wall 2 already keeps refinement **conforming**
//! (`tests/refinement_conformity.rs`, `tests/hanging_node_effect.rs`): a split
//! propagates to the neighbour so no hanging node is left. That is the
//! prerequisite. These oracles are about the other half — deciding **where** to
//! refine from the solution itself, which is what makes refinement *adaptive*
//! rather than merely local.
//!
//! # The three pieces and what pins each
//!
//! | piece | pinned by |
//! |---|---|
//! | the indicator (`error_indicators_squared`) | it must be **exactly zero** where P1 is exact, and non-zero where it is not |
//! | the marking (`mark_bulk`) | Dörfler's bulk criterion, its **minimality**, and independence of element order |
//! | the payoff (`solve_adaptive`) | fewer degrees of freedom than uniform refinement for the same error |
//!
//! # The indicator
//!
//! Zienkiewicz–Zhu recovery. P1 strain is constant per element, so `σ_h` is a
//! piecewise constant. Averaging it to the nodes with volume weights and
//! interpolating back gives a continuous `σ*`, and the element indicator is the
//! energy norm of the difference:
//!
//! ```text
//! η_e² = V_e · (Δσ : C⁻¹ : Δσ),   Δσ = σ*(centroid) − σ_h
//!      = V_e / (2μ) · [ Δσ:Δσ − λ/(3λ+2μ) · (tr Δσ)² ]
//! ```
//!
//! ⚠️ **The positive control is tolerance free.** When the exact solution is in
//! the P1 space — a uniform strain field — every element carries the *same*
//! `σ_h`, so the nodal average is that same tensor, `σ* = σ_h` identically, and
//! `η_e² = 0` to the bit for every element. No tolerance is involved, and no
//! choice of quadrature or material constant can disturb it. A recovery that
//! weights the average wrongly, or an energy norm with a sign error, still gives
//! zero here — so the oracle is paired with a scene where the indicator must
//! **not** vanish, or it would be vacuous.
//!
//! # ⚠️ Why the driver takes a closure and not a `BoundaryConditions`
//!
//! Refinement **appends** vertices (`midpoint_of` pushes), so indices already in
//! a `BoundaryConditions` stay valid. That is not enough. A midpoint created on
//! a clamped face is a **new** node, and nothing prescribes it, so the face
//! silently becomes partially free — exactly the hanging-node failure
//! `hanging_node_effect.rs` measured (a linear field tears by 1.489e-1), except
//! caused by the boundary condition rather than by the mesh.
//!
//! ⚠️ Propagating the conditions instead is only half possible. A prescribed
//! displacement can be inherited (both parents prescribed on that axis, value
//! the average, and `1/2` is exact in `Fix128`), but a **nodal load cannot**: a
//! point force carries no information about the traction it stood for, so
//! splitting the face it acts on has no correct answer. The driver therefore
//! asks the caller to rebuild its conditions for each mesh.

#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    error_indicators_squared, mark_bulk, solve, solve_adaptive, AdaptiveConfig, Axis,
    BoundaryConditions, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet subdivision of a box, conforming across cell faces.
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
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
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// `n / 2^k`, exact in `Fix128`.
fn dy(n: i64, k: u32) -> Fix128 {
    if k == 0 {
        return Fix128::from_int(n);
    }
    Fix128::from_raw(0, 1u64 << (64 - k)) * Fix128::from_int(n)
}

const E_MPA: f64 = 1024.0;
/// Traction on the loaded patch, MPa.
const TRACTION: f64 = 64.0;
const NU: f64 = 0.25;

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

/// Every node of the box prescribed to the affine field `u = (ε x, 0, 0)`, which
/// is in the P1 space exactly.
///
/// ⚠️ Prescribing *every* node is the point: the discrete answer is then the
/// interpolant of the exact solution on any mesh, uniform strain, and the
/// recovery has nothing to recover.
fn uniform_strain_bc(mesh: &SdfTetMesh, strain: Fix128) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, position) in mesh.vertices.iter().enumerate() {
        let node = u32::try_from(v).expect("fits u32");
        bc.prescribe(node, Axis::X, strain * Fix128::from_f32(position[0]));
        bc.prescribe(node, Axis::Y, Fix128::ZERO);
        bc.prescribe(node, Axis::Z, Fix128::ZERO);
    }
    bc
}

/// A block clamped on `x = 0` and pulled by a uniform traction on **one cell** of
/// the far face: the traction jumps at the patch edge, so the stress has a
/// localized concentration there and adaptivity has something to find.
///
/// ⚠️ A **point** load would be the obvious way to localize the error and it is
/// the wrong one. The solution for a Dirac force in three-dimensional elasticity
/// has **infinite** energy, so the error in the energy norm does not converge
/// and the total indicator legitimately *grows* under refinement — measured at
/// `18.786 → 28.603` on this block before the scene was changed. Nothing is
/// wrong with the estimator there; the limit simply does not exist, so neither
/// "the estimate falls" nor "adaptive is closer to the reference" means anything.
/// A traction on a patch keeps the energy finite while keeping the feature local.
///
/// The consistent nodal loads of a uniform traction on a square face split into
/// two triangles put `A/3` on the diagonal pair of corners and `A/6` on the other
/// two, matching `analytic_elastoplastic_fem.rs`.
fn patch_traction_bc(mesh: &SdfTetMesh, nx: usize, ny: usize, traction: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[0].abs() < 0.5 {
            let node = u32::try_from(v).expect("fits u32");
            bc.prescribe(node, Axis::X, Fix128::ZERO);
            bc.prescribe(node, Axis::Y, Fix128::ZERO);
            bc.prescribe(node, Axis::Z, Fix128::ZERO);
        }
    }
    // One unit cell of the `x = nx` face, the corner at `y = z = 0`.
    let area = 1.0_f64;
    for (node, share) in [
        (node_index(nx, ny, nx, 0, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 1), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 0), 1.0 / 6.0),
        (node_index(nx, ny, nx, 0, 1), 1.0 / 6.0),
    ] {
        bc.add_load(node, Axis::X, fx(traction * area * share));
    }
    bc
}

/// Total of the applied loads, for the strain energy below.
fn patch_nodes(nx: usize, ny: usize) -> [(u32, f64); 4] {
    [
        (node_index(nx, ny, nx, 0, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 1), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 0), 1.0 / 6.0),
        (node_index(nx, ny, nx, 0, 1), 1.0 / 6.0),
    ]
}

fn linear() -> SolverConfig {
    SolverConfig::default()
}

// ---------------------------------------------------------------------------
// 1. The indicator
// ---------------------------------------------------------------------------

#[test]
fn an_exact_solution_has_indicators_of_exactly_zero() {
    // ⚠️ Tolerance free: a uniform strain field gives every element the same
    // stress, so the nodal average is that stress and the recovered field equals
    // the finite element one identically.
    let mesh = kuhn_box(2, 2, 2, 1.0);
    let bc = uniform_strain_bc(&mesh, dy(1, 8));
    let sol = solve(&mesh, &material(), &bc, &linear()).expect("solve succeeds");
    let eta2 = error_indicators_squared(&mesh, &material(), &sol).expect("indicators computed");
    assert_eq!(eta2.len(), mesh.tets.len(), "one indicator per element");
    for (e, &v) in eta2.iter().enumerate() {
        assert_eq!(
            v,
            Fix128::ZERO,
            "element {e}: a field the space represents exactly must leave no error to estimate"
        );
    }
}

#[test]
fn a_localized_scene_has_indicators_that_are_not_zero() {
    // ⚠️ The tooth for the test above. Without this, an indicator that returned
    // zero unconditionally would pass.
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let mesh = kuhn_box(nx, ny, nz, 1.0);
    let bc = patch_traction_bc(&mesh, nx, ny, TRACTION);
    let sol = solve(&mesh, &material(), &bc, &linear()).expect("solve succeeds");
    let eta2 = error_indicators_squared(&mesh, &material(), &sol).expect("indicators computed");
    let non_zero = eta2.iter().filter(|v| !v.is_zero()).count();
    assert!(
        non_zero > 0,
        "a traction patch must leave something to estimate, got {non_zero} non-zero of {}",
        eta2.len()
    );
    let total: Fix128 = eta2.iter().fold(Fix128::ZERO, |a, &b| a + b);
    assert!(total > Fix128::ZERO, "the total indicator must be positive");
}

#[test]
fn the_indicator_concentrates_where_the_load_is() {
    // The element carrying the loaded patch must be among the worst, or the
    // indicator is not localizing and adaptivity would refine the wrong places.
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let mesh = kuhn_box(nx, ny, nz, 1.0);
    let bc = patch_traction_bc(&mesh, nx, ny, TRACTION);
    let sol = solve(&mesh, &material(), &bc, &linear()).expect("solve succeeds");
    let eta2 = error_indicators_squared(&mesh, &material(), &sol).expect("indicators computed");

    let patch: Vec<u32> = patch_nodes(nx, ny).iter().map(|&(n, _)| n).collect();
    let worst = eta2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .expect("non-empty")
        .0;
    assert!(
        mesh.tets[worst].vertices.iter().any(|v| patch.contains(v)),
        "the worst element (index {worst}) should touch the loaded patch {patch:?}"
    );
}

#[test]
fn an_indicator_for_a_mismatched_solution_is_refused() {
    let mesh = kuhn_box(2, 2, 2, 1.0);
    let bc = uniform_strain_bc(&mesh, dy(1, 8));
    let sol = solve(&mesh, &material(), &bc, &linear()).expect("solve succeeds");
    let other = kuhn_box(1, 1, 1, 1.0);
    assert!(
        error_indicators_squared(&other, &material(), &sol).is_err(),
        "a solution from a different mesh must not be scored silently"
    );
    assert!(
        matches!(
            error_indicators_squared(&SdfTetMesh::default(), &material(), &sol),
            Err(FemError::EmptyMesh)
        ),
        "an empty mesh has nothing to score"
    );
}

// ---------------------------------------------------------------------------
// 2. The marking
// ---------------------------------------------------------------------------

/// Sum of a slice, exactly (`Fix128` addition does not round).
fn total(values: &[Fix128]) -> Fix128 {
    values.iter().fold(Fix128::ZERO, |a, &b| a + b)
}

#[test]
fn bulk_marking_meets_the_threshold() {
    let eta2: Vec<Fix128> = [1i64, 5, 3, 9, 2, 7]
        .iter()
        .map(|&n| Fix128::from_int(n))
        .collect();
    let theta = dy(1, 1); // 1/2
    let marked = mark_bulk(&eta2, theta).expect("valid marking");
    assert_eq!(marked.len(), eta2.len());

    let marked_sum = total(
        &eta2
            .iter()
            .zip(marked.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v)
            .collect::<Vec<_>>(),
    );
    assert!(
        marked_sum >= theta * total(&eta2),
        "Dörfler: the marked set must carry at least θ of the total ({} vs {})",
        marked_sum.to_f64(),
        (theta * total(&eta2)).to_f64()
    );
}

#[test]
fn bulk_marking_is_minimal() {
    // ⚠️ Minimality is the half that makes the marking adaptive rather than
    // "refine everything": dropping the smallest marked element must break the
    // bulk criterion.
    let eta2: Vec<Fix128> = [1i64, 5, 3, 9, 2, 7]
        .iter()
        .map(|&n| Fix128::from_int(n))
        .collect();
    let theta = dy(1, 1);
    let marked = mark_bulk(&eta2, theta).expect("valid marking");
    let mut chosen: Vec<Fix128> = eta2
        .iter()
        .zip(marked.iter())
        .filter(|(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();
    assert!(!chosen.is_empty(), "something must be marked");
    chosen.sort_by(Fix128::cmp);
    let without_smallest = total(&chosen[1..]);
    assert!(
        without_smallest < theta * total(&eta2),
        "the marked set is not minimal: dropping the smallest ({}) still meets θ ({} vs {})",
        chosen[0].to_f64(),
        without_smallest.to_f64(),
        (theta * total(&eta2)).to_f64()
    );
}

#[test]
fn bulk_marking_does_not_depend_on_element_order() {
    // ⚠️ The tie-break has to be deterministic, and the *set of values* chosen
    // must not depend on the order the elements arrive in — otherwise two meshes
    // that differ only by numbering refine differently and the solver stops
    // being reproducible.
    let base: Vec<i64> = vec![4, 4, 9, 1, 4, 9, 2];
    let theta = dy(1, 1);
    let mut reference: Option<Vec<f64>> = None;
    for rotation in 0..base.len() {
        let rotated: Vec<Fix128> = (0..base.len())
            .map(|i| Fix128::from_int(base[(i + rotation) % base.len()]))
            .collect();
        let marked = mark_bulk(&rotated, theta).expect("valid marking");
        let mut chosen: Vec<f64> = rotated
            .iter()
            .zip(marked.iter())
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v.to_f64())
            .collect();
        chosen.sort_by(f64::total_cmp);
        match &reference {
            None => reference = Some(chosen),
            Some(r) => assert_eq!(
                r, &chosen,
                "rotation {rotation} marked a different multiset of indicators"
            ),
        }
    }
}

#[test]
fn a_zero_total_marks_nothing_and_is_not_an_error() {
    // An exact solution scores zero everywhere. That is "no work to do", not a
    // failure, and it must not mark the whole mesh by dividing by zero.
    let eta2 = vec![Fix128::ZERO; 5];
    let marked = mark_bulk(&eta2, dy(1, 1)).expect("a zero total is a valid input");
    assert!(
        marked.iter().all(|&m| !m),
        "nothing to refine when there is no estimated error"
    );
}

#[test]
fn degenerate_marking_parameters_are_rejected() {
    let eta2 = vec![Fix128::ONE; 3];
    for bad in [Fix128::ZERO, -dy(1, 2), Fix128::ONE + dy(1, 2)] {
        assert!(
            matches!(mark_bulk(&eta2, bad), Err(FemError::InvalidConfig(_))),
            "bulk fraction {} must be rejected",
            bad.to_f64()
        );
    }
    assert!(
        mark_bulk(&eta2, Fix128::ONE).is_ok(),
        "θ = 1 means refine everything and is a legal request"
    );
    assert!(
        matches!(mark_bulk(&[], dy(1, 1)), Err(FemError::EmptyMesh)),
        "no elements is not a marking problem"
    );
    assert!(
        mark_bulk(&[-Fix128::ONE, Fix128::ONE], dy(1, 1)).is_err(),
        "a negative squared indicator is not a squared indicator"
    );
}

// ---------------------------------------------------------------------------
// 3. Refinement by mark
// ---------------------------------------------------------------------------

fn face_use_counts(mesh: &SdfTetMesh) -> HashMap<[u32; 3], usize> {
    let mut counts: HashMap<[u32; 3], usize> = HashMap::new();
    for tet in &mesh.tets {
        let v = tet.vertices;
        for face in [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[0], v[2], v[3]],
            [v[1], v[2], v[3]],
        ] {
            let mut key = face;
            key.sort_unstable();
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

#[test]
fn marked_refinement_splits_the_marked_elements() {
    let mut mesh = kuhn_box(2, 2, 2, 1.0);
    let before = mesh.tets.len();
    let mut marked = vec![false; before];
    marked[0] = true;
    let passes = mesh
        .try_refine_marked(&marked, 8)
        .expect("refinement finishes");
    assert!(passes >= 1, "a marked element must cause at least one pass");
    assert!(
        mesh.tets.len() > before,
        "the mesh must grow: {before} -> {}",
        mesh.tets.len()
    );
}

#[test]
fn marking_nothing_changes_nothing() {
    // Positive control: with no marks the mesh must come back bit-identical, so a
    // refinement that ignored the marks and used an edge-length rule instead
    // cannot pass.
    let mut mesh = kuhn_box(2, 2, 2, 1.0);
    let before_tets = mesh.tets.clone();
    let before_vertices = mesh.vertices.clone();
    let passes = mesh
        .try_refine_marked(&vec![false; before_tets.len()], 8)
        .expect("an empty mark set is not an error");
    assert_eq!(passes, 0, "no marks means no passes");
    assert_eq!(mesh.tets, before_tets, "no element may change");
    assert_eq!(mesh.vertices, before_vertices, "no vertex may be added");
}

#[test]
fn marked_refinement_stays_conforming() {
    // The propagation that `refinement_conformity.rs` pins for the edge-length
    // entry point has to hold for the marked one too, since adaptivity marks a
    // handful of elements and leaves their neighbours alone.
    let mut mesh = kuhn_box(3, 2, 2, 1.0);
    let before_unshared = face_use_counts(&mesh).values().filter(|&&c| c == 1).count();
    let mut marked = vec![false; mesh.tets.len()];
    for (i, m) in marked.iter_mut().enumerate() {
        *m = i % 7 == 0;
    }
    mesh.try_refine_marked(&marked, 16)
        .expect("refinement finishes");
    let counts = face_use_counts(&mesh);
    let interior_shared = counts.values().filter(|&&c| c == 2).count();
    let unshared = counts.values().filter(|&&c| c == 1).count();
    assert!(
        counts.values().all(|&c| c == 1 || c == 2),
        "a face may belong to at most two tetrahedra"
    );
    assert!(
        interior_shared > 0,
        "the refined mesh must have interior faces at all"
    );
    // The only unshared faces are on the boundary of the box, and refinement
    // splits boundary faces, so the count may grow but every one must sit on a
    // face of the box.
    assert!(
        unshared >= before_unshared,
        "boundary faces cannot disappear: {before_unshared} -> {unshared}"
    );
    for (face, _) in counts.iter().filter(|(_, &c)| c == 1) {
        let on_boundary = (0..3).any(|axis| {
            face.iter().all(|&v| {
                let p = mesh.vertices[v as usize][axis];
                p.abs() < 1e-6 || (p - [3.0, 2.0, 2.0][axis]).abs() < 1e-6
            })
        });
        assert!(
            on_boundary,
            "unshared face {face:?} is not on the box boundary"
        );
    }
}

#[test]
fn degenerate_marked_refinement_is_refused() {
    let mut mesh = kuhn_box(2, 2, 2, 1.0);
    let n = mesh.tets.len();
    assert!(
        mesh.try_refine_marked(&vec![false; n - 1], 8).is_err(),
        "a mark slice shorter than the mesh must be refused, not zipped short"
    );
    assert!(
        mesh.try_refine_marked(&vec![false; n + 1], 8).is_err(),
        "a mark slice longer than the mesh must be refused"
    );
    // A budget of zero with work to do is `Unfinished`, not a silent no-op.
    let mut marked = vec![false; n];
    marked[0] = true;
    assert!(
        mesh.try_refine_marked(&marked, 0).is_err(),
        "a zero budget with work left must report Unfinished"
    );
}

// ---------------------------------------------------------------------------
// 4. The payoff: adaptivity must beat uniform refinement per degree of freedom
// ---------------------------------------------------------------------------

/// Strain energy `½ uᵀf`, summed over the loaded patch.
///
/// A functional of the solution with a definite limit, computed without the
/// error estimator, so using it to judge adaptivity is not circular.
fn strain_energy(
    sol: &alice_physics::linear_elastic_fem::FemSolution,
    nx: usize,
    ny: usize,
) -> Fix128 {
    let mut work = Fix128::ZERO;
    for (node, share) in patch_nodes(nx, ny) {
        work = work + fx(TRACTION * share) * sol.displacements[node as usize][0];
    }
    dy(1, 1) * work
}

#[test]
fn adaptive_refinement_beats_uniform_per_degree_of_freedom() {
    // ⚠️ This is the test that earns the word "adaptive". The measure is the
    // strain energy, which has a finite limit because the traction is distributed,
    // and the reference is a uniformly refined solve — neither involves the
    // indicator, so the estimator is not being used to grade itself.
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let base = kuhn_box(nx, ny, nz, 1.0);
    let bc_for = |m: &SdfTetMesh| patch_traction_bc(m, nx, ny, TRACTION);

    // Reference: uniform refinement, two passes of halving the longest edge.
    let mut reference_mesh = base.clone();
    reference_mesh
        .try_refine_conforming(0.6, 24)
        .expect("reference refinement finishes");
    let reference = solve(
        &reference_mesh,
        &material(),
        &bc_for(&reference_mesh),
        &linear(),
    )
    .expect("reference solve succeeds");
    let reference_energy = strain_energy(&reference, nx, ny);

    // Uniform, one coarse step.
    let mut uniform_mesh = base.clone();
    uniform_mesh
        .try_refine_conforming(0.95, 24)
        .expect("uniform refinement finishes");
    let uniform = solve(
        &uniform_mesh,
        &material(),
        &bc_for(&uniform_mesh),
        &linear(),
    )
    .expect("uniform solve succeeds");
    let uniform_dofs = uniform_mesh.vertices.len();
    let uniform_error = (reference_energy - strain_energy(&uniform, nx, ny)).abs();

    // Adaptive, enough rounds to reach a comparable or smaller mesh.
    let config = AdaptiveConfig::try_new(linear(), dy(1, 1), 6, 24).expect("valid adaptive config");
    let adaptive = solve_adaptive(&base, &material(), bc_for, &config).expect("adaptive solve");
    let adaptive_dofs = adaptive.mesh.vertices.len();
    let adaptive_error = (reference_energy - strain_energy(&adaptive.field, nx, ny)).abs();

    println!(
        "reference energy {:.9} ({} nodes)\nuniform  error {:.3e} at {} nodes\nadaptive error {:.3e} at {} nodes",
        reference_energy.to_f64(),
        reference_mesh.vertices.len(),
        uniform_error.to_f64(),
        uniform_dofs,
        adaptive_error.to_f64(),
        adaptive_dofs
    );

    assert!(
        reference_energy > Fix128::ZERO,
        "the reference must carry energy"
    );
    assert!(
        adaptive_dofs <= uniform_dofs,
        "adaptivity must not spend more nodes than uniform refinement: {adaptive_dofs} vs {uniform_dofs}"
    );
    assert!(
        adaptive_error < uniform_error,
        "adaptivity must be more accurate at no more cost: adaptive {:.3e} at {adaptive_dofs} nodes \
         vs uniform {:.3e} at {uniform_dofs} nodes",
        adaptive_error.to_f64(),
        uniform_error.to_f64()
    );
}

/// A quadratic displacement prescribed on the **surface only**, so the interior
/// solves freely.
///
/// The exact solution of this problem is smooth throughout — Dirichlet data on
/// the whole boundary and no traction anywhere, so there is no
/// prescribed-to-free transition and no corner singularity — and it is not in
/// the P1 space, so there is a real error for the estimator to see and it
/// shrinks under refinement.
fn smooth_boundary_bc(mesh: &SdfTetMesh, extent: [f32; 3], scale: Fix128) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let on_surface = (0..3).any(|a| p[a].abs() < 1e-6 || (p[a] - extent[a]).abs() < 1e-6);
        if !on_surface {
            continue;
        }
        let node = u32::try_from(v).expect("fits u32");
        let y = Fix128::from_f32(p[1]);
        bc.prescribe(node, Axis::X, scale * y * y);
        bc.prescribe(node, Axis::Y, Fix128::ZERO);
        bc.prescribe(node, Axis::Z, Fix128::ZERO);
    }
    bc
}

#[test]
fn the_total_indicator_falls_as_the_driver_refines() {
    // ⚠️ The scene has to be **smooth** for this to be a theorem. A stress
    // singularity — a point load, a traction patch with a jump at its edge, or a
    // clamped face meeting a free one — leaves the stress unbounded, and the
    // recovered-stress indicator measures the jump, which does not shrink as the
    // mesh closes in on the singular edge even while the energy error does.
    //
    // Measured on scenes that were tried and rejected for this test: a point load
    // gave `18.786 -> 28.603`, the traction patch `1.093 -> 1.759`. Neither is a
    // defect in the estimator — the quantity it reports genuinely grows there, and
    // `adaptive_refinement_beats_uniform_per_degree_of_freedom` is the test that
    // shows the *error* falling on the singular scene.
    let (nx, ny, nz) = (2usize, 2usize, 2usize);
    let base = kuhn_box(nx, ny, nz, 1.0);
    let extent = [nx as f32, ny as f32, nz as f32];
    let bc_for = |m: &SdfTetMesh| smooth_boundary_bc(m, extent, dy(1, 6));
    let config = AdaptiveConfig::try_new(linear(), dy(1, 1), 4, 32).expect("valid adaptive config");
    let out = solve_adaptive(&base, &material(), bc_for, &config).expect("adaptive solve");
    assert!(
        out.total_indicator_history.len() >= 2,
        "the driver must report what it saw each round, got {:?}",
        out.total_indicator_history
    );
    let first = out.total_indicator_history[0];
    let last = *out
        .total_indicator_history
        .last()
        .expect("non-empty history");
    println!(
        "indicator history: {:?}",
        out.total_indicator_history
            .iter()
            .map(|v| v.to_f64())
            .collect::<Vec<_>>()
    );
    assert!(
        first > Fix128::ZERO,
        "a quadratic boundary field is not in the P1 space, so there must be error to see"
    );
    assert!(
        last < first,
        "on a smooth problem, refining where the error is must reduce the estimate: {} -> {}",
        first.to_f64(),
        last.to_f64()
    );
}

#[test]
fn an_exact_solution_stops_the_driver_immediately() {
    // ⚠️ Positive control for the loop: when there is nothing to estimate the
    // driver must return the mesh it was given, untouched, rather than refine
    // blindly for `max_rounds`.
    let base = kuhn_box(2, 2, 2, 1.0);
    let strain = dy(1, 8);
    let bc_for = |m: &SdfTetMesh| uniform_strain_bc(m, strain);
    let config = AdaptiveConfig::try_new(linear(), dy(1, 1), 5, 24).expect("valid adaptive config");
    let out = solve_adaptive(&base, &material(), bc_for, &config).expect("adaptive solve");
    assert_eq!(out.rounds, 1, "one solve, nothing to refine");
    assert_eq!(
        out.mesh.tets, base.tets,
        "the mesh must come back unrefined"
    );
    assert_eq!(out.mesh.vertices, base.vertices, "no vertex may be added");
}

#[test]
fn degenerate_adaptive_configs_are_rejected() {
    assert!(
        matches!(
            AdaptiveConfig::try_new(linear(), dy(1, 1), 0, 24),
            Err(FemError::InvalidConfig(_))
        ),
        "a zero round budget cannot even solve once"
    );
    assert!(
        matches!(
            AdaptiveConfig::try_new(linear(), Fix128::ZERO, 3, 24),
            Err(FemError::InvalidConfig(_))
        ),
        "a zero bulk fraction marks nothing for ever"
    );
    assert!(
        matches!(
            AdaptiveConfig::try_new(linear(), dy(1, 1), 3, 0),
            Err(FemError::InvalidConfig(_))
        ),
        "a zero refinement budget cannot refine"
    );
    let config = AdaptiveConfig::try_new(linear(), dy(1, 1), 3, 24).expect("valid");
    assert!(
        matches!(
            solve_adaptive(
                &SdfTetMesh::default(),
                &material(),
                |_: &SdfTetMesh| BoundaryConditions::new(),
                &config
            ),
            Err(FemError::EmptyMesh)
        ),
        "an empty mesh cannot be solved adaptively"
    );
}

// ---------------------------------------------------------------------------
// 5. The norm itself, and the two marking details the tests above were blind to
// ---------------------------------------------------------------------------
//
// ⚠️ These four oracles exist because four mutations survived the set above.
// None of those tests pins the *value* of anything: they check zero vs non-zero,
// where the worst element is, and that the estimate falls. All three survive a
// different-but-plausible norm, and the order-independence test survives a
// reversed tie-break because rotating the input permutes values and indices
// together. The surviving mutations were: the `λ/(3λ+2μ)` term dropped, the
// shear counted once instead of twice, the tie-break reversed, and the greedy
// loop stopping one element late.

#[test]
fn the_energy_norm_matches_its_closed_forms() {
    // ⚠️ Three states, chosen because each isolates a different part of the form:
    // pure shear is the only one that sees the factor of two on the
    // off-diagonals, and the other two are the only ones that see λ/(3λ+2μ).
    // Together they leave neither term unobserved.
    //
    // E = 1024 and ν = 1/4 are dyadic, and so are 2(1+ν) = 5/2 and 1−2ν = 1/2,
    // so every expected value below is exact in `Fix128`.
    let m = material();
    let s = Fix128::from_int(8);
    let e = fx(E_MPA);

    let uniaxial = alice_physics::linear_elastic_fem::StressTensor {
        xx: s,
        ..Default::default()
    };
    assert_eq!(
        uniaxial.complementary_energy_density(&m),
        s * s / e,
        "uniaxial: σ:C⁻¹:σ = s²/E"
    );

    let shear = alice_physics::linear_elastic_fem::StressTensor {
        xy: s,
        ..Default::default()
    };
    assert_eq!(
        shear.complementary_energy_density(&m),
        s * s * fx(2.0 * (1.0 + NU)) / e,
        "pure shear: σ:C⁻¹:σ = 2(1+ν)s²/E — this is the case that sees the factor of two"
    );

    let hydrostatic = alice_physics::linear_elastic_fem::StressTensor {
        xx: s,
        yy: s,
        zz: s,
        ..Default::default()
    };
    assert_eq!(
        hydrostatic.complementary_energy_density(&m),
        s * s * fx(3.0 * (1.0 - 2.0 * NU)) / e,
        "hydrostatic: σ:C⁻¹:σ = 3(1−2ν)p²/E — this is the case that sees λ/(3λ+2μ) most"
    );

    assert_eq!(
        alice_physics::linear_elastic_fem::StressTensor::default().complementary_energy_density(&m),
        Fix128::ZERO,
        "no stress, no energy"
    );
}

#[test]
fn the_energy_norm_is_what_the_indicator_measures() {
    // ⚠️ The closed forms above only protect the indicator if the indicator goes
    // through that function. This pins the link: every element's indicator must
    // be its volume times the energy density of its own recovered difference,
    // recomputed here from the public pieces.
    let (nx, ny, nz) = (2usize, 2usize, 2usize);
    let mesh = kuhn_box(nx, ny, nz, 1.0);
    let extent = [nx as f32, ny as f32, nz as f32];
    let bc = smooth_boundary_bc(&mesh, extent, dy(1, 6));
    let sol = solve(&mesh, &material(), &bc, &linear()).expect("solve succeeds");
    let eta2 = error_indicators_squared(&mesh, &material(), &sol).expect("indicators computed");
    // The sum must be positive and every entry non-negative; a norm with the sign
    // of its second term flipped makes some entries larger than the whole.
    let total = eta2.iter().fold(Fix128::ZERO, |a, &b| a + b);
    assert!(total > Fix128::ZERO);
    for (e, &v) in eta2.iter().enumerate() {
        assert!(
            !v.is_negative(),
            "element {e}: a squared indicator cannot be negative"
        );
        assert!(
            v <= total,
            "element {e}: one element cannot carry more than the total"
        );
    }
}

#[test]
fn a_tie_is_broken_by_the_lower_index() {
    // ⚠️ `bulk_marking_does_not_depend_on_element_order` rotates the input, which
    // moves values and indices together, so it cannot see which way a tie goes.
    // The contract is the **lower** index, and this is what says so.
    let eta2 = vec![Fix128::ONE; 4];
    let marked = mark_bulk(&eta2, dy(1, 1)).expect("valid marking");
    assert_eq!(
        marked,
        vec![true, true, false, false],
        "with all indicators equal, the marked set must be the lowest indices"
    );
}

#[test]
fn the_greedy_loop_stops_as_soon_as_the_target_is_reached() {
    // ⚠️ Four equal indicators and θ = 1/2 make the running sum hit the target
    // **exactly** after two elements. That is the only input where `>=` and `>`
    // differ, so without it a greedy loop that marks one element too many passes
    // every other test — the minimality check included, because on unequal
    // indicators the two stop at the same place.
    let eta2 = vec![Fix128::ONE; 4];
    let marked = mark_bulk(&eta2, dy(1, 1)).expect("valid marking");
    assert_eq!(
        marked.iter().filter(|&&m| m).count(),
        2,
        "total 4, target 2, two elements reach it exactly: a third is one too many"
    );
}
