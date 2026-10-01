//! Degenerate-input behaviour of the hyperelastic P2 / P3 entry points.
//!
//! The closed-form oracles live in `analytic_quadratic_hyperelastic.rs` and
//! `analytic_cubic_hyperelastic.rs`; this file asks the other question: what
//! happens on input the solver is **not** meant for.
//!
//! ⚠️ **The contract asserted here is "an `Err`, never a panic".** Every one of
//! these entry points returns `Result<_, FemError>`, so a degenerate mesh, an
//! out-of-range boundary index, a configuration with no material law, or a
//! deformation that turns an element inside out must all come back as a named
//! variant. An index panic, a divide by zero or an arithmetic overflow would be
//! a defect even though the input is nonsense — a library cannot unwind through
//! a caller that is not expecting it.
//!
//! ⚠️ **Each guard is run against its own absence.** A test that says "this
//! input is refused" is worth nothing if the input would be refused anyway by
//! something further down, so the reason the refusal happens is pinned by the
//! variant, not just by `is_err()`. The mutation evidence for the guards
//! themselves is recorded in
//! `memory/feedback_hyperelastic_on_high_order_elements.md`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// ⚠️ **Excluded from the `parallel` feature build on purpose.**
// `cargo test --features "parallel"` (ci.yml:77) would otherwise run this file
// a second time, and `parallel` reaches none of the code it exercises:
// `grep -cE 'feature = "parallel"|rayon|par_iter'` is **0** for every module in
// the transitive closure of these tests (`quadratic_elastic_fem`,
// `cubic_elastic_fem`, `linear_elastic_fem`, `hyperelastic`, `math`,
// `sdf_fem_mesh`, `coupled_field`, `sdf_collider`, `sim_field`, `collider`,
// `metric`). The feature bites only the rigid-body solvers (`solver.rs`,
// `solver_tgs*.rs`, `query.rs`, `multi_world.rs`, `laminate.rs`,
// `eulerian_grid.rs`). ⚠️ The cost of the duplicate run is real and the
// information it adds is zero, so the file opts out — which also means it does
// **not** run under `--all-features`.
#![cfg(not(feature = "parallel"))]

use alice_physics::cubic_elastic_fem::{solve_cubic_hyperelastic, CubicMesh};
use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics::math::{Fix128, PolarError};
use alice_physics::quadratic_elastic_fem::{solve_quadratic_hyperelastic, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const SIDE: f64 = 4.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(10.0), fx(0.45)).expect("valid")
}

fn model() -> HyperelasticModel {
    HyperelasticModel::NeoHookean { mu_mpa: fx(3.0) }
}

fn config() -> CorotationalConfig {
    let linear = SolverConfig::try_new(50_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(500, Fix128::from_raw(0, 1 << 54))
        .expect("valid");
    CorotationalConfig::try_new(linear, 20, fx(1.0e-6), 2, 64)
        .expect("valid")
        .with_hyperelastic(model())
}

/// One cube split into six tetrahedra.
fn unit_cube() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                mesh.vertices.push([
                    i as f32 * SIDE as f32,
                    j as f32 * SIDE as f32,
                    k as f32 * SIDE as f32,
                ]);
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
    for path in PATHS {
        let mut step = [0usize; 3];
        let mut corners = [0u32; 4];
        corners[0] = 0;
        for (m, axis) in path.into_iter().enumerate() {
            step[axis] = 1;
            corners[m + 1] = u32::try_from(step[0] + step[1] * 2 + step[2] * 4).expect("fits");
        }
        mesh.tets.push(Tetrahedron { vertices: corners });
    }
    mesh
}

/// Clamp every node, which is a valid (if trivial) problem: it exercises the
/// fully-prescribed path without a solve.
fn clamp_all_p2(mesh: &QuadraticMesh) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for n in 0..mesh.node_count() {
        bc.prescribe_all(
            u32::try_from(n).expect("fits"),
            [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
        );
    }
    bc
}

// ---------------------------------------------------------------------------
// empty input
// ---------------------------------------------------------------------------

/// An empty mesh is refused **at construction**, so the solver never sees one.
///
/// ⚠️⚠️ **This test was written on a wrong premise and the measurement
/// corrected it.** The first version asserted that `from_tet_mesh` accepts an
/// empty mesh and that the solver returns `FemError::EmptyMesh`; it failed with
/// `an empty mesh is well formed: EmptyMesh`, because the **constructor** is
/// what refuses it. Recorded rather than quietly rewritten, because the
/// consequence is load bearing:
///
/// ⚠️ **`QuadraticMesh` and `CubicMesh` have private fields and exactly one
/// constructor each, so there is no way to hand an empty mesh to
/// `solve_*_hyperelastic` through the public API.** The `if node_count == 0 ||
/// mesh.elements.is_empty()` check inside the solvers is therefore a defensive
/// duplicate that no caller can reach — which is why removing it is a
/// *surviving* mutation rather than a caught one, and why the guard that
/// actually protects callers is the one asserted here.
#[test]
fn an_empty_mesh_is_refused_at_construction() {
    let empty = SdfTetMesh::default();
    match QuadraticMesh::from_tet_mesh(&empty) {
        Err(FemError::EmptyMesh) => {}
        other => panic!(
            "P2 from an empty mesh: expected EmptyMesh, got {:?}",
            other.map(|m| m.node_count())
        ),
    }
    match CubicMesh::from_tet_mesh(&empty) {
        Err(FemError::EmptyMesh) => {}
        other => panic!(
            "P3 from an empty mesh: expected EmptyMesh, got {:?}",
            other.map(|m| m.node_count())
        ),
    }

    // A mesh with vertices but no tetrahedra is the other half of "empty", and
    // it is refused by the same guard rather than producing a zero-element mesh.
    let mut no_tets = SdfTetMesh::default();
    no_tets.vertices.push([0.0, 0.0, 0.0]);
    match QuadraticMesh::from_tet_mesh(&no_tets) {
        Err(FemError::EmptyMesh) => {}
        other => panic!(
            "P2 from a mesh with no tets: expected EmptyMesh, got {:?}",
            other.map(|m| m.element_count())
        ),
    }
}

/// A zero-volume tetrahedron is refused **at construction**, with the element
/// index, not by a divide-by-zero inside the solver.
///
/// ⚠️ **Which layer refuses this was measured, not assumed.** The shape
/// function gradients are `∇λ = (face normal) / (3V)`, so `V = 0` is a division
/// by zero; the question is whether it is caught where the geometry is built or
/// where it is used. It is the former — `from_tet_mesh` returns
/// `FemError::DegenerateElement { tet }` — which means, as with the empty mesh,
/// that the solver never sees one through the public API.
#[test]
fn a_zero_volume_element_is_refused_at_construction() {
    let mut flat = SdfTetMesh::default();
    // four coplanar points: the tetrahedron they span has no volume
    flat.vertices.push([0.0, 0.0, 0.0]);
    flat.vertices.push([1.0, 0.0, 0.0]);
    flat.vertices.push([0.0, 1.0, 0.0]);
    flat.vertices.push([1.0, 1.0, 0.0]);
    flat.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });

    match QuadraticMesh::from_tet_mesh(&flat) {
        Err(FemError::DegenerateElement { tet }) => assert_eq!(tet, 0),
        other => panic!(
            "P2 from a flat tetrahedron: expected DegenerateElement, got {:?}",
            other.map(|m| m.element_count())
        ),
    }
    match CubicMesh::from_tet_mesh(&flat) {
        Err(FemError::DegenerateElement { tet }) => assert_eq!(tet, 0),
        other => panic!(
            "P3 from a flat tetrahedron: expected DegenerateElement, got {:?}",
            other.map(|m| m.element_count())
        ),
    }
}

// ---------------------------------------------------------------------------
// indices out of range
// ---------------------------------------------------------------------------

/// A boundary condition naming a node the mesh does not have is refused with
/// the index and the count, not with a slice panic.
///
/// ⚠️ Both the prescribed list **and** the load list are checked: they are
/// separate vectors and an implementation that validated only one of them would
/// pass half of this.
#[test]
fn a_boundary_index_past_the_end_is_refused_and_does_not_panic() {
    let tets = unit_cube();
    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let past = u32::try_from(p2.node_count()).expect("fits");

    let mut prescribed = BoundaryConditions::new();
    prescribed.prescribe(past, Axis::X, Fix128::ZERO);
    match solve_quadratic_hyperelastic(&p2, &material(), &prescribed, &config()) {
        Err(FemError::VertexOutOfRange {
            vertex,
            vertex_count,
        }) => {
            assert_eq!(vertex, past);
            assert_eq!(vertex_count, p2.node_count());
        }
        other => panic!("prescribed past the end: expected VertexOutOfRange, got {other:?}"),
    }

    let mut loaded = BoundaryConditions::new();
    loaded.add_load(past, Axis::Y, fx(1.0));
    match solve_quadratic_hyperelastic(&p2, &material(), &loaded, &config()) {
        Err(FemError::VertexOutOfRange { vertex, .. }) => assert_eq!(vertex, past),
        other => panic!("loaded past the end: expected VertexOutOfRange, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// not enough constraints
// ---------------------------------------------------------------------------

/// Fewer than six prescribed degrees of freedom cannot remove the six rigid
/// body modes, and the solver says so instead of iterating into a singular
/// system.
#[test]
fn an_under_constrained_problem_is_refused_and_does_not_panic() {
    let tets = unit_cube();
    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let mut bc = BoundaryConditions::new();
    // five, one short of the necessary six
    bc.prescribe(0, Axis::X, Fix128::ZERO);
    bc.prescribe(0, Axis::Y, Fix128::ZERO);
    bc.prescribe(0, Axis::Z, Fix128::ZERO);
    bc.prescribe(1, Axis::X, Fix128::ZERO);
    bc.prescribe(1, Axis::Y, Fix128::ZERO);
    match solve_quadratic_hyperelastic(&p2, &material(), &bc, &config()) {
        Err(FemError::UnderConstrained) => {}
        other => panic!("five constraints: expected UnderConstrained, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// a deformation with no positive determinant
// ---------------------------------------------------------------------------

/// Turning an element inside out is reported as `RotationFailed` with
/// [`PolarError::Inverted`], not as a panic and not as a silent answer.
///
/// ⚠️ **This is the guard the whole hyperelastic path rests on.** Every model in
/// `crate::hyperelastic` divides by `J = det F`, so `J ≤ 0` has no stress under
/// any of them; `cauchy_stress` returns `None` and the caller has to turn that
/// into an error rather than unwrapping it. The scene prescribes a reflection
/// (every node mapped to `−X`), which is the cleanest way to reach `det F < 0`
/// without relying on an iteration wandering there.
#[test]
fn a_reflected_element_is_refused_and_does_not_panic() {
    let tets = unit_cube();
    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let mut bc = BoundaryConditions::new();
    for n in 0..p2.node_count() {
        let p = p2
            .node_position(u32::try_from(n).expect("fits"))
            .expect("node in range");
        // x ↦ −x: the map has det = −1 everywhere
        bc.prescribe_all(
            u32::try_from(n).expect("fits"),
            [-p[0] - p[0], -p[1] - p[1], -p[2] - p[2]],
        );
    }
    match solve_quadratic_hyperelastic(&p2, &material(), &bc, &config()) {
        Err(FemError::RotationFailed {
            cause: PolarError::Inverted,
            ..
        }) => {}
        other => panic!("a reflection: expected RotationFailed/Inverted, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// a configuration with no law
// ---------------------------------------------------------------------------

/// The hyperelastic entry points refuse a configuration carrying no material
/// law rather than quietly solving the small-strain problem.
///
/// ⚠️ The silent fall-back is the failure this guards: it would make every
/// closed-form oracle in the sibling files pass against the wrong physics.
#[test]
fn a_configuration_without_a_law_is_refused_on_both_elements() {
    let tets = unit_cube();
    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let p3 = CubicMesh::from_tet_mesh(&tets).expect("well formed");
    let linear = SolverConfig::try_new(50_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let no_law = CorotationalConfig::try_new(linear, 20, fx(1.0e-6), 2, 64).expect("valid");

    match solve_quadratic_hyperelastic(&p2, &material(), &clamp_all_p2(&p2), &no_law) {
        Err(FemError::InvalidConfig(_)) => {}
        other => panic!("P2 without a law: expected InvalidConfig, got {other:?}"),
    }
    let mut bc3 = BoundaryConditions::new();
    for n in 0..p3.node_count() {
        bc3.prescribe_all(
            u32::try_from(n).expect("fits"),
            [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
        );
    }
    match solve_cubic_hyperelastic(&p3, &material(), &bc3, &no_law) {
        Err(FemError::InvalidConfig(_)) => {}
        other => panic!("P3 without a law: expected InvalidConfig, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// the trivial problem
// ---------------------------------------------------------------------------

/// A fully clamped body is a valid problem with a zero answer, and the
/// fully-prescribed shortcut must take it without dividing by a zero load norm.
///
/// ⚠️ This is the degenerate case most likely to be a divide by zero rather
/// than a panic: the external load is zero, so any `‖r‖/‖b‖` on this path has a
/// zero denominator.
#[test]
fn a_fully_clamped_body_returns_zero_without_dividing_by_zero() {
    let tets = unit_cube();
    let p2 = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let solution = solve_quadratic_hyperelastic(&p2, &material(), &clamp_all_p2(&p2), &config())
        .expect("a clamped body is a valid problem");
    assert_eq!(solution.newton_iterations, 0);
    for d in &solution.field.displacements {
        for axis in d {
            assert_eq!(*axis, Fix128::ZERO, "a clamped body must not move");
        }
    }
    // σ(I) = 0 exactly, for every model and every bulk modulus — the module doc
    // of `crate::hyperelastic` states it as a check a caller can make.
    for s in &solution.field.element_stress {
        for component in [s.xx, s.yy, s.zz, s.xy, s.yz, s.zx] {
            assert_eq!(
                component,
                Fix128::ZERO,
                "the undeformed state must be stress free"
            );
        }
    }
}
