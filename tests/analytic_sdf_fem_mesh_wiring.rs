//! Oracle: wiring for the ten `sdf_fem_mesh` items that had no production
//! caller (`scripts/wiring-baseline.txt`): `BoundaryFaceError`,
//! `CUBE_FIVE_TETS`, `boundary_faces`, `cell_parity`, `generate`,
//! `generate_marching_tets`, `max_edge_length`, `refine_by_max_edge_length`,
//! `tet_count`, `vertex_count`.
//!
//! The production entry point is `examples/sdf_fem_mesh_generation.rs`
//! (`cargo run --example sdf_fem_mesh_generation --features std`).
//! `CUBE_FIVE_TETS` and `cell_parity` are `pub(crate)`, so they cannot be
//! named from here or from the example — they are reached transitively, from
//! inside `generate`'s and `generate_marching_tets`'s own bodies, once those
//! two functions themselves have a production caller. This file checks that
//! what they compute is right, not merely that something calls them.
//!
//! `tests/analytic_boundary_faces.rs` already holds a thorough,
//! independently-derived oracle for `generate` / `tet_count` / `vertex_count`
//! / `boundary_faces` / `BoundaryFaceError` on cubic `n^3` blocks. This file
//! does not repeat that: it covers what that file does not —
//! `generate_marching_tets`, `max_edge_length`, the two decompositions
//! `CUBE_FIVE_TETS` alternates between (via a *non-cubic* block, where a
//! fixed decomposition would be visibly wrong on more than one axis), the
//! deprecated `refine_by_max_edge_length`, and the degenerate inputs.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The reference computations below work in plain f32/f64 geometry; the
// crate's own determinism gate (`Fix128` + `alice_physics::det_math`) is not
// in scope for an independent check written against it.
#![allow(clippy::disallowed_methods)]

use std::panic::{self, AssertUnwindSafe};

use alice_physics::det_math::sqrt64;
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{
    generate, generate_marching_tets, BoundaryFaceError, SdfTetMesh,
};

// ---------------------------------------------------------------------------
// SDFs
// ---------------------------------------------------------------------------

/// Axis-aligned box `[min, max]`. Distance is the usual "max of the six
/// slab distances" box SDF; every point of `[min, max]` itself — including
/// its boundary — has distance `<= 0.0`.
fn box_region_sdf(min: [f32; 3], max: [f32; 3]) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| {
            let dx = (min[0] - x).max(x - max[0]);
            let dy = (min[1] - y).max(y - max[1]);
            let dz = (min[2] - z).max(z - max[2]);
            dx.max(dy).max(dz)
        },
        |_x, _y, _z| (1.0, 0.0, 0.0),
    )
}

/// Half-space `x < plane_x`, as a (non-compact, intentionally not a closed
/// shape) SDF: `distance(x, y, z) = x - plane_x`.
///
/// Used only with `generate_marching_tets` on a single cube, where it stands
/// in for a flat cut through the cell. Since the field is affine, the
/// crossing it produces on every edge is the *exact* intersection with
/// `x = plane_x`, not merely a linear approximation of a curved one, so the
/// clipped volume has a closed form.
fn half_space_sdf(plane_x: f32) -> ClosureSdf {
    ClosureSdf::new(move |x, _y, _z| x - plane_x, |_x, _y, _z| (1.0, 0.0, 0.0))
}

// ---------------------------------------------------------------------------
// Independent reference: tetrahedron volume by the scalar triple product.
//
// This is the same formula `tet_signed_volume_x6` uses, written fresh here
// rather than called, per the "don't call the implementation to build the
// oracle" rule — the point is to catch the implementation computing the
// wrong clip, not to re-measure its own arithmetic.
// ---------------------------------------------------------------------------

fn tet_volume(p: [[f32; 3]; 4]) -> f64 {
    let e = |i: usize, k: usize| f64::from(p[i][k]) - f64::from(p[0][k]);
    let a = [e(1, 0), e(1, 1), e(1, 2)];
    let b = [e(2, 0), e(2, 1), e(2, 2)];
    let c = [e(3, 0), e(3, 1), e(3, 2)];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    det.abs() / 6.0
}

fn mesh_volume(mesh: &SdfTetMesh) -> f64 {
    mesh.tets
        .iter()
        .map(|t| tet_volume(t.vertices.map(|i| mesh.vertices[i as usize])))
        .sum()
}

// ---------------------------------------------------------------------------
// generate / tet_count / vertex_count / max_edge_length: one lattice cell
// ---------------------------------------------------------------------------

/// Oracle, derived by hand (not from running the generator):
///
/// A cube whose corners coincide exactly with one lattice cell is diced,
/// whole, into 5 tetrahedra built only from its 8 existing corners (no new
/// vertex is ever introduced by the 5-tet decomposition) — so
/// `tet_count == 5` and `vertex_count == 8` for *any* single fully-interior
/// cell, regardless of which of the two `CUBE_FIVE_TETS` rows is used.
///
/// The longest edge in either row is a face diagonal of the cube
/// (`cell * sqrt(2)`): enumerating both rows by hand against the corner
/// numbering in the module doc shows every tetrahedron has at least one such
/// diagonal and none has the cube's body diagonal (`cell * sqrt(3)`) or an
/// edge longer than that.
#[test]
fn one_lattice_cell_has_five_tets_eight_vertices_and_a_face_diagonal_max_edge() {
    let cell = 2.0_f32;
    let sdf = box_region_sdf([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let mesh = generate(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], cell);

    assert_eq!(mesh.tet_count(), 5, "oracle: one cell dices into 5 tets");
    assert_eq!(mesh.vertex_count(), 8, "oracle: no new vertex is created");

    let expected_max_edge = f64::from(cell) * sqrt64(2.0);
    let got = f64::from(mesh.max_edge_length());
    assert!(
        (got - expected_max_edge).abs() < 1.0e-5,
        "max_edge_length = {got}, expected cell*sqrt(2) = {expected_max_edge}"
    );

    // And the other decomposition-independent fact: the vertex set is
    // exactly the 8 lattice corners, no more, no fewer.
    let mut corners: Vec<[f32; 3]> = Vec::new();
    for dz in [-1.0_f32, 1.0] {
        for dy in [-1.0_f32, 1.0] {
            for dx in [-1.0_f32, 1.0] {
                corners.push([dx, dy, dz]);
            }
        }
    }
    let mut got_vertices = mesh.vertices.clone();
    got_vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    corners.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        got_vertices, corners,
        "oracle: vertex set is exactly the 8 corners"
    );
}

// ---------------------------------------------------------------------------
// cell_parity / CUBE_FIVE_TETS: a non-cubic block
//
// A cubic n^3 block is already covered, independently, in
// `tests/analytic_boundary_faces.rs`. A non-cubic block is used here on
// purpose: `cell_parity` alternates by `(ix + iy + iz) mod 2`, so a block
// that is not the same length on every axis exercises the alternation
// differently along each one, whereas an n^3 block cannot distinguish
// "alternates along every axis" from "alternates along one axis and happens
// to look right on the others" (all three look the same by symmetry there).
// ---------------------------------------------------------------------------

/// Oracle, derived by hand from the generator's own occupancy and dicing
/// rules (not from running it):
///
/// `(nx, ny, nz)` fully-interior cells give `5 * nx * ny * nz` tets on
/// `(nx+1) * (ny+1) * (nz+1)` lattice corners (no new vertex, as above).
///
/// For the boundary count: every cell contributes `5 * 4 = 20` triangle
/// slots. A face on the exterior of the `nx * ny * nz` block is used once; a
/// face shared between two adjacent interior cells is used twice *provided*
/// the two cells agree on how they triangulated it — which is exactly what
/// alternating `CUBE_FIVE_TETS` by `cell_parity` guarantees (see the module
/// doc's `# Conformity` section). Assuming that, the boundary is the
/// surface of the solid block, 2 triangles per unit square: the block has
/// `2*(nx*ny) + 2*(nx*nz) + 2*(ny*nz)` exposed unit squares, so
/// `4 * (nx*ny + nx*nz + ny*nz)` boundary triangles.
///
/// If the alternation were wrong, the quad shared between two adjacent
/// cells would instead be triangulated two different ways from either side,
/// and the resulting 2 mismatched triangles on each side would each be used
/// *once* rather than paired up — i.e. counted as boundary — inflating this
/// count by 4 for every internal face along the axis that stopped
/// alternating. A `(3, 2, 1)` block has internal faces on all three axes
/// (2 along x, 1 along y, 0 along z — z has only one cell, so it alone
/// cannot catch a broken alternation), so getting the exact closed-form
/// count right is evidence the x and y alternation are both working.
#[test]
fn a_non_cubic_block_matches_the_closed_form_tet_vertex_and_boundary_counts() {
    let (nx, ny, nz) = (3_i32, 2_i32, 1_i32);
    let min = [0.0_f32, 0.0, 0.0];
    let max = [nx as f32, ny as f32, nz as f32];
    let sdf = box_region_sdf(min, max);
    let mesh = generate(&sdf, min, max, 1.0);

    assert_eq!(
        mesh.tet_count(),
        5 * (nx * ny * nz) as usize,
        "oracle: 5 tets per cell"
    );
    assert_eq!(
        mesh.vertex_count(),
        ((nx + 1) * (ny + 1) * (nz + 1)) as usize,
        "oracle: lattice corners, no new vertex"
    );

    let expected_boundary = 4 * (nx * ny + nx * nz + ny * nz);
    let got_boundary = mesh
        .boundary_faces()
        .expect("generate() produces a conforming mesh")
        .len();
    assert_eq!(
        got_boundary, expected_boundary as usize,
        "oracle: 4*(nx*ny + nx*nz + ny*nz) boundary triangles (non-conforming dicing would \
         inflate this)"
    );
}

// ---------------------------------------------------------------------------
// generate_marching_tets
// ---------------------------------------------------------------------------

/// Oracle: with no zero crossing anywhere in the sampled region, marching
/// tetrahedra has nothing to clip, so it reproduces `generate`'s mesh
/// exactly — same tet and vertex counts (the *same* closed form as the
/// previous test, since the shape still fills the block exactly).
///
/// The shape is padded half a cell past the meshed block on every side, so
/// every lattice corner has distance strictly less than zero. Without the
/// padding every corner sits exactly *on* the shape's boundary (distance
/// `== 0.0`), which is a different, and differently handled, case — see
/// the next test.
#[test]
fn generate_marching_tets_matches_generate_exactly_when_nothing_crosses() {
    let (nx, ny, nz) = (3_i32, 2_i32, 1_i32);
    let min = [0.0_f32, 0.0, 0.0];
    let max = [nx as f32, ny as f32, nz as f32];
    let pad = 0.5_f32;
    let sdf = box_region_sdf(
        [min[0] - pad, min[1] - pad, min[2] - pad],
        [max[0] + pad, max[1] + pad, max[2] + pad],
    );

    let interior = generate(&sdf, min, max, 1.0);
    let surface = generate_marching_tets(&sdf, min, max, 1.0);

    assert_eq!(surface.tet_count(), interior.tet_count());
    assert_eq!(surface.vertex_count(), interior.vertex_count());
    assert_eq!(
        surface.tet_count(),
        5 * (nx * ny * nz) as usize,
        "oracle: same closed form as generate() alone"
    );
}

/// ⚠️ Found while building the oracle above, and reported rather than fixed
/// (out of scope for this wiring pass): drop the
/// padding, so every lattice corner sits *exactly on* the shape's boundary
/// (distance `== 0.0`), and the two generators go from agreeing to
/// disagreeing in the direction the module doc does not describe.
/// `generate`'s occupancy rule is `distance <= 0.0`, so it meshes every
/// cell. `generate_marching_tets`'s inside/outside mask
/// (`marching_tet_emit`) tests `d < 0.0` (strict), so a corner at exactly
/// `0.0` counts as outside; with every corner at `0.0`, every
/// sub-tetrahedron's mask is `0b0000` and nothing is emitted at all. That
/// contradicts both the module doc ("also clips the cubes the surface
/// crosses... see `generate` for the fully-interior variant") and the
/// crate's own unit test
/// `marching_tets_produces_more_tets_than_interior_only`, which asserts
/// `marching.tet_count() >= interior.tet_count()` — here it is `0 >= 30`.
///
/// This pins the measured behaviour as a fact, not as a claim that it is
/// correct.
#[test]
fn exact_zero_distance_at_every_corner_is_meshed_by_generate_but_not_by_marching_tets() {
    let (nx, ny, nz) = (3_i32, 2_i32, 1_i32);
    let min = [0.0_f32, 0.0, 0.0];
    let max = [nx as f32, ny as f32, nz as f32];
    let sdf = box_region_sdf(min, max);

    let interior = generate(&sdf, min, max, 1.0);
    let surface = generate_marching_tets(&sdf, min, max, 1.0);

    assert_eq!(
        interior.tet_count(),
        5 * (nx * ny * nz) as usize,
        "generate meshes every cell (<=0.0 occupancy)"
    );
    assert_eq!(
        surface.tet_count(),
        0,
        "generate_marching_tets discards every cell: every corner is at distance exactly 0.0, \
         which its mask (strict d < 0.0) does not count as inside"
    );
}

/// Oracle, derived from the SDF being affine (not from running the mesher):
///
/// `distance(x, y, z) = x - 0.375` cuts the unit cube `[0,1]^3` at exactly
/// `x = 0.375`. Marching tetrahedra places every crossing vertex at the
/// *exact* linear interpolation of the (here, already linear) field along
/// each lattice edge, so the clipped solid is exactly the slab
/// `{x < 0.375}`, whatever the 5-tet decomposition did internally — its
/// volume is `0.375 * 1 * 1`.
///
/// `0.375 = 3/8` is exactly representable in `f32`, so the comparison is
/// tight (not merely "close enough for an iso-surface").
#[test]
fn generate_marching_tets_volume_matches_the_half_space_closed_form() {
    let plane_x = 0.375_f32;
    let sdf = half_space_sdf(plane_x);
    let mesh = generate_marching_tets(&sdf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);

    assert!(mesh.tet_count() > 0, "the plane crosses this cube");
    let got = mesh_volume(&mesh);
    let expected = f64::from(plane_x);
    assert!(
        (got - expected).abs() < 1.0e-5,
        "clipped volume = {got}, expected plane_x*1*1 = {expected}"
    );
}

/// Degenerate input: a region with no zero crossing and nothing inside it —
/// the lattice has no corner the mesher calls "inside", so both generators
/// agree the mesh is empty.
#[test]
fn a_region_with_no_zero_crossing_yields_an_empty_mesh_from_both_generators() {
    let sdf = box_region_sdf([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let far = ([10.0_f32, 10.0, 10.0], [11.0_f32, 11.0, 11.0]);

    let interior = generate(&sdf, far.0, far.1, 0.5);
    assert_eq!(interior.tet_count(), 0);
    assert_eq!(interior.vertex_count(), 0);
    assert_eq!(
        interior.max_edge_length(),
        0.0,
        "oracle: documented empty-mesh value"
    );

    let surface = generate_marching_tets(&sdf, far.0, far.1, 0.5);
    assert_eq!(surface.tet_count(), 0);
    assert_eq!(surface.vertex_count(), 0);
}

// ---------------------------------------------------------------------------
// boundary_faces / BoundaryFaceError
// ---------------------------------------------------------------------------

/// Degenerate input: an empty mesh has no faces to claim and no face can be
/// non-manifold, so `boundary_faces()` is `Ok(vec![])`, not an error.
#[test]
fn boundary_faces_on_an_empty_mesh_is_ok_and_empty() {
    let mesh = SdfTetMesh::default();
    assert_eq!(mesh.boundary_faces(), Ok(Vec::new()));
}

/// The error variant itself, matched by field: three tetrahedra sharing one
/// triangular face is refused with the exact use count and the exact
/// (sorted) face it disagreed about — not merely "an error".
#[test]
fn boundary_faces_reports_which_face_and_how_many_tets_claimed_it() {
    let mut mesh = SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        tets: Vec::new(),
    };
    for fourth in [3_u32, 4, 5] {
        mesh.tets.push(alice_physics::sdf_fem_mesh::Tetrahedron {
            vertices: [0, 1, 2, fourth],
        });
    }
    match mesh.boundary_faces() {
        Err(BoundaryFaceError::NonManifoldFace { face, uses }) => {
            assert_eq!(face, [0, 1, 2], "oracle: the shared face, sorted");
            assert_eq!(uses, 3, "oracle: three tetrahedra claimed it");
        }
        other => panic!("expected NonManifoldFace{{ face: [0,1,2], uses: 3 }}, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// refine_by_max_edge_length (deprecated) vs try_refine_conforming
// ---------------------------------------------------------------------------

fn one_tet_mesh(side: f32) -> SdfTetMesh {
    SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, side],
        ],
        tets: vec![alice_physics::sdf_fem_mesh::Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    }
}

/// Oracle: the module doc says the deprecated function "performs the same
/// refinement... only the reporting is worse" — so on an identical starting
/// mesh and parameters where the budget suffices, the two must leave the
/// mesh in the exact same state (same tet/vertex count, same
/// `max_edge_length`), and the deprecated one's pass count must equal
/// `try_refine_conforming`'s `Ok` value.
#[test]
#[allow(deprecated)]
fn deprecated_refine_matches_try_refine_conforming_when_the_budget_suffices() {
    let mut via_new = one_tet_mesh(1.0);
    let new_passes = via_new
        .try_refine_conforming(0.7, 32)
        .expect("thirty-two passes finishes a single tetrahedron");

    let mut via_old = one_tet_mesh(1.0);
    let old_passes = via_old.refine_by_max_edge_length(0.7, 32);

    assert_eq!(
        old_passes, new_passes,
        "oracle: same pass count when it finishes"
    );
    assert_eq!(via_old.tet_count(), via_new.tet_count());
    assert_eq!(via_old.vertex_count(), via_new.vertex_count());
    assert_eq!(via_old.max_edge_length(), via_new.max_edge_length());
}

/// Oracle, from the deprecation note itself: when the budget runs out with
/// work left, the deprecated function returns `max_passes` (indistinguishable
/// from a clean finish), which is exactly the ambiguity
/// `try_refine_conforming` was written to remove.
#[test]
#[allow(deprecated)]
fn deprecated_refine_reports_max_passes_on_an_exhausted_budget_old_signature_behaviour() {
    let mut via_new = one_tet_mesh(1.0);
    let result = via_new.try_refine_conforming(0.7, 4);
    assert!(
        matches!(
            result,
            Err(alice_physics::sdf_fem_mesh::RefineError::Unfinished { passes: 4, .. })
        ),
        "oracle: four passes is not enough here: {result:?}"
    );

    let mut via_old = one_tet_mesh(1.0);
    let old_passes = via_old.refine_by_max_edge_length(0.7, 4);
    assert_eq!(
        old_passes, 4,
        "oracle: exhausted budget reports max_passes, same as a finish"
    );
}

// ---------------------------------------------------------------------------
// Degenerate inputs: panics, measured rather than assumed
// ---------------------------------------------------------------------------

#[test]
fn generate_panics_on_nonpositive_cell_zero_and_negative() {
    let sdf = box_region_sdf([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    for bad_cell in [0.0_f32, -1.0_f32, -0.5_f32] {
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            generate(&sdf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], bad_cell)
        }));
        assert!(
            outcome.is_err(),
            "generate({bad_cell}) should panic, not return a mesh"
        );
    }
}

#[test]
fn generate_marching_tets_panics_on_nonpositive_cell_zero_and_negative() {
    let sdf = box_region_sdf([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    for bad_cell in [0.0_f32, -1.0_f32, -0.5_f32] {
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            generate_marching_tets(&sdf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], bad_cell)
        }));
        assert!(
            outcome.is_err(),
            "generate_marching_tets({bad_cell}) should panic, not return a mesh"
        );
    }
}

#[test]
#[allow(deprecated)]
fn deprecated_refine_panics_on_nonpositive_threshold() {
    for bad_threshold in [0.0_f32, -1.0_f32] {
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            let mut m = one_tet_mesh(1.0);
            m.refine_by_max_edge_length(bad_threshold, 4)
        }));
        assert!(
            outcome.is_err(),
            "refine_by_max_edge_length(.., {bad_threshold}, ..) should panic"
        );
    }
}

/// Extreme size: a cell far larger than the bounding box does not panic —
/// `generate`'s grid dimensions are `((max-min)/cell).max(1.0)`, which stays
/// `1` however large `cell` gets, so this is one ordinary cube with very
/// large corner coordinates, not an overflow. Measured rather than assumed:
/// `catch_unwind` reports whether it actually panics.
#[test]
fn generate_with_a_cell_far_larger_than_the_bounding_box_does_not_panic() {
    // `box_region_sdf` spans the whole finite f32 range, so every corner of
    // the one oversized cube below — whatever its magnitude — lands well
    // inside it rather than at the boundary, keeping this test about the
    // overflow path in `generate`, not about the box SDF's own edges.
    let sdf = box_region_sdf(
        [-f32::MAX, -f32::MAX, -f32::MAX],
        [f32::MAX, f32::MAX, f32::MAX],
    );
    let cell = f32::MAX / 2.0;
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        generate(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], cell)
    }));
    assert!(
        outcome.is_ok(),
        "an oversized cell should clamp to one cube, not panic: {outcome:?}"
    );
    let mesh = outcome.unwrap();
    assert_eq!(
        mesh.tet_count(),
        5,
        "oracle: still exactly one cell's worth of cube"
    );
    assert_eq!(mesh.vertex_count(), 8);
}
