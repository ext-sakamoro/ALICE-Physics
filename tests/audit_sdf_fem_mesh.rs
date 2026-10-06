//! Audit oracles for `alice_physics::sdf_fem_mesh`.
//! Reference values: exact volumes of half-space / box regions (polytope formulas), the
//! topological identities of a closed surface (Euler characteristic), and brute-force
//! recomputation of the generators' own occupancy rule. Nothing is read back from the mesh
//! to define what the mesh should be.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_fem_mesh::{generate, generate_marching_tets, RefineError, SdfTetMesh};
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn plane(n: [f32; 3], d: f32) -> ClosureSdf {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let u = [n[0] / len, n[1] / len, n[2] / len];
    ClosureSdf::new(
        move |x, y, z| u[0] * x + u[1] * y + u[2] * z - d,
        move |_, _, _| (u[0], u[1], u[2]),
    )
}

fn ball(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / l, y / l, z / l)
        },
    )
}

fn everywhere_inside() -> ClosureSdf {
    ClosureSdf::new(|_, _, _| -1.0, |_, _, _| (0.0, 0.0, 1.0))
}

fn signed_vol6(m: &SdfTetMesh, vs: [u32; 4]) -> f64 {
    let p = vs.map(|v| m.vertices[v as usize].map(f64::from));
    let e = |i: usize, k: usize| p[i][k] - p[0][k];
    let (a, b, c) = (
        [e(1, 0), e(1, 1), e(1, 2)],
        [e(2, 0), e(2, 1), e(2, 2)],
        [e(3, 0), e(3, 1), e(3, 2)],
    );
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn volume(m: &SdfTetMesh) -> f64 {
    m.tets
        .iter()
        .map(|t| signed_vol6(m, t.vertices) / 6.0)
        .sum()
}

/// Sum of the absolute tet volumes (orientation-independent region volume).
fn volume_abs(m: &SdfTetMesh) -> f64 {
    m.tets
        .iter()
        .map(|t| signed_vol6(m, t.vertices).abs() / 6.0)
        .sum()
}

fn face_uses(m: &SdfTetMesh) -> HashMap<[u32; 3], usize> {
    let mut u = HashMap::new();
    for t in &m.tets {
        let v = t.vertices;
        for f in [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[0], v[2], v[3]],
            [v[1], v[2], v[3]],
        ] {
            let mut k = f;
            k.sort_unstable();
            *u.entry(k).or_insert(0) += 1;
        }
    }
    u
}

/// Would any lattice corner be warped by `generate_marching_tets` (a zero crossing within 0.3
/// cell of a corner along one of the 26 lattice directions)? Used only as a precondition so
/// that exact polytope volumes can be asserted.
fn warp_free(sdf: &dyn SdfField, min: [f32; 3], cell: f32, dims: [i32; 3]) -> bool {
    let d = |i: i32, j: i32, k: i32| {
        sdf.distance(
            min[0] + i as f32 * cell,
            min[1] + j as f32 * cell,
            min[2] + k as f32 * cell,
        )
    };
    for k in 0..=dims[2] {
        for j in 0..=dims[1] {
            for i in 0..=dims[0] {
                let s0 = d(i, j, k);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let (a, b, c) = (i + dx, j + dy, k + dz);
                            if a < 0 || b < 0 || c < 0 || a > dims[0] || b > dims[1] || c > dims[2]
                            {
                                continue;
                            }
                            let s1 = d(a, b, c);
                            if (s0 < 0.0) == (s1 < 0.0) {
                                continue;
                            }
                            let t = s0 / (s0 - s1);
                            let len = ((dx * dx + dy * dy + dz * dz) as f32).sqrt() * cell;
                            if t * len < 0.30 * cell {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

/// Exact volume of `{x + y + z <= s}` inside `[0, l]^3`.
fn tilted_cube_volume(l: f64, s: f64) -> f64 {
    let c = |x: f64| x.max(0.0).powi(3);
    (c(s) - 3.0 * c(s - l) + 3.0 * c(s - 2.0 * l) - c(s - 3.0 * l)) / 6.0
}

// ---------------------------------------------------------------------------
// generate_marching_tets: exact polytope volumes, orientation, surface placement
// ---------------------------------------------------------------------------

/// Axis-aligned half-space `x <= s` through the middle of a cell column: the clipped region is
/// the exact box (s - x0) * Ly * Lz and every element is positively oriented.
#[test]
fn marching_tets_axis_half_space_volume_is_exact_and_elements_are_positive() {
    let (cell, n) = (0.5f32, 4);
    for s in [0.25f32, 0.75, 1.25, 1.75] {
        let sdf = plane([1.0, 0.0, 0.0], s);
        let max = [n as f32 * cell; 3];
        assert!(warp_free(&sdf, [0.0; 3], cell, [n, n, n]));
        let m = generate_marching_tets(&sdf, [0.0; 3], max, cell);
        let want = f64::from(s) * 2.0 * 2.0;
        assert!(
            (volume(&m) - want).abs() < 1e-5 * want,
            "s {s}: {} vs {want}",
            volume(&m)
        );
        for t in &m.tets {
            assert!(signed_vol6(&m, t.vertices) > 0.0, "inverted element");
        }
        // surface vertices are exact linear interpolations: on a linear SDF they lie ON the plane
        let on_plane = m
            .vertices
            .iter()
            .filter(|v| (v[0] - s).abs() < 1e-6)
            .count();
        assert!(
            on_plane >= (n as usize + 1) * (n as usize + 1),
            "s {s}: {on_plane} plane vertices"
        );
        for v in &m.vertices {
            assert!(v[0] <= s + 1e-6, "vertex outside the region: {v:?}");
        }
    }
}

/// Tilted half-spaces exercise the 1-, 2- and 3-vertices-inside clip cases.
#[test]
fn marching_tets_tilted_half_space_volumes_match_polytope_formulas() {
    let (cell, n) = (0.5f32, 4);
    let l = f64::from(cell) * f64::from(n);
    let max = [n as f32 * cell; 3];
    let sdf_xy = |s: f32| plane([1.0, 1.0, 0.0], s / std::f32::consts::SQRT_2);
    // x + y <= s  : area s^2/2 (s <= L), L^2 - (2L-s)^2/2 (L < s <= 2L), times Lz
    let mut checked = 0;
    for s in (1..400).map(|i| i as f32 * 0.01) {
        let sdf = sdf_xy(s);
        if !warp_free(&sdf, [0.0; 3], cell, [n, n, n]) {
            continue;
        }
        let m = generate_marching_tets(&sdf, [0.0; 3], max, cell);
        let sd = f64::from(s);
        let area = if sd <= l {
            sd * sd / 2.0
        } else {
            l * l - (2.0 * l - sd).powi(2) / 2.0
        };
        let want = area * l;
        assert!(
            (volume(&m) - want).abs() < 2e-5 * want,
            "xy s {s}: {} vs {want}",
            volume(&m)
        );
        for t in &m.tets {
            assert!(signed_vol6(&m, t.vertices) > 0.0);
        }
        checked += 1;
    }
    assert!(checked >= 2, "too few warp-free xy planes ({checked})");
    // x + y + z <= s : cube cut by an oblique plane (all three clip cases)
    let sdf_xyz = |s: f32| plane([1.0, 1.0, 1.0], s / 3.0f32.sqrt());
    let mut checked = 0;
    for s in (1..590).map(|i| i as f32 * 0.01) {
        let sdf = sdf_xyz(s);
        if !warp_free(&sdf, [0.0; 3], cell, [n, n, n]) {
            continue;
        }
        let m = generate_marching_tets(&sdf, [0.0; 3], max, cell);
        let want = tilted_cube_volume(l, f64::from(s));
        assert!(
            (volume(&m) - want).abs() < 2e-5 * want.max(1.0),
            "xyz s {s}: {} vs {want}",
            volume(&m)
        );
        for t in &m.tets {
            assert!(signed_vol6(&m, t.vertices) > 0.0);
        }
        checked += 1;
    }
    assert!(checked >= 4, "too few warp-free xyz planes ({checked})");
}

/// Closed surface: the boundary of a meshed ball is a 2-manifold without boundary
/// (every boundary edge shared by exactly two boundary faces) of Euler characteristic 2.
#[test]
fn marching_tets_ball_boundary_is_a_closed_sphere() {
    let m = generate_marching_tets(&ball(1.5), [-2.0; 3], [2.0; 3], 0.25);
    let faces = m.boundary_faces().expect("manifold");
    assert!(!faces.is_empty());
    let mut edge_uses: HashMap<(u32, u32), usize> = HashMap::new();
    let mut verts = HashSet::new();
    for f in &faces {
        for (a, b) in [(f[0], f[1]), (f[0], f[2]), (f[1], f[2])] {
            *edge_uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        verts.extend(f.iter().copied());
    }
    assert!(
        edge_uses.values().all(|&c| c == 2),
        "open or non-manifold boundary"
    );
    let chi = verts.len() as i64 - edge_uses.len() as i64 + faces.len() as i64;
    assert_eq!(
        chi,
        2,
        "V {} E {} F {}",
        verts.len(),
        edge_uses.len(),
        faces.len()
    );
    // volume within the discretisation + warp budget of the exact ball volume
    let exact = 4.0 / 3.0 * std::f64::consts::PI * 1.5f64.powi(3);
    assert!(
        (volume(&m) - exact).abs() / exact < 0.04,
        "{} vs {exact}",
        volume(&m)
    );
    for t in &m.tets {
        assert!(signed_vol6(&m, t.vertices) > 0.0);
    }
}

/// Doc: "a cube is kept only when all eight corners are inside": the tet count of `generate`
/// is five times the number of such cubes, recomputed here by brute force, and the volume is
/// that many cubes exactly. Elements are positively oriented.
#[test]
fn generate_counts_match_brute_force_occupancy_and_elements_are_positive() {
    let (cell, n) = (0.5f32, 8);
    let min = [-2.0f32; 3];
    let sdf = ball(1.6);
    let inside = |i: i32, j: i32, k: i32| {
        sdf.distance(
            min[0] + i as f32 * cell,
            min[1] + j as f32 * cell,
            min[2] + k as f32 * cell,
        ) <= 0.0
    };
    let mut cubes = 0;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if (0..8).all(|c| inside(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))) {
                    cubes += 1;
                }
            }
        }
    }
    assert!(cubes > 0);
    let m = generate(&sdf, min, [2.0; 3], cell);
    assert_eq!(m.tet_count(), 5 * cubes);
    let want = f64::from(cubes as i32) * 0.125;
    assert!(
        (volume(&m) - want).abs() < 1e-5 * want,
        "{} vs {want}",
        volume(&m)
    );
    for t in &m.tets {
        assert!(signed_vol6(&m, t.vertices) > 0.0);
    }
    // faces: at most 2 uses, boundary faces strictly fewer than all faces
    assert!(face_uses(&m).values().all(|&c| c <= 2));
}

/// Marching tets contains the interior-only mesh for a convex shape away from degenerate
/// (exactly-on-lattice) boundaries: volume(marching) >= volume(generate).
#[test]
fn marching_volume_contains_the_staircase_volume_for_a_ball() {
    let sdf = ball(1.55);
    let a = generate(&sdf, [-2.0; 3], [2.0; 3], 0.25);
    let b = generate_marching_tets(&sdf, [-2.0; 3], [2.0; 3], 0.25);
    assert!(
        volume(&b) >= volume(&a) - 1e-9,
        "{} vs {}",
        volume(&b),
        volume(&a)
    );
}

/// `generate` / `generate_marching_tets` walk `(max - min) / cell` cells. The quotient is computed
/// in f32 and truncated, so a box whose width is an exact multiple of the cell loses its last
/// layer when the quotient comes out one ulp below the integer: 1.3 / 0.1 = 12.999999.
#[test]
// AUD-A-S2W3-010
fn lattice_cell_count_is_robust_to_f32_rounding_of_the_quotient() {
    let sdf = everywhere_inside();
    // (max extent, cell, expected cells)
    for (len, cell, n) in [(1.3f32, 0.1f32, 13usize), (0.9, 0.3, 3), (0.3, 0.1, 3)] {
        let m = generate(&sdf, [0.0; 3], [len, cell, cell], cell);
        assert_eq!(m.tet_count(), 5 * n, "generate len {len} cell {cell}");
        let m2 = generate_marching_tets(&everywhere_inside(), [0.0; 3], [len, cell, cell], cell);
        assert_eq!(volume(&m2).round(), 0.0);
        let expect = f64::from(cell).powi(3) * n as f64;
        assert!(
            (volume(&m2) - expect).abs() < 1e-3 * expect,
            "marching len {len}: {} vs {expect}",
            volume(&m2)
        );
    }
}

/// The partial last cell is dropped, silently: an AABB 2.2 wide at cell 0.5 meshes 4 cells (2.0).
/// This is the documented floor behaviour of the lattice (pinned, not a defect by itself).
#[test]
fn a_partial_last_cell_is_dropped_not_meshed() {
    let m = generate(&everywhere_inside(), [0.0; 3], [2.2, 0.5, 0.5], 0.5);
    assert_eq!(m.tet_count(), 5 * 4);
    let m = generate(&everywhere_inside(), [0.0; 3], [0.2, 0.2, 0.2], 0.5);
    assert_eq!(
        m.tet_count(),
        5,
        "an AABB smaller than a cell still gets one cube"
    );
}

// ---------------------------------------------------------------------------
// try_refine_conforming / try_refine_marked
// ---------------------------------------------------------------------------

fn block() -> SdfTetMesh {
    generate(&everywhere_inside(), [0.0; 3], [1.5, 1.0, 1.0], 0.5)
}

fn boundary_area(m: &SdfTetMesh) -> f64 {
    let faces = m.boundary_faces().expect("manifold");
    faces
        .iter()
        .map(|f| {
            let p = f.map(|v| m.vertices[v as usize].map(f64::from));
            let (a, b) = (
                [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]],
                [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]],
            );
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
        })
        .sum()
}

/// Refinement bisects edges: afterwards no edge exceeds the threshold, the region (volume and
/// boundary area) is unchanged, old vertices keep their index and position (append only), every
/// face is used at most twice, and the minimal pass budget is exactly the returned count.
#[test]
fn conforming_refinement_bounds_edges_preserves_the_region_and_has_a_minimal_budget() {
    let original = block();
    let (v0, a0) = (volume_abs(&original), boundary_area(&original));
    let limit = 0.45f32;
    assert!(original.max_edge_length() > limit);
    let mut m = original.clone();
    let passes = m.try_refine_conforming(limit, 50).expect("finishes");
    assert!(passes >= 1);
    assert!(
        m.max_edge_length() <= limit + 1e-6,
        "{}",
        m.max_edge_length()
    );
    assert!((volume_abs(&m) - v0).abs() < 1e-6 * v0);
    assert!((boundary_area(&m) - a0).abs() < 1e-5 * a0);
    assert_eq!(
        &m.vertices[..original.vertices.len()],
        &original.vertices[..],
        "append only"
    );
    assert!(face_uses(&m).values().all(|&c| c <= 2));
    // idempotent: nothing left to do, unchanged
    let again = m.clone();
    assert_eq!(m.try_refine_conforming(limit, 50), Ok(0));
    assert_eq!(m, again);
    // minimal budget == returned pass count
    let mut tight = original.clone();
    assert_eq!(tight.try_refine_conforming(limit, passes), Ok(passes));
    let mut short = original.clone();
    match short.try_refine_conforming(limit, passes - 1) {
        Err(RefineError::Unfinished {
            passes: p,
            tets_left,
        }) => {
            assert_eq!(p, passes - 1);
            assert!(tets_left > 0);
        }
        other => panic!("budget {} should be short: {other:?}", passes - 1),
    }
    // a roomy budget returns the same count (the extra pass that finds nothing is not counted)
    let mut roomy = original.clone();
    assert_eq!(roomy.try_refine_conforming(limit, passes + 10), Ok(passes));
    assert_eq!(roomy, tight);
}

#[test]
fn conforming_refinement_rejects_nonpositive_threshold() {
    for bad in [0.0f32, -1.0] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut m = block();
            let _ = m.try_refine_conforming(bad, 3);
        }));
        assert!(r.is_err(), "threshold {bad}");
    }
}

/// Marked refinement: no marks -> Ok(0) and untouched; marks with a zero budget -> Unfinished
/// with the marked count; the longest edge of a marked element is bisected; region preserved;
/// every face used at most twice afterwards (no hanging node).
#[test]
fn marked_refinement_contracts() {
    let original = block();
    let none = vec![false; original.tet_count()];
    let mut m = original.clone();
    assert_eq!(m.try_refine_marked(&none, 5), Ok(0));
    assert_eq!(m, original);

    let mut marks = vec![false; original.tet_count()];
    marks[0] = true;
    marks[7] = true;
    let mut m = original.clone();
    assert_eq!(
        m.try_refine_marked(&marks, 0),
        Err(RefineError::Unfinished {
            passes: 0,
            tets_left: 2
        })
    );
    assert_eq!(m, original, "a refused budget leaves the mesh untouched");

    let mut m = original.clone();
    let passes = m.try_refine_marked(&marks, 40).expect("finishes");
    assert!(passes >= 1);
    assert!(
        m.tet_count() >= original.tet_count() + 2,
        "each marked element splits in two at least"
    );
    assert!((volume_abs(&m) - volume_abs(&original)).abs() < 1e-6 * volume_abs(&original));
    assert!(face_uses(&m).values().all(|&c| c <= 2));
    assert_eq!(
        &m.vertices[..original.vertices.len()],
        &original.vertices[..]
    );
    // the midpoint of the marked element's longest edge exists as a new vertex
    let t = original.tets[0].vertices;
    let pos = |v: u32| original.vertices[v as usize];
    let mut best = (0.0f32, (0u32, 0u32));
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let (pa, pb) = (pos(t[a]), pos(t[b]));
        let l = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
        if l > best.0 + 1e-9 {
            best = (l, (t[a], t[b]));
        }
    }
    let (pa, pb) = (pos(best.1 .0), pos(best.1 .1));
    let mid = [
        0.5 * (pa[0] + pb[0]),
        0.5 * (pa[1] + pb[1]),
        0.5 * (pa[2] + pb[2]),
    ];
    assert!(
        m.vertices[original.vertices.len()..].contains(&mid),
        "no vertex at the midpoint {mid:?} of the longest edge"
    );
    // wrong mark count is refused with both numbers
    let mut m = original.clone();
    assert_eq!(
        m.try_refine_marked(&[true; 3], 5),
        Err(RefineError::MarkCountDoesNotMatch {
            marks: 3,
            tets: original.tet_count()
        })
    );
}

// ---------------------------------------------------------------------------
// boundary_faces / accessors
// ---------------------------------------------------------------------------

/// Faces come back with sorted vertex indices, in ascending order, once each; the box surface
/// of an n x m x l block has 4 (nm + ml + nl) triangles.
#[test]
fn boundary_faces_are_sorted_unique_and_match_the_box_surface_count() {
    let m = block(); // 3 x 2 x 2 cells
    let f = m.boundary_faces().expect("manifold");
    assert_eq!(f.len(), 4 * (3 * 2 + 2 * 2 + 3 * 2));
    for face in &f {
        assert!(face[0] < face[1] && face[1] < face[2], "{face:?}");
    }
    assert!(f.windows(2).all(|w| w[0] < w[1]), "ascending and unique");
    assert_eq!(m.vertex_count(), 4 * 3 * 3);
    assert_eq!(m.tet_count(), 5 * 3 * 2 * 2);
    assert_eq!(SdfTetMesh::default().max_edge_length(), 0.0);
    let diag = (0.5f32 * 0.5 * 2.0).sqrt();
    assert!(
        (m.max_edge_length() - diag).abs() < 1e-6,
        "{}",
        m.max_edge_length()
    );
}

/// `push_tet` documents that every emitted element is wound the same way ("what makes
/// `inverted == 0` a proposition worth asserting") and `Tetrahedron::vertices` says the vertex
/// order defines the orientation. The refiners push their two children straight onto the list:
/// `[a, mid, c, d]` / `[mid, b, c, d]` keep the parent's winding only when the split edge is
/// (v0, v1) or an even permutation of it; for the other edges half the children are inverted.
#[test]
// AUD-A-S2W3-012
fn refinement_keeps_every_element_positively_wound() {
    let mut m = block();
    assert!(m.tets.iter().all(|t| signed_vol6(&m, t.vertices) > 0.0));
    m.try_refine_conforming(0.45, 50).expect("finishes");
    let inverted = m
        .tets
        .iter()
        .filter(|t| signed_vol6(&m, t.vertices) < 0.0)
        .count();
    assert_eq!(
        inverted,
        0,
        "{inverted} of {} elements inverted",
        m.tet_count()
    );
    let mut m = block();
    let mut marks = vec![false; m.tet_count()];
    marks[0] = true;
    marks[7] = true;
    m.try_refine_marked(&marks, 40).expect("finishes");
    let inverted = m
        .tets
        .iter()
        .filter(|t| signed_vol6(&m, t.vertices) < 0.0)
        .count();
    assert_eq!(
        inverted,
        0,
        "marked: {inverted} of {} elements inverted",
        m.tet_count()
    );
}

/// Zero pass budget: `tets_left` is exactly the number of tetrahedra with an edge above the limit,
/// recounted here from the geometry.
#[test]
fn unfinished_reports_the_exact_number_of_tets_with_a_long_edge() {
    let mut m = block();
    let limit = 0.6f32; // face diagonals (0.707) exceed it, axis edges (0.5) do not
    let long = m
        .tets
        .iter()
        .filter(|t| {
            let v = t.vertices.map(|i| m.vertices[i as usize]);
            (0..4).any(|a| {
                (a + 1..4).any(|b| {
                    let d = [v[a][0] - v[b][0], v[a][1] - v[b][1], v[a][2] - v[b][2]];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() > limit
                })
            })
        })
        .count();
    assert!(long > 0);
    assert_eq!(
        m.try_refine_conforming(limit, 0),
        Err(RefineError::Unfinished {
            passes: 0,
            tets_left: long
        })
    );
}

/// Marked refinement has the same minimal-budget contract as the conforming one.
#[test]
fn marked_refinement_minimal_budget_equals_returned_passes() {
    let original = block();
    let mut marks = vec![false; original.tet_count()];
    marks[3] = true;
    let mut a = original.clone();
    let p = a.try_refine_marked(&marks, 40).expect("finishes");
    assert!(p >= 1);
    let mut roomy = original.clone();
    assert_eq!(roomy.try_refine_marked(&marks, p + 7), Ok(p));
    let mut tight = original.clone();
    assert_eq!(tight.try_refine_marked(&marks, p), Ok(p));
    let mut short = original.clone();
    assert!(
        matches!(
            short.try_refine_marked(&marks, p - 1),
            Err(RefineError::Unfinished { .. })
        ) || p == 1
    );
}

/// `max_edge_length` sees edges along every axis (here a 3-long edge along z only).
#[test]
fn max_edge_length_covers_each_axis() {
    use alice_physics::sdf_fem_mesh::Tetrahedron;
    for (i, want) in [(0usize, 3.0f32), (1, 3.0), (2, 3.0)] {
        let mut tip = [0.0f32; 3];
        tip[i] = 3.0;
        let m = SdfTetMesh {
            vertices: vec![[0.0, 0.0, 0.0], tip, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            tets: vec![Tetrahedron {
                vertices: [0, 1, 2, 3],
            }],
        };
        assert!((m.max_edge_length() - want).abs() < 1e-6, "axis {i}");
    }
}
