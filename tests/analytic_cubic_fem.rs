//! Oracles for `alice_physics::cubic_elastic_fem` — the twenty-node P3
//! tetrahedron.
//!
//! The headline property, and the one that separates this element from the P2
//! one: **a P3 element reproduces a cubic displacement field exactly**. Not
//! "converges to", not "to within a tolerance that shrinks under refinement" —
//! exactly, at every node, on any mesh, at any resolution. That is a statement
//! about the element's function space and it is either true or the assembly is
//! wrong.
//!
//! The field used is
//!
//! ```text
//! u = c·(y³ − 3yz²,  z³ − 3zx²,  x³ − 3xy²)
//! ```
//!
//! chosen so that **no body force is needed**. Each component is independent of
//! its own coordinate, so every normal strain and the trace vanish and
//! equilibrium reduces to `μ·Δ₂uᵢ` in the other two coordinates; `y³ − 3yz² =
//! Re((y + iz)³)` is harmonic there, so `div σ ≡ 0` identically. There is no load
//! vector to get wrong, and a failure is the element and not the quadrature of a
//! source term.
//!
//! ⚠️ **The source-free requirement is sharper here than it was for P2.** The
//! consistent load of a body force is `∫Nᵢ·f`, which for a cubic `Nᵢ` and a
//! *constant* `f` is degree 3 and well inside the degree-5 rule this element
//! uses — but a *linear* `f` would be degree 4, and a quadratic one degree 5, so
//! an oracle with a source term would be measuring how far up that ladder the
//! rule reaches rather than what the function space contains. A source-free field
//! has no ladder.
//!
//! ⚠️ **P2 fails this field by four orders of magnitude, and that is asserted
//! here rather than assumed** (`cubic_field_is_beyond_p2`). An oracle that only
//! ever sees the element it was written for cannot tell "correct" from
//! "vacuous" — and for *this* degree the trap is wide open, because
//! it has been measured that on a uniform
//! Kuhn lattice even **P1** reproduces any polynomial of degree ≤ 3 at the nodes.
//! A cubic field is exactly the degree where that blindness peaks, so every
//! exactness oracle below runs on a **perturbed** lattice.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeSet;

use alice_physics::cubic_elastic_fem::{shape_values, solve_cubic, CubicMesh};
use alice_physics::linear_elastic_fem::{solve, BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{solve_quadratic, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
/// Side of the cube, in mm.
///
/// A multiple of three at every resolution used below (`n` ∈ {1, 2}), so that an
/// *unperturbed* mesh places the edge nodes `(2a + b)/3` exactly. See
/// [`interior_node_positions_are_exact_only_on_a_lattice_of_multiples_of_three`].
const SIDE: f64 = 6.0;
/// Amplitude, so displacements land in the thousandths of a millimetre.
///
/// The cubic field reaches `|y³ − 3yz²| = 432` at the far corner of a 6 mm cube,
/// so `1e-5` keeps the solution near `4e-3` mm — the same order as the P2
/// oracle's, which keeps the conjugate gradient's floor in the same place.
const AMPLITUDE: f64 = 1.0e-5;

/// Interior-vertex displacement, as a fraction of `h`.
///
/// ⚠️ **Every exactness oracle below is run on a perturbed lattice, and for a
/// cubic field it is not optional.** On a uniform Kuhn lattice the P1 stiffness
/// reproduces a 7-point Laplacian stencil whose truncation error is proportional
/// to the *fourth* derivative of the solution, so P1 — never mind P2 — returns
/// any polynomial of degree ≤ 3 exactly at the nodes. `cubic_field_is_beyond_p2`
/// is what notices if this is ever set back to zero.
///
/// ⚠️ **The asymmetry is the point.** P3 exactness on a cubic field is a property
/// of the *function space*, so it survives any mesh. P1's and P2's agreement on a
/// uniform lattice is an artefact of the *lattice*, so it does not. Perturbing
/// separates them.
const JITTER: f64 = 0.25;

/// What "exactly" means here, in mm.
///
/// Not the arithmetic floor: the field is reproduced by the *element*, but it
/// still has to be found by the *conjugate gradient*, which stops at a relative
/// residual of `2⁻³⁰`. The residual is printed on every row so the two can be
/// told apart, and P2 misses this field by four orders of magnitude more, which
/// is what `cubic_field_is_beyond_p2` pins.
const EXACTNESS_BOUND: f64 = 1.0e-9;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid")
}

/// Raw two's-complement integer behind a `Fix128`; differences in it are ulps.
fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

fn config() -> SolverConfig {
    SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid")
}

// ---------------------------------------------------------------------------
// the manufactured fields
// ---------------------------------------------------------------------------

/// `u = c·(y³ − 3yz², z³ − 3zx², x³ − 3xy²)`.
///
/// Divergence-free, each component harmonic in the two coordinates it depends
/// on, so the manufactured stress is divergence-free and no body force is
/// needed. **In the P3 space and outside the P2 one.**
fn cubic_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE;
    [
        c * (y * y * y - 3.0 * y * z * z),
        c * (z * z * z - 3.0 * z * x * x),
        c * (x * x * x - 3.0 * x * y * y),
    ]
}

/// `u = c·(y² − z², z² − x², x² − y²)`. The P2 oracle's field; P3 contains P2 so
/// it must come back exactly here too.
fn quadratic_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE * 10.0;
    [
        c * (y * y - z * z),
        c * (z * z - x * x),
        c * (x * x - y * y),
    ]
}

/// A linear field, which all three elements must reproduce exactly.
fn linear_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE * 100.0;
    [
        c * (1.0 + 2.0 * x - y),
        c * (3.0 * y + z),
        c * (x + y + 4.0 * z),
    ]
}

// ---------------------------------------------------------------------------
// mesh
// ---------------------------------------------------------------------------

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Deterministic displacement in `[-1, 1]` for one lattice node and axis.
fn jitter_unit(i: usize, j: usize, k: usize, axis: usize) -> f64 {
    let mut h = 0x9E37_79B9_7F4A_7C15_u64;
    for v in [i as u64, j as u64, k as u64, axis as u64] {
        h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
    }
    ((h >> 32) as f64) / ((1u64 << 32) as f64) * 2.0 - 1.0
}

/// Kuhn 6-tet cube on `[0, n·h]³`, conforming at every `n`.
///
/// `jitter` displaces the **interior** vertices by that fraction of `h`; the
/// boundary, the topology and the element count are untouched. See [`JITTER`] for
/// why every oracle here needs it.
fn kuhn_cube(n: usize, h: f64, jitter: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let boundary = i == 0 || j == 0 || k == 0 || i == n || j == n || k == n;
                let mut p = [i as f64 * h, j as f64 * h, k as f64 * h];
                if !boundary && jitter != 0.0 {
                    for (axis, c) in p.iter_mut().enumerate() {
                        *c += jitter * h * jitter_unit(i, j, k, axis);
                    }
                }
                mesh.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
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
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(n, i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = node_index(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// Distinct edges and faces of a tetrahedral mesh, counted independently of the
/// element implementation.
fn count_edges_and_faces(mesh: &SdfTetMesh) -> (usize, usize) {
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut faces: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for tet in &mesh.tets {
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (p, q) = (tet.vertices[a], tet.vertices[b]);
                edges.insert(if p <= q { (p, q) } else { (q, p) });
            }
        }
        for skip in 0..4 {
            let mut tri: Vec<u32> = (0..4)
                .filter(|&i| i != skip)
                .map(|i| tet.vertices[i])
                .collect();
            tri.sort_unstable();
            faces.insert((tri[0], tri[1], tri[2]));
        }
    }
    (edges.len(), faces.len())
}

// ---------------------------------------------------------------------------
// the mesh the element needs
// ---------------------------------------------------------------------------

/// The node tables, before any solve depends on them.
///
/// Four properties, each of which a later oracle would otherwise be silently
/// resting on: corners keep their indices, the node count is exactly
/// `corners + 2·edges + faces` (so nothing is duplicated per element), the edge
/// pair does not depend on the order of the two arguments, and the two nodes of
/// an edge are distinct and placed at one and two thirds.
#[test]
fn the_cubic_mesh_adds_two_nodes_per_edge_and_one_per_face_and_keeps_the_corners() {
    for n in [1usize, 2] {
        let h = SIDE / n as f64;
        let mesh = kuhn_cube(n, h, JITTER);
        let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
        let (edges, faces) = count_edges_and_faces(&mesh);

        assert_eq!(
            c.corner_count(),
            mesh.vertex_count(),
            "n={n}: the corners are the vertices of the source mesh"
        );
        assert_eq!(
            c.edge_node_count(),
            2 * edges,
            "n={n}: a cubic element puts two nodes on every edge, and an edge shared by \
             several elements must be built once"
        );
        assert_eq!(
            c.face_node_count(),
            faces,
            "n={n}: one node per distinct face, counted from the element list independently"
        );
        assert_eq!(
            c.node_count(),
            mesh.vertex_count() + 2 * edges + faces,
            "n={n}: no node is unaccounted for"
        );

        for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
            let want = mesh.vertices[v as usize];
            let got = c.node_position(v).expect("in range");
            for axis in 0..3 {
                assert_eq!(
                    raw(got[axis]),
                    raw(Fix128::from_f32(want[axis])),
                    "n={n}: corner {v} must come back bit-identical to the source vertex"
                );
            }
        }

        // The pair is defined against the corner numbering, not against the call.
        for tet in &mesh.tets {
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let (p, q) = (tet.vertices[a], tet.vertices[b]);
                    let forward = c.edge_nodes(p, q).expect("edge present");
                    let backward = c.edge_nodes(q, p).expect("edge present");
                    assert_eq!(
                        forward, backward,
                        "n={n}: the edge pair must not depend on the argument order"
                    );
                    assert_ne!(
                        forward.0, forward.1,
                        "n={n}: the two nodes of an edge are distinct"
                    );
                    assert!(
                        c.face_node(tet.vertices[0], tet.vertices[1], tet.vertices[2])
                            .is_some(),
                        "n={n}: every face of every element has a node"
                    );
                }
            }
        }
        eprintln!(
            "[p3fem] n={n}: {} corners + {} edge nodes + {} face nodes = {} nodes, \
             {} elements",
            c.corner_count(),
            c.edge_node_count(),
            c.face_node_count(),
            c.node_count(),
            c.element_count()
        );
    }
}

/// ⚠️ The API contract for the one thing P2 did not have to think about.
///
/// The edge nodes are at `(2a + b)/3` and the face nodes at `(a + b + c)/3`, and
/// `1/3` is not representable in a binary fixed-point type. An inexact third is
/// **truncated and nothing in the solve notices**, because the element geometry
/// is built from the four corners — so without a way to ask, the degradation
/// would be invisible to a caller that needs to know where a node is.
///
/// `CubicMesh::interior_node_positions_are_exact` is that way to ask, and this
/// measures both of its answers together with the error each corresponds to: zero
/// ulp on a lattice of multiples of three, non-zero on a lattice of ones.
#[test]
fn interior_node_positions_are_exact_only_on_a_lattice_of_multiples_of_three() {
    for (h, want_exact) in [(3.0_f64, true), (1.0_f64, false)] {
        let mesh = kuhn_cube(1, h, 0.0);
        let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
        assert_eq!(
            c.interior_node_positions_are_exact(),
            want_exact,
            "h={h}: a lattice of multiples of three places (2a+b)/3 exactly and one of ones \
             does not"
        );

        // The error the flag corresponds to: recompute one edge node from its two
        // corners in exact integer arithmetic over the raw encoding.
        let mut worst = 0i128;
        for tet in &mesh.tets {
            for a in 0..4 {
                for b in 0..4 {
                    if a == b {
                        continue;
                    }
                    let (p, q) = (tet.vertices[a], tet.vertices[b]);
                    let (lo, hi) = c.edge_nodes(p.min(q), p.max(q)).expect("edge present");
                    let near = if p <= q { lo } else { hi };
                    let got = c.node_position(near).expect("in range");
                    let pa = c.node_position(p).expect("in range");
                    let pb = c.node_position(q).expect("in range");
                    for axis in 0..3 {
                        // 3·node − (2·a + b) is zero exactly when the third was.
                        let want = 2 * raw(pa[axis]) + raw(pb[axis]);
                        let diff = 3 * raw(got[axis]) - want;
                        if diff.abs() > worst {
                            worst = diff.abs();
                        }
                    }
                }
            }
        }
        eprintln!(
            "[p3fem] h={h}: interior positions exact = {}, worst 3·node − (2a+b) = {worst} raw",
            c.interior_node_positions_are_exact()
        );
        if want_exact {
            assert_eq!(
                worst, 0,
                "h={h}: the flag says exact, so the recomputation must agree to the bit"
            );
        } else {
            assert_ne!(
                worst, 0,
                "h={h}: the flag says inexact, so there must be an error to find — a zero \
                 here means the flag and the measurement disagree"
            );
            assert!(
                worst < 8,
                "h={h}: the error must be the truncation of one division ({worst} raw)"
            );
        }
    }
}

/// The interpolation itself, against the closed form, with no solver involved.
///
/// The twenty shape functions are a basis for the cubics on a tetrahedron, so
/// interpolating a cubic function from its twenty nodal values must return it
/// **at any point**, not only at the nodes. Checked on the reference tetrahedron
/// against the closed form at the quadrature points and the centroid.
///
/// ⚠️ This is the function-space statement without the Galerkin step, so a
/// failure here localises to `shape_values` and a failure in
/// `cubic_field_is_exact_on_cubic_elements` but not here localises to the
/// assembly or the solver.
#[test]
fn the_cubic_basis_interpolates_a_cubic_function_at_interior_points() {
    // A cubic in barycentric coordinates: f = 6λ₂³ − λ₁λ₂λ₃ + 2λ₀²λ₃.
    let f = |l: [f64; 4]| 6.0 * l[2].powi(3) - l[1] * l[2] * l[3] + 2.0 * l[0] * l[0] * l[3];

    // Nodal barycentric coordinates in element slot order: corners, then the two
    // nodes of each edge, then the face nodes.
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];
    let mut nodal = [[0.0f64; 4]; 20];
    for (i, slot) in nodal.iter_mut().take(4).enumerate() {
        slot[i] = 1.0;
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        nodal[4 + 2 * e][i] = 2.0 / 3.0;
        nodal[4 + 2 * e][j] = 1.0 / 3.0;
        nodal[5 + 2 * e][j] = 2.0 / 3.0;
        nodal[5 + 2 * e][i] = 1.0 / 3.0;
    }
    for (fi, &(i, j, k)) in FACES.iter().enumerate() {
        for axis in [i, j, k] {
            nodal[16 + fi][axis] = 1.0 / 3.0;
        }
    }

    let probes: [[f64; 4]; 4] = [
        [0.25, 0.25, 0.25, 0.25],
        [0.625, 0.125, 0.125, 0.125],
        [0.0625, 0.3125, 0.3125, 0.3125],
        [0.375, 0.375, 0.125, 0.125],
    ];
    let mut worst = 0.0f64;
    for probe in probes {
        let n = shape_values(&[fx(probe[0]), fx(probe[1]), fx(probe[2]), fx(probe[3])]);
        let mut got = 0.0f64;
        for (value, point) in n.iter().zip(nodal.iter()) {
            got += value.to_f64() * f(*point);
        }
        let want = f(probe);
        worst = worst.max((got - want).abs());
        eprintln!("[p3fem] interpolation at {probe:?}: got {got:.12}, want {want:.12}");
    }
    assert!(
        worst < 1.0e-12,
        "the twenty cubic shape functions are a basis for the cubics, so interpolating a \
         cubic must return it at interior points too; worst error {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// exactness
// ---------------------------------------------------------------------------

/// Prescribe `field` on every boundary node, solve with P3, and return the
/// largest error over the interior nodes (mm).
fn interior_error_p3(n: usize, field: fn([f64; 3]) -> [f64; 3]) -> (f64, u32, f64, usize) {
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    let eps = h * 1e-4;

    let mut bc = BoundaryConditions::new();
    let mut interior = 0usize;
    let mut positions = Vec::with_capacity(c.node_count());
    for node in 0..u32::try_from(c.node_count()).expect("fits") {
        let q = c.node_position(node).expect("in range");
        let p = [q[0].to_f64(), q[1].to_f64(), q[2].to_f64()];
        positions.push(p);
        if p.iter().any(|&v| v < eps || v > SIDE - eps) {
            let u = field(p);
            bc.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior += 1;
        }
    }

    let out = solve_cubic(&c, &pla(), &bc, &config()).expect("well posed");
    let mut worst = 0.0f64;
    for (p, got) in positions.iter().zip(out.displacements.iter()) {
        if p.iter().any(|&v| v < eps || v > SIDE - eps) {
            continue;
        }
        for (g, want) in got.iter().zip(field(*p).iter()) {
            worst = worst.max((g.to_f64() - want).abs());
        }
    }
    (
        worst,
        out.iterations,
        out.relative_residual.to_f64(),
        interior,
    )
}

/// P3 contains P1, so a linear field has to come back exactly.
#[test]
fn linear_field_is_exact_on_cubic_elements() {
    for n in [1usize, 2] {
        let (worst, iters, residual, interior) = interior_error_p3(n, linear_field);
        eprintln!(
            "[p3fem] n={n} linear: {interior} interior nodes, worst interior error \
             {worst:.3e} mm, {iters} iterations, relative residual {residual:.3e}"
        );
        assert!(interior > 0, "n={n}: the study needs interior nodes");
        assert!(
            worst < EXACTNESS_BOUND,
            "n={n}: a linear field lies in the P3 space, so it must be reproduced to the \
             solver's floor; got {worst:.3e} mm against a bound of {EXACTNESS_BOUND:.0e}"
        );
    }
}

/// P3 contains P2, so the P2 oracle's field has to come back exactly as well.
#[test]
fn quadratic_field_is_exact_on_cubic_elements() {
    for n in [1usize, 2] {
        let (worst, iters, residual, interior) = interior_error_p3(n, quadratic_field);
        eprintln!(
            "[p3fem] n={n} quadratic: {interior} interior nodes, worst interior error \
             {worst:.3e} mm, {iters} iterations, relative residual {residual:.3e}"
        );
        assert!(interior > 0, "n={n}: the study needs interior nodes");
        assert!(
            worst < EXACTNESS_BOUND,
            "n={n}: a quadratic field lies in the P3 space, so it must be reproduced to the \
             solver's floor; got {worst:.3e} mm against a bound of {EXACTNESS_BOUND:.0e}"
        );
    }
}

/// The property that is the point of the element.
#[test]
fn cubic_field_is_exact_on_cubic_elements() {
    for n in [1usize, 2] {
        let (worst, iters, residual, interior) = interior_error_p3(n, cubic_field);
        eprintln!(
            "[p3fem] n={n} cubic: {interior} interior nodes, worst interior error \
             {worst:.3e} mm, {iters} iterations, relative residual {residual:.3e}"
        );
        assert!(interior > 0, "n={n}: the study needs interior nodes");
        assert!(
            worst < EXACTNESS_BOUND,
            "n={n}: a cubic field lies in the P3 space and the manufactured stress is \
             divergence-free, so the solution must be reproduced to the solver's floor; got \
             {worst:.3e} mm against a bound of {EXACTNESS_BOUND:.0e}. This is the element's \
             defining property, not a tolerance to be relaxed"
        );
    }
}

/// ⚠️ The same field through the P2 and P1 solvers, so the test above is known to
/// have teeth.
///
/// Without this, `cubic_field_is_exact_on_cubic_elements` could be passing
/// because the field is easy rather than because the element is cubic. For this
/// degree that is not a hypothetical: on a **uniform** Kuhn lattice both P1 and
/// P2 return a cubic field exactly at the nodes, so the entire difference
/// between the three elements here rests on [`JITTER`] being non-zero.
#[test]
fn cubic_field_is_beyond_p2() {
    let n = 2usize;
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let eps = h * 1e-4;

    // --- P2 on the same mesh and the same field -----------------------------
    let q = QuadraticMesh::from_tet_mesh(&mesh).expect("well formed");
    let mut bc2 = BoundaryConditions::new();
    let mut interior2 = Vec::new();
    for node in 0..u32::try_from(q.node_count()).expect("fits") {
        let f = q.node_position(node).expect("in range");
        let p = [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()];
        if p.iter().any(|&v| v < eps || v > SIDE - eps) {
            let u = cubic_field(p);
            bc2.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior2.push((node as usize, p));
        }
    }
    let out2 = solve_quadratic(&q, &pla(), &bc2, &config()).expect("well posed");
    let mut worst2 = 0.0f64;
    let mut scale = 0.0f64;
    for &(node, p) in &interior2 {
        for (g, want) in out2.displacements[node].iter().zip(cubic_field(p).iter()) {
            worst2 = worst2.max((g.to_f64() - want).abs());
            scale = scale.max(want.abs());
        }
    }

    // --- P1 on the same mesh and the same field -----------------------------
    let mut bc1 = BoundaryConditions::new();
    let mut interior1 = Vec::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let c = mesh.vertices[v as usize];
        let p = [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])];
        if p.iter().any(|&x| x < eps || x > SIDE - eps) {
            let u = cubic_field(p);
            bc1.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior1.push((v as usize, p));
        }
    }
    let out1 = solve(&mesh, &pla(), &bc1, &config()).expect("well posed");
    let mut worst1 = 0.0f64;
    for &(v, p) in &interior1 {
        for (g, want) in out1.displacements[v].iter().zip(cubic_field(p).iter()) {
            worst1 = worst1.max((g.to_f64() - want).abs());
        }
    }

    let (worst3, _, _, _) = interior_error_p3(n, cubic_field);
    eprintln!(
        "[p3fem] the same cubic field, same mesh: P1 {worst1:.3e} mm ({:.2}% of the field), \
         P2 {worst2:.3e} mm ({:.2}%), P3 {worst3:.3e} mm",
        100.0 * worst1 / scale,
        100.0 * worst2 / scale
    );

    assert!(
        worst2 > 1.0e3 * EXACTNESS_BOUND,
        "P2 must *not* reproduce this field — if it does, the field is not exercising the \
         cubic part of the space and cubic_field_is_exact_on_cubic_elements is vacuous. Got \
         {worst2:.3e} mm. ⚠️ On a *uniform* lattice this is exactly what happens, for P1 as \
         well as P2, so check that JITTER is still non-zero before blaming the element"
    );
    assert!(
        worst1 > worst2,
        "the three elements must order: P1 ({worst1:.3e}) worse than P2 ({worst2:.3e}) worse \
         than P3 ({worst3:.3e}). An inversion means the comparison is measuring the solver \
         and not the function space"
    );
    assert!(
        worst3 < EXACTNESS_BOUND,
        "P3 must reproduce it: got {worst3:.3e} mm"
    );
}

/// Constraining only the corners of a clamped face leaves **seven of its ten
/// nodes** free, which is the mistake this element makes easiest.
///
/// Reported as a measured number rather than a warning in prose: the point is
/// that the solve *succeeds* and returns a different answer, so nothing flags it
/// for the caller. A clamped triangular face of a P3 mesh carries three corners,
/// six edge nodes and one face node.
#[test]
fn constraining_corners_only_is_a_different_problem() {
    let n = 1usize;
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    let eps = h * 1e-4;

    let mut full = BoundaryConditions::new();
    let mut corners_only = BoundaryConditions::new();
    let mut boundary_nodes = 0usize;
    let mut boundary_corners = 0usize;
    for node in 0..u32::try_from(c.node_count()).expect("fits") {
        let f = c.node_position(node).expect("in range");
        let p = [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()];
        if !p.iter().any(|&v| v < eps || v > SIDE - eps) {
            continue;
        }
        let u = cubic_field(p);
        let value = [fx(u[0]), fx(u[1]), fx(u[2])];
        full.prescribe_all(node, value);
        boundary_nodes += 1;
        if (node as usize) < c.corner_count() {
            corners_only.prescribe_all(node, value);
            boundary_corners += 1;
        }
    }

    let a = solve_cubic(&c, &pla(), &full, &config()).expect("well posed");
    let b = solve_cubic(&c, &pla(), &corners_only, &config()).expect("also well posed");

    let mut worst = 0.0f64;
    for (x, y) in a.displacements.iter().zip(b.displacements.iter()) {
        for (p, q) in x.iter().zip(y.iter()) {
            worst = worst.max((p.to_f64() - q.to_f64()).abs());
        }
    }
    eprintln!(
        "[p3fem] boundary nodes {boundary_nodes}, of which corners {boundary_corners}: \
         leaving the other {} free changes the answer by {worst:.3e} mm and both solves \
         succeed",
        boundary_nodes - boundary_corners
    );
    assert!(
        boundary_nodes > boundary_corners,
        "the study needs edge and face nodes on the boundary"
    );
    assert!(
        worst > EXACTNESS_BOUND,
        "constraining only the corners must give a *different* answer; it gave the same one \
         to {worst:.3e} mm, which would mean the edge and face degrees of freedom on a \
         clamped face do not matter"
    );
}
