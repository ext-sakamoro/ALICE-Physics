//! Oracles for `alice_physics::quadratic_elastic_fem` — the ten-node P2
//! tetrahedron.
//!
//! The headline property, and the one that separates this element from the P1
//! one already in the crate: **a P2 element reproduces a quadratic displacement
//! field exactly**. Not "converges to", not "to within a tolerance that shrinks
//! under refinement" — exactly, at every node and at every point inside every
//! element, on any mesh, at any resolution. That is a statement about the
//! element's function space and it is either true or the assembly is wrong.
//!
//! The field used is
//!
//! ```text
//! u = c·(y² − z²,  z² − x²,  x² − y²)
//! ```
//!
//! chosen so that **no body force is needed**. Each component is independent of
//! its own coordinate, so every normal strain and the trace vanish and
//! equilibrium reduces to `μ·Δ₂uᵢ` in the other two coordinates; `y² − z² =
//! Re((y + iz)²)` is harmonic there, so `div σ ≡ 0` identically. There is no
//! load vector to get wrong, and a failure is the element and not the
//! quadrature of a source term.
//!
//! ⚠️ **P1 fails this by a wide margin, and that is asserted here rather than
//! assumed.** `quadratic_field_is_beyond_p1` runs the same field through
//! `linear_elastic_fem::solve` on the same mesh and requires a large error. An
//! oracle that only ever sees the element it was written for cannot tell
//! "correct" from "vacuous".
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{solve, BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{solve_quadratic, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
const SIDE: f64 = 4.0;
/// Amplitude, so displacements land in the hundredths of a millimetre.
const AMPLITUDE: f64 = 1.0e-3;

/// Interior-vertex displacement, as a fraction of `h`.
///
/// ⚠️ **Every exactness oracle below is run on a perturbed lattice, and it has
/// to be.** On a *uniform* Kuhn lattice the P1 stiffness reproduces a 7-point
/// Laplacian stencil whose truncation error is proportional to the fourth
/// derivative of the solution, so **P1 reproduces any polynomial of degree ≤ 3
/// exactly at the nodes**. Measured here before the perturbation was added: P1
/// returned the quadratic field below to **1.475e-17 mm** — which made
/// `quadratic_field_is_exact_on_quadratic_elements` pass for a reason that had
/// nothing to do with the element, and `quadratic_field_is_beyond_p1` caught
/// it.
///
/// ⚠️ **The asymmetry is the point.** P2 exactness on a quadratic field is a
/// property of the *function space*, so it survives any mesh. P1's was an
/// artefact of the *lattice*, so it does not. Perturbing separates them.
/// (`tests/p2_oracle_design.rs` measures the same separation from the
/// convergence-rate side.)
const JITTER: f64 = 0.25;

/// What "exactly" means here, in mm.
///
/// Not the arithmetic floor: the field is reproduced by the *element*, but it
/// still has to be found by the *conjugate gradient*, which stops at a relative
/// residual of `2⁻³⁰`. The measured interior errors track that and not the
/// element — 0 at `n = 1`, 6.9e-18 at `n = 2`, 4.9e-12 at `n = 3` — so the
/// bound sits above the solver's floor and the residual is printed on every row
/// so the two can be told apart. P1 misses this field by ten orders of
/// magnitude more, which is what `quadratic_field_is_beyond_p1` pins.
const EXACTNESS_BOUND: f64 = 1.0e-10;

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

// ---------------------------------------------------------------------------
// the manufactured fields
// ---------------------------------------------------------------------------

/// `u = c·(y² − z², z² − x², x² − y²)`. Divergence-free stress, no body force.
fn quadratic_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE;
    [
        c * (y * y - z * z),
        c * (z * z - x * x),
        c * (x * x - y * y),
    ]
}

/// A linear field, which both elements must reproduce exactly.
fn linear_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE;
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
/// boundary, the topology and the element count are untouched. See
/// [`JITTER`] for why every oracle here needs it.
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

// ---------------------------------------------------------------------------
// the mesh the element needs
// ---------------------------------------------------------------------------

/// The edge table, before any solve depends on it.
///
/// Three properties, each of which a later oracle would otherwise be silently
/// resting on: corners keep their indices, every edge is shared rather than
/// duplicated per element, and the midpoints are exact.
#[test]
fn the_quadratic_mesh_adds_one_node_per_edge_and_keeps_the_corners() {
    for n in [1usize, 2, 3] {
        let h = SIDE / n as f64;
        let mesh = kuhn_cube(n, h, JITTER);
        let q = QuadraticMesh::from_tet_mesh(&mesh).expect("well formed");

        assert!(
            q.corner_count() == mesh.vertex_count(),
            "n={n}: corners must be the source vertices; {} vs {}",
            q.corner_count(),
            mesh.vertex_count()
        );
        assert!(
            q.element_count() == mesh.tet_count(),
            "n={n}: one element per tetrahedron"
        );

        // Corners keep their positions bit for bit, so boundary data written
        // against the tetrahedral mesh addresses the same points.
        for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
            let got = q.node_position(v).expect("in range");
            let want = mesh.vertices[v as usize];
            assert!(
                got[0] == Fix128::from_f32(want[0])
                    && got[1] == Fix128::from_f32(want[1])
                    && got[2] == Fix128::from_f32(want[2]),
                "n={n}: corner {v} moved"
            );
        }

        // Every element's six edge slots resolve through the shared table, so
        // an edge between two elements is one node and not two. Counted rather
        // than assumed: a per-element table would still pass every test that
        // only looks at one element.
        let mut distinct = std::collections::BTreeSet::new();
        for e in 0..q.element_count() {
            let nodes = q.element_nodes(e).expect("in range");
            for node in nodes.iter().skip(4) {
                distinct.insert(*node);
            }
        }
        assert!(
            distinct.len() == q.edge_count(),
            "n={n}: the element edge slots must name exactly the edge nodes; \
             {} distinct against {} edge nodes",
            distinct.len(),
            q.edge_count()
        );
        assert!(
            q.node_count() == q.corner_count() + q.edge_count(),
            "n={n}: node count must be corners plus edges"
        );

        // Midpoints are exact: `1/2` is a power of two, so the average of two
        // corner positions is representable with no rounding at all. This is
        // what makes an edge node reached from either side bit-identical.
        for e in 0..q.element_count() {
            let nodes = q.element_nodes(e).expect("in range");
            const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
            for (slot, &(i, j)) in EDGES.iter().enumerate() {
                let a = q.node_position(nodes[i]).expect("in range");
                let b = q.node_position(nodes[j]).expect("in range");
                let m = q.node_position(nodes[4 + slot]).expect("in range");
                for axis in 0..3 {
                    let want = raw(a[axis]) + raw(b[axis]);
                    assert!(
                        want % 2 == 0 && raw(m[axis]) * 2 == want,
                        "n={n} element {e} edge {slot} axis {axis}: midpoint is not exact"
                    );
                }
                assert!(
                    q.edge_node(nodes[i], nodes[j]) == Some(nodes[4 + slot])
                        && q.edge_node(nodes[j], nodes[i]) == Some(nodes[4 + slot]),
                    "n={n}: edge_node must be symmetric and agree with the element table"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// exactness
// ---------------------------------------------------------------------------

/// Prescribe `field` on every boundary node, solve, and return the largest
/// error over the interior nodes (mm), together with the iteration count.
fn interior_error(n: usize, field: fn([f64; 3]) -> [f64; 3]) -> (f64, u32, f64, usize) {
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let q = QuadraticMesh::from_tet_mesh(&mesh).expect("well formed");
    let eps = h * 1e-4;

    let mut bc = BoundaryConditions::new();
    let mut interior = 0usize;
    let mut positions = Vec::with_capacity(q.node_count());
    for node in 0..u32::try_from(q.node_count()).expect("fits") {
        let f = q.node_position(node).expect("in range");
        let p = [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()];
        positions.push(p);
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            let u = field(p);
            bc.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior += 1;
        }
    }

    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid");
    let out = solve_quadratic(&q, &pla(), &bc, &config).expect("well posed");

    let mut worst = 0.0_f64;
    for (p, got) in positions.iter().zip(out.displacements.iter()) {
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
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

/// P2 contains P1, so a linear field has to come back exactly.
#[test]
fn linear_field_is_exact_on_quadratic_elements() {
    for n in [1usize, 2, 3] {
        let (worst, iters, residual, interior) = interior_error(n, linear_field);
        eprintln!(
            "[p2fem] n={n} linear: {interior} interior nodes, worst interior error {worst:.3e} mm, \
             {iters} iterations, relative residual {residual:.3e}"
        );
        assert!(
            interior > 0,
            "n={n}: the study needs interior nodes to say anything"
        );
        assert!(
            worst < EXACTNESS_BOUND,
            "n={n}: a linear field lies in the P2 space, so it must be reproduced to the \
             solver's floor; got {worst:.3e} mm against a bound of {EXACTNESS_BOUND:.0e}"
        );
    }
}

/// The property that is the point of the element.
#[test]
fn quadratic_field_is_exact_on_quadratic_elements() {
    for n in [1usize, 2, 3] {
        let (worst, iters, residual, interior) = interior_error(n, quadratic_field);
        eprintln!(
            "[p2fem] n={n} quadratic: {interior} interior nodes, worst interior error \
             {worst:.3e} mm, {iters} iterations, relative residual {residual:.3e}"
        );
        assert!(interior > 0, "n={n}: the study needs interior nodes");
        assert!(
            worst < EXACTNESS_BOUND,
            "n={n}: a quadratic field lies in the P2 space and the manufactured stress is \
             divergence-free, so the solution must be reproduced to the solver's floor; got \
             {worst:.3e} mm against a bound of {EXACTNESS_BOUND:.0e}. This is the element's \
             defining property, not a tolerance to be relaxed"
        );
    }
}

/// ⚠️ The same field through the P1 solver, so the test above is known to have
/// teeth.
///
/// Without this, `quadratic_field_is_exact_on_quadratic_elements` could be
/// passing because the field is easy rather than because the element is
/// quadratic — the failure mode where an oracle never sees anything it should
/// reject.
#[test]
fn quadratic_field_is_beyond_p1() {
    let n = 3usize;
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let eps = h * 1e-4;

    let mut bc = BoundaryConditions::new();
    let mut interior = Vec::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let q = mesh.vertices[v as usize];
        let p = [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])];
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            let u = quadratic_field(p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior.push((v as usize, p));
        }
    }
    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let out = solve(&mesh, &pla(), &bc, &config).expect("well posed");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for &(v, p) in &interior {
        for (g, want) in out.displacements[v].iter().zip(quadratic_field(p).iter()) {
            worst = worst.max((g.to_f64() - want).abs());
            scale = scale.max(want.abs());
        }
    }
    eprintln!(
        "[p2fem] P1 on the same quadratic field: worst interior error {worst:.3e} mm \
         ({:.2}% of the field), {} interior nodes",
        100.0 * worst / scale,
        interior.len()
    );
    assert!(
        worst > 1.0e3 * EXACTNESS_BOUND,
        "P1 must *not* reproduce this field — if it does, the field is not exercising the \
         quadratic part of the space and quadratic_field_is_exact_on_quadratic_elements is \
         vacuous. Got {worst:.3e} mm, which is inside the bound P2 is held to \
         ({EXACTNESS_BOUND:.0e}). ⚠️ On a *uniform* lattice this is exactly what happens \
         (measured 1.475e-17) — check that JITTER is still non-zero before blaming P1"
    );
}

/// Constraining only the corners of a clamped face leaves the edge nodes on it
/// free, which is the mistake this element makes easy.
///
/// Reported as a measured number rather than a warning in prose: the point is
/// that the solve *succeeds* and returns a different answer, so nothing flags
/// it for the caller.
#[test]
fn constraining_corners_only_is_a_different_problem() {
    let n = 2usize;
    let h = SIDE / n as f64;
    let mesh = kuhn_cube(n, h, JITTER);
    let q = QuadraticMesh::from_tet_mesh(&mesh).expect("well formed");
    let eps = h * 1e-4;
    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid");

    let on_boundary = |p: [f64; 3]| p.iter().any(|&c| c < eps || c > SIDE - eps);
    let pos = |node: u32| {
        let f = q.node_position(node).expect("in range");
        [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()]
    };

    let mut full = BoundaryConditions::new();
    let mut corners_only = BoundaryConditions::new();
    for node in 0..u32::try_from(q.node_count()).expect("fits") {
        let p = pos(node);
        if !on_boundary(p) {
            continue;
        }
        let u = quadratic_field(p);
        let value = [fx(u[0]), fx(u[1]), fx(u[2])];
        full.prescribe_all(node, value);
        if (node as usize) < q.corner_count() {
            corners_only.prescribe_all(node, value);
        }
    }
    assert!(
        corners_only.prescribed_count() < full.prescribed_count(),
        "the two boundary sets must differ, or this test compares a thing with itself"
    );

    let a = solve_quadratic(&q, &pla(), &full, &config).expect("well posed");
    let b = solve_quadratic(&q, &pla(), &corners_only, &config)
        .expect("corner-only constraints still leave a solvable problem — that is the point");

    let mut worst = 0.0_f64;
    for node in 0..q.node_count() {
        for axis in 0..3 {
            worst = worst.max(
                (a.displacements[node][axis].to_f64() - b.displacements[node][axis].to_f64()).abs(),
            );
        }
    }
    eprintln!(
        "[p2fem] corners-only vs full boundary: {} vs {} prescribed dofs, largest difference \
         {worst:.3e} mm (both solves succeeded)",
        corners_only.prescribed_count(),
        full.prescribed_count()
    );
    assert!(
        worst > 1.0e-9,
        "leaving the edge nodes of a prescribed face free must change the answer; got \
         {worst:.3e} mm. If this is now zero, the edge nodes are not carrying degrees of \
         freedom and the element has collapsed to P1"
    );
}

// ---------------------------------------------------------------------------
// does the quadratic element actually relieve the stiffness P1 shows?
// ---------------------------------------------------------------------------

/// A cantilever, meshed from an SDF box, solved with both elements.
///
/// `tests/locking_p1.rs` measures where the P1 element is too stiff and which
/// knob moves it. This is the other half of that question: whether a quadratic
/// element helps, and by how much, on exactly those two knobs — elements
/// through the thickness, and the Poisson ratio.
///
/// # The traction has to be consistent for the element, not for the mesh
///
/// ⚠️ On a **six-node** boundary triangle under a constant traction, the
/// consistent nodal load is **zero at the corners and `A·t/3` at each midside
/// node** — not `A·t/3` at the corners as it is for a three-node triangle.
/// `∫Nᵢ dA / A` is `2·(1/6) − 1/3 = 0` for a corner and `4·(1/12) = 1/3` for a
/// midside. Handing the corners a third each would be a different load, and it
/// would show up as a stiffness difference and be read as an element property.
/// `p2_traction_puts_nothing_on_the_corners` pins the split.
mod cantilever {
    use super::*;
    use alice_physics::linear_elastic_fem::{Axis, FemSolution};
    use alice_physics::sdf_collider::ClosureSdf;
    use alice_physics::sdf_fem_mesh::generate;
    use std::collections::HashMap;

    pub const WIDTH: f64 = 2.0;
    pub const THICKNESS: f64 = 2.0;
    pub const LOAD: f64 = 400.0;

    pub fn beam_sdf(length: f64) -> ClosureSdf {
        let (l, w, t) = (length as f32, WIDTH as f32, THICKNESS as f32);
        ClosureSdf::new(
            move |x, y, z| {
                let dx = (0.0 - x).max(x - l);
                let dy = (0.0 - y).max(y - w);
                let dz = (0.0 - z).max(z - t);
                dx.max(dy).max(dz)
            },
            |_x, _y, _z| (0.0, 0.0, 1.0),
        )
    }

    /// `δ = PL³/(3EI) + PL/(κGA)`, bending plus shear, `κ = 5/6`.
    pub fn beam_tip_mm(length: f64, nu: f64) -> f64 {
        let i = WIDTH * THICKNESS * THICKNESS * THICKNESS / 12.0;
        let bending = LOAD * length * length * length / (3.0 * E_MPA * i);
        let g = E_MPA / (2.0 * (1.0 + nu));
        let shear = LOAD * length / ((5.0 / 6.0) * g * (WIDTH * THICKNESS));
        bending + shear
    }

    pub fn mesh(length: f64, cell: f64) -> SdfTetMesh {
        let m = generate(
            &beam_sdf(length),
            [0.0, 0.0, 0.0],
            [length as f32, WIDTH as f32, THICKNESS as f32],
            cell as f32,
        );
        let nx = (length / cell).round() as usize;
        let ny = (WIDTH / cell).round() as usize;
        let nz = (THICKNESS / cell).round() as usize;
        assert!(
            m.tet_count() == 5 * nx * ny * nz && m.vertex_count() == (nx + 1) * (ny + 1) * (nz + 1),
            "L={length} cell={cell}: the meshed region is not the beam"
        );
        m
    }

    /// Triangles used by exactly one tetrahedron.
    pub fn boundary_faces(mesh: &SdfTetMesh) -> Vec<[u32; 3]> {
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
        let mut out: Vec<[u32; 3]> = counts
            .into_iter()
            .filter(|(_, n)| *n == 1)
            .map(|(f, _)| f)
            .collect();
        // HashMap iteration order is not stable; the loads below are summed in
        // this order, so it has to be fixed before it reaches the solver.
        out.sort_unstable();
        out
    }

    pub fn triangle_area(p: [[f64; 3]; 3]) -> f64 {
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let c = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
    }

    pub struct Run {
        pub ratio: f64,
        pub iterations: u32,
        pub residual: f64,
        pub dofs: usize,
    }

    fn tip(length: f64, nodes: &[[f64; 3]], out: &FemSolution) -> f64 {
        let eps = 1.0e-4;
        let mut sum = 0.0;
        let mut count = 0usize;
        for (n, p) in nodes.iter().enumerate() {
            if (p[0] - length).abs() < eps {
                sum += out.displacements[n][Axis::Z.index()].to_f64();
                count += 1;
            }
        }
        assert!(count > 0, "the loaded end must carry nodes");
        -sum / count as f64
    }

    /// Solve one cantilever with the quadratic element.
    pub fn run_p2(length: f64, cell: f64, nu: f64, config: &SolverConfig) -> Run {
        let m = mesh(length, cell);
        let q = QuadraticMesh::from_tet_mesh(&m).expect("well formed");
        let eps = 1.0e-4;
        let nodes: Vec<[f64; 3]> = (0..q.node_count())
            .map(|n| {
                let f = q.node_position(n as u32).expect("in range");
                [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()]
            })
            .collect();

        let mut bc = BoundaryConditions::new();
        // Clamp every node on x = 0, edge nodes included. Clamping only the
        // corners would leave the midside nodes of the built-in face free.
        for (n, p) in nodes.iter().enumerate() {
            if p[0].abs() < eps {
                bc.fix(u32::try_from(n).expect("fits"));
            }
        }

        let traction = LOAD / (WIDTH * THICKNESS);
        let mut applied = 0.0_f64;
        for face in boundary_faces(&m) {
            let p = [
                nodes[face[0] as usize],
                nodes[face[1] as usize],
                nodes[face[2] as usize],
            ];
            if !p.iter().all(|c| (c[0] - length).abs() < eps) {
                continue;
            }
            let share = -traction * triangle_area(p) / 3.0;
            // Corners get nothing; each midside node gets a third.
            for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                let mid = q
                    .edge_node(face[a], face[b])
                    .expect("a boundary triangle's edges are mesh edges");
                bc.add_load(mid, Axis::Z, fx(share));
                applied += share;
            }
        }
        assert!(
            (applied + LOAD).abs() < 1.0e-6 * LOAD,
            "the consistent nodal loads must sum to the applied force: got {applied}"
        );

        let material = ElasticMaterial::new(fx(E_MPA), fx(nu)).expect("nu in (-1, 0.5)");
        let out = solve_quadratic(&q, &material, &bc, config)
            .unwrap_or_else(|e| panic!("P2 L={length} cell={cell} nu={nu}: {e:?}"));
        Run {
            ratio: tip(length, &nodes, &out) / beam_tip_mm(length, nu),
            iterations: out.iterations,
            residual: out.relative_residual.to_f64(),
            dofs: q.node_count() * 3,
        }
    }

    /// The same cantilever with the linear element, for the comparison.
    pub fn run_p1(length: f64, cell: f64, nu: f64, config: &SolverConfig) -> Run {
        let m = mesh(length, cell);
        let eps = 1.0e-4;
        let nodes: Vec<[f64; 3]> = m
            .vertices
            .iter()
            .map(|q| [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])])
            .collect();
        let mut bc = BoundaryConditions::new();
        for (n, p) in nodes.iter().enumerate() {
            if p[0].abs() < eps {
                bc.fix(u32::try_from(n).expect("fits"));
            }
        }
        let traction = LOAD / (WIDTH * THICKNESS);
        let mut applied = 0.0_f64;
        for face in boundary_faces(&m) {
            let p = [
                nodes[face[0] as usize],
                nodes[face[1] as usize],
                nodes[face[2] as usize],
            ];
            if !p.iter().all(|c| (c[0] - length).abs() < eps) {
                continue;
            }
            let share = -traction * triangle_area(p) / 3.0;
            for n in face {
                bc.add_load(n, Axis::Z, fx(share));
                applied += share;
            }
        }
        assert!(
            (applied + LOAD).abs() < 1.0e-6 * LOAD,
            "loads must sum to the force"
        );
        let material = ElasticMaterial::new(fx(E_MPA), fx(nu)).expect("nu in (-1, 0.5)");
        let out = solve(&m, &material, &bc, config)
            .unwrap_or_else(|e| panic!("P1 L={length} cell={cell} nu={nu}: {e:?}"));
        Run {
            ratio: tip(length, &nodes, &out) / beam_tip_mm(length, nu),
            iterations: out.iterations,
            residual: out.relative_residual.to_f64(),
            dofs: m.vertex_count() * 3,
        }
    }
}

fn sweep_config() -> SolverConfig {
    SolverConfig::try_new(400_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(20_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid")
        .with_preconditioner(alice_physics::linear_elastic_fem::Preconditioner::JacobiScaled)
}

/// The consistent traction split, before any stiffness conclusion rests on it.
#[test]
fn p2_traction_puts_nothing_on_the_corners() {
    // `∫Nᵢ dA / A` over a six-node triangle: 2·(1/6) − 1/3 at a corner,
    // 4·(1/12) at a midside. Checked as arithmetic here so that the loads
    // assembled above are traceable to a number rather than to a memory of one.
    let corner: f64 = 2.0 * (1.0 / 6.0) - 1.0 / 3.0;
    let midside: f64 = 4.0 * (1.0 / 12.0);
    eprintln!("[p2fem] six-node triangle: corner weight {corner:.6}, midside {midside:.6}");
    assert!(
        corner.abs() < 1e-15,
        "corner weight must be zero, got {corner}"
    );
    assert!(
        (midside - 1.0 / 3.0).abs() < 1e-15,
        "midside weight must be one third, got {midside}"
    );
    assert!(
        (3.0 * corner + 3.0 * midside - 1.0).abs() < 1e-15,
        "the six weights must sum to one"
    );
}

/// Elements through the thickness — the knob that sets the bending stiffness.
#[test]
fn quadratic_elements_relieve_the_bending_stiffness() {
    let config = sweep_config();
    eprintln!("[p2fem] ---- cantilever L=10 t=2 w=2 (slenderness 5), nu=0.3 ----");
    eprintln!(
        "[p2fem] {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}  {:>8}",
        "n_t", "P1", "P2", "P1 dofs", "P2 dofs", "P1 iters", "P2 iters"
    );
    let mut rows = Vec::new();
    for (n_t, cell) in [(1usize, 2.0), (2, 1.0), (4, 0.5)] {
        let a = cantilever::run_p1(10.0, cell, 0.3, &config);
        let b = cantilever::run_p2(10.0, cell, 0.3, &config);
        eprintln!(
            "[p2fem] {n_t:>5}  {:>7.4}  {:>7.4}  {:>7}  {:>7}  {:>8}  {:>8}",
            a.ratio, b.ratio, a.dofs, b.dofs, a.iterations, b.iterations
        );
        for r in [&a, &b] {
            assert!(
                r.residual <= 1.0e-9,
                "n_t={n_t}: a solve stopped at residual {:.3e} after {} iterations",
                r.residual,
                r.iterations
            );
        }
        rows.push((n_t, a.ratio, b.ratio));
    }

    for (n_t, p1, p2) in &rows {
        assert!(
            p2 > p1,
            "n_t={n_t}: the quadratic element must be less stiff than the linear one on the \
             same mesh; got P1 {p1:.4} and P2 {p2:.4}"
        );
        assert!(
            *p2 < 1.02,
            "n_t={n_t}: P2 must not overshoot the beam solution by more than the shear \
             model's own error; got {p2:.4}"
        );
    }
    // The headline: one element through the thickness, where P1 is worst.
    let (_, p1_coarse, p2_coarse) = rows[0];
    assert!(
        p2_coarse > 0.9 && p1_coarse < 0.4,
        "at one element through the thickness P1 measured 0.3514 and P2 is expected to \
         recover most of the beam deflection; got P1 {p1_coarse:.4}, P2 {p2_coarse:.4}"
    );
}

/// The Poisson ratio — the knob that drives volumetric locking.
#[test]
fn quadratic_elements_relieve_volumetric_locking() {
    let config = sweep_config();
    eprintln!("[p2fem] ---- cantilever L=10 t=2 cell=1.0 (n_t=2), nu varies ----");
    eprintln!(
        "[p2fem] {:>8}  {:>7}  {:>7}  {:>8}  {:>8}",
        "nu", "P1", "P2", "P1 iters", "P2 iters"
    );
    let nus = [0.3, 0.49, 0.4999];
    let mut rows = Vec::new();
    for nu in nus {
        let a = cantilever::run_p1(10.0, 1.0, nu, &config);
        let b = cantilever::run_p2(10.0, 1.0, nu, &config);
        eprintln!(
            "[p2fem] {nu:>8}  {:>7.4}  {:>7.4}  {:>8}  {:>8}",
            a.ratio, b.ratio, a.iterations, b.iterations
        );
        for r in [&a, &b] {
            assert!(
                r.residual <= 1.0e-9,
                "nu={nu}: a solve stopped at residual {:.3e} after {} iterations",
                r.residual,
                r.iterations
            );
        }
        rows.push((nu, a.ratio, b.ratio));
    }

    for (nu, p1, p2) in &rows {
        assert!(
            p2 > p1,
            "nu={nu}: the quadratic element must be less stiff; got P1 {p1:.4}, P2 {p2:.4}"
        );
    }
    // What "relieves" means, as a number: the loss from ν = 0.3 to the
    // near-incompressible end has to be smaller for P2 than for P1.
    let p1_loss = rows[0].1 - rows[2].1;
    let p2_loss = rows[0].2 - rows[2].2;
    eprintln!("[p2fem] loss from nu=0.3 to nu=0.4999: P1 {p1_loss:.4}, P2 {p2_loss:.4}");
    assert!(
        p2_loss < p1_loss,
        "approaching incompressibility must cost the quadratic element less than the \
         linear one; got P1 {p1_loss:.4} and P2 {p2_loss:.4}"
    );
}

// ---------------------------------------------------------------------------
// determinism
// ---------------------------------------------------------------------------

/// The node numbering and the solution are reproducible, and pinned.
///
/// Two things can drift here and neither shows up as a wrong answer:
///
/// - **The edge node numbering.** It comes from a `BTreeMap` keyed on the
///   sorted corner pair, so it is ordered by the key and not by a hasher. A
///   `HashMap` would renumber the nodes between runs — every displacement index
///   would move, and a caller holding node indices across a rebuild would read
///   the wrong ones. Pinned by hashing the whole `(key → node)` table.
/// - **The solution bits.** `Fix128` is integer arithmetic, so a correct
///   implementation is bit-identical across targets; that is the crate's
///   premise and this is the quadratic element's share of it.
///
/// ⚠️ Pinned in the **raw representation**. A golden taken through `to_f64`
/// carries 53 bits of a 64-bit fraction and would accept a drift of up to 2⁻⁵³.
#[test]
fn quadratic_solve_is_deterministic_and_pinned() {
    fn fnv(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    let build = || {
        let mesh = kuhn_cube(2, SIDE / 2.0, JITTER);
        QuadraticMesh::from_tet_mesh(&mesh).expect("well formed")
    };

    // (a) the numbering, hashed over every element's ten slots in order
    let numbering = |q: &QuadraticMesh| {
        let mut bytes = Vec::new();
        for e in 0..q.element_count() {
            for n in q.element_nodes(e).expect("in range") {
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        }
        fnv(&bytes)
    };

    let first = build();
    let second = build();
    let (h1, h2) = (numbering(&first), numbering(&second));
    eprintln!("[p2fem] node numbering hash {h1:#018x} (rebuild {h2:#018x})");
    assert!(
        h1 == h2,
        "rebuilding the same mesh must produce the same node numbering; {h1:#x} vs {h2:#x}"
    );
    assert!(
        h1 == 0xffbf_cedf_0804_fb53,
        "the edge node numbering moved: got {h1:#018x}. If the edge table stopped being \
         ordered — a `HashMap` in place of the `BTreeMap` — every node index downstream \
         moved with it"
    );

    // (b) the solution, hashed over the raw parts of every nodal displacement
    let (worst, _, _, _) = interior_error(2, quadratic_field);
    assert!(
        worst < EXACTNESS_BOUND,
        "the golden below is only meaningful on a solve that is still correct"
    );

    let solve_hash = |q: &QuadraticMesh| {
        let eps = (SIDE / 2.0) * 1e-4;
        let mut bc = BoundaryConditions::new();
        for node in 0..u32::try_from(q.node_count()).expect("fits") {
            let f = q.node_position(node).expect("in range");
            let p = [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()];
            if p.iter().any(|&c| c < eps || c > SIDE - eps) {
                let u = quadratic_field(p);
                bc.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
            }
        }
        let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
            .expect("valid")
            .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
            .expect("valid");
        let out = solve_quadratic(q, &pla(), &bc, &config).expect("well posed");
        let mut bytes = Vec::new();
        for d in &out.displacements {
            for c in d {
                bytes.extend_from_slice(&c.hi.to_le_bytes());
                bytes.extend_from_slice(&c.lo.to_le_bytes());
            }
        }
        (fnv(&bytes), out.iterations)
    };
    let (s1, i1) = solve_hash(&first);
    let (s2, i2) = solve_hash(&second);
    eprintln!("[p2fem] solution hash {s1:#018x} after {i1} iterations (repeat {s2:#018x}, {i2})");
    assert!(
        s1 == s2 && i1 == i2,
        "the same problem must give the same bits and the same iteration count"
    );
    assert!(
        s1 == 0x93cf_96c2_be0f_f85f,
        "the quadratic solution moved: got {s1:#018x} after {i1} iterations. This is a \
         change in the element, the quadrature or `Fix128` — not a tolerance"
    );
}
