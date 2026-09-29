//! Small-strain linear elastic FEM on **ten-node quadratic tetrahedra** (P2).
//!
//! The companion of [`crate::linear_elastic_fem`], which solves the same
//! boundary value problem with four-node constant-strain tetrahedra. Both are
//! kept: P1 is cheaper and its behaviour is pinned by a family of oracles, and
//! this module does not change it by one bit. [`solve_quadratic`] is a separate
//! entry point, not a replacement.
//!
//! # Why a second element
//!
//! A P1 tetrahedron carries one constant strain, so a through-thickness strain
//! gradient costs elements. Measured on a 5:1 cantilever against the beam
//! solution (`tests/locking_p1.rs`):
//!
//! | elements through the thickness | 1 | 2 | 4 | 8 |
//! |---|---|---|---|---|
//! | FEM / beam | 0.3514 | 0.6607 | 0.8749 | 0.9547 |
//!
//! and at a fixed mesh, approaching incompressibility:
//!
//! | ν | 0.3 | 0.45 | 0.49 | 0.499 | 0.4999 |
//! |---|---|---|---|---|---|
//! | FEM / beam | 0.8749 | 0.8275 | 0.7795 | 0.7268 | 0.7068 |
//!
//! ⚠️ **What this is not a remedy for.** The same file measures that
//! slenderness does *not* worsen the stiffness at a fixed element shape
//! (0.6457 → 0.6794 over `L/t` 2 → 20, drifting the *helpful* way), because
//! [`crate::sdf_fem_mesh::generate`] takes one cell size for all three axes and
//! so cannot produce a long thin element at all. Classical aspect-ratio shear
//! locking is not reachable through that mesher, and a quadratic element is not
//! being introduced to cure it.
//!
//! # The nodes live here, not in the mesh
//!
//! [`crate::sdf_fem_mesh::SdfTetMesh`] stays P1: its `vertices` are the lattice
//! corners and `tests/analytic_fem_convergence.rs` asserts that count against
//! the closed form for the lattice. Edge nodes are therefore built here, by
//! [`QuadraticMesh::from_tet_mesh`], and numbered **after** the corners — so a
//! corner keeps the index it has in the tetrahedral mesh and existing boundary
//! data indexes the same nodes it always did.
//!
//! # Quadrature
//!
//! On a straight-edged tetrahedron the Jacobian is constant, the shape function
//! gradients are linear in the barycentric coordinates, and `BᵀDB` is therefore
//! quadratic — a degree-2 rule integrates the element stiffness **exactly**.
//! Hammer–Stroud's four points are used. Their abscissae are irrational
//! (`(5 ± √5)/20`), which is not a problem: `Fix128::sqrt` is an exact integer
//! square root over a fixed 96 steps, and `tests/p2_quadrature_fix128.rs`
//! measures the rule reproducing every barycentric moment of degree ≤ 2 to
//! within **2 ulp**.
//!
//! ⚠️ That file also measures the *rational* alternative (Keast's five points,
//! weights `−4/5` and `9/20`) coming out **worse**, at 5 ulp. Avoiding `√5`
//! costs accuracy here, because Hammer–Stroud's weight is `1/4` — exactly
//! representable in a binary fixed-point type — while `1/6`, `4/5` and `9/20`
//! are not, and Keast's negative centroid weight makes its five terms cancel.
//! **Do not "simplify" this rule to the rational one.**
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::collections::BTreeMap;

use crate::linear_elastic_fem::{
    BoundaryConditions, ElasticMaterial, FemError, FemSolution, Preconditioner, SolverConfig,
    StressTensor, RESIDUAL_NORM_FLOOR,
};
use crate::math::Fix128;
use crate::sdf_fem_mesh::SdfTetMesh;

/// Nodes per element: four corners then six edge midpoints.
const NODES_PER_ELEMENT: usize = 10;

/// The six edges of a tetrahedron, as corner index pairs, in the order the
/// edge nodes occupy slots 4..10 of an element.
///
/// Fixed and index-ordered: the numbering of the global edge nodes follows this
/// table, so two runs on the same mesh produce the same node indices and the
/// same assembly order.
const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn half() -> Fix128 {
    Fix128::from_raw(0, 1 << 63)
}

fn sub3(a: [Fix128; 3], b: [Fix128; 3]) -> [Fix128; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [Fix128; 3], b: [Fix128; 3]) -> [Fix128; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [Fix128; 3], b: [Fix128; 3]) -> Fix128 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn div3(a: [Fix128; 3], d: Fix128) -> [Fix128; 3] {
    [a[0] / d, a[1] / d, a[2] / d]
}

// ---------------------------------------------------------------------------
// quadrature
// ---------------------------------------------------------------------------

/// Hammer–Stroud four-point rule, exact to degree 2 on the reference
/// tetrahedron.
///
/// Each point carries `b = (5 + 3√5)/20` on one barycentric coordinate and
/// `a = (5 − √5)/20` on the other three; every weight is `1/4`, so the caller
/// multiplies the weighted sum by the element volume and nothing else.
fn quadrature_points() -> [[Fix128; 4]; 4] {
    let five = Fix128::from_int(5);
    let root5 = five.sqrt();
    let twenty = Fix128::from_int(20);
    let a = (five - root5) / twenty;
    let b = (five + Fix128::from_int(3) * root5) / twenty;
    let mut points = [[a; 4]; 4];
    for (i, point) in points.iter_mut().enumerate() {
        point[i] = b;
    }
    points
}

/// The common weight `1/4`, exact in a binary fixed-point type.
fn quadrature_weight() -> Fix128 {
    half() * half()
}

// ---------------------------------------------------------------------------
// the mesh
// ---------------------------------------------------------------------------

/// One quadratic element: its ten global node indices, the (constant)
/// barycentric gradients of the underlying tetrahedron, and its volume.
#[derive(Clone, Copy, Debug)]
struct QuadraticElement {
    nodes: [usize; NODES_PER_ELEMENT],
    /// `∇λ_i` (1/mm), constant over a straight-edged tetrahedron.
    grad_lambda: [[Fix128; 3]; 4],
    /// `|det J| / 6` (mm³).
    volume: Fix128,
}

/// A [`SdfTetMesh`] with the edge nodes a P2 element needs.
///
/// Corners keep their tetrahedral-mesh indices; edge nodes follow, numbered by
/// first appearance in element order. Build it once and solve on it repeatedly:
/// the edge table and the element geometry do not depend on the material or the
/// boundary data.
#[derive(Clone, Debug)]
pub struct QuadraticMesh {
    corner_count: usize,
    positions: Vec<[Fix128; 3]>,
    elements: Vec<QuadraticElement>,
    /// `(min, max)` corner pair → edge node index. `BTreeMap` and not a hash
    /// map: the iteration order of a hash map depends on the hasher, and every
    /// downstream index would inherit that.
    edge_nodes: BTreeMap<(u32, u32), u32>,
}

impl QuadraticMesh {
    /// Build the quadratic mesh from a tetrahedral one.
    ///
    /// # Errors
    ///
    /// [`FemError::EmptyMesh`] for a mesh with no vertices or no tetrahedra,
    /// [`FemError::VertexOutOfRange`] for a tetrahedron naming a vertex the
    /// mesh does not have, and [`FemError::DegenerateElement`] for a
    /// tetrahedron of zero volume — the same three the P1 assembly reports, for
    /// the same reasons.
    pub fn from_tet_mesh(mesh: &SdfTetMesh) -> Result<Self, FemError> {
        let corner_count = mesh.vertices.len();
        if corner_count == 0 || mesh.tets.is_empty() {
            return Err(FemError::EmptyMesh);
        }
        let six = Fix128::from_int(6);

        let mut positions: Vec<[Fix128; 3]> = mesh
            .vertices
            .iter()
            .map(|q| {
                [
                    Fix128::from_f32(q[0]),
                    Fix128::from_f32(q[1]),
                    Fix128::from_f32(q[2]),
                ]
            })
            .collect();
        let mut edge_nodes: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut elements = Vec::with_capacity(mesh.tets.len());

        for (t, tet) in mesh.tets.iter().enumerate() {
            let mut corner = [0usize; 4];
            let mut p = [[Fix128::ZERO; 3]; 4];
            for (i, &v) in tet.vertices.iter().enumerate() {
                let idx = v as usize;
                if idx >= corner_count {
                    return Err(FemError::VertexOutOfRange {
                        vertex: v,
                        vertex_count: corner_count,
                    });
                }
                corner[i] = idx;
                p[i] = positions[idx];
            }

            let e1 = sub3(p[1], p[0]);
            let e2 = sub3(p[2], p[0]);
            let e3 = sub3(p[3], p[0]);
            let det = dot3(e1, cross3(e2, e3));
            if det.is_zero() {
                return Err(FemError::DegenerateElement { tet: t });
            }
            let g1 = div3(cross3(e2, e3), det);
            let g2 = div3(cross3(e3, e1), det);
            let g3 = div3(cross3(e1, e2), det);
            let g0 = [
                -(g1[0] + g2[0] + g3[0]),
                -(g1[1] + g2[1] + g3[1]),
                -(g1[2] + g2[2] + g3[2]),
            ];

            let mut nodes = [0usize; NODES_PER_ELEMENT];
            nodes[..4].copy_from_slice(&corner);
            for (slot, &(i, j)) in EDGES.iter().enumerate() {
                let (a, b) = (tet.vertices[i], tet.vertices[j]);
                let key = if a <= b { (a, b) } else { (b, a) };
                let next = u32::try_from(positions.len()).map_err(|_| FemError::EmptyMesh)?;
                let node = *edge_nodes.entry(key).or_insert(next);
                if node == next {
                    // The midpoint. `1/2` is exact in a binary fixed-point type,
                    // so the edge node of a shared edge is bit-identical however
                    // it is reached.
                    let (pa, pb) = (p[i], p[j]);
                    positions.push([
                        half() * (pa[0] + pb[0]),
                        half() * (pa[1] + pb[1]),
                        half() * (pa[2] + pb[2]),
                    ]);
                }
                nodes[4 + slot] = node as usize;
            }

            elements.push(QuadraticElement {
                nodes,
                grad_lambda: [g0, g1, g2, g3],
                volume: det.abs() / six,
            });
        }

        Ok(Self {
            corner_count,
            positions,
            elements,
            edge_nodes,
        })
    }

    /// Total node count: corners plus edge nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of corner nodes, which are the vertices of the source mesh and
    /// occupy indices `0..corner_count()`.
    #[must_use]
    pub const fn corner_count(&self) -> usize {
        self.corner_count
    }

    /// Number of edge nodes.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.positions.len() - self.corner_count
    }

    /// Element count, equal to the source mesh's tetrahedron count.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Position of one node (mm). Corners come back exactly as the source mesh
    /// holds them; edge nodes are the midpoints.
    #[must_use]
    pub fn node_position(&self, node: u32) -> Option<[Fix128; 3]> {
        self.positions.get(node as usize).copied()
    }

    /// The edge node between two corners, if that edge is in the mesh. The
    /// order of the two arguments does not matter.
    #[must_use]
    pub fn edge_node(&self, a: u32, b: u32) -> Option<u32> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.edge_nodes.get(&key).copied()
    }

    /// The ten node indices of one element, corners first.
    #[must_use]
    pub fn element_nodes(&self, element: usize) -> Option<[u32; NODES_PER_ELEMENT]> {
        let e = self.elements.get(element)?;
        let mut out = [0u32; NODES_PER_ELEMENT];
        for (slot, &n) in e.nodes.iter().enumerate() {
            out[slot] = u32::try_from(n).ok()?;
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// shape functions
// ---------------------------------------------------------------------------

/// `∇N` for all ten shape functions at one barycentric point.
///
/// Corner `i`: `N = λ_i(2λ_i − 1)`, so `∇N = (4λ_i − 1)∇λ_i`.
/// Edge `(i,j)`: `N = 4λ_iλ_j`, so `∇N = 4(λ_i∇λ_j + λ_j∇λ_i)`.
fn shape_gradients(
    element: &QuadraticElement,
    lambda: &[Fix128; 4],
) -> [[Fix128; 3]; NODES_PER_ELEMENT] {
    let four = Fix128::from_int(4);
    let mut grad = [[Fix128::ZERO; 3]; NODES_PER_ELEMENT];
    for (i, (out, gl)) in grad.iter_mut().zip(element.grad_lambda.iter()).enumerate() {
        let s = four * lambda[i] - Fix128::ONE;
        for (o, c) in out.iter_mut().zip(gl.iter()) {
            *o = s * *c;
        }
    }
    for (slot, &(i, j)) in EDGES.iter().enumerate() {
        let (li, lj) = (lambda[i], lambda[j]);
        let (gi, gj) = (element.grad_lambda[i], element.grad_lambda[j]);
        for (o, (ci, cj)) in grad[4 + slot].iter_mut().zip(gi.iter().zip(gj.iter())) {
            *o = four * (li * *cj + lj * *ci);
        }
    }
    grad
}

/// `σ = D B u_e` at one point inside an element.
fn stress_at(
    element: &QuadraticElement,
    grad: &[[Fix128; 3]; NODES_PER_ELEMENT],
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
) -> StressTensor {
    let mut exx = Fix128::ZERO;
    let mut eyy = Fix128::ZERO;
    let mut ezz = Fix128::ZERO;
    let mut gxy = Fix128::ZERO;
    let mut gyz = Fix128::ZERO;
    let mut gzx = Fix128::ZERO;
    for (g, &node) in grad.iter().zip(element.nodes.iter()) {
        let base = node * 3;
        let (ux, uy, uz) = (u[base], u[base + 1], u[base + 2]);
        exx = exx + g[0] * ux;
        eyy = eyy + g[1] * uy;
        ezz = ezz + g[2] * uz;
        gxy = gxy + g[1] * ux + g[0] * uy;
        gyz = gyz + g[2] * uy + g[1] * uz;
        gzx = gzx + g[2] * ux + g[0] * uz;
    }
    let trace = exx + eyy + ezz;
    let two_mu = mu + mu;
    StressTensor {
        xx: lambda * trace + two_mu * exx,
        yy: lambda * trace + two_mu * eyy,
        zz: lambda * trace + two_mu * ezz,
        xy: mu * gxy,
        yz: mu * gyz,
        zx: mu * gzx,
    }
}

/// `out = K u`, accumulated element by element and quadrature point by
/// quadrature point, both in index order.
fn apply_stiffness(
    elements: &[QuadraticElement],
    points: &[[Fix128; 4]; 4],
    weight: Fix128,
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    out: &mut [Fix128],
) {
    out.fill(Fix128::ZERO);
    for element in elements {
        let scale = element.volume * weight;
        for point in points {
            let grad = shape_gradients(element, point);
            let s = stress_at(element, &grad, u, lambda, mu);
            for (g, &node) in grad.iter().zip(element.nodes.iter()) {
                let base = node * 3;
                out[base] = out[base] + scale * (g[0] * s.xx + g[1] * s.xy + g[2] * s.zx);
                out[base + 1] = out[base + 1] + scale * (g[1] * s.yy + g[0] * s.xy + g[2] * s.yz);
                out[base + 2] = out[base + 2] + scale * (g[2] * s.zz + g[1] * s.yz + g[0] * s.zx);
            }
        }
    }
}

/// Diagonal of the global stiffness, without forming `K`.
///
/// Same per-node expression as the P1 assembly — `(λ+2μ)g_a² + μ(g_b² + g_c²)`
/// — summed over the quadrature points rather than taken once, because the
/// gradients are no longer constant over the element.
fn stiffness_diagonal(
    elements: &[QuadraticElement],
    points: &[[Fix128; 4]; 4],
    weight: Fix128,
    lambda: Fix128,
    mu: Fix128,
    ndof: usize,
) -> Vec<Fix128> {
    let mut diag = vec![Fix128::ZERO; ndof];
    let lambda_2mu = lambda + mu + mu;
    for element in elements {
        let scale = element.volume * weight;
        for point in points {
            let grad = shape_gradients(element, point);
            for (g, &node) in grad.iter().zip(element.nodes.iter()) {
                let sq = [g[0] * g[0], g[1] * g[1], g[2] * g[2]];
                let base = node * 3;
                for axis in 0..3 {
                    let others = sq[(axis + 1) % 3] + sq[(axis + 2) % 3];
                    diag[base + axis] =
                        diag[base + axis] + scale * (lambda_2mu * sq[axis] + mu * others);
                }
            }
        }
    }
    diag
}

fn dot(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc + *x * *y;
    }
    acc
}

fn relative(residual: Fix128, b_norm: Fix128) -> Fix128 {
    if b_norm.is_zero() {
        Fix128::ZERO
    } else {
        residual / b_norm
    }
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

/// Solve the linear elastic boundary value problem on a quadratic mesh.
///
/// `boundary` indexes **nodes of `mesh`**, not vertices of the tetrahedral mesh
/// it came from. Corner indices coincide, so boundary data written for the P1
/// solver constrains the same corners here — but the edge nodes on a clamped
/// face are then left free, which is a different problem. Use
/// [`QuadraticMesh::edge_node`] to reach them.
///
/// # Errors
///
/// See [`FemError`]. [`FemError::UnderConstrained`] covers the case this
/// element makes easy to hit by accident: constraining only the corners of a
/// face leaves the edge nodes on it free.
pub fn solve_quadratic(
    mesh: &QuadraticMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &SolverConfig,
) -> Result<FemSolution, FemError> {
    let node_count = mesh.node_count();
    if node_count == 0 || mesh.elements.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    for &(node, _, _) in boundary.prescribed().iter().chain(boundary.loads().iter()) {
        if node as usize >= node_count {
            return Err(FemError::VertexOutOfRange {
                vertex: node,
                vertex_count: node_count,
            });
        }
    }

    let (lambda, mu) = material.lame();
    let ndof = node_count * 3;
    let points = quadrature_points();
    let weight = quadrature_weight();

    let mut prescribed_value = vec![Fix128::ZERO; ndof];
    let mut is_free = vec![true; ndof];
    for &(node, axis, value) in boundary.prescribed() {
        let d = node as usize * 3 + axis.index();
        prescribed_value[d] = value;
        is_free[d] = false;
    }
    if is_free.iter().all(|f| !*f) {
        // Fully prescribed: the answer is the boundary data itself.
        // Nothing to solve, so nothing was relaxed: the tolerance that was
        // actually honoured is the one that was asked for.
        return Ok(finish(
            mesh,
            &prescribed_value,
            lambda,
            mu,
            0,
            Fix128::ZERO,
            config.relative_tolerance(),
        ));
    }

    let mut scratch = vec![Fix128::ZERO; ndof];

    // b = f − K·u_prescribed, restricted to the free degrees of freedom.
    apply_stiffness(
        &mesh.elements,
        &points,
        weight,
        &prescribed_value,
        lambda,
        mu,
        &mut scratch,
    );
    let mut b = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            b[d] = -scratch[d];
        }
    }
    for &(node, axis, force) in boundary.loads() {
        let d = node as usize * 3 + axis.index();
        if is_free[d] {
            b[d] = b[d] + force;
        }
    }

    let precond = match config.preconditioner() {
        Preconditioner::None => vec![Fix128::ONE; ndof],
        Preconditioner::JacobiScaled => {
            let diag = stiffness_diagonal(&mesh.elements, &points, weight, lambda, mu, ndof);
            let mut sum = Fix128::ZERO;
            let mut free = 0i64;
            for (d, value) in diag.iter().enumerate() {
                if is_free[d] {
                    sum = sum + *value;
                    free += 1;
                }
            }
            if free == 0 || sum.is_zero() {
                vec![Fix128::ONE; ndof]
            } else {
                let mean = sum / Fix128::from_int(free);
                let mut m = vec![Fix128::ONE; ndof];
                for (d, value) in diag.iter().enumerate() {
                    if is_free[d] && !value.is_zero() {
                        m[d] = mean / *value;
                    }
                }
                m
            }
        }
    };

    // Conjugate gradient on the free degrees of freedom.
    let mut x = vec![Fix128::ZERO; ndof];
    let mut r = b.clone();
    let mut z = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            z[d] = r[d] * precond[d];
        }
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(&b, &b).sqrt();
    let requested = config.relative_tolerance() * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };
    let effective = relative(target, b_norm);

    let mut iterations = 0u32;
    let mut residual_norm = dot(&r, &r).sqrt();
    let mut best_residual = residual_norm;
    let mut since_improvement = 0u32;

    while residual_norm > target {
        if iterations >= config.max_iterations() {
            return Err(FemError::NotConverged {
                iterations,
                relative_residual: relative(residual_norm, b_norm),
            });
        }
        let window = {
            let scaled =
                config.stagnation_window_fraction() * Fix128::from_int(i64::from(iterations));
            let scaled = if scaled.is_negative() {
                0
            } else {
                u32::try_from(scaled.hi).unwrap_or(u32::MAX)
            };
            scaled.max(config.stagnation_min_window())
        };
        if since_improvement >= window {
            return Err(FemError::Stagnated {
                iterations,
                relative_residual: relative(best_residual, b_norm),
                without_improvement: since_improvement,
            });
        }
        apply_stiffness(
            &mesh.elements,
            &points,
            weight,
            &p,
            lambda,
            mu,
            &mut scratch,
        );
        for (d, value) in scratch.iter_mut().enumerate() {
            if !is_free[d] {
                *value = Fix128::ZERO;
            }
        }
        let pkp = dot(&p, &scratch);
        if pkp <= Fix128::ZERO {
            if iterations == 0 {
                return Err(FemError::UnderConstrained);
            }
            return Err(FemError::Stagnated {
                iterations,
                relative_residual: relative(best_residual, b_norm),
                without_improvement: since_improvement,
            });
        }
        let alpha = rz / pkp;
        for d in 0..ndof {
            if is_free[d] {
                x[d] = x[d] + alpha * p[d];
                r[d] = r[d] - alpha * scratch[d];
            }
        }
        for d in 0..ndof {
            if is_free[d] {
                z[d] = r[d] * precond[d];
            }
        }
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        for d in 0..ndof {
            if is_free[d] {
                p[d] = z[d] + beta * p[d];
            }
        }
        rz = rz_next;
        residual_norm = dot(&r, &r).sqrt();
        iterations += 1;

        if residual_norm < best_residual - best_residual * config.stagnation_min_improvement() {
            best_residual = residual_norm;
            since_improvement = 0;
        } else {
            if residual_norm < best_residual {
                best_residual = residual_norm;
            }
            since_improvement += 1;
        }
    }

    let mut u = prescribed_value;
    for d in 0..ndof {
        if is_free[d] {
            u[d] = x[d];
        }
    }
    Ok(finish(
        mesh,
        &u,
        lambda,
        mu,
        iterations,
        relative(residual_norm, b_norm),
        effective,
    ))
}

/// Package the displacement field and the per-element stress.
///
/// `effective_relative_tolerance` is the **floor-adjusted** tolerance, matching
/// [`crate::linear_elastic_fem::solve`]: when the requested tolerance would
/// demand a residual norm below [`RESIDUAL_NORM_FLOOR`], what the iteration was
/// actually held to is the floor. Reporting the requested value instead would
/// make the field agree with `SolverConfig` always and therefore say nothing.
///
/// ⚠️ **`element_stress` is evaluated at the element centroid.** P1 strain is
/// constant and its per-element stress is the whole truth; P2 strain is linear,
/// so one tensor per element is a sample and not a summary. The centroid is the
/// sample with the best accuracy for a linear field — it is where the linear
/// part averages out — but a caller looking for the peak stress in a bending
/// element will not find it here.
fn finish(
    mesh: &QuadraticMesh,
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    iterations: u32,
    relative_residual: Fix128,
    effective_relative_tolerance: Fix128,
) -> FemSolution {
    let quarter = half() * half();
    let centroid = [quarter; 4];
    let displacements = (0..mesh.node_count())
        .map(|n| [u[n * 3], u[n * 3 + 1], u[n * 3 + 2]])
        .collect();
    let element_stress = mesh
        .elements
        .iter()
        .map(|e| {
            let grad = shape_gradients(e, &centroid);
            stress_at(e, &grad, u, lambda, mu)
        })
        .collect();
    FemSolution {
        displacements,
        element_stress,
        iterations,
        relative_residual,
        effective_relative_tolerance,
    }
}
