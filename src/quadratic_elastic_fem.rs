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

use crate::hyperelastic::HyperelasticModel;
use crate::linear_elastic_fem::{
    hyperelastic_volumetric_modulus, BoundaryConditions, CorotationalConfig, CorotationalSolution,
    ElasticMaterial, FemError, FemSolution, Preconditioner, SolverConfig, StressTensor,
    RESIDUAL_NORM_FLOOR,
};
use crate::math::{Fix128, Mat3Fix, PolarError, Vec3Fix};
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

// ---------------------------------------------------------------------------
// hyperelasticity on the quadratic element
// ---------------------------------------------------------------------------

/// `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ` at one point inside an element.
///
/// ⚠️ **Unlike the P1 gradient this is a function of the point, not of the
/// element.** `Σᵢ Xᵢ ⊗ ∇Nᵢ = I` still holds — P1 ⊂ P2, so the quadratic shape
/// functions reproduce a linear field exactly on a straight-edged tetrahedron
/// whose edge nodes sit at the midpoints — but `Σᵢ uᵢ ⊗ ∇Nᵢ` varies across the
/// element because `∇Nᵢ` does. That is the whole difference between the
/// non-linear machinery here and the one in
/// [`crate::linear_elastic_fem`](crate::linear_elastic_fem), where one gradient
/// per element is the whole truth.
///
/// The identity is not assumed: `the_quadratic_gradient_of_an_affine_field_is_exact`
/// in `tests/analytic_quadratic_hyperelastic.rs` measures it.
fn deformation_gradient_at(
    element: &QuadraticElement,
    grad: &[[Fix128; 3]; NODES_PER_ELEMENT],
    u: &[Fix128],
) -> Mat3Fix {
    // `cols[b][a]` is the entry in row `a`, column `b` — the layout `Mat3Fix`
    // stores.
    let mut cols = [[Fix128::ZERO; 3]; 3];
    for (g, &node) in grad.iter().zip(element.nodes.iter()) {
        let base = node * 3;
        let node_u = [u[base], u[base + 1], u[base + 2]];
        for (b, gb) in g.iter().enumerate() {
            for (a, ua) in node_u.iter().enumerate() {
                cols[b][a] = cols[b][a] + *ua * *gb;
            }
        }
    }
    for (k, col) in cols.iter_mut().enumerate() {
        col[k] = col[k] + Fix128::ONE;
    }
    Mat3Fix::from_cols(
        Vec3Fix::new(cols[0][0], cols[0][1], cols[0][2]),
        Vec3Fix::new(cols[1][0], cols[1][1], cols[1][2]),
        Vec3Fix::new(cols[2][0], cols[2][1], cols[2][2]),
    )
}

/// `σ` and `P = J σ F⁻ᵀ` under a hyperelastic law, or `None` when `F` has no
/// positive determinant or no inverse.
///
/// The same pair, computed the same way, as the private `hyperelastic_stress`
/// of [`crate::linear_elastic_fem`]. It is repeated rather than shared because
/// the P1 one is not on the public surface; the arithmetic is a function of `F`
/// alone, so nothing about the element enters here.
fn hyperelastic_stress(
    model: &HyperelasticModel,
    bulk_modulus: Fix128,
    gradient: Mat3Fix,
) -> Option<(StressTensor, Mat3Fix)> {
    let sigma = crate::hyperelastic::cauchy_stress(model, bulk_modulus, gradient)?;
    let inverse_transpose = gradient.inverse()?.transpose();
    let piola = sigma
        .mul_mat(inverse_transpose)
        .scale(gradient.determinant());
    let cauchy = StressTensor {
        xx: sigma.col0.x,
        yy: sigma.col1.y,
        zz: sigma.col2.z,
        xy: sigma.col1.x,
        yz: sigma.col2.y,
        zx: sigma.col2.x,
    };
    Some((cauchy, piola))
}

/// The material law a hyperelastic solve runs under: the model and the bulk
/// modulus that fixes the pressure its incompressible energy leaves free.
#[derive(Clone, Copy, Debug)]
struct MaterialLaw {
    model: HyperelasticModel,
    bulk_modulus: Fix128,
}

/// `Σₑ Σ_q w_q V₀ P(F(ξ_q)) ∇₀N(ξ_q)` — the total-Lagrangian internal force.
///
/// ⚠️ **The quadrature is not exact here and cannot be made so.** For
/// Neo-Hookean `P = μF + [κJ(J−1) − μ]·F⁻ᵀ`, which is degree five in the
/// entries of `F`, so with `F` linear in the barycentric coordinates the
/// integrand `P : ∇N` is degree six against a rule exact to degree two. The one
/// case that *is* exact is a **uniform** `F`: `P` then comes out of the integral
/// and what is left is `∫∇N`, degree one. Every closed-form oracle in
/// `tests/analytic_quadratic_hyperelastic.rs` is built on a uniform `F` for that
/// reason, and the module there says so.
fn material_internal_force(
    elements: &[QuadraticElement],
    points: &[[Fix128; 4]; 4],
    weight: Fix128,
    u: &[Fix128],
    law: MaterialLaw,
    out: &mut [Fix128],
) -> Result<(), FemError> {
    out.fill(Fix128::ZERO);
    for (tet, element) in elements.iter().enumerate() {
        for point in points {
            let scale = element.volume * weight;
            let grad = shape_gradients(element, point);
            let gradient = deformation_gradient_at(element, &grad, u);
            let (_, piola) = hyperelastic_stress(&law.model, law.bulk_modulus, gradient).ok_or(
                FemError::RotationFailed {
                    tet,
                    cause: PolarError::Inverted,
                },
            )?;
            let rows = [
                [piola.col0.x, piola.col1.x, piola.col2.x],
                [piola.col0.y, piola.col1.y, piola.col2.y],
                [piola.col0.z, piola.col1.z, piola.col2.z],
            ];
            for (g, &node) in grad.iter().zip(element.nodes.iter()) {
                let base = node * 3;
                for (axis, row) in rows.iter().enumerate() {
                    out[base + axis] =
                        out[base + axis] + scale * (row[0] * g[0] + row[1] * g[1] + row[2] * g[2]);
                }
            }
        }
    }
    Ok(())
}

/// Largest absolute entry, which is the norm the Newton stop uses.
fn max_abs(v: &[Fix128]) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for entry in v {
        let a = entry.abs();
        if a > worst {
            worst = a;
        }
    }
    worst
}

/// Smallest `det F` a frame may be built from, as `2⁻²⁰ ≈ 9.5e-7`.
///
/// ⚠️ The same value as `linear_elastic_fem::POLAR_DET_FLOOR`, repeated because
/// that one is private and this path must not widen the crate's public surface
/// to reach it. Kept identical on purpose: a frame refused on P1 and accepted
/// on P2 for the same `F` would be a difference nobody asked for.
const POLAR_DET_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 44);

/// `R Kₑ⁰ Rᵀ` — the co-rotational tangent a Newton step is solved with.
///
/// ⚠️ **This is still not the material tangent**, so the iteration stays a
/// *modified* Newton one and the fixed point is unchanged: it is
/// `f_ext = f_mat(u)` whatever operator steps towards it. What the rotation
/// buys is the **domain the iteration converges on**, which the plain
/// small-strain operator loses as soon as the deformation carries a rotation —
/// measured, in `tests/analytic_quadratic_hyperelastic.rs`, as a superposed
/// rotation of five degrees that does not converge in 200 Newton steps and
/// whose residual is bit identical at four increments and at sixteen.
///
/// ⚠️ **One rotation per element, from `F` at the centroid** — not one per
/// quadrature point. The internal force has to be integrated point by point
/// because it decides the answer; the tangent only decides the path, so a
/// coarser rotation costs iterations and not correctness. P3 carries 24 points
/// per element, where a polar decomposition each would be 24 times the cost of
/// this for no change in what the iteration converges to.
struct RotatedOperator<'a> {
    elements: &'a [QuadraticElement],
    points: &'a [[Fix128; 4]; 4],
    weight: Fix128,
    lame: (Fix128, Fix128),
    /// One per element, recomputed from the current displacement at the start
    /// of every Newton step and held fixed for the duration of that step —
    /// which is what makes `f_lin` affine within the step, and therefore what
    /// makes the step exact for the linear law.
    rotations: Vec<Mat3Fix>,
}

impl<'a> RotatedOperator<'a> {
    fn new(
        elements: &'a [QuadraticElement],
        points: &'a [[Fix128; 4]; 4],
        weight: Fix128,
        lame: (Fix128, Fix128),
    ) -> Self {
        Self {
            elements,
            points,
            weight,
            lame,
            rotations: vec![Mat3Fix::IDENTITY; elements.len()],
        }
    }

    /// Rebuild the frames from `u`, or report the element that has no rotation.
    ///
    /// ⚠️ A refusal is **not** propagated as an error here. The state a Newton
    /// step opens on can be a prediction rather than a candidate solution, and
    /// a prediction is allowed to be degenerate; keeping the frames already in
    /// hand makes that step a small-strain step, which is the right thing to do
    /// from a guess that carries no rotation information. The *force* still
    /// refuses on a non-positive `det F`, which is where a genuinely inverted
    /// element is caught.
    fn refresh(&mut self, u: &[Fix128], polar_iterations: u32) {
        let centroid = [half() * half(); 4];
        for (slot, element) in self.elements.iter().enumerate() {
            let grad = shape_gradients(element, &centroid);
            let gradient = deformation_gradient_at(element, &grad, u);
            if let Ok(r) = gradient.polar_rotation(POLAR_DET_FLOOR, polar_iterations) {
                self.rotations[slot] = r;
            }
        }
    }

    /// `out = Σₑ R Kₑ⁰ Rᵀ u`, matrix free.
    fn apply(&self, u: &[Fix128], out: &mut [Fix128]) {
        let (lambda, mu) = self.lame;
        out.fill(Fix128::ZERO);
        for (element, rotation) in self.elements.iter().zip(self.rotations.iter()) {
            let transpose = rotation.transpose();
            let mut local = [[Fix128::ZERO; 3]; NODES_PER_ELEMENT];
            for (slot, &node) in local.iter_mut().zip(element.nodes.iter()) {
                let base = node * 3;
                *slot = mul3(transpose, [u[base], u[base + 1], u[base + 2]]);
            }
            for point in self.points {
                let scale = element.volume * self.weight;
                let grad = shape_gradients(element, point);
                let s = stress_at_local(&grad, &local, lambda, mu);
                for (g, &node) in grad.iter().zip(element.nodes.iter()) {
                    let force = [
                        g[0] * s.xx + g[1] * s.xy + g[2] * s.zx,
                        g[1] * s.yy + g[0] * s.xy + g[2] * s.yz,
                        g[2] * s.zz + g[1] * s.yz + g[0] * s.zx,
                    ];
                    let global = mul3(*rotation, force);
                    let base = node * 3;
                    for (axis, value) in global.iter().enumerate() {
                        out[base + axis] = out[base + axis] + scale * *value;
                    }
                }
            }
        }
    }

    /// `diag(Σₑ R Kₑ⁰ Rᵀ)`.
    ///
    /// Rotating the diagonal needs the whole nodal block, not just its
    /// diagonal: `diag(R K Rᵀ)_a = Σ_{m,n} R_{am} K_{mn} R_{an}` mixes every
    /// entry. The same expression `linear_elastic_fem::rotated_stiffness_diagonal`
    /// uses, summed over the quadrature points instead of taken once.
    fn diagonal(&self, ndof: usize) -> Vec<Fix128> {
        let (lambda, mu) = self.lame;
        let lambda_2mu = lambda + mu + mu;
        let lambda_mu = lambda + mu;
        let mut diag = vec![Fix128::ZERO; ndof];
        for (element, rotation) in self.elements.iter().zip(self.rotations.iter()) {
            let r = [
                [rotation.col0.x, rotation.col1.x, rotation.col2.x],
                [rotation.col0.y, rotation.col1.y, rotation.col2.y],
                [rotation.col0.z, rotation.col1.z, rotation.col2.z],
            ];
            for point in self.points {
                let scale = element.volume * self.weight;
                let grad = shape_gradients(element, point);
                for (g, &node) in grad.iter().zip(element.nodes.iter()) {
                    let sq = [g[0] * g[0], g[1] * g[1], g[2] * g[2]];
                    let mut block = [[Fix128::ZERO; 3]; 3];
                    for (m, row) in block.iter_mut().enumerate() {
                        for (n, entry) in row.iter_mut().enumerate() {
                            *entry = if m == n {
                                scale
                                    * (lambda_2mu * sq[m]
                                        + mu * (sq[(m + 1) % 3] + sq[(m + 2) % 3]))
                            } else {
                                scale * (lambda_mu * g[m] * g[n])
                            };
                        }
                    }
                    let base = node * 3;
                    for (a, r_row) in r.iter().enumerate() {
                        let mut acc = Fix128::ZERO;
                        for (m, block_row) in block.iter().enumerate() {
                            for (n, entry) in block_row.iter().enumerate() {
                                acc = acc + r_row[m] * *entry * r_row[n];
                            }
                        }
                        diag[base + a] = diag[base + a] + acc;
                    }
                }
            }
        }
        diag
    }
}

/// `M v` for a 3-vector held as an array.
#[inline]
fn mul3(m: Mat3Fix, v: [Fix128; 3]) -> [Fix128; 3] {
    let out = m.mul_vec(Vec3Fix::new(v[0], v[1], v[2]));
    [out.x, out.y, out.z]
}

/// `σ = D B u_e` at one point, from nodal displacements already gathered (and,
/// on the co-rotational path, already rotated into the element frame).
fn stress_at_local(
    grad: &[[Fix128; 3]; NODES_PER_ELEMENT],
    local: &[[Fix128; 3]; NODES_PER_ELEMENT],
    lambda: Fix128,
    mu: Fix128,
) -> StressTensor {
    let mut exx = Fix128::ZERO;
    let mut eyy = Fix128::ZERO;
    let mut ezz = Fix128::ZERO;
    let mut gxy = Fix128::ZERO;
    let mut gyz = Fix128::ZERO;
    let mut gzx = Fix128::ZERO;
    for (g, d) in grad.iter().zip(local.iter()) {
        let (ux, uy, uz) = (d[0], d[1], d[2]);
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

/// What one conjugate gradient solve produced.
struct CgOutcome {
    x: Vec<Fix128>,
    iterations: u32,
    residual_norm: Fix128,
    b_norm: Fix128,
    target: Fix128,
}

/// Jacobi-preconditioned conjugate gradient on the free degrees of freedom.
///
/// The same iteration [`solve_quadratic`] runs inline; it is a separate function
/// here because a Newton step runs it repeatedly, and `solve_quadratic` is left
/// untouched so that the linear oracles stay bit for bit where they are.
fn conjugate_gradient<A: Fn(&[Fix128], &mut [Fix128])>(
    apply: A,
    b: &[Fix128],
    is_free: &[bool],
    precond: &[Fix128],
    config: &SolverConfig,
) -> Result<CgOutcome, FemError> {
    let ndof = b.len();
    let mut scratch = vec![Fix128::ZERO; ndof];
    let mut x = vec![Fix128::ZERO; ndof];
    let mut r = b.to_vec();
    let mut z = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            z[d] = r[d] * precond[d];
        }
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(b, b).sqrt();
    let requested = config.relative_tolerance() * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };

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
        apply(&p, &mut scratch);
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

    Ok(CgOutcome {
        x,
        iterations,
        residual_norm,
        b_norm,
        target,
    })
}

/// The Jacobi preconditioner for a given stiffness diagonal, or all ones.
fn preconditioner(diag: &[Fix128], is_free: &[bool], mode: Preconditioner) -> Vec<Fix128> {
    let ndof = diag.len();
    match mode {
        Preconditioner::None => vec![Fix128::ONE; ndof],
        Preconditioner::JacobiScaled => {
            let mut sum = Fix128::ZERO;
            let mut free = 0i64;
            for (d, value) in diag.iter().enumerate() {
                if is_free[d] {
                    sum = sum + *value;
                    free += 1;
                }
            }
            if free == 0 || sum.is_zero() {
                return vec![Fix128::ONE; ndof];
            }
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
}

/// Package a hyperelastic solution: the displacement field and the **Cauchy**
/// stress of the material law at each element centroid.
///
/// ⚠️ The stress reported here is `cauchy_stress` of the law, **not** the
/// small-strain `stress_at` that [`finish`] reports. At the stretches this
/// solver exists for the two differ by more than the discretisation error, so
/// reporting the linear one would make the field agree with a law the solve did
/// not use.
fn finish_hyperelastic(
    mesh: &QuadraticMesh,
    u: &[Fix128],
    law: MaterialLaw,
    cg: (u32, Fix128, Fix128),
) -> Result<FemSolution, FemError> {
    let quarter = half() * half();
    let centroid = [quarter; 4];
    let displacements = (0..mesh.node_count())
        .map(|n| [u[n * 3], u[n * 3 + 1], u[n * 3 + 2]])
        .collect();
    let mut element_stress = Vec::with_capacity(mesh.elements.len());
    for (tet, element) in mesh.elements.iter().enumerate() {
        let grad = shape_gradients(element, &centroid);
        let gradient = deformation_gradient_at(element, &grad, u);
        let (cauchy, _) = hyperelastic_stress(&law.model, law.bulk_modulus, gradient).ok_or(
            FemError::RotationFailed {
                tet,
                cause: PolarError::Inverted,
            },
        )?;
        element_stress.push(cauchy);
    }
    let (iterations, relative_residual, effective_relative_tolerance) = cg;
    Ok(FemSolution {
        displacements,
        element_stress,
        iterations,
        relative_residual,
        effective_relative_tolerance,
    })
}

/// Solve the **finite-strain** boundary value problem on a quadratic mesh under
/// a hyperelastic law.
///
/// `boundary` indexes **nodes of `mesh`**, with the same caution
/// [`solve_quadratic`] gives: constraining only the corners of a face leaves the
/// edge nodes on it free, which is a different problem.
///
/// # The iteration
///
/// Each step solves a **co-rotational small-strain** operator `A = Σₑ R Kₑ⁰ Rᵀ`
/// for the displacement itself, with the right hand side carrying the
/// difference between what that operator would exert and what the material law
/// does:
///
/// ```text
/// A·u_free = f_ext − f_lin(bf) + f_lin(u) − f_mat(u)
/// ```
///
/// Substituting `f_lin(u) = f_lin(bf) + A·u_free` collapses it to
/// `f_ext = f_mat(u)`: **a fixed point is equilibrium under the material law,
/// exactly**, with `A` deciding only how many steps it takes to get there —
/// which is why `A` is allowed to be an approximation at all. The scheme is the
/// one `solve_corotational` uses for P1 with a material law set.
///
/// ⚠️ **The frames are in the tangent, not in the stress.** A hyperelastic
/// stress is objective on its own, so nothing is rotated into place when the
/// internal force is formed; `R` appears only in the operator that steps
/// towards the root. Dropping it costs no accuracy and a great deal of
/// convergence: with the plain small-strain operator a superposed rotation of
/// five degrees does not converge in 200 Newton steps, and the residual is bit
/// identical at four increments and at sixteen, so it is the contraction of the
/// iteration and not the step size that fails. Measured in
/// `tests/analytic_quadratic_hyperelastic.rs`.
///
/// ⚠️ `CorotationalConfig::polar_iterations` **is** read on this path — it is
/// the budget for the one polar decomposition per element per Newton step.
///
/// ⚠️ **A prediction is carried between increments**, the previous converged
/// field scaled by the ratio of the two load factors, exactly as
/// `solve_corotational` does and for the same reason: the frames of the first
/// step of an increment are read off it.
///
/// # What the increments are for
///
/// `A` is the small-strain stiffness, so the correction term grows with the
/// stretch and the iteration slows as it does. Applying the prescribed
/// displacement in fractions keeps each solve near a state the previous one
/// already found, and keeps `det F` positive on the way — a single step to a
/// large stretch can turn an element inside out at the prediction and be
/// refused outright.
///
/// # Errors
///
/// See [`FemError`]. [`FemError::InvalidConfig`] when the configuration carries
/// no material law — this entry point exists for the law and silently falling
/// back to the linear one would report a solve that was never run.
/// [`FemError::RotationFailed`] with [`PolarError::Inverted`] when an element
/// reaches a non-positive `det F`, at which point no hyperelastic law has a
/// stress.
pub fn solve_quadratic_hyperelastic(
    mesh: &QuadraticMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &CorotationalConfig,
) -> Result<CorotationalSolution, FemError> {
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
    let model = config.hyperelastic().ok_or(FemError::InvalidConfig(
        "solve_quadratic_hyperelastic needs CorotationalConfig::with_hyperelastic",
    ))?;

    let (lambda, mu) = material.lame();
    // `κ = λ − offset(model)` — the same choice `solve_corotational` makes.
    let law = MaterialLaw {
        model,
        bulk_modulus: hyperelastic_volumetric_modulus(&model, lambda)?,
    };
    let ndof = node_count * 3;
    let points = quadrature_points();
    let weight = quadrature_weight();
    let mut op = RotatedOperator::new(&mesh.elements, &points, weight, (lambda, mu));

    let mut prescribed_value = vec![Fix128::ZERO; ndof];
    let mut is_free = vec![true; ndof];
    for &(node, axis, value) in boundary.prescribed() {
        let d = node as usize * 3 + axis.index();
        prescribed_value[d] = value;
        is_free[d] = false;
    }
    if is_free.iter().filter(|f| !**f).count() < 6 {
        return Err(FemError::UnderConstrained);
    }

    let mut f_ext = vec![Fix128::ZERO; ndof];
    for &(node, axis, force) in boundary.loads() {
        let d = node as usize * 3 + axis.index();
        if is_free[d] {
            f_ext[d] = f_ext[d] + force;
        }
    }

    if is_free.iter().all(|f| !*f) {
        // Fully prescribed: the answer is the boundary data itself, and the
        // stress is the law read off it. Nothing was relaxed, so the tolerance
        // honoured is the one that was asked for.
        let field = finish_hyperelastic(
            mesh,
            &prescribed_value,
            law,
            (0, Fix128::ZERO, config.linear().relative_tolerance()),
        )?;
        return Ok(CorotationalSolution {
            field,
            newton_iterations: 0,
            increments: config.increments(),
        });
    }

    // The reference residual: the full prescribed displacement read by the
    // small-strain operator. It does not depend on the increment count or the
    // Newton budget, so neither does the threshold built from it.
    let mut scratch = vec![Fix128::ZERO; ndof];
    // ⚠️ The reference is read by the **unrotated** operator, exactly as
    // `solve_corotational` does: it fixes the Newton threshold and must not
    // depend on the frames, which change with every step.
    apply_stiffness(
        &mesh.elements,
        &points,
        weight,
        &prescribed_value,
        lambda,
        mu,
        &mut scratch,
    );
    let mut reference = f_ext.clone();
    for (d, value) in reference.iter_mut().enumerate() {
        if is_free[d] {
            *value = *value - scratch[d];
        } else {
            *value = Fix128::ZERO;
        }
    }
    let newton_target = max_abs(&reference) * config.newton_tolerance();

    let mut boundary_field = vec![Fix128::ZERO; ndof];
    let mut u = vec![Fix128::ZERO; ndof];
    let mut material_force = vec![Fix128::ZERO; ndof];
    let mut residual = vec![Fix128::ZERO; ndof];
    let mut newton_iterations = 0u32;
    let mut cg_iterations = 0u32;
    let mut relative_residual = Fix128::ZERO;
    let mut effective_relative_tolerance = Fix128::ZERO;

    // Jacobi scaling for the Newton–Krylov step; the diagonal of the small-strain
    // operator at the undeformed frames is a fixed, positive scaling.
    let krylov_precond = preconditioner(
        &op.diagonal(ndof),
        &is_free,
        config.linear().preconditioner(),
    );

    for increment in 1..=config.increments() {
        // Predict the whole field, not only the boundary.
        //
        // ⚠️ With the **rotated** tangent this is load bearing, where with a
        // plain small-strain one it was not. The frames of the first step of an
        // increment are built from whatever `u` holds at that moment; moving
        // the prescribed nodes and leaving the interior where the last
        // increment left it flattens the deformation gradient of an element
        // straddling the boundary, and the frame that comes out of that is not
        // a frame of any state the iteration is near. Carrying the previous
        // converged field forward by the ratio of the two load factors keeps
        // the interior with the boundary.
        if increment > 1 {
            let ratio =
                Fix128::from_int(i64::from(increment)) / Fix128::from_int(i64::from(increment - 1));
            for (d, value) in u.iter_mut().enumerate() {
                if is_free[d] {
                    *value = *value * ratio;
                }
            }
        }
        boundary_field.fill(Fix128::ZERO);
        // The last increment carries the prescribed values through unscaled, so
        // the boundary data the answer is built on is bit for bit what the
        // caller asked for however many increments there were.
        if increment == config.increments() {
            for (d, value) in prescribed_value.iter().enumerate() {
                if !is_free[d] {
                    boundary_field[d] = *value;
                }
            }
        } else {
            let scale = Fix128::from_int(i64::from(increment))
                / Fix128::from_int(i64::from(config.increments()));
            for (d, value) in prescribed_value.iter().enumerate() {
                if !is_free[d] {
                    boundary_field[d] = *value * scale;
                }
            }
        }
        for (d, value) in boundary_field.iter().enumerate() {
            if !is_free[d] {
                u[d] = *value;
            }
        }

        let mut step = 0u32;
        loop {
            if step > 0 {
                material_internal_force(
                    &mesh.elements,
                    &points,
                    weight,
                    &u,
                    law,
                    &mut material_force,
                )?;
                for (d, value) in residual.iter_mut().enumerate() {
                    *value = if is_free[d] {
                        f_ext[d] - material_force[d]
                    } else {
                        Fix128::ZERO
                    };
                }
                if max_abs(&residual) <= newton_target {
                    break;
                }
            }
            if step >= config.newton_iterations() {
                return Err(FemError::NotConverged {
                    iterations: step,
                    relative_residual: relative(max_abs(&residual), newton_target),
                });
            }

            // `with_consistent_tangent`: a Newton–Krylov step whose tangent action is
            // the central difference of the material internal force. The modified
            // iteration below contracts by a factor that grows with the stretch and
            // does not converge once it passes one (measured: a smooth non-affine
            // field at `|∇u| ≈ 0.4` ends `NotConverged` at 400 steps and at 16
            // increments alike), so the tangent is the thing that has to change.
            // Step 0 is a prediction, not a candidate, and keeps the linear solve.
            if config.consistent_tangent() && step > 0 {
                if let Some(report) =
                    crate::linear_elastic_fem::consistent_tangent::newton_krylov_step(
                        &mut u,
                        &f_ext,
                        &is_free,
                        &krylov_precond,
                        &config.linear(),
                        |x, out| {
                            material_internal_force(&mesh.elements, &points, weight, x, law, out)
                        },
                    )?
                {
                    cg_iterations = cg_iterations.saturating_add(report.cg_iterations);
                    relative_residual = report.relative_residual;
                    effective_relative_tolerance = report.effective_relative_tolerance;
                    step += 1;
                    newton_iterations = newton_iterations.saturating_add(1);
                    continue;
                }
            }

            // The frames come from the current displacement and are then held
            // for the whole step, which is what keeps `f_lin` affine in `u`
            // within the step and therefore keeps the step exact for the
            // linear law.
            //
            // ⚠️ **Not at step 0.** What `u` holds there is a *prediction* —
            // at increment 1 it is the undeformed interior with the boundary
            // already moved — and a frame read off a prediction is a frame of
            // a state the iteration is not near. Keeping the frames already in
            // hand (the identity at the start, the previous increment's
            // otherwise) makes the first solve of an increment a small-strain
            // solve from a guess that carries no rotation information yet,
            // which is what `solve_corotational` does for the same reason.
            if step > 0 {
                op.refresh(&u, config.polar_iterations());
            }
            let precond = preconditioner(
                &op.diagonal(ndof),
                &is_free,
                config.linear().preconditioner(),
            );

            // `f_ext − f_lin(bf)`, the right hand side a linear solve would use.
            op.apply(&boundary_field, &mut scratch);
            for (d, value) in residual.iter_mut().enumerate() {
                *value = if is_free[d] {
                    f_ext[d] - scratch[d]
                } else {
                    Fix128::ZERO
                };
            }
            // `+ f_lin(u) − f_mat(u)`, which turns the exact step of the linear
            // law into a modified Newton step for the material one. Skipped at
            // step 0, where `u` is a boundary field and not yet a candidate.
            if step > 0 {
                op.apply(&u, &mut scratch);
                material_internal_force(
                    &mesh.elements,
                    &points,
                    weight,
                    &u,
                    law,
                    &mut material_force,
                )?;
                for (d, value) in residual.iter_mut().enumerate() {
                    if is_free[d] {
                        *value = *value + scratch[d] - material_force[d];
                    }
                }
            }

            let cg = conjugate_gradient(
                |p, out| op.apply(p, out),
                &residual,
                &is_free,
                &precond,
                &config.linear(),
            )?;
            cg_iterations = cg_iterations.saturating_add(cg.iterations);
            relative_residual = relative(cg.residual_norm, cg.b_norm);
            effective_relative_tolerance = relative(cg.target, cg.b_norm);
            for (d, value) in u.iter_mut().enumerate() {
                *value = if is_free[d] {
                    cg.x[d]
                } else {
                    boundary_field[d]
                };
            }
            step += 1;
            newton_iterations = newton_iterations.saturating_add(1);
        }
    }

    // Asked once, on the answer, so that it cannot decide *where* the iteration
    // stops — only whether what it stopped on is acceptable.
    material_internal_force(
        &mesh.elements,
        &points,
        weight,
        &u,
        law,
        &mut material_force,
    )?;
    for (d, value) in residual.iter_mut().enumerate() {
        *value = if is_free[d] {
            f_ext[d] - material_force[d]
        } else {
            Fix128::ZERO
        };
    }
    let final_residual = max_abs(&residual);
    if final_residual > newton_target {
        return Err(FemError::NotConverged {
            iterations: newton_iterations,
            relative_residual: relative(final_residual, newton_target),
        });
    }

    let field = finish_hyperelastic(
        mesh,
        &u,
        law,
        (
            cg_iterations,
            relative_residual,
            effective_relative_tolerance,
        ),
    )?;
    Ok(CorotationalSolution {
        field,
        newton_iterations,
        increments: config.increments(),
    })
}

// ---------------------------------------------------------------------------
// reactions
// ---------------------------------------------------------------------------

/// Support forces at the prescribed degrees of freedom of a [`solve_quadratic`]
/// or [`solve_quadratic_hyperelastic`] answer (N), indexed like the nodes of
/// `mesh`.
///
/// # What it is
///
/// ```text
/// R_d = f_int(u)_d − f_ext_d     on a prescribed degree of freedom
/// R_d = 0                        on a free one
/// ```
///
/// with `f_int(u) = K u` when `law` is `None` (the small-strain internal force
/// [`solve_quadratic`] balances) and `f_int(u) = Σₑ Σ_q w_q V₀ P ∇₀N` when it is
/// `Some` (the total-Lagrangian internal force [`solve_quadratic_hyperelastic`]
/// balances, with the volumetric modulus `κ = λ − offset(model)` that solver pairs the model
/// with). The sign is the force the body exerts on its supports, so that
/// `Σ R + Σ f_ext = 0` over the whole mesh. A free row carries no reaction by
/// construction: that row *is* the equilibrium equation the solve satisfied.
///
/// `law` has to be the one the solution was produced under. A hyperelastic
/// answer read with `None` reports the support forces of a linear body held at
/// the same displacement — a different problem, with a different answer.
///
/// # ⚠️ Why this exists for the higher-order elements in particular
///
/// Measured 2026-10-01 on this element and the cubic one: dropping the `J`
/// from `P = J σ F⁻ᵀ` in `hyperelastic_stress` left **7 of 7** oracles in
/// `tests/analytic_quadratic_hyperelastic.rs` green, including the one asserting
/// a closed form for `σ_xx`. With `f_ext = 0` the discrete problem is
/// `f_mat(u) = 0` on the free rows, so a uniform scale on the internal force
/// does not move the root, and the reported stress is the Cauchy `σ` from
/// *before* the Piola conversion. The reaction is linear in that scale, and it
/// is the only quantity on this module's surface that is.
///
/// # ⚠️ The shared assembly is load-bearing — do not re-derive it here
///
/// This goes through `apply_stiffness` and `material_internal_force` (both
/// private), the very functions the two solvers build their systems with, and
/// **not** through a second copy of the same formulae. That sharing is what
/// makes the closed-form oracle work: a mistake in the element integral has to
/// *reach* this value before anything can compare it against the analytic
/// traction. Re-deriving it here would delete the oracle while every test
/// stayed green.
///
/// # ⚠️ What it does not check
///
/// Nothing ties `solution` to the arguments beside it. Only the node count is
/// checked. A load placed on a prescribed degree of freedom is subtracted here
/// even though the solvers ignore it — the support carries it.
///
/// # Errors
///
/// [`FemError::EmptyMesh`], [`FemError::VertexOutOfRange`] for boundary data
/// naming a node the mesh does not have,
/// [`FemError::SolutionDoesNotMatchMesh`] when `solution` has a different
/// node count, and — with a law — [`FemError::RotationFailed`] with
/// [`PolarError::Inverted`] at a quadrature point where `det F ≤ 0`, where no
/// hyperelastic law has a stress.
#[must_use = "the support forces are the whole point of calling this; a solve that \
     drops them has not been checked against equilibrium at all"]
pub fn reactions(
    mesh: &QuadraticMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    law: Option<HyperelasticModel>,
    solution: &FemSolution,
) -> Result<Vec<[Fix128; 3]>, FemError> {
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
    let nodes = solution.displacements.len();
    if nodes != node_count {
        return Err(FemError::SolutionDoesNotMatchMesh {
            nodes,
            vertex_count: node_count,
        });
    }

    let (lambda, mu) = material.lame();
    let ndof = node_count * 3;
    let points = quadrature_points();
    let weight = quadrature_weight();

    let mut u = vec![Fix128::ZERO; ndof];
    for (n, d) in solution.displacements.iter().enumerate() {
        u[n * 3] = d[0];
        u[n * 3 + 1] = d[1];
        u[n * 3 + 2] = d[2];
    }

    // `f_int`, through the function the matching solver balances.
    let mut force = vec![Fix128::ZERO; ndof];
    match law {
        None => apply_stiffness(&mesh.elements, &points, weight, &u, lambda, mu, &mut force),
        Some(model) => {
            // The same volumetric modulus `solve_quadratic_hyperelastic` pairs the
            // model with.
            let law = MaterialLaw {
                model,
                bulk_modulus: hyperelastic_volumetric_modulus(&model, lambda)?,
            };
            material_internal_force(&mesh.elements, &points, weight, &u, law, &mut force)?;
        }
    }

    for &(node, axis, applied) in boundary.loads() {
        let d = node as usize * 3 + axis.index();
        force[d] = force[d] - applied;
    }
    let mut is_free = vec![true; ndof];
    for &(node, axis, _) in boundary.prescribed() {
        is_free[node as usize * 3 + axis.index()] = false;
    }
    for (d, value) in force.iter_mut().enumerate() {
        if is_free[d] {
            *value = Fix128::ZERO;
        }
    }
    Ok((0..node_count)
        .map(|n| [force[n * 3], force[n * 3 + 1], force[n * 3 + 2]])
        .collect())
}

/// What an adaptive quadratic run produced.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveQuadraticSolution {
    /// The final straight-edged corner mesh, refined where the estimator asked.
    pub mesh: SdfTetMesh,
    /// The quadratic mesh built from [`Self::mesh`] — the one `field` is indexed by.
    pub high_order: QuadraticMesh,
    /// The solution on that mesh.
    pub field: FemSolution,
    /// Solves performed, at least one and at most `max_rounds`.
    pub rounds: u32,
    /// `Σ_e η_e²` after each solve, oldest first.
    pub total_indicator_history: Vec<Fix128>,
}

/// Solve, estimate, mark, refine, repeat — on the quadratic element.
///
/// The quadratic counterpart of [`crate::linear_elastic_fem::solve_adaptive`].
/// Refinement is performed on the straight-edged corner mesh by
/// [`SdfTetMesh::try_refine_marked`] and the QuadraticMesh is rebuilt from it each
/// round, so the conforming guarantee of that refinement carries over unchanged.
/// The error indicator is the P1 stress-recovery estimator applied to the
/// element's centroid stress (see `corner_indicators_squared`).
///
/// ⚠️ `boundary_for` receives the **quadratic mesh**, not the corner mesh: a
/// higher-order boundary condition has to constrain the edge (and face) nodes of
/// the clamped faces too, and only that mesh knows where they are. As in the
/// linear driver the conditions are rebuilt per mesh, because a nodal load cannot
/// be split between children.
///
/// # Errors
///
/// Whatever [`solve_quadratic`], the estimator or `mark_bulk` return, and
/// [`FemError::InvalidConfig`] if a refinement pass runs out of budget.
pub fn solve_adaptive_quadratic<F>(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary_for: F,
    adaptive: &crate::linear_elastic_fem::AdaptiveConfig,
) -> Result<AdaptiveQuadraticSolution, FemError>
where
    F: Fn(&QuadraticMesh) -> BoundaryConditions,
{
    let (corner, field, rounds, total_indicator_history) =
        crate::linear_elastic_fem::adaptive_refinement_loop(mesh, adaptive, |current| {
            let high_order = QuadraticMesh::from_tet_mesh(current)?;
            let boundary = boundary_for(&high_order);
            let field = solve_quadratic(&high_order, material, &boundary, &adaptive.linear())?;
            let indicators =
                crate::linear_elastic_fem::corner_indicators_squared(current, material, &field)?;
            Ok((field, indicators))
        })?;
    let high_order = QuadraticMesh::from_tet_mesh(&corner)?;
    Ok(AdaptiveQuadraticSolution {
        mesh: corner,
        high_order,
        field,
        rounds,
        total_indicator_history,
    })
}
