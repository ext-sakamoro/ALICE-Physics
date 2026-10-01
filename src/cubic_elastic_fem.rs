//! Small-strain linear elastic FEM on **twenty-node cubic tetrahedra** (P3).
//!
//! The third element in this family, after [`crate::linear_elastic_fem`] (P1,
//! four nodes, constant strain) and [`crate::quadratic_elastic_fem`] (P2, ten
//! nodes, linear strain). All three are kept and none of them changes the others
//! by one bit: [`solve_cubic`] is a separate entry point, and the P1 and P2
//! oracle families are untouched.
//!
//! # Why a third element
//!
//! P2 reproduces a quadratic displacement field exactly; P3 reproduces a
//! **cubic** one. That is a statement about the function space, so it holds on
//! any mesh at any resolution, and `tests/analytic_cubic_fem.rs` asserts it with
//! a cubic field that P2 misses by four orders of magnitude on the same mesh.
//! Where it pays off in practice is bending: the through-thickness strain of a
//! bent plate is linear, so its displacement is cubic, and a single P3 element
//! through the thickness carries it.
//!
//! # The nodes live here, not in the mesh
//!
//! As for P2. [`crate::sdf_fem_mesh::SdfTetMesh`] stays P1 — its `vertices` are
//! the lattice corners and `tests/analytic_fem_convergence.rs` asserts that count
//! against the closed form for the lattice. The sixteen extra nodes of each
//! element (two per edge, one per face) are built here by
//! [`CubicMesh::from_tet_mesh`] and numbered **after** the corners, so a corner
//! keeps the index it has in the tetrahedral mesh.
//!
//! ⚠️ Twelve edge nodes, not six: a cubic element puts **two** nodes on each
//! edge, at one and two thirds. [`CubicMesh::edge_nodes`] returns them as a pair
//! ordered from the lower-numbered corner, because "the edge node" is no longer
//! a well-defined phrase.
//!
//! # Quadrature — why this rule and not the textbook one
//!
//! On a straight-edged tetrahedron the Jacobian is constant and the shape
//! function gradients are **quadratic** in the barycentric coordinates, so
//! `BᵀDB` is **quartic** and a degree-4 rule integrates the element stiffness
//! exactly. P2 needed degree 2; this is two degrees further up, and in a
//! fixed-point type that is not free.
//!
//! ⚠️ **No tetrahedral rule of degree ≥ 2 can be fully exact in `Fix128`.** The
//! moments it has to match are `1/10` at degree 2 and `1/35`, `1/140`, `1/210`,
//! `1/420`, `1/840` at degree 4, and **none of those is a dyadic rational**. A
//! rule with dyadic points and dyadic weights computes a sum of products of
//! dyadics, which is dyadic, so it can never land on them. The question is
//! therefore not "is there an exact rule" but "how few roundings can this be
//! done in" — and `tests/p3_quadrature_fix128.rs` answers it by measurement:
//!
//! | rule | degree | points | worst ulp within its degree |
//! |---|---|---|---|
//! | Hammer–Stroud 4pt (what P2 uses) | 2 | 4 | −2 |
//! | Keast 11pt (the textbook degree-4 rule) | 4 | 11 | −11 |
//! | **the rule below** | 4 (5 in fact) | 24 | **−7** |
//!
//! So P3 costs a factor of 3.5 against P2 in the quadrature constants, not a
//! factor of a thousand.
//!
//! ⚠️ **The rule used here is not from the literature**, and the reason to
//! prefer it over Keast's eleven points is not the 7-against-11 above. It is that
//! every one of its abscissae is dyadic, and **so is every coefficient of the
//! cubic shape functions** (`1/2`, `9/2`, `27`). Together those mean the twenty
//! shape functions and their barycentric derivatives evaluate at the quadrature
//! points **with no rounding at all** — measured as a partition of unity exact to
//! **0 ulp** at all 24 points, against a non-zero error at Hammer–Stroud's and
//! Keast's irrational abscissae. The only rounding left in the integrand comes
//! from the element geometry. That property does not appear in a table of
//! moments, which is why it has a test of its own.
//!
//! ⚠️ **Do not "simplify" this to an all-positive rule.** The obvious heuristic —
//! negative weights cancel, so prefer a rule without them — was measured and is
//! **false here**: over degree-4 rules with dyadic abscissae, the all-positive
//! ones come out at −14 ulp because forcing positivity pushes the weights onto
//! much larger odd denominators (`25279/694575` and the like). One small negative
//! weight (`−16/315`) is cheaper than five badly-rounded positive ones.
//!
//! # ⚠️ A scene constraint P2 did not have, and what the API does about it
//!
//! The edge nodes sit at `(2a + b)/3` and the face nodes at `(a + b + c)/3`, and
//! **`1/3` is not dyadic**. P2's edge midpoints `(a + b)/2` were exact on any
//! mesh; P3's interior nodes are exact only when the corner coordinates are
//! multiples of three. A mesh on a 3 mm lattice has exact node positions; a mesh
//! on a 1 mm lattice does not.
//!
//! ⚠️ **It would be silent.** The division is a `Fix128` division, so an
//! inexact third is simply truncated and nothing reports it — the failure mode
//! where the implementation quietly degrades below what the documentation
//! promises. Rather than leave that to prose,
//! [`CubicMesh::interior_node_positions_are_exact`] answers it for the mesh that
//! was actually built, and a caller that needs exact node positions can assert
//! on it. `tests/analytic_cubic_fem.rs` measures both sides: `true` with zero ulp
//! of error on a 3 mm lattice, `false` with a non-zero error on a 1 mm one.
//!
//! The flag covers the **reported node positions only**, which is the whole
//! extent of the problem: the element geometry (`∇λ`, the volume) is built from
//! the four corners, so the thirds never enter the stiffness. What they do affect
//! is any caller — an oracle prescribing a closed-form field, a post-processor
//! plotting the field — that asks where a node is.
//!
//! # ⚠️⚠️ Three exactness properties, and only two can be had at once
//!
//! | | exact thing | condition | conflicts with |
//! |---|---|---|---|
//! | **A** | node positions `(2a+b)/3`, `(a+b+c)/3` | every edge vector is `3 ×` dyadic | ⚠️ **B** |
//! | **B** | `∇λ`, the only geometric quantity the integrand reads | the inverse edge matrix is dyadic (axis-aligned: legs a power of two) | ⚠️ **A**, **C** |
//! | **C** | the oracle's teeth — a lower-order element must *fail* the same field | the lattice is **perturbed** | ⚠️ **B** |
//!
//! **A against B is not a measurement, it is a proof.** A needs the edge matrix
//! to be `3D` with `D` dyadic, and then `∇λ = adj(3D)/det(3D) = adj(D)/(3·det D)`
//! has determinant `1/(27·det D)`, which cannot be dyadic because `det D` is.
//! Conversely dyadic `∇λ` puts the legs on powers of two and the thirds stop
//! dividing. Pinned by construction in
//! `the_two_exactness_properties_cannot_both_hold`.
//!
//! ⚠️ **This is specific to P3 among the three elements here.** The mechanism —
//! an inexact `∇λ` rounding the product `s·∇λ` — is shared, but P1 has nodes only
//! at the corners and P2's are at `(a+b)/2` where `1/2` *is* dyadic, so neither
//! has any reason to want a lattice of multiples of three and the conflict never
//! arises. The general statement is that **an element whose node placement needs
//! a denominator with an odd factor cannot have both exact node positions and
//! exact barycentric gradients.**
//!
//! Which two each test takes:
//!
//! | test | takes | and therefore does not measure |
//! |---|---|---|
//! | `dyadic_abscissae_lower_the_rounding_of_the_whole_assembly` (unit) | **B** | node positions; it uses a constant field, which needs none |
//! | `the_shape_gradient_identity_degrades_with_the_element_geometry` (unit) | **A** | bit exactness — it *measures the loss*, 110 raw units, and fails if it ever becomes zero |
//! | `tests/analytic_cubic_fem.rs` exactness oracles | **C** | bit exactness; they assert against the conjugate gradient's floor, not against zero |
//!
//! ⚠️ So **"the dyadic rule rounds less" holds only on elements with dyadic
//! `∇λ`**, and the oracle that proves the element is a cubic one runs on a mesh
//! where that does not hold. Neither claim covers the other, and neither is
//! stated more widely than it was measured.
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

/// Nodes per element: four corners, then twelve edge nodes, then four face
/// nodes.
const NODES_PER_ELEMENT: usize = 20;

/// Quadrature points, and therefore weights, per element.
const QUADRATURE_POINTS: usize = 24;

/// The six edges of a tetrahedron, as corner index pairs.
///
/// Edge `e` owns element slots `4 + 2e` (the node two thirds of the way to
/// `EDGES[e].0`) and `5 + 2e` (two thirds of the way to `EDGES[e].1`). Fixed and
/// index-ordered, so two runs on the same mesh produce the same node indices and
/// the same assembly order.
const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// The four faces of a tetrahedron, as corner index triples, face `f` being the
/// one opposite corner `f`. Face `f` owns element slot `16 + f`.
const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];

/// `numer / 2^shift`, built from the bit pattern: no division, exact by
/// construction.
///
/// Every abscissa of the quadrature rule and every coefficient of the shape
/// functions goes through here, which is what makes their evaluation exact.
fn dyadic(numer: i64, shift: u32) -> Fix128 {
    let r = (numer as i128) << (64 - shift);
    Fix128::from_raw((r >> 64) as i64, r as u64)
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

/// One symmetric orbit of the quadrature rule.
///
/// `a` and `b` are `(numerator, shift)` pairs for `dyadic`, and `weight` is an
/// exact `(numerator, denominator)` pair — the only constant of the rule that is
/// not dyadic, and therefore the only one that rounds.
struct Orbit {
    /// `true` for an `S31` orbit (four points, `a` once and `b = (1−a)/3` three
    /// times), `false` for an `S22` orbit (six points, `a` twice and
    /// `b = 1/2 − a` twice).
    s31: bool,
    a: (i64, u32),
    b: (i64, u32),
    weight: (i64, i64),
}

/// The five symmetric orbits of the rule.
///
/// | orbit | points | barycentric | weight |
/// |---|---|---|---|
/// | `S31(1)` | 4 | the vertices `(1, 0, 0, 0)` | `1/300` |
/// | `S31(5/8)` | 4 | `(5/8, 1/8, 1/8, 1/8)` | `8/63` |
/// | `S31(1/16)` | 4 | `(1/16, 5/16, 5/16, 5/16)` | `256/1575` |
/// | `S22(1/2)` | 6 | the edge midpoints `(1/2, 1/2, 0, 0)` | `1/45` |
/// | `S22(3/8)` | 6 | `(3/8, 3/8, 1/8, 1/8)` | `−16/315` |
///
/// Obtained by fixing the abscissae on dyadic values and solving the five
/// moment conditions for degree 4 in exact rational arithmetic; the result is
/// exact through degree **5** and breaks at 6. See the module header for why a
/// dyadic-abscissa rule is worth 24 points.
const ORBITS: [Orbit; 5] = [
    Orbit {
        s31: true,
        a: (1, 0),
        b: (0, 0),
        weight: (1, 300),
    },
    Orbit {
        s31: true,
        a: (5, 3),
        b: (1, 3),
        weight: (8, 63),
    },
    Orbit {
        s31: true,
        a: (1, 4),
        b: (5, 4),
        weight: (256, 1575),
    },
    Orbit {
        s31: false,
        a: (1, 1),
        b: (0, 0),
        weight: (1, 45),
    },
    Orbit {
        s31: false,
        a: (3, 3),
        b: (1, 3),
        weight: (-16, 315),
    },
];

/// The six index patterns of an `S22` orbit, in a fixed order.
const S22_PATTERN: [[u8; 4]; 6] = [
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
];

/// Barycentric coordinates of the 24 quadrature points, in orbit order.
fn quadrature_points() -> [[Fix128; 4]; QUADRATURE_POINTS] {
    let mut points = [[Fix128::ZERO; 4]; QUADRATURE_POINTS];
    let mut at = 0usize;
    for orbit in &ORBITS {
        let a = dyadic(orbit.a.0, orbit.a.1);
        let b = dyadic(orbit.b.0, orbit.b.1);
        if orbit.s31 {
            for i in 0..4 {
                let mut p = [b; 4];
                p[i] = a;
                points[at] = p;
                at += 1;
            }
        } else {
            for pattern in S22_PATTERN {
                let mut p = [a; 4];
                for (slot, &s) in p.iter_mut().zip(pattern.iter()) {
                    *slot = if s == 0 { a } else { b };
                }
                points[at] = p;
                at += 1;
            }
        }
    }
    debug_assert!(at == QUADRATURE_POINTS);
    points
}

/// Weights of the 24 quadrature points, in the same order, summing to one.
fn quadrature_weights() -> [Fix128; QUADRATURE_POINTS] {
    let mut weights = [Fix128::ZERO; QUADRATURE_POINTS];
    let mut at = 0usize;
    for orbit in &ORBITS {
        let w = Fix128::from_int(orbit.weight.0) / Fix128::from_int(orbit.weight.1);
        let count = if orbit.s31 { 4 } else { 6 };
        for _ in 0..count {
            weights[at] = w;
            at += 1;
        }
    }
    debug_assert!(at == QUADRATURE_POINTS);
    weights
}

// ---------------------------------------------------------------------------
// the mesh
// ---------------------------------------------------------------------------

/// One cubic element: its twenty global node indices, the (constant) barycentric
/// gradients of the underlying tetrahedron, and its volume.
#[derive(Clone, Copy, Debug)]
struct CubicElement {
    nodes: [usize; NODES_PER_ELEMENT],
    /// `∇λ_i` (1/mm), constant over a straight-edged tetrahedron.
    grad_lambda: [[Fix128; 3]; 4],
    /// `|det J| / 6` (mm³).
    volume: Fix128,
}

/// A [`SdfTetMesh`] with the edge and face nodes a P3 element needs.
///
/// Corners keep their tetrahedral-mesh indices; the sixteen extra nodes per
/// element follow, numbered by first appearance in element order. Build it once
/// and solve on it repeatedly: the node tables and the element geometry do not
/// depend on the material or the boundary data.
#[derive(Clone, Debug)]
pub struct CubicMesh {
    corner_count: usize,
    positions: Vec<[Fix128; 3]>,
    elements: Vec<CubicElement>,
    /// `(min, max)` corner pair → the two edge nodes, the first being the one
    /// nearer `min`.
    ///
    /// `BTreeMap` and not a hash map: the iteration order of a hash map depends
    /// on the hasher and every downstream index would inherit that.
    edge_nodes: BTreeMap<(u32, u32), (u32, u32)>,
    /// Sorted corner triple → face node.
    face_nodes: BTreeMap<(u32, u32, u32), u32>,
    /// Whether every edge and face node position came out of its division by
    /// three exactly. See
    /// [`CubicMesh::interior_node_positions_are_exact`].
    interior_positions_exact: bool,
}

/// `numerator / 3`, together with whether the division was exact.
///
/// ⚠️ **Division, not multiplication by a stored `1/3`.** `1/3` is not
/// representable, so `9 · (1/3 rounded)` truncates to just below `3` and *no*
/// coordinate would come out exact — the instrument would report `false`
/// everywhere and say nothing. `Fix128::Div` computes
/// `floor(|numerator|·2⁶⁴ / 3)`, which is exact whenever the true quotient is.
///
/// The recomputation `q · 3 == numerator` is what makes the inexact case
/// observable instead of silent: a truncated quotient multiplies back to
/// something strictly below the numerator.
fn divide_by_three(numerator: Fix128) -> (Fix128, bool) {
    let three = Fix128::from_int(3);
    let q = numerator / three;
    (q, q * three == numerator)
}

impl CubicMesh {
    /// Build the cubic mesh from a tetrahedral one.
    ///
    /// # Errors
    ///
    /// [`FemError::EmptyMesh`] for a mesh with no vertices or no tetrahedra,
    /// [`FemError::VertexOutOfRange`] for a tetrahedron naming a vertex the mesh
    /// does not have, and [`FemError::DegenerateElement`] for a tetrahedron of
    /// zero volume — the same three the P1 and P2 assemblies report, for the same
    /// reasons.
    pub fn from_tet_mesh(mesh: &SdfTetMesh) -> Result<Self, FemError> {
        let corner_count = mesh.vertices.len();
        if corner_count == 0 || mesh.tets.is_empty() {
            return Err(FemError::EmptyMesh);
        }
        let six = Fix128::from_int(6);
        let two = Fix128::from_int(2);
        let mut interior_positions_exact = true;

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
        let mut edge_nodes: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
        let mut face_nodes: BTreeMap<(u32, u32, u32), u32> = BTreeMap::new();
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

            for (e, &(i, j)) in EDGES.iter().enumerate() {
                let (va, vb) = (tet.vertices[i], tet.vertices[j]);
                let forward = va <= vb;
                let key = if forward { (va, vb) } else { (vb, va) };
                let next = u32::try_from(positions.len()).map_err(|_| FemError::EmptyMesh)?;
                let pair = *edge_nodes.entry(key).or_insert((next, next + 1));
                if pair.0 == next {
                    // The two nodes at one and two thirds, in the order of the
                    // *key*: `pair.0` is nearer the lower-numbered corner. An
                    // edge shared by several elements is therefore built once
                    // and from the same two endpoints however it is reached.
                    let (lo, hi) = if forward { (p[i], p[j]) } else { (p[j], p[i]) };
                    for (first, second) in [(lo, hi), (hi, lo)] {
                        let mut at = [Fix128::ZERO; 3];
                        for (axis, slot) in at.iter_mut().enumerate() {
                            let (q, exact) = divide_by_three(two * first[axis] + second[axis]);
                            *slot = q;
                            interior_positions_exact &= exact;
                        }
                        positions.push(at);
                    }
                }
                let (near_lo, near_hi) = pair;
                let (near_i, near_j) = if forward {
                    (near_lo, near_hi)
                } else {
                    (near_hi, near_lo)
                };
                nodes[4 + 2 * e] = near_i as usize;
                nodes[5 + 2 * e] = near_j as usize;
            }

            for (f, &(i, j, k)) in FACES.iter().enumerate() {
                let mut tri = [tet.vertices[i], tet.vertices[j], tet.vertices[k]];
                tri.sort_unstable();
                let key = (tri[0], tri[1], tri[2]);
                let next = u32::try_from(positions.len()).map_err(|_| FemError::EmptyMesh)?;
                let node = *face_nodes.entry(key).or_insert(next);
                if node == next {
                    let (pa, pb, pc) = (p[i], p[j], p[k]);
                    let mut at = [Fix128::ZERO; 3];
                    for (axis, slot) in at.iter_mut().enumerate() {
                        let (q, exact) = divide_by_three(pa[axis] + pb[axis] + pc[axis]);
                        *slot = q;
                        interior_positions_exact &= exact;
                    }
                    positions.push(at);
                }
                nodes[16 + f] = node as usize;
            }

            elements.push(CubicElement {
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
            face_nodes,
            interior_positions_exact,
        })
    }

    /// Whether every edge and face node position landed on its exact value.
    ///
    /// The edge nodes are at `(2a + b)/3` and the face nodes at `(a + b + c)/3`,
    /// and `1/3` is not representable in a binary fixed-point type, so those
    /// divisions are exact only when the corner coordinates are multiples of
    /// three. On a 3 mm lattice this is `true`; on a 1 mm lattice it is `false`
    /// and each affected coordinate is low by a bit.
    ///
    /// ⚠️ **This exists so the degradation is not silent.** Nothing else reports
    /// it: an inexact third is truncated and the solve proceeds, because the
    /// element geometry comes from the four corners and never touches these
    /// positions. The callers it matters to are the ones that ask *where* a node
    /// is — an oracle prescribing a closed-form field at the boundary, a
    /// post-processor sampling the solution — and such a caller should assert on
    /// this rather than discover a few ulps of unexplained error later.
    #[must_use]
    pub const fn interior_node_positions_are_exact(&self) -> bool {
        self.interior_positions_exact
    }

    /// Total node count: corners plus edge nodes plus face nodes.
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

    /// Number of edge nodes — **two per edge**, so twice the edge count.
    #[must_use]
    pub fn edge_node_count(&self) -> usize {
        self.edge_nodes.len() * 2
    }

    /// Number of face nodes, one per distinct face of the mesh.
    #[must_use]
    pub fn face_node_count(&self) -> usize {
        self.face_nodes.len()
    }

    /// Element count, equal to the source mesh's tetrahedron count.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Position of one node (mm).
    ///
    /// Corners come back exactly as the source mesh holds them. The edge nodes
    /// are at one and two thirds and the face nodes at the face centroid, so
    /// ⚠️ **they carry a rounding unless the corner coordinates are multiples of
    /// three** — `1/3` is not representable in a binary fixed-point type.
    #[must_use]
    pub fn node_position(&self, node: u32) -> Option<[Fix128; 3]> {
        self.positions.get(node as usize).copied()
    }

    /// The two edge nodes between two corners, if that edge is in the mesh,
    /// ordered **from the lower-numbered corner**: the first is at two thirds of
    /// the way from `min(a, b)` to `max(a, b)`'s side — that is, nearer
    /// `min(a, b)`.
    ///
    /// The order of the two arguments does not change the returned pair, which
    /// is why it is defined against the corner numbering rather than against the
    /// call.
    #[must_use]
    pub fn edge_nodes(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.edge_nodes.get(&key).copied()
    }

    /// The face node of the triangle on three corners, if that face is in the
    /// mesh. The order of the three arguments does not matter.
    #[must_use]
    pub fn face_node(&self, a: u32, b: u32, c: u32) -> Option<u32> {
        let mut tri = [a, b, c];
        tri.sort_unstable();
        self.face_nodes.get(&(tri[0], tri[1], tri[2])).copied()
    }

    /// The twenty node indices of one element: four corners, then twelve edge
    /// nodes two per edge in the order `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`,
    /// then four face nodes in the order of the corner each face is opposite.
    ///
    /// Within an edge the first of the two is the node two thirds of the way to
    /// the edge's first corner in that list.
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

/// The twenty cubic shape function values at one barycentric point.
///
/// Corner `i`: `½ λᵢ(3λᵢ − 1)(3λᵢ − 2)`. Edge node two thirds of the way to `i`
/// along `(i, j)`: `9/2 λᵢλⱼ(3λᵢ − 1)`. Face node of `(i, j, k)`:
/// `27 λᵢλⱼλₖ`.
///
/// Public because it is the interpolation this element defines, and a caller
/// that wants a field anywhere other than a node needs it.
///
/// ⚠️ Every coefficient here is dyadic, so at the quadrature points of this
/// module — which are dyadic too — the whole evaluation is exact. `Σ Nᵢ = 1`
/// holds to **0 ulp** there, which `tests/p3_quadrature_fix128.rs` measures
/// against the non-zero error at an irrational-abscissa rule.
#[must_use]
pub fn shape_values(lambda: &[Fix128; 4]) -> [Fix128; NODES_PER_ELEMENT] {
    let half = dyadic(1, 1);
    let nine_halves = dyadic(9, 1);
    let three = Fix128::from_int(3);
    let two = Fix128::from_int(2);
    let twenty_seven = Fix128::from_int(27);
    let mut n = [Fix128::ZERO; NODES_PER_ELEMENT];
    for (slot, &li) in n.iter_mut().zip(lambda.iter()) {
        *slot = half * li * (three * li - Fix128::ONE) * (three * li - two);
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        let (li, lj) = (lambda[i], lambda[j]);
        n[4 + 2 * e] = nine_halves * li * lj * (three * li - Fix128::ONE);
        n[5 + 2 * e] = nine_halves * lj * li * (three * lj - Fix128::ONE);
    }
    for (f, &(i, j, k)) in FACES.iter().enumerate() {
        n[16 + f] = twenty_seven * lambda[i] * lambda[j] * lambda[k];
    }
    n
}

/// `∇N` for all twenty shape functions at one barycentric point.
///
/// Differentiating the expressions in [`shape_values`]:
/// corner `i` gives `½(27λᵢ² − 18λᵢ + 2)∇λᵢ`;
/// the edge node nearer `i` gives `9/2[λⱼ(6λᵢ − 1)∇λᵢ + λᵢ(3λᵢ − 1)∇λⱼ]`;
/// the face node gives `27(λⱼλₖ∇λᵢ + λᵢλₖ∇λⱼ + λᵢλⱼ∇λₖ)`.
fn shape_gradients(
    element: &CubicElement,
    lambda: &[Fix128; 4],
) -> [[Fix128; 3]; NODES_PER_ELEMENT] {
    let half = dyadic(1, 1);
    let nine_halves = dyadic(9, 1);
    let two = Fix128::from_int(2);
    let three = Fix128::from_int(3);
    let six = Fix128::from_int(6);
    let eighteen = Fix128::from_int(18);
    let twenty_seven = Fix128::from_int(27);

    let mut grad = [[Fix128::ZERO; 3]; NODES_PER_ELEMENT];
    for (i, (out, gl)) in grad
        .iter_mut()
        .take(4)
        .zip(element.grad_lambda.iter())
        .enumerate()
    {
        let li = lambda[i];
        let s = half * (twenty_seven * li * li - eighteen * li + two);
        for (o, c) in out.iter_mut().zip(gl.iter()) {
            *o = s * *c;
        }
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        let (li, lj) = (lambda[i], lambda[j]);
        let (gi, gj) = (element.grad_lambda[i], element.grad_lambda[j]);
        let near_i = (
            nine_halves * lj * (six * li - Fix128::ONE),
            nine_halves * li * (three * li - Fix128::ONE),
        );
        let near_j = (
            nine_halves * li * (six * lj - Fix128::ONE),
            nine_halves * lj * (three * lj - Fix128::ONE),
        );
        for (axis, (ci, cj)) in gi.iter().zip(gj.iter()).enumerate() {
            grad[4 + 2 * e][axis] = near_i.0 * *ci + near_i.1 * *cj;
            grad[5 + 2 * e][axis] = near_j.0 * *cj + near_j.1 * *ci;
        }
    }
    for (f, &(i, j, k)) in FACES.iter().enumerate() {
        let (li, lj, lk) = (lambda[i], lambda[j], lambda[k]);
        let (gi, gj, gk) = (
            element.grad_lambda[i],
            element.grad_lambda[j],
            element.grad_lambda[k],
        );
        for axis in 0..3 {
            grad[16 + f][axis] =
                twenty_seven * (lj * lk * gi[axis] + li * lk * gj[axis] + li * lj * gk[axis]);
        }
    }
    grad
}

/// `σ = D B u_e` at one point inside an element.
fn stress_at(
    element: &CubicElement,
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
    elements: &[CubicElement],
    points: &[[Fix128; 4]; QUADRATURE_POINTS],
    weights: &[Fix128; QUADRATURE_POINTS],
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    out: &mut [Fix128],
) {
    out.fill(Fix128::ZERO);
    for element in elements {
        for (point, weight) in points.iter().zip(weights.iter()) {
            let scale = element.volume * *weight;
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
/// Same per-node expression as the P1 and P2 assemblies —
/// `(λ+2μ)g_a² + μ(g_b² + g_c²)` — summed over the quadrature points, each with
/// its own weight: unlike Hammer–Stroud's, this rule's weights are not all
/// equal, so the weight cannot be hoisted out of the point loop.
fn stiffness_diagonal(
    elements: &[CubicElement],
    points: &[[Fix128; 4]; QUADRATURE_POINTS],
    weights: &[Fix128; QUADRATURE_POINTS],
    lambda: Fix128,
    mu: Fix128,
    ndof: usize,
) -> Vec<Fix128> {
    let mut diag = vec![Fix128::ZERO; ndof];
    let lambda_2mu = lambda + mu + mu;
    for element in elements {
        for (point, weight) in points.iter().zip(weights.iter()) {
            let scale = element.volume * *weight;
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

/// Solve the linear elastic boundary value problem on a cubic mesh.
///
/// `boundary` indexes **nodes of `mesh`**, not vertices of the tetrahedral mesh
/// it came from. Corner indices coincide, so boundary data written for the P1
/// solver constrains the same corners here — but the edge and face nodes on a
/// clamped face are then left free, which is a different problem that still
/// solves. Use [`CubicMesh::edge_nodes`] and [`CubicMesh::face_node`] to reach
/// them.
///
/// # Errors
///
/// See [`FemError`]. [`FemError::UnderConstrained`] covers the case this element
/// makes easiest to hit by accident: a clamped face carries three corners, six
/// edge nodes and one face node, and constraining only the corners leaves seven
/// of the ten free.
pub fn solve_cubic(
    mesh: &CubicMesh,
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
    let weights = quadrature_weights();

    let mut prescribed_value = vec![Fix128::ZERO; ndof];
    let mut is_free = vec![true; ndof];
    for &(node, axis, value) in boundary.prescribed() {
        let d = node as usize * 3 + axis.index();
        prescribed_value[d] = value;
        is_free[d] = false;
    }
    if is_free.iter().all(|f| !*f) {
        // Fully prescribed: the answer is the boundary data itself. Nothing was
        // solved, so nothing was relaxed, and the tolerance that was honoured is
        // the one that was asked for.
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
        &weights,
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
            let diag = stiffness_diagonal(&mesh.elements, &points, &weights, lambda, mu, ndof);
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
            &weights,
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
/// [`crate::linear_elastic_fem::solve`] and
/// [`crate::quadratic_elastic_fem::solve_quadratic`].
///
/// ⚠️ **`element_stress` is evaluated at the element centroid, and for this
/// element that is one sample of a quadratic field.** P1 strain is constant and
/// its per-element stress is the whole truth; P2 strain is linear and the
/// centroid is where the linear part averages out; P3 strain is quadratic, so
/// the centroid is neither the mean nor the peak. A caller looking for the peak
/// stress in a bending element should evaluate [`shape_values`] where it wants
/// it.
fn finish(
    mesh: &CubicMesh,
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    iterations: u32,
    relative_residual: Fix128,
    effective_relative_tolerance: Fix128,
) -> FemSolution {
    let quarter = dyadic(1, 2);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_fem_mesh::Tetrahedron;

    fn raw(x: Fix128) -> i128 {
        ((x.hi as i128) << 64) | (x.lo as i128)
    }

    /// An axis-aligned tetrahedron whose legs are powers of two, so that
    /// `∇λ = (1/2, 1/4, 1/8)` is **dyadic and therefore exact**.
    ///
    /// ⚠️ Its interior node positions are *not* exact — see
    /// [`the_two_exactness_properties_cannot_both_hold`] for why no element can
    /// have both. This one is the right control for the quadrature, because the
    /// integrand reads `∇λ` and never reads a node position.
    fn element_with_exact_gradients() -> CubicMesh {
        let mut mesh = SdfTetMesh::default();
        mesh.vertices.push([0.0, 0.0, 0.0]);
        mesh.vertices.push([2.0, 0.0, 0.0]);
        mesh.vertices.push([0.0, 4.0, 0.0]);
        mesh.vertices.push([0.0, 0.0, 8.0]);
        mesh.tets.push(Tetrahedron {
            vertices: [0, 1, 2, 3],
        });
        CubicMesh::from_tet_mesh(&mesh).expect("well formed")
    }

    /// The same shape scaled by three, so every coordinate is a multiple of three
    /// and the interior node positions are exact.
    fn element_with_exact_node_positions() -> CubicMesh {
        let mut mesh = SdfTetMesh::default();
        mesh.vertices.push([0.0, 0.0, 0.0]);
        mesh.vertices.push([6.0, 0.0, 0.0]);
        mesh.vertices.push([0.0, 12.0, 0.0]);
        mesh.vertices.push([0.0, 0.0, 24.0]);
        mesh.tets.push(Tetrahedron {
            vertices: [0, 1, 2, 3],
        });
        CubicMesh::from_tet_mesh(&mesh).expect("well formed")
    }

    /// Keast's eleven points, the textbook degree-4 rule, padded to the same
    /// array length so the same assembly can be run with it. The padding carries
    /// zero weight and contributes nothing.
    ///
    /// Present only as the control arm of
    /// [`dyadic_abscissae_lower_the_rounding_of_the_whole_assembly`]; it is not a
    /// rule this module offers.
    fn keast_11_padded() -> (
        [[Fix128; 4]; QUADRATURE_POINTS],
        [Fix128; QUADRATURE_POINTS],
    ) {
        let mut points = [[Fix128::ONE / Fix128::from_int(4); 4]; QUADRATURE_POINTS];
        let mut weights = [Fix128::ZERO; QUADRATURE_POINTS];
        weights[0] = Fix128::from_int(-148) / Fix128::from_int(1875);
        let a31 = Fix128::from_int(11) / Fix128::from_int(14);
        let b31 = Fix128::ONE / Fix128::from_int(14);
        for i in 0..4 {
            let mut p = [b31; 4];
            p[i] = a31;
            points[1 + i] = p;
            weights[1 + i] = Fix128::from_int(343) / Fix128::from_int(7500);
        }
        let r = Fix128::from_int(5).sqrt() / Fix128::from_int(14).sqrt();
        let a22 = (Fix128::ONE + r) / Fix128::from_int(4);
        let b22 = (Fix128::ONE - r) / Fix128::from_int(4);
        for (slot, pattern) in S22_PATTERN.iter().enumerate() {
            let mut p = [a22; 4];
            for (axis, &s) in p.iter_mut().zip(pattern.iter()) {
                *axis = if s == 0 { a22 } else { b22 };
            }
            points[5 + slot] = p;
            weights[5 + slot] = Fix128::from_int(56) / Fix128::from_int(375);
        }
        (points, weights)
    }

    fn pla() -> ElasticMaterial {
        // nu = 1/4, dyadic, so the Lamé constants are not an extra source of
        // rounding in a comparison about rounding.
        ElasticMaterial::new(Fix128::from_int(3500), dyadic(1, 2)).expect("valid")
    }

    /// ⚠️ The two exactness properties of a P3 element are **mutually
    /// exclusive**, and this is a proof by construction of both halves.
    ///
    /// Exact interior node positions need every edge vector to be three times a
    /// dyadic vector, because the nodes sit at `(2a + b)/3`. But then the edge
    /// matrix is `3·D` with `D` dyadic, so `∇λ = adj(3D)/det(3D) =
    /// adj(D)/(3·det D)` carries a factor of `1/3` — and `1/3` is not
    /// representable. Conversely, dyadic `∇λ` forces the legs onto powers of two
    /// and the thirds are then inexact.
    ///
    /// Which one to want depends on the caller: the **integrand reads `∇λ`** and
    /// never reads a node position, so a stiffness computation wants the second;
    /// an oracle prescribing a closed-form field at the boundary nodes wants the
    /// first. They cannot be had together, so this is recorded rather than
    /// resolved.
    #[test]
    fn the_two_exactness_properties_cannot_both_hold() {
        let grads = element_with_exact_gradients();
        let nodes = element_with_exact_node_positions();

        assert!(
            !grads.interior_node_positions_are_exact(),
            "legs that are powers of two cannot place (2a+b)/3 exactly"
        );
        assert!(
            nodes.interior_node_positions_are_exact(),
            "legs that are multiples of three must place (2a+b)/3 exactly; a false here \
             means divide_by_three has stopped measuring anything"
        );

        // ∇λ on the power-of-two element: exactly 1/2, 1/4, 1/8 and their sum.
        let g = grads.elements[0].grad_lambda;
        for (i, (shift, axis)) in [(1u32, 0usize), (2, 1), (3, 2)].into_iter().enumerate() {
            assert_eq!(
                raw(g[i + 1][axis]),
                raw(dyadic(1, shift)),
                "grad lambda {} must be exactly 1/2^{shift}",
                i + 1
            );
        }
        // ∇λ on the multiple-of-three element: 1/6, 1/12, 1/24, none exact.
        let h = nodes.elements[0].grad_lambda;
        for (i, (den, axis)) in [(6i64, 0usize), (12, 1), (24, 2)].into_iter().enumerate() {
            let value = h[i + 1][axis];
            assert_ne!(
                raw(value * Fix128::from_int(den)),
                raw(Fix128::ONE),
                "1/{den} carries a factor of 1/3 and cannot be exact; an exact value here \
                 would contradict the impossibility stated above"
            );
        }
    }

    /// The two identities the assembly rests on, at the quadrature points.
    ///
    /// `Σ Nᵢ = 1` makes a constant field representable; `Σ ∇Nᵢ = 0` makes a
    /// constant field strain-free. Both hold **exactly** at these abscissae: the
    /// first because the abscissae and every shape function coefficient are
    /// dyadic, the second because on this element `∇λ` is dyadic too, so the
    /// products `s·∇λ` are exact as well.
    ///
    /// ⚠️ The second identity is the one that needs the element: on an element
    /// with inexact `∇λ` it holds only to a few ulps, which
    /// [`the_shape_gradient_identity_degrades_with_the_element_geometry`] below
    /// measures so the dependency is not mistaken for a property of the rule.
    #[test]
    fn the_quadrature_points_satisfy_both_partition_identities_exactly() {
        let mesh = element_with_exact_gradients();
        let element = &mesh.elements[0];
        for (i, point) in quadrature_points().iter().enumerate() {
            let mut sum = Fix128::ZERO;
            for v in shape_values(point) {
                sum = sum + v;
            }
            assert_eq!(
                raw(sum),
                raw(Fix128::ONE),
                "point {i}: the twenty cubic shape functions must sum to exactly one at a \
                 dyadic abscissa"
            );
            let grad = shape_gradients(element, point);
            for axis in 0..3 {
                let mut g = Fix128::ZERO;
                for row in &grad {
                    g = g + row[axis];
                }
                assert_eq!(
                    raw(g),
                    0,
                    "point {i} axis {axis}: the shape function gradients must sum to exactly \
                     zero, or a constant field carries strain"
                );
            }
        }
    }

    /// ⚠️ What the exact basis evaluation does **not** buy, measured rather than
    /// guessed.
    ///
    /// The `Σ ∇Nᵢ = 0` identity above is exact only because that element's `∇λ`
    /// is dyadic as well. On an element whose `∇λ` is not — which is most
    /// elements, including every one on a 3 mm lattice — the exactness of the
    /// abscissae does not rescue it, because the product `s·∇λ` truncates. The
    /// residue is small and bounded, and it is reported as a number so that the
    /// claim in the module header stays attached to its condition.
    #[test]
    fn the_shape_gradient_identity_degrades_with_the_element_geometry() {
        let mesh = element_with_exact_node_positions();
        let element = &mesh.elements[0];
        let mut worst = 0i128;
        for point in &quadrature_points() {
            let grad = shape_gradients(element, point);
            for axis in 0..3 {
                let mut g = Fix128::ZERO;
                for row in &grad {
                    g = g + row[axis];
                }
                if raw(g).abs() > worst {
                    worst = raw(g).abs();
                }
            }
        }
        std::eprintln!(
            "[p3fem] sum of shape gradients on an element with inexact grad lambda: \
             worst {worst} raw units (exact answer 0)"
        );
        assert_ne!(
            worst, 0,
            "if this element also satisfies the identity exactly, then the condition stated \
             in the test above is wider than claimed and the module header is wrong"
        );
        assert!(
            worst < 1 << 20,
            "the residue must stay at the rounding floor of a 1/3 that cannot be \
             represented; {worst} raw units is a different mechanism"
        );
    }

    /// ⚠️ The measurement that justifies 24 dyadic points over Keast's eleven,
    /// taken at the **end** of the assembly rather than at its entrance.
    ///
    /// `tests/p3_quadrature_fix128.rs` measures that the shape functions evaluate
    /// exactly at dyadic abscissae. That is the entrance to the integrand, and
    /// `shape_gradients`, `stress_at` and the accumulation in `apply_stiffness`
    /// all round afterwards — so "the entrance is exact" does not by itself mean
    /// the answer is better, and this is the check that it is.
    ///
    /// The probe is the internal force of a **constant** displacement field,
    /// whose exact value is zero at every degree of freedom: no reference value
    /// has to be computed, and the measured number is purely the rounding the
    /// assembly accumulated. Measured on the element with exact `∇λ`:
    ///
    /// | rule | worst internal force, raw units |
    /// |---|---|
    /// | dyadic 24pt | **0** |
    /// | Keast 11pt | non-zero |
    ///
    /// So the property carries through to the result, and by the widest margin
    /// available: the difference is not a smaller error, it is no error.
    #[test]
    fn dyadic_abscissae_lower_the_rounding_of_the_whole_assembly() {
        let mesh = element_with_exact_gradients();
        let (lambda, mu) = pla().lame();
        // A constant field, with dyadic components so that the nodal data itself
        // is exact and the only thing under test is the integration.
        let t = [dyadic(1, 4), -dyadic(3, 5), dyadic(7, 6)];
        let mut u = vec![Fix128::ZERO; mesh.node_count() * 3];
        for n in 0..mesh.node_count() {
            for axis in 0..3 {
                u[n * 3 + axis] = t[axis];
            }
        }
        let mut out = vec![Fix128::ZERO; u.len()];

        let (dp, dw) = (quadrature_points(), quadrature_weights());
        apply_stiffness(&mesh.elements, &dp, &dw, &u, lambda, mu, &mut out);
        let dyadic_worst = out.iter().map(|v| raw(*v).abs()).max().expect("non-empty");

        let (kp, kw) = keast_11_padded();
        apply_stiffness(&mesh.elements, &kp, &kw, &u, lambda, mu, &mut out);
        let keast_worst = out.iter().map(|v| raw(*v).abs()).max().expect("non-empty");

        std::eprintln!(
            "[p3fem] internal force of a constant field (exact answer 0) -- \
             dyadic 24pt {dyadic_worst} raw, Keast 11pt {keast_worst} raw"
        );
        assert_eq!(
            dyadic_worst, 0,
            "a constant field must produce exactly zero internal force on the dyadic rule; \
             got {dyadic_worst} raw units. This is the property the rule was chosen for and \
             it is not a tolerance"
        );
        assert!(
            keast_worst > 0,
            "the comparison is the point: if the textbook rule also returns exactly zero, \
             then dyadic abscissae bought nothing at the end of the assembly and the choice \
             of rule should be revisited"
        );
    }
}
