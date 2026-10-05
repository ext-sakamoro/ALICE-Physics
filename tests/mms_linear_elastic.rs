//! Method of Manufactured Solutions on the existing linear-elastic FEM.
//!
//! This exists to answer one question for the strong-coupling wall: **is MMS
//! compatible with `Fix128`?** Coupled multiphysics has essentially no closed
//! forms, so MMS is the only general way to state what "correct" means — pick a
//! solution, substitute it into the governing equations, and take whatever
//! source term the equality demands. If the source term needed transcendental
//! functions evaluated more accurately than the deterministic arithmetic can
//! manage, the whole plan would be blocked before it started.
//!
//! The answer measured here is **yes, with a polynomial manufactured solution**.
//! A polynomial field and the source term derived from it are built out of
//! `+ − × ÷` only, which is exactly the set `Fix128` is exact in;
//! `polynomial_manufactured_fields_need_no_transcendental` measures the
//! agreement between the `Fix128` and `f64` evaluations. `Fix128` does also carry
//! CORDIC `sin` / `cos` and a `powf_pos`-based `exp`, so a trigonometric
//! manufactured solution is expressible — but `exp` is documented at ≲ 1e-6
//! relative error, which would put a floor under a convergence study long before
//! the discretisation error reached it. Polynomials have no such floor.
//!
//! # ⚠️ The degree has to be chosen against the *mesh*, not just the element
//!
//! Two degrees were measured blind before this one, and the reason is worth more
//! than the tests:
//!
//! | manufactured degree | nodal error at `h = 1 mm` | what it actually was |
//! |---|---|---|
//! | 2 (`c(yz, zx, xy)`, `c(x², 0, 0)`) | 1.7e-18 mm | the arithmetic floor |
//! | 3 (`c(y³−3yz², …)`, `c(x³, 0, 0)`) | 8.6e-12 mm | the conjugate-gradient residual |
//! | 4 (below) | see the tables | discretisation error |
//!
//! At degrees 2 and 3 the "error" *rose* under refinement, tracking the iteration
//! count rather than `h`. The cause is not P1's polynomial reproduction — P1
//! reproduces degree 1 — it is that **on a uniform Kuhn lattice the P1 stiffness
//! matrix reproduces the 7-point Laplacian stencil, whose truncation error is
//! proportional to the fourth derivative of the solution**. A manufactured field
//! of degree ≤ 3 has no fourth derivative, so the nodal values come out exact and
//! the oracle measures nothing.
//!
//! That is the same shape of failure as a patch test on non-conforming faces
//! — machine-epsilon green from an instrument that does not reach the property
//! — with a different cause,
//! and it generalises: **the blind degree depends on the mesh, so the element
//! order is not the only thing that sets the degree an MMS needs.**
//!
//! # What is measured
//!
//! - **zero source**: `u = c·(y⁴−6y²z²+z⁴, z⁴−6z²x²+x⁴, x⁴−6x²y²+y⁴)`. Each
//!   component is independent of its own coordinate, so every normal strain and
//!   the trace vanish and equilibrium reduces to `μ·Δ₂uᵢ` in the other two
//!   coordinates; `y⁴−6y²z²+z⁴ = Re((y+iz)⁴)` is harmonic there, so `div σ ≡ 0`
//!   exactly. **No body force and no load vector to get wrong** — and the fourth
//!   derivatives are non-zero, so the stencil's truncation error is live.
//! - **linear source**: `u = c·(x³, 0, 0)`, body force `−6c(λ+2μ)·x`. Kept at
//!   degree 3 deliberately: a body force linear in space is integrated *exactly*
//!   by the P1 consistent load `M·f`, `M_ij = V/20·(1+δ_ij)`, so this arm
//!   exercises the manufactured-source path with no quadrature error of its own.
//!   Its nodal values are superconvergent, so it is judged on the
//!   **element-interior** error instead.
//!
//! Both errors are reported at the nodes *and* at element centroids, because the
//! two differ by orders of magnitude and only one of them is a discretisation
//! error at any given degree.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// mesh
// ---------------------------------------------------------------------------

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet cube on `[0, n·h]³`, conforming at every `n` (the face census in
/// `tests/refinement_conformity.rs` pins that).
fn kuhn_cube(n: usize, h: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push([
                    (i as f64 * h) as f32,
                    (j as f64 * h) as f32,
                    (k as f64 * h) as f32,
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

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn vert(mesh: &SdfTetMesh, i: u32) -> [f64; 3] {
    let v = mesh.vertices[i as usize];
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
/// Side of the cube in mm — fixed across levels so refinement changes only `h`.
const SIDE: f64 = 4.0;

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

// ---------------------------------------------------------------------------
// the manufactured solutions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Manufactured {
    /// Degree 4, harmonic per component: `div σ ≡ 0`, no body force.
    QuarticZeroSource,
    /// Degree 3 with a body force linear in `x`, integrated exactly by `M·f`.
    CubicLinearSource,
}

impl Manufactured {
    fn name(self) -> &'static str {
        match self {
            Self::QuarticZeroSource => "u = c(y⁴−6y²z²+z⁴, …), zero body force",
            Self::CubicLinearSource => "u = c(x³, 0, 0), body force −6c(λ+2μ)x",
        }
    }

    fn degree(self) -> u32 {
        match self {
            Self::QuarticZeroSource => 4,
            Self::CubicLinearSource => 3,
        }
    }

    /// Amplitude, per case, so the displacements land in the hundredths of a
    /// millimetre: far above the `2⁻⁶⁴` resolution, far below where small strain
    /// would become the thing under test.
    fn amplitude(self) -> f64 {
        match self {
            Self::QuarticZeroSource => 1.0e-4,
            Self::CubicLinearSource => 1.0e-3,
        }
    }

    /// The exact displacement, in mm. `+ − ×` only.
    fn exact(self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = p;
        let c = self.amplitude();
        match self {
            Self::QuarticZeroSource => {
                // Re((a + i b)^4) = a⁴ − 6a²b² + b⁴
                let re4 = |a: f64, b: f64| a * a * a * a - 6.0 * a * a * b * b + b * b * b * b;
                [c * re4(y, z), c * re4(z, x), c * re4(x, y)]
            }
            Self::CubicLinearSource => [c * x * x * x, 0.0, 0.0],
        }
    }

    /// The body force per unit volume the equality demands (N/mm³).
    fn body_force_at(self, p: [f64; 3]) -> [f64; 3] {
        let (lambda, mu) = lame();
        match self {
            Self::QuarticZeroSource => [0.0, 0.0, 0.0],
            Self::CubicLinearSource => [
                -6.0 * self.amplitude() * (lambda + 2.0 * mu) * p[0],
                0.0,
                0.0,
            ],
        }
    }

    fn has_source(self) -> bool {
        self != Self::QuarticZeroSource
    }
}

/// Volume of a tetrahedron, mm³.
fn tet_volume(mesh: &SdfTetMesh, tet: &Tetrahedron) -> f64 {
    let p: Vec<[f64; 3]> = tet.vertices.iter().map(|&i| vert(mesh, i)).collect();
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    det.abs() / 6.0
}

// ---------------------------------------------------------------------------
// one refinement level
// ---------------------------------------------------------------------------

struct Level {
    cells: usize,
    h: f64,
    nodes: usize,
    tets: usize,
    interior: usize,
    /// Max / rms error of the solved nodal displacements (mm).
    node_max: f64,
    node_rms: f64,
    /// Max / volume-weighted error of the P1 field at element centroids (mm).
    /// This is the `L²`-style measure: it sees the interpolation error the nodes
    /// hide whenever the nodal values are superconvergent.
    cell_max: f64,
    cell_l2: f64,
    iterations: u32,
}

fn run_level(case: Manufactured, cells: usize) -> Level {
    let h = SIDE / cells as f64;
    let mesh = kuhn_cube(cells, h);
    let eps = h * 1e-4;
    let on_boundary = |p: [f64; 3]| p.iter().any(|&c| c < eps || c > SIDE - eps);

    let mut bc = BoundaryConditions::new();
    let mut interior = 0usize;
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(&mesh, v);
        if on_boundary(p) {
            let u = case.exact(p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior += 1;
        }
    }

    // Consistent nodal load of a body force varying linearly over the element:
    // ∫_e N_i f dV = Σ_j M_ij f_j with the P1 mass matrix M_ij = V/20·(1 + δ_ij),
    // which collapses to V/20·(f_i + Σ_j f_j).
    // `consistent_load_reduces_to_a_quarter_for_a_constant_force` pins that
    // collapse against the independently known constant-force answer f·V/4.
    if case.has_source() {
        for tet in &mesh.tets {
            let volume = tet_volume(&mesh, tet);
            let nodal: Vec<[f64; 3]> = tet
                .vertices
                .iter()
                .map(|&i| case.body_force_at(vert(&mesh, i)))
                .collect();
            let mut total = [0.0_f64; 3];
            for n in &nodal {
                for axis in 0..3 {
                    total[axis] += n[axis];
                }
            }
            for (slot, &v) in tet.vertices.iter().enumerate() {
                for (axis_i, axis) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
                    let load = volume / 20.0 * (nodal[slot][axis_i] + total[axis_i]);
                    if load != 0.0 {
                        bc.add_load(v, axis, fx(load));
                    }
                }
            }
        }
    }

    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let out = solve(&mesh, &pla(), &bc, &config).expect("well posed");

    // --- nodal error, interior nodes only -----------------------------------
    let mut node_max = 0.0_f64;
    let mut node_sq = 0.0_f64;
    let mut counted = 0usize;
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(&mesh, v);
        if on_boundary(p) {
            continue;
        }
        let got = out.displacements[v as usize];
        for (axis, want) in case.exact(p).into_iter().enumerate() {
            let e = (got[axis].to_f64() - want).abs();
            node_max = node_max.max(e);
            node_sq += e * e;
            counted += 1;
        }
    }

    // --- element-interior error at the centroid ------------------------------
    // The P1 field at the centroid is the mean of the four nodal values, so this
    // needs no shape-function evaluation and no extra machinery.
    let mut cell_max = 0.0_f64;
    let mut weighted = 0.0_f64;
    let mut total_volume = 0.0_f64;
    for tet in &mesh.tets {
        let volume = tet_volume(&mesh, tet);
        let mut centroid = [0.0_f64; 3];
        let mut interp = [0.0_f64; 3];
        for &i in &tet.vertices {
            let p = vert(&mesh, i);
            for axis in 0..3 {
                centroid[axis] += p[axis] / 4.0;
                interp[axis] += out.displacements[i as usize][axis].to_f64() / 4.0;
            }
        }
        let want = case.exact(centroid);
        let mut sq = 0.0_f64;
        for axis in 0..3 {
            let e = (interp[axis] - want[axis]).abs();
            cell_max = cell_max.max(e);
            sq += e * e;
        }
        weighted += sq * volume;
        total_volume += volume;
    }

    Level {
        cells,
        h,
        nodes: mesh.vertex_count(),
        tets: mesh.tet_count(),
        interior,
        node_max,
        node_rms: (node_sq / counted.max(1) as f64).sqrt(),
        cell_max,
        cell_l2: (weighted / total_volume.max(f64::MIN_POSITIVE)).sqrt(),
        iterations: out.iterations,
    }
}

/// Richardson order from three errors on successively halved cells.
fn order(coarse: f64, mid: f64, fine: f64) -> f64 {
    if mid <= 0.0 || fine <= 0.0 {
        return f64::NAN;
    }
    ((coarse / mid).log2() + (mid / fine).log2()) / 2.0
}

fn report(case: Manufactured, levels: &[Level]) {
    eprintln!("  {}  (degree {})", case.name(), case.degree());
    eprintln!(
        "  {:>5} {:>7} {:>6} {:>7} {:>8} {:>13} {:>13} {:>13} {:>13} {:>7}",
        "cells",
        "h",
        "nodes",
        "tets",
        "interior",
        "node max",
        "node rms",
        "cell max",
        "cell L2",
        "cg it"
    );
    for l in levels {
        eprintln!(
            "  {:>5} {:>7.4} {:>6} {:>7} {:>8} {:>13.6e} {:>13.6e} {:>13.6e} {:>13.6e} {:>7}",
            l.cells,
            l.h,
            l.nodes,
            l.tets,
            l.interior,
            l.node_max,
            l.node_rms,
            l.cell_max,
            l.cell_l2,
            l.iterations
        );
    }
    eprintln!(
        "  order: node rms {:.3}, cell L2 {:.3}",
        order(levels[0].node_rms, levels[1].node_rms, levels[2].node_rms),
        order(levels[0].cell_l2, levels[1].cell_l2, levels[2].cell_l2)
    );
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

/// A manufactured field must produce an error the refinement can actually
/// reduce, or the study is measuring the solver's stopping rule.
///
/// This is the check that caught degrees 2 and 3: both gave a nodal error at the
/// conjugate-gradient floor that *grew* under refinement. The property asserted
/// is the one that matters — the error must fall when `h` halves.
#[test]
fn the_manufactured_error_is_discretisation_not_residual() {
    for (case, metric) in [
        (Manufactured::QuarticZeroSource, "node"),
        (Manufactured::CubicLinearSource, "cell"),
    ] {
        let coarse = run_level(case, 4);
        let fine = run_level(case, 8);
        let (c, f) = if metric == "node" {
            (coarse.node_rms, fine.node_rms)
        } else {
            (coarse.cell_l2, fine.cell_l2)
        };
        eprintln!(
            "  {}: {metric} error {:.6e} -> {:.6e} mm over h 1.0 -> 0.5 ({} -> {} cg it)",
            case.name(),
            c,
            f,
            coarse.iterations,
            fine.iterations
        );
        assert!(
            f < c,
            "{}: the {metric} error grew under refinement ({c:.6e} -> {f:.6e} mm) while the \
             iteration count went {} -> {}. That is the conjugate-gradient residual, not a \
             discretisation error — raise the manufactured degree (degree ≤ 3 is exact at the \
             nodes of a uniform Kuhn lattice, see the module doc)",
            case.name(),
            coarse.iterations,
            fine.iterations
        );
    }
}

/// MMS with a zero source: the error must fall at the P1 rate.
///
/// The band is the second-order one a P1 scheme gives. It is not narrowed to a
/// point, because what this test has to establish is that a manufactured solution
/// *works* in this arithmetic — a rate inside the band establishes it, and a rate
/// near zero would mean the boundary data and the field disagree.
#[test]
fn zero_source_mms_converges_at_the_p1_rate() {
    let levels: Vec<Level> = [4usize, 8, 16]
        .into_iter()
        .map(|c| run_level(Manufactured::QuarticZeroSource, c))
        .collect();
    report(Manufactured::QuarticZeroSource, &levels);

    for w in levels.windows(2) {
        assert!(
            w[1].node_rms < w[0].node_rms,
            "refinement must reduce the nodal error: {:.6e} -> {:.6e}",
            w[0].node_rms,
            w[1].node_rms
        );
    }
    let p = order(levels[0].node_rms, levels[1].node_rms, levels[2].node_rms);
    assert!(
        p > 1.5 && p < 2.5,
        "the nodal convergence order is {p:.3}, outside the band a P1 scheme gives. Either the \
         manufactured field and its boundary data disagree, or the arithmetic has put a floor \
         under the error"
    );
}

/// MMS with a non-zero source: the same, through the load vector.
///
/// This is the arm that matters for coupled physics, where the manufactured
/// source term is the whole point — it shows a manufactured right-hand side goes
/// through the load path without leaving the exact `+ − × ÷` regime. Judged on
/// the element-interior error, because a cubic's nodal values are exact on this
/// lattice (module doc).
#[test]
fn source_term_mms_converges_at_the_p1_rate() {
    let levels: Vec<Level> = [4usize, 8, 16]
        .into_iter()
        .map(|c| run_level(Manufactured::CubicLinearSource, c))
        .collect();
    report(Manufactured::CubicLinearSource, &levels);
    let f_end = Manufactured::CubicLinearSource.body_force_at([SIDE, 0.0, 0.0]);
    eprintln!(
        "  body force at x = {SIDE} mm: ({:.6e}, {:.1}, {:.1}) N/mm³",
        f_end[0], f_end[1], f_end[2]
    );

    for w in levels.windows(2) {
        assert!(
            w[1].cell_l2 < w[0].cell_l2,
            "refinement must reduce the element-interior error: {:.6e} -> {:.6e}",
            w[0].cell_l2,
            w[1].cell_l2
        );
    }
    let p = order(levels[0].cell_l2, levels[1].cell_l2, levels[2].cell_l2);
    assert!(
        p > 1.5 && p < 2.5,
        "the element-interior convergence order is {p:.3}, outside the band a P1 scheme gives \
         — the manufactured source and the field do not match"
    );
}

/// The consistent-load formula is checked against the one case it has an
/// independent answer for.
///
/// `∫_e N_i f dV = Σ_j M_ij f_j` with `M_ij = V/20·(1 + δ_ij)` must collapse to
/// `f·V/4` when `f` is equal at all four nodes: `V/20·(f + 4f) = V/4·f`. This is
/// the only part of the manufactured-source path that is the test's arithmetic
/// rather than the solver's, so it gets its own check — a wrong mass matrix would
/// show up in the convergence study as a plausible rate rather than as a failure.
#[test]
fn consistent_load_reduces_to_a_quarter_for_a_constant_force() {
    let mesh = kuhn_cube(2, SIDE / 2.0);
    let f = -7.5_f64;
    let total = 4.0 * f;
    for tet in &mesh.tets {
        let volume = tet_volume(&mesh, tet);
        let load = volume / 20.0 * (f + total);
        let want = f * volume / 4.0;
        assert!(
            (load - want).abs() <= 1e-12 * want.abs().max(1.0),
            "mass-matrix load {load:.12e} disagrees with the constant-force answer {want:.12e}"
        );
    }
    eprintln!(
        "  V/20·(f + Σf) = f·V/4 confirmed over {} tets",
        mesh.tet_count()
    );
}

/// No transcendental is reachable from this file's arithmetic.
///
/// The manufactured fields and both source terms are polynomial, so every value
/// handed to the solver is a sum of products of exactly represented quantities.
/// This states that as a property of the data rather than leaving it to the prose
/// above: it builds the exact field twice, once in `f64` and once with `Fix128`
/// multiplication, and requires agreement at the `Fix128` resolution. A field
/// needing `sin` or `exp` could not pass, because those carry ≳ 1e-6 relative
/// error in this arithmetic.
#[test]
fn polynomial_manufactured_fields_need_no_transcendental() {
    let mut worst_overall = 0.0_f64;
    for case in [
        Manufactured::QuarticZeroSource,
        Manufactured::CubicLinearSource,
    ] {
        let c = fx(case.amplitude());
        let six = fx(6.0);
        let mut worst = 0.0_f64;
        for i in 0..=8 {
            for j in 0..=8 {
                for k in 0..=8 {
                    let p = [
                        SIDE * f64::from(i) / 8.0,
                        SIDE * f64::from(j) / 8.0,
                        SIDE * f64::from(k) / 8.0,
                    ];
                    let want = case.exact(p);
                    let (x, y, z) = (fx(p[0]), fx(p[1]), fx(p[2]));
                    let re4 =
                        |a: Fix128, b: Fix128| a * a * a * a - six * a * a * b * b + b * b * b * b;
                    let got = match case {
                        Manufactured::QuarticZeroSource => {
                            [c * re4(y, z), c * re4(z, x), c * re4(x, y)]
                        }
                        Manufactured::CubicLinearSource => {
                            [c * x * x * x, Fix128::ZERO, Fix128::ZERO]
                        }
                    };
                    for axis in 0..3 {
                        worst = worst.max((got[axis].to_f64() - want[axis]).abs());
                    }
                }
            }
        }
        eprintln!("  {}: |Fix128 − f64| ≤ {:.3e} mm", case.name(), worst);
        worst_overall = worst_overall.max(worst);
    }

    // Four Fix128 multiplications truncate at 2⁻⁶⁴ each and the f64 reference
    // rounds at its own 53-bit precision, so a few ulp of the larger of the two
    // is all the agreement that can be asked for — many orders below the
    // discretisation errors measured above.
    assert!(
        worst_overall < 1e-14,
        "the polynomial fields disagree between f64 and Fix128 by {worst_overall:.3e} mm, far \
         above truncation — the field is then not a polynomial in the exact operations"
    );
}
