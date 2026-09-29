//! What a P2 convergence oracle has to look like, measured before P2 exists.
//!
//! `mms_linear_elastic.rs` records that a manufactured solution of degree ≤ 3 is
//! reproduced **exactly at the nodes** by P1 on a uniform Kuhn lattice, so an
//! oracle built on it measures the conjugate-gradient residual instead of the
//! discretisation. The note there draws the right conclusion — the blind degree
//! depends on the mesh — and then offers a remedy for P2 that has never been
//! measured: *raise the degree*, to 5 or beyond.
//!
//! ⚠️ **Raising the degree is not the only remedy, and this file measures which
//! one is actually doing the work.** The stated cause is that a uniform lattice
//! makes the P1 stiffness reproduce a 7-point Laplacian whose truncation error
//! is proportional to the fourth derivative. If that is the cause, then the
//! blindness belongs to the *lattice*, and destroying the lattice must restore
//! the error at degree 3 without touching the degree at all. That is a testable
//! statement, it can be tested today with the P1 element that already exists,
//! and its answer decides the P2 oracle:
//!
//! - **If the lattice is the cause**, a P2 oracle on a uniform lattice will be
//!   blind at some higher degree too, and chasing it by raising the degree is
//!   chasing a moving target. Perturbing the mesh fixes it once, for every
//!   element order.
//! - **If the element is the cause**, the degree has to go up and the mesh is
//!   irrelevant.
//!
//! The perturbation used below moves only *interior* vertices, and only by a
//! fraction of `h`. The topology, the element count, the domain and the
//! boundary nodes are all untouched, so the two meshes differ in exactly one
//! property: whether the stencil is uniform.

//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// `log2` / `powi` on `f64` are disallowed crate-wide because the platform libm
// is not bit-exact across targets. Every use here is in the *reporting* of a
// convergence order or a closed-form beam deflection — the solve itself is
// `Fix128` throughout — so a last-bit difference between targets changes a
// printed slope and nothing that is asserted to the bit. Same exemption, and
// same reason, as `tests/mms_linear_elastic.rs` and
// `tests/analytic_fem_convergence.rs`.
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
const SIDE: f64 = 4.0;
/// Amplitude of `u = c·x³`, chosen as in `mms_linear_elastic.rs`.
const AMPLITUDE: f64 = 1.0e-3;
/// Interior-vertex displacement, as a fraction of `h`.
///
/// Large enough to destroy the uniform stencil, small enough that no Kuhn
/// tetrahedron can invert — which `the_perturbed_mesh_is_still_a_valid_mesh`
/// checks rather than assumes.
const JITTER: f64 = 0.25;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid")
}

fn lame() -> (f64, f64) {
    let lambda = E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let mu = E_MPA / (2.0 * (1.0 + NU));
    (lambda, mu)
}

fn vert(mesh: &SdfTetMesh, i: u32) -> [f64; 3] {
    let v = mesh.vertices[i as usize];
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

// ---------------------------------------------------------------------------
// the manufactured solution: degree 3, the one that is blind
// ---------------------------------------------------------------------------

/// `u = c·(x³, 0, 0)`.
fn exact(p: [f64; 3]) -> [f64; 3] {
    [AMPLITUDE * p[0] * p[0] * p[0], 0.0, 0.0]
}

/// `f = −6c(λ+2μ)·x`, the body force the equality demands. Linear in `x`, so
/// the P1 consistent load `M·f` integrates it exactly on any straight-edged
/// tetrahedron — perturbed or not. That is what keeps the quadrature out of the
/// comparison below.
fn body_force_at(p: [f64; 3]) -> [f64; 3] {
    let (lambda, mu) = lame();
    [-6.0 * AMPLITUDE * (lambda + 2.0 * mu) * p[0], 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// meshes
// ---------------------------------------------------------------------------

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Deterministic displacement in `[-1, 1]` for one lattice node and axis.
///
/// A hash, not a random number generator: the mesh has to be the same on every
/// run and every target, or the convergence table is not reproducible.
fn jitter_unit(i: usize, j: usize, k: usize, axis: usize) -> f64 {
    let mut h = 0x9E37_79B9_7F4A_7C15_u64;
    for v in [i as u64, j as u64, k as u64, axis as u64] {
        h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
    }
    // Top 32 bits to a fraction, then centred.
    ((h >> 32) as f64) / ((1u64 << 32) as f64) * 2.0 - 1.0
}

/// Kuhn 6-tet cube on `[0, n·h]³`. `jitter = 0` gives the uniform lattice.
fn kuhn_cube(n: usize, h: f64, jitter: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let on_boundary = i == 0 || j == 0 || k == 0 || i == n || j == n || k == n;
                let base = [i as f64 * h, j as f64 * h, k as f64 * h];
                let mut p = base;
                if !on_boundary && jitter != 0.0 {
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

/// Signed volume × 6.
fn tet_det(mesh: &SdfTetMesh, tet: &Tetrahedron) -> f64 {
    let p: Vec<[f64; 3]> = tet.vertices.iter().map(|&i| vert(mesh, i)).collect();
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn tet_volume(mesh: &SdfTetMesh, tet: &Tetrahedron) -> f64 {
    tet_det(mesh, tet).abs() / 6.0
}

// ---------------------------------------------------------------------------
// one level
// ---------------------------------------------------------------------------

struct Level {
    h: f64,
    /// rms error at the interior nodes (mm). Superconvergent on the uniform
    /// lattice, which is the whole point.
    node_rms: f64,
    /// Volume-weighted error of the P1 field at element centroids (mm). Sees
    /// the interpolation error the nodes hide.
    cell_l2: f64,
    iterations: u32,
    residual: f64,
}

fn run_level(cells: usize, jitter: f64) -> Level {
    let h = SIDE / cells as f64;
    let mesh = kuhn_cube(cells, h, jitter);
    let eps = h * 1e-4;
    let on_boundary = |p: [f64; 3]| p.iter().any(|&c| c < eps || c > SIDE - eps);

    let mut bc = BoundaryConditions::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(&mesh, v);
        if on_boundary(p) {
            let u = exact(p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        }
    }

    // Consistent load, P1 mass matrix `M_ij = V/20·(1 + δ_ij)`.
    for tet in &mesh.tets {
        let volume = tet_volume(&mesh, tet);
        let nodal: Vec<[f64; 3]> = tet
            .vertices
            .iter()
            .map(|&i| body_force_at(vert(&mesh, i)))
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

    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let out = solve(&mesh, &pla(), &bc, &config).expect("well posed");

    let mut node_sq = 0.0_f64;
    let mut counted = 0usize;
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(&mesh, v);
        if on_boundary(p) {
            continue;
        }
        let got = out.displacements[v as usize];
        for (axis, want) in exact(p).into_iter().enumerate() {
            let e = got[axis].to_f64() - want;
            node_sq += e * e;
            counted += 1;
        }
    }

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
        let want = exact(centroid);
        let mut sq = 0.0_f64;
        for axis in 0..3 {
            let e = interp[axis] - want[axis];
            sq += e * e;
        }
        weighted += sq * volume;
        total_volume += volume;
    }

    Level {
        h,
        node_rms: (node_sq / counted.max(1) as f64).sqrt(),
        cell_l2: (weighted / total_volume).sqrt(),
        iterations: out.iterations,
        residual: out.relative_residual.to_f64(),
    }
}

/// Richardson order from three errors on successively halved cells.
fn order(coarse: f64, mid: f64, fine: f64) -> f64 {
    if mid <= 0.0 || fine <= 0.0 {
        return f64::NAN;
    }
    (coarse / mid).log2().min((mid / fine).log2())
}

fn slope(coarse: f64, fine: f64) -> f64 {
    (coarse / fine).log2()
}

fn study(label: &str, jitter: f64, cells: &[usize]) -> Vec<Level> {
    eprintln!("[p2oracle] ---- {label} ----");
    eprintln!(
        "[p2oracle] {:>7}  {:>13}  {:>13}  {:>6}  {:>10}",
        "h(mm)", "node rms", "cell L2", "iters", "residual"
    );
    let levels: Vec<Level> = cells.iter().map(|c| run_level(*c, jitter)).collect();
    for l in &levels {
        eprintln!(
            "[p2oracle] {:>7.4}  {:>13.6e}  {:>13.6e}  {:>6}  {:>10.3e}",
            l.h, l.node_rms, l.cell_l2, l.iterations, l.residual
        );
    }
    for w in levels.windows(2) {
        eprintln!(
            "[p2oracle]   h {:.4}->{:.4}: node slope {:+.3}, cell slope {:+.3}",
            w[0].h,
            w[1].h,
            slope(w[0].node_rms, w[1].node_rms),
            slope(w[0].cell_l2, w[1].cell_l2)
        );
    }
    levels
}

// ---------------------------------------------------------------------------
// the perturbed mesh has to be a mesh before it can be an argument
// ---------------------------------------------------------------------------

#[test]
fn the_perturbed_mesh_is_still_a_valid_mesh() {
    for cells in [4usize, 8, 16] {
        let h = SIDE / cells as f64;
        let uniform = kuhn_cube(cells, h, 0.0);
        let perturbed = kuhn_cube(cells, h, JITTER);

        assert!(
            uniform.tet_count() == perturbed.tet_count()
                && uniform.vertex_count() == perturbed.vertex_count(),
            "the perturbation must not change the topology: {} vs {} tets, {} vs {} vertices",
            uniform.tet_count(),
            perturbed.tet_count(),
            uniform.vertex_count(),
            perturbed.vertex_count()
        );

        // Same orientation everywhere, and no element collapsed: a sign flip or
        // a near-zero volume would make the comparison below about element
        // quality instead of about the stencil.
        let mut min_ratio = f64::INFINITY;
        let mut total_u = 0.0;
        let mut total_p = 0.0;
        for (a, b) in uniform.tets.iter().zip(&perturbed.tets) {
            let du = tet_det(&uniform, a);
            let dp = tet_det(&perturbed, b);
            assert!(
                du.signum() == dp.signum(),
                "cells={cells}: perturbation inverted a tetrahedron"
            );
            min_ratio = min_ratio.min((dp / du).abs());
            total_u += du.abs() / 6.0;
            total_p += dp.abs() / 6.0;
        }
        // The domain is unchanged: only interior vertices moved, so the volumes
        // must still sum to the cube.
        let cube = SIDE * SIDE * SIDE;
        assert!(
            (total_u - cube).abs() < 1e-6 * cube && (total_p - cube).abs() < 1e-6 * cube,
            "cells={cells}: total volume must stay {cube}: uniform {total_u}, \
             perturbed {total_p}"
        );
        eprintln!(
            "[p2oracle] cells={cells}: perturbed mesh valid, smallest volume ratio \
             {min_ratio:.4} of the uniform element"
        );
        assert!(
            min_ratio > 0.05,
            "cells={cells}: an element shrank to {min_ratio:.4} of its uniform volume; \
             the study would be measuring element quality"
        );

        // ⚠️ Without this, the test passes for `JITTER = 0` — a perturbation that
        // does nothing satisfies every check above, and the study downstream
        // would then be comparing a mesh with itself. Measured: with
        // `JITTER = 0` this test stayed green while
        // `degree_three_blindness_...` reported "perturbed 2.583e-11 against
        // uniform 2.583e-11". With the check, `JITTER = 0` fails here first,
        // saying "0 moved, 27 are interior".
        let moved = uniform
            .vertices
            .iter()
            .zip(&perturbed.vertices)
            .filter(|(a, b)| a != b)
            .count();
        let interior = (cells - 1) * (cells - 1) * (cells - 1);
        assert!(
            moved == interior,
            "cells={cells}: the perturbation must move every interior vertex and no \
             boundary vertex; {moved} moved, {interior} are interior"
        );
    }
}

// ---------------------------------------------------------------------------
// the question this file exists for
// ---------------------------------------------------------------------------

/// Is the degree-3 blindness the lattice or the element?
///
/// Same element, same degree, same domain, same element count, same boundary
/// data. The only difference is whether the interior vertices sit on a uniform
/// lattice. If the note in `mms_linear_elastic.rs` is right about the cause,
/// the uniform run is blind and the perturbed one is not.
#[test]
fn degree_three_blindness_belongs_to_the_lattice_not_to_the_element() {
    let cells = [4usize, 8, 16];
    let uniform = study("degree 3, uniform Kuhn lattice", 0.0, &cells);
    let perturbed = study(
        "degree 3, interior vertices perturbed by 0.25 h",
        JITTER,
        &cells,
    );

    let u_node = order(
        uniform[0].node_rms,
        uniform[1].node_rms,
        uniform[2].node_rms,
    );
    let p_node = order(
        perturbed[0].node_rms,
        perturbed[1].node_rms,
        perturbed[2].node_rms,
    );
    let p_cell = order(
        perturbed[0].cell_l2,
        perturbed[1].cell_l2,
        perturbed[2].cell_l2,
    );
    eprintln!(
        "[p2oracle] nodal order: uniform {u_node:+.3}, perturbed {p_node:+.3}; \
         perturbed cell order {p_cell:+.3}"
    );

    // The uniform lattice is blind: the nodal error is at the level of the
    // solver, so it does not fall at a rate, and the finest level is not even
    // the most accurate.
    assert!(
        uniform[2].node_rms > 1e-13,
        "the uniform run is supposed to be residual-limited, not exact; got {:.3e} — \
         if this is now the arithmetic floor the table above needs re-reading",
        uniform[2].node_rms
    );

    // The perturbed mesh restores a real discretisation error at the same
    // degree with the same element.
    assert!(
        perturbed[2].node_rms > uniform[2].node_rms * 10.0,
        "perturbing the lattice must expose an error the uniform lattice hides; \
         got perturbed {:.3e} against uniform {:.3e}",
        perturbed[2].node_rms,
        uniform[2].node_rms
    );
    assert!(
        p_node > 1.5,
        "with the lattice destroyed, degree 3 must converge at a real rate; got \
         {p_node:+.3}"
    );
}

/// The oracle a P2 element will have to pass, written now so that it is red now.
///
/// `cell_l2` for a P1 element converges at second order. A P2 element on the
/// same meshes has to reach third order in that norm; nothing less distinguishes
/// it from the element that is already there. The mesh is the perturbed one,
/// because the test above measures that a uniform lattice hides the very thing
/// this is trying to see.
///
/// ⚠️ **This test fails on the current crate**, and that is its present job: it
/// is the red that the P2 work has to turn green. It is not a regression and
/// must not be "fixed" by lowering the exponent.
///
/// Measured red, `d90b18e`, before any P2 code exists:
///
/// ```text
/// [p2oracle] cell L2 order = +1.895 (P1 gives ~2, the P2 target is ~3)
/// a second-order element cannot pass this: cell L2 converged at +1.895.
/// ```
///
/// `#[ignore]` only so that a committed red does not stop every other gate in
/// the repository. Run it with `cargo test --test p2_oracle_design -- --ignored`
/// and delete the attribute the moment P2 is wired in; the assertion itself is
/// the acceptance criterion and does not change.
#[test]
#[ignore = "the P2 acceptance criterion: red until P2 exists (cell L2 order +1.895 on P1)"]
fn p2_must_reach_third_order_and_p1_does_not() {
    let cells = [4usize, 8, 16];
    let levels = study(
        "the P2 target: third order in cell L2 on a perturbed mesh",
        JITTER,
        &cells,
    );
    let p = order(levels[0].cell_l2, levels[1].cell_l2, levels[2].cell_l2);
    eprintln!("[p2oracle] cell L2 order = {p:+.3} (P1 gives ~2, the P2 target is ~3)");
    assert!(
        p > 2.7,
        "a second-order element cannot pass this: cell L2 converged at {p:+.3}. When P2 \
         lands, this is the assertion that says so"
    );
}
