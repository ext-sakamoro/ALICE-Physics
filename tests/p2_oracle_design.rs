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
use alice_physics::quadratic_elastic_fem::{solve_quadratic, QuadraticMesh};
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

/// A degree-3 field with **no body force**, for comparing the two elements.
///
/// `u = c·(Re((y+iz)³), Re((z+ix)³), Re((x+iy)³))`, i.e.
/// `c·(y³−3yz², z³−3zx², x³−3xy²)`. Each component is independent of its own
/// coordinate, so every normal strain and the trace vanish and equilibrium
/// reduces to `μ·Δ₂uᵢ`; each component is harmonic in its two coordinates, so
/// `div σ ≡ 0` exactly.
///
/// ⚠️ **The absence of a body force is what makes the comparison fair.** A
/// consistent load for a *linear* body force is exact on P1 through the closed
/// form `M·f`, but on P2 the product `N_i·f` is **cubic**, which the
/// degree-2 Hammer–Stroud rule the element assembles with does not integrate
/// exactly. Comparing the two elements on a field with a source term would
/// therefore be comparing their load quadrature as much as their function
/// spaces. With no source there is no load vector at all.
fn cubic_zero_source(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = AMPLITUDE;
    let re3 = |a: f64, b: f64| a * a * a - 3.0 * a * b * b;
    [c * re3(y, z), c * re3(z, x), c * re3(x, y)]
}

/// Volume-weighted error at the element centroids, for one element type.
///
/// The centroid value of a P1 field is the mean of its four nodal values; of a
/// P2 field it is `Σ Nᵢ(¼,¼,¼,¼) uᵢ`, which is `−1/8` on each corner and `1/4`
/// on each edge node (they sum to one, as they must).
fn cubic_cell_l2(cells: usize, quadratic: bool) -> (f64, u32, f64) {
    let h = SIDE / cells as f64;
    let mesh = kuhn_cube(cells, h, JITTER);
    let eps = h * 1e-4;
    let on_boundary = |p: [f64; 3]| p.iter().any(|&c| c < eps || c > SIDE - eps);
    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid");

    // Displacement at every node of whichever discretisation is in use.
    let (positions, displacements, iterations, residual) = if quadratic {
        let q = QuadraticMesh::from_tet_mesh(&mesh).expect("well formed");
        let mut positions = Vec::with_capacity(q.node_count());
        let mut bc = BoundaryConditions::new();
        for node in 0..u32::try_from(q.node_count()).expect("fits") {
            let f = q.node_position(node).expect("in range");
            let p = [f[0].to_f64(), f[1].to_f64(), f[2].to_f64()];
            positions.push(p);
            if on_boundary(p) {
                let u = cubic_zero_source(p);
                bc.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
            }
        }
        let out = solve_quadratic(&q, &pla(), &bc, &config).expect("well posed");
        let mut weighted = 0.0_f64;
        let mut total = 0.0_f64;
        for e in 0..q.element_count() {
            let nodes = q.element_nodes(e).expect("in range");
            // Volume from the four corners; the element is straight-edged.
            let corner = |slot: usize| positions[nodes[slot] as usize];
            let volume = tet_volume_from(corner(0), corner(1), corner(2), corner(3));
            let mut centroid = [0.0_f64; 3];
            let mut interp = [0.0_f64; 3];
            for slot in 0..10 {
                let w = if slot < 4 { -0.125 } else { 0.25 };
                let p = positions[nodes[slot] as usize];
                let u = out.displacements[nodes[slot] as usize];
                for axis in 0..3 {
                    interp[axis] += w * u[axis].to_f64();
                    if slot < 4 {
                        centroid[axis] += p[axis] / 4.0;
                    }
                }
            }
            let want = cubic_zero_source(centroid);
            let mut sq = 0.0_f64;
            for axis in 0..3 {
                let e = interp[axis] - want[axis];
                sq += e * e;
            }
            weighted += sq * volume;
            total += volume;
        }
        return (
            (weighted / total).sqrt(),
            out.iterations,
            out.relative_residual.to_f64(),
        );
    } else {
        let mut bc = BoundaryConditions::new();
        let mut positions = Vec::with_capacity(mesh.vertex_count());
        for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
            let p = vert(&mesh, v);
            positions.push(p);
            if on_boundary(p) {
                let u = cubic_zero_source(p);
                bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
            }
        }
        let out = solve(&mesh, &pla(), &bc, &config).expect("well posed");
        let d = (0..mesh.vertex_count())
            .map(|n| {
                [
                    out.displacements[n][0].to_f64(),
                    out.displacements[n][1].to_f64(),
                    out.displacements[n][2].to_f64(),
                ]
            })
            .collect::<Vec<_>>();
        (positions, d, out.iterations, out.relative_residual.to_f64())
    };

    let mut weighted = 0.0_f64;
    let mut total = 0.0_f64;
    for tet in &mesh.tets {
        let p: Vec<[f64; 3]> = tet
            .vertices
            .iter()
            .map(|&i| positions[i as usize])
            .collect();
        let volume = tet_volume_from(p[0], p[1], p[2], p[3]);
        let mut centroid = [0.0_f64; 3];
        let mut interp = [0.0_f64; 3];
        for (slot, &i) in tet.vertices.iter().enumerate() {
            for axis in 0..3 {
                centroid[axis] += p[slot][axis] / 4.0;
                interp[axis] += displacements[i as usize][axis] / 4.0;
            }
        }
        let want = cubic_zero_source(centroid);
        let mut sq = 0.0_f64;
        for axis in 0..3 {
            let e = interp[axis] - want[axis];
            sq += e * e;
        }
        weighted += sq * volume;
        total += volume;
    }
    ((weighted / total).sqrt(), iterations, residual)
}

fn tet_volume_from(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]);
    det.abs() / 6.0
}

/// The acceptance criterion for the quadratic element, and the P1 measurement
/// that makes it non-vacuous.
///
/// Both elements solve the **same** manufactured problem on the **same**
/// perturbed meshes, with no body force on either side. The claim is about the
/// function spaces and nothing else: P1 converges at second order in the
/// element-interior `L²` norm, P2 at third.
///
/// ⚠️ The P1 arm is not decoration. Without it, a P2 order of 3 could come from
/// the mesh sequence, the norm or the field rather than from the element, and
/// the test would have no way to say so.
#[test]
fn p2_reaches_third_order_where_p1_reaches_second() {
    let cells = [2usize, 4, 8];
    eprintln!("[p2oracle] ---- cubic zero-source field, perturbed lattice, both elements ----");
    eprintln!(
        "[p2oracle] {:>7}  {:>6}  {:>13}  {:>6}  {:>10}",
        "h(mm)", "elem", "cell L2", "iters", "residual"
    );
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    for &c in &cells {
        for quadratic in [false, true] {
            let (l2, iters, residual) = cubic_cell_l2(c, quadratic);
            eprintln!(
                "[p2oracle] {:>7.4}  {:>6}  {:>13.6e}  {:>6}  {:>10.3e}",
                SIDE / c as f64,
                if quadratic { "P2" } else { "P1" },
                l2,
                iters,
                residual
            );
            if quadratic {
                p2.push((l2, residual));
            } else {
                p1.push((l2, residual));
            }
        }
    }

    for (label, rows) in [("P1", &p1), ("P2", &p2)] {
        for (l2, residual) in rows.iter() {
            assert!(
                *residual <= 1.0e-9,
                "{label}: a solve stopped at residual {residual:.3e}, so its error {l2:.3e}                  measures the conjugate gradient and not the element"
            );
        }
    }

    let order_p1 = order(p1[0].0, p1[1].0, p1[2].0);
    let order_p2 = order(p2[0].0, p2[1].0, p2[2].0);
    eprintln!("[p2oracle] cell L2 order: P1 {order_p1:+.3}, P2 {order_p2:+.3}");

    assert!(
        order_p1 > 1.6 && order_p1 < 2.4,
        "P1 must converge at second order on this field for the comparison to mean \
         anything; got {order_p1:+.3}"
    );
    assert!(
        order_p2 > 2.7,
        "the quadratic element must reach third order in cell L2; got {order_p2:+.3} \
         against P1's {order_p1:+.3}. This is the acceptance criterion for P2 and is not \
         a band to widen"
    );
}

/// What the **P1** element does on this study, pinned so that it keeps doing it.
///
/// ⚠️ **This test outlived the contract it was written under, and the change is
/// worth recording.** It was first written as the guard half of a pair whose
/// other half, `p2_must_reach_third_order_and_p1_does_not`, was `#[ignore]`d
/// until a quadratic element existed — on the assumption that P2 would
/// *replace* P1 in `run_level`, so that exactly one of the two would be green
/// at any time and the upper edge of the band below would force the ignore
/// attribute off in the same diff.
///
/// **P2 landed as a separate entry point instead** ([`solve_quadratic`]), so
/// P1 stays, and "P1 converges at second order on a perturbed lattice" stays
/// true. Deleting this test would throw away a measurement that is still
/// correct. The pair was therefore dissolved:
/// [`p2_reaches_third_order_where_p1_reaches_second`] now carries **both** arms
/// itself, on one field and one mesh sequence, and this test keeps its own
/// separate job.
///
/// That job is two-sided:
///
/// - The **lower edge** catches a P1 regression — a broken `B` matrix, a wrong
///   consistent load, a mesh that stopped being conforming. Measured: dividing
///   the consistent load by 24 instead of 20 drops the order to `+0.074`.
/// - The **upper edge** catches the study being quietly re-pointed at the
///   quadratic element. `run_level` here is P1 by construction and the numbers
///   below are P1 numbers; if a future change routes it through
///   `solve_quadratic`, the order jumps past 2.4 and this test says so rather
///   than silently re-labelling P2 results as P1 ones.
#[test]
fn characterises_the_p1_second_order_limit_on_a_perturbed_mesh() {
    let cells = [4usize, 8, 16];
    let levels = study(
        "the P1 guard: second order in cell L2 on the same perturbed mesh",
        JITTER,
        &cells,
    );

    // The solves have to have converged before the order means anything — the
    // same precondition the locking sweeps carry, for the same reason.
    for l in &levels {
        assert!(
            l.residual <= 1.0e-9,
            "h={:.4}: the solve stopped at residual {:.3e} after {} iterations, so the \
             order below would be measuring the conjugate gradient",
            l.h,
            l.residual,
            l.iterations
        );
    }

    let p = order(levels[0].cell_l2, levels[1].cell_l2, levels[2].cell_l2);
    eprintln!("[p2oracle] P1 cell L2 order = {p:+.3} (measured +1.895 at d90b18e, P1 study)");
    assert!(
        p > 1.6,
        "P1 must still converge at second order in cell L2; got {p:+.3}. This is a \
         regression in the element, the consistent load or the mesh — not a signal \
         to widen the band"
    );
    assert!(
        p < 2.4,
        "cell L2 converged at {p:+.3}, which is past what a second-order element can \
         do. `run_level` in this file is the P1 study; if it has been re-pointed at \
         `quadratic_elastic_fem::solve_quadratic`, this test has done its job — put it \
         back, and put the P2 measurement in \
         `p2_reaches_third_order_where_p1_reaches_second`, which already runs both \
         elements on one field"
    );
}
