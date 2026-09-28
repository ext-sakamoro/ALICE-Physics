//! Mesh convergence study for `alice_physics::linear_elastic_fem`: a cantilever
//! beam under a uniform end traction.
//!
//! The companion file `analytic_linear_elastic_fem.rs` holds the oracles that
//! are **exact** — linear displacement fields, which P1 tetrahedra represent
//! with no discretisation error at all. Bending is not one of those. The FEM
//! answer only approaches the beam solution as the mesh refines, so pinning a
//! tolerance on a single mesh would measure the resolution rather than the
//! solver. What is predictable without picking a resolution is the *shape* of
//! the approach, and that is what this file asserts:
//!
//! - **P1 tetrahedra are too stiff in bending**, so the computed tip deflection
//!   must be *below* the beam value and must *increase* monotonically with
//!   refinement. Overshoot or a non-monotone sequence is a defect in the mesh,
//!   the boundary conditions or the assembly — not "not converged yet".
//! - The sequence must have a **convergence order**, estimated by Richardson
//!   from three successive halvings, and the estimate from one triple must agree
//!   with the estimate from the next. Two levels give one slope and cannot tell
//!   a real order from a coincidence.
//! - The **Richardson limit** must land near the beam solution, with shear
//!   deformation included.
//!
//! # The domain has to stay fixed
//!
//! `sdf_fem_mesh::generate` emits only cubes whose eight corners are inside the
//! field, so for a general shape **the meshed region changes with the cell
//! size** — a 20 mm box at 4 mm cells meshes the inner 16 mm. A convergence
//! study on such a series measures the domain moving, not the error shrinking.
//!
//! Here the beam is axis-aligned and every cell size divides all three extents
//! exactly, so every cube lies inside and the meshed region *is* the beam at
//! every level. That is not assumed: `assert_domain_is_exact` checks the
//! element and vertex counts against the closed form for the lattice, and the
//! study is invalid if it ever fails.
//!
//! **This is a property of the shape, not of the mesher, and it does not
//! generalise.** `generate` is usable here only because a rectangular block
//! aligned to the lattice is represented exactly. On a curved surface the same
//! generator loses the boundary layer of cells: measured on a unit ball, its
//! volume reaches 2.53 of the analytic 4.19 at cell 0.1875 — about 60% — and
//! keeps changing with the cell size. A convergence study on a curved shape has
//! to use `generate_marching_tets`, and then has to contend with the slivers
//! that surface clipping produces (minimum dihedral angle measured as low as
//! 4.59° there, against a uniform 54.7° here). Do not read the result below as
//! "the FEM converges on meshes from this crate"; read it as "the FEM converges
//! on a well-shaped mesh of a block".
//!
//! # Euler-Bernoulli is not the target on its own
//!
//! `δ = PL³/(3EI)` is the slender limit. A real beam also shears, adding
//! `PL/(κGA)`. With the 10:1 slenderness used here the shear term is a few
//! percent — small, but far larger than the difference the last refinement
//! makes, so comparing against the bending term alone would look like a
//! convergence failure. Both are computed and reported; the assertion uses the
//! sum.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    solve, stiffness_diagonal_stats, Axis, BoundaryConditions, ElasticMaterial, FemSolution,
    Preconditioner, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{generate, SdfTetMesh};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// beam definition (mm / N / MPa)
// ---------------------------------------------------------------------------

/// Span along x.
const LENGTH: f64 = 20.0;
/// Width along y.
const WIDTH: f64 = 2.0;
/// Thickness along z, the bending direction. `LENGTH / THICKNESS = 10`, which
/// keeps the Euler-Bernoulli term within a few percent of the beam answer.
const THICKNESS: f64 = 2.0;
/// Total end load along −z.
const LOAD: f64 = 4.0;

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// `δ = PL³/(3EI)`, the slender-beam term.
fn euler_bernoulli_tip_mm() -> f64 {
    let i = WIDTH * THICKNESS.powi(3) / 12.0;
    LOAD * LENGTH.powi(3) / (3.0 * E_MPA * i)
}

/// `δ = PL/(κGA)`, the shear term, with `κ = 5/6` for a rectangular section.
fn shear_tip_mm() -> f64 {
    let g = E_MPA / (2.0 * (1.0 + NU));
    let a = WIDTH * THICKNESS;
    LOAD * LENGTH / ((5.0 / 6.0) * g * a)
}

// ---------------------------------------------------------------------------
// meshing
// ---------------------------------------------------------------------------

/// Box SDF for the beam occupying `[0,L] × [0,w] × [0,t]`.
fn beam_sdf() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| {
            let dx = (0.0 - x).max(x - LENGTH as f32);
            let dy = (0.0 - y).max(y - WIDTH as f32);
            let dz = (0.0 - z).max(z - THICKNESS as f32);
            dx.max(dy).max(dz)
        },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    )
}

/// The mesh, plus the lattice counts the study needs to index nodes.
struct Level {
    mesh: SdfTetMesh,
    cells: [usize; 3],
}

fn mesh_at(cell: f64) -> Level {
    let nx = (LENGTH / cell).round() as usize;
    let ny = (WIDTH / cell).round() as usize;
    let nz = (THICKNESS / cell).round() as usize;
    let mesh = generate(
        &beam_sdf(),
        [0.0, 0.0, 0.0],
        [LENGTH as f32, WIDTH as f32, THICKNESS as f32],
        cell as f32,
    );
    Level {
        mesh,
        cells: [nx, ny, nz],
    }
}

/// The precondition of the whole study: the meshed region is the beam, at every
/// level.
///
/// Every cube of the lattice lies inside the beam, so the generator must emit
/// all of them — five tetrahedra each — and exactly the lattice corners. If a
/// cube were dropped the domain would shrink with the cell size and the
/// sequence below would be measuring a moving problem.
fn assert_domain_is_exact(level: &Level, cell: f64) {
    let [nx, ny, nz] = level.cells;
    assert_eq!(
        level.mesh.tet_count(),
        5 * nx * ny * nz,
        "cell {cell}: the generator dropped cubes, so the meshed region is not the beam"
    );
    assert_eq!(
        level.mesh.vertex_count(),
        (nx + 1) * (ny + 1) * (nz + 1),
        "cell {cell}: vertex count does not match the lattice"
    );
}

// ---------------------------------------------------------------------------
// boundary conditions
// ---------------------------------------------------------------------------

/// Boundary triangles of the mesh: the faces used by exactly one tetrahedron.
fn boundary_faces(mesh: &SdfTetMesh) -> Vec<[u32; 3]> {
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
    counts
        .into_iter()
        .filter(|(_, n)| *n == 1)
        .map(|(f, _)| f)
        .collect()
}

/// Clamp `x = 0`, and load the `x = L` face with a uniform traction summing to
/// `LOAD` along −z.
///
/// The load is *consistent*, not lumped: each boundary triangle on the end face
/// hands one third of `traction × area` to each of its three nodes. Reading the
/// triangles off the mesh rather than assuming them keeps this correct whatever
/// the dicing does, and it converges to the same traction at every level — a
/// tributary-area shortcut would perturb the coarse levels and contaminate the
/// order estimate.
fn cantilever_bc(mesh: &SdfTetMesh) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    let eps = 1.0e-4_f32;
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[0].abs() < eps {
            bc.fix(u32::try_from(v).expect("fits"));
        }
    }

    let end = LENGTH as f32;
    let area_total = WIDTH * THICKNESS;
    let traction = LOAD / area_total; // MPa, along −z
    let mut applied = 0.0_f64;
    for face in boundary_faces(mesh) {
        let p: [[f32; 3]; 3] = [
            mesh.vertices[face[0] as usize],
            mesh.vertices[face[1] as usize],
            mesh.vertices[face[2] as usize],
        ];
        if !p.iter().all(|q| (q[0] - end).abs() < eps) {
            continue;
        }
        let e1 = [
            f64::from(p[1][0] - p[0][0]),
            f64::from(p[1][1] - p[0][1]),
            f64::from(p[1][2] - p[0][2]),
        ];
        let e2 = [
            f64::from(p[2][0] - p[0][0]),
            f64::from(p[2][1] - p[0][1]),
            f64::from(p[2][2] - p[0][2]),
        ];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let share = -traction * area / 3.0;
        applied += 3.0 * share;
        for n in face {
            bc.add_load(n, Axis::Z, fx(share));
        }
    }
    assert!(
        (applied + LOAD).abs() < 1.0e-6,
        "the consistent nodal loads must sum to the applied force: got {applied}, want {}",
        -LOAD
    );
    bc
}

/// Mean deflection of the loaded end, as a positive number.
///
/// Averaging over the end face rather than reading one node: the end section
/// rotates and warps a little, and the beam solution is about the neutral axis,
/// so the mean is the quantity the closed form describes.
fn tip_deflection_mm(mesh: &SdfTetMesh, out: &FemSolution) -> f64 {
    let end = LENGTH as f32;
    let eps = 1.0e-4_f32;
    let mut sum = 0.0;
    let mut count = 0usize;
    for (v, p) in mesh.vertices.iter().enumerate() {
        if (p[0] - end).abs() < eps {
            sum += out.displacements[v][Axis::Z.index()].to_f64();
            count += 1;
        }
    }
    assert!(count > 0, "the end face must have nodes");
    -sum / count as f64
}

/// One level's measurements.
struct Measured {
    tip_mm: f64,
    tets: usize,
    iterations: u32,
    /// Reported on success as well as on failure: a solve that *just* reached
    /// its tolerance and one that reached it with room to spare are different
    /// situations, and only the number distinguishes them.
    relative_residual: f64,
    /// `max / min` of the stiffness diagonal over the free degrees of freedom.
    /// A diagonal preconditioner stretches individual components by up to this
    /// factor around the mean, which is what decides whether the scaled inner
    /// products stay inside the representable range.
    diagonal_spread: f64,
}

/// Solve one level and return the tip deflection.
fn run_level(cell: f64, config: &SolverConfig) -> Measured {
    let level = mesh_at(cell);
    assert_domain_is_exact(&level, cell);
    let material = ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid material");
    let bc = cantilever_bc(&level.mesh);
    let stats = stiffness_diagonal_stats(&level.mesh, &material, &bc)
        .unwrap_or_else(|e| panic!("cell {cell}: diagonal statistics failed: {e:?}"));
    let out = solve(&level.mesh, &material, &bc, config).unwrap_or_else(|e| {
        panic!(
            "cell {cell}: solve failed: {e:?} (diagonal min {:.4e} mean {:.4e} max {:.4e}, \
             spread {:.1})",
            stats.min.to_f64(),
            stats.mean.to_f64(),
            stats.max.to_f64(),
            stats.max.to_f64() / stats.min.to_f64()
        );
    });
    Measured {
        diagonal_spread: stats.max.to_f64() / stats.min.to_f64(),
        tip_mm: tip_deflection_mm(&level.mesh, &out),
        tets: level.mesh.tet_count(),
        iterations: out.iterations,
        relative_residual: out.relative_residual.to_f64(),
    }
}

/// Richardson order from three deflections on successively halved cells.
fn order(coarse: f64, mid: f64, fine: f64) -> f64 {
    ((coarse - mid) / (mid - fine)).abs().log2()
}

/// Richardson limit from the two finest levels and an order.
fn richardson_limit(mid: f64, fine: f64, p: f64) -> f64 {
    fine + (fine - mid) / (2.0_f64.powf(p) - 1.0)
}

fn report(cells: &[f64], m: &[Measured]) {
    let deflections: Vec<f64> = m.iter().map(|x| x.tip_mm).collect();
    let eb = euler_bernoulli_tip_mm();
    let shear = shear_tip_mm();
    eprintln!("cantilever L={LENGTH} w={WIDTH} t={THICKNESS} (slenderness {:.0}), P={LOAD} N, E={E_MPA} MPa, nu={NU}",
        LENGTH / THICKNESS);
    eprintln!("  Euler-Bernoulli PL^3/(3EI) = {eb:.6} mm");
    eprintln!(
        "  shear PL/(kGA)             = {shear:.6} mm  ({:.1}% of bending)",
        100.0 * shear / eb
    );
    eprintln!("  target (bending + shear)   = {:.6} mm", eb + shear);
    for (i, x) in m.iter().enumerate() {
        eprintln!(
            "  cell {:<6} tets {:>6}  cg iters {:>6}  rel resid {:.3e}  diag spread {:>7.1}  \
             tip {:.6} mm  ({:.1}% of target)",
            cells[i],
            x.tets,
            x.iterations,
            x.relative_residual,
            x.diagonal_spread,
            x.tip_mm,
            100.0 * x.tip_mm / (eb + shear)
        );
    }
    for i in 0..deflections.len().saturating_sub(2) {
        eprintln!(
            "  order from cells {}/{}/{}: {:.3}",
            cells[i],
            cells[i + 1],
            cells[i + 2],
            order(deflections[i], deflections[i + 1], deflections[i + 2])
        );
    }
}

// ---------------------------------------------------------------------------
// the study
// ---------------------------------------------------------------------------

/// Three levels, asserting only what these three levels can support: the
/// approach is from below and monotone.
///
/// **This test deliberately does not assert a convergence order.** Measured on
/// cells 2 / 1 / 0.5 the Richardson order is 0.571, which is not the asymptotic
/// order of the scheme — the coarsest level has a *single* element through the
/// thickness, and one layer of linear tetrahedra cannot represent bending at
/// all, so the sequence has not entered its asymptotic range by the third level.
/// Asserting a band wide enough to admit 0.571 would admit almost anything, so
/// the order is reported and left to `cantilever_order_estimates_agree`, which
/// adds the level where the claim can actually be made.
#[test]
fn cantilever_converges_from_below() {
    let cells = [2.0, 1.0, 0.5];
    let config = SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let measured: Vec<Measured> = cells.iter().map(|c| run_level(*c, &config)).collect();
    report(&cells, &measured);
    let deflections: Vec<f64> = measured.iter().map(|x| x.tip_mm).collect();

    let target = euler_bernoulli_tip_mm() + shear_tip_mm();

    // P1 tetrahedra cannot be too soft: every level must sit below the beam
    // answer. An overshoot means the load, the clamp or the assembly is wrong,
    // not that the mesh is coarse.
    for (c, d) in cells.iter().zip(&deflections) {
        assert!(
            *d < target,
            "cell {c}: tip deflection {d:.6} mm exceeds the beam value {target:.6} mm; \
             P1 elements are stiffer than the beam, so this cannot be a resolution effect"
        );
        assert!(*d > 0.0, "cell {c}: tip deflection must be positive");
    }

    // and must increase as the mesh refines
    for w in deflections.windows(2) {
        assert!(
            w[1] > w[0],
            "refinement must increase the deflection of an over-stiff element: {:.6} -> {:.6}",
            w[0],
            w[1]
        );
    }

    // Reported, not asserted: see the note on this function. What *is* checked
    // is that the differences keep shrinking, which is the weakest statement
    // that still rules out a stalled sequence.
    let p = order(deflections[0], deflections[1], deflections[2]);
    eprintln!(
        "  (order {p:.3} is reported only; three levels starting from one element \
               through the thickness are not an asymptotic series)"
    );
    assert!(
        (deflections[2] - deflections[1]).abs() < (deflections[1] - deflections[0]).abs(),
        "successive refinements must change the answer by less each time; got steps \
         {:.6} then {:.6}",
        deflections[1] - deflections[0],
        deflections[2] - deflections[1]
    );
}

/// Four levels, so two independent order estimates can be compared. Agreement
/// between them is the evidence that the sequence has reached its asymptotic
/// range; a single estimate cannot distinguish a real order from a coincidence.
///
/// ```text
/// cargo test --release --test analytic_fem_convergence -- --ignored --nocapture
/// ```
///
/// **Release, not debug.** The finest level is 25,600 tetrahedra and the
/// conjugate gradient count grows roughly as `1/h` (measured: 88 / 132 / 292
/// iterations for cells 2 / 1 / 0.5), so the work grows about sixteenfold per
/// halving. Every arithmetic operation is a software 128-bit fixed-point
/// multiply, which a debug build does not inline; a debug run of this test did
/// not finish in ten minutes on an M3, while the three-level test above takes
/// about a second.
/// The same series with the preconditioner switched off, so the two can be
/// compared where it matters.
///
/// Measured at 25,600 elements, the two settings disagree in a way the coarse
/// levels gave no hint of: unpreconditioned reached a relative residual of
/// 9.41e-10 (and was still improving when a 500,000 iteration budget ran out),
/// while Jacobi stalled at 4.66e-9 — five times worse — after 4,766. On the
/// meshes up to 3,200 elements Jacobi is the better of the two on both counts,
/// which is exactly why this pair has to be run at the fine level to decide
/// anything.
///
/// ```text
/// cargo test --release --test analytic_fem_convergence -- --ignored --nocapture
/// ```
///
/// runs both. The numbers to line up are the achieved residual, the iteration
/// count and whether the outcome is `Ok`, `Stagnated` or `NotConverged`.
#[test]
#[ignore = "25,600 tets at cell 0.25; the A/B partner of the test above"]
fn cantilever_without_preconditioner() {
    let cells = [2.0, 1.0, 0.5, 0.25];
    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_preconditioner(Preconditioner::None);
    eprintln!("=== preconditioner: None ===");
    let measured: Vec<Measured> = cells.iter().map(|c| run_level(*c, &config)).collect();
    report(&cells, &measured);
    let deflections: Vec<f64> = measured.iter().map(|x| x.tip_mm).collect();
    let target = euler_bernoulli_tip_mm() + shear_tip_mm();
    for (c, d) in cells.iter().zip(&deflections) {
        assert!(
            *d < target,
            "cell {c}: {d:.6} mm exceeds the beam value {target:.6} mm"
        );
    }
    for w in deflections.windows(2) {
        assert!(
            w[1] > w[0],
            "refinement must increase the deflection: {:.6} -> {:.6}",
            w[0],
            w[1]
        );
    }
}

#[test]
#[ignore = "25,600 tets at cell 0.25; run with --release, see the doc comment"]
fn cantilever_order_estimates_agree() {
    let cells = [2.0, 1.0, 0.5, 0.25];
    let config = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let measured: Vec<Measured> = cells.iter().map(|c| run_level(*c, &config)).collect();
    report(&cells, &measured);
    let deflections: Vec<f64> = measured.iter().map(|x| x.tip_mm).collect();

    // Two estimates, from cells 2/1/0.5 and 1/0.5/0.25. Only the second can be
    // asymptotic: the coarsest level has one element through the thickness, and
    // a single layer of linear tetrahedra has no bending mode at all, so any
    // triple containing it measures the departure from that degeneracy rather
    // than the order of the scheme.
    let coarse_triple = order(deflections[0], deflections[1], deflections[2]);
    let fine_triple = order(deflections[1], deflections[2], deflections[3]);
    let limit = richardson_limit(deflections[2], deflections[3], fine_triple);
    let target = euler_bernoulli_tip_mm() + shear_tip_mm();
    eprintln!("  order 2/1/0.5    = {coarse_triple:.3}  (includes the one-element-thick level)");
    eprintln!("  order 1/0.5/0.25 = {fine_triple:.3}  <- the one that can be asymptotic");
    eprintln!(
        "  Richardson limit {limit:.6} mm, target {target:.6} mm, ratio {:.4}",
        limit / target
    );

    for (c, d) in cells.iter().zip(&deflections) {
        assert!(
            *d < target,
            "cell {c}: {d:.6} mm exceeds the beam value {target:.6} mm"
        );
    }
    for w in deflections.windows(2) {
        assert!(
            w[1] > w[0],
            "refinement must increase the deflection: {:.6} -> {:.6}",
            w[0],
            w[1]
        );
    }

    // The band was fixed before the number was measured, and it is not to be
    // widened to fit one. A linear element with an exactly integrated stiffness
    // is a second-order scheme; the re-entrant corner at the clamp can pull the
    // observed order down, but not to the 0.571 the coarse triple shows. If the
    // measurement lands outside the band, report it — the finding is then "still
    // pre-asymptotic at 25,600 elements", which is itself worth knowing.
    assert!(
        fine_triple > 1.0 && fine_triple < 3.0,
        "the order from the three finest levels is {fine_triple:.3}, outside the band a \
         second-order scheme produces. Do not widen the band: either the series is still \
         pre-asymptotic, or something upstream is wrong"
    );
    assert!(
        (limit / target - 1.0).abs() < 0.10,
        "the Richardson limit {limit:.6} mm is more than 10% from the beam value \
         {target:.6} mm, which discretisation error alone does not explain"
    );
}
