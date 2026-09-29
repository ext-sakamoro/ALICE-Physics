//! Locking measurement for the P1 tetrahedra in `alice_physics::linear_elastic_fem`.
//!
//! "P1 tetrahedra are too stiff in bending" is already measured by
//! `analytic_fem_convergence.rs`, but that study varies only the cell size, so
//! the stiffness it sees is a single number with two causes mixed into it.
//! There are two separate mechanisms, they respond to different sweeps, and a
//! higher-order element is a remedy for one of them and not the other:
//!
//! - **Shear locking** — a constant-strain element cannot bend. Under a pure
//!   bending field a linear tetrahedron develops a spurious shear strain whose
//!   energy grows with the element's length-to-thickness ratio, so the element
//!   stiffens as the beam gets more slender *at a fixed mesh*. The sweep that
//!   isolates it holds the element shape and the Poisson ratio fixed and varies
//!   the slenderness of the beam.
//! - **Volumetric locking** — as `ν → 1/2` the bulk modulus diverges and the
//!   element, which has only one strain state, must satisfy the incompressibility
//!   constraint on every element at once. The constraint count overwhelms the
//!   degrees of freedom and the displacement is driven towards zero. The sweep
//!   that isolates it holds the geometry and the mesh fixed and varies `ν`.
//!
//! Both are reported as `FEM / beam`, a ratio below one. A locking element
//! drives the ratio towards zero along its own sweep while the other sweep
//! stays put; that separation is the point of the file.
//!
//! # The iteration count is part of the measurement
//!
//! Conditioning worsens along *both* sweeps — as `(L/t)²` for slenderness and
//! as `1/(1−2ν)` for the Poisson ratio — so a small ratio has two possible
//! causes: the element locked, or the conjugate gradient stopped early. These
//! are not distinguishable from the deflection alone. Every row therefore
//! carries its iteration count and its achieved relative residual, and the
//! tests assert that the solve reached its tolerance before they read anything
//! into the ratio. A run that stops on the iteration budget is a failed
//! measurement, not a stiff element.

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
    solve, stiffness_diagonal_stats, Axis, BoundaryConditions, ElasticMaterial, FemSolution,
    Preconditioner, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{generate, SdfTetMesh};
use std::collections::HashMap;

/// Young's modulus (MPa). Shared by every row so the ratios are comparable.
const E_MPA: f64 = 3500.0;

/// The end load every sweep uses (N).
///
/// Far above anything a 10 mm printed part would carry, and deliberately so:
/// the problem is linear, so this does not move `FEM / beam`, and it divides the
/// reachable relative residual by the same factor. See `Beam::load`.
///
/// ⚠️ The usable range is **two-sided**, and both ends were measured by
/// `reachable_residual_scales_with_the_load` at `ν = 0.49`:
///
/// | load (N) | outcome |
/// |---|---|
/// | 4 | stagnates at `1.035e-9`, above the `2⁻³⁰` tolerance |
/// | 400 | converges, 408 iterations, ratio 0.7795 |
/// | 40 000 | converges, 406 iterations, ratio 0.7795 — **the same ratio** |
/// | 4 000 000 | stagnates at relative residual **exactly 1.0**, 201 iterations |
///
/// The bottom end is the `2⁻³² / ‖b‖` floor. The top end is saturation: the
/// residual never decreases at all, so a load chosen "generously large" is not
/// safe either. 400 sits in the middle of the measured window.
const LOAD_FOR_HEADROOM: f64 = 400.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

// ---------------------------------------------------------------------------
// the specimen
// ---------------------------------------------------------------------------

/// One cantilever: geometry, material and resolution.
#[derive(Clone, Copy, Debug)]
struct Beam {
    /// Span along x (mm).
    length: f64,
    /// Width along y (mm).
    width: f64,
    /// Thickness along z (mm), the bending direction.
    thickness: f64,
    /// Lattice cell (mm). Must divide all three extents exactly.
    cell: f64,
    /// Poisson's ratio.
    nu: f64,
    /// Total end load (N) along −z.
    ///
    /// The problem is linear, so the ratio `FEM / beam` does not depend on it.
    /// The *relative residual floor* does: `‖r‖` cannot fall below about
    /// `2⁻³²` in `Fix128`, so the smallest reachable `‖r‖/‖b‖` is that floor
    /// divided by `‖b‖`. Raising the load is therefore the one lever that buys
    /// residual headroom without changing the answer being measured.
    load: f64,
}

impl Beam {
    fn slenderness(&self) -> f64 {
        self.length / self.thickness
    }

    /// `δ = PL³/(3EI) + PL/(κGA)`, bending plus shear, `κ = 5/6`.
    ///
    /// The bending term does not involve `ν`; the shear term does, through
    /// `G = E/(2(1+ν))`. Over the Poisson sweep below the target therefore
    /// moves by well under one percent, which is why a ratio that falls by a
    /// factor of several is the element and not the yardstick.
    fn beam_tip_mm(&self) -> f64 {
        let i = self.width * self.thickness.powi(3) / 12.0;
        let bending = self.load * self.length.powi(3) / (3.0 * E_MPA * i);
        let g = E_MPA / (2.0 * (1.0 + self.nu));
        let a = self.width * self.thickness;
        let shear = self.load * self.length / ((5.0 / 6.0) * g * a);
        bending + shear
    }

    fn sdf(&self) -> ClosureSdf {
        let (l, w, t) = (self.length as f32, self.width as f32, self.thickness as f32);
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

    /// Mesh the block, and assert the meshed region *is* the block.
    ///
    /// `generate` emits only cubes whose eight corners are inside the field, so
    /// for a general shape the domain moves with the cell size. Here every
    /// extent is a whole number of cells, so the closed form for the lattice
    /// holds exactly, and a sweep that silently lost a layer of cells would be
    /// comparing different specimens.
    fn mesh(&self) -> SdfTetMesh {
        let nx = (self.length / self.cell).round() as usize;
        let ny = (self.width / self.cell).round() as usize;
        let nz = (self.thickness / self.cell).round() as usize;
        let mesh = generate(
            &self.sdf(),
            [0.0, 0.0, 0.0],
            [self.length as f32, self.width as f32, self.thickness as f32],
            self.cell as f32,
        );
        assert!(
            mesh.tet_count() == 5 * nx * ny * nz,
            "L={} t={} cell={}: the generator emitted {} tets, not the {} the \
             lattice calls for, so the meshed region is not the beam",
            self.length,
            self.thickness,
            self.cell,
            mesh.tet_count(),
            5 * nx * ny * nz
        );
        assert!(
            mesh.vertex_count() == (nx + 1) * (ny + 1) * (nz + 1),
            "L={} t={} cell={}: the generator emitted {} vertices, not the {} the \
             lattice calls for",
            self.length,
            self.thickness,
            self.cell,
            mesh.vertex_count(),
            (nx + 1) * (ny + 1) * (nz + 1)
        );
        mesh
    }
}

// ---------------------------------------------------------------------------
// boundary conditions (same construction as analytic_fem_convergence.rs)
// ---------------------------------------------------------------------------

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

/// Clamp `x = 0`; load the `x = L` face with a consistent traction summing to
/// the beam's `load` along −z.
fn cantilever_bc(beam: &Beam, mesh: &SdfTetMesh) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    let eps = 1.0e-4_f32;
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[0].abs() < eps {
            bc.fix(u32::try_from(v).expect("fits"));
        }
    }

    let end = beam.length as f32;
    let traction = beam.load / (beam.width * beam.thickness);
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
        (applied + beam.load).abs() < 1.0e-6 * beam.load.max(1.0),
        "the consistent nodal loads must sum to the applied force: got {applied}, want {}",
        -beam.load
    );
    bc
}

/// Mean −z displacement over the loaded end face.
fn tip_deflection_mm(beam: &Beam, mesh: &SdfTetMesh, out: &FemSolution) -> f64 {
    let end = beam.length as f32;
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

// ---------------------------------------------------------------------------
// one row
// ---------------------------------------------------------------------------

/// One measured specimen. `ratio` is the headline; the rest is what decides
/// whether the headline means anything.
#[derive(Clone, Copy, Debug)]
struct Row {
    ratio: f64,
    tip_mm: f64,
    beam_mm: f64,
    tets: usize,
    free_dofs: usize,
    iterations: u32,
    residual: f64,
    diagonal_spread: f64,
}

fn measure(beam: &Beam, config: &SolverConfig) -> Row {
    let mesh = beam.mesh();
    let material = ElasticMaterial::new(fx(E_MPA), fx(beam.nu)).expect("nu in (-1, 0.5)");
    let bc = cantilever_bc(beam, &mesh);
    let stats = stiffness_diagonal_stats(&mesh, &material, &bc).unwrap_or_else(|e| {
        panic!(
            "L={} nu={}: diagonal stats failed: {e:?}",
            beam.length, beam.nu
        )
    });
    let out = solve(&mesh, &material, &bc, config).unwrap_or_else(|e| {
        panic!(
            "L={} t={} nu={} cell={}: solve failed: {e:?} (free dofs {}, diagonal spread {:.1})",
            beam.length,
            beam.thickness,
            beam.nu,
            beam.cell,
            stats.free_dofs,
            stats.max.to_f64() / stats.min.to_f64()
        )
    });
    let tip_mm = tip_deflection_mm(beam, &mesh, &out);
    let beam_mm = beam.beam_tip_mm();
    Row {
        ratio: tip_mm / beam_mm,
        tip_mm,
        beam_mm,
        tets: mesh.tet_count(),
        free_dofs: stats.free_dofs,
        iterations: out.iterations,
        residual: out.relative_residual.to_f64(),
        diagonal_spread: stats.max.to_f64() / stats.min.to_f64(),
    }
}

fn header(label: &str) {
    eprintln!("[locking] ---- {label} ----");
    eprintln!(
        "[locking] {:>9}  {:>7}  {:>6}  {:>8}  {:>10}  {:>10}  {:>6}  {:>10}  {:>9}",
        "sweep",
        "ratio",
        "tets",
        "freedof",
        "tip(mm)",
        "beam(mm)",
        "iters",
        "residual",
        "diagspread"
    );
}

fn line(sweep: String, r: &Row) {
    eprintln!(
        "[locking] {:>9}  {:>7.4}  {:>6}  {:>8}  {:>10.6}  {:>10.6}  {:>6}  {:>10.3e}  {:>9.1}",
        sweep,
        r.ratio,
        r.tets,
        r.free_dofs,
        r.tip_mm,
        r.beam_mm,
        r.iterations,
        r.residual,
        r.diagonal_spread
    );
}

/// The tolerance the sweeps solve to, and the budget.
///
/// `2⁻³⁰` is the tolerance the convergence study uses; the budget is large
/// because conditioning is the thing being pushed on.
fn sweep_config() -> SolverConfig {
    SolverConfig::try_new(400_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_preconditioner(Preconditioner::JacobiScaled)
}

// ---------------------------------------------------------------------------
// sweep 1: volumetric locking — geometry fixed, ν varies
// ---------------------------------------------------------------------------

/// The highest Poisson ratio this solver reaches on this problem.
///
/// Not a modelling choice: `the_load_window_closes_as_the_poisson_ratio_approaches_one_half`
/// and `at_nu_0_499_the_iteration_takes_no_step` measure that 0.499 does not
/// solve at any load, with either preconditioner, while the stiffness diagonal
/// stays perfectly ordinary. The sweep stops here because past here there is no
/// answer to read, not because the element stops being interesting.
const HIGHEST_SOLVABLE_NU: f64 = 0.49;

/// Volumetric locking: the same beam and the same mesh, stiffened only by
/// pushing `ν` towards the incompressible limit.
///
/// `ν` enters the yardstick only through the shear term, which is 3.2% of the
/// tip deflection and changes by 15% across the sweep — so the target moves by
/// about half a percent while the ratio moves by several. Anything that large
/// is the element.
#[test]
fn volumetric_locking_by_poisson_ratio() {
    let config = sweep_config();
    let nus = [0.3, 0.45, HIGHEST_SOLVABLE_NU];
    header("volumetric: L=10 t=2 (slenderness 5), cell 0.5, nu varies");
    let rows: Vec<Row> = nus
        .iter()
        .map(|nu| {
            let beam = Beam {
                length: 10.0,
                width: 2.0,
                thickness: 2.0,
                cell: 0.5,
                nu: *nu,
                load: LOAD_FOR_HEADROOM,
            };
            let r = measure(&beam, &config);
            line(format!("nu={nu}"), &r);
            r
        })
        .collect();

    for (nu, r) in nus.iter().zip(&rows) {
        assert!(
            r.residual <= 1.0e-9,
            "nu={nu}: the solve stopped at residual {:.3e} after {} iterations, so its \
             ratio {:.4} measures the conjugate gradient and not the element",
            r.residual,
            r.iterations,
            r.ratio
        );
    }

    // The signature of volumetric locking: at a fixed mesh, approaching
    // incompressibility makes the element stiffer.
    for w in rows.windows(2) {
        assert!(
            w[1].ratio < w[0].ratio,
            "approaching incompressibility must stiffen a locking element; \
             got {:.4} then {:.4}",
            w[0].ratio,
            w[1].ratio
        );
    }

    // The mesh is the one the through-thickness sweep calls `n_t = 4`, so the
    // nu = 0.3 row is the same specimen and must agree with it. A drift here
    // would mean the two sweeps are not measuring the same thing.
    assert!(
        (rows[0].ratio - 0.8749).abs() < 5.0e-4,
        "nu=0.3 at cell 0.5 is the n_t=4 row of the other sweep; got {:.4}, want 0.8749",
        rows[0].ratio
    );
}

/// Pins the limit itself, so that a solver change reopens the study.
///
/// If this test ever fails because `nu = 0.499` *solved*, that is not a
/// regression: it means `HIGHEST_SOLVABLE_NU` can be raised and the volumetric
/// sweep extended. The value of pinning it is that the extension is not
/// forgotten.
#[test]
fn nu_0_499_does_not_solve_at_any_load_in_the_window() {
    let beam = |load| Beam {
        length: 10.0,
        width: 2.0,
        thickness: 2.0,
        cell: 0.5,
        nu: 0.499,
        load,
    };
    let config = sweep_config();
    for load in [0.4, 4.0, 40.0, 400.0, 4_000.0, 40_000.0] {
        let b = beam(load);
        let mesh = b.mesh();
        let material = ElasticMaterial::new(fx(E_MPA), fx(b.nu)).expect("nu in (-1, 0.5)");
        let bc = cantilever_bc(&b, &mesh);
        let outcome = solve(&mesh, &material, &bc, &config);
        assert!(
            outcome.is_err(),
            "nu=0.499 solved at load {load} — the conjugate gradient has improved, so \
             raise HIGHEST_SOLVABLE_NU and extend volumetric_locking_by_poisson_ratio"
        );
    }
}

// ---------------------------------------------------------------------------
// sweep 2: the bending stiffness excess — which knob actually moves it
// ---------------------------------------------------------------------------

/// Slenderness at a fixed element shape and a fixed count through the
/// thickness.
///
/// This is the sweep that classical shear locking responds to — but only when
/// the element is allowed to become long and thin. `sdf_fem_mesh::generate`
/// takes **one** cell size for all three axes, so every element it can produce
/// is a cube diced five ways and the element aspect ratio is pinned at 1. The
/// sweep below therefore holds the element shape fixed and varies only how many
/// of them span the beam.
///
/// ⚠️ **The measured answer is that the ratio does not degrade.** It drifts
/// *upward* with slenderness (0.6457 → 0.6794 over 2 → 20), which is the
/// opposite of locking: the Euler-Bernoulli yardstick becomes more valid as the
/// beam gets more slender, and more elements span it. So the "P1 is too stiff in
/// bending" that `analytic_fem_convergence.rs` records is **not** aspect-ratio
/// shear locking, and it cannot be provoked into becoming that through this
/// mesher. What sets it is the count through the thickness — see
/// `bending_stiffness_excess_by_elements_through_thickness`.
#[test]
fn slenderness_does_not_worsen_the_bending_stiffness_excess() {
    let config = sweep_config();
    let lengths = [4.0, 10.0, 20.0, 40.0];
    header("slenderness: t=2 w=2, cell 1.0 (2 elements through thickness), nu=0.3, L varies");
    let rows: Vec<Row> = lengths
        .iter()
        .map(|l| {
            let beam = Beam {
                length: *l,
                width: 2.0,
                thickness: 2.0,
                cell: 1.0,
                nu: 0.3,
                load: LOAD_FOR_HEADROOM,
            };
            let r = measure(&beam, &config);
            line(format!("L/t={}", beam.slenderness()), &r);
            r
        })
        .collect();

    for (l, r) in lengths.iter().zip(&rows) {
        assert!(
            r.residual <= 1.0e-9,
            "L={l}: the solve stopped at residual {:.3e} after {} iterations, so its \
             ratio {:.4} measures the conjugate gradient and not the element",
            r.residual,
            r.iterations,
            r.ratio
        );
    }

    // The statement this file exists to make about slenderness: it does not
    // make things worse. A ratio that fell along this sweep would mean the
    // mesher had started producing elongated elements, which it cannot.
    for w in rows.windows(2) {
        assert!(
            w[1].ratio >= w[0].ratio,
            "the element shape is fixed at a cube, so slenderness must not stiffen the \
             result; got {:.4} then {:.4}",
            w[0].ratio,
            w[1].ratio
        );
    }
}

/// Elements through the thickness, at a fixed slenderness.
///
/// This is the knob that sets the bending stiffness excess, and the one a
/// higher-order element is meant to relieve: a P1 tetrahedron carries one
/// constant strain, so resolving a through-thickness strain gradient costs
/// elements. The row at one element through the thickness is the number a P2
/// element has to beat.
#[test]
fn bending_stiffness_excess_by_elements_through_thickness() {
    let config = sweep_config();
    let cells = [2.0, 1.0, 0.5, 0.25];
    header("through-thickness: L=10 t=2 w=2 (slenderness 5), nu=0.3, cell varies");
    let rows: Vec<Row> = cells
        .iter()
        .map(|cell| {
            let beam = Beam {
                length: 10.0,
                width: 2.0,
                thickness: 2.0,
                cell: *cell,
                nu: 0.3,
                load: LOAD_FOR_HEADROOM,
            };
            let r = measure(&beam, &config);
            line(format!("n_t={}", (2.0 / cell).round() as usize), &r);
            r
        })
        .collect();

    for (c, r) in cells.iter().zip(&rows) {
        assert!(
            r.residual <= 1.0e-9,
            "cell={c}: the solve stopped at residual {:.3e} after {} iterations, so its \
             ratio {:.4} measures the conjugate gradient and not the element",
            r.residual,
            r.iterations,
            r.ratio
        );
    }
    for w in rows.windows(2) {
        assert!(
            w[1].ratio > w[0].ratio,
            "refinement through the thickness must soften an over-stiff element; \
             got {:.4} then {:.4}",
            w[0].ratio,
            w[1].ratio
        );
    }
    for r in &rows {
        assert!(
            r.ratio < 1.0,
            "P1 tetrahedra cannot be softer than the beam; got {:.4}",
            r.ratio
        );
    }
}

// ---------------------------------------------------------------------------
// probe: the load window, and where it closes
// ---------------------------------------------------------------------------

/// Maps the two-sided load window against the Poisson ratio.
///
/// The stopping rule is relative, and `Fix128` bounds the residual *norm* from
/// below (`rᵀr` bottoms out at `2⁻⁶⁴`, so `‖r‖` cannot go under about `2⁻³²`).
/// The smallest reachable `‖r‖/‖b‖` is therefore `2⁻³² / ‖b‖`, which **falls**
/// as the load rises. At the other end the inner products saturate and the
/// residual stops moving at all. So there is a window, not a direction.
///
/// ⚠️ The window is not fixed: as `ν → ½` the first Lamé parameter diverges, `K`
/// grows with it, and the saturating end comes down to meet the floor. Where
/// the two meet, **no load solves the problem** — and the failure is not a
/// message about incompressibility, it is `Stagnated`. That is the limit this
/// probe exists to locate, because every ratio reported above it would
/// otherwise be read as an element property.
///
/// Printed as a map. What is asserted is the part that is arithmetic and not
/// conditioning: where two loads both converge, they must agree on the answer.
#[test]
fn the_load_window_closes_as_the_poisson_ratio_approaches_one_half() {
    let config = sweep_config();
    let nus = [0.49, 0.499, 0.4999];
    let loads = [0.4, 4.0, 40.0, 400.0, 4_000.0, 40_000.0];
    eprintln!("[locking] ---- probe: load window, L=10 t=2 cell=0.5 ----");
    eprintln!(
        "[locking] {:>8}  {:>10}  {:>8}  {:>6}  {:>10}",
        "nu", "load(N)", "ratio", "iters", "residual"
    );
    let mut solved_any = false;
    for nu in nus {
        let mut ratios: Vec<(f64, f64)> = Vec::new();
        for load in loads {
            let beam = Beam {
                length: 10.0,
                width: 2.0,
                thickness: 2.0,
                cell: 0.5,
                nu,
                load,
            };
            let mesh = beam.mesh();
            let material = ElasticMaterial::new(fx(E_MPA), fx(beam.nu)).expect("nu in (-1, 0.5)");
            let bc = cantilever_bc(&beam, &mesh);
            match solve(&mesh, &material, &bc, &config) {
                Ok(out) => {
                    let ratio = tip_deflection_mm(&beam, &mesh, &out) / beam.beam_tip_mm();
                    eprintln!(
                        "[locking] {nu:>8}  {load:>10.1}  {ratio:>8.4}  {:>6}  {:>10.3e}",
                        out.iterations,
                        out.relative_residual.to_f64()
                    );
                    ratios.push((load, ratio));
                    solved_any = true;
                }
                Err(e) => {
                    let tag = match e {
                        alice_physics::linear_elastic_fem::FemError::Stagnated {
                            relative_residual,
                            iterations,
                            ..
                        } => format!(
                            "stagnated {:>6}  {:>10.3e}",
                            iterations,
                            relative_residual.to_f64()
                        ),
                        other => format!("{other:?}"),
                    };
                    eprintln!("[locking] {nu:>8}  {load:>10.1}  {:>8}  {tag}", "-");
                }
            }
        }
        // Where the window is open at all, its width is a free choice: the
        // problem is linear, so two loads inside it must give the same answer.
        if let Some((first_load, first_ratio)) = ratios.first().copied() {
            for (load, ratio) in &ratios[1..] {
                assert!(
                    (ratio - first_ratio).abs() < 1.0e-6,
                    "nu={nu}: the problem is linear, so loads {first_load} and {load} must \
                     agree; got {first_ratio:.8} then {ratio:.8}"
                );
            }
        }
    }
    assert!(
        solved_any,
        "the probe measures nothing if no (nu, load) pair solves at all"
    );
}

/// Which side of the solve stops at `ν = 0.499`.
///
/// The map above shows `Stagnated` with a relative residual of **exactly 1.0**
/// and `without_improvement` equal to the whole window, at every load from 0.4
/// to 40 000. A residual that never leaves its starting value is not a stiff
/// element and not a saturating load: the iteration is taking no step at all.
///
/// This narrows it to two candidates, and reports both so the next reader does
/// not have to re-derive them — the stiffness is unrepresentable (visible in the
/// diagonal), or the step length underflows (the diagonal is fine and both
/// preconditioners behave the same).
#[test]
fn at_nu_0_499_the_iteration_takes_no_step() {
    let beam = Beam {
        length: 10.0,
        width: 2.0,
        thickness: 2.0,
        cell: 0.5,
        nu: 0.499,
        load: 400.0,
    };
    let mesh = beam.mesh();
    let material = ElasticMaterial::new(fx(E_MPA), fx(beam.nu)).expect("nu in (-1, 0.5)");
    let bc = cantilever_bc(&beam, &mesh);

    let lame_lambda = E_MPA * beam.nu / ((1.0 + beam.nu) * (1.0 - 2.0 * beam.nu));
    let stats = stiffness_diagonal_stats(&mesh, &material, &bc).expect("stats");
    eprintln!(
        "[locking] nu=0.499: lame lambda {:.4e} MPa, diagonal min {:.6e} mean {:.6e} \
         max {:.6e}, spread {:.2}, free dofs {}",
        lame_lambda,
        stats.min.to_f64(),
        stats.mean.to_f64(),
        stats.max.to_f64(),
        stats.max.to_f64() / stats.min.to_f64(),
        stats.free_dofs
    );
    eprintln!(
        "[locking] nu=0.499: diagonal min raw {:?}, max raw {:?}",
        (stats.min.hi, stats.min.lo),
        (stats.max.hi, stats.max.lo)
    );

    for pc in [Preconditioner::None, Preconditioner::JacobiScaled] {
        let config = SolverConfig::try_new(400_000, Fix128::from_raw(0, 1 << 34))
            .expect("valid")
            .with_preconditioner(pc);
        match solve(&mesh, &material, &bc, &config) {
            Ok(out) => eprintln!(
                "[locking] nu=0.499 {pc:?}: converged in {} iterations, residual {:.3e}",
                out.iterations,
                out.relative_residual.to_f64()
            ),
            Err(e) => eprintln!("[locking] nu=0.499 {pc:?}: {e:?}"),
        }
    }

    // The diagonal is the only part that can be pinned without deciding which
    // candidate is right: if it were unrepresentable the stats call would not
    // have produced finite, positive, well-separated numbers.
    assert!(
        stats.min.to_f64() > 0.0 && stats.max.to_f64() > stats.min.to_f64(),
        "the stiffness diagonal at nu=0.499 must at least be positive and ordered \
         before 'the iteration takes no step' can mean anything: min {:?} max {:?}",
        stats.min,
        stats.max
    );
}
