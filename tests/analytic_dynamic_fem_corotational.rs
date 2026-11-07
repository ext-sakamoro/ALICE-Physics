//! The dynamic co-rotational Newmark solver ([`dynamic_fem::CorotationalTransientSolver`]):
//! per-element rotation by polar decomposition each step, the element stiffness warped by
//! it, the implicit average-acceleration step of [`dynamic_fem::TransientSolver`] otherwise.
//! Three oracles the row asks for, plus the same period oracle against the linear solver as
//! the sanity check of the reference:
//!
//! * **rigid rotation gives zero elastic force**: a body rotated as a whole and released
//!   with no load does not move (the linear solver, by contrast, sees a rotation as a strain
//!   and springs back);
//! * **sag matches the static solve**: a column under its own weight from rest oscillates
//!   about the static sag with no damping, so the mean of its top displacement over whole
//!   periods is the static sag;
//! * **period matches modal analysis**: the oscillation's period is `2π/ω̄` with
//!   `ω̄ = (2/dt)·atan(ω₁ dt/2)`, `ω₁` the lowest axial mode of the same discretisation
//!   (inverse iteration on `K φ = ω² M φ`, `tests/support/modal_axial.rs`) — the
//!   average-acceleration scheme's period elongation is a closed form, so this is exact to the
//!   crossing resolution. The thin-rod closed form `ω₁ = (π/2H)√(E/ρ)` holds only for a
//!   slender rod with lateral inertia neglected and no lateral constraint at the base; it is
//!   checked against the discrete mode as a stated-assumption bound, not used as the oracle;
//! * **Rayleigh damping decays at the closed-form rate**: with `C = α M + β K'` the first
//!   mode's coordinate decays and oscillates as the trapezoidal rule (which the
//!   average-acceleration step is) maps the continuous root `λ = −ζω₁ ± i ω₁√(1 − ζ²)`,
//!   `ζ = α/(2ω₁) + βω₁/2`, to `z = (1 + λ dt/2) / (1 − λ dt/2)` per step: amplitude `|z|ⁿ`,
//!   frequency `arg z / dt`. Checked for `α` alone and `β` alone, so each coefficient is seen
//!   on its own;
//! * **ground contact**: a column with no base fix standing on the plane settles (with
//!   damping) to the static solve of the same mesh with a roller base (bottom nodes held in
//!   `y` only, the frictionless contact's exact counterpart) within 1 %, with every bottom
//!   node held on the plane; dropped from 10 mm it never penetrates the plane at any step's
//!   end, touches down near the free-fall time and settles to the same sag. The closed form
//!   `ρ g H² / (2E)` is the limit the P1 mesh converges to at `O(h)` (the stress varies
//!   linearly under self-weight, a P1 tetrahedron holds one strain), stated for the record,
//!   not used as the oracle;
//! * **external impulses conserve momentum**: a free body (no base, no ground, no load) given
//!   an impulse `J` at one vertex has total momentum `Σ M v = J` — bit for bit with the lumped
//!   mass, to the solve tolerance with the consistent one — and keeps it over the following
//!   steps (nothing acts on it), its centre of mass moving at `J / M_total`.
//!
//! Material: `E = 1.2725e6 Pa` (`G = 4.27e5 Pa`, `ν = 0.49`), `ρ = 1100 kg/m³`; in the finite
//! element module's units `E = 1.2725 MPa`, `ρ = 1.1e-9 t/mm³`.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: periods, means and the Newmark frequency relation are computed in f64 from the recorded displacements, never by the solver under test"
)]

use alice_physics_core::damping_rayleigh::RayleighCoefficients;
use alice_physics_core::dynamic_fem::{
    CorotationalTransientSolver, DynamicsConfig, MassLumping, TransientSolver,
};
use alice_physics_core::linear_elastic_fem::{
    self, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;

#[path = "support/deformable_column.rs"]
mod column;
use column::{base_vertices, column_mesh, lumped_masses, top_vertices};
#[path = "support/modal_axial.rs"]
mod modal;
use alice_physics_core::linear_elastic_fem::RESIDUAL_NORM_FLOOR;
use modal::{
    axial_mode, axial_mode_held, held_dofs, newmark_frequency, newmark_operator_times,
    stiffness_eigenvalue_bound, warped_axial_mode, warped_operator_times,
};

const SHEAR_MODULUS_PA: f64 = 4.27e5;
const POISSON: f64 = 0.49;
const YOUNG_PA: f64 = 2.0 * SHEAR_MODULUS_PA * (1.0 + POISSON);
const DENSITY: f64 = 1100.0;
const G: f64 = 9.81;
/// the column: 0.2 m tall, 0.025 m wide, 1.25 cm cells (16 × 2 × 2 cubes, 384 tetrahedra)
const HALF_H: f64 = 0.1;
const HALF_W: f64 = 0.0125;
const CELL: f32 = 0.0125;

struct Column {
    mm: SdfTetMesh,
    material: ElasticMaterial,
    /// lumped vertex masses, tonnes
    masses_t: Vec<f64>,
    base: Vec<usize>,
    top: Vec<usize>,
}

fn column() -> Column {
    let mesh = column_mesh(CELL, HALF_W, HALF_H);
    let mm = SdfTetMesh {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| [v[0] * 1000.0, v[1] * 1000.0, v[2] * 1000.0])
            .collect(),
        tets: mesh.tets.clone(),
    };
    let material =
        ElasticMaterial::new(Fix128::from_f64(YOUNG_PA * 1e-6), Fix128::from_f64(POISSON))
            .expect("E > 0, ν in range");
    // kg → tonnes
    let masses_t = lumped_masses(&mesh, DENSITY)
        .iter()
        .map(|m| m * 1e-3)
        .collect();
    let base = base_vertices(&mesh);
    let top = top_vertices(&mesh);
    Column {
        mm,
        material,
        masses_t,
        base,
        top,
    }
}

fn gravity_boundary(c: &Column) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for &i in &c.base {
        bc.fix(i as u32);
    }
    for (i, m) in c.masses_t.iter().enumerate() {
        // N = t·mm/s²: m[t] × g[mm/s²]
        bc.add_load(
            i as u32,
            linear_elastic_fem::Axis::Y,
            Fix128::from_f64(-m * G * 1000.0),
        );
    }
    bc
}

/// the steps per first-mode period of every sweep: `T₁/32`, `T₁/64`, `T₁/128`
const STEPS_PER_PERIOD: [usize; 3] = [32, 64, 128];
/// periods of the first mode each run covers
const PERIODS: usize = 8;
/// the conjugate gradient's relative tolerance in the runs below
const CG_TOL: f64 = 1e-9;

fn dynamics(dt: f64) -> DynamicsConfig {
    DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_f64(dt),
        MassLumping::Lumped,
    )
    .expect("density and dt positive")
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(20_000, Fix128::from_f64(CG_TOL)).expect("budget and tolerance")
}

fn to_f64(u: &[Fix128]) -> Vec<f64> {
    u.iter().map(|x| x.to_f64()).collect()
}

/// a transient run: the displacement field after every step, step 0 the start (mm)
struct Trace {
    u: Vec<Vec<f64>>,
}

impl Trace {
    fn sag(&self, c: &Column) -> Vec<f64> {
        self.u.iter().map(|u| top_of(c, u)).collect()
    }
    fn q(&self, c: &Column, phi: &[f64]) -> Vec<f64> {
        self.u
            .iter()
            .map(|u| modal_coordinate_f64(c, phi, u))
            .collect()
    }
    /// the state of largest `M`-norm: the most deformed, hence most warped, state of the run
    fn most_deformed(&self, c: &Column) -> &[f64] {
        self.u
            .iter()
            .max_by(|a, b| m_norm(c, a).partial_cmp(&m_norm(c, b)).expect("finite"))
            .expect("a step")
    }
}

/// `e_topᵀ u`: the mean sag of the top face, mm, positive downward
fn top_of(c: &Column, u: &[f64]) -> f64 {
    c.top.iter().map(|&i| -u[3 * i + 1]).sum::<f64>() / c.top.len() as f64
}

fn modal_coordinate_f64(c: &Column, phi: &[f64], u: &[f64]) -> f64 {
    (0..c.mm.vertices.len())
        .map(|i| c.masses_t[i] * (0..3).map(|k| phi[3 * i + k] * u[3 * i + k]).sum::<f64>())
        .sum()
}

/// `‖u‖_M` with the lumped masses
fn m_norm(c: &Column, u: &[f64]) -> f64 {
    (0..c.mm.vertices.len())
        .map(|i| c.masses_t[i] * (0..3).map(|k| u[3 * i + k] * u[3 * i + k]).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// `‖f‖_{M⁻¹}` of a nodal vector (loads in N give `√(N²/t)`)
fn m_inv_norm(c: &Column, f: &[f64]) -> f64 {
    (0..c.mm.vertices.len())
        .map(|i| (0..3).map(|k| f[3 * i + k] * f[3 * i + k]).sum::<f64>() / c.masses_t[i])
        .sum::<f64>()
        .sqrt()
}

/// the gravity loads, N per degree of freedom
fn gravity_loads(c: &Column) -> Vec<f64> {
    let mut f = vec![0.0; 3 * c.mm.vertices.len()];
    for (i, m) in c.masses_t.iter().enumerate() {
        f[3 * i + 1] = -m * G * 1000.0;
    }
    f
}

/// the functional `e_top` as a nodal vector: `−1/n_top` on the `y` of each top vertex
fn top_functional(c: &Column) -> Vec<f64> {
    let mut e = vec![0.0; 3 * c.mm.vertices.len()];
    for &i in &c.top {
        e[3 * i + 1] = -1.0 / c.top.len() as f64;
    }
    e
}

/// the static solution (mm per degree of freedom) under the nodal loads `f` (N) with the
/// `held` degrees of freedom at zero
fn static_solution(c: &Column, held: &[bool], f: &[f64]) -> Vec<f64> {
    let mut bc = BoundaryConditions::new();
    for i in 0..c.mm.vertices.len() {
        for (axis, k) in [
            (linear_elastic_fem::Axis::X, 0),
            (linear_elastic_fem::Axis::Y, 1),
            (linear_elastic_fem::Axis::Z, 2),
        ] {
            if held[3 * i + k] {
                bc.prescribe(i as u32, axis, Fix128::ZERO);
            } else if f[3 * i + k] != 0.0 {
                bc.add_load(i as u32, axis, Fix128::from_f64(f[3 * i + k]));
            }
        }
    }
    let sol = linear_elastic_fem::solve(&c.mm, &c.material, &bc, &SolverConfig::default())
        .expect("the static solve");
    sol.displacements
        .iter()
        .flat_map(|d| d.iter().map(|x| x.to_f64()))
        .collect()
}

/// the held degrees of freedom of the fixed base
fn fixed_base(c: &Column) -> Vec<bool> {
    held_dofs(&c.base, &[], c.mm.vertices.len())
}

/// the held degrees of freedom of the roller base (bottom vertices in `y` only, plus the
/// centre bottom vertex in `x` and `z` and one more in `z` against the lateral rigid modes
/// the roller cannot see)
fn roller_base(c: &Column) -> Vec<bool> {
    let n = c.mm.vertices.len();
    let mut held = held_dofs(&[], &c.base, n);
    let centre = bottom_centre(c);
    held[3 * centre] = true;
    held[3 * centre + 2] = true;
    let other = *c
        .base
        .iter()
        .find(|&&i| i != centre && (c.mm.vertices[i][2] - c.mm.vertices[centre][2]).abs() < 1e-6)
        .expect("a second bottom vertex on the centre's z line");
    held[3 * other + 2] = true;
    held
}

fn bottom_centre(c: &Column) -> usize {
    *c.base
        .iter()
        .min_by(|&&a, &&b| {
            let r = |i: usize| {
                let v = c.mm.vertices[i];
                v[0] * v[0] + v[2] * v[2]
            };
            r(a).partial_cmp(&r(b)).expect("finite")
        })
        .expect("a base")
}

/// What one completed step's solve can have moved the state by, in the `M`-norm: the solve
/// `A u' = b` stops at `‖r‖₂ ≤ max(tol ‖b‖₂, floor)`, `A = (4/dt² + 2α/dt) M + (1 + 2β/dt) K'`
/// (`K'` warped at `u'` when `warped`, the unwarped stiffness otherwise), and
/// `Â = M^{-1/2} A M^{-1/2} ≥ (4/dt²) I`, so `‖δu'‖_M ≤ (dt²/4) ‖r‖₂ / √m_min`; `‖b‖₂` is
/// `‖A u'‖₂` evaluated in `f64` from the measured `u'`, with `2⁻¹⁹ λ_K ‖u'‖₂` added for the
/// difference between the solver's polar factors (to `2⁻⁶²`) and `f64`'s when warped.
fn solve_error_m(c: &Column, dt: f64, damping: (f64, f64), warped: bool, u: &[f64]) -> f64 {
    let (lambda, mu) = lame_mpa();
    let zero = vec![0.0; u.len()];
    let warp_state: &[f64] = if warped { u } else { &zero };
    let ku = warped_operator_times(&c.mm, lambda, mu, &c.masses_t, false, 0.0, warp_state, u);
    let (alpha, beta) = damping;
    let a_u: f64 = (0..u.len())
        .map(|d| {
            let m_u = c.masses_t[d / 3] * u[d];
            let x = (4.0 / (dt * dt) + 2.0 * alpha / dt) * m_u + (1.0 + 2.0 * beta / dt) * ku[d];
            x * x
        })
        .sum::<f64>()
        .sqrt();
    let polar = if warped {
        2f64.powi(-19)
            * stiffness_eigenvalue_bound(&c.mm, lambda, mu)
            * u.iter().map(|x| x * x).sum::<f64>().sqrt()
    } else {
        0.0
    };
    let m_min = c.masses_t.iter().copied().fold(f64::INFINITY, f64::min);
    (dt * dt / 4.0) * residual_bound(CG_TOL, a_u + polar) / m_min.sqrt()
}

/// `(|z|, arg z)` of the trapezoidal map of a mode `ω` at damping ratio `ζ` (any `ζ ≥ 0`:
/// above 1 the two roots are real and the larger modulus is returned with phase 0)
fn trapezoidal_root(omega: f64, zeta: f64, dt: f64) -> (f64, f64) {
    let map = |re: f64, im: f64| {
        // z = (1 + λ dt/2) / (1 − λ dt/2), λ = re + i im
        let (nr, ni) = (1.0 + re * dt / 2.0, im * dt / 2.0);
        let (dr, di) = (1.0 - re * dt / 2.0, -im * dt / 2.0);
        let d2 = dr * dr + di * di;
        let zr = (nr * dr + ni * di) / d2;
        let zi = (ni * dr - nr * di) / d2;
        ((zr * zr + zi * zi).sqrt(), zi.atan2(zr))
    };
    if zeta < 1.0 {
        map(-zeta * omega, omega * (1.0 - zeta * zeta).sqrt())
    } else {
        let s = omega * (zeta * zeta - 1.0).sqrt();
        let (a, _) = map(-zeta * omega + s, 0.0);
        let (b, _) = map(-zeta * omega - s, 0.0);
        (a.max(b), 0.0)
    }
}

/// the largest value of `g(ω)` over the frequencies a mode of the mesh can have,
/// `[ω_lo, ω_hi]`, on a geometric grid of 4,000 points with the endpoints (the functions
/// bounded here are smooth and unimodal or monotone in `ω`, and the grid's neighbours are
/// 0.2 % apart, which the bounds below absorb by the factor `grid_margin`)
fn max_over_modes(omega_lo: f64, omega_hi: f64, g: impl Fn(f64) -> f64) -> f64 {
    let n = 4000;
    (0..=n)
        .map(|k| omega_lo * (omega_hi / omega_lo).powf(k as f64 / n as f64))
        .map(&g)
        .fold(0.0, f64::max)
}
const GRID_MARGIN: f64 = 1.01;

/// the highest frequency any mode of the lumped-mass mesh can have, `√(λ_K / m_min)` with
/// `λ_K` the Gershgorin bound of the stiffness
fn omega_max(c: &Column) -> f64 {
    let (lambda, mu) = lame_mpa();
    let m_min = c.masses_t.iter().copied().fold(f64::INFINITY, f64::min);
    (stiffness_eigenvalue_bound(&c.mm, lambda, mu) / m_min).sqrt()
}

/// the period read from the level crossings of a single-mode signal, and what it can be
/// off by: linear interpolation between samples puts each crossing within `dt³ ω̄² / 8`
/// (the interpolation error `(dt²/8) max|q''|` with `|q''| ≤ A ω̄² sin(ω̄ dt)` across the
/// crossing's interval, over the slope `A ω̄`), and the period is twice the span of
/// `n_half` half periods: `δT ≤ 4 · dt³ ω̄² / (8 n_half)`
fn crossing_period(signal: &[f64], level: f64, dt: f64, omega_bar: f64) -> (f64, f64) {
    let mut crossings = Vec::new();
    for k in 1..signal.len() {
        let (a, b) = (signal[k - 1] - level, signal[k] - level);
        if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0) {
            let frac = a / (a - b);
            crossings.push((k as f64 - 1.0 + frac) * dt);
        }
    }
    assert!(crossings.len() >= 3, "{} crossings", crossings.len());
    let n_half = (crossings.len() - 1) as f64;
    let period = 2.0 * (crossings[crossings.len() - 1] - crossings[0]) / n_half;
    (period, dt.powi(3) * omega_bar * omega_bar / (2.0 * n_half))
}

/// the trajectory from rest under gravity: `linear` the Newmark solver, otherwise the
/// co-rotational one with the given Rayleigh damping
fn gravity_trace(
    c: &Column,
    dt: f64,
    steps: usize,
    linear: bool,
    damping: Option<RayleighCoefficients>,
) -> Trace {
    let dynamics = dynamics(dt);
    let config = solver_config();
    let mut u = Vec::with_capacity(steps + 1);
    if linear {
        let mut solver =
            TransientSolver::new(&c.mm, &c.material, &gravity_boundary(c), &dynamics, &config)
                .expect("the column solves");
        u.push(to_f64(solver.displacements()));
        for _ in 0..steps {
            solver.step().expect("the step converges");
            u.push(to_f64(solver.displacements()));
        }
    } else {
        let mut solver = CorotationalTransientSolver::new(
            &c.mm,
            &c.material,
            &gravity_boundary(c),
            &dynamics,
            &config,
        )
        .expect("the column solves");
        solver.set_damping(damping);
        u.push(to_f64(solver.displacements()));
        for _ in 0..steps {
            solver.step().expect("the step converges");
            u.push(to_f64(solver.displacements()));
        }
    }
    Trace { u }
}

/// The undamped oscillation of the column under its own weight at three step sizes, for the
/// linear Newmark solver (`linear`) or the co-rotational one. Checked at every `dt`, with
/// the tolerances computed in place:
///
/// * the period of the first modal coordinate `q₁ = φ₁ᵀ M u` against `2π/ω̄`,
///   `ω̄ = (2/dt)·atan(ω₁ dt/2)` (the scheme and the modal decomposition commute for the
///   linear solver, so `q₁` is a single-mode signal): the measurement's crossing error
///   (`crossing_period`), plus the phase the solves can have drifted by, each step's
///   `‖δu'‖_M (1 + 2/(ω̄ dt)) / A₁` (a state error `(δq, δq̇)`, `δq̇ = (2/dt) δq`, moves the
///   phase by at most `|δq| + |δq̇|/ω̄` over the amplitude `A₁ = |q₁*|`, the static modal
///   coordinate, which is the amplitude of a start from rest), summed and divided by
///   `2π × periods`; for the co-rotational solver the stiffness is warped by the
///   deformation, so its first frequency lies between the unwarped `ω₁` and `ω₁'` of the
///   stiffness warped at the most deformed state of the run (`warped_axial_mode`), and the
///   period is checked against that range;
/// * the mean of `q₁` over the run against `q₁*`: a start from rest gives
///   `q₁(n) = q₁*(1 − cos(ω̄ n dt))` and the mean of `cos` over `N + 1` samples is at most
///   `1 / ((N + 1) sin(ω̄ dt / 2)) = √(1 + x²) / (x (N + 1))`, `x = ω₁ dt / 2` (Dirichlet),
///   plus the solves' drift `Σ ‖δu'‖_M / (N + 1)` and, co-rotational, the static's own
///   change under the warped stiffness, `|ω₁'²/ω₁² − 1| |q₁*|`;
/// * the mean of the top sag over the run against the static sag: every mode `k` starts the
///   same way with amplitude `aₖ = (e_topᵀ φₖ)(φₖᵀ f) / ωₖ²`, so the bias is at most
///   `Σₖ |aₖ| (1/xₖ + 1) / (N + 1)`, and `Σₖ |aₖ| / ωₖ ≤ ‖u_unit‖_M ‖u_static‖_M`,
///   `Σₖ |aₖ| ≤ √((e_topᵀ u_unit) (fᵀ u_static))` by Cauchy–Schwarz over the
///   `M`-orthonormal modes (`u_unit = K⁻¹ e_top`, `u_static = K⁻¹ f`, both static solves),
///   plus the same drift and warping terms.
fn check_undamped(c: &Column, linear: bool) {
    let held = fixed_base(c);
    let (omega, phi, omega_err) = axial_mode_held(&c.mm, &c.material, &c.masses_t, &held, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let f = gravity_loads(c);
    let u_static = static_solution(c, &held, &f);
    let u_unit = static_solution(c, &held, &top_functional(c));
    let static_sag = top_of(c, &u_static);
    let q_static = modal_coordinate_f64(c, &phi, &u_static);
    let sum_a =
        (top_of(c, &u_unit) * f.iter().zip(&u_static).map(|(a, b)| a * b).sum::<f64>()).sqrt();
    let sum_a_over_omega = m_norm(c, &u_unit) * m_norm(c, &u_static);
    let (lambda, mu) = lame_mpa();
    let label = if linear { "linear" } else { "co-rotational" };
    for &per_period in &STEPS_PER_PERIOD {
        let dt = period / per_period as f64;
        let steps = per_period * PERIODS;
        let trace = gravity_trace(c, dt, steps, linear, None);
        let q = trace.q(c, &phi);
        let sag = trace.sag(c);
        let drift: Vec<f64> = trace.u[1..]
            .iter()
            .map(|u| solve_error_m(c, dt, (0.0, 0.0), !linear, u))
            .collect();
        let drift_sum: f64 = drift.iter().sum();
        let n_samples = q.len() as f64;
        // the frequency range: the unwarped mode, and the mode of the stiffness warped at the
        // most deformed state
        let omega_warped = if linear {
            omega
        } else {
            warped_axial_mode(
                &c.mm,
                lambda,
                mu,
                &c.masses_t,
                &held,
                trace.most_deformed(c),
                12,
            )
        };
        let (omega_lo, omega_hi) = (omega.min(omega_warped), omega.max(omega_warped));
        let warping = (omega_hi * omega_hi / (omega_lo * omega_lo) - 1.0).abs();
        let omega_bar = newmark_frequency(omega, dt);
        let amplitude = q_static.abs();
        let q_mean = q.iter().sum::<f64>() / n_samples;
        let (measured, crossing) = crossing_period(&q, q_mean, dt, omega_bar);
        let expected_lo = 2.0 * std::f64::consts::PI / newmark_frequency(omega_hi, dt);
        let expected_hi = 2.0 * std::f64::consts::PI / newmark_frequency(omega_lo, dt);
        let phase = drift_sum * (1.0 + 2.0 / (omega_bar * dt))
            / amplitude
            / (2.0 * std::f64::consts::PI * PERIODS as f64);
        // the reference frequency's own error moves the expected period by the same share
        let period_tol = crossing / expected_lo + phase + omega_err;
        let x = omega * dt / 2.0;
        let q_tol = amplitude * (1.0 + x * x).sqrt() / (x * n_samples)
            + drift_sum / n_samples
            + warping * amplitude;
        let sag_mean = sag.iter().sum::<f64>() / n_samples;
        let sag_tol = (sum_a_over_omega * 2.0 / dt + sum_a) / n_samples
            + drift_sum
                / c.masses_t
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
                    .sqrt()
            + warping * static_sag;
        eprintln!(
            "{label} dt = T/{per_period}: period {measured:.6e} s vs [{expected_lo:.6e}, {expected_hi:.6e}] (tolerance {period_tol:.2e} relative, of which ω₁'s own {omega_err:.1e}; continuous {period:.6e}); q₁ mean {q_mean:.5e} vs static {q_static:.5e} (tolerance {q_tol:.2e}); top sag mean {sag_mean:.5e} mm vs static {static_sag:.5e} mm (tolerance {sag_tol:.2e} mm); warping {warping:.2e}"
        );
        assert!(
            measured >= expected_lo * (1.0 - period_tol) && measured <= expected_hi * (1.0 + period_tol),
            "{label} dt = T/{per_period}: period {measured:.6e} outside [{expected_lo:.6e}, {expected_hi:.6e}] ± {period_tol:.2e}"
        );
        assert!(
            (q_mean - q_static).abs() <= q_tol,
            "{label} dt = T/{per_period}: q₁ mean {q_mean:.5e} vs static {q_static:.5e} (tolerance {q_tol:.2e})"
        );
        assert!(
            (sag_mean - static_sag).abs() <= sag_tol,
            "{label} dt = T/{per_period}: top sag mean {sag_mean:.5e} vs static {static_sag:.5e} (tolerance {sag_tol:.2e})"
        );
        // the sweep itself: the elongation the Newmark relation predicts at this dt must be
        // resolved, that is larger than the tolerance it is checked with, at the coarse steps
        if per_period <= 64 {
            assert!(
                (expected_lo / period - 1.0).abs() > period_tol,
                "{label} dt = T/{per_period}: the period elongation {:.2e} is not resolved by the tolerance {period_tol:.2e}",
                expected_lo / period - 1.0
            );
        }
    }
}

#[test]
fn the_linear_newmark_oscillation_has_the_modal_period_and_the_static_mean_at_three_step_sizes() {
    check_undamped(&column(), true);
}

#[test]
fn the_corotational_oscillation_has_the_modal_period_and_the_static_mean_at_three_step_sizes() {
    check_undamped(&column(), false);
}

/// The trapezoidal map of one damped mode on the state `(q, q̇)`:
/// `T = (I − (dt/2) L)⁻¹ (I + (dt/2) L)`, `L = [[0, 1], [−ω², −2ζω]]`
fn trapezoidal_map(omega: f64, zeta: f64, dt: f64) -> [[f64; 2]; 2] {
    let h = dt / 2.0;
    let l = [[0.0, 1.0], [-omega * omega, -2.0 * zeta * omega]];
    let a = [
        [1.0 - h * l[0][0], -h * l[0][1]],
        [-h * l[1][0], 1.0 - h * l[1][1]],
    ];
    let b = [
        [1.0 + h * l[0][0], h * l[0][1]],
        [h * l[1][0], 1.0 + h * l[1][1]],
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let inv = [
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det],
    ];
    let mut t = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            t[i][j] = inv[i][0] * b[0][j] + inv[i][1] * b[1][j];
        }
    }
    t
}

/// The three-term recurrence a single damped mode obeys, `q_{n+1} = a q_n + b q_{n−1} + c`
/// (`a = tr T`, `b = −det T`, `c` from the constant load), fitted by least squares to a
/// signal, with the fit's sensitivity: `(a, b, c)` and `√((XᵀX)⁻¹ᵢᵢ)`, the 2-norm of the
/// `i`-th row of the pseudo-inverse, so that a residual vector `δ` moves coefficient `i` by
/// at most `√((XᵀX)⁻¹ᵢᵢ) ‖δ‖₂`
fn recurrence_fit(q: &[f64]) -> ([f64; 3], [f64; 3]) {
    let mut xtx = [[0.0f64; 3]; 3];
    let mut xty = [0.0f64; 3];
    for n in 1..q.len() - 1 {
        let row = [q[n], q[n - 1], 1.0];
        for i in 0..3 {
            for j in 0..3 {
                xtx[i][j] += row[i] * row[j];
            }
            xty[i] += row[i] * q[n + 1];
        }
    }
    // inverse of the symmetric 3 × 3 by cofactors
    let cof = |i: usize, j: usize| {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
        xtx[i1][j1] * xtx[i2][j2] - xtx[i1][j2] * xtx[i2][j1]
    };
    let det = xtx[0][0] * cof(0, 0) + xtx[0][1] * cof(0, 1) + xtx[0][2] * cof(0, 2);
    let inv: [[f64; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| cof(j, i) / det));
    let mut theta = [0.0f64; 3];
    for i in 0..3 {
        theta[i] = (0..3).map(|j| inv[i][j] * xty[j]).sum();
    }
    (
        theta,
        [inv[0][0].sqrt(), inv[1][1].sqrt(), inv[2][2].sqrt()],
    )
}

/// The damped first mode at three step sizes, for one Rayleigh coefficient at a time
/// (`ζ = 0.05` on the first mode): `q₁` obeys the three-term recurrence of the trapezoidal
/// map (the Rayleigh damping is proportional, so the modes stay decoupled), whose fitted
/// coefficients give the per-step decay `|z| = √(−b)` and phase `arg z = acos(a / 2|z|)`;
/// the log-decrement per period `D = −ln|z| · 2π / arg z` and the period `2π dt / arg z`
/// are checked against the closed form at `ω₁`, and for the co-rotational solver against
/// the range the warping spans (`ω₁` to `ω₁'` at the most deformed state). The tolerance is
/// propagated from the solves: a step's error `(δq, δq̇)` enters the recurrence as a residual
/// `|δₙ| ≤ εₙ + |T_{q q̇}| (2/dt) εₙ₋₁ + |T_{qq} − a| εₙ₋₁` (`εₙ` from `solve_error_m`), the
/// least squares moves `(a, b)` by at most `√((XᵀX)⁻¹ᵢᵢ) ‖δ‖₂`, and
/// `δ|z| ≤ δb / 2|z|`, `δ(arg z) ≤ (δa / 2|z| + |a| δ|z| / 2|z|²) / sin(arg z)`,
/// `δD ≤ (2π / arg z) δ|z| / |z| + (2π |ln|z|| / arg z²) δ(arg z)`.
fn check_damped(c: &Column, alpha_or_beta: bool) {
    let held = fixed_base(c);
    let (omega, phi, omega_err) = axial_mode_held(&c.mm, &c.material, &c.masses_t, &held, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let (alpha, beta) = if alpha_or_beta {
        (2.0 * 0.05 * omega, 0.0)
    } else {
        (0.0, 2.0 * 0.05 / omega)
    };
    let damping = RayleighCoefficients {
        alpha: Fix128::from_f64(alpha),
        beta: Fix128::from_f64(beta),
    };
    let zeta_of = |w: f64| alpha / (2.0 * w) + beta * w / 2.0;
    let (lambda, mu) = lame_mpa();
    let label = if alpha_or_beta { "α only" } else { "β only" };
    for &per_period in &STEPS_PER_PERIOD {
        let dt = period / per_period as f64;
        let steps = per_period * PERIODS;
        let trace = gravity_trace(c, dt, steps, false, Some(damping));
        let q = trace.q(c, &phi);
        let eps: Vec<f64> = trace
            .u
            .iter()
            .map(|u| solve_error_m(c, dt, (alpha, beta), true, u))
            .collect();
        let (theta, sens) = recurrence_fit(&q);
        let (a, b) = (theta[0], theta[1]);
        assert!(
            b < 0.0,
            "{label} dt = T/{per_period}: the fit has no complex root (b = {b:.3e})"
        );
        let abs_z = (-b).sqrt();
        let cos_arg = a / (2.0 * abs_z);
        assert!(
            cos_arg.abs() < 1.0,
            "{label} dt = T/{per_period}: cos(arg z) = {cos_arg}"
        );
        let arg_z = cos_arg.acos();
        let measured_decrement = -abs_z.ln() * 2.0 * std::f64::consts::PI / arg_z;
        let measured_period = 2.0 * std::f64::consts::PI * dt / arg_z;
        // the closed form over the frequency range the warping spans
        let omega_warped = warped_axial_mode(
            &c.mm,
            lambda,
            mu,
            &c.masses_t,
            &held,
            trace.most_deformed(c),
            12,
        );
        let mut expected = [omega, omega_warped].map(|w| {
            let (z, arg) = trapezoidal_root(w, zeta_of(w), dt);
            (
                -z.ln() * 2.0 * std::f64::consts::PI / arg,
                2.0 * std::f64::consts::PI * dt / arg,
            )
        });
        expected.sort_by(|p, q| p.0.partial_cmp(&q.0).expect("finite"));
        let (decrement_lo, decrement_hi) = (expected[0].0, expected[1].0);
        let (period_lo, period_hi) = (
            expected[0].1.min(expected[1].1),
            expected[0].1.max(expected[1].1),
        );
        // the residual the solves can have put into the recurrence
        let t = trapezoidal_map(omega, zeta_of(omega), dt);
        let a_exact = t[0][0] + t[1][1];
        let residual_norm: f64 = (1..q.len() - 1)
            .map(|n| {
                let d = eps[n + 1]
                    + t[0][1].abs() * (2.0 / dt) * eps[n]
                    + (t[0][0] - a_exact).abs() * eps[n];
                d * d
            })
            .sum::<f64>()
            .sqrt();
        let (da, db) = (sens[0] * residual_norm, sens[1] * residual_norm);
        let d_abs_z = db / (2.0 * abs_z);
        let d_arg = (da / (2.0 * abs_z) + a.abs() * d_abs_z / (2.0 * abs_z * abs_z)) / arg_z.sin();
        let d_decrement = 2.0 * std::f64::consts::PI / arg_z * d_abs_z / abs_z
            + 2.0 * std::f64::consts::PI * abs_z.ln().abs() / (arg_z * arg_z) * d_arg;
        let d_period = measured_period * d_arg / arg_z;
        // the reference frequency's own error: the decrement is proportional to ω (β) or 1/ω
        // (α), the period to 1/ω̄, so both move by its relative share
        let d_decrement = d_decrement + omega_err * decrement_hi;
        let d_period = d_period + omega_err * period_hi;
        eprintln!(
            "{label} dt = T/{per_period}: log-decrement per period {measured_decrement:.5e} vs [{decrement_lo:.5e}, {decrement_hi:.5e}] ± {d_decrement:.2e}; period {measured_period:.6e} vs [{period_lo:.6e}, {period_hi:.6e}] ± {d_period:.2e}"
        );
        assert!(
            measured_decrement >= decrement_lo - d_decrement && measured_decrement <= decrement_hi + d_decrement,
            "{label} dt = T/{per_period}: decrement {measured_decrement:.5e} outside [{decrement_lo:.5e}, {decrement_hi:.5e}] ± {d_decrement:.2e}"
        );
        assert!(
            measured_period >= period_lo - d_period && measured_period <= period_hi + d_period,
            "{label} dt = T/{per_period}: period {measured_period:.6e} outside [{period_lo:.6e}, {period_hi:.6e}] ± {d_period:.2e}"
        );
    }
}

#[test]
fn stiffness_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate_at_three_step_sizes(
) {
    check_damped(&column(), false);
}

#[test]
fn mass_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate_at_three_step_sizes() {
    check_damped(&column(), true);
}

/// The discrete axial mode of the column against the thin-rod closed form, with the
/// assumptions stated: the rod formula neglects lateral inertia and the base's lateral
/// constraint, both of which raise the stiffness-to-mass ratio of a short thick column a
/// little, so the discrete mode must lie above the rod's frequency and within the base
/// layer's share (`(f − f²/2)/(1/2)`, `f = w/H`, the same bound the static sag uses).
#[test]
fn the_discrete_axial_mode_lies_above_the_thin_rod_frequency_by_at_most_the_base_layer_share() {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let h = 2.0 * HALF_H;
    let rod = std::f64::consts::PI / (2.0 * h) * (YOUNG_PA / DENSITY).sqrt();
    let f = 2.0 * HALF_W / h;
    let share = (f - f * f / 2.0) / 0.5;
    eprintln!(
        "discrete axial ω₁ {omega:.3} rad/s, thin rod {rod:.3} rad/s ({:+.4})",
        omega / rod - 1.0
    );
    assert!(
        omega > rod,
        "discrete {omega:.3} must exceed the rod's {rod:.3}"
    );
    // a stiffer base layer raises ω² by at most its share of the compliance
    assert!(
        omega / rod - 1.0 < share / 2.0,
        "discrete {omega:.3} vs rod {rod:.3}: above the bound {}",
        share / 2.0
    );
}

/// A body rotated as a whole and released with no load stays where it is: the co-rotational
/// element sees no strain in a rotation, so the elastic force is zero and nothing moves
/// (the prescribed rotation is applied as the initial displacement of every vertex, the base
/// held at its rotated place).
#[test]
fn a_rigidly_rotated_body_feels_no_elastic_force() {
    let c = column();
    // rotate the column by 30° about z around its base centre: the displacement field is
    // u = R x − x, prescribed on the base, the initial state of the rest
    let (s, co) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
    let y0 = f64::from(
        c.mm.vertices
            .iter()
            .map(|v| v[1])
            .fold(f32::INFINITY, f32::min),
    );
    let rotated = |v: [f32; 3]| {
        let (x, y) = (f64::from(v[0]), f64::from(v[1]) - y0);
        [
            co * x - s * y - f64::from(v[0]),
            s * x + co * y + y0 - f64::from(v[1]),
            0.0,
        ]
    };
    let mut bc = BoundaryConditions::new();
    for &i in &c.base {
        let d = rotated(c.mm.vertices[i]);
        bc.prescribe_all(
            i as u32,
            [
                Fix128::from_f64(d[0]),
                Fix128::from_f64(d[1]),
                Fix128::from_f64(d[2]),
            ],
        );
    }
    let dt = 1.0 / 240.0;
    let mut solver =
        CorotationalTransientSolver::new(&c.mm, &c.material, &bc, &dynamics(dt), &solver_config())
            .expect("the rotated column solves");
    let initial: Vec<[f64; 3]> = c.mm.vertices.iter().map(|v| rotated(*v)).collect();
    solver
        .set_displacements(
            &initial
                .iter()
                .flatten()
                .map(|x| Fix128::from_f64(*x))
                .collect::<Vec<_>>(),
        )
        .expect("the rotated state has a polar factor");
    // what the solves can have moved any vertex by: each step's `‖δu'‖_M / √m_min`
    // (`solve_error_m`), summed (an undamped body keeps every perturbation), plus the polar
    // factors' own precision, `2⁻⁶²` of the rotation over the body's extent (an eigenstrain
    // `ε` moves a body of size `L` by at most `ε L` in linear elasticity)
    let m_min = c.masses_t.iter().copied().fold(f64::INFINITY, f64::min);
    let extent = f64::from(
        c.mm.vertices
            .iter()
            .map(|v| v[1])
            .fold(f32::NEG_INFINITY, f32::max),
    ) - y0;
    let mut tol = 2f64.powi(-62) * extent;
    for _ in 0..60 {
        solver.step().expect("the step converges");
        tol +=
            solve_error_m(&c, dt, (0.0, 0.0), true, &to_f64(solver.displacements())) / m_min.sqrt();
    }
    let u = solver.displacements();
    let worst = (0..c.mm.vertices.len())
        .map(|i| {
            (0..3)
                .map(|k| (u[3 * i + k].to_f64() - initial[i][k]).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    eprintln!("rotated and released: the largest move after 60 steps is {worst:.3e} mm (tolerance {tol:.3e} mm)");
    assert!(
        worst <= tol,
        "a vertex moved {worst:.3e} mm under a rigid rotation (tolerance {tol:.3e})"
    );
}

/// the column with no base fix on the ground plane at `ground_mm` below its bottom (0 =
/// standing on it), gravity, `ζ = 0.2` stiffness damping on the first mode so it settles;
/// returns the solver, the plane's height (mm) and `β`
fn standing_solver(
    c: &Column,
    omega: f64,
    ground_below_mm: f64,
    dt: f64,
) -> (CorotationalTransientSolver, f64, f64) {
    let mut bc = BoundaryConditions::new();
    for (i, m) in c.masses_t.iter().enumerate() {
        bc.add_load(
            i as u32,
            linear_elastic_fem::Axis::Y,
            Fix128::from_f64(-m * G * 1000.0),
        );
    }
    let mut solver =
        CorotationalTransientSolver::new(&c.mm, &c.material, &bc, &dynamics(dt), &solver_config())
            .expect("the column solves");
    let beta = 2.0 * 0.2 / omega;
    solver.set_damping(Some(RayleighCoefficients {
        alpha: Fix128::ZERO,
        beta: Fix128::from_f64(beta),
    }));
    let bottom = f64::from(
        c.mm.vertices
            .iter()
            .map(|v| v[1])
            .fold(f32::INFINITY, f32::min),
    );
    let ground = bottom - ground_below_mm;
    solver.set_ground(Some(Fix128::from_f64(ground)));
    (solver, ground, beta)
}

fn lowest_y_mm(c: &Column, u: &[Fix128]) -> f64 {
    (0..c.mm.vertices.len())
        .map(|i| f64::from(c.mm.vertices[i][1]) + u[3 * i + 1].to_f64())
        .fold(f64::INFINITY, f64::min)
}

/// The roller problem's data: the static solution under gravity (mm), its top sag, its
/// first axial frequency, and the two norms the settling bounds use
struct Roller {
    held: Vec<bool>,
    u_static: Vec<f64>,
    sag: f64,
    omega: f64,
    e_top_norm: f64,
    f_norm: f64,
}

fn roller(c: &Column) -> Roller {
    let held = roller_base(c);
    let f = gravity_loads(c);
    let u_static = static_solution(c, &held, &f);
    let (omega, _, _) = axial_mode_held(&c.mm, &c.material, &c.masses_t, &held, 12);
    Roller {
        sag: top_of(c, &u_static),
        e_top_norm: m_inv_norm(c, &top_functional(c)),
        f_norm: m_inv_norm(c, &f),
        held,
        u_static,
        omega,
    }
}

/// The column standing on the plane with no base fix, at three step sizes: every bottom
/// vertex is held from the first step on and no step ends below the plane (exact checks),
/// and after `n` steps the top sag is the roller static sag (the same mesh, the
/// frictionless contact's exact counterpart; the closed form `ρ g H² / (2E)` is the O(h)
/// limit, printed) within a computed bound: with the whole bottom held the system is the
/// linear roller problem under a constant load from rest, so every mode `k` oscillates about
/// the static with amplitude `|aₖ| = |e_topᵀ φₖ| |φₖᵀ f| / ωₖ²` decaying as `|zₖ|ⁿ`
/// (`trapezoidal_root` at `ζₖ = β ωₖ / 2`), and
/// `Σₖ |aₖ| |zₖ|ⁿ ≤ max_ω (|z(ω)|ⁿ / ω²) ‖e_top‖_{M⁻¹} ‖f‖_{M⁻¹}` by Cauchy–Schwarz over the
/// `M`-orthonormal modes (the load and the mesh are mirror symmetric, so only the symmetric
/// modes are excited, whose lowest is the roller axial mode `ω₁ʳ`; the highest is below
/// `omega_max`); plus the solves' drift and the warping of the static,
/// `|ω₁ʳ′² / ω₁ʳ² − 1|` of the sag with `ω₁ʳ′` at the final state.
#[test]
fn a_column_standing_on_the_ground_settles_to_the_roller_static_sag_with_its_whole_bottom_in_contact(
) {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let r = roller(&c);
    let (lambda, mu) = lame_mpa();
    let w_max = omega_max(&c);
    let closed = DENSITY * G * (2.0 * HALF_H).powi(2) / (2.0 * YOUNG_PA) * 1000.0;
    for &per_period in &STEPS_PER_PERIOD {
        let dt = period / per_period as f64;
        let steps = per_period * PERIODS;
        let (mut solver, ground, beta) = standing_solver(&c, omega, 0.0, dt);
        let mut drift = 0.0;
        for step in 1..=steps {
            solver.step().expect("the step converges");
            assert!(
                lowest_y_mm(&c, solver.displacements()) >= ground,
                "dt = T/{per_period} step {step}: penetration"
            );
            assert!(
                c.base.iter().all(|&i| solver.on_ground()[i]),
                "dt = T/{per_period} step {step}: a bottom vertex is not held"
            );
            drift += solve_error_m(&c, dt, (0.0, beta), true, &to_f64(solver.displacements()));
        }
        let u = to_f64(solver.displacements());
        let sag = top_of(&c, &u);
        let decay = max_over_modes(r.omega, w_max, |w| {
            trapezoidal_root(w, beta * w / 2.0, dt).0.powi(steps as i32) / (w * w)
        }) * GRID_MARGIN;
        let omega_warped = warped_axial_mode(&c.mm, lambda, mu, &c.masses_t, &r.held, &u, 12);
        let warping = (omega_warped * omega_warped / (r.omega * r.omega) - 1.0).abs();
        let tol = decay * r.e_top_norm * r.f_norm
            + drift
                / c.masses_t
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
                    .sqrt()
            + warping * r.sag;
        eprintln!(
            "standing dt = T/{per_period}: sag {sag:.5e} mm vs roller static {:.5e} mm ({:+.2e}, tolerance {tol:.2e} mm: decay {decay:.2e}, warping {warping:.2e}); closed form {closed:.5e} mm ({:+.4} of roller)",
            r.sag,
            sag - r.sag,
            closed / r.sag - 1.0
        );
        assert!(
            (sag - r.sag).abs() <= tol,
            "dt = T/{per_period}: sag {sag:.5e} vs roller {:.5e} (tolerance {tol:.2e})",
            r.sag
        );
    }
}

/// Dropped from 10 mm at three step sizes: no step ends below the plane and touchdown is
/// within one step of the free-fall time `√(2h/g)` (exact: the scheme integrates a constant
/// acceleration exactly, so only the step's discreteness remains), and once the whole bottom
/// is held the column settles to the roller static sag (measured against the plane) within a
/// computed bound: from the moment `n₀` the bottom is held, the state is the static plus the
/// free response to `(δu₀, v₀)` (the deviation and velocity at that moment, measured), each
/// mode decaying as `|zₖ|ⁿ`; the sag is read as the mean over the last `W` samples (two
/// roller periods), where mode `k`'s contribution is at most its amplitude at the window's
/// start times `√(1 + xₖ²) / (xₖ W)` (Dirichlet, `xₖ = ωₖ dt / 2`), so the bias is at most
/// `max_ω [|z|ⁿ √(1 + x²) / x] / W · ‖e_top‖_{M⁻¹} ‖δu₀‖_M +
///  max_ω [|z|ⁿ √(1 + x²) / (x ω̄)] / W · ‖e_top‖_{M⁻¹} ‖v₀‖_M` plus the solves' drift and
/// the warping of the static. The column can bounce (at `T₁/32` it does): `n₀` is the final
/// contact, after which the bottom stays held to the end.
#[test]
fn a_dropped_column_never_penetrates_the_ground_and_settles_to_the_roller_static_sag() {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let r = roller(&c);
    let (lambda, mu) = lame_mpa();
    let w_max = omega_max(&c);
    let drop_mm = 10.0;
    let fall_time = (2.0 * drop_mm / (G * 1000.0)).sqrt();
    for &per_period in &STEPS_PER_PERIOD {
        let dt = period / per_period as f64;
        let steps = per_period * 3 * PERIODS;
        let (mut solver, ground, beta) = standing_solver(&c, omega, drop_mm, dt);
        let mut touched_at = None;
        // the step the whole bottom was last found held after not being held: the column can
        // bounce (at the coarse step it does), and the bound starts from the final contact
        let mut held_at = None;
        let mut deviation = 0.0;
        let mut speed = 0.0;
        let mut drift = 0.0;
        let mut sags = Vec::with_capacity(steps);
        for step in 1..=steps {
            solver.step().expect("the step converges");
            let u = to_f64(solver.displacements());
            let lowest = lowest_y_mm(&c, solver.displacements());
            assert!(
                lowest >= ground,
                "dt = T/{per_period} step {step}: lowest {lowest:.6e} below ground {ground:.6e}"
            );
            if touched_at.is_none() && solver.on_ground().iter().any(|&h| h) {
                touched_at = Some(step as f64 * dt);
            }
            let whole_bottom = c.base.iter().all(|&i| solver.on_ground()[i]);
            if !whole_bottom {
                held_at = None;
            } else if held_at.is_none() {
                held_at = Some(step);
                // the deviation from the static on the ground (the static shifted down by
                // the drop) and the velocity, both in the M-norm
                let delta: Vec<f64> = (0..u.len())
                    .map(|d| u[d] - r.u_static[d] + if d % 3 == 1 { drop_mm } else { 0.0 })
                    .collect();
                deviation = m_norm(&c, &delta);
                speed = m_norm(&c, &to_f64(solver.velocities()));
                drift = 0.0;
            }
            if held_at.is_some() {
                drift += solve_error_m(&c, dt, (0.0, beta), true, &u);
            }
            // the sag against the plane: the column's height above it
            let top_y = c
                .top
                .iter()
                .map(|&i| f64::from(c.mm.vertices[i][1]) + u[3 * i + 1])
                .sum::<f64>()
                / c.top.len() as f64;
            sags.push(ground + 2.0 * HALF_H * 1000.0 - top_y);
        }
        let touched = touched_at.expect("the column touches down");
        let held_at = held_at.expect("the whole bottom is held at the end");
        assert!(
            (touched - fall_time).abs() <= dt,
            "dt = T/{per_period}: touchdown {touched:.5e} vs free fall {fall_time:.5e}"
        );
        let window = (2.0 * 2.0 * std::f64::consts::PI / r.omega / dt).round() as usize;
        assert!(
            held_at + window < steps,
            "dt = T/{per_period}: the final contact at step {held_at} leaves no whole window"
        );
        let n_before = steps - window - held_at;
        let sag = sags[steps - window..].iter().sum::<f64>() / window as f64;
        let kernel = |w: f64| {
            let x = w * dt / 2.0;
            trapezoidal_root(w, beta * w / 2.0, dt)
                .0
                .powi(n_before as i32)
                * (1.0 + x * x).sqrt()
                / x
        };
        let bias_u = max_over_modes(r.omega, w_max, kernel) * GRID_MARGIN / window as f64;
        let bias_v = max_over_modes(r.omega, w_max, |w| kernel(w) / newmark_frequency(w, dt))
            * GRID_MARGIN
            / window as f64;
        let u = to_f64(solver.displacements());
        let omega_warped = warped_axial_mode(&c.mm, lambda, mu, &c.masses_t, &r.held, &u, 12);
        let warping = (omega_warped * omega_warped / (r.omega * r.omega) - 1.0).abs();
        let tol = r.e_top_norm * (bias_u * deviation + bias_v * speed)
            + drift
                / c.masses_t
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
                    .sqrt()
            + warping * r.sag;
        eprintln!(
            "drop dt = T/{per_period}: touchdown at {touched:.5e} s vs free fall {fall_time:.5e} s (dt {dt:.3e}), whole bottom held from step {held_at}; sag over the last {window} steps {sag:.5e} mm vs roller static {:.5e} mm ({:+.2e}, tolerance {tol:.2e} mm: ‖δu₀‖_M {deviation:.2e}, ‖v₀‖_M {speed:.2e}, bias {bias_u:.2e} / {bias_v:.2e})",
            r.sag,
            sag - r.sag
        );
        assert!(
            (sag - r.sag).abs() <= tol,
            "dt = T/{per_period}: sag {sag:.5e} vs roller {:.5e} (tolerance {tol:.2e})",
            r.sag
        );
    }
}

/// a free column (no base, no ground, no load) of the given mass distribution
fn free_solver(c: &Column, lumping: MassLumping, dt: f64) -> CorotationalTransientSolver {
    let bc = BoundaryConditions::new();
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_f64(dt),
        lumping,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-9)).expect("budget and tolerance");
    CorotationalTransientSolver::new(&c.mm, &c.material, &bc, &dynamics, &config)
        .expect("a free body solves")
}

fn total_momentum(solver: &CorotationalTransientSolver) -> [f64; 3] {
    let p = solver.momentum();
    let mut sum = [0.0; 3];
    for (d, x) in p.iter().enumerate() {
        sum[d % 3] += x.to_f64();
    }
    sum
}

/// the Lamé constants of the column's material in MPa
fn lame_mpa() -> (f64, f64) {
    let e = YOUNG_PA * 1e-6;
    (
        e * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON)),
        e / (2.0 * (1.0 + POISSON)),
    )
}

/// The smallest eigenvalue of the mass matrix the solver uses, from the lumped vertex
/// masses: the lumped matrix is their diagonal; the consistent one is assembled from element
/// matrices `(ρ Vₑ / 20)(1 + δᵢⱼ)` whose eigenvalues are at least `ρ Vₑ / 20`, and
/// `uᵀ M u = Σₑ uₑᵀ Mₑ uₑ ≥ Σᵢ ‖uᵢ‖² Σ_{e ∋ i} ρ Vₑ / 20 = Σᵢ (mᵢ / 5) ‖uᵢ‖²`
/// (`mᵢ = ρ Σ_{e ∋ i} Vₑ / 4`)
fn mass_eigenvalue_floor(masses_t: &[f64], lumping: MassLumping) -> f64 {
    let lo = masses_t.iter().copied().fold(f64::INFINITY, f64::min);
    match lumping {
        MassLumping::Lumped => lo,
        MassLumping::Consistent => lo / 5.0,
    }
}

/// what a completed conjugate gradient solve guarantees for its residual: at most the
/// relative tolerance times the right-hand side's norm, or the representation's floor
fn residual_bound(tol: f64, rhs_norm: f64) -> f64 {
    (tol * rhs_norm).max(RESIDUAL_NORM_FLOOR.to_f64())
}

/// An oblique impulse on the top centre of a free column (no fix, no ground, no load):
/// `Σ M v = J` right after, and over 200 free steps. The bounds are derived from what the
/// conjugate gradient guarantees (`‖r‖₂ ≤ max(tol ‖b‖₂, floor)` for a completed solve), not
/// chosen:
///
/// * the mass solve `Ṁ y = J` leaves `Ṁ y = J − r`, and the momentum is `M Δv = Ṁ y`, so its
///   error is `Σᵢ rᵢ ≤ √n ‖r‖₂` whatever the mass distribution;
/// * each free step solves `A u' = b`, `A = (4/dt²) M + K'`, so `A δu' = r` and the
///   corrector gives `δv' = (2/dt) δu'`; the momentum error is `1ᵀ M δv' =
///   (2/dt) (M^{1/2} 1)ᵀ Â⁻¹ M^{-1/2} r` with `Â = M^{-1/2} A M^{-1/2} ≥ (4/dt²) I`, so at most
///   `(2/dt) √M_total (dt²/4) ‖r‖₂ / √λ_min(M) = (dt/2) √(M_total / λ_min(M)) ‖r‖₂`; the
///   scheme itself keeps `1ᵀ M v` exactly for a free body (the rows of every warped element
///   matrix sum to zero), so the errors add over the steps. `‖b‖₂ = ‖A u'‖₂` is evaluated in
///   `f64` from the measured `u'` with the same warping (its polar factors differ from the
///   solver's by at most `2⁻²⁰`, allowed for as `2⁻¹⁹ λ_K ‖u'‖₂` with `λ_K` the Gershgorin
///   bound of the stiffness).
///
/// `max‖u'‖₂` and `max|u'|∞` are measured over the run and printed.
fn check_impulse(c: &Column, lumping: MassLumping, label: &str) {
    let dt = 1.0 / 2000.0;
    let tol = 1e-9;
    let steps = 200;
    let mut solver = free_solver(c, lumping, dt);
    let n_v = c.mm.vertices.len();
    let n_dof = 3 * n_v;
    // an oblique impulse at the top centre vertex: 2e-3 N·s (t·mm/s) along (3, 4, 0)/5
    let top_centre = *c
        .top
        .iter()
        .min_by(|&&a, &&b| {
            let r = |i: usize| {
                let v = c.mm.vertices[i];
                v[0] * v[0] + v[2] * v[2]
            };
            r(a).partial_cmp(&r(b)).expect("finite")
        })
        .expect("a top");
    let j: [f64; 3] = [1.2e-3, 1.6e-3, 0.0];
    let j_norm = (j[0] * j[0] + j[1] * j[1]).sqrt();
    solver
        .apply_impulses(&[(
            top_centre as u32,
            [Fix128::from_f64(j[0]), Fix128::from_f64(j[1]), Fix128::ZERO],
        )])
        .expect("the mass solve converges");
    let p0 = total_momentum(&solver);
    let total_mass: f64 = c.masses_t.iter().sum();
    let immediate = (n_dof as f64).sqrt() * residual_bound(tol, j_norm);
    let (lambda, mu) = lame_mpa();
    let lambda_k = stiffness_eigenvalue_bound(&c.mm, lambda, mu);
    let mass_floor = mass_eigenvalue_floor(&c.masses_t, lumping);
    let inv_beta_dt2 = 4.0 / (dt * dt);
    let consistent = matches!(lumping, MassLumping::Consistent);
    let mut u_norm_max = 0.0f64;
    let mut u_inf_max = 0.0f64;
    let mut kept = 0.0f64;
    for _ in 0..steps {
        solver.step().expect("the step converges");
        let u: Vec<f64> = solver.displacements().iter().map(|x| x.to_f64()).collect();
        let u_norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        u_norm_max = u_norm_max.max(u_norm);
        u_inf_max = u_inf_max.max(u.iter().map(|x| x.abs()).fold(0.0, f64::max));
        let au =
            newmark_operator_times(&c.mm, lambda, mu, &c.masses_t, consistent, inv_beta_dt2, &u);
        let rhs_norm =
            au.iter().map(|x| x * x).sum::<f64>().sqrt() + 2f64.powi(-19) * lambda_k * u_norm;
        kept += (dt / 2.0) * (total_mass / mass_floor).sqrt() * residual_bound(tol, rhs_norm);
    }
    let p1 = total_momentum(&solver);
    // centre of mass velocity from the lumped vertex masses (the same total for both distributions)
    let v = solver.velocities();
    let mut com_v = [0.0; 3];
    for i in 0..n_v {
        for k in 0..3 {
            com_v[k] += c.masses_t[i] * v[3 * i + k].to_f64();
        }
    }
    for x in &mut com_v {
        *x /= total_mass;
    }
    eprintln!(
        "{label}: momentum right after {p0:?} vs J {j:?} (bound {immediate:.3e}), after {steps} steps {p1:?} (bound {kept:.3e}; max‖u‖₂ {u_norm_max:.3e} mm, max|u|∞ {u_inf_max:.3e} mm, λ_K {lambda_k:.3e} N/mm)"
    );
    for k in 0..3 {
        assert!(
            (p0[k] - j[k]).abs() <= immediate,
            "{label}: momentum {p0:?} vs J {j:?} (bound {immediate:.3e})"
        );
        assert!(
            (p1[k] - j[k]).abs() <= kept,
            "{label}: momentum after {steps} steps {p1:?} vs J {j:?} (bound {kept:.3e})"
        );
        assert!(
            (com_v[k] - j[k] / total_mass).abs() <= kept / total_mass,
            "{label}: centre of mass velocity"
        );
    }
}

/// With the lumped mass the impulse lands to rounding (the preconditioned mass solve of a
/// diagonal system is one iteration), and is kept over 200 free steps within the bound the
/// solve tolerance allows (derived at `check_impulse`).
#[test]
fn an_impulse_on_a_free_lumped_body_is_its_momentum_exactly_and_is_kept() {
    let c = column();
    check_impulse(&c, MassLumping::Lumped, "lumped");
}

/// With the consistent mass the impulse is distributed by the mass solve: `Σ M v = J` to the
/// conjugate gradient tolerance times the mass matrix's condition number, kept over 200 free
/// steps within the derived bound.
#[test]
fn an_impulse_on_a_free_consistent_body_is_its_momentum_to_the_solve_tolerance() {
    let c = column();
    check_impulse(&c, MassLumping::Consistent, "consistent");
}

/// An impulse on a vertex held on the ground (asserted held, with zero vertical velocity,
/// right before each impulse, since the hold is re-decided every step): the upward part
/// releases it (its velocity becomes the impulse over its mass, the next step finds it above
/// the plane, no longer held), the downward part is taken by the ground (the vertex's vertical
/// velocity stays zero, the body's vertical momentum does not change, no step ends below the
/// plane), and the horizontal part is the body's momentum as for a free vertex. The bound on
/// the kept horizontal momentum is the mass solve's residual guarantee, as in
/// `check_impulse`.
#[test]
fn an_impulse_on_a_held_vertex_releases_it_upward_and_is_taken_by_the_ground_downward() {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let dt = 2.0 * std::f64::consts::PI / omega / 64.0;
    let (mut solver, ground, _) = standing_solver(&c, omega, 0.0, dt);
    for _ in 0..400 {
        solver.step().expect("the step converges");
    }
    assert!(
        c.base.iter().all(|&i| solver.on_ground()[i]),
        "the whole bottom is held"
    );
    let n_v = c.mm.vertices.len();
    let bottom_centre = *c
        .base
        .iter()
        .min_by(|&&a, &&b| {
            let r = |i: usize| {
                let v = c.mm.vertices[i];
                v[0] * v[0] + v[2] * v[2]
            };
            r(a).partial_cmp(&r(b)).expect("finite")
        })
        .expect("a base");
    let corner = c.base[0];
    assert_ne!(corner, bottom_centre);
    assert_eq!(
        solver.velocities()[3 * corner + 1],
        Fix128::ZERO,
        "the corner is held with no vertical velocity before the downward impulse"
    );
    let p_before = total_momentum(&solver);
    // the mass solve's guarantee for one impulse, as in `check_impulse` (tolerance 1e-7 here)
    let bound = |j_norm: f64| (3.0 * n_v as f64).sqrt() * residual_bound(1e-7, j_norm);

    // downward and sideways on the corner: the ground takes the vertical part
    let j = [0.0, -2e-3, 1e-3];
    solver
        .apply_impulses(&[(
            corner as u32,
            [Fix128::ZERO, Fix128::from_f64(j[1]), Fix128::from_f64(j[2])],
        )])
        .expect("the mass solve converges");
    assert_eq!(
        solver.velocities()[3 * corner + 1],
        Fix128::ZERO,
        "vertical velocity untouched"
    );
    let p = total_momentum(&solver);
    let rounding = bound(j[2].abs());
    assert!(
        (p[1] - p_before[1]).abs() <= rounding,
        "vertical momentum {p:?} vs {p_before:?}"
    );
    assert!(
        (p[2] - p_before[2] - j[2]).abs() <= rounding,
        "sideways momentum {p:?}"
    );
    for _ in 0..20 {
        solver.step().expect("the step converges");
        assert!(
            lowest_y_mm(&c, solver.displacements()) >= ground,
            "no step ends below the plane"
        );
    }
    assert!(
        solver.on_ground()[corner],
        "a pushed-down vertex stays held"
    );

    // let the kick die out, then check the target vertex is held with no vertical velocity
    // right before the upward impulse: the hold is re-decided every step, so without this the
    // impulse could land on a free vertex and the release would be asserted of nothing
    for _ in 0..400 {
        solver.step().expect("the step converges");
    }
    assert!(
        c.base.iter().all(|&i| solver.on_ground()[i]),
        "the whole bottom is held again before the upward impulse"
    );
    assert_eq!(
        solver.velocities()[3 * bottom_centre + 1],
        Fix128::ZERO,
        "a held vertex has no vertical velocity"
    );

    // upward on the bottom centre: released, and above the plane at the next step
    let up = 2e-3;
    solver
        .apply_impulses(&[(
            bottom_centre as u32,
            [Fix128::ZERO, Fix128::from_f64(up), Fix128::ZERO],
        )])
        .expect("the mass solve converges");
    let v_up = solver.velocities()[3 * bottom_centre + 1].to_f64();
    // one vertex's velocity from the diagonal solve: the momentum guarantee over its mass
    assert!(
        (v_up - up / c.masses_t[bottom_centre]).abs() <= bound(up) / c.masses_t[bottom_centre],
        "the upward impulse is the vertex's velocity: {v_up:.3e}"
    );
    solver.step().expect("the step converges");
    let y = f64::from(c.mm.vertices[bottom_centre][1])
        + solver.displacements()[3 * bottom_centre + 1].to_f64();
    assert!(!solver.on_ground()[bottom_centre], "released");
    assert!(y > ground, "above the plane: {y:.3e} vs {ground:.3e}");
}

/// Two runs of the same inputs give the same bits: the linear and the co-rotational solver,
/// the latter with damping, the ground and impulses, stepped twice from the same start,
/// agree in every raw `Fix128` of displacement, velocity and acceleration at every step and
/// in the state digest; a run with one input changed (one impulse component `1e-8 N·s`
/// larger: above the mass solve's residual floor, so the solve sees it) has another digest.
#[test]
fn two_runs_of_the_same_inputs_replay_to_the_same_bits_and_digest() {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let dt = 2.0 * std::f64::consts::PI / omega / 64.0;
    let raw = |u: &[Fix128]| -> Vec<(i64, u64)> { u.iter().map(|x| (x.hi, x.lo)).collect() };

    let linear = || {
        TransientSolver::new(
            &c.mm,
            &c.material,
            &gravity_boundary(&c),
            &dynamics(dt),
            &solver_config(),
        )
        .expect("the column solves")
    };
    let (mut a, mut b) = (linear(), linear());
    assert_eq!(a.state_digest(), b.state_digest(), "the same start");
    for step in 0..40 {
        a.step().expect("the step converges");
        b.step().expect("the step converges");
        assert_eq!(
            raw(a.displacements()),
            raw(b.displacements()),
            "linear step {step}: u"
        );
        assert_eq!(
            raw(a.velocities()),
            raw(b.velocities()),
            "linear step {step}: v"
        );
        assert_eq!(
            raw(a.accelerations()),
            raw(b.accelerations()),
            "linear step {step}: a"
        );
        assert_eq!(
            a.state_digest(),
            b.state_digest(),
            "linear step {step}: digest"
        );
    }

    let corotational = |nudge: f64| {
        let (mut solver, _, _) = standing_solver(&c, omega, 2.0, dt);
        let corner = c.base[0] as u32;
        solver
            .apply_impulses(&[(
                corner,
                [
                    Fix128::from_f64(1e-4 + nudge),
                    Fix128::ZERO,
                    Fix128::from_f64(-5e-5),
                ],
            )])
            .expect("the mass solve converges");
        solver
    };
    let (mut a, mut b) = (corotational(0.0), corotational(0.0));
    let digest_at_start = a.state_digest();
    assert_eq!(digest_at_start, b.state_digest(), "the same start");
    for step in 0..200 {
        a.step().expect("the step converges");
        b.step().expect("the step converges");
        assert_eq!(
            raw(a.displacements()),
            raw(b.displacements()),
            "co-rotational step {step}: u"
        );
        assert_eq!(
            raw(a.velocities()),
            raw(b.velocities()),
            "co-rotational step {step}: v"
        );
        assert_eq!(
            raw(a.accelerations()),
            raw(b.accelerations()),
            "co-rotational step {step}: a"
        );
        assert_eq!(
            a.on_ground(),
            b.on_ground(),
            "co-rotational step {step}: hold"
        );
        assert_eq!(
            a.state_digest(),
            b.state_digest(),
            "co-rotational step {step}: digest"
        );
    }
    assert!(
        a.on_ground().iter().any(|&h| h),
        "the run reached the ground"
    );
    assert_ne!(a.state_digest(), digest_at_start, "the state moved");
    let mut nudged = corotational(1e-8);
    assert_ne!(
        nudged.state_digest(),
        digest_at_start,
        "a changed impulse changes the digest"
    );
    for _ in 0..200 {
        nudged.step().expect("the step converges");
    }
    assert_ne!(
        nudged.state_digest(),
        a.state_digest(),
        "and the run after it"
    );
    eprintln!(
        "replay: linear and co-rotational runs agree to the bit; digest {}",
        hex(&a.state_digest())
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The hold is not adhesive: a column standing settled on the plane, kicked upward by an
/// impulse on every top vertex (`J = 2 M_total √(2 g h)` for a rise `h` of 2 mm, half the
/// momentum going into the body's rise), leaves the plane with every bottom vertex released
/// within the time the kick needs to lift the bottom (`√(2h/g)` plus the compression wave's
/// transit, two periods at most), rises above it, and lands again (every bottom vertex held
/// once more within the flight time plus two periods); no step ends below the plane.
#[test]
fn a_column_kicked_upward_leaves_the_plane_with_every_bottom_vertex_released() {
    let c = column();
    let (omega, _) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = period / 64.0;
    let (mut solver, ground, _) = standing_solver(&c, omega, 0.0, dt);
    for _ in 0..400 {
        solver.step().expect("the step converges");
    }
    assert!(
        c.base.iter().all(|&i| solver.on_ground()[i]),
        "settled on the plane"
    );
    let total_mass: f64 = c.masses_t.iter().sum();
    let rise_mm = 2.0;
    let g = G * 1000.0;
    let kick = 2.0 * total_mass * (2.0 * g * rise_mm).sqrt();
    let per_vertex = Fix128::from_f64(kick / c.top.len() as f64);
    let impulses: Vec<(u32, [Fix128; 3])> = c
        .top
        .iter()
        .map(|&i| (i as u32, [Fix128::ZERO, per_vertex, Fix128::ZERO]))
        .collect();
    solver
        .apply_impulses(&impulses)
        .expect("the mass solve converges");
    let flight = 2.0 * (2.0 * rise_mm / g).sqrt();
    let two_periods = (2.0 * period / dt).round() as usize;
    let mut released_at = None;
    let mut landed_at = None;
    let mut highest = f64::NEG_INFINITY;
    let steps = (flight / dt).round() as usize + 2 * two_periods;
    for step in 1..=steps {
        solver.step().expect("the step converges");
        let lowest = lowest_y_mm(&c, solver.displacements());
        assert!(lowest >= ground, "step {step}: penetration");
        highest = highest.max(lowest - ground);
        let any_held = solver.on_ground().iter().any(|&h| h);
        if released_at.is_none() && !any_held {
            released_at = Some(step);
        }
        if released_at.is_some()
            && landed_at.is_none()
            && c.base.iter().all(|&i| solver.on_ground()[i])
        {
            landed_at = Some(step);
        }
    }
    let released = released_at.expect("every bottom vertex is released");
    let landed = landed_at.expect("the column lands again");
    eprintln!(
        "kick: every vertex free from step {released} ({:.3e} s), bottom rises {highest:.3e} mm (kick sized for {rise_mm} mm), lands at step {landed} ({:.3e} s, flight {flight:.3e} s)",
        released as f64 * dt,
        landed as f64 * dt
    );
    assert!(
        released as f64 * dt <= (2.0 * rise_mm / g).sqrt() + 2.0 * period,
        "released at {released} steps"
    );
    assert!(highest > 0.0, "the bottom leaves the plane");
    assert!(
        landed as f64 * dt <= flight + 2.0 * period,
        "landed at {landed} steps vs flight {flight:.3e} s"
    );
}
