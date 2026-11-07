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
//!   on its own.
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
    self, BoundaryConditions, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;

#[path = "support/deformable_column.rs"]
mod column;
use column::{base_vertices, column_mesh, lumped_masses, top_vertices};
#[path = "support/modal_axial.rs"]
mod modal;
use modal::{axial_mode, newmark_frequency};

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

/// static sag, mm, positive downward
fn static_sag_mm(c: &Column) -> f64 {
    let sol = linear_elastic_fem::solve(
        &c.mm,
        &c.material,
        &gravity_boundary(c),
        &SolverConfig::default(),
    )
    .expect("the column solves");
    c.top
        .iter()
        .map(|&i| -sol.displacements[i][1].to_f64())
        .sum::<f64>()
        / c.top.len() as f64
}

fn top_sag_mm(c: &Column, u: &[Fix128]) -> f64 {
    c.top.iter().map(|&i| -u[3 * i + 1].to_f64()).sum::<f64>() / c.top.len() as f64
}

/// the mean period of a signal from its crossings of `level`, both directions, linear
/// interpolation between samples; `None` with fewer than three crossings
fn period_from_crossings(signal: &[f64], level: f64, dt: f64) -> Option<f64> {
    let mut crossings = Vec::new();
    for k in 1..signal.len() {
        let (a, b) = (signal[k - 1] - level, signal[k] - level);
        if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0) {
            let frac = a / (a - b);
            crossings.push((k as f64 - 1.0 + frac) * dt);
        }
    }
    if crossings.len() < 3 {
        return None;
    }
    // successive crossings are half periods
    Some(2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64)
}

/// the modal coordinate `q₁ = φ₁ᵀ M u` of a displacement field (mm), with `φ₁` of unit
/// `M`-norm: the single-mode signal the period is read from (the top sag mixes the higher
/// axial modes in, which shift its crossings)
fn modal_coordinate(c: &Column, phi: &[f64], u: &[Fix128]) -> f64 {
    (0..c.mm.vertices.len())
        .map(|i| {
            c.masses_t[i]
                * (0..3)
                    .map(|k| phi[3 * i + k] * u[3 * i + k].to_f64())
                    .sum::<f64>()
        })
        .sum()
}

/// run `steps` Newmark steps of `dt` seconds from rest under gravity, recording the top sag
/// (mm) and the modal coordinate `q₁`
fn linear_run(c: &Column, phi: &[f64], dt: f64, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_f64(dt),
        MassLumping::Lumped,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-7)).expect("budget and tolerance");
    let mut solver =
        TransientSolver::new(&c.mm, &c.material, &gravity_boundary(c), &dynamics, &config)
            .expect("the column solves");
    let mut sag = Vec::with_capacity(steps + 1);
    let mut q = Vec::with_capacity(steps + 1);
    sag.push(top_sag_mm(c, solver.displacements()));
    q.push(modal_coordinate(c, phi, solver.displacements()));
    for _ in 0..steps {
        solver.step().expect("the step converges");
        sag.push(top_sag_mm(c, solver.displacements()));
        q.push(modal_coordinate(c, phi, solver.displacements()));
    }
    (sag, q)
}

fn corotational_run(
    c: &Column,
    phi: &[f64],
    dt: f64,
    steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), FemError> {
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_f64(dt),
        MassLumping::Lumped,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-7)).expect("budget and tolerance");
    let mut solver = CorotationalTransientSolver::new(
        &c.mm,
        &c.material,
        &gravity_boundary(c),
        &dynamics,
        &config,
    )?;
    let mut sag = Vec::with_capacity(steps + 1);
    let mut q = Vec::with_capacity(steps + 1);
    sag.push(top_sag_mm(c, solver.displacements()));
    q.push(modal_coordinate(c, phi, solver.displacements()));
    for _ in 0..steps {
        solver.step()?;
        sag.push(top_sag_mm(c, solver.displacements()));
        q.push(modal_coordinate(c, phi, solver.displacements()));
    }
    Ok((sag, q))
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

/// The linear Newmark solver's first axial modal coordinate `q₁(t) = φ₁ᵀ M u(t)` oscillates
/// about its mean with the period `2π/ω̄`, `ω̄ = (2/dt)·atan(ω₁ dt/2)` (the scheme and the
/// modal decomposition commute, so `q₁` is a single-mode signal; the top sag is not, the
/// higher axial modes shift its crossings by 1.5 %, measured), with `dt = T₁/64`: within the
/// crossing resolution, one 64th of a period interpolated linearly and averaged over the
/// crossings of 8 periods, below 0.5 %. The mean of the top sag over the run is the static
/// sag within 1 % (8 periods plus the higher modes' small share).
#[test]
fn the_linear_newmark_oscillation_has_the_modal_period_and_the_static_mean() {
    let c = column();
    let (omega, phi) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = period / 64.0;
    let steps = 64 * 8;
    let (sag, q) = linear_run(&c, &phi, dt, steps);
    let static_sag = static_sag_mm(&c);
    let expected = 2.0 * std::f64::consts::PI / newmark_frequency(omega, dt);
    let q_mean = q.iter().sum::<f64>() / q.len() as f64;
    let measured = period_from_crossings(&q, q_mean, dt).expect("at least three crossings");
    let mean = sag.iter().sum::<f64>() / sag.len() as f64;
    eprintln!(
        "linear: period {measured:.6e} s vs 2π/ω̄ {expected:.6e} s ({:+.4}); mean sag {mean:.4e} mm vs static {static_sag:.4e} mm ({:+.4})",
        measured / expected - 1.0,
        mean / static_sag - 1.0
    );
    assert!(
        (measured / expected - 1.0).abs() < 0.005,
        "period {measured:.6e} vs {expected:.6e}"
    );
    assert!(
        (mean / static_sag - 1.0).abs() < 0.01,
        "mean {mean:.4e} vs static {static_sag:.4e}"
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
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_ratio(1, 240),
        MassLumping::Lumped,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-7)).expect("budget and tolerance");
    let mut solver = CorotationalTransientSolver::new(&c.mm, &c.material, &bc, &dynamics, &config)
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
    for _ in 0..60 {
        solver.step().expect("the step converges");
    }
    let u = solver.displacements();
    let worst = (0..c.mm.vertices.len())
        .map(|i| {
            (0..3)
                .map(|k| (u[3 * i + k].to_f64() - initial[i][k]).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    // the polar decomposition is determined to 2⁻⁶⁴ × 2⁴ and the mesh is 200 mm tall: nothing
    // above the solve tolerance times the size
    assert!(
        worst < 1e-4,
        "a vertex moved {worst:.3e} mm under a rigid rotation"
    );
}

/// The co-rotational solver's sag under its own weight: the mean of its oscillation over
/// whole periods is the static linear sag within 1 % (the strain is `4e-4`, so the rotation
/// correction is below that), and the period is the modal one through the Newmark relation.
#[test]
fn the_corotational_oscillation_has_the_modal_period_and_the_static_mean() {
    let c = column();
    let (omega, phi) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = period / 64.0;
    let (sag, q) = corotational_run(&c, &phi, dt, 64 * 8).expect("the column solves");
    let static_sag = static_sag_mm(&c);
    let expected = 2.0 * std::f64::consts::PI / newmark_frequency(omega, dt);
    let q_mean = q.iter().sum::<f64>() / q.len() as f64;
    let measured = period_from_crossings(&q, q_mean, dt).expect("at least three crossings");
    let mean = sag.iter().sum::<f64>() / sag.len() as f64;
    eprintln!(
        "co-rotational: period {measured:.6e} s vs 2π/ω̄ {expected:.6e} s ({:+.4}); mean sag {mean:.4e} mm vs static {static_sag:.4e} mm ({:+.4})",
        measured / expected - 1.0,
        mean / static_sag - 1.0
    );
    assert!(
        (measured / expected - 1.0).abs() < 0.005,
        "period {measured:.6e} vs {expected:.6e}"
    );
    assert!(
        (mean / static_sag - 1.0).abs() < 0.01,
        "mean {mean:.4e} vs static {static_sag:.4e}"
    );
}

/// the solver with Rayleigh damping, from rest under gravity, recording `q₁`
fn damped_run(
    c: &Column,
    phi: &[f64],
    damping: RayleighCoefficients,
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_f64(dt),
        MassLumping::Lumped,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-7)).expect("budget and tolerance");
    let mut solver = CorotationalTransientSolver::new(
        &c.mm,
        &c.material,
        &gravity_boundary(c),
        &dynamics,
        &config,
    )
    .expect("the column solves");
    solver.set_damping(Some(damping));
    let mut q = Vec::with_capacity(steps + 1);
    q.push(modal_coordinate(c, phi, solver.displacements()));
    for _ in 0..steps {
        solver.step().expect("the step converges");
        q.push(modal_coordinate(c, phi, solver.displacements()));
    }
    q
}

/// the local maxima of a signal, parabolically interpolated: `(time, value)`
fn peaks(signal: &[f64], dt: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for k in 1..signal.len() - 1 {
        let (a, b, c) = (signal[k - 1], signal[k], signal[k + 1]);
        if b > a && b >= c {
            let denom = a - 2.0 * b + c;
            let offset = if denom == 0.0 {
                0.0
            } else {
                0.5 * (a - c) / denom
            };
            let value = b - 0.25 * (a - c) * offset;
            out.push(((k as f64 + offset) * dt, value));
        }
    }
    out
}

/// `(|z|, arg z)` of the trapezoidal map of the damped root: the per-step amplitude factor
/// and phase advance of the first mode
fn trapezoidal_root(omega: f64, zeta: f64, dt: f64) -> (f64, f64) {
    // λ = −ζω + i ω√(1 − ζ²); z = (1 + λ dt/2) / (1 − λ dt/2)
    let (re, im) = (
        -zeta * omega * dt / 2.0,
        omega * (1.0 - zeta * zeta).sqrt() * dt / 2.0,
    );
    let (nr, ni) = (1.0 + re, im);
    let (dr, di) = (1.0 - re, -im);
    let d2 = dr * dr + di * di;
    let zr = (nr * dr + ni * di) / d2;
    let zi = (ni * dr - nr * di) / d2;
    ((zr * zr + zi * zi).sqrt(), zi.atan2(zr))
}

/// The decay and the frequency of the damped first mode against the trapezoidal closed form,
/// for one Rayleigh coefficient at a time (`ζ = 0.05` on the first mode): the log-decrement
/// per period between successive peaks of `q₁ − mean` within 2 % of `−ln|z|` per step times
/// the steps per period (the peaks are parabolically interpolated at 64 samples per period,
/// 0.1 % in amplitude, and the mean is the tail's static value), the period within 0.5 % of
/// `2π dt / arg z`.
fn check_damped(c: &Column, omega: f64, phi: &[f64], damping: RayleighCoefficients, label: &str) {
    let zeta = damping.damping_ratio(Fix128::from_f64(omega)).to_f64();
    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = period / 64.0;
    let steps = 64 * 8;
    let q = damped_run(c, phi, damping, dt, steps);
    // the mean the oscillation settles to: the static modal coordinate, read from the tail
    let tail = &q[q.len() - 64..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let centred: Vec<f64> = q.iter().map(|x| x - mean).collect();
    let pk = peaks(&centred, dt);
    assert!(pk.len() >= 4, "{label}: {} peaks", pk.len());
    let (abs_z, arg_z) = trapezoidal_root(omega, zeta, dt);
    let expected_period = 2.0 * std::f64::consts::PI * dt / arg_z;
    let expected_decrement = -abs_z.ln() * expected_period / dt;
    let first = pk[1];
    let last = pk[pk.len() - 1];
    let periods = (last.0 - first.0) / expected_period;
    let measured_period = (last.0 - first.0) / periods.round();
    let measured_decrement = (first.1 / last.1).ln() / periods.round();
    eprintln!(
        "{label}: ζ {zeta:.4}; log-decrement per period {measured_decrement:.5} vs {expected_decrement:.5} ({:+.4}); period {measured_period:.6e} vs {expected_period:.6e} ({:+.5})",
        measured_decrement / expected_decrement - 1.0,
        measured_period / expected_period - 1.0
    );
    assert!(
        (measured_decrement / expected_decrement - 1.0).abs() < 0.02,
        "{label}: decrement"
    );
    assert!(
        (measured_period / expected_period - 1.0).abs() < 0.005,
        "{label}: period"
    );
}

#[test]
fn stiffness_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate() {
    let c = column();
    let (omega, phi) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    // β alone: ζ = β ω / 2 = 0.05
    let damping = RayleighCoefficients {
        alpha: Fix128::ZERO,
        beta: Fix128::from_f64(2.0 * 0.05 / omega),
    };
    check_damped(&c, omega, &phi, damping, "β only");
}

#[test]
fn mass_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate() {
    let c = column();
    let (omega, phi) = axial_mode(&c.mm, &c.material, &c.masses_t, &c.base, 12);
    // α alone: ζ = α / (2 ω) = 0.05
    let damping = RayleighCoefficients {
        alpha: Fix128::from_f64(2.0 * 0.05 * omega),
        beta: Fix128::ZERO,
    };
    check_damped(&c, omega, &phi, damping, "α only");
}
