//! Closed-form oracles for the plastic dissipation reported by
//! `linear_elastic_fem::solve_elastoplastic`, and for the heat it becomes.
//!
//! # The discrete identity, derived from the return mapping
//!
//! The radial return commits `Δε_p = (3/2) Δε̄ · s_tr/q` and
//! `σ^{n+1} = σ_tr − 3μ Δε̄ · s_tr/q`. The plastic work of one step is
//!
//! ```text
//! ΔW_p = σ^{n+1} : Δε_p
//!      = (3/2)(Δε̄/q)(1 − 3μΔε̄/q)(s_tr : s_tr)
//!      = Δε̄ (q − 3μΔε̄)                        [since s:s = (2/3) q²]
//!      = Δε̄ · q^{n+1}
//! ```
//!
//! and consistency (`Δε̄ = f/(3μ+H)`, `f = q − σ_y − H ε̄ⁿ`) gives
//! `q^{n+1} = σ_y + H ε̄^{n+1}` exactly. Summing over the path, with
//! `ε̄_p = Σ Δε̄_k` and `Σ Δε̄_k ε̄_k = (ε̄_p² + Σ Δε̄_k²)/2`:
//!
//! ```text
//! W_p = σ_y·ε̄_p + (H/2)(ε̄_p² + Σ Δε̄_k²)
//! ```
//!
//! ⚠️ This is **not** the continuous `∫(σ_y + Hε̄) dε̄ = σ_y ε̄_p + (H/2) ε̄_p²`.
//! The discrete form exceeds it by `(H/2) Σ Δε̄_k²`, which is `O(1/N)` in the
//! number of plastic steps and vanishes only in the limit. Both bounds below
//! are closed forms in quantities the test already knows, so no internal
//! state has to be exposed to check them:
//!
//! ```text
//! σ_y ε̄_p + (H/2) ε̄_p²   ≤   W_p   ≤   σ_y ε̄_p + H ε̄_p²
//! ```
//!
//! because `0 < Σ Δε̄_k² ≤ ε̄_p²`, with the upper bound **attained exactly** at
//! `N = 1` (one plastic step) and the lower bound approached as `N → ∞`.
//!
//! # ⚠️ Why `ε̄_p` alone cannot be the oracle
//!
//! `ε̄_p` is a function of the end state and is the same for every `N`, while
//! `W_p` depends on how the path was cut into steps. A test that checks only
//! `ε̄_p` therefore **cannot detect a wrong integration rule for the work** —
//! it is structurally blind to it, in the same way a uniform field cannot see
//! where a quantity is sampled. `dissipation_depends_on_the_step_count_while_
//! the_equivalent_strain_does_not` pins exactly that difference, so the blind
//! spot stays documented rather than rediscovered.
//!
//! # `H = 0` is the tolerance-free case
//!
//! With no hardening `W_p = σ_y·ε̄_p` holds for **any** `N` and **any** path,
//! so the oracle needs no tolerance beyond the arithmetic itself: the
//! implementation accumulates `Σ_k (Δε̄_k · σ_y)` while the oracle computes
//! `(Σ_k Δε̄_k) · σ_y`, and `Fix128` multiplication truncates, so the two differ
//! by fewer than `N` ulp. That bound is derived from the arithmetic, not tuned
//! to an observation.

#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    deposit_plastic_heat, plastic_temperature_rise, solve_elastoplastic, solve_with_eigenstrain,
    Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig, ElastoplasticSolution,
    FemError, PlasticHeating, SolverConfig, ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// Scene, shared with `analytic_elastoplastic_fem.rs`
// ---------------------------------------------------------------------------

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet subdivision of `[0,nx·h] × [0,ny·h] × [0,nz·h]` (conforming).
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
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
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
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

/// `n / 2^k`, exact in `Fix128`.
fn dy(n: i64, k: u32) -> Fix128 {
    if k == 0 {
        return Fix128::from_int(n);
    }
    Fix128::from_raw(0, 1u64 << (64 - k)) * Fix128::from_int(n)
}

const E_MPA: f64 = 1024.0;
const NU: f64 = 0.25;
const SIGMA_Y: f64 = 2.0;
const H_PLASTIC: f64 = 1024.0;

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

fn newton_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24) // 2^-40
}

fn config(sigma_y: f64, h: f64) -> ElastoplasticConfig {
    ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        newton_tol(),
        fx(sigma_y),
        fx(h),
    )
    .expect("valid elastoplastic config")
}

fn bar_mesh() -> SdfTetMesh {
    kuhn_box(2, 1, 1, 2.0)
}

fn bar_displacement_bc(eps_ref: Fix128) -> BoundaryConditions {
    let (nx, ny, nz) = (2usize, 1usize, 1usize);
    let mut bc = BoundaryConditions::new();
    let length = Fix128::from_int(4);
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node_index(nx, ny, nx, j, k), Axis::X, eps_ref * length);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    bc
}

/// Strain path `ε_ref = 2⁻⁷`; the yield strain is `2⁻⁹`, so factor 1 is four
/// times yield and the bar is well into the plastic range.
fn eps_ref() -> Fix128 {
    dy(1, 7)
}

/// A monotone path of `n` equal steps ending at load factor 1.
fn uniform_path(n: i64) -> Vec<Fix128> {
    (1..=n)
        .map(|k| Fix128::from_int(k) / Fix128::from_int(n))
        .collect()
}

fn run(cfg: &ElastoplasticConfig, path: &[Fix128]) -> ElastoplasticSolution {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    solve_elastoplastic(&mesh, &material(), &bc, cfg, path).expect("elastoplastic solve succeeds")
}

/// One ulp of `Fix128`.
fn ulp() -> Fix128 {
    Fix128::from_raw(0, 1)
}

// ---------------------------------------------------------------------------
// 1. `H = 0`: `W_p = σ_y · ε̄_p`, with no tolerance beyond the arithmetic
// ---------------------------------------------------------------------------

#[test]
fn perfect_plasticity_dissipates_yield_stress_times_equivalent_strain() {
    // ⚠️ Both a dyadic and a non-dyadic yield stress. With `σ_y = 2` the
    // accumulation `Σ(Δε̄·σ_y)` is a shift and the identity holds to **0 ulp**
    // at every `N` (measured). `σ_y = 2.1` truncates each product, so it is the
    // case the `N + 1` bound below is actually derived for; without it the
    // bound would never be exercised and could be silently wrong.
    for &sy in &[SIGMA_Y, 2.1] {
        perfect_plasticity_at_yield_stress(sy);
    }
}

fn perfect_plasticity_at_yield_stress(sigma_y_value: f64) {
    let sigma_y = fx(sigma_y_value);
    for n in [1i64, 4, 100] {
        let sol = run(&config(sigma_y_value, 0.0), &uniform_path(n));
        assert_eq!(
            sol.dissipation.len(),
            sol.equivalent_plastic_strain.len(),
            "one dissipation per element"
        );
        let mut yielded = 0usize;
        for (e, (&w, &eq)) in sol
            .dissipation
            .iter()
            .zip(sol.equivalent_plastic_strain.iter())
            .enumerate()
        {
            let want = sigma_y * eq;
            // Each step truncates one multiply, so at most `n` ulp separate
            // `Σ(Δε̄·σ_y)` from `(ΣΔε̄)·σ_y`. Derived, not tuned.
            let bound = ulp() * Fix128::from_int(n + 1);
            assert!(
                (w - want).abs() <= bound,
                "element {e}, N = {n}: W_p = {} but σ_y·ε̄_p = {} (gap {}, bound {})",
                w.to_f64(),
                want.to_f64(),
                (w - want).abs().to_f64(),
                bound.to_f64()
            );
            if !eq.is_zero() {
                yielded += 1;
            }
        }
        assert!(
            yielded > 0,
            "N = {n}: the scene must actually yield, or this oracle is vacuous"
        );
    }
}

#[test]
fn perfect_plasticity_dissipation_is_exact_in_a_single_step() {
    // With `N = 1` there is one multiply and no accumulation, so the identity
    // holds to the bit.
    let sol = run(&config(SIGMA_Y, 0.0), &uniform_path(1));
    let sigma_y = fx(SIGMA_Y);
    for (e, (&w, &eq)) in sol
        .dissipation
        .iter()
        .zip(sol.equivalent_plastic_strain.iter())
        .enumerate()
    {
        assert_eq!(
            w,
            sigma_y * eq,
            "element {e}: W_p must equal σ_y·ε̄_p exactly"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. `H > 0`: the two-sided closed-form bracket
// ---------------------------------------------------------------------------

#[test]
fn hardening_dissipation_lies_between_the_continuous_and_single_step_forms() {
    let (sigma_y, h) = (fx(SIGMA_Y), fx(H_PLASTIC));
    let half = Fix128::from_raw(0, 1 << 63);
    for n in [1i64, 4, 100] {
        let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(n));
        for (e, (&w, &eq)) in sol
            .dissipation
            .iter()
            .zip(sol.equivalent_plastic_strain.iter())
            .enumerate()
        {
            if eq.is_zero() {
                assert_eq!(w, Fix128::ZERO, "element {e}: no yield means no work");
                continue;
            }
            let lower = sigma_y * eq + half * h * eq * eq; // continuous
            let upper = sigma_y * eq + h * eq * eq; // ΣΔε̄² = ε̄_p², i.e. N = 1
            let slack = ulp() * Fix128::from_int(4 * (n + 1));
            assert!(
                w + slack >= lower && w <= upper + slack,
                "element {e}, N = {n}: W_p = {} outside [{}, {}]",
                w.to_f64(),
                lower.to_f64(),
                upper.to_f64()
            );
        }
    }
}

#[test]
fn a_single_plastic_step_attains_the_upper_bound() {
    // `N = 1` makes `Σ Δε̄_k² = ε̄_p²`, so the bracket collapses onto its upper
    // end and the oracle has no slack left to hide an error in.
    let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(1));
    let (sigma_y, h) = (fx(SIGMA_Y), fx(H_PLASTIC));
    let mut yielded = 0usize;
    for (e, (&w, &eq)) in sol
        .dissipation
        .iter()
        .zip(sol.equivalent_plastic_strain.iter())
        .enumerate()
    {
        if eq.is_zero() {
            continue;
        }
        yielded += 1;
        let want = sigma_y * eq + h * eq * eq;
        let bound = ulp() * Fix128::from_int(8);
        assert!(
            (w - want).abs() <= bound,
            "element {e}: W_p = {} but σ_y ε̄_p + H ε̄_p² = {} (gap {})",
            w.to_f64(),
            want.to_f64(),
            (w - want).abs().to_f64()
        );
    }
    assert!(yielded > 0, "the scene must yield");
}

#[test]
fn the_excess_over_the_continuous_form_is_first_order_in_the_step() {
    // excess = (H/2) Σ Δε̄_k² ≈ (H/2) ε̄_p² / N for equal increments, so
    // `excess · N` is bounded and `excess` falls by about the step ratio.
    let (sigma_y, h) = (fx(SIGMA_Y), fx(H_PLASTIC));
    let half = Fix128::from_raw(0, 1 << 63);
    let mut row = Vec::new();
    for n in [1i64, 4, 100, 1000] {
        let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(n));
        let (&w, &eq) = sol
            .dissipation
            .iter()
            .zip(sol.equivalent_plastic_strain.iter())
            .find(|(_, e)| !e.is_zero())
            .expect("some element yields");
        let continuous = sigma_y * eq + half * h * eq * eq;
        row.push((n, (w - continuous).to_f64()));
    }
    for w in row.windows(2) {
        let (n0, e0) = w[0];
        let (n1, e1) = w[1];
        assert!(
            e1 < e0,
            "excess must fall as the path is refined: N={n0} gave {e0:e}, N={n1} gave {e1:e}"
        );
    }
    let (_, first) = row[0];
    let (_, last) = row[row.len() - 1];
    assert!(
        last < first / 100.0,
        "a 1000x refinement must cut the excess by far more than 100x: {first:e} -> {last:e}"
    );
}

#[test]
fn dissipation_depends_on_the_step_count_while_the_equivalent_strain_does_not() {
    // ⚠️ The structural blind spot, pinned: an oracle that only reads `ε̄_p`
    // sees the same number for every `N` and therefore cannot detect a wrong
    // integration rule for the work.
    let coarse = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(1));
    let fine = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(1000));
    let idx = coarse
        .equivalent_plastic_strain
        .iter()
        .position(|e| !e.is_zero())
        .expect("some element yields");

    let (eq_c, eq_f) = (
        coarse.equivalent_plastic_strain[idx],
        fine.equivalent_plastic_strain[idx],
    );
    assert!(
        (eq_c - eq_f).abs().to_f64() < 1e-8,
        "ε̄_p must not depend on the step count: {} vs {}",
        eq_c.to_f64(),
        eq_f.to_f64()
    );

    let (w_c, w_f) = (coarse.dissipation[idx], fine.dissipation[idx]);
    let spread = (w_c - w_f).abs().to_f64() / w_f.to_f64();
    assert!(
        spread > 1e-3,
        "W_p must depend on the step count, or this test is not pinning the blind spot \
         (relative spread {spread:e})"
    );
}

#[test]
fn an_elastic_path_dissipates_nothing() {
    // Positive control for the whole family: below yield the dissipation must
    // be exactly zero, not merely small.
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(dy(1, 11)); // 2^-11 < yield strain 2^-9
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");
    for (e, &w) in sol.dissipation.iter().enumerate() {
        assert_eq!(w, Fix128::ZERO, "element {e} stayed elastic");
    }
}

#[test]
fn the_committed_stress_sits_on_the_yield_surface() {
    // ⚠️ The implementation accumulates `Δε̄ · (σ_y + H·ε̄^{n+1})` rather than
    // `Δε̄ · (q − 3μΔε̄)`. The two are the same quantity only because the
    // return mapping lands exactly on the yield surface, so that identity is
    // what holds the dissipation to the stress actually reported. Pinning it
    // here means the choice made in `return_map` cannot drift into a
    // dissipation computed against a surface the stress is not on.
    let (sigma_y, h) = (fx(SIGMA_Y), fx(H_PLASTIC));
    for n in [1i64, 4, 100] {
        let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(n));
        let mut checked = 0usize;
        for (e, (stress, &eq)) in sol
            .field
            .element_stress
            .iter()
            .zip(sol.equivalent_plastic_strain.iter())
            .enumerate()
        {
            if eq.is_zero() {
                continue;
            }
            checked += 1;
            let want = sigma_y + h * eq;
            let got = stress.von_mises();
            // The solve's own stress tolerance, from the sibling oracle's
            // header: the Newton and CG floors leave about 1e-9 MPa, and the
            // comparison here is against a 2 MPa surface.
            assert!(
                (got - want).abs().to_f64() <= 1e-6,
                "element {e}, N = {n}: von Mises {} is off the surface σ_y + H ε̄_p = {}",
                got.to_f64(),
                want.to_f64()
            );
        }
        assert!(checked > 0, "N = {n}: no element yielded");
    }
}

// ---------------------------------------------------------------------------
// 3. Plastic work becomes heat
// ---------------------------------------------------------------------------

/// Taylor–Quinney 0.9 and a volumetric heat capacity of 4 MPa/K (dyadic, near
/// steel's 3.82) so `ΔT = β W_p / c_v` is exact in `Fix128`.
fn heating() -> PlasticHeating {
    PlasticHeating::try_new(dy(9, 0) / Fix128::from_int(10), Fix128::from_int(4))
        .expect("valid heating")
}

#[test]
fn temperature_rise_is_taylor_quinney_work_over_heat_capacity() {
    let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(4));
    let h = heating();
    let rise = plastic_temperature_rise(&sol, &h);
    assert_eq!(rise.len(), sol.dissipation.len());
    let mut hot = 0usize;
    for (e, (&t, &w)) in rise.iter().zip(sol.dissipation.iter()).enumerate() {
        let want = h.taylor_quinney() * w / h.volumetric_heat_capacity_mpa_per_k();
        assert_eq!(t, want, "element {e}: ΔT must be β·W_p/c_v");
        if !t.is_zero() {
            hot += 1;
        }
    }
    assert!(
        hot > 0,
        "some element must heat up, or the oracle is vacuous"
    );
}

#[test]
fn a_zero_conversion_fraction_produces_no_heat_and_is_not_an_error() {
    // β = 0 is "all plastic work stored as cold work": a defined physical
    // choice, so it must give exactly zero rather than fail.
    let sol = run(&config(SIGMA_Y, H_PLASTIC), &uniform_path(4));
    let h = PlasticHeating::try_new(Fix128::ZERO, Fix128::from_int(4)).expect("β = 0 is valid");
    for (e, &t) in plastic_temperature_rise(&sol, &h).iter().enumerate() {
        assert_eq!(t, Fix128::ZERO, "element {e}: β = 0 means no rise");
    }
}

#[test]
fn degenerate_heating_parameters_are_rejected() {
    let four = Fix128::from_int(4);
    let one = Fix128::ONE;
    assert!(
        matches!(
            PlasticHeating::try_new(one, Fix128::ZERO),
            Err(FemError::InvalidConfig(_))
        ),
        "a zero heat capacity would divide by zero"
    );
    assert!(
        matches!(
            PlasticHeating::try_new(one, -four),
            Err(FemError::InvalidConfig(_))
        ),
        "a negative heat capacity would cool the material under dissipation"
    );
    assert!(
        matches!(
            PlasticHeating::try_new(-one, four),
            Err(FemError::InvalidConfig(_))
        ),
        "a negative Taylor-Quinney fraction is not a fraction"
    );
    assert!(
        matches!(
            PlasticHeating::try_new(one + one, four),
            Err(FemError::InvalidConfig(_))
        ),
        "more heat than work violates the energy balance"
    );
    assert!(
        PlasticHeating::try_new(one, four).is_ok(),
        "β = 1 is the full conversion limit and must be allowed"
    );
}

// ---------------------------------------------------------------------------
// 4. The deposit onto the grid conserves energy
// ---------------------------------------------------------------------------

/// Volume of tetrahedron `t`, computed from the mesh by the test itself.
fn tet_volume(mesh: &SdfTetMesh, t: usize) -> f64 {
    let v = mesh.tets[t].vertices;
    let p = |i: usize| {
        let a = mesh.vertices[v[i] as usize];
        [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
    };
    let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
    let e = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (a, b, c) = (e(p1, p0), e(p2, p0), e(p3, p0));
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    det.abs() / 6.0
}

/// Covers the 4 x 2 x 2 bar with a margin, at a dyadic cell size of 1 mm:
/// `(hi - lo) / (n - 1)` is `8/8`, `6/6`, `6/6`.
fn grid() -> CoupledField {
    let lo = -Fix128::ONE;
    CoupledField::try_new(
        9,
        7,
        7,
        (lo, lo, lo),
        (
            Fix128::from_int(7),
            Fix128::from_int(5),
            Fix128::from_int(5),
        ),
    )
    .expect("valid grid")
}

#[test]
fn depositing_the_heat_conserves_the_total_energy() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");
    let h = heating();

    let mut field = grid();
    deposit_plastic_heat(&mesh, &sol, &h, &mut field).expect("deposit succeeds");

    // Energy on the grid: Σ_cell ΔT · c_v · V_cell.
    let (cx, cy, cz) = field.cell_size();
    let cell_volume = (cx * cy * cz).to_f64();
    let on_grid =
        field.sum().to_f64() * cell_volume * h.volumetric_heat_capacity_mpa_per_k().to_f64();

    // Energy in the solution: Σ_e β W_p,e V_e, with V_e from the mesh.
    let in_solution: f64 = sol
        .dissipation
        .iter()
        .enumerate()
        .map(|(e, &w)| h.taylor_quinney().to_f64() * w.to_f64() * tet_volume(&mesh, e))
        .sum();

    assert!(in_solution > 0.0, "the scene must dissipate something");
    let relative = (on_grid - in_solution).abs() / in_solution;
    assert!(
        relative < 1e-9,
        "the trilinear deposit must conserve the heat: grid {on_grid:e} vs solution \
         {in_solution:e} (relative {relative:e})"
    );
}

#[test]
fn depositing_nothing_leaves_the_grid_untouched() {
    // Positive control for the deposit: an elastic solve must not warm the
    // grid at all. Without this, a deposit that ignored `dissipation` and
    // wrote a constant would still pass the conservation test on a scene whose
    // total happened to match.
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(dy(1, 11));
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");
    let mut field = grid();
    deposit_plastic_heat(&mesh, &sol, &heating(), &mut field).expect("deposit succeeds");
    assert_eq!(
        field.sum(),
        Fix128::ZERO,
        "an elastic solve deposits no heat"
    );
    assert_eq!(field.max_value(), Fix128::ZERO, "no cell may be warmed");
}

#[test]
fn a_deposit_onto_a_mismatched_mesh_is_rejected() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(2),
    )
    .expect("solve succeeds");
    let other = kuhn_box(1, 1, 1, 2.0); // 6 tets, not 12
    let mut field = grid();
    assert!(
        deposit_plastic_heat(&other, &sol, &heating(), &mut field).is_err(),
        "a solution from a different mesh must not be deposited silently"
    );
}

// ---------------------------------------------------------------------------
// 5. The loop closes: dissipation -> heat -> eigenstrain -> deformation
// ---------------------------------------------------------------------------

#[test]
fn the_dissipated_heat_drives_a_thermal_eigenstrain() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");

    let mut field = grid();
    deposit_plastic_heat(&mesh, &sol, &heating(), &mut field).expect("deposit succeeds");
    let rise = TemperatureRise::from_absolute(&field, Fix128::ZERO);

    // A free bar: only enough constraints to remove the rigid body modes, so a
    // thermal expansion shows up as displacement rather than as stress.
    let mut free = BoundaryConditions::new();
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::X, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::Y, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::Z, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 1, 0), Axis::X, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 1, 0), Axis::Z, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 1), Axis::X, Fix128::ZERO);

    let alpha = dy(1, 10); // 2^-10 per K, dyadic
    let hot = solve_with_eigenstrain(
        &mesh,
        &material(),
        &free,
        &SolverConfig::default(),
        Some(ThermalExpansion::from_rise(&rise, alpha)),
    )
    .expect("thermoelastic solve succeeds");

    let cold = solve_with_eigenstrain(&mesh, &material(), &free, &SolverConfig::default(), None)
        .expect("reference solve succeeds");

    let moved = hot
        .displacements
        .iter()
        .zip(cold.displacements.iter())
        .map(|(a, b)| {
            (a[0] - b[0])
                .abs()
                .max((a[1] - b[1]).abs())
                .max((a[2] - b[2]).abs())
        })
        .fold(Fix128::ZERO, Fix128::max);
    assert!(
        moved.to_f64() > 1e-6,
        "the plastic heat must move the bar: largest change {} mm",
        moved.to_f64()
    );
}

#[test]
fn an_elastic_solve_closes_the_loop_with_no_motion() {
    // ⚠️ The positive control of the loop test above: with no dissipation the
    // deposited field is zero, so the thermoelastic solve must reproduce the
    // reference to the bit. A deposit that leaked a constant, or an
    // eigenstrain wired to the wrong field, breaks this and not the test above.
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(dy(1, 11));
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");

    let mut field = grid();
    deposit_plastic_heat(&mesh, &sol, &heating(), &mut field).expect("deposit succeeds");
    let rise = TemperatureRise::from_absolute(&field, Fix128::ZERO);

    let mut free = BoundaryConditions::new();
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::X, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::Y, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 0), Axis::Z, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 1, 0), Axis::X, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 1, 0), Axis::Z, Fix128::ZERO);
    free.prescribe(node_index(2, 1, 0, 0, 1), Axis::X, Fix128::ZERO);

    let hot = solve_with_eigenstrain(
        &mesh,
        &material(),
        &free,
        &SolverConfig::default(),
        Some(ThermalExpansion::from_rise(&rise, dy(1, 10))),
    )
    .expect("solve succeeds");
    let cold = solve_with_eigenstrain(&mesh, &material(), &free, &SolverConfig::default(), None)
        .expect("solve succeeds");
    assert_eq!(
        hot.displacements, cold.displacements,
        "no dissipation must mean no thermal motion, to the bit"
    );
}

// ---------------------------------------------------------------------------
// 6. Degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn an_empty_mesh_is_rejected_by_the_deposit() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(2),
    )
    .expect("solve succeeds");
    let empty = SdfTetMesh::default();
    let mut field = grid();
    assert!(
        matches!(
            deposit_plastic_heat(&empty, &sol, &heating(), &mut field),
            Err(FemError::EmptyMesh)
        ),
        "an empty mesh has nowhere to deposit"
    );
}

#[test]
fn a_solution_with_no_elements_yields_an_empty_rise() {
    // `plastic_temperature_rise` on a solution with no plastic elements must
    // produce a vector of the same length, not an empty one: the caller
    // indexes it by element.
    let sol = run(&config(SIGMA_Y, 0.0), &uniform_path(1));
    let rise = plastic_temperature_rise(&sol, &heating());
    assert_eq!(rise.len(), sol.field.element_stress.len());
}

// ---------------------------------------------------------------------------
// 7. The deposit's ledger must be the one `CoupledField::diffuse` conserves
// ---------------------------------------------------------------------------
//
// ⚠️ `CoupledField` is **node centred**: nodes sit on `min` and `max` and the
// spacing is `(max − min) / (n − 1)`, so the material the field represents is
// exactly `[min, max]` and a node on a face owns **half** a cell, an edge node a
// quarter, a corner node an eighth. The finite-volume dual weight of a node is
// therefore `2^−b` with `b` the number of axes on which it sits at an end.
//
// That weight is not a convention this crate is free to pick. `diffuse` uses a
// seven-point stencil with a **mirror** ghost node (`T₋₁ = T₁`, zero gradient at
// the node), and such a scheme conserves `Σ 2^−b T` exactly — measured below,
// and measured to drift by 0 over six steps while the plain sum `Σ T` moves by
// −12.7 % (heat on a face node) and +22.0 % (heat in the interior).
//
// A deposit whose own ledger is the plain sum therefore disagrees with the very
// operator the field is going to be stepped with, and a body touching the grid
// boundary loses up to a factor of eight. These oracles pin the deposit to the
// dual weight instead.
//
// ⚠️ A degenerate axis (`n == 1`) has both ends on the same node, so counting it
// in `b` would halve the node twice over. Its dual length is one whole cell and
// it carries no flux, so **an axis with `n == 1` is not counted**. The deposit
// and the ledger below have to agree on that or the oracle fails for a reason
// that has nothing to do with the physics.

/// Finite-volume dual weight of a node: `2^−b`, `b` = number of axes with more
/// than one node on which this node sits at an end.
///
/// Written from the definition of a node-centred grid, not from the deposit.
fn dual_weight(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize, nz: usize) -> f64 {
    let mut b = 0u32;
    for (i, n) in [(ix, nx), (iy, ny), (iz, nz)] {
        if n > 1 && (i == 0 || i == n - 1) {
            b += 1;
        }
    }
    0.5f64.powi(b as i32)
}

/// `Σ 2^−b · T`, the quantity `diffuse` conserves.
fn lumped_sum(f: &CoupledField) -> f64 {
    let (nx, ny, nz) = (f.nx(), f.ny(), f.nz());
    let mut total = 0.0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                total += f.get(ix, iy, iz).to_f64() * dual_weight(ix, iy, iz, nx, ny, nz);
            }
        }
    }
    total
}

/// The grid is exactly the bar: `[0,4] × [0,2] × [0,2]` at a cell size of 1, so
/// every surface node of the bar **is** a boundary node of the grid.
///
/// ⚠️ The interior grid used by the oracles above keeps a one-cell margin, so it
/// never puts anything on a boundary node and cannot see this at all.
fn flush_grid() -> CoupledField {
    CoupledField::try_new(
        5,
        3,
        3,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(4),
            Fix128::from_int(2),
            Fix128::from_int(2),
        ),
    )
    .expect("valid grid")
}

/// `Σ_e β · W_p,e · V_e`, the heat the solution says was released.
fn released_heat(mesh: &SdfTetMesh, sol: &ElastoplasticSolution, h: &PlasticHeating) -> f64 {
    sol.dissipation
        .iter()
        .enumerate()
        .map(|(e, &w)| h.taylor_quinney().to_f64() * w.to_f64() * tet_volume(mesh, e))
        .sum()
}

fn yielded_solution() -> (SdfTetMesh, ElastoplasticSolution) {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let sol = solve_elastoplastic(
        &mesh,
        &material(),
        &bc,
        &config(SIGMA_Y, H_PLASTIC),
        &uniform_path(4),
    )
    .expect("solve succeeds");
    (mesh, sol)
}

#[test]
fn a_body_flush_with_the_grid_deposits_its_whole_heat_into_the_dual_ledger() {
    let (mesh, sol) = yielded_solution();
    let h = heating();
    let mut field = flush_grid();
    deposit_plastic_heat(&mesh, &sol, &h, &mut field).expect("deposit succeeds");

    let (cx, cy, cz) = field.cell_size();
    let cell_volume = (cx * cy * cz).to_f64();
    let on_grid =
        lumped_sum(&field) * cell_volume * h.volumetric_heat_capacity_mpa_per_k().to_f64();
    let released = released_heat(&mesh, &sol, &h);

    assert!(released > 0.0, "the scene must dissipate something");
    let relative = (on_grid - released).abs() / released;
    assert!(
        relative < 1e-12,
        "the dual ledger must hold the whole released heat: ledger {on_grid:e} vs released \
         {released:e} (relative {relative:e}) — a deposit that writes the plain sum loses the \
         half, quarter and eighth cells on the boundary"
    );
}

#[test]
fn the_deposited_heat_survives_diffusion_in_the_dual_ledger() {
    let (mesh, sol) = yielded_solution();
    let h = heating();
    let mut field = flush_grid();
    deposit_plastic_heat(&mesh, &sol, &h, &mut field).expect("deposit succeeds");

    let before_dual = lumped_sum(&field);
    let before_plain = field.sum().to_f64();
    // Six explicit steps, well inside the stability limit for h = 1.
    for _ in 0..6 {
        field.diffuse(Fix128::from_ratio(1, 16), Fix128::ONE);
    }
    let after_dual = lumped_sum(&field);
    let after_plain = field.sum().to_f64();

    let drift = (after_dual - before_dual).abs() / before_dual;
    assert!(
        drift < 1e-15,
        "diffuse must conserve the dual ledger: {before_dual:e} -> {after_dual:e} \
         (relative {drift:e})"
    );
    // ⚠️ The tooth. If the plain sum did not move, the scene is not touching the
    // boundary and neither of these tests is exercising the correction.
    let plain_move = (after_plain - before_plain).abs() / before_plain;
    assert!(
        plain_move > 1e-3,
        "the plain sum must move under diffusion, or this scene never reaches a boundary \
         node: {before_plain:e} -> {after_plain:e} (relative {plain_move:e})"
    );
}

#[test]
fn a_degenerate_axis_is_refused_rather_than_silently_rescaled() {
    // ⚠️ A single node on an axis is refused, not weighted. `CoupledField` gives
    // a degenerate axis a stored cell size of one, so `V_cell` would make the
    // ledger "per unit thickness" while the element volumes it is balanced
    // against are the real three-dimensional ones.
    //
    // ⚠️ The balance would still *close* — both sides use the same `V_cell` — so
    // a conservation oracle cannot see this. What comes out wrong is the
    // temperature, scaled by the body's true thickness on that axis (here a
    // factor of two for a 2 mm bar on a grid that can only say 1 mm). That is
    // why this asserts the refusal rather than the balance.
    let (mesh, sol) = yielded_solution();
    let h = heating();
    for (nx, ny, nz, axis) in [
        (1usize, 3usize, 3usize, Axis::X),
        (5, 1, 3, Axis::Y),
        (5, 3, 1, Axis::Z),
    ] {
        let mut field = CoupledField::try_new(
            nx,
            ny,
            nz,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (
                Fix128::from_int(4),
                Fix128::from_int(2),
                Fix128::from_int(2),
            ),
        )
        .expect("valid grid");
        let got = deposit_plastic_heat(&mesh, &sol, &h, &mut field);
        assert_eq!(
            got,
            Err(FemError::DepositGridHasDegenerateAxis { axis }),
            "a single node on {axis:?} must be refused, got {got:?}"
        );
        assert_eq!(
            field.sum(),
            Fix128::ZERO,
            "a refused deposit must not have written anything"
        );
    }
}

#[test]
fn an_existing_field_is_added_to_and_not_rescaled() {
    // ⚠️ The dual-weight correction applies to what this call deposits, never to
    // what the caller already had. Without the staging buffer the boundary
    // multiply would scale the pre-existing content too, so a caller
    // accumulating two bodies would see the first one doubled at the boundary.
    let (mesh, sol) = yielded_solution();
    let h = heating();

    let mut once = flush_grid();
    deposit_plastic_heat(&mesh, &sol, &h, &mut once).expect("deposit succeeds");

    let mut twice = flush_grid();
    deposit_plastic_heat(&mesh, &sol, &h, &mut twice).expect("deposit succeeds");
    deposit_plastic_heat(&mesh, &sol, &h, &mut twice).expect("second deposit succeeds");

    let (nx, ny, nz) = (once.nx(), once.ny(), once.nz());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let single = once.get(ix, iy, iz);
                let double = twice.get(ix, iy, iz);
                assert_eq!(
                    double,
                    single + single,
                    "node ({ix},{iy},{iz}): two deposits must be exactly twice one"
                );
            }
        }
    }
}

#[test]
fn a_non_uniform_cell_still_balances() {
    // Different spacing on each axis: the dual weight is a product over axes, so
    // it does not depend on the spacings being equal, but the `V_cell` the
    // ledger uses does.
    let (mesh, sol) = yielded_solution();
    let h = heating();
    // Bar is 4 x 2 x 2; nodes at x 0,1,2,3,4 / y 0,1,2 / z 0,2 -> hz = 2.
    let mut field = CoupledField::try_new(
        5,
        3,
        2,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(4),
            Fix128::from_int(2),
            Fix128::from_int(2),
        ),
    )
    .expect("valid grid");
    deposit_plastic_heat(&mesh, &sol, &h, &mut field).expect("deposit succeeds");
    let (cx, cy, cz) = field.cell_size();
    assert_eq!(cz, Fix128::from_int(2), "the z spacing must really differ");
    let on_grid = lumped_sum(&field)
        * (cx * cy * cz).to_f64()
        * h.volumetric_heat_capacity_mpa_per_k().to_f64();
    let released = released_heat(&mesh, &sol, &h);
    let relative = (on_grid - released).abs() / released;
    assert!(
        relative < 1e-12,
        "a non-uniform cell must still balance: ledger {on_grid:e} vs released {released:e} \
         (relative {relative:e})"
    );
}

#[test]
fn an_unstable_step_still_conserves_the_dual_ledger() {
    // Conservation is independent of stability: the dual ledger is exact even
    // when the explicit step is far past its limit and the field blows up.
    let (mesh, sol) = yielded_solution();
    let mut field = flush_grid();
    deposit_plastic_heat(&mesh, &sol, &heating(), &mut field).expect("deposit succeeds");
    let before = lumped_sum(&field);
    for _ in 0..4 {
        field.diffuse(Fix128::from_int(2), Fix128::ONE); // dt = 2, wildly unstable
    }
    let after = lumped_sum(&field);
    let drift = (after - before).abs() / before.abs();
    assert!(
        drift < 1e-12,
        "the ledger must not depend on stability: {before:e} -> {after:e} (relative {drift:e})"
    );
}

#[test]
fn an_interior_body_is_unaffected_by_the_boundary_correction() {
    // ⚠️ Regression evidence: when nothing lands on a boundary node the dual
    // weight is 1 everywhere that matters, so the plain-sum identity of
    // `depositing_the_heat_conserves_the_total_energy` must still hold exactly
    // and the two ledgers must agree.
    let (mesh, sol) = yielded_solution();
    let h = heating();
    let mut field = grid(); // the one-cell-margin grid
    deposit_plastic_heat(&mesh, &sol, &h, &mut field).expect("deposit succeeds");
    let plain = field.sum().to_f64();
    let dual = lumped_sum(&field);
    assert!(
        (plain - dual).abs() / plain < 1e-15,
        "with a margin the two ledgers must coincide: plain {plain:e} vs dual {dual:e}"
    );
}
