//! Which temperature the thermal eigenstrain reads, when two subsystems own one.
//!
//! # The question this file answers
//!
//! `thermal::ThermalModifier` and `phase_change::PhaseChangeModifier` each own
//! a `sim_field::ScalarField3D` called `temperature` over the same region, both
//! claim the same channel name `"temperature"` through
//! `coupled_field::CoupledScalar`, and `sim_modifier::ModifiedSdf::update`
//! never relates them — that absence is pinned by
//! `tests/coupling_channel_inventory.rs`. Separately,
//! `linear_elastic_fem::ThermalExpansion` drives an eigenstrain from a
//! temperature field, so a scene carrying both owners raises the question of
//! **which of the two the eigenstrain is loaded from**.
//!
//! The answer is neither, and it is structural rather than conventional:
//! `ThermalExpansion` holds a `&CoupledField`, so it cannot reach either
//! owner's `ScalarField3D` at all. What reaches the residual is whatever the
//! caller put in the channel. This file pins that the composition
//!
//! ```text
//! two disagreeing owners → reconcile_mean → subtract the reference → ThermalExpansion
//! ```
//!
//! produces the stress of the *agreed rise*, and — the part an absence test
//! cannot give — that each of the four other things a caller could plausibly
//! route in produces a visibly different stress instead. Five closed forms that
//! are far apart are what make "which one was read" a question the test can
//! answer.
//!
//! # ⚠️⚠️ Reconciling is not the same as getting the unit right
//!
//! The two steps in that chain are independent, and only together are they
//! correct:
//!
//! - **`reconcile_mean` fixes the multiplicity.** Two owners become one agreed
//!   field, so a point stops having two temperatures.
//! - **Subtracting the reference fixes the unit.** `ThermalExpansion` is
//!   specified on `ΔT` above the stress-free reference, and its documentation
//!   states that the type cannot subtract the reference itself because the
//!   reference belongs to the configuration the mesh was built in. Both owners
//!   are filled with `config.ambient_temperature`, i.e. an **absolute**
//!   temperature, and `CoupledScalar::publish` copies values through verbatim.
//!
//! ⚠️ **The mean of two absolute temperatures is still an absolute
//! temperature**, so reconciling alone leaves the unit wrong and loads the
//! ambient offset as if it were a rise.
//! [`reconciling_does_not_fix_the_unit_only_subtracting_the_reference_does`]
//! measures that, so this file cannot be read as "reconcile and the eigenstrain
//! becomes correct". Every [`Source`] variant says in its own name whether the
//! reference was subtracted.
//!
//! ⚠️ Nothing in the crate stops a caller from skipping the subtraction: `ΔT`
//! and `T` are the same type on the same grid, and the owners' channel name
//! does not distinguish them. That gap is open on
//! `8ddb608` and is filed in the backlog rather than closed here — closing it
//! means giving the rise its own type, which is a design change.
//!
//! # Why this needed its own oracle
//!
//! `tests/analytic_thermoelastic.rs` drives the eigenstrain from a
//! `CoupledField` built by hand, so it never touches a temperature owner;
//! `tests/analytic_coupled_field.rs` reconciles the two owners but never
//! reaches the FEM, and mentions the eigenstrain nowhere. The junction between
//! them — the only place where "a point has two temperatures" can become a
//! wrong stress — had no coverage in either file, and no caller in `src/`
//! composes the two.
//!
//! # Scope
//!
//! Static, uniform temperatures and one isotropic material. Nothing here
//! exercises diffusion, phase transitions, latent heat, or the reverse
//! direction (deformation heating): the eigenstrain is the only coupling term
//! that exists, and `reconcile_mean` is the only thing that joins the owners.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::{reconcile_mean, CoupledField, CoupledScalar};
use alice_physics::linear_elastic_fem::{
    solve_with_eigenstrain, BoundaryConditions, ElasticMaterial, FemSolution, SolverConfig,
    StressTensor, ThermalExpansion,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

// ---------------------------------------------------------------------------
// material, scene and the temperatures in play
// ---------------------------------------------------------------------------

/// Young's modulus, MPa (PLA).
const E_MPA: f64 = 3500.0;
/// Poisson's ratio.
const NU: f64 = 0.35;
/// Side of the cube, mm, as the mesh and the closed forms use it.
const SIDE: f64 = 4.0;
/// The same side as the owners' grid bounds need it.
const SIDE_MM: f32 = 4.0;
/// Linear expansion coefficient, K⁻¹.
const ALPHA_PER_K: f64 = 1.0 / 1000.0;

/// The stress-free reference the mesh was built at, K.
///
/// Absolute, like the two owners' fields. Every rise below is a difference
/// against this, which is the quantity `ThermalExpansion` wants.
const REFERENCE_K: f32 = 25.0;
/// The absolute temperature the thermal owner holds, K.
const THERMAL_ABS_K: f32 = 65.0;
/// The absolute temperature the phase-change owner holds, K.
const PHASE_ABS_K: f32 = 105.0;

/// Field resolution of both owners. Uniform fields, so the value only has to
/// put the outer nodes on the cube's faces.
const RES: usize = 5;
/// Mesh refinements every scene is run at. Both closed forms lie in the P1
/// space exactly, so the answer must not move between them.
const REFINEMENTS: [usize; 2] = [2, 4];

/// Stress tolerance, MPa. Three orders of magnitude below the smallest gap
/// between the closed forms, which is what makes those gaps readable.
const SIGMA_TOL_MPA: f64 = 1e-2;
/// Displacement tolerance, mm.
const U_TOL_MM: f64 = 1e-9;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

/// 200,000 iterations at a relative residual of `2⁻³⁴`, matching
/// `tests/analytic_thermoelastic.rs` so the two files' numbers are comparable.
fn solver_config() -> SolverConfig {
    SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 30)).expect("valid linear config")
}

// ---------------------------------------------------------------------------
// closed forms, derived here and cross-checked by a second route
// ---------------------------------------------------------------------------

/// `λ` and `μ` from `E` and `ν`.
fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

/// `σ = −E α ΔT / (1 − 2ν)`, MPa — every normal component of the fully
/// suppressed stress.
///
/// oracle: the constitutive law is
/// `σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th)` with `ε_th = α ΔT · I`. Clamping
/// every boundary node of a body under a *uniform* eigenstrain admits `u ≡ 0`,
/// so `ε = 0` and `σ = −(3λ + 2μ) α ΔT · I`. Substituting
/// `3λ + 2μ = 3K = E/(1 − 2ν)` gives the form above, with no shear.
fn constrained_stress_mpa(delta_t_k: f64) -> f64 {
    -E_MPA * (ALPHA_PER_K * delta_t_k) / (1.0 - 2.0 * NU)
}

/// The same number through `−(3λ + 2μ) α ΔT`, which shares no factor with the
/// bulk-modulus route above.
fn constrained_stress_via_lame_mpa(delta_t_k: f64) -> f64 {
    let (lambda, mu) = lame();
    -(3.0 * lambda + 2.0 * mu) * (ALPHA_PER_K * delta_t_k)
}

/// Assert the two independent routes agree and return the value.
fn constrained_stress_checked(delta_t_k: f64) -> f64 {
    let want = constrained_stress_mpa(delta_t_k);
    let want_lame = constrained_stress_via_lame_mpa(delta_t_k);
    assert!(
        (want - want_lame).abs() < 1e-9,
        "the two closed forms must agree at ΔT = {delta_t_k} K: \
         −E α ΔT/(1 − 2ν) = {want} MPa, −(3λ + 2μ) α ΔT = {want_lame} MPa"
    );
    want
}

// ---------------------------------------------------------------------------
// the five fields the eigenstrain could be loaded from
// ---------------------------------------------------------------------------

/// Which field a caller routes into the channel that drives the eigenstrain.
///
/// ⚠️ Each name carries both decisions: **which owners** reached the channel,
/// and **whether the reference was subtracted** afterwards. The `*Rise`
/// variants are correct in unit; the `*Absolute` variants are the mistake
/// `ThermalExpansion`'s documentation warns about, and are present so that
/// "reconciled" cannot be read as "correct".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// `ThermalModifier` alone, reference subtracted.
    ThermalRise,
    /// `PhaseChangeModifier` alone, reference subtracted.
    PhaseRise,
    /// Both owners through `reconcile_mean`, reference subtracted. The only
    /// variant that is right about both multiplicity and unit.
    ReconciledRise,
    /// `ThermalModifier` alone, reference **not** subtracted.
    ThermalAbsolute,
    /// Both owners through `reconcile_mean`, reference **not** subtracted —
    /// one temperature, wrong unit.
    ReconciledAbsolute,
}

impl Source {
    /// The `ΔT` in K that this source presents to the eigenstrain.
    ///
    /// oracle: arithmetic on the scene's own constants, not on anything the
    /// crate computes. `reconcile_mean` is defined as the arithmetic mean, and
    /// the mean commutes with subtracting a constant reference, so the
    /// reconciled rise is both `mean(abs) − ref` and `mean(rise)`.
    fn delta_t_k(self) -> f64 {
        let thermal = f64::from(THERMAL_ABS_K);
        let phase = f64::from(PHASE_ABS_K);
        let reference = f64::from(REFERENCE_K);
        let mean = (thermal + phase) / 2.0;
        match self {
            Self::ThermalRise => thermal - reference,
            Self::PhaseRise => phase - reference,
            Self::ReconciledRise => mean - reference,
            Self::ThermalAbsolute => thermal,
            Self::ReconciledAbsolute => mean,
        }
    }

    /// Whether the caller subtracts the stress-free reference before handing
    /// the channel to `ThermalExpansion`.
    fn subtracts_reference(self) -> bool {
        !matches!(self, Self::ThermalAbsolute | Self::ReconciledAbsolute)
    }

    /// Whether both owners reached the channel through `reconcile_mean`.
    fn reconciles(self) -> bool {
        matches!(self, Self::ReconciledRise | Self::ReconciledAbsolute)
    }

    fn label(self) -> &'static str {
        match self {
            Self::ThermalRise => "thermal owner, rise",
            Self::PhaseRise => "phase owner, rise",
            Self::ReconciledRise => "reconciled, rise",
            Self::ThermalAbsolute => "thermal owner, absolute",
            Self::ReconciledAbsolute => "reconciled, absolute",
        }
    }
}

/// Every source, in one place so no test can quietly cover a subset.
const ALL_SOURCES: [Source; 5] = [
    Source::ThermalRise,
    Source::PhaseRise,
    Source::ReconciledRise,
    Source::ThermalAbsolute,
    Source::ReconciledAbsolute,
];

/// A fresh pair of owners, each holding a uniform absolute temperature over the
/// cube, disagreeing by construction.
///
/// Both are built on the same grid and left unstepped: the disagreement is the
/// scene, and stepping would only add decay toward two different ambients.
/// `tests/coupling_channel_inventory.rs` is where stepping-without-exchange is
/// pinned; here the gap only has to exist.
fn disagreeing_owners() -> (ThermalModifier, PhaseChangeModifier) {
    let min = (0.0_f32, 0.0_f32, 0.0_f32);
    let max = (SIDE_MM, SIDE_MM, SIDE_MM);
    let thermal = ThermalModifier::new(
        ThermalConfig {
            ambient_temperature: THERMAL_ABS_K,
            ..ThermalConfig::default()
        },
        RES,
        min,
        max,
    );
    let phase = PhaseChangeModifier::new(
        PhaseChangeConfig {
            ambient_temperature: PHASE_ABS_K,
            ..PhaseChangeConfig::default()
        },
        RES,
        min,
        max,
    );
    (thermal, phase)
}

/// Build the channel the eigenstrain will read, from `source`.
///
/// Returns the channel together with the two owners, so a caller can check what
/// `reconcile_mean` did to them. The two decisions are applied in order and
/// separately: reconcile (or not), then subtract the reference (or not).
fn channel_from(source: Source) -> (CoupledField, ThermalModifier, PhaseChangeModifier) {
    let (mut thermal, mut phase) = disagreeing_owners();
    let mut channel = thermal
        .coupled_channel()
        .expect("the channel matches the owner's own grid");

    if source.reconciles() {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut thermal, &mut phase];
        reconcile_mean(&mut participants, &mut channel).expect("both owners share a grid");
    } else if source == Source::PhaseRise {
        phase.publish(&mut channel).expect("same grid");
    } else {
        thermal.publish(&mut channel).expect("same grid");
    }

    // ⚠️ The reference subtraction the eigenstrain cannot do for itself, and
    // that reconcile_mean does not do either: the mean of two absolute
    // temperatures is still absolute.
    if source.subtracts_reference() {
        let reference = Fix128::from_f32(REFERENCE_K);
        for cell in channel.as_mut_slice() {
            *cell = *cell - reference;
        }
    }

    (channel, thermal, phase)
}

/// Assert the channel really is the uniform rise the closed form assumes, and
/// that it covers the mesh.
///
/// Guards the two ways a plausible-looking scene stops matching its closed
/// form: a field that is not actually uniform, and a mesh poking outside the
/// grid, where `CoupledField::sample` clamps onto the nearest boundary node
/// rather than refusing.
fn assert_channel_drives(channel: &CoupledField, mesh: &SdfTetMesh, want_delta_t_k: f64) {
    for p in [
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(fx(1.5), fx(2.5), fx(0.5)),
        Vec3Fix::new(fx(SIDE), fx(SIDE), fx(SIDE)),
    ] {
        let t = channel.sample(p).to_f64();
        assert!(
            (t - want_delta_t_k).abs() < 1e-9,
            "the channel must be the uniform value the closed form assumes; sampled \
             {t} K instead of {want_delta_t_k} K"
        );
    }
    for v in 0..vertex_count(mesh) {
        let x = vert(mesh, v);
        let p = Vec3Fix::new(fx(x[0]), fx(x[1]), fx(x[2]));
        assert!(
            channel.contains(p),
            "node {v} at {x:?} mm lies outside the channel, where sampling clamps onto \
             the boundary node; the closed form assumes the channel covers the mesh"
        );
    }
}

/// Solve the clamped cube with the eigenstrain driven by `source`.
///
/// ⚠️ **The one place the channel has to reach the residual.** A field that is
/// built, asserted and then not forwarded leaves every stress at zero while the
/// scene still looks right; [`every_source_actually_reaches_the_residual`] is
/// the control that rules that out by comparing against a `None` run.
fn solve_from(source: Source, n: usize) -> (FemSolution, f64) {
    let mesh = kuhn_cube(n, SIDE / n as f64);
    let bc = clamped_boundary(&mesh, n);
    let (channel, _thermal, _phase) = channel_from(source);
    let want_delta_t = source.delta_t_k();
    assert_channel_drives(&channel, &mesh, want_delta_t);

    let out = solve_with_eigenstrain(
        &mesh,
        &pla(),
        &bc,
        &solver_config(),
        Some(ThermalExpansion::new(&channel, fx(ALPHA_PER_K))),
    )
    .unwrap_or_else(|e| panic!("{} at n = {n}: {e:?}", source.label()));

    (out, want_delta_t)
}

// ---------------------------------------------------------------------------
// mesh and boundary conditions
// ---------------------------------------------------------------------------

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet cube on `[0, n·h]³`, conforming at every `n`.
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

fn vert(mesh: &SdfTetMesh, v: u32) -> [f64; 3] {
    let p = mesh.vertices[v as usize];
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn vertex_count(mesh: &SdfTetMesh) -> u32 {
    u32::try_from(mesh.vertices.len()).expect("lattice fits u32")
}

/// `u = 0` on every boundary node, with the interior count checked against the
/// `(n − 1)³` the lattice must have.
fn clamped_boundary(mesh: &SdfTetMesh, n: usize) -> BoundaryConditions {
    let eps = SIDE * 1e-9;
    let mut bc = BoundaryConditions::new();
    let mut interior = 0_usize;
    for v in 0..vertex_count(mesh) {
        let p = vert(mesh, v);
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            bc.prescribe_all(v, [Fix128::ZERO; 3]);
        } else {
            interior += 1;
        }
    }
    assert_eq!(
        interior,
        (n - 1) * (n - 1) * (n - 1),
        "n = {n}: an n-cell cube has (n−1)³ interior nodes"
    );
    bc
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

/// Worst deviation of any normal component from `want`, MPa.
fn worst_normal_deviation(stresses: &[StressTensor], want: f64) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz] {
            worst = worst.max((c.to_f64() - want).abs());
        }
    }
    worst
}

fn worst_shear(stresses: &[StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

fn worst_stress_component(stresses: &[StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz, s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

fn worst_displacement_mm(displacements: &[[Fix128; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for u in displacements {
        for c in u {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

/// Worst component-wise difference between two stress fields on the same mesh.
fn worst_stress_gap(a: &[StressTensor], b: &[StressTensor]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "the two runs must be on the same mesh to be compared element by element"
    );
    let mut worst = 0.0_f64;
    for (sa, sb) in a.iter().zip(b.iter()) {
        for (ca, cb) in [sa.xx, sa.yy, sa.zz, sa.xy, sa.yz, sa.zx]
            .into_iter()
            .zip([sb.xx, sb.yy, sb.zz, sb.xy, sb.yz, sb.zx])
        {
            worst = worst.max((ca.to_f64() - cb.to_f64()).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// 0. the scene is discriminating in the first place
// ---------------------------------------------------------------------------

/// The five sources present five `ΔT`s whose stresses are far apart.
///
/// Without this the rest of the file could pass while every source produced the
/// same answer, which is exactly the state "a point has two temperatures" would
/// be harmless in. The closed form is `−11.66̄ MPa/K · ΔT`, so the smallest gap
/// here is `5 K · 11.66̄ ≈ 58.3 MPa` against a `1e-2 MPa` tolerance.
#[test]
fn the_five_candidate_temperatures_give_well_separated_stresses() {
    // Closed forms spelled out, so a change to the scene constants has to be
    // acknowledged here rather than silently carried through.
    let expected: [(Source, f64, f64); 5] = [
        (Source::ThermalRise, 40.0, -1400.0 / 3.0),
        (Source::PhaseRise, 80.0, -2800.0 / 3.0),
        (Source::ReconciledRise, 60.0, -700.0),
        (Source::ThermalAbsolute, 65.0, -2275.0 / 3.0),
        (Source::ReconciledAbsolute, 85.0, -2975.0 / 3.0),
    ];
    for (source, want_delta_t, want_sigma) in expected {
        assert!(
            (source.delta_t_k() - want_delta_t).abs() < 1e-12,
            "{}: must present ΔT = {want_delta_t} K; it presents {} K",
            source.label(),
            source.delta_t_k()
        );
        let want = constrained_stress_checked(source.delta_t_k());
        assert!(
            (want - want_sigma).abs() < 1e-9,
            "{}: ΔT = {want_delta_t} K must give {want_sigma} MPa; the closed form says \
             {want} MPa",
            source.label()
        );
    }

    for (i, a) in ALL_SOURCES.iter().enumerate() {
        let sa = constrained_stress_checked(a.delta_t_k());
        assert!(
            sa.abs() > 1000.0 * SIGMA_TOL_MPA,
            "vacuity guard: {} expects {sa} MPa, which must be far above the \
             {SIGMA_TOL_MPA} MPa tolerance or a zero-stress solve would pass",
            a.label()
        );
        for b in ALL_SOURCES.iter().skip(i + 1) {
            let sb = constrained_stress_checked(b.delta_t_k());
            assert!(
                (sa - sb).abs() > 1000.0 * SIGMA_TOL_MPA,
                "{} ({sa} MPa) and {} ({sb} MPa) are {} MPa apart, which is not enough \
                 for the solve to tell them apart at a {SIGMA_TOL_MPA} MPa tolerance",
                a.label(),
                b.label(),
                (sa - sb).abs()
            );
        }
    }
}

/// `reconcile_mean` gives the eigenstrain one temperature to read, and leaves
/// that temperature absolute.
///
/// ⚠️ Both halves matter. The first is the multiplicity fix this file depends
/// on; the second is why the reconciled field still needs the reference
/// subtracted before it is a valid `ΔT`. Asserting only the first is what would
/// make this file read as "reconcile and the coupling becomes correct".
#[test]
fn reconcile_mean_gives_one_temperature_and_leaves_it_absolute() {
    let probe = (1.0_f32, 1.0_f32, 1.0_f32);

    let (thermal, phase) = disagreeing_owners();
    let before_t = thermal.temperature_at(probe.0, probe.1, probe.2);
    let before_p = phase.temperature_at(probe.0, probe.1, probe.2);
    assert!(
        (f64::from(before_t) - f64::from(before_p)).abs() > 1.0,
        "the two owners must disagree before reconciling, or the agreement below is \
         trivial; they read {before_t} K and {before_p} K"
    );

    let (channel, thermal, phase) = channel_from(Source::ReconciledAbsolute);
    let after_t = thermal.temperature_at(probe.0, probe.1, probe.2);
    let after_p = phase.temperature_at(probe.0, probe.1, probe.2);

    // (1) multiplicity: one temperature where there were two.
    assert_eq!(
        after_t.to_bits(),
        after_p.to_bits(),
        "after reconcile_mean the probe still reads two temperatures ({after_t} vs \
         {after_p}); the owners were not joined"
    );
    assert!(
        after_t > before_t.min(before_p) && after_t < before_t.max(before_p),
        "the agreed temperature {after_t} K is not between the two inputs ({before_t}, \
         {before_p}); one field was copied over the other rather than reconciled"
    );

    // (2) unit: it is the mean of the absolutes, so it is still absolute.
    // Checked against the scene constants, not against anything the crate
    // computed.
    let want_abs = Source::ReconciledAbsolute.delta_t_k();
    assert!(
        (f64::from(after_t) - want_abs).abs() < 1e-6,
        "the agreed temperature must be {want_abs} K, the mean of {THERMAL_ABS_K} and \
         {PHASE_ABS_K}; the owners read {after_t} K"
    );
    let in_channel = channel
        .sample(Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)))
        .to_f64();
    assert!(
        (in_channel - want_abs).abs() < 1e-6,
        "the reconciled channel must carry the absolute mean {want_abs} K; it carries \
         {in_channel} K"
    );
    // ⚠️ And it is therefore *not* the rise: the channel sits a whole reference
    // offset above the ΔT the eigenstrain is specified on. This is the half
    // that stops "reconciled" from being read as "correct".
    let want_rise = Source::ReconciledRise.delta_t_k();
    assert!(
        (in_channel - want_rise - f64::from(REFERENCE_K)).abs() < 1e-6,
        "the reconciled channel reads {in_channel} K, which is not the agreed rise \
         {want_rise} K plus the {REFERENCE_K} K reference; reconcile_mean is not \
         supposed to change the unit"
    );
    assert!(
        (in_channel - want_rise).abs() > 1000.0 * SIGMA_TOL_MPA,
        "the reconciled channel {in_channel} K and the agreed rise {want_rise} K are \
         indistinguishable, so this test cannot tell the two units apart"
    );
}

// ---------------------------------------------------------------------------
// 1. the channel reaches the residual at all
// ---------------------------------------------------------------------------

/// Every source changes the answer, and a `None` eigenstrain leaves it at zero.
///
/// ⚠️ The failure this rules out is the one the previous iteration of the wall-3
/// oracle actually had: a temperature field that is constructed, asserted
/// against the scene, and then never forwarded to the solve. That leaves every
/// stress at zero, which an `== want` assert catches but a "the two agree"
/// assert does not. Here the zero case is measured explicitly and each source
/// is required to move away from it by about its own closed form.
#[test]
fn every_source_actually_reaches_the_residual() {
    let n = REFINEMENTS[0];
    let mesh = kuhn_cube(n, SIDE / n as f64);
    let bc = clamped_boundary(&mesh, n);

    let unloaded = solve_with_eigenstrain(&mesh, &pla(), &bc, &solver_config(), None)
        .expect("the unloaded clamped cube solves");
    let unloaded_sigma = worst_stress_component(&unloaded.element_stress);
    assert!(
        unloaded_sigma < SIGMA_TOL_MPA,
        "a clamped cube with no eigenstrain must be stress free; worst component is \
         {unloaded_sigma:.6e} MPa"
    );

    for source in ALL_SOURCES {
        let (out, delta_t) = solve_from(source, n);
        let want = constrained_stress_checked(delta_t);
        let gap = worst_stress_gap(&out.element_stress, &unloaded.element_stress);
        eprintln!(
            "  {:<26} ΔT = {delta_t:>5.1} K, σ = {:>12.6} MPa, gap vs None = {gap:.6e} MPa",
            source.label(),
            out.element_stress[0].xx.to_f64()
        );
        assert!(
            gap > want.abs() / 2.0,
            "{}: the loaded and unloaded solves differ by only {gap:.6e} MPa, so the \
             channel did not reach the residual (expected about {} MPa)",
            source.label(),
            want.abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. which temperature was read — the question this file exists for
// ---------------------------------------------------------------------------

/// The stress is the closed form of the field that was routed in, and of no
/// other candidate.
///
/// Read as a whole this is the answer to "which of the two temperatures does
/// the eigenstrain use": whichever one reached the channel. Routing the thermal
/// owner in gives the thermal owner's stress and *not* the phase owner's, the
/// reconciled stress, or either unsubtracted absolute — each of which is
/// checked by name, so a failure says which field was read instead.
#[test]
fn the_eigenstrain_reads_exactly_the_field_that_reached_the_channel() {
    for source in ALL_SOURCES {
        let want = constrained_stress_checked(source.delta_t_k());
        for n in REFINEMENTS {
            let (out, delta_t) = solve_from(source, n);
            assert!(
                (delta_t - source.delta_t_k()).abs() < 1e-12,
                "the scene's ΔT drifted from the source's own definition"
            );

            let worst_u = worst_displacement_mm(&out.displacements);
            let worst_normal = worst_normal_deviation(&out.element_stress, want);
            let worst_shear_c = worst_shear(&out.element_stress);
            eprintln!(
                "  {:<26} n = {n}: worst |u| = {worst_u:.6e} mm, \
                 worst |σ_nn − {want:.6}| = {worst_normal:.6e} MPa, \
                 worst |σ_shear| = {worst_shear_c:.6e} MPa",
                source.label()
            );

            assert!(
                worst_u < U_TOL_MM,
                "{} at n = {n}: a uniform eigenstrain on a fully clamped body loads no \
                 interior degree of freedom, so the displacement must be zero; worst \
                 node is {worst_u:.6e} mm",
                source.label()
            );
            assert!(
                worst_shear_c < SIGMA_TOL_MPA,
                "{} at n = {n}: an isotropic eigenstrain produces no shear; worst \
                 component is {worst_shear_c:.6e} MPa",
                source.label()
            );
            assert!(
                worst_normal < SIGMA_TOL_MPA,
                "{} at n = {n}: every normal stress must be {want} MPa; worst element is \
                 {worst_normal:.6e} MPa away",
                source.label()
            );

            // ⚠️ And it is not any of the other candidates. This is the assert
            // that fails when the wrong temperature is read.
            for other in ALL_SOURCES {
                if other == source {
                    continue;
                }
                let wrong = constrained_stress_checked(other.delta_t_k());
                let deviation = worst_normal_deviation(&out.element_stress, wrong);
                assert!(
                    deviation > SIGMA_TOL_MPA,
                    "{} at n = {n}: the stress matches {} ({wrong} MPa) to within \
                     {deviation:.6e} MPa, so the eigenstrain read the wrong field",
                    source.label(),
                    other.label()
                );
            }
        }
    }
}

/// Routing the two owners in separately gives two different stresses, and the
/// reconciled rise gives the mean of them.
///
/// The linearity is not decoration: it is what makes `reconcile_mean`'s
/// "arithmetic mean of the fields" equal to "the mean of the two answers", so a
/// caller can reason about the agreed temperature without re-solving. It also
/// pins the direction of the only remaining ambiguity — the reconciled answer
/// lies strictly between the two owners' answers, so neither owner was
/// silently preferred.
#[test]
fn the_reconciled_rise_gives_the_mean_of_the_two_owners_stresses() {
    let n = REFINEMENTS[1];
    let sigma = |source: Source| -> f64 {
        let (out, _) = solve_from(source, n);
        out.element_stress[0].xx.to_f64()
    };

    let s_thermal = sigma(Source::ThermalRise);
    let s_phase = sigma(Source::PhaseRise);
    let s_both = sigma(Source::ReconciledRise);
    let mean = (s_thermal + s_phase) / 2.0;
    eprintln!(
        "  σ(thermal rise) = {s_thermal:.6}, σ(phase rise) = {s_phase:.6}, \
         σ(reconciled rise) = {s_both:.6}, mean = {mean:.6} MPa"
    );

    assert!(
        (s_thermal - s_phase).abs() > 1000.0 * SIGMA_TOL_MPA,
        "the two owners must give visibly different stresses ({s_thermal} vs {s_phase} \
         MPa) or the mean below is trivial"
    );
    assert!(
        (s_both - mean).abs() < SIGMA_TOL_MPA,
        "the reconciled stress {s_both} MPa must be the mean {mean} MPa of the two \
         owners' stresses; it is {} MPa away",
        (s_both - mean).abs()
    );
    assert!(
        s_both > s_phase && s_both < s_thermal,
        "the reconciled stress {s_both} MPa must lie strictly between the phase owner's \
         {s_phase} MPa and the thermal owner's {s_thermal} MPa; one owner was preferred"
    );
}

// ---------------------------------------------------------------------------
// 3. the unit the channel carries — reconciling does not supply it
// ---------------------------------------------------------------------------

/// ⚠️⚠️ Agreeing on a temperature does not make it the right quantity.
///
/// `reconcile_mean` averages two absolute fields and produces an absolute
/// field, so the reconciled channel is wrong by the whole reference offset
/// until the caller subtracts it. The two decisions are independent, and this
/// test measures that independence in stress:
///
/// - reconciled **absolute** differs from reconciled **rise** by the stress of
///   the reference, `E α T_ref / (1 − 2ν)`;
/// - the same offset appears on the single-owner route, so it is a property of
///   the subtraction and not of reconciling.
///
/// Nothing in the crate can catch the omission: `ΔT` and `T` are the same type
/// on the same grid, and both owners publish under the same channel name
/// `"temperature"`. The cost is recorded here so the requirement in
/// `ThermalExpansion`'s documentation has a measured number attached.
#[test]
fn reconciling_does_not_fix_the_unit_only_subtracting_the_reference_does() {
    let n = REFINEMENTS[0];
    // oracle: skipping the subtraction adds the stress of the reference
    // temperature itself, whichever route reached the channel.
    let want_error = constrained_stress_checked(f64::from(REFERENCE_K));

    for (rise, absolute) in [
        (Source::ThermalRise, Source::ThermalAbsolute),
        (Source::ReconciledRise, Source::ReconciledAbsolute),
    ] {
        assert!(
            rise.reconciles() == absolute.reconciles(),
            "the pair must differ only in the subtraction, not in the routing"
        );
        let correct = constrained_stress_checked(rise.delta_t_k());
        let forgotten = constrained_stress_checked(absolute.delta_t_k());
        assert!(
            ((forgotten - correct) - want_error).abs() < 1e-9,
            "{} vs {}: the error from skipping the subtraction must be the stress of \
             the reference temperature, {want_error} MPa; the closed forms differ by \
             {} MPa",
            absolute.label(),
            rise.label(),
            forgotten - correct
        );

        let (out_correct, _) = solve_from(rise, n);
        let (out_forgotten, _) = solve_from(absolute, n);
        let measured = worst_stress_gap(&out_correct.element_stress, &out_forgotten.element_stress);
        eprintln!(
            "  {:<26} σ(rise) = {correct:>12.6}, σ(absolute) = {forgotten:>12.6}, \
             measured gap = {measured:.6} MPa, closed form = {:.6} MPa",
            rise.label(),
            want_error.abs()
        );
        assert!(
            (measured - want_error.abs()).abs() < SIGMA_TOL_MPA,
            "{}: the measured gap {measured:.6} MPa must match the closed-form \
             {:.6} MPa",
            rise.label(),
            want_error.abs()
        );
        assert!(
            measured / correct.abs() > 0.3,
            "vacuity guard: the error is {:.1}% of the correct stress, which is too \
             small for this test to be saying anything",
            100.0 * measured / correct.abs()
        );
    }
}
