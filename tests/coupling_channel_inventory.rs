//! What the coupling channels between the physics subsystems actually carry.
//!
//! The crate advertises several physics that run over the same region of space
//! (thermal, pressure, fracture, phase change, erosion, and the Fix128 CFD
//! solver). This file measures what they can and cannot hand to each other.
//!
//! ⚠️ **Measured: the only channel between the `PhysicsModifier` implementors
//! is the scalar SDF distance, and there is no channel at all between them and
//! the `Fix128` core.** `update` receives `dt` and nothing else, so a modifier
//! has no argument through which another modifier's field could arrive, and
//! `modify_distance` takes `&self`, so nothing can be written back. The
//! composition in `ModifiedSdf::eval_distance` folds one `f32` through the
//! chain. That is a one-way composition of geometry, weaker than a staggered
//! (weak) coupling.
//!
//! ⚠️ **Measured: "temperature" exists three times over, in two number
//! systems.** `thermal::ThermalModifier.temperature` and
//! `phase_change::PhaseChangeModifier.temperature` are both `ScalarField3D`
//! (`f32`); `cfd_solver::CfdSolver.temperature` is an `Option<Grid3d>`
//! (`Fix128`). No file under `src/` names both field types, so no code path
//! can relate them.
//!
//! # Why these tests are not vacuous
//!
//! Two unrelated structs trivially do not affect each other, so "they stayed
//! equal" on its own measures nothing. Every null result below is asserted as a
//! **conjunction of three facts in the same run**:
//!
//! 1. the perturbation is real — the subsystem it was applied to **diverges**
//!    from the unperturbed run,
//! 2. the observed subsystem is **live** — its state moves away from its own
//!    initial condition while the run proceeds,
//! 3. and yet the observed subsystem is **bit-identical** across the two runs.
//!
//! (1) rules out "the perturbation did nothing", (2) rules out "we compared two
//! frozen arrays", and only then does (3) mean "there is no return path". Drop
//! any one of the three and the test stops carrying its claim; each is asserted
//! explicitly rather than assumed.
//!
//! # What these tests are for
//!
//! They fix the present architecture so that **wiring a coupling channel makes
//! them fail**. A red here is not a regression; it is the signal that the
//! premise of the strong-coupling analysis has changed and the analysis has to
//! be redone.
//!
//! Author: Moroya Sakamoto

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::Fix128;
use alice_physics::multiphase::Grid3d;
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sim_modifier::ModifiedSdf;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

// ============================================================================
// Shared scene construction
// ============================================================================

const RES: usize = 6;
const MIN: (f32, f32, f32) = (-2.0, -2.0, -2.0);
const MAX: (f32, f32, f32) = (2.0, 2.0, 2.0);
const DT: f32 = 0.016;
const STEPS: usize = 40;

/// Unit sphere at the origin, the geometry every modifier chain rides on.
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / l, y / l, z / l)
        },
    )
}

/// A thermal modifier with a heat source, so its field evolves under `update`.
fn heated_thermal() -> ThermalModifier {
    let mut t = ThermalModifier::new(ThermalConfig::default(), RES, MIN, MAX);
    t.add_heat_point(0.3, 0.0, 0.0, 400.0, 1.0);
    t
}

/// A CFD solver over the same region, optionally carrying a temperature grid.
fn cfd(with_temperature: Option<Fix128>) -> CfdSolver {
    let dx = Fix128::from_ratio(1, 10);
    let mut s = CfdSolver::new(4, 4, 4, dx);
    if let Some(t) = with_temperature {
        s.temperature = Some(Grid3d::new(4, 4, 4, dx, t));
    }
    s
}

/// Bit pattern of an `f32` field, for exact comparison.
fn bits(field: &[f32]) -> Vec<u32> {
    field.iter().map(|v| v.to_bits()).collect()
}

/// Every MAC-grid velocity component plus the temperature grid, as raw words.
fn cfd_state(s: &CfdSolver) -> Vec<(i64, u64)> {
    let mut out = Vec::new();
    for v in s
        .grid
        .u
        .iter()
        .chain(s.grid.v.iter())
        .chain(s.grid.w.iter())
        .chain(s.grid.pressure.iter())
    {
        out.push((v.hi, v.lo));
    }
    if let Some(t) = s.temperature.as_ref() {
        for v in &t.data {
            out.push((v.hi, v.lo));
        }
    }
    out
}

// ============================================================================
// 1. The channel surface itself
// ============================================================================

/// `PhysicsModifier` has no method through which one physics could hand a
/// field to another, and no method that could write one back.
///
/// This is a guard on the premise, not on behaviour: the behavioural tests
/// below only mean "there is no coupling" because the type surface admits
/// none. If a method is added that takes or returns simulation state, this
/// fails and the behavioural conclusions have to be re-derived.
#[test]
fn physics_modifier_offers_no_channel_for_another_physics_state() {
    let src = include_str!("../src/sim_modifier.rs");
    let start = src
        .find("pub trait PhysicsModifier")
        .expect("PhysicsModifier trait declaration");
    let body = &src[start..];
    let end = body.find("\n}\n").expect("end of trait block");
    let body = &body[..end];

    // Collect the declared method signatures, comments stripped.
    let methods: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("fn "))
        .collect();

    assert_eq!(
        methods,
        vec![
            "fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32;",
            "fn update(&mut self, dt: f32);",
            "fn name(&self) -> &str;",
            "fn is_active(&self) -> bool {",
        ],
        "the PhysicsModifier surface changed; re-derive what the coupling \
         channel can carry before updating this list"
    );

    // The two consequences the analysis rests on, stated separately so a
    // failure says which one broke.
    assert!(
        methods[1] == "fn update(&mut self, dt: f32);",
        "`update` is the only mutating entry point; its sole argument is `dt`, \
         so no modifier can be handed another modifier's field"
    );
    assert!(
        methods[0].starts_with("fn modify_distance(&self,"),
        "`modify_distance` takes `&self`, so a modifier cannot write back into \
         its own field while composing, let alone into another's"
    );

    // Nothing in the surface mentions a field or grid type.
    for m in &methods {
        for banned in ["Field", "Grid", "Modifier", "Fix128"] {
            assert!(
                !m.contains(banned),
                "PhysicsModifier::{m} mentions `{banned}`: a state channel may \
                 have been opened"
            );
        }
    }
}

/// No module under `src/` names both `ScalarField3D` (the `f32` field the
/// modifiers own) and `Grid3d` (the `Fix128` grid the CFD solver owns), so no
/// function can hold one of each and relate them.
#[test]
fn no_module_names_both_field_types() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut both: Vec<String> = Vec::new();
    let mut scalar_field = 0usize;
    let mut grid3d = 0usize;

    for entry in std::fs::read_dir(root).expect("src/ is readable") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let body = std::fs::read_to_string(&path).expect("readable source file");
        let has_scalar = body.contains("ScalarField3D");
        let has_grid = body.contains("Grid3d");
        if has_scalar {
            scalar_field += 1;
        }
        if has_grid {
            grid3d += 1;
        }
        if has_scalar && has_grid {
            both.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }

    // The two populations must both be non-empty, or "no overlap" is vacuous.
    assert!(
        scalar_field >= 5 && grid3d >= 3,
        "expected both field types to be in real use (ScalarField3D in \
         {scalar_field} modules, Grid3d in {grid3d}); the disjointness below \
         says nothing if one of them is unused"
    );
    assert!(
        both.is_empty(),
        "these modules now name both field types: {both:?} — a conversion or a \
         coupling path may exist; re-derive the layer analysis"
    );
}

// ============================================================================
// 2. The Fix128 CFD temperature and the f32 thermal temperature are disjoint
// ============================================================================

/// Perturbing the fluid temperature moves the fluid and leaves the thermal
/// modifier bit-identical.
#[test]
fn fluid_temperature_moves_the_fluid_and_not_the_thermal_field() {
    let t0 = Fix128::from_int(293);
    let run = |hot: bool| {
        let mut solver = cfd(Some(t0));
        if hot {
            // One cell far from the reference temperature. Boussinesq buoyancy
            // reads exactly this deviation, so the perturbation is one the
            // fluid is defined to respond to.
            if let Some(g) = solver.temperature.as_mut() {
                let i = g.idx(2, 2, 2);
                g.data[i] = Fix128::from_int(1_500);
            }
        }
        let mut thermal = heated_thermal();
        let initial = bits(&thermal.temperature.data);

        let dt_fix = Fix128::from_ratio(16, 1000);
        for _ in 0..STEPS {
            solver.step(dt_fix);
            alice_physics::sim_modifier::PhysicsModifier::update(&mut thermal, DT);
        }
        (
            cfd_state(&solver),
            bits(&thermal.temperature.data),
            bits(&thermal.melt_accumulator.data),
            initial,
        )
    };

    let (cold_fluid, cold_temp, cold_melt, initial) = run(false);
    let (hot_fluid, hot_temp, hot_melt, _) = run(true);

    // (1) the perturbation is real: the fluid diverged.
    let fluid_diff = cold_fluid
        .iter()
        .zip(&hot_fluid)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        fluid_diff > 0,
        "the 1500 K cell did not move the fluid at all; the perturbation is \
         not one this scene responds to, so the null result below measures \
         nothing"
    );

    // (2) the observed subsystem is live: the thermal field left its initial
    //     state, so we are not comparing two frozen arrays.
    let thermal_moved = initial
        .iter()
        .zip(&cold_temp)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        thermal_moved > 0,
        "the thermal field never moved during the run ({} cells); a bit-exact \
         match afterwards would be trivial",
        cold_temp.len()
    );

    // (3) and yet nothing of the fluid reached it.
    assert_eq!(
        cold_temp, hot_temp,
        "the thermal temperature field responded to the fluid temperature; a \
         channel between Grid3d (Fix128) and ScalarField3D (f32) now exists"
    );
    assert_eq!(
        cold_melt, hot_melt,
        "the accumulated melt responded to the fluid temperature; a channel now \
         exists"
    );
}

/// The mirror image: perturbing the thermal field moves the SDF it modifies
/// and leaves the fluid bit-identical.
#[test]
fn thermal_field_moves_the_sdf_and_not_the_fluid() {
    let t0 = Fix128::from_int(293);
    let probe = (0.3_f32, 0.0_f32, 0.0_f32);

    let run = |hot: bool| {
        let mut solver = cfd(Some(t0));
        let mut thermal = heated_thermal();
        if hot {
            thermal.apply_heat_at(probe.0, probe.1, probe.2, 5_000.0, 1.5);
        }
        let mut sdf = ModifiedSdf::new(Box::new(unit_sphere()));
        sdf.add_modifier(Box::new(thermal));

        let dt_fix = Fix128::from_ratio(16, 1000);
        let mut fluid_initial = cfd_state(&solver);
        let mut distances = Vec::new();
        for _ in 0..STEPS {
            solver.step(dt_fix);
            sdf.update(DT);
            distances.push(sdf.distance(probe.0, probe.1, probe.2).to_bits());
        }
        fluid_initial.truncate(fluid_initial.len());
        (cfd_state(&solver), distances, fluid_initial)
    };

    let (cold_fluid, cold_dist, fluid_initial) = run(false);
    let (hot_fluid, hot_dist, _) = run(true);

    // (1) the perturbation is real: the SDF the thermal modifier drives moved.
    assert_ne!(
        cold_dist, hot_dist,
        "5000 units of heat did not change the modified SDF distance at the \
         probe; the perturbation is invisible even to its own subsystem, so \
         the null result below measures nothing"
    );

    // (2) the fluid is live: gravity has moved it away from rest.
    let fluid_moved = fluid_initial
        .iter()
        .zip(&cold_fluid)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        fluid_moved > 0,
        "the fluid state never left its initial condition; a bit-exact match \
         afterwards would be trivial"
    );

    // (3) and yet nothing of the thermal field reached it.
    assert_eq!(
        cold_fluid, hot_fluid,
        "the fluid responded to the thermal modifier's temperature; a channel \
         between ScalarField3D (f32) and the Fix128 core now exists"
    );
}

// ============================================================================
// 3. The two f32 temperatures are also disjoint from each other
// ============================================================================

/// `ThermalModifier` and `PhaseChangeModifier` each own a temperature field
/// over the same region. Composed in one `ModifiedSdf`, they can disagree
/// about the temperature at a point without limit, and neither moves toward
/// the other.
///
/// This is the same conjunction as above, within the `f32` layer: the
/// perturbation changes the modifier it was applied to, both fields are live,
/// and the other field is bit-identical.
#[test]
fn the_two_f32_temperature_fields_never_reconcile() {
    let probe = (0.0_f32, 0.0_f32, 0.0_f32);

    // Both modifiers start from the same ambient so the disagreement below is
    // produced by the run, not by construction.
    let thermal_cfg = ThermalConfig {
        ambient_temperature: 20.0,
        ..ThermalConfig::default()
    };
    let phase_cfg = PhaseChangeConfig {
        ambient_temperature: 20.0,
        ..PhaseChangeConfig::default()
    };

    let run = |heat_thermal: f32, heat_phase: f32| {
        let mut thermal = ThermalModifier::new(thermal_cfg, RES, MIN, MAX);
        let mut phase = PhaseChangeModifier::new(phase_cfg, RES, MIN, MAX);
        thermal.apply_heat_at(probe.0, probe.1, probe.2, heat_thermal, 1.5);
        phase.apply_heat_at(probe.0, probe.1, probe.2, heat_phase, 1.5);

        let mut sdf = ModifiedSdf::new(Box::new(unit_sphere()));
        sdf.add_modifier(Box::new(thermal));
        sdf.add_modifier(Box::new(phase));

        let mut thermal_trace = Vec::new();
        let mut phase_trace = Vec::new();
        for _ in 0..STEPS {
            sdf.update(DT);
            // Read both fields back out of the chain.
            let m0 = sdf.modifier_mut(0).expect("thermal modifier at index 0");
            assert_eq!(m0.name(), "thermal");
            let t = m0.modify_distance(probe.0, probe.1, probe.2, 0.0);
            let m1 = sdf.modifier_mut(1).expect("phase modifier at index 1");
            assert_eq!(m1.name(), "phase_change");
            let p = m1.modify_distance(probe.0, probe.1, probe.2, 0.0);
            thermal_trace.push(t.to_bits());
            phase_trace.push(p.to_bits());
        }
        (thermal_trace, phase_trace)
    };

    // Baseline: both hot.
    let (base_thermal, base_phase) = run(3_000.0, 3_000.0);
    // Perturb only the phase-change temperature.
    let (thermal_vs_cold_phase, phase_cold) = run(3_000.0, 0.0);
    // Perturb only the thermal temperature.
    let (thermal_cold, phase_vs_cold_thermal) = run(0.0, 3_000.0);

    // (1) each perturbation is real in its own subsystem.
    assert_ne!(
        base_phase, phase_cold,
        "removing the phase-change heat did not change the phase-change \
         modifier's own contribution; the perturbation measures nothing"
    );
    assert_ne!(
        base_thermal, thermal_cold,
        "removing the thermal heat did not change the thermal modifier's own \
         contribution; the perturbation measures nothing"
    );

    // (2) both traces are live — they are not constant across the run.
    assert!(
        base_thermal.iter().any(|v| *v != base_thermal[0]),
        "the thermal contribution is constant over {STEPS} steps; comparing it \
         across runs would be trivial"
    );
    assert!(
        base_phase.iter().any(|v| *v != base_phase[0]),
        "the phase-change contribution is constant over {STEPS} steps; \
         comparing it across runs would be trivial"
    );

    // (3) and neither sees the other's field.
    assert_eq!(
        base_thermal, thermal_vs_cold_phase,
        "the thermal modifier responded to the phase-change temperature; the \
         two f32 temperature fields are now coupled"
    );
    assert_eq!(
        base_phase, phase_vs_cold_thermal,
        "the phase-change modifier responded to the thermal temperature; the \
         two f32 temperature fields are now coupled"
    );
}

/// The same quantity read through the two `f32` owners disagrees by an
/// unbounded amount, and `ModifiedSdf::update` does not narrow the gap.
///
/// `ModifiedSdf::update` calls each modifier's `update(dt)` in turn, so this is
/// a Jacobi sweep over subsystems with no exchange: the gap can only close by
/// each field independently decaying toward its own ambient.
#[test]
fn a_point_has_two_temperatures_and_the_gap_does_not_close() {
    let probe = (0.0_f32, 0.0_f32, 0.0_f32);
    let cfg_t = ThermalConfig {
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        diffusion_rate: 0.0,
        ..ThermalConfig::default()
    };
    let cfg_p = PhaseChangeConfig {
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        diffusion_rate: 0.0,
        ..PhaseChangeConfig::default()
    };

    let mut thermal = ThermalModifier::new(cfg_t, RES, MIN, MAX);
    let phase = PhaseChangeModifier::new(cfg_p, RES, MIN, MAX);
    // Contradictory statements about the temperature at the same world point.
    thermal.apply_heat_at(probe.0, probe.1, probe.2, 800.0, 1.5);

    let gap_before = (thermal.temperature_at(probe.0, probe.1, probe.2)
        - phase.temperature_at(probe.0, probe.1, probe.2))
    .abs();
    assert!(
        gap_before > 100.0,
        "the two fields were set up to disagree; measured gap {gap_before}"
    );

    // Advance both through the composition, the only place they meet.
    let mut sdf = ModifiedSdf::new(Box::new(unit_sphere()));
    sdf.add_modifier(Box::new(thermal));
    sdf.add_modifier(Box::new(phase));
    for _ in 0..STEPS {
        sdf.update(DT);
    }

    // Read the fields back out. Both cooling and diffusion are off, so any
    // narrowing would have to come from an exchange between them.
    let thermal_after = {
        let m = sdf.modifier_mut(0).expect("thermal modifier");
        // Temperature is not on the trait; go through the SDF contribution,
        // which is a strictly monotone function of it here (expansion term).
        m.modify_distance(probe.0, probe.1, probe.2, 0.0)
    };
    let phase_after = {
        let m = sdf.modifier_mut(1).expect("phase modifier");
        m.modify_distance(probe.0, probe.1, probe.2, 0.0)
    };

    // 820 K against a 200 K melt point: the thermal modifier is well past
    // melting and reports a large recession. Measured 3.34 at 40 steps.
    assert!(
        thermal_after > 1.0,
        "the thermal modifier stopped reporting melt ({thermal_after}); its \
         temperature was drained somewhere"
    );
    // The phase-change modifier sat next to that for the whole run with its own
    // cooling and diffusion switched off, so the only way its offset could
    // become non-zero is by reading the other field. It is exactly zero.
    assert_eq!(
        phase_after, 0.0,
        "the phase-change modifier developed an offset ({phase_after}) without \
         ever being heated; it read the thermal field"
    );
}

// ============================================================================
// 4. What the one channel that does exist actually carries
// ============================================================================

/// The scalar distance is a real channel for **exactly one** of the five
/// production implementors.
///
/// ⚠️ Reading the trait signature suggests the chain is a plain sum of
/// independent offsets. It is not. `modify_distance` receives the distance the
/// upstream modifiers produced, and `fracture` **branches on it**
/// (`if d < width * 2.0 { d = d.max(-crack_dist) }`, `src/fracture.rs`), so a
/// crack only cuts where the incoming surface was already close. That makes
/// fracture the one physics in this layer that observes what another physics
/// did — one-way, through a single `f32`, and order dependent.
///
/// The test measures, per implementor, whether `f(d) - d` depends on `d`. A
/// modifier that ignores the channel has a constant residual; one that reads it
/// does not. The partition is pinned so that a modifier newly reading (or
/// ceasing to read) the channel is a failure and not a silent change.
#[test]
fn the_composed_distance_is_read_by_exactly_one_implementor() {
    use alice_physics::erosion::{ErosionConfig, ErosionModifier};
    use alice_physics::fracture::{FractureConfig, FractureModifier};
    use alice_physics::pressure::{PressureConfig, PressureModifier};
    use alice_physics::sim_modifier::PhysicsModifier;

    let probe = (0.0_f32, 0.0_f32, 0.0_f32);
    // Distances spanning well inside, at, and well outside the surface.
    let probes: [f32; 7] = [-4.0, -1.0, -0.05, 0.0, 0.05, 1.0, 4.0];

    // Each modifier is driven into a state where it actually contributes;
    // an inert modifier returns `original_dist` unchanged and would land in
    // the "ignores the channel" bucket for the wrong reason.
    let mut built: Vec<Box<dyn PhysicsModifier>> = Vec::new();
    {
        let mut m = ThermalModifier::new(ThermalConfig::default(), RES, MIN, MAX);
        m.apply_heat_at(probe.0, probe.1, probe.2, 3_000.0, 1.5);
        for _ in 0..STEPS {
            m.update(DT);
        }
        built.push(Box::new(m));
    }
    {
        let mut m = PhaseChangeModifier::new(PhaseChangeConfig::default(), RES, MIN, MAX);
        m.apply_heat_at(probe.0, probe.1, probe.2, 6_000.0, 1.5);
        for _ in 0..STEPS {
            m.update(DT);
        }
        built.push(Box::new(m));
    }
    {
        let mut m = PressureModifier::new(PressureConfig::default(), RES, MIN, MAX);
        m.apply_pressure_at(probe.0, probe.1, probe.2, 5_000.0, 1.5);
        for _ in 0..STEPS {
            m.update(DT);
        }
        built.push(Box::new(m));
    }
    {
        let mut m = ErosionModifier::new(ErosionConfig::default(), RES, MIN, MAX);
        for _ in 0..STEPS {
            // Exposure decays every update (`EXPOSURE_DECAY_PER_S`), so the
            // caller has to keep supplying it for depth to accumulate.
            m.set_exposure_at(probe.0, probe.1, probe.2, 1.0, 1.5);
            m.update(DT);
        }
        built.push(Box::new(m));
    }
    {
        let mut m = FractureModifier::new(FractureConfig::default(), RES, MIN, MAX);
        m.apply_stress_at(probe.0, probe.1, probe.2, 400.0, 1.5);
        for _ in 0..STEPS {
            m.update(DT);
        }
        assert!(
            m.active_crack_count() > 0 || !m.cracks.is_empty(),
            "the fracture modifier grew no cracks, so it returns early and \
             cannot read the channel for a reason unrelated to its code"
        );
        built.push(Box::new(m));
    }

    let mut reads_channel: Vec<&str> = Vec::new();
    let mut ignores_channel: Vec<&str> = Vec::new();
    let mut inert: Vec<&str> = Vec::new();

    for m in &built {
        let residuals: Vec<f32> = probes
            .iter()
            .map(|d| m.modify_distance(probe.0, probe.1, probe.2, *d) - d)
            .collect();
        // Non-vacuity per modifier: it has to be doing something, or "constant
        // residual" would just mean "returns its input".
        let contributes = residuals.iter().any(|r| r.abs() > 1e-6);
        // Rounding-tolerant constancy: `f(d) = d + c` computed in f32 gives a
        // residual that can wobble by an ulp of the larger operand.
        let spread = residuals.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
            - residuals.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        let tolerance = 8.0 * f32::EPSILON * 4.0; // 4.0 = largest |d| probed

        if !contributes {
            inert.push(m.name());
        } else if spread > tolerance {
            reads_channel.push(m.name());
        } else {
            ignores_channel.push(m.name());
        }
    }

    assert!(
        inert.is_empty(),
        "these modifiers contributed nothing and were not classified: {inert:?} \
         — the partition below says nothing about them"
    );
    assert_eq!(
        ignores_channel,
        vec!["thermal", "phase_change", "pressure", "erosion"],
        "the set of modifiers that ignore the composed distance changed"
    );
    assert_eq!(
        reads_channel,
        vec!["fracture"],
        "the set of modifiers that read the composed distance changed; this is \
         the only data path between two physics in this layer, so a change here \
         changes what the coupling analysis has to account for"
    );
}

/// Because `fracture` reads the channel, the chain is order dependent in a way
/// that is not float rounding: putting a crack before or after the modifiers
/// that move the surface changes where it cuts.
///
/// The two magnitudes are asserted apart so the test distinguishes them: a
/// chain of offset-only modifiers differs by a few ulp when reordered (f32
/// addition is not associative), whereas inserting fracture differs by orders
/// of magnitude more.
#[test]
fn chain_order_matters_materially_only_because_of_fracture() {
    use alice_physics::fracture::{FractureConfig, FractureModifier};

    let probe = (0.4_f32, 0.1_f32, -0.2_f32);

    let heated_pair = || {
        let mut thermal = ThermalModifier::new(ThermalConfig::default(), RES, MIN, MAX);
        thermal.apply_heat_at(0.3, 0.0, 0.0, 2_000.0, 1.5);
        let mut phase = PhaseChangeModifier::new(PhaseChangeConfig::default(), RES, MIN, MAX);
        phase.apply_heat_at(0.3, 0.0, 0.0, 2_000.0, 1.5);
        (thermal, phase)
    };
    let cracked = || {
        let mut f = FractureModifier::new(FractureConfig::default(), RES, MIN, MAX);
        f.apply_stress_at(0.3, 0.0, 0.0, 400.0, 1.5);
        f
    };

    // (a) offset-only chain, both orders.
    let offsets_only = |swapped: bool| {
        let (thermal, phase) = heated_pair();
        let mut sdf = ModifiedSdf::new(Box::new(unit_sphere()));
        if swapped {
            sdf.add_modifier(Box::new(phase));
            sdf.add_modifier(Box::new(thermal));
        } else {
            sdf.add_modifier(Box::new(thermal));
            sdf.add_modifier(Box::new(phase));
        }
        for _ in 0..STEPS {
            sdf.update(DT);
        }
        sdf.distance(probe.0, probe.1, probe.2)
    };
    let a0 = offsets_only(false);
    let a1 = offsets_only(true);
    let offset_gap = (a0 - a1).abs();

    // (b) the same chain with fracture at the front or at the back.
    let with_fracture = |fracture_first: bool| {
        let (thermal, phase) = heated_pair();
        let f = cracked();
        let mut sdf = ModifiedSdf::new(Box::new(unit_sphere()));
        if fracture_first {
            sdf.add_modifier(Box::new(f));
            sdf.add_modifier(Box::new(thermal));
            sdf.add_modifier(Box::new(phase));
        } else {
            sdf.add_modifier(Box::new(thermal));
            sdf.add_modifier(Box::new(phase));
            sdf.add_modifier(Box::new(f));
        }
        for _ in 0..STEPS {
            sdf.update(DT);
        }
        sdf.distance(probe.0, probe.1, probe.2)
    };
    let b0 = with_fracture(true);
    let b1 = with_fracture(false);
    let fracture_gap = (b0 - b1).abs();

    // Non-vacuity: the modifiers are contributing, so neither chain is the
    // bare sphere.
    let bare = unit_sphere().distance(probe.0, probe.1, probe.2);
    assert!(
        (a0 - bare).abs() > 1e-3,
        "the offset-only chain contributed nothing at the probe ({a0} vs \
         {bare}); the comparison below is vacuous"
    );

    // Reordering offset-only modifiers costs at most float rounding.
    assert!(
        offset_gap <= 8.0 * f32::EPSILON * a0.abs().max(1.0),
        "reordering the offset-only chain moved the distance by {offset_gap}, \
         which is more than f32 non-associativity accounts for; a modifier \
         started reading the composed distance"
    );
    // Moving fracture across them does not.
    assert!(
        fracture_gap > 1e-3,
        "moving fracture from the front of the chain to the back changed \
         nothing ({fracture_gap}); it stopped reading the composed distance, \
         and this layer no longer has any data path between two physics"
    );
}
