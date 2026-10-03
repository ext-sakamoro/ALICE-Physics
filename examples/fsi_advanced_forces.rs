//! Advanced fluid-structure interaction: per-sample drag + buoyancy,
//! aggregation into a net force/torque, and Newton-III back-reaction onto
//! the fluid, for `alice_physics::fsi_advanced`.
//!
//! ⚠️ **Why this example exists.** `scripts/wiring_guard.py` reported five
//! items as unwired: `SolidSample`, `aggregate_forces`, `buoyancy_force`,
//! `drag_force`, `react_back_pressure`. `tests/engineering_oracles_fluid.rs`
//! and `tests/fsi_advanced_sub_iteration.rs` already drive all five against
//! closed forms (terminal velocity, Archimedes buoyancy, a two-sample
//! dumbbell torque, Newton's third law, and a sub-iteration contraction
//! ratio) — but tests do not count as production callers for the wiring
//! guard, and nothing in `src/` / `examples/` / `benches/` called any of
//! the five before this file existed. This example is that caller.
//! `tests/analytic_fsi_advanced_wiring.rs` holds additional closed-form
//! oracles for cases the existing test files do not cover: a dispatcher
//! cross-check between `aggregate_forces` and manually-summed direct calls
//! at a fully 3D (non-single-axis) sample layout, drag/buoyancy zeroed by
//! `area_m2 = 0` / `fluid_density = 0` independently, and extreme-magnitude
//! wraparound for all three force functions.
//!
//! # Scenario
//!
//! A three-point sample of a submerged articulated frame (the kind of
//! "bag of sample points" the module's doc comment describes standing in
//! for a deformable or articulated solid) drifting in a uniform water
//! current of `(2, 0, 0)` m/s, `ρ = 1000 kg/m³`, `C_d = 1`, `g = 10 m/s²`.
//! Sample C's velocity is set equal to the current exactly, so its drag is
//! the module's zero-relative-velocity early return, and its volume is
//! zero, so its buoyancy is also zero — both exercised in place rather
//! than in an isolated unit.
//!
//! ```bash
//! cargo run --example fsi_advanced_forces --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example fsi_advanced_forces --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::fsi_advanced::{
        aggregate_forces, buoyancy_force, drag_force, react_back_pressure, SolidSample,
    };
    use alice_physics::math::{Fix128, Vec3Fix};

    let fluid_density = Fix128::from_int(1000);
    let drag_coefficient = Fix128::ONE;
    let gravity = Fix128::from_int(10);
    let current = Vec3Fix::from_int(2, 0, 0);
    let fluid_velocity_sampler = |_p: Vec3Fix| current;

    let sample_a = SolidSample {
        position: Vec3Fix::from_int(2, 0, 0),
        velocity: Vec3Fix::from_int(5, 0, 0),
        area_m2: Fix128::from_int(2),
        volume_m3: Fix128::from_int(1),
    };
    let sample_b = SolidSample {
        position: Vec3Fix::from_int(-2, 0, 0),
        velocity: Vec3Fix::from_int(2, 3, 4),
        area_m2: Fix128::ONE,
        volume_m3: Fix128::from_int(2),
    };
    let sample_c = SolidSample {
        position: Vec3Fix::from_int(0, 0, 5),
        velocity: current, // exactly the fluid velocity: v_rel = 0
        area_m2: Fix128::from_int(3),
        volume_m3: Fix128::ZERO, // no displaced volume: buoyancy = 0
    };
    println!(
        "[fsi_advanced] samples: A pos={:?} vel={:?}, B pos={:?} vel={:?}, C pos={:?} vel={:?} (vel == current)",
        (sample_a.position.x.to_f64(), sample_a.position.y.to_f64(), sample_a.position.z.to_f64()),
        (sample_a.velocity.x.to_f64(), sample_a.velocity.y.to_f64(), sample_a.velocity.z.to_f64()),
        (sample_b.position.x.to_f64(), sample_b.position.y.to_f64(), sample_b.position.z.to_f64()),
        (sample_b.velocity.x.to_f64(), sample_b.velocity.y.to_f64(), sample_b.velocity.z.to_f64()),
        (sample_c.position.x.to_f64(), sample_c.position.y.to_f64(), sample_c.position.z.to_f64()),
        (sample_c.velocity.x.to_f64(), sample_c.velocity.y.to_f64(), sample_c.velocity.z.to_f64()),
    );

    // --- Part 1: drag_force + buoyancy_force, called directly -----------
    //
    // Sample A: v_rel = (5,0,0) - (2,0,0) = (3,0,0), |v_rel| = 3 exactly.
    // F_d = -0.5 * rho * Cd * A * |v_rel| * v_rel
    //     = -0.5 * 1000 * 1 * 2 * 3 * (3,0,0) = -3000 * (3,0,0) = (-9000,0,0)
    // F_b = rho * V * g = 1000 * 1 * 10 = 10000, along +Y.
    let drag_a = drag_force(&sample_a, current, fluid_density, drag_coefficient);
    let buoyancy_a = buoyancy_force(&sample_a, fluid_density, gravity);
    assert_eq!(
        drag_a,
        Vec3Fix::new(Fix128::from_int(-9000), Fix128::ZERO, Fix128::ZERO)
    );
    assert_eq!(
        buoyancy_a,
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10_000), Fix128::ZERO)
    );
    println!(
        "[fsi_advanced] sample A: drag={:?} buoyancy={:?} (hand: drag=(-9000,0,0), buoyancy=(0,10000,0))",
        (drag_a.x.to_f64(), drag_a.y.to_f64(), drag_a.z.to_f64()),
        (buoyancy_a.x.to_f64(), buoyancy_a.y.to_f64(), buoyancy_a.z.to_f64()),
    );

    // Sample C: v_rel = 0 exactly (velocity == current) -> drag's early
    // return, independent of area_m2 = 3 (a nonzero area that would
    // otherwise make the prefactor nonzero). volume_m3 = 0 -> buoyancy 0
    // independent of the nonzero density/gravity.
    let drag_c = drag_force(&sample_c, current, fluid_density, drag_coefficient);
    let buoyancy_c = buoyancy_force(&sample_c, fluid_density, gravity);
    assert_eq!(drag_c, Vec3Fix::ZERO, "zero relative velocity -> zero drag");
    assert_eq!(buoyancy_c, Vec3Fix::ZERO, "zero volume -> zero buoyancy");
    println!(
        "[fsi_advanced] sample C: drag={drag_c:?} buoyancy={buoyancy_c:?} (both exactly zero)"
    );

    // --- Part 2: aggregate_forces over all three samples -----------------
    //
    // Sample B: v_rel = (2,3,4) - (2,0,0) = (0,3,4), |v_rel| = 5 exactly.
    // F_d = -0.5*1000*1*1*5 * (0,3,4) = -2500*(0,3,4) = (0,-7500,-10000)
    // F_b = 1000*2*10 = 20000, along +Y -> total_B = (0,12500,-10000)
    //
    // total_A = drag_a + buoyancy_a = (-9000,10000,0); total_C = (0,0,0).
    //
    // Net force = total_A + total_B + total_C = (-9000,22500,-10000).
    //
    // Torque about the origin (r x F per sample, A and B only since C's
    // force is zero):
    //   A: r=(2,0,0), F=(-9000,10000,0)  -> tau = (0,0,20000)
    //   B: r=(-2,0,0), F=(0,12500,-10000) -> tau = (0,-20000,-25000)
    //   sum                                = (0,-20000,-5000)
    let reference_point = Vec3Fix::ZERO;
    let samples = [sample_a, sample_b, sample_c];
    let (net_force, net_torque) = aggregate_forces(
        &samples,
        fluid_velocity_sampler,
        fluid_density,
        drag_coefficient,
        gravity,
        reference_point,
    );
    let expected_net_force = Vec3Fix::new(
        Fix128::from_int(-9000),
        Fix128::from_int(22_500),
        Fix128::from_int(-10_000),
    );
    let expected_net_torque = Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(-20_000),
        Fix128::from_int(-5_000),
    );
    assert_eq!(net_force, expected_net_force, "net force");
    assert_eq!(net_torque, expected_net_torque, "net torque");
    println!(
        "[fsi_advanced] aggregate: net_force={:?} net_torque={:?} (hand: force=(-9000,22500,-10000), torque=(0,-20000,-5000))",
        (net_force.x.to_f64(), net_force.y.to_f64(), net_force.z.to_f64()),
        (net_torque.x.to_f64(), net_torque.y.to_f64(), net_torque.z.to_f64()),
    );

    // --- Part 3: react_back_pressure deposits -F at each sample's position
    //
    // Deposit the per-sample totals computed above; the reaction must be
    // the exact negation at the exact sample position, matching Newton's
    // third law.
    let total_a = Vec3Fix::new(
        Fix128::from_int(-9000),
        Fix128::from_int(10_000),
        Fix128::ZERO,
    );
    let total_b = Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(12_500),
        Fix128::from_int(-10_000),
    );
    let total_c = Vec3Fix::ZERO;
    let solid_forces = [total_a, total_b, total_c];
    let mut deposits: Vec<(Vec3Fix, Vec3Fix)> = Vec::new();
    react_back_pressure(&samples, &solid_forces, |pos, reaction| {
        deposits.push((pos, reaction));
    });
    assert_eq!(deposits.len(), 3);
    assert_eq!(deposits[0].0, sample_a.position);
    assert_eq!(
        deposits[0].1,
        Vec3Fix::new(
            Fix128::from_int(9000),
            Fix128::from_int(-10_000),
            Fix128::ZERO
        )
    );
    assert_eq!(deposits[1].0, sample_b.position);
    assert_eq!(
        deposits[1].1,
        Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_int(-12_500),
            Fix128::from_int(10_000)
        )
    );
    assert_eq!(deposits[2].0, sample_c.position);
    assert_eq!(deposits[2].1, Vec3Fix::ZERO);
    // Momentum bookkeeping: the deposited reactions must sum to exactly
    // -(net force) computed in Part 2, independent of per-sample totals.
    let deposit_sum = deposits.iter().fold(Vec3Fix::ZERO, |acc, (_, f)| {
        Vec3Fix::new(acc.x + f.x, acc.y + f.y, acc.z + f.z)
    });
    assert_eq!(
        deposit_sum,
        Vec3Fix::new(
            Fix128::ZERO - net_force.x,
            Fix128::ZERO - net_force.y,
            Fix128::ZERO - net_force.z
        )
    );
    println!(
        "[fsi_advanced] react_back_pressure: {} deposits, sum={:?} (== -net_force, Newton's third law)",
        deposits.len(),
        (deposit_sum.x.to_f64(), deposit_sum.y.to_f64(), deposit_sum.z.to_f64()),
    );

    // --- Part 4: two's-complement self-negating boundary -----------------
    //
    // Fix128's most negative representable value, -2^63 exactly, is its
    // own two's-complement negation (confirmed against `src/math.rs`'s
    // `Sub`/`Neg` impls, not assumed): `react_back_pressure` deposits
    // `ZERO - f`, so depositing this exact force returns the *same* value
    // instead of its arithmetic negative. No panic, deterministic.
    let min_value = Fix128::from_raw(i64::MIN, 0);
    let boundary_sample = SolidSample {
        position: Vec3Fix::ZERO,
        velocity: Vec3Fix::ZERO,
        area_m2: Fix128::ZERO,
        volume_m3: Fix128::ZERO,
    };
    let mut boundary_reaction = Vec3Fix::ZERO;
    react_back_pressure(
        &[boundary_sample],
        &[Vec3Fix::new(min_value, Fix128::ZERO, Fix128::ZERO)],
        |_pos, reaction| boundary_reaction = reaction,
    );
    assert_eq!(
        boundary_reaction.x, min_value,
        "ZERO - MIN must wrap back to MIN, not to a positive value"
    );
    println!(
        "[fsi_advanced] boundary: depositing Fix128::MIN reacts to {:?} (self-negating wrap, no panic)",
        boundary_reaction.x
    );

    println!(
        "[fsi_advanced] all 5 wiring targets exercised: SolidSample, aggregate_forces, buoyancy_force, drag_force, react_back_pressure"
    );
}
