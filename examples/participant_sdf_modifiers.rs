//! An SDF modifier as a world participant: a thermal modifier with a point
//! heat source advances once per substep through
//! [`alice_physics::world_participant::run_substep`], reports its
//! observations, is snapshotted half way and restored into a participant
//! built with another resolution, and the restored copy finishes the run
//! with the same bytes as the original.
//!
//! Run: `cargo run --example participant_sdf_modifiers`

use alice_physics::math::Fix128;
use alice_physics::solver::RigidBody;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};
use alice_physics::world_participant::{
    run_substep, FieldBoard, ForceAccumulator, ObservationSink, Participant, ParticipantPlan,
    SubstepTime,
};

/// Runs `substeps` substeps of width `h` over `ps` (no bodies, no fields).
fn run(ps: &mut [Box<dyn Participant>], substeps: usize, h: Fix128) {
    let mut board = FieldBoard::new();
    let plan = ParticipantPlan::new(ps, &board).expect("modifiers declare no ports");
    let mut frozen = vec![false; ps.len()];
    let bodies: Vec<RigidBody> = Vec::new();
    let mut forces = ForceAccumulator::new(0);
    for index in 0..substeps {
        let time = SubstepTime {
            index,
            count: substeps,
            h,
        };
        let faults = run_substep(
            ps,
            &plan,
            &mut frozen,
            &bodies,
            &mut board,
            &mut forces,
            time,
        )
        .expect("inputs fit");
        assert!(faults.is_empty(), "a modifier substep never fails");
    }
}

fn print_observations(label: &str, p: &dyn Participant) {
    let mut sink = ObservationSink::new();
    p.observe(&mut sink);
    let named: Vec<String> = sink
        .values()
        .iter()
        .map(|&(channel, v)| {
            let name = match channel {
                0 => "max temperature",
                1 => "total melt",
                _ => "?",
            };
            format!("{name} = {:.4}", v.to_f64())
        })
        .collect();
    println!("{label}: {}", named.join(", "));
}

fn state(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

fn main() {
    let config = ThermalConfig {
        melt_temperature: 60.0,
        ..ThermalConfig::default()
    };
    let lo = (-2.0, -2.0, -2.0);
    let hi = (2.0, 2.0, 2.0);
    let scene = || {
        let mut m = ThermalModifier::new(config, 6, lo, hi);
        m.add_heat_point(0.0, 0.0, 0.0, 4000.0, 1.5);
        m
    };
    let h = Fix128::from_ratio(1, 240);
    println!(
        "kind {:#010x} (\"THRM\"), step rule {:?}",
        ThermalModifier::PARTICIPANT_KIND.get(),
        scene().step_rule()
    );

    // uninterrupted: 40 substeps
    let mut whole: Vec<Box<dyn Participant>> = vec![Box::new(scene())];
    print_observations("start", whole[0].as_ref());
    run(&mut whole, 40, h);
    print_observations("after 40 substeps", whole[0].as_ref());

    // 20 substeps, snapshot, restore into a fresh 2³ modifier, 20 more
    let mut first: Vec<Box<dyn Participant>> = vec![Box::new(scene())];
    run(&mut first, 20, h);
    print_observations("after 20 substeps", first[0].as_ref());
    let snapshot = state(first[0].as_ref());
    println!("snapshot: {} bytes", snapshot.len());

    let mut restored = ThermalModifier::new(ThermalConfig::default(), 2, lo, lo);
    restored
        .check_state(&snapshot)
        .expect("a payload write_state produced");
    restored.read_state(&snapshot);
    let short = &snapshot[..snapshot.len() - 1];
    println!(
        "a payload one byte short: {:?}",
        restored.check_state(short).expect_err("refused")
    );

    let mut second: Vec<Box<dyn Participant>> = vec![Box::new(restored)];
    run(&mut second, 20, h);
    print_observations("restored + 20 substeps", second[0].as_ref());

    let same = state(second[0].as_ref()) == state(whole[0].as_ref());
    println!("restored run matches the uninterrupted run bit for bit: {same}");
    assert!(same);
}
