//! A crowd and a molecular-dynamics system advanced together through
//! `run_substep`, the participant step the world calls once per substep.
//!
//! Run with `cargo run --example participant_crowd_md`.

use alice_physics::crowd_force::{
    CrowdParticipant, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{MdParticipant, PeriodicBox, VelocityVerlet};
use alice_physics::pair_potential::{LennardJones, ShiftMode, Truncated};
use alice_physics::physics2d::Vec2Fix;
use alice_physics::solver::RigidBody;
use alice_physics::world_participant::{
    run_substep, FieldBoard, ForceAccumulator, ObservationSink, Participant, ParticipantPlan,
    SubstepTime,
};

fn hfv() -> InteractionParams {
    InteractionParams {
        strength_n: Fix128::from_int(2000),
        range_m: Fix128::from_ratio(2, 25),
        body_stiffness: Fix128::from_int(120_000),
        sliding_friction: Fix128::from_int(240_000),
    }
}

fn walker(x: i64, y: Fix128, dir: i64) -> Pedestrian {
    Pedestrian {
        position: Vec2Fix::new(Fix128::from_int(x), y),
        velocity: Vec2Fix::new(Fix128::ZERO, Fix128::ZERO),
        radius_m: Fix128::from_ratio(3, 10),
        mass_kg: Fix128::from_int(80),
        desired_speed_m_s: Fix128::from_ratio(13, 10),
        desired_direction: Vec2Fix::new(Fix128::from_int(dir), Fix128::ZERO),
        relaxation_time_s: Fix128::from_ratio(1, 2),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = SocialForce::new(hfv(), hfv(), Fix128::from_ratio(1, 2), Fix128::from_int(2))?;
    let walls = vec![
        WallSegment {
            start: Vec2Fix::new(Fix128::from_int(-10), Fix128::from_int(-2)),
            end: Vec2Fix::new(Fix128::from_int(10), Fix128::from_int(-2)),
        },
        WallSegment {
            start: Vec2Fix::new(Fix128::from_int(-10), Fix128::from_int(2)),
            end: Vec2Fix::new(Fix128::from_int(10), Fix128::from_int(2)),
        },
    ];
    let mut crowd = CrowdParticipant::new(
        model,
        vec![
            walker(-3, Fix128::from_ratio(1, 5), 1),
            walker(-4, Fix128::from_ratio(-3, 5), 1),
            walker(3, Fix128::ZERO, -1),
            walker(4, Fix128::from_ratio(4, 5), -1),
        ],
        walls,
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )?;

    let lj = Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE)?,
        Fix128::from_ratio(5, 2),
        ShiftMode::ForceShift,
    )?;
    let mut positions = Vec::new();
    for i in 0..2i64 {
        for j in 0..2i64 {
            for k in 0..2i64 {
                positions.push(Vec3Fix::new(
                    Fix128::from_ratio(20 + 12 * i, 10),
                    Fix128::from_ratio(20 + 12 * j, 10),
                    Fix128::from_ratio(20 + 12 * k, 10) + Fix128::from_ratio(i + j, 10),
                ));
            }
        }
    }
    let velocities: Vec<Vec3Fix> = (0..8i64)
        .map(|n| {
            Vec3Fix::new(
                Fix128::from_ratio(n % 3 - 1, 5),
                Fix128::from_ratio(1 - n % 2 * 2, 10),
                Fix128::ZERO,
            )
        })
        .collect();
    let md = MdParticipant::new(
        VelocityVerlet::new(
            lj,
            PeriodicBox::cubic(Fix128::from_int(6))?,
            positions,
            velocities,
            vec![Fix128::ONE; 8],
        )?,
        Fix128::ONE,
    )?;

    // route choice stays with the caller: the last walker turns slightly
    // towards the upper wall before the run starts
    if let Some(p) = crowd.pedestrians_mut().last_mut() {
        p.desired_direction = Vec2Fix::new(Fix128::from_int(-4), Fix128::ONE);
    }
    println!(
        "crowd: {} pedestrians, {} walls, model {:?}",
        crowd.pedestrians().len(),
        crowd.walls().len(),
        crowd.model()
    );
    let p = md.system().momentum();
    println!(
        "md: {} particles, k_B = {}, momentum = ({:.6}, {:.6}, {:.6})",
        md.system().positions().len(),
        md.boltzmann_constant().to_f64(),
        p.x.to_f64(),
        p.y.to_f64(),
        p.z.to_f64()
    );

    let mut participants: Vec<Box<dyn Participant>> = vec![Box::new(crowd), Box::new(md)];
    let bodies: Vec<RigidBody> = Vec::new();
    let mut board = FieldBoard::new();
    let plan = ParticipantPlan::new(&participants, &board).map_err(|e| format!("plan: {e:?}"))?;
    let mut frozen = vec![false; participants.len()];
    let h = Fix128::from_ratio(1, 256);
    for frame in 0..=8usize {
        if frame > 0 {
            for index in 0..64 {
                let mut forces = ForceAccumulator::new(bodies.len());
                let time = SubstepTime {
                    index,
                    count: 64,
                    h,
                };
                let faults = run_substep(
                    &mut participants,
                    &plan,
                    &mut frozen,
                    &bodies,
                    &mut board,
                    &mut forces,
                    time,
                )
                .map_err(|e| format!("substep: {e:?}"))?;
                if !faults.is_empty() {
                    println!("faults: {faults:?}");
                }
            }
        }
        for p in &participants {
            let mut obs = ObservationSink::new();
            p.observe(&mut obs);
            let values: Vec<String> = obs
                .values()
                .iter()
                .map(|(c, v)| format!("{c}={:.6}", v.to_f64()))
                .collect();
            println!(
                "t={:.2}s kind={:#010x} {}",
                frame as f64 * 0.25,
                p.kind().get(),
                values.join(" ")
            );
        }
    }
    Ok(())
}
