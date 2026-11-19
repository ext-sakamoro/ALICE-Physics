//! Compare two ways of making a cloth a resident of the world.
//!
//! B: the world's own cloth list (`PhysicsWorld::cloths`), stepped once per frame
//! after the rigid substeps. A: [`ClothParticipant`], stepped once per world
//! substep through the participant contract.
//!
//! The scene is a thrown letter E tied to its rest shape (the site's mochi tether),
//! no gravity: it should come back to rest. Oracles are closed forms or the other
//! path's bits, never this file's own output.

use alice_physics_core::cloth::{Cloth, ClothConfig, OutlineDesc};
use alice_physics_core::cloth_participant::{ClothParticipant, CLOTH_PARTICIPANT_KIND};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

fn letter_e() -> Vec<Vec<[Fix128; 2]>> {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    vec![vec![
        p(0, 0),
        p(10, 0),
        p(10, 2),
        p(2, 2),
        p(2, 4),
        p(10, 4),
        p(10, 6),
        p(2, 6),
        p(2, 8),
        p(10, 8),
        p(10, 10),
        p(0, 10),
    ]]
}

fn letter() -> Cloth {
    letter_at(1)
}

/// The letter stepping `m` times per frame (its own substeps), the same 4
/// iterations per step.
fn letter_at(m: usize) -> Cloth {
    let omega = Fix128::from_f32(core::f32::consts::TAU / 0.5);
    let tether = (omega * omega, omega * Fix128::from_f32(2.0 * 0.42));
    let mut c = Cloth::from_outline(&OutlineDesc {
        origin: Vec3Fix::ZERO,
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours: letter_e(),
        spacing: r(1, 10),
        mass_per_particle: Fix128::ONE,
        config: ClothConfig {
            iterations: 4,
            substeps: m,
            gravity: Vec3Fix::ZERO,
            damping: r(985, 1000),
            stretch_compliance: r(1, 100),
            bend_compliance: r(1, 4),
            ..ClothConfig::default()
        },
        tether: Some(tether),
    })
    .expect("E");
    // thrown: every particle starts with 1 m/s out of the page
    for v in &mut c.velocities {
        *v = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    }
    c
}

fn world(substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        ..SolverConfig::default()
    })
}

fn world_b(substeps: usize) -> PhysicsWorld {
    world_b_at(substeps, 1)
}

fn world_b_at(substeps: usize, m: usize) -> PhysicsWorld {
    let mut w = world(substeps);
    w.add_cloth(letter_at(m));
    w
}

fn world_a(substeps: usize) -> PhysicsWorld {
    world_a_at(substeps, 1)
}

fn world_a_at(substeps: usize, m: usize) -> PhysicsWorld {
    let mut w = world(substeps);
    w.add_participant(Box::new(ClothParticipant::new(letter_at(m))))
        .expect("register");
    w
}

fn fix_at(b: &[u8], at: usize) -> Fix128 {
    let hi = i64::from_le_bytes(b[at..at + 8].try_into().expect("8"));
    let lo = u64::from_le_bytes(b[at + 8..at + 16].try_into().expect("8"));
    Fix128::from_raw(hi, lo)
}

/// Positions out of the participant's payload: version u32, count u32, structure
/// hash u64, then per particle position / previous / velocity / inverse mass.
fn positions_a(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    let b = w.participant_state(0).expect("payload");
    let n = u32::from_le_bytes(b[4..8].try_into().expect("4")) as usize;
    (0..n)
        .map(|i| {
            let at = 16 + i * (3 * 48 + 16);
            Vec3Fix::new(fix_at(&b, at), fix_at(&b, at + 16), fix_at(&b, at + 32))
        })
        .collect()
}

fn positions_b(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    w.cloths[0].positions.clone()
}

fn max_from_rest(p: &[Vec3Fix], rest: &[Vec3Fix]) -> f64 {
    p.iter()
        .zip(rest)
        .map(|(a, b)| (*a - *b).length().to_f64())
        .fold(0.0, f64::max)
}

fn step(w: &mut PhysicsWorld, frames: usize) {
    for _ in 0..frames {
        w.step(dt());
    }
}

/// A cloth of one step per frame takes, on the resident path, one step of the frame's
/// width whatever the world's substeps: it must reproduce the world's own cloth step
/// bit for bit at world substeps 1, 2, 4 and 8.
#[test]
fn one_step_per_frame_matches_the_world_cloth_bit_for_bit_at_any_substeps() {
    for n in [1usize, 2, 4, 8] {
        let (mut a, mut b) = (world_a(n), world_b(n));
        for frame in 0..300 {
            step(&mut a, 1);
            step(&mut b, 1);
            assert_eq!(
                positions_a(&a),
                positions_b(&b),
                "substeps {n} frame {frame}"
            );
        }
        assert_eq!(a.participant_kinds(), vec![CLOTH_PARTICIPANT_KIND]);
    }
}

/// A snapshot taken mid-run and restored replays the rest of the run bit for bit
/// (rewind), and a world built from the snapshot does too (fork).
#[test]
fn the_resident_rewinds_and_forks_bit_for_bit() {
    let mut a = world_a(8);
    step(&mut a, 30);
    let snap = a.snapshot_world();
    step(&mut a, 90);
    let end = positions_a(&a);
    a.restore_world(&snap).expect("rewind");
    step(&mut a, 90);
    assert_eq!(positions_a(&a), end, "rewind");
    let mut fork = world_a(8);
    fork.restore_world(&snap).expect("fork");
    step(&mut fork, 90);
    assert_eq!(positions_a(&fork), end, "fork");
}

/// The thrown letter comes back to rest within 5 s on either path, under the
/// world's default 8 substeps. Printed: the largest distance from rest at 0.5 s
/// steps, so the two paths' settling can be compared.
#[test]
fn both_paths_bring_the_thrown_letter_back_to_rest() {
    let rest = letter().rest_positions().to_vec();
    let (mut a, mut b) = (world_a(8), world_b(8));
    let mut rows = Vec::new();
    for half in 1..=10 {
        step(&mut a, 30);
        step(&mut b, 30);
        let (da, db) = (
            max_from_rest(&positions_a(&a), &rest),
            max_from_rest(&positions_b(&b), &rest),
        );
        rows.push(format!(
            "{:.1} s  A {da:.6}  B {db:.6}",
            f64::from(half) / 2.0
        ));
    }
    println!("[cloth resident] from rest (m):\n{}", rows.join("\n"));
    assert!(max_from_rest(&positions_b(&b), &rest) < 1e-3, "B settles");
    assert!(max_from_rest(&positions_a(&a), &rest) < 1e-3, "A settles");
}

/// Snapshot bytes of each path (printed); the frame cost is measured by
/// `examples/cloth_resident_cost.rs` (a wall clock does not belong in a test).
#[test]
fn snapshot_size_is_measured() {
    for substeps in [2usize, 8] {
        let (mut a, mut b) = (world_a_at(substeps, 2), world_b(substeps));
        step(&mut a, 120);
        step(&mut b, 120);
        let (sa, sb) = (a.snapshot_world().len(), b.snapshot_world().len());
        println!("[cloth resident] substeps {substeps}: snapshot A {sa} B {sb} bytes");
        assert!(sa > 0 && sb > 0);
    }
}

/// The settle of the thrown letter does not depend on how many substeps the world
/// cuts a frame into: for 1, 2 and 4 cloth steps per frame, the largest distance from
/// rest at 0.25 / 0.5 / 1.0 s at more world substeps, including counts that `m` does
/// not divide (3, 5, 6: uneven step widths), is within 2 % of the value at world
/// substeps `m`. (Oracle: the cloth's steps are placed per frame, so only the
/// rounding of `k·h` against `dt/m` may differ.)
#[test]
fn the_settle_does_not_depend_on_the_world_substeps() {
    let rest = letter().rest_positions().to_vec();
    let cases: [(usize, &[usize]); 3] = [
        (1, &[1, 2, 3, 4, 8]),
        (2, &[2, 3, 4, 5, 8]),
        (4, &[4, 6, 8]),
    ];
    for (m, ns) in cases {
        let mut table = Vec::new();
        for &n in ns {
            let mut a = world_a_at(n, m);
            let mut row = Vec::new();
            for frames in [15usize, 15, 30] {
                step(&mut a, frames);
                row.push(max_from_rest(&positions_a(&a), &rest));
            }
            table.push((n, row));
        }
        let base = table[0].1.clone();
        for (n, row) in &table {
            println!(
                "[cloth resident] m {m} n {n}: 0.25 s {:.7}  0.5 s {:.7}  1.0 s {:.7}",
                row[0], row[1], row[2]
            );
            for (k, (x, b)) in row.iter().zip(&base).enumerate() {
                assert!(*b > 0.0, "the letter moved (m {m} sample {k})");
                assert!(
                    ((x - b) / b).abs() <= 0.02,
                    "m {m} n {n} sample {k}: {x} vs {b}"
                );
            }
        }
    }
}

/// More cloth steps per frame than world substeps is refused, at registration and at
/// a step after the world's substeps were lowered; the refused step changes nothing.
#[test]
fn more_steps_per_frame_than_substeps_is_refused() {
    use alice_physics_core::world_participant::RegisterError;
    let mut w = world(2);
    let err = w
        .add_participant(Box::new(ClothParticipant::new(letter_at(4))))
        .unwrap_err();
    assert_eq!(
        err,
        RegisterError::StepsPerFrameAboveSubsteps {
            steps: 4,
            substeps: 2
        }
    );
    assert_eq!(w.participant_count(), 0);
    let mut w = world_a_at(4, 4);
    w.config.substeps = 2;
    let before = positions_a(&w);
    assert!(w.try_step(dt()).is_err(), "the step is refused");
    w.step(dt());
    assert_eq!(positions_a(&w), before, "a refused step runs nothing");
}

#[test]
fn per_frame_places_the_calls_evenly() {
    use alice_physics_core::world_participant::StepRule;
    let at = |m: u32, n: usize| -> Vec<Option<usize>> {
        (0..n).map(|i| StepRule::PerFrame(m).covers(i, n)).collect()
    };
    // oracle: m calls cover the n substeps in contiguous runs summing to n
    // each call sits at the first substep of its span
    assert_eq!(at(1, 4), vec![Some(4), None, None, None]);
    assert_eq!(at(2, 4), vec![Some(2), None, Some(2), None]);
    assert_eq!(at(2, 3), vec![Some(2), None, Some(1)]);
    assert_eq!(at(3, 3), vec![Some(1), Some(1), Some(1)]);
    assert_eq!(StepRule::PerFrame(0).covers(0, 4), None);
    for (m, n) in [(1u32, 1usize), (2, 5), (3, 7), (4, 8), (5, 8)] {
        let v = at(m, n);
        assert_eq!(
            v.iter().filter(|c| c.is_some()).count(),
            m as usize,
            "m {m} n {n}"
        );
        assert_eq!(v.iter().flatten().sum::<usize>(), n, "m {m} n {n}");
        assert!(
            v[0].is_some(),
            "the first substep starts the frame with a call"
        );
    }
}

/// A body and a resident cloth in one world act on each other: a ball dropped
/// through the letter's plane moves differently from the same ball in a world
/// without the cloth, and the cloth differently from the same cloth in a world
/// without the ball (the contact itself is measured in `tests/cloth_body_contact.rs`).
#[test]
fn a_body_and_a_resident_cloth_act_on_each_other() {
    let g = Vec3Fix::new(Fix128::ZERO, r(-981, 100), Fix128::ZERO);
    let make = |with_ball: bool, with_cloth: bool| {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: g,
            substeps: 4,
            ..SolverConfig::default()
        });
        if with_ball {
            let b = w.add_body(RigidBody::new_dynamic(
                Vec3Fix::new(r(1, 2), r(3, 10), r(1, 2)),
                r(1, 100),
            ));
            w.set_body_collision_radius(b, r(1, 10));
        }
        if with_cloth {
            w.add_participant(Box::new(ClothParticipant::new(letter_at(2))))
                .expect("register");
        }
        w
    };
    let (mut both, mut ball, mut cloth) = (make(true, true), make(true, false), make(false, true));
    for _ in 0..120 {
        step(&mut both, 1);
        step(&mut ball, 1);
        step(&mut cloth, 1);
    }
    assert_ne!(
        both.bodies[0].position, ball.bodies[0].position,
        "the ball feels the cloth"
    );
    assert_ne!(
        positions_a(&both),
        positions_a(&cloth),
        "the cloth feels the ball"
    );
}
