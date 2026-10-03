//! Prediction, reconciliation and rollback
//!
//! A server runs the full deterministic simulation. A client predicts its own player
//! with a light model (position moves by `movement · speed · dt` each tick), keeps
//! its unacknowledged inputs in a `PredictionBuffer`, and when the server's snapshot
//! for an older tick arrives it replays the inputs after it with `reconcile_checked`.
//! With no gravity and no collisions the light model is exact, so the corrected
//! prediction equals the server's body to the last bit of rounding; the printout
//! shows the difference.
//!
//! The second half is the heavy path: the server snapshots a frame, runs on, rolls
//! back to the snapshot and replays the same inputs, and gets the same checksum.
//!
//! ```bash
//! cargo run --example rollback_netcode --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::netcode::{DeterministicSimulation, FrameInput, NetcodeConfig};
use alice_physics::netcode_prediction::{
    reconcile_checked, PredictedInput, PredictionBuffer, ReconcileError, Snapshot,
};
use alice_physics::solver::{RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// Player 0 walks along +x for 30 frames, then along +z.
fn input(frame: u64) -> FrameInput {
    let movement = if frame <= 30 {
        v3(1.0, 0.0, 0.0)
    } else {
        v3(0.0, 0.0, 1.0)
    };
    FrameInput::new(0).with_movement(movement)
}

fn main() {
    let config = NetcodeConfig {
        physics: SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        },
        max_snapshots: 20,
        ..NetcodeConfig::default()
    };
    let speed = Fix128::from_int(5); // the default applicator's movement speed
    let mut server = DeterministicSimulation::new(config);
    let body = server.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    server.assign_player_body(0, body);
    let dt = server.dt();

    // The client: predict each tick, remember the input and the predicted position.
    // The server's answer for tick 40 reaches it at tick 50, ten ticks late.
    let mut buffer: PredictionBuffer<FrameInput, Vec3Fix> = PredictionBuffer::new(64);
    let mut predicted = Vec3Fix::ZERO;
    let model = move |state: Vec3Fix, input: PredictedInput<FrameInput>| {
        state + input.action.movement * speed * dt
    };
    let mut server_at_40 = Vec3Fix::ZERO;
    for frame in 1..=60u64 {
        predicted = model(
            predicted,
            PredictedInput {
                tick: frame,
                action: input(frame),
            },
        );
        buffer.push(
            PredictedInput {
                tick: frame,
                action: input(frame),
            },
            Snapshot {
                tick: frame,
                state: predicted,
            },
        );
        server.advance_frame(&[input(frame)]);
        if frame == 40 {
            server_at_40 = server.world.bodies[body].position;
        }
        if frame == 50 {
            let authoritative = Snapshot {
                tick: 40,
                state: server_at_40,
            };
            let corrected = reconcile_checked(authoritative, &mut buffer, model)
                .expect("the buffer holds every input after tick 40");
            println!(
                "snapshot of tick 40 reconciled at tick 50: {} inputs replayed, head at tick {}",
                buffer.inputs().len(),
                corrected.tick
            );
            predicted = corrected.state;
        }
    }
    let head = buffer.head_snapshot().expect("inputs are buffered");
    let server_at = server.world.bodies[body].position;
    println!(
        "client head (tick {}): ({:.9}, {:.9}, {:.9})",
        head.tick,
        head.state.x.to_f64(),
        head.state.y.to_f64(),
        head.state.z.to_f64()
    );
    println!(
        "server body        : ({:.9}, {:.9}, {:.9})",
        server_at.x.to_f64(),
        server_at.y.to_f64(),
        server_at.z.to_f64()
    );
    println!(
        "difference: {:.2e} m (the server steps eight substeps of dt/8)",
        (head.state - server_at).length().to_f64()
    );

    // A server that is too slow: the client's ring has dropped the inputs between
    // the snapshot and its oldest entry. The checked reconcile says so.
    let mut small: PredictionBuffer<FrameInput, Vec3Fix> = PredictionBuffer::new(4);
    for frame in 1..=8u64 {
        small.push(
            PredictedInput {
                tick: frame,
                action: input(frame),
            },
            Snapshot {
                tick: frame,
                state: Vec3Fix::ZERO,
            },
        );
    }
    let late = Snapshot {
        tick: 2,
        state: Vec3Fix::ZERO,
    };
    match reconcile_checked(late, &mut small, model) {
        Err(ReconcileError::MissingInputs { needed, oldest }) => println!(
            "a snapshot for tick 2 against a ring that starts at tick {oldest}: input {needed} is lost"
        ),
        Ok(_) => println!("unexpectedly reconciled"),
        Err(_) => println!("reconcile refused"),
    }

    // The heavy path: snapshot, run on, roll back, replay the same inputs.
    let mut rewound = DeterministicSimulation::new(NetcodeConfig {
        physics: SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        },
        max_snapshots: 20,
        ..NetcodeConfig::default()
    });
    let rb = rewound.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    rewound.assign_player_body(0, rb);
    let mut checksums = Vec::new();
    for frame in 1..=60u64 {
        checksums.push(rewound.advance_frame(&[input(frame)]));
        if frame == 30 {
            rewound.save_snapshot();
        }
    }
    println!("snapshots held: {}", rewound.snapshot_count());
    let at_30 = rewound
        .get_snapshot(30)
        .expect("saved at frame 30")
        .checksum;
    assert!(rewound.load_snapshot(30));
    let replay: Vec<_> = (31..=60u64)
        .map(|frame| rewound.advance_frame(&[input(frame)]))
        .collect();
    println!(
        "rolled back to frame 30 (checksum {:#x}) and replayed 30 frames: identical = {}",
        at_30.0,
        replay[..] == checksums[30..]
    );
}
