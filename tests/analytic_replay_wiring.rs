//! Degenerate-input oracles for the wiring of `alice_physics::replay`
//! (`examples/replay_recording.rs`): `ReplayRecorder::{new, record_frame,
//! record_positions, frame_count, close}` and `ReplayPlayer::{open,
//! get_position, get_velocity, scan_positions, close}`.
//!
//! The happy-path exact round trip (hand-built positions/velocities through
//! `record_frame` / `record_positions`, read back bit-for-bit via
//! `get_position` / `get_velocity` / `scan_positions`) is covered by the
//! example and by `src/replay.rs`'s own
//! `scan_positions_matches_get_position_per_frame_and_body` unit test. This
//! file covers the inputs that are not the recorded happy path:
//!
//! * **Zero frames recorded**: `frame_count() == 0`, `scan_positions`
//!   yields nothing, `get_position`/`get_velocity` for frame 0 are `None`
//!   — there is no key at that timestamp for `AliceDB::get` to find
//!   (`src/replay.rs`'s `get_position`/`get_velocity` return `Ok(None)`
//!   whenever the underlying `db.get` call returns `None`, which is what
//!   an absent key produces).
//! * **A frame past the recorded range**: same reasoning — `get_position`/
//!   `get_velocity` return `None` for a timestamp whose keys were never
//!   written by `put_batch`.
//! * **`scan_positions` with `end_frame < start_frame`**: `src/replay.rs`
//!   documents and implements an explicit early return
//!   (`end_frame < start_frame { return Ok(Vec::new()) }`), independent of
//!   whether the body/frames exist.
//! * **`scan_positions` with `body_id >= body_count`**: same explicit
//!   early-return branch, documented as "`None` if the frame/body wasn't
//!   recorded" for the point queries and mirrored for the range query.
//! * **Mixing `record_frame` and `record_positions` on one recorder**:
//!   `ReplayRecorder::set_components` documents and implements rejecting
//!   this with `io::ErrorKind::InvalidInput` ("do not mix") once the first
//!   call has fixed the per-body component count.
//! * **`body_count == 0`**: the dense-key-fill loop in `record_frame`/
//!   `record_positions` is `for i in 0..self.body_count`, i.e. empty; nothing
//!   is written, no arithmetic on `body_count - 1` is reached (the
//!   `body_id >= body_count` guards in `get_position`/`get_velocity`/
//!   `scan_positions` are unconditionally true for `body_count == 0`, so
//!   they return before any such subtraction). Measured via `catch_unwind`
//!   rather than assumed.
//! * **Extreme position/velocity magnitude**: `RigidBody::position`/
//!   `velocity` are `Vec3Fix` (`Fix128`, 64.64 fixed point); `record_frame`
//!   converts each component with `Fix128::to_f32`, documented in
//!   `src/math.rs` as `(hi as f64 + lo / 2^64) as f32` — i.e. for an
//!   integer `Fix128` (`lo == 0`, as built by `Vec3Fix::from_int`) this is
//!   `hi as f64 as f32`. This file recomputes that same cast chain
//!   independently (never by calling `to_f32`/`to_f64`) as the closed-form
//!   expected value, for an `i64` magnitude (`~2e9` / `~-3e9`) comfortably
//!   above `f32`'s 24-bit exact-integer range (so the conversion is
//!   genuinely lossy — the test pins the documented rounding, not just
//!   "no panic") and comfortably below `2^53` (so the two-step
//!   `as f64 as f32` chain used here cannot double-round relative to a
//!   direct `as f32`, because the first step loses no precision — i64::MAX
//!   itself is `catch_unwind`-measured separately for the "does not panic"
//!   property only, since above `2^53` the two-step chain's result is not
//!   guaranteed identical to a direct single-step cast).
//!
//! `ReplayRecorder::close`/`ReplayPlayer::close` both take `self` by value,
//! so "double close" and "write/read after close" are prevented by
//! ownership at compile time, not by a runtime `Err`/panic — there is no
//! degenerate-input test for either here because the type system makes
//! the input impossible to construct.

#![cfg(all(feature = "std", feature = "replay"))]

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn world_with_bodies(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    for _ in 0..n {
        world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    }
    world
}

// ---------------------------------------------------------------------------
// Zero frames recorded
// ---------------------------------------------------------------------------

#[test]
fn zero_frames_recorded_reads_back_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty");
    let recorder = ReplayRecorder::new(&path, 2).unwrap();
    assert_eq!(
        recorder.frame_count(),
        0,
        "no record_frame/record_positions calls were made"
    );
    recorder.close().unwrap();

    let player = ReplayPlayer::open(&path, 2).unwrap();
    assert_eq!(
        player.get_position(0, 0).unwrap(),
        None,
        "frame 0 was never written"
    );
    assert_eq!(player.get_velocity(0, 0).unwrap(), None);
    assert!(
        player.scan_positions(0, 0, 100).unwrap().is_empty(),
        "no frames exist to scan"
    );
    player.close().unwrap();
}

// ---------------------------------------------------------------------------
// A frame past the recorded range
// ---------------------------------------------------------------------------

#[test]
fn frame_past_recorded_range_is_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short");
    let mut world = world_with_bodies(1);
    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    for k in 0..3i64 {
        world.bodies[0].position = Vec3Fix::from_int(k, 0, 0);
        recorder.record_frame(&world).unwrap();
    }
    assert_eq!(recorder.frame_count(), 3);
    recorder.close().unwrap();

    let player = ReplayPlayer::open(&path, 1).unwrap();
    // frames 0, 1, 2 exist...
    for k in 0..3u64 {
        assert!(
            player.get_position(k, 0).unwrap().is_some(),
            "frame {k} was recorded"
        );
    }
    // ...frame 3 and beyond were never written.
    assert_eq!(
        player.get_position(3, 0).unwrap(),
        None,
        "frame 3 is past the 3 recorded frames (0..=2)"
    );
    assert_eq!(player.get_velocity(3, 0).unwrap(), None);
    assert_eq!(player.get_position(1_000_000, 0).unwrap(), None);
    player.close().unwrap();
}

// ---------------------------------------------------------------------------
// scan_positions degenerate ranges/bodies
// ---------------------------------------------------------------------------

#[test]
fn scan_positions_reversed_range_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reversed");
    let mut world = world_with_bodies(1);
    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    for k in 0..5i64 {
        world.bodies[0].position = Vec3Fix::from_int(k, 0, 0);
        recorder.record_frame(&world).unwrap();
    }
    recorder.close().unwrap();

    let player = ReplayPlayer::open(&path, 1).unwrap();
    // sanity: the forward range is non-empty
    assert_eq!(player.scan_positions(0, 0, 4).unwrap().len(), 5);
    // end_frame < start_frame: documented early return, independent of what
    // is actually recorded at those frames.
    assert!(
        player.scan_positions(0, 4, 0).unwrap().is_empty(),
        "end_frame < start_frame must short-circuit to empty"
    );
    assert!(player.scan_positions(0, 3, 2).unwrap().is_empty());
    player.close().unwrap();
}

#[test]
fn scan_positions_out_of_range_body_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oob_body");
    let mut world = world_with_bodies(2);
    let mut recorder = ReplayRecorder::new(&path, 2).unwrap();
    for k in 0..3i64 {
        world.bodies[0].position = Vec3Fix::from_int(k, 0, 0);
        world.bodies[1].position = Vec3Fix::from_int(-k, 0, 0);
        recorder.record_frame(&world).unwrap();
    }
    recorder.close().unwrap();

    let player = ReplayPlayer::open(&path, 2).unwrap();
    assert!(
        player.scan_positions(2, 0, 2).unwrap().is_empty(),
        "body_id 2 is out of range for a 2-body recording (indices 0, 1)"
    );
    assert!(player.scan_positions(999, 0, 2).unwrap().is_empty());
    assert_eq!(
        player.get_position(0, 2).unwrap(),
        None,
        "same guard, point query"
    );
    assert_eq!(player.get_velocity(0, 2).unwrap(), None);
    player.close().unwrap();
}

// ---------------------------------------------------------------------------
// `ReplayPlayer::open`'s `body_count` argument is only a fallback for a
// missing `replay_layout`; once the layout exists it governs, and this
// scene deliberately passes a *mismatched*, non-default argument so a
// mutation that makes `open` ignore the layout (and use the argument
// unconditionally) is observable.
// ---------------------------------------------------------------------------

#[test]
fn open_with_mismatched_body_count_argument_is_governed_by_the_recorded_layout() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatched_open");
    let mut world = world_with_bodies(2);
    let mut recorder = ReplayRecorder::new(&path, 2).unwrap();
    for k in 0..3i64 {
        world.bodies[0].position = Vec3Fix::from_int(k, 0, 0);
        world.bodies[1].position = Vec3Fix::from_int(-k, 0, 0);
        recorder.record_frame(&world).unwrap();
    }
    recorder.close().unwrap();

    // recorded with body_count = 2; open with a deliberately different,
    // non-default argument (5) -- the layout file written by the recorder
    // must win.
    let player = ReplayPlayer::open(&path, 5).unwrap();
    assert_eq!(
        player.body_count(),
        2,
        "replay_layout (written by the recorder) must govern over open()'s argument"
    );
    // body 2, 3, 4 do not exist in the recording, regardless of the 5 passed
    // to open()
    for body in 2..5usize {
        assert_eq!(player.get_position(0, body).unwrap(), None);
    }
    // body 0 and 1 still read back exactly what was recorded
    for k in 0..3u64 {
        let (x0, ..) = player.get_position(k, 0).unwrap().expect("recorded");
        let (x1, ..) = player.get_position(k, 1).unwrap().expect("recorded");
        assert_eq!(x0, k as f32);
        assert_eq!(x1, -(k as f32));
    }
    player.close().unwrap();
}

// ---------------------------------------------------------------------------
// Mixing record_frame and record_positions on one recorder
// ---------------------------------------------------------------------------

#[test]
fn mixing_record_frame_then_record_positions_errs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_a");
    let world = world_with_bodies(1);
    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    recorder.record_frame(&world).unwrap();
    let err = recorder.record_positions(&world).unwrap_err();
    assert_eq!(
        err.kind(),
        io::ErrorKind::InvalidInput,
        "record_frame (6 components) then record_positions (3) must be rejected, not silently \
         re-laid-out"
    );
}

#[test]
fn mixing_record_positions_then_record_frame_errs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_b");
    let world = world_with_bodies(1);
    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    recorder.record_positions(&world).unwrap();
    let err = recorder.record_frame(&world).unwrap_err();
    assert_eq!(
        err.kind(),
        io::ErrorKind::InvalidInput,
        "the mix check is symmetric"
    );
    // the first (successful) call still counts, the rejected call does not
    assert_eq!(recorder.frame_count(), 1);
}

#[test]
fn repeating_the_same_mode_is_not_a_mix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("same_mode");
    let world = world_with_bodies(1);
    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    for _ in 0..3 {
        recorder.record_frame(&world).unwrap();
    }
    assert_eq!(recorder.frame_count(), 3);
    recorder.close().unwrap();
}

// ---------------------------------------------------------------------------
// body_count == 0
// ---------------------------------------------------------------------------

#[test]
fn zero_body_count_does_not_panic_and_records_nothing_observable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero_bodies");
    let world = world_with_bodies(0);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut recorder = ReplayRecorder::new(&path, 0).unwrap();
        recorder.record_frame(&world).unwrap();
        recorder.record_frame(&world).unwrap();
        assert_eq!(recorder.frame_count(), 2);
        recorder.close().unwrap();

        let player = ReplayPlayer::open(&path, 0).unwrap();
        assert_eq!(player.get_position(0, 0).unwrap(), None);
        assert!(player.scan_positions(0, 0, 10).unwrap().is_empty());
        player.close().unwrap();
    }));
    assert!(
        result.is_ok(),
        "body_count == 0 must not panic anywhere in the record/playback path: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Extreme magnitude
// ---------------------------------------------------------------------------

/// `src/math.rs`'s documented `Fix128::to_f32` law for an integer value
/// (`lo == 0`): `(hi as f64) as f32`. Computed here independently of
/// calling `to_f32`/`to_f64`.
fn integer_fix128_to_f32_closed_form(hi: i64) -> f32 {
    (hi as f64) as f32
}

#[test]
fn extreme_but_f64_exact_magnitude_round_trips_to_the_documented_f32_rounding() {
    // Magnitudes chosen deliberately above f32's 24-bit exact-integer range
    // (so the Fix128 -> f32 conversion is genuinely lossy, not merely
    // "large") and below 2^53 (so `hi as f64` loses nothing, meaning the
    // two-step `as f64 as f32` chain used both here and by `to_f32` cannot
    // disagree with a direct single-step `as f32` cast via double rounding).
    const POS_HI: i64 = 2_000_000_011;
    const VEL_HI: i64 = -3_000_000_029;
    assert!(POS_HI.unsigned_abs() > (1i64 << 24) as u64);
    assert!(POS_HI.unsigned_abs() < (1i64 << 53) as u64);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extreme");
    let mut world = world_with_bodies(1);
    world.bodies[0].position = Vec3Fix::from_int(POS_HI, -POS_HI, 0);
    world.bodies[0].velocity = Vec3Fix::from_int(VEL_HI, 0, -VEL_HI);

    let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
    recorder.record_frame(&world).unwrap();
    recorder.close().unwrap();

    let player = ReplayPlayer::open(&path, 1).unwrap();
    let (x, y, z) = player.get_position(0, 0).unwrap().expect("recorded");
    let (vx, vy, vz) = player.get_velocity(0, 0).unwrap().expect("recorded");
    player.close().unwrap();

    let want_pos_x = integer_fix128_to_f32_closed_form(POS_HI);
    let want_pos_y = integer_fix128_to_f32_closed_form(-POS_HI);
    let want_vel_x = integer_fix128_to_f32_closed_form(VEL_HI);
    let want_vel_z = integer_fix128_to_f32_closed_form(-VEL_HI);

    // the conversion really is lossy at this magnitude -- otherwise the test
    // would not be exercising the documented rounding at all
    assert_ne!(
        want_pos_x as f64, POS_HI as f64,
        "sanity: f32 cannot hold this integer exactly"
    );

    assert_eq!(
        x, want_pos_x,
        "lossy Fix128->f32 rounding must survive the lossless DB round trip bit-for-bit"
    );
    assert_eq!(y, want_pos_y);
    assert_eq!(z, 0.0);
    assert_eq!(vx, want_vel_x);
    assert_eq!(vy, 0.0);
    assert_eq!(vz, want_vel_z);
}

#[test]
fn i64_extreme_magnitude_does_not_panic() {
    // i64::MAX / i64::MIN are above 2^53, so the two-step `as f64 as f32`
    // chain is not guaranteed to match a direct single-step cast (possible
    // double rounding) -- this test measures only the documented "does not
    // panic" property, not bit-exactness.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64_extreme");
    let mut world = world_with_bodies(1);
    world.bodies[0].position = Vec3Fix::from_int(i64::MAX, i64::MIN, 0);
    world.bodies[0].velocity = Vec3Fix::from_int(i64::MIN, 0, i64::MAX);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut recorder = ReplayRecorder::new(&path, 1).unwrap();
        recorder.record_frame(&world).unwrap();
        recorder.close().unwrap();
        let player = ReplayPlayer::open(&path, 1).unwrap();
        let pos = player.get_position(0, 0).unwrap().expect("recorded");
        let vel = player.get_velocity(0, 0).unwrap().expect("recorded");
        player.close().unwrap();
        (pos, vel)
    }));
    let (pos, vel) = result.expect("i64::MAX/MIN magnitudes must not panic anywhere in the path");
    assert!(pos.0.is_finite() && pos.1.is_finite() && pos.2.is_finite());
    assert!(vel.0.is_finite() && vel.1.is_finite() && vel.2.is_finite());
}
