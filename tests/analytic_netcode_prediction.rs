//! Oracles for client-side prediction and server reconciliation.
//!
//! # What is measured
//!
//! Prediction state here is a plain integer and an input adds its `action` to it,
//! so every expected value is a sum written from the definition: after
//! reconciling with an authoritative snapshot at tick `t`, the corrected head is
//! `authoritative.state + Σ action` over the buffered inputs with tick `> t`.
//!
//! The silent failure the unchecked `reconcile` has is an *incomplete* history: when
//! the buffer's ring has dropped the inputs between the authoritative tick and the
//! oldest one it still holds, or when the buffer itself skips a tick, replaying
//! "every buffered input" applies the wrong inputs and returns a wrong state without
//! a word. `reconcile_checked` must report that instead.
//!
//! Author: Moroya Sakamoto

use alice_physics::netcode_prediction::{
    reconcile, reconcile_checked, PredictedInput, PredictionBuffer, ReconcileError, Snapshot,
};

fn step(state: i64, input: PredictedInput<i64>) -> i64 {
    state + input.action
}

/// A buffer holding the inputs `actions[i]` at ticks `first + i`, with the
/// snapshots a client would have predicted (running sum from `start`).
fn buffer_of(
    first: u64,
    actions: &[i64],
    start: i64,
    capacity: usize,
) -> PredictionBuffer<i64, i64> {
    let mut buffer = PredictionBuffer::new(capacity);
    let mut state = start;
    for (i, &a) in actions.iter().enumerate() {
        let tick = first + i as u64;
        state += a;
        buffer.push(PredictedInput { tick, action: a }, Snapshot { tick, state });
    }
    buffer
}

fn auth(tick: u64, state: i64) -> Snapshot<i64> {
    Snapshot { tick, state }
}

/// The head after reconciling is the authoritative state plus the actions of the
/// ticks after it, and the buffer keeps exactly those inputs.
#[test]
fn reconcile_replays_the_inputs_after_the_authoritative_tick() {
    // ticks 1..=6 with actions 1,2,3,4,5,6; the server's state at tick 3 is 100 (not
    // the client's predicted 6): the corrected head is 100 + 4 + 5 + 6.
    let mut b = buffer_of(1, &[1, 2, 3, 4, 5, 6], 0, 16);
    let head = reconcile(auth(3, 100), &mut b, step);
    assert_eq!(
        head,
        Snapshot {
            tick: 6,
            state: 115
        }
    );
    assert_eq!(b.head_snapshot(), Some(head));
    let ticks: Vec<u64> = b.inputs().iter().map(|i| i.tick).collect();
    assert_eq!(ticks, vec![4, 5, 6], "the acknowledged inputs are gone");
}

/// `drop_acknowledged` drops ticks `<= authoritative`, no more and no fewer.
#[test]
fn drop_acknowledged_keeps_exactly_the_ticks_after_it() {
    let mut b = buffer_of(10, &[1, 1, 1, 1], 0, 16); // ticks 10, 11, 12, 13
    b.drop_acknowledged(9); // before all: nothing dropped
    assert_eq!(b.len(), 4);
    b.drop_acknowledged(10); // tick 10 is acknowledged (<=)
    assert_eq!(b.inputs()[0].tick, 11);
    assert_eq!(b.len(), 3);
    b.drop_acknowledged(12);
    assert_eq!(b.inputs()[0].tick, 13);
    b.drop_acknowledged(99); // after all: everything dropped
    assert!(b.is_empty());
    assert_eq!(b.head_snapshot(), None);
}

/// An authoritative snapshot newer than everything buffered is the new head, and the
/// buffer is emptied.
#[test]
fn a_newer_authoritative_snapshot_empties_the_buffer() {
    let mut b = buffer_of(1, &[1, 2, 3], 0, 8);
    let head = reconcile(auth(9, -7), &mut b, step);
    assert_eq!(head, auth(9, -7));
    assert!(b.is_empty());
}

/// The ring drops the oldest entry first and keeps the order.
#[test]
fn the_ring_drops_the_oldest_input_first() {
    let b = buffer_of(1, &[1, 2, 3, 4, 5], 0, 3);
    let ticks: Vec<u64> = b.inputs().iter().map(|i| i.tick).collect();
    assert_eq!(ticks, vec![3, 4, 5]);
    assert_eq!(b.head_snapshot().map(|s| s.tick), Some(5));
}

/// With a complete history the checked reconcile agrees with the unchecked one.
#[test]
fn the_checked_reconcile_agrees_when_the_history_is_complete() {
    let mut a = buffer_of(1, &[3, 1, 4, 1, 5], 0, 16);
    let mut b = a.clone();
    let plain = reconcile(auth(2, 50), &mut a, step);
    let checked = reconcile_checked(auth(2, 50), &mut b, step).expect("complete");
    assert_eq!(plain, checked);
    assert_eq!(plain.state, 50 + 4 + 1 + 5);
}

/// The server's snapshot is older than the oldest input the ring still holds: the
/// inputs for the ticks in between are lost, so the replay would be wrong; the checked
/// reconcile says which input it needed.
#[test]
fn the_checked_reconcile_reports_inputs_the_ring_has_dropped() {
    // ticks 1..=8 into a ring of 4 keep ticks 5..=8; the server answers for tick 2.
    let mut b = buffer_of(1, &[1; 8], 0, 4);
    let wrong = reconcile(auth(2, 0), &mut b.clone(), step);
    assert_eq!(
        wrong.state, 4,
        "the unchecked reconcile replays four inputs and says 4"
    );
    let err = reconcile_checked(auth(2, 0), &mut b, step).expect_err("ticks 3 and 4 are lost");
    assert_eq!(
        err,
        ReconcileError::MissingInputs {
            needed: 3,
            oldest: 5
        }
    );
    // The buffer is left as it was: nothing was dropped or rewritten.
    assert_eq!(b.len(), 4);
}

/// A tick missing in the middle of the buffer is reported too.
#[test]
fn the_checked_reconcile_reports_a_gap_inside_the_buffer() {
    let mut b = PredictionBuffer::new(8);
    for &(tick, action) in &[(1u64, 1i64), (2, 1), (4, 1), (5, 1)] {
        b.push(PredictedInput { tick, action }, Snapshot { tick, state: 0 });
    }
    let err = reconcile_checked(auth(0, 0), &mut b, step).expect_err("tick 3 is missing");
    assert_eq!(
        err,
        ReconcileError::MissingInputs {
            needed: 3,
            oldest: 4
        }
    );
}

/// The authoritative snapshot is exactly one tick before the oldest input, or older
/// inputs were already acknowledged: no gap.
#[test]
fn the_checked_reconcile_accepts_a_history_that_starts_right_after() {
    let mut b = buffer_of(5, &[2, 2], 0, 8); // ticks 5, 6
    let head = reconcile_checked(auth(4, 10), &mut b, step).expect("contiguous");
    assert_eq!(head, Snapshot { tick: 6, state: 14 });
    // And when the server is ahead of all of it, there is nothing to replay: ok.
    let mut b = buffer_of(5, &[2, 2], 0, 8);
    let head = reconcile_checked(auth(6, 10), &mut b, step).expect("nothing to replay");
    assert_eq!(head, auth(6, 10));
}
