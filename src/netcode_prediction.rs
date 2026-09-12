//! Client-side prediction primitives with server reconciliation.
//!
//! Complements [`crate::netcode`] (checksum-based rollback for the
//! full simulation) with a lighter-weight input-buffer / snapshot-
//! reconciliation core that individual game systems (character
//! controllers, projectiles) can wire up without invoking the entire
//! world rewind.
//!
//! # Usage pattern
//!
//! ```text
//! 1. Client records inputs into a `PredictionBuffer` each tick and
//!    stores a matching `Snapshot` produced by applying that input.
//! 2. Client sends inputs to the authoritative server.
//! 3. Server replies with the authoritative `Snapshot` for the tick
//!    it just simulated.
//! 4. Client calls `reconcile` with the server snapshot + the list of
//!    inputs it has retained since that tick. `reconcile` rewinds the
//!    local snapshot to the authoritative one, replays each stored
//!    input, and returns the corrected head snapshot.
//! ```
//!
//! The trait interface is generic over the game's input, state, and
//! integration function so callers can plug in any deterministic
//! step.

use core::fmt::Debug;

/// A single client-side input tagged with the tick it applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictedInput<A>
where
    A: Copy,
{
    /// Simulation tick the input applies to.
    pub tick: u64,
    /// Player action / control vector.
    pub action: A,
}

/// A snapshot of the predicted state at a particular tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snapshot<S>
where
    S: Copy,
{
    /// Simulation tick this snapshot represents.
    pub tick: u64,
    /// Concrete state (position, velocity, health — game-specific).
    pub state: S,
}

/// Rolling buffer of unacknowledged inputs + matching predicted
/// snapshots, sized so the oldest entry is dropped when the ring is
/// full.
#[derive(Debug, Clone)]
pub struct PredictionBuffer<A, S>
where
    A: Copy,
    S: Copy,
{
    inputs: Vec<PredictedInput<A>>,
    snapshots: Vec<Snapshot<S>>,
    capacity: usize,
}

impl<A, S> PredictionBuffer<A, S>
where
    A: Copy + Debug,
    S: Copy + Debug,
{
    /// Construct a buffer that retains at most `capacity` entries.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            inputs: Vec::with_capacity(capacity),
            snapshots: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Number of buffered entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// True when no inputs are buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Push a `(input, snapshot)` pair. Drops the oldest entry when
    /// the ring is full. The snapshot MUST be the result of applying
    /// `input` to the previous state — the buffer does not verify
    /// this.
    pub fn push(&mut self, input: PredictedInput<A>, snapshot: Snapshot<S>) {
        debug_assert_eq!(
            input.tick, snapshot.tick,
            "input and snapshot ticks must match"
        );
        if self.inputs.len() == self.capacity {
            self.inputs.remove(0);
            self.snapshots.remove(0);
        }
        self.inputs.push(input);
        self.snapshots.push(snapshot);
    }

    /// Head (newest) snapshot, if any.
    #[must_use]
    pub fn head_snapshot(&self) -> Option<Snapshot<S>> {
        self.snapshots.last().copied()
    }

    /// All buffered inputs.
    #[must_use]
    pub fn inputs(&self) -> &[PredictedInput<A>] {
        &self.inputs
    }

    /// Drop every entry with `tick <= authoritative_tick`. Used after
    /// reconcile to discard now-obsolete predictions.
    pub fn drop_acknowledged(&mut self, authoritative_tick: u64) {
        let cutoff = self
            .inputs
            .iter()
            .position(|i| i.tick > authoritative_tick)
            .unwrap_or(self.inputs.len());
        self.inputs.drain(..cutoff);
        self.snapshots.drain(..cutoff);
    }
}

/// Rewind a prediction buffer to `authoritative` and replay every
/// buffered input with a caller-supplied integration function,
/// returning the corrected head snapshot.
///
/// - `authoritative`: last authoritative snapshot received from the
///   server. Its tick MUST equal the tick just before the first
///   buffered input.
/// - `buffer`: rolling buffer of unacknowledged `(input, snapshot)`
///   pairs. `reconcile` drops acknowledged entries and replaces the
///   remaining snapshots with the replayed ones.
/// - `step`: closure that applies a single input to a state and
///   returns the updated state.
///
/// Returns the head snapshot after replay. If the authoritative
/// snapshot is newer than every buffered entry, the buffer is
/// cleared and the authoritative snapshot is returned directly.
pub fn reconcile<A, S, F>(
    authoritative: Snapshot<S>,
    buffer: &mut PredictionBuffer<A, S>,
    mut step: F,
) -> Snapshot<S>
where
    A: Copy + Debug,
    S: Copy + Debug,
    F: FnMut(S, PredictedInput<A>) -> S,
{
    buffer.drop_acknowledged(authoritative.tick);
    if buffer.is_empty() {
        return authoritative;
    }
    let mut state = authoritative.state;
    let mut new_snapshots = Vec::with_capacity(buffer.len());
    let inputs_copy: Vec<_> = buffer.inputs.to_vec();
    for input in &inputs_copy {
        state = step(state, *input);
        new_snapshots.push(Snapshot {
            tick: input.tick,
            state,
        });
    }
    buffer.snapshots = new_snapshots;
    buffer
        .head_snapshot()
        .expect("buffer non-empty by construction")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test state: a 1-D scalar position advanced by an integer
    // delta each tick.
    fn step(state: i32, input: PredictedInput<i32>) -> i32 {
        state + input.action
    }

    #[test]
    fn buffer_rejects_zero_capacity() {
        let result = std::panic::catch_unwind(|| PredictionBuffer::<i32, i32>::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn buffer_pushes_up_to_capacity_and_drops_oldest() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(3);
        for t in 0..5 {
            buffer.push(
                PredictedInput { tick: t, action: 1 },
                Snapshot { tick: t, state: 0 },
            );
        }
        assert_eq!(buffer.len(), 3);
        // Ticks 0, 1 should have been dropped.
        assert_eq!(buffer.inputs()[0].tick, 2);
    }

    #[test]
    fn drop_acknowledged_removes_stale_entries() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(8);
        for t in 0..5 {
            buffer.push(
                PredictedInput { tick: t, action: 1 },
                Snapshot { tick: t, state: 0 },
            );
        }
        buffer.drop_acknowledged(2);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.inputs()[0].tick, 3);
    }

    #[test]
    fn reconcile_with_empty_buffer_returns_authoritative() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(4);
        let authoritative = Snapshot {
            tick: 10,
            state: 99,
        };
        let result = reconcile(authoritative, &mut buffer, step);
        assert_eq!(result, authoritative);
    }

    #[test]
    fn reconcile_replays_buffered_inputs() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(8);
        for t in 5..10 {
            buffer.push(
                PredictedInput { tick: t, action: 1 },
                Snapshot { tick: t, state: 0 },
            );
        }
        let authoritative = Snapshot { tick: 4, state: 0 };
        let head = reconcile(authoritative, &mut buffer, step);
        // Authoritative was 0, then five +1 inputs → 5.
        assert_eq!(head.state, 5);
        assert_eq!(head.tick, 9);
    }

    #[test]
    fn reconcile_discards_acknowledged_before_replay() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(8);
        for t in 0..10 {
            buffer.push(
                PredictedInput { tick: t, action: 1 },
                Snapshot { tick: t, state: 0 },
            );
        }
        let authoritative = Snapshot {
            tick: 5,
            state: 100,
        };
        let head = reconcile(authoritative, &mut buffer, step);
        // Ticks 6, 7, 8, 9 remain; each +1 from base 100 → 104.
        assert_eq!(head.state, 104);
        assert_eq!(head.tick, 9);
        assert_eq!(buffer.len(), 4);
    }

    #[test]
    fn buffer_head_snapshot_returns_newest() {
        let mut buffer: PredictionBuffer<i32, i32> = PredictionBuffer::new(4);
        buffer.push(
            PredictedInput { tick: 1, action: 0 },
            Snapshot { tick: 1, state: 11 },
        );
        buffer.push(
            PredictedInput { tick: 2, action: 0 },
            Snapshot { tick: 2, state: 22 },
        );
        let head = buffer.head_snapshot().unwrap();
        assert_eq!(head.tick, 2);
        assert_eq!(head.state, 22);
    }
}
