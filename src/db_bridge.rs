//! ALICE-DB bridge: Physics state snapshot persistence
//!
//! Records per-step physics metrics (kinetic energy, body count, contact
//! count) as ALICE-DB exact series: every step reads back the `f32` bits it
//! was written with, whatever the spacing of the steps.

use crate::semantics::{check_peer, PeerMismatch};
use alice_db::AliceDB;
use std::io;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::path::Path;

/// Exact series (`alice_db::Series`) the sink writes: one record per step,
/// keyed by the step, holding the value's `f32` bits (no model fit, so every
/// step reads back the value written at it, however far apart the steps are)
const ENERGY: &str = "alice-physics/energy";
const BODIES: &str = "alice-physics/bodies";
const CONTACTS: &str = "alice-physics/contacts";

/// Error of a constructor of an ALICE-DB bridge ([`PhysicsMetricsSink`],
/// `crate::replay`): the linked `alice-db` is not the one this build expects
/// ([`PeerMismatch`], naming both identifiers), or the database failed.
#[derive(Debug)]
pub enum BridgeError {
    /// The linked `alice-db` exports another identifier than
    /// [`crate::PHYSICS_PEER_IDS`] expects: no value is exchanged with it.
    Peer(PeerMismatch),
    /// The database could not be opened or read.
    Io(io::Error),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Peer(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Peer(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<io::Error> for BridgeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<PeerMismatch> for BridgeError {
    fn from(e: PeerMismatch) -> Self {
        Self::Peer(e)
    }
}

/// The handshake of every ALICE-DB bridge: the linked `alice-db`'s identifier
/// against the one this build expects of it.
pub(crate) fn alice_db_handshake(bridge: &'static str) -> Result<(), PeerMismatch> {
    check_peer(bridge, "alice-db", &alice_db::SEMANTICS_ID)
}

/// Physics metrics sink backed by ALICE-DB.
///
/// Stores per-step simulation metrics as three exact series of one database:
/// total kinetic energy, active body count and contact count per step. A
/// value reads back with the bits it was written with, at any step (sparse
/// or negative steps included).
pub struct PhysicsMetricsSink {
    db: AliceDB,
}

impl PhysicsMetricsSink {
    /// Open the physics metrics database at the given directory.
    ///
    /// # Errors
    ///
    /// [`BridgeError::Peer`] when the linked `alice-db` is not the one this
    /// build expects (checked first, nothing is opened);
    /// [`BridgeError::Io`] when the database cannot be opened.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self, BridgeError> {
        alice_db_handshake("db_bridge")?;
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            db: AliceDB::open(dir.join("metrics"))?,
        })
    }

    /// Keep the metrics in process memory (no filesystem; available on
    /// `wasm32-unknown-unknown`).
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub fn in_memory() -> Result<Self, BridgeError> {
        alice_db_handshake("db_bridge")?;
        Ok(Self {
            db: AliceDB::in_memory(alice_db::StorageConfig::default())?,
        })
    }

    /// Record a simulation step's metrics.
    pub fn record_step(
        &self,
        step: i64,
        kinetic_energy: f32,
        body_count: f32,
        contact_count: f32,
    ) -> io::Result<()> {
        self.db.series(ENERGY)?.put_f32(step, kinetic_energy)?;
        self.db.series(BODIES)?.put_f32(step, body_count)?;
        self.db.series(CONTACTS)?.put_f32(step, contact_count)
    }

    /// Record only kinetic energy.
    pub fn record_energy(&self, step: i64, energy: f32) -> io::Result<()> {
        self.db.series(ENERGY)?.put_f32(step, energy)
    }

    /// Query energy history for a step range (inclusive on both ends).
    pub fn query_energy(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db.series(ENERGY)?.scan_f32(start, end)
    }

    /// Query body count history.
    pub fn query_bodies(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db.series(BODIES)?.scan_f32(start, end)
    }

    /// Query contact count history.
    pub fn query_contacts(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db.series(CONTACTS)?.scan_f32(start, end)
    }

    /// Flush the database.
    pub fn flush(&self) -> io::Result<()> {
        self.db.flush_blobs()?;
        self.db.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_record_and_query() {
        let dir = tempdir().unwrap();
        let sink = PhysicsMetricsSink::open(dir.path()).unwrap();

        for step in 0..100 {
            sink.record_step(step, step as f32 * 0.1, 10.0, step as f32 % 5.0)
                .unwrap();
        }
        sink.flush().unwrap();

        let energy = sink.query_energy(0, 99).unwrap();
        assert!(!energy.is_empty());
    }

    /// `query_bodies` / `query_contacts` return exactly the recorded
    /// `(step, value)` pairs of the requested inclusive range, in step order,
    /// from their own series (not the energy one); `record_energy` writes
    /// only the energy series.
    #[test]
    fn query_bodies_contacts_and_record_energy_hit_their_own_series() {
        let dir = tempdir().unwrap();
        let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
        for step in 0..10 {
            sink.record_step(
                step,
                1.5 * step as f32,
                100.0 + step as f32,
                7.0 - step as f32,
            )
            .unwrap();
        }
        // energy-only record at the next step: bodies / contacts must not gain a
        // row (each metric is its own exact series)
        sink.record_energy(10, 123.5).unwrap();
        sink.flush().unwrap();

        let bodies = sink.query_bodies(3, 6).unwrap();
        assert_eq!(bodies, vec![(3, 103.0), (4, 104.0), (5, 105.0), (6, 106.0)]);
        let contacts = sink.query_contacts(0, 2).unwrap();
        assert_eq!(contacts, vec![(0, 7.0), (1, 6.0), (2, 5.0)]);
        assert!(sink.query_bodies(10, 10).unwrap().is_empty());
        assert!(sink.query_contacts(10, 10).unwrap().is_empty());
        let energy = sink.query_energy(10, 10).unwrap();
        assert_eq!(energy, vec![(10, 123.5)]);
        // ranges are independent per database: the energy scan of 0..2 is the step formula
        assert_eq!(
            sink.query_energy(0, 2).unwrap(),
            vec![(0, 0.0), (1, 1.5), (2, 3.0)]
        );
    }
}
