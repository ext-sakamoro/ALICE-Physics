//! ALICE-DB bridge: Physics state snapshot persistence
//!
//! Records physics simulation state (body positions, velocities, energies)
//! into ALICE-DB time-series for replay, debugging, and analysis.

use alice_db::AliceDB;
use std::io;
use std::path::Path;

/// Physics metrics sink backed by ALICE-DB.
///
/// Stores per-step simulation metrics in separate DB instances:
/// - `energy_db`: Total kinetic energy per step
/// - `bodies_db`: Active body count per step
/// - `contacts_db`: Contact count per step
pub struct PhysicsMetricsSink {
    energy_db: AliceDB,
    bodies_db: AliceDB,
    contacts_db: AliceDB,
}

impl PhysicsMetricsSink {
    /// Open physics metrics databases at the given directory.
    pub fn open<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            energy_db: AliceDB::open(dir.join("energy"))?,
            bodies_db: AliceDB::open(dir.join("bodies"))?,
            contacts_db: AliceDB::open(dir.join("contacts"))?,
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
        self.energy_db.put(step, kinetic_energy)?;
        self.bodies_db.put(step, body_count)?;
        self.contacts_db.put(step, contact_count)?;
        Ok(())
    }

    /// Record only kinetic energy.
    pub fn record_energy(&self, step: i64, energy: f32) -> io::Result<()> {
        self.energy_db.put(step, energy)
    }

    /// Query energy history for a step range.
    pub fn query_energy(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.energy_db.scan(start, end)
    }

    /// Query body count history.
    pub fn query_bodies(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.bodies_db.scan(start, end)
    }

    /// Query contact count history.
    pub fn query_contacts(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.contacts_db.scan(start, end)
    }

    /// Flush all databases.
    pub fn flush(&self) -> io::Result<()> {
        self.energy_db.flush()?;
        self.bodies_db.flush()?;
        self.contacts_db.flush()
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
}
