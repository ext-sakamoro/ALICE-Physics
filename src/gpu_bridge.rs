//! GPU solver bridge trait for external offload backends.
//!
//! Downstream crates (e.g. ALICE-TRT with its compute-shader Fix128
//! kernel) can implement [`GpuSolverBridge`] and inject the runtime
//! into the sub-stepping TGS dispatch layer. Default `alice-physics`
//! builds do not enable any bridge — the CPU-native
//! [`crate::articulation::FeatherstoneSolver`] and TGS pipeline
//! remain the only code path, so the trait is a pure opt-in
//! extension surface (no runtime overhead when the feature is off).
//!
//! # Feature gate
//!
//! Enable with `--features gpu-solver-bridge`. The trait definition
//! itself does not link any GPU backend so consumers can stub it out
//! for testing purposes.
//!
//! # Determinism contract
//!
//! Any implementation must guarantee bit-exact equivalence with the
//! CPU-side solver for a supplied [`DiffFixture`]. When the backend
//! diverges (e.g. GPU vendor-specific rounding, subgroup reduce
//! ordering) it must surface the mismatch via [`GpuDivergence`]. See
//! the private
//! `deterministic-physics-lockstep-discipline` skill §1 経路 3
//! (SIMD / subgroup reduce ordering) and §1 経路 5 (traversal
//! ordering) for the reference reasoning.

use crate::math::Fix128;

/// A reference input fixture used to certify bit-exact equivalence
/// between the CPU-side solver and an external GPU backend. Consumers
/// typically wire this against the golden pairs stored alongside the
/// FFI byte-for-byte replay contract (see
/// `alice_physics::ffi::AliceVec3Fix128Raw`).
#[derive(Debug, Clone)]
pub struct DiffFixture {
    /// Human-readable label carried through in error reports.
    pub description: &'static str,
    /// Allowed absolute deviation in the CPU vs GPU output.
    /// `Fix128::ZERO` requests strict byte-for-byte equality.
    pub tolerance: Fix128,
}

/// Report returned when a GPU backend fails the bit-exact contract.
#[derive(Debug, Clone)]
pub struct GpuDivergence {
    /// Which output component diverged (e.g. `"body[3].position.y"`).
    pub axis: &'static str,
    /// CPU-side reference `hi` half.
    pub cpu_hi: i64,
    /// CPU-side reference `lo` half.
    pub cpu_lo: u64,
    /// GPU-side observed `hi` half.
    pub gpu_hi: i64,
    /// GPU-side observed `lo` half.
    pub gpu_lo: u64,
}

/// External GPU offload backend for the sub-stepping TGS solver.
///
/// Implementations are responsible for the entire island lifecycle
/// (upload → dispatch → readback) and must expose a diff-check API
/// that the host uses to gate production use.
///
/// # Skeleton
///
/// The trait is intentionally minimal for the initial rollout; each
/// method is documented with the deterministic behaviour expected
/// from the implementer even though the associated compute shaders
/// live in the downstream crate.
pub trait GpuSolverBridge {
    /// Upload an island (positions + velocities as
    /// `[Fix128; 3]` per body) into the GPU-side buffer. The
    /// implementer must preserve the caller's index ordering.
    fn send_island(&mut self, positions: &[[Fix128; 3]], velocities: &[[Fix128; 3]]);

    /// Run `iters` PGS iterations against the currently uploaded
    /// island using the given time step. Iterations must run in
    /// index-ordered dispatch to satisfy the skill §1 経路 5 contract.
    fn dispatch_iterations(&mut self, iters: u32, dt: Fix128);

    /// Read back the post-solve positions and velocities from the
    /// GPU buffer into the caller-provided slices. Element indices
    /// must line up with the [`Self::send_island`] upload order.
    fn recv_island(&self, positions: &mut [[Fix128; 3]], velocities: &mut [[Fix128; 3]]);

    /// Assert that this backend produces byte-identical results to
    /// the CPU-side reference for `fixture`. Implementers should
    /// return `Ok(())` on success and `Err(GpuDivergence { .. })`
    /// pinpointing the mismatched axis / value pair on failure.
    ///
    /// This is the production gate that Unity / UE5 host bindings
    /// consult before enabling the GPU path at run time.
    fn assert_bit_exact_vs_cpu(&self, fixture: &DiffFixture) -> Result<(), GpuDivergence>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the diff types can be constructed with the
    /// documented invariants (zero tolerance means strict equality).
    #[test]
    fn diff_fixture_defaults_are_strict() {
        let f = DiffFixture {
            description: "gravity_fall_60_steps",
            tolerance: Fix128::ZERO,
        };
        assert_eq!(f.description, "gravity_fall_60_steps");
        assert_eq!(f.tolerance.hi, 0);
        assert_eq!(f.tolerance.lo, 0);
    }

    /// A trivial stub implementer so the trait shape is exercised at
    /// compile time; a real backend will replace these with real
    /// dispatch code.
    struct StubBridge;
    impl GpuSolverBridge for StubBridge {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
    }

    #[test]
    fn stub_bridge_reports_no_divergence() {
        let bridge = StubBridge;
        let f = DiffFixture {
            description: "identity",
            tolerance: Fix128::ZERO,
        };
        assert!(bridge.assert_bit_exact_vs_cpu(&f).is_ok());
    }
}
