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
use crate::solver::ContactConstraint;

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
/// # Pipeline stages
///
/// The trait covers two pipeline stages:
///
/// - **Integrate + distance** (v0.7+): `send_island` uploads the
///   body positions and velocities, `dispatch_iterations` runs the
///   integrate + floor + distance projection pipeline, and
///   `recv_island` reads back the updated state.
/// - **Contact solve** (v0.9+, opt-in): `send_contact_constraints`
///   uploads the contact list, `send_body_state` uploads the body
///   positions + inverse masses used by the PGS iteration,
///   `dispatch_contact_solve_iteration` runs one sequential
///   Gauss-Seidel pass, and `recv_contact_constraints` +
///   `recv_body_positions` read back the updated warm-start state
///   and position corrections. Every contact-solve method has a
///   `panic!` default implementation so pre-v0.9 backends that
///   don't implement contact solve stay compilable but fail
///   fast when a caller tries to route contact solve through
///   them.
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

    // ---- v0.9.0: contact-solve pipeline (opt-in, default panics) ----

    /// Upload the contact constraint list into the GPU-side buffer.
    /// Element indices must line up with the caller's
    /// `PhysicsWorld::contact_constraints` slot ordering so
    /// subsequent [`Self::recv_contact_constraints`] can write the
    /// updated `cached_lambda` warm-start values back in place.
    ///
    /// # Default
    ///
    /// The default implementation panics with a "not implemented by
    /// this backend" message. Backends that support the v0.9+
    /// contact solve pipeline must override.
    fn send_contact_constraints(&mut self, _constraints: &[ContactConstraint]) {
        panic!("send_contact_constraints not implemented by this GpuSolverBridge backend");
    }

    /// Upload the per-body state that the contact solve iteration
    /// reads. `positions[i]` is the position of body id `i` (indexed
    /// by the `body_a` / `body_b` fields of the constraints uploaded
    /// via [`Self::send_contact_constraints`]);
    /// `inv_masses[i]` is body `i`'s inverse mass, with
    /// `Fix128::ZERO` marking a static body.
    ///
    /// # Default
    ///
    /// Panics; backends that support contact solve must override.
    fn send_body_state(&mut self, _positions: &[[Fix128; 3]], _inv_masses: &[Fix128]) {
        panic!("send_body_state not implemented by this GpuSolverBridge backend");
    }

    /// Run one sequential Gauss-Seidel PGS contact-solve iteration
    /// against the currently uploaded constraints and body state.
    /// Updates in-place the GPU-side `cached_lambda` on each
    /// constraint and applies position corrections to bodies.
    /// Callers loop this method for multiple iterations, mirroring
    /// the CPU `for _ in 0..config.iterations { ... }` pattern
    /// inside `PhysicsWorld::substep`.
    ///
    /// `warm_start_factor` is `SolverConfig::warm_start_factor`
    /// (default 0.85 per `ContactConstraint::new`).
    ///
    /// # Default
    ///
    /// Panics; backends that support contact solve must override.
    fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
        panic!("dispatch_contact_solve_iteration not implemented by this GpuSolverBridge backend");
    }

    /// Read back the post-solve `cached_lambda` warm-start state
    /// into the caller-provided constraint slice. Only the
    /// `cached_lambda` field of each element is updated; other
    /// fields (body indices, contact geometry, friction,
    /// restitution) are left untouched — those are the caller's
    /// inputs that the kernel does not modify.
    ///
    /// # Default
    ///
    /// Panics; backends that support contact solve must override.
    fn recv_contact_constraints(&self, _constraints: &mut [ContactConstraint]) {
        panic!("recv_contact_constraints not implemented by this GpuSolverBridge backend");
    }

    /// Read back the post-solve body positions into the
    /// caller-provided slice. Element indices must line up with
    /// the upload order of [`Self::send_body_state`].
    ///
    /// # Default
    ///
    /// Panics; backends that support contact solve must override.
    fn recv_body_positions(&self, _positions: &mut [[Fix128; 3]]) {
        panic!("recv_body_positions not implemented by this GpuSolverBridge backend");
    }

    // ---- v0.12.0: joint-solve pipeline (opt-in, default panics) ----

    /// Upload the joint list into the GPU-side buffer. Element
    /// indices must line up with the caller's `PhysicsWorld::joints`
    /// slot ordering.
    ///
    /// Not every joint variant is required to be handled by every
    /// backend; ALICE-TRT v3.1.0 supports only [`Joint::Ball`] and
    /// fails-fast with a `panic!` on other variants. Future backend
    /// revisions add Hinge / Fixed / D6 support.
    ///
    /// # Default
    ///
    /// Panics; backends that support joint solve must override.
    fn send_joints(&mut self, _joints: &[crate::joint::Joint]) {
        panic!("send_joints not implemented by this GpuSolverBridge backend");
    }

    /// Upload the per-body rotations that the joint-solve iteration
    /// reads. `rotations[i]` is body `i`'s orientation quaternion,
    /// stored as `[x, y, z, w]` matching the CPU `QuatFix` layout.
    /// Callers who use only the contact-solve pipeline (positions +
    /// inv_masses) do not need to upload rotations.
    ///
    /// # Default
    ///
    /// Panics; backends that support joint solve must override.
    fn send_body_rotations(&mut self, _rotations: &[[Fix128; 4]]) {
        panic!("send_body_rotations not implemented by this GpuSolverBridge backend");
    }

    /// Run one joint-solve pass over the currently uploaded joint
    /// list, applying position corrections to bodies in place. Unlike
    /// contact solve (which iterates N times per substep), joint
    /// solve runs exactly once per substep — the correction is a
    /// Baumgarte-stabilised projection, not a Gauss-Seidel iteration.
    /// `dt` is the substep length used to compute the compliance
    /// stabilisation term `compliance / (dt * dt)`.
    ///
    /// # Default
    ///
    /// Panics; backends that support joint solve must override.
    fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {
        panic!("dispatch_joint_solve_iteration not implemented by this GpuSolverBridge backend");
    }
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
