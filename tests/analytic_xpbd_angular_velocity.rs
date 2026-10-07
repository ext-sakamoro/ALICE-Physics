//! Exact angular velocity of free rigid bodies under the XPBD backend.
//!
//! A body that no constraint or contact turns during a substep must leave the
//! substep with exactly the angular velocity it was predicted with:
//! `ω = ω_pred + log(q · q_pred⁻¹) / h`, `q_pred` being the rotation the
//! substep predicted and `q` the rotation after the solve (so the second term
//! is zero for a free body). Re-deriving `ω` from `log(q · q_prev⁻¹) / h`
//! instead goes through `from_axis_angle`, a normalisation and the logarithm,
//! each of which rounds, so the low bits of `ω` changed every substep.
//!
//! This covers the bodies the gyroscopic splitting does not advance:
//! isotropic inertia (`ω × Iω = 0`) and infinite moments (`inv_inertia = 0`).
//! A body with anisotropic inertia is advanced by the splitting, whose end
//! velocity is already kept (its `ω` precesses, so it has no constant closed
//! form and is covered by `tests/analytic_gyroscopic.rs`).
//!
//! Closed form: no gravity, no damping (`damping = 1`, `v · 1` exact), no
//! contacts, so the free body's `ω` is constant: `ω_n = ω_0` bit for bit after
//! any number of frames, for any `dt` and `substeps`. The isotropic and
//! infinite-moment bodies have no torque-free precession, so this is the
//! exact solution, not only a rounding statement.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::SolverBackend;

const FRAMES: usize = 240;
const SUBSTEPS: [usize; 4] = [1, 3, 4, 8];

fn dts() -> [Fix128; 2] {
    [Fix128::from_ratio(1, 60), Fix128::from_ratio(1, 64)]
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn config(substeps: usize) -> SolverConfig {
    SolverConfig {
        substeps,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        solver_backend: SolverBackend::Xpbd,
        ..SolverConfig::default()
    }
}

/// Non-dyadic angular velocities, slow enough that `|ω| h` stays far below
/// the branch points of the logarithm.
fn omegas() -> [Vec3Fix; 3] {
    [
        Vec3Fix::new(r(4, 3), r(-16, 7), r(16, 11)),
        Vec3Fix::new(r(-13, 9), r(22, 13), r(-3, 17)),
        Vec3Fix::new(r(7, 5), r(5, 19), r(-29, 23)),
    ]
}

/// One free body at `x` (1000 m apart from the others, never in contact)
/// with inverse inertia `inv_inertia` and angular velocity `w`.
fn add_spinner(w: &mut PhysicsWorld, x: i64, inv_inertia: Vec3Fix, omega: Vec3Fix) -> usize {
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(x, 0, 0), Fix128::ONE);
    b.inv_inertia = inv_inertia;
    b.angular_velocity = omega;
    w.add_body(b)
}

fn run(inv_inertia: Vec3Fix, label: &str) {
    for dt in dts() {
        for substeps in SUBSTEPS {
            let mut w = PhysicsWorld::new(config(substeps));
            let ids: Vec<usize> = omegas()
                .iter()
                .enumerate()
                .map(|(k, &o)| add_spinner(&mut w, 1000 * k as i64, inv_inertia, o))
                .collect();
            let w0 = omegas();
            for frame in 1..=FRAMES {
                w.step(dt);
                for (k, &i) in ids.iter().enumerate() {
                    assert_eq!(
                        w.bodies[i].angular_velocity, w0[k],
                        "{label}: dt {dt:?} substeps {substeps} frame {frame} body {k}"
                    );
                }
            }
        }
    }
}

/// Isotropic inertia (`I = 2/5 m r²` of a unit sphere of radius 1/2, i.e.
/// `inv_inertia = 10` on every axis): `ω` is constant bit for bit.
#[test]
fn xpbd_free_isotropic_body_keeps_its_angular_velocity_bit_for_bit() {
    run(Vec3Fix::from_int(10, 10, 10), "isotropic");
}

/// Infinite moments (`inv_inertia = 0`): no torque can change `ω`, and the
/// splitting does not apply; `ω` is constant bit for bit.
#[test]
fn xpbd_free_body_with_infinite_moments_keeps_its_angular_velocity_bit_for_bit() {
    run(Vec3Fix::ZERO, "infinite moments");
}
