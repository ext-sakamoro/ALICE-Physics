//! The GPU-bridge stepping path applies rolling and torsional friction exactly
//! as `step` does.
//!
//! A bridge replaces the position-level contact solve only; the velocity pass
//! that carries rolling resistance and torsional friction runs on the CPU in
//! both paths. Measured against a reference backend that performs the crate's
//! CPU contact correction on what it receives (accumulated multiplier
//! `λ += max(depth − λ, 0)`, positions moved along the normal by the inverse
//! mass shares, the same fixed-point operations in the same order):
//!
//! - a ball rolling and spinning about the normal on a flat static box, with
//!   nonzero rolling (`c`) and torsional (`μ_t`) coefficients: every frame,
//!   position, velocity, rotation and angular velocity of `step_with_bridge`
//!   equal those of `step` bit for bit;
//! - the scene is not vacuous: after the run, the ball's speed and spin differ
//!   from the run with both coefficients zero.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "std", feature = "gpu-solver-bridge"))]

use alice_physics_core::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const R: f64 = 0.5;
const FRAMES: usize = 120;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// The CPU contact correction of one solver iteration, run on the uploaded
/// buffers.
#[derive(Default)]
struct CpuContactReference {
    island_p: Vec<[Fix128; 3]>,
    island_v: Vec<[Fix128; 3]>,
    constraints: Vec<ContactConstraint>,
    positions: Vec<[Fix128; 3]>,
    inv_masses: Vec<Fix128>,
}

fn to_v(p: [Fix128; 3]) -> Vec3Fix {
    Vec3Fix::new(p[0], p[1], p[2])
}

fn from_v(v: Vec3Fix) -> [Fix128; 3] {
    [v.x, v.y, v.z]
}

impl GpuSolverBridge for CpuContactReference {
    fn semantics_id(&self) -> [u8; 32] {
        alice_physics_core::PHYSICS_SEMANTICS_ID
    }
    fn send_island(&mut self, p: &[[Fix128; 3]], v: &[[Fix128; 3]]) {
        self.island_p = p.to_vec();
        self.island_v = v.to_vec();
    }
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, p: &mut [[Fix128; 3]], v: &mut [[Fix128; 3]]) {
        p.copy_from_slice(&self.island_p);
        v.copy_from_slice(&self.island_v);
    }
    fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
        self.constraints = c.to_vec();
    }
    fn send_body_state(&mut self, positions: &[[Fix128; 3]], inv_masses: &[Fix128]) {
        self.positions = positions.to_vec();
        self.inv_masses = inv_masses.to_vec();
    }
    fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
        for c in &mut self.constraints {
            let contact = c.contact;
            if contact.depth <= Fix128::ZERO {
                continue;
            }
            let (ia, ib) = (self.inv_masses[c.body_a], self.inv_masses[c.body_b]);
            let w_sum = ia + ib;
            if w_sum.is_zero() {
                continue;
            }
            let inv_w_sum = Fix128::ONE / w_sum;
            let dlambda = contact.depth - c.cached_lambda;
            if dlambda > Fix128::ZERO {
                c.cached_lambda = c.cached_lambda + dlambda;
                let correction = contact.normal * dlambda;
                if !ia.is_zero() {
                    let p = to_v(self.positions[c.body_a]) + correction * (ia * inv_w_sum);
                    self.positions[c.body_a] = from_v(p);
                }
                if !ib.is_zero() {
                    let p = to_v(self.positions[c.body_b]) - correction * (ib * inv_w_sum);
                    self.positions[c.body_b] = from_v(p);
                }
            }
        }
    }
    fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
        for (dst, src) in c.iter_mut().zip(&self.constraints) {
            dst.cached_lambda = src.cached_lambda;
        }
    }
    fn recv_body_positions(&self, positions: &mut [[Fix128; 3]]) {
        positions.copy_from_slice(&self.positions);
    }
}

/// a flat static box (top at `y = 0`) and a solid ball on it rolling along
/// `+x` at 2 m/s and spinning about `y` at 6 rad/s
fn scene(c: f64, mu_t: f64) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(v3(0.0, -0.5, 0.0)));
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(50.0, 0.5, 50.0),
        }
    ));
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, R, 0.0), Fix128::ONE), fx(R));
    let inv_i = 1.0 / (0.4 * R * R);
    w.bodies[ball].inv_inertia = v3(inv_i, inv_i, inv_i);
    w.bodies[ball].velocity = v3(2.0, 0.0, 0.0);
    w.bodies[ball].angular_velocity = v3(0.0, 6.0, -2.0 / R);
    let id = w.material_table.register(
        PhysicsMaterial::new(0, fx(0.8), Fix128::ZERO)
            .with_rolling_friction(fx(c))
            .with_torsional_friction(fx(mu_t)),
    );
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

fn body_state(w: &PhysicsWorld, i: usize) -> [Fix128; 13] {
    let b = &w.bodies[i];
    [
        b.position.x,
        b.position.y,
        b.position.z,
        b.velocity.x,
        b.velocity.y,
        b.velocity.z,
        b.rotation.x,
        b.rotation.y,
        b.rotation.z,
        b.rotation.w,
        b.angular_velocity.x,
        b.angular_velocity.y,
        b.angular_velocity.z,
    ]
}

/// oracle: bridged and CPU steps agree bit for bit every frame with nonzero
/// rolling and torsional coefficients, and the coefficients change the run.
#[test]
fn the_bridged_step_applies_rolling_and_torsional_friction_like_the_cpu_step() {
    let (c, mu_t) = (0.05, 0.01);
    let (mut cpu, ball) = scene(c, mu_t);
    let (mut bridged, _) = scene(c, mu_t);
    let mut bridge = CpuContactReference::default();
    for frame in 1..=FRAMES {
        cpu.step(dt());
        bridged
            .step_with_bridge(&mut bridge, dt())
            .expect("the bridge computes under this build's semantics");
        assert_eq!(
            body_state(&cpu, ball),
            body_state(&bridged, ball),
            "frame {frame}: the bridged step diverged from the CPU step"
        );
    }
    let (mut free, _) = scene(0.0, 0.0);
    for _ in 0..FRAMES {
        free.step(dt());
    }
    let (a, b) = (&cpu.bodies[ball], &free.bodies[ball]);
    assert!(
        a.velocity.x < b.velocity.x && a.angular_velocity.y.abs() < b.angular_velocity.y.abs(),
        "the coefficients must slow the roll and the spin: v {} vs {}, ω_y {} vs {}",
        a.velocity.x.to_f64(),
        b.velocity.x.to_f64(),
        a.angular_velocity.y.to_f64(),
        b.angular_velocity.y.to_f64()
    );
}
