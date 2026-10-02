//! A game / robotics host driving `PhysicsWorld` through its public API
//!
//! Walks the body builders, the per-body mutators a host reaches through
//! `get_body_mut`, force fields, the contact / trigger event pipeline with a
//! pre-solve hook and a contact modifier, per-body collision radii and
//! filters, an SDF floor, joints and constraint batches, sleeping, body
//! removal, and (behind `gpu-solver-bridge`) the GPU bridge lifecycle.
//! Every line it prints is a quantity the closed-form oracles in
//! `tests/analytic_world_api.rs` pin, so the printout doubles as a
//! human-readable run of those oracles.
//!
//! ```bash
//! cargo run --example world_api_tour --features std
//! cargo run --example world_api_tour --features std,gpu-solver-bridge
//! ```

use alice_physics::collider::Contact;
use alice_physics::joint::BallJoint;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{
    CollisionFilter, ContactEventType, DistanceConstraint, ForceField, ForceFieldInstance, Joint,
    PhysicsMaterial,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// `1/64 s`: exact in `Fix128`, so one frame of eight substeps advances
/// positions by exactly `8 * v * (1/512)` (`tests/analytic_world_api.rs`).
fn frame_dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// Zero gravity, otherwise the default solver configuration.
fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    }
}

/// Two radius-2 spheres whose centres are 3 apart: penetration depth 1.
fn two_overlapping_spheres(world: &mut PhysicsWorld) -> (usize, usize) {
    let r = Fix128::from_int(2);
    let a = world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r);
    let b = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
        r,
    );
    (a, b)
}

fn xs(world: &PhysicsWorld, a: usize, b: usize) -> (f64, f64) {
    (
        world.bodies[a].position.x.to_f64(),
        world.bodies[b].position.x.to_f64(),
    )
}

/// A contact modifier that halves every penetration depth (a soft-contact
/// host rule) and counts how often the solver consulted it.
struct HalfDepth {
    calls: Arc<AtomicUsize>,
}

impl ContactModifier for HalfDepth {
    fn modify_contact(
        &self,
        _body_a: usize,
        _body_b: usize,
        contact: &mut Contact,
        _friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        contact.depth = contact.depth.half();
        true
    }
}

fn builders_and_mutators() {
    println!("[world_api] -- body builders and host-side mutators --");
    let mut world = PhysicsWorld::new(PhysicsConfig::default());

    // A projectile launched at 512 m/s along +x with a 180-degree yaw, full
    // gravity, a custom friction coefficient and angular damping.
    let yaw_180 = QuatFix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let shell = world.add_body(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(4))
            .with_velocity(Vec3Fix::from_int(512, 0, 0))
            .with_rotation(yaw_180)
            .with_gravity_scale(Fix128::ONE)
            .with_friction(Fix128::from_ratio(1, 4))
            .with_angular_damping(Fix128::ZERO),
    );
    // A buoy that ignores gravity entirely.
    let buoy = world.add_body(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE)
            .with_gravity_scale(Fix128::ZERO),
    );
    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    let platform = world.add_body(RigidBody::new_kinematic(Vec3Fix::from_int(5, 0, 0)));

    for (name, idx) in [
        ("shell", shell),
        ("buoy", buoy),
        ("anchor", anchor),
        ("platform", platform),
    ] {
        let body = world.get_body(idx).expect("index came from add_body");
        println!(
            "[world_api] {name}: is_dynamic={} is_kinematic={} mass={} speed={}",
            body.is_dynamic(),
            body.is_kinematic(),
            body.mass().to_f64(),
            body.speed().to_f64(),
        );
    }

    // Host-side impulses through the mutable accessor: a thruster force for
    // one frame, a reaction-wheel torque, then direct state writes.
    let dt = frame_dt();
    if let Some(body) = world.get_body_mut(shell) {
        body.add_force(Vec3Fix::from_int(0, 0, 256), dt);
        body.add_torque(Vec3Fix::from_int(0, 4, 0), dt);
    }
    if let Some(body) = world.get_body_mut(buoy) {
        body.set_velocity(Vec3Fix::from_int(3, 4, 0));
        body.set_angular_velocity(Vec3Fix::from_int(0, 0, 1));
        body.set_rotation(QuatFix::IDENTITY);
    }
    let shell_body = world.get_body(shell).expect("still present");
    println!(
        "[world_api] shell after add_force/add_torque: v=({}, {}, {}) w_y={}",
        shell_body.velocity.x.to_f64(),
        shell_body.velocity.y.to_f64(),
        shell_body.velocity.z.to_f64(),
        shell_body.angular_velocity.y.to_f64(),
    );
    let buoy_body = world.get_body(buoy).expect("still present");
    println!(
        "[world_api] buoy after set_velocity: speed={} w_z={}",
        buoy_body.speed().to_f64(),
        buoy_body.angular_velocity.z.to_f64(),
    );

    world.step(dt);
    let shell_body = world.get_body(shell).expect("still present");
    let buoy_body = world.get_body(buoy).expect("still present");
    println!(
        "[world_api] after one frame: shell x={} y={} w_y={} | buoy y={} (gravity_scale 0)",
        shell_body.position.x.to_f64(),
        shell_body.position.y.to_f64(),
        shell_body.angular_velocity.y.to_f64(),
        buoy_body.position.y.to_f64(),
    );
    println!(
        "[world_api] out-of-range get_body(99) is None: {}",
        world.get_body(99).is_none()
    );
}

fn force_fields() {
    println!("[world_api] -- force fields --");
    let mut world = PhysicsWorld::new(weightless());
    let crate_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));
    let wind = world.add_force_field(ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(8),
    }));
    let dt = frame_dt();
    world.step(dt);
    let x1 = world.bodies[crate_idx].position.x.to_f64();
    let removed = world.remove_force_field(wind).is_some();
    world.step(dt);
    let x2 = world.bodies[crate_idx].position.x.to_f64();
    println!("[world_api] wind on for 1 frame: x={x1}; removed={removed}; coasting frame: x={x2}");
    println!(
        "[world_api] remove_force_field again -> None: {}",
        world.remove_force_field(wind).is_none()
    );
}

fn contacts_hooks_and_events() {
    println!("[world_api] -- contact pipeline: hooks, modifiers, events --");
    let dt = frame_dt();

    // Plain contact: the two spheres separate by substeps * depth / 2 each.
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = two_overlapping_spheres(&mut world);
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    let events = world.drain_contact_events();
    let begin = events
        .iter()
        .filter(|e| e.event_type == ContactEventType::Begin)
        .count();
    println!(
        "[world_api] plain contact: x_a={xa} x_b={xb} events={} (begin={begin}) depth={} drained_again={}",
        events.len(),
        events.first().map_or(0.0, |e| e.depth.to_f64()),
        world.drain_contact_events().len(),
    );

    // Pre-solve hook that vetoes everything (one-way platform style) and
    // counts calls: nothing moves, the hook is consulted every substep.
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = two_overlapping_spheres(&mut world);
    let veto_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&veto_calls);
    world.add_pre_solve_hook(Box::new(move |_a, _b, _c: &Contact| {
        counter.fetch_add(1, Ordering::SeqCst);
        false
    }));
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!(
        "[world_api] veto hook: x_a={xa} x_b={xb} hook_calls={}",
        veto_calls.load(Ordering::SeqCst)
    );
    world.clear_pre_solve_hooks();
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!("[world_api] after clear_pre_solve_hooks: x_a={xa} x_b={xb}");

    // Contact modifier halving the depth: half the separation per substep.
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = two_overlapping_spheres(&mut world);
    let calls = Arc::new(AtomicUsize::new(0));
    world.add_contact_modifier(Box::new(HalfDepth {
        calls: Arc::clone(&calls),
    }));
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!(
        "[world_api] half-depth modifier: x_a={xa} x_b={xb} modifier_calls={}",
        calls.load(Ordering::SeqCst)
    );
    world.clear_contact_modifiers();
    // Put the pair back where it started and run the plain solver again.
    for (idx, x) in [(a, 0), (b, 3)] {
        if let Some(body) = world.get_body_mut(idx) {
            body.set_position(Vec3Fix::from_int(x, 0, 0));
            body.set_velocity(Vec3Fix::ZERO);
        }
    }
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!("[world_api] after clear_contact_modifiers (pair reset): x_a={xa} x_b={xb}");

    // Collision filter: a body that collides with nothing never contacts.
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = two_overlapping_spheres(&mut world);
    world.set_body_filter(a, CollisionFilter::NONE);
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!(
        "[world_api] filter NONE: x_a={xa} x_b={xb} contact_events={}",
        world.contact_events().len()
    );

    // Sensor: overlap reports a trigger, no response, enter once then exit.
    let mut world = PhysicsWorld::new(weightless());
    let r = Fix128::from_int(2);
    let zone = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_sensor(true),
        r,
    );
    let visitor = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
        r,
    );
    world.step(dt);
    let entered = world.trigger_events().iter().filter(|t| t.entered).count();
    let (xz, xv) = xs(&world, zone, visitor);
    let drained = world.drain_trigger_events().len();
    println!(
        "[world_api] sensor frame 1: trigger_events={drained} entered={entered} x_zone={xz} x_visitor={xv}"
    );
    world.step(dt);
    let persisting = world.trigger_events().len();
    if let Some(body) = world.get_body_mut(visitor) {
        body.set_position(Vec3Fix::from_int(10, 0, 0));
    }
    world.step(dt);
    let exits = world
        .drain_trigger_events()
        .iter()
        .filter(|t| !t.entered)
        .count();
    println!("[world_api] sensor frame 2: new trigger events={persisting}; after leaving: exit events={exits}");
}

fn radii_and_materials() {
    println!("[world_api] -- per-body collision radius and material --");
    let dt = frame_dt();
    let mut world = PhysicsWorld::new(weightless());
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::ONE,
    ));
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!("[world_api] no radius: x_a={xa} x_b={xb}");
    world.set_body_collision_radius(a, Fix128::from_int(2));
    world.set_body_collision_radius(b, Fix128::from_int(2));
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!("[world_api] radius 2 on both: x_a={xa} x_b={xb}");

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = two_overlapping_spheres(&mut world);
    world.clear_body_collision_radius(b);
    world.step(dt);
    let (xa, xb) = xs(&world, a, b);
    println!("[world_api] radius cleared on b: x_a={xa} x_b={xb}");

    let rubber = world.material_table.register(PhysicsMaterial::new(
        0,
        Fix128::ONE,
        Fix128::from_ratio(1, 2),
    ));
    world.set_body_material(a, rubber);
    let combined = world.combined_material(a, b);
    println!(
        "[world_api] rubber vs default (average rule): friction={} restitution={}",
        combined.friction.to_f64(),
        combined.restitution.to_f64(),
    );
}

fn sdf_floor() {
    println!("[world_api] -- SDF floor --");
    let dt = frame_dt();
    let plane = || Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0)));
    let mut world = PhysicsWorld::new(weightless());
    let ball = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 4), Fix128::ZERO),
        Fix128::ONE,
    ));
    let floor = world.add_sdf_collider(SdfCollider::new_static(
        plane(),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    world.step(dt);
    let y_default = world.bodies[ball].position.y.to_f64();

    let mut world = PhysicsWorld::new(weightless());
    let ball2 = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 4), Fix128::ZERO),
        Fix128::ONE,
    ));
    let floor2 = world.add_sdf_collider(SdfCollider::new_static(
        plane(),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    world.set_sdf_collision_radius(Fix128::from_ratio(3, 4));
    world.step(dt);
    let y_wide = world.bodies[ball2].position.y.to_f64();
    let removed = world.remove_sdf_collider(floor2).is_some();
    let y_before = world.bodies[ball2].position.y.to_f64();
    world.step(dt);
    let y_after = world.bodies[ball2].position.y.to_f64();
    println!(
        "[world_api] ball at y=0.25: radius 0.5 -> y={y_default}; radius 0.75 -> y={y_wide}; floor {floor} removed={removed}: y {y_before} -> {y_after}"
    );
}

fn joints_and_batches() {
    println!("[world_api] -- joints and constraint batches --");
    let mut world = PhysicsWorld::new(weightless());
    let ids: Vec<usize> = (0..4)
        .map(|i| {
            world.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(i, 0, 0),
                Fix128::ONE,
            ))
        })
        .collect();
    let j = world.add_joint(Joint::Ball(BallJoint::new(
        ids[0],
        ids[1],
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    let joints_after_add = world.joint_count();
    let removed = world.remove_joint(j).is_some();
    let removed_twice = world.remove_joint(j).is_none();
    println!(
        "[world_api] joints: after add={joints_after_add} removed={removed} after remove={} remove again -> None: {removed_twice}",
        world.joint_count()
    );

    for w in ids.windows(2) {
        world.add_distance_constraint(DistanceConstraint {
            body_a: w[0],
            body_b: w[1],
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::ONE,
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        });
    }
    world.rebuild_batches();
    println!(
        "[world_api] chain of 3 distance constraints over 4 bodies: num_batches={}",
        world.num_batches()
    );
}

fn sleeping_and_removal() {
    println!("[world_api] -- sleeping, waking, removal --");
    let dt = frame_dt();
    let mut world = PhysicsWorld::new(weightless());
    let resting = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let wall = world.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)));
    let frames = world.islands.config.frames_to_sleep;
    for _ in 0..frames {
        world.step(dt);
    }
    println!(
        "[world_api] after {frames} idle frames: resting sleeping={} wall sleeping={} active_body_count={} of {}",
        world.is_sleeping(resting),
        world.is_sleeping(wall),
        world.active_body_count(),
        world.body_count(),
    );
    world.wake_body(resting);
    println!(
        "[world_api] wake_body: sleeping={} active_body_count={}",
        world.is_sleeping(resting),
        world.active_body_count()
    );
    let removed = world.remove_body(wall);
    println!(
        "[world_api] remove_body(wall): returned static={} bodies={} active_body_count={} get_body(wall) none={}",
        removed.is_some_and(|b| b.is_static()),
        world.body_count(),
        world.active_body_count(),
        world.get_body(wall).is_none(),
    );
}

#[cfg(feature = "gpu-solver-bridge")]
mod gpu {
    //! GPU bridge lifecycle as a host would drive it. `TracingBridge` only
    //! counts the uploads and dispatches the world routes through it and
    //! leaves the uploaded state untouched — it is not a solver (ALICE-TRT
    //! provides the real one), so the bodies it is given do not separate.
    use super::{frame_dt, two_overlapping_spheres, weightless, xs};
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::math::Fix128;
    use alice_physics::solver::{ContactConstraint, PhysicsWorld};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    pub(super) struct TracingBridge {
        pub(super) dispatches: Arc<AtomicUsize>,
    }

    impl GpuSolverBridge for TracingBridge {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
        }
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    pub(super) fn tour() {
        println!("[world_api] -- gpu solver bridge --");
        let dt = frame_dt();
        let dispatches = Arc::new(AtomicUsize::new(0));

        // Explicit routing with a host-owned bridge.
        let mut bridge = TracingBridge {
            dispatches: Arc::clone(&dispatches),
        };
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = two_overlapping_spheres(&mut world);
        world.substep_with_bridge(&mut bridge, dt);
        let after_substep = dispatches.load(Ordering::SeqCst);
        world.step_with_bridge(&mut bridge, dt);
        let (xa, xb) = xs(&world, a, b);
        println!(
            "[world_api] host-owned bridge: substep dispatches={after_substep} step dispatches={} x_a={xa} x_b={xb}",
            dispatches.load(Ordering::SeqCst) - after_substep
        );

        // Installed bridge: plain `step` routes through it until taken.
        let dispatches = Arc::new(AtomicUsize::new(0));
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = two_overlapping_spheres(&mut world);
        world.set_gpu_solver_bridge(Some(Box::new(TracingBridge {
            dispatches: Arc::clone(&dispatches),
        })));
        let installed = world.gpu_solver_bridge_installed();
        world.step(dt);
        let routed = dispatches.load(Ordering::SeqCst);
        let taken = world.take_gpu_solver_bridge().is_some();
        let taken_twice = world.take_gpu_solver_bridge().is_none();
        world.step(dt);
        let (xa, xb) = xs(&world, a, b);
        println!(
            "[world_api] installed={installed} routed dispatches={routed} taken={taken} installed after take={} take again -> None: {taken_twice} cpu frame: x_a={xa} x_b={xb}",
            world.gpu_solver_bridge_installed()
        );
    }
}

fn main() {
    builders_and_mutators();
    force_fields();
    contacts_hooks_and_events();
    radii_and_materials();
    sdf_floor();
    joints_and_batches();
    sleeping_and_removal();
    #[cfg(feature = "gpu-solver-bridge")]
    gpu::tour();
}
