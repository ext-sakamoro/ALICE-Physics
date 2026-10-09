//! A sleeping body wakes when the SDF it rests on changes shape
//! (`SdfField::generation`), and a field that never changes leaves the
//! simulation bit for bit as before.
//!
//! Closed form of the fall after the wake: a body at rest integrates
//! `v += g·h`, `x += v·h` over the `n` substeps of the step, so after one step
//! `y = y₀ + g·h²·n(n+1)/2` (frame damping acts on the velocity only, after
//! the substeps). The counts each test compares are asserted to be non-zero.

#![cfg(feature = "std")]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
use alice_physics::solver::{Broadphase, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

/// The floor `y = height` whose height can be moved; `generation` is bumped
/// by the test when it wants the move announced.
struct MovableFloor {
    height: Arc<AtomicU32>,
    generation: Arc<AtomicU64>,
}

impl SdfField for MovableFloor {
    fn distance(&self, _x: f32, y: f32, _z: f32) -> f32 {
        y - f32::from_bits(self.height.load(Ordering::Relaxed))
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }
}

struct Handles {
    height: Arc<AtomicU32>,
    generation: Arc<AtomicU64>,
}

impl Handles {
    fn lower_to(&self, y: f32) {
        self.height.store(y.to_bits(), Ordering::Relaxed);
    }
    fn announce(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

const SUBSTEPS: usize = 8;
const G: f64 = -10.0;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// A ball of radius 1/2 resting on the movable floor `y = 0` at `x = 0`.
fn scene(broadphase: Broadphase) -> (PhysicsWorld, Handles) {
    let config = PhysicsConfig {
        substeps: SUBSTEPS,
        gravity: Vec3Fix::from_int(0, G as i64, 0),
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_broadphase(broadphase);
    w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
    let height = Arc::new(AtomicU32::new(0.0f32.to_bits()));
    let generation = Arc::new(AtomicU64::new(0));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(MovableFloor {
            height: Arc::clone(&height),
            generation: Arc::clone(&generation),
        }),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
        Fix128::ONE,
    ));
    (w, Handles { height, generation })
}

/// Step until body 0 sleeps; panics if it never does.
fn settle(w: &mut PhysicsWorld, step: fn(&mut PhysicsWorld, Fix128)) -> usize {
    for k in 0..2000 {
        if w.is_sleeping(0) {
            return k;
        }
        step(w, dt());
    }
    panic!("the ball never fell asleep");
}

fn y(w: &PhysicsWorld) -> f64 {
    w.bodies[0].position.y.to_f64()
}

type StepPath = (&'static str, fn(&mut PhysicsWorld, Fix128));

fn xpbd_paths() -> Vec<StepPath> {
    #[cfg_attr(not(feature = "parallel"), allow(unused_mut))]
    let mut paths: Vec<StepPath> = vec![("step", |w, dt| w.step(dt))];
    #[cfg(feature = "parallel")]
    paths.push(("step_parallel", |w, dt| w.step_parallel(dt)));
    paths
}

/// After the floor drops by 1 m and announces it, the next step wakes the
/// ball and it falls `g·h²·n(n+1)/2` from rest (`h = dt / n`). Both with the
/// sleep skip parking the ball (BVH broadphase) and without.
#[test]
fn an_announced_shape_change_wakes_the_sleeping_ball_and_it_falls_the_closed_form_distance() {
    let h = 1.0 / (60.0 * SUBSTEPS as f64);
    let n = SUBSTEPS as f64;
    let drop = G * h * h * n * (n + 1.0) / 2.0;
    let mut compared = 0;
    for broadphase in [Broadphase::Bvh, Broadphase::DynamicTree] {
        for (name, step) in xpbd_paths() {
            let (mut w, floor) = scene(broadphase);
            settle(&mut w, step);
            let y0 = y(&w);
            // At rest on the floor up to the f32 of the field query.
            assert!((y0 - 0.5).abs() < 1e-6, "{name}: rests at {y0}");
            // The velocity the ball fell asleep with (a residue of the f32
            // push-out, of order 1e-7) is the input of the step: y₀ + v₀·dt + drop.
            let v0 = w.bodies[0].velocity.y.to_f64();
            assert!(v0.abs() < 1e-5, "{name}: asleep with velocity {v0}");
            let expected = y0 + v0 / 60.0 + drop;
            floor.lower_to(-1.0);
            floor.announce();
            step(&mut w, dt());
            assert!(!w.is_sleeping(0), "{name} {broadphase:?}: still asleep");
            let got = y(&w);
            assert!(
                (got - expected).abs() < 1e-9,
                "{name} {broadphase:?}: y {got} vs closed form {expected}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0);
}

/// The same drop without announcing it: the ball stays asleep where it was
/// (a field whose generation does not change is never re-examined for
/// sleeping bodies, as before generations existed).
#[test]
fn an_unannounced_change_leaves_the_sleeping_ball_where_it_was() {
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, floor) = scene(Broadphase::DynamicTree);
        settle(&mut w, step);
        let before = w.bodies[0].position;
        floor.lower_to(-1.0);
        for _ in 0..30 {
            step(&mut w, dt());
        }
        assert!(w.is_sleeping(0), "{name}");
        assert_eq!(w.bodies[0].position, before, "{name}");
        runs += 1;
    }
    assert!(runs > 0);
}

/// Woken once per change: after the ball lands on the lowered floor it falls
/// asleep again (a generation that stays at its new value does not keep
/// waking it), and rests on the new floor at `y = −1 + 1/2`.
#[test]
fn after_the_wake_the_ball_settles_on_the_new_floor_and_sleeps_again() {
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, floor) = scene(Broadphase::DynamicTree);
        settle(&mut w, step);
        floor.lower_to(-1.0);
        floor.announce();
        step(&mut w, dt());
        settle(&mut w, step);
        let got = y(&w);
        assert!((got + 0.5).abs() < 1e-6, "{name}: rests at {got}");
        // and stays asleep: the change is not seen again in later steps
        for _ in 0..5 {
            step(&mut w, dt());
            assert!(w.is_sleeping(0), "{name}: woken again by the old change");
        }
        runs += 1;
    }
    assert!(runs > 0);
}

/// Adding a collider whose generation is already non-zero is not a change:
/// the sleeping ball stays asleep.
#[test]
fn adding_a_collider_with_a_nonzero_generation_wakes_nothing() {
    let mut runs = 0;
    for (name, step) in xpbd_paths() {
        let (mut w, _floor) = scene(Broadphase::DynamicTree);
        settle(&mut w, step);
        let far = MovableFloor {
            height: Arc::new(AtomicU32::new((-100.0f32).to_bits())),
            generation: Arc::new(AtomicU64::new(5)),
        };
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(far),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        step(&mut w, dt());
        assert!(w.is_sleeping(0), "{name}: woken by an added collider");
        runs += 1;
    }
    assert!(runs > 0);
}

/// TGS: the announced change wakes the ball too (it moves down in the step).
/// The ball is put to sleep with the XPBD backend, then the backend is
/// switched.
#[test]
fn tgs_wakes_the_ball_on_an_announced_change() {
    let (mut w, floor) = scene(Broadphase::DynamicTree);
    settle(&mut w, |w, dt| w.step(dt));
    w.config.solver_backend = SolverBackend::Tgs;
    let y0 = y(&w);
    floor.lower_to(-1.0);
    floor.announce();
    w.step(dt());
    assert!(!w.is_sleeping(0));
    assert!(y(&w) < y0, "TGS: {} not below {y0}", y(&w));
}

fn bits(w: &PhysicsWorld) -> Vec<[Fix128; 6]> {
    w.bodies
        .iter()
        .map(|b| {
            [
                b.position.x,
                b.position.y,
                b.position.z,
                b.velocity.x,
                b.velocity.y,
                b.velocity.z,
            ]
        })
        .collect()
}

/// A field whose generation is constant (here 7) steps bit for bit like the
/// same floor as a `ClosureSdf` (default generation 0), over a scene where
/// bodies fall, land, sleep and are parked.
#[test]
fn a_constant_generation_changes_nothing() {
    fn build(constant: bool) -> PhysicsWorld {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.set_broadphase(Broadphase::Bvh);
        w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
        let field: Box<dyn SdfField> = if constant {
            Box::new(MovableFloor {
                height: Arc::new(AtomicU32::new(0.0f32.to_bits())),
                generation: Arc::new(AtomicU64::new(7)),
            })
        } else {
            Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0)))
        };
        w.add_sdf_collider(SdfCollider::new_static(
            field,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        for k in 0..6i64 {
            w.add_body(RigidBody::new_dynamic(
                Vec3Fix::new(
                    Fix128::from_int(3 * k),
                    Fix128::from_ratio(1 + 2 * k, 2),
                    Fix128::ZERO,
                ),
                Fix128::ONE,
            ));
        }
        w
    }
    let (mut a, mut b) = (build(false), build(true));
    let mut slept = 0;
    for _ in 0..400 {
        a.step(dt());
        b.step(dt());
        slept += (0..a.bodies.len()).filter(|&i| a.is_sleeping(i)).count();
    }
    assert!(
        slept > 0,
        "no body ever slept: the scene does not test the wake"
    );
    assert_eq!(bits(&a), bits(&b));
}

/// A union of two fields changes when one part changes: the floor wrapped in
/// `SdfUnion` with a far plane still wakes the ball when it announces a drop.
#[test]
fn a_union_reports_the_change_of_a_part() {
    use alice_physics::sdf_collider::SdfUnion;
    let config = PhysicsConfig {
        substeps: SUBSTEPS,
        gravity: Vec3Fix::from_int(0, G as i64, 0),
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
    let height = Arc::new(AtomicU32::new(0.0f32.to_bits()));
    let generation = Arc::new(AtomicU64::new(0));
    let floor = MovableFloor {
        height: Arc::clone(&height),
        generation: Arc::clone(&generation),
    };
    let far = ClosureSdf::new(|x, _, _| 100.0 - x, |_, _, _| (-1.0, 0.0, 0.0));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(SdfUnion::new(floor, far)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
        Fix128::ONE,
    ));
    let handles = Handles { height, generation };
    settle(&mut w, |w, dt| w.step(dt));
    handles.lower_to(-1.0);
    handles.announce();
    w.step(dt());
    assert!(!w.is_sleeping(0), "the union hid the part's change");
}

/// A `DestructibleSdf` the test keeps a handle to while the world holds it.
struct SharedDestructible(Arc<std::sync::Mutex<alice_physics::sdf_destruction::DestructibleSdf>>);

impl SdfField for SharedDestructible {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.0.lock().expect("lock").distance(x, y, z)
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        self.0.lock().expect("lock").normal(x, y, z)
    }
    fn generation(&self) -> u64 {
        self.0.lock().expect("lock").generation()
    }
}

/// A ball asleep on a destructible floor `y = 0`; carving a crater of radius
/// 3 under it wakes it at the next step, and it falls the closed-form
/// distance `v₀·dt + g·h²·n(n+1)/2` (the crater bottom is 3 m down, out of
/// reach in one step). Without carving, the same world steps bit for bit like
/// one whose floor is a plain `ClosureSdf`.
#[test]
fn carving_a_destructible_floor_wakes_the_sleeping_ball() {
    use alice_physics::sdf_destruction::{DestructibleSdf, DestructionShape};
    fn build() -> (PhysicsWorld, Arc<std::sync::Mutex<DestructibleSdf>>) {
        let config = PhysicsConfig {
            substeps: SUBSTEPS,
            gravity: Vec3Fix::from_int(0, G as i64, 0),
            ..PhysicsConfig::default()
        };
        let mut w = PhysicsWorld::new(config);
        w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
        let floor = Arc::new(std::sync::Mutex::new(DestructibleSdf::new(Box::new(
            ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0)),
        ))));
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(SharedDestructible(Arc::clone(&floor))),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
            Fix128::ONE,
        ));
        (w, floor)
    }

    // Carved: woken, falls.
    let (mut w, floor) = build();
    settle(&mut w, |w, dt| w.step(dt));
    let (y0, v0) = (y(&w), w.bodies[0].velocity.y.to_f64());
    floor
        .lock()
        .expect("lock")
        .apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 3.0));
    w.step(dt());
    assert!(!w.is_sleeping(0), "carving did not wake the ball");
    let h = 1.0 / (60.0 * SUBSTEPS as f64);
    let n = SUBSTEPS as f64;
    let expected = y0 + v0 / 60.0 + G * h * h * n * (n + 1.0) / 2.0;
    assert!((y(&w) - expected).abs() < 1e-9, "y {} vs {expected}", y(&w));

    // Not carved: bit for bit like the plain floor.
    let (mut d, _floor) = build();
    let mut plain = PhysicsWorld::new(PhysicsConfig {
        substeps: SUBSTEPS,
        gravity: Vec3Fix::from_int(0, G as i64, 0),
        ..PhysicsConfig::default()
    });
    plain.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
    plain.add_sdf_collider(SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    plain.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
        Fix128::ONE,
    ));
    let mut slept = 0;
    for _ in 0..200 {
        d.step(dt());
        plain.step(dt());
        slept += usize::from(d.is_sleeping(0));
    }
    assert!(
        slept > 0,
        "the ball never slept: the comparison does not cover the wake"
    );
    assert_eq!(bits(&d), bits(&plain));
}
