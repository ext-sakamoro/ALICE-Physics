//! Lib tests of `src/solver.rs` paths no other lib test observed: each one is
//! named after the behaviour it pins, and its doc says which closed form the
//! expected value comes from.
//!
//! Included from `src/solver.rs` as a `#[cfg(test)]` module, so they run with
//! `cargo test --lib`, the test set the mutation run uses.

use super::*;
use crate::joint::BallJoint;
use crate::sdf_collider::{ClosureSdf, SdfCollider};

fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn world_with(gravity: Vec3Fix, substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity,
        damping: Fix128::ONE,
        substeps,
        ..SolverConfig::default()
    })
}

fn ground_plane() -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

/// `step` solves joints through `solve_joints_dispatch`; with that call gone
/// a ball joint stops holding. A body hung from a static anchor by a ball
/// joint at distance 1 under gravity stays at distance 1 (the joint is
/// stiff, compliance 0) instead of falling `g t² / 2 = 1.25` in half a second.
#[test]
fn step_solves_joints() {
    let mut world = world_with(v3(0, -10, 0), 8);
    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let bob = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
    world.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        bob,
        Vec3Fix::ZERO,
        v3(-1, 0, 0),
    )));
    for _ in 0..30 {
        world.step(r(1, 60));
    }
    let d = world.bodies[bob].position.length();
    assert!((d - Fix128::ONE).abs() < r(1, 100), "distance {d:?}");
}

/// The SDF push-out leaves static bodies and sensors where they are and
/// logs no contact for them; a dynamic body is pushed out of the ground to
/// the surface (`y = radius`).
#[test]
fn sdf_push_out_skips_static_bodies_and_sensors() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    world.add_sdf_collider(ground_plane());
    world.set_sdf_collision_radius(r(1, 2));
    let at = Vec3Fix::new(Fix128::ZERO, r(1, 4), Fix128::ZERO);
    let fixed = world.add_body(RigidBody::new_static(at));
    let mut sensor = RigidBody::new_dynamic(
        Vec3Fix::new(v3(5, 0, 0).x, r(1, 4), Fix128::ZERO),
        Fix128::ONE,
    );
    sensor.is_sensor = true;
    let sensor = world.add_body(sensor);
    let free = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(v3(10, 0, 0).x, r(1, 4), Fix128::ZERO),
        Fix128::ONE,
    ));
    world.resolve_sdf_collisions();
    assert_eq!(world.bodies[fixed].position, at);
    assert_eq!(world.bodies[sensor].position.y, r(1, 4));
    assert!((world.bodies[free].position.y - r(1, 2)).abs() < r(1, 1000));
    let logged: Vec<usize> = world.sdf_contact_log.iter().map(|c| c.body_index).collect();
    assert_eq!(logged, vec![free]);
}

/// `approach_speed` of an SDF contact is the speed into the surface,
/// `-v · n`: a body moving down at 3 into a ground whose normal is `+y`
/// approaches at `+3`.
#[test]
fn sdf_contact_approach_speed_is_speed_into_the_surface() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    world.add_sdf_collider(ground_plane());
    world.set_sdf_collision_radius(r(1, 2));
    let mut body = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, r(1, 4), Fix128::ZERO),
        Fix128::ONE,
    );
    body.velocity = v3(0, -3, 0);
    world.add_body(body);
    world.resolve_sdf_collisions();
    assert_eq!(world.sdf_contact_log.len(), 1);
    let speed = world.sdf_contact_log[0].approach_speed;
    assert!(
        (speed - Fix128::from_int(3)).abs() < r(1, 1000),
        "approach {speed:?}"
    );
}

/// The per-stage work counters count, not multiply: one step of two
/// overlapping spheres with 2 substeps integrates, derives velocities for and
/// pushes out of the ground SDF 2 bodies per substep (4 each), builds 2
/// broad-phase primitives per substep (4) and finds their one candidate pair
/// per substep (2).
#[test]
fn stage_work_counts_bodies_primitives_and_pairs() {
    let mut world = world_with(Vec3Fix::ZERO, 2);
    world.add_body_with_radius(
        RigidBody::new_dynamic(v3(0, 5, 0), Fix128::ONE),
        Fix128::ONE,
    );
    world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(r(3, 2), v3(0, 5, 0).y, Fix128::ZERO),
            Fix128::ONE,
        ),
        Fix128::ONE,
    );
    world.add_sdf_collider(ground_plane());
    world.step(r(1, 60));
    let w = world.stage_work();
    assert_eq!(w.integrated, 4, "{w:?}");
    assert_eq!(w.velocity_bodies, 4, "{w:?}");
    assert_eq!(w.broadphase_primitives, 4, "{w:?}");
    assert_eq!(w.broadphase_pairs, 2, "{w:?}");
    assert_eq!(w.resolution_bodies, 4, "{w:?}");
}

/// Restitution applies only to an approach faster than `2 |g| h` (Müller et
/// al. 2020): with `|g| = 10` and `h = 1/4` the threshold is 5. A contact
/// with restitution 1 and pre-solve normal velocity `pre` (A moving into B
/// along `-x`, the contact normal is `+x`) ends at relative normal velocity
/// `-pre` above the threshold and at 0 at or below it.
#[test]
fn restitution_threshold_is_twice_gravity_times_substep() {
    let h = r(1, 4);
    for (pre, bounces) in [(-6, true), (-5, false), (-4, false), (-2, false)] {
        let mut world = world_with(v3(0, -10, 0), 1);
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let contact = Contact {
            depth: r(1, 100),
            normal: v3(1, 0, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        let mut c = ContactConstraint::new(a, b, contact);
        c.friction = Fix128::ZERO;
        c.restitution = Fix128::ONE;
        world.contact_constraints.push(c);
        // velocity (pre, 0, 0) for A, B at rest: the pre-solve normal velocity
        // is read from `velocity`, the derived one from `(position - prev) / h`
        let va = v3(pre, 0, 0);
        world.bodies[a].velocity = va;
        world.bodies[a].prev_position = world.bodies[a].position - va * h;
        world.bodies[b].prev_position = world.bodies[b].position;
        world.update_velocities(h);
        let vn = (world.bodies[a].velocity - world.bodies[b].velocity).x;
        let want = if bounces {
            Fix128::from_int(-pre)
        } else {
            Fix128::ZERO
        };
        assert_eq!(vn, want, "pre {pre}");
    }
}

/// Friction acts on the relative tangential velocity `v_a - v_b`: two
/// bodies sliding past each other along `y` at the same velocity have no
/// relative tangential velocity and keep it; moving apart at `±1` with an
/// unlimited friction budget, both stop sliding relative to each other
/// (equal masses: each ends at the mean, 0).
#[test]
fn friction_uses_the_relative_tangential_velocity() {
    let h = r(1, 4);
    for (vb, want_a) in [(1, 1), (-1, 0)] {
        let mut world = world_with(Vec3Fix::ZERO, 1);
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let contact = Contact {
            depth: r(1, 100),
            normal: v3(1, 0, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        let mut c = ContactConstraint::new(a, b, contact);
        c.friction = Fix128::ONE;
        c.restitution = Fix128::ZERO;
        c.cached_lambda = Fix128::from_int(100);
        world.contact_constraints.push(c);
        let (va, vbv) = (v3(0, 1, 0), v3(0, vb, 0));
        world.bodies[a].velocity = va;
        world.bodies[b].velocity = vbv;
        world.bodies[a].prev_position = world.bodies[a].position - va * h;
        world.bodies[b].prev_position = world.bodies[b].position - vbv * h;
        world.update_velocities(h);
        assert_eq!(
            world.bodies[a].velocity.y,
            Fix128::from_int(want_a),
            "vb {vb}"
        );
    }
}

fn tgs_world(gravity: Vec3Fix, substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity,
        damping: Fix128::ONE,
        substeps,
        solver_backend: SolverBackend::Tgs,
        ..SolverConfig::default()
    })
}

/// Free fall from rest over `N` semi-implicit Euler substeps of `h = dt/N`
/// (`v += g h`, then `x += v h`) drops `g h² N(N+1)/2 = g dt² (N+1)/(2N)`.
fn free_fall_drop(g: i64, dt: Fix128, n: i64) -> Fix128 {
    Fix128::from_int(g) * dt * dt * Fix128::from_ratio(n + 1, 2 * n)
}

/// `step_tgs` sub-steps by `config.substeps` (not the TGS default of 4):
/// a free body falls the closed-form semi-implicit drop for `N` substeps,
/// with no joint (one solver call over `N` substeps) and with an unrelated
/// joint in the world (the per-substep loop, one solver substep each).
#[test]
fn tgs_substeps_follow_the_config() {
    for with_joint in [false, true] {
        for n in [1i64, 3] {
            let mut world = tgs_world(v3(0, -10, 0), n as usize);
            let free = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
            if with_joint {
                let anchor = world.add_body(RigidBody::new_static(v3(100, 0, 0)));
                let bob = world.add_body(RigidBody::new_dynamic(v3(101, 0, 0), Fix128::ONE));
                world.add_joint(Joint::Ball(BallJoint::new(
                    anchor,
                    bob,
                    Vec3Fix::ZERO,
                    v3(-1, 0, 0),
                )));
            }
            world.step(Fix128::ONE);
            let drop = -world.bodies[free].position.y;
            let want = free_fall_drop(10, Fix128::ONE, n);
            assert!(
                (drop - want).abs() < r(1, 1_000_000),
                "joint {with_joint} N {n}: {drop:?} vs {want:?}"
            );
        }
    }
}

/// `step_tgs` passes `config.iterations` as the velocity iterations: a
/// three-sphere stack on a static floor ends in a different state after one
/// step with 1 and with 8 iterations (the stack is not solved exactly by one
/// sweep), so the count reaches the solver.
#[test]
fn tgs_velocity_iterations_follow_the_config() {
    let run = |iterations: usize| {
        let mut world = PhysicsWorld::new(SolverConfig {
            gravity: v3(0, -10, 0),
            damping: Fix128::ONE,
            iterations,
            solver_backend: SolverBackend::Tgs,
            ..SolverConfig::default()
        });
        world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
        for k in 1..=3 {
            let y = Fix128::from_ratio(19 * k, 10);
            world.add_body_with_radius(
                RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO), Fix128::ONE),
                Fix128::ONE,
            );
        }
        world.step(r(1, 60));
        world
            .bodies
            .iter()
            .map(|b| b.position.y)
            .collect::<Vec<_>>()
    };
    assert_ne!(run(1), run(8));
}

/// `step_tgs` applies force fields: a body of mass 1 in a directional field
/// of strength 6 along `+x` (no gravity) gains velocity `6 dt` in one step.
#[test]
fn tgs_applies_force_fields() {
    let mut world = tgs_world(Vec3Fix::ZERO, 4);
    let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world.add_force_field(crate::force::ForceFieldInstance::new(
        crate::force::ForceField::Directional {
            direction: v3(1, 0, 0),
            strength: Fix128::from_int(6),
        },
    ));
    world.step(r(1, 2));
    assert!(
        (world.bodies[b].velocity.x - Fix128::from_int(3)).abs() < r(1, 1_000_000),
        "{:?}",
        world.bodies[b].velocity
    );
}

/// `step_tgs` resolves SDF overlap: a sphere of radius 1/2 sunk to
/// `y = 1/4` in the ground plane is pushed up to the surface (`y ≥ 1/2`
/// less the one step of gravity it then falls).
#[test]
fn tgs_resolves_sdf_overlap() {
    let mut world = tgs_world(Vec3Fix::ZERO, 4);
    world.add_sdf_collider(ground_plane());
    world.set_sdf_collision_radius(r(1, 2));
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, r(1, 4), Fix128::ZERO),
        Fix128::ONE,
    ));
    world.step(r(1, 60));
    assert!(
        (world.bodies[b].position.y - r(1, 2)).abs() < r(1, 1000),
        "{:?}",
        world.bodies[b].position
    );
}

/// `step_tgs` keeps a ball joint: a bob hung at distance 1 from a static
/// anchor stays at distance 1 under gravity over half a second.
#[test]
fn tgs_keeps_joints() {
    let mut world = tgs_world(v3(0, -10, 0), 8);
    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let bob = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
    world.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        bob,
        Vec3Fix::ZERO,
        v3(-1, 0, 0),
    )));
    for _ in 0..30 {
        world.step(r(1, 60));
    }
    let d = world.bodies[bob].position.length();
    assert!((d - Fix128::ONE).abs() < r(1, 100), "distance {d:?}");
}

/// Two spheres of radius `2³²` whose centres are `3·2³¹` apart overlap by
/// `2³¹`; their squared distance (`≈ 4.1·10¹⁹`) does not fit `Fix128`, so the
/// scaled-distance path decides. It reports the contact with that depth.
#[test]
fn far_apart_huge_spheres_still_collide() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    let big = Fix128::from_int(1 << 32);
    world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), big);
    world.add_body_with_radius(RigidBody::new_dynamic(v3(3 << 31, 0, 0), Fix128::ONE), big);
    world.detect_collisions();
    let ev = world.contact_events();
    assert_eq!(ev.len(), 1);
    assert!(
        (ev[0].depth - Fix128::from_int(1 << 31)).abs() < Fix128::ONE,
        "{:?}",
        ev[0].depth
    );
}

/// Spheres of radius 1 and `2⁻⁶⁴` (one ulp) whose centres are exactly the
/// radius sum `1 + 2⁻⁶⁴` apart touch without overlapping: no contact. The
/// squared radius sum `1 + 2⁻⁶³ + 2⁻¹²⁸` is not representable, so it and the
/// squared distance both truncate to `1 + 2⁻⁶³`; the equality of the
/// squares is the only thing that keeps the pair out (`√(1 + 2⁻⁶³)`
/// truncates to 1, below the radius sum).
#[test]
fn touching_spheres_with_an_unrepresentable_squared_sum_do_not_collide() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    let ulp = Fix128::from_raw(0, 1);
    world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ONE + ulp, Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        ulp,
    );
    world.detect_collisions();
    assert!(
        world.contact_events().is_empty(),
        "{:?}",
        world.contact_events()
    );
    assert!(world.contact_constraints.is_empty());
}

fn unit_box() -> crate::shape::Shape {
    crate::shape::Shape::Box {
        half_extents: Vec3Fix::from_int(1, 1, 1),
    }
}

/// A box body and a sphere body with coincident centres overlap: the pair
/// has a collider, so the shapes decide (the sphere-sphere path cannot give
/// a normal there and would drop it).
#[test]
fn a_collider_pair_with_coincident_centres_collides() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    world
        .add_shaped_body(&unit_box(), Fix128::ONE, Vec3Fix::ZERO)
        .unwrap();
    world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r(1, 2));
    world.detect_collisions();
    assert_eq!(
        world.contact_events().len(),
        1,
        "{:?}",
        world.contact_events()
    );
}

/// A collider contact reports the relative velocity `(v_a - v_b) · n`: a box
/// and a sphere beside it moving together at `(2, 0, 0)` approach at 0.
#[test]
fn a_collider_contact_reports_relative_velocity() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    let a = world
        .add_shaped_body(&unit_box(), Fix128::ONE, Vec3Fix::ZERO)
        .unwrap();
    let b = world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        Fix128::ONE,
    );
    world.bodies[a].velocity = v3(2, 0, 0);
    world.bodies[b].velocity = v3(2, 0, 0);
    world.detect_collisions();
    let ev = world.contact_events();
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].relative_velocity, Fix128::ZERO);
}

/// A collider pair where either body is a sensor reports a trigger and the
/// contact event, but adds no contact constraint (as the sphere path does).
#[test]
fn a_collider_pair_with_a_sensor_reports_a_trigger() {
    for sensor_on_box in [false, true] {
        let mut world = world_with(Vec3Fix::ZERO, 1);
        let a = world
            .add_shaped_body(&unit_box(), Fix128::ONE, Vec3Fix::ZERO)
            .unwrap();
        let b = world.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            Fix128::ONE,
        );
        world.bodies[if sensor_on_box { a } else { b }].is_sensor = true;
        world.detect_collisions();
        assert_eq!(
            world.contact_events().len(),
            1,
            "sensor on box {sensor_on_box}"
        );
        assert_eq!(
            world.trigger_events().len(),
            1,
            "sensor on box {sensor_on_box}"
        );
        assert!(world.contact_constraints.is_empty());
    }
}

/// A box of half extent 1 and a sphere of radius 1 whose centre is 2 from
/// the box centre touch at one point without overlapping: no contact.
#[test]
fn a_collider_pair_that_only_touches_does_not_collide() {
    let mut world = world_with(Vec3Fix::ZERO, 1);
    world
        .add_shaped_body(&unit_box(), Fix128::ONE, Vec3Fix::ZERO)
        .unwrap();
    world.add_body_with_radius(
        RigidBody::new_dynamic(v3(2, 0, 0), Fix128::ONE),
        Fix128::ONE,
    );
    world.detect_collisions();
    assert!(
        world.contact_events().is_empty(),
        "{:?}",
        world.contact_events()
    );
}

/// In the first substep of a step that parked a body, detection reads the
/// parked body's velocity as the force-field kick it would have had
/// (`v + F/m · frame_dt`); an awake body's and every later substep's
/// velocity is `velocity` itself (the kick is already in it, or zeroed).
/// Body 0 (mass 1, parked) and body 1 (mass 2, awake) overlap along `x`, a
/// field of strength 6 along `+x` acts on both, `frame_dt = 1/2`: the kick
/// is 3 for body 0 and would be 3/2 for body 1. The contact normal points
/// from body 1 to body 0 (`-x`).
#[test]
fn parked_bodies_see_their_force_field_kick_in_the_first_substep() {
    let run = |parked0: bool, first: bool| {
        let mut world = world_with(Vec3Fix::ZERO, 1);
        world.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            Fix128::ONE,
        );
        world.add_body_with_radius(
            RigidBody::new_dynamic(v3(1, 0, 0), Fix128::from_int(2)),
            Fix128::ONE,
        );
        world.add_force_field(crate::force::ForceFieldInstance::new(
            crate::force::ForceField::Directional {
                direction: v3(1, 0, 0),
                strength: Fix128::from_int(6),
            },
        ));
        world.park.active = true;
        world.park.first_substep = first;
        world.park.frame_dt = r(1, 2);
        world.park.parked = vec![parked0, false];
        world.park.awake = if parked0 { vec![1] } else { vec![0, 1] };
        if parked0 {
            let aabb = world.broadphase_box(0, Fix128::ONE);
            world.park.proxies = vec![Some(world.park.tree.insert(aabb, 0)), None];
            world.park.proxy_live = 1;
        }
        world.detect_collisions();
        let ev = world.contact_events();
        assert_eq!(ev.len(), 1, "parked {parked0} first {first}");
        ev[0].relative_velocity
    };
    // parked, first substep: (3 - 0) · (-1)
    assert_eq!(run(true, true), Fix128::from_int(-3));
    // parked, later substep: velocities as they are
    assert_eq!(run(true, false), Fix128::ZERO);
    // nothing parked, first substep: velocities as they are
    assert_eq!(run(false, true), Fix128::ZERO);
}

/// A ball resting on a static sphere under TGS takes the same next step
/// whether or not a `remove_body` has just swapped it to a lower index than
/// the sphere: the contact is re-oriented from the lower stable id to the
/// higher one, so last frame's warm-start impulse is applied in the frame it
/// was stored in. The removed body is far away in its own island, so the twin
/// world that keeps it solves the same contact with the same numbers.
#[test]
fn tgs_warm_start_survives_an_index_swap() {
    let build = || {
        let mut world = tgs_world(v3(0, -10, 0), 4);
        world.add_body_with_radius(
            RigidBody::new_dynamic(v3(100, 0, 0), Fix128::ONE),
            Fix128::ONE,
        );
        world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
        world.add_body_with_radius(
            RigidBody::new_dynamic(v3(0, 2, 0), Fix128::ONE),
            Fix128::ONE,
        );
        for _ in 0..60 {
            world.step(r(1, 60));
        }
        world
    };
    let mut kept = build();
    let mut swapped = build();
    swapped.remove_body(0);
    // swap_remove: the ball is now index 0, the ground index 1
    assert_eq!(swapped.bodies[0].position, kept.bodies[2].position);
    assert_eq!(swapped.bodies[1].position, Vec3Fix::ZERO);
    for _ in 0..3 {
        kept.step(r(1, 60));
        swapped.step(r(1, 60));
        assert_eq!(swapped.bodies[0].position, kept.bodies[2].position);
        assert_eq!(swapped.bodies[0].velocity, kept.bodies[2].velocity);
    }
}
