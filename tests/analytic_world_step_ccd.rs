//! Continuous collision in the world step
//! ([`PhysicsWorld::set_continuous_collision`]), through `PhysicsWorld::step`.
//!
//! oracle: closed-form kinematics of translating spheres. Unless a test says
//! otherwise the world has one substep of `h = 1/64` s, no gravity, no
//! damping, and every body the material `μ = 0`, `e = 1/2`, so one step is one
//! straight sweep and the outcome of an impact is the textbook one:
//!
//! - a sphere of radius `r` reaching the face `x = x_f` of a still wall
//!   stops at `x_f − r` and leaves with `−e` times its approach speed;
//! - two spheres closing head on touch at `t* = (gap − r_a − r_b) / |u_a − u_b|`
//!   of the step, keep their momentum and separate at `e |u_a − u_b|`:
//!   `v_a = (m_a u_a + m_b u_b + m_b e (u_b − u_a)) / (m_a + m_b)`;
//! - a sphere hitting a still sphere off centre with impact parameter `b`
//!   touches where `x² + b² = (R + r)²`, the normal is the line of centres
//!   there, and the velocity becomes `v − (1 + e)(v·n) n`.
//!
//! The obstacle-to-body normal convention is the one of
//! `PhysicsWorld::time_of_impact`. The cast conventions assumed for the
//! boundary cases (a cast that starts touching hits only when it moves into
//! the surface, a grazing path must enter it to hit) are the ones of
//! [`alice_physics::world_shape_query`].

#![cfg(feature = "std")]

use alice_physics::event::ContactEvent;
use alice_physics::material::PhysicsMaterial;
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldCcdConfig};
use sha2::{Digest, Sha256};

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}

fn h() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn close(got: Vec3Fix, want: Vec3Fix, tol: f64, what: &str) {
    let d = [
        (got.x - want.x).to_f64(),
        (got.y - want.y).to_f64(),
        (got.z - want.z).to_f64(),
    ];
    assert!(
        d.iter().all(|e| e.abs() <= tol),
        "{what}: got ({}, {}, {}), want ({}, {}, {})",
        got.x.to_f64(),
        got.y.to_f64(),
        got.z.to_f64(),
        want.x.to_f64(),
        want.y.to_f64(),
        want.z.to_f64()
    );
}

/// Position tolerance: the casts and the sphere TOI are exact to far below
/// this; the velocities divide by `h`, so their tolerance is 64 times larger.
const POS_TOL: f64 = 1e-9;
const VEL_TOL: f64 = 64e-9;

/// One substep of `1/64` s, no gravity, no damping, material `μ = 0, e = 1/2`.
fn bare(ccd: bool) -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    if ccd {
        w.set_continuous_collision(WorldCcdConfig::on());
    }
    w
}

fn half_bouncy(w: &mut PhysicsWorld) {
    let id = w.material_table.register(PhysicsMaterial::new(
        1,
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    ));
    for i in 0..w.bodies.len() {
        w.set_body_material(i, id);
    }
}

fn sphere(w: &mut PhysicsWorld, at: Vec3Fix, mass: i64, radius: f64, velocity: Vec3Fix) -> usize {
    let mut b = RigidBody::new_dynamic(at, Fix128::from_int(mass));
    b.velocity = velocity;
    w.add_body_with_radius(b, f(radius))
}

/// A still plate 1/32 thick (faces at `x = 5 ∓ 1/64`), 4 × 4 wide.
fn plate(w: &mut PhysicsWorld) -> usize {
    let p = w.add_body(RigidBody::new_static(v3(5.0, 0.0, 0.0)));
    w.set_body_shape(
        p,
        &Shape::Box {
            half_extents: v3(1.0 / 64.0, 2.0, 2.0),
        },
    );
    p
}

/// A sphere of radius 1/4 at the origin moving at `speed` along `+x` toward
/// [`plate`].
fn plate_scene(ccd: bool, speed: Fix128) -> (PhysicsWorld, usize) {
    let mut w = bare(ccd);
    plate(&mut w);
    let s = sphere(
        &mut w,
        Vec3Fix::ZERO,
        1,
        0.25,
        Vec3Fix::new(speed, Fix128::ZERO, Fix128::ZERO),
    );
    half_bouncy(&mut w);
    (w, s)
}

// ── Thin plate ──────────────────────────────────────────────────────────────

/// Off: the sphere moves 10 m in the step, through the plate (characterization).
#[test]
fn off_a_fast_sphere_passes_through_a_thin_plate() {
    let (mut w, s) = plate_scene(false, Fix128::from_int(640));
    w.step(h());
    close(
        w.bodies[s].position,
        v3(10.0, 0.0, 0.0),
        POS_TOL,
        "position",
    );
    close(
        w.bodies[s].velocity,
        v3(640.0, 0.0, 0.0),
        VEL_TOL,
        "velocity",
    );
}

/// On: it stops on the near face, `x = 5 − 1/64 − 1/4`, and bounces with `e`.
// covers: COV-RIGID-074
#[test]
fn on_a_fast_sphere_stops_on_a_thin_plate_and_bounces() {
    let (mut w, s) = plate_scene(true, Fix128::from_int(640));
    w.step(h());
    close(
        w.bodies[s].position,
        v3(5.0 - 1.0 / 64.0 - 0.25, 0.0, 0.0),
        POS_TOL,
        "position",
    );
    close(
        w.bodies[s].velocity,
        v3(-320.0, 0.0, 0.0),
        VEL_TOL,
        "velocity",
    );
}

/// The default configuration (8 substeps, gravity, damping, default
/// materials) over a second at 552 m/s (1.15 m per substep, so the substep
/// poses 4.6 and 5.75 straddle the plate without touching it): off, the
/// sphere ends past the plate; on, it never crosses the plate's near face.
// covers: COV-RIGID-074
#[test]
fn default_config_never_crosses_the_plate_when_on() {
    for ccd in [false, true] {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        if ccd {
            w.set_continuous_collision(WorldCcdConfig::on());
        }
        plate(&mut w);
        let s = sphere(&mut w, v3(0.0, 0.0, 0.0), 1, 0.25, v3(552.0, 0.0, 0.0));
        let mut max_x = f64::MIN;
        for _ in 0..60 {
            w.step(Fix128::from_ratio(1, 60));
            max_x = max_x.max(w.bodies[s].position.x.to_f64());
        }
        if ccd {
            assert!(
                max_x <= 5.0 - 1.0 / 64.0 - 0.25 + 1e-9,
                "on: reached x = {max_x}"
            );
            assert!(
                w.bodies[s].velocity.x.to_f64() < 0.0,
                "on: did not bounce back"
            );
        } else {
            assert!(
                w.bodies[s].position.x.to_f64() > 5.0 + 1.0 / 64.0 + 0.25,
                "off: did not pass"
            );
        }
    }
}

/// A static triangle mesh (zero thickness) has no body to carry a contact:
/// the sphere stops on its surface and does not cross it.
#[test]
fn on_a_fast_sphere_stops_on_a_static_triangle() {
    for ccd in [false, true] {
        let mut w = bare(ccd);
        let tri = TriMesh::from_indexed(
            &[v3(5.0, -4.0, -4.0), v3(5.0, 4.0, -4.0), v3(5.0, 0.0, 4.0)],
            &[0, 1, 2],
        );
        w.add_static_collider(StaticCollider::TriMesh(tri));
        let s = sphere(&mut w, Vec3Fix::ZERO, 1, 0.25, v3(640.0, 0.0, 0.0));
        w.step(h());
        let want = if ccd { 4.75 } else { 10.0 };
        close(
            w.bodies[s].position,
            v3(want, 0.0, 0.0),
            POS_TOL,
            "position",
        );
    }
}

// ── Two spheres head on ─────────────────────────────────────────────────────

fn head_on(ccd: bool, mass_b: i64) -> (PhysicsWorld, usize, usize) {
    let mut w = bare(ccd);
    let a = sphere(&mut w, v3(-6.0, 0.0, 0.0), 1, 0.5, v3(640.0, 0.0, 0.0));
    let b = sphere(&mut w, v3(6.0, 0.0, 0.0), mass_b, 0.5, v3(-640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    (w, a, b)
}

#[test]
fn off_two_fast_spheres_pass_through_each_other() {
    let (mut w, a, b) = head_on(false, 1);
    w.step(h());
    close(w.bodies[a].position, v3(4.0, 0.0, 0.0), POS_TOL, "a");
    close(w.bodies[b].position, v3(-4.0, 0.0, 0.0), POS_TOL, "b");
}

/// Equal masses: `t* = (12 − 1) / 20 = 0.55`, both stop at the touching
/// poses `∓1/2` and leave at `∓e · 640`.
// covers: COV-RIGID-074
#[test]
fn on_two_equal_spheres_stop_at_the_time_of_impact() {
    let (mut w, a, b) = head_on(true, 1);
    w.step(h());
    close(
        w.bodies[a].position,
        v3(-0.5, 0.0, 0.0),
        POS_TOL,
        "a position",
    );
    close(
        w.bodies[b].position,
        v3(0.5, 0.0, 0.0),
        POS_TOL,
        "b position",
    );
    close(
        w.bodies[a].velocity,
        v3(-320.0, 0.0, 0.0),
        VEL_TOL,
        "a velocity",
    );
    close(
        w.bodies[b].velocity,
        v3(320.0, 0.0, 0.0),
        VEL_TOL,
        "b velocity",
    );
}

/// Masses 1 and 3: the pair touches at the same `t* = 0.55`, then moves as
/// one with the centre-of-mass velocity `−320` for the rest of the step
/// (`−5 · 0.45` m), and leaves with the 1-D restitution result `−800 / −160`.
// covers: COV-RIGID-074
#[test]
fn on_unequal_spheres_keep_momentum_and_separate_at_e_times_approach() {
    let (mut w, a, b) = head_on(true, 3);
    w.step(h());
    close(
        w.bodies[a].position,
        v3(-0.5 - 2.25, 0.0, 0.0),
        POS_TOL,
        "a position",
    );
    close(
        w.bodies[b].position,
        v3(0.5 - 2.25, 0.0, 0.0),
        POS_TOL,
        "b position",
    );
    close(
        w.bodies[a].velocity,
        v3(-800.0, 0.0, 0.0),
        VEL_TOL,
        "a velocity",
    );
    close(
        w.bodies[b].velocity,
        v3(-160.0, 0.0, 0.0),
        VEL_TOL,
        "b velocity",
    );
    let p = w.bodies[a].velocity.x.to_f64() + 3.0 * w.bodies[b].velocity.x.to_f64();
    assert!(
        (p - (640.0 - 3.0 * 640.0)).abs() <= 4.0 * VEL_TOL,
        "momentum {p}"
    );
}

// ── Oblique impact on a still sphere ────────────────────────────────────────

/// A still sphere of radius 1 at the origin and a sphere of radius 1/2 at
/// `(−10, b, 0)` moving at 640 m/s along `+x`.
fn oblique(ccd: bool, b: f64) -> (PhysicsWorld, usize, usize) {
    let mut w = bare(ccd);
    let o = w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    let s = sphere(&mut w, v3(-10.0, b, 0.0), 1, 0.5, v3(640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    (w, o, s)
}

fn first_contact(w: &PhysicsWorld) -> ContactEvent {
    *w.events.contact_events().first().expect("a contact event")
}

/// `b = 0.9`: contact at `x = −1.2` (`t = 0.88`), normal and contact point
/// `(−0.8, 0.6, 0)`; the sphere then slides along the tangent plane for the
/// rest of the step and leaves with `v − (1 + e)(v·n) n = (25.6, 460.8, 0)`.
// covers: COV-RIGID-074
#[test]
fn on_an_oblique_impact_has_the_closed_form_point_normal_and_velocity() {
    let (mut w, o, s) = oblique(true, 0.9);
    w.step(h());
    let ev = first_contact(&w);
    let n = v3(-0.8, 0.6, 0.0);
    let n_event = if ev.body_a == s { n } else { -n };
    assert_eq!(
        (ev.body_a.min(ev.body_b), ev.body_a.max(ev.body_b)),
        (o.min(s), o.max(s))
    );
    close(ev.normal, n_event, 1e-12, "normal");
    close(ev.point, n, 1e-9, "contact point");
    // (−10, 0.9) + 10·0.88 x̂ + 0.12 · tangent((10, 0, 0)) = (−0.768, 1.476)
    close(
        w.bodies[s].position,
        v3(-0.768, 1.476, 0.0),
        POS_TOL,
        "position",
    );
    close(
        w.bodies[s].velocity,
        v3(25.6, 460.8, 0.0),
        VEL_TOL,
        "velocity",
    );
}

/// A path that misses the obstacle by `2⁻²⁰` is free flight; one that enters
/// it by `2⁻⁸` is caught (without the setting it passes through).
#[test]
fn grazing_paths_hit_only_when_they_enter_the_obstacle() {
    let miss = 1.5 + 1.0 / 1_048_576.0; // 2⁻²⁰
    let (mut w, _, s) = oblique(true, miss);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(0.0, miss, 0.0),
        POS_TOL,
        "missing path",
    );
    close(
        w.bodies[s].velocity,
        v3(640.0, 0.0, 0.0),
        VEL_TOL,
        "missing path velocity",
    );

    let enter = 1.5 - 1.0 / 256.0; // 2⁻⁸
    let x_c = -(1.5f64 * 1.5 - enter * enter).sqrt();
    let (mut w, _, s) = oblique(true, enter);
    w.step(h());
    let ev = first_contact(&w);
    let n = v3(x_c / 1.5, enter / 1.5, 0.0);
    close(ev.point, n, 1e-9, "grazing contact point");
    assert!(
        w.bodies[s].velocity.y.to_f64() > 1.0,
        "grazing hit did not deflect"
    );

    let (mut w, _, s) = oblique(false, enter);
    w.step(h());
    assert!(w.events.contact_events().is_empty() || w.bodies[s].position.x.to_f64() > -1.0);
}

/// Starting exactly in contact: moving into the obstacle is a hit at `t = 0`
/// (the sphere stays and bounces), moving away is free flight.
#[test]
fn starting_in_contact_hits_only_when_moving_into_the_obstacle() {
    let mut w = bare(true);
    w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    let s = sphere(&mut w, v3(-1.5, 0.0, 0.0), 1, 0.5, v3(640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(-1.5, 0.0, 0.0),
        POS_TOL,
        "into: position",
    );
    close(
        w.bodies[s].velocity,
        v3(-320.0, 0.0, 0.0),
        VEL_TOL,
        "into: velocity",
    );

    let mut w = bare(true);
    w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    let s = sphere(&mut w, v3(-1.5, 0.0, 0.0), 1, 0.5, v3(-640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(-11.5, 0.0, 0.0),
        POS_TOL,
        "away: position",
    );
    close(
        w.bodies[s].velocity,
        v3(-640.0, 0.0, 0.0),
        VEL_TOL,
        "away: velocity",
    );
}

// ── Threshold ───────────────────────────────────────────────────────────────

/// With a threshold of 40 radii (10 m for `r = 1/4`), a displacement of
/// exactly 10 m is not swept (it passes through) and one `2⁻²⁰` m/s faster is.
#[test]
fn a_displacement_equal_to_the_threshold_is_not_swept() {
    let config = WorldCcdConfig::on().with_motion_threshold(Fix128::from_int(40));
    let (mut w, s) = plate_scene(false, Fix128::from_int(640));
    w.set_continuous_collision(config);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(10.0, 0.0, 0.0),
        POS_TOL,
        "at the threshold",
    );

    let faster = Fix128::from_int(640) + Fix128::from_f64(1.0 / 1_048_576.0);
    let (mut w, s) = plate_scene(false, faster);
    w.set_continuous_collision(config);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(5.0 - 1.0 / 64.0 - 0.25, 0.0, 0.0),
        POS_TOL,
        "past the threshold",
    );
}

/// A threshold `≤ 0` sweeps every moving body: a negative one gives the same
/// world as `0`, and both sweep a body that moves `1/8` m per step, under the
/// default threshold of one radius. A swept impact is reported at depth `0`
/// (the body is stopped at the surface); the discrete detection of the same
/// impact reports the overlap it found, `1/8 − 0.034375 = 0.090625`.
#[test]
fn a_non_positive_threshold_sweeps_every_moving_body() {
    let run = |config: WorldCcdConfig| {
        let (mut w, s) = plate_scene(false, Fix128::from_int(8));
        w.set_continuous_collision(config);
        w.bodies[s].position = v3(4.7, 0.0, 0.0);
        w.step(h());
        let depth = first_contact(&w).depth.to_f64();
        (w.serialize_state(), depth)
    };
    let (zero, zero_depth) = run(WorldCcdConfig::on().with_motion_threshold(Fix128::ZERO));
    let (negative, _) = run(WorldCcdConfig::on().with_motion_threshold(Fix128::from_int(-1)));
    let (_, default_depth) = run(WorldCcdConfig::on());
    assert_eq!(zero, negative);
    assert_eq!(zero_depth, 0.0, "threshold 0 did not sweep");
    assert!(
        (default_depth - 0.090625).abs() < 1e-7,
        "default threshold: depth {default_depth}"
    );
}

// ── Degenerate inputs ───────────────────────────────────────────────────────

/// Bodies the sweep must leave alone, each with its expected outcome: a body
/// without a collision radius and a sensor fly through; an empty world and a
/// world of resting bodies step as with the setting off.
#[test]
fn bodies_without_radius_sensors_and_resting_worlds_are_left_alone() {
    // no radius: free flight
    let mut w = bare(true);
    plate(&mut w);
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.velocity = v3(640.0, 0.0, 0.0);
    let s = w.add_body(b);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(10.0, 0.0, 0.0),
        POS_TOL,
        "no radius",
    );

    // sensor: free flight
    let mut w = bare(true);
    plate(&mut w);
    let mut b = RigidBody::new_sensor(Vec3Fix::ZERO);
    b.velocity = v3(640.0, 0.0, 0.0);
    let s = w.add_body_with_radius(b, f(0.25));
    w.step(h());
    let x = w.bodies[s].position.x.to_f64();
    assert!(x > 5.0 || x == 0.0, "sensor swept: x = {x}");

    // empty world
    let mut w = bare(true);
    w.step(h());
    assert!(w.bodies.is_empty());

    // resting: on and off identical
    let state = |ccd: bool| {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.set_continuous_collision(WorldCcdConfig::new().with_enabled(ccd));
        w.add_body_with_radius(
            RigidBody::new_static(v3(0.0, -10.0, 0.0)),
            Fix128::from_int(10),
        );
        w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
            f(0.5),
        );
        for _ in 0..30 {
            w.step(Fix128::from_ratio(1, 60));
        }
        w.serialize_state()
    };
    assert_eq!(state(true), state(false));
}

/// A speed of `2³⁰` m/s (`2²⁴` m per step) is still swept and stopped on the
/// plate.
#[test]
fn a_huge_speed_is_still_stopped() {
    let (mut w, s) = plate_scene(true, Fix128::from_int(1 << 30));
    w.step(h());
    close(
        w.bodies[s].position,
        v3(5.0 - 1.0 / 64.0 - 0.25, 0.0, 0.0),
        1e-6,
        "position",
    );
    assert!(w.bodies[s].velocity.x.to_f64() < 0.0);
}

// ── Starting in contact with something else ─────────────────────────────────

/// A static box plate `1/32 × 6 × 6` at `x = 5` (near face `5 − 1/64`).
fn tall_plate(w: &mut PhysicsWorld) -> usize {
    let p = w.add_body(RigidBody::new_static(v3(5.0, 0.0, 0.0)));
    w.set_body_shape(
        p,
        &Shape::Box {
            half_extents: v3(1.0 / 64.0, 3.0, 3.0),
        },
    );
    p
}

/// A sphere rolling on a floor (overlapping the plane `y = −3` by `2⁻¹²`)
/// thrown at the plate: the overlap with the floor must not hide the plate.
/// It stops on the plate's near face and bounces with `e`.
// covers: COV-RIGID-074
#[test]
fn an_overlap_with_the_floor_does_not_hide_the_wall() {
    let mut w = bare(true);
    tall_plate(&mut w);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        f(-3.0),
    )));
    let y = -3.0 + 0.25 - 1.0 / 4096.0;
    let s = sphere(&mut w, v3(0.0, y, 0.0), 1, 0.25, v3(640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    w.step(h());
    let p = w.bodies[s].position;
    assert!(
        (p.x.to_f64() - (5.0 - 1.0 / 64.0 - 0.25)).abs() <= POS_TOL,
        "x = {}",
        p.x.to_f64()
    );
    assert!((w.bodies[s].velocity.x.to_f64() + 320.0).abs() <= VEL_TOL);
}

/// The same with a still sphere the moving sphere starts inside of.
#[test]
fn an_overlap_with_a_still_body_does_not_hide_the_wall() {
    let mut w = bare(true);
    tall_plate(&mut w);
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.2, 0.0)), Fix128::ONE);
    let s = sphere(&mut w, Vec3Fix::ZERO, 1, 0.25, v3(640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    w.step(h());
    let x = w.bodies[s].position.x.to_f64();
    assert!((x - (5.0 - 1.0 / 64.0 - 0.25)).abs() <= POS_TOL, "x = {x}");
    assert!((w.bodies[s].velocity.x.to_f64() + 320.0).abs() <= VEL_TOL);
}

/// A sphere `2⁻¹⁰` into a thin floor (half thickness `1/64`) thrown down
/// through it at 640 m/s: on, it does not pass the floor (it is held at its
/// start and leaves upward with `e` times its approach speed); off, it passes.
#[test]
fn a_sphere_already_in_a_thin_floor_and_moving_in_does_not_pass() {
    for ccd in [false, true] {
        let mut w = bare(ccd);
        let floor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        w.set_body_shape(
            floor,
            &Shape::Box {
                half_extents: v3(3.0, 1.0 / 64.0, 3.0),
            },
        );
        let y0 = 1.0 / 64.0 + 0.25 - 1.0 / 1024.0;
        let s = sphere(&mut w, v3(0.0, y0, 0.0), 1, 0.25, v3(0.0, -640.0, 0.0));
        half_bouncy(&mut w);
        w.step(h());
        let y = w.bodies[s].position.y.to_f64();
        if ccd {
            assert!(y >= y0 - POS_TOL, "on: y = {y}");
            assert!(w.bodies[s].velocity.y.to_f64() > 0.0, "on: did not bounce");
        } else {
            assert!(y < -1.0 / 64.0 - 0.25, "off: did not pass, y = {y}");
        }
    }
}

/// A sphere touching a floor exactly and sliding along it is free flight
/// (a cast that starts touching hits only when it moves in).
#[test]
fn a_sphere_sliding_on_a_floor_is_not_stopped() {
    let mut w = bare(true);
    let floor = w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
    w.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(50.0, 1.0, 50.0),
        },
    );
    let s = sphere(&mut w, v3(0.0, 0.25, 0.0), 1, 0.25, v3(640.0, 0.0, 0.0));
    half_bouncy(&mut w);
    w.step(h());
    close(
        w.bodies[s].position,
        v3(10.0, 0.25, 0.0),
        POS_TOL,
        "position",
    );
    close(
        w.bodies[s].velocity,
        v3(640.0, 0.0, 0.0),
        VEL_TOL,
        "velocity",
    );
}

// ── Kinematic bodies ────────────────────────────────────────────────────────

/// A kinematic sphere (radius 1/2) driven from `x = −6` to `x = 4` in one
/// step hits a still dynamic sphere (radius 1/2, at the origin): the
/// kinematic body reaches its target, the dynamic one is carried to touch it
/// (`x = 5`) and leaves at `(1 + e) · 640 = 960` m/s (an immovable body
/// hitting a free one). Off, the kinematic body passes through it.
// covers: COV-RIGID-074
#[test]
fn a_fast_kinematic_body_pushes_a_still_dynamic_body() {
    for ccd in [false, true] {
        let mut w = bare(ccd);
        let mut k = RigidBody::new_kinematic(v3(-6.0, 0.0, 0.0));
        k.kinematic_target = Some((v3(4.0, 0.0, 0.0), k.rotation));
        let k = w.add_body_with_radius(k, f(0.5));
        let d = sphere(&mut w, Vec3Fix::ZERO, 1, 0.5, Vec3Fix::ZERO);
        half_bouncy(&mut w);
        w.step(h());
        close(
            w.bodies[k].position,
            v3(4.0, 0.0, 0.0),
            0.0,
            "kinematic target",
        );
        if ccd {
            close(
                w.bodies[d].position,
                v3(5.0, 0.0, 0.0),
                POS_TOL,
                "dynamic position",
            );
            close(
                w.bodies[d].velocity,
                v3(960.0, 0.0, 0.0),
                VEL_TOL,
                "dynamic velocity",
            );
        } else {
            close(
                w.bodies[d].position,
                Vec3Fix::ZERO,
                POS_TOL,
                "off: untouched",
            );
        }
    }
}

/// An off-centre hit: the kinematic body still lands on its target bit for
/// bit (it is never placed by the sweep), and the dynamic body is pushed away
/// from it with a positive speed along the line of centres.
#[test]
fn a_kinematic_body_reaches_its_target_exactly_after_an_oblique_hit() {
    let mut w = bare(true);
    let target = v3(4.3, 0.7, -0.2);
    let mut k = RigidBody::new_kinematic(v3(-6.1, -0.4, 0.3));
    k.kinematic_target = Some((target, k.rotation));
    let k = w.add_body_with_radius(k, f(0.5));
    let d = sphere(&mut w, v3(0.1, 0.3, 0.1), 3, 0.5, Vec3Fix::ZERO);
    half_bouncy(&mut w);
    w.step(h());
    assert_eq!(w.bodies[k].position, target);
    let gap = w.bodies[d].position - w.bodies[k].position;
    assert!(gap.length().to_f64() >= 1.0 - 1e-9, "still overlapping");
    assert!(w.bodies[d].velocity.dot(gap).to_f64() > 0.0);
}

// ── Large displacements ─────────────────────────────────────────────────────

/// `2⁴⁰` m/s (`2³⁴` m in the step) is still stopped on the plate.
#[test]
fn a_speed_of_two_to_the_forty_is_stopped() {
    let (mut w, s) = plate_scene(true, Fix128::from_int(1 << 40));
    w.step(h());
    close(
        w.bodies[s].position,
        v3(5.0 - 1.0 / 64.0 - 0.25, 0.0, 0.0),
        POS_TOL,
        "position",
    );
    assert!(w.bodies[s].velocity.x.to_f64() < 0.0);
}

// ── Determinism ─────────────────────────────────────────────────────────────

/// Fast spheres in every direction among a plate, a still sphere, a triangle
/// and each other, with the default configuration.
fn crowd() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.set_continuous_collision(WorldCcdConfig::on());
    plate(&mut w);
    w.add_body_with_radius(RigidBody::new_static(v3(-4.0, 1.0, 0.0)), Fix128::ONE);
    let tri = TriMesh::from_indexed(
        &[v3(-6.0, -4.0, 6.0), v3(6.0, -4.0, 6.0), v3(0.0, 6.0, 6.0)],
        &[0, 1, 2],
    );
    w.add_static_collider(StaticCollider::TriMesh(tri));
    // a ring of 12 starts, each thrown across the middle at about 400 m/s
    const RING: [(f64, f64); 12] = [
        (3.0, 0.0),
        (2.0, 2.0),
        (0.0, 3.0),
        (-2.0, 2.0),
        (-3.0, 0.0),
        (-2.0, -2.0),
        (0.0, -3.0),
        (2.0, -2.0),
        (3.0, 1.0),
        (1.0, 3.0),
        (-1.0, -3.0),
        (-3.0, -1.0),
    ];
    for (k, &(cx, cz)) in RING.iter().enumerate() {
        let k = k as i64;
        let at = v3(cx, (k % 3) as f64 - 1.0, cz);
        let vel = v3(-cx * 130.0 + 37.0, (k % 5) as f64 * 11.0, -cz * 130.0);
        sphere(&mut w, at, 1 + k % 3, 0.25 + 0.05 * (k % 4) as f64, vel);
    }
    w
}

fn crowd_hash(parallel: bool) -> String {
    let mut w = crowd();
    for _ in 0..60 {
        if parallel {
            #[cfg(feature = "parallel")]
            w.step_parallel(Fix128::from_ratio(1, 60));
        } else {
            w.step(Fix128::from_ratio(1, 60));
        }
    }
    Sha256::digest(w.serialize_state())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The `serialize_state` of [`crowd`] after 60 steps, recorded once: the same
/// with and without `--features parallel`, with `step` and `step_parallel`.
const GOLDEN_CROWD: &str = "0d56880d7cfd1c4b0b086952b084fd98f2c9e243d3cbeedaf28f21f2845e5a1e";

#[test]
fn on_is_deterministic_and_independent_of_the_parallel_feature() {
    let a = crowd_hash(false);
    let b = crowd_hash(false);
    println!("crowd: {a}");
    assert_eq!(a, b);
    assert_eq!(a, GOLDEN_CROWD);
    #[cfg(feature = "parallel")]
    assert_eq!(crowd_hash(true), GOLDEN_CROWD);
}

// ── Snapshot ────────────────────────────────────────────────────────────────

/// The world `tests/fixtures/world_snapshot_v{1,2}_stacked.bin` were taken
/// from, after its first step.
fn stacked() -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 4,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..3 {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 1 + 2 * i, 0), Fix128::ONE),
            Fix128::ONE,
        );
    }
    w.step(Fix128::from_ratio(1, 60));
    w
}

/// A version 1 or 2 blob has no continuous collision section: it is read as
/// the setting off, and the restored world is the world it was taken from,
/// byte for byte, now and after 60 more steps.
#[test]
fn old_snapshots_read_as_off_and_match_the_world_they_came_from() {
    let fixtures: [(&str, &[u8]); 2] = [
        (
            "v1",
            include_bytes!("fixtures/world_snapshot_v1_stacked.bin"),
        ),
        (
            "v2",
            include_bytes!("fixtures/world_snapshot_v2_stacked.bin"),
        ),
    ];
    for (name, blob) in fixtures {
        let mut restored = PhysicsWorld::from_world_snapshot(blob).expect(name);
        assert_eq!(
            restored.continuous_collision(),
            WorldCcdConfig::new(),
            "{name}"
        );
        let mut original = stacked();
        assert_eq!(
            restored.snapshot_world(),
            original.snapshot_world(),
            "{name}"
        );
        for _ in 0..60 {
            original.step(Fix128::from_ratio(1, 60));
            restored.step(Fix128::from_ratio(1, 60));
        }
        assert_eq!(
            restored.serialize_state(),
            original.serialize_state(),
            "{name}"
        );
        assert_eq!(
            restored.snapshot_world(),
            original.snapshot_world(),
            "{name}"
        );
    }
}

/// A version 3 blob carries the setting: a branch restored mid-flight keeps
/// sweeping and stays bit-identical to the original.
#[test]
fn snapshot_carries_the_setting() {
    let mut w = crowd();
    let config = WorldCcdConfig::on().with_motion_threshold(Fix128::from_ratio(3, 4));
    w.set_continuous_collision(config);
    w.step(Fix128::from_ratio(1, 60));
    let blob = w.snapshot_world();
    assert_eq!(&blob[4..6], &3u16.to_le_bytes());
    let mut branch = PhysicsWorld::from_world_snapshot(&blob).expect("v3 blob");
    assert_eq!(branch.continuous_collision(), config);
    for _ in 0..20 {
        w.step(Fix128::from_ratio(1, 60));
        branch.step(Fix128::from_ratio(1, 60));
    }
    assert_eq!(branch.serialize_state(), w.serialize_state());
    assert_eq!(branch.snapshot_world(), w.snapshot_world());
}

/// `reset_world` returns the setting to off, like the rest of the world.
#[test]
fn reset_world_turns_it_off() {
    let mut w = bare(true);
    w.reset_world();
    assert!(!w.continuous_collision().is_enabled());
}
