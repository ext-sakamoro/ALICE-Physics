//! `PhysicsWorld2D` stepped against closed forms the example computes itself.
//!
//! 1. **Free fall** (gravity `g = (0, −8)`, `dt = 1/16`, 4 substeps ⇒ `h = 1/64`,
//!    damping 1). The step is semi-implicit Euler per substep (`v += g h`, `x += v h`),
//!    so after `n` substeps from rest `v_n = n g h` and
//!    `y_n = y_0 + g h² n (n + 1) / 2`. All of `g`, `h`, `y_0` are dyadic, so the
//!    discrete solution is reproduced exactly. It differs from the continuous
//!    `y_0 − |g| t² / 2` by exactly `|g| h t / 2`.
//! 2. **Kinematic and static bodies** in the same world: a kinematic body moves
//!    `x = x_0 + v t` with no gravity, a static body never moves.
//! 3. **`check_collision_2d`**: two unit circles 3/2 apart overlap by `depth = 1/2`
//!    along `n = (1, 0)` with the contact point at `x_a + n (r_a − depth / 2)`; a unit
//!    circle 3/4 above a static box top overlaps by `1/4`; separated pairs give `None`.
//! 4. **Resting contact**: a unit circle dropped onto a static circle of radius 10
//!    (top at `y = 0`) settles with its centre one radius above the top and (almost)
//!    zero velocity.
//! 5. **Pendulums**: a `Joint2D::Distance` rod (point bob) swings with the
//!    small-angle period `T = 2π √(L / g)`; a `Joint2D::Revolute` pin at distance `L`
//!    from a disc's centre swings as a physical pendulum,
//!    `T = 2π √((I + m L²) / (m g L))` with `I = m r² / 2`.
//! 6. **`remove_body`** swaps the last body into the removed slot and leaves every
//!    other body's state untouched; an out-of-range index returns `None`.
//!
//! ```bash
//! cargo run --release --example physics2d_world_closed_forms --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    Contact2D, Joint2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};
use std::f64::consts::PI;

fn circle(radius: Fix128) -> Shape2D {
    Shape2D::Circle { radius }
}

/// Axis-aligned box with half extents `(hx, hy)`, CCW winding.
fn box_shape(hx: Fix128, hy: Fix128) -> Shape2D {
    Shape2D::Polygon {
        vertices: vec![
            Vec2Fix::new(-hx, -hy),
            Vec2Fix::new(hx, -hy),
            Vec2Fix::new(hx, hy),
            Vec2Fix::new(-hx, hy),
        ],
    }
}

fn config(gravity_y: i64, substeps: usize, damping: Fix128) -> PhysicsConfig2D {
    PhysicsConfig2D {
        gravity: Vec2Fix::from_int(0, gravity_y),
        substeps,
        iterations: 8,
        damping,
    }
}

fn free_fall_kinematic_static() {
    let (g, substeps, steps) = (8_i64, 4_usize, 16_i64);
    let mut world = PhysicsWorld2D::new(config(-g, substeps, Fix128::ONE));
    let y0 = 100_i64;
    let ball = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(0, y0),
        Fix128::ONE,
        circle(Fix128::ONE),
    ));
    let mut cart = RigidBody2D::new_kinematic(Vec2Fix::from_int(50, 0), circle(Fix128::ONE));
    cart.velocity = Vec2Fix::from_int(2, 0);
    let cart = world.add_body(cart);
    let wall = world.add_body(RigidBody2D::new_static(
        Vec2Fix::from_int(-50, 0),
        box_shape(Fix128::ONE, Fix128::ONE),
    ));
    let dt = Fix128::from_ratio(1, 16);
    for _ in 0..steps {
        world.step(dt);
    }

    // discrete closed form: n substeps of h = dt / substeps
    let n = steps * substeps as i64;
    let h_den = 16 * substeps as i64; // h = 1 / h_den
    let v_expected = Fix128::from_ratio(-g * n, h_den);
    let drop = Fix128::from_ratio(g * n * (n + 1), 2 * h_den * h_den);
    let y_expected = Fix128::from_int(y0) - drop;
    let b = &world.bodies[ball];
    assert_eq!(b.velocity.y, v_expected, "v_n = n g h");
    assert_eq!(b.position.y, y_expected, "y_n = y0 + g h² n(n+1)/2");
    assert_eq!(b.position.x, Fix128::ZERO, "no horizontal drift");

    // relation to the continuous parabola: discrete drop − |g| t²/2 = |g| h t / 2
    let t = steps as f64 / 16.0;
    let h = 1.0 / h_den as f64;
    let continuous_drop = g as f64 * t * t / 2.0;
    let gap = drop.to_f64() - continuous_drop;
    assert!(
        (gap - g as f64 * h * t / 2.0).abs() < 1e-15,
        "discrete − continuous drop = |g| h t / 2 (got {gap})"
    );

    // kinematic: x = x0 + v t, no gravity; static: unchanged
    let c = &world.bodies[cart];
    assert_eq!(
        c.position,
        Vec2Fix::from_int(50 + 2 * steps / 16, 0),
        "kinematic x = x0 + v t"
    );
    assert_eq!(c.velocity, Vec2Fix::from_int(2, 0), "kinematic v unchanged");
    assert_eq!(
        world.bodies[wall].position,
        Vec2Fix::from_int(-50, 0),
        "static unmoved"
    );
    println!(
        "free fall: y = {} after t = {t} s (discrete closed form {}, continuous {}), v = {}",
        b.position.y.to_f64(),
        y_expected.to_f64(),
        y0 as f64 - continuous_drop,
        b.velocity.y.to_f64()
    );
}

fn collision_queries() {
    let mut world = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let one = Fix128::ONE;
    let a = world.add_body(RigidBody2D::new_dynamic(Vec2Fix::ZERO, one, circle(one)));
    let b = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO),
        one,
        circle(one),
    ));
    let far = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(10, 0),
        one,
        circle(one),
    ));
    let floor = world.add_body(RigidBody2D::new_static(
        Vec2Fix::new(Fix128::from_int(40), Fix128::from_ratio(-1, 2)),
        box_shape(Fix128::from_int(4), Fix128::from_ratio(1, 2)),
    ));
    let disc = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::new(Fix128::from_int(40), Fix128::from_ratio(3, 4)),
        one,
        circle(one),
    ));

    // two unit circles at distance d = 3/2: depth = r_a + r_b − d = 1/2
    let c: Contact2D = world.check_collision_2d(a, b).expect("overlapping circles");
    let (ra, d) = (1.0, 1.5);
    let depth = 2.0 - d;
    let close = |x: Fix128, e: f64, what: &str| {
        assert!(
            (x.to_f64() - e).abs() < 1e-12,
            "{what}: {} vs {e}",
            x.to_f64()
        );
    };
    close(c.depth, depth, "circle-circle depth");
    close(c.normal.x, 1.0, "normal points from a to b (x)");
    close(c.normal.y, 0.0, "normal points from a to b (y)");
    close(
        c.point.x,
        ra - depth / 2.0,
        "contact point x_a + n (r_a − depth/2)",
    );
    assert_eq!(
        (c.body_a, c.body_b),
        (a, b),
        "contact carries the body indices"
    );
    assert!(
        world.check_collision_2d(a, far).is_none(),
        "separated circles: None"
    );

    // unit circle 3/4 above a box top at y = 0: depth = 1/4 in both orders
    let up = world
        .check_collision_2d(disc, floor)
        .expect("circle on box");
    let down = world
        .check_collision_2d(floor, disc)
        .expect("box under circle");
    close(up.depth, 0.25, "circle-box depth");
    close(down.depth, 0.25, "box-circle depth");
    close(up.normal.x, 0.0, "circle-box normal is vertical");
    close(up.normal.y.abs(), 1.0, "circle-box normal is unit");
    assert_eq!(
        up.normal, -down.normal,
        "swapping the pair flips the normal"
    );
    println!(
        "check_collision_2d: depth {} (expect {depth}), circle-box depth {} (expect 0.25)",
        c.depth.to_f64(),
        up.depth.to_f64()
    );
}

fn resting_contact() {
    let mut world = PhysicsWorld2D::new(PhysicsConfig2D::default());
    // A static circle, not a box: the circle-polygon contact normal currently points
    // from the polygon to the circle (against the `Contact2D::normal` doc, "from
    // body_a toward body_b") in both pair orders, so a circle stepped onto a static
    // polygon is pushed through it. Only the orientation-free facts of that pair
    // (depth, |n| = 1, sign flip on swap) are asserted in `collision_queries`.
    let floor = world.add_body(RigidBody2D::new_static(
        Vec2Fix::from_int(0, -10),
        circle(Fix128::from_int(10)),
    ));
    let ball = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::new(Fix128::ZERO, Fix128::from_ratio(3, 2)),
        Fix128::ONE,
        circle(Fix128::ONE),
    ));
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..300 {
        world.step(dt);
    }
    let b = &world.bodies[ball];
    let (y, v) = (b.position.y.to_f64(), b.velocity.length().to_f64());
    // equilibrium: centre one radius above the floor top (y = 0)
    assert!((y - 1.0).abs() < 0.02, "resting height = radius (got {y})");
    assert!(v < 0.05, "resting speed ≈ 0 (got {v})");
    assert!(b.position.x.to_f64().abs() < 1e-9, "no sideways drift");
    assert_eq!(
        world.bodies[floor].position,
        Vec2Fix::from_int(0, -10),
        "static floor unmoved"
    );
    println!("resting contact: y = {y:.5} (expect 1), |v| = {v:.2e}");
}

/// Times at which `x(t)` crosses zero, linearly interpolated within a step.
fn zero_crossings(xs: &[f64], dt: f64) -> Vec<f64> {
    xs.windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] != 0.0 && w[0].signum() != w[1].signum())
        .map(|(i, w)| (i as f64 + w[0] / (w[0] - w[1])) * dt)
        .collect()
}

/// Mean full period from successive zero crossings (two crossings per period).
fn period_from(xs: &[f64], dt: f64) -> f64 {
    let z = zero_crossings(xs, dt);
    assert!(
        z.len() >= 5,
        "need at least two full periods, got {} crossings",
        z.len()
    );
    2.0 * (z[z.len() - 1] - z[0]) / (z.len() - 1) as f64
}

fn pendulums() {
    let g = 10.0;
    let length = 1.0;
    let theta0: f64 = 0.1;
    let (steps_per_s, seconds) = (120_i64, 7_i64);
    let dt = 1.0 / steps_per_s as f64;
    let small = Fix128::from_ratio(1, 10);

    // distance rod, point bob
    let mut world = PhysicsWorld2D::new(config(-10, 8, Fix128::ONE));
    let pivot = world.add_body(RigidBody2D::new_static(Vec2Fix::ZERO, circle(small)));
    // initial angle from the crate's deterministic sin/cos
    let (sin0, cos0) = Fix128::from_f64(theta0).sin_cos();
    let start = Vec2Fix::new(
        Fix128::from_f64(length) * sin0,
        -(Fix128::from_f64(length) * cos0),
    );
    let x0 = length * sin0.to_f64();
    let bob = world.add_body(RigidBody2D::new_dynamic(start, Fix128::ONE, circle(small)));
    world.add_joint(Joint2D::Distance {
        body_a: pivot,
        body_b: bob,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::ZERO,
        target_distance: Fix128::from_f64(length),
        compliance: Fix128::ZERO,
    });

    // revolute pin at distance L above the centre of a disc of radius r
    let r = 0.1;
    let mut phys = PhysicsWorld2D::new(config(-10, 8, Fix128::ONE));
    let pin = phys.add_body(RigidBody2D::new_static(Vec2Fix::ZERO, circle(small)));
    let mut disc = RigidBody2D::new_dynamic(start, Fix128::ONE, circle(Fix128::from_f64(r)));
    disc.angle = Fix128::from_f64(theta0);
    disc.prev_angle = disc.angle;
    let disc = phys.add_body(disc);
    phys.add_joint(Joint2D::Revolute {
        body_a: pin,
        body_b: disc,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::new(Fix128::ZERO, Fix128::from_f64(length)),
        compliance: Fix128::ZERO,
    });

    let step = Fix128::from_ratio(1, steps_per_s);
    let (mut xs, mut xs_phys) = (vec![], vec![]);
    let mut max_len_err: f64 = 0.0;
    let mut max_pin_err: f64 = 0.0;
    for _ in 0..steps_per_s * seconds {
        world.step(step);
        phys.step(step);
        let p = world.bodies[bob].position;
        xs.push(p.x.to_f64());
        max_len_err = max_len_err.max((p.length().to_f64() - length).abs());
        let anchor =
            phys.bodies[disc].world_point(Vec2Fix::new(Fix128::ZERO, Fix128::from_f64(length)));
        max_pin_err = max_pin_err.max(anchor.length().to_f64());
        xs_phys.push(phys.bodies[disc].position.x.to_f64());
    }

    let t_simple = 2.0 * PI * (length / g).sqrt();
    let t_meas = period_from(&xs, dt);
    let inertia_ratio = 0.5 * r * r / (length * length); // I / (m L²)
    let t_physical = t_simple * (1.0 + inertia_ratio).sqrt();
    let t_meas_phys = period_from(&xs_phys, dt);
    println!(
        "pendulum (distance): T = {t_meas:.5} s vs 2π√(L/g) = {t_simple:.5} s, max |len − L| = {max_len_err:.2e}"
    );
    println!(
        "pendulum (revolute): T = {t_meas_phys:.5} s vs 2π√((I+mL²)/(mgL)) = {t_physical:.5} s, max pin gap = {max_pin_err:.2e}"
    );
    assert!(
        ((t_meas - t_simple) / t_simple).abs() < 0.01,
        "distance pendulum period within 1 % of 2π√(L/g)"
    );
    assert!(
        ((t_meas_phys - t_physical) / t_physical).abs() < 0.01,
        "revolute pendulum period within 1 % of the physical-pendulum period"
    );
    assert!(
        max_len_err < 1e-3,
        "rod length held (max err {max_len_err:e})"
    );
    assert!(max_pin_err < 1e-3, "pin held (max gap {max_pin_err:e})");
    let amp = xs.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(
        amp <= x0 * 1.01,
        "amplitude does not grow (got {amp}, start {})",
        x0
    );
}

fn remove_keeps_others() {
    let mut world = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let one = Fix128::ONE;
    let mut bodies = vec![];
    for i in 0..3 {
        let mut b = RigidBody2D::new_dynamic(Vec2Fix::from_int(10 * i, i), one, circle(one));
        b.velocity = Vec2Fix::from_int(i, -i);
        bodies.push(b);
    }
    for b in &bodies {
        world.add_body(b.clone());
    }
    let removed = world.remove_body(0).expect("index 0 is valid");
    assert_eq!(
        removed.position, bodies[0].position,
        "returns the removed body"
    );
    assert_eq!(world.bodies.len(), 2, "one body fewer");
    // swap_remove: the last body moves into slot 0, slot 1 is untouched
    assert_eq!(
        world.bodies[0].position, bodies[2].position,
        "last moved into slot 0"
    );
    assert_eq!(
        world.bodies[0].velocity, bodies[2].velocity,
        "with its velocity"
    );
    assert_eq!(
        world.bodies[1].position, bodies[1].position,
        "slot 1 untouched"
    );
    assert_eq!(
        world.bodies[1].velocity, bodies[1].velocity,
        "slot 1 velocity untouched"
    );
    assert!(
        world.remove_body(5).is_none(),
        "out-of-range index returns None"
    );
    assert_eq!(world.bodies.len(), 2, "failed removal changes nothing");
    println!("remove_body: swap-remove semantics hold, out-of-range = None");
}

fn main() {
    free_fall_kinematic_static();
    collision_queries();
    resting_contact();
    pendulums();
    remove_keeps_others();
    println!("physics2d_world_closed_forms: all closed forms hold");
}
