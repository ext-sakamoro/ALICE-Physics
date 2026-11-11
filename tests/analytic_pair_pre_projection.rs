//! Two bodies set down overlapping are pushed apart before the substep moves
//! them, and the push makes no motion: it is not an approach of this substep.
//!
//! oracle (momentum and the centre of mass): the push-out is the rigid
//! least-constraint one between the two bodies, impulses `+y` on one and `−y`
//! on the other (`Δx = W y`). Equal and opposite impulses leave the total
//! linear momentum and the mass-weighted centre where they were (`Σ mᵢ Δxᵢ =
//! 0`); the poses move with their previous poses, so no velocity appears and
//! the total linear and angular momentum stay zero. After the push no point of
//! the overlap is left (the rigid push leaves at most `2⁻⁵⁰` of it).
//!
//! What it catches: (1) an overlap turned into velocity (a pop-out of
//! `depth / h`, here metres per second); (2) a push that moves only one body,
//! or both by the same amount (the centre of mass moves).

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Two boxes of densities 1 and 3, the upper one turned about `z` and off
/// centre, set down overlapping by `0.1` along `y`, without gravity.
fn scene(substeps: usize) -> (PhysicsWorld, usize, usize) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        ..SolverConfig::default()
    });
    let lower = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(1.0, 0.5, 0.8),
            },
            fx(1.0),
            v3(0.0, 0.0, 0.0),
        )
        .expect("lower");
    let upper = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(0.4, 0.3, 0.5),
            },
            fx(3.0),
            v3(0.35, 0.7, -0.2),
        )
        .expect("upper");
    w.get_body_mut(upper).expect("upper").rotation =
        QuatFix::from_axis_angle(v3(0.0, 1.0, 0.0), fx(0.3));
    (w, lower, upper)
}

fn momentum(w: &PhysicsWorld, ids: &[usize]) -> (f64, f64, [f64; 3]) {
    let (mut lin, mut ang) = (Vec3Fix::ZERO, Vec3Fix::ZERO);
    let (mut mass, mut moment) = (Fix128::ZERO, Vec3Fix::ZERO);
    for &i in ids {
        let b = w.get_body(i).expect("body");
        let m = Fix128::ONE / b.inv_mass;
        lin = lin + b.velocity * m;
        let iw = Vec3Fix::new(
            b.angular_velocity.x / b.inv_inertia.x,
            b.angular_velocity.y / b.inv_inertia.y,
            b.angular_velocity.z / b.inv_inertia.z,
        );
        ang = ang + b.position.cross(b.velocity * m) + iw;
        mass = mass + m;
        moment = moment + b.position * m;
    }
    let c = moment / mass;
    (
        lin.length().to_f64(),
        ang.length().to_f64(),
        [c.x.to_f64(), c.y.to_f64(), c.z.to_f64()],
    )
}

/// After one step at 1 and 8 substeps: linear and angular momentum stay 0
/// (`1e-12`), the centre of mass stays (`1e-12` m), and the two boxes no
/// longer overlap (the step's contacts carry no depth over `1e-12` m).
#[test]
fn an_overlapping_pair_is_pushed_apart_without_momentum() {
    let mut wrong = Vec::new();
    for substeps in [1usize, 8] {
        let (mut w, a, b) = scene(substeps);
        let (_, _, c0) = momentum(&w, &[a, b]);
        w.step(fx(1.0 / 60.0));
        let (lin, ang, c1) = momentum(&w, &[a, b]);
        let shift = (0..3).map(|k| (c1[k] - c0[k]).abs()).fold(0.0, f64::max);
        let depth = w
            .contact_constraints
            .iter()
            .map(|c| c.contact.depth.to_f64())
            .fold(0.0, f64::max);
        let top = w.get_body(b).expect("upper").position.y.to_f64();
        eprintln!("{substeps}: |p| {lin:.3e} |L| {ang:.3e} centre moved {shift:.3e} depth {depth:.3e} upper y {top:.6}");
        if lin > 1e-12 || ang > 1e-12 || shift > 1e-12 || depth > 1e-12 {
            wrong.push(format!(
                "{substeps} substeps: |p| {lin:.3e}, |L| {ang:.3e}, centre moved {shift:.3e} m, depth left {depth:.3e}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
