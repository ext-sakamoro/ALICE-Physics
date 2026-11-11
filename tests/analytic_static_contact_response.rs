//! Oracles for the response of a body touching what it cannot move: a static
//! collider (plane, triangle mesh, height field), a static body with a shape
//! (the body–body contact), and a static SDF collider: the bounce, the friction
//! and the turn the contact gives, for every one of these pairings (each law is
//! checked on each, `scripts/contact_law_matrix.py`).
//!
//! # What is measured (closed forms, independent of the solver)
//!
//! - **Bounce.** A ball dropped from rest `h` above a floor, with restitution `e`
//!   for the ball–floor pair, leaves the floor at `e·v` (`v = √(2 g h)`) and rises
//!   `e² h`. Two restitutions give two heights.
//! - **Sliding friction.** A flat box sliding at `v0` on a level floor with
//!   friction `μ` decelerates at `μ g` and stops after `v0² / (2 μ g)`.
//! - **Rolling.** A solid ball (`I = 2/5 m r²`) set sliding at `v0` with no spin
//!   is slowed by friction and spun up by its torque until it rolls: from then on
//!   `v = 5 v0 / 7` and `ω r = v`, the spin about the axis `up × direction`
//!   reversed (rolling toward `+x` on a floor facing `+y` turns about `−z`).
//! - **Tipping.** A box of width `w` and height `h` standing on one bottom edge,
//!   tilted by `θ`, falls over when its centre of mass is past the edge
//!   (`θ > atan(w / h)`) and falls back onto its base otherwise.
//!
//! The friction / restitution of the pair is set with a pair override between the
//! body's material and [`DEFAULT_MATERIAL`], the material a static collider
//! counts as.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::heightfield::HeightField;
use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::trimesh::{TriMesh, Triangle};

const G: f64 = 10.0;
const DT: f64 = 1.0 / 240.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn backends() -> [(&'static str, SolverBackend); 2] {
    [("xpbd", SolverBackend::Xpbd), ("tgs", SolverBackend::Tgs)]
}

/// What the body stands on: a static collider, a static body with a box shape
/// (its top face at `y = 0`), or a static SDF collider (the half-space `y ≤ 0`).
enum Floor {
    Static(StaticCollider),
    Body,
    Sdf,
}

/// Put `floor` into `w`.
fn install(w: &mut PhysicsWorld, floor: Floor) {
    match floor {
        Floor::Static(c) => {
            w.add_static_collider(c);
        }
        Floor::Body => {
            let b = w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
            w.set_body_shape(
                b,
                &Shape::Box {
                    half_extents: v3(40.0, 1.0, 40.0),
                },
            );
        }
        Floor::Sdf => {
            let field = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
            w.add_sdf_collider(SdfCollider::new_static(
                Box::new(field),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            ));
        }
    }
}

fn plane() -> Floor {
    Floor::Static(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )))
}

/// Every kind of floor at `y = 0`, facing up
fn floors() -> Vec<(&'static str, Floor)> {
    let (a, b, c, d) = (
        v3(-20.0, 0.0, -20.0),
        v3(20.0, 0.0, -20.0),
        v3(20.0, 0.0, 20.0),
        v3(-20.0, 0.0, 20.0),
    );
    vec![
        ("plane", plane()),
        (
            "mesh",
            Floor::Static(StaticCollider::TriMesh(TriMesh::from_triangles(vec![
                Triangle::new(a, c, b),
                Triangle::new(a, d, c),
            ]))),
        ),
        (
            "height field",
            Floor::Static(StaticCollider::HeightField(HeightField::flat(
                41,
                41,
                fx(1.0),
                v3(-20.0, 0.0, -20.0),
                fx(0.0),
            ))),
        ),
        ("static body", Floor::Body),
        ("sdf", Floor::Sdf),
    ]
}

/// A world on `floor` with gravity `−G` along `y`, no damping, and a material
/// whose pair with the floor has `friction` and `restitution`. Returns the world
/// and the material.
fn world(
    backend: SolverBackend,
    floor: Floor,
    friction: f64,
    restitution: f64,
) -> (PhysicsWorld, u16) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    install(&mut w, floor);
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(friction), fx(restitution)));
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, fx(friction), fx(restitution));
    (w, m)
}

fn step(w: &mut PhysicsWorld, seconds: f64) {
    let n = (seconds / DT).round() as usize;
    for _ in 0..n {
        w.step(fx(DT));
    }
}

/// The height the ball's lowest point reaches after its first bounce, from a
/// drop of `h` (ball radius 0.25)
fn bounce_height(backend: SolverBackend, floor: Floor, e: f64, h: f64) -> f64 {
    let r = 0.25;
    let (mut w, m) = world(backend, floor, 0.0, e);
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, r + h, 0.0), Fix128::ONE),
        fx(r),
    );
    w.set_body_material(b, m);
    let mut rising = false;
    let mut apex = f64::NEG_INFINITY;
    for _ in 0..(3.0 / DT) as usize {
        w.step(fx(DT));
        let body = w.get_body(b).expect("ball");
        let (y, vy) = (body.position.y.to_f64(), body.velocity.y.to_f64());
        if vy > 0.0 {
            rising = true;
        }
        if rising {
            apex = apex.max(y - r);
            if vy < 0.0 {
                break;
            }
        }
    }
    apex
}

/// oracle: e² h (free fall, the bounce reverses e·v). Tolerance: the bounce is
/// resolved at the substep of the touch, `h_s = DT / 8`; the speed at the touch is
/// off by at most `g h_s` and the rise by `2 e² v g h_s / g`, which for `h = 1`
/// is `≤ 2 e² √(2 g) h_s` < 0.01 h. 0.02 h covers it with margin.
#[test]
#[ignore = "known defect: static colliders and SDF colliders only push out (no restitution), TGS static contacts too; body-body TGS overshoots at 240 Hz"]
fn a_ball_bounces_to_e_squared_h_on_every_floor_and_backend() {
    let h = 1.0;
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (bname, backend) in backends() {
        for e in [0.15, 0.8] {
            for (fname, floor) in floors() {
                let got = bounce_height(backend, floor, e, h);
                let want = e * e * h;
                compared += 1;
                if !((got - want).abs() <= 0.02 * h) {
                    wrong.push(format!(
                        "{bname} {fname} e {e}: rises {got:.4} (closed form {want:.4})"
                    ));
                }
            }
        }
    }
    assert_eq!(compared, 2 * floors().len() * 2);
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// oracle: a flat box sliding at v0 with friction μ stops after v0² / (2 μ g).
/// Tolerance: the error of the step is first order in `DT`. Measured on the
/// body–body pairing of the temporal Gauss-Seidel backend (a contact that already
/// had friction): 0.0387, 0.0196 and 0.0097 m past the closed form at 240, 480
/// and 960 Hz, halving with the step, so it is the discretisation, about
/// `3.1 v0 DT`. The bound is `4 v0 DT` (0.050 m at 240 Hz, above all three
/// measurements scaled to their step). A
/// friction that is wrong rather than coarse keeps its error as `DT` shrinks
/// (the XPBD static plane stopped at 1.261 m at all three rates).
#[test]
#[ignore = "known defect: static colliders and SDF colliders apply no friction (the box slides on at v0)"]
fn a_sliding_box_stops_after_v0_squared_over_two_mu_g() {
    let (v0, mu) = (3.0, 0.4);
    let want = v0 * v0 / (2.0 * mu * G);
    let shape = Shape::Box {
        half_extents: v3(0.45, 0.06, 0.45),
    };
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (bname, backend) in backends() {
        for (fname, floor) in floors() {
            let (mut w, m) = world(backend, floor, mu, 0.0);
            let b = w
                .add_shaped_body(&shape, fx(500.0), v3(0.0, 0.06, 0.0))
                .expect("pad");
            w.set_body_material(b, m);
            w.get_body_mut(b).expect("pad").velocity = v3(v0, 0.0, 0.0);
            step(&mut w, 2.0);
            let body = w.get_body(b).expect("pad");
            let (x, speed) = (body.position.x.to_f64(), body.velocity.length().to_f64());
            compared += 1;
            if !((x - want).abs() <= 4.0 * v0 * DT && speed <= 1e-3) {
                wrong.push(format!(
                    "{bname} {fname}: stops at {x:.4} (closed form {want:.4}), speed {speed:.3e}"
                ));
            }
        }
    }
    assert_eq!(compared, 2 * floors().len());
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// oracle: a solid ball sliding at v0 with no spin rolls at 5 v0 / 7 with
/// ω r = v once t > 2 v0 / (7 μ g) (here 0.286 s; measured at 1 s). Tolerance:
/// the friction impulse of one substep changes v by at most μ g h_s
/// (= 0.0016 m/s) past the switch to rolling; 0.01 of 5 v0 / 7 covers it.
#[test]
#[ignore = "known defect: no friction torque on static colliders, SDF colliders or the XPBD body-body contact (the ball slides, or stops without spinning)"]
fn a_sliding_ball_rolls_at_five_sevenths_of_v0() {
    let (v0, mu, r) = (3.0, 0.3, 0.25);
    let want = 5.0 * v0 / 7.0;
    let shape = Shape::Ellipsoid { radii: v3(r, r, r) };
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (bname, backend) in backends() {
        for (fname, floor) in floors() {
            let (mut w, m) = world(backend, floor, mu, 0.0);
            let b = w
                .add_shaped_body(&shape, fx(500.0), v3(0.0, r, 0.0))
                .expect("ball");
            w.set_body_material(b, m);
            w.get_body_mut(b).expect("ball").velocity = v3(v0, 0.0, 0.0);
            step(&mut w, 1.0);
            let body = w.get_body(b).expect("ball");
            let (vx, wz) = (body.velocity.x.to_f64(), body.angular_velocity.z.to_f64());
            compared += 1;
            if !((vx - want).abs() <= 0.01 * want && (-wz * r - want).abs() <= 0.01 * want) {
                wrong.push(format!(
                    "{bname} {fname}: v {vx:.4}, -ω_z r {:.4} (closed form both {want:.4})",
                    -wz * r
                ));
            }
        }
    }
    assert_eq!(compared, 2 * floors().len());
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The angle (radians) between the body's local `y` axis and world `y`
fn tilt(rotation: QuatFix) -> f64 {
    let up = rotation.rotate_vec(v3(0.0, 1.0, 0.0));
    up.y.to_f64().clamp(-1.0, 1.0).acos()
}

/// A box (half extents 0.05 × 0.25 × 0.2) standing on its bottom edge at `x =
/// +0.05`, tilted by `theta` toward `+x` (about `−z`), at rest. Its tilt after 2 s.
fn domino_tilt_after(backend: SolverBackend, floor: Floor, theta: f64) -> f64 {
    let (hw, hh) = (0.05, 0.25);
    let (mut w, m) = world(backend, floor, 0.8, 0.0);
    let q = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(-theta));
    // the edge (hw, −hh) of the box in its frame stays at (hw, 0) in the world:
    // centre = edge − R (hw, −hh)
    let (s, c) = theta.sin_cos();
    let centre = v3(hw - (hw * c + hh * s), -(-hw * s - hh * c), 0.0);
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(hw, hh, 0.2),
            },
            fx(500.0),
            centre,
        )
        .expect("domino");
    w.set_body_material(b, m);
    w.get_body_mut(b).expect("domino").set_rotation(q);
    step(&mut w, 2.0);
    tilt(w.get_body(b).expect("domino").rotation)
}

/// oracle: the critical tilt is atan(w / h) = atan(0.1 / 0.5) = 11.31°. 9° falls
/// back upright (tilt < 1°), 14° falls over (tilt within 1° of 90°, lying on its
/// side).
#[test]
#[ignore = "known defect: static colliders, SDF colliders and the XPBD body-body contact give no torque (a tilted box keeps its tilt)"]
fn a_domino_falls_over_only_past_atan_w_over_h() {
    let critical = (0.1f64 / 0.5).atan().to_degrees();
    assert!((critical - 11.31).abs() < 0.01);
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (bname, backend) in backends() {
        for (k, (fname, floor)) in floors().into_iter().enumerate() {
            let back = domino_tilt_after(backend, floor, 9f64.to_radians()).to_degrees();
            let (_, again) = floors().swap_remove(k);
            let over = domino_tilt_after(backend, again, 14f64.to_radians()).to_degrees();
            compared += 2;
            if !(back < 1.0) {
                wrong.push(format!(
                    "{bname} {fname}: 9° ends at {back:.2}° (want upright, < 1°)"
                ));
            }
            if !((over - 90.0).abs() < 1.0) {
                wrong.push(format!(
                    "{bname} {fname}: 14° ends at {over:.2}° (want on its side, 90°)"
                ));
            }
        }
    }
    assert_eq!(compared, 2 * floors().len() * 2);
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Degenerate pair coefficients: restitution 0 gives no bounce, friction 0 keeps
/// the slide speed. Both are the closed forms at the end of their range.
#[test]
fn zero_restitution_does_not_bounce_and_zero_friction_does_not_slow() {
    for (bname, backend) in backends() {
        let rise = bounce_height(backend, plane(), 0.0, 1.0);
        assert!(rise <= 0.005, "{bname}: e 0 rises {rise} (want no bounce)");

        let (mut w, m) = world(backend, plane(), 0.0, 0.0);
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: v3(0.45, 0.06, 0.45),
                },
                fx(500.0),
                v3(0.0, 0.06, 0.0),
            )
            .expect("pad");
        w.set_body_material(b, m);
        w.get_body_mut(b).expect("pad").velocity = v3(2.0, 0.0, 0.0);
        step(&mut w, 1.0);
        let vx = w.get_body(b).expect("pad").velocity.x.to_f64();
        assert!(
            (vx - 2.0).abs() <= 1e-6,
            "{bname}: μ 0 slides at {vx} after 1 s (want 2)"
        );
    }
}

/// The same scene stepped twice ends in the same bits, on both backends.
#[test]
fn the_contact_response_scene_replays_to_the_same_bits() {
    for (_, backend) in backends() {
        let run = || {
            let (mut w, m) = world(backend, plane(), 0.5, 0.4);
            let mut ids = Vec::new();
            for k in 0..4 {
                let b = w
                    .add_shaped_body(
                        &Shape::Box {
                            half_extents: v3(0.05, 0.25, 0.2),
                        },
                        fx(500.0),
                        v3(k as f64, 0.6, 0.0),
                    )
                    .expect("box");
                w.set_body_material(b, m);
                let q = QuatFix::from_axis_angle(v3(0.3, 0.0, 1.0).normalize(), fx(0.2 * k as f64));
                let body = w.get_body_mut(b).expect("box");
                body.set_rotation(q);
                body.velocity = v3(1.0, 0.0, 0.5);
                ids.push(b);
            }
            step(&mut w, 1.0);
            ids.iter()
                .map(|&b| {
                    let body = w.get_body(b).expect("box");
                    format!(
                        "{:?} {:?} {:?} {:?}",
                        body.position, body.velocity, body.rotation, body.angular_velocity
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
