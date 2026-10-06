//! Independent closed-form checks of the contacts between a shaped (or
//! compound) body and a plain sphere body, decided by GJK/EPA, and of the
//! tight world boxes of the shape helpers.
//!
//! Expected values come from the geometry of each scene, computed here:
//!
//! - a ball of radius `r` resting on the top of a static solid rests with its
//!   centre at the solid's top plus `r` (box turned about `y`, upright
//!   cylinder, ellipsoid); a ball cradled in the hole of a flat torus
//!   (`R − ρ < r`) rests where it touches the tube circle all round:
//!   `y = √((r + ρ)² − R²)`;
//! - a flat-bottomed shaped box, or a compound of two boxes with a flat
//!   bottom, resting on the top of a large static plain sphere of radius `S`
//!   has its centre at `S + h_y` (stable: `h_y < S`);
//! - a ball falling past a long rod inside the rod's bounding sphere but
//!   clear of the rod itself is not touched: its fall is the free fall;
//!   a ball whose path meets the rod stops on it at the rod's top plus `r`;
//! - the world box of a turned cylinder, cone, ellipsoid and torus contains
//!   every point of a dense sampling of its extreme curves (parametrised
//!   here) and exceeds their extent by no more than the sampling error.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::AABB;
use alice_physics::compound::CompoundShape;
use alice_physics::cone::Cone;
use alice_physics::cylinder::Cylinder;
use alice_physics::ellipsoid::Ellipsoid;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::torus::Torus;

type V = [f64; 3];

const DT: f64 = 1.0 / 60.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}

fn to_f(v: Vec3Fix) -> V {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// Resting tolerance: the steady penetration of a resting contact under the
/// default 8 substeps of 1/480 s is of order `g h² = 4e-5`; `5e-3` also
/// covers a GJK/EPA depth error, and is far below any of the wrong answers
/// (bounding sphere: tens of centimetres; missed contact: falls through).
const REST_TOL: f64 = 5e-3;

/// Steps `frames` frames and returns the position and speed of `body`.
fn settle(w: &mut PhysicsWorld, body: usize, frames: usize) -> (V, f64) {
    for _ in 0..frames {
        w.step(fx(DT));
    }
    let b = w.get_body(body).expect("body");
    (to_f(b.position), b.velocity.length().to_f64())
}

fn static_shape(w: &mut PhysicsWorld, shape: &Shape, at: V, rotation: QuatFix) -> usize {
    let k = w.add_body(RigidBody::new_static(v3(at)));
    w.bodies[k].rotation = rotation;
    assert!(w.set_body_shape(k, shape));
    k
}

fn ball(w: &mut PhysicsWorld, at: V, r: f64) -> usize {
    w.add_body_with_radius(RigidBody::new_dynamic(v3(at), Fix128::ONE), fx(r))
}

/// A ball on a turned box, an upright cylinder, an ellipsoid and in the hole
/// of a torus, with the ball added before and after the solid.
#[test]
fn a_ball_rests_at_the_closed_form_height_on_each_static_shape() {
    let r = 0.5;
    let yaw =
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(std::f64::consts::FRAC_PI_6)).normalize();
    let torus_r = (1.0f64, 0.3f64);
    // (name, shape, rotation, ball radius, ball start, rest height)
    let scenes: Vec<(&str, Shape, QuatFix, f64, V, f64)> = vec![
        (
            "box turned 30 deg about y",
            Shape::Box {
                half_extents: v3([1.5, 0.4, 1.0]),
            },
            yaw,
            r,
            [0.2, 1.5, -0.1],
            0.4 + r,
        ),
        (
            "upright cylinder",
            Shape::Cylinder {
                radius: fx(1.2),
                half_height: fx(0.6),
            },
            QuatFix::IDENTITY,
            r,
            [0.3, 1.6, 0.2],
            0.6 + r,
        ),
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: v3([1.5, 0.7, 1.1]),
            },
            QuatFix::IDENTITY,
            r,
            [0.0, 1.8, 0.0],
            0.7 + r,
        ),
        (
            "torus, on the top of the tube",
            Shape::Torus {
                major_radius: fx(torus_r.0),
                minor_radius: fx(torus_r.1),
            },
            QuatFix::IDENTITY,
            r,
            [torus_r.0, 1.5, 0.0],
            torus_r.1 + r,
        ),
    ];
    let mut failures = Vec::new();
    for (name, shape, rotation, rb, start, rest) in &scenes {
        for ball_first in [false, true] {
            let mut w = PhysicsWorld::new(SolverConfig::default());
            let k = if ball_first {
                let k = ball(&mut w, *start, *rb);
                static_shape(&mut w, shape, [0.0; 3], *rotation);
                k
            } else {
                static_shape(&mut w, shape, [0.0; 3], *rotation);
                ball(&mut w, *start, *rb)
            };
            let (p, speed) = settle(&mut w, k, 300);
            eprintln!(
                "{name} ball_first={ball_first}: y {} (want {rest}), speed {speed:.2e}",
                p[1]
            );
            if (p[1] - rest).abs() > REST_TOL || speed > 1e-2 {
                failures.push(format!(
                    "{name} ball_first={ball_first}: rests at y {} speed {speed:.3e}, want {rest}",
                    p[1]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A ball of radius `r = 1` in the hole of a flat torus `R = 1, ρ = 0.3`
/// touches the tube circle all round at `y = √((r + ρ)² − R²) = 0.8307`.
/// Measured: it rests at `y = 1.3000 = ρ + r` in both index orders, on the
/// flat top of the torus's convex hull: GJK/EPA works on the support function,
/// which is the hull's, so the hole is filled for every narrow-phase pair.
#[test]
#[ignore = "known limitation: a torus collides as its convex hull, a ball cannot enter the hole (rests at rho + r = 1.3, want 0.8307)"]
fn a_ball_is_cradled_in_the_hole_of_a_torus() {
    let (big_r, rho, r): (f64, f64, f64) = (1.0, 0.3, 1.0);
    let want = ((r + rho) * (r + rho) - big_r * big_r).sqrt();
    let mut w = PhysicsWorld::new(SolverConfig::default());
    static_shape(
        &mut w,
        &Shape::Torus {
            major_radius: fx(big_r),
            minor_radius: fx(rho),
        },
        [0.0; 3],
        QuatFix::IDENTITY,
    );
    let k = ball(&mut w, [0.0, 2.5, 0.0], r);
    let (p, _) = settle(&mut w, k, 300);
    assert!(
        (p[1] - want).abs() < REST_TOL,
        "rests at y {}, want {want}",
        p[1]
    );
}

/// A dynamic shaped box and a dynamic compound with the same flat bottom rest
/// on the top of a static plain sphere of radius `S = 3` at `S + h_y`; the
/// pair is decided by GJK on the collider and the sphere's support, in both
/// index orders.
#[test]
fn a_shaped_box_and_a_compound_rest_on_top_of_a_plain_static_sphere() {
    let s = 3.0;
    let hy = 0.25;
    let mut failures = Vec::new();
    for compound in [false, true] {
        for sphere_first in [false, true] {
            let mut w = PhysicsWorld::new(SolverConfig::default());
            let add_sphere = |w: &mut PhysicsWorld| {
                w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), fx(s))
            };
            let add_box = |w: &mut PhysicsWorld| {
                let start = v3([0.0, s + hy + 0.3, 0.0]);
                if compound {
                    let mut c = CompoundShape::new();
                    for x in [-0.5, 0.5] {
                        c.add_box(
                            OrientedBox::new(Vec3Fix::ZERO, v3([0.5, hy, 0.5]), QuatFix::IDENTITY),
                            v3([x, 0.0, 0.0]),
                            QuatFix::IDENTITY,
                        );
                    }
                    w.add_compound_body(&c, Fix128::from_int(500), start)
                        .expect("a valid compound")
                } else {
                    w.add_shaped_body(
                        &Shape::Box {
                            half_extents: v3([1.0, hy, 0.5]),
                        },
                        Fix128::from_int(500),
                        start,
                    )
                    .expect("a valid box")
                }
            };
            let k = if sphere_first {
                add_sphere(&mut w);
                add_box(&mut w)
            } else {
                let k = add_box(&mut w);
                add_sphere(&mut w);
                k
            };
            let (p, speed) = settle(&mut w, k, 300);
            let want = s + hy;
            eprintln!(
                "compound={compound} sphere_first={sphere_first}: y {} (want {want})",
                p[1]
            );
            if (p[1] - want).abs() > REST_TOL || p[0].abs() > REST_TOL || speed > 1e-2 {
                failures.push(format!(
                    "compound={compound} sphere_first={sphere_first}: at {p:?} speed {speed:.3e}, \
                     want y {want}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A static rod of half-extents `3 × 0.2 × 0.2` (bounding radius `√9.08 ≈
/// 3.013`). A ball of radius 0.3 dropped at `x = 2, z = 0.8` (clear of the rod
/// by `0.8 − 0.2 − 0.3 = 0.3`, well inside the bounding sphere) falls freely:
/// under no damping and `g = 10`, the semi-implicit Euler substeps give the
/// exact discrete free fall `v = −g t`, and `y` below the rod. A ball dropped at
/// `z = 0.15` (over the top face) stops on the rod at `0.2 + 0.3`.
#[test]
fn a_ball_passes_the_rod_inside_its_bounding_sphere_and_stops_on_the_rod_itself() {
    let run = |z: f64| {
        let mut w = PhysicsWorld::new(SolverConfig {
            damping: Fix128::ONE,
            ..SolverConfig::default()
        });
        static_shape(
            &mut w,
            &Shape::Box {
                half_extents: v3([3.0, 0.2, 0.2]),
            },
            [0.0; 3],
            QuatFix::IDENTITY,
        );
        let k = ball(&mut w, [2.0, 1.0, z], 0.3);
        let frames = 60;
        for _ in 0..frames {
            w.step(fx(DT));
        }
        let b = w.get_body(k).expect("ball");
        (to_f(b.position), to_f(b.velocity), frames as f64 * DT)
    };
    let (p, v, t) = run(0.8);
    let g = 10.0;
    assert!(
        (v[1] + g * t).abs() < 1e-9 && v[0].abs() < 1e-12 && v[2].abs() < 1e-12,
        "the clear ball was touched: velocity {v:?}, free fall {}",
        -g * t
    );
    assert!(
        p[1] < -2.0,
        "the clear ball did not fall past the rod: {p:?}"
    );
    let (p, v, _) = run(0.15);
    assert!(
        (p[1] - 0.5).abs() < REST_TOL && v[1].abs() < 1e-2,
        "the ball over the rod rests at {p:?} with {v:?}, want y 0.5"
    );
}

// ---------------------------------------------------------------------------
// Tight boxes of the shape helpers
// ---------------------------------------------------------------------------

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

fn random_rotation(s: &mut u64) -> QuatFix {
    loop {
        let k = [lcg(s) - 0.5, lcg(s) - 0.5, lcg(s) - 0.5];
        let n = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        if n > 0.1 {
            return QuatFix::from_axis_angle(
                v3([k[0] / n, k[1] / n, k[2] / n]),
                fx(lcg(s) * std::f64::consts::PI),
            )
            .normalize();
        }
    }
}

/// Rotation matrix of a quaternion, from its components.
fn matrix(q: QuatFix) -> [[f64; 3]; 3] {
    let (x, y, z, w) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64());
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn turn(m: &[[f64; 3]; 3], p: V) -> V {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
    ]
}

/// Per-axis extent of local points turned by `m` about `c`, each point
/// inflated by `pad` (a sphere swept along the curve).
fn extent(m: &[[f64; 3]; 3], c: V, pts: &[V], pad: f64) -> (V, V) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        let q = turn(m, *p);
        for i in 0..3 {
            lo[i] = lo[i].min(c[i] + q[i] - pad);
            hi[i] = hi[i].max(c[i] + q[i] + pad);
        }
    }
    (lo, hi)
}

fn circle(radius: f64, y: f64, n: usize) -> Vec<V> {
    (0..n)
        .map(|k| {
            let u = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [radius * u.cos(), y, radius * u.sin()]
        })
        .collect()
}

/// The box contains the sampled extent (to `1e-9`) and exceeds it by at most
/// `slack`: the sampling error of the curves (`size · (π/n)² / 2`).
fn check_tight(what: &str, aabb: AABB, (lo, hi): (V, V), slack: f64, failures: &mut Vec<String>) {
    let (bl, bh) = (to_f(aabb.min), to_f(aabb.max));
    for i in 0..3 {
        let loose = (lo[i] - bl[i]).max(bh[i] - hi[i]);
        let short = (bl[i] - lo[i]).max(hi[i] - bh[i]);
        if short > 1e-9 || loose > slack {
            failures.push(format!(
                "{what} axis {i}: box [{:.6}, {:.6}] vs sampled [{:.6}, {:.6}]",
                bl[i], bh[i], lo[i], hi[i]
            ));
        }
    }
}

/// Extreme curves: a cylinder's two cap rims, a cone's apex and base rim, an
/// ellipsoid's surface (a dense latitude-longitude net) and a torus's tube
/// centre circle swept by the tube radius. 40 random rotations each.
#[test]
fn turned_shape_boxes_are_the_tight_extent_of_their_surfaces() {
    let n = 2048;
    let slack_ring = |size: f64| size * (std::f64::consts::PI / n as f64).powi(2);
    let mut s = 0x7157_a11du64;
    let mut failures = Vec::new();
    let c = [0.7, -1.3, 2.1];
    for _ in 0..40 {
        let q = random_rotation(&mut s);
        let m = matrix(q);
        // cylinder r 0.8, hh 1.5
        let mut pts = circle(0.8, 1.5, n);
        pts.extend(circle(0.8, -1.5, n));
        let cyl = Cylinder {
            rotation: q,
            ..Cylinder::new(v3(c), fx(1.5), fx(0.8))
        };
        check_tight(
            "cylinder",
            cyl.aabb(),
            extent(&m, c, &pts, 0.0),
            slack_ring(0.8),
            &mut failures,
        );
        // cone r 0.9, hh 1.2 (apex +y)
        let mut pts = circle(0.9, -1.2, n);
        pts.push([0.0, 1.2, 0.0]);
        let cone = Cone::with_rotation(v3(c), fx(0.9), fx(1.2), q);
        check_tight(
            "cone",
            cone.aabb(),
            extent(&m, c, &pts, 0.0),
            slack_ring(0.9),
            &mut failures,
        );
        // ellipsoid 1.4 x 0.5 x 0.9: latitude-longitude net of 512 x 1024
        let radii = [1.4, 0.5, 0.9];
        let mut pts = Vec::new();
        for a in 0..=512 {
            let th = std::f64::consts::PI * a as f64 / 512.0;
            for b in 0..1024 {
                let ph = 2.0 * std::f64::consts::PI * b as f64 / 1024.0;
                pts.push([
                    radii[0] * th.sin() * ph.cos(),
                    radii[1] * th.cos(),
                    radii[2] * th.sin() * ph.sin(),
                ]);
            }
        }
        let ell = Ellipsoid {
            rotation: q,
            ..Ellipsoid::new(v3(c), v3(radii))
        };
        let slack_ell = 1.4 * (std::f64::consts::PI / 512.0).powi(2);
        check_tight(
            "ellipsoid",
            ell.aabb(),
            extent(&m, c, &pts, 0.0),
            slack_ell,
            &mut failures,
        );
        // torus R 1.1, ρ 0.25, ring in local xz
        let pts = circle(1.1, 0.0, n);
        let tor = Torus {
            rotation: q,
            ..Torus::new(v3(c), fx(1.1), fx(0.25))
        };
        check_tight(
            "torus",
            tor.aabb(),
            extent(&m, c, &pts, 0.25),
            slack_ring(1.1),
            &mut failures,
        );
    }
    assert!(
        failures.is_empty(),
        "{} misfits:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
