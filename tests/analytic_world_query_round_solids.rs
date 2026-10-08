//! Oracles for world queries against a cylinder and a cone at the tolerance
//! `2⁻³²`: small spheres and capsules cast onto them stop at the surface, and a
//! small box beside a curved surface overlaps it only when it reaches in.
//!
//! # Expected values
//!
//! The reference distances are computed here in `f64`, independently of the code
//! under test: a cylinder and a cone are solids of revolution, so the distance of
//! a point is the distance in its meridian half-plane `(ρ, y)` to a rectangle or
//! a triangle; a segment's distance is the minimum of that convex function along
//! it, found by golden-section search. The bounds for the cone are the worst gaps
//! of the previous implementation on the same inputs (its side is not yet within
//! `2⁻³²`), so they guard against it getting worse.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// `2⁻³²`.
const TOL: f64 = 1.0 / 4_294_967_296.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2]))
}

struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// `2⁻ᵉ·(1 + u)` with `e` uniform in `4..=16`.
    fn radius(&mut self) -> f64 {
        let e = 4 + ((self.u() * 13.0) as u32).min(12);
        (1.0 + self.u()) / ((1u64 << e) as f64)
    }
}

/// `v` rotated by the inverse of the unit quaternion `q = (w, x, y, z)`.
fn rotate_back(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], -q[1], -q[2], -q[3]);
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

fn segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let u = (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
    let (dx, dy) = (p.0 - a.0 - u * ex, p.1 - a.1 - u * ey);
    (dx * dx + dy * dy).sqrt()
}

/// A solid of revolution about the local `Y` axis through its geometric centre
/// (`offset` along `Y` from the body position), outside distance in `(ρ, y)`.
#[derive(Clone, Copy)]
enum Solid {
    /// Radius and half height.
    Cylinder(f64, f64),
    /// Base radius and half height; the geometric centre is `h/2` above the body
    /// position (the centre of mass is a quarter of the height above the base).
    Cone(f64, f64),
}

impl Solid {
    fn shape(self) -> Shape {
        match self {
            Self::Cylinder(r, h) => Shape::Cylinder {
                radius: fx(r),
                half_height: fx(h),
            },
            Self::Cone(r, h) => Shape::Cone {
                radius: fx(r),
                half_height: fx(h),
            },
        }
    }

    /// The distance from a local point outside the solid (`0` inside).
    fn distance_local(self, l: [f64; 3]) -> f64 {
        let rho = (l[0] * l[0] + l[2] * l[2]).sqrt();
        match self {
            Self::Cylinder(r, h) => {
                let dx = (rho - r).max(0.0);
                let dy = (l[1].abs() - h).max(0.0);
                (dx * dx + dy * dy).sqrt()
            }
            Self::Cone(r, h) => {
                let p = (rho, l[1] - h / 2.0);
                let (a, b, c) = ((0.0, -h), (0.0, h), (r, -h));
                // inside the triangle (ρ ≥ 0, above the base, below the side)
                if p.1 >= -h && p.0 * 2.0 * h <= r * (h - p.1) {
                    return 0.0;
                }
                segment_distance(p, a, b)
                    .min(segment_distance(p, b, c))
                    .min(segment_distance(p, c, a))
            }
        }
    }
}

/// The distance from the segment `a`–`b` (world) to the solid turned by `q`, by
/// golden-section search on the convex distance along it.
fn core_distance(solid: Solid, q: [f64; 4], a: [f64; 3], b: [f64; 3]) -> f64 {
    let at = |s: f64| {
        let p = [
            a[0] + (b[0] - a[0]) * s,
            a[1] + (b[1] - a[1]) * s,
            a[2] + (b[2] - a[2]) * s,
        ];
        solid.distance_local(rotate_back(q, p))
    };
    if a == b {
        return at(0.0);
    }
    let ratio = 0.618_033_988_749_894_9;
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let x1 = hi - (hi - lo) * ratio;
        let x2 = lo + (hi - lo) * ratio;
        if at(x1) <= at(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    at(0.5 * (lo + hi)).min(at(0.0)).min(at(1.0))
}

/// The gaps (`/2⁻³²`) at the hit of small spheres and capsules cast from 4 away
/// toward the solid, aligned and turned.
fn cast_gaps(solid: Solid, seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for tilted in [false, true] {
        let (w, q) = solid_world(solid, tilted);
        for _ in 0..n {
            let r = rng.radius();
            let mut c = [
                2.0 * rng.u() - 1.0,
                2.0 * rng.u() - 1.0,
                2.0 * rng.u() - 1.0,
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-3);
            for x in &mut c {
                *x *= 4.0 / l;
            }
            let aim = [
                0.8 * rng.u() - 0.4,
                0.8 * rng.u() - 0.4,
                0.8 * rng.u() - 0.4,
            ];
            let mut d = [aim[0] - c[0], aim[1] - c[1], aim[2] - c[2]];
            let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for x in &mut d {
                *x /= dl;
            }
            let half = if rng.u() < 0.5 { 0.0 } else { 0.25 * rng.u() };
            let mut e = [
                2.0 * rng.u() - 1.0,
                2.0 * rng.u() - 1.0,
                2.0 * rng.u() - 1.0,
            ];
            let el = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt().max(1e-3);
            for x in &mut e {
                *x *= 2.0 * half / el;
            }
            let b = [c[0] + e[0], c[1] + e[1], c[2] + e[2]];
            let f = RayFilter::default();
            let h = if half == 0.0 {
                w.cast_sphere(p3(c), fx(r), p3(d), fx(10.0), &f)
            } else {
                w.cast_capsule(p3(c), p3(b), fx(r), p3(d), fx(10.0), &f)
            };
            let Some(h) = h else { continue };
            let t = h.t.to_f64();
            if t == 0.0 {
                continue;
            }
            let mv = |p: [f64; 3]| [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]];
            // oracle: the core is r from the solid at the contact
            out.push((core_distance(solid, q, mv(c), mv(b)) - r) / TOL);
        }
    }
    out
}

/// The world for `solid`, aligned or turned, and its rotation as `(w, x, y, z)`.
fn solid_world(solid: Solid, tilted: bool) -> (PhysicsWorld, [f64; 4]) {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert!(w.set_body_shape(i, &solid.shape()));
    if tilted {
        let k = (0.36f64 + 0.49 + 0.04).sqrt();
        w.bodies[i].rotation = QuatFix::from_axis_angle(p3([0.6 / k, 0.7 / k, -0.2 / k]), fx(0.9));
    }
    let q = w.bodies[i].rotation;
    (w, [q.w.to_f64(), q.x.to_f64(), q.y.to_f64(), q.z.to_f64()])
}

/// The gap (`/2⁻³²`) at the hit of one capsule cast (`a = b` for a sphere).
fn one_gap(solid: Solid, tilted: bool, a: [f64; 3], b: [f64; 3], d: [f64; 3], r: f64) -> f64 {
    let (w, q) = solid_world(solid, tilted);
    let h = w
        .cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(10.0), &RayFilter::default())
        .expect("hit");
    let t = h.t.to_f64();
    let mv = |p: [f64; 3]| [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]];
    (core_distance(solid, q, mv(a), mv(b)) - r) / TOL
}

/// Checks that no gap is below `lo` and at most `early` gaps are above `hi`.
#[track_caller]
fn assert_range(what: &str, gaps: &[f64], lo: f64, hi: f64, early: usize) {
    let min = gaps.iter().cloned().fold(f64::MAX, f64::min);
    let over: Vec<f64> = gaps.iter().cloned().filter(|&g| g > hi).collect();
    assert!(
        min >= lo && over.len() <= early,
        "{what}: deepest gap {min:+.3}·2⁻³² (allowed {lo:+.3}), {} gaps above {hi:+.3} \
         (allowed {early}): {over:?}",
        over.len()
    );
}

/// The deepest gap (`/2⁻³²`, rounded down to `0.01`) and the number of casts
/// stopping more than [`EARLY`] short, of the previous implementation on the
/// random casts of each test: the sides of a cylinder and a cone are not yet
/// within `2⁻³²` (and a cone's side can stop a cast early), and must not get
/// worse.
const CYLINDER_PREVIOUS: (f64, usize) = (-17.92, 0);
const CONE_PREVIOUS: (f64, usize) = (-24.43, 1);

/// A cast that stops more than `2⁻³²` short of the surface: the bound on the
/// early side (`2·2⁻³²`, the touching tolerance plus rounding).
const EARLY: f64 = 2.0;

#[test]
fn small_spheres_and_capsules_on_a_cylinder_are_no_worse_than_before() {
    let gaps = cast_gaps(Solid::Cylinder(0.9, 0.7), 0x0c71_0001, 400);
    // not vacuous: most casts aimed at the centre hit
    assert!(gaps.len() >= 600, "{} hits", gaps.len());
    assert_range(
        "cylinder",
        &gaps,
        CYLINDER_PREVIOUS.0,
        EARLY,
        CYLINDER_PREVIOUS.1,
    );
    // oracle: a capsule onto the side of the cylinder (the contact is on the side,
    // so it is reached at the gap 0, not 0.08 before it)
    let g = one_gap(
        Solid::Cylinder(0.9, 0.7),
        false,
        [-2.411086445858354, 1.0743258079875604, 3.0054094910494333],
        [-2.3191785665722886, 1.1898855244611029, 2.6439791865236733],
        [
            0.5870054376953389,
            -0.32836162989029366,
            -0.7400022000858465,
        ],
        3.099845350011054e-4,
    );
    assert_range("cylinder side", &[g], -1.0, EARLY, 0);
}

#[test]
fn small_spheres_and_capsules_on_a_cone_are_no_worse_than_before() {
    let gaps = cast_gaps(Solid::Cone(1.0, 0.8), 0x0c0e_0001, 400);
    assert!(gaps.len() >= 600, "{} hits", gaps.len());
    assert_range("cone", &gaps, CONE_PREVIOUS.0, EARLY, CONE_PREVIOUS.1);
    // oracle: a sphere onto the turned cone's side reaches it at about t = 3.589
    // (with the support direction scaled up, it stopped at 2.994)
    let c = [
        -3.8511054637745077,
        -0.017231054551754537,
        1.0810595717374534,
    ];
    let g = one_gap(
        Solid::Cone(1.0, 0.8),
        true,
        c,
        c,
        [0.972619985166704, 0.11309695260785699, -0.2030257219298494],
        1.0 / 4096.0,
    );
    // (the previous implementation reaches −1.97·2⁻³² into the side here)
    assert_range("cone side", &[g], -2.0, EARLY, 0);
}

/// A box of half width `2⁻¹²` beside a curved surface, `δ` off it along the
/// normal at the nearest point: it overlaps exactly when `δ < 0`.
#[test]
fn small_box_beside_a_curved_surface_overlaps_only_when_it_reaches_in() {
    let w_half = 1.0 / 4096.0;
    let shapes = [
        // the top of an ellipsoid of semi axes (1.2, 0.5, 0.8): y = 0.5 at x = z = 0
        // is its highest point, so a box above it is δ off it
        (
            Shape::Ellipsoid {
                radii: p3([1.2, 0.5, 0.8]),
            },
            [0.0, 0.5, 0.0],
            1usize,
        ),
        // the side of a cylinder of radius 0.9: x = 0.9 at z = 0 is its farthest
        // point along +X, so a box beside it is δ off it
        (
            Shape::Cylinder {
                radius: fx(0.9),
                half_height: fx(0.7),
            },
            [0.9, 0.0, 0.0],
            0usize,
        ),
    ];
    for (shape, top, axis) in shapes {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(i, &shape));
        for k in [20, 24, 28, 30] {
            let delta = 1.0 / ((1u64 << k) as f64);
            for (sign, want) in [(1.0, false), (-1.0, true)] {
                let mut lo = [top[0] - w_half, top[1] - w_half, top[2] - w_half];
                let mut hi = [top[0] + w_half, top[1] + w_half, top[2] + w_half];
                // the box's near face δ off the surface along the axis
                lo[axis] = top[axis] + sign * delta;
                hi[axis] = lo[axis] + 2.0 * w_half;
                let got = w.overlap_aabb(&AABB::new(p3(lo), p3(hi)), &RayFilter::default());
                assert_eq!(
                    got == vec![RayTarget::Body(i)],
                    want,
                    "{shape:?} δ = {}2^-{k}: {got:?}",
                    if sign > 0.0 { "+" } else { "−" }
                );
            }
        }
    }
}
