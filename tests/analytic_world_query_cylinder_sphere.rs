//! Oracles for spheres cast onto a cylinder along paths that graze its caps and
//! rims: each hit is within `[−1, +2]·2⁻³²` of the surface, its point is on the
//! surface and its normal is the surface normal there, and each path that goes
//! more than `2⁻³²` in hits.
//!
//! # Expected values
//!
//! The reference is computed here in `f64`, independently of the code under
//! test: the distance of a point to a cylinder is the distance in its meridian
//! half-plane `(ρ, y)` to a rectangle; the gap along a straight path (the
//! distance less the radius) is convex in time, so its minimum and first zero
//! are found by ternary search and bisection. The surface normal at a hit is the
//! direction from the nearest point of the cylinder to the sphere's centre. The
//! reference sees the `Fix128` inputs converted to `f64` (off by `2⁻⁵³` of their
//! size).
//!
//! The single casts below each failed before: a sphere sliding onto a cap at
//! `10⁻⁶` rad went `6873·2⁻³²` into it, one sliding over a cap at a height just
//! inside the rounded rim was reported on the rim's vertical line (point `r` off
//! the surface, normal `90°` off), and one grazing a rim went `4.4·10⁵·2⁻³²` in
//! without a hit.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// `2⁻³²`.
const TOL: f64 = 1.0 / 4_294_967_296.0;

/// The bounds of a hit's gap, in `2⁻³²`.
const DEEP: f64 = -1.0;
const EARLY: f64 = 2.0;

/// The largest distance of a hit's point from the surface, and the largest
/// angle of its normal from the surface normal (rad).
const POINT_OFF: f64 = 1e-9;
const NORMAL_OFF: f64 = 1e-6;

const MAX_T: f64 = 10.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2]))
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn add(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
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

    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.u()
    }
}

/// `v` rotated by the unit quaternion `q = (w, x, y, z)`, or by its inverse.
fn rotate(q: [f64; 4], v: [f64; 3], inverse: bool) -> [f64; 3] {
    let s = if inverse { -1.0 } else { 1.0 };
    let (w, x, y, z) = (q[0], s * q[1], s * q[2], s * q[3]);
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

/// A cylinder of radius `radius` and half height `hh` on a static body at
/// `position` turned about `axis` by `angle`.
struct Scene {
    radius: f64,
    hh: f64,
    world: PhysicsWorld,
    position: [f64; 3],
    q: [f64; 4],
}

impl Scene {
    fn new(radius: f64, hh: f64, position: [f64; 3], axis: [f64; 3], angle: f64) -> Self {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        let i = world.add_body(RigidBody::new_static(p3(position)));
        let shape = Shape::Cylinder {
            radius: fx(radius),
            half_height: fx(hh),
        };
        assert!(world.set_body_shape(i, &shape));
        world.bodies[i].rotation = QuatFix::from_axis_angle(p3(unit(axis)), fx(angle));
        let r = world.bodies[i].rotation;
        Self {
            radius,
            hh,
            world,
            position: f3(p3(position)),
            q: [r.w.to_f64(), r.x.to_f64(), r.y.to_f64(), r.z.to_f64()],
        }
    }

    fn to_local(&self, p: [f64; 3]) -> [f64; 3] {
        let o = self.position;
        rotate(self.q, [p[0] - o[0], p[1] - o[1], p[2] - o[2]], true)
    }

    /// The signed distance of a point (negative inside).
    fn signed_distance(&self, p: [f64; 3]) -> f64 {
        let l = self.to_local(p);
        let rho = (l[0] * l[0] + l[2] * l[2]).sqrt();
        let (dx, dy) = (rho - self.radius, l[1].abs() - self.hh);
        let out = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
        out + dx.max(dy).min(0.0)
    }

    /// The outward unit normal at the point of the surface nearest `p`, for a
    /// `p` outside: the direction from that point to `p`.
    fn normal_toward(&self, p: [f64; 3]) -> [f64; 3] {
        let l = self.to_local(p);
        let rho = (l[0] * l[0] + l[2] * l[2]).sqrt();
        let k = rho.min(self.radius) / rho;
        let near = [l[0] * k, l[1].clamp(-self.hh, self.hh), l[2] * k];
        let n = unit([l[0] - near[0], l[1] - near[1], l[2] - near[2]]);
        rotate(self.q, n, false)
    }

    fn gap(&self, c: &Cast, t: f64) -> f64 {
        self.signed_distance(add(c.o, c.d, t)) - c.r
    }

    /// The least gap on `[0, MAX_T]` and the first time the gap reaches 0.
    fn reference(&self, c: &Cast) -> (f64, Option<f64>) {
        let (mut lo, mut hi) = (0.0f64, MAX_T);
        for _ in 0..120 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if self.gap(c, m1) <= self.gap(c, m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let tm = 0.5 * (lo + hi);
        let least = self.gap(c, tm);
        if least > 0.0 {
            return (least, None);
        }
        let (mut lo, mut hi) = (0.0f64, tm);
        for _ in 0..80 {
            let m = 0.5 * (lo + hi);
            if self.gap(c, m) > 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        (least, Some(hi))
    }
}

/// A sphere of radius `r` from `o` along `d` (normalized by the cast).
#[derive(Debug)]
struct Cast {
    fixed: [Vec3Fix; 2],
    o: [f64; 3],
    d: [f64; 3],
    r: f64,
}

impl Cast {
    fn new(o: [f64; 3], d: [f64; 3], r: f64) -> Self {
        let fixed = [p3(o), p3(d)];
        Self {
            o: f3(fixed[0]),
            d: unit(f3(fixed[1])),
            fixed,
            r,
        }
    }
}

/// What a cast did against the reference.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// A hit at gap `g` (`/2⁻³²`), its point `p` from the surface and its normal
    /// `n` rad from the surface normal.
    Hit { g: f64, p: f64, n: f64 },
    /// A hit at `t = 0`.
    Start,
    /// No hit, and the path does not go `2⁻³²` in.
    Clear,
    /// A hit where the path never reaches the surface.
    Extra,
    /// No hit, but the path goes `g` (`/2⁻³²`) in.
    Missed(f64),
}

fn outcome(scene: &Scene, c: &Cast) -> Outcome {
    let (least, contact) = scene.reference(c);
    let h = scene.world.cast_sphere(
        c.fixed[0],
        fx(c.r),
        c.fixed[1],
        fx(MAX_T),
        &RayFilter::default(),
    );
    match h {
        Some(_) if contact.is_none() => Outcome::Extra,
        Some(h) if h.t.is_zero() => Outcome::Start,
        Some(h) => {
            let t = h.t.to_f64();
            let centre = add(c.o, c.d, t);
            let want = scene.normal_toward(centre);
            let got = unit(f3(h.normal));
            let dot = got[0] * want[0] + got[1] * want[1] + got[2] * want[2];
            Outcome::Hit {
                g: scene.gap(c, t) / TOL,
                p: scene.signed_distance(f3(h.point)).abs(),
                n: dot.clamp(-1.0, 1.0).acos(),
            }
        }
        None if least / TOL < DEEP => Outcome::Missed(least / TOL),
        None => Outcome::Clear,
    }
}

fn good(o: &Outcome) -> bool {
    match *o {
        Outcome::Hit { g, p, n } => {
            (DEEP..=EARLY).contains(&g) && p <= POINT_OFF && n <= NORMAL_OFF
        }
        Outcome::Start | Outcome::Clear => true,
        Outcome::Extra | Outcome::Missed(_) => false,
    }
}

#[track_caller]
fn assert_at_surface(o: Outcome) {
    assert!(
        matches!(o, Outcome::Hit { .. }) && good(&o),
        "{o:?}, expected a hit within [{DEEP}, {EARLY}]·2⁻³², point within {POINT_OFF}, \
         normal within {NORMAL_OFF} rad"
    );
}

/// A sphere sliding onto a turned cylinder's cap at `10⁻⁶` rad: it went
/// `6873·2⁻³²` into the cap (its path met the rounded rim first, and the ray
/// through that rim was not found).
#[test]
fn a_sphere_sliding_onto_a_cap_stops_at_it() {
    let scene = Scene::new(
        1.322639427431568,
        0.8833547745225587,
        [0.8757124063295123, -1.4809189221923589, 1.107182173174806],
        [0.4895207024201227, -0.7647938836253161, 0.41887921585021104],
        2.643228861633361,
    );
    let c = Cast::new(
        [1.3625507867945998, -0.19350011872757023, 3.254362783446595],
        [0.3003632220252257, -0.2868026233254568, -0.9096846652046224],
        0.0625,
    );
    assert_at_surface(outcome(&scene, &c));
}

/// A sphere sliding over a turned cylinder's cap at a height just within its
/// rounded rim: the time was right, but the point was on the rim's vertical
/// line `r` from the sphere's centre, the normal horizontal.
#[test]
fn a_sphere_sliding_over_a_cap_gets_the_cap_normal() {
    let scene = Scene::new(
        0.5181143623331081,
        0.8667111466975257,
        [1.075814958846422, 1.2710040447982465, -1.8224955914092789],
        [
            -0.3765408733570439,
            -0.09379973336126568,
            -0.9216390729088744,
        ],
        2.9000697912515534,
    );
    let o = [1.0957662486143818, 2.093937713336345, -3.69258705395805];
    let d = [
        -0.37951040787902457,
        -0.037562364287623495,
        0.9244246421967546,
    ];
    // that path goes only 0.999·2⁻³² in (a hit or none are both right); moved
    // 10⁻⁶ toward the cap (along the body's Y), it goes in
    let up = rotate(scene.q, [0.0, 1e-6, 0.0], false);
    let c = Cast::new(add(o, up, 1.0), d, 0.0625);
    assert_at_surface(outcome(&scene, &c));
}

/// A sphere grazing a turned cylinder's rim, `4.4·10⁵·2⁻³²` deep at its
/// deepest: it was not a hit.
#[test]
fn a_sphere_grazing_a_rim_hits_it() {
    let scene = Scene::new(
        0.7467231900400293,
        1.1617931724786104,
        [1.618755172060446, 0.4206591868523901, -0.8537734834953881],
        [0.46596195393885864, 0.298953402528241, -0.8327702688002319],
        2.7461600719370836,
    );
    let c = Cast::new(
        [-1.9616743186597887, 2.3771735302116213, -2.2270289525486078],
        [0.9640397648827275, -0.08290410194695141, 0.252496022950254],
        0.5,
    );
    assert_at_surface(outcome(&scene, &c));
}

/// Spheres on paths that pass a cap or a rim at a small angle and a small
/// depth, in random turned cylinders.
#[test]
fn spheres_grazing_caps_and_rims_stop_at_the_surface() {
    let mut rng = Rng(0x0c71_5a5e);
    let mut bad = Vec::new();
    let (mut casts, mut hits) = (0usize, 0usize);
    for _ in 0..1200 {
        let (radius, hh) = (rng.r(0.5, 1.4), rng.r(0.5, 1.2));
        let axis = [rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)];
        let scene = Scene::new(radius, hh, [0.0; 3], axis, rng.r(0.0, 3.0));
        let r = [1.0 / 4096.0, 1.0 / 256.0, 1.0 / 16.0, 0.5][(rng.u() * 4.0) as usize];
        // the path's nearest point to the cap's plane, in the body frame: above
        // the cap or beside the rim, a little inside the rounded surface
        let phi = rng.r(0.0, std::f64::consts::TAU);
        let rho = if rng.u() < 0.5 {
            rng.r(0.0, radius)
        } else {
            radius + r * rng.r(0.0, 1.0)
        };
        let depth = [0.0, 1e-9, 1e-6, 1e-3][(rng.u() * 4.0) as usize];
        let sign = if rng.u() < 0.5 { 1.0 } else { -1.0 };
        let local_y = sign * (hh + r * rng.r(0.0, 1.0) - depth);
        let near = [rho * phi.cos(), local_y, rho * phi.sin()];
        // along the cap, turned toward it by a small angle
        let slope = [0.0, 1e-9, 1e-6, 1e-3, 2e-2][(rng.u() * 5.0) as usize];
        let psi = rng.r(0.0, std::f64::consts::TAU);
        let dl = unit([psi.cos(), -sign * slope, psi.sin()]);
        let o_l = add(near, dl, -3.0);
        let o = add(rotate(scene.q, o_l, false), scene.position, 1.0);
        let d = rotate(scene.q, dl, false);
        let c = Cast::new(o, d, r);
        casts += 1;
        let out = outcome(&scene, &c);
        if matches!(out, Outcome::Hit { .. }) {
            hits += 1;
        }
        if !good(&out) {
            bad.push((out, radius, hh, c));
        }
    }
    // not vacuous: most of these paths go into the rounded cylinder
    assert!(hits * 2 >= casts, "{hits} hits of {casts} casts");
    assert!(bad.is_empty(), "{} of {casts} casts: {bad:#?}", bad.len());
}
