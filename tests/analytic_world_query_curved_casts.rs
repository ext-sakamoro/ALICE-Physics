//! Oracles for small spheres and capsules cast onto a cone and a cylinder,
//! aligned and turned: each hit is within `[−1, +2]·2⁻³²` of the surface, each
//! hit has a contact, and each cast whose path goes more than `2⁻³²` into the
//! solid hits.
//!
//! # Expected values
//!
//! The reference is computed here in `f64`, independently of the code under
//! test: a cone and a cylinder are solids of revolution, so the distance of a
//! point is the distance in its meridian half-plane `(ρ, y)` to a triangle or a
//! rectangle; a segment's distance is the minimum of that convex function along
//! it, and the gap along a straight path (the distance minus the radius) is
//! convex in time, so its minimum and first zero are found by ternary search
//! (`(2/3)¹⁰⁰` of the segment, `(2/3)¹²⁰` of the path) and bisection (`2⁻⁸⁰`
//! of the path). `f64` is exact to about `2⁻⁵²` here, far below the `2⁻³²`
//! bounds. The reference sees the `Fix128` inputs converted to `f64` (off by
//! `2⁻⁵³` of their size, `2⁻⁵⁰` along a path of length 10).
//!
//! The single casts below each stopped outside the bounds before: a sphere
//! that went `11·2⁻³²` into a cone's side, one stopped `0.03` before a turned
//! cone's side, a capsule stopped `0.19` before a cylinder, one stopped
//! `8.7·2⁻³²` before a cylinder's rim, and a sphere that went `38152·2⁻³²`
//! into a turned cone near its tangent was not a hit.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// `2⁻³²`.
const TOL: f64 = 1.0 / 4_294_967_296.0;

/// The bounds of a hit's gap, in `2⁻³²`: touching tolerance plus rounding.
const DEEP: f64 = -1.0;
const EARLY: f64 = 2.0;

/// How far the casts may travel.
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

fn segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let u = (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
    let (dx, dy) = (p.0 - a.0 - u * ex, p.1 - a.1 - u * ey);
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// Its geometric centre is `hh/2` above the body position (the centre of
    /// mass): apex at `1.5·hh`, base at `−0.5·hh`.
    Cone,
    /// About the body position.
    Cylinder,
}

/// A cone or a cylinder of radius `r` and half height `hh`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Solid {
    kind: Kind,
    r: f64,
    hh: f64,
}

/// The cone and the cylinder of the random casts.
const CONE: Solid = Solid {
    kind: Kind::Cone,
    r: 1.0,
    hh: 0.8,
};
const CYLINDER: Solid = Solid {
    kind: Kind::Cylinder,
    r: 0.9,
    hh: 0.7,
};

impl Solid {
    fn shape(self) -> Shape {
        let (radius, half_height) = (fx(self.r), fx(self.hh));
        match self.kind {
            Kind::Cone => Shape::Cone {
                radius,
                half_height,
            },
            Kind::Cylinder => Shape::Cylinder {
                radius,
                half_height,
            },
        }
    }

    /// The signed distance of a point in the body frame (negative inside).
    fn signed_distance(self, l: [f64; 3]) -> f64 {
        let (r, h) = (self.r, self.hh);
        let rho = (l[0] * l[0] + l[2] * l[2]).sqrt();
        match self.kind {
            Kind::Cone => {
                let p = (rho, l[1] - 0.5 * h);
                let (a, b, c) = ((0.0, -h), (0.0, h), (r, -h));
                let to_side = segment_distance(p, b, c);
                let to_base = segment_distance(p, c, a);
                // inside the triangle: above the base, below the side
                if p.1 >= -h && p.0 * 2.0 * h <= r * (h - p.1) {
                    -to_side.min(to_base)
                } else {
                    segment_distance(p, a, b).min(to_side).min(to_base)
                }
            }
            Kind::Cylinder => {
                let (dx, dy) = (rho - r, l[1].abs() - h);
                let out = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
                out + dx.max(dy).min(0.0)
            }
        }
    }
}

/// A cone or a cylinder in a world, its body at `position` turned by `q`.
struct Scene {
    solid: Solid,
    world: PhysicsWorld,
    position: [f64; 3],
    q: [f64; 4],
}

impl Scene {
    /// Aligned (a turn by 0 about `Y`, as a pose would be built: the identity
    /// to within rounding) or turned, at the origin.
    fn new(solid: Solid, tilted: bool) -> Self {
        if tilted {
            Self::posed(solid, [0.0; 3], [0.6, 0.7, -0.2], 0.9)
        } else {
            Self::posed(solid, [0.0; 3], [0.0, 1.0, 0.0], 0.0)
        }
    }

    fn posed(solid: Solid, position: [f64; 3], axis: [f64; 3], angle: f64) -> Self {
        let turn = QuatFix::from_axis_angle(p3(unit(axis)), fx(angle));
        Self::turned(solid, position, turn)
    }

    /// The body turned by the quaternion `turn` as given.
    fn turned(solid: Solid, position: [f64; 3], turn: QuatFix) -> Self {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        let i = world.add_body(RigidBody::new_static(p3(position)));
        assert!(world.set_body_shape(i, &solid.shape()));
        world.bodies[i].rotation = turn;
        let r = world.bodies[i].rotation;
        let q = [r.w.to_f64(), r.x.to_f64(), r.y.to_f64(), r.z.to_f64()];
        Self {
            solid,
            world,
            position: f3(p3(position)),
            q,
        }
    }

    /// The signed distance of the segment `a`–`b` (world) to the solid.
    fn segment_distance(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        let at = |s: f64| {
            let p = [
                a[0] + s * (b[0] - a[0]),
                a[1] + s * (b[1] - a[1]),
                a[2] + s * (b[2] - a[2]),
            ];
            let o = self.position;
            let rel = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
            self.solid.signed_distance(rotate(self.q, rel, true))
        };
        if a == b {
            return at(0.0);
        }
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..100 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if at(m1) <= at(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        at(0.5 * (lo + hi)).min(at(0.0)).min(at(1.0))
    }

    /// The gap of the core `a`–`b` grown by `r`, moved by `t·d`.
    fn gap(&self, c: &Cast, t: f64) -> f64 {
        let mv = |p: [f64; 3]| [p[0] + t * c.d[0], p[1] + t * c.d[1], p[2] + t * c.d[2]];
        self.segment_distance(mv(c.a), mv(c.b)) - c.r
    }

    /// The least gap on `[0, max_t]` and the first time the gap reaches 0, if
    /// it does (the gap is convex in time).
    fn reference(&self, c: &Cast) -> (f64, Option<f64>) {
        let (mut lo, mut hi) = (0.0f64, c.max_t);
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

    /// The time of the hit of a cast.
    fn cast(&self, c: &Cast) -> Option<f64> {
        let f = RayFilter::default();
        let (r, max_t) = (fx(c.r), fx(c.max_t));
        let h = if c.fixed[0] == c.fixed[1] {
            self.world.cast_sphere(c.fixed[0], r, c.fixed[2], max_t, &f)
        } else {
            self.world
                .cast_capsule(c.fixed[0], c.fixed[1], r, c.fixed[2], max_t, &f)
        };
        h.map(|h| h.t.to_f64())
    }
}

/// A cast of the core `a`–`b` (a sphere when `a = b`) grown by `r` along the
/// `d` (normalized by the cast): the inputs given to the code under test (`fixed`: `a`, `b`, `d`)
/// and the same in `f64` for the reference (`r` is a power of two, exact in
/// both).
#[derive(Debug)]
struct Cast {
    fixed: [Vec3Fix; 3],
    a: [f64; 3],
    b: [f64; 3],
    d: [f64; 3],
    r: f64,
    /// How far the cast goes (`MAX_T` unless set).
    max_t: f64,
}

impl Cast {
    fn new(a: [f64; 3], b: [f64; 3], d: [f64; 3], r: f64) -> Self {
        let fixed = [p3(a), p3(b), p3(d)];
        Self {
            a: f3(fixed[0]),
            b: f3(fixed[1]),
            // the cast normalizes its direction
            d: unit(f3(fixed[2])),
            fixed,
            r,
            max_t: MAX_T,
        }
    }

    fn up_to(self, max_t: f64) -> Self {
        Self { max_t, ..self }
    }
}

/// What a cast did against the reference.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// A hit at gap `g` (`/2⁻³²`); a hit at `t = 0` (the core starting within
    /// the tolerance) is `Start`.
    Hit(f64),
    Start,
    /// No hit, and the path does not go `2⁻³²` in.
    Clear,
    /// A hit where the path never reaches the surface.
    Extra,
    /// No hit, but the path goes `g` (`/2⁻³²`, below −1) in.
    Missed(f64),
}

fn outcome(scene: &Scene, c: &Cast) -> Outcome {
    let (least, contact) = scene.reference(c);
    match scene.cast(c) {
        Some(_) if contact.is_none() => Outcome::Extra,
        Some(0.0) => Outcome::Start,
        Some(t) => Outcome::Hit(scene.gap(c, t) / TOL),
        None if least / TOL < DEEP => Outcome::Missed(least / TOL),
        None => Outcome::Clear,
    }
}

/// A cast from 4 away toward a point near the solid's side, rim, apex or cap.
fn random_cast(rng: &mut Rng, scene: &Scene, r: f64, capsule: bool) -> Cast {
    let az = rng.r(0.0, std::f64::consts::TAU);
    let el = rng.r(-1.2, 1.2);
    let c = [
        4.0 * az.cos() * el.cos(),
        4.0 * el.sin(),
        4.0 * az.sin() * el.cos(),
    ];
    let pick = rng.u();
    let ph = rng.r(0.0, std::f64::consts::TAU);
    let local = match scene.solid.kind {
        Kind::Cone if pick < 0.6 => {
            let s = rng.r(0.0, 1.0);
            [s * ph.cos(), 1.2 - 1.6 * s, s * ph.sin()]
        }
        Kind::Cone if pick < 0.8 => [ph.cos(), -0.4, ph.sin()],
        Kind::Cone => [0.0, 1.2, 0.0],
        Kind::Cylinder if pick < 0.6 => [0.9 * ph.cos(), rng.r(-0.7, 0.7), 0.9 * ph.sin()],
        Kind::Cylinder => {
            let y = if rng.u() < 0.5 { 0.7 } else { -0.7 };
            [0.9 * ph.cos(), y, 0.9 * ph.sin()]
        }
    };
    let jitter = [rng.r(-0.03, 0.03), rng.r(-0.03, 0.03), rng.r(-0.03, 0.03)];
    let aim = rotate(
        scene.q,
        [
            local[0] + jitter[0],
            local[1] + jitter[1],
            local[2] + jitter[2],
        ],
        false,
    );
    let d = unit([aim[0] - c[0], aim[1] - c[1], aim[2] - c[2]]);
    if !capsule {
        return Cast::new(c, c, d, r);
    }
    let e = unit([rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)]);
    let hl = rng.r(0.05, 0.3);
    let a = [c[0] - hl * e[0], c[1] - hl * e[1], c[2] - hl * e[2]];
    let b = [c[0] + hl * e[0], c[1] + hl * e[1], c[2] + hl * e[2]];
    Cast::new(a, b, d, r)
}

/// Casts per solid, orientation, core kind and radius.
const CASTS: usize = 24;

#[test]
fn small_spheres_and_capsules_stop_at_a_cone_and_a_cylinder() {
    let mut rng = Rng(0x5340_c0e5);
    let mut bad = Vec::new();
    let (mut hits, mut casts) = (0usize, 0usize);
    for solid in [CONE, CYLINDER] {
        for tilted in [true, false] {
            let scene = Scene::new(solid, tilted);
            for capsule in [false, true] {
                for k in [16, 14, 12, 10, 8] {
                    let r = 1.0 / ((1u64 << k) as f64);
                    for _ in 0..CASTS {
                        let c = random_cast(&mut rng, &scene, r, capsule);
                        casts += 1;
                        let o = outcome(&scene, &c);
                        match o {
                            Outcome::Hit(g) if (DEEP..=EARLY).contains(&g) => hits += 1,
                            Outcome::Clear | Outcome::Start => {}
                            _ => bad.push((solid, tilted, k, o, c)),
                        }
                    }
                }
            }
        }
    }
    // not vacuous: most casts are aimed at the surface
    assert!(hits * 4 >= casts * 3, "{hits} hits of {casts} casts");
    assert!(
        bad.is_empty(),
        "{} of {casts} casts outside [{DEEP}, {EARLY}]·2⁻³², a hit without a contact or a \
         path that goes in without a hit: {bad:#?}",
        bad.len()
    );
}

/// The outcome of one cast.
fn one(solid: Solid, tilted: bool, a: [f64; 3], b: [f64; 3], d: [f64; 3], r: f64) -> Outcome {
    outcome(&Scene::new(solid, tilted), &Cast::new(a, b, d, r))
}

#[track_caller]
fn assert_at_surface(o: Outcome) {
    assert!(
        matches!(o, Outcome::Hit(g) if (DEEP..=EARLY).contains(&g)),
        "{o:?}, expected a hit within [{DEEP}, {EARLY}]·2⁻³²"
    );
}

/// Spheres onto a cone's side: the nearest point of GJK's long thin triangles
/// (support points on the apex and the rim) was off by up to `2⁻²⁶`; they went
/// `1.9`, `11.2` and `2.0·2⁻³²` in.
#[test]
fn spheres_onto_a_cone_side_stop_at_it() {
    let cases: [(bool, [f64; 3], [f64; 3], f64); 3] = [
        (
            true,
            [0.477799047306774, -3.9052878250067056, 3.0112373368542884],
            [
                -0.05235262586114163,
                0.7794760898125248,
                -0.6242405209340548,
            ],
            2.093220555230235e-05,
        ),
        (
            false,
            [-0.8017093517167726, 3.41359356257783, 1.270047874110987],
            [
                0.004513553686136947,
                -0.966737762989219,
                -0.25572978990278544,
            ],
            3.281904777169284e-05,
        ),
        (
            true,
            [
                -3.8511054637745077,
                -0.017231054551754537,
                1.0810595717374534,
            ],
            [0.972619985166704, 0.11309695260785699, -0.2030257219298494],
            1.0 / 4096.0,
        ),
    ];
    for (tilted, c, d, r) in cases {
        assert_at_surface(one(CONE, tilted, c, c, d, r));
    }
}

/// A sphere onto a turned cone near its apex: GJK's tetrahedron of the apex
/// and three nearly coplanar points counted the origin inside by the sign of
/// a rounding-sized volume, and the cast stopped at `t = 3.51783` instead of
/// `3.55063` (`0.03` before the surface).
#[test]
fn a_flat_tetrahedron_does_not_stop_a_sphere_before_a_cone() {
    let c = [0.6798685716009641, 1.4536636933206846, 3.9417989199538406];
    let d = [
        -0.14042024359157537,
        -0.3040988874759272,
        -0.9422345895930514,
    ];
    assert_at_surface(one(CONE, true, c, c, d, 1.0 / 1024.0));
}

/// A capsule onto an aligned cylinder: it stopped at `t = 3.00683` instead of
/// `3.35170` (`0.19` before the surface).
#[test]
fn a_capsule_reaches_a_cylinder() {
    assert_at_surface(one(
        CYLINDER,
        false,
        [0.9722042145558407, 0.4872755148012834, -3.6061370143046063],
        [1.1179553214974085, 0.43868468227385105, -4.060272238225437],
        [
            -0.04592945850301522,
            -0.26157394675276135,
            0.964090014066012,
        ],
        1.0 / 65536.0,
    ));
}

/// A capsule onto an aligned cylinder's bottom rim: the contact normal is
/// nearly along the axis, so the support direction's horizontal part is about
/// `2⁻²⁰` long, and the rim points normalized from it were `2⁻²⁴` of the radius
/// outside the cylinder; the cast stopped `8.7·2⁻³²` short.
#[test]
fn a_capsule_reaches_a_cylinder_rim() {
    assert_at_surface(one(
        CYLINDER,
        false,
        [-2.0745705723964463, -3.0390753557258097, -1.58484393111156],
        [-2.227742123142524, -3.0606503022738942, -1.2932548105830826],
        [0.7201672512993514, 0.6628242737539644, 0.20499539574943265],
        1.0 / 65536.0,
    ));
}

/// A sphere that grazes a turned cone at a slope of `0.009` and then goes
/// `38152·2⁻³²` in: the first step of the near-tangent search dropped the gap
/// by `0.01·2⁻³²`, below the rounding of the distance, which made it seem to
/// grow; it was not a hit.
#[test]
fn a_grazing_sphere_that_goes_into_a_cone_hits_it() {
    let c = [3.49676953785352, -1.9311419641957024, -0.20807093324643489];
    let d = [
        -0.8417749934677569,
        0.49646776600120957,
        -0.21197787076514807,
    ];
    assert_at_surface(one(CONE, true, c, c, d, 1.0 / 65536.0));
}

/// A sphere moving at `9·10⁻¹⁰` rad to a turned cone's base, `3·2⁻³²` from it
/// at `t = 1.92`, whose path then goes `1.7·2⁻³²` in (first contact at
/// `t = 2.662`): the slope there came out `+3·10⁻⁹` (away from the base) for
/// `−1.5·10⁻⁹`, and the cast was not a hit.
#[test]
fn a_sphere_nearly_parallel_to_a_cone_base_that_goes_in_hits_it() {
    let cone = Solid {
        kind: Kind::Cone,
        r: 0.6428203322684567,
        hh: 0.7475816177966408,
    };
    let scene = Scene::posed(
        cone,
        [
            -1.6526587531452606,
            -1.2186799371102097,
            0.18938048544987396,
        ],
        [
            -0.5803070933934578,
            0.7790505106677301,
            -0.23732673508311564,
        ],
        1.8984897774453202,
    );
    let c = [
        -0.8965821042720563,
        -3.3486672559565704,
        -0.9839086836373099,
    ];
    let cast = Cast::new(
        c,
        c,
        [-0.1482674077997217, 0.8136896687128683, 0.5620728589901773],
        1.0 / 65536.0,
    );
    // not vacuous: the path goes more than 2⁻³² in
    let (least, _) = scene.reference(&cast);
    assert!(least / TOL < DEEP, "the path goes only {} in", least / TOL);
    assert_at_surface(outcome(&scene, &cast));
}

/// A capsule sliding onto a turned cylinder's cap at `9·10⁻¹⁰` rad, whose path
/// goes `7.5·2⁻³²` in: GJK's face near the contact is a thin triangle of rim
/// points, and taking its point by projection on the face's plane (whose tilt
/// is the rounding of its short edge) put the gap below the distance, so the
/// path seemed not to go in.
#[test]
fn a_capsule_sliding_onto_a_cylinder_cap_hits_it() {
    let cylinder = Solid {
        kind: Kind::Cylinder,
        r: 0.9012093363780878,
        hh: 0.5179403340707722,
    };
    let scene = Scene::posed(
        cylinder,
        [
            -0.4023466290173019,
            -0.46594330930383876,
            -1.2957395342946256,
        ],
        [
            -0.6714485362126652,
            -0.09612558018852724,
            -0.7347902667097514,
        ],
        0.5010374581070209,
    );
    let cast = Cast::new(
        [1.8608221708009296, -2.0500185708660865, -1.4757067685941365],
        [1.3062351220451092, -1.5734718199873896, -0.7798910803267063],
        [
            -0.9249401521738037,
            0.2939600468880599,
            -0.24098382877309632,
        ],
        1.0 / 65536.0,
    );
    // not vacuous: the path goes more than 2⁻³² in
    let (least, _) = scene.reference(&cast);
    assert!(least / TOL < DEEP, "the path goes only {} in", least / TOL);
    assert_at_surface(outcome(&scene, &cast));
}

/// One cast onto a posed solid, from `a`–`b` along `d` with radius `r`.
#[allow(clippy::too_many_arguments)]
fn posed_cast(
    solid: Solid,
    position: [f64; 3],
    axis: [f64; 3],
    angle: f64,
    a: [f64; 3],
    b: [f64; 3],
    d: [f64; 3],
    r: f64,
) -> (Scene, Cast) {
    (
        Scene::posed(solid, position, axis, angle),
        Cast::new(a, b, d, r).up_to(4.0),
    )
}

/// Segments of radius 0 that enter a cylinder through its side just below a
/// cap, `1.125·2⁻³²` and `1.25·2⁻³²` deep (the second turned): GJK stopped on
/// a face `2` to `25·2⁻³²` from the origin with the segment already in, and
/// the step from there went `1.1` and `1.25·2⁻³²` past the contact.
#[test]
fn segments_entering_a_cylinder_below_its_cap_stop_at_its_side() {
    let cylinder = Solid {
        kind: Kind::Cylinder,
        r: 0.9,
        hh: 0.7,
    };
    let casts = [
        (
            Scene::turned(cylinder, [0.25, -0.5, 0.75], QuatFix::IDENTITY),
            Cast::new(
                [-1.695761194159812, 0.19999999973806548, 0.2143570449433858],
                [-1.695761194159812, 0.4999999997380655, 0.2143570449433858],
                [0.9909759596810406, 0.0, 0.1340397229713666],
                0.0,
            )
            .up_to(4.0),
        ),
        posed_cast(
            cylinder,
            [0.25, -0.5, 0.75],
            [0.6, 0.7, -0.2],
            0.9,
            [
                -0.7916938670894744,
                1.1912886520345445,
                -0.03569496775777914,
            ],
            [-0.6883045505593128, 1.4402697713519372, 0.09590689944358055],
            [0.7223971058724328, -0.5328868033551468, 0.4406518764706059],
            0.0,
        ),
    ];
    for (scene, cast) in &casts {
        // not vacuous: the path goes more than 2⁻³² in
        let (least, _) = scene.reference(cast);
        assert!(least / TOL < DEEP, "the path goes only {} in", least / TOL);
        assert_at_surface(outcome(scene, cast));
    }
}

/// Casts at the edge of what the time of impact can tell apart, each of which
/// the cast got wrong while one part of it was missing (the part named): the
/// path goes more than `2⁻³²` in, and the hit must be within `[−1, +2]·2⁻³²`.
#[test]
fn casts_at_the_edge_of_the_tolerance_stop_at_the_surface() {
    // (what it pins, cone, radius, half height, turned, position, a, b, r, d, max_t)
    type Case = (
        &'static str,
        bool,
        f64,
        f64,
        bool,
        [f64; 3],
        [f64; 3],
        [f64; 3],
        f64,
        [f64; 3],
        f64,
    );
    let p = [0.25, -0.5, 0.75];
    let cases: [Case; 7] = [
        (
            "the barycentric form only where it is precise (segment over a rim)",
            false,
            0.9,
            0.7,
            false,
            p,
            [-1.6640986290561979, 0.19999999975304572, 1.8206196513237867],
            [-1.834506720909574, 0.41213203410901, 1.6942824745669378],
            0.0,
            [0.5955591626715423, 0.0, -0.8033114487905494],
            4.0,
        ),
        (
            "the lower bound less the support slack, its step (segment over a rim)",
            false,
            0.9,
            0.7,
            false,
            p,
            [1.8673886946205231, 0.19999999967072768, -0.7312338808982968],
            [2.0622378155658776, 0.41213203402669196, -0.6473660082245325],
            0.0,
            [-0.3953569432753914, 0.0, 0.9185275648579845],
            4.0,
        ),
        (
            "the step by the lower bound and the near-tangent search (turned rim)",
            false,
            0.9,
            0.7,
            true,
            p,
            [-0.21852465418609512, -0.6507157640535539, 2.998948511156243],
            [
                -0.34405370476815655,
                -0.41720885309588485,
                3.139365675331331,
            ],
            0.0,
            [-0.06648810702703647, 0.487707135982694, -0.8704717578046512],
            4.0,
        ),
        (
            "the exact distance along a near-tangent path (cone rim)",
            true,
            1.0,
            0.8,
            false,
            p,
            [-1.7015908184354425, -0.9027621356993638, 1.8439950540065662],
            [-1.7015908184354425, -0.9027621356993638, 1.8439950540065662],
            1.0 / 256.0,
            [0.5606127132316385, 0.0, -0.8280781278134695],
            4.0,
        ),
        (
            "the exact distance in the frame of a quaternion not quite unit",
            true,
            1.0,
            0.8,
            true,
            p,
            [0.634936017399154, 0.008205912075554433, -1.4320403873864982],
            [0.634936017399154, 0.008205912075554433, -1.4320403873864982],
            1.0 / 256.0,
            [0.19818663470094447, -0.5210934105185997, 0.8301708952619384],
            4.0,
        ),
        (
            "a rise in the gap counted only past its rounding (turned cone base)",
            true,
            1.0,
            0.8,
            true,
            p,
            [1.4096423077353353, -1.9430159060992016, 1.6571642739508496],
            [1.4096423077353353, -1.9430159060992016, 1.6571642739508496],
            1.0 / 65536.0,
            [-0.7361296676461404, 0.528908761536265, -0.422337109879486],
            4.0,
        ),
        (
            "the barycentric form only where it is precise (cone side)",
            true,
            1.2283558239978447,
            1.1400073115892155,
            true,
            [0.32309568328946625, -1.9802558012543159, 1.9984246786671065],
            [3.1858960097542877, -1.5962567395663427, 3.710440242238292],
            [3.1858960097542877, -1.5962567395663427, 3.710440242238292],
            1.0 / 65536.0,
            [-0.7613638764805728, 0.2934675184214939, -0.5781019488140373],
            10.0,
        ),
    ];
    for (what, cone, r, hh, turned, position, a, b, radius, d, max_t) in cases {
        let solid = Solid {
            kind: if cone { Kind::Cone } else { Kind::Cylinder },
            r,
            hh,
        };
        let scene = if !turned {
            Scene::turned(solid, position, QuatFix::IDENTITY)
        } else if what.contains("cone side") {
            Scene::posed(
                solid,
                position,
                [
                    -0.18230836962811736,
                    -0.9522042122798666,
                    -0.24509344438229164,
                ],
                0.5744975833404169,
            )
        } else {
            Scene::posed(solid, position, [0.6, 0.7, -0.2], 0.9)
        };
        let cast = Cast::new(a, b, d, radius).up_to(max_t);
        let (least, _) = scene.reference(&cast);
        assert!(
            least / TOL < DEEP,
            "{what}: the path goes only {} in",
            least / TOL
        );
        let o = outcome(&scene, &cast);
        assert!(
            matches!(o, Outcome::Hit(g) if (DEEP..=EARLY).contains(&g)),
            "{what}: {o:?}, expected a hit within [{DEEP}, {EARLY}]·2⁻³²"
        );
    }
}
