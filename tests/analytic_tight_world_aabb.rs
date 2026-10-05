//! Tight world boxes for the convex shapes and for a body's collider.
//!
//! Every expected value is a closed form evaluated in `f64`, with the rotation
//! matrix built from the quaternion's components; nothing is read back from the
//! code under test:
//!
//! - box: half-extent `Σ_j |R_ij| h_j`
//! - cylinder along `â = R ê_y`: `|â_i| hh + r √(1 − â_i²)`
//! - cone: the hull of the apex `c + hh â` and the base disc about `c − hh â`,
//!   per axis `[min(a_i, −a_i − s_i), max(a_i, −a_i + s_i)]` with `a_i = hh â_i`,
//!   `s_i = r √(1 − â_i²)`
//! - ellipsoid: `√(Σ_j R_ij² r_j²)`
//! - torus: `R √(1 − â_i²) + r`
//! - wedge: min / max over its six vertices
//! - compound: the union of its children's boxes
//!
//! Each box must contain the closed-form support point of the solid along the
//! six axes and along random directions, reach the support along each axis (so it
//! is tight, not merely containing), and be no larger than the cube of the
//! bounding sphere. The body-level box is observed through
//! `WorldRayCaster::candidates`, which keeps a body exactly when the ray segment
//! meets the body's collider box: bisecting the offset of an axis-parallel ray
//! measures each face of the box.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{contact, gjk, Capsule, Sphere, Support, AABB};
use alice_physics::compound::{CompoundShape, ShapeRef, TransformedCompound};
use alice_physics::cone::Cone;
use alice_physics::cylinder::Cylinder;
use alice_physics::ellipsoid::Ellipsoid;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::{PosedShape, Shape};
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsWorld, SolverConfig};
use alice_physics::torus::Torus;
use alice_physics::wedge::Wedge;

type V = [f64; 3];
type M = [[f64; 3]; 3];

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}
fn arr(v: Vec3Fix) -> V {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

/// A unit quaternion for a random axis and angle in `(−π, π)`.
fn random_rotation(s: &mut u64) -> QuatFix {
    loop {
        let k = [lcg(s), lcg(s), lcg(s)];
        let n = norm(k);
        if n > 0.2 {
            let ang = lcg(s) * std::f64::consts::PI;
            return QuatFix::from_axis_angle(v3(scale(k, 1.0 / n)), fx(ang)).normalize();
        }
    }
}

/// The rotation matrix of a quaternion, in `f64` from its components.
fn matrix(q: QuatFix) -> M {
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
fn apply(m: &M, v: V) -> V {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
fn apply_t(m: &M, v: V) -> V {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// A convex solid in closed form: centre, rotation and dimensions.
#[derive(Clone, Copy, Debug)]
enum Solid {
    Box(V),
    Cylinder { r: f64, hh: f64 },
    Cone { r: f64, hh: f64 },
    Ellipsoid(V),
    Torus { big: f64, small: f64 },
    Wedge { w: f64, h: f64, d: f64 },
}

impl Solid {
    fn shape(self) -> Shape {
        match self {
            Self::Box(h) => Shape::Box {
                half_extents: v3(h),
            },
            Self::Cylinder { r, hh } => Shape::Cylinder {
                radius: fx(r),
                half_height: fx(hh),
            },
            Self::Cone { r, hh } => Shape::Cone {
                radius: fx(r),
                half_height: fx(hh),
            },
            Self::Ellipsoid(r) => Shape::Ellipsoid { radii: v3(r) },
            Self::Torus { big, small } => Shape::Torus {
                major_radius: fx(big),
                minor_radius: fx(small),
            },
            Self::Wedge { w, h, d } => Shape::Wedge {
                width: fx(w),
                height: fx(h),
                depth: fx(d),
            },
        }
    }

    /// The shape's own world box helper, centred at `c` and turned by `q`.
    fn helper_aabb(self, c: Vec3Fix, q: QuatFix) -> AABB {
        match self {
            Self::Box(h) => OrientedBox::new(c, v3(h), q).aabb(),
            Self::Cylinder { r, hh } => Cylinder::with_rotation(c, fx(hh), fx(r), q).aabb(),
            Self::Cone { r, hh } => Cone::with_rotation(c, fx(r), fx(hh), q).aabb(),
            Self::Ellipsoid(r) => Ellipsoid::with_rotation(c, v3(r), q).aabb(),
            Self::Torus { big, small } => Torus::with_rotation(c, fx(big), fx(small), q).aabb(),
            Self::Wedge { w, h, d } => Wedge::with_rotation(c, fx(w), fx(h), fx(d), q).aabb(),
        }
    }

    /// The wedge's six vertices in its own frame.
    fn wedge_vertices(w: f64, h: f64, d: f64) -> [V; 6] {
        let (hw, hh, hd) = (w / 2.0, h / 2.0, d / 2.0);
        [
            [-hw, -hh, -hd],
            [hw, -hh, -hd],
            [0.0, hh, -hd],
            [-hw, -hh, hd],
            [hw, -hh, hd],
            [0.0, hh, hd],
        ]
    }

    /// The closed-form box about the geometric centre `c` (min, max).
    fn closed_form_box(self, c: V, m: &M) -> (V, V) {
        let col = |j: usize| [m[0][j], m[1][j], m[2][j]];
        let a_hat = col(1);
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for i in 0..3 {
            let perp = (1.0 - a_hat[i] * a_hat[i]).max(0.0).sqrt();
            let (l, h) = match self {
                Self::Box(h) => {
                    let e = (0..3).map(|j| m[i][j].abs() * h[j]).sum::<f64>();
                    (-e, e)
                }
                Self::Cylinder { r, hh } => {
                    let e = a_hat[i].abs() * hh + r * perp;
                    (-e, e)
                }
                Self::Cone { r, hh } => {
                    let (ai, si) = (hh * a_hat[i], r * perp);
                    (ai.min(-ai - si), ai.max(-ai + si))
                }
                Self::Ellipsoid(r) => {
                    let e = (0..3).map(|j| (m[i][j] * r[j]).powi(2)).sum::<f64>().sqrt();
                    (-e, e)
                }
                Self::Torus { big, small } => {
                    let e = big * perp + small;
                    (-e, e)
                }
                Self::Wedge { w, h, d } => {
                    let vs = Self::wedge_vertices(w, h, d).map(|p| apply(m, p)[i]);
                    let l = vs.iter().copied().fold(f64::INFINITY, f64::min);
                    let h = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (l, h)
                }
            };
            lo[i] = c[i] + l;
            hi[i] = c[i] + h;
        }
        (lo, hi)
    }

    /// The closed-form support point of the solid along world `d`.
    fn support(self, c: V, m: &M, d: V) -> V {
        let dl = apply_t(m, d);
        let rho = (dl[0] * dl[0] + dl[2] * dl[2]).sqrt();
        let local = match self {
            Self::Box(h) => [
                h[0].copysign(dl[0]),
                h[1].copysign(dl[1]),
                h[2].copysign(dl[2]),
            ],
            Self::Cylinder { r, hh } => {
                let y = if dl[1] >= 0.0 { hh } else { -hh };
                if rho == 0.0 {
                    [0.0, y, 0.0]
                } else {
                    [r * dl[0] / rho, y, r * dl[2] / rho]
                }
            }
            Self::Cone { r, hh } => {
                let rim = if rho == 0.0 {
                    [r, -hh, 0.0]
                } else {
                    [r * dl[0] / rho, -hh, r * dl[2] / rho]
                };
                let apex = [0.0, hh, 0.0];
                if dot(apex, dl) >= dot(rim, dl) {
                    apex
                } else {
                    rim
                }
            }
            Self::Ellipsoid(r) => {
                let s = (0..3).map(|j| (r[j] * dl[j]).powi(2)).sum::<f64>().sqrt();
                [
                    r[0] * r[0] * dl[0] / s,
                    r[1] * r[1] * dl[1] / s,
                    r[2] * r[2] * dl[2] / s,
                ]
            }
            Self::Torus { big, small } => {
                let ring = if rho == 0.0 {
                    [big, 0.0, 0.0]
                } else {
                    [big * dl[0] / rho, 0.0, big * dl[2] / rho]
                };
                add(ring, scale(dl, small / norm(dl)))
            }
            Self::Wedge { w, h, d } => {
                let vs = Self::wedge_vertices(w, h, d);
                let mut best = vs[0];
                for v in vs {
                    if dot(v, dl) > dot(best, dl) {
                        best = v;
                    }
                }
                best
            }
        };
        add(c, apply(m, local))
    }

    /// Radius of the sphere about the centre of mass that encloses the solid.
    fn bounding_radius(self) -> f64 {
        self.shape().bounding_radius().to_f64()
    }

    /// Whether the solid is round in every direction about its centre of mass
    /// (its box can equal the bounding-sphere cube).
    fn is_round(self) -> bool {
        match self {
            Self::Ellipsoid(r) => r[0] == r[1] && r[1] == r[2],
            _ => false,
        }
    }
}

fn solids() -> Vec<Solid> {
    vec![
        Solid::Box([0.5, 1.5, 3.0]),
        Solid::Cylinder { r: 0.4, hh: 3.0 },
        Solid::Cylinder { r: 2.0, hh: 0.25 },
        Solid::Cone { r: 1.0, hh: 2.5 },
        Solid::Cone { r: 2.5, hh: 0.5 },
        Solid::Ellipsoid([0.5, 1.0, 3.0]),
        Solid::Ellipsoid([1.5, 1.5, 1.5]),
        Solid::Torus {
            big: 3.0,
            small: 0.5,
        },
        Solid::Wedge {
            w: 2.0,
            h: 1.0,
            d: 4.0,
        },
    ]
}

/// Six axis directions and `n` random ones.
fn directions(s: &mut u64, n: usize) -> Vec<V> {
    let mut out = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    while out.len() < n + 6 {
        let d = [lcg(s), lcg(s), lcg(s)];
        if norm(d) > 0.1 {
            out.push(d);
        }
    }
    out
}

const TOL: f64 = 1e-9;

fn assert_box_close(got: (V, V), want: (V, V), tol: f64, what: &str) {
    for i in 0..3 {
        assert!(
            (got.0[i] - want.0[i]).abs() < tol && (got.1[i] - want.1[i]).abs() < tol,
            "{what}: axis {i} [{}, {}] vs closed form [{}, {}]",
            got.0[i],
            got.1[i],
            want.0[i],
            want.1[i]
        );
    }
}

fn assert_contains_supports(solid: Solid, c: V, m: &M, b: (V, V), s: &mut u64, what: &str) {
    for d in directions(s, 64) {
        let p = solid.support(c, m, d);
        for i in 0..3 {
            assert!(
                p[i] >= b.0[i] - TOL && p[i] <= b.1[i] + TOL,
                "{what}: support along {d:?} at {p:?} outside axis {i} [{}, {}]",
                b.0[i],
                b.1[i]
            );
        }
    }
    // tight: the support along +-e_i reaches the face
    for i in 0..3 {
        let mut e = [0.0; 3];
        e[i] = 1.0;
        let up = solid.support(c, m, e)[i];
        e[i] = -1.0;
        let down = solid.support(c, m, e)[i];
        assert!(
            (up - b.1[i]).abs() < 1e-7 && (down - b.0[i]).abs() < 1e-7,
            "{what}: axis {i} box [{}, {}] is not reached by the support [{down}, {up}]",
            b.0[i],
            b.1[i]
        );
    }
}

fn box_of(b: AABB) -> (V, V) {
    (arr(b.min), arr(b.max))
}

// ------------------------------------------------------------------ shape helpers

#[test]
fn each_helper_box_is_the_closed_form_and_contains_every_support_point() {
    let mut s = 0x5eed_0001u64;
    for solid in solids() {
        for case in 0..24 {
            let q = if case == 0 {
                QuatFix::IDENTITY
            } else {
                random_rotation(&mut s)
            };
            let c = [lcg(&mut s) * 10.0, lcg(&mut s) * 10.0, lcg(&mut s) * 10.0];
            let m = matrix(q);
            let got = box_of(solid.helper_aabb(v3(c), q));
            let what = format!("{solid:?} case {case}");
            assert_box_close(got, solid.closed_form_box(c, &m), TOL, &what);
            assert_contains_supports(solid, c, &m, got, &mut s, &what);
        }
    }
}

#[test]
fn a_long_thin_cylinder_lying_along_x_has_the_radius_as_its_cross_extent() {
    // Y -> X by a quarter turn about Z
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI).normalize();
    let b = Cylinder::with_rotation(v3([1.0, 2.0, 3.0]), fx(10.0), fx(0.25), q).aabb();
    let (lo, hi) = box_of(b);
    let want_half = [10.0, 0.25, 0.25];
    for i in 0..3 {
        assert!(
            (0.5 * (hi[i] - lo[i]) - want_half[i]).abs() < TOL,
            "axis {i}"
        );
    }
    // the bounding-sphere cube would be sqrt(10^2 + 0.25^2) on every axis
    assert!(0.5 * (hi[1] - lo[1]) < 0.1 * (100.0f64 + 0.0625).sqrt());
}

#[test]
fn a_rotated_box_has_half_extent_abs_r_times_h() {
    let mut s = 77u64;
    for _ in 0..16 {
        let q = random_rotation(&mut s);
        let m = matrix(q);
        let h = [0.3, 1.1, 2.7];
        let (lo, hi) = box_of(OrientedBox::new(Vec3Fix::ZERO, v3(h), q).aabb());
        for i in 0..3 {
            let want = (0..3).map(|j| m[i][j].abs() * h[j]).sum::<f64>();
            assert!((hi[i] - want).abs() < TOL && (lo[i] + want).abs() < TOL);
        }
    }
}

#[test]
fn degenerate_dimensions_and_exact_rotations_give_exact_boxes() {
    let c = v3([1.0, -2.0, 0.5]);
    // 180 degrees about X: (1, 0, 0, 0) is exact, flips Y and Z
    let flip = QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    for q in [QuatFix::IDENTITY, flip] {
        // zero-radius cylinder: the segment of its axis
        let b = Cylinder::with_rotation(c, fx(2.0), Fix128::ZERO, q).aabb();
        assert_eq!(box_of(b), ([1.0, -4.0, 0.5], [1.0, 0.0, 0.5]));
        // zero-size ellipsoid: a point
        let b = Ellipsoid::with_rotation(c, Vec3Fix::ZERO, q).aabb();
        assert_eq!(b, AABB::new(c, c));
        // zero major radius torus: the ball of the tube
        let b = Torus::with_rotation(c, Fix128::ZERO, fx(0.5), q).aabb();
        assert_eq!(box_of(b), ([0.5, -2.5, 0.0], [1.5, -1.5, 1.0]));
        // zero-height cone: its base disc
        let b = Cone::with_rotation(c, fx(1.0), Fix128::ZERO, q).aabb();
        assert_eq!(box_of(b), ([0.0, -2.0, -0.5], [2.0, -2.0, 1.5]));
        // zero-size box
        let b = OrientedBox::new(c, Vec3Fix::ZERO, q).aabb();
        assert_eq!(b, AABB::new(c, c));
    }
    // a cone turned 180 degrees has its apex at -Y: the box is the same shape
    // mirrored, apex side down
    let up = box_of(Cone::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(2.0), QuatFix::IDENTITY).aabb());
    let down = box_of(Cone::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(2.0), flip).aabb());
    assert_eq!(up, ([-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]));
    assert_eq!(down, ([-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]));
    // a cone lying along +X (apex at +X): asymmetric along X only through the disc
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, -Fix128::HALF_PI).normalize();
    let (lo, hi) = box_of(Cone::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(2.0), q).aabb());
    assert!((hi[0] - 2.0).abs() < TOL && (lo[0] + 2.0).abs() < TOL);
    assert!((hi[1] - 1.0).abs() < TOL && (lo[1] + 1.0).abs() < TOL);
}

#[test]
fn a_tilted_cone_box_is_not_symmetric_about_the_centre() {
    // axis tilted 45 degrees in the XY plane: apex at hh (s, s), disc about -hh (s, s)
    let q =
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, -Fix128::PI / Fix128::from_int(4)).normalize();
    let (lo, hi) = box_of(Cone::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(3.0), q).aabb());
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // x: apex 3s, disc reaches -3s - s
    assert!((hi[0] - 3.0 * s).abs() < TOL, "{hi:?}");
    assert!((lo[0] + 4.0 * s).abs() < TOL, "{lo:?}");
    assert!((lo[0] + hi[0]).abs() > 0.5);
}

// ------------------------------------------------------------------ body colliders

/// Measures the collider box of body `i` through the ray caster's candidate set:
/// an axis-parallel ray through the body's position is kept exactly when its
/// offset along axis `k` lies inside the box.
fn probe_box(world: &PhysicsWorld, i: usize, reach: f64) -> (V, V) {
    let caster = world.ray_caster(RayFilter::new());
    let p = arr(world.bodies[i].position);
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for k in 0..3 {
        // the ray runs along axis j != k, through p on the third axis
        let j = (k + 1) % 3;
        let hit = |x: f64| {
            let mut o = p;
            o[k] = x;
            o[j] -= 4.0 * reach;
            let mut d = [0.0; 3];
            d[j] = 1.0;
            caster
                .candidates(v3(o), v3(d), fx(8.0 * reach))
                .contains(&i)
        };
        assert!(
            hit(p[k]),
            "body {i}: the box does not contain its own position"
        );
        assert!(!hit(p[k] + 2.0 * reach) && !hit(p[k] - 2.0 * reach));
        for (sign, out) in [(1.0, &mut hi), (-1.0, &mut lo)] {
            let (mut inside, mut outside) = (0.0f64, 2.0 * reach);
            for _ in 0..60 {
                let mid = 0.5 * (inside + outside);
                if hit(p[k] + sign * mid) {
                    inside = mid;
                } else {
                    outside = mid;
                }
            }
            out[k] = p[k] + sign * inside;
        }
    }
    (lo, hi)
}

fn quiet_world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    })
}

#[test]
fn a_shaped_body_box_is_its_shape_closed_form_box_not_the_sphere_cube() {
    let mut s = 0x5eed_0002u64;
    let mut world = quiet_world();
    let mut placed = Vec::new();
    for (n, solid) in solids().into_iter().enumerate() {
        for k in 0..3 {
            let pos = [40.0 * n as f64, 40.0 * k as f64, lcg(&mut s)];
            let i = world
                .add_shaped_body(&solid.shape(), Fix128::ONE, v3(pos))
                .expect("valid shape");
            let q = if k == 0 {
                QuatFix::IDENTITY
            } else {
                random_rotation(&mut s)
            };
            world.bodies[i].rotation = q;
            placed.push((i, solid, q));
        }
    }
    for (i, solid, q) in placed {
        let m = matrix(q);
        let com = arr(world.bodies[i].position);
        // the geometric centre: the centre of mass moved back by the turned offset
        let offset = arr(solid.shape().center_of_mass_offset());
        let c = add(com, scale(apply(&m, offset), -1.0));
        let r = solid.bounding_radius();
        let got = probe_box(&world, i, r);
        let what = format!("body {i} {solid:?}");
        assert_box_close(got, solid.closed_form_box(c, &m), 1e-7, &what);
        assert_contains_supports(solid, c, &m, got, &mut s, &what);
        // never larger than the bounding-sphere cube about the centre of mass
        for k in 0..3 {
            assert!(got.1[k] <= com[k] + r + 1e-7 && got.0[k] >= com[k] - r - 1e-7);
        }
        // and strictly smaller in volume for every solid that is not round
        let vol = |b: (V, V)| (0..3).map(|k| b.1[k] - b.0[k]).product::<f64>();
        if !solid.is_round() {
            assert!(
                vol(got) < 0.95 * (2.0 * r).powi(3),
                "{what}: box volume {} vs sphere cube {}",
                vol(got),
                (2.0 * r).powi(3)
            );
        }
    }
}

/// The f64 box of a compound child (sphere, box or capsule) placed at
/// `pos + q (local)` with rotation `q * child`.
fn sample_compound() -> CompoundShape {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(v3([0.0, 0.0, 0.0]), fx(0.75)),
        v3([2.0, 0.0, 0.0]),
        QuatFix::IDENTITY,
    );
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3([3.0, 0.25, 0.5]), QuatFix::IDENTITY),
        v3([-1.0, 0.5, 0.0]),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(0.4)).normalize(),
    );
    c.add_capsule(
        Capsule::new(v3([0.0, -1.0, 0.0]), v3([0.0, 1.5, 0.0]), fx(0.3)),
        v3([0.5, -0.5, 1.0]),
        QuatFix::IDENTITY,
    );
    c
}

fn child_box_f64(shape: &ShapeRef, pos: V, m: &M) -> (V, V) {
    match shape {
        ShapeRef::Sphere(sp) => {
            let c = add(pos, apply(m, arr(sp.center)));
            let r = sp.radius.to_f64();
            (
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + r, c[1] + r, c[2] + r],
            )
        }
        ShapeRef::Box(b) => {
            let bm = matrix(b.rotation);
            let mut full = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    full[i][j] = (0..3).map(|k| m[i][k] * bm[k][j]).sum();
                }
            }
            let c = add(pos, apply(m, arr(b.center)));
            Solid::Box(arr(b.half_extents)).closed_form_box(c, &full)
        }
        ShapeRef::Capsule(cap) => {
            let a = add(pos, apply(m, arr(cap.a)));
            let b = add(pos, apply(m, arr(cap.b)));
            let r = cap.radius.to_f64();
            let mut lo = [0.0; 3];
            let mut hi = [0.0; 3];
            for i in 0..3 {
                lo[i] = a[i].min(b[i]) - r;
                hi[i] = a[i].max(b[i]) + r;
            }
            (lo, hi)
        }
        ShapeRef::ConvexHull(_) => unreachable!("not in the sample"),
    }
}

#[test]
fn a_compound_body_box_is_the_union_of_its_children_boxes() {
    let mut s = 0x5eed_0003u64;
    let compound = sample_compound();
    let density = Fix128::ONE;
    let com = arr(compound.mass_properties(density).center_of_mass);
    for case in 0..6 {
        let mut world = quiet_world();
        let pos = [lcg(&mut s) * 5.0, lcg(&mut s) * 5.0, lcg(&mut s) * 5.0];
        let i = world
            .add_compound_body(&compound, density, v3(pos))
            .expect("valid compound");
        let q = if case == 0 {
            QuatFix::IDENTITY
        } else {
            random_rotation(&mut s)
        };
        // the body's frame is the compound's own, turned by q, with the centre of
        // mass at the body's position
        let frame = world.bodies[i].rotation;
        world.bodies[i].rotation = q.mul(frame);
        let m = matrix(q);
        let mut want: Option<(V, V)> = None;
        for child in &compound.children {
            let cm = matrix(child.local_rotation);
            let mut full = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    full[a][b] = (0..3).map(|k| m[a][k] * cm[k][b]).sum();
                }
            }
            let at = add(
                pos,
                apply(&m, add(arr(child.local_position), scale(com, -1.0))),
            );
            let b = child_box_f64(&child.shape, at, &full);
            want = Some(match want {
                None => b,
                Some(w) => {
                    let mut u = w;
                    for k in 0..3 {
                        u.0[k] = u.0[k].min(b.0[k]);
                        u.1[k] = u.1[k].max(b.1[k]);
                    }
                    u
                }
            });
        }
        let got = probe_box(&world, i, 10.0);
        assert_box_close(got, want.expect("children"), 1e-7, &format!("case {case}"));
    }
}

// ------------------------------------------------------------------ no missed contact / hit

/// One convex piece of a scene body, posed in the world, for the brute-force
/// reference.
enum Piece {
    Posed(PosedShape),
    Compound(CompoundShape, Vec3Fix, QuatFix, usize),
}

impl Support for Piece {
    fn support(&self, d: Vec3Fix) -> Vec3Fix {
        match self {
            Self::Posed(p) => p.support(d),
            Self::Compound(c, pos, rot, k) => c.children[*k].support_world(d, *pos, *rot),
        }
    }
}

struct Scene {
    world: PhysicsWorld,
    pieces: Vec<Vec<Piece>>,
    /// Bounding radius of each shaped body, `None` for a compound.
    radii: Vec<Option<f64>>,
}

/// Shaped bodies of every kind and two compounds on a jittered grid, close enough
/// that many bounding spheres overlap while far fewer solids do.
fn mixed_scene() -> Scene {
    let mut s = 0x5eed_0004u64;
    let mut world = quiet_world();
    let mut pieces = Vec::new();
    let mut radii = Vec::new();
    let kinds = solids();
    let mut n = 0usize;
    for gx in 0..5 {
        for gy in 0..4 {
            for gz in 0..3 {
                let pos = [
                    gx as f64 * 3.2 + lcg(&mut s) * 0.6,
                    gy as f64 * 3.2 + lcg(&mut s) * 0.6,
                    gz as f64 * 3.2 + lcg(&mut s) * 0.6,
                ];
                let q = random_rotation(&mut s);
                if n % 23 == 7 {
                    let compound = sample_compound();
                    let i = world
                        .add_compound_body(&compound, Fix128::ONE, v3(pos))
                        .expect("valid compound");
                    let rot = q.mul(world.bodies[i].rotation);
                    world.bodies[i].rotation = rot;
                    // the stored compound is in the body's principal frame; place the
                    // original children equivalently: pos + q (local - com)
                    let com = compound.mass_properties(Fix128::ONE).center_of_mass;
                    let origin = v3(pos) - q.rotate_vec(com);
                    pieces.push(
                        (0..compound.children.len())
                            .map(|k| Piece::Compound(compound.clone(), origin, q, k))
                            .collect(),
                    );
                    radii.push(None);
                } else {
                    let solid = kinds[n % kinds.len()];
                    let i = world
                        .add_shaped_body(&solid.shape(), Fix128::ONE, v3(pos))
                        .expect("valid shape");
                    world.bodies[i].rotation = q;
                    pieces.push(vec![Piece::Posed(PosedShape {
                        shape: solid.shape(),
                        position: v3(pos),
                        rotation: q,
                    })]);
                    radii.push(Some(solid.bounding_radius()));
                }
                n += 1;
            }
        }
    }
    Scene {
        world,
        pieces,
        radii,
    }
}

fn brute_contact_pairs(scene: &Scene) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for a in 0..scene.pieces.len() {
        for b in a + 1..scene.pieces.len() {
            let touching = scene.pieces[a].iter().any(|pa| {
                scene.pieces[b]
                    .iter()
                    .any(|pb| contact(pa, pb).is_some_and(|c| c.depth > Fix128::ZERO))
            });
            if touching {
                out.push((a, b));
            }
        }
    }
    out
}

/// Pairs of shaped bodies whose bounding spheres overlap, and how many of them
/// the solids themselves touch.
fn sphere_pairs(scene: &Scene, touching: &[(usize, usize)]) -> (usize, usize) {
    let w = &scene.world;
    let (mut spheres, mut solids) = (0, 0);
    for a in 0..w.bodies.len() {
        for b in a + 1..w.bodies.len() {
            let (Some(ra), Some(rb)) = (scene.radii[a], scene.radii[b]) else {
                continue;
            };
            let d = norm(add(
                arr(w.bodies[a].position),
                scale(arr(w.bodies[b].position), -1.0),
            ));
            if d < ra + rb {
                spheres += 1;
                if touching.contains(&(a, b)) {
                    solids += 1;
                }
            }
        }
    }
    (spheres, solids)
}

#[test]
fn the_world_finds_every_overlapping_pair_of_a_mixed_scene() {
    let mut scene = mixed_scene();
    let want = brute_contact_pairs(&scene);
    assert!(
        want.len() >= 10,
        "scene too sparse: {} contacts",
        want.len()
    );
    // the scene is one where the sphere test alone over-reports
    let (spheres, solids) = sphere_pairs(&scene, &want);
    assert!(
        spheres > solids,
        "{spheres} sphere pairs vs {solids} contacts"
    );
    scene.world.step(fx(1.0 / 60.0));
    let mut got: Vec<(usize, usize)> = scene
        .world
        .contact_events()
        .iter()
        .map(|e| (e.body_a.min(e.body_b), e.body_a.max(e.body_b)))
        .collect();
    got.sort_unstable();
    got.dedup();
    for p in &want {
        assert!(got.contains(p), "missed contact {p:?}; found {got:?}");
    }
}

#[test]
fn the_ray_caster_misses_no_body_a_ray_meets() {
    let scene = mixed_scene();
    let caster = scene.world.ray_caster(RayFilter::new());
    let mut s = 0x5eed_0005u64;
    let mut checked = 0;
    for _ in 0..200 {
        let o = [lcg(&mut s) * 20.0 + 6.0, lcg(&mut s) * 20.0 + 5.0, -10.0];
        let target = [
            lcg(&mut s) * 8.0 + 6.0,
            lcg(&mut s) * 6.0 + 5.0,
            lcg(&mut s) * 4.0 + 3.0,
        ];
        let d = add(target, scale(o, -1.0));
        let len = norm(d);
        let dir = scale(d, 1.0 / len);
        let reach = 2.0 * len;
        let seg = Capsule::new(v3(o), v3(add(o, scale(dir, reach))), Fix128::ZERO);
        let candidates = caster.candidates(v3(o), v3(dir), fx(reach));
        let hits: Vec<usize> = caster
            .all(v3(o), v3(dir), fx(reach))
            .iter()
            .filter_map(|h| match h.target {
                RayTarget::Body(i) => Some(i),
                _ => None,
            })
            .collect();
        for (i, body) in scene.pieces.iter().enumerate() {
            if body.iter().any(|p| gjk(p, &seg).colliding) {
                checked += 1;
                assert!(candidates.contains(&i), "ray misses the box of body {i}");
                // GJK meets the convex hull; a torus is not convex, and the ray
                // caster tests its real surface (a ray through the hole misses it)
                let torus = matches!(
                    body[0],
                    Piece::Posed(PosedShape {
                        shape: Shape::Torus { .. },
                        ..
                    })
                );
                if !torus {
                    assert!(hits.contains(&i), "ray misses body {i}");
                }
            }
        }
    }
    assert!(checked > 50, "only {checked} ray-body hits");
}

// ------------------------------------------------------------------ compound support

#[test]
fn a_transformed_compound_supports_with_its_farthest_child() {
    let compound = sample_compound();
    let mut s = 0x5eed_0006u64;
    for _ in 0..8 {
        let q = random_rotation(&mut s);
        let pos = v3([lcg(&mut s), lcg(&mut s), lcg(&mut s)]);
        let t = TransformedCompound {
            compound: &compound,
            position: pos,
            rotation: q,
        };
        let world_box = compound.world_aabb(pos, q);
        for d in directions(&mut s, 32) {
            let dir = v3(d);
            let p = t.support(dir);
            assert_eq!(p, compound.support_world(dir, pos, q));
            let best = compound
                .children
                .iter()
                .map(|c| c.support_world(dir, pos, q).dot(dir))
                .max()
                .expect("children");
            assert_eq!(p.dot(dir), best);
            let (a, lo, hi) = (arr(p), arr(world_box.min), arr(world_box.max));
            for k in 0..3 {
                assert!(a[k] >= lo[k] - TOL && a[k] <= hi[k] + TOL, "axis {k}");
            }
        }
        // the local box of each child, moved, is its world box when unturned
        for k in 0..compound.children.len() {
            let child = &compound.children[k];
            if child.local_rotation == QuatFix::IDENTITY {
                let local = child.shape.aabb();
                let moved = AABB::new(
                    local.min + child.local_position + pos,
                    local.max + child.local_position + pos,
                );
                assert_eq!(moved, compound.child_world_aabb(k, pos, QuatFix::IDENTITY));
            }
        }
    }
}
