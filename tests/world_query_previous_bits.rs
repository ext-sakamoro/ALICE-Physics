//! Change detectors for world casts whose core is grown by `r ≥ 2⁻¹⁶`.
//!
//! - Against polytopes (a box, a wedge, a triangle mesh, a plane) the results
//!   are pinned bit for bit to those of the implementation before the
//!   short-distance precision changes: those apply below `|v| = 2⁻¹⁶`, to GJK
//!   simplices with edges below `2⁻⁸` and to the support directions of rounded
//!   solids, none of which these casts reach.
//! - Against curved solids the results changed at every radius: small GJK
//!   simplices (which a curved surface produces) are scaled up for their
//!   barycentric weights, and an ellipsoid's support is given a direction
//!   scaled up. None of a cone, a cylinder, a height field and an ellipsoid keeps
//!   the previous
//!   bits, so they are pinned to this implementation instead, which also keeps
//!   the time a bracketed gap ends on at the clear end for these radii.
//!
//! # Expected values
//!
//! The digests below are not derived from geometry: they were recorded by running
//! this file (the same inputs, the same digest) against the implementation each
//! table names. Correctness of the values is covered by the closed-form oracles
//! in the `analytic_world_query_*` files; this file only guards against an
//! unintended change of bits.
//!
//! The inputs come from a fixed linear congruential generator and use only `+`,
//! `−`, `×`, `/` and `sqrt` in `f64` (correctly rounded everywhere), so they are
//! the same on every platform.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::world_shape_query::WorldShapeHit;

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

    fn unit(&mut self) -> [f64; 3] {
        loop {
            let v = [self.r(-1.0, 1.0), self.r(-1.0, 1.0), self.r(-1.0, 1.0)];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.1 && l <= 1.0 {
                return [v[0] / l, v[1] / l, v[2] / l];
            }
        }
    }

    /// `2⁻ᵉ·(1 + u)` with `e` uniform in `1..=16`: `[2⁻¹⁶, 1)`.
    fn radius(&mut self) -> f64 {
        let e = 1 + ((self.u() * 16.0) as u32).min(15);
        (1.0 + self.u()) / ((1u64 << e) as f64)
    }
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2]))
}

fn unit_of(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// FNV-1a over the raw bits of a result.
struct Digest(u64);

impl Digest {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn fix(&mut self, f: Fix128) {
        self.bytes(&f.hi.to_le_bytes());
        self.bytes(&f.lo.to_le_bytes());
    }

    fn hit(&mut self, h: &Option<WorldShapeHit>) {
        match h {
            None => self.bytes(&[0]),
            Some(h) => {
                self.bytes(&[1]);
                self.fix(h.t);
                for v in [h.point, h.normal] {
                    self.fix(v.x);
                    self.fix(v.y);
                    self.fix(v.z);
                }
                self.bytes(format!("{:?}", h.target).as_bytes());
            }
        }
    }
}

fn scenes(rng: &mut Rng) -> Vec<(&'static str, PhysicsWorld)> {
    let mut out = Vec::new();
    let shapes = [
        (
            "box",
            Shape::Box {
                half_extents: p3([1.0, 0.6, 0.8]),
            },
        ),
        (
            "wedge",
            Shape::Wedge {
                width: fx(1.6),
                height: fx(1.2),
                depth: fx(1.0),
            },
        ),
    ];
    for (name, s) in shapes {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(i, &s));
        let axis = rng.unit();
        let angle = rng.r(0.0, 3.0);
        w.bodies[i].rotation = QuatFix::from_axis_angle(p3(axis), fx(angle));
        out.push((name, w));
    }
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let mut vs = Vec::new();
    let mut idx = Vec::new();
    for k in 0..24u32 {
        for _ in 0..3 {
            vs.push(p3([rng.r(-2.0, 2.0), rng.r(-1.0, 1.0), rng.r(-2.0, 2.0)]));
        }
        idx.extend_from_slice(&[3 * k, 3 * k + 1, 3 * k + 2]);
    }
    w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(&vs, &idx)));
    out.push(("mesh", w));
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let n = rng.unit();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        p3(n),
        fx(rng.r(-0.5, 0.5)),
    )));
    out.push(("plane", w));
    out
}

/// Rounded solids (a cone, a cylinder, a height field, an ellipsoid), whose casts the
/// short-distance changes do move.
fn round_scenes(rng: &mut Rng) -> Vec<(&'static str, PhysicsWorld)> {
    let mut out = Vec::new();
    let shapes = [
        (
            "cone",
            Shape::Cone {
                radius: fx(1.0),
                half_height: fx(0.8),
            },
        ),
        (
            "cylinder",
            Shape::Cylinder {
                radius: fx(0.9),
                half_height: fx(0.7),
            },
        ),
    ];
    for (name, s) in shapes {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(i, &s));
        let axis = rng.unit();
        let angle = rng.r(0.0, 3.0);
        w.bodies[i].rotation = QuatFix::from_axis_angle(p3(axis), fx(angle));
        out.push((name, w));
    }
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let hs: Vec<Fix128> = (0..81).map(|_| fx(rng.r(-0.8, 0.8))).collect();
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        hs,
        9,
        9,
        fx(0.5),
        p3([-2.0, 0.0, -2.0]),
    )));
    out.push(("field", w));
    // last, turned by a fixed rotation, so the scenes above draw as before
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert!(w.set_body_shape(
        i,
        &Shape::Ellipsoid {
            radii: p3([1.2, 0.5, 0.8]),
        }
    ));
    let k = (0.36f64 + 0.49 + 0.04).sqrt();
    w.bodies[i].rotation = QuatFix::from_axis_angle(p3([0.6 / k, 0.7 / k, -0.2 / k]), fx(0.9));
    out.push(("ellipsoid", w));
    out
}

/// `(scene, casts, hits, digest)` for every scene of `worlds`, `n` shell casts
/// and `n / 4` surface casts each.
fn run(
    rng: &mut Rng,
    worlds: &[(&'static str, PhysicsWorld)],
    n: usize,
) -> Vec<(&'static str, usize, usize, u64)> {
    let f = RayFilter::default();
    let mut out = Vec::new();
    for (name, w) in worlds {
        let mut dg = Digest(0xcbf2_9ce4_8422_2325);
        let (mut casts, mut hits) = (0, 0);
        let mut record = |h: &Option<WorldShapeHit>, casts: &mut usize, hits: &mut usize| {
            dg.hit(h);
            *casts += 1;
            *hits += usize::from(h.is_some());
        };
        // Casts from a shell toward a random point near the solid.
        for _ in 0..n {
            let r = rng.radius();
            let dir0 = rng.unit();
            let dist = rng.r(2.5, 6.0);
            let c = [dir0[0] * dist, dir0[1] * dist, dir0[2] * dist];
            let aim = [rng.r(-1.5, 1.5), rng.r(-0.8, 0.8), rng.r(-1.5, 1.5)];
            let d = unit_of([aim[0] - c[0], aim[1] - c[1], aim[2] - c[2]]);
            let max_t = if rng.u() < 0.2 {
                rng.r(0.5, 3.0)
            } else {
                rng.r(5.0, 15.0)
            };
            let seg = rng.unit();
            let sl = if rng.u() < 0.5 { 0.0 } else { rng.r(0.0, 1.0) };
            let b = [c[0] + seg[0] * sl, c[1] + seg[1] * sl, c[2] + seg[2] * sl];
            let h = if sl == 0.0 {
                w.cast_sphere(p3(c), fx(r), p3(d), fx(max_t), &f)
            } else {
                w.cast_capsule(p3(c), p3(b), fx(r), p3(d), fx(max_t), &f)
            };
            record(&h, &mut casts, &mut hits);
        }
        // A cast from above, then from its contact along the surface, turned in
        // or out by at least 2⁻⁵ (steeper than the near-tangent bound).
        for _ in 0..n / 4 {
            let r = rng.radius();
            let c = [rng.r(-1.0, 1.0), 3.0, rng.r(-1.0, 1.0)];
            let aim = [rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)];
            let d = unit_of([aim[0] - c[0], aim[1] - c[1], aim[2] - c[2]]);
            let h = w.cast_sphere(p3(c), fx(r), p3(d), fx(rng.r(1.0, 10.0)), &f);
            record(&h, &mut casts, &mut hits);
            let tangent = rng.unit();
            let inward = rng.r(1.0 / 32.0, 0.25) * if rng.u() < 0.5 { 1.0 } else { -1.0 };
            let max_t = rng.r(0.1, 5.0);
            if let Some(h) = h.filter(|h| h.t.to_f64() > 1e-3) {
                let t = h.t.to_f64();
                let c2 = [c[0] + d[0] * t, c[1] + d[1] * t, c[2] + d[2] * t];
                let n = [
                    h.normal.x.to_f64(),
                    h.normal.y.to_f64(),
                    h.normal.z.to_f64(),
                ];
                let dn = tangent[0] * n[0] + tangent[1] * n[1] + tangent[2] * n[2];
                let mut tg = [
                    tangent[0] - dn * n[0],
                    tangent[1] - dn * n[1],
                    tangent[2] - dn * n[2],
                ];
                let tl = (tg[0] * tg[0] + tg[1] * tg[1] + tg[2] * tg[2]).sqrt();
                if tl > 1e-3 {
                    for k in 0..3 {
                        tg[k] = tg[k] / tl - inward * n[k];
                    }
                    let h2 = w.cast_sphere(p3(c2), fx(r), p3(tg), fx(max_t), &f);
                    record(&h2, &mut casts, &mut hits);
                }
            }
        }
        out.push((*name, casts, hits, dg.0));
    }
    out
}

/// `(scene, casts, hits, digest)` recorded on the previous implementation.
const PREVIOUS: [(&str, usize, usize, u64); 4] = [
    ("box", 223, 123, 4_909_276_495_861_602_550),
    ("wedge", 217, 75, 8_396_969_157_840_317_528),
    ("mesh", 233, 172, 10_044_748_902_636_547_745),
    ("plane", 231, 158, 10_948_560_710_893_612_363),
];

#[test]
fn polytope_casts_with_radius_from_2_pow_minus_16_keep_their_bits() {
    let mut rng = Rng(0x005e_edd4_0003);
    let worlds = scenes(&mut rng);
    let got = run(&mut rng, &worlds, 160);
    for (g, want) in got.iter().zip(PREVIOUS.iter()) {
        // not vacuous: a good share of the casts hit
        assert!(g.2 * 4 >= g.1, "{}: only {} of {} casts hit", g.0, g.2, g.1);
        assert_eq!(
            g, want,
            "{}: (casts, hits, digest) changed from the previous implementation; all: {got:?}",
            g.0
        );
    }
    assert_eq!(got.len(), PREVIOUS.len());
}

/// `(scene, casts, hits, digest)` of the rounded solids, recorded on this
/// implementation (the scaled small simplices, then the curved faces' nearest
/// points, the flat tetrahedra, the lower bound and the exact distance of
/// cones and cylinders moved them; against an independent distance the same
/// casts went from 66, 16 and 6 below `−2⁻³²` on the cone, the cylinder and
/// the ellipsoid to none): a change detector for the time of impact that
/// a cast ends on when its gap is bracketed (it is the clear end for a core of
/// reach `2⁻¹⁶` or more, and the tangent's root from it only below).
const CURRENT_ROUND: [(&str, usize, usize, u64); 4] = [
    ("cone", 1102, 513, 5_852_212_246_854_234_555),
    ("cylinder", 1144, 624, 598_540_554_569_869_253),
    ("field", 1145, 870, 9_444_083_892_263_037_640),
    ("ellipsoid", 1122, 525, 17_000_410_992_680_744_878),
];

#[test]
fn round_casts_with_radius_from_2_pow_minus_16_keep_their_bits() {
    let mut rng = Rng(0x005e_edd4_0004);
    let worlds = round_scenes(&mut rng);
    let got = run(&mut rng, &worlds, 800);
    for (g, want) in got.iter().zip(CURRENT_ROUND.iter()) {
        assert!(g.2 * 4 >= g.1, "{}: only {} of {} casts hit", g.0, g.2, g.1);
        assert_eq!(
            g, want,
            "{}: (casts, hits, digest) changed; all: {got:?}",
            g.0
        );
    }
    assert_eq!(got.len(), CURRENT_ROUND.len());
}
