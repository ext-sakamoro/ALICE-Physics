//! Change detector for the two ray queries, `PhysicsWorld::raycast` (bounding
//! spheres) and `PhysicsWorld::cast_ray` (the real geometry), on scenes of size
//! 1–100: every result is pinned bit for bit.
//!
//! # Why
//!
//! The ray queries are being given a range-safe path (a body whose box or
//! arithmetic leaves the `Fix128` range is still found, with a stated bound).
//! That path must not touch a query that is in range: these digests were
//! recorded on the implementation before the change and must not move.
//!
//! # Expected values
//!
//! The digests are not derived from geometry: they were recorded by running this
//! file against the implementation named in the table. Correctness is covered by
//! the closed-form oracles (`tests/analytic_shape_raycast.rs`, the raycast unit
//! tests); this file only guards against an unintended change of bits.
//!
//! The inputs come from a fixed linear congruential generator and use only `+`,
//! `−`, `×`, `/` and `sqrt` in `f64` (correctly rounded everywhere), so they are
//! the same on every platform.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::collider::{Capsule, ConvexHull, Sphere};
use alice_physics_core::compound::CompoundShape;
use alice_physics_core::heightfield::HeightField;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::shape_raycast::{RayFilter, WorldRayHit};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::trimesh::TriMesh;

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

    fn sphere_hit(&mut self, h: &Option<(usize, Fix128)>) {
        match h {
            None => self.bytes(&[0]),
            Some((i, t)) => {
                self.bytes(&[1]);
                self.bytes(&(*i as u64).to_le_bytes());
                self.fix(*t);
            }
        }
    }

    fn ray_hit(&mut self, h: &Option<WorldRayHit>) {
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
                self.bytes(format!("{:?} {:?}", h.target, h.body).as_bytes());
            }
        }
    }
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    })
}

fn turned(w: &mut PhysicsWorld, i: usize, rng: &mut Rng) {
    let axis = rng.unit();
    let angle = rng.r(0.0, 3.0);
    w.bodies[i].rotation = QuatFix::from_axis_angle(p3(axis), fx(angle));
}

fn scenes(rng: &mut Rng) -> Vec<(&'static str, PhysicsWorld)> {
    let mut out = Vec::new();
    // a body with a collision radius and no shape: the only thing `raycast` sees
    let mut w = world();
    w.add_body_with_radius(RigidBody::new(p3([0.3, -0.2, 0.1]), Fix128::ONE), fx(1.1));
    out.push(("sphere", w));
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
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: p3([1.2, 0.5, 0.8]),
            },
        ),
        (
            "torus",
            Shape::Torus {
                major_radius: fx(1.0),
                minor_radius: fx(0.35),
            },
        ),
    ];
    for (name, s) in shapes {
        let mut w = world();
        let i = w.add_body(RigidBody::new_static(p3([0.2, 0.1, -0.3])));
        assert!(w.set_body_shape(i, &s));
        turned(&mut w, i, rng);
        out.push((name, w));
    }
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere {
            center: p3([1.2, 0.0, 0.0]),
            radius: fx(0.6),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    c.add_capsule(
        Capsule::new(p3([-1.5, 0.0, 0.0]), p3([0.0, 0.8, 0.0]), fx(0.4)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    c.add_convex_hull(
        ConvexHull::new(vec![
            p3([0.0, -1.2, 0.5]),
            p3([0.9, -1.0, -0.6]),
            p3([-0.8, -0.9, -0.7]),
            p3([0.1, -0.2, 0.0]),
            p3([0.3, -1.6, 0.1]),
        ]),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let i = w
        .add_compound_body(&c, Fix128::ONE, p3([0.1, 0.2, 0.0]))
        .expect("valid compound");
    turned(&mut w, i, rng);
    out.push(("compound", w));
    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        p3(unit_of([0.2, 1.0, -0.1])),
        fx(-0.4),
    )));
    out.push(("plane", w));
    let mut w = world();
    let hs: Vec<Fix128> = (0..81).map(|_| fx(rng.r(-0.8, 0.8))).collect();
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        hs,
        9,
        9,
        fx(0.5),
        p3([-2.0, 0.0, -2.0]),
    )));
    out.push(("field", w));
    let mut w = world();
    let verts = [
        p3([-1.5, 0.0, -1.5]),
        p3([1.5, 0.1, -1.5]),
        p3([1.5, -0.1, 1.5]),
        p3([-1.5, 0.2, 1.5]),
        p3([0.0, 1.3, 0.0]),
    ];
    let mesh = TriMesh::from_indexed(
        &verts,
        &[0, 1, 2, 0, 2, 3, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
    );
    w.add_static_collider(StaticCollider::TriMesh(mesh));
    out.push(("mesh", w));
    // many bodies, mixed: the nearest of several, in whatever order the BVH visits
    let mut w = world();
    for k in 0..12 {
        let p = p3([rng.r(-3.0, 3.0), rng.r(-3.0, 3.0), rng.r(-3.0, 3.0)]);
        let i = w.add_body_with_radius(RigidBody::new(p, Fix128::ONE), fx(rng.r(0.3, 1.0)));
        if k % 3 == 1 {
            assert!(w.set_body_shape(
                i,
                &Shape::Box {
                    half_extents: p3([0.5, 0.3, 0.4])
                }
            ));
            turned(&mut w, i, rng);
        }
    }
    out.push(("many", w));
    out
}

/// `(scene, casts, sphere hits, geometry hits, digest)` over `n` rays per scene.
fn run(
    rng: &mut Rng,
    worlds: &[(&'static str, PhysicsWorld)],
    n: usize,
) -> Vec<(&'static str, usize, usize, usize, u64)> {
    let f = RayFilter::default();
    let mut out = Vec::new();
    for (name, w) in worlds {
        let mut dg = Digest(0xcbf2_9ce4_8422_2325);
        let (mut sphere_hits, mut geometry_hits) = (0, 0);
        for _ in 0..n {
            let dir0 = rng.unit();
            let dist = rng.r(2.5, 6.0);
            let c = [dir0[0] * dist, dir0[1] * dist, dir0[2] * dist];
            let aim = [rng.r(-1.5, 1.5), rng.r(-0.8, 0.8), rng.r(-1.5, 1.5)];
            let d = unit_of([aim[0] - c[0], aim[1] - c[1], aim[2] - c[2]]);
            let max_t = if rng.u() < 0.2 {
                rng.r(0.5, 3.0)
            } else {
                rng.r(5.0, 100.0)
            };
            let s = w.raycast(p3(c), p3(d), fx(max_t));
            dg.sphere_hit(&s);
            sphere_hits += usize::from(s.is_some());
            let g = w.cast_ray(p3(c), p3(d), fx(max_t), &f);
            dg.ray_hit(&g);
            geometry_hits += usize::from(g.is_some());
            let all = w.cast_ray_all(p3(c), p3(d), fx(max_t), &f);
            dg.bytes(&(all.len() as u64).to_le_bytes());
            for h in &all {
                dg.ray_hit(&Some(*h));
            }
        }
        out.push((*name, n, sphere_hits, geometry_hits, dg.0));
    }
    out
}

/// Recorded on main `06b1e84b` (2026-11-07), before the range-safe query path.
const EXPECTED: &[(&str, usize, usize, usize, u64)] = &[
    ("sphere", 200, 101, 101, 0x6a703a7e2cf0ab9b),
    ("box", 200, 141, 109, 0x54fd501b00a6759f),
    ("wedge", 200, 83, 42, 0x5bb5f9c6fb0f4c06),
    ("cone", 200, 103, 63, 0xfc1af673cfd9d084),
    ("cylinder", 200, 115, 92, 0x03e724ed0fd5f248),
    ("ellipsoid", 200, 104, 45, 0x56198af3d7f1fb3f),
    ("torus", 200, 116, 82, 0xf4a8f03532b86db7),
    ("compound", 200, 175, 74, 0x61a60ae38eed49d9),
    ("plane", 200, 0, 152, 0x46cf9a9e117a113d),
    ("field", 200, 0, 151, 0x4e96553c5aad0af2),
    ("mesh", 200, 0, 126, 0x931e62495ca1fd31),
    ("many", 200, 85, 66, 0xbf82005d8c186013),
];

#[test]
fn ray_queries_keep_their_previous_bits() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let worlds = scenes(&mut rng);
    let got = run(&mut rng, &worlds, 200);
    let table: Vec<String> = got
        .iter()
        .map(|(n, c, s, g, d)| format!("    ({n:?}, {c}, {s}, {g}, 0x{d:016x}),"))
        .collect();
    assert!(
        got.iter()
            .all(|r| r.2 > 0 || r.0 == "plane" || r.0 == "field" || r.0 == "mesh"),
        "a scene has no sphere hit at all:\n{}",
        table.join("\n")
    );
    assert!(
        got.iter().all(|r| r.3 > 0),
        "a scene has no geometry hit at all:\n{}",
        table.join("\n")
    );
    assert_eq!(
        got.len(),
        EXPECTED.len(),
        "scene count\n{}",
        table.join("\n")
    );
    for (g, e) in got.iter().zip(EXPECTED) {
        assert_eq!(
            g,
            e,
            "{}: bits moved; recorded table:\n{}",
            g.0,
            table.join("\n")
        );
    }
}
