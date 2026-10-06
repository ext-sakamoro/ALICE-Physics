//! Oracles for world queries against twisted (non-planar) bilinear height-field
//! cells: the distance a cast advances by must never exceed the true distance,
//! or the cast moves past the first contact and the shape ends inside the
//! surface.
//!
//! Random terrain is checked by brute force in `f64`: points sampled on the
//! bilinear surface give an upper bound on the true distance, so "the sampled
//! distance is not below `r`" can only fail when the swept shape really overlaps
//! the surface.
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion. Nothing here calls the code under test to
//! make an expected value.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics::static_collider::StaticCollider;
use alice_physics::world_shape_query::WorldShapeHit;

const ITER: f64 = 1e-9;
const R: f64 = 0.3;
const SKIN: f64 = 0.01;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    v3(p[0], p[1], p[2])
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn sphere_cast(
    w: &PhysicsWorld,
    c: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_sphere(p3(c), fx(r), p3(d), fx(max), &RayFilter::default())
}

fn capsule_cast(
    w: &PhysicsWorld,
    a: [f64; 3],
    b: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(max), &RayFilter::default())
}

fn overlap_s(w: &PhysicsWorld, c: [f64; 3], r: f64) -> Vec<RayTarget> {
    w.overlap_sphere(p3(c), fx(r), &RayFilter::default())
}

#[track_caller]
fn assert_vec(got: Vec3Fix, want: [f64; 3], tol: f64, what: &str) {
    let g = f3(got);
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() < tol,
            "{what} {g:?} but the closed form is {want:?}"
        );
    }
}

#[track_caller]
fn assert_t(hit: Option<WorldShapeHit>, t: f64, tol: f64) -> WorldShapeHit {
    let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
    assert!(
        (h.t.to_f64() - t).abs() < tol,
        "t = {} but the closed form is {t}",
        h.t.to_f64()
    );
    h
}

/// One twisted cell, heights `1, −1, −1, 1` at `(0,0), (2,0), (0,2), (2,2)`:
/// with `u = x/2`, `v = z/2` the surface is `y = (1 − 2u)(1 − 2v) = (1 − x)(1 − z)`,
/// a saddle through `(1, 0, 1)` with principal curvatures `±1` there.
fn saddle_world() -> (PhysicsWorld, usize) {
    let mut w = world();
    let h = w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        vec![fx(1.0), fx(-1.0), fx(-1.0), fx(1.0)],
        2,
        2,
        fx(2.0),
        Vec3Fix::ZERO,
    )));
    (w, h)
}

#[test]
fn saddle_cell_sphere_cast_stops_where_the_sphere_touches() {
    let (w, h) = saddle_world();
    for r in [0.25, 0.1, 0.05] {
        // oracle: the saddle's tangent plane at (1, 0, 1) is y = 0 and its radii
        // of curvature are 1 > r, so a sphere of radius r coming straight down
        // over (1, ·, 1) first touches at (1, 0, 1): its centre is then at y = r,
        // t = 5 − r, normal +Y.
        let hit = assert_t(
            sphere_cast(&w, [1.0, 5.0, 1.0], r, [0.0, -1.0, 0.0], 100.0),
            5.0 - r,
            ITER,
        );
        assert_eq!(hit.target, RayTarget::StaticCollider(h));
        assert_vec(hit.normal, [0.0, 1.0, 0.0], 1e-6, "normal");
        assert_vec(hit.point, [1.0, 0.0, 1.0], 1e-6, "point");
    }
}

#[test]
fn saddle_cell_overlap_uses_the_true_distance() {
    let (w, h) = saddle_world();
    // oracle: over the centre the distance to the saddle is the height (the
    // tangent plane is y = 0 and the radii of curvature 1 exceed it):
    // 0.3 > 0.25 does not overlap, 0.2 < 0.25 does.
    assert!(overlap_s(&w, [1.0, 0.3, 1.0], 0.25).is_empty());
    assert_eq!(
        overlap_s(&w, [1.0, 0.2, 1.0], 0.25),
        vec![RayTarget::StaticCollider(h)]
    );
    // oracle: a sphere of radius 0.04 whose centre is 0.01 above the surface
    // y = (1 − x)(1 − z) reaches into it (the vertical distance bounds the true
    // one); brute force on the surface confirms it (sampled points bound the
    // distance from above).
    for (x, z) in [(1.0, 1.0), (1.03, 0.97), (1.1, 0.95), (0.9, 1.2)] {
        let c = [x, (1.0 - x) * (1.0 - z) + 0.01, z];
        let r = 0.04;
        let field = Terrain::saddle();
        assert!(field.sampled_distance(c) < r, "the oracle scene overlaps");
        assert_eq!(
            overlap_s(&w, c, r),
            vec![RayTarget::StaticCollider(h)],
            "sphere at {c:?} of radius {r} reaches the saddle"
        );
    }
}

/// A bilinear height field in `f64`, for brute-force distances.
struct Terrain {
    heights: Vec<f64>,
    n: usize,
    spacing: f64,
}

impl Terrain {
    fn saddle() -> Self {
        Self {
            heights: vec![1.0, -1.0, -1.0, 1.0],
            n: 2,
            spacing: 2.0,
        }
    }

    /// Heights from a fixed linear congruential generator, rounded to `1/1024`
    /// (exact in `Fix128` and `f64`).
    fn random(n: usize, amplitude: f64, seed: &mut u64) -> Self {
        let heights = (0..n * n)
            .map(|_| (lcg(seed) * amplitude * 1024.0).round() / 1024.0)
            .collect();
        Self {
            heights,
            n,
            spacing: 1.0,
        }
    }

    fn collider(&self) -> StaticCollider {
        StaticCollider::HeightField(HeightField::new(
            self.heights.iter().map(|&h| fx(h)).collect(),
            self.n as u32,
            self.n as u32,
            fx(self.spacing),
            Vec3Fix::ZERO,
        ))
    }

    fn height(&self, x: f64, z: f64) -> f64 {
        let last = (self.n - 2) as f64;
        let gx = (x / self.spacing).floor().clamp(0.0, last);
        let gz = (z / self.spacing).floor().clamp(0.0, last);
        let (u, v) = (x / self.spacing - gx, z / self.spacing - gz);
        let (i, j) = (gx as usize, gz as usize);
        let h = |a: usize, b: usize| self.heights[b * self.n + a];
        h(i, j) * (1.0 - u) * (1.0 - v)
            + h(i + 1, j) * u * (1.0 - v)
            + h(i, j + 1) * (1.0 - u) * v
            + h(i + 1, j + 1) * u * v
    }

    /// The smallest distance from `f(point)` over sampled surface points: a
    /// coarse grid over the field, then repeated zooms around the best samples.
    fn sampled_min(&self, f: impl Fn([f64; 3]) -> f64) -> f64 {
        let end = self.spacing * (self.n - 1) as f64;
        let k = 48 * (self.n - 1);
        let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
        for i in 0..=k {
            for j in 0..=k {
                let (x, z) = (end * i as f64 / k as f64, end * j as f64 / k as f64);
                seeds.push((f([x, self.height(x, z), z]), x, z));
            }
        }
        seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best = seeds[0].0;
        for &(_, x0, z0) in seeds.iter().take(6) {
            let (mut cx, mut cz, mut half) = (x0, z0, end / k as f64);
            for _ in 0..40 {
                let mut local = (f64::INFINITY, cx, cz);
                for i in -4i32..=4 {
                    for j in -4i32..=4 {
                        let x = (cx + half * f64::from(i) / 4.0).clamp(0.0, end);
                        let z = (cz + half * f64::from(j) / 4.0).clamp(0.0, end);
                        let d = f([x, self.height(x, z), z]);
                        if d < local.0 {
                            local = (d, x, z);
                        }
                    }
                }
                best = best.min(local.0);
                cx = local.1;
                cz = local.2;
                half *= 0.5;
            }
        }
        best
    }

    fn sampled_distance(&self, c: [f64; 3]) -> f64 {
        self.sampled_min(|p| dist(p, c))
    }

    fn sampled_segment_distance(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        self.sampled_min(|p| dist(p, closest_on_segment(a, b, p)))
    }
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn closest_on_segment(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let s = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / len2)
        .clamp(0.0, 1.0);
    [a[0] + ab[0] * s, a[1] + ab[1] * s, a[2] + ab[2] * s]
}

#[test]
fn random_terrain_sphere_casts_touch_without_penetrating() {
    let mut seed = 12_345u64;
    for amplitude in [0.6, 1.5, 3.0] {
        let terrain = Terrain::random(6, amplitude, &mut seed);
        let mut w = world();
        w.add_static_collider(terrain.collider());
        for _ in 0..20 {
            let x = 1.0 + lcg(&mut seed) * 3.0;
            let z = 1.0 + lcg(&mut seed) * 3.0;
            for r in [0.05, 0.2, 0.5] {
                let hit = sphere_cast(&w, [x, 6.0, z], r, [0.0, -1.0, 0.0], 100.0)
                    .unwrap_or_else(|| panic!("a sphere over ({x}, {z}) falls onto the field"));
                let c = [x, 6.0 - hit.t.to_f64(), z];
                let d = terrain.sampled_distance(c);
                // oracle: at the reported contact the sphere touches the surface:
                // the sampled distance (an upper bound) is not below r (no
                // overlap) and within 1e-6 of r (it touches).
                assert!(
                    d >= r - 1e-9,
                    "amplitude {amplitude}, r = {r}: the sphere at {c:?} reaches {} into the surface",
                    r - d
                );
                assert!(
                    d <= r + 1e-6,
                    "amplitude {amplitude}, r = {r}: stopped {} short",
                    d - r
                );
            }
        }
    }
}

#[test]
fn random_terrain_capsule_casts_touch_without_penetrating() {
    let mut seed = 777u64;
    let terrain = Terrain::random(6, 3.0, &mut seed);
    let mut w = world();
    w.add_static_collider(terrain.collider());
    for _ in 0..12 {
        let x = 1.5 + lcg(&mut seed) * 2.0;
        let z = 1.5 + lcg(&mut seed) * 2.0;
        let r = 0.2;
        let (a, b) = ([x - 0.3, 6.0, z - 0.1], [x + 0.3, 6.0, z + 0.1]);
        let hit = capsule_cast(&w, a, b, r, [0.0, -1.0, 0.0], 100.0)
            .unwrap_or_else(|| panic!("a capsule over ({x}, {z}) falls onto the field"));
        let t = hit.t.to_f64();
        let d = terrain.sampled_segment_distance([a[0], a[1] - t, a[2]], [b[0], b[1] - t, b[2]]);
        // oracle: as for spheres, with the distance to the segment.
        assert!(
            d >= r - 1e-9,
            "the capsule reaches {} into the surface",
            r - d
        );
        assert!(d <= r + 1e-6, "the capsule stopped {} short", d - r);
    }
}

fn ctrl(p: [f64; 3], config: CharacterConfig) -> CharacterController {
    CharacterController::new(v3(p[0], p[1], p[2]), config)
}

#[test]
fn landing_on_a_saddle_keeps_the_capsule_off_the_surface() {
    let mut w = world();
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        vec![fx(1.0), fx(-1.0), fx(-1.0), fx(1.0)],
        2,
        2,
        fx(2.0),
        Vec3Fix::ZERO,
    )));
    let surface = |x: f64, z: f64| (1.0 - x) * (1.0 - z);
    for (x, z) in [(1.03, 0.97), (1.0, 1.0), (0.97, 1.02)] {
        let mut c = ctrl([x, 6.0, z], CharacterConfig::default());
        let res = w.move_character(&mut c, v3(0.0, -10.0, 0.0));
        let p = f3(res.position);
        let lower = [p[0], p[1] - (0.9 - R), p[2]];
        let upper = [p[0], p[1] + (0.9 - R), p[2]];
        // oracle: brute-force distance from the capsule's segment to the saddle
        // y = (1 − x)(1 − z) over [0, 2]² (sampled points bound it from above): at
        // least r, no overlap. The capsule lands on a slope and the rest of the
        // move slides it downhill; what is left after the last slide is dropped,
        // so it may end above the surface, but within the ground probe (0.1) of
        // it, grounded.
        let mut d = f64::INFINITY;
        let n = 800;
        for i in 0..=n {
            for j in 0..=n {
                let (sx, sz) = (
                    2.0 * f64::from(i) / f64::from(n),
                    2.0 * f64::from(j) / f64::from(n),
                );
                let q = [sx, surface(sx, sz), sz];
                let c = closest_on_segment(lower, upper, q);
                d = d.min(dist(q, c));
            }
        }
        assert!(
            d >= R - 1e-9,
            "({x}, {z}): the capsule reaches {} into the saddle",
            R - d
        );
        assert!(
            d <= R + SKIN + 0.1,
            "({x}, {z}): it stopped {} above the surface",
            d - R
        );
        assert!(res.grounded, "({x}, {z}): grounded on the saddle");
    }
}
