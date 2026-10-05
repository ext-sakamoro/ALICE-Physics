//! Oracles for `Broadphase::Hybrid` driven by `PhysicsWorld::step`.
//!
//! # What is measured
//!
//! Every broad-phase hands its candidate pairs to the same exact narrow-phase in
//! ascending order, and a pair whose boxes do not meet has no contact, so a world
//! steps to *bit-identical* states whichever broad-phase it uses. The hybrid
//! reports exactly the overlapping pairs (no quantisation, no fat margin), so it
//! hands over the fewest candidates; the reference is the other two kinds.
//!
//! - four scenes (uniform crowd, a pile falling onto a static floor, a crowd with
//!   a few large bodies, a mostly static world) at 100 and 1000 bodies: every
//!   field of every body equal under `Bvh`, `DynamicTree` and `Hybrid` after each
//!   step;
//! - the candidate count of the hybrid equals the brute-force count of pairs of
//!   bodies whose boxes overlap and that are not both static (closed form:
//!   `|p_a − p_b|_k < h_a,k + h_b,k` on every axis, `h` the sphere radius or a
//!   shaped box's half-extents), and the other two hand over at least that many;
//! - a snapshot keeps the choice (`snapshot_world` / `restore_world`, tag 2) and a
//!   restored hybrid world continues exactly like one that never stopped;
//! - switching between the three kinds mid-run changes nothing.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// A 64-bit LCG (high bits), no `rand`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Spheres and boxes of similar size spread through a cube, random velocities.
    Uniform,
    /// Bodies dropped onto a static floor under gravity.
    Pile,
    /// Like `Uniform`, with every 50th body a sphere of radius 3.
    Mixed,
    /// 95% static spheres, the rest moving through them.
    MostlyStatic,
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene(kind: Kind, n: usize, bp: Broadphase) -> PhysicsWorld {
    let gravity = match kind {
        Kind::Pile => v3(0.0, -9.81, 0.0),
        _ => Vec3Fix::ZERO,
    };
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity,
        substeps: 2,
        ..SolverConfig::default()
    });
    w.set_broadphase(bp);
    let mut rng = Lcg(0x51ED_270B_u64.wrapping_mul(n as u64 + 1));
    // keep the density of the crowd the same at every n
    let side = 1.6 * (n as f64).cbrt();
    if let Kind::Pile = kind {
        let floor = w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
        w.set_body_shape(
            floor,
            &Shape::Box {
                half_extents: v3(side, 0.5, side),
            },
        );
    }
    for i in 0..n {
        let p = match kind {
            Kind::Pile => v3(
                rng.between(-side / 2.0, side / 2.0),
                rng.between(0.5, side),
                rng.between(-side / 2.0, side / 2.0),
            ),
            _ => v3(
                rng.between(-side / 2.0, side / 2.0),
                rng.between(-side / 2.0, side / 2.0),
                rng.between(-side / 2.0, side / 2.0),
            ),
        };
        let is_static = matches!(kind, Kind::MostlyStatic) && i % 20 != 0;
        let idx = if is_static {
            w.add_body_with_radius(RigidBody::new_static(p), fx(0.5))
        } else if matches!(kind, Kind::Mixed) && i % 50 == 0 {
            w.add_body_with_radius(RigidBody::new(p, Fix128::from_int(20)), fx(3.0))
        } else if i % 5 == 4 {
            w.add_shaped_body(
                &Shape::Box {
                    half_extents: v3(0.4, 0.3, 0.5),
                },
                Fix128::from_int(1000),
                p,
            )
            .expect("a valid solid")
        } else {
            w.add_body_with_radius(RigidBody::new(p, Fix128::ONE), fx(0.5))
        };
        if !is_static {
            let speed = if let Kind::Pile = kind { 0.5 } else { 2.0 };
            w.get_body_mut(idx).expect("body").velocity = v3(
                rng.between(-speed, speed),
                rng.between(-speed, speed),
                rng.between(-speed, speed),
            );
        }
    }
    w
}

fn assert_same(a: &PhysicsWorld, b: &PhysicsWorld, what: &str) {
    assert_eq!(a.bodies.len(), b.bodies.len(), "{what}: body count");
    for (i, (x, y)) in a.bodies.iter().zip(&b.bodies).enumerate() {
        assert_eq!(x.position, y.position, "{what}: body {i} position");
        assert_eq!(x.velocity, y.velocity, "{what}: body {i} velocity");
        assert_eq!(x.rotation, y.rotation, "{what}: body {i} rotation");
        assert_eq!(
            x.angular_velocity, y.angular_velocity,
            "{what}: body {i} angular velocity"
        );
    }
}

/// Steps the scene under the three broad-phases, comparing every body after every
/// step; returns the candidate pairs each handed over and the contacts seen.
fn run_three(kind: Kind, n: usize, steps: usize) -> ([u64; 3], usize) {
    let kinds = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];
    let mut worlds: Vec<PhysicsWorld> = kinds.iter().map(|&bp| scene(kind, n, bp)).collect();
    let mut pairs = [0u64; 3];
    let mut contacts = 0;
    for step in 0..steps {
        for (k, w) in worlds.iter_mut().enumerate() {
            w.step(dt());
            pairs[k] += w.stage_work().broadphase_pairs;
        }
        contacts += worlds[0].contact_constraints.len();
        for (k, w) in worlds.iter().enumerate().skip(1) {
            assert_same(
                &worlds[0],
                w,
                &format!("{kind:?} n={n} {:?} step {step}", kinds[k]),
            );
        }
    }
    assert_eq!(worlds[0].broadphase(), Broadphase::Bvh);
    assert_eq!(worlds[2].broadphase(), Broadphase::Hybrid);
    (pairs, contacts)
}

fn check(kind: Kind, n: usize, steps: usize) {
    let (pairs, contacts) = run_three(kind, n, steps);
    eprintln!(
        "{kind:?} n={n} steps={steps}: candidates bvh {} tree {} hybrid {}, contacts {contacts}",
        pairs[0], pairs[1], pairs[2]
    );
    assert!(contacts > 0, "{kind:?} n={n}: no body ever touched");
    // the hybrid reports exactly the overlapping boxes: no other kind hands fewer
    assert!(pairs[2] <= pairs[0] && pairs[2] <= pairs[1], "{pairs:?}");
}

#[test]
fn hybrid_uniform_100_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Uniform, 100, 300);
}

#[test]
fn hybrid_pile_100_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Pile, 100, 300);
}

#[test]
fn hybrid_mixed_100_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Mixed, 100, 300);
}

#[test]
fn hybrid_mostly_static_100_is_bit_identical_to_bvh_and_tree() {
    check(Kind::MostlyStatic, 100, 300);
}

#[test]
fn hybrid_uniform_1000_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Uniform, 1000, 30);
}

#[test]
fn hybrid_pile_1000_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Pile, 1000, 30);
}

#[test]
fn hybrid_mixed_1000_is_bit_identical_to_bvh_and_tree() {
    check(Kind::Mixed, 1000, 30);
}

#[test]
fn hybrid_mostly_static_1000_is_bit_identical_to_bvh_and_tree() {
    check(Kind::MostlyStatic, 1000, 30);
}

/// The hybrid's candidates on one step are exactly the brute-force pairs whose
/// boxes (sphere cubes, the shaped boxes' own boxes) overlap and that are not
/// both static.
#[test]
fn hybrid_candidates_are_the_brute_force_box_overlaps() {
    for kind in [Kind::Uniform, Kind::Mixed, Kind::MostlyStatic] {
        let mut w = scene(kind, 300, Broadphase::Hybrid);
        w.config.substeps = 1;
        // no motion in the first substep: the boxes are those of the start
        for b in &mut w.bodies {
            b.velocity = Vec3Fix::ZERO;
        }
        let n = w.bodies.len();
        // half-extents of each body's box: the sphere cube, or for the shaped
        // 0.4×0.3×0.5 box (unrotated at the start) its own half-extents, the
        // closed-form box of its collider
        let half = |i: usize| -> [f64; 3] {
            if matches!(kind, Kind::Mixed) && i % 50 == 0 {
                [3.0; 3]
            } else if !(matches!(kind, Kind::MostlyStatic) && i % 20 != 0) && i % 5 == 4 {
                [0.4, 0.3, 0.5]
            } else {
                [0.5; 3]
            }
        };
        let mut want = 0u64;
        let mut nearest_tie = f64::INFINITY;
        for a in 0..n {
            for b in a + 1..n {
                if w.bodies[a].is_static() && w.bodies[b].is_static() {
                    continue;
                }
                let (pa, pb) = (w.bodies[a].position, w.bodies[b].position);
                let d = [
                    (pa.x - pb.x).to_f64().abs(),
                    (pa.y - pb.y).to_f64().abs(),
                    (pa.z - pb.z).to_f64().abs(),
                ];
                let (ha, hb) = (half(a), half(b));
                let gaps: Vec<f64> = (0..3).map(|k| d[k] - (ha[k] + hb[k])).collect();
                for g in &gaps {
                    nearest_tie = nearest_tie.min(g.abs());
                }
                if gaps.iter().all(|g| *g < 0.0) {
                    want += 1;
                }
            }
        }
        assert!(
            nearest_tie > 1e-9,
            "{kind:?}: a pair touches within {nearest_tie}"
        );
        w.step(dt());
        let got = w.stage_work();
        assert_eq!(got.broadphase_pairs, want, "{kind:?}");
        assert!(
            got.broadphase_box_tests >= want,
            "{kind:?}: {} box tests for {want} pairs",
            got.broadphase_box_tests
        );
        assert!(want > 0, "{kind:?}: no overlap");
    }
}

/// The other kinds report no box tests.
#[test]
fn only_the_hybrid_counts_box_tests() {
    for bp in [Broadphase::Bvh, Broadphase::DynamicTree] {
        let mut w = scene(Kind::Uniform, 100, bp);
        w.step(dt());
        assert!(w.stage_work().broadphase_pairs > 0);
        assert_eq!(w.stage_work().broadphase_box_tests, 0, "{bp:?}");
    }
}

/// A snapshot records the hybrid choice (tag 2), and a restored run continues
/// exactly like one that never stopped.
#[test]
fn a_snapshot_keeps_the_hybrid_and_its_run() {
    let mut reference = scene(Kind::Uniform, 100, Broadphase::Hybrid);
    for _ in 0..20 {
        reference.step(dt());
    }
    let blob = reference.snapshot_world();
    let mut restored = scene(Kind::Uniform, 100, Broadphase::Bvh);
    restored.restore_world(&blob).expect("the snapshot loads");
    assert_eq!(restored.broadphase(), Broadphase::Hybrid);
    assert_same(&reference, &restored, "restored");
    for step in 0..40 {
        reference.step(dt());
        restored.step(dt());
        assert_same(
            &reference,
            &restored,
            &format!("after restore, step {step}"),
        );
    }
    // and the other two tags still decode to their own kinds
    for bp in [Broadphase::Bvh, Broadphase::DynamicTree] {
        let mut w = scene(Kind::Uniform, 10, bp);
        w.step(dt());
        let mut back = scene(Kind::Uniform, 10, Broadphase::Hybrid);
        back.restore_world(&w.snapshot_world()).expect("loads");
        assert_eq!(back.broadphase(), bp);
    }
}

/// Switching among the three kinds every few steps changes nothing.
#[test]
fn switching_among_the_three_mid_run_changes_nothing() {
    let order = [
        Broadphase::Hybrid,
        Broadphase::DynamicTree,
        Broadphase::Hybrid,
        Broadphase::Bvh,
        Broadphase::Hybrid,
    ];
    for kind in [Kind::Uniform, Kind::Pile] {
        let mut steady = scene(kind, 100, Broadphase::Bvh);
        let mut switching = scene(kind, 100, Broadphase::Bvh);
        for step in 0..100 {
            if step % 20 == 5 {
                switching.set_broadphase(order[step / 20]);
            }
            steady.step(dt());
            switching.step(dt());
            assert_same(&steady, &switching, &format!("{kind:?} step {step}"));
        }
        assert_eq!(switching.broadphase(), Broadphase::Hybrid);
    }
}
