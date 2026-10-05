//! Contacts between a shaped body and a plain sphere body, and the tight
//! broad-phase boxes that become possible once they are decided by the shape.
//!
//! # What is measured
//!
//! - **Brute force.** On a crowd of plain spheres (radius 0.5) and randomly turned
//!   boxes (half-extents 0.4 × 0.3 × 0.5) with no motion, one step finds exactly
//!   the pairs a brute force over every pair finds: GJK/EPA (`collider::contact`,
//!   depth > 0) on the posed box and the other body's box or sphere, and
//!   `|p_a − p_b| < r_a + r_b` for two spheres. Both `step` and `step_parallel`,
//!   under every broad-phase.
//! - **Candidates.** The hybrid broad-phase hands over exactly the pairs whose
//!   boxes overlap, where a box's box is its closed form `Σ_j |R_ij| h_j` and a
//!   sphere's is its cube; the count is below the count of overlapping sphere
//!   cubes (the boxes before).
//! - **Resting height.** A sphere of radius `r` dropped on the top face of a
//!   static box comes to rest with its centre at the face height plus `r`
//!   (within the solver's resting penetration), not on the box's bounding sphere
//!   (`√(hx² + hy² + hz²) + r` above the box centre).
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::collider::{contact, Sphere};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::{PosedShape, Shape};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

type V = [f64; 3];

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f64) / f64::from(1u32 << 31)
}

fn random_rotation(s: &mut u64) -> QuatFix {
    loop {
        let k = [lcg(s) - 0.5, lcg(s) - 0.5, lcg(s) - 0.5];
        let n = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        if n > 0.1 {
            let axis = [k[0] / n, k[1] / n, k[2] / n];
            return QuatFix::from_axis_angle(v3(axis), fx(lcg(s) * std::f64::consts::PI))
                .normalize();
        }
    }
}

/// The rotation matrix of a quaternion, in `f64` from its components.
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

const H: V = [0.4, 0.3, 0.5];
const R: f64 = 0.5;

fn box_shape() -> Shape {
    Shape::Box {
        half_extents: v3(H),
    }
}

/// What the brute force needs of each body.
#[derive(Clone, Copy)]
enum Body {
    Sphere(Vec3Fix),
    Box(Vec3Fix, QuatFix),
}

/// `n` bodies, every third a turned box, in a cube dense enough that many pairs
/// touch; nothing moves and there is no gravity, so one substep sees the start.
fn crowd(n: usize, bp: Broadphase) -> (PhysicsWorld, Vec<Body>) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    w.set_broadphase(bp);
    let mut s = 0x5eed_0a11u64;
    let side = 1.5 * (n as f64).cbrt();
    let mut bodies = Vec::new();
    for i in 0..n {
        let p = v3([lcg(&mut s) * side, lcg(&mut s) * side, lcg(&mut s) * side]);
        if i % 3 == 1 {
            let q = random_rotation(&mut s);
            let k = w
                .add_shaped_body(&box_shape(), Fix128::from_int(1000), p)
                .expect("a valid box");
            w.bodies[k].rotation = q;
            bodies.push(Body::Box(p, q));
        } else {
            w.add_body_with_radius(RigidBody::new(p, Fix128::ONE), fx(R));
            bodies.push(Body::Sphere(p));
        }
    }
    (w, bodies)
}

fn brute_contacts(bodies: &[Body]) -> Vec<(usize, usize)> {
    let posed = |p, q| PosedShape {
        shape: box_shape(),
        position: p,
        rotation: q,
    };
    let mut out = Vec::new();
    for a in 0..bodies.len() {
        for b in a + 1..bodies.len() {
            let deep = |c: Option<alice_physics::collider::Contact>| {
                c.is_some_and(|c| c.depth > Fix128::ZERO)
            };
            let hit = match (bodies[a], bodies[b]) {
                (Body::Sphere(pa), Body::Sphere(pb)) => {
                    let reach = fx(2.0 * R);
                    let d = (pa - pb).length_squared();
                    d < reach * reach && !d.is_zero()
                }
                (Body::Box(pa, qa), Body::Box(pb, qb)) => {
                    deep(contact(&posed(pa, qa), &posed(pb, qb)))
                }
                (Body::Box(pa, qa), Body::Sphere(pb)) => {
                    deep(contact(&posed(pa, qa), &Sphere::new(pb, fx(R))))
                }
                (Body::Sphere(pa), Body::Box(pb, qb)) => {
                    deep(contact(&Sphere::new(pa, fx(R)), &posed(pb, qb)))
                }
            };
            if hit {
                out.push((a, b));
            }
        }
    }
    out
}

/// The closed-form box of each body: `Σ_j |R_ij| h_j` for a box, the cube for a
/// sphere; plus the bounding-sphere cube every body had before.
fn boxes(bodies: &[Body], tight: bool) -> Vec<(V, V)> {
    let bound = (H[0] * H[0] + H[1] * H[1] + H[2] * H[2]).sqrt();
    bodies
        .iter()
        .map(|b| {
            let (p, half) = match *b {
                Body::Sphere(p) => (p, [R; 3]),
                Body::Box(p, q) if tight => {
                    let m = matrix(q);
                    let mut half = [0.0; 3];
                    for i in 0..3 {
                        half[i] = (0..3).map(|j| m[i][j].abs() * H[j]).sum();
                    }
                    (p, half)
                }
                Body::Box(p, _) => (p, [bound; 3]),
            };
            let c = [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()];
            (
                [c[0] - half[0], c[1] - half[1], c[2] - half[2]],
                [c[0] + half[0], c[1] + half[1], c[2] + half[2]],
            )
        })
        .collect()
}

/// Pairs whose boxes overlap, and the smallest gap to touching over all pairs and
/// axes (so a near tie, where f64 and the fixed point could disagree, is seen).
fn overlapping(boxes: &[(V, V)]) -> (u64, f64) {
    let mut count = 0;
    let mut nearest = f64::INFINITY;
    for a in 0..boxes.len() {
        for b in a + 1..boxes.len() {
            let mut all = true;
            for k in 0..3 {
                let gap = (boxes[a].0[k] - boxes[b].1[k]).max(boxes[b].0[k] - boxes[a].1[k]);
                nearest = nearest.min(gap.abs());
                all &= gap < 0.0;
            }
            if all {
                count += 1;
            }
        }
    }
    (count, nearest)
}

fn found(w: &PhysicsWorld) -> Vec<(usize, usize)> {
    let mut got: Vec<(usize, usize)> = w
        .contact_constraints
        .iter()
        .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
        .collect();
    got.sort_unstable();
    got.dedup();
    got
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn check_contacts(step: fn(&mut PhysicsWorld), what: &str) {
    for bp in [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid] {
        let (mut w, bodies) = crowd(150, bp);
        let want = brute_contacts(&bodies);
        let mixed = want
            .iter()
            .filter(|&&(a, b)| {
                matches!(bodies[a], Body::Box(..)) != matches!(bodies[b], Body::Box(..))
            })
            .count();
        assert!(mixed >= 10, "{what}: only {mixed} box-sphere contacts");
        step(&mut w);
        let got = found(&w);
        let missed: Vec<_> = want.iter().filter(|p| !got.contains(p)).collect();
        let extra: Vec<_> = got.iter().filter(|p| !want.contains(p)).collect();
        assert!(
            missed.is_empty() && extra.is_empty(),
            "{what} {bp:?}: {} contacts, brute {}; missed {missed:?}, extra {extra:?}",
            got.len(),
            want.len()
        );
    }
}

/// `step` finds exactly the brute-force contacts, shaped × sphere included.
#[test]
fn step_finds_exactly_the_brute_force_contacts_of_boxes_and_spheres() {
    check_contacts(|w| w.step(dt()), "step");
}

/// `step_parallel` finds the same.
#[cfg(feature = "parallel")]
#[test]
fn step_parallel_finds_exactly_the_brute_force_contacts_of_boxes_and_spheres() {
    check_contacts(|w| w.step_parallel(dt()), "step_parallel");
}

/// The hybrid's candidates are exactly the overlapping tight boxes, fewer than
/// the overlapping bounding-sphere cubes.
#[test]
fn the_broadphase_hands_over_the_tight_box_overlaps() {
    let (_, bodies) = crowd(150, Broadphase::Bvh);
    let (tight, tie) = overlapping(&boxes(&bodies, true));
    let (cubes, _) = overlapping(&boxes(&bodies, false));
    assert!(tie > 1e-9, "a pair of boxes touches within {tie}");
    assert!(tight < cubes, "{tight} tight vs {cubes} cube overlaps");
    let mut counts = Vec::new();
    for bp in [Broadphase::Hybrid, Broadphase::Bvh, Broadphase::DynamicTree] {
        let (mut w, _) = crowd(150, bp);
        w.step(dt());
        counts.push(w.stage_work().broadphase_pairs);
    }
    eprintln!("tight {tight} cubes {cubes} hybrid/bvh/tree {counts:?}");
    // the BVH (quantised leaf boxes) and the tree (fattened proxies) hand over
    // supersets whose size depends on their own slack, so only the hybrid's
    // count is a closed form
    assert_eq!(counts[0], tight, "hybrid");
}

/// A sphere dropped on the top face of a static box rests at the face height plus
/// its radius, whichever index it has; not on the bounding sphere of the box.
#[test]
fn a_sphere_rests_on_the_face_of_a_box_not_on_its_bounding_sphere() {
    let half: V = [2.0, 0.5, 2.0];
    let face = half[1];
    let bound = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    for sphere_first in [false, true] {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        let add_box = |w: &mut PhysicsWorld| {
            let k = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            assert!(w.set_body_shape(
                k,
                &Shape::Box {
                    half_extents: v3(half)
                }
            ));
            k
        };
        let add_ball = |w: &mut PhysicsWorld| {
            w.add_body_with_radius(RigidBody::new(v3([0.3, 1.2, -0.2]), Fix128::ONE), fx(R))
        };
        let ball = if sphere_first {
            let k = add_ball(&mut w);
            add_box(&mut w);
            k
        } else {
            add_box(&mut w);
            add_ball(&mut w)
        };
        for _ in 0..240 {
            w.step(dt());
        }
        let y = w.bodies[ball].position.y.to_f64();
        let speed = w.bodies[ball].velocity.length().to_f64();
        eprintln!(
            "sphere_first={sphere_first}: rest y {y} (face + r = {}, bounding = {})",
            face + R,
            bound + R
        );
        assert!(
            (y - (face + R)).abs() < 0.02,
            "sphere_first={sphere_first}: rests at y {y}, want {}",
            face + R
        );
        assert!(speed < 0.05, "still moving at {speed}");
    }
}

/// A static, sleeping shaped body is parked with its tight box; turning it while
/// it is parked re-boxes it, so an awake sphere meets it where it now is. The
/// run equals one with the sleep skip off, and the sphere stops on the turned
/// rod's top end (y = 3 + r) instead of where the rod used to be (y = 0.2 + r).
#[test]
fn turning_a_parked_static_shape_re_boxes_it() {
    let run = |skip: bool| {
        // no gravity and no damping: the ball keeps its speed until it hits
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            damping: Fix128::ONE,
            ..SolverConfig::default()
        });
        w.set_sleep_skip(skip);
        let rod = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(
            rod,
            &Shape::Box {
                half_extents: v3([3.0, 0.2, 0.2])
            }
        ));
        let ball = w.add_body_with_radius(RigidBody::new(v3([0.0, 40.0, 0.0]), Fix128::ONE), fx(R));
        w.bodies[ball].velocity = v3([0.0, -2.0, 0.0]);
        let mut parked = false;
        let mut lowest = f64::INFINITY;
        for frame in 0..1300 {
            if frame == 300 {
                // the ball is still far above (y = 40 − 2·5 = 30); the rod now
                // stands along y
                assert!(w.bodies[ball].position.y.to_f64() > 29.0);
                w.bodies[rod].rotation =
                    QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(std::f64::consts::FRAC_PI_2))
                        .normalize();
            }
            w.step(dt());
            if frame < 300 {
                parked |= w.stage_work().parked > 0;
            }
            lowest = lowest.min(w.bodies[ball].position.y.to_f64());
        }
        let ball_v = w.bodies[ball].velocity.y.to_f64();
        (w, parked, lowest, ball_v)
    };
    let (skipping, parked, lowest, vy) = run(true);
    let (plain, _, _, _) = run(false);
    // the ball reached the rod and bounced back up, no faster than it came
    assert!((0.0..=2.0).contains(&vy), "the ball leaves at {vy} m/s");
    assert!(parked, "the rod was never parked");
    for (i, (x, y)) in skipping.bodies.iter().zip(&plain.bodies).enumerate() {
        assert_eq!(x.position, y.position, "body {i} position");
        assert_eq!(x.velocity, y.velocity, "body {i} velocity");
    }
    assert!(
        lowest > 3.0 + R - 0.05,
        "the ball went down to y = {lowest}, through the turned rod"
    );
}
