//! Secondary motion chains: the step's stages and constraints against closed forms worked by
//! hand, and the shortest-arc rotation they use.
//!
//! oracles (each independent of the code under test):
//! - at its pose with no gravity a chain stays put bit for bit, step after step;
//! - `QuatFix::from_to` turns `from`'s direction into `to`'s (90°, an arbitrary pair, 180° with a
//!   perpendicular axis, parallel and zero inputs give the identity);
//! - one free particle (no parent constraint can bind: its rest length is its distance) with pull
//!   P, momentum M and no gravity moves by the linear recurrence
//!   `e_{n+1} = (1 − P)(e_n + M (e_n − e_{n−1}))` of its offset from the pose, small amplitude;
//! - under gravity the particle settles where a further step leaves it, and that point is the
//!   root of `f(θ) = 0`, the fixed-point condition on the circle, solved by bisection in `f64`;
//! - the rest length to the parent holds after every step to the resolution of `Fix128`;
//! - an angle limit is never exceeded; a capsule is never entered when the push-out is
//!   perpendicular to the link; with immobile 1 a translated anchor and pose translate the
//!   chain exactly; with immobile 0 and nothing else the particle lags where it was, on its
//!   rest length;
//! - `curve_value` is the cubic Hermite of its keys, held outside them;
//! - two runs give the same bits; bad inputs are refused.
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "a test: failures panic, and exact values are compared on purpose"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the f64 recurrence, the f64 fixed point on the circle and f64 angles, independent of the Fix128 code under test"
)]

use alice_physics_core::collider::Capsule;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::secondary_chain::{
    curve_value, ChainConfig, ChainError, ChainInput, ChainLimit, ChainParams, ChainParticle,
    CurveKey, SecondaryChain,
};

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}

fn to3(p: Vec3Fix) -> [f64; 3] {
    [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()]
}

fn params() -> ChainParams {
    ChainParams {
        pull: f(0.2),
        momentum: f(0.6),
        stiffness: Fix128::ZERO,
        gravity: Fix128::ZERO,
        gravity_falloff: Fix128::ZERO,
        immobile: Fix128::ZERO,
        limit: ChainLimit::None,
        radius: Fix128::ZERO,
        attached_inside: 0,
    }
}

/// A chain of three hanging down from the anchor at (0, 1, 0), 0.1 apart
fn three(p: ChainParams) -> (SecondaryChain, Vec<Vec3Fix>) {
    let pose = vec![v(0.0, 0.9, 0.0), v(0.0, 0.8, 0.0), v(0.0, 0.7, 0.0)];
    // rest lengths measured from the pose, as a caller measures them
    let parts = (0..3)
        .map(|i| ChainParticle {
            parent: if i == 0 { None } else { Some(i - 1) },
            rest_length: (pose[i] - if i == 0 { anchor() } else { pose[i - 1] }).length(),
            params: p,
            p: pose[i],
            q: pose[i],
        })
        .collect();
    let cfg = ChainConfig {
        gravity: v(0.0, -9.81, 0.0),
        iterations: 1,
    };
    (SecondaryChain::new(parts, cfg).unwrap(), pose)
}

const ANCHOR: (f64, f64, f64) = (0.0, 1.0, 0.0);

/// No turn of any bone frame (the identity for every particle)
fn idf(a: &[Vec3Fix]) -> Vec<QuatFix> {
    vec![QuatFix::IDENTITY; a.len()]
}

fn anchor() -> Vec3Fix {
    v(ANCHOR.0, ANCHOR.1, ANCHOR.2)
}

#[test]
fn at_its_pose_without_gravity_a_chain_stays_put_bit_for_bit() {
    let mut p = params();
    p.stiffness = f(0.3);
    p.immobile = f(0.5);
    p.limit = ChainLimit::Angle { max: f(0.5) };
    let (mut c, pose) = three(p);
    let start = c.particles().to_vec();
    for _ in 0..200 {
        c.step(
            f(1.0 / 60.0),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[],
            },
        )
        .unwrap();
    }
    for (a, b) in c.particles().iter().zip(&start) {
        assert_eq!(a.p, b.p);
        assert_eq!(a.q, b.q);
    }
}

#[test]
fn from_to_turns_one_direction_into_the_other() {
    let close = |a: Vec3Fix, b: [f64; 3]| {
        let a = to3(a);
        assert!(
            (0..3).all(|k| (a[k] - b[k]).abs() < 1e-15),
            "{a:?} vs {b:?}"
        );
    };
    // 90° about z: x to y
    let q = QuatFix::from_to(v(1.0, 0.0, 0.0), v(0.0, 2.0, 0.0));
    close(q.rotate_vec(v(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
    close(q.rotate_vec(v(0.0, 0.0, 1.0)), [0.0, 0.0, 1.0]);
    // an arbitrary pair: (1, 2, 3) to (-2, 0.5, 1), lengths differ
    let (a, b) = (v(1.0, 2.0, 3.0), v(-2.0, 0.5, 1.0));
    let q = QuatFix::from_to(a, b);
    let n = (4.0f64 + 0.25 + 1.0).sqrt();
    let s = (1.0f64 + 4.0 + 9.0).sqrt();
    close(q.rotate_vec(a), [-2.0 / n * s, 0.5 / n * s, 1.0 / n * s]);
    // 180°: x to -x, about a perpendicular axis (x's smallest component is y or z: y first)
    let q = QuatFix::from_to(v(1.0, 0.0, 0.0), v(-1.0, 0.0, 0.0));
    close(q.rotate_vec(v(1.0, 0.0, 0.0)), [-1.0, 0.0, 0.0]);
    assert_eq!(q.w, Fix128::ZERO);
    assert_eq!(q.x, Fix128::ZERO, "the axis is perpendicular to x");
    // the identity for parallel or zero inputs
    assert_eq!(
        QuatFix::from_to(v(0.0, 3.0, 0.0), v(0.0, 1.0, 0.0)),
        QuatFix::IDENTITY
    );
    assert_eq!(
        QuatFix::from_to(Vec3Fix::ZERO, v(0.0, 1.0, 0.0)),
        QuatFix::IDENTITY
    );
    assert_eq!(
        QuatFix::from_to(v(1.0, 0.0, 0.0), Vec3Fix::ZERO),
        QuatFix::IDENTITY
    );
}

/// One root particle at rest length r below the anchor, its pose straight down
fn one(p: ChainParams, start: Vec3Fix, r: f64) -> SecondaryChain {
    one_k(p, start, r, 1)
}

fn one_k(p: ChainParams, start: Vec3Fix, r: f64, iterations: u32) -> SecondaryChain {
    SecondaryChain::new(
        vec![ChainParticle {
            parent: None,
            rest_length: f(r),
            params: p,
            p: start,
            q: start,
        }],
        ChainConfig {
            gravity: v(0.0, -9.81, 0.0),
            iterations,
        },
    )
    .unwrap()
}

#[test]
fn a_small_offset_follows_the_linear_recurrence_of_pull_and_momentum() {
    let (pp, mm) = (0.2, 0.6);
    let mut p = params();
    p.pull = f(pp);
    p.momentum = f(mm);
    // the pose 0.1 below the anchor; start 1e-6 aside (rest length kept to the start's distance
    // so the length projection moves it only by the second order, 5e-12)
    let r = 0.1f64;
    let e0 = 1e-6;
    let start = v(e0, 0.9, 0.0);
    let mut c = one(p, start, (r * r + e0 * e0).sqrt());
    let pose = [v(0.0, 0.9, 0.0)];
    let (mut prev, mut cur) = (e0, e0);
    for n in 0..40 {
        c.step(
            f(1.0 / 60.0),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[],
            },
        )
        .unwrap();
        let next = (1.0 - pp) * (cur + mm * (cur - prev));
        let got = c.particles()[0].p.x.to_f64();
        assert!((got - next).abs() < 1e-10, "step {n}: {got} vs {next}");
        prev = cur;
        cur = next;
    }
}

#[test]
fn under_gravity_the_particle_settles_at_the_fixed_point_on_its_circle() {
    // the pose 0.1 to the side (+x) of the anchor; gravity pulls down; pull 0.3, no momentum
    let (pull, grav, dt, r) = (0.3f64, 0.5f64, 1.0f64 / 60.0, 0.1f64);
    let mut p = params();
    p.pull = f(pull);
    p.momentum = Fix128::ZERO;
    p.gravity = f(grav);
    let pose = [v(r, 1.0, 0.0)];
    let mut c = one(p, pose[0], r);
    for _ in 0..4000 {
        c.step(
            f(dt),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[],
            },
        )
        .unwrap();
    }
    let a = c.particles()[0].p;
    let mut c2 = c.clone();
    c2.step(
        f(dt),
        &ChainInput {
            anchor: anchor(),
            anim_anchor: anchor(),
            anim: &pose,
            anim_frames: &idf(&pose),
            anchor_delta: Vec3Fix::ZERO,
            colliders: &[],
        },
    )
    .unwrap();
    let moved = to3(c2.particles()[0].p - a);
    assert!(
        moved.iter().all(|m| m.abs() < 1e-15),
        "not settled: {moved:?}"
    );
    // in f64, independently: the point at angle θ below the pose on the circle,
    // x = (r cos θ, −r sin θ); one step: y = x + g·grav·dt² + (t − x − g·grav·dt²)·pull,
    // then back to the circle; the fixed point has y parallel to x: y_x · x_y − y_y · x_x = 0
    let g = 9.81 * grav * dt * dt;
    let cross = |th: f64| {
        let (x, z) = (r * th.cos(), -r * th.sin());
        let (yx, yz) = (x + (r - x) * pull, (z - g) + (0.0 - (z - g)) * pull);
        yx * z - yz * x
    };
    let (mut lo, mut hi) = (0.0f64, std::f64::consts::FRAC_PI_2);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cross(lo).signum() == cross(mid).signum() {
            lo = mid
        } else {
            hi = mid
        }
    }
    let th = 0.5 * (lo + hi);
    let want = [r * th.cos(), 1.0 - r * th.sin(), 0.0];
    let got = to3(a);
    assert!(
        (0..3).all(|k| (got[k] - want[k]).abs() < 1e-12),
        "{got:?} vs {want:?} (θ {th})"
    );
}

/// A fixed sequence of anchor positions and poses that whip a chain around
fn whip(k: usize) -> (Vec3Fix, Vec<Vec3Fix>) {
    let t = k as f64 * 0.07;
    let a = v(
        0.3 * t.sin(),
        1.0 + 0.2 * (2.0 * t).cos(),
        0.25 * (1.3 * t).sin(),
    );
    let dir = [0.4 * (0.7 * t).cos(), -1.0, 0.5 * (1.1 * t).sin()];
    let n = (dir[0] * dir[0] + 1.0 + dir[2] * dir[2]).sqrt();
    let d = v(dir[0] / n * 0.1, dir[1] / n * 0.1, dir[2] / n * 0.1);
    let pose = vec![a + d, a + d + d, a + d + d + d];
    (a, pose)
}

#[test]
fn gravity_falloff_scales_gravity_by_the_rest_direction_along_it() {
    // the pose 45° down-right; falloff 0.6 scales gravity by 1 − 0.6 · cos 45°; the fixed point on
    // the circle as in the test above, with that share, solved in f64
    let (pull, grav, fall, dt, r) = (0.3f64, 0.5f64, 0.6f64, 1.0f64 / 60.0, 0.1f64);
    let h = r / 2f64.sqrt();
    let mut p = params();
    p.pull = f(pull);
    p.momentum = Fix128::ZERO;
    p.gravity = f(grav);
    p.gravity_falloff = f(fall);
    let pose = [v(h, 1.0 - h, 0.0)];
    let rest = (pose[0] - anchor()).length();
    let mut c = SecondaryChain::new(
        vec![ChainParticle {
            parent: None,
            rest_length: rest,
            params: p,
            p: pose[0],
            q: pose[0],
        }],
        ChainConfig {
            gravity: v(0.0, -9.81, 0.0),
            iterations: 1,
        },
    )
    .unwrap();
    for _ in 0..4000 {
        c.step(
            f(dt),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[],
            },
        )
        .unwrap();
    }
    let got = to3(c.particles()[0].p);
    let share = 1.0 - fall * (1.0 / 2f64.sqrt());
    let g = 9.81 * grav * share * dt * dt;
    let r = rest.to_f64();
    // angle φ below level of the point on the circle; the pose is at φ = 45°
    let (tx, ty) = (h, -h);
    let cross = |ph: f64| {
        let (x, y) = (r * ph.cos(), -r * ph.sin());
        let (yx, yy) = (x + (tx - x) * pull, (y - g) + (ty - (y - g)) * pull);
        yx * y - yy * x
    };
    let (mut lo, mut hi) = (std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cross(lo).signum() == cross(mid).signum() {
            lo = mid
        } else {
            hi = mid
        }
    }
    let ph = 0.5 * (lo + hi);
    let want = [r * ph.cos(), 1.0 - r * ph.sin(), 0.0];
    assert!(
        (0..3).all(|k| (got[k] - want[k]).abs() < 1e-12),
        "{got:?} vs {want:?}"
    );
}

#[test]
fn pull_and_stiffness_aim_from_the_simulated_parent() {
    // two particles; the parent has swung away from its pose; one step of the child with only
    // pull (then only stiffness) moves it toward the simulated parent plus the pose link (turned
    // with the parent link's rotation from its pose, for stiffness), then onto its rest length
    let pose = [v(0.0, 0.9, 0.0), v(0.0, 0.8, 0.0)];
    let parent_now = v(0.06, 0.92, 0.0);
    for (pull, stiff) in [(0.4, 0.0), (0.0, 0.4)] {
        let mut p = params();
        p.pull = Fix128::ZERO;
        p.momentum = Fix128::ZERO;
        let mut pc = p;
        pc.pull = f(pull);
        pc.stiffness = f(stiff);
        let rest0 = (pose[0] - anchor()).length();
        let rest1 = (pose[1] - pose[0]).length();
        let child_now = parent_now + v(0.0, -0.1, 0.0);
        let mut c = SecondaryChain::new(
            vec![
                ChainParticle {
                    parent: None,
                    rest_length: rest0,
                    params: p,
                    p: parent_now,
                    q: parent_now,
                },
                ChainParticle {
                    parent: Some(0),
                    rest_length: rest1,
                    params: pc,
                    p: child_now,
                    q: child_now,
                },
            ],
            ChainConfig {
                gravity: Vec3Fix::ZERO,
                iterations: 1,
            },
        )
        .unwrap();
        c.step(
            f(1.0 / 60.0),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[],
            },
        )
        .unwrap();
        // the parent first: no pull, no momentum: it stays where it is, put on its rest length
        let a = [0.0, 1.0, 0.0];
        let pn = to3(parent_now);
        let d = [pn[0] - a[0], pn[1] - a[1], pn[2] - a[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let par = [
            a[0] + d[0] / n * 0.1,
            a[1] + d[1] / n * 0.1,
            a[2] + d[2] / n * 0.1,
        ];
        // the child's target: the parent (as simulated) plus the pose link (0, −0.1, 0), turned by
        // the rotation from the parent's pose link (0, −0.1, 0) to its simulated link for stiffness
        let link = [0.0, -0.1, 0.0];
        let target = if stiff > 0.0 {
            let u = [par[0] / 0.1 - 0.0, (par[1] - 1.0) / 0.1, par[2] / 0.1];
            // the pose link's direction is (0, −1, 0); turned onto u, (0, −0.1, 0) becomes 0.1 u
            [
                par[0] + 0.1 * u[0],
                par[1] + 0.1 * u[1],
                par[2] + 0.1 * u[2],
            ]
        } else {
            [par[0] + link[0], par[1] + link[1], par[2] + link[2]]
        };
        let k = if stiff > 0.0 { stiff } else { pull };
        let cn = to3(child_now);
        let x = [
            cn[0] + (target[0] - cn[0]) * k,
            cn[1] + (target[1] - cn[1]) * k,
            cn[2] + (target[2] - cn[2]) * k,
        ];
        let d = [x[0] - par[0], x[1] - par[1], x[2] - par[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let want = [
            par[0] + d[0] / n * 0.1,
            par[1] + d[1] / n * 0.1,
            par[2] + d[2] / n * 0.1,
        ];
        let got = to3(c.particles()[1].p);
        assert!(
            (0..3).all(|j| (got[j] - want[j]).abs() < 1e-12),
            "pull {pull} stiffness {stiff}: {got:?} vs {want:?}"
        );
    }
}

#[test]
fn the_rest_length_holds_after_every_step() {
    let mut p = params();
    p.gravity = f(0.8);
    p.stiffness = f(0.2);
    let (mut c, _) = three(p);
    let mut prev = anchor();
    for k in 0..300 {
        let (a, pose) = whip(k);
        c.step(
            f(1.0 / 60.0),
            &ChainInput {
                anchor: a,
                anim_anchor: a,
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: a - prev,
                colliders: &[],
            },
        )
        .unwrap();
        prev = a;
        let ps = c.particles();
        for (i, pt) in ps.iter().enumerate() {
            let parent = pt.parent.map_or(a, |j| ps[j].p);
            let len = (pt.p - parent).length().to_f64();
            assert!((len - 0.1).abs() < 1e-15, "step {k} particle {i}: {len}");
        }
    }
}

#[test]
fn an_angle_limit_is_never_exceeded() {
    let max = 0.35f64;
    let mut p = params();
    p.gravity = f(1.0);
    p.momentum = f(0.9);
    p.pull = f(0.05);
    p.limit = ChainLimit::Angle { max: f(max) };
    let (mut c, _) = three(p);
    let mut prev = anchor();
    let mut worst = 0.0f64;
    for k in 0..300 {
        let (a, pose) = whip(k);
        c.step(
            f(1.0 / 60.0),
            &ChainInput {
                anchor: a,
                anim_anchor: a,
                anim: &pose,
                anim_frames: &idf(&pose),
                anchor_delta: a - prev,
                colliders: &[],
            },
        )
        .unwrap();
        prev = a;
        // the first particle's rest direction is its pose link (it hangs from the anchor)
        let r = to3(pose[0] - a);
        let d = to3(c.particles()[0].p - a);
        let dot = (r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / (0.1 * 0.1);
        worst = worst.max(dot.clamp(-1.0, 1.0).acos());
    }
    assert!(worst <= max + 1e-12, "{worst} > {max}");
    assert!(
        worst > max - 0.05,
        "the limit was never reached ({worst}): the scene does not test it"
    );
}

#[test]
fn a_capsule_is_entered_at_most_by_a_bound_that_shrinks_with_the_passes() {
    // the pose level with the anchor (+x); gravity swings the particle down its circle of
    // radius 0.1 onto a capsule along z whose axis sits on that circle at 60° below level; the
    // final rest-length pass pulls a pushed-out particle back onto its circle, so a particle can
    // stay inside by the curvature of that move (about push² / (2 · length)), less with more
    // passes. Measured bounds: 1 pass < 3e-5, 4 passes < 1e-9
    let (cx, cy) = (
        0.1 * 60f64.to_radians().cos(),
        1.0 - 0.1 * 60f64.to_radians().sin(),
    );
    let cap = Capsule::new(v(cx, cy, -1.0), v(cx, cy, 1.0), f(0.02));
    let mut p = params();
    p.gravity = f(1.0);
    p.pull = f(0.02);
    p.momentum = f(0.5);
    p.radius = f(0.01);
    let pose = [v(0.1, 1.0, 0.0)];
    let mut depths = [0.0f64; 3];
    for (slot, k) in [1u32, 2, 4].into_iter().enumerate() {
        let mut c = one_k(p, pose[0], 0.1, k);
        let (mut worst, mut touched) = (0.0f64, false);
        for _ in 0..600 {
            c.step(
                f(1.0 / 60.0),
                &ChainInput {
                    anchor: anchor(),
                    anim_anchor: anchor(),
                    anim: &pose,
                    anim_frames: &idf(&pose),
                    anchor_delta: Vec3Fix::ZERO,
                    colliders: &[cap],
                },
            )
            .unwrap();
            let q = to3(c.particles()[0].p);
            let d = ((q[0] - cx).powi(2) + (q[1] - cy).powi(2)).sqrt();
            worst = worst.max(0.03 - d);
            // the rest length holds after the push-out too (the last pass is the length)
            let len = (q[0] * q[0] + (q[1] - 1.0).powi(2) + q[2] * q[2]).sqrt();
            assert!((len - 0.1).abs() < 1e-15, "{k} passes: length {len}");
            touched |= d < 0.03 + 1e-6;
        }
        assert!(
            touched,
            "{k} passes: the particle never reached the capsule"
        );
        depths[slot] = worst;
    }
    println!("deepest inside the capsule for 1, 2, 4 passes: {depths:?}");
    assert!(depths[0] < 3e-5, "{depths:?}");
    assert!(depths[2] < 1e-9, "{depths:?}");
    assert!(
        depths[1] <= depths[0] && depths[2] <= depths[1],
        "{depths:?}"
    );
}

#[test]
fn immobile_one_carries_the_chain_with_its_anchor_and_zero_leaves_it() {
    let shift = v(0.05, -0.02, 0.03);
    let mut p = params();
    p.immobile = Fix128::ONE;
    p.pull = Fix128::ZERO;
    p.momentum = Fix128::ZERO;
    let (mut c, pose) = three(p);
    let before = c.particles().to_vec();
    let moved: Vec<Vec3Fix> = pose.iter().map(|&x| x + shift).collect();
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: anchor() + shift,
            anim_anchor: anchor() + shift,
            anim: &moved,
            anim_frames: &idf(&moved),
            anchor_delta: shift,
            colliders: &[],
        },
    )
    .unwrap();
    for (a, b) in c.particles().iter().zip(&before) {
        assert_eq!(a.p, b.p + shift);
    }
    // immobile 0, nothing else: the first particle stays where it was, pulled back onto its
    // rest length from the moved anchor
    p.immobile = Fix128::ZERO;
    let (mut c, pose) = three(p);
    let moved: Vec<Vec3Fix> = pose.iter().map(|&x| x + shift).collect();
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: anchor() + shift,
            anim_anchor: anchor() + shift,
            anim: &moved,
            anim_frames: &idf(&moved),
            anchor_delta: shift,
            colliders: &[],
        },
    )
    .unwrap();
    let a = to3(anchor() + shift);
    let was = [0.0, 0.9, 0.0];
    let d = [was[0] - a[0], was[1] - a[1], was[2] - a[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let want = [
        a[0] + d[0] / n * 0.1,
        a[1] + d[1] / n * 0.1,
        a[2] + d[2] / n * 0.1,
    ];
    let got = to3(c.particles()[0].p);
    assert!(
        (0..3).all(|k| (got[k] - want[k]).abs() < 1e-15),
        "{got:?} vs {want:?}"
    );
}

#[test]
fn a_curve_is_the_hermite_of_its_keys_held_outside() {
    let key = |t: f64, val: f64, i: f64, o: f64| CurveKey {
        time: f(t),
        value: f(val),
        in_slope: f(i),
        out_slope: f(o),
    };
    let keys = [key(0.0, 1.0, 0.0, 0.0), key(1.0, 0.0, 0.0, 0.0)];
    // s = 0.25: h00 = 2/64 − 3/16 + 1 = 0.84375, h01 = 3/16 − 2/64 = 0.15625
    assert_eq!(curve_value(&keys, f(0.25)).to_f64(), 0.84375);
    assert_eq!(curve_value(&keys, f(0.5)).to_f64(), 0.5);
    assert_eq!(curve_value(&keys, f(-1.0)).to_f64(), 1.0);
    assert_eq!(curve_value(&keys, f(2.0)).to_f64(), 0.0);
    // slopes: keys (0.5, 0.8, out 2) and (1, 1, in 0); at t = 0.75, s = 0.5, span 0.5:
    // h00 0.5·0.8 + h10 0.125·0.5·2 + h01 0.5·1 + h11 (−0.125)·0.5·0 = 0.4 + 0.125 + 0.5
    let keys = [key(0.5, 0.8, 0.0, 2.0), key(1.0, 1.0, 0.0, 0.0)];
    assert!((curve_value(&keys, f(0.75)).to_f64() - 1.025).abs() < 1e-15);
    assert_eq!(
        curve_value(&keys, f(0.25)).to_f64(),
        0.8,
        "held before the first key"
    );
    assert_eq!(curve_value(&[], f(0.3)), Fix128::ONE, "no key scales by 1");
}

#[test]
fn two_runs_give_the_same_bits() {
    let run = || {
        let mut p = params();
        p.gravity = f(0.7);
        p.stiffness = f(0.4);
        p.limit = ChainLimit::Polar {
            x_axis: v(1.0, 0.0, 0.0),
            max_x: f(0.6),
            max_z: f(0.3),
        };
        let (mut c, _) = three(p);
        let mut prev = anchor();
        for k in 0..200 {
            let (a, pose) = whip(k);
            c.step(
                f(1.0 / 60.0),
                &ChainInput {
                    anchor: a,
                    anim_anchor: a,
                    anim: &pose,
                    anim_frames: &idf(&pose),
                    anchor_delta: a - prev,
                    colliders: &[],
                },
            )
            .unwrap();
            prev = a;
        }
        c.particles().to_vec()
    };
    assert_eq!(run(), run());
}

#[test]
fn bad_chains_and_inputs_are_refused() {
    let pt = |parent, params| ChainParticle {
        parent,
        rest_length: f(0.1),
        params,
        p: Vec3Fix::ZERO,
        q: Vec3Fix::ZERO,
    };
    let cfg = ChainConfig {
        gravity: Vec3Fix::ZERO,
        iterations: 1,
    };
    assert_eq!(
        SecondaryChain::new(vec![pt(Some(0), params())], cfg).unwrap_err(),
        ChainError::BadParent(0)
    );
    let mut p = params();
    p.pull = f(1.5);
    assert_eq!(
        SecondaryChain::new(vec![pt(None, p)], cfg).unwrap_err(),
        ChainError::BadParam(0, "pull")
    );
    let mut p = params();
    p.immobile = f(-0.1);
    assert_eq!(
        SecondaryChain::new(vec![pt(None, p)], cfg).unwrap_err(),
        ChainError::BadParam(0, "immobile")
    );
    let mut p = params();
    p.radius = f(-0.1);
    assert!(matches!(
        SecondaryChain::new(vec![pt(None, p)], cfg),
        Err(ChainError::BadParam(0, _))
    ));
    let mut p = params();
    p.limit = ChainLimit::Polar {
        x_axis: v(1.0, 0.0, 0.0),
        max_x: Fix128::ZERO,
        max_z: f(0.2),
    };
    assert!(matches!(
        SecondaryChain::new(vec![pt(None, p)], cfg),
        Err(ChainError::BadParam(0, "limit angle"))
    ));
    assert_eq!(
        SecondaryChain::new(
            vec![pt(None, params())],
            ChainConfig {
                gravity: Vec3Fix::ZERO,
                iterations: 0
            }
        )
        .unwrap_err(),
        ChainError::NoIterations
    );
    let mut c = SecondaryChain::new(vec![pt(None, params())], cfg).unwrap();
    assert_eq!(
        c.step(
            f(0.01),
            &ChainInput {
                anchor: anchor(),
                anim_anchor: anchor(),
                anim: &[],
                anim_frames: &idf(&[]),
                anchor_delta: Vec3Fix::ZERO,
                colliders: &[]
            }
        )
        .unwrap_err(),
        ChainError::BadInput("anim")
    );
}

#[test]
fn degenerate_inputs_do_not_panic_and_keep_the_chain_finite() {
    // a zero-length bone, a capsule of coincident ends, dt 0 and an opposite pose link
    let mut p = params();
    p.gravity = f(1.0);
    p.stiffness = f(0.5);
    p.radius = f(0.01);
    let parts = vec![
        ChainParticle {
            parent: None,
            rest_length: Fix128::ZERO,
            params: p,
            p: anchor(),
            q: anchor(),
        },
        ChainParticle {
            parent: Some(0),
            rest_length: f(0.1),
            params: p,
            p: v(0.0, 0.9, 0.0),
            q: v(0.0, 0.9, 0.0),
        },
        ChainParticle {
            parent: Some(1),
            rest_length: f(0.1),
            params: p,
            p: v(0.0, 0.8, 0.0),
            q: v(0.0, 0.8, 0.0),
        },
    ];
    let mut c = SecondaryChain::new(
        parts,
        ChainConfig {
            gravity: v(0.0, -9.81, 0.0),
            iterations: 2,
        },
    )
    .unwrap();
    let cap = Capsule::new(v(0.0, 0.85, 0.0), v(0.0, 0.85, 0.0), f(0.01));
    // the pose turns the second link fully over (antiparallel to the simulated one)
    let pose = [anchor(), v(0.0, 1.1, 0.0), v(0.0, 1.2, 0.0)];
    for dt in [0.0, 1.0 / 60.0] {
        for _ in 0..10 {
            c.step(
                f(dt),
                &ChainInput {
                    anchor: anchor(),
                    anim_anchor: anchor(),
                    anim: &pose,
                    anim_frames: &idf(&pose),
                    anchor_delta: Vec3Fix::ZERO,
                    colliders: &[cap],
                },
            )
            .unwrap();
        }
    }
    for pt in c.particles() {
        let q = to3(pt.p);
        assert!(q.iter().all(|x| x.is_finite() && x.abs() < 10.0), "{q:?}");
    }
}

// --- limit axes live in the bone's animated frame ---------------------------------------------
// oracle: the hinge projects the bone direction onto the plane perpendicular to its axis, the axis
// being the local axis turned by the bone's animated frame (and, below a moved parent, by the
// parent link's turn from its pose); computed here in f64 from those definitions

fn qrot(axis: [f64; 3], angle: f64, p: [f64; 3]) -> [f64; 3] {
    // Rodrigues' rotation
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let kxp = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];
    let kp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
    [0, 1, 2].map(|i| p[i] * c + kxp[i] * s + k[i] * kp * (1.0 - c))
}

fn unit(p: [f64; 3]) -> [f64; 3] {
    let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    [p[0] / n, p[1] / n, p[2] / n]
}

/// One constraint pass on a particle placed off its rest direction, nothing else acting
fn limited_once(
    link: [f64; 3],
    start_dir: [f64; 3],
    limit: ChainLimit,
    frame: QuatFix,
) -> [f64; 3] {
    let mut p = params();
    p.pull = Fix128::ZERO;
    p.momentum = Fix128::ZERO;
    p.limit = limit;
    let a = anchor();
    let len = (link[0] * link[0] + link[1] * link[1] + link[2] * link[2]).sqrt();
    let d = unit(start_dir);
    let start = a + v(d[0] * len, d[1] * len, d[2] * len);
    let anim = [a + v(link[0], link[1], link[2])];
    let mut c = SecondaryChain::new(
        vec![ChainParticle {
            parent: None,
            rest_length: f(len),
            params: p,
            p: start,
            q: start,
        }],
        ChainConfig {
            gravity: Vec3Fix::ZERO,
            iterations: 1,
        },
    )
    .unwrap();
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: a,
            anim_anchor: a,
            anim: &anim,
            anim_frames: &[frame],
            anchor_delta: Vec3Fix::ZERO,
            colliders: &[],
        },
    )
    .unwrap();
    let q = to3(c.particles()[0].p);
    [q[0], q[1] - 1.0, q[2]]
}

fn hinge_expect(len: f64, start_dir: [f64; 3], axis_world: [f64; 3]) -> [f64; 3] {
    let d = unit(start_dir);
    let a = unit(axis_world);
    let k = d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
    let ip = unit([d[0] - a[0] * k, d[1] - a[1] * k, d[2] - a[2] * k]);
    [ip[0] * len, ip[1] * len, ip[2] * len]
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < tol)
}

#[test]
fn a_hinge_turns_with_the_bone_when_the_character_turns() {
    // a bone held out sideways (+x), hinge axis local z: at rest the bone swings in the xy plane;
    // the character turns 90° about the vertical, so the bone points along -z and the hinge axis
    // along +x: the bone must swing in the yz plane
    let turn = std::f64::consts::FRAC_PI_2;
    let frame = QuatFix::from_axis_angle(v(0.0, 1.0, 0.0), f(turn));
    let link = qrot([0.0, 1.0, 0.0], turn, [0.1, 0.0, 0.0]);
    let start = qrot([0.0, 1.0, 0.0], turn, [1.0, -0.4, 0.5]);
    let limit = ChainLimit::Hinge {
        axis: v(0.0, 0.0, 1.0),
        max: Fix128::PI,
    };
    let got = limited_once(link, start, limit, frame);
    let want = hinge_expect(0.1, start, qrot([0.0, 1.0, 0.0], turn, [0.0, 0.0, 1.0]));
    assert!(close(got, want, 1e-12), "{got:?} vs {want:?}");
}

#[test]
fn a_hinge_follows_a_twist_of_the_bone_about_its_own_axis() {
    // a hanging bone (-y), hinge axis local x; the frame twists 90° about the bone itself: the
    // positions do not change, but the hinge axis becomes -z (rot_y(90°) x = -z)
    let twist = std::f64::consts::FRAC_PI_2;
    let frame = QuatFix::from_axis_angle(v(0.0, 1.0, 0.0), f(twist));
    let start = [0.6, -1.0, 0.3];
    let limit = ChainLimit::Hinge {
        axis: v(1.0, 0.0, 0.0),
        max: Fix128::PI,
    };
    let got = limited_once([0.0, -0.1, 0.0], start, limit, frame);
    let want = hinge_expect(0.1, start, qrot([0.0, 1.0, 0.0], twist, [1.0, 0.0, 0.0]));
    assert!(close(got, want, 1e-12), "{got:?} vs {want:?}");
    // and without the twist the axis is x itself
    let got0 = limited_once([0.0, -0.1, 0.0], start, limit, QuatFix::IDENTITY);
    let want0 = hinge_expect(0.1, start, [1.0, 0.0, 0.0]);
    assert!(close(got0, want0, 1e-12), "{got0:?} vs {want0:?}");
}

#[test]
fn a_polar_limit_measures_its_angles_from_the_turned_reference_axis() {
    // a hanging bone, polar limit with the reference axis local x, 0.3 rad toward it and 1.2 rad
    // toward the other axis; the frame twists 60° about the bone, so the reference axis is
    // rot_y(60°) x. A bone pushed 0.8 rad toward that axis is brought back to exactly 0.3 rad
    // toward it (a 60° twist, unlike 90° or 180°, tells the frame from its inverse)
    let twist = std::f64::consts::FRAC_PI_3;
    let frame = QuatFix::from_axis_angle(v(0.0, 1.0, 0.0), f(twist));
    let limit = ChainLimit::Polar {
        x_axis: v(1.0, 0.0, 0.0),
        max_x: f(0.3),
        max_z: f(1.2),
    };
    let xa = qrot([0.0, 1.0, 0.0], twist, [1.0, 0.0, 0.0]);
    let toward = |a: f64| [xa[0] * a.sin(), -a.cos(), xa[2] * a.sin()];
    let start = toward(0.8);
    let got = limited_once([0.0, -0.1, 0.0], start, limit, frame);
    let want = toward(0.3).map(|c| c * 0.1);
    assert!(close(got, want, 1e-12), "{got:?} vs {want:?}");
}

#[test]
fn below_a_moved_parent_a_hinge_axis_also_takes_the_parent_links_turn() {
    // two particles hanging (-y); the parent has swung to point along +x (its link turned by 90°
    // about z); the child's hinge axis local x, frame identity: its axis is turned by the parent
    // link's turn, rot_z(90°) x = +y
    let mut pp = params();
    pp.pull = Fix128::ZERO;
    pp.momentum = Fix128::ZERO;
    let mut pc = pp;
    pc.limit = ChainLimit::Hinge {
        axis: v(1.0, 0.0, 0.0),
        max: Fix128::PI,
    };
    let a = anchor();
    let parent_now = a + v(0.1, 0.0, 0.0);
    let d = unit([1.0, 0.3, 0.5]);
    let child_now = parent_now + v(d[0] * 0.1, d[1] * 0.1, d[2] * 0.1);
    let anim = [a + v(0.0, -0.1, 0.0), a + v(0.0, -0.2, 0.0)];
    let mut c = SecondaryChain::new(
        vec![
            ChainParticle {
                parent: None,
                rest_length: f(0.1),
                params: pp,
                p: parent_now,
                q: parent_now,
            },
            ChainParticle {
                parent: Some(0),
                rest_length: f(0.1),
                params: pc,
                p: child_now,
                q: child_now,
            },
        ],
        ChainConfig {
            gravity: Vec3Fix::ZERO,
            iterations: 1,
        },
    )
    .unwrap();
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: a,
            anim_anchor: a,
            anim: &anim,
            anim_frames: &idf(&anim),
            anchor_delta: Vec3Fix::ZERO,
            colliders: &[],
        },
    )
    .unwrap();
    let q = to3(c.particles()[1].p);
    let got = [q[0] - 1.0 * 0.1, q[1] - 1.0, q[2]];
    let want = hinge_expect(0.1, d, [0.0, 1.0, 0.0]);
    assert!(close(got, want, 1e-12), "{got:?} vs {want:?}");
}

#[test]
fn a_frame_list_of_the_wrong_length_is_refused() {
    let (mut c, pose) = three(params());
    let r = c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: anchor(),
            anim_anchor: anchor(),
            anim: &pose,
            anim_frames: &[QuatFix::IDENTITY],
            anchor_delta: Vec3Fix::ZERO,
            colliders: &[],
        },
    );
    assert_eq!(r, Err(ChainError::BadInput("anim_frames")));
}

#[test]
fn the_parents_turn_is_taken_after_the_bones_own_frame() {
    // as above (the parent swung from -y to +x, a turn of 90° about z), and the child's own frame
    // twisted 90° about y: its axis is turn · frame · x = rot_z(90°) rot_y(90°) x = -z (the other
    // order would give +y)
    let mut pp = params();
    pp.pull = Fix128::ZERO;
    pp.momentum = Fix128::ZERO;
    let mut pc = pp;
    pc.limit = ChainLimit::Hinge {
        axis: v(1.0, 0.0, 0.0),
        max: Fix128::PI,
    };
    let a = anchor();
    let parent_now = a + v(0.1, 0.0, 0.0);
    let d = unit([1.0, 0.3, 0.5]);
    let child_now = parent_now + v(d[0] * 0.1, d[1] * 0.1, d[2] * 0.1);
    let anim = [a + v(0.0, -0.1, 0.0), a + v(0.0, -0.2, 0.0)];
    let frames = [
        QuatFix::IDENTITY,
        QuatFix::from_axis_angle(v(0.0, 1.0, 0.0), f(std::f64::consts::FRAC_PI_2)),
    ];
    let mut c = SecondaryChain::new(
        vec![
            ChainParticle {
                parent: None,
                rest_length: f(0.1),
                params: pp,
                p: parent_now,
                q: parent_now,
            },
            ChainParticle {
                parent: Some(0),
                rest_length: f(0.1),
                params: pc,
                p: child_now,
                q: child_now,
            },
        ],
        ChainConfig {
            gravity: Vec3Fix::ZERO,
            iterations: 1,
        },
    )
    .unwrap();
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: a,
            anim_anchor: a,
            anim: &anim,
            anim_frames: &frames,
            anchor_delta: Vec3Fix::ZERO,
            colliders: &[],
        },
    )
    .unwrap();
    let q = to3(c.particles()[1].p);
    let got = [q[0] - 0.1, q[1] - 1.0, q[2]];
    let axis = qrot(
        [0.0, 0.0, 1.0],
        std::f64::consts::FRAC_PI_2,
        qrot(
            [0.0, 1.0, 0.0],
            std::f64::consts::FRAC_PI_2,
            [1.0, 0.0, 0.0],
        ),
    );
    assert!(close(axis, [0.0, 0.0, -1.0], 1e-12), "{axis:?}");
    let want = hinge_expect(0.1, d, axis);
    assert!(close(got, want, 1e-12), "{got:?} vs {want:?}");
}

// --- attached inside: a link that the authored rest attaches inside a collider is not acted on
// by it ----------------------------------------------------------------------------------------
// oracle: the masks are judged once at the rest (the parent end strictly within the capsule's own
// radius) and then hold in every frame, whatever the pose does

fn rest_chain(p: ChainParams, pose: Vec3Fix, colliders: &[Capsule]) -> SecondaryChain {
    let probe = ChainParticle {
        parent: None,
        rest_length: f(0.1),
        params: p,
        p: pose,
        q: pose,
    };
    let masks = SecondaryChain::attached_inside(&[probe], &[pose], anchor(), colliders).unwrap();
    let mut q = p;
    q.attached_inside = masks[0];
    one(q, pose, 0.1)
}

fn step_with(c: &mut SecondaryChain, anchor_now: Vec3Fix, pose: &[Vec3Fix], caps: &[Capsule]) {
    c.step(
        f(1.0 / 60.0),
        &ChainInput {
            anchor: anchor_now,
            anim_anchor: anchor_now,
            anim: pose,
            anim_frames: &idf(pose),
            anchor_delta: Vec3Fix::ZERO,
            colliders: caps,
        },
    )
    .unwrap();
}

fn held_params() -> ChainParams {
    let mut p = params();
    p.pull = f(0.5);
    p.momentum = Fix128::ZERO;
    p.radius = f(0.01);
    p
}

#[test]
fn the_masks_mark_the_colliders_that_hold_each_links_parent_end_at_rest() {
    // a root (parent end = the anchor) and a child (parent end = the root's rest place 0.1 along x);
    // capsule 0 holds the anchor only, capsule 1 holds the root's place only, capsule 2 neither
    let a = anchor();
    let rest = [a + v(0.1, 0.0, 0.0), a + v(0.2, 0.0, 0.0)];
    let caps = [
        Capsule::new(a + v(0.0, 0.0, -1.0), a + v(0.0, 0.0, 1.0), f(0.05)),
        Capsule::new(a + v(0.1, 0.0, -1.0), a + v(0.1, 0.0, 1.0), f(0.05)),
        Capsule::new(a + v(0.5, 0.5, -1.0), a + v(0.5, 0.5, 1.0), f(0.05)),
    ];
    let pts = [
        ChainParticle {
            parent: None,
            rest_length: f(0.1),
            params: params(),
            p: rest[0],
            q: rest[0],
        },
        ChainParticle {
            parent: Some(0),
            rest_length: f(0.1),
            params: params(),
            p: rest[1],
            q: rest[1],
        },
    ];
    let m = SecondaryChain::attached_inside(&pts, &rest, a, &caps).unwrap();
    assert_eq!(m, vec![0b001, 0b010]);
}

#[test]
fn a_masked_collider_leaves_its_link_as_if_it_were_not_there() {
    // a horizontal bone held at its pose by pull; a capsule along z through the anchor holds the
    // anchor at rest: the particle moves bit for bit as a chain with no capsule, while it sits
    // within the capsule's reach
    let a = anchor();
    let pose = [a + v(0.1, 0.0, 0.0)];
    let cap = Capsule::new(a + v(0.0, 0.0, -1.0), a + v(0.0, 0.0, 1.0), f(0.15));
    let mut c = rest_chain(held_params(), pose[0], &[cap]);
    assert_eq!(c.particles()[0].params.attached_inside, 1);
    let mut free = one(held_params(), pose[0], 0.1);
    for _ in 0..30 {
        step_with(&mut c, a, &pose, &[cap]);
        step_with(&mut free, a, &pose, &[]);
        assert_eq!(c.particles()[0].p, free.particles()[0].p);
        let q = to3(c.particles()[0].p);
        assert!(
            (q[0].powi(2) + (q[1] - 1.0).powi(2)).sqrt() < 0.16,
            "{q:?} left the capsule's reach"
        );
    }
}

#[test]
fn an_unmasked_collider_still_acts() {
    // the capsule holds the particle's place but not the anchor (its axis 1 cm above the particle,
    // where the push has a direction): it moves the particle away from where a free chain is
    let a = anchor();
    let pose = [a + v(0.1, 0.0, 0.0)];
    let cap = Capsule::new(a + v(0.1, 0.01, -1.0), a + v(0.1, 0.01, 1.0), f(0.03));
    let mut c = rest_chain(held_params(), pose[0], &[cap]);
    assert_eq!(c.particles()[0].params.attached_inside, 0);
    step_with(&mut c, a, &pose, &[cap]);
    let q = to3(c.particles()[0].p);
    assert!((q[1] - 1.0).abs() > 1e-3, "the capsule did not act: {q:?}");
}

#[test]
fn a_collider_masked_at_rest_stays_masked_in_a_posed_frame() {
    // masked at rest (it holds the anchor); in the posed frame the anchor has moved 0.3 up, out of
    // the capsule, and the pose places the particle within its reach: still no action (bit for bit
    // as a free chain)
    let a = anchor();
    let cap = Capsule::new(a + v(0.0, 0.0, -1.0), a + v(0.0, 0.0, 1.0), f(0.25));
    let rest_pose = a + v(0.1, 0.0, 0.0);
    let mut c = rest_chain(held_params(), rest_pose, &[cap]);
    assert_eq!(c.particles()[0].params.attached_inside, 1);
    let mut free = one(held_params(), rest_pose, 0.1);
    let moved = a + v(0.0, 0.3, 0.0);
    let pose = [moved + v(0.0, -0.1, 0.0)];
    let mut reached = false;
    for _ in 0..30 {
        step_with(&mut c, moved, &pose, &[cap]);
        step_with(&mut free, moved, &pose, &[]);
        assert_eq!(c.particles()[0].p, free.particles()[0].p);
        let q = to3(c.particles()[0].p);
        reached |= (q[0].powi(2) + (q[1] - 1.0).powi(2)).sqrt() < 0.26;
    }
    assert!(
        reached,
        "the particle never came within the capsule's reach, so the test proves nothing"
    );
}

#[test]
fn an_unmasked_collider_acts_when_it_rises_around_the_root() {
    // unmasked at rest (it is far away); in the posed frame the capsule holds the anchor and the
    // particle: it acts (the particle ends away from where a free chain is)
    let a = anchor();
    let far = Capsule::new(a + v(0.5, 0.5, -1.0), a + v(0.5, 0.5, 1.0), f(0.25));
    let pose = [a + v(0.1, 0.0, 0.0)];
    let mut c = rest_chain(held_params(), pose[0], &[far]);
    assert_eq!(c.particles()[0].params.attached_inside, 0);
    let mut free = one(held_params(), pose[0], 0.1);
    let risen = Capsule::new(a + v(0.0, 0.05, -1.0), a + v(0.0, 0.05, 1.0), f(0.25));
    step_with(&mut c, a, &pose, &[risen]);
    step_with(&mut free, a, &pose, &[]);
    let (q, r) = (to3(c.particles()[0].p), to3(free.particles()[0].p));
    assert!(
        (q[0] - r[0]).abs() + (q[1] - r[1]).abs() > 1e-3,
        "the risen capsule did not act: {q:?} vs {r:?}"
    );
}

#[test]
fn the_masks_use_the_capsules_own_radius_and_refuse_more_than_64() {
    // the anchor 0.0354 from a capsule of radius 0.03: outside it (even though within the capsule
    // plus a particle radius): not masked
    let a = anchor();
    let pose = a + v(0.1, 0.0, 0.0);
    let cap = Capsule::new(a + v(0.035, 0.005, 0.0), a + v(0.2, 0.005, 0.0), f(0.03));
    let mut p = held_params();
    p.radius = f(0.01);
    let c = rest_chain(p, pose, &[cap]);
    assert_eq!(c.particles()[0].params.attached_inside, 0);
    let probe = ChainParticle {
        parent: None,
        rest_length: f(0.1),
        params: p,
        p: pose,
        q: pose,
    };
    let many = vec![cap; 65];
    assert!(SecondaryChain::attached_inside(&[probe], &[pose], a, &many).is_err());
    assert!(SecondaryChain::attached_inside(&[probe], &[pose], a, &many[..64]).is_ok());
}

// --- fit to rest: the authored rest is collision free -------------------------------------------
// oracle: after `fit_to_rest`, a particle's radius against each collider is at most its rest
// clearance (distance from its rest place to the capsule's surface), and a particle whose centre
// the rest puts inside a capsule is masked from it; a chain at its rest then stays there bit for bit
// with the colliders present, and a collider still acts on what comes closer than the rest did

#[test]
fn a_particle_within_reach_at_rest_keeps_its_rest_after_fitting() {
    // the particle 0.035 from a capsule axis of radius 0.03 (clearance 0.005), its radius 0.01:
    // unfitted, the capsule acts at rest; fitted, the chain moves bit for bit as a free chain
    let a = anchor();
    let pose = [a + v(0.1, 0.0, 0.0)];
    let cap = Capsule::new(a + v(0.1, 0.035, -1.0), a + v(0.1, 0.035, 1.0), f(0.03));
    let mut c = one(held_params(), pose[0], 0.1);
    c.fit_to_rest(a, &[cap]).unwrap();
    assert_eq!(c.particles()[0].params.attached_inside, 0);
    let mut free = one(held_params(), pose[0], 0.1);
    let mut unfitted = one(held_params(), pose[0], 0.1);
    for _ in 0..30 {
        step_with(&mut c, a, &pose, &[cap]);
        step_with(&mut free, a, &pose, &[]);
        step_with(&mut unfitted, a, &pose, &[cap]);
        assert_eq!(c.particles()[0].p, free.particles()[0].p);
    }
    assert_ne!(
        unfitted.particles()[0].p,
        free.particles()[0].p,
        "the scene does not make the capsule act unfitted"
    );
}

#[test]
fn a_fitted_collider_still_acts_on_what_comes_closer_than_the_rest_did() {
    // fitted at rest with clearance 0.005 to the capsule's surface; in the posed frame the capsule
    // has moved 0.004 toward the particle (the gap would be 0.001): the particle is moved back to the
    // fitted reach, 0.03 + 0.005 from the axis (then back onto its circle, within the single
    // pass's bound)
    let a = anchor();
    let pose = [a + v(0.1, 0.0, 0.0)];
    let cap = Capsule::new(a + v(0.1, 0.035, -1.0), a + v(0.1, 0.035, 1.0), f(0.03));
    let mut c = one(held_params(), pose[0], 0.1);
    c.fit_to_rest(a, &[cap]).unwrap();
    let near = Capsule::new(a + v(0.1, 0.031, -1.0), a + v(0.1, 0.031, 1.0), f(0.03));
    step_with(&mut c, a, &pose, &[near]);
    let q = to3(c.particles()[0].p);
    let d = ((q[0] - 0.1).powi(2) + (q[1] - 1.031).powi(2)).sqrt();
    assert!(
        d > 0.035 - 1e-4,
        "the capsule did not keep the fitted reach: {d}"
    );
    assert!(
        d < 0.035 + 1e-3,
        "the capsule used the unfitted radius: {d}"
    );
}

#[test]
fn a_particle_whose_centre_is_inside_a_capsule_at_rest_is_masked_from_it() {
    // the particle's rest place inside the capsule (0.02 from an axis of radius 0.03), its parent
    // end (the anchor) outside: the fit masks that capsule
    let a = anchor();
    let pose = a + v(0.1, 0.0, 0.0);
    let cap = Capsule::new(a + v(0.1, 0.02, -1.0), a + v(0.1, 0.02, 1.0), f(0.03));
    let far = Capsule::new(a + v(0.5, 0.5, -1.0), a + v(0.5, 0.5, 1.0), f(0.03));
    let mut c = one(held_params(), pose, 0.1);
    c.fit_to_rest(a, &[far, cap]).unwrap();
    assert_eq!(c.particles()[0].params.attached_inside, 0b10);
}

#[test]
fn a_far_collider_leaves_the_radius_and_the_mask_alone() {
    // nothing near at rest: the fitted chain acts like the unfitted one against a capsule that
    // comes close later
    let a = anchor();
    let pose = [a + v(0.1, 0.0, 0.0)];
    let far = Capsule::new(a + v(0.5, 0.5, -1.0), a + v(0.5, 0.5, 1.0), f(0.03));
    let mut fitted = one(held_params(), pose[0], 0.1);
    fitted.fit_to_rest(a, &[far]).unwrap();
    assert_eq!(fitted.particles()[0].params.attached_inside, 0);
    let mut plain = one(held_params(), pose[0], 0.1);
    let near = Capsule::new(a + v(0.1, 0.035, -1.0), a + v(0.1, 0.035, 1.0), f(0.03));
    step_with(&mut fitted, a, &pose, &[near]);
    step_with(&mut plain, a, &pose, &[near]);
    assert_eq!(fitted.particles()[0].p, plain.particles()[0].p);
}

#[test]
fn fitting_keeps_the_parent_end_rule_and_refuses_more_than_64() {
    // the anchor inside the capsule: masked as by `attached_inside`
    let a = anchor();
    let pose = a + v(0.1, 0.0, 0.0);
    let cap = Capsule::new(a + v(0.0, 0.0, -1.0), a + v(0.0, 0.0, 1.0), f(0.05));
    let mut c = one(held_params(), pose, 0.1);
    c.fit_to_rest(a, &[cap]).unwrap();
    assert_eq!(c.particles()[0].params.attached_inside, 1);
    let mut d = one(held_params(), pose, 0.1);
    assert!(d.fit_to_rest(a, &vec![cap; 65]).is_err());
}
