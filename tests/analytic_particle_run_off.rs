//! Oracles for `ParticleSystem::step_with_run_off` (a drop that lands stays on what it
//! hit, runs down it and drips) and `ParticleSystem::state_digest`.
//!
//! - on an incline the speed reaches the closed form `g sin(theta) / drag` and the drop
//!   stays on the plane
//! - on a sphere the drop drips where `n·ĝ` reaches the cosine it was told
//! - past the edge of a shelf the drop falls free
//! - a query surface holds the drop only where it is occupied
//! - held drops stay on square-edged and curved strokes with a blended normal, and a slide
//!   round a curve covers its arc (it does not cut across it)
//! - the same calls give the same bits; every part of the state moves the digest; a
//!   digest is pinned

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the closed forms are evaluated in f64 by the test, not by the engine"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::particle::{
    LandingTarget, Particle, ParticleEmitter, ParticleSystem, RunOff,
};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::ClosureSdf;

const G: f64 = 10.0;

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn f(x: Fix128) -> f64 {
    x.to_f64()
}

/// One drop falling from `from`, nothing emitted, no air damping.
fn one_drop(from: Vec3Fix) -> ParticleSystem {
    let mut s = ParticleSystem::new(4, v(0.0, -G, 0.0));
    s.damping = Fix128::ONE;
    s.particles.push(Particle::new(
        from,
        Vec3Fix::ZERO,
        Fix128::from_int(1000),
        Fix128::ONE,
    ));
    s
}

fn run_off(drag: f64, drip_cos: f64) -> RunOff {
    RunOff {
        drag: Fix128::from_f64(drag),
        drip_cos: Fix128::from_f64(drip_cos),
    }
}

const DT: f64 = 1.0 / 240.0;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 240)
}

fn travel() -> Fix128 {
    Fix128::from_ratio(1, 4)
}

/// oracle: on an incline of angle `theta` the drop ends at `g sin(theta) / drag` down the
/// slope, within the f32 of the field, and stays on the plane, for every step: the drag is
/// implicit, so the fixed point of `v ← (v + g_t dt) / (1 + c dt)` is `g_t / c` whatever `dt`
/// is (swept over 1/60, 1/240 and 1/960 s, the same 10 s of sim time each)
#[test]
fn on_an_incline_the_speed_reaches_the_closed_form() {
    let theta = 30.0_f64.to_radians();
    let (ny, nx) = (theta.cos() as f32, theta.sin() as f32);
    let plane = ClosureSdf::new(move |x, y, _| nx * x + ny * y, move |_, _, _| (nx, ny, 0.0));
    let targets = [LandingTarget::Field(&plane)];
    let drag = 2.0;
    let closed = G * theta.sin() / drag;
    for per_second in [60, 240, 960] {
        let step = Fix128::from_ratio(1, per_second);
        let mut s = one_drop(v(0.0, 1.0, 0.0));
        let mut rng = DeterministicRng::new(1);
        let mut landed = 0;
        for _ in 0..10 * per_second {
            landed += s
                .step_with_run_off(step, &mut rng, &targets, travel(), run_off(drag, 0.9))
                .landed
                .len();
        }
        assert_eq!(landed, 1, "1/{per_second} s");
        assert_eq!(s.particle_on_surface(0), Some(0), "1/{per_second} s");
        let p = &s.particles[0];
        let speed = f(p.velocity.length());
        assert!(
            (speed - closed).abs() < closed * 1e-3,
            "1/{per_second} s: {speed} against {closed}"
        );
        let off = f64::from(nx) * f(p.position.x) + f64::from(ny) * f(p.position.y);
        assert!(off.abs() < 1e-2, "1/{per_second} s: {off} off the plane");
        assert!(f(p.velocity.x) > 0.0, "down the slope");
        assert!(f(p.velocity.y) < 0.0, "down the slope");
    }
}

/// A drop that ran off a surface, died in another step and came back in its slot is a free
/// drop: the surface it was on is not carried into the new life, so it falls (gravity, no
/// drag) and nothing drips. Both ways back: revived by hand through the public particles, and
/// reused by an emitter for a new particle.
#[test]
fn a_drop_that_died_and_came_back_is_free() {
    let floor = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    let targets = [LandingTarget::Field(&floor)];
    for by_emitter in [false, true] {
        let mut s = one_drop(v(0.0, 0.05, 0.0));
        s.particles[0].lifetime = Fix128::from_f64(0.5);
        let mut rng = DeterministicRng::new(3);
        // it lands and runs (the floor is flat, so it rests)
        for _ in 0..60 {
            s.step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(2.0, 0.9));
        }
        assert_eq!(s.particle_on_surface(0), Some(0), "on the floor");
        // it dies of age in the plain step
        while s.particles[0].alive {
            s.step(dt(), &mut rng);
        }
        let high = v(0.0, 2.0, 0.0);
        if by_emitter {
            // particles straight up at no speed, from 2 m: with room for one particle
            // only, it reuses the dead slot
            s.max_particles = 1;
            s.add_emitter(ParticleEmitter::new(
                high,
                v(0.0, 1.0, 0.0),
                Fix128::ZERO,
                // two a step, so the rounding of 1/240 s cannot leave the first step empty; the
                // cap of one takes the first
                Fix128::from_int(480),
                Fix128::ZERO,
                Fix128::from_int(10),
                Fix128::ONE,
            ));
        } else {
            let p = &mut s.particles[0];
            p.alive = true;
            p.age = Fix128::ZERO;
            p.position = high;
            p.velocity = Vec3Fix::ZERO;
        }
        assert_eq!(
            s.particle_on_surface(0),
            None,
            "a new life is not on the floor"
        );
        let events = s.step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(2.0, 0.9));
        assert!(
            events.dripped.is_empty(),
            "nothing drips: {:?}",
            events.dripped
        );
        assert!(
            events.landed.is_empty(),
            "nothing lands from 2 m in one step"
        );
        assert_eq!(s.particle_on_surface(0), None);
        assert_eq!(s.particles.len(), 1, "the slot was reused");
        assert!(s.particles[0].alive);
        let fell = f(high.y) - f(s.particles[0].position.y);
        assert!(fell > 0.0, "it falls free (emitter {by_emitter}): {fell}");
    }
}

/// A unit sphere at the origin.
/// One change to a particle system, for the digest teeth.
type Change = Box<dyn Fn(&mut ParticleSystem)>;

fn sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

/// oracle: running round a sphere the drop leaves where `n·ĝ` reaches `drip_cos`, within the
/// angle it moves in one step.
#[test]
fn on_a_sphere_the_drop_drips_at_the_cosine_it_was_told() {
    let ball = sphere();
    let targets = [LandingTarget::Field(&ball)];
    for drip_cos in [0.0, 0.5, 0.9] {
        let mut s = one_drop(v(0.2, 2.0, 0.0));
        let mut rng = DeterministicRng::new(1);
        let mut dripped = None;
        for _ in 0..24_000 {
            let e = s.step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(3.0, drip_cos));
            if !e.dripped.is_empty() {
                assert_eq!(e.dripped, vec![0]);
                dripped = Some(s.particles[0].position);
                break;
            }
        }
        let p = dripped.expect("it dripped");
        let along = -f(p.y) / f(p.length());
        let step = f(s.particles[0].velocity.length()) * DT;
        assert!(
            (along - drip_cos).abs() < step + 2e-3,
            "{along} against {drip_cos}"
        );
        assert_eq!(s.particle_on_surface(0), None);
        // free again: it falls
        let before = f(s.particles[0].velocity.y);
        s.step_with_run_off(dt(), &mut rng, &[], travel(), run_off(3.0, drip_cos));
        assert!(f(s.particles[0].velocity.y) < before);
    }
}

/// oracle: past the end of a shelf nothing holds the drop any more, and it falls free.
#[test]
fn past_an_edge_the_drop_falls_free() {
    // a tilted shelf: the plane reaches only to x = 1
    let (nx, ny) = (0.6_f32, 0.8_f32);
    let shelf = ClosureSdf::new(
        move |x, y, _| {
            if x > 1.0 {
                1000.0
            } else {
                nx * x + ny * y
            }
        },
        move |_, _, _| (nx, ny, 0.0),
    );
    let targets = [LandingTarget::Field(&shelf)];
    let mut s = one_drop(v(0.0, 0.5, 0.0));
    let mut rng = DeterministicRng::new(1);
    let mut dripped_at = None;
    for _ in 0..4800 {
        let e = s.step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(1.0, 0.9));
        if !e.dripped.is_empty() {
            dripped_at = Some(s.particles[0].position);
            break;
        }
    }
    let p = dripped_at.expect("it left the shelf");
    assert!(f(p.x) > 1.0 && f(p.x) < 1.1, "left at x = {}", f(p.x));
}

/// oracle: a query surface holds the drop while the point is occupied, and only then.
#[test]
fn a_query_surface_holds_the_drop_where_it_is_occupied() {
    let up = v(0.0, 1.0, 0.0);
    let ink = move |p: Vec3Fix| (f(p.y) < 0.0 && f(p.x) < 1.0).then_some(up);
    let targets = [LandingTarget::Query(&ink)];
    let mut s = one_drop(v(0.0, 0.5, 0.0));
    s.particles[0].velocity = v(1.0, 0.0, 0.0);
    let mut rng = DeterministicRng::new(1);
    let mut left = false;
    for _ in 0..4800 {
        left |= !s
            .step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(0.0, 2.0))
            .dripped
            .is_empty();
        if left {
            break;
        }
    }
    assert!(left, "it slid out of the occupied part");
    assert!(f(s.particles[0].position.x) >= 1.0);
}

/// A rain scene: an emitter over a sphere, drops landing and dripping.
fn rain(seed: u64) -> (ParticleSystem, DeterministicRng) {
    let mut s = ParticleSystem::new(64, v(0.0, -G, 0.0));
    s.add_emitter(ParticleEmitter::new(
        v(0.0, 3.0, 0.0),
        v(0.0, -1.0, 0.0),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(60),
        Fix128::from_int(1),
        Fix128::from_int(4),
        Fix128::ONE,
    ));
    (s, DeterministicRng::new(seed))
}

fn run_rain(s: &mut ParticleSystem, rng: &mut DeterministicRng, frames: usize) -> (usize, usize) {
    let ball = sphere();
    let targets = [LandingTarget::Field(&ball)];
    let (mut landed, mut dripped) = (0, 0);
    for _ in 0..frames {
        let e = s.step_with_run_off(
            Fix128::from_ratio(1, 60),
            rng,
            &targets,
            travel(),
            run_off(3.0, 0.5),
        );
        landed += e.landed.len();
        dripped += e.dripped.len();
    }
    (landed, dripped)
}

/// oracle: the same calls give the same digest; the scene does land and drip, so the digest
/// compares something; and the emitted count stays under the cap.
#[test]
fn the_same_calls_give_the_same_bits() {
    let (mut a, mut ra) = rain(7);
    let (mut b, mut rb) = rain(7);
    let (landed, dripped) = run_rain(&mut a, &mut ra, 240);
    assert_eq!(run_rain(&mut b, &mut rb, 240), (landed, dripped));
    assert!(landed > 10, "landed {landed}");
    assert!(dripped > 10, "dripped {dripped}");
    assert_eq!(a.state_digest(&ra), b.state_digest(&rb));
    assert!(a.alive_count() <= 64);
    let (mut c, mut rc) = rain(8);
    run_rain(&mut c, &mut rc, 240);
    assert_ne!(a.state_digest(&ra), c.state_digest(&rc), "another seed");
}

/// oracle: each part of the state moves the digest, the one that decides the next step.
#[test]
fn every_part_of_the_state_moves_the_digest() {
    let (mut s, mut rng) = rain(7);
    run_rain(&mut s, &mut rng, 120);
    let base = s.state_digest(&rng);
    let on = (0..s.particles.len())
        .find(|i| s.particle_on_surface(*i).is_some())
        .expect("one on the sphere");
    let free = |s: &ParticleSystem| {
        let mut copy = ParticleSystem::new(s.max_particles, s.gravity);
        copy.damping = s.damping;
        copy.particles = s.particles.clone();
        copy.emitters = s.emitters.clone();
        copy
    };
    // the same particles, none on a surface
    assert_ne!(free(&s).state_digest(&rng), base, "which surface");
    let mut changed: Vec<Change> = vec![
        Box::new(|s| s.gravity.x = Fix128::ONE),
        Box::new(|s| s.damping = Fix128::ONE),
        Box::new(|s| s.max_particles += 1),
        Box::new(|s| s.emitters[0].emission_rate = Fix128::ONE),
        Box::new(move |s| s.particles[on].position.x = Fix128::ONE),
        Box::new(move |s| s.particles[on].velocity.y = Fix128::ONE),
        Box::new(move |s| s.particles[on].age = Fix128::ZERO),
        Box::new(move |s| s.particles[on].lifetime = Fix128::ONE),
        Box::new(move |s| s.particles[on].mass = Fix128::from_int(9)),
        Box::new(move |s| s.particles[on].alive = false),
    ];
    for (i, change) in changed.drain(..).enumerate() {
        let (mut t, mut r) = rain(7);
        run_rain(&mut t, &mut r, 120);
        change(&mut t);
        assert_ne!(t.state_digest(&r), base, "change {i}");
    }
    let mut r = rng.clone();
    r.next_u32();
    assert_ne!(s.state_digest(&r), base, "the generator");
}

const RAIN_DIGEST: &str = "2d012502649dd65e4573b45d33bf1fe49aec65375cfbdbd840b1a36ca2a65373";

/// golden (determinism): the digest of the rain scene after 240 frames, printed so the log
/// shows the bits.
#[test]
fn rain_scene_digest_is_pinned() {
    let (mut s, mut rng) = rain(7);
    run_rain(&mut s, &mut rng, 240);
    let hex: String = s
        .state_digest(&rng)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("rain digest {hex}");
    assert_eq!(hex, RAIN_DIGEST);
}

/// A field read only by its value: its normal from a central difference 2 mm across, which blends the faces at an
/// edge and leans on a tight curve, as a renderer's field gives it
fn by_value(distance: fn(f32, f32, f32) -> f32) -> ClosureSdf {
    ClosureSdf::new(distance, move |x, y, z| {
        let e = 2e-3;
        let g = [
            distance(x + e, y, z) - distance(x - e, y, z),
            distance(x, y + e, z) - distance(x, y - e, z),
            distance(x, y, z + e) - distance(x, y, z - e),
        ];
        let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
        (g[0] / l, g[1] / l, g[2] / l)
    })
}

/// A stroke 20 mm deep with square edges (a box 0.5 × 0.08 × 0.02)
fn square_stroke(x: f32, y: f32, z: f32) -> f32 {
    let q = [x.abs() - 0.25, y.abs() - 0.04, z.abs() - 0.01];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// A curved stroke 20 mm deep: a quarter-round bar of radius 0.15 about the z axis, 40 mm across, square at its ends
fn curved_stroke(x: f32, y: f32, z: f32) -> f32 {
    let r = (x * x + y * y).sqrt() - 0.15;
    let ring = [r.abs() - 0.02, z.abs() - 0.01];
    let o = [ring[0].max(0.0), ring[1].max(0.0)];
    let tube = (o[0] * o[0] + o[1] * o[1]).sqrt() + ring[0].max(ring[1]).min(0.0);
    // only the upper half: cut by the plane y = 0
    tube.max(-y)
}

/// The drops held on `shape` under wind-blown rain, every step they are held: their distance from it
fn held_on(shape: fn(f32, f32, f32) -> f32, max_travel: Fix128) -> Vec<f64> {
    let field = by_value(shape);
    let targets = [LandingTarget::Field(&field)];
    let mut out = Vec::new();
    for i in 0..=80 {
        for (wx, wz) in [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)] {
            let x = -0.3 + 0.0075 * f64::from(i);
            let mut s = one_drop(v(x, 0.25, 0.0));
            s.gravity = v(0.0, -9.81, 0.0);
            s.particles[0].velocity = v(wx, -6.5, wz);
            let mut rng = DeterministicRng::new(1);
            for _ in 0..90 {
                s.step_with_run_off(
                    Fix128::from_ratio(1, 60),
                    &mut rng,
                    &targets,
                    max_travel,
                    run_off(6.0, 0.5),
                );
                if s.particle_on_surface(0).is_some() {
                    let (px, py, pz) = s.particles[0].position.to_f32();
                    out.push(f64::from(shape(px, py, pz)));
                }
            }
        }
    }
    out
}

/// oracle: a held drop is on the surface, step after step, on square edges and on a curved stroke, with a normal
/// that blends faces and leans on the curve. Bound: after each piece of its slide the drop is projected until a
/// projection moves it no more than `max_travel / 2^24`, so it is within about that of the surface, and the field is
/// read in f32 (8 ulps at 0.5): `2 max_travel / 2^24 + 8 eps 0.5`
#[test]
fn held_drops_stay_on_square_and_curved_strokes() {
    let max_travel = Fix128::from_ratio(2, 100);
    let bound = 2.0 * 0.02 / f64::from(1u32 << alice_physics_core::particle::LANDING_BISECTIONS)
        + 8.0 * f64::from(f32::EPSILON) * 0.5;
    for (name, shape) in [
        ("square", square_stroke as fn(f32, f32, f32) -> f32),
        ("curved", curved_stroke),
    ] {
        let held = held_on(shape, max_travel);
        let worst = held.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        println!(
            "{name}: {} held steps, the farthest {worst:e} off, bound {bound:e}",
            held.len()
        );
        assert!(held.len() > 500, "{name}: {} held steps", held.len());
        assert!(
            worst <= bound,
            "{name}: a held drop is {worst} off the surface, the bound is {bound}"
        );
    }
}

/// oracle: a slide does not cut across a curve. Round a horizontal cylinder of radius `r` a drop moving `c = |v| dt`
/// in a step goes in pieces of at most `max_travel`, each along the surface where it starts and put back on it: a
/// piece of length `a` put back on the cylinder covers the arc `r atan(a / r) >= a (1 - (a / r)^2 / 3)`, so the step's
/// arc is at least `c (1 - (max_travel / r)^2 / 3)` and not more than `c`. One straight move of `c` (r 0.05, c about
/// 0.017) covers 3.5 % less; the bound here allows 0.33 %
#[test]
fn a_slide_round_a_curve_does_not_cut_across_it() {
    let r = 0.05_f64;
    let cylinder = ClosureSdf::new(
        |x, y, _| (x * x + y * y).sqrt() - 0.05,
        |x, y, _| {
            let l = (x * x + y * y).sqrt();
            (x / l, y / l, 0.0)
        },
    );
    let targets = [LandingTarget::Field(&cylinder)];
    let max_travel = Fix128::from_ratio(5, 1000);
    let mut s = one_drop(v(0.0, r + 1e-3, 0.0));
    s.particles[0].velocity = v(1.0, -0.1, 0.0);
    let mut rng = DeterministicRng::new(1);
    let rule = run_off(0.1, 0.5);
    let dt = Fix128::from_ratio(1, 60);
    let mut checked = 0;
    let mut worst = 1.0f64;
    let mut was: Option<f64> = None;
    for _ in 0..200 {
        s.step_with_run_off(dt, &mut rng, &targets, max_travel, rule);
        if s.particle_on_surface(0).is_none() {
            if was.is_some() {
                break;
            }
            continue;
        }
        let p = s.particles[0].position;
        let angle = f(p.y).atan2(f(p.x));
        if let Some(a0) = was {
            let arc = r * (a0 - angle).abs();
            let c = f(s.particles[0].velocity.length()) / 60.0;
            let ratio = arc / c;
            worst = worst.min(ratio);
            let least = 1.0 - (0.005 / r).powi(2) / 3.0 - 1e-5;
            assert!(
                ratio >= least && ratio <= 1.0 + 1e-5,
                "step arc {arc} for a move of {c}: {ratio}, at least {least}"
            );
            checked += 1;
        }
        was = Some(angle);
    }
    println!("{checked} steps round the cylinder, the least arc / move {worst}");
    assert!(checked >= 5, "only {checked} steps were checked");
}
