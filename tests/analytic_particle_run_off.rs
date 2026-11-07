//! Oracles for `ParticleSystem::step_with_run_off` (a drop that lands stays on what it
//! hit, runs down it and drips) and `ParticleSystem::state_digest`.
//!
//! - on an incline the speed reaches the closed form `g sin(theta) / drag` and the drop
//!   stays on the plane
//! - on a sphere the drop drips where `n·ĝ` reaches the cosine it was told
//! - past the edge of a shelf the drop falls free
//! - a query surface holds the drop only where it is occupied
//! - the same calls give the same bits; every part of the state moves the digest; a
//!   digest is pinned

#![allow(
    clippy::disallowed_methods,
    reason = "the closed forms are evaluated in f64 by the test, not by the engine"
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
/// slope, within the f32 of the field, and stays on the plane.
#[test]
fn on_an_incline_the_speed_reaches_the_closed_form() {
    let theta = 30.0_f64.to_radians();
    let (ny, nx) = (theta.cos() as f32, theta.sin() as f32);
    let plane = ClosureSdf::new(move |x, y, _| nx * x + ny * y, move |_, _, _| (nx, ny, 0.0));
    let targets = [LandingTarget::Field(&plane)];
    let mut s = one_drop(v(0.0, 1.0, 0.0));
    let mut rng = DeterministicRng::new(1);
    let drag = 2.0;
    let mut landed = 0;
    for _ in 0..2400 {
        landed += s
            .step_with_run_off(dt(), &mut rng, &targets, travel(), run_off(drag, 0.9))
            .landed
            .len();
    }
    assert_eq!(landed, 1);
    assert_eq!(s.particle_on_surface(0), Some(0));
    let p = &s.particles[0];
    let closed = G * theta.sin() / drag;
    let speed = f(p.velocity.length());
    assert!(
        (speed - closed).abs() < closed * 1e-3,
        "{speed} against {closed}"
    );
    let off = f64::from(nx) * f(p.position.x) + f64::from(ny) * f(p.position.y);
    assert!(off.abs() < 1e-2, "{off} off the plane");
    assert!(f(p.velocity.x) > 0.0, "down the slope");
    assert!(f(p.velocity.y) < 0.0, "down the slope");
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

const RAIN_DIGEST: &str = "1801e895a99e5182e2f2a9f56bf7c5693d7484648310c84cdbb8d6bcc0b5f664";

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
