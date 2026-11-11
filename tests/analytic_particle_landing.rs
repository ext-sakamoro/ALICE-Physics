//! Oracles for `ParticleSystem::step_with_landing` (particles such as rain hitting
//! shapes and being re-emitted).
//!
//! - free fall onto a floor: the landing step and time agree with the closed form of
//!   the semi-implicit Euler trajectory, and the landing time is within the derived
//!   discretisation bound of the continuous time `sqrt(2h/g)`
//! - a band thinner than one step's travel is never skipped, for any `dt`
//! - a landing is on the surface and on the drop's path, at faces, edges and corners, with an
//!   exact normal and with a blended one
//! - same seed, same landing sequence (bit-identical)
//! - landed particles are re-emitted from the emitter, so the live count is conserved
//! - with no targets the call is bit-identical to `step`
//! - degenerate inputs: `max_travel <= 0`, bad emitter index, `dt = 0`, sample cap

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::particle::{
    LandingEvent, LandingTarget, Particle, ParticleEmitter, ParticleSystem,
};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};

fn floor_sdf() -> ClosureSdf {
    ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))
}

/// Emitter that never emits on its own (rate 0): only used as the respawn source.
fn respawn_only(position: Vec3Fix) -> ParticleEmitter {
    ParticleEmitter::new(
        position,
        Vec3Fix::from_int(0, -1, 0),
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::from_int(1000),
        Fix128::ONE,
    )
}

/// Drop one particle from rest at height `h` onto the floor `y = 0`; returns the step
/// index (0-based) of the landing and the event.
fn drop_onto_floor(h: i64, g: i64, dt: Fix128, max_travel: Fix128) -> (usize, LandingEvent) {
    let floor = floor_sdf();
    let mut ps = ParticleSystem::new(4, Vec3Fix::from_int(0, -g, 0));
    ps.damping = Fix128::ONE;
    ps.add_emitter(respawn_only(Vec3Fix::from_int(0, 1000, 0)));
    ps.particles.push(Particle::new(
        Vec3Fix::from_int(0, h, 0),
        Vec3Fix::ZERO,
        Fix128::from_int(1000),
        Fix128::ONE,
    ));
    let mut rng = DeterministicRng::new(1);
    let targets = [LandingTarget::Field(&floor)];
    for step in 0..100_000 {
        let events = ps.step_with_landing(dt, &mut rng, &targets, max_travel, 0);
        if let Some(e) = events.first() {
            return (step, *e);
        }
    }
    panic!("particle never landed");
}

/// oracle: closed form of semi-implicit Euler from rest, `v_m = -g m dt`,
/// `y_m = h - g dt^2 m (m + 1) / 2`, with the path of step `m` sampled at `k/n`,
/// `n = ceil(|v_m dt| / max_travel)`. Returns `(step index, k, n, crossing)`: the first sample
/// with `y < 0` and where in the step the straight path of that step meets `y = 0`
/// (`y_prev / (y_prev - y_end)`, independent of the implementation, evaluated in f64).
fn closed_form_landing(h: f64, g: f64, dt: f64, max_travel: f64) -> (usize, usize, usize, f64) {
    let mut m = 1usize;
    loop {
        let mf = m as f64;
        let y_prev = h - g * dt * dt * (mf - 1.0) * mf / 2.0;
        let y_end = h - g * dt * dt * mf * (mf + 1.0) / 2.0;
        if y_end < 0.0 {
            let n = ((g * mf * dt * dt) / max_travel).ceil().max(1.0) as usize;
            for k in 1..=n {
                let y = y_prev + (y_end - y_prev) * (k as f64) / (n as f64);
                if y < 0.0 {
                    return (m - 1, k, n, y_prev / (y_prev - y_end));
                }
            }
            unreachable!("the end sample is below the floor");
        }
        m += 1;
    }
}

#[test]
fn free_fall_landing_matches_the_closed_form_and_sqrt_2h_over_g() {
    // oracle: t* = sqrt(2h/g) = 1 s for h = 5, g = 10.
    //
    // Bound: semi-implicit Euler from rest gives y_m = h + g dt^2/8 - g (t_m + dt/2)^2 / 2,
    // so the node sequence lies on a parabola crossing zero at
    // t_c = sqrt(2h/g + dt^2/4) - dt/2, with t* - dt/2 <= t_c <= t*. The chord between
    // two nodes is below the (concave) parabola, so the path crosses in the same step
    // [t_{m-1}, t_m] that contains t_c; the reported time is where the step's straight path
    // meets the floor, found by halving the piece between the last free sample and the first
    // occupied one `LANDING_BISECTIONS` (24) times: within dt / (n 2^24) of the crossing, so
    // 1e-9 s holds for every dt here. Hence t* - 3dt/2 <= t_hit <= t* + dt. Fix128 rounding
    // (64 fractional bits, a few thousand operations) is below 1e-15 s.
    let (h, g) = (5_i64, 10_i64);
    let t_star = (2.0 * h as f64 / g as f64).sqrt();
    for (num, den) in [(1_i64, 30_i64), (1, 64), (1, 240), (1, 7)] {
        let dt = Fix128::from_ratio(num, den);
        let dt_f = num as f64 / den as f64;
        let max_travel = Fix128::from_ratio(1, 50);
        let (step, e) = drop_onto_floor(h, g, dt, max_travel);
        let t_hit = step as f64 * dt_f + e.time_in_step.to_f64();

        let (m_exp, k_exp, n_exp, crossing) = closed_form_landing(h as f64, g as f64, dt_f, 0.02);
        assert_eq!(step, m_exp, "landing step, dt = {num}/{den}");
        // the crossing lies before the first occupied sample and after the one before it
        assert!(
            crossing <= k_exp as f64 / n_exp as f64
                && crossing >= (k_exp - 1) as f64 / n_exp as f64,
            "crossing {crossing} k {k_exp}/{n_exp}"
        );
        let t_exp = m_exp as f64 * dt_f + dt_f * crossing;
        assert!(
            (t_hit - t_exp).abs() < 1e-9,
            "landing time: got {t_hit}, the step's path meets the floor at {t_exp}, dt = {num}/{den}"
        );
        assert!(
            t_hit >= t_star - 1.5 * dt_f && t_hit <= t_star + dt_f,
            "t_hit {t_hit} outside [t* - 3dt/2, t* + dt] for dt = {num}/{den}"
        );
        // projected onto the surface: |y| <= f32 rounding of a sample within max_travel
        assert!(
            e.position.y.to_f64().abs() < 1e-6,
            "landing y {}",
            e.position.y.to_f64()
        );
        assert_eq!(e.normal, Vec3Fix::UNIT_Y);
        assert_eq!(e.target, 0);
        assert_eq!(e.particle, 0);
    }
}

#[test]
fn landing_time_error_shrinks_with_dt() {
    // first-order integrator: halving dt keeps |t_hit - t*| within the shrinking bound
    let t_star = 1.0;
    let mut prev_bound = f64::INFINITY;
    for den in [16_i64, 32, 64, 128, 256] {
        let dt_f = 1.0 / den as f64;
        let (step, e) = drop_onto_floor(
            5,
            10,
            Fix128::from_ratio(1, den),
            Fix128::from_ratio(1, 200),
        );
        let err = (step as f64 * dt_f + e.time_in_step.to_f64() - t_star).abs();
        let bound = 1.5 * dt_f;
        assert!(
            err <= bound && bound < prev_bound,
            "dt = 1/{den}: error {err} > {bound}"
        );
        prev_bound = bound;
    }
}

/// Band `|y| < eps/2`, outward normal pointing away from the mid-plane.
fn band_sdf(eps: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |_, y, _| y.abs() - eps / 2.0,
        |_, y, _| (0.0, if y >= 0.0 { 1.0 } else { -1.0 }, 0.0),
    )
}

#[test]
fn a_band_thinner_than_one_step_of_travel_is_never_skipped() {
    // oracle: samples are at most max_travel = eps/2 apart along the path, and the band is
    // eps thick along a vertical path, so at least one sample falls strictly inside it
    let eps = Fix128::from_ratio(1, 1000);
    let band = band_sdf(0.001);
    let targets = [LandingTarget::Field(&band)];
    for (num, den) in [(1_i64, 1_i64), (1, 30), (1, 60), (1, 240)] {
        let dt = Fix128::from_ratio(num, den);
        for travel_per_step in [100_i64, 1000, 12_345] {
            for phase in 0..7_i64 {
                let speed = eps * Fix128::from_int(travel_per_step) / dt;
                let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
                ps.damping = Fix128::ONE;
                ps.add_emitter(respawn_only(Vec3Fix::from_int(0, 50, 0)));
                let y0 = Fix128::ONE + eps * Fix128::from_ratio(37 * phase, 100);
                ps.particles.push(Particle::new(
                    Vec3Fix::new(Fix128::ZERO, y0, Fix128::ZERO),
                    Vec3Fix::new(Fix128::ZERO, -speed, Fix128::ZERO),
                    Fix128::from_int(1000),
                    Fix128::ONE,
                ));
                let mut rng = DeterministicRng::new(3);
                let mut landed = None;
                // the band is reached within 1 / (travel * eps) + 1 steps
                for _ in 0..(1000 / travel_per_step + 2) {
                    let ev = ps.step_with_landing(dt, &mut rng, &targets, eps.half(), 0);
                    if let Some(e) = ev.first() {
                        landed = Some(*e);
                        break;
                    }
                    assert!(
                        ps.particles[0].position.y > Fix128::ZERO,
                        "skipped the band: dt {num}/{den}, travel {travel_per_step} eps, phase {phase}"
                    );
                }
                let e = landed.unwrap_or_else(|| {
                    panic!("never landed: dt {num}/{den}, travel {travel_per_step}, phase {phase}")
                });
                assert_eq!(e.normal, Vec3Fix::UNIT_Y, "hit from above");
                let top = 0.0005;
                assert!(
                    (e.position.y.to_f64() - top).abs() < 1e-6,
                    "on the top face"
                );
            }
        }
    }
}

/// `(events, final (position, velocity, alive) per slot, landings per target)`
type RainRun = (Vec<LandingEvent>, Vec<(Vec3Fix, Vec3Fix, bool)>, [usize; 3]);

fn rain_scene(seed: u64) -> RainRun {
    let floor = floor_sdf();
    let ball = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::from_int(0, 2, 0),
        QuatFix::IDENTITY,
    );
    // a non-SDF occupant: the slab 1 < x < 2, 0 < y < 1 (stand-in for cloth particles)
    let slab = |p: Vec3Fix| {
        let inside = p.x > Fix128::ONE
            && p.x < Fix128::from_int(2)
            && p.y > Fix128::ZERO
            && p.y < Fix128::ONE;
        inside.then_some(Vec3Fix::UNIT_Y)
    };
    let mut ps = ParticleSystem::new(48, Vec3Fix::from_int(0, -10, 0));
    ps.add_emitter(ParticleEmitter::new(
        Vec3Fix::from_int(0, 6, 0),
        Vec3Fix::from_int(0, -1, 0),
        Fix128::from_ratio(3, 2),
        Fix128::from_int(300),
        Fix128::from_int(4),
        Fix128::from_int(100),
        Fix128::ONE,
    ));
    let targets = [
        LandingTarget::Field(&floor),
        LandingTarget::Collider(&ball),
        LandingTarget::Query(&slab),
    ];
    let mut rng = DeterministicRng::new(seed);
    let mut all = Vec::new();
    let mut per_target = [0usize; 3];
    for _ in 0..300 {
        let events = ps.step_with_landing(
            Fix128::from_ratio(1, 60),
            &mut rng,
            &targets,
            Fix128::from_ratio(1, 20),
            0,
        );
        for e in &events {
            per_target[e.target] += 1;
            match e.target {
                0 => assert!(e.normal.y > Fix128::ZERO, "floor normal points up"),
                1 => {
                    let out = e.position - Vec3Fix::from_int(0, 2, 0);
                    assert!(
                        out.dot(e.normal) > Fix128::ZERO,
                        "ball normal points outward"
                    );
                }
                _ => assert_eq!(e.normal, Vec3Fix::UNIT_Y),
            }
        }
        all.extend(events);
    }
    let state = ps
        .particles
        .iter()
        .map(|p| (p.position, p.velocity, p.alive))
        .collect();
    (all, state, per_target)
}

#[test]
fn same_seed_gives_a_bit_identical_landing_sequence() {
    let (a, sa, ta) = rain_scene(2024);
    let (b, sb, _) = rain_scene(2024);
    assert!(
        ta.iter().all(|&n| n > 0),
        "every target kind is hit: {ta:?}"
    );
    assert_eq!(a, b);
    assert_eq!(sa, sb);
    let (c, _, _) = rain_scene(2025);
    assert_ne!(
        a, c,
        "a different seed changes the sequence (the test is not vacuous)"
    );
}

#[test]
fn landed_particles_return_to_the_emitter_and_the_live_count_is_conserved() {
    let floor = floor_sdf();
    let source = Vec3Fix::from_int(3, 4, -1);
    let mut ps = ParticleSystem::new(32, Vec3Fix::from_int(0, -10, 0));
    ps.add_emitter(ParticleEmitter::new(
        source,
        Vec3Fix::from_int(0, -1, 0),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(600),
        Fix128::from_int(2),
        Fix128::from_int(1000), // never expires within the test
        Fix128::ONE,
    ));
    let targets = [LandingTarget::Field(&floor)];
    let mut rng = DeterministicRng::new(9);
    let mut total = 0;
    let mut saturated = false;
    for _ in 0..600 {
        let events = ps.step_with_landing(
            Fix128::from_ratio(1, 60),
            &mut rng,
            &targets,
            Fix128::from_ratio(1, 10),
            0,
        );
        if saturated {
            assert_eq!(
                ps.alive_count(),
                32,
                "landing must not change the live count"
            );
        }
        saturated |= ps.alive_count() == 32;
        for e in &events {
            let p = &ps.particles[e.particle];
            assert!(p.alive);
            assert_eq!(p.position, source, "re-emitted at the emitter");
            assert!(p.age.is_zero());
        }
        total += events.len();
    }
    assert!(saturated && total > 100, "landings: {total}");
    assert!(ps.particles.len() <= 32);
}

#[test]
fn without_targets_it_is_bit_identical_to_step() {
    let make = || {
        let mut ps = ParticleSystem::new(64, Vec3Fix::from_int(0, -10, 0));
        ps.add_emitter(ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::from_ratio(1, 3),
            Fix128::from_int(90),
            Fix128::from_int(5),
            Fix128::from_ratio(1, 2),
            Fix128::ONE,
        ));
        ps
    };
    let (mut a, mut b) = (make(), make());
    let (mut ra, mut rb) = (DeterministicRng::new(5), DeterministicRng::new(5));
    for _ in 0..120 {
        a.step(Fix128::from_ratio(1, 60), &mut ra);
        let ev = b.step_with_landing(Fix128::from_ratio(1, 60), &mut rb, &[], Fix128::ONE, 0);
        assert!(ev.is_empty());
    }
    let key = |ps: &ParticleSystem| -> Vec<_> {
        ps.particles
            .iter()
            .map(|p| (p.position, p.velocity, p.age, p.alive))
            .collect()
    };
    assert_eq!(key(&a), key(&b));
}

// ---- degenerate inputs ----

#[test]
#[should_panic(expected = "max_travel must be positive")]
fn zero_max_travel_panics() {
    let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
    ps.add_emitter(respawn_only(Vec3Fix::ZERO));
    let _ = ps.step_with_landing(
        Fix128::ONE,
        &mut DeterministicRng::new(1),
        &[],
        Fix128::ZERO,
        0,
    );
}

#[test]
#[should_panic(expected = "respawn_emitter 1 out of range")]
fn bad_respawn_emitter_panics() {
    let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
    ps.add_emitter(respawn_only(Vec3Fix::ZERO));
    let _ = ps.step_with_landing(
        Fix128::ONE,
        &mut DeterministicRng::new(1),
        &[],
        Fix128::ONE,
        1,
    );
}

#[test]
#[should_panic(expected = "samples exceeds")]
fn a_step_needing_more_than_the_sample_cap_panics() {
    // |v dt| / max_travel = 2^21 > 2^20: panic instead of an unbounded loop
    let floor = floor_sdf();
    let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
    ps.damping = Fix128::ONE;
    ps.add_emitter(respawn_only(Vec3Fix::ZERO));
    ps.particles.push(Particle::new(
        Vec3Fix::from_int(0, 10, 0),
        Vec3Fix::from_int(1 << 21, 0, 0),
        Fix128::from_int(100),
        Fix128::ONE,
    ));
    let _ = ps.step_with_landing(
        Fix128::ONE,
        &mut DeterministicRng::new(1),
        &[LandingTarget::Field(&floor)],
        Fix128::ONE,
        0,
    );
}

#[test]
fn zero_dt_is_a_no_op_and_a_resting_particle_inside_lands_at_the_end_of_the_step() {
    let floor = floor_sdf();
    let mut ps = ParticleSystem::new(4, Vec3Fix::ZERO);
    ps.damping = Fix128::ONE;
    ps.add_emitter(respawn_only(Vec3Fix::from_int(0, 9, 0)));
    let inside = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-1, 4), Fix128::ZERO);
    ps.particles.push(Particle::new(
        inside,
        Vec3Fix::ZERO,
        Fix128::from_int(10),
        Fix128::ONE,
    ));
    let targets = [LandingTarget::Field(&floor)];
    let mut rng = DeterministicRng::new(1);

    // dt = 0: documented early return, nothing moves and nothing lands
    let ev = ps.step_with_landing(Fix128::ZERO, &mut rng, &targets, Fix128::ONE, 0);
    assert!(ev.is_empty());
    assert_eq!(ps.particles[0].position, inside);

    // zero displacement still probes the single (end) sample
    let ev = ps.step_with_landing(Fix128::ONE, &mut rng, &targets, Fix128::ONE, 0);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].time_in_step, Fix128::ONE);
    assert!(
        ev[0].position.y.to_f64().abs() < 1e-6,
        "projected onto y = 0"
    );
    assert_eq!(ps.particles[0].position, Vec3Fix::from_int(0, 9, 0));
}

/// A box of half extents `h` about the origin: its exact distance (outside and in)
fn box_distance(p: [f32; 3], h: [f32; 3]) -> f32 {
    let q = [p[0].abs() - h[0], p[1].abs() - h[1], p[2].abs() - h[2]];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// Its exact gradient: toward the nearest point outside, the nearest face's axis inside
fn box_gradient(p: [f32; 3], h: [f32; 3]) -> (f32, f32, f32) {
    let q = [p[0].abs() - h[0], p[1].abs() - h[1], p[2].abs() - h[2]];
    let s = [p[0].signum(), p[1].signum(), p[2].signum()];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    let l = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    if l > 0.0 {
        return (o[0] / l * s[0], o[1] / l * s[1], o[2] / l * s[2]);
    }
    let a = if q[0] >= q[1] && q[0] >= q[2] {
        0
    } else if q[1] >= q[2] {
        1
    } else {
        2
    };
    let mut n = [0.0f32; 3];
    n[a] = s[a];
    (n[0], n[1], n[2])
}

/// The same box with its normal from a central difference 2 mm across, as a field read only by its
/// value gives it: at an edge or a corner the difference spans two or three faces and the normal blends them
fn stencil_normal(p: [f32; 3], h: [f32; 3]) -> (f32, f32, f32) {
    let e = 2e-3;
    let d = |dx: f32, dy: f32, dz: f32| box_distance([p[0] + dx, p[1] + dy, p[2] + dz], h);
    let g = [
        d(e, 0.0, 0.0) - d(-e, 0.0, 0.0),
        d(0.0, e, 0.0) - d(0.0, -e, 0.0),
        d(0.0, 0.0, e) - d(0.0, 0.0, -e),
    ];
    let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
    (g[0] / l, g[1] / l, g[2] / l)
}

const STROKE: [f32; 3] = [0.25, 0.04, 0.01];

/// Rain slanted by a 2.5 m/s wind (each of four ways) over a stroke-shaped box (tops, sides, edges and corners), one drop at a time, until it
/// lands: each landing point's distance from the box, the point, and its distance from the straight path of the
/// step it landed in (semi-implicit Euler: from where the step began along the velocity after gravity)
fn landings_on_the_stroke(field: &ClosureSdf, max_travel: Fix128) -> Vec<(f32, [f32; 3], f64)> {
    let mut out = Vec::new();
    for i in 0..=240 {
        for j in 0..=24 {
            for (vx, vz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let x = -0.3 + 0.0025 * i as f64;
                let z = -0.03 + 0.0025 * j as f64;
                let mut ps = ParticleSystem::new(4, Vec3Fix::from_f32(0.0, -9.81, 0.0));
                ps.damping = Fix128::ONE;
                ps.add_emitter(respawn_only(Vec3Fix::from_int(0, 1000, 0)));
                ps.particles.push(Particle::new(
                    Vec3Fix::from_f32(x as f32, 0.06, z as f32),
                    Vec3Fix::from_f32(2.5 * vx as f32, -6.5, 2.5 * vz as f32),
                    Fix128::from_int(1000),
                    Fix128::ONE,
                ));
                let mut rng = DeterministicRng::new(1);
                let targets = [LandingTarget::Field(field)];
                let dt = Fix128::from_ratio(1, 60);
                for _ in 0..40 {
                    let from = ps.particles[0].position;
                    let travel = (ps.particles[0].velocity + ps.gravity * dt) * dt;
                    let ev = ps.step_with_landing(dt, &mut rng, &targets, max_travel, 0);
                    if let Some(e) = ev.first() {
                        let (px, py, pz) = e.position.to_f32();
                        let off = off_segment(e.position, from, travel);
                        out.push((box_distance([px, py, pz], STROKE), [px, py, pz], off));
                        break;
                    }
                }
            }
        }
    }
    out
}

/// How far `p` is from the segment `from → from + travel`
fn off_segment(p: Vec3Fix, from: Vec3Fix, travel: Vec3Fix) -> f64 {
    let f = |v: Vec3Fix| (v.x.to_f64(), v.y.to_f64(), v.z.to_f64());
    let (a, d, q) = (f(from), f(travel), f(p));
    let r = [q.0 - a.0, q.1 - a.1, q.2 - a.2];
    let dd = d.0 * d.0 + d.1 * d.1 + d.2 * d.2;
    let t = ((r[0] * d.0 + r[1] * d.1 + r[2] * d.2) / dd).clamp(0.0, 1.0);
    let e = [r[0] - d.0 * t, r[1] - d.1 * t, r[2] - d.2 * t];
    (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
}

/// oracle: a landing is on the surface it hit and on the drop's path where it met it, faces, edges and corners alike, whatever the field's normal is
/// like there (a projection from a sample deep inside can put it on the surface but on another face, off the path).
/// Bound: the crossing is bracketed to `max_travel / 2^LANDING_BISECTIONS` (the occupied end is no
/// deeper), the projection from there moves it by that depth along a unit normal, so the point is within that
/// depth of the surface; the field is read in f32, a few ulps of the coordinates (|p| < 0.4): 8 ulps at 0.5
#[test]
fn a_landing_is_on_the_surface_at_faces_edges_and_corners() {
    use alice_physics_core::particle::LANDING_BISECTIONS;
    let max_travel = Fix128::from_ratio(5, 100);
    let bound = 0.05 / f64::from(1u32 << LANDING_BISECTIONS) + 8.0 * f64::from(f32::EPSILON) * 0.5;
    let exact = ClosureSdf::new(
        |x, y, z| box_distance([x, y, z], STROKE),
        |x, y, z| box_gradient([x, y, z], STROKE),
    );
    let blended = ClosureSdf::new(
        |x, y, z| box_distance([x, y, z], STROKE),
        |x, y, z| stencil_normal([x, y, z], STROKE),
    );
    for (name, field) in [
        ("exact normal", &exact),
        ("2 mm difference normal", &blended),
    ] {
        let got = landings_on_the_stroke(field, max_travel);
        // tops, sides, edges and corners all met: a point within 1 mm of two faces' planes is on an edge
        let near = |p: [f32; 3], a: usize| (p[a].abs() - STROKE[a]).abs() < 1e-3;
        let edges = got
            .iter()
            .filter(|(_, p, _)| (0..3).filter(|&a| near(*p, a)).count() >= 2)
            .count();
        let sides = got
            .iter()
            .filter(|(_, p, _)| near(*p, 0) || near(*p, 2))
            .count();
        let worst = got
            .iter()
            .map(|(d, _, _)| f64::from(d.abs()))
            .fold(0.0, f64::max);
        let off_path = got.iter().map(|(_, _, o)| *o).fold(0.0, f64::max);
        assert!(
            worst <= bound,
            "{name}: a landing is {worst} from the surface, the bound is {bound}"
        );
        assert!(
            off_path <= 2.0 * bound,
            "{name}: a landing is {off_path} off its drop's path, the bound is {}",
            2.0 * bound
        );
        assert!(
            got.len() > 1000 && edges > 20 && sides > 20,
            "{name}: {} landings, {edges} on edges, {sides} on sides",
            got.len()
        );
        println!("{name}: {} landings ({edges} on edges, {sides} on sides), the farthest {worst:e} off the surface and {off_path:e} off the path, bound {bound:e}", got.len());
    }
}

/// The bound has teeth: putting the first occupied sample on the surface in one step, as landing did before the
/// bisection, misses it by far more near an edge, where that sample lies deep and the normal blends two faces
#[test]
fn one_projection_from_a_deep_sample_misses_the_surface_at_an_edge() {
    let bound = 0.05 / f64::from(1u32 << alice_physics_core::particle::LANDING_BISECTIONS)
        + 8.0 * f64::from(f32::EPSILON) * 0.5;
    // 9 mm under the top and 8 mm in from the end, as a sample a 50 mm step leaves can lie: the 2 mm difference
    // there still reads the end's face, the nearest face is the top
    let p = [STROKE[0] - 0.008, STROKE[1] - 0.009, 0.0];
    let d = box_distance(p, STROKE);
    let n = stencil_normal(p, STROKE);
    let on = [p[0] - n.0 * d, p[1] - n.1 * d, p[2] - n.2 * d];
    let miss = f64::from(box_distance(on, STROKE).abs());
    assert!(
        miss > 100.0 * bound,
        "one projection lands {miss} off, the bound is {bound}"
    );
}
