//! Oracles for `particle::ParticleSystem::{add_emitter, apply_force_field}`
//! (previously unwired) and the surrounding emitter / integrator contract.
//!
//! Closed forms: the semi-implicit recursion `v' = (v + g dt) d`, `x' = x + v' dt`
//! of `step`; emission count `floor(rate t)` (dyadic rate and dt, so exact);
//! `apply_force_field`: `dv = F(x, v) / m * (1/60)` with the force laws of
//! `ForceField` written out from their doc comments.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::force::ForceField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::particle::{Particle, ParticleEmitter, ParticleSystem};
use alice_physics::rng::DeterministicRng;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn v3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn fx(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn emitter(
    rate: i64,
    lifetime: Fix128,
    dir: Vec3Fix,
    speed: i64,
    spread: Fix128,
) -> ParticleEmitter {
    ParticleEmitter::new(
        Vec3Fix::from_int(1, 2, 3),
        dir,
        spread,
        Fix128::from_int(rate),
        Fix128::from_int(speed),
        lifetime,
        Fix128::ONE,
    )
}

const DT: (i64, i64) = (1, 16);

fn system(max: usize, gravity: Vec3Fix, damping: Fix128) -> ParticleSystem {
    let mut ps = ParticleSystem::new(max, gravity);
    ps.damping = damping;
    ps
}

// -------------------------------------------------------------- emitters

#[test]
fn add_emitter_returns_consecutive_indices_and_stores_in_order() {
    let mut ps = system(10, Vec3Fix::ZERO, Fix128::ONE);
    for k in 0..4 {
        let idx = ps.add_emitter(emitter(
            k + 1,
            Fix128::ONE,
            Vec3Fix::UNIT_Y,
            1,
            Fix128::ZERO,
        ));
        assert_eq!(idx, k as usize);
        assert_eq!(ps.emitters.len(), k as usize + 1);
        assert_eq!(ps.emitters[idx].emission_rate, Fix128::from_int(k + 1));
    }
}

#[test]
fn emitter_normalises_its_direction() {
    let e = emitter(1, Fix128::ONE, Vec3Fix::from_int(0, 3, 4), 1, Fix128::ZERO);
    let d = v3(e.direction);
    assert!(
        rel(d[1], 0.6) < 1e-12 && rel(d[2], 0.8) < 1e-12 && d[0] == 0.0,
        "{d:?}"
    );
}

#[test]
fn cumulative_emission_is_floor_of_rate_times_time() {
    // rate 20 /s, dt = 1/16: 1.25 per step; after n steps floor(1.25 n) particles
    let mut ps = system(1000, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        20,
        Fix128::from_int(100),
        Vec3Fix::UNIT_Y,
        1,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(1);
    for n in 1..=32 {
        ps.step(r(DT.0, DT.1), &mut rng);
        assert_eq!(ps.alive_count(), (5 * n) / 4, "after {n} steps");
    }
}

#[test]
fn two_emitters_add_their_rates() {
    let mut ps = system(1000, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        32,
        Fix128::from_int(100),
        Vec3Fix::UNIT_Y,
        1,
        Fix128::ZERO,
    ));
    ps.add_emitter(emitter(
        16,
        Fix128::from_int(100),
        Vec3Fix::UNIT_X,
        1,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(1);
    for _ in 0..4 {
        ps.step(r(DT.0, DT.1), &mut rng);
    }
    assert_eq!(ps.alive_count(), 4 * (2 + 1));
}

#[test]
fn zero_dt_and_zero_rate_emit_nothing() {
    let mut ps = system(10, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(100, Fix128::ONE, Vec3Fix::UNIT_Y, 1, Fix128::ZERO));
    let mut rng = DeterministicRng::new(1);
    ps.step(Fix128::ZERO, &mut rng);
    assert!(ps.particles.is_empty());
    let mut quiet = system(10, Vec3Fix::ZERO, Fix128::ONE);
    quiet.add_emitter(emitter(0, Fix128::ONE, Vec3Fix::UNIT_Y, 1, Fix128::ZERO));
    quiet.step(r(DT.0, DT.1), &mut rng);
    assert!(quiet.particles.is_empty());
}

// ------------------------------------------------------------ integrator

#[test]
fn single_particle_follows_the_semi_implicit_recursion() {
    // one particle per step: rate 16, dt 1/16. Track particle 0 (emitted at step 1).
    let g = Vec3Fix::from_int(0, -8, 0);
    let damping = r(15, 16);
    let mut ps = system(100, g, damping);
    ps.add_emitter(emitter(
        16,
        Fix128::from_int(100),
        Vec3Fix::from_int(0, 3, 4),
        5,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(7);
    let dt = 1.0 / 16.0;
    let mut v = [0.0, 3.0, 4.0]; // direction (0, .6, .8) * 5
    let mut x = [1.0, 2.0, 3.0];
    for step in 1..=12 {
        ps.step(r(DT.0, DT.1), &mut rng);
        v[1] += -8.0 * dt;
        for c in &mut v {
            *c *= 15.0 / 16.0;
        }
        for i in 0..3 {
            x[i] += v[i] * dt;
        }
        let p = &ps.particles[0];
        let (pv, px) = (v3(p.velocity), v3(p.position));
        for i in 0..3 {
            assert!(
                (pv[i] - v[i]).abs() < 1e-12,
                "step {step} v[{i}]: {} vs {}",
                pv[i],
                v[i]
            );
            assert!(
                (px[i] - x[i]).abs() < 1e-12,
                "step {step} x[{i}]: {} vs {}",
                px[i],
                x[i]
            );
        }
        assert_eq!(p.age, Fix128::from_ratio(step, 16));
    }
}

#[test]
fn particle_dies_exactly_when_age_reaches_its_lifetime() {
    // lifetime 1/2 = 8 steps of 1/16: alive after 7 updates, dead after the 8th
    let mut ps = system(100, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(16, r(1, 2), Vec3Fix::UNIT_Y, 1, Fix128::ZERO));
    let mut rng = DeterministicRng::new(1);
    for n in 1..=8 {
        ps.step(r(DT.0, DT.1), &mut rng);
        let first_alive = ps.particles[0].alive;
        assert_eq!(first_alive, n < 8, "after {n} steps");
    }
}

#[test]
fn dead_particles_are_frozen() {
    let mut ps = system(100, Vec3Fix::from_int(0, -8, 0), Fix128::ONE);
    let mut dead = Particle::new(
        Vec3Fix::from_int(1, 1, 1),
        Vec3Fix::from_int(2, 0, 0),
        Fix128::ONE,
        Fix128::ONE,
    );
    dead.alive = false;
    ps.particles.push(dead.clone());
    let mut rng = DeterministicRng::new(1);
    ps.step(r(DT.0, DT.1), &mut rng);
    assert_eq!(ps.particles[0].position, dead.position);
    assert_eq!(ps.particles[0].velocity, dead.velocity);
    assert_eq!(ps.alive_count(), 0);
}

#[test]
fn live_cap_holds_under_a_burst_emitter() {
    // 4 per step, lifetime 2 steps, cap 4: alive never exceeds 4
    let mut ps = system(4, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(64, r(1, 8), Vec3Fix::UNIT_Y, 1, Fix128::ZERO));
    let mut rng = DeterministicRng::new(1);
    for _ in 0..40 {
        ps.step(r(DT.0, DT.1), &mut rng);
        assert!(ps.alive_count() <= 4);
    }
}

#[test]
fn live_cap_holds_with_long_lived_particles_and_the_pool_does_not_grow() {
    let mut ps = system(4, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        64,
        Fix128::from_int(100),
        Vec3Fix::UNIT_Y,
        1,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(1);
    for _ in 0..20 {
        ps.step(r(DT.0, DT.1), &mut rng);
    }
    assert_eq!(ps.alive_count(), 4);
    assert_eq!(ps.particles.len(), 4);
}

#[test]
fn emitted_particles_carry_the_emitter_mass_lifetime_and_position() {
    let mut ps = system(10, Vec3Fix::ZERO, Fix128::ONE);
    let e = ParticleEmitter::new(
        Vec3Fix::from_int(4, 5, 6),
        Vec3Fix::UNIT_X,
        Fix128::ZERO,
        Fix128::from_int(16),
        Fix128::from_int(2),
        Fix128::from_int(7),
        Fix128::from_int(3),
    );
    ps.add_emitter(e);
    let mut rng = DeterministicRng::new(1);
    ps.step(r(DT.0, DT.1), &mut rng);
    let p = &ps.particles[0];
    assert_eq!(p.mass, Fix128::from_int(3));
    assert_eq!(p.lifetime, Fix128::from_int(7));
    // emitted at (4,5,6), moved by v dt = (2, 0, 0)/16 during the same step
    assert_eq!(
        p.position,
        Vec3Fix::new(
            Fix128::from_int(4) + r(1, 8),
            Fix128::from_int(5),
            Fix128::from_int(6)
        )
    );
}

#[test]
fn zero_spread_does_not_consume_random_numbers() {
    let mut ps = system(100, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        64,
        Fix128::from_int(10),
        Vec3Fix::UNIT_Y,
        1,
        Fix128::ZERO,
    ));
    let mut used = DeterministicRng::new(77);
    ps.step(r(DT.0, DT.1), &mut used);
    let mut fresh = DeterministicRng::new(77);
    assert_eq!(
        used.next_u64(),
        fresh.next_u64(),
        "no draws for a focused beam"
    );
    // a spread emitter does draw
    let mut ps2 = system(100, Vec3Fix::ZERO, Fix128::ONE);
    ps2.add_emitter(emitter(
        64,
        Fix128::from_int(10),
        Vec3Fix::UNIT_Y,
        1,
        r(1, 1),
    ));
    let mut used2 = DeterministicRng::new(77);
    ps2.step(r(DT.0, DT.1), &mut used2);
    let mut fresh2 = DeterministicRng::new(77);
    assert_ne!(used2.next_u64(), fresh2.next_u64());
}

/// The live cap is never exceeded even when dead slots exist (before the fix a dead slot was
/// recycled at `alive >= max_particles`: 4 alive + 1 dead with cap 4 ended with 5 alive).
#[test]
fn live_count_never_exceeds_the_cap_when_dead_slots_exist() {
    let mut ps = system(4, Vec3Fix::ZERO, Fix128::ONE);
    for _ in 0..4 {
        ps.particles.push(Particle::new(
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(100),
            Fix128::ONE,
        ));
    }
    let mut dead = Particle::new(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        Fix128::ONE,
    );
    dead.alive = false;
    ps.particles.push(dead);
    ps.add_emitter(emitter(
        16,
        Fix128::from_int(100),
        Vec3Fix::UNIT_Y,
        1,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(1);
    ps.step(r(DT.0, DT.1), &mut rng);
    assert!(ps.alive_count() <= 4, "alive {} > cap 4", ps.alive_count());
}

/// Dead slots are reused: when the pool is not full, a new
/// particle is pushed even though dead slots exist, so a steady emitter below the
/// cap grows `particles` without bound (a dead entry per emission, iterated every step).
#[test]
fn steady_emission_below_the_cap_does_not_grow_the_pool() {
    let mut ps = system(64, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(16, r(1, 8), Vec3Fix::UNIT_Y, 1, Fix128::ZERO)); // 1 per step, lives 2 steps
    let mut rng = DeterministicRng::new(1);
    for _ in 0..200 {
        ps.step(r(DT.0, DT.1), &mut rng);
    }
    assert!(ps.alive_count() <= 3);
    assert!(
        ps.particles.len() <= 4,
        "pool length {}",
        ps.particles.len()
    );
}

// ----------------------------------------------------------------- spread

#[test]
fn zero_spread_emits_exactly_along_the_direction_at_the_initial_speed() {
    let mut ps = system(100, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        64,
        Fix128::from_int(10),
        Vec3Fix::from_int(3, 0, 4),
        10,
        Fix128::ZERO,
    ));
    let mut rng = DeterministicRng::new(5);
    ps.step(r(DT.0, DT.1), &mut rng);
    assert_eq!(ps.alive_count(), 4);
    for p in &ps.particles {
        let v = v3(p.velocity);
        assert!(
            (v[0] - 6.0).abs() < 1e-12 && v[1].abs() < 1e-12 && (v[2] - 8.0).abs() < 1e-12,
            "{v:?}"
        );
    }
}

fn max_deflection(spread: Fix128, n: usize) -> (f64, f64) {
    let mut ps = system(n + 8, Vec3Fix::ZERO, Fix128::ONE);
    ps.add_emitter(emitter(
        16 * n as i64,
        Fix128::from_int(10),
        Vec3Fix::UNIT_Y,
        1,
        spread,
    ));
    let mut rng = DeterministicRng::new(2024);
    ps.step(r(DT.0, DT.1), &mut rng);
    let mut max_angle = 0.0f64;
    let mut speed_dev = 0.0f64;
    for p in &ps.particles {
        let v = v3(p.velocity);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        speed_dev = speed_dev.max((len - 1.0).abs());
        max_angle = max_angle.max((v[1] / len).clamp(-1.0, 1.0).acos());
    }
    (max_angle, speed_dev)
}

/// `spread_angle` is the full apex angle of the emission cone (doc: "0 = focused beam,
/// PI = hemisphere": a full apex angle of PI is a hemisphere), so the largest deflection
/// from the direction is `spread / 2`: no particle goes beyond it and the cone is filled up to it.
#[test]
fn spread_is_the_full_cone_angle_and_keeps_the_speed() {
    use std::f64::consts::PI;
    for (spread, bound) in [
        (Fix128::from_f64(PI / 6.0), PI / 12.0),
        (Fix128::from_f64(PI / 4.0), PI / 8.0),
        (Fix128::HALF_PI, PI / 4.0),
        (Fix128::PI, PI / 2.0),
        (Fix128::from_int(5), PI / 2.0), // beyond PI the cone stays a hemisphere
    ] {
        let (angle, speed_dev) = max_deflection(spread, 4000);
        assert!(speed_dev < 1e-9, "speed is initial_speed: {speed_dev}");
        assert!(angle <= bound + 1e-6, "max deflection {angle} > {bound}");
        assert!(angle > bound - 0.05, "cone not filled: {angle} vs {bound}");
    }
}

// ------------------------------------------------------- force fields

fn dv(field: &ForceField, pos: Vec3Fix, vel: Vec3Fix, mass: Fix128) -> [f64; 3] {
    let mut ps = system(10, Vec3Fix::ZERO, Fix128::ONE);
    ps.particles
        .push(Particle::new(pos, vel, Fix128::from_int(10), mass));
    ps.apply_force_field(field);
    let p = &ps.particles[0];
    let d = p.velocity - vel;
    v3(d)
}

fn close(got: [f64; 3], want: [f64; 3], what: &str) {
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() < 1e-12 * (1.0 + want[i].abs()),
            "{what}[{i}]: {} vs {}",
            got[i],
            want[i]
        );
    }
}

const IMPULSE_DT: f64 = 1.0 / 60.0;

#[test]
fn directional_field_is_force_over_mass_times_one_sixtieth() {
    let f = ForceField::Directional {
        direction: Vec3Fix::from_int(0, 0, -1),
        strength: Fix128::from_int(12),
    };
    close(
        dv(&f, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(3)),
        [0.0, 0.0, -12.0 / 3.0 * IMPULSE_DT],
        "directional",
    );
}

#[test]
fn point_field_is_inverse_square_clamped_and_signed_by_repulsion() {
    let centre = Vec3Fix::from_int(0, 0, 0);
    let attract = ForceField::Point {
        center: centre,
        strength: Fix128::from_int(8),
        repulsive: false,
        max_force: Fix128::from_int(100),
    };
    // particle at (2,0,0): dist 2, |F| = 8 / 4 = 2 toward the centre (-x)
    close(
        dv(
            &attract,
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ONE,
        ),
        [-2.0 * IMPULSE_DT, 0.0, 0.0],
        "attract",
    );
    let repel = ForceField::Point {
        center: centre,
        strength: Fix128::from_int(8),
        repulsive: true,
        max_force: Fix128::from_int(100),
    };
    close(
        dv(
            &repel,
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ONE,
        ),
        [2.0 * IMPULSE_DT, 0.0, 0.0],
        "repel",
    );
    // clamp: at dist 0.5 the raw magnitude 32 exceeds max_force 5
    let clamp = ForceField::Point {
        center: centre,
        strength: Fix128::from_int(8),
        repulsive: false,
        max_force: Fix128::from_int(5),
    };
    close(
        dv(&clamp, fx(0.0, 0.5, 0.0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0, -5.0 * IMPULSE_DT, 0.0],
        "clamped",
    );
    // singular point: no force
    close(
        dv(&attract, centre, Vec3Fix::ZERO, Fix128::ONE),
        [0.0; 3],
        "at centre",
    );
}

#[test]
fn drag_opposes_velocity_and_scales_with_inverse_mass() {
    let f = ForceField::Drag {
        coefficient: Fix128::from_int(6),
    };
    close(
        dv(
            &f,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(2, -4, 1),
            Fix128::from_int(2),
        ),
        [
            -6.0 * 2.0 / 2.0 * IMPULSE_DT,
            6.0 * 4.0 / 2.0 * IMPULSE_DT,
            -6.0 * 1.0 / 2.0 * IMPULSE_DT,
        ],
        "drag",
    );
}

#[test]
fn buoyancy_field_acts_only_below_the_surface_with_depth_proportional_lift() {
    let f = ForceField::Buoyancy {
        surface_y: Fix128::from_int(5),
        density: Fix128::from_int(10),
        drag: Fix128::from_int(2),
    };
    // depth 3, v = (1, 0, 0): F = (0, 10*3, 0) - 2 v
    close(
        dv(
            &f,
            Vec3Fix::from_int(0, 2, 0),
            Vec3Fix::from_int(1, 0, 0),
            Fix128::ONE,
        ),
        [-2.0 * IMPULSE_DT, 30.0 * IMPULSE_DT, 0.0],
        "below",
    );
    close(
        dv(
            &f,
            Vec3Fix::from_int(0, 6, 0),
            Vec3Fix::from_int(1, 0, 0),
            Fix128::ONE,
        ),
        [0.0; 3],
        "above",
    );
    close(
        dv(
            &f,
            Vec3Fix::from_int(0, 5, 0),
            Vec3Fix::from_int(1, 0, 0),
            Fix128::ONE,
        ),
        [0.0; 3],
        "on the surface (strictly below only)",
    );
}

#[test]
fn vortex_force_is_tangential_with_inverse_distance_falloff() {
    let f = ForceField::Vortex {
        center: Vec3Fix::ZERO,
        axis: Vec3Fix::UNIT_Y,
        strength: Fix128::from_int(9),
        falloff_radius: Fix128::from_int(2),
    };
    // axis x diff: y x (1,0,0) = (0, 0, -1): inside the falloff radius full strength
    close(
        dv(&f, Vec3Fix::from_int(1, 0, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0, 0.0, -9.0 * IMPULSE_DT],
        "inside",
    );
    // at distance 4 the strength is scaled by falloff_radius / dist = 1/2
    close(
        dv(&f, Vec3Fix::from_int(4, 0, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0, 0.0, -4.5 * IMPULSE_DT],
        "outside",
    );
    close(
        dv(&f, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ONE),
        [0.0; 3],
        "on the axis point",
    );
}

#[test]
fn explosion_force_follows_strength_times_one_minus_d_over_r_to_the_power() {
    let f = ForceField::Explosion {
        center: Vec3Fix::ZERO,
        strength: Fix128::from_int(20),
        radius: Fix128::from_int(4),
        falloff_power: Fix128::from_int(2),
    };
    // d = 1, direction +y: 20 * (1 - 1/4)^2
    close(
        dv(&f, Vec3Fix::from_int(0, 1, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0, 20.0 * 0.5625 * IMPULSE_DT, 0.0],
        "inside",
    );
    close(
        dv(&f, Vec3Fix::from_int(0, 5, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0; 3],
        "outside",
    );
    close(
        dv(&f, Vec3Fix::from_int(0, 4, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0; 3],
        "on the radius",
    );
    // linear falloff
    let lin = ForceField::Explosion {
        center: Vec3Fix::ZERO,
        strength: Fix128::from_int(20),
        radius: Fix128::from_int(4),
        falloff_power: Fix128::ONE,
    };
    close(
        dv(
            &lin,
            Vec3Fix::from_int(0, 0, -2),
            Vec3Fix::ZERO,
            Fix128::ONE,
        ),
        [0.0, 0.0, -10.0 * IMPULSE_DT],
        "linear",
    );
}

#[test]
fn magnetic_field_points_along_the_moment_with_inverse_cube_strength() {
    let f = ForceField::Magnetic {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::from_int(0, 0, 5),
        strength: Fix128::from_int(16),
    };
    // r = 2: 16 / 8 = 2 along +z (moment is normalised)
    close(
        dv(&f, Vec3Fix::from_int(2, 0, 0), Vec3Fix::ZERO, Fix128::ONE),
        [0.0, 0.0, 2.0 * IMPULSE_DT],
        "magnetic",
    );
    close(
        dv(&f, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ONE),
        [0.0; 3],
        "at the dipole",
    );
}

#[test]
fn dead_and_massless_particles_ignore_force_fields() {
    let f = ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(60),
    };
    let mut ps = system(10, Vec3Fix::ZERO, Fix128::ONE);
    let mut dead = Particle::new(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
    dead.alive = false;
    ps.particles.push(dead);
    ps.particles.push(Particle::new(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ZERO,
    ));
    ps.particles.push(Particle::new(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
    ));
    ps.apply_force_field(&f);
    assert_eq!(ps.particles[0].velocity, Vec3Fix::ZERO);
    assert_eq!(ps.particles[1].velocity, Vec3Fix::ZERO);
    close(
        v3(ps.particles[2].velocity),
        [60.0 * IMPULSE_DT, 0.0, 0.0],
        "live",
    );
}

#[test]
fn emission_is_reproducible_for_a_given_seed_and_differs_across_seeds() {
    let run = |seed: u64| {
        let mut ps = system(100, Vec3Fix::from_int(0, -8, 0), r(15, 16));
        ps.add_emitter(emitter(
            64,
            Fix128::from_int(10),
            Vec3Fix::UNIT_Y,
            3,
            r(1, 1),
        ));
        let mut rng = DeterministicRng::new(seed);
        for _ in 0..8 {
            ps.step(r(DT.0, DT.1), &mut rng);
        }
        ps.particles
            .iter()
            .map(|p| (p.position, p.velocity))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}

#[test]
fn a_dead_slot_is_revived_with_the_emitter_state_before_the_pool_grows() {
    let mut ps = system(4, Vec3Fix::ZERO, Fix128::ONE);
    ps.particles.push(Particle::new(
        Vec3Fix::from_int(9, 9, 9),
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        Fix128::ONE,
    ));
    let mut dead = Particle::new(
        Vec3Fix::from_int(7, 7, 7),
        Vec3Fix::from_int(5, 5, 5),
        Fix128::from_int(1),
        Fix128::from_int(9),
    );
    dead.alive = false;
    dead.age = Fix128::from_int(5);
    ps.particles.push(dead);
    let e = ParticleEmitter::new(
        Vec3Fix::from_int(4, 5, 6),
        Vec3Fix::UNIT_X,
        Fix128::ZERO,
        Fix128::from_int(16),
        Fix128::from_int(2),
        Fix128::from_int(7),
        Fix128::from_int(3),
    );
    ps.add_emitter(e);
    let mut rng = DeterministicRng::new(1);
    ps.step(r(DT.0, DT.1), &mut rng);
    assert_eq!(
        ps.particles.len(),
        2,
        "the dead slot is reused, the pool does not grow"
    );
    let p = &ps.particles[1];
    assert!(p.alive);
    assert_eq!(p.mass, Fix128::from_int(3));
    assert_eq!(p.lifetime, Fix128::from_int(7));
    assert_eq!(p.velocity, Vec3Fix::from_int(2, 0, 0));
    assert_eq!(
        p.age,
        r(1, 16),
        "age restarts at 0 and is advanced once by the same step"
    );
    assert_eq!(
        p.position,
        Vec3Fix::new(
            Fix128::from_int(4) + r(1, 8),
            Fix128::from_int(5),
            Fix128::from_int(6)
        )
    );
}
