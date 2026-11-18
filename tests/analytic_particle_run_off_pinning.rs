//! Oracles for the retention of a drop on a surface (contact-angle hysteresis, the Furmidge
//! relation) in `ParticleSystem::step_with_run_off`, and for handing a pinned drop's water to
//! the water film.
//!
//! - the retention per unit mass is `γ w (cos θ_r − cos θ_a) / (ρ V)` (Furmidge 1962)
//! - on an incline a drop pins when `g sin α ≤ a_r` and slides when `g sin α > a_r`, on both
//!   sides of the critical angle, at three step lengths
//! - a sliding drop ends at `(g sin α − a_r) / c` down the slope (the retention opposes the
//!   motion like dry friction, the drag is the film's viscous drag)
//! - a drop that slides onto level ground stops within `v₀² / (2 a_r)` and pins
//! - without hysteresis (`a_r = 0`) a drop on any tilt slides, as before
//! - every pinned drop's water goes to the film cell under it: the volume born equals the volume
//!   deposited plus the parts below one quantum each cell still holds plus the water that pinned
//!   off the bed plus what is still in the air, exactly (Q64.64); the film's own ledger counts
//!   every quantum deposited; water off the bed is never put on it
//! - a run on a level face ends with the drop pinned, never at its lifetime

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the closed forms are evaluated in f64 by the test, not by the engine"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::particle::{
    retention_acceleration, FilmHandOff, LandingTarget, Particle, ParticleSystem, RunOff,
};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::ClosureSdf;
use alice_physics_core::water_film::{
    quanta_to_metres, FilmBed, FilmConfig, FilmEdge, FilmScheme, WaterFilm,
};

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

fn run_off(drag: f64, retention: f64) -> RunOff {
    RunOff {
        drag: Fix128::from_f64(drag),
        drip_cos: Fix128::from_f64(0.9),
        retention: Fix128::from_f64(retention),
    }
}

fn travel() -> Fix128 {
    Fix128::from_ratio(1, 4)
}

/// An incline of `sin α = s` falling toward +x, through the origin: its distance and normal
fn incline(s: f64) -> ClosureSdf {
    let c = (1.0 - s * s).sqrt();
    let (nx, ny) = (s as f32, c as f32);
    ClosureSdf::new(move |x, y, _| nx * x + ny * y, move |_, _, _| (nx, ny, 0.0))
}

/// oracle (Furmidge 1962, the retention force of a drop of contact width `w` with advancing and
/// receding angles `θ_a`, `θ_r`): `F = γ w (cos θ_r − cos θ_a)`, so per unit mass of a drop of
/// volume `V` and density `ρ`, `a_r = γ w (cos θ_r − cos θ_a) / (ρ V)`. Water (γ 0.072 N/m,
/// ρ 1000 kg/m³), w 3 mm, V 25 mm³, θ_a 80°, θ_r 50°: `0.072 · 0.003 · (cos 50° − cos 80°) /
/// (1000 · 2.5e-8)`, computed here in f64
#[test]
fn the_retention_is_the_furmidge_force_per_unit_mass() {
    let (a, r) = (80.0_f64.to_radians(), 50.0_f64.to_radians());
    let want = 0.072 * 0.003 * (r.cos() - a.cos()) / (1000.0 * 2.5e-8);
    let got = retention_acceleration(
        Fix128::from_f64(0.072),
        Fix128::from_f64(1000.0),
        Fix128::from_f64(2.5e-8),
        Fix128::from_f64(0.003),
        Fix128::from_f64(r.cos()),
        Fix128::from_f64(a.cos()),
    );
    assert!(
        (f(got) - want).abs() < want * 1e-9,
        "{} against {want}",
        f(got)
    );
    // no hysteresis, no retention; a receding angle above the advancing one is not a surface
    let none = retention_acceleration(
        Fix128::from_f64(0.072),
        Fix128::from_f64(1000.0),
        Fix128::from_f64(2.5e-8),
        Fix128::from_f64(0.003),
        Fix128::from_f64(a.cos()),
        Fix128::from_f64(a.cos()),
    );
    assert_eq!(none, Fix128::ZERO);
    let inverted = retention_acceleration(
        Fix128::from_f64(0.072),
        Fix128::from_f64(1000.0),
        Fix128::from_f64(2.5e-8),
        Fix128::from_f64(0.003),
        Fix128::from_f64(a.cos()),
        Fix128::from_f64(r.cos()),
    );
    assert_eq!(
        inverted,
        Fix128::ZERO,
        "θ_r > θ_a is clamped to no retention"
    );
}

/// oracle: the critical angle is `sin α_c = a_r / g`. With `a_r = 1 m/s²` and `g = 10 m/s²`,
/// `sin α_c = 0.1`. A drop set on the incline at `sin α = 0.098` pins (it is handed off: an event,
/// and it is no longer alive); at `sin α = 0.102` it slides and ends at `(g sin α − a_r) / c`. At
/// step lengths 1/60, 1/240 and 1/960 s. The speed matches to 1e-3 of itself: the field's normal
/// is f32 (1e-7 of sin α, 5e-5 of the driving 0.02 m/s²)
#[test]
fn a_drop_pins_below_the_critical_angle_and_slides_above_it() {
    let (drag, retention) = (2.0, 1.0);
    for per_second in [60_i64, 240, 960] {
        let step = Fix128::from_ratio(1, per_second);
        for (s, pins) in [
            (0.098, true),
            (0.102, false),
            (0.0, true),
            (0.05, true),
            (0.3, false),
        ] {
            let plane = incline(s);
            let targets = [LandingTarget::Field(&plane)];
            // just above the plane, so it lands in the first step at almost no speed
            let mut d = one_drop(v(0.0, 1e-4, 0.0));
            let mut rng = DeterministicRng::new(7);
            let mut pinned = 0;
            for _ in 0..20 * per_second {
                pinned += d
                    .step_with_run_off(step, &mut rng, &targets, travel(), run_off(drag, retention))
                    .pinned
                    .len();
            }
            let alive = d.particles[0].alive;
            if pins {
                assert_eq!(
                    (pinned, alive),
                    (1, false),
                    "sin α {s}, 1/{per_second} s: it must pin once"
                );
            } else {
                assert_eq!(
                    (pinned, alive),
                    (0, true),
                    "sin α {s}, 1/{per_second} s: it must slide"
                );
                let closed = (G * s - retention) / drag;
                let speed = f(d.particles[0].velocity.length());
                assert!(
                    (speed - closed).abs() < closed * 1e-3,
                    "sin α {s}, 1/{per_second} s: {speed} against {closed}"
                );
            }
        }
    }
}

/// oracle: without hysteresis a drop on any tilt slides (the retention is the only thing that
/// can hold it), and its end speed is `g sin α / c`, the closed form of the run-off without
/// retention
#[test]
fn without_hysteresis_a_drop_on_a_tilt_slides() {
    let per_second = 240;
    let step = Fix128::from_ratio(1, per_second);
    for s in [0.001, 0.02, 0.5] {
        let plane = incline(s);
        let targets = [LandingTarget::Field(&plane)];
        let mut d = one_drop(v(0.0, 1e-4, 0.0));
        let mut rng = DeterministicRng::new(7);
        let mut pinned = 0;
        for _ in 0..20 * per_second {
            pinned += d
                .step_with_run_off(step, &mut rng, &targets, travel(), run_off(2.0, 0.0))
                .pinned
                .len();
        }
        assert_eq!(pinned, 0, "sin α {s}");
        let closed = G * s / 2.0;
        let speed = f(d.particles[0].velocity.length());
        assert!(
            (speed - closed).abs() < closed * 1e-3,
            "sin α {s}: {speed} against {closed}"
        );
    }
}

/// oracle: a drop sliding at `v₀` onto level ground decelerates by at least `a_r` (and the
/// drag), so it stops within `v₀² / (2 a_r)` and pins there; without drag the stop is that
/// distance. The drop is set on level ground at x = 0.01 m moving +x at 0.5 m/s, so the bound
/// is 0.125 m past it with a_r = 1
#[test]
fn a_drop_sliding_onto_level_ground_stops_and_pins() {
    let ground = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    let targets = [LandingTarget::Field(&ground)];
    let (v0, retention) = (0.5, 1.0);
    for drag in [0.0, 2.0] {
        let mut d = one_drop(v(0.01, 1e-4, 0.0));
        d.particles[0].velocity = v(v0, 0.0, 0.0);
        let mut rng = DeterministicRng::new(5);
        let step = Fix128::from_ratio(1, 240);
        let mut at = None;
        for _ in 0..2400 {
            let e =
                d.step_with_run_off(step, &mut rng, &targets, travel(), run_off(drag, retention));
            if let Some(p) = e.pinned.first() {
                at = Some(f(p.position.x));
                break;
            }
        }
        let x = at.unwrap_or_else(|| panic!("drag {drag}: it never pinned"));
        // the step it lands in it is still in the air (no retention): `v₀ dt` more
        let stop = 0.01 + v0 * v0 / (2.0 * retention);
        let bound = stop + v0 / 240.0;
        assert!(
            x > 0.01 && x <= bound + 1e-9,
            "drag {drag}: pinned at {x}, bound {bound}"
        );
        if drag == 0.0 {
            // without drag the stop is the closed form itself, to one step's travel
            assert!(
                (x - stop).abs() <= v0 / 240.0,
                "pinned at {x}, closed form {stop}"
            );
        }
    }
}

/// oracle (conservation, exact): rain on a level bed at a hysteresis that pins every drop. Each
/// pinned drop's volume is handed to the film cell under it (`FilmHandOff`): the film takes whole
/// quanta, the part below one quantum stays with that cell for its next drop, and a drop that
/// pinned off the bed keeps its water off it. After the run: the volume born equals deposited +
/// held below a quantum + off the bed + still in the air, bit for bit; the film's ledger counts
/// the quanta deposited; and no drop reached its lifetime
#[test]
fn a_pinned_drop_hands_all_its_water_to_the_film() {
    let cell = Fix128::from_ratio(1, 100);
    let bed = FilmBed::from_height_fn(20, 20, cell, (Fix128::ZERO, Fix128::ZERO), 1, |_, _| 0.0)
        .expect("bed");
    let mut film = WaterFilm::new(
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open),
        bed,
    )
    .expect("film");
    let ground = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    let targets = [LandingTarget::Field(&ground)];
    let volume = Fix128::from_f64(2.5e-8);
    let lifetime = Fix128::from_int(5);
    let mut s = ParticleSystem::new(usize::MAX, v(0.0, -G, 0.0));
    let mut rng = DeterministicRng::new(11);
    let mut hand = FilmHandOff::default();
    let mut born = Fix128::ZERO;
    let mut handed = 0;
    let step = Fix128::from_ratio(1, 60);
    for n in 0..600_u32 {
        if n % 3 == 0 {
            // a drop over the bed, inside and (every 9th) outside it: water off the bed is kept
            let k = f64::from(n / 3);
            let x = if n % 27 == 0 {
                -0.05
            } else {
                0.01 + (k * 0.037) % 0.18
            };
            let z = 0.01 + (k * 0.053) % 0.18;
            s.particles.push(Particle::new(
                v(x, 0.3, z),
                Vec3Fix::ZERO,
                lifetime,
                Fix128::ONE,
            ));
            born = born + volume;
        }
        let e = s.step_with_run_off(step, &mut rng, &targets, travel(), run_off(2.0, 5.0));
        handed += hand.hand_off(&mut film, &e.pinned, volume);
        for p in &s.particles {
            assert!(
                p.age < lifetime,
                "a drop reached its lifetime: water removed without a landing"
            );
        }
    }
    let in_air = s.particles.iter().filter(|p| p.alive).count();
    let air = volume * Fix128::from_int(in_air as i64);
    assert!(handed > 100, "only {handed} drops handed off");
    assert_eq!(
        hand.deposited() + hand.held() + hand.off_bed() + air,
        born,
        "volume born against deposited + held below a quantum + off the bed + in the air"
    );
    assert_eq!(
        film.total(),
        film.ledger().expected(),
        "the film's own ledger"
    );
    // what the film took is its whole quanta: each deposit's volume is its quanta times the cell's area, rounded
    // once in the last bit of Q64.64, so the sum is the quanta's volume to one bit a deposit
    let whole = quanta_to_metres(hand.quanta()) * cell * cell;
    let apart = (hand.deposited() - whole).abs();
    assert!(
        apart <= Fix128::from_raw(0, handed as u64),
        "deposited {:?} against the quanta's volume {whole:?}",
        hand.deposited()
    );
    // the drops off the bed: every 9th born, all pinned by now except those still falling
    assert!(
        hand.off_bed() >= volume * Fix128::from_int(5),
        "the drops off the bed are kept, not lost"
    );
    assert!(
        hand.held() < quanta_to_metres(1) * cell * cell * Fix128::from_int(400),
        "at most a quantum a cell is held"
    );
}

/// A V valley along z with faces of slope `k` (the solid below `y = k |x| − s z`): its crease is
/// the line `x = 0, y = −s z`, falling toward +z when `s > 0`
fn valley(k: f32, s: f32) -> ClosureSdf {
    let n = (1.0 + k * k + s * s).sqrt();
    ClosureSdf::new(
        move |x, y, z| (y - k * x.abs() + s * z) / n,
        move |x, _, _| {
            let sx = if x >= 0.0 { 1.0 } else { -1.0 };
            (-k * sx / n, 1.0 / n, s / n)
        },
    )
}

/// oracle: in a crease (the bottom of a letter's counter, two faces at 45°) the faces drive the
/// drop into the crease, and only the pull along the crease line, `g s / √(1 + s²)`, can move it
/// on. A drop let go on one face slides into the crease; there it pins when that pull is at most
/// the retention, and runs on along the crease when it is more: a level crease (s = 0) and
/// `g s / √(1 + s²) = 0.5 a_r` pin it, `2 a_r` lets it run (`a_r = 1 m/s²`)
#[test]
fn in_a_crease_the_pull_along_it_decides() {
    let retention = 1.0;
    let along = |s: f64| G * s / (1.0 + s * s).sqrt();
    // the s giving a pull `p` along the crease: s = p / √(g² − p²)
    let slope_for = |p: f64| p / (G * G - p * p).sqrt();
    for (s, pins) in [(0.0, true), (slope_for(0.5), true), (slope_for(2.0), false)] {
        assert!(s == 0.0 || (along(s) / retention - if pins { 0.5 } else { 2.0 }).abs() < 1e-12);
        let ground = valley(1.0, s as f32);
        let targets = [LandingTarget::Field(&ground)];
        // released at rest a millimetre above one face, 0.05 m from the crease
        let mut d = one_drop(v(0.05, 0.051, 0.0));
        let mut rng = DeterministicRng::new(3);
        let step = Fix128::from_ratio(1, 240);
        let mut pinned = Vec::new();
        for _ in 0..2400 {
            pinned.extend(
                d.step_with_run_off(step, &mut rng, &targets, travel(), run_off(2.0, retention))
                    .pinned,
            );
        }
        if pins {
            assert_eq!(pinned.len(), 1, "s {s}: it must pin in the crease");
            assert!(
                f(pinned[0].position.x).abs() < 1e-3,
                "s {s}: pinned off the crease at x {}",
                f(pinned[0].position.x)
            );
        } else {
            assert!(
                pinned.is_empty() && d.particles[0].alive,
                "s {s}: it must run along the crease"
            );
            assert!(
                f(d.particles[0].position.z) > 0.1,
                "s {s}: it did not run down the crease"
            );
        }
    }
}
