//! Oracles for `fracture::FractureModifier::{apply_stress_at, stress_at, active_crack_count}`
//! (`examples/fracture_impact.rs`) and the crack lifecycle they feed.
//!
//! Grid: 5^3 nodes over `[0, 4]^3` (cell 1). Hand-derived closed forms:
//! * splat at a node, radius `R`, amount `A`: node at distance `d < R` gets
//!   `A * t^2 (3 - 2t)` with `t = 1 - d/R` (smoothstep); `R = 1.5` gives centre `A`,
//!   6 face neighbours `A * 7/27`, 12 edge neighbours `t = 1 - sqrt2/1.5`, corners 0
//! * `stress_at` between two nodes on a grid line is their average at the midpoint
//! * seeded crack direction = `normalize(-gz, 0, gx)` of the central-difference gradient
//! * a crack grows `propagation_speed * dt` per update and stops at `max_crack_length`
//! * capsule distance of a point at perpendicular distance `p` from the axis: `p - crack_width`
#![allow(clippy::disallowed_methods)]

use alice_physics::fracture::{Crack, FractureConfig, FractureModifier};
use alice_physics::sim_modifier::PhysicsModifier;

fn modifier(config: FractureConfig) -> FractureModifier {
    FractureModifier::new(config, 5, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
}
fn quiet() -> FractureConfig {
    FractureConfig {
        stress_diffusion: 0.0,
        stress_decay: 0.0,
        ..FractureConfig::default()
    }
}
fn smooth(d: f32, r: f32) -> f32 {
    let t = 1.0 - d / r;
    t * t * (3.0 - 2.0 * t)
}
fn near(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn splat_follows_the_smoothstep_closed_form_on_every_node() {
    let mut m = modifier(quiet());
    m.apply_stress_at(2.0, 2.0, 2.0, 100.0, 1.5);
    for iz in 0..5 {
        for iy in 0..5 {
            for ix in 0..5 {
                let d = ((ix as f32 - 2.0).powi(2)
                    + (iy as f32 - 2.0).powi(2)
                    + (iz as f32 - 2.0).powi(2))
                .sqrt();
                let want = if d < 1.5 { 100.0 * smooth(d, 1.5) } else { 0.0 };
                let got = m.stress_at(ix as f32, iy as f32, iz as f32);
                assert!(
                    near(got, want, 1e-4),
                    "node ({ix},{iy},{iz}) d={d}: {got} vs {want}"
                );
            }
        }
    }
    assert!(near(m.stress_at(3.0, 2.0, 2.0), 100.0 * 7.0 / 27.0, 1e-5));
}

#[test]
fn stress_is_additive_signed_and_interpolated() {
    let mut m = modifier(quiet());
    m.apply_stress_at(2.0, 2.0, 2.0, 40.0, 1.5);
    m.apply_stress_at(2.0, 2.0, 2.0, 60.0, 1.5);
    assert!(near(m.stress_at(2.0, 2.0, 2.0), 100.0, 1e-5));
    m.apply_stress_at(2.0, 2.0, 2.0, -25.0, 1.5);
    assert!(near(m.stress_at(2.0, 2.0, 2.0), 75.0, 1e-5));
    // midpoint between centre (75) and +x neighbour (100 * 7/27 - 25 * 7/27 = 75 * 7/27)
    let nb = 75.0 * 7.0 / 27.0;
    assert!(near(m.stress_at(2.5, 2.0, 2.0), 0.5 * (75.0 + nb), 1e-5));
    // zero radius deposits nothing; points outside the splat read 0
    let mut z = modifier(quiet());
    z.apply_stress_at(2.0, 2.0, 2.0, 100.0, 0.0);
    assert_eq!(z.stress_at(2.0, 2.0, 2.0), 0.0);
    let mut far = modifier(quiet());
    far.apply_stress_at(0.0, 0.0, 0.0, 100.0, 1.5);
    assert_eq!(far.stress_at(4.0, 4.0, 4.0), 0.0);
    assert_eq!(far.stress_at(0.0, 0.0, 0.0), 100.0);
}

#[test]
fn update_diffuses_then_decays_with_the_closed_form() {
    // centre-only stress c: after diffusion (rate*dt = 0.05) centre = c (1 - 6*0.05), face neighbour = 0.05 c
    let cfg = FractureConfig {
        stress_diffusion: 0.1,
        stress_decay: 0.0,
        fracture_toughness: 1e9,
        ..FractureConfig::default()
    };
    let mut m = modifier(cfg);
    m.stress.set(2, 2, 2, 100.0);
    m.update(0.5);
    assert!(near(m.stress_at(2.0, 2.0, 2.0), 70.0, 1e-5));
    assert!(near(m.stress_at(3.0, 2.0, 2.0), 5.0, 1e-5));
    // decay only: value * exp(-rate * dt)
    let cfg = FractureConfig {
        stress_diffusion: 0.0,
        stress_decay: 0.1,
        fracture_toughness: 1e9,
        ..FractureConfig::default()
    };
    let mut m = modifier(cfg);
    m.stress.set(2, 2, 2, 100.0);
    m.update(2.0);
    assert!(near(
        m.stress_at(2.0, 2.0, 2.0),
        100.0 * (-0.2f32).exp(),
        1e-4
    ));
}

#[test]
fn one_hot_node_seeds_one_crack_that_grows_then_stops() {
    let cfg = FractureConfig {
        propagation_speed: 2.0,
        max_crack_length: 1.0,
        ..quiet()
    };
    let mut m = modifier(cfg);
    assert_eq!(m.active_crack_count(), 0);
    m.apply_stress_at(2.0, 2.0, 2.0, 120.0, 1.5); // 31.1 on neighbours, below toughness 50
    m.update(0.25);
    assert_eq!((m.active_crack_count(), m.cracks.len()), (1, 1));
    let c = m.cracks[0];
    assert_eq!(c.start, (2.0, 2.0, 2.0));
    assert!(
        near(c.length, 0.5, 1e-6),
        "speed * dt = 0.5, got {}",
        c.length
    );
    // seed consumed the stress at its node
    assert_eq!(m.stress.get(2, 2, 2), 0.0);
    // symmetric field -> hash direction: unit vector in the XZ plane
    let (dx, dy, dz) = c.direction;
    assert_eq!(dy, 0.0);
    assert!(near((dx * dx + dz * dz).sqrt(), 1.0, 1e-5));
    // the tip is start + direction * length
    assert!(
        near(c.end.0, 2.0 + dx * 0.5, 1e-5)
            && near(c.end.2, 2.0 + dz * 0.5, 1e-5)
            && c.end.1 == 2.0
    );
    // second update reaches max length exactly and deactivates; a third changes nothing
    m.update(0.25);
    assert_eq!((m.active_crack_count(), m.cracks.len()), (0, 1));
    assert_eq!(m.cracks[0].length, 1.0);
    let end = m.cracks[0].end;
    m.update(0.25);
    assert_eq!(m.cracks[0].end, end);
    assert_eq!(m.cracks.len(), 1, "stress was consumed; no re-seeding");
}

#[test]
fn seed_threshold_is_strict_and_per_node() {
    for (amount, expect) in [(49.0f32, 0usize), (50.0, 0), (50.01, 1)] {
        let mut m = modifier(quiet());
        m.stress.set(1, 1, 1, amount);
        m.update(0.0);
        assert_eq!(m.cracks.len(), expect, "amount {amount}");
    }
}

#[test]
fn crack_direction_follows_the_stress_gradient() {
    // centre 100, +x neighbour 20 -> central-difference gradient gx = (20 - 0) / (2 cell) = 10,
    // crack direction = normalize(-gz, 0, gx) = (0, 0, 1)
    let mut m = modifier(quiet());
    m.stress.set(2, 2, 2, 100.0);
    m.stress.set(3, 2, 2, 20.0);
    m.update(0.0);
    assert_eq!(m.cracks.len(), 1);
    let (dx, dy, dz) = m.cracks[0].direction;
    assert!(
        dx.abs() < 1e-6 && dy == 0.0 && near(dz, 1.0, 1e-6),
        "{:?}",
        m.cracks[0].direction
    );
    // gradient along z only -> direction (-1, 0, 0)
    let mut m = modifier(quiet());
    m.stress.set(2, 2, 2, 100.0);
    m.stress.set(2, 2, 3, 20.0);
    m.update(0.0);
    let (dx, dy, dz) = m.cracks[0].direction;
    assert!(
        near(dx, -1.0, 1e-6) && dy == 0.0 && dz.abs() < 1e-6,
        "{:?}",
        m.cracks[0].direction
    );
    // negative slope flips the sign: neighbour at -x is hot
    let mut m = modifier(quiet());
    m.stress.set(2, 2, 2, 100.0);
    m.stress.set(1, 2, 2, 20.0);
    m.update(0.0);
    assert!(near(m.cracks[0].direction.2, -1.0, 1e-6));
}

#[test]
fn seeds_closer_than_ten_crack_widths_are_suppressed() {
    // hot nodes 1 apart along x; iteration order is x first, so (1,2,2) seeds first
    let run = |width: f32| {
        let mut m = modifier(FractureConfig {
            crack_width: width,
            ..quiet()
        });
        m.stress.set(1, 2, 2, 100.0);
        m.stress.set(2, 2, 2, 100.0);
        m.update(0.0);
        m
    };
    let close = run(0.15); // 10 * 0.15 = 1.5 > 1
    assert_eq!(close.cracks.len(), 1);
    assert_eq!(close.cracks[0].start, (1.0, 2.0, 2.0));
    assert_eq!(
        close.stress.get(2, 2, 2),
        100.0,
        "suppressed node keeps its stress"
    );
    let apart = run(0.05); // 0.5 < 1
    assert_eq!(apart.cracks.len(), 2);
    // exactly ten widths apart (1.0 vs 10 * 0.1): the test is strict `<`, so it seeds
    assert_eq!(run(0.1).cracks.len(), 2);
}

#[test]
fn max_cracks_caps_the_total_including_finished_cracks() {
    let cfg = FractureConfig {
        max_cracks: 1,
        propagation_speed: 10.0,
        max_crack_length: 0.5,
        ..quiet()
    };
    let mut m = modifier(cfg);
    m.stress.set(1, 1, 1, 100.0);
    m.update(0.1);
    m.update(0.1);
    assert_eq!(
        (m.active_crack_count(), m.cracks.len()),
        (0, 1),
        "finished crack"
    );
    m.stress.set(3, 3, 3, 100.0);
    m.update(0.1);
    assert_eq!(
        m.cracks.len(),
        1,
        "finished cracks stay in the SDF and occupy their slot"
    );
    assert_eq!(m.stress.get(3, 3, 3), 100.0);
    // with room for two the second node seeds
    let mut m = modifier(FractureConfig {
        max_cracks: 2,
        ..quiet()
    });
    m.stress.set(1, 1, 1, 100.0);
    m.stress.set(3, 3, 3, 100.0);
    m.update(0.0);
    assert_eq!(m.cracks.len(), 2);
}

fn manual_crack(m: &mut FractureModifier, active: bool) {
    m.cracks.push(Crack {
        start: (1.0, 2.0, 2.0),
        end: (3.0, 2.0, 2.0),
        direction: (1.0, 0.0, 0.0),
        length: 2.0,
        active,
    });
}

#[test]
fn active_count_counts_only_growing_cracks() {
    let mut m = modifier(quiet());
    manual_crack(&mut m, true);
    manual_crack(&mut m, false);
    manual_crack(&mut m, true);
    assert_eq!((m.active_crack_count(), m.cracks.len()), (2, 3));
}

#[test]
fn modify_distance_subtracts_the_capsule_only_inside_material() {
    let w = 0.1;
    let mut m = modifier(FractureConfig {
        crack_width: w,
        ..quiet()
    });
    manual_crack(&mut m, false);
    // on the axis: crack_dist = -w -> d = max(orig, w)
    assert!(near(m.modify_distance(2.0, 2.0, 2.0, -1.0), w, 1e-6));
    // off axis by p = 0.3 (within the span): crack_dist = p - w = 0.2 -> max(-1, -0.2) = -0.2
    assert!(near(
        m.modify_distance(2.0, 2.3, 2.0, -1.0),
        -(0.3 - w),
        1e-5
    ));
    // beyond the tip: distance to the end point
    assert!(near(
        m.modify_distance(3.4, 2.0, 2.0, -1.0),
        -(0.4 - w),
        1e-5
    ));
    // far away the crack does not change the field
    assert_eq!(m.modify_distance(2.0, 3.9, 2.0, -0.5), -0.5);
    // outside the material by more than 2 widths: untouched even on the axis
    assert_eq!(m.modify_distance(2.0, 2.0, 2.0, 0.25), 0.25);
    // outside material the max() keeps the larger original distance (the 2w guard never changes the result:
    // -crack_dist <= w < 2w <= orig there)
    assert_eq!(m.modify_distance(2.0, 2.0, 2.0, 0.19), 0.19);
    assert!(near(m.modify_distance(2.0, 2.05, 2.0, -0.5), 0.05, 1e-5));
    // disabled / no cracks / zero-length crack are identities
    m.enabled = false;
    assert_eq!(m.modify_distance(2.0, 2.0, 2.0, -1.0), -1.0);
    assert!(!m.is_active());
    m.enabled = true;
    assert!(m.is_active());
    m.cracks[0].length = 0.0;
    assert_eq!(m.modify_distance(2.0, 2.0, 2.0, -1.0), -1.0);
    assert_eq!(m.name(), "fracture");
    let empty = modifier(quiet());
    assert_eq!(empty.modify_distance(2.0, 2.0, 2.0, -1.0), -1.0);
}

#[test]
fn disabled_modifier_neither_diffuses_nor_seeds() {
    let mut m = modifier(FractureConfig {
        stress_decay: 1.0,
        ..FractureConfig::default()
    });
    m.enabled = false;
    m.stress.set(2, 2, 2, 100.0);
    m.update(1.0);
    assert_eq!(m.stress.get(2, 2, 2), 100.0);
    assert!(m.cracks.is_empty());
}

fn seg_dist(p: (f64, f64, f64), a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let ba = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let pa = (p.0 - a.0, p.1 - a.1, p.2 - a.2);
    let len2 = ba.0 * ba.0 + ba.1 * ba.1 + ba.2 * ba.2;
    let h = ((pa.0 * ba.0 + pa.1 * ba.1 + pa.2 * ba.2) / len2).clamp(0.0, 1.0);
    let d = (pa.0 - ba.0 * h, pa.1 - ba.1 * h, pa.2 - ba.2 * h);
    (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt()
}

#[test]
fn oblique_crack_matches_the_point_to_segment_distance() {
    let w = 0.05f32;
    let mut m = modifier(FractureConfig {
        crack_width: w,
        ..quiet()
    });
    let (a, b) = ((0.5f32, 1.0f32, 1.5f32), (3.0f32, 2.0f32, 3.5f32));
    m.cracks.push(Crack {
        start: a,
        end: b,
        direction: (0.0, 0.0, 1.0),
        length: 3.0,
        active: false,
    });
    let f = |t: (f32, f32, f32)| (f64::from(t.0), f64::from(t.1), f64::from(t.2));
    let mut checked = 0;
    for ix in 0..9 {
        for iy in 0..9 {
            for iz in 0..9 {
                let p = (ix as f32 * 0.5, iy as f32 * 0.5, iz as f32 * 0.5);
                // original deep inside material so max(orig, -crack_dist) = -crack_dist when it exceeds orig
                let orig = -10.0f32;
                let want = (f64::from(w) - seg_dist(f(p), f(a), f(b))).max(f64::from(orig));
                let got = f64::from(m.modify_distance(p.0, p.1, p.2, orig));
                assert!((got - want).abs() < 2e-4, "p={p:?}: {got} vs {want}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 729);
    // a zero-extent crack (start == end, length forced > 0) degenerates to a sphere of radius `width`
    let mut z = modifier(FractureConfig {
        crack_width: w,
        ..quiet()
    });
    z.cracks.push(Crack {
        start: (2.0, 2.0, 2.0),
        end: (2.0, 2.0, 2.0),
        direction: (1.0, 0.0, 0.0),
        length: 1.0,
        active: false,
    });
    assert!(near(
        z.modify_distance(2.0, 2.0, 2.3, -10.0),
        -(0.3 - w),
        1e-4
    ));
}

#[test]
fn small_gradients_still_orient_the_crack_and_tiny_ones_use_the_hash_direction() {
    // gx = 0.002 (neighbour 0.004 / (2 cell) per the central difference): gradient branch -> (0,0,1)
    let mut m = modifier(quiet());
    m.stress.set(2, 2, 2, 100.0);
    m.stress.set(3, 2, 2, 0.004);
    m.update(0.0);
    assert!(
        near(m.cracks[0].direction.2, 1.0, 1e-4),
        "{:?}",
        m.cracks[0].direction
    );
    // gradient 5e-6 < 1e-5: hash direction from the node index
    let mut h = modifier(quiet());
    h.stress.set(2, 2, 2, 100.0);
    h.stress.set(3, 2, 2, 1e-5);
    h.update(0.0);
    let hash = ((2usize * 73_856_093) ^ (2usize * 19_349_663) ^ (2usize * 83_492_791)) as f32;
    let angle = f64::from(hash * 0.0001);
    let (dx, dy, dz) = h.cracks[0].direction;
    assert_eq!(dy, 0.0);
    assert!(
        (f64::from(dx) - angle.cos()).abs() < 5e-3 && (f64::from(dz) - angle.sin()).abs() < 5e-3,
        "{dx} {dz} vs {} {}",
        angle.cos(),
        angle.sin()
    );
}

#[test]
fn growth_clamps_at_max_length_and_the_tip_is_measured_from_the_start() {
    let cfg = FractureConfig {
        propagation_speed: 1.5,
        max_crack_length: 1.0,
        ..quiet()
    };
    let mut m = modifier(cfg);
    m.stress.set(1, 2, 3, 100.0);
    m.stress.set(2, 2, 3, 20.0); // gradient along +x -> direction (0, 0, 1)
    m.update(0.5); // length 0.75
    let c = m.cracks[0];
    assert_eq!(c.start, (1.0, 2.0, 3.0));
    assert!(near(c.length, 0.75, 1e-6) && c.active);
    assert!(near(c.end.2, 3.0 + 0.75 * c.direction.2, 1e-5));
    m.update(0.5); // 1.5 overshoots -> clamped to exactly 1.0
    let c = m.cracks[0];
    assert_eq!(c.length, 1.0, "clamped to max_crack_length");
    assert!(!c.active);
    assert!(
        near(c.end.0, 1.0 + c.direction.0, 1e-5)
            && near(c.end.1, 2.0 + c.direction.1, 1e-5)
            && near(c.end.2, 3.0 + c.direction.2, 1e-5),
        "{c:?}"
    );
}

#[test]
fn cap_applies_inside_one_scan_too() {
    let mut m = modifier(FractureConfig {
        max_cracks: 1,
        ..quiet()
    });
    m.stress.set(1, 1, 1, 100.0);
    m.stress.set(3, 3, 3, 100.0);
    m.update(0.0);
    assert_eq!(m.cracks.len(), 1);
    assert_eq!(m.stress.get(3, 3, 3), 100.0);
}

#[test]
fn tip_of_a_crack_along_x_is_measured_from_the_start_each_update() {
    let cfg = FractureConfig {
        propagation_speed: 1.0,
        max_crack_length: 3.0,
        ..quiet()
    };
    let mut m = modifier(cfg);
    m.stress.set(3, 2, 1, 100.0);
    m.stress.set(3, 2, 2, 20.0); // gradient along +z -> direction (-1, 0, 0)
    m.update(0.5);
    m.update(0.5);
    m.update(0.5);
    let c = m.cracks[0];
    assert!(near(c.direction.0, -1.0, 1e-6), "{:?}", c.direction);
    assert!(near(c.length, 1.5, 1e-6));
    assert!(
        near(c.end.0, 3.0 - 1.5, 1e-5) && near(c.end.1, 2.0, 1e-6) && near(c.end.2, 1.0, 1e-6),
        "{c:?}"
    );
}
