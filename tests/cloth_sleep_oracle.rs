//! Cloth sleep (exact) and unpin.
//!
//! Sleep is defined so that it can never change a trajectory: a cloth sleeps only once a
//! `step` left its whole state (positions, previous positions, velocities) unchanged, bit
//! for bit, with the same inputs (masses, pins, wind, config, rest tether, rest shape,
//! `dt`). `step` is a function of that state and those inputs, so every further step from
//! there would be the identity too; skipping it is exact. Any change to the state or the
//! inputs (a grab, a poke, a config change, another `dt`) is seen on the next `step`, which
//! then simulates again. So a cloth that slept and was woken follows, bit for bit, the
//! trajectory of a cloth that never slept.
//!
//! The scene is the one the soft lettering on the Extoria site runs: a 7×7 sheet tethered to
//! its rest shape (period 0.5 s, damping ratio 0.42), no gravity, damping 0.985, grabbed at a
//! corner, dragged for half a second and released. It reaches its exact fixed point (rest
//! shape, zero velocity) at frame 480; that frame is pinned below.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::{Cloth, ClothConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

const RES: usize = 7;
/// First frame (0-based, counted from the start of the script) whose `step` changes no bit
/// of the state; the cloth must report asleep right after that step and not before.
///
/// It is the step in which every particle's correction rounds to zero, so it is set by the
/// rounding of `Fix128`'s product: with the product floored (before 4.0), a correction in
/// `(−2⁻⁶⁴, 0)` became `−2⁻⁶⁴` and never vanished, and the sheet's first identity step was
/// frame 480; with the product rounded to nearest, every correction within `±2⁻⁶⁵` is zero,
/// and the sheet reaches it at frame 485 (measured; the change of the product alone gives
/// the same frame, the contact solve plays no part in a free sheet).
const FIRST_IDENTITY_STEP: u32 = 485;

fn sheet() -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        RES,
        RES,
        Fix128::ONE,
    );
    c.config = ClothConfig {
        iterations: 4,
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::from_ratio(985, 1000),
        stretch_compliance: Fix128::from_ratio(1, 100),
        bend_compliance: Fix128::from_ratio(1, 4),
        ..ClothConfig::default()
    };
    // k = m ω², c = 2 ζ m ω with ω = 2π / 0.5 s, ζ = 0.42 (from integer ratios: no f32)
    let omega =
        Fix128::from_ratio(4, 1) * Fix128::from_ratio(314_159_265_358_979, 100_000_000_000_000);
    c.set_rest_tether(omega * omega, Fix128::from_ratio(84, 100) * omega);
    c
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// The particle the hand takes: the corner farthest along +x and +z.
fn corner(c: &Cloth) -> usize {
    let rest = c.rest_positions();
    (0..rest.len())
        .max_by(|&a, &b| (rest[a].x + rest[a].z).cmp(&(rest[b].x + rest[b].z)))
        .unwrap()
}

/// Frame `f` of the script: grab at 10, drag (+0.6, 0, +0.3) over 30 frames, release at 40.
fn drive(c: &mut Cloth, g: usize, g0: Vec3Fix, f: u32) {
    if (10..40).contains(&f) {
        c.pin(g);
        let t = Fix128::from_ratio(i64::from(f) - 10, 30);
        c.positions[g] = g0
            + Vec3Fix::new(
                Fix128::from_ratio(6, 10) * t,
                Fix128::ZERO,
                Fix128::from_ratio(3, 10) * t,
            );
        c.velocities[g] = Vec3Fix::ZERO;
    }
    if f == 40 {
        c.unpin(g);
    }
}

/// A light tap: particle 3 gets an upward velocity.
fn poke(c: &mut Cloth) {
    c.velocities[3] =
        c.velocities[3] + Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO);
}

#[derive(PartialEq, Debug)]
struct State(Vec<Vec3Fix>, Vec<Vec3Fix>, Vec<Vec3Fix>);

fn state(c: &Cloth) -> State {
    State(
        c.positions.clone(),
        c.prev_positions.clone(),
        c.velocities.clone(),
    )
}

/// Run the script for `frames` frames with a poke at `wake`; `sleep` turns sleep on.
/// Returns every frame's state and the number of frames the cloth was asleep before a step.
fn run(frames: u32, wake: Option<u32>, sleep: bool) -> (Vec<State>, u32) {
    let mut c = sheet();
    c.set_sleep(sleep);
    let g = corner(&c);
    let g0 = c.rest_positions()[g];
    let mut out = Vec::new();
    let mut asleep = 0;
    for f in 0..frames {
        drive(&mut c, g, g0, f);
        if Some(f) == wake {
            poke(&mut c);
        }
        if c.is_asleep() {
            asleep += 1;
        }
        c.step(dt());
        out.push(state(&c));
    }
    (out, asleep)
}

#[test]
fn falls_asleep_right_after_the_first_identity_step() {
    let mut c = sheet();
    c.set_sleep(true);
    let g = corner(&c);
    let g0 = c.rest_positions()[g];
    let mut first = None;
    for f in 0..700 {
        drive(&mut c, g, g0, f);
        let before = state(&c);
        c.step(dt());
        let unchanged = before == state(&c);
        if f > 40 && first.is_none() && unchanged {
            first = Some(f);
        }
        // asleep exactly from the first identity step after release (the untouched sheet
        // before the grab is a fixed point too and may sleep there)
        if f > 40 {
            assert_eq!(c.is_asleep(), first.is_some(), "frame {f}");
        }
    }
    assert_eq!(first, Some(FIRST_IDENTITY_STEP));
}

#[test]
fn a_woken_cloth_follows_the_never_slept_trajectory_bit_for_bit() {
    let mut compared = 0;
    let n = FIRST_IDENTITY_STEP;
    for wake in [100, n - 1, n, n + 1, n + 2, 700] {
        let frames = wake + 120;
        let (never, slept0) = run(frames, Some(wake), false);
        let (woken, slept) = run(frames, Some(wake), true);
        assert_eq!(slept0, 0);
        // the wakes after the fixed point must actually have slept, or nothing was tested
        if wake > FIRST_IDENTITY_STEP + 1 {
            assert!(
                slept >= wake - FIRST_IDENTITY_STEP - 1,
                "wake {wake}: asleep {slept} frames"
            );
        }
        for (f, (a, b)) in never.iter().zip(&woken).enumerate() {
            assert_eq!(a, b, "wake {wake}: frame {f} differs");
            compared += 1;
        }
    }
    assert!(compared > 0);
}

#[test]
fn changing_an_input_while_asleep_wakes_the_cloth() {
    let (_, _) = run(1, None, true);
    let mut c = sheet();
    c.set_sleep(true);
    let g = corner(&c);
    let g0 = c.rest_positions()[g];
    for f in 0..=FIRST_IDENTITY_STEP + 1 {
        drive(&mut c, g, g0, f);
        c.step(dt());
    }
    assert!(c.is_asleep());
    // gravity is an input: the next step must simulate and move the sheet
    c.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    assert!(!c.is_asleep());
    let before = state(&c);
    c.step(dt());
    assert_ne!(before, state(&c));
    // a direct write to a position wakes it too
    let mut d = sheet();
    d.set_sleep(true);
    for f in 0..=FIRST_IDENTITY_STEP + 1 {
        drive(&mut d, g, g0, f);
        d.step(dt());
    }
    assert!(d.is_asleep());
    d.positions[5] =
        d.positions[5] + Vec3Fix::new(Fix128::from_ratio(1, 100), Fix128::ZERO, Fix128::ZERO);
    assert!(!d.is_asleep());
    let before = state(&d);
    d.step(dt());
    assert_ne!(before, state(&d));
    // another dt is an input as well: asleep for 1/60 s says nothing about 1/30 s
    let mut e = sheet();
    e.set_sleep(true);
    for f in 0..=FIRST_IDENTITY_STEP + 1 {
        drive(&mut e, g, g0, f);
        e.step(dt());
    }
    assert!(e.is_asleep());
    poke(&mut e);
    let mut never = sheet();
    for f in 0..=FIRST_IDENTITY_STEP + 1 {
        drive(&mut never, g, g0, f);
        never.step(dt());
    }
    poke(&mut never);
    for _ in 0..30 {
        e.step(Fix128::from_ratio(1, 30));
        never.step(Fix128::from_ratio(1, 30));
        assert_eq!(state(&e), state(&never));
    }
}

#[test]
fn sleep_is_off_by_default_and_turning_it_off_forgets_it() {
    let c = sheet();
    assert!(!c.is_asleep());
    let mut d = sheet();
    d.set_sleep(true);
    d.step(dt());
    d.step(dt());
    // the untouched sheet is a fixed point: asleep after its first identity step
    assert!(d.is_asleep());
    d.set_sleep(false);
    assert!(!d.is_asleep());
}

#[test]
fn unpin_restores_the_mass_pin_took() {
    let mut c = sheet();
    let m = c.inv_masses[4];
    assert!(!m.is_zero());
    c.pin(4);
    c.pin(4);
    assert!(c.inv_masses[4].is_zero());
    c.unpin(4);
    assert_eq!(c.inv_masses[4], m);
    assert!(!c.pinned.contains(&4));
    // unpinning a particle that is not pinned changes nothing
    let before = c.inv_masses.clone();
    c.unpin(5);
    assert_eq!(c.inv_masses, before);
    // a particle pinned from construction (pin_top_row) is unpinned to the grid's mass
    let mut d = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        RES,
        RES,
        Fix128::from_int(2),
    );
    d.pin_top_row(RES);
    let p = d.pinned[0];
    d.unpin(p);
    assert_eq!(d.inv_masses[p], Fix128::ONE / Fix128::from_int(2));
}

/// The script's whole trajectory with sleep on, digested (bit pattern of every frame). The
/// same on every target; and equal to the run without sleep (checked above frame by frame).
#[test]
fn trajectory_digest_with_sleep_is_pinned() {
    use sha2::{Digest, Sha256};
    // 540 frames, the reference's length (it does not follow FIRST_IDENTITY_STEP)
    let (states, slept) = run(540, None, true);
    assert!(slept > 0);
    let mut h = Sha256::new();
    for s in &states {
        for v in s.0.iter().chain(&s.1).chain(&s.2) {
            for c in [v.x, v.y, v.z] {
                h.update(c.hi.to_le_bytes());
                h.update(c.lo.to_le_bytes());
            }
        }
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, TRAJECTORY_DIGEST);
}

// from Physics main bc76ee2d without any sleep code: the same scene never slept (release by
// restoring the inverse mass by hand), 540 frames
const TRAJECTORY_DIGEST: &str = "154ff855bc90da827373172562aa9471c050948e6df59e6340353a87ced2885c";
