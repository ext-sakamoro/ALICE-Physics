//! Oracles of the thick chain's bending law
//! ([`alice_physics_core::thick_chain::ThickChainBend`]).
//!
//! 1. The law is set by [`ThickChain::bending`] alone. A chain as made has none and its bytes
//!    are those of a chain before the law existed (the digest below was taken on the commit
//!    before this file); with the law the bytes say so (another version, the law's tag).
//! 2. Given the law after steps, a chain is the chain made in that state with the law.
//! 3. A straight chain with a fixed pair of particles and no load does not move by a bit.
//! 4. A straight chain is not a special place: pushed off its line by `+v` or by `−v` at the
//!    head it runs mirrored bit for bit, and not as a chain without the law does.
//! 5. What cannot be bent is refused where it is asked for: the law on links that are not
//!    neo-Hookean, a link of no length, a joint folded flat or bent past the radius of its
//!    half-width, when the law is set and when a step meets it (the step then writes
//!    nothing); a chain of one link has no joint and swings as without the law (to a
//!    tolerance: with the law a substep is solved another way, see the MODEL section).
//!
//! Integration test only: the chains of the files that follow this one settle over many
//! frames.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainBend, ThickChainConfig, ThickChainError, ThickChainFit,
    ThickChainStretch, P2,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn f(v: Fix128) -> f64 {
    v.to_f64()
}

const ZERO2: P2 = [Fix128::ZERO, Fix128::ZERO];
const LINK: f64 = 0.01;
const RADIUS: f64 = 0.02;
const DENSITY: f64 = 1000.0;
const SHEAR: f64 = 1000.0;

fn frame() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn config() -> ThickChainConfig {
    ThickChainConfig {
        substeps: 32,
        iterations: 8,
        gravity: ZERO2,
        stretch: ThickChainStretch::NeoHookean { shear: fx(SHEAR) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.002),
        splits: 0,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    }
}

fn row(links: usize) -> Vec<P2> {
    (0..=links)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect()
}

fn made(cfg: ThickChainConfig, points: &[P2]) -> ThickChain {
    ThickChain::new(cfg, points, fx(RADIUS), fx(DENSITY)).expect("a chain")
}

/// a straight chain of `links` links with the law, its first two particles fixed
fn clamped(links: usize) -> ThickChain {
    let mut chain = made(config(), &row(links))
        .bending(ThickChainBend::Section)
        .expect("the law");
    chain.pin(0);
    chain.pin(1);
    chain
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The digest of `made(config(), &row(4))` on the commit before the bending law
/// (`ThickChain::state_digest`, printed there by a run of that commit's library).
const BEFORE_THE_LAW: &str = "da9c95622dcb4131b1ba4da840b4deffd8b02e2f81bdba37bec07ae86005560b";

#[test]
fn a_chain_is_made_without_the_law_and_its_bytes_are_those_before_the_law() {
    let chain = made(config(), &row(4));
    assert_eq!(chain.bend(), ThickChainBend::None);
    assert_eq!(hex(&chain.state_digest()), BEFORE_THE_LAW);
    let bytes = chain.snapshot();
    assert_eq!(&bytes[..4], b"TCHN");
    assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
    // the settings (two counts, the particles' count, eleven numbers), five numbers a
    // particle, two a link: nothing else
    assert_eq!(
        bytes.len(),
        8 + 4 + 4 + 8 + 11 * 16 + 5 * 5 * 16 + 4 * 2 * 16
    );
    // asked for no law, the same chain
    let asked = made(config(), &row(4))
        .bending(ThickChainBend::None)
        .expect("no law");
    assert_eq!(asked.snapshot(), bytes);
}

#[test]
fn with_the_law_the_bytes_say_so() {
    let without = made(config(), &row(4));
    let with = made(config(), &row(4))
        .bending(ThickChainBend::Section)
        .expect("the law");
    assert_eq!(with.bend(), ThickChainBend::Section);
    let (a, b) = (without.snapshot(), with.snapshot());
    assert_ne!(a, b);
    assert_ne!(without.state_digest(), with.state_digest());
    assert_eq!(&b[..4], b"TCHN");
    assert_eq!(&b[4..8], &2u32.to_le_bytes());
    // the law's tag is one number after the settings; the rest is the chain without the law
    assert_eq!(b.len(), a.len() + 16);
    let settings = 8 + 4 + 4 + 8 + 11 * 16;
    assert_eq!(&b[8..settings], &a[8..settings]);
    assert_eq!(&b[settings + 16..], &a[settings..]);
    assert_ne!(&b[settings..settings + 16], &[0u8; 16]);
    // (a law of no stiffness cannot be made to look like no law: a shear modulus of 0 is
    // refused when the chain is made)
    let mut soft = config();
    soft.stretch = ThickChainStretch::NeoHookean {
        shear: Fix128::ZERO,
    };
    assert_eq!(
        ThickChain::new(soft, &row(4), fx(RADIUS), fx(DENSITY)).err(),
        Some(ThickChainError::Invalid("shear"))
    );
}

#[test]
fn given_the_law_after_steps_a_chain_is_the_chain_made_in_that_state_with_the_law() {
    let mut cfg = config();
    cfg.gravity = [Fix128::ZERO, fx(-0.5)];
    let mut first = made(cfg, &row(6));
    first.pin(0);
    first.pin(1);
    for _ in 0..5 {
        first.step(frame()).expect("a step");
    }
    let mut second = made(cfg, &row(6));
    second.pin(0);
    second.pin(1);
    for i in 0..first.particle_count() {
        second.set_position(i, first.positions()[i]);
        second.set_velocity(i, first.velocities()[i]);
    }
    let mut first = first.bending(ThickChainBend::Section).expect("the law");
    let mut second = second.bending(ThickChainBend::Section).expect("the law");
    assert_eq!(first.state_digest(), second.state_digest());
    for _ in 0..5 {
        first.step(frame()).expect("a step");
        second.step(frame()).expect("a step");
    }
    assert_eq!(first.state_digest(), second.state_digest());
    // and the law acted in those steps: the hanging part is not where it is without it
    let mut none = made(cfg, &row(6));
    none.pin(0);
    none.pin(1);
    for _ in 0..10 {
        none.step(frame()).expect("a step");
    }
    assert_ne!(none.positions(), first.positions());
}

#[test]
fn a_straight_chain_held_by_two_particles_does_not_move_by_a_bit() {
    let mut chain = clamped(12);
    let before = chain.positions().to_vec();
    for _ in 0..60 {
        chain.step(frame()).expect("a step");
    }
    assert_eq!(chain.positions(), &before[..]);
    assert!(chain.velocities().iter().all(|v| *v == ZERO2));
}

#[test]
fn off_its_line_either_way_a_straight_chain_runs_mirrored_bit_for_bit() {
    let run = |v: f64, law: bool| {
        let mut chain = if law {
            clamped(12)
        } else {
            let mut chain = made(config(), &row(12));
            chain.pin(0);
            chain.pin(1);
            chain
        };
        chain.set_velocity(12, [Fix128::ZERO, fx(v)]);
        for _ in 0..30 {
            chain.step(frame()).expect("a step");
        }
        chain.positions().to_vec()
    };
    // the push is small: the joints turn by about 1e-5 rad, where a law with a corner at the
    // straight chain would answer by its side, not by its size
    let (up, down) = (run(1.0e-5, true), run(-1.0e-5, true));
    for (a, b) in up.iter().zip(&down) {
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], -b[1]);
    }
    // the head came off the line, and the law held it back: without the law it is further
    let free = run(1.0e-5, false);
    assert!(f(up[12][1]) > 0.0);
    assert!(
        f(up[12][1]) < 0.9 * f(free[12][1]),
        "{} with the law, {} without",
        f(up[12][1]),
        f(free[12][1])
    );
}

/// the points of a chain whose joint at particle 1 turns by `turn` rad
fn bent_once(turn: f64) -> Vec<P2> {
    vec![
        [Fix128::ZERO, Fix128::ZERO],
        [fx(LINK), Fix128::ZERO],
        [fx(LINK + LINK * turn.cos()), fx(LINK * turn.sin())],
        [
            fx(LINK + 2.0 * LINK * turn.cos()),
            fx(2.0 * LINK * turn.sin()),
        ],
    ]
}

#[test]
fn what_cannot_be_bent_is_refused_when_the_law_is_set() {
    // links that are not neo-Hookean have no section to take the law from
    let mut linear = config();
    linear.stretch = ThickChainStretch::Linear {
        compliance: fx(1.0e-4),
    };
    assert_eq!(
        made(linear, &row(4)).bending(ThickChainBend::Section).err(),
        Some(ThickChainError::Invalid("bend"))
    );
    // a joint folded flat: the link goes back the way the one before it came
    let flat = vec![
        [Fix128::ZERO, Fix128::ZERO],
        [fx(LINK), Fix128::ZERO],
        [Fix128::ZERO, Fix128::ZERO],
    ];
    assert_eq!(
        made(config(), &flat).bending(ThickChainBend::Section).err(),
        Some(ThickChainError::Folded { joint: 1 })
    );
    // a joint bent to the radius of the half-width or less: with links of `LINK` and the
    // radius `RADIUS` the joint's curvature `2 tan(θ/2) / LINK` reaches `1 / RADIUS` at
    // `θ = 2 atan(LINK / (2 RADIUS))`
    let limit = 2.0 * (LINK / (2.0 * RADIUS)).atan();
    assert_eq!(
        made(config(), &bent_once(1.01 * limit))
            .bending(ThickChainBend::Section)
            .err(),
        Some(ThickChainError::Folded { joint: 1 })
    );
    assert_eq!(
        made(config(), &bent_once(-1.01 * limit))
            .bending(ThickChainBend::Section)
            .err(),
        Some(ThickChainError::Folded { joint: 1 })
    );
    // just inside it the law is taken, and without the law any of them is a chain
    assert!(made(config(), &bent_once(0.98 * limit))
        .bending(ThickChainBend::Section)
        .is_ok());
    assert!(made(config(), &flat).bending(ThickChainBend::None).is_ok());
}

#[test]
fn a_step_that_meets_a_fold_is_refused_and_writes_nothing() {
    let mut chain = clamped(6);
    // put the head's link back along the link before it
    let back = chain.positions()[4];
    chain.set_position(6, back);
    let before = chain.snapshot();
    assert_eq!(
        chain.step(frame()),
        Err(ThickChainError::Folded { joint: 5 })
    );
    assert_eq!(chain.snapshot(), before);
}

#[test]
fn a_chain_of_one_link_has_no_joint_and_swings_as_without_the_law() {
    // one link hung by an end and let go level: no joint, so the law adds no force. The bits
    // are not those of the chain without the law: with the law a substep's stretch is put
    // along the link as it lies at each visit (the module's MODEL section), without it along
    // the link as it lay at each step. What the two can differ by in a substep is the
    // stretch's move times the angle the link turns in it: the move is at most the tension
    // over the mass times h² (the tension of a link let go level is at most three times its
    // weight, at the bottom of its swing), the turn at most the speed over the length times
    // h (the speed at most √(2 g l)). Summed over the substeps, as if every substep's
    // difference went the same way:
    let (g, substeps) = (9.81f64, 30.0 * 32.0);
    let h = 1.0 / (60.0 * 32.0);
    let bound = substeps * (3.0 * g * h * h) * ((2.0 * g / LINK).sqrt() * h);
    assert!(bound < 0.02 * LINK, "the bound {bound}");
    let mut cfg = config();
    cfg.gravity = [Fix128::ZERO, fx(-g)];
    let run = |law: ThickChainBend| {
        let mut chain = made(cfg, &row(1)).bending(law).expect("the law");
        chain.pin(0);
        for _ in 0..30 {
            chain.step(frame()).expect("a step");
        }
        let head = chain.positions()[1];
        [f(head[0]), f(head[1])]
    };
    let (with, without) = (run(ThickChainBend::Section), run(ThickChainBend::None));
    let apart = (with[0] - without[0]).hypot(with[1] - without[1]);
    assert!(
        apart < bound,
        "{with:?} with the law, {without:?} without: {apart} apart, the bound {bound}"
    );
    // (the two are different runs: the comparison is not of a thing with itself)
    assert!(apart > 0.0);
    // it swung: the head is well below where it was let go
    assert!(without[1] < -0.5 * LINK, "{without:?}");
}
