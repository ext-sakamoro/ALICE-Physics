//! A strand of hair swinging behind a moving head with `SecondaryChain`.
//!
//! Five particles hang from the anchor (the head's back). Their settings come from curves along
//! the strand (`curve_value`): stiffer at the root, freer at the tip. The neck is a capsule the
//! strand starts touching at its root, so `fit_to_rest` masks it there (`attached_inside` shows
//! the same mask) and caps the radius elsewhere to the rest clearance. The head turns left and
//! right; the strand lags behind, stays out of the neck and keeps its length.
//!
//! Run: `cargo run --example secondary_chain_hair --features std`

use alice_physics_core::collider::Capsule;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::secondary_chain::{
    curve_value, ChainConfig, ChainError, ChainInput, ChainLimit, ChainParams, ChainParticle,
    CurveKey, SecondaryChain,
};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn main() -> Result<(), ChainError> {
    let head = Vec3Fix::new(Fix128::ZERO, r(16, 10), r(-1, 10));
    // the strand at rest: straight down behind the head, 6 cm apart
    let n: usize = 5;
    let rest: Vec<Vec3Fix> = (1..=n)
        .map(|i| head + Vec3Fix::new(Fix128::ZERO, r(-6 * i as i64, 100), r(-1, 100)))
        .collect();
    // pull from 0.3 at the root to 0.05 at the tip; the angle limit from 0.4 rad to 1.2 rad
    let pull = [
        CurveKey {
            time: Fix128::ZERO,
            value: r(3, 10),
            in_slope: Fix128::ZERO,
            out_slope: Fix128::ZERO,
        },
        CurveKey {
            time: Fix128::ONE,
            value: r(5, 100),
            in_slope: Fix128::ZERO,
            out_slope: Fix128::ZERO,
        },
    ];
    let angle = [
        CurveKey {
            time: Fix128::ZERO,
            value: r(4, 10),
            in_slope: Fix128::ZERO,
            out_slope: Fix128::ZERO,
        },
        CurveKey {
            time: Fix128::ONE,
            value: r(12, 10),
            in_slope: Fix128::ZERO,
            out_slope: Fix128::ZERO,
        },
    ];
    let particles: Vec<ChainParticle> = (0..n)
        .map(|i| {
            let t = r(i as i64, (n - 1) as i64);
            ChainParticle {
                parent: i.checked_sub(1),
                rest_length: (rest[i] - if i == 0 { head } else { rest[i - 1] }).length(),
                params: ChainParams {
                    pull: curve_value(&pull, t),
                    momentum: r(7, 10),
                    stiffness: r(1, 10),
                    gravity: r(1, 2),
                    gravity_falloff: Fix128::ZERO,
                    immobile: Fix128::ZERO,
                    limit: ChainLimit::Angle {
                        max: curve_value(&angle, t),
                    },
                    radius: r(1, 100),
                    attached_inside: 0,
                },
                p: rest[i],
                q: rest[i],
            }
        })
        .collect();
    let mut strand = SecondaryChain::new(
        particles,
        ChainConfig {
            gravity: Vec3Fix::new(Fix128::ZERO, r(-981, 100), Fix128::ZERO),
            iterations: 2,
        },
    )?;
    // the neck: a short vertical capsule in front of the strand, holding its root end (the head)
    let neck = [Capsule {
        a: head + Vec3Fix::new(Fix128::ZERO, r(-4, 100), r(2, 100)),
        b: head + Vec3Fix::new(Fix128::ZERO, r(2, 100), r(2, 100)),
        radius: r(3, 100),
    }];
    let masks = SecondaryChain::attached_inside(strand.particles(), &rest, head, &neck)?;
    strand.fit_to_rest(head, &neck)?;
    println!("[secondary_chain_hair] {n} particles, masks from the rest {masks:?}");

    let dt = r(1, 60);
    let mut prev = head;
    for frame in 0..120 {
        // the head sways along x: a triangle wave of ±8 cm, period 40 frames
        let m = frame % 40;
        let u = if m < 20 { m } else { 40 - m };
        let anchor = head + Vec3Fix::new(r(8 * (u - 10), 1000), Fix128::ZERO, Fix128::ZERO);
        let anim: Vec<Vec3Fix> = rest.iter().map(|&p| p - head + anchor).collect();
        // the bones turn with the sway (from straight down toward the anchor's offset)
        let down = Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO);
        let turn = QuatFix::from_to(down, down + (anchor - head));
        let frames = vec![turn; n];
        strand.step(
            dt,
            &ChainInput {
                anchor,
                anim_anchor: anchor,
                anim: &anim,
                anim_frames: &frames,
                anchor_delta: anchor - prev,
                colliders: &neck,
            },
        )?;
        prev = anchor;
        if frame % 20 == 19 {
            let tip = strand.particles()[n - 1].p;
            println!(
                "[secondary_chain_hair] frame {:3} anchor x {:+.3} tip ({:+.3}, {:.3}, {:+.3})",
                frame + 1,
                anchor.x.to_f64(),
                tip.x.to_f64(),
                tip.y.to_f64(),
                tip.z.to_f64()
            );
        }
    }
    // the bone lengths hold exactly after every step
    let mut parent = prev;
    for p in strand.particles() {
        let len = (p.p - parent).length();
        assert!(
            (len - p.rest_length).abs() < r(1, 1_000_000),
            "length {}",
            len.to_f64()
        );
        parent = p.p;
    }
    println!(
        "[secondary_chain_hair] lengths held, {} m each",
        strand.particles()[1].rest_length.to_f64()
    );
    Ok(())
}
