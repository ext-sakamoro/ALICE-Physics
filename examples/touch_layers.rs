//! Layered touch playback on soft sheets: record a top layer of touches live over a lower layer, write both logs as
//! bytes, read them back with bounds, replay them, check the playback against limits (weakening the top layer if it
//! goes beyond them), and print the history digest. Then the step cost: one busy scene driven on the bodies
//! directly, the step and the digest timed apart, with the most bodies moving at once.
//!
//! Run: `cargo run --release --example touch_layers`

use std::time::Instant;

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::{InputLog, LogDecodeError, TOUCH_LOG_MAX_BYTES};
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::touch_layers::{
    history_digest, replay, LayerError, LayeredPlayback, PlaybackConfig, Recording, TouchBodies,
};
use alice_physics_core::touch_limits::{check, within_limits, TouchLimits, Violation};
use alice_physics_core::{Fix128, Vec3Fix};

const STEPS: u64 = 300;
const RATE: u64 = 60;

fn rest_centre(i: usize) -> Vec3Fix {
    let i = i64::try_from(i).expect("few bodies");
    Vec3Fix::new(
        Fix128::from_ratio(6 * i, 5) + Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    )
}

/// `n` unit sheets of 4 × 4 particles (10 g each), no gravity, held to their rest shape (period 0.5 s, ζ 0.42)
fn sheets(n: usize) -> Vec<Cloth> {
    let mass = Fix128::from_ratio(1, 100);
    let omega = Fix128::from_ratio(12_566, 1000);
    let half = Fix128::from_ratio(1, 2);
    (0..n)
        .map(|i| {
            let origin = rest_centre(i) - Vec3Fix::new(half, Fix128::ZERO, half);
            let mut c = Cloth::new_grid(origin, Fix128::ONE, Fix128::ONE, 4, 4, mass);
            c.config.gravity = Vec3Fix::ZERO;
            c.config.damping = Fix128::ONE;
            c.config.iterations = 4;
            c.config.substeps = 1;
            c.set_rest_tether(
                mass * omega * omega,
                Fix128::from_ratio(84, 100) * mass * omega,
            );
            c
        })
        .collect()
}

fn at(g: usize, dx: i64, dz: i64) -> Vec3Fix {
    rest_centre(g)
        + Vec3Fix::new(
            Fix128::from_ratio(dx, 100),
            Fix128::ZERO,
            Fix128::from_ratio(dz, 100),
        )
}

fn hex(d: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    d.iter().fold(String::new(), |mut h, b| {
        let _ = write!(h, "{b:02x}");
        h
    })
}

fn lower_layer() -> InputLog<Touch> {
    let mut l = InputLog::new();
    l.record(10, Touch::grab(0, at(0, -40, -40)))
        .expect("in order");
    for s in 11..70u64 {
        let k = i64::try_from(s - 10).expect("small");
        l.record(s, Touch::drag(at(0, -40 + k / 2, -40 + k / 3)))
            .expect("in order");
    }
    l.record(70, Touch::Release).expect("in order");
    l
}

fn record(
    n: usize,
    lower: &InputLog<Touch>,
    config: PlaybackConfig,
) -> Result<Recording, LayerError> {
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let mut p = LayeredPlayback::new(sheets(n), &hand, vec![lower.clone()], config)?;
    while p.step_now() < STEPS {
        let s = p.step_now();
        let k = i64::try_from(s).expect("small");
        match s {
            40 => p.touch(&Touch::grab(0, at(0, 20, 20)))?,
            41..=59 => p.touch(&Touch::drag(at(0, 20 + k - 40, 20)))?,
            60 | 160 => p.touch(&Touch::Release)?,
            120 => p.touch(&Touch::grab(3, at(3, 0, 0)))?,
            121..=159 => p.touch(&Touch::drag(at(3, 0, -(k - 120))))?,
            _ => {}
        }
        p.advance()?;
    }
    println!(
        "recorded: {} bodies, {} moving at the end",
        p.bodies().len(),
        p.bodies().active().len()
    );
    Ok(p.finish().0)
}

#[allow(
    clippy::disallowed_methods,
    reason = "timing harness: the time per step and per digest that this program prints"
)]
fn step_cost() {
    let config = PlaybackConfig::new(
        STEPS,
        Fix128::from_ratio(1, 60),
        5,
        Fix128::from_ratio(1, 65_536),
    )
    .expect("config");
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let mut b = TouchBodies::new(sheets(20), &hand, 1, config).expect("bodies");
    assert!(!b.is_empty());
    let mut steps = Vec::new();
    let mut digests = Vec::new();
    let mut most = 0;
    for s in 0..STEPS {
        let g = usize::try_from(s / 12).expect("small") % 20;
        let id = u32::try_from(g).expect("small");
        match s % 12 {
            0 => {
                b.input(0, &Touch::grab(id, at(g, 30, 30)));
            }
            1..=8 => {
                let k = i64::try_from(s % 12).expect("small");
                b.input(0, &Touch::drag(at(g, 30 + 4 * k, 30 - 3 * k)));
            }
            9 => {
                b.input(0, &Touch::Release);
            }
            _ => {}
        }
        let t0 = Instant::now();
        b.step();
        steps.push(t0.elapsed().as_secs_f64() * 1e3);
        let t1 = Instant::now();
        let d = b.digest();
        digests.push(t1.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(d);
        most = most.max(b.active().len());
    }
    let warm = usize::try_from(RATE).expect("small");
    let stat = |v: &[f64]| {
        let mut w: Vec<f64> = v[warm..].to_vec();
        w.sort_by(f64::total_cmp);
        let p99 = w[(w.len() * 99) / 100 - 1];
        let count = u32::try_from(w.len()).expect("few samples");
        let mean = w.iter().sum::<f64>() / f64::from(count);
        (mean, p99, w[w.len() - 1])
    };
    let (sm, sp, sw) = stat(&steps);
    let (dm, dp, dw) = stat(&digests);
    let first_worst = steps[..warm].iter().copied().fold(0.0, f64::max);
    let centre = b.centre(19).expect("body 19");
    let particles = b.cloth(19).map_or(0, |c| c.positions.len());
    println!(
        "step: mean {sm:.3} p99 {sp:.3} worst {sw:.3} ms | digest: mean {dm:.3} p99 {dp:.3} worst {dw:.3} ms | first second worst step {first_worst:.3} ms | most moving {most} (budget 5)"
    );
    println!(
        "body 19 ({particles} particles) centre x {:.4} m | final state {}",
        centre.x.to_f64(),
        hex(&b.digest())
    );
}

fn main() -> Result<(), LayerError> {
    let n = 6;
    let config = PlaybackConfig::new(
        STEPS,
        Fix128::from_ratio(1, 60),
        5,
        Fix128::from_ratio(1, 65_536),
    )?;
    let lower = lower_layer();
    let rec = record(n, &lower, config)?;

    // the logs as bytes, read back with bounds
    let bound = u32::try_from(n).expect("few bodies");
    let lower_bytes = lower.to_bytes();
    let top_bytes = rec.top.to_bytes();
    println!(
        "logs: lower {} bytes, top {} bytes (the most read: {TOUCH_LOG_MAX_BYTES})",
        lower_bytes.len(),
        top_bytes.len()
    );
    let read = |b: &[u8]| -> Result<InputLog<Touch>, LogDecodeError> {
        InputLog::from_bytes_bounded(b, STEPS, bound)
    };
    let (Ok(lower_read), Ok(top_read)) = (read(&lower_bytes), read(&top_bytes)) else {
        println!("a log did not read back");
        return Ok(());
    };
    assert_eq!(
        read(&top_bytes[..top_bytes.len() - 1]),
        Err(LogDecodeError::Truncated)
    );

    // the replay goes through the same frames
    let hand = Hand::new(Fix128::from_ratio(4, 10));
    let layers = [lower_read, top_read];
    let again = replay(&sheets(n), &hand, &layers, config)?;
    assert_eq!(again.frames, rec.frames);
    assert_eq!(again.history, history_digest(&again.frames));
    println!(
        "history digest {} ({} frames, replay agrees)",
        hex(&again.history),
        again.frames.len()
    );

    // limits: 6 reversals a second, 0.6 m from rest, swings under 1/32 m not counted
    let limits = TouchLimits::new(
        6,
        Fix128::from_ratio(6, 10),
        Fix128::from_ratio(1, 32),
        RATE,
    )?;
    let found: Vec<Violation> = check(&sheets(n), &hand, &layers, config, &limits)?;
    println!("beyond the limits: {found:?}");
    match within_limits(&sheets(n), &hand, &layers, config, &limits)? {
        Some((top, scale)) => println!(
            "sent at scale {:.4} ({} top-layer touches)",
            scale.to_f64(),
            top.entries().len()
        ),
        None => println!("the lower layer alone is beyond the limits: sent without motion"),
    }

    step_cost();
    Ok(())
}
