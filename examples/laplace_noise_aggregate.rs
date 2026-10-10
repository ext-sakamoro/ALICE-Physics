//! Lattice Laplace mechanism with an entropy-keyed generator, averaged by an
//! aggregator
//!
//! Reaches `DpNoise::try_from_entropy`, `DpNoise::lattice`,
//! `DpNoise::effective_epsilon` and `PrivateAggregator::count`. `DpNoise`
//! rounds to the lattice `Λ = 2^(⌊log2 Δf⌋ − 20)` and adds `Λ ·` discrete
//! Laplace noise sampled with integer arithmetic; to within `2^-20` that is
//! Laplace(0, b), so the closed forms below hold.
//!
//! Closed forms (Laplace(0, b) with `b = Δf / ε`, independent of the code):
//! - `scale = sensitivity / epsilon`, so `Δf = 2`, `ε = 0.5` gives `b = 4`
//! - `E[X] = 0`, `Var[X] = 2 b^2`, so the mean of `n` draws has standard
//!   error `b √2 / √n`
//! - `E[|X|] = b`, `Var[|X|] = b^2`, so the mean absolute draw has standard
//!   error `b / √n`
//!
//! The key comes from OS entropy, so the run is not reproducible; the statistical
//! checks use an 8-standard-error band (two-sided tail below 1e-14 under the
//! normal approximation), which a wrong scale (any factor off by 10 %) leaves.
//!
//! Run with: `cargo run --example laplace_noise_aggregate`

use alice_physics::privacy::DpNoise;
use alice_physics::PrivateAggregator;

fn main() {
    let sensitivity = 2.0_f64;
    let epsilon = 0.5_f64;
    let b = 4.0_f64; // Δf / ε, worked by hand

    let mut noise = DpNoise::try_from_entropy(sensitivity, epsilon).expect("entropy and valid ε");
    assert_eq!(noise.sensitivity() / noise.epsilon(), b, "scale is Δf / ε");
    // the lattice for Δf = 2 is 2^(1 − 20); rounding costs at most ε·2^-20
    assert_eq!(noise.lattice(), 1.0 / 524_288.0); // 2^-19
    assert!(noise.effective_epsilon() >= epsilon);
    assert!(noise.effective_epsilon() <= epsilon * (1.0 + 1.0 / 1_048_576.0)); // 2^-20

    let n: u32 = 40_000;
    let true_value = 100.0_f64;
    let mut agg = PrivateAggregator::new(b);
    let mut abs_sum = 0.0_f64;
    for _ in 0..n {
        let released = noise.privatize(true_value).expect("value fits the lattice");
        abs_sum += (released - true_value).abs();
        agg.add(released);
    }
    assert_eq!(agg.count(), u64::from(n), "one report per release");

    let nf = f64::from(n);
    let se_mean = b * core::f64::consts::SQRT_2 / nf.sqrt();
    let se_abs = b / nf.sqrt();
    let mean = agg.estimate_mean();
    let mean_abs = abs_sum / nf;
    println!(
        "[privacy] n = {n}, b = {b}: mean = {mean:.4} (true {true_value}, SE {se_mean:.4}), \
         mean |noise| = {mean_abs:.4} (closed form {b}, SE {se_abs:.4})"
    );

    assert!(
        (mean - true_value).abs() < 8.0 * se_mean,
        "noisy mean {mean} is outside 8 SE of the true value {true_value}"
    );
    assert!(
        (mean_abs - b).abs() < 8.0 * se_abs,
        "mean |noise| {mean_abs} is outside 8 SE of E|X| = b = {b}"
    );
    assert!(
        (agg.standard_error() - se_mean).abs() < 1e-12,
        "aggregator SE {} matches b √2 / √n = {se_mean}",
        agg.standard_error()
    );

    agg.reset();
    assert_eq!(agg.count(), 0, "reset clears the count");
    println!("[privacy] scale, count and the noise moments match the Laplace closed forms");
}
