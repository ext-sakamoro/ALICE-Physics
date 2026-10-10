//! Local differential privacy: budget tracking, lattice Laplace noise,
//! randomized response and RAPPOR, each printed next to its closed form.
//!
//! The noise comes from a keyed ChaCha20 stream (`SecureRng`, a 32-byte secret
//! key), so every line is reproducible with the key and unpredictable without
//! it; the closed forms are
//!
//! * budget: `remaining = ε_max − Σ accepted`, exhausted exactly when
//!   `Σ accepted ≥ ε_max`, a spend is refused exactly when it would exceed,
//! * Laplace: `b = Δf / ε`; `DpNoise` rounds to the lattice
//!   `Λ = 2^(⌊log2 Δf⌋ − 20)` and adds `Λ ·` discrete Laplace noise, so the
//!   aggregator's standard error is `b·√2/√n` to within `2^-20`,
//! * randomized response: the bit is kept with probability `e^ε / (1 + e^ε)`
//!   and flipped otherwise,
//! * RAPPOR: `f = 0`, `p = 1`, `q = 0` is the identity on the Bloom filter.
//!
//! ```bash
//! cargo run --example privacy_budget_and_rappor --features std
//! ```

use alice_physics::det_math::exp64;
use alice_physics::privacy::{
    dp_int, randomized_response, DpNoise, KeyedRappor, PrivacyBudget, PrivateAggregator, SecureRng,
    RAPPOR_BITS,
};

fn main() {
    // ---- privacy budget (sequential composition) ---------------------------
    let eps_max = 1.0;
    let mut budget = PrivacyBudget::new(eps_max);
    let mut accepted_sum = 0.0;
    for eps in [0.5, 0.25, 0.25, 0.125, 0.0] {
        let ok = budget.try_spend(eps);
        if ok {
            accepted_sum += eps;
        }
        println!(
            "[privacy] try_spend({eps}) -> {ok} | spent={} (closed form Σ={accepted_sum}) \
             remaining={} (closed form {}) exhausted={} queries={}",
            budget.spent(),
            budget.remaining(),
            eps_max - accepted_sum,
            budget.is_exhausted(),
            budget.query_count()
        );
    }
    budget.reset();
    println!(
        "[privacy] reset -> spent={} remaining={} exhausted={} queries={} (fresh: 0 {eps_max} false 0)",
        budget.spent(),
        budget.remaining(),
        budget.is_exhausted(),
        budget.query_count()
    );

    // ---- lattice Laplace mechanism + aggregator ----------------------------
    let (sensitivity, epsilon, key) = (1.0, 2.0, [42u8; 32]);
    let scale = sensitivity / epsilon;
    let mut laplace = DpNoise::with_key(sensitivity, epsilon, key);
    let mut agg = PrivateAggregator::new(scale);
    let truth = 100.0;
    let n = 8u64;
    for _ in 0..n {
        agg.add(laplace.privatize(truth).expect("valid value"));
    }
    let mut rng = SecureRng::from_key(key);
    let rounded = dp_int(100, 1, epsilon, &mut rng).expect("valid ε");
    println!(
        "[privacy] laplace b={scale} (closed form Δf/ε={}) lattice={} ε_eff={} n={n}: mean={} \
         sum={} se={} (closed form b√2/√n={}) dp_int(100)={rounded}",
        sensitivity / epsilon,
        laplace.lattice(),
        laplace.effective_epsilon(),
        agg.estimate_mean(),
        agg.estimate_sum(),
        agg.standard_error(),
        scale * core::f64::consts::SQRT_2 / (n as f64).sqrt()
    );
    agg.reset();
    println!(
        "[privacy] aggregator reset -> sum={} se={} (fresh: 0 inf)",
        agg.estimate_sum(),
        agg.standard_error()
    );

    // ---- randomized response ----------------------------------------------
    let eps_rr = 1.0;
    let keep = exp64(eps_rr) / (1.0 + exp64(eps_rr));
    let n_reports = 4000u32;
    let kept = (0..n_reports)
        .filter(|&i| {
            let truth = i % 2 == 0;
            randomized_response(truth, eps_rr, &mut rng).expect("valid ε") == truth
        })
        .count();
    let observed = kept as f64 / f64::from(n_reports);
    let se = (keep * (1.0 - keep) / f64::from(n_reports)).sqrt();
    println!(
        "[privacy] randomized response ε={eps_rr}: kept {kept}/{n_reports} = {observed:.4} \
         (closed form e^ε/(1+e^ε)={keep:.4}, SE {se:.4})"
    );
    assert!(
        (observed - keep).abs() < 8.0 * se,
        "kept fraction {observed} is outside 8 SE of e^ε/(1+e^ε) = {keep}"
    );

    // ---- RAPPOR ------------------------------------------------------------
    let mut rappor = KeyedRappor::with_key((1, 2), (3, 4), (1, 4), key).expect("valid fractions");
    let report = rappor.privatize(12345);
    let ones = report.iter().filter(|&&bit| bit == 1).count();
    println!(
        "[privacy] rappor params={:?} (f=1/2 p=3/4 q=1/4) bits={} (RAPPOR_BITS={RAPPOR_BITS}) ones={ones}",
        rappor.params(),
        report.len(),
    );
    let mut identity = KeyedRappor::with_key((0, 1), (1, 1), (0, 1), key).expect("valid fractions");
    let bloom = identity.privatize(12345);
    let set: Vec<usize> = (0..RAPPOR_BITS).filter(|&i| bloom[i] == 1).collect();
    println!(
        "[privacy] rappor f=0 p=1 q=0 is the identity on the Bloom filter: set bits {set:?} \
         (3 hash positions, fewer on collision) params={:?}",
        identity.params()
    );
    assert!(
        (1..=3).contains(&set.len()),
        "the identity RAPPOR report is the Bloom filter (1 to 3 bits set): {set:?}"
    );
}
