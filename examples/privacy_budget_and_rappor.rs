//! Local differential privacy: budget tracking, Laplace noise, randomized
//! response and RAPPOR, each printed next to its closed form.
//!
//! Everything but RAPPOR runs on an explicit seed, so every line is
//! reproducible; the closed forms are
//!
//! * budget: `remaining = ε_max − Σ accepted`, exhausted exactly when
//!   `Σ accepted ≥ ε_max`, a spend is refused exactly when it would exceed,
//! * Laplace: `b = Δf / ε`, the sample is `−sign(u)·b·ln(1 − 2|u|)` with
//!   `u = U − ½`, the aggregator's standard error is `b·√2/√n`,
//! * randomized response: truthful with probability `p`, otherwise a fair
//!   coin, so the report probabilities are `(1 ± p) / 2` and
//!   `ε = ln((1 + p) / (1 − p))`, i.e. `p_true = (e^ε − 1) / (e^ε + 1)`
//!   `= 1 − 2 / (e^ε + 1)`; with `k` positive
//!   reports out of `n` the unbiased proportion is `(k/n − (1 − p)/2) / p`,
//! * RAPPOR: `f = 0`, `p = 1`, `q = 0` is the identity on the Bloom filter.
//!
//! ```bash
//! cargo run --example privacy_budget_and_rappor --features std
//! ```

use alice_physics::det_math::exp64;
use alice_physics::privacy::{
    LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
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

    // ---- Laplace mechanism + aggregator ------------------------------------
    let (sensitivity, epsilon, seed) = (1.0, 2.0, 42u64);
    let scale = sensitivity / epsilon;
    let mut laplace = LaplaceNoise::with_seed(sensitivity, epsilon, seed);
    let mut agg = PrivateAggregator::new(scale);
    let truth = 100.0;
    let n = 8u64;
    for _ in 0..n {
        agg.add(laplace.privatize(truth));
    }
    let rounded = laplace.privatize_int(100);
    println!(
        "[privacy] laplace b={scale} (closed form Δf/ε={}) n={n}: mean={} sum={} se={} \
         (closed form b√2/√n={}) privatize_int(100)={rounded}",
        sensitivity / epsilon,
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
    let rr_from_eps = RandomizedResponse::new(eps_rr);
    let e = exp64(eps_rr);
    let p_closed = (e - 1.0) / (e + 1.0);
    println!(
        "[privacy] randomized response ε={eps_rr}: p_true={} (closed form (e^ε−1)/(e^ε+1)={p_closed})",
        rr_from_eps.p_true(),
    );
    assert!(
        (rr_from_eps.p_true() - p_closed).abs() < 1e-12,
        "RandomizedResponse::new(ε).p_true() = {} vs (e^ε−1)/(e^ε+1) = {p_closed}",
        rr_from_eps.p_true()
    );
    let p = 0.75;
    let mut rr = RandomizedResponse::with_probability(p, 7);
    let n_reports = 8u64;
    let mut k = 0u64;
    let mut bits = Vec::with_capacity(n_reports as usize);
    for i in 0..n_reports {
        let truth = i % 2 == 0;
        let report = rr.privatize(truth);
        bits.push(rr.privatize_bit(u8::from(truth)));
        k += u64::from(report);
    }
    let observed = k as f64 / n_reports as f64;
    println!(
        "[privacy] p_true={} reports={n_reports} positives={k} bits={bits:?} \
         estimate={} (closed form (k/n − (1−p)/2)/p={})",
        rr.p_true(),
        RandomizedResponse::estimate_proportion(p, n_reports, k),
        (observed - (1.0 - p) / 2.0) / p
    );

    // ---- deterministic generator -------------------------------------------
    let mut rng = XorShift64::new(seed);
    let (lo, hi) = (4.0, 8.0);
    let x = rng.next_f64_range(lo, hi);
    let b = rng.next_bool(0.5);
    println!("[privacy] xorshift seed={seed}: next_f64_range({lo},{hi})={x} (in [{lo},{hi})) next_bool(0.5)={b}");

    // ---- RAPPOR ------------------------------------------------------------
    let mut rappor = Rappor::default_params();
    let report = rappor.privatize(12345);
    let ones = report.iter().filter(|&&bit| bit == 1).count();
    println!(
        "[privacy] rappor params={:?} (default 0.5 0.75 0.25) bits={} (BITS={} RAPPOR_BITS={RAPPOR_BITS}) ones={ones}",
        rappor.params(),
        report.len(),
        Rappor::BITS
    );
    let mut identity = Rappor::new(0.0, 1.0, 0.0);
    let bloom = identity.privatize(12345);
    let set: Vec<usize> = (0..RAPPOR_BITS).filter(|&i| bloom[i] == 1).collect();
    println!(
        "[privacy] rappor f=0 p=1 q=0 is the identity on the Bloom filter: set bits {set:?} \
         (3 hash positions, fewer on collision) params={:?}",
        identity.params()
    );
}
