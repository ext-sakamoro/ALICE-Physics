//! Reading a metric: `MetricWeights::{weights, norm, lipschitz}`.
//!
//! A clearance of `r` measured in the cube metric `LINF` is a Euclidean region of
//! radius `r / minimum = sqrt(3) r`. The other direction is bounded by the
//! Lipschitz constant: a Euclidean step of length `s` changes the metric
//! distance by at most `lipschitz * s`, which for `LINF` is exactly `1`.
//!
//! Hand values for the weights `(w1, w2, winf) = (1, 2, 3)`:
//! `lipschitz = sqrt((1+3)^2 + 2*1^2) + 2 = sqrt(18) + 2 = 6.2426407`,
//! `minimum   = min(1*1+3, (2+3)/sqrt2, (3+3)/sqrt3) + 2 = min(4, 3.5355339, 3.4641016) + 2 = 5.4641016`,
//! `norm((1,-2,3)) = 1*6 + 2*sqrt(14) + 3*3 = 22.4833`.
//!
//! ```bash
//! cargo run --release --example metric_clearance_bounds --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::metric::MetricWeights;

fn main() {
    let m = MetricWeights::new(Fix128::ONE, Fix128::from_int(2), Fix128::from_int(3))
        .expect("non-negative weights form a metric");
    let (w1, w2, winf) = m.weights();
    println!(
        "weights ({}, {}, {})",
        w1.to_f64(),
        w2.to_f64(),
        winf.to_f64()
    );
    println!("lipschitz = {:.7}", m.lipschitz().to_f64());
    println!("minimum   = {:.7}", m.minimum().to_f64());
    let v = Vec3Fix::from_int(1, -2, 3);
    println!("norm(1,-2,3) = {:.4}", m.norm(v).to_f64());

    // A 1 m clearance in the cube metric needs a sqrt(3) m Euclidean search radius,
    // while a 1 m Euclidean step moves the cube-metric distance by at most 1.
    let cube = MetricWeights::LINF;
    println!(
        "LINF: euclidean radius of a 1 m clearance = {:.7}, lipschitz = {:.7}",
        cube.euclidean_radius(Fix128::ONE).to_f64(),
        cube.lipschitz().to_f64()
    );
}
