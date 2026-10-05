//! Scalar Helpers Closed Forms Example
//!
//! Production entry point for three small scalar helpers no other example
//! reaches: `Fix128::ceil` (`src/math.rs`), `IdealGas::pressure`
//! (`src/compressible.rs`) and `LoadDirection::length` /
//! `LoadDirection::length_squared` (`src/print_orientation.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - `ceil` leaves integers alone, rounds any fraction up (also for negative
//!   values, where "up" is toward zero), satisfies `ceil(x) = −floor(−x)` for
//!   values whose negation is representable, and saturates at the top of the
//!   range instead of wrapping
//! - the ideal gas law `p = ρ R T` with `R = 287` J/(kg K) for air and
//!   `R = 2077` for helium; `density` and `temperature` invert it; sea-level
//!   air (1.225 kg/m³, 288.15 K) is within 0.02 % of 101 325 Pa
//! - a Pythagorean quadruple `(2, 3, 6)` has length 7 and squared length 49,
//!   and the axis directions have length 1
//!
//! Run with: `cargo run --example scalar_helpers_closed_forms`

use alice_physics::compressible::IdealGas;
use alice_physics::math::Fix128;
use alice_physics::print_orientation::LoadDirection;

fn ceil_rounding() {
    let cases: [(Fix128, Fix128, &str); 7] = [
        (
            Fix128::from_int(3),
            Fix128::from_int(3),
            "an integer is unchanged",
        ),
        (
            Fix128::from_int(-3),
            Fix128::from_int(-3),
            "a negative integer is unchanged",
        ),
        (
            Fix128::from_ratio(9, 4),
            Fix128::from_int(3),
            "2.25 rounds up to 3",
        ),
        (
            Fix128::from_ratio(-9, 4),
            Fix128::from_int(-2),
            "-2.25 rounds up to -2",
        ),
        (Fix128::from_raw(0, 1), Fix128::ONE, "2^-64 rounds up to 1"),
        (
            Fix128::from_raw(-1, u64::MAX),
            Fix128::ZERO,
            "-2^-64 rounds up to 0",
        ),
        (Fix128::ZERO, Fix128::ZERO, "zero is unchanged"),
    ];
    for (x, want, what) in cases {
        assert_eq!(x.ceil(), want, "{what}");
        assert_eq!(x.ceil(), -((-x).floor()), "ceil(x) = -floor(-x): {what}");
        assert!(x.ceil() >= x, "ceil is never below its argument: {what}");
        assert!(
            x.ceil() - x < Fix128::ONE,
            "ceil is less than one above its argument: {what}"
        );
    }

    // Above i64::MAX with a fraction the next integer does not exist: the
    // largest representable value is returned, still >= the argument.
    let top = Fix128::from_raw(i64::MAX, 1);
    assert_eq!(
        top.ceil(),
        Fix128::from_raw(i64::MAX, u64::MAX),
        "saturates"
    );
    assert!(
        top.ceil() >= top,
        "the saturated value is still >= the argument"
    );
    println!(
        "ceil: 2.25 -> {}, -2.25 -> {}, 2^-64 -> {}",
        Fix128::from_ratio(9, 4).ceil().to_f64(),
        Fix128::from_ratio(-9, 4).ceil().to_f64(),
        Fix128::from_raw(0, 1).ceil().to_f64()
    );
}

fn ideal_gas() {
    let air = IdealGas::air();
    let p = air.pressure(Fix128::ONE, Fix128::from_int(300));
    assert_eq!(p, Fix128::from_int(287 * 300), "air: 1 kg/m³ at 300 K");
    let back = air.density(p, Fix128::from_int(300));
    assert!(
        (back - Fix128::ONE).abs() < Fix128::from_raw(0, 1 << 8),
        "density inverts pressure"
    );
    let t = air.temperature(p, Fix128::ONE);
    assert!(
        (t - Fix128::from_int(300)).abs() < Fix128::from_raw(0, 1 << 16),
        "temperature inverts pressure"
    );

    let helium = IdealGas::helium();
    let p_he = helium.pressure(Fix128::from_ratio(1, 2), Fix128::from_int(4));
    assert_eq!(p_he, Fix128::from_int(2077 * 2), "helium: 1/2 kg/m³ at 4 K");

    // Doubling either the density or the temperature doubles the pressure.
    let rho = Fix128::from_ratio(5, 4);
    let temp = Fix128::from_int(250);
    let base = air.pressure(rho, temp);
    assert_eq!(
        air.pressure(rho * Fix128::from_int(2), temp),
        base * Fix128::from_int(2)
    );
    assert_eq!(
        air.pressure(rho, temp * Fix128::from_int(2)),
        base * Fix128::from_int(2)
    );

    let sea_level = air
        .pressure(
            Fix128::from_ratio(1225, 1000),
            Fix128::from_ratio(28815, 100),
        )
        .to_f64();
    let want = 1.225 * 287.0 * 288.15;
    assert!(
        (sea_level - want).abs() <= 1e-9 * want,
        "sea level: got {sea_level}, ρ R T = {want}"
    );
    assert!(
        (sea_level - 101_325.0).abs() / 101_325.0 < 2e-4,
        "sea-level air is within 0.02 % of one standard atmosphere, got {sea_level}"
    );
    println!(
        "ideal gas: air 1 kg/m³ 300 K = {} Pa, sea level = {sea_level:.1} Pa",
        p.to_f64()
    );
}

fn load_direction() {
    let d = LoadDirection {
        x: Fix128::from_int(2),
        y: Fix128::from_int(3),
        z: Fix128::from_int(6),
    };
    assert_eq!(d.length_squared(), Fix128::from_int(49), "2² + 3² + 6²");
    assert_eq!(d.length(), Fix128::from_int(7), "the (2, 3, 6) quadruple");

    let tilted = LoadDirection {
        x: Fix128::from_int(-1),
        y: Fix128::from_int(-4),
        z: Fix128::from_int(8),
    };
    assert_eq!(
        tilted.length(),
        Fix128::from_int(9),
        "the (1, 4, 8) quadruple, signs ignored"
    );

    for (axis, name) in [
        (LoadDirection::axis_x(), "x"),
        (LoadDirection::axis_y(), "y"),
        (LoadDirection::axis_z(), "z"),
    ] {
        assert_eq!(
            axis.length_squared(),
            Fix128::ONE,
            "axis {name}: squared length"
        );
        assert_eq!(axis.length(), Fix128::ONE, "axis {name}: length");
    }
    println!(
        "LoadDirection: |(2,3,6)| = {}, |(-1,-4,8)| = {}",
        d.length().to_f64(),
        tilted.length().to_f64()
    );
}

fn main() {
    ceil_rounding();
    ideal_gas();
    load_direction();
    println!("all closed forms hold");
}
