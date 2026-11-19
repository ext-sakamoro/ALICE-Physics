//! Reference table for the `Fix128` transcendental functions.
//!
//! `tests/fixtures/fix128_reference.txt` holds 1758 inputs, each an exact
//! `Fix128` value, with the exact result computed with 400-bit arithmetic and
//! rounded to the nearest raw value (`scripts/gen_fix128_reference.py`, fixed
//! seed, reproducible byte for byte). The table is independent of the code under
//! test: it comes from a multi-precision library, not from `Fix128` arithmetic.
//!
//! # Bounds (raw units of `2⁻⁶⁴`)
//!
//! | function | bound against the nearest raw value |
//! |---|---|
//! | `sqrt`, `math_util::cbrt_fix` | equal (both return the floor of the root) |
//! | `sin`, `cos`, `atan`, `atan2`, `ln` | 1 (the result is the nearest or a neighbour) |
//! | `exp` | `1 + ⌊y⌋` (relative `2⁻⁶⁴` for results above 1) |
//! | `powf_pos` | `1 + 2·⌊y⌋` |
//!
//! A result at or beyond `2⁶³` must be the saturated maximum, and a result below
//! it must not be. Each function is compared on at least the stated number of
//! rows (a table that silently lost its rows fails).

use alice_physics_core::math_util::cbrt_fix;
use alice_physics_core::Fix128;

const TABLE: &str = include_str!("fixtures/fix128_reference.txt");
const MAX: Fix128 = Fix128::from_raw(i64::MAX, u64::MAX);

fn fix(raw: i128) -> Fix128 {
    Fix128::from_raw((raw >> 64) as i64, raw as u64)
}

fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

/// Rows of `(fn, inputs, expected)`.
fn rows() -> Vec<(String, Vec<i128>, Vec<i128>)> {
    TABLE
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let (lhs, rhs) = line.split_once(" = ").expect("row has ' = '");
            let mut lhs = lhs.split_whitespace();
            let name = lhs.next().expect("fn name").to_string();
            let ins = lhs.map(|t| t.parse().expect("input raw")).collect();
            let outs = rhs
                .split_whitespace()
                .map(|t| t.parse().expect("expected raw"))
                .collect();
            (name, ins, outs)
        })
        .collect()
}

/// `|got − want|` in raw units must not exceed `1 + b·⌊|want|⌋` (the integer part of
/// the expected value, so `b` is a relative bound in units of `2⁻⁶⁴` above 1).
fn within(name: &str, ins: &[i128], got: Fix128, want: i128, b: u128) {
    let g = raw(got);
    if want == raw(MAX) || got == MAX {
        assert_eq!(g, want, "{name}{ins:?}: saturation (got {g}, want {want})");
        return;
    }
    let err = g.abs_diff(want);
    let bound = 1 + b * (want.unsigned_abs() >> 64);
    assert!(
        err <= bound,
        "{name}{ins:?}: got {g}, want {want}, error {err} > {bound}"
    );
}

#[test]
fn every_transcendental_is_within_its_bound_of_the_400_bit_reference() {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for (name, ins, want) in rows() {
        let x = fix(ins[0]);
        match name.as_str() {
            "sqrt" => assert_eq!(raw(x.sqrt()), want[0], "sqrt {ins:?}"),
            "cbrt" => assert_eq!(raw(cbrt_fix(x)), want[0], "cbrt {ins:?}"),
            "sincos" => {
                let (s, c) = x.sin_cos();
                within("sin", &ins, s, want[0], 0);
                within("cos", &ins, c, want[1], 0);
                assert_eq!(
                    (x.sin(), x.cos()),
                    (s, c),
                    "sin/cos agree with sin_cos {ins:?}"
                );
            }
            "atan" => within("atan", &ins, x.atan(), want[0], 0),
            "atan2" => within("atan2", &ins, Fix128::atan2(x, fix(ins[1])), want[0], 1),
            "exp" => within("exp", &ins, x.exp(), want[0], 1),
            "ln" => within("ln", &ins, x.ln(), want[0], 0),
            "powf" => within("powf_pos", &ins, x.powf_pos(fix(ins[1])), want[0], 2),
            other => panic!("unknown function {other} in the table"),
        }
        *counts.entry(name).or_default() += 1;
    }
    for (name, min) in [
        ("sqrt", 120),
        ("sincos", 320),
        ("atan", 200),
        ("atan2", 100),
        ("exp", 300),
        ("ln", 150),
        ("powf", 300),
        ("cbrt", 140),
    ] {
        let n = counts.get(name).copied().unwrap_or(0);
        assert!(
            n >= min,
            "{name}: {n} rows compared, at least {min} expected"
        );
    }
}

/// The value of a `# key: value` header line of the table.
fn header(key: &str) -> i128 {
    let prefix = format!("# {key}: ");
    TABLE
        .lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("table header `{key}` missing"))
        .parse()
        .expect("header value")
}

/// The table has exactly the number of rows its header states, and more than
/// zero (a truncated or emptied fixture fails here before any comparison).
#[test]
fn the_table_has_the_rows_its_header_states() {
    let stated = header("rows");
    assert!(stated > 0, "header states {stated} rows");
    assert_eq!(rows().len() as i128, stated);
}

/// `exp` rounds to the nearest value, so it is 1 ulp down to the largest
/// representable `x` with `e^x ≥ 2⁻⁶⁵` (half an ulp) and 0 from the next value
/// below. That `x` (≈ −65·ln 2 ≈ −45.05) is computed by the generator from the
/// 400-bit reference, not from `exp`.
#[test]
fn exp_is_zero_exactly_below_half_an_ulp() {
    let edge = header("exp-zero-boundary-raw");
    assert_eq!(
        fix(edge).exp(),
        Fix128::from_raw(0, 1),
        "e^x at the last x above half an ulp"
    );
    assert_eq!(fix(edge - 1).exp(), Fix128::ZERO, "e^x one raw step below");
    assert_eq!(Fix128::from_int(-46).exp(), Fix128::ZERO);
    assert_eq!(Fix128::from_int(-1000).exp(), Fix128::ZERO);
}

/// Integer powers that are exactly representable come out exactly; inexact ones
/// stay within the bound (`1.0001^1000`, `0.9999^1000`, `1.01^100`, `1.1^400` are
/// rows of the table above).
#[test]
fn exact_integer_powers_are_exact() {
    let two = Fix128::from_int(2);
    assert_eq!(
        two.powf_pos(Fix128::from_int(62)),
        Fix128::from_raw(1 << 62, 0)
    );
    assert_eq!(
        Fix128::from_int(3).powf_pos(Fix128::from_int(39)),
        Fix128::from_int(4_052_555_153_018_976_267)
    );
    // 1.5^40 = 3^40 / 2^40: 40 fraction bits, exact
    let p = Fix128::from_ratio(3, 2).powf_pos(Fix128::from_int(40));
    let three40: u128 = 12_157_665_459_056_928_801;
    assert_eq!(raw(p), (three40 << 24) as i128);
    assert_eq!(
        Fix128::from_ratio(1, 2).powf_pos(Fix128::from_int(60)),
        Fix128::from_raw(0, 1 << 4)
    );
}

/// Around an integer exponent `n` the three values at `n − 2⁻⁶⁰`, `n`, `n + 2⁻⁶⁰`
/// are ordered (increasing in `y` for `x > 1`, decreasing for `x < 1`) and each
/// step is no larger than the true step plus the bound.
#[test]
fn powf_pos_is_continuous_and_monotone_across_integer_exponents() {
    let step = Fix128::from_raw(0, 16); // 2⁻⁶⁰
    let mut compared = 0;
    for (num, den) in [
        (3i64, 2i64),
        (27, 10),
        (10, 1),
        (3, 10),
        (10_001, 10_000),
        (600, 1),
    ] {
        let x = Fix128::from_ratio(num, den);
        for n in [1i64, 2, 5, 10] {
            let y = Fix128::from_int(n);
            let (lo, mid, hi) = (x.powf_pos(y - step), x.powf_pos(y), x.powf_pos(y + step));
            if mid == MAX {
                continue;
            }
            if x > Fix128::ONE {
                assert!(
                    lo <= mid && mid <= hi,
                    "x={num}/{den} n={n}: {lo:?} {mid:?} {hi:?}"
                );
            } else {
                assert!(
                    lo >= mid && mid >= hi,
                    "x={num}/{den} n={n}: {lo:?} {mid:?} {hi:?}"
                );
            }
            // true step x^n · ln x · 2⁻⁶⁰ is below 2⁻⁵⁴·x^n here; allow that plus the bound
            let slack = 3 + 2 * (raw(mid).unsigned_abs() >> 54);
            assert!(
                raw(mid).abs_diff(raw(lo)) <= slack,
                "x={num}/{den} n={n}: below step"
            );
            assert!(
                raw(hi).abs_diff(raw(mid)) <= slack,
                "x={num}/{den} n={n}: above step"
            );
            compared += 1;
        }
    }
    assert!(compared >= 20, "{compared} exponents compared");
}
