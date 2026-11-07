//! The facts the rows of `scripts/mutants-equivalent.txt` for `src/ccd.rs`,
//! `src/math.rs`, `src/bvh.rs` and the cone-twist joint rest on. An equivalent
//! mutant has to give the same bits for every input, so these compare bits, and
//! include the wrap-around ends of the fixed-point range.
//!
//! `cordic_sin_cos` and `find_split` are private, so their tests repeat the code
//! with each mutation switched on. The function hash in the list ties the copy to
//! the source: a changed function makes its rows stale until someone reads them
//! again.

use alice_physics_core::ccd::swept_aabb;
use alice_physics_core::{Fix128, Vec3Fix, AABB};

struct Seq(u64);

impl Seq {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 ^ (self.0 >> 29)
    }

    /// Any bits, at a magnitude between the fraction only and the whole range.
    fn fix(&mut self) -> Fix128 {
        let shift = self.next() % 64;
        Fix128 {
            hi: (self.next() as i64) >> shift,
            lo: self.next(),
        }
    }
}

const ULP: Fix128 = Fix128 { hi: 0, lo: 1 };
const MIN: Fix128 = Fix128 {
    hi: i64::MIN,
    lo: 0,
};
const MAX: Fix128 = Fix128 {
    hi: i64::MAX,
    lo: u64::MAX,
};

fn sample() -> Vec<Fix128> {
    let mut seq = Seq(0x5eed);
    let mut out = vec![Fix128::ZERO, Fix128::ONE, -Fix128::ONE, MIN, MAX, ULP, -ULP];
    out.extend((0..4000).map(|_| seq.fix()));
    out
}

#[test]
fn equal_compare_means_same_bits() {
    let values = sample();
    for pair in values.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        assert_eq!(a.cmp(&a), core::cmp::Ordering::Equal);
        assert_eq!(
            a.cmp(&b) == core::cmp::Ordering::Equal,
            (a.hi, a.lo) == (b.hi, b.lo),
            "{a:?} {b:?}"
        );
    }
}

#[test]
fn unit_mul_equals_unit_div() {
    let values = sample();
    for chunk in values.chunks_exact(3) {
        let v = Vec3Fix::new(chunk[0], chunk[1], chunk[2]);
        for side in [Fix128::ONE, -Fix128::ONE] {
            assert_eq!(v * side, v / side, "{v:?} {side:?}");
        }
    }
}

#[test]
fn neg_zero_is_zero() {
    assert_eq!(-Fix128::ZERO, Fix128::ZERO);
}

#[test]
fn zero_times_anything_is_zero() {
    for x in sample() {
        assert_eq!(Fix128::ZERO * x, Fix128::ZERO, "{x:?}");
        assert_eq!(x * Fix128::ZERO, Fix128::ZERO, "{x:?}");
        assert_eq!(x - x, Fix128::ZERO, "{x:?}");
    }
}

/// The twist row: at `|t| == limit` both arms of the error give 0, for either sign of
/// `t`. `abs` saturates at the most negative value, where the sum is not 0; the twist
/// angle is `atan2(..).double()`, which is never that value.
#[test]
fn twist_error_at_the_limit_is_zero() {
    let values = sample();
    for pair in values.windows(2) {
        assert_ne!(Fix128::atan2(pair[0], pair[1]).double(), MIN, "{pair:?}");
    }
    for t in values.into_iter().filter(|t| *t != MIN) {
        let limit = t.abs();
        let error = if t > Fix128::ZERO {
            t - limit
        } else {
            t + limit
        };
        assert_eq!(error, Fix128::ZERO, "{t:?}");
    }
}

#[test]
fn union_with_itself_is_the_same_box() {
    let values = sample();
    for chunk in values.chunks_exact(6) {
        let b = AABB {
            min: Vec3Fix::new(chunk[0], chunk[1], chunk[2]),
            max: Vec3Fix::new(chunk[3], chunk[4], chunk[5]),
        };
        let u = b.union(&b);
        assert_eq!((u.min, u.max), (b.min, b.max));
    }
}

/// The row for `t_enter < 0`: an entry time of exactly 0 is returned by the branch
/// before that comparison, so the comparison never sees 0.
#[test]
fn swept_aabb_entering_at_zero_returns_zero() {
    let moving = AABB {
        min: Vec3Fix::from_int(0, 0, 0),
        max: Vec3Fix::from_int(1, 1, 1),
    };
    let target = AABB {
        min: Vec3Fix::from_int(1, 0, 0),
        max: Vec3Fix::from_int(2, 1, 1),
    };
    let got = swept_aabb(&moving, Vec3Fix::from_int(1, 0, 0), &target);
    assert_eq!(got, Some(Fix128::ZERO));
}

#[test]
fn packed_escape_fields_are_disjoint() {
    let mut seq = Seq(0xe5c);
    for _ in 0..4000 {
        let (count, right) = (seq.next() as u32, seq.next() as u32);
        let (a, b) = (count << 24, right & 0x00FF_FFFF);
        assert_eq!(a & b, 0);
        assert_eq!(a | b, a ^ b);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Reduction {
    None,
    EnterAtPi,
    EnterAtMinusPi,
    ClampAbove(ClampAbove),
    ClampBelow(ClampBelow),
    /// The site next to these that is not listed, because only one of the two additions
    /// wraps near the top of the range: the sample has to tell this one apart.
    ShiftByMinusPi,
}

#[derive(Clone, Copy, PartialEq)]
enum ClampAbove {
    Equal,
    GreaterOrEqual,
    AddTwoPi,
    DivideTwoPi,
}

#[derive(Clone, Copy, PartialEq)]
enum ClampBelow {
    Equal,
    LessOrEqual,
}

/// The angle `cordic_sin_cos` hands to its iterations, and whether the cosine is
/// negated: all that the rest of the function reads.
fn quadrant_input(angle: Fix128, m: Reduction) -> (Fix128, bool) {
    let (pi, minus_pi) = (Fix128::PI, -Fix128::PI);
    let mut theta = angle;
    let enter_above = if m == Reduction::EnterAtPi {
        theta >= pi
    } else {
        theta > pi
    };
    let enter_below = if m == Reduction::EnterAtMinusPi {
        theta <= minus_pi
    } else {
        theta < minus_pi
    };
    if enter_above || enter_below {
        let shifted = if m == Reduction::ShiftByMinusPi {
            theta - pi
        } else {
            theta + pi
        };
        let k = shifted / Fix128::TWO_PI;
        let k_int = Fix128::from_int(k.hi);
        theta = theta - Fix128::TWO_PI * k_int;
        let above = match m {
            Reduction::ClampAbove(ClampAbove::Equal) => theta == pi,
            Reduction::ClampAbove(ClampAbove::GreaterOrEqual) => theta >= pi,
            _ => theta > pi,
        };
        let below = match m {
            Reduction::ClampBelow(ClampBelow::Equal) => theta == minus_pi,
            Reduction::ClampBelow(ClampBelow::LessOrEqual) => theta <= minus_pi,
            _ => theta < minus_pi,
        };
        if above {
            theta = match m {
                Reduction::ClampAbove(ClampAbove::AddTwoPi) => theta + Fix128::TWO_PI,
                Reduction::ClampAbove(ClampAbove::DivideTwoPi) => theta / Fix128::TWO_PI,
                _ => theta - Fix128::TWO_PI,
            };
        } else if below {
            theta = theta + Fix128::TWO_PI;
        }
    }
    if theta > Fix128::HALF_PI {
        (pi - theta, true)
    } else if theta < -Fix128::HALF_PI {
        (minus_pi - theta, true)
    } else {
        (theta, false)
    }
}

fn angles() -> Vec<Fix128> {
    let mut out = sample();
    let offsets = [
        Fix128::ZERO,
        ULP,
        -ULP,
        ULP + ULP,
        -(ULP + ULP),
        Fix128 { hi: 0, lo: 1 << 32 },
        -Fix128 { hi: 0, lo: 1 << 32 },
    ];
    let mut turns: Vec<i64> = (-2000..=2000).collect();
    for bit in 11..63 {
        turns.extend([
            1i64 << bit,
            (1i64 << bit) + 1,
            -(1i64 << bit),
            -(1i64 << bit) - 1,
        ]);
    }
    for n in turns {
        let whole = Fix128::TWO_PI * Fix128::from_int(n);
        for base in [Fix128::ZERO, Fix128::PI, -Fix128::PI, Fix128::HALF_PI] {
            for offset in offsets {
                out.push(whole + base + offset);
            }
        }
    }
    for edge in [MIN, MAX] {
        for step in 0..64 {
            let near = Fix128::from_int(step);
            out.extend([
                edge + near,
                edge - near,
                edge + Fix128::PI,
                edge - Fix128::PI,
            ]);
        }
    }
    out
}

/// The rows of the range reduction: each mutation gives the iterations the same bits,
/// including after the additions wrap at the ends of the range.
#[test]
fn cordic_reduction_mutants_give_the_same_quadrant_input() {
    let mutations = [
        Reduction::EnterAtPi,
        Reduction::EnterAtMinusPi,
        Reduction::ClampAbove(ClampAbove::Equal),
        Reduction::ClampAbove(ClampAbove::GreaterOrEqual),
        Reduction::ClampAbove(ClampAbove::AddTwoPi),
        Reduction::ClampAbove(ClampAbove::DivideTwoPi),
        Reduction::ClampBelow(ClampBelow::Equal),
        Reduction::ClampBelow(ClampBelow::LessOrEqual),
    ];
    let all = angles();
    assert!(all.len() > 100_000, "compared {}", all.len());
    assert!(all.contains(&Fix128::PI) && all.contains(&-Fix128::PI));
    assert!(
        all.iter()
            .any(|a| quadrant_input(*a, Reduction::ShiftByMinusPi)
                != quadrant_input(*a, Reduction::None)),
        "the sample does not reach the wrap-around that tells the unlisted site apart"
    );
    for angle in all {
        let want = quadrant_input(angle, Reduction::None);
        for (i, m) in mutations.iter().enumerate() {
            assert_eq!(quadrant_input(angle, *m), want, "mutation {i} at {angle:?}");
        }
    }
}

#[derive(Clone, Copy)]
enum Split {
    None,
    UpperBoundIsEnd,
    FirstBitShiftedLeft,
    LowerClampAtStart,
    UpperClampPastEnd,
    UpperClampAtEnd,
}

fn find_split(codes: &[u64], start: usize, end: usize, m: Split) -> usize {
    let first_code = codes[start];
    let last_code = codes[end - 1];
    if first_code == last_code {
        return (start + end) / 2;
    }
    let diff = first_code ^ last_code;
    let highest_bit = 63 - diff.leading_zeros() as usize;
    let mut lo = start;
    let mut hi = if matches!(m, Split::UpperBoundIsEnd) {
        end
    } else {
        end - 1
    };
    while lo < hi {
        let mid = (lo + hi) / 2;
        let split_bit = (codes[mid] >> highest_bit) & 1;
        let first_bit = if matches!(m, Split::FirstBitShiftedLeft) {
            (first_code << highest_bit) & 1
        } else {
            (first_code >> highest_bit) & 1
        };
        if split_bit == first_bit {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let low = if matches!(m, Split::LowerClampAtStart) {
        start
    } else {
        start + 1
    };
    let high = match m {
        Split::UpperClampPastEnd => end + 1,
        Split::UpperClampAtEnd => end,
        _ => end - 1,
    };
    lo.max(low).min(high)
}

/// The rows of `find_split`, which `build_recursive` calls on codes sorted ascending,
/// for ranges of more than 4 primitives.
#[test]
fn find_split_mutants_give_the_same_split() {
    let mutations = [
        Split::UpperBoundIsEnd,
        Split::FirstBitShiftedLeft,
        Split::LowerClampAtStart,
        Split::UpperClampPastEnd,
        Split::UpperClampAtEnd,
    ];
    let mut seq = Seq(0xb1d);
    let mut compared = 0u32;
    for round in 0..600 {
        let len = 5 + (seq.next() % 40) as usize;
        let keep = [u64::MAX, 0xFF, 0xFFFF_0000_0000_0000, 1, 0b1010][round % 5];
        let mut codes: Vec<u64> = (0..len).map(|_| seq.next() & keep).collect();
        codes.sort_unstable();
        for start in 0..len {
            for end in (start + 5)..=len {
                let want = find_split(&codes, start, end, Split::None);
                compared += u32::from(codes[start] != codes[end - 1]);
                for (i, m) in mutations.iter().enumerate() {
                    assert_eq!(
                        find_split(&codes, start, end, *m),
                        want,
                        "mutation {i} on {codes:?} {start}..{end}"
                    );
                }
            }
        }
    }
    assert!(compared > 10_000, "compared {compared}");
}
