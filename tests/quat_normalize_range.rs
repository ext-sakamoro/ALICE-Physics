//! `QuatFix::normalize` and `unit_rotation` for quaternions of any size
//!
//! A rotation does not depend on the scale of its quaternion. When the largest component lies in
//! `[2^-16, 2^30]` the quaternion is normalized as before (bit for bit, checked against the
//! previous body written out here); outside it, the squared length would wrap or round to zero,
//! so the quaternion is scaled by a power of two first. The expected values are closed forms.
#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "raw Fix128 words are built from and split into i128 and u64 on purpose"
)]

use alice_physics_core::math::{Fix128, QuatFix};

/// the body of `normalize` before sizes outside `[2^-16, 2^30]` were scaled
fn previous_normalize(q: QuatFix) -> QuatFix {
    let len = q.length();
    if len.is_zero() {
        QuatFix::IDENTITY
    } else {
        let inv_len = Fix128::ONE / len;
        QuatFix::new(q.x * inv_len, q.y * inv_len, q.z * inv_len, q.w * inv_len)
    }
}

fn raw(r: i128) -> Fix128 {
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

fn bits(q: QuatFix) -> [(i64, u64); 4] {
    [
        (q.x.hi, q.x.lo),
        (q.y.hi, q.y.lo),
        (q.z.hi, q.z.lo),
        (q.w.hi, q.w.lo),
    ]
}

/// |a - b| in units of 2^-64
fn ulps(a: Fix128, b: Fix128) -> u128 {
    let d = ((i128::from(a.hi) << 64) | i128::from(a.lo))
        - ((i128::from(b.hi) << 64) | i128::from(b.lo));
    d.unsigned_abs()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// a raw component with its top bit at `bit` (bit 64 is 1.0), random sign and lower bits
    fn component(&mut self, bit: u32) -> i128 {
        let low = (u128::from(self.next()) << 64 | u128::from(self.next())) & ((1u128 << bit) - 1);
        let m = (1u128 << bit) | low;
        if self.next() & 1 == 0 {
            m as i128
        } else {
            -(m as i128)
        }
    }
}

#[test]
fn inside_the_plain_sizes_the_bits_are_the_previous_ones() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut compared = 0;
    for _ in 0..20_000 {
        // largest component's top bit in 48..=93 (2^-16 ..< 2^30), the others at or below it
        let top = 48 + (rng.next() % 46) as u32;
        let mut c = [0i128; 4];
        c[(rng.next() % 4) as usize] = rng.component(top);
        for v in &mut c {
            if *v == 0 {
                let bit = 1 + (rng.next() % u64::from(top)) as u32;
                *v = rng.component(bit);
            }
        }
        let q = QuatFix::new(raw(c[0]), raw(c[1]), raw(c[2]), raw(c[3]));
        assert_eq!(bits(q.normalize()), bits(previous_normalize(q)), "{q:?}");
        compared += 1;
    }
    // the ends of the plain sizes are inside it
    for m in [1i128 << 48, 1i128 << 94] {
        let q = QuatFix::new(raw(m), raw(m / 3), raw(-m / 5), raw(m / 7));
        assert_eq!(bits(q.normalize()), bits(previous_normalize(q)));
        compared += 1;
    }
    assert_eq!(compared, 20_002);
}

#[test]
fn outside_the_plain_sizes_the_closed_forms_hold() {
    // an f64 reference carries 53 bits: within 2^12 of 2^-64 covers its rounding
    let close = |got: Fix128, want: f64| {
        let w = Fix128::from_f64(want);
        assert!(ulps(got, w) <= 1 << 12, "{got:?} vs {want}");
    };
    let exact =
        |got: Fix128, want: Fix128| assert!(ulps(got, want) <= 1 << 8, "{got:?} vs {want:?}");
    // (3, 4, 0, 0) at 2^33 and at 2^-40: the unit quaternion (0.6, 0.8, 0, 0)
    for shift in [33i32, 60, -40, -60] {
        let s = |v: i128| {
            raw(if shift >= 0 {
                (v << 64) << shift
            } else {
                (v << 64) >> -shift
            })
        };
        let q = QuatFix::new(s(3), s(4), s(0), s(0)).normalize();
        exact(q.x, Fix128::from_ratio(3, 5));
        exact(q.y, Fix128::from_ratio(4, 5));
        assert_eq!(bits(q)[2], (0, 0));
        assert_eq!(bits(q)[3], (0, 0));
    }
    // a large axis quaternion is that axis, within the rounding of 1 / len and one product
    // (bringing the size to 2^29 instead of 1 would leave 1 / len 35 bits: x = 1 - 2.5e-11)
    let q = QuatFix::new(
        Fix128::from_int(1_000_000_000_000),
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
    )
    .normalize();
    exact(q.x, Fix128::ONE);
    assert_eq!(bits(q)[1..], [(0, 0); 3]);
    // a tiny half turn about x and w: each component 1/sqrt(2)
    let t = raw(1i128 << 20); // 2^-44
    let q = QuatFix::new(t, Fix128::ZERO, Fix128::ZERO, t).normalize();
    close(q.x, core::f64::consts::FRAC_1_SQRT_2);
    close(q.w, core::f64::consts::FRAC_1_SQRT_2);
    // the zero quaternion is the identity, as before
    assert_eq!(
        QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO).normalize(),
        QuatFix::IDENTITY
    );
}

#[test]
fn outside_the_plain_sizes_every_power_of_two_gives_the_same_bits_and_a_unit_length() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut compared = 0;
    for _ in 0..2_000 {
        // components that fit 40 bits, so every scaling below loses no bit
        let c: [i128; 4] = core::array::from_fn(|_| (rng.next() >> 24) as i128 - (1 << 39));
        if c.iter().all(|v| *v == 0) {
            continue;
        }
        let at = |shift: i32| {
            let s = |v: i128| raw(if shift >= 0 { v << shift } else { v >> -shift });
            QuatFix::new(s(c[0]), s(c[1]), s(c[2]), s(c[3])).normalize()
        };
        // the components as raw values times 2^shift: largest above 2^30 for shift 60 and 80,
        // below 2^-16 for shift 0 (raw < 2^40, so below 2^-24)
        let (a, b, d) = (at(60), at(80), at(0));
        assert_eq!(bits(a), bits(b));
        assert_eq!(bits(a), bits(d));
        // unit length: |q|^2 - 1 within the rounding of four products
        let len2 = a.length_squared();
        assert!(ulps(len2, Fix128::ONE) <= 1 << 8, "{len2:?}");
        compared += 1;
    }
    assert!(compared > 1_900);
}

#[test]
fn unit_rotation_normalizes_outside_the_plain_sizes() {
    // 2^40 on every component: the squared length 2^82 wraps, so it must not be read as near 1
    let big = raw(1i128 << 104);
    let q = QuatFix::new(big, big, big, big);
    let u = q.unit_rotation();
    assert_eq!(bits(u), bits(q.normalize()));
    assert!(ulps(u.x, Fix128::from_ratio(1, 2)) <= 1 << 8);
    // (2^32, 0, 0, 1): the squared length 2^64 + 1 wraps to exactly 1, which used to read as
    // already unit and be kept unnormalized
    let wraps = QuatFix::new(
        Fix128::from_int(1 << 32),
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ONE,
    );
    assert_eq!(wraps.length_squared(), Fix128::ONE);
    let u = wraps.unit_rotation();
    assert_eq!(bits(u), bits(wraps.normalize()));
    assert!(ulps(u.x, Fix128::ONE) <= 1 << 8, "{u:?}");
    // a unit quaternion inside the plain sizes is left bit for bit
    let half = Fix128::from_ratio(1, 2);
    let unit = QuatFix::new(half, half, half, half);
    assert_eq!(bits(unit.unit_rotation()), bits(unit));
}
