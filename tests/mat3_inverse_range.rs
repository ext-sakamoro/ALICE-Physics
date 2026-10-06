//! `Mat3Fix::inverse` の範囲外の判定
//!
//! 旧実装は `det` / 余因子 / `1/det` / 最終の積を全て wrap する演算で作っていたので、
//! `|det| ≥ 2⁶³` や `|det| ≤ 2⁻⁶³` で **符号の反転した逆行列を `Some` で返した**
//! 本 file は次を固定する
//!
//! - 仕様: 余因子・`det`・`1/det`・各成分の積のどれかが Q64.64 に収まらなければ `None`
//!   (`det` と余因子は「積の floor の和」を厳密に計算して収まるかを見る 途中の積が
//!   単独で範囲を超えても和が収まれば旧実装の結果は正しいので、そこは `Some` のまま)
//! - 範囲内は旧実装と bit 一致 (旧式を本 file に写して `assert_eq!`)
//! - 範囲内の可逆な行列は `M·M⁻¹ ≈ I` (旧式とは独立の検算)
//!
//! 厳密な参照値は u32 limb の 256 bit 整数で計算する (src 側とは別の書き方)

use alice_physics::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};

// ---------------------------------------------------------------------------
// 256 bit の厳密演算 (2 の補数、little endian u64 × 4)
// ---------------------------------------------------------------------------

type W = [u64; 4];

fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

fn from_raw(r: i128) -> Fix128 {
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

fn w_add(a: W, b: W) -> W {
    let mut out = [0u64; 4];
    let mut carry = 0u128;
    for i in 0..4 {
        let s = a[i] as u128 + b[i] as u128 + carry;
        out[i] = s as u64;
        carry = s >> 64;
    }
    out
}

fn w_neg(a: W) -> W {
    w_add([!a[0], !a[1], !a[2], !a[3]], [1, 0, 0, 0])
}

/// `floor(a·b / 2⁶⁴)` (raw 単位、つまり `Fix128` の積の真値の floor)
fn w_prod(a: Fix128, b: Fix128) -> W {
    let (ra, rb) = (raw(a), raw(b));
    let neg = (ra < 0) != (rb < 0);
    let limbs = |u: u128| {
        [
            u as u32,
            (u >> 32) as u32,
            (u >> 64) as u32,
            (u >> 96) as u32,
        ]
    };
    let (la, lb) = (limbs(ra.unsigned_abs()), limbs(rb.unsigned_abs()));
    let mut acc = [0u64; 9];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let t = la[i] as u64 * lb[j] as u64 + acc[i + j] + carry;
            acc[i + j] = t & 0xFFFF_FFFF;
            carry = t >> 32;
        }
        acc[i + 4] += carry;
    }
    let mut p: W = [0; 4];
    for k in 0..4 {
        p[k] = acc[2 * k] | (acc[2 * k + 1] << 32);
    }
    if neg {
        p = w_neg(p);
    }
    // 算術右 shift 64 = floor
    let s = if (p[3] as i64) < 0 { u64::MAX } else { 0 };
    [p[1], p[2], p[3], s]
}

fn w_fit(a: W) -> Option<Fix128> {
    let s = if (a[1] as i64) < 0 { u64::MAX } else { 0 };
    (a[2] == s && a[3] == s).then(|| from_raw(((a[1] as u128) << 64 | a[0] as u128) as i128))
}

/// `floor(p) - floor(q)` が収まるか (余因子 1 つ)
fn cof(a: Fix128, b: Fix128, c: Fix128, d: Fix128) -> Option<Fix128> {
    w_fit(w_add(w_prod(a, b), w_neg(w_prod(c, d))))
}

/// 仕様: 厳密に計算して範囲内なら旧式の値、どこかが範囲外なら `None`
fn spec_inverse(m: Mat3Fix) -> Option<Mat3Fix> {
    let (a, b, c) = (m.col0, m.col1, m.col2);
    let c00 = cof(b.y, c.z, b.z, c.y)?;
    let c01 = cof(a.z, c.y, a.y, c.z)?;
    let c02 = cof(a.y, b.z, a.z, b.y)?;
    let c10 = cof(b.z, c.x, b.x, c.z)?;
    let c11 = cof(a.x, c.z, a.z, c.x)?;
    let c12 = cof(a.z, b.x, a.x, b.z)?;
    let c20 = cof(b.x, c.y, b.y, c.x)?;
    let c21 = cof(a.y, c.x, a.x, c.y)?;
    let c22 = cof(a.x, b.y, a.y, b.x)?;
    // det の第 2 項の余因子は旧式どおり `a.y·c.z − a.z·c.y` で別に作る
    let n = cof(a.y, c.z, a.z, c.y)?;
    let det = w_fit(w_add(
        w_add(w_prod(a.x, c00), w_neg(w_prod(b.x, n))),
        w_prod(c.x, c02),
    ))?;
    if det.is_zero() {
        return None;
    }
    // 1/det は |raw(det)| ≥ 3 で Q64.64 に収まる (2⁻⁶³ 以下は収まらない)
    if raw(det).unsigned_abs() <= 2 {
        return None;
    }
    let inv = Fix128::ONE / det;
    let e = |x: Fix128| w_fit(w_prod(x, inv));
    Some(Mat3Fix::from_cols(
        Vec3Fix::new(e(c00)?, e(c01)?, e(c02)?),
        Vec3Fix::new(e(c10)?, e(c11)?, e(c12)?),
        Vec3Fix::new(e(c20)?, e(c21)?, e(c22)?),
    ))
}

/// 修正前の `Mat3Fix::inverse` と `Mat3Fix::determinant` の写し (wrap する演算のまま)
fn old_inverse(m: Mat3Fix) -> Option<Mat3Fix> {
    let det = m.col0.x * (m.col1.y * m.col2.z - m.col1.z * m.col2.y)
        - m.col1.x * (m.col0.y * m.col2.z - m.col0.z * m.col2.y)
        + m.col2.x * (m.col0.y * m.col1.z - m.col0.z * m.col1.y);
    if det.is_zero() {
        return None;
    }
    let inv_det = Fix128::ONE / det;
    let c00 = m.col1.y * m.col2.z - m.col1.z * m.col2.y;
    let c01 = m.col0.z * m.col2.y - m.col0.y * m.col2.z;
    let c02 = m.col0.y * m.col1.z - m.col0.z * m.col1.y;
    let c10 = m.col1.z * m.col2.x - m.col1.x * m.col2.z;
    let c11 = m.col0.x * m.col2.z - m.col0.z * m.col2.x;
    let c12 = m.col0.z * m.col1.x - m.col0.x * m.col1.z;
    let c20 = m.col1.x * m.col2.y - m.col1.y * m.col2.x;
    let c21 = m.col0.y * m.col2.x - m.col0.x * m.col2.y;
    let c22 = m.col0.x * m.col1.y - m.col0.y * m.col1.x;
    Some(Mat3Fix::from_cols(
        Vec3Fix::new(c00 * inv_det, c01 * inv_det, c02 * inv_det),
        Vec3Fix::new(c10 * inv_det, c11 * inv_det, c12 * inv_det),
        Vec3Fix::new(c20 * inv_det, c21 * inv_det, c22 * inv_det),
    ))
}

// ---------------------------------------------------------------------------
// 入力の生成 (決定的 LCG)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = self.0;
        (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (x >> 32)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// 符号付き、raw の bit 長 `bits` (≤ 127) の乱数
    fn fix_bits(&mut self, bits: u32) -> Fix128 {
        let u = ((self.next() as u128) << 64) | self.next() as u128;
        let mag = if bits == 0 { 0 } else { u >> (128 - bits) };
        let r = mag as i128;
        from_raw(if self.next() & 1 == 1 { -r } else { r })
    }
    /// 2⁻⁴⁰ .. 2⁴⁰ 付近の値
    fn fix_scaled(&mut self, center_bits: u32, spread: u32) -> Fix128 {
        let lo = center_bits.saturating_sub(spread);
        let b = lo + self.below(u64::from(2 * spread + 1)) as u32;
        self.fix_bits(b.min(127))
    }
}

fn pow2(n: i32) -> Fix128 {
    if n >= 0 {
        Fix128::from_raw(1i64 << n, 0)
    } else {
        Fix128::from_raw(0, 1u64 << (64 + n))
    }
}

fn diag(x: Fix128, y: Fix128, z: Fix128) -> Mat3Fix {
    Mat3Fix::diagonal(x, y, z)
}

fn rotation(r: &mut Lcg) -> Mat3Fix {
    let axis = Vec3Fix::new(r.fix_bits(65), r.fix_bits(65), r.fix_bits(65));
    let axis = if axis.length_squared().is_zero() {
        Vec3Fix::UNIT_X
    } else {
        axis.normalize()
    };
    let q = QuatFix::from_axis_angle(axis, r.fix_bits(66));
    Mat3Fix::from_cols(
        q.rotate_vec(Vec3Fix::UNIT_X),
        q.rotate_vec(Vec3Fix::UNIT_Y),
        q.rotate_vec(Vec3Fix::UNIT_Z),
    )
}

/// 全入力 (≥ 10k) 範囲の両端・境界付近・回転・対角・特異を含む
fn corpus() -> Vec<Mat3Fix> {
    let mut r = Lcg(0x1234_5678_9ABC_DEF0);
    let mut v = Vec::new();
    // 1. 成分ごとに bit 長が全域でばらばら
    for _ in 0..4000 {
        let mut f = || {
            let b = r.below(128) as u32;
            r.fix_bits(b)
        };
        v.push(Mat3Fix::from_cols(
            Vec3Fix::new(f(), f(), f()),
            Vec3Fix::new(f(), f(), f()),
            Vec3Fix::new(f(), f(), f()),
        ));
    }
    // 2. 全成分が同じ大きさ (2⁻⁴⁴ .. 2⁴⁴、境界 2²¹ / 2³¹·⁵ / 2⁻²¹ の付近を含む)
    for center in [20u32, 43, 50, 64, 74, 84, 85, 86, 95, 96, 97, 108] {
        for _ in 0..400 {
            let mut f = || r.fix_scaled(center, 1);
            v.push(Mat3Fix::from_cols(
                Vec3Fix::new(f(), f(), f()),
                Vec3Fix::new(f(), f(), f()),
                Vec3Fix::new(f(), f(), f()),
            ));
        }
    }
    // 3. 回転とその定数倍
    for _ in 0..800 {
        let m = rotation(&mut r);
        let s = r.fix_scaled(64, 40);
        v.push(m);
        v.push(m.scale(s.abs()));
    }
    // 4. 対角 (境界の両側を密に)
    for _ in 0..1200 {
        let b = 30 + r.below(98) as u32;
        let x = r.fix_bits(b);
        let y = r.fix_bits(b);
        let z = r.fix_bits(b);
        v.push(diag(x, y, z));
    }
    for k in -24..=24 {
        let s = pow2(k);
        v.push(diag(s, s, s));
        v.push(diag(-s, s, s));
        v.push(diag(s + Fix128::from_raw(0, 1), s, s));
    }
    // 5. 特異 (det = 0 厳密: 整数成分で第 3 列 = 第 1 列 + 第 2 列)
    for _ in 0..300 {
        let mut i = || Fix128::from_int(r.below(2001) as i64 - 1000);
        let c0 = Vec3Fix::new(i(), i(), i());
        let c1 = Vec3Fix::new(i(), i(), i());
        v.push(Mat3Fix::from_cols(c0, c1, c0 + c1));
    }
    v.push(Mat3Fix::ZERO);
    // 6. 積が単独では範囲を超えるが和は収まる (旧式でも正しい)
    for k in 30..40 {
        let a = pow2(k);
        let e = pow2(k - 8);
        v.push(Mat3Fix::from_cols(
            Vec3Fix::new(a, a, Fix128::ZERO),
            Vec3Fix::new(a, a + Fix128::ONE, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, e),
        ));
    }
    // 7. 負の窓 (x² < 2⁶³ だが整数部の floor は 2⁶³ を越える)
    let w = Fix128::from_raw(-3_037_000_500, 1 << 63); // −3037000499.5
    v.push(diag(w, w, Fix128::ONE));
    v.push(diag(Fix128::ONE, w, w));
    v.push(diag(w, Fix128::ONE, w));
    v
}

// ---------------------------------------------------------------------------
// (1) det が wrap する境界 ⇒ None
// ---------------------------------------------------------------------------

#[test]
fn det_at_or_above_two_pow_63_is_none() {
    // diag(2²¹): det = 2⁶³ (旧: −2⁶³ に wrap し、逆行列の対角が −2⁻²¹)
    let s = pow2(21);
    assert_eq!(diag(s, s, s).inverse(), None);
    // 余因子は収まり det だけが範囲外: diag(2³², 2³¹, 1) は det = 2⁶³
    assert_eq!(diag(pow2(32), pow2(31), Fix128::ONE).inverse(), None);
    // 負の側: det = −2⁶³ − 2⁻⁶⁴ 相当まで
    let n = -pow2(21);
    assert_eq!(diag(n, s, s + Fix128::from_raw(0, 1)).inverse(), None);
    // 成分 ≈ 2³¹·⁵: 余因子 x² が範囲外
    let a = Fix128::from_int(3_037_000_500);
    assert_eq!(diag(a, a, Fix128::ONE).inverse(), None);
    assert_eq!(diag(Fix128::ONE, a, a).inverse(), None);
    assert_eq!(diag(a, Fix128::ONE, a).inverse(), None);
    // 負の窓のすぐ外: −3037000500 の 2 乗は 2⁶³ を越える
    let w = Fix128::from_int(-3_037_000_500);
    assert_eq!(diag(w, w, Fix128::ONE).inverse(), None);
}

#[test]
fn old_formula_returned_sign_flipped_inverse_at_the_boundary() {
    // 修正前の挙動の記録 (旧式の写しで再現): 範囲外なのに Some で符号が反転する
    let s = pow2(21);
    let old = old_inverse(diag(s, s, s)).expect("old returned Some");
    assert_eq!(old.col0.x, -pow2(-21));
    let t = pow2(-21);
    let old = old_inverse(diag(t, t, t)).expect("old returned Some");
    assert_eq!(old.col0.x, -pow2(21));
}

// ---------------------------------------------------------------------------
// (2) det が小さすぎて 1/det が収まらない ⇒ None
// ---------------------------------------------------------------------------

#[test]
fn tiny_det_whose_reciprocal_overflows_is_none() {
    // diag(2⁻²¹): det = 2⁻⁶³ = raw 2、1/det = 2⁶³ は収まらない
    let t = pow2(-21);
    assert_eq!(diag(t, t, t).inverse(), None);
    // det = raw 1 (= 2⁻⁶⁴)
    assert_eq!(diag(pow2(-32), pow2(-32), Fix128::ONE).inverse(), None);
    // det = −2⁻⁶³
    assert_eq!(diag(-pow2(-32), pow2(-31), Fix128::ONE).inverse(), None);
    // raw 3 は 1/det ≈ 6.1e18 < 2⁶³ で収まる: 余因子 × 1/det が収まる限り Some
    let three = Fix128::from_raw(0, 3);
    let inv = diag(three, Fix128::ONE, Fix128::ONE)
        .inverse()
        .expect("|det| = 3·2⁻⁶⁴ is invertible");
    assert_eq!(inv.col0.x, Fix128::ONE / three);
}

// ---------------------------------------------------------------------------
// (3) 仕様との一致 + 範囲内は旧式と bit 一致
// ---------------------------------------------------------------------------

#[test]
fn inverse_matches_the_exact_spec_and_old_formula_in_range() {
    let corpus = corpus();
    assert!(corpus.len() >= 10_000, "corpus {}", corpus.len());
    let (mut some, mut none, mut old_some_out_of_range, mut singular) =
        (0usize, 0usize, 0usize, 0usize);
    for m in &corpus {
        let new = m.inverse();
        let spec = spec_inverse(*m);
        assert_eq!(new, spec, "spec mismatch for {m:?}");
        let old = old_inverse(*m);
        match spec {
            Some(_) => {
                assert_eq!(new, old, "in-range result changed for {m:?}");
                some += 1;
            }
            None => {
                none += 1;
                if old.is_some() {
                    old_some_out_of_range += 1;
                } else {
                    singular += 1;
                }
            }
        }
    }
    eprintln!("corpus {} in-range {some} old-some-out-of-range {old_some_out_of_range} old-none {singular} none {none}", corpus.len());
    // 各領域が十分に踏まれていること (空振り防止)
    assert!(some >= 4_500, "in-range {some}");
    assert!(
        old_some_out_of_range >= 1_000,
        "old Some but out of range {old_some_out_of_range}"
    );
    assert!(singular >= 300, "old None {singular}");
    assert!(none >= 1_000, "none {none}");
}

#[test]
fn intermediate_products_that_cancel_stay_in_range() {
    // 個々の積は 2⁶³ を越えるが余因子 / det は収まる: 旧式の値が正しいので Some のまま
    let a = pow2(40);
    let m = Mat3Fix::from_cols(
        Vec3Fix::new(a, a, Fix128::ZERO),
        Vec3Fix::new(a, a + Fix128::ONE, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
    );
    let inv = m.inverse().expect("det = 2⁴⁰");
    assert_eq!(Some(inv), old_inverse(m));
    // 負の窓: x² < 2⁶³ なので範囲内
    let w = Fix128::from_raw(-3_037_000_500, 1 << 63);
    let m = diag(w, w, Fix128::ONE);
    let inv = m.inverse().expect("(−3037000499.5)² < 2⁶³");
    assert_eq!(Some(inv), old_inverse(m));
}

/// `Fix128::checked_mul` の範囲内の偽陽性 (中央 128 bit の和の i128 wrap で `None`)
///
/// `−2⁶³ + (1 − 2⁻⁶⁴)` と `−2⁻⁶⁴` の積は約 0.5 で収まるが、`hl + lh` が i128 を
/// 越えるので `None` になる `Mat3Fix::inverse` はこのため `checked_mul` を使わず
/// 積を 256 bit で厳密に計算する (本 test は `checked_mul` 側の未修正の記録)
#[test]
#[ignore = "src gap: checked_mul returns None for an in-range product when hl + lh wraps i128"]
fn checked_mul_has_no_false_positive_when_the_middle_sum_wraps() {
    for (a, b) in [
        (
            Fix128::from_raw(i64::MIN, 1 << 63),
            Fix128::from_raw(-1, u64::MAX),
        ),
        (
            Fix128::from_raw(i64::MIN, u64::MAX),
            Fix128::from_raw(-1, u64::MAX),
        ),
    ] {
        let exact = w_fit(w_prod(a, b));
        assert!(exact.is_some());
        assert_eq!(a.checked_mul(b), exact, "{a:?} * {b:?}");
    }
}

#[test]
fn checked_mul_agrees_with_the_exact_product() {
    // checked_mul と厳密な積の突合 (中央の和が wrap する窓は上の ignore test)
    let mut r = Lcg(42);
    let mut edges = vec![
        (
            Fix128::from_raw(-3_037_000_500, 1 << 63),
            Fix128::from_raw(-3_037_000_500, 1 << 63),
        ),
        (Fix128::from_raw(i64::MIN, 1 << 63), Fix128::from_raw(-1, 1)),
        (
            Fix128::from_raw(i64::MIN, u64::MAX),
            Fix128::from_raw(-1, 1),
        ),
        (Fix128::from_raw(i64::MIN, 0), Fix128::from_raw(-1, 0)),
        (
            Fix128::from_raw(i64::MAX, u64::MAX),
            Fix128::from_raw(1, u64::MAX),
        ),
        (Fix128::from_raw(i64::MIN, 0), Fix128::from_raw(0, 1 << 63)),
    ];
    for _ in 0..50_000 {
        let (ba, bb) = (r.below(128) as u32, r.below(128) as u32);
        edges.push((r.fix_bits(ba), r.fix_bits(bb)));
    }
    for (a, b) in edges {
        assert_eq!(a.checked_mul(b), w_fit(w_prod(a, b)), "{a:?} * {b:?}");
        if let Some(p) = a.checked_mul(b) {
            assert_eq!(p, a * b);
        }
    }
}

// ---------------------------------------------------------------------------
// (4) M·M⁻¹ ≈ I (独立の検算)
// ---------------------------------------------------------------------------

#[test]
fn product_with_inverse_is_identity_for_well_conditioned_matrices() {
    let mut r = Lcg(7);
    let mut checked = 0;
    for i in 0..3000 {
        let m = if i % 2 == 0 {
            // 対角優位: 成分 |x| < 1 + 4·I
            let mut f = || r.fix_bits(64);
            Mat3Fix::from_cols(
                Vec3Fix::new(f() + Fix128::from_int(4), f(), f()),
                Vec3Fix::new(f(), f() + Fix128::from_int(4), f()),
                Vec3Fix::new(f(), f(), f() + Fix128::from_int(4)),
            )
        } else {
            rotation(&mut r)
        };
        // 2⁻¹⁰ .. 2¹⁰ 倍
        let k = r.below(21) as i32 - 10;
        let s = pow2(k);
        let ms = m.scale(s);
        let inv = ms.inverse().expect("well conditioned");
        let p = ms.mul_mat(inv);
        let id = Mat3Fix::IDENTITY;
        // 1/det の丸め (2⁻⁶⁴) が余因子倍され、さらに M の成分倍される
        let max_abs = |a: Mat3Fix| {
            [a.col0, a.col1, a.col2]
                .iter()
                .flat_map(|c| [c.x, c.y, c.z])
                .map(|x| x.to_f64().abs())
                .fold(0.0f64, f64::max)
        };
        // 一次の誤差評価 (ε = 2⁻⁶⁴): 余因子の誤差 2ε、det の誤差 δ ≲ (6·|m| + 3)·ε、
        // 1/det の誤差 ≲ δ/det² + ε、成分 = 余因子 × 1/det、行の和で 3·|m| 倍、余裕 4 倍
        let m_abs = max_abs(ms);
        let eps = 1.0 / 18_446_744_073_709_551_616.0; // 2⁻⁶⁴
        let inv_det = 1.0 / ms.determinant().to_f64().abs();
        let c_abs = 2.0 * m_abs * m_abs;
        let delta = (6.0 * m_abs + 3.0) * eps;
        let entry_err = 2.0 * eps * inv_det + c_abs * (delta * inv_det * inv_det + eps) + eps;
        let tol = 4.0 * 3.0 * m_abs * entry_err + 16.0 * eps;
        for (x, y) in [(p.col0, id.col0), (p.col1, id.col1), (p.col2, id.col2)] {
            for (u, w) in [(x.x, y.x), (x.y, y.y), (x.z, y.z)] {
                let e = (u - w).to_f64().abs();
                assert!(e < tol, "M·M⁻¹ off by {e} for s=2^{k} {ms:?}");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 3000);
}
