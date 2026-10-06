//! 演算ごとの許容座標範囲 — 境界の入力で実際に何が起きるかの実測
//!
//! `Fix128` (Q64.64) の分解能は座標に依らず一様 (2⁻⁶⁴) なので、大きな座標で
//! 問題になるのは精度でなく **中間積のあふれ** である 本 file は演算ごとに
//! 境界の入力 (2³⁰, 2³¹−ε, 2³¹, 2³¹+ε, 2³², 2⁶²) を与え、出力を独立に計算した
//! 正しい値 (i128 / 256 bit の整数演算 / f64) と比べて、超えた時の振る舞いを
//! **正しい / wrap (一周) / saturate (上限で止まる) / panic / 明示の Err** の
//! どれかに分類する
//!
//! # 読み方
//!
//! - `characterization_*` は **現状の挙動を pin する** test (wrap するなら wrap する
//!   ことを値で assert する) 挙動が変わったら red になり、表を更新する合図になる
//! - `#[ignore = "src gap: WORLD-V1-RANGE ..."]` の test は **物理として期待する挙動**
//!   (範囲外なら fault を立てる、または正しい値を返す) を assert する 今は red
//! - それ以外は **範囲内で正しい** ことの oracle
//!
//! # 要約 (本 file の実測)
//!
//! | 演算 | 正しい範囲 | 超えた時 |
//! |---|---|---|
//! | `Fix128` `*` | `\|a·b\| < 2⁶³` | wrap (mod 2¹²⁸)、`checked_mul` は `None` |
//! | `Fix128` `+` `-` | `\|a ± b\| < 2⁶³` | wrap |
//! | `Fix128` `/` | `\|a/b\| < 2⁶³` | wrap (`checked_div` は 0 除算しか見ない) |
//! | `sqrt` | 非負の全域 | (超えない) |
//! | `sin` `cos` | 全域で wrap しない | 絶対誤差が `\|x\|·2⁻⁷⁰` 程度で増える |
//! | `exp` | `x < 43` | saturate (文書どおり) |
//! | `powf_pos` 整数部 | 結果 `< 2⁶³` | wrap |
//! | `Vec3Fix::length_squared` / `length` / `normalize` | `\|v\| < 2³¹·⁵ ≈ 3.04e9` | wrap、`length` は 0、`normalize` は ZERO (`checked_*` 版は `None`) |
//! | `Vec3Fix::checked_length_scaled` / `try_normalize_scaled` | 長さ `< 2⁶³` / 全域 | `None` (範囲内は `length` / `try_normalize` と bit 一致) |
//! | `Vec3Fix::cross` | 各積 `< 2⁶²` | wrap |
//! | `QuatFix::rotate_vec` (単位 q) | `\|v\| ≤ 2⁶²` で wrap しない | 絶対誤差が `\|v\|` に比例 |
//! | `Mat3Fix::inverse` | `2⁻⁶³ < \|det\| < 2⁶³` (余因子・各成分も範囲内) | `None` |
//! | `Shape::mass_and_inertia` | `ρ·8·L⁵ < 2⁶¹` | 明示の `Err` |
//! | 剛体の位置積分 | `\|x\| < 2⁶³` | 位置を据え置いて overflow flag (XPBD / TGS) |
//! | 角速度の積分 | `\|ω\|·h < 2⁶³` (`\|ω\| ≥ 2³¹·⁵` は 2 乗を経ない長さ) | 回さずに overflow flag |
//! | `apply_impulse_at` の torque | `\|r\|·\|J\| < 2⁶²` | wrap |
//! | 球同士の接触 | 半径和 `< 2⁶³` (2 乗が範囲外の対は 2 乗を経ない長さで比べる) | 接触を作らずに overflow flag |
//! | `ForceField::Point` | 全域 (距離 `≥ 2³¹·⁵` は 2 乗を経ない式、範囲内は従来の式と bit 一致) | — |
//! | `LinearBvh` の節点 AABB | 座標 `< 2³¹` (i32) | 候補が全対になる (結果は不変) |
//! | `SpatialGrid::hash` | `\|x / cell\| < 2⁶³` かつ `hi + half` が i64 に収まる | wrap / debug で panic |
//! | SDF の問い合わせ | field 原点からの距離で f32 の精度 | 2²⁴ 付近で 0.5 単位に丸まる |
//! | world 全体の step | 2 体・格子の scene は 2⁶¹ まで平行移動で bit 一致 | — |

// 参照値と許容幅の計算にだけ f64 の libm を使う (判定は緩い許容幅で、bit 一致は求めない)
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::collide_sphere_sdf;
use alice_physics::solver::Broadphase;
use alice_physics::{
    ClosureSdf, Fix128, ForceField, Mat3Fix, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody,
    SdfCollider, Shape, ShapeError, SolverBackend, SpatialGrid, Vec3Fix,
};

// ---------------------------------------------------------------------------
// 独立な参照計算
// ---------------------------------------------------------------------------

fn raw(f: Fix128) -> i128 {
    (i128::from(f.hi) << 64) | i128::from(f.lo)
}

fn from_raw128(r: i128) -> Fix128 {
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

/// `2ⁿ` (n < 63)
fn pow2(n: u32) -> Fix128 {
    Fix128::from_raw(1i64 << n, 0)
}

/// 最小の正の値 `2⁻⁶⁴`
fn eps() -> Fix128 {
    Fix128::from_raw(0, 1)
}

const M64: u128 = u64::MAX as u128;

/// u128 × u128 の 256 bit 積 (上位, 下位)
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    let (a1, a0) = (a >> 64, a & M64);
    let (b1, b0) = (b >> 64, b & M64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & M64) + (p10 & M64);
    let lo = (p00 & M64) | ((mid & M64) << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// Q64.64 積の参照値: 厳密な `floor(a·b / 2⁶⁴)` を絶対値の 256 bit 積から作る
/// (`Fix128::mul` の符号付き 4 分割とは別の算法)
///
/// 戻り値: (`2¹²⁸` を法とした raw, 真の値が i128 に収まるか)
fn mul_ref(a: Fix128, b: Fix128) -> (i128, bool) {
    let (a, b) = (raw(a), raw(b));
    let neg = (a < 0) != (b < 0);
    let (hi, lo) = mul_wide(a.unsigned_abs(), b.unsigned_abs());
    // |a·b| >> 64 = hi·2⁶⁴ + (lo >> 64)
    let q_fits_u128 = hi >> 64 == 0;
    let q = (hi << 64) | (lo >> 64);
    let rem = lo & M64 != 0;
    let fits = q_fits_u128
        && if neg {
            q + u128::from(rem) <= 1u128 << 127
        } else {
            q < 1u128 << 127
        };
    let r = if neg {
        (q as i128).wrapping_neg().wrapping_sub(i128::from(rem))
    } else {
        q as i128
    };
    (r, fits)
}

fn zero_gravity_world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    })
}

fn frame() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

// ---------------------------------------------------------------------------
// Fix128 の四則
// ---------------------------------------------------------------------------

/// 積が `2⁶³` 未満なら `*` は厳密な `floor(a·b)` と bit 一致する
#[test]
fn mul_matches_the_exact_product_while_it_fits() {
    let xs = [
        pow2(30),
        pow2(31) - eps(),
        pow2(31),
        pow2(31) + eps(),
        Fix128::from_int(3_037_000_499), // floor(√2⁶³)
        Fix128::from_ratio(-7, 3),
    ];
    for x in xs {
        let (want, fits) = mul_ref(x, x);
        assert!(fits, "前提: {x:?}² は範囲内");
        assert_eq!(raw(x * x), want, "{x:?}²");
        assert_eq!(x.checked_mul(x), Some(x * x));
    }
    // 異符号 (floor が −∞ 側) も参照と一致する
    let (a, b) = (-(pow2(31) + eps()), pow2(31) - eps());
    assert_eq!(raw(a * b), mul_ref(a, b).0);
}

/// `checked_mul` は積が範囲内なら必ず `Some(a * b)` を返す (偽陽性が無い)
///
/// 負の値は整数部が floor なので、`|x| ∈ [3037000499, √2⁶³)` の負の x では
/// 整数部どうしの積だけが i64 を越え、中央の項がそれを打ち消す 判定は和で
/// 行う必要がある (整数部の積だけで判定すると範囲内の積を None にしていた)
#[test]
fn checked_mul_has_no_false_overflow_in_range() {
    let window = [
        Fix128::from_raw(-3_037_000_500, 1 << 63),  // −3037000499.5
        Fix128::from_raw(-3_037_000_500, 1),        // −3037000499.99..
        Fix128::from_raw(-3_037_000_500, u64::MAX), // −3037000499 − 2⁻⁶⁴
    ];
    let mut compared = 0;
    for x in window {
        for y in [x, Fix128::from_raw(-3_037_000_500, 1 << 62), -x] {
            let (want, fits) = mul_ref(x, y);
            if fits {
                assert_eq!(x.checked_mul(y), Some(x * y), "{x:?} * {y:?} は範囲内");
                assert_eq!(raw(x * y), want);
                compared += 1;
            } else {
                assert_eq!(x.checked_mul(y), None, "{x:?} * {y:?} は範囲外");
            }
        }
    }
    assert!(compared >= 6, "範囲内の組を比べていない ({compared})");
    // 境界の両側を細かく走査し、参照で fits の時は必ず Some が返ること
    let mut fits_seen = 0;
    let mut out_seen = 0;
    for k in 0..4096u64 {
        let x = Fix128::from_raw(-3_037_000_500, k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (_, fits) = mul_ref(x, x);
        if fits {
            assert_eq!(x.checked_mul(x), Some(x * x), "{x:?}²");
            fits_seen += 1;
        } else {
            assert_eq!(x.checked_mul(x), None, "{x:?}²");
            out_seen += 1;
        }
    }
    assert!(
        fits_seen > 0 && out_seen > 0,
        "境界の両側を走査していない ({fits_seen}/{out_seen})"
    );
}

/// `checked_mul` は中央の和が i128 を越える組でも、真の積が範囲内なら `Some(a * b)`
///
/// 例: (−2⁶³ + 1 − 2⁻⁶⁴)·(−2⁻⁶⁴) の真値は ≈ 0.5 だが、中央の 2 項がどちらも
/// −2¹²⁶ 級の負で、和が −2¹²⁷ を下回る 端の値と乱数を 256 bit 参照と突き合わせる
#[test]
fn checked_mul_is_exact_when_the_middle_sum_leaves_i128() {
    let a = Fix128::from_raw(i64::MIN, u64::MAX);
    let b = Fix128::from_raw(-1, u64::MAX);
    let (want, fits) = mul_ref(a, b);
    assert!(fits, "前提: 真の積 ≈ 0.5 は範囲内");
    assert_eq!(a.checked_mul(b), Some(a * b));
    assert_eq!(raw(a * b), want);

    let edge_hi = [i64::MIN, i64::MIN + 1, -2, -1, 0, 1, i64::MAX - 1, i64::MAX];
    let edge_lo = [0, 1, 1 << 63, u64::MAX - 1, u64::MAX];
    let mut vals = Vec::new();
    for &h in &edge_hi {
        for &l in &edge_lo {
            vals.push(Fix128::from_raw(h, l));
        }
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..2000 {
        vals.push(Fix128::from_raw(next() as i64, next()));
    }
    let (mut inside, mut outside) = (0u32, 0u32);
    for &x in &vals {
        for &y in vals.iter().take(60) {
            let (want, fits) = mul_ref(x, y);
            if fits {
                assert_eq!(x.checked_mul(y), Some(x * y), "{x:?} * {y:?}");
                assert_eq!(raw(x * y), want);
                inside += 1;
            } else {
                assert_eq!(x.checked_mul(y), None, "{x:?} * {y:?} は範囲外");
                outside += 1;
            }
        }
    }
    assert!(
        inside > 1000 && outside > 1000,
        "両側を比べていない ({inside}/{outside})"
    );
}

/// characterization: 積が `2⁶³` 以上になると `*` は `2¹²⁸` を法に wrap する
/// (panic しない、debug でも同じ) `checked_mul` だけが `None` で知らせる
///
/// 2 乗の境界は `|x| < 2³¹·⁵` (`3_037_000_500² > 2⁶³`)
#[test]
fn characterization_mul_wraps_from_2_pow_31_5() {
    for x in [Fix128::from_int(3_037_000_500), pow2(32), pow2(62)] {
        let (want, fits) = mul_ref(x, x);
        assert!(!fits, "前提: {x:?}² は範囲外");
        assert_eq!(raw(x * x), want, "wrap した値が 2¹²⁸ を法とした値と違う");
        assert_eq!(x.checked_mul(x), None);
    }
    // 符号が反転する例と、厳密に 0 に落ちる例
    assert!((Fix128::from_int(3_037_000_500) * Fix128::from_int(3_037_000_500)).is_negative());
    assert_eq!(pow2(32) * pow2(32), Fix128::ZERO);
    assert_eq!(pow2(62) * pow2(62), Fix128::ZERO);
}

/// characterization: 加算は `±2⁶³` で wrap する
#[test]
fn characterization_add_wraps_at_2_pow_63() {
    assert_eq!(pow2(61) + pow2(61), pow2(62));
    assert_eq!(pow2(62) + pow2(62), Fix128::from_raw(i64::MIN, 0));
    let max = Fix128::from_raw(i64::MAX, u64::MAX);
    assert_eq!(max + eps(), Fix128::from_raw(i64::MIN, 0));
}

/// characterization: 商が `2⁶³` 以上になると `/` は wrap する
/// (`u128 → i64` の切り捨て) `checked_div` は 0 除算しか見ないので `Some` を返す
#[test]
fn characterization_div_quotient_wraps_from_2_pow_63() {
    let half = Fix128::from_ratio(1, 2);
    let quarter = Fix128::from_ratio(1, 4);
    // 範囲内: 正しい
    assert_eq!(pow2(61) / half, pow2(62));
    assert_eq!(pow2(31) / quarter, pow2(33));
    // 2⁶² / 0.5 = 2⁶³ は表せず i64::MIN (負) になる
    assert_eq!(pow2(62) / half, Fix128::from_raw(i64::MIN, 0));
    // 2⁶² / 0.25 = 2⁶⁴ は厳密に 0
    assert_eq!(pow2(62) / quarter, Fix128::ZERO);
    // 1 / 2⁻⁶⁴ = 2⁶⁴ も 0
    assert_eq!(Fix128::ONE / eps(), Fix128::ZERO);
    assert_eq!(Fix128::ONE.checked_div(eps()), Some(Fix128::ZERO));
}

/// `sqrt` は非負の全域で厳密な floor (`r² ≤ N < (r+1)²`, `N = raw·2⁶⁴`)
#[test]
fn sqrt_is_the_exact_floor_over_the_whole_positive_range() {
    assert_eq!(pow2(62).sqrt(), pow2(31));
    for x in [
        pow2(31) - eps(),
        pow2(31) + eps(),
        pow2(62),
        Fix128::from_raw(i64::MAX, u64::MAX),
    ] {
        let r = raw(x.sqrt()) as u128;
        let n_hi = (raw(x) as u128) >> 64;
        let n_lo = (raw(x) as u128) << 64;
        let le = |(h, l): (u128, u128)| h < n_hi || (h == n_hi && l <= n_lo);
        assert!(le(mul_wide(r, r)), "r² > N for {x:?}");
        assert!(!le(mul_wide(r + 1, r + 1)), "(r+1)² ≤ N for {x:?}");
    }
}

/// characterization: `sin` / `cos` は wrap しないが、引数の還元で絶対誤差が
/// `|x|` に比例して増える (2π の表現誤差 × 周回数、実測 ≈ `|x|·2⁻⁷⁰`)
#[test]
fn characterization_sin_absolute_error_grows_with_the_argument() {
    let err = |n: u32| (pow2(n).sin().to_f64() - ((1u64 << n) as f64).sin()).abs();
    assert!(err(20) < 1e-12, "2²⁰: {}", err(20));
    assert!(err(40) < 1e-7, "2⁴⁰: {}", err(40));
    assert!(err(60) > 1e-5, "2⁶⁰: {} (誤差が増えていない)", err(60));
    // 増えても単位円からは外れない
    let (s, c) = pow2(60).sin_cos();
    assert!(((s * s + c * c).to_f64() - 1.0).abs() < 1e-9);
}

/// `exp` は 43 以上で上限に saturate (文書どおり)、`ln` / `atan` は全域で範囲内
#[test]
fn exp_saturates_and_ln_atan_stay_in_range() {
    let max = Fix128::from_raw(i64::MAX, u64::MAX);
    assert_eq!(Fix128::from_int(43).exp(), max);
    assert_eq!(Fix128::from_int(50).exp(), max);
    assert!((max.ln().to_f64() - 63.0 * core::f64::consts::LN_2).abs() < 1e-9);
    assert!((pow2(62).atan().to_f64() - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

/// characterization: `powf_pos` の整数部は `*` の繰り返しなので同じく wrap する
#[test]
fn characterization_powf_pos_integer_power_wraps() {
    let two = Fix128::from_int(2);
    assert_eq!(pow2(31).powf_pos(two), pow2(62));
    assert_eq!(pow2(32).powf_pos(two), Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// Vec3Fix / QuatFix / Mat3Fix
// ---------------------------------------------------------------------------

/// characterization: `length_squared` は `|v|² ≥ 2⁶³` (`|v| ≥ 2³¹·⁵`) で wrap し、
/// `length` は 0、`normalize` は ZERO、`try_normalize` は `None` になる
///
/// 3 軸とも 2³¹ の点 (`|v| ≈ 3.7e9`) は位置としてはまだ範囲内だが、その点の
/// 長さ・向きは取れない
#[test]
fn characterization_length_squared_wraps_from_norm_2_pow_31_5() {
    let z = Fix128::ZERO;
    // 範囲内
    let v = Vec3Fix::new(pow2(30), pow2(30), pow2(30));
    assert_eq!(v.length_squared(), Fix128::from_int(3 << 60));
    assert!((v.length().to_f64() - 3f64.sqrt() * 2f64.powi(30)).abs() < 1.0);
    assert_eq!(Vec3Fix::new(pow2(31), z, z).length(), pow2(31));
    // 範囲外
    let v = Vec3Fix::new(pow2(31), pow2(31), pow2(31));
    assert_eq!(v.length_squared(), Fix128::from_int(-(1 << 62)));
    assert_eq!(v.length(), Fix128::ZERO);
    assert_eq!(v.normalize(), Vec3Fix::ZERO);
    assert_eq!(v.try_normalize(), None);
    assert_eq!(v.dot(v), v.length_squared());
    assert_eq!(Vec3Fix::new(pow2(32), z, z).length(), Fix128::ZERO);
}

/// characterization: `cross` は成分の積と同じ境界で wrap する
#[test]
fn characterization_cross_wraps_with_the_product() {
    let z = Fix128::ZERO;
    let x = Fix128::from_int(3_037_000_499);
    let c = Vec3Fix::new(x, z, z).cross(Vec3Fix::new(z, x, z));
    assert_eq!(c.z, Fix128::from_int(3_037_000_499 * 3_037_000_499));
    let x = Fix128::from_int(3_037_000_500);
    let c = Vec3Fix::new(x, z, z).cross(Vec3Fix::new(z, x, z));
    assert!(c.z.is_negative(), "2⁶³ を超える積が wrap していない");
}

/// 単位四元数の `rotate_vec` は `|v| ≤ 2⁶²` で wrap しない (往復が相対 2⁻⁴⁰ 以内)
#[test]
fn rotate_vec_does_not_wrap_up_to_2_pow_62() {
    let q = QuatFix::from_axis_angle(Vec3Fix::from_int(1, 1, 1), Fix128::from_ratio(7, 10));
    let v = Vec3Fix::new(pow2(62), pow2(61), Fix128::ZERO);
    let back = q.conjugate().rotate_vec(q.rotate_vec(v));
    let tol = 2f64.powi(62 - 40);
    for (a, b) in [(back.x, v.x), (back.y, v.y), (back.z, v.z)] {
        assert!((a - b).abs().to_f64() < tol, "{a} vs {b}");
    }
    let r = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), Fix128::HALF_PI)
        .rotate_vec(Vec3Fix::new(pow2(62), Fix128::ZERO, Fix128::ZERO));
    assert!((r.y.to_f64() / 2f64.powi(62) - 1.0).abs() < 1e-12);
}

/// characterization: 回転の絶対誤差は `|v|` に比例する (四元数の成分が厳密でない)
/// 2³¹ の点を 90° 回すと 1e-5 m 級ずれる ⇒ 絶対座標を回さず相対ベクトルを回す
#[test]
fn characterization_rotate_vec_absolute_error_scales_with_magnitude() {
    let q = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), Fix128::HALF_PI);
    let small = q.rotate_vec(Vec3Fix::from_int(1, 0, 0)).x.abs().to_f64();
    let big = q
        .rotate_vec(Vec3Fix::new(pow2(31), Fix128::ZERO, Fix128::ZERO))
        .x
        .abs()
        .to_f64();
    assert!(small < 1e-12, "|v|=1: {small}");
    assert!(big > 1e-7 && big < 1e-3, "|v|=2³¹: {big}");
}

/// `Mat3Fix::inverse` は `|det|` が `2⁶³` 以上でも `2⁻⁶³` 以下でも `None`
/// (以前は境界の直後で符号の反転した逆行列を `Some` で返していた)
///
/// 対角 `s` の 3×3 なら `det = s³` ⇒ 正しい範囲は `2⁻²¹ < s < 2²¹`
#[test]
fn mat3_inverse_is_none_outside_the_det_range() {
    let d = |f: Fix128| Mat3Fix::diagonal(f, f, f);
    let small = |n: u32| Fix128::from_raw(0, 1u64 << (64 - n));
    // 範囲内
    assert_eq!(d(pow2(20)).inverse().unwrap().col0.x, small(20));
    assert_eq!(d(small(20)).inverse().unwrap().col0.x, pow2(20));
    // det = 2⁶³ は収まらない
    assert_eq!(d(pow2(21)).inverse(), None);
    // 1/det = 2⁶³ は収まらない
    assert_eq!(d(small(21)).inverse(), None);
    // det が 0 に落ちる
    assert_eq!(d(pow2(22)).inverse(), None);
    assert_eq!(d(small(22)).inverse(), None);
}

// ---------------------------------------------------------------------------
// 剛体 (質量・積分・衝撃)
// ---------------------------------------------------------------------------

/// 形状の質量は `ρ·8·L⁵ < 2⁶¹` を越えると明示の `Err` (wrap しない)
#[test]
fn shape_mass_beyond_the_limit_is_an_explicit_err() {
    let cube = |n: u32| Shape::Box {
        half_extents: Vec3Fix::new(pow2(n), pow2(n), pow2(n)),
    };
    let (mass, _) = cube(11).mass_and_inertia(Fix128::ONE).unwrap();
    assert_eq!(mass, Fix128::from_int(1 << 36));
    assert_eq!(
        cube(12).mass_and_inertia(Fix128::ONE),
        Err(ShapeError::MassNotRepresentable)
    );
}

fn edge_world(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    let mut b = RigidBody::new(
        Vec3Fix::new(Fix128::from_raw(i64::MAX, 0), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    );
    b.velocity = Vec3Fix::from_int(120, 0, 0);
    w.add_body(b);
    w
}

/// `+x` の端 (`2⁶³ − 1`) を越えて進む物体は、位置の加算が範囲外になった
/// substep で **位置を据え置き** overflow flag を立てる (XPBD / TGS とも)
/// 端に置いた物体はまず 1 未満の余りを進み、越える substep で止まる
///
/// 修正前は `−2⁶³` 側へ wrap して反対側の端から動き続け、flag は立たなかった
/// (速度導出 `(x − x_prev)/h` も wrap した差で正しい速度を出すので、どの
/// 不変条件でも見えなかった) 据え置きは積 `v·dt` が範囲外の時の扱いと同じ
#[test]
fn position_wrap_at_2_pow_63_raises_the_overflow_flag() {
    let edge = Fix128::from_raw(i64::MAX, 0);
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        let mut w = edge_world(backend);
        w.step(frame());
        assert!(w.overflow_detected(), "{backend:?}");
        let b = w.get_body(0).unwrap();
        // 加算が範囲内の substep だけ進み (端の 1 未満の余り)、越える substep で止まる
        assert!(
            b.position.x >= edge && !b.position.x.is_negative(),
            "{backend:?}: 端で wrap した ({:?})",
            b.position.x
        );
        // 対照: 端から 1 離れて範囲内で動く物体は flag を立てずに進む
        let mut near = edge_world(backend);
        near.bodies[0].position.x = edge - Fix128::from_int(1000);
        near.step(frame());
        assert!(!near.overflow_detected(), "{backend:?}: 範囲内で flag");
        assert!(
            near.get_body(0).unwrap().position.x > edge - Fix128::from_int(1000),
            "{backend:?}: 範囲内で進まない"
        );
    }
}

fn washing_world(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, -Fix128::from_int(1 << 30), Fix128::ZERO),
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w
}

/// 積 `v·dt` が範囲外になる scene (重力 `−2³⁰`, `dt = 2²⁰`) で、XPBD も TGS も
/// flag を立て、落下中の物体が正の側へ飛ばない
///
/// 修正前の TGS は flag を立てず、位置が wrap して `y ≈ +6.8e18` に飛んでいた
/// (TGS は world の body の複製の上で積分するので、複製に溜めた印を戻す時に
/// world の flag へ畳み込む)
#[test]
fn tgs_backend_raises_the_overflow_flag() {
    let dt = Fix128::from_int(1 << 20);
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        let mut w = washing_world(backend);
        let mut flagged_at = None;
        for k in 0..4 {
            w.step(dt);
            if flagged_at.is_none() && w.overflow_detected() {
                flagged_at = Some(k);
            }
            // 落下だけの scene なので y は増えない (wrap すると正に飛ぶ)
            assert!(
                w.get_body(0).unwrap().position.y <= Fix128::ZERO,
                "{backend:?} step {k}: y が正"
            );
        }
        assert!(flagged_at.is_some(), "{backend:?}: flag が立たない");
        // sticky: 立った後の step でも落ちない
        w.step(Fix128::from_ratio(1, 60));
        assert!(w.overflow_detected(), "{backend:?}: flag が落ちた");
    }
    // 対照: 同じ重力でも dt が小さければ範囲内で、どちらも flag は立たない
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        let mut w = washing_world(backend);
        w.step(frame());
        assert!(!w.overflow_detected(), "{backend:?}");
        assert!(
            w.get_body(0).unwrap().position.y < Fix128::ZERO,
            "{backend:?}"
        );
    }
}

fn spinning_world(w: Fix128) -> PhysicsWorld {
    let mut world = zero_gravity_world();
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.angular_velocity = Vec3Fix::new(w, w, w);
    world.add_body(b);
    world
}

/// 参照: 軸 `(1,1,1)/√3` 回りに `ω_axis·√3·h` を substep の数だけ回した
/// 単位 quaternion (f64、角は substep ごとに Fix128 の `h` から作る)
fn spin_reference(w_axis: f64, substeps: usize, h: f64) -> [f64; 4] {
    let angle = w_axis * 3f64.sqrt() * h;
    // 同じ軸の回転は角の和 (順序に依らない)
    let total = angle * substeps as f64;
    let half = total / 2.0;
    let s = half.sin() / 3f64.sqrt();
    [s, s, s, half.cos()]
}

/// `|ω| ≥ 2³¹·⁵` rad/s でも、軸 `(1,1,1)` 回りに `|ω|·h` ずつ正しく回り、
/// ω は 0 に消えない
///
/// 修正前は `normalize_with_length` の長さが wrap して 0 になり、回転が止まり
/// 速度導出で ω 自体も 0 に消えていた (flag なし) 長さと向きは 2 乗を経ない
/// 版で求める (`|ω|·h` が表せる限り flag は立てない)
#[test]
fn angular_speed_beyond_range_is_kept_or_flagged() {
    let substeps = PhysicsConfig::default().substeps;
    let h = frame() / Fix128::from_int(substeps as i64);
    for (k, w_axis) in [(30u32, pow2(30)), (31, pow2(31)), (40, pow2(40))] {
        let mut w = spinning_world(w_axis);
        w.step(frame());
        let b = w.get_body(0).unwrap();
        assert!(!w.overflow_detected(), "2^{k}: flag");
        let r = spin_reference(w_axis.to_f64(), substeps, h.to_f64());
        let q = b.rotation;
        let got = [q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64()];
        // q と −q は同じ回転
        let sign = if got[3] * r[3] + got[0] * r[0] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for c in 0..4 {
            assert!(
                (sign * got[c] - r[c]).abs() < 1e-6,
                "2^{k}: q[{c}] = {} 参照 {}",
                got[c],
                r[c]
            );
        }
        // ω は軸 (1,1,1) の向きのまま残る (大きさは h ごとの回転から導くので
        // |ω|·h > π では別の値に折り返す、範囲内の 2^30 と同じ扱い)
        let om = b.angular_velocity;
        assert_ne!(om, Vec3Fix::ZERO, "2^{k}: ω が消えた");
        assert!(om.x == om.y && om.y == om.z, "2^{k}: ω の向き {om:?}");
    }
}

/// `|ω|` 自体か `|ω|·h` が `≥ 2⁶³` で表せない時は回さずに flag を立てる
#[test]
fn angular_step_beyond_representable_raises_the_overflow_flag() {
    // |ω| = 3·2⁶¹·√3 ≈ 1.2e19 ≥ 2⁶³: 長さが表せない
    let mut w = spinning_world(Fix128::from_raw(3 << 61, 0));
    w.step(frame());
    assert!(w.overflow_detected());
    assert_eq!(w.get_body(0).unwrap().rotation, QuatFix::IDENTITY);

    // |ω| = 2⁶² (1 軸) は表せるが、h = 2 で |ω|·h = 2⁶³ が表せない
    let mut w = zero_gravity_world();
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.angular_velocity = Vec3Fix::new(pow2(62), Fix128::ZERO, Fix128::ZERO);
    w.add_body(b);
    let substeps = PhysicsConfig::default().substeps as i64;
    w.step(Fix128::from_int(2 * substeps));
    assert!(w.overflow_detected());

    // 対照: 同じ ω でも h が小さく |ω|·h が表せれば flag は立たない
    let mut w = zero_gravity_world();
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.angular_velocity = Vec3Fix::new(pow2(62), Fix128::ZERO, Fix128::ZERO);
    w.add_body(b);
    w.step(frame());
    assert!(!w.overflow_detected());
    assert_ne!(w.get_body(0).unwrap().rotation, QuatFix::IDENTITY);
}

/// characterization: `apply_impulse_at` の torque `r × J` は `|r|·|J| ≥ 2⁶³` で wrap
#[test]
fn characterization_apply_impulse_at_torque_wraps() {
    let z = Fix128::ZERO;
    let hit = |n: u32| {
        let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        b.apply_impulse_at(Vec3Fix::new(z, pow2(n), z), Vec3Fix::new(pow2(n), z, z));
        b
    };
    // 範囲内: ω_z = 2⁶⁰ · inv_I (= 2.5)
    let b = hit(30);
    assert_eq!(b.angular_velocity.z, pow2(60) * b.inv_inertia.z);
    // 2³² × 2³² = 2⁶⁴ は 0 に wrap
    assert_eq!(hit(32).angular_velocity.z, Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// 接触 / 力場
// ---------------------------------------------------------------------------

/// 半径 `r` の球 2 個を中心距離 `d` で置いて 1 frame 回し、両者の x を返す
fn sphere_pair(r: i64, d: i64) -> (Fix128, Fix128) {
    let mut w = zero_gravity_world();
    w.add_body_with_radius(
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::from_int(r),
    );
    w.add_body_with_radius(
        RigidBody::new(Vec3Fix::from_int(d, 0, 0), Fix128::ONE),
        Fix128::from_int(r),
    );
    w.step(frame());
    (
        w.get_body(0).unwrap().position.x,
        w.get_body(1).unwrap().position.x,
    )
}

/// 半径和が `2³¹·⁵` を越える球の対も、重なっていれば押し離される
///
/// 修正前は `dist² < (r_a + r_b)²` の両辺が wrap して接触を見逃していた
/// oracle: 重力なしの 2 体の接触は長さについて線形なので、同じ配置を 2⁻¹⁰ に
/// 縮めた scene (2 乗が範囲内、従来の経路) の結果の 2¹⁰ 倍と一致する
/// (相対 1e-9、丸めの差だけ)
#[test]
fn sphere_pair_beyond_range_is_separated_or_flagged() {
    // 中心距離 2_999_999_488 = 2_929_687 · 2¹⁰ (縮めても整数のまま)
    let small_d = 2_929_687;
    let (sa, sb) = sphere_pair(3 << 19, small_d);
    assert!(
        sa < Fix128::ZERO && sb > Fix128::from_int(small_d),
        "縮めた scene が離れていない"
    );
    let (a, b) = sphere_pair(3 << 29, small_d << 10);
    assert!(a < Fix128::ZERO, "押し離されていない");
    let scale = 1024.0;
    for (big, small) in [(a, sa), (b, sb)] {
        let want = small.to_f64() * scale;
        let rel = (big.to_f64() - want).abs() / want.abs();
        assert!(
            rel < 1e-9,
            "{} vs 2^10 × {}: 相対 {rel}",
            big.to_f64(),
            small.to_f64()
        );
    }
}

/// 半径和そのものが `≥ 2⁶³` で表せない対は、接触を作らずに flag を立てる
/// (wrap した負の半径和で比べない)
#[test]
fn sphere_radius_sum_beyond_2_pow_63_raises_the_overflow_flag() {
    let mut w = zero_gravity_world();
    let r = Fix128::from_raw(1i64 << 62, 0);
    w.add_body_with_radius(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE), r);
    w.add_body_with_radius(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE), r);
    w.step(frame());
    assert!(w.overflow_detected());
    // 対照: 半径 2⁴⁰ (和は表せ、2 乗は範囲外) なら flag は立たず押し離される
    let mut w = zero_gravity_world();
    let r = Fix128::from_raw(1i64 << 40, 0);
    w.add_body_with_radius(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE), r);
    w.add_body_with_radius(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE), r);
    w.step(frame());
    assert!(!w.overflow_detected());
    assert!(w.get_body(0).unwrap().position.x < Fix128::ZERO);
}

fn point_force(pos: Vec3Fix, strength: Fix128) -> Vec3Fix {
    let field = ForceField::Point {
        center: Vec3Fix::ZERO,
        strength,
        repulsive: false,
        max_force: Fix128::from_raw(i64::MAX, 0),
    };
    alice_physics::force::compute_force(&field, &RigidBody::new(pos, Fix128::ONE))
}

/// 点力場の参照値 (f64): 中心向き (引力) に `strength / r²`、上限 `max_force`
fn point_force_ref(pos: Vec3Fix, strength: f64) -> [f64; 3] {
    let (x, y, z) = (pos.x.to_f64(), pos.y.to_f64(), pos.z.to_f64());
    let r = (x * x + y * y + z * z).sqrt();
    let m = strength / (r * r);
    [-x / r * m, -y / r * m, -z / r * m]
}

fn assert_close_to_ref(pos: Vec3Fix, strength: Fix128) {
    let f = point_force(pos, strength);
    let e = point_force_ref(pos, strength.to_f64());
    let mag = e.iter().map(|c| c * c).sum::<f64>().sqrt();
    for (got, want) in [f.x, f.y, f.z].iter().zip(e) {
        // 絶対 2⁻⁵⁰ か相対 2⁻⁴⁰ (参照の f64 の丸めが支配的)
        let tol = (mag * 2f64.powi(-40)).max(2f64.powi(-50));
        assert!(
            (got.to_f64() - want).abs() <= tol,
            "pos {pos}: got {f}, want {e:?}"
        );
    }
}

/// 点力場は中心からの距離が `2³¹·⁵` を越えても `strength / r²` を返す
/// (以前は `dist²` が wrap し、3·2³⁰ で上限値 (真値の 2e19 倍)、2³² で 0 だった)
///
/// oracle: 逆 2 乗則の閉形式を f64 で計算した参照値
#[test]
fn point_field_matches_the_inverse_square_across_the_square_range_edge() {
    let z = Fix128::ZERO;
    // 範囲内: 2⁶⁰ / (2³⁰)² = 1、中心向き
    assert_eq!(
        point_force(Vec3Fix::new(pow2(30), z, z), pow2(60)),
        Vec3Fix::new(-Fix128::ONE, z, z)
    );
    // 範囲外: 2⁶² / (2³²)² = 2⁻² ちょうど
    assert_eq!(
        point_force(Vec3Fix::new(pow2(32), z, z), pow2(62)),
        Vec3Fix::new(-Fix128::from_ratio(1, 4), z, z)
    );
    let big = Fix128::from_int(3_037_000_499); // 2³¹·⁵ の直下
    for pos in [
        Vec3Fix::new(pow2(31) + pow2(30), z, z),
        Vec3Fix::new(pow2(31), pow2(31), pow2(31)),
        Vec3Fix::new(big, big, z),
        Vec3Fix::new(big, z, z),
        Vec3Fix::new(-pow2(40), pow2(39), -pow2(20)),
        Vec3Fix::new(pow2(61), -pow2(61), pow2(61)),
        Vec3Fix::new(pow2(62), pow2(62), pow2(62)),
        // 距離そのものが 2⁶³ を越える (半分の長さで計算する経路)
        Vec3Fix::new(Fix128::from_int(3 << 61), Fix128::from_int(3 << 61), z),
    ] {
        assert_close_to_ref(pos, pow2(62));
        assert_close_to_ref(pos, Fix128::from_int(7));
    }
    // 斥力は向きだけ反転する
    let pos = Vec3Fix::new(pow2(33), -pow2(32), z);
    let field = ForceField::Point {
        center: Vec3Fix::ZERO,
        strength: pow2(62),
        repulsive: true,
        max_force: Fix128::from_raw(i64::MAX, 0),
    };
    let f = alice_physics::force::compute_force(&field, &RigidBody::new(pos, Fix128::ONE));
    let g = point_force(pos, pow2(62));
    // `Mul` は −∞ 向きの切り捨てなので、符号の反転は 1 raw 単位までずれる
    for (a, b) in [(f.x, g.x), (f.y, g.y), (f.z, g.z)] {
        assert!((raw(a) + raw(b)).abs() <= 1, "{f} vs {g}");
    }
    assert!(f.x > Fix128::ZERO && f.y < Fix128::ZERO);
    // 上限は範囲外でも効く
    let field = ForceField::Point {
        center: Vec3Fix::ZERO,
        strength: pow2(62),
        repulsive: false,
        max_force: Fix128::from_ratio(1, 8),
    };
    let f = alice_physics::force::compute_force(
        &field,
        &RigidBody::new(Vec3Fix::new(pow2(32), z, z), Fix128::ONE),
    );
    assert_eq!(f, Vec3Fix::new(-Fix128::from_ratio(1, 8), z, z));
}

#[test]
fn point_field_beyond_range_matches_the_inverse_square() {
    let z = Fix128::ZERO;
    let f = point_force(Vec3Fix::new(pow2(32), z, z), pow2(62));
    // 2⁶² / 2⁶⁴ = 0.25、中心向き
    assert!((f.x.to_f64() + 0.25).abs() < 1e-9, "{f}");
}

/// 修正前の `ForceField::Point` の式 (範囲内の bit 不変を確かめる参照)
///
/// `src/force.rs` の修正前の本体をそのまま写したもの
fn point_force_previous(
    center: Vec3Fix,
    strength: Fix128,
    repulsive: bool,
    max_force: Fix128,
    position: Vec3Fix,
) -> Vec3Fix {
    let delta = center - position;
    let dist_sq = delta.length_squared();
    if delta == Vec3Fix::ZERO {
        return Vec3Fix::ZERO;
    }
    let direction = if dist_sq < Fix128::ONE {
        let mut scaled = delta;
        let mut scaled_dist_sq = dist_sq;
        for _ in 0..128 {
            if scaled_dist_sq >= Fix128::ONE {
                break;
            }
            scaled = scaled + scaled;
            scaled_dist_sq = scaled.length_squared();
        }
        scaled / scaled_dist_sq.sqrt()
    } else {
        delta / dist_sq.sqrt()
    };
    let dist_sq_for_force = if max_force.is_zero() {
        dist_sq
    } else {
        let floor = (strength / max_force).abs();
        if dist_sq < floor {
            floor
        } else {
            dist_sq
        }
    };
    let force_mag = strength / dist_sq_for_force;
    let clamped = if force_mag > max_force {
        max_force
    } else {
        force_mag
    };
    if repulsive {
        -direction * clamped
    } else {
        direction * clamped
    }
}

/// 疑似乱数 (LCG) で raw の上位 `bits` bit までの符号付き値
fn lcg_fix(state: &mut u64, bits: u32) -> Fix128 {
    let mut next = || {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    };
    let a = ((u128::from(next()) << 64) | u128::from(next())) >> (128 - bits);
    let v = if next() & 1 == 1 {
        -(a as i128)
    } else {
        a as i128
    };
    from_raw128(v)
}

/// 範囲内 (`|delta|² < 2⁶³`) では修正後の点力場が修正前の式と bit 一致する
///
/// 近距離の 2 倍寄せ (`dist² < 1`)、上限による下限 (`dist² < strength / max_force`)、
/// 上限なし (`max_force = 0`)、斥力、範囲の境界の直下を含む
#[test]
fn point_field_in_range_is_bit_identical_to_the_previous_formula() {
    let mut s = 0x5eed_u64;
    let mut checked = 0;
    for bits in [20u32, 40, 56, 64, 72, 80, 88, 93, 94] {
        for i in 0..3000 {
            let center = Vec3Fix::new(
                lcg_fix(&mut s, 80),
                lcg_fix(&mut s, 80),
                lcg_fix(&mut s, 80),
            );
            let delta = Vec3Fix::new(
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
            );
            if delta.checked_length_squared().is_none() {
                continue;
            }
            let position = center - delta;
            let strength = lcg_fix(&mut s, 100);
            let max_force = match i % 3 {
                0 => Fix128::ZERO,
                1 => lcg_fix(&mut s, 100).abs(),
                _ => Fix128::from_raw(i64::MAX, 0),
            };
            let repulsive = i % 2 == 0;
            let field = ForceField::Point {
                center,
                strength,
                repulsive,
                max_force,
            };
            let got =
                alice_physics::force::compute_force(&field, &RigidBody::new(position, Fix128::ONE));
            let want = point_force_previous(center, strength, repulsive, max_force, position);
            assert_eq!(got, want, "delta {delta} strength {strength}");
            checked += 1;
        }
    }
    // 境界の直下 (3 軸) と 1 軸
    let big = Fix128::from_int(1_753_413_056); // 3·big² < 2⁶³
    for delta in [
        Vec3Fix::new(big, big, big),
        Vec3Fix::new(Fix128::from_int(3_037_000_499), Fix128::ZERO, Fix128::ZERO),
    ] {
        assert!(delta.checked_length_squared().is_some());
        let field = ForceField::Point {
            center: Vec3Fix::ZERO,
            strength: pow2(62),
            repulsive: false,
            max_force: Fix128::from_raw(i64::MAX, 0),
        };
        let got = alice_physics::force::compute_force(&field, &RigidBody::new(-delta, Fix128::ONE));
        let want = point_force_previous(
            Vec3Fix::ZERO,
            pow2(62),
            false,
            Fix128::from_raw(i64::MAX, 0),
            -delta,
        );
        assert_eq!(got, want);
        checked += 1;
    }
    assert!(checked > 20_000, "比較件数が少ない: {checked}");
}

// ---------------------------------------------------------------------------
// Vec3Fix の範囲検査版
// ---------------------------------------------------------------------------

/// `checked_dot` / `checked_length_squared` / `checked_length` / `checked_normalize` は
/// 範囲内で既存の関数と bit 一致し、範囲外 (積・和が `2⁶³` 以上) で `None` を返す
///
/// oracle: 範囲の境界は整数の 2 乗 (`3_037_000_499² < 2⁶³ < 3_037_000_500²`)
#[test]
fn checked_vector_products_match_inside_and_refuse_outside() {
    let z = Fix128::ZERO;
    let lo = Fix128::from_int(3_037_000_499);
    let hi = Fix128::from_int(3_037_000_500);
    // 1 成分: 積の境界
    let v = Vec3Fix::new(lo, z, z);
    assert_eq!(v.checked_length_squared(), Some(v.length_squared()));
    assert_eq!(v.checked_length(), Some(v.length()));
    assert_eq!(v.checked_normalize(), v.try_normalize());
    let v = Vec3Fix::new(hi, z, z);
    assert_eq!(v.checked_length_squared(), None);
    assert_eq!(v.checked_length(), None);
    assert_eq!(v.checked_normalize(), None);
    // 積は収まるが和が 2⁶³ を越える
    let v = Vec3Fix::new(pow2(31), pow2(31), z);
    assert_eq!(v.checked_dot(v), None);
    // 各積は 2⁶³ の直下でも、和が越えれば None
    let v = Vec3Fix::new(lo, lo, z);
    assert_eq!(v.checked_dot(v), None);
    // 符号の違う積の和は範囲内なら Some
    let a = Vec3Fix::new(lo, lo, z);
    let b = Vec3Fix::new(lo, -lo, z);
    assert_eq!(a.checked_dot(b), Some(z));
    assert_eq!(a.checked_dot(b), Some(a.dot(b)));
    // 範囲内の乱数で既存の関数と bit 一致
    let mut s = 0xd07_u64;
    let mut n = 0;
    for bits in [30u32, 60, 80, 90, 93] {
        for _ in 0..2000 {
            let a = Vec3Fix::new(
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
            );
            let b = Vec3Fix::new(
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
            );
            if let Some(d) = a.checked_dot(b) {
                assert_eq!(d, a.dot(b));
                n += 1;
            }
            if let Some(l) = a.checked_length() {
                assert_eq!(l, a.length());
                assert_eq!(a.checked_normalize(), a.try_normalize());
                n += 1;
            }
        }
    }
    assert!(n > 10_000, "比較件数が少ない: {n}");
}

/// `checked_length_scaled` / `try_normalize_scaled` は範囲内で `length` /
/// `try_normalize` と bit 一致し、範囲外でも正しい長さ・向きを返す
///
/// oracle: f64 の `sqrt(x² + y² + z²)` と `v / |v|`
#[test]
fn scaled_length_and_direction_are_correct_beyond_the_square_range() {
    let z = Fix128::ZERO;
    // 厳密に分かる値
    assert_eq!(
        Vec3Fix::new(pow2(32), z, z).checked_length_scaled(),
        Some(pow2(32))
    );
    assert_eq!(
        Vec3Fix::new(z, -pow2(62), z).checked_length_scaled(),
        Some(pow2(62))
    );
    assert_eq!(
        Vec3Fix::new(
            pow2(40) * Fix128::from_int(3),
            z,
            -pow2(40) * Fix128::from_int(4)
        )
        .checked_length_scaled(),
        Some(pow2(40) * Fix128::from_int(5))
    );
    assert_eq!(
        Vec3Fix::new(pow2(32), z, z).try_normalize_scaled(),
        Some(Vec3Fix::UNIT_X)
    );
    // 長さが 2⁶³ 以上 (表せない): 3·2⁶¹·√2 ≈ 9.78e18
    assert_eq!(
        Vec3Fix::new(Fix128::from_int(3 << 61), Fix128::from_int(3 << 61), z)
            .checked_length_scaled(),
        None
    );
    // 2⁶²·√2 ≈ 6.52e18 は表せる
    assert!(
        (Vec3Fix::new(pow2(62), pow2(62), z)
            .checked_length_scaled()
            .expect("2⁶³ 未満")
            .to_f64()
            - 2f64.powf(62.5))
        .abs()
            <= 2f64.powf(62.5) * 1e-15
    );
    // 零ベクトル
    assert_eq!(Vec3Fix::ZERO.checked_length_scaled(), Some(z));
    assert_eq!(Vec3Fix::ZERO.try_normalize_scaled(), None);
    // 長さの 2 乗が下位桁で 0 になる非零ベクトルにも向きを返す
    let tiny = Vec3Fix::new(eps(), z, z);
    assert_eq!(tiny.try_normalize(), None);
    assert_eq!(tiny.try_normalize_scaled(), Some(Vec3Fix::UNIT_X));

    let mut s = 0x5ca1e_u64;
    let (mut inside, mut outside) = (0, 0);
    for bits in [20u32, 50, 70, 90, 93, 94, 96, 100, 110, 120, 126] {
        for _ in 0..2000 {
            let v = Vec3Fix::new(
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
                lcg_fix(&mut s, bits),
            );
            if v.checked_length_squared().is_some() {
                assert_eq!(v.checked_length_scaled(), Some(v.length()));
                if let Some(n) = v.try_normalize() {
                    assert_eq!(v.try_normalize_scaled(), Some(n));
                }
                inside += 1;
                continue;
            }
            outside += 1;
            let (x, y, zz) = (v.x.to_f64(), v.y.to_f64(), v.z.to_f64());
            let t = (x * x + y * y + zz * zz).sqrt();
            match v.checked_length_scaled() {
                Some(l) => assert!((l.to_f64() - t).abs() <= t * 1e-15, "{v}: {l} vs {t}"),
                None => assert!(t >= 2f64.powi(63) * (1.0 - 1e-12), "{v}: None at {t}"),
            }
            let d = v.try_normalize_scaled().expect("非零");
            for (got, want) in [(d.x, x / t), (d.y, y / t), (d.z, zz / t)] {
                assert!((got.to_f64() - want).abs() <= 1e-15, "{v}: {d}");
            }
        }
    }
    assert!(inside > 5_000 && outside > 5_000, "{inside} / {outside}");
}

// ---------------------------------------------------------------------------
// broadphase / SpatialGrid
// ---------------------------------------------------------------------------

/// 4×4×4 の球格子 (間隔 2.5、半径 1、1 体だけ動く) を `offset` に置いて 10 frame
fn grid_scene(offset: Vec3Fix) -> (Vec<(Vec3Fix, Vec3Fix)>, u64) {
    let mut w = zero_gravity_world();
    w.set_broadphase(Broadphase::Bvh);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let p = Vec3Fix::new(
                    Fix128::from_ratio(5 * i, 2),
                    Fix128::from_ratio(5 * j, 2),
                    Fix128::from_ratio(5 * k, 2),
                );
                let mut b = RigidBody::new(offset + p, Fix128::ONE);
                if i + j + k == 0 {
                    b.velocity = Vec3Fix::from_int(3, 0, 0);
                }
                w.add_body_with_radius(b, Fix128::ONE);
            }
        }
    }
    let mut pairs = 0;
    for _ in 0..10 {
        w.step(frame());
        pairs += w.stage_work().broadphase_pairs;
    }
    let state = (0..64)
        .map(|i| {
            let b = w.get_body(i).unwrap();
            (b.position - offset, b.velocity)
        })
        .collect();
    (state, pairs)
}

/// characterization: `LinearBvh` の節点 AABB は i32 に clamp されるので、3 軸とも
/// `2³¹` を越えた場所では全 body の箱が 1 点に潰れ、候補が全対 (64·63/2) になる
/// 結果は原点と bit 一致する (候補が増えるだけで、狭域判定が落とす)
#[test]
fn characterization_bvh_candidates_become_all_pairs_beyond_i32() {
    let substeps = PhysicsConfig::default().substeps as u64;
    let per_substep = |p: u64| p / (10 * substeps);
    let (origin, p0) = grid_scene(Vec3Fix::ZERO);
    assert!(
        per_substep(p0) < 2016,
        "前提: 原点では全対より少ない ({})",
        per_substep(p0)
    );
    let t = pow2(31);
    let (far, p1) = grid_scene(Vec3Fix::new(t, t, t));
    assert_eq!(per_substep(p1), 2016);
    assert!(far == origin, "候補の増加で結果が変わった");
}

#[test]
fn spatial_grid_hash_clamps_to_the_edge_cell() {
    let z = Fix128::ZERO;
    let g = SpatialGrid::new(Fix128::from_ratio(1, 4), 16);
    assert_eq!(g.hash(Vec3Fix::new(pow2(62), z, z)) % 16, 15);
    let g1 = SpatialGrid::new(Fix128::ONE, 16);
    assert_eq!(
        g1.hash(Vec3Fix::new(Fix128::from_raw(i64::MAX, 0), z, z)) % 16,
        15
    );
}

// ---------------------------------------------------------------------------
// SDF
// ---------------------------------------------------------------------------

/// field 原点から `wall` の位置にある平面 (`d = x − wall`) と、その外側 0.25 に
/// 中心を置いた半径 0.5 の球の接触深さ (真値 0.25)
fn sdf_wall_depth(field_origin: Vec3Fix, wall: f32) -> f64 {
    let sdf = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            move |x, _, _| x - wall,
            |_, _, _| (1.0, 0.0, 0.0),
        )),
        field_origin,
        QuatFix::IDENTITY,
    );
    let center = field_origin
        + Vec3Fix::new(
            Fix128::from_f64(f64::from(wall)) + Fix128::from_ratio(1, 4),
            Fix128::ZERO,
            Fix128::ZERO,
        );
    collide_sphere_sdf(center, Fix128::from_ratio(1, 2), &sdf)
        .expect("接触がない")
        .depth
        .to_f64()
}

/// characterization: SDF は field 原点からの **局所座標を f32 で** 評価するので、
/// 精度は world 座標でなく field 原点からの距離で決まる 原点を一緒に動かせば
/// 2⁴⁰ でも正確、局所 2²⁴ では 0.25 が丸められて深さが 0.5 になる
#[test]
fn characterization_sdf_query_is_f32_relative_to_the_field_origin() {
    let t = pow2(40);
    assert!((sdf_wall_depth(Vec3Fix::ZERO, 0.0) - 0.25).abs() < 1e-6);
    assert!((sdf_wall_depth(Vec3Fix::new(t, t, t), 0.0) - 0.25).abs() < 1e-6);
    let d = sdf_wall_depth(Vec3Fix::ZERO, 16_777_216.0);
    assert!((d - 0.5).abs() < 1e-6, "局所 2²⁴: {d}");
}

// ---------------------------------------------------------------------------
// world の step 全体: 平行移動の不変性
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Scene {
    /// 球 2 個が斜めに衝突
    Collide,
    /// 箱 2 個 (回転あり) が衝突、GJK / EPA を通る
    Boxes,
    /// 自由落下
    Fall,
    /// 回転だけ
    Spin,
}

type BodyState = (Vec3Fix, Vec3Fix, QuatFix, Vec3Fix);

fn run_scene(scene: Scene, offset: Vec3Fix, backend: SolverBackend) -> (Vec<BodyState>, bool) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: if matches!(scene, Scene::Fall) {
            PhysicsConfig::default().gravity
        } else {
            Vec3Fix::ZERO
        },
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    let one = Fix128::ONE;
    let n = match scene {
        Scene::Collide => {
            let mut a = RigidBody::new(offset + Vec3Fix::from_int(-3, 0, 0), one);
            a.velocity = Vec3Fix::from_int(4, 0, 0);
            let mut b = RigidBody::new(
                offset + Vec3Fix::new(Fix128::from_int(3), Fix128::from_ratio(1, 2), Fix128::ZERO),
                one,
            );
            b.velocity = Vec3Fix::from_int(-4, 0, 0);
            w.add_body_with_radius(a, one);
            w.add_body_with_radius(b, one);
            2
        }
        Scene::Boxes => {
            let s = Shape::Box {
                half_extents: Vec3Fix::from_int(1, 1, 1),
            };
            let i = w
                .add_shaped_body(&s, one, offset + Vec3Fix::from_int(-3, 0, 0))
                .unwrap();
            let j = w
                .add_shaped_body(
                    &s,
                    one,
                    offset
                        + Vec3Fix::new(
                            Fix128::from_int(3),
                            Fix128::from_ratio(1, 3),
                            Fix128::from_ratio(1, 5),
                        ),
                )
                .unwrap();
            let a = w.get_body_mut(i).unwrap();
            a.velocity = Vec3Fix::from_int(4, 0, 0);
            a.angular_velocity = Vec3Fix::from_int(0, 1, 2);
            let b = w.get_body_mut(j).unwrap();
            b.velocity = Vec3Fix::from_int(-4, 0, 0);
            b.rotation =
                QuatFix::from_axis_angle(Vec3Fix::from_int(1, 1, 0), Fix128::from_ratio(1, 3));
            2
        }
        Scene::Fall => {
            w.add_body_with_radius(RigidBody::new(offset, one), one);
            1
        }
        Scene::Spin => {
            let mut a = RigidBody::new(offset, one);
            a.angular_velocity = Vec3Fix::from_int(1, 2, 3);
            w.add_body(a);
            1
        }
    };
    for _ in 0..60 {
        w.step(frame());
    }
    let state = (0..n)
        .map(|i| {
            let b = w.get_body(i).unwrap();
            (
                b.position - offset,
                b.velocity,
                b.rotation,
                b.angular_velocity,
            )
        })
        .collect();
    (state, w.overflow_detected())
}

/// 衝突・自由落下・回転の scene を 2²⁰〜2⁶¹ (3 軸とも) に平行移動しても、
/// 原点の scene と **bit 一致** する (XPBD / TGS)
///
/// world の剛体経路は位置を差 (`x_a − x_b`, `p − x`) でしか掛け算に使わないので、
/// 分解能が一様な固定小数点では平行移動で何も変わらない 中間積のあふれは
/// 位置の絶対値でなく **差・半径・速度・力の大きさ** で決まる
#[test]
fn world_step_is_translation_invariant_up_to_2_pow_61() {
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        for scene in [Scene::Collide, Scene::Boxes, Scene::Fall, Scene::Spin] {
            let (base, flag0) = run_scene(scene, Vec3Fix::ZERO, backend);
            assert!(!flag0);
            for k in [20u32, 25, 30, 31, 40, 61] {
                let t = pow2(k);
                let (got, flag) = run_scene(scene, Vec3Fix::new(t, t, t), backend);
                assert!(got == base, "{backend:?} {scene:?} 2^{k}: 原点と一致しない");
                assert!(!flag, "{backend:?} {scene:?} 2^{k}: flag");
            }
        }
    }
}

/// 参照計算自体の検算: 既知の積で `mul_ref` が正しい
#[test]
fn reference_product_is_correct_on_known_values() {
    assert_eq!(mul_ref(pow2(31), pow2(31)), (raw(pow2(62)), true));
    assert_eq!(
        mul_ref(Fix128::from_ratio(-1, 2), eps()),
        (-1, true),
        "floor は −∞ 側"
    );
    assert_eq!(mul_ref(pow2(32), pow2(32)), (0, false));
    assert_eq!(from_raw128(raw(Fix128::from_int(-5))), Fix128::from_int(-5));
}

// ---------------------------------------------------------------------------
// 範囲内は修正前と bit 一致
// ---------------------------------------------------------------------------

/// FNV-1a 64 (blob の指紋、比較のためだけ)
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// 範囲の検査を足した経路 (積分の位置の加算、角速度の長さ、球の接触の 2 乗と
/// 半径和、TGS の積分と関節投影) をすべて通る、範囲内の代表 scene
///
/// 境界の直下 (|ω| = 3_037_000_000 < 2³¹·⁵、半径和 2³¹、位置 2⁶²) を含む
fn in_range_scene(id: u32, backend: SolverBackend) -> PhysicsWorld {
    let one = Fix128::ONE;
    let mut w = PhysicsWorld::new(PhysicsConfig {
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    match id {
        // 床の上の 3×3×3 の球の山 (接触・摩擦・反発・sleep)
        0 => {
            w.add_body_with_radius(
                RigidBody::new_static(Vec3Fix::from_int(0, -100, 0)),
                Fix128::from_int(100),
            );
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let p = Vec3Fix::new(
                            Fix128::from_ratio(2 * i64::from(i) * 101, 100),
                            Fix128::from_ratio(1 + 2 * i64::from(j) * 101, 100) + one,
                            Fix128::from_ratio(2 * i64::from(k) * 103, 100),
                        );
                        w.add_body_with_radius(RigidBody::new_dynamic(p, one), one);
                    }
                }
            }
        }
        // 回る箱 2 個の衝突 (collider の経路と gyroscopic の分割)
        1 => {
            w.config.gravity = Vec3Fix::ZERO;
            let s = Shape::Box {
                half_extents: Vec3Fix::new(one, Fix128::from_ratio(1, 2), Fix128::from_int(2)),
            };
            let i = w
                .add_shaped_body(&s, one, Vec3Fix::from_int(-3, 0, 0))
                .unwrap();
            let j = w
                .add_shaped_body(&s, one, Vec3Fix::from_int(3, 1, 0))
                .unwrap();
            let a = w.get_body_mut(i).unwrap();
            a.velocity = Vec3Fix::from_int(4, 0, 0);
            a.angular_velocity = Vec3Fix::from_int(3, 1, 2);
            let b = w.get_body_mut(j).unwrap();
            b.velocity = Vec3Fix::from_int(-4, 0, 0);
            b.angular_velocity = Vec3Fix::from_int(-1, 5, 0);
        }
        // 境界の直下の角速度 (等方な慣性、|ω|² < 2⁶³) と速い回転
        2 => {
            w.config.gravity = Vec3Fix::ZERO;
            let mut a = RigidBody::new(Vec3Fix::ZERO, one);
            a.angular_velocity =
                Vec3Fix::new(Fix128::from_int(3_037_000_000), Fix128::ZERO, Fix128::ZERO);
            w.add_body(a);
            let mut b = RigidBody::new(Vec3Fix::from_int(10, 0, 0), one);
            b.angular_velocity = Vec3Fix::from_int(1 << 30, 1 << 30, 1 << 30);
            w.add_body(b);
            let mut c = RigidBody::new(Vec3Fix::from_int(20, 0, 0), one);
            c.angular_velocity = Vec3Fix::from_int(1, -2, 3);
            w.add_body(c);
        }
        // 半径和 2³¹ の球の重なり (2 乗 2⁶² は範囲内)
        3 => {
            w.config.gravity = Vec3Fix::ZERO;
            let r = Fix128::from_int(1 << 30);
            w.add_body_with_radius(RigidBody::new(Vec3Fix::ZERO, one), r);
            w.add_body_with_radius(
                RigidBody::new(Vec3Fix::from_int((1 << 31) - (1 << 20), 1 << 10, 0), one),
                r,
            );
        }
        // 位置 2⁶² 付近の衝突 (位置は差でしか積に入らない)
        //
        // ⚠️ 3 と同じ world に置くと、2⁶² 離れた対の `|delta|²` が範囲外になり
        // (修正前は wrap した値で比べていた) 範囲内の scene でなくなる
        5 => {
            w.config.gravity = Vec3Fix::ZERO;
            let far = Fix128::from_raw(1 << 62, 0);
            let mut a = RigidBody::new(Vec3Fix::new(far, far, far), one);
            a.velocity = Vec3Fix::from_int(5, 0, 0);
            w.add_body_with_radius(a, one);
            let mut b = RigidBody::new(
                Vec3Fix::new(
                    far + Fix128::from_int(3),
                    far,
                    far + Fix128::from_ratio(1, 3),
                ),
                one,
            );
            b.velocity = Vec3Fix::from_int(-5, 0, 0);
            w.add_body_with_radius(b, one);
        }
        // 関節でつないだ振り子 (TGS は substep ごとに関節を投影する)
        4 => {
            let a = w.add_body_with_radius(RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)), one);
            let mut bb = RigidBody::new_dynamic(Vec3Fix::from_int(2, 5, 0), one);
            bb.angular_velocity = Vec3Fix::from_int(0, 0, 1);
            let b = w.add_body_with_radius(bb, Fix128::from_ratio(1, 2));
            w.add_joint(alice_physics::Joint::Ball(alice_physics::BallJoint::new(
                a,
                b,
                Vec3Fix::ZERO,
                Vec3Fix::from_int(-2, 0, 0),
            )));
            let c = w.add_body_with_radius(
                RigidBody::new_dynamic(Vec3Fix::from_int(4, 5, 1), one),
                Fix128::from_ratio(1, 2),
            );
            w.add_joint(alice_physics::Joint::Ball(alice_physics::BallJoint::new(
                b,
                c,
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::new(-one, Fix128::ZERO, -one),
            )));
        }
        // 既存の検出経路 (積 v·dt が範囲外、XPBD は修正前から flag を立てる)
        _ => {
            w.config.gravity = Vec3Fix::new(Fix128::ZERO, -Fix128::from_int(1 << 30), Fix128::ZERO);
            w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, one));
        }
    }
    w
}

/// 各 scene を 60 frame 進めた `serialize_state` の指紋
fn in_range_fingerprints() -> Vec<(u32, SolverBackend, u64)> {
    let mut out = Vec::new();
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        for id in 0..7 {
            // 範囲外の検出経路の scene は修正で TGS の挙動が変わる (それが修正) ので XPBD だけ
            if id == 6 && backend == SolverBackend::Tgs {
                continue;
            }
            let mut w = in_range_scene(id, backend);
            let dt = if id == 6 {
                Fix128::from_int(1 << 20)
            } else {
                frame()
            };
            let steps = if id == 6 { 4 } else { 60 };
            for _ in 0..steps {
                w.step(dt);
            }
            out.push((id, backend, fnv1a(&w.serialize_state())));
        }
    }
    out
}

/// 範囲内の scene は、範囲の検査を足す前の solver と `serialize_state` が
/// bit 一致する (指紋は修正前の commit で同じ関数を走らせて記録した値)
///
/// `parallel` は積分と速度導出を rayon で回すが、body ごとに独立な計算なので
/// 同じ値になる (修正前の commit で両方を測って一致を確認)
#[test]
fn in_range_scenes_are_bit_identical_to_the_previous_solver() {
    let got = in_range_fingerprints();
    for (id, backend, h) in &got {
        println!("scene {id} {backend:?} {h:#018x}");
    }
    let want: [(u32, SolverBackend, u64); 13] = PREVIOUS_FINGERPRINTS;
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g, w, "scene {} {:?} の状態が修正前と違う", w.0, w.1);
    }
    // 範囲内の scene で flag は立たない (範囲外の scene 6 は除く)
    for id in 0..6 {
        let mut w = in_range_scene(id, SolverBackend::default());
        for _ in 0..60 {
            w.step(frame());
        }
        assert!(!w.overflow_detected(), "scene {id}");
    }
}

/// [`in_range_fingerprints`] を範囲の検査を足す前の solver で走らせた値
const PREVIOUS_FINGERPRINTS: [(u32, SolverBackend, u64); 13] = [
    (0, SolverBackend::Xpbd, 0x96d8862902a5e20e_u64),
    (1, SolverBackend::Xpbd, 0x5f0450309bb6bbd8_u64),
    (2, SolverBackend::Xpbd, 0x7413325ab6d09153_u64),
    (3, SolverBackend::Xpbd, 0xee886f5765d11d9f_u64),
    (4, SolverBackend::Xpbd, 0xad1a2717d4f33e7d_u64),
    (5, SolverBackend::Xpbd, 0x001b5e81c7e8bae2_u64),
    (6, SolverBackend::Xpbd, 0xbc92d7768c7f84fa_u64),
    (0, SolverBackend::Tgs, 0x98fab73c15c719bd_u64),
    (1, SolverBackend::Tgs, 0x4ffae91e2b8e5e12_u64),
    (2, SolverBackend::Tgs, 0x1dddca010d2df650_u64),
    (3, SolverBackend::Tgs, 0xa26ed4428533826f_u64),
    (4, SolverBackend::Tgs, 0x95bf9815df272c77_u64),
    (5, SolverBackend::Tgs, 0x8375e72116bbe2e0_u64),
];
