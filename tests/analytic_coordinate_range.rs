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
//! | `Vec3Fix::length_squared` / `length` / `normalize` | `\|v\| < 2³¹·⁵ ≈ 3.04e9` | wrap、`length` は 0、`normalize` は ZERO |
//! | `Vec3Fix::cross` | 各積 `< 2⁶²` | wrap |
//! | `QuatFix::rotate_vec` (単位 q) | `\|v\| ≤ 2⁶²` で wrap しない | 絶対誤差が `\|v\|` に比例 |
//! | `Mat3Fix::inverse` | `2⁻⁶³ < \|det\| < 2⁶³` | 符号反転した逆行列 / `None` |
//! | `Shape::mass_and_inertia` | `ρ·8·L⁵ < 2⁶¹` | 明示の `Err` |
//! | 剛体の位置積分 | `\|x\| < 2⁶³` | wrap、overflow flag は立たない |
//! | 角速度の積分 | `\|ω\| < 2³¹·⁵` | 回転が止まり ω が 0 に消える |
//! | `apply_impulse_at` の torque | `\|r\|·\|J\| < 2⁶²` | wrap |
//! | 球同士の接触 | 半径和・中心距離 `< 2³¹·⁵` | 重なりを見逃す |
//! | `ForceField::Point` | 中心からの距離 `< 2³¹·⁵` | 力が 0 / 上限値に飛ぶ |
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

/// characterization: `Mat3Fix::inverse` は `|det|` が `2⁶³` 以上でも `2⁻⁶³` 以下でも
/// `Some` で **符号の反転した** 逆行列を返し、その外側では `None`
///
/// 対角 `s` の 3×3 なら `det = s³` ⇒ 正しい範囲は `2⁻²¹ < s < 2²¹`
#[test]
fn characterization_mat3_inverse_sign_flips_outside_the_det_range() {
    let d = |f: Fix128| Mat3Fix::diagonal(f, f, f);
    let small = |n: u32| Fix128::from_raw(0, 1u64 << (64 - n));
    // 範囲内
    assert_eq!(d(pow2(20)).inverse().unwrap().col0.x, small(20));
    assert_eq!(d(small(20)).inverse().unwrap().col0.x, pow2(20));
    // det = 2⁶³ が -2⁶³ に wrap ⇒ 逆行列が負
    assert_eq!(d(pow2(21)).inverse().unwrap().col0.x, -small(21));
    // 1/det = 2⁶³ が -2⁶³ に wrap ⇒ 逆行列が負
    assert_eq!(d(small(21)).inverse().unwrap().col0.x, -pow2(21));
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

/// characterization: `+x` の端 (`2⁶³ − 1`) を越えて進む物体は `−2⁶³` 側へ wrap し、
/// overflow flag は立たない (XPBD / TGS とも)
///
/// flag の検査は `velocity·dt` と速度導出の積だけで、位置の加算は見ない
/// 速度導出 `(x − x_prev)/h` も wrap した差で正しい速度を出すので、物体は
/// 反対側の端から何事もなく動き続ける
#[test]
fn characterization_position_wraps_at_2_pow_63_without_the_flag() {
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        let mut w = edge_world(backend);
        w.step(frame());
        let b = w.get_body(0).unwrap();
        assert!(
            b.position.x.is_negative(),
            "{backend:?}: 端で wrap していない"
        );
        assert!(b.velocity.x > Fix128::from_int(100), "{backend:?}");
        assert!(
            !w.overflow_detected(),
            "{backend:?}: flag が立った (表を更新)"
        );
    }
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE position add at +-2^63 wraps without raising overflow_detected"]
fn position_wrap_at_2_pow_63_raises_the_overflow_flag() {
    for backend in [SolverBackend::default(), SolverBackend::Tgs] {
        let mut w = edge_world(backend);
        w.step(frame());
        assert!(w.overflow_detected(), "{backend:?}");
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

/// characterization: 積 `v·dt` が範囲外になる scene (重力 `−2³⁰`, `dt = 2²⁰`) で、
/// XPBD は flag を立てて位置を止めるが、TGS は flag を立てず位置が wrap して
/// 正の側へ飛ぶ (落下しているのに `y > 0`)
#[test]
fn characterization_tgs_backend_does_not_raise_the_overflow_flag() {
    let dt = Fix128::from_int(1 << 20);
    let mut xpbd = washing_world(SolverBackend::default());
    let mut tgs = washing_world(SolverBackend::Tgs);
    for _ in 0..4 {
        xpbd.step(dt);
        tgs.step(dt);
    }
    assert!(xpbd.overflow_detected());
    assert_eq!(xpbd.get_body(0).unwrap().position.y, Fix128::ZERO);
    assert!(!tgs.overflow_detected());
    assert!(tgs.get_body(0).unwrap().position.y > pow2(60));
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE TGS backend never raises overflow_detected"]
fn tgs_backend_raises_the_overflow_flag() {
    let mut tgs = washing_world(SolverBackend::Tgs);
    for _ in 0..4 {
        tgs.step(Fix128::from_int(1 << 20));
    }
    assert!(tgs.overflow_detected());
}

fn spinning_world(w: Fix128) -> PhysicsWorld {
    let mut world = zero_gravity_world();
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.angular_velocity = Vec3Fix::new(w, w, w);
    world.add_body(b);
    world
}

/// characterization: `|ω| ≥ 2³¹·⁵` rad/s では `normalize_with_length` の長さが 0 に
/// なり回転が積分されず、速度導出で ω 自体も 0 に消える (flag なし)
#[test]
fn characterization_angular_speed_from_2_pow_31_5_is_erased() {
    let mut ok = spinning_world(pow2(30));
    ok.step(frame());
    assert_ne!(ok.get_body(0).unwrap().rotation, QuatFix::IDENTITY);

    let mut w = spinning_world(pow2(31));
    w.step(frame());
    let b = w.get_body(0).unwrap();
    assert_eq!(b.rotation, QuatFix::IDENTITY);
    assert_eq!(b.angular_velocity, Vec3Fix::ZERO);
    assert!(!w.overflow_detected());
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE angular speed beyond 2^31.5 is erased without a fault"]
fn angular_speed_beyond_range_is_kept_or_flagged() {
    let mut w = spinning_world(pow2(31));
    w.step(frame());
    let b = w.get_body(0).unwrap();
    assert!(w.overflow_detected() || b.angular_velocity != Vec3Fix::ZERO);
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

/// characterization: 半径和が `2³¹·⁵` を越える球の対は、重なっていても
/// `dist² < (r_a + r_b)²` の両辺が wrap して接触を見逃す
///
/// 同じ配置を 2⁻¹⁰ に縮めると押し離される (対照)
#[test]
fn characterization_sphere_pair_beyond_2_pow_31_5_is_missed() {
    let (a, b) = sphere_pair(3 << 19, 3_000_000_000 >> 10);
    assert!(
        a < Fix128::ZERO && b > Fix128::from_int(3_000_000_000 >> 10),
        "対照が離れていない"
    );

    let (a, b) = sphere_pair(3 << 29, 3_000_000_000);
    assert_eq!(a, Fix128::ZERO);
    assert_eq!(b, Fix128::from_int(3_000_000_000));
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE sphere pair with combined radius beyond 2^31.5 is not separated and no fault is raised"]
fn sphere_pair_beyond_range_is_separated_or_flagged() {
    let (a, _) = sphere_pair(3 << 29, 3_000_000_000);
    assert!(a < Fix128::ZERO);
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

/// characterization: 点力場は中心からの距離が `2³¹·⁵` を越えると `dist²` が wrap し、
/// 力が 0 になるか上限値 (真値の 2e19 倍) に飛ぶ
#[test]
fn characterization_point_field_beyond_2_pow_31_5() {
    let z = Fix128::ZERO;
    // 範囲内: 2⁶⁰ / (2³⁰)² = 1、中心向き
    assert_eq!(
        point_force(Vec3Fix::new(pow2(30), z, z), pow2(60)),
        Vec3Fix::new(-Fix128::ONE, z, z)
    );
    // 3·2³⁰: 真値 0.44、実測は上限へ
    let f = point_force(Vec3Fix::new(pow2(31) + pow2(30), z, z), pow2(62));
    assert!(f.x < -pow2(60), "{f}");
    // 2³² / 3 軸 2³¹: 0
    assert_eq!(
        point_force(Vec3Fix::new(pow2(32), z, z), pow2(62)),
        Vec3Fix::ZERO
    );
    assert_eq!(
        point_force(Vec3Fix::new(pow2(31), pow2(31), pow2(31)), pow2(62)),
        Vec3Fix::ZERO
    );
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE point force field beyond 2^31.5 returns zero or a capped kick instead of strength/r^2"]
fn point_field_beyond_range_matches_the_inverse_square() {
    let z = Fix128::ZERO;
    let f = point_force(Vec3Fix::new(pow2(32), z, z), pow2(62));
    // 2⁶² / 2⁶⁴ = 0.25、中心向き
    assert!((f.x.to_f64() + 0.25).abs() < 1e-9, "{f}");
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

/// characterization: `SpatialGrid::hash` は `x / cell` の積が wrap すると中央の cell に
/// 落ち、`hi + half` が i64 を越えると **debug では panic、release では端の cell**
/// になる (profile で挙動が違う唯一の箇所として実測)
#[test]
fn characterization_spatial_grid_hash_wraps_and_differs_by_profile() {
    let z = Fix128::ZERO;
    let g = SpatialGrid::new(Fix128::from_ratio(1, 4), 16);
    // 2⁶² · 4 = 2⁶⁴ → 0 ⇒ 原点と同じ cell
    assert_eq!(g.hash(Vec3Fix::new(pow2(62), z, z)), g.hash(Vec3Fix::ZERO));

    let g1 = SpatialGrid::new(Fix128::ONE, 16);
    let edge = Vec3Fix::new(Fix128::from_raw(i64::MAX, 0), z, z);
    let r = std::panic::catch_unwind(|| g1.hash(edge));
    if cfg!(debug_assertions) {
        assert!(r.is_err(), "debug で panic しなかった");
    } else {
        // wrap して負 → x の cell は 0 (正しくは 15)
        assert_eq!(r.unwrap() % 16, 0);
    }
}

#[test]
#[ignore = "src gap: WORLD-V1-RANGE SpatialGrid::hash wraps to a wrong cell (and panics in debug) instead of clamping"]
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
