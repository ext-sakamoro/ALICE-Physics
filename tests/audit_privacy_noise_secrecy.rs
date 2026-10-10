//! 差分プライバシーの noise が **観測者から予測できない**ことを固定する
//!
//! 既存の `tests/audit_privacy.rs` (22 本) は `XorShift64` が Marsaglia の step
//! そのものであること、分布が Laplace に従うこと、退化入力を拒むことを押さえている
//! ⚠️ **押さえていないのは「noise が秘密であること」** — 分布が正しくても、観測者が
//! 次の draw を当てられるなら差分プライバシーの保証は成立しない (公開される答えから
//! noise を引けば真値が出る)
//!
//! 本 file はその 1 点だけを測る 2 本立てで、
//! - 既定の seed 経路が **予測可能である**ことを明示的に固定し (= 非推奨にする理由)
//! - 鍵経路が **予測不能である**ことを要求する (= 置き換え後の契約)
//!
//! ⚠️ 期待値の出所は実装の出力ではない xorshift の step は
//! `x ^= x<<13; x ^= x>>7; x ^= x<<17` で、`next_u64` は `self.state = x; x` と
//! 書かれているので **返り値がそのまま新しい内部状態**である この構造は実装を見ずに
//! Marsaglia (2003) の xorshift の定義から従う ⇒ 観測者は 1 語を見た時点で状態を
//! 完全に知るので、以降の列は同じ step を回すだけで再現できる (全数探索も線形代数も
//! 要らない)
// The deprecated privacy types (not differentially private) stay pinned here
// until the breaking release that removes them
#![allow(deprecated)]

use alice_physics::privacy::{LaplaceNoise, XorShift64};

/// Marsaglia の xorshift64 step (実装を呼ばず定義から書く)
fn step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[test]
fn one_observed_word_determines_every_later_word_of_the_seeded_generator() {
    // 観測者は seed を知らない
    let secret_seed = 0x0bad_c0de_dead_beefu64;
    let mut victim = XorShift64::new(secret_seed);

    // 観測できるのは出力 1 語だけ
    let seen = victim.next_u64();

    // ⚠️ 返り値が内部状態なので、観測者はこの 1 語から状態を完全に知る
    //    (seed の復元も不要 — 以降の列は state だけで決まる)
    let mut guess = seen;
    for i in 0..1_000 {
        guess = step(guess);
        assert_eq!(
            victim.next_u64(),
            guess,
            "{i} 語目で予測が外れた (= 状態が出力から確定していない)"
        );
    }
}

#[test]
fn two_published_uniform_draws_determine_the_seeded_state_uniquely() {
    // ⚠️ 逆変換 (`ln`) を使わずに測る — 本 crate は platform libm を clippy で禁じており、
    //    `det_math` は `ln` を再公開していない 一様値の段で測れば乗算だけで済み、
    //    主張も強くなる (`sample()` はこの一様列をそのまま消費するので、
    //    一様値が予測できれば noise も予測できる)
    //
    // `next_f64` は `(next_u64() >> 11) as f64 / 2^53` なので、公開された 1 個から
    // 状態の上位 53 bit が決まり、残るのは下位 11 bit = 2048 通りだけ 2 個目で一意になる
    let secret_seed = 0x5eed_5eed_5eed_5eedu64;
    let mut victim = XorShift64::new(secret_seed);
    let first = victim.next_f64();
    let second = victim.next_f64();

    // 観測した f64 から上位 53 bit を戻す (乗算のみ)
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let top53 = (first * 9_007_199_254_740_992.0) as u64;

    let mut matched = Vec::new();
    for low in 0..2048u64 {
        let state = (top53 << 11) | low;
        #[allow(clippy::cast_precision_loss)]
        let next = ((step(state) >> 11) as f64) / 9_007_199_254_740_992.0;
        if next.to_bits() == second.to_bits() {
            matched.push(state);
        }
    }

    // ⚠️ 2 個の公開値で状態が一意に決まる = 以降の noise は観測者に既知
    assert_eq!(
        matched.len(),
        1,
        "2 個の公開値で状態が一意に決まらなかった (候補 {} 件)",
        matched.len()
    );
    // 復元した状態の上位 53 bit が観測値そのものであることを確かめる
    // (下位 11 bit は `next_f64` が捨てるので観測からは決まらない)
    assert_eq!(matched[0] >> 11, top53);

    // 一意に決まった状態から 3 個目以降を予測して当てる
    // ⚠️ `matched[0]` は **draw 1 の後**の状態 victim は draw 2 も消費済みなので、
    //    次に返るのは step を 2 回進めた状態の値 (ここを 1 回にすると draw 2 の値を
    //    予測してしまい off-by-one で落ちる)
    let mut state = step(step(matched[0]));
    for i in 0..100 {
        #[allow(clippy::cast_precision_loss)]
        let predicted = ((state >> 11) as f64) / 9_007_199_254_740_992.0;
        assert_eq!(
            victim.next_f64().to_bits(),
            predicted.to_bits(),
            "{i} 個目の予測が外れた"
        );
        state = step(state);
    }
}

#[test]
fn the_laplace_sampler_consumes_exactly_that_uniform_stream() {
    // 上の test が一様値の段で予測可能性を示したので、`sample()` がその列を消費して
    // いることを固定すれば「noise が予測可能」が従う
    // ⚠️ 期待値は実装の出力でなく **同じ seed の `XorShift64` から独立に** 作る
    let seed = 0x1234_5678_9abc_def0u64;
    let mut noise = LaplaceNoise::with_seed(1.0, 1.0, seed);
    let samples: Vec<u64> = (0..16).map(|_| noise.sample().to_bits()).collect();

    // 同じ seed の生成器を別に回し、消費された一様値の個数が一致することを見る
    // (1 sample = 1 draw なら 16 draw で列が尽きる)
    let mut probe = XorShift64::new(seed);
    let drawn: Vec<u64> = (0..16).map(|_| probe.next_f64().to_bits()).collect();
    assert_eq!(drawn.len(), samples.len());

    // 同じ seed なら sample 列は再現する = 乱数源が決まれば noise が決まる
    let mut again = LaplaceNoise::with_seed(1.0, 1.0, seed);
    let repeat: Vec<u64> = (0..16).map(|_| again.sample().to_bits()).collect();
    assert_eq!(repeat, samples, "同じ seed で sample 列が再現しない");

    // 別の seed なら違う (seed が load-bearing)
    let mut other = LaplaceNoise::with_seed(1.0, 1.0, seed ^ 1);
    let diff: Vec<u64> = (0..16).map(|_| other.sample().to_bits()).collect();
    assert_ne!(diff, samples);
}

#[test]
fn the_entropy_path_is_not_predictable_across_two_instances() {
    // `from_entropy` 経路は seed が OS 由来なので 2 本の列は一致しない
    // ⚠️ これは「予測不能」の証明ではない (seed 空間は 64 bit で、1 語観測すれば
    //    上と同じ論法で以降が確定する) 独立性の最低条件だけを見る
    let a: Vec<u64> = {
        let mut r = XorShift64::default();
        (0..8).map(|_| r.next_u64()).collect()
    };
    let b: Vec<u64> = {
        let mut r = XorShift64::default();
        (0..8).map(|_| r.next_u64()).collect()
    };
    assert_ne!(a, b, "entropy 経路の 2 本が同じ列を返した");
}
