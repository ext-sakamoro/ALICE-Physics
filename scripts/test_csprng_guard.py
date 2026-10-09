#!/usr/bin/env python3
"""test_csprng_guard.py — `csprng_guard.py` の歯を陽性対照つきで確かめる

⚠️ 検査器は「何も検出しないまま green」になりうるので、**検出すべき形を与えて
検出することと、検出すべきでない形を素通りすることの両方**を測る
(CI / preflight では本 file を guard 本体の **前** に走らせる)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csprng_guard  # noqa: E402

# --- 検査 A ---
BAD_STATE = """
pub struct Rng { state: u64 }
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        self.state = x;
        x
    }
}
"""
GOOD_STATE = """
pub struct Rng { state: u64 }
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        // DETERMINISTIC-RNG: lockstep の再現性のため、秘匿は要らない経路
        self.state = x;
        x
    }
}
"""
SHORT_REASON = """
pub struct Rng { state: u64 }
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        // DETERMINISTIC-RNG: 再現用
        self.state = 1;
        1
    }
}
"""
NAMED = """
pub struct XorShift64 { s: u64 }
"""

# --- 検査 B ---
CLOCK_SEED = """
impl Rng {
    pub fn from_entropy() -> Self {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        Self::new(seed)
    }
}
"""
CLOCK_ONLY = """
pub fn elapsed_ms(&self) -> u64 {
    let t = Instant::now();
    t.elapsed().as_millis() as u64
}
"""
CLOCK_BUT_CSPRNG = """
impl Rng {
    pub fn from_entropy() -> Self {
        let _unused = SystemTime::now();
        let seed = getrandom::getrandom(&mut buf);
        Self::new(seed)
    }
}
"""

# --- 検査 C ---
FLOAT_DP = """
pub fn sample(&mut self) -> f64 {
    let u = self.rng.next_f64() - 0.5;
    -self.scale * ln64(1.0 - 2.0 * u.abs())
}
"""
FLOAT_DP_MARKED = """
// FLOAT-DP-ACCEPTED: 公開面でなく内部の試験専用、監査済の経路
pub fn sample(&mut self) -> f64 {
    let u = self.rng.next_f64() - 0.5;
    -self.scale * ln64(1.0 - 2.0 * u.abs())
}
"""
# ⚠️ CSPRNG 由来の一様値でも浮動小数点の逆変換なら Mironov の対象 (検出されるのが正しい)
FLOAT_DP_CSPRNG = """
pub fn sample(&mut self) -> f64 {
    let u = SecureRng::next_f64_open01(&mut self.rng);
    -self.scale * ln64(1.0 - 2.0 * u.abs())
}
"""
# 対策済とみなすのは snapping / 離散 noise だけ
FLOAT_DP_SNAPPED = """
pub fn sample(&mut self) -> f64 {
    let u = SecureRng::next_f64_open01(&mut self.rng);
    snapping(-self.scale * ln64(1.0 - 2.0 * u.abs()), self.lattice)
}
"""
NO_TRANSCENDENTAL = """
pub fn sample(&mut self) -> i64 {
    self.rng.next_u64() as i64 % 7
}
"""

CFG_TEST_TIMING = """
#[cfg(test)]
mod tests {
    #[test]
    fn timing_of_a_seeded_scene() {
        let seed = 7u64;
        let t = Instant::now();
        let _ = build(seed);
        assert!(t.elapsed().as_millis() < 500);
    }
}
"""

CASES: list[tuple[str, str, str, int, str]] = [
    # (ラベル, 置く path, 中身, 期待違反数, 補足)
    ("A 予測可能な step を検出", "src/rng.rs", BAD_STATE, 1, ""),
    ("A 理由つきは通す", "src/rng.rs", GOOD_STATE, 0, ""),
    ("A 理由が短いと fail", "src/rng.rs", SHORT_REASON, 1, ""),
    ("A 名前つき PRNG の型も検出", "src/rng.rs", NAMED, 1, ""),
    ("A 秘匿経路では理由を認めない", "src/privacy.rs", GOOD_STATE, 1, "path が privacy"),
    ("B 時刻 seed を検出", "src/rng.rs", CLOCK_SEED, 1, ""),
    # ⚠️ この case は検査 0 件が正しい (時刻の値が seed に流れないので検査 B の対象外)
    #    「検査したうえで通る」側の陰性対照は下の CSPRNG 併用が担う
    (
        "B 時刻の計測だけは当たらない",
        "src/metrics.rs",
        CLOCK_ONLY,
        0,
        "seed に流れない / 検査 0 件",
    ),
    ("B CSPRNG 併用は通す", "src/rng.rs", CLOCK_BUT_CSPRNG, 0, ""),
    ("C 浮動小数点の逆変換を検出", "src/dpnoise.rs", FLOAT_DP, 1, ""),
    ("C 理由つきは通す", "src/noisegen.rs", FLOAT_DP_MARKED, 0, ""),
    # ⚠️ 旧 case は「CSPRNG なら通す」としていたが、それは検査器のバグを固定していた
    #    Mironov は乱数源の質では直らないので、CSPRNG でも検出するのが正しい
    ("C CSPRNG でも逆変換は対象", "src/noisegen.rs", FLOAT_DP_CSPRNG, 1, "Mironov は源の質で直らない"),
    ("C snapping は通す", "src/noisegen.rs", FLOAT_DP_SNAPPED, 0, ""),
    ("C 超越関数が無ければ対象外", "src/noisegen.rs", NO_TRANSCENDENTAL, 0, "検査 0 件"),
    # ⚠️ `#[cfg(test)]` の中で時刻を計測しつつ `seed` という名前も出てくる形
    #    これを落とさないと本番経路でない計測 fn が全部「時刻 seed」として当たる
    #    (本 repo の実測で 5 件の偽陽性)
    (
        "B cfg(test) の計測は対象外",
        "src/solver.rs",
        CFG_TEST_TIMING,
        0,
        "cfg(test) を落とす / 検査 0 件",
    ),
]


def run_case(path: str, body: str) -> tuple[list[str], int]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
        return csprng_guard.audit(root)


def main() -> int:
    failures = 0
    for label, path, body, want, note in CASES:
        got, checked = run_case(path, body)
        ok = len(got) == want
        # ⚠️ 「違反 0 が正解」の case では、検査箇所が 0 でないことも確かめる
        #    (何も見ずに 0 件を返すのと区別する) 検査 0 件が正しい case は note に書く
        if want == 0 and checked == 0 and "検査 0 件" not in note:
            ok = False
            note = (note + " ⚠️ 検査 0 箇所 = 空振り").strip()
        mark = "ok  " if ok else "FAIL"
        extra = f"  ({note})" if note else ""
        print(f"{mark} {label}: 違反 {len(got)}/{want} 検査 {checked} 箇所{extra}")
        if not ok:
            failures += 1
            for v in got:
                print(f"       {v}")

    # 検査 D (0 件 fail) 自体の歯 — 空の tree で検査 0 件になることを確かめる
    with tempfile.TemporaryDirectory() as td:
        got, checked = csprng_guard.audit(Path(td))
        ok = checked == 0 and not got
        print(f"{'ok  ' if ok else 'FAIL'} D 空の tree では検査 0 箇所 (本体が fail する側)")
        if not ok:
            failures += 1

    total = len(CASES) + 1
    print(f"\n{total - failures}/{total} pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
