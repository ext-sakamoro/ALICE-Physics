#!/usr/bin/env python3
"""csprng_guard.py — 予測されて困る値が予測可能な源から来ていないかを検査する

## なぜ要るのか

差分プライバシーの noise は「分布が正しい」だけでは保証にならない 観測者が次の draw を
当てられるなら、公開した答えから noise を引いて真値が出る 2026-10-09 の横断実測では、
同一の xorshift 実装が 4 crate に写っており、どれも次の 3 つを同時に持っていた

1. `next_u64` が `self.state = x; x` = **返り値がそのまま内部状態** ⇒ 出力 1 語で以降が確定
   (線形代数も全数探索も要らない)
2. `from_entropy` が時刻を seed にする ⇒ 呼び出し時刻の前後を試すだけで seed が復元できる
3. Laplace を浮動小数点の逆変換で引く ⇒ 出力の下位 bit が一様乱数を漏らす
   (Mironov, "On Significance of the Least Significant Bits for Differential Privacy",
   CCS 2012) 対策は snapping mechanism か整数 (離散) noise

⚠️ **「決定論 PRNG」と「秘匿が要る乱数」は別物** simulation の再現性のための PRNG は
正しい設計で、同じ型を秘匿が要る経路が共有しているのが欠陥 ⇒ 本 guard は定義を一律に
禁じず、**理由の明示**を求める 秘匿が要る経路だけは理由を認めない

## 検査

- 検査 A: 予測可能な PRNG の定義 (`self.state = …` の step / xorshift / splitmix / LCG)
    `// DETERMINISTIC-RNG: <12 字以上の理由>` を直前に置くか baseline に載せる
    ⚠️ **秘匿が要る経路 (privacy / dp / auth / token / nonce / secret / key) では理由を認めない**
    CSPRNG (ChaCha20 等) を使う
- 検査 B: 時刻を seed にする経路 (同じ関数内で時刻の値が `seed` / `from_entropy` に流れる)
    ⚠️ sink に `::new(` を入れてはいけない — あらゆる構築で現れるので、時刻を計測して
    いるだけの関数が全部当たる (本 repo の `#[cfg(test)]` の計測 fn 5 件で実測)
    marker 不可 hash を挟んでも seed 空間が時刻幅に縮むので、CSPRNG を使う
- 検査 C: 差分プライバシーの浮動小数点逆変換 (noise を返す関数の中の `ln` / `exp`)
    `// FLOAT-DP-ACCEPTED: <12 字以上の理由>` を直前に置くか baseline に載せる
    ⚠️⚠️ **CSPRNG を使っていても対象** Mironov の漏洩は乱数源の質ではなく
    浮動小数点の逆変換そのものが原因なので、検査 A / B を満たしても独立に残る
- 検査 D: **検査対象が 0 件なら fail** (検査器が空振りして green になるのを防ぐ)

baseline (`scripts/csprng-baseline.txt`) は既存の違反を記録するラチェットで、新規の違反
だけが fail する 解消された entry が残っていても fail (stale_baseline)

## 使い方

    python3 scripts/csprng_guard.py                 # 検査 (exit 0 = 新規違反なし)
    python3 scripts/csprng_guard.py --baseline-write # 既存分を baseline に記録
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "csprng-baseline.txt"

#: 理由つき marker
MARKER_RNG = re.compile(r"//\s*DETERMINISTIC-RNG\s*:(.*)$")
MARKER_DP = re.compile(r"//\s*FLOAT-DP-ACCEPTED\s*:(.*)$")
MIN_REASON = 12

#: 検査 A — 予測可能な PRNG の step
#: ⚠️ `self.state =` だけでは足りない 実測した偽陽性 3 件 (本 repo):
#:    - `self.state == SleepState::Sleeping` ... `=` が `==` の 1 文字目に当たる (`sleeping.rs`)
#:    - `self.state = self.state.wrapping_mul(FNV_PRIME)` ... FNV **hash** で PRNG ではない (`sketch.rs`)
#:    ⇒ (a) `=` は `==` を除く (b) **同じ impl が乱数を配る method を持つこと**を条件にする
STATE_ASSIGN = re.compile(r"\bself\.state\s*=(?!=)")
#: 乱数を配る method (これが同じ impl に無ければ、状態の代入は PRNG ではない)
GENERATOR_FN = re.compile(r"\bfn\s+(?:next_(?:u8|u16|u32|u64|u128|usize|i\d+|f32|f64|bool|range)\w*|gen_\w+|random\w*)\s*[(<]")
NAMED_PRNG = re.compile(r"\b(?:struct|enum)\s+(XorShift\w*|SplitMix\w*|Lcg\w*|Pcg\w*|WyRand\w*)\b")

#: 検査 B — 時刻の値
CLOCK = re.compile(r"\b(?:SystemTime::now|Instant::now)\s*\(")
#: その値が seed に流れる形 (同じ関数の中)
#: ⚠️ `::new(` を入れてはいけない — あらゆる構築で現れるので、時刻を計測しているだけの
#:    関数が全部当たる (Physics の `#[cfg(test)]` の計測 fn 5 件で実測した偽陽性)
SEED_SINK = re.compile(r"\b(?:seed|from_entropy|from_seed)\b")

#: 検査 C — noise を返す関数の名前
DP_FN = re.compile(r"\bfn\s+(\w*(?:laplace|sample|privatize|noise|randomize)\w*)\s*[(<]", re.I)
#: 浮動小数点の逆変換
TRANSCENDENTAL = re.compile(r"\b(?:ln|ln64|ln32|exp|exp64|exp32)\s*\(|\.ln\(\)|\.exp\(\)")
#: 既に対策済とみなす形
#: ⚠️⚠️ **CSPRNG を入れてはいけない** Mironov の漏洩は乱数源の質ではなく
#:    **浮動小数点の逆変換そのもの**が原因なので、`SecureRng` 由来の一様値でも起きる
#:    (実測: `alice-crypto` の `laplace_from` は CSPRNG 由来の `u` を `ln64` に通しており、
#:     module doc に "Neither is implemented here yet" と自分で書いている)
#:    ⇒ 対策と数えるのは **snapping / 離散 (整数) noise** だけ
DP_SAFE = re.compile(r"\b(?:snapping|discrete_laplace|geometric_mechanism|integer_laplace)\b")

#: 秘匿が要る経路 (理由を認めない) — path か module 名で判定
SECRET_PATH = re.compile(r"(?:privacy|\bdp\b|auth|token|nonce|secret|keygen|csprng)", re.I)

#: CSPRNG を使っていれば検査 A/B の対象外
CSPRNG = re.compile(r"\b(?:ChaCha20|SecureRng|OsRng|getrandom|alice_crypto::dp)\b")


def remove_cfg_test(code: str) -> str:
    """`#[cfg(test)]` の直後の module / 関数を落とす

    ⚠️ `src/` の中に test が同居するのが Rust の既定なので、これを落とさないと
    計測用の `Instant::now()` が「時刻 seed」として当たる (Physics で 5 件実測)
    中括弧の対応で範囲を取り、同じ長さの空白に置き換えて行番号を保つ
    """
    out = code
    for m in list(re.finditer(r"#\[cfg\(test\)\]", code)):
        brace = out.find("{", m.end())
        if brace < 0:
            continue
        depth, i = 0, brace
        while i < len(out):
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        span = out[m.start() : i + 1]
        out = out[: m.start()] + re.sub(r"[^\n]", " ", span) + out[i + 1 :]
    return out


def strip_line_comments(src: str) -> str:
    """`//` 以降を空白に潰す (marker の検出は別に行うので、ここでは落としてよい)

    ⚠️ 文字列リテラル中の `//` は潰さない (URL 等が壊れる)
    """
    out = []
    for line in src.splitlines():
        in_str = False
        esc = False
        cut = len(line)
        i = 0
        while i < len(line) - 1:
            c = line[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str and c == "/" and line[i + 1] == "/":
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def marker_above(lines: list[str], idx: int, pat: re.Pattern[str]) -> str | None:
    """直前の非空行 (attribute / doc comment は飛ばす) に marker があるか"""
    j = idx - 1
    while j >= 0:
        s = lines[j].strip()
        if not s:
            j -= 1
            continue
        m = pat.search(lines[j])
        if m:
            return m.group(1).strip()
        if s.startswith("#[") or s.startswith("///") or s.startswith("//!"):
            j -= 1
            continue
        return None
    return None


def impl_bodies(code: str) -> list[tuple[int, str]]:
    """(開始位置, 本体) を `impl` ごとに返す

    ⚠️ 状態の代入が PRNG かどうかは「同じ impl が乱数を配る method を持つか」で決まる
    (FNV hash も state machine も `self.state = …` を書くので、代入だけでは区別できない)
    """
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"\bimpl\b", code):
        brace = code.find("{", m.end())
        if brace < 0:
            continue
        # `impl` から `{` までに `;` があれば宣言ではない (別の構文を拾った)
        if ";" in code[m.end() : brace]:
            continue
        depth, i = 0, brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((brace, code[brace : i + 1]))
    return out


def fn_bodies(code: str) -> list[tuple[str, int, str]]:
    """(関数名, 開始行 index, 本体) を返す 中括弧の対応で本体を切る"""
    out: list[tuple[str, int, str]] = []
    for m in re.finditer(r"\bfn\s+(\w+)", code):
        brace = code.find("{", m.end())
        if brace < 0:
            continue
        depth = 0
        i = brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((m.group(1), code[: m.start()].count("\n"), code[brace : i + 1]))
    return out


def rs_files(root: Path) -> list[Path]:
    """`src/` 配下の .rs (test / bench / example は対象外 — 本番経路だけを見る)"""
    src = root / "src"
    if not src.is_dir():
        return []
    return sorted(p for p in src.rglob("*.rs") if "/target/" not in str(p))


def audit(root: Path) -> tuple[list[str], int]:
    """(違反の一覧, 検査した箇所の件数) を返す"""
    violations: list[str] = []
    checked = 0

    for path in rs_files(root):
        rel = str(path.relative_to(root))
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        code = remove_cfg_test(strip_line_comments(raw))
        secret = bool(SECRET_PATH.search(rel))
        has_csprng = bool(CSPRNG.search(code))

        # --- 検査 A: 予測可能な PRNG の定義 ---
        # 乱数を配る method を持つ impl の中の状態代入だけを見る (+ 名前つき PRNG の型)
        sites: list[int] = []
        for brace, body in impl_bodies(code):
            if not GENERATOR_FN.search(body):
                continue
            for m in STATE_ASSIGN.finditer(body):
                sites.append(brace + m.start())
        sites.extend(m.start() for m in NAMED_PRNG.finditer(code))
        for pos in sorted(set(sites)):
                ln = code[:pos].count("\n")
                checked += 1
                if has_csprng and not secret:
                    continue
                reason = marker_above(lines, ln, MARKER_RNG)
                if secret:
                    violations.append(
                        f"{rel}:{ln + 1}: 秘匿が要る経路に予測可能な PRNG がある "
                        f"(理由を書いても認めない — CSPRNG を使う)"
                    )
                elif reason is None:
                    violations.append(
                        f"{rel}:{ln + 1}: 予測可能な PRNG の定義に "
                        f"`// DETERMINISTIC-RNG: <理由>` が無い"
                    )
                elif len(reason) < MIN_REASON:
                    violations.append(
                        f"{rel}:{ln + 1}: DETERMINISTIC-RNG の理由が短い "
                        f"({len(reason)} 字、{MIN_REASON} 字以上)"
                    )

        # --- 検査 B: 時刻を seed にする経路 ---
        # ⚠️ 証拠は本体の `seed` 語だけでは足りない 本 repo の `from_entropy` は
        #    `Self::new(h.finish())` と書くので本体に `seed` が現れない
        #    ⇒ **関数名**も証拠に数える (`from_entropy` / `*seed*` / `*rng*`)
        for name, ln, body in fn_bodies(code):
            if not CLOCK.search(body):
                continue
            if not (SEED_SINK.search(body) or SEED_SINK.search(name) or "rng" in name.lower()):
                continue
            checked += 1
            if CSPRNG.search(body):
                continue
            violations.append(
                f"{rel}:{ln + 1}: `{name}` が時刻を seed にしている "
                f"(hash を挟んでも seed 空間が時刻幅に縮む — CSPRNG を使う)"
            )

        # --- 検査 C: 差分プライバシーの浮動小数点逆変換 ---
        for m in DP_FN.finditer(code):
            name = m.group(1)
            brace = code.find("{", m.end())
            if brace < 0:
                continue
            depth, i = 0, brace
            while i < len(code):
                if code[i] == "{":
                    depth += 1
                elif code[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = code[brace : i + 1]
            if not TRANSCENDENTAL.search(body):
                continue
            ln = code[: m.start()].count("\n")
            checked += 1
            if DP_SAFE.search(body):
                continue
            reason = marker_above(lines, ln, MARKER_DP)
            if reason is None:
                violations.append(
                    f"{rel}:{ln + 1}: `{name}` が浮動小数点の逆変換で noise を作っている "
                    f"(Mironov CCS 2012) snapping / 離散 noise にするか "
                    f"`// FLOAT-DP-ACCEPTED: <理由>` を置く"
                )
            elif len(reason) < MIN_REASON:
                violations.append(
                    f"{rel}:{ln + 1}: FLOAT-DP-ACCEPTED の理由が短い "
                    f"({len(reason)} 字、{MIN_REASON} 字以上)"
                )

    return violations, checked


def load_baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {
        ln.strip()
        for ln in BASELINE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def main() -> int:
    violations, checked = audit(ROOT)

    if "--baseline-write" in sys.argv:
        BASELINE.write_text(
            "# csprng_guard.py の baseline — 既存の違反を記録するラチェット\n"
            "# 新規の違反だけが fail する 解消したら行を消す (残っていても fail)\n"
            "# ⚠️ ここに載っている行は「予測可能な noise が残っている」という負債の一覧\n"
            + "".join(f"{v}\n" for v in violations),
            encoding="utf-8",
        )
        print(f"[csprng-guard] baseline を書いた: {len(violations)} 件 (検査 {checked} 箇所)")
        return 0

    # 検査 D: 対象 0 件で fail (空振りを green と読ませない)
    if checked == 0:
        print(
            "[csprng-guard] FAIL: 検査対象を 1 件も見つけなかった 検査器が空振りしている",
            file=sys.stderr,
        )
        return 1

    base = load_baseline()
    now = set(violations)
    new = sorted(now - base)
    stale = sorted(base - now)

    rc = 0
    if new:
        print(f"[csprng-guard] FAIL: 新規の違反が {len(new)} 件", file=sys.stderr)
        for v in new:
            print(f"  {v}", file=sys.stderr)
        rc = 1
    if stale:
        print(
            f"[csprng-guard] FAIL: baseline に解消済の entry が {len(stale)} 件残っている "
            f"(`--baseline-write` で締め直す)",
            file=sys.stderr,
        )
        for v in stale:
            print(f"  {v}", file=sys.stderr)
        rc = 1
    if rc == 0:
        print(
            f"[csprng-guard] OK: 検査 {checked} 箇所 / 新規違反 0 "
            f"(baseline に既存 {len(base)} 件)"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
