# 配線負債台帳 (WIRING_DEBT)

実装されているが production path から呼ばれていない item を、解消予定つきで追跡する台帳
`scripts/wiring_guard.py` (CI の `fmt` job + preflight) が機械検査する違反のうち、**内容を分析して閉じ方が分かっているもの**を ID つきで記録する

## この台帳と baseline の役割分担

| | 対象 | 形式 |
|---|---|---|
| `scripts/wiring-baseline.txt` | 既存の違反 **全件** (現在 `unwired` 587 / `dead_code` 17) | 機械可読のラチェット 新規違反のみ fail / 解消済 entry の残置も fail (`stale_baseline`) |
| **本台帳** | そのうち **分析済で閉じ方が決まっているもの** | 人間可読 ID (`WD-nnn`) + 証跡 + 閉じる条件 |

本台帳は baseline を置き換えない 587 件すべてを転記する意図もない (ID を振る価値があるのは「何が足りないか分かっているもの」だけ)

## marker の実態 (⚠️ rule と実装で語彙が違う)

`scripts/wiring_guard.py` の doc が契約の正本:

| 検査 | 対象 | 免除の書き方 |
|---|---|---|
| **A** `dead_code` | `#[allow(dead_code)]` / `#![allow(dead_code)]` | `// ALLOW-DEAD: <12 字以上の理由>` を直前に置く or baseline |
| **B** `unwired` | `src/` の `pub` / `pub(crate)` な fn / struct / enum / const / static / trait / type / union で、test と自身の宣言と `use` 行を除いた production code に 1 度も現れないもの | `// ALLOW-UNWIRED: <12 字以上の理由>` を直前に置く or baseline |
| **C** 空振り防止 | 検査対象が 0 件 | fail (検査器が空振りして green になるのを防ぐ) |

⚠️ `rules/analytic-oracle-tests.md` (2026-10-01 制定) は marker を **`// wiring: <ID>`** / **`// api: <理由>`** と書いているが、**実装は `ALLOW-UNWIRED:` / `ALLOW-DEAD:` で、`api:` に相当する分類を持たない** (恒久的な公開入口も `ALLOW-UNWIRED` か baseline に入る) rule 自身が「機械検査 (設計中)」と書いている部分で、**どちらの語彙に寄せるかは未決** (下記 `WD-000`)

⚠️ 既存の使用実績: `ALLOW-UNWIRED` **1 file** / `ALLOW-DEAD` **0 file** (ほぼ全部 baseline 側)

## 検査器の限界 (fail-open 側)

- **名前の字面一致で数える**ので、他の item / field / method と同名なら「配線済」と誤判定する (`new` 等) 偽陽性より偽陰性を選んでいる
- ⚠️ **推移的な未配線** (未配線 item からしか呼ばれない item) は検出されない
- ⚠️ **caller 数で捕まらない残り** (P2 には繋いだが P3 には無い / `thermal` は `solve` に届くが FEM 残差には無い) は **caller 数では原理的に見えない** ⇒ 破壊試験の **配線変異** が担当する (`rules/analytic-oracle-tests.md`)

## 運用

1. 未配線を残す時は marker (上表) か baseline に載せる + **本台帳に `open` で起票**する
2. 対の oracle を **`#[ignore = "src gap: <ID>"]`** で同時に置く (閉じ方が test で表現できる場合)
3. 配線したら **ignore を外し / 本台帳を `closed` にし / marker を消す** の 3 点セット
4. ⚠️ `src gap:` の ignore が残ったまま通るのは stale (配線したのに外し忘れ)

---

## open

### `WD-000` — rule と実装で marker 語彙が食い違っている (⚠️ user 判断待ち)

| | |
|---|---|
| 状態 | **open** (2026-10-01 起票) |
| 事象 | `rules/analytic-oracle-tests.md` は `// wiring: <ID>` / `// api: <理由>` を要求、`scripts/wiring_guard.py` は `// ALLOW-UNWIRED:` / `// ALLOW-DEAD:` を要求し `api:` 相当を持たない |
| 影響 | rule どおりに書くと guard が通さず、guard どおりに書くと rule の ID 対応が付かない |
| 閉じる条件 | どちらかに寄せる (a) guard を `wiring:` / `api:` に対応させる (b) rule を `ALLOW-*` 語彙に合わせる ⚠️ **(a) なら既存 baseline 604 件の移行方針が要る** |

### `WD-001` — Jacobi / BiCGStab 圧力解法と P2G scatter が production から呼ばれない

| | |
|---|---|
| 状態 | **open** (2026-10-01 起票、実測 `8b3ecdd`) |
| 証跡 | `src/eulerian_grid.rs:99-102` が逐語で `// Reserved algorithm variants (Jacobi / BiCGStab pressure, P2G scatter) are pub(crate) but currently unused outside their own unit tests — awaiting cfd_solver integration.` + `#![allow(dead_code)]` `project_pressure_bicgstab` は `pub(crate)` で、呼び出し 4 件すべてが `#[cfg(test)] mod tests` の内側 |
| 影響 | ⚠️ 外部 crate から呼べないので、1e8 規模での反復数を外部 probe で測れない |
| 閉じる条件 | `cfd_solver` への配線 ⚠️ **ただし優先度は低い** — BiCGStab の反復は `O(n)` ≈ 500 で、`project_pressure_multigrid` (`02309ed` で landing) の `O(1)` には届かない (実測: multigrid 19/20/21 cycles vs GS 167/602/2271、n=8/16/32) |
| 対の oracle | 未設置 (配線しないなら不要、配線するなら `#[ignore = "src gap: WD-001"]` で反復数の上限を pin する) |

### `WD-002` — RANS / 壁関数は実装 587 行があるが u_τ の出し元が crate に無い

| | |
|---|---|
| 状態 | **open** (2026-10-01 起票、実測 `8b3ecdd`) |
| 証跡 | `src/turbulence.rs:43` が `#![allow(dead_code)] // Reserved RANS/wall-function API — awaiting cfd_solver integration` ⚠️ **module の一部は配線済** — `src/cfd_solver.rs:38` が `use crate::turbulence::{smagorinsky_eddy_viscosity, strain_rate_magnitude, SMAGORINSKY_CS}` (LES / Smagorinsky 側) ⇒ **未配線なのは RANS / 壁関数側** |
| ⚠️ 真の blocker | **u_τ (摩擦速度) を計算して出す code が crate に 1 つも無い** `u_tau` / `friction_velocity` / `tau_w` / `wall_shear` の `src/` 7 ヒットは**すべて `turbulence.rs` 内の引数名** (`:263` field / `:270` 使用 / `:326` 引数 / `:327` / `:334` / `:336` / `:337`) ⇒ **受け取る側しか無い** |
| 閉じる条件 | (1) u_τ の算出 (壁隣接 cell の速度勾配から) を実装 → (2) `cfd_solver` に配線 ⚠️ **「既存 587 行を繋ぐだけ」では閉じない** |
| ⚠️ 設計上の注意 | **u_τ を u⁺ プロファイルの逆解で求めてから「u⁺ が対数則に乗る」を assert すると循環する** (逆関数を順関数で検査しているだけで、κ / B / プロファイル自体の誤りが見えない) ⇒ `rules/analytic-oracle-tests.md` の「基準が内側」に該当 |
| 対の oracle | 未設置 (u_τ の実装と同時に、外部の基準 = 実験相関 or 別実装と突合する形で置く) |

### `WD-003` — `gpu_sdf::batch_size()` / `SIMD_WIDTH` は値を消費する内部 code が無い

| | |
|---|---|
| 状態 | **open** (2026-10-01 起票、実測 `8b3ecdd`) |
| 証跡 | `batch_size()` (`src/gpu_sdf.rs:306`) の `src/` ヒットは **doc 例 (`:300` / `:301`) と `lib.rs:556` / `:708` の re-export のみ** ⇒ **production caller 0** 返す値は `crate::math::SIMD_WIDTH` |
| ⚠️ 性質 | **これは「配線」で閉じる負債ではない** — 恒久的な公開入口 (外部 caller が chunk 幅として使う) なので、rule の `// api: <理由>` 側に分類されるべきもの ⇒ `WD-000` の決着待ち |
| ⚠️ 併発する実問題 | `simd_width()` は `target_feature = "avx2"` で **8**、それ以外で **4** を返すが、⚠️ **`-C target-feature=+avx2` が CI / preflight のどこにも無い**ので **8 の枝は未 compile** 実測 (Paperspace x86_64): `+avx2` を付けると `SIMD_WIDTH=8` になり lib test **1818 passed / 0 failed** ⚠️ **green の理由は「正しいから」でなく「値を誰も消費していないから」** `math.rs:3651` の `assert!(matches!(simd_width(), 4 \| 8))` は**両方を受けるので degenerate** |
| 閉じる条件 | ⚠️ **CI job を足すより test を足す** — `#[cfg(target_feature = "avx2")] assert_eq!(simd_width(), 8)` / `#[cfg(not(...))] assert_eq!(simd_width(), 4)` で **flag ごとに値を pin する** (現状は「4 か 8 のどちらか」しか言っていない) |

### `WD-004` — slab 局所化と越境 transport が `pub(crate)` で外部から規模計測できない

| | |
|---|---|
| 状態 | **open** (2026-10-01 起票、実測 `8b3ecdd`) |
| 証跡 | `src/eulerian_grid.rs` の slab 系 `pub(crate) fn` / `pub(crate) struct` が **20 件** 越境 driver は **test 名** (`eulerian_grid::tests::cross_process_slab_rank_worker`) |
| 影響 | ⚠️ **「複数プロセスで 1e8」が外部 probe crate からも `examples/` からも測れない** 越境の bit 一致は 9³ / 8³ / 3³ (最大 729 cell) までしか実測されていない |
| ⚠️ 性質 | 内部実装なので `pub` 化は公開面の増加になる (v1.0 の pub audit 6 回の反転) ⇒ **`pub` 化で閉じるのは採らない** |
| 閉じる条件 | **越境 test の格子寸法を env で可変にする** (test 側の小変更) ⚠️ ただし優先度は multigrid の後 — 律速は並列度でなく反復数で、GS に分散を掛けても 1e8 は 1,024 rank で 3 時間、multigrid は 1 台で 22.8 分 (実測) |

### `WD-005` — CFD Armaly backward-facing step の再付着長が文献値に届かない

| | |
|---|---|
| 状態 | **open** (既存、本台帳には 2026-10-01 に転記) |
| 証跡 | `tests/armaly_backward_step.rs:1939` の `#[ignore = "src gap: x_1 = 4.3921 against Gartling's 6.10 (72.0 %, short by 13.7 cells) and no upper-wall bubble at all. ..."]` |
| ⚠️ 性質 | **配線負債ではなく solver 精度の負債** (`src gap:` の ignore 語彙を共有しているので本台帳に併記する) ⚠️ ID 形式の `src gap: WD-005` には**書き換えていない** (既存 message に診断が入っており、書き換えると情報が落ちる) ⇒ `WD-000` の決着時に形式を揃えるか判断する |
| 閉じる条件 | 解像度 (`ny=32` は 1 点 2.4〜9.5h で CI 予算外) ⚠️ **scheme を変えても `ny=8` 固定では文献値を跨ぐ**ので、残る軸は解像度のみ |

---

## closed

(まだ無い)
