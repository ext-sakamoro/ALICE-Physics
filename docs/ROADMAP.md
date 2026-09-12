# ALICE-Physics ROADMAP

Canonical roadmap for the alice-physics crate. Primary source of truth.
Memory index pointer: `[[reference-alice-physics-v1-roadmap]]` in claude-config.

## 現在位置 (2026-09-12)

**v0.13.0 landed** (commit `f37df0e`) — Session 4 push で 19 module 追加 (Tier ★★★ 4 + ★★ 7 + ★ 8)、ALICE-SDF v1.7.7 の `morphology` を S1 tier-★★★ integration partner として組合わせて 20 module 完備

- module 総数: 146 src file、`pub mod` 144
- Session 1-3 (v0.10-0.12) baseline: 1175 lib tests + 53 alice-bamboo 統合 tests
- Cargo.toml: `publish = false` (crates.io 未公開状態)、`rust-version = "1.70.0"`

Related sub-roadmaps:
- [`GPU_OFFLOAD_ROADMAP.md`](GPU_OFFLOAD_ROADMAP.md) — ALICE-TRT companion GPU offload progression

## Phase 一覧

### ✅ v0.10.0 - v0.12.0 (Session 1-3、shipped 2026-07-08)

Engineering-solver 基盤 35 module + 3 統合 solver loop — 3D プリント安全性 (warp / thin-wall / stress / bridging) から composite / plastic / fatigue 力学、乱流、VOF / level-set 多相流、実行可能な CFD 時間ステップ loop、GpuSolverBridge joint-solve pipeline (ALICE-TRT v3.1.0 協調)

### ✅ v0.13.0 (Session 4、19 module、shipped 2026-09-12)

commit `f37df0e`

- **Tier ★★★ (5)**: G1 ragdoll / G2 buoyancy_zone / R1 laminate_failure / R2 transient_thermal / S1 morphology (ALICE-SDF v1.7.7 側)
- **Tier ★★ (7)**: G3 wind_zone / S4 sdf_character / R3 rolling_contact / G5 netcode_prediction / G6 character_state / S3 sdf_sph / R4 kinematic_loop
- **Tier ★ (8)**: G4 ik_physics_bridge / G7 anisotropic_friction / R5 aeroelasticity / R6 piezoelectric / R7 acoustic_wave / R8 electromagnetic / S2 sdf_fem_mesh / S5 sdf_wind_field
- **CFD refinement**: MacCormack advection (Fedkiw monotone limiter) / velocity self-advection / pressure Poisson RHS scale fix

### 🚧 v0.14.0 (推定 2-4 週間) — quick wins + API surface audit 前半

- **G. MSRV policy 明記** (1-2 日、quick win) — `rust-version = "1.70.0"` 既記述、README / docs に policy 説明追加のみ (「N-3 stable channel まで支援」等)
- **B. Public API surface freeze 前半** — priority module (net / character / SDF 系) の `pub` → `pub(crate)` audit 着手 (144 pub mod のうち、`net_prediction` / `character*` / `sdf_*` 系から)
- **D. Documentation 完備** — `#![deny(missing_docs)]` を lib.rs に追加、`cargo doc --no-deps 2>&1 | grep warning` で残 warning 列挙 → fix
- **新 example 追加** — ragdoll / SPH / joint / character の代表 example (現状 7 → 12 個目標)

自己採点 target: 品質 90/100 (現状 100/100 optimization scorecard は維持、public API 完成度で -10)

### 🚧 v0.15.0 (推定 2-4 週間) — API stability signaling

- **B. Public API surface freeze 後半** — 残 module audit + `#[non_exhaustive]` 戦略適用 (struct / enum の一部) + `#[deprecated]` alias で v1.0 名確定 (v0.x で名前変えたい API に marker + alias、v1.0 で確定名だけ残す)
- **C. cargo-semver-checks / cargo-public-api CI 通し** — `.github/workflows/security-audit.yml` に既に semver-checks job あり、拡張して cargo-public-api snapshot を repo に commit、PR diff で API surface 変化を可視化
- **F. Fuzz coverage 拡張** — 現状 `fuzz_collision` + `fuzz_step` の 2 target → joint / SDF CCD / trimesh / `cfd_solver` / `structural_solver` で 5-8 target 追加、24h 実行 crash 0 実績を CHANGELOG に記載

### 🚧 v0.16.0 (推定 3-4 週間) — determinism guarantee + ecosystem + publish 準備

- **E. Determinism CI 6 環境 matrix** — macOS ARM + macOS x86 + Linux ARM + Linux x86 + Windows + WASM で毎 PR bit-exact snapshot golden test を run、joint / cloth / fluid / SDF CCD / trimesh に拡張
- **H. Ecosystem 契約 freeze** — ALICE-TRT `GpuSolverBridge` trait / ALICE-SDF `SdfField` trait / ALICE-Bamboo / ALICE-Anima / ALICE-Kinematics との integration point の method signature freeze、各 partner crate と semver policy 契約書化
- **J. crates.io publish 準備** — 依存 chain (`alice-ml` / `alice-db` / `alice-analytics`) の crates.io 公開状況調査、ALICE-SDF v1.7.7 で経験した「bridge feature 削って crates.io 対応」パターンの pre-experience

### ⏳ v0.16.1 or v0.17.0 (推定 3-5 日) — crates.io publish 実績作り

- **J 続き** — `Cargo.toml` の `publish = false` を解除、`cargo publish --dry-run` → 依存 chain の残タスク処理 → `cargo publish` 実行
- crates.io に初 publish、`cargo add alice-physics` で外部 downstream が使えるようになる
- v1.0.0 前の publish trial-and-error 完了 (0.x のうちに済ませて stable 直前の risk 排除、ADR-002 準拠)

### ⏳ v1.0.0-rc.1 (推定 6-8 週間)

- **I. Migration guide 執筆** — `docs/MIGRATION_0.x_TO_1.0.md` に API rename / removal / deprecation 一覧 + downstream 対応手順
- crates.io に rc.1 publish (`0.16.1` → `1.0.0-rc.1` の pre-release version)
- 実 downstream (ALICE-Bamboo / ALICE-Anima / SBR ゲーム側) の 1.0 対応調整、feedback 期間 4 週間

### ⏳ v1.0.0-rc.2 (推定 2-3 週間)

- rc.1 feedback 反映、breaking change 発生時は rc.3 も許容

### ⏳ v1.0.0 stable

- **semver 契約発動** — `alice-physics = "1"` で下流 pin 可能に、breaking change は必ず major bump

## 未決 (Open Questions)

### OQ1: pub API の scope 決定

- 144 pub mod 中 何個を `pub(crate)` に格下げすべきか?
- prelude に入れる型は canonical か? (現状の `alice_physics::prelude` の中身を再確認必要)
- **判断保留、v0.14.0 で B の priority module (net / character / SDF 系) audit 先行実施 → 結果次第で決定**

### OQ2: crates.io publish 前の依存 chain 対応

- `alice-ml` / `alice-db` / `alice-analytics` の crates.io 未公開状況の確認 (ALICE-SDF v1.7.7 が experienced した「bridge feature 削って crates.io 対応」の焼き直しになる可能性大)
- 選択肢: (a) feature flag で optional dep 化、(b) bridge feature 分離 (SDF v1.7.7 と同 pattern)、(c) パラメータ化 trait で外部注入
- **v0.16.0 で J の pre-flight 調査時に決定**

### OQ3: MSRV 昇格戦略

- 現状 `rust-version = "1.70.0"`、v1.0 時点で 1.75+ に昇格するか
- v1.x 中で MSRV 昇格したら minor bump で告知するか (Serde と同じ運用パターン)
- **v0.14.0 の G 実施時に policy 決定**

### OQ4: v1.0 直行 (γ) vs 段階昇格 (α) の判断軸再確認

- 「downstream crate が今から `alice-physics = "1"` で pin して 6 ヶ月 breaking change なしを我々が保証できるか?」の y/n で決定
- 現状は「Unreleased 反映済だが実運用検証未了」で α 判定継続
- **v0.16.0 landing 時点で再評価**

## 判断記録 (ADR)

### ADR-001 (2026-09-12): v0.13.0 で Session 4 20 module を単一 release として ship する

**背景**: v0.12.0 (2026-07-08) 以降、Unreleased で 20 module (Tier ★★★ 5 + ★★ 7 + ★ 8) が積み上がっていた

**決定**: v0.13.0 として 1 release にまとめて ship、tier 構造を CHANGELOG / README に明記

**代替案**:
- v0.13.0 = Tier ★★★ のみ、v0.14.0 = Tier ★★、v0.15.0 = Tier ★ (段階 ship) → **reject**: overhead 大、20 module は依存関係が交差してるので分離コスト高
- v1.0.0 直行 → **reject**: 実運用検証未了、public API surface freeze 未完 (詳細は OQ1 + Item B 参照)

**根拠**: Session 4 として 3 tier 構造で発表することで「1.0 に向けた最後の module 追加 wave」の signaling になる v1.0.0 は API freeze の意味を持たせ、feature 追加は v0.14.0-v0.16.0 で緩やかに

### ADR-002 (2026-09-12): v1.0.0 前に crates.io publish を試験実施する (v0.16.x 前後)

**背景**: 現状 Cargo.toml `publish = false` 指定、外部 downstream は git dep 経由のみ

**決定**: v0.14.0 - v0.16.0 の polishing 期間中に依存 chain (`alice-ml` / `alice-db` / `alice-analytics`) の crates.io 公開対応を進め、v0.16.x で `publish = false` 解除 → `cargo publish`

**代替案**:
- v1.0.0 で初 publish → **reject**: publish 経路の trial-and-error を stable 直前にやると risk 大
- 永遠に `publish = false` のまま → **reject**: 1.0 の意味が薄い、外部 OSS contributor が cargo add できない

**根拠**: publish の trial-and-error は 0.x のうちに済ませる (ALICE-SDF が v1.7.7 で経験した「bridge feature 削って crates.io 対応」パターンの焼き直しになる想定、pre-experience 蓄積が本命)

### ADR-003 (2026-09-12): 4-6 ヶ月 α timeline を採用 (γ 直行 / β RC 短縮 の代わりに)

**背景**: user が「Physics 1.0 並であればバージョンあげてもいいかもね」と可能性示唆

**決定**: α (v0.13.0 → v0.14.0 → ... → v0.16.x → rc.1 → rc.2 → v1.0.0 の段階昇格) を採用

**代替案**:
- **γ (直行)**: Cargo.toml `0.13.0 → 1.0.0` に bump、README も 1.0 表記に、Session 4 module も全部「v1.0.0 included」に格上げ → reject: 実運用ドッグフーディング未実施、public API surface freeze 未完
- **β (RC 短縮)**: 現在の main を `v1.0.0-rc.1` として freeze publish、feedback 経て v1.0.0 → reject: Unreleased backlog は既に v0.13.0 で吸収済だが、新 module (client_prediction 等) の実運用検証がない状態で RC 出すと feedback の意味が薄い

**根拠**: 「downstream crate が今から `alice-physics = "1"` で pin して 6 ヶ月 breaking change なしを我々が保証できるか?」 = 現状 n 判定 (実運用検証未了)、ALICE-Bamboo と ALICE-Anima の実運用で 3 ヶ月連続 breaking change なしを先に達成することが 1.0 コミットの根拠になる (semver 契約は約束ではなく実績の追認)

## 最短優先候補 (v0.14.0-preview として 1 週間内 landing 可能)

以下 3 項目は独立で並行実施可能、user 明示指示で個別着手:

1. **G. MSRV policy 明記** (1-2 日、quick win) — Cargo.toml `rust-version = "1.70.0"` 既記述、README + `docs/MSRV.md` に policy 説明追加のみ
2. **D. `#![deny(missing_docs)]` 追加** (半日 + fix) — lib.rs に attribute 追加、`cargo doc --no-deps 2>&1 | grep warning` で残 warning 列挙 → module ごとに fix (推定 1-2 週間)
3. **J. publish 前調査** (半日) — `cargo publish --dry-run` で依存 chain の未公開 crate 洗い出し → 対応方針決定 (OQ2)

## 一番効くマイルストーン (α timeline の根拠)

**「ALICE-Bamboo と ALICE-Anima の実運用で 3 ヶ月連続 breaking change なし」を先に達成する** — B-I の作業を進めながらも、実 downstream での使用実績が 1.0 コミットの根拠になる semver 契約は約束ではなく実績の追認

## 関連

- ALICE-SDF v1.7.7 CHANGELOG — S1 morphology integration partner
- [ALICE-TRT v3.1.0](https://github.com/ext-sakamoro/ALICE-TRT/releases/tag/v3.1.0) — GpuSolverBridge joint-solve GPU offload companion release
- CLAUDE.md § ROADMAP 自律作成 / 更新規律 — 本 ROADMAP.md 作成の根拠
- CLAUDE.md § auto memory loop 遅延禁止 — Phase 完了時の memory 更新規律
- Memory pointer: `reference_alice_physics_v1_roadmap.md` in `~/.claude/projects/-Users-ys/memory/` (横断 view、詳細は本 file 参照)
