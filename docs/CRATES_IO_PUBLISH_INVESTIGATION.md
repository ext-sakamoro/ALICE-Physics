# crates.io publish 前調査 (v1.0 Roadmap Item J)

**調査日**: 2026-09-13
**調査 base**: alice-physics v0.13.0 (commit `d175d67` + v0.14.0-preview.3 landing)
**目的**: `cargo publish` に必要な整備タスクを特定、ROADMAP OQ2 の 3 択 (a/b/c) を実データから決定

## 前提: 現状の `publish = false`

`Cargo.toml` 13 行目で `publish = false` を明示指定、crates.io publish 不可の状態。本調査は該当行を **一時コメントアウトして dry-run** → **findings 集約後に復元** の手順で実施 (復元済、diff 0)。

## 実施した dry-run

```bash
$ cargo publish --dry-run --allow-dirty
    Updating crates.io index
error: failed to verify manifest at `/Users/ys/ALICE-Physics/Cargo.toml`

Caused by:
  all dependencies must have a version requirement specified when publishing.
  dependency `alice-analytics` does not specify a version
  Note: The published dependency will use the version from crates.io,
  the `path` specification will be removed from the dependency declaration.
```

## 判明した blocker

### B1: path dep に `version = "..."` fallback 未記述 (最上位 blocker)

該当 4 dep (全て optional / non-default feature gate):

| dep | path | feature | crates.io 状態 |
|--|--|--|--|
| `alice-ml` | `../ALICE-ML` v0.1.0 (local) | `neural` | ❌ 未 publish |
| `alice-db` | `../ALICE-DB` v0.1.0 (local) | `replay` | ❌ 未 publish |
| `alice-analytics` | `../ALICE-Analytics` v0.1.0 (local) | `analytics` | ❌ 未 publish |

Cargo requirement: `cargo publish` 時に **全 dep に `version = "..."` 指定必須**、path のみでは reject 現状のいずれかを追加すれば dry-run が次段に進む:

```toml
# 例: version fallback 追加パターン
alice-analytics = { path = "../ALICE-Analytics", version = "0.1", optional = true, default-features = false, features = ["std"] }
```

ただし **version = "0.1" fallback が有効になるのは、対応 crate が crates.io に publish されている場合のみ** 未 publish の状態で version 指定しても、cargo は crates.io に該当 version を探しに行き not found で失敗する

### B2: sibling repo が crates.io 未 publish

3 sibling (`alice-ml` / `alice-db` / `alice-analytics`) は全て local `v0.1.0` = crates.io に上がっていない、`cargo search` で 0 hit 確認済:

```bash
$ cargo search alice-ml
(no results)
$ cargo search alice-db
(no results)
$ cargo search alice-analytics
(no results)
```

### B3: `--all-features` build で path dep drift 検知

`cargo doc --no-deps --all-features` 実行時、以下 3 dep で **import path drift** を検知:

- `alice_db::AliceDB` — root にない
- `alice_ml::Ternary` / `alice_ml::TernaryWeight` — root にない
- `alice_analytics::prelude` — root にない

これは alice-physics が書かれた当時の sibling API と現在の sibling `v0.1.0` の間で drift が発生している事を示す (sibling 側で rename / restructure されたが、alice-physics の bridge module が未追従)

**影響範囲**: default features build と `alice-physics-std` build (which excludes optional deps) は clean、all-features build のみが失敗、通常運用 (feature 未指定) では顕在化しない latent 破損

## OQ2 の 3 択 (α/β/γ) 実データからの判定

ROADMAP.md OQ2 で挙げた 3 案:

### 案 (a): 依存 chain publish 先行 (sibling を先に crates.io へ)

**手順**:
1. `alice-analytics v0.1.0` を crates.io に publish (最も dep 少ない、まず起点として)
2. `alice-db v0.1.0` を publish
3. `alice-ml v0.1.0` を publish
4. `alice-physics` の Cargo.toml に `version = "0.1"` fallback 追加
5. `alice-physics` dry-run → 成功 → publish

**block**: 3 sibling がそれぞれ `publish = false` かつ MSRV / API / docs 未整備の可能性大 (別途 3 crate 分の準備工数、v1.0 roadmap の scope 外)

**評価**: **reject** — sibling を先に crates.io まで持っていく = 3 crate 分の polishing (MSRV / docs / feature audit) が必要、alice-physics v1.0 前の scope として大きすぎる

### 案 (b): bridge feature 削除 (ALICE-SDF v1.7.7 pattern)

**手順**:
1. `Cargo.toml` の `[features]` から `neural` / `replay` / `analytics` を削除 (feature 定義そのもの)
2. `[dependencies]` から `alice-ml` / `alice-db` / `alice-analytics` の 3 行を削除
3. `src/lib.rs` の bridge module 宣言に `#[cfg(feature = "neural")]` 等の gate 追加 (もし未 gate なら)
4. CHANGELOG に「temporarily removed for crates.io v0.14.x publish, restoration in v0.14.y once siblings publish」 note
5. Downstream (SBR ゲーム / ALICE-Bamboo) は継続して `git`/`path` dep 経由で feature 有効化可能
6. `cargo publish --dry-run` → 次段の issue (recovery) → 段階 fix

**block**: なし (ALICE-SDF v1.7.7 で実証済のパターン、pre-experience あり)

**評価**: **accept** — 案 (a) と比べて alice-physics 単独で完結、v0.14.x - v0.16.x の間で実施可能

### 案 (c): パラメータ化 trait で外部注入

**手順**:
1. 各 bridge (neural / replay / analytics) を `trait AliceMlBackend` / `trait AliceDbBackend` / `trait AliceAnalyticsBackend` として定義
2. dep を optional から完全削除、trait 定義のみ alice-physics に残置
3. 実装は downstream (alice-ml / alice-db / alice-analytics 側) で提供
4. Runtime injection: `PhysicsWorld::new().with_ml_backend(Box::new(SomeMlImpl))` 等

**block**: 大幅リファクタリング必要、bridge module 3 個 (`neural.rs` / replay 経路 / analytics 経路) の re-design、work 4-6 週間

**評価**: **defer** — architectural 改善として価値は高いが、v1.0 前 scope として heavy、v1.x post-release で検討

## 推奨判定

**案 (b) を v0.16.x で採用** (ADR-002 準拠、crates.io publish 事前試験を 0.x のうちに済ませる原則):

- v0.14.0 - v0.15.x: 引き続き `publish = false`、bridge feature 現状維持
- v0.16.x で bridge feature 削除 preview → dry-run → publish 実行
- v1.x post-release で案 (c) 移行検討 (bridge を trait 化 = downstream への切り離し)

## B3 (`--all-features` drift) の即応

path dep import 4 箇所を fix する追加作業が **v0.14.0 stable landing 前に必要**:

- `alice_db::AliceDB` → sibling API 側の renamed 版に更新 (`ALICE-DB` repo 側で確認要)
- `alice_ml::Ternary` / `TernaryWeight` → 同上
- `alice_analytics::prelude` → 同上

**scope**: v0.14.0 stable landing 前の必須修正 (現在 default build clean、all-features build 破損の latent bug、`cargo test --all-features` CI job が導入されたら失敗する)

## 実施タイムライン更新

ROADMAP.md § v0.16.0 section に本 findings を反映、以下を追加:

- **J-1** (v0.14.0 内): B3 の path dep drift 4 箇所を修正 (sibling API を最新に追従)
- **J-2** (v0.15.0 内): bridge feature の削除 preview commit (`neural` / `replay` / `analytics` 削除案の PR draft、CI で all-features build が pass することを確認)
- **J-3** (v0.16.x): `publish = false` 解除 + `cargo publish --dry-run` → 次 issue 洗い出し → `cargo publish` 実行
- **J-4** (v0.17.x): sibling 3 crate が crates.io publish された段階で bridge feature を段階復帰 (SDF v1.8.0 と同 pattern)

## 復元 verification

- ✅ `Cargo.toml` 13 行目: `publish = false` に復元済
- ✅ `git diff Cargo.toml` は本 preview 実施前と一致 (v0.14.0-preview.3 の Cargo.toml 変更は無し)
- ✅ この doc 自体が新規追加のみ、既存 file は非破壊

## 関連

- ROADMAP.md § v0.16.0 / OQ2 (本 doc の受け皿)
- ADR-002 (2026-09-12): v1.0.0 前に crates.io publish 事前試験実施
- ALICE-SDF v1.7.7 CHANGELOG: bridge feature 削除 pattern の pre-experience source
