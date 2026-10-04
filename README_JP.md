# ALICE-Physics

Rust 向けの決定論的物理エンジン
剛体コアは 128 bit 固定小数点で計算するため、同じ入力からは CPU・コンパイラ・OS によらず同じビット列が得られる

[English](README.md) | 日本語

[![crates.io](https://img.shields.io/crates/v/alice-physics.svg)](https://crates.io/crates/alice-physics)
[![docs.rs](https://img.shields.io/docsrs/alice-physics)](https://docs.rs/alice-physics)
[![MSRV](https://img.shields.io/crates/msrv/alice-physics)](#最小サポート-rust-バージョン)
[![CI](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/alice-physics.svg)](#ライセンス)

ビット単位の再現が必須になる用途を想定している
ロールバック / ロックステップ型のネットコード、サーバー側でのリプレイ検証、再現性が求められる工学・研究計算などである
剛体ソルバーのほかに工学系モジュール (FEM、CFD、伝熱、複合材、3D プリント検査) を含み、これらにも同じクロスプラットフォーム決定論を適用している
<!-- claim-test: golden_sim_field -->
<!-- claim-test: determinism_freefall -->

浮動小数点のゲームエンジンをそのまま置き換えるものではない
固定小数点演算には時間がかかり、数千体が相互作用するシーンを 60 fps で回す用途は設計の想定外

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [決定論](#決定論)
- [含まれるもの](#含まれるもの)
- [検証と既知の不具合](#検証と既知の不具合)
- [Cargo feature](#cargo-feature)
- [バインディング](#バインディング)
- [性能](#性能)
- [最小サポート Rust バージョン](#最小サポート-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## インストール

```sh
cargo add alice-physics
```

標準ライブラリなし (`alloc` が必要):

```sh
cargo add alice-physics --no-default-features
```

## 使用例

重力で 1 秒間落下する物体の例
同じコードが `src/lib.rs` の crate レベル doctest になっており、`cargo test` でコンパイル・実行される

```rust
use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Fix128, Vec3Fix};

// Create physics world
let config = PhysicsConfig::default();
let mut world = PhysicsWorld::new(config);

// Add a dynamic body
let body = RigidBody::new_dynamic(
    Vec3Fix::from_int(0, 10, 0),  // position
    Fix128::ONE,                   // mass = 1
);
let body_id = world.add_body(body);

// Step simulation (60 frames at 1/60 second)
let dt = Fix128::from_ratio(1, 60);
for _ in 0..60 {
    world.step(dt);
}

// Body should have fallen under gravity
let pos = world.bodies[body_id].position;
assert!(pos.y < Fix128::from_int(10), "Body fell under gravity");
```

[`examples/`](examples/) に実行可能なプログラムがある
最初に見るなら次のあたり:

| Example | 内容 |
|---------|------|
| [`basic_physics`](examples/basic_physics.rs) | ワールドの作成、物体の追加、ステップ |
| [`world_api_tour`](examples/world_api_tour.rs) | `PhysicsWorld` の setter・クエリ・イベント取得をひと通り |
| [`rollback_netcode`](examples/rollback_netcode.rs) | スナップショット、ロールバック、チェックサム照合 |
| [`joint_limits_and_breaking`](examples/joint_limits_and_breaking.rs) | ジョイントの種類、リミット、モーター、破断 |
| [`cloth_simulation`](examples/cloth_simulation.rs) | XPBD の布 |
| [`cfd_smoke_plume`](examples/cfd_smoke_plume.rs) | 統合 CFD ソルバー |
| [`print_full_safety`](examples/print_full_safety.rs) | 3D プリント安全性パイプライン |

```sh
cargo run --release --example rollback_netcode
```

## 決定論

全モジュールがプラットフォーム間でビット一致する
理由は次の 2 通り:

| 区分 | モジュール | ビットが一致する理由 |
|------|-----------|---------------------|
| **Fix128 コア** | 剛体ソルバー、ジョイント、距離・接触拘束、BVH ブロードフェーズ、GJK / EPA、CCD、スリープ、シーン I/O、ネットコードのスナップショットとロールバック、ニューラルコントローラー、公開 API が `Fix128` / `Vec3Fix` / `QuatFix` のモジュール全般 | 整数演算のみ `tests/determinism_golden.rs` で固定 |
| **`f32` / `f64` を使うモジュール** (一覧は [英語版](README.md#determinism) を参照、`scripts/f32_modules.py` が生成) | IEEE 754 の `+ - * / sqrt` と FMA は Rust の全ターゲットで結果が一意 超越関数 (`sin`、`exp`、`ln`、`powf` など) はすべて [`alice-det-math`](https://crates.io/crates/alice-det-math) (`alice_physics::det_math` として再エクスポート) を通し、プラットフォームの `libm` は使わない `clippy.toml` の設定で `libm` の呼び出しは CI エラーになる `tests/determinism_golden_f32.rs` で固定 |

2 行目のモジュール一覧は `scripts/f32_modules.py` が生成し、英語版の記述が `src/` とずれると CI が失敗する
ほかに 9 モジュールが入出力の境界 (FFI、Python、リプレイなど) でだけ浮動小数点を扱い、シミュレーション計算には使わない

2 つの golden テストは CI 上で macOS (ARM / x86)、Linux (ARM / x86)、Windows、`wasm32-wasip1` で実行している
<!-- claim-test: test_determinism_golden_hash -->

**利用者が渡すコード** `ClosureSdf` は利用者のクロージャを受け取る
ソルバー自体は決定論的だが、クロージャの中では `f32::sin` などの代わりに `alice_physics::det_math::{sin, exp, …}` を呼ぶ必要がある

**対象外** 基本演算で IEEE 754 に従わないターゲット (SSE2 を使わない 32 bit x86 `i586` など) と、fast-math 系のフラグを付けたビルド

**決定論は正しさを意味しない** ビットが一致するのは全員が同じ数値を計算しているということで、その数値が正しいかどうかは別に検証している
[検証と既知の不具合](#検証と既知の不具合) を参照

## 含まれるもの

公開モジュールの全一覧 (分野別、1 行説明付き) は [`docs/MODULES.md`](docs/MODULES.md) (英語) にある
API の詳細は [docs.rs](https://docs.rs/alice-physics) を参照

| 分野 | 主な内容 |
|------|---------|
| 剛体 | XPBD ソルバー (時間方向 Gauss-Seidel バックエンドも選択可)、スリープとアイランド、CCD、ロールバック用の状態シリアライズ |
| 衝突 | GJK / EPA、線形 BVH または永続的な動的 AABB 木のブロードフェーズ、箱・球・カプセル・円柱・円錐・楕円体・トーラス・くさび・凸包・複合形状・三角形メッシュ・高さ場・SDF コライダー |
| ジョイント | ボール、ヒンジ、固定、スライダー、ばね、D6、コーンツイストと、プーリー、ギア、溶接、ラックアンドピニオン、マウス 破断とPD モーター |
| 柔軟体 | XPBD のロープと布 (自己衝突あり)、位置ベース流体、FEM-XPBD 変形体、切断 |
| ゲーム用途 | キャラクターコントローラー、車両、ラグドール、IK 連携、クライアント側予測、決定論的乱数、接触イベント |
| 固体力学 | P1 / P2 / P3 四面体の線形弾性 FEM、共回転による大回転、J2 塑性、超弾性、熱-構造連成、適応細分化、梁、座屈、疲労、複合材 |
| 流体と場 | 複数の圧力ソルバーを持つ MAC 格子 CFD、RANS / LES 乱流モデル、VOF とレベルセット、SPH、圧縮性流れ、伝熱、Maxwell FDTD |
| 3D プリント | 材料データベース、薄肉・オーバーハング検査、反り、層間接着、造形向き、それらをまとめた安全性パイプライン |
| 2D | 独立した 2D XPBD エンジン (専用の形状とジョイント) |

**領域分割** CFD の圧力投影は `z` 方向のスラブに分けて 1 層のハローを交換しながら実行でき、ランク数によらず単一プロセスの解とビット一致する
測定は 1 台のホスト内のみ (ループバック TCP で最大 8 プロセス)
複数ホストにまたがる実行は行っておらず、MPI バックエンドもない

## 検証と既知の不具合

`tests/` の解析解テストと参照テストは、閉形式解や公表されている参照データ (教科書の公式、キャビティ流れの Ghia らの表など) と結果を比較している
golden ハッシュは変化を検出するだけなので、これらとは別に扱っている

[`docs/oracle-status.md`](docs/oracle-status.md) は `tests/` から自動生成した一覧で、全テストを状態別に並べている
**既知の不具合** もここに載っている
実装が直るまで意図的に red のまま残しているテスト (`#[ignore = "known defect: …"]`) である
本番で数値を使う前に、該当モジュールの行を確認してほしい

参照テストのないモジュール (経験式の `warp_risk`、`layer_adhesion` など) は、引用した式を実装したものであって検証済みの予測ではない
[`docs/MODULES.md`](docs/MODULES.md) に明記している

## Cargo feature

<!-- readme-sync: features -->
| Feature | 既定 | 内容 |
|---------|:----:|------|
| `std` | ○ | 標準ライブラリ 外すと `no_std` (`alloc` が必要) |
| `simd` | | `x86_64` 上で `Vec3Fix` 演算に SSE2 を使う 結果はスカラー版とビット一致 |
| `parallel` | | グラフ彩色したバッチを Rayon で並列に解く |
| `ffi` | | Unity、Unreal Engine などから使う C ABI `wasm` とは併用不可 |
| `wasm` | | `wasm-bindgen` による WebAssembly バインディング `std` が必要、`ffi` とは併用不可 |
| `python` | | Python バインディング (PyO3 + NumPy) |
| `gpu-solver-bridge` | | 外部 GPU ソルバー用の `GpuSolverBridge` トレイト 無効時のコストはない |
| `neural` | | [`alice-ml`](https://crates.io/crates/alice-ml) を使う決定論的ニューラルコントローラー |
| `replay` | | [`alice-db`](https://crates.io/crates/alice-db) を使うリプレイの記録と再生 |
| `analytics` | | [`alice-analytics`](https://crates.io/crates/alice-analytics) を使うシミュレーションのプロファイリング |

## バインディング

| 対象 | 場所 | 備考 |
|------|------|------|
| C / C++ | [`include/alice_physics.h`](include/alice_physics.h) | `--features ffi` `cdylib` と `staticlib` を生成 |
| Unity (C#) | [`bindings/AlicePhysics.cs`](bindings/AlicePhysics.cs) | C ABI への P/Invoke |
| Unreal Engine 5 | [`unreal-plugin/`](unreal-plugin/README.md) | C ABI を包むプラグイン |
| Python | `src/python.rs` | `--features python` `PhysicsWorld` と `DeterministicSimulation` クラス、NumPy の一括 API |
| WebAssembly | [`web/`](web/) | `--features wasm` `wasm-pack` でビルドする Three.js ビューア |

```sh
cargo build --release --features ffi
```

## 性能

`cargo bench --bench physics_bench` (criterion、release プロファイル) を Apple シリコン上、バージョン 1.2.0 で測定した値
絶対値は機械に依存するので、引用する前に対象環境で計測し直してほしい
<!-- perf-measured: 2026-09-15 benches/physics_bench.rs -->

| 処理 | 測定値 |
|------|--------|
| `Fix128` 乗算 | 1.1 ns |
| `Fix128` 除算 | 141 ns |
| `Fix128` 平方根 | 193–354 ns |
| `Vec3Fix` 正規化 | 618 ns |
| 10 物体 × 60 ステップ、既定設定 | 398 µs |
| 重なった球 1000 個、最初のフレーム | 65 ms |

## 最小サポート Rust バージョン

最小サポート Rust バージョン: **1.85** (`Cargo.toml` の `rust-version`) <!-- readme-sync: msrv -->

CI でこのバージョンちょうどを使い、既定 feature、`no_std`、`std,simd,parallel,ffi,gpu-solver-bridge` のライブラリビルドを確認している
MSRV の引き上げはマイナーバージョンの変更として扱い、パッチでは行わない
`neural`、`replay`、`analytics` は依存先 crate の MSRV に従う

## ビルドとテスト

```sh
cargo build --release
cargo test
cargo test --features "simd,parallel,ffi,gpu-solver-bridge"
cargo bench --bench physics_bench
```

`wasm` と `ffi` は同時に有効にできないので `--all-features` はビルドできない
`wasm` は別に実行する
`scripts/preflight.sh` で CI と同じ検査を手元で実行できる (`--quick` は結合テストと解析解テストを省く)

## 関連 crate

| Crate | 役割 |
|-------|------|
| [alice-det-math](https://github.com/ext-sakamoro/ALICE-DetMath) | 本 crate と ALICE-SDF が共通で使う決定論的な超越関数 |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | 符号付き距離関数 同じ数学で評価される SDF コライダーを提供する |
| [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) | ALICE-SDF の木にコンパイルされる言語と法則検証器 |

依存グラフ内の `alice-det-math` は 1 バージョンに揃えること
2 バージョンが混在すると同じ関数の実装が 2 つになり、決定論が崩れる
`cargo tree -i alice-det-math` で確認できる

変更履歴は [`CHANGELOG.md`](CHANGELOG.md)、今後の予定は [`docs/ROADMAP.md`](docs/ROADMAP.md) にある

## ライセンス

`AGPL-3.0-or-later OR LicenseRef-Commercial` のデュアルライセンス
どちらかを選んで使う

| 選択肢 | 向いている場合 |
|--------|---------------|
| [AGPL-3.0-or-later](LICENSE-AGPL) | プロジェクト自体が AGPL 互換のオープンソース、または社内利用のみ |
| [商用ライセンス](LICENSE-COMMERCIAL.md) | クローズドソース製品、プロプライエタリな SaaS、ファームウェア、エンジンプラグインの再配布 |

AGPL は `alice-physics` をリンクするものすべてに及ぶ
C ABI、Unity / Unreal バインディング、Python バインディング、WebAssembly ビルド経由のリンクも含む
商用ライセンスの問い合わせ: <contact@extoria.co.jp>

任意 feature の `neural` と `replay` は、追加で AGPL ライセンスの crate を取り込む
`analytics` と必須依存の `alice-det-math` は `MIT OR Apache-2.0`

Copyright (C) 2024-2026 Moroya Sakamoto

### 参考文献

- Müller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- Ericson, *Real-Time Collision Detection* (GJK / EPA)
- Volder, "The CORDIC Trigonometric Computing Technique"
