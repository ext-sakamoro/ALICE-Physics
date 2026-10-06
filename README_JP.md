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
ロールバック / ロックステップ型のネットコード、サーバー側でのリプレイ検証、シミュレーションを分岐・巻き戻しながら進める探索・計画や世界モデル、再現性が求められる工学・研究計算などである
剛体ソルバーのほかに工学系モジュール (FEM、CFD、伝熱、複合材、3D プリント検査) を含み、これらにも同じクロスプラットフォーム決定論を適用している
<!-- claim-test: golden_sim_field -->
<!-- claim-test: determinism_freefall -->

浮動小数点のゲームエンジンをそのまま置き換えるものではない
固定小数点演算には時間がかかり、数千体が相互作用するシーンを 60 fps で回す用途は設計の想定外

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [決定論](#決定論)
- [探索・計画・世界モデルのための決定論的な世界](#探索計画世界モデルのための決定論的な世界)
- [含まれるもの](#含まれるもの)
- [車両運動](#車両運動)
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
| [`world_snapshot_branching`](examples/world_snapshot_branching.rs) | 世界全体のスナップショット、そこからの分岐、SDF コライダーを持つ世界の復元 |
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

## 探索・計画・世界モデルのための決定論的な世界

1 ステップがビット単位で一意なので、`PhysicsWorld` は確率的に学習した近似ではなく、法則そのものを遷移関数として使える
探索や計画のループから何度も呼び、行動を試す → 結果を読む → 巻き戻す → 別の行動を試す、を繰り返せる
そのための API:

| API | 内容 |
|-----|------|
| `PhysicsWorld::reset_world()` | 全フィールドを `PhysicsWorld::new` 直後の状態に戻す 同じ初期状態と同じ入力なら 2 回目もビット一致する |
| `observe_body` / `observe_bodies` | 型付きの `BodyObservation` (位置、速度、回転、角速度、`sleeping`、`in_contact`) を返す ゴール判定が状態 blob を解析せずに読める |
| `serialize_state` / `deserialize_state` | 分岐点の保存と復元 物体の数が同じでも中身 (質量、形状、フィルタ、マテリアル) が保存時と違えば復元を拒否する |
| `snapshot_world` / `from_world_snapshot` / `restore_world` | 世界全体 (物体、ジョイント、拘束、コライダー、力場、マテリアル、フィルタ、イベント、スリープ状態、broad-phase の木、warm-start のキャッシュ、オーバーフローフラグ) を版番号とチェックサム付きの 1 つの blob に保存し、新しい世界または既存の世界に復元する 復元後のステップは元の世界とビット一致する 不正な blob は理由を示す `WorldSnapshotError` で拒否する |
| `step_n(n, dt)` | `step(dt)` を `n` 回実行する Python の `step_n` と WASM の `stepN` も同じ関数を呼ぶ |
| `overflow_detected()` | `Fix128` の演算が範囲を外れたことを報告し、発散した実行を正しい結果と取り違えないようにする フラグはロールバック後も残る |
| `netcode::SimulationChecksum` | 保存する状態と同じバイト列から求めるチェックサム |

[`examples/world_auditor_observation.rs`](examples/world_auditor_observation.rs) でリセットと観測の使い方を示している
契約は `tests/wm0*_*.rs` で固定している

**範囲** `serialize_state` が保存するのは剛体 (運動状態、スリープ状態、オーバーフローフラグ) だけで、ジョイント、力場、衝突フィルタ、マテリアルはロールバック型ネットコードと同様に呼び出し側が組み立て直す
`snapshot_world` は `PhysicsWorld` の全フィールドを保存する ただしデータでなくコードであるもの (SDF の場、pre-solve hook、contact modifier、GPU bridge) は復元先の世界にあるものを使い、その数が一致しなければ拒否する フィールドごとの分類表は `PhysicsWorld::snapshot_world` の doc にある
モーター、キャラクターコントローラー、布、流体、FEM、車両は `PhysicsWorld` のフィールドではないので呼び出し側が保存する
本 crate が提供するのは遷移関数で、探索アルゴリズムそのものは含まない

## 含まれるもの

公開モジュールの全一覧 (分野別、1 行説明付き) は [`docs/MODULES.md`](docs/MODULES.md) (英語) にある
API の詳細は [docs.rs](https://docs.rs/alice-physics) を参照

| 分野 | 主な内容 |
|------|---------|
| 剛体 | XPBD ソルバー (時間方向 Gauss-Seidel バックエンドも選択可)、スリープとアイランド、CCD、ロールバック用の状態シリアライズ |
| 衝突 | GJK / EPA、線形 BVH または永続的な動的 AABB 木のブロードフェーズ、箱・球・カプセル・円柱・円錐・楕円体・トーラス・くさび・凸包・複合形状・三角形メッシュ・高さ場・SDF コライダー、それらの実形状に対する world の ray クエリ (`PhysicsWorld::cast_ray`) |
| ジョイント | ボール、ヒンジ、固定、スライダー、ばね、D6、コーンツイストと、プーリー、ギア、溶接、ラックアンドピニオン、マウス 破断とPD モーター |
| 柔軟体 | XPBD のロープと布 (自己衝突あり)、位置ベース流体、FEM-XPBD 変形体、切断 |
| ゲーム用途 | キャラクターコントローラー、車両 (簡易モデルと、タイヤ・ブレーキ・ABS・路面・天候を持つ輪ごとの車両運動モデル)、ラグドール、IK 連携、クライアント側予測、決定論的乱数 (正規分布を含む)、接触イベント、模擬センサー (lidar / 接触 / IMU) |
| 固体力学 | P1 / P2 / P3 四面体の線形弾性 FEM、共回転による大回転、J2 塑性、超弾性、熱-構造連成、適応細分化、梁、座屈、疲労、複合材 |
| 流体と場 | 複数の圧力ソルバーを持つ MAC 格子 CFD、RANS / LES 乱流モデル、VOF とレベルセット、SPH、圧縮性流れ、伝熱、Maxwell FDTD |
| 空力 | 標準大気 (ISA 1976、20 km まで)、失速を含む翼の揚力と抗力、ロータの推力とトルク |
| 分子動力学 | Lennard-Jones・Morse・Coulomb・遮蔽 Coulomb の対ポテンシャル (カットオフとシフト)、周期境界のセルリスト (最小像規約) を使う速度 Verlet |
| 群衆 | 歩行者の social force model: 駆動項、視野角で重みづけた反発、接触時の体圧と滑り摩擦、壁 |
| 3D プリント | 材料データベース、薄肉・オーバーハング検査、反り、層間接着、造形向き、それらをまとめた安全性パイプライン |
| 2D | 独立した 2D XPBD エンジン (専用の形状とジョイント) |

**領域分割** CFD の圧力投影は `z` 方向のスラブに分けて 1 層のハローを交換しながら実行でき、ランク数によらず単一プロセスの解とビット一致する
測定は 1 台のホスト内のみ (ループバック TCP で最大 8 プロセス)
複数ホストにまたがる実行は行っておらず、MPI バックエンドもない

すべてのモジュールが `PhysicsWorld` に組み込まれているわけではない 下の表は、公開モジュールをその item が実際にどこから呼ばれているかで数えたもの (`examples/` からの呼び出しは数えない) CI が `scripts/integration_levels.py` で測る モジュールごとの区分は [`docs/MODULES.md`](docs/MODULES.md) の Integration 列、詳細は [`docs/integration-levels.md`](docs/integration-levels.md) (英語) にある

<!-- integration-levels: summary -->
| 使われ方 | モジュール数 |
|----------|-------------:|
| step: `PhysicsWorld` の step で実行される | 20 |
| world API: `PhysicsWorld` の他のメソッドから使われる | 7 |
| binding: C ABI・Python・WebAssembly のバインディングから使われる | 2 |
| standalone: 利用者が直接呼ぶ Rust API で、`PhysicsWorld` は呼ばない | 131 |
| unused: テスト以外に呼び出し元がない | 0 |

## 車両運動

`vehicle_dynamics::DynamicVehicle` は、各車輪がそれぞれの接地点で車体に力を加える車両モデル

- サスペンションとタイヤの力を車輪ごとに加えるので、操舵でヨーが生じ、制動や旋回で前後・左右に荷重が移る
- 各車輪は回転状態を持ち、駆動トルク、ブレーキトルク、タイヤ力で回転が変わる ブレーキで車輪をロックでき、ABS はスリップ率を目標値付近に保つ
- タイヤ力は brush モデルまたは Magic Formula モデルで求め、摩擦楕円で制限する
- 路面は平面、斜面、高さ場、三角形メッシュ、SDF のいずれか グリップは路面材料に天候係数 (乾燥・湿潤・積雪・凍結) を掛けたもので、冠水路ではハイドロプレーニングによる低下が加わる
- ロックした車輪は静止摩擦の限界より緩い斜面で車を止めたまま保持し、限界を超えると動摩擦係数で滑る
- エンジンのトルク曲線、変速機、エンジンブレーキ、オープンまたはロックのデファレンシャル、空気抵抗と揚力
- `vehicle_dynamics::scenario`: 1 つの `PhysicsWorld` で複数の `DynamicVehicle` を入力列または frame ごとの入力関数で走らせる (処理順は index 順で固定) / 追従指標: TTC (車間 ÷ 接近速度、接近していなければ `None`) と車間時間、停止距離計 / 無損失 replay: 初期状態と frame ごとの `DriverInput` を `Fix128` の raw 値で記録し全車の車体と全輪の状態を bit 一致で再現 (版番号付きバイト列) / 壊れた・切り詰めたバイト列や形の合わないシナリオは `ReplayError` で拒否

従来の `vehicle::Vehicle` は変更していない
こちらは `ground_height` の平面上だけを走り、全車輪の力の合計を重心に加えるので、車輪ごとの荷重移動、車輪のロック、タイヤモデルを持たない
停止距離や旋回を物理に従わせる必要がある場合は `vehicle_dynamics` を使う

```rust
use alice_physics::vehicle_dynamics::surface::{FlatGround, RoadCondition};
use alice_physics::vehicle_dynamics::{DynamicVehicle, DynamicVehicleConfig, Environment};

let mut car = DynamicVehicle::new(DynamicVehicleConfig::passenger_car());
let road = FlatGround { height: Fix128::ZERO };
let condition = RoadCondition::dry_asphalt();
let env = Environment { condition: &condition, wind: None, time: Fix128::ZERO };
car.input.brake = Fix128::ONE;
// 毎フレーム、world.step(dt) の前に:
car.update(&mut world.bodies[chassis], &road, &env, dt);
```

[`examples/vehicle_dynamics.rs`](examples/vehicle_dynamics.rs) はこの構成を実行し、乾燥・湿潤・凍結路で 4 輪をロックした停止距離を `v0² / (2 μ_k g)` と比較し、ABS の有無で制動を比べ、斜面で車を保持する
[`examples/vehicle_scenario.rs`](examples/vehicle_scenario.rs) は同じ車線に 2 台を走らせ、TTC で後続車を制動し、記録した走行を bit 一致で再生する
閉形式との比較テストは `tests/analytic_vehicle_dynamics.rs` と `tests/analytic_vehicle_scenario.rs` にある

**既知の制限**

- 完全滑りのとき、brush モデルは力の向きを滑り方向へ寄せるが、Magic Formula モデルは寄せない (純スリップの力を摩擦楕円上へ縮めるだけ)
- 車輪の力はフレーム先頭で 1 回の撃力として加える
  サスペンションが安定なのは `(ω_n dt)² + 2 c dt / m_share < 4` (`ω_n = √(k / m_share)`、`m_share` は 1 輪が受け持つ質量) の範囲だけで、軽い車体に硬いばねを大きな `dt` で使うと振幅が増大する
- `HeightField` 自体は `origin.y` を反映せず、格子の境界で法線に既知の不具合がある
  高さ場の路面は高さ場自身の `sample_height` に従い (路面の高さはその戻り値で、`origin.y` は加わらない)、境界では片側差分で法線を求めるので、車輪の探査は境界の不具合の影響を受けない

## 検証と既知の不具合

`tests/` の解析解テストと参照テストは、閉形式解や公表されている参照データ (教科書の公式、キャビティ流れの Ghia らの表など) と結果を比較している
golden ハッシュは変化を検出するだけなので、これらとは別に扱っている

[`docs/oracle-status.md`](docs/oracle-status.md) は `tests/` から自動生成した一覧で、全テストを状態別に並べている
**既知の不具合** もここに載っている
実装が直るまで意図的に red のまま残しているテスト (`#[ignore = "known defect: …"]`) である
本番で数値を使う前に、該当モジュールの行を確認してほしい

[`docs/integration-status.md`](docs/integration-status.md) は rust-analyzer が解決した参照から自動生成した一覧で、公開 item ごとにテスト以外から到達されているか、example からだけ到達されているかを示す

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
| Unreal Engine 5 | [`unreal-plugin/`](unreal-plugin/README.md) | C ABI の一部を包む Blueprint コンポーネント |
| Python | `src/python.rs` | `--features python` `PhysicsWorld` (body・衝突半径と形状・静的コライダー・ジョイント) と `DeterministicSimulation` クラス、NumPy の一括 API |
| WebAssembly | [`web/`](web/) | `--features wasm` `WasmPhysicsWorld` (body・衝突半径と形状・静的コライダー・ジョイント) と `wasm-pack` でビルドする Three.js ビューア |

バインディングが覆うのはシーンを組んで進めるのに要る範囲 (body・衝突半径と形状・静的コライダー・ジョイント・撃力・状態の直列化) で、全モジュールではない 上の表の standalone のモジュールは Rust からのみ使える C ABI と各利用側の対応は CI が `scripts/integration_levels.py` で検査する C ヘッダ・`bindings/AlicePhysics.h`・Unity のバインディングは公開関数をすべて宣言し、Unreal Engine のコンポーネントはその一部を包む 包まない関数とその理由は [`docs/integration-levels.md`](docs/integration-levels.md) (英語) にある

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

### 休止中の物体

`step` は、休止中で静止しており joint や distance 拘束に繋がっていない物体を、全ての段階から外す (`PhysicsWorld::set_sleep_skip`、既定で有効、無効にしても結果は bit 一致)
起きている物体との接触は休止中の物体を収めた永続的な木から探すので、broad-phase は起きている物体だけで組む
`PhysicsWorld::stage_work` が直前の step の段階ごとの作業量を返す

`cargo bench --bench world_scale` (criterion、release プロファイル、8 substep の `step` 1 回) を arm64 / 10 コア / 32 GiB で、他のプロセスが動いている状態で測定した値 (±15 % 程度の揺れを見込んでほしい)
半径 0.5 の球を 2 m 間隔の格子に置き、起きている球は休止中の球の上方を漂って接触しない
<!-- perf-measured: 2026-10-04 benches/world_scale.rs -->

| 物体数 | 休止中 | skip 有効 | skip 無効 |
|-------:|-------:|----------:|----------:|
| 10 000 | 0 % | 113 ms | 122 ms |
| 10 000 | 90 % | 9.7 ms | 96 ms |
| 10 000 | 99 % | 0.88 ms | 121 ms |
| 100 000 | 0 % | 1.31 s | 1.24 s |
| 100 000 | 90 % | 105 ms | 1.31 s |
| 100 000 | 99 % | 10.8 ms | 1.15 s |

休止中の物体 1 個あたり、step ごとに約 30 ns が残る (step 間に書き換えられていないかの検査 (`bodies` は公開 field) と、状態 snapshot に含まれる `idle_frames` の加算)

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
cargo bench --bench world_scale
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
