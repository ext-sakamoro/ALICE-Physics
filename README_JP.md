# ALICE-Physics

**決定論的128bit固定小数点物理エンジン** - v0.3.0

[English](README.md) | 日本語

異なるプラットフォームやハードウェア間で決定論的なシミュレーションを実現する高精度物理エンジン。128bit固定小数点演算を使用し、CPU、コンパイラ、OSに関わらずビット精度の結果を保証します。

## 機能一覧

| 機能 | 説明 |
|------|------|
| **128bit固定小数点** | I64F64形式（64bit整数部 + 64bit小数部）による超高精度 |
| **CORDIC三角関数** | FPU命令を使わない決定論的 sin/cos/atan |
| **XPBDソルバー** | Extended Position Based Dynamics による安定した拘束解決 |
| **GJK/EPA衝突判定** | 凸形状に対するロバストな衝突検出 |
| **スタックレスBVH** | モートンコードベースの空間加速（エスケープポインタ付き） |
| **拘束バッチング** | グラフ彩色による並列拘束解決 |
| **ロールバック対応** | ネットコード用の完全なステートシリアライズ |
| **ニューラルコントローラ** | ALICE-ML三値重み + Fix128推論による決定論的AI |
| **5種のジョイント** | Ball, Hinge, Fixed, Slider, Spring（角度制限・モーター付き） |
| **レイキャスティング** | Sphere, AABB, Capsule, Planeに対するレイ・シェイプキャスト |
| **CCD** | 連続衝突検出（TOI、保守的前進法） |
| **スリープ/アイランド** | Union-Findアイランド管理による自動スリープ |
| **三角メッシュ** | BVH加速三角メッシュ衝突（Moller-Trumboreアルゴリズム） |
| **ハイトフィールド** | バイリニア補間によるグリッド地形 |
| **多関節体** | 多関節チェーン、ラグドール、ロボットアーム（FK伝播） |
| **フォースフィールド** | 風、重力井戸、ドラッグ、浮力、ボルテックス |
| **PDコントローラ** | 1D/3D 比例-微分関節モーター |
| **衝突フィルタリング** | レイヤー/マスクビットマスクによる衝突グループ |
| **決定論的RNG** | PCG-XSH-RR 疑似乱数生成器 |
| **接触イベント** | Begin/Persist/End 接触・トリガーイベント追跡 |
| **no_std対応** | 組み込みシステム・WebAssemblyで動作 |

## 最適化（"黒焦げ" エディション）

ALICE-Physicsは以下のパフォーマンス最適化を含みます：

### 1. スタックレスBVHトラバーサル

従来のBVHトラバーサルはスタックを使用しますが、本実装では各ノードに**エスケープポインタ**を埋め込みます：

```
┌──────────────────────────────────────────────────────┐
│  BvhNode レイアウト（32バイト、キャッシュライン整列）      │
├──────────────────────────────────────────────────────┤
│  aabb_min[3]        (12 bytes) - バウンディングボックス最小 │
│  first_child/prim   (4 bytes)  - 子ノード or プリミティブ   │
│  aabb_max[3]        (12 bytes) - バウンディングボックス最大 │
│  prim_count_escape  (4 bytes)  - [count:8|escape:24]      │
└──────────────────────────────────────────────────────┘

トラバーサル: 単一インデックス変数、スタック割り当てなし
  if (ヒット) → first_child へ降下
  if (ミス)   → escape_idx へジャンプ（サブツリー全体をスキップ）
```

**利点:**
- クエリ中のヒープ割り当てゼロ
- トラバーサル状態は単一レジスタ
- 分岐予測の改善
- i32 AABB比較（Fix128復元不要）

### 2. SIMD高速化（オプション）

`--features simd` で有効化：

```rust
// x86_64 with SSE2
impl Fix128 {
    pub unsafe fn add_simd(self, rhs: Self) -> Self;
    pub unsafe fn sub_simd(self, rhs: Self) -> Self;
}

impl Vec3Fix {
    pub fn dot_simd(self, rhs: Self) -> Fix128;
    pub fn cross_simd(self, rhs: Self) -> Self;
    pub fn dot_batch_4(a: [Self; 4], b: [Self; 4]) -> [Fix128; 4];
}
```

### 3. 拘束バッチング（オプション）

`--features parallel` で有効化：

**グラフ彩色**で拘束をグループ化 — ボディを共有しない拘束は同じ「色」に配置され、独立に解決可能：

```rust
// 拘束バッチを再構築（貪欲グラフ彩色）
world.rebuild_batches();

// バッチ拘束解決でステップ
world.step_parallel(dt);

// カラーバッチ数を確認
println!("Batches: {}", world.num_batches());
```

**利点:**
- 拘束ループ内のヒープ割り当てゼロ（インデックスベース反復）
- Rayon対応の並列解決
- ロック競合の低減
- キャッシュ効率の向上

## なぜ決定論的物理なのか？

IEEE 754浮動小数点を使用する従来の物理エンジンは、以下の条件で異なる結果を生成する可能性があります：
- 異なるCPUアーキテクチャ（x86 vs ARM）
- 異なるコンパイラ（GCC vs Clang vs MSVC）
- 異なる最適化レベル（-O0 vs -O3）
- 異なる命令セット（SSE vs AVX）

ALICE-Physicsは**どこでもビット精度の結果**を保証し、以下を実現します：

- **ロックステップマルチプレイ**: 全クライアントが同一のシミュレーションを計算
- **ロールバックネットコード**: 入力を決定論的に再生
- **リプレイシステム**: ゲームセッションの完全な再現
- **分散シミュレーション**: 一貫した結果による並列計算

## アーキテクチャ

```
┌──────────────────────────────────────────────────────────────────────┐
│                       ALICE-Physics v0.3.0                           │
├──────────────────────────────────────────────────────────────────────┤
│  コアレイヤー                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  math    │ │ collider │ │  solver  │ │   bvh    │ │sdf_colldr│  │
│  │ Fix128   │ │ AABB     │ │ RigidBody│ │ Morton   │ │ SdfField │  │
│  │ Vec3Fix  │ │ Sphere   │ │ XPBD     │ │Stackless │ │ Gradient │  │
│  │ QuatFix  │ │ Capsule  │ │ Batching │ │ Zero-    │ │ Early-out│  │
│  │ CORDIC   │ │ GJK/EPA  │ │ Rollback │ │  alloc   │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  拘束・ダイナミクスレイヤー                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  joint   │ │  motor   │ │articulatn│ │  force   │ │ sleeping │  │
│  │ Ball     │ │ PD 1D/3D │ │ Ragdoll  │ │ Wind     │ │ Islands  │  │
│  │ Hinge    │ │ Position │ │ FK Chain │ │ Gravity  │ │ Union-   │  │
│  │ Fixed    │ │ Velocity │ │ Robotic  │ │ Buoyancy │ │  Find    │  │
│  │ Slider   │ │ Max Torq │ │ 12-body  │ │ Drag     │ │ Auto     │  │
│  │ Spring   │ │          │ │          │ │ Vortex   │ │  Sleep   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  クエリ・衝突レイヤー                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ raycast  │ │   ccd    │ │ trimesh  │ │heightfld │ │  filter  │  │
│  │ Sphere   │ │ TOI      │ │ Triangle │ │ Bilinear │ │ Layer    │  │
│  │ AABB     │ │ Conserv. │ │ BVH加速   │ │ Normal   │ │ Mask     │  │
│  │ Capsule  │ │ Advance  │ │ Moller-  │ │ Sphere   │ │ Group    │  │
│  │ Plane    │ │ Swept    │ │ Trumbore │ │ Collide  │ │ Bidirect │  │
│  │ Sweep    │ │ AABB     │ │ Closest  │ │ Signed   │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  ユーティリティレイヤー                                                  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────────────────────┐  │
│  │   rng    │ │  event   │ │    neural (ALICE-ML × Physics)      │  │
│  │ PCG-XSH  │ │ Begin    │ │ 三値 {-1,0,+1} → Fix128 加算/減算    │  │
│  │ Fix128   │ │ Persist  │ │ 決定論的AI                           │  │
│  │ Direction│ │ End      │ │                                     │  │
│  └──────────┘ └──────────┘ └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## 使い方

### 基本シミュレーション

```rust
use alice_physics::prelude::*;

fn main() {
    // デフォルト設定で物理ワールドを作成
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // 動的ボディを追加（落下する球体）
    let sphere = RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 100, 0),  // 位置: (0, 100, 0)
        Fix128::ONE,                    // 質量: 1.0
    );
    let sphere_id = world.add_body(sphere);

    // 静的な地面を追加
    let ground = RigidBody::new_static(Vec3Fix::ZERO);
    world.add_body(ground);

    // 60 FPSでシミュレーション
    let dt = Fix128::from_ratio(1, 60);  // 1/60秒

    for frame in 0..300 {  // 5秒間
        world.step(dt);

        let pos = world.bodies[sphere_id].position;
        println!("Frame {}: y = {}", frame, pos.y.hi);
    }
}
```

### 距離拘束（ロープ/チェーン）

```rust
use alice_physics::prelude::*;

fn create_rope(world: &mut PhysicsWorld, segments: usize) {
    let mut prev_id = None;

    for i in 0..segments {
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_int(i as i64 * 2, 50, 0),
            Fix128::ONE,
        );
        let id = world.add_body(body);

        if let Some(prev) = prev_id {
            let constraint = DistanceConstraint {
                body_a: prev,
                body_b: id,
                local_anchor_a: Vec3Fix::ZERO,
                local_anchor_b: Vec3Fix::ZERO,
                target_distance: Fix128::from_int(2),
                compliance: Fix128::from_ratio(1, 1000),  // ソフト拘束
            };
            world.add_distance_constraint(constraint);
        }

        prev_id = Some(id);
    }

    // 最初のセグメントを固定
    world.bodies[0].inv_mass = Fix128::ZERO;
}
```

### ロールバックネットコード

```rust
use alice_physics::prelude::*;

struct GameState {
    physics: PhysicsWorld,
    frame: u64,
    input_buffer: Vec<PlayerInput>,
}

impl GameState {
    fn save_snapshot(&self) -> Vec<u8> {
        self.physics.serialize_state()
    }

    fn load_snapshot(&mut self, data: &[u8]) {
        self.physics.deserialize_state(data);
    }

    fn rollback_and_resimulate(&mut self, to_frame: u64, new_input: PlayerInput) {
        // フレームのスナップショットをロード
        let snapshot = self.get_snapshot(to_frame);
        self.load_snapshot(&snapshot);

        // 修正された入力でリプレイ
        self.input_buffer[to_frame as usize] = new_input;

        for frame in to_frame..self.frame {
            let input = &self.input_buffer[frame as usize];
            self.apply_input(input);
            self.physics.step(Fix128::from_ratio(1, 60));
        }
    }
}
```

### BVHブロードフェーズ衝突

```rust
use alice_physics::bvh::{LinearBvh, BvhPrimitive};

// プリミティブからBVHを構築
let primitives: Vec<BvhPrimitive> = bodies.iter().enumerate()
    .map(|(i, body)| BvhPrimitive {
        aabb: body.compute_aabb(),
        index: i as u32,
        morton: 0,  // ビルド時に計算
    })
    .collect();

let bvh = LinearBvh::build(primitives);

// ヒープ割り当てゼロのクエリ（コールバック版）
bvh.query_callback(&query_aabb, |prim_idx| {
    // プリミティブとの衝突候補を処理
});

// または結果を収集
let hits = bvh.query(&query_aabb);

// BVH統計情報を取得
let stats = bvh.stats();
println!("ノード数: {}, リーフ数: {}", stats.node_count, stats.leaf_count);
```

## モジュール

### `math` - 固定小数点プリミティブ

| 型 | 説明 |
|----|------|
| `Fix128` | 128bit固定小数点数（I64F64） |
| `Vec3Fix` | Fix128成分の3Dベクトル |
| `QuatFix` | 回転用クォータニオン |
| `Mat3Fix` | 慣性テンソル用3x3行列 |

**定数:**
- `Fix128::ZERO`, `Fix128::ONE`, `Fix128::NEG_ONE`
- `Fix128::PI`, `Fix128::HALF_PI`, `Fix128::TWO_PI`

**CORDIC関数（決定論的、FPU不使用）:**
- `Fix128::sin()`, `Fix128::cos()`, `Fix128::sin_cos()`
- `Fix128::atan()`, `Fix128::atan2()`
- `Fix128::sqrt()`（Newton-Raphson法、64回反復）

### `collider` - 衝突検出

| 形状 | 説明 |
|------|------|
| `AABB` | 軸平行バウンディングボックス |
| `Sphere` | 球体コライダー |
| `Capsule` | カプセル（円柱 + 半球） |
| `ConvexHull` | 任意の凸多面体 |

**アルゴリズム:**
- **GJK**: 交差判定（最大64回反復）
- **EPA**: 貫通深度算出（最大64回反復）

### `solver` - XPBD物理

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `position` | `Vec3Fix` | 重心位置 |
| `rotation` | `QuatFix` | 姿勢クォータニオン |
| `velocity` | `Vec3Fix` | 線形速度 |
| `angular_velocity` | `Vec3Fix` | 角速度 |
| `inv_mass` | `Fix128` | 逆質量（0 = 静的） |
| `restitution` | `Fix128` | 反発係数（0-1） |
| `friction` | `Fix128` | 摩擦係数 |

### `joint` - 5種のジョイント

| 型 | 説明 |
|----|------|
| `BallJoint` | 球面ジョイント（回転3自由度） |
| `HingeJoint` | ヒンジジョイント（回転1自由度、角度制限付き） |
| `FixedJoint` | 固定ジョイント（0自由度） |
| `SliderJoint` | スライダージョイント（並進1自由度、制限付き） |
| `SpringJoint` | 減衰スプリング拘束 |

### その他のモジュール

| モジュール | 説明 |
|-----------|------|
| `bvh` | モートンコード、エスケープポインタ、i32 AABBによるスタックレストラバーサル |
| `raycast` | Sphere/AABB/Capsule/Planeに対するレイキャスト |
| `ccd` | 連続衝突検出（TOI、保守的前進法、Swept AABB） |
| `trimesh` | BVH加速三角メッシュ衝突（Moller-Trumbore） |
| `heightfield` | バイリニア補間地形、球体衝突、符号付き距離 |
| `filter` | レイヤー/マスクビットマスク衝突フィルタ |
| `force` | 風、重力井戸、ドラッグ、浮力、ボルテックス |
| `motor` | 1D/3D PDコントローラ、ジョイントモーター |
| `articulation` | 多関節チェーン、FK伝播、12体ラグドール |
| `rng` | PCG-XSH-RR 決定論的乱数 |
| `event` | Begin/Persist/End 接触イベント追跡 |
| `sleeping` | Union-Findアイランド、自動スリープ |

## SDF コライダー（ALICE-SDF連携）

ALICE-Physicsは[ALICE-SDF](../ALICE-SDF)の距離場を衝突形状として使用できます。凸包（GJK/EPA）で近似する代わりに、SDFを直接サンプリングし、O(1)コストで数学的に正確な曲面を得ます。

```rust
use alice_physics::prelude::*;
use alice_physics::sdf_collider::SdfCollider;
use alice_sdf::physics_bridge::CompiledSdfField;
use alice_sdf::prelude::*;

// SDF形状をALICE-SDFで作成
let terrain = SdfNode::plane(0.0, 1.0, 0.0, 0.0)
    .union(SdfNode::sphere(2.0).translate(0.0, -1.5, 0.0));

let field = CompiledSdfField::new(terrain);

// SDFを静的コライダーとして登録
let collider = SdfCollider::new_static(
    Box::new(field),
    Vec3Fix::ZERO,
    QuatFix::IDENTITY,
);
world.add_sdf_collider(collider);
```

## 決定論的ニューラルコントローラ（ALICE-ML連携）

[ALICE-ML](../ALICE-ML)と連携し、1.58bit三値重み {-1, 0, +1} と128bit固定小数点演算を組み合わせた**ビット精度決定論的AI**を提供します。ニューラル推論が純粋な加算/減算に集約され、全クライアントで同一のAI動作を保証します。

```rust
// 1. 学習済み重みを三値に量子化
let (w1, _) = quantize_to_ternary(&trained_weights_l1, hidden_size, input_size);

// 2. 固定小数点に変換（一度だけ）
let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);

// 3. 決定論的ネットワークを構築
let network = DeterministicNetwork::new(
    vec![ftw1, ftw2],
    vec![Activation::ReLU, Activation::HardTanh],
);

// 4. 物理ループで使用 — 全クライアントで決定論的
for frame in 0..3600 {
    let output = controller.compute(&world.bodies);
    world.step(dt);
}
```

## 設定

```rust
let config = PhysicsConfig {
    substeps: 8,       // フレームあたりのXPBDサブステップ数
    iterations: 4,     // サブステップあたりの拘束反復回数
    gravity: Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(-10),  // -10 m/s²
        Fix128::ZERO,
    ),
    damping: Fix128::from_ratio(99, 100),  // 0.99 速度保持率
};
```

## 性能特性

| 演算 | 計算量 | 備考 |
|------|--------|------|
| Fix128 加算/減算 | O(1) | ~2-3サイクル |
| Fix128 乗算 | O(1) | ~10サイクル（128bit乗算） |
| Fix128 除算 | O(1) | ~40サイクル（128bit除算） |
| CORDIC sin/cos | O(48) | 48回反復、決定論的 |
| GJK 交差判定 | O(64) | 最大64回反復 |
| EPA 貫通深度 | O(64) | 最大64回反復 |
| BVH 構築 | O(n log n) | モートンコードソート |
| BVH クエリ | O(log n) | スタックレストラバーサル |

## ビルド

```bash
# 標準ビルド
cargo build --release

# no_stdビルド（組み込み/WASM向け）
cargo build --release --no-default-features

# テスト実行
cargo test

# 全フィーチャー組み合わせ
cargo test --features simd
cargo test --features parallel
cargo test --features neural
cargo test --features "simd,parallel"
```

## Cargo Features

| Feature | デフォルト | 説明 |
|---------|----------|------|
| `std` | Yes | 標準ライブラリサポート |
| `simd` | No | SIMD高速化 Fix128/Vec3Fix 演算（x86_64） |
| `parallel` | No | Rayonによる拘束バッチング（グラフ彩色並列解決） |
| `neural` | No | ALICE-ML三値推論による決定論的ニューラルコントローラ |
| `python` | No | Pythonバインディング（PyO3 + NumPyゼロコピー） |
| `replay` | No | ALICE-DB経由のリプレイ録画/再生 |
| `ffi` | No | C FFI（Unity、UE5等のゲームエンジン向け） |

```bash
# SIMD最適化
cargo build --release --features simd

# 並列拘束解決
cargo build --release --features parallel

# ゲームエンジン向け共有ライブラリのビルド
cargo build --release --features ffi
```

## ゲームエンジン統合（C FFI / Unity / UE5）

ALICE-Physicsは、Unity、Unreal Engine、およびC関数を呼べる全ての言語向けにC FFIレイヤーを提供します。

### 共有ライブラリのビルド

```bash
cargo build --release --features ffi
# 出力: target/release/ 内に .dylib / .so / .dll
```

### C API

Cヘッダーは `include/alice_physics.h` にあります。FFI境界では全て `f64` を使用し、内部で `Fix128` に変換します。

```c
#include "alice_physics.h"

AlicePhysicsWorld* world = alice_physics_world_create();
AliceVec3 pos = {0.0, 10.0, 0.0};
uint32_t body = alice_physics_body_add_dynamic(world, pos, 1.0);
alice_physics_world_step(world, 1.0 / 60.0);

// ステートシリアライズ（ロールバックネットコード用）
uint32_t len;
uint8_t* state = alice_physics_state_serialize(world, &len);
alice_physics_state_deserialize(world, state, len);
alice_physics_state_free(state, len);

alice_physics_world_destroy(world);
```

### Unity C# バインディング

`bindings/AlicePhysics.cs` とネイティブライブラリをUnityプロジェクトにコピー：

```csharp
using AlicePhysics;

var world = new AlicePhysicsWorld();
uint body = world.AddDynamicBody(new Vector3(0, 10, 0), 1.0);
world.Step(1.0 / 60.0);
Vector3 pos = world.GetBodyPosition(body);

// ロールバックネットコード
byte[] state = world.SerializeState();
world.DeserializeState(state);

world.Dispose();
```

### Unreal Engine 5 プラグイン

`unreal-plugin/` をUE5プロジェクトの `Plugins/AlicePhysics/` にコピーし、ネイティブライブラリを `ThirdParty/AlicePhysics/lib/<Platform>/` に配置します。

Blueprint対応の `UAlicePhysicsWorldComponent` を提供:
- ボディ作成・状態取得・力の適用
- ロールバックネットコード用のステートシリアライズ
- 座標系の自動変換（UE5 Z-up cm → ALICE Y-up m）

### リリースワークフロー

タグをプッシュすると自動的にクロスプラットフォームビルドが実行されます：

```bash
git tag v0.3.0 && git push origin v0.3.0
```

GitHub Actionsが macOS (ARM + Intel)、Windows、Linux 向けにビルドし、UE5プラグインZIPとUnityパッケージZIPをリリースに添付します。

## 浮動小数点エンジンとの比較

| 観点 | ALICE-Physics | 浮動小数点エンジン |
|------|---------------|-----------------|
| 決定論性 | 保証 | プラットフォーム依存 |
| 精度 | 64bit小数部 | 23bit (f32) / 52bit (f64) |
| 速度 | やや遅い（2-5x） | 速い |
| ロールバック | 容易 | 注意が必要 |
| 組み込み | no_std | FPU必要 |
| 数値範囲 | ±9.2×10^18 | ±3.4×10^38 (f32) |

## テスト結果

```
v0.3.0 テストサマリ:
  - 42モジュール全体で230ユニットテスト
  - 8ドキュメントテスト
  - 全フィーチャー組み合わせパス
```

## ライセンス

AGPL-3.0 - 詳細は [LICENSE](LICENSE) を参照。

Copyright (C) 2024-2026 Moroya Sakamoto

## 謝辞

- XPBD: Müller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- GJK/EPA: Ericson, "Real-Time Collision Detection"
- Morton Codes: Morton, "A Computer Oriented Geodetic Data Base"
- CORDIC: Volder, "The CORDIC Trigonometric Computing Technique"
