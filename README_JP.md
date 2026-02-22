# ALICE-Physics

**決定論的128bit固定小数点物理エンジン** - v0.4.0

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
| **7種のジョイント** | Ball, Hinge, Fixed, Slider, Spring, D6, Cone-Twist（角度制限・モーター・破壊可能拘束付き） |
| **レイキャスティング** | Sphere, AABB, Capsule, Planeに対するレイ・シェイプキャスト |
| **シェイプキャスト/オーバーラップ** | 球体キャスト、カプセルキャスト、オーバーラップ球体/AABB クエリ |
| **CCD** | 連続衝突検出（TOI、保守的前進法） |
| **スリープ/アイランド** | Union-Findアイランド管理による自動スリープ |
| **三角メッシュ** | BVH加速三角メッシュ衝突（Moller-Trumboreアルゴリズム） |
| **ハイトフィールド** | バイリニア補間によるグリッド地形 |
| **多関節体** | 多関節チェーン、ラグドール、ロボットアーム（FK伝播） |
| **フォースフィールド** | 風、重力井戸、ドラッグ、浮力、ボルテックス |
| **PDコントローラ** | 1D/3D 比例-微分関節モーター |
| **衝突フィルタリング** | レイヤー/マスクビットマスクによる衝突グループ |
| **トリガー/センサー** | 物理応答なしでオーバーラップを検出するセンサーボディ |
| **キャラクターコントローラ** | キネマティックカプセルベースのmove-and-slide（階段昇降・SDF地形対応） |
| **ロープ** | XPBD距離チェーンによるロープ・ケーブルシミュレーション |
| **クロス** | XPBDメッシュクロス（自己衝突対応、空間ハッシュグリッド） |
| **流体** | Position-Based Fluids (PBF)（空間ハッシュグリッド） |
| **変形体** | FEM-XPBD変形ボディ（四面体メッシュ） |
| **車両** | ホイール、サスペンション、エンジン、ステアリング、ギアシフト |
| **アニメーションブレンド** | ラグドール⇔アニメーションのSLERPブレンド |
| **オーディオ物理** | 物理ベースのオーディオパラメータ生成（衝突、摩擦、転がり） |
| **SDFマニフォールド** | SDF曲面からのマルチポイント接触マニフォールド |
| **SDF CCD** | SDF向け球体トレーシング連続衝突検出 |
| **SDFフォースフィールド** | SDF駆動フォースフィールド（引力、斥力、封じ込め、フロー） |
| **SDF破壊** | リアルタイムCSGブーリアン破壊 |
| **SDFアダプティブ** | 距離ベースLODによる適応的SDF評価 |
| **凸包分解** | SDFボクセルグリッドからの凸包分解 |
| **GPU SDF** | GPUコンピュートシェーダーによるバッチSDF評価 |
| **流体ネットコード** | デルタ圧縮付き決定論的流体ネットコード |
| **シミュレーションフィールド** | トリリニア補間・拡散付き3Dスカラー/ベクトルフィールド |
| **熱伝導** | 熱拡散、融解、熱膨張、凍結 |
| **圧力** | 接触力蓄積、圧壊、膨張、凹み変形 |
| **浸食** | 風食、水食、化学腐食、アブレーション |
| **破砕** | 応力駆動亀裂伝播（CSG減算） |
| **相変化** | 温度駆動の固体/液体/気体遷移 |
| **決定論的RNG** | PCG-XSH-RR 疑似乱数生成器 |
| **接触イベント** | Begin/Persist/End 接触・トリガーイベント追跡 |
| **ボックス/OBBコライダー** | GJK対応の方向付きバウンディングボックス |
| **複合形状** | ローカル変換付きマルチシェイプ複合コライダー |
| **接触キャッシュ** | HashMap O(1) ルックアップ付き永続マニフォールドキャッシュ |
| **動的AABBツリー** | O(log n) 挿入/削除/更新付きインクリメンタルBVH |
| **D6ジョイント** | 軸ごとにロック/フリー/リミット設定可能な6自由度ジョイント |
| **コーンツイストジョイント** | コーンスイング制限+ツイスト制限付きボールジョイント |
| **マテリアルテーブル** | ペアごとの摩擦/反発係数（合成ルール: 平均, 最小, 最大, 乗算） |
| **スケール形状** | Support実装形状への均一スケールラッパー |
| **推測的CCD** | 時間巻き戻し不要の高速移動体向け推測的接触 |
| **Featherstone** | 多関節体のO(n)順動力学 |
| **デバッグレンダー** | ワイヤーフレーム可視化API（ボディ、接触、ジョイント、BVH、力） |
| **プロファイリング** | ステージ別タイマーとフレーム統計API |
| **サブステップ補間** | NLERP四元数ブレンド付きWorldSnapshotでスムーズレンダリング |
| **確率的スケッチ** | HyperLogLog、DDSketch、Count-Min Sketch、Heavy Hitters |
| **ストリーミング異常検出** | MAD、EWMA、Zスコア複合検出器 |
| **ローカル差分プライバシー** | ラプラスノイズ、RAPPOR、ランダム化応答 |
| **メトリックパイプライン** | ロックフリーリングバッファによるメトリック集約 |
| **no_std対応** | 組み込みシステム・WebAssemblyで動作 |

## 最適化（"黒焦げ" エディション） — 100/100

ALICE-Physicsは6層にわたる最適化で **100/100 の完璧なスコア** を達成しています：

### 最適化スコアカード

| レイヤー | スコア | 主要手法 |
|---------|--------|---------|
| **L1: メモリレイアウト** | 15/15 | RigidBody/Cloth/Fluid `#[repr(C, align(64))]`、hot/coldフィールド分離 |
| **L2: 実行モデル** | 20/20 | GJK/BVH/拘束に `#[inline(always)]`、ブランチレスselect、`grid_half`事前計算 |
| **L3: 計算戦略** | 20/20 | ウォームスタート `cached_lambda`、逆数事前計算（`inv_rest_length`、`inv_rest_density`） |
| **L4: GPU・スループット** | 15/15 | `SIMD_WIDTH`定数 + `simd_width()`、`GpuSdfInstancedBatch`/`GpuSdfMultiDispatch`、`batch_size()` |
| **L5: ビルドプロファイル** | 10/10 | `opt-level=3`、`lto="fat"`、`codegen-units=1`、`panic="abort"`、`strip=true` |
| **L6: コード品質** | 20/20 | 379テスト（358ユニット + 10統合 + 11ドキュメント）、clippy 0警告 |
| **合計** | **100/100** | |

### L1: メモリレイアウト (15/15)

全ホットデータ構造体が64バイトキャッシュライン整列 + hot/coldフィールド分離:

```rust
#[repr(C, align(64))]
pub struct RigidBody {
    // HOTフィールド（毎サブステップアクセス）— 最初のキャッシュライン
    pub position: Vec3Fix,
    pub velocity: Vec3Fix,
    pub inv_mass: Fix128,
    pub inv_inertia: Vec3Fix,
    pub prev_position: Vec3Fix,
    // COLDフィールド（低頻度アクセス）
    pub rotation: QuatFix,
    pub angular_velocity: Vec3Fix,
    pub restitution: Fix128,
    pub friction: Fix128,
    // ...
}
```

- **RigidBody** (`solver.rs`): `#[repr(C, align(64))]` + hot/coldフィールド再配置
- **Cloth** (`cloth.rs`): `#[repr(C, align(64))]` + エッジ拘束ごとに `inv_rest_length` 事前計算
- **Fluid** (`fluid.rs`): `#[repr(C, align(64))]` + `inv_rest_density` と `grid_half` 事前計算

### L2: 実行モデル (20/20)

全ホットパスでの積極的インライン化とブランチレス実行:

- **GJK/BVH**: 全サポート関数とトラバーサルに `#[inline(always)]`（計35箇所）
- **拘束解決**: 距離/接触拘束カーネルに `#[inline(always)]` 昇格
- **流体空間グリッド**: `grid_half` を構築時に事前計算、`hash()` ホットパスの除算を排除
- **ブランチレスプリミティブ** (`math.rs`): `select_fix128()`、`select_vec3()` — ビットマスクによるCMOV相当、パイプラインフラッシュゼロ

### L3: 計算戦略 (20/20)

除算排除とウォームスタートによる高速収束:

- **距離拘束ウォームスタート**: `cached_lambda` フィールドが前サブステップのラグランジュ乗数を保存、初期推定値をバイアスして劇的な収束高速化
- **布の逆数事前計算**: `inv_rest_length` を構築時に1回計算、拘束解決は除算の代わりに乗算を使用
- **流体の逆数事前計算**: `inv_rest_density` が非圧縮性拘束でパーティクルあたり2回以上の除算を排除
- **接触マニフォールドウォームスタート**: `lambda_n`、`lambda_t1`、`lambda_t2` をフレーム間で `apply_warm_start()` 経由で蓄積
- **接触モディファイア並列前処理パス**: `pre_process_contacts()` がRayon並列ディスパッチ前に逐次実行され、プリソルブフック・接触モディファイアをデータ競合なしに安全に適用

### L4: GPU・スループット (15/15)

SIMD幅対応バッチングによる最適なGPU/CPU活用:

```rust
// コンパイル時CPU機能検出
pub const SIMD_WIDTH: usize = simd_width();
// AVX2=8, SSE=4, NEON=4, スカラー=1

#[inline(always)]
pub fn batch_size() -> usize { crate::math::SIMD_WIDTH }
```

- **`SIMD_WIDTH`定数** (`math.rs`): コンパイル時CPU機能検出（AVX2=8, SSE=4, NEON=4, スカラー=1）
- **`GpuSdfInstancedBatch`** (`gpu_sdf.rs`): SDF IDごとのクエリグルーピングでGPUカーネル起動を最小化
- **`GpuSdfMultiDispatch`** (`gpu_sdf.rs`): 複数GPUバッチ管理、`total_queries()` と `total_dispatches()` 統計
- **`batch_size()`** (`gpu_sdf.rs`): レジスタ幅整列データストリーム用に `SIMD_WIDTH` を返却

### L5: ビルドプロファイル (10/10)

```toml
[profile.release]
opt-level = 3          # 最大最適化
lto = "fat"            # リンク時最適化（全クレート統合）
codegen-units = 1      # 単一コード生成ユニット（最適化の機会最大化）
panic = "abort"        # パニック時即abort（unwindオーバーヘッド排除）
strip = true           # シンボル除去
```

### L6: コード品質 (20/20)

- **358ユニットテスト**（67モジュール）
- **10統合テスト**（エンドツーエンド物理シナリオ）
- **11ドキュメントテスト**（実行可能な例: sketch, anomaly, privacy, pipeline）
- **合計: 379テストパス**、clippy: 0警告

---

### その他の最適化

#### スタックレスBVHトラバーサル

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

#### SIMD高速化（オプション）

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

#### 拘束バッチング（オプション）

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

#### HashMap接触キャッシュ

`BodyPairKey → usize` HashMapによるO(1)接触マニフォールドルックアップ:

```rust
// O(1) マニフォールドルックアップ（従来のO(n)線形走査を置き換え）
pub fn find(&self, a: usize, b: usize) -> Option<&ContactManifold> {
    let key = BodyPairKey::new(a, b);
    self.pair_index.get(&key).map(|&i| &self.manifolds[i])
}
```

`no_std` 環境ではリニアスキャンにフォールバックします。

#### Rayon並列インテグレーション

`--features parallel` で有効化：

位置積分と速度更新が `par_iter_mut()` で並列実行されます：

```rust
// 並列位置積分（重力 + 減衰 + オイラー）
bodies.par_iter_mut().for_each(|body| {
    body.velocity = body.velocity + gravity * dt;
    body.velocity = body.velocity * damping;
    body.position = body.position + body.velocity * dt;
});
```

#### Pythonバッチ API

GILリリース付きゼロコピーNumPyバッチ操作:

```python
# (N,4) 配列 [x, y, z, mass] からバッチボディ生成
world.add_bodies_batch(np.array([[0,10,0,1.0], [5,10,0,2.0]]))

# GILリリース付きバッチ速度更新
world.set_velocities_batch(velocities_array)  # (N,3)

# 結合状態出力 (N,10) [px,py,pz,vx,vy,vz,qx,qy,qz,qw]
states = world.states()
```

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
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALICE-Physics v0.4.0                                │
│              67モジュール、358ユニットテスト、10統合、11ドキュメントテスト         │
├─────────────────────────────────────────────────────────────────────────────┤
│  コアレイヤー                                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  math    │ │ collider │ │  solver  │ │   bvh    │ │sdf_colldr│          │
│  │ Fix128   │ │ AABB     │ │ RigidBody│ │ Morton   │ │ SdfField │          │
│  │ Vec3Fix  │ │ Sphere   │ │ XPBD     │ │Stackless │ │ Gradient │          │
│  │ QuatFix  │ │ Capsule  │ │ Sensor   │ │ Zero-    │ │ Early-out│          │
│  │ CORDIC   │ │ GJK/EPA  │ │ Rollback │ │  alloc   │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  AAAエンジンレイヤー (v0.4.0)                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │box_colldr│ │ compound │ │cont_cache│ │dynamic_bv│ │ material │          │
│  │ OBB      │ │ Multi-   │ │ HashMap  │ │ Incr BVH │ │ Pair Tbl │          │
│  │ GJK Supp │ │ Shape    │ │ O(1) Get │ │ AVL Bal  │ │ Combine  │          │
│  │ Inertia  │ │ Local Tx │ │ Warm Str │ │ O(log n) │ │ Friction │          │
│  │ Corners  │ │ AABB Mrg │ │ 4-point  │ │ Fat AABB │ │ Restit   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │dbg_rendr │ │profiling │ │ interp   │                                    │
│  │ Wireframe│ │ Timers   │ │ Snapshot │                                    │
│  │ Contacts │ │ Per-Stage│ │ NLERP    │                                    │
│  │ Joints   │ │ Stats    │ │ Blend    │                                    │
│  │ BVH/AABB │ │ History  │ │ Alpha    │                                    │
│  └──────────┘ └──────────┘ └──────────┘                                    │
│                                                                              │
│  拘束・ダイナミクスレイヤー                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  joint   │ │  motor   │ │articulatn│ │  force   │ │ sleeping │          │
│  │ Ball     │ │ PD 1D/3D │ │ Ragdoll  │ │ Wind     │ │ Islands  │          │
│  │ Hinge    │ │ Position │ │ FK Chain │ │ Gravity  │ │ Union-   │          │
│  │ Fixed    │ │ Velocity │ │ Robotic  │ │ Buoyancy │ │  Find    │          │
│  │ Slider   │ │ Max Torq │ │ Feather- │ │ Drag     │ │ Auto     │          │
│  │ Spring   │ │          │ │  stone   │ │ Vortex   │ │  Sleep   │          │
│  │ D6       │ │          │ │ 12-body  │ │          │ │          │          │
│  │ ConeTwst │ │          │ │          │ │          │ │          │          │
│  │ Breakable│ │          │ │          │ │          │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  クエリ・衝突レイヤー                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ raycast  │ │   ccd    │ │ trimesh  │ │heightfld │ │  filter  │          │
│  │ Sphere   │ │ TOI      │ │ Triangle │ │ Bilinear │ │ Layer    │          │
│  │ AABB     │ │ Conserv. │ │ BVH加速   │ │ Normal   │ │ Mask     │          │
│  │ Capsule  │ │ Advance  │ │ Moller-  │ │ Sphere   │ │ Group    │          │
│  │ Plane    │ │ Swept    │ │ Trumbore │ │ Collide  │ │ Bidirect │          │
│  │ Sweep    │ │ Specultv │ │ Closest  │ │ Signed   │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │  query   │ │character │                                                  │
│  │ SphCast  │ │ Move&Sld │                                                  │
│  │ CapCast  │ │ Stair    │                                                  │
│  │ Overlap  │ │ Ground   │                                                  │
│  │ AABB Ovr │ │ SDF Terr │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  ソフトボディ・シミュレーションレイヤー                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  rope    │ │  cloth   │ │  fluid   │ │deformable│ │ vehicle  │          │
│  │ XPBD     │ │ XPBD     │ │ PBF      │ │ FEM-XPBD │ │ Wheel    │          │
│  │ Distance │ │ Triangle │ │ SPH Hash │ │ Tetrahedr│ │ Suspensn │          │
│  │ Chain    │ │ Self-Col │ │ Density  │ │ Volume   │ │ Engine   │          │
│  │ Cable    │ │ SpatHash │ │ Viscosty │ │ Neo-Hook │ │ Steering │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  SDF拡張レイヤー                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │sdf_mnfld │ │ sdf_ccd  │ │sdf_force │ │sdf_destr │ │sdf_adapt │          │
│  │ Manifold │ │ SphTrace │ │ Attract  │ │ CSG Bool │ │ LOD      │          │
│  │ N-point  │ │ March    │ │ Repel    │ │ Subtract │ │ Distance │          │
│  │ Contact  │ │ TOI      │ │ Contain  │ │ Real-time│ │ Adaptive │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │cvx_decomp│ │ gpu_sdf  │                                                  │
│  │ Voxel    │ │ Compute  │                                                  │
│  │ Flood    │ │ Batch    │                                                  │
│  │ Convex   │ │ Shader   │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  SDFシミュレーションモディファイアレイヤー                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │sim_field │ │sim_modif │ │ thermal  │ │ pressure │ │ erosion  │          │
│  │ Scalar3D │ │ Modifier │ │ Heat Eq  │ │ Crush    │ │ Wind     │          │
│  │ Vector3D │ │ Chain    │ │ Melt     │ │ Bulge    │ │ Water    │          │
│  │ Trilin   │ │ Modified │ │ Freeze   │ │ Dent     │ │ Chemical │          │
│  │ Diffuse  │ │ SDF      │ │ Expand   │ │ Yield    │ │ Ablation │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │ fracture │ │phase_chg │                                                  │
│  │ Stress   │ │ Solid    │                                                  │
│  │ Crack    │ │ Liquid   │                                                  │
│  │ CSG Sub  │ │ Gas      │                                                  │
│  │ Voronoi  │ │ Latent H │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  ゲームシステムレイヤー                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │anim_blnd │ │audio_phys│ │ netcode  │                                    │
│  │ SLERP    │ │ Impact   │ │ FrameInp │                                    │
│  │ Ragdoll  │ │ Friction │ │ Checksum │                                    │
│  │ Blend    │ │ Rolling  │ │ Snapshot │                                    │
│  │ IK Mix   │ │ Material │ │ Rollback │                                    │
│  └──────────┘ └──────────┘ └──────────┘                                    │
│                                                                              │
│  アナリティクス & プライバシーレイヤー (v0.4.0)                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │  sketch  │ │ anomaly  │ │ privacy  │ │ pipeline │                       │
│  │ HyperLog │ │ MAD      │ │ Laplace  │ │ MetricPi │                       │
│  │ DDSketch │ │ EWMA     │ │ RAPPOR   │ │ RingBuf  │                       │
│  │ CountMin │ │ Z-score  │ │ RandResp │ │ Registry │                       │
│  │ HeavyHit │ │ Composit │ │ XorShift │ │ Snapshot │                       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
│                                                                              │
│  ユーティリティレイヤー                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────────────────────┐│
│  │   rng    │ │  event   │ │fluid_net │ │      neural (ALICE-ML × Phys)   ││
│  │ PCG-XSH  │ │ Begin    │ │ Delta    │ │ 三値 {-1,0,+1} → Fix128        ││
│  │ Fix128   │ │ Persist  │ │ Compress │ │ 決定論的AI                       ││
│  │ Direction│ │ End      │ │ Snapshot │ │ ラグドールコントローラ              ││
│  └──────────┘ └──────────┘ └──────────┘ └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
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

**ユーティリティ:**
- `Fix128::from_ratio(num, denom)` - 分数から生成
- `Fix128::half()`, `Fix128::double()` - 正確なビットシフト
- `Fix128::abs()`, `Fix128::floor()`, `Fix128::ceil()`

### `collider` - 衝突検出

| 形状 | 説明 |
|------|------|
| `AABB` | 軸平行バウンディングボックス |
| `Sphere` | 球体コライダー |
| `Capsule` | カプセル（円柱 + 半球） |
| `ConvexHull` | 任意の凸多面体 |
| `ScaledShape` | 均一スケールラッパー |
| `CollisionResult` | 接触情報 |

**アルゴリズム:**
- **GJK**: Gilbert-Johnson-Keerthi 交差判定（最大64回反復）
- **EPA**: Expanding Polytope Algorithm 貫通深度算出（最大64回反復）

### `solver` - XPBD物理

**RigidBodyフィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `position` | `Vec3Fix` | 重心位置 |
| `rotation` | `QuatFix` | 姿勢クォータニオン |
| `velocity` | `Vec3Fix` | 線形速度 |
| `angular_velocity` | `Vec3Fix` | 角速度 |
| `inv_mass` | `Fix128` | 逆質量（0 = 静的） |
| `inv_inertia` | `Vec3Fix` | 逆慣性テンソル（対角） |
| `restitution` | `Fix128` | 反発係数（0-1） |
| `friction` | `Fix128` | 摩擦係数 |
| `is_sensor` | `bool` | センサーモード: 物理応答なしでオーバーラップ検出 |

**拘束:**
- `DistanceConstraint`: アンカー点間の固定距離拘束
- `ContactConstraint`: 摩擦/反発付き衝突応答

**メソッド:**
- `RigidBody::new(position, mass)` - 動的ボディ生成
- `RigidBody::new_dynamic(position, mass)` - newのエイリアス
- `RigidBody::new_static(position)` - 不動ボディ生成
- `RigidBody::new_sensor(position)` - センサー/トリガーボディ生成

### `joint` - 7種のジョイント + 破壊可能拘束

| 型 | 説明 |
|----|------|
| `BallJoint` | 球面ジョイント（回転3自由度） |
| `HingeJoint` | ヒンジジョイント（回転1自由度、角度制限付き） |
| `FixedJoint` | 固定ジョイント（0自由度） |
| `SliderJoint` | スライダージョイント（並進1自由度、制限付き） |
| `SpringJoint` | 減衰スプリング拘束 |
| `D6Joint` | 6自由度設定可能ジョイント（軸ごとにロック/フリー/リミット） |
| `ConeTwistJoint` | コーンスイング制限+ツイスト制限付きボールジョイント |

全ジョイントは `with_break_force(max_force)` で**破壊可能拘束**をサポート。拘束力が閾値を超えるとジョイントが破壊されます。

```rust
use alice_physics::joint::*;
// 角度制限付きヒンジ
let hinge = HingeJoint::new(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
    .with_limits(-Fix128::HALF_PI, Fix128::HALF_PI);

// D6ジョイント（X並進ロック、Y回転フリー、Z回転リミット）
let d6 = D6Joint::new(body_a, body_b, anchor_a, anchor_b)
    .with_axis(Axis::LinearX, D6Mode::Locked)
    .with_axis(Axis::AngularY, D6Mode::Free)
    .with_axis(Axis::AngularZ, D6Mode::Limited(-Fix128::HALF_PI, Fix128::HALF_PI));

// 破壊可能ボールジョイント（力 > 100で破壊）
let ball = BallJoint::new(body_a, body_b, anchor_a, anchor_b)
    .with_break_force(Fix128::from_int(100));

// 破壊対応ソルブ — 破壊されたジョイントのインデックスを返す
let broken = solve_joints_breakable(&joints, &mut bodies, dt);
```

### `raycast` - レイ & シェイプキャスト

| 関数 | 説明 |
|------|------|
| `ray_sphere` | レイ vs 球体 |
| `ray_aabb` | レイ vs AABB（スラブ法） |
| `ray_capsule` | レイ vs カプセル |
| `ray_plane` | レイ vs 無限平面 |
| `sweep_sphere` | 移動球体 vs 球体（ミンコフスキー展開） |

### `query` - シェイプキャスト & オーバーラップクエリ

| 関数 | 説明 |
|------|------|
| `sphere_cast` | 球体を方向にスイープ |
| `capsule_cast` | カプセルを方向にスイープ（3点球体キャスト） |
| `overlap_sphere` | 球体内の全ボディを検出 |
| `overlap_aabb` | AABB内の全ボディを検出 |
| `batch_raycast`, `batch_sphere_cast` | バッチクエリ |

### `ccd` - 連続衝突検出

| 関数 | 説明 |
|------|------|
| `sphere_sphere_toi` | 球体-球体TOI（二次方程式） |
| `sphere_plane_toi` | 球体-平面TOI |
| `conservative_advancement` | 反復安全ステッピングによるTOI |
| `swept_aabb` | Swept AABBバウンディングボリューム |
| `speculative_contact` | ソルバー統合用推測的接触 |
| `needs_ccd` | CCD有効化のための速度閾値チェック |

### その他のモジュール

| モジュール | 説明 |
|-----------|------|
| `bvh` | モートンコード、エスケープポインタ、i32 AABBによるスタックレストラバーサル |
| `dynamic_bvh` | O(log n) 挿入/削除/更新付きインクリメンタルBVH（AVLバランシング） |
| `sleeping` | Union-Findアイランド、自動スリープ |
| `trimesh` | BVH加速三角メッシュ衝突（Moller-Trumbore） |
| `heightfield` | バイリニア補間地形、球体衝突、符号付き距離 |
| `filter` | レイヤー/マスクビットマスク衝突フィルタ |
| `force` | 風、重力井戸、ドラッグ、浮力、ボルテックス |
| `motor` | 1D/3D PDコントローラ、ジョイントモーター |
| `articulation` | 多関節チェーン、FK伝播、Featherstone O(n)順動力学、12体ラグドール |
| `rng` | PCG-XSH-RR 決定論的乱数 |
| `event` | Begin/Persist/End 接触イベント追跡 |
| `box_collider` | GJK対応OBB（慣性テンソル、体積、表面積） |
| `compound` | ローカル変換付きマルチシェイプ複合コライダー |
| `contact_cache` | HashMap O(1)マニフォールドルックアップ付き永続接触キャッシュ |
| `material` | ペアごとの摩擦/反発テーブル（合成ルール: 平均, 最小, 最大, 乗算） |
| `character` | キネマティックカプセルベースmove-and-slide（階段昇降・SDF地形対応） |
| `rope` | XPBD距離チェーンロープ・ケーブルシミュレーション |
| `cloth` | XPBDメッシュクロス（自己衝突対応、空間ハッシュグリッド） |
| `fluid` | Position-Based Fluids (PBF)（空間ハッシュグリッド） |
| `deformable` | FEM四面体メッシュ変形体 |
| `vehicle` | 車両物理（ホイール、サスペンション、エンジン、ステアリング） |
| `animation_blend` | ラグドール⇔アニメーションブレンド（SLERP） |
| `audio_physics` | 物理ベースオーディオパラメータ生成（衝突、摩擦、転がり） |
| `debug_render` | ワイヤーフレーム可視化API |
| `profiling` | ステージ別タイマーとフレーム統計 |
| `interpolation` | NLERP四元数補間付きWorldSnapshot |

### SDFモジュール

| モジュール | 説明 |
|-----------|------|
| `sdf_collider` | SDF衝突形状（距離+法線評価インターフェース） |
| `sdf_manifold` | SDF曲面からのマルチポイント接触マニフォールド |
| `sdf_ccd` | SDF向け球体トレーシングCCD |
| `sdf_force` | SDF駆動フォースフィールド（引力、斥力、封じ込め、フロー） |
| `sdf_destruction` | リアルタイムCSGブーリアン破壊 (std) |
| `sdf_adaptive` | 距離ベースLODによる適応的SDF評価 (std) |
| `convex_decompose` | SDFボクセルグリッドからの凸包分解 (std) |
| `gpu_sdf` | GPUコンピュートシェーダーバッチSDF評価 (std) |

### SDFシミュレーションモディファイア

| モジュール | 説明 |
|-----------|------|
| `sim_field` | トリリニア補間付き3Dスカラー/ベクトルフィールド (std) |
| `sim_modifier` | 物理駆動SDFモディファイアチェーン (std) |
| `thermal` | 熱拡散、融解、凍結、熱膨張 (std) |
| `pressure` | 接触力駆動変形（圧壊、膨張、凹み） (std) |
| `erosion` | 風食、水食、化学腐食、アブレーション (std) |
| `fracture` | 応力駆動亀裂伝播（Voronoi断片化、CSG減算） (std) |
| `phase_change` | 温度駆動相変化（固体/液体/気体） (std) |

### アナリティクス & プライバシー

| モジュール | 説明 |
|-----------|------|
| `sketch` | 確率的スケッチ: HyperLogLog, DDSketch, Count-Min Sketch, Heavy Hitters (std) |
| `anomaly` | ストリーミング異常検出: MAD, EWMA, Zスコア, 複合検出器 (std) |
| `privacy` | ローカル差分プライバシー: ラプラスノイズ, RAPPOR, ランダム化応答 (std) |
| `pipeline` | ロックフリーリングバッファメトリック集約パイプライン (std) |

### ゲームシステム

| モジュール | 説明 |
|-----------|------|
| `netcode` | 決定論的シミュレーション、FrameInput、チェックサム、スナップショット |
| `fluid_netcode` | デルタ圧縮付き決定論的流体ネットコード (std) |

## SDFコライダー（ALICE-SDF連携）

ALICE-Physicsは[ALICE-SDF](../ALICE-SDF)の距離場を衝突形状として使用できます。凸包（GJK/EPA）で近似する代わりに、SDFを直接サンプリングし、O(1)コストで数学的に正確な曲面を得ます。

### 仕組み

```
Body (球体)                      SdfCollider
  ┌───┐                         ┌──────────────────────┐
  │ ● │──world_to_local(pos)──▶│ SdfField::distance() │─── >0 → ヒットなし（早期脱出）
  └───┘                         │ SdfField::normal()   │─── ≤0 → 接触 + 解決
                                │ cached inv_rotation   │
                                │ cached scale_f32      │
                                └──────────────────────┘
```

### 主要な最適化

| 最適化 | 説明 | 効果 |
|--------|------|------|
| **早期脱出** | `distance()`（1回評価）を先に呼び、衝突時のみ `normal()`（4回評価）を計算 | 非衝突ボディで80%少ない評価 |
| **キャッシュ済み不変量** | 事前計算された `inv_rotation`、`scale_f32`、`inv_scale_f32` | クエリごとの再計算なし |
| **Rayon並列** | `--features parallel` で `par_iter_mut` | コア数に比例したスピードアップ |
| **4回評価結合** | 四面体勾配で距離+法線を4回の評価から取得 | ナイーブな1+4より1回少ない |

### 使い方

```rust
use alice_physics::prelude::*;
use alice_physics::sdf_collider::SdfCollider;
use alice_sdf::physics_bridge::CompiledSdfField;
use alice_sdf::prelude::*;

// 1. ALICE-SDFでSDF形状を作成
let terrain = SdfNode::plane(0.0, 1.0, 0.0, 0.0)  // 地面
    .union(SdfNode::sphere(2.0).translate(0.0, -1.5, 0.0));  // 丘

let field = CompiledSdfField::new(terrain);

// 2. 物理ワールドを作成
let mut world = PhysicsWorld::new(PhysicsConfig::default());

// 3. SDFを静的コライダーとして登録
let collider = SdfCollider::new_static(
    Box::new(field),
    Vec3Fix::ZERO,       // 位置
    QuatFix::IDENTITY,   // 回転
);
world.add_sdf_collider(collider);

// 4. 動的ボディを追加 — SDF表面と衝突します
let ball = RigidBody::new_dynamic(
    Vec3Fix::from_int(0, 10, 0),  // 地形の上から開始
    Fix128::ONE,                   // 質量
);
world.add_body(ball);

// 5. シミュレーション — SDF衝突はstep()で自動解決
let dt = Fix128::from_ratio(1, 60);
for _ in 0..300 {
    world.step(dt);
}
```

## 決定論的ニューラルコントローラ（ALICE-ML連携）

[ALICE-ML](../ALICE-ML)と連携し、1.58bit三値重み {-1, 0, +1} と128bit固定小数点演算を組み合わせた**ビット精度決定論的AI**を提供します。ニューラル推論が純粋な加算/減算に集約され、全クライアントで同一のAI動作を保証します。

ネットワーク格闘ゲームやアクションゲームの「聖杯」: 同期なしで全クライアントが同一のAI動作を計算します。

### 仕組み

```
三値重み {-1, 0, +1}:
  +1 → Fix128 加算
  -1 → Fix128 減算
   0 → スキップ（無料のスパーシティ）

結果: 推論パイプライン全体で浮動小数点乗算ゼロ
```

### ラグドールコントローラの例

```rust
use alice_physics::prelude::*;
use alice_ml::{TernaryWeight, quantize_to_ternary};

// 1. 学習済み重みを三値に量子化
let (w1, _) = quantize_to_ternary(&trained_weights_l1, hidden_size, input_size);
let (w2, _) = quantize_to_ternary(&trained_weights_l2, output_size, hidden_size);

// 2. 固定小数点に変換（一度だけ）
let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);
let ftw2 = FixedTernaryWeight::from_ternary_weight(w2);

// 3. 決定論的ネットワークを構築
let network = DeterministicNetwork::new(
    vec![ftw1, ftw2],
    vec![Activation::ReLU, Activation::HardTanh],
);

// 4. ラグドールコントローラを作成
let config = ControllerConfig {
    max_torque: Fix128::from_int(100),
    num_joints: 8,    // 8ジョイント × 3軸 = 24出力
    num_bodies: 9,    // 9パーツ × 13特徴 = 117入力
    features_per_body: 13,  // pos(3) + vel(3) + rot(4) + angvel(3)
};
let mut controller = RagdollController::new(network, config);

// 5. 物理ループ — 全クライアントで決定論的
for frame in 0..3600 {
    let output = controller.compute(&world.bodies);
    for (joint_idx, torque) in output.torques.iter().enumerate() {
        world.bodies[joint_idx].apply_impulse(*torque);
    }
    world.step(dt);
}
```

## 決定論的ネットコード

ALICE-Physicsはフレームベースの決定論的ネットコード基盤を含んでいます。エンジンがビット精度の結果を保証するため、**プレイヤー入力のみの同期で済み**、状態同期は不要です。

### 帯域幅削減

| 方式 | フレームあたり（10ボディ、2プレイヤー） |
|------|--------------------------------------|
| 状態同期（従来） | ~1,600バイト |
| **入力同期（ALICE）** | **~40バイト** |
| 削減率 | **97.5%** |

### コア型

| 型 | 説明 |
|----|------|
| `FrameInput` | 20バイトシリアライズ可能プレイヤー入力（移動、アクション、エイム） |
| `SimulationChecksum` | XORローリングハッシュによる物理状態チェックサム |
| `SimulationSnapshot` | ロールバック用完全状態キャプチャ |
| `DeterministicSimulation` | PhysicsWorldラッパー（フレームカウンタ、チェックサム履歴、スナップショットリングバッファ） |
| `InputApplicator` | ゲーム固有の入力→物理力マッピング用トレイト |

### 使い方

```rust
use alice_physics::prelude::*;

// 両クライアントが同一のシミュレーションを作成
let mut sim = DeterministicSimulation::new(NetcodeConfig::default());

// ボディを追加してプレイヤーに割り当て
let body0 = sim.add_body(RigidBody::new_dynamic(
    Vec3Fix::from_int(0, 10, 0), Fix128::ONE,
));
sim.assign_player_body(0, body0);

// 各フレーム: 入力を収集、進行、チェックサム比較
let inputs = vec![
    FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0)),
    FrameInput::new(1).with_movement(Vec3Fix::from_int(0, 0, -1)),
];
let checksum = sim.advance_frame(&inputs);

// ロールバック用スナップショット保存
sim.save_snapshot();

// リモートクライアントのチェックサムを検証
assert_eq!(sim.verify_checksum(1, checksum), Some(true));
```

## ALICE-Sync連携（ゲームエンジンパイプライン）

ALICE-Physicsは[ALICE-Sync](../ALICE-Sync)と連携し、完全なマルチプレイヤーゲームネットワーキングを実現します。

```
プレイヤー入力 ──► InputFrame (i16, 24B) ──► FrameInput (Fix128) ──► PhysicsWorld
                     ALICE-Sync                 bridge                 step()
                                                  │
PhysicsWorld ──► SimulationChecksum ──► WorldHash ──► Desync検証
                   from_world()           bridge        ALICE-Sync
```

### 帯域幅

| 方式 | フレームあたり（4プレイヤー、60fps） |
|------|--------------------------------------|
| 状態同期 | ~960 KB/s |
| **入力同期（ALICE）** | **5.6 KB/s** |
| 削減率 | **99.4%** |

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

// またはデフォルトを使用
let config = PhysicsConfig::default();
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
| `wasm` | No | WebAssemblyバインディング（wasm-bindgen） |
| `analytics` | No | ALICE-Analytics経由のシミュレーションプロファイリング |

```bash
# SIMD最適化
cargo build --release --features simd

# 並列拘束解決
cargo build --release --features parallel

# 両方有効化
cargo build --release --features "simd,parallel"

# ニューラルコントローラ（ALICE-ML必要）
cargo build --release --features neural

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

// ワールド作成
AlicePhysicsWorld* world = alice_physics_world_create();

// ボディ追加
AliceVec3 pos = {0.0, 10.0, 0.0};
uint32_t body = alice_physics_body_add_dynamic(world, pos, 1.0);

// シミュレーションステップ
alice_physics_world_step(world, 1.0 / 60.0);

// 位置取得
AliceVec3 out_pos;
alice_physics_body_get_position(world, body, &out_pos);

// ステートシリアライズ（ロールバックネットコード用）
uint32_t len;
uint8_t* state = alice_physics_state_serialize(world, &len);
alice_physics_state_deserialize(world, state, len);
alice_physics_state_free(state, len);

// クリーンアップ
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
git tag v0.4.0
git push origin v0.4.0
```

GitHub Actionsが macOS (ARM + Intel)、Windows、Linux 向けにビルドし、UE5プラグインZIPとUnityパッケージZIPをリリースに添付します。

## Pythonバインディング（PyO3 + NumPyゼロコピー）

インストール：

```bash
pip install maturin
maturin develop --release --features python
```

### 最適化レイヤー

| レイヤー | 手法 | 効果 |
|---------|------|------|
| L1 | GILリリース (`py.allow_threads`) | 並列物理ステッピング |
| L2 | ゼロコピーNumPy (`into_pyarray_bound`) | バルク位置/速度のmemcpyなし |
| L3 | バッチAPI (`step_n`, `positions`, `states`) | FFI償却 |
| L4 | バッチ変更 (`add_bodies_batch`, `set_velocities_batch`, `apply_impulses_batch`) | GILリリース付きバルク操作 |
| L5 | Rustバックエンド (Fix128, XPBD, BVH) | ハードウェア速度シミュレーション |

### Python API

```python
import alice_physics

# 基本的な物理ワールド
world = alice_physics.PhysicsWorld()
body0 = world.add_dynamic_body(0.0, 10.0, 0.0, mass=1.0)
ground = world.add_static_body(0.0, 0.0, 0.0)

# GILリリース付きステップ（他のPythonスレッドが実行可能）
world.step(1.0 / 60.0)

# トレーニングループ向けバッチステップ
world.step_n(1.0 / 60.0, steps=300)

# 全位置をNumPy (N, 3) float64配列で取得（ゼロコピー）
positions = world.positions()  # shape: (N, 3)
velocities = world.velocities()  # shape: (N, 3)

# 決定論的ネットコードシミュレーション
sim = alice_physics.DeterministicSimulation(player_count=2, fps=60)
body = sim.add_body(0.0, 10.0, 0.0, mass=1.0)
sim.assign_player(0, body)

# プレイヤー入力で進行: (player_id, move_x, move_y, move_z, actions)
checksum = sim.advance_frame([(0, 1.0, 0.0, 0.0, 0), (1, 0.0, 0.0, -1.0, 0)])

# ロールバック用スナップショット
frame = sim.save_snapshot()
sim.load_snapshot(frame)

# シリアライゼーション
state = world.serialize_state()  # NumPy uint8配列
world.deserialize_state(state.tolist())

# フレーム入力エンコード（20バイト、ネットワーク対応）
data = alice_physics.encode_frame_input(player_id=0, move_x=1.0, actions=0x3)
player_id, mx, my, mz, actions, ax, ay, az = alice_physics.decode_frame_input(data)

# === バッチAPI (v0.4.0) ===

import numpy as np

# (N,4) 配列 [x, y, z, mass] からバッチボディ生成
ids = world.add_bodies_batch(np.array([
    [0.0, 10.0, 0.0, 1.0],
    [5.0, 10.0, 0.0, 2.0],
    [10.0, 10.0, 0.0, 0.5],
]))

# GILリリース付きバッチ速度更新 (N,3)
world.set_velocities_batch(np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 5.0, 0.0],
]))

# バッチインパルス (M,4) [body_id, ix, iy, iz]（GILリリース付き）
world.apply_impulses_batch(np.array([
    [0.0, 100.0, 0.0, 0.0],
    [2.0, 0.0, 50.0, 0.0],
]))

# 結合状態出力 (N,10) [px,py,pz, vx,vy,vz, qx,qy,qz,qw]
states = world.states()  # shape: (N, 10), ゼロコピーNumPy
```

## リプレイ録画（ALICE-DB連携）

ALICE-Physicsは[ALICE-DB](../ALICE-DB)経由でシミュレーション軌跡の録画・再生が可能です。`--features replay` で有効化。

### 録画

```rust
use alice_physics::replay::ReplayRecorder;

// レコーダー作成（パス、ボディ数）
let mut recorder = ReplayRecorder::new("./replay_data", 3)?;

// ゲームループ: 毎フレーム記録
for _ in 0..300 {
    world.step(dt);
    recorder.record_frame(&world)?;
}

recorder.flush()?;
recorder.close()?;
```

### 再生

```rust
use alice_physics::replay::ReplayPlayer;

let player = ReplayPlayer::open("./replay_data", 3)?;

// ランダムアクセス: 任意フレームの位置を取得
let pos = player.get_position(frame, body_id)?;

// 範囲クエリ: フレーム範囲の位置をスキャン
let trajectory = player.scan_positions(0, 299, body_id)?;

player.close()?;
```

## 浮動小数点エンジンとの比較

| 観点 | ALICE-Physics | 浮動小数点エンジン |
|------|---------------|-----------------|
| 決定論性 | 保証 | プラットフォーム依存 |
| 精度 | 64bit小数部 | 23bit (f32) / 52bit (f64) |
| 速度 | やや遅い（2-5x） | 速い |
| ロールバック | 容易 | 注意が必要 |
| 組み込み | no_std | FPU必要 |
| 数値範囲 | ±9.2×10^18 | ±3.4×10^38 (f32) |

## クロスクレートブリッジ

ALICE-Physicsはフィーチャーゲート付きブリッジモジュールで他のALICEエコシステムクレートと接続します：

| ブリッジ | Feature | 対象クレート | 説明 |
|---------|---------|------------|------|
| Physics Visualization | `view` | [ALICE-View](../ALICE-View) | リアルタイム物理デバッグオーバーレイ |
| GPU Physics Controller | `trt` | [ALICE-TRT](../ALICE-TRT) | GPU三値推論による物理制御ポリシー |
| Physics State Streaming | `asp` | [ALICE-Streaming-Protocol](../ALICE-Streaming-Protocol) | 物理ボディ状態のASP D-パケットデルタエンコード |
| `db_bridge` | `replay` | [ALICE-DB](../ALICE-DB) | 物理状態スナップショット永続化 |
| `analytics_bridge` | `analytics` | [ALICE-Analytics](../ALICE-Analytics) | シミュレーションプロファイリング |

## 完全APIリファレンス

### コア型

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `math` | `Fix128` | 128bit固定小数点数（I64F64） |
| `math` | `Vec3Fix` | Fix128成分の3Dベクトル |
| `math` | `QuatFix` | 回転用クォータニオン |
| `math` | `Mat3Fix` | 慣性テンソル用3x3行列 |
| `math` | `SIMD_WIDTH` | コンパイル時SIMDレーン数（AVX2=8, NEON=4, スカラー=1） |
| `math` | `simd_width()` | コンパイル時SIMD幅を返す |
| `math` | `select_fix128()` | ブランチレス条件選択 |
| `math` | `select_vec3()` | ブランチレスVec3条件選択 |

### 衝突検出

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `collider` | `AABB` | 軸平行バウンディングボックス |
| `collider` | `Sphere` | 球体コライダー |
| `collider` | `Capsule` | カプセルコライダー |
| `collider` | `ConvexHull` | 任意の凸多面体 |
| `collider` | `ScaledShape` | 均一スケールラッパー |
| `collider` | `CollisionResult` | 接触点、法線、深度 |
| `collider` | `Support` trait | GJKサポート関数インターフェース |
| `box_collider` | `OrientedBox` | 中心、半径、回転付きOBB |
| `compound` | `CompoundShape` | マルチシェイプ複合コライダー |
| `cylinder` | `Cylinder` | GJK対応シリンダーコライダー |
| `filter` | `CollisionFilter` | レイヤー/マスク衝突グループ |

### ソルバー & ダイナミクス

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `solver` | `PhysicsWorld` | メインシミュレーションワールド |
| `solver` | `PhysicsConfig` | サブステップ、反復、重力、減衰 |
| `solver` | `RigidBody` | 動的/静的/センサー剛体 |
| `solver` | `DistanceConstraint` | アンカー点間の固定距離 |
| `solver` | `ContactConstraint` | 摩擦/反発付き衝突応答 |
| `solver` | `ContactModifier` trait | カスタム接触修正 (std) |
| `joint` | `BallJoint` | 球面ジョイント（3自由度） |
| `joint` | `HingeJoint` | 角度制限付きヒンジジョイント |
| `joint` | `FixedJoint` | 溶接ジョイント（0自由度） |
| `joint` | `SliderJoint` | 制限付きスライダージョイント |
| `joint` | `SpringJoint` | 減衰スプリング拘束 |
| `joint` | `D6Joint` | 6自由度設定可能ジョイント |
| `joint` | `ConeTwistJoint` | コーン+ツイスト制限ジョイント |
| `joint` | `solve_joints_breakable()` | 破壊力対応ソルブ |
| `motor` | `PdController` | 1D比例-微分コントローラ |
| `motor` | `JointMotor` | 位置/速度/トルクモードモーター |
| `force` | `ForceField` | 風、重力井戸、ドラッグ、浮力、ボルテックス |
| `material` | `MaterialTable` | ペアごと摩擦/反発テーブル |
| `material` | `CombineRule` | 平均、最小、最大、乗算 |

### 空間加速

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `bvh` | `LinearBvh` | スタックレストラバーサル付きフラット配列BVH |
| `bvh` | `BvhNode` | 32バイトキャッシュ整列ノード |
| `dynamic_bvh` | `DynamicAabbTree` | O(log n) 挿入/削除付きインクリメンタルBVH |
| `spatial` | `SpatialGrid` | 近傍クエリ用ハッシュグリッド |

### クエリ

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `raycast` | `Ray`, `RayHit` | レイ原点+方向、ヒット結果 |
| `raycast` | `ray_sphere()`, `ray_aabb()`, `ray_capsule()`, `ray_plane()` | 形状別レイテスト |
| `query` | `sphere_cast()`, `capsule_cast()` | シェイプスイープ |
| `query` | `overlap_sphere()`, `overlap_aabb()` | オーバーラップクエリ |
| `query` | `batch_raycast()`, `batch_sphere_cast()` | バッチクエリ |
| `ccd` | `sphere_sphere_toi()` | 衝突時刻（球体-球体） |
| `ccd` | `conservative_advancement()` | 反復安全ステッピングTOI |
| `ccd` | `speculative_contact()` | 推測的CCD接触 |

### ソフトボディ & シミュレーション

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `rope` | `Rope`, `RopeConfig` | XPBD距離チェーンロープ |
| `cloth` | `Cloth`, `ClothConfig` | XPBDメッシュクロス |
| `fluid` | `Fluid`, `FluidConfig` | Position-Based Fluids |
| `deformable` | `DeformableBody`, `DeformableConfig` | FEM四面体メッシュ |
| `vehicle` | `Vehicle`, `VehicleConfig` | 車両シミュレーション |
| `character` | `CharacterController`, `CharacterConfig` | キネマティックmove-and-slide |

### SDF統合

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `sdf_collider` | `SdfCollider`, `SdfField` trait | SDF衝突形状 |
| `sdf_manifold` | `SdfManifold`, `ManifoldConfig` | マルチポイント接触 |
| `sdf_ccd` | `SdfCcdConfig` | 球体トレーシングCCD |
| `sdf_force` | `SdfForceField`, `SdfForceType` | SDF駆動力場 |
| `sdf_destruction` | `DestructibleSdf`, `DestructionShape` | CSGブーリアン破壊 (std) |
| `sdf_adaptive` | `AdaptiveSdfEvaluator` | 距離ベースLOD (std) |
| `gpu_sdf` | `GpuSdfBatch`, `GpuSdfInstancedBatch`, `GpuSdfMultiDispatch` | GPUバッチSDF (std) |

### SDFシミュレーションモディファイア

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `sim_field` | `ScalarField3D`, `VectorField3D` | 3Dフィールド (std) |
| `sim_modifier` | `PhysicsModifier` trait, `ModifiedSdf` | SDFモディファイアチェーン (std) |
| `thermal` | `ThermalModifier`, `ThermalConfig` | 熱拡散・融解・凍結 (std) |
| `pressure` | `PressureModifier`, `PressureConfig` | 接触力変形 (std) |
| `erosion` | `ErosionModifier`, `ErosionType` | 風食・水食・化学腐食 (std) |
| `fracture` | `FractureModifier`, `FractureConfig` | 応力駆動亀裂伝播 (std) |
| `phase_change` | `PhaseChangeModifier`, `Phase` | 固体/液体/気体遷移 (std) |

### ゲームシステム

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `animation_blend` | `AnimationBlender`, `BlendMode`, `SkeletonPose` | アニメーションブレンド |
| `audio_physics` | `AudioGenerator`, `AudioEvent`, `AudioMaterial` | 物理ベースオーディオ |
| `netcode` | `DeterministicSimulation`, `FrameInput`, `SimulationChecksum` | 決定論的ネットコード |
| `fluid_netcode` | `FluidSnapshot`, `FluidDelta` | 流体ネットコード (std) |

### アナリティクス & プライバシー

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `sketch` | `HyperLogLog`, `DDSketch`, `CountMinSketch`, `HeavyHitters` | 確率的スケッチ (std) |
| `sketch` | `FnvHasher`, `Mergeable` trait | ハッシュ・分散マージ (std) |
| `anomaly` | `MadDetector`, `EwmaDetector`, `ZScoreDetector`, `CompositeDetector` | 異常検出 (std) |
| `privacy` | `LaplaceNoise`, `Rappor`, `RandomizedResponse` | 差分プライバシー (std) |
| `privacy` | `PrivacyBudget`, `PrivateAggregator`, `XorShift64` | 予算管理・集約 (std) |
| `pipeline` | `MetricPipeline`, `MetricRegistry`, `MetricEvent`, `MetricType` | メトリック集約 (std) |

### ユーティリティ

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `rng` | `DeterministicRng` | PCG-XSH-RR 32bit生成器 |
| `event` | `EventCollector`, `ContactEvent` | 接触イベント追跡 |
| `sleeping` | `IslandManager`, `SleepConfig` | Union-Findアイランド管理 |
| `trimesh` | `TriMesh`, `Triangle` | BVH加速三角メッシュ |
| `heightfield` | `HeightField` | バイリニア補間地形 |
| `articulation` | `ArticulatedBody`, `FeatherstoneSolver` | 多関節体・O(n)順動力学 |
| `contact_cache` | `ContactCache`, `ContactManifold` | HashMap O(1)マニフォールド |
| `interpolation` | `WorldSnapshot` | 物理状態補間 |
| `debug_render` | `DebugDrawData`, `DebugDrawFlags` | ワイヤーフレーム可視化 |
| `profiling` | `PhysicsProfiler`, `StepStats` | ステージ別タイマー |
| `error` | `PhysicsError` | 型付き物理エラー列挙 |

### フィーチャーゲート付きモジュール

| モジュール | Feature | 説明 |
|-----------|---------|------|
| `neural` | `neural` | 決定論的三値ニューラルコントローラ（ALICE-ML） |
| `python` | `python` | PyO3 + NumPyゼロコピーPythonバインディング |
| `ffi` | `ffi` | Unity/UE5向けC FFI |
| `wasm` | `wasm` | WebAssemblyバインディング（wasm-bindgen） |
| `replay` | `replay` | ALICE-DB経由リプレイ録画/再生 |
| `db_bridge` | `replay` | 物理状態永続化ブリッジ |
| `analytics_bridge` | `analytics` | シミュレーションプロファイリングブリッジ |

### Python API (PyO3)

| クラス / 関数 | 説明 |
|--------------|------|
| `PhysicsWorld` | 物理シミュレーションのPythonラッパー |
| `PhysicsWorld.add_dynamic_body(x, y, z, mass)` | 動的ボディ追加 |
| `PhysicsWorld.add_static_body(x, y, z)` | 静的ボディ追加 |
| `PhysicsWorld.step(dt)` | GILリリース付きステップ |
| `PhysicsWorld.step_n(dt, steps)` | トレーニング向けバッチステップ |
| `PhysicsWorld.positions()` | NumPy (N,3) ゼロコピー |
| `PhysicsWorld.velocities()` | NumPy (N,3) ゼロコピー |
| `PhysicsWorld.states()` | NumPy (N,10) 結合状態 |
| `PhysicsWorld.add_bodies_batch(array)` | (N,4)からバッチ生成 |
| `PhysicsWorld.set_velocities_batch(array)` | (N,3) バッチ速度更新 |
| `PhysicsWorld.apply_impulses_batch(array)` | (M,4) バッチインパルス |
| `PhysicsWorld.serialize_state()` | NumPy uint8への状態出力 |
| `PhysicsWorld.deserialize_state(data)` | バイトから状態復元 |
| `DeterministicSimulation` | ネットコードシミュレーションラッパー |
| `encode_frame_input(...)` | 20バイト入力エンコード |
| `decode_frame_input(data)` | 20バイト入力デコード |

### C FFI API

| 関数 | 説明 |
|------|------|
| `alice_physics_world_create()` | 物理ワールド作成 |
| `alice_physics_world_destroy(world)` | 物理ワールド破棄 |
| `alice_physics_world_step(world, dt)` | シミュレーションステップ |
| `alice_physics_body_add_dynamic(world, pos, mass)` | 動的ボディ追加 |
| `alice_physics_body_add_static(world, pos)` | 静的ボディ追加 |
| `alice_physics_body_get_position(world, id, out)` | ボディ位置取得 |
| `alice_physics_body_apply_impulse(world, id, impulse)` | インパルス適用 |
| `alice_physics_state_serialize(world, len)` | 状態シリアライズ |
| `alice_physics_state_deserialize(world, data, len)` | 状態デシリアライズ |
| `alice_physics_state_free(data, len)` | シリアライズバッファ解放 |

## テスト結果

```
v0.4.0 テストサマリ:
  - 67モジュール全体で358ユニットテスト
  - 10統合テスト（エンドツーエンド物理シナリオ）
  - 11ドキュメントテスト（実行可能な例）
  - 合計: 379テストパス
  - Clippy: 0警告
  - 全フィーチャー組み合わせパス（default, parallel, simd）
```

## ライセンス

AGPL-3.0 - 詳細は [LICENSE](LICENSE) を参照。

Copyright (C) 2024-2026 Moroya Sakamoto

## 謝辞

- XPBD: Muller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- GJK/EPA: Ericson, "Real-Time Collision Detection"
- Morton Codes: Morton, "A Computer Oriented Geodetic Data Base"
- CORDIC: Volder, "The CORDIC Trigonometric Computing Technique"
