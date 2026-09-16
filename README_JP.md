# ALICE-Physics

**決定論的128bit固定小数点物理エンジン** - v0.14.0-preview.6

[English](README.md) | 日本語

[![crates.io](https://img.shields.io/crates/v/alice-physics.svg)](https://crates.io/crates/alice-physics)
[![docs.rs](https://img.shields.io/docsrs/alice-physics)](https://docs.rs/alice-physics)
[![License: AGPL-3.0-or-later](https://img.shields.io/crates/l/alice-physics.svg)](#license)
[![CI](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml)

> **ライセンス**: AGPL-3.0-or-later AGPL 義務なしの商用ライセンスは別途用意 — <sakamoro@alicelaw.net> まで 詳細は [ライセンス](#ライセンス)

異なるプラットフォームやハードウェア間で決定論的なシミュレーションを実現する高精度物理エンジン。rigid-body core は 128bit 固定小数点演算 (`Fix128`)、周辺の engineering / field module は IEEE `f32` / `f64` + 超越関数を crate 自前の `det_math` に統一 いずれも CPU、コンパイラ、OS に関わらずビット精度の結果を保証する — [決定論の範囲](#決定論の範囲) 参照

**crates.io で v1.1.0 公開中** (1.0.0 semver-locked stable は 2026-09-14、1.1.0 は 2026-09-15 = `det_math` で全 module cross-platform bit-exact) `cargo add alice-physics` でインストール可能 v1.0 roadmap 9 項目完了 — release 全容は [`CHANGELOG.md`](CHANGELOG.md)、0.x → 1.0 移行は [`docs/MIGRATION_0.x_TO_1.0.md`](docs/MIGRATION_0.x_TO_1.0.md)、凍結済 partner API contract は [`docs/ECOSYSTEM_CONTRACTS.md`](docs/ECOSYSTEM_CONTRACTS.md) 参照

### 決定論の範囲

crate 内の全 module が **プラットフォーム跨ぎで bit-exact** 理由は 2 系統:

| Tier | Module | bit が一致する理由 |
|------|--------|--------------------|
| **Fix128 core** | rigid-body solver / joint / distance・contact constraint / BVH broad-phase / GJK・EPA / CCD / sleeping / scene I/O / netcode snapshot・rollback / neural controller、および public API が `Fix128` / `Vec3Fix` / `QuatFix` の全 module | 純 `i64` / `u64` 整数演算 `tests/determinism_golden.rs` で pin |
| **`f32` / `f64` field module** (27: `acoustic_wave` / `aeroelasticity` / `anomaly` / `convex_decompose` / `erosion` / `fracture` / `gpu_sdf` / `phase_change` / `piezoelectric` / `pressure` / `privacy` / `rolling_contact` / `sdf_adaptive` / `sdf_ccd` / `sdf_character` / `sdf_collider` / `sdf_destruction` / `sdf_fem_mesh` / `sdf_manifold` / `sdf_sph` / `sdf_wind_field` / `sim_field` / `sim_modifier` / `sketch` / `thermal` / `thin_wall` / `transient_thermal`、一覧は `scripts/f32_modules.py --check` が生成し CI がこの行を gate、他に I/O 境界で float を渡すだけの 8 module: `ffi` / `pipeline` / `analytics_bridge` / `db_bridge` / `replay` / `fluid_netcode` / `character_state` / `solver_tgs`) | IEEE 754 の `+ - * / sqrt` と fused multiply-add は Rust が支援する全 target (SSE2+ / aarch64 / wasm32) で bit 単位に規定済 超越関数 (`sin` / `exp` / `ln` / `powf` / `cbrt` / `hypot` …) は全て [`det_math`](src/det_math.rs) (整数引数還元 + 固定順序多項式) を経由し、プラットフォーム `libm` を呼ばない `clippy.toml` の `disallowed-methods` で `libm` 呼出は CI error `tests/determinism_golden_f32.rs` (13 scenario / 29 module) で pin |

両 golden suite は CI で macOS ARM/x86 / Linux ARM/x86 / Windows / `wasm32-wasip1` を通る

**境界 — 渡す側の code** `ClosureSdf` は user closure `Fn(f32, f32, f32) -> f32` を受ける solver 側の決定論は保たれるが closure は呼出側の code なので、内部では `f32::sin` 等でなく `alice_physics::det_math::{sin, exp, …}` を使うか、同じ規律の evaluator を持つ SDF を渡すこと (ALICE-SDF の CPU evaluator は `det_math` への整合を別途進行中、それまでは同一 binary 内決定論として扱う)

**保証対象外** IEEE 754 の基本演算を守らない target: x87 向け 32-bit x86 (`i586`、SSE2 なし) と fast-math 系 flag 付き build

### 正しさの範囲 — 「決定論的」が言っていること / 言っていないこと

bit 一致は「全 peer が同じ数値を出す」ことしか保証しない、その数値が正しいかは別の性質 2026-09-15 の外部レビューは 1456 test + golden 44 が全 green のまま既定 config の物理バグ 4 件を見つけた (`CHANGELOG.md` 1.2.0)、以後この 2 つを明示的に分けて扱う 下表の **検証** は「`tests/` で閉形式解 or 独立参照と突合済」の意味、golden hash は *変化* しか検出しない

| 層 | module | 検証 |
|----|--------|------|
| **Core** — 決定論保証の本体 | `math` (`Fix128` / `Vec3Fix` / `QuatFix` / CORDIC)、`solver` (XPBD 剛体、距離 / 接触拘束、sleeping)、`joint`、`bvh`、`collider` (GJK / EPA)、`ccd`、`contact_cache`、`sdf_collider`、`netcode` / snapshot | `tests/analytic_physics.rs` — 自由落下 / 放物 / 終端速度 / `mg/k` 伸び / バネと振り子の周期 / 衝突の運動量 + energy 上限 / kinematic 目標 / torque-free 回転 / 静止接触、`math::tests` の sqrt / 超越関数 oracle、`det_math` の correctly-rounded 参照 sweep |
| **Engineering / field module** — 教科書公式の決定論実装 | `transient_thermal`、`fatigue` | `tests/engineering_oracles.rs` — cosine 固有 mode 減衰 (Carslaw & Jaeger、explicit / Crank–Nicolson)、Basquin / Miner 閉形式 |
| | `cloth`、`rope`、`deformable`、`fluid` (PBF)、`character`、`vehicle` (`WheelConfig` / `EngineConfig`)、`sleeping`、`netcode`、`audio_physics`、`support`、`neural` | `tests/default_configs.rs` — 全 `Config::default()` を最も普通の scenario で: カーテン / ロープの幾何、deformable 立方体と fluid block 重心の自由落下閉形式、wheel 圧縮 `mg/k`、engine 初 frame `Δv = T·gear/(r·m)·dt`、`frames_to_sleep` ちょうどで sleep、lockstep checksum + rollback bit 一致、support 体積 1e-6 |
| | *固体 / 構造*: `beam_stress`、`buckling`、`thin_wall`、`fillet_stress`、`modal`、`vibration_wall`、`damping_rayleigh`、`hyperelastic`、`plastic`、`creep_longterm`、`structural_solver`、`laminate`、`laminate_failure` (Tsai-Wu / Hashin / Puck)、`anisotropic`、`bimaterial`、`prestressed`、`bridging`、`thermal_stress`、`mass_properties`、`print_orientation`、`filament_db` | `tests/engineering_oracles_solid.rs` (34 本) — Roark 梁たわみ / 応力 / 断面性能、Euler–Johnson 座屈 (Shigley)、Rayleigh 減衰 (Chopra)、CLT 積層板 ABD と ply 変換 (Jones)、一軸強度点での Tsai-Wu / Hashin 包絡、Shigley ボルト継手と Irvine ケーブル、Kirsch / Inglis 厳密解と Peterson fit 不変量、Norton クリープ、Findley / WLF、超弾性一軸応力 = ひずみエネルギー微分、1 次モード (Blevins 梁 / Leissa 板 / Rao 1 自由度)、閉形式慣性テンソルと Steiner 平行軸、混合則 / 熱ミスマッチ、熱応力の大きさと符号 |
| | *流体 / 熱 / 場*: `cfd_solver`、`eulerian_grid`、`sdf_sph`、`multiphase`、`interface_capture`、`surface_tension_csf`、`compressible`、`non_newtonian`、`turbulence`、`buoyancy_zone`、`wind_zone`、`sdf_wind_field`、`cloth_fluid`、`fsi_advanced`、`smoke_fire`、`wave_ship`、`aeroelasticity` (Facchinetti 2004)、`acoustic_wave`、`electromagnetic`、`piezoelectric`、`thermal`、`phase_change`、`pressure`、`erosion` | `tests/engineering_oracles_fluid.rs` (39 本) — 粘性せん断 mode 減衰、圧力投影の発散除去と solenoidal 場での恒等性、SPH kernel 正規化と格子静止密度、CSF band 越しの Young–Laplace、ISA 密度 / 音速、垂直衝撃波 / 等エントロピー表 (Anderson)、Bingham / Herschel–Bulkley / power-law / Carreau 流動曲線 (整数 + 分数指数)、Smagorinsky 単純せん断、Archimedes、2 乗抗力、Courant 1 での d'Alembert、Coulomb / Lorentz / 双極子、d33 圧電関係、Arrhenius / Boussinesq、深水分散 / Froude–Krylov / JONSWAP、van der Pol 振幅と Strouhal 周波数、Newton 冷却と伝導固有 mode、エンタルピー法の plateau と保存、level-set 曲率と fast-sweeping 距離、侵食 rate law 不変量 |
| | *機構 / 場 / netcode*: `rolling_contact`、`fracture`、`kinematic_loop`、`sdf_force`、`anisotropic_friction`、`physics2d`、`sdf_destruction`、`soft_body_cut`、`layer_adhesion`、`fluid_netcode` | `tests/engineering_oracles_misc.rs` (27 本) — Hertz 球接触と Basquin 寿命 (Johnson §4.2)、亀裂成長 `L = min(vt, L_max)` と Quilez capsule 減算、PBD loop closure (質量重み射影 / 残差の幾何収束 / 四節リンク長)、`F = −kφ∇φ` と contain / repel / vortex 閉形式、直交異方性摩擦楕円 (Zmitrowicz) の等方 Coulomb 極限、2-D 自由落下 / `θ = ωt` / 運動量 / Newton 反発 `v_sep = e·v_app` / Coulomb 滑り `μ g` / 慣性 (Meriam & Kraige)、半球クレーターと円柱孔の体積、平面切断の分割不変量、snapshot の bit 一致 round trip と `τ = 0` lossless delta |
| | `warp_risk` (経験 fit: ALICE-Bamboo 実測 2 点)、`sim_field`、`layer_adhesion` (経験 FDM 係数、不変量のみ) と残りの utility / bridge module | **未検証** — 公式は文献通りに実装され bit-exact だが、参照 solver / 教科書例題 / 実測との突合 test がまだ無い 数値は「引用式の実装」であって validated prediction ではない、module 単位で oracle を追加していく (1.2.0 の batch が pattern: `Config::default()` で閉形式、精度 parameter を振る) |

**向いている用途** bit-exact replay が要件そのものである領域: rollback netcode (格闘 / RTS / .io、数十〜数百 body)、server 側 replay 検証 / anti-cheat、再現性が要る研究 / 監査 batch 1000 体重なり球 scene (`cargo bench --bench physics_bench` `thousand_overlapping_spheres_1_step`) で既定 config 数 ms/frame **向いていない用途** 一般ゲーム物理の置換: 破壊表現、数千体の群衆、60 fps の VFX 級接触数は設計点の外、決定論が要らないなら float engine が固定小数点のコストを払う理由はない

**v0.10-0.14 の主な追加**: 5 wave の完全実装プッシュで **54 module + 3 統合 solver loop + Session 4 19 module (3 tier 分類)** を追加 3D プリント安全性検証 (warp / thin-wall / stress / bridging) から composite / plastic / fatigue 力学、乱流、VOF / level-set 多相流、実行可能な CFD 時間ステップ loop、humanoid ragdoll、SDF-boundary SPH、transient thermal、composite failure、VIV / piezoelectric / acoustic / electromagnetic、IK / anisotropic friction / netcode prediction / character FSM / kinematic loop / buoyancy zone / wind zone / SDF FEM / SDF wind までカバー 全て platform 跨ぎで bit-exact (Fix128 は整数演算、`f32` は `det_math`、[決定論の範囲](#決定論の範囲)) 詳細は [Session 1-3 追加](#session-1-3-追加-v010-012) と [v0.13.0 Session 4 追加](#v0130-session-4-追加-19-module--3-tier-構成) を参照

**v0.14.0 preview series (crates.io landing)**

- **preview.6** (2026-09-13) — `cargo-public-api` CI gate + fuzz coverage 5 → 7 (`fuzz_ccd` / `fuzz_trimesh`)
- **preview.5** (2026-09-13) — Phase 1 quick wins: `[package.metadata.docs.rs]` + README recommended feature table、example 7 → 10 (`ragdoll_demo` / `bfecc_advection_demo` / `sph_boundary_demo`)、fuzz 3 → 5 (`fuzz_joint` / `fuzz_cfd`)
- **preview.4** (2026-09-13) — 破損 `neural` / `replay` / `analytics` bridge feature を**一時削除** (sibling repo `alice-ml` / `alice-db` / `alice-analytics` が現時点 crates.io 未 publish、ALICE-SDF v1.7.7 pattern で bridge を綺麗に外し、v0.17.x での復帰予定を確保) bridge が必要な downstream は preview.4 直前の revision に `git` / `path` dep で pin
- **preview.3** (2026-09-13) — `#![deny(missing_docs)]` escalation (0 warning)、MSRV policy 明文化、clippy `approx_constant` cleanup、crates.io publish 事前調査
- **preview.1 / preview.2** (2026-09-12) — Physics v2 Priority 1/2 (Marching Tets、ALICE-Fluid の 3D Spectral IPM、spatial-hash SPH、Crank-Nicolson thermal、BFECC scalar / MAC-face BFECC velocity、nonlinear Crank-Nicolson、3D transient thermal、BiCGStab pressure、adaptive dt、edge-split refinement) — **+37 lib test、合計 1364**

**v0.12.0 の追加**: `GpuSolverBridge` に joint-solve パイプライン (`send_joints` / `send_body_rotations` / `dispatch_joint_solve_iteration`) を追加、`PhysicsWorld` が contact / joint solve の両方を装着済 bridge 経由に auto-route する。ALICE-TRT v3.1.0 の `FIX128_BALL_SOCKET_JOINT_SOLVE_WGSL` kernel と協調 (CPU `solve_ball_joint` と byte-exact 一致)。詳細は [CHANGELOG.md](CHANGELOG.md) 参照。

## v0.13.0 Session 4 追加 (19 module / 3 tier 構成)

v0.10-0.12 の engineering-solver 基盤の上に、game-physics 仕上げ (ragdoll / character state / netcode prediction / IK)、soft-body simulation (SDF SPH / SDF character / SDF FEM / SDF wind)、engineering research (composite failure / transient thermal / rolling contact fatigue / VIV / piezoelectric / acoustic / electromagnetic)、multi-material coupling (buoyancy zone / anisotropic friction / kinematic loop) をカバーする 19 module を追加 全て platform 跨ぎで bit-exact ([決定論の範囲](#決定論の範囲))、出典 formula は module doc に明記

### Tier ★★★ — 5 module (実運用 critical、downstream 直接依存)

| Module | 用途 |
|--------|------|
| `ragdoll` | 人型 ragdoll ビルダー (pose target / joint limit / breakable constraint) (G1) |
| `buoyancy_zone` | 境界付き 3-D 流体体積の浮力 + drag、fluid-surface CSF coupling (G2) |
| `laminate_failure` | Tsai-Wu / Tsai-Hill / Hashin / Puck の composite failure index (R1) |
| `transient_thermal` | 温度依存材質特性 + 1-D 非定常熱ソルバ + phase boundary tracking (R2) |
| ALICE-SDF `morphology` | 印刷 clearance 用の signed offset + tolerance fit check (S1、ALICE-SDF v1.7.7 で ship) |

### Tier ★★ — 7 module (広範な utility、ドメイン横断 glue)

| Module | 用途 |
|--------|------|
| `wind_zone` | 境界付き 3-D 風領域 (rigid / soft body への drag + lift) (G3) |
| `sdf_character` | SDF 地形対応キャラクターコントローラ (slope / step / air-time state) (S4) |
| `rolling_contact` | 転がり接触疲労 (Hertz + subsurface shear + Basquin S-N cycle counting) (R3) |
| `netcode_prediction` | クライアント予測 + 再和解 + input replay (G5) |
| `character_state` | キャラクター FSM (idle / walk / run / jump / fall / crouch) + 遷移検証 (G6) |
| `sdf_sph` | SDF 境界 SPH 流体 (density / viscosity / surface tension) (S3) |
| `kinematic_loop` | 閉ループ機構の kinematic body ループ制約ソルバ (R4) |

### Tier ★ — 8 module (専門分野 engineering / research)

| Module | 用途 |
|--------|------|
| `ik_physics_bridge` | joint-limit 対応 IK ソルバ (FABRIK / CCD backend) (G4) |
| `anisotropic_friction` | 方向依存摩擦係数 (rolling vs sliding、木目、布) (G7) |
| `aeroelasticity` | VIV (Vortex-Induced Vibration) + flutter + galloping (R5) |
| `piezoelectric` | センサー / アクチュエータ用 piezoelectric coupling (voltage ↔ strain) (R6) |
| `acoustic_wave` | 1-D / 2-D / 3-D 音波伝播 + 材質インピーダンス境界 (R7) |
| `electromagnetic` | 導体 body への電磁力場 (Lorentz / induction / eddy current) (R8) |
| `sdf_fem_mesh` | 大変形シミュレーション用 SDF → 四面体 FEM メッシュ生成 (S2) |
| `sdf_wind_field` | SDF 境界対応風場 (turbulence intensity + gust モデル) (S5) |

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
| **12種のジョイント** | Ball, Hinge, Fixed, Slider, Spring, D6, Cone-Twist, Pulley, Gear, Weld（破壊可能）, Rack-and-Pinion, Mouse |
| **レイキャスティング** | Sphere, AABB, Capsule, Planeに対するレイ・シェイプキャスト |
| **シェイプキャスト/オーバーラップ** | 球体キャスト、カプセルキャスト、オーバーラップ球体/AABB クエリ |
| **CCD** | 連続衝突検出（TOI、保守的前進法） |
| **スリープ/アイランド** | Union-Findアイランド管理による自動スリープ |
| **三角メッシュ** | BVH加速三角メッシュ衝突（Moller-Trumboreアルゴリズム） |
| **ハイトフィールド** | バイリニア補間によるグリッド地形 |
| **多関節体** | 多関節チェーン、ラグドール、ロボットアーム（FK伝播） |
| **フォースフィールド** | 風、重力井戸、ドラッグ、浮力、ボルテックス、爆発、磁気双極子 |
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
| **コーンコライダー** | GJK対応コーン形状（頂点+Y、底面-Y） |
| **楕円体コライダー** | 3軸独立半径の異方性サポート |
| **トーラスコライダー** | 主半径/副半径のミンコフスキー和分解 |
| **平面コライダー** | 無限平面（ヘッセ標準形）球体/AABB交差判定 |
| **くさびコライダー** | 三角柱（6頂点GJKサポート） |
| **凸包ビルダー** | 任意の点集合からインクリメンタル凸包構築 |
| **2D物理** | 完全な2Dサブシステム: SAT衝突、XPBDソルバー、Circle/Polygon/Capsule/Edge形状 |
| **質量特性** | 球体、ボックス、シリンダー、カプセル、凸包の慣性テンソル計算 |
| **衝突メッシュ生成** | マーチングキューブSDF→メッシュ変換（エッジ崩壊簡略化） |
| **シーンI/O** | バイナリ (.aphys) + JSONシーンシリアライズ（Fix128ビット精度往復） |
| **クロス-流体カップリング** | 双方向クロス-流体相互作用（ドラッグ、浮力、境界反発） |
| **ロープアタッチメント** | 剛体ロープ接続（コンプライアンス、破壊力設定） |
| **ソフトボディ切断** | 平面ベースの変形体・クロス切断 |
| **応力ヒートマップ** | 応力/温度/圧力の2Dスライス可視化（viridisカラーマップ） |
| **フロー可視化** | 流体速度場の矢印と流線生成 |
| **接触可視化** | 接触力矢印と摩擦コーンレンダリング |
| **マルチワールド** | 複数の独立した物理ワールドとボディ転送 |
| **パーティクルシステム** | 汎用エミッター、ライフタイム、フォースフィールド統合 |
| **no_std対応** | 組み込みシステム・WebAssemblyで動作 |

## Session 1-3 追加 (v0.10-0.12)

3 セッションの完全実装プッシュで **35 module + 3 統合 solver loop + 3
実行可能 example** を追加、「物理プリミティブ」から「実運用エンジニアリング
ソルバー」へのギャップを埋めました 全追加は platform 跨ぎで bit-exact ([決定論の範囲](#決定論の範囲) 参照) で、
出典 formula を明記 (Roark / Timoshenko / Simo & Hughes / Jones / Tsai-Wu /
Hill / Norton / Findley / WLF / Brackbill / Smagorinsky / Launder-Spalding /
Wilcox / Hasselmann / Turns / Anderson 等)

### Session 1 — 3D プリント安全性 + 剛性 Tier 1 (v0.10)

| Module | 用途 | Formula 出典 |
|--------|------|--------------|
| `filament_db` | 10 材料 property DB (Young's / yield / tensile / density / Tg / 異方性) | MatWeb / ASM Metals Handbook |
| `thin_wall` | SDF sphere marching による壁厚検出 | Bambu/Prusa 最小壁厚基準 |
| `beam_stress` | 断面 + 荷重ケース + Euler 座屈 + FoS | Roark's Formulas for Stress and Strain |
| `support_volume` | overhang → filament mm³ + 印刷時間見積 | Bambu Studio support manual |
| `anisotropic` | 9 定数 orthotropic + Hill + Tsai-Wu failure | Jones, *Mechanics of Composite Materials* |
| `plastic` | von Mises + isotropic/kinematic/combined hardening + Norton creep | Simo & Hughes; Norton (1929) |
| `buckling` | Johnson / Euler / plate / snap-through | Timoshenko & Gere; Bažant & Cedolin |
| `hyperelastic` | Neo-Hookean / Mooney-Rivlin / Yeoh | Ogden (1984); Yeoh (1990) |
| `bimaterial` | Timoshenko bimetal residual + Voigt/Reuss 境界 | Timoshenko (1925) |
| `layer_adhesion` | XY vs Z 6-成分実効強度 envelope | FDM 実測データ |
| `print_orientation` | 荷重方向最適化 + Euler grid search | 異方性変換 |
| `bridging` | 材料別最大 bridge distance check | Bambu/Prusa knowledge base |
| `warp_risk` | 冷却収縮 × footprint → Low/Medium/High/Critical | ALICE-Bamboo docs 事案校正済 |

### Session 2 — 剛性 Tier 2 + 流体 Tier 1-2 (v0.11-0.12)

| Module | 用途 | Formula 出典 |
|--------|------|--------------|
| `fatigue` | Basquin S-N + Miner 累積損傷 | Basquin (1910); Miner (1945) |
| `modal` | 1-DOF / 梁 / Warburton 板 / ねじり固有周波数 | Blevins; Warburton (1954) |
| `damping_rayleigh` | C = αM + βK + fit_two_modes | Clough & Penzien |
| `laminate` | Classical Laminate Theory ABD 行列 | Jones eq. 2.84 |
| `prestressed` | Motosh ボルト preload + parabolic cable pretension | Shigley; VDI 2230 |
| `fillet_stress` | Kirsch / Inglis / Peterson K_t 応力集中係数 | Peterson; Pilkey; Norton |
| `vibration_wall` | 薄壁共振 vs 6 印刷機振動 preset | Blevins; Bambu X1C spec |
| `thermal_stress` | 拘束応力 σ = c·E·α·ΔT + Tg 近傍警告 | Timoshenko & Goodier |
| `creep_longterm` | Findley 3-parameter + WLF time-temperature superposition | Findley (1989); Williams et al. (1955) |
| `non_newtonian` | Power-law / Carreau / Bingham / Herschel-Bulkley | Bird, Stewart, Lightfoot |
| `multiphase` | VOF advection + level set reinit + curvature | Hirt & Nichols (1981); Osher & Sethian (1988) |
| `compressible` | Ideal gas + Rankine-Hugoniot 衝撃波 + Riemann invariants | Anderson, *Modern Compressible Flow* |
| `eulerian_grid` | Staggered MAC + Jacobi/red-black GS 圧力射影 | Harlow & Welch (1965) |
| `turbulence` | Smagorinsky LES + k-ε + k-ω + wall function | Pope; Wilcox; Launder & Spalding |
| `surface_tension_csf` | Continuum Surface Force | Brackbill, Kothe & Zemach (1992) |
| `fsi_advanced` | 固体 ↔ 流体 drag / buoyancy / reaction | Peskin immersed boundary |
| `smoke_fire` | Arrhenius 反応 + soot + Boussinesq buoyancy | Turns; Kuo |
| `wave_ship` | JONSWAP spectrum + Froude-Krylov + 2-DOF | Hasselmann (1973); Faltinsen |
| `interface_capture` | Fast Sweeping FSM + PLIC (Rider-Kothe 解析解) | Zhao (2005); Youngs (1982) |

### Session 3 — Solver Loop + 12 改善 + 3 Demo (v0.12)

**統合 Solver Loop** — Session 1-2 module を組み合わせた 1 発 `step(dt)`
呼び出しで駆動:

| Solver | 統合先 |
|--------|--------|
| `cfd_solver::CfdSolver` | MAC + turbulence + non-Newtonian + level_set + CSF + Boussinesq + gravity |
| `structural_solver::StructuralSolver` | beam + plastic + creep + fatigue + buckling + history tracking |
| `print_pipeline_solver` | 10 印刷 safety check (warp + layer + orientation + thermal + beam + bridging + support + fillet + bimaterial) を one shot |

**12 改善** — 精度 / 性能 / 堅牢性:

- I1 `math_util` に `exp_fix` / `cbrt_fix` / `pow_int` / `clamp_fix` 共通化
- I2 `eulerian_grid` trilinear P2G / G2P (旧 nearest-cell)
- I3 `multiphase` semi-Lagrangian advection (旧 1 次 upwind)
- I4 PLIC Rider-Kothe 解析解 + cbrt (bisection 縮小)
- I5 FSM Godunov 3-neighbour quadratic Eikonal (旧 min + dx)
- I6 `safety::sdf_aabb` NaN/degenerate/min-dim guard
- I7 `safety_validate` に `thin_wall` + `layer_adhesion` + `thermal_stress` 統合
- I8 turbulence log-law wall function + `ln_fix`
- I9 圧力射影 red-black Gauss-Seidel (~2× 収束)
- I10 dynamic Smagorinsky Germano estimator
- I11 `laminate::compute_abd` rayon 並列 (feature-gated)
- I12 `fatigue::stress_at_cycles` Newton 早期終了

**3 実行可能 example** で solver を end-to-end 動作証明:

```bash
cargo run --example cfd_smoke_plume --release          # 12³ MAC grid で gravity settling
cargo run --example structural_pla_shelf_creep --release  # 20h PLA 棚 20N centre load
cargo run --example print_full_safety --release        # SKADIS 板 完全安全性 report
```

`cfd_smoke_plume` は `v_y = -g·t` を bit-exact に再現、圧力射影が grid
中央の divergence を 0 に保つ `print_full_safety` は 10 個の安全性 check
を実行し、300 × 300 × 5 mm PLA 板に対して現実的な UNSAFE 判定 (Warp
Critical / Beam FoS 1.16 / Fillet K_t 3.30) を返す — ALICE-Bamboo の
warp 事案で記録された failure mode と一致

### Mutation score (cargo-mutants、`quality-deep.yml`)

「test が通る」は「test が値を検証している」を意味しない core module は [cargo-mutants](https://mutants.rs) (`-- --lib`、test helper は `.cargo/mutants.toml` で除外) で測る 変異後にどれかの test が落ちれば *caught*、score = caught / (caught + missed)、unviable は除外 開始時 **32.0 %** (2026-09-15、16 shard 週次 run)、途中で本体 bug 6 件を検出 (`remove_body` の拘束付替え / `Fix128::atan` CORDIC shift / cloth bending 符号 / `LinearBvh::find_pairs` n² / joint 角補正の慣性分配 + 符号なし twist / EPA normal 符号)

| module | score | 測定 |
|---|---|---|
| `ccd` | **95.4 %** (208 / 218) | batch 8 後の local run |
| `joint` | **94.5 %** (358 / 379) | batch 8 後の scoped run 34969836733 |
| `collider` | **93.5 %** (145 / 155) | batch 8 後の local run |
| `math` | **91.3 %** (570 / 624、timeout 13) | scoped run 34960080446 |
| `contact_cache` | 88.0 % (66 / 75) | `3c12ee6` scoped run、batch 7 test 前 |
| `bvh` | **87.1 %** (210 / 241、timeout 10) | batch 8 後の scoped run 34969844505 |
| `solver` | **94.9 %** (default 軸の compiled code、485 / 511、残り 175 変異は `cfg(feature = "parallel" / "gpu-solver-bridge")` 側で同軸では非 compile、raw 70.7 %) / **82.4 %** (`parallel,gpu-solver-bridge` 軸、509 / 618、`cfg(not(feature = "parallel"))` 側 49 変異は非 compile、raw 76.3 %) | batch 8 後の scoped run 34969828635、両軸 |
| `solver_tgs` | 81.7 % (89 / 109) | `b187144` 週次 run、batch 7 test 前 |

「batch N 前」は最後に *測定* した値 以後に書いた test は列挙された miss を狙ったもの (等価変異は各 test module に記載) で、scoped `quality-deep` dispatch で再測定する 測定値だけを書く、timeout は caught にも missed にも数えない

### Test 数

Session 1 baseline (v0.9): 719 → Session 1 end (v0.10): 904 → Session 2
end (v0.11): 1170 → **Session 3 end (v0.12): alice-physics 1175 test +
alice-bamboo 53 統合 test、全 pass**

### サブステッピング TGS ソルバー（プレビュー）

具体的な剛体表現から独立した、trait 抽象のサブステッピング TGS ソルバー基盤群:

- **`TgsHooks` ドライバ** — `begin_substep` / `velocity_iteration` / `position_iteration` / `end_substep` の 4 コールバック、フレームを `N` サブステップに分割することで、質量比の大きい積み上げ剛体でも安定にシミュレート
- **インパルスウォームスタート** — `ImpulseCache` によりフレーム間で接触点ごとの適用インパルスを記憶、静止しているパイル向けの PGS 収束を大幅に高速化
- **連結成分アイランド** — union-find による `build_islands` が接触やジョイントで結合された剛体を disjoint な island に分割
- **6-DOF リファレンスフック** — Baumgarte による位置補正 + Coulomb 摩擦 (`√(τ₁² + τ₂²) ≤ μ · N_acc` cone clamp) + Newton の反発係数を実装した projected Gauss-Seidel
- **アイランド単位スコープソルブ** — body slice の分割 + world→local index remap により、per-island 呼び出しで重力が重複積算されない
- **rayon per-island 並列化** — Fix128 演算と canonical island ordering により、serial 版と byte-identical (`parallel_matches_serial_bit_perfect` test で証明)
- **クォータニオン姿勢積分** — `q_new = normalize(q + 0.5 · dt · (ω × q))` プリミティブによる、diagonal-inertia リファレンスフック上への完全 6-DOF シミュレーション積層用

rayon 並列版は `--features parallel` で有効化。全バリアントで Fix128 の byte-identical 決定性を維持。

### 高度なソルバー基盤（Turn D / Phase E / Phase F）

サブステッピング TGS コアの上に積み上げた追加ビルディングブロック:

- **適応的サブステッピング** — `AdaptiveSubStepConfig` + `HasVelocity` + `adaptive_substeps_for` により、最速 body の L∞ 速度から決定論的なサブステップ数を計算（Fix128 pure、closed-form、除算なし — lockstep / rollback で安全）
- **CCD 最適化適応的サブステッピング** — `adaptive_substeps_for_ccd` は per-step 変位上限を `最小 collider 半径 × 安全係数` に絞り、高速 body が薄い壁を通り抜けないことを保証
- **適応的サブ-TOI** — `adaptive_toi_substeps` は適応的サブステッピングと既存の `speculative_contact` TOI を連携するペア別 CCD の bridge、collision course pair は即 `max_substeps` を要求（Phase F 11.1、body 完成）
- **`ImpulseCache` 観測性** — `hits/misses/stats/hit_rate/reset_stats` によりウォームスタートの有効性を可視化、シーンが安定（高 hit rate）か churn 状態（低 hit rate）かを判別
- **アイランド別スコープ oriented ソルブ** — `solver_tgs_hooks_6dof_oriented_scoped` は完全 6-DOF oriented hook を island 単位で分離、重力の重複積算を防ぎ serial-parallel の bit-perfect 一致を保証
- **`LinearBvh::refit_leaves`（bottom-up）** — flat tree 構造を保持したまま leaf AABB を requantise、internal node へ i32 domain で union を伝播（CORDIC / rounding 依存なし）、完全 rebuild の代わりに `O(N)` refit を提供
- **`BroadphaseHybrid`** — 動的 body 用 `SpatialGrid` (hash grid) と静的 body 用 `LinearBvh` の 2 層構造（Turn D 5' 案）、per-frame broad-phase コストを `O(N_total log N_total)` から `O(N_dynamic + log N_static)` に削減。推奨 per-frame flow: `clear_dynamic` → `insert_dynamic` × N → `build_dynamic` → `query_pairs`
- **`FeatherstoneSolver::solve_with_mass_splitting`** — O(n) forward-dynamics ソルバーに mass-ratio 分割 threshold を追加（heavy / light > threshold → `dt / 2` × 2 recursion）、極端な質量比スタックの stiffness 由来 ill-conditioning を緩和。決定論は lookup 不要な乗算比較 + `Fix128::half` で保持（Phase F 11.2、body + 3 tests 完成）
- **FFI byte-for-byte 決定性** — `AliceVec3Fix128Raw` + `alice_physics_body_get_position_fix128_raw` により Fix128 hi/lo ペアを C ABI 経由で公開。`ffi_contract_gravity_fall_deterministic` は bracket 検証 + world 再構築リプレイの hi/lo 完全一致を assert、Unity / UE5 host が bit パターン一致を検証可能（Phase F 11.3、runtime determinism gate 完成）

- **`GpuSolverBridge` トレイト** (`--features gpu-solver-bridge`) — 外部 GPU オフロードバックエンド (例: ALICE-TRT `TrtSolverAdapter`) 用の opt-in 拡張面。`DiffFixture` / `GpuDivergence` 型を提供し、実装は CPU 側 solver との byte-for-byte 等価性を certify するまで runtime に受理されない。default build は CPU-native TGS pipeline のみを唯一のコードパスとして維持

#### コンパニオンクレート — ALICE-TRT（GPU ソルバーオフロード）

[ALICE-TRT v1.0.0+](https://github.com/ext-sakamoro/ALICE-TRT) の `--features physics-solver` とペア。リファレンス実装として `TrtSolverAdapter` を提供します。ALICE-TRT 側で `physics-solver` を有効化すると本クレートの `gpu-solver-bridge` も自動的に enable されるので、feature 行 1 本で両側カバーできます。**ALICE-TRT v1.x は semver 安定** — 公開 adapter 表面 (`send_island` / `dispatch_iterations` / `set_gravity` / `set_floor` / `push_distance_constraint` / `recv_island` / `assert_bit_exact_vs_cpu`) は v1.x 系で bytes-stable。

```toml
[dependencies]
# Pre-release identifier `-preview.N` は v0.14.0 stable までは version 完全一致必須
# (Cargo は pre-release 系列を横断して auto-upgrade しない)
alice-physics = { version = "0.14.0-preview.6", features = ["gpu-solver-bridge"] }
alice-trt     = { version = "3.1", features = ["physics-solver"] }
```

```rust
use alice_physics::gpu_bridge::{DiffFixture, GpuSolverBridge};
use alice_physics::math::Fix128;
use alice_trt::{GpuDevice, TrtSolverAdapter};

// Live PGS ディスパッチ (integrate + 重力 + 床 + N 個距離制約)
let device = GpuDevice::new()?;
let mut adapter = TrtSolverAdapter::new(&device);

adapter.send_island(&positions, &velocities);
adapter.set_gravity(Some([Fix128::ZERO, Fix128::from_ratio(-98, 10), Fix128::ZERO]));
adapter.set_floor(Some(Fix128::ZERO));
adapter.push_distance_constraint(0, 1, Fix128::from_int(2));   // Gauss-Seidel 順
adapter.push_distance_constraint(1, 2, Fix128::from_int(2));

adapter.dispatch_iterations(10, Fix128::from_ratio(1, 60));    // 実 GPU dispatch、no-op ではない
adapter.recv_island(&mut positions, &mut velocities);

// production gate: CPU 側 solver と byte-for-byte 等価を検証
adapter.assert_bit_exact_vs_cpu(&DiffFixture {
    description: "3body_triangle_10iter",
    tolerance: Fix128::ZERO,
})?;
```

対応リリース: [ALICE-TRT v3.1.0](https://github.com/ext-sakamoro/ALICE-TRT/releases/tag/v3.1.0) (alice-physics v0.12.0 と協調 joint-solve GPU offload リリース)。全 ALICE-TRT リリースを macOS (Metal) / Ubuntu (Vulkan lavapipe) / Windows (DX12 WARP) 3 プラットフォームで 37 Fix128 単体テスト + 170 physics-solver テスト、CPU golden との byte-exact 検証済み。

これらのプリミティブの決定論保証ガードレールは [`deterministic-physics-lockstep-discipline`](https://github.com/ext-sakamoro/claude-config/blob/main/claude-skills/deterministic-physics-lockstep-discipline/SKILL.md) スキル（private reference）に集約されています。

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
| **L6: コード品質** | 20/20 | 1730 lib テスト + 解析解 oracle 13 本 + engineering oracle 105 本 (thermal/fatigue 5 + solid 34 + fluid 39 + misc 27) + default config oracle 13 本 + 53 alice-bamboo 統合テスト + 8 fuzz target + 44 決定論テスト、clippy `-D warnings` (default + 全 native feature set、all targets)、MSRV 1.85 CI job (default / no_std / native)、`#![deny(missing_docs)]`、cargo-semver-checks hard-gate |
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

- **1730 lib テスト** (alice-physics crate、Session 4 + v0.14.0 preview 1/2 + v1.0.1 coloring / island / Fix128 + 1.2.0 mutation batch 1-8 / 解析解 oracle)
- **53 alice-bamboo 統合テスト** (3D プリント安全性のエンドツーエンド)
- **7 fuzz target** (`fuzz_step` / `fuzz_collision` / `fuzz_deterministic_roundtrip` / `fuzz_joint` / `fuzz_cfd` / `fuzz_ccd` / `fuzz_trimesh`)
- **合計: 2036 テストパス** (`cargo test` default feature: 1730 lib + 44 決定論 (Fix128 golden 9 + f32 golden 13 + semantic 22) + 75 統合 + 21 doctest、全 native feature set では lib 1426)、clippy: `-D warnings` で 0 警告 (default + `std,simd,parallel,ffi,gpu-solver-bridge,neural,replay,analytics`、`--all-targets`)、`#![deny(missing_docs)]` (0 warning)

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

ALICE-Physics は全 module で platform 跨ぎの **ビット精度の結果** を保証し ([決定論の範囲](#決定論の範囲) 参照)、以下を実現します：

- **ロックステップマルチプレイ**: 全クライアントが同一のシミュレーションを計算
- **ロールバックネットコード**: 入力を決定論的に再生
- **リプレイシステム**: ゲームセッションの完全な再現
- **分散シミュレーション**: 一貫した結果による並列計算

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALICE-Physics v0.14.0-preview.6                     │
│         140 pub mod (preview.4 bridge 削除後)、1364 lib テスト                 │
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
│  AAAエンジンレイヤー                                                          │
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
│  衝突形状レイヤー (v0.5.0)                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  cone    │ │ellipsoid │ │  torus   │ │plane_col │ │  wedge   │          │
│  │ Apex/Base│ │ 3-Axis   │ │ Maj/Min  │ │ Hessian  │ │ TriPrism │          │
│  │ GJK Supp│ │ Aniso    │ │ Minkowsk │ │ Sph/AABB │ │ 6-Vertex │          │
│  │ Inertia  │ │ Support  │ │ Support  │ │ Intersct │ │ GJK Supp│          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐                                                               │
│  │cvx_mesh  │                                                               │
│  │ Incr Hull│                                                               │
│  │ Tetra    │                                                               │
│  │ Horizon  │                                                               │
│  └──────────┘                                                               │
│                                                                              │
│  拘束・ダイナミクスレイヤー                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  joint   │ │joint_extr│ │  motor   │ │articulatn│ │  force   │          │
│  │ Ball     │ │ Pulley   │ │ PD 1D/3D │ │ Ragdoll  │ │ Wind     │          │
│  │ Hinge    │ │ Gear     │ │ Position │ │ FK Chain │ │ Gravity  │          │
│  │ Fixed    │ │ Weld     │ │ Velocity │ │ Robotic  │ │ Buoyancy │          │
│  │ Slider   │ │ Rack&Pin │ │ Max Torq │ │ Feather- │ │ Drag     │          │
│  │ Spring   │ │ Mouse    │ │          │ │  stone   │ │ Vortex   │          │
│  │ D6       │ │ Breakabl │ │          │ │ 12-body  │ │ Explosion│          │
│  │ ConeTwst │ │          │ │          │ │          │ │ Magnetic │          │
│  │ Breakable│ │          │ │          │ │          │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐                                                               │
│  │ sleeping │                                                               │
│  │ Islands  │                                                               │
│  │ Union-   │                                                               │
│  │  Find    │                                                               │
│  └──────────┘                                                               │
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
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │clth_flud │ │rope_atth │ │soft_cut  │ │ particle │                       │
│  │ Drag     │ │ RigidBdy │ │ Plane    │ │ Emitters │                       │
│  │ Buoyancy │ │ Complnce │ │ Deform   │ │ Lifetime │                       │
│  │ Boundary │ │ Break    │ │ Cloth    │ │ ForceInt │                       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
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
│  2D物理サブシステム (v0.5.0)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ physics2d                                                           │    │
│  │ Vec2Fix, Shape2D (Circle/Polygon/Capsule/Edge), RigidBody2D         │    │
│  │ SAT + Voronoi collision, XPBD 2D solver, Joint2D (Rev/Dist/Weld/M) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  メッシュ & I/O ユーティリティ (v0.5.0)                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │mass_prop │ │col_mesh  │ │ scene_io │ │multi_wrld│                       │
│  │ Sphere   │ │ Marching │ │ Binary   │ │ N worlds │                       │
│  │ Box      │ │ Cubes    │ │ APHYS    │ │ Transfer │                       │
│  │ Cylinder │ │ EdgeClps │ │ JSON     │ │ Isolate  │                       │
│  │ PAxis    │ │ Simplify │ │ Fix128   │ │ Step All │                       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
│                                                                              │
│  可視化レイヤー (v0.5.0)                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │ heatmap  │ │ flow_viz │ │contct_viz│                                    │
│  │ Stress   │ │ Arrows   │ │ Force    │                                    │
│  │ Temp     │ │ Streamln │ │ Friction │                                    │
│  │ Viridis  │ │ Velocity │ │ Cone     │                                    │
│  └──────────┘ └──────────┘ └──────────┘                                    │
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
│  アナリティクス & プライバシーレイヤー                                          │
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

### パスを選ぶ

役割ごとに最適な入り口を選んでください。全パスは同じ決定論的 `PhysicsWorld` を共有するので、シンプルに始めて後から rollback / GPU オフロード / joint を積み増しできます。

| あなたは… | パス | 用途 | セクション |
|-----------|------|-----|-----------|
| **初めて触る** | [Hello, first physics](#hello-first-physics-30-秒) ↓ | インストール確認、落下する球体 1 個 | ↓ |
| **物理シミュ入門** | [基本シミュレーション](#基本シミュレーション) | 剛体・重力・地面・毎フレームステップ | ↓ |
| **ネットコード / lockstep ゲーム** | [ロールバック ネットコード](#ロールバック-ネットコード) | Snapshot 保存/読込、決定論的リプレイ、Fix128 bit-exact | ↓ |
| **ロープ / cloth / 制約** | [距離制約](#距離制約-ロープチェーン) + `joint` モジュール | チェーン、ロープ、ragdoll、破断可能制約 | ↓ |
| **多数の body を捌く** | [BVH ブロードフェーズ](#bvhブロードフェーズ衝突) | 空間加速、zero-alloc クエリ、Morton | ↓ |
| **GPU オフロード (上級)** | [GPU Solver Bridge](#コンパニオンクレート--alice-trtgpu-ソルバーオフロード) | ALICE-TRT 経由の GPU PGS、CPU と byte-exact | ↑ |
| **Unity / UE5 / Godot 統合** | C-ABI FFI | Fix128 hi/lo ペアを C ABI で公開、bit パターン一致 | `alice_physics_ffi::*` |
| **CCD / raycast / trimesh** | モジュール個別 | `ccd` / `raycast` / `trimesh` 専用 API | [モジュール](#モジュール) 参照 |

### Hello, First Physics (30 秒)

最小の動くサンプル — 静的地面に球体 1 個を落として y 座標を毎フレーム出力:

```rust
use alice_physics::prelude::*;

let mut world = PhysicsWorld::new(PhysicsConfig::default());

let sphere = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
let _sphere_id = world.add_body(sphere);
world.add_body(RigidBody::new_static(Vec3Fix::ZERO));   // y=0 の地面

let dt = Fix128::from_ratio(1, 60);
for frame in 0..60 {
    world.step(dt);
    println!("frame {frame}: y = {}", world.bodies[0].position.y.hi);
}
```

これが動いたら、表の中から自分の役割に合ったパスに進んでください。

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

### よく使うレシピ

繰り返し出てくるパターンをコピペしやすい形で。

**1. 世界に対する raycast**

```rust
use alice_physics::prelude::*;

let ray = Ray {
    origin:    Vec3Fix::from_int(0, 5, 0),
    direction: Vec3Fix::from_int(0, -1, 0),   // 真下
    max_distance: Fix128::from_int(100),
};
if let Some(hit) = world.raycast(&ray) {
    println!("body {} に距離 {} で hit", hit.body_id, hit.distance.hi);
}
```

**2. 停止した body を sleep させる (性能)**

```rust
// PhysicsConfig は sleeping デフォルト有効、閾値だけ調整
let mut config = PhysicsConfig::default();
config.sleeping.linear_threshold  = Fix128::from_ratio(1, 100);   // 0.01 m/s
config.sleeping.angular_threshold = Fix128::from_ratio(1, 100);
config.sleeping.time_to_sleep     = Fix128::from_ratio(1, 2);     // 0.5 秒
let mut world = PhysicsWorld::new(config);
```

**3. 破断する制約 (負荷でロープが千切れる)**

```rust
let constraint = DistanceConstraint {
    body_a: a,
    body_b: b,
    local_anchor_a: Vec3Fix::ZERO,
    local_anchor_b: Vec3Fix::ZERO,
    target_distance: Fix128::from_int(2),
    compliance:      Fix128::from_ratio(1, 1000),
};
let id = world.add_distance_constraint(constraint);
world.set_constraint_break_force(id, Fix128::from_int(500));   // > 500 N で破断
```

**4. 静的地面のクイックセットアップ**

```rust
let ground = RigidBody::new_static(Vec3Fix::ZERO);
let ground_id = world.add_body(ground);
world.set_collider(ground_id, Collider::plane(Vec3Fix::from_int(0, 1, 0), Fix128::ZERO));
```

**5. Snapshot 保存 + 再読込 (rollback プリミティブ)**

```rust
let snapshot: Vec<u8> = world.serialize_state();     // arch 越しに byte-exact
// ... 後で、または別マシンで:
world.deserialize_state(&snapshot);                  // frame 境界から再開
```

**6. Continuous collision (高速射出物)**

```rust
use alice_physics::ccd;

let bullet = RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE);
let bullet_id = world.add_body(bullet);
world.enable_ccd(bullet_id);   // sub-sweep 判定、貫通防止
```

### Session 3 Solver Loop 例 (v0.12)

**CFD gravity settling** — 非圧縮 Navier-Stokes を圧力射影付きで 1 step:

```rust
use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::Fix128;

let mut solver = CfdSolver::new(12, 12, 12, Fix128::from_ratio(1, 10));
solver.jacobi_iterations = 100;

// 閉じた箱の中の水、重力で settling → divergence ≈ 0
for _ in 0..30 {
    solver.step(Fix128::from_ratio(1, 1000));  // 0.001 s / step
}
// solver.grid.v[…] 中心 column は t = 0.03 s で −0.294 m/s
// 自由落下解 −g·t と bit-exact 一致
```

**Structural creep + fatigue 履歴** — PLA 棚に 20 h 継続荷重 @ 55°C の
診断 table:

```rust
use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

let section = CrossSection::Rectangular {
    width_mm: Fix128::from_int(150),
    height_mm: Fix128::from_int(10),
};
let load = LoadCase::SimplySupportedCenter {
    load_n: Fix128::from_int(20),
    length_mm: Fix128::from_int(300),
};
let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
solver.operating_temp_c = Fix128::from_int(55);
solver.dt_s = Fix128::from_int(3600);   // 1 時間 step

let history = solver.run(20);
println!(
    "creep {:.5}, fatigue D = {:.4}, failure step = {:?}",
    history.plastic_state.creep_strain.to_f32(),
    history.fatigue_damage.to_f32(),
    history.failure_step,
);
```

**Print pipeline safety** — 10 印刷 safety check を 1 call:

```rust
use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::math::Fix128;
use alice_physics::print_orientation::LoadDirection;
use alice_physics::print_pipeline_solver::{analyze_print_pipeline, PrintPipelineInputs};
use alice_physics::support_volume::OverhangRegion;
use alice_physics::warp_risk::Footprint;

let footprint = Footprint {
    area_mm2: Fix128::from_int(300 * 300),
    max_dimension_mm: Fix128::from_int(300),
};
let inputs = PrintPipelineInputs {
    beam_load: Some((
        CrossSection::Rectangular {
            width_mm: Fix128::from_int(50),
            height_mm: Fix128::from_int(5),
        },
        LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(30),
            length_mm: Fix128::from_int(300),
        },
    )),
    load_direction: Some(LoadDirection::axis_z()),
    overhangs: vec![OverhangRegion {
        projected_area_mm2: Fix128::from_int(50 * 50),
        support_height_mm: Fix128::from_int(15),
    }],
    fillet: Some((Fix128::from_ratio(3, 10), Fix128::from_int(20), Fix128::from_int(40))),
    ..Default::default()
};

let report = analyze_print_pipeline(footprint, "PLA", &inputs);
report.print();
// warp / strength envelope / thermal / beam FoS / bridging /
// support volume / K_t / bimaterial + 総合 SAFE / UNSAFE 判定を出力
```

### 次に見るべきドキュメント

| やりたいこと | 参照先 |
|------------|-------|
| 型表面を全体把握 | 下の [モジュール](#モジュール) セクション |
| 決定性保証を深掘り | [決定論的物理エンジンが必要な理由](#決定論的物理エンジンが必要な理由) |
| PGS を GPU にオフロード | [コンパニオンクレート — ALICE-TRT](#コンパニオンクレート--alice-trtgpu-ソルバーオフロード) |
| Unity / UE5 ネイティブプラグイン統合 | `alice_physics_ffi::*` (本 crate 内モジュール) |
| Joint カタログ (12 基本 + 5 応用) | [`joint`](#joint---12-種類のジョイント--breakable-constraints) / [`joint_extra`](#joint_extra---5種類の高度なジョイントv050) セクション |
| 3 プラットフォーム CI byte-exact 保証 | ALICE-TRT の Fix128 + physics-solver test マトリクス (companion release) |
| Rapier / PhysX / Box2D と比較 | [決定論的物理エンジンが必要な理由](#決定論的物理エンジンが必要な理由) ベンチマーク表 |

`PhysicsWorld` に直接プラグインできるコンパニオンクレート:

- **[ALICE-TRT](https://github.com/ext-sakamoro/ALICE-TRT)** (`gpu-solver-bridge` feature) — GPU PGS ソルバーオフロード、CPU と byte-exact
- **[ALICE-Cloth](https://github.com/ext-sakamoro/ALICE-Cloth)** — 同じ Fix128 プリミティブ上の XPBD cloth
- **[ALICE-Vehicle](https://github.com/ext-sakamoro/ALICE-Vehicle)** — 決定論的な車両リグ
- **[ALICE-Netcode](https://github.com/ext-sakamoro/ALICE-Netcode)** — `serialize_state` / `deserialize_state` に載せた rollback ネットコード配管

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

### `joint` - 12種のジョイント + 破壊可能拘束

| 型 | 説明 |
|----|------|
| `BallJoint` | 球面ジョイント（回転3自由度） |
| `HingeJoint` | ヒンジジョイント（回転1自由度、角度制限付き） |
| `FixedJoint` | 固定ジョイント（0自由度） |
| `SliderJoint` | スライダージョイント（並進1自由度、制限付き） |
| `SpringJoint` | 減衰スプリング拘束 |
| `D6Joint` | 6自由度設定可能ジョイント（軸ごとにロック/フリー/リミット） |
| `ConeTwistJoint` | コーンスイング制限+ツイスト制限付きボールジョイント |

### `joint_extra` - 5種の高度なジョイント (v0.5.0)

| 型 | 説明 |
|----|------|
| `PulleyJoint` | 2アンカー+ロープ比率付きプーリー（総距離拘束） |
| `GearJoint` | ギアカップリング（2ボディ間の角速度比率） |
| `WeldJoint` | `break_force` / `break_torque` 閾値付き剛体溶接 |
| `RackAndPinionJoint` | ピッチ半径による回転-並進変換 |
| `MouseJoint` | マウス/タッチドラッグ操作用ターゲット追従ジョイント |

全ジョイントは `with_break_force(max_force)` で**破壊可能拘束**をサポート。拘束力が閾値を超えるとジョイントが破壊されシミュレーションから除去されます。

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

// コーンツイストジョイント（肩のような関節: コーン+ツイスト制限）
let cone = ConeTwistJoint::new(body_a, body_b, anchor_a, anchor_b, twist_axis)
    .with_cone_limit(Fix128::from_ratio(45, 1))  // 45度コーン
    .with_twist_limit(Fix128::from_ratio(30, 1)); // 30度ツイスト

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
| `force` | 風、重力井戸、ドラッグ、浮力、ボルテックス、爆発、磁気双極子 |
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

### `cone` - コーンコライダー (v0.5.0)

| 型 | 説明 |
|----|------|
| `Cone` | 頂点+Y、底面-Yのコーンコライダー |

**機能:**
- GJK `Support` トレイト実装（頂点/底面中心切り替え）
- AABB計算、体積、表面積、慣性テンソル対角
- 半径と半高さ+回転の設定可能

### `ellipsoid` - 楕円体コライダー (v0.5.0)

| 型 | 説明 |
|----|------|
| `Ellipsoid` | 3軸独立半径(rx, ry, rz)の楕円体 |

**機能:**
- 異方性GJKサポート（方向を半径でスケール、正規化、スケールバック）
- 体積、AABB、慣性テンソル対角の計算

### `torus` - トーラスコライダー (v0.5.0)

| 型 | 説明 |
|----|------|
| `Torus` | 主半径（リング）と副半径（チューブ）のトーラス |

**機能:**
- ミンコフスキー和分解によるGJKサポート
- 体積とAABBの計算

### `plane_collider` - 無限平面コライダー (v0.5.0)

| 型 | 説明 |
|----|------|
| `PlaneCollider` | ヘッセ標準形（法線+オフセット）の無限平面 |

**機能:**
- `intersect_sphere()` -- 球体-平面交差判定（接触点付き）
- `intersect_aabb()` -- AABB-平面オーバーラップテスト
- `signed_distance()` -- 点-平面符号付き距離

### `wedge` - くさびコライダー (v0.5.0)

| 型 | 説明 |
|----|------|
| `Wedge` | 6頂点の三角柱（くさび） |

**機能:**
- GJKサポート（6頂点の最大ドット積反復）
- 体積、AABB、重心の計算

### `convex_mesh_builder` - 凸包ビルダー (v0.5.0)

| 関数 | 説明 |
|------|------|
| `build_convex_hull(points)` | 任意の点集合から凸包を構築 |
| `compute_centroid(faces, vertices)` | 凸包の重心を計算 |

**アルゴリズム:** インクリメンタル凸包 -- 初期四面体を見つけ、点を順次挿入、可視面を除去、ホライズンエッジでパッチ。

### `physics2d` - 2D物理エンジン (v0.5.0)

| 型 | 説明 |
|----|------|
| `Vec2Fix` | Fix128成分の2Dベクトル |
| `Shape2D` | Circle, Polygon（最大16頂点）, Capsule, Edge |
| `RigidBody2D` | 位置、角度、速度、角速度の2D剛体 |
| `PhysicsWorld2D` | XPBDソルバー付き完全な2D物理ワールド |
| `PhysicsConfig2D` | 重力、サブステップ、反復、スリープ閾値 |
| `Contact2D` | 法線、深度、ボディインデックス付き接触点 |
| `Joint2D` | Revolute, Distance, Weld, Mouse ジョイントバリアント |
| `BodyType2D` | Dynamic, Static, Kinematic |

**アルゴリズム:**
- **SAT**（分離軸定理）+ Voronoi領域分類
- **XPBD** 位置ベースソルバー（設定可能なサブステップ）
- ブロードフェーズAABBオーバーラップ、ナローフェーズポリゴン射影

### `mass_properties` - 質量 & 慣性 (v0.5.0)

| 関数 | 説明 |
|------|------|
| `sphere_mass_properties(radius, density)` | 球体の質量+慣性テンソル |
| `box_mass_properties(half_extents, density)` | ボックスの質量+慣性テンソル |
| `cylinder_mass_properties(radius, half_h, density)` | シリンダーの質量+慣性 |
| `capsule_mass_properties(radius, half_h, density)` | カプセルの質量+慣性 |
| `convex_hull_mass_properties(vertices, faces, density)` | 凸包の質量+慣性 |
| `translate_inertia(inertia, mass, offset)` | 平行軸の定理 |

### `collision_mesh_gen` - SDF→メッシュ (v0.5.0)

| 型 / 関数 | 説明 |
|-----------|------|
| `CollisionMeshConfig` | グリッド解像度、バウンディングボックス、簡略化ターゲット |
| `CollisionMesh` | 頂点+三角形インデックス出力 |
| `generate_collision_mesh(sdf, config)` | マーチングキューブメッシュ生成 |
| `simplify_mesh(mesh, target)` | エッジ崩壊メッシュ簡略化 |

### `scene_io` - シーンシリアライズ (v0.5.0, std)

| 関数 | 説明 |
|------|------|
| `save_binary(path, scene)` | `.aphys` バイナリ形式で保存（ビット精度Fix128） |
| `load_binary(path)` | `.aphys` バイナリ形式から読み込み |
| `save_json(path, scene)` | `.aphys.json` 人間可読形式で保存 |
| `load_json(path)` | `.aphys.json` 形式から読み込み |

**バイナリ形式:** マジック `APHYS\0`、リトルエンディアン、生の Fix128 `{hi: i64, lo: u64}` でビット精度決定論的往復。

### `cloth_fluid` - クロス-流体カップリング (v0.5.0)

| 型 | 説明 |
|----|------|
| `ClothFluidCoupling` | 双方向クロス-流体相互作用 |

**効果:** クロス頂点への流体ドラッグ力、水没に基づく浮力、貫通防止の境界反発。

### `rope_attach` - ロープアタッチメント (v0.5.0)

| 型 | 説明 |
|----|------|
| `RopeAttachment` | ロープ端点を剛体に接続 |

**機能:** 設定可能なコンプライアンス、破壊力閾値、自動デタッチメント。

### `soft_body_cut` - ソフトボディ切断 (v0.5.0)

| 関数 | 説明 |
|------|------|
| `cut_deformable(body, plane)` | 平面に沿って変形体を切断 |
| `cut_cloth(cloth, plane)` | 平面に沿ってクロスメッシュを切断 |

**アルゴリズム:** 平面ベースのトポロジー分割 -- 頂点を分類、交差点に新頂点を生成、接続性を再構築。

### `heatmap` - 応力/温度ヒートマップ (v0.5.0)

| 型 | 説明 |
|----|------|
| `Heatmap` | カラーマップ付き2Dスカラーフィールド可視化 |
| `HeatmapConfig` | 解像度、値範囲、カラーマップ選択 |

**機能:** Viridisカラーマップ、RGBAピクセル出力、設定可能な最小/最大範囲。

### `flow_viz` - フロー可視化 (v0.5.0)

| 型 | 説明 |
|----|------|
| `FlowArrow` | 位置、方向、大きさ付き速度矢印 |
| `Streamline` | フローパスを辿る順序付き点リスト |

**関数:** 速度場から `generate_flow_arrows()`、`generate_streamlines()`。

### `contact_viz` - 接触可視化 (v0.5.0)

| 型 | 説明 |
|----|------|
| `ContactArrow` | 接触点の力矢印 |
| `FrictionCone` | 摩擦限界を表すコーンジオメトリ |

**関数:** 接触マニフォールドから `visualize_contacts()`、`visualize_friction_cones()`。

### `multi_world` - 複数物理ワールド (v0.5.0)

| 型 | 説明 |
|----|------|
| `MultiWorld` | N個の独立した物理ワールドのコンテナ |

**機能:**
- `add_world()` / `remove_world()` -- 独立ワールドの管理
- `step_all(dt)` -- 全ワールドを同時に進行
- `transfer_body(from, to, body_id)` -- ワールド間のボディ移動

### `particle` - パーティクルシステム (v0.5.0)

| 型 | 説明 |
|----|------|
| `ParticleSystem` | 汎用パーティクルシミュレーション |
| `ParticleEmitter` | コーンスプレッド付き方向性エミッター |
| `Particle` | 位置、速度、年齢、ライフタイム、質量 |

**機能:**
- 設定可能なエミッションレート、スプレッド角、速度範囲
- 自動リサイクル付きライフタイム管理
- 完全な `ForceField` 統合（爆発と磁気を含む）
- `DeterministicRng` による決定論的動作

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

## レンダリングパイプライン連携（ALICE-SDF v1.4.0+）

ALICE-Physicsの物理状態データは、ALICE-SDFのレンダリングパイプラインの破壊unifromを駆動します。物理エンジンが熱・破砕・浸食・圧力フィールドをシミュレーションし、アプリケーション層がこれをサンプリングしてシェーダーuniformにアップロードすることで、リアルタイムの視覚フィードバックを実現します。

### アーキテクチャ

```
ALICE-Physics (CPU)                    ALICE-SDF シェーダー (GPU)
┌─────────────────────┐                ┌──────────────────────┐
│ ThermalModifier      │──→ sample ──→│ uniform float uShatter│
│ FractureModifier     │──→ sample ──→│ uniform float uEntropy│
│ PressureModifier     │──→ sample ──→│ uniform float uImpact │
│ ErosionModifier      │──→ sample ──→│ uniform vec2 uShake   │
│ ForceField::Explosion│──→ sample ──→│ uniform float uMeteorY│
└─────────────────────┘                └──────────────────────┘
         ↓ アプリケーション層がフレーム毎にブリッジ
```

### Uniformマッピング

| 物理ソース | フィールド / メソッド | レンダリングUniform | 説明 |
|-----------|---------------------|-------------------|------|
| `ThermalModifier` | `temperature.sample(pos)` | `uShatter` | 融点超過温度 → 破壊強度 |
| `ThermalModifier` | `melt_accumulator.sample(pos)` | `uEntropy` | 累積材料損失量 (0-1) |
| `FractureModifier` | `stress_field.sample(pos)` | `uShatter` | 応力駆動のクラック伝播 |
| `PressureModifier` | `pressure_field.sample(pos)` | `uImpact` | 接触力による変形深度 |
| `ForceField::Explosion` | `center`, `radius` | `uMeteorImpact`, `uImpactRing` | 爆発中心と爆風半径 |
| 衝撃イベント | 衝突速度 | `uShake` | 衝撃力によるカメラシェイク |
| 投射体ボディ | `body.position.y` | `uMeteorY` | 落下物体の高度 |

### 使用例：物理駆動の破壊演出

```rust
use alice_physics::{PhysicsWorld, ThermalModifier, ThermalConfig, HeatSource};
use alice_sdf::compiled::glsl::RenderConfig;

// 物理側: 熱シミュレーション
let mut thermal = ThermalModifier::new(ThermalConfig {
    melt_temperature: 800.0,
    melt_rate: 0.01,
    droop_strength: 0.5,
    ..Default::default()
});
thermal.add_heat_source(HeatSource::point(impact_pos, 2000.0));

// 毎フレーム:
thermal.step(&mut world, dt);

// アプリケーションブリッジ: 物理をサンプリング → シェーダーにアップロード
let shatter = thermal.melt_accumulator.sample(surface_pos).min(1.0);
let entropy = thermal.temperature.sample(surface_pos) / 1000.0;
gl.uniform1f(u_shatter_loc, shatter);
gl.uniform1f(u_entropy_loc, entropy.min(1.0));
```

### 関連モジュール

| モジュール | 説明 | 駆動するUniform |
|-----------|------|----------------|
| `thermal.rs` | 熱拡散、融解、熱膨張、凍結 | `uShatter`, `uEntropy` |
| `fracture.rs` | 応力駆動クラック伝播（CSG減算） | `uShatter` |
| `pressure.rs` | 接触力 → 圧壊/膨出/凹み変形 | `uImpact` |
| `erosion.rs` | 風食、水食、化学侵食、削磨 | `uEntropy` |
| `sdf_destruction.rs` | リアルタイムCSGブーリアン破壊 | `uImpactRing` |
| `force.rs` | 爆発フォースフィールド（中心、半径、減衰） | `uMeteorImpact`, `uImpactRing` |
| `sim_field.rs` | トリリニア補間付き3Dスカラー/ベクトルフィールド | 全て（インフラ） |

### RenderConfig互換性

ALICE-SDFの `RenderConfig` で `destruction: true` を有効にすると、シェーダー側の破壊機能がアクティブになります：

```rust
let config = RenderConfig {
    destruction: true,       // Voronoiクラック、瓦礫、衝撃波の視覚効果
    spectral_rendering: true, // 加熱表面の黒体放射
    vfx_effects: true,       // 融解表面のドメインワープ
    ..Default::default()
};
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
    iterations: 1,     // サブステップあたりの solver pass (既定 1: Small Steps、増やすなら iterations でなく substeps)
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

計測値 (推定でなく実測): `cargo bench --bench physics_bench` (criterion、release profile)、Apple M 系、2026-09-15、alice-physics 1.2.0 絶対値は環境依存、Algorithm 列が実装の実態
<!-- perf-measured: 2026-09-15 benches/physics_bench.rs -->

| 演算 | アルゴリズム | 実測 |
|------|------------|------|
| Fix128 加算/減算 | 128bit add with carry | < 1 ns |
| Fix128 乗算 | 64×64→128 部分積 3 回 | 1.1 ns |
| Fix128 除算 | u128 整数商 + 小数部 64 step 長除算 | 141 ns |
| Fix128 sqrt | 96 step restoring digit recurrence (exact floor) | 193〜354 ns (run 間ばらつき、1.1.0: 9,676 ns、Newton × 除算 64 回) |
| Vec3Fix normalize | sqrt 1 回 + 除算 3 回 | 618 ns (1.1.0: 10,390 ns) |
| CORDIC sin/cos / atan | 48 反復固定 | ~1 µs |
| GJK 交差 | 最大 64 反復 | — |
| EPA 侵入深度 | 最大 64 反復 | — |
| BVH 構築 | Morton code sort、O(n log n) | — |
| BVH query / find_pairs | leaf AABB 毎の stackless 走査、O(n log n) 期待 | 1.1.0 は root AABB で query → O(n²) |
| World step、10 body × 60 step (default config) | | 398 µs (1.1.0: 5.11 ms) |
| World step、1000 重なり球 (10³ grid、default config: 8 substep 毎に detection) | `thousand_overlapping_spheres_1_step` | 初 frame 65 ms (接触 2,700、broad-phase 候補 57k × 8)、body が離れる 2〜10 frame 目: 7.7 ms/frame |

1.1.0 の外部レビュー (Linux x86_64) では frame 1 回 detection で重なり 1000 体 432 ms/frame、1.2.0 は substep 毎 detection (正しさの要件、CHANGELOG 参照) なので密な初 frame は重く定常は軽い 数値を引用する前に対象環境で bench を再実行すること

## MSRV ポリシー

**最小サポート Rust バージョン: 1.85**

ALICE-Physics は Serde 流の MSRV ポリシーを採用:

- 0.x 系での MSRV bump は **minor version bump** (`0.13.x → 0.14.0`) 扱い、patch bump にはしない
- 1.x 安定 line での MSRV bump も **minor version bump** (`1.x → 1.(x+1)`) 扱い、patch にはしない
- MSRV コミットメントは **最新 3 stable Rust channel** (N-2 policy) をカバー、リリース時点で current stable + 過去 2 個を支援
- `alice-physics` は nightly toolchain を要求しない
- CI の専用 `msrv` job が宣言 MSRV を実 compile: `cargo +1.85 check` を default feature / `no_std` (rlib) / core native feature set の 3 通り、`resolver = "3"` (MSRV-aware) で解決 1.0.x〜1.1.0 は 1.70.0 を宣言していたが、その toolchain は lockfile を parse できず `no_std` build は `core::f32::abs` (1.85) を要求していたため 1.2.0 で訂正 (上記 policy 通り minor bump)

**MSRV 保証の範囲** — *library* を default feature / `no_std` (`--no-default-features`、rlib) / core native feature (`std` / `simd` / `parallel` / `ffi` / `gpu-solver-bridge`) で build する範囲が 1.85 対象 範囲外:

| 対象 | 実効 MSRV | 理由 |
|------|-----------|------|
| `neural` / `replay` / `analytics` bridge feature | sibling crate (`alice-ml` / `alice-db` / `alice-analytics`) に従う、`alice-db 0.2.0-beta.2` → `alice-zip 0.3.0` 時点で 1.87 | sibling crate が独自に MSRV を持つ |
| test / bench (dev-dependencies) | 1.85+ | `criterion` / `serde_derive` |
| `wasm` feature | stable channel 推奨 | `wasm-bindgen` の更新が速い |

Downstream crate は `alice-physics` が patch release で MSRV を上げないことに依存でき、下流の MSRV window を破壊しない

## ビルド

**`--all-features` に関する注意**: `alice-physics` は `wasm` と `ffi` を相互排他 feature として設計 (それぞれ `wasm-bindgen` と C ABI FFI を link、単一バイナリで共存不可)。`cargo build --all-features` は意図的な `compile_error!` で失敗する どちらか片方で build

**推奨 feature 組合わせ:**

| ターゲット | コマンド |
|--------|---------|
| ネイティブアプリ / ゲームエンジン host (Unity、UE5、Godot) | `cargo build --features "std,simd,parallel,ffi,gpu-solver-bridge"` |
| ブラウザ (WebGL / WebGPU、wasm-bindgen 経由) | `cargo build --features "std,simd,parallel,wasm,gpu-solver-bridge"` |
| 組み込み / no_std | 依存として使う場合は `default-features = false`、単体 build は `cargo rustc --lib --no-default-features --crate-type rlib` (package は C ABI 用に `cdylib`/`staticlib` も宣言しており、これらは `std` が必要) |
| Python バインディング | `cargo build --features "std,simd,parallel,python"` |

docs.rs はネイティブ feature set でビルド (Cargo.toml の `[package.metadata.docs.rs]` 参照)

```bash
# 標準ビルド
cargo build --release

# no_stdビルド（組み込み/WASM向け）
cargo rustc --release --lib --no-default-features --crate-type rlib

# Unity / UE5 向け C ABI 成果物 (target/release/libalice_physics.{so,dylib,a}、alice_physics.dll)
cargo build --release --features ffi

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
| `simd` | No | `*_simd` / `dot_batch_4` API surface + `SIMD_WIDTH` **1.2.0 時点でスカラ等価**: SSE2/AVX2 に 128bit 乗算がなく lo→hi の carry も伝播できないため、intrinsic 経路がスカラ ADC chain より速くなることはない 将来の AVX-512 / NEON batch 経路のために API を保持 |
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

## テスト済みフィーチャー組み合わせ

全フィーチャー組み合わせがmacOS、Ubuntu、Windows上のCIでテストされています：

| 組み合わせ | ステータス | テスト数 |
|-----------|----------|---------|
| `--no-default-features` (no_std) | ✅ | 9 |
| `--features std` (default) | ✅ | 1730 unit + 72 integration + 解析解 13 + engineering 105 + default config oracle 13 + 決定論 44 + 21 doc |
| `--features simd` | ✅ | 20 |
| `--features parallel` | ✅ | 20 |
| `--features "simd,parallel"` | ✅ | 20 |
| `--features ffi` | ✅ | ビルドのみ |
| `--features python` | ✅ | 1 |
| `--features replay` | ✅ | ビルドのみ |
| `--features analytics` | ✅ | ビルドのみ |
| `--features neural` | ✅ | ビルドのみ |
| `--features wasm` | ✅ | ビルドのみ |

フィーチャー互換性マトリクス：

| | std | simd | parallel | neural | python | ffi | wasm | replay | analytics |
|---|---|---|---|---|---|---|---|---|---|
| **std** | - | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **simd** | ✅ | - | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **parallel** | ✅ | ✅ | - | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **wasm** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | - | ❌ | ❌ |

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
git tag v0.6.0
git push origin v0.6.0
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
| `cone` | `Cone` | コーンコライダー（頂点+Y、底面-Y） |
| `ellipsoid` | `Ellipsoid` | 3軸独立半径の楕円体 |
| `torus` | `Torus` | 主半径/副半径のトーラス |
| `plane_collider` | `PlaneCollider` | ヘッセ標準形の無限平面 |
| `wedge` | `Wedge` | 6頂点の三角柱（くさび） |
| `convex_mesh_builder` | `build_convex_hull()` | インクリメンタル凸包構築 |
| `filter` | `CollisionFilter` | レイヤー/マスク衝突グループ |

### ソルバー & ダイナミクス

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `solver` | `PhysicsWorld` | メインシミュレーションワールド |
| `solver` | `PhysicsConfig` | サブステップ、反復、重力、減衰 |
| `solver` | `RigidBody` | 動的/静的/センサー剛体 |
| `solver` | `BodyType` | Dynamic, Static, Kinematic |
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
| `joint_extra` | `PulleyJoint` | プーリージョイント（総距離拘束） |
| `joint_extra` | `GearJoint` | ギアカップリング（角速度比率） |
| `joint_extra` | `WeldJoint` | 破壊力閾値付き剛体溶接 |
| `joint_extra` | `RackAndPinionJoint` | 回転-並進変換ジョイント |
| `joint_extra` | `MouseJoint` | ターゲット追従ジョイント |
| `motor` | `PdController` | 1D比例-微分コントローラ |
| `motor` | `JointMotor` | 位置/速度/トルクモードモーター |
| `force` | `ForceField` | 風、重力井戸、ドラッグ、浮力、ボルテックス、爆発、磁気双極子 |
| `force` | `ForceFieldInstance` | シミュレーション内のアクティブなフォースフィールド |
| `material` | `MaterialTable` | ペアごと摩擦/反発テーブル |
| `material` | `PhysicsMaterial` | マテリアルプロパティ |
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
| `cloth_fluid` | `ClothFluidCoupling` | 双方向クロス-流体カップリング |
| `rope_attach` | `RopeAttachment` | 剛体ロープ接続 |
| `soft_body_cut` | `cut_deformable()`, `cut_cloth()` | 平面ベース切断 |
| `particle` | `ParticleSystem`, `ParticleEmitter`, `Particle` | 汎用パーティクルシステム |

### 2D物理

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `physics2d` | `Vec2Fix` | Fix128成分の2Dベクトル |
| `physics2d` | `Shape2D` | Circle, Polygon, Capsule, Edge |
| `physics2d` | `RigidBody2D` | 2D剛体 |
| `physics2d` | `PhysicsWorld2D` | 2D物理ワールド（SAT + XPBD） |
| `physics2d` | `Joint2D` | 2Dジョイント（Revolute, Distance, Weld, Mouse） |

### メッシュ & I/O

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `mass_properties` | `sphere_mass_properties()`, `box_mass_properties()` 等 | 質量・慣性テンソル計算 |
| `collision_mesh_gen` | `generate_collision_mesh()`, `simplify_mesh()` | SDF→メッシュ生成 |
| `scene_io` | `save_binary()`, `load_binary()`, `save_json()`, `load_json()` | シーンシリアライズ (std) |
| `multi_world` | `MultiWorld` | 複数独立物理ワールド |

### 可視化

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `heatmap` | `Heatmap`, `HeatmapConfig` | 応力/温度ヒートマップ |
| `flow_viz` | `FlowArrow`, `Streamline` | フロー可視化 |
| `contact_viz` | `ContactArrow`, `FrictionCone` | 接触可視化 |

### SDF統合

| モジュール | 型 / 関数 | 説明 |
|-----------|----------|------|
| `sdf_collider` | `SdfCollider` | SDFを衝突形状として使用 |
| `sdf_collider` | `SdfField` trait | 距離+法線評価インターフェース |
| `sdf_collider` | `ClosureSdf` | クロージャベースSDF実装 |
| `sdf_manifold` | `SdfManifold`, `ManifoldConfig` | SDF曲面からのマルチポイント接触 |
| `sdf_ccd` | `SdfCcdConfig` | SDF向け球体トレーシングCCD |
| `sdf_force` | `SdfForceField`, `SdfForceType` | SDF駆動力場（Attract, Repel, Contain, Flow） |
| `sdf_destruction` | `DestructibleSdf`, `DestructionShape` | CSGブーリアン破壊 (std) |
| `sdf_adaptive` | `AdaptiveSdfEvaluator` | 距離ベースLOD評価 (std) |
| `convex_decompose` | `DecomposeConfig` | ボクセルグリッド分解設定 (std) |
| `gpu_sdf` | `GpuSdfBatch` | GPUコンピュートSDFバッチ (std) |
| `gpu_sdf` | `GpuSdfInstancedBatch` | インスタンスGPU SDFバッチ |
| `gpu_sdf` | `GpuSdfMultiDispatch` | マルチバッチGPUディスパッチ |

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
v0.6.0 テストサマリ:
  - 84モジュール全体で645ユニットテスト
  - 72統合テスト（エンドツーエンド物理シナリオ）
  - 20ドキュメントテスト（実行可能な例）
  - 合計: 737テストパス
  - Clippy: 0警告
  - CI: Format + Clippy + Test (macOS/Ubuntu/Windows)
  - 全フィーチャー組み合わせパス（default, parallel, simd, no_std）
```

## ライセンス

AGPL-3.0-or-later - 詳細は [LICENSE](LICENSE) を参照。

AGPL は強いコピーレフト: `alice-physics` を (Unity / UE5 binding 経由を含め) link して配布 / 提供するゲーム・アプリケーションは AGPL で公開する義務がある これはオープンなエコシステムのための意図的な選択 AGPL 義務なしの商用利用は別ライセンスを用意するので <sakamoro@alicelaw.net> まで連絡

Copyright (C) 2024-2026 Moroya Sakamoto

## 謝辞

- XPBD: Muller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- GJK/EPA: Ericson, "Real-Time Collision Detection"
- Morton Codes: Morton, "A Computer Oriented Geodetic Data Base"
- CORDIC: Volder, "The CORDIC Trigonometric Computing Technique"
