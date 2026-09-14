# ALICE-Physics ROADMAP

Canonical roadmap for the alice-physics crate. Primary source of truth.
Memory index pointer: `[[reference-alice-physics-v1-roadmap]]` in claude-config.

## 現在位置 (2026-09-12)

**v0.13.0 landed** (commit `f37df0e`) — Session 4 push で 19 module 追加 (Tier ★★★ 4 + ★★ 7 + ★ 8)、ALICE-SDF v1.7.7 の `morphology` を S1 tier-★★★ integration partner として組合わせて 20 module 完備

**v0.14.0-preview.1 landed** (commit `155bdbf` + ALICE-Fluid `935b7a6`) — Physics 拡張 v2 Priority 1 5 items (Marching Tets / 3D IPM / spatial hash SPH / Crank-Nicolson / BFECC scalar)、+16 test Physics + 7 test Fluid

**v0.14.0-preview.2 landed** (commit `bc9ea4a`) — Physics 拡張 v2 Priority 2 6 items (MAC-face BFECC / nonlinear C-N / 3D thermal / BiCGStab / adaptive dt / edge-split refinement)、+21 test

**v0.14.0-preview.3 landed** (commit `a4d475b`、2026-09-13) — v1.0 roadmap 最短優先候補 3 項目 + clippy fix: Item G MSRV policy 明記 (README EN/JP) / Item D `#![deny(missing_docs)]` escalation (0 warning 実測、`warn` → `deny` 1-char) / clippy `approx_constant` fix (solver_tgs_hooks_6dof_oriented.rs:820 の `1.5708` → `FRAC_PI_2`) / Item J crates.io publish 前調査 (`docs/CRATES_IO_PUBLISH_INVESTIGATION.md` に集約、案 (b) SDF v1.7.7 pattern を v0.16.x で採用推奨)

**v0.14.0-preview.4 landed** (commit `69ced3d`、2026-09-13) — J-1 実態調査で sibling API drift が想定以上と判明 (`Ternary` / `AliceDB` / `prelude` / `DDSketch256` / `HyperLogLog12` 全て sibling で削除済、`~/ALICE-ML/src/lib.rs` と `~/ALICE-DB/src/lib.rs` は空)、案 (b) SDF v1.7.7 pattern を v0.16.x → **v0.14.0 に前倒し実施**: 4 bridge file 削除 (`analytics_bridge` / `db_bridge` / `neural` / `replay`、1424 行)、`Cargo.toml` から `neural` / `replay` / `analytics` 3 feature + 3 path dep 削除、`src/lib.rs` から 4 pub mod + neural prelude re-export 削除 検証: 1364 lib test 全 pass (regression 0)、`cargo publish --dry-run` PASS (195 file / 3.0 MiB packaged) — B1/B3 blocker 完全解消

**J-3 実 publish 完了** (2026-09-13) — `alice-physics v0.14.0-preview.4` を **crates.io に初 publish**、Cargo.toml `version` `0.13.0` → `0.14.0-preview.4` + `publish = false` 削除、194 file / 3.0 MiB / 718.6 KiB compressed uploaded (`cargo publish` 成功、`Published alice-physics v0.14.0-preview.4 at registry crates-io`)、`cargo search` で `alice-physics = "0.14.0-preview.4"` 反映確認済 crates.io URL: https://crates.io/crates/alice-physics ADR-002 の「v1.0.0 前に crates.io publish 事前試験実施」を **v0.14.0 preview で達成**

**v0.14.0-preview.5 landed** (commit `7d5d214`、2026-09-13) — Phase 1 quick wins landing: (1) B4 wasm × ffi mutual exclusion を `[package.metadata.docs.rs]` + README recommended feature table で ergonomic 化、(2) 3 新 example 追加 (`ragdoll_demo` / `bfecc_advection_demo` / `sph_boundary_demo`、現状 7 → 10 個)、(3) 2 新 fuzz target (`fuzz_joint` / `fuzz_cfd`、現状 3 → 5 個)、(4) crates.io に `0.14.0-preview.5` publish 成功 (197 file / 3.1 MiB / 724.3 KiB compressed) 検証: 1364 lib test 全 pass (regression 0)、10 example release build PASS、`cargo doc` clean cargo-public-api snapshot は nightly toolchain install 時間の関係で v0.15.0 に defer

- module 総数: 146 src file、`pub mod` 144
- lib test: 1364 (1327 → 1343 → 1364、v2 Priority 1+2 sprint で +37)
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

### ✅ v0.14.0-preview.1 (Physics 拡張 v2 Priority 1、shipped 2026-09-12)

commit `155bdbf` (+ ALICE-Fluid `935b7a6`)

- **Marching Tets** `sdf_fem_mesh.rs` `generate_marching_tets` — surface-conforming mesh (5-tet cube × 16-case LUT)
- **3D Spectral IPM** `spectral_ipm_3d.rs` (ALICE-Fluid) — T³ Darcy multiplier + rustfft row/col/slab pass
- **Spatial Hash SPH** `sdf_sph.rs` `SphSpatialHash` + `step_hashed` — O(N²) → O(N·k) 近傍探索
- **Crank-Nicolson thermal** `transient_thermal.rs` `crank_nicolson_step_1d` — A-stable + Thomas algorithm 三重対角
- **BFECC scalar advection** `cfd_solver.rs` `AdvectionScheme::Bfecc` + `advect_temperature_bfecc`

+16 test (1327 → 1343) Physics, +7 test (173 → 180) Fluid

### ✅ v0.14.0-preview.2 (Physics 拡張 v2 Priority 2、shipped 2026-09-12)

commit `bc9ea4a`

- **MAC-face BFECC** `cfd_solver.rs` `advect_velocity_bfecc` — u/v/w 面 3-pass BFECC 本実装
- **Nonlinear Crank-Nicolson** `transient_thermal.rs` `crank_nicolson_step_1d_nonlinear` — Picard iteration α(T)
- **3D transient thermal** `transient_thermal.rs` `transient_step_3d` + `stable_dt_3d` — Cartesian 7-stencil
- **BiCGStab pressure solver** `eulerian_grid.rs` `project_pressure_bicgstab` — Krylov subspace + diagonal precond
- **Adaptive time step** `cfd_solver.rs` `compute_max_dt` + `step_adaptive` — CFL 逆算 + ceiling
- **Edge-split refinement** `sdf_fem_mesh.rs` `SdfTetMesh::refine_by_max_edge_length` — 長辺 midpoint split (non-Delaunay)

+21 test (1343 → 1364)

### ✅ v0.14.0-preview.3 (Item G/D + clippy + J investigation、shipped 2026-09-13)

commit `a4d475b`

- **Item G MSRV policy 明記** — README EN/JP に MSRV Policy section 追加 (Serde-style: MSRV bump は minor version bump 扱い、N-2 stable channel 支援、nightly 非要求)
- **Item D `#![deny(missing_docs)]` escalation** — `src/lib.rs:189` で `warn` → `deny` (実測 0 warning、`cargo doc --no-deps` clean 確認済、file-level `#![allow(missing_docs)]` も 0 件 = 逃げ道なし)
- **clippy pre-existing block fix** — `src/solver_tgs_hooks_6dof_oriented.rs:820` `Fix128::from_f32(1.5708)` → `Fix128::from_f32(core::f32::consts::FRAC_PI_2)` (Priority 2 で拡張中に露呈した `approx_constant` block を解消)
- **Item J crates.io publish 前調査** — `docs/CRATES_IO_PUBLISH_INVESTIGATION.md` に集約:
  - B1 blocker: 4 path dep (`alice-ml` / `alice-db` / `alice-analytics`) に `version = "..."` fallback 未記述、`cargo publish --dry-run` の manifest verify で reject
  - B2 blocker: 3 sibling は全て local v0.1.0、crates.io 未 publish (`cargo search` 0 hit)
  - B3 latent bug: `--all-features` build で 4 個の path dep import drift (sibling 側 API rename に未追従、default build では顕在化しない)
  - **推奨**: v0.16.x で **案 (b) SDF v1.7.7 pattern** (`neural` / `replay` / `analytics` feature 削除 preview → dry-run → publish) 採用
  - v0.14.0 内追加タスク J-1 として B3 path dep drift 4 箇所修正を予定 (下記)

### ✅ v0.14.0-preview.4 (J-1 → 案 (b) 前倒し実施、shipped 2026-09-13)

commit `69ced3d`

- **J-1 実態調査**: sibling API が想定以上に drift (`Ternary` / `AliceDB` / `prelude` / `DDSketch256` / `HyperLogLog12` 全て削除、sibling `lib.rs` は空) — 単純 import fix では復旧不可と判明
- **案 (b) 前倒し実施** (SDF v1.7.7 pattern): v0.16.x で予定していた J-2 を **v0.14.0 に前倒し**
  - `src/{analytics_bridge,db_bridge,neural,replay}.rs` 4 file 削除 (合計 1424 行)
  - `Cargo.toml` から `neural` / `replay` / `analytics` 3 feature + `alice-ml` / `alice-db` / `alice-analytics` 3 path dep 削除
  - `src/lib.rs` から 4 `pub mod` + `#[cfg(feature = "neural")]` prelude re-export 削除
  - `[features]` に削除経緯 comment 追加 (v0.17.x J-4 での段階復帰への pointer)
- **検証**: 1364 lib test 全 pass (regression 0)、`cargo doc` clean (pre-existing 8 warning は無関係)、**`cargo publish --dry-run` PASS** (195 file / 3.0 MiB packaged、B1/B3 blocker 完全解消)

### 🚧 v0.14.0 (推定 1-2 週間) — API surface audit 前半 + 残 preview 済

Preview 4 wave が landed 済み、残作業:

- **B. Public API surface freeze 前半** — priority module (net / character / SDF 系) の `pub` → `pub(crate)` audit 着手 (144 pub mod のうち、`net_prediction` / `character*` / `sdf_*` 系から)
- **新 example 追加** — ragdoll / SPH / joint / character / BFECC velocity / BiCGStab pressure の代表 example (現状 7 → 14 個目標)
- **B4 (新規発覚) 対応**: `wasm` × `ffi` mutual exclusion (pre-existing `compile_error!`) の設計見直し — CI で `--all-features` を除外し続けるか、feature 設計を分割するか判断

自己採点 target: 品質 90/100 (現状 100/100 optimization scorecard は維持、public API 完成度で -10)

### 🚧 v0.15.0 (推定 2-4 週間) — API stability signaling

- **B. Public API surface freeze 後半** — 残 module audit + `#[non_exhaustive]` 戦略適用 (struct / enum の一部) + `#[deprecated]` alias で v1.0 名確定 (v0.x で名前変えたい API に marker + alias、v1.0 で確定名だけ残す)
- **C. cargo-semver-checks / cargo-public-api CI 通し** — `.github/workflows/security-audit.yml` に既に semver-checks job あり、拡張して cargo-public-api snapshot を repo に commit、PR diff で API surface 変化を可視化
- **F. Fuzz coverage 拡張** — 現状 `fuzz_collision` + `fuzz_step` の 2 target → joint / SDF CCD / trimesh / `cfd_solver` / `structural_solver` で 5-8 target 追加、24h 実行 crash 0 実績を CHANGELOG に記載

### 🚧 v0.16.0 (推定 2-3 週間) — determinism guarantee + ecosystem (J-2 は v0.14.0-preview.4 で前倒し完了済)

- **E. Determinism CI 6 環境 matrix** — macOS ARM + macOS x86 + Linux ARM + Linux x86 + Windows + WASM で毎 PR bit-exact snapshot golden test を run、joint / cloth / fluid / SDF CCD / trimesh に拡張
- **H. Ecosystem 契約 freeze** — ALICE-TRT `GpuSolverBridge` trait / ALICE-SDF `SdfField` trait / ALICE-Bamboo / ALICE-Anima / ALICE-Kinematics との integration point の method signature freeze、各 partner crate と semver policy 契約書化
- ~~**J-2. bridge feature 削除 preview commit**~~ ✅ **v0.14.0-preview.4 で前倒し実施済** (詳細は [`docs/CRATES_IO_PUBLISH_INVESTIGATION.md`](CRATES_IO_PUBLISH_INVESTIGATION.md) の追記 section 参照)

### ✅ v0.14.0-preview.4 crates.io publish 完了 (J-3 landed 2026-09-13)

- ~~**J-3. publish 実行**~~ ✅ 完了: `alice-physics v0.14.0-preview.4` を crates.io に初 publish
- crates.io URL: https://crates.io/crates/alice-physics
- 外部 downstream から `cargo add alice-physics --pre` で使用可 (pre-release identifier `-preview.4` のため default は除外、明示的な `--pre` or `@0.14.0-preview.4` で opt-in)
- ADR-002 の「v1.0.0 前に crates.io publish 事前試験実施」原則を **v0.14.0 preview で達成**
- **J-4** (v0.17.x 以降): sibling 3 crate (`alice-ml` / `alice-db` / `alice-analytics`) が crates.io publish された段階で、削除した bridge feature を段階復帰 (ALICE-SDF v1.8.0 と同 pattern) 削除したコードは git 履歴 (v0.14.0-preview.4 commit `69ced3d`) から参照可能

### ✅ v0.14.0-preview.5 (Phase 1 quick wins、shipped 2026-09-13)

commit `7d5d214` / crates.io: `alice-physics = "0.14.0-preview.5"`

- **B4 wasm × ffi mutual exclusion**: `[package.metadata.docs.rs]` 追加 + README EN/JP に recommended feature combinations table 追加 (mutual exclusion は正当な設計、ergonomic 化のみ)
- **新 example 3 個** (7 → 10): `ragdoll_demo` / `bfecc_advection_demo` / `sph_boundary_demo`
- **新 fuzz target 2 個** (3 → 5): `fuzz_joint` / `fuzz_cfd`
- crates.io に 2 nd publish 成功 (197 file / 3.1 MiB / 724.3 KiB compressed)
- 検証: 1364 lib test PASS、10 example release build PASS、`cargo doc` clean

### ✅ CI hotfix (fmt + release runner、2026-09-13)

- **fmt fix** commit `c0e3880` — preview.5 提出時に `cargo fmt --all` を先に実行しなかった漏れ、3 example (`bfecc_advection_demo` / `ragdoll_demo` / `sph_boundary_demo`) に multi-line 化 fmt 適用、CI Format check job 通過
- **release runner fix** commit `727d854` — `macos-15` (Apple Silicon ARM) → `macos-15-intel` (Intel-native) に置換、`x86_64-apple-darwin` target が cross-toolchain なしで build 可能に (preview.4 / preview.5 の Release workflow 失敗の pre-existing infrastructure bug fix)、次回 tag push (`v0.14.0-preview.6` 以降) で復旧確認

### ✅ Public API snapshot 生成 (C 準備、2026-09-13)

- **`docs/PUBLIC_API_SNAPSHOT.txt`** — `cargo +nightly public-api --simplified` 出力を repo に commit、**20,201 public API item** の baseline を確立 (B/C 用)
- **`docs/PUBLIC_API_SNAPSHOT.md`** — snapshot の位置付け + regenerate command + diff pattern + roadmap wiring 集約
- v1.0 Item C (cargo-semver-checks / cargo-public-api CI) の入力側整備完了、CI job 追加は v0.15.0 phase で実施

### ✅ v0.14.0-preview.6 (Phase 2、shipped 2026-09-13)

commit `d215f4a` / crates.io: `alice-physics = "0.14.0-preview.6"`

- **Item C 入力側**: `.github/workflows/security-audit.yml` に `public-api-diff` job 追加 — `cargo +nightly public-api --simplified` の出力を `docs/PUBLIC_API_SNAPSHOT.txt` と diff、非空なら fail 意図的 API 変更時は開発者側で snapshot 再生成 + commit を要求
- **新 fuzz target 2 個** (5 → 7): `fuzz_ccd` (sphere_sphere_toi / sphere_plane_toi swept 入力耐性) / `fuzz_trimesh` (from_indexed + raycast + closest_point degenerate 入力耐性)
- crates.io に 3rd publish 成功 (199 file / 4.6 MiB / 868.5 KiB compressed、PUBLIC_API_SNAPSHOT.txt 20,201 行増分)
- 検証: 1364 lib test PASS、`cargo fmt --all --check` clean、fuzz target cargo check PASS

### ✅ v0.14.0-preview.7 (F 8/8 完全達成、shipped 2026-09-13)

commit `f1b4209` / crates.io: `alice-physics = "0.14.0-preview.7"`

- **Item F 完全達成** — 新 fuzz target `fuzz_structural` (StructuralSolver + Rectangular + CantileverEndPoint + PLA、extreme aspect ratio / heavy load 耐性) 追加、fuzz coverage **7 → 8** (v1.0 目標 5-8 range を full 到達)
- 検証: 1364 lib test PASS + `cargo fmt --check` clean + `cargo publish --dry-run` PASS + `fuzz_structural` cargo check PASS

### 🚧 v0.14.0-preview.8+ / v0.14.0 stable — 継続開発

Phase 1+2+F 完了、以降は最重量の B に集中:

- **B Iteration 1 (priority modules 調査、2026-09-13 完了)** — `netcode_prediction` / `character_state` / `character` / `sdf_character` / `sdf_sph` / `sdf_wind_field` / `sdf_fem_mesh` の 7 module (53 pub item) を survey、**全て clean な public API と判定**、`pub(crate)` 格下げ候補 0 見つかった 詳細は [`docs/PUB_AUDIT_ITERATION_1.md`](PUB_AUDIT_ITERATION_1.md) 参照
- **B Iteration 2 (P1 solver internals 調査、2026-09-13 完了)** — 9 module 127 pub item を survey (`solver` / `contact_cache` / `dynamic_bvh` / `solver_tgs` / `solver_tgs_hooks` / `solver_tgs_hooks_6dof` / `solver_tgs_hooks_6dof_oriented` / `solver_tgs_hooks_6dof_scoped` / `solver_tgs_hooks_6dof_oriented_scoped`)、**4 items pub(crate) 格下げ実施** (`NULL_NODE` / `DynamicNode` / `MAX_MANIFOLD_POINTS` / `tangent_frame`、2 commit)、snapshot 20,201 → 20,179 items (−22)、`solver_tgs*` extension mechanism 6 module 60+ items は **architectural decision required** で deferred (Option A: feature-gate / Option B: keep pub + unstable caveat / Option C: pub(crate) 全撤去 の 3 択、user 判断待ち) 詳細は [`docs/PUB_AUDIT_ITERATION_2.md`](PUB_AUDIT_ITERATION_2.md) 参照
- **B Iteration 3 (P2 math / BVH / spatial 調査、2026-09-13 完了)** — 3 module 108 pub item を survey (`math` / `bvh` / `spatial`)、**8 items pub(crate) 格下げ実施** (math: `pack_pair` / `select_fix128` / `select_vec3`、bvh: `morton_code` / `point_to_morton` / `ESCAPE_NONE` / `MAX_PRIMS_PER_LEAF` / `BroadphaseHybrid`、2 commit `dc32236` + `3f0d12c`)、snapshot 20,179 → 20,155 items (−24)、`BvhStats` は `LinearBvh::stats()` の return type leak で格下げ不可 (keep pub)、`spatial` は全 pub items が prelude commitment + 内部 cross-module 利用で保護 (0 downgrades)、`CachedContactPoint` field visibility は **v1.0-rc.1 で `#[non_exhaustive]` 化検討** として resolved 詳細は [`docs/PUB_AUDIT_ITERATION_3.md`](PUB_AUDIT_ITERATION_3.md) 参照
- **B Iteration 4 (P3 CFD 内部 調査、2026-09-13 完了)** — 4 module 58 pub item を survey (`eulerian_grid` / `multiphase` / `interface_capture` / `turbulence`)、**31 items pub(crate) 格下げ実施** (eulerian_grid 6 + multiphase 4 + interface_capture 3 + turbulence 18、4 commit `1d72044` + `72da7f2` + `0612a51` + `32acdce`)、snapshot 20,155 → 20,064 items (−91、reserved RANS/wall-function/PLIC 全 auto-impl 削減)、cross-module internal helpers (`MacGrid`/`Grid3d` leak 経由 + `project_pressure`/`g2p_velocity`/`sample_*`/`trilinear_*`/`fast_sweeping_reinit`/`SMAGORINSKY_CS` 等) は keep pub (v1.0-rc.1 で design 判断)、4 module に module-level or item-level `#[allow(dead_code)]` + "Integration status" 明記 詳細は [`docs/PUB_AUDIT_ITERATION_4.md`](PUB_AUDIT_ITERATION_4.md) 参照
- **B Iteration 5 (P4 structural 内部 調査、2026-09-13 完了)** — 5 module 65 pub item を survey (`beam_stress` / `plastic` / `buckling` / `fatigue` / `creep_longterm`)、**30 items pub(crate) 格下げ実施** (beam_stress 1 + plastic 9 + buckling 6 + fatigue 7 + creep_longterm 7、5 commit `4374397` + `7a0e7f5` + `4052faa` + `cd87aef` + `19d017f`)、reserved RANS/WLF/PLIC 相似の "downstream 0 で future integration 待ち" pattern が structural にも存在 (`StressTensor` / `KEpsilonState`-analog `WlfConstants` / `FatigueReport` 等)、4 module に module-level `#![allow(dead_code)]` + "Integration status" doc note、keep pub: Bamboo downstream 使用 (`BeamAnalysis` / `CrossSection` / `LoadCase`) + structural_solver 内部使用 (`PlasticModel` / `NortonCreep` / `analyze_column` / `ColumnBucklingReport` / `SnCurve` / `miner_damage` / `FindleyParameters` / `predict_strain`) 詳細は [`docs/PUB_AUDIT_ITERATION_5.md`](PUB_AUDIT_ITERATION_5.md) 参照
- **B Iteration 6 (P5 I/O + serialization 調査、2026-09-13 完了)** — 4 module ~41 pub item を survey (`scene_io` / `collision_mesh_gen` / `debug_render` / `heatmap`)、**0 items 格下げ**、全 pub items が prelude commitment or prelude 経路 leak (`DebugDrawData.lines/.points` 経由 `DebugLine`/`DebugPoint` + `PhysicsScene.config` 経由 `scene_io::PhysicsConfig`) で mechanical downgrade 不可、field visibility design task は v1.0-rc.1 に defer 詳細は [`docs/PUB_AUDIT_ITERATION_6.md`](PUB_AUDIT_ITERATION_6.md) 参照
- **B mechanical audit complete** — Iter 1-6 で 32 module ~452 pub item survey、73 items 格下げ、snapshot 20,201 → 19,956 items (−245)
- **B Final iteration (2026-09-13 完了、user "推奨で" 指示で自律遂行)** — 2 architectural decision を landing:
  1. `solver_tgs*` Option C 選択 (commit `ee3efb0`): 6 module 60+ items を pub → pub(crate)、`pub mod` → `pub(crate) mod` へ、cleanest v1.0 surface、demand 出たら semver-minor で再展開 downstream 0 usage が 6 iteration にわたって unchanged だった実測データが選択根拠
  2. `#[non_exhaustive]` を 7 struct に追加 (commit `427d378`): prelude 経路 leak の forward-compat hedge (`ContactManifold`/`CachedContactPoint`/`DebugDrawData`/`DebugLine`/`DebugPoint`/`PhysicsScene`/`PhysicsConfig`)、pub field observability 維持しつつ future 拡張余地確保、accessor migration は v2.0-scope に defer
  - snapshot 19,956 → 19,406 (−550)、累積 20,201 → 19,406 (**−795 items across 全 audit campaign**)
  - 詳細は [`docs/PUB_AUDIT_FINAL.md`](PUB_AUDIT_FINAL.md) 参照
- **✅ v1.0 Item B COMPLETE** — mechanical audit + architectural decisions 全 landing 済
- **新 example 4 個追加** (10 → 14) — joint / character / BiCGStab pressure / adaptive dt
- **Item C 出力側**: semver-checks の hard-gate 化 (現状 `continue-on-error: true`、B Iteration 3+ landing 後に有効化)
- **F 24h 実行**: 8 target 全てで 24h fuzz run + crash 0 実績 (v1.0-rc validation の一環)
- 完了次第 `0.14.0-preview.8` → ... → **`0.14.0` stable** publish

### ⏳ v1.0.0-rc.1 (推定 6-8 週間)

- **✅ I. Migration guide 執筆完了 (2026-09-14)** — [`docs/MIGRATION_0.x_TO_1.0.md`](MIGRATION_0.x_TO_1.0.md) 起草済 (~340 行、per-module 削除項目テーブル + `#[non_exhaustive]` 影響 + Cargo.toml migration + 6-environment determinism promise + post-1.0 stability guarantees + 再曝露リクエスト手順)
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
