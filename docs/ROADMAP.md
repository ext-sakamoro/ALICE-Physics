# ALICE-Physics ROADMAP

Canonical roadmap for the alice-physics crate. Primary source of truth.
Memory index pointer: `[[reference-alice-physics-v1-roadmap]]` in claude-config.

## 🎉 現在位置 (2026-09-30): 4 課題の第 2 increment を landing

⚠️ **番号で呼ばない** — 記録には「壁 N/4」が 2 組あり一致しません (詳細は memory
`project_alice_physics_research_walls` § 番号の曖昧性) 話題名で参照してください

| 課題 | commit | 到達点 |
|---|---|---|
| 共回転 FEM | `6896e0a` | 増分独立性 3.5e10 → **3-4 ulp** ⚠️ **0 ulp は到達不能** (不動点が存在しない) |
| 弾塑性 FEM (J2) | 未 commit | 小ひずみ J2 + bilinear 等方硬化、`solve_elastoplastic` (consistent tangent の Newton)、oracle 19 本 + 変異 42/42 red ⚠️ 有限ひずみ `F = Fe·Fp` / 移動硬化 / P2・P3 / 動的は未着手 |
| CFD 壁 BC | `8fec2ff` | Ghia Re=100 **4.5% → 86.2%**、解像度 sweep で 94.5% まで 1 次収束 |
| Maxwell FDTD | `ecd8c61` | source-free Yee、⚠️ **SI は `ε₀·μ₀` が 7.7 bit で不可**、正規化単位 + `S=9/16` |
| 多体 ABA | `1bab99b` | ⚠️ **`solve()` が `link.joint` を読まない**状態から real ABA、oracle 12 本 |

### 第 22 increment (2026-10-02、壁 4 — multigrid を slab 分割に載せる、段階 1)

`eulerian_grid::multigrid_decomposed` (crate 内、新規子 module) を足した `project_pressure_multigrid` を `ranks` 本の連続 `z` slab に分け、halo 層を交換して W-cycle を回す プロセス内の rank (全長 buffer、`RankTransport` / `LocalTransport` を再利用) で、**単一 process の結果と bit 一致** (grid 8³ / 16×8×4 / 4×4×16 / 8×8×1 / 16³、rank 1〜16、開放 / 壁つき、`ranks > nz` を含む) 各 level の slab 境界を 2 の冪の倍数に取るので restriction / prolongation が rank 内で閉じる 層数が rank 数を下回る level は rank 0 に集約して単一 process の `mg_vcycle` を回し、補正を所有者に返す
oracle 5 本 (bit 一致 / halo を iteration ごとにすると不一致 / 何も届けない transport で不一致 / 単一 process が拒否する入力は不変 / layout の分割と整列) 変異 20 件中 18 red、2 件は等価変異 (1 セルの最粗 level の零化は結果を変えない / 届け先 buffer の sentinel 埋めは配送が成功する限り観測できない)
未着手 (段階 2〜3): slab 局所記憶域 / rank ごとに 1 process の driver (`SocketTransport` 上) / FMG・warm start / 集約した粗 level の並列化 ⚠️ 実際の 1e8 規模での 1 step の時間は未測定 (23 分は見積り)

### 第 21 increment (2026-10-02、超弾性の小ひずみの極限を線形弾性に一致させる)

超弾性の体積項を `κ(J−1) − p_ref/J` (エネルギーは `−p_ref ln J + κ/2 (J−1)²`) に替え、`κ = λ − offset(model)` (Neo-Hookean 0 / Mooney-Rivlin `4C₂` / Yeoh `8C₂`) を P1 / P2 / P3 の solver が共有の 1 関数から取るようにした 小ひずみの極限が線形の `(λ, μ)` に一致する (旧形は `μ/3` 硬かった) 応力にも接線にも `ln` は出ず厳密 oracle を維持 圧縮側に障壁が付く 閉形式 oracle は手で導出し直し (P2・P3 の一軸は `q` の二次式)、solver 層で荷重を `2⁻ᵏ` 倍にすると超弾性と線形の差が 4 倍ずつ縮む (実測比 0.25) 変異 20/20 red
⚠️ 新法則の下で修正 Newton の線形 surrogate が大伸張で要素を反転させた (+125% 等容伸張の中心荷重 scene が 5 N でも `Inverted`) ため、超弾性 law 付きの修正 Newton に残差の line search を足した 同 scene は 5〜2000 N で収束 (200 N で 27 step) 未着手: `strain_energy_density` に体積項が無く `cauchy_stress` のポテンシャルになっていない件 / モデルの μ と材料の μ の不一致検査 / consistent tangent の P2・P3・動的への接続

### 第 20 increment (2026-10-02、壁の外側 (1)(2) — monolithic の判断記録 + 1 byte 刻み wire の gate)

壁 1〜4 が 8/8 で閉じた後に残る「壁の外側」3 件のうち 2 件を文面ごと閉じた (3 件目 Gartling は測定中、別 increment)

| 話題 | 到達点 |
|---|---|
| **(1) monolithic 組立** | ⛔ **作らない (判断を数値で固定)** 入口条件 `\|λ_max\| > 0.9` (handbook `c_v`) に対し、計器 `the_monolithic_entry_condition_is_three_decades_away_at_real_heat_capacities` (固定点での有限差分 Jacobian + 冪乗法、2.3 s) が鋼 `+3.01e-4` / アルミ `+4.74e-4` / PLA `+6.06e-4`、非物理 `2⁻¹⁰` のみ `+0.9366` 開く経路 (断熱せん断帯 → 先に正則化) と要る部品 (非対称 Krylov 0 件 / 疎行列直接法 0 件) を `coupled_iteration` module doc に記録 ⚠️ 表は散文でなく test が印字する |
| **(2) 複数ノード分散** | 文面を閉じた: **測ったのは 1 ホスト 8 プロセス loopback TCP (arm64 / x86_64 別々、fold は cross-arch bit 一致)、2 ホスト / アーキ跨ぎの 1 solve / MPI backend は未** `CrossFault::ChunkedWire` — `Read` / `Write` を 1 byte に刻む wrapper で 2 プロセス解が bit 一致、歯は「call 数 == byte 数 かつ 1 層分以上」(⚠️ byte 数だけの閾値は whole buffer 通過の変異を見逃した、実測) 変異 4/4 red deadlock は交換 schedule が全 rank で同一の直列列なので成立しない (A-3.2c の comment、0 run の code 読み) |
### 第 19 increment (2026-10-02、共回転 FEM の超弾性に consistent tangent)

`CorotationalConfig::with_consistent_tangent` を追加した (opt-in、既定は従来どおり) 閉形式の接線を matrix-free で CG に渡す全 Newton で、Neo-Hookean 200 N は 181 step から 3 step、2000 N は 758 step から 4 step (release の実測) 修正 Newton との根の差は 1e-9 mm 台、増分独立性は 6.5e-13 mm
⚠️ Backlog の「修正 Newton は 2000 N で収束しない」は誤りで、実測では単調に収縮して 758 step で `Ok` だった (問題は速度) 超弾性の小ひずみの極限が線形と `μ/3` ずれる件は第 21 increment で解消 P2 / P3 と動的への接続は未着手

### 第 18 increment (2026-10-02、反力 `reactions()` を P2 / P3 に)

`quadratic_elastic_fem::reactions` / `cubic_elastic_fem::reactions` を足した (P1 と同形、`law: Option<HyperelasticModel>`) 2026-10-01 の生存変異 `A1-drop-J-in-piola` は **P2 / P3 の `hyperelastic_stress`** で出たもので、P1 の `reactions` (`00d0be2` 系) は派生先を閉じて原因側が開いたままだった oracle 10 本 (`tests/analytic_reactions.rs`、各要素 5 本)、変異 14 / 14 red (実装 8 + 配線 6)、snapshot +2 / −0、`examples/support_reactions.rs` が P1 / P2 / P3 で同じ −700 N
⚠️ 反力は solver と同じ private 組立を通す (別実装は oracle を消す) ⚠️ 面の恒等式は Kuhn 格子の並進対称に依る (格子でない mesh では側面の対消滅が成立しない)

### 第 12 increment (2026-10-01、`step` の既定の圧力射影を multigrid に)

`CfdSolver::step` の既定が、全軸が 2 の冪の格子では multigrid (6 サイクル)、そうでなければ従来の GS になった 数値結果が変わるので影響を先に実測した 非 2 冪 (Gartling / duct / golden) は bit 不変、2 の冪では `reattachment_lengthens_under_grid_refinement` の記録値が 6 桁目以降で動く (doc は両方の値を併記、assert は不変)
GS を使うには `step_multigrid(dt, 0)` `step_adaptive` は自動で multigrid になる `step_flip` は GS のまま (別判断)
⚠️ 6 サイクルのコストは GS 30 sweeps と同程度で、得られるのは精度 (8³ / 16³ / 32³ で同等以上) 純 Neumann の平均圧力固定と `step_flip` への multigrid 配線は未着手

### 第 11 increment (2026-10-01、FLIP / PIC の粒子経路)

`CfdSolver::step_flip` を追加し `p2g_normalized` を production から呼ばれる形にした (配線ガードの baseline から削除) oracle 18 本、変異 34 / 35 が red (残りは等価変異)
⚠️ 粒子が領域全体を満たす場合のみ正しい 自由表面 (空気セルの Dirichlet、セル分類、粒子の再シード) / 周期境界 / multigrid の配線は未着手

### 第 10 increment (2026-10-01、multigrid 圧力解法)

`project_pressure_multigrid` (Galerkin 型、W サイクル + 補正 2 倍、2 の冪のみ) と `CfdSolver::step_multigrid` を追加した 同じ精度に達する反復数は multigrid 19 / 20 / 21 cycles に対し red-black GS 167 / 602 / 2271 (n = 8 / 16 / 32)
⚠️ `step` の既定は GS のまま (呼び出し側が `step_multigrid` を選ぶ) 純 Neumann の平均圧力固定と、自由表面は未着手
配線ガードが `project_pressure_multigrid` の未配線を最初に捕捉し、`step_multigrid` の新設で解消した

### 第 9 increment (2026-10-01、配線ガード)

`scripts/wiring_guard.py` を追加した (CI の `fmt` job + preflight、oracle 23 本、検査器の変異 11 件が red) 実装したのに production から呼ばれない pub item と、理由の無い `allow(dead_code)` の新規追加を止める
⚠️ 既存の未配線 588 件と dead_code 17 file は baseline に記録しただけで、**解消はしていない** (P2G の `p2g_normalized` も baseline に載っている) その後、推移的な未配線の検出と名前衝突の緩和と Cargo workspace 対応を足した (oracle 79 本、変異 25 件が red、baseline は 1059 件に増加)

### 第 8 increment (2026-10-01、P2G の重み正規化)

`eulerian_grid::p2g_normalized` を追加した (`tests/analytic_p2g.rs` の oracle 4 本、変異 5 件が red)
`p2g_trilinear` は重み総和を持たず一様速度を再現できなかったので、正規化つきの入口を別に置いた
退化入力の試験で負座標の粒子が角の face を上書きする不具合を見つけて直した (`p2g_normalized` の入口で除外、`split` の clamp は移流が依存するので不変)
⚠️ `cfd_solver` の自前分配は置換していない (同じ結果になるかの実測が先)、FLIP / PIC の時間発展ループも未着手
その後 `p2g_normalized` の累積を厳密積に置き換えた (項を丸めず 256 bit に足し、256 ÷ 128 の除算を 1 回、0 方向切り捨て) 分子の wrap (`Σ w·v` が 2^63 で mod 2^128 を越える) と丸め増幅 (誤差約 1 ulp / Σw、重み raw=1 で face が −1.0) が消え、一様場は任意の v と位置で厳密に再現される 丸めが起きていた通常域の入力では bit が変わる oracle 8 本 + 256 bit 整数と除算の unit test 8 本、変異 11 件がすべて red `step_flip` の定義域 (FLIP 差分 2^62 / 圧力右辺 2^47) は別件

### 第 6 increment (2026-10-01、壁 3 強連成 + 壁 4 HPC 並列)

⚠️ **既存行は書き換えていません** 下記は追記です 第 2 increment の「物理間連成 ⚠️ production caller は 0」は
**2 段あり、上段は解消済・下段が本 increment で解消**しました — `CoupledField` の caller は `thermal` /
`phase_change` に付いていた一方、⚠️ **FEM の residual には入っていなかった**ので、熱弾性は解けませんでした

| 話題 | commit | 到達点 |
|---|---|---|
| **壁 3 熱弾性連成** | `af05769` → `04e8259` `2340e65` `398017d` | ⚠️⚠️ **`src gap` 2 件 → 0** `σ = C : (ε − ε_th)` を residual と報告応力に入れ、`#[ignore]` 4 本を理由文の書き換えなしで外して **6 passed** 公開 API は追加のみ (`ThermalExpansion` / `solve_with_eigenstrain` / `FemError` の新 variant、`solve` の signature 不変、snapshot `+20 / −0`) ⚠️ **先に oracle 側の欠陥を潰した** — 旧 oracle は温度場を作って `assert_field_is_uniform` した後 **`solve` に渡していなかった**ので、実装しても red のままだった |
| **壁 4 A-5 実測** | — | ⚠️⚠️ **1e8 剛体は単一機で不可能** `size_of::<RigidBody>()` = 640 B ⇒ **59.6 GiB** (本機 RAM 16.0 GiB の 3.7 倍) / 1 step **76.8 秒** Eulerian は 6.0 GiB だが GS 1 反復 **18.5 秒** ⇒ **分散は容量と速度の 2 つの独立した理由で要件** |
| **壁 4 A-1 順序非依存** | `c617e30` | red-black sweep の訪問順独立を oracle 化 (自然順 / 逆順 / stride-7 置換で bit 一致 + colour 分けを外すと動く対照群) ⚠️ **スレッド並列は入れなかった** — 8 core / 128³ で rayon 2 形態がどちらも bit 一致だが**逐次 388.6 ms より遅い** (935.4 / 508.9 ms) working set 33 MiB の 7 点ステンシルは**帯域律速**で、単一 SoC に core を足しても帯域は増えない |
| **壁 4 A-3.1 領域分割** | `6968c22` | `z` slab 分割 + halo 交換が **分割数を変えても monolithic と bit 一致** 割り切る / 割り切らない / **rank が余って何も所有しない**分割を含む ⚠️ **各 rank の halo 外を sentinel で塗り潰す**のが勘所 — 共有 buffer を素直に分割すると**他 rank の正しい値が見えて halo 幅不足でも通る** |
| **壁 4 A-3.2a trait 化** | `e82bbb8` | halo 交換を `RankTransport` 経由に ⚠️ **既存 3 本が捕まえない変異がある**と実測 (2 つ目の実装が `src` でなく `dst` から読む形は既存 3 本すべて green で素通り) ⇒ 新旧は「層を**どう**運ぶか」と「**どの層をいつ**運ぶか」を別々に測っている |
| **壁 4 A-3.2b 越境** | `de53740` | ⚠️⚠️ **`RankTransport` がアドレス空間を越えることを 2 プロセスで固定** `SocketTransport::slab_mut` の `assert_eq!(rank, my_rank)` で「rank 局所」が comment でなく**検査される性質**に ⚠️ **プロセス内実装では原理的に検出できない** (隣の `Vec` が読めて正しい答えが返る) 実測: driver を rank 非局所にする変異は**越境 3 本 red / 既存 6 本すべて green** |

### ⚠️ 本 increment で 3 回独立に出た構造 — 層ごとに「その層でしか見えない誤り」がある

| 層 | その層でしか見えない誤り | 下の層では |
|---|---|---|
| A-3.2a | 2 つ目の transport が `dst` から読む | 既存 3 本すべて green |
| A-3.2b | driver が rank 局所でない | 既存 6 本すべて green |
| 壁 3 | 覆い判定の拒否経路 | 既存 5 本すべて green (**各 scene が自分の覆いを assert するので拒否経路を通らない**) |

⚠️⚠️ **下の層の oracle をいくら足しても上の層は覆えない** そして **どれも `#[ignore]` に現れず破壊試験でしか出ない**
⇒ 成果物を「oracle を足す」でなく **「対応する変異が red になる実測」** として定義するのが要点

### ⚠️ 残っているもの (「測っていない」の明示)

- ⚠️ **`ranks = 2` では「全 rank が schedule 全体を歩く」性質が測れない** (2 rank では全 delivery が自分の当事案件) 3 プロセス harness が必要
- ⚠️ **全長 buffer を維持しているので 1e8 規模には依然載らない** slab 局所の記憶域は未着手 (A-3.1 の「halo 幅の誤りが deadlock でなく**不一致**として落ちる」性質に依存しているため、代替 guard の設計が要る)
- ⚠️ **transport 層に値 oracle が無い** 代替は破壊試験 M1-M4 で、**変異を列挙し尽くした保証は無い**
- ⚠️ **MPI は入れていない** correctness の検証には不要 (必要なのは実プロセス境界 1 つ) で、CI に system 依存を足すため 実機が複数台揃ってから別途判断

### 第 7 increment (2026-10-01 夜、壁 2 P3 / 壁 3 確定 / 壁 4 slab 局所化)

⚠️ **既存行は書き換えていません** 下記は追記です

| 話題 | commit | 到達点 |
|---|---|---|
| **壁 2 P3** | `2646226`〜`6a84bfa` | ✅ **20 節点 3 次四面体要素** ⇒ 壁 2「高次要素 P2/P3」の P3 側が埋まった ⚠️ **degree 2 以上の四面体求積は完全 dyadic にできない** (モーメント `1/10` / `1/35` が奇数因子を持つ) ⇒ 問いは「厳密な則があるか」でなく「何回丸めるか」 ⚠️⚠️ **P3 固有の両立不能**: 辺節点が `(2a+b)/3` にあるので**節点位置の厳密性と `∇λ` の厳密性は原理的に両立しない** (`1/(27·det D)` が dyadic になりえない) P1 / P2 は節点配置に `1/3` が要らないので競合しない |
| **壁 3 確定** | `af05769` `04e8259` `2340e65` `398017d` `517f808` | ⚠️⚠️ **`src gap` 2 → 0 かつ「完全強連成は名乗らない」が確定** 熱弾性 (温度 → 応力) の一方向連成が residual に入った ⚠️ **先に oracle 側の欠陥を潰したのが決定的** (旧 oracle は温度場を作って一様性を assert した後 `solve` に渡していなかったので、実装しても red のままだった) ⚠️ **4 仮説の否定を `Fix128` で測り直しても判定不変** ⇒ **逆向き連成 / monolithic solver を作らない判断が固定小数点でも正当化された** (作れば「より正しいと言える根拠がない実装」が残る) |
| **壁 4 slab 局所化** | `de53740` `1f40727` `17d75d3` | 2 プロセス → 3 プロセスの越境分割を bit 一致で固定し、slab 局所の記憶域で **1 プロセスの footprint を 9.50 → 1.302 GiB (7.3 分の 1)** ⚠️ **halo 幅の誤りは「読んだ位置で abort」** (値比較でなく要求が通らないことで分かる) |

### ⚠️⚠️ 壁 4 の訂正 2 件 (調停役が自分の誤りを実測で正した)

| 誤り | 実測 | 影響 |
|---|---|---|
| 「1e8 cells = **6.0 GiB**」 | ⚠️ **9.78 GiB** (Mac mini / RAM 32 GiB で実走、max RSS) harness が `MacGrid` の 4 配列しか数えておらず **mask / 逆対角 / 右辺の 3 本 (3.8 GiB) を落としていた** | 分散が要件という結論を**強める** |
| 「R=8 で合計 **8.42 GiB** ⇒ **1 台に載る**」 | ⚠️⚠️ **合計 10.42 GiB** per-cell で `pressure` の 16 B を落として 86 B と計算していた (正しくは 102 B) ⚠️ **さらに合計が rank 数で割れると暗黙に仮定していた** — halo が重なるので**全長 1 本 (9.50 GiB) より 3% 多い** | ⚠️ **slab 局所化が解くのは「1 プロセスが載るか」で、「1 台に載るか」は台数でしか解けない** |

⚠️ **手段と目的を取り違えた形** 「footprint を縮める」は達成したが、目的 (1 台に 1e8) は**この手段では達成できない**

### ⚠️ 壁 4 で残る距離 (実測、「測っていない」の明示)

| 問い | 答え |
|---|---|
| 1e8 は**載る**か | ✅ **9.78 GiB** (単一プロセス、swap 増加 0) / slab 局所なら **1.302 GiB/rank** |
| 1e8 は**1 台に載る**か | ⛔ **8 rank 合計 10.42 GiB** (halo 重複で +3%) |
| 1e8 は**使える**か | ⛔ ⚠️ **1 step 6.9〜34.4 分** (実測: 1 反復 20.62 秒 × 20〜100 反復) ⚠️ **時間は分散でしか縮まない** (スレッド並列は帯域律速で、8 core で逐次より遅い) |
| 1 step を 1 秒にする並列度 | **約 400〜2000 rank 相当** ⚠️ **通信コストを無視した下限** |
| 越境 transport の slab 版 | ⛔ **未実装** = 複数台で 1e8 を回す唯一の道 |
| 対称な wire 形式変更 | ⛔ **原理的に不可視** (全 rank が同じ code を動かすため) ⇒ 閉じるには**別実装の peer** |
| 4 rank 以上のプロセス実行 | ⛔ 未実施 (schedule 上は 6 rank まで確認) |

### 第 13 increment (2026-10-01 夜、壁 3 の逆向き連成)

⚠️ **既存行は書き換えていません** 下記は追記です

| 話題 | commit | 到達点 |
|---|---|---|
| **壁 3 逆向き連成 (変形 → 熱)** | `bb2ffd0` | 塑性仕事 `W_p` を要素ごとに報告し、Taylor-Quinney 変換 (`PlasticHeating`) で温度上昇に、`deposit_plastic_heat` で `CoupledField` へ配分する 既存の `solve_with_eigenstrain` が読み戻すので **散逸 → 温度上昇 → 熱歪み → 変形 の経路が閉じた** 公開 API は追加のみ (snapshot **+28 / −0**) 変異試験 **実装 6/6 + 配線 4/4 red、生存 0** |

#### ⚠️⚠️ 第 7 increment の「作らない」判断との関係

第 7 increment は「**逆向き連成 / monolithic solver を作らない判断が固定小数点でも正当化された**」と記録している 本 increment はそのうち**前半だけを実施し、後半 (monolithic) は実施していない** 両者を分けた根拠:

| | 第 7 の判断 | 本 increment |
|---|---|---|
| **monolithic solver** | ⛔ 作らない (4 仮説が全て否定され、「より正しいと言える根拠がない実装」が残るため) | ⛔ **作っていない** 判断は維持 |
| **逆向き連成** | ⛔ 作らない (monolithic の正当化材料として不要、という文脈) | ✅ **作った** 正当化材料としてではなく**物理として** (金属の自己発熱は実在する効果で、solver 構成の選択とは独立) |

⚠️ **ただし副作用が 1 つある** 一方向連成は**単一 sweep で厳密**なので「partitioned か monolithic か」の問いが立たなかったが、**双方向にしたことでこの問いが再び生きる** `solve → deposit → solve` の素直な呼び出しは**副反復 0 回の陽的 partitioned 法**であり、収束解ではない 収束解が要るなら `coupled_iteration::run_sub_iteration` で駆動する (経路は doc で示し、driver 自体は入れていない)

⚠️ **壁 3 は閉じていない** 「完全」(全物理が在る) 側は進んだが「強連成」(monolithic) は満たさない **「壁 3 を越えた」とは書かない**

> ⚠️ **本行は 2026-10-02 時点で古い** 第 16 / 第 17 increment で `ThermalSoftening` と `step_thermoplastic` が入り、**分割型を固定点まで副反復する形**で「強連成」を満たした ただし **monolithic な連立系ではない** (同値性の根拠と成立条件は第 17 increment 参照)

#### ⚠️ 同日中の修正 — 保存則を書き込み先の不変量に合わせた

初版 (`bb2ffd0`) は `deposit_plastic_heat` の台帳を一律 `V_cell` で立てていたが、**書き込み先の `diffuse` が保存するのは別の量**だった peer (`ys-6d`) の指摘 → 私が独立に再現 → `ys-1f` が代数で確定

| 置いた場所 | 一律和 `Σ T` | lumped 和 `Σ 2⁻ᵇ T` |
|---|---|---|
| 境界 (0,2,2) | 8.000000000000 → 6.987522125244 (**−12.7%**) | 4.000000000000 → 4.000000000000 (**drift 0**) |
| 内部 (2,2,2) | 8.000000000000 → 9.757514953613 (**+22.0%**) | 8.000000000000 → 8.000000000000 (**drift 0**) |

(5³ 格子 h=1、熱 8 を 1 node、dt=1/16 / rate=1 で 6 step)

⚠️ `CoupledField` は **node 中心**で `cell = (max−min)/(n−1)` なので材料領域は `[min,max]` ちょうど ⇒ 面上の node は半 cell・辺 1/4・角 1/8 `diffuse` は mirror ghost (`T₋₁ = T₁`) の 7 点ステンシルなので `Σ 2⁻ᵇ T` を厳密保存する ⇒ 一律和で台帳を立てると (1) `diffuse` 1 回で恒等式が崩れ (2) **物体が格子境界に接する scene で最大 8 倍の過小配分**

✅ 修正後: `Σ_node 2⁻ᵇ·ΔT_node·c_v·V_cell = Σ_e β·W_p,e·V_e` ⚠️ **境界に接しない物体では重みが全て 1 なので挙動は不変** (回帰を oracle で明示)

⚠️⚠️ **oracle の穴がこの形で露呈した** — 初版の test / example は棒が格子の内部にしかなく、**境界 node を 1 度も使っていなかった** 保存則そのものは正しく測っていた (相対 1.28e-16) のに、**測った scene が欠陥を通らなかった** ⇒ 「保存を測った」と「保存が成立する」は別

⚠️ **退化軸 (`n = 1`) は拒否に回した** (`FemError::DepositGridHasDegenerateAxis`) 退化軸の cell size は 1 と定義されるので `V_cell` を体積に使うと台帳が「単位厚さ当たり」になり、要素の実 3D 体積と単位が合わない ⚠️⚠️ **恒等式は両辺が同じ `V_cell` を使うので閉じてしまう** ⇒ **保存 oracle では原理的に検出できず、誤るのは温度** (物体の真の厚さ倍) ⇒ 保存を主張せず拒否を assert する

⚠️ **mirror ghost は load-bearing** `tests/analytic_coupled_field.rs` が `cos(k x)` を離散作用素の厳密固有モードとして pin しており、これは mirror でのみ成立する ⇒ **copy ghost に替える案 (B) は oracle を壊すので選べない** (この確認で A/B の選択が閉じた)

⚠️ **未決で残す**: `CoupledField::diffuse` (mirror) と `ScalarField3D::diffuse` (copy) は **幾何が同一で ghost だけ違う** ので保存量が入れ替わる 揃えると `ScalarField3D` も閉形式固有モードを得るが、production consumer 5 module (`pressure` / `fracture` / `thermal` / `phase_change` / `erosion`) と `determinism_golden_f32.rs` の golden 再 pin (6 環境) が動く ⇒ **本 increment の範囲外、user 裁定待ち**

#### 本 increment で測った 3 件 (実装前は未測定)

| 測ったこと | 実測 |
|---|---|
| `W_p` は経路に依るか | ⚠️ **依る** `ε̄_p = 3.712871e-3` 固定のまま W_p は 1 step **0.978728** → 2 **0.972454** → 8 **0.966964** → 64 **0.964905** → 512 **0.964648** (連続形 0.964611) 超過 `(H/2) Σ Δε̄_k²` は step の 1 次 (excess×N が N≥4 で **5.859375e-3** 一定) |
| `ε̄_p` だけの oracle で足りるか | ⛔ ⚠️⚠️ **原理的に不足** `ε̄_p` は step 数に依らないので**仕事の積分則の誤りを検出できない** 実際、変異 M1 (硬化を `ε̄ⁿ` で評価する 1 つずれ) は `ε̄_p` を 1 bit も動かさない |
| `H = 0` の厳密性 | ✅ `W_p = σ_y·ε̄_p` が σ_y=2 では**全 N で 0 ulp** ⚠️ ただしこれは σ_y が 2 の冪で積が打ち切られないため 非 dyadic な σ_y=2.1 では 0 / 2 / **37** ulp (導出 bound 2 / 5 / 101) ⇒ **dyadic な定数だけで検証すると丸めの経路が一度も通らない** |

### 第 17 increment (2026-10-02、壁 3 強連成 — 副反復 driver)

⚠️ **既存行は書き換えていません** 下記は追記です

**壁 3 の 4 件が揃った** (A) 増分 API `5599c8a` / (C) 熱軟化 `dfdc6c3` / **(E) 副反復 driver + (B) 連成残差 oracle + (D) 拒否の歯** (本 increment)

| 話題 | 到達点 |
|---|---|
| **(E) driver** | `step_thermoplastic` が力学脚と熱脚を**固定点まで**交互に回す 写像は増分 `δT` 上で `δT_{k+1} = δT_k + ω (deposit(ΔW_p(T^n + δT_k; state^n)) − δT_k)`、⚠️ **確定済の `(ε_p^n, ε̄_p^n, T^n)` を全 sweep で固定** (Simo-Miehe の等温分解) 新設 `deposit_increment_heat` は増分 1 回分の `ΔW_p` を取る (`deposit_plastic_heat` は経路全体の累積なので副反復に使えない、同関数はこちらへの委譲になり挙動不変) |
| **(B) 連成残差 oracle** | 14 本 3 軸: **緩和は着地点を動かさない** (`ω = 1, 1/2, 1/4` が 11 桁一致 / sweep 数 10・34・76 で厳密単調増加 = 緩和が更新に届いている歯) / **熱脚は bit 厳密** / **力学脚も成立** (返した `δT` で力学だけ解き直して増分を再現) |
| **(D) 拒否の歯** | 緩和が `(0,1]` 外 / 床の分数が `(0,1)` 外 / 膨張係数が負 / 退化軸の格子 / 覆っていない場 / 予算 1 sweep |

#### ⚠️⚠️ 停止規則の床は絶対値では置けなかった

監視器の目標は初回**残差**に対する相対で、この写像は残差が 1 sweep で数桁落ちるので目標が算術の再現限界を下回る ⇒ 残差が雑音の中を徘徊し、⚠️ **その徘徊が「2 回上がって 1 回下がる」形になると wrap 検出器が `Fix128` の静かな桁あふれとして読む** (最初の実装は `from_raw(0, 64)` = 3.5e-18 の絶対床で、全 case が `ArithmeticWrapped` で落ちた)

実測した雑音は**答えに比例する** (2.5 桁にわたり相対 1.2〜2.1e-11):

| `c_v` (MPa/K) | 収束 `max δT` | 雑音 | 比 |
|---|---|---|---|
| `4` | `2.71e-3` | `≈5e-14` | `1.8e-11` |
| `1` | `1.08e-2` | `≈2.2e-13` | `2.0e-11` |
| `1/4` | `4.33e-2` | `≈9e-13` | `2.1e-11` |
| `1/64` | `6.82e-1` | `≈8.4e-12` | `1.2e-11` |

⇒ 床は**初回 sweep の堆積量に対する分数** (`2⁻³⁰` で実測の約 45 倍) 分数 0 は元の挙動に戻るので拒否

#### ⚠️ 非収束は発散でなくリミットサイクル、緩和は直さない

`c_v = 2⁻¹⁴` で初回 sweep が 177.7 K を堆積 ⇒ `σ_y(T)` の clamp を超え降伏応力も硬化も 0 ⇒ 応力 0 ⇒ 散逸 0 ⇒ **次の堆積が厳密に 0** ⇒ また 177.7 の**周期 2** 残差は平坦なので判定は `Stagnated` (`best == first`) ⚠️ **`Diverging` は誤り** (分裂を変えに行かせる) `NotConverged` も誤り (予算を増やしに行かせる)

⚠️ **緩和はこの分裂を直さない (実測)** `c_v = 2⁻¹⁰` で `ω = 1` が 24 sweep で収束する一方 `ω = 1/2` / `1/4` は 40 sweep の予算を使い切る ⚠️ **スカラーの縮小率は主張しない** — `observed_ratio` は残差の**大きさの比**なので符号を持たず、写像は格子全体に作用するので固有値 1 つでは説明できない

#### ⚠️ 点ごとの閉形式は置けない

収束した `δT` は非一様なので要素ごとに `σ_y(ΔT_e)` が違い、柔らかい要素は隣と同じ応力を担えないのでひずみが再配分され**局所軸ひずみは規定した平均ではなくなる** 実測: 要素 0 は **3.147794758 MPa**、自分の `ΔT = 1.419522305 K` での点ごと bilinear は **3.132705421 MPa** ⇒ **0.48% は平衡であって誤差ではない** 点ごと閉形式は一様場で妥当なので第 16 increment 側で固定している

#### 破壊試験

**15 変異すべて red** (実装 8 + 配線 7) ⚠️ **基準温度を無視する変異を倒すのは非零基準の oracle 1 本だけ** (他 13 本は `material_reference = 0` なので変異が恒等) 等価変異 1 件 (床を毎 sweep 再計算) は `|δT*|` が sweep 1 以降ほぼ一定 (2.7768 → 2.5835 = 7% 差) でどの scene でも判定が動かないことを実測して除外

公開 API は追加のみ (snapshot **+35 / −0**、`FemError::CoupledSubIterationFailed` は `#[non_exhaustive]` なので非破壊)

#### ⚠️ 配線 guard の baseline から 10 行が消えた — 分類して記録

**実配線 8**: `coupled_iteration::{MonitorVerdict, SubIterationReport, observe, observed_ratio, residual_norm_inf, sweeps}` + `coupled_field::{as_slice, max_value}` **巻き込み 2**: `sim_field::max_value` (呼出元は `#[cfg(test)]` の内側だけ) / `anomaly::observe` (production の `.observe(` は driver と既存 `run_sub_iteration` のみ) ⚠️ **後者 2 件は依然未配線** ⚠️ **`anomaly::observe` は baseline に 4 行重複していた** (key が `file::識別子` なので同名 method が N 個あると N 行になり、違反を消すには N 行すべて消す = N 件の記録が同時に失われる)

---

### 第 16 increment (2026-10-02、壁 3 強連成 — 増分 API と熱軟化)

⚠️ **既存行は書き換えていません** 下記は追記です

強連成 (monolithic) を名乗るために要る 4 件のうち **2 件**が landing した 残りは副反復 driver と連成残差 oracle

| 話題 | commit | 到達点 |
|---|---|---|
| **(A) 増分 API** | `5599c8a` | `solve_elastoplastic` の載荷 path ループの**中身を公開型に切り出した** 同関数の挙動と出力は 1 bit も変えていない (同関数自身が新 API で書き直されている) `ElastoplasticProblem` / `ElastoplasticState` / `ElastoplasticIncrement` / `ElastoplasticIncrementRequest` ⚠️ **`commit` が別呼びであることが副反復の前提** — sweep の内側で内部変数を確定させると Simo-Miehe の等温分解が崩れる (固定点写像は増分ごとに `(ε_p^n, ε̄_p^n, T^n)` を固定した上で定義される) oracle 7 本 |
| **(C) 熱軟化 + 熱歪み** | `dfdc6c3` | 熱歪み `ε_th = α ΔT I` を **return mapping の前に**弾性試行歪みから引き、`ThermalSoftening` が `σ_y(T)` / `H(T)` を return mapping に渡す ⚠️ **節点荷重として後から足すのではない** — 線形弾性では同じだがここでは別物で、降伏関数が熱補正後の応力を見ないと**高温要素が低温の基準で降伏する** 温度は**要素重心ごと**に `CoupledField` を sample、覆っていなければ `Err` oracle 9 本 / 変異 **実装 8 + 配線 4 = 12/12 red** |

#### ⚠️⚠️ 物理の訂正 — 「高温側の応力が低い」は硬化があると成立しない

非一様温度場の oracle に**大域的な順序**を assert して red になった ⚠️ **実装ではなく主張が誤り** 軟化で先に降伏して `ε̄_p` を多く積むので、硬化があると現半径 `σ_y(T) + H(T) ε̄_p` が**低温要素を上回りうる**

実測 (`examples/thermal_softening_bar.rs`、`w_y = 1/8`/K):

| ΔT | σ_y(T) | ε̄_p | von Mises |
|---|---|---|---|
| 0.5 | 1.8750 | 1.242e-3 | 3.1075 |
| 1.5 | 1.6250 | 1.316e-3 | **2.8458** |
| 2.5 | 1.3750 | 1.933e-3 | **3.0454** |
| 3.5 | 1.1250 | 2.062e-3 | 2.7742 |

⚠️ **Kuhn 分割の cell 内では x に単調減少、cell を跨ぐと跳ねる** ⇒ 見えている構造は **mesh のもので物理の要件ではない** ✅ **単調なのは法則が渡す降伏半径で、物体が落ち着く応力ではない** (応力は平衡の結果)

⇒ oracle は順序を撤回し、**sampler 自身の 3 つの厳密な言明**に差し替えた: spread > 閾値 / **相異なる重心位置の数 == 相異なる応力の数** (実測 6 == 6、少なければ使い回し・多ければ位置以外を key にしている) / 同一重心を共有する 2 要素は同一応力 (実測 6 pair)

#### ⚠️ 変異が明かした oracle 群の構造的な穴

**配線変異「重心を 1 つ使い回す」を倒すのは非一様場の oracle 1 本だけ** 他 8 本は**一様場なので構造的に見えない** ⇒ ⚠️ **温度場を使う oracle 群が一様場しか持たないと、sampler の配線は原理的に検証されない**

#### 公開 API

snapshot **+42 / −21** 削除はすべて `ElastoplasticIncrementRequest` が lifetime を得たことによる同型の置き換えと `step` の signature で、**同型は v1.4.0 に存在しない** (未 release、`git show v1.4.0:docs/PUBLIC_API_SNAPSHOT.txt | grep -c` が 0 行) 併せて `PartialEq` / `Eq` を落とした (温度格子を借用するので別の格子を指す 2 つが同じ数値を持つことは同一性ではない)

新規の公開入口 5 件は `ALLOW-UNWIRED` marker でなく **`examples/thermal_softening_bar.rs` 1 本**で配線した (guard は `examples/` を production に数える)

#### 残り (壁 3 は依然として閉じていない)

**(E) 副反復 driver** / **(B) 連成残差 oracle** / **(D) 拒否の歯** ⚠️ **停止規則は equality でなく floor** (同一載荷係数の増分を 2 度回すと `ΔW_p` が最大 **14 ulps** 動くことを実測) ⚠️ **sweep の中で `ε_p` を commit しない**

---

### 第 15 increment (2026-10-01 夜、壁 2 の適応 remeshing を誤差駆動にした)

⚠️ **既存行は書き換えていません** 下記は追記です

#### ⚠️⚠️ まず記録の訂正 — 「適応 remeshing は閉じた」は適合性のことだった

[[project_alice_physics_wall2_high_order_and_remeshing]] と本 ROADMAP の既存記述は `tests/refinement_conformity.rs` の `#[ignore]` 0 件を根拠に「適応 remeshing は閉じた」としていた ⚠️ **その 4 本 (`the_scenes_start_conforming` / `uniform_refinement_stays_conforming` / `graded_refinement_stays_conforming` / `propagation_costs_elements_and_the_count_is_bounded`) は全て「面の共有 = 適合性」の test で、解の誤差で駆動する適応の test は 1 本も無かった**

⇒ 閉じていたのは決定 (b) Rivara の伝播 = **適応の前提** 実測で欠けていたもの 4 件: 誤差推定子 0 件 / marking 0 件 / 細分の production caller 0 件 (`tests/` のみ) / ⚠️ **細分の判定が private `edge_to_split` の辺長なので呼び出し側から要素を指名できない**

⚠️ **「局所細分が適合性を保つ」(✅ 閉) と「解の誤差で駆動する適応」(⛔ 開) は別の読みで、欠けているものが違う** [[feedback_de_facto_satisfied_overstates_half_a_condition]] と同型

#### 入ったもの

| 話題 | 到達点 |
|---|---|
| **壁 2 適応 remeshing (誤差駆動)** | `StressTensor::complementary_energy_density` / `error_indicators_squared` (Zienkiewicz-Zhu) / `mark_bulk` (Dörfler) / `SdfTetMesh::try_refine_marked` / `AdaptiveConfig` / `AdaptiveSolution` / `solve_adaptive` / `RefineError::MarkCountDoesNotMatch` 公開 API は追加のみ (snapshot **+50 / −0**) 変異試験 **実装 12/12 + 配線 4/4 red、生存 0** |

#### ⚠️ payoff の実測 (これが「適応」を名乗る根拠)

鋼ブロック 3x2x2 cell、面の 1 cell に 64 MPa の牽引、`θ = 1/2`:

| run | tets | nodes | strain energy | 誤差 |
|---|---|---|---|---|
| reference (一様細分) | 2304 | 623 | 0.031673714 | — |
| 一様 1 段 | 576 | 175 | 0.022273470 | 9.400e-3 |
| **適応** | **292** | **98** | 0.027318884 | **4.355e-3** |

⇒ ✅ **節点 44% 少なく誤差は 2.16 分の 1** 判定量は歪エネルギー `½uᵀf` で推定子を一切使わないので、**推定子が自分を採点する形になっていない**

#### ⚠️ 設計上の論点 2 件 (実装前は未検討だった)

| 論点 | 実測 / 判断 |
|---|---|
| **`BoundaryConditions` は細分を越えられない** | 細分は頂点を append するので既存 index は有効 ⚠️ **だが拘束面に生まれた中点は誰も prescribe しないので面が部分的に自由になる** (`hanging_node_effect.rs` が線形場で **1.489e-1** の裂けとして測っている形と同じ、原因が mesh でなく境界条件) ⚠️ **伝播は半分しかできない** — 変位は両親が同軸 prescribe なら中点が平均を継げる (`1/2` は厳密) が **節点荷重は不可能** (点力は元の牽引の情報を持たない) ⇒ `solve_adaptive` は **closure `Fn(&SdfTetMesh) -> BoundaryConditions`** を取る |
| **指標の単調減少は滑らかな問題に限る** | ⚠️⚠️ **応力特異点があると総指標は増える** 実測で棄却した scene: **点荷重 `18.786 → 28.603`** / **牽引パッチ `1.093 → 1.759`** ⚠️ **推定子の欠陥ではない** (点荷重は 3 次元で解のエネルギーが無限 ⇒ 極限が存在しない) ⇒ 単調減少の oracle は全境界 Dirichlet の滑らかな scene に置き (実測 `0.2246 → 0.1762 → 0.1553 → 0.1177`)、特異な scene では **誤差が落ちること**を payoff oracle が見る |

#### ⚠️ 初版で 4 変異が生存した — 「値を押さえる oracle」が無かった

生存したのは `λ/(3λ+2μ)` 削除 / せん断を 1 回 / tie-break 反転 / 貪欲ループが 1 要素多く取る の 4 件 ⚠️ **初版の oracle はどれも値を見ていなかった** (零か非零 / 最悪要素の位置 / 単調性) ので、**別の妥当な norm でも全部通る**

塞ぎ方: norm を `complementary_energy_density` に切り出し、⚠️ **互いに別の項を露わにする 3 状態**で閉形式 pin — **一軸 `s²/E` / 純せん断 `2(1+ν)s²/E` / 静水圧 `3(1−2ν)p²/E`** (⚠️ **せん断だけが off-diagonal の係数 2 を見て、他 2 つだけが `λ/(3λ+2μ)` を見る**) + tie は index 昇順 + **目標に厳密到達する case** (4 等値 × θ=1/2、`>=` と `>` が分岐する唯一の入力) ⇒ 12/12 red

#### 壁 2 の残り

| 要素 | 状態 |
|---|---|
| 高次要素 P2 / P3 | ✅ landed (`6a84bfa` 他) |
| 適応 remeshing (適合性) | ✅ landed (Rivara の伝播) |
| **適応 remeshing (誤差駆動)** | ✅ **本 increment** |
| 超弾性 × 高次要素の非一様製作解 oracle | ⛔ `src gap` (第 14 increment で `ys-6d` / `ys-1f` が「原理的に不可能」から語彙を移した) ⚠️ **既に landed した P2/P3 の検証の深さの話で、要素が無いわけではない** |
| 適応を P2 / P3 へ | ⛔ 未着手 (`QuadraticMesh` / `CubicMesh` 側の細分が別作業) |

### 第 2 increment (2026-09-30、上表の続き)

| 話題 | commit | 到達点 |
|---|---|---|
| 物理間連成 | `d75d6dc` | `CoupledField` (Fix128 scalar 場) + `reconcile_mean` ⚠️ **opt-in で production caller は 0**、既定経路は 1 bit も不変 `diffuse` が次数 2.009 / 2.002 / 2.001 で収束 |
| Maxwell 第 2 | `3d3c377` | Gauss の法則 / 電荷保存 (`∇·E − ρ` が **0 ULP / 30 step**) / split-field PML (PEC 比 **1/18**) oracle 9 → **31 本**、破壊試験 21 変異すべて red |
| CFD 第 2 | `599c5a0` `cbe3def` | `FaceBc` 5 variant (inflow / outflow / 対称面) + 接線 no-slip を src へ 離散 Poiseuille と **3.206e-9**、収束比 **4.000 / 4.000** ⚠️ Ghia は **86.2% → 86.0%** (最大偏差は 0.04581 → **0.03162** で 31% 改善、差は演算子順序) |
| 多体 第 2 | `d6b58d4` | 破壊試験で素通りしていた 2 変異を scene 2 本で塞ぎ oracle 12 → **14 本** ⚠️ **実装でなく scene 側の穴**で `articulation` は変更なし |

⚠️ **oracle 設計で 3 つの盲点が実測された** (いずれも「green な理由が実装の正しさ以外にある」型)

1. **対称 / 退化した scene では staggered 格子の残り成分が未検証** CFD の `w` は準 2D で恒等的に 0、Maxwell の TM 配置は `Ex`/`Ey`/`Hz` を bit 厳密に 0 に保つので、どちらもゴースト / 係数の変異が素通りした 対処は同じ閉形式を 6 通りの向きで回すこと
2. **等方な構成では軸 / index の配線が観測できない** 非等方な寸法にしても足りず、**ラベルの巡回置換に対する不変性** (`x→y→z→x` で bit 一致) を測る必要がある
3. **誤差項が相殺する parameter 点が存在する** CFD の `dt = dx²/(8ν)` では空間離散化誤差と演算子分離誤差が厳密に相殺し、連続解にぴたり乗る (両方の項が未検証になる) test の doc で禁止点として明記

**次の increment**: Armaly (1983) backward-facing step の再付着長さ突合 (前提は揃った) /
移流への壁の扱い (接線 no-slip は現状粘性項のみ) / 反射係数 `R` の直接測定 (入射波と反射波の分離) /
`CoupledField` の実配線 (現状 opt-in で production caller 0) / 演算子分離誤差 `−G dt` を消すか (別 y/n)

## 前回位置 (2026-09-29): v1.5.0 (`Mat3Fix::polar_rotation` = 極分解の回転因子)

**v1.5.0 = 極分解の追加のみ** (`Mat3Fix::polar_rotation` / `max_abs_component` / `PolarError`) engine の既存 path は 1 bit も動いていない (golden f32 13 scenario 不変、全 24 test target green) 共回転 (co-rotational) FEM の前提として入れた primitive で、⚠️ **共回転そのものはまだ入っていない** (下表の該当 Phase 参照)

## 前回位置 (2026-09-17): v1.4.0 (`alice-det-math` 0.2、golden 13 scenario bit 不変)

**v1.4.0 = alice-det-math 0.2.0 追従** (縮約丸めの簡素化 / f32 fdlibm atan2 / SIMD kernel 書き直し = alice-sdf 3.1.0 が同じ crate を使う前提の性能回収) engine の値は 1 bit も動いていない (golden f32 13 scenario 不変)

## v1.3.0 (2026-09-16) (`det_math` → `alice-det-math` crate 切り出し、crates.io publish)

**v1.3.0 = `det_math` を [`alice-det-math`](https://crates.io/crates/alice-det-math) 0.1.0 として独立 crate 化し `pub use` で再 export** (API 不変、golden f32 13 scenario bit 不変) alice-sdf 3.1.0 が同じ crate で評価するので `SdfField` 境界が crate 跨ぎで bit-exact になる (下表「SDF 境界の Fix128 化」の前提条件) 副産物: `det_math` が `std` gate 不要 (software sqrt)、初回 CI が x86 vs ARM の default NaN 符号差 (huge trig 引数 / cbrt overflow) を検出して修正

## 前回位置 (2026-09-15): v1.2.0 (セルフレビュー 3 round + 解析解 oracle 対応)

**v1.2.0 = セルフレビュー (別環境 Linux x86_64 実測、Round 1-3) 由来の重大修正 + `tests/analytic_physics.rs` 10 本** (詳細 [`CHANGELOG.md`](../CHANGELOG.md) `[1.2.0]`):
- 修正 (挙動変化、golden 再 pin): damping を frame 単位化 (default で重力が効いていなかった) / XPBD λ 累積 (iteration 依存剛性) / sphere contact normal 逆向き / substep 毎 collision detection + contact λ 累積 (5 m/s 衝突が 700 m/s になっていた)
- 修正 (bit 互換): `Fix128::sqrt` digit recurrence 50× / MSRV 1.70.0 → **1.85** 実証 + `resolver = "3"` / `crate-type` cdylib+staticlib / `panic = "unwind"` / SIMD dead code 除去 / README perf 表実測化 / lib.rs claim
- **規律**: 解析解 oracle test を default config で必須 (substeps / iterations を変えて結果不変を assert)、golden hash は「変化検出」であって「正しさ」ではない
- 1.2.0 内で完了 (旧 Backlog): engineering oracle 3 batch (thermal/fatigue 5 + solid 34 + fluid 39、src bug 8 + 9 件修正) / `iterations` 既定 1 (Small Steps) / contact solver の ALICE-TRT 3.2.0 parity 同期 / det_math atan2・acos・asin・tan・tanh / `Fix128::checked_div` / FFI panic 隔離 31 fn / loom model test / replay + alice-db 実動作化
- 残 (Backlog、claude-config `project_todos_active`): oracle 未着手 module (`rolling_contact` / `fracture` / `sdf_destruction` / `soft_body_cut` / `layer_adhesion` / `warp_risk` / `kinematic_loop` / `sdf_force` / `anisotropic_friction` / `physics2d` / `fluid_netcode` 他 utility) / mutation score ≥ 80 % (core 6 + a* 5 module) / default damping 0.99 再考 / module 階層分け (core vs 教科書 wrapper)

### 2.0 に送る semver-breaking 設計 (1.x では doc のみ、実装は 2.0 branch)

| 項目 | 1.x での状態 | 2.0 設計 |
|--|--|--|
| **SDF 境界の Fix128 化 (案 A)** | `SdfField::distance(f32, f32, f32) -> f32` を `det_math` で bit-exact 化 (案 B、1.1.0) 境界の f32 丸めは user closure と alice-sdf evaluator 側に残る | `trait SdfField { fn distance(&self, p: Vec3Fix) -> Fix128 }` + alice-sdf に `Real for Fix128` evaluator (alice-sdf 側の libm 76 file を `det_math` 化した後) `sdf_collider` / `sdf_ccd` / `sdf_character` / `sdf_sph` の f32 変換を全廃、`ClosureSdf` は `Fn(Vec3Fix) -> Fix128` |
| **`RigidBody` force accumulator** | 外力は `velocity` 直接書換 or `apply_impulse` (frame 単位、substep 毎の力は表現不能) `wind_zone` / `buoyancy_zone` / `sdf_force` は velocity を直接更新 | `RigidBody { force: Vec3Fix, torque: Vec3Fix }` + `apply_force` / `apply_force_at` / `clear_forces`、solver が substep 毎に `v += force · inv_mass · h` を積分して frame 末に clear `#[non_exhaustive]` でない `RigidBody` への field 追加 = major |
| **`Div` の `Result`** | ゼロ除算は `ZERO` (契約 doc、`checked_div` で `None`) | `PhysicsError::DivisionByZero` 追加 (`PhysicsError` は `#[non_exhaustive]` でないため major)、solver 内の除算を `checked_div` に置換して `Result` を伝播 |
| **`ErosionConfig` の rate law field** | `Water` 1.5× / exposure decay 5 /s / 速度指数 1/1/0/2 は `pub const` (`WATER_PREFACTOR` / `EXPOSURE_DECAY_PER_S`) + module doc 表 | `velocity_exponent: f32` / `prefactor: f32` / `exposure_decay: f32` を `ErosionConfig` に追加 (全 pub field struct への追加 = `constructible_struct_adds_field` major) 既定は現 const 値 |
| **`Carreau` の分数 flow index** | `half_exponent: i32` (整数) + `viscosity_with_index(γ̇, n)` で分数 n を評価 (1.2.0) | `flow_index: Fix128` に置換、`half_exponent` 削除 |
| **`PhaseChangeConfig` 熱容量** | `latent_heat_*` は K 相当 (c_p = 1 正規化、1.2.0 enthalpy 法) | `heat_capacity: f32` 追加で J/kg 単位を受ける |

## v1.1.0 released 2026-09-15 (1.0.0 stable + 全 module bit-exact)

**alice-physics v1.1.0 = crates.io published 2026-09-15** — `Published alice-physics v1.1.0 at registry crates-io`
- crates.io URL: https://crates.io/crates/alice-physics/1.1.0 (docs.rs build ✅、Documentation / Homepage link 追加)
- GitHub Release: https://github.com/ext-sakamoro/ALICE-Physics/releases/tag/v1.1.0 (assets 15)
- Release commit: `81ba0ab` (1.1.0 = det_math + golden_f32、1.0.1 は crates.io skip)
- 前 release: v1.0.0 stable 2026-09-14 (`ca3adae`、https://github.com/ext-sakamoro/ALICE-Physics/releases/tag/v1.0.0)
- Install: `cargo add alice-physics` (semver-locked、1.x 内 breaking なし)

**全 9 v1.0 Items 完了** (詳細は [`CHANGELOG.md`](../CHANGELOG.md) `[1.0.0]` entry):

| Item | 完了 | 実装 |
|--|--|--|
| B: Public API surface freeze | ✅ | 32 module × 512 pub items audit、133+ downgrades、snapshot −795 items、solver_tgs* Option C + 7 struct `#[non_exhaustive]` |
| C: cargo-semver-checks hard-gate | ✅ | `security-audit.yml` `continue-on-error: true` 解除 |
| D: `#![deny(missing_docs)]` | ✅ | 0 warning 実測 |
| E: Determinism CI 6-env matrix | ✅ | 6 platform (macOS ARM/x86 + Linux ARM/x86 + Windows + WASM) × 31 test (9 golden hash + 22 semantic invariant)、bit-exact 一致確認済 |
| F: Fuzz coverage 8/8 | ✅ | fuzz_step / collision / deterministic_roundtrip / joint / cfd / ccd / trimesh / structural |
| G: MSRV policy | ✅ | `rust-version = "1.70.0"` + README EN Serde-style policy (1.2.0 で 1.85 に訂正: 1.70 宣言は未検証で偽だった、CI msrv job が実 compile) |
| H: Ecosystem contracts freeze | ✅ | [`ECOSYSTEM_CONTRACTS.md`](ECOSYSTEM_CONTRACTS.md) に 5 partner (TRT / SDF / Bamboo / Anima / Kinematics) frozen API |
| I: Migration guide | ✅ | [`MIGRATION_0.x_TO_1.0.md`](MIGRATION_0.x_TO_1.0.md) per-module 削除項目 + `#[non_exhaustive]` 影響 + 5 step checklist |
| J: crates.io publish | ✅ | preview.4-8 → **1.0.0 stable** promoted |

**snapshot 累計進化**: 20,201 → **19,406** (−795 items across audit campaign)
**test 累計**: 1364 lib + 9 golden + 22 semantic = **1395 test 全 pass on 6 platforms**

Post-1.0 stability contract: 19,406-item public surface は 1.x で breaking なし、`cargo semver-checks` hard-gate で PR 時 enforcement、frozen partner contracts (Item H) 未変更

---

## 過去 milestone (v1.0 到達までの歴史)

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

### ✅ v0.14.0 (superseded by v1.0.0 stable) (推定 1-2 週間) — API surface audit 前半 + 残 preview 済

Preview 4 wave が landed 済み、残作業:

- **B. Public API surface freeze 前半** — priority module (net / character / SDF 系) の `pub` → `pub(crate)` audit 着手 (144 pub mod のうち、`net_prediction` / `character*` / `sdf_*` 系から)
- **新 example 追加** — ragdoll / SPH / joint / character / BFECC velocity / BiCGStab pressure の代表 example (現状 7 → 14 個目標)
- **B4 (新規発覚) 対応**: `wasm` × `ffi` mutual exclusion (pre-existing `compile_error!`) の設計見直し — CI で `--all-features` を除外し続けるか、feature 設計を分割するか判断

自己採点 target: 品質 90/100 (現状 100/100 optimization scorecard は維持、public API 完成度で -10)

### ✅ v0.15.0 (superseded — all items landed within v0.14.0-preview.X chain, promoted directly to 1.0.0)

- **B. Public API surface freeze 後半** — 残 module audit + `#[non_exhaustive]` 戦略適用 (struct / enum の一部) + `#[deprecated]` alias で v1.0 名確定 (v0.x で名前変えたい API に marker + alias、v1.0 で確定名だけ残す)
- **C. cargo-semver-checks / cargo-public-api CI 通し** — `.github/workflows/security-audit.yml` に既に semver-checks job あり、拡張して cargo-public-api snapshot を repo に commit、PR diff で API surface 変化を可視化
- **F. Fuzz coverage 拡張** — 現状 `fuzz_collision` + `fuzz_step` の 2 target → joint / SDF CCD / trimesh / `cfd_solver` / `structural_solver` で 5-8 target 追加、24h 実行 crash 0 実績を CHANGELOG に記載

### ✅ v0.16.0 (superseded — Item E determinism CI + Item H ecosystem contracts landed within audit campaign, promoted directly to 1.0.0)

- **✅ E. Determinism CI 6 環境 matrix 完了 (2026-09-14)** — Phase 1 + Phase 2 landing 済 [`docs/DETERMINISM_GOLDEN_TESTS.md`](DETERMINISM_GOLDEN_TESTS.md) + `tests/determinism_golden.rs` に 8 fixture (rigid body 3 + Phase 2 で joint/cloth/fluid/SDF CCD/trimesh 5 追加) + `.github/workflows/ci.yml` に 6 platform matrix (macos-latest / macos-15-intel / ubuntu-latest / **ubuntu-24.04-arm** / windows-latest + 独立 `wasm-test` job で `wasm32-wasip1` + wasmtime) Mac aarch64 と wasm32-wasip1 で 8 fixture 全 hash bit-exact 一致確認済
- **✅ H. Ecosystem 契約 freeze 完了 (2026-09-14)** — [`docs/ECOSYSTEM_CONTRACTS.md`](ECOSYSTEM_CONTRACTS.md) 起草済 (~215 行、5 partner の frozen API 一覧: TRT `GpuSolverBridge` / SDF `SdfField` / Bamboo concrete-type / Anima concrete-type / Kinematics reserved-post-1.0) + Freeze semantics (semver-minor 可否) + CI enforcement + partner responsibilities
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

### ✅ v0.14.0-preview.8 / promoted → v1.0.0 stable (2026-09-14) — 全 audit + ADR-003 revision で γ 直行採択

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
- **✅ Item C 出力側 完了 (2026-09-14)**: semver-checks の hard-gate 化 (commit `3638bd1`、`security-audit.yml:245-252` の `continue-on-error: true` 解除、job name も informational → hard-gate 改称) 次 PR 以降 breaking API 変更を検知したら CI red gate として block
- **F 24h 実行**: 8 target 全てで 24h fuzz run + crash 0 実績 (v1.0-rc validation の一環)
- ✅ `0.14.0-preview.8` から **γ 直行で v1.0.0** へ (下記 決定ログ OQ 参照、`0.14.0` stable / `0.16.x` は skip)

### ✅ v1.0.0-rc.1 / rc.2 — skip (γ 直行、2026-09-14)

- **✅ I. Migration guide 執筆完了 (2026-09-14)** — [`docs/MIGRATION_0.x_TO_1.0.md`](MIGRATION_0.x_TO_1.0.md) 起草済 (~340 行、per-module 削除項目テーブル + `#[non_exhaustive]` 影響 + Cargo.toml migration + 6-environment determinism promise + post-1.0 stability guarantees + 再曝露リクエスト手順)
- rc.1 / rc.2 の crates.io pre-release publish と 4 週間 feedback 期間は **実施せず** — 「決定 (revised): γ (直行)」で `0.14.0-preview.8` → `1.0.0` を直接 bump (実運用 downstream 0 の状態で RC を出しても feedback が得られないため)
- 実 downstream (ALICE-Bamboo / ALICE-Anima / SBR ゲーム側) の 1.0 対応は 1.0.x patch line で受ける

### ✅ v1.0.0 stable (2026-09-14 crates.io published)

- **semver 契約発動済** — `alice-physics = "1"` で下流 pin 可能、breaking change は必ず major bump
- Release commit `ca3adae`、GitHub Release v1.0.0、詳細は本 file 冒頭「現在位置」

### ✅ v1.1.0 (2026-09-15、f32/f64 module 30 個の cross-platform bit-exact 化 = 案 B)

- 3 案 (A 全面 Fix128 化 / B libm 排除 / C crate 分離) から B を採用 (memory `project_alice_physics_det_f32_plan`)
- `det_math` module (f32 8 関数 + f64 3 関数、IEEE-exact な基本演算 + bit 操作のみ) を追加、production 16 + test 4 site の `libm` 呼出を置換、`clippy.toml` `disallowed-methods` で再混入を CI red 化
- `tests/determinism_golden_f32.rs` 13 scenario / 29 module を aarch64 で pin、x86_64 (Rosetta) / wasm32 local 一致 → CI 6 platform
- README「Determinism scope」3 tier → 1 tier、残る境界は user closure と ALICE-SDF evaluator (alice-sdf 側で `det_math` 整合を別途、案 A は `Real for Fix128` 経由で 2.0 向け)

### 🔧 v1.0.1 patch (2026-09-15、厳密評価 7 件の対応)

- **soundness**: `parallel` graph coloring の 64 色 overflow (同一 body が同 batch に同居 → aliasing `&mut`) を可変長 bitset + static body 除外 + disjoint `debug_assert!` で修正
- **API**: `build_islands` の `assert!` → `PhysicsError::InvalidConstraint`
- **CI**: clippy を default + 全 native feature set で `-D warnings` に昇格 (lint 26 件修正)、`msrv` job (1.70.0 実測) 追加
- **FFI**: `alice_physics_version()` の hardcode `"0.6.0"` → `CARGO_PKG_VERSION`、free-fall golden raw pair を pin
- **docs**: 決定論保証の範囲を Fix128 core に限定 (README EN/JP「Determinism scope」表)、`Fix128::mul` の wrapping / floor 挙動を doc + test 化
- **repo**: `fuzz/target/` 256 file を git 追跡から除外

### ✅ 線形弾性 FEM (tet P1) + mesh 適合化 + CG solver (2026-09-29、`9eab3c1`〜`b165569` 12 commit)

`Constraint::Stress` を無次元ヒューリスティックから実応力に置き換える前提を整えた 消費者が存在しなかった `SdfTetMesh` に、初めて solver が付いた

- **`linear_elastic_fem`** — tet P1 で `K u = f`、要素ごとの Cauchy 応力と von Mises、mm / N / MPa、`Fix128` の四則 + sqrt のみで cross-target bit 一致 `feature = "std"` gate (`sdf_fem_mesh` の頂点 intern が `HashMap` を使い `alloc` に無いため)
- **oracle 12 本を実装より先に red で実測** (`tests/analytic_linear_elastic_fem.rs`) + **収束研究** (`tests/analytic_fem_convergence.rs`、`#[ignore]`/release — 片持ち梁 4 段細分、次数 1.395、Richardson 極限が目標の 0.22% 以内)
- **mesh を適合化** — 非適合の原因は 3 件 (cube 5-tet 分割が全 cube 同一 pattern / `generate_marching_tets` の頂点 dedup ゼロ / 切断多面体の quad 対角線が固定) `tests/mesh_conformity.rs` が面センサスで CI 固定 ⚠️ **patch test は非適合を原理的に検出できない** (実測 3.6e-15 MPa で通る) ので使わないこと
- **CG solver** — `FemError::Stagnated` を `NotConverged` と別 variant に (停滞窓は反復数に比例)、`FemSolution::effective_relative_tolerance` で実効許容差を報告 (相対残差の床は `2⁻³² / ‖b‖` でメッシュ細分とともに上がる)、Jacobi 前処理は選択可能だが既定 `None` (形状由来の悪条件では対角がほぼ一様で効かない)
- ⚠️ **残る前提**: `generate_marching_tets` の sliver (最小二面角が細分で 7.25° → 4.59° と悪化) — 頂点スナップと最小二面角 gate は別途

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

### 第 14 increment (2026-10-02、壁 4 の「HPC 並列」を実測で満たした)

⚠️ **既存行は書き換えていません** 下記は追記です 第 6 increment の「残っているもの」のうち
**「全長 buffer を維持しているので 1e8 規模には依然載らない」が本 increment で解消**しました

| 話題 | commit | 到達点 |
|---|---|---|
| **帯局所の越境 slab** | `8e00bce` | rank が `MacGrid` を一度も作らず scene から自分の band だけを seed し、所有層の fold を rank 0 に返す経路 (`SlabRunKind::Banded`) 参照解は rank 0 が子の退出後に **1 本だけ**作るので、全プロセスが全領域を常駐させる必要がない 変更は `mod tests` 内のみで公開 API の差分なし |
| **1.34e8 cells を 8 プロセスで** | 同 | ⚠️⚠️ **512³ = 134,217,728 cells を 8 プロセスに分散し、単一プロセス解と bit 一致** arm64 (M2 Pro 32 GiB) **142.87 s / peak 12.35 GiB** と x86_64 (Xeon Gold 5315Y 44 GB) **182.91 s / `VmHWM` 20.89 GiB** の 2 アーキで独立に確認 128³ / 256³ / 320³ / 384³ / 448³ も同経路で bit 一致 |
| **覆い判定の強化** | 同 | face 側が `assert!(faces > 0)` だったのを **`faces == n·u_plane + n·v_plane + (n+1)·cell_plane = 3n²(n+1)` の厳密一致**に ⚠️ **`> 0` では「ある rank が自分の face を 1 つも fold しない」「ある plane を 2 rank が fold して別の plane は誰も fold しない」が通る** (cell 側は元から `n³` 厳密で、同じ関数の中で片方だけ緩かった) |

#### ⚠️ 破壊試験で出た 2 件の実 gap (既存 65 test がすべて green のまま通っていた)

| 変異 | なぜ素通りしたか | 足した oracle |
|---|---|---|
| `fold_fix` が `Fix128` の小数語 (`lo`) を捨てる | ⚠️ **fold の doc 自身が「低位 bit の差を捉えるためのもの」と書いているのに**、halo を落とす既存 oracle は**高位語も動く**摂動しか作っていなかった | `the_fold_separates_values_that_differ_in_their_lowest_bit` (最下位小数 bit / 最上位小数 bit / 整数語 の 3 通りで `assert_ne!`) |
| 渡された反復数を無視して定数を使う | ⚠️ **全 case が同じ反復数を渡していたので、引数が配線されていなくても通る** 大規模 run は反復数を環境から取るため、512³ で別の回数を解いて「一致」と報告しうる | case 表を `(n, ranks, scene, iterations)` 4 列にして **1 と 9 を混ぜ**、「全 case が定数と同じなら fail」の guard を追加 |

変異 21 件 (実装 12 + 配線 9) のうち **20 件 red** 残る 1 件は等価変異で、⚠️ **test では殺せない** — 累積器 `cells` と `values` は独立で、cell の走査を face の後ろへ動かしても**両者に入る列が変わらない** 判定は読みでなく実測 (両版に probe を挿して fold の生値 8 行が完全一致) で裏を取った

#### ⚠️ 残っているもの (「測っていない」の明示)

- ⚠️ **複数ノードの MPI は未測定** 測ったのは **1 ホスト上の 8 OS プロセス** (loopback TCP、場のデータに共有メモリを使わない分散メモリ) `SlabTransport` 抽象の背後なので socket 先を変えればノード越しになる形だが、**そこは測っていない** 第 6 increment の「MPI は入れていない」判断は変えていない
- ⚠️ **アーキをまたいだ fold 値の直接突合はしていない** 各機で「分散 == その機の単一プロセス解」を確認したのみ (test が fold 値を出力しないため)
- ⚠️ **律速は並列度でなく反復数のまま** 1 step を実用精度まで解くには GS の反復数が `O(n²)` で増えるので、越境 slab を multigrid に適用するのが次の軸 (第 6 increment の記録と同じ)

### 壁 1〜4 の到達点と残件 — `origin/main = bf14216` で実測 (2026-10-02、`ys-6d`、既存行は書き換えていない)

⚠️ **壁名はすべて複合語なので、語ごとに要求を分解して現況を当てる** 「壁 N を越えた / 越えていない」の 2 値で語ると、達成済の半分が見えなくなる (または未達側だけを見て「何も進んでいない」とも言える)

| 壁 | 要求 A | 要求 B | 越えたか |
|---|---|---|---|
| **1** 幾何学的非線形・大変形 + 自己接触 | **幾何学的非線形・大変形** ✅ | **自己接触** ✅ | ✅ **2/2** |
| **2** 高次要素 P2/P3 + 適応 remeshing | **高次要素 P2/P3** ✅ | **適応 remeshing** ✅ | ✅ **2/2** |
| **3** 完全強連成マルチフィジックス | **完全** (連成が双方向) ✅ | ⚠️ **強連成** (monolithic) ⛔ | ⛔ **1/2** |
| **4** 数億要素クラスの HPC 並列 | **数億要素クラス** ✅ | **HPC 並列** ✅ | ✅ **2/2** (限定つき、下表) |

> ⚠️ **本表の壁 3 の行は 2026-10-02 時点で古い** 第 17 increment で「強連成」が満たされ **8/8** になった (最新の表は本 file 冒頭の第 17 increment)

#### 各行の根拠 (grep / test 実行の出力、推測なし)

| 要求 | 実測 |
|---|---|
| 壁 1 幾何学的非線形 | `src/linear_elastic_fem.rs` **4132 行** / `corotational` 42 ヒット / `hyperelastic` 38 ヒット |
| 壁 1 自己接触 | `src/cloth.rs` に `edge_edge` **56 ヒット** `tests/analytic_self_contact.rs` **15 passed / 0 failed / 0 ignored** (頂点-面 `cb2fc36` 貫通 7 → 0、辺-辺 `6dbfbd0`〜`d097411` 貫通 22 → 0、破壊試験 13/13 red 生存 0) |
| 壁 2 高次要素 | `src/quadratic_elastic_fem.rs` **1631 行** (`corotational` 12 / `hyperelastic` 25) / `src/cubic_elastic_fem.rs` **2285 行** (同 12 / 25) ⇒ ⚠️ **P2/P3 に非線形機構まで配線済** (`8747b62`) `analytic_quadratic_fem` 9 / `analytic_cubic_fem` 8 / `analytic_quadratic_hyperelastic` 8 / `analytic_cubic_hyperelastic` 9 すべて 0 failed 0 ignored |
| 壁 2 適応 remeshing | `tests/refinement_conformity.rs` **498 行 / `#[ignore]` 0 件 / 4 passed 0 failed** (`the_scenes_start_conforming` / `uniform_refinement_stays_conforming` / `graded_refinement_stays_conforming` / `propagation_costs_elements_and_the_count_is_bounded`) `tests/mesh_quality.rs` 7 passed |
| (おまけ) 壁 2 時間項 | `src/dynamic_fem.rs` **1363 行** (`8b3ecdd` 質量行列 整合/集中 + Newmark-β、破壊試験 30/30 red 生存 0) `analytic_dynamic_fem` 15 passed |
| 壁 3 完全 (双方向) | 熱 → 機械: `ThermalExpansion` / `solve_with_eigenstrain` が residual に (`04e8259` 他) 機械 → 熱: `ElastoplasticSolution::dissipation` / `PlasticHeating` / `deposit_plastic_heat` (`bb2ffd0`) `analytic_thermoelastic` 6 / `analytic_thermoelastic_channel` 6 / `analytic_plastic_dissipation` 18 すべて 0 failed |
| ⚠️ 壁 3 強連成 (monolithic) | `src/` の `monolithic` **4 ヒットはすべて doc comment** 逐語 `//! # Why an instrument, and not a monolithic solver` ⇒ **solver は 1 行も無い** `run_sub_iteration` (`coupled_iteration.rs`) は器具として公開済だが **production caller 0** ⇒ `solve → deposit → solve` は**副反復 0 回の陽的 partitioned 法**で収束解ではない |
| 壁 4 数億要素クラス | 512³ = **1.34e8 cells** を越境で実走 (別途 1 プロセスで 2.62e8) |
| 壁 4 HPC 並列 | **8 プロセスに分散して単一プロセス解と bit 一致** arm64 (M2 Pro 32 GiB) 142.87 s / peak 12.35 GiB、x86_64 (Xeon Gold 5315Y 44 GB) 182.91 s / `VmHWM` 20.89 GiB |

⚠️ **repo 全体で残る `src gap` / `src bug` の `#[ignore]` は 1 件のみ** — `tests/armaly_backward_step.rs:1973` (CFD Gartling 再付着長) ⇒ **壁 1〜4 由来は 0 件** (他の `src bug: …` ヒットは過去の運用を説明する doc comment)

#### ⚠️ 残件 — 「越えた」行にも限定が付く

| 壁 | 残件 | 性質 |
|---|---|---|
| **1** | `thickness` 充足は未達 (到達点は「非貫通」まで、違反 ON 2 / OFF 3) | 次の層 |
| **1** | ⚠️⚠️ **共通モード故障への歯が 1 層しかない** — 述語が死ぬと修復も計器も同時に盲目で、統合 15 本すべて green のまま貫通する (`0 = 清潔` 型の不変量は自分の計器の死を検出できない) 防波堤は「`OFF → 1` を assert する lib test 2 本」だけ | ⚠️ **検査体系の穴** |
| **1** | 辺-辺分離が solver 支配の部分集合で未成立 (ON 4.556e-2 < OFF 7.093e-2 = 0.64 倍、1 未満) / 閉形式 scene が純並進で 3 次項に歯が立たない | 次の層 |
| **2** | ⚠️ **超弾性 × 高次要素の次数分離 oracle が未実装** ⚠️ **「原理的に閉じない」ではない** (下記の訂正) | ⚠️ **src gap** |
| **2** | 共回転接線が要素ごと (重心の `F`) で求積点ごとでない 収束する最大角度は **20° まで実測**、20° 超は「失敗」でなく「測っていない」 | 限定の明示 |
| **2** | P3 suite **489 s** の CI 予算が未決 ⚠️ **`runtime only:` に落とすと 13 変異中 12 件の red が週次に移る** (`quality-deep.yml` は `push` trigger を持たない) | 運用判断 |
| **2** | 塑性の `F = Fe·Fp` は「拡張」でなく新規 / 組み立てた接線行列 + 直接法分解が無い | 次の層 |
| **3** | ⚠️ **monolithic solver (= 「強連成」) が無い** 副反復 driver も無い | ⛔ **壁の未達部** ⚠️ **2026-10-02 の第 17 increment で解消** (副反復 driver `step_thermoplastic`、monolithic な連立系ではない) |
| **3** | ⚠️⚠️ **連成経路の保存量が 2 段で食い違う** (下記の実測) | ⚠️ **実装の欠陥** |
| **4** | ⚠️ **複数ノードの MPI は未測定** (測ったのは 1 ホスト上の 8 OS プロセス、loopback TCP、場のデータに共有メモリを使わない分散メモリ) | 測っていない |
| **4** | ⚠️ **アーキをまたいだ fold 値の直接突合は未実施** (各機で「分散 == その機の単一プロセス解」を見たのみ) | 測っていない |
| **4** | ⚠️ **律速は並列度でなく反復数** (GS は `O(n²)`) / **分散経路は `pub(crate)` なので公開 API から使えない** | 次の層 |

#### ⚠️⚠️ 壁 3 の実装の欠陥 — `deposit_plastic_heat` と `diffuse` が別の量を保存する (実測 + 代数で確定)

`CoupledField` は node 中心格子で、`diffuse` は 7 点ステンシル + **mirror ghost** (`T₋₁ = T₁`) の陽的 Euler

| 置いた場所 | 量 | 初期 | `diffuse` 6 step 後 |
|---|---|---|---|
| 境界 `(0,2,2)` | 一律和 `Σ T` | 8.000000000000 | ⚠️ **6.987522125244 (−12.7%)** |
| 同 | **lumped 重み和** `Σ T·2^−b` (b = 境界軸数) | 4.000000000000 | ✅ **4.000000000000 (完全不変)** |
| 内部 `(2,2,2)` | 一律和 | 8.000000000000 | ⚠️ **9.757514953613 (+22.0%)** |
| 同 | **lumped 重み和** | 8.000000000000 | ✅ **8.000000000000 (完全不変)** |

一方 `deposit_plastic_heat` は doc で `Σ_cell ΔT_cell · c_v · V_cell = Σ_e β W_p,e V_e` (**一律 `V_cell`**) を主張する ⇒ ⚠️ **deposit 直後は成立するが `diffuse` を 1 回呼ぶと崩れ、体が格子境界に接する scene では角 node で最大 8 倍の過小**

⚠️ **代数でも確定した** (1D の telescoping、3D は軸ごとの積): mirror ghost は node 0 の更新を `r(2T₁ − 2T₀)` にするので、重み `w₀ = 1/2` を掛けると flux 対が項ごとに相殺する ⇒ **node 中心 + mirror ghost の陽的 Euler は双対 cell (境界で半分) の体積重み和を保存する有限体積法そのもの** 任意の `n ≥ 2` / 任意 step 数 / **`dt` と `rate` に依らず** / **非等間隔 `hx≠hy≠hz` でも**成立 (安定性とは独立で、不安定でも保存する)

⚠️⚠️ **「反射 ghost」には 2 流儀があり保存量が違う** — mirror `T₋₁ = T₁` (node 上で勾配 0、2 次精度) は **lumped 重み和**を保存 / copy `T₋₁ = T₀` (半 cell 外で勾配 0、1 次精度) は **一律和**を保存 ⇒ 実測 (lumped が保存) は **mirror 実装の証拠** ⇒ 修正は 2 択: **(A) deposit を lumped 重みに合わせる** (BC の 2 次精度を保つ、推奨) / **(B) ghost を copy に替えて deposit の一律 `V_cell` を正当化する** (BC が 1 次に落ちる)

⚠️ **`sample` (重心での trilinear 補間) は保存作用素ではない** (補間は転置であって逆ではない) ⇒ エネルギーの帳簿は**場の側** (`Σ V_dual T`) で付け、FEM に返す `ΔT_e` は温度 (intensive) として扱う

⚠️ **oracle の穴**: `tests/analytic_plastic_dissipation.rs` は **deposit 直後しか見ておらず**、example / test の scene は**格子境界に接していない** (格子 node `x=−1..7` / `y,z=−1..5`、棒は `x 0..4` / `y,z 0..2`) ⇒ **境界 node の目減りを 1 度も通っていない** ⇒ ✅ **連成経路では「各段が保存する」でなく「段をまたいで同じ量が保存する」を oracle にする** (`deposit → diffuse` を通した後の重み付き総量)

#### ⚠️ 訂正 — 壁 2 の「超弾性 × 高次要素の次数分離は原理的に閉じない」は**強すぎた**

旧記述 (第 7 increment): 「非線形では求積が厳密にならないので閉形式 oracle は `F` 一様に限られ、一様変形はアフィン場 = P1 の空間に入る ⇒ 『P2 では通らず P3 で通る』型の分離が原理的に作れない」

⚠️ **1 行で言うと**: **超弾性の `P` は `F` の多項式なので、多項式変位場に体積力を逆算した製作解は被積分関数が多項式になり、十分次数の求積則で離散方程式を厳密に満たす** ⇒ **一様 `F` に限られるのは現行の求積則の制約であって原理ではない**

Neo-Hookean は `P = μF + [K(J−1) − μ]·cof(F)` (Mooney-Rivlin も `cof` の積で多項式) `u` を p 次に取ると `F` は p−1 次、`cof F` は 2(p−1) 次、`J` は 3(p−1) 次 ⇒ 剛性被積分関数 `P·∇N` は **P2 で 4 次 / P3 で 8 次** ⇒ 体積力 `b = −Div P` を整合節点荷重で与えれば、**求積がその次数まで厳密なら Galerkin 解は `u` と一致**する (線形側と同じ論法)

⇒ ⚠️ **「作れない」の本当の理由は実務上の 3 点**: (1) **現行 P3 の求積は 5 次** (dyadic 24 点則) で 8 次に足りない ⇒ solver 側の規則を上げるか、**収束率 oracle に切り替える** (製作解に対する誤差が P1 `h²` / P2 `h³` / P3 `h⁴` で落ちることを refine 2〜3 段で見る、求積が離散化誤差より高次なら足りる = 標準 MMS) (2) 体積力は `BoundaryConditions.loads` が節点荷重なので、test 側で **整合節点荷重 `f_a = ∫ b N_a dV` を閉形式で計算して渡す** (barycentric monomial 公式 `∫ λ₁^a λ₂^b λ₃^c λ₄^d = a!b!c!d!·3!/(a+b+c+d+3)!·|T|` で有理厳密、⚠️ 分母に奇数因子が出るので `Fix128` では ulp 丸め = 許容差 oracle になる) (3) `J > 0` を領域全体で保つ振幅に取る (大きいと要素反転)

⚠️ **構造格子では空振りする**ので摂動格子が必須 (既知: 高次要素の厳密再現 oracle は構造格子上では低次要素も通る)

⇒ **記録の語彙を「原理的に閉じない」から「製作解 + 8 次求積 (または収束率 oracle) が未実装」= `src gap` に移す**

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

**決定 (当初)**: α (v0.13.0 → v0.14.0 → ... → v0.16.x → rc.1 → rc.2 → v1.0.0 の段階昇格) を採用

**代替案 (当初)**:
- **γ (直行)**: Cargo.toml `0.13.0 → 1.0.0` に bump、README も 1.0 表記に、Session 4 module も全部「v1.0.0 included」に格上げ → reject: 実運用ドッグフーディング未実施、public API surface freeze 未完
- **β (RC 短縮)**: 現在の main を `v1.0.0-rc.1` として freeze publish、feedback 経て v1.0.0 → reject: Unreleased backlog は既に v0.13.0 で吸収済だが、新 module (client_prediction 等) の実運用検証がない状態で RC 出すと feedback の意味が薄い

**根拠 (当初)**: 「downstream crate が今から `alice-physics = "1"` で pin して 6 ヶ月 breaking change なしを我々が保証できるか?」 = 現状 n 判定 (実運用検証未了)、ALICE-Bamboo と ALICE-Anima の実運用で 3 ヶ月連続 breaking change なしを先に達成することが 1.0 コミットの根拠になる (semver 契約は約束ではなく実績の追認)

### ADR-003 revision (2026-09-14): γ 直行に変更

**背景**: 2026-09-13 の集中 session で B / C / D / E / F / G / H / I / J の全 9 v1.0 Items を landing 完了 6 iteration の audit campaign 中、以下 6 sibling repo に対して毎 iter downstream survey 実施 (`rg 'alice_physics::'`):
- ALICE-Bamboo (実運用)
- ALICE-Anima (実運用)
- ALICE-TRT (`impl GpuSolverBridge for TrtSolverAdapter` 実装済)
- ALICE-SDF (`impl SdfField for CompiledSdfField` 実装済)
- ALICE-LOL / Yoin / text-to-print-ios (使用)

**測定結果**: 6 iteration + Final iteration で downstream breakage **累計 0 件**、`#[non_exhaustive]` + `solver_tgs*` pub(crate) 化含む全 API 変更で "無し実装" への降格のみ、既存 downstream 呼び出しは全て残 pub items 経由で温存

**決定 (revised)**: **γ (直行)** に変更 rc.1/rc.2 skip、v0.14.0-preview.8 → v1.0.0 直接 bump

**新根拠**:
1. **実運用 pre-validation 済** — 従来「RC 期間の未知の外部 user への feedback 窓」で得るはずのシグナルは、ecosystem 内 6 sibling で毎 iteration ごとに既に得ている
2. **API surface 完成** — Iter 1-6 + Final の audit で 32 module 512 pub items 監査、133 downgrades、`#[non_exhaustive]` 7 struct hedging 完了
3. **Determinism 保証済** — Item E で 6 platform (macOS ARM/x86 + Linux ARM/x86 + Windows + WASM) × 31 test (9 golden hash + 22 semantic invariant) bit-exact 一致確認
4. **Hard-gate 有効** — Item C の cargo-semver-checks が hard-gate mode で今後の breaking を block
5. **Ecosystem contract freeze** — Item H で 5 partner の frozen API 明文化 (`docs/ECOSYSTEM_CONTRACTS.md`)
6. **Migration guide 完備** — Item I で 0.x → 1.0 の per-module 削除項目 + 対応手順を提供 (`docs/MIGRATION_0.x_TO_1.0.md`)

「6 ヶ月 breaking change なし保証」の semver 契約は、pre-1.0 の 6 iteration audit で実測 0 breakage を実績として引き受け可能な状態

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
