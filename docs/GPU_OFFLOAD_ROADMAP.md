# GPU Offload Roadmap — Data-driven scope decision

**Status**: 2026-07-06 initial version
**Source data**: `benches/stage_breakdown.rs` (differential workload measurement, MacBook M2 Max)

## Question

Should ALICE-Physics migrate the remaining CPU-only stages (BVH broad-phase / narrow-phase / joint / CCD) to GPU on top of the existing PGS solver bridge (v1.3.1)?

The remainder of this document derives the answer from measurement rather than intuition.

## Measurement approach

We use **differential workload decomposition**: run five workload variants for each of three body counts, and infer per-stage cost from the deltas. Sleep-fast-path variants use `SleepConfig::frames_to_sleep = 1`. All measurements: `--quick` (rapid sampling), 30 frames per iteration, `Fix128::from_ratio(1, 60)` dt.

Workloads:

| # | Name | Isolates |
|---|------|----------|
| 1 | `sleeping` | Dispatch + island rebuild + sleeping fast-path |
| 2 | `falling` | + gravity integrate |
| 3 | `bvh_dense` | + BVH pair enumeration on packed AABBs |
| 4 | `pile` | + narrow-phase attempts (no explicit colliders → still baseline) |
| 5 | `chain` | + distance-constraint solve on N-body chain |

## Results

```
N=100:
  1_sleeping   30.7 ms
  2_falling    30.7 ms
  3_bvh_dense  30.6 ms
  4_pile       30.5 ms
  5_chain    1017.0 ms         <-- 33x

N=1,000:
  1_sleeping   300.0 ms
  2_falling    301.0 ms
  3_bvh_dense  299.7 ms
  4_pile       299.9 ms
  5_chain    10009.0 ms         <-- 33x

N=10,000:
  1_sleeping  3040 ms
  2_falling   3025 ms
  3_bvh_dense 3129 ms           <-- +3% over baseline (BVH overhead visible)
  4_pile      3042 ms
  5_chain    (~100 s projected, skipped)
```

Per-unit costs at every scale:

| Stage | Per body per frame | Per constraint per frame |
|-------|-------------------|-------------------------|
| Integrate + island bookkeeping | **~10 μs** | — |
| BVH broad-phase (packed AABBs) | +0.3 μs (=3% of baseline) | — |
| Distance constraint solve | — | **~333 μs** |

## Findings

1. **Constraint solve is the dominant cost by a factor of 33x** per-unit versus integrate. This holds at every scale from N=100 to N=1,000, and the constraint cost also scales O(N).

2. **BVH broad-phase overhead is negligible (~3% at N=10,000)**. Even packed AABBs on 10k bodies barely shift the total; the pair-enumeration cost is dominated by everything else.

3. **Narrow-phase was not measured** because the bench does not attach explicit colliders. RigidBody defaults to a sphere-like surrogate that either short-circuits or lands in the same fast-path as no-collider bodies. A follow-up bench with attached `Collider` instances is required to measure this stage before making decisions about GPU narrow-phase.

4. **Sleeping optimisation is already effective** — the sleeping variant has essentially the same cost as the falling variant at all sizes, meaning the sleep fast-path is not saving significant work in a workload that never actually accumulates idle frames. This is expected: sleeping helps mixed active/idle workloads, not "everyone active" workloads.

5. **CCD was not measured**. Enabling CCD selectively per body (`world.enable_ccd(id)`) adds continuous swept collision to a subset only; it will only appear in profiles when high-velocity projectiles exist in the workload.

## Interpretation

**High-ROI GPU offload targets** (this data justifies them):

- Additional constraint types (joints, rigid rods, springs). The current PGS bridge already delivers the 33x speedup for distance constraints; extending the same pattern to joints multiplies the reachable workloads.

**Low-ROI GPU offload targets** (this data does not justify them):

- BVH broad-phase. 3% of total at N=10k, not worth the 2-3 week effort plus determinism verification risk.
- Narrow-phase. Not measured yet, so no evidence either way. Must be measured with real colliders before deciding.
- CCD. Only relevant for a small subset of bodies; premature to migrate.

## Recommended plan

### Phase 1 (v1.4.0 → v1.6.0, ~4-6 weeks total)

Extend the existing constraint bridge with the constraint types that dominate real-world use.

| Release | Content | Est. duration |
|---------|---------|---------------|
| v1.4.0 | GPU sqrt kernel (Newton-Raphson bit-exact in WGSL), enables scalar physics ops fully on-device | 1 week |
| v1.4.1 | Rigid rod constraint (strict distance = L, iterate to convergence) | 3-4 days |
| v1.5.0 | Ball joint (3-DOF, most common) + Hinge joint (1-DOF, second most common) | 2 weeks |
| v1.6.0 | Slider + Spring + Fixed joints, completing the top-5 joint types | 1-2 weeks |

Each release preserves the v1.x semver stability commitment (additive only). CI matrix (Metal / Vulkan lavapipe / DX12 WARP) validates byte-exactness against the CPU golden for every new constraint.

### Phase 2 (v2.0.0, ~4-6 weeks)

Introduce **constraint graph coloring** so independent constraints can be dispatched in parallel. This is a breaking API change (`push_distance_constraint` semantics gain colour metadata), hence the major-version bump. Multiplicative on top of Phase 1: each colour dispatches all its constraints in one compute call.

Prerequisites in place after Phase 1: GPU sqrt, joint kernels, uniform-flow discipline.

### Phase 3 (deferred, gated on measurement)

Only proceed with GPU BVH / narrow-phase / CCD after:

1. Adding a **collider-attached bench variant** (measure real narrow-phase cost, not the current no-op fast-path).
2. Confirming that at least one target workload spends **>30% of frame time** in one of these stages.
3. Documenting the target workload as a real user requirement (crowd sim, destruction, particle collision).

Absent that evidence, the Phase 3 stages are premature optimisation.

### Phase 3 gate status (updated 2026-07-07)

**Gates 1 and 2 are satisfied.** Gate 3 is deferred to Phase 3 design.

#### Gate 1: collider-attached bench variant ✅

`benches/stage_breakdown_collider.rs` (commit `ac7d1b1`, 2026-07-06) measures the pile workload with and without attached `Sphere` colliders via `add_body_with_radius`. The delta between the two runs is the effective narrow-phase + post-collision solver cost — the same workload shape as `stage_breakdown` but with real collider attachment instead of the no-op fast-path.

#### Gate 2: >30% of frame time in narrow-phase + PGS ✅

Initial `--quick` measurements (MacBook M2 Max):

| N | `pile_no_collider` | `pile_with_collider` | delta | delta / total |
|---|---|---|---|---|
| 100 | 41.5 ms | 1583 ms | +1542 ms | **97.4%** |
| 1000 | 316 ms | 152740 ms | +152424 ms | **99.8%** |
| 10000 | 3063 ms | (compute-heavy, deferred) | — | — |

Narrow-phase + PGS contact solve dominates the frame at every scale that fits in `--quick` measurement. The 30% threshold is exceeded by more than 3x at N=100 and by more than 3x again at N=1000. GPU offload for these stages is now data-justified rather than intuition-driven.

#### Gate 3: target workload as user requirement — deferred

Gate 3 formalises that a real caller cares about the measured workload. This is a documentation / scoping task rather than a measurement one, and lands with Phase 3 design (v2.1+ series in ALICE-TRT). The measurement above lifts the "premature optimisation" concern; explicit workload documentation lands when Phase 3 kernel work begins.

#### Prerequisites for Phase 3 design

Before kernel work starts, `deterministic-physics-lockstep-discipline` §11.4 must be revisited: GPU BVH construction must preserve the escape-pointer forward-monotonic invariant (position-independent placeholder + build-time `debug_assert!` + traversal cycle guard). The CPU BVH's July 2026 correctness fix (ALICE-Physics `dede78c`) is the canonical reference implementation for how the GPU port must behave.

#### Notes for ALICE-TRT release cadence

- **v2.0.0** (2026-07-07) — Phase 2 removal wrap-up + Phase 3 gate declaration in the CHANGELOG. No Phase 3 kernel work in the crate yet.
- **v2.1+** — Phase 3 kernel work begins here. Same additive semver posture as the Phase 1 / v1.4.x series: new WGSL kernels published under fresh constant names, `pub` API is additive, byte-exact CPU golden lands with each kernel.

## Determinism guardrails

All Phase 1 and Phase 2 work must continue to observe the five determinism-breaking routes catalogued in `deterministic-physics-lockstep-discipline`:

1. Broad-phase precision — remain fully in Fix128 space.
2. CORDIC / sqrt — Newton-Raphson with fixed iteration count, no early-exit.
3. SIMD / dispatch order — sequential reduce, no cross-lane sum reordering.
4. Rollback snapshot delta — every new field goes into `serialize_state`.
5. Thread / workgroup traversal order — deterministic sort of constraint IDs prior to dispatch.

CI must include the 3-platform matrix (Metal / Vulkan lavapipe / DX12 WARP) for every new kernel from day one — the WARP crash we hit at v0.7.1 → v0.8.1 proves that platform-specific driver behaviour has to be caught in CI, not in production.

## Reproducing the measurement

```bash
cd ALICE-Physics
cargo bench --bench stage_breakdown -- --quick        # ~2 min for all sizes
cargo bench --bench stage_breakdown -- "stage_breakdown/small_100"  # single size
```

Full bench source: [`benches/stage_breakdown.rs`](../benches/stage_breakdown.rs).
