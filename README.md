# ALICE-Physics

A deterministic physics engine for Rust. The rigid-body core runs on 128-bit
fixed-point numbers, so the same inputs produce the same bits on every CPU,
compiler and operating system.

English | [日本語](README_JP.md)

[![crates.io](https://img.shields.io/crates/v/alice-physics.svg)](https://crates.io/crates/alice-physics)
[![docs.rs](https://img.shields.io/docsrs/alice-physics)](https://docs.rs/alice-physics)
[![MSRV](https://img.shields.io/crates/msrv/alice-physics)](#minimum-supported-rust-version)
[![CI](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Physics/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/alice-physics.svg)](#license)

ALICE-Physics is built for cases where bit-exact replay is a requirement:
rollback and lockstep netcode, server-side replay verification, search and
planning loops that need to branch and rewind a simulation, and reproducible
engineering or research batches. Next to the rigid-body solver it
includes a set of engineering modules (FEM, CFD, heat transfer, composites,
3D-print checks), all held to the same cross-platform determinism.
<!-- claim-test: golden_sim_field -->
<!-- claim-test: determinism_freefall -->

It is not a drop-in replacement for a floating-point game engine. Fixed-point
arithmetic costs time, and scenes with thousands of interacting bodies at
60 fps are outside its design point.

## Contents

- [Installation](#installation)
- [Example](#example)
- [Determinism](#determinism)
- [Reset, observation and rollback](#reset-observation-and-rollback)
- [What is included](#what-is-included)
- [Vehicle dynamics](#vehicle-dynamics)
- [Validation and known defects](#validation-and-known-defects)
- [Cargo features](#cargo-features)
- [Bindings](#bindings)
- [Performance](#performance)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [Related crates](#related-crates)
- [License](#license)

## Installation

```sh
cargo add alice-physics
```

Without the standard library (requires `alloc`):

```sh
cargo add alice-physics --no-default-features
```

## Example

A body falling under gravity for one second. The same code is the crate-level
doctest in `src/lib.rs`, so it is compiled and run by `cargo test`.

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

Runnable programs live in [`examples/`](examples/). Good places to start:

| Example | Shows |
|---------|-------|
| [`basic_physics`](examples/basic_physics.rs) | world setup, bodies, stepping |
| [`world_api_tour`](examples/world_api_tour.rs) | every `PhysicsWorld` setter, query and event drain |
| [`rollback_netcode`](examples/rollback_netcode.rs) | snapshot, rollback and checksum verification |
| [`world_snapshot_branching`](examples/world_snapshot_branching.rs) | whole-world snapshot, branching from it, restoring a world that holds an SDF collider |
| [`joint_limits_and_breaking`](examples/joint_limits_and_breaking.rs) | joint types, limits, motors, breakable joints |
| [`cloth_simulation`](examples/cloth_simulation.rs) | XPBD cloth |
| [`cfd_smoke_plume`](examples/cfd_smoke_plume.rs) | the integrated CFD solver |
| [`print_full_safety`](examples/print_full_safety.rs) | the 3D-print safety pipeline |

```sh
cargo run --release --example rollback_netcode
```

## Determinism

Every module is bit-exact across platforms, for one of two reasons:

| Tier | Modules | Why the bits agree |
|------|---------|--------------------|
| **Fix128 core** | rigid-body solver, joints, distance and contact constraints, BVH broad-phase, GJK / EPA, CCD, sleeping, scene I/O, netcode snapshot and rollback, neural controller, and every module whose public API uses `Fix128` / `Vec3Fix` / `QuatFix` | Pure integer arithmetic. Pinned by `tests/determinism_golden.rs`. |
| **`f32` / `f64` field modules** (28: `acoustic_wave`, `aeroelasticity`, `anomaly`, `convex_decompose`, `erosion`, `fracture`, `gpu_sdf`, `phase_change`, `piezoelectric`, `pressure`, `privacy`, `rolling_contact`, `sdf_adaptive`, `sdf_ccd`, `sdf_character`, `sdf_collider`, `sdf_destruction`, `sdf_fem_mesh`, `sdf_manifold`, `sdf_sph`, `sdf_wind_field`, `sim_field`, `sim_modifier`, `sketch`, `spherical_terrain`, `thermal`, `thin_wall`, `transient_thermal`) | IEEE 754 `+ - * / sqrt` and fused multiply-add are exact on every target Rust supports. Every transcendental (`sin`, `exp`, `ln`, `powf`, …) goes through [`alice-det-math`](https://crates.io/crates/alice-det-math), re-exported as `alice_physics::det_math`, never the platform `libm`. `clippy.toml` turns a stray `libm` call into a CI error. Pinned by `tests/determinism_golden_f32.rs`. |

The module list in the second row is generated by `scripts/f32_modules.py`,
and CI fails when it disagrees with `src/`. Nine more modules carry floats only
across an I/O boundary (FFI, Python, replay and similar) and do no simulation
arithmetic in `f32`.

Both golden suites run in CI on macOS (ARM and x86), Linux (ARM and x86),
Windows and `wasm32-wasip1`.
<!-- claim-test: test_determinism_golden_hash -->

**Your own code.** `ClosureSdf` takes a user closure. The solver stays
deterministic, but inside the closure you need to call
`alice_physics::det_math::{sin, exp, …}` instead of `f32::sin` and friends.

**Not covered.** Targets that do not follow IEEE 754 for the basic operations,
such as 32-bit x86 built for x87 (`i586`, no SSE2), and builds that use
fast-math style flags.

**Determinism is not correctness.** Matching bits say every peer computes the
same numbers, not that the numbers are right. That is checked separately; see
[Validation and known defects](#validation-and-known-defects).

## Reset, observation and rollback

Because a step is bit-exact, a `PhysicsWorld` can be used as a step function
that a search or planning loop calls many times: try an action, read the
result, rewind, try another. These APIs support that use:

| API | What it gives you |
|-----|-------------------|
| `PhysicsWorld::reset_world()` | every field back to the state `PhysicsWorld::new` produces, so the same start and the same inputs give the same bits on a second run |
| `observe_body` / `observe_bodies` | a typed `BodyObservation` (position, velocity, rotation, angular velocity, `sleeping`, `in_contact`) for a goal check to read, instead of parsing the state blob |
| `serialize_state` / `deserialize_state` | save and restore a branch point; restoring is refused when the bodies in the world (mass, shape, filter, material) differ from the ones that were saved, even if the count matches |
| `snapshot_world` / `from_world_snapshot` / `restore_world` | save the whole world (bodies, joints, constraints, colliders, force fields, materials, filters, events, sleep state, broad-phase tree, warm-start caches, overflow flag) in one versioned blob with a checksum, and restore it into a new or existing world; every later step is bit-identical to the original's. A bad blob is rejected with a `WorldSnapshotError` that says why |
| `step_n(n, dt)` | `step(dt)` run `n` times, the same call the Python `step_n` and WASM `stepN` bindings make |
| `overflow_detected()` | reports when `Fix128` arithmetic left its range, so a diverged run is not mistaken for a valid one; the flag survives a rollback |
| `Vec3Fix::checked_*` / `checked_length_scaled` / `try_normalize_scaled` | a length or dot product whose square would leave the `Fix128` range (`\|v\| ≥ 2^31.5 ≈ 3.04e9`) returns `None` instead of wrapping; the scaled versions return the correct length and direction there and are bit-identical to `length` / `try_normalize` inside the range |
| `netcode::SimulationChecksum` | a checksum derived from the same bytes as the saved state |

[`examples/world_auditor_observation.rs`](examples/world_auditor_observation.rs)
shows reset and observation. The contracts are pinned by the `tests/wm0*_*.rs`
files.

**Scope.** `serialize_state` covers rigid bodies only: their motion, sleep
state and the overflow flag; the caller rebuilds joints, force fields, filters
and materials, as in rollback netcode. `snapshot_world` covers every
`PhysicsWorld` field except what is code rather than data: SDF fields, pre-solve
hooks, contact modifiers and a GPU bridge stay in the world being restored into,
and their counts must match. The field-by-field table is in the docs of
`PhysicsWorld::snapshot_world`. Motors, character controllers, cloth, fluids,
FEM and vehicles are not `PhysicsWorld` fields and are saved by the caller. The
crate provides the step function, not a search algorithm.

## What is included

Every public module, grouped by area and with a one-line summary, is listed in [`docs/MODULES.md`](docs/MODULES.md). API details are on
[docs.rs](https://docs.rs/alice-physics).

| Area | Highlights |
|------|-----------|
| Rigid bodies | XPBD solver with an optional temporal Gauss-Seidel backend, sleeping and islands, CCD, rollback-ready state serialization |
| Collision | GJK / EPA, linear BVH or persistent dynamic AABB tree broad-phase, box, sphere, capsule, cylinder, cone, ellipsoid, torus, wedge, convex hull, compound, triangle mesh, height field, SDF colliders; world ray queries against those shapes (`PhysicsWorld::cast_ray`) |
| Joints | ball, hinge, fixed, slider, spring, D6, cone-twist, plus pulley, gear, weld, rack-and-pinion and mouse joints; breakable joints and PD motors |
| Soft bodies | XPBD rope and cloth (with self-collision), position-based fluids, FEM-XPBD deformables, cutting |
| Gameplay | character controller, vehicles (a simple model and a per-wheel dynamics model with tyres, brakes, ABS, road surfaces and weather), ragdolls, IK bridge, client-side prediction, deterministic RNG with Gaussian draws, contact events, simulated lidar / contact / IMU sensors |
| Solid mechanics | linear-elastic FEM on P1 / P2 / P3 tetrahedra, corotational large rotation, J2 plasticity, hyperelasticity, thermo-mechanical coupling, adaptive refinement, beams, buckling, fatigue, composites |
| Fluids and fields | MAC-grid CFD with several pressure solvers, RANS / LES turbulence closures, VOF and level set, SPH, compressible flow, heat transfer, Maxwell FDTD with per-cell materials |
| Aerodynamics | standard atmosphere (ISA 1976, up to 20 km), wing lift and drag with stall, rotor thrust and torque |
| Molecular dynamics | Lennard-Jones, Morse, Coulomb and screened Coulomb pair potentials with cutoff and shift, velocity Verlet with a periodic cell list (minimum image) |
| Crowds | social force model of pedestrian motion: driving term, repulsion weighted by the view angle, body force and sliding friction in contact, walls |
| 3D printing | material database, thin-wall and overhang checks, warp risk, layer adhesion, print orientation, a combined safety pipeline |
| 2D | a separate 2D XPBD engine with its own shapes and joints |

**Domain decomposition.** The CFD pressure projection can run as `z` slabs
that exchange one halo layer, with results bit-identical to the single-process
solve for any number of ranks. This has been measured on one host only (up to
eight processes over loopback TCP). Runs across several hosts have not been
done, and there is no MPI backend.

Not every module is wired into `PhysicsWorld`. The table counts public modules by where their
items are actually called from; calls from `examples/` do not count. CI measures it with
`scripts/integration_levels.py`; each module's label is in the Integration column of
[`docs/MODULES.md`](docs/MODULES.md) and the details are in
[`docs/integration-levels.md`](docs/integration-levels.md).

<!-- integration-levels: summary -->
| How a module is used | Modules |
|----------------------|--------:|
| step: runs when `PhysicsWorld` steps | 24 |
| world API: used through another `PhysicsWorld` method | 13 |
| binding: reached from the C ABI, Python or WebAssembly bindings | 2 |
| standalone: a Rust API you call yourself; `PhysicsWorld` does not call it | 125 |
| unused: no caller outside tests | 0 |

## Vehicle dynamics

`vehicle_dynamics::DynamicVehicle` is a car model in which each wheel acts on
the chassis at its own contact point.

- suspension and tyre forces are applied per wheel, so steering produces yaw
  and braking or cornering shifts load between axles and sides
- each wheel has a spin state driven by drive torque, brake torque and the
  tyre force; brakes can lock a wheel, and ABS holds the slip ratio near a
  target
- tyre forces come from a brush model or a Magic Formula model, limited by a
  friction ellipse
- the road is a plane, a slope, a height field, a triangle mesh or an SDF;
  grip is the road material times a weather factor (dry, wet, snow, ice),
  with a hydroplaning loss on flooded roads
- a locked wheel holds the car in place on a slope below the static friction
  limit, and slides at the kinetic coefficient above it
- engine torque curve, gearbox, engine braking, open or locked differential,
  and aerodynamic drag and lift
- `vehicle_dynamics::scenario` — run several `DynamicVehicle`s in one
  `PhysicsWorld` from input tracks or a per-frame control function (fixed
  index order); following metrics: time to collision (`gap / closing speed`,
  `None` when not closing) and time headway; a stopping-distance meter
  (horizontal path from brake-on until forward speed reaches zero); lossless
  replay — initial state plus per-frame `DriverInput`s stored as raw `Fix128`
  reproduce every chassis and wheel state bit for bit, with a versioned byte
  encoding; malformed / truncated recordings and mismatched scenarios are
  rejected with a typed `ReplayError`

The older `vehicle::Vehicle` is unchanged. It drives on a flat plane at
`ground_height` and applies the sum of all wheel forces at the centre of mass,
so it has no per-wheel load transfer, wheel lock or tyre model. Use
`vehicle_dynamics` when stopping distance or cornering has to follow the
physics.

```rust
use alice_physics::vehicle_dynamics::surface::{FlatGround, RoadCondition};
use alice_physics::vehicle_dynamics::{DynamicVehicle, DynamicVehicleConfig, Environment};

let mut car = DynamicVehicle::new(DynamicVehicleConfig::passenger_car());
let road = FlatGround { height: Fix128::ZERO };
let condition = RoadCondition::dry_asphalt();
let env = Environment { condition: &condition, wind: None, time: Fix128::ZERO };
car.input.brake = Fix128::ONE;
// every frame, before world.step(dt):
car.update(&mut world.bodies[chassis], &road, &env, dt);
```

[`examples/vehicle_dynamics.rs`](examples/vehicle_dynamics.rs) runs this setup
and checks locked-wheel stopping distances on dry, wet and icy roads against
`v0² / (2 μ_k g)`, compares braking with and without ABS, and holds a car on a
slope. [`examples/vehicle_scenario.rs`](examples/vehicle_scenario.rs) runs two
cars in one lane, brakes the follower on time to collision, and replays the
recorded run bit for bit. The closed-form tests are in
`tests/analytic_vehicle_dynamics.rs` and `tests/analytic_vehicle_scenario.rs`.

**Known limitations**

- In full sliding, the brush model turns the force towards the sliding
  direction; the Magic Formula model does not (it scales the pure-slip forces
  onto the friction ellipse).
- Wheel forces are applied as one impulse at the start of each frame. The
  suspension is stable only while `(ω_n dt)² + 2 c dt / m_share < 4`
  (`ω_n = √(k / m_share)`, `m_share` the mass carried by one wheel); a light
  chassis on stiff springs at a large `dt` oscillates with growing amplitude.
- `HeightField` itself does not apply `origin.y` and has a known normal
  defect at the grid border. The height-field road follows the field's own
  `sample_height` (so the road sits at the height it returns, without
  `origin.y`) and computes its normal with one-sided differences at the
  border, so a wheel probe is not affected by the border defect.

## Validation and known defects

Analytic and reference tests in `tests/` compare results against closed-form
solutions or published reference data (textbook formulas, Ghia et al. for the
lid-driven cavity, and others). Golden hashes only detect change, so they are
kept separate from these tests.

[`docs/oracle-status.md`](docs/oracle-status.md) is generated from `tests/` and
lists every test by status. It includes the **known defects**: tests that are
kept red on purpose (`#[ignore = "known defect: …"]`) until the implementation
is fixed. Read that list before relying on a module for production numbers.

[`docs/integration-status.md`](docs/integration-status.md) is generated from
rust-analyzer's resolved references and shows, for every public item, whether
anything outside the tests reaches it, and whether only an example does.

Modules without a reference test, such as `warp_risk` and `layer_adhesion`
(empirical fits), implement the cited equation but are not validated
predictions. [`docs/MODULES.md`](docs/MODULES.md) marks them.

## Cargo features

<!-- readme-sync: features -->
| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | yes | Standard library. Disable for `no_std` (needs `alloc`). |
| `simd` | | SSE2 paths for `Vec3Fix` operations on `x86_64`. Results are bit-identical to the scalar paths. |
| `parallel` | | Parallel constraint solving with Rayon over graph-coloured batches. |
| `ffi` | | C ABI for Unity, Unreal Engine and other hosts. Cannot be combined with `wasm`. |
| `wasm` | | WebAssembly bindings through `wasm-bindgen`. Requires `std`; cannot be combined with `ffi`. |
| `python` | | Python bindings (PyO3 + NumPy). |
| `gpu-solver-bridge` | | `GpuSolverBridge` trait for external GPU solver backends. No cost when off. |
| `neural` | | Deterministic neural controller using [`alice-ml`](https://crates.io/crates/alice-ml). |
| `replay` | | Replay recording and playback using [`alice-db`](https://crates.io/crates/alice-db). |
| `analytics` | | Simulation profiling using [`alice-analytics`](https://crates.io/crates/alice-analytics). |

## Bindings

| Target | Where | Notes |
|--------|-------|-------|
| C / C++ | [`include/alice_physics.h`](include/alice_physics.h) | `--features ffi`; builds `cdylib` and `staticlib` |
| Unity (C#) | [`bindings/AlicePhysics.cs`](bindings/AlicePhysics.cs) | P/Invoke over the C ABI |
| Unreal Engine 5 | [`unreal-plugin/`](unreal-plugin/README.md) | Blueprint component over part of the C ABI |
| Python | `src/python.rs` | `--features python`; `PhysicsWorld` (bodies, collision radius and shapes, static colliders, joints) and `DeterministicSimulation` classes, NumPy batch APIs |
| WebAssembly | [`web/`](web/) | `--features wasm`; `WasmPhysicsWorld` (bodies, collision radius and shapes, static colliders, joints) and a Three.js viewer built with `wasm-pack` |

The bindings cover what you need to build and step a scene (bodies, collision radius and shapes,
static colliders, joints, impulses, state serialization), not every module: the standalone modules
in the table above are Rust only. CI checks the C ABI against its consumers
(`scripts/integration_levels.py`): the C header, `bindings/AlicePhysics.h` and the Unity bindings
declare every exported function; the Unreal Engine component wraps part of them, and the functions
it does not wrap are listed with the reason in
[`docs/integration-levels.md`](docs/integration-levels.md).

```sh
cargo build --release --features ffi
```

## Performance

Measured with `cargo bench --bench physics_bench` (criterion, release profile)
on Apple silicon at version 1.2.0. Absolute numbers depend on the machine, so
re-run the benchmark on your target before quoting them.
<!-- perf-measured: 2026-09-15 benches/physics_bench.rs -->

| Operation | Measured |
|-----------|----------|
| `Fix128` multiply | 1.1 ns |
| `Fix128` divide | 141 ns |
| `Fix128` square root | 193–354 ns |
| `Vec3Fix` normalize | 618 ns |
| 10 bodies × 60 steps, default config | 398 µs |
| 1000 overlapping spheres, first frame | 65 ms |

### Sleeping bodies

`step` leaves a sleeping body out of every stage while it is at rest and not
attached to a joint or distance constraint (`PhysicsWorld::set_sleep_skip`, on
by default; the results are bit-identical with it off). Its contacts with
awake bodies are found through a persistent tree of the sleeping bodies, so the
broad-phase is built over the awake bodies only. `PhysicsWorld::stage_work`
reports the work of the last step per stage.

Measured with `cargo bench --bench world_scale` (criterion, release profile,
one `step` of 8 substeps) on arm64 / 10 cores / 32 GiB, with other processes
running (expect ±15 %). Spheres of radius 0.5 on a 2 m grid; the awake ones
drift above the sleeping ones and never touch them.
<!-- perf-measured: 2026-10-04 benches/world_scale.rs -->

| Bodies | Asleep | Skip on | Skip off |
|-------:|-------:|--------:|---------:|
| 10 000 | 0 % | 113 ms | 122 ms |
| 10 000 | 90 % | 9.7 ms | 96 ms |
| 10 000 | 99 % | 0.88 ms | 121 ms |
| 100 000 | 0 % | 1.31 s | 1.24 s |
| 100 000 | 90 % | 105 ms | 1.31 s |
| 100 000 | 99 % | 10.8 ms | 1.15 s |

A sleeping body still costs about 30 ns per step: a check that it was not
edited between steps (`bodies` is a public field) and its `idle_frames` count,
which the state snapshot carries.

## Minimum supported Rust version

Minimum supported Rust version: **1.85** (the `rust-version` in `Cargo.toml`). <!-- readme-sync: msrv -->

A CI job builds the library with exactly this version for the default
features, `no_std`, and `std,simd,parallel,ffi,gpu-solver-bridge`. Raising the
MSRV is a minor-version change, never a patch. The `neural`, `replay` and
`analytics` features follow the MSRV of the crates they pull in.

## Building and testing

```sh
cargo build --release
cargo test
cargo test --features "simd,parallel,ffi,gpu-solver-bridge"
cargo bench --bench physics_bench
cargo bench --bench world_scale
```

`wasm` and `ffi` cannot be enabled together, so `--all-features` does not
build; test `wasm` in a separate run.
`scripts/preflight.sh` reproduces the CI checks locally (`--quick` skips the
integration and oracle tests).

## Related crates

| Crate | Role |
|-------|------|
| [alice-det-math](https://github.com/ext-sakamoro/ALICE-DetMath) | the deterministic transcendental functions both this crate and ALICE-SDF use |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | signed distance functions; supplies SDF colliders evaluated with the same math |
| [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) | a language and law verifier that compiles to ALICE-SDF trees |

Keep a single version of `alice-det-math` in your dependency graph. Two
versions mean two implementations of the same function, and determinism is
lost. Check with `cargo tree -i alice-det-math`.

Release history is in [`CHANGELOG.md`](CHANGELOG.md). Planned work is in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

Dual-licensed under `AGPL-3.0-or-later OR LicenseRef-Commercial`. Pick either.

| Option | Use it when |
|--------|-------------|
| [AGPL-3.0-or-later](LICENSE-AGPL) | your project is AGPL-compatible open source, or you use it internally |
| [Commercial](LICENSE-COMMERCIAL.md) | closed-source products, proprietary SaaS, firmware, or engine plugin redistribution |

The AGPL applies to anything that links `alice-physics`, including through the
C ABI, the Unity / Unreal bindings, the Python bindings or the WebAssembly
build. Commercial licence enquiries: <contact@extoria.co.jp>

The optional `neural` and `replay` features pull in additional AGPL-licensed
crates. `analytics` and the required `alice-det-math` are `MIT OR Apache-2.0`.

Copyright (C) 2024-2026 Moroya Sakamoto

### References

- Müller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- Ericson, *Real-Time Collision Detection* (GJK / EPA)
- Volder, "The CORDIC Trigonometric Computing Technique"
