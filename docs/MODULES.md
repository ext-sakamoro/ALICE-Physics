# Module overview

Every public module of `alice-physics`, grouped by area, with a one-line
summary. API details are on [docs.rs](https://docs.rs/alice-physics); test
status for each module is in [`oracle-status.md`](oracle-status.md).

- **Feature**: the Cargo feature the module needs. Blank means it is always
  available, including under `no_std`.
- **Example**: a program in [`examples/`](../examples/) that uses the module
  (`cargo run --release --example <name>`).

`scripts/readme_sync.py --check` fails CI when a public module is missing from
this file or listed twice, or when a linked example or test does not exist.

## Contents

- [Math and utilities](#math-and-utilities)
- [World, solver and runtime](#world-solver-and-runtime)
- [Collision shapes and queries](#collision-shapes-and-queries)
- [Joints and articulated bodies](#joints-and-articulated-bodies)
- [Soft bodies and particles](#soft-bodies-and-particles)
- [Gameplay](#gameplay)
- [SDF integration](#sdf-integration)
- [Simulation fields and coupling](#simulation-fields-and-coupling)
- [Solid mechanics and FEM](#solid-mechanics-and-fem)
- [Fluids and waves](#fluids-and-waves)
- [Electromagnetics](#electromagnetics)
- [3D printing](#3d-printing)
- [2D physics](#2d-physics)
- [Visualization](#visualization)
- [Telemetry and analytics](#telemetry-and-analytics)
- [Bindings and bridges](#bindings-and-bridges)
- [Validation notes](#validation-notes)

## Math and utilities

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `math` | `Fix128` (I64F64), `Vec3Fix`, `QuatFix`, `Mat3Fix`, CORDIC trigonometry, optional SSE2 paths | | [`math_simd_and_transcendentals`](../examples/math_simd_and_transcendentals.rs) |
| `math_util` | `Fix128` transcendentals and solvers (`exp_fix`, `cbrt_fix`, `pow_int`, root finding) | | [`math_util_roots_powers`](../examples/math_util_roots_powers.rs) |
| `det_math` | re-export of `alice-det-math`: deterministic `sin`, `exp`, `ln`, `powf`, … for `f32` / `f64` | | |
| `metric` | the norm a distance is measured in (`‖·‖₁`, `‖·‖₂`, `‖·‖∞` mixes) and conversion to Euclidean bounds | | [`metric_clearance_bounds`](../examples/metric_clearance_bounds.rs) |
| `rng` | PCG-XSH-RR deterministic random number generator | | [`rng_streams_and_bounded`](../examples/rng_streams_and_bounded.rs) |
| `error` | the crate-wide `PhysicsError` type | | |
| `prelude` | commonly used types in one import | | |

## World, solver and runtime

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `solver` | `PhysicsWorld`, `RigidBody`, XPBD solver (default) or temporal Gauss-Seidel backend, constraint batching, rollback state | | [`basic_physics`](../examples/basic_physics.rs) |
| `shape` | solid shapes with mass properties, and bodies built from them | | [`shaped_bodies`](../examples/shaped_bodies.rs) |
| `static_collider` | immovable planes, height fields and triangle meshes for a `PhysicsWorld` | | [`static_colliders`](../examples/static_colliders.rs) |
| `mass_properties` | mass, centre of mass and inertia tensors for primitive shapes and convex hulls | | |
| `material` | per-pair friction and restitution with combine rules | | [`material_registry_presets`](../examples/material_registry_presets.rs) |
| `filter` | collision layers, masks and groups | | |
| `force` | force fields: wind, gravity wells, drag, buoyancy, vortex, explosion, magnetic dipole | | [`particle_emitter_forces`](../examples/particle_emitter_forces.rs) |
| `motor` | 1D / 3D PD controllers for joint motors | | [`articulated_body_chain`](../examples/articulated_body_chain.rs) |
| `sleeping` | sleep detection and union-find islands | | |
| `event` | begin / persist / end contact and trigger events | | [`world_events_and_islands`](../examples/world_events_and_islands.rs) |
| `interpolation` | world snapshots and quaternion blending for rendering between steps | | [`substep_interpolation`](../examples/substep_interpolation.rs) |
| `multi_world` | several independent worlds with body transfer | | [`multi_world_management`](../examples/multi_world_management.rs) |
| `netcode` | lockstep frame inputs, snapshots, checksums and rollback | | [`rollback_netcode`](../examples/rollback_netcode.rs) |
| `netcode_prediction` | client-side prediction with server reconciliation | std | [`rollback_netcode`](../examples/rollback_netcode.rs) |
| `scene_io` | binary and JSON scene files with exact `Fix128` round trip | std | [`scene_snapshot_roundtrip`](../examples/scene_snapshot_roundtrip.rs) |
| `profiling` | per-stage timers and per-frame statistics | | [`profiling_stages`](../examples/profiling_stages.rs) |
| `debug_render` | wireframe primitives for bodies, contacts, joints, BVH and forces | | [`debug_render_primitives`](../examples/debug_render_primitives.rs) |
| `gpu_bridge` | `GpuSolverBridge` trait for external GPU solvers that must match the CPU result bit for bit | gpu-solver-bridge | [`world_api_tour`](../examples/world_api_tour.rs) |

## Collision shapes and queries

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `collider` | sphere, capsule, convex hull and AABB shapes; GJK / EPA detection | | [`convex_contacts`](../examples/convex_contacts.rs) |
| `box_collider` | oriented box (OBB) | | [`compound_shapes`](../examples/compound_shapes.rs) |
| `compound` | several shapes with local transforms | | [`compound_shapes`](../examples/compound_shapes.rs) |
| `cone` | cone (apex `+Y`) | | [`geometry_queries`](../examples/geometry_queries.rs) |
| `cylinder` | cylinder | | |
| `ellipsoid` | ellipsoid with three semi-axes | | |
| `torus` | torus | | |
| `wedge` | triangular prism | | |
| `plane_collider` | infinite plane | | [`static_colliders`](../examples/static_colliders.rs) |
| `convex_mesh_builder` | incremental convex hull from a point set | | |
| `trimesh` | BVH-accelerated triangle mesh collision | | [`static_colliders`](../examples/static_colliders.rs) |
| `heightfield` | grid terrain with bilinear interpolation | | [`static_colliders`](../examples/static_colliders.rs) |
| `bvh` | linear BVH over Morton codes with stackless traversal | | [`bvh_leaf_aabb_roundtrip`](../examples/bvh_leaf_aabb_roundtrip.rs) |
| `dynamic_bvh` | incremental AABB tree (insert / remove / update in O(log n)) | | |
| `spatial` | spatial hash grid | | |
| `raycast` | ray and shape casts | | [`spatial_raycast_queries`](../examples/spatial_raycast_queries.rs) |
| `shape_raycast` | world ray queries against the geometry bodies collide as: shapes, compound children, static colliders and SDF colliders (closest / all / any, layer filter, BVH culling) | | [`shape_raycast_sensors`](../examples/shape_raycast_sensors.rs) |
| `query` | sphere / capsule casts and overlap queries | | [`spatial_queries`](../examples/spatial_queries.rs) |
| `ccd` | continuous collision detection (time of impact, conservative advancement, speculative contacts) | | [`continuous_collision_detection`](../examples/continuous_collision_detection.rs) |
| `contact_cache` | persistent contact manifolds with warm starting, as an opt-in tool outside `PhysicsWorld::step` | | [`contact_warm_start_cache`](../examples/contact_warm_start_cache.rs) |

## Joints and articulated bodies

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `joint` | ball, hinge, fixed, slider, spring, D6 and cone-twist joints; limits, motors, breaking | | [`joint_limits_and_breaking`](../examples/joint_limits_and_breaking.rs) |
| `joint_extra` | pulley, gear, weld, rack-and-pinion and mouse joints | | [`joint_extra_wiring`](../examples/joint_extra_wiring.rs) |
| `articulation` | multi-joint chains with Featherstone's articulated-body algorithm | | [`articulated_body_chain`](../examples/articulated_body_chain.rs) |
| `ragdoll` | humanoid ragdoll builder | | [`ragdoll_demo`](../examples/ragdoll_demo.rs) |
| `ik_physics_bridge` | drives an IK target chain through physics joints | std | [`ragdoll_ik_targets`](../examples/ragdoll_ik_targets.rs) |
| `kinematic_loop` | position-based closure for closed-chain mechanisms | | [`kinematic_loop_four_bar`](../examples/kinematic_loop_four_bar.rs) |
| `animation_blend` | blending between ragdoll and animation poses | | [`animation_blend_state_machine`](../examples/animation_blend_state_machine.rs) |

## Soft bodies and particles

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `rope` | XPBD rope and cable | | [`rope_pin_constraints`](../examples/rope_pin_constraints.rs) |
| `rope_attach` | ropes attached to rigid bodies, with compliance and break force | | [`rope_body_attachment`](../examples/rope_body_attachment.rs) |
| `cloth` | XPBD triangle-mesh cloth with self-collision | | [`cloth_simulation`](../examples/cloth_simulation.rs) |
| `cloth_fluid` | two-way cloth and fluid coupling | | |
| `fluid` | position-based fluids (PBF) | | [`fluid_block_column`](../examples/fluid_block_column.rs) |
| `fluid_netcode` | fluid state for netcode with delta compression | std | [`fluid_netcode_roundtrip`](../examples/fluid_netcode_roundtrip.rs) |
| `deformable` | FEM-XPBD tetrahedral deformable bodies | | [`deformable_cube_impact`](../examples/deformable_cube_impact.rs) |
| `soft_body_cut` | cutting deformables and cloth along a plane | | [`soft_body_cutting`](../examples/soft_body_cutting.rs) |
| `particle` | particle emitters with lifetime and force fields | | [`particle_emitter_forces`](../examples/particle_emitter_forces.rs) |

## Gameplay

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `character` | kinematic capsule controller with move-and-slide and stair stepping | | [`character_controller`](../examples/character_controller.rs) |
| `character_state` | locomotion state machine (idle, walk, run, jump, fall, crouch) | | [`character_state_machine`](../examples/character_state_machine.rs) |
| `vehicle` | wheels, suspension, engine and steering | | [`vehicle_drive`](../examples/vehicle_drive.rs) |
| `vehicle_dynamics` | per-wheel contact forces, wheel spin, brakes and ABS, brush and Magic Formula tyres, road surfaces and weather, powertrain | | [`vehicle_dynamics`](../examples/vehicle_dynamics.rs) |
| `audio_physics` | audio parameters from impacts, sliding and rolling | | [`audio_physics_events`](../examples/audio_physics_events.rs) |
| `anisotropic_friction` | direction-dependent friction (tyres, skis, ice blades) | | [`anisotropic_friction_presets`](../examples/anisotropic_friction_presets.rs) |
| `buoyancy_zone` | bounded fluid volume applying buoyancy and drag | | [`buoyancy_zone_pool`](../examples/buoyancy_zone_pool.rs) |
| `wind_zone` | bounded wind volume applying drag and lift | | [`wind_zone_forces`](../examples/wind_zone_forces.rs) |
| `sensors` | simulated lidar, contact sensor and IMU with seeded Gaussian noise | | [`shape_raycast_sensors`](../examples/shape_raycast_sensors.rs) |

## SDF integration

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `sdf_collider` | collision against signed distance fields | | [`convex_decomposition`](../examples/convex_decomposition.rs) |
| `sdf_manifold` | multi-point contact manifolds from SDF surfaces | | [`sdf_manifold_patch`](../examples/sdf_manifold_patch.rs) |
| `sdf_ccd` | sphere-tracing continuous collision detection | | [`sdf_ccd_sweep`](../examples/sdf_ccd_sweep.rs) |
| `sdf_force` | force fields driven by an SDF (attract, repel, contain, flow) | | [`sdf_force_fields`](../examples/sdf_force_fields.rs) |
| `sdf_destruction` | CSG boolean destruction | std | [`sdf_destruction_events`](../examples/sdf_destruction_events.rs) |
| `sdf_adaptive` | distance-based level of detail for SDF evaluation | std | [`sdf_adaptive_lod`](../examples/sdf_adaptive_lod.rs) |
| `sdf_character` | character controller swept against an SDF | | [`character_state_machine`](../examples/character_state_machine.rs) |
| `spherical_terrain` | sphere-world ground as a radial height field, constant gravity toward a centre | | [`spherical_planet_walk`](../examples/spherical_planet_walk.rs) |
| `sdf_sph` | SPH particle fluid with SDF boundaries | std | [`sph_boundary_demo`](../examples/sph_boundary_demo.rs) |
| `sdf_wind_field` | wind field shaped by an SDF | | |
| `sdf_fem_mesh` | conforming tetrahedral meshes from an SDF, with marked refinement | std | [`sdf_fem_mesh_generation`](../examples/sdf_fem_mesh_generation.rs) |
| `convex_decompose` | convex decomposition from an SDF voxel grid | std | [`convex_decomposition`](../examples/convex_decomposition.rs) |
| `collision_mesh_gen` | marching-cubes mesh from an SDF, with simplification | | [`collision_mesh_from_sdf`](../examples/collision_mesh_from_sdf.rs) |
| `gpu_sdf` | batched SDF evaluation interface for compute shaders | std | [`gpu_sdf_batch_queries`](../examples/gpu_sdf_batch_queries.rs) |

## Simulation fields and coupling

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `sim_field` | 3D scalar and vector fields with trilinear sampling and diffusion | std | |
| `sim_modifier` | the modifier chain that applies fields to bodies | std | [`sim_modifier_management`](../examples/sim_modifier_management.rs) |
| `thermal` | heat diffusion, melting, expansion, freezing | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) |
| `pressure` | contact pressure, crushing, bulging and denting | std | [`pressure_contact_deformation`](../examples/pressure_contact_deformation.rs) |
| `erosion` | wind, water and chemical erosion, ablation | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) |
| `fracture` | stress-driven crack propagation | std | [`fracture_impact`](../examples/fracture_impact.rs) |
| `phase_change` | solid / liquid / gas transitions from temperature | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) |
| `coupled_field` | a `Fix128` scalar field shared between solvers, with order-independent reconciliation | | [`temperature_reconciliation`](../examples/temperature_reconciliation.rs) |
| `coupled_iteration` | convergence monitoring for partitioned (staggered) multiphysics coupling | | [`thermoplastic_sub_iteration`](../examples/thermoplastic_sub_iteration.rs) |

## Solid mechanics and FEM

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `linear_elastic_fem` | small-strain FEM on linear (P1) tetrahedra; thermal eigenstrain, J2 plasticity, corotational large rotation, adaptive refinement | std | [`linear_elastic_fem_config_diagnostics`](../examples/linear_elastic_fem_config_diagnostics.rs) |
| `quadratic_elastic_fem` | FEM on ten-node quadratic (P2) tetrahedra | | [`quadratic_mesh_edge_nodes`](../examples/quadratic_mesh_edge_nodes.rs) |
| `cubic_elastic_fem` | FEM on twenty-node cubic (P3) tetrahedra | | [`cubic_elastic_fem_topology`](../examples/cubic_elastic_fem_topology.rs) |
| `dynamic_fem` | transient FEM with a mass matrix and Newmark-β time stepping | std | [`dynamic_fem_cantilever`](../examples/dynamic_fem_cantilever.rs) |
| `structural_solver` | time-stepping driver combining beam, plasticity, creep, fatigue and buckling | | [`structural_pla_shelf_creep`](../examples/structural_pla_shelf_creep.rs) |
| `beam_stress` | beam sections, load cases, deflection and safety factor | | [`beam_end_condition_min_fos`](../examples/beam_end_condition_min_fos.rs) |
| `buckling` | Euler / Johnson column, plate and snap-through buckling | | |
| `plastic` | von Mises yield, hardening and Norton creep | | |
| `hyperelastic` | neo-Hookean, Mooney-Rivlin and Yeoh models | | [`hyperelastic_material_presets`](../examples/hyperelastic_material_presets.rs) |
| `anisotropic` | orthotropic materials with Hill and Tsai-Wu failure | | [`anisotropic_failure_analysis`](../examples/anisotropic_failure_analysis.rs) |
| `laminate` | classical laminate theory (ABD matrix) | std | [`laminate_abd_matrix`](../examples/laminate_abd_matrix.rs) |
| `laminate_failure` | Tsai-Wu, Tsai-Hill, Hashin and Puck ply failure | | [`laminate_failure_criteria`](../examples/laminate_failure_criteria.rs) |
| `bimaterial` | bimetal residual stress and Voigt / Reuss bounds | | |
| `prestressed` | bolt preload and cable pretension | | [`prestressed_joints_and_cables`](../examples/prestressed_joints_and_cables.rs) |
| `fillet_stress` | stress concentration factors (Kirsch, Inglis, Peterson) | | [`fillet_stress_concentration`](../examples/fillet_stress_concentration.rs) |
| `modal` | natural frequencies of beams, plates and springs | | [`modal_frequency_analysis`](../examples/modal_frequency_analysis.rs) |
| `vibration_wall` | thin-wall resonance against printer excitation | | [`vibration_wall_resonance`](../examples/vibration_wall_resonance.rs) |
| `damping_rayleigh` | Rayleigh damping and two-mode fitting | | [`damping_rayleigh_fit`](../examples/damping_rayleigh_fit.rs) |
| `creep_longterm` | Findley creep and time-temperature superposition | | |
| `fatigue` | Basquin S-N curves and Miner's rule | | |
| `thermal_stress` | thermal stress in constrained parts | | [`thermal_stress_envelope`](../examples/thermal_stress_envelope.rs) |
| `transient_thermal` | 1D transient conduction with temperature-dependent properties | std | [`transient_thermal_materials`](../examples/transient_thermal_materials.rs) |
| `rolling_contact` | Hertzian rolling contact stress and fatigue life | std | [`rolling_contact_fatigue`](../examples/rolling_contact_fatigue.rs) |

## Fluids and waves

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `cfd_solver` | grid CFD driver: pressure projection (several solvers), turbulence, level set, surface tension, buoyancy, boundary conditions | std | [`cfd_smoke_plume`](../examples/cfd_smoke_plume.rs) |
| `eulerian_grid` | staggered MAC grid with FLIP / PIC transfer and domain decomposition | | [`flip_scatter`](../examples/flip_scatter.rs) |
| `multiphase` | VOF and level-set multiphase flow | | [`vof_level_set_transport`](../examples/vof_level_set_transport.rs) |
| `interface_capture` | fast-sweeping level set and PLIC interface reconstruction | | [`plic_interface_reconstruction`](../examples/plic_interface_reconstruction.rs) |
| `surface_tension_csf` | continuum surface force | | [`csf_surface_tension_presets`](../examples/csf_surface_tension_presets.rs) |
| `compressible` | ideal gas, normal shocks, Riemann invariants | | [`compressible_gas_dynamics`](../examples/compressible_gas_dynamics.rs) |
| `non_newtonian` | power-law, Carreau, Bingham and Herschel-Bulkley fluids | | [`non_newtonian_rheology`](../examples/non_newtonian_rheology.rs) |
| `turbulence` | Smagorinsky LES, k-ε, k-ω and wall functions | | [`wall_model`](../examples/wall_model.rs) |
| `fsi_advanced` | fluid-structure coupling for deformables and articulations | std | [`fsi_advanced_forces`](../examples/fsi_advanced_forces.rs) |
| `smoke_fire` | Arrhenius combustion, soot and buoyancy | | [`smoke_fire_combustion`](../examples/smoke_fire_combustion.rs) |
| `wave_ship` | JONSWAP ocean waves and Froude-Krylov ship forces | | [`wave_ship_spectrum`](../examples/wave_ship_spectrum.rs) |
| `aeroelasticity` | vortex-induced vibration of slender structures | | [`aeroelasticity_viv_lock_in`](../examples/aeroelasticity_viv_lock_in.rs) |
| `acoustic_wave` | acoustic wave equation solver | | [`acoustic_wave_propagation`](../examples/acoustic_wave_propagation.rs) |

## Electromagnetics

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `electromagnetic` | Lorentz force on charged rigid bodies | | [`em_lorentz_cyclotron`](../examples/em_lorentz_cyclotron.rs) |
| `maxwell_fdtd` | Maxwell solver on a Yee lattice with sources and a PML absorber | | [`maxwell_sources_and_absorber`](../examples/maxwell_sources_and_absorber.rs) |
| `piezoelectric` | piezoelectric force and voltage coupling | std | [`piezoelectric_materials`](../examples/piezoelectric_materials.rs) |

## 3D printing

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `filament_db` | material and sheet-metal property database | | [`filament_database_properties`](../examples/filament_database_properties.rs) |
| `thin_wall` | wall thickness detection by sphere marching | std | [`thin_wall_detection`](../examples/thin_wall_detection.rs) |
| `support_volume` | support material volume and print time estimates | std | [`support_volume_presets`](../examples/support_volume_presets.rs) |
| `warp_risk` | warp risk from cooling shrinkage (empirical fit, see below) | | [`warp_risk_enclosure`](../examples/warp_risk_enclosure.rs) |
| `layer_adhesion` | layer adhesion strength by print and load direction (empirical factors, see below) | | [`layer_adhesion_fos`](../examples/layer_adhesion_fos.rs) |
| `print_orientation` | print orientation search for load-aligned strength | | [`print_orientation_axes`](../examples/print_orientation_axes.rs) |
| `bridging` | maximum bridge distance per material | std | [`bridging_span_check`](../examples/bridging_span_check.rs) |
| `print_pipeline_solver` | runs the print checks above as one safety report | std | [`print_full_safety`](../examples/print_full_safety.rs) |

## 2D physics

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `physics2d` | a separate 2D XPBD engine with circle, polygon, capsule and edge shapes and 2D joints | | [`physics2d_impulse_spin`](../examples/physics2d_impulse_spin.rs) |

## Visualization

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `heatmap` | stress / temperature / pressure slice images | | [`heatmap_visualization`](../examples/heatmap_visualization.rs) |
| `flow_viz` | velocity arrows and streamlines | std | [`flow_visualization`](../examples/flow_visualization.rs) |
| `contact_viz` | contact force arrows and friction cones | | [`contact_visualization`](../examples/contact_visualization.rs) |

## Telemetry and analytics

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `anomaly` | streaming anomaly detection (EWMA, MAD, z-score) | std | [`anomaly_detectors`](../examples/anomaly_detectors.rs) |
| `pipeline` | ring-buffer metric aggregation | std | [`pipeline_events`](../examples/pipeline_events.rs) |
| `privacy` | local differential privacy (Laplace noise, RAPPOR, randomized response) | std | [`privacy_budget_and_rappor`](../examples/privacy_budget_and_rappor.rs) |
| `sketch` | Count-Min, HyperLogLog, DDSketch and heavy hitters | std | [`sketch_streams`](../examples/sketch_streams.rs) |

## Bindings and bridges

| Module | Summary | Feature | Example |
|--------|---------|---------|---------|
| `ffi` | C ABI for Unity, Unreal Engine and other hosts | ffi | |
| `neural` | deterministic neural controller with ternary weights | neural | [`neural_ternary_controller`](../examples/neural_ternary_controller.rs) |
| `replay` | replay recording and playback | replay | [`replay_recording`](../examples/replay_recording.rs) |
| `db_bridge` | physics state snapshots stored in ALICE-DB | replay | [`db_bridge_roundtrip`](../examples/db_bridge_roundtrip.rs) |
| `analytics_bridge` | simulation metrics sent to ALICE-Analytics | analytics | [`analytics_bridge_telemetry`](../examples/analytics_bridge_telemetry.rs) |

## Validation notes

Tests in `tests/` named `analytic_*` and `engineering_oracles*` compare modules
with closed-form solutions or published reference data. `audit_*` tests probe
edge cases. Tests kept red on purpose for known defects are listed in
[`oracle-status.md`](oracle-status.md).

Two modules are empirical models, not physical laws: `warp_risk` (fitted to a
small set of observed prints) and `layer_adhesion` (published FDM strength
factors). Their tests check that the formula is implemented as written; they
do not show that the predictions match real parts.
