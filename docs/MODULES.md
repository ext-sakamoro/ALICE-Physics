# Module overview

Every public module of `alice-physics`, grouped by area, with a one-line
summary. API details are on [docs.rs](https://docs.rs/alice-physics); test
status for each module is in [`oracle-status.md`](oracle-status.md).

- **Feature**: the Cargo feature the module needs. Blank means it is always
  available, including under `no_std`.
- **Example**: a program in [`examples/`](../examples/) that uses the module
  (`cargo run --release --example <name>`).
- **Integration**: where the module's items are called from, examples not counted:
  `step` (runs when `PhysicsWorld` steps), `world API` (another `PhysicsWorld` method),
  `binding` (the C ABI, Python or WebAssembly), `standalone` (a Rust API you call yourself),
  `unused`. The label is the level most of its items have; items above it are noted in
  brackets. Measured by `scripts/integration_levels.py --check` in CI, details in
  [`integration-levels.md`](integration-levels.md).

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
- [Orbital mechanics](#orbital-mechanics)
- [3D printing](#3d-printing)
- [2D physics](#2d-physics)
- [Visualization](#visualization)
- [Telemetry and analytics](#telemetry-and-analytics)
- [Bindings and bridges](#bindings-and-bridges)
- [Validation notes](#validation-notes)

## Math and utilities

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `math` | `Fix128` (I64F64), `Vec3Fix`, `QuatFix`, `Mat3Fix`, CORDIC trigonometry, optional SSE2 paths |  | [`math_simd_and_transcendentals`](../examples/math_simd_and_transcendentals.rs) | step |
| `math_util` | `Fix128` transcendentals and solvers (`exp_fix`, `cbrt_fix`, `pow_int`, root finding) |  | [`math_util_roots_powers`](../examples/math_util_roots_powers.rs) | standalone |
| `linear_solver` | general Krylov solvers on `Fix128`: restarted GMRES(m) and BiCGStab, Jacobi / block Jacobi preconditioning, mandatory block equilibration for coupled systems |  | [`linear_solver_krylov`](../examples/linear_solver_krylov.rs) | standalone |
| `det_math` | re-export of `alice-det-math`: deterministic `sin`, `exp`, `ln`, `powf`, … for `f32` / `f64` |  |  | — |
| `metric` | the norm a distance is measured in (`‖·‖₁`, `‖·‖₂`, `‖·‖∞` mixes) and conversion to Euclidean bounds |  | [`metric_clearance_bounds`](../examples/metric_clearance_bounds.rs) | standalone (step 5 of 13 items) |
| `rng` | PCG-XSH-RR deterministic random number generator |  | [`rng_streams_and_bounded`](../examples/rng_streams_and_bounded.rs) | standalone |
| `error` | the crate-wide `PhysicsError` type |  |  | step |
| `prelude` | commonly used types in one import |  |  | — |

## World, solver and runtime

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `solver` | `PhysicsWorld`, `RigidBody`, XPBD solver (default) or temporal Gauss-Seidel backend, constraint batching, rollback state |  | [`basic_physics`](../examples/basic_physics.rs) | world API (step 28 of 148 items) |
| `shape` | solid shapes with mass properties, and bodies built from them |  | [`shaped_bodies`](../examples/shaped_bodies.rs) | world API (step 4 of 9 items) |
| `static_collider` | immovable planes, height fields and triangle meshes for a `PhysicsWorld` |  | [`static_colliders`](../examples/static_colliders.rs) | step |
| `mass_properties` | mass, centre of mass and inertia tensors for primitive shapes and convex hulls |  |  | world API |
| `material` | per-pair friction and restitution with combine rules |  | [`material_registry_presets`](../examples/material_registry_presets.rs) | step |
| `filter` | collision layers, masks and groups |  |  | standalone (step 3 of 18 items) |
| `force` | force fields: wind, gravity wells, drag, buoyancy, vortex, explosion, magnetic dipole |  | [`particle_emitter_forces`](../examples/particle_emitter_forces.rs) | step |
| `motor` | 1D / 3D PD controllers for joint motors |  | [`articulated_body_chain`](../examples/articulated_body_chain.rs) | step |
| `sleeping` | sleep detection and union-find islands |  |  | step |
| `event` | begin / persist / end contact and trigger events |  | [`world_events_and_islands`](../examples/world_events_and_islands.rs) | step |
| `interpolation` | world snapshots and quaternion blending for rendering between steps |  | [`substep_interpolation`](../examples/substep_interpolation.rs) | standalone (step 1 of 19 items) |
| `multi_world` | several independent worlds with body transfer |  | [`multi_world_management`](../examples/multi_world_management.rs) | standalone |
| `netcode` | lockstep frame inputs, snapshots, checksums and rollback |  | [`rollback_netcode`](../examples/rollback_netcode.rs) | binding |
| `netcode_prediction` | client-side prediction with server reconciliation | std | [`rollback_netcode`](../examples/rollback_netcode.rs) | standalone |
| `scene_io` | binary and JSON scene files with exact `Fix128` round trip | std | [`scene_snapshot_roundtrip`](../examples/scene_snapshot_roundtrip.rs) | standalone |
| `profiling` | per-stage timers and per-frame statistics |  | [`profiling_stages`](../examples/profiling_stages.rs) | standalone |
| `debug_render` | wireframe primitives for bodies, contacts, joints, BVH and forces |  | [`debug_render_primitives`](../examples/debug_render_primitives.rs) | standalone |
| `gpu_bridge` | `GpuSolverBridge` trait for external GPU solvers that must match the CPU result bit for bit | gpu-solver-bridge | [`world_api_tour`](../examples/world_api_tour.rs) | standalone (step 1 of 3 items) |

## Collision shapes and queries

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `collider` | sphere, capsule, convex hull and AABB shapes; GJK / EPA detection |  | [`convex_contacts`](../examples/convex_contacts.rs) | step |
| `box_collider` | oriented box (OBB) |  | [`compound_shapes`](../examples/compound_shapes.rs) | step |
| `compound` | several shapes with local transforms |  | [`compound_shapes`](../examples/compound_shapes.rs) | step |
| `cone` | cone (apex `+Y`) |  | [`geometry_queries`](../examples/geometry_queries.rs) | step |
| `cylinder` | cylinder |  |  | world API (step 2 of 7 items) |
| `ellipsoid` | ellipsoid with three semi-axes |  |  | step |
| `torus` | torus |  |  | world API (step 2 of 7 items) |
| `wedge` | triangular prism |  |  | step |
| `plane_collider` | infinite plane |  | [`static_colliders`](../examples/static_colliders.rs) | step |
| `convex_mesh_builder` | incremental convex hull from a point set |  |  | world API |
| `trimesh` | BVH-accelerated triangle mesh collision |  | [`static_colliders`](../examples/static_colliders.rs) | standalone (step 6, binding 3 of 16 items) |
| `heightfield` | grid terrain with bilinear interpolation |  | [`static_colliders`](../examples/static_colliders.rs) | step |
| `bvh` | linear BVH over Morton codes with stackless traversal |  | [`bvh_leaf_aabb_roundtrip`](../examples/bvh_leaf_aabb_roundtrip.rs) | step |
| `dynamic_bvh` | incremental AABB tree (insert / remove / update in O(log n)) |  |  | step |
| `spatial` | spatial hash grid |  |  | standalone |
| `raycast` | ray and shape casts |  | [`spatial_raycast_queries`](../examples/spatial_raycast_queries.rs) | standalone (binding 5 of 16 items) |
| `shape_raycast` | world ray queries against the geometry bodies collide as: shapes, compound children, static colliders and SDF colliders (closest / all / any, layer filter, BVH culling) |  | [`shape_raycast_sensors`](../examples/shape_raycast_sensors.rs) | standalone |
| `query` | sphere / capsule casts and overlap queries |  | [`spatial_queries`](../examples/spatial_queries.rs) | standalone |
| `ccd` | continuous collision detection (time of impact, conservative advancement, speculative contacts) |  | [`continuous_collision_detection`](../examples/continuous_collision_detection.rs) | standalone |
| `contact_cache` | persistent contact manifolds with warm starting, as an opt-in tool outside `PhysicsWorld::step` |  | [`contact_warm_start_cache`](../examples/contact_warm_start_cache.rs) | step |

## Joints and articulated bodies

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `joint` | ball, hinge, fixed, slider, spring, D6 and cone-twist joints; limits, motors, breaking |  | [`joint_limits_and_breaking`](../examples/joint_limits_and_breaking.rs) | standalone (step 12, binding 5 of 40 items) |
| `joint_extra` | pulley, gear, weld, rack-and-pinion and mouse joints |  | [`joint_extra_wiring`](../examples/joint_extra_wiring.rs) | standalone |
| `articulation` | multi-joint chains with Featherstone's articulated-body algorithm |  | [`articulated_body_chain`](../examples/articulated_body_chain.rs) | standalone |
| `ragdoll` | humanoid ragdoll builder |  | [`ragdoll_demo`](../examples/ragdoll_demo.rs) | standalone |
| `ik_physics_bridge` | drives an IK target chain through physics joints | std | [`ragdoll_ik_targets`](../examples/ragdoll_ik_targets.rs) | standalone |
| `kinematic_loop` | position-based closure for closed-chain mechanisms |  | [`kinematic_loop_four_bar`](../examples/kinematic_loop_four_bar.rs) | standalone |
| `animation_blend` | blending between ragdoll and animation poses |  | [`animation_blend_state_machine`](../examples/animation_blend_state_machine.rs) | standalone |

## Soft bodies and particles

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `rope` | XPBD rope and cable |  | [`rope_pin_constraints`](../examples/rope_pin_constraints.rs) | standalone |
| `rope_attach` | ropes attached to rigid bodies, with compliance and break force |  | [`rope_body_attachment`](../examples/rope_body_attachment.rs) | standalone |
| `cloth` | XPBD triangle-mesh cloth with self-collision |  | [`cloth_simulation`](../examples/cloth_simulation.rs) | standalone |
| `cloth_fluid` | two-way cloth and fluid coupling |  |  | standalone |
| `fluid` | position-based fluids (PBF) |  | [`fluid_block_column`](../examples/fluid_block_column.rs) | standalone |
| `fluid_netcode` | fluid state for netcode with delta compression | std | [`fluid_netcode_roundtrip`](../examples/fluid_netcode_roundtrip.rs) | standalone |
| `deformable` | FEM-XPBD tetrahedral deformable bodies |  | [`deformable_cube_impact`](../examples/deformable_cube_impact.rs) | standalone |
| `soft_body_cut` | cutting deformables and cloth along a plane |  | [`soft_body_cutting`](../examples/soft_body_cutting.rs) | standalone |
| `particle` | particle emitters with lifetime and force fields |  | [`particle_emitter_forces`](../examples/particle_emitter_forces.rs) | standalone |
| `pair_potential` | pair potentials: Lennard-Jones 12-6, Morse, Coulomb, screened Coulomb (Yukawa), cutoff with energy / force shift, Lorentz–Berthelot mixing |  | [`lj_dimer`](../examples/lj_dimer.rs) | standalone |
| `molecular_dynamics` | velocity Verlet for point particles under a pair potential, periodic cell list with minimum image, kinetic / potential energy and temperature |  | [`lj_dimer`](../examples/lj_dimer.rs) | standalone |

## Gameplay

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `character` | kinematic capsule controller with move-and-slide and stair stepping |  | [`character_controller`](../examples/character_controller.rs) | standalone |
| `character_state` | locomotion state machine (idle, walk, run, jump, fall, crouch) |  | [`character_state_machine`](../examples/character_state_machine.rs) | standalone |
| `vehicle` | wheels, suspension, engine and steering |  | [`vehicle_drive`](../examples/vehicle_drive.rs) | standalone |
| `vehicle_dynamics` | per-wheel contact forces, wheel spin, brakes and ABS, brush and Magic Formula tyres, road surfaces and weather, powertrain |  | [`vehicle_dynamics`](../examples/vehicle_dynamics.rs) | standalone |
| `audio_physics` | audio parameters from impacts, sliding and rolling |  | [`audio_physics_events`](../examples/audio_physics_events.rs) | standalone |
| `anisotropic_friction` | direction-dependent friction (tyres, skis, ice blades) |  | [`anisotropic_friction_presets`](../examples/anisotropic_friction_presets.rs) | standalone |
| `buoyancy_zone` | bounded fluid volume applying buoyancy and drag |  | [`buoyancy_zone_pool`](../examples/buoyancy_zone_pool.rs) | standalone |
| `wind_zone` | bounded wind volume applying drag and lift |  | [`wind_zone_forces`](../examples/wind_zone_forces.rs) | standalone |
| `sensors` | simulated lidar, contact sensor and IMU with seeded Gaussian noise |  | [`shape_raycast_sensors`](../examples/shape_raycast_sensors.rs) | standalone |
| `crowd_force` | social force model of pedestrians: driving term, exponential repulsion with a view-angle weight, body force and sliding friction in contact, walls, cell-list neighbour search |  | [`crowd_corridor`](../examples/crowd_corridor.rs) | standalone |

## SDF integration

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `sdf_collider` | collision against signed distance fields |  | [`convex_decomposition`](../examples/convex_decomposition.rs) | step |
| `sdf_manifold` | multi-point contact manifolds from SDF surfaces |  | [`sdf_manifold_patch`](../examples/sdf_manifold_patch.rs) | standalone |
| `sdf_ccd` | sphere-tracing continuous collision detection |  | [`sdf_ccd_sweep`](../examples/sdf_ccd_sweep.rs) | standalone |
| `sdf_force` | force fields driven by an SDF (attract, repel, contain, flow) |  | [`sdf_force_fields`](../examples/sdf_force_fields.rs) | standalone |
| `sdf_destruction` | CSG boolean destruction | std | [`sdf_destruction_events`](../examples/sdf_destruction_events.rs) | standalone |
| `sdf_adaptive` | distance-based level of detail for SDF evaluation | std | [`sdf_adaptive_lod`](../examples/sdf_adaptive_lod.rs) | standalone |
| `sdf_character` | character controller swept against an SDF |  | [`character_state_machine`](../examples/character_state_machine.rs) | standalone |
| `spherical_terrain` | sphere-world ground as a radial height field, constant gravity toward a centre |  | [`spherical_planet_walk`](../examples/spherical_planet_walk.rs) | standalone (step 1 of 10 items) |
| `sdf_sph` | SPH particle fluid with SDF boundaries | std | [`sph_boundary_demo`](../examples/sph_boundary_demo.rs) | standalone |
| `sdf_wind_field` | wind field shaped by an SDF |  |  | standalone |
| `sdf_fem_mesh` | conforming tetrahedral meshes from an SDF, with marked refinement | std | [`sdf_fem_mesh_generation`](../examples/sdf_fem_mesh_generation.rs) | standalone |
| `convex_decompose` | convex decomposition from an SDF voxel grid | std | [`convex_decomposition`](../examples/convex_decomposition.rs) | standalone |
| `collision_mesh_gen` | marching-cubes mesh from an SDF, with simplification |  | [`collision_mesh_from_sdf`](../examples/collision_mesh_from_sdf.rs) | standalone |
| `gpu_sdf` | batched SDF evaluation interface for compute shaders | std | [`gpu_sdf_batch_queries`](../examples/gpu_sdf_batch_queries.rs) | standalone (step 2 of 23 items) |

## Simulation fields and coupling

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `sim_field` | 3D scalar and vector fields with trilinear sampling and diffusion | std |  | standalone |
| `sim_modifier` | the modifier chain that applies fields to bodies | std | [`sim_modifier_management`](../examples/sim_modifier_management.rs) | standalone |
| `thermal` | heat diffusion, melting, expansion, freezing | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) | standalone |
| `pressure` | contact pressure, crushing, bulging and denting | std | [`pressure_contact_deformation`](../examples/pressure_contact_deformation.rs) | standalone |
| `erosion` | wind, water and chemical erosion, ablation | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) | standalone |
| `fracture` | stress-driven crack propagation | std | [`fracture_impact`](../examples/fracture_impact.rs) | standalone |
| `phase_change` | solid / liquid / gas transitions from temperature | std | [`thermal_phase_erosion_chain`](../examples/thermal_phase_erosion_chain.rs) | standalone |
| `coupled_field` | a `Fix128` scalar field shared between solvers, with order-independent reconciliation |  | [`temperature_reconciliation`](../examples/temperature_reconciliation.rs) | standalone |
| `coupled_iteration` | convergence monitoring for partitioned (staggered) multiphysics coupling |  | [`thermoplastic_sub_iteration`](../examples/thermoplastic_sub_iteration.rs) | standalone |

## Solid mechanics and FEM

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `linear_elastic_fem` | small-strain FEM on linear (P1) tetrahedra; thermal eigenstrain, J2 plasticity, corotational large rotation, adaptive refinement | std | [`linear_elastic_fem_config_diagnostics`](../examples/linear_elastic_fem_config_diagnostics.rs) | standalone (step 1 of 126 items) |
| `quadratic_elastic_fem` | FEM on ten-node quadratic (P2) tetrahedra |  | [`quadratic_mesh_edge_nodes`](../examples/quadratic_mesh_edge_nodes.rs) | standalone |
| `cubic_elastic_fem` | FEM on twenty-node cubic (P3) tetrahedra |  | [`cubic_elastic_fem_topology`](../examples/cubic_elastic_fem_topology.rs) | standalone |
| `dynamic_fem` | transient FEM with a mass matrix and Newmark-β time stepping | std | [`dynamic_fem_cantilever`](../examples/dynamic_fem_cantilever.rs) | standalone |
| `structural_solver` | time-stepping driver combining beam, plasticity, creep, fatigue and buckling; creep presets for PLA only, other materials need `with_creep`; S-N curve per material (SUS304 / A5052 presets, FDM rule otherwise), `with_sn_curve` overrides |  | [`structural_pla_shelf_creep`](../examples/structural_pla_shelf_creep.rs), [`structural_creep_per_material`](../examples/structural_creep_per_material.rs), [`structural_fatigue_buckling_per_material`](../examples/structural_fatigue_buckling_per_material.rs) | standalone |
| `beam_stress` | beam sections, load cases, deflection and safety factor |  | [`beam_end_condition_min_fos`](../examples/beam_end_condition_min_fos.rs) | standalone |
| `buckling` | Euler / Johnson column, plate and snap-through buckling |  | [`structural_fatigue_buckling_per_material`](../examples/structural_fatigue_buckling_per_material.rs) | standalone |
| `plastic` | von Mises yield, hardening and Norton creep |  |  | standalone |
| `hyperelastic` | neo-Hookean, Mooney-Rivlin and Yeoh models |  | [`hyperelastic_material_presets`](../examples/hyperelastic_material_presets.rs) | standalone |
| `anisotropic` | orthotropic materials with Hill and Tsai-Wu failure |  | [`anisotropic_failure_analysis`](../examples/anisotropic_failure_analysis.rs) | standalone |
| `laminate` | classical laminate theory (ABD matrix) | std | [`laminate_abd_matrix`](../examples/laminate_abd_matrix.rs) | standalone |
| `laminate_failure` | Tsai-Wu, Tsai-Hill, Hashin and Puck ply failure |  | [`laminate_failure_criteria`](../examples/laminate_failure_criteria.rs) | standalone |
| `bimaterial` | bimetal residual stress and Voigt / Reuss bounds |  |  | standalone |
| `prestressed` | bolt preload and cable pretension |  | [`prestressed_joints_and_cables`](../examples/prestressed_joints_and_cables.rs) | standalone |
| `fillet_stress` | stress concentration factors (Kirsch, Inglis, Peterson) |  | [`fillet_stress_concentration`](../examples/fillet_stress_concentration.rs) | standalone |
| `modal` | natural frequencies of beams, plates and springs |  | [`modal_frequency_analysis`](../examples/modal_frequency_analysis.rs) | standalone |
| `vibration_wall` | thin-wall resonance against printer excitation |  | [`vibration_wall_resonance`](../examples/vibration_wall_resonance.rs) | standalone |
| `damping_rayleigh` | Rayleigh damping and two-mode fitting |  | [`damping_rayleigh_fit`](../examples/damping_rayleigh_fit.rs) | standalone |
| `creep_longterm` | Findley creep and time-temperature superposition |  |  | standalone |
| `fatigue` | Basquin S-N curves and Miner's rule |  | [`structural_fatigue_buckling_per_material`](../examples/structural_fatigue_buckling_per_material.rs) | standalone |
| `thermal_stress` | thermal stress in constrained parts |  | [`thermal_stress_envelope`](../examples/thermal_stress_envelope.rs) | standalone |
| `transient_thermal` | 1D transient conduction with temperature-dependent properties | std | [`transient_thermal_materials`](../examples/transient_thermal_materials.rs) | standalone |
| `rolling_contact` | Hertzian rolling contact stress and fatigue life | std | [`rolling_contact_fatigue`](../examples/rolling_contact_fatigue.rs) | standalone |

## Fluids and waves

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `cfd_solver` | grid CFD driver: pressure projection (several solvers), turbulence, level set, surface tension, buoyancy, boundary conditions | std | [`cfd_smoke_plume`](../examples/cfd_smoke_plume.rs) | standalone |
| `eulerian_grid` | staggered MAC grid with FLIP / PIC transfer and domain decomposition |  | [`flip_scatter`](../examples/flip_scatter.rs) | standalone |
| `multiphase` | VOF and level-set multiphase flow |  | [`vof_level_set_transport`](../examples/vof_level_set_transport.rs) | standalone |
| `interface_capture` | fast-sweeping level set and PLIC interface reconstruction |  | [`plic_interface_reconstruction`](../examples/plic_interface_reconstruction.rs) | standalone |
| `surface_tension_csf` | continuum surface force |  | [`csf_surface_tension_presets`](../examples/csf_surface_tension_presets.rs) | standalone |
| `compressible` | ideal gas, normal shocks, Riemann invariants |  | [`compressible_gas_dynamics`](../examples/compressible_gas_dynamics.rs) | standalone |
| `non_newtonian` | power-law, Carreau, Bingham and Herschel-Bulkley fluids |  | [`non_newtonian_rheology`](../examples/non_newtonian_rheology.rs) | standalone |
| `turbulence` | Smagorinsky LES, k-ε, k-ω and wall functions |  | [`wall_model`](../examples/wall_model.rs) | standalone |
| `fsi_advanced` | fluid-structure coupling for deformables and articulations | std | [`fsi_advanced_forces`](../examples/fsi_advanced_forces.rs) | standalone |
| `smoke_fire` | Arrhenius combustion, soot and buoyancy |  | [`smoke_fire_combustion`](../examples/smoke_fire_combustion.rs) | standalone |
| `wave_ship` | JONSWAP ocean waves and Froude-Krylov ship forces |  | [`wave_ship_spectrum`](../examples/wave_ship_spectrum.rs) | standalone |
| `aeroelasticity` | vortex-induced vibration of slender structures |  | [`aeroelasticity_viv_lock_in`](../examples/aeroelasticity_viv_lock_in.rs) | standalone |
| `atmosphere` | U.S. Standard Atmosphere 1976 up to 20 km: temperature, pressure, density, speed of sound |  | [`aero_glider`](../examples/aero_glider.rs) | standalone |
| `lift_drag` | lift and drag of a wing surface: finite-wing slope, induced drag, stall to flat plate, force at the centre of pressure |  | [`aero_glider`](../examples/aero_glider.rs) | standalone |
| `rotor` | rotor thrust and torque, reaction torque on the body, momentum-theory hover power |  | [`rotor_hover`](../examples/rotor_hover.rs) | standalone |
| `acoustic_wave` | acoustic wave equation solver |  | [`acoustic_wave_propagation`](../examples/acoustic_wave_propagation.rs) | standalone |

## Electromagnetics

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `electromagnetic` | Lorentz force on charged rigid bodies |  | [`em_lorentz_cyclotron`](../examples/em_lorentz_cyclotron.rs) | standalone |
| `maxwell_fdtd` | Maxwell solver on a Yee lattice with sources, per-cell materials and a PML absorber |  | [`maxwell_sources_and_absorber`](../examples/maxwell_sources_and_absorber.rs), [`maxwell_dielectric_slab`](../examples/maxwell_dielectric_slab.rs) | standalone |
| `piezoelectric` | piezoelectric force and voltage coupling | std | [`piezoelectric_materials`](../examples/piezoelectric_materials.rs) | standalone |

## Orbital mechanics

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `kepler` | two-body problem: Kepler's equation, orbital elements and state vectors, propagation, vis-viva, J2 node and periapsis drift |  | [`orbit_two_body`](../examples/orbit_two_body.rs) | standalone |
| `nbody` | direct-sum mutual gravity with Plummer softening and a velocity-Verlet integrator, also on `PhysicsWorld` bodies |  | [`nbody_figure_eight`](../examples/nbody_figure_eight.rs) | standalone |

## 3D printing

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `filament_db` | material and sheet-metal property database |  | [`filament_database_properties`](../examples/filament_database_properties.rs) | standalone |
| `thin_wall` | wall thickness detection by sphere marching | std | [`thin_wall_detection`](../examples/thin_wall_detection.rs) | standalone |
| `support_volume` | support material volume and print time estimates | std | [`support_volume_presets`](../examples/support_volume_presets.rs) | standalone |
| `warp_risk` | warp risk from cooling shrinkage (empirical fit, see below) |  | [`warp_risk_enclosure`](../examples/warp_risk_enclosure.rs) | standalone |
| `layer_adhesion` | layer adhesion strength by print and load direction (empirical factors, see below) |  | [`layer_adhesion_fos`](../examples/layer_adhesion_fos.rs) | standalone |
| `print_orientation` | print orientation search for load-aligned strength |  | [`print_orientation_axes`](../examples/print_orientation_axes.rs) | standalone |
| `bridging` | maximum bridge distance per material | std | [`bridging_span_check`](../examples/bridging_span_check.rs) | standalone |
| `print_pipeline_solver` | runs the print checks above as one safety report | std | [`print_full_safety`](../examples/print_full_safety.rs) | standalone |

## 2D physics

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `physics2d` | a separate 2D XPBD engine with circle, polygon, capsule and edge shapes and 2D joints |  | [`physics2d_impulse_spin`](../examples/physics2d_impulse_spin.rs) | standalone |

## Visualization

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `heatmap` | stress / temperature / pressure slice images |  | [`heatmap_visualization`](../examples/heatmap_visualization.rs) | standalone |
| `flow_viz` | velocity arrows and streamlines | std | [`flow_visualization`](../examples/flow_visualization.rs) | standalone |
| `contact_viz` | contact force arrows and friction cones |  | [`contact_visualization`](../examples/contact_visualization.rs) | world API |

## Telemetry and analytics

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `anomaly` | streaming anomaly detection (EWMA, MAD, z-score) | std | [`anomaly_detectors`](../examples/anomaly_detectors.rs) | standalone |
| `pipeline` | ring-buffer metric aggregation | std | [`pipeline_events`](../examples/pipeline_events.rs) | standalone |
| `privacy` | local differential privacy (Laplace noise, RAPPOR, randomized response) | std | [`privacy_budget_and_rappor`](../examples/privacy_budget_and_rappor.rs) | standalone (step 2 of 44 items) |
| `sketch` | Count-Min, HyperLogLog, DDSketch and heavy hitters | std | [`sketch_streams`](../examples/sketch_streams.rs) | standalone (step 1 of 12 items) |

## Bindings and bridges

| Module | Summary | Feature | Example | Integration |
|--------|---------|---------|---------|-------------|
| `ffi` | C ABI for Unity, Unreal Engine and other hosts | ffi |  | binding |
| `neural` | deterministic neural controller with ternary weights | neural | [`neural_ternary_controller`](../examples/neural_ternary_controller.rs) | standalone |
| `replay` | replay recording and playback | replay | [`replay_recording`](../examples/replay_recording.rs) | standalone |
| `db_bridge` | physics state snapshots stored in ALICE-DB | replay | [`db_bridge_roundtrip`](../examples/db_bridge_roundtrip.rs) | standalone |
| `analytics_bridge` | simulation metrics sent to ALICE-Analytics | analytics | [`analytics_bridge_telemetry`](../examples/analytics_bridge_telemetry.rs) | standalone |

## Validation notes

Tests in `tests/` named `analytic_*` and `engineering_oracles*` compare modules
with closed-form solutions or published reference data. `audit_*` tests probe
edge cases. Tests kept red on purpose for known defects are listed in
[`oracle-status.md`](oracle-status.md).

Two modules are empirical models, not physical laws: `warp_risk` (fitted to a
small set of observed prints) and `layer_adhesion` (published FDM strength
factors). Their tests check that the formula is implemented as written; they
do not show that the predictions match real parts.
