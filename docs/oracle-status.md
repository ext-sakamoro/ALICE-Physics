# ALICE-Physics Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | 3051 |
| 🔴 Red by design | 88 |
| ⏱ Gated (runtime / diagnostic / manual) | 24 |
| ⚪ Pending (bare `#[ignore]`) | 0 |
| **Total** | **3163** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

## 🔴 Red by design (88)

Oracles kept red on purpose: the implementation is not there yet, and a companion test pins
today's behaviour so CI coverage is not lost. The fix is in `src/`; the expected value is never loosened.

- `a_contact_modifier_can_change_the_friction` (audit_solver.rs) — known defect: AUD-A-S1W2-005: a ContactModifier's friction change is discarded; modifier friction 0 on a frict…
- `a_contact_modifier_can_change_the_restitution` (audit_solver.rs) — known defect: AUD-A-S1W2-005: a ContactModifier's restitution change is discarded (CPU solve keeps it in an un…
- `a_sliver_with_a_fixed_base_is_not_reported_as_under_constrained` (audit_linear_elastic_fem.rs) — known defect: AUD-A-S1W3-004: a sliver tet (apex height 1e-8, base fixed) returns UnderConstrained although 9 …
- `a_zero_extent_grid_with_use_turbulence_steps_without_panic` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-007: use_turbulence on a zero-extent grid evaluates `1..self.grid.nz - 1` with nz - 1…
- `aabb_plane_toi_is_consistent_with_the_two_sided_sphere_version` (audit_ccd.rs) — known defect: AUD-A-S2W2-012: aabb_plane_toi treats the back side of the plane as solid (support distance <= 0…
- `adaptive_toi_substeps_bounds_euclidean_travel_per_substep` (audit_ccd.rs) — known defect: AUD-A-S2W2-014: adaptive_toi_substeps sizes sub-steps from the L-infinity speed, so a body movin…
- `add_torque_uses_the_body_frame_inertia_of_a_rotated_body` (audit_solver.rs) — known defect: AUD-A-S1W2-001: add_torque / apply_impulse_at multiply the world-frame torque by the body-frame …
- `apply_impulse_at_uses_the_body_frame_inertia_of_a_rotated_body` (audit_solver.rs) — known defect: AUD-A-S1W2-001: same root cause through apply_impulse_at: r = (0,1,0), J = (0,0,1) gives torque …
- `ball_offset_anchor_splits_the_correction_between_translation_and_rotation` (audit_joint.rs) — known defect: AUD-A-S1W6-006: ball/hinge/fixed/slider/cone-twist/D6 positional corrections use w = inv_m_a + i…
- `bicgstab_breakdown_keeps_a_stale_iterate` (audit_eulerian_grid.rs) — known defect: AUD-A-S1W1-002: BiCGStab breaks down on Fix128 underflow of t.t (tolerance below ~1e-10) and ret…
- `body_restitution_changes_the_bounce` (audit_solver.rs) — known defect: AUD-A-S1W2-008: RigidBody.restitution / friction (with_restitution, with_friction, wasm setResti…
- `boundary_residual_covers_accumulated_pushes` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-004: boundary residual under-reports accumulated pushes (two same-direction pushes of…
- `boundary_residual_is_the_norm_of_the_net_velocity_change` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-004: boundary residual reports the largest single-pair correction component, not the …
- `boussinesq_face_force_is_the_mean_of_the_two_cells_sharing_the_face` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-002: Boussinesq reads the cell below the face (jj = j - 1) instead of the mean of the…
- `capsule_plane_toi_behind_the_plane_uses_the_nearer_endpoint` (audit_ccd.rs) — known defect: AUD-A-S2W2-010: capsule_plane_toi always takes the endpoint with the smaller signed distance, co…
- `capsule_plane_toi_straddling_capsule_is_already_touching` (audit_ccd.rs) — known defect: AUD-A-S2W2-010: capsule_plane_toi judges only one endpoint sphere, so a capsule a=(0,-5,0) b=(0,…
- `ceil_and_abs_at_the_range_extremes` (audit_math.rs) — known defect: AUD-A-S1W5-028: Fix128::ceil panics in debug (hi + 1 overflow; wraps to i64::MIN in release) for…
- `checked_mul_is_complete` (audit_math.rs) — known defect: AUD-A-S1W5-026: checked_mul returns None for in-range products: the guard `i64::try_from(hh)` te…
- `compute_force_is_a_force_not_a_separation` (audit_joint.rs) — known defect: AUD-A-S1W6-009: Joint::compute_force returns the anchor SEPARATION (metres) for ball/hinge/fixed…
- `conservative_advancement_does_not_give_up_on_a_shallow_approach` (audit_ccd.rs) — known defect: AUD-A-S2W2-013: conservative_advancement returns None after max_iterations without converging (i…
- `critical_load_does_not_depend_on_how_the_rectangle_is_labelled` (audit_buckling.rs) — known defect: AUD-A-S2W2-006: analyze_column uses I about the horizontal axis (b h^3/12), not the weak axis mi…
- `csf_x_line_impulse_matches_young_laplace_to_2_percent` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-006: the solver smears the delta over eps = 1.5 dx, a non-integer multiple of dx; the…
- `csf_y_and_z_line_impulses_equal_the_x_one_by_sphere_symmetry` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-001 (cfd_solver.rs:1567): CSF is applied through fx only; the y and z line integrals …
- `d6_local_frame_b_defines_the_zero_error_pose` (audit_joint.rs) — known defect: AUD-A-S1W6-010 / AUD-B-S1W6-001: D6Joint.local_frame_b is never read (grep: only its definition …
- `degenerate_bounds_with_more_than_one_node_are_rejected_or_have_nonzero_cell` (audit_coupled_field.rs) — known defect: AUD-A-S1W5-016: try_new accepts n>1 with max==min (and max<min): cell size 0 (negative), so worl…
- `density_si_of_decimal_datasheet_values_is_exact` (audit_filament_db.rs) — known defect: AUD-A-S1W5-011: from_ratio floors, so density_si() of PLA is 1239.99999999999995 (raw lo=u64::MA…
- `diffusivity_is_never_negative_outside_the_calibrated_range` (audit_transient_thermal.rs) — known defect: AUD-A-S2W2-016: diffusivity_at guards only rho*cp <= 0; steel_1018 at 2500 K gives k = -15 W/mK …
- `drag_never_overshoots_the_fluid_velocity` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-003: drag is explicit Euler with no clamp; C_d=0.5, rho=1000, N=1, dt=1/60 gives v: 1…
- `empty_source_list_does_not_panic` (audit_vibration_wall.rs) — known defect: AUD-A-S2W1-002: analyze_wall_resonance(&[]) panics (index out of bounds) with no documented prec…
- `exp_is_accurate_up_to_the_representable_maximum` (audit_math.rs) — known defect: AUD-A-S1W5-020: Fix128::exp saturates from x >= 43.0 (hi >= 43) but e^43 = 4.73e18 .. e^43.66 = …
- `explicit_step_conserves_enthalpy_with_temperature_dependent_conductivity` (audit_transient_thermal.rs) — known defect: AUD-A-S2W2-015: transient_step_1d advances alpha(T_i) * d2T/dx2 (non-conservative form) instead …
- `find_after_direct_manifolds_mutation_does_not_panic` (audit_contact_cache.rs) — known defect: AUD-A-S1W5-015: ContactCache.manifolds is pub but find()/get_or_create() trust a private pair_in…
- `fit_two_modes_never_returns_negative_damping_coefficients` (audit_damping_rayleigh.rs) — known defect: AUD-A-S2W2-001: fit_two_modes returns beta=-2.083e-5 for (100 rad/s, z=0.05) and (500 rad/s, z=0…
- `from_f64_non_finite_and_out_of_range_are_not_silently_plausible` (audit_math.rs) — known defect: AUD-A-S1W5-018: Fix128::from_f64(NaN) returns ZERO silently, from_f64(+/-inf) and from_f64(-1e30…
- `gamma_one_degenerate_is_not_silent_identity` (audit_compressible.rs) — known defect: AUD-A-S1W5-007: riemann_invariants(gamma=1) silently returns (u,u) (J+ == J-) although 2a/(gamma…
- `grid_pipeline_detects_thin_plate_parallel_to_scan_axis` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-004: analyze_thickness_grid only scans X-lines; a 0.4 mm z-thin plate parallel to X i…
- `hinge_with_only_a_minimum_still_enforces_it` (audit_joint.rs) — known defect: AUD-A-S1W6-007: hinge limits are applied only when BOTH angle_min and angle_max are Some; a lone…
- `in_contact_is_false_on_the_frame_the_contact_ends` (audit_solver.rs) — known defect: AUD-A-S1W2-003: observe_body.in_contact is true on the frame of an End event (any contact event …
- `joint_solve_through_a_reference_bridge_matches_the_cpu_solve` (audit_solver.rs) — known defect: AUD-A-S1W2-006: solve_joints_with_bridge writes back positions only; the rotation corrections of…
- `k_epsilon_point_source_is_one_explicit_euler_step_from_the_start_of_the_step_state` (audit_cfd_solver_rans.rs) — known defect: AUD-A-S1W4-009: the k-eps point source is documented as `explicit` (advance_epsilon: `one explic…
- `kinematic_velocity_after_a_step_is_the_displacement_over_dt` (audit_solver.rs) — known defect: AUD-A-S1W2-002: kinematic body velocity after step() is 0, not (target - position) / dt (the vel…
- `lame_never_returns_a_wrapped_value_inside_the_accepted_interval` (audit_linear_elastic_fem.rs) — known defect: AUD-A-S1W3-001: ElasticMaterial::new accepts nu near 0.5 / -1 and lame() wraps silently (E=1e6, …
- `mark_bulk_carries_at_least_theta_times_total_even_at_ulp_scale` (audit_linear_elastic_fem.rs) — known defect: AUD-A-S1W3-002: mark_bulk truncates total*theta, so at ulp-scale indicators the marked set carri…
- `mat3_inverse_of_small_scale_matrix` (audit_math.rs) — known defect: AUD-A-S1W5-024: Mat3Fix::inverse of a small non-singular matrix (diag 1e-5 I: det 1e-15 quantise…
- `outflow_on_the_low_layer_reads_a_stale_neighbour` (audit_eulerian_grid.rs) — known defect: AUD-A-S1W1-001: enforce_face_boundaries Outflow at index 0 reads the not-yet-enforced neighbour …
- `point_slightly_outside_surface_is_not_reported_as_a_hairline_wall` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-005: measure_thickness_at returns Some(0.01) for a surface point 0.011 mm outside the…
- `polar_rotation_is_idempotent_bit_for_bit_on_general_gradients` (audit_math.rs) — known defect: AUD-A-S1W5-027: polar_rotation is not idempotent for general (non-diagonal) gradients: 105 of 20…
- `powf_pos_large_integer_exponent` (audit_math.rs) — known defect: AUD-A-S1W5-021: powf_pos silently caps the integer part of the exponent at 64: 1.01^100 returns …
- `quat_from_axis_angle_zero_axis_is_a_unit_quaternion` (audit_math.rs) — known defect: AUD-A-S1W5-023: QuatFix::from_axis_angle with a zero axis returns (0,0,0,cos(angle/2)), a non-un…
- `recommended_fillet_trivial_target_gives_zero_radius` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-006: recommended_fillet_radius_mm returns ~0.01*d for a target every radius satisfies…
- `recommended_fillet_unreachable_target_not_silently_returned` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-005: recommended_fillet_radius_mm returns d/2 (K_t=1.43 > target 1.2) for an unreacha…
- `reconcile_weighted_is_atomic_on_error` (audit_coupled_field.rs) — known defect: AUD-A-S1W5-017: reconcile_weighted (and reconcile_mean) adopt participant by participant; if par…
- `register_beyond_u16_range_still_returns_an_id_that_finds_the_material` (audit_material.rs) — known defect: AUD-A-S1W6-002: register() returns `len as u16`; the 65537th material (index 65536) gets id 0 an…
- `register_ids_never_alias_after_u16_range` (audit_filament_db.rs) — known defect: AUD-A-S1W5-013: FilamentDb::register uses `len() as u16`; after 65536 materials ids wrap (65537t…
- `reinit_every_two_steps_reinitialises_at_the_end_of_the_second_step` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-004: reinit_every_n_steps = N first reinitialises on step N + 1 (the test is step_cou…
- `remove_body_keeps_the_contact_history_of_the_survivors` (audit_solver.rs) — known defect: AUD-A-S1W2-004: remove_body does not remap the event pair history; the surviving pair re-reports…
- `reported_force_does_not_vanish_at_zero_step` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-017: apply_fluid_forces_to_cloth_with_residual returns 0 at dt == 0 although the repo…
- `reserve_factor_scales_hill_to_incipient_failure` (audit_anisotropic.rs) — known defect: AUD-A-S1W6-003: Hill reserve_factor = 1/f although f scales as R^2: stress (X/2,0,0) gives f=0.2…
- `reserve_factor_scales_tsai_wu_to_incipient_failure` (audit_anisotropic.rs) — known defect: AUD-A-S1W6-004: Tsai-Wu reserve_factor = 1/index although the index is quadratic+linear in load:…
- `sampled_points_stay_inside_the_aabb` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-006: sample_surface_points returns points beyond aabb_max (ceil in axis_steps): x=5 >…
- `separation_load_for_c_above_one_is_not_negative` (audit_prestressed.rs) — known defect: AUD-A-S1W6-001: separation_load_n(1000, C=1.2) returns -5000 (negative load, no clamp/guard for …
- `shoulder_bending_at_d_over_d_two_matches_documented_table` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-001: kt_shaft_shoulder_bending at D/d=2 returns 1.1x the module's own D/d=2 table (r/…
- `shoulder_bending_is_continuous_in_fillet_radius` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-003: kt_shaft_shoulder_bending is a piecewise-CONSTANT step table (jump 2.9->2.2 at r…
- `shoulder_bending_without_step_is_unity` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-002: kt_shaft_shoulder_bending(D=d) returns the fillet table value (1.3..3.0), not 1
- `slider_with_only_a_maximum_still_enforces_it` (audit_joint.rs) — known defect: AUD-A-S1W6-007: lone SliderJoint.limit_max (limit_min None) is silently ignored, see hinge_with_…
- `solve_cubic_never_reports_ok_for_a_body_nothing_holds` (audit_cubic_elastic_fem.rs) — known defect: AUD-A-S2W1-008: solve_cubic returns Ok (u ~ 1e11 mm, relative_residual 0) for a body with no con…
- `speculative_contact_normal_follows_the_b_to_a_contract` (audit_ccd.rs) — known defect: AUD-A-S2W2-008: speculative_contact returns normal = (pos_b - pos_a)/dist (A->B) but collider::C…
- `speculative_contact_reports_coincident_overlapping_spheres` (audit_ccd.rs) — known defect: AUD-A-S2W2-009: speculative_contact returns None for coincident centres (dist == 0 guard) althou…
- `sphere_capsule_toi_sliding_approach_hits_at_the_right_time` (audit_ccd.rs) — known defect: AUD-A-S2W2-011: sphere_capsule_toi freezes the closest axis point at the start position (treats …
- `spring_correction_scales_with_dt_squared` (audit_joint.rs) — known defect: AUD-A-S1W6-008: solve_spring_joint adds `F*dt*inv_mass` to the POSITION (a velocity-sized quanti…
- `stagnation_pressure_ratio_is_monotone_and_positive_at_large_mach` (audit_compressible.rs) — known defect: AUD-A-S1W5-008: stagnation_pressure_ratio wraps silently when p0/p exceeds the Fix128 range (M=1…
- `step_with_options_refuses_or_bounds_an_unstable_explicit_diffusion_number` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-005: step / step_with_options accept nu dt / dx^2 = 0.6 > 1/6 and the checkerboard mo…
- `strain_is_monotone_and_nonnegative_across_documented_wlf_domain` (audit_creep_longterm.rs) — known defect: AUD-A-S2W1-003: predict_strain wraps (Fix128 overflow of t_eff^3) inside the documented WLF doma…
- `surface_tension_pulls_the_cloth_toward_the_fluid_centre` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-005: surface_tension term is mean_fluid_velocity * k (zero for resting fluid), not a …
- `symmetric_stack_has_exactly_zero_b` (audit_laminate.rs) — known defect: AUD-A-S1W5-009: compute_abd leaves B != 0 (rounding residue b11 = -2^-64 (raw hi=-1,lo=u64::MAX-…
- `the_answer_does_not_depend_on_the_increment_count` (analytic_corotational.rs) — the red is correct: bit-identical displacements across increment counts need an exact fixed point of the frame…
- `the_clamped_counter_of_k_omega_counts_cells_as_documented` (audit_cfd_solver_rans.rs) — known defect: AUD-A-S1W4-010: `TurbulenceSummary::clamped` is documented as the `Number of cells` whose k, eps…
- `the_default_fluid_is_water_with_water_s_thermal_expansion` (audit_cfd_solver_rans.rs) — known defect: AUD-A-S1W4-008: `CfdSolver::new` documents `default fluid = water at 20 C` but beta_per_k = 3.4e…
- `the_reattachment_length_matches_gartling` (armaly_backward_step.rs) — src gap: measured 2026-10-02 on this scene (ny = 8, default SemiLagrangian, L = 16 so nx = 128 is a power of t…
- `thickening_stress_stays_positive_and_increasing_at_high_rate` (audit_non_newtonian.rs) — known defect: AUD-A-S2W2-002: shear_thickening(1,3).stress(5e6) = 1.25e20 exceeds the Fix128 range and wraps t…
- `to_f64_is_relatively_accurate_for_small_negative_values` (audit_math.rs) — known defect: AUD-A-S1W5-025: Fix128::to_f64 computes hi + lo/2^64 in f64, so a small negative value loses its…
- `u_notch_kt_never_below_one` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-004: kt_u_notch_axial returns 0.913 at h/r=1e-3 and 0.85 at h=0 (<1); formula has no …
- `vec3_normalize_small_nonzero_vectors` (audit_math.rs) — known defect: AUD-A-S1W5-022: Vec3Fix::normalize / try_normalize / normalize_with_length treat any vector with…
- `wall_within_band_of_a_non_nearest_source_is_risky` (audit_vibration_wall.rs) — known defect: AUD-A-S2W1-001: is_risky checks only the abs-nearest source; w within +-20% of a farther (larger…
- `warm_start_normal_impulse_separates_with_documented_a_to_b_normal` (audit_contact_cache.rs) — known defect: AUD-A-S1W5-014: CachedContactPoint::normal doc says A->B but the cached value is Contact.normal …
- `youngs_at_angle_is_a_lower_bound_reuss_form` (audit_filament_db.rs) — known defect: AUD-A-S1W5-012: youngs_at_angle doc says 'Reuss-like lower bound' but E_xy cos2 + E_z sin2 is th…
- `zero_length_column_is_governed_by_yield_not_zero` (audit_buckling.rs) — known defect: AUD-A-S2W2-007: zero-length column reports critical_stress = 0 and critical_load = 0 with regime…
- `zero_strength_with_nonzero_stress_is_not_safe` (audit_anisotropic.rs) — known defect: AUD-A-S1W6-005: a zero strength is treated as unlimited (max-stress skips the component, Hill/Ts…

## ⏱ Gated (24)

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

- `adaptive_cubic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 40 s in release (P3 reference at two uniform passes, a uniform coarse solve and a six-round ada…
- `adaptive_quadratic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 2 s in release, measured 2026-10-03 (P2 reference at two uniform passes, a uniform coarse solve…
- `amplification_growth_is_problem_size_or_element_shape` (mesh_to_fem_stress.rs) — diagnostic: run when the amplification threshold is in question
- `cantilever_order_estimates_agree` (analytic_fem_convergence.rs) — 12,800 tets at cell 0.25 (about 1 s in release, measured 2026-10-03); run with --release, see the doc comment
- `cantilever_without_preconditioner` (analytic_fem_convergence.rs) — 12,800 tets at cell 0.25 (about 1 s in release, measured 2026-10-03); the A/B partner of the test above
- `cavity_gap_to_ghia_shrinks_with_resolution` (analytic_cfd_wall_bc.rs) — resolution sweep: three cavity runs to t = 30, minutes in a debug build
- `cavity_size_sweep` (mesh_quality.rs) — diagnostic: the table behind a_cavity_narrower_than_a_cell_leaves_no_trace
- `cavity_trace_by_size` (mesh_quality.rs) — diagnostic: the evidence table for what each cavity size leaves behind
- `channel_error_is_second_order_in_the_cell_size` (analytic_cfd_flow_bc.rs) — runtime only: 43k steps across three resolutions, release-only; the accuracy it asks for is reached, see the d…
- `duct_profile_develops_toward_the_parabola` (analytic_cfd_flow_bc.rs) — runtime only: 4096 steps on 24x16 cells with 200 GS sweeps, release-only; the accuracy it asks for is reached,…
- `finest_level_residual_floor_by_preconditioner` (analytic_fem_convergence.rs) — 4 solves at 12,800 tets (about 45 s in release, measured 2026-10-03); the stopping-rule diagnostic
- `gs_control_rates` (analytic_multigrid.rs) — diagnostic: GS control rates for oracle 2
- `gs_satisfies_exact_solution_oracle` (analytic_multigrid.rs) — diagnostic: oracle validity against the existing GS
- `iterations_to_same_precision` (analytic_multigrid.rs) — diagnostic: cycles vs GS iterations to the same precision (use --release)
- `maccormack_converges_and_the_schemes_bracket_the_limit_at_ny_8_16_32` (armaly_backward_step.rs) — manual: about 4 h in release (ny = 8 / 16 / 32, both schemes, each to t_settle), longer than the 180-min ignor…
- `mg_rate_report` (analytic_multigrid.rs) — diagnostic: multigrid rates used to fix the thresholds
- `p2_error_decreases_with_refinement_at_better_than_second_order` (analytic_hyperelastic_mms_order.rs) — runtime: about 5 s in release, about 60 s in debug (P2 n = 2, 3, 4 with a Fix128 Newton solve each); run by ru…
- `p3_separates_from_p2_at_large_amplitude` (analytic_hyperelastic_mms_order.rs) — runtime: about 35 s in release (P3 n = 3 at A = 0.08 with the Newton-Krylov step, plus P2 n = 3); run by run_i…
- `p3_separates_from_p2_in_order_and_in_error` (analytic_hyperelastic_mms_order.rs) — runtime: about 30 s in release (P3 n = 2 and 3 plus P2 n = 3, Fix128 Newton on 20-node elements); run by run_i…
- `reattachment_lengthens_under_grid_refinement` (armaly_backward_step.rs) — runtime only: 256x16 cells for 8192 steps, release-only (about 190 s, measured 2026-10-03); the measured value…
- `the_two_schemes_bracket_the_reattachment_length_at_ny_8_and_16` (armaly_backward_step.rs) — runtime: about 30 min in release (ny = 8 and 16, both advection schemes, each to t_settle at Courant 0.75); ru…
- `tolerance_floor_rises_as_the_mesh_refines` (analytic_fem_convergence.rs) — 4 solves (finest level 12,800 tets, about 1 s in release, measured 2026-10-03); the tolerance-floor measuremen…
- `tolerance_measurement` (analytic_step_multigrid.rs) — diagnostic: the measurements the two tolerances above are fixed from
- `x_1_time_trace` (armaly_backward_step.rs) — diagnostic: x_1(t) trace for one resolution and scheme, settings from ARM_NY / ARM_SCHEME / ARM_DT_RECIP / ARM…

## 🟢 Not ignored (3051)

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
| `integration_physics.rs` | 75 |
| `engineering_oracles_fluid.rs` | 40 |
| `audit_joint.rs` | 39 |
| `analytic_maxwell_fdtd.rs` | 38 |
| `analytic_raycast_wiring.rs` | 37 |
| `analytic_sdf_character_up_axis.rs` | 35 |
| `analytic_world_api.rs` | 35 |
| `engineering_oracles_solid.rs` | 34 |
| `audit_solver.rs` | 32 |
| `analytic_static_collider.rs` | 27 |
| `engineering_oracles_misc.rs` | 27 |
| `analytic_maxwell_wiring.rs` | 26 |
| `analytic_particle_wiring.rs` | 26 |
| `audit_math.rs` | 26 |
| `analytic_character_wiring.rs` | 25 |
| `analytic_p2g.rs` | 25 |
| `analytic_plastic_dissipation.rs` | 25 |
| `audit_cfd_solver_rans.rs` | 25 |
| `analytic_compound_wiring.rs` | 24 |
| `analytic_query_wiring.rs` | 24 |
| `analytic_transient_thermal_wiring.rs` | 24 |
| `analytic_compound.rs` | 23 |
| `analytic_convex_contact.rs` | 23 |
| `analytic_reactions.rs` | 22 |
| `determinism_semantic.rs` | 22 |
| `analytic_adaptive_refinement.rs` | 21 |
| `analytic_corotational.rs` | 21 |
| `analytic_multiphase_wiring.rs` | 21 |
| `analytic_neural_wiring.rs` | 21 |
| `analytic_rope_wiring.rs` | 21 |
| `audit_cfd_solver.rs` | 21 |
| `audit_linear_elastic_fem.rs` | 21 |
| `analytic_buoyancy_zone_wiring.rs` | 20 |
| `analytic_fluid_netcode_wiring.rs` | 20 |
| `analytic_joint_extra_wiring.rs` | 20 |
| `analytic_wave_ship_wiring.rs` | 20 |
| `analytic_compressible_wiring.rs` | 19 |
| `analytic_elastoplastic_fem.rs` | 19 |
| `analytic_heatmap_wiring.rs` | 19 |
| `analytic_laminate_wiring.rs` | 19 |
| `analytic_prestressed_wiring.rs` | 19 |
| `analytic_rolling_contact_wiring.rs` | 19 |
| `analytic_tgs_wiring.rs` | 19 |
| `analytic_filament_db_wiring.rs` | 18 |
| `analytic_flip.rs` | 18 |
| `analytic_hyperelastic_wiring.rs` | 18 |
| `analytic_multi_world_wiring.rs` | 18 |
| `analytic_sdf_force_wiring.rs` | 18 |
| `analytic_thin_wall_wiring.rs` | 18 |
| `analytic_wind_zone_wiring.rs` | 18 |
| `audit_hyperelastic.rs` | 18 |
| `analytic_acoustic_wave_wiring.rs` | 17 |
| `analytic_anomaly_wiring.rs` | 17 |
| `analytic_ccd_wiring.rs` | 17 |
| `analytic_debug_render_wiring.rs` | 17 |
| `analytic_gpu_sdf_wiring.rs` | 17 |
| `analytic_interpolation_wiring.rs` | 17 |
| `analytic_modal_wiring.rs` | 17 |
| `analytic_phase_change_wiring.rs` | 17 |
| `audit_coupled_field.rs` | 17 |
| `audit_cubic_elastic_fem.rs` | 17 |
| `audit_thin_wall.rs` | 17 |
| `analytic_audio_physics_wiring.rs` | 16 |
| `analytic_damping_rayleigh_wiring.rs` | 16 |
| `analytic_fracture_wiring.rs` | 16 |
| `analytic_linear_elastic_fem.rs` | 16 |
| `analytic_profiling_wiring.rs` | 16 |
| `analytic_shape.rs` | 16 |
| `analytic_structural_solver_wiring.rs` | 16 |
| `analytic_vehicle_wiring.rs` | 16 |
| `armaly_backward_step.rs` | 16 |
| `audit_eulerian_grid.rs` | 16 |
| `audit_maxwell_fdtd.rs` | 16 |
| `analytic_articulation_wiring.rs` | 15 |
| `analytic_dynamic_fem.rs` | 15 |
| `analytic_multibody_dynamics.rs` | 15 |
| `analytic_pipeline_wiring.rs` | 15 |
| `analytic_privacy_wiring.rs` | 15 |
| `analytic_rope_attach_wiring.rs` | 15 |
| `analytic_sdf_body_collider.rs` | 15 |
| `analytic_self_contact.rs` | 15 |
| `analytic_thermoplastic_coupling.rs` | 15 |
| `audit_anisotropic.rs` | 15 |
| `audit_ccd.rs` | 15 |
| `analytic_collision_mesh.rs` | 14 |
| `analytic_convex_decompose.rs` | 14 |
| `analytic_joint_wiring.rs` | 14 |
| `analytic_scene_io_wiring.rs` | 14 |
| `analytic_sdf_destruction_wiring.rs` | 14 |
| `analytic_sdf_fem_mesh_wiring.rs` | 14 |
| `analytic_sdf_manifold_wiring.rs` | 14 |
| `analytic_smoke_fire_wiring.rs` | 14 |
| `analytic_thermal_wiring.rs` | 14 |
| `analytic_anisotropic_wiring.rs` | 13 |
| `analytic_csf_wiring.rs` | 13 |
| `analytic_cubic_elastic_fem_wiring.rs` | 13 |
| `analytic_erosion_wiring.rs` | 13 |
| `analytic_filter_wiring.rs` | 13 |
| `analytic_interface_capture_wiring.rs` | 13 |
| `analytic_kinematic_loop_wiring.rs` | 13 |
| `analytic_math_wiring.rs` | 13 |
| `analytic_physics.rs` | 13 |
| `analytic_rans.rs` | 13 |
| `analytic_sdf_ccd_wiring.rs` | 13 |
| `analytic_sim_modifier_wiring.rs` | 13 |
| `analytic_sketch_wiring.rs` | 13 |
| `default_configs.rs` | 13 |
| `determinism_golden_f32.rs` | 13 |
| `analytic_aeroelasticity_wiring.rs` | 12 |
| `analytic_beam_stress_wiring.rs` | 12 |
| `analytic_collision_mesh_gen_wiring.rs` | 12 |
| `analytic_contact_cache_wiring.rs` | 12 |
| `analytic_deformable_wiring.rs` | 12 |
| `analytic_layer_adhesion_wiring.rs` | 12 |
| `audit_material.rs` | 12 |
| `audit_plastic.rs` | 12 |
| `audit_turbulence.rs` | 12 |
| `analytic_animation_blend_wiring.rs` | 11 |
| `analytic_contact_viz_wiring.rs` | 11 |
| `analytic_coupled_field.rs` | 11 |
| `analytic_flow_viz_wiring.rs` | 11 |
| `analytic_mass_properties.rs` | 11 |
| `analytic_polar_decomposition.rs` | 11 |
| `analytic_pressure_wiring.rs` | 11 |
| `analytic_replay_wiring.rs` | 11 |
| `analytic_sdf_adaptive_wiring.rs` | 11 |
| `analytic_temperature_rise.rs` | 11 |
| `analytic_vibration_wall_wiring.rs` | 11 |
| `audit_cloth_fluid.rs` | 11 |
| `audit_laminate.rs` | 11 |
| `analytic_bridging_wiring.rs` | 10 |
| `analytic_cfd_flow_bc.rs` | 10 |
| `analytic_cubic_hyperelastic.rs` | 10 |
| `analytic_db_bridge_wiring.rs` | 10 |
| `analytic_fsi_advanced_wiring.rs` | 10 |
| `analytic_linear_elastic_fem_wiring_additional.rs` | 10 |
| `analytic_physics2d_wiring.rs` | 10 |
| `analytic_pressure_solvers.rs` | 10 |
| `analytic_sdf_sph_wiring.rs` | 10 |
| `audit_buckling.rs` | 10 |
| `audit_creep_longterm.rs` | 10 |
| `coupling_channel_inventory.rs` | 10 |
| `fix128_vs_f64_coupling_hypotheses.rs` | 10 |
| `analytic_adaptive_dt.rs` | 9 |
| `analytic_anisotropic_friction_wiring.rs` | 9 |
| `analytic_character_state_wiring.rs` | 9 |
| `analytic_contact_forces.rs` | 9 |
| `analytic_event_sleeping_wiring.rs` | 9 |
| `analytic_flip_scatter.rs` | 9 |
| `analytic_laminate_failure_wiring.rs` | 9 |
| `analytic_material_wiring.rs` | 9 |
| `analytic_quadratic_fem.rs` | 9 |
| `analytic_quadratic_hyperelastic.rs` | 9 |
| `analytic_ragdoll_wiring.rs` | 9 |
| `analytic_thermoplastic_softening.rs` | 9 |
| `audit_compressible.rs` | 9 |
| `audit_vibration_wall.rs` | 9 |
| `determinism_golden.rs` | 9 |
| `analytic_added_mass_coupling.rs` | 8 |
| `analytic_broadphase.rs` | 8 |
| `analytic_cubic_fem.rs` | 8 |
| `analytic_fillet_stress_wiring.rs` | 8 |
| `analytic_fluid_block_wiring.rs` | 8 |
| `analytic_ik_physics_bridge_wiring.rs` | 8 |
| `analytic_math_util_wiring.rs` | 8 |
| `analytic_metric_broadphase.rs` | 8 |
| `analytic_netcode_prediction.rs` | 8 |
| `analytic_print_orientation_wiring.rs` | 8 |
| `analytic_step_multigrid.rs` | 8 |
| `analytic_thermal_stress_wiring.rs` | 8 |
| `analytic_wall_model.rs` | 8 |
| `audit_non_newtonian.rs` | 8 |
| `audit_prestressed.rs` | 8 |
| `mesh_conformity.rs` | 8 |
| `analytic_ccd_adaptive_substeps_wiring.rs` | 7 |
| `analytic_coupled_wiring.rs` | 7 |
| `analytic_electromagnetic_wiring.rs` | 7 |
| `analytic_hyperelastic_degenerate.rs` | 7 |
| `analytic_multigrid.rs` | 7 |
| `analytic_non_newtonian_wiring.rs` | 7 |
| `analytic_piezoelectric_wiring.rs` | 7 |
| `analytic_rng_wiring.rs` | 7 |
| `analytic_soft_body_cut_wiring.rs` | 7 |
| `audit_contact_cache.rs` | 7 |
| `audit_damping_rayleigh.rs` | 7 |
| `audit_fillet_stress.rs` | 7 |
| `audit_transient_thermal.rs` | 7 |
| `elastoplastic_increment_api.rs` | 7 |
| `fsi_advanced_sub_iteration.rs` | 7 |
| `mesh_quality.rs` | 7 |
| `wm07_reset_and_rollback_contract.rs` | 7 |
| `analytic_analytics_bridge_wiring.rs` | 6 |
| `analytic_bvh_leaf_aabb_wiring.rs` | 6 |
| `analytic_geometry_helpers.rs` | 6 |
| `analytic_metric_wiring.rs` | 6 |
| `analytic_support_volume_wiring.rs` | 6 |
| `analytic_thermoelastic.rs` | 6 |
| `analytic_thermoelastic_channel.rs` | 6 |
| `analytic_warp_risk_wiring.rs` | 6 |
| `audit_filament_db.rs` | 6 |
| `cloth_fluid_sub_iteration.rs` | 6 |
| `hanging_node_effect.rs` | 6 |
| `p3_quadrature_fix128.rs` | 6 |
| `wm01_overflow_is_not_silent.rs` | 6 |
| `wm08_checksum_coverage.rs` | 6 |
| `analytic_cfd_wall_bc.rs` | 5 |
| `analytic_cloth_crossings_wiring.rs` | 5 |
| `analytic_hyperelastic_mms_order.rs` | 5 |
| `analytic_math_ln.rs` | 5 |
| `analytic_quadratic_mesh_edges_wiring.rs` | 5 |
| `analytic_solver_tgs_dispatch_wiring.rs` | 5 |
| `engineering_oracles.rs` | 5 |
| `mms_linear_elastic.rs` | 5 |
| `reduction_order_independence.rs` | 5 |
| `analytic_adaptive_refinement_high_order.rs` | 4 |
| `analytic_step_default_projection.rs` | 4 |
| `locking_p1.rs` | 4 |
| `p2_oracle_design.rs` | 4 |
| `p2_quadrature_fix128.rs` | 4 |
| `refinement_conformity.rs` | 4 |
| `analytic_boundary_faces.rs` | 3 |
| `analytic_large_rotation.rs` | 3 |
| `parallel_batch_coloring.rs` | 3 |
| `wm01_flag_survives_rollback.rs` | 3 |
| `wm08_prev_state_coverage.rs` | 3 |
| `wm08_state_coverage.rs` | 3 |
| `mesh_to_fem_stress.rs` | 2 |
| `wm07_rollback_event_parity.rs` | 2 |
| `analytic_fem_convergence.rs` | 1 |

---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

For details: [CLAUDE.md](../CLAUDE.md)
