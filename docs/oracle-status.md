# ALICE-Physics Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | 5318 |
| 🔴 Red by design | 199 |
| ⏱ Gated (runtime / diagnostic / manual) | 26 |
| ⚪ Pending (bare `#[ignore]`) | 0 |
| **Total** | **5543** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

## 🔴 Red by design (199)

Oracles kept red on purpose: the implementation is not there yet, and a companion test pins
today's behaviour so CI coverage is not lost. The fix is in `src/`; the expected value is never loosened.

- `a_solid_ball_slides_then_rolls_at_five_sevenths_of_its_launch_speed` (indep_contact_restitution_friction.rs) — src gap: contact friction is translational only, a solid ball gets no spin and never rolls
- `a_zero_dt_update_does_not_cancel_a_return_to_animation` (audit_animation_blend.rs) — known defect: AUD-A-S2W3-011: mirror case, go_animated() from full ragdoll then update(dt = 0) flips mode to R…
- `a_zero_dt_update_does_not_cancel_a_started_transition` (audit_animation_blend.rs) — known defect: AUD-A-S2W3-011: go_ragdoll() then update(dt = 0) (or transition_speed 0) flips mode Blend -> Ani…
- `aabb_of_a_negative_spacing_field_is_not_inverted` (audit_heightfield.rs) — known defect: AUD-A-S4W3-007: HeightField::new accepts spacing <= 0 (debug_assert checks only the length); col…
- `aabb_plane_toi_is_consistent_with_the_two_sided_sphere_version` (audit_ccd.rs) — known defect: AUD-A-S2W2-012: aabb_plane_toi treats the back side of the plane as solid (support distance <= 0…
- `adaptive_toi_substeps_bounds_euclidean_travel_per_substep` (audit_ccd.rs) — known defect: AUD-A-S2W2-014: adaptive_toi_substeps sizes sub-steps from the L-infinity speed, so a body movin…
- `applied_cycles_above_i64_max_do_not_flip_the_damage_sign` (audit_fatigue.rs) — known defect: AUD-A-S4W2-007: n as i64 で n > i64::MAX が負に wrap し damage が負になる (n = u64::MAX -> D < 0)
- `arrow_records_head_geometry_besides_the_shaft` (audit_debug_render.rs) — known defect: AUD-A-S4W3-008: arrow() documents 'line + arrowhead' but the computed head_point is discarded (`…
- `bond_strength_is_symmetric_for_equal_names_with_different_yield` (audit_bimaterial.rs) — known defect: AUD-A-S4W3-011: interfacial_bond_strength_mpa is not symmetric when the two materials share a na…
- `buoyancy_field_holds_a_body_at_its_equilibrium_depth` (analytic_external_force_substep.rs) — src gap: the buoyancy field is a frame-head impulse while gravity is per substep, so a floating body at equili…
- `buoyancy_uses_gravity_magnitude` (audit_fsi_advanced.rs) — known defect: AUD-A-S4W1-002: buoyancy_force doc says rho*V*|g| but g=-10 gives F_y=-10000 (downward); no abs
- `capsule_bottom_does_not_sink_into_a_static_body_it_lands_on` (audit_character.rs) — known defect: AUD-A-S3W3-009: sweep_against_bodies casts a point-sphere from the capsule centre, so landing on…
- `capsule_cast_misses_nothing_along_the_segment` (audit_query.rs) — known defect: AUD-A-S3W3-014: capsule_cast casts only 3 spheres (both ends + midpoint) so a body above the qua…
- `capsule_plane_toi_behind_the_plane_uses_the_nearer_endpoint` (audit_ccd.rs) — known defect: AUD-A-S2W2-010: capsule_plane_toi always takes the endpoint with the smaller signed distance, co…
- `capsule_plane_toi_straddling_capsule_is_already_touching` (audit_ccd.rs) — known defect: AUD-A-S2W2-010: capsule_plane_toi judges only one endpoint sphere, so a capsule a=(0,-5,0) b=(0,…
- `checked_mul_is_complete` (audit_math.rs) — known defect: AUD-A-S1W5-026: checked_mul returns None for in-range products: the guard `i64::try_from(hh)` te…
- `checksum_distinguishes_where_the_position_stream_ends` (audit_fluid_netcode.rs) — known defect: AUD-A-S5W2-012: the checksum hashes positions then velocities as one byte stream with no length,…
- `closest_and_all_agree_on_equidistant_aabbs` (audit_raycast.rs) — known defect: AUD-A-S5W1-004: tie-breaking differs: raycast_aabbs picks the last of equal-t candidates, raycas…
- `closest_and_all_agree_on_equidistant_spheres` (audit_raycast.rs) — known defect: AUD-A-S5W1-004: tie-breaking differs: raycast_spheres picks the last of equal-t candidates (max_…
- `collide_aabb_contact_points_are_separated_by_depth_along_the_normal` (audit_trimesh.rs) — known defect: AUD-A-S4W2-017: collide_aabb の point_a が center - normal * depth (箱の表面でも最深点でもない)。床の上の箱で point_a …
- `collide_aabb_reports_contact_iff_the_triangle_overlaps_the_box` (audit_trimesh.rs) — known defect: AUD-A-S4W2-013: 重なる箱の 30 / 239 (12.6%) を接触なしと答える (箱中心への最近点が箱の外なら棄却するため)。標本点で箱内を確認済
- `collide_aabb_slanted_triangle_clipping_a_box_corner_is_a_contact` (audit_trimesh.rs) — known defect: AUD-A-S4W2-013: 平面 x + 0.1y = 1.05 の三角形が箱 [-1,1]^3 と交わるのに接触なし (中心からの垂線の足 x = 1.0396 が箱の外)
- `collide_capsule_depth_equals_radius_minus_segment_triangle_distance` (audit_trimesh.rs) — known defect: AUD-A-S4W2-014: 線分-三角形の最近点を 2 回の交互射影で近似するため距離を過大評価する (250 配置中 63 件で標本より悪い、最悪 +0.55)
- `collide_capsule_integer_example_with_oblique_segment_is_a_contact` (audit_trimesh.rs) — known defect: AUD-A-S4W2-014: 整数例 (線分 (6,8,-7)-(-5,0,5)、三角形 (6,2,-2) (-5,-2,5) (6,4,5)、半径 2.5) で真の距離 1.756 < r…
- `composite_score_matches_verdict_during_warm_up` (audit_anomaly.rs) — known defect: AUD-A-S4W1-017: CompositeDetector::anomaly_score = +inf with 0 observations although is_anomaly …
- `compute_force_is_a_force_not_a_separation` (audit_joint.rs) — known defect: AUD-A-S1W6-009: Joint::compute_force returns the anchor SEPARATION (metres) for ball/hinge/fixed…
- `conservative_advancement_does_not_give_up_on_a_shallow_approach` (audit_ccd.rs) — known defect: AUD-A-S2W2-013: conservative_advancement returns None after max_iterations without converging (i…
- `constant_force_follows_the_closed_form_trajectory` (analytic_external_force_substep.rs) — src gap: a frame-head force impulse adds ½ (F/m) dt t (s-1)/s to the position, which more substeps do not redu…
- `constant_force_position_does_not_depend_on_the_frame_length` (analytic_external_force_substep.rs) — src gap: with a frame-head force impulse the position error is set by the frame length dt, not the substep len…
- `constant_torque_spins_up_as_tau_t_over_i` (analytic_rotation_integration.rs) — src gap: add_torque is applied once at the frame head, so θ carries the splitting term ½ (τ/I) dt t (s − 1)/s …
- `contact_concentric_sphere_pair_reports_the_radius_sum` (audit_collider.rs) — known defect: AUD-A-S3W3-016: EPA depth for near-concentric spheres is 5.5 percent shallow (concentric r 1.0 +…
- `contact_upload_keeps_slot_alignment_with_a_sensor_in_slot_zero` (audit_gpu_bridge.rs) — known defect: AUD-A-S2W3-003: send_contact_constraints doc says indices line up with contact_constraints slots…
- `contain_is_force_free_inside_as_documented` (audit_sdf_force.rs) — known defect: AUD-A-S4W1-009: SdfForceType::Contain doc says zero force inside; code returns -damping*velocity…
- `countmin_total_does_not_overflow_when_counters_saturate` (audit_sketch.rs) — known defect: AUD-A-S3W1-010: CountMinSketch::insert_hash(_, u64::MAX) twice: counters saturate but `total += …
- `covering_is_the_smallest_power_of_two_at_or_above_a_fractional_magnitude` (audit_coupled_iteration.rs) — known defect: AUD-A-S3W3-012: EquilibrationScale::covering(0.3) returns factor 1 (exponent is u32), so the doc…
- `critical_load_does_not_depend_on_how_the_rectangle_is_labelled` (audit_buckling.rs) — known defect: AUD-A-S2W2-006: analyze_column uses I about the horizontal axis (b h^3/12), not the weak axis mi…
- `curvature_is_invariant_under_scaling_of_phi` (audit_multiphase.rs) — known defect: AUD-A-S2W3-009: curvature_at documents a 1/|grad phi| scaling but returns the bare Laplacian / d…
- `ddsketch_quantile_zero_is_inside_the_data_range` (audit_sketch.rs) — known defect: AUD-A-S3W1-006: DDSketch::quantile(0.0) (rank 0) returns the outermost negative bucket edge, -2.…
- `ddsketch_relative_error_holds_above_the_last_bucket` (audit_sketch.rs) — known defect: AUD-A-S3W1-009: values above the top bin (gamma^(3 BINS/4) = 2e13 at alpha 0.01) are dropped fro…
- `ddsketch_relative_error_holds_below_the_first_bucket` (audit_sketch.rs) — known defect: AUD-A-S3W1-009: values below gamma^-(BINS/4) (3.6e-5 at alpha 0.01, DDSketch2048) are clamped in…
- `degenerate_bounds_with_more_than_one_node_are_rejected_or_have_nonzero_cell` (audit_coupled_field.rs) — known defect: AUD-A-S1W5-016: try_new accepts n>1 with max==min (and max<min): cell size 0 (negative), so worl…
- `density_si_of_decimal_datasheet_values_is_exact` (audit_filament_db.rs) — known defect: AUD-A-S1W5-011: from_ratio floors, so density_si() of PLA is 1239.99999999999995 (raw lo=u64::MA…
- `diffusion_step_never_produces_negative_pressure` (audit_pressure.rs) — known defect: AUD-A-S6W1-004: update() with diffusion_rate*dt/h^2 = 0.5 turns a +100 spike into negative press…
- `diffusivity_is_never_negative_outside_the_calibrated_range` (audit_transient_thermal.rs) — known defect: AUD-A-S2W2-016: diffusivity_at guards only rho*cp <= 0; steel_1018 at 2500 K gives k = -15 W/mK …
- `directional_force_magnitude_is_the_strength_for_a_non_unit_direction` (audit_force.rs) — known defect: AUD-A-S4W3-014: Directional documents `direction` as normalized but does not normalize or check …
- `dispatch_covers_every_query_with_the_shader_workgroup_size` (audit_gpu_sdf.rs) — known defect: AUD-A-S5W2-021: num_workgroups() divides by config.workgroup_size (documented as typically 64 or…
- `displacement_shorter_than_skin_width_is_not_discarded` (audit_character.rs) — known defect: AUD-A-S3W3-004: move_and_slide breaks out when |displacement| < skin_width before moving, so a 5…
- `dissimilar_bond_never_exceeds_the_weaker_partners_yield` (audit_bimaterial.rs) — known defect: AUD-A-S4W3-012: design question: a dissimilar bond (half the geometric mean of the yields) can e…
- `doc_exponent_5_is_also_inexpressible_at_hertz_pressures` (audit_rolling_contact.rs) — known defect: AUD-A-S4W2-008: m = 5 でも 3 GPa で N < 1 (f32::MAX の C でも 0 cycle)
- `doc_exponent_range_9_to_10_is_expressible_in_pa_f32` (audit_rolling_contact.rs) — known defect: AUD-A-S4W2-008: doc の m = 9-10 (高強度合金) は Pa / f32 では表せない (C = N0 s0^9 = 3.8e60 > f32::MAX、f32::M…
- `dof_count_sums_joint_degrees_of_freedom` (audit_articulation.rs) — known defect: AUD-A-S3W1-019: dof_count returns the number of jointed links (2 for Ball + Hinge), not the degr…
- `draw_aabbs_flag_draws_something` (audit_debug_render.rs) — known defect: AUD-A-S4W3-033: DebugDrawFlags documents draw_aabbs (Draw body AABBs, on by default) and draw_bv…
- `draw_joints_also_draws_the_world_joint_list` (audit_debug_render.rs) — known defect: AUD-A-S4W3-009: draw_joints draws only distance_constraints; the world's joint list (ball / hing…
- `editing_the_public_triangles_keeps_queries_consistent` (audit_trimesh.rs) — known defect: AUD-A-S4W2-016: triangles は pub field だが BVH は構築時の AABB のまま。triangles[i] を書き換えると raycast / colli…
- `empty_source_list_does_not_panic` (audit_vibration_wall.rs) — known defect: AUD-A-S2W1-002: analyze_wall_resonance(&[]) panics (index out of bounds) with no documented prec…
- `entry_name_truncated_inside_a_multibyte_character_is_not_lost` (audit_pipeline.rs) — known defect: AUD-A-S5W3-018: MetricEntry::new truncates the name at byte 64 even inside a multi-byte characte…
- `entry_total_never_decreases_on_overflow` (audit_profiling.rs) — known defect: AUD-A-S4W1-001: ProfileEntry::record `+=` overflows (debug panic / release wrap u64::MAX+u64::MA…
- `escape_index_above_the_24_bit_field_is_not_silently_truncated` (audit_bvh.rs) — known defect: AUD-A-S3W2-011: the escape index is stored in 24 bits and ESCAPE_NONE (u32::MAX) is masked to 0x…
- `exp_is_accurate_up_to_the_representable_maximum` (audit_math.rs) — known defect: AUD-A-S1W5-020: Fix128::exp saturates from x >= 43.0 (hi >= 43) but e^43 = 4.73e18 .. e^43.66 = …
- `explosion_fractional_power_follows_the_documented_formula` (audit_force.rs) — known defect: AUD-A-S4W3-015: Explosion documents the falloff as (1 - dist/radius)^falloff_power but truncates…
- `explosion_with_a_huge_power_returns_promptly` (audit_force.rs) — known defect: AUD-A-S4W3-016: Explosion evaluates (1 - d/R)^n by an n-iteration loop on the truncated exponent…
- `feet_position_is_the_capsule_bottom` (audit_character.rs) — known defect: AUD-A-S3W3-003: feet_position doc says capsule bottom but returns the lower hemisphere centre (c…
- `fit_two_modes_never_returns_negative_damping_coefficients` (audit_damping_rayleigh.rs) — known defect: AUD-A-S2W2-001: fit_two_modes returns beta=-2.083e-5 for (100 rad/s, z=0.05) and (500 rad/s, z=0…
- `fos_is_never_below_one_when_applied_is_below_allowable` (audit_layer_adhesion.rs) — known defect: AUD-A-S1W5-030 (downstream: component_fos): for applied = 2^-64 or 2^-63 the quotient allowable …
- `from_f64_non_finite_and_out_of_range_are_not_silently_plausible` (audit_math.rs) — known defect: AUD-A-S1W5-018: Fix128::from_f64(NaN) returns ZERO silently, from_f64(+/-inf) and from_f64(-1e30…
- `froude_krylov_doc_excess_over_mean_is_zero_at_mean_level` (audit_wave_ship.rs) — known defect: AUD-A-S2W3-005: doc says result is buoyancy in excess of the mean, but eta = 0 returns the full …
- `froude_krylov_is_not_negative_when_the_keel_is_out_of_the_water` (audit_wave_ship.rs) — known defect: AUD-A-S2W3-004: froude_krylov_vertical_n returns negative force (rho g A (d+eta) = -2.4e5 N at d…
- `gamma_one_riemann_invariants_are_not_a_silent_identity` (audit_compressible.rs) — known defect: AUD-A-S1W5-007: riemann_invariants(gamma=1) silently returns (u,u) (J+ == J-) although 2a/(gamma…
- `gapped_series_returns_exactly_the_recorded_pairs` (audit_db_bridge.rs) — known defect: AUD-A-S5W1-001: non-contiguous steps are re-spaced uniformly by the storage layer; a query for a…
- `grid_pipeline_detects_thin_plate_parallel_to_scan_axis` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-004: analyze_thickness_grid only scans X-lines; a 0.4 mm z-thin plate parallel to X i…
- `hover_with_add_force_keeps_altitude_for_every_substep_count` (analytic_external_force_substep.rs) — src gap: add_force is a frame-head impulse while gravity is per substep, so a hovering body climbs n g dt^2 (s…
- `hover_with_force_field_keeps_altitude_for_every_substep_count` (analytic_external_force_substep.rs) — src gap: force fields are applied once at the head of the frame while gravity is per substep, so a field-held …
- `huge_stress_must_not_wrap_to_safe` (audit_laminate_failure.rs) — known defect: AUD-A-S4W2-004: sigma1 = 3.2e9 MPa で sigma1^2 が 2^63 を超え wrap、Tsai-Hill FI = -3.6e12 (負)
- `in_contact_is_false_on_the_frame_the_contact_ends` (audit_solver.rs) — known defect: AUD-A-S1W2-003: observe_body.in_contact is true on the frame of an End event (any contact event …
- `invalid_parameters_do_not_produce_non_finite_output` (audit_privacy.rs) — known defect: AUD-A-S4W3-024: parameters that make a mechanism meaningless are accepted silently: Laplace with…
- `joint_solve_through_a_reference_bridge_matches_the_cpu_solve` (audit_solver.rs) — known defect: AUD-A-S1W2-006: solve_joints_with_bridge writes back positions only; the rotation corrections of…
- `k_epsilon_point_source_is_one_explicit_euler_step_from_the_start_of_the_step_state` (audit_cfd_solver_rans.rs) — known defect: AUD-A-S1W4-009: the k-eps point source is documented as `explicit` (advance_epsilon: `one explic…
- `laplace_sample_is_always_finite_even_when_the_uniform_draw_is_zero` (audit_privacy.rs) — known defect: AUD-A-S4W3-025: LaplaceNoise::sample can return -inf: the uniform draw U = 0 (reachable, probabi…
- `lateral_margin_is_the_same_on_both_sides` (audit_heightfield.rs) — known defect: AUD-A-S4W3-004: collide_sphere margin is 2 cells before the origin but 3 cells past the last ver…
- `lattice_cell_count_is_robust_to_f32_rounding_of_the_quotient` (audit_sdf_fem_mesh.rs) — known defect: AUD-A-S2W3-010: nx = trunc((max-min)/cell) in f32 drops the last layer when the quotient is 1 ul…
- `magnetic_force_on_the_axis_pulls_toward_the_dipole_as_the_code_comment_says` (audit_force.rs) — known defect: AUD-A-S4W3-017: the Magnetic code comment says a body on the dipole axis 'is attracted', but wit…
- `mark_bulk_carries_at_least_theta_times_total_even_at_ulp_scale` (audit_linear_elastic_fem.rs) — known defect: AUD-A-S1W3-002: mark_bulk truncates total*theta, so at ulp-scale indicators the marked set carri…
- `mat3_inverse_of_small_scale_matrix` (audit_math.rs) — known defect: AUD-A-S1W5-024: Mat3Fix::inverse of a small non-singular matrix (diag 1e-5 I: det 1e-15 quantise…
- `matvec_output_is_the_exact_product_of_the_integer_sum_and_the_scale` (audit_neural.rs) — known defect: AUD-A-S3W2-016: fix128_ternary_matvec doc says 'No rounding error', but the final `acc * scale` …
- `max_queries_bounds_what_one_dispatch_covers` (audit_gpu_sdf.rs) — known defect: AUD-A-S5W2-009: GpuDispatchConfig::max_queries is documented as \"Maximum queries per dispatch\"…
- `mesh_closest_point_is_correct_for_queries_far_from_the_mesh` (audit_trimesh.rs) — known defect: AUD-A-S4W2-010: 候補を点の ±1000 の箱で絞るため、メッシュから 1000 より遠い点では候補が 0 件になり triangles[0].v0 (index 0) を返す …
- `missing_velocity_does_not_dilute_the_mean` (audit_flow_viz.rs) — known defect: AUD-A-S4W1-007: generate_flow_arrows counts position-only particles in the mean (magnitude 1 ins…
- `nan_courant_is_rejected` (audit_acoustic_wave.rs) — known defect: AUD-A-S5W1-002: leapfrog_step accepts NaN / unstable Courant numbers (C > 1) with no check and n…
- `nan_sdf_distance_is_not_masked_as_still_air` (audit_sdf_wind_field.rs) — known defect: AUD-A-S6W1-002: NaN SDF distance is masked to zero wind (finite 0.0) instead of propagating
- `negative_dimensions_do_not_break_volume_or_support` (audit_cylinder.rs) — known defect: AUD-A-S6W1-008: half_height=-1 gives volume -6.283, radius=-2 gives support x=-2 for +X (no dime…
- `negative_internal_pressure_contracts_the_surface` (audit_pressure.rs) — known defect: AUD-A-S6W1-006: negative internal_pressure has no effect (no inward contraction), doc says only …
- `negative_max_force_must_not_produce_force_at_zero_error` (audit_motor.rs) — known defect: AUD-A-S5W2-001: negative max_force makes clamp(v, -max, +max) an empty interval and compute retu…
- `negative_max_torque_must_not_reverse_the_torque` (audit_motor.rs) — known defect: AUD-A-S5W2-004: negative max_torque is not rejected; `mag > max` is always true and torque * (ma…
- `negative_minor_radius_is_rejected_or_still_maximises` (audit_torus.rs) — known defect: AUD-A-S5W3-005: Torus::new accepts a negative minor radius; support(+X) with R=5, r=-1 returns x…
- `negative_threshold_does_not_flag_the_centre` (audit_anomaly.rs) — known defect: AUD-A-S4W1-016: negative threshold_k flags the median itself (deviation 0 > negative threshold)
- `non_unit_rotation_does_not_scale_the_support_point` (audit_cylinder.rs) — known defect: AUD-A-S6W1-009: rotation quaternion (0,0,0,2) gives support (4,4,0) instead of (1,1,0), 4x too f…
- `non_unit_rotation_must_not_scale_the_box` (audit_box_collider.rs) — known defect: AUD-A-S5W2-018: the orientation quaternion is not normalised or checked, so a non-unit rotation …
- `non_unit_rotation_must_not_scale_the_ellipsoid` (audit_ellipsoid.rs) — known defect: AUD-A-S5W2-019: the orientation quaternion is not normalised or checked, so a non-unit rotation …
- `non_unit_rotation_must_not_scale_the_wedge` (audit_wedge.rs) — known defect: AUD-A-S5W2-020: the orientation quaternion is not normalised or checked, so a non-unit rotation …
- `non_unit_target_quaternion_must_not_scale_the_torque` (audit_motor.rs) — known defect: AUD-A-S5W2-005: a non-unit target or current quaternion is not normalised; scaling the target qu…
- `normal_on_the_border_of_a_plane_is_the_plane_normal` (audit_heightfield.rs) — known defect: AUD-A-S4W3-003: sample_normal at a grid border point of a plane returns half the slope (clamped …
- `opening_a_player_on_a_missing_directory_does_not_create_it` (audit_replay.rs) — known defect: AUD-A-S4W2-009: ReplayPlayer::open は存在しない dir を作成し (alice-db の open が create)、存在しない replay でも Ok…
- `optimize_does_not_restore_material_carved_by_disjoint_craters` (audit_sdf_destruction.rs) — known defect: AUD-A-S5W3-009: optimize() is documented as removing shapes fully contained by newer ones, but d…
- `origin_y_offsets_the_surface` (audit_heightfield.rs) — known defect: AUD-A-S4W3-006: origin.y is never read (sample_height / signed_distance / aabb use the stored he…
- `out_of_range_quotient_keeps_its_sign_and_magnitude` (audit_math.rs) — known defect: AUD-A-S1W5-030: Fix128::div truncates an out-of-range integer quotient to its low 64 bits, so 1 …
- `overlap_aabb_expanded_does_not_report_spheres_that_miss_the_corner` (audit_query.rs) — known defect: AUD-A-S3W3-015: overlap_aabb_expanded inflates the AABB into a bigger box (Minkowski sum with a …
- `particle_inside_a_non_cubic_cell_contributes` (audit_flow_viz.rs) — known defect: AUD-A-S4W1-008: averaging radius uses dx for all axes; particle inside a dy=100 cell, 5 from its…
- `particle_system_agrees_with_compute_force_for_magnetic_explosion_and_vortex` (audit_force.rs) — known defect: AUD-B-S4W3-001: particle.rs keeps a private second implementation of the ForceField laws that di…
- `per_query_radius_contributes_to_penetration` (audit_gpu_sdf.rs) — known defect: AUD-A-S5W2-010: GpuSdfQuery::radius (\"for sphere-SDF test\") is carried to the buffer but never…
- `pipeline_colliding_metric_names_do_not_corrupt_each_other` (audit_pipeline.rs) — known defect: AUD-A-S5W3-016: two metrics whose hashes agree modulo SLOTS share one slot: the second metric's …
- `pipeline_histogram_with_alpha_one_percent_keeps_quantiles_for_values_above_fifty` (audit_pipeline.rs) — known defect: AUD-A-S5W3-019: MetricPipeline::new documents alpha = 0.01 as a normal choice, but a 256-bin ske…
- `plate_with_negative_dimension_is_rejected_like_the_beam_with_negative_length` (audit_modal.rs) — known defect: AUD-A-S3W2-002: plate_natural_frequency_hz treats negative thickness / side as its absolute valu…
- `point_slightly_outside_surface_is_not_reported_as_a_hairline_wall` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-005: measure_thickness_at returns Some(0.01) for a surface point 0.011 mm outside the…
- `polar_rotation_is_idempotent_bit_for_bit_on_general_gradients` (audit_math.rs) — known defect: AUD-A-S1W5-027: polar_rotation is not idempotent for general (non-diagonal) gradients: 105 of 20…
- `puck_pure_shear_above_s_is_inter_fibre_not_fibre_tension` (audit_laminate_failure.rs) — known defect: AUD-A-S4W2-005: sigma1 = 0, sigma2 = -50, tau = 75 (> S) の puck が FibreTension を返す (Hashin の fib…
- `pulley_ratio_two_conserves_the_rope_length` (audit_c_joint_extra.rs) — known defect: AUD-A-S34-012: same root as AUD-A-S3W1-011 for ratio 2: `solve_pulley` uses `error = Fix128::ZER…
- `rack_and_pinion_uses_the_axis_inverse_inertia` (audit_joint_extra.rs) — known defect: AUD-A-S3W1-013: rack-and-pinion (and gear / weld) use |inv_inertia| (sqrt(3) for isotropic i = 1…
- `ragdoll_joint_anchors_coincide_at_build_time` (audit_articulation.rs) — known defect: AUD-A-S3W1-018: build_ragdoll spine / chest / head / leg joints put anchor_a at +1 (or -1) on th…
- `rappor_default_params_epsilon_is_about_two` (audit_privacy.rs) — known defect: AUD-A-S4W3-023: Rappor::default_params documents 'approximately epsilon = 2' but with f = 0.5, p…
- `ray_capsule_along_axis_hits_near_cap` (audit_raycast.rs) — known defect: AUD-A-S5W1-003: ray_capsule returns None for a ray parallel to the capsule axis (a_coeff == 0 ea…
- `ray_capsule_parallel_offset_inside_radius_hits_cap` (audit_raycast.rs) — known defect: AUD-A-S5W1-003: ray_capsule returns None for a ray parallel to the capsule axis (a_coeff == 0 ea…
- `recommended_fillet_trivial_target_gives_zero_radius` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-006: recommended_fillet_radius_mm returns ~0.01*d for a target every radius satisfies…
- `recommended_fillet_unreachable_target_not_silently_returned` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-005: recommended_fillet_radius_mm returns d/2 (K_t=1.43 > target 1.2) for an unreacha…
- `reconcile_weighted_is_atomic_on_error` (audit_coupled_field.rs) — known defect: AUD-A-S1W5-017: reconcile_weighted (and reconcile_mean) adopt participant by participant; if par…
- `recorded_position_equals_the_engine_state_for_integers_above_2_pow_24` (audit_replay.rs) — known defect: AUD-A-S4W2-015: 記録値は Fix128 状態ではなく to_f32 の丸め (24 bit 仮数)。x = 16777217 を記録すると 16777216 で戻る (エンジン…
- `refinement_keeps_every_element_positively_wound` (audit_sdf_fem_mesh.rs) — known defect: AUD-A-S2W3-012: try_refine_conforming / try_refine_marked emit children with mixed winding (372 …
- `refit_keeps_the_world_bounds_field_in_sync_with_the_primitives` (audit_bvh.rs) — known defect: AUD-A-S3W2-009: LinearBvh::refit_leaves refreshes every node box but leaves the public `bounds` …
- `reinit_every_two_steps_reinitialises_at_the_end_of_the_second_step` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-004: reinit_every_n_steps = N first reinitialises on step N + 1 (the test is step_cou…
- `remove_body_keeps_the_contact_history_of_the_survivors` (audit_solver.rs) — known defect: AUD-A-S1W2-004: remove_body does not remap the event pair history; the surviving pair re-reports…
- `repeated_impacts_never_exceed_max_deformation_before_update` (audit_pressure.rs) — known defect: AUD-A-S6W1-003: two apply_impact calls at one node give deformation 2.0 with max_deformation 1.0…
- `reported_force_does_not_vanish_at_zero_step` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-017: apply_fluid_forces_to_cloth_with_residual returns 0 at dt == 0 although the repo…
- `residual_rotates_the_local_anchor_into_world_space` (audit_kinematic_loop.rs) — known defect: AUD-A-S3W3-002: residual/apply add local_anchor to position without rotating by body.rotation, s…
- `residual_stress_accounts_for_unequal_layer_thickness` (audit_bimaterial.rs) — known defect: AUD-A-S4W3-010: thermal_residual_stress_mpa ignores layer thickness (equal-thickness formula) al…
- `restore_rejects_a_snapshot_whose_buffer_no_longer_matches_its_checksum` (audit_fluid_netcode.rs) — known defect: AUD-A-S5W2-014: restore() does not check the snapshot checksum, so a snapshot whose buffer was a…
- `result_velocity_has_no_component_into_the_wall_after_a_head_on_slide` (audit_character.rs) — known defect: AUD-A-S3W3-005: MoveResult.velocity doc says velocity after sliding but move_and_slide always re…
- `rigid_half_turn_in_an_even_number_of_increments_is_stress_free` (audit_c_linear_elastic_fem.rs) — known defect: AUD-A-S34-030: `solve_corotational` applies a rigid 180 degree rotation in an even number of inc…
- `rigid_weld_closes_a_small_angular_error_in_one_projection` (audit_joint_extra.rs) — known defect: AUD-A-S3W1-013: solve_weld's angular effective inverse mass is |inv_inertia| (vector norm, sqrt(…
- `rotation_error_magnitude_equals_the_angle_for_a_quarter_turn` (audit_motor.rs) — known defect: AUD-A-S5W2-002: the in-code comment calls the position error \"axis-angle\" but the controller u…
- `sampled_points_stay_inside_the_aabb` (audit_thin_wall.rs) — known defect: AUD-A-S2W1-006: sample_surface_points returns points beyond aabb_max (ceil in axis_steps): x=5 >…
- `self_referential_ball_joint_with_distinct_anchors_is_a_free_body` (audit_c_joint.rs) — known defect: AUD-A-S34-010: `solve_ball_joint` with `BallJoint(a, a)` and distinct local anchors (1,0,0) / (-…
- `separating_normal_velocity_is_not_reversed_by_the_contact` (audit_rope.rs) — known defect: AUD-A-S3W1-004: resolve_sdf_collisions sets v_n' = -0.1 v_n unconditionally, reversing a separat…
- `separation_distance_matches_closed_form_diagonal_equal_radius` (analytic_gjk_separation_distance.rs) — src gap: fixed-iteration GJK converges inexactly for a diagonal separation even at equal radii (dist=5 r1=1 r2…
- `separation_distance_matches_closed_form_off_axis_unequal_radius` (analytic_gjk_separation_distance.rs) — src gap: fixed-iteration GJK converges inexactly for unequal-radius separated spheres along a non-x axis (dist…
- `separation_load_for_c_above_one_is_not_negative` (audit_prestressed.rs) — known defect: AUD-A-S1W6-001: separation_load_n(1000, C=1.2) returns -5000 (negative load, no clamp/guard for …
- `shoulder_bending_at_d_over_d_two_matches_documented_table` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-001: kt_shaft_shoulder_bending at D/d=2 returns 1.1x the module's own D/d=2 table (r/…
- `shoulder_bending_is_continuous_in_fillet_radius` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-003: kt_shaft_shoulder_bending is a piecewise-CONSTANT step table (jump 2.9->2.2 at r…
- `shoulder_bending_without_step_is_unity` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-002: kt_shaft_shoulder_bending(D=d) returns the fillet table value (1.3..3.0), not 1
- `slerp_has_constant_angular_velocity` (audit_interpolation.rs) — known defect: AUD-A-S2W3-006: slerp is NLERP; 90 deg arc at t=1/4 is 21.6 deg not 22.5 (angular speed non-unif…
- `slot_merge_keeps_the_gauge_when_the_other_slot_never_wrote_one` (audit_pipeline.rs) — known defect: AUD-A-S5W3-015: MetricSlot::merge overwrites the gauge with the other slot's value whenever that…
- `small_triangles_are_not_invisible_to_rays` (audit_trimesh.rs) — known defect: AUD-A-S4W2-012: MT_EPSILON (2^-24) との比較が絶対値で、det = 2 * 面積 * |cos| が小さい三角形 (辺 1e-4 m、面積 5e-9) は真上…
- `snapshot_of_a_slot_without_histogram_data_has_finite_extrema` (audit_pipeline.rs) — known defect: AUD-A-S5W3-020: a snapshot of a slot without histogram data reports min = +inf and max = -inf wh…
- `snapshot_restore_is_bit_identical_with_participants_in_parallel` (world_participant_conformance.rs) — src gap: step_parallel stores batch bookkeeping in the snapshot that step does not (single pipeline, separate …
- `solve_cubic_never_reports_ok_for_a_body_nothing_holds` (audit_cubic_elastic_fem.rs) — known defect: AUD-A-S2W1-008: solve_cubic returns Ok (u ~ 1e11 mm, relative_residual 0) for a body with no con…
- `speculative_contact_reports_coincident_overlapping_spheres` (audit_ccd.rs) — known defect: AUD-A-S2W2-009: speculative_contact returns None for coincident centres (dist == 0 guard) althou…
- `sphere_capsule_toi_sliding_approach_hits_at_the_right_time` (audit_ccd.rs) — known defect: AUD-A-S2W2-011: sphere_capsule_toi freezes the closest axis point at the start position (treats …
- `spring_by_add_force_holds_its_static_extension` (analytic_external_force_substep.rs) — src gap: a spring force applied by add_force is a frame-head impulse, so the static extension m g / k is not h…
- `stagnation_pressure_ratio_is_monotone_and_positive_at_large_mach` (audit_compressible.rs) — known defect: AUD-A-S1W5-008: stagnation_pressure_ratio wraps silently when p0/p exceeds the Fix128 range (M=1…
- `stair_step_with_a_short_capsule_does_not_lower_the_character` (audit_character.rs) — known defect: AUD-A-S3W3-019: the stair snap-down ray is cast against the character's own previous feet plane …
- `static_body_grounds_within_probe_of_the_capsule_bottom_like_an_sdf_floor` (audit_character.rs) — known defect: AUD-A-S3W3-006: detect_ground body branch rays from the hemisphere centre against a sphere of ra…
- `static_failure_at_ultimate_strength_is_one_cycle` (audit_fatigue.rs) — known defect: AUD-A-S4W2-006: ultimate_tensile_mpa は寿命計算で一切使われない。PLA 曲線で S = UTS の寿命は N_e (0.3)^5 = 2430 cycle…
- `step_height_limit_blocks_obstacles_taller_than_step_height` (audit_character.rs) — known defect: AUD-A-S3W3-008: stair step check samples only the capsule centre at test_pos so step_height = 0.…
- `step_with_options_refuses_or_bounds_an_unstable_explicit_diffusion_number` (audit_cfd_solver.rs) — known defect: AUD-A-S1W4-005: step / step_with_options accept nu dt / dx^2 = 0.6 > 1/6 and the checkerboard mo…
- `strain_far_above_tg_does_not_collapse_to_the_elastic_strain` (audit_c_creep_longterm.rs) — known defect: AUD-A-S34-031: `predict_strain` returns `epsilon_0` (no creep) once `ln10 * log10 a_T <= -40`, m…
- `strain_is_monotone_in_temperature_across_the_shift_factor_underflow` (audit_c_creep_longterm.rs) — known defect: AUD-A-S34-031: strain drops from 0.2061 at `T_g + 13100` to 0.001 at `T_g + 13200` (`m = 2^-60`,…
- `streamline_tracing_is_scale_covariant` (audit_flow_viz.rs) — known defect: AUD-A-S4W1-006: influence radius hard-coded to 1.0; uniform flow with 2.5 m particle spacing yie…
- `stress_map_is_covariant_under_uniform_scaling_of_the_scene` (audit_heatmap.rs) — known defect: AUD-A-S5W3-006: the stress kernel radius is a hard-coded 2.0 world units (not in HeatmapConfig),…
- `sub_unit_objects_keep_the_broad_phase_selective` (audit_bvh.rs) — known defect: AUD-A-S3W2-013: node boxes are quantised to whole world units (floor/ceil to i32), so a scene wh…
- `support_huge_direction_does_not_wrap` (audit_cone.rs) — known defect: AUD-A-S4W3-002: support() with |dir|=5e9 (xz_len_sq wraps in Fix128) returns x=1.953 instead of …
- `surface_tension_pulls_the_cloth_toward_the_fluid_centre` (audit_cloth_fluid.rs) — known defect: AUD-A-S2W2-005: surface_tension term is mean_fluid_velocity * k (zero for resting fluid), not a …
- `symmetric_stack_b_is_exactly_zero_as_documented` (audit_c_laminate.rs) — known defect: AUD-A-S34-050: `compute_abd` returns B = -(n/2) * 2^-64 in every entry for a mirrored stack of n…
- `symmetric_stack_has_exactly_zero_b` (audit_laminate.rs) — known defect: AUD-A-S1W5-009: compute_abd leaves B != 0 (rounding residue b11 = -2^-64 (raw hi=-1,lo=u64::MAX-…
- `tangential_loss_per_frame_is_independent_of_the_substep_count` (audit_rope.rs) — known defect: AUD-A-S3W1-005: sdf_friction is a per-contact velocity retention (1-mu) applied each substep and…
- `tanh_approx_is_bounded_by_one_and_monotone` (audit_neural.rs) — known defect: AUD-A-S3W2-004: fix128_tanh_approx (Activation::TanhApprox, doc 'smooth bounded output') returns…
- `tanh_approx_max_error_matches_the_documented_0_004` (audit_neural.rs) — known defect: AUD-A-S3W2-003: fix128_tanh_approx doc claims a Pade approximant with ~0.004 max error for |x| <…
- `tgs_isotropic_spin_turns_by_omega_t` (indep_free_rotation_closed_forms.rs) — src gap: TGS turns an isotropic body by 2 atan(|w| h / 2) per substep, the angle lags by |w|^3 h^2 t / 12
- `the_answer_does_not_depend_on_the_increment_count` (analytic_corotational.rs) — the red is correct: bit-identical displacements across increment counts need an exact fixed point of the frame…
- `the_clamped_counter_of_k_omega_counts_cells_as_documented` (audit_cfd_solver_rans.rs) — known defect: AUD-A-S1W4-010: `TurbulenceSummary::clamped` is documented as the `Number of cells` whose k, eps…
- `the_reattachment_length_matches_gartling` (armaly_backward_step.rs) — src gap: measured 2026-10-02 on this scene (ny = 8, default SemiLagrangian, L = 16 so nx = 128 is a power of t…
- `the_undamped_pendulum_keeps_its_amplitude` (indep_tgs_pendulum_static_plane.rs) — src gap: joint projection loses swing energy at first order in h (amplitude 0.6 to 0.517 in 12 s at 4 substeps…
- `to_f64_is_relatively_accurate_for_small_negative_values` (audit_math.rs) — known defect: AUD-A-S1W5-025: Fix128::to_f64 computes hi + lo/2^64 in f64, so a small negative value loses its…
- `total_dispatches_counts_unique_sdf_ids` (audit_gpu_sdf.rs) — known defect: AUD-A-S5W2-011: GpuSdfMultiDispatch docs say one dispatch per unique sdf_id, but add_batch never…
- `total_particle_mass_is_mass_per_unit_times_length` (audit_rope.rs) — known defect: AUD-A-S3W1-003: Rope::new total particle mass = mpu*L*(N+1)/N, not mpu*L (measured N=1: 12 vs 6)…
- `transfer_preserves_collision_filter` (audit_multi_world.rs) — known defect: AUD-A-S4W1-005: transfer_body drops the body's collision filter (custom -> DEFAULT)
- `transfer_preserves_collision_radius` (audit_multi_world.rs) — known defect: AUD-A-S4W1-005: transfer_body drops the collision radius, so the transferred body no longer coll…
- `transfer_preserves_material` (audit_multi_world.rs) — known defect: AUD-A-S4W1-005: transfer_body drops the body's material id (3 -> 0)
- `u_notch_kt_never_below_one` (audit_fillet_stress.rs) — known defect: AUD-A-S1W5-004: kt_u_notch_axial returns 0.913 at h/r=1e-3 and 0.85 at h=0 (<1); formula has no …
- `upwind_slab_translates_k_cells_for_k_greater_than_one` (audit_multiphase.rs) — known defect: AUD-A-S2W3-008: advect_vof_rigid doc promises a k-cell translation 'under either scheme' for dt …
- `values_that_fill_the_window_are_not_anomalous_when_mad_is_zero` (audit_anomaly.rs) — known defect: AUD-A-S4W1-013: MAD = 0 branch flags 10.5 (40 of 100 window samples) as anomalous with score +in…
- `vec3_normalize_small_nonzero_vectors` (audit_math.rs) — known defect: AUD-A-S1W5-022: Vec3Fix::normalize / try_normalize / normalize_with_length treat any vector with…
- `wall_within_band_of_a_non_nearest_source_is_risky` (audit_vibration_wall.rs) — known defect: AUD-A-S2W1-001: is_risky checks only the abs-nearest source; w within +-20% of a farther (larger…
- `weld_angular_correction_is_split_by_inverse_inertia` (audit_joint_extra.rs) — known defect: AUD-A-S3W1-013: apply_angular_correction rotates both bodies by the same angle regardless of inv…
- `world_hover_holds_altitude` (analytic_rotor.rs) — src gap: frame 先頭の外力 impulse と substep 重力の splitting (world 側に substep 内で外力を掛ける経路が無い)
- `world_reaction_torque_spin_rate_matches_free_rotation` (analytic_rotor.rs) — src gap: 回転の積分で角速度が substep ごとに ω ← (2/h) sin(ωh/2) と目減りする (率 ω³h/24、h は substep 幅)
- `youngs_at_angle_is_a_lower_bound_reuss_form` (audit_filament_db.rs) — known defect: AUD-A-S1W5-012: youngs_at_angle doc says 'Reuss-like lower bound' but E_xy cos2 + E_z sin2 is th…
- `zero_decay_scale_still_blocks_wind_inside_the_obstacle` (audit_sdf_wind_field.rs) — known defect: AUD-A-S6W1-001: decay_scale_m = 0 returns full base wind (10 m/s) inside the obstacle (d = -1) w…
- `zero_layer_network_forward_does_not_panic` (audit_neural.rs) — known defect: AUD-A-S3W2-007: DeterministicNetwork::forward on a zero-layer network panics (index out of bound…
- `zero_length_column_is_governed_by_yield_not_zero` (audit_buckling.rs) — known defect: AUD-A-S2W2-007: zero-length column reports critical_stress = 0 and critical_load = 0 with regime…
- `zero_max_velocity_saturates_volume_instead_of_silencing` (audit_audio_physics.rs) — known defect: AUD-A-S3W2-008: with config.max_velocity = 0 every speed is 'above max velocity' (doc: velocitie…
- `zero_strength_with_nonzero_stress_is_not_safe` (audit_anisotropic.rs) — known defect: AUD-A-S1W6-005: a zero strength is treated as unlimited (max-stress skips the component, Hill/Ts…

## 📌 Pins: tests that turn red when a known defect is fixed (4)

Tests marked `// PIN: <defect id>` assert today's defective behaviour on purpose. Fixing the
defect makes them red; update them in the same change, after checking that the new behaviour
is the intended one.

| Defect | Pin test | Defect test |
|--------|----------|-------------|
| AUD-A-S34-030 | `rigid_half_turn_in_two_increments_is_currently_refused_as_inverted` (audit_c_linear_elastic_fem.rs) | `rigid_half_turn_in_an_even_number_of_increments_is_stress_free` (audit_c_linear_elastic_fem.rs) |
| AUD-A-S3W1-010 | `count_min_degenerate_inputs` (analytic_sketch_wiring.rs) | `countmin_total_does_not_overflow_when_counters_saturate` (audit_sketch.rs) |
| AUD-A-S3W2-007 | `deterministic_network_zero_layers_constructs_but_forward_panics` (analytic_neural_wiring.rs) | `zero_layer_network_forward_does_not_panic` (audit_neural.rs) |
| AUD-A-S4W1-001 | `entry_record_overflow_panics_in_debug_and_wraps_in_release` (analytic_profiling_wiring.rs) | `entry_total_never_decreases_on_overflow` (audit_profiling.rs) |

## 🌐 Root cause outside this repository (1)

Known defects whose reason says `root: external <crate> <version>`: the fix belongs in that
dependency. When Cargo resolves a different version (Cargo.lock, or `cargo metadata --all-features`
when the lock is not committed), re-check whether the defect remains.

| Defect | Test | Crate | Reason says | Resolved | Status |
|--------|------|-------|-------------|----------|--------|
| AUD-A-S5W1-001 | `gapped_series_returns_exactly_the_recorded_pairs` (audit_db_bridge.rs) | `alice-db` | 0.2.0-beta.3 | 0.3.0-beta.1 | ⚠️ re-check |

## ⏱ Gated (26)

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

- `a_ball_is_cradled_in_the_hole_of_a_torus` (indep_shaped_sphere_rest_heights.rs) — known limitation: a torus collides as its convex hull, a ball cannot enter the hole (rests at rho + r = 1.3, w…
- `adaptive_cubic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 40 s in release (P3 reference at two uniform passes, a uniform coarse solve and a six-round ada…
- `adaptive_quadratic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 2 s in release, measured 2026-10-03 (P2 reference at two uniform passes, a uniform coarse solve…
- `amplification_growth_is_problem_size_or_element_shape` (mesh_to_fem_stress.rs) — diagnostic: run when the amplification threshold is in question
- `bottleneck_specific_flow_matches_experiments_in_order_of_magnitude` (analytic_crowd_force.rs) — runtime: about 30 s in release (2 widths, 40 pedestrians, up to 15000 Fix128 steps each); run by run_ignored.p…
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

## 🟢 Not ignored (5318)

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
| `integration_physics.rs` | 75 |
| `world_participant_conformance.rs` | 51 |
| `audit_joint.rs` | 46 |
| `audit_solver.rs` | 42 |
| `engineering_oracles_fluid.rs` | 40 |
| `analytic_maxwell_fdtd.rs` | 38 |
| `analytic_raycast_wiring.rs` | 37 |
| `analytic_world_participant_wiring.rs` | 37 |
| `analytic_coordinate_range.rs` | 36 |
| `analytic_shape_raycast.rs` | 36 |
| `analytic_sdf_character_up_axis.rs` | 35 |
| `analytic_world_api.rs` | 35 |
| `audit_coupled_iteration.rs` | 34 |
| `engineering_oracles_solid.rs` | 34 |
| `analytic_world_character.rs` | 31 |
| `analytic_world_shape_query.rs` | 29 |
| `audit_math.rs` | 29 |
| `analytic_crowd_force.rs` | 28 |
| `audit_raycast.rs` | 28 |
| `audit_trimesh.rs` | 28 |
| `analytic_static_collider.rs` | 27 |
| `audit_motor.rs` | 27 |
| `engineering_oracles_misc.rs` | 27 |
| `analytic_euler_fv.rs` | 26 |
| `analytic_maxwell_wiring.rs` | 26 |
| `analytic_particle_wiring.rs` | 26 |
| `audit_cfd_solver_rans.rs` | 26 |
| `audit_query.rs` | 26 |
| `analytic_character_wiring.rs` | 25 |
| `analytic_p2g.rs` | 25 |
| `analytic_plastic_dissipation.rs` | 25 |
| `analytic_query_wiring.rs` | 25 |
| `audit_cfd_solver.rs` | 25 |
| `audit_character.rs` | 25 |
| `audit_collider.rs` | 25 |
| `audit_heightfield.rs` | 25 |
| `audit_joint_extra.rs` | 25 |
| `analytic_compound_wiring.rs` | 24 |
| `analytic_physics2d_pairs.rs` | 24 |
| `analytic_transient_thermal_wiring.rs` | 24 |
| `analytic_compound.rs` | 23 |
| `analytic_convex_contact.rs` | 23 |
| `analytic_kepler.rs` | 23 |
| `audit_linear_elastic_fem.rs` | 23 |
| `analytic_pair_potential.rs` | 22 |
| `analytic_reactions.rs` | 22 |
| `analytic_world_sweeps.rs` | 22 |
| `audit_fluid_netcode.rs` | 22 |
| `audit_force.rs` | 22 |
| `determinism_semantic.rs` | 22 |
| `analytic_adaptive_refinement.rs` | 21 |
| `analytic_corotational.rs` | 21 |
| `analytic_multiphase_wiring.rs` | 21 |
| `analytic_neural_wiring.rs` | 21 |
| `analytic_rope_wiring.rs` | 21 |
| `analytic_vehicle_dynamics.rs` | 21 |
| `audit_torus.rs` | 21 |
| `analytic_buoyancy_zone_wiring.rs` | 20 |
| `analytic_fluid_netcode_wiring.rs` | 20 |
| `analytic_joint_extra_wiring.rs` | 20 |
| `analytic_linear_solver.rs` | 20 |
| `analytic_wave_ship_wiring.rs` | 20 |
| `audit_pipeline.rs` | 20 |
| `audit_print_pipeline_solver.rs` | 20 |
| `audit_sdf_destruction.rs` | 20 |
| `analytic_compressible_wiring.rs` | 19 |
| `analytic_debug_render_wiring.rs` | 19 |
| `analytic_elastoplastic_fem.rs` | 19 |
| `analytic_heatmap_wiring.rs` | 19 |
| `analytic_laminate_wiring.rs` | 19 |
| `analytic_prestressed_wiring.rs` | 19 |
| `analytic_rolling_contact_wiring.rs` | 19 |
| `analytic_tgs_wiring.rs` | 19 |
| `audit_privacy.rs` | 19 |
| `audit_sim_modifier.rs` | 19 |
| `participant_crowd_md.rs` | 19 |
| `analytic_filament_db_wiring.rs` | 18 |
| `analytic_flip.rs` | 18 |
| `analytic_hyperelastic_wiring.rs` | 18 |
| `analytic_multi_world_wiring.rs` | 18 |
| `analytic_sdf_force_wiring.rs` | 18 |
| `analytic_spherical_terrain.rs` | 18 |
| `analytic_thin_wall_wiring.rs` | 18 |
| `analytic_wind_zone_wiring.rs` | 18 |
| `audit_anisotropic.rs` | 18 |
| `audit_bvh.rs` | 18 |
| `audit_eulerian_grid.rs` | 18 |
| `audit_hyperelastic.rs` | 18 |
| `audit_wave_ship.rs` | 18 |
| `analytic_acoustic_wave_wiring.rs` | 17 |
| `analytic_anomaly_wiring.rs` | 17 |
| `analytic_ccd_wiring.rs` | 17 |
| `analytic_gpu_sdf_wiring.rs` | 17 |
| `analytic_interpolation_wiring.rs` | 17 |
| `analytic_modal_wiring.rs` | 17 |
| `analytic_nbody.rs` | 17 |
| `analytic_phase_change_wiring.rs` | 17 |
| `audit_anomaly.rs` | 17 |
| `audit_cone.rs` | 17 |
| `audit_coupled_field.rs` | 17 |
| `audit_cubic_elastic_fem.rs` | 17 |
| `audit_neural.rs` | 17 |
| `audit_thin_wall.rs` | 17 |
| `indep_participant_laws.rs` | 17 |
| `participant_sdf_modifiers.rs` | 17 |
| `analytic_audio_physics_wiring.rs` | 16 |
| `analytic_damping_rayleigh_wiring.rs` | 16 |
| `analytic_fdtd_materials.rs` | 16 |
| `analytic_fracture_wiring.rs` | 16 |
| `analytic_lift_drag.rs` | 16 |
| `analytic_linear_elastic_fem.rs` | 16 |
| `analytic_profiling_wiring.rs` | 16 |
| `analytic_shape.rs` | 16 |
| `analytic_structural_solver_wiring.rs` | 16 |
| `analytic_vehicle_wiring.rs` | 16 |
| `analytic_world_views.rs` | 16 |
| `armaly_backward_step.rs` | 16 |
| `audit_audio_physics.rs` | 16 |
| `audit_bimaterial.rs` | 16 |
| `audit_ccd.rs` | 16 |
| `audit_interpolation.rs` | 16 |
| `audit_maxwell_fdtd.rs` | 16 |
| `analytic_articulation_wiring.rs` | 15 |
| `analytic_dynamic_fem.rs` | 15 |
| `analytic_md_crowd_try_step.rs` | 15 |
| `analytic_multibody_dynamics.rs` | 15 |
| `analytic_pipeline_wiring.rs` | 15 |
| `analytic_privacy_wiring.rs` | 15 |
| `analytic_rope_attach_wiring.rs` | 15 |
| `analytic_sdf_body_collider.rs` | 15 |
| `analytic_self_contact.rs` | 15 |
| `analytic_thermoplastic_coupling.rs` | 15 |
| `audit_cloth_fluid.rs` | 15 |
| `audit_ellipsoid.rs` | 15 |
| `audit_sdf_force.rs` | 15 |
| `analytic_collision_mesh.rs` | 14 |
| `analytic_convex_decompose.rs` | 14 |
| `analytic_joint_motor_world.rs` | 14 |
| `analytic_joint_wiring.rs` | 14 |
| `analytic_scene_io_wiring.rs` | 14 |
| `analytic_sdf_destruction_wiring.rs` | 14 |
| `analytic_sdf_fem_mesh_wiring.rs` | 14 |
| `analytic_sdf_manifold_wiring.rs` | 14 |
| `analytic_sensors.rs` | 14 |
| `analytic_smoke_fire_wiring.rs` | 14 |
| `analytic_thermal_wiring.rs` | 14 |
| `analytic_vehicle_scenario.rs` | 14 |
| `analytic_world_query_margins.rs` | 14 |
| `audit_animation_blend.rs` | 14 |
| `audit_articulation.rs` | 14 |
| `audit_fsi_advanced.rs` | 14 |
| `audit_laminate_failure.rs` | 14 |
| `audit_material.rs` | 14 |
| `audit_modal.rs` | 14 |
| `audit_multiphase.rs` | 14 |
| `audit_rope.rs` | 14 |
| `audit_sketch.rs` | 14 |
| `audit_soft_body_cut.rs` | 14 |
| `sleep_skip.rs` | 14 |
| `world_snapshot_v2.rs` | 14 |
| `analytic_anisotropic_wiring.rs` | 13 |
| `analytic_coupling_medium.rs` | 13 |
| `analytic_csf_wiring.rs` | 13 |
| `analytic_cubic_elastic_fem_wiring.rs` | 13 |
| `analytic_erosion_wiring.rs` | 13 |
| `analytic_filter_wiring.rs` | 13 |
| `analytic_gyroscopic.rs` | 13 |
| `analytic_interface_capture_wiring.rs` | 13 |
| `analytic_kinematic_loop_wiring.rs` | 13 |
| `analytic_math_wiring.rs` | 13 |
| `analytic_physics.rs` | 13 |
| `analytic_pressure_solvers.rs` | 13 |
| `analytic_rans.rs` | 13 |
| `analytic_sdf_ccd_wiring.rs` | 13 |
| `analytic_sim_modifier_wiring.rs` | 13 |
| `analytic_sketch_wiring.rs` | 13 |
| `analytic_spring_joint.rs` | 13 |
| `audit_box_collider.rs` | 13 |
| `audit_dynamic_fem.rs` | 13 |
| `audit_flow_viz.rs` | 13 |
| `audit_gpu_sdf.rs` | 13 |
| `audit_sdf_fem_mesh.rs` | 13 |
| `audit_spatial.rs` | 13 |
| `audit_wind_zone.rs` | 13 |
| `default_configs.rs` | 13 |
| `determinism_golden_f32.rs` | 13 |
| `analytic_aeroelasticity_wiring.rs` | 12 |
| `analytic_beam_stress_wiring.rs` | 12 |
| `analytic_broadphase_hybrid.rs` | 12 |
| `analytic_collision_mesh_gen_wiring.rs` | 12 |
| `analytic_contact_cache_wiring.rs` | 12 |
| `analytic_deformable_wiring.rs` | 12 |
| `analytic_layer_adhesion_wiring.rs` | 12 |
| `analytic_structural_fatigue_buckling_wiring.rs` | 12 |
| `analytic_sweep_sphere_inside.rs` | 12 |
| `audit_analytics_bridge.rs` | 12 |
| `audit_plane_collider.rs` | 12 |
| `audit_plastic.rs` | 12 |
| `audit_turbulence.rs` | 12 |
| `indep_molecular_pair_potential.rs` | 12 |
| `indep_orbit_kepler.rs` | 12 |
| `analytic_animation_blend_wiring.rs` | 11 |
| `analytic_contact_viz_wiring.rs` | 11 |
| `analytic_coupled_field.rs` | 11 |
| `analytic_flow_viz_wiring.rs` | 11 |
| `analytic_mass_properties.rs` | 11 |
| `analytic_physics2d_tethers.rs` | 11 |
| `analytic_polar_decomposition.rs` | 11 |
| `analytic_pressure_wiring.rs` | 11 |
| `analytic_replay_wiring.rs` | 11 |
| `analytic_sdf_adaptive_wiring.rs` | 11 |
| `analytic_sdf_ccd_borrowed_field.rs` | 11 |
| `analytic_temperature_rise.rs` | 11 |
| `analytic_tight_world_aabb.rs` | 11 |
| `analytic_vibration_wall_wiring.rs` | 11 |
| `audit_anisotropic_friction.rs` | 11 |
| `audit_creep_longterm.rs` | 11 |
| `audit_cylinder.rs` | 11 |
| `audit_debug_render.rs` | 11 |
| `audit_laminate.rs` | 11 |
| `audit_piezoelectric.rs` | 11 |
| `audit_pressure.rs` | 11 |
| `audit_rolling_contact.rs` | 11 |
| `audit_wedge.rs` | 11 |
| `indep_linear_solver_krylov.rs` | 11 |
| `indep_participant_order.rs` | 11 |
| `analytic_atmosphere_isa1976.rs` | 10 |
| `analytic_bridging_wiring.rs` | 10 |
| `analytic_cfd_flow_bc.rs` | 10 |
| `analytic_cubic_hyperelastic.rs` | 10 |
| `analytic_db_bridge_wiring.rs` | 10 |
| `analytic_fsi_advanced_wiring.rs` | 10 |
| `analytic_linear_elastic_fem_wiring_additional.rs` | 10 |
| `analytic_molecular_dynamics.rs` | 10 |
| `analytic_particle_landing.rs` | 10 |
| `analytic_physics2d_contact_normals.rs` | 10 |
| `analytic_physics2d_wiring.rs` | 10 |
| `analytic_sdf_sph_wiring.rs` | 10 |
| `analytic_snapshot_material_table.rs` | 10 |
| `audit_buckling.rs` | 10 |
| `audit_compressible.rs` | 10 |
| `audit_contact_cache.rs` | 10 |
| `audit_db_bridge.rs` | 10 |
| `audit_smoke_fire.rs` | 10 |
| `audit_transient_thermal.rs` | 10 |
| `coupling_channel_inventory.rs` | 10 |
| `fix128_vs_f64_coupling_hypotheses.rs` | 10 |
| `indep_aero_atmosphere.rs` | 10 |
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
| `analytic_rans_wall_function.rs` | 9 |
| `analytic_structural_creep_per_material.rs` | 9 |
| `analytic_thermoplastic_softening.rs` | 9 |
| `audit_c_trimesh.rs` | 9 |
| `audit_fatigue.rs` | 9 |
| `audit_heatmap.rs` | 9 |
| `audit_non_newtonian.rs` | 9 |
| `audit_replay.rs` | 9 |
| `audit_sdf_wind_field.rs` | 9 |
| `audit_vibration_wall.rs` | 9 |
| `determinism_golden.rs` | 9 |
| `indep_crowd_social_force.rs` | 9 |
| `vehicle_dynamics_degenerate.rs` | 9 |
| `analytic_added_mass_coupling.rs` | 8 |
| `analytic_broadphase.rs` | 8 |
| `analytic_contact_event_normals.rs` | 8 |
| `analytic_cubic_fem.rs` | 8 |
| `analytic_fillet_stress_wiring.rs` | 8 |
| `analytic_fluid_block_wiring.rs` | 8 |
| `analytic_ik_physics_bridge_wiring.rs` | 8 |
| `analytic_math_util_wiring.rs` | 8 |
| `analytic_metric_broadphase.rs` | 8 |
| `analytic_netcode_prediction.rs` | 8 |
| `analytic_pressure_distributed.rs` | 8 |
| `analytic_print_orientation_wiring.rs` | 8 |
| `analytic_rotor.rs` | 8 |
| `analytic_step_multigrid.rs` | 8 |
| `analytic_tgs_joints_in_substep.rs` | 8 |
| `analytic_thermal_stress_wiring.rs` | 8 |
| `analytic_wall_model.rs` | 8 |
| `audit_acoustic_wave.rs` | 8 |
| `audit_c_cfd_solver.rs` | 8 |
| `audit_error.rs` | 8 |
| `audit_filament_db.rs` | 8 |
| `audit_filter.rs` | 8 |
| `audit_kinematic_loop.rs` | 8 |
| `audit_prestressed.rs` | 8 |
| `indep_aero_lift_drag.rs` | 8 |
| `indep_euler_fv_riemann.rs` | 8 |
| `indep_molecular_dynamics.rs` | 8 |
| `indep_orbit_nbody.rs` | 8 |
| `mat3_inverse_range.rs` | 8 |
| `mesh_conformity.rs` | 8 |
| `analytic_ccd_adaptive_substeps_wiring.rs` | 7 |
| `analytic_coupled_wiring.rs` | 7 |
| `analytic_electromagnetic_wiring.rs` | 7 |
| `analytic_hyperelastic_degenerate.rs` | 7 |
| `analytic_multigrid.rs` | 7 |
| `analytic_non_newtonian_wiring.rs` | 7 |
| `analytic_physics2d_joints.rs` | 7 |
| `analytic_piezoelectric_wiring.rs` | 7 |
| `analytic_rng_wiring.rs` | 7 |
| `analytic_sdf_dynamic_collider_world.rs` | 7 |
| `analytic_sketch_generic.rs` | 7 |
| `analytic_soft_body_cut_wiring.rs` | 7 |
| `audit_damping_rayleigh.rs` | 7 |
| `audit_fillet_stress.rs` | 7 |
| `audit_multi_world.rs` | 7 |
| `elastoplastic_increment_api.rs` | 7 |
| `fsi_advanced_sub_iteration.rs` | 7 |
| `indep_aero_rotor.rs` | 7 |
| `mesh_quality.rs` | 7 |
| `wm07_reset_and_rollback_contract.rs` | 7 |
| `analytic_analytics_bridge_wiring.rs` | 6 |
| `analytic_bvh_leaf_aabb_wiring.rs` | 6 |
| `analytic_geometry_helpers.rs` | 6 |
| `analytic_mass_properties_exactness.rs` | 6 |
| `analytic_metric_wiring.rs` | 6 |
| `analytic_replay_in_memory.rs` | 6 |
| `analytic_sdf_distance_query.rs` | 6 |
| `analytic_support_volume_wiring.rs` | 6 |
| `analytic_thermoelastic.rs` | 6 |
| `analytic_thermoelastic_channel.rs` | 6 |
| `analytic_warp_risk_wiring.rs` | 6 |
| `analytic_world_query_touching.rs` | 6 |
| `audit_contact_modifier_once.rs` | 6 |
| `audit_layer_adhesion.rs` | 6 |
| `audit_profiling.rs` | 6 |
| `cloth_fluid_sub_iteration.rs` | 6 |
| `hanging_node_effect.rs` | 6 |
| `indep_fdtd_materials.rs` | 6 |
| `p3_quadrature_fix128.rs` | 6 |
| `wm01_overflow_is_not_silent.rs` | 6 |
| `wm08_checksum_coverage.rs` | 6 |
| `analytic_cfd_wall_bc.rs` | 5 |
| `analytic_cloth_crossings_wiring.rs` | 5 |
| `analytic_contact_static_friction.rs` | 5 |
| `analytic_hyperelastic_mms_order.rs` | 5 |
| `analytic_math_ln.rs` | 5 |
| `analytic_quadratic_mesh_edges_wiring.rs` | 5 |
| `analytic_shaped_sphere_contacts.rs` | 5 |
| `analytic_solver_tgs_dispatch_wiring.rs` | 5 |
| `analytic_world_query_long_paths.rs` | 5 |
| `analytic_world_query_twisted_cells.rs` | 5 |
| `audit_c_buckling.rs` | 5 |
| `audit_c_collider.rs` | 5 |
| `audit_c_force.rs` | 5 |
| `audit_solver_bridge_modifier.rs` | 5 |
| `determinism_golden_contacts.rs` | 5 |
| `engineering_oracles.rs` | 5 |
| `indep_free_rotation_closed_forms.rs` | 5 |
| `mms_linear_elastic.rs` | 5 |
| `reduction_order_independence.rs` | 5 |
| `analytic_adaptive_refinement_high_order.rs` | 4 |
| `analytic_contact_friction_cap.rs` | 4 |
| `analytic_critically_damped_tether.rs` | 4 |
| `analytic_sdf_dynamic_collider_pose.rs` | 4 |
| `analytic_step_default_projection.rs` | 4 |
| `analytic_world_query_boundaries.rs` | 4 |
| `audit_c_joint_extra.rs` | 4 |
| `audit_c_laminate.rs` | 4 |
| `audit_c_linear_elastic_fem.rs` | 4 |
| `audit_c_plane_collider.rs` | 4 |
| `indep_contact_restitution_friction.rs` | 4 |
| `indep_shaped_sphere_rest_heights.rs` | 4 |
| `locking_p1.rs` | 4 |
| `p2_oracle_design.rs` | 4 |
| `p2_quadrature_fix128.rs` | 4 |
| `refinement_conformity.rs` | 4 |
| `spatial_hash_range.rs` | 4 |
| `tgs_stable_cache_keys.rs` | 4 |
| `analytic_boundary_faces.rs` | 3 |
| `analytic_contact_filter_parallel_once.rs` | 3 |
| `analytic_contact_filter_velocity_pass.rs` | 3 |
| `analytic_electromagnetic_world_motion.rs` | 3 |
| `analytic_joint_compliance_xpbd.rs` | 3 |
| `analytic_large_rotation.rs` | 3 |
| `analytic_restitution_phase.rs` | 3 |
| `analytic_shape_with_rotation_wiring.rs` | 3 |
| `audit_bvh_alloc.rs` | 3 |
| `audit_c_coupled_field.rs` | 3 |
| `audit_c_hyperelastic.rs` | 3 |
| `audit_c_laminate_failure.rs` | 3 |
| `audit_c_math.rs` | 3 |
| `audit_c_turbulence.rs` | 3 |
| `audit_gpu_bridge.rs` | 3 |
| `audit_nonunit_quaternion_compound.rs` | 3 |
| `determinism_physics2d_step_digest.rs` | 3 |
| `indep_tgs_pendulum_static_plane.rs` | 3 |
| `parallel_batch_coloring.rs` | 3 |
| `sleep_skip_snapshot.rs` | 3 |
| `wm01_flag_survives_rollback.rs` | 3 |
| `wm08_prev_state_coverage.rs` | 3 |
| `wm08_state_coverage.rs` | 3 |
| `analytic_gjk_separation_distance.rs` | 2 |
| `analytic_rotation_integration.rs` | 2 |
| `analytic_tgs_backend_coverage.rs` | 2 |
| `analytic_tgs_rotation.rs` | 2 |
| `audit_c_fatigue.rs` | 2 |
| `audit_c_joint.rs` | 2 |
| `audit_c_replay.rs` | 2 |
| `audit_neural_alloc.rs` | 2 |
| `mesh_to_fem_stress.rs` | 2 |
| `tgs_joints_static_colliders.rs` | 2 |
| `wm07_rollback_event_parity.rs` | 2 |
| `analytic_fem_convergence.rs` | 1 |
| `analytic_wind_zone_terminal_velocity.rs` | 1 |
| `analytic_world_query_near_contact.rs` | 1 |
| `audit_c_creep_longterm.rs` | 1 |
| `audit_c_plastic.rs` | 1 |

---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

