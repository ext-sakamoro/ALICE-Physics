//! What sub-iterating the existing cloth-fluid coupling can and cannot show.
//!
//! `cloth_fluid` applies one one-directional force function and then the other,
//! with no sub-iteration. The intent of this file was to drive that as a
//! fixed-point iteration and measure its contraction ratio.
//!
//! ⚠️ **Measured: within a single step there is nothing to iterate.**
//! `apply_cloth_boundary_to_fluid` takes cloth *positions* and *normals* and no
//! cloth velocity, so the fluid's update does not depend on the cloth iterate.
//! The fixed-point map is constant after the first sweep, and the cloth result
//! is bit-identical from sweep one onwards.
//!
//! The loop does close — but through **positions**, over successive time steps,
//! once an integrator has moved the cloth. It does not close **within** a step,
//! which is where a sub-iteration would live.
//!
//! # What follows from that
//!
//! - The added-mass instability **cannot be exhibited here**, and that is not
//!   evidence that the splitting is well conditioned. It is evidence that the
//!   feedback path a sub-iteration would traverse is absent.
//! - Sub-iterating this coupling as it stands would converge in one sweep by
//!   construction — a monitor wrapped around it would report contraction while
//!   measuring nothing. ⚠️ That is a scene that cannot support the claim, not a
//!   passing result.
//! - The monitoring machinery is nonetheless in place and exercised here for
//!   the parts that *are* meaningful: the reporting variants leave the state
//!   untouched, and the residual plumbing works.
//!
//! Closing the loop (having the cloth-to-fluid direction read cloth velocity,
//! or integrating positions inside the sweep) is a change to the coupling's
//! formulation and is out of scope for this file; it is filed in the backlog.
//!
//! # Where the closed-form claims live
//!
//! In `analytic_added_mass_coupling.rs`, on the one-degree-of-freedom piston,
//! where an exact answer exists. This coupling is particle-based and has no
//! closed form, so no claim about the correctness of a converged state is made
//! here.

#![cfg(feature = "std")]

use alice_physics::cloth_fluid::{
    apply_cloth_boundary_to_fluid, apply_cloth_boundary_to_fluid_with_residual,
    apply_fluid_forces_to_cloth, apply_fluid_forces_to_cloth_with_residual, ClothFluidCoupling,
};
use alice_physics::math::{Fix128, Vec3Fix};

/// A submerged patch with every force term live.
struct Scene {
    coupling: ClothFluidCoupling,
    cloth_positions: Vec<Vec3Fix>,
    cloth_velocities: Vec<Vec3Fix>,
    cloth_normals: Vec<Vec3Fix>,
    fluid_positions: Vec<Vec3Fix>,
    fluid_velocities: Vec<Vec3Fix>,
    fluid_density: Fix128,
    repulsion: Fix128,
    dt: Fix128,
}

impl Scene {
    fn submerged() -> Self {
        let step = Fix128::from_ratio(1, 8);
        let cloth_positions = (0..4i64)
            .map(|i| Vec3Fix::new(Fix128::from_int(i) * step, Fix128::ZERO, Fix128::ZERO))
            .collect();
        let fluid_positions = (0..4i64)
            .map(|i| {
                Vec3Fix::new(
                    Fix128::from_int(i) * step,
                    Fix128::from_ratio(1, 16),
                    Fix128::ZERO,
                )
            })
            .collect();
        Self {
            coupling: ClothFluidCoupling::default(),
            cloth_positions,
            cloth_velocities: vec![Vec3Fix::from_int(1, 0, 0); 4],
            cloth_normals: vec![Vec3Fix::UNIT_Y; 4],
            fluid_positions,
            fluid_velocities: vec![Vec3Fix::ZERO; 4],
            fluid_density: Fix128::from_ratio(1, 4),
            repulsion: Fix128::from_ratio(1, 64),
            dt: Fix128::from_ratio(1, 60),
        }
    }

    /// One fluid-to-cloth pass from the step's initial cloth state.
    fn cloth_pass(&self, fluid_velocities: &[Vec3Fix]) -> (Vec<Vec3Fix>, Fix128) {
        let mut cloth = self.cloth_velocities.clone();
        let force = apply_fluid_forces_to_cloth_with_residual(
            &self.coupling,
            &self.cloth_positions,
            &mut cloth,
            &self.fluid_positions,
            fluid_velocities,
            self.fluid_density,
            self.dt,
        );
        (cloth, force)
    }

    /// One cloth-to-fluid pass from the step's initial fluid state.
    fn fluid_pass(&self) -> (Vec<Vec3Fix>, Fix128) {
        let mut fluid = self.fluid_velocities.clone();
        let correction = apply_cloth_boundary_to_fluid_with_residual(
            &self.cloth_positions,
            &self.cloth_normals,
            &self.fluid_positions,
            &mut fluid,
            self.repulsion,
        );
        (fluid, correction)
    }
}

#[test]
fn the_cloth_to_fluid_direction_ignores_the_cloth_velocity() {
    // The structural fact, asserted directly rather than inferred from the
    // signature: two wildly different cloth velocity fields must produce the
    // same fluid result, because the velocity never reaches that computation.
    let scene = Scene::submerged();
    let (baseline, correction) = scene.fluid_pass();

    let mut moving_fast = Scene::submerged();
    moving_fast.cloth_velocities = vec![Vec3Fix::from_int(1_000, -500, 250); 4];
    let (with_fast_cloth, _) = moving_fast.fluid_pass();

    assert_eq!(
        baseline, with_fast_cloth,
        "the fluid update must be identical, which is exactly why a \
         sub-iteration has nothing to converge within a step"
    );
    assert!(
        correction > Fix128::ZERO,
        "the repulsion must actually fire, or this compares two no-ops"
    );
    assert_ne!(
        baseline, scene.fluid_velocities,
        "the fluid state must have moved, or this compares two no-ops"
    );
}

#[test]
fn the_cloth_iterate_is_bit_identical_from_the_first_sweep() {
    // The consequence: iterating cannot change the answer. A monitor wrapped
    // around this would report convergence while measuring nothing, so the
    // absence of a divergence verdict here says nothing about the splitting.
    let scene = Scene::submerged();
    let (fluid_after_reaction, _) = scene.fluid_pass();

    let (first, first_force) = scene.cloth_pass(&fluid_after_reaction);
    for sweep in 2..=4u32 {
        let (again, force) = scene.cloth_pass(&fluid_after_reaction);
        assert_eq!(
            again, first,
            "sweep {sweep} differs from sweep 1, so the loop does close after all \
             and this file's premise needs revisiting"
        );
        assert_eq!(force, first_force);
    }

    // The pass must do something, or the bit-identity above is vacuous.
    assert_ne!(first, scene.cloth_velocities);
    assert!(first_force > Fix128::ZERO);
}

#[test]
fn the_reporting_variants_leave_the_state_a_bare_call_would_leave() {
    // The monitoring must not perturb the physics, in either direction.
    let scene = Scene::submerged();

    let mut reported_cloth = scene.cloth_velocities.clone();
    let force = apply_fluid_forces_to_cloth_with_residual(
        &scene.coupling,
        &scene.cloth_positions,
        &mut reported_cloth,
        &scene.fluid_positions,
        &scene.fluid_velocities,
        scene.fluid_density,
        scene.dt,
    );
    let mut bare_cloth = scene.cloth_velocities.clone();
    apply_fluid_forces_to_cloth(
        &scene.coupling,
        &scene.cloth_positions,
        &mut bare_cloth,
        &scene.fluid_positions,
        &scene.fluid_velocities,
        scene.fluid_density,
        scene.dt,
    );
    assert_eq!(reported_cloth, bare_cloth);
    assert_ne!(reported_cloth, scene.cloth_velocities);
    assert!(force > Fix128::ZERO);

    let mut reported_fluid = scene.fluid_velocities.clone();
    let correction = apply_cloth_boundary_to_fluid_with_residual(
        &scene.cloth_positions,
        &scene.cloth_normals,
        &scene.fluid_positions,
        &mut reported_fluid,
        scene.repulsion,
    );
    let mut bare_fluid = scene.fluid_velocities.clone();
    apply_cloth_boundary_to_fluid(
        &scene.cloth_positions,
        &scene.cloth_normals,
        &scene.fluid_positions,
        &mut bare_fluid,
        scene.repulsion,
    );
    assert_eq!(reported_fluid, bare_fluid);
    assert_ne!(reported_fluid, scene.fluid_velocities);
    assert!(correction > Fix128::ZERO);
}

/// A single cloth particle with a single fluid neighbour, so every intermediate
/// quantity can be written out independently in the assertions below.
fn one_on_one() -> (ClothFluidCoupling, Vec3Fix, Vec3Fix, Fix128, Fix128) {
    let coupling = ClothFluidCoupling::default();
    let cloth_velocity = Vec3Fix::from_int(3, -1, 2);
    let fluid_velocity = Vec3Fix::from_int(-1, 2, 0);
    let fluid_density = Fix128::from_ratio(3, 4);
    let dt = Fix128::from_ratio(1, 60);
    (coupling, cloth_velocity, fluid_velocity, fluid_density, dt)
}

/// The three force terms, written from the documented formulas rather than
/// taken from the implementation.
fn expected_terms(
    coupling: &ClothFluidCoupling,
    cloth_velocity: Vec3Fix,
    fluid_velocity: Vec3Fix,
    fluid_density: Fix128,
) -> (Vec3Fix, Vec3Fix, Vec3Fix) {
    // One neighbour, so the neighbour count is one and the average fluid
    // velocity is that neighbour's velocity.
    let density_factor = Fix128::ONE * fluid_density;
    let drag = (cloth_velocity - fluid_velocity) * (coupling.drag_coefficient * density_factor);
    let buoyancy = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO)
        * (coupling.buoyancy_factor * Fix128::ONE);
    let tension = fluid_velocity * coupling.surface_tension;
    (drag, buoyancy, tension)
}

/// A small deterministic set of one-on-one scenes. No single scene can pin all
/// three force terms through an L-infinity norm (a term acting only on `y`
/// cannot move the maximum when `x` dominates), so the assertions below require
/// the *set* to distinguish each term, and say so when it does not.
fn scene_set() -> Vec<(Vec3Fix, Vec3Fix, Fix128)> {
    vec![
        (
            Vec3Fix::from_int(3, -1, 2),
            Vec3Fix::from_int(-1, 2, 0),
            Fix128::from_ratio(3, 4),
        ),
        // y dominates, so the buoyancy term reaches the maximum.
        (
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 32), Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 32), Fix128::ZERO),
            Fix128::from_ratio(1, 64),
        ),
        // the cloth matches the fluid, so drag vanishes and tension shows.
        (
            Vec3Fix::from_int(0, 0, 5),
            Vec3Fix::from_int(0, 0, 5),
            Fix128::from_ratio(1, 3),
        ),
        // Non-dyadic components, so every product truncates.
        (
            Vec3Fix::new(
                Fix128::from_ratio(1, 7),
                Fix128::from_ratio(-2, 5),
                Fix128::from_ratio(1, 11),
            ),
            Vec3Fix::new(
                Fix128::from_ratio(-3, 7),
                Fix128::from_ratio(1, 5),
                Fix128::from_ratio(2, 11),
            ),
            Fix128::from_ratio(5, 7),
        ),
    ]
}

fn inf_norm(v: Vec3Fix) -> Fix128 {
    [v.x, v.y, v.z]
        .into_iter()
        .map(Fix128::abs)
        .fold(Fix128::ZERO, |worst, m| if m > worst { m } else { worst })
}

#[test]
fn the_reported_force_is_the_sum_of_all_three_terms() {
    // Pins the *value*, not merely that it is non-zero.
    let (coupling, _, _, _, dt) = one_on_one();
    let mut detects = [false; 3];

    for (cloth_velocity, fluid_velocity, fluid_density) in scene_set() {
        let (drag, buoyancy, tension) =
            expected_terms(&coupling, cloth_velocity, fluid_velocity, fluid_density);
        let expected = inf_norm(buoyancy + tension - drag);

        let mut state = vec![cloth_velocity];
        let reported = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &[Vec3Fix::ZERO],
            &mut state,
            &[Vec3Fix::ZERO],
            &[fluid_velocity],
            fluid_density,
            dt,
        );
        assert_eq!(
            reported, expected,
            "scene cloth={cloth_velocity:?} fluid={fluid_velocity:?}"
        );

        // Record which terms this scene could detect the loss of.
        for (index, without) in [buoyancy + tension, tension - drag, buoyancy - drag]
            .into_iter()
            .enumerate()
        {
            if inf_norm(without) != expected {
                detects[index] = true;
            }
        }
    }

    for (name, detected) in ["drag", "buoyancy", "tension"].iter().zip(detects) {
        assert!(
            detected,
            "no scene in the set detects the loss of the {name} term, so this              test cannot pin it"
        );
    }
}

#[test]
fn each_force_term_keeps_its_own_multiplication_by_the_step() {
    // The doc claims the applied update is byte-for-byte unchanged, and the
    // reason is that `Fix128` multiplication truncates, so three separate
    // products do not equal one product of the sum. Asserted here rather than
    // left as prose. Not every scene distinguishes the two orders (the
    // truncations can coincide), so the set must contain one that does.
    let (coupling, _, _, _, dt) = one_on_one();
    let mut distinguishing_scenes = 0u32;

    for (cloth_velocity, fluid_velocity, fluid_density) in scene_set() {
        let (_, buoyancy, tension) =
            expected_terms(&coupling, cloth_velocity, fluid_velocity, fluid_density);
        // Drag is one implicit Euler step on the relative velocity, so its
        // applied fraction is x / (1 + x) with x = C_d rho N dt (N = 1 here);
        // buoyancy and tension keep one product each.
        let x = coupling.drag_coefficient * fluid_density * dt;
        let relative = cloth_velocity - fluid_velocity;
        let drag_delta = relative * (x / (Fix128::ONE + x));
        let term_by_term = cloth_velocity - drag_delta + buoyancy * dt + tension * dt;
        let folded = cloth_velocity + (buoyancy + tension) * dt - drag_delta;
        if term_by_term != folded {
            distinguishing_scenes += 1;
        }

        let mut state = vec![cloth_velocity];
        let _ = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &[Vec3Fix::ZERO],
            &mut state,
            &[Vec3Fix::ZERO],
            &[fluid_velocity],
            fluid_density,
            dt,
        );
        assert_eq!(
            state[0], term_by_term,
            "the applied update must keep one multiplication per force term              (scene cloth={cloth_velocity:?})"
        );
    }

    assert!(
        distinguishing_scenes > 0,
        "no scene distinguishes the two multiplication orders, so this test          cannot pin which one is used"
    );
}

#[test]
fn the_reported_force_is_independent_of_the_step() {
    // Why the fluid-to-cloth direction reports a force rather than a velocity
    // change: the force must not move when only the step does.
    let mut coarse = Scene::submerged();
    coarse.dt = Fix128::from_ratio(1, 60);
    let (coarse_state, coarse_force) = coarse.cloth_pass(&coarse.fluid_velocities);

    let mut fine = Scene::submerged();
    fine.dt = Fix128::from_ratio(1, 240);
    let (fine_state, fine_force) = fine.cloth_pass(&fine.fluid_velocities);

    assert_eq!(coarse_force, fine_force);

    // And the applied change must shrink, or the assertion above is empty.
    let coarse_change = (coarse_state[0] - coarse.cloth_velocities[0]).x.abs();
    let fine_change = (fine_state[0] - fine.cloth_velocities[0]).x.abs();
    assert!(
        fine_change < coarse_change,
        "quartering the step must shrink what is applied: \
         {coarse_change:?} -> {fine_change:?}"
    );
}
