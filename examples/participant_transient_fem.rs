//! A transient FEM body as a world participant: one tetrahedron with a
//! single free degree of freedom under a suddenly applied load, stepped by
//! average-acceleration Newmark inside the world's substep loop. The response
//! swings between 0 and twice the static deflection.
//!
//! Run: `cargo run --example participant_transient_fem`

use alice_physics_core::dynamic_fem::{
    DynamicsConfig, MassLumping, TransientFemParticipant, TransientSolver,
    TRANSIENT_FEM_OBS_ITERATIONS, TRANSIENT_FEM_OBS_STEPS, TRANSIENT_FEM_OBS_TIME,
};
use alice_physics_core::linear_elastic_fem::{
    Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.extend([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    let mut boundary = BoundaryConditions::new();
    for v in 0..3u32 {
        boundary.fix(v);
    }
    boundary.prescribe(3, Axis::Y, Fix128::ZERO);
    boundary.prescribe(3, Axis::Z, Fix128::ZERO);
    boundary.add_load(3, Axis::X, Fix128::ONE);
    let dt = Fix128::from_raw(0, 1u64 << (64 - 23));
    let solver = TransientSolver::new(
        &mesh,
        &ElasticMaterial::new(Fix128::from_int(3500), Fix128::ZERO)
            .map_err(|e| format!("material: {e:?}"))?,
        &boundary,
        &DynamicsConfig::try_new(Fix128::from_raw(0, 1u64 << 34), dt, MassLumping::Lumped)
            .map_err(|e| format!("dynamics: {e:?}"))?,
        &SolverConfig::try_new(20_000, Fix128::from_f64(1e-12))
            .map_err(|e| format!("solver: {e:?}"))?,
    )
    .map_err(|e| format!("transient solver: {e:?}"))?;
    let resident = TransientFemParticipant::new(solver);
    println!(
        "start: {} steps at {:.3e} s, u_x = {:.3e}",
        resident.steps(),
        resident.time().to_f64(),
        resident.solver().displacements()[9].to_f64()
    );

    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 2,
        ..PhysicsConfig::default()
    });
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("register: {e:?}"))?;
    let mut peak = 0.0_f64;
    for _ in 0..40 {
        world
            .try_step(Fix128::from_int(4) * dt)
            .map_err(|e| format!("step: {e:?}"))?;
        if let Some(state) = world.participant_state(index) {
            let at = 4 + 8 + 16 + 8 + 8 + 9 * 16;
            let hi = i64::from_le_bytes(state[at..at + 8].try_into()?);
            let lo = u64::from_le_bytes(state[at + 8..at + 16].try_into()?);
            peak = peak.max(Fix128 { hi, lo }.to_f64());
        }
    }
    println!(
        "peak u_x over 160 steps: {peak:.6e} (static deflection 1/K = {:.6e})",
        9.0 / 875.0
    );
    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            let name = match channel {
                TRANSIENT_FEM_OBS_TIME => "time (s)",
                TRANSIENT_FEM_OBS_STEPS => "Newmark steps",
                TRANSIENT_FEM_OBS_ITERATIONS => "CG iterations, last step",
                _ => continue,
            };
            println!("{name}: {:.6e}", value.to_f64());
        }
    }
    Ok(())
}
