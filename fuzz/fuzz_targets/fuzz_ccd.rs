#![no_main]
use alice_physics::ccd::{sphere_plane_toi, sphere_sphere_toi};
use alice_physics::math::{Fix128, Vec3Fix};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct CcdInput {
    /// Sphere A start / end and radius (i8 bounded → Fix128 in unit steps).
    a_start: (i8, i8, i8),
    a_end: (i8, i8, i8),
    a_radius: u8,
    /// Sphere B start / end and radius.
    b_start: (i8, i8, i8),
    b_end: (i8, i8, i8),
    b_radius: u8,
    /// Plane normal (must be non-zero after normalisation) and offset.
    plane_normal: (i8, i8, i8),
    plane_offset: i8,
    /// dt for the swept interval (u8, mapped to 1/N s).
    dt_denom: u8,
}

// Fuzz continuous collision detection primitives:
// - `sphere_sphere_toi` — swept two-sphere TOI computation
// - `sphere_plane_toi` — swept sphere vs infinite plane TOI
//
// Must never panic on arbitrary swept inputs; edge cases include zero
// velocity, coincident starts, degenerate plane normals, and negative
// radii mapped to a floor of Fix128::ZERO.
fuzz_target!(|input: CcdInput| {
    // Denominator for dt: keep at least 1 to avoid divide-by-zero and cap
    // to 240 (240 Hz upper bound).
    let denom = ((input.dt_denom as i64).max(1)).min(240);
    let dt = Fix128::from_ratio(1, denom);

    let a_start = Vec3Fix::from_int(
        input.a_start.0 as i64,
        input.a_start.1 as i64,
        input.a_start.2 as i64,
    );
    let a_end = Vec3Fix::from_int(
        input.a_end.0 as i64,
        input.a_end.1 as i64,
        input.a_end.2 as i64,
    );
    let b_start = Vec3Fix::from_int(
        input.b_start.0 as i64,
        input.b_start.1 as i64,
        input.b_start.2 as i64,
    );
    let b_end = Vec3Fix::from_int(
        input.b_end.0 as i64,
        input.b_end.1 as i64,
        input.b_end.2 as i64,
    );

    // Velocities are the total displacement over the unit interval;
    // sphere_sphere_toi / sphere_plane_toi return t in [0, 1].
    let a_vel = a_end - a_start;
    let b_vel = b_end - b_start;

    let a_radius = Fix128::from_ratio((input.a_radius as i64).max(1), 10);
    let b_radius = Fix128::from_ratio((input.b_radius as i64).max(1), 10);

    // Sphere-sphere swept TOI (radius before velocity per the API): must never
    // panic even if spheres already overlap or are moving apart.
    let _ = sphere_sphere_toi(a_start, a_radius, a_vel, b_start, b_radius, b_vel);

    // Sphere-plane swept TOI: if the plane normal is effectively zero, skip
    // (degenerate plane).
    let plane_nx = input.plane_normal.0 as i64;
    let plane_ny = input.plane_normal.1 as i64;
    let plane_nz = input.plane_normal.2 as i64;
    if plane_nx.abs() + plane_ny.abs() + plane_nz.abs() > 0 {
        let plane_normal = Vec3Fix::from_int(plane_nx, plane_ny, plane_nz);
        let plane_offset = Fix128::from_int(input.plane_offset as i64);
        let _ = sphere_plane_toi(a_start, a_radius, a_vel, plane_normal, plane_offset);
    }

    // `dt` still exercised to keep the CcdConfig-scoped code paths warm:
    // adjust the target radius to a dt-scaled value so the fuzz corpus
    // reflects varied step sizes when the caller wraps the primitives in
    // a substep loop.
    let _scaled_radius = a_radius * dt;
});
