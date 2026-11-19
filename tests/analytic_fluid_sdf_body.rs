//! PBF fluid coupled to static solids through SDF density maps (`Fluid::with_sdf_body`).
//!
//! Oracles (closed forms, s = 0.1 m, h = 2 s, a cube of 4 spacings in a tank 1.6 m wide, i.e.
//! 3 h of fluid beside the cube):
//! - cube lying on the floor: no fluid under it, so the vertical force on it is the weight of
//!   the column above its top face, `-rho_0 g (y_s - y_top) A` (down), with the free surface
//!   `y_s = (fluid volume + cube volume) / W^2`;
//! - cube 2 h above the floor: Archimedes, `+rho_0 g V`;
//! - Newton 3: the forces on the solids sum to minus the solids' push on the fluid, which the
//!   fluid accumulates separately (`Fluid::boundary_force`);
//! - determinism: two builds of the same map are bit-equal.
//!
//! The 1 h gap (two particle layers under the cube) is an open limitation, see the MODEL of
//! `alice_physics_core::fluid_boundary_map`.

use alice_physics_core::fluid::FluidConfig;
use alice_physics_core::fluid_boundary_map::BoundaryMap;
use alice_physics_core::math::{Fix128, Vec3Fix};
use sha2::{Digest, Sha256};

#[path = "support/fluid_cube_fixture.rs"]
mod fixture;

use fixture::{collider, cube, fx, largest, newton_bound, scene, v, Scene, CUBE, RHO0, S, W};

/// The frames of the floor-cube run that is checked every time: the force is the mean over its
/// last 60. The lattice the fluid starts on is close to hydrostatic, and the mean wanders by 2
/// to 4 % whatever the length (measured 2026-11-19 on main's model, floor cube / cube 2 h up:
/// 0.983 / 1.018 at 120 frames, 1.013 / 0.996 at 240, 1.011 / 1.037 at 480), so a longer run is
/// not a closer one; the 900-frame runs are the `runtime:` tests below. The cube 2 h up is
/// checked every time from a stored state instead (see its test).
const FRAMES: usize = 120;

fn frames(default: usize) -> usize {
    std::env::var("ALICE_FLUID_SDF_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Mean vertical force on the cube over the last 60 frames, and the mean total check
fn settle(sc: &mut Scene, default_frames: usize) -> (f64, f64) {
    let dt = Fix128::from_ratio(1, 60);
    let total = frames(default_frames);
    let mut cube_y = 0.0;
    let mut newton = 0.0_f64;
    for f in 0..total {
        sc.fluid.step(dt);
        let bf = sc.fluid.body_forces();
        assert_eq!(bf.len(), 2, "both solids report a force");
        // Newton 3 on every frame: the solids' forces sum to minus their push on the fluid
        let sum = bf[0] + bf[1] + sc.fluid.boundary_force();
        // Newton 3: the solids' forces sum to minus their push on the fluid, to rounding
        let bound = newton_bound(&sc.fluid, 1.0 / 60.0);
        assert!(
            largest(sum) <= bound,
            "frame {f}: Newton 3 {} N over the rounding bound {bound} N",
            largest(sum)
        );
        newton = newton.max(largest(sum) / bound);
        if f + 60 >= total {
            cube_y += bf[1].y.to_f64() / 60.0;
        }
    }
    (cube_y, newton)
}

fn g() -> f64 {
    FluidConfig::default().gravity.y.to_f64().abs()
}

#[test]
fn the_cube_lying_on_the_floor_carries_only_the_column_above_it() {
    cube_on_the_floor(FRAMES);
}

#[test]
#[ignore = "runtime: about 20 min on the slowest test runner, measured 2026-11-18; the 120-frame test above runs every time; run by the weekly job"]
fn the_cube_lying_on_the_floor_carries_only_the_column_above_it_over_900_frames() {
    cube_on_the_floor(900);
}

fn cube_on_the_floor(default_frames: usize) {
    let top = 0.8;
    let mut sc = scene(0.0, top);
    let (fy, newton) = settle(&mut sc, default_frames);
    let vol = CUBE * CUBE * CUBE;
    let fluid_vol = sc.n as f64 * S * S * S;
    let ys = (fluid_vol + vol) / (W * W);
    let closed = -RHO0 * g() * (ys - CUBE) * CUBE * CUBE;
    let ratio = fy / closed;
    eprintln!(
        "cube on the floor: F_y / closed form = {ratio} ({fy} N vs {closed} N), Newton 3 |sum| / rounding bound {newton}"
    );
    assert!((ratio - 1.0).abs() < 0.05, "F_y / closed form = {ratio}");
}

/// The cube's vertical force by face over the last 60 frames of a run: top, bottom and side
/// faces (the particle above, below or beside the cube), and the bottom face in four x-bins
/// (printed, not checked: they move with where the particles start, 0.46 to 1.22 of their share
/// over four placements, measured 2026-11-19), from `Fluid::body_particle_forces`
struct Faces {
    top: f64,
    bottom: f64,
    side: f64,
    bottom_bins: [f64; 4],
}

fn settle_with_faces(sc: &mut Scene, bottom: f64, total: usize) -> (f64, f64, Faces) {
    let dt = Fix128::from_ratio(1, 60);
    let lo = (W - CUBE) / 2.0;
    let (mut cube_y, mut newton) = (0.0, 0.0_f64);
    let mut faces = Faces {
        top: 0.0,
        bottom: 0.0,
        side: 0.0,
        bottom_bins: [0.0; 4],
    };
    for f in 0..total {
        sc.fluid.step(dt);
        let bf = sc.fluid.body_forces();
        let sum = bf[0] + bf[1] + sc.fluid.boundary_force();
        // Newton 3: the solids' forces sum to minus their push on the fluid, to rounding
        let bound = newton_bound(&sc.fluid, 1.0 / 60.0);
        assert!(
            largest(sum) <= bound,
            "frame {f}: Newton 3 {} N over the rounding bound {bound} N",
            largest(sum)
        );
        newton = newton.max(largest(sum) / bound);
        if f + 60 < total {
            continue;
        }
        cube_y += bf[1].y.to_f64() / 60.0;
        let per = sc.fluid.body_particle_forces();
        for (i, p) in sc.fluid.positions.iter().enumerate() {
            let fy = per[i * 2 + 1].y.to_f64() / 60.0;
            let (x, y) = (p.x.to_f64(), p.y.to_f64());
            if y > bottom + CUBE {
                faces.top += fy;
            } else if y < bottom {
                faces.bottom += fy;
                let bin = (((x - lo) / S).floor().max(0.0) as usize).min(3);
                faces.bottom_bins[bin] += fy;
            } else {
                faces.side += fy;
            }
        }
    }
    (cube_y, newton, faces)
}

/// The cube two kernel radii above the floor carries the Archimedes force, and its faces are
/// read one by one so the total cannot pass through cancellation alone. Closed forms with
/// `H = (N s^3 + CUBE^3) / W^2`: the bottom face is pushed up by `rho_0 g (H - y_b) A`, the
/// top face down by `rho_0 g (H - y_b - CUBE) A`, the side faces carry no vertical load.
///
/// The window is the last 60 frames of the 900-frame run (mean of the force over it). The test
/// that runs every time steps them from the stored state at frame 840
/// (`tests/fixtures/fluid_cube_f840.bin`, written by the example `fluid_cube_state`), so it
/// gives the bits of the 900-frame run; the weekly test steps the 840 frames from the lattice,
/// checks that the state it reaches is the stored one (SHA-256), and steps the same window.
/// A run of 120 frames from the lattice is not used here: on this model its mean is 0.932 of
/// Archimedes (measured 2026-11-19), outside the row, while the settled window is 0.990.
///
/// The faces are stated rows at the values measured in that window (2026-11-19, C >= 0
/// clamped, inelastic contact, grid rebuilt every iteration): top 0.808, bottom 0.789 of their
/// closed forms, sides +140.3 N; bounds 0.01 of the face's closed form from them, and they may
/// only tighten (the structural row of the fluid MODEL: the face loads are not hydrostatic at
/// the edges of a solid). The ratio depends on where the particles start: the initial lattice
/// shifted by a tenth of a spacing gives 0.930 to 1.032 at 900 frames (the open limitation in
/// the MODEL of `alice_physics_core::fluid_boundary_map`); this test holds one placement.
#[test]
fn the_cube_two_kernel_radii_above_the_floor_feels_archimedes() {
    let mut sc = fixture::SETTLED.restore(SETTLED_STATE);
    assert_eq!(sha256(SETTLED_STATE), SETTLED_SHA256, "the stored state");
    cube_two_kernel_radii_up(&mut sc);
}

#[test]
#[ignore = "runtime: about 20 min on the slowest test runner, measured 2026-11-18; the stored-state test above runs every time; run by the weekly job"]
fn the_cube_two_kernel_radii_above_the_floor_feels_archimedes_over_900_frames() {
    let mut sc = fixture::SETTLED.run();
    let state = fixture::SETTLED.encode(&sc.fluid);
    assert_eq!(
        state.len(),
        SETTLED_STATE.len(),
        "length of the state at frame 840"
    );
    assert_eq!(
        sha256(&state),
        SETTLED_SHA256,
        "the state at frame 840 is the stored one"
    );
    cube_two_kernel_radii_up(&mut sc);
}

const SETTLED_STATE: &[u8] = include_bytes!("fixtures/fluid_cube_f840.bin");
const SETTLED_SHA256: &str = "9503530f19fbfc5711dff0194cbc0f874be3df044d120ad940de4325da8fca7c";
/// SHA-256 of the state after the window (frame 900), in the stored form of [`fixture::SETTLED`]
const WINDOW_END_SHA256: &str = "674d7ea0cb451d36953913f4ad09a96da6d61963cca5fc5acc2270f96fb48fc9";

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The checks on the window: the 60 frames after the state in `sc`
fn cube_two_kernel_radii_up(sc: &mut Scene) {
    let bottom = fixture::BOTTOM;
    let (fy, newton, faces) = settle_with_faces(sc, bottom, 60);
    let closed = RHO0 * g() * CUBE * CUBE * CUBE;
    let ratio = fy / closed;
    let h = (sc.n as f64 * S * S * S + CUBE * CUBE * CUBE) / (W * W);
    let bottom_closed = RHO0 * g() * (h - bottom) * CUBE * CUBE;
    let top_closed = RHO0 * g() * (h - bottom - CUBE) * CUBE * CUBE;
    let bins: Vec<f64> = faces
        .bottom_bins
        .iter()
        .map(|b| b / (bottom_closed / 4.0))
        .collect();
    eprintln!(
        "cube 2h up: F_y / (rho g V) = {ratio}, Newton 3 |sum| / rounding bound {newton}; top {} / bottom {} of closed, sides {} N, bottom bins {bins:?}",
        -faces.top / top_closed,
        faces.bottom / bottom_closed,
        faces.side
    );
    // the stepping is Fix128 and the solids' maps are built from f32 SDFs with correctly
    // rounded operations only, so every target continues the stored state to the same bits;
    // the state after the window, in the stored form, is pinned so a target that does not
    // fails here on every push
    assert_eq!(
        sha256(&fixture::SETTLED.encode(&sc.fluid)),
        WINDOW_END_SHA256,
        "the state at frame 900"
    );
    assert!((ratio - 1.0).abs() < 0.05, "F_y / (rho g V) = {ratio}");
    assert!(
        -faces.top / top_closed >= 0.798,
        "top face {} of closed (measured 0.808)",
        -faces.top / top_closed
    );
    assert!(
        faces.bottom / bottom_closed >= 0.779,
        "bottom face {} of closed (measured 0.789)",
        faces.bottom / bottom_closed
    );
    assert!(
        faces.side.abs() <= 140.3 + 12.8,
        "side faces carry {} N vertically (measured 140.3)",
        faces.side
    );
}

#[test]
fn two_builds_of_the_same_map_are_bit_equal() {
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let step = s / Fix128::from_int(4);
    let build = || {
        BoundaryMap::from_sdf(
            &cube(0.2),
            v(0.3, -0.1, 0.3),
            v(1.3, 0.9, 1.3),
            step,
            fx(RHO0),
            h,
        )
    };
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the map build time printed for the record, never asserted"
    )]
    let (a, ms) = {
        let t0 = std::time::Instant::now();
        let a = build();
        (a, t0.elapsed().as_secs_f64() * 1e3)
    };
    let b = build();
    eprintln!(
        "map build: {} nodes, {ms:.1} ms, {} bytes",
        a.node_count(),
        a.node_count() * (16 + 48)
    );
    assert!(a.node_count() > 0);
    assert_eq!(a, b);
}

#[test]
fn a_flat_wall_map_reads_half_the_rest_density_at_the_surface() {
    // oracle: rho_0 * integral of W over the half ball = rho_0 / 2 at the plane, 0 at h out
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let step = s / Fix128::from_int(4);
    let floor = collider(|p| p[1]);
    let map = BoundaryMap::from_sdf(
        &floor,
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.5, 0.5),
        step,
        fx(RHO0),
        h,
    );
    let (at_surface, g_surface) = map.sample(v(0.0, 0.0, 0.0));
    let (outside, _) = map.sample(v(0.0, 2.0 * S + 0.05, 0.0));
    let ratio = at_surface.to_f64() / RHO0;
    eprintln!(
        "flat wall: rho_b(0) / rho_0 = {ratio}, grad = {:?}",
        g_surface.y.to_f64()
    );
    assert!(
        (ratio - 0.5).abs() < 0.03,
        "rho_b at the surface / rho_0 = {ratio}"
    );
    assert!(outside.is_zero(), "zero beyond the kernel radius");
    assert!(
        g_surface.y.to_f64() < 0.0,
        "the gradient points into the solid (-y)"
    );
    assert!(g_surface.x.to_f64().abs() < 1e-6 && g_surface.z.to_f64().abs() < 1e-6);
}

#[test]
fn degenerate_map_inputs_read_zero() {
    let h = fx(0.2);
    let floor = collider(|p| p[1]);
    for (step, kernel, lo, hi) in [
        (fx(0.0), h, v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0)),
        (fx(-0.1), h, v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0)),
        (fx(0.05), fx(0.0), v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0)),
        (fx(0.05), h, v(1.0, 1.0, 1.0), v(-1.0, -1.0, -1.0)),
    ] {
        let map = BoundaryMap::from_sdf(&floor, lo, hi, step, fx(RHO0), kernel);
        assert_eq!(map.node_count(), 0);
        let (r, g) = map.sample(v(0.0, 0.0, 0.0));
        assert!(r.is_zero() && g == Vec3Fix::ZERO);
    }
}
