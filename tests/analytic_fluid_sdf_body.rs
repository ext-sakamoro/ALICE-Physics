//! PBF fluid coupled to static solids through SDF density maps (`Fluid::with_sdf_body`).
//!
//! Oracles (closed forms, s = 0.1 m, h = 2 s, a cube of 4 spacings in a tank 1.6 m wide, i.e.
//! 3 h of fluid beside the cube):
//! - cube lying on the floor: no fluid under it, so the vertical force on it is the weight of
//!   the column above its top face, `-rho_0 g (y_s - y_top) A` (down), with the free surface
//!   `y_s = (fluid volume + cube volume) / W^2`;
//! - cube 2 h above the floor: Archimedes, `+rho_0 g V`;
//! - Newton 3: the forces on the solids sum to minus the solids' push on the fluid, which the
//!   fluid accumulates separately ([`Fluid::boundary_force`]);
//! - determinism: two builds of the same map are bit-equal.
//!
//! The 1 h gap (two particle layers under the cube) is an open limitation, see the MODEL of
//! `alice_physics_core::fluid_boundary_map`.

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::fluid_boundary_map::BoundaryMap;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};

const S: f64 = 0.1;
const W: f64 = 1.6;
const CUBE: f64 = 0.4;
const RHO0: f64 = 1000.0;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Exact SDF of an axis-aligned box `[lo, hi]` (negative inside)
fn box_sdf(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let q: Vec<f32> = (0..3).map(|i| (lo[i] - p[i]).max(p[i] - hi[i])).collect();
    let outside = q
        .iter()
        .map(|x| {
            let m = x.max(0.0);
            m * m
        })
        .sum::<f32>()
        .sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

/// Central-difference normal of an f32 SDF closure
fn fd_normal(f: &dyn Fn([f32; 3]) -> f32, p: [f32; 3]) -> (f32, f32, f32) {
    let e = 1e-4_f32;
    let mut g = [0.0_f32; 3];
    for a in 0..3 {
        let mut p1 = p;
        let mut p0 = p;
        p1[a] += e;
        p0[a] -= e;
        g[a] = (f(p1) - f(p0)) / (2.0 * e);
    }
    let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
    (g[0] / n, g[1] / n, g[2] / n)
}

fn collider(f: impl Fn([f32; 3]) -> f32 + Send + Sync + Clone + 'static) -> SdfCollider {
    let g = f.clone();
    let field = ClosureSdf::new(
        move |x, y, z| f([x, y, z]),
        move |x, y, z| fd_normal(&g, [x, y, z]),
    );
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// Open-topped tank: floor at y = 0 and four walls of thickness 2 s around `[0, W]^2`
fn tank(top: f64) -> SdfCollider {
    let (w, l, t) = (W as f32, (2.0 * S) as f32, top as f32);
    collider(move |p| {
        let outer = box_sdf(p, [-l, -l, -l], [w + l, t, w + l]);
        let hole = box_sdf(p, [0.0, 0.0, 0.0], [w, t + 1.0, w]);
        outer.max(-hole)
    })
}

fn cube(bottom: f64) -> SdfCollider {
    let lo = ((W - CUBE) / 2.0) as f32;
    let (b, c) = (bottom as f32, CUBE as f32);
    collider(move |p| box_sdf(p, [lo, b, lo], [lo + c, b + c, lo + c]))
}

struct Scene {
    fluid: Fluid,
    n: usize,
}

/// Fluid on the lattice `(i + 1/2) s` filling `[0, W]^2 x [0, top]` outside the cube
fn scene(bottom: f64, top: f64) -> Scene {
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let config = FluidConfig {
        rest_spacing: s,
        kernel_radius: h,
        rest_density: fx(RHO0),
        particle_mass: FluidConfig::lattice_mass(fx(RHO0), h, s),
        ..FluidConfig::default()
    };
    let nx = (W / S).round() as i64;
    let ny = (top / S).round() as i64;
    let lo = ((W - CUBE) / 2.0 / S).round() as i64;
    let size = (CUBE / S).round() as i64;
    let b0 = (bottom / S).round() as i64;
    let mut pts = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nx {
                let inside = (lo..lo + size).contains(&i)
                    && (b0..b0 + size).contains(&j)
                    && (lo..lo + size).contains(&k);
                if !inside {
                    pts.push(v(
                        (i as f64 + 0.5) * S,
                        (j as f64 + 0.5) * S,
                        (k as f64 + 0.5) * S,
                    ));
                }
            }
        }
    }
    let n = pts.len();
    let r = 2.0 * S + 0.3;
    let fluid = Fluid::new(pts, config)
        .with_sdf_body(
            tank(2.0 * top),
            v(-r, -r, -r),
            v(W + r, top + 2.0 * S, W + r),
        )
        .with_sdf_body(
            cube(bottom),
            v((W - CUBE) / 2.0 - 0.3, bottom - 0.3, (W - CUBE) / 2.0 - 0.3),
            v(
                (W + CUBE) / 2.0 + 0.3,
                bottom + CUBE + 0.3,
                (W + CUBE) / 2.0 + 0.3,
            ),
        );
    Scene { fluid, n }
}

/// The rounding bound on Newton 3 between the fluid and the solids, per component, in N
/// (the fluid MODEL "Newton 3"): each particle correction is rounded twice (by the particle
/// volume and by the relaxation, round to nearest), at most one ulp `2^-64` m in all, for
/// every particle on every constraint iteration of the last substep; the fluid-side total and
/// each solid's share are then multiplied by `m / dt^2` once each (half an ulp of force
/// apiece). The pair terms cancel exactly, so nothing else separates the two sides.
fn newton_bound(f: &Fluid, frame_dt: f64) -> f64 {
    let ulp = 2.0_f64.powi(-64);
    let dt = frame_dt / f.config.substeps as f64;
    let force_per_correction = f.config.particle_mass.to_f64() / (dt * dt);
    let count = (f.particle_count() * f.config.iterations) as f64;
    (count * force_per_correction + f.body_count() as f64 + 1.0) * ulp
}

/// The largest component of `v`, in N
fn largest(v: Vec3Fix) -> f64 {
    v.x.to_f64()
        .abs()
        .max(v.y.to_f64().abs())
        .max(v.z.to_f64().abs())
}

/// The frames of a run that is checked every time: the force is the mean over its last 60.
/// The lattice the fluid starts on is close to hydrostatic, and the mean wanders by 2 to 4 %
/// whatever the length (measured 2026-11-19, floor cube / cube 2 h up: 0.983 / 1.018 at 120
/// frames, 1.013 / 0.996 at 240, 1.011 / 1.037 at 480), so a longer run is not a closer one;
/// the 900-frame runs are the `runtime:` tests below.
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

/// The cube's vertical force by face over the same window as [`settle`]: top, bottom and
/// side faces (the particle above, below or beside the cube), and the bottom face in four
/// x-bins, from [`Fluid::body_particle_forces`]
struct Faces {
    top: f64,
    bottom: f64,
    side: f64,
    bottom_bins: [f64; 4],
}

fn settle_with_faces(sc: &mut Scene, bottom: f64, default_frames: usize) -> (f64, f64, Faces) {
    let dt = Fix128::from_ratio(1, 60);
    let total = frames(default_frames);
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
/// top face down by `rho_0 g (H - y_b - CUBE) A`, the side faces carry no vertical load; the
/// bottom face in four bins, each `rho_0 g (H - y_b) A / 4`. The faces are stated rows at the
/// values measured with C >= 0 clamped and inelastic contact (2026-11-18, 900 frames, mean of
/// the last 60): top 0.791, bottom 0.798 of their closed forms, sides +121.8 N, bottom bins
/// 0.639 / 1.107 / 0.853 / 0.593; bounds 0.01 of the face's closed form from them, and they
/// may only tighten (the structural row of the fluid MODEL: the face loads are not hydrostatic
/// at the edges of a solid).
#[test]
fn the_cube_two_kernel_radii_above_the_floor_feels_archimedes() {
    cube_two_kernel_radii_up(FRAMES);
}

#[test]
#[ignore = "runtime: about 20 min on the slowest test runner, measured 2026-11-18; the 120-frame test above runs every time; run by the weekly job"]
fn the_cube_two_kernel_radii_above_the_floor_feels_archimedes_over_900_frames() {
    cube_two_kernel_radii_up(900);
}

fn cube_two_kernel_radii_up(default_frames: usize) {
    let bottom = 0.4;
    let mut sc = scene(bottom, bottom + 0.8);
    let (fy, newton, faces) = settle_with_faces(&mut sc, bottom, default_frames);
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
    assert!((ratio - 1.0).abs() < 0.05, "F_y / (rho g V) = {ratio}");
    assert!(
        -faces.top / top_closed >= 0.781,
        "top face {} of closed (measured 0.791)",
        -faces.top / top_closed
    );
    assert!(
        faces.bottom / bottom_closed >= 0.788,
        "bottom face {} of closed (measured 0.798)",
        faces.bottom / bottom_closed
    );
    assert!(
        faces.side.abs() <= 121.8 + 12.8,
        "side faces carry {} N vertically (measured 121.8)",
        faces.side
    );
    for (b, (got, measured)) in bins.iter().zip([0.639, 1.107, 0.853, 0.593]).enumerate() {
        assert!(
            (got - 1.0).abs() <= (measured - 1.0_f64).abs() + 0.01,
            "bottom bin {b}: {got} (measured {measured})"
        );
    }
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
