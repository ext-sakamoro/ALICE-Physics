//! Thin-wall detection driven end to end: `ThinWallConfig::for_nozzle` →
//! `measure_thickness_at` / `analyze_thickness` on a hand-built "stepped
//! slab" (thin on one side, thick on the other), and `sample_surface_points`
//! / `analyze_thickness_grid` on a sphere sampled on a coarse grid.
//!
//! ⚠️ **Why this example exists.** `scripts/wiring_guard.py` reported all
//! seven `pub` items of `alice_physics::thin_wall` — `for_nozzle`,
//! `measure_thickness_at`, `analyze_thickness`, `analyze_thickness_grid`,
//! `sample_surface_points`, `ThinWallReport::has_thin_walls` and
//! `ThinWallReport::thin_fraction` — as having no caller outside the
//! module's own `#[cfg(test)]` block. This example is that caller, and
//! `tests/analytic_thin_wall_wiring.rs` holds the closed-form oracles for
//! the same seven items (plus their degenerate-input behaviour).
//!
//! ```text
//! cargo run --example thin_wall_detection --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example thin_wall_detection --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::math::{Fix128, Vec3Fix};
    use alice_physics::sdf_collider::{ClosureSdf, SdfField};
    use alice_physics::thin_wall::{
        analyze_thickness, analyze_thickness_grid, measure_thickness_at, sample_surface_points,
        ThinWallConfig,
    };

    // ------------------------------------------------------------------
    // Scene 1: "stepped slab" — SDF(x,y,z) = |z| - h(x), h(x)=0.3mm for
    // x<0 (thin side, thickness 0.6mm), h(x)=0.5mm for x>=0 (thick side,
    // thickness 1.0mm). The surface normal used for every query here is
    // always the top face's (0,0,1), so the sphere-march for a given query
    // never moves in x/y and sees a constant-h 1-D slab for its own x.
    // ------------------------------------------------------------------
    let stepped_slab = ClosureSdf::new(
        |x, _y, z| {
            let h: f32 = if x < 0.0 { 0.3 } else { 0.5 };
            z.abs() - h
        },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    );

    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10)); // 0.4mm nozzle
    println!(
        "[thin_wall] for_nozzle(0.4mm) -> min_thickness_mm={:.4} (expect 0.8 = 2x nozzle)",
        cfg.min_thickness_mm.to_f64()
    );

    // 3 thin-side points (thickness 0.6mm) + 3 thick-side points (thickness
    // 1.0mm). Hand-derived thin_fraction = 3/6 = 0.5 exactly.
    let xs: [f32; 6] = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
    let points: Vec<Vec3Fix> = xs
        .iter()
        .map(|&x| {
            let h: f32 = if x < 0.0 { 0.3 } else { 0.5 };
            Vec3Fix::from_f32(x, 0.0, h)
        })
        .collect();

    let report = analyze_thickness(&stepped_slab, &points, &cfg);
    println!(
        "[thin_wall] analyze_thickness(stepped slab): sampled={} thin_regions={} thin_fraction={:.4} has_thin_walls={} (expect 6 / 3 / 0.5 / true)",
        report.sampled_count,
        report.regions.len(),
        report.thin_fraction().to_f64(),
        report.has_thin_walls()
    );
    println!(
        "[thin_wall]   min_thickness_seen={:.4}mm max_thickness_seen={:.4}mm (expect ~0.6 / ~1.0)",
        report.min_thickness_seen.to_f64(),
        report.max_thickness_seen.to_f64()
    );

    let p_thin = Vec3Fix::from_f32(-2.0, 0.0, 0.3);
    let t_thin = measure_thickness_at(&stepped_slab, p_thin, (0.0, 0.0, 1.0), &cfg)
        .expect("thin side must hit the opposite face");
    println!(
        "[thin_wall] measure_thickness_at(thin side, x=-2) = {:.4}mm (expect ~0.6 = 2x0.3)",
        t_thin.to_f64()
    );

    let p_thick = Vec3Fix::from_f32(2.0, 0.0, 0.5);
    let t_thick = measure_thickness_at(&stepped_slab, p_thick, (0.0, 0.0, 1.0), &cfg)
        .expect("thick side must hit the opposite face");
    println!(
        "[thin_wall] measure_thickness_at(thick side, x=2) = {:.4}mm (expect ~1.0 = 2x0.5)",
        t_thick.to_f64()
    );

    // ------------------------------------------------------------------
    // Scene 2: a sphere of radius 5mm, grid-sampled on a coarse 3x3x3 grid
    // (step=6mm over [-6,6]^3) so the only (y,z) grid line through the
    // sphere is (0,0); SDF(x,0,0)=|x|-5 is exactly piecewise-linear there,
    // so the sampler's linear interpolation lands exactly on x=-5,+5.
    // ------------------------------------------------------------------
    let sphere = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 5.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
            (x / len, y / len, z / len)
        },
    );

    let surface_pts = sample_surface_points(
        &sphere,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::from_int(6),
    );
    println!(
        "[thin_wall] sample_surface_points(sphere r=5, grid=6mm): count={} (expect 2)",
        surface_pts.len()
    );
    for p in &surface_pts {
        let (px, py, pz) = p.to_f32();
        let r = (px * px + py * py + pz * pz).sqrt();
        println!(
            "[thin_wall]   point=({px:.3},{py:.3},{pz:.3}) |p|={r:.4} sdf={:.6} (expect |p|~5, sdf~0)",
            sphere.distance(px, py, pz)
        );
    }

    let grid_report = analyze_thickness_grid(
        &sphere,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::from_int(6),
        &cfg,
    );
    println!(
        "[thin_wall] analyze_thickness_grid(sphere r=5): sampled={} thin_regions={} has_thin_walls={} min_thickness={:.4}mm (expect 2 / 0 / false / ~10)",
        grid_report.sampled_count,
        grid_report.regions.len(),
        grid_report.has_thin_walls(),
        grid_report.min_thickness_seen.to_f64()
    );
}
