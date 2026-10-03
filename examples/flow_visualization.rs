//! Production entry point for `alice_physics::flow_viz`: `FlowArrow`,
//! `FlowVizConfig`, `generate_flow_arrows`, and `generate_streamlines`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired` --
//! the module's own `#[cfg(test)]` block exercises them, but tests do not
//! count as production callers for the wiring guard, and nothing in
//! `src/` / `examples/` / `benches/` called any of the four before this
//! file existed. This example is that caller.
//!
//! `tests/analytic_flow_viz_wiring.rs` holds the closed-form / degenerate
//! / boundary-case oracles that this file's diagnostic prints do not
//! repeat (uniform-field exact match, zero-velocity fields, streamline
//! termination boundaries, extreme magnitudes).
//!
//! # Scenario
//!
//! A single rigid-body rotation ("vortex") velocity field about the z
//! axis, `v(p) = omega * (-p.y, p.x, 0)`, is used for both halves of the
//! module:
//!
//! * `generate_flow_arrows` is checked against a grid with exactly one
//!   fluid particle placed at each sampling cell's own center, so the
//!   function's neighbor-averaging sees only that one particle (its
//!   neighbors are exactly one sampling-cell-width away, which is never
//!   closer than the search radius) and reduces to exact point
//!   evaluation of the field -- compared against `v(center)` computed
//!   independently via the field's own definition, not via
//!   `generate_flow_arrows` itself.
//! * `generate_streamlines` is checked the same way: fluid particles are
//!   placed at the exact points of a hand-rolled forward-Euler
//!   integration of the same field, spaced far enough apart that the
//!   function's inverse-distance-weighted sampling always sees exactly
//!   one particle (itself, at distance zero) at every traced position.
//!   That hand-rolled path is cross-checked against an independent
//!   closed-form identity of the field itself (a conformal radius-growth
//!   factor), not merely re-derived by running the same recursion twice.
//!
//! ```bash
//! cargo run --example flow_visualization --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::flow_viz::{
    generate_flow_arrows, generate_streamlines, FlowArrow, FlowVizConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};

/// Rigid-body rotation about the z axis: `v(p) = omega * (-p.y, p.x, 0)`.
///
/// This is the field's own closed-form definition -- not a call into
/// anything `flow_viz` exports.
fn vortex_velocity(p: Vec3Fix, omega: Fix128) -> Vec3Fix {
    Vec3Fix::new(-(omega * p.y), omega * p.x, Fix128::ZERO)
}

fn main() {
    // ------------------------------------------------------------------
    // 1. FlowArrow -- a hand-built struct literal, fields read back
    //    unchanged. Not derived from calling `generate_flow_arrows`.
    // ------------------------------------------------------------------
    let hand_arrow = FlowArrow {
        position: Vec3Fix::from_int(1, 2, 3),
        direction: Vec3Fix::UNIT_X,
        magnitude: Fix128::from_int(7),
    };
    assert_eq!(
        hand_arrow.position,
        Vec3Fix::from_int(1, 2, 3),
        "[flow_viz] MISMATCH FlowArrow.position round-trip"
    );
    assert_eq!(
        hand_arrow.direction,
        Vec3Fix::UNIT_X,
        "[flow_viz] MISMATCH FlowArrow.direction round-trip"
    );
    assert_eq!(
        hand_arrow.magnitude,
        Fix128::from_int(7),
        "[flow_viz] MISMATCH FlowArrow.magnitude round-trip"
    );
    println!(
        "[flow_viz] ok FlowArrow struct literal round-trip: position={:?} direction={:?} magnitude={:.1}",
        hand_arrow.position,
        hand_arrow.direction,
        hand_arrow.magnitude.to_f64()
    );

    // ------------------------------------------------------------------
    // 2. FlowVizConfig / generate_flow_arrows -- a 4^3 grid over
    //    [-2,2]^3 (grid spacing dx = 1 exactly), vortex strength
    //    omega = 1/2, one fluid particle placed at each cell's own
    //    center with that exact velocity. `generate_flow_arrows`'s
    //    averaging radius equals the grid spacing, so no two distinct
    //    cell centers (always exactly a multiple of dx apart) can both
    //    fall inside a query cell's search radius -- confirmed below by
    //    computing every pairwise distance, not assumed.
    // ------------------------------------------------------------------
    let resolution = 4_usize;
    let bounds_min = Vec3Fix::from_int(-2, -2, -2);
    let bounds_max = Vec3Fix::from_int(2, 2, 2);
    let omega = Fix128::from_ratio(1, 2);
    let config = FlowVizConfig {
        grid_resolution: resolution,
        bounds_min,
        bounds_max,
        arrow_scale: Fix128::ONE,
    };

    // dx = (max - min) / resolution, re-derived from the grid-resolution
    // documented in `generate_flow_arrows`'s own doc comment ("samples
    // velocity on a regular 3D grid") -- a cube, so the same spacing
    // applies on every axis.
    let dx = (bounds_max.x - bounds_min.x) / Fix128::from_int(resolution as i64);
    let half_dx = dx.half();

    let mut cell_centers: Vec<Vec3Fix> = Vec::with_capacity(resolution * resolution * resolution);
    for iz in 0..resolution {
        for iy in 0..resolution {
            for ix in 0..resolution {
                cell_centers.push(Vec3Fix::new(
                    bounds_min.x + dx * Fix128::from_int(ix as i64) + half_dx,
                    bounds_min.y + dx * Fix128::from_int(iy as i64) + half_dx,
                    bounds_min.z + dx * Fix128::from_int(iz as i64) + half_dx,
                ));
            }
        }
    }

    let sampling_radius_sq = dx * dx;
    for (i, &a) in cell_centers.iter().enumerate() {
        for &b in &cell_centers[i + 1..] {
            let d = a - b;
            let dist_sq = d.dot(d);
            assert!(
                dist_sq >= sampling_radius_sq,
                "[flow_viz] construction error: cell centers {a:?} and {b:?} are closer than the sampling radius"
            );
        }
    }

    let fluid_positions: Vec<Vec3Fix> = cell_centers.clone();
    let fluid_velocities: Vec<Vec3Fix> = cell_centers
        .iter()
        .map(|&c| vortex_velocity(c, omega))
        .collect();

    let arrows = generate_flow_arrows(&fluid_positions, &fluid_velocities, &config);
    assert_eq!(
        arrows.len(),
        cell_centers.len(),
        "[flow_viz] MISMATCH generate_flow_arrows(vortex): expected exactly one arrow per grid cell"
    );

    let mut max_perp_dev = 0.0_f64;
    for (arrow, &c) in arrows.iter().zip(cell_centers.iter()) {
        assert_eq!(
            arrow.position, c,
            "[flow_viz] MISMATCH generate_flow_arrows(vortex) arrow position"
        );
        let expected_v = vortex_velocity(c, omega);
        let expected_dir = expected_v.normalize();
        let expected_mag = expected_v.length() * config.arrow_scale;
        assert_eq!(
            arrow.direction, expected_dir,
            "[flow_viz] MISMATCH generate_flow_arrows(vortex) direction at {c:?}"
        );
        assert_eq!(
            arrow.magnitude, expected_mag,
            "[flow_viz] MISMATCH generate_flow_arrows(vortex) magnitude at {c:?}"
        );

        // Diagnostic (not the oracle above): a solid-body rotation's
        // velocity is exactly perpendicular to the radius vector in the
        // xy-plane. `normalize()` on the radial vector floors a sqrt, so
        // this is checked with a tolerance tied to that rounding, not a
        // tuned fudge factor.
        let radial = Vec3Fix::new(c.x, c.y, Fix128::ZERO).normalize();
        let perp_dev = arrow.direction.dot(radial).to_f64().abs();
        if perp_dev > max_perp_dev {
            max_perp_dev = perp_dev;
        }
    }
    assert!(
        max_perp_dev < 1e-9,
        "[flow_viz] MISMATCH generate_flow_arrows(vortex): tangential direction not perpendicular to radius, max |dot| = {max_perp_dev:e}"
    );
    println!(
        "[flow_viz] ok generate_flow_arrows(vortex, omega={:.2}, grid={}^3): {} arrows, \
         exact direction/magnitude match at every cell, max |tangent . radial| = {max_perp_dev:e}",
        omega.to_f64(),
        resolution,
        arrows.len()
    );

    // ------------------------------------------------------------------
    // 3. generate_streamlines -- forward-Euler integration of the SAME
    //    vortex field, verified two independent ways:
    //
    //    (a) bit-exact match against a hand-rolled copy of the Euler
    //        recursion `p_next = p + omega * perp(p) * dt`, and
    //    (b) that hand-rolled path itself satisfies a closed-form
    //        identity that follows from the field's definition alone:
    //        expanding |p + k*perp(p)|^2 (k = omega*dt) gives
    //        |p|^2 + 2k*(p . perp(p)) + k^2*|perp(p)|^2. Since
    //        p . perp(p) = p.x*(-p.y) + p.y*p.x = 0 and
    //        |perp(p)|^2 = p.y^2 + p.x^2 = |p|^2 for every p, this is
    //        exactly |p_next|^2 = (1 + k^2) * |p|^2 -- a property of the
    //        rotation field itself, not of either implementation.
    // ------------------------------------------------------------------
    let dt = Fix128::ONE;
    let k = omega * dt;
    let steps = 4_usize;
    let seed = Vec3Fix::from_int(10, 0, 0);

    let mut hand_path: Vec<Vec3Fix> = Vec::with_capacity(steps + 1);
    hand_path.push(seed);
    for _ in 0..steps {
        let last = *hand_path
            .last()
            .expect("hand_path always has at least the seed");
        hand_path.push(last + vortex_velocity(last, omega) * dt);
    }

    let growth = Fix128::ONE + k * k;
    for i in 0..steps {
        let got = hand_path[i + 1].length_squared();
        let want = hand_path[i].length_squared() * growth;
        assert_eq!(
            got, want,
            "[flow_viz] construction error: hand-rolled Euler step {i} violates the \
             conformal growth identity |p_next|^2 = (1 + k^2)|p|^2"
        );
    }

    // Sanity: every pair of the first `steps` hand-path points is at
    // least the hardcoded influence radius (`Fix128::ONE`) apart, so
    // placing one fluid particle at each -- with the field's own
    // velocity there -- makes `generate_streamlines`'s inverse-distance
    // weighting see exactly one particle (itself, distance zero) at
    // every traced position.
    let influence_radius_sq = Fix128::ONE;
    for i in 0..steps {
        for j in (i + 1)..steps {
            let d = hand_path[i] - hand_path[j];
            let dist_sq = d.dot(d);
            assert!(
                dist_sq >= influence_radius_sq,
                "[flow_viz] construction error: hand-path points {i} and {j} are closer than the influence radius"
            );
        }
    }

    let fluid_positions: Vec<Vec3Fix> = hand_path[..steps].to_vec();
    let fluid_velocities: Vec<Vec3Fix> = fluid_positions
        .iter()
        .map(|&p| vortex_velocity(p, omega))
        .collect();

    let lines = generate_streamlines(&fluid_positions, &fluid_velocities, &[seed], steps, dt);
    assert_eq!(
        lines.len(),
        1,
        "[flow_viz] MISMATCH generate_streamlines(vortex): expected exactly one streamline"
    );
    let traced = lines.line(0);
    assert_eq!(
        traced.len(),
        hand_path.len(),
        "[flow_viz] MISMATCH generate_streamlines(vortex): point count"
    );
    for (got, want) in traced.iter().zip(hand_path.iter()) {
        assert_eq!(
            got, want,
            "[flow_viz] MISMATCH generate_streamlines(vortex): traced point does not match the hand-rolled Euler spiral"
        );
    }
    println!(
        "[flow_viz] ok generate_streamlines(vortex, omega={:.2}, dt={:.1}, steps={}): traced {} points, \
         bit-exact match to the hand-rolled Euler spiral; radius grows by a factor of {:.6} per step \
         (closed form: 1 + (omega*dt)^2)",
        omega.to_f64(),
        dt.to_f64(),
        steps,
        traced.len(),
        growth.to_f64()
    );

    println!("[flow_viz] all checks passed");
}
