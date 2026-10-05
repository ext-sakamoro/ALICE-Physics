//! Two groups walking in opposite directions along a corridor, with the
//! social force model (Helbing–Molnár 1995 / Helbing–Farkas–Vicsek 2000).
//!
//! ```text
//! m dv/dt = m (v0 ê − v)/τ + Σ_j f_ij + Σ_W f_iW
//! f_ij = w_i A e^{(r_ij − d_ij)/B} n_ij + k g(r_ij − d_ij) n_ij + κ g(r_ij − d_ij) Δv^t_ji t_ij
//! ```
//!
//! Before the corridor run the example checks a few closed forms against
//! the crate in `f64`: the driving force, the pair repulsion with the view
//! weight, the wall repulsion, and the distance at which a pedestrian walking
//! into a wall stops, `d* = r + B ln(A τ/(m v0))`. During the run it compares
//! the cell-list and direct-sum forces bit for bit and reports how many
//! pedestrians got through the oncoming group.
//!
//! ```bash
//! cargo run --release --example crowd_corridor
//! ```

// f64 `exp` / `ln` compute closed-form references, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::crowd_force::{
    CrowdForceError, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
};
use alice_physics::math::Fix128;
use alice_physics::physics2d::Vec2Fix;

// Representative values of Helbing, Farkas, Vicsek, Nature 407, 487 (2000),
// and a view weight of the order fitted by Johansson, Helbing, Shukla (2007).
const A: f64 = 2000.0;
const B: f64 = 0.08;
const K: f64 = 1.2e5;
const KAPPA: f64 = 2.4e5;
const LAMBDA: f64 = 0.1;
const MASS: f64 = 80.0;
const TAU: f64 = 0.5;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v2(x: f64, y: f64) -> Vec2Fix {
    Vec2Fix::new(fx(x), fx(y))
}

fn pedestrian(x: f64, y: f64, heading: f64, radius: f64, v0: f64) -> Pedestrian {
    Pedestrian {
        position: v2(x, y),
        velocity: Vec2Fix::ZERO,
        radius_m: fx(radius),
        mass_kg: fx(MASS),
        desired_speed_m_s: fx(v0),
        desired_direction: v2(heading, 0.0),
        relaxation_time_s: fx(TAU),
    }
}

fn check(label: &str, actual: f64, expected: f64, rel_tol: f64) {
    let err = ((actual - expected) / expected).abs();
    println!("  {label:<34} {actual:>14.6} (closed form {expected:.6}, rel err {err:.1e})");
    assert!(err <= rel_tol, "{label}: rel err {err} > {rel_tol}");
}

fn main() -> Result<(), CrowdForceError> {
    let params = InteractionParams {
        strength_n: fx(A),
        range_m: fx(B),
        body_stiffness: fx(K),
        sliding_friction: fx(KAPPA),
    };
    let model = SocialForce::new(params, params, fx(LAMBDA), fx(1.2))?;

    // `Fix128::exp` is relative 1e-6 (src/math.rs); the rest is 2⁻⁶⁴ per product.
    let tol = 2e-6;
    println!("closed forms");
    let walker = pedestrian(0.0, 0.0, 1.0, 0.3, 1.34);
    check(
        "driving force m v0/τ (N)",
        model.driving_force(&walker)?.x.to_f64(),
        MASS * 1.34 / TAU,
        1e-12,
    );
    let ahead = pedestrian(0.9, 0.0, -1.0, 0.3, 1.34);
    let (on_walker, on_ahead) = model.pair_forces(&walker, &ahead);
    let social = A * ((0.6 - 0.9) / B).exp();
    check(
        "repulsion, neighbour ahead (N)",
        -on_walker.x.to_f64(),
        social,
        tol,
    );
    // `ahead` heads −x and sees `walker` in front too: weight 1 on both sides.
    check(
        "repulsion on the other (N)",
        on_ahead.x.to_f64(),
        social,
        tol,
    );
    let behind_weight = model.anisotropy_weight(&walker, v2(1.0, 0.0)).to_f64();
    check(
        "view weight, neighbour behind",
        behind_weight,
        LAMBDA,
        1e-12,
    );
    let wall = WallSegment {
        start: v2(-5.0, 0.0),
        end: v2(5.0, 0.0),
    };
    let near_wall = pedestrian(0.0, 0.5, 1.0, 0.3, 1.34);
    check(
        "wall repulsion (N)",
        model.wall_force(&near_wall, &wall).y.to_f64(),
        A * ((0.3 - 0.5) / B).exp(),
        tol,
    );
    let front = WallSegment {
        start: v2(0.0, -5.0),
        end: v2(0.0, 5.0),
    };
    let mut stopper = [pedestrian(-3.0, 0.0, 1.0, 0.3, 0.8)];
    for _ in 0..4000 {
        model.step(
            &mut stopper,
            &[front],
            fx(0.01),
            NeighborSearch::Direct,
            None,
        )?;
    }
    check(
        "stopping distance from a wall (m)",
        -stopper[0].position.x.to_f64(),
        0.3 + B * (A * TAU / (MASS * 0.8)).ln(),
        2e-6,
    );

    // Corridor 80 m × 3 m, 10 walking +x and 10 walking −x in staggered files.
    // The model has no noise: two blocks meeting head-on in a mirror-symmetric
    // start can stall, so the files are offset sideways.
    let walls = [
        WallSegment {
            start: v2(-40.0, -1.5),
            end: v2(40.0, -1.5),
        },
        WallSegment {
            start: v2(-40.0, 1.5),
            end: v2(40.0, 1.5),
        },
    ];
    let mut crowd = Vec::new();
    for k in 0..10 {
        let r = 0.25 + 0.02 * ((k * 3 % 5) as f64);
        let side = if k % 2 == 0 { 0.5 } else { -0.5 };
        let x = 2.0 * k as f64;
        crowd.push(pedestrian(
            -6.0 - x,
            side + 0.05 * (k % 3) as f64,
            1.0,
            r,
            1.3,
        ));
        crowd.push(pedestrian(
            6.0 + x,
            -side - 0.04 * (k % 4) as f64,
            -1.0,
            r,
            1.2,
        ));
    }
    let h = fx(0.004);
    let (mut direct, mut cells) = (Vec::new(), Vec::new());
    println!("\ncorridor: {} pedestrians, h = 4 ms", crowd.len());
    // Past the oncoming group: +x walkers beyond x = 25, −x walkers below x = −25 (every start lies within |x| ≤ 24).
    let passed = |crowd: &[Pedestrian]| {
        crowd
            .iter()
            .filter(|p| (p.position.x * p.desired_direction.x).to_f64() > 25.0)
            .count()
    };
    for step in 1..=15_000 {
        if step % 3000 == 0 {
            model.total_forces(&crowd, &walls, NeighborSearch::Direct, &mut direct)?;
            model.total_forces(&crowd, &walls, NeighborSearch::CellList, &mut cells)?;
            assert_eq!(direct, cells, "cell list and direct sum differ");
            let mean_speed: f64 = crowd
                .iter()
                .map(|p| (p.velocity.x * p.desired_direction.x).to_f64())
                .sum::<f64>()
                / crowd.len() as f64;
            println!(
                "  t = {:>4.1} s: through the other group {:>2}/{}, mean speed towards the goal {mean_speed:.3} m/s (forces bit-identical)",
                step as f64 * 0.004,
                passed(&crowd),
                crowd.len()
            );
        }
        model.step(
            &mut crowd,
            &walls,
            h,
            NeighborSearch::CellList,
            Some(fx(1.3)),
        )?;
    }
    assert_eq!(
        passed(&crowd),
        crowd.len(),
        "the two groups did not get through"
    );
    for p in &crowd {
        let y = p.position.y.to_f64();
        assert!(y.abs() < 1.5, "a pedestrian left the corridor: y = {y}");
    }
    Ok(())
}
