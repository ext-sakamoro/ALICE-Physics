#![cfg(feature = "mpm2d")]
use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel, V2};
fn fx(v: f64) -> Fix128 { Fix128::from_f64(v) }
/// an L-shaped room: the box [0.1, 0.5]² less the wedge x > C and y > C
const C: f64 = 0.36;
fn walls(x: f64, y: f64) -> (f64, [f64; 2]) {
    let (a, b) = (x - C, y - C);
    if a > 0.0 && b > 0.0 && x < 0.5 + a.min(b) && y < 0.5 + a.min(b) {
        return if a < b { (-a, [-1.0, 0.0]) } else { (-b, [0.0, -1.0]) };
    }
    let out = [(0.1 - x, [1.0, 0.0]), (x - 0.5, [-1.0, 0.0]), (0.1 - y, [0.0, 1.0]), (y - 0.5, [0.0, -1.0])];
    let worst = out.iter().fold(out[0], |m, o| if o.0 > m.0 { *o } else { m });
    if worst.0 >= 0.0 { (-worst.0, worst.1) } else { (1.0, [1.0, 0.0]) }
}
#[test]
fn corner_depth() {
    for (cell, e, push) in [(0.01, 2.0e4, 30.0), (0.01, 2.0e4, 100.0), (0.01, 2.0e4, 300.0), (0.01, 2.0e3, 30.0), (0.01, 2.0e3, 100.0), (0.005, 2.0e4, 100.0)] {
        let (nu, rho) = (0.4f64, 1000.0f64);
        let wave = (e / ((1.0 - nu * nu) * rho)).sqrt();
        let frame = 1.0 / 60.0;
        let per = (frame / (cell / wave / 3.0)).ceil() as usize;
        let dt = frame / per as f64;
        let n = (0.64 / cell) as usize;
        let s = push / 2.0f64.sqrt();
        let mut w = Mpm2d::new(
            Mpm2dConfig { origin: [Fix128::ZERO; 2], cell: fx(cell), nodes: [n, n], gravity: [fx(s), fx(s)], dt: fx(dt), ground: None },
            Mpm2dMaterial { youngs: fx(e), poisson: fx(nu), density: fx(rho), model: Mpm2dModel::NeoHookean, maxwell: None },
        ).unwrap();
        w.set_walls(|x: V2| { let (d, nn) = walls(x[0].to_f64(), x[1].to_f64()); (fx(d), [fx(nn[0]), fx(nn[1])]) }, Fix128::ZERO).unwrap();
        w.seed(|x: V2| { let (a, b) = (x[0].to_f64() - 0.27, x[1].to_f64() - 0.27); fx((a * a + b * b).sqrt() - 0.06) }, 2);
        let mut deepest = f64::NEG_INFINITY;
        let mut end = String::from("ran 120 frames");
        'run: for fr in 0..120 {
            for _ in 0..per { if let Err(er) = w.step() { end = format!("frame {fr} {er:?}"); break 'run; } }
            let d = w.particles().iter().map(|p| (p.x[0].to_f64() - C).min(p.x[1].to_f64() - C)).fold(f64::NEG_INFINITY, f64::max);
            deepest = deepest.max(d);
        }
        let box_depth = w.particles().iter().map(|p| { let (x, y) = (p.x[0].to_f64(), p.x[1].to_f64()); (0.1 - x).max(x - 0.5).max(0.1 - y).max(y - 0.5) }).fold(f64::NEG_INFINITY, f64::max);
        println!("cell {cell} E {e} push {push}: wedge worst {:.2} cells, flat wall at the end {:.2} cells, {end}", deepest / cell, box_depth / cell);
    }
}
