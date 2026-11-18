//! The laminar point plume, measured: a 2 cm cube heating the floor of a closed box of air
//! (0.32 × 0.64 × 0.32 m, `ν = κ`), run until the plume is steady, then averaged over a window.
//! A laminar point plume is self-similar: its width grows as `R² ∝ z`, its centreline speed is
//! the same at every height, and its centreline excess temperature falls as `1/z` (Fujii 1963,
//! Pr = 1). This run prints the three exponents fitted over a band of heights, and how many
//! cells the plume's radius spans there; the oracle compares the exponents at two cells
//! (10 mm and 5 mm) and asks that they converge to (1, 0, −1).
//!
//! It is long (minutes at 10 mm, hours at 5 mm), so it saves a checkpoint (the plume's state
//! and the window sums so far) every few seconds of simulated time and resumes from it.
//!
//! Run: `cargo run --release --example smoke_laminar_plume -- <dx_mm> <nu_um2> <heat_k_per_s>
//! <seconds> <window_from_s> <z_lo_m> <z_hi_m> [checkpoint]`
//! e.g. `-- 10 200 100 20 12 0.10 0.30 /tmp/plume10.ckpt`

#[path = "../tests/support/smoke_laminar_scene.rs"]
mod scene;

use alice_physics_core::smoke_plume::{from_quanta, SmokePlume, HEAT_QUANTUM_BITS};
use alice_physics_core::world_participant::Participant;
use std::io::Write;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"SMKLAM01";

/// The sums over the averaging window, per layer `j`.
struct Window {
    samples: u64,
    r2: Vec<f64>,
    w: Vec<f64>,
    t: Vec<f64>,
}

fn slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (sxx, sxy) = pts
        .iter()
        .fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.0 * p.1));
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn save(path: &str, step: u64, plume: &SmokePlume, win: &Window) -> std::io::Result<()> {
    let mut state = Vec::new();
    plume.write_state(&mut state);
    let mut out = Vec::with_capacity(state.len() + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&step.to_le_bytes());
    out.extend_from_slice(&(state.len() as u64).to_le_bytes());
    out.extend_from_slice(&state);
    out.extend_from_slice(&win.samples.to_le_bytes());
    for v in win.r2.iter().chain(&win.w).chain(&win.t) {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    // write beside, then rename: a run stopped mid-write keeps the previous checkpoint
    let tmp = format!("{path}.tmp");
    std::fs::File::create(&tmp)?.write_all(&out)?;
    std::fs::rename(&tmp, path)
}

fn load(path: &str, plume: &mut SmokePlume, win: &mut Window) -> Option<u64> {
    let b = std::fs::read(path).ok()?;
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    assert_eq!(&b[..8], MAGIC, "not a laminar plume checkpoint");
    let step = u64_at(8);
    let len = u64_at(16) as usize;
    plume.read_state(&b[24..24 + len]);
    let mut o = 24 + len;
    win.samples = u64_at(o);
    o += 8;
    let n = win.r2.len();
    for v in win.r2.iter_mut().chain(win.w.iter_mut()).chain(win.t.iter_mut()) {
        *v = f64::from_bits(u64_at(o));
        o += 8;
    }
    assert_eq!(o, b.len(), "checkpoint length for {n} layers");
    Some(step)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 7 {
        return Err("args: dx_mm nu_um2 heat_k_per_s seconds window_from_s z_lo_m z_hi_m [checkpoint]".into());
    }
    let int = |i: usize| a[i].parse::<i64>();
    let real = |i: usize| a[i].parse::<f64>();
    let (dx_mm, nu, heat, seconds, from) = (int(0)?, int(1)?, int(2)?, int(3)?, int(4)?);
    let (z_lo, z_hi) = (real(5)?, real(6)?);
    let ckpt = a.get(7).cloned();

    let mut plume = scene::scene(dx_mm, nu, heat);
    let step = scene::step(dx_mm);
    let per_second = (1200 / dx_mm) as u64;
    let total = seconds as u64 * per_second;
    let window_from = from as u64 * per_second;
    let (nx, ny, nz) = {
        let g = plume.velocity();
        (g.nx, g.ny, g.nz)
    };
    let mut win = Window {
        samples: 0,
        r2: vec![0.0; ny],
        w: vec![0.0; ny],
        t: vec![0.0; ny],
    };
    let mut n = 0u64;
    if let Some(p) = &ckpt {
        if let Some(s) = load(p, &mut plume, &mut win) {
            println!("resumed at step {s} ({:.2} s)", s as f64 / per_second as f64);
            n = s;
        }
    }

    let dx = dx_mm as f64 / 1000.0;
    let (ci, ck) = (nx / 2, nz / 2);
    let clock = Instant::now();
    let mut max_sub = 0;
    while n < total {
        plume.advance(step).map_err(|e| format!("advance at step {n}: {e:?}"))?;
        n += 1;
        max_sub = max_sub.max(plume.last_transport_substeps());
        if n > window_from {
            win.samples += 1;
            let f = plume.heat();
            let g = plume.velocity();
            for j in 0..ny {
                let (mut m0, mut m2) = (0.0, 0.0);
                for k in 0..nz {
                    for i in 0..nx {
                        let q = from_quanta(f.quanta()[f.index(i, j, k)], HEAT_QUANTUM_BITS).to_f64();
                        let x = (i as f64 + 0.5) * dx - 0.5 * nx as f64 * dx;
                        let z = (k as f64 + 0.5) * dx - 0.5 * nz as f64 * dx;
                        m0 += q;
                        m2 += q * (x * x + z * z);
                    }
                }
                if m0 > 0.0 {
                    win.r2[j] += m2 / m0;
                }
                // the centreline: the four cells around the box's axis
                let (mut w, mut t) = (0.0, 0.0);
                for (i, k) in [(ci - 1, ck - 1), (ci, ck - 1), (ci - 1, ck), (ci, ck)] {
                    w += g.cell_velocity(i, j, k).1.to_f64() / 4.0;
                    t += from_quanta(f.quanta()[f.index(i, j, k)], HEAT_QUANTUM_BITS).to_f64() / 4.0;
                }
                win.w[j] += w;
                win.t[j] += t;
            }
        }
        if n % (2 * per_second) == 0 {
            if let Some(p) = &ckpt {
                save(p, n, &plume, &win)?;
            }
            println!(
                "t {:.1} s, {:.0} ms/step, max transport substeps {max_sub}",
                n as f64 / per_second as f64,
                clock.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64
            );
        }
    }
    let ledger = plume.ledger();
    assert_eq!(plume.heat().total(), Some(ledger.heat_expected()), "heat ledger");
    if win.samples == 0 {
        return Err("the window holds no samples".into());
    }

    let s = win.samples as f64;
    let z0 = scene::source_top_m(dx_mm);
    let (mut pr, mut pw, mut pt, mut min_cells) = (vec![], vec![], vec![], f64::INFINITY);
    println!(" j   z (m)   R (cells)  R²/z (m)   w_c (m/s)   ΔT_c·z (K·m)");
    for j in 0..ny {
        let z = (j as f64 + 0.5) * dx - z0;
        if z <= 0.0 {
            continue;
        }
        let (r2, w, t) = (win.r2[j] / s, win.w[j] / s, win.t[j] / s);
        let cells = r2.sqrt() / dx;
        println!("{j:3} {z:7.4} {cells:9.2} {:11.4e} {w:11.4e} {:11.4e}", r2 / z, t * z);
        if z >= z_lo && z <= z_hi && r2 > 0.0 && w > 0.0 && t > 0.0 {
            pr.push((z.ln(), r2.ln()));
            pw.push((z.ln(), w.ln()));
            pt.push((z.ln(), t.ln()));
            min_cells = min_cells.min(cells);
        }
    }
    if pr.len() < 3 {
        return Err(format!("only {} layers in the band {z_lo}..{z_hi} m", pr.len()).into());
    }
    println!(
        "RESULT dx_mm {dx_mm} nu_um2 {nu} heat {heat} band {z_lo}..{z_hi} layers {} min_radius_cells {min_cells:.2} \
         exp_R2 {:.4} exp_w {:.4} exp_T {:.4} (laminar 1 0 -1) ms_per_step {:.1}",
        pr.len(),
        slope(&pr),
        slope(&pw),
        slope(&pt),
        clock.elapsed().as_secs_f64() * 1e3 / total as f64,
    );
    Ok(())
}
