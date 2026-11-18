//! The laminar point plume, measured: a 2 cm cube heating the floor of a box of air
//! (0.32 × 0.64 × 0.32 m, `ν = κ`, the top open as an outflow), run until the plume is steady,
//! then averaged over a window. A laminar point plume is self-similar: its width grows as
//! `R² ∝ z`, its centreline speed is the same at every height, and its centreline excess
//! temperature falls as `1/z` (Fujii 1963, Pr = 1). This run prints the three exponents fitted
//! over a band of heights and how many cells the plume's radius spans there; the oracle compares
//! the exponents as the cell shrinks (10, 5, 2.5 mm) and asks that they converge to (1, 0, −1).
//!
//! The closed form holds only in an unbounded, unstratified ambient around a steady plume, so the
//! run checks its own scene against that before it reports exponents, and fails as "scene
//! invalid" (never as a pass) when one of the three does not hold:
//! - **reach inside the box**: per layer, the window-mean radial profile `P(r)` falls to the far
//!   field, the first radius where `P − P_min ≤ f · (P(0) − P_min)` (`f` = 1 %), and at least
//!   2 cm of far field lie between that radius and the wall (the box is at least
//!   `2 (reach + 2 cm)` wide, a length whatever the cell);
//! - **far field unstratified**: the far field (the plateau's mean) changes with height so little
//!   that `N · t_window < 0.1`, `N² = g β dT∞/dz`;
//! - **outflow steady**: over the window the heat leaving through the top equals the heat the
//!   source adds, within 5 % (a closed box fills from the top down and is never valid).
//!
//! The width and the centreline excess are measured on the excess over that far field.
//!
//! The window is set by a rule, not by hand: it starts at the first whole second `t0` at which the
//! heat that left has matched what the source added within 5 % (on both sides: the plume's first
//! arrival at the top overshoots) over a span of one transit of the box (its height at the
//! fastest centreline speed, in whole seconds), at each of the last transit's seconds (an
//! outflow passing through the band while the box fills does not stay), and it
//! lasts 8 s; a window over which the outflow was not steady after all is dropped and the rule
//! looks for the next start, and the run fails when no steady window ends within the given maximum
//! time.
//!
//! It is long (minutes at 10 mm, hours at 5 mm), so it saves a checkpoint (the plume's state, the
//! outflow history and the window sums so far) every 2 s of simulated time and resumes from it.
//!
//! Run: `cargo run --release --example smoke_laminar_plume -- <dx_mm> <nu_um2> <heat_k_per_s>
//! <max_seconds> <z_lo_m> <z_hi_m> <closed|open> [checkpoint] [coarse checkpoint]`,
//! e.g. `-- 10 200 400 40 0.10 0.40 open /tmp/plume10.ckpt`. With a coarse checkpoint (a run
//! of the same scene at twice the cell) and no checkpoint of its own yet, the run starts from that
//! state prolonged to this cell (`SmokePlume::prolonged`) instead of from rest; the window should
//! start a while after it.

#[path = "../tests/support/smoke_laminar_scene.rs"]
mod scene;

use alice_physics_core::smoke_plume::{from_quanta, SmokeConfig, SmokePlume, HEAT_QUANTUM_BITS};
use alice_physics_core::world_participant::Participant;
use std::io::Write;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"SMKLAM06";
/// The format before the window's span was stored: its outflow history resumes, its window
/// (opened under an earlier rule) is dropped.
const MAGIC_V5: &[u8; 8] = b"SMKLAM05";
/// Earlier formats: a run resumes from their plume's state alone (its outflow history and
/// window start afresh), and a coarse run's state prolongs from any of them.
const MAGIC_OLD: [&[u8; 8]; 4] = [b"SMKLAM01", b"SMKLAM02", b"SMKLAM03", b"SMKLAM04"];
use scene::{sustained_steady, transit_span_s, STEADY_TOLERANCE};
/// The window's length (s).
const WINDOW_S: u64 = 8;
/// The far field is where the window-mean excess is within this fraction of the centreline's
/// excess over the layer's coolest radius.
const FAR_FIELD_FRACTION: f64 = 0.01;
/// The largest `N · t_window` of the far field for which the scene counts as unstratified.
const MAX_N_T: f64 = 0.1;
/// The far field must reach this far (m) between the plume's reach and the wall: a length, so
/// it is the same whatever the cell.
const FAR_FIELD_MARGIN_M: f64 = 0.02;

/// The outflow history and the sums over the averaging window.
struct Window {
    /// The heat that had left and the heat added, at each whole second from `hist_from`.
    hist: Vec<(i64, i64)>,
    hist_from: u64,
    /// The step at which the window starts (0 while not yet found), and the span (s) its
    /// steady rule used.
    start: u64,
    span: u64,
    samples: u64,
    /// The heat that had left through the top, and the heat the source had added, when the
    /// window began.
    out0: i64,
    added0: i64,
    /// The centreline speed summed per layer `j`.
    w: Vec<f64>,
    /// The excess temperature summed per layer and radius bin (`j · bins + b`, bin `b` holds
    /// the cells whose centre is `b..b+1` cells from the box's axis).
    prof: Vec<f64>,
}

/// Natural log of a measured quantity, for the log-log fits.
#[allow(
    clippy::disallowed_methods,
    reason = "measured quantity: the log of the simulated plume's width, speed and excess, fitted against the laminar exponents"
)]
fn ln(v: f64) -> f64 {
    v.ln()
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
    out.extend_from_slice(&win.hist_from.to_le_bytes());
    out.extend_from_slice(&(win.hist.len() as u64).to_le_bytes());
    for &(l, a) in &win.hist {
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&a.to_le_bytes());
    }
    out.extend_from_slice(&win.start.to_le_bytes());
    out.extend_from_slice(&win.span.to_le_bytes());
    out.extend_from_slice(&win.samples.to_le_bytes());
    out.extend_from_slice(&win.out0.to_le_bytes());
    out.extend_from_slice(&win.added0.to_le_bytes());
    for v in win.w.iter().chain(&win.prof) {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    // write beside, then rename: a run stopped mid-write keeps the previous checkpoint
    let tmp = format!("{path}.tmp");
    std::fs::File::create(&tmp)?.write_all(&out)?;
    std::fs::rename(&tmp, path)
}

/// The step and the plume's state of a checkpoint of any format, read into `plume` after it
/// accepts them; the rest of the bytes is returned.
fn read_plume<'a>(b: &'a [u8], plume: &mut SmokePlume) -> Result<(u64, bool, &'a [u8]), String> {
    let old = MAGIC_OLD.iter().any(|m| &b[..8] == *m);
    if !(old || &b[..8] == MAGIC || &b[..8] == MAGIC_V5) {
        return Err("not a laminar plume checkpoint".into());
    }
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let len = u64_at(16) as usize;
    let state = &b[24..24 + len];
    plume
        .check_state(state)
        .map_err(|e| format!("the scene refuses the state: {e:?}"))?;
    plume.read_state(state);
    Ok((u64_at(8), old, &b[24 + len..]))
}

/// Forget the window (start, span and sums); the outflow history stays.
fn drop_window(win: &mut Window) {
    win.start = 0;
    win.span = 0;
    win.samples = 0;
    win.w
        .iter_mut()
        .chain(win.prof.iter_mut())
        .for_each(|v| *v = 0.0);
}

fn load(path: &str, plume: &mut SmokePlume, win: &mut Window) -> Result<Option<u64>, String> {
    let Ok(b) = std::fs::read(path) else {
        return Ok(None);
    };
    let (step, old, rest) = read_plume(&b, plume).map_err(|e| format!("{path}: {e}"))?;
    if old {
        // its sums are not this format's: the plume's state resumes, the history starts afresh
        println!("{path}: an earlier format, resuming from the plume's state alone");
        return Ok(Some(step));
    }
    let at = |o: usize| u64::from_le_bytes(rest[o..o + 8].try_into().unwrap());
    win.hist_from = at(0);
    let len = at(8) as usize;
    win.hist = (0..len)
        .map(|h| (at(16 + 16 * h) as i64, at(24 + 16 * h) as i64))
        .collect();
    let v5 = &b[..8] == MAGIC_V5;
    let mut o = 16 + 16 * len;
    win.start = at(o);
    o += 8;
    if !v5 {
        win.span = at(o);
        o += 8;
    }
    win.samples = at(o);
    win.out0 = at(o + 8) as i64;
    win.added0 = at(o + 16) as i64;
    o += 24;
    for v in win.w.iter_mut().chain(win.prof.iter_mut()) {
        *v = f64::from_bits(at(o));
        o += 8;
    }
    if o != rest.len() {
        return Err(format!("{path}: checkpoint length"));
    }
    if v5 && win.start > 0 {
        // opened under the rule before the sustained one: dropped, the history stays
        println!("{path}: its window was opened under an earlier rule and is dropped");
        drop_window(win);
    }
    Ok(Some(step))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 7 {
        return Err(
            "args: dx_mm nu_um2 heat_k_per_s max_seconds z_lo_m z_hi_m closed|open \
                    [checkpoint] [coarse checkpoint]"
                .into(),
        );
    }
    let int = |i: usize| a[i].parse::<i64>();
    let real = |i: usize| a[i].parse::<f64>();
    // the cell in millimetres (10, 5, 2.5): held in micrometres, exactly
    let dx_um = (real(0)? * 1000.0).round() as i64;
    let dx_mm = dx_um as f64 / 1000.0;
    let (nu, heat, seconds) = (int(1)?, int(2)?, int(3)?);
    let (z_lo, z_hi) = (real(4)?, real(5)?);
    let open = match a[6].as_str() {
        "open" => true,
        "closed" => false,
        other => return Err(format!("the top is closed or open, not {other}").into()),
    };
    let ckpt = a.get(7).cloned();
    let coarse_ckpt = a.get(8).cloned();

    let mut plume = scene::scene_um(dx_um, nu, heat);
    if open {
        plume
            .set_open_top()
            .map_err(|e| format!("open the top: {e:?}"))?;
    }
    assert_eq!(plume.open_top(), open);
    let step = scene::step_um(dx_um);
    let per_second = (1_200_000 / dx_um) as u64;
    let total = seconds as u64 * per_second;
    let window_steps = WINDOW_S * per_second;
    let (nx, ny, nz) = {
        let g = plume.velocity();
        (g.nx, g.ny, g.nz)
    };
    // radius bins out to the box's corner
    let bins = ((nx.max(nz) as f64) * 0.5 * std::f64::consts::SQRT_2).ceil() as usize + 1;
    let bin_of = |i: usize, k: usize| {
        let x = i as f64 + 0.5 - 0.5 * nx as f64;
        let z = k as f64 + 0.5 - 0.5 * nz as f64;
        (x * x + z * z).sqrt() as usize
    };
    let mut bin_cells = vec![0.0; bins];
    for k in 0..nz {
        for i in 0..nx {
            bin_cells[bin_of(i, k)] += 1.0;
        }
    }
    let mut win = Window {
        hist: vec![],
        hist_from: 0,
        start: 0,
        span: 0,
        samples: 0,
        out0: 0,
        added0: 0,
        w: vec![0.0; ny],
        prof: vec![0.0; ny * bins],
    };
    let mut n = 0u64;
    if let Some(p) = &ckpt {
        if let Some(s) = load(p, &mut plume, &mut win)? {
            println!(
                "resumed at step {s} ({:.2} s)",
                s as f64 / per_second as f64
            );
            n = s;
        }
    }
    if n == 0 {
        if let Some(c) = &coarse_ckpt {
            let mut coarse = scene::scene_um(2 * dx_um, nu, heat);
            if open {
                coarse
                    .set_open_top()
                    .map_err(|e| format!("open the coarse top: {e:?}"))?;
            }
            let b = std::fs::read(c).map_err(|e| format!("{c}: {e}"))?;
            let (s, _, _) = read_plume(&b, &mut coarse).map_err(|e| format!("{c}: {e}"))?;
            let config = *plume.config();
            plume = coarse
                .prolonged(config)
                .map_err(|e| format!("prolong: {e:?}"))?;
            // a coarse step is two of this cell's
            n = 2 * s;
            println!(
                "prolonged from {c} at {:.2} s ({} cells to {} per axis)",
                n as f64 / per_second as f64,
                coarse.velocity().nx,
                plume.velocity().nx
            );
            assert_eq!(
                plume.heat().total(),
                Some(plume.heat_held_expected()),
                "prolonged heat"
            );
        }
    }

    let dx = dx_um as f64 / 1e6;
    let (ci, ck) = (nx / 2, nz / 2);
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the time per velocity step that this run prints"
    )]
    let clock = Instant::now();
    // the steps this process runs (a resumed run starts past 0)
    let first = n;
    let mut max_sub = 0;
    let window_done = |win: &Window, n: u64| win.start > 0 && n >= win.start + window_steps;
    while n < total && !window_done(&win, n) {
        plume
            .advance(step)
            .map_err(|e| format!("advance at step {n}: {e:?}"))?;
        n += 1;
        max_sub = max_sub.max(plume.last_transport_substeps());
        if n % per_second == 0 {
            if win.hist.is_empty() {
                win.hist_from = n / per_second;
            }
            win.hist
                .push((plume.outflow().heat_left, plume.ledger().heat_added));
            // the plume's transit: the box's height at the fastest centreline speed
            let g = plume.velocity();
            let w_max = (0..ny)
                .map(|j| {
                    [(ci - 1, ck - 1), (ci, ck - 1), (ci - 1, ck), (ci, ck)]
                        .iter()
                        .map(|&(i, k)| g.cell_velocity(i, j, k).1.to_f64() / 4.0)
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            let h = &win.hist;
            if let Some(span) = transit_span_s(w_max) {
                if win.start == 0 && sustained_steady(h, span) {
                    let now = h[h.len() - 1];
                    win.start = n;
                    win.span = span as u64;
                    win.out0 = now.0;
                    win.added0 = now.1;
                    println!(
                        "window starts at {:.1} s (steady for a transit of {span} s)",
                        n as f64 / per_second as f64
                    );
                }
            }
        }
        if win.start > 0 && n > win.start {
            win.samples += 1;
            let f = plume.heat();
            let g = plume.velocity();
            let temp = |i: usize, j: usize, k: usize| {
                from_quanta(f.quanta()[f.index(i, j, k)], HEAT_QUANTUM_BITS).to_f64()
            };
            for j in 0..ny {
                for k in 0..nz {
                    for i in 0..nx {
                        win.prof[j * bins + bin_of(i, k)] += temp(i, j, k);
                    }
                }
                // the centreline: the four cells around the box's axis
                let w: f64 = [(ci - 1, ck - 1), (ci, ck - 1), (ci - 1, ck), (ci, ck)]
                    .iter()
                    .map(|&(i, k)| g.cell_velocity(i, j, k).1.to_f64() / 4.0)
                    .sum();
                win.w[j] += w;
            }
        }
        if window_done(&win, n) {
            let out = (plume.outflow().heat_left - win.out0) as f64;
            let added = (plume.ledger().heat_added - win.added0) as f64;
            let ratio = out / added.max(1.0);
            if ratio.is_nan() || (ratio - 1.0).abs() > STEADY_TOLERANCE {
                // not steady over the whole window after all: look for the next start
                println!(
                    "the window from {:.1} s was not steady (out / added {ratio:.4}): dropped",
                    win.start as f64 / per_second as f64
                );
                drop_window(&mut win);
            }
        }
        if n % (2 * per_second) == 0 {
            if let Some(p) = &ckpt {
                save(p, n, &plume, &win)?;
            }
            #[allow(
                clippy::disallowed_methods,
                reason = "timing harness: the time per velocity step that this run prints"
            )]
            let ms = clock.elapsed().as_secs_f64() * 1e3 / (n - first).max(1) as f64;
            println!(
                "t {:.1} s, {ms:.0} ms/step, max transport substeps {max_sub}",
                n as f64 / per_second as f64,
            );
        }
    }
    assert_eq!(
        plume.heat().total(),
        Some(plume.heat_held_expected()),
        "heat ledger"
    );
    assert_eq!(
        plume.soot().total(),
        Some(plume.soot_held_expected()),
        "soot ledger"
    );
    if win.start == 0 {
        return Err(format!(
            "scene invalid: outflow never steady: the heat leaving did not stay within \
             {STEADY_TOLERANCE} of the heat added for a transit of the box, and a whole window, by \
             {seconds} s"
        )
        .into());
    }
    if n < win.start + window_steps {
        return Err(format!(
            "the window from {:.1} s does not end within {seconds} s",
            win.start as f64 / per_second as f64
        )
        .into());
    }

    let s = win.samples as f64;
    let z0 = scene::source_top_m();
    let inscribed = nx.min(nz) / 2;
    let (mut reach_max, mut missing, mut min_cells) = (0usize, 0usize, f64::INFINITY);
    let (mut pr, mut pw, mut pt, mut pa) = (vec![], vec![], vec![], vec![]);
    println!(" j   z (m)  reach (cells)  plateau  far field (K)  R (cells)   w_c (m/s)   ΔT_c (K)");
    for j in 0..ny {
        let z = (j as f64 + 0.5) * dx - z0;
        if z < z_lo || z > z_hi {
            continue;
        }
        let pm: Vec<f64> = (0..bins)
            .map(|b| {
                if bin_cells[b] > 0.0 {
                    win.prof[j * bins + b] / s / bin_cells[b]
                } else {
                    f64::NAN
                }
            })
            .collect();
        let inside = &pm[..=inscribed.min(bins - 1)];
        let p0 = pm[0];
        let pmin = inside.iter().copied().fold(f64::INFINITY, f64::min);
        let cut = pmin + FAR_FIELD_FRACTION * (p0 - pmin);
        let Some(r_f) = inside.iter().position(|&v| v <= cut) else {
            missing += 1;
            continue;
        };
        let plateau: Vec<f64> = inside[r_f..]
            .iter()
            .copied()
            .take_while(|&v| v <= cut)
            .collect();
        if (plateau.len() as f64) * dx < FAR_FIELD_MARGIN_M - 1e-9 {
            missing += 1;
            continue;
        }
        reach_max = reach_max.max(r_f);
        let far = plateau.iter().sum::<f64>() / plateau.len() as f64;
        let (mut m0, mut m2) = (0.0, 0.0);
        for b in 0..bins {
            if bin_cells[b] == 0.0 {
                continue;
            }
            let e = (pm[b] - far).max(0.0) * bin_cells[b];
            let r = (b as f64 + 0.5) * dx;
            m0 += e;
            m2 += e * r * r;
        }
        let (tc, w) = (p0 - far, win.w[j] / s);
        let cells = (m2 / m0).sqrt() / dx;
        if j % 4 == 0 {
            println!(
                "{j:3} {z:7.4} {r_f:10} {:8} {far:13.4e} {cells:10.2} {w:11.4e} {tc:11.4e}",
                plateau.len()
            );
        }
        if m0 > 0.0 && tc > 0.0 && w > 0.0 {
            pr.push((ln(z), ln(m2 / m0)));
            pw.push((ln(z), ln(w)));
            pt.push((ln(z), ln(tc)));
            pa.push((z, far));
            min_cells = min_cells.min(cells);
        }
    }

    // the scene's validity for the closed form, each checked before the exponents count
    let mut invalid = vec![];
    // the reach must leave FAR_FIELD_MARGIN_M of far field before the wall on every layer: the
    // box must be at least 2 (reach + margin) wide
    let needed = 2.0 * (reach_max as f64 * dx + FAR_FIELD_MARGIN_M);
    if missing > 0 || pr.len() < 3 || needed > 2.0 * inscribed as f64 * dx + 1e-9 {
        invalid.push(format!(
            "reach: {reach_max} cells of the {inscribed} to the wall ({missing} of the band's layers \
             without a far field); the plume needs a box at least {needed:.3} m wide"
        ));
    }
    let cfg = SmokeConfig::air();
    let window_s = WINDOW_S as f64;
    let grad = if pa.len() >= 2 { slope(&pa) } else { f64::NAN };
    let n_t = (cfg.gravity_m_s2.to_f64() * cfg.beta_per_k.to_f64() * grad)
        .max(0.0)
        .sqrt()
        * window_s;
    if n_t.is_nan() || n_t >= MAX_N_T {
        invalid.push(format!(
            "far field stratified: N·t {n_t:.3} (at most {MAX_N_T})"
        ));
    }
    let out = (plume.outflow().heat_left - win.out0) as f64;
    let added = (plume.ledger().heat_added - win.added0) as f64;
    let steady = out / added.max(1.0);
    if !open {
        invalid.push("outflow: the top is closed, so the box fills".into());
    } else if steady.is_nan() || (steady - 1.0).abs() > STEADY_TOLERANCE {
        invalid.push(format!(
            "outflow not steady: out / added {steady:.4} over the window (within {STEADY_TOLERANCE})"
        ));
    }
    println!(
        "SCENE reach_max_cells {reach_max} of {inscribed} needed_width_m {needed:.3} far_field_dT_dz_K_per_m {grad:.4} \
         N_t_window {n_t:.3} outflow_over_added {steady:.4} {}",
        if invalid.is_empty() { "valid" } else { "INVALID" }
    );
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the time per velocity step that this run prints"
    )]
    let ms_per_step = clock.elapsed().as_secs_f64() * 1e3 / (n - first).max(1) as f64;
    if pr.len() >= 3 {
        println!(
            "RESULT dx_mm {dx_mm} nu_um2 {nu} heat {heat} top {} band {z_lo}..{z_hi} layers {} \
             min_radius_cells {min_cells:.2} exp_R2 {:.4} exp_w {:.4} exp_T {:.4} (laminar 1 0 -1) \
             ms_per_step {ms_per_step:.1}",
            if open { "open" } else { "closed" },
            pr.len(),
            slope(&pr),
            slope(&pw),
            slope(&pt),
        );
    }
    if !invalid.is_empty() {
        return Err(format!("scene invalid: {}", invalid.join("; ")).into());
    }
    Ok(())
}
