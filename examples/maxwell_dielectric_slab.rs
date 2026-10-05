//! A pulse hitting a dielectric slab on the Yee lattice, printed next to the
//! Fresnel coefficient it should approach.
//!
//! The lattice is a 2-D TM waveguide (`nz = 1`, only `Ez`, `Hx`, `Hy` move)
//! excited in its lowest mode `Ez ∝ sin(πj/NY)`. A PEC box has no TEM mode, so
//! this is the closest the lattice gets to a plane wave; with `NY = 16` the
//! cut-off frequency (≈ 0.2) is well below the pulse band (around 0.6).
//!
//! The slab is `ε_r = 4` (`n = 2`) and 30 cells thick. The reflected wave is
//! separated from the incident one by running the same source twice — once
//! through vacuum only, once with the slab — and subtracting at a probe in
//! front of the slab. Normal-incidence Fresnel gives `r = (1 − 2)/(1 + 2) =
//! −1/3` at the front face and `(2 − 1)/(2 + 1) = +1/3` at the back face, so
//! the two reflected pulses have opposite signs.
//!
//! ⚠️ The printed peak ratio is **not** expected to be `1/3`. At this
//! resolution the wave has about 6 cells per wavelength inside the slab, and
//! the lattice's own interface correction is second order in `kΔx` — about 13%
//! here (the `analytic_fdtd_materials` test derives the exact discrete value
//! and shows it converging to Fresnel). Refining the pulse toward longer
//! wavelengths moves the ratio toward `1/3`.
//!
//! ```bash
//! cargo run --example maxwell_dielectric_slab --features std
//! ```

use alice_physics::det_math::exp64;
use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    Component, Material, MaterialError, MaterialMap, YeeGrid, COURANT_3D,
};

const TAG: &str = "[maxwell_fdtd materials]";
const NX: usize = 240;
const NY: usize = 16;
const SOURCE: usize = 10;
const PROBE: usize = 70;
const FRONT: usize = 120;
const BACK: usize = 150;
const STEPS: usize = 700;

/// `exp(−((t − 4τ)/τ)²)·sin(ω₀(t − 4τ))`, `τ = 12`, `ω₀ = 0.6`, `t = n·S`.
fn pulse(n: usize) -> Fix128 {
    let s = COURANT_3D.to_f64();
    let t = n as f64 * s - 48.0;
    let envelope = exp64(-(t / 12.0) * (t / 12.0));
    let (sin, _) = Fix128::from_f64(0.6 * t).sin_cos();
    Fix128::from_f64(envelope) * sin
}

fn run(map: &MaterialMap) -> Result<Vec<f64>, MaterialError> {
    let mut grid = YeeGrid::new(NX, NY, 1, COURANT_3D).with_materials(map)?;
    let profile: Vec<Fix128> = (0..=NY)
        .map(|j| {
            let (sin, _) = (Fix128::from_ratio(j as i64, NY as i64) * Fix128::PI).sin_cos();
            sin
        })
        .collect();
    let mut probe = Vec::with_capacity(STEPS);
    for n in 0..STEPS {
        let a = pulse(n);
        for (j, &shape) in profile.iter().enumerate().take(NY).skip(1) {
            grid.set_current(Component::Ez, SOURCE, j, 0, a * shape);
        }
        grid.step();
        probe.push(grid.get(Component::Ez, PROBE, NY / 2, 0).to_f64());
    }
    if map.get(FRONT, 0, 0) != Material::VACUUM {
        let front = grid.effective_material(Component::Ez, FRONT, NY / 2, 0);
        let inside = grid.effective_material(Component::Ez, FRONT + 1, NY / 2, 0);
        println!(
            "{TAG} Ez on the front face sees eps = {} (arithmetic mean of 1 and 4), one cell in sees {}",
            front.eps_r.to_f64(),
            inside.eps_r.to_f64()
        );
    }
    Ok(probe)
}

fn main() {
    let (nx, ny, nz) = (NX, NY, 1);
    let vacuum = MaterialMap::vacuum(nx, ny, nz);
    let mut slab = MaterialMap::vacuum(nx, ny, nz);
    let glass = Material::dielectric(Fix128::from_int(4));
    slab.fill([FRONT, 0, 0], [BACK, NY, 1], glass);
    assert_eq!(slab.dims(), (nx, ny, nz));
    println!(
        "{TAG} slab cells {FRONT}..{BACK}, eps_r = {}, n = {}",
        glass.eps_r.to_f64(),
        glass.refractive_index().to_f64()
    );

    let incident = run(&vacuum).expect("a vacuum map is always accepted");
    let with_slab = run(&slab).expect("eps_r = 4 satisfies the Courant bound");
    let reflected: Vec<f64> = with_slab
        .iter()
        .zip(&incident)
        .map(|(a, b)| a - b)
        .collect();

    let peak = |x: &[f64]| {
        x.iter()
            .enumerate()
            .fold((0usize, 0.0_f64), |(bi, bv), (i, &v)| {
                if v.abs() > bv.abs() {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
    };
    let (ni, vi) = peak(&incident);
    // the front-face reflection arrives first; the back-face one 2·30·n later
    let split = ni + ((2 * (FRONT - PROBE) + 60) as f64 / COURANT_3D.to_f64()) as usize;
    let (nf, vf) = peak(&reflected[..split.min(STEPS)]);
    let (nb, vb) = peak(&reflected[split.min(STEPS)..]);
    println!("{TAG} incident peak {vi:+.5} at step {ni}");
    println!(
        "{TAG} front reflection {vf:+.5} at step {nf}: ratio {:+.4} (Fresnel -1/3 = -0.3333, lattice-corrected about -0.29)",
        vf / vi.abs() * vi.signum()
    );
    println!(
        "{TAG} back reflection {vb:+.5} at step {}: sign opposite to the front one: {}",
        nb + split,
        vf.signum() != vb.signum()
    );

    // invalid materials are refused up front, not discovered as a blow-up
    let mut bad = MaterialMap::vacuum(4, 4, 4);
    bad.set(
        1,
        1,
        1,
        Material::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
    );
    match YeeGrid::new(4, 4, 4, COURANT_3D).with_materials(&bad) {
        Err(e) => println!("{TAG} rejected: {e}"),
        Ok(_) => println!("{TAG} unexpectedly accepted a zero permittivity"),
    }
    let mut slow = MaterialMap::vacuum(4, 4, 4);
    slow.set(
        2,
        2,
        2,
        Material::new(Fix128::from_ratio(1, 2), Fix128::ONE, Fix128::ZERO),
    );
    if let Err(e) = YeeGrid::new(4, 4, 4, COURANT_3D).with_materials(&slow) {
        println!("{TAG} rejected: {e}");
    }

    // Gauss's law with materials: div(eps E) − rho stays at the truncation level
    let mut cube = MaterialMap::vacuum(6, 6, 6);
    cube.fill([3, 0, 0], [6, 6, 6], glass);
    let mut g = YeeGrid::new(6, 6, 6, COURANT_3D)
        .with_materials(&cube)
        .expect("valid map");
    g.set_current(Component::Ex, 2, 3, 3, Fix128::from_ratio(1, 3));
    for _ in 0..50 {
        g.step();
    }
    println!(
        "{TAG} after 50 steps of a current at the interface: max |div(eps E) - rho| = {:.2e}, div D at (3,3,3) = {:+.4}, rho there = {:+.4}",
        g.max_abs_gauss_residual().to_f64(),
        g.div_d(3, 3, 3).to_f64(),
        g.charge(3, 3, 3).to_f64()
    );
}
