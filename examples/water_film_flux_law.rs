//! The water film's integer flux law, face by face: for a few pairs of neighbouring cells
//! (bed, depth in 2⁻³² m quanta), the depth over the higher bed, the join step, and the water
//! a substep moves by the film law (lubrication capped by critical) and by the free-overfall
//! weir, at a 1 cm cell and a 1/256 s substep.
//!
//! Run: `cargo run --example water_film_flux_law`

use alice_physics_core::water_film::{
    face_depth, isqrt_u128, join_step, metres_to_quanta, simultaneous_transfer,
    simultaneous_transfer_report, weir_transfer, FluxConstants, MAX_FACE_DEPTH,
    MAX_LUBRICATION_DIVISOR,
};
use alice_physics_core::Fix128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mm = |v: i64| metres_to_quanta(Fix128::from_ratio(v, 1000));
    let k = FluxConstants::new(
        Fix128::from_ratio(1, 100),
        Fix128::from_ratio(1, 256),
        Fix128::from_ratio(980_665, 100_000),
        Fix128::from_ratio(1, 1_000_000),
    )
    .ok_or("constants")?;
    println!(
        "D_l = {} (limit {MAX_LUBRICATION_DIVISOR}), N_c = {}, N_w = {}, deepest face {MAX_FACE_DEPTH} quanta",
        k.lubrication, k.critical, k.weir
    );
    // (bed, depth) on each side of a face
    let faces = [
        (
            "a film on a 1/200 slope",
            (mm(0), mm(1) / 2),
            (-mm(1) / 20, mm(1) / 2),
        ),
        (
            "a pool beside a bump of 0.1 mm",
            (mm(-30), mm(20)),
            (mm(-30), mm(20) + mm(1) / 10),
        ),
        (
            "a pool over a sill, 15 mm head",
            (mm(-10), mm(15)),
            (mm(-15), 0),
        ),
    ];
    for (name, a, b) in faces {
        let hf = face_depth(a.0, a.1, b.0, b.1);
        let h3 = u128::from(hf.unsigned_abs()).pow(3);
        let film = simultaneous_transfer(a, b, k).ok_or("film law")?;
        let (_, capped) = simultaneous_transfer_report(a, b, k).ok_or("report")?;
        let weir = weir_transfer(a, b, k).ok_or("weir law")?;
        println!(
            "{name}: h_f {hf} quanta (⌊√h_f³⌋ = {}), join step {}, film {film} quanta{}, weir {weir} quanta",
            isqrt_u128(h3),
            join_step(Fix128::from_ratio(1, 100), a.1, b.1),
            if capped { " (held by a cap)" } else { "" },
        );
    }
    Ok(())
}
