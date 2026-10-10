// Generator of the world snapshot fixtures. Not built by `cargo test`: it is
// copied into `examples/` of the checkout whose writer is to be frozen, next
// to `world_snapshot_scenes.rs`, and run there (see `README.md`).
//
// For every scene of `world_snapshot_scenes.rs` it writes
// `world_snapshot_v{V}_{scene}.bin`, where V is the version the checkout's
// writer emits; for V below 5 it also writes `world_snapshot_v{V}_{scene}_later.bin`,
// the same world stepped `FIXTURE_LATER_STEPS` more times by that checkout.

include!("world_snapshot_scenes.rs");

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: world_snapshot_gen <output directory>");
    let out = std::path::Path::new(&out);
    for (name, mut w) in fixture_scenes() {
        let blob = w.snapshot_world();
        let version = u16::from_le_bytes([blob[4], blob[5]]);
        let path = out.join(format!("world_snapshot_v{version}_{name}.bin"));
        std::fs::write(&path, &blob).expect("write fixture");
        println!("{} {} bytes", path.display(), blob.len());
        if version < 5 {
            w.step_n(FIXTURE_LATER_STEPS, fixture_dt());
            let later = w.snapshot_world();
            let path = out.join(format!("world_snapshot_v{version}_{name}_later.bin"));
            std::fs::write(&path, &later).expect("write fixture");
            println!("{} {} bytes", path.display(), later.len());
        }
    }
}
