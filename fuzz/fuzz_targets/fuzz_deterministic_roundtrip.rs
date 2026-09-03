//! Fuzz target: 128-bit fixed-point scene serialize → deserialize が bit-exact 保証されることを検証
//!
//! 決定論 lockstep が physics engine の core value なので、serialize/deserialize 経路で
//! 1 bit でもズレたら panic / 状態不整合 / lockstep desync に直結
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 罠 catalog #10 準拠
//! (scene 内部フォーマット直接 fuzz で panic ゼロ + roundtrip bit-exact 保証)
//!
//! 起こり得る危険:
//! - malformed bytes で load_scene が Vec 過大 allocation → panic
//! - fixed-point i64 pair (hi, lo) の byte order 誤解でズレ
//! - version 番号無検証で future format 読込 → panic

#![no_main]

// scene_io::PhysicsConfig は runtime の PhysicsConfig とは別 struct
// (scene 用: substeps + gravity 等の raw i64 配列版)
use alice_physics::scene_io::{
    load_scene, save_scene, PhysicsConfig as SceneConfig, PhysicsScene, SerializedBody,
    SerializedJoint,
};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

#[derive(Arbitrary, Debug)]
enum Op {
    /// 任意 byte 列を load_scene に食わせて panic しないことを検証
    RawBytes(Vec<u8>),
    /// 有効な scene を構築 → save → load して roundtrip 一致を検証
    Roundtrip {
        body_count: u8,
        position_seeds: Vec<(i16, i16, i16)>,
        version: u32,
    },
}

fuzz_target!(|op: Op| {
    let path = std::env::temp_dir().join(format!(
        "alice_physics_fuzz_scene_{}.bin",
        std::process::id()
    ));

    match op {
        Op::RawBytes(bytes) => {
            let Ok(mut file) = std::fs::File::create(&path) else {
                return;
            };
            if file.write_all(&bytes).is_err() {
                let _ = std::fs::remove_file(&path);
                return;
            }
            drop(file);

            let _ = load_scene(&path);
        }

        Op::Roundtrip {
            body_count,
            position_seeds,
            version,
        } => {
            let count = (body_count as usize).min(16);
            let bodies: Vec<SerializedBody> = (0..count)
                .map(|i| {
                    let (px, py, pz) = position_seeds.get(i).copied().unwrap_or((0, 0, 0));
                    SerializedBody {
                        // position: [x.hi, x.lo, y.hi, y.lo, z.hi, z.lo]
                        position: [px as i64, 0, py as i64, 0, pz as i64, 0],
                        velocity: [0; 6],
                        // rotation quaternion (identity: w=1)
                        rotation: [0, 0, 0, 0, 0, 0, 1, 0],
                        mass: [1, 0],
                        body_type: 0,  // Dynamic
                    }
                })
                .collect();

            let scene = PhysicsScene {
                bodies,
                joints: Vec::<SerializedJoint>::new(),
                config: SceneConfig::default(),
                version,
            };

            let Ok(()) = save_scene(&scene, &path) else {
                let _ = std::fs::remove_file(&path);
                return;
            };
            let Ok(loaded) = load_scene(&path) else {
                let _ = std::fs::remove_file(&path);
                return;
            };

            // bit-exact 検証 (決定論 lockstep の要)
            assert_eq!(
                scene, loaded,
                "roundtrip drift detected — determinism lockstep broken"
            );
        }
    }

    let _ = std::fs::remove_file(&path);
});
