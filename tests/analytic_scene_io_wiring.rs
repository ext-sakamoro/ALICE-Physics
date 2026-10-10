//! Oracles for `scene_io::{CURRENT_SCENE_VERSION, save_scene_json, load_scene_json}`
//! (`examples/scene_snapshot_roundtrip.rs`) and the binary / JSON formats around them.
//!
//! * the JSON text of a tiny scene is spelled out by hand below (layout: version, config, bodies, joints)
//! * the binary layout is rebuilt byte by byte (magic `APHYS\0`, little-endian `u32` / `i64`)
//! * a JSON value that is present but not a `u32` / `u8` is `Err(InvalidData)`; an absent key takes
//!   its documented default (`version` 1, `substeps` 8, `iterations` 4, everything else 0)
//! * every strict prefix of a valid binary file is an `Err`, never a panic and never `Ok`
//! * a version outside `SUPPORTED_SCENE_VERSIONS` (only 1) is refused by both loaders with
//!   `InvalidData` carrying `UnsupportedSceneVersion { found }`; the committed fixtures
//!   `tests/fixtures/scene_v{1,2}.{aphys,json}` (written byte by byte outside the crate) pin one
//!   readable and one refused file per format
//! * the JSON version is the top-level object's `version` member only, read after the whole
//!   document is checked against the JSON grammar: a nested `version` is not the scene version,
//!   a duplicated top-level member, a non-JSON number (`+1`, `01`) and a value that is not a
//!   `u32` written as a plain integer are `InvalidData` carrying `InvalidSceneJsonVersion`
//! * every other field is read from its own object's members: a key that appears only inside a
//!   nested value is not the member, a key spelled with escapes is the same key, and a repeated
//!   key in any object is `InvalidData` carrying `InvalidSceneJson::DuplicateKey`
//! * a member the format does not define, at any object of the scene (top level, `config`,
//!   `bodies[i]`, `joints[i]`), is `InvalidData` carrying `InvalidSceneJson::UnknownMember` with
//!   its path and, within edit distance 2 (case-insensitive), the intended key; the checks run in
//!   the order grammar / depth, version, repeated keys, unknown members, field values
#![allow(clippy::disallowed_methods)]

use alice_physics::scene_io::{
    load_scene, load_scene_json, save_scene, save_scene_json, InvalidSceneJson,
    InvalidSceneJsonVersion, PhysicsConfig, PhysicsScene, SerializedBody, SerializedJoint,
    UnsupportedSceneVersion, CURRENT_SCENE_VERSION, SUPPORTED_SCENE_VERSIONS,
};
use std::io::ErrorKind;
use std::path::PathBuf;

/// The version carried by an `UnsupportedSceneVersion` refusal, `None` for any other error.
fn refused_version(e: &std::io::Error) -> Option<u32> {
    e.get_ref()
        .and_then(|i| i.downcast_ref::<UnsupportedSceneVersion>())
        .map(|u| u.found)
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("w4_scene_io_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}
fn body(seed: i64, ty: u8) -> SerializedBody {
    SerializedBody {
        position: [seed, -seed, i64::MAX, i64::MIN, 0, seed * 3],
        velocity: [1, 2, 3, 4, 5, 6],
        rotation: [0, 0, 0, 0, 0, 0, 1, seed],
        mass: [seed + 1, -1],
        body_type: ty,
    }
}
fn joint(a: u32, b: u32, ty: u8) -> SerializedJoint {
    SerializedJoint {
        body_a: a,
        body_b: b,
        joint_type: ty,
        anchor_a: [1, 2, 3, 4, 5, 6],
        anchor_b: [-1, -2, -3, -4, -5, -6],
    }
}
fn config() -> PhysicsConfig {
    PhysicsConfig::new(2, 3, [10, 20, 30, 40, 50, 60], [7, 8])
}
fn scene() -> PhysicsScene {
    PhysicsScene::new(
        vec![body(1, 0), body(2, 1)],
        vec![joint(0, 1, 4)],
        config(),
        CURRENT_SCENE_VERSION,
    )
}
fn json_of(body_a: &str, joint_type: &str) -> String {
    format!(
        "{{\"version\": 1, \"config\": {{\"substeps\": 2, \"iterations\": 3, \"gravity\": [0,0,0,0,0,0], \"damping\": [1, 0]}}, \"bodies\": [], \"joints\": [{{\"body_a\": {body_a}, \"body_b\": 1, \"joint_type\": {joint_type}, \"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,0]}}]}}"
    )
}
fn load_text(name: &str, text: &str) -> std::io::Result<PhysicsScene> {
    let p = tmp(name);
    std::fs::write(&p, text).unwrap();
    let r = load_scene_json(&p);
    std::fs::remove_file(&p).ok();
    r
}

#[test]
fn current_version_is_one_and_is_what_the_file_records() {
    assert_eq!(CURRENT_SCENE_VERSION, 1);
    let p = tmp("ver.json");
    save_scene_json(
        &PhysicsScene::new(vec![], vec![], config(), CURRENT_SCENE_VERSION),
        &p,
    )
    .unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("\"version\": 1,"));
    assert_eq!(load_scene_json(&p).unwrap().version, 1);
    // an older / newer version is written as given but refused on reading (no other layout exists)
    for v in [0u32, 7, u32::MAX] {
        save_scene_json(&PhysicsScene::new(vec![], vec![], config(), v), &p).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains(&format!("\"version\": {v},")));
        let e = load_scene_json(&p).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert_eq!(refused_version(&e), Some(v));
    }
    std::fs::remove_file(&p).ok();
}

#[test]
fn json_text_matches_the_hand_written_layout() {
    let p = tmp("layout.json");
    let s = PhysicsScene::new(vec![], vec![], config(), CURRENT_SCENE_VERSION);
    save_scene_json(&s, &p).unwrap();
    let want = "{\n  \"version\": 1,\n  \"config\": {\n    \"substeps\": 2,\n    \"iterations\": 3,\n    \"gravity\": [10, 20, 30, 40, 50, 60],\n    \"damping\": [7, 8]\n  },\n  \"bodies\": [\n  ],\n  \"joints\": [\n  ]\n}\n";
    assert_eq!(std::fs::read_to_string(&p).unwrap(), want);
    let one = PhysicsScene::new(vec![body(1, 2)], vec![joint(3, 4, 1)], config(), 1);
    save_scene_json(&one, &p).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains(
        "      \"position\": [1, -1, 9223372036854775807, -9223372036854775808, 0, 3],\n"
    ));
    assert!(text.contains("      \"rotation\": [0, 0, 0, 0, 0, 0, 1, 1],\n"));
    assert!(text.contains("      \"mass\": [2, -1],\n      \"body_type\": 2\n    }\n  ],"));
    assert!(text.contains("      \"body_a\": 3,\n      \"body_b\": 4,\n      \"joint_type\": 1,\n"));
    std::fs::remove_file(&p).ok();
}

#[test]
fn json_round_trip_is_exact_including_extreme_limbs() {
    let p = tmp("rt.json");
    let mut bodies = Vec::new();
    for i in 0..40i64 {
        bodies.push(body(i - 20, (i % 3) as u8));
    }
    let s = PhysicsScene::new(
        bodies,
        vec![joint(0, 39, 0), joint(u32::MAX, 0, 255), joint(5, 6, 2)],
        config(),
        1,
    );
    save_scene_json(&s, &p).unwrap();
    assert_eq!(load_scene_json(&p).unwrap(), s);
    // empty scene
    let e = PhysicsScene::new(vec![], vec![], PhysicsConfig::default(), 1);
    save_scene_json(&e, &p).unwrap();
    assert_eq!(load_scene_json(&p).unwrap(), e);
    std::fs::remove_file(&p).ok();
}

#[test]
fn valid_values_at_the_boundaries_load() {
    for (a, want) in [
        ("0", 0u32),
        ("1", 1),
        ("4294967295", u32::MAX),
        ("  7  ", 7),
    ] {
        let s = load_text("ok.json", &json_of(a, "0")).unwrap();
        assert_eq!(s.joints[0].body_a, want, "body_a = {a:?}");
    }
    for (t, want) in [("0", 0u8), ("1", 1), ("255", 255)] {
        let s = load_text("ok2.json", &json_of("0", t)).unwrap();
        assert_eq!(s.joints[0].joint_type, want);
    }
}

#[test]
fn present_but_invalid_values_are_errors_not_defaults() {
    for bad in [
        "-1",
        "4294967296",
        "1.5",
        "\"abc\"",
        "5abc",
        "",
        "0x10",
        "99999999999999999999",
    ] {
        let e = load_text("bad.json", &json_of(bad, "0")).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData, "body_a = {bad:?}");
    }
    for bad in ["256", "257", "-1", "65536", "x", "2.0"] {
        let e = load_text("bad2.json", &json_of("0", bad)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData, "joint_type = {bad:?}");
    }
    // the same rule for the header fields
    for key in ["version", "substeps", "iterations"] {
        let good = json_of("0", "0");
        let bad = good.replacen(
            &format!("\"{key}\": "),
            &format!("\"{key}\": -3, \"x\": "),
            1,
        );
        assert_eq!(
            load_text("bad3.json", &bad).unwrap_err().kind(),
            ErrorKind::InvalidData,
            "{key}"
        );
    }
    let bad_type = json_of("0", "0").replace("\"bodies\": []", "\"bodies\": [{\"position\":[0,0,0,0,0,0],\"velocity\":[0,0,0,0,0,0],\"rotation\":[0,0,0,0,0,0,0,0],\"mass\":[0,0],\"body_type\": 300}]");
    assert_eq!(
        load_text("bad4.json", &bad_type).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn absent_keys_take_the_documented_defaults() {
    // no version, substeps, iterations; body without body_type; joint without body_a / body_b / joint_type
    let text = "{\"config\": {\"gravity\": [1,2,3,4,5,6], \"damping\": [9, 9]}, \"bodies\": [{\"position\":[0,0,0,0,0,0],\"velocity\":[0,0,0,0,0,0],\"rotation\":[0,0,0,0,0,0,0,0],\"mass\":[1,0]}], \"joints\": [{\"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,1]}]}";
    let s = load_text("def.json", text).unwrap();
    assert_eq!(s.version, 1);
    assert_eq!((s.config.substeps, s.config.iterations), (8, 4));
    assert_eq!(s.config.gravity, [1, 2, 3, 4, 5, 6]);
    assert_eq!(s.bodies[0].body_type, 0);
    assert_eq!(
        (
            s.joints[0].body_a,
            s.joints[0].body_b,
            s.joints[0].joint_type
        ),
        (0, 0, 0)
    );
    assert_eq!(s.joints[0].anchor_b, [0, 0, 0, 0, 0, 1]);
    // missing bodies / joints arrays are empty lists; a missing config array is an error
    let min = "{\"config\": {\"gravity\": [1,2,3,4,5,6], \"damping\": [9, 9]}}";
    let s = load_text("def2.json", min).unwrap();
    assert!(s.bodies.is_empty() && s.joints.is_empty());
    assert_eq!(
        load_text("def3.json", "{\"config\": {\"damping\": [9, 9]}}")
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        load_text("def4.json", "{}").unwrap_err().kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn malformed_documents_are_errors() {
    for text in [
        "",
        "[]",
        "not json",
        "{\"config\": {\"gravity\": [1,2,3], \"damping\": [1,2]}}",
        "{\"config\": {\"gravity\": [1,2,3,4,5,x], \"damping\": [1,2]}}",
    ] {
        assert_eq!(
            load_text("mal.json", text).unwrap_err().kind(),
            ErrorKind::InvalidData,
            "{text:?}"
        );
    }
    assert_eq!(
        load_scene_json(&tmp("does-not-exist.json"))
            .unwrap_err()
            .kind(),
        ErrorKind::NotFound
    );
}

fn le32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn le64s(a: &[i64]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn binary_layout_is_byte_exact() {
    let s = scene();
    let p = tmp("layout.aphys");
    save_scene(&s, &p).unwrap();
    let got = std::fs::read(&p).unwrap();
    let mut want = b"APHYS\0".to_vec();
    want.extend(le32(1)); // version
    want.extend(le32(2)); // bodies
    want.extend(le32(1)); // joints
    want.extend(le32(2)); // substeps
    want.extend(le32(3)); // iterations
    want.extend(le64s(&[10, 20, 30, 40, 50, 60]));
    want.extend(le64s(&[7, 8]));
    for b in &s.bodies {
        want.extend(le64s(&b.position));
        want.extend(le64s(&b.velocity));
        want.extend(le64s(&b.rotation));
        want.extend(le64s(&b.mass));
        want.push(b.body_type);
    }
    for j in &s.joints {
        want.extend(le32(j.body_a));
        want.extend(le32(j.body_b));
        want.push(j.joint_type);
        want.extend(le64s(&j.anchor_a));
        want.extend(le64s(&j.anchor_b));
    }
    assert_eq!(got, want);
    assert_eq!(load_scene(&p).unwrap(), s);
    std::fs::remove_file(&p).ok();
}

#[test]
fn every_strict_prefix_of_a_binary_file_is_an_error() {
    let p = tmp("full.aphys");
    save_scene(&scene(), &p).unwrap();
    let bytes = std::fs::read(&p).unwrap();
    let cut = tmp("cut.aphys");
    for n in 0..bytes.len() {
        std::fs::write(&cut, &bytes[..n]).unwrap();
        let e = load_scene(&cut).expect_err(&format!("prefix of {n} bytes must fail"));
        assert!(
            matches!(e.kind(), ErrorKind::UnexpectedEof | ErrorKind::InvalidData),
            "{n}: {e:?}"
        );
    }
    std::fs::remove_file(&p).ok();
    std::fs::remove_file(&cut).ok();
}

#[test]
fn a_header_claiming_four_billion_entries_ends_in_an_error() {
    // body count 0xFFFFFFFF with a few bytes of payload; also joints
    for (nb, nj) in [(u32::MAX, 0u32), (0, u32::MAX), (u32::MAX, u32::MAX)] {
        let mut bytes = b"APHYS\0".to_vec();
        bytes.extend(le32(1));
        bytes.extend(le32(nb));
        bytes.extend(le32(nj));
        bytes.extend(le32(8));
        bytes.extend(le32(4));
        bytes.extend(le64s(&[0; 8]));
        let p = tmp("huge.aphys");
        std::fs::write(&p, bytes).unwrap();
        let e = load_scene(&p).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::UnexpectedEof);
        std::fs::remove_file(&p).ok();
    }
}

#[test]
fn files_with_more_entries_than_the_preallocation_hint_still_load() {
    let bodies: Vec<_> = (0..10_000i64).map(|i| body(i, (i % 3) as u8)).collect();
    let joints: Vec<_> = (0..5_000u32)
        .map(|i| joint(i, i + 1, (i % 5) as u8))
        .collect();
    let s = PhysicsScene::new(bodies, joints, config(), 1);
    let p = tmp("big.aphys");
    save_scene(&s, &p).unwrap();
    let back = load_scene(&p).unwrap();
    assert_eq!(back.bodies.len(), 10_000);
    assert_eq!(back.joints.len(), 5_000);
    assert_eq!(back, s);
    let pj = tmp("big.json");
    save_scene_json(&s, &pj).unwrap();
    assert_eq!(load_scene_json(&pj).unwrap(), s);
    std::fs::remove_file(&p).ok();
    std::fs::remove_file(&pj).ok();
}

#[test]
fn binary_rejects_bad_magic_and_missing_files() {
    let p = tmp("badmagic.aphys");
    std::fs::write(&p, b"NOPE!!\x01\0\0\0").unwrap();
    assert_eq!(load_scene(&p).unwrap_err().kind(), ErrorKind::InvalidData);
    std::fs::remove_file(&p).ok();
    assert_eq!(
        load_scene(&tmp("nope.aphys")).unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[test]
fn binary_reads_version_one_and_refuses_every_other_version() {
    assert_eq!(SUPPORTED_SCENE_VERSIONS, &[CURRENT_SCENE_VERSION]);
    let p = tmp("ver.aphys");
    for v in [0u32, 1, 2, 0xDEAD_BEEF] {
        let s = PhysicsScene::new(vec![], vec![], config(), v);
        save_scene(&s, &p).unwrap();
        // the writer stores the version as given (bytes 6..10, little-endian)
        assert_eq!(std::fs::read(&p).unwrap()[6..10], v.to_le_bytes());
        if v == 1 {
            assert_eq!(load_scene(&p).unwrap(), s);
        } else {
            let e = load_scene(&p).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidData);
            assert_eq!(refused_version(&e), Some(v));
        }
    }
    std::fs::remove_file(&p).ok();
}

/// The scene the committed fixtures `tests/fixtures/scene_v{1,2}.{aphys,json}` hold (the
/// version-2 files differ from the version-1 files only in the version).
fn fixture_scene() -> PhysicsScene {
    PhysicsScene::new(
        vec![SerializedBody {
            position: [1, 2, 3, 4, 5, 6],
            velocity: [-1, 0, 0, 0, 0, 7],
            rotation: [0, 0, 0, 0, 0, 0, 1, 0],
            mass: [2, 0],
            body_type: 1,
        }],
        vec![SerializedJoint {
            body_a: 0,
            body_b: 0,
            joint_type: 3,
            anchor_a: [1, 0, 0, 0, 0, 0],
            anchor_b: [0, 0, -1, 0, 0, 0],
        }],
        PhysicsConfig::new(4, 6, [0, 0, -10, 0, 0, 0], [0, -2]),
        1,
    )
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn committed_version_one_fixtures_load() {
    assert_eq!(
        load_scene(&fixture("scene_v1.aphys")).unwrap(),
        fixture_scene()
    );
    assert_eq!(
        load_scene_json(&fixture("scene_v1.json")).unwrap(),
        fixture_scene()
    );
    // the binary fixture is exactly what the writer produces for that scene
    let p = tmp("fixture.aphys");
    save_scene(&fixture_scene(), &p).unwrap();
    assert_eq!(
        std::fs::read(&p).unwrap(),
        std::fs::read(fixture("scene_v1.aphys")).unwrap()
    );
    std::fs::remove_file(&p).ok();
}

#[test]
fn committed_version_two_fixtures_are_refused() {
    let bin = std::fs::read(fixture("scene_v2.aphys")).unwrap();
    let one = std::fs::read(fixture("scene_v1.aphys")).unwrap();
    // only the version field differs from the readable fixture
    assert_eq!(bin.len(), one.len());
    let differ: Vec<usize> = (0..bin.len()).filter(|&i| bin[i] != one[i]).collect();
    assert_eq!(differ, vec![6]);
    assert_eq!(bin[6..10], 2u32.to_le_bytes());
    for e in [
        load_scene(&fixture("scene_v2.aphys")).unwrap_err(),
        load_scene_json(&fixture("scene_v2.json")).unwrap_err(),
    ] {
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert_eq!(refused_version(&e), Some(2));
        assert!(e.to_string().contains("unsupported scene version 2"), "{e}");
    }
}

#[test]
fn compact_and_truncated_documents() {
    // a value directly followed by the closing brace (no space, comma or newline)
    let compact = "{\"config\":{\"gravity\":[1,2,3,4,5,6],\"damping\":[1,2]},\"bodies\":[{\"position\":[0,0,0,0,0,0],\"velocity\":[0,0,0,0,0,0],\"rotation\":[0,0,0,0,0,0,0,0],\"mass\":[1,0],\"body_type\":2}],\"joints\":[{\"anchor_a\":[0,0,0,0,0,0],\"anchor_b\":[0,0,0,0,0,0],\"body_b\":9}]}";
    let s = load_text("compact.json", compact).unwrap();
    assert_eq!(s.bodies[0].body_type, 2);
    assert_eq!(s.joints[0].body_b, 9);
    // a key name that only appears as a string value is not a key: the member is `note`, which
    // the format does not define (the version gate passed, so it is not read as a version)
    let stray = "{\"config\":{\"gravity\":[1,2,3,4,5,6],\"damping\":[1,2]},\"note\":\"version\"}";
    assert_eq!(
        field_error(&load_text("stray.json", stray).unwrap_err()),
        Some(unknown("note", None))
    );
    // truncated document (no closing brace) and over-long arrays are errors
    let cut = "{\"config\":{\"gravity\":[1,2,3,4,5,6],\"damping\":[1,2]},\"bodies\":[";
    assert_eq!(
        load_text("cut.json", cut).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    let long = "{\"config\":{\"gravity\":[1,2,3,4,5,6,7],\"damping\":[1,2]}}";
    assert_eq!(
        load_text("long.json", long).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
}

#[test]
#[allow(deprecated)]
fn deprecated_checked_loaders_behave_as_the_default_loaders() {
    use alice_physics::scene_io::{load_scene_checked, load_scene_json_checked};
    let p = tmp("checked.aphys");
    let pj = tmp("checked.json");
    save_scene(&scene(), &p).unwrap();
    save_scene_json(&scene(), &pj).unwrap();
    assert_eq!(load_scene_checked(&p).unwrap(), scene());
    assert_eq!(load_scene_json_checked(&pj).unwrap(), scene());
    for v in [0u32, 2, 0xDEAD_BEEF] {
        let s = PhysicsScene::new(vec![body(1, 0)], vec![], config(), v);
        save_scene(&s, &p).unwrap();
        save_scene_json(&s, &pj).unwrap();
        for e in [
            load_scene_checked(&p).unwrap_err(),
            load_scene_json_checked(&pj).unwrap_err(),
            load_scene(&p).unwrap_err(),
            load_scene_json(&pj).unwrap_err(),
        ] {
            assert_eq!(e.kind(), ErrorKind::InvalidData);
            assert_eq!(refused_version(&e), Some(v));
        }
    }
    std::fs::remove_file(&p).ok();
    std::fs::remove_file(&pj).ok();
}

/// The `InvalidSceneJsonVersion` carried by a refusal, `None` for any other error.
fn json_version_error(e: &std::io::Error) -> Option<InvalidSceneJsonVersion> {
    e.get_ref()
        .and_then(|i| i.downcast_ref::<InvalidSceneJsonVersion>())
        .cloned()
}

/// A loadable scene whose top level is `head` followed by the usual members.
fn with_head(head: &str) -> String {
    let body = json_of("0", "0");
    format!("{{{head}{}", &body[body.find("\"config\"").unwrap()..])
}

#[test]
fn json_version_is_read_from_the_top_level_member_only() {
    // valid version 1, with whitespace (including newlines) around the member
    let s = load_text("ws.json", &with_head(" \n\t\"version\"\r\n :\n 1 \n,")).unwrap();
    assert_eq!(s.version, 1);
    assert_eq!(s.config.substeps, 2);
    // `version` only inside a nested object: the top level has none, so the version gate sees
    // version 1 and passes; the nested member is then refused as unknown, not as version 2
    let nested = with_head("").replace("\"config\": {", "\"config\": {\"version\": 2, ");
    let e = load_text("nested.json", &nested).unwrap_err();
    assert_eq!(refused_version(&e), None);
    assert_eq!(field_error(&e), Some(unknown("config.version", None)));
    let nested_body = with_head("").replace(
        "\"bodies\": []",
        "\"bodies\": [], \"meta\": {\"version\": 2}",
    );
    let e = load_text("nested2.json", &nested_body).unwrap_err();
    assert_eq!(refused_version(&e), None);
    assert_eq!(field_error(&e), Some(unknown("meta", None)));
    // a top-level 2 after a nested 1 is still refused as version 2
    let late = with_head("")
        .replace("\"joints\"", "\"version\": 2, \"joints\"")
        .replace("\"config\": {", "\"config\": {\"version\": 1, ");
    let e = load_text("late.json", &late).unwrap_err();
    assert_eq!(refused_version(&e), Some(2));
}

#[test]
fn json_version_values_that_are_not_a_plain_u32_are_refused() {
    use InvalidSceneJsonVersion as E;
    let probe =
        |value: &str| load_text("probe.json", &with_head(&format!("\"version\": {value}, ")));
    // (value, expected refusal)
    let cases: [(&str, E); 10] = [
        ("+1", E::MalformedJson { offset: 12 }),
        ("01", E::MalformedJson { offset: 13 }),
        (
            "1e0",
            E::NotAnUnsignedInteger {
                value: "1e0".into(),
            },
        ),
        (
            "1.0",
            E::NotAnUnsignedInteger {
                value: "1.0".into(),
            },
        ),
        (
            "\"1\"",
            E::NotAnUnsignedInteger {
                value: "\"1\"".into(),
            },
        ),
        ("-1", E::NotAnUnsignedInteger { value: "-1".into() }),
        (
            "null",
            E::NotAnUnsignedInteger {
                value: "null".into(),
            },
        ),
        (
            "4294967296",
            E::OutOfRange {
                value: "4294967296".into(),
            },
        ),
        (
            "99999999999999999999",
            E::OutOfRange {
                value: "99999999999999999999".into(),
            },
        ),
        ("1 1", E::MalformedJson { offset: 14 }),
    ];
    for (value, want) in cases {
        let e = probe(value).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData, "{value}");
        assert_eq!(json_version_error(&e), Some(want), "{value}");
        assert_eq!(refused_version(&e), None, "{value}");
    }
    assert_eq!(probe("1").unwrap().version, 1);
    assert_eq!(
        refused_version(&probe("4294967295").unwrap_err()),
        Some(u32::MAX)
    );
}

#[test]
fn duplicate_top_level_version_members_are_refused_in_either_order() {
    for head in [
        "\"version\": 1, \"version\": 2, ",
        "\"version\": 2, \"version\": 1, ",
        "\"version\": 1, \"version\": 1, ",
    ] {
        let e = load_text("dup.json", &with_head(head)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData, "{head}");
        assert_eq!(
            json_version_error(&e),
            Some(InvalidSceneJsonVersion::DuplicateVersion),
            "{head}"
        );
    }
    // one before the other members and one after them
    let split = with_head("\"version\": 1, ").replace("\"joints\"", "\"version\": 2, \"joints\"");
    assert_eq!(
        json_version_error(&load_text("dup2.json", &split).unwrap_err()),
        Some(InvalidSceneJsonVersion::DuplicateVersion)
    );
}

/// `InvalidSceneJson::UnknownMember` at `path`.
fn unknown(path: &str, suggestion: Option<&'static str>) -> InvalidSceneJson {
    InvalidSceneJson::UnknownMember {
        path: path.into(),
        suggestion,
    }
}

/// The `InvalidSceneJson` carried by a refusal, `None` for any other error.
fn field_error(e: &std::io::Error) -> Option<InvalidSceneJson> {
    e.get_ref()
        .and_then(|i| i.downcast_ref::<InvalidSceneJson>())
        .cloned()
}

const CFG4: &str =
    "\"substeps\": 4, \"iterations\": 6, \"gravity\": [0,0,-10,0,0,0], \"damping\": [0,-2]";

#[test]
fn a_nested_key_is_not_the_member_being_read() {
    // `extra.substeps` comes first and is valid JSON; it is not read as the config's member:
    // the whole `extra` member is refused (nothing inside it is read)
    let text = format!(
        "{{\"config\": {{\"extra\": {{\"substeps\": 99, \"gravity\": [9,9,9,9,9,9]}}, {CFG4}}}}}"
    );
    let e = load_text("nested_member.json", &text).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert_eq!(field_error(&e), Some(unknown("config.extra", None)));
    // a body's nested object repeating `mass` is refused at the nested member's path
    let text = format!("{{\"config\": {{{CFG4}}}, \"bodies\": [{{\"meta\": {{\"mass\": [9,9], \"body_type\": 2}}, \"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [3,0]}}]}}");
    assert_eq!(
        field_error(&load_text("nested_body.json", &text).unwrap_err()),
        Some(unknown("bodies[0].meta", None))
    );
    // a config that exists only inside a body is not the scene config: the unknown member is
    // reported before the missing top-level `config`
    let text = format!("{{\"bodies\": [{{\"config\": {{{CFG4}}}}}]}}");
    let e = load_text("nested_config.json", &text).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert_eq!(field_error(&e), Some(unknown("bodies[0].config", None)));
    // the same body without the stray member loads, with its own `mass`
    let text = format!("{{\"config\": {{{CFG4}}}, \"bodies\": [{{\"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [3,0]}}]}}");
    let s = load_text("own_body.json", &text).unwrap();
    assert_eq!((s.bodies[0].mass, s.bodies[0].body_type), ([3, 0], 0));
}

#[test]
fn keys_spelled_with_escapes_are_the_same_key() {
    let text = format!("{{\"\\u0063onfig\": {{{CFG4}}}}}");
    assert_eq!(
        load_text("escaped_key.json", &text)
            .unwrap()
            .config
            .substeps,
        4
    );
    let text = format!("{{\"config\": {{{CFG4}, \"subst\\u0065ps\": 5}}}}");
    assert_eq!(
        field_error(&load_text("escaped_dup.json", &text).unwrap_err()),
        Some(InvalidSceneJson::DuplicateKey {
            path: "config.substeps".into()
        })
    );
}

#[test]
fn duplicate_keys_are_refused_at_every_level() {
    let cfg = format!("\"config\": {{{CFG4}}}");
    for (text, path) in [
        // two complete copies of the config
        (format!("{{{cfg}, {cfg}}}"), "config"),
        (format!("{{{cfg}, \"bodies\": [], \"bodies\": []}}"), "bodies"),
        (format!("{{{cfg}, \"joints\": [], \"joints\": []}}"), "joints"),
        (format!("{{\"config\": {{{CFG4}, \"substeps\": 4}}}}"), "config.substeps"),
        (
            format!("{{{cfg}, \"joints\": [{{\"body_a\": 1, \"body_a\": 2, \"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,0]}}]}}"),
            "joints[0].body_a",
        ),
        (format!("{{{cfg}, \"meta\": [{{\"k\": 1, \"k\": 1}}]}}"), "meta[0].k"),
    ] {
        let e = load_text("dup_any.json", &text).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData, "{text}");
        assert_eq!(
            field_error(&e),
            Some(InvalidSceneJson::DuplicateKey { path: path.into() }),
            "{text}"
        );
    }
}

#[test]
fn braces_inside_strings_do_not_split_the_body_list() {
    let body = |ty: u8, note: &str| {
        format!("{{\"note\": \"{note}\", \"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [1,0], \"body_type\": {ty}}}")
    };
    let plain = |ty: u8| {
        format!("{{\"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [1,0], \"body_type\": {ty}}}")
    };
    // the braces inside the note do not split the list: the note is found in body 2 (index 1)
    for note in ["}{", "]\\\"[{"] {
        let text = format!(
            "{{\"config\": {{{CFG4}}}, \"bodies\": [{}, {}, {}]}}",
            plain(1),
            body(2, note),
            plain(3)
        );
        assert_eq!(
            field_error(&load_text("strings.json", &text).unwrap_err()),
            Some(unknown("bodies[1].note", None)),
            "{text}"
        );
    }
    let text = format!(
        "{{\"config\": {{{CFG4}}}, \"bodies\": [{}, {}]}}",
        plain(1),
        plain(2)
    );
    let s = load_text("strings_ok.json", &text).unwrap();
    assert_eq!((s.bodies[0].body_type, s.bodies[1].body_type), (1, 2));
}

#[test]
fn huge_numbers_are_refused_with_a_bounded_message() {
    let big = format!("9{}", "9".repeat(999));
    let text = format!(
        "{{\"config\": {{{}}}}}",
        CFG4.replace("\"substeps\": 4", &format!("\"substeps\": {big}"))
    );
    let e = load_text("huge.json", &text).unwrap_err();
    assert_eq!(
        field_error(&e),
        Some(InvalidSceneJson::OutOfRange {
            path: "config.substeps".into(),
            value: big.clone()
        })
    );
    let shown = e.to_string();
    assert!(
        shown.len() < 120 && shown.contains("(1000 bytes)"),
        "{shown}"
    );
    let e = load_text("huge_version.json", &format!("{{\"version\": {big}}}")).unwrap_err();
    assert!(
        e.to_string().len() < 120 && e.to_string().contains("(1000 bytes)"),
        "{e}"
    );
}

#[test]
fn nesting_at_the_depth_limit_loads_and_one_more_level_is_refused() {
    use alice_physics::scene_io::MAX_SCENE_JSON_DEPTH;
    let cfg = format!("\"config\": {{{CFG4}}}");
    // the top-level object is level 1, so `meta` may hold MAX - 1 nested arrays: the grammar /
    // depth stage passes and the document reaches the member check, which refuses `meta`
    // (the depth limit on its own is pinned by the lib test on the parser)
    let meta = |n: usize| format!("{{{cfg}, \"meta\": {}{}}}", "[".repeat(n), "]".repeat(n));
    let e = load_text("depth_ok.json", &meta(MAX_SCENE_JSON_DEPTH - 1)).unwrap_err();
    assert_eq!(json_version_error(&e), None);
    assert_eq!(field_error(&e), Some(unknown("meta", None)));
    let e = load_text("depth_over.json", &meta(MAX_SCENE_JSON_DEPTH)).unwrap_err();
    assert!(
        matches!(
            json_version_error(&e),
            Some(InvalidSceneJsonVersion::TooDeep { .. })
        ),
        "{e}"
    );
    let e = load_text("depth_huge.json", &meta(500_000)).unwrap_err();
    assert!(
        matches!(
            json_version_error(&e),
            Some(InvalidSceneJsonVersion::TooDeep { .. })
        ),
        "{e}"
    );
}

/// A scene with one body and one joint, every member spelled out (each probe below edits one
/// key of it).
fn full_scene_text() -> String {
    format!(
        "{{\"version\": 1, \"config\": {{{CFG4}}}, \"bodies\": [{{\"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [1,0], \"body_type\": 1}}, {{\"position\": [0,0,0,0,0,0], \"velocity\": [0,0,0,0,0,0], \"rotation\": [0,0,0,0,0,0,1,0], \"mass\": [1,0], \"body_type\": 2}}], \"joints\": [{{\"body_a\": 0, \"body_b\": 1, \"joint_type\": 3, \"anchor_a\": [0,0,0,0,0,0], \"anchor_b\": [0,0,0,0,0,0]}}]}}"
    )
}

/// The `InvalidSceneJson` of loading `text`, which must be refused.
fn refusal(text: &str) -> Option<InvalidSceneJson> {
    // tests run in parallel, so each probe gets its own file
    static PROBE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = PROBE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let e = load_text(&format!("unknown_probe_{n}.json"), text).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidData, "{text}");
    field_error(&e)
}

#[test]
fn misspelled_keys_are_refused_with_the_intended_key() {
    let full = full_scene_text();
    assert_eq!(load_text("full.json", &full).unwrap().config.substeps, 4);
    // the typos that used to load as another scene (substeps 8, iterations 4, no bodies)
    for (from, to, path, hint) in [
        (
            "\"substeps\"",
            "\"subsetps\"",
            "config.subsetps",
            "substeps",
        ),
        (
            "\"iterations\"",
            "\"iteration\"",
            "config.iteration",
            "iterations",
        ),
        ("\"bodies\"", "\"bodys\"", "bodys", "bodies"),
        ("\"joints\"", "\"joint\"", "joint", "joints"),
        ("\"version\"", "\"verison\"", "verison", "version"),
        (
            "\"body_type\": 1",
            "\"bodytype\": 1",
            "bodies[0].bodytype",
            "body_type",
        ),
        (
            "\"joint_type\"",
            "\"join_type\"",
            "joints[0].join_type",
            "joint_type",
        ),
        (
            "\"anchor_b\"",
            "\"anchorb\"",
            "joints[0].anchorb",
            "anchor_b",
        ),
    ] {
        let text = full.replacen(from, to, 1);
        assert_eq!(refusal(&text), Some(unknown(path, Some(hint))), "{to}");
    }
    // case variants match no key exactly but are suggested (compared case-insensitively)
    for variant in ["Gravity", "GRAVITY", "gRaViTy", "Gravty"] {
        let text = full.replacen("\"gravity\"", &format!("\"{variant}\""), 1);
        assert_eq!(
            refusal(&text),
            Some(unknown(&format!("config.{variant}"), Some("gravity"))),
            "{variant}"
        );
    }
    assert_eq!(
        refusal(&full.replacen("\"config\"", "\"CONFIG\"", 1)),
        Some(unknown("CONFIG", Some("config")))
    );
    // the suggestion is the key of the same object, not of another one
    assert_eq!(
        refusal(&full.replacen("\"substeps\"", "\"bodies\"", 1)),
        Some(unknown("config.bodies", None))
    );
}

#[test]
fn the_suggestion_stops_at_edit_distance_two() {
    use alice_physics::scene_io::UNKNOWN_MEMBER_SUGGESTION_DISTANCE;
    assert_eq!(UNKNOWN_MEMBER_SUGGESTION_DISTANCE, 2);
    let full = full_scene_text();
    // distance 1, 2 and 3 from `substeps` (substitutions, an insertion, a deletion)
    for (key, hint) in [
        ("substep", Some("substeps")),
        ("substepsX", Some("substeps")),
        ("subxteps", Some("substeps")),
        ("sbstep", Some("substeps")),
        ("subXtepsYZ", None),
        ("sstep", None),
        ("xxbstepsx", None),
        ("note", None),
        ("", None),
    ] {
        let text = full.replacen("\"substeps\"", &format!("\"{key}\""), 1);
        assert_eq!(
            refusal(&text),
            Some(unknown(&format!("config.{key}"), hint)),
            "{key:?}"
        );
    }
    // the message names the path and, when there is one, the suggestion
    let e = load_text(
        "msg.json",
        &full.replacen("\"substeps\"", "\"subsetps\"", 1),
    )
    .unwrap_err();
    let shown = e.to_string();
    assert!(
        shown.contains("unknown member \"config.subsetps\"")
            && shown.contains("did you mean `substeps`?"),
        "{shown}"
    );
    let e = load_text("msg2.json", &full.replacen("\"substeps\"", "\"zzz\"", 1)).unwrap_err();
    let shown = e.to_string();
    assert!(
        shown.contains("unknown member \"config.zzz\"") && !shown.contains("did you mean"),
        "{shown}"
    );
    // a very long unknown key is shown as a bounded prefix
    let long = "k".repeat(5000);
    let e = load_text(
        "msg3.json",
        &full.replacen("\"substeps\"", &format!("\"{long}\""), 1),
    )
    .unwrap_err();
    assert!(e.to_string().len() < 160, "{}", e.to_string().len());
}

#[test]
fn unknown_members_are_refused_at_every_object() {
    let full = full_scene_text();
    let insert = |after: &str, nth: usize, member: &str| {
        let at = full.match_indices(after).nth(nth).unwrap().0 + after.len();
        format!("{}{member}, {}", &full[..at], &full[at..])
    };
    for (text, path) in [
        // an extra top-level key
        (insert("{", 0, "\"comment\": \"x\""), "comment"),
        // an extra key in config
        (
            insert("\"config\": {", 0, "\"timestep\": 1"),
            "config.timestep",
        ),
        // a nested extra key under bodies[i] (the second body) and under joints[i]
        (
            insert("\"bodies\": [{", 0, "\"name\": \"a\""),
            "bodies[0].name",
        ),
        (insert("}, {", 0, "\"name\": \"b\""), "bodies[1].name"),
        (
            insert("\"joints\": [{", 0, "\"limit\": [1,2]"),
            "joints[0].limit",
        ),
        // a nested object inside an unknown member: refused at the unknown member's path,
        // whatever its value holds (here a key that would be valid in config, and a bad value)
        (
            insert(
                "{",
                0,
                "\"meta\": {\"substeps\": 99, \"x\": {\"y\": [true]}}",
            ),
            "meta",
        ),
        (
            insert(
                "\"joints\": [{",
                0,
                "\"spring\": {\"stiffness\": {\"k\": 1}}",
            ),
            "joints[0].spring",
        ),
    ] {
        assert_eq!(refusal(&text), Some(unknown(path, None)), "{text}");
    }
    // the parent object is reported before an object inside it, then document order
    let both = insert("\"config\": {", 0, "\"c\": 1").replacen("{", "{\"z\": 1, ", 1);
    assert_eq!(refusal(&both), Some(unknown("z", None)));
    let two = insert("\"joints\": [{", 0, "\"j\": 1");
    let two = two.replacen("\"bodies\": [{", "\"bodies\": [{\"b\": 1, ", 1);
    assert_eq!(refusal(&two), Some(unknown("bodies[0].b", None)));
}

#[test]
fn a_duplicate_key_wins_over_an_unknown_member() {
    let full = full_scene_text();
    // the repeated key is inside an unknown member
    let text = full.replacen("{", "{\"meta\": {\"a\": 1, \"a\": 2}, ", 1);
    assert_eq!(
        refusal(&text),
        Some(InvalidSceneJson::DuplicateKey {
            path: "meta.a".into()
        })
    );
    // an unknown member before a repeated known key elsewhere
    let text = full.replacen("{", "{\"meta\": 1, ", 1).replacen(
        "\"iterations\": 6",
        "\"iterations\": 6, \"iterations\": 6",
        1,
    );
    assert_eq!(
        refusal(&text),
        Some(InvalidSceneJson::DuplicateKey {
            path: "config.iterations".into()
        })
    );
    // a repeated unknown key
    let text = full.replacen("{", "{\"meta\": 1, \"meta\": 1, ", 1);
    assert_eq!(
        refusal(&text),
        Some(InvalidSceneJson::DuplicateKey {
            path: "meta".into()
        })
    );
}

/// One document per pair of adjacent stages (grammar / depth → version → repeated keys →
/// unknown members → field values) that fails both stages; the earlier stage is reported.
#[test]
fn the_checks_run_in_the_documented_order() {
    use alice_physics::scene_io::MAX_SCENE_JSON_DEPTH;
    let full = full_scene_text();
    // grammar vs version: version 2 then a trailing comma
    let e = load_text(
        "o1.json",
        &full
            .replacen("\"version\": 1", "\"version\": 2", 1)
            .replacen("}]}", "}],}", 1),
    )
    .unwrap_err();
    assert!(
        matches!(
            json_version_error(&e),
            Some(InvalidSceneJsonVersion::MalformedJson { .. })
        ),
        "{e}"
    );
    // depth vs version: version 2 and an over-deep unknown member
    let deep = format!(
        "{{\"version\": 2, \"meta\": {}{}}}",
        "[".repeat(MAX_SCENE_JSON_DEPTH),
        "]".repeat(MAX_SCENE_JSON_DEPTH)
    );
    let e = load_text("o2.json", &deep).unwrap_err();
    assert!(
        matches!(
            json_version_error(&e),
            Some(InvalidSceneJsonVersion::TooDeep { .. })
        ),
        "{e}"
    );
    // version vs repeated keys, for each version refusal
    let dup_cfg = full.replacen(
        "\"bodies\"",
        &format!("\"config\": {{{CFG4}}}, \"bodies\""),
        1,
    );
    assert_eq!(
        refused_version(
            &load_text(
                "o3.json",
                &dup_cfg.replacen("\"version\": 1", "\"version\": 2", 1)
            )
            .unwrap_err()
        ),
        Some(2)
    );
    for (version, want) in [
        (
            "\"version\": 1, \"version\": 1",
            InvalidSceneJsonVersion::DuplicateVersion,
        ),
        (
            "\"version\": \"1\"",
            InvalidSceneJsonVersion::NotAnUnsignedInteger {
                value: "\"1\"".into(),
            },
        ),
        (
            "\"version\": 4294967296",
            InvalidSceneJsonVersion::OutOfRange {
                value: "4294967296".into(),
            },
        ),
    ] {
        let text = dup_cfg.replacen("\"version\": 1", version, 1);
        assert_eq!(
            json_version_error(&load_text("o4.json", &text).unwrap_err()),
            Some(want),
            "{version}"
        );
    }
    // version vs unknown members, and version vs field values
    let unknown_v2 = full
        .replacen("\"version\": 1", "\"version\": 2", 1)
        .replacen("\"substeps\"", "\"subsetps\"", 1);
    assert_eq!(
        refused_version(&load_text("o5.json", &unknown_v2).unwrap_err()),
        Some(2)
    );
    // repeated keys vs unknown members
    let text = full.replacen("\"substeps\"", "\"subsetps\"", 1).replacen(
        "\"mass\": [1,0], \"body_type\": 2",
        "\"mass\": [1,0], \"mass\": [1,0], \"body_type\": 2",
        1,
    );
    assert_eq!(
        refusal(&text),
        Some(InvalidSceneJson::DuplicateKey {
            path: "bodies[1].mass".into()
        })
    );
    // unknown members vs field values: wrong type, wrong length, out of range, missing
    let late = |t: &str| t.replacen("\"anchor_b\"", "\"anchor_c\"", 1);
    for broken in [
        full.replacen("\"gravity\": [0,0,-10,0,0,0]", "\"gravity\": null", 1),
        full.replacen("\"gravity\": [0,0,-10,0,0,0]", "\"gravity\": [0]", 1),
        full.replacen("\"body_type\": 1", "\"body_type\": 256", 1),
        full.replacen("\"mass\": [1,0], \"body_type\": 1", "\"body_type\": 1", 1),
    ] {
        // without the unknown member each one is a field-value error
        let alone = refusal(&broken).unwrap();
        assert!(
            matches!(
                alone,
                InvalidSceneJson::WrongType { .. }
                    | InvalidSceneJson::WrongLength { .. }
                    | InvalidSceneJson::OutOfRange { .. }
                    | InvalidSceneJson::MissingMember { .. }
            ),
            "{alone:?}"
        );
        assert_eq!(
            refusal(&late(&broken)),
            Some(unknown("joints[0].anchor_c", Some("anchor_a"))),
            "{broken}"
        );
    }
}
