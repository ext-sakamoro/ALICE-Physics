//! ALICE-Law G7: the closed list of the places where a value of another
//! Law-bearing crate crosses into this one, each with its decision.
//!
//! Every `src/` file that names a Law-bearing peer crate in code (comments and
//! string literals removed), and every `src/*_bridge.rs`, must be in
//! [`BOUNDARIES`]; a new one fails here until it is decided. A `handshake`
//! entry lists the constructors that must check the peer's identifier first,
//! and each of them must call the check named for it.

use std::collections::BTreeSet;
use std::path::Path;

/// The Law-bearing crates whose values can cross into this one
/// (`alice-det-math` is the arithmetic itself, covered by G1 and G6).
const PEERS: &[&str] = &[
    "alice_db",
    "alice_analytics",
    "alice_ml",
    "alice_crypto",
    "alice_zip",
    "alice_sdf",
    "alice_kinematics",
    "alice_lol",
    "alice_sync",
];

enum Decision {
    /// each listed function calls the check before it exchanges anything
    Handshake {
        check: &'static str,
        fns: &'static [&'static str],
    },
    /// not a boundary of G7, for the stated reason
    NotABoundary(&'static str),
}

const BOUNDARIES: &[(&str, Decision)] = &[
    (
        "src/analytics_bridge.rs",
        Decision::Handshake {
            check: "check_peer(",
            fns: &["new"],
        },
    ),
    (
        "src/gpu_bridge.rs",
        Decision::Handshake {
            check: "found == crate::PHYSICS_SEMANTICS_ID",
            fns: &["check_bridge"],
        },
    ),
    (
        "src/neural.rs",
        Decision::Handshake {
            check: "check_peer(",
            fns: &["from_ternary_weight", "from_ternary_weight_with_scale"],
        },
    ),
    (
        "src/privacy.rs",
        Decision::NotABoundary(
            "re-exports alice-crypto's functions for the caller; no value of alice-crypto \
             crosses into this crate's computation",
        ),
    ),
    (
        "src/ik_physics_bridge.rs",
        Decision::NotABoundary(
            "targets arrive as this crate's own Fix128 / Vec3Fix, built by the caller: an \
             explicit input, not another crate's Law-bearing value",
        ),
    ),
    (
        "src/sdf_collider.rs",
        Decision::NotABoundary(
            "no alice-sdf dependency: distances come from caller closures (alice-sdf appears \
             only in a documentation example)",
        ),
    ),
];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Source with `//` comments, block comments and string literals blanked.
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else if b[i] == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            out.push_str("\"\"");
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn src_files() -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).expect("read src") {
            let p = e.expect("entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let rel = p.strip_prefix(root()).expect("under root");
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut v = Vec::new();
    walk(&root().join("src"), &mut v);
    v.sort();
    v
}

fn names_a_peer(code: &str) -> bool {
    PEERS.iter().any(|p| {
        code.match_indices(&format!("{p}::")).any(|(i, _)| {
            i == 0
                || !code.as_bytes()[i - 1].is_ascii_alphanumeric() && code.as_bytes()[i - 1] != b'_'
        })
    })
}

/// The body of the first `fn <name>` in `code` (from its `{` to the matching `}`).
fn fn_body<'a>(code: &'a str, name: &str) -> Option<&'a str> {
    let at = code
        .match_indices(&format!("fn {name}"))
        .find(|(i, _)| {
            let next = code
                .as_bytes()
                .get(i + 3 + name.len())
                .copied()
                .unwrap_or(b' ');
            !next.is_ascii_alphanumeric() && next != b'_'
        })?
        .0;
    let open = at + code[at..].find('{')?;
    let mut depth = 0usize;
    for (k, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&code[open..=open + k]);
                }
            }
            _ => {}
        }
    }
    None
}

#[test]
fn every_file_naming_a_peer_or_called_a_bridge_has_a_decision() {
    let decided: BTreeSet<&str> = BOUNDARIES.iter().map(|(f, _)| *f).collect();
    assert_eq!(decided.len(), BOUNDARIES.len(), "a file is listed twice");
    let mut found = Vec::new();
    let mut scanned = 0;
    for rel in src_files() {
        let code = code_only(&std::fs::read_to_string(root().join(&rel)).expect("read"));
        scanned += 1;
        if rel == "src/lib.rs" {
            continue; // module declarations and re-exports of this crate's own items
        }
        if names_a_peer(&code) || rel.ends_with("_bridge.rs") {
            found.push(rel);
        }
    }
    assert!(
        scanned > 100,
        "scanned {scanned} files: the walk read almost nothing"
    );
    assert!(
        found.len() >= 5,
        "found {} boundaries: the scan found almost nothing",
        found.len()
    );
    let undecided: Vec<_> = found
        .iter()
        .filter(|f| !decided.contains(f.as_str()))
        .collect();
    assert!(
        undecided.is_empty(),
        "boundaries without a G7 decision: {undecided:?}"
    );
    for f in &decided {
        assert!(root().join(f).exists(), "{f} is listed but does not exist");
    }
}

#[test]
fn every_handshake_constructor_checks_first() {
    let mut checked = 0;
    for (file, decision) in BOUNDARIES {
        if let Decision::Handshake { check, fns } = decision {
            let code = code_only(&std::fs::read_to_string(root().join(file)).expect("read"));
            let check = code_only(check);
            for name in *fns {
                let body = fn_body(&code, name).unwrap_or_else(|| panic!("{file}: no fn {name}"));
                assert!(
                    body.contains(check.as_str()),
                    "{file}: fn {name} does not call {check}"
                );
                checked += 1;
            }
        }
        if let Decision::NotABoundary(why) = decision {
            assert!(why.len() > 40, "{file}: the reason is too short to be one");
        }
    }
    assert_eq!(checked, 4, "handshake constructors checked");
}

#[test]
fn the_scan_sees_a_peer_in_code_but_not_in_comments_or_strings() {
    assert!(names_a_peer(&code_only(
        "let d = alice_db::AliceDB::in_memory(c);"
    )));
    assert!(!names_a_peer(&code_only(
        "//! use alice_sdf::CompiledSdf;\nfn f() {}"
    )));
    assert!(!names_a_peer(&code_only(
        "/* alice_ml::Ternary */ fn f() {}"
    )));
    assert!(!names_a_peer(&code_only("let s = \"alice_zip::x\";")));
    assert!(!names_a_peer(&code_only("my_alice_db::x();")));
    let code = "fn open() { alice_db_handshake(\"x\")?; }\nfn open_more() {}";
    assert_eq!(
        fn_body(code, "open"),
        Some("{ alice_db_handshake(\"x\")?; }")
    );
}

#[cfg(feature = "gpu-solver-bridge")]
mod gpu {
    use alice_physics_core::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics_core::math::{Fix128, Vec3Fix};
    use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
    use alice_physics_core::{PeerMismatch, PHYSICS_SEMANTICS_ID};

    struct Other([u8; 32]);
    impl GpuSolverBridge for Other {
        fn semantics_id(&self) -> [u8; 32] {
            self.0
        }
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {
            panic!("a refused bridge was used");
        }
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {
            panic!("a refused bridge was used");
        }
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {
            panic!("a refused bridge was used");
        }
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
    }

    fn world() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 5, 0),
            Fix128::ONE,
        ));
        w
    }

    #[test]
    fn a_bridge_of_another_semantics_is_refused_on_every_path() {
        let mut other = PHYSICS_SEMANTICS_ID;
        other[31] ^= 1;
        let expect = |bridge: &'static str| PeerMismatch {
            bridge,
            peer: "gpu-solver-bridge",
            found: other,
            expected: PHYSICS_SEMANTICS_ID,
        };
        let dt = Fix128::from_ratio(1, 60);
        let mut w = world();
        let before = w.snapshot_world();
        assert_eq!(
            w.step_with_bridge(&mut Other(other), dt),
            Err(expect("step_with_bridge"))
        );
        assert_eq!(
            w.substep_with_bridge(&mut Other(other), dt),
            Err(expect("substep_with_bridge"))
        );
        assert_eq!(
            w.solve_contact_constraints_with_bridge(&mut Other(other)),
            Err(expect("solve_contact_constraints_with_bridge"))
        );
        assert_eq!(
            w.set_gpu_solver_bridge(Some(Box::new(Other(other)))),
            Err(expect("set_gpu_solver_bridge"))
        );
        assert!(
            !w.gpu_solver_bridge_installed(),
            "a refused bridge was installed"
        );
        assert_eq!(
            w.snapshot_world(),
            before,
            "a refused path changed the world"
        );
        // this build's identifier is accepted
        assert_eq!(
            w.set_gpu_solver_bridge(Some(Box::new(Other(PHYSICS_SEMANTICS_ID)))),
            Ok(())
        );
        assert!(w.gpu_solver_bridge_installed());
    }
}
