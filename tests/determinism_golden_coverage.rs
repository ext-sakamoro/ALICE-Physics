//! Which stepping paths the determinism goldens pin, checked against the code.
//!
//! The goldens in `tests/determinism_*.rs` pin the bits of a world after it
//! has been stepped. A path that no golden runs can change its results
//! without any golden noticing, so the set of pinned paths has to cover every
//! way the crate steps a world. This file lists that set and checks it.
//!
//! # What is required
//!
//! * Every solver backend (read from `pub enum SolverBackend` in
//!   `src/solver.rs`) on every stepping entry point of `PhysicsWorld`.
//!   The entry points are the `pub fn` items taking `&mut self` inside
//!   `impl PhysicsWorld` blocks (also written `impl crate::solver::PhysicsWorld`)
//!   anywhere under `src/`;
//!   each one must be classified, by hand, in [`STEP_ENTRIES`] or in
//!   [`NOT_STEP`]. A new method that is in neither list fails the check, so a
//!   new way to step a world cannot appear unclassified whatever its name.
//! * Every broadphase kind (read from `pub enum Broadphase`).
//! * The scene features in [`FEATURES`] (listed by hand).
//!
//! # What counts as a pin
//!
//! Every row of [`PINS`] names a golden test and is checked mechanically on
//! the test and the same-file helpers it calls, with string literals and
//! comments blanked first:
//!
//! * the test exists and is not ignored, and its `#[cfg(feature = ..)]`
//!   attributes are exactly the features the row declares;
//! * a constant digest (a 64-digit hex string or a `u64` hex constant) of the
//!   file is used inside an `assert` statement it reaches;
//! * for a backend × entry row: the entry point is called on a value of type
//!   `PhysicsWorld` (a parameter of that type, `PhysicsWorld::new(..)`, or the
//!   result of a same-file function returning `PhysicsWorld`), and the
//!   backend is named as `SolverBackend::<name>`, or, for the default backend
//!   only, no backend is named anywhere the test reaches and the test calls
//!   no helper of another module (whose backend would be invisible here);
//! * for a feature row: the code listed in [`FEATURES`] for it.
//!
//! A row of [`RELATION_PINS`] pins a combination by its relation to a `PINS`
//! row: its test passes the checks above for its own combination and asserts
//! a digest constant (of the same file) that the test of the named `PINS` row
//! also asserts, and it hands the stepped world to `assert_stepped_with(..)`
//! with its own backend and that digest. The helper asserts the backend and
//! the digest on the one world it is given, so the evidence that the
//! combination ran is checked when the test runs, not read from names: a
//! second world, or a world whose backend was changed before stepping, fails
//! the test. As cheap static guards the test must also build exactly one
//! world and must not name the backend of the named row.
//! The four TGS rows on the paths that do not consult the
//! backend are pinned this way; they are not `PINS` rows, so the entries
//! folded into `PHYSICS_SEMANTICS_ID` do not change.
//!
//! Every required combination is either pinned (by [`PINS`] or
//! [`RELATION_PINS`]) or listed in [`KNOWN_GAPS`]
//! (with the reason), never both, and both tables name only required
//! combinations. `golden_coverage_has_no_gaps` is ignored while
//! [`KNOWN_GAPS`] is not empty and fails until every combination has a golden.
//!
//! # Limits
//!
//! The check reads source text; it does not run the tests.
//!
//! * A helper in another file is not followed (a test that calls one cannot
//!   pin the default backend by omission).
//! * The other operand of the digest assertion is not proven to be a hash of
//!   the stepped world; only that the digest constant is asserted.
//! * A test that names two backends (one loop over both) pins both rows; the
//!   check does not see which of them the digest belongs to.
//! * A row whose test is behind a feature (`parallel`, `gpu-solver-bridge`) is
//!   only executed by CI lanes that enable that feature; the row records the
//!   feature so the gate is visible.
//! * A `step_parallel` row on a scene where no two constraints share a body
//!   pins the parity with `step`, not the batched ordering; the ordering has
//!   its own row, `step_parallel shared-body order`.
//! * For a [`RELATION_PINS`] row, the static guards (one world built, the
//!   other backend not named) count call sites and names only; a loop or a
//!   later change of `config.solver_backend` passes them. The runtime check
//!   of `assert_stepped_with` is what catches those. Its body is checked only
//!   for asserting `.config.solver_backend`.
//! * Feature evidence is code presence: `contacts` needs two bodies added
//!   with a collision radius, not a measured contact.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// `PhysicsWorld` methods that advance the simulation.
const STEP_ENTRIES: &[&str] = &[
    "step",
    "step_n",
    "try_step",
    "step_parallel",
    "try_step_parallel",
    "step_with_bridge",
    "substep_with_bridge",
];

/// `PhysicsWorld` methods taking `&mut self` that do not advance the
/// simulation (setup, state transfer, queries with side effects).
/// `solve_contact_constraints_with_bridge` and `solve_joints_with_bridge` do
/// move bodies: each runs one solve pass through the bridge, and a host can
/// call them directly. They are classified here because they integrate no
/// time and run no participant, so they are a stage of a step rather than a
/// step law; their result is pinned through `step_with_bridge`, which runs them.
const NOT_STEP: &[&str] = &[
    "add_body",
    "add_body_with_radius",
    "add_compound_body",
    "add_contact",
    "add_contact_modifier",
    "add_contact_with_material",
    "add_distance_constraint",
    "add_force_field",
    "add_joint",
    "add_joint_motor",
    "add_joint_motor_3d",
    "add_participant",
    "add_pre_solve_hook",
    "add_sdf_collider",
    "add_shaped_body",
    "add_static_collider",
    "begin_frame",
    "clear_body_collision_radius",
    "clear_contact_modifiers",
    "clear_contacts",
    "clear_fault",
    "clear_pre_solve_hooks",
    "declare_field",
    "deserialize_state",
    "disable_joint_motor",
    "drain_contact_events",
    "drain_trigger_events",
    "end_frame",
    "get_body_mut",
    "joint_motor_3d_mut",
    "joint_motor_mut",
    "rebuild_batches",
    "remove_body",
    "remove_force_field",
    "remove_joint",
    "remove_sdf_collider",
    "remove_static_collider",
    "reset_tgs_cache_stats",
    "reset_world",
    "restore_world",
    "set_body_collision_radius",
    "set_body_filter",
    "set_body_material",
    "set_body_shape",
    "set_broadphase",
    "set_continuous_collision",
    "set_field",
    "set_gpu_solver_bridge",
    "set_joint_motor_3d_rotation_target",
    "set_joint_motor_velocity_target",
    "set_sdf_collision_radius",
    "set_sleep_config",
    "set_sleep_skip",
    "solve_contact_constraints_with_bridge",
    "solve_joints_with_bridge",
    "take_gpu_solver_bridge",
    "wake_body",
];

/// A golden that pins a combination:
/// (combination, file under `tests/`, test fn, `#[cfg(feature)]` of the test or "").
const PINS: &[(&str, &str, &str, &str)] = &[
    (
        "Xpbd step",
        "determinism_golden.rs",
        "determinism_freefall",
        "",
    ),
    (
        "Xpbd step_n",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_step_n",
        "",
    ),
    (
        "Xpbd try_step",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_try_step",
        "",
    ),
    (
        "Xpbd step_parallel",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_step_parallel",
        "parallel",
    ),
    (
        "Xpbd try_step_parallel",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_try_step_parallel",
        "parallel",
    ),
    (
        "Xpbd step_with_bridge",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_step_with_bridge",
        "gpu-solver-bridge",
    ),
    (
        "Xpbd substep_with_bridge",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_substep_with_bridge",
        "gpu-solver-bridge",
    ),
    (
        "Tgs step",
        "determinism_golden_paths.rs",
        "golden_path_tgs_step",
        "",
    ),
    (
        "Tgs step_n",
        "determinism_golden_paths.rs",
        "golden_path_tgs_step_n",
        "",
    ),
    (
        "Tgs try_step",
        "determinism_golden_paths.rs",
        "golden_path_tgs_try_step",
        "",
    ),
    (
        "contacts",
        "determinism_golden_contacts.rs",
        "determinism_contact_stack",
        "",
    ),
    (
        "distance constraint",
        "determinism_golden.rs",
        "determinism_joint_pendulum",
        "",
    ),
    (
        "joint",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_step",
        "",
    ),
    (
        "continuous collision",
        "determinism_golden_paths.rs",
        "golden_continuous_collision",
        "",
    ),
    (
        "sleeping",
        "determinism_golden_paths.rs",
        "golden_sleeping",
        "",
    ),
    (
        "broadphase Bvh",
        "determinism_golden_paths.rs",
        "golden_path_xpbd_step",
        "",
    ),
    (
        "broadphase DynamicTree",
        "determinism_golden_paths.rs",
        "golden_broadphase_dynamic_tree",
        "",
    ),
    (
        "broadphase Hybrid",
        "determinism_golden_paths.rs",
        "golden_broadphase_hybrid",
        "",
    ),
    (
        "participant",
        "determinism_golden_paths.rs",
        "golden_participant",
        "",
    ),
    (
        "cloth",
        "determinism_golden.rs",
        "determinism_cloth_drape",
        "",
    ),
    (
        "physics2d",
        "determinism_physics2d_step_digest.rs",
        "step_digest_is_unchanged",
        "",
    ),
];

/// Required combinations pinned by a relation to another pinned combination:
/// (combination, the `PINS` combination it gives the bits of, file, test,
/// feature). The paths below do not consult `config.solver_backend` and run
/// the XPBD substep loop (documented on each entry point), so a TGS world
/// stepped through them is pinned to the XPBD digest of the same path.
///
/// A row counts as a pin when its test exercises the combination (as a `PINS`
/// row would) and asserts a digest constant that the test of the named `PINS`
/// row also asserts. The rows are not part of `PINS`, so they add no entry to
/// `PHYSICS_SEMANTICS_PINS`: their bits are those of the named row, which is
/// already folded into `PHYSICS_SEMANTICS_ID`.
const RELATION_PINS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "Tgs step_parallel",
        "Xpbd step_parallel",
        "determinism_golden_paths.rs",
        "golden_path_tgs_step_parallel_gives_the_xpbd_bits",
        "parallel",
    ),
    (
        "Tgs try_step_parallel",
        "Xpbd try_step_parallel",
        "determinism_golden_paths.rs",
        "golden_path_tgs_try_step_parallel_gives_the_xpbd_bits",
        "parallel",
    ),
    (
        "Tgs step_with_bridge",
        "Xpbd step_with_bridge",
        "determinism_golden_paths.rs",
        "golden_path_tgs_step_with_bridge_gives_the_xpbd_bits",
        "gpu-solver-bridge",
    ),
    (
        "Tgs substep_with_bridge",
        "Xpbd substep_with_bridge",
        "determinism_golden_paths.rs",
        "golden_path_tgs_substep_with_bridge_gives_the_xpbd_bits",
        "gpu-solver-bridge",
    ),
];

/// Required combinations that no golden pins yet: (combination, reason).
const KNOWN_GAPS: &[(&str, &str)] = &[
    (
        "step_parallel shared-body order",
        "no golden runs step_parallel on a scene where two constraints share a body",
    ),
    (
        "installed bridge",
        "no golden steps a world with a bridge installed by set_gpu_solver_bridge, which reroutes its contact solve",
    ),
];

/// Scene features every stepping law must have a golden for, with the code
/// that shows a test exercises them (checked on comment- and string-free
/// text): (feature, needles, minimum count of the first needle).
const FEATURES: &[(&str, &[&str], usize)] = &[
    ("contacts", &["add_body_with_radius("], 2),
    ("distance constraint", &["add_distance_constraint("], 1),
    ("joint", &["add_joint("], 1),
    ("continuous collision", &["set_continuous_collision("], 1),
    ("sleeping", &["is_sleeping("], 1),
    ("participant", &["add_participant("], 1),
    ("cloth", &["Cloth::"], 1),
    ("physics2d", &["PhysicsWorld2D"], 1),
    // an installed bridge reroutes the contact solve of `step` / `substep`
    ("installed bridge", &["set_gpu_solver_bridge("], 1),
    // two joints on one chain share their middle body
    (
        "step_parallel shared-body order",
        &["add_joint(", ".step_parallel("],
        2,
    ),
];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
        .replace("\r\n", "\n")
}

// ── Source scanning ─────────────────────────────────────────────────────────

/// The source with the contents of comments, string literals and char
/// literals replaced by spaces (newlines kept), so braces and words inside
/// them are not taken for code. Byte offsets are unchanged.
fn blank_literals(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for c in &mut out[from..to] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    };
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                let end = src[i..].find('\n').map_or(b.len(), |n| i + n);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let end = src[i + 2..].find("*/").map_or(b.len(), |n| i + 2 + n + 2);
                blank(&mut out, i, end);
                i = end;
            }
            b'r' if (b.get(i + 1) == Some(&b'"') || b.get(i + 1) == Some(&b'#'))
                && (i == 0 || !is_ident(b[i - 1])) =>
            {
                let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
                let open = i + 1 + hashes;
                if b.get(open) != Some(&b'"') {
                    i += 1;
                    continue;
                }
                let close = format!("\"{}", "#".repeat(hashes));
                let end = src[open + 1..]
                    .find(&close)
                    .map_or(b.len(), |n| open + 1 + n + close.len());
                blank(
                    &mut out,
                    open + 1,
                    end.saturating_sub(close.len()).max(open + 1),
                );
                i = end;
            }
            b'"' => {
                let mut j = i + 1;
                while j < b.len() && b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                blank(&mut out, i + 1, j.min(b.len()));
                i = j + 1;
            }
            b'\'' => {
                // a char literal ('x', '\n', '\u{7b}'); a lifetime has no closing quote nearby
                let close = if b.get(i + 1) == Some(&b'\\') {
                    src[i + 2..].find('\'').map(|n| i + 2 + n)
                } else if b.get(i + 2) == Some(&b'\'') {
                    Some(i + 2)
                } else {
                    None
                };
                match close {
                    Some(c) if c - i <= 10 => {
                        blank(&mut out, i + 1, c);
                        i = c + 1;
                    }
                    _ => i += 1,
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("blanking keeps ASCII positions")
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn ident_at(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// The end (exclusive) of the bracketed group opening at `open`.
fn matching(code: &str, open: usize) -> usize {
    let b = code.as_bytes();
    let (o, c) = (b[open], if b[open] == b'(' { b')' } else { b'}' });
    let mut depth = 0usize;
    for (k, &ch) in b[open..].iter().enumerate() {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return open + k + 1;
            }
        }
    }
    code.len()
}

/// A `fn` item of a blanked source.
struct FnItem {
    /// From the name to the opening brace (parameters and return type).
    signature: String,
    /// Between the braces.
    body: String,
    /// The attribute lines right above it.
    attrs: String,
}

/// Every `fn` item with a body: name -> item. Nested fns are items too.
fn fn_items(code: &str) -> BTreeMap<String, FnItem> {
    let b = code.as_bytes();
    let mut out = BTreeMap::new();
    let mut from = 0;
    while let Some(n) = code[from..].find("fn ") {
        let at = from + n;
        from = at + 3;
        if at > 0 && is_ident(b[at - 1]) {
            continue;
        }
        let name = ident_at(&code[at + 3..]);
        if name.is_empty() {
            continue;
        }
        let Some(open_rel) = code[at..].find(['{', ';']) else {
            break;
        };
        let open = at + open_rel;
        if b[open] == b';' {
            continue; // a declaration without a body (trait item)
        }
        let end = matching(code, open);
        // attributes: the lines right above `fn` that start with `#[`
        let line_start = code[..at].rfind('\n').map_or(0, |p| p + 1);
        let mut attrs = String::new();
        let mut cursor = line_start;
        while cursor > 0 {
            let prev_start = code[..cursor - 1].rfind('\n').map_or(0, |p| p + 1);
            let line = code[prev_start..cursor - 1].trim();
            if line.starts_with("#[") {
                attrs.insert_str(0, &format!("{line}\n"));
                cursor = prev_start;
            } else {
                break;
            }
        }
        out.entry(name).or_insert_with(|| FnItem {
            signature: code[at + 3..open].to_string(),
            body: code[open + 1..end.saturating_sub(1)].to_string(),
            attrs,
        });
    }
    out
}

/// The same-file fns `test` reaches (itself included), transitively.
fn reachable_fns(items: &BTreeMap<String, FnItem>, test: &str) -> Vec<String> {
    let mut seen = BTreeSet::from([test.to_string()]);
    let mut stack = vec![test.to_string()];
    let mut order = Vec::new();
    while let Some(name) = stack.pop() {
        let Some(item) = items.get(&name) else {
            continue;
        };
        order.push(name.clone());
        let body = item.body.as_bytes();
        for (i, _) in item.body.match_indices('(') {
            let start = body[..i]
                .iter()
                .rposition(|&c| !is_ident(c))
                .map_or(0, |p| p + 1);
            let callee = &item.body[start..i];
            if items.contains_key(callee) && seen.insert(callee.to_string()) {
                stack.push(callee.to_string());
            }
        }
    }
    order
}

/// Names of constant digests in a raw source: `const X: &str = "<64 hex>"`
/// and `const X: u64 = 0x<16 hex digits>`.
fn digest_consts(raw: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for item in raw.split("const ").skip(1) {
        let name = ident_at(item);
        let decl = item.split(';').next().unwrap_or("");
        let value: String = decl
            .split('=')
            .nth(1)
            .unwrap_or("")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        let sha = value.len() == 66
            && value.starts_with('"')
            && value[1..65].bytes().all(|c| c.is_ascii_hexdigit());
        let word = value.starts_with("0x")
            && value[2..].bytes().filter(|&c| c != b'_').count() == 16
            && value[2..]
                .bytes()
                .all(|c| c.is_ascii_hexdigit() || c == b'_');
        if sha || word {
            out.insert(name);
        }
    }
    out
}

fn mentions_ident(text: &str, ident: &str) -> bool {
    let b = text.as_bytes();
    text.match_indices(ident).any(|(i, _)| {
        let before = i == 0 || !is_ident(b[i - 1]);
        let after = b.get(i + ident.len()).is_none_or(|&c| !is_ident(c));
        before && after
    })
}

/// Whether a digest constant is used inside an `assert` statement of `body`.
fn asserts_digest(body: &str, consts: &BTreeSet<String>) -> bool {
    body.split(';')
        .any(|stmt| stmt.contains("assert") && consts.iter().any(|c| mentions_ident(stmt, c)))
}

/// The return type of a fn signature (after `->`), trimmed.
fn return_type(signature: &str) -> String {
    signature
        .rsplit_once("->")
        .map(|(_, r)| r.trim().to_string())
        .unwrap_or_default()
}

/// Names bound to a `PhysicsWorld` inside one fn: parameters of that type,
/// `let` bindings of `PhysicsWorld::new(..)` or of a same-file fn returning
/// `PhysicsWorld` (or a tuple whose first element is one).
fn world_names(item: &FnItem, items: &BTreeMap<String, FnItem>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let sig = &item.signature;
    if let Some(open) = sig.find('(') {
        let close = matching(sig, open);
        for param in sig[open + 1..close.saturating_sub(1)].split(',') {
            let Some((pat, ty)) = param.split_once(':') else {
                continue;
            };
            let ty = ty
                .trim()
                .trim_start_matches('&')
                .trim_start_matches("mut ")
                .trim();
            if ty.starts_with("PhysicsWorld") && !ty.starts_with("PhysicsWorld2D") {
                out.insert(pat.trim().trim_start_matches("mut ").trim().to_string());
            }
        }
    }
    let returns_world = |callee: &str, tuple: bool| {
        items.get(callee).is_some_and(|f| {
            let r = return_type(&f.signature);
            if tuple {
                r.starts_with("(PhysicsWorld,")
            } else {
                r == "PhysicsWorld"
            }
        })
    };
    for (i, _) in item.body.match_indices("let ") {
        let rest = &item.body[i + 4..];
        let Some(eq) = rest.find('=') else { continue };
        let pat = rest[..eq].trim();
        let rhs = rest[eq + 1..].trim_start();
        let callee = ident_at(rhs);
        if let Some(tuple) = pat.strip_prefix('(') {
            let first = tuple
                .split(',')
                .next()
                .unwrap_or("")
                .trim()
                .trim_start_matches("mut ")
                .trim();
            if returns_world(&callee, true) {
                out.insert(first.to_string());
            }
            continue;
        }
        let (name, annotated) = match pat.split_once(':') {
            Some((n, t)) => (n, t.trim() == "PhysicsWorld"),
            None => (pat, false),
        };
        let name = name.trim().trim_start_matches("mut ").trim();
        if annotated || rhs.starts_with("PhysicsWorld::new(") || returns_world(&callee, false) {
            out.insert(name.to_string());
        }
    }
    out
}

// ── What a test exercises ───────────────────────────────────────────────────

/// What the code a golden test reaches shows.
struct TestFacts {
    active: bool,
    features: BTreeSet<String>,
    asserts_digest: bool,
    /// The digest constants used inside an `assert` statement it reaches.
    asserted: BTreeSet<String>,
    /// Entry points called on a `PhysicsWorld` value.
    world_calls: BTreeSet<String>,
    /// All reached code, blanked.
    text: String,
    /// The test calls a helper of another module.
    foreign_helper: bool,
}

fn test_facts(raw: &str, test: &str, entries: &BTreeSet<String>) -> TestFacts {
    let code = blank_literals(raw);
    let items = fn_items(&code);
    let Some(item) = items.get(test) else {
        return TestFacts {
            active: false,
            features: BTreeSet::new(),
            asserts_digest: false,
            asserted: BTreeSet::new(),
            world_calls: BTreeSet::new(),
            text: String::new(),
            foreign_helper: false,
        };
    };
    let active = item.attrs.contains("#[test]") && !item.attrs.contains("#[ignore");
    let features = item
        .attrs
        .lines()
        .filter_map(|l| l.strip_prefix("#[cfg(feature ="))
        .map(|l| l.trim_end_matches(")]").trim().to_string())
        .collect();
    // feature names are inside a string literal, blanked in `code`: read them raw
    let raw_items = fn_items_attrs_raw(raw, test);
    let modules: Vec<String> = code
        .lines()
        .filter_map(|l| l.trim().strip_prefix("mod "))
        .map(ident_at)
        .collect();
    let consts = digest_consts(raw);
    let mut text = String::new();
    let mut asserted = BTreeSet::new();
    let mut calls = BTreeSet::new();
    for name in reachable_fns(&items, test) {
        let f = &items[&name];
        text.push_str(&f.body);
        text.push('\n');
        for c in &consts {
            if asserts_digest(&f.body, &BTreeSet::from([c.clone()])) {
                asserted.insert(c.clone());
            }
        }
        for w in world_names(f, &items) {
            for e in entries {
                if f.body.contains(&format!("{w}.{e}(")) {
                    calls.insert(e.clone());
                }
            }
        }
    }
    let foreign_helper = modules.iter().any(|m| text.contains(&format!("{m}::")));
    TestFacts {
        active,
        features: if raw_items.is_empty() {
            features
        } else {
            raw_items
        },
        asserts_digest: !asserted.is_empty(),
        asserted,
        world_calls: calls,
        text,
        foreign_helper,
    }
}

/// The `#[cfg(feature = "x")]` features on the attribute lines above `fn <test>`, read raw.
fn fn_items_attrs_raw(raw: &str, test: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(at) = raw.find(&format!("fn {test}(")) else {
        return out;
    };
    let mut cursor = raw[..at].rfind('\n').map_or(0, |p| p + 1);
    while cursor > 0 {
        let prev = raw[..cursor - 1].rfind('\n').map_or(0, |p| p + 1);
        let line = raw[prev..cursor - 1].trim();
        if !line.starts_with("#[") && !line.starts_with("///") {
            break;
        }
        if let Some(rest) = line.strip_prefix("#[cfg(feature = \"") {
            out.insert(rest.split('"').next().unwrap_or("").to_string());
        }
        cursor = prev;
    }
    out
}

// ── Requirements ────────────────────────────────────────────────────────────

/// Variants of `pub enum <name>` in `src`, and the one marked `#[default]`.
fn enum_variants(src: &str, name: &str) -> (Vec<String>, String) {
    let code = blank_literals(src);
    let start = code
        .find(&format!("pub enum {name} {{"))
        .unwrap_or_else(|| panic!("pub enum {name}"));
    let body = &code[start..start + code[start..].find("\n}").expect("end of enum")];
    let mut variants = Vec::new();
    let mut default = String::new();
    let mut next_is_default = false;
    for line in body.lines().skip(1) {
        let t = line.trim();
        if t.starts_with("#[default]") {
            next_is_default = true;
            continue;
        }
        if line.starts_with("    ")
            && !line.starts_with("     ")
            && t.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            let v = ident_at(t);
            if next_is_default {
                default = v.clone();
                next_is_default = false;
            }
            variants.push(v);
        }
    }
    assert!(!default.is_empty(), "no #[default] variant in {name}");
    (variants, default)
}

/// `pub fn` names taking `&mut self` inside `impl PhysicsWorld` blocks of `src`.
fn world_mut_methods(src: &str) -> BTreeSet<String> {
    let code = blank_literals(src);
    let mut out = BTreeSet::new();
    let mut from = 0;
    while let Some(n) = code[from..].find("impl") {
        let at = from + n;
        from = at + 4;
        if at > 0 && is_ident(code.as_bytes()[at - 1]) {
            continue;
        }
        let Some(open_rel) = code[at..].find('{') else {
            break;
        };
        let header: String = code[at + 4..at + open_rel]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        // `impl PhysicsWorld`, `impl crate::solver::PhysicsWorld`, `impl<..> PhysicsWorld`;
        // not `impl Trait for PhysicsWorld` and not other types (`PyPhysicsWorld`)
        let path = if header.starts_with('<') {
            header[matching_angle(&header)..].trim()
        } else {
            header.as_str()
        };
        let is_world =
            !path.contains(" for ") && (path == "PhysicsWorld" || path.ends_with("::PhysicsWorld"));
        let open = at + open_rel;
        let end = matching(&code, open);
        if is_world {
            let body = &code[open..end];
            for (i, _) in body.match_indices("pub fn ") {
                let rest = &body[i + 7..];
                let name = ident_at(rest);
                let params = &rest[name.len()..];
                let params = params.trim_start();
                let params = if params.starts_with('<') {
                    &params[matching_angle(params)..]
                } else {
                    params
                };
                if params
                    .trim_start()
                    .trim_start_matches('(')
                    .trim_start()
                    .starts_with("&mut self")
                {
                    out.insert(name);
                }
            }
        }
        from = end.max(from);
    }
    out
}

/// The end (exclusive) of the `<...>` group at the start of `s`.
fn matching_angle(s: &str) -> usize {
    let mut depth = 0usize;
    for (k, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return k + 1;
                }
            }
            _ => {}
        }
    }
    s.len()
}

struct Required {
    combos: BTreeSet<String>,
    backends: Vec<String>,
    default_backend: String,
    broadphases: Vec<String>,
    default_broadphase: String,
    entries: BTreeSet<String>,
    unclassified: Vec<String>,
    stale_classification: Vec<String>,
}

/// `src/solver.rs` first (it defines the enums), then every other `.rs` file
/// under `src/`: an `impl PhysicsWorld` block can live in any module.
fn solver_sources() -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read src").flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&root().join("src"), &mut files);
    files.sort();
    let mut out = vec![read("src/solver.rs")];
    for p in files {
        let rel = p
            .strip_prefix(root())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if rel != "src/solver.rs" {
            out.push(read(&rel));
        }
    }
    out
}

fn required_from(sources: &[String]) -> Required {
    let (backends, default_backend) = enum_variants(&sources[0], "SolverBackend");
    let (broadphases, default_broadphase) = enum_variants(&sources[0], "Broadphase");
    let methods: BTreeSet<String> = sources.iter().flat_map(|s| world_mut_methods(s)).collect();
    let classified: BTreeSet<String> = STEP_ENTRIES
        .iter()
        .chain(NOT_STEP)
        .map(|s| (*s).to_string())
        .collect();
    let unclassified = methods.difference(&classified).cloned().collect();
    let stale_classification = classified.difference(&methods).cloned().collect();
    let entries: BTreeSet<String> = STEP_ENTRIES
        .iter()
        .map(|s| (*s).to_string())
        .filter(|e| methods.contains(e))
        .collect();
    let mut combos = BTreeSet::new();
    for b in &backends {
        for e in &entries {
            combos.insert(format!("{b} {e}"));
        }
    }
    for k in &broadphases {
        combos.insert(format!("broadphase {k}"));
    }
    for (f, _, _) in FEATURES {
        combos.insert((*f).to_string());
    }
    Required {
        combos,
        backends,
        default_backend,
        broadphases,
        default_broadphase,
        entries,
        unclassified,
        stale_classification,
    }
}

/// Whether a golden test with these facts exercises `combo`; the reason when it does not.
fn exercises(req: &Required, t: &TestFacts, combo: &str) -> Result<(), String> {
    let names = |kind: &str, v: &str| mentions_ident(&t.text, &format!("{kind}::{v}"));
    if let Some(kind) = combo.strip_prefix("broadphase ") {
        let any = req.broadphases.iter().any(|k| names("Broadphase", k));
        return if names("Broadphase", kind)
            || (kind == req.default_broadphase && !any && !t.foreign_helper)
        {
            Ok(())
        } else {
            Err(format!("does not select Broadphase::{kind}"))
        };
    }
    if let Some((_, needles, min)) = FEATURES.iter().find(|(f, _, _)| *f == combo) {
        if t.text.matches(needles[0]).count() < *min {
            return Err(format!("uses `{}` fewer than {min} times", needles[0]));
        }
        if let Some(missing) = needles[1..].iter().find(|n| !t.text.contains(*n)) {
            return Err(format!("does not use `{missing}`"));
        }
        if combo == "sleeping"
            && (t.text.contains("frames_to_sleep: u32::MAX")
                || t.text.contains("set_sleep_skip(false)"))
        {
            return Err("disables sleeping".into());
        }
        return Ok(());
    }
    let (backend, entry) = combo
        .split_once(' ')
        .ok_or_else(|| format!("unknown combination {combo:?}"))?;
    if !t.world_calls.contains(entry) {
        return Err(format!("does not call .{entry}( on a PhysicsWorld"));
    }
    let named = names("SolverBackend", backend);
    let any = req.backends.iter().any(|b| names("SolverBackend", b));
    if named || (backend == req.default_backend && !any && !t.foreign_helper) {
        Ok(())
    } else {
        Err(format!("does not select SolverBackend::{backend}"))
    }
}

/// How many worlds `test` builds: calls of a same-file function that returns
/// a `PhysicsWorld` without taking one (a constructor such as `path_scene`),
/// plus `PhysicsWorld::new(` outside such constructors, counted over every
/// function the test reaches. A constructor's own body is one construction.
fn world_constructions(raw: &str, test: &str) -> usize {
    let code = blank_literals(raw);
    let items = fn_items(&code);
    let takes_world = |f: &FnItem| {
        let sig = &f.signature;
        sig.find('(')
            .map(|open| &sig[open..matching(sig, open)])
            .is_some_and(|params| params.contains("PhysicsWorld"))
    };
    let constructors: BTreeSet<&String> = items
        .iter()
        .filter(|(_, f)| return_type(&f.signature) == "PhysicsWorld" && !takes_world(f))
        .map(|(name, _)| name)
        .collect();
    let calls = |body: &str, callee: &str| {
        let b = body.as_bytes();
        body.match_indices(callee)
            .filter(|&(i, _)| {
                (i == 0 || !is_ident(b[i - 1])) && body[i + callee.len()..].starts_with('(')
            })
            .count()
    };
    reachable_fns(&items, test)
        .iter()
        .filter(|name| !constructors.contains(name))
        .map(|name| {
            let body = &items[name].body;
            body.matches("PhysicsWorld::new(").count()
                + constructors.iter().map(|c| calls(body, c)).sum::<usize>()
        })
        .sum()
}

/// Whether `test` calls `assert_stepped_with(..)` with the backend of `combo`
/// and one of `digests`, and the file defines that helper so that it asserts
/// the backend of the world it is given.
fn stepped_with(
    raw: &str,
    test: &str,
    combo: &str,
    digests: &BTreeSet<String>,
) -> Result<(), String> {
    const HELPER: &str = "assert_stepped_with";
    let code = blank_literals(raw);
    let items = fn_items(&code);
    let helper_ok = items
        .get(HELPER)
        .is_some_and(|f| f.body.contains(".config.solver_backend") && f.body.contains("assert"));
    if !helper_ok {
        return Err(format!(
            "the file has no {HELPER} that asserts `.config.solver_backend`"
        ));
    }
    let backend = combo.split_once(' ').map_or(combo, |(b, _)| b);
    let body = items.get(test).map(|f| f.body.as_str()).unwrap_or("");
    let called = body.match_indices(&format!("{HELPER}(")).any(|(i, m)| {
        let b = body.as_bytes();
        if i > 0 && is_ident(b[i - 1]) {
            return false;
        }
        let open = i + m.len() - 1;
        let args = &body[open..matching(body, open)];
        mentions_ident(args, &format!("SolverBackend::{backend}"))
            && digests.iter().any(|d| mentions_ident(args, d))
    });
    if called {
        Ok(())
    } else {
        Err(format!(
            "does not call {HELPER}(.., SolverBackend::{backend}, <digest of the named row>)"
        ))
    }
}

/// Checks one `RELATION_PINS` row against the `PINS` combinations already
/// pinned (`pinned`).
fn check_relation(
    req: &Required,
    (combo, same_as, file, test, feature): (&str, &str, &str, &str, &str),
    read_test: &dyn Fn(&str) -> String,
    pinned: &BTreeSet<String>,
) -> Result<(), String> {
    let row = format!("RELATION_PINS row {combo:?}: {file}::{test}");
    if !req.combos.contains(combo) {
        return Err(format!("{row}: not a required combination"));
    }
    if pinned.contains(combo) {
        return Err(format!("{row}: it is also a PINS row"));
    }
    let Some(&(_, same_file, same_test, _)) = PINS.iter().find(|r| r.0 == same_as) else {
        return Err(format!("{row}: {same_as:?} is not a PINS row"));
    };
    if !pinned.contains(same_as) {
        return Err(format!("{row}: {same_as:?} is not pinned"));
    }
    let t = test_facts(&read_test(file), test, &req.entries);
    if !t.active {
        return Err(format!("{row} is not an active #[test]"));
    }
    let want: BTreeSet<String> = if feature.is_empty() {
        BTreeSet::new()
    } else {
        BTreeSet::from([feature.to_string()])
    };
    if t.features != want {
        return Err(format!(
            "{row} is behind features {:?}, the row says {want:?}",
            t.features
        ));
    }
    exercises(req, &t, combo).map_err(|why| format!("{row} {why}"))?;
    // The relation is "this backend gives the other backend's bits": a test
    // that also builds a world with the other backend could assert that
    // world's hash instead, so naming the other backend refuses the row.
    let backend = |c: &str| c.split_once(' ').map(|(b, _)| b.to_string());
    if let (Some(own), Some(other)) = (backend(combo), backend(same_as)) {
        if own != other && mentions_ident(&t.text, &format!("SolverBackend::{other}")) {
            return Err(format!(
                "{row} names SolverBackend::{other}, the backend whose bits it claims to give"
            ));
        }
    }
    // A second world (built with the other backend, by name or by default)
    // could supply the asserted hash instead of the stepped one.
    let worlds = world_constructions(&read_test(file), test);
    if worlds != 1 {
        return Err(format!(
            "{row} builds {worlds} worlds; a relation test steps exactly one"
        ));
    }
    let same = test_facts(&read_test(same_file), same_test, &req.entries);
    if file != same_file || t.asserted.is_disjoint(&same.asserted) {
        return Err(format!(
            "{row} asserts {:?}, not a digest constant of {same_file}::{same_test} ({:?})",
            t.asserted, same.asserted
        ));
    }
    // Runtime evidence: the test hands the stepped world to
    // `assert_stepped_with`, which asserts its backend and its digest on the
    // same value, naming its own backend and the digest of the named row.
    stepped_with(&read_test(file), test, combo, &same.asserted)
        .map_err(|why| format!("{row} {why}"))?;
    Ok(())
}

fn check(req: &Required, read_test: &dyn Fn(&str) -> String) -> (Vec<String>, BTreeSet<String>) {
    let mut errors = Vec::new();
    for m in &req.unclassified {
        errors.push(format!(
            "PhysicsWorld::{m} takes &mut self and is in neither STEP_ENTRIES nor NOT_STEP: classify it"
        ));
    }
    for m in &req.stale_classification {
        errors.push(format!(
            "{m} is classified but is no PhysicsWorld method taking &mut self"
        ));
    }
    let mut pinned = BTreeSet::new();
    for &(combo, file, test, feature) in PINS {
        if !req.combos.contains(combo) {
            errors.push(format!("PINS row {combo:?}: not a required combination"));
            continue;
        }
        let t = test_facts(&read_test(file), test, &req.entries);
        let row = format!("PINS row {combo:?}: {file}::{test}");
        if !t.active {
            errors.push(format!("{row} is not an active #[test]"));
            continue;
        }
        let want: BTreeSet<String> = if feature.is_empty() {
            BTreeSet::new()
        } else {
            BTreeSet::from([feature.to_string()])
        };
        if t.features != want {
            errors.push(format!(
                "{row} is behind features {:?}, the row says {want:?}",
                t.features
            ));
        }
        if !t.asserts_digest {
            errors.push(format!("{row} asserts no digest constant"));
        }
        match exercises(req, &t, combo) {
            Ok(()) => {
                pinned.insert(combo.to_string());
            }
            Err(why) => errors.push(format!("{row} {why}")),
        }
    }
    for &row in RELATION_PINS {
        match check_relation(req, row, read_test, &pinned) {
            Ok(()) => {
                pinned.insert(row.0.to_string());
            }
            Err(why) => errors.push(why),
        }
    }
    let gaps: BTreeSet<String> = KNOWN_GAPS.iter().map(|(g, _)| (*g).to_string()).collect();
    if gaps.len() != KNOWN_GAPS.len() {
        errors.push("a KNOWN_GAPS row is listed twice".into());
    }
    for g in &gaps {
        if !req.combos.contains(g) {
            errors.push(format!("KNOWN_GAPS row {g:?}: not a required combination"));
        }
        if pinned.contains(g) {
            errors.push(format!(
                "KNOWN_GAPS row {g:?}: it is pinned (remove it from KNOWN_GAPS)"
            ));
        }
    }
    for c in &req.combos {
        if !pinned.contains(c) && !gaps.contains(c) {
            errors.push(format!(
                "{c:?}: no golden pins it and it is not in KNOWN_GAPS"
            ));
        }
    }
    (errors, pinned)
}

#[test]
fn every_stepping_combination_has_a_golden_pin_or_is_a_listed_gap() {
    let req = required_from(&solver_sources());
    assert!(
        req.backends.len() >= 2,
        "backends read from src: {:?}",
        req.backends
    );
    assert!(
        req.entries.contains("step"),
        "`step` not found among {:?}",
        req.entries
    );
    let (errors, pinned) = check(&req, &|file| read(&format!("tests/{file}")));
    assert!(
        !pinned.is_empty(),
        "no PINS row matched: the check compared nothing"
    );
    assert!(
        errors.is_empty(),
        "golden coverage ({} required, {} pinned, {} listed gaps):\n  {}",
        req.combos.len(),
        pinned.len(),
        KNOWN_GAPS.len(),
        errors.join("\n  ")
    );
}

#[test]
#[ignore = "src gap: stepping combinations without a determinism golden are listed in KNOWN_GAPS of this file"]
fn golden_coverage_has_no_gaps() {
    assert!(
        KNOWN_GAPS.is_empty(),
        "combinations without a golden: {KNOWN_GAPS:?}"
    );
}

// ── Tests of the checker itself ─────────────────────────────────────────────

/// A synthetic solver source with the real enums and the given methods in an
/// `impl PhysicsWorld` block.
#[test]
fn a_relation_row_must_assert_the_digest_of_the_row_it_names() {
    let req = required_from(&solver_sources());
    let pinned = BTreeSet::from(["Xpbd step_parallel".to_string()]);
    let file = "const GOLDEN_XPBD_STEP_PARALLEL: &str = \"7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39\";\n\
                const GOLDEN_OTHER: &str = \"cfb9f3e814a4b0a55c96019f1df345eb4aaba233fa5321e021dbcb916ff730fe\";\n\
                fn scene(b: SolverBackend) -> PhysicsWorld { PhysicsWorld::new(b) }\n\
                fn assert_stepped_with(s: &str, w: &PhysicsWorld, b: SolverBackend, e: &str) { assert_eq!(w.config.solver_backend, b); assert_eq!(hash(w), e); }\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn golden_path_xpbd_step_parallel() {\n    let mut w = scene(SolverBackend::Xpbd);\n    w.step_parallel(dt);\n    assert_eq!(hash(&w), GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn same_bits() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    assert_stepped_with(\"t\", &w, SolverBackend::Tgs, GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn unbound() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    assert_eq!(hash(&w), GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn other_digest() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    assert_eq!(hash(&w), GOLDEN_OTHER);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn two_worlds() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    let mut x = scene(SolverBackend::Xpbd);\n    x.step_parallel(dt);\n    assert_eq!(hash(&x), GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn default_world() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    let mut x = scene(SolverBackend::default());\n    x.step_parallel(dt);\n    assert_eq!(hash(&x), GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn new_world() {\n    let mut w = scene(SolverBackend::Tgs);\n    w.step_parallel(dt);\n    let mut x = PhysicsWorld::new(Default::default());\n    x.step_parallel(dt);\n    assert_eq!(hash(&x), GOLDEN_XPBD_STEP_PARALLEL);\n}\n\
                #[cfg(feature = \"parallel\")]\n#[test]\nfn wrong_backend() {\n    let mut w = scene(SolverBackend::Xpbd);\n    w.step_parallel(dt);\n    assert_eq!(hash(&w), GOLDEN_XPBD_STEP_PARALLEL);\n}\n";
    let read_test = |_: &str| file.to_string();
    let row = |test| {
        (
            "Tgs step_parallel",
            "Xpbd step_parallel",
            "determinism_golden_paths.rs",
            test,
            "parallel",
        )
    };
    assert_eq!(
        check_relation(&req, row("same_bits"), &read_test, &pinned),
        Ok(())
    );
    let other = check_relation(&req, row("other_digest"), &read_test, &pinned);
    assert!(
        other
            .as_ref()
            .is_err_and(|e| e.contains("not a digest constant")),
        "{other:?}"
    );
    let backend = check_relation(&req, row("wrong_backend"), &read_test, &pinned);
    assert!(
        backend
            .as_ref()
            .is_err_and(|e| e.contains("SolverBackend::Tgs")),
        "{backend:?}"
    );
    let two_worlds = check_relation(&req, row("two_worlds"), &read_test, &pinned);
    assert!(
        two_worlds
            .as_ref()
            .is_err_and(|e| e.contains("names SolverBackend::Xpbd")),
        "{two_worlds:?}"
    );
    for second in ["default_world", "new_world"] {
        let got = check_relation(&req, row(second), &read_test, &pinned);
        assert!(
            got.as_ref().is_err_and(|e| e.contains("builds 2 worlds")),
            "{second}: {got:?}"
        );
    }
    let unbound = check_relation(&req, row("unbound"), &read_test, &pinned);
    assert!(
        unbound
            .as_ref()
            .is_err_and(|e| e.contains("does not call assert_stepped_with")),
        "{unbound:?}"
    );
    let unpinned = check_relation(&req, row("same_bits"), &read_test, &BTreeSet::new());
    assert!(
        unpinned
            .as_ref()
            .is_err_and(|e| e.contains("is not pinned")),
        "{unpinned:?}"
    );
}

fn synthetic_solver(methods: &[&str]) -> Vec<String> {
    let real = read("src/solver.rs");
    let code = blank_literals(&real);
    let enum_src = |name: &str| {
        let s = code.find(&format!("pub enum {name} {{")).expect("enum");
        let e = s + code[s..].find("\n}").expect("end") + 2;
        real[s..e].to_string()
    };
    let mut src = format!(
        "{}\n{}\nimpl PhysicsWorld {{\n",
        enum_src("SolverBackend"),
        enum_src("Broadphase")
    );
    for m in methods {
        src.push_str(&format!("    pub fn {m}(&mut self, dt: Fix128) {{}}\n"));
    }
    src.push_str("    pub fn body_count(&self) -> usize { 0 }\n}\n");
    vec![src]
}

#[test]
fn a_new_mut_method_must_be_classified_whatever_its_name() {
    let mut methods: Vec<&str> = STEP_ENTRIES.iter().chain(NOT_STEP).copied().collect();
    methods.push("substep_twice");
    methods.push("advance");
    let req = required_from(&synthetic_solver(&methods));
    assert_eq!(req.unclassified, ["advance", "substep_twice"]);
    // `&self` methods are not stepping entry points and need no classification
    assert!(!req.unclassified.iter().any(|m| m == "body_count"));
    // generic methods are seen through their parameter list
    let src = "impl PhysicsWorld {\n    pub fn step_with_bridge<B: Bridge + ?Sized>(\n        &mut self,\n        b: &mut B,\n    ) {}\n}\n";
    assert!(world_mut_methods(src).contains("step_with_bridge"));
    // a path-qualified impl in another module counts; another type does not
    assert!(world_mut_methods(
        "impl crate::solver::PhysicsWorld {\n    pub fn advance_elsewhere(&mut self) {}\n}\n"
    )
    .contains("advance_elsewhere"));
    assert!(
        world_mut_methods("impl PyPhysicsWorld {\n    pub fn step(&mut self) {}\n}\n").is_empty()
    );
    assert!(
        world_mut_methods("impl Default for PhysicsWorld {\n    pub fn x(&mut self) {}\n}\n")
            .is_empty()
    );
    // a method of another type is not a PhysicsWorld method
    assert!(
        world_mut_methods("impl Cloth {\n    pub fn step(&mut self, dt: F) {}\n}\n").is_empty()
    );
}

#[test]
fn a_call_on_another_type_does_not_pin_a_world_entry() {
    let req = required_from(&solver_sources());
    let entries = req.entries.clone();
    let file = "const GOLDEN_X: &str = \"95d1f0805b7b5b2cae4030ba7bd74749cfe7bc60869d1478eeeac67fab27d381\";\n\
                #[test]\nfn cloth_case() {\n    let mut cloth = Cloth::new(cfg);\n    cloth.step(dt);\n    assert_golden(\"c\", hash(&cloth), GOLDEN_X);\n}\n\
                #[test]\nfn world_case() {\n    let mut w = PhysicsWorld::new(cfg);\n    w.step(dt);\n    assert_golden(\"w\", hash(&w), GOLDEN_X);\n}\n\
                fn make() -> PhysicsWorld { PhysicsWorld::new(c) }\n\
                fn run(mut w: PhysicsWorld) -> PhysicsWorld { w.step_n(3, dt); w }\n\
                #[test]\nfn helper_case() {\n    let w = run(make());\n    assert_eq!(hash(&w), GOLDEN_X);\n}\n";
    let cloth = test_facts(file, "cloth_case", &entries);
    assert!(
        exercises(&req, &cloth, "Xpbd step").is_err(),
        "cloth.step( is not a world step"
    );
    let world = test_facts(file, "world_case", &entries);
    assert!(exercises(&req, &world, "Xpbd step").is_ok());
    let helper = test_facts(file, "helper_case", &entries);
    assert!(
        exercises(&req, &helper, "Xpbd step_n").is_ok(),
        "parameter of type PhysicsWorld"
    );
    assert!(helper.asserts_digest);
}

#[test]
fn backends_are_told_apart_and_a_loop_over_both_pins_both() {
    let req = required_from(&solver_sources());
    let entries = req.entries.clone();
    let both = "const G: &str = \"95d1f0805b7b5b2cae4030ba7bd74749cfe7bc60869d1478eeeac67fab27d381\";\n\
                #[test]\nfn both() {\n    for b in [SolverBackend::Xpbd, SolverBackend::Tgs] {\n        let mut w = PhysicsWorld::new(cfg(b));\n        w.step(dt);\n        assert_eq!(hash(&w), G);\n    }\n}\n\
                #[test]\nfn tgs_only() {\n    let mut w = PhysicsWorld::new(PhysicsConfig { solver_backend: SolverBackend::Tgs, ..d });\n    w.step(dt);\n    assert_eq!(hash(&w), G);\n}\n\
                mod common;\n#[test]\nfn foreign() {\n    let mut w = common::world();\n    let mut v = PhysicsWorld::new(d);\n    v.step(dt);\n    assert_eq!(hash(&v), G);\n}\n";
    let t = test_facts(both, "both", &entries);
    assert!(exercises(&req, &t, "Xpbd step").is_ok() && exercises(&req, &t, "Tgs step").is_ok());
    let t = test_facts(both, "tgs_only", &entries);
    assert!(exercises(&req, &t, "Tgs step").is_ok());
    assert!(
        exercises(&req, &t, "Xpbd step").is_err(),
        "naming Tgs is not the default"
    );
    let t = test_facts(both, "foreign", &entries);
    assert!(
        exercises(&req, &t, "Xpbd step").is_err(),
        "a helper of another module hides the backend"
    );
}

#[test]
fn the_source_scanner_sees_code_and_skips_literals() {
    let src = "fn a() { let s = \"} .step( {\"; let c = '{'; // .try_step(\n b(); }\n\
               fn b() { w.step(dt); /* } */ }\nfn c() { r#\"}\"#; }\n\
               #[test]\nfn t() { a(); }\n#[test]\n#[ignore = \"x\"]\nfn u() {}\n";
    let code = blank_literals(src);
    assert_eq!(code.len(), src.len());
    let items = fn_items(&code);
    assert_eq!(
        items.keys().cloned().collect::<Vec<_>>(),
        ["a", "b", "c", "t", "u"]
    );
    assert_eq!(reachable_fns(&items, "t"), ["t", "a", "b"]);
    assert!(items["u"].attrs.contains("#[ignore"));
    let raw = "const A: &str =\n    \"95d1f0805b7b5b2cae4030ba7bd74749cfe7bc60869d1478eeeac67fab27d381\";\n\
               const B: u64 = 0x545b_6a9d_803d_31ef;\nconst C: usize = 180;\nconst D: &str = \"abc\";\n";
    let consts = digest_consts(raw);
    assert_eq!(consts, BTreeSet::from(["A".to_string(), "B".to_string()]));
    // the digest has to be asserted, not only named
    assert!(asserts_digest(
        "let h = hash(&w); assert_eq!(h, B)",
        &consts
    ));
    assert!(!asserts_digest("println!(\"{}\", A); let x = B", &consts));
    assert!(mentions_ident("x(GOLDEN_A)", "GOLDEN_A") && !mentions_ident("GOLDEN_AB", "GOLDEN_A"));
    let (v, d) = enum_variants(
        "pub enum SolverBackend {\n    #[default]\n    Xpbd,\n    Tgs,\n    /// new\n    Probe { x: u8 },\n}\n",
        "SolverBackend",
    );
    assert_eq!(
        (v, d.as_str()),
        (
            vec!["Xpbd".to_string(), "Tgs".to_string(), "Probe".to_string()],
            "Xpbd"
        )
    );
}

#[test]
fn the_feature_of_a_gated_golden_is_read() {
    let raw = "#[cfg(feature = \"parallel\")]\n#[test]\nfn g() {}\n#[test]\nfn h() {}\n";
    assert_eq!(
        fn_items_attrs_raw(raw, "g"),
        BTreeSet::from(["parallel".to_string()])
    );
    assert!(fn_items_attrs_raw(raw, "h").is_empty());
}
