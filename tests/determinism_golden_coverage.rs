//! Which stepping paths the determinism goldens pin, checked against the code.
//!
//! The goldens in `tests/determinism_*.rs` pin the bits of a world after it
//! has been stepped. A path that no golden runs can change its results
//! without any golden noticing, so the set of pinned paths has to cover every
//! way the crate steps a world. This file lists that set and checks it:
//!
//! * the solver backends are read from `pub enum SolverBackend` in
//!   `src/solver.rs`, and the stepping entry points (`step`, `step_n`,
//!   `try_step`, `step_parallel`, `try_step_parallel`, `step_with_bridge`, ...) from the
//!   `pub fn` items of `src/solver.rs` and `src/solver/*.rs`, so a new
//!   backend or entry point becomes a new required combination without
//!   editing this file;
//! * the scene features below (contacts, joints, continuous collision,
//!   sleeping, each broadphase kind, participants, cloth, the 2D world) are
//!   listed here by hand;
//! * every row of [`PINS`] names a golden test and is checked mechanically:
//!   the test exists, is not ignored, compares against a constant digest
//!   (a SHA-256 hex string or a `u64` hex constant) and, in its own body or
//!   in the helpers of the same file it calls, uses the backend, entry point
//!   or feature the row claims;
//! * every required combination is either pinned or listed in
//!   [`KNOWN_GAPS`], never both, and both tables name only required
//!   combinations.
//!
//! `golden_coverage_has_no_gaps` is ignored while [`KNOWN_GAPS`] is not empty:
//! it fails until every combination has a golden. Adding a golden for a gap
//! means moving its row from [`KNOWN_GAPS`] to [`PINS`]; the main test fails
//! if a listed gap turns out to be pinned or a pin stops matching its row.
//!
//! The check reads source text. It decides "uses `SolverBackend::Tgs`" or
//! "calls `.try_step(`" from the code of the test and its same-file helpers,
//! with string literals and comments blanked first; it does not run the
//! tests, and a helper in another file is not followed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A golden that pins a combination: (combination, file under `tests/`, test fn).
const PINS: &[(&str, &str, &str)] = &[
    ("Xpbd step", "determinism_golden.rs", "determinism_freefall"),
    (
        "Xpbd step_parallel",
        "determinism_golden_contacts.rs",
        "determinism_contact_bounce",
    ),
    (
        "contacts",
        "determinism_golden_contacts.rs",
        "determinism_contact_stack",
    ),
    (
        "distance constraint",
        "determinism_golden.rs",
        "determinism_joint_pendulum",
    ),
    (
        "broadphase Bvh",
        "determinism_golden.rs",
        "determinism_cascade",
    ),
    ("cloth", "determinism_golden.rs", "determinism_cloth_drape"),
    (
        "physics2d",
        "determinism_physics2d_step_digest.rs",
        "step_digest_is_unchanged",
    ),
];

/// Required combinations that no golden pins yet.
const KNOWN_GAPS: &[&str] = &[
    "Xpbd step_n",
    "Xpbd try_step",
    "Xpbd try_step_parallel",
    "Xpbd step_with_bridge",
    "Tgs step",
    "Tgs step_n",
    "Tgs try_step",
    "Tgs step_parallel",
    "Tgs try_step_parallel",
    "Tgs step_with_bridge",
    "joint",
    "continuous collision",
    "sleeping",
    "broadphase DynamicTree",
    "broadphase Hybrid",
    "participant",
];

/// Scene features every stepping law must have a golden for, with the code
/// that shows a test exercises them (checked on comment- and string-free text).
const FEATURES: &[(&str, &[&str])] = &[
    ("contacts", &["add_body_with_radius("]),
    ("distance constraint", &["add_distance_constraint("]),
    ("joint", &["add_joint("]),
    ("continuous collision", &["set_continuous_collision("]),
    // sleeping must be reachable: a scene that sets `frames_to_sleep` to
    // `u32::MAX` or turns the skip off is rejected in `feature_holds`
    ("sleeping", &["is_sleeping(", ".sleeping"]),
    ("participant", &["add_participant("]),
    ("cloth", &["Cloth::"]),
    ("physics2d", &["PhysicsWorld2D"]),
];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
        .replace("\r\n", "\n")
}

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
                && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) =>
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

/// Every `fn` item of a blanked source: name -> body (between the braces).
fn fn_bodies(code: &str) -> BTreeMap<String, String> {
    let b = code.as_bytes();
    let mut out = BTreeMap::new();
    let mut from = 0;
    while let Some(n) = code[from..].find("fn ") {
        let at = from + n;
        from = at + 3;
        if at > 0 && is_ident(b[at - 1]) {
            continue;
        }
        let name: String = code[at + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let Some(open_rel) = code[at..].find(['{', ';']) else {
            break;
        };
        let open = at + open_rel;
        if b[open] == b';' {
            continue; // a declaration without a body
        }
        let mut depth = 0usize;
        let mut end = open;
        for (k, &c) in b[open..].iter().enumerate() {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.entry(name)
            .or_insert_with(|| code[open + 1..end].to_string());
    }
    out
}

/// Names of the `#[test]` fns that carry no `#[ignore]`.
fn active_tests(code: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut from = 0;
    while let Some(n) = code[from..].find("#[test]") {
        let at = from + n + "#[test]".len();
        from = at;
        let Some(f) = code[at..].find("fn ") else {
            break;
        };
        let attrs = &code[at..at + f];
        if attrs.contains("#[ignore") {
            continue;
        }
        let name: String = code[at + f + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        out.insert(name);
    }
    out
}

/// The body of `test` and of every same-file fn it calls, transitively.
fn reachable(bodies: &BTreeMap<String, String>, test: &str) -> String {
    let mut seen = BTreeSet::from([test.to_string()]);
    let mut stack = vec![test.to_string()];
    let mut text = String::new();
    while let Some(name) = stack.pop() {
        let Some(body) = bodies.get(&name) else {
            continue;
        };
        text.push_str(body);
        text.push('\n');
        let bb = body.as_bytes();
        for (i, _) in body.match_indices('(') {
            let start = bb[..i]
                .iter()
                .rposition(|&c| !is_ident(c))
                .map_or(0, |p| p + 1);
            let callee = &body[start..i];
            if bodies.contains_key(callee) && seen.insert(callee.to_string()) {
                stack.push(callee.to_string());
            }
        }
    }
    text
}

/// Constant digests of a raw source: names of `const X: &str = "<64 hex>"`
/// and `const X: u64 = 0x...` items.
fn digest_consts(raw: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for item in raw.split("const ").skip(1) {
        let name: String = item
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
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
            let v: String = t
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
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

/// `pub fn` names in `src` that step a world: `step`, `step_*`, `try_step`, `try_step_*`
/// taking `&mut self` (the stepping entry points of `PhysicsWorld`).
fn step_entries(src: &str) -> BTreeSet<String> {
    let code = blank_literals(src);
    let mut out = BTreeSet::new();
    for (i, _) in code.match_indices("pub fn ") {
        let rest = &code[i + "pub fn ".len()..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let sig = &rest[..rest.find('{').unwrap_or(rest.len())];
        let stepping = name == "step"
            || name == "try_step"
            || name.starts_with("step_")
            || name.starts_with("try_step_");
        if stepping && sig.contains("&mut self") {
            out.insert(name);
        }
    }
    out
}

struct Required {
    combos: BTreeSet<String>,
    backends: Vec<String>,
    default_backend: String,
    broadphases: Vec<String>,
    default_broadphase: String,
    entries: BTreeSet<String>,
}

fn required() -> Required {
    let solver = read("src/solver.rs");
    let (backends, default_backend) = enum_variants(&solver, "SolverBackend");
    let (broadphases, default_broadphase) = enum_variants(&solver, "Broadphase");
    let mut entries = step_entries(&solver);
    let mut dir: Vec<_> = std::fs::read_dir(root().join("src/solver"))
        .expect("src/solver")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    dir.sort();
    for p in dir {
        let rel = p
            .strip_prefix(root())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        entries.extend(step_entries(&read(&rel)));
    }
    let mut combos = BTreeSet::new();
    for b in &backends {
        for e in &entries {
            combos.insert(format!("{b} {e}"));
        }
    }
    for k in &broadphases {
        combos.insert(format!("broadphase {k}"));
    }
    for (f, _) in FEATURES {
        combos.insert((*f).to_string());
    }
    Required {
        combos,
        backends,
        default_backend,
        broadphases,
        default_broadphase,
        entries,
    }
}

/// Whether the code a test reaches exercises `combo`; the reason when it does not.
fn exercises(req: &Required, text: &str, combo: &str) -> Result<(), String> {
    let uses_backend = |b: &str| mentions_ident(text, &format!("SolverBackend::{b}"));
    let uses_broadphase = |k: &str| mentions_ident(text, &format!("Broadphase::{k}"));
    if let Some(kind) = combo.strip_prefix("broadphase ") {
        let others = req
            .broadphases
            .iter()
            .any(|k| k != kind && uses_broadphase(k));
        return if kind == req.default_broadphase {
            if others {
                Err(format!("selects another broadphase than {kind}"))
            } else {
                Ok(())
            }
        } else if uses_broadphase(kind) {
            Ok(())
        } else {
            Err(format!("does not select Broadphase::{kind}"))
        };
    }
    if let Some((_, needles)) = FEATURES.iter().find(|(f, _)| *f == combo) {
        if !needles.iter().any(|n| text.contains(n)) {
            return Err(format!("does not use any of {needles:?}"));
        }
        if combo == "sleeping"
            && (text.contains("frames_to_sleep: u32::MAX")
                || text.contains("set_sleep_skip(false)"))
        {
            return Err("disables sleeping".into());
        }
        return Ok(());
    }
    let (backend, entry) = combo
        .split_once(' ')
        .ok_or_else(|| format!("unknown combination {combo:?}"))?;
    if !text.contains(&format!(".{entry}(")) {
        return Err(format!("does not call .{entry}("));
    }
    let others = req.backends.iter().any(|b| b != backend && uses_backend(b));
    if backend == req.default_backend {
        if others {
            Err(format!("selects another backend than {backend}"))
        } else {
            Ok(())
        }
    } else if uses_backend(backend) {
        Ok(())
    } else {
        Err(format!("does not select SolverBackend::{backend}"))
    }
}

#[test]
fn every_stepping_combination_has_a_golden_pin_or_is_a_listed_gap() {
    let req = required();
    assert!(
        req.backends.len() >= 2,
        "backends read from src: {:?}",
        req.backends
    );
    assert!(
        req.entries.len() >= 3,
        "entry points read from src: {:?}",
        req.entries
    );
    assert!(
        req.entries.contains("step"),
        "`step` not found among {:?}",
        req.entries
    );

    let mut errors = Vec::new();
    let mut pinned = BTreeSet::new();
    let mut files: BTreeMap<&str, (String, String)> = BTreeMap::new();
    for &(combo, file, test) in PINS {
        if !req.combos.contains(combo) {
            errors.push(format!("PINS row {combo:?}: not a required combination"));
            continue;
        }
        let (raw, code) = files.entry(file).or_insert_with(|| {
            let raw = read(&format!("tests/{file}"));
            let code = blank_literals(&raw);
            (raw, code)
        });
        let bodies = fn_bodies(code);
        if !active_tests(code).contains(test) {
            errors.push(format!(
                "PINS row {combo:?}: {file}::{test} is not an active #[test]"
            ));
            continue;
        }
        let text = reachable(&bodies, test);
        let consts = digest_consts(raw);
        if !consts.iter().any(|c| mentions_ident(&text, c)) {
            errors.push(format!(
                "PINS row {combo:?}: {file}::{test} compares against no digest constant"
            ));
        }
        if !text.contains("assert") {
            errors.push(format!(
                "PINS row {combo:?}: {file}::{test} asserts nothing"
            ));
        }
        match exercises(&req, &text, combo) {
            Ok(()) => {
                pinned.insert(combo.to_string());
            }
            Err(why) => errors.push(format!("PINS row {combo:?}: {file}::{test} {why}")),
        }
    }
    let gaps: BTreeSet<String> = KNOWN_GAPS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        gaps.len(),
        KNOWN_GAPS.len(),
        "a KNOWN_GAPS row is listed twice"
    );
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
    assert!(
        !pinned.is_empty(),
        "no PINS row matched: the check compared nothing"
    );
    assert!(
        errors.is_empty(),
        "golden coverage ({} required, {} pinned, {} listed gaps):\n  {}",
        req.combos.len(),
        pinned.len(),
        gaps.len(),
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

#[test]
fn the_source_scanner_sees_code_and_skips_literals() {
    // braces and calls inside strings, chars and comments are not code
    let src = "fn a() { let s = \"} .step( {\"; let c = '{'; // .try_step(\n b(); }\n\
               fn b() { w.step(dt); /* } */ }\nfn c() { r#\"}\"#; }\n\
               #[test]\nfn t() { a(); }\n#[test]\n#[ignore = \"x\"]\nfn u() {}\n";
    let code = blank_literals(src);
    assert_eq!(code.len(), src.len());
    let bodies = fn_bodies(&code);
    assert_eq!(
        bodies.keys().cloned().collect::<Vec<_>>(),
        ["a", "b", "c", "t", "u"]
    );
    let text = reachable(&bodies, "t");
    assert!(text.contains(".step("), "helper b is followed: {text}");
    assert!(!text.contains(".try_step("), "a comment is not code");
    assert_eq!(active_tests(&code), BTreeSet::from(["t".to_string()]));
    // digest constants of both forms
    let raw = "const A: &str =\n    \"95d1f0805b7b5b2cae4030ba7bd74749cfe7bc60869d1478eeeac67fab27d381\";\n\
               const B: u64 = 0x545b_6a9d_803d_31ef;\nconst C: usize = 180;\nconst D: &str = \"abc\";\n";
    assert_eq!(
        digest_consts(raw),
        BTreeSet::from(["A".to_string(), "B".to_string()])
    );
    assert!(mentions_ident("x(GOLDEN_A)", "GOLDEN_A") && !mentions_ident("GOLDEN_AB", "GOLDEN_A"));
}

#[test]
fn a_new_backend_variant_or_entry_point_is_read_as_a_new_requirement() {
    // adding a variant to the real enum stops the crate from compiling (its
    // matches are exhaustive), so the reader is checked on a synthetic source
    let src = "pub enum SolverBackend {\n    /// doc\n    #[default]\n    Xpbd,\n    Tgs,\n    /// new\n    Probe { x: u8 },\n}\n";
    let (v, d) = enum_variants(src, "SolverBackend");
    assert_eq!(v, ["Xpbd", "Tgs", "Probe"]);
    assert_eq!(d, "Xpbd");
    let entries = step_entries("impl W {\n    pub fn step(&mut self, dt: F) {}\n    pub fn try_step_rollback(&mut self) -> R {}\n    pub fn step_count(&self) -> u64 { 0 }\n    pub fn stepper() {}\n}\n");
    assert_eq!(
        entries,
        BTreeSet::from(["step".to_string(), "try_step_rollback".to_string()])
    );
}

#[test]
fn backends_entries_and_broadphases_are_read_from_the_source() {
    let req = required();
    assert!(req.backends.iter().any(|b| b == "Xpbd") && req.backends.iter().any(|b| b == "Tgs"));
    assert_eq!(req.default_backend, "Xpbd");
    assert_eq!(req.default_broadphase, "Bvh");
    for e in [
        "step",
        "step_n",
        "try_step",
        "step_parallel",
        "try_step_parallel",
    ] {
        assert!(
            req.entries.contains(e),
            "{e} not read from src: {:?}",
            req.entries
        );
    }
    // a selected non-default backend is told apart from the default
    let tgs = "let c = SolverConfig { solver_backend: SolverBackend::Tgs, ..d }; w.step(dt);";
    assert!(exercises(&req, tgs, "Tgs step").is_ok());
    assert!(exercises(&req, tgs, "Xpbd step").is_err());
    assert!(exercises(&req, "w.step(dt);", "Xpbd step").is_ok());
    assert!(exercises(&req, "w.step(dt);", "Xpbd step_n").is_err());
    let no_sleep =
        "w.set_sleep_config(SleepConfig { frames_to_sleep: u32::MAX, ..d }); w.is_sleeping(0);";
    assert!(exercises(&req, no_sleep, "sleeping").is_err());
}
