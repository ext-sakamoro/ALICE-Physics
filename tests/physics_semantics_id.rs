//! `PHYSICS_SEMANTICS_ID` and its table, checked against the goldens.
//!
//! `alice_physics::PHYSICS_SEMANTICS_PINS` holds one SHA-256 digest per
//! stepping combination the determinism goldens pin, plus the semantics
//! identifier of `alice-det-math`; `PHYSICS_SEMANTICS_ID` is their fold.
//! This file checks, on the source of the goldens (no value is copied here):
//!
//! * every entry equals the digest constant asserted by the golden test that
//!   the `PINS` table of `tests/determinism_golden_coverage.rs` names for it;
//! * the entry names are exactly the `PINS` combinations plus
//!   `"alice-det-math"` (the known gaps are not entries);
//! * the det-math entry equals `alice_det_math::SEMANTICS_ID`;
//! * the fold of the table is `PHYSICS_SEMANTICS_ID`, and both equal the hex
//!   literal pinned here. The table and the identifier are not behind any
//!   `cfg`, and CI runs this file with and without `parallel` and
//!   `gpu-solver-bridge`, so the value is the same in every build;
//! * the fold separates what it should (one changed bit, length-prefix
//!   collisions) and ignores what it should (entry order).
//!
//! # Fold
//!
//! The `SEMANTICS_ID` fold of `alice-det-math`: entries sorted by name,
//! an empty table, a repeated name, an empty name or a non-ASCII name is
//! rejected, then SHA-256 over `u32` big-endian name length ‖ name ‖ 32
//! digest bytes for each entry. A known-answer vector computed outside this
//! crate pins the byte layout (`the_fold_matches_a_known_answer`).
//!
//! # How a digest is located
//!
//! As in the coverage check: comments and string literals are blanked, the
//! `fn` items of the golden file are collected, the functions the named test
//! reaches (same file, transitively) are followed, and a digest constant
//! (`const X: &str = "<64 hex digits>"`, or `const X: u64 = 0x<16 hex>`)
//! counts when it is named inside a statement containing `assert`. Exactly
//! one 64-digit constant must be asserted. A test that asserts only a `u64`
//! constant (`physics2d`) resolves to the one other test of the same file
//! that asserts that same `u64` constant and exactly one 64-digit constant,
//! so the SHA-256 entry is tied to the state the `u64` pin already fixes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use alice_physics::{PHYSICS_SEMANTICS_ID, PHYSICS_SEMANTICS_PINS};
use sha2::{Digest, Sha256};

/// The identifier, pinned. A change here is a change of the stepping
/// semantics (a golden moved, a gap was filled, or det-math changed).
const PINNED_ID_HEX: &str = "574e7586cbdf75bb3ed2fb0f304ceea091ebf190a47617f8ec51012b4604fbdd";

const DET_MATH: &str = "alice-det-math";

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
        .replace("\r\n", "\n")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex32(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64, "{s:?} is not 64 hex digits");
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or_else(|e| panic!("{s:?}: {e}"));
    }
    out
}

// ── Fold ────────────────────────────────────────────────────────────────────

fn fold(entries: &[(&str, [u8; 32])]) -> Result<[u8; 32], String> {
    if entries.is_empty() {
        return Err("the table is empty".into());
    }
    let mut sorted = entries.to_vec();
    sorted.sort_unstable_by_key(|(name, _)| *name);
    for pair in sorted.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(format!("`{}` appears twice", pair[0].0));
        }
    }
    let mut h = Sha256::new();
    for (name, digest) in sorted {
        if name.is_empty() || !name.is_ascii() {
            return Err(format!("name {name:?} is empty or not ASCII"));
        }
        let len = u32::try_from(name.len()).map_err(|e| e.to_string())?;
        h.update(len.to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    Ok(h.finalize().into())
}

// ── Source scanning (the method of tests/determinism_golden_coverage.rs) ────

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn ident_at(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Comments, string and char literals replaced by spaces (newlines and byte
/// offsets kept).
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
    // multi-byte characters only occur inside comments and literals, which
    // are blanked byte by byte, so the result is ASCII
    String::from_utf8(out).expect("blanked source is ASCII")
}

/// The end (exclusive) of the brace group opening at `open`.
fn matching_brace(code: &str, open: usize) -> usize {
    let mut depth = 0usize;
    for (k, &ch) in code.as_bytes()[open..].iter().enumerate() {
        if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                return open + k + 1;
            }
        }
    }
    code.len()
}

struct FnItem {
    body: String,
    attrs: String,
}

/// Every `fn` item with a body (blanked source): name -> item.
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
            continue;
        }
        let end = matching_brace(code, open);
        let line_start = code[..at].rfind('\n').map_or(0, |p| p + 1);
        let mut attrs = String::new();
        let mut cursor = line_start;
        while cursor > 0 {
            let prev = code[..cursor - 1].rfind('\n').map_or(0, |p| p + 1);
            let line = code[prev..cursor - 1].trim();
            if line.starts_with("#[") {
                attrs.insert_str(0, &format!("{line}\n"));
                cursor = prev;
            } else {
                break;
            }
        }
        out.entry(name).or_insert_with(|| FnItem {
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

/// A digest constant of a golden file.
#[derive(Clone, Debug, PartialEq, Eq)]
enum DigestConst {
    Sha256([u8; 32]),
    Word(u64),
}

/// `const X: &str = "<64 hex>"` and `const X: u64 = 0x<16 hex digits>`, read raw.
fn digest_consts(raw: &str) -> BTreeMap<String, DigestConst> {
    let mut out = BTreeMap::new();
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
        if value.len() == 66
            && value.starts_with('"')
            && value.ends_with('"')
            && value[1..65].bytes().all(|c| c.is_ascii_hexdigit())
        {
            out.insert(name, DigestConst::Sha256(unhex32(&value[1..65])));
        } else if let Some(h) = value.strip_prefix("0x") {
            let digits: String = h.chars().filter(|&c| c != '_').collect();
            if digits.len() == 16 && digits.bytes().all(|c| c.is_ascii_hexdigit()) {
                let w = u64::from_str_radix(&digits, 16).expect("16 hex digits");
                out.insert(name, DigestConst::Word(w));
            }
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

/// The digest constants named inside an `assert` statement of the code `test` reaches.
fn asserted_consts(
    items: &BTreeMap<String, FnItem>,
    consts: &BTreeMap<String, DigestConst>,
    test: &str,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for name in reachable_fns(items, test) {
        for stmt in items[&name].body.split(';') {
            if !stmt.contains("assert") {
                continue;
            }
            for c in consts.keys() {
                if mentions_ident(stmt, c) {
                    out.insert(c.clone());
                }
            }
        }
    }
    out
}

fn is_active_test(item: &FnItem) -> bool {
    item.attrs.contains("#[test]") && !item.attrs.contains("#[ignore")
}

/// The 32-byte golden the test `test` of `raw` pins (see the module doc).
fn golden_of(raw: &str, test: &str) -> Result<[u8; 32], String> {
    let code = blank_literals(raw);
    let items = fn_items(&code);
    let item = items.get(test).ok_or(format!("no fn {test}"))?;
    if !is_active_test(item) {
        return Err(format!("{test} is not an active #[test]"));
    }
    let consts = digest_consts(raw);
    let asserted = asserted_consts(&items, &consts, test);
    let sha = |names: &BTreeSet<String>| -> Vec<(String, [u8; 32])> {
        names
            .iter()
            .filter_map(|n| match &consts[n] {
                DigestConst::Sha256(d) => Some((n.clone(), *d)),
                DigestConst::Word(_) => None,
            })
            .collect()
    };
    let direct = sha(&asserted);
    match direct.len() {
        1 => return Ok(direct[0].1),
        0 => {}
        _ => {
            return Err(format!(
                "{test} asserts several SHA-256 digests: {direct:?}"
            ))
        }
    }
    let words: BTreeSet<String> = asserted
        .iter()
        .filter(|n| matches!(consts[*n], DigestConst::Word(_)))
        .cloned()
        .collect();
    if words.is_empty() {
        return Err(format!("{test} asserts no digest constant"));
    }
    // a u64 pin: the other test of the file pinning the same state as SHA-256
    let mut found = Vec::new();
    for (name, other) in &items {
        if name == test || !is_active_test(other) {
            continue;
        }
        let a = asserted_consts(&items, &consts, name);
        let s = sha(&a);
        if words.is_subset(&a) && s.len() == 1 {
            found.push((name.clone(), s[0].1));
        }
    }
    match found.as_slice() {
        [(_, d)] => Ok(*d),
        [] => Err(format!(
            "{test} pins only {words:?}; no other test asserts it together with one SHA-256 digest"
        )),
        many => Err(format!(
            "{test}: several SHA-256 tests pin {words:?}: {many:?}"
        )),
    }
}

/// The rows of `PINS` in tests/determinism_golden_coverage.rs: (combination, file, test).
fn pins_rows() -> Vec<(String, String, String)> {
    let src = read("tests/determinism_golden_coverage.rs");
    let start = src
        .find("const PINS: &[(&str, &str, &str, &str)] = &[")
        .expect("PINS table not found");
    let end = start + src[start..].find("\n];").expect("PINS table not closed");
    let lits: Vec<String> = src[start..end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect();
    assert_eq!(lits.len() % 4, 0, "a PINS row is not four strings");
    let rows: Vec<_> = lits
        .chunks(4)
        .map(|r| (r[0].clone(), r[1].clone(), r[2].clone()))
        .collect();
    assert!(
        !rows.is_empty(),
        "no PINS row read: the check would compare nothing"
    );
    rows
}

/// The combination names of `KNOWN_GAPS`.
fn known_gaps() -> BTreeSet<String> {
    let src = read("tests/determinism_golden_coverage.rs");
    let start = src
        .find("const KNOWN_GAPS: &[(&str, &str)] = &[")
        .expect("KNOWN_GAPS table not found");
    let end = start + src[start..].find("\n];").expect("KNOWN_GAPS not closed");
    let lits: Vec<String> = src[start..end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect();
    assert_eq!(lits.len() % 2, 0, "a KNOWN_GAPS row is not two strings");
    lits.chunks(2).map(|r| r[0].clone()).collect()
}

fn table() -> BTreeMap<&'static str, [u8; 32]> {
    let map: BTreeMap<_, _> = PHYSICS_SEMANTICS_PINS.iter().copied().collect();
    assert_eq!(
        map.len(),
        PHYSICS_SEMANTICS_PINS.len(),
        "a name appears twice in PHYSICS_SEMANTICS_PINS"
    );
    map
}

// ── (a) every entry is the golden its pin names ─────────────────────────────

#[test]
fn every_entry_is_the_golden_its_pin_names() {
    let table = table();
    let mut errors = Vec::new();
    let mut compared = 0usize;
    for (combo, file, test) in pins_rows() {
        let raw = read(&format!("tests/{file}"));
        let golden = match golden_of(&raw, &test) {
            Ok(g) => g,
            Err(e) => {
                errors.push(format!("{combo:?} ({file}::{test}): {e}"));
                continue;
            }
        };
        match table.get(combo.as_str()) {
            Some(entry) if *entry == golden => compared += 1,
            Some(entry) => errors.push(format!(
                "{combo:?}: table {} != golden {} ({file}::{test})",
                hex(entry),
                hex(&golden)
            )),
            None => errors.push(format!("{combo:?}: no table entry")),
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    assert_eq!(
        compared,
        table.len() - 1,
        "every non-det-math entry compared"
    );
    println!("{compared} entries equal their golden");
}

// ── (b) the names are the pins plus det-math ────────────────────────────────

#[test]
fn entry_names_are_the_pinned_combinations_and_det_math() {
    let rows = pins_rows();
    let mut expected: BTreeSet<String> = rows.iter().map(|r| r.0.clone()).collect();
    assert_eq!(
        expected.len(),
        rows.len(),
        "a PINS combination is listed twice"
    );
    assert!(!expected.contains(DET_MATH));
    expected.insert(DET_MATH.to_string());
    let names: BTreeSet<String> = table().keys().map(|s| (*s).to_string()).collect();
    assert_eq!(names, expected);
    assert_eq!(PHYSICS_SEMANTICS_PINS.len(), rows.len() + 1);
    let gaps = known_gaps();
    assert!(!gaps.is_empty(), "no KNOWN_GAPS row read");
    let both: Vec<_> = gaps.intersection(&names).collect();
    assert!(both.is_empty(), "known gaps must not be entries: {both:?}");
}

// ── (c) the identifier is the fold of the table ─────────────────────────────

#[test]
fn the_identifier_is_the_fold_of_the_table() {
    let id = fold(PHYSICS_SEMANTICS_PINS).expect("the table folds");
    assert_eq!(
        hex(&id),
        hex(&PHYSICS_SEMANTICS_ID),
        "PHYSICS_SEMANTICS_ID is not the fold of PHYSICS_SEMANTICS_PINS"
    );
}

// ── (d) the det-math entry ──────────────────────────────────────────────────

#[test]
fn the_det_math_entry_is_the_det_math_identifier() {
    assert_eq!(
        table().get(DET_MATH).map(|d| hex(d)),
        Some(hex(&alice_det_math::SEMANTICS_ID)),
        "the alice-det-math entry is not alice_det_math::SEMANTICS_ID \
         (the resolved det-math changed its semantics)"
    );
}

// ── (e) the pinned value, the same in every feature set ─────────────────────

#[test]
fn the_identifier_is_pinned_and_independent_of_features() {
    let recomputed = fold(PHYSICS_SEMANTICS_PINS).expect("the table folds");
    assert_eq!(hex(&recomputed), PINNED_ID_HEX);
    assert_eq!(hex(&PHYSICS_SEMANTICS_ID), PINNED_ID_HEX);
    // the doc states the hex form too
    let src = read("src/semantics.rs");
    assert!(
        src.matches(PINNED_ID_HEX).count() >= 3,
        "the hex form in src/semantics.rs is not the pinned value"
    );
}

#[test]
fn the_table_and_identifier_are_not_behind_a_cfg() {
    let code = blank_literals(&read("src/semantics.rs"));
    assert!(code.contains("pub const PHYSICS_SEMANTICS_ID"));
    assert!(code.contains("pub const PHYSICS_SEMANTICS_PINS"));
    assert!(!code.contains("cfg"), "src/semantics.rs contains a cfg");
    let lib = blank_literals(&read("src/lib.rs"));
    let mut seen = 0;
    let lines: Vec<&str> = lib.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t == "pub mod semantics;" || t.starts_with("pub use semantics::") {
            seen += 1;
            let attr = lines[..i]
                .iter()
                .rev()
                .map(|l| l.trim())
                .take_while(|l| l.starts_with("#[") || l.is_empty())
                .any(|l| l.contains("cfg"));
            assert!(!attr, "src/lib.rs line {}: `{t}` is behind a cfg", i + 1);
        }
    }
    assert_eq!(
        seen, 2,
        "expected `pub mod semantics;` and its re-export in src/lib.rs"
    );
}

// ── (f) what the fold separates and what it ignores ─────────────────────────

/// SHA-256 over u32 BE length ‖ name ‖ digest, computed outside this crate
/// for `[("a", [0; 32]), ("b", [0xff; 32])]`.
#[test]
fn the_fold_matches_a_known_answer() {
    let id = fold(&[("a", [0u8; 32]), ("b", [0xffu8; 32])]).unwrap();
    assert_eq!(
        hex(&id),
        "824ff1276f3b99b079e47a648bb1b22fc67d1164f3f72ca399874f10ece7515b"
    );
}

#[test]
fn one_changed_bit_in_any_entry_changes_the_identifier() {
    let base = fold(PHYSICS_SEMANTICS_PINS).unwrap();
    let mut ids = BTreeSet::from([base]);
    let mut flips = 0usize;
    for i in 0..PHYSICS_SEMANTICS_PINS.len() {
        let spots: BTreeSet<(usize, u8)> = [(0, 0), (31, 7), (i % 32, (i % 8) as u8)].into();
        for (byte, bit) in spots {
            let mut t = PHYSICS_SEMANTICS_PINS.to_vec();
            t[i].1[byte] ^= 1 << bit;
            ids.insert(fold(&t).unwrap());
            flips += 1;
        }
    }
    // every flip gives its own identifier, different from the base, also for
    // the combinations that share one digest
    assert_eq!(ids.len(), flips + 1);
}

#[test]
fn the_order_of_the_entries_does_not_matter() {
    let base = fold(PHYSICS_SEMANTICS_PINS).unwrap();
    let mut rev = PHYSICS_SEMANTICS_PINS.to_vec();
    rev.reverse();
    assert_eq!(fold(&rev).unwrap(), base);
    for k in 1..PHYSICS_SEMANTICS_PINS.len() {
        let mut rot = PHYSICS_SEMANTICS_PINS.to_vec();
        rot.rotate_left(k);
        assert_eq!(fold(&rot).unwrap(), base, "rotation {k}");
    }
}

#[test]
fn a_repeated_or_malformed_name_is_rejected() {
    let mut dup = PHYSICS_SEMANTICS_PINS.to_vec();
    dup.push(PHYSICS_SEMANTICS_PINS[0]);
    assert!(fold(&dup).unwrap_err().contains("twice"));
    // the same name with another digest is a repeat too
    let mut dup2 = PHYSICS_SEMANTICS_PINS.to_vec();
    dup2.push((PHYSICS_SEMANTICS_PINS[3].0, [7u8; 32]));
    assert!(fold(&dup2).is_err());
    assert!(fold(&[("", [0u8; 32])]).is_err());
    assert!(fold(&[("caf\u{e9}", [0u8; 32])]).is_err());
    assert!(fold(&[]).is_err());
}

/// `"a" ‖ P ‖ "bb" ‖ Q` and `"aa" ‖ R ‖ "b" ‖ Q` are the same bytes for
/// `P = 'a' ‖ R[..31]` and `R[31] = 'b'`: two different sorted tables that
/// collide without the length prefix.
#[test]
fn the_length_prefix_separates_tables_that_would_otherwise_collide() {
    let mut r = [0u8; 32];
    for (i, x) in r.iter_mut().enumerate() {
        *x = 0x40 + i as u8;
    }
    r[31] = b'b';
    let mut p = [0u8; 32];
    p[0] = b'a';
    p[1..].copy_from_slice(&r[..31]);
    let q = [0x5au8; 32];
    let t1: [(&str, [u8; 32]); 2] = [("a", p), ("bb", q)];
    let t2: [(&str, [u8; 32]); 2] = [("aa", r), ("b", q)];
    let concat = |t: &[(&str, [u8; 32])]| -> Vec<u8> {
        t.iter()
            .flat_map(|(n, d)| n.bytes().chain(d.iter().copied()))
            .collect()
    };
    assert_eq!(
        concat(&t1),
        concat(&t2),
        "the pair does not collide unprefixed"
    );
    assert_ne!(t1, t2);
    assert_ne!(fold(&t1).unwrap(), fold(&t2).unwrap());
}

// ── the locator itself ──────────────────────────────────────────────────────

#[test]
fn the_locator_follows_helpers_and_ignores_comments_and_unasserted_constants() {
    // `@[` stands for an attribute, so the status generators that scan this
    // file do not count the fixture's functions as tests of their own
    let fixture = r##"
const A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const B: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const W: u64 = 0x0123_4567_89ab_cdef;
fn check(h: &str) { assert_eq!(h, A); }
@[test]
fn via_helper() { let b = B; check("x"); println!("{}", b); }
@[test]
fn word_only() { assert_eq!(1, W); }
@[test]
fn word_and_sha() { assert_eq!(1, W); assert_eq!("", B); }
@[test]
@[ignore = "off"]
fn ignored() { assert_eq!("", A); }
// fn commented() { assert_eq!("", B); }
"##;
    let raw = &fixture.replace("@[", "#[");
    assert_eq!(golden_of(raw, "via_helper").unwrap(), [0x11; 32]);
    assert_eq!(golden_of(raw, "word_only").unwrap(), [0x22; 32]);
    assert!(golden_of(raw, "ignored").is_err());
    assert!(golden_of(raw, "commented").is_err());
}
