//! `Fix128` has one product: the operator in `src/math.rs` (rounding to nearest, ties to even,
//! odd in its operands). No other part of `src/` may define its own product of two `Fix128`
//! values with another rounding.
//!
//! Why: while the operator floored toward −∞, modules that needed mirror symmetry wrote local
//! sign-symmetric products (`|a|·|b|` with the sign put back), and the matrix inverse had a
//! floored wide product of its own, so the crate had several meanings of `a·b`. A fix to one did
//! not reach the others, and two codes computing "the same" product could disagree in the last
//! bit.
//!
//! Scanned: every `.rs` file under `src/`. Flagged:
//! - a function outside `src/math.rs` that takes two `Fix128` and returns a `Fix128` and whose
//!   name contains `mul` (a local product),
//! - a function named `*_odd` that multiplies (the local odd products),
//! - the sign-symmetric product pattern `x.abs() * y.abs()`.
//!
//! Allowed, by name and with the reason: `linear_elastic_fem::mul_ceil_fraction`, a ceiling
//! `⌈x·f⌉` for a marking bound (a different operation from the product, chosen so the marked
//! set is never short). An allowance that no longer matches anything fails, and 0 files
//! scanned fails.

use std::fs;
use std::path::{Path, PathBuf};

const ALLOWED: &[(&str, &str, &str)] = &[(
    "src/linear_elastic_fem.rs",
    "mul_ceil_fraction",
    "a ceiling for the Dörfler marking bound, not the product",
)];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read src") {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// (function name, its one-line signature) of every `fn` in `text`
fn functions(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        let t = t
            .strip_prefix("pub(crate) ")
            .or_else(|| t.strip_prefix("pub "))
            .unwrap_or(t);
        let t = t.strip_prefix("const ").unwrap_or(t);
        if let Some(rest) = t.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            out.push((name, t.to_string()));
        }
    }
    out
}

fn findings(path: &str, text: &str) -> Vec<String> {
    let mut bad = Vec::new();
    for (name, sig) in functions(text) {
        let two_fix_to_fix = sig.matches(": Fix128").count() >= 2 && sig.contains("-> Fix128");
        if path != "src/math.rs" && name.contains("mul") && two_fix_to_fix {
            bad.push(format!("{path}: fn {name} is a local Fix128 product"));
        }
        // the odd arithmetic helpers (mul_odd, scale_odd, cross_odd, dot_odd,
        // inv_inertia_odd), not other names that end in `_odd` (cloth's even-odd rule)
        let odd_arith = ["mul", "scale", "cross", "dot", "inv_inertia", "product"]
            .iter()
            .any(|p| name.starts_with(p));
        if name.ends_with("_odd") && odd_arith {
            bad.push(format!("{path}: fn {name} is a local odd product"));
        }
    }
    for (k, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        let compact: String = code.chars().filter(|c| !c.is_whitespace()).collect();
        let symmetric = compact.split(".abs()*").skip(1).any(|r| {
            r.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_')
                .starts_with(".abs()")
        });
        if symmetric {
            bad.push(format!(
                "{path}:{}: sign-symmetric product `|a|·|b|`",
                k + 1
            ));
        }
    }
    bad
}

#[test]
fn src_has_one_fix128_product() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    assert!(
        !files.is_empty(),
        "no source scanned (the check compared nothing)"
    );
    let mut bad = Vec::new();
    let mut allowance_used = vec![false; ALLOWED.len()];
    for f in &files {
        let rel = f
            .strip_prefix(root)
            .expect("under root")
            .to_string_lossy()
            .replace('\\', "/");
        let text = fs::read_to_string(f).expect("read");
        for finding in findings(&rel, &text) {
            match ALLOWED
                .iter()
                .position(|(p, n, _)| rel == *p && finding.contains(&format!("fn {n} ")))
            {
                Some(i) => allowance_used[i] = true,
                None => bad.push(finding),
            }
        }
    }
    assert!(
        bad.is_empty(),
        "other Fix128 products in src/:\n{}",
        bad.join("\n")
    );
    for (used, (p, n, why)) in allowance_used.iter().zip(ALLOWED) {
        assert!(
            used,
            "the allowance {p}::{n} ({why}) matches nothing; remove it"
        );
    }
}

#[test]
fn the_check_catches_the_forms_it_names() {
    let local = "fn mul(a: Fix128, b: Fix128) -> Fix128 {\n    a * b\n}\n";
    assert_eq!(findings("src/x.rs", local).len(), 1);
    assert!(
        findings("src/math.rs", local).is_empty(),
        "math.rs owns the product"
    );
    let odd =
        "pub(crate) fn mul_odd(a: Fix128, b: Fix128) -> Fix128 {\n    let p = a.abs() * b.abs();\n";
    // the local product, the odd name and the |a|·|b| pattern
    assert_eq!(findings("src/x.rs", odd).len(), 3);
    let cross = "pub(crate) fn cross_odd(u: Vec3Fix, v: Vec3Fix) -> Vec3Fix {\n";
    assert_eq!(findings("src/x.rs", cross).len(), 1);
    let fine = "fn scale(v: Vec3Fix, s: Fix128) -> Vec3Fix { v * s }\nlet r = x.abs() + y;\n";
    assert!(findings("src/x.rs", fine).is_empty());
}
