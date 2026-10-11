#!/usr/bin/env python3
"""Tests for scripts/mutants_in_diff_plan.py (one rule per case)."""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_in_diff_plan as mp  # noqa: E402

LIB = """\
pub mod math;
#[cfg(feature = "replay")]
pub mod replay;
/// doc between the attribute and the module
#[cfg(feature = "neural")]
/// more doc
pub mod neural;
#[cfg(all(feature = "wasm", not(feature = "std")))]
pub mod wasm_only;
#[cfg(any(feature = "ffi", feature = "python"))]
pub mod bindings_common;
#[cfg(feature = "python")]
pub mod python;
#[cfg(not(feature = "std"))]
pub mod no_std_alloc;
"""
CARGO = """\
[package]
name = "x"

[[test]]
name = "audit_replay"
required-features = ["std", "replay"]

[[test]]
name = "audit_math"
required-features = ["analytics"]

[[test]]
name = "python_replay_binding"
required-features = ["python"]
"""


def diff_of(*paths: str) -> str:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n" for p in paths)


class Plan(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src/lib.rs").write_text(LIB, encoding="utf-8")
        (self.root / "Cargo.toml").write_text(CARGO, encoding="utf-8")
        for t in ["audit_replay", "analytic_replay_in_memory", "audit_replayer_x", "audit_math",
                  "python_replay_binding", "audit_neural_net"]:
            (self.root / f"tests/{t}.rs").write_text("", encoding="utf-8")

    def plan(self, *paths, base=()):
        return mp.plan(diff_of(*paths), list(base), self.root)

    def test_a_gated_module_turns_its_feature_on(self):
        self.assertEqual(self.plan("src/replay.rs")["features"], ["replay"])

    def test_an_ungated_module_adds_no_feature_and_keeps_the_matrix_ones(self):
        p = self.plan("src/math.rs", base=["parallel"])
        # the module itself adds nothing; its test brings its required feature
        self.assertEqual(p["features"], ["analytics", "parallel"])
        self.assertEqual(p["tests"], ["audit_math"])

    def test_doc_comments_between_the_cfg_and_the_mod_line_are_skipped(self):
        self.assertIn("neural", self.plan("src/neural.rs")["features"])

    def test_tests_named_after_the_module_run_but_not_longer_names(self):
        # audit_replayer_x is a different word; the python binding test needs a
        # host-unbuildable feature and is left out
        self.assertEqual(self.plan("src/replay.rs")["tests"], ["analytic_replay_in_memory", "audit_replay"])

    def test_required_features_of_a_selected_test_are_added(self):
        self.assertEqual(self.plan("src/replay/store.rs")["features"], ["replay"])

    def test_a_module_that_needs_not_std_is_skipped_and_removed_from_the_diff(self):
        p = self.plan("src/wasm_only.rs", "src/math.rs")
        self.assertEqual(p["skipped"], ["src/wasm_only.rs"])
        self.assertNotIn("wasm_only", p["diff"])
        self.assertIn("src/math.rs", p["diff"])

    def test_a_not_std_module_is_skipped(self):
        self.assertEqual(self.plan("src/no_std_alloc.rs")["skipped"], ["src/no_std_alloc.rs"])

    def test_a_host_unbuildable_feature_skips_the_module(self):
        self.assertEqual(self.plan("src/python.rs")["skipped"], ["src/python.rs"])

    def test_any_takes_its_first_feature(self):
        self.assertEqual(self.plan("src/bindings_common.rs")["features"], ["ffi"])

    def test_files_outside_src_and_lib_rs_pass_through_unchanged(self):
        p = self.plan("src/lib.rs", "Cargo.toml")
        self.assertEqual((p["features"], p["tests"], p["skipped"]), ([], [], []))
        self.assertIn("Cargo.toml", p["diff"])

    # src/geom.rs is not gated in lib.rs; one function and one impl inside it are
    MATH = """\
pub fn plain() -> u8 {
    1
}

#[cfg(feature = "gpu-solver-bridge")]
/// doc between the attribute and the item
pub fn bridged(x: u8) -> u8 {
    let s = "a { brace in a string";
    x + 1 // } a brace in a comment
}

#[cfg(feature = "simd")]
impl Thing {
    fn inner(&self) -> u8 {
        if true {
            2
        } else {
            3
        }
    }
}
"""

    def changed(self, line: int) -> str:
        return f"diff --git a/src/geom.rs b/src/geom.rs\n--- a/src/geom.rs\n+++ b/src/geom.rs\n@@ -{line} +{line} @@\n-old\n+new\n"

    def plan_math(self, line: int):
        (self.root / "src/geom.rs").write_text(self.MATH, encoding="utf-8")
        return mp.plan(self.changed(line), [], self.root)

    def test_a_changed_line_in_an_item_gated_fn_turns_its_feature_on(self):
        # line 9: `x + 1` inside `bridged`
        self.assertEqual(self.plan_math(9)["features"], ["gpu-solver-bridge"])

    def test_a_changed_line_outside_the_gated_items_adds_no_feature(self):
        self.assertEqual(self.plan_math(2)["features"], [])
        # the blank line after the gated fn closes, before the impl's attribute
        self.assertEqual(self.plan_math(11)["features"], [])

    def test_a_changed_line_deep_in_a_gated_impl_turns_its_feature_on(self):
        # line 18: `3` inside the else of `inner`, two blocks below the impl
        self.assertEqual(self.plan_math(18)["features"], ["simd"])

    def test_changed_lines_are_counted_in_the_new_file(self):
        d = "@@ -1,3 +5,4 @@\n ctx\n-gone\n+new1\n+new2\n ctx\n@@ -20 +30 @@\n-x\n+y\n"
        self.assertEqual(mp.changed_lines(d), [6, 7, 30])

    def test_the_real_lib_rs_gates_replay_behind_its_feature(self):
        gates = mp.module_gates(mp.cfg_spans(HERE.parent / "src/lib.rs"))
        self.assertEqual(mp.features_for(gates["replay"]), (["replay"], True))
        self.assertGreater(len(gates), 5, "the real lib.rs has gated modules")



# Every shape of an item header rustfmt writes, and the nodes inside an item that
# can carry their own cfg. A line ending in `// expect: <features>` (comma
# separated, `-` for none) is planned on its own and must turn on exactly those.
SHAPES = """\
#[cfg(feature = "alpha")]
pub fn long_signature<B: core::fmt::Debug + ?Sized>(
    first: &mut B, // expect: alpha
    second: u32,
) -> u32 {
    second + 1 // expect: alpha
}

#[cfg(feature = "alpha")]
pub fn with_where<T>(x: T) -> T
where
    T: Copy, // expect: alpha
{
    x // expect: alpha
}

#[cfg(feature = "alpha")]
pub fn brace_on_its_own_line()
{
    let _ = 1; // expect: alpha
}

#[cfg(all(
    feature = "alpha",
    feature = "beta",
))]
pub fn multi_line_cfg() -> u8 {
    7 // expect: alpha,beta
}

#[cfg(feature = "beta")]
pub const TABLE: [u8; 3] = [
    1, // expect: beta
    2,
    3,
];

pub struct Thing {
    plain: u8, // expect: -
    #[cfg(feature = "gamma")]
    gated: u8, // expect: gamma
}

impl Thing {
    pub fn plain(&self) -> u8 {
        self.plain // expect: -
    }

    #[cfg(feature = "gamma")]
    pub fn gated_method(
        &self,
    ) -> u8 {
        self.gated // expect: gamma
    }
}

pub enum Shape {
    Plain,
    #[cfg(feature = "delta")]
    Gated {
        side: u8, // expect: delta
    },
}

pub fn statements(s: &Shape, v: &mut Vec<u8>) -> u8 {
    #[cfg(feature = "delta")]
    let doubled = v
        .iter()
        .map(|x| x * 2) // expect: delta
        .count();
    #[cfg(feature = "delta")]
    v.push(
        3, // expect: delta
    );
    #[cfg(feature = "delta")]
    println!(
        "{}", 1 // expect: delta
    );
    match s {
        #[cfg(feature = "delta")]
        Shape::Gated { side } => {
            *side // expect: delta
        }
        _ => 0, // expect: -
    }
}

#[cfg(feature = "epsilon")]
pub fn outer() -> u8 {
    fn nested() -> u8 {
        5 // expect: epsilon
    }
    nested()
}

#[cfg(feature = "epsilon")]
mod inline_module {
    pub fn inside() -> u8 {
        6 // expect: epsilon
    }
}

pub trait Speak {
    #[cfg(feature = "epsilon")]
    fn default_method(&self) -> u8 {
        8 // expect: epsilon
    }
}

#[cfg(not(test))]
pub fn not_test() -> u8 {
    9 // expect: -
}

#[cfg_attr(feature = "zeta", derive(Debug))]
pub struct Derived {
    field: u8, // expect: -
}

pub fn decoys() -> &'static str {
    // #[cfg(feature = "zeta")] in a comment
    "#[cfg(feature = \\"zeta\\")] in a string" // expect: -
}

#[cfg(feature = "alpha")]
pub fn tricky_tokens<'a>(s: &'a str) -> char {
    let _raw = r#"} { } "#;
    let _ = s; // expect: alpha
    '{' // expect: alpha
}
pub fn after_tricky() -> u8 {
    0 // expect: -
}

#[cfg(any(feature = "beta", feature = "gamma"))]
pub fn either() -> u8 {
    1 // expect: beta
}

#[cfg(feature = "gamma")]
/// doc between the attribute and the item
pub fn documented() -> u8 {
    2 // expect: gamma
}
"""

FILE_GATED = """\
#![cfg(feature = "zeta")]

pub fn anything() -> u8 {
    1 // expect: zeta
}
"""


class Shapes(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "src/outer").mkdir(parents=True)
        (self.root / "src/lib.rs").write_text(
            "pub mod shapes;\npub mod file_gated;\npub mod outer;\n", encoding="utf-8")
        (self.root / "Cargo.toml").write_text("[package]\nname = \"x\"\n", encoding="utf-8")
        (self.root / "src/shapes.rs").write_text(SHAPES, encoding="utf-8")
        (self.root / "src/file_gated.rs").write_text(FILE_GATED, encoding="utf-8")
        # a nested module gated where it is declared, by a multi-line attribute
        (self.root / "src/outer.rs").write_text(
            "#[cfg(all(\n    feature = \"eta\",\n))]\npub mod inner;\n", encoding="utf-8")
        (self.root / "src/outer/inner.rs").write_text("pub fn f() -> u8 {\n    1\n}\n", encoding="utf-8")

    def expectations(self, text: str) -> list[tuple[int, list[str]]]:
        out = []
        for n, line in enumerate(text.splitlines(), 1):
            if "// expect: " in line:
                want = line.split("// expect: ")[1].strip()
                out.append((n, [] if want == "-" else want.split(",")))
        return out

    def plan_line(self, path: str, line: int) -> list[str]:
        diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -{line} +{line} @@\n-old\n+new\n"
        return mp.plan(diff, [], self.root)["features"]

    def test_every_marked_line_turns_on_exactly_its_features(self):
        checked = 0
        for path, text in (("src/shapes.rs", SHAPES), ("src/file_gated.rs", FILE_GATED)):
            for line, want in self.expectations(text):
                with self.subTest(path=path, line=line, text=text.splitlines()[line - 1].strip()):
                    self.assertEqual(self.plan_line(path, line), sorted(want))
                checked += 1
        self.assertGreaterEqual(checked, 25, "the fixtures mark every shape")

    def test_a_nested_module_gated_where_it_is_declared(self):
        self.assertEqual(self.plan_line("src/outer/inner.rs", 2), ["eta"])

    def test_a_file_that_does_not_parse_fails_the_plan(self):
        (self.root / "src/shapes.rs").write_text("pub fn broken( {\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.plan_line("src/shapes.rs", 1)


class Cfg(unittest.TestCase):
    def test_not_test_code_is_not_compiled_by_the_test_run(self):
        # the mutants are tested with `cargo test`, where cfg(test) holds
        self.assertEqual(mp.features_for("not (test)"), ([], False))

    def test_not_std_is_unbuildable(self):
        self.assertEqual(mp.features_for('not (feature = "std")'), ([], False))

    def test_any_takes_its_first_satisfiable_alternative(self):
        self.assertEqual(mp.features_for('any (feature = "python" , feature = "ffi")'), (["ffi"], True))

    def test_not_any_needs_every_feature_off(self):
        self.assertEqual(mp.features_for('all (feature = "a" , not (any (feature = "b" , feature = "a")))'),
                         ([], False))

    def test_a_wasm_target_is_unbuildable(self):
        self.assertEqual(mp.features_for('all (target_arch = "wasm32" , feature = "a")'), ([], False))

    def test_several_cfgs_joined_must_all_hold(self):
        self.assertEqual(mp.features_for('feature = "a" , feature = "b"'), (["a", "b"], True))


# --- the real src/: an attribute count the parser must match, and every gated span planned ---

def mask(src: str) -> str:
    """The source with comments, strings, raw strings and char literals blanked
    (newlines kept), by a lexer that does not share code with the parser."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "r" and (m := re.match(r'r(#*)"', src[i:])) and not (i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            j = src.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "'" and (m := re.match(r"'(\\.[^']*|[^\\'])'", src[i:])):
            out.append(" " * len(m.group(0)))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def feature_attributes(src: str) -> int:
    """`#[cfg(...)]` outer and inner attributes whose expression names a feature,
    joined across lines by bracket balance."""
    code, count = mask(src), 0
    for m in re.finditer(r"#!?\s*\[\s*cfg\s*\(", code):
        depth, j = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(code[j], 0)
            j += 1
        if re.search(r"\bfeature\b", code[m.end():j]):
            count += 1
    return count


class RealSource(unittest.TestCase):
    SRC = HERE.parent / "src"

    def test_the_parser_finds_every_feature_attribute_of_src(self):
        compared = 0
        for f in sorted(self.SRC.rglob("*.rs")):
            text = f.read_text(encoding="utf-8")
            want = feature_attributes(text)
            got = sum(1 for s in mp.cfg_spans(f) if re.search(r"\bfeature\b", s.cfg))
            with self.subTest(file=str(f.relative_to(self.SRC))):
                self.assertEqual(got, want)
            compared += want
        self.assertGreater(compared, 200, "src/ has feature-gated items; none compared = blind")

    def test_every_gated_span_closes_on_balanced_brackets(self):
        checked = 0
        for f in sorted(self.SRC.rglob("*.rs")):
            lines = mask(f.read_text(encoding="utf-8")).splitlines()
            for s in mp.cfg_spans(f):
                body = "\n".join(lines[s.start - 1:s.end])
                for o, c in ("{}", "()", "[]"):
                    with self.subTest(file=f.name, span=(s.start, s.end), bracket=o):
                        self.assertEqual(body.count(o), body.count(c))
                last = lines[s.end - 1].rstrip()
                with self.subTest(file=f.name, span=(s.start, s.end), last=last):
                    self.assertTrue(last.endswith(("}", ";", ",", ")", "]")) or s.kind == "file")
                checked += 1
        self.assertGreater(checked, 200)

    def test_a_change_on_the_last_line_of_every_gated_span_turns_its_feature_on(self):
        """236 multi-line feature-gated fns were not seen by the line scanner; plan a
        one-line change at the closing line of every buildable gated span and
        require its features (the closing line is the one a line scanner loses)."""
        root = HERE.parent
        checked = 0
        for f in sorted(self.SRC.rglob("*.rs")):
            rel = str(f.relative_to(root))
            module_feats, module_ok = mp.features_for(" , ".join(mp.module_path_gates(rel, root)))
            if not module_ok:
                continue
            for s in mp.cfg_spans(f):
                feats, ok = mp.features_for(s.cfg)
                if not ok or not feats or s.kind == "mod":
                    continue
                diff = f"diff --git a/{rel} b/{rel}\n--- a/{rel}\n+++ b/{rel}\n@@ -{s.end} +{s.end} @@\n-o\n+n\n"
                got = mp.plan(diff, [], root)["features"]
                with self.subTest(file=rel, span=(s.start, s.end)):
                    self.assertTrue(set(feats) <= set(got), f"{feats} not in {got}")
                checked += 1
        self.assertGreater(checked, 50, "buildable gated spans planned; none = blind")


if __name__ == "__main__":
    unittest.main()
