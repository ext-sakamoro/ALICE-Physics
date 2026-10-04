#!/usr/bin/env python3
"""Tests for scripts/scip_reach.py.

Each case writes a few Rust source files and a hand-built SCIP index (encoded
here, so rust-analyzer is not needed) into a temporary directory, then checks
the level the analysis assigns. One rule per case; breaking that rule in
scip_reach.py turns exactly that case red.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scip_reach as sr  # noqa: E402

P = "rust-analyzer cargo demo 0.1.0 "


# --- tiny SCIP encoder ---------------------------------------------------------

def _v(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(field: int, payload) -> bytes:
    if isinstance(payload, int):
        return _v(field << 3 | 0) + _v(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _v(field << 3 | 2) + _v(len(payload)) + payload


def _packed(xs) -> bytes:
    return b"".join(_v(x) for x in xs)


class Doc:
    def __init__(self, path: str, text: str):
        self.path, self.lines = path, text.split("\n")
        self.occ: list[bytes] = []
        self.syms: list[bytes] = []

    def _pos(self, line: int, token: str, nth: int = 0) -> list[int]:
        start = -1
        for _ in range(nth + 1):
            start = self.lines[line].index(token, start + 1)
        return [line, start, start + len(token)]

    def define(self, sym: str, line: int, token: str, end_line: int | None = None, nth: int = 0) -> "Doc":
        """Definition of `sym` at `token`; its body spans line..end_line."""
        r = self._pos(line, token, nth)
        end_line = line if end_line is None else end_line
        enc = [line, 0, end_line, len(self.lines[end_line])]
        self.occ.append(_f(1, _packed(r)) + _f(2, P + sym) + _f(3, 1) + _f(7, _packed(enc)))
        return self

    def ref(self, sym: str, line: int, token: str, nth: int = 0) -> "Doc":
        self.occ.append(_f(1, _packed(self._pos(line, token, nth))) + _f(2, P + sym))
        return self

    def implements(self, sym: str, target: str, external: bool = False) -> "Doc":
        tgt = ("rust-analyzer cargo std 1.0.0 " + target) if external else P + target
        rel = _f(1, tgt) + _f(3, 1)
        self.syms.append(_f(1, P + sym) + _f(4, rel))
        return self

    def encode(self) -> bytes:
        body = _f(1, self.path) + b"".join(_f(2, o) for o in self.occ) + b"".join(_f(3, s) for s in self.syms)
        return _f(2, body)


def build(docs: list[Doc]) -> Path:
    d = Path(tempfile.mkdtemp())
    for doc in docs:
        p = d / doc.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(doc.lines), encoding="utf-8")
    (d / "target" / "scip").mkdir(parents=True)
    (d / "target" / "scip" / "native.scip").write_bytes(b"".join(doc.encode() for doc in docs))
    (d / "target" / "scip" / "wasm.scip").write_bytes(b"")
    return d


def levels(docs: list[Doc]) -> dict[str, str]:
    d = build(docs)
    return sr.analyze(d, [d / "target/scip/native.scip", d / "target/scip/wasm.scip"]).level


# --- fixtures -----------------------------------------------------------------

LIB = """pub fn used() {}
pub fn unused() {}
pub fn helper() {}
pub fn via_binding() { helper(); }"""


def lib_doc() -> Doc:
    return (Doc("src/lib.rs", LIB)
            .define("used().", 0, "used")
            .define("unused().", 1, "unused")
            .define("helper().", 2, "helper")
            .define("via_binding().", 3, "via_binding")
            .ref("helper().", 3, "helper"))


class Levels(unittest.TestCase):
    def test_example_only_is_l1_and_unreferenced_is_l0(self):
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        lv = levels([lib_doc(), ex])
        self.assertEqual(lv["src/lib.rs::used"], "L1")
        self.assertEqual(lv["src/lib.rs::unused"], "L0")

    def test_binding_reaches_transitively_and_is_live(self):
        ffi = Doc("src/ffi.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        lv = levels([lib_doc(), ffi])
        self.assertEqual(lv["src/lib.rs::via_binding"], "live")
        self.assertEqual(lv["src/lib.rs::helper"], "live")
        self.assertNotIn("src/ffi.rs::api", lv)  # binding surface is a root, not reported

    def test_same_name_on_different_types_is_not_conflated(self):
        text = "pub struct A;\npub struct B;\nimpl A { pub fn update(&self) {} }\nimpl B { pub fn update(&self) {} }"
        src = (Doc("src/m.rs", text)
               .define("m/impl#[A]update().", 2, "update")
               .define("m/impl#[B]update().", 3, "update"))
        ex = Doc("examples/e.rs", "fn main() { a.update(); }").ref("m/impl#[A]update().", 0, "update")
        d = build([src, ex])
        a = sr.analyze(d, [d / "target/scip/native.scip", d / "target/scip/wasm.scip"])
        self.assertEqual(a.items["src/m.rs::update"], {P + "m/impl#[A]update().", P + "m/impl#[B]update()."})
        # the key is reached through A; the B symbol alone would be L0
        self.assertEqual(a.level["src/m.rs::update"], "L1")
        only_b = Doc("examples/e.rs", "fn main() { b.update(); }").ref("m/impl#[B]update().", 0, "update")
        src2 = Doc("src/m.rs", text).define("m/impl#[A]update().", 2, "update")
        self.assertEqual(levels([src2, only_b])["src/m.rs::update"], "L0")

    def test_impl_header_is_not_a_use_of_the_type(self):
        text = "pub struct Foo;\nimpl Foo {\n    fn x(&self) {}\n}"
        src = Doc("src/m.rs", text).define("m/Foo#", 0, "Foo").ref("m/Foo#", 1, "Foo")
        self.assertEqual(levels([src])["src/m.rs::Foo"], "L0")

    def test_reference_inside_cfg_test_is_ignored(self):
        text = "pub fn target() {}\n#[cfg(test)]\nmod tests {\n    fn t() { super::target(); }\n}"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 3, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "L0")

    def test_reference_inside_comment_is_ignored(self):
        text = "pub fn target() {}\n// see target() for details"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 1, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "L0")

    def test_pub_use_reexport_is_not_a_use(self):
        lib = Doc("src/lib.rs", "pub use m::target;").ref("m/target().", 0, "target")
        src = Doc("src/m.rs", "pub fn target() {}").define("m/target().", 0, "target")
        self.assertEqual(levels([lib, src])["src/m.rs::target"], "L0")

    def test_module_level_code_is_a_root(self):
        text = "pub fn target() {}\nconst_assert!(target());"
        src = Doc("src/m.rs", text).define("m/target().", 0, "target").ref("m/target().", 1, "target")
        self.assertEqual(levels([src])["src/m.rs::target"], "live")

    def test_call_through_trait_reaches_the_implementation(self):
        text = ("pub trait Tr { fn run(&self); }\npub fn inner() {}\n"
                "impl Tr for X { fn run(&self) { inner(); } }\npub fn go(t: &dyn Tr) { t.run(); }")

        def src(with_relationship: bool) -> Doc:
            d = (Doc("src/m.rs", text)
                 .define("m/Tr#run().", 0, "run")
                 .define("m/inner().", 1, "inner")
                 .define("m/impl#[X][Tr]run().", 2, "run")
                 .ref("m/inner().", 2, "inner")
                 .define("m/go().", 3, "go")
                 .ref("m/Tr#run().", 3, "run"))
            return d.implements("m/impl#[X][Tr]run().", "m/Tr#run().") if with_relationship else d

        ex = Doc("examples/e.rs", "fn main() { go(&x); }").ref("m/go().", 0, "go")
        # go -> Tr::run -> (is_implementation) X::run -> inner
        self.assertEqual(levels([src(True), ex])["src/m.rs::inner"], "L1")
        self.assertEqual(levels([src(False), ex])["src/m.rs::inner"], "L0")

    def test_impl_of_external_trait_is_a_root(self):
        text = "pub fn helper() {}\npub struct X;\nimpl Drop for X {\n    fn drop(&mut self) { helper(); }\n}"
        src = (Doc("src/m.rs", text)
               .define("m/helper().", 0, "helper")
               .define("m/impl#[X][Drop]drop().", 3, "drop", 3)
               .implements("m/impl#[X][Drop]drop().", "core/ops/Drop#drop().", external=True)
               .ref("m/helper().", 3, "helper"))
        self.assertEqual(levels([src])["src/m.rs::helper"], "live")

    def test_reached_struct_reaches_its_field_types(self):
        text = "pub struct Cfg;\npub struct W {\n    pub cfg: Cfg,\n}"
        src = (Doc("src/m.rs", text)
               .define("m/Cfg#", 0, "Cfg")
               .define("m/W#", 1, "W", 3)
               .define("m/W#cfg.", 2, "cfg")
               .ref("m/Cfg#", 2, "Cfg"))
        ex = Doc("examples/e.rs", "fn main() { let w: W; }").ref("m/W#", 0, "W")
        self.assertEqual(levels([src, ex])["src/m.rs::Cfg"], "L1")
        self.assertEqual(levels([src])["src/m.rs::Cfg"], "L0")

    def test_tests_directory_is_not_a_root(self):
        t = Doc("tests/t.rs", "fn t() { used(); }").ref("used().", 0, "used")
        self.assertEqual(levels([lib_doc(), t])["src/lib.rs::used"], "L0")


class Main(unittest.TestCase):
    def test_missing_index_fails(self):
        d = Path(tempfile.mkdtemp())
        self.assertEqual(sr.main(["--root", str(d)]), 1)

    # each zero-count guard is tested with the other two conditions satisfied,
    # so removing one guard turns exactly one case green-when-it-should-fail
    EX = staticmethod(lambda: Doc("examples/e.rs", "fn main() { used(); }").ref("used().", 0, "used"))
    FFI = staticmethod(lambda: Doc("src/ffi.rs", "pub extern \"C\" fn api() { used(); }").ref("used().", 0, "used"))

    def test_all_three_conditions_met_passes(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX(), self.FFI()]))]), 0)

    def test_no_example_references_fails(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.FFI()]))]), 1)

    def test_no_binding_references_fails(self):
        self.assertEqual(sr.main(["--root", str(build([lib_doc(), self.EX()]))]), 1)

    def test_no_items_fails(self):
        self.assertEqual(sr.main(["--root", str(build([self.EX(), self.FFI()]))]), 1)

    def test_writes_ledger_and_compares_with_baseline(self):
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        ffi = Doc("src/ffi.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        d = build([lib_doc(), ex, ffi])
        (d / "scripts").mkdir()
        (d / "scripts" / "wiring-baseline.txt").write_text("unwired src/lib.rs::via_binding\n", encoding="utf-8")
        out = d / "ledger.md"
        self.assertEqual(sr.main(["--root", str(d), "--write", str(out)]), 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("| L0 | not reached by any non-test code, examples included | 1 |", text)
        self.assertIn("- `src/lib.rs::unused`", text)
        self.assertIn("`src/lib.rs::via_binding` (live)", text)  # baseline says unwired, a binding reaches it


class Baseline(unittest.TestCase):
    """--check-baseline: a ratchet on L0 items, like scripts/wiring-baseline.txt."""

    def crate(self, baseline: str | None) -> Path:
        ex = Doc("examples/demo.rs", "fn main() { used(); }").ref("used().", 0, "used")
        ffi = Doc("src/ffi.rs", "pub extern \"C\" fn api() { via_binding(); }").ref("via_binding().", 0, "via_binding")
        d = build([lib_doc(), ex, ffi])  # L0: unused
        if baseline is not None:
            (d / "scripts").mkdir()
            (d / "scripts" / "integration-baseline.txt").write_text(baseline, encoding="utf-8")
        return d

    def check(self, baseline):
        return sr.main(["--root", str(self.crate(baseline)), "--check-baseline"])

    def test_baseline_listing_every_l0_passes(self):
        self.assertEqual(self.check("# header\nsrc/lib.rs::unused\n"), 0)

    def test_a_new_l0_item_fails(self):
        self.assertEqual(self.check("# nothing recorded\n"), 1)

    def test_a_stale_entry_fails(self):
        self.assertEqual(self.check("src/lib.rs::unused\nsrc/lib.rs::used\n"), 1)  # used is L1 now

    def test_missing_baseline_fails(self):
        self.assertEqual(self.check(None), 1)

    def test_write_baseline_then_check_round_trips(self):
        d = self.crate(None)
        (d / "scripts").mkdir()
        self.assertEqual(sr.main(["--root", str(d), "--write-baseline"]), 0)
        text = (d / "scripts" / "integration-baseline.txt").read_text(encoding="utf-8")
        self.assertIn("src/lib.rs::unused\n", text)
        self.assertEqual(sr.main(["--root", str(d), "--check-baseline"]), 0)


if __name__ == "__main__":
    unittest.main()
