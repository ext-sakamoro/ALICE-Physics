#!/usr/bin/env python3
"""Tests for scripts/scip_inputs_check.py: a reader's file that no pattern matches, and a reader that read
nothing, fail; caches are not inputs; the tracer records the files a script opens."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scip_inputs_check as sc  # noqa: E402

PATS = ["**/*.rs", "scripts/integration_levels.py", "docs/MODULES.md"]


class Problems(unittest.TestCase):
    def test_covered_reads_pass(self):
        self.assertEqual(sc.problems({"r": ["src/lib.rs", "docs/MODULES.md"]}, PATS), [])

    def test_an_unmatched_read_fails_and_names_the_file(self):
        errs = sc.problems({"r": ["src/lib.rs", "scripts/abi-consumer-gaps.txt"]}, PATS)
        self.assertEqual(len(errs), 1)
        self.assertIn("scripts/abi-consumer-gaps.txt", errs[0])

    def test_a_reader_that_read_nothing_fails(self):
        self.assertTrue(any("opened no file" in e for e in sc.problems({"r": []}, PATS)))

    def test_caches_only_count_as_nothing(self):
        errs = sc.problems({"r": ["scripts/__pycache__/x.cpython-314.pyc", ".git/HEAD"]}, PATS)
        self.assertTrue(any("opened no file" in e for e in errs), errs)

    def test_the_index_directory_is_not_an_input(self):
        # CI builds the index inside the repository (target/scip): reading it is not reading an input
        reads = {"r": ["src/lib.rs", "target/scip/native.scip", "target/scip/wasm.scip"]}
        self.assertEqual(sc.problems(reads, PATS, "target/scip"), [])
        self.assertEqual(len(sc.problems(reads, PATS)), 2)
        # (only that directory: another file under target/ is still an input)
        self.assertEqual(len(sc.problems({"r": ["src/lib.rs", "target/other.txt"]}, PATS, "target/scip")), 1)

    def test_index_rel(self):
        root = Path(tempfile.mkdtemp(prefix="idx-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, True))
        (root / "target" / "scip").mkdir(parents=True)
        self.assertEqual(sc.index_rel(root, "target/scip"), "target/scip")
        self.assertIsNone(sc.index_rel(root, tempfile.gettempdir()))

    def test_no_pattern_fails(self):
        self.assertEqual(sc.problems({"r": ["src/lib.rs"]}, []), ["the patterns file holds no pattern"])


class TraceRules(unittest.TestCase):
    """the rules that read the trace of each reader (exit code, processes, listings, the index itself)"""

    def t(self, **k):
        base = {"exit": 0, "read": ["src/lib.rs", "target/scip/native.scip"], "listed": [], "spawned": []}
        base.update(k)
        return {"r": base}

    def check(self, **k):
        tr = self.t(**k)
        return sc.problems({"r": tr["r"]["read"]}, PATS, "target/scip", tr)

    def test_a_clean_trace_passes(self):
        self.assertEqual(self.check(), [])

    def test_a_reader_that_crashed_fails(self):
        self.assertTrue(any("exited 1" in e for e in self.check(exit=1)))
        self.assertTrue(any("exited" in e for e in self.check(exit="KeyError('x')")))

    def test_a_reader_that_started_a_process_fails(self):
        self.assertTrue(any("started a process" in e for e in self.check(spawned=["git"])))

    REAL = (Path(__file__).resolve().parent / "land-scip-inputs.txt")

    def real_patterns(self):
        import changed_paths as cp
        return cp.read_patterns(str(self.REAL), [])

    def with_real(self, **k):
        tr = self.t(**k)
        return sc.problems({"r": tr["r"]["read"]}, self.real_patterns(), "target/scip", tr)

    def test_an_unfiltered_listing_needs_every_new_file_to_be_an_input(self):
        # with the real file (which has **/*.rs): a listing of src/ or docs/ sees new files of any kind
        for d in ("src", "docs", "assets", "docs/private", ".github"):
            self.assertTrue(any(f"lists {d}/" in e for e in self.with_real(listed=[d])), d)
        self.assertEqual(self.with_real(listed=["bindings", "unreal-plugin/Source"]), [])   # bindings/**, unreal-plugin/**

    def test_a_glob_is_judged_by_what_it_picks_up(self):
        self.assertEqual(self.with_real(globbed=["src/**/*.rs", "bindings/AlicePhysics.cs",
                                                 "unreal-plugin/Source/**/*.h"]), [])
        for g in ("assets/*.png", "docs/*.md", "src/**/*.toml.bak", "scripts/*.py"):
            self.assertTrue(any(f"globs {g}" in e for e in self.with_real(globbed=[g])), g)

    def test_the_index_read_check_has_a_directory_boundary(self):
        self.assertTrue(any("did not read the index" in e for e in self.check(read=["src/lib.rs", "target/scipX"])))

    def test_a_reader_that_did_not_read_the_index_fails(self):
        self.assertTrue(any("did not read the index" in e for e in self.check(read=["src/lib.rs"])))

    def test_an_index_outside_the_repository_counts(self):
        tr = self.t(read=["src/lib.rs", "<outside>/tmp/idx/native.scip"])
        self.assertEqual(sc.problems({"r": tr["r"]["read"]}, PATS, None, tr, "/tmp/idx"), [])

    def test_anchored_patterns_are_refused(self):
        for p in ("/src/**", "src/"):
            self.assertTrue(any("leading or trailing /" in e for e in sc.problems({"r": ["src/lib.rs"]}, [p])), p)


class Trace(unittest.TestCase):
    def test_the_tracer_records_what_a_script_opens(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data.txt").write_text("x\n", encoding="utf-8")
            (root / "reader.py").write_text("open('data.txt').read()\n", encoding="utf-8")
            t = sc.trace(root, ["reader.py"])
            self.assertEqual(t["exit"], 0)
            self.assertIn("data.txt", t["read"])

    def test_the_tracer_records_listings_and_processes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            (root / "reader.py").write_text("import os, subprocess\nos.listdir('sub')\n"
                                            "subprocess.run(['true'])\n", encoding="utf-8")
            t = sc.trace(root, ["reader.py"])
            self.assertIn("sub", t["listed"])
            self.assertTrue(t["spawned"])

    def test_the_tracer_records_a_glob_as_its_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            (root / "sub" / "a.txt").write_text("x\n", encoding="utf-8")
            (root / "reader.py").write_text("import glob\nglob.glob('sub/*.txt')\n", encoding="utf-8")
            t = sc.trace(root, ["reader.py"])
            self.assertEqual(t["globbed"], ["sub/*.txt"])
            self.assertEqual(t["listed"], [])          # (glob's own directory walk is not a listing of the reader)

    def test_an_import_is_traced_as_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "helper.py").write_text("X = 1\n", encoding="utf-8")
            (root / "reader.py").write_text("import sys; sys.path.insert(0, '.')\nimport helper\n", encoding="utf-8")
            sc.trace(root, ["reader.py"])            # (a first run would write a .pyc if caches were on)
            self.assertIn("helper.py", sc.trace(root, ["reader.py"])["read"])

    def test_a_failing_script_still_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "reader.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
            self.assertEqual(sc.trace(root, ["reader.py"])["exit"], 3)


if __name__ == "__main__":
    unittest.main()
