#!/usr/bin/env python3
"""Tests of scripts/reader_map_core.py (python3 scripts/test_reader_map_core.py), the shared reading of a reader map:
a file is named only by its whole path as one token, outside comments and docstrings, and a script is run only by a
`run:` step. The first three cases are the ways a reader claim was made without reading: a free word that is not
the path, a shell comment, and a script named in a comment."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reader_map_core as c  # noqa: E402


class Core(unittest.TestCase):
    def script(self, name: str, text: str) -> Path:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(d)])
        p = d / name
        p.write_text(text, encoding="utf-8")
        return p

    # --- the three ways a reader was claimed without reading the file ---

    def test_a_free_word_that_is_not_the_path_names_nothing(self):
        for text in ("cargo test --workspace\n", "cargo test\n", "src\n", "tests\n"):
            self.assertFalse(c.names_path(self.script("r.sh", text), "src/lib.rs"), text)
        self.assertFalse(c.token("src/lib.rs").search("cargo test --workspace"))

    def test_a_shell_comment_does_not_name_a_path(self):
        self.assertFalse(c.names_path(self.script("r.sh", "# reads conformance/a.toml\necho hi\n"), "conformance/a.toml"))
        self.assertFalse(c.names_path(self.script("r.sh", "echo hi  # conformance/a.toml\n"), "conformance/a.toml"))
        self.assertTrue(c.names_path(self.script("r.sh", "cat conformance/a.toml\n"), "conformance/a.toml"))
        self.assertTrue(c.names_path(self.script("r.sh", "echo '# conformance/a.toml'\n"), "conformance/a.toml"),
                        "a # inside quotes is text, not a comment")

    def test_a_script_named_in_a_comment_is_not_run(self):
        wf = ("on:\n  push:\n    branches: ['ci/**']\njobs:\n  a:\n    steps:\n"
              "      - run: |\n          # python3 scripts/old.py\n          python3 scripts/new.py\n")
        runs = [c.without_shell_comments(r) for r in c.run_texts(wf)]
        named = set().union(*(set(c.RUN_SCRIPT.findall(r)) for r in runs))
        self.assertEqual(named, {"new.py"})
        self.assertFalse(c.names_path(self.script("r.py", "# see conformance/a.toml\nX = 1\n"), "conformance/a.toml"))
        self.assertFalse(c.names_path(self.script("r.py", '"""reads conformance/a.toml"""\nX = 1\n'), "conformance/a.toml"),
                         "a docstring is not a use")

    # --- what does name a file ---

    def test_the_whole_path_is_one_token(self):
        t = c.token("conformance/a.toml")
        self.assertTrue(t.search("--table conformance/a.toml --x"))
        for text in ("conformance/a.toml.bak", "xconformance/a.toml", "old/conformance/a.toml", "conformance/a.tomls"):
            self.assertFalse(t.search(text), text)

    def test_python_strings_and_path_chains_name_a_file(self):
        for src in ('open("conformance/a.toml")\n', 'P = ROOT / "conformance" / "a.toml"\n',
                    'from pathlib import Path\nP = Path("conformance", "a.toml")\n',
                    'import pathlib\nP = pathlib.Path("conformance") / "a.toml"\n', 'F = f"conformance/a.toml"\n'):
            self.assertTrue(c.names_path(self.script("r.py", src), "conformance/a.toml"), src)
        self.assertFalse(c.names_path(self.script("r.py", 'P = print("conformance", "a.toml")\n'), "conformance/a.toml"),
                         "only a path call's arguments are parts of a path")

    def test_a_path_name_imported_from_a_module_is_a_use(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(d)])
        (d / "lib.py").write_text('PATH = "conformance/a.toml"\nOTHER = 1\n', encoding="utf-8")
        (d / "uses.py").write_text("import lib\nopen(lib.PATH)\n", encoding="utf-8")
        (d / "other.py").write_text("import lib\nprint(lib.OTHER)\n", encoding="utf-8")
        (d / "frm.py").write_text("from lib import PATH as P\nopen(P)\n", encoding="utf-8")
        self.assertTrue(c.names_path(d / "uses.py", "conformance/a.toml"))
        self.assertFalse(c.names_path(d / "other.py", "conformance/a.toml"), "importing for something else is not a use")
        self.assertTrue(c.names_path(d / "frm.py", "conformance/a.toml"))

    def test_run_texts_reads_run_steps_only(self):
        wf = ("on:\n  push:\n    branches: ['ci/**']\n    paths:\n      - scripts/listed.py\njobs:\n  a:\n    steps:\n"
              "      - run: python3 scripts/one.py\n      - name: b\n        run: |\n          python3 scripts/two.py\n"
              "      - uses: actions/checkout@v5\n")
        runs = c.run_texts(wf)
        self.assertEqual(len(runs), 2)
        named = set().union(*(set(c.RUN_SCRIPT.findall(r)) for r in runs))
        self.assertEqual(named, {"one.py", "two.py"}, "a paths: list does not run a script")

    def test_on_ci_branches(self):
        self.assertTrue(c.on_ci_branches("on:\n  push:\n    branches: [main, 'ci/**']\njobs:\n"))
        self.assertFalse(c.on_ci_branches("on:\n  push:\n    branches: [main]\njobs:\n"))
        self.assertFalse(c.on_ci_branches("on:\n  workflow_dispatch:\njobs:\n"))

    def test_a_script_that_does_not_parse_uses_nothing(self):
        self.assertEqual(c.python_strings("def (:\n"), [])
        self.assertFalse(c.names_path(self.script("r.py", 'def (:\n"conformance/a.toml"\n'), "conformance/a.toml"))


if __name__ == "__main__":
    unittest.main()
