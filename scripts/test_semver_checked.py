#!/usr/bin/env python3
"""Tests for scripts/semver_checked.py: a semver run that compared nothing must fail loudly."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semver_checked as sc  # noqa: E402

PLAIN = "     Checked [   0.580s] 223 checks: 223 pass, 31 skip\n     Summary no semver update required\n"
# what cargo prints with CARGO_TERM_COLOR=always (captured from a run)
COLOURED = "\x1b[1m\x1b[32m     Checked\x1b[0m [   0.580s] 223 checks: 223 pass, 31 skip\n"
RUSTDOC_FAILED = "error: running cargo-doc on crate alice-physics failed\n"


class CheckedCount(unittest.TestCase):
    def test_plain_and_coloured_logs_give_the_same_count(self):
        self.assertEqual(sc.checked_count(PLAIN), 223)
        self.assertEqual(sc.checked_count(COLOURED), 223)

    def test_a_failed_rustdoc_build_has_no_count(self):
        self.assertIsNone(sc.checked_count(RUSTDOC_FAILED))

    def test_the_last_count_wins(self):
        self.assertEqual(sc.checked_count("Checked [1s] 5 checks\nChecked [2s] 7 checks\n"), 7)


class Main(unittest.TestCase):
    def run_main(self, text: str, rc: int = 0) -> tuple[int, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
        out = io.StringIO()
        with redirect_stdout(out):
            code = sc.main([f.name, "--exit-code", str(rc)])
        Path(f.name).unlink()
        return code, out.getvalue()

    def test_coloured_log_passes_and_prints_the_count(self):
        self.assertEqual(self.run_main(COLOURED), (0, "223\n"))

    def test_no_count_fails_with_an_error_line(self):
        code, out = self.run_main(RUSTDOC_FAILED, rc=101)
        self.assertEqual(code, 1)
        self.assertIn("::error::", out)
        self.assertIn("exit 101", out)

    def test_zero_checks_fails(self):
        code, out = self.run_main("     Checked [   0.1s] 0 checks: 0 pass\n")
        self.assertEqual(code, 1)
        self.assertIn("::error::", out)

    def test_empty_log_fails(self):
        self.assertEqual(self.run_main("")[0], 1)


if __name__ == "__main__":
    unittest.main()
