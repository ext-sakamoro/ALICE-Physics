#!/usr/bin/env python3
"""Tests for scripts/bench_counts_check.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_counts_check as bc  # noqa: E402

# the shape of gungraun's terminal output, colour codes included
OK = ("\x1b[32mhot::world_step\x1b[0m\n\x1b[90m  \x1b[0mInstructions:     \x1b[1m1234567\x1b[0m|1234000   (+0.04%)\n"
      "\x1b[32mhot::bvh\x1b[0m\n  Instructions:     88|N/A   (*********)\n")
EMPTY = "hot::world_step\n  Instructions:     0|0    (No change)\n  Instructions:  42|40 (+5%)\n"


class Counts(unittest.TestCase):
    def run_on(self, text: str) -> int:
        p = Path(tempfile.mkdtemp()) / "b.txt"
        p.write_text(text, encoding="utf-8")
        return bc.main(["x", str(p)])

    def test_counts_are_read_through_colour_codes(self):
        self.assertEqual(bc.counts(OK), [1234567, 88])

    def test_non_zero_counts_pass(self):
        self.assertEqual(self.run_on(OK), 0)

    def test_a_zero_count_fails(self):
        self.assertEqual(self.run_on(EMPTY), 1)

    def test_no_count_at_all_fails(self):
        self.assertEqual(self.run_on("Compiling alice-physics\nFinished\n"), 1)


if __name__ == "__main__":
    unittest.main()
