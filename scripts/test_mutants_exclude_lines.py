#!/usr/bin/env python3
"""Tests of scripts/mutants_exclude_lines.py on a small tree written to a temporary dir."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants_exclude_lines as mel  # noqa: E402
from mutants_equivalent import line_digest  # noqa: E402

SRC = """\
pub fn other() -> u32 {
    1
}

pub fn clamp_pos(a: i64, b: i64) -> i64 {
    let t = a * 2;
    if t > b {
        return b;
    }
    if a > 0 {
        return a;
    }
    0
}
"""


def tree(tmp: Path, src: str = SRC, anchors: str | None = None) -> Path:
    (tmp / "src").mkdir(parents=True, exist_ok=True)
    (tmp / "scripts").mkdir(exist_ok=True)
    (tmp / ".cargo").mkdir(exist_ok=True)
    (tmp / "src/geo.rs").write_text(src, encoding="utf-8")
    if anchors is None:
        d1 = line_digest("    if t > b {")
        d2 = line_digest("    if a > 0 {")
        anchors = (
            "# header\n"
            f"src/geo.rs\tclamp_pos\t2\t{d1}\treplace > with >= in clamp_pos\n"
            f"src/geo.rs\tclamp_pos\t2,5\t{d1},{d2}\treplace > with < in clamp_pos\n"
        )
    (tmp / mel.ANCHORS).write_text(anchors, encoding="utf-8")
    (tmp / mel.CONFIG).write_text(
        "exclude_re = [\n"
        '    "tests::",\n'
        f"{mel.BEGIN}\n"
        '    "geo\\\\.rs:7:.*replace > with >= in clamp_pos",\n'
        '    "geo\\\\.rs:(7|10):.*replace > with < in clamp_pos",\n'
        f"{mel.END}\n"
        "]\n",
        encoding="utf-8",
    )
    (tmp / mel.BASELINE).write_text(
        "# counts\n"
        "5\ttests::\n"
        "1\tgeo\\.rs:7:.*replace > with >= in clamp_pos\n"
        "2\tgeo\\.rs:(7|10):.*replace > with < in clamp_pos\n",
        encoding="utf-8",
    )
    return tmp


class MutantsExcludeLines(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_current_block_passes(self) -> None:
        tree(self.root)
        self.assertEqual(mel.run(self.root, write=False), [])

    def test_code_above_the_function_makes_the_block_stale_and_write_moves_it(self) -> None:
        tree(self.root, src="// a new line\n// and another\n" + SRC)
        found = mel.run(self.root, write=False)
        self.assertEqual(len(found), 1)
        self.assertIn("stale", found[0])
        self.assertEqual(mel.run(self.root, write=True), [])
        self.assertEqual(mel.run(self.root, write=False), [])
        config = (self.root / mel.CONFIG).read_text(encoding="utf-8")
        self.assertIn('"geo\\\\.rs:9:.*replace > with >= in clamp_pos"', config)
        self.assertIn('"geo\\\\.rs:(9|12):.*replace > with < in clamp_pos"', config)
        base = (self.root / mel.BASELINE).read_text(encoding="utf-8")
        self.assertIn("1\tgeo\\.rs:9:.*replace > with >= in clamp_pos", base)
        self.assertIn("2\tgeo\\.rs:(9|12):.*replace > with < in clamp_pos", base)
        self.assertIn("5\ttests::", base)

    def test_a_changed_anchored_line_is_a_finding_and_write_refuses(self) -> None:
        tree(self.root, src=SRC.replace("if t > b {", "if t > b + 1 {"))
        before = (self.root / mel.CONFIG).read_text(encoding="utf-8")
        found = mel.run(self.root, write=True)
        self.assertTrue(any("no longer has the text" in f for f in found), found)
        self.assertEqual((self.root / mel.CONFIG).read_text(encoding="utf-8"), before)

    def test_a_function_found_twice_is_a_finding(self) -> None:
        tree(self.root, src=SRC + "\nmod m {\n    pub fn clamp_pos() {}\n}\n")
        found = mel.run(self.root, write=False)
        self.assertTrue(any("found 2 times" in f for f in found), found)

    def test_a_missing_function_is_a_finding(self) -> None:
        tree(self.root, src=SRC.replace("clamp_pos", "clamp_neg"))
        found = mel.run(self.root, write=False)
        self.assertTrue(any("found 0 times" in f for f in found), found)

    def test_no_anchor_compared_is_a_finding(self) -> None:
        tree(self.root, anchors="# only a header\n")
        found = mel.run(self.root, write=False)
        self.assertEqual(found, [f"{mel.ANCHORS}: no anchor compared"])

    def test_a_missing_block_is_a_finding(self) -> None:
        tree(self.root)
        (self.root / mel.CONFIG).write_text('exclude_re = ["tests::"]\n', encoding="utf-8")
        found = mel.run(self.root, write=False)
        self.assertTrue(any("missing or malformed" in f for f in found), found)

    def test_a_malformed_anchor_is_a_finding(self) -> None:
        tree(self.root, anchors="src/geo.rs\tclamp_pos\tx\td\treplace\n")
        found = mel.run(self.root, write=False)
        self.assertTrue(any("integers" in f for f in found), found)

    def test_the_repository_block_is_current(self) -> None:
        self.assertEqual(mel.run(mel.ROOT, write=False), [])


if __name__ == "__main__":
    unittest.main()
