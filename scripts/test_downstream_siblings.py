"""Oracles of scripts/downstream_siblings.py.

run: python3 scripts/test_downstream_siblings.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("downstream_siblings", os.path.join(HERE, "downstream_siblings.py"))
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

SHA = "8a1b00d800082187a8693732a9ffb806c98814e7"
ROOT = "/w/ALICE-LOL"
CORE = (
    '[package]\nname = "core"\nversion = "0.1.0"\n\n[dependencies]\n'
    'inner = { path = "../inner" }\n'
    'phys = { path = "../../ALICE-Physics/crates/alice-physics", optional = true }\n'
    'sign = { path = "../../ALICE-Sign", version = "0.1", optional = true }\n'
    'plain = "1"\n'
)
MANIFESTS = {"Cargo.toml": '[workspace]\nmembers = ["core"]\n', "core/Cargo.toml": CORE}


class Outside(unittest.TestCase):
    def test_path_dependencies_leaving_the_checkout_name_their_repositories(self):
        self.assertEqual(sorted(ds.outside_repos(ROOT, MANIFESTS)), ["ALICE-Physics", "ALICE-Sign"])

    def test_target_and_patch_tables_are_read(self):
        m = {"Cargo.toml": '[target.\'cfg(unix)\'.dev-dependencies]\nk = { path = "../ALICE-K" }\n'
                           '[patch.crates-io]\nz = { path = "../ALICE-Z" }\n'}
        self.assertEqual(sorted(ds.outside_repos(ROOT, m)), ["ALICE-K", "ALICE-Z"])

    def test_a_path_leaving_the_layout_directory_fails(self):
        with self.assertRaises(ValueError):
            ds.outside_repos(ROOT, {"Cargo.toml": '[dependencies]\nx = { path = "../../../elsewhere" }\n'})


class Plan(unittest.TestCase):
    NEEDED = {"ALICE-Physics": ["c"], "ALICE-Sign": ["c"]}

    def test_a_pinned_sibling_is_placed_at_its_pin(self):
        out, problems, _ = ds.plan("ALICE-LOL", self.NEEDED, set(), {"ALICE-Sign": SHA}, True, False, ["ALICE-Physics"])
        self.assertEqual((out, problems), ([("ALICE-Sign", SHA)], []))

    def test_the_candidate_is_never_placed_even_when_pinned(self):
        out, _, _ = ds.plan("ALICE-LOL", self.NEEDED, set(), {"ALICE-Sign": SHA, "ALICE-Physics": SHA}, True, False, [])
        self.assertEqual(out, [("ALICE-Sign", SHA)])

    def test_a_missing_pin_file_fails(self):
        _, problems, _ = ds.plan("ALICE-LOL", self.NEEDED, set(), {}, False, False, [])
        self.assertEqual(len(problems), 1)
        self.assertIn("is missing", problems[0])

    def test_without_a_pin_file_unpinned_main_places_at_main_with_a_notice(self):
        out, problems, notices = ds.plan("ALICE-TRT", self.NEEDED, set(), {}, False, True, [])
        self.assertEqual((out, problems), ([("ALICE-Sign", "main")], []))
        self.assertEqual(len(notices), 1)
        self.assertIn("placed at main", notices[0])

    def test_unpinned_main_does_not_apply_when_the_pin_file_exists(self):
        _, problems, _ = ds.plan("ALICE-TRT", self.NEEDED, set(), {}, True, True, [])
        self.assertEqual(len(problems), 1)
        self.assertIn("is not pinned", problems[0])

    def test_a_required_repository_not_taken_by_path_fails_as_comparing_nothing(self):
        _, problems, _ = ds.plan("ALICE-LOL", {}, set(), {}, True, False, ["ALICE-Physics"])
        self.assertTrue(any("0 compared" in p for p in problems))

    def test_a_present_sibling_is_not_placed_again(self):
        out, problems, _ = ds.plan("ALICE-LOL", self.NEEDED, {"ALICE-Sign"}, {"ALICE-Sign": SHA}, True, False, [])
        self.assertEqual((out, problems), ([], []))

    def test_a_malformed_or_repeated_pin_line_fails(self):
        _, problems = ds.read_pins(f"ALICE-Sign {SHA[:12]}\nALICE-X {SHA}\nALICE-X {SHA}\n")
        self.assertEqual(len(problems), 2)


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(d, msg):
    git("-C", d, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", msg)
    return git("-C", d, "rev-parse", "HEAD")


class Cli(unittest.TestCase):
    """the command against real git checkouts in a layout directory"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def repo(self, name, manifests=None, pin_text=None):
        d = os.path.join(self.layout, name)
        os.makedirs(d, exist_ok=True)
        git("init", "-q", d)
        for rel, text in (manifests or {}).items():
            os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
            with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                f.write(text)
        if pin_text is not None:
            os.makedirs(os.path.join(d, ".github"), exist_ok=True)
            with open(os.path.join(d, ds.PIN_FILE), "w", encoding="utf-8") as f:
                f.write(pin_text)
        git("-C", d, "add", "-A")
        return commit(d, "one")

    def run_cli(self, consumer="ALICE-LOL", *extra):
        return subprocess.run([sys.executable, os.path.join(HERE, "downstream_siblings.py"),
                               "--layout", self.layout, "--consumer", consumer, *extra],
                              capture_output=True, text=True)

    def lol(self, pin_text, manifests=MANIFESTS):
        return self.repo("ALICE-LOL", manifests, pin_text)

    def test_prints_the_siblings_to_place(self):
        self.lol(f"ALICE-Sign {SHA}\n")
        r = self.run_cli("ALICE-LOL", "--require", "ALICE-Physics")
        self.assertEqual((r.returncode, r.stdout), (0, f"ALICE-Sign {SHA}\n"), r.stderr)

    def test_fails_without_the_pin_file(self):
        self.lol(None)
        r = self.run_cli()
        self.assertEqual((r.returncode, r.stdout), (1, ""), r.stderr)
        self.assertIn("is missing", r.stderr)

    def test_fails_with_an_empty_pin_file(self):
        self.lol("")
        r = self.run_cli()
        self.assertEqual((r.returncode, r.stdout), (1, ""), r.stderr)
        self.assertIn("ALICE-Sign (path dependency of ALICE-LOL/core/Cargo.toml) is not pinned", r.stderr)

    def test_fails_with_a_malformed_pin_line(self):
        self.lol(f"ALICE-Sign {SHA[:12]}\n")
        r = self.run_cli()
        self.assertEqual((r.returncode, r.stdout), (1, ""), r.stderr)
        self.assertIn("is not '<Repo> <40-hex SHA>'", r.stderr)

    def test_fails_on_a_path_sibling_the_pin_file_does_not_list(self):
        m = dict(MANIFESTS, **{"core/Cargo.toml": CORE + 'more = { path = "../../ALICE-Unlisted", optional = true }\n'})
        self.lol(f"ALICE-Sign {SHA}\n", m)
        r = self.run_cli()
        self.assertEqual((r.returncode, r.stdout), (1, ""), r.stderr)
        self.assertIn("ALICE-Unlisted (path dependency of ALICE-LOL/core/Cargo.toml) is not pinned", r.stderr)

    def test_a_placed_sibling_that_needs_another_is_followed(self):
        sign = self.repo("ALICE-Sign", {"Cargo.toml": '[dependencies]\nh = { path = "../ALICE-Hash" }\n'})
        self.lol(f"ALICE-Sign {sign}\nALICE-Hash {SHA}\n")
        r = self.run_cli()
        self.assertEqual((r.returncode, r.stdout), (0, f"ALICE-Hash {SHA}\n"), r.stderr)

    def test_unpinned_main_for_a_consumer_without_pins_says_so(self):
        self.repo("ALICE-TRT", {"Cargo.toml": '[dependencies]\nml = { path = "../ALICE-ML" }\n'})
        r = self.run_cli("ALICE-TRT", "--unpinned-main")
        self.assertEqual((r.returncode, r.stdout), (0, "ALICE-ML main\n"), r.stderr)
        self.assertIn("notice: ALICE-TRT has no", r.stderr)

    def test_verify_passes_when_every_sibling_is_at_its_pin(self):
        sign = self.repo("ALICE-Sign")
        self.lol(f"ALICE-Sign {sign}\n")
        r = self.run_cli("ALICE-LOL", "--verify")
        self.assertEqual((r.returncode, r.stdout), (0, ""), r.stderr)

    def test_verify_fails_when_a_sibling_head_is_not_the_pin(self):
        sign = self.repo("ALICE-Sign")
        self.lol(f"ALICE-Sign {sign}\n")
        commit(os.path.join(self.layout, "ALICE-Sign"), "two")
        r = self.run_cli("ALICE-LOL", "--verify")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn(f"is not ALICE-LOL's pin {sign}", r.stderr)

    def test_verify_fails_when_a_needed_sibling_is_not_placed(self):
        self.lol(f"ALICE-Sign {SHA}\n")
        r = self.run_cli("ALICE-LOL", "--verify")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("not placed yet: ALICE-Sign", r.stderr)

    def test_verify_fails_on_a_placed_repository_the_consumer_does_not_pin(self):
        self.repo("ALICE-Sign")
        self.repo("ALICE-Stray")
        sign = git("-C", os.path.join(self.layout, "ALICE-Sign"), "rev-parse", "HEAD")
        self.lol(f"ALICE-Sign {sign}\n")
        r = self.run_cli("ALICE-LOL", "--verify")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("ALICE-Stray: placed but not pinned", r.stderr)


if __name__ == "__main__":
    unittest.main()
