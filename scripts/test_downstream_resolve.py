#!/usr/bin/env python3
"""Tests for scripts/downstream_resolve.sh: the lock move and the resolve guard of
downstream_check.sh.

The functions are sourced into bash and run against a small consumer crate that
takes `alice-physics` 3.0.1 from a local git repository (a file:// source, so
nothing is fetched from the network) with its Cargo.lock committed to that
version, and a `[patch]` that points at a path package 3.1.0, the shape of
ALICE-SDF against a minor bump of this crate: cargo keeps the locked version and
leaves the patch unused. The guard must refuse that, `relock` must move the lock
so the guard accepts it, and a failed update must be reported. The order in
downstream_check.sh is checked too: every step that patches alice-physics
relocks before its resolve check, and at least one does.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_downstream_trt_probe import lf_copy, posix, resolve_bash  # noqa: E402

RESOLVE = os.path.join(HERE, "downstream_resolve.sh")
CHECK = os.path.join(HERE, "downstream_check.sh")
BASH = resolve_bash()

DRIVER = r"""
set -euo pipefail
. "$RESOLVE"
failed=()
for step in $STEPS; do
    case "$step" in
        relock) relock c "$DIR" --config "$PATCH" ;;
        resolve) check_resolved c "$DIR" --config "$PATCH" ;;
    esac
done
printf 'FAILED=%s\n' "${failed[*]:-}"
"""


def manifest(name: str, version: str, deps: str = "") -> str:
    return f'[package]\nname = "{name}"\nversion = "{version}"\nedition = "2021"\npublish = false\n\n[workspace]\n{deps}'


def git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=cwd, check=True, capture_output=True)


class Fixture:
    """upstream (git, 3.0.1), physics (path, 3.1.0), consumer locked to the upstream."""

    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="downstream-resolve-")
        up, phys, self.consumer = (os.path.join(self.root, d) for d in ("up", "phys", "consumer"))
        for d, text in ((up, manifest("alice-physics", "3.0.1")), (phys, manifest("alice-physics", "3.1.0"))):
            os.makedirs(os.path.join(d, "src"))
            with open(os.path.join(d, "Cargo.toml"), "w", encoding="utf-8") as f:
                f.write(text)
            with open(os.path.join(d, "src", "lib.rs"), "w", encoding="utf-8") as f:
                f.write("pub fn f() {}\n")
        git(up, "init", "-q")
        git(up, "add", "-A")
        git(up, "commit", "-qm", "upstream")
        url = "file:///" + posix(up).lstrip("/")
        os.makedirs(os.path.join(self.consumer, "src"))
        with open(os.path.join(self.consumer, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write(manifest("consumer", "0.0.0", f'\n[dependencies]\nalice-physics = {{ git = "{url}", version = "3" }}\n'))
        with open(os.path.join(self.consumer, "src", "lib.rs"), "w", encoding="utf-8") as f:
            f.write("")
        subprocess.run(["cargo", "generate-lockfile", "-q"], cwd=self.consumer, check=True, capture_output=True)
        self.physics = posix(phys)
        self.patch = f'patch."{url}".alice-physics.path="{self.physics}"'

    def lock(self) -> str:
        with open(os.path.join(self.consumer, "Cargo.lock"), encoding="utf-8") as f:
            return f.read()

    def run(self, steps: str, directory: str | None = None, resolve: str = RESOLVE,
            physics_dir: str | None = None) -> tuple[str, str]:
        env = dict(os.environ, RESOLVE=posix(lf_copy(resolve)), DIR=posix(directory or self.consumer),
                   PATCH=self.patch, STEPS=steps, PHYSICS_DIR=physics_dir or self.physics,
                   PYTHON=posix(sys.executable))
        env.pop("CARGO_TARGET_DIR", None)
        p = subprocess.run([BASH, "-c", DRIVER], env=env, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise AssertionError(f"driver exited {p.returncode}: {p.stderr}")
        failed = re.search(r"^FAILED=(.*)$", p.stdout, re.M)
        return (failed.group(1) if failed else "<none>"), p.stdout + p.stderr

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class Resolve(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_a_lock_on_the_older_version_leaves_the_patch_unused_and_the_guard_refuses(self):
        failed, out = self.fx.run("resolve")
        self.assertEqual(failed, "c (resolve)", out)
        self.assertIn("3.0.1", out)

    def test_relock_moves_the_lock_and_the_guard_accepts(self):
        before = self.fx.lock()
        failed, out = self.fx.run("relock resolve")
        self.assertEqual(failed, "", out)
        self.assertIn("v3.1.0", out)
        self.assertNotEqual(before, self.fx.lock())
        self.assertIn('version = "3.1.0"', self.fx.lock())

    def test_the_guard_compares_directories_not_spellings(self):
        # cargo prints the path in its own form (on Windows a backslash path with 8.3
        # short names); another spelling of the same directory is the same checkout
        other = self.fx.physics + "/../" + os.path.basename(self.fx.physics)
        failed, out = self.fx.run("relock resolve", physics_dir=other)
        self.assertEqual(failed, "", out)

    def test_the_guard_refuses_another_directory_with_the_same_version(self):
        copy = os.path.join(self.fx.root, "copy")
        shutil.copytree(self.fx.physics, copy)
        failed, out = self.fx.run("relock resolve", physics_dir=posix(copy))
        self.assertEqual(failed, "c (resolve)", out)
        self.assertIn("v3.1.0", out)

    def test_a_failed_update_is_reported(self):
        empty = tempfile.mkdtemp(prefix="downstream-resolve-empty-")
        try:
            failed, out = self.fx.run("relock", directory=empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
        self.assertEqual(failed, "c (lock)", out)

    def test_mutant_relock_without_the_update_is_refused(self):
        with open(RESOLVE, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(text.count('(cd "$dir" && cargo update -p alice-physics "$@")'), 1)
        mutant = os.path.join(self.fx.root, "mutant.sh")
        with open(mutant, "w", encoding="utf-8") as f:
            f.write(text.replace('(cd "$dir" && cargo update -p alice-physics "$@")', "true"))
        failed, out = self.fx.run("relock resolve", resolve=mutant)
        self.assertEqual(failed, "c (resolve)", out)


def patched_steps(text: str) -> list[tuple[str, int, bool]]:
    """(label, line, relocked first) for every check_resolved that patches alice-physics."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = re.match(r'\s*check_resolved\s+(\S+)\s+(\S+).*--config\s+"\$PATCH"', line)
        if not m:
            continue
        label, directory = m.groups()
        prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        relocked = bool(re.match(rf'\s*relock\s+{re.escape(label)}\s+{re.escape(directory)}\s+--config\s+"\$PATCH"\s*$', prev))
        out.append((label, i + 1, relocked))
    return out


class Order(unittest.TestCase):
    def read(self) -> str:
        with open(CHECK, encoding="utf-8") as f:
            return f.read()

    def test_every_patched_resolve_check_is_relocked_first(self):
        steps = patched_steps(self.read())
        self.assertGreater(len(steps), 0, "no step patches alice-physics (compared nothing)")
        for label, line, relocked in steps:
            with self.subTest(label=label):
                self.assertTrue(relocked, f"downstream_check.sh:{line}: {label} is not relocked before check_resolved")

    def test_mutant_without_the_relock_line_is_caught(self):
        text = re.sub(r'(?m)^\s*relock sdf .*\n', "", self.read())
        self.assertTrue(any(not r for _, _, r in patched_steps(text)))

    def test_the_functions_come_from_the_sourced_file_only(self):
        text = self.read()
        self.assertIn('. "$PHYSICS_DIR/scripts/downstream_resolve.sh"', text)
        self.assertNotRegex(text, r"(?m)^\s*(relock|check_resolved)\s*\(\)")


if __name__ == "__main__":
    unittest.main()
