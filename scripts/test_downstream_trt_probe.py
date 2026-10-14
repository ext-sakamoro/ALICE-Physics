#!/usr/bin/env python3
"""Tests for scripts/downstream_trt_probe.sh: the GPU probe of downstream_check.sh.

The function is sourced into bash with a stand-in `cargo` that either fails to build
(the message cargo gives for a dependency whose rust-version is newer than the
toolchain) or builds and prints / does not print an adapter. A probe that does not
build must be reported as that build failure with the compiler's message, never as a
missing adapter; a probe that builds and prints nothing must still be a missing
adapter. test_mutant_* turn the script into its wrong forms and require the checks to
notice, so they cannot pass on a script that merges the two.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "downstream_trt_probe.sh")

MSRV = (
    "error: rustc 1.92.0 is not supported by the following package:\n"
    "  alice-ml@0.3.0 requires rustc 1.98.1\n"
)

# FAKE_BUILD: ok | fail ; FAKE_RUN: adapter | empty | exit1
DRIVER = r"""
set -uo pipefail
. "$PROBE"
cargo() {
    case "$1" in
        build)
            if [ "$FAKE_BUILD" = fail ]; then printf '%s' "$FAKE_BUILD_ERR" >&2; return 101; fi
            return 0 ;;
        run)
            case "$FAKE_RUN" in
                adapter) echo "Mesa llvmpipe (Vulkan)"; return 0 ;;
                empty) return 0 ;;
                exit1) echo "no adapter found" >&2; return 1 ;;
            esac ;;
    esac
}
failed=(); summary=()
trt_gpu_probe
echo "RC_FAILED=${failed[*]-}"
echo "RC_SUMMARY=${summary[*]-}"
"""


def run_probe(build: str, run: str, script: str = PROBE) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "logs"))
        os.makedirs(os.path.join(root, "ALICE-TRT"))
        env = dict(
            os.environ,
            PROBE=script,
            ROOT=root,
            TRT=os.path.join(root, "ALICE-TRT"),
            FAKE_BUILD=build,
            FAKE_RUN=run,
            FAKE_BUILD_ERR=MSRV,
        )
        env.pop("TRT_GPU_PROBE", None)
        p = subprocess.run(["bash", "-c", DRIVER], env=env, capture_output=True, text=True, check=False)
        return p.stdout, p.stderr


def failures(script: str = PROBE) -> list[str]:
    """What the script gets wrong (empty list: nothing)."""
    wrong = []
    out, err = run_probe("fail", "adapter", script)
    if "RC_FAILED=trt (GPU probe build failed)" not in out:
        wrong.append(f"build error not its own failure: {out!r}")
    if "alice-ml@0.3.0 requires rustc 1.98.1" not in err or "rustc 1.92.0 is not supported" not in err:
        wrong.append(f"build error message lost: {err!r}")
    if "no GPU adapter" in err or "no GPU adapter" in out:
        wrong.append("build error reported as a missing adapter")
    for run in ("empty", "exit1"):
        out, err = run_probe("ok", run, script)
        if "RC_FAILED=trt (no GPU adapter)" not in out:
            wrong.append(f"{run}: not reported as no adapter: {out!r}")
        if "ALICE-TRT finds no GPU adapter" not in err:
            wrong.append(f"{run}: no-adapter message missing: {err!r}")
        if "failed to build" in err:
            wrong.append(f"{run}: reported as a build failure")
    out, err = run_probe("ok", "adapter", script)
    if "RC_FAILED=\n" not in out:
        wrong.append(f"adapter found but failed: {out!r}")
    if "trt: GPU adapter: Mesa llvmpipe (Vulkan)" not in out:
        wrong.append(f"adapter not reported: {out!r}")
    return wrong


class ProbeTest(unittest.TestCase):
    def test_script_is_correct(self) -> None:
        self.assertEqual(failures(), [])

    def mutated(self, old: str, new: str, *more: str) -> list[str]:
        with open(PROBE, encoding="utf-8") as f:
            text = f.read()
        self.assertIn(old, text, "mutation target missing: the test is out of date")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.sh")
            with open(path, "w", encoding="utf-8") as f:
                text = text.replace(old, new, 1)
                pairs = list(zip(more[::2], more[1::2]))
                for o, n in pairs:
                    self.assertIn(o, text, "mutation target missing: the test is out of date")
                    text = text.replace(o, n, 1)
                f.write(text)
            return failures(path)

    def test_mutant_build_error_reported_as_no_adapter(self) -> None:
        # the old behaviour: the build and the run are one step, any failure is "no adapter"
        self.assertNotEqual(self.mutated('failed+=("trt (GPU probe build failed)")',
                                         'failed+=("trt (no GPU adapter)")'), [])

    def test_mutant_build_error_ignored(self) -> None:
        self.assertNotEqual(self.mutated('if ! (cd "$TRT" && cargo build', 'if (cd "$TRT" && cargo build'), [])

    def test_mutant_message_dropped(self) -> None:
        self.assertNotEqual(self.mutated('asked: ${msg:-$(tail -n 1 "$build_log")}', 'asked',
                                         '    cat "$build_log" >&2\n', ""), [])

    def test_mutant_empty_output_passes(self) -> None:
        self.assertNotEqual(self.mutated('&& [ -n "$out" ]', ""), [])

    def test_mutant_build_failure_not_red(self) -> None:
        self.assertNotEqual(self.mutated('failed+=("trt (GPU probe build failed)")\n', ""), [])


if __name__ == "__main__":
    sys.exit(unittest.main())
