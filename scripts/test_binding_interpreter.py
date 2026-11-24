#!/usr/bin/env python3
"""Tests of scripts/binding_interpreter.py and of the two places scripts/preflight.sh uses it.

Author: Moroya Sakamoto
"""
from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binding_interpreter as bi  # noqa: E402

PREFLIGHT = (Path(__file__).resolve().parent / "preflight.sh").read_text(encoding="utf-8")


class Probe(unittest.TestCase):
    def test_the_probe_reads_a_real_interpreter(self):
        self.assertEqual(bi.probe(sys.executable)[1:], tuple(sys.version_info[:2]))


# the stand-in interpreters are sh scripts and the venv layout is bin/python: scripts/preflight.sh, the one
# caller, is a bash script of the unix legs
@unittest.skipIf(os.name == "nt", "sh stand-ins; preflight.sh does not run on Windows")
class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, name: str, answer: str = "CPython 3 12", rc: int = 0) -> str:
        """an executable that answers the probe with `answer`"""
        p = self.d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"#!/bin/sh\necho '{answer}'\nexit {rc}\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return str(p)

    def venv(self, answer: str = "CPython 3 12", rc: int = 0) -> Path:
        self.fake("venv/bin/python", answer, rc)
        return self.d / "venv"


    def test_the_venv_s_interpreter_is_the_one_named_when_the_venv_exists(self):
        line = bi.check(self.venv(), {}, which=lambda _: self.fake("other/python3", "CPython 3 9"))
        self.assertIn(f"{self.d / 'venv' / 'bin' / 'python'} (CPython 3.12; the venv's interpreter)", line)
        self.assertIn("PYO3_PYTHON is not set", line)

    def test_without_a_venv_it_is_the_python3_that_will_make_it_and_none_is_an_error(self):
        made_by = self.fake("path/python3", "CPython 3 13")
        line = bi.check(self.d / "venv", {}, which=lambda _: made_by)
        self.assertIn(f"{made_by} (CPython 3.13; python3, which will make the venv)", line)
        with self.assertRaisesRegex(bi.Mismatch, "no python3 on PATH"):
            bi.check(self.d / "venv", {}, which=lambda _: None)

    def test_a_pyo3_python_of_another_version_is_refused_and_the_same_version_passes(self):
        venv = self.venv("CPython 3 12")
        for answer in ("CPython 3 14", "CPython 2 12", "PyPy 3 12"):
            with self.subTest(answer=answer):
                other = self.fake("elsewhere/python", answer)
                with self.assertRaisesRegex(bi.Mismatch, rf"PYO3_PYTHON={other} is .* and the binding step uses "
                                                         r".*venv/bin/python \(CPython 3\.12.*Unset PYO3_PYTHON"):
                    bi.check(venv, {"PYO3_PYTHON": other})
        same = self.fake("elsewhere/python", "CPython 3 12")
        self.assertIn(f"PYO3_PYTHON={same} is the same version", bi.check(venv, {"PYO3_PYTHON": same}))
        # and against the python3 that will make the venv, when there is none yet
        with self.assertRaisesRegex(bi.Mismatch, "python3, which will make the venv"):
            bi.check(self.d / "none", {"PYO3_PYTHON": same}, which=lambda _: self.fake("path/python3", "CPython 3 14"))

    def test_an_interpreter_that_does_not_run_or_does_not_answer_is_an_error_not_a_pass(self):
        with self.assertRaisesRegex(bi.Mismatch, "PYO3_PYTHON: .*does not run.*unset it"):
            bi.check(self.venv(), {"PYO3_PYTHON": str(self.d / "missing")})
        for answer, rc in (("", 0), ("CPython 3", 0), ("CPython three 12", 0), ("CPython 3 12", 1)):
            with self.subTest(answer=answer, rc=rc):
                with self.assertRaisesRegex(bi.Mismatch, "does not answer as a Python interpreter"):
                    bi.check(self.venv(), {"PYO3_PYTHON": self.fake("bad/python", answer, rc)})
                with self.assertRaisesRegex(bi.Mismatch, "does not answer.*remove .*venv"):
                    bi.check(self.venv(answer, rc), {})

    def test_exit_codes(self):
        out, err = io.StringIO(), io.StringIO()
        old = os.environ.pop("PYO3_PYTHON", None)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(bi.main(["--venv", str(self.venv())]), 0)
                os.environ["PYO3_PYTHON"] = self.fake("elsewhere/python", "CPython 3 14")
                self.assertEqual(bi.main(["--venv", str(self.d / "venv")]), 1)
        finally:
            os.environ.pop("PYO3_PYTHON", None)
            if old is not None:
                os.environ["PYO3_PYTHON"] = old
        self.assertIn("binding interpreter: ", out.getvalue())
        self.assertIn("binding_interpreter: PYO3_PYTHON=", err.getvalue())


class Preflight(unittest.TestCase):
    VENV = '"${CARGO_TARGET_DIR:-target}/python-binding-venv"'

    def test_the_full_run_asks_before_its_first_step_and_the_other_modes_do_not(self):
        ask = ('if [[ $quick -eq 0 && $fast -eq 0 ]]; then\n'
               f'  python3 scripts/binding_interpreter.py --venv {self.VENV}\nfi\n')
        self.assertEqual(PREFLIGHT.count(ask), 1)
        self.assertLess(PREFLIGHT.index(ask), PREFLIGHT.index('\nstep "'), "before the first step")
        self.assertEqual(PREFLIGHT.count("python3 scripts/binding_interpreter.py"), 1)
        self.assertIn("set -euo pipefail\n", PREFLIGHT[:PREFLIGHT.index(ask)])

    def test_the_binding_step_gives_maturin_the_venv_s_interpreter_and_its_venv_is_the_one_asked_about(self):
        step = PREFLIGHT[PREFLIGHT.index('step "Python binding (maturin wheel'):]
        step = step[:step.index('\nstep "', 10)]
        self.assertIn('py_target="${CARGO_TARGET_DIR:-target}"\npy_venv="$py_target/python-binding-venv"\n', step)
        build = [ln for ln in step.splitlines() if "maturin" in ln and " build " in ln]
        self.assertEqual(len(build), 1, build)
        self.assertTrue(build[0].startswith('"$py_venv/bin/maturin" build -i "$py_venv/bin/python" '), build[0])
        self.assertIn('"$py_venv/bin/pip" install -q --force-reinstall', step)
        self.assertIn('"$py_venv/bin/python" python/tests/test_binding.py', step)


if __name__ == "__main__":
    unittest.main()
