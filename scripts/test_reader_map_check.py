#!/usr/bin/env python3
"""Oracle for scripts/reader_map_check.py, on throwaway repositories: a mapped tree passes; an
unmapped file, a stale entry, an unknown reader, a reader that does not run on ci/**, a
reader without the entry's evidence, a script no job runs, a short or stale `[[unread]]`
reason, an empty map and an empty tree each fail with their own message. The evidence of a
reader is its own steps or a script it runs: a path a script only mentions does not count."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_map_check as rm  # noqa: E402

CLASSES = '[workflows]\n"ci.yml" = "landing"\n"weekly.yml" = "schedule-only"\n'
CI = """name: CI
on:
  push:
    branches: [main, 'ci/**']
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
      - run: python3 scripts/runner.py
"""
WEEKLY = """name: Weekly
on:
  schedule:
    - cron: '0 0 * * 0'
jobs:
  deep:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test --release
"""
MAP = """[[entry]]
name = "sources"
paths = ["src/**"]
readers = ["ci.yml:test"]
evidence = "cargo test"

[[entry]]
name = "scripts"
paths = ["scripts/*"]
scripts = true

[[entry]]
name = "repository files"
paths = [".github/**"]
readers = ["ci.yml:test"]
evidence = "runner.py"
"""


class Repo:
    def __init__(self, files: dict[str, str]):
        self.d = Path(tempfile.mkdtemp())
        for path, text in files.items():
            (self.d / path).parent.mkdir(parents=True, exist_ok=True)
            (self.d / path).write_text(text, encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.d)], check=True)
        subprocess.run(["git", "-C", str(self.d), "add", "-A"], check=True)

    def check(self):
        return rm.check(self.d)


def base(**over) -> dict[str, str]:
    files = {
        ".github/ci-evidence.toml": CLASSES,
        ".github/workflows/ci.yml": CI,
        ".github/workflows/weekly.yml": WEEKLY,
        ".github/reader-map.toml": MAP,
        "src/lib.rs": "pub fn f() {}\n",
        "scripts/runner.py": "import helper\nprint(open('scripts/baseline.txt').read())\n",
        "scripts/helper.py": "X = 1\n",
        "scripts/baseline.txt": "0\n",
    }
    files.update(over)
    return {k: v for k, v in files.items() if v is not None}


class ReaderMap(unittest.TestCase):
    def errors(self, **over) -> list[str]:
        try:
            return Repo(base(**over)).check()[0]
        except Exception as e:  # noqa: BLE001 - a crash is a failed assertion, not a pass
            self.fail(f"the checker raised {e!r}")

    def test_a_mapped_tree_passes_and_counts(self):
        errors, counts = Repo(base()).check()
        self.assertEqual(errors, [])
        self.assertEqual(counts["files"], 8)
        self.assertEqual(counts["entries"], 3)
        self.assertGreaterEqual(counts["scripts"], 3)

    def test_an_unmapped_file_fails(self):
        self.assertIn("docs/x.md: no reader (add it to .github/reader-map.toml with the job that reads it)",
                      self.errors(**{"docs/x.md": "x\n"}))

    def test_a_stale_entry_fails(self):
        errs = self.errors(**{".github/reader-map.toml": MAP + '[[entry]]\nname = "gone"\npaths = ["old/**"]\n'
                              'readers = ["ci.yml:test"]\nevidence = "cargo"\n'})
        self.assertTrue(any("(gone): matches no tracked file" in e for e in errs), errs)

    def test_an_unknown_reader_fails(self):
        m = MAP.replace('readers = ["ci.yml:test"]\nevidence = "cargo test"', 'readers = ["ci.yml:nope"]\nevidence = "cargo test"')
        errs = self.errors(**{".github/reader-map.toml": m})
        self.assertTrue(any("reader ci.yml:nope is not a job" in e for e in errs), errs)
        self.assertIn("src/lib.rs: no reader (add it to .github/reader-map.toml with the job that reads it)", errs)

    def test_a_reader_that_does_not_run_on_ci_fails(self):
        m = MAP.replace('readers = ["ci.yml:test"]\nevidence = "cargo test"', 'readers = ["weekly.yml:deep"]\nevidence = "cargo test"')
        errs = self.errors(**{".github/reader-map.toml": m})
        self.assertTrue(any("reader weekly.yml:deep is class schedule-only" in e for e in errs), errs)

    def test_a_reader_without_the_evidence_fails(self):
        m = MAP.replace('evidence = "cargo test"', 'evidence = "cargo bench"')
        errs = self.errors(**{".github/reader-map.toml": m})
        self.assertTrue(any("does not show `cargo bench`" in e for e in errs), errs)

    def test_the_evidence_may_be_in_a_script_the_job_runs_or_imports(self):
        m = MAP.replace('evidence = "cargo test"', 'evidence = "X = 1"')  # (in helper.py, imported by runner.py)
        self.assertEqual(self.errors(**{".github/reader-map.toml": m}), [])

    def test_a_path_a_script_only_mentions_does_not_make_the_job_a_reader(self):
        # runner.py mentions scripts/other.py by path but does not run or import it
        errs = self.errors(**{"scripts/runner.py": "import helper\n# see scripts/other.py\nprint(open('scripts/baseline.txt').read())\n",
                              "scripts/other.py": "SECRET_MARK = 2\n",
                              ".github/reader-map.toml": MAP.replace('evidence = "cargo test"', 'evidence = "SECRET_MARK"')})
        self.assertTrue(any("does not show `SECRET_MARK`" in e for e in errs), errs)

    def test_a_script_no_job_runs_fails(self):
        errs = self.errors(**{"scripts/orphan.py": "print(1)\n"})
        self.assertTrue(any("scripts/orphan.py is run or named by no job" in e for e in errs), errs)

    def test_data_named_by_a_script_a_job_runs_is_read(self):
        self.assertEqual(self.errors(), [])  # (baseline.txt is named by runner.py)
        errs = self.errors(**{"scripts/runner.py": "import helper\n"})
        self.assertTrue(any("scripts/baseline.txt is run or named by no job" in e for e in errs), errs)

    def test_unread_with_a_reason_is_counted_not_silent(self):
        m = MAP + '[[unread]]\npaths = ["docs/**"]\nreason = "kept for the record, not checked"\n'
        errors, counts = Repo(base(**{"docs/x.md": "x\n", ".github/reader-map.toml": m})).check()
        self.assertEqual(errors, [])
        self.assertEqual(counts["unread"], 1)

    def test_unread_without_a_reason_stale_or_over_a_read_file_fails(self):
        short = MAP + '[[unread]]\npaths = ["docs/**"]\nreason = "later"\n'
        self.assertTrue(any("no reason" in e for e in self.errors(**{"docs/x.md": "x\n", ".github/reader-map.toml": short})))
        stale = MAP + '[[unread]]\npaths = ["old/**"]\nreason = "kept for the record, not checked"\n'
        self.assertTrue(any("unread 1: matches no tracked file" in e for e in self.errors(**{".github/reader-map.toml": stale})))
        over = MAP + '[[unread]]\npaths = ["src/**"]\nreason = "kept for the record, not checked"\n'
        self.assertTrue(any("src/lib.rs has a reader" in e for e in self.errors(**{".github/reader-map.toml": over})))

    def crate(self, **over):
        ci = CI + "      - run: cargo test --quiet --manifest-path tools/lint/Cargo.toml\n"
        m = MAP + ('[[entry]]\nname = "probes"\npaths = ["tools/probes/**"]\nreaders = ["ci.yml:test"]\n'
                   'crate-tests = "tools/lint"\nevidence = "probes"\nread-ext = "rs"\n'
                   '[[entry]]\nname = "lint"\npaths = ["tools/lint/**"]\nreaders = ["ci.yml:test"]\nevidence = "tools/lint"\n')
        files = {".github/workflows/ci.yml": ci, ".github/reader-map.toml": m,
                 "tools/lint/Cargo.toml": "[package]\n", "tools/lint/src/main.rs": 'const D: &str = "../probes";\n',
                 "tools/probes/a.rs": "// a\n", "tools/probes/b.rs": "// b\n"}
        files.update(over)
        return self.errors(**files)

    def test_a_crate_whose_tests_read_the_files_is_their_reader(self):
        self.assertEqual(self.crate(), [])

    def test_a_crate_reader_that_does_not_run_the_crate_tests_fails(self):
        errs = self.crate(**{".github/workflows/ci.yml": CI})
        self.assertTrue(any("does not run `cargo test --manifest-path tools/lint/Cargo.toml`" in e for e in errs), errs)

    def test_a_crate_whose_source_does_not_name_the_files_fails(self):
        errs = self.crate(**{"tools/lint/src/main.rs": "fn main() {}\n"})
        self.assertTrue(any("no source of tools/lint names `probes`" in e for e in errs), errs)

    def test_a_file_the_crate_tests_skip_fails(self):
        errs = self.crate(**{"tools/probes/notes.txt": "x\n"})
        self.assertTrue(any("1 matched file(s) are not .rs" in e for e in errs), errs)

    def test_an_empty_map_and_an_empty_tree_fail(self):
        self.assertTrue(any("no entry" in e for e in self.errors(**{".github/reader-map.toml": "# nothing\n"})))
        d = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        errors, counts = rm.check(d)
        self.assertEqual(counts["files"], 0)
        self.assertTrue(errors)
        self.assertEqual(rm.main(["--root", str(d)]), 1)


if __name__ == "__main__":
    unittest.main()
