#!/usr/bin/env python3
"""Oracle for scripts/reader_map_check.py, on throwaway repositories. A file is read only when a landing job names its
path (in a run step outside comments, or in a script that job runs) or a declared read passes its token check. The
first three cases are the ways a reader was claimed without reading: a free word as evidence, a shell comment, a
script named in a comment."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_map_check as rm  # noqa: E402

CLASSES = '[workflows]\n"ci.yml" = "landing"\n"weekly.yml" = "schedule-only"\n'


def workflow(*runs: str, name: str = "CI", job: str = "test", extra: str = "") -> str:
    steps = "".join(f"      - run: |\n" + "".join(f"          {line}\n" for line in r.split("\n")) for r in runs)
    return (f"name: {name}\non:\n  push:\n    branches: [main, 'ci/**']\njobs:\n  {job}:\n    runs-on: ubuntu-latest\n"
            f"    steps:\n{steps}{extra}")


WEEKLY = "name: Weekly\non:\n  schedule:\n    - cron: '0 0 * * 0'\njobs:\n  deep:\n    runs-on: ubuntu-latest\n    steps:\n" \
         "      - run: python3 scripts/weekly.py\n"


class Repo:
    def __init__(self, files: dict[str, str]):
        self.d = Path(tempfile.mkdtemp())
        for path, text in files.items():
            (self.d / path).parent.mkdir(parents=True, exist_ok=True)
            (self.d / path).write_text(text, encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.d)], check=True)
        subprocess.run(["git", "-C", str(self.d), "add", "-A"], check=True)


# the weekly job runs scripts/weekly.py: a scheduled read, declared in every case's map
SCHEDULED = ('[[entry]]\nname = "weekly"\nkind = "scheduled"\npaths = ["scripts/weekly.py"]\n'
             'readers = ["weekly.yml:deep"]\n')


def base(ci: str, **over) -> dict[str, str]:
    if ".github/reader-map.toml" in over:
        over[".github/reader-map.toml"] = SCHEDULED + over[".github/reader-map.toml"]
    files = {
        ".github/ci-evidence.toml": CLASSES,
        ".github/workflows/ci.yml": ci,
        ".github/workflows/weekly.yml": WEEKLY,
        ".github/reader-map.toml": SCHEDULED,
        "scripts/weekly.py": "print(1)\n",
    }
    files.update(over)
    return {k: v for k, v in files.items() if v is not None}


# the repository files every case has: the workflows, the classes and the map are read by the check itself only in
# the real repository, so each case declares them read through a step that names them
FRAME = "cat .github/ci-evidence.toml .github/workflows/ci.yml .github/workflows/weekly.yml .github/reader-map.toml"


class ReaderMap(unittest.TestCase):
    def run_check(self, ci_runs: list[str], **files) -> tuple[list[str], dict[str, int]]:
        try:
            return rm.check(Repo(base(workflow(FRAME, *ci_runs), **files)).d)
        except Exception as e:  # noqa: BLE001 - a crash is a failed assertion, not a pass
            self.fail(f"the checker raised {e!r}")

    def unread(self, errs: list[str]) -> list[str]:
        return [e.split(": no reader")[0] for e in errs if ": no reader" in e]

    # --- the three ways a reader was claimed without reading ---

    def test_a_free_word_is_not_a_read(self):
        m = '[[entry]]\nname = "src"\nkind = "dir"\ndir = "src"\npaths = ["src/**"]\nreaders = ["ci.yml:test"]\n'
        errs, _ = self.run_check(["cargo test --workspace"], **{"src/lib.rs": "", ".github/reader-map.toml": m})
        self.assertIn("src/lib.rs", self.unread(errs))
        self.assertTrue(any("names no `src`" in e for e in errs), errs)

    def test_a_shell_comment_names_nothing(self):
        errs, _ = self.run_check(["echo hi  # reads data/a.txt", "# cat data/b.txt"],
                                 **{"data/a.txt": "", "data/b.txt": ""})
        self.assertEqual(sorted(self.unread(errs)), ["data/a.txt", "data/b.txt"])
        errs, _ = self.run_check(["cat data/a.txt data/b.txt"], **{"data/a.txt": "", "data/b.txt": ""})
        self.assertEqual(errs, [])

    def test_a_script_named_in_a_comment_is_not_run(self):
        files = {"scripts/reader.py": 'open("data/a.txt")\n', "data/a.txt": ""}
        errs, _ = self.run_check(["# python3 scripts/reader.py", "true"], **files)
        self.assertEqual(sorted(self.unread(errs)), ["data/a.txt", "scripts/reader.py"])
        errs, _ = self.run_check(["python3 scripts/reader.py"], **files)
        self.assertEqual(errs, [])

    # --- derived reads ---

    def test_a_python_comment_or_docstring_is_not_a_use(self):
        files = {"scripts/reader.py": '"""reads data/a.txt"""\n# data/b.txt\nX = 1\n', "data/a.txt": "", "data/b.txt": ""}
        errs, _ = self.run_check(["python3 scripts/reader.py"], **files)
        self.assertEqual(sorted(self.unread(errs)), ["data/a.txt", "data/b.txt"])

    def test_a_path_name_imported_from_a_module_is_a_use(self):
        files = {"scripts/lib.py": 'PATH = "data/a.txt"\nOTHER = "data/b.txt"\n', "scripts/reader.py": "import lib\nopen(lib.PATH)\n",
                 "data/a.txt": "", "data/b.txt": ""}
        errs, _ = self.run_check(["python3 scripts/reader.py"], **files)
        self.assertEqual(self.unread(errs), ["data/b.txt"], "only the name the reader uses is read")

    def test_a_sibling_named_by_its_file_name_is_read_and_a_named_script_runs(self):
        files = {"scripts/a.sh": 'here=$(dirname "$0")\npython3 "$here/b.py"\n',
                 "scripts/b.py": 'from pathlib import Path\nopen(Path(__file__).parent / "base.txt")\n',
                 "scripts/base.txt": "0\n", "scripts/c.py": "print(1)\n"}
        errs, _ = self.run_check(["bash scripts/a.sh"], **files)
        self.assertEqual(self.unread(errs), ["scripts/c.py"])

    def test_a_directory_path_a_reader_uses_is_read_whole(self):
        files = {"scripts/reader.py": 'ROOT = None\nD = ROOT / "docs" / "coverage"\n', "docs/coverage/a.toml": "",
                 "docs/coverage/b.toml": "", "docs/other.md": ""}
        errs, _ = self.run_check(["python3 scripts/reader.py"], **files)
        self.assertEqual(self.unread(errs), ["docs/other.md"], "a two-component directory is read whole, not its parent")

    def test_a_job_that_does_not_run_on_ci_reads_nothing(self):
        errs, _ = self.run_check(["true"], **{"scripts/weekly.py": 'open("data/a.txt")\n', "data/a.txt": ""})
        self.assertEqual(self.unread(errs), ["data/a.txt"])

    def test_a_working_directory_is_a_cd(self):
        m = '[[entry]]\nname = "web"\nkind = "dir"\ndir = "web"\npaths = ["web/**"]\nreaders = ["ci.yml:test"]\n'
        extra = "      - working-directory: web\n        run: npm ci\n"
        files = base(workflow(FRAME, extra=extra), **{"web/a.ts": "", ".github/reader-map.toml": m})
        errs, _ = rm.check(Repo(files).d)
        self.assertEqual(errs, [])

    # --- declared reads ---

    def cargo(self, runs: list[str], manifest: str, **files):
        m = (f'[[entry]]\nname = "crate"\nkind = "cargo"\nmanifest = "{manifest}"\npaths = ["Cargo.toml", "src/**", "t/**"]\n'
             'readers = ["ci.yml:test"]\n')
        return self.run_check(runs, **{"Cargo.toml": "", "src/lib.rs": "", "t/Cargo.toml": "", "t/src/main.rs": "",
                                       ".github/reader-map.toml": m, **files})

    def test_cargo_reads_the_root_workspace_or_a_named_manifest(self):
        self.assertEqual(self.cargo(["cargo test --workspace"], "Cargo.toml")[0], [])
        errs, _ = self.cargo(["cargo test --manifest-path t/Cargo.toml"], "Cargo.toml")
        self.assertTrue(any("runs no cargo command on Cargo.toml" in e for e in errs), errs)
        self.assertEqual(self.cargo(["cargo test --manifest-path t/Cargo.toml"], "t/Cargo.toml")[0], [])
        self.assertEqual(self.cargo(["cd t\ncargo build"], "t/Cargo.toml")[0], [])
        errs, _ = self.cargo(["cargo test"], "t/Cargo.toml")
        self.assertTrue(any("runs no cargo command on t/Cargo.toml" in e for e in errs), errs)

    def test_rust_literals_of_cargo_sources_resolve_relative_to_the_file(self):
        rs = 'const A: &[u8] = include_bytes!("../data/a.bin");\n// include_bytes!("../data/c.bin")\n'
        errs, _ = self.cargo(["cargo test"], "Cargo.toml", **{"src/lib.rs": rs, "data/a.bin": "", "data/c.bin": ""})
        self.assertEqual(self.unread(errs), ["data/c.bin"], "a literal in a comment is not a read")

    def test_all_tracked_needs_a_script_that_lists_the_tracked_files(self):
        m = ('[[entry]]\nname = "text"\nkind = "all-tracked"\npaths = ["NOTES.md"]\nreaders = ["ci.yml:test"]\n')
        lister = 'import subprocess\nsubprocess.run(["git", "ls-files", "-z"])\n'
        errs, _ = self.run_check(["python3 scripts/lint.py"], **{"NOTES.md": "", ".github/reader-map.toml": m,
                                                                 "scripts/lint.py": lister})
        self.assertEqual(errs, [])
        errs, _ = self.run_check(["python3 scripts/lint.py"], **{"NOTES.md": "", ".github/reader-map.toml": m,
                                                                 "scripts/lint.py": "# git ls-files\nprint(1)\n"})
        self.assertTrue(any("lists the tracked files" in e for e in errs), errs)

    def test_a_declared_reader_must_exist_and_run_on_ci(self):
        for reader, msg in (("ci.yml:nope", "is not a job"), ("weekly.yml:deep", "is class schedule-only")):
            m = f'[[entry]]\nname = "x"\nkind = "dir"\ndir = "web"\npaths = ["web/**"]\nreaders = ["{reader}"]\n'
            errs, _ = self.run_check(["cd web"], **{"web/a.ts": "", ".github/reader-map.toml": m})
            self.assertTrue(any(msg in e for e in errs), (reader, errs))

    def test_a_stale_entry_and_an_unknown_kind_fail(self):
        m = '[[entry]]\nname = "gone"\nkind = "dir"\ndir = "old"\npaths = ["old/**"]\nreaders = ["ci.yml:test"]\n'
        errs, _ = self.run_check(["true"], **{".github/reader-map.toml": m})
        self.assertTrue(any("(gone): matches no tracked file" in e for e in errs), errs)
        m = '[[entry]]\nname = "k"\nkind = "magic"\npaths = ["web/**"]\nreaders = ["ci.yml:test"]\n'
        errs, _ = self.run_check(["true"], **{"web/a.ts": "", ".github/reader-map.toml": m})
        self.assertTrue(any("unknown kind 'magic'" in e for e in errs), errs)

    def test_unread_with_a_reason_is_counted_and_a_bad_one_fails(self):
        m = '[[unread]]\npaths = ["notes/**"]\nreason = "kept for the record, not checked"\n'
        errs, counts = self.run_check(["true"], **{"notes/a.md": "", ".github/reader-map.toml": m})
        self.assertEqual(errs, [])
        self.assertEqual(counts["unread"], 1)
        short = '[[unread]]\npaths = ["notes/**"]\nreason = "later"\n'
        errs, _ = self.run_check(["true"], **{"notes/a.md": "", ".github/reader-map.toml": short})
        self.assertTrue(any("no reason" in e for e in errs), errs)
        over = '[[unread]]\npaths = ["data/**"]\nreason = "kept for the record, not checked"\n'
        errs, _ = self.run_check(["cat data/a.txt"], **{"data/a.txt": "", ".github/reader-map.toml": over})
        self.assertTrue(any("data/a.txt has a reader" in e for e in errs), errs)

    def test_a_scheduled_read_is_counted_apart_and_must_still_be_named(self):
        errs, counts = self.run_check(["true"])
        self.assertEqual(errs, [])
        self.assertEqual(counts["scheduled"], 1)
        weekly = WEEKLY.replace("python3 scripts/weekly.py", "true")
        errs, _ = rm.check(Repo(base(workflow(FRAME), **{".github/workflows/weekly.yml": weekly})).d)
        self.assertTrue(any("reader weekly.yml:deep does not name scripts/weekly.py" in e for e in errs), errs)
        m = '[[entry]]\nname = "x"\nkind = "scheduled"\npaths = ["data/a.txt"]\nreaders = ["ci.yml:test"]\n'
        errs, _ = self.run_check(["cat data/a.txt"], **{"data/a.txt": "", ".github/reader-map.toml": m})
        self.assertTrue(any("is class landing, not schedule-only" in e for e in errs), errs)

    def pinned(self, checker_text: str, sha: str | None = None, names_pin: bool = True):
        import hashlib
        import os
        src = Path(tempfile.mkdtemp())
        (src / "law").mkdir()
        g = ["git", "-C", str(src / "law"), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", str(src / "law")], check=True)
        (src / "law" / "scripts").mkdir()
        (src / "law" / "scripts" / "check.py").write_text(checker_text, encoding="utf-8")
        subprocess.run(g + ["add", "-A"], check=True)
        subprocess.run(g + ["commit", "-qm", "c"], check=True)
        rev = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        subprocess.run(g + ["config", "uploadpack.allowReachableSHA1InWant", "true"], check=True)
        want = sha or hashlib.sha256(checker_text.encode()).hexdigest()
        m = ('[[entry]]\nname = "g11"\nkind = "pinned-checker"\npin = ".github/pin"\nrepo = "law"\n'
             'checker = "scripts/check.py"\npaths = ["conformance/t.toml"]\nreaders = ["ci.yml:test"]\n')
        old = os.environ.get("READER_MAP_GIT_BASE")
        os.environ["READER_MAP_GIT_BASE"] = str(src)
        try:
            return self.run_check(["bash scripts/g.sh"], **{
                "conformance/t.toml": "", ".github/pin": f"rev {rev}\nchecker-sha256 {want}\n",
                "scripts/g.sh": "awk '{print}' .github/pin\n" if names_pin else "true\n", ".github/reader-map.toml": m})[0]
        finally:
            if old is None:
                os.environ.pop("READER_MAP_GIT_BASE")
            else:
                os.environ["READER_MAP_GIT_BASE"] = old

    def test_a_pinned_checker_reads_what_its_pinned_bytes_name(self):
        self.assertEqual(self.pinned('T = "conformance/t.toml"\n'), [])
        errs = self.pinned('# conformance/t.toml\nT = 1\n')
        self.assertTrue(any("does not name conformance/t.toml" in e for e in errs), errs)
        errs = self.pinned('T = "conformance/t.toml"\n', sha="0" * 64)
        self.assertTrue(any("does not have the pinned sha256" in e for e in errs), errs)
        errs = self.pinned('T = "conformance/t.toml"\n', names_pin=False)
        self.assertTrue(any("runs nothing that names .github/pin" in e for e in errs), errs)

    def test_nothing_read_fails(self):
        d = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        errs, counts = rm.check(d)
        self.assertEqual(counts["files"], 0)
        self.assertTrue(errs)
        self.assertEqual(rm.main(["--root", str(d)]), 1)


if __name__ == "__main__":
    unittest.main()
