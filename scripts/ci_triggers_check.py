#!/usr/bin/env python3
"""docs/CI_TRIGGERS.md and .github/ci-evidence.toml agree with the workflows.

For every workflow in .github/workflows the triggers are read from its `on:` block
(as text, no YAML library on the runners) and compared with its row in the table of
docs/CI_TRIGGERS.md; the evidence class of the row is compared with
.github/ci-evidence.toml. The checks:

  * the workflows, the rows of the table and the classes of the evidence file name
    the same set of workflows;
  * every cell of a row equals what the workflow file says (push branches or tags,
    path gate, pull_request, schedule, dispatch, behaviour on a ci/** push, class);
  * the class fits the triggers: landing and after-landing run on ci/**, schedule-only
    has no push trigger, tag-only is pushed by tags only;
  * every after-landing leg names a landing workflow, one of its jobs and a matrix value
    of that job, and the second table of the document lists the same legs;
  * a workflow that runs on ci/** cannot cancel a main run: its concurrency group
    includes the ref, or `cancel-in-progress` is false.

Fails (exit 1) on a finding, and when no row was compared.

  python3 scripts/ci_triggers_check.py            # exit 1 on a finding
  python3 scripts/ci_triggers_check.py --root DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    print("error: ci_triggers_check.py needs Python 3.11 or newer (tomllib)", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths as cp  # noqa: E402
from ci_load_check import jobs, scalar, top_block  # noqa: E402

DOC = "docs/CI_TRIGGERS.md"
EVIDENCE = ".github/ci-evidence.toml"
CLASSES = ("landing", "after-landing", "schedule-only", "tag-only")
COLUMNS = ("workflow", "push", "path gate", "pull_request", "schedule", "dispatch", "ci/** push", "class")
LEG_COLUMNS = ("workflow", "job", "matrix")
WF_TABLE = "ci-triggers:workflows"
LEG_TABLE = "ci-triggers:after-landing-legs"


def derive(name: str, text: str) -> dict[str, str]:
    """the cells of a workflow's row, from its file"""
    on = cp.on_block(text)
    push = cp.sub_block(on, "push", 2)
    branches = cp.push_branches(push) if push is not None else None
    if push is None:
        push_cell, on_ci = "-", False
    elif branches is not None:
        push_cell, on_ci = ", ".join(branches), "ci/**" in branches
    else:
        tags = cp.sub_block(push, "tags", 4)
        if tags is not None:
            push_cell, on_ci = "tags " + ", ".join(cp.list_items(tags)), False
        else:
            push_cell, on_ci = "every branch", True
    rel = f".github/paths/{name.rsplit('.', 1)[0]}.txt"
    gate = rel if f"--patterns-file {rel}" in text else "-"
    pr = cp.sub_block(on, "pull_request", 2)
    if pr is None:
        pr_cell = "-"
    else:
        pr_branches = cp.list_items(cp.sub_block(pr, "branches", 4) or [])
        m = next((re.match(r"^    branches:\s*\[(.*)\]", l) for l in pr if re.match(r"^    branches:\s*\[", l)), None)
        if m:
            pr_branches = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        pr_cell = ", ".join(pr_branches) or "every branch"
        if cp.sub_block(pr, "paths", 4) is not None:
            pr_cell += " (paths)"
    has = lambda key: any(re.match(rf"^  {key}:", l) for l in on)  # noqa: E731
    ci_cell = "-" if not on_ci else ("runs when the paths match" if gate != "-" else "runs")
    return {
        "workflow": name,
        "push": push_cell,
        "path gate": gate,
        "pull_request": pr_cell,
        "schedule": "yes" if has("schedule") else "-",
        "dispatch": "yes" if has("workflow_dispatch") else "-",
        "ci/** push": ci_cell,
    }


def table(doc: str, marker: str, columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    """the rows of the table between `<!-- marker:begin -->` and `<!-- marker:end -->`"""
    m = re.search(rf"<!-- {re.escape(marker)}:begin -->\n(.*?)<!-- {re.escape(marker)}:end -->", doc, re.S)
    if not m:
        return [], [f"{DOC}: no table between <!-- {marker}:begin --> and <!-- {marker}:end -->"]
    lines = [l for l in m.group(1).splitlines() if l.strip().startswith("|")]
    errors: list[str] = []
    if len(lines) < 2:
        return [], [f"{DOC}: table {marker} has no header"]
    head = [c.strip().lower() for c in lines[0].strip().strip("|").split("|")]
    if tuple(head) != columns:
        return [], [f"{DOC}: table {marker} columns {head} != {list(columns)}"]
    rows = []
    for l in lines[2:]:
        cells = [c.strip().replace("`", "") for c in l.strip().strip("|").split("|")]
        if len(cells) != len(columns):
            errors.append(f"{DOC}: row {l.strip()!r} has {len(cells)} cells, expected {len(columns)}")
            continue
        rows.append(dict(zip(columns, cells)))
    return rows, errors


def concurrency_errors(name: str, text: str) -> list[str]:
    conc = top_block(text, "concurrency")
    group, cancel = scalar(conc, "group"), scalar(conc, "cancel-in-progress")
    if group is None or "github.ref" in group or cancel in (None, "false"):
        return []
    return [f"{name}: runs on ci/** with concurrency group {group} shared by every ref and "
            f"cancel-in-progress {cancel}: a ci/** push can cancel the main run"]


def check(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    wf_dir = root / ".github" / "workflows"
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(wf_dir.glob("*.yml"))}
    try:
        ev = tomllib.loads((root / EVIDENCE).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"{EVIDENCE}: {e}"], 0
    classes = ev.get("workflows", {})
    legs = ev.get("after-landing-legs", [])
    try:
        doc = (root / DOC).read_text(encoding="utf-8")
    except OSError as e:
        return [f"{DOC}: {e}"], 0
    rows, errs = table(doc, WF_TABLE, COLUMNS)
    errors += errs
    leg_rows, errs = table(doc, LEG_TABLE, LEG_COLUMNS)
    errors += errs

    names = set(files)
    for label, got in ((EVIDENCE, set(classes)), (DOC, {r["workflow"] for r in rows})):
        for n in sorted(names - got):
            errors.append(f"{label}: no entry for workflow {n}")
        for n in sorted(got - names):
            errors.append(f"{label}: entry for {n}, which is not a workflow")
    seen = set()
    for r in rows:
        if r["workflow"] in seen:
            errors.append(f"{DOC}: workflow {r['workflow']} listed twice")
        seen.add(r["workflow"])

    compared = 0
    for r in rows:
        name = r["workflow"]
        if name not in files:
            continue
        compared += 1
        want = derive(name, files[name])
        cls = classes.get(name)
        want["class"] = cls if isinstance(cls, str) else "?"
        for col in COLUMNS[1:]:
            if r[col] != want[col]:
                errors.append(f"{DOC}: {name}: `{col}` is {r[col]!r}, the workflow says {want[col]!r}")
        if cls not in CLASSES:
            errors.append(f"{EVIDENCE}: {name}: class {cls!r} is not one of {', '.join(CLASSES)}")
        elif cls in ("landing", "after-landing") and want["ci/** push"] == "-":
            errors.append(f"{name}: class {cls} but it does not run on a ci/** push")
        elif cls == "schedule-only" and want["push"] != "-":
            errors.append(f"{name}: class schedule-only but it triggers on push")
        elif cls == "tag-only" and not want["push"].startswith("tags "):
            errors.append(f"{name}: class tag-only but its push trigger is {want['push']!r}")
        if want["ci/** push"] != "-":
            errors += concurrency_errors(name, files[name])

    want_legs = set()
    for leg in legs:
        w, j, mval = leg.get("workflow"), leg.get("job"), leg.get("matrix")
        want_legs.add((w, j, mval))
        if classes.get(w) != "landing":
            errors.append(f"{EVIDENCE}: after-landing leg of {w!r}, which is not a landing workflow")
            continue
        js = jobs(files.get(w, ""))
        if j not in js:
            errors.append(f"{EVIDENCE}: after-landing leg {w}/{j}: no job {j!r}")
        elif not re.search(rf"(?<![\w.-]){re.escape(str(mval))}(?![\w.-])", js[j]):
            errors.append(f"{EVIDENCE}: after-landing leg {w}/{j}: matrix value {mval!r} not in the job")
    doc_legs = {(r["workflow"], r["job"], r["matrix"]) for r in leg_rows}
    for leg in sorted(want_legs - doc_legs, key=str):
        errors.append(f"{DOC}: after-landing leg {leg} missing from the table")
    for leg in sorted(doc_legs - want_legs, key=str):
        errors.append(f"{DOC}: after-landing leg {leg} is not in {EVIDENCE}")

    if compared == 0:
        errors.append("no workflow row compared (compared nothing)")
    return errors, compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    errors, n = check(Path(args.root))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: workflow rows {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
