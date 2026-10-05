#!/usr/bin/env python3
"""Lint the public documents for vocabulary and CHANGELOG structure.

The repository is public; its documents are read by people and by language
models judging whether the project is real. Two classes of problem are
checked mechanically:

  * vocabulary  development-process words that do not belong in a public
                document (agent / process names, internal tracker names,
                device names, private network addresses, unrelated private
                projects) in README.md, README_JP.md, docs/MODULES.md and
                CHANGELOG.md
  * changelog   version headings: no duplicates, `[Unreleased]` first,
                released versions in descending semver order, the Cargo.toml
                version either released or covered by `[Unreleased]`; inside
                `[Unreleased]`, each Keep a Changelog category at most once
                and no emoji status markers

Released sections are history and are not rewritten, so the per-category and
emoji checks apply to `[Unreleased]` only. docs/ROADMAP.md is not linted yet:
its existing text is kept as it is, and a gate that is red on day one checks
nothing new.

Every check must compare at least one item; a check that compared nothing
fails, so a renamed file or heading cannot turn the gate into a no-op.

Usage: `python3 scripts/docs_lint.py --check` (exit 1 on any finding).
`--root DIR` runs against another tree (used by scripts/test_docs_lint.py).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

DOCS = ("README.md", "README_JP.md", "docs/MODULES.md", "CHANGELOG.md")

# (label, pattern). Common English words such as "memory" and "session" are
# matched only in their internal-process phrasings, so "memory footprint" or a
# netcode "session" are not flagged.
FORBIDDEN = [
    ("agent process", re.compile(r"\bworkers?\b|ワーカー|\bsubagents?\b|調停役|並走\s*session|他\s*session|session\s*名")),
    ("session name", re.compile(r"\bys-[0-9a-f]{2}\b|\bsakamoro-[0-9a-f]{2}\b")),
    ("internal tracker", re.compile(r"\bBacklog\b|memory\s*(?:側|化|に|へ|file)")),
    ("instruction source", re.compile(r"user\s*(?:指示|裁定|指摘)")),
    ("device", re.compile(r"\bJetson\b|\bMac\s?mini\b|\bMacBook\b|Raspberry\s*Pi|EAC-4000|reCamera")),
    ("private address", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]

# Names of unrelated private projects and of internal infrastructure are not written
# here in plain text: this file
# is public, and a list of names would itself disclose them. Each entry is the
# SHA-256 of a lower-case word, or of a two- or three-word phrase joined by one
# space; documents are tokenised and every 1-, 2- and 3-gram is hashed and looked up.
PRIVATE_NAME_HASHES = {
    "76ed632eacc9561fddba93918e12fa408df40a2f9e8e038c7803346d9e2b9ab3",
    "8c2dda2f006c7c4f0e8875c47387901590ff92de46261dbf15c325dc5b3ff8aa",
    "2ef7d5809068e897d756dd958a81b31e3811673a716eb80b96680436550e096b",
    "db526bfe062629b1fd8b8bf3cd70bd6cdbd37e1dd1e81d080fdd29de2a6873cd",
    "80d387ccb3289295ecace879442a42cc8cdc1ff10a0047b9b2b9b11f569142c1",
    "d79d1f8b76c9a4f5546474bf8cfea05cf1a9d3ef73698de05d5b6ae502861708",
    "4a3eb33262fe03814a0fdc3ca9bb5ed5cdff99d8e28fc5788ebe767c0b2158fa",
    "c7b00200a27d82bed0ee6fcaa71a29c535f41e839248aee0e3526c81396fd64d",
    "e1eeef0077e8f80d6f71c92be9758592699ed9171c22e9be78978616ea8e5686",
    "a3c1038a8770ed804b6544ad1a0100415f7ef7ae012f89253d0cd09675c59107",
    "3925ad1a1f1767a30bfb7b606c8bd43bed9123e245bc46fd11950ffb7f0a9745",
    "1de7dcbdf455feccbb694bf1fba9a65830d84ab96b16603624ea33f8cb417802",
    "28bb90e451eea46b2ef4000ce1872e29f7e5a8aecbf94aaf184f5536ebcdb7b5",
    "b4a959fb606739afc8c9f69bc1a3fca53807bdefd44ea46e352c6c84b45c05e0",
    "e78a14d392a171bb45a5bcb8d6d4e3a685893ea6e1a04b811066c34873e0c311",
    "33bdb6df44ce6cd93063bc0ca38b52808f72825c837bac136a3b45720022ef56",
    "9c118c2d1b0d10df8c67942b4cb330fbcaeee47dabb27d9729a706825036ab74",
    # assistant / tooling names: no form of them belongs anywhere in the tree
    "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
    "b5cf43ae07a7364e0c0ca9e838f01f278fc6a71c207a6f8c3de8d908608b2db1",
    "28e174396028f226b3bead259d19749205378d9204ce12fd9b1918ab6032a15d",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u3040-\u30ff\u4e00-\u9fff]+")


def private_names(line: str) -> list[str]:
    """Tokens / phrases of the line whose hash is a private-project name."""
    toks = TOKEN_RE.findall(line)
    found = []
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            phrase = " ".join(toks[i:i + n])
            if hashlib.sha256(phrase.lower().encode("utf-8")).hexdigest() in PRIVATE_NAME_HASHES:
                found.append(phrase)
    return found

CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
EMOJI_RE = re.compile("[⚠✅❌⭐\U0001f300-\U0001faff]")
VERSION_RE = re.compile(r"^## \[([^\]]+)\]", re.M)


def read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def strip_code(text: str) -> str:
    """Blank fenced code blocks (identifiers in code are not prose), keeping line numbers."""
    return re.sub(r"^```.*?^```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.M | re.S)


def semver_key(v: str) -> tuple:
    core, _, pre = v.partition("-")
    nums = tuple(int(x) for x in re.findall(r"\d+", core)[:3])
    # a pre-release sorts below the release it precedes
    return nums + ((1,) if not pre else (0, tuple(int(x) if x.isdigit() else x for x in re.split(r"[.]", pre))))


def cargo_version(root: str) -> str | None:
    m = re.search(r'^\[package\]\s*$(.*?)^\[', read(root, "Cargo.toml"), re.M | re.S)
    if not m:
        return None
    v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
    return v.group(1) if v else None


def unreleased_body(changelog: str) -> str | None:
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog, re.M | re.S)
    return m.group(1) if m else None


def check(root: str) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"vocabulary": 0, "versions": 0, "categories": 0}

    for rel in DOCS:
        if not os.path.exists(os.path.join(root, rel)):
            errors.append(f"{rel}: missing")
            continue
        text = strip_code(read(root, rel))
        for i, line in enumerate(text.splitlines(), 1):
            counts["vocabulary"] += 1
            for label, pat in FORBIDDEN:
                for m in pat.finditer(line):  # every hit: one name must not hide another on the same line
                    errors.append(f"{rel}:{i}: {label} `{m.group(0)}` in a public document")
            for name in private_names(line):
                errors.append(f"{rel}:{i}: private or internal name in a public document (`{name}`)")

    if os.path.exists(os.path.join(root, "CHANGELOG.md")):
        cl = read(root, "CHANGELOG.md")
        heads = VERSION_RE.findall(cl)
        counts["versions"] = len(heads)
        dup = sorted({h for h in heads if heads.count(h) > 1})
        if dup:
            errors.append(f"CHANGELOG.md: version headings appear twice: {dup}")
        if heads and heads[0] != "Unreleased":
            errors.append(f"CHANGELOG.md: the first version heading is [{heads[0]}], not [Unreleased]")
        released = [h for h in heads if h != "Unreleased"]
        for a, b in zip(released, released[1:]):
            if semver_key(a) <= semver_key(b):
                errors.append(f"CHANGELOG.md: [{a}] is listed above [{b}] but is not newer")
        cv = cargo_version(root)
        if cv and cv not in released and "Unreleased" not in heads:
            errors.append(f"CHANGELOG.md: Cargo.toml version {cv} has no section and there is no [Unreleased]")
        if cv and released and cv not in released and semver_key(cv) < semver_key(released[0]):
            errors.append(f"CHANGELOG.md: Cargo.toml version {cv} is older than the newest section [{released[0]}]")
        body = unreleased_body(cl)
        if body is not None:
            cats = re.findall(r"^### (\w+)", body, re.M)
            counts["categories"] = len(cats)
            for c in sorted(set(cats)):
                if c not in CATEGORIES:
                    errors.append(f"CHANGELOG.md [Unreleased]: `### {c}` is not a Keep a Changelog category")
                elif cats.count(c) > 1:
                    errors.append(f"CHANGELOG.md [Unreleased]: `### {c}` appears {cats.count(c)} times (append to one list)")
            for i, line in enumerate(body.splitlines(), 1):
                if EMOJI_RE.search(line):
                    errors.append(f"CHANGELOG.md [Unreleased] line {i}: emoji status marker (use **Behavior change:** / **Breaking:**)")
        else:
            counts["categories"] = 1  # no unreleased work: nothing to compare, not an error

    # private / internal names (hashed above) anywhere in the tree: file paths and
    # the text of every tracked file, not only the four documents. Generated
    # ledgers, comments, workflows and scripts are published too.
    counts["tree files"] = 0
    for rel in tree_files(root):
        counts["tree files"] += 1
        for name in private_names(" ".join(re.split(r"[/_.\-]+", rel))):
            errors.append(f"{rel}: private or internal name in a file path (`{name}`)")
        if rel in DOCS:
            continue  # their text is checked above (with code blocks set aside)
        text = read_text_file(os.path.join(root, rel))
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name in private_names(line):
                errors.append(f"{rel}:{i}: private or internal name (`{name}`)")

    for name, c in counts.items():
        if c == 0:
            errors.append(f"check `{name}` compared nothing")
    return errors, counts


SKIP_TREE = {".git", "target", "node_modules"}


def tree_files(root: str) -> list[str]:
    """Tracked files when `root` is a git work tree, otherwise every file under it
    (the tests run on plain directories)."""
    import subprocess
    r = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True)
    if r.returncode == 0 and r.stdout:
        return sorted(p for p in r.stdout.decode("utf-8", "replace").split("\0") if p)
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_TREE]
        out += [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for f in files]
    return sorted(out)


def read_text_file(path: str) -> str | None:
    """The file as UTF-8 text, or None for a binary / large / missing file."""
    try:
        if os.path.getsize(path) > 4 * 1024 * 1024:
            return None
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    errors, counts = check(root)
    print("compared: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if "--check" in sys.argv and errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
