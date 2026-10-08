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

  * contracts   every `pub trait` written in a ```rust block of
                docs/ECOSYSTEM_CONTRACTS.md matches the trait in src/: the
                header (bounds), the set of methods, each method signature
                (whitespace and `crate::` paths normalised) and which methods
                have a default body (written `{ ... }` in the document)

The development-process, tracker, instruction-source and device vocabulary is
also checked in the text of every other tracked file (docs/ROADMAP.md, comments,
workflows, generated ledgers); private names are checked in every file path and
text.

Released sections are history and are not rewritten, so the per-category and
emoji checks apply to `[Unreleased]` only.

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
# English words written next to Japanese text: `\b` sees no boundary between
# "worker" and "が" (both are word characters), so ASCII-letter boundaries are used
A = r"(?<![A-Za-z])"
Z = r"(?![A-Za-z])"


def words(*alts: str) -> str:
    return A + "(?:" + "|".join(alts) + ")" + Z


FORBIDDEN = [
    ("agent process", re.compile(words(r"workers?", r"subagents?", r"multi[- ]?agents?", r"skills?")
                                 # `git worktree add` is a command, not a description of how the work was done
                                 + r"|(?<!git )" + words(r"worktrees?")
                                 + r"|ワーカー|マルチエージェント|エージェント|調停役|並走\s*session|他\s*session|session\s*名")),
    ("session name", re.compile(r"(?<![A-Za-z0-9])(?:ys|sakamoro)-[0-9a-f]{2}(?![A-Za-z0-9])")),
    ("internal tracker", re.compile(words(r"Backlog") + r"|memory\s*(?:側|化|に|へ|file)|(?<![A-Za-z0-9_])memory/")),
    # names of private notes: `[[snake_case]]` links and the note-file prefixes
    ("internal note", re.compile(r"\[\[[a-z0-9]+_[a-z0-9_]+\]\]"
                                 r"|(?<![A-Za-z0-9_])(?:feedback|discipline|success|handoff)_[a-z0-9_]{4,}"
                                 r"|(?<![A-Za-z0-9_])(?:project|reference)_alice_[a-z0-9_]+"
                                 # numbered entries of an internal pitfall catalogue / CI template
                                 r"|罠\s*#\s*\d+|canonical\s+(?:CI\s+)?template")),
    ("instruction source", re.compile(r"user\s*(?:指示|裁定|指摘|判断)")),
    ("device", re.compile(words(r"Jetson", r"Mac\s?mini", r"MacBook") + r"|Raspberry\s*Pi")),
    ("private address", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]

# Labels checked in the text of every tracked file, not only the public documents:
# how the work was organised does not belong in comments, ledgers or the roadmap
# either. Addresses stay document-only (dotted numbers occur in data files).
TREE_LABELS = ("agent process", "session name", "internal tracker", "internal note", "instruction source", "device")
# These files define and test the vocabulary, so they spell it out
TREE_VOCAB_EXEMPT = {"scripts/docs_lint.py", "scripts/test_docs_lint.py", "scripts/test_land.py"}

# Names of unrelated private projects and of internal infrastructure are not written
# here in plain text: this file
# is public, and a list of names would itself disclose them. Each entry is the
# SHA-256 of a lower-case word, or of a two- or three-word phrase joined by one
# space; documents are tokenised and every 1-, 2- and 3-gram is hashed and looked up.
# Owned equipment and private network names (generic products stay in the
# `device` pattern above); hashed like the private names below and merged into them.
DEVICE_NAME_HASHES = {
    "1034f65141df5f4ca8ef4b136e1a4d67f70ef8f7bfd16f91f6d94d980c8061c3",
    "af1c54d629fb3db8ecaa658f846b5b3f8ef314e6b01d1494bafac6af8a9183f1",
    "699074f67aea6aa9733bfbf42ff376f92b7762049c5ef5136256908c298b7514",
    "795b104abe3e4134960ca245ded0e617f162c751209347b7f003bc35e062f43e",
    "791dc82c39c048682e2c8bcc953dba44838997987e5678c161f257d6fc27bf4a",
}
PRIVATE_NAME_HASHES = DEVICE_NAME_HASHES | {
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
    # titles and settings of a work that are not to appear in public text
    "19f798ce9f70706dd4334369cbf056e51922830d480360b8ace258e818ba7dd4",
    "f20f5b0dfb30b315a2adacb9fd240ce53301a01a4cbc4c16b26c1c64c815eaa8",
    "e3638dd52d300c1393f35ef855ed00ae35e66b48acc8a8fbfc6b5dfba335593b",
    "1f770aff55b29cfc7d0e86b0d15f0ca06b523420221794dbb9a4f028a698203a",
    "cb49d686203bf58098b7db4cc309f826ee749366c217d7f108cad562507feccd",
    "f59367a3ffb393334570507ae7d5d0f7b1b92b0b7dd8a9a229c0d3f1df416388",
    "96ac2ef53544a3e2fbe76274f9124be1e78c878f03bdf2ae6ed9aa7864dddecd",
    "8653d44dc6fd3b625cba9865fc10f6b8d2d9d871d5da9e3624bc6d87179ed8f1",
    # internal review / rule names
    "d44eb131a9fc729aae3aed377733d242c0796290775259142e0bd8d26e1e3132",
    "9330fafbe2c0a50b50ab63cbc545e79f87b76394b7efa9b6f1dc90f4ce76bfb1",
    "3834bde4d50c16031f0b0c9ff9371cd8aaa10155dbbf964f199a5b965e369925",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u3040-\u30ff\u4e00-\u9fff]+")
# Japanese is written without spaces, so a name is usually glued to a particle or
# another word ("<name>の", "<name>編"); the second tokenisation splits at every
# change of script (Latin / katakana / hiragana / kanji) so such a name is still a token
SCRIPT_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u30a0-\u30ff]+|[\u3040-\u309f]+|[\u4e00-\u9fff]+")


def private_names(line: str) -> list[str]:
    """Tokens / phrases of the line whose hash is a private-project name."""
    found: list[str] = []
    for toks in (TOKEN_RE.findall(line), SCRIPT_TOKEN_RE.findall(line)):
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                phrase = " ".join(toks[i:i + n])
                if phrase not in found and \
                        hashlib.sha256(phrase.lower().encode("utf-8")).hexdigest() in PRIVATE_NAME_HASHES:
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
        if body is not None and body.strip():
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
    counts["tree vocabulary"] = 0
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
            if rel in TREE_VOCAB_EXEMPT:
                continue
            counts["tree vocabulary"] += 1
            for label, pat in FORBIDDEN:
                if label in TREE_LABELS:
                    for m in pat.finditer(line):
                        errors.append(f"{rel}:{i}: {label} `{m.group(0)}`")

    for name, c in counts.items():
        if c == 0:
            errors.append(f"check `{name}` compared nothing")
    return errors, counts


CONTRACT_DOCS = ("docs/ECOSYSTEM_CONTRACTS.md",)
RUST_BLOCK_RE = re.compile(r"^```rust[^\n]*\n(.*?)^```", re.M | re.S)
TRAIT_NAME_RE = re.compile(r"\bpub\s+(?:unsafe\s+)?trait\s+(\w+)")
# a char literal (`'"'`, `'\''`, `'\u{1F600}'`); a lifetime (`'a`) does not match
CHAR_LIT_RE = re.compile(r"'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]{1,6}\}|.)|[^\\'\n])'")
RAW_STR_RE = re.compile(r'b?r(#*)"')


def strip_rust_comments(text: str) -> str:
    """Drop `/* */` (nested) and `//` comments (doc comments included), outside
    string, raw string and char literals."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        raw = RAW_STR_RE.match(text, i) if c in "br" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) else None
        if raw:
            close = '"' + raw.group(1)
            j = text.find(close, raw.end())
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif c == "'":
            m = CHAR_LIT_RE.match(text, i)
            j = m.end() if m else i + 1
            out.append(text[i:j])
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def normalise_sig(sig: str) -> str:
    s = re.sub(r"\bcrate::(?:\w+::)*", "", sig)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([(),:;<>&\[\]=+!{}])\s*", r"\1", s)
    s = s.replace(",)", ")").replace(",>", ">")
    return s


FN_ITEM_RE = re.compile(r'(?:(?:const|async|unsafe|extern(?:\s*"[^"]*")?)\s+)*fn\s+(\w+)')
TYPE_ITEM_RE = re.compile(r"type\s+(\w+)")
CONST_ITEM_RE = re.compile(r"const\s+(\w+)\s*:")


def _matching(text: str, i: int, open_: str, close: str) -> int:
    """Index just past the bracket that closes the one at `i`."""
    depth = 0
    while i < len(text):
        depth += text[i] == open_
        depth -= text[i] == close
        i += 1
        if depth == 0:
            break
    return i


def trait_items(text: str, name: str) -> tuple[str, list[tuple[str, str, bool]]] | None:
    """(header, [(item key, signature, has default)]) of `pub trait name` in
    comment-free `text`, or None when the trait is not there. The items are the
    associated fns (key `fn name`, signature with `const` / `async` / `unsafe` /
    `extern "ABI"`, default = a body), types (`type name`, signature with the
    bounds, default = `= T`) and consts (`const name`, signature with the type,
    default = `= value`); anything else in the body is kept under its own text so
    a document that adds or drops it differs."""
    m = re.search(rf"\bpub\s+(?:unsafe\s+)?trait\s+{re.escape(name)}\b", text)
    if not m:
        return None
    brace = text.find("{", m.end())
    if brace < 0:
        return None
    header = normalise_sig(text[m.start():brace])
    end = _matching(text, brace, "{", "}") - 1
    items, i = [], brace + 1
    while i < end:
        if text[i].isspace() or text[i] == ";":
            i += 1
            continue
        if text.startswith("#", i):  # attribute
            j = text.find("[", i)
            i = _matching(text, j, "[", "]") if 0 <= j < end else i + 1
            continue
        # the item ends at `;`, or at the end of a `{ }` body, outside () [];
        # after `=` (a default type or value) braces are part of the value
        # (`<` `>` count as brackets in a type or const item, so `Item = u8` in a
        # bound is not taken for a default)
        assoc = bool(TYPE_ITEM_RE.match(text, i) or CONST_ITEM_RE.match(text, i))
        j, nest, eq = i, 0, -1
        while j < end:
            ch = text[j]
            if ch in "([" or (assoc and ch == "<"):
                nest += 1
            elif ch in ")]" or (assoc and ch == ">" and text[j - 1] != "-"):
                nest -= 1
            elif assoc and nest == 0 and ch == "=" and eq < 0:
                eq = j
            elif nest == 0 and ch == "{" and eq < 0:
                break
            elif ch == "{":
                j = _matching(text, j, "{", "}")
                continue
            elif nest == 0 and ch == ";":
                break
            j += 1
        head = text[i:j]
        body = j < end and text[j] == "{"
        fm, tm, cm = FN_ITEM_RE.match(head), TYPE_ITEM_RE.match(head), CONST_ITEM_RE.match(head)
        if fm:
            items.append((f"fn {fm.group(1)}", normalise_sig(head), body))
        elif tm or cm:
            sig = head if eq < 0 else text[i:eq]
            key = f"type {tm.group(1)}" if tm else f"const {cm.group(1)}"
            items.append((key, normalise_sig(sig), eq >= 0))
        else:
            items.append((normalise_sig(head), normalise_sig(head), body))
        i = _matching(text, j, "{", "}") if body else j + 1
    return header, items


def rust_sources(root: str) -> list[str]:
    out = []
    for d, dirs, files in os.walk(os.path.join(root, "src")):
        dirs.sort()
        out += [os.path.join(d, f) for f in sorted(files) if f.endswith(".rs")]
    return out


def check_contracts(root: str) -> tuple[list[str], dict[str, int]]:
    """Compare the traits written in the contract documents with src/."""
    errors: list[str] = []
    counts = {"contract traits": 0, "contract items": 0}
    sources = {p: strip_rust_comments(read_text_file(p) or "") for p in rust_sources(root)}
    for rel in CONTRACT_DOCS:
        if not os.path.exists(os.path.join(root, rel)):
            errors.append(f"{rel}: missing")
            continue
        for block in RUST_BLOCK_RE.findall(read(root, rel)):
            block = strip_rust_comments(block)
            for name in TRAIT_NAME_RE.findall(block):
                doc = trait_items(block, name)
                found = [(p, t) for p, t in ((p, trait_items(t, name)) for p, t in sources.items()) if t]
                if len(found) != 1:
                    where = ", ".join(os.path.relpath(p, root) for p, _ in found) or "nowhere"
                    errors.append(f"{rel}: `pub trait {name}` is defined {len(found)} times in src/ ({where})")
                    continue
                counts["contract traits"] += 1
                src_path, (src_header, src_items) = found[0]
                src_rel = os.path.relpath(src_path, root).replace(os.sep, "/")
                doc_header, doc_items = doc
                if doc_header != src_header:
                    errors.append(f"{rel}: trait header `{doc_header}` differs from {src_rel} `{src_header}`")
                def item(key: str, trait: str = name) -> str:
                    kind, _, ident = key.partition(" ")
                    return f"`{trait}::{ident}`" + ("" if kind == "fn" else f" ({kind})") if ident else f"`{key}`"

                src_by = {n: (s, d) for n, s, d in src_items}
                doc_by = {n: (s, d) for n, s, d in doc_items}
                for n in sorted(set(src_by) - set(doc_by)):
                    errors.append(f"{rel}: {item(n)} is in {src_rel} but not in the document")
                for n in sorted(set(doc_by) - set(src_by)):
                    errors.append(f"{rel}: {item(n)} is in the document but not in {src_rel}")
                for n in sorted(set(src_by) & set(doc_by)):
                    counts["contract items"] += 1
                    (ss, sd), (ds, dd) = src_by[n], doc_by[n]
                    if ss != ds:
                        errors.append(f"{rel}: {item(n)} is `{ds}` in the document but `{ss}` in {src_rel}")
                    if sd != dd:
                        has = lambda d: "has a default" + (" body" if n.startswith("fn ") else "") if d \
                            else "has no default" + (" body" if n.startswith("fn ") else "")  # noqa: E731
                        errors.append(f"{rel}: {item(n)} {has(dd)} in the document but {has(sd)} in {src_rel}")
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
    c_errors, c_counts = check_contracts(root)
    errors += c_errors
    counts.update(c_counts)
    print("compared: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if "--check" in sys.argv and errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
