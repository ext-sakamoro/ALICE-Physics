# shellcheck shell=bash
# Sourced by downstream_check.sh (and by test_downstream_resolve.py).
#
# relock moves a fetched copy's Cargo.lock to this checkout. ALICE-SDF takes
# alice-physics from crates.io through the path patch (ALICE-LOL and ALICE-TRT
# depend on it by path and re-resolve on their own). On a minor or patch over the
# published release its requirement stays as fetched, so its committed Cargo.lock
# still names the published alice-physics, and cargo keeps the locked version and
# leaves the patch unused; `cargo update -p alice-physics` with the same patch
# moves the lock to the patched package. Only the fetched copy's Cargo.lock
# changes. A failed update is reported as "<label> (lock)".
#
# check_resolved then requires the resolved alice-physics to be a path package at
# this checkout's facade (PHYSICS_PACKAGE_DIR, default PHYSICS_DIR): a patch still
# unused after the update (or one cargo cannot apply, which falls back to crates.io
# with only a warning) is reported as "<label> (resolve)".
#
# Needs from the caller: PHYSICS_DIR (and PHYSICS_PACKAGE_DIR when the package is not
# at the root) and the array failed (PYTHON, default
# python3, runs the directory comparison).
relock() {
    local label=$1 dir=$2
    shift 2
    echo "==> $label: cargo update -p alice-physics (the fetched copy's lock follows this checkout)"
    if ! (cd "$dir" && cargo update -p alice-physics "$@"); then
        failed+=("$label (lock)")
    fi
}

# same_dir A B: true when A and B name the same directory. cargo prints the path
# in its own form (on Windows a backslash path, possibly with 8.3 short names such
# as RUNNER~1), so the two are compared as directories, not as strings: Python's
# os.path.samefile (file identity), after cygpath -m turns an MSYS path (/c/...)
# into one Windows Python reads
same_dir() {
    local a=$1 b=$2
    if command -v cygpath >/dev/null 2>&1; then
        a=$(cygpath -m "$a")
        b=$(cygpath -m "$b")
    fi
    "${PYTHON:-python3}" -c 'import os, sys
a, b = sys.argv[1:3]
sys.exit(0 if os.path.isdir(a) and os.path.isdir(b) and os.path.samefile(a, b) else 1)' "$a" "$b"
}

check_resolved() {
    local label=$1 dir=$2
    shift 2
    local line path
    if ! line=$(cd "$dir" && cargo tree -q -i alice-physics -e normal --depth 0 "$@" 2>&1); then
        echo "error: $label: cargo tree -i alice-physics failed: $line" >&2
        failed+=("$label (resolve)")
        return
    fi
    # `alice-physics vX.Y.Z (<source>)`: a path package names its directory; a
    # registry package has no parentheses and a git one a URL, neither a directory
    # (an empty or non-directory path is refused by same_dir)
    path=$(printf '%s\n' "$line" | tr -d '\r' | sed -n '1s/^alice-physics v[^ ]* (\(.*\))$/\1/p')
    if same_dir "$path" "${PHYSICS_PACKAGE_DIR:-$PHYSICS_DIR}"; then
        echo "$label: $line"
    else
        echo "error: $label: alice-physics does not resolve to this checkout: $line" >&2
        failed+=("$label (resolve)")
    fi
}
