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
# this checkout: a patch still unused after the update (or one cargo cannot
# apply, which falls back to crates.io with only a warning) is reported as
# "<label> (resolve)".
#
# Needs from the caller: PHYSICS_DIR, LINK and the array failed.
relock() {
    local label=$1 dir=$2
    shift 2
    echo "==> $label: cargo update -p alice-physics (the fetched copy's lock follows this checkout)"
    if ! (cd "$dir" && cargo update -p alice-physics "$@"); then
        failed+=("$label (lock)")
    fi
}

check_resolved() {
    local label=$1 dir=$2
    shift 2
    local line
    if ! line=$(cd "$dir" && cargo tree -q -i alice-physics -e normal --depth 0 "$@" 2>&1); then
        echo "error: $label: cargo tree -i alice-physics failed: $line" >&2
        failed+=("$label (resolve)")
        return
    fi
    case "$line" in
        *"($LINK)"* | *"($PHYSICS_DIR)"*) echo "$label: $line" ;;
        *)
            echo "error: $label: alice-physics does not resolve to this checkout: $line" >&2
            failed+=("$label (resolve)")
            ;;
    esac
}
