#!/usr/bin/env bash
# Build and test the downstream crates that depend on alice-physics against
# this checkout, so a change here that breaks them fails here.
#
#   ALICE-SDF  depends on alice-physics from crates.io (`version = "1.1"`,
#              feature `physics`); the registry copy is replaced with this
#              checkout through `--config patch.crates-io.alice-physics.path`.
#   ALICE-LOL  depends on it by path (`../../ALICE-Physics`, feature
#              `physics`). Its workspace also needs ALICE-SDF, ALICE-Kinematics,
#              ALICE-LLM and ALICE-Zip next to it, so all of them are fetched
#              into one directory with this checkout linked in as ALICE-Physics.
#
# Which version of each repository is placed:
#
#   ALICE-SDF, ALICE-LOL   main (the code that depends on alice-physics)
#   ALICE-LLM, ALICE-Zip   the newest release tag (pre-releases skipped) that
#                          satisfies every requirement on the crate in the
#                          ALICE-LOL manifests, and whose own manifests accept the
#                          alice-lol / alice-sdf / ... versions placed next to it
#                          (scripts/downstream_select_tag.py); with no such tag
#                          the check fails and lists the requirements and tags
#   ALICE-Kinematics       main for now (see the comment where it is fetched)
#
# Only the tests that reach alice-physics are run (the rest of each downstream
# suite does not depend on this crate and is tested in its own repository):
#
#   ALICE-SDF  lib tests of `physics_bridge::` and `sim_bridge::` (the two
#              modules behind the feature), tests/test_physics_bridge_determinism
#              and tests/test_sim_bridge_oracle, and the `sim_bridge` example build
#   ALICE-LOL  every test of alice-world-auditor with `physics` (the crate is a
#              planner over alice-physics worlds), alice-lol tests/analytic_thermal
#              and the alice-lol lib tests of `law::` and
#              `stdlib::hardsurface::reinforcement` (the modules that call it)
#
# Each step must run at least one test, and the resolved alice-physics must be
# this checkout (a patch cargo cannot apply falls back to crates.io with only a
# warning, which would test the published version instead).
#
#   scripts/downstream_check.sh               fetch, then build and test
#   scripts/downstream_check.sh --fetch-only  only fetch (CI restores its cache in between)
#   scripts/downstream_check.sh --no-fetch    use what DOWNSTREAM_ROOT already holds
#
# Environment:
#   DOWNSTREAM_ROOT   directory to fetch into (default: a new temporary directory)
#   DOWNSTREAM_URL    base URL of the repositories (default https://github.com/ext-sakamoro)
#   SDF_REF LOL_REF   branch, tag or commit of ALICE-SDF / ALICE-LOL (default main)
#   LLM_REF ZIP_REF   ref of ALICE-LLM / ALICE-Zip, instead of the selected tag
#   KINEMATICS_REF    ref of ALICE-Kinematics (default main; set it to the empty
#                     string to select a release tag as for the other two)
#   DOWNSTREAM_TARGET_DIR
#                     build output, one subdirectory per downstream (default
#                     $DOWNSTREAM_ROOT/target); the two are kept apart because
#                     alice-sdf builds a cdylib, whose rlib has no hash in its
#                     file name, so a shared directory lets one workspace's build
#                     overwrite the other's (E0463 on the next build)

set -euo pipefail

mode=all
case "${1:-}" in
    "") ;;
    --fetch-only) mode=fetch ;;
    --no-fetch) mode=run ;;
    *)
        echo "usage: $0 [--fetch-only | --no-fetch]" >&2
        exit 2
        ;;
esac

PHYSICS_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
ROOT=${DOWNSTREAM_ROOT:-$(mktemp -d)}
mkdir -p "$ROOT/logs"
ROOT=$(cd "$ROOT" && pwd -P)
URL=${DOWNSTREAM_URL:-https://github.com/ext-sakamoro}

fetch() {
    local name=$1 ref=$2 dir="$ROOT/$1"
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$URL/$name.git"
    fi
    git -C "$dir" fetch -q --depth 1 origin "$ref"
    git -C "$dir" checkout -q --detach FETCH_HEAD
    echo "fetched $name $(git -C "$dir" rev-parse --short=12 HEAD) ($ref)"
}

fetch_sibling() {
    local name=$1 crate=$2 override=$3 ref
    if [ -n "$override" ]; then
        echo "$crate: ${override} (given ref, no tag selection)"
        ref=$override
    else
        ref=$(python3 "$PHYSICS_DIR/scripts/downstream_select_tag.py" --url "$URL/$name.git" \
            --repo "$ROOT/$name" --crate "$crate" --workspace "$ROOT/ALICE-LOL" \
            --provider "$ROOT/ALICE-LOL" --provider "$ROOT/ALICE-SDF")
    fi
    fetch "$name" "$ref"
}

if [ "$mode" != run ]; then
    fetch ALICE-SDF "${SDF_REF:-main}"
    fetch ALICE-LOL "${LOL_REF:-main}"
    # the other repositories ALICE-LOL needs next to it: the newest release tag
    # that fits both ways (scripts/downstream_select_tag.py), unless a ref is given.
    # ALICE-Kinematics defaults to main: its only release tag (0.1.0) requires
    # alice-lol 0.3 while ALICE-LOL main is 0.4. Drop the default once a release
    # tag accepts the current alice-lol.
    fetch_sibling ALICE-Kinematics alice-kinematics "${KINEMATICS_REF-main}"
    fetch_sibling ALICE-LLM alice-llm "${LLM_REF:-}"
    fetch_sibling ALICE-Zip alice-zip "${ZIP_REF:-}"
else
    for name in ALICE-SDF ALICE-LOL ALICE-Kinematics ALICE-LLM ALICE-Zip; do
        if [ ! -d "$ROOT/$name/.git" ]; then
            echo "error: --no-fetch but $ROOT/$name is missing" >&2
            exit 1
        fi
        echo "using $name $(git -C "$ROOT/$name" rev-parse --short=12 HEAD)"
    done
fi

LINK="$ROOT/ALICE-Physics"
if [ -e "$LINK" ] || [ -L "$LINK" ]; then
    if [ "$(cd "$LINK" && pwd -P)" != "$PHYSICS_DIR" ]; then
        echo "error: $LINK exists and is not this checkout ($PHYSICS_DIR)" >&2
        exit 1
    fi
else
    ln -s "$PHYSICS_DIR" "$LINK"
fi

if [ "$mode" = fetch ]; then
    exit 0
fi

PATCH="patch.crates-io.alice-physics.path=\"$LINK\""
failed=()
summary=()

# the resolved alice-physics must be a path package at this checkout
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

# run one cargo step, keep its log, count the tests it ran
run() {
    local label=$1 dir=$2
    shift 2
    local log="$ROOT/logs/$label.log" start=$SECONDS n
    echo "==> $label: $*"
    if ! (cd "$dir" && "$@") 2>&1 | tee "$log"; then
        failed+=("$label")
    fi
    n=$({ grep -E '^test result: ' "$log" || true; } | sed -E 's/.* ([0-9]+) passed.*/\1/' | awk '{s += $1} END {print s + 0}' || true)
    if [ "$n" -eq 0 ]; then
        echo "error: $label ran 0 tests" >&2
        failed+=("$label (0 tests)")
    fi
    summary+=("$label: $n passed, $((SECONDS - start)) s")
}

# build-only step (no test count)
build() {
    local label=$1 dir=$2
    shift 2
    echo "==> $label: $*"
    if ! (cd "$dir" && "$@") 2>&1 | tee "$ROOT/logs/$label.log"; then
        failed+=("$label")
    fi
}

SDF="$ROOT/ALICE-SDF"
LOL="$ROOT/ALICE-LOL"
TARGET=${DOWNSTREAM_TARGET_DIR:-$ROOT/target}
unset CARGO_TARGET_DIR

export CARGO_TARGET_DIR="$TARGET/sdf"

check_resolved sdf "$SDF" --features physics --config "$PATCH"
run sdf-lib "$SDF" cargo test --features physics --config "$PATCH" --lib -- physics_bridge:: sim_bridge::
run sdf-tests "$SDF" cargo test --features physics --config "$PATCH" \
    --test test_physics_bridge_determinism --test test_sim_bridge_oracle
build sdf-example "$SDF" cargo build --features physics --config "$PATCH" --example sim_bridge

export CARGO_TARGET_DIR="$TARGET/lol"
check_resolved lol-world-auditor "$LOL" -p alice-world-auditor --features physics
check_resolved lol "$LOL" -p alice-lol --features physics
run lol-world-auditor "$LOL" cargo test -p alice-world-auditor --features physics
run lol-thermal "$LOL" cargo test -p alice-lol --features physics --test analytic_thermal
run lol-lib "$LOL" cargo test -p alice-lol --features physics --lib -- law:: stdlib::hardsurface::reinforcement

echo
echo "downstream ($ROOT):"
for s in "${summary[@]}"; do echo "  $s"; done
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "downstream: all green"
