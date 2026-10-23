#!/usr/bin/env bash
# Build and test the downstream crates that depend on alice-physics against
# this checkout, so a change here that breaks them fails here.
#
#   ALICE-SDF  depends on alice-physics from crates.io (`version = "1.1"`,
#              feature `physics`); the registry copy is replaced with this
#              checkout through `--config patch.crates-io.alice-physics.path`.
#   ALICE-LOL  depends on it by path (`../../ALICE-Physics`, feature
#              `physics`). Its workspace also needs ALICE-SDF, ALICE-Kinematics,
#              ALICE-LLM, ALICE-Zip and ALICE-Sign next to it, so all of them are
#              fetched into one directory with this checkout linked in as
#              ALICE-Physics.
#   ALICE-TRT  depends on it by path (`../ALICE-Physics`, no version; feature
#              `physics-solver`) and implements `gpu_bridge::GpuSolverBridge`.
#              Its manifest also needs ALICE-ML, and ALICE-DB with ALICE-Crypto
#              behind optional features (cargo reads every path dependency),
#              placed at main: none of them is required with a version.
#
# Which version of each repository is placed: each consumer gets a layout directory
# of its own ($DOWNSTREAM_ROOT/sdf, /lol, /trt) holding
#
#   the consumer          its main (ALICE-SDF, ALICE-LOL, ALICE-TRT)
#   ALICE-Physics         this checkout, linked in (the candidate, in every layout)
#   every other           the commit the consumer pins in its .github/alice-siblings,
#   repository the        followed through the siblings' own path dependencies
#   layout takes by path  (scripts/downstream_siblings.py, which fails on a path
#                         dependency that is not pinned, a missing or empty pin
#                         file, a malformed pin line, a placed repository that is
#                         not at its pin, or ALICE-LOL / ALICE-TRT taking no path
#                         dependency on alice-physics). ALICE-LOL: ALICE-SDF,
#                         ALICE-Zip, ALICE-Kinematics, ALICE-LLM, ALICE-Sign.
#                         ALICE-SDF takes none
#   ALICE-TRT's siblings  main (ALICE-ML, ALICE-SDF and what they take by path):
#                         ALICE-TRT has no .github/alice-siblings yet, and the log
#                         names each repository placed at main for that reason
#
# A sibling two consumers pin at different commits (ALICE-SDF: main in its own
# layout, ALICE-LOL's pin in the lol layout) is placed once per layout.
#
# Only the tests that reach alice-physics are run (the rest of each downstream
# suite does not depend on this crate and is tested in its own repository):
#
#   ALICE-SDF  lib tests of `physics_bridge::` and `sim_bridge::` (the two
#              modules behind the feature), tests/test_physics_bridge_determinism
#              and tests/test_sim_bridge_oracle, and the `sim_bridge` example build
#   ALICE-LOL  every integration test of alice-world-auditor with `physics` (the
#              crate is a planner over alice-physics worlds; its lib has no unit
#              tests), alice-lol tests/analytic_thermal
#              and the alice-lol lib tests of `law::` and
#              `stdlib::hardsurface::reinforcement` (the modules that call it)
#   ALICE-TRT  lib tests of `physics_bridge::` (the GpuSolverBridge
#              implementation) and `fix128::` (the GPU Fix128 kernels checked
#              against alice-physics) with `physics-solver`, one test at a time
#              as in ALICE-TRT's own GPU lanes. Of the 102 tests, 56 create a
#              `GpuDevice` and return early, still counted as passed, when
#              `GpuDevice::new` finds no adapter (wgpu, every backend: on Linux
#              Vulkan or GL). So before the tests a probe built against the
#              placed ALICE-TRT calls the same `GpuDevice::new` and prints the
#              adapter; with no adapter the step fails, because the GPU tests
#              would pass without running. CI provides the adapter with Mesa's
#              software Vulkan driver (lavapipe); locally it is the machine's
#              GPU. TRT_GPU_PROBE=0 skips the probe (the tests then prove only
#              that the trait implementation builds and the CPU tests pass)
#
# When this checkout is a major bump over the newest alice-physics on crates.io
# (2.2.0 published, 3.0.0 here), the downstream repositories cannot accept it yet:
# their requirements name the published major, and raising them in their own
# repositories needs the new version to be published first. In the fetched
# copies only, every requirement on alice-physics that accepts the published
# release is raised to this checkout's version (scripts/downstream_major_rewrite.py,
# which prints each edited line), so the downstream code is built and tested
# against the new major with nothing changed but the requirement. Any other
# commit (a minor or patch over the published release, or a version that is
# already published) leaves the manifests as fetched.
#
# Before building, every requirement on alice-physics in the placed ALICE-SDF,
# ALICE-LOL and ALICE-TRT manifests must accept this checkout's version
# (scripts/downstream_select_tag.py --check); a downstream that still requires
# an older major fails here with the requirement, and its steps are skipped.
# Each test binary of a step must run at least one test and ignore none (a
# `Running` section with 0 tests, or with a test marked #[ignore] in the selected
# set, fails even when the other binaries of the step ran some;
# scripts/downstream_test_counts.py), and the resolved alice-physics must be
# this checkout (a patch cargo cannot apply falls back to crates.io with only a
# warning, which would test the published version instead). ALICE-SDF's committed
# Cargo.lock names the published alice-physics, which cargo keeps over the patch
# on a minor or patch bump, so its lock is first moved to this checkout with
# `cargo update -p alice-physics` (scripts/downstream_resolve.sh, the fetched
# copy only).
#
#   scripts/downstream_check.sh               fetch, then build and test
#   scripts/downstream_check.sh --fetch-only  only fetch (CI restores its cache in between)
#   scripts/downstream_check.sh --no-fetch    use what DOWNSTREAM_ROOT already holds
#
# Environment:
#   DOWNSTREAM_ROOT   directory to fetch into (default: a new temporary directory)
#   DOWNSTREAM_URL    base URL of the repositories (default https://github.com/ext-sakamoro)
#   TRT_GPU_PROBE     1 (default): fail when ALICE-TRT finds no GPU adapter; 0:
#                     run its tests without one (the GPU tests return early)
#   DOWNSTREAM_TARGET_DIR
#                     build output, one subdirectory per downstream (default
#                     $DOWNSTREAM_ROOT/target); they are kept apart because
#                     alice-sdf and alice-trt build a cdylib, whose rlib has no
#                     hash in its file name, so a shared directory lets one
#                     workspace's build overwrite another's (E0463 on the next
#                     build)

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
# the engine (alice-physics-core) is the root package; the downstream crates depend on the
# facade (alice-physics), which re-exports it
FACADE_DIR="$PHYSICS_DIR/crates/alice-physics"
PHYSICS_PACKAGE_DIR="$FACADE_DIR"
ROOT=${DOWNSTREAM_ROOT:-$(mktemp -d)}
mkdir -p "$ROOT/logs"
ROOT=$(cd "$ROOT" && pwd -P)
URL=${DOWNSTREAM_URL:-https://github.com/ext-sakamoro}

# one layout directory per consumer: each direct consumer at its main, the candidate
# (this checkout) linked in as ALICE-Physics, and every repository the layout takes by
# path at the commit the consumer pins in its .github/alice-siblings
# (scripts/downstream_siblings.py). A sibling two consumers pin at different commits is
# placed once per layout, so no consumer is built against another's pin
LAYOUTS=(sdf lol trt)
declare_layout() {
    case $1 in
        sdf) echo ALICE-SDF ;;
        lol) echo ALICE-LOL ;;
        trt) echo ALICE-TRT ;;
    esac
}
# ALICE-LOL takes alice-physics by path, so a layout without it compared nothing;
# ALICE-SDF takes it from crates.io (patched below) and needs no path sibling.
# ALICE-TRT has no .github/alice-siblings yet: its siblings are placed at main, and
# the script names each one in the job log
layout_args() {
    case $1 in
        sdf) ;;
        lol) echo --require ALICE-Physics ;;
        trt) echo --require ALICE-Physics --unpinned-main ;;
    esac
}

fetch() {
    local layout=$1 name=$2 ref=$3 dir="$ROOT/$1/$2"
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$URL/$name.git"
    fi
    git -C "$dir" fetch -q --depth 1 origin "$ref"
    git -C "$dir" checkout -q --detach FETCH_HEAD
    echo "fetched $layout/$name $(git -C "$dir" rev-parse --short=12 HEAD) ($ref)"
}

siblings() {
    local layout=$1
    shift
    # shellcheck disable=SC2046
    python3 "$PHYSICS_DIR/scripts/downstream_siblings.py" --layout "$ROOT/$layout" \
        --consumer "$(declare_layout "$layout")" $(layout_args "$layout") "$@"
}

for layout in "${LAYOUTS[@]}"; do
    consumer=$(declare_layout "$layout")
    mkdir -p "$ROOT/$layout"
    link="$ROOT/$layout/ALICE-Physics"
    if [ -e "$link" ] || [ -L "$link" ]; then
        if [ "$(cd "$link" && pwd -P)" != "$PHYSICS_DIR" ]; then
            echo "error: $link exists and is not this checkout ($PHYSICS_DIR)" >&2
            exit 1
        fi
    else
        ln -s "$PHYSICS_DIR" "$link"
    fi
    if [ "$mode" != run ]; then
        fetch "$layout" "$consumer" main
        # siblings can take further siblings by path: place until nothing is missing
        for _ in 1 2 3 4 5 6 7 8; do
            todo=$(siblings "$layout")
            [ -n "$todo" ] || break
            while read -r name ref; do
                fetch "$layout" "$name" "$ref"
            done <<< "$todo"
        done
    elif [ ! -d "$ROOT/$layout/$consumer/.git" ]; then
        echo "error: --no-fetch but $ROOT/$layout/$consumer is missing" >&2
        exit 1
    fi
    # every needed sibling placed, each at the consumer's pin (or main, named in the log)
    siblings "$layout" --verify
    for d in "$ROOT/$layout"/*/; do
        name=$(basename "$d")
        [ "$name" = ALICE-Physics ] && continue
        echo "using $layout/$name $(git -C "$d" rev-parse --short=12 HEAD)"
    done
done

if [ "$mode" = fetch ]; then
    exit 0
fi

PATCH="patch.crates-io.alice-physics.path=\"$FACADE_DIR\""
failed=()
summary=()
# downstream crates whose steps ran; fewer than the 3 at the end fails (a crate
# skipped for any reason, or none compared at all)
compared=0

# a major bump over the published release: raise the requirements that accept the
# published release to this version in the fetched copies (no change otherwise)
for layout in "${LAYOUTS[@]}"; do
    trees=()
    for d in "$ROOT/$layout"/*/; do
        [ "$(basename "$d")" = ALICE-Physics ] || trees+=(--tree "${d%/}")
    done
    # a downstream path dependency written for the layout before the facade
    # (`path = "../ALICE-Physics"`, now the engine's directory) points at the facade
    # in the fetched copies, the edit each downstream repository makes itself
    echo "==> $layout: path dependencies on alice-physics that name this checkout's root"
    if ! python3 "$PHYSICS_DIR/scripts/downstream_path_rewrite.py" --provider "$ROOT/$layout/ALICE-Physics" "${trees[@]}"; then
        echo "error: the path rewrite of the downstream dependencies failed ($layout)" >&2
        exit 1
    fi
    echo "==> $layout: requirements on alice-physics against the published release"
    if ! python3 "$PHYSICS_DIR/scripts/downstream_major_rewrite.py" --provider "$FACADE_DIR" "${trees[@]}"; then
        echo "error: the major-bump rewrite of the downstream requirements failed ($layout)" >&2
        exit 1
    fi
done

# relock / check_resolved: scripts/downstream_resolve.sh (sourced, so its tests can
# run them against a small crate that takes the package from a git source)
# shellcheck source=downstream_resolve.sh
. "$PHYSICS_DIR/scripts/downstream_resolve.sh"

# run one cargo step, keep its log, count the tests of each binary it ran
run() {
    local label=$1 dir=$2
    shift 2
    local log="$ROOT/logs/$label.log" start=$SECONDS n
    echo "==> $label: $*"
    if ! (cd "$dir" && "$@") 2>&1 | tee "$log"; then
        failed+=("$label")
    fi
    echo "$label: tests per binary"
    if ! n=$(python3 "$PHYSICS_DIR/scripts/downstream_test_counts.py" "$log"); then
        failed+=("$label (a binary ran 0 tests)")
    fi
    summary+=("$label: ${n:-0} passed, $((SECONDS - start)) s")
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

SDF="$ROOT/sdf/ALICE-SDF"
LOL="$ROOT/lol/ALICE-LOL"
TRT="$ROOT/trt/ALICE-TRT"
TARGET=${DOWNSTREAM_TARGET_DIR:-$ROOT/target}
unset CARGO_TARGET_DIR

# every requirement on alice-physics in a placed downstream must accept this
# checkout's version; a downstream that does not is reported and skipped (cargo
# would only fail to resolve, or fall back to crates.io for the patched one)
requirements_ok() {
    local label=$1 dir=$2
    echo "==> $label: requirements on alice-physics"
    if python3 "$PHYSICS_DIR/scripts/downstream_select_tag.py" --check "$dir" --provider "$PHYSICS_DIR" --provider "$FACADE_DIR"; then
        return 0
    fi
    failed+=("$label (requires another alice-physics version, steps skipped)")
    summary+=("$label: skipped, its manifests do not accept alice-physics $(physics_version)")
    return 1
}

physics_version() {
    sed -n 's/^version *= *"\(.*\)"/\1/p' "$PHYSICS_DIR/Cargo.toml" | head -n 1
}

export CARGO_TARGET_DIR="$TARGET/sdf"
if requirements_ok sdf "$SDF"; then
    compared=$((compared + 1))
    relock sdf "$SDF" --config "$PATCH"
    check_resolved sdf "$SDF" --features physics --config "$PATCH"
    run sdf-lib "$SDF" cargo test --features physics --config "$PATCH" --lib -- physics_bridge:: sim_bridge::
    run sdf-tests "$SDF" cargo test --features physics --config "$PATCH" \
        --test test_physics_bridge_determinism --test test_sim_bridge_oracle
    build sdf-example "$SDF" cargo build --features physics --config "$PATCH" --example sim_bridge
fi

export CARGO_TARGET_DIR="$TARGET/lol"
if requirements_ok lol "$LOL"; then
    compared=$((compared + 1))
    check_resolved lol-world-auditor "$LOL" -p alice-world-auditor --features physics
    check_resolved lol "$LOL" -p alice-lol --features physics
    # its integration tests (`--test '*'`); the lib has no unit tests, and a
    # binary with 0 tests fails the step
    run lol-world-auditor "$LOL" cargo test -p alice-world-auditor --features physics --test '*'
    run lol-thermal "$LOL" cargo test -p alice-lol --features physics --test analytic_thermal
    run lol-lib "$LOL" cargo test -p alice-lol --features physics --lib -- law:: stdlib::hardsurface::reinforcement
fi

# trt_gpu_probe: scripts/downstream_trt_probe.sh (sourced, so its tests can run it
# against a stand-in cargo)
# shellcheck source=downstream_trt_probe.sh
. "$PHYSICS_DIR/scripts/downstream_trt_probe.sh"

export CARGO_TARGET_DIR="$TARGET/trt"
if requirements_ok trt "$TRT"; then
    compared=$((compared + 1))
    check_resolved trt "$TRT" --features physics-solver
    trt_gpu_probe
    run trt-lib "$TRT" cargo test --features physics-solver --lib -- physics_bridge:: fix128:: --test-threads=1
fi

if [ "$compared" -eq 0 ]; then
    failed+=("no downstream crate was built and tested")
elif [ "$compared" -ne 3 ]; then
    failed+=("only $compared of 3 downstream crates were built and tested")
fi

echo
echo "downstream ($ROOT): $compared of 3 compared"
for s in "${summary[@]}"; do echo "  $s"; done
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "downstream: all green"
