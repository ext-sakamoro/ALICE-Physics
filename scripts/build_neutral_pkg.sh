#!/usr/bin/env bash
# Build the WebAssembly package for the browser with no name of the engine in it, named by its
# content hash (scripts/neutral_pkg.py), and check it (scripts/neutral_leak_check.py).
#
#   scripts/build_neutral_pkg.sh <out-dir> <private-dir>
#
# <out-dir> receives p-<h12>.wasm / .js / .d.ts / .manifest.json and nothing else; <out-dir>.tokens.json
# is the SHA-256 token set the leak check reads (copy both to the consumer). <private-dir> must be
# outside every git repository: it receives name-map.json (mode 600), the engine name -> neutral
# name map. The HMAC key comes from the macOS Keychain (service p-artifact-key, account k1); the
# build fails without it. Re-running on the same commit with the same key gives the same bytes.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:?usage: build_neutral_pkg.sh <out-dir> <private-dir>}
private=${2:?usage: build_neutral_pkg.sh <out-dir> <private-dir>}
mkdir -p "$private"
if git -C "$private" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: <private-dir> is inside a git work tree; put the name map outside every repository" >&2
  exit 2
fi

want=$(awk '/^\[\[package\]\]/{p=0} $0=="name = \"wasm-bindgen\""{p=1} p&&/^version = /{gsub(/"/,"",$3); print $3; exit}' Cargo.lock)
have=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}') || { echo "error: wasm-bindgen CLI not found" >&2; exit 2; }
[ "$have" = "$want" ] || { echo "error: wasm-bindgen CLI $have, Cargo.lock has $want" >&2; exit 2; }

# its own target directory: the path-remapping flags would otherwise rebuild the normal target
target_dir="${CARGO_TARGET_DIR:-target}/neutral"
repo=$(pwd)
remap="--remap-path-prefix=$repo=/s --remap-path-prefix=$HOME=/h"
[ -n "${CARGO_HOME:-}" ] && remap="--remap-path-prefix=$CARGO_HOME=/c $remap"
RUSTFLAGS="$remap" CARGO_TARGET_DIR="$target_dir" \
  cargo rustc --locked -p alice-physics --lib --target wasm32-unknown-unknown --release --features wasm --crate-type cdylib
wasm="$target_dir/wasm32-unknown-unknown/release/alice_physics.wasm"
[ -f "$wasm" ] || { echo "error: $wasm was not produced" >&2; exit 1; }

raw=$(mktemp -d)
trap 'rm -rf "$raw"' EXIT
wasm-bindgen --target web --out-name p --out-dir "$raw" "$wasm"

# the tokens the package must not carry: the engine's name, who built it and where, every
# source file name, every crate of the engine's family in the build (by name)
tokens=(--name alice --token "$(id -un)" --token "$(basename "$repo")" --token rustup)
while IFS= read -r f; do tokens+=(--token "$(basename "$f")"); done < <(git ls-files 'src/*.rs' 'crates/*.rs')
while IFS= read -r c; do tokens+=(--token "$c"); done < <(awk '/^name = /{gsub(/"/,"",$3); print $3}' Cargo.lock | grep '^alice' | sort -u)

rm -rf "$out"
python3 scripts/neutral_pkg.py --in "$raw" --out "$out" --map "$private/name-map.json" \
  --tokens "$out.tokens.json" --identifiers docs/PUBLIC_API_SNAPSHOT.txt "${tokens[@]}"
python3 scripts/neutral_leak_check.py --tokens "$out.tokens.json" "$out"
# the neutral package must behave as the package it was made from, bit for bit
node scripts/neutral_equivalence.mjs "$raw" "$out" "$private/name-map.json"
