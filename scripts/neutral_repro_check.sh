#!/usr/bin/env bash
# Build the browser package without names (scripts/build_neutral_pkg.sh) twice from clean target
# directories and require the same bytes.
#
#   scripts/neutral_repro_check.sh
#
# Compared: the full content id of each build (the SHA-256 of its wasm, of which the file names
# carry the first 12 hex digits), the SHA-256 of every file of the package and of the token set,
# the SHA-256 of a zip of the package written with fixed names, order and times, and the SHA-256 of
# the name map. Exit 0 when every item is equal, 1 when one differs, 2 when nothing was compared.
# The name maps and the build directories are temporary and removed at the end; only hashes are
# printed.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

build() {
  local n=$1
  mkdir -p "$work/private$n" "$work/out$n"
  chmod 700 "$work/private$n"
  CARGO_TARGET_DIR="$work/target$n" bash scripts/build_neutral_pkg.sh "$work/out$n/pkg" "$work/private$n" \
    > "$work/build$n.log" 2>&1 || { tail -20 "$work/build$n.log" >&2; echo "error: build $n failed" >&2; exit 1; }
}

# one line per compared item: "<item> <sha256>"
digest() {
  local n=$1 dir="$work/out$n/pkg"
  python3 -I - "$dir" "$dir.tokens.json" "$work/private$n/name-map.json" <<'EOF'
import hashlib, io, sys, zipfile
from pathlib import Path
pkg, tokens, name_map = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
files = sorted(p for p in pkg.iterdir() if p.is_file())
wasm = [p for p in files if p.suffix == ".wasm"]
if len(wasm) != 1:
    sys.exit(f"error: {len(wasm)} wasm files in the package")
sha = lambda b: hashlib.sha256(b).hexdigest()
print("content-id", sha(wasm[0].read_bytes()))
for p in files:
    print(f"file:{p.name}", sha(p.read_bytes()))
print("tokens", sha(tokens.read_bytes()))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        info = zipfile.ZipInfo(p.name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, p.read_bytes())
print("zip", sha(buf.getvalue()))
print("name-map", sha(name_map.read_bytes()))
EOF
}

build 1
build 2
digest 1 > "$work/d1"
digest 2 > "$work/d2"
compared=$(wc -l < "$work/d1" | tr -d ' ')
if [ "$compared" = 0 ] || [ ! -s "$work/d2" ]; then
  echo "neutral_repro_check: compared nothing" >&2
  exit 2
fi
differ=0
while read -r item h1; do
  h2=$(awk -v k="$item" '$1==k{print $2}' "$work/d2")
  if [ "$h1" = "$h2" ]; then echo "$item: equal ${h1:0:16}"; else echo "$item: DIFFERENT"; differ=$((differ + 1)); fi
done < "$work/d1"
[ "$(wc -l < "$work/d2" | tr -d ' ')" = "$compared" ] || { echo "the two builds have different file sets"; differ=$((differ + 1)); }
echo "neutral repro check: $compared items compared, $differ different"
[ "$differ" = 0 ]
