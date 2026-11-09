#!/bin/bash
# Fetch and unpack the data bundle into runs/ and data/ at the repository root, for Level 2 of REPRODUCING.md.
#   scripts/fetch_data.sh                download the release asset with the GitHub CLI
#   scripts/fetch_data.sh BUNDLE.tar.gz  use a copy you already have
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=swarm-surplus-data-v1.tar.gz
BUNDLE=${1:-}
if [ -z "$BUNDLE" ]; then
  gh release download data-v1 --repo ayseozdincer/swarm-surplus --pattern "$NAME" --dir . --clobber
  BUNDLE=$NAME
fi
expected=$(cut -d' ' -f1 scripts/data_bundle.sha256)
actual=$(shasum -a 256 "$BUNDLE" | cut -d' ' -f1)
[ "$expected" = "$actual" ] || { echo "checksum mismatch: expected $expected, got $actual" >&2; exit 1; }
tar -xzf "$BUNDLE"
echo "unpacked runs/ and data/"
