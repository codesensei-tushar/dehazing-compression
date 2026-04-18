#!/bin/bash
# Download RESIDE ITS (train) and SOTS (test) on the cluster.
#
# Canonical source: https://sites.google.com/view/reside-dehaze-datasets
# Dropbox short-links (from the project brief):
#   ITS:  https://bit.ly/3iwHmh0
#   SOTS: https://bit.ly/2XZH498
#   OTS:  https://bit.ly/3k8a0Gf   (313K pairs — opt-in via --with-ots)
#
# bit.ly redirects to a Dropbox share page; wget --content-disposition follows
# the chain. If Dropbox serves an HTML preview instead of the zip, resolve the
# URL with `curl -Ls -o /dev/null -w '%{url_effective}' <bit.ly>`, then append
# `&dl=1` (or `?dl=1`) and retry with wget.
#
# Usage:
#   ./scripts/download_reside.sh                # ITS + SOTS
#   ./scripts/download_reside.sh --with-ots     # + OTS (huge)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$PROJECT_DIR/data/RESIDE"
mkdir -p "$DATA"

fetch() {
    local url="$1" out="$2"
    if [ -s "$out" ]; then
        echo "exists: $out (skip)"
        return
    fi
    echo "Downloading $url -> $out"
    wget --content-disposition -c -O "$out" "$url" || {
        echo "wget failed. Try:"
        echo "  1) curl -Ls -o /dev/null -w '%{url_effective}\\n' '$url'"
        echo "  2) append dl=1 to the resolved Dropbox URL and wget that."
        return 1
    }
}

fetch "https://bit.ly/3iwHmh0"  "$DATA/ITS.zip"
fetch "https://bit.ly/2XZH498"  "$DATA/SOTS.zip"

if [ "${1:-}" = "--with-ots" ]; then
    fetch "https://bit.ly/3k8a0Gf" "$DATA/OTS.zip"
fi

echo "Unzipping..."
cd "$DATA"
for z in ITS.zip SOTS.zip OTS.zip; do
    [ -f "$z" ] && unzip -q -n "$z" || true
done

echo
echo "Layout:"
find "$DATA" -maxdepth 2 -type d | head -30
