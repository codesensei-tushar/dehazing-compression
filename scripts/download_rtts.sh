#!/usr/bin/env bash
# Download RTTS (4,322 real hazy images) from the public RESIDE mirror.
#
# RESIDE bit.ly links are dead (see notes in CLAUDE.md). The most reliable
# mirror is the RESIDE-RTTS section of the official Google Drive:
#   https://sites.google.com/view/reside-dehaze-datasets
#
# Folder ID is `0B7PPbXPJRQp3OUVTUm9LREhRdjQ` (legacy) or one of the
# DeHamer/RESIDE-V0 mirrors. The cleanest 2023 mirror lives at:
#   GDrive ID 1Iqz_jXmzqz4eP6X1mRpFEi9b6gZh-h6X  (RTTS.zip, ~278MB)
#
# If GDrive throttles, the Kaggle alternative is:
#   https://www.kaggle.com/datasets/akaramachan/reside (includes RTTS).
#
# Output: data/RTTS/<images>.{png,jpg}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/teaching/miniconda3/envs/adu/bin/python
DATA_DIR="$ROOT/data/RTTS"
DATA_PHYSICAL="/DATA/datasets/dehazing/RTTS"

mkdir -p "$DATA_PHYSICAL"
if [ ! -L "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
    ln -sfn "$DATA_PHYSICAL" "$DATA_DIR"
fi

if [ -f "$DATA_PHYSICAL/RTTS.zip" ] && [ ! -s "$DATA_PHYSICAL/RTTS.zip" ]; then
    rm -f "$DATA_PHYSICAL/RTTS.zip"
fi

cd "$DATA_PHYSICAL"

# Primary attempt: gdown by file id
RTTS_GDRIVE_ID="${RTTS_GDRIVE_ID:-1Iqz_jXmzqz4eP6X1mRpFEi9b6gZh-h6X}"

if [ ! -f RTTS.zip ]; then
    echo "=== $(date) downloading RTTS via gdown id=$RTTS_GDRIVE_ID ==="
    $PY -m gdown --id "$RTTS_GDRIVE_ID" -O RTTS.zip || {
        echo "gdown failed. Either the ID is stale or quota was exceeded."
        echo "Alternatives:"
        echo "  1. Search Kaggle for 'RESIDE RTTS' and place RTTS.zip here:"
        echo "     $DATA_PHYSICAL/RTTS.zip"
        echo "  2. Find an alt GDrive ID and re-run with RTTS_GDRIVE_ID=<id>"
        exit 1
    }
fi

if [ ! -d images ] && [ ! -d JPEGImages ]; then
    echo "=== $(date) unzipping RTTS.zip ==="
    unzip -q -o RTTS.zip
fi

# Normalize to flat dir: data/RTTS/<images>
# RTTS distributions ship as either:
#   RTTS/JPEGImages/*.jpg   (Pascal-VOC style)
#   RTTS/images/*.png
#   RTTS/*.jpg              (flat)
SRC=""
for cand in "$DATA_PHYSICAL/JPEGImages" "$DATA_PHYSICAL/RTTS/JPEGImages" \
            "$DATA_PHYSICAL/images"     "$DATA_PHYSICAL/RTTS/images"; do
    if [ -d "$cand" ] && [ "$(ls -A "$cand" 2>/dev/null | wc -l)" -gt 0 ]; then
        SRC="$cand"; break
    fi
done

if [ -n "$SRC" ] && [ "$SRC" != "$DATA_PHYSICAL" ]; then
    echo "=== $(date) flattening $SRC → $DATA_PHYSICAL ==="
    find "$SRC" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPG' -o -name '*.PNG' \) -print0 \
        | xargs -0 -I{} mv -n {} "$DATA_PHYSICAL/"
fi

count=$(find "$DATA_PHYSICAL" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
echo "=== $(date) RTTS ready ==="
echo "directory : $DATA_PHYSICAL"
echo "symlink   : $DATA_DIR"
echo "images    : $count"
