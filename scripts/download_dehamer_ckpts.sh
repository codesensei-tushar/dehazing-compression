#!/bin/bash
# Download DeHamer pretrained checkpoints (indoor + outdoor + NH + dense)
# from the authors' Google Drive folder. Run ON THE CLUSTER.
#
# Layout produced:
#   experiments/teachers/dehamer/indoor/PSNR3663_ssim09881.pt
#   experiments/teachers/dehamer/outdoor/PSNR3518_SSIM09860.pt
#   experiments/teachers/dehamer/NH/PSNR2066_SSIM06844.pt
#   experiments/teachers/dehamer/dense/PSNR1662_SSIM05602.pt

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$PROJECT_DIR/experiments/teachers/dehamer"
mkdir -p "$OUT"

if ! command -v gdown >/dev/null; then
    echo "Installing gdown..."
    pip install --user -q gdown
fi

FOLDER_URL="https://drive.google.com/drive/folders/1YZnKreDfqbs_GHB76Ko4qtifpPWPbCwU"
echo "Downloading DeHamer ckpt folder to $OUT ..."
gdown --folder "$FOLDER_URL" -O "$OUT"

echo
echo "Contents of $OUT:"
find "$OUT" -maxdepth 2 -name "*.pt" -exec ls -lh {} \;
