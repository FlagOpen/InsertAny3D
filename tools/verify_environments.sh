#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
THIRD_PARTY="$PROJECT_ROOT/third_party"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run_test() {
  local name=$1
  local repo=$2
  local python=$3
  local code=$4
  echo "正在验证：$name"
  (
    cd "$repo"
    CUDA_VISIBLE_DEVICES="$GPU" "$python" -c "$code"
  )
}

MAIN="$THIRD_PARTY/TRELLIS/.venv/bin/python"
HUNYUAN="$THIRD_PARTY/Hunyuan3D-2/.venv/bin/python"
GIM="$THIRD_PARTY/gim/.venv/bin/python"

run_test "TRELLIS 主环境" "$THIRD_PARTY/TRELLIS" "$MAIN" '
import torch, trellis, flash_attn, xformers, kaolin, nvdiffrast, diffoctreerast
import diff_gaussian_rasterization, diff_gaussian_rasterization_absdepth, simple_knn
assert torch.ones(1, device="cuda").item() == 1
print("ENV_OK TRELLIS", torch.__version__, torch.version.cuda)
'

run_test "SAGS 共用主环境" "$THIRD_PARTY/SAGS" "$MAIN" '
import torch, groundingdino, segment_anything
import diff_gaussian_rasterization, simple_knn
assert torch.ones(1, device="cuda").item() == 1
print("ENV_OK SAGS", torch.__version__, torch.version.cuda)
'

run_test "TRELLIS-old 共用主环境" "$THIRD_PARTY/TRELLIS-old" "$MAIN" '
import torch, trellis, lang_sam, sam2
import flash_attn, kaolin, nvdiffrast, diffoctreerast
assert torch.ones(1, device="cuda").item() == 1
print("ENV_OK TRELLIS-old", torch.__version__, torch.version.cuda)
'

run_test "Hunyuan 独立环境" "$THIRD_PARTY/Hunyuan3D-2" "$HUNYUAN" '
import torch, hy3dgen, custom_rasterizer, mesh_processor
assert torch.ones(1, device="cuda").item() == 1
print("ENV_OK Hunyuan3D-2", torch.__version__, torch.version.cuda)
'

run_test "GIM 独立环境" "$THIRD_PARTY/gim" "$GIM" '
import torch, cv2, albumentations, kornia, pytorch_lightning
from networks.roma.dino import Attention
assert torch.ones(1, device="cuda").item() == 1
print("ENV_OK GIM", torch.__version__, torch.version.cuda)
'

echo "INSERTANY3D_THREE_ENVIRONMENTS_READY"
