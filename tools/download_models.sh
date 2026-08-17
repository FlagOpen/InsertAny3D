#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
THIRD_PARTY="$PROJECT_ROOT/third_party"

verify_sha256() {
  local file=$1
  local expected=$2
  printf '%s  %s\n' "$expected" "$file" | sha256sum --check --status
}

download_http() {
  local url=$1
  local target=$2
  local sha256=$3
  if [[ -f "$target" ]] && verify_sha256 "$target" "$sha256"; then
    echo "已存在并通过校验: $target"
    return
  fi
  mkdir -p "$(dirname "$target")"
  rm -f "${target}.part"
  curl --fail --location --retry 3 --output "${target}.part" "$url"
  verify_sha256 "${target}.part" "$sha256" || {
    echo "下载文件校验失败: $target" >&2
    exit 1
  }
  mv "${target}.part" "$target"
}

download_gim() {
  local target="$THIRD_PARTY/gim/weights/gim_roma_100h.ckpt"
  local sha256=0568a39fa9154bc60ab67badf1a49d4da7860761f87eb2149dc5e9daa02e5a05
  if [[ -f "$target" ]] && verify_sha256 "$target" "$sha256"; then
    echo "已存在并通过校验: $target"
    return
  fi
  local python="$THIRD_PARTY/gim/.venv/bin/python"
  [[ -x "$python" ]] || {
    echo "请先安装 GIM 环境: bash tools/install_environments.sh gim" >&2
    exit 1
  }
  mkdir -p "$(dirname "$target")"
  rm -f "${target}.part"
  "$python" -m gdown --fuzzy \
    'https://drive.google.com/file/d/1OGNbJdw9zn5zHC4WNQ0IMqdGCS0HMUfe/view' \
    --output "${target}.part"
  verify_sha256 "${target}.part" "$sha256" || {
    echo "GIM 权重校验失败" >&2
    exit 1
  }
  mv "${target}.part" "$target"
}

download_huggingface() {
  local executable=$1
  local repo=$2
  local revision=$3
  local python="$(dirname "$executable")/python"
  [[ -x "$executable" ]] || {
    echo "缺少 $executable，请先安装对应环境" >&2
    exit 1
  }
  if "$python" -c \
    'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True)' \
    "$repo" "$revision" >/dev/null 2>&1; then
    echo "已存在: $repo@$revision"
    return
  fi
  "$executable" download "$repo" --revision "$revision"
}

[[ -d "$THIRD_PARTY/SAGS/.git" ]] || bash "$PROJECT_ROOT/tools/bootstrap_third_party.sh"

download_http \
  'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth' \
  "$THIRD_PARTY/SAGS/gaussiansplatting/dependencies/GroundingDINO/weights/groundingdino_swint_ogc.pth" \
  3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
download_http \
  'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' \
  "$THIRD_PARTY/SAGS/gaussiansplatting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth" \
  a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e
download_gim
download_huggingface \
  "$THIRD_PARTY/TRELLIS/.venv/bin/hf" \
  microsoft/TRELLIS-image-large \
  25e0d31ffbebe4b5a97464dd851910efc3002d96
download_huggingface \
  "$THIRD_PARTY/Hunyuan3D-2/.venv/bin/hf" \
  tencent/Hunyuan3D-2 \
  9cd649ba6913f7a852e3286bad86bfa9a2d83dcf

echo "InsertAny3D 主流程模型已下载并校验"
