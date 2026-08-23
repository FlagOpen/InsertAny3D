#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
THIRD_PARTY="$PROJECT_ROOT/third_party"
ENV_FILES="$PROJECT_ROOT/environment"
BUILD_ROOT="${INSERTANY3D_BUILD_ROOT:-$PROJECT_ROOT/.build/environment_sources}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CUDA_12_HOME="${CUDA_12_HOME:-/usr/local/cuda-12.4}"
CUDA_11_HOME="${CUDA_11_HOME:-/usr/local/cuda-11.8}"
TARGET="${1:-all}"

if [[ -z "$CONDA_BIN" ]]; then
  echo "未找到 conda，请通过 CONDA_BIN 指定其可执行文件。" >&2
  exit 1
fi

export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$BUILD_ROOT"

ensure_env() {
  local prefix=$1
  local python_version=$2
  if [[ ! -x "$prefix/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$prefix" "python=$python_version" pip
  fi
}

clone_exact() {
  local url=$1
  local commit=$2
  local path=$3
  if [[ ! -d "$path/.git" ]]; then
    git clone --recursive "$url" "$path"
  fi
  git -C "$path" checkout "$commit"
  git -C "$path" submodule update --init --recursive
}

install_main() {
  local env="$THIRD_PARTY/TRELLIS/.venv"
  ensure_env "$env" 3.11
  local py="$env/bin/python"

  export CUDA_HOME="$CUDA_12_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  "$py" -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
  "$py" -m pip install -r "$ENV_FILES/requirements-main.txt"
  "$py" -m pip install xformers==0.0.29.post1 --no-deps \
    --index-url https://download.pytorch.org/whl/cu121
  "$py" -m pip install flash-attn==2.7.4.post1 --no-deps --no-build-isolation
  "$py" -m pip install kaolin==0.18.0 --no-deps \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

  clone_exact https://github.com/NVlabs/nvdiffrast.git \
    253ac4fcea7de5f396371124af597e6cc957bfae "$BUILD_ROOT/nvdiffrast"
  clone_exact https://github.com/JeffreyXiang/diffoctreerast.git \
    b09c20b84ec3aace4729e6e18a613112320eca3a "$BUILD_ROOT/diffoctreerast"
  clone_exact https://github.com/autonomousvision/mip-splatting.git \
    dda02ab5ecf45d6edb8c540d9bb65c7e451345a9 "$BUILD_ROOT/mip-splatting"
  clone_exact https://github.com/g-truc/glm.git \
    33b4a621a697a305bc3a7610d290677b96beb181 "$BUILD_ROOT/glm"
  clone_exact https://github.com/facebookresearch/segment-anything.git \
    dca509fe793f601edb92606367a655c15ac00fdf "$BUILD_ROOT/segment-anything"
  clone_exact https://github.com/IDEA-Research/GroundingDINO.git \
    856dde20aee659246248e20734ef9ba5214f5e44 "$BUILD_ROOT/GroundingDINO"
  clone_exact https://github.com/facebookresearch/segment-anything-2.git \
    c2ec8e14a185632b0a5d8b161928ceb50197eddc "$BUILD_ROOT/segment-anything-2"
  clone_exact https://github.com/luca-medeiros/lang-segment-anything.git \
    d20193576bf7e9d179a823d5bc6e096b0eb3fa84 "$BUILD_ROOT/lang-segment-anything"
  clone_exact https://github.com/EasternJournalist/utils3d.git \
    9a4eb15e4021b67b12c460c7057d642626897ec8 "$BUILD_ROOT/utils3d"

  local glm_target="$THIRD_PARTY/TRELLIS/trellis/diff-gaussian-rasterization-absdepth/third_party/glm"
  mkdir -p "$glm_target"
  cp -a "$BUILD_ROOT/glm/glm" "$glm_target/"

  "$py" -m pip install "$BUILD_ROOT/nvdiffrast" --no-deps --no-build-isolation
  "$py" -m pip install "$BUILD_ROOT/diffoctreerast" --no-deps --no-build-isolation
  "$py" -m pip install "$BUILD_ROOT/utils3d" --no-deps --no-build-isolation
  "$py" -m pip install \
    "$BUILD_ROOT/mip-splatting/submodules/diff-gaussian-rasterization" \
    "$THIRD_PARTY/TRELLIS/trellis/diff-gaussian-rasterization-absdepth" \
    "$THIRD_PARTY/SAGS/gaussiansplatting/submodules/simple-knn" \
    --no-deps --no-build-isolation
  "$py" -m pip install -e "$BUILD_ROOT/segment-anything" --no-deps --no-build-isolation
  "$py" -m pip install -e "$BUILD_ROOT/GroundingDINO" --no-deps --no-build-isolation
  "$py" -m pip install -e "$BUILD_ROOT/segment-anything-2" --no-deps --no-build-isolation
  "$py" -m pip install -e "$BUILD_ROOT/lang-segment-anything" --no-deps --no-build-isolation

  for module in SAGS; do
    local link="$THIRD_PARTY/$module/.venv"
    if [[ -L "$link" ]]; then
      ln -sfn ../TRELLIS/.venv "$link"
    elif [[ ! -e "$link" ]]; then
      ln -s ../TRELLIS/.venv "$link"
    else
      echo "$link 已存在且不是符号链接，请确认旧环境不再需要后再替换。" >&2
    fi
  done

  echo "主环境安装完成：$env"
}

install_hunyuan() {
  local env="$THIRD_PARTY/Hunyuan3D-2/.venv"
  local repo="$THIRD_PARTY/Hunyuan3D-2"
  ensure_env "$env" 3.12
  local py="$env/bin/python"

  export CUDA_HOME="$CUDA_11_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  "$py" -m pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
  "$py" -m pip install -r "$ENV_FILES/requirements-hunyuan.txt"
  "$py" -m pip install -e "$repo" --no-deps --no-build-isolation
  "$py" -m pip install \
    "$repo/hy3dgen/texgen/custom_rasterizer" \
    "$repo/hy3dgen/texgen/differentiable_renderer" \
    --no-deps --no-build-isolation

  echo "Hunyuan 环境安装完成：$env"
}

install_gim() {
  local env="$THIRD_PARTY/gim/.venv"
  ensure_env "$env" 3.9
  local py="$env/bin/python"

  "$py" -m pip install \
    pip==24.0 setuptools==59.5.0 wheel==0.45.1
  "$py" -m pip install \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
  "$py" -m pip install -r "$ENV_FILES/requirements-gim.txt"
  "$CONDA_BIN" install -y -p "$env" --no-deps -c xformers \
    'xformers=0.0.22=py39_cu11.6.2_pyt1.12.1'

  echo "GIM 环境安装完成：$env"
}

case "$TARGET" in
  main) install_main ;;
  hunyuan) install_hunyuan ;;
  gim) install_gim ;;
  all)
    install_main
    install_hunyuan
    install_gim
    ;;
  *)
    echo "用法：$0 {main|hunyuan|gim|all}" >&2
    exit 2
    ;;
esac
