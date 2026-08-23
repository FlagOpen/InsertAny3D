#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
THIRD_PARTY="$PROJECT_ROOT/third_party"
SOURCE_ROOT="$PROJECT_ROOT/code/third_party"

clone_and_prepare() {
  local name=$1
  local url=$2
  local commit=$3
  local patch_file="$SOURCE_ROOT/patches/$name.patch"
  local overlay_file="$SOURCE_ROOT/overlays/$name.tar"
  local target="$THIRD_PARTY/$name"

  mkdir -p "$THIRD_PARTY"
  if [[ ! -e "$target/.git" ]]; then
    git clone --recursive "$url" "$target"
  fi
  git -C "$target" fetch --all --tags
  git -C "$target" checkout "$commit"
  git -C "$target" submodule update --init --recursive

  if [[ -s "$patch_file" ]]; then
    if git -C "$target" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
      echo "$name: patch already applied"
    elif git -C "$target" apply --check "$patch_file" >/dev/null 2>&1; then
      git -C "$target" apply --binary "$patch_file"
    elif [[ "$name" == "SAGS" ]] &&
      (cd "$target" && patch --dry-run --batch --reverse -p1 -F3 < "$patch_file" >/dev/null 2>&1); then
      echo "$name: patch already applied with fuzzy context"
    elif [[ "$name" == "SAGS" ]] &&
      (cd "$target" && patch --dry-run --batch --forward -p1 -F3 < "$patch_file" >/dev/null 2>&1); then
      (cd "$target" && patch --batch --forward -p1 -F3 < "$patch_file")
    else
      echo "$name: patch does not apply cleanly" >&2
      exit 1
    fi
  fi
  if [[ -f "$overlay_file" ]]; then
    tar -xf "$overlay_file" -C "$target"
  fi
}

clone_and_prepare Hunyuan3D-2 \
  https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git \
  b173994017b1ab9559792fbdfa6194952e2ae2e0
clone_and_prepare MVInpainter \
  https://github.com/ewrfcas/MVInpainter.git \
  323d7f6ce3f73b0f263eb7f07dc48aefa6f27f34
clone_and_prepare TRELLIS \
  https://github.com/MrHandsomeljn/TRELLIS \
  1c4ab02e359f991d949cc527b81f065f2f266b92
clone_and_prepare TRELLIS-old \
  https://github.com/microsoft/TRELLIS.git \
  eb83038919f6e1feb63accf3a97a377a608c497d

git -C "$PROJECT_ROOT" submodule update --init third_party/SAGS third_party/gim
ln -sfn ../TRELLIS/.venv "$THIRD_PARTY/SAGS/.venv"
ln -sfn ../TRELLIS/.venv "$THIRD_PARTY/TRELLIS-old/.venv"
echo "InsertAny3D third-party sources are ready"
