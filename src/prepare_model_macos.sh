#!/usr/bin/env bash
# Called from a macOS arm64e build environment. The source checkpoint is a
# third-party mirror of NudeNet weights; pinned digest protects against drift.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname)" != Darwin ]]; then echo 'Requires macOS/Xcode to compile a Core ML model.' >&2; exit 1; fi
: "${MODEL_URL:=https://huggingface.co/vladmandic/nudenet/resolve/main/nudenet-v34-320n.pt?download=true}"
EXPECTED_SHA='1d25e219d536dcd6994651020d3c7cba642d13990e6eef934ed7a8ba650fb582'
if [[ ! -f 320n.pt ]]; then
  echo 'Downloading source model checkpoint...'
  curl --fail --location --retry 3 --output 320n.pt "$MODEL_URL"
fi
printf '%s  %s\n' "$EXPECTED_SHA" 320n.pt | shasum --algorithm 256 --check
python3 export_coreml.py
src=$(python3 - <<'PY'
from pathlib import Path
c=list(Path('.').glob('320n.mlpackage'))+list(Path('.').glob('320n.mlmodel'))
if len(c)!=1: raise SystemExit(f'Expected one converted 320n Core ML model, found: {c}')
print(c[0])
PY
)
rm -rf model_compile_out NudeNet.mlmodelc
mkdir -p model_compile_out
xcrun coremlc compile "$src" model_compile_out
compiled=$(find model_compile_out -maxdepth 1 -name '*.mlmodelc' -type d -print -quit)
[[ -n "$compiled" ]] || { echo 'Core ML compilation did not create an .mlmodelc bundle.' >&2; exit 1; }
cp -R "$compiled" NudeNet.mlmodelc
mkdir -p 'layout/Library/Application Support/PixelGuard'
rm -rf 'layout/Library/Application Support/PixelGuard/NudeNet.mlmodelc'
cp -R NudeNet.mlmodelc 'layout/Library/Application Support/PixelGuard/NudeNet.mlmodelc'
[[ -d 'layout/Library/Application Support/PixelGuard/NudeNet.mlmodelc' ]] || exit 1
printf 'Compiled Core ML model staged for Theos packaging.\n'
