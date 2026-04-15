# Handy CUDA GPU Build (Ubuntu 22.04+)

This fork replaces Vulkan/Whisper with **ONNX Runtime CUDA EP** for GPU-accelerated inference of GigaAM, Parakeet, and other ORT-based models on NVIDIA GPUs.

## Quick Install (prebuilt)

```bash
cd dist-cuda
sudo ./install.sh
```

This installs the `.deb` package and ONNX Runtime 1.24.1 GPU libraries to `/usr/local/lib/`.

**Requirements:** NVIDIA GPU, CUDA Toolkit 12.x, Ubuntu 22.04+.

## Build from Source

### Prerequisites

```bash
# Rust & Bun
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -fsSL https://bun.sh/install | bash

# System dependencies
sudo apt install -y libgtk-3-dev libwebkit2gtk-4.1-dev libclang-dev \
    libappindicator3-dev librsvg2-dev patchelf

# CUDA Toolkit (if not installed)
# https://developer.nvidia.com/cuda-downloads

# ONNX Runtime 1.24.1 GPU
cd /tmp
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-gpu-1.24.1.tgz
tar xzf onnxruntime-linux-x64-gpu-1.24.1.tgz
sudo cp onnxruntime-linux-x64-gpu-1.24.1/lib/libonnxruntime*.so* /usr/local/lib/
sudo ln -sf /usr/local/lib/libonnxruntime.so.1.24.1 /usr/local/lib/libonnxruntime.so.1
sudo ln -sf /usr/local/lib/libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so
sudo ldconfig

# VAD model
mkdir -p src-tauri/resources/models
curl -o src-tauri/resources/models/silero_vad_v4.onnx https://blob.handy.computer/silero_vad_v4.onnx
```

### Build

```bash
bun install

export ORT_LIB_LOCATION=/usr/local/lib
export ORT_PREFER_DYNAMIC_LINK=1
export PATH="$HOME/.bun/bin:/usr/local/cuda/bin:$PATH"

cargo tauri build --bundles deb
```

Output: `src-tauri/target/release/bundle/deb/Handy_*.deb`

### Install

```bash
sudo dpkg -i src-tauri/target/release/bundle/deb/Handy_*.deb
```

## What Changed vs Upstream

| File | Change |
|------|--------|
| `src-tauri/Cargo.toml` | `whisper-vulkan` → `ort-cuda` for Linux |
| `src-tauri/tauri.conf.json` | `bun run build` → `bun --bun run build` |

## Troubleshooting

**`libonnxruntime.so.1: not found`** — install ONNX Runtime libraries (see Prerequisites).

**Model hangs at loading** — version mismatch. `ort-sys v2` requires ONNX Runtime **1.24.x**, not 1.20.x.

**`VERS_X.Y.Z not found`** — rebuild after installing the correct ONNX Runtime version.
