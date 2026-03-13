# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Stable Diffusion image generation toolkit. Supports SD 1.5 and SDXL models loaded from `.safetensors`/`.ckpt` files in the `models/` directory, with optional LoRA weights from `loras/`.

## Setup & Running

```bash
# Install (uses uv/pip with pyproject.toml)
pip install -e .

# Generate an image
python -m ai_image_tools.generate --prompt "..." --model meinamix-v12

# SDXL (auto-detected by file size >4GB, or force with --sdxl)
python -m ai_image_tools.generate --prompt "..." --model pony-v6 --sdxl --clip-skip 2

# With LoRA
python -m ai_image_tools.generate --prompt "..." --model meinamix-v12 --lora my-lora

# Via installed entrypoint
generate --prompt "..." --model meinamix-v12
```

## Architecture

Single module: `src/ai_image_tools/generate.py`

- **Device detection**: MPS (Apple Silicon, float32) → CUDA (float16) → CPU
- **Model detection**: SDXL auto-detected if file size > 4GB
- **Schedulers**: Euler for SDXL/Pony, DPMSolver for SD 1.5
- **Output**: PNG + sidecar `.txt` with full generation params (prompt, seed, model, etc.)
- **Directory layout**: `models/` for checkpoints, `loras/` for LoRA weights, `outputs/` for results (timestamped)

## Key Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | 30 | |
| `--guidance` | 7.5 | |
| `--width/height` | 512x768 (SD1.5), 768x1024 (SDXL) | MPS struggles above 896x1152 |
| `--clip-skip` | None | Pony models need `--clip-skip 2` |
| `--seed` | random | Printed to stdout and saved in sidecar |
| `--face-ref PATH` | None | Reference face image — anchors identity via IP-Adapter FaceID Plus v2 |
| `--face-weight` | 0.8 | FaceID anchoring strength |
| `--style-ref PATH` | None | Reference style image — transfers hair/clothing/accessories via IP-Adapter Plus |
| `--style-weight` | 0.6 | Style transfer strength (lower = more prompt influence) |

## Reference Image Modes

Two mutually exclusive IP-Adapter modes (cannot combine):

**`--face-ref`** → IP-Adapter FaceID Plus v2 (`h94/IP-Adapter-FaceID`)
- Uses InsightFace `buffalo_l` to extract face identity embedding
- Captures: face structure, skin tone — NOT hair or accessories
- InsightFace always runs on CPU (ONNX, no MPS provider)
- Weights auto-downloaded and cached in `~/.cache/huggingface/hub/`

**`--style-ref`** → IP-Adapter Plus (`h94/IP-Adapter`)
- Encodes full image via CLIP ViT-H
- Captures: hair, outfit, accessories, collar, jewelry, colors, textures
- Passes PIL Image directly via `ip_adapter_image` kwarg
- Use this for appearance/style transfer from a selfie

## Notes

- No safety checker (disabled for SD 1.5; SDXL has none by default)
- MPS uses `enable_attention_slicing()` for memory; SDXL uses `enable_vae_slicing()`
- clip_skip for SDXL is passed via `gen_kwargs["clip_skip"]` — diffusers applies it to the second text encoder
