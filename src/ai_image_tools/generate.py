#!/usr/bin/env python3
"""
Local image generation using Stable Diffusion with custom models.

Supports:
- SD 1.5 models (.safetensors from CivitAI)
- SDXL models (Pony, Illustrious, etc.)
- LoRA style tuning
- IP-Adapter FaceID Plus v2 face reference
- IP-Adapter Plus style reference (hair, clothing, accessories, colors)
- MPS (Mac), CUDA (Linux/Windows), CPU fallback
- No content filters

Usage:
    # SD 1.5 (default)
    python -m ai_image_tools.generate --prompt "..." --model meinamix-v12

    # SDXL (auto-detected or explicit)
    python -m ai_image_tools.generate --prompt "..." --model pony-v6 --sdxl

    # With face reference
    python -m ai_image_tools.generate --prompt "..." --model meinamix-v12 --face-ref photo.jpg

    # With style reference (full appearance transfer via IP-Adapter Plus)
    python -m ai_image_tools.generate --prompt "..." --model meinamix-v12 --style-ref outfit.jpg
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image


# SDXL models are typically >4GB, SD 1.5 models are ~2GB
SDXL_SIZE_THRESHOLD = 4 * 1024 * 1024 * 1024  # 4GB

# IP-Adapter FaceID Plus v2 HuggingFace repo
FACEID_REPO = "h94/IP-Adapter-FaceID"

# Model-type-specific adapter and LoRA filenames
FACEID_ADAPTER_SD15 = "ip-adapter-faceid-plusv2_sd15.bin"
FACEID_LORA_SD15 = "ip-adapter-faceid-plusv2_sd15_lora.safetensors"
FACEID_ADAPTER_SDXL = "ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_LORA_SDXL = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"

# IP-Adapter Plus (style/appearance transfer via CLIP ViT-H, no InsightFace)
IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_PLUS_SD15 = "models/ip-adapter-plus_sd15.bin"
IPADAPTER_PLUS_SDXL = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"


def get_device():
    """Detect best available device."""
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # MPS needs float32
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


def is_sdxl_model(model_path: Path) -> bool:
    """Detect if model is SDXL based on file size."""
    size = model_path.stat().st_size
    return size > SDXL_SIZE_THRESHOLD


def find_model(model_name: str, models_dir: Path) -> Path:
    """Find model file by name (with or without extension)."""
    # Try exact match
    if (models_dir / model_name).exists():
        return models_dir / model_name

    # Try with common extensions
    for ext in [".safetensors", ".ckpt"]:
        path = models_dir / f"{model_name}{ext}"
        if path.exists():
            return path

    # List available models
    available = list(models_dir.glob("*.safetensors")) + list(models_dir.glob("*.ckpt"))
    available_names = [p.stem for p in available]

    raise FileNotFoundError(
        f"Model '{model_name}' not found in {models_dir}\n"
        f"Available models: {available_names}"
    )


def find_lora(lora_name: str, loras_dir: Path) -> Path:
    """Find LoRA file by name."""
    if (loras_dir / lora_name).exists():
        return loras_dir / lora_name

    for ext in [".safetensors", ".pt"]:
        path = loras_dir / f"{lora_name}{ext}"
        if path.exists():
            return path

    available = list(loras_dir.glob("*.safetensors")) + list(loras_dir.glob("*.pt"))
    available_names = [p.stem for p in available]

    raise FileNotFoundError(
        f"LoRA '{lora_name}' not found in {loras_dir}\n"
        f"Available LoRAs: {available_names}"
    )


def disable_safety_checker(pipe):
    """Disable the safety checker for NSFW content (SD 1.5 only)."""
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = dummy_checker
    return pipe


def extract_face_embedding(face_ref_path: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Extract face embedding from a reference image using InsightFace buffalo_l.

    InsightFace uses ONNX under the hood, so CPUExecutionProvider is used
    even on MPS systems — the face analysis always runs on CPU.
    """
    import cv2
    from insightface.app import FaceAnalysis

    print(f"Extracting face embedding from: {face_ref_path}")

    # InsightFace requires CPUExecutionProvider on MPS (no native MPS ONNX support)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(face_ref_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {face_ref_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError(
            f"No face detected in reference image: {face_ref_path}\n"
            "Make sure the image contains a clear, visible face."
        )

    face = faces[0]
    embedding = face.normed_embedding  # numpy array, shape (512,)

    # Convert to tensor with correct dtype and device
    face_embed = torch.from_numpy(embedding).unsqueeze(0).to(device=device, dtype=dtype)
    print(f"Face embedding extracted (shape: {face_embed.shape})")
    return face_embed


def load_faceid_weights(pipe, sdxl: bool, device: str, dtype: torch.dtype):
    """
    Download (if needed) and load IP-Adapter FaceID Plus v2 weights and paired LoRA.

    Uses hf_hub_download so files are cached in ~/.cache/huggingface/hub/.
    """
    from huggingface_hub import hf_hub_download

    adapter_filename = FACEID_ADAPTER_SDXL if sdxl else FACEID_ADAPTER_SD15
    lora_filename = FACEID_LORA_SDXL if sdxl else FACEID_LORA_SD15

    print(f"Downloading IP-Adapter FaceID Plus v2 weights ({adapter_filename})...")
    adapter_path = hf_hub_download(repo_id=FACEID_REPO, filename=adapter_filename)

    print(f"Downloading FaceID LoRA weights ({lora_filename})...")
    lora_path = hf_hub_download(repo_id=FACEID_REPO, filename=lora_filename)

    # Load the paired LoRA first (affects UNet attention layers)
    print("Loading FaceID LoRA into UNet...")
    pipe.load_lora_weights(lora_path)

    # Load IP-Adapter
    print("Loading IP-Adapter FaceID Plus v2...")
    pipe.load_ip_adapter(
        FACEID_REPO,
        subfolder=None,
        weight_name=adapter_filename,
    )

    return adapter_path, lora_path


def load_ipadapter_plus_weights(pipe, sdxl: bool):
    """
    Download (if needed) and load IP-Adapter Plus adapter weights only.
    Image encoding is handled separately via encode_style_image().
    """
    from huggingface_hub import hf_hub_download

    if sdxl:
        weight_name = IPADAPTER_PLUS_SDXL
        subfolder = "sdxl_models"
    else:
        weight_name = IPADAPTER_PLUS_SD15
        subfolder = "models"

    filename = weight_name.split("/")[-1]
    print(f"Downloading IP-Adapter Plus weights ({filename})...")
    hf_hub_download(repo_id=IPADAPTER_REPO, filename=weight_name)

    print("Loading IP-Adapter Plus adapter weights...")
    pipe.load_ip_adapter(
        IPADAPTER_REPO,
        subfolder=subfolder,
        weight_name=filename,
    )


def encode_style_image(style_ref_path: str, sdxl: bool, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Encode a style reference image using CLIP ViT-H from the IP-Adapter repo.

    IP-Adapter Plus uses hidden_states[-2] (penultimate layer), NOT the final
    projected output — this is what distinguishes Plus from basic IP-Adapter.
    Bypasses the pipeline's own CLIP to avoid SDXL's ViT-bigG (1664-dim)
    being used instead of ViT-H (1280-dim).
    """
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    # Both SD1.5 and SDXL _vit-h variants use the same ViT-H encoder at models/image_encoder.
    # sdxl_models/image_encoder contains ViT-bigG (1664-dim) used by the non-vit-h SDXL adapter.
    encoder_subfolder = "models/image_encoder"
    print(f"Loading CLIP ViT-H encoder from {encoder_subfolder}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IPADAPTER_REPO,
        subfolder=encoder_subfolder,
        torch_dtype=dtype,
    ).to(device)

    clip_processor = CLIPImageProcessor()
    pil_image = Image.open(style_ref_path).convert("RGB")
    pixel_values = clip_processor(images=pil_image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        # Plus variant needs penultimate hidden state, not final projection
        image_embeds = image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]
        # Negative embed is zeros (uncond)
        negative_image_embeds = torch.zeros_like(image_embeds)

    # MultiIPAdapterImageProjection expects [batch, num_images, seq, dim].
    # Add num_images=1 dimension before stacking [uncond, cond] on batch axis.
    # Without this, [2, 257, 1280] is misread as batch=2, num_images=257.
    image_embeds = image_embeds.unsqueeze(1)           # [1, 1, 257, 1280]
    negative_image_embeds = negative_image_embeds.unsqueeze(1)  # [1, 1, 257, 1280]
    embeds = torch.cat([negative_image_embeds, image_embeds])   # [2, 1, 257, 1280]
    print(f"Style embedding extracted (shape: {embeds.shape})")

    # Free encoder from device memory
    del image_encoder
    if device == "mps":
        torch.mps.empty_cache()

    return embeds


def generate(
    prompt: str,
    model_path: Path,
    output_path: Path,
    lora_path: Path = None,
    negative_prompt: str = None,
    width: int = None,
    height: int = None,
    steps: int = 30,
    guidance: float = 7.5,
    seed: int = None,
    sdxl: bool = None,
    clip_skip: int = None,
    face_ref: str = None,
    face_weight: float = 0.8,
    style_ref: str = None,
    style_weight: float = 0.6,
):
    """Generate image from prompt using specified model."""
    device, dtype = get_device()
    print(f"Using device: {device} (dtype: {dtype})")

    # Auto-detect SDXL if not specified
    if sdxl is None:
        sdxl = is_sdxl_model(model_path)

    model_type = "SDXL" if sdxl else "SD 1.5"
    print(f"Model type: {model_type}")

    # Set default dimensions based on model type
    # Note: MPS struggles with 896x1152, 768x1024 is more stable
    if width is None:
        width = 768 if sdxl else 512
    if height is None:
        height = 1024 if sdxl else 768

    # Load model
    print(f"Loading model: {model_path.name}")

    if sdxl:
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
        # Set clip_skip for Pony and similar models (they need clip_skip=2)
        if clip_skip:
            # In diffusers, we set this via the text encoder config
            # For SDXL, clip_skip affects the second text encoder
            print(f"Using clip_skip: {clip_skip}")
    else:
        pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=True,
        )
        # Disable safety checker for SD 1.5
        pipe = disable_safety_checker(pipe)

    # Use appropriate scheduler (Euler for SDXL/Pony, DPM for SD 1.5)
    if sdxl:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        print("Using scheduler: Euler (recommended for Pony/SDXL)")
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Using scheduler: DPMSolver")

    # Move to device
    pipe = pipe.to(device)

    # For SDXL on limited VRAM, enable more aggressive optimizations
    if sdxl:
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    # Load user LoRA if specified (loaded before FaceID LoRA to avoid conflicts)
    if lora_path:
        print(f"Loading LoRA: {lora_path.name}")
        pipe.load_lora_weights(str(lora_path))

    # Guard: face_ref and style_ref can't be combined (two IP-adapters simultaneously
    # requires multi-adapter loading, not supported in this version)
    if face_ref and style_ref:
        raise ValueError(
            "--face-ref and --style-ref cannot be used together. "
            "Combining two IP-Adapters simultaneously is not supported in this version."
        )

    # Face reference: extract embedding and load IP-Adapter FaceID + paired LoRA
    face_embeds = None
    if face_ref:
        face_embeds = extract_face_embedding(face_ref, device=device, dtype=dtype)
        load_faceid_weights(pipe, sdxl=sdxl, device=device, dtype=dtype)
        pipe.set_ip_adapter_scale(face_weight)
        print(f"Face reference enabled (weight: {face_weight})")

    # Style reference: full appearance transfer via IP-Adapter Plus (CLIP ViT-H)
    style_embeds = None
    if style_ref:
        load_ipadapter_plus_weights(pipe, sdxl=sdxl)
        style_embeds = encode_style_image(style_ref, sdxl=sdxl, device=device, dtype=dtype)
        pipe.set_ip_adapter_scale(style_weight)
        print(f"Style reference enabled (weight: {style_weight})")

    # Attention slicing is incompatible with IP-Adapter: enable_attention_slicing()
    # replaces all attention processors (including IPAdapterAttnProcessor2_0) with
    # SlicedAttnProcessor, breaking the IP-Adapter cross-attention path.
    # Only enable when not using IP-Adapter.
    if device == "mps" and face_ref is None and style_ref is None:
        pipe.enable_attention_slicing()

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Seed: {seed}")
    print(f"Generating: {width}x{height}, {steps} steps, guidance {guidance}")
    print(f"Prompt: {prompt[:100]}...")

    # Generate
    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "low quality, bad anatomy, worst quality",
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }

    # Add clip_skip for SDXL if specified
    if sdxl and clip_skip:
        gen_kwargs["clip_skip"] = clip_skip

    # Pass face embedding via ip_adapter_image_embeds
    if face_embeds is not None:
        gen_kwargs["ip_adapter_image_embeds"] = [face_embeds]

    # Pass style embeds (encoded manually with ViT-H to avoid SDXL's ViT-bigG mismatch)
    if style_embeds is not None:
        gen_kwargs["ip_adapter_image_embeds"] = [style_embeds]

    result = pipe(**gen_kwargs)

    image = result.images[0]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved: {output_path}")

    # Save prompt info
    prompt_path = output_path.with_suffix(".txt")
    with open(prompt_path, "w") as f:
        f.write(f"prompt: {prompt}\n")
        f.write(f"negative: {negative_prompt}\n")
        f.write(f"model: {model_path.name}\n")
        f.write(f"model_type: {model_type}\n")
        if lora_path:
            f.write(f"lora: {lora_path.name}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"guidance: {guidance}\n")
        f.write(f"size: {width}x{height}\n")
        if clip_skip:
            f.write(f"clip_skip: {clip_skip}\n")
        if face_ref:
            f.write(f"face_ref: {Path(face_ref).name}\n")
            f.write(f"face_weight: {face_weight}\n")
        if style_ref:
            f.write(f"style_ref: {Path(style_ref).name}\n")
            f.write(f"style_weight: {style_weight}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using local Stable Diffusion models"
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name (looks in ./models/ directory)"
    )
    parser.add_argument(
        "--lora", "-l",
        help="Optional LoRA name (looks in ./loras/ directory)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: ./outputs/[timestamp].png)"
    )
    parser.add_argument(
        "--negative", "-n",
        default="low quality, bad anatomy, worst quality, blurry",
        help="Negative prompt"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        help="Image width (default: 512 for SD1.5, 768 for SDXL)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        help="Image height (default: 768 for SD1.5, 1024 for SDXL)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=30,
        help="Inference steps (default: 30)"
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Force SDXL mode (auto-detected by file size otherwise)"
    )
    parser.add_argument(
        "--clip-skip",
        type=int,
        default=None,
        help="Clip skip value (Pony models need --clip-skip 2)"
    )
    parser.add_argument(
        "--face-ref",
        metavar="PATH",
        help="Path to reference face image for IP-Adapter FaceID Plus v2"
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.8,
        help="IP-Adapter face anchoring strength (default: 0.8)"
    )
    parser.add_argument(
        "--style-ref",
        metavar="PATH",
        help="Path to reference image for IP-Adapter Plus style/appearance transfer "
             "(hair, clothing, accessories, colors — cannot combine with --face-ref)"
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=0.6,
        help="IP-Adapter Plus style transfer strength (default: 0.6)"
    )

    args = parser.parse_args()

    # Find project root (where models/ and loras/ are)
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    loras_dir = project_root / "loras"
    outputs_dir = project_root / "outputs"

    # Find model
    model_path = find_model(args.model, models_dir)

    # Find LoRA if specified
    lora_path = None
    if args.lora:
        lora_path = find_lora(args.lora, loras_dir)

    # Validate face-ref path if provided
    if args.face_ref and not Path(args.face_ref).exists():
        raise FileNotFoundError(f"Face reference image not found: {args.face_ref}")

    # Validate style-ref path if provided
    if args.style_ref and not Path(args.style_ref).exists():
        raise FileNotFoundError(f"Style reference image not found: {args.style_ref}")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = outputs_dir / f"{timestamp}.png"

    # Generate
    generate(
        prompt=args.prompt,
        model_path=model_path,
        output_path=output_path,
        lora_path=lora_path,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        sdxl=args.sdxl if args.sdxl else None,
        clip_skip=args.clip_skip,
        face_ref=args.face_ref,
        face_weight=args.face_weight,
        style_ref=args.style_ref,
        style_weight=args.style_weight,
    )


if __name__ == "__main__":
    main()
