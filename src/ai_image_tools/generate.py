#!/usr/bin/env python3
"""
Local image generation using Stable Diffusion with custom models.

Supports:
- Custom .safetensors models from CivitAI
- LoRA style tuning
- MPS (Mac), CUDA (Linux/Windows), CPU fallback
- No content filters

Usage:
    python -m ai_image_tools.generate --prompt "anime girl with pink hair" --model anime-v5
    python -m ai_image_tools.generate --prompt "..." --model anime-v5 --lora style-lora
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


def get_device():
    """Detect best available device."""
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # MPS needs float32
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


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
    """Disable the safety checker for NSFW content."""
    def dummy_checker(images, **kwargs):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_checker
    return pipe


def generate(
    prompt: str,
    model_path: Path,
    output_path: Path,
    lora_path: Path = None,
    negative_prompt: str = None,
    width: int = 512,
    height: int = 768,
    steps: int = 30,
    guidance: float = 7.5,
    seed: int = None,
):
    """Generate image from prompt using specified model."""
    device, dtype = get_device()
    print(f"Using device: {device} (dtype: {dtype})")

    # Load model
    print(f"Loading model: {model_path.name}")
    pipe = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Disable safety checker
    pipe = disable_safety_checker(pipe)

    # Move to device
    pipe = pipe.to(device)

    # Enable memory optimization
    if device == "mps":
        pipe.enable_attention_slicing()

    # Load LoRA if specified
    if lora_path:
        print(f"Loading LoRA: {lora_path.name}")
        pipe.load_lora_weights(str(lora_path))

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
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "low quality, bad anatomy, worst quality",
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

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
        if lora_path:
            f.write(f"lora: {lora_path.name}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"guidance: {guidance}\n")
        f.write(f"size: {width}x{height}\n")

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
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=768,
        help="Image height (default: 768)"
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
    )


if __name__ == "__main__":
    main()
