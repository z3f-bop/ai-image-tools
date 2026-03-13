# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-03-13

### Added
- `encode_style_image()` — manual ViT-H encoding that bypasses diffusers' broken auto-encode path

### Fixed
- Wrong CLIP encoder path: use `models/image_encoder` (ViT-H) not `sdxl_models/image_encoder` (ViT-bigG)
- Missing `num_images` dimension: embeds now shaped `[batch, num_images, seq, dim]` as required by `MultiIPAdapterImageProjection`
- `enable_attention_slicing()` no longer overwrites `IPAdapterAttnProcessor2_0` — skipped when IP-Adapter is active
- Style reference now passed as pre-encoded embeds via `ip_adapter_image_embeds` instead of raw PIL image

## [0.1.0] - 2026-03-13

### Added
- Initial image generation toolkit
