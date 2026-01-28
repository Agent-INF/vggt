# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference utilities for the unified 3D generation pipeline.

This module provides easy-to-use inference functions for:
1. Phase 1: AutoEncoder encoding/decoding
2. Phase 2: DiT generation (unconditional)
3. Phase 3: LLM-conditioned generation

Example usage:
    from vggt.inference.pipeline import UnifiedInferencePipeline
    
    pipeline = UnifiedInferencePipeline.from_pretrained(
        vggt_path="path/to/vggt",
        autoencoder_path="path/to/autoencoder",
        dit_path="path/to/dit",
        llm_path="Qwen/Qwen3-4B",
    )
    
    # Generate from single image + text
    result = pipeline.generate(
        image=pil_image,
        prompt="Generate a 360 degree view of this object",
        num_frames=16,
    )
"""

from .pipeline import (
    UnifiedInferencePipeline,
    GenerationConfig,
    GenerationOutput,
    visualize_depth_sequence,
    create_point_cloud_from_depth,
)

__all__ = [
    "UnifiedInferencePipeline",
    "GenerationConfig",
    "GenerationOutput",
    "visualize_depth_sequence",
    "create_point_cloud_from_depth",
]
