# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM-DiT Integration for unified 3D generation and understanding.

This module implements the integration between an LLM (e.g., Qwen3) and the
Latent DiT for multimodal 3D generation. It includes:
- Visual Token Resampler: Aligns DINO visual features to LLM text token space
- LLM Wrapper: Generates condition tokens for DiT
- Full Pipeline: End-to-end generation from image + text to 3D

Phase 3 of the 3D generation pipeline.
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class VisualResampler(nn.Module):
    """
    Resampler to align visual features (DINO latent_1) to LLM token space.
    
    Uses cross-attention with learnable queries to compress and align
    visual features to the text embedding space.
    
    Args:
        visual_dim: Dimension of visual features from DINO (default: 1024)
        llm_dim: Dimension of LLM text embeddings (default: 4096)
        num_queries: Number of learnable query tokens (default: 64)
        num_layers: Number of cross-attention layers (default: 4)
        num_heads: Number of attention heads (default: 16)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        llm_dim: int = 4096,
        num_queries: int = 64,
        num_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.zeros(1, num_queries, llm_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # Project visual features to LLM dimension
        self.visual_proj = nn.Linear(visual_dim, llm_dim)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            ResamplerBlock(
                dim=llm_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.norm = nn.LayerNorm(llm_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Resample visual features to LLM token space.
        
        Args:
            visual_features: DINO features [B, N_img, P, visual_dim]
                where N_img is number of input images, P is num patches
            visual_mask: Optional mask for visual features
            
        Returns:
            resampled_tokens: Visual tokens aligned to LLM space [B, num_queries, llm_dim]
        """
        B = visual_features.shape[0]
        
        # Flatten image and patch dimensions
        if len(visual_features.shape) == 4:
            B, N_img, P, C = visual_features.shape
            visual_features = visual_features.view(B, N_img * P, C)
        
        # Project visual features
        visual_features = self.visual_proj(visual_features)  # [B, N, llm_dim]
        
        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, num_queries, llm_dim]
        
        # Apply cross-attention layers
        x = queries
        for layer in self.layers:
            x = layer(x, visual_features, visual_mask)
        
        # Output normalization
        x = self.norm(x)
        
        return x


class ResamplerBlock(nn.Module):
    """
    A single block of the visual resampler with cross-attention and MLP.
    
    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor [B, N_q, D]
            context: Key-value tensor (visual features) [B, N_kv, D]
            context_mask: Optional mask for context
            
        Returns:
            Output tensor [B, N_q, D]
        """
        # Cross-attention to visual features
        x_norm = self.norm1(x)
        context_norm = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            x_norm, context_norm, context_norm,
            key_padding_mask=context_mask
        )
        x = x + attn_out
        
        # Self-attention
        x_norm = self.norm2(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self_attn_out
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class ConditionTokenGenerator(nn.Module):
    """
    Generates condition tokens for DiT from LLM hidden states.
    
    Takes the LLM output and projects it to frame-specific condition tokens
    that guide the DiT generation process.
    
    Args:
        llm_dim: Dimension of LLM hidden states (default: 4096)
        condition_dim: Dimension of condition tokens for DiT (default: 1024)
        num_frames: Number of output frames (default: 16)
        use_temporal_encoding: Whether to add temporal position encoding
    """
    
    def __init__(
        self,
        llm_dim: int = 4096,
        condition_dim: int = 1024,
        num_frames: int = 16,
        use_temporal_encoding: bool = True,
    ):
        super().__init__()
        
        self.llm_dim = llm_dim
        self.condition_dim = condition_dim
        self.num_frames = num_frames
        self.use_temporal_encoding = use_temporal_encoding
        
        # Project LLM output to condition tokens
        self.proj = nn.Linear(llm_dim, condition_dim * num_frames)
        
        # Optional temporal encoding
        if use_temporal_encoding:
            self.temporal_embed = nn.Parameter(
                torch.zeros(1, num_frames, condition_dim)
            )
            nn.init.normal_(self.temporal_embed, std=0.02)
        
        # Refinement layers
        self.refine = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        
    def forward(
        self,
        llm_output: torch.Tensor,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate condition tokens from LLM output.
        
        Args:
            llm_output: LLM hidden state [B, llm_dim] or [B, seq_len, llm_dim]
            num_frames: Number of frames to generate (if different from default)
            
        Returns:
            condition_tokens: Frame-specific condition tokens [B, T, condition_dim]
        """
        if num_frames is None:
            num_frames = self.num_frames
            
        # Handle sequence input by taking the last token or pooling
        if len(llm_output.shape) == 3:
            # Use last token (or could use mean pooling)
            llm_output = llm_output[:, -1, :]  # [B, llm_dim]
        
        B = llm_output.shape[0]
        
        # Project to all frame conditions
        conditions = self.proj(llm_output)  # [B, condition_dim * num_frames]
        conditions = conditions.view(B, num_frames, self.condition_dim)
        
        # Add temporal encoding
        if self.use_temporal_encoding:
            if num_frames != self.num_frames:
                # Interpolate temporal embeddings
                temporal_embed = F.interpolate(
                    self.temporal_embed.transpose(1, 2),
                    size=num_frames,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                temporal_embed = self.temporal_embed
            conditions = conditions + temporal_embed
        
        # Refine
        conditions = self.refine(conditions)
        
        return conditions


class LLMDiTWrapper(nn.Module):
    """
    Full wrapper integrating LLM with DiT for 3D generation.
    
    This is the main module for Phase 3, combining:
    1. Frozen DINO for visual feature extraction (latent_1)
    2. Visual Resampler for alignment to LLM space
    3. LLM (Qwen3) for understanding and condition generation
    4. DiT for latent space generation
    
    Note: The actual LLM module should be passed in or loaded separately
    due to its large size. This wrapper handles the interface.
    
    Args:
        visual_dim: Dimension of DINO features (default: 1024)
        llm_dim: Dimension of LLM hidden states (default: 4096)
        latent_dim: Dimension of latent space (default: 512)
        dit_hidden_dim: Hidden dimension for DiT (default: 1024)
        num_visual_queries: Number of visual resampler queries (default: 64)
        num_frames: Number of output frames (default: 16)
        num_patches: Number of patches per frame (default: 256)
        dit_depth: Number of DiT blocks (default: 12)
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        llm_dim: int = 4096,
        latent_dim: int = 512,
        dit_hidden_dim: int = 1024,
        num_visual_queries: int = 64,
        num_frames: int = 16,
        num_patches: int = 256,
        dit_depth: int = 12,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.num_patches = num_patches
        
        # Visual Resampler
        self.visual_resampler = VisualResampler(
            visual_dim=visual_dim,
            llm_dim=llm_dim,
            num_queries=num_visual_queries,
            num_layers=4,
            num_heads=16,
        )
        
        # Condition Token Generator (processes LLM output for DiT)
        self.condition_generator = ConditionTokenGenerator(
            llm_dim=llm_dim,
            condition_dim=dit_hidden_dim,
            num_frames=num_frames,
        )
        
        # Import DiT here to avoid circular imports
        from vggt.models.latent_dit import LatentDiT
        
        # Latent DiT
        self.dit = LatentDiT(
            latent_dim=latent_dim,
            hidden_dim=dit_hidden_dim,
            depth=dit_depth,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=num_frames,
            num_patches=num_patches,
            use_cross_attention=True,
            condition_dim=dit_hidden_dim,
        )
        
    def prepare_visual_tokens(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare visual tokens for LLM input.
        
        Args:
            visual_features: DINO features [B, N_img, P, visual_dim]
            
        Returns:
            visual_tokens: Tokens for LLM [B, num_queries, llm_dim]
        """
        return self.visual_resampler(visual_features)
    
    def generate_condition_tokens(
        self,
        llm_hidden_states: torch.Tensor,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate condition tokens for DiT from LLM output.
        
        Args:
            llm_hidden_states: LLM output hidden states
            num_frames: Number of frames to generate
            
        Returns:
            condition_tokens: Frame-specific conditions for DiT
        """
        return self.condition_generator(llm_hidden_states, num_frames)
    
    def forward_dit(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through DiT for denoising.
        
        Args:
            noisy_latent: Noisy latent [B, S, P, latent_dim]
            timestep: Diffusion timestep [B]
            condition_tokens: Condition tokens from LLM [B, T, condition_dim]
            
        Returns:
            Predicted noise or denoised latent
        """
        return self.dit(noisy_latent, timestep, condition=condition_tokens)


class Unified3DModel(nn.Module):
    """
    Complete unified model for 3D generation and understanding.
    
    Integrates all components:
    - Frozen VGGT (DINO + Aggregator + DPT Head)
    - Latent AutoEncoder
    - LLM-DiT Wrapper
    
    This is the full end-to-end model for Phase 3.
    
    Args:
        vggt_model: Frozen VGGT model (optional, can be loaded separately)
        autoencoder: Trained LatentAutoEncoder
        llm_dit_wrapper: LLMDiTWrapper instance
        diffusion: GaussianDiffusion instance
    """
    
    def __init__(
        self,
        vggt_model: Optional[nn.Module] = None,
        autoencoder: Optional[nn.Module] = None,
        llm_dit_wrapper: Optional[LLMDiTWrapper] = None,
        diffusion: Optional[Any] = None,
    ):
        super().__init__()
        
        self.vggt = vggt_model
        self.autoencoder = autoencoder
        self.llm_dit_wrapper = llm_dit_wrapper
        self.diffusion = diffusion
        
        # Freeze VGGT if provided
        if self.vggt is not None:
            for param in self.vggt.parameters():
                param.requires_grad = False
    
    def extract_dino_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract DINO features (latent_1) from images using frozen VGGT.
        
        Args:
            images: Input images [B, N_img, 3, H, W]
            
        Returns:
            dino_features: DINO features [B, N_img, P, 1024]
        """
        if self.vggt is None:
            raise ValueError("VGGT model not loaded")
        
        with torch.no_grad():
            B, N_img, C, H, W = images.shape
            
            # Reshape for batch processing
            images_flat = images.view(B * N_img, C, H, W)
            
            # Get patch tokens from DINO
            patch_tokens = self.vggt.aggregator.patch_embed(images_flat)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            
            # Reshape back
            P = patch_tokens.shape[1]
            patch_tokens = patch_tokens.view(B, N_img, P, -1)
            
        return patch_tokens
    
    def extract_latent2_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Extract full latent_2 features from images using frozen VGGT.
        
        Args:
            images: Input images [B, S, 3, H, W]
            
        Returns:
            aggregated_tokens_list: List of aggregated token tensors
            patch_start_idx: Index where patch tokens start
        """
        if self.vggt is None:
            raise ValueError("VGGT model not loaded")
        
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
            
        return aggregated_tokens_list, patch_start_idx
    
    def encode_to_latent(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        Encode latent_2 features to compressed latent z.
        
        Args:
            aggregated_tokens_list: Aggregated tokens from VGGT
            patch_start_idx: Index where patch tokens start
            
        Returns:
            z: Compressed latent [B, S, P', latent_dim]
        """
        if self.autoencoder is None:
            raise ValueError("AutoEncoder not loaded")
        
        layer_features = self.autoencoder.extract_layers_from_aggregated(
            aggregated_tokens_list, patch_start_idx
        )
        return self.autoencoder.encode(layer_features)
    
    def decode_from_latent(
        self,
        z: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Decode compressed latent z back to latent_2 features.
        
        Args:
            z: Compressed latent [B, S, P', latent_dim]
            
        Returns:
            layer_features: Reconstructed 4-layer features
        """
        if self.autoencoder is None:
            raise ValueError("AutoEncoder not loaded")
        
        return self.autoencoder.decode(z)
    
    def generate_latent(
        self,
        visual_features: torch.Tensor,
        llm_hidden_states: torch.Tensor,
        num_frames: int = 16,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate latent representations using DiT with LLM conditioning.
        
        Args:
            visual_features: DINO features from input images
            llm_hidden_states: Hidden states from LLM
            num_frames: Number of frames to generate
            num_steps: Number of diffusion sampling steps
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            generated_z: Generated latent [B, num_frames, P', latent_dim]
        """
        if self.llm_dit_wrapper is None or self.diffusion is None:
            raise ValueError("LLM-DiT wrapper or diffusion not loaded")
        
        B = visual_features.shape[0]
        
        # Generate condition tokens
        condition_tokens = self.llm_dit_wrapper.generate_condition_tokens(
            llm_hidden_states, num_frames
        )
        
        # Sample from diffusion with condition tokens
        shape = (B, num_frames, self.llm_dit_wrapper.num_patches, 
                 self.llm_dit_wrapper.latent_dim)
        
        generated_z = self.diffusion.ddim_sample_loop(
            self.llm_dit_wrapper.dit, 
            shape,
            condition=condition_tokens,
            num_steps=num_steps,
            progress=True,
        )
        
        return generated_z
    
    def generate_depth_from_latent(
        self,
        z: torch.Tensor,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate depth maps from latent representations.
        
        Args:
            z: Latent representations [B, S, P', latent_dim]
            images: Reference images for DPT Head [B, S, 3, H, W]
            
        Returns:
            depth: Predicted depth maps [B, S, H, W, 1]
            depth_conf: Depth confidence [B, S, H, W]
        """
        if self.vggt is None or self.autoencoder is None:
            raise ValueError("VGGT or AutoEncoder not loaded")
        
        # Decode to latent_2 features
        layer_features = self.decode_from_latent(z)
        
        # Reconstruct full aggregated_tokens_list format
        # This is a simplified version - in practice, you may need to handle
        # the full 24-layer structure
        B, S, P, C = layer_features[0].shape
        
        # Get patch_start_idx safely with default
        patch_start_idx = getattr(self.vggt.aggregator, 'patch_start_idx', 5)
        
        # Create dummy camera/register tokens
        dummy_special = torch.zeros(B, S, patch_start_idx, C, device=z.device)
        
        aggregated_tokens_list = []
        layer_idx_mapping = {4: 0, 11: 1, 17: 2, 23: 3}
        
        for i in range(24):
            if i in layer_idx_mapping:
                feat = layer_features[layer_idx_mapping[i]]
                feat_with_special = torch.cat([dummy_special, feat], dim=2)
            else:
                # Use interpolated features for non-key layers
                feat_with_special = torch.zeros(B, S, patch_start_idx + P, C, device=z.device)
            aggregated_tokens_list.append(feat_with_special)
        
        # Use DPT Head to generate depth
        with torch.no_grad():
            depth, depth_conf = self.vggt.depth_head(
                aggregated_tokens_list, images, patch_start_idx
            )
        
        return depth, depth_conf


def build_llm_dit_wrapper(
    visual_dim: int = 1024,
    llm_dim: int = 4096,
    latent_dim: int = 512,
    num_frames: int = 16,
    spatial_size: int = 16,
) -> LLMDiTWrapper:
    """
    Build an LLMDiTWrapper with default configurations.
    
    Args:
        visual_dim: Dimension of DINO features
        llm_dim: Dimension of LLM hidden states
        latent_dim: Dimension of latent space
        num_frames: Number of output frames
        spatial_size: Spatial size after autoencoder downsampling
        
    Returns:
        wrapper: Configured LLMDiTWrapper instance
    """
    return LLMDiTWrapper(
        visual_dim=visual_dim,
        llm_dim=llm_dim,
        latent_dim=latent_dim,
        dit_hidden_dim=1024,
        num_visual_queries=64,
        num_frames=num_frames,
        num_patches=spatial_size * spatial_size,
        dit_depth=12,
    )
