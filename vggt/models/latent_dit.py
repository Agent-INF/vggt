# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Latent Diffusion Transformer (DiT) for geometry generation.

This module implements a Diffusion Transformer for generating geometric
latent representations in the compressed space learned by the AutoEncoder.

Phase 2 of the 3D generation pipeline:
- DiT learns to generate valid geometric structures in the latent space
- Can be conditioned on image features (DINO latent_1) or class labels
- Outputs can be decoded via AutoEncoder and DPT Head for depth/point cloud
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vggt.utils.diffusion import timestep_embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    
    This block uses adaptive layer normalization to inject timestep and
    optional conditioning information into the transformer.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to hidden dim
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # AdaLN-Zero modulation: produces shift, scale, gate for both attn and mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input tensor [B, N, D]
            c: Conditioning tensor [B, D] (timestep + optional condition)
            mask: Optional attention mask
            
        Returns:
            Output tensor [B, N, D]
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning on external features.
    
    Used to inject LLM-generated condition tokens into the DiT.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention to context.
        
        Args:
            x: Input tensor [B, N, D]
            context: Context tensor [B, M, D] (e.g., LLM condition tokens)
            context_mask: Optional mask for context
            
        Returns:
            Output tensor [B, N, D]
        """
        # Cross-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attn(
            x_norm, context, context, key_padding_mask=context_mask
        )
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT for projecting to output.
    
    Args:
        hidden_dim: Hidden dimension
        output_dim: Output dimension
    """
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Final layer forward pass.
        
        Args:
            x: Input tensor [B, N, D]
            c: Conditioning tensor [B, D]
            
        Returns:
            Output tensor [B, N, output_dim]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class LatentDiT(nn.Module):
    """
    Latent Diffusion Transformer for 3D geometry generation.
    
    This model learns to denoise latent representations in the compressed
    space, enabling generation of geometric structures.
    
    Args:
        latent_dim: Dimension of the latent space (from AutoEncoder)
        hidden_dim: Hidden dimension for transformer
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to hidden dim
        num_frames: Number of frames to generate
        num_patches: Number of spatial patches per frame
        dropout: Dropout rate
        use_cross_attention: Whether to use cross-attention for conditioning
        condition_dim: Dimension of external condition (e.g., LLM tokens)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_frames: int = 16,
        num_patches: int = 256,  # After spatial downsampling
        dropout: float = 0.0,
        use_cross_attention: bool = False,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.use_cross_attention = use_cross_attention
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * num_patches, hidden_dim)
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Optional condition projection (for LLM outputs, DINO features, etc.)
        self.condition_proj = None
        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Optional cross-attention blocks (interleaved)
        self.cross_attn_blocks = None
        if use_cross_attention and condition_dim is not None:
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_dim, latent_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values."""
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize adaLN modulation to zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.
        
        Args:
            x: Noisy latent tensor [B, S, P, latent_dim] or [B, N, latent_dim]
                where S=frames, P=patches, N=S*P
            t: Timestep tensor [B]
            condition: Optional conditioning tensor [B, M, condition_dim]
            condition_mask: Optional mask for condition
            
        Returns:
            Predicted noise [B, S, P, latent_dim] or [B, N, latent_dim]
        """
        input_shape = x.shape
        
        # Flatten spatial dimensions if needed
        if len(x.shape) == 4:
            B, S, P, C = x.shape
            x = x.view(B, S * P, C)
        else:
            B, N, C = x.shape
            
        # Input projection
        x = self.input_proj(x)  # [B, N, hidden_dim]
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Timestep embedding
        t_emb = timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]
        
        # Add global condition if provided (for unconditional or simple conditioning)
        c = t_emb
        projected_context = None  # For cross-attention
        
        if condition is not None and self.condition_proj is not None:
            if len(condition.shape) == 2:
                # Global condition [B, condition_dim]
                cond_emb = self.condition_proj(condition)
                c = c + cond_emb
            elif len(condition.shape) == 3:
                # Sequence condition [B, M, condition_dim] - project once for cross-attention
                projected_context = self.condition_proj(condition)
        
        # Process through DiT blocks
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            
            # Optional cross-attention with sequence condition
            if self.cross_attn_blocks is not None and projected_context is not None:
                x = self.cross_attn_blocks[i](x, projected_context, condition_mask)
        
        # Final layer
        x = self.final_layer(x, c)  # [B, N, latent_dim]
        
        # Reshape to input shape
        if len(input_shape) == 4:
            x = x.view(input_shape)
            
        return x
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        
        Args:
            x: Noisy latent tensor
            t: Timestep tensor
            condition: Conditioning tensor
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Predicted noise with CFG applied
        """
        # Unconditional prediction
        uncond_pred = self.forward(x, t, condition=None)
        
        if condition is None or cfg_scale == 1.0:
            return uncond_pred
        
        # Conditional prediction
        cond_pred = self.forward(x, t, condition=condition)
        
        # CFG combination
        return uncond_pred + cfg_scale * (cond_pred - uncond_pred)


class ConditionedLatentDiT(nn.Module):
    """
    Latent DiT with conditioning from LLM outputs.
    
    This version is designed for Phase 3, where condition tokens from
    the LLM (Qwen3) guide the generation of multi-frame latent sequences.
    
    Args:
        latent_dim: Dimension of the latent space
        hidden_dim: Hidden dimension for transformer
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim
        num_frames: Maximum number of frames to generate
        num_patches: Number of patches per frame
        llm_hidden_dim: Dimension of LLM hidden states (e.g., 4096 for Qwen3)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_frames: int = 16,
        num_patches: int = 256,
        llm_hidden_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.num_patches = num_patches
        
        # Base DiT with cross-attention
        self.dit = LatentDiT(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_frames=num_frames,
            num_patches=num_patches,
            dropout=dropout,
            use_cross_attention=True,
            condition_dim=llm_hidden_dim,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        llm_condition_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with LLM conditioning.
        
        Args:
            x: Noisy latent tensor [B, S, P, latent_dim]
            t: Timestep tensor [B]
            llm_condition_tokens: LLM output tokens [B, T, llm_hidden_dim]
                where T is the number of condition tokens (one per frame)
            condition_mask: Optional mask for condition tokens
            
        Returns:
            Predicted noise [B, S, P, latent_dim]
        """
        return self.dit(
            x, t,
            condition=llm_condition_tokens,
            condition_mask=condition_mask,
        )


def build_latent_dit(
    latent_dim: int = 512,
    hidden_dim: int = 1024,
    depth: int = 12,
    num_frames: int = 16,
    spatial_size: int = 16,  # After autoencoder downsampling
    use_llm_conditioning: bool = False,
    llm_hidden_dim: int = 4096,
) -> nn.Module:
    """
    Build a LatentDiT configured for VGGT latent space.
    
    Args:
        latent_dim: Dimension of the compressed latent space
        hidden_dim: Hidden dimension for transformer
        depth: Number of DiT blocks
        num_frames: Number of frames to generate
        spatial_size: Spatial size after autoencoder downsampling
        use_llm_conditioning: Whether to use LLM conditioning
        llm_hidden_dim: Dimension of LLM hidden states
        
    Returns:
        dit: Configured DiT model
    """
    num_patches = spatial_size * spatial_size
    
    if use_llm_conditioning:
        return ConditionedLatentDiT(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=num_frames,
            num_patches=num_patches,
            llm_hidden_dim=llm_hidden_dim,
            dropout=0.0,
        )
    else:
        return LatentDiT(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=num_frames,
            num_patches=num_patches,
            dropout=0.0,
            use_cross_attention=False,
            condition_dim=None,
        )
