# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Latent AutoEncoder for compressing VGGT's latent_2 features.

This module implements an encoder-decoder architecture to compress the
4-layer 2048D latent_2 features from VGGT's Alternating-Attention Transformer
into a compact latent space suitable for diffusion-based generation.

Phase 1 of the 3D generation pipeline:
- Encoder: Fuses and downsamples the 4-layer features into a compact latent z
- Decoder: Reconstructs the latent_2 features from z
- Training uses both feature reconstruction loss and geometric consistency loss
  (via frozen DPT Head)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers.block import Block


class LatentEncoder(nn.Module):
    """
    Encoder that compresses VGGT's 4-layer latent_2 features into a compact latent space.
    
    The encoder takes 4 layers of 2048D features (concatenated frame + global attention outputs)
    and produces a compact latent representation z.
    
    Args:
        input_dim: Dimension of each layer's features (default: 2048)
        num_layers: Number of input layers to fuse (default: 4)
        hidden_dim: Hidden dimension for processing (default: 1024)
        latent_dim: Dimension of the output latent space (default: 512)
        num_blocks: Number of transformer blocks for processing (default: 4)
        num_heads: Number of attention heads (default: 16)
        spatial_downsample: Factor to downsample spatial dimensions (default: 4)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        num_layers: int = 4,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 16,
        spatial_downsample: int = 4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.spatial_downsample = spatial_downsample
        
        # Layer fusion: project each layer and combine
        self.layer_projs = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Combine fused layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Spatial downsampling using strided convolution on the token sequence
        # This reduces the number of tokens by spatial_downsample^2
        self.spatial_down = nn.Conv1d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=spatial_downsample,
            stride=spatial_downsample,
            padding=0
        )
        
        # Transformer blocks for processing
        self.blocks = nn.ModuleList([
            Block(dim=hidden_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(num_blocks)
        ])
        
        # Final projection to latent space
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(
        self,
        layer_features: List[torch.Tensor],
        return_pre_spatial: bool = False,
    ) -> torch.Tensor:
        """
        Encode latent_2 features into compact latent z.
        
        Args:
            layer_features: List of 4 tensors, each with shape [B, S, P, C]
                where B=batch, S=sequence (frames), P=patches, C=2048
            return_pre_spatial: If True, also return features before spatial downsampling
                
        Returns:
            z: Latent representation with shape [B, S, P', latent_dim]
                where P' = P // spatial_downsample
        """
        B, S, P, C = layer_features[0].shape
        
        # Project each layer
        projected = []
        for i, (feat, proj) in enumerate(zip(layer_features, self.layer_projs)):
            # Reshape for projection
            feat_flat = feat.view(B * S, P, C)
            projected.append(proj(feat_flat))
        
        # Concatenate and fuse
        fused = torch.cat(projected, dim=-1)  # [B*S, P, hidden_dim * num_layers]
        fused = self.fusion(fused)  # [B*S, P, hidden_dim]
        
        if return_pre_spatial:
            pre_spatial = fused.view(B, S, P, self.hidden_dim)
        
        # Spatial downsampling
        # Reshape for conv1d: [B*S, hidden_dim, P]
        fused = fused.transpose(1, 2)
        fused = self.spatial_down(fused)  # [B*S, hidden_dim, P']
        fused = fused.transpose(1, 2)  # [B*S, P', hidden_dim]
        
        P_down = fused.shape[1]
        
        # Process through transformer blocks
        for block in self.blocks:
            fused = block(fused)
        
        # Project to latent space
        z = self.to_latent(fused)  # [B*S, P', latent_dim]
        z = z.view(B, S, P_down, self.latent_dim)
        
        if return_pre_spatial:
            return z, pre_spatial
        return z


class LatentDecoder(nn.Module):
    """
    Decoder that reconstructs VGGT's 4-layer latent_2 features from compact latent z.
    
    Args:
        latent_dim: Dimension of the input latent space (default: 512)
        output_dim: Dimension of each output layer's features (default: 2048)
        num_layers: Number of output layers to generate (default: 4)
        hidden_dim: Hidden dimension for processing (default: 1024)
        num_blocks: Number of transformer blocks for processing (default: 4)
        num_heads: Number of attention heads (default: 16)
        spatial_upsample: Factor to upsample spatial dimensions (default: 4)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_dim: int = 2048,
        num_layers: int = 4,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        num_heads: int = 16,
        spatial_upsample: int = 4,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.spatial_upsample = spatial_upsample
        
        # Project from latent space
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Transformer blocks for processing
        self.blocks = nn.ModuleList([
            Block(dim=hidden_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(num_blocks)
        ])
        
        # Spatial upsampling using transposed convolution
        self.spatial_up = nn.ConvTranspose1d(
            hidden_dim,
            hidden_dim,
            kernel_size=spatial_upsample,
            stride=spatial_upsample,
            padding=0
        )
        
        # Layer-specific output heads
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            ) for _ in range(num_layers)
        ])
        
    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Decode latent z back to 4-layer latent_2 features.
        
        Args:
            z: Latent representation with shape [B, S, P', latent_dim]
                
        Returns:
            layer_features: List of 4 tensors, each with shape [B, S, P, output_dim]
        """
        B, S, P_down, _ = z.shape
        
        # Project from latent space
        x = z.view(B * S, P_down, self.latent_dim)
        x = self.from_latent(x)  # [B*S, P', hidden_dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Spatial upsampling
        x = x.transpose(1, 2)  # [B*S, hidden_dim, P']
        x = self.spatial_up(x)  # [B*S, hidden_dim, P]
        x = x.transpose(1, 2)  # [B*S, P, hidden_dim]
        
        P = x.shape[1]
        
        # Generate each layer's output
        layer_features = []
        for head in self.layer_heads:
            feat = head(x)  # [B*S, P, output_dim]
            feat = feat.view(B, S, P, self.output_dim)
            layer_features.append(feat)
        
        return layer_features


class LatentAutoEncoder(nn.Module):
    """
    Full AutoEncoder for VGGT latent_2 compression.
    
    Combines encoder and decoder with optional geometric consistency loss
    computation using a frozen DPT Head.
    
    Args:
        input_dim: Dimension of each layer's features (default: 2048)
        num_layers: Number of layers to encode/decode (default: 4)
        hidden_dim: Hidden dimension for processing (default: 1024)
        latent_dim: Dimension of the latent space (default: 512)
        num_encoder_blocks: Number of transformer blocks in encoder (default: 4)
        num_decoder_blocks: Number of transformer blocks in decoder (default: 4)
        num_heads: Number of attention heads (default: 16)
        spatial_downsample: Factor for spatial downsampling (default: 4)
        intermediate_layer_idx: Indices of layers to use (default: [4, 11, 17, 23])
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        num_layers: int = 4,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_encoder_blocks: int = 4,
        num_decoder_blocks: int = 4,
        num_heads: int = 16,
        spatial_downsample: int = 4,
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
    ):
        super().__init__()
        
        self.intermediate_layer_idx = intermediate_layer_idx
        self.num_layers = num_layers
        
        self.encoder = LatentEncoder(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=num_encoder_blocks,
            num_heads=num_heads,
            spatial_downsample=spatial_downsample,
        )
        
        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_blocks=num_decoder_blocks,
            num_heads=num_heads,
            spatial_upsample=spatial_downsample,
        )
        
    def encode(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """Encode latent_2 features to compact latent z."""
        return self.encoder(layer_features)
    
    def decode(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Decode compact latent z back to latent_2 features."""
        return self.decoder(z)
    
    def forward(
        self,
        layer_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: encode and decode.
        
        Args:
            layer_features: List of 4 tensors from VGGT's aggregated_tokens_list
                
        Returns:
            z: The latent representation
            reconstructed: List of 4 reconstructed feature tensors
        """
        z = self.encode(layer_features)
        reconstructed = self.decode(z)
        return z, reconstructed
    
    def extract_layers_from_aggregated(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> List[torch.Tensor]:
        """
        Extract the relevant layers from VGGT's aggregated_tokens_list.
        
        Args:
            aggregated_tokens_list: Full list of aggregated tokens from VGGT
            patch_start_idx: Index where patch tokens start
                
        Returns:
            layer_features: List of 4 feature tensors for encoding
        """
        layer_features = []
        for idx in self.intermediate_layer_idx:
            # Extract patch tokens only (skip camera and register tokens)
            feat = aggregated_tokens_list[idx][:, :, patch_start_idx:]
            layer_features.append(feat)
        return layer_features
    
    def compute_reconstruction_loss(
        self,
        original: List[torch.Tensor],
        reconstructed: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature reconstruction loss (MSE).
        
        Args:
            original: List of original feature tensors
            reconstructed: List of reconstructed feature tensors
                
        Returns:
            loss: MSE reconstruction loss
        """
        total_loss = 0.0
        for orig, recon in zip(original, reconstructed):
            total_loss = total_loss + F.mse_loss(recon, orig)
        return total_loss / len(original)


class AutoEncoderLoss(nn.Module):
    """
    Loss module for training the LatentAutoEncoder.
    
    Combines:
    1. Feature Reconstruction Loss: MSE between original and reconstructed features
    2. Geometric Consistency Loss: Error between depth maps from original and
       reconstructed features (using frozen DPT Head)
    
    Args:
        feature_weight: Weight for feature reconstruction loss (default: 1.0)
        geometric_weight: Weight for geometric consistency loss (default: 1.0)
    """
    
    def __init__(
        self,
        feature_weight: float = 1.0,
        geometric_weight: float = 1.0,
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        
    def forward(
        self,
        original_features: List[torch.Tensor],
        reconstructed_features: List[torch.Tensor],
        original_depth: Optional[torch.Tensor] = None,
        reconstructed_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            original_features: Original latent_2 features
            reconstructed_features: Reconstructed features from autoencoder
            original_depth: Depth from original features (via DPT Head)
            reconstructed_depth: Depth from reconstructed features (via DPT Head)
            gt_depth: Ground truth depth (optional)
            depth_mask: Valid depth mask (optional)
            
        Returns:
            loss_dict: Dictionary containing individual and total losses
        """
        loss_dict = {}
        
        # Feature reconstruction loss
        feature_loss = 0.0
        for orig, recon in zip(original_features, reconstructed_features):
            feature_loss = feature_loss + F.mse_loss(recon, orig)
        feature_loss = feature_loss / len(original_features)
        loss_dict["loss_feature_recon"] = feature_loss
        
        # Geometric consistency loss
        geometric_loss = torch.tensor(0.0, device=feature_loss.device)
        
        if original_depth is not None and reconstructed_depth is not None:
            # Consistency between original and reconstructed depth
            depth_consistency = F.mse_loss(reconstructed_depth, original_depth)
            geometric_loss = geometric_loss + depth_consistency
            loss_dict["loss_depth_consistency"] = depth_consistency
            
        if gt_depth is not None and reconstructed_depth is not None:
            # Loss against ground truth depth
            if depth_mask is not None:
                gt_depth_loss = F.mse_loss(
                    reconstructed_depth[depth_mask],
                    gt_depth[depth_mask]
                )
            else:
                gt_depth_loss = F.mse_loss(reconstructed_depth, gt_depth)
            geometric_loss = geometric_loss + gt_depth_loss
            loss_dict["loss_gt_depth"] = gt_depth_loss
        
        loss_dict["loss_geometric"] = geometric_loss
        
        # Total loss
        total_loss = (
            self.feature_weight * feature_loss +
            self.geometric_weight * geometric_loss
        )
        loss_dict["loss_total"] = total_loss
        
        return loss_dict


def build_autoencoder_for_vggt(
    latent_dim: int = 512,
    hidden_dim: int = 1024,
    spatial_downsample: int = 4,
) -> LatentAutoEncoder:
    """
    Build a LatentAutoEncoder configured for VGGT's architecture.
    
    Args:
        latent_dim: Dimension of the compressed latent space
        hidden_dim: Hidden dimension for processing
        spatial_downsample: Factor for spatial downsampling
        
    Returns:
        autoencoder: Configured LatentAutoEncoder instance
    """
    return LatentAutoEncoder(
        input_dim=2048,  # VGGT uses 2048D features (1024 frame + 1024 global)
        num_layers=4,  # 4 intermediate layers
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_encoder_blocks=4,
        num_decoder_blocks=4,
        num_heads=16,
        spatial_downsample=spatial_downsample,
        intermediate_layer_idx=[4, 11, 17, 23],  # Standard VGGT layer indices
    )
