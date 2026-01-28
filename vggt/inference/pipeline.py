# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference pipeline for the unified 3D generation model.
"""

import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


@dataclass
class GenerationConfig:
    """Configuration for 3D generation."""
    num_frames: int = 16
    num_diffusion_steps: int = 50
    cfg_scale: float = 7.5
    eta: float = 0.0  # DDIM eta (0 = deterministic)
    seed: Optional[int] = None


@dataclass
class GenerationOutput:
    """Output from 3D generation."""
    latent_z: torch.Tensor  # Generated latent [B, S, P, latent_dim]
    depth: Optional[torch.Tensor] = None  # Predicted depth [B, S, H, W]
    depth_conf: Optional[torch.Tensor] = None  # Depth confidence [B, S, H, W]
    world_points: Optional[torch.Tensor] = None  # 3D points [B, S, H, W, 3]
    camera_poses: Optional[torch.Tensor] = None  # Camera poses [B, S, 9]


class UnifiedInferencePipeline:
    """
    Unified inference pipeline for 3D generation and understanding.
    
    This pipeline integrates all components for end-to-end inference:
    - VGGT for visual feature extraction and depth decoding
    - AutoEncoder for latent compression/decompression
    - DiT for diffusion-based generation
    - LLM for text understanding and condition generation
    
    Args:
        vggt_model: VGGT model instance
        autoencoder: LatentAutoEncoder instance
        dit_model: LatentDiT or LLMDiTWrapper instance
        llm_model: Optional LLM model instance
        diffusion: GaussianDiffusion instance
        device: Device to run inference on
    """
    
    def __init__(
        self,
        vggt_model: nn.Module,
        autoencoder: nn.Module,
        dit_model: nn.Module,
        llm_model: Optional[nn.Module] = None,
        diffusion: Any = None,
        device: str = "cuda",
    ):
        self.device = device
        
        # Set up models
        self.vggt = vggt_model.to(device).eval()
        self.autoencoder = autoencoder.to(device).eval()
        self.dit = dit_model.to(device).eval()
        self.llm = llm_model
        if self.llm is not None:
            self.llm = self.llm.to(device).eval()
        
        self.diffusion = diffusion
        
        # Extract model dimensions from DiT
        self.latent_dim = getattr(dit_model, 'latent_dim', 512)
        self.num_patches = getattr(dit_model, 'num_patches', 256)
        self.num_frames = getattr(dit_model, 'num_frames', 16)
        
        # Freeze all models
        for model in [self.vggt, self.autoencoder, self.dit]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Image preprocessing
        self._setup_preprocessing()
    
    def _setup_preprocessing(self):
        """Set up image preprocessing transforms."""
        try:
            import torchvision.transforms as T
            self.transform = T.Compose([
                T.Resize((518, 518)),
                T.ToTensor(),
            ])
        except ImportError:
            self.transform = None
    
    @classmethod
    def from_pretrained(
        cls,
        vggt_path: str,
        autoencoder_path: str,
        dit_path: str,
        llm_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "UnifiedInferencePipeline":
        """
        Load pipeline from pretrained checkpoints.
        
        Args:
            vggt_path: Path to VGGT checkpoint or HuggingFace model ID
            autoencoder_path: Path to AutoEncoder checkpoint
            dit_path: Path to DiT checkpoint
            llm_path: Optional path/ID for LLM model
            device: Device to load models on
            
        Returns:
            pipeline: Loaded UnifiedInferencePipeline instance
        """
        from vggt.models.vggt import VGGT
        from vggt.models.latent_autoencoder import LatentAutoEncoder
        from vggt.models.latent_dit import LatentDiT
        from vggt.utils.diffusion import GaussianDiffusion
        
        # Load VGGT
        print(f"Loading VGGT from {vggt_path}...")
        if os.path.isfile(vggt_path):
            vggt = VGGT()
            state_dict = torch.load(vggt_path, map_location="cpu")
            vggt.load_state_dict(state_dict)
        else:
            # Try loading from HuggingFace Hub
            vggt = VGGT.from_pretrained(vggt_path)
        
        # Load AutoEncoder
        print(f"Loading AutoEncoder from {autoencoder_path}...")
        autoencoder = LatentAutoEncoder(
            input_dim=2048,
            num_layers=4,
            hidden_dim=1024,
            latent_dim=512,
        )
        state_dict = torch.load(autoencoder_path, map_location="cpu")
        autoencoder.load_state_dict(state_dict)
        
        # Load DiT
        print(f"Loading DiT from {dit_path}...")
        dit = LatentDiT(
            latent_dim=512,
            hidden_dim=1024,
            depth=12,
            num_frames=16,
            num_patches=256,
        )
        state_dict = torch.load(dit_path, map_location="cpu")
        dit.load_state_dict(state_dict)
        
        # Load LLM if specified
        llm = None
        if llm_path is not None:
            print(f"Loading LLM from {llm_path}...")
            try:
                from transformers import AutoModelForCausalLM
                llm = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            except ImportError:
                print("Warning: transformers not installed, LLM not loaded")
        
        # Create diffusion
        diffusion = GaussianDiffusion(
            num_timesteps=1000,
            beta_schedule="cosine",
            model_mean_type="epsilon",
        )
        
        return cls(
            vggt_model=vggt,
            autoencoder=autoencoder,
            dit_model=dit,
            llm_model=llm,
            diffusion=diffusion,
            device=device,
        )
    
    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess input image for the pipeline.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            
        Returns:
            tensor: Preprocessed image tensor [1, 1, 3, H, W]
        """
        if isinstance(image, Image.Image):
            if self.transform is not None:
                tensor = self.transform(image)
            else:
                # Manual preprocessing
                image = image.resize((518, 518))
                tensor = torch.from_numpy(np.array(image)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
            if tensor.max() > 1:
                tensor = tensor / 255.0
            if len(tensor.shape) == 3:
                if tensor.shape[-1] == 3:
                    tensor = tensor.permute(2, 0, 1)
        else:
            tensor = image
        
        # Add batch and sequence dimensions
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 4:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def extract_visual_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract DINO visual features from images.
        
        Args:
            images: Input images [B, N_img, 3, H, W]
            
        Returns:
            features: DINO features [B, N_img, P, 1024]
        """
        B, N_img, C, H, W = images.shape
        
        # Reshape for batch processing
        images_flat = images.view(B * N_img, C, H, W)
        
        # Normalize images
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_norm = (images_flat - mean) / std
        
        # Get patch tokens from DINO
        patch_tokens = self.vggt.aggregator.patch_embed(images_norm)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        
        # Reshape back
        P = patch_tokens.shape[1]
        features = patch_tokens.view(B, N_img, P, -1)
        
        return features
    
    @torch.no_grad()
    def encode_to_latent(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to compressed latent z via VGGT + AutoEncoder.
        
        Args:
            images: Input images [B, S, 3, H, W]
            
        Returns:
            z: Compressed latent [B, S, P', latent_dim]
        """
        # Get VGGT aggregated features
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
        
        # Extract relevant layers and encode
        layer_features = self.autoencoder.extract_layers_from_aggregated(
            aggregated_tokens_list, patch_start_idx
        )
        z = self.autoencoder.encode(layer_features)
        
        return z
    
    @torch.no_grad()
    def decode_to_depth(
        self,
        z: torch.Tensor,
        reference_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent z to depth maps via AutoEncoder + DPT Head.
        
        Args:
            z: Latent representation [B, S, P', latent_dim]
            reference_images: Reference images for DPT [B, S, 3, H, W]
            
        Returns:
            outputs: Dict with 'depth' and 'depth_conf'
        """
        # Decode latent to layer features
        layer_features = self.autoencoder.decode(z)
        
        # Reconstruct aggregated_tokens_list format
        B, S, P, C = layer_features[0].shape
        
        # Get patch_start_idx safely with default
        patch_start_idx = getattr(self.vggt.aggregator, 'patch_start_idx', 5)
        
        # Create dummy special tokens
        dummy_special = torch.zeros(B, S, patch_start_idx, C, device=z.device)
        
        # Build full list (simplified - key layers only)
        aggregated_tokens_list = []
        layer_idx_mapping = {4: 0, 11: 1, 17: 2, 23: 3}
        
        for i in range(24):
            if i in layer_idx_mapping:
                feat = layer_features[layer_idx_mapping[i]]
                feat_with_special = torch.cat([dummy_special, feat], dim=2)
            else:
                # Interpolate from nearest key layer
                nearest_key = min(layer_idx_mapping.keys(), key=lambda x: abs(x - i))
                feat = layer_features[layer_idx_mapping[nearest_key]]
                feat_with_special = torch.cat([dummy_special, feat], dim=2)
            aggregated_tokens_list.append(feat_with_special)
        
        # Use DPT Head to generate depth
        depth, depth_conf = self.vggt.depth_head(
            aggregated_tokens_list, reference_images, patch_start_idx
        )
        
        return {
            "depth": depth,
            "depth_conf": depth_conf,
        }
    
    @torch.no_grad()
    def generate_unconditional(
        self,
        batch_size: int = 1,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """
        Generate latent z unconditionally using DiT.
        
        Args:
            batch_size: Number of samples to generate
            config: Generation configuration
            
        Returns:
            output: GenerationOutput with generated latent
        """
        if config is None:
            config = GenerationConfig()
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        # Determine shape from model config
        num_frames = config.num_frames if config.num_frames else self.num_frames
        shape = (batch_size, num_frames, self.num_patches, self.latent_dim)
        
        # Sample using diffusion
        z = self.diffusion.ddim_sample_loop(
            self.dit,
            shape,
            num_steps=config.num_diffusion_steps,
            eta=config.eta,
            device=self.device,
            progress=True,
        )
        
        return GenerationOutput(latent_z=z)
    
    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """
        Generate 3D content from image and optional text prompt.
        
        This is the main generation function for Phase 3.
        
        Note: Text prompt conditioning requires LLM integration, which is not
        yet fully implemented. Currently, the prompt parameter is ignored and
        generation is unconditional.
        
        Args:
            image: Input image (single image or sparse images)
            prompt: Optional text prompt for generation (not yet fully supported)
            config: Generation configuration
            
        Returns:
            output: GenerationOutput with generated latent and decoded outputs
        """
        if config is None:
            config = GenerationConfig()
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        # Preprocess image
        images = self.preprocess_image(image)
        B = images.shape[0]
        
        # Extract visual features
        visual_features = self.extract_visual_features(images)
        
        # Check if we have LLM for text conditioning
        # Note: Full LLM integration is a future enhancement
        condition_tokens = None
        if self.llm is not None and prompt is not None:
            import warnings
            warnings.warn(
                "Text prompt conditioning is not yet fully implemented. "
                "The prompt will be ignored and generation will be unconditional.",
                UserWarning
            )
        
        # Determine shape from model config
        num_frames = config.num_frames if config.num_frames else self.num_frames
        shape = (B, num_frames, self.num_patches, self.latent_dim)
        
        # Sample using diffusion with the DiT model
        z = self.diffusion.ddim_sample_loop(
            self.dit,
            shape,
            condition=condition_tokens,
            num_steps=config.num_diffusion_steps,
            eta=config.eta,
            device=self.device,
            progress=True,
        )
        
        # Create reference images for depth decoding
        # Expand input image to match generated frames
        reference_images = images[:, 0:1].expand(-1, num_frames, -1, -1, -1)
        
        # Decode to depth
        depth_outputs = self.decode_to_depth(z, reference_images)
        
        return GenerationOutput(
            latent_z=z,
            depth=depth_outputs["depth"],
            depth_conf=depth_outputs["depth_conf"],
        )
    
    def to(self, device: str) -> "UnifiedInferencePipeline":
        """Move pipeline to specified device."""
        self.device = device
        self.vggt = self.vggt.to(device)
        self.autoencoder = self.autoencoder.to(device)
        self.dit = self.dit.to(device)
        if self.llm is not None:
            self.llm = self.llm.to(device)
        return self


def visualize_depth_sequence(
    depths: torch.Tensor,
    output_dir: str,
    colormap: str = "turbo",
) -> List[str]:
    """
    Visualize a sequence of depth maps.
    
    Args:
        depths: Depth tensor [B, S, H, W] or [B, S, H, W, 1]
        output_dir: Directory to save visualizations
        colormap: Matplotlib colormap name
        
    Returns:
        paths: List of saved image paths
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Warning: matplotlib not installed, cannot visualize")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(depths.shape) == 5:
        depths = depths[..., 0]
    
    depths = depths.cpu().numpy()
    B, S, H, W = depths.shape
    
    paths = []
    for b in range(B):
        for s in range(S):
            depth = depths[b, s]
            
            # Normalize for visualization
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Apply colormap
            cmap = cm.get_cmap(colormap)
            depth_colored = cmap(depth_norm)
            
            # Save
            path = os.path.join(output_dir, f"depth_b{b}_s{s:03d}.png")
            plt.imsave(path, depth_colored)
            paths.append(path)
    
    return paths


def create_point_cloud_from_depth(
    depth: torch.Tensor,
    intrinsics: Optional[torch.Tensor] = None,
    extrinsics: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Create a 3D point cloud from depth maps.
    
    Args:
        depth: Depth tensor [B, S, H, W] or [H, W]
        intrinsics: Camera intrinsics [B, S, 3, 3] or [3, 3]
        extrinsics: Camera extrinsics [B, S, 4, 4] or [4, 4]
        
    Returns:
        points: Point cloud as numpy array [N, 3]
    """
    depth = depth.cpu().numpy()
    
    if len(depth.shape) == 2:
        depth = depth[None, None]
    elif len(depth.shape) == 3:
        depth = depth[None]
    
    B, S, H, W = depth.shape
    
    # Create pixel grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Default intrinsics if not provided (assume fx=fy=518, cx=cy=259)
    if intrinsics is None:
        fx = fy = 518.0
        cx, cy = W / 2, H / 2
    else:
        intrinsics = intrinsics.cpu().numpy()
        if len(intrinsics.shape) == 2:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        else:
            # Use first intrinsic
            fx, fy = intrinsics[0, 0, 0, 0], intrinsics[0, 0, 1, 1]
            cx, cy = intrinsics[0, 0, 0, 2], intrinsics[0, 0, 1, 2]
    
    all_points = []
    
    for b in range(B):
        for s in range(S):
            d = depth[b, s]
            
            # Back-project to 3D
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = d
            
            # Stack and reshape
            points = np.stack([x, y, z], axis=-1)
            valid = d > 0.01  # Filter invalid depth
            points = points[valid]
            
            # Apply extrinsics if provided
            if extrinsics is not None:
                ext = extrinsics
                if len(ext.shape) == 4:
                    ext = ext[b, s].cpu().numpy()
                elif len(ext.shape) == 3:
                    ext = ext[s].cpu().numpy()
                else:
                    ext = ext.cpu().numpy()
                
                # Transform points
                R = ext[:3, :3]
                t = ext[:3, 3]
                points = points @ R.T + t
            
            all_points.append(points)
    
    return np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))
