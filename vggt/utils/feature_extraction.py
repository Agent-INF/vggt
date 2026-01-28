# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Feature extraction utilities for preprocessing video data.

This module provides utilities to extract and cache VGGT features for training:
- Extract DINO latent_1 features from images
- Extract full latent_2 features from video sequences
- Compress and cache latent_z using trained AutoEncoder
"""

import os
import json
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    batch_size: int = 8
    num_workers: int = 4
    img_size: int = 518
    patch_size: int = 14
    device: str = "cuda"
    save_latent1: bool = True
    save_latent2: bool = True
    save_latent_z: bool = False  # Requires trained AutoEncoder
    compression_level: int = 9  # For npz compression


class VGGTFeatureExtractor:
    """
    Extract features from VGGT model for training downstream models.
    
    This class handles:
    1. DINO latent_1 extraction (from patch embeddings)
    2. Full latent_2 extraction (from Aggregator)
    3. Compressed latent_z extraction (using AutoEncoder)
    
    Args:
        vggt_model: VGGT model instance (will be frozen)
        autoencoder: Optional AutoEncoder for latent_z extraction
        config: ExtractionConfig instance
    """
    
    def __init__(
        self,
        vggt_model: nn.Module,
        autoencoder: Optional[nn.Module] = None,
        config: Optional[ExtractionConfig] = None,
    ):
        self.config = config or ExtractionConfig()
        self.device = self.config.device
        
        # Set up VGGT model
        self.vggt = vggt_model.to(self.device).eval()
        for param in self.vggt.parameters():
            param.requires_grad = False
        
        # Set up AutoEncoder if provided
        self.autoencoder = None
        if autoencoder is not None:
            self.autoencoder = autoencoder.to(self.device).eval()
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        
        # Get patch info
        self.patch_size = self.config.patch_size
        self.num_patches_per_dim = self.config.img_size // self.patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Normalization constants
        self._setup_normalization()
    
    def _setup_normalization(self):
        """Set up image normalization constants."""
        # Store normalization tensors as regular attributes (not buffers since we're not nn.Module)
        # Shape [1, 3, 1, 1] for proper broadcasting with [B*S, C, H, W] images
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    @torch.no_grad()
    def extract_dino_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract DINO patch features (latent_1) from images.
        
        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1]
            
        Returns:
            features: DINO features [B, S, P, 1024]
        """
        B, S, C, H, W = images.shape
        images = images.to(self.device)
        
        # Reshape to [B*S, C, H, W] first, then normalize
        images_flat = images.view(B * S, C, H, W)
        images_norm = (images_flat - self.mean) / self.std
        
        # Get patch tokens
        patch_tokens = self.vggt.aggregator.patch_embed(images_norm)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        
        P = patch_tokens.shape[1]
        features = patch_tokens.view(B, S, P, -1)
        
        return features
    
    @torch.no_grad()
    def extract_aggregated_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Extract full aggregated features (latent_2) from images.
        
        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1]
            
        Returns:
            aggregated_tokens_list: List of 24 tensors [B, S, P, 2048]
            patch_start_idx: Index where patch tokens start
        """
        images = images.to(self.device)
        
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
        
        return aggregated_tokens_list, patch_start_idx
    
    @torch.no_grad()
    def extract_key_layer_features(
        self,
        images: torch.Tensor,
        layer_indices: List[int] = [4, 11, 17, 23],
    ) -> List[torch.Tensor]:
        """
        Extract features from specific layers (for AutoEncoder training).
        
        Args:
            images: Input images [B, S, 3, H, W]
            layer_indices: Indices of layers to extract
            
        Returns:
            layer_features: List of feature tensors [B, S, P, 2048]
        """
        aggregated_tokens_list, patch_start_idx = self.extract_aggregated_features(images)
        
        layer_features = []
        for idx in layer_indices:
            # Extract only patch tokens (skip camera and register tokens)
            feat = aggregated_tokens_list[idx][:, :, patch_start_idx:]
            layer_features.append(feat)
        
        return layer_features
    
    @torch.no_grad()
    def extract_compressed_latent(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract compressed latent z using AutoEncoder.
        
        Args:
            images: Input images [B, S, 3, H, W]
            
        Returns:
            z: Compressed latent [B, S, P', latent_dim]
        """
        if self.autoencoder is None:
            raise ValueError("AutoEncoder not loaded")
        
        layer_features = self.extract_key_layer_features(images)
        z = self.autoencoder.encode(layer_features)
        
        return z
    
    def process_video_sequence(
        self,
        video_frames: torch.Tensor,
        output_path: str,
        extract_depth: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video sequence and save features.
        
        Args:
            video_frames: Video frames [S, 3, H, W] or [B, S, 3, H, W]
            output_path: Path to save extracted features
            extract_depth: Whether to also extract depth predictions
            
        Returns:
            info: Dictionary with extraction info
        """
        import numpy as np
        
        # Ensure batch dimension
        if len(video_frames.shape) == 4:
            video_frames = video_frames.unsqueeze(0)
        
        video_frames = video_frames.to(self.device)
        B, S, C, H, W = video_frames.shape
        
        results = {"shape": list(video_frames.shape)}
        
        # Extract DINO features
        if self.config.save_latent1:
            latent1 = self.extract_dino_features(video_frames)
            results["latent1_shape"] = list(latent1.shape)
        
        # Extract aggregated features
        if self.config.save_latent2:
            layer_features = self.extract_key_layer_features(video_frames)
            results["latent2_shapes"] = [list(f.shape) for f in layer_features]
        
        # Extract compressed latent
        if self.config.save_latent_z and self.autoencoder is not None:
            latent_z = self.extract_compressed_latent(video_frames)
            results["latent_z_shape"] = list(latent_z.shape)
        
        # Extract depth if requested
        depth = None
        depth_head = getattr(self.vggt, 'depth_head', None)
        if extract_depth and depth_head is not None:
            aggregated_tokens_list, patch_start_idx = self.extract_aggregated_features(video_frames)
            depth, depth_conf = depth_head(
                aggregated_tokens_list, video_frames, patch_start_idx
            )
            results["depth_shape"] = list(depth.shape)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_dict = {}
        if self.config.save_latent1:
            save_dict["latent1"] = latent1.cpu().numpy().astype(np.float16)
        if self.config.save_latent2:
            for i, feat in enumerate(layer_features):
                save_dict[f"latent2_layer{i}"] = feat.cpu().numpy().astype(np.float16)
        if self.config.save_latent_z and self.autoencoder is not None:
            save_dict["latent_z"] = latent_z.cpu().numpy().astype(np.float16)
        if depth is not None:
            save_dict["depth"] = depth.cpu().numpy().astype(np.float16)
            save_dict["depth_conf"] = depth_conf.cpu().numpy().astype(np.float16)
        
        np.savez_compressed(output_path, **save_dict)
        results["output_path"] = output_path
        
        return results
    
    def process_dataset(
        self,
        dataset,
        output_dir: str,
        num_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process an entire dataset and save features.
        
        Args:
            dataset: PyTorch dataset yielding (video_frames, metadata)
            output_dir: Directory to save extracted features
            num_samples: Optional limit on number of samples
            
        Returns:
            results: List of extraction results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        num_to_process = num_samples or len(dataset)
        
        for idx in tqdm(range(min(num_to_process, len(dataset))), desc="Extracting features"):
            try:
                sample = dataset[idx]
                
                # Handle different dataset formats
                if isinstance(sample, dict):
                    video_frames = sample.get("images", sample.get("video"))
                    sample_id = sample.get("id", sample.get("name", str(idx)))
                elif isinstance(sample, tuple):
                    video_frames = sample[0]
                    sample_id = str(idx)
                else:
                    video_frames = sample
                    sample_id = str(idx)
                
                output_path = os.path.join(output_dir, f"{sample_id}.npz")
                
                result = self.process_video_sequence(video_frames, output_path)
                result["sample_id"] = sample_id
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        # Save index file
        index_path = os.path.join(output_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results


class FeatureCacheDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-extracted features from cache.
    
    Args:
        cache_dir: Directory containing cached features
        load_latent1: Whether to load latent_1 features
        load_latent2: Whether to load latent_2 features
        load_latent_z: Whether to load compressed latent_z
        load_depth: Whether to load depth predictions
    """
    
    def __init__(
        self,
        cache_dir: str,
        load_latent1: bool = True,
        load_latent2: bool = True,
        load_latent_z: bool = False,
        load_depth: bool = False,
    ):
        import numpy as np
        
        self.cache_dir = cache_dir
        self.load_latent1 = load_latent1
        self.load_latent2 = load_latent2
        self.load_latent_z = load_latent_z
        self.load_depth = load_depth
        
        # Load index
        index_path = os.path.join(cache_dir, "index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.index = json.load(f)
        else:
            # Find all npz files
            self.index = [
                {"sample_id": Path(f).stem, "output_path": os.path.join(cache_dir, f)}
                for f in os.listdir(cache_dir)
                if f.endswith(".npz")
            ]
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import numpy as np
        
        info = self.index[idx]
        path = info.get("output_path", os.path.join(self.cache_dir, f"{info['sample_id']}.npz"))
        
        data = np.load(path)
        
        result = {"sample_id": info.get("sample_id", str(idx))}
        
        if self.load_latent1 and "latent1" in data:
            result["latent1"] = torch.from_numpy(data["latent1"].astype(np.float32))
        
        if self.load_latent2:
            latent2_layers = []
            for i in range(4):
                key = f"latent2_layer{i}"
                if key in data:
                    latent2_layers.append(torch.from_numpy(data[key].astype(np.float32)))
            if latent2_layers:
                result["latent2"] = latent2_layers
        
        if self.load_latent_z and "latent_z" in data:
            result["latent_z"] = torch.from_numpy(data["latent_z"].astype(np.float32))
        
        if self.load_depth:
            if "depth" in data:
                result["depth"] = torch.from_numpy(data["depth"].astype(np.float32))
            if "depth_conf" in data:
                result["depth_conf"] = torch.from_numpy(data["depth_conf"].astype(np.float32))
        
        return result


def extract_features_from_video_dir(
    video_dir: str,
    output_dir: str,
    vggt_path: str,
    autoencoder_path: Optional[str] = None,
    config: Optional[ExtractionConfig] = None,
):
    """
    Extract features from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files or frame directories
        output_dir: Directory to save extracted features
        vggt_path: Path to VGGT model checkpoint
        autoencoder_path: Optional path to AutoEncoder checkpoint
        config: Extraction configuration
    """
    import cv2
    import numpy as np
    
    from vggt.models.vggt import VGGT
    
    config = config or ExtractionConfig()
    
    # Load VGGT
    logger.info(f"Loading VGGT from {vggt_path}")
    if os.path.isfile(vggt_path):
        vggt = VGGT()
        state_dict = torch.load(vggt_path, map_location="cpu")
        vggt.load_state_dict(state_dict)
    else:
        vggt = VGGT.from_pretrained(vggt_path)
    
    # Load AutoEncoder if provided
    autoencoder = None
    if autoencoder_path is not None:
        from vggt.models.latent_autoencoder import LatentAutoEncoder
        logger.info(f"Loading AutoEncoder from {autoencoder_path}")
        autoencoder = LatentAutoEncoder()
        state_dict = torch.load(autoencoder_path, map_location="cpu")
        autoencoder.load_state_dict(state_dict)
    
    # Create extractor
    extractor = VGGTFeatureExtractor(vggt, autoencoder, config)
    
    # Find all videos/frame dirs
    entries = []
    for item in os.listdir(video_dir):
        item_path = os.path.join(video_dir, item)
        if os.path.isdir(item_path):
            # Directory of frames
            entries.append(("frames", item_path, item))
        elif item.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Video file
            entries.append(("video", item_path, Path(item).stem))
    
    logger.info(f"Found {len(entries)} entries to process")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for entry_type, entry_path, entry_name in tqdm(entries, desc="Processing"):
        try:
            # Load frames
            if entry_type == "frames":
                frame_files = sorted([
                    f for f in os.listdir(entry_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                frames = []
                for ff in frame_files:
                    img = cv2.imread(os.path.join(entry_path, ff))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (config.img_size, config.img_size))
                    frames.append(img)
                frames = np.stack(frames)
            else:
                # Video file
                cap = cv2.VideoCapture(entry_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (config.img_size, config.img_size))
                    frames.append(frame)
                cap.release()
                frames = np.stack(frames)
            
            # Convert to tensor [S, 3, H, W] normalized to [0, 1]
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)
            
            # Process
            output_path = os.path.join(output_dir, f"{entry_name}.npz")
            result = extractor.process_video_sequence(frames, output_path)
            result["sample_id"] = entry_name
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to process {entry_name}: {e}")
            continue
    
    # Save index
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Processed {len(results)} entries, saved to {output_dir}")
    return results
