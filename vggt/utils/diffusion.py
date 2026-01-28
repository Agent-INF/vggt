# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Diffusion utilities for latent space generation.

This module provides noise schedules, diffusion processes, and sampling utilities
for the latent DiT model used in 3D generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> torch.Tensor:
    """
    Get a pre-defined beta schedule for the given name.
    
    Args:
        schedule_name: Name of the schedule ('linear', 'cosine', 'sqrt')
        num_diffusion_timesteps: Number of diffusion steps
        
    Returns:
        A 1-D tensor of betas for each timestep
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al 2020.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - (t + 0.001) ** 0.5,
        )
    
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar_fn, max_beta: float = 0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_bar function.
    
    Args:
        num_diffusion_timesteps: Number of timesteps
        alpha_bar_fn: A function that takes a timestep from 0 to 1 and produces
                      the cumulative product of (1-beta) up to that timestep
        max_beta: The maximum beta value to use
        
    Returns:
        A 1-D tensor of betas
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


class GaussianDiffusion:
    """
    Gaussian Diffusion process for training and sampling.
    
    This class implements the forward and reverse diffusion processes
    for training diffusion models on latent representations.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        model_mean_type: str = "epsilon",  # "epsilon" or "x0" or "v"
        model_var_type: str = "fixed_small",  # "fixed_small", "fixed_large", "learned"
        loss_type: str = "mse",  # "mse" or "huber"
    ):
        """
        Initialize the diffusion process.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Type of beta schedule
            model_mean_type: What the model predicts ("epsilon", "x0", or "v")
            model_var_type: How to handle variance
            loss_type: Type of loss function
        """
        self.num_timesteps = num_timesteps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # Get beta schedule
        betas = get_named_beta_schedule(beta_schedule, num_timesteps)
        
        # Calculate alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Store as buffers (will be moved to device when needed)
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        ).float()
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).float()
        
    def _extract(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple) -> torch.Tensor:
        """
        Extract values from a 1-D array for a batch of indices.
        
        Args:
            arr: The 1-D array to extract from
            timesteps: A tensor of indices into the array
            broadcast_shape: The shape to broadcast the result to
            
        Returns:
            A tensor of values extracted from arr
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        
        In other words, sample from q(x_t | x_0).
        
        Args:
            x_start: The initial data batch [B, ...]
            t: A 1-D tensor of timesteps [B]
            noise: Optional pre-generated noise
            
        Returns:
            A noisy version of x_start
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:
        
            q(x_{t-1} | x_t, x_0)
            
        Args:
            x_start: The initial data x_0
            x_t: The noisy data at timestep t
            t: The timesteps
            
        Returns:
            A tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from epsilon prediction.
        
        Args:
            x_t: The noisy data at timestep t
            t: The timesteps
            eps: The predicted noise
            
        Returns:
            The predicted x_0
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_eps_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict epsilon from x_0 prediction.
        
        Args:
            x_t: The noisy data at timestep t
            t: The timesteps
            x_0: The predicted clean data
            
        Returns:
            The predicted noise
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t) parameters.
        
        Args:
            model: The denoising model
            x_t: The noisy data at timestep t
            t: The timesteps
            condition: Optional conditioning information
            clip_denoised: Whether to clip the denoised sample
            
        Returns:
            A tuple of (mean, variance, log_variance, x_0_pred)
        """
        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, condition)
        else:
            model_output = model(x_t, t)
        
        # Handle different prediction types
        if self.model_mean_type == "epsilon":
            x_0_pred = self.predict_x0_from_eps(x_t, t, model_output)
        elif self.model_mean_type == "x0":
            x_0_pred = model_output
        elif self.model_mean_type == "v":
            # v-prediction: v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            x_0_pred = sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")
        
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)
        
        # Get posterior mean and variance
        mean, variance, log_variance = self.q_posterior_mean_variance(x_0_pred, x_t, t)
        
        return mean, variance, log_variance, x_0_pred
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from the model at the given timestep.
        
        Args:
            model: The denoising model
            x_t: The noisy data at timestep t
            t: The timesteps (same value for all samples)
            condition: Optional conditioning information
            clip_denoised: Whether to clip the denoised sample
            
        Returns:
            The sampled x_{t-1}
        """
        mean, _, log_variance, _ = self.p_mean_variance(
            model, x_t, t, condition, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from the model by iteratively denoising.
        
        Args:
            model: The denoising model
            shape: The shape of the samples to generate
            condition: Optional conditioning information
            clip_denoised: Whether to clip the denoised samples
            device: The device to generate samples on
            progress: Whether to show a progress bar
            
        Returns:
            The generated samples
        """
        if device is None:
            device = next(model.parameters()).device
            
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Optionally use tqdm for progress
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            try:
                from tqdm import tqdm
                indices = tqdm(indices, desc="Sampling")
            except ImportError:
                pass
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                x = self.p_sample(model, x, t, condition, clip_denoised)
                
        return x
    
    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Sample x_{t_prev} from x_t using DDIM.
        
        Args:
            model: The denoising model
            x_t: The noisy data at timestep t
            t: Current timesteps
            t_prev: Previous timesteps
            condition: Optional conditioning information
            clip_denoised: Whether to clip the denoised sample
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            
        Returns:
            The sampled x_{t_prev}
        """
        _, _, _, x_0_pred = self.p_mean_variance(
            model, x_t, t, condition, clip_denoised
        )
        
        # Get alpha values
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        
        # Compute predicted epsilon
        eps_pred = self.predict_eps_from_x0(x_t, t, x_0_pred)
        
        # Compute sigma for stochasticity
        sigma = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * eps_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
            
        return x_prev
    
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        device: Optional[torch.device] = None,
        eta: float = 0.0,
        num_steps: int = 50,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM sampling.
        
        Args:
            model: The denoising model
            shape: The shape of the samples to generate
            condition: Optional conditioning information
            clip_denoised: Whether to clip the denoised samples
            device: The device to generate samples on
            eta: DDIM eta parameter
            num_steps: Number of DDIM steps
            progress: Whether to show a progress bar
            
        Returns:
            The generated samples
        """
        if device is None:
            device = next(model.parameters()).device
            
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep sequence
        skip = self.num_timesteps // num_steps
        seq = list(range(0, self.num_timesteps, skip))
        seq_prev = [0] + seq[:-1]
        
        indices = list(zip(reversed(seq), reversed(seq_prev)))
        if progress:
            try:
                from tqdm import tqdm
                indices = tqdm(indices, desc="DDIM Sampling")
            except ImportError:
                pass
        
        for t_cur, t_prev in indices:
            t = torch.tensor([t_cur] * shape[0], device=device)
            t_prev_tensor = torch.tensor([t_prev] * shape[0], device=device)
            with torch.no_grad():
                x = self.ddim_sample(model, x, t, t_prev_tensor, condition, clip_denoised, eta)
                
        return x
    
    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute training losses for a single timestep.
        
        Args:
            model: The denoising model
            x_start: The clean data
            t: The timesteps
            condition: Optional conditioning information
            noise: Optional pre-generated noise
            
        Returns:
            A dictionary containing loss values
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_t = self.q_sample(x_start, t, noise)
        
        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, condition)
        else:
            model_output = model(x_t, t)
        
        # Compute target based on prediction type
        if self.model_mean_type == "epsilon":
            target = noise
        elif self.model_mean_type == "x0":
            target = x_start
        elif self.model_mean_type == "v":
            # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")
        
        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        loss = loss.mean()
        
        return {"loss": loss, "mse": F.mse_loss(model_output, target)}


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: A 1-D tensor of N indices, one per batch element
        dim: The dimension of the output
        max_period: Controls the minimum frequency of the embeddings
        
    Returns:
        An [N x dim] tensor of positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
