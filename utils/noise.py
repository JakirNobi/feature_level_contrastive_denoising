"""
Noise injection utilities for robust face detection training.
Supports Gaussian, Poisson, Salt & Pepper, and Speckle noise.
"""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class NoiseConfig:
    """Configuration for noise injection during training."""
    noise_types: Tuple[str, ...] = ('gaussian',)
    noise_params: Tuple[int, ...] = (10, 25, 50)


def add_noise(
    images: torch.Tensor,
    noise_type: str = 'gaussian',
    param: float = 25.0
) -> torch.Tensor:
    """
    Inject noise into a batch of images.

    Args:
        images: Tensor of shape (B, C, H, W) in range [0, 1] or [0, 255]
        noise_type: 'gaussian', 'poisson', 'salt_pepper', or 'speckle'
        param: Noise intensity parameter

    Returns:
        Noisy images of same shape and range as input
    """
    scale: float = 255.0 if images.max() > 1.5 else 1.0

    if noise_type == 'gaussian':
        sigma: float = param / 255.0 * scale
        noise: torch.Tensor = torch.randn_like(images) * sigma
        return torch.clamp(images + noise, 0.0, scale)

    if noise_type == 'poisson':
        if scale == 1.0:
            img_scaled: torch.Tensor = images * 255.0
        else:
            img_scaled = images
        noisy: torch.Tensor = torch.poisson(torch.clamp(img_scaled, min=1e-5))
        noisy = noisy / 255.0 * scale
        return torch.clamp(noisy, 0.0, scale)

    if noise_type == 'salt_pepper':
        p: float = param / 100.0
        mask: torch.Tensor = torch.rand_like(images)
        noisy = images.clone()
        noisy[mask < p / 2] = 0.0
        noisy[mask > 1 - p / 2] = scale
        return noisy

    if noise_type == 'speckle':
        sigma = param / 255.0 * scale
        noise = torch.randn_like(images) * sigma
        return torch.clamp(images + images * noise, 0.0, scale)

    if noise_type == 'clean':
        return images.clone()

    raise ValueError(f"Unknown noise type: {noise_type}")