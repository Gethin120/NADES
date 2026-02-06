"""
Differential Privacy utilities for Federated Learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def clip_gradients(
    gradients: torch.Tensor, max_norm: float, norm_type: float = 2.0
) -> Tuple[torch.Tensor, float]:
    """
    Clip gradients to a maximum norm.

    Args:
        gradients: Gradient tensor to clip
        max_norm: Maximum norm value
        norm_type: Type of norm (default: 2.0 for L2 norm)

    Returns:
        Tuple of (clipped_gradients, clip_coef):
        - clipped_gradients: Clipped gradient tensor
        - clip_coef: Clipping coefficient applied
    """
    if max_norm <= 0:
        return gradients, 1.0

    # Compute gradient norm
    grad_norm = torch.norm(gradients, p=norm_type)

    # Compute clipping coefficient
    clip_coef = max_norm / (grad_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)

    # Clip gradients
    clipped_gradients = gradients * clip_coef

    return clipped_gradients, clip_coef.item()


def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_multiplier: float,
    sensitivity: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor for differential privacy.

    Args:
        tensor: Input tensor
        noise_multiplier: Noise multiplier (sigma)
        sensitivity: Sensitivity of the function (default: 1.0)
        device: Device to generate noise on (default: same as tensor)

    Returns:
        Noisy tensor
    """
    if device is None:
        device = tensor.device

    # Compute noise scale: sigma * sensitivity
    noise_scale = noise_multiplier * sensitivity

    # Generate Gaussian noise
    noise = torch.normal(mean=0.0, std=noise_scale, size=tensor.shape, device=device)

    # Add noise to tensor
    noisy_tensor = tensor + noise

    return noisy_tensor


def per_sample_gradient_norm(
    model: nn.Module, loss_fn: callable, batch: Tuple, max_norm: float
) -> torch.Tensor:
    """
    Compute per-sample gradient norms for a batch.

    This is used for per-sample gradient clipping in DP-SGD.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        batch: Input batch (data, labels)
        max_norm: Maximum norm for clipping

    Returns:
        Clipping coefficients for each sample
    """
    model.train()
    batch_size = (
        batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
    )

    # Compute per-sample gradients
    clip_coefs = torch.ones(batch_size, device=next(model.parameters()).device)

    # For each sample in the batch
    for i in range(batch_size):
        # Zero gradients
        model.zero_grad()

        # Get single sample
        if isinstance(batch[0], torch.Tensor):
            sample_x = batch[0][i : i + 1]
            sample_y = batch[1][i : i + 1]
        else:
            sample_x = [b[i : i + 1] for b in batch[0]]
            sample_y = batch[1][i : i + 1]

        # Forward pass
        output = model(sample_x)
        loss = loss_fn(output, sample_y)

        # Backward pass
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2.0)

        # Compute clipping coefficient
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coefs[i] = min(clip_coef, 1.0)

    return clip_coefs

