"""Learning rate schedulers for training.

This module provides custom learning rate scheduler implementations
with warmup phases for training neural networks.
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine annealing.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of the initial LR.
        last_epoch: The index of last epoch when resuming training.

    Returns:
        LambdaLR scheduler with linear warmup and cosine annealing.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with linear warmup and linear decay.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        last_epoch: The index of last epoch when resuming training.

    Returns:
        LambdaLR scheduler with linear warmup and linear decay.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay after warmup
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
