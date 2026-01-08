"""Parallel execution utilities for validation and inference.

This module provides helper functions for parallel task execution with
timeout handling and CUDA memory management.
"""

import gc
import multiprocessing as mp
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
from lightning.pytorch.utilities import rank_zero_info
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError
from torch import Tensor
from tqdm import tqdm


@contextmanager
def handle_cuda_oom(action_name: str = "operation"):
    """Context manager to handle CUDA out-of-memory errors gracefully.

    Usage:
        with handle_cuda_oom("sampling"):
            result = model(batch)
        # If OOM occurs, logs warning and clears cache

    Args:
        action_name: Description of the operation for logging.

    Yields:
        None

    Raises:
        RuntimeError: Re-raises if the error is not OOM-related.
    """
    try:
        yield
    except RuntimeError as e:
        # Check for CUDA OOM using torch's built-in method when available
        is_oom = False
        if hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError):
            is_oom = True
        elif "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            is_oom = True

        if is_oom:
            rank_zero_info(f"| WARNING: CUDA out of memory during {action_name}. Clearing cache.")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise


def expand_tensor_for_multiplicity(
    tensor: Tensor,
    multiplicity: int,
    target_size: Optional[int] = None,
    pad_value: float = 0.0,
) -> Tensor:
    """Expand a tensor for multiplicity and optionally resize to target_size.

    Args:
        tensor: Input tensor of shape (B, N, ...) or (B, N)
        multiplicity: Number of times to repeat each sample
        target_size: Optional target size for dimension 1. If provided, pads or slices.
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Expanded tensor of shape (B*multiplicity, target_size or N, ...)
    """
    if target_size is not None:
        current_size = tensor.shape[1]
        if target_size > current_size:
            # Pad
            pad_shape = list(tensor.shape)
            pad_shape[1] = target_size - current_size
            padding = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=1)
        elif target_size < current_size:
            # Slice
            tensor = tensor[:, :target_size]

    return tensor.repeat_interleave(multiplicity, dim=0)


def run_parallel_tasks(
    tasks: list,
    task_fn: Callable,
    num_workers: int,
    timeout_seconds: float,
    task_name: str,
    default_result: Any,
) -> list:
    """Execute tasks in parallel using ProcessPool with consistent error handling.

    Args:
        tasks: List of task arguments (each element is passed to task_fn)
        task_fn: Function to execute for each task
        num_workers: Number of parallel workers
        timeout_seconds: Timeout per task in seconds
        task_name: Name for logging (e.g., "Prim RMSD matching")
        default_result: Result to use when a task fails or times out

    Returns:
        List of results, one per task
    """
    if not tasks:
        return []

    rank_zero_info(f"| Computing {task_name} for {len(tasks)} structures (workers={num_workers})...")
    results = []

    with ProcessPool(max_workers=num_workers, context=mp.get_context("spawn")) as pool:
        future_map = {
            pool.schedule(task_fn, args=(task,)): task
            for task in tasks
        }

        for future in tqdm(
            future_map,
            desc=task_name,
            unit="struct",
            total=len(tasks),
            leave=False,
        ):
            try:
                result = future.result(timeout=timeout_seconds)
                results.append(result)
            except PebbleTimeoutError:
                future.cancel()
                rank_zero_info(f"| WARNING: A {task_name} task timed out. Skipping.")
                results.append(default_result)
            except Exception as e:
                future.cancel()
                rank_zero_info(f"| WARNING: A {task_name} task failed: {e}")
                results.append(default_result)

    rank_zero_info(f"| {task_name} computation completed.")
    return results
