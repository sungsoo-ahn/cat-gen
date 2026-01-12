import torch


def stratify_loss_by_time(
    batch_t: torch.Tensor, batch_loss: torch.Tensor, loss_name: str
) -> dict[str, float]:
    """Stratify losses into time bins for analysis.

    Args:
        batch_t: (B,) timesteps
        batch_loss: (B,) losses
        loss_name: name prefix for the losses

    Returns:
        Dictionary with stratified losses
    """
    time_bins = [
        (0.0, 0.25, "t_0_025"),
        (0.25, 0.5, "t_025_05"),
        (0.5, 0.75, "t_05_075"),
        (0.75, 1.0, "t_075_1"),
    ]

    results = {}
    for low, high, bin_name in time_bins:
        mask = (batch_t >= low) & (batch_t < high)
        if mask.sum() > 0:
            results[f"{loss_name}_{bin_name}"] = batch_loss[mask].mean().item()
        # Skip empty bins instead of logging NaN (avoids WandB NaN warnings)

    return results
