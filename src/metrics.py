"""RMSE metrics between predicted and observed vote fractions."""

import torch
from torch import Tensor


def masked_rmse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """RMSE over all answers of questions that received votes, each answer weighted equally."""
    diff = pred[mask] - target[mask]
    return diff.pow(2).mean().sqrt()


def per_column_rmse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Masked RMSE computed separately for each answer; 0 for answers with no votes."""
    m = mask.float()
    sse = ((pred - target).pow(2) * m).sum(dim=0)
    return (sse / m.sum(dim=0).clamp(min=1)).sqrt()


def vote_weights(counts: Tensor, question_slices: list[slice]) -> Tensor:
    """Number of votes N_q of each answer's question, broadcast to all answers."""
    w = torch.empty_like(counts)
    for sl in question_slices:
        w[:, sl] = counts[:, sl].sum(dim=1, keepdim=True)
    return w


def vote_weighted_rmse(
    pred: Tensor,
    target: Tensor,
    counts: Tensor,
    question_slices: list[slice],
) -> Tensor:
    """RMSE with each squared error weighted by the number of votes on its question."""
    w = vote_weights(counts, question_slices)
    return ((w * (pred - target).pow(2)).sum() / w.sum().clamp(min=1e-8)).sqrt()
