"""Dirichlet-Multinomial loss over the Galaxy Zoo 2 decision tree."""

import torch
from torch import Tensor

_ALPHA_MIN = 1e-6


def dirichlet_multinomial_loss(
    alpha: Tensor,
    counts: Tensor,
    question_slices: list[slice],
    class_weights: list[Tensor] | None = None,
) -> Tensor:
    """D-M negative log-likelihood, summed over questions and averaged over galaxies.

    Questions with no votes contribute exactly zero, so no masking is needed.
    `class_weights` is an optional list of [C_q] tensors, one per question; each
    galaxy's term is then scaled by the weight of its majority-vote answer.
    """
    alpha = alpha.clamp(min=_ALPHA_MIN)
    loss = 0.0

    for q, sl in enumerate(question_slices):
        a, k = alpha[:, sl], counts[:, sl]
        a_sum, n = a.sum(dim=1), k.sum(dim=1)
        # The multinomial coefficient log(N! / prod k_i!) is omitted: it does not
        # depend on alpha, so it only shifts the loss by a constant.
        nll = (
            torch.lgamma(a_sum + n)
            - torch.lgamma(a_sum)
            - (torch.lgamma(a + k) - torch.lgamma(a)).sum(dim=1)
        )
        if class_weights is not None:
            nll = nll * class_weights[q].to(nll)[k.argmax(dim=1)]
        loss = loss + nll

    return loss.mean()


def alpha_to_fractions(alpha: Tensor, question_slices: list[slice]) -> Tensor:
    """Expected vote fractions under the predicted Dirichlet: alpha_i / sum(alpha) per question."""
    alpha = alpha.clamp(min=_ALPHA_MIN)
    fracs = torch.empty_like(alpha)
    for sl in question_slices:
        a = alpha[:, sl]
        fracs[:, sl] = a / a.sum(dim=1, keepdim=True)
    return fracs
