"""Class reweighting and oversampling for imbalanced answers deep in the decision tree.

Both interventions use the majority-vote answer of each question, the same label
that accuracy and F1 are computed on in `src.evaluate`.
"""

import numpy as np
import torch
from torch import Tensor

from src import data as schema


def _count_matrix(hf_split) -> np.ndarray:
    df = hf_split.select_columns(schema.campaign_cols()).to_pandas()
    return np.nan_to_num(df.to_numpy(dtype=np.float64))


def question_labels(hf_split) -> np.ndarray:
    """Majority-vote answer per question, [n_galaxies, 10]; -1 where the question got no votes."""
    counts = _count_matrix(hf_split)
    labels = np.full((len(counts), len(schema.QUESTION_SLICES)), -1, dtype=np.int64)
    for q, sl in enumerate(schema.QUESTION_SLICES):
        k = counts[:, sl]
        active = k.sum(axis=1) > 0
        labels[active, q] = k[active].argmax(axis=1)
    return labels


def class_priors(labels: np.ndarray) -> list[np.ndarray]:
    """Frequency of each answer within its question, over galaxies that reached it."""
    priors = []
    for q, sl in enumerate(schema.QUESTION_SLICES):
        lab = labels[:, q]
        lab = lab[lab >= 0]
        cnt = np.bincount(lab, minlength=sl.stop - sl.start).astype(np.float64)
        # An absent class gets a count of 1, so its weight stays finite.
        priors.append(np.maximum(cnt, 1.0) / max(len(lab), 1))
    return priors


def class_balanced_weights(labels: np.ndarray) -> list[Tensor]:
    """Per-question class weights 1 / (C * prior) for `dirichlet_multinomial_loss`.

    Every class gets the same total weight, and the mean weight over galaxies is 1,
    so the loss scale (and the effective learning rate) stays unchanged.
    """
    return [torch.tensor(1.0 / (len(pi) * pi), dtype=torch.float32) for pi in class_priors(labels)]


def deep_sample_weights(
    labels: np.ndarray,
    questions: tuple[int, ...] = (7, 8, 9),
    clip: float = 10.0,
) -> Tensor:
    """Per-galaxy weights for `WeightedRandomSampler` that oversample rare deep answers.

    A galaxy gets the largest 1 / prior of its answers to `questions`
    (zero-based, Q8-Q10 by default), or 1 if it reached none of them.
    `clip` caps the raw weight so a few hundred rare galaxies cannot dominate an
    epoch; the weights are then normalised to mean 1.
    """
    priors = class_priors(labels)

    w = np.ones(len(labels), dtype=np.float64)
    for q in questions:
        lab = labels[:, q]
        active = lab >= 0
        w[active] = np.maximum(w[active], 1.0 / priors[q][lab[active]])

    w = np.minimum(w, clip)
    return torch.tensor(w / w.mean(), dtype=torch.double)
