"""Model inference and per-question classification metrics."""

from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import data
from src.loss import alpha_to_fractions
from src.train import GalaxyModel, build_model


def load_checkpoint(
    path: Path, arch: str | None = None, device: torch.device | str = "cpu"
) -> tuple[GalaxyModel, dict]:
    """Rebuild a model from a checkpoint saved by `fit`, in eval mode.

    `arch` is used only for older checkpoints that do not store their architecture.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_cfg"]
    arch = cfg.get("arch") or arch
    if arch is None:
        raise ValueError(f"{path} does not store its architecture; pass arch")
    if not timm.is_model(arch):
        raise ValueError(f"{path}: {arch!r} is not a timm model")
    model = build_model(
        arch,
        data.N_OUTPUTS,
        pretrained=False,
        head_hidden=cfg["head_hidden"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def collect_predictions(
    model: torch.nn.Module,
    dl: DataLoader,
    question_slices: list[slice],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader; return predicted fractions, true fractions and vote counts."""
    all_pred, all_frac, all_count = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, counts, fracs, _ in tqdm(dl, desc="inference", leave=False):
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                alpha = model(imgs).float()
            pred = alpha_to_fractions(alpha, question_slices).cpu()
            all_pred.append(pred)
            all_frac.append(fracs)
            all_count.append(counts)

    return (
        torch.cat(all_pred).numpy(),
        torch.cat(all_frac).numpy(),
        torch.cat(all_count).numpy(),
    )


def _one_vs_rest_auc(y_true: np.ndarray, scores: np.ndarray) -> list[float | None]:
    """AUC of each answer's predicted fraction for telling its galaxies from the rest.

    None for an answer that is never, or always, the majority vote.
    """
    out = []
    for i in range(scores.shape[1]):
        is_i = y_true == i
        out.append(float(roc_auc_score(is_i, scores[:, i])) if 0 < is_i.sum() < len(is_i) else None)
    return out


def classification_metrics(
    pred_all: np.ndarray,
    frac_all: np.ndarray,
    count_all: np.ndarray,
    questions: list[tuple[str, list[str]]],
    question_slices: list[slice],
) -> list[dict]:
    """Per-question classification metrics over galaxies that reached the question.

    The true label is the answer with the most votes, the predicted label the answer
    with the highest predicted fraction. Ties go to the first answer in the list.
    Returns one dict per question with accuracy, macro precision/recall/F1, per-answer
    F1 and AUC, the confusion matrix, and y_true/y_pred.
    """
    rows = []
    for q, ((qname, answers), sl) in enumerate(
        zip(questions, question_slices, strict=True), start=1
    ):
        active = count_all[:, sl].sum(axis=1) > 0
        scores = pred_all[active, sl]
        y_pred = scores.argmax(axis=1)
        y_true = frac_all[active, sl].argmax(axis=1)
        labels = list(range(len(answers)))

        acc = accuracy_score(y_true, y_pred)
        # Macro average for every question, including two-answer ones, so rare answers
        # count as much as common ones and all questions report the same metric.
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        rows.append(
            dict(
                q=q,
                name=qname,
                n=int(active.sum()),
                acc=acc,
                prec=prec,
                rec=rec,
                f1=f1,
                f1_per_answer=f1_score(
                    y_true, y_pred, average=None, labels=labels, zero_division=0
                ),
                auc=_one_vs_rest_auc(y_true, scores),
                confusion=confusion_matrix(y_true, y_pred, labels=labels),
                n_cls=len(answers),
                answers=answers,
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    return rows
