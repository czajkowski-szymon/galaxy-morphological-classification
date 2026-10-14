"""Evaluate trained checkpoints on the validation and test sets.

Writes one JSON file per checkpoint with run details (parameter count, training
history) and, for both splits, the vote-weighted RMSE and, for every question,
accuracy, macro F1, per-answer F1 and AUC, and the confusion matrix.
Checkpoints that already have a result file are skipped unless --overwrite is given.

Examples:
    python -m scripts.evaluate models/gz2_*_best.pth
    python -m scripts.evaluate models/imbalance/*_best.pth --arch densenet121
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, data
from src.dataset import GalaxyZoo2Dataset, load_splits
from src.evaluate import classification_metrics, collect_predictions, load_checkpoint
from src.metrics import vote_weighted_rmse
from src.transforms import test_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument(
        "--arch",
        help="timm model name for checkpoints that do not store it; "
        "defaults to <arch> in gz2_<arch>_best.pth",
    )
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "runs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def evaluate_split(model: torch.nn.Module, dl: DataLoader) -> dict:
    pred, frac, count = collect_predictions(model, dl, data.QUESTION_SLICES, config.DEVICE)
    wrmse = vote_weighted_rmse(
        *(torch.from_numpy(a) for a in (pred, frac, count)), data.QUESTION_SLICES
    ).item()
    rows = classification_metrics(pred, frac, count, data.QUESTIONS, data.QUESTION_SLICES)
    questions = [
        {
            "q": r["q"],
            "name": r["name"],
            "answers": r["answers"],
            "n": r["n"],
            "acc": float(r["acc"]),
            "f1_macro": float(r["f1"]),
            "f1": [float(x) for x in r["f1_per_answer"]],
            "auc": r["auc"],
            "confusion": r["confusion"].tolist(),
        }
        for r in rows
    ]
    return {
        "wrmse": wrmse,
        "mean_f1": float(np.mean([q["f1_macro"] for q in questions])),
        "questions": questions,
    }


def main() -> None:
    args = parse_args()
    _, hf_val, hf_test = load_splits()
    loader_kw = dict(batch_size=config.BATCH_SIZE, num_workers=args.num_workers, pin_memory=True)
    loaders = {
        "val": DataLoader(GalaxyZoo2Dataset(hf_val, transform=test_transform), **loader_kw),
        "test": DataLoader(GalaxyZoo2Dataset(hf_test, transform=test_transform), **loader_kw),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.checkpoints:
        run_name = path.stem.removesuffix("_best")
        out = args.out_dir / f"{run_name}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {run_name} (already evaluated)")
            continue

        arch = args.arch or run_name.removeprefix("gz2_")
        try:
            model, ckpt = load_checkpoint(path, arch=arch, device=config.DEVICE)
        except ValueError as e:
            print(f"skip {run_name}: {e}")
            continue
        result = {
            "run_name": run_name,
            "arch": model.config["arch"],
            "checkpoint": str(path),
            "best_epoch": int(ckpt["epoch"]),
            "val_loss": float(ckpt["val_loss"]),
            "weighted_loss": bool(ckpt.get("weighted_loss", False)),
            "minutes": float(sum(ckpt["history"]["epoch_time"]) / 60),
            "n_params": sum(p.numel() for p in model.parameters()),
            "history": ckpt["history"],
        }
        for split, dl in loaders.items():
            result[split] = evaluate_split(model, dl)
        out.write_text(json.dumps(result, indent=1))

        print(
            f"{run_name}: val wRMSE {result['val']['wrmse']:.4f} | "
            f"test wRMSE {result['test']['wrmse']:.4f} | "
            f"test mean F1 {result['test']['mean_f1']:.3f}"
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
