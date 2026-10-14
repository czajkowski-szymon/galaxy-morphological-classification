"""Train one model on Galaxy Zoo 2 with the Dirichlet-Multinomial loss.

Examples:
    python -m scripts.train --arch densenet121.tv_in1k
    python -m scripts.train --arch densenet121.tv_in1k --seed 101 --class-weights
    python -m scripts.train --arch densenet121.tv_in1k --seed 101 --oversample
"""

import argparse
import json
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import config, data, sampling
from src.dataset import GalaxyZoo2Dataset, load_splits
from src.train import build_model, fit, make_optimizer
from src.transforms import test_transform, train_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", required=True, help="timm model name")
    p.add_argument("--run-name", help="checkpoint name, defaults to gz2_<arch>")
    p.add_argument("--out-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--layer-decay", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    p.add_argument("--lr-patience", type=int, default=5, help="ReduceLROnPlateau patience")
    p.add_argument("--num-workers", type=int, default=4)

    imbalance = p.add_mutually_exclusive_group()
    imbalance.add_argument(
        "--class-weights",
        action="store_true",
        help="reweight the loss by inverse class frequency within each question",
    )
    imbalance.add_argument(
        "--oversample",
        action="store_true",
        help="oversample galaxies with rare answers to Q8-Q10",
    )
    p.add_argument("--oversample-clip", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_name = args.run_name or f"gz2_{args.arch.split('.')[0]}"

    hf_train, hf_val, _ = load_splits()
    ds_train = GalaxyZoo2Dataset(hf_train, transform=train_transform)
    ds_val = GalaxyZoo2Dataset(hf_val, transform=test_transform)

    class_weights, sampler = None, None
    if args.class_weights or args.oversample:
        labels = sampling.question_labels(hf_train)
    if args.class_weights:
        class_weights = sampling.class_balanced_weights(labels)
    if args.oversample:
        weights = sampling.deep_sample_weights(labels, clip=args.oversample_clip)
        # Same epoch length as without oversampling, so runs cost the same.
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(ds_train),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loader_kw = dict(batch_size=config.BATCH_SIZE, num_workers=args.num_workers, pin_memory=True)
    dl_train = DataLoader(ds_train, shuffle=sampler is None, sampler=sampler, **loader_kw)
    dl_val = DataLoader(ds_val, shuffle=False, **loader_kw)

    model = build_model(
        args.arch,
        data.N_OUTPUTS,
        pretrained=True,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
    )
    optimizer = make_optimizer(
        model,
        head_lr=args.head_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    history = fit(
        model,
        dl_train,
        dl_val,
        args.epochs,
        optimizer,
        scheduler,
        data.QUESTION_SLICES,
        config.DEVICE,
        accum_steps=args.accum_steps,
        patience=args.patience,
        save_dir=args.out_dir,
        run_name=run_name,
        class_weights=class_weights,
    )

    # Written only after fit returns, so it also marks the run as finished.
    best = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
    summary = {
        "run_name": run_name,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "epochs": len(history["val_loss"]),
        "best_epoch": best + 1,
        "val_loss": history["val_loss"][best],
        "val_wrmse": history["val_wrmse"][best],
        "minutes": sum(history["epoch_time"]) / 60,
        # The checkpoint's history stops at the best epoch; this one is complete.
        "history": history,
    }
    (args.out_dir / f"{run_name}_run.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
