"""Model construction, optimizer and training loop with early stopping."""

import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss import alpha_to_fractions, dirichlet_multinomial_loss
from src.metrics import vote_weights


class GalaxyModel(nn.Module):
    """Pretrained backbone with an MLP head producing positive Dirichlet concentrations."""

    def __init__(
        self,
        backbone: nn.Module,
        n_outputs: int,
        head_hidden: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = {"head_hidden": head_hidden, "dropout": dropout}
        self.head = nn.Sequential(
            nn.Linear(backbone.num_features, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.head(self.backbone(x)))


def build_model(
    arch: str,
    n_outputs: int,
    pretrained: bool = True,
    head_hidden: int = 1024,
    dropout: float = 0.3,
) -> GalaxyModel:
    backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
    model = GalaxyModel(backbone, n_outputs, head_hidden=head_hidden, dropout=dropout)
    model.config["arch"] = arch  # saved in checkpoints so they can be rebuilt
    return model


def make_optimizer(
    model: GalaxyModel,
    head_lr: float = 1e-3,
    layer_decay: float = 0.3,
    weight_decay: float = 0.2,
) -> torch.optim.Optimizer:
    """AdamW with discriminative learning rates in three groups.

    The head trains at `head_lr`, the last third of the backbone's parameter tensors
    at `head_lr * layer_decay` and the first two thirds at `head_lr * layer_decay**2`.
    The split counts parameter tensors in definition order, not layers, so it is
    only an approximation of depth.
    """
    backbone_params = [p for _, p in model.backbone.named_parameters()]
    cut = len(backbone_params) * 2 // 3

    return torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": head_lr},
            {"params": backbone_params[cut:], "lr": head_lr * layer_decay},
            {"params": backbone_params[:cut], "lr": head_lr * layer_decay**2},
        ],
        weight_decay=weight_decay,
    )


_BAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar:30}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def _run_epoch(
    model: nn.Module,
    dl: DataLoader,
    question_slices: list[slice],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accum_steps: int = 4,
    desc: str = "",
    class_weights: list[Tensor] | None = None,
) -> tuple[float, float, float]:
    """One pass over `dl`; trains if an optimizer is given. Returns (loss, rmse, wrmse)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    sum_loss = 0.0
    sse, cnt = 0.0, 0.0
    wsse, wsum = 0.0, 0.0

    if train_mode:
        optimizer.zero_grad()

    pbar = tqdm(dl, desc=desc, leave=False, bar_format=_BAR_FMT, dynamic_ncols=True)

    with torch.set_grad_enabled(train_mode):
        for step, (imgs, counts, fracs, mask) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)
            fracs = fracs.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Mixed precision for the backbone, float32 for lgamma in the D-M loss.
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                alpha = model(imgs)
            alpha = alpha.float()
            loss = dirichlet_multinomial_loss(
                alpha, counts, question_slices, class_weights=class_weights
            )

            if train_mode:
                scaler.scale(loss / accum_steps).backward()
                if step % accum_steps == 0 or step == len(dl):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            with torch.no_grad():
                pred = alpha_to_fractions(alpha, question_slices)
                d2 = (pred - fracs).pow(2)
                m = mask.float()
                w = vote_weights(counts, question_slices)

                sse += (d2 * m).sum().item()
                cnt += m.sum().item()
                wsse += (d2 * w).sum().item()
                wsum += w.sum().item()

            sum_loss += loss.item()

            pbar.set_postfix(
                loss=f"{sum_loss / step:.4f}",
                rmse=f"{(sse / max(cnt, 1)) ** 0.5:.4f}",
                wrmse=f"{(wsse / max(wsum, 1)) ** 0.5:.4f}",
            )

    return (
        sum_loss / len(dl),
        (sse / max(cnt, 1)) ** 0.5,
        (wsse / max(wsum, 1)) ** 0.5,
    )


def fit(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    question_slices: list[slice],
    device: torch.device,
    accum_steps: int = 4,
    patience: int = 8,
    save_dir: Path | None = None,
    run_name: str = "model",
    class_weights: list[Tensor] | None = None,
) -> dict[str, list]:
    """Train with early stopping on validation loss and restore the best weights.

    `class_weights` apply to the training loss only; validation loss stays unweighted
    so early stopping and model selection are comparable across runs.
    """
    model.to(device)
    if class_weights is not None:
        class_weights = [w.to(device) for w in class_weights]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    history: dict[str, list] = {
        k: []
        for k in (
            "train_loss",
            "train_rmse",
            "train_wrmse",
            "val_loss",
            "val_rmse",
            "val_wrmse",
            "epoch_time",
        )
    }

    for epoch in range(1, n_epochs + 1):
        lrs = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        print(f"Epoch {epoch}/{n_epochs}  lr=[{lrs}]")

        t0 = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            dl_train,
            question_slices,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accum_steps=accum_steps,
            desc="  train",
            class_weights=class_weights,
        )
        val_metrics = _run_epoch(model, dl_val, question_slices, device, desc="    val")
        elapsed = time.perf_counter() - t0
        val_loss = val_metrics[0]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        for split, values in (("train", train_metrics), ("val", val_metrics)):
            for name, value in zip(("loss", "rmse", "wrmse"), values, strict=True):
                history[f"{split}_{name}"].append(value)
        history["epoch_time"].append(elapsed)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": best_state,
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "history": history,
                        "elapsed_total": sum(history["epoch_time"]),
                        "model_cfg": getattr(model, "config", {}),
                        "weighted_loss": class_weights is not None,
                    },
                    save_dir / f"{run_name}_best.pth",
                )
            status = "  ✓ saved"
        else:
            epochs_without_improvement += 1
            status = f"  ({epochs_without_improvement}/{patience})"

        print(
            f"  {elapsed:.0f}s | "
            f"train {'/'.join(f'{v:.4f}' for v in train_metrics)} | "
            f"val {'/'.join(f'{v:.4f}' for v in val_metrics)}"
            f"{status}\n"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement in val_loss for {patience} epochs.\n")
            break

    total = sum(history["epoch_time"])
    print(f"Training time: {total / 60:.1f} min ({len(history['epoch_time'])} epochs)")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})\n")

    return history
