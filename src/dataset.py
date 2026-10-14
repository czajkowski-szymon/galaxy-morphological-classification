from collections.abc import Callable

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import Dataset

from src import config
from src import data as schema


def load_splits():
    """Train, validation and test HF splits; validation is a fixed 20% of the HF train set.

    The dataset is downloaded from HuggingFace on first use and cached on disk.
    """
    if config.DATA_PATH_GZ2.exists():
        ds = load_from_disk(str(config.DATA_PATH_GZ2))
    else:
        ds = load_dataset(config.HF_DATASET_GZ2)
        ds.save_to_disk(str(config.DATA_PATH_GZ2))
    schema.validate_columns(ds["train"].column_names)
    split = ds["train"].train_test_split(test_size=config.VAL_SIZE, seed=config.SEED)
    return split["train"], split["test"], ds["test"]


def read_columns(row: dict, cols: list[str]) -> np.ndarray:
    """Read the given columns of a dataset row as float32, with missing values set to 0."""
    values = np.array([row[c] for c in cols], dtype=np.float32)  # None -> NaN
    return np.nan_to_num(values, nan=0.0)


class GalaxyZoo2Dataset(Dataset):
    """Galaxy Zoo 2 split from HuggingFace wrapped as a PyTorch dataset.

    Each item is a tuple (image, counts, fractions, mask):
    - counts: vote count per answer, float32 [30]
    - fractions: vote fraction per answer, float32 [30]
    - mask: bool [30], True for answers of questions that received at least one vote
    """

    def __init__(self, hf_split, transform: Callable | None = None) -> None:
        self.split = hf_split
        self.transform = transform

        self._count_cols = schema.campaign_cols(fraction=False)
        self._frac_cols = schema.campaign_cols(fraction=True)
        self._total_vote_cols = schema.total_vote_cols()

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.split[idx]

        image = row["image"]
        if self.transform is not None:
            image = self.transform(image)

        counts = torch.from_numpy(read_columns(row, self._count_cols))
        fractions = torch.from_numpy(read_columns(row, self._frac_cols))

        total_votes = read_columns(row, self._total_vote_cols)
        mask = torch.zeros(schema.N_OUTPUTS, dtype=torch.bool)
        for sl, n in zip(schema.QUESTION_SLICES, total_votes, strict=True):
            if n > 0:
                mask[sl] = True

        return image, counts, fractions, mask
