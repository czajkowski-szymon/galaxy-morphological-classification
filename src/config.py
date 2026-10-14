from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH_GZ2 = REPO_ROOT / "data" / "galaxy_zoo_2"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

HF_DATASET_GZ2 = "mwalmsley/gz2"

SEED = 42
VAL_SIZE = 0.2
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
