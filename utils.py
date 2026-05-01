import random
from pathlib import Path

import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

