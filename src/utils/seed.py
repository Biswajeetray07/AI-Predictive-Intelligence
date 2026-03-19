"""
Centralized seed utility for deterministic reproducibility.
Import this instead of duplicating set_seed() across training scripts.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set seeds for full reproducibility across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
