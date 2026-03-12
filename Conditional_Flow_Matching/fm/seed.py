# fm/seed.py

import numpy as np
import torch


def set_seed():
    """
    Set global seed to 0 for reproducible training (CPU).
    """
    seed = 0

    np.random.seed(seed)          # per split e permutazioni 
    torch.manual_seed(seed)       # per pesi modello e sampling 