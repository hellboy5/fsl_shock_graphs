import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Locks all random seeds across Python, NumPy, and PyTorch to ensure 
    strict academic reproducibility.
    """
    # 1. Lock the Python hash seed (affects dictionary ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Lock Python's built-in random module
    random.seed(seed)
    
    # 3. Lock NumPy's random number generator
    np.random.seed(seed)
    
    # 4. Lock PyTorch's CPU/GPU generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Ensures safety if using multiple GPUs
    
    # 5. Lock CuDNN backend (Forces deterministic convolutional algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
