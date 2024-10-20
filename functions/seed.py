import numpy as np
import os
import random as r
import torch

'''
This function initialises the seed for the relevant random number generators to ensure reproducibility
'''
def generate(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    r.seed(seed)  # Python random module.
    # In general seed PyTorch operations
    torch.manual_seed(0)
    # If you are using CUDA on 1 GPU, seed it
    torch.cuda.manual_seed(0)
    # If you are using CUDA on more than 1 GPU, seed them all
    torch.cuda.manual_seed_all(0)
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = False
    # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
    torch.backends.cudnn.deterministic = True