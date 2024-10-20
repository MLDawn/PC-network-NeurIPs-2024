import numpy as np
import torch

def generate(size, mu, sigma):
    num_samples, dim = size[0], size[1]
    white_noise = []
    for _ in range(dim):
        # num_samples number of noisy samples generated PER dimension, independently.
        white_noise.append(torch.from_numpy(np.random.normal(size=num_samples, loc=mu, scale=sigma)))

    white_noise = torch.stack(white_noise)
    return white_noise