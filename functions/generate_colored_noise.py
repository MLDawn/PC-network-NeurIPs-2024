import numpy as np
from scipy.signal import convolve
from functions import generate_white_noise
import torch

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()

def generate(size, white_noise_mu, white_noise_sigma, conv_kernel_size=51, conv_kernel_sigma=2.0):
    # white_noise has shape (dim, num_samples) e.g., (2,1000)
    white_noise = generate_white_noise.generate(size, white_noise_mu, white_noise_sigma)
    # Step 2: Create a Gaussian kernel
    kernel = gaussian_kernel(conv_kernel_size, conv_kernel_sigma)
    # Step 3: Convolve the white noise with the Gaussian kernel to generate colored noise
    colored_noise = []
    for dim in range(white_noise.shape[0]):
        colored_noise.append(torch.from_numpy(convolve(white_noise[dim], kernel, mode='same')))
    colored_noise = torch.stack(colored_noise)
    return colored_noise
