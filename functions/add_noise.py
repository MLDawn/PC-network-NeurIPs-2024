import torch
from functions import generate_white_noise, generate_colored_noise
import matplotlib.pyplot as plt
def add(x, mu, sigma, noise_type, conv_kernel_size=51, conv_kernel_sigma=2.0):
    '''
    :param x: The hidden state
    :type x: Torch tensor (N x dx) where N is the number of data points and dx is the dimension of each point in x
    :param mu: The mean of the white noise Gaussian
    :type mu: float
    :param sigma: The standard deviation of the white noise Gaussian
    :type sigma: float
    :param noise_type: 'color' or 'white'
    :type noise_type: string
    :param conv_kernel_size: The kernel size of the convolution kernel for creating color noise
    :type conv_kernel_size: integer
    :param conv_kernel_sigma: The standard deviation of the convolution kernel for creating color noise
    :type conv_kernel_sigma: float
    :return: Noisy observations (N x dx) where N is the number of data points and dx is the dimension of each point in x
    :rtype: Torch tensor
    '''
    # Holds the final noisy observations
    noisy_y = []
    # Holds an independent random noise tensor vector, per each dimension of x
    noise = torch.tensor([])
    if noise_type == 'white':
        # The white noise values independently generated per each dimension of x
        noise = generate_white_noise.generate(x.shape, mu, sigma)
    elif noise_type == 'color':
        # The color noise values independently generated per each dimension of x
        noise = generate_colored_noise.generate(x.shape, mu, sigma, conv_kernel_size, conv_kernel_sigma)
    # noise is now of the shape (dim, num_samples) whereas x is reversed, that is, (1000,2)
    # So, we will swap the dimensions of the noise just to make the code more straightforward
    noise = torch.transpose(noise, 0, 1)

    for dim in range(x.shape[1]):
        # Add each noise sequence to its corresponding dimension of x. This ensures independent noise per channel
        noisy_y.append(x[:, dim] + noise[:, dim])

    noisy_y = torch.transpose(torch.stack(noisy_y), 0, 1)

    return noisy_y