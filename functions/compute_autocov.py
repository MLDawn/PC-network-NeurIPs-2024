import sympy as sp
import torch


def gaussian_function(h_value, lambda_value):
    return torch.exp(-0.5 * lambda_value * h_value ** 2)

def nth_derivative_gaussian_torch(h_value, lambda_value,N):

    # Calculate the Gaussian function
    y = gaussian_function(h_value, lambda_value)

    # Compute the nth derivative w.r.t the mean
    for _ in range(N):
        y = torch.autograd.grad(outputs=y, inputs=h_value, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return y


def compute(k, h_value, lambda_value):
    '''
    :param k: The number of generalised coordinates for either X or Y, that is, kx or ky (scalar)
    :param h_value: Determines the centre of the Gaussian kernel for computing the auto-covariance matrix (scalar)
    :param lambda_value: Determines the variance of the Gaussian kernel used for calculating the auto-covariance matrix (scalar)
    :return: The auto-covariance matrix (k*dx * k*dx)
    '''
    autocov_array = torch.zeros((k, k))
    for n in range(k):
        for m in range(k):
            # compute the current (n+m)th derivative
            kernel_derivative = nth_derivative_gaussian_torch(h_value, lambda_value, n + m)
            # Compute the final value by multiplying the (-1)**n and place this in row n and column m of cov_array
            autocov_array[n, m] = (-1) ** n * kernel_derivative
    return autocov_array