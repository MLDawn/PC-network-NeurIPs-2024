import torch
from functions import generate_white_noise, generate_colored_noise

def build(dt, T, a=0.7, b=0.5, c=0.3, e=0.2):
    '''
    :param dt: Used for solving the ODEs in the lotka-volterra generative process
    :param T: Total amount of time, which divided by dt, will give the total number of steps taken for solving the ODEs
    :return: The true hidden state solutions torch tensor x
    '''
    N = int(T / dt)
    # Assign the initial value
    x_1 = torch.zeros(N)
    x_2 = torch.zeros(N)
    x_1[0], x_2[0] = 1.0, 0.5
    # The hidden state list
    x = [torch.tensor([x_1[0], x_2[0]])]
    for i in range(1, N):
        # calc new values for x
        x_1[i] = x_1[i - 1] + (a * x_1[i - 1] - b * x_1[i - 1] * x_2[i - 1]) * dt
        x_2[i] = x_2[i - 1] + (-c * x_2[i - 1] + e * x_1[i - 1] * x_2[i - 1]) * dt
        # store new values in list
        x.append(torch.tensor([x_1[i], x_2[i]]))
    x = torch.stack(x)
    return x