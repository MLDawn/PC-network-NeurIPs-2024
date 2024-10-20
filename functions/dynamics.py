import torch


def pullback(x):
    L = len(x)

    A = 0.50*torch.eye(L)
    centre = torch.tensor([1.0, 1.0])

    output = torch.matmul(-A, (x - centre))
    return output

def trigonometric(x):
    '''
    Inputs:
    - x: the state vector
    Output:
    -f(x): Which is a trigonometric function
    '''
    l = []
    for idx in range(len(x)):
        l.append(torch.sin(x[idx]))
    output = torch.stack(l)
    return output