import torch

def compute(kx,dx):
    '''
    :param kx: Number of generalised coordinates in x
    :param dx: Dimension of x
    :return: D torch tensor of shape (kx*dx * kx*dx)
    '''

    D = [torch.zeros((dx,dx))]*(kx*kx)
    D[1] = torch.eye(dx)
    i=1
    while(True):
        i += kx+1
        try:
            D[i] = torch.eye(dx)
        except:
            break
    D = torch.stack(D)
    temp = []
    i=0
    while(i<kx*kx):
        for j in range(dx):
            temp.append(D[i:i+kx][:,j].flatten())
        i += kx
    D = torch.stack(temp)
    return D