import torch
from functions import compute_e_y, compute_e_x, compute_vfe
def compute(t, gen_mu, gen_y, f, g, D, gen_pi_x, gen_pi_y, kx, ky, dx, dy):
    # Convert NumPy array to a PyTorch tensor
    gen_mu = torch.from_numpy(gen_mu).reshape((kx,dx))
    # Set requires_grad=True
    gen_mu.requires_grad_(True)
    # compute the error terms using the best solution gen_mu
    gen_e_y, gen_y_hat = compute_e_y.compute(gen_mu, gen_y, g, ky)
    gen_e_x = compute_e_x.compute(gen_mu, f, kx, dx)
    # Calculate the vfe
    vfe = compute_vfe.compute(kx, dx, ky, dy, gen_e_y, gen_pi_y, gen_e_x, gen_pi_x)
    # Compute negative gradients with respect to gen_mu
    negative_grad_vfe = - torch.autograd.grad(outputs=vfe, inputs=gen_mu)[0]
    with torch.no_grad():
        result = torch.matmul(D, gen_mu.data.reshape(kx * dx)).reshape((kx, dx)) + negative_grad_vfe
    return result.reshape((kx*dx))