import torch
def compute(kx, dx, ky, dy, gen_e_y, gen_pi_y, gen_e_x, gen_pi_x):
    '''
    :param kx: The number of genenralised coordinates in x
    :param dx: The number of dimensions in x
    :param ky: The number of genenralised coordinates in y
    :param dy: The number of dimensions in y
    :param gen_e_y: The calculated generalised error gen_e_y (ky*dy) in observations
    :param gen_pi_y: The generalised precision of observations (ky*dy * ky*dy)
    :param gen_e_x: The generalised error in hidden state (kx*dx)
    :param gen_pi_x: The generalised precision of hidden states (kx*dx * kx*dx)
    :return: The variational free energy (a scalar)
    '''

    # compute the quadratic terms in vfe
    quad_y = torch.matmul(torch.matmul(gen_e_y.reshape(ky*dy).T, gen_pi_y),gen_e_y.reshape(ky*dy))
    quad_x = torch.matmul(torch.matmul(gen_e_x.reshape(kx*dx).T, gen_pi_x),gen_e_x.reshape(kx*dx))

    # confute the final vfe
    vfe = 0.50*(quad_y + quad_x)

    return vfe