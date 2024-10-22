import torch
from torch.autograd.functional import jacobian
def compute(gen_mu, gen_y, g, ky):
    '''
    :param gen_mu: The vector of generalised posterior expectations (mu_x, mu_x_dot, mu_x_dotdot, etc)
    :param gen_y: The vector of generalised observations (y,y',y'', etc.)
    :param g: The function describing the observation likelihood
    :param ky: The number of generalised coordinates in y
    :return: The calculated generalised error gen_e_y (ky*dy) in observations and the generalised predictions for the generalised observations
    '''
    # Evaluate the Jacobian of g() at mu_x
    jacob_g_eval = jacobian(g, gen_mu[0])
    pred = []
    for i in range(ky):
        # i=0 then we are generating predictions for y, which is g(mu_x)
        if i == 0:
            pred = [g(gen_mu[0])]
        # if i > 0, then it means we are generating predictions for y',y'', etc.
        # This requires the Jacobian of g() evaluated at mu_x
        else:
            pred.append(torch.matmul(jacob_g_eval,gen_mu[i]))

    pred = torch.stack(pred)
    # calculate the prediction error
    gen_e_y = gen_y - pred
    return gen_e_y, pred