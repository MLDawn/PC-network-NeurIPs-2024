import torch
from torch.autograd.functional import jacobian
import inspect
def compute(gen_mu,f, kx, dx):
    '''
    :param gen_mu: The tensor of generalised posterior expectations (mu_x, mu_x_dot, mu_x_dotdot, etc.)
    :param f: The function describing flow on state dynamics
    :param kx: The number of generalised coordinates in x
    :param dx: The dimension of hidden state x
    :return: The calculated error e_x
    '''
    # # Use a lambda to fix mu_theta
    # jac_f = jacobian(lambda x: f(x, q_theta_mu))
    # # Evaluate the Jacobian of f() at mu_x
    # jacob_f_eval = jac_f(gen_mu[0])

    #Crucial to set create_graph=True: This argument ensures that the operations involved
    # in computing the Jacobian are tracked in the computational graph. As a result, you
    # can backpropagate through jacob_f_eval to calculate the gradients with respect to
    # the inputs (like q_theta_mu) during parameter learning
    #jacob_f_eval = torch.autograd.functional.jacobian(lambda x: f(x, q_theta_mu), gen_mu[0], create_graph=True)

    jacob_f_eval = jacobian(f, gen_mu[0])

    pred = []
    for i in range(kx):
        # i=0 then we are generating predictions for x', which is f(mu_x)
        if i == 0:
            pred = [f(gen_mu[0])]
        # if i > 0, then it means we are generating predictions for x'',x''', etc.
        # This requries the Jacobian of f() evaluated at mu_x
        else:
            pred.append(torch.matmul(jacob_f_eval.T, gen_mu[i]))
    pred = torch.stack(pred)
    #Build the ground-truth vector (it should have a zero vector at the end, which is the regulariser)
    gt = torch.cat((gen_mu[1:], torch.zeros(1,dx)),dim=0)
    e_x = gt - pred
    return e_x
