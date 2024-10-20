from data_utils.params import Params
import importlib
import torch

def set(config_name="parameters.yaml"):
    params = Params(config_name)
    params = params.yaml_map

    gm = params['gm']
    kx, ky = gm['kx'], gm['ky']

    lambda_value = torch.tensor(gm['lambda_value'])
    h_value= torch.tensor(gm['h_value'], requires_grad=True)

    f_name, g_name = gm['dynamics'], gm['likelihood']
    # Import the module dynamically
    dynamics, likelihood = importlib.import_module('functions.dynamics'), importlib.import_module(
        'functions.likelihood')
    # Get the function object by name
    f, g = getattr(dynamics, f_name), getattr(likelihood, g_name)
    # Generative Process parameters
    gp = params['gp']
    gp_name = gp['name']

    # gm_noise parameters
    gm_white_noise_mu, gm_white_noise_sigma = gm['noise']['gm_white_noise_mu'], gm['noise']['gm_white_noise_sigma']
    gm_noise_type = gm['noise']['gm_noise_type']
    gm_color_noise_kernel_size, gm_color_noise_kernel_sigma = gm['noise']['gm_color_noise_kernel_size'], gm['noise'][
        'gm_color_noise_kernel_sigma']
    # the time step, used in generating hidden states from the generative process AND in estimating y',y'',...
    dt, T = gp['dt'], gp['T']
    opt = params['optimizer']
    # Num of update steps during learning
    state_lr, num_steps = opt['state_lr'], opt['num_steps']

    # Assert commands for making sure the input parameters are set correctly
    #assert schedule[0] >=1 and type(schedule[0]) == int, "The state_estimation_rounds needs to be an integer, equal or greater than 1"

    assert type(kx) == int and type(ky) == int and kx - ky == 1, "kx and ky must be integers such that kx=ky+1"
    assert gp_name in ['lotka'], "The generative process should be: lotka"
    assert f_name in ['pullback', 'trigonometric'], "The model dynamics type can be either of: pullback, trigonometric"
    assert g_name in ['identity'], "The likelihood flow can be either of: identity"
    assert gm_noise_type in ['white',
                          'color'], "The noise type for both observations and multiplicative state noise can be either of: white, color"

    assert gm_color_noise_kernel_size % 2 == 1 and type(
        gm_color_noise_kernel_size) == int, "The kernel size for the colored noise of observations should be an odd integer"


    assert type(num_steps) == int, "The number of steps for solving the ODEs in the generative model should be an integer"

    return (kx, ky, f, g, lambda_value, h_value,gp_name,
            gm_white_noise_mu, gm_white_noise_sigma, gm_noise_type, gm_color_noise_kernel_size,
            gm_color_noise_kernel_sigma,
            dt, T, state_lr, num_steps, f_name, g_name)