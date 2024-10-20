import os
import yaml
from tqdm import tqdm
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20,
                    'lines.linewidth': 2,
                     'xtick.labelsize' : 20,
                     'ytick.labelsize' : 20})

import matplotlib as mpl
mpl.rc('lines', linewidth=3.0)

import torch
from torch.autograd.functional import jacobian
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)


from functions import (seed,add_noise,compute_D, compute_autocov, compute_vfe, gradient,
                       update_gen_y, compute_e_x,compute_e_y, plot_sensation_predictions,
                       plot_vfe, plot_state_estimation, initialise)


# Seed all sources of randomness to ensure reproducibility
seed.generate()

# Take a snapshot of the current yaml file of parameters
# This will be saved as one of the output files.
with open('parameters.yaml', 'r') as file:
    snapshot = yaml.safe_load(file)

# Calls the initialise.py script to set the initial values of the experiment parameters, as specified in parameters.yaml file
(kx, ky, f, g, lambda_value, h_value,gp_name, gm_white_noise_mu, gm_white_noise_sigma, gm_noise_type, gm_color_noise_kernel_size,
gm_color_noise_kernel_sigma, dt, T, state_lr, num_steps, f_name, g_name) = initialise.set("parameters.yaml")

# Load the selected generative process, determined by gp_name. In this script, we only have Lotka-Volterra as the GP.
# GP is now a .py function that contains the Ordinaty Differential Equations (ODEs) describing the GP.
GP = __import__('.'.join(['functions', gp_name]), fromlist=['object'])

# Solve the ODEs in GP by integrating them over a time span T and with step size dt.
# x now holds the true external states of the world (i.e., true trajectories of the world)
x = GP.build(dt, T=T)

# Add noise to x, to build the noisy observations
y = add_noise.add(x, gm_white_noise_mu, gm_white_noise_sigma, gm_noise_type,
                        gm_color_noise_kernel_size, gm_color_noise_kernel_sigma)

# The dimensions of x and y, which are both equal to 2 for a Lotka-Volterra process.
dx, dy = x.shape[1], y.shape[1]

# Hidden state vector (Since in our generative model kx=2, this holds the initial expected
# value for the position and velocity of the external world: mu_x, mu_xdot)
# Note that if we are stepping into the world generalised coordinates of motion where kx>2,
# then gen_mu will hold initial values for higher temporal derivative expectations such as
# acceleration, jerk, etc. as well.
gen_mu = torch.stack([torch.rand(dx) for _ in range(kx)]).requires_grad_(True)

# Create a placeholders for the observations.
# Note: since ky=1, this will only hold a place-holder for the actual observation, however, if ky>1, then
# we are in the world of generalised coordinates of motion where gen_y will also contain the temporal derivatives of y,
# such as velocity, acceleration, jerk, etc. (i.e., y',y'', y''',...).
gen_y = torch.stack([torch.zeros(len(y[0])) for _ in range(ky)])

# Construct the generalised precisions for x and y, which for now will remain to be fixed.
# Note: gen_pi_y and gen_pi_x express the correlation between random fluctuations along different temporal derivatives
# For the purpose of this script, since kx=2, ky=1, you can safely ignore their construction. However, if kx>2 and ky>1,
# Then these will become essential.
p_y, p_x = torch.eye(dy), torch.eye(dx)
s_y, s_x = torch.inverse(compute_autocov.compute(ky, h_value, lambda_value)), torch.inverse(compute_autocov.compute(kx, h_value, lambda_value))
gen_pi_y, gen_pi_x = torch.kron(s_y, torch.inverse(p_y)), torch.kron(s_x, torch.inverse(p_x))

# Compute the block-matrix derivative operator D
D = compute_D.compute(kx, dx)

# Initialise free action as 0
free_action = 0

# These will hold, the estimates for x, the actual sensations, and predicted sensations, respectively, for plotting purposes.
gen_x_estimates, gen_sensations, gen_predictions = [], [], []

# This will hold the values of variational free energy at every time step
VFE = []

# The inference loop starts, which goes over each sensation/observation in y.
for i, obs in tqdm(enumerate(y)):
    # Update gen_y given current observation obs. This serves as the ground truth for calculating the error term e_y
    # Given any new observation, gen_y needs to be updated. If ky>1, then not only the observation in gen_y needs to be
    # updated but also the estimates of velocity y', acceleration y'', jerk y''', etc. In our script ky=1, though.
    gen_y = update_gen_y.update(obs, gen_y, dt)

    # Update the estimated generalised expected posteriors
    # This is where the ODE describing the belief update (See: gradient.py) needs to be integrated.
    # The conversion of gen_mu to numpy is necessary as solve_ivp cannot work with torch tensors.
    # NOTE: The 'y0=gen_mu.ravel().detach().numpy()' line is needed, since:
    # 1) solve_ivp cannot work with torch tensors so need conversion to numpy
    # 2) ravel() is needed to flatten the tensor
    # In the gradient.compute() function, we will have to convert this back to torch.tensor(requires_grad=True)
    # so we can use autograd to calculate the gradients
    result = solve_ivp(gradient.compute, args=(gen_y, f, g, D, gen_pi_x, gen_pi_y, kx, ky, dx, dy),
                       t_span=(0, state_lr*num_steps), y0=gen_mu.ravel().detach().numpy(), method='RK45')

    # Pick the final value of the result trajectory as the final value, result.y[:, -1], of gen_mu and convert it to torch tensor
    gen_mu = torch.from_numpy(result.y[:, -1]).reshape((kx,dx))

    # Final evaluation of vfe
    gen_e_y, gen_y_hat = compute_e_y.compute(gen_mu, gen_y, g, ky)
    gen_e_x = compute_e_x.compute(gen_mu, f, kx, dx)

    # Calculate the new value of vfe
    vfe = compute_vfe.compute(kx, dx, ky, dy, gen_e_y, gen_pi_y, gen_e_x, gen_pi_x)
    VFE.append(vfe)

    # Accumulate vfe into free action
    free_action += vfe

    if i % 10 == 0:
        print('\nFree Action=%.2f, VFE: %.2f' % (free_action, vfe))

    # store the current gen_y, gen_y_hat and gen_mu for plotting purposes
    gen_sensations.append(gen_y.data)
    gen_predictions.append(gen_y_hat.data)
    gen_x_estimates.append(gen_mu.data)

# Convert to numpy for the plots and analysis
VFE = np.stack([t.detach().cpu().numpy() for t in VFE])
gen_sensations = np.stack([t.detach().cpu().numpy() for t in gen_sensations])
gen_x_estimates = np.stack([t.detach().cpu().numpy() for t in gen_x_estimates])
gen_predictions = np.stack([t.detach().cpu().numpy() for t in gen_predictions])

x = np.stack([t.detach().cpu().numpy() for t in x])
y = np.stack([t.detach().cpu().numpy() for t in y])


# Evaluate the performance through MSE and Free Action
# gen_x_estimates[:,0,:] holds all the estimates for x (i.e., mu_x) ignoring mu_x_dot, mu_x_dotdot, etc.
x_estimates = gen_x_estimates[:, 0, :]
mse = np.mean((x - x_estimates) ** 2)
print('Final Free Action=%.2f , Final MSE=%.2f' % (free_action, mse))

# Get the current date and time to be used in naming the result folder
now = datetime.now()
# Format the date and time
# Example format: "2023-06-20_14-30-00"
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
dir_name= '(%d, %d, %s, %s, %s, %.2f, %.2f)' % (kx, ky, gp_name, f_name, g_name, free_action.item(), mse)
# Define the directory name using os.path.join to be OS agnostic
directory_name = os.path.join("results", dir_name)
# Create the subdirectory
os.makedirs(directory_name, exist_ok=True)
# Plot and save vfe and free action
plot_vfe.plot(VFE,directory_name)
# Plot and save the predicted sensations against true sensations
plot_sensation_predictions.plot(ky, dy, gen_sensations, gen_predictions,directory_name)
#  Plot and save the estimated hidden states
plot_state_estimation.plot(kx, x, y, dx, dy, dt, gen_x_estimates,directory_name)

# Add the free action and mse loss to the current snapshot as key:value pairs
snapshot['fa'] = float(free_action)
snapshot['mse'] = float(mse)

# Save the snapshot
with open(os.path.join(directory_name, 'snapshot.yaml'), 'w') as file:
    yaml.safe_dump(snapshot, file)
