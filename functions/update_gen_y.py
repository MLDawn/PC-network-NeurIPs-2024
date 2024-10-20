import torch
def update(obs, gen_y, dt):
    '''
    :param obs: The current observation
    :param gen_y: The current generalised observation tensor y
    :param dt: The small time step used to estimate y', y'', etc.
    :return: the updated gen_y where y,y',y'',etc. have been updated
    '''
    # for current observation, update gen_y
    temp = [obs]
    for j in range(len(gen_y)-1):
        # Division by dt is used to estimate higher temporal derivatives of y
        # For instance, change in y divided by dt, is velocity y'
        # And change in y' divided by dt, is the acceleration y''
        next_ = (obs - gen_y[j])/dt
        temp.append(next_)
        obs = next_
    new_gen_y = torch.stack(temp)
    return new_gen_y