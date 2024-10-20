import matplotlib.pyplot as plt
import numpy as np
import os
def plot(kx, x, y, dx, dy, dt, gen_x_estimates,directory,legend_size=22, tick_size = 22):
    colors = ['#ff7f00', '#984ea3', '#377eb8']
    true_gen_x = [x]
    for i in range(kx - 1):
        true_gen_x.append(np.diff(true_gen_x[-1], axis=0, prepend=0) / dt)
    true_gen_x = np.array(true_gen_x)

    # directory = './figures'
    # file_name = 'RK45-LorenzAttractor-linear_f(%d-%d-%d-%.3f-%.3f).pdf' % (kx, ky, num_steps, free_action, mse)

    titles = ['x']
    t = 'x'
    for i in range(kx - 1):
        t = t + '`'
        titles.append(t)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    for idx in range(dy):
        ax.plot(y[:, idx], c=colors[idx], label='y[%d]' % idx)
        plt.grid(True)
    ax.legend(loc=2, prop={'size': legend_size})
    ylim = plt.ylim()
    plt.xlim(left=0)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    fig.savefig(os.path.join(directory, 'y.pdf'), bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    for idx in range(dx):
        ax.plot(x[:, idx], colors[idx], label='x[%d]' % idx)
        plt.grid(True)
    ax.legend(loc=2, prop={'size': legend_size})
    plt.ylim(ylim)
    plt.xlim(left=0)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    fig.savefig(os.path.join(directory, 'x.pdf'), bbox_inches='tight', pad_inches=0)

    coordinate_idx = 0
    while (coordinate_idx < kx):

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        for idx in range(dx):
            ax.plot(true_gen_x[coordinate_idx][:, idx], c=colors[idx], label=titles[coordinate_idx]+'[%d]' % idx)
            plt.grid(True)
        ax.legend(loc=2, prop={'size': legend_size})
        plt.xlim(left=0)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        ylim = plt.ylim()
        fig.savefig(os.path.join(directory, '%s.pdf' % titles[coordinate_idx]), bbox_inches='tight', pad_inches=0)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        estimate = gen_x_estimates[:, coordinate_idx, :]
        for idx in range(dx):
            ax.plot(estimate[:, idx], c=colors[idx],label=titles[coordinate_idx]+'hat'+'[%d]' % idx)
            plt.grid(True)
        ax.legend(loc=2, prop={'size': legend_size})
        plt.ylim(ylim)
        plt.xlim(left=0)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        fig.savefig(os.path.join(directory, '%s' % titles[coordinate_idx]+'hat'+'.pdf'), bbox_inches='tight', pad_inches=0)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        diff = true_gen_x[coordinate_idx] - gen_x_estimates[:, coordinate_idx, :]
        for idx in range(dx):
            ax.plot(diff[:, idx], c=colors[idx], label='['+titles[coordinate_idx]+'-'+titles[coordinate_idx]+'hat'+']'+'[%d]' % idx)
            plt.grid(True)
        ax.legend(loc=2, prop={'size': legend_size})
        plt.xlim(left=0)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        fig.savefig(os.path.join(directory, '%s' % (titles[coordinate_idx] + '-' + titles[coordinate_idx] + 'hat')+'.pdf'), bbox_inches='tight', pad_inches=0)

        coordinate_idx += 1