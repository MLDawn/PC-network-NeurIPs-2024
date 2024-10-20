import matplotlib.pyplot as plt
import os
def plot(vfe, directory, legend_size=22, tick_size = 22):

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.plot(vfe, c='#a65628', label='VFE')
    ax.legend(loc=2, prop={'size': legend_size})
    plt.ylim(bottom=0)
    plt.yticks(fontsize=tick_size)
    plt.xlim(left=0)
    plt.xticks(fontsize=tick_size)
    plt.grid(True)
    fig.tight_layout()
    # Save the plot to the constructed file path
    fig.savefig(os.path.join(directory, 'vfe.pdf'), bbox_inches='tight', pad_inches=0)

    free_action = []
    temp = vfe[0]
    for idx in range(len(vfe)):
        if idx == 0:
            free_action.append(temp)
        else:
            temp += vfe[idx]
            free_action.append(temp)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.plot(free_action, c='#a65628', label='Free Action')
    ax.legend(loc=2, prop={'size': legend_size})
    plt.xticks(fontsize=14)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=tick_size)
    plt.xlim(left=0)
    plt.xticks(fontsize=tick_size)
    plt.grid(True)
    # Save the plot to the constructed file path
    fig.savefig(os.path.join(directory, 'fa.pdf'), bbox_inches='tight', pad_inches=0)