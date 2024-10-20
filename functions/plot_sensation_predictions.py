import matplotlib.pyplot as plt
import numpy as np
import os
def plot(ky, dy, gen_sensations, gen_predictions, directory, legend_size=22, tick_size = 22):
    t = 'y'
    titles = [t]
    for i in range(ky - 1):
        t = t + '`'
        titles.append(t)

    if ky == 1:
        counter = 1
        gen_sensations = np.squeeze(gen_sensations, axis=1)
        gen_predictions = np.squeeze(gen_predictions, axis=1)

        for j in range(dy):
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            ax.plot(gen_sensations[:, j], c='#009E73', label=titles[0] + '[%d]' % j)
            ax.plot(gen_predictions[:, j], c='#CC79A7', alpha=0.80, label=titles[0]+ 'hat' + '[%d]' % j)
            ax.legend(loc=2, prop={'size': legend_size})
            plt.grid()
            plt.xlim(left=0)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            fig.savefig(os.path.join(directory, '%s' % titles[0] + '[%d]' % j + '.pdf'), bbox_inches='tight', pad_inches=0)
            counter += 1
    else:
        counter = 1
        for i in range(ky):
            for j in range(dy):
                fig = plt.figure(figsize=(16, 9))
                ax = fig.add_subplot(111)
                ax.plot(gen_sensations[:, i, j], c='#009E73', label=titles[i] + '[%d]' % j)
                ax.plot(gen_predictions[:, i, j], c='#CC79A7', alpha=0.80, label=titles[i]+'hat'+'[%d]' % j)
                ax.legend(loc=2, prop={'size': legend_size})
                plt.grid()
                plt.xlim(left=0)
                plt.xticks(fontsize=tick_size)
                plt.yticks(fontsize=tick_size)
                fig.savefig(os.path.join(directory, '%s' % titles[i] + '[%d]' % j + '.pdf'), bbox_inches='tight', pad_inches=0)
                counter += 1
