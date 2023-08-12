from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path
import numpy as np



def clean_ax(ax: plt.Axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def plot_confmx(cfm, ax, show_absolute=False, show_normed=True, class_names=None):
    plot_confusion_matrix(np.array(cfm),hide_spines=False,
                          hide_ticks=False, axis=ax, 
                          show_absolute=show_absolute, 
                          show_normed=show_normed, class_names=class_names)
    clean_ax(ax)
    ax.axis('off')
    
    
def main():
    confmx = {
        "RF": [
            [35820, 2780, 340],
            [9817, 3523, 34],
            [5839, 132, 115]
        ],
        "Proposed": [
            [23519, 8675, 6746],
            [3996, 7852, 1526],
            [3027, 798, 2260],
        ]
    }
    class_names = ['Medium','Old','Young']
    for k, v in confmx.items():
        fig, ax = plt.subplots()
        plot_confmx(v, ax, show_absolute=True, show_normed=True, class_names=class_names)
        plt.savefig(f'results/temp_confmx_{k}.png')
        
main()