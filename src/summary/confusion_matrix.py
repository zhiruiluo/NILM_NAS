import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path
import numpy as np
import scienceplots
plt.style.use(["ieee","no-latex"])

confusion_matrix = {
    'TSNet': {
        "microwave": [[200,10],[10,200]],
        "dishwasher": [[200,100],[200,10]]
    },
    "BitcnNILM": {
        "microwave": [[100,200],[200,100]],
        "dishwasher": [[100,200],[200,100]]
    },
}

def set_spins_color(ax,color):
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

def clean_ax(ax: plt.Axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
def display_table(ax, text, spins_color=None, facecolor=None, rotation=0, axis_off=False):
    # ax.table(text, loc='center', edges='open')
    ax.text(0.5,0.5, text, va='center', ha='center',rotation=rotation)
    if spins_color is not None:
        set_spins_color(ax,spins_color)
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    clean_ax(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    if axis_off:
        ax.axis('off')

def plot_row(appliance: str, fig: plt.Figure, gs: gridspec.GridSpec, row_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs[row_idx])
    
    ax0 = fig.add_subplot(gs0[:, 0])
    display_table(ax0, appliance, rotation=90, spins_color='white',facecolor='grey')

    ax1 = fig.add_subplot(gs0[0, 1])
    display_table(ax1, '0', spins_color='white', facecolor='lightgrey')
    
    ax2 = fig.add_subplot(gs0[1, 1])
    display_table(ax2, '1', spins_color='white', facecolor='lightgrey')
    
def plot_right_row(appliance: str, fig: plt.Figure, gs: gridspec.GridSpec, row_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[row_idx])
    
    ax0 = fig.add_subplot(gs0[:, 0])
    display_table(ax0, appliance, rotation=270, spins_color='white',facecolor='grey')

def plot_left_row(fig: plt.Figure, gs: gridspec.GridSpec, row_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[row_idx])
    ax1 = fig.add_subplot(gs0[0])
    display_table(ax1, '0', spins_color='white', facecolor='lightgrey', axis_off=True)
    
    ax2 = fig.add_subplot(gs0[1])
    display_table(ax2, '1', spins_color='white', facecolor='lightgrey', axis_off=True)
    
def plot_col(model: str, fig: plt.figure, gs: gridspec.GridSpec, col_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs[col_idx])
    
    ax0 = fig.add_subplot(gs0[0, :])
    display_table(ax0, model, spins_color='white',facecolor='grey')

    ax1 = fig.add_subplot(gs0[1, 0])
    display_table(ax1, '0', spins_color='white', facecolor='lightgrey')
    
    ax2 = fig.add_subplot(gs0[1, 1])
    display_table(ax2, '1', spins_color='white', facecolor='lightgrey')

def plot_top_col(model, fig, gs, col_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs[col_idx])
    ax0 = fig.add_subplot(gs0[0])
    display_table(ax0, model, spins_color='white',facecolor='grey')

def plot_bottom_col(fig, gs, col_idx):
    gs0 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[col_idx])
    
    ax1 = fig.add_subplot(gs0[0])
    display_table(ax1, '0', spins_color='white', facecolor='lightgrey', axis_off=True)
    
    ax2 = fig.add_subplot(gs0[1])
    display_table(ax2, '1', spins_color='white', facecolor='lightgrey', axis_off=True)


def plot_confmx(cfm, ax, show_absolute=False, show_normed=True):
    plot_confusion_matrix(np.array(cfm),hide_spines=False, hide_ticks=False, axis=ax, show_absolute=show_absolute, show_normed=show_normed)
    clean_ax(ax)
    ax.axis('off')

def plot_all():
    n_models = 2
    n_dataset = 2
    
    fig = plt.figure(figsize=(3,3))
    gs0 = gridspec.GridSpec(2,2, width_ratios=(1,4),height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05, figure=fig)
    gs_col = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs0[0,1])
    gs_row = gridspec.GridSpecFromSubplotSpec(n_dataset, 1, subplot_spec=gs0[1,0])
    
    grid_cfm = gridspec.GridSpecFromSubplotSpec(n_dataset, n_models, wspace=0.05, hspace=0.05, subplot_spec=gs0[1,1])
    
    row_idx = 0
    col_idx = 0
    for model_n, datasets_cfm in confusion_matrix.items():
        row_idx = 0
        plot_col(model_n, fig, gs_col, col_idx=col_idx)
        for dataset_n, cfm in datasets_cfm.items():
            subspec = grid_cfm[row_idx, col_idx]
            ax = fig.add_subplot(subspec)
            plot_confmx(cfm, ax)
            if col_idx == 0:
                plot_row(dataset_n,fig, gs_row, row_idx)
            row_idx += 1
        col_idx += 1
    
    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()
    path = Path('results').joinpath('confmax.png')
    fig.savefig(path.as_posix())
    print(f"save confmx {path}")
    
def plot_all_v2():
    n_models = 2
    n_dataset = 2
    
    
    fig = plt.figure(figsize=(3,3))
    gs0 = gridspec.GridSpec(3,3, width_ratios=(1,6,1),height_ratios=(1,6,1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05, figure=fig)
    gs_top_col = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs0[0,1])
    gs_bottom_col = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs0[2,1])
    
    gs_left_row = gridspec.GridSpecFromSubplotSpec(n_dataset, 1, subplot_spec=gs0[1,0])
    gs_right_row = gridspec.GridSpecFromSubplotSpec(n_dataset, 1, subplot_spec=gs0[1,2])
    
    grid_cfm = gridspec.GridSpecFromSubplotSpec(n_dataset, n_models, wspace=0.05, hspace=0.05, subplot_spec=gs0[1,1])
    
    row_idx = 0
    col_idx = 0
    for model_n, datasets_cfm in confusion_matrix.items():
        row_idx = 0
        plot_top_col(model_n, fig, gs_top_col, col_idx)
        plot_bottom_col(fig, gs_bottom_col, col_idx)
        for dataset_n, cfm in datasets_cfm.items():
            subspec = grid_cfm[row_idx, col_idx]
            ax = fig.add_subplot(subspec)
            plot_confmx(cfm, ax)
            if col_idx == 0:
                plot_left_row(fig, gs_left_row, row_idx)
                plot_right_row(dataset_n, fig, gs_right_row, row_idx)
            row_idx += 1
        col_idx += 1
    
    fig.supxlabel("Actual Appliance On/Off")
    fig.supylabel("Detected Appliance On/Off")
    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()
    path = Path('results').joinpath('confmax.png')
    fig.tight_layout()
    fig.savefig(path.as_posix())
    print(f"save confmx {path}")
    
def plot_confusion_matrix_api(dc_confusion_matrix: dict, n_models: int, n_dataset: int, path: str, figsize, show_absolute, show_normed):
    
    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(3,3, width_ratios=(1,6,1),height_ratios=(1,6,1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05, figure=fig)
    gs_top_col = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs0[0,1])
    gs_bottom_col = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs0[2,1])
    
    gs_left_row = gridspec.GridSpecFromSubplotSpec(n_dataset, 1, subplot_spec=gs0[1,0])
    gs_right_row = gridspec.GridSpecFromSubplotSpec(n_dataset, 1, subplot_spec=gs0[1,2])
    
    grid_cfm = gridspec.GridSpecFromSubplotSpec(n_dataset, n_models, wspace=0.05, hspace=0.05, subplot_spec=gs0[1,1])
    
    row_idx = 0
    col_idx = 0
    for model_n, datasets_cfm in dc_confusion_matrix.items():
        row_idx = 0
        plot_top_col(model_n, fig, gs_top_col, col_idx)
        plot_bottom_col(fig, gs_bottom_col, col_idx)
        for dataset_n, cfm in datasets_cfm.items():
            subspec = grid_cfm[row_idx, col_idx]
            ax = fig.add_subplot(subspec)
            plot_confmx(cfm, ax, show_absolute, show_normed)
            if col_idx == 0:
                plot_left_row(fig, gs_left_row, row_idx)
                plot_right_row(dataset_n, fig, gs_right_row, row_idx)
            row_idx += 1
        col_idx += 1
    
    fig.supxlabel("Actual Appliance On/Off")
    fig.supylabel("Detected Appliance On/Off")
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig(path)
    print(f"save confmx {path}")
        
if __name__ == '__main__':
    plot_all_v2()