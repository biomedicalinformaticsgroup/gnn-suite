#!/usr/bin/env python3
import typer
from pathlib import Path
from typing import List 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_epochs(filename: str, data: List[Path], metric='loss'):
    dfs = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(dfs)
    
    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=data, x='epoch', y=metric, markers=True, ax=ax)
    ax.set(xlabel='Epoch', ylabel=metric.capitalize())
    ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=16, fontweight='bold')
    ax.set_xlim(1, max(data['epoch']))
    
    fig.suptitle("GNN Training Progress", fontsize=20, fontweight='bold', y=0.95)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")




def plot_metrics(filename: str, data: List[Path],
                 metrics=('loss', 'prec', 'rec', 'acc', 'bacc', 'auc'), model: str = '', task_type: str = ''):
    dfs = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(dfs)

    available_metrics = [m for m in metrics if m in data.columns]

    if not available_metrics:
        print(f"Warning: None of the specified metrics found in data. Available columns: {list(data.columns)}")
        return

    print(data.head())

    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols

    num_plots = len(available_metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    # Define a color palette with distinct colors for each subplot
    palette = sns.color_palette("tab10", num_plots)

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        color = palette[i % len(palette)]  # Get a distinct color from the palette

        sns.lineplot(data=data, x='epoch', y=metric, markers=False,
                    ax=ax, color=color, label=metric.capitalize())

        # Plot the error bars
       # ax.fill_between(metric_mean.index, metric_mean - metric_std,
       #                 metric_mean + metric_std, color=color, alpha=0.2)

        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=14, fontweight='bold')
        ax.set_xlim(1, max(data['epoch']))

    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("GNN Training Progress", fontsize=16, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    info_text = f"Model: {model}"
    if task_type:
        info_text += f" | Task Type: {task_type.capitalize()}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=12)

    plt.savefig(filename)
    plt.close()
    print(f"Plots saved as {filename}")


def main(
    filename: str,  # Filename for single plot or metrics plot
    data: List[Path],  # List of data file paths
    metric: str = 'loss',  # Metric to plot (default: 'loss')
    metrics: List[str] = None,  # Metrics to plot (auto-detected if None)
    model: str = '',  # Model information
    task_type: str = 'binary',  # 'binary', 'multiclass', 'regression'
):
    if metrics is None:
        if task_type == 'regression':
            metrics = ['loss', 'mse', 'rmse', 'mae', 'r2']
        elif task_type == 'multiclass':
            metrics = ['loss', 'prec', 'rec', 'acc', 'bacc', 'f1', 'auc']
        else:  # binary
            metrics = ['loss', 'prec', 'rec', 'acc', 'bacc', 'auc']

    plot_metrics(filename, data, metrics, model, task_type)


if __name__ == "__main__":
    typer.run(main)





