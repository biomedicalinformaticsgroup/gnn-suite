#!/usr/bin/env python3


import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

"""
compare_models_local.py - Compare GNN Training Metrics Line Plots

This script reads GNN training metrics from data files and generates line plots
to compare the training progress of different models over epochs.

Usage:
    python compare_models_local.py --line_plot <output_file> --folder <data_folder> --base_name <base_name> --task_type <task_type>

Arguments:
    --line_plot <output_file>
        Path to save the generated line plot (PDF format recommended).

    --folder <data_folder>
        Location of the data files containing GNN training metrics.

    --base_name <base_name>
        Base name of the files to be analyzed. The script will search for files
        that start with this base name and have a specific naming convention.

    --task_type <task_type>
        Task type: 'binary', 'multiclass', or 'regression' (default: binary).
        This determines which metrics are plotted:
        - binary: loss, prec, rec, acc, bacc, auc
        - multiclass: loss, prec, rec, acc, bacc, f1, auc
        - regression: loss, mse, rmse, mae, r2

Examples:
    # Binary classification
    python bin/compare_models_local.py --line_plot results/comparison/comp.pdf --folder results/data/string --base_name binary-test.txt

    # Multiclass classification
    python bin/compare_models_local.py --line_plot results/comparison/comp.pdf --folder results/data/string --base_name multiclass-test.txt --task_type multiclass

    # Regression
    python bin/compare_models_local.py --line_plot results/comparison/comp.pdf --folder results/data/sfari --base_name regression-test.txt --task_type regression

Note:
    This script extracts model names, epochs, and runs from the file names
    in the specified folder based on a specific naming convention. It then
    generates line plots to compare the training progress of the models.
"""


def plot_line_plots(line_plot, data_files, metrics, model_names, runs):
    # Read data from the files
    all_data = []
    for file, model_name, run in zip(data_files, model_names, runs):
        df = pd.read_table(file, delimiter="\s+", comment='#')
        df['model'] = model_name
        df['run'] = run
        all_data.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(all_data, ignore_index=True)
    
    # print(all_data)

    sns.set(style='whitegrid', font_scale=1.6)
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each model
    palette = sns.color_palette("tab10", len(set(model_names)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Using sns.lineplot to plot the mean and standard deviation of metrics
        sns.lineplot(data=all_data, x='epoch', y=metric, hue='model', errorbar='se', ax=ax, palette=palette)

        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=16, fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [label.upper() for label in labels])

    # Hide unused subplot axes
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("GNN Training Progress", fontsize=22, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(line_plot, dpi=300)
    plt.close()
    print(f"Line plots saved as {line_plot}")


def extract_info_from_file_name(file_name):
    parts = os.path.splitext(file_name)[0].split('-')
    print("Parts:")
    print(parts)
    parts = parts[2:]
    print("Sliced parts:")
    print(parts)
    model_name, epoch_str, run_prefix, run_str, base_name = parts[:5]
    epoch = int(epoch_str)
    run = int(run_str)  
    
    return base_name, model_name, epoch, run

def max_metric_stats(filename, data_files, model_names, runs, task_type='binary'):
    """
    Compute statistics for the best epoch based on the primary comparison metric.

    For binary/multiclass: uses 'bacc' (balanced accuracy)
    For regression: uses 'r2' (R-squared)
    """
    # Read data from the files
    all_data = []
    for file, model_name, run in zip(data_files, model_names, runs):
        df = pd.read_table(file, delimiter="\s+", comment='#')
        df['model'] = model_name
        df['run'] = run
        all_data.append(df)

    # Concatenate all dataframes
    all_data = pd.concat(all_data, ignore_index=True)

    # Get the appropriate comparison metric for this task type
    comparison_metric = get_comparison_metric(task_type)

    # Group by model and epoch, then compute mean and std of the metric for each group
    grouped = all_data.groupby(['model', 'epoch'])[comparison_metric].agg(['mean', 'std']).reset_index()

    # Identify the epoch with max mean metric for each model
    idx = grouped.groupby('model')['mean'].idxmax()
    max_metric_df = grouped.loc[idx]

    # Combine mean and std in one column with "±" symbol
    max_metric_df['mean ± std'] = max_metric_df['mean'].round(3).astype(str) + " ± " + max_metric_df['std'].round(3).astype(str)

    # Save the stats to a file
    max_metric_df.to_latex(filename, index=False, columns=['model', 'epoch', 'mean ± std'])
    print(f"Stats saved as {filename}")
    print(f"Comparison metric: {comparison_metric}")
    print(max_metric_df[['model', 'epoch', 'mean ± std']])
    return max_metric_df



def get_metrics_for_task(task_type):
    """Return appropriate metrics based on task type."""
    if task_type == 'regression':
        return ['loss', 'mse', 'rmse', 'mae', 'r2']
    elif task_type == 'multiclass':
        return ['loss', 'prec', 'rec', 'acc', 'bacc', 'f1', 'auc']
    else:  # binary
        return ['loss', 'prec', 'rec', 'acc', 'bacc', 'auc']


def get_comparison_metric(task_type):
    """Return the primary comparison metric for each task type."""
    if task_type == 'regression':
        return 'r2'
    else:  # binary or multiclass
        return 'bacc'


def main():
    parser = argparse.ArgumentParser(description='Compare GNN training metrics')
    parser.add_argument('--line_plot', type=str, help='Path to save the line plot')
    parser.add_argument('--folder', type=str, default='../results/data', help='Location of the data files (default: ./results/)')
    parser.add_argument('--base_name', type=str, help='Base name for extracting information')
    parser.add_argument('--task_type', type=str, default='binary',
                        choices=['binary', 'multiclass', 'regression'],
                        help='Task type: binary, multiclass, or regression (default: binary)')

    args = parser.parse_args()
    args.line_plot = os.path.abspath(args.line_plot)
    args.folder = os.path.abspath(args.folder)

    data_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith(args.base_name)]

    file_info = [extract_info_from_file_name(file) for file in data_files]
    base_names, model_names, epochs, runs = zip(*file_info)
    print(runs)

    metrics = get_metrics_for_task(args.task_type)

    plot_line_plots(args.line_plot, data_files, metrics, model_names, runs)

    # Uncomment to generate statistics table:
    # stats_file = args.line_plot.replace('.pdf', '_stats.tex')
    # max_metric_stats(stats_file, data_files, model_names, runs, args.task_type)

if __name__ == "__main__":
    main()



