import typer
from pathlib import Path
from typing import List 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_line_plots(line_plot: str, data_files: str, model_names: str, metrics=('loss', 'prec', 'rec', 'acc', 'bacc', 'auc')):
    dfs = [pd.read_table(file, delimiter="\s+", comment='#') for file in data_files]
    
    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each model
    palette = sns.color_palette("tab10", len(model_names))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, model_data in enumerate(dfs):
            model_name = model_names[j]
            color = palette[j % len(palette)]  
            
            sns.lineplot(data=model_data, x='epoch', y=metric, markers=True, ax=ax, color=color, label=model_name)
            
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=14, fontweight='bold')
        ax.set_xlim(1, max(max(df['epoch']) for df in dfs))
        ax.legend()
        
    fig.suptitle("GNN Training Progress - Line Plots", fontsize=16, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    plt.savefig(line_plot)
    plt.close()
    print(f"Line plots saved as {line_plot}")


def plot_box_plots(box_plot: str, data_files: str, model_names: str, metrics=('loss', 'prec', 'rec', 'acc', 'bacc', 'auc')):
    dfs = [pd.read_table(file, delimiter="\s+", comment='#') for file in data_files]
    
    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each model
    palette = sns.color_palette("tab10", len(model_names))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, model_data in enumerate(dfs):
            model_name = model_names[j]
            color = palette[j % len(palette)]  
            
            sns.boxplot(data=model_data, x='epoch', y=metric, ax=ax, color=color, showfliers=False, width=0.3)
            
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=14, fontweight='bold')
        ax.set_xlim(1, max(max(df['epoch']) for df in dfs))
        
    fig.suptitle("GNN Training Progress - Box Plots", fontsize=16, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    plt.savefig(box_plot)
    plt.close()
    print(f"Box plots saved as {box_plot}")


def plot_violin_plots(violin_plot: str, data_files: str, model_names: str, metrics=('loss', 'prec', 'rec', 'acc', 'bacc', 'auc')):
    dfs = [pd.read_table(file, delimiter="\s+", comment='#') for file in data_files]
    
    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each model
    palette = sns.color_palette("tab10", len(model_names))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, model_data in enumerate(dfs):
            model_name = model_names[j]
            color = palette[j % len(palette)]  # Get a distinct color from the palette
            
            sns.violinplot(data=model_data, x='epoch', y=metric, ax=ax, color=color, inner='stick')
            
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=14, fontweight='bold')
        ax.set_xlim(1, max(max(df['epoch']) for df in dfs))
        
    fig.suptitle("GNN Training Progress - Violin Plots", fontsize=16, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    plt.savefig(violin_plot)
    plt.close()
    print(f"Violin plots saved as {violin_plot}")


def compare_metrics(line_plot: str, box_plot: str, violin_plot: str, data_files: List[str], model_names: List[str], metrics: List[str]):
    plot_line_plots(line_plot, data_files, model_names, metrics)
    plot_box_plots(box_plot, data_files, model_names, metrics)
    plot_violin_plots(violin_plot, data_files, model_names, metrics)

if __name__ == "__main__":
    typer.run(compare_metrics)
