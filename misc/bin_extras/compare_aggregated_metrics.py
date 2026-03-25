#!/usr/bin/env python3
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_aggregated_stats(file_path):
    """
    Loads an aggregated stats file and returns a DataFrame where
    the index is the metric and the columns are the models.
    Also adds a column for the network type, extracted from the file name.
    Expected file name format:
      <network_type>_aggregated_stats.csv
    """
    # Extract network type from filename
    base = os.path.basename(file_path)
    network_type = base.split('_aggregated_stats.csv')[0]
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Split the header and check if the first element is empty
    header = [x.strip() for x in lines[0].strip().split(',')]
    if header[0] == "":
        models = header[1:]
    else:
        models = header

    data = {}
    # Each subsequent line: first element is the metric, followed by the values.
    for line in lines[1:]:
        parts = [x.strip() for x in line.strip().split(',')]
        # Expect exactly 1 metric + len(models) values
        if len(parts) != len(models) + 1:
            print(f"Skipping malformed line in {file_path}: {line.strip()}")
            continue
        metric = parts[0]
        try:
            values = [float(x) for x in parts[1:]]
        except ValueError as e:
            print(f"Error converting values to float in {file_path} for metric {metric}: {e}")
            continue
        data[metric] = dict(zip(models, values))
    
    df = pd.DataFrame(data).T
    df['network'] = network_type
    return df


def gather_metric_data(metric_name, files):
    """
    For the given metric (e.g. 'prec_mean' or 'bacc_mean'),
    load all files and extract that metric row, then melt into long format.
    """
    all_dfs = []
    for f in files:
        df = load_aggregated_stats(f)
        if metric_name in df.index:
            # Extract the row for the metric
            metric_df = df.loc[[metric_name]].reset_index().rename(columns={'index': 'metric'})
            all_dfs.append(metric_df)
        else:
            print(f"Warning: {metric_name} not found in {f}")
    
    if not all_dfs:
        return None
    # Concatenate all the DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)
    # Melt the DataFrame: each row is a (network, model, value) triple.
    # We ignore the 'network' column when melting since we want to keep it as an identifier.
    long_df = combined.melt(id_vars=['network', 'metric'], var_name='model', value_name=metric_name)
    return long_df

def make_boxplot(df, metric_name, output_file):
    """
    Create a box plot for the specified metric with the following customizations:
      - Title: "Balanced Accuracy Across Models"
      - X-axis tick labels (models) in uppercase
      - Custom color (#FFABAB)
      - Larger fonts
      - Saved at 300 dpi
    """
    # Increase the base font size
    plt.rcParams.update({'font.size': 14})
    
    plt.figure(figsize=(10, 6))
    
    # Set fixed title and axis labels
    title_text = "BALANCED ACCURACY ACROSS MODELS"
    xlabel_text = "MODEL"
    ylabel_text = metric_name.replace('_', ' ').upper()  # e.g., BACC MEAN
    
    # Create the boxplot using the custom color
    ax = sns.boxplot(x='model', y=metric_name, data=df, color="#FFABAB")
    
    # Set title and axis labels with larger font sizes
    ax.set_title(title_text, fontsize=18, fontweight='bold')
    ax.set_xlabel(xlabel_text, fontsize=16)
    ax.set_ylabel(ylabel_text, fontsize=16)
    
    # Convert x-axis tick labels to uppercase
    ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
    
    plt.tight_layout()
    # Save the figure at 300 dpi
    plt.savefig(output_file, dpi=300)
    plt.savefig('bacc_asd_boxplot.pdf', dpi=300)
    plt.close()
    print(f"Box plot saved to {output_file} at 300 dpi")

def create_bacc_summary_table(files, output_file="bacc_summary.csv"):
    """
    Creates a summary table of bacc_mean and bacc_std for each network type and model.
    Each cell in the table will be formatted as "bacc_mean ± bacc_std".
    The table is saved as a CSV file.
    """
    summary = []  # Will hold a list of dictionaries, one per (network, model)
    for f in files:
        df = load_aggregated_stats(f)
        network_type = df['network'].iloc[0]  # All rows in the file have the same network type
        if "bacc_mean" not in df.index or "bacc_std" not in df.index:
            print(f"bacc metrics not found in {f}")
            continue
        
        # For each model column, extract the bacc_mean and bacc_std values.
        for model in df.columns.drop("network"):
            bacc_mean = df.loc["bacc_mean", model]
            bacc_std = df.loc["bacc_std", model]
            summary.append({
                "network": network_type,
                "model": model,
                "bacc_summary": f"{bacc_mean:.3f} ± {bacc_std:.3f}"
            })
    
    summary_df = pd.DataFrame(summary)
    # Optionally, pivot the table to have one row per network with models as columns:
    pivot_table = summary_df.pivot(index="network", columns="model", values="bacc_summary")
    
    # Save the pivot table to a CSV file.
    pivot_table.to_csv(output_file)
    print(f"Summary table saved to {output_file}")
    return pivot_table

# Then in your main() function, after processing the plots, you can add:
def main():
    folder = "results/comparison/aggr_stats"  # update as needed
    files = glob.glob(os.path.join(folder, "*_aggregated_stats.csv"))
    if not files:
        print("No aggregated stats files found in the folder.")
        return

    # Specify which metrics you want to compare:
    metrics_to_compare = ['prec_mean', 'bacc_mean']
    for metric in metrics_to_compare:
        df_metric = gather_metric_data(metric, files)
        if df_metric is not None:
            output_file = f"{metric}_boxplot.png"
            make_boxplot(df_metric, metric, output_file)
        else:
            print(f"No data found for {metric}.")
    
    # Create and save the summary table for bacc metrics:
    create_bacc_summary_table(files)

if __name__ == "__main__":
    main()
