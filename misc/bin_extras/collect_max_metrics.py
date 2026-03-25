#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def extract_info_from_file_name(file_name):
    """
    Extracts the model name, epoch, replicate and set type from the file name.
    Expected file name format: full-<model>-<epoch>-run-<replicate>-<set_type>.txt
    For example: full-gat-200-run-1-test.txt
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split('-')
    if len(parts) < 6:
        raise ValueError(f"Unexpected file name format: {file_name}")
    # parts: ["full", "<model>", "<epoch>", "run", "<replicate>", "<set_type>"]
    model = parts[1]
    epoch = int(parts[2])
    run = int(parts[4])
    set_type = parts[5]
    return model, epoch, run, set_type

def process_file(file, metrics):
    """
    Reads a file, computes the maximum value for each metric over all epochs,
    and returns a dictionary with the model, replicate, final epoch (as a sanity check),
    and the maximum values for each specified metric.
    Only files where the set type is 'test' are processed.
    """
    model, epoch, run, set_type = extract_info_from_file_name(file)
    if set_type != "test":
        return None

    # Read the file: skip comment lines (lines starting with '#') and use whitespace delimiter.
    df = pd.read_table(file, delim_whitespace=True, comment='#')
    
    max_values = {}
    for metric in metrics:
        if metric in df.columns:
            max_values[metric] = df[metric].max()
        else:
            print(f"Warning: Metric '{metric}' not found in {file}.")
            max_values[metric] = None

    record = {"model": model, "replicate": run, "final_epoch": epoch}
    record.update(max_values)
    return record

def main():
    parser = argparse.ArgumentParser(
        description="Collect maximum test metrics from GNN experiment files and aggregate statistics."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="../results/data",
        help="Folder containing the data files."
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default="test.txt",
        help="Only process files ending with this string (e.g., 'test.txt')."
    )
    parser.add_argument(
        "--output_table",
        type=str,
        default="aggregated_stats.csv",
        help="Output file for the aggregated statistics (CSV format)."
    )
    args = parser.parse_args()
    
    folder = os.path.abspath(args.folder)
    # Get all files in the folder that end with the specified base name (e.g., '-test.txt')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(args.base_name)]
    
    if not files:
        print("No files found matching the criteria.")
        return

    # Define the metrics you want to analyze
    metrics = ['loss', 'prec', 'rec', 'acc', 'bacc', 'auc']
    records = []
    for file in files:
        try:
            rec = process_file(file, metrics)
            if rec is not None:
                records.append(rec)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not records:
        print("No valid test files were processed.")
        return

    # Create a DataFrame where each row corresponds to one replicate's max metrics
    df_records = pd.DataFrame(records)
    
    # Group by model and calculate the mean and std for each metric over replicates.
    agg_funcs = {metric: ['mean', 'std'] for metric in metrics}
    aggregated = df_records.groupby("model").agg(agg_funcs)
    # Flatten the MultiIndex columns
    aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
    aggregated = aggregated.reset_index()

    # Transpose the table so that models become columns and metrics (mean, std) are rows.
    transposed = aggregated.set_index('model').T

    # Save the aggregated table to the specified output file (CSV)
    output_file = os.path.abspath(args.output_table)
    transposed.to_csv(output_file, index=True)
    print(f"Aggregated stats saved to {output_file}")
    print(transposed)

if __name__ == "__main__":
    main()

# Example usage:
# python3 bin/collect_max_metrics.py --folder results/data/asd_graph_knn_20 --base_name test.txt --output_table results/comparison/aggr_stats/asd_graph_knn_20_aggregated_stats.csv
