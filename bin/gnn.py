#!/usr/bin/env python3
import sys
import typer
import csv
import os

import numpy as np
from sklearn import metrics, model_selection

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import mlflow

from models import GCN, GAT, HGCN, PHGCN, GraphSAGE, GraphIsomorphismNetwork, GCNII, GraphTransformer

WITH_MLFLOW = os.environ.get("WITH_MLFLOW", "0") == "1"



def read_gene_file(gene_filename):
    """
    Reads the gene file and returns the feature matrix and labels.

    Parameters:
    - gene_filename: Path to the gene file.

    Returns:
    - feature_matrix: List of features for each gene.
    - labels: List of labels for each gene.
    - gene_to_id: Dictionary mapping gene names to IDs.
    """
    feature_matrix = []
    labels = []

    gene_to_id = {}
    gene_id = 0

    # Open the gene file
    with open(gene_filename, 'r') as f:
        reader = csv.reader(f)

        # Iterate over each row in the file
        for row in reader:
            # Skip the header row
            if row[-1] == "label":
                continue

            # Extract the gene name, features, and label from the row
            gene_name = row[0]
            features = list(map(float, row[1:-1]))
            label = float(row[-1])

            # If the gene name is not in the dictionary, add it
            if gene_name not in gene_to_id:
                gene_to_id[gene_name] = gene_id

                # Add the features and label to their respective lists
                feature_matrix.append(features)
                labels.append([label])

                # Increment the gene ID for the next new gene
                gene_id += 1

    return feature_matrix, labels, gene_to_id



def read_network_file(network_filename, gene_to_id):
    """
    Reads the network file and returns the edge matrix.

    Parameters:
    - network_filename: Path to the network file.
    - gene_to_id: Dictionary mapping gene names to IDs.

    Returns:
    - edge_matrix: List of edges, each represented as a pair of gene IDs.
    """

    edge_matrix = []
    for row, record in enumerate(csv.reader(open(network_filename), delimiter="\t")):
        gene_name_1, gene_name_2 = record
        edge_matrix.append([gene_to_id[gene_name_1], gene_to_id[gene_name_2]])
        edge_matrix.append([gene_to_id[gene_name_2], gene_to_id[gene_name_1]])

    return edge_matrix



def load_data(gene_filename, network_filename, train_size=0.7):
    """
    Load data into PyTorch-Geometric format.

    Parameters:
    - gene_filename: Path to the gene file.
    - network_filename: Path to the network file.
    - train_size: Proportion of the dataset to include in the train split.

    Returns:
    - Data object with features, edge indices, labels, and train/test masks.
    """
    # Read the gene and network files
    feature_matrix, label_list, gene_to_id = read_gene_file(gene_filename)
    edge_matrix = read_network_file(network_filename, gene_to_id)

    # Convert the data to tensors
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float)
    label_tensor = torch.tensor(label_list, dtype=torch.float)
    edge_tensor = torch.tensor(edge_matrix, dtype=torch.long)

    # Split the dataset into positive and negative samples
    positive_indices = torch.where(label_tensor == 1)[0]
    negative_indices = torch.where(label_tensor == 0)[0]

    positive_train_indices, positive_test_indices = model_selection.train_test_split(
        positive_indices, train_size=train_size
    )
    negative_train_indices, negative_test_indices = model_selection.train_test_split(
        negative_indices, train_size=train_size
    )

    # Create masks for the train and test sets
    train_mask = torch.zeros(label_tensor.size(0), dtype=torch.bool)
    train_mask[positive_train_indices] = True
    train_mask[negative_train_indices] = True

    test_mask = torch.zeros(label_tensor.size(0), dtype=torch.bool)
    test_mask[positive_test_indices] = True
    test_mask[negative_test_indices] = True

    # Create a PyTorch-Geometric data object
    data = Data(
        x=feature_tensor,
        edge_index=edge_tensor.t().contiguous(),
        y=label_tensor,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    return data


def build_model(name, data, dropout=0.5, alpha=0.1, theta=0.5):
    """
    Instantiate a model based on the name.

    Parameters:
    - name: Name of the model to instantiate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.

    Returns:
    - An instance of the specified model.
    """
    num_classes = data.y.size(1)
    

    num_features = data.x.size(1)

    if name not in ["gcn", "gat", "gat3h", "hgcn", "phgcn", "sage",
                    "gin", "gcn2", "gtn"]:
        print("Unknown model: {}.".format(name))
        sys.exit(1)
    elif name == "gat":
        return GAT(num_features, num_classes, dropout=dropout)
    elif name == "gat3h":
        return GAT(num_features, num_classes, num_heads=3, dropout=dropout)
    elif name == "hgcn":
        return HGCN(num_features, num_classes, dropout=dropout)
    elif name == "phgcn":
        return PHGCN(num_features, num_classes, dropout=dropout)
    elif name == "sage":
        return GraphSAGE(num_features, num_classes, dropout=dropout)
    elif name == "gtn":
        return GraphTransformer(num_features, num_classes, dropout=dropout)
    elif name == "gin":
        return GraphIsomorphismNetwork(num_features, num_classes)
    elif name == "gcn2":
        return GCNII(num_features, num_classes, dropout=dropout, alpha=alpha, theta = None)
    else:
        return GCN(num_features, num_classes, dropout=dropout)

def evaluate_all(model, data, thq=0.95):
    """
    Perform an evaluation step on the entire dataset.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.

    Returns:
    - Evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(data))

        th = np.quantile(out.cpu().numpy(), thq)
        truth, prob, pred = (
            data.y.cpu().numpy(),
            out.cpu().numpy(),
            (out >= th).cpu().numpy().astype(int),
        )

        acc = metrics.accuracy_score(truth, pred)
        bacc = metrics.balanced_accuracy_score(truth, pred)
        precision = metrics.precision_score(truth, pred)
        recall = metrics.recall_score(truth, pred)
        auc = metrics.roc_auc_score(truth, prob, average="weighted")
        tn, fp, fn, tp = metrics.confusion_matrix(truth, pred).ravel()
        
        return tn, fp, fn, tp, precision, recall, acc, bacc, auc


def evaluate_train(model, data, thq=0.95):
    """
    Perform an evaluation step on the training data.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.

    Returns:
    - Evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(data))

        th = np.quantile(out[data.train_mask].cpu().numpy(), thq)
        truth, prob, pred = (
            data.y[data.train_mask].cpu().numpy(),
            out[data.train_mask].cpu().numpy(),
            (out[data.train_mask] >= th).cpu().numpy().astype(int),
        )

        acc = metrics.accuracy_score(truth, pred)
        bacc = metrics.balanced_accuracy_score(truth, pred)
        precision = metrics.precision_score(truth, pred)
        recall = metrics.recall_score(truth, pred)
        auc = metrics.roc_auc_score(truth, prob, average="weighted")
        tn, fp, fn, tp = metrics.confusion_matrix(truth, pred).ravel()
        
        return tn, fp, fn, tp, precision, recall, acc, bacc, auc



def evaluate(model, data, thq=0.95):
    """
    Perform an evaluation on the test data.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.

    Returns:
    - Evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(data))

        th = np.quantile(out[data.train_mask].cpu().numpy(), thq)
        truth, prob, pred = (
            data.y[data.test_mask].cpu().numpy(),
            out[data.test_mask].cpu().numpy(),
            (out[data.test_mask] >= th).cpu().numpy().astype(int),
        )

        acc = metrics.accuracy_score(truth, pred)
        bacc = metrics.balanced_accuracy_score(truth, pred)
        precision = metrics.precision_score(truth, pred)
        recall = metrics.recall_score(truth, pred)
        auc = metrics.roc_auc_score(truth, prob, average="weighted")
        tn, fp, fn, tp = metrics.confusion_matrix(truth, pred).ravel()
        
        return tn, fp, fn, tp, precision, recall, acc, bacc, auc

def train(model, data, optimizer, weight=1.0):
    """
    Perform a training step.

    Parameters:
    - model: The model to train.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - optimizer: The optimizer to use for training.
    - weight: Weight for positive examples in the loss function.

    Returns:
    - Loss value.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.binary_cross_entropy_with_logits(
        out[data.train_mask], data.y[data.train_mask], pos_weight=weight
    )
    loss.backward()
    optimizer.step()

    return loss

def compute_positive_sample_weight(data):
    """
    Compute the weight for positive samples in the data.

    Parameters:
    - data: The data object containing the labels and the training mask.

    Returns:
    - The computed weight for positive samples.
    """
    num_samples = data.y[data.train_mask].size(0)
    num_positive_samples = data.y[data.train_mask].sum()

    pos_weight = (num_samples - num_positive_samples) / num_positive_samples

    return pos_weight

from torch_geometric.utils import to_undirected

def print_network_statistics(data):
    """
    Print network statistics of a PyG data object.

    Parameters:
    - data: The data object containing the features, edge indices, labels, and train/test masks.
    """
    # Calculate basic statistics
    num_nodes = data.num_nodes
    
    # Get unique undirected edges
    undirected_edges = to_undirected(data.edge_index)
    num_unique_undirected_edges = undirected_edges.size(1) // 2  # Each edge is represented twice
    
    density = num_unique_undirected_edges / (num_nodes * (num_nodes - 1) / 2)

    # Compute the average degree based on unique undirected edges
    avg_degree = 2 * num_unique_undirected_edges / num_nodes

    num_pos_samples = int(data.y.sum().item())
    num_neg_samples = num_nodes - num_pos_samples
    balance_ratio = num_pos_samples / num_neg_samples

    # Print statistics
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Unique Undirected Edges: {num_unique_undirected_edges}")
    print(f"Graph Density: {density:.5f}")
    print(f"Average Degree based on Unique Edges: {avg_degree:.5f}")
    print(f"Number of Positive Samples (labels): {num_pos_samples}")
    print(f"Number of Negative Samples (labels): {num_neg_samples}")
    print(f"Data Balance (Pos/Neg Ratio): {balance_ratio:.5f}")

    # Print data in LaTeX table format
    print(f"\n{num_nodes} & {num_unique_undirected_edges} & {density:.5f} & {avg_degree:.5f} & {num_pos_samples} & {num_neg_samples} & {balance_ratio:.5f}")


def run(
    gene_filename: str,
    network_filename: str,
    train_size: float = 0.8,
    model_name: str = "gcn",
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    eval_threshold: float = 0.9,
    verbose_interval: int = 10,
    dropout: float = 0.5,
    alpha: float = 0.1,
    theta: float = 0.5
):
    """
    Train a graph neural network with MLflow tracking for binary classification.

    Parameters:
    - gene_filename: Path to the gene file.
    - network_filename: Path to the network file.
    - train_size: Proportion of the dataset to include in the train split.
    - model_name: Name of the model to train.
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - weight_decay: Weight decay for the optimizer.
    - eval_threshold: Threshold for quantile function in evaluation.
    - verbose_interval: Interval for printing training progress.
    - dropout: Dropout rate for regularization.
    - alpha: Alpha parameter (for GCN2).
    - theta: Theta parameter (for GCN2).
    """
    if WITH_MLFLOW:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "gnn-binary-classification")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run_name = f"{model_name}"

        mlflow.start_run(run_name=run_name)

    # Load the data
    data = load_data(gene_filename, network_filename, train_size)

    # Print basic information about the data and model
    print(
        f"# Number of nodes={data.num_nodes}; Number of edges={data.num_edges}; "
        f"Number of node features={data.num_features}; Model: {model_name}."
    )

    # Print network statistics
    # print_network_statistics(data)

    # Determine the device to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model and move it to the appropriate device
    model = build_model(model_name, data, dropout, alpha, theta).to(device)
    data = data.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if WITH_MLFLOW:
        mlflow.log_params({
            "model": model_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "alpha": alpha,
            "theta": theta,
            "epochs": epochs,
            "train_size": train_size,
            "eval_threshold": eval_threshold
        })
        mlflow.set_tag("task_type", "binary")
        mlflow.set_tag("model_name", model_name)

    # Compute the weight for positive samples
    pos_weight = compute_positive_sample_weight(data)

    # Print the header for the training progress output
    print(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} ".format(
            "epoch",
            "loss",
            "tn",
            "fp",
            "fn",
            "tp",
            "prec",
            "rec",
            "acc",
            "bacc",
            "auc",
        )
    )

    # instantiate BACC array for hyperparameter search
    bacc_array = []

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, weight=pos_weight)

        if WITH_MLFLOW:
            mlflow.log_metric("train_loss", float(loss), step=epoch)

        # Evaluate the model and print the results at the specified interval
        if (epoch % verbose_interval == 0) or (epoch == 1):
            tn, fp, fn, tp, precision, recall, acc, bacc, auc = evaluate(
                model, data, eval_threshold
            )
            tn_train, fp_train, fn_train, tp_train, precision_train, recall_train, acc_train, bacc_train, auc_train = evaluate_train(
                model, data, eval_threshold
            )
            tn_all, fp_all, fn_all, tp_all, precision_all, recall_all, acc_all, bacc_all, auc_all = evaluate_all(
                model, data, eval_threshold
            )
            print(
                "Test: {:>10} {:>10.5g} {:>10} {:>10} {:>10} {:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                    epoch, loss, tn, fp, fn, tp, precision, recall, acc, bacc, auc
                )
            )

            # Append the balanced accuracy to the array
            bacc_array.append(bacc)

            if WITH_MLFLOW:
                mlflow.log_metric("val_bacc", bacc, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_auc", auc, step=epoch)
                mlflow.log_metric("val_accuracy", acc, step=epoch)

            print(
                "Train: {:>10} {:>10.5g} {:>10} {:>10} {:>10} {:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                    epoch, loss, tn_train, fp_train, fn_train, tp_train, precision_train, recall_train, acc_train, bacc_train, auc_train
                )
            )
            print(
                "All: {:>10} {:>10.5g} {:>10} {:>10} {:>10} {:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                    epoch, loss, tn_all, fp_all, fn_all, tp_all, precision_all, recall_all, acc_all, bacc_all, auc_all
                )
            )

    # Get the max balanced accuracy
    max_bacc = max(bacc_array)

    if WITH_MLFLOW:
        mlflow.log_metric("final_bacc", max_bacc)
        mlflow.log_metric("final_precision", precision)
        mlflow.log_metric("final_recall", recall)
        mlflow.log_metric("final_auc", auc)
        mlflow.log_metric("final_accuracy", acc)
        mlflow.end_run()

    return max_bacc

if __name__ == "__main__":
    typer.run(run)
    torch.cuda.empty_cache()