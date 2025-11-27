#!/usr/bin/env python3
import sys
import typer
import csv

import numpy as np
from sklearn import metrics, model_selection

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models import GCN, GAT, HGCN, PHGCN, GraphSAGE, GraphIsomorphismNetwork, GCNII, GraphTransformer


def read_gene_file(gene_filename, task_type='binary'):
    """
    Reads the gene file and returns the feature matrix and labels.

    Parameters:
    - gene_filename: Path to the gene file.
    - task_type: 'binary', 'multiclass', or 'regression' task.

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

                if task_type == 'multiclass':
                    labels.append(int(label))
                elif task_type == 'regression':
                    labels.append(label)
                else:
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



def load_data(gene_filename, network_filename, train_size=0.7, task_type='binary'):
    """
    Load data into PyTorch-Geometric format.

    Parameters:
    - gene_filename: Path to the gene file.
    - network_filename: Path to the network file.
    - train_size: Proportion of the dataset to include in the train split.
    - task_type: 'binary', 'multiclass', or 'regression' task.

    Returns:
    - Data object with features, edge indices, labels, and train/test masks.
    """
    # Read the gene and network files
    feature_matrix, label_list, gene_to_id = read_gene_file(gene_filename, task_type=task_type)
    edge_matrix = read_network_file(network_filename, gene_to_id)

    # Convert the data to tensors
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float)

    if task_type == 'multiclass':
        label_tensor = torch.tensor(label_list, dtype=torch.long)
    elif task_type == 'regression':
        label_tensor = torch.tensor(label_list, dtype=torch.float).unsqueeze(1)
    else:
        label_tensor = torch.tensor(label_list, dtype=torch.float)

    edge_tensor = torch.tensor(edge_matrix, dtype=torch.long)

    if task_type == 'multiclass':
        all_indices = torch.arange(len(label_list))
        train_indices, test_indices = model_selection.train_test_split(
            all_indices, train_size=train_size, stratify=label_list
        )
    elif task_type == 'regression':
        all_indices = torch.arange(len(label_list))
        train_indices, test_indices = model_selection.train_test_split(
            all_indices, train_size=train_size, shuffle=True, random_state=42
        )
    else:
        # Split the dataset into positive and negative samples
        positive_indices = torch.where(label_tensor == 1)[0]
        negative_indices = torch.where(label_tensor == 0)[0]

        positive_train_indices, positive_test_indices = model_selection.train_test_split(
            positive_indices, train_size=train_size
        )
        negative_train_indices, negative_test_indices = model_selection.train_test_split(
            negative_indices, train_size=train_size
        )

        train_indices = torch.cat([positive_train_indices, negative_train_indices])
        test_indices = torch.cat([positive_test_indices, negative_test_indices])

    # Create masks for the train and test sets
    train_mask = torch.zeros(label_tensor.size(0), dtype=torch.bool)
    train_mask[train_indices] = True

    test_mask = torch.zeros(label_tensor.size(0), dtype=torch.bool)
    test_mask[test_indices] = True

    # Create a PyTorch-Geometric data object
    data = Data(
        x=feature_tensor,
        edge_index=edge_tensor.t().contiguous(),
        y=label_tensor,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    return data


def build_model(name, data, dropout=0.5, alpha=0.1, theta=0.5, task_type='binary'):
    """
    Instantiate a model based on the name.

    Parameters:
    - name: Name of the model to instantiate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - dropout: Dropout probability for regularization.
    - alpha: Alpha parameter (for GCN2).
    - theta: Theta parameter (for GCN2).
    - task_type: 'binary', 'multiclass', or 'regression' task.

    Returns:
    - An instance of the specified model.
    """
    if task_type == 'multiclass':
        num_classes = int(data.y.max().item()) + 1
    elif task_type == 'regression':
        num_classes = 1
    else:
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

def evaluate_all(model, data, thq=0.95, task_type='binary'):
    """
    Perform an evaluation step on the entire dataset.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.
    - task_type: Type of task ('binary', 'multiclass', 'regression').

    Returns:
    - Binary: (tn, fp, fn, tp, precision, recall, acc, bacc, auc)
    - Multiclass: (precision, recall, acc, bacc, f1, auc)
    - Regression: (mse, rmse, mae, r2)
    """
    model.eval()
    with torch.no_grad():
        out_logits = model(data)

        if task_type == 'multiclass':
            out = F.softmax(out_logits, dim=1)
            pred = out.argmax(dim=1).cpu().numpy()
            truth = data.y.cpu().numpy()
            prob = out.cpu().numpy()

            acc = metrics.accuracy_score(truth, pred)
            bacc = metrics.balanced_accuracy_score(truth, pred)
            precision = metrics.precision_score(truth, pred, average='macro', zero_division=0)
            recall = metrics.recall_score(truth, pred, average='macro', zero_division=0)
            f1 = metrics.f1_score(truth, pred, average='macro', zero_division=0)

            try:
                auc = metrics.roc_auc_score(truth, prob, average="macro", multi_class="ovr")
            except:
                auc = 0.0

            return precision, recall, acc, bacc, f1, auc

        elif task_type == 'regression':
            pred = out_logits.cpu().numpy()
            truth = data.y.cpu().numpy()

            mse = metrics.mean_squared_error(truth, pred)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(truth, pred)
            r2 = metrics.r2_score(truth, pred)

            return mse, rmse, mae, r2

        else:  # binary
            out = torch.sigmoid(out_logits)

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


def evaluate_train(model, data, thq=0.95, task_type='binary'):
    """
    Perform an evaluation step on the training data.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.
    - task_type: Type of task ('binary', 'multiclass', 'regression').

    Returns:
    - Binary: (tn, fp, fn, tp, precision, recall, acc, bacc, auc)
    - Multiclass: (precision, recall, acc, bacc, f1, auc)
    - Regression: (mse, rmse, mae, r2)
    """
    model.eval()
    with torch.no_grad():
        out_logits = model(data)

        if task_type == 'multiclass':
            out = F.softmax(out_logits, dim=1)
            pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
            truth = data.y[data.train_mask].cpu().numpy()
            prob = out[data.train_mask].cpu().numpy()

            acc = metrics.accuracy_score(truth, pred)
            bacc = metrics.balanced_accuracy_score(truth, pred)
            precision = metrics.precision_score(truth, pred, average='macro', zero_division=0)
            recall = metrics.recall_score(truth, pred, average='macro', zero_division=0)
            f1 = metrics.f1_score(truth, pred, average='macro', zero_division=0)

            try:
                auc = metrics.roc_auc_score(truth, prob, average="macro", multi_class="ovr")
            except:
                auc = 0.0

            return precision, recall, acc, bacc, f1, auc

        elif task_type == 'regression':
            pred = out_logits[data.train_mask].cpu().numpy()
            truth = data.y[data.train_mask].cpu().numpy()

            mse = metrics.mean_squared_error(truth, pred)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(truth, pred)
            r2 = metrics.r2_score(truth, pred)

            return mse, rmse, mae, r2

        else:  # binary
            out = torch.sigmoid(out_logits)

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



def evaluate(model, data, thq=0.95, task_type='binary'):
    """
    Perform an evaluation on the test data.

    Parameters:
    - model: The model to evaluate.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - thq: Threshold for quantile function.
    - task_type: Type of task ('binary', 'multiclass', 'regression').

    Returns:
    - Binary: (tn, fp, fn, tp, precision, recall, acc, bacc, auc)
    - Multiclass: (precision, recall, acc, bacc, f1, auc)
    - Regression: (mse, rmse, mae, r2)
    """
    model.eval()
    with torch.no_grad():
        out_logits = model(data)

        if task_type == 'multiclass':
            out = F.softmax(out_logits, dim=1)
            pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            truth = data.y[data.test_mask].cpu().numpy()
            prob = out[data.test_mask].cpu().numpy()

            acc = metrics.accuracy_score(truth, pred)
            bacc = metrics.balanced_accuracy_score(truth, pred)
            precision = metrics.precision_score(truth, pred, average='macro', zero_division=0)
            recall = metrics.recall_score(truth, pred, average='macro', zero_division=0)
            f1 = metrics.f1_score(truth, pred, average='macro', zero_division=0)

            try:
                auc = metrics.roc_auc_score(truth, prob, average="macro", multi_class="ovr")
            except:
                auc = 0.0

            return precision, recall, acc, bacc, f1, auc

        elif task_type == 'regression':
            pred = out_logits[data.test_mask].cpu().numpy()
            truth = data.y[data.test_mask].cpu().numpy()

            mse = metrics.mean_squared_error(truth, pred)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(truth, pred)
            r2 = metrics.r2_score(truth, pred)

            return mse, rmse, mae, r2

        else:  # binary
            out = torch.sigmoid(out_logits)

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

def train(model, data, optimizer, pos_weight=None, task_type='binary'):
    """
    Perform a training step.

    Parameters:
    - model: The model to train.
    - data: Data object containing the features, edge indices, labels, and train/test masks.
    - optimizer: The optimizer to use for training.
    - pos_weight: Weight tensor for positive examples in the loss function (binary only).
    - task_type: Type of task ('binary', 'multiclass', 'regression').

    Returns:
    - Loss value.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)

    if task_type == 'multiclass':
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    elif task_type == 'regression':
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    else:
        loss = F.binary_cross_entropy_with_logits(
            out[data.train_mask], data.y[data.train_mask],
            pos_weight=pos_weight if pos_weight is not None else None
        )

    loss.backward()
    optimizer.step()

    return loss

def compute_positive_sample_weight(data, task_type='binary'):
    """
    Compute the weight for positive samples in the data.
    Only applicable for binary classification.

    Parameters:
    - data: The data object containing the labels and the training mask.
    - task_type: Type of task ('binary', 'multiclass', 'regression').

    Returns:
    - The computed weight for positive samples as a tensor, or None for non-binary tasks.
    """
    if task_type != 'binary':
        return None

    num_samples = data.y[data.train_mask].size(0)
    num_positive_samples = data.y[data.train_mask].sum().item()

    if num_positive_samples == 0:
        return torch.tensor([1.0])

    pos_weight = (num_samples - num_positive_samples) / num_positive_samples
    return torch.tensor([pos_weight])

from torch_geometric.utils import to_undirected

def print_network_statistics(data, task_type='binary'):
    """
    Print network statistics of a PyG data object.

    Parameters:
    - data: The data object containing the features, edge indices, labels, and train/test masks.
    - task_type: Type of task ('binary', 'multiclass', 'regression').
    """
    # Calculate basic statistics
    num_nodes = data.num_nodes

    # Get unique undirected edges
    undirected_edges = to_undirected(data.edge_index)
    num_unique_undirected_edges = undirected_edges.size(1) // 2  # Each edge is represented twice

    density = num_unique_undirected_edges / (num_nodes * (num_nodes - 1) / 2)

    # Compute the average degree based on unique undirected edges
    avg_degree = 2 * num_unique_undirected_edges / num_nodes

    # Print statistics
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Unique Undirected Edges: {num_unique_undirected_edges}")
    print(f"Graph Density: {density:.5f}")
    print(f"Average Degree based on Unique Edges: {avg_degree:.5f}")

    if task_type == 'binary':
        num_pos_samples = int(data.y.sum().item())
        num_neg_samples = num_nodes - num_pos_samples
        balance_ratio = num_pos_samples / num_neg_samples if num_neg_samples > 0 else 0.0

        print(f"Number of Positive Samples (labels): {num_pos_samples}")
        print(f"Number of Negative Samples (labels): {num_neg_samples}")
        print(f"Data Balance (Pos/Neg Ratio): {balance_ratio:.5f}")

        # Print data in LaTeX table format
        print(f"\n{num_nodes} & {num_unique_undirected_edges} & {density:.5f} & {avg_degree:.5f} & {num_pos_samples} & {num_neg_samples} & {balance_ratio:.5f}")
    elif task_type == 'multiclass':
        # Count samples per class
        unique_classes, class_counts = torch.unique(data.y, return_counts=True)
        print(f"Number of classes: {len(unique_classes)}")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {int(cls)}: {int(count)} samples")

        # Print data in LaTeX table format
        print(f"\n{num_nodes} & {num_unique_undirected_edges} & {density:.5f} & {avg_degree:.5f} & {len(unique_classes)} classes")
    else:  # regression
        # Show label statistics
        mean_val = data.y.mean().item()
        std_val = data.y.std().item()
        min_val = data.y.min().item()
        max_val = data.y.max().item()

        print(f"Label statistics:")
        print(f"  Mean: {mean_val:.5f}")
        print(f"  Std: {std_val:.5f}")
        print(f"  Min: {min_val:.5f}")
        print(f"  Max: {max_val:.5f}")

        # Print data in LaTeX table format
        print(f"\n{num_nodes} & {num_unique_undirected_edges} & {density:.5f} & {avg_degree:.5f} & [{min_val:.2f}, {max_val:.2f}]")


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
    theta: float = 0.5,
    task_type: str = 'binary'
):
    """
    Train a graph neural network.

    Parameters:
    - gene_filename: Path to the gene file.
    - network_filename: Path to the network file.
    - train_size: Proportion of the dataset to include in the train split.
    - model_name: Name of the model to train.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - weight_decay: Weight decay for the optimizer.
    - eval_threshold: Threshold for quantile function in evaluation.
    - verbose_interval: Interval for printing training progress.
    - task_type: Type of task ('binary', 'multiclass', 'regression').
    """
    # Load the data
    data = load_data(gene_filename, network_filename, train_size, task_type=task_type)

    # Print basic information about the data and model
    print(
        f"# Number of nodes={data.num_nodes}; Number of edges={data.num_edges}; "
        f"Number of node features={data.num_features}; Model: {model_name}."
    )

    # Print network statistics
    # print_network_statistics(data, task_type=task_type)

    # Determine the device to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to device first
    data = data.to(device)

    # Build the model and move it to the appropriate device
    model = build_model(model_name, data, dropout, alpha, theta, task_type=task_type).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Compute the weight for positive samples (binary only)
    # Note: data is already on device, so pos_weight will be created on the same device
    pos_weight = compute_positive_sample_weight(data, task_type=task_type)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    # Print the header for the training progress output
    if task_type == 'regression':
        print(
            "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} ".format(
                "epoch",
                "loss",
                "mse",
                "rmse",
                "mae",
                "r2",
            )
        )
    elif task_type == 'multiclass':
        print(
            "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} ".format(
                "epoch",
                "loss",
                "prec",
                "rec",
                "acc",
                "bacc",
                "f1",
                "auc",
            )
        )
    else:
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

    # instantiate metric array for hyperparameter search
    metric_array = []

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, pos_weight=pos_weight, task_type=task_type)

        # Evaluate the model and print the results at the specified interval
        if (epoch % verbose_interval == 0) or (epoch == 1):
            if task_type == 'multiclass':
                precision, recall, acc, bacc, f1, auc = evaluate(
                    model, data, eval_threshold, task_type=task_type
                )
                precision_train, recall_train, acc_train, bacc_train, f1_train, auc_train = evaluate_train(
                    model, data, eval_threshold, task_type=task_type
                )
                precision_all, recall_all, acc_all, bacc_all, f1_all, auc_all = evaluate_all(
                    model, data, eval_threshold, task_type=task_type
                )

                print(
                    "Test: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, precision, recall, acc, bacc, f1, auc
                    )
                )
                print(
                    "Train: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, precision_train, recall_train, acc_train, bacc_train, f1_train, auc_train
                    )
                )
                print(
                    "All: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, precision_all, recall_all, acc_all, bacc_all, f1_all, auc_all
                    )
                )

                metric_array.append(bacc)

            elif task_type == 'regression':
                mse, rmse, mae, r2 = evaluate(
                    model, data, eval_threshold, task_type=task_type
                )
                mse_train, rmse_train, mae_train, r2_train = evaluate_train(
                    model, data, eval_threshold, task_type=task_type
                )
                mse_all, rmse_all, mae_all, r2_all = evaluate_all(
                    model, data, eval_threshold, task_type=task_type
                )

                print(
                    "Test: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, mse, rmse, mae, r2
                    )
                )
                print(
                    "Train: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, mse_train, rmse_train, mae_train, r2_train
                    )
                )
                print(
                    "All: {:>10} {:>10.5g} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, mse_all, rmse_all, mae_all, r2_all
                    )
                )

                metric_array.append(r2)

            else:  # binary
                tn, fp, fn, tp, precision, recall, acc, bacc, auc = evaluate(
                    model, data, eval_threshold, task_type=task_type
                )
                tn_train, fp_train, fn_train, tp_train, precision_train, recall_train, acc_train, bacc_train, auc_train = evaluate_train(
                    model, data, eval_threshold, task_type=task_type
                )
                tn_all, fp_all, fn_all, tp_all, precision_all, recall_all, acc_all, bacc_all, auc_all = evaluate_all(
                    model, data, eval_threshold, task_type=task_type
                )

                print(
                    "Test: {:>10} {:>10.5g} {:>10} {:>10} {:>10} {:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(
                        epoch, loss, tn, fp, fn, tp, precision, recall, acc, bacc, auc
                    )
                )
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

                metric_array.append(bacc)

    max_metric = max(metric_array)
    return max_metric

if __name__ == "__main__":
    typer.run(run)
    torch.cuda.empty_cache()