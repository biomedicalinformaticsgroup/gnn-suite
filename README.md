# GNN-suite

![](https://img.shields.io/badge/current_version-v0.2.21-blue)


![](https://github.com/stracquadaniolab/gnn-suite/workflows/build/badge.svg)

![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

GNN-Suite is a robust and modular framework for constructing and benchmarking Graph Neural Network (GNN) architectures for computational biology applications. Built on the [Nextflow](https://www.nextflow.io/) scientific workflow system, it standardises experimentation and reproducibility for evaluating GNN model performance.

The framework supports **binary classification**, **multi-class classification**, and **regression** tasks, and integrates [MLflow](https://mlflow.org/) for experiment tracking, metric logging, and model versioning. Automated hyperparameter optimisation is provided via [Optuna](https://optuna.org/) with YAML-based configuration of search spaces, samplers, and pruners.

To illustrate its utility, we applied the system to the identification of cancer-driver genes by constructing molecular networks from protein-protein interaction (PPI) data sourced from STRING and BioGRID and annotating nodes with features extracted from PCAWG, PID, and COSMIC-CGC repositories. Our experiments showed that all GNN architectures consistently outperformed a baseline logistic regression model, with GCN2 achieving the highest balanced accuracy (0.807 ± 0.035) on a STRING-based network.

# Paper

If you use **GNN-Suite** in your research, please cite the following preprint:

> **GNN-Suite: a Graph Neural Network Benchmarking Framework for Biomedical Informatics**  
> Sebestyén Kamp, Giovanni Stracquadanio, T. Ian Simpson  
> [arXiv:2505.10711](https://arxiv.org/abs/2505.10711), 2025  
> DOI: [10.48550/arXiv.2505.10711](https://doi.org/10.48550/arXiv.2505.10711)

```bibtex
@article{kamp2025gnnsuite,
  title     = {GNN-Suite: a Graph Neural Network Benchmarking Framework for Biomedical Informatics},
  author    = {Kamp, Sebestyén and Stracquadanio, Giovanni and Simpson, T. Ian},
  journal   = {arXiv preprint arXiv:2505.10711},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.10711},
  doi       = {10.48550/arXiv.2505.10711}
}
```


## Models

The following models are included:

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT) 
- Hierarchical Graph Convolutional Networks (HGCN)
- Parallel Hierarchical Graph Convolutional Networks (PHGCN)
- Graph SAmpling and aggreGatE (GraphSAGE) 
- Graph Transformer Networks (GTN) 
- Graph Isomorphism Networks (GIN)
- Graph Convolutional Networks II (GCNII)

## Task Types

The pipeline supports three types of node classification/prediction tasks:

- **Binary Classification** (`binary`): Two class classification
- **Multiclass Classification** (`multiclass`): Multi class classification with more than two classes
- **Regression** (`regression`): Continuous value prediction

Specify the task type using the `--task_type` parameter:

```bash
# Binary classification
nextflow run main.nf -profile docker,test --task_type binary

# Multiclass classification
nextflow run main.nf -profile docker \
  --geneFile testdata/multiclass.csv \
  --networkFile testdata/multiclass_network.tsv \
  --task_type multiclass

# Regression
nextflow run main.nf -profile docker \
  --geneFile testdata/regression.csv \
  --networkFile testdata/regression_network.tsv \
  --task_type regression
```

## Architecture


![New Architecture](assets/nextflow_pipeline_with_bg.png)


## Running the workflow

### Install or update the workflow

```bash
nextflow pull stracquadaniolab/gnn-suite
```

### Run a test

```bash
nextflow run stracquadaniolab/gnn-suite -profile docker,test
```

### Run an experiment

```bash
nextflow run stracquadaniolab/gnn-suite -profile docker,<experiment_file>
```
The results of the experimetn will be stored in the `results/data/<experiment_file>/` and `results/figures/<experiment_file>/` directory.

For more information on `Nextflow`, you can visit the official documentation at [nextflow.io/docs](https://www.nextflow.io/docs/latest/index.html).


## Parameters Reference

### Data Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--geneFile` | null | Path to gene features CSV file |
| `--networkFile` | null | Path to network edges TSV file |
| `--resultsDir` | null | Results directory (defaults to `./results`) |
| `--dataSet` | null | Dataset name (auto generated from geneFile if not provided genefile_tasktype) |

### Experimental Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--models` | `["gcn2", "gcn", "gat"]` | List of GNN models to train |
| `--epochs` | `[100]` | Number of training epochs |
| `--learning_rate` | `0.01` | Learning rate |
| `--weight_decay` | `1e-4` | Weight decay for regularization |
| `--train_size` | `0.8` | Train/test split ratio |
| `--replicates` | `1` | Number of experiment replicates |
| `--verbose_interval` | `10` | Logging interval (epochs) |
| `--dropout` | `0.2` | Dropout rate |
| `--alpha` | `0.1` | Alpha parameter (GCNII) |
| `--theta` | `0` | Theta parameter (GCNII) |
| `--num_heads` | `1` | Number of attention heads (GAT) |
| `--task_type` | `binary` | Task type: `binary`, `multiclass`, `regression` |

### Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--enable_plots` | `false` | Generate PDF plots |


## Output Description

The `--dataSet` parameter determines the output folder name. If not provided, it defaults to `<geneFile_basename>_<task_type>` (e.g., `genes_binary`, `genes_multiclass`).

After running the pipeline, results are organized in the following structure:

```
results/
├── data/<dataSet>/
│   ├── full-<model>-<epochs>-run-<replicate>-<task_type>.txt        # Full training output
│   ├── full-<model>-<epochs>-run-<replicate>-<task_type>-train.txt  # Training set metrics
│   ├── full-<model>-<epochs>-run-<replicate>-<task_type>-test.txt   # Test set metrics
│   └── full-<model>-<epochs>-run-<replicate>-<task_type>-all.txt    # All nodes metrics
├── hyperparameters/<dataSet>/
│   └── best_trial_<model>_<dataSet>.json    # Optimized hyperparameters from hyperopt
└── figures/<dataSet>/
    └── <model>-<epochs>-split-<split>-<task_type>.pdf   # Training plots (if enabled)
```



## Docker Image
 
[View the `gnn-suite` Docker image on GitHub Container Registry](https://github.com/orgs/stracquadaniolab/packages/container/package/gnn-suite), you can also download it using:

```bash
docker pull ghcr.io/stracquadaniolab/gnn-suite:latest
```

## Adding a New Experiment

1. **Create a Config File**: Create a new configuration file `<experiment_file>.config` with the parameters for the experiment:
    ```groovy
    // profile to test the string workflow
    params {
      resultsDir = "${baseDir}/results/"
      networkFile = "${baseDir}/data/<network_file>.tsv"
      geneFile = "${baseDir}/data/<feature_file>.csv"
      epochs = [300]
      models = ["gcn2", "gcn", "gat", "gat3h", "hgcn", "phgcn", "sage", "gin", "gtn"]
      replicates = 10
      verbose_interval = 1
      dropout = 0.2
      alpha = 0.1
      theta = 1
      dataSet = "<experiment_file_tag>"
    }
    ```

2. **Update `base.config`**: Add a new profile for your experiment in `base.config`:
    ```groovy
    profiles {
      // existing profiles...

      // test profile for the biogrid cosmic network defining some data
      <config_file> {
        includeConfig '<experiment_file>.config'
      }
    }
    ```

3. **Run the Experiment**: Execute the pipeline with the new profile using:
    ```bash
    nextflow run main.nf -profile docker, <experiment_file>
    ```

    or
    ```bash
    nextflow run stracquadaniolab/gnn-suite -profile docker,<experiment_file>
    ```

## Adding a New Model

1. **Create Model**: Implement the new model class in `models.py`:
    ```python
    class NewModel(torch.nn.Module):
        def __init__(self, num_features, num_classes, num_hidden=16, num_layers=2, dropout=0.5):
            super(NewModel, self).__init__()
            # Define layers
        def forward(self, data):
            # Define forward pass
    ```

2. **Import Model**: Add your model to the imports in `gnn.py`:
    ```python
    from models import GCN, GAT, ..., NewModel
    ```

3. **Update `build_model`**: Add your model to the `build_model` function in `gnn.py`:
    ```python
    elif name == "new_model":
        return NewModel(num_features, num_classes, dropout=dropout)
    ```

4. **Include in Experiment**: Add the new model name to the `models` list in your experiment config (`<experiment_file>.config`):
    ```groovy
    models = ["gcn", "gat", ..., "new_model"]
    ```

## Hyperparameter Optimization with Optuna

To run the hyperparameter optimization workflow using `optuna` defined in `hyperopt.py`:

```bash
# Binary classification
nextflow run main.nf -profile docker -entry hyperopt \
  --geneFile testdata/genes.csv \
  --networkFile testdata/network.tsv \
  --task_type binary

# Multiclass classification
nextflow run main.nf -profile docker -entry hyperopt \
  --geneFile testdata/genes.csv \
  --networkFile testdata/multiclass_network.tsv \
  --task_type multiclass

# Regression
nextflow run main.nf -profile docker -entry hyperopt \
  --geneFile testdata/genes.csv \
  --networkFile testdata/regression_network.tsv \
  --task_type regression
```

### Search Space Configuration

The hyperparameter search space is defined in `conf/hyperparams.yaml`:

| Parameter | Range | Type |
|-----------|-------|------|
| `learning_rate` | 0.001 - 0.5 | float (log scale) |
| `weight_decay` | 0.00001 - 0.5 | float (log scale) |
| `dropout` | 0.0 - 0.8 | float |
| `epochs` | 100 - 300 | int |

### Model-Specific Parameters

| Model | Parameter | Range |
|-------|-----------|-------|
| GCNII | `alpha` | 0.001 - 10.0 (log scale) |
| GCNII | `theta` | 0.001 - 10.0 (log scale) |
| GAT | `num_heads` | [1, 2, 4, 8] |

### Optimization Settings

| Setting | Value |
|---------|-------|
| `n_trials` | 5 |
| `sampler` | TPE |

### Output

The results are stored in `results/hyperparameters/<dataSet>/`:
- `best_trial_<model>_<dataSet>.json` - Best hyperparameters (auto used during training)

When running the main training workflow, optimized hyperparameters are automatically used if available.


## MLflow Integration

The pipeline supports [MLflow](https://mlflow.org/) for experiment tracking, metrics logging, and model registry.

### Enabling MLflow

```bash
# Using SQLite database
nextflow run main.nf -profile docker,test \
  --with_mlflow true \
  --mlflow_tracking_uri "sqlite:///mlflow.db"
```

### MLflow Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--with_mlflow` | `false` | Enable MLflow tracking |
| `--mlflow_tracking_uri` | `file:./mlruns` | MLflow tracking URI (`sqlite:///mlflow.db` or `http://localhost:5000`) |
| `--mlflow_experiment_name` | `gnn-suite-default` | Experiment name in MLflow |
| `--mlflow_register_model` | `false` | Register trained models in MLflow Model Registry |

### Using a Tracking Server

1. Start the MLflow server using the provided script:
    ```bash
    bash mlflow_server.sh
    ```
    This starts a server with SQLite backend (`sqlite:///mlflow.db`) and artifact storage (`./mlflow-artifacts`).

2. Run the pipeline with the server URI:
    ```bash
    nextflow run main.nf -profile docker,test \
      --with_mlflow true \
      --mlflow_tracking_uri "http://localhost:5000"
    ```

3. View experiments at `http://localhost:5000`

### Logged Metrics

MLflow logs the following metrics per epoch:
- **Binary/Multiclass**: precision, recall, accuracy, balanced accuracy, F1, AUC
- **Regression**: MSE, RMSE, MAE, R²


## FAQ
If you encounter the following error message when attempting to execute the script:

```groovy
Command error:
  .command.sh: line 2: ../gnn-suite/bin/plot.py: Permission denied
```
You need to grant the necessary execution permissions to the specific python scripts. You can do this by running (e.g. `plot.py`):
```groovy
 chmod +x /home/<path>/code/gnn-suite/bin/plot.py
```

## Authors

- Sebestyén Kamp
- Ian Simpson
- Giovanni Stracquadanio


