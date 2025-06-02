# GNNsuite

![](https://img.shields.io/badge/current_version-v0.2.21-blue)


![](https://github.com/stracquadaniolab/gnn-suite/workflows/build/badge.svg)

![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

We design and create a framework for benchmarking and comparing Graph Neural Network (GNN) architectures implemented in a robust and reproducible way using the scientific workflow system Nextflow, popular with computational biologists. We include support for nine different GNN architectures on binary node classification tasks. To demonstrate the versatility of our framework, we consider a task of significant biological importance - that of identifying cancer-driver genes (CDG) in a protein-protein interaction (PPI) network. Data was sourced from the Pan-Cancer Analysis of Whole Genomes (PCAWG), the Pathway Indicated Drivers (PID), the COSMIC Cancer Gene Census (COSMIC-CGC), and STRING and BioGRID PPI databases. On this task, GNNs were able to effectively make use of the network structure of the data. Nevertheless, different architectures performed remarkably similar, emphasising the importance of the quality of the training data for such tasks. We make our pipeline publically available to enable other researchers to perform similar investigations into other areas of computational biology. We believe that this will lead to improved benchmarking standards in the GNN literature.

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

To run the hyperparameter optimization workflow using `optuna` defined in `hyperopt.py`, run the hyperparameter optimization workflow:

```bash
nextflow run main.nf -profile docker,<experiment_file> -entry hyperopt
```

The results of the search will be stored in the `results/hyperparameters/<experiment_file>/` directory. You can find the best trial information in the `best_trial_<model>_<experiment>.txt` file.

For more information on `optuna`, you can visit the official documentation at [https://optuna.readthedocs.io](https://optuna.readthedocs.io).


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


