# MLflow Integration - Quick Start Guide

## Commands Used

### 1. Git Operations
```bash
git branch -a                    # List all branches (local and remote)
git checkout feature/mlflow      # Switch to the feature/mlflow branch
```

### 2. Environment Setup
```bash
conda info --envs               # List available conda environments
conda activate asd              # Activate the 'asd' conda environment
```

### 3. Start MLflow Server
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### 4. Run the Pipeline with MLflow Tracking
```bash
nextflow run main.nf \
  --geneFile 'testdata/sfari_multiclass_genes.csv' \
  --networkFile 'testdata/sfari_multiclass_network.tsv' \
  --task_type 'multiclass' \
  --with_mlflow true \
  --mlflow_tracking_uri 'http://localhost:5000' \
  --mlflow_experiment_name 'multiclass-experiment'
```

## Notes

- MLflow UI is accessible at `http://localhost:5000` after starting the server
- Artifacts are stored in `./mlflow-artifacts`
- Experiment data is stored in `sqlite:///mlflow.db`
