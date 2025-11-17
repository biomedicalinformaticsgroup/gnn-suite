mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./results/mlruns \
  --host 0.0.0.0 \
  --port 5000