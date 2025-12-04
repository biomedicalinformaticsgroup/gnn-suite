#!/usr/bin/env python3
import sys
import json
import os

if len(sys.argv) != 10:
    print("Usage: load_hyperparams.py MODEL DATASET RESULTS_DIR LR WD DROPOUT ALPHA THETA NUM_HEADS", file=sys.stderr)
    sys.exit(1)

# Arguments passed from main.nf (must match order):
# sys.argv[1] = model (e.g., 'gcn', 'gat', 'gcn2')
# sys.argv[2] = dataSet
# sys.argv[3] = resultsDir
# sys.argv[4] = params.learning_rate
# sys.argv[5] = params.weight_decay
# sys.argv[6] = params.dropout
# sys.argv[7] = params.alpha
# sys.argv[8] = params.theta
# sys.argv[9] = params.num_heads

model = sys.argv[1]
dataset = sys.argv[2]
results_dir = sys.argv[3]

params = {
    'learning_rate': float(sys.argv[4]),
    'weight_decay': float(sys.argv[5]),
    'dropout': float(sys.argv[6]),
    'alpha': float(sys.argv[7]),
    'theta': float(sys.argv[8]),
    'num_heads': int(sys.argv[9])
}

json_path = f"{results_dir}/hyperparameters/{dataset}/best_trial_{model}_{dataset}.json"

if os.path.exists(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        params.update(data.get('best_params', {}))
        print(f"# Using optimized hyperparameters from {json_path}", file=sys.stderr)
    except:
        print(f"# Using default hyperparameters", file=sys.stderr)
else:
    print(f"# Using default hyperparameters", file=sys.stderr)

print(f"LEARNING_RATE={params['learning_rate']}")
print(f"WEIGHT_DECAY={params['weight_decay']}")
print(f"DROPOUT={params['dropout']}")
print(f"ALPHA={params['alpha']}")
print(f"THETA={params['theta']}")
print(f"NUM_HEADS={params['num_heads']}")
