#!/usr/bin/env python3
import os
import json
import yaml
import torch
import optuna
import typer

from gnn import run


def load_hyperparam_config(config_path="conf/hyperparams.yaml"):
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', config_path)

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {config_path}: {e}")
        print("Using hardcoded defaults")
        return None


def suggest_hyperparameters(trial, model_name, config):
    if config is None:
        return None

    params = {}

    default_params = config.get('default', {})
    model_params = config.get(model_name, {})

    if model_params == 'pass' or model_params is None:
        model_params = {}

    all_params = {**default_params, **model_params}

    for param_name, param_config in all_params.items():
        if param_config is None or param_config == 'pass':
            continue

        param_type = param_config.get('type')

        if param_type == 'float':
            params[param_name] = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=param_config.get('log', False)
            )
        elif param_type == 'int':
            params[param_name] = trial.suggest_int(
                param_name,
                param_config['low'],
                param_config['high'],
                step=param_config.get('step', 1),
                log=param_config.get('log', False)
            )
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config['choices']
            )

    return params


def objective_gnn(trial, model_name, gene_filename, network_filename, num_epochs=300, hyperparam_config=None, task_type='binary'):
    params = suggest_hyperparameters(trial, model_name, hyperparam_config)

    if params is None:
        params = {
            'dropout': trial.suggest_float('dropout', 0.0, 0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True),
            'epochs': num_epochs
        }

    max_metric = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= params.get('epochs', num_epochs),
        learning_rate = params['learning_rate'],
        weight_decay= params['weight_decay'],
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= params['dropout'],
        alpha= params.get('alpha', 0.1),
        theta= params.get('theta', 0.5),
        num_heads= params.get('num_heads', 1),
        task_type=task_type,
        manage_mlflow_run=False,
        trial=trial
    )

    return max_metric

def objective_gcn2(trial, model_name, gene_filename, network_filename, num_epochs=300, hyperparam_config=None, task_type='binary'):
    params = suggest_hyperparameters(trial, model_name, hyperparam_config)

    if params is None:
        params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'theta': trial.suggest_float('theta', 1e-3, 10.0, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True),
            'epochs': num_epochs
        }

    max_metric = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= params.get('epochs', num_epochs),
        learning_rate = params['learning_rate'],
        weight_decay= params['weight_decay'],
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= params['dropout'],
        alpha= params.get('alpha', 0.1),
        theta= params.get('theta', 0.5),
        num_heads= params.get('num_heads', 1),
        task_type=task_type,
        manage_mlflow_run=False,
        trial=trial
    )

    return max_metric


def run_optuna(data_pair, model, task_type='binary', hyperparam_config_path="conf/hyperparams.yaml"):
    gene_filename = data_pair['geneFile']
    network_filename = data_pair['networkFile']
    data_name = data_pair['name']
    model_name = model

    hyperparam_config = load_hyperparam_config(hyperparam_config_path)

    if hyperparam_config and 'optimization' in hyperparam_config:
        n_trials = hyperparam_config['optimization'].get('n_trials', 300)
        n_jobs = hyperparam_config['optimization'].get('n_jobs', -1)
        sampler_name = hyperparam_config['optimization'].get('sampler', 'TPE')
        pruner_name = hyperparam_config['optimization'].get('pruner', 'MedianPruner')
    else:
        n_trials = 300
        n_jobs = -1
        sampler_name = 'TPE'
        pruner_name = 'MedianPruner'

    num_epochs = 250

    if n_jobs == -1 and torch.cuda.is_available():
        n_jobs = 1
        print("Note: Using n_jobs=1 (GPU detected). Set n_jobs explicitly in config to override.")

    sampler = optuna.samplers.TPESampler() if sampler_name == 'TPE' else None

    pruner_map = {
        'MedianPruner': optuna.pruners.MedianPruner,
        'PercentilePruner': optuna.pruners.PercentilePruner,
        'SuccessiveHalvingPruner': optuna.pruners.SuccessiveHalvingPruner,
        'HyperbandPruner': optuna.pruners.HyperbandPruner,
        'ThresholdPruner': optuna.pruners.ThresholdPruner,
        'NopPruner': optuna.pruners.NopPruner,
    }
    pruner_class = pruner_map.get(pruner_name)
    pruner = pruner_class() if pruner_class else None

    print(f"\n{'='*60}")
    print(f"Starting hyperparameter optimization for: {model_name}")
    print(f"Dataset: {data_name}")
    print(f"Task type: {task_type}")
    print(f"Number of trials: {n_trials}")
    print(f"Sampler: {sampler_name}")
    print(f"Pruner: {pruner_name}")
    print(f"Config file: {hyperparam_config_path}")
    print(f"{'='*60}\n")

    direction = "minimize" if task_type == 'regression' else "maximize"
    study = optuna.create_study(
        study_name=model_name+"_hp_search",
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )

    if model_name == "gcn2":
        study.optimize(lambda trial: objective_gcn2(trial,
                                                    model_name,
                                                    gene_filename,
                                                    network_filename,
                                                    num_epochs,
                                                    hyperparam_config,
                                                    task_type),
                                                n_jobs=n_jobs,
                                                n_trials=n_trials)
    else:
        study.optimize(lambda trial: objective_gnn(trial, model_name,
                                                   gene_filename,
                                                    network_filename,
                                                    num_epochs,
                                                    hyperparam_config,
                                                    task_type),
                                                n_jobs=n_jobs,
                                                n_trials=n_trials)

    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print(f"Optimization completed for: {model_name}")
    print(f"{'='*60}")
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    json_output = {
        "model": model_name,
        "dataset": data_name,
        "best_value": best_trial.value,
        "best_params": best_trial.params
    }

    json_filename = f"best_trial_{model_name}_{data_name}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"\nSaved hyperparameters to {json_filename}")


def run_hyperopt(
        gene_filename: str,
        network_filename: str,
        model_name: str,
        data_set: str,
        task_type: str = 'binary'):

    data_pairs = [{'name': data_set,
                   'networkFile': network_filename,
                   'geneFile': gene_filename}]

    models = [model_name]

    for data_pair in data_pairs:
        for model in models:
            print(f"Running hyperparameter optimization for model '{model}' with data pair '{data_pair['name']}'")
            run_optuna(data_pair, model, task_type=task_type)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    typer.run(run_hyperopt)
    

