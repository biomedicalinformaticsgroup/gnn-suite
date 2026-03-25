# This script is used to run hyperparameter optimization for all models on all datasets
import sys
import torch
import optuna
sys.path.append('models.py')

from gnn import run

def objective_gnn(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize

    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    bacc = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= num_epochs,
        learning_rate = learning_rate,
        weight_decay= weight_decay,
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= dropout
    )
    
    return bacc

def objective_gcn2(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    theta = trial.suggest_float('theta', 1e-3, 10.0, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    bacc = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= num_epochs,
        learning_rate = learning_rate,
        weight_decay= weight_decay,
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= dropout,
        alpha= alpha,
        theta= theta
    )
    
    return bacc


def run_optuna(data_pair, model):
    gene_filename = data_pair['geneFile']
    network_filename = data_pair['networkFile']
    data_name = data_pair['name']
    model_name = model

    num_epochs = 25
    # RENAME FOR DIFFERENT MODELS AND IN OBJECTIVE FUNCTION above
    #model_name = "GCN2"

    
    study = optuna.create_study(study_name=model_name+"_hp_search",
                                direction="maximize")
    

    if model_name == "gcn2":
        study.optimize(lambda trial: objective_gcn2(trial,
                                                    model_name,
                                                    gene_filename,
                                                    network_filename,
                                                    num_epochs),
                                                n_jobs=-1,
                                                n_trials=300)
    else:
        study.optimize(lambda trial: objective_gnn(trial, model_name,
                                                   gene_filename,
                                                    network_filename,
                                                    num_epochs),
                                                n_jobs=-1,
                                                n_trials=10)


    # Print the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best trial to a file
    save_dir = '/home/essharom/code/cancer-gnn-nf/results/hyperparameters'

    # CHANGE FILENAME TO DIFFERENT MODELS
    # Save the best trial to a file
    with open(f'{save_dir}/best_trial_{model_name}_{data_name}.txt', 'w') as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {best_trial.value}\n")
        f.write("  Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")


if __name__ == "__main__":
    data_pairs = [
    {
        'name': 'string',
        'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_stringhc.tsv',
        'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nstringhc_lbailey.csv'
    }#,
    # {
    #     'name': 'string_cosmic',
    #     'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_stringhc.tsv',
    #     'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nstringhc_lcosmic.csv'
    # },
    # {
    #     'name': 'biogrid',
    #     'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_biogridhc.tsv',
    #     'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nbiogridhc_lbailey.csv'
    # },
    # {
    #     'name': 'biogrid_cosmic',
    #     'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_biogridhc.tsv',
    #     'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nbiogridhc_lcosmic.csv'
    # }
    ]

    #models = ["sage", "gin", "gtn", "gcn2", "gcn", "gat", "gat3h", "hgcn", "phgcn"]
    models = ["gcn"]

    for data_pair in data_pairs:
        for model in models:
            print(f"Running hyperparameter optimization for model '{model}' with data pair '{data_pair['name']}'")
            run_optuna(data_pair, model)
            torch.cuda.empty_cache()

