import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'String': {
        'means': [0.769, 0.761, 0.777, 0.807, 0.796, 0.798, 0.802, 0.794, 0.796],
        'stds': [0.021, 0.033, 0.04, 0.035, 0.032, 0.036, 0.028, 0.032, 0.032]
    },
    'String Cosmic': {
        'means': [0.65, 0.664, 0.662, 0.68, 0.665, 0.674, 0.671, 0.678, 0.666],
        'stds': [0.021, 0.03, 0.033, 0.03, 0.031, 0.02, 0.02, 0.024, 0.022]
    },
    'Biogrid': {
        'means': [0.77, 0.751, 0.78, 0.759, 0.754, 0.75, 0.786, 0.783, 0.772],
        'stds': [0.046, 0.037, 0.041, 0.047, 0.031, 0.045, 0.04, 0.03, 0.03]
    },
    'Biogrid Cosmic': {
        'means': [0.629, 0.646, 0.662, 0.677, 0.677, 0.655, 0.648, 0.652, 0.651],
        'stds': [0.024, 0.019, 0.019, 0.027, 0.027, 0.023, 0.014, 0.025, 0.035]
    }
}

for dataset in data:
    sems = [std / np.sqrt(10) for std in data[dataset]['stds']]
    data[dataset]['sems'] = sems

print(pd.DataFrame(data))

models = [str.upper(i) for i in['gat', 'gat3h', 'gcn', 'gcn2', 'gin', 'gtn', 'hgcn', 'phgcn', 'sage']]
datasets = ['String', 'String Cosmic', 'Biogrid', 'Biogrid Cosmic']

# Ranking models by performance
def rank_models_by_performance(models, data, datasets):
    scores = {model: 0 for model in models}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model_A = models[i]
            model_B = models[j]
            model_A_means = [data[dataset]['means'][i] for dataset in datasets]
            model_B_means = [data[dataset]['means'][j] for dataset in datasets]
            for k in range(len(datasets)):
                if model_A_means[k] > model_B_means[k]:
                    scores[model_A] += 1
                elif model_A_means[k] < model_B_means[k]:
                    scores[model_B] += 1
    ranked_models = sorted(scores, key=scores.get, reverse=True)
    return ranked_models

ordered_models = rank_models_by_performance(models, data, datasets)

# Plotting

width = 0.2
all_positions = [np.arange(len(ordered_models))]
for i in range(1, len(datasets)):
    pos = [x + width for x in all_positions[-1]]
    all_positions.append(pos)

# Get the colors from Seaborn's "Set2" palette
colors = sns.color_palette("Set2", len(datasets))

# Plotting with the new color scheme
fig, ax = plt.subplots(figsize=(14, 8))

for i, dataset in enumerate(datasets):
    means = [data[dataset]['means'][models.index(model)] for model in ordered_models]
    stds = [data[dataset]['sems'][models.index(model)] for model in ordered_models]
    ax.bar(all_positions[i], means, yerr=stds, width=width, label=str.upper(dataset), alpha=0.75, capsize=7, color=colors[i])

ax.set_xlabel('Model', fontweight='bold', fontsize=15)
ax.set_ylabel('Balanced Accuracy', fontweight='bold', fontsize=15)
ax.set_title('Comparison of GNN Models Across Datasets', fontweight='bold', fontsize=16)
ax.set_xticks([r + 1.5*width for r in range(len(ordered_models))])
ax.set_xticklabels(ordered_models)
ax.set_ylim(0.5, 0.85)
ax.legend()
plt.tight_layout()
plt.savefig('/home/essharom/code/cancer-gnn-nf/results/comparison/comparison.pdf')
plt.show()

