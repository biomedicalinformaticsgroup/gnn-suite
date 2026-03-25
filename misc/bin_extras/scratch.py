# temporary way to run a model for testing purposes

import sys
sys.path.append('models.py')

from gnn import run


data_pairs = [
    {
        'name': 'string',
        'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_stringhc.tsv',
        'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nstringhc_lbailey.csv'
    },
    {
        'name': 'string_cosmic',
        'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_stringhc.tsv',
        'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nstringhc_lcosmic.csv'
    },
    {
        'name': 'biogrid',
        'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_biogridhc.tsv',
        'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nbiogridhc_lbailey.csv'
    },
    {
        'name': 'biogrid_cosmic',
        'networkFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_biogridhc.tsv',
        'geneFile': '/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nbiogridhc_lcosmic.csv'
    }
]


if __name__ == "__main__":
    run(
        gene_filename ="/home/essharom/code/cancer-gnn-nf/data/entrez_fpancanall_nbiogridhc_lcosmic.csv",
        network_filename = "/home/essharom/code/cancer-gnn-nf/data/entrez_biogridhc.tsv",
        train_size = 1,
        model_name= "gcn",
        epochs= 200,
        learning_rate = 0.01,
        weight_decay= 1e-4,
        eval_threshold= 0.9,
        verbose_interval= 1   
    )