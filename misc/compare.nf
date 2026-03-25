// nextflow run compare.nf --numEpochs 100
// Define the process to list the data files and extract the model names

// TODO: Debug the process to extract the model names and create the comparison plots

// enabling nextflow DSL v2
nextflow.enable.dsl=2

// defaults results directory
// nextflow run compare.nf --numEpochs 100

println "do i work?"

// Define the process to list the data files and extract the model names
process ExtractModelInfo {
    tag "extract-model-names"

    input:
    folder 'results/data/*'
    
    output:
    file 'results/comparison/models.txt'
    file 'results/comparison/model_names.txt'
    
    script:
    """
    find results/data -name '*.txt' | awk -F '/' '{print \$3}' | awk -F '-' '{print \$2}' | sort | uniq > results/comparison/models.txt
    find results/data -name '*.txt' | awk -F '/' '{print \$3}' | awk -F '-' '{print \$2}' | awk -F '.' '{print substr(\$1, 1, length(\$1)-4)}' > results/comparison/model_names.txt
    """
}

// Define the process to run the comparison script
process CompareMetrics {
    tag "compare-metrics"
    
    input:
    file linePlotFile
    file boxPlotFile
    file violinPlotFile
    file modelFile
    file modelNamesFile
    
    output:
    file "*"
    
    script:
    """
    python compare_models.py --line_plot ${linePlotFile} --box_plot ${boxPlotFile} --violin_plot ${violinPlotFile} \
        --data_files ${modelFile} --model_names ${modelNamesFile} --num_epochs ${params.numEpochs}
    """
}

// Define the workflow
workflow {
    params.resultsDir = params.resultsDir ? params.resultsDir : "${workflow.launchDir}/results"
    
    // Extract model names process
    ExtractModelInfo
    
    // Input files
    linePlotFile = file("${params.resultsDir}/comparison/line_plot.pdf")
    boxPlotFile = file("${params.resultsDir}/comparison/box_plot.pdf")
    violinPlotFile = file("${params.resultsDir}/comparison/violin_plot.pdf")
    modelFile = file("${params.resultsDir}/comparison/models.txt")
    modelNamesFile = file("${params.resultsDir}/comparison/model_names.txt")
    
    // Run the comparison process
    CompareMetrics(linePlotFile, boxPlotFile, violinPlotFile, modelFile, modelNamesFile)
}
