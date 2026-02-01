// enabling nextflow DSL v2
nextflow.enable.dsl=2

resultsDir = params.resultsDir ? params.resultsDir : "${workflow.launchDir}/results"
dataSet = params.dataSet ? params.dataSet : "${file(params.geneFile).baseName}_${params.task_type}"

hparamDir = file("${resultsDir}/hyperparameters/${dataSet}")
hparamStatus = hparamDir.exists() ? "Using optimized (from ${hparamDir})" : "Using defaults"

println ""
println "Gene file: ${params.geneFile}"
println "Network file: ${params.networkFile}"
println "Hyperparameters: ${hparamStatus}"
println "Results dir: ${resultsDir}"
println "Data set: ${dataSet}"
println "Task type: ${params.task_type}"
println "Training set size: ${params.train_size}"
println "Models: ${params.models}"
println "Learning rate: ${params.learning_rate}"
println "Weight decay: ${params.weight_decay}"
println "Epochs: ${params.epochs}"
println "Replicates: ${params.replicates}"
println "Metrics: ${params.metrics}"
println "Eval threshold: ${params.eval_threshold}"
println "Verbose interval: ${params.verbose_interval}"
println ""

process TrainGNN {
    tag "${model}-${epoch}-${params.task_type}"

    publishDir "${resultsDir}/data/${dataSet}", pattern: "full-${model}-${epoch}-run-${run}-${params.task_type}*.txt", mode: 'copy'

    input:
        tuple path(geneFile), path(networkFile), val(model), val(epoch)
        each run

    output:
        tuple val(model), val(epoch), path("full-${model}-${epoch}-run-${run}-${params.task_type}.txt"), emit: full_output
        tuple val(model), val(epoch), val("train"), path("full-${model}-${epoch}-run-${run}-${params.task_type}-train.txt"), emit: train_output
        tuple val(model), val(epoch), val("test"), path("full-${model}-${epoch}-run-${run}-${params.task_type}-test.txt"), emit: test_output
        tuple val(model), val(epoch), val("all"), path("full-${model}-${epoch}-run-${run}-${params.task_type}-all.txt"), emit: all_output

    """
        export WITH_MLFLOW="${params.with_mlflow ? '1' : '0'}"
        export MLFLOW_TRACKING_URI="${params.mlflow_tracking_uri}"
        export MLFLOW_EXPERIMENT_NAME="${params.mlflow_experiment_name}"
        export MLFLOW_REGISTER_MODEL="${params.mlflow_register_model ? '1' : '0'}"
        export DATASET="${dataSet}"
        export REPLICATE="${run}"

        # Load model-specific hyperparameters (or use defaults)
        eval \$(load_hyperparams.py ${model} ${dataSet} ${resultsDir} \
            ${params.learning_rate} \
            ${params.weight_decay} \
            ${params.dropout} \
            ${params.alpha} \
            ${params.theta} \
            ${params.num_heads})

        echo "# Using hyperparameters for ${model}: LR=\${LEARNING_RATE}, WD=\${WEIGHT_DECAY}, DROPOUT=\${DROPOUT}, ALPHA=\${ALPHA}, THETA=\${THETA}, NUM_HEADS=\${NUM_HEADS}" >&2

        gnn.py ${geneFile} ${networkFile} \
                --train-size ${params.train_size} \
                --model-name ${model} \
                --learning-rate \${LEARNING_RATE} \
                --weight-decay \${WEIGHT_DECAY} \
                --epochs ${epoch} \
                --eval-threshold ${params.eval_threshold} \
                --verbose-interval ${params.verbose_interval} \
                --dropout \${DROPOUT} \
                --alpha \${ALPHA} \
                --theta \${THETA} \
                --num-heads \${NUM_HEADS} \
                --task-type ${params.task_type} \
                 > full-${model}-${epoch}-run-${run}-${params.task_type}.txt

        split_data.py full-${model}-${epoch}-run-${run}-${params.task_type}.txt
    """
}



process PlotEpochMetrics {
    tag "${model}-${epoch}-${split}-${params.task_type}"

    publishDir "${resultsDir}/figures/${dataSet}", pattern: "${model}-${epoch}-split-${split}-${params.task_type}*.pdf", mode: 'copy'

    input:
        tuple val(model), val(epoch), val(split), path(files)


    output:
        path "${model}-${epoch}-split-${split}-${params.task_type}.pdf"

    script:
    """
        plot.py --model ${model} --task-type ${params.task_type} ${model}-${epoch}-split-${split}-${params.task_type}.pdf ${files}
    """
}


process ComputeStats {
    tag "${model}-${params.task_type}"

    input:
        tuple val(model), path(results,stageAs: "?/*")

    output:
        path "stats-${model}-${params.task_type}.txt"

    """
        stats.py compute stats-${model}-${params.task_type}.txt ${results} ${model}
    """
}

process CollectStats {
    tag "Final stats-${params.task_type}"

    publishDir "${resultsDir}", pattern: "stats-${params.task_type}.tex", mode: 'copy'

    input:
        path results

    output:
        path "stats-${params.task_type}.tex"

    """
        # collect stats
        stats.py collect stats-${params.task_type}.tex ${results}
    """
}

process HyperparameterOptimization {

    tag "${dataSet}-${model}-${params.task_type}"

    publishDir "${resultsDir}/hyperparameters/${dataSet}", pattern: "best_trial_${model}_${dataSet}.*", mode: 'copy'

    input:
        tuple path(geneFile),
        path(networkFile),
        val(model),
        val(dataSet)

    output:
        path "best_trial_${model}_${dataSet}_${params.task_type}.txt", emit: best_trial_output
        path "best_trial_${model}_${dataSet}.json", emit: best_trial_json

    """
        echo "Starting hyperparameter optimization for ${model} on ${dataSet}..."

        hyperopt.py ${geneFile} ${networkFile} \
            ${model} \
            ${dataSet} \
            --task-type ${params.task_type} \
            2>&1 | tee best_trial_${model}_${dataSet}_${params.task_type}.txt

        echo "Completed hyperparameter optimization for ${model}"
        clean_hparams.py best_trial_${model}_${dataSet}_${params.task_type}.txt
    """
}

workflow hyperopt {
    geneChan = channel.fromPath(params.geneFile)
    networkChan = channel.fromPath(params.networkFile)
    modelChan = channel.from(params.models)
    dataSetChan = channel.value(dataSet)
    hparamChan = geneChan.combine(networkChan).combine(modelChan).combine(dataSetChan)

    hparams = HyperparameterOptimization(
        hparamChan
    )
    hparams.best_trial_output.view()
}



workflow {
    // building channels for experiments
    geneChan = channel.fromPath(params.geneFile)
    networkChan = channel.fromPath(params.networkFile)
    modelChan = channel.from(params.models)
    epochChan = channel.from(params.epochs)
    replicatesChan = channel.of(1..params.replicates)
    metricsChan = channel.from(params.metrics)
    
    // benchmark channel
    benchmarkChan = geneChan.combine(networkChan).combine(modelChan).combine(epochChan) 
    
    // train the GNN
    results = TrainGNN(benchmarkChan, replicatesChan)
    //results.train_output.view()

    // group the splitDataResults by model and epoch
    groupedResult_train = results.train_output
        .groupTuple(by: [0,1, 2]) // group by model and epoch: [model, epoch, list(runs), list(train-files), list(test-files)]
    groupedResult_test = results.test_output
        .groupTuple(by: [0,1, 2]) 
    groupedResult_all = results.all_output
        .groupTuple(by: [0,1, 2]) 
    
    groupedResult = groupedResult_train.concat(groupedResult_test, groupedResult_all)
    groupedResult.view()

    // plot the results
    if (params.enable_plots) {
        PlotEpochMetrics(groupedResult).view()
    }
}

