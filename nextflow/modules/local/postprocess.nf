process EBBF_POSTPROCESS {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(parquet_files), path(model_dir), path(tma_file)

    output:
    tuple val(meta), path("results"), emit: results
    path "versions.yml"             , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # Copy model directory to make it writable
    cp -rL ${model_dir} model_work

    # Create tx directory and move parquet files
    mkdir -p model_work/tx
    cp ${parquet_files} model_work/tx/

    # Create tma directory with tma.npz
    mkdir -p tma_work
    cp ${tma_file} tma_work/tma.npz

    ebbf-postprocess \\
        --model_directory model_work \\
        --tma_directory tma_work \\
        --cluster_distance ${params.cluster_distance} \\
        ${args}

    # Move results to output
    mv model_work/results .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
    END_VERSIONS
    """
}
