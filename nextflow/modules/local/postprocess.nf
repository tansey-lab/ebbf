process EBBF_POSTPROCESS {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(parquet_files), path(tma_file)

    output:
    tuple val(meta), path("*.parquet"), emit: results
    path "versions.yml"               , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    ebbf-postprocess \\
        ${parquet_files} \\
        --tma ${tma_file} \\
        --cluster_distance ${params.cluster_distance} \\
        --output ${prefix}.parquet \\
        --work_dir . \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
