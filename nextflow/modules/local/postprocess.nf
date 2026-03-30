process EBBF_POSTPROCESS {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(parquet_files), path(dataset_file)

    output:
    tuple val(meta), path("*.parquet"), emit: results
    tuple val(meta), path("*.h5ad")   , emit: anndata
    path "versions.yml"               , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def cd_cutoff = params.cd_cutoff ?: 2.0
    """
    ebbf-postprocess \\
        ${parquet_files} \\
        --dataset ${dataset_file} \\
        --cluster_distance ${params.cluster_distance} \\
        --output ${prefix}.parquet \\
        --anndata_output ${prefix}.h5ad \\
        --cd_cutoff ${cd_cutoff} \\
        --work_dir . \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
