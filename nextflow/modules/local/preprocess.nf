process EBBF_PREPROCESS {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(zarr_dir)

    output:
    tuple val(meta), path("stdata.npz"), emit: dataset
    path "versions.yml"             , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    ebbf-preprocess \\
        --zarr_dir ${zarr_dir} \\
        --output stdata.npz \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
