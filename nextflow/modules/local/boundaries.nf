process EBBF_BOUNDARIES {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(ef_parquet), path(original_transcripts)

    output:
    tuple val(meta), path("cell_boundaries_ebbf.parquet"), emit: boundaries
    path "versions.yml"                                  , emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    ebbf-build-boundaries \\
        --ef_tx ${ef_parquet} \\
        --original_transcripts ${original_transcripts} \\
        --output cell_boundaries_ebbf.parquet \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
