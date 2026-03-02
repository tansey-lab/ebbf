process EBBF_INFER {
    tag "$meta.id-worker${worker_id}"
    label 'process_high'
    label 'process_gpu'

    input:
    tuple val(meta), path(model_dir), path(transcripts), val(n_workers), val(worker_id)

    output:
    tuple val(meta), path("part-*.parquet"), emit: results
    path "versions.yml"                    , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # Copy model directory to make it writable
    cp -rL ${model_dir} model_work

    ebbf-infer \\
        --model_directory model_work \\
        --transcripts_fname ${transcripts} \\
        --n_workers ${n_workers} \\
        --worker_id ${worker_id} \\
        ${args}

    # Move output parquet to working directory
    mv model_work/tx/*.parquet .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
    END_VERSIONS
    """
}
