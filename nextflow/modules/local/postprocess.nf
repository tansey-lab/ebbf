process EBBF_POSTPROCESS_PREPARE {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(parquet_files), path(dataset_file)

    output:
    tuple val(meta), path("*.step4.parquet"), emit: step4
    tuple val(meta), path("*.noise_meta.parquet"), emit: noise_meta
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def n_workers_arg = params.n_postprocess_workers ? "--n_workers ${params.n_postprocess_workers}" : ""
    """
    ebbf-postprocess \\
        prepare \\
        ${parquet_files} \\
        --dataset ${dataset_file} \\
        --cluster_distance ${params.cluster_distance} \\
        --output ${prefix}.step4.parquet \\
        --noise_meta_output ${prefix}.noise_meta.parquet \\
        ${n_workers_arg} \\
        --work_dir . \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}

process EBBF_POSTPROCESS_MAP {
    tag "$meta.id-worker${worker_id}"
    label 'process_medium'

    input:
    tuple val(meta), path(step4_parquet), path(dataset_file), val(n_workers), val(worker_id)

    output:
    tuple val(meta), path("map-*.parquet"), emit: mapped
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    ebbf-postprocess \\
        map \\
        --dataset ${dataset_file} \\
        --input_step4 ${step4_parquet} \\
        --n_workers ${n_workers} \\
        --worker_id ${worker_id} \\
        --output map-${prefix}-worker-${worker_id}.parquet \\
        --work_dir . \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}

process EBBF_POSTPROCESS_MERGE {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(mapped_parts), path(dataset_file), path(original_transcripts), path(zarr_dir), path(noise_meta)

    output:
    tuple val(meta), path("${prefix}.parquet"), emit: results
    tuple val(meta), path("${prefix}.h5ad"), emit: anndata
    tuple val(meta), path("${prefix}.denovo_filter_report.parquet"), emit: denovo_filter_report
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    def cd_cutoff = params.cd_cutoff ?: 2.0
    def orig_tx_arg = "--original_transcripts ${original_transcripts}"
    def noise_meta_arg = "--noise_meta ${noise_meta}"
    """
    ebbf-postprocess \\
        merge \\
        ${mapped_parts} \\
        --dataset ${dataset_file} \\
        --output ${prefix}.parquet \\
        --anndata_output ${prefix}.h5ad \\
        --denovo_filter_report ${prefix}.denovo_filter_report.parquet \\
        --cd_cutoff ${cd_cutoff} \\
        --work_dir . \\
        ${orig_tx_arg} \\
        ${noise_meta_arg} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
