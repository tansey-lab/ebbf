process EBBF_POSTPROCESS {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(parquet_files), path(dataset_file), path(original_transcripts), path(zarr_dir)

    output:
    tuple val(meta), path("*.parquet"), emit: results
    tuple val(meta), path("*.h5ad")   , emit: anndata
    tuple val(meta), path("viz/*.pdf"), emit: visualizations, optional: true
    path "versions.yml"               , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def cd_cutoff = params.cd_cutoff ?: 2.0
    def orig_tx_arg = "--original_transcripts ${original_transcripts}"
    def visualize = params.visualize ? "--visualize --xenium_dir ${zarr_dir} --viz_output_dir viz --viz_patch_size ${params.viz_patch_size}" : ""
    """
    mkdir -p viz

    ebbf-postprocess \\
        ${parquet_files} \\
        --dataset ${dataset_file} \\
        --cluster_distance ${params.cluster_distance} \\
        --output ${prefix}.parquet \\
        --anndata_output ${prefix}.h5ad \\
        --cd_cutoff ${cd_cutoff} \\
        --work_dir . \\
        ${orig_tx_arg} \\
        ${visualize} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
