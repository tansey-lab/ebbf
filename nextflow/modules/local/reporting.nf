process EBBF_REPORTING {
    tag "$meta.id"
    label 'process_high'

    input:
    tuple val(meta), path(anndata_file), path(combined_boundaries), path(zarr_dir)

    output:
    tuple val(meta), path("reports/*.html") , emit: reports
    tuple val(meta), path("reports/*.h5ad") , emit: diff_anndata, optional: true
    tuple val(meta), path("reports/*.pdf")  , emit: visualizations, optional: true
    path "versions.yml"                     , emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p reports

    ebbf-reporting \\
        ${anndata_file} \\
        --xenium_dir ${zarr_dir} \\
        --output_dir reports \\
        --combined_boundaries ${combined_boundaries} \\
        --viz_patch_size ${params.viz_patch_size} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
