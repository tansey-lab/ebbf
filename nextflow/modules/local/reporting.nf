process EBBF_REPORTING {
    tag "$meta.id"
    label 'process_high'

    input:
    tuple val(meta), path(anndata_file), path(anndata_unfiltered), path(ebbf_boundaries), path(zarr_dir), path(stdata)

    output:
    tuple val(meta), path("*.html")               , emit: reports
    tuple val(meta), path("ellipse_clusters.html"), emit: ellipse_clusters
    // narrow to the specific output filename — the staged input ebbf.h5ad lives
    // in the same workdir and would otherwise match a `*.h5ad` glob.
    tuple val(meta), path("adata_diff.h5ad"), emit: diff_anndata, optional: true
    tuple val(meta), path("*.pdf")          , emit: visualizations, optional: true
    path "versions.yml"                     , emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    ebbf-reporting \\
        ${anndata_file} \\
        --xenium_dir ${zarr_dir} \\
        --output_dir . \\
        --ebbf_boundaries ${ebbf_boundaries} \\
        --ebbf_unfiltered ${anndata_unfiltered} \\
        --viz_patch_size ${params.viz_patch_size} \\
        --stdata ${stdata} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
