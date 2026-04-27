process EBBF_INFER {
    tag "$meta.id-worker${worker_id}"
    label 'process_high'
    label 'process_gpu'

    input:
    tuple val(meta), path(weights), path(dataset), path(zarr_dir), val(n_workers), val(worker_id)

    output:
    tuple val(meta), path("part-*.parquet"), emit: results
    path "versions.yml"                    , emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def attn_res = params.attn_resolutions.tokenize(',').collect { it.trim() as int }
    def chan_mult = params.channel_mult.tokenize(',').collect { it.trim() as int }
    """
    # Generate config from params
    cat > infer_config.yaml << 'EOFCONFIG'
data:
  patch_size: ${params.patch_size}
  patch_padding: ${params.patch_padding}
  max_tx: ${params.max_tx}

model:
  conditioned: nuc
  img_resolution: ${params.img_resolution}
  in_channels: ${params.in_channels}
  model_channels: ${params.model_channels}
  out_channels: ${params.out_channels}
  num_blocks: ${params.num_blocks}
  attn_resolutions: ${attn_res}
  dropout: ${params.dropout}
  channel_mult: ${chan_mult}
  channel_mult_emb: ${params.channel_mult_emb}
  channels_per_head: ${params.channels_per_head}
  nuc_channel_frac: ${params.nuc_channel_frac}
  pred_noise: ${params.pred_noise}
  splat: true

train:
  use_ema: true
  batch_size: ${params.batch_size}

infer:
  device: ${params.infer_device}
  atol: ${params.infer_atol}
  rtol: ${params.infer_rtol}
EOFCONFIG

    ebbf-infer \\
        --config infer_config.yaml \\
        --weights ${weights} \\
        --dataset ${dataset} \\
        --zarr_dir ${zarr_dir} \\
        --n_workers ${n_workers} \\
        --worker_id ${worker_id} \\
        --output_dir . \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
