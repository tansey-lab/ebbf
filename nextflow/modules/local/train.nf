process EBBF_TRAIN {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    errorStrategy 'retry'
    maxRetries 3

    input:
    tuple val(meta), path(tma)

    output:
    tuple val(meta), path("model_output"), emit: model
    path "versions.yml"                  , emit: versions

    script:
    def args = task.ext.args ?: ''
    // Build resume command for retry attempts - finds latest checkpoint from previous attempt
    def resume_cmd = ""
    def prevWorkDir = task.previousAttempt?.workDir
    if (task.attempt > 1 && prevWorkDir) {
        resume_cmd = """
        PREV_DIR="${prevWorkDir}"
        LATEST_CKPT="\${PREV_DIR}/model_output/chkpoint.pt"
        if [ -f "\$LATEST_CKPT" ]; then
            RESUME_FLAG="--resume_from \$LATEST_CKPT"
            echo "Resuming from \$LATEST_CKPT (attempt ${task.attempt})"
        fi
        """
    }
    def attn_res = params.attn_resolutions.tokenize(',').collect { it.trim() as int }
    def chan_mult = params.channel_mult.tokenize(',').collect { it.trim() as int }
    def wandb_key = params.wandb_api_key ?: ''
    """
    mkdir -p model_output
    RESUME_FLAG=""
    ${resume_cmd}

    # Generate config from params
    cat > train_config.yaml << 'EOFCONFIG'
experiment_name: ${params.experiment_name}
seed: ${params.seed}

data:
  tma_fname: ${tma}
  patch_size: ${params.patch_size}
  patch_padding: ${params.patch_padding}
  max_tx: ${params.max_tx}
  dataset_size: ${params.dataset_size}
  min_nucs: ${params.min_nucs}
  min_tx_factor: ${params.min_tx_factor}
  max_tx_factor: ${params.max_tx_factor}
  oversample_tx_factor: ${params.oversample_tx_factor}
  max_nuc_dropout_prob: ${params.max_nuc_dropout_prob}
  triangle_dirichlet_scale: ${params.triangle_dirichlet_scale}
  max_cell_expansion_factor: ${params.max_cell_expansion_factor}
  max_noise_prob: ${params.max_noise_prob}
  max_noise_scale: ${params.max_noise_scale}

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
  splat: true

optim:
  optimizer: AdamW
  lr: ${params.lr}
  beta1: 0.9
  beta2: 0.999
  clip_grad: ${params.clip_grad}
  eps: 1.0e-8
  use_scaler: false

train:
  use_ema: true
  ema_warmup_epochs: 100
  save_freq: 1
  batch_size: ${params.batch_size}
  n_epochs: ${params.n_epochs}
  grad_accum_steps: ${params.grad_accum_steps}
  time_skew: ${params.time_skew}
  num_workers: ${params.train_num_workers}
  wandb_enabled: ${params.wandb_enabled}

output_folder: model_output

infer:
  tx_fname: null
  batch_size: ${params.infer_batch_size}
  model_checkpoint: null
  atol: 1.0e-5
  rtol: 1.0e-5
  device: ${params.infer_device}
EOFCONFIG

    # Set wandb API key if provided
    if [ -n "${wandb_key}" ]; then
        export WANDB_API_KEY="${wandb_key}"
    fi

    ebbf-train --config train_config.yaml \$RESUME_FLAG

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
