process EBBF_TRAIN {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    errorStrategy 'retry'
    maxRetries 10

    input:
    // resume_weights is optional: pass [] from the workflow when no weights are
    // available, or a file path to resume training from. Staged as a fixed
    // filename so the script body does not depend on the source path.
    tuple val(meta), path(dataset), path(resume_weights, stageAs: 'provided_weights.pt')

    // Stage checkpoint from persistent location before task starts (for retries)
    beforeScript {
        """
        CKPT_SRC='${params.outdir}/checkpoints/${meta.id}/chkpoint.pt'
        if [ -f "\$CKPT_SRC" ]; then
            # A checkpoint half-written by a preempted/killed attempt is a truncated
            # zip with no End-of-Central-Directory record (PK\\x05\\x06). torch.load
            # then raises "PytorchStreamReader failed ... central directory" and
            # crashes on EVERY retry — an unrecoverable loop. Validate before staging
            # (coreutils only, no python: beforeScript runs outside the conda env);
            # quarantine a corrupt file so training restarts clean and afterScript can
            # replace it with a good checkpoint.
            if tail -c 128 "\$CKPT_SRC" | od -An -tx1 | tr -d ' \\n' | grep -q 504b0506; then
                cp "\$CKPT_SRC" prev_checkpoint.pt
                echo "Staged checkpoint from previous attempt"
            else
                echo "WARNING: persistent checkpoint \$CKPT_SRC is corrupt; quarantining and training from scratch"
                mv "\$CKPT_SRC" "\$CKPT_SRC.corrupt" || rm -f "\$CKPT_SRC"
            fi
        fi
        """
    }

    // Save checkpoint to persistent location after each attempt (runs even on failure).
    // Write to a temp file then atomically rename so an interrupted copy can never
    // leave a truncated checkpoint behind (which would poison every future retry).
    afterScript {
        """
        CKPT_DIR='${params.outdir}/checkpoints/${meta.id}'
        mkdir -p "\$CKPT_DIR"
        if [ -f model_output/chkpoint.pt ]; then
            cp model_output/chkpoint.pt "\$CKPT_DIR/.chkpoint.pt.tmp" && mv -f "\$CKPT_DIR/.chkpoint.pt.tmp" "\$CKPT_DIR/chkpoint.pt"
            echo "Saved checkpoint to persistent location"
        fi
        """
    }

    output:
    tuple val(meta), path("model_output"), emit: model
    tuple val(meta), path("model_output/chkpoint.pt"), emit: weights
    path "versions.yml"                  , emit: versions

    script:
    def args = task.ext.args ?: ''
    def attn_res = params.attn_resolutions.tokenize(',').collect { it.trim() as int }
    def chan_mult = params.channel_mult.tokenize(',').collect { it.trim() as int }
    def wandb_key = params.wandb_api_key ?: ''
    """
    mkdir -p model_output

    # Check for checkpoint staged by beforeScript from previous attempt.
    # Priority: prev_checkpoint.pt (retry) > provided_weights.pt (per-sample
    # or global --model_weights, staged via the resume_weights input).
    RESUME_FLAG=""
    if [ -f "prev_checkpoint.pt" ]; then
        RESUME_FLAG="--resume_from prev_checkpoint.pt"
        echo "Resuming from staged checkpoint (attempt ${task.attempt})"
    elif [ -f "provided_weights.pt" ]; then
        RESUME_FLAG="--resume_from provided_weights.pt"
        echo "Resuming from provided weights"
    fi

    # Generate config from params
    cat > train_config.yaml << 'EOFCONFIG'
experiment_name: ${params.experiment_name}
seed: ${params.seed}

data:
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
  pred_noise: ${params.pred_noise}

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
  noise_scale_factor: ${params.noise_scale_factor}

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

    # Persist a wandb run ID alongside the checkpoint so retries (which get
    # fresh work dirs) resume the same wandb run. The nonce is generated at
    # bash runtime, not Groovy render time, so it does not enter the task
    # hash — that keeps -resume working across Nextflow invocations.
    # A fresh Nextflow invocation that actually re-executes this task (cache
    # miss) will generate a new ID iff no prior run-id file exists; delete
    # the file under \$CKPT_DIR to force a brand-new wandb run.
    CKPT_DIR='${params.outdir}/checkpoints/${meta.id}'
    mkdir -p "\$CKPT_DIR"
    WANDB_ID_FILE="\$CKPT_DIR/wandb_run_id.txt"
    if [ -f "\$WANDB_ID_FILE" ]; then
        export WANDB_RUN_ID="\$(cat "\$WANDB_ID_FILE")"
    else
        export WANDB_RUN_ID="${params.experiment_name}_${meta.id}_\$(date +%s)_\$\$"
        echo "\$WANDB_RUN_ID" > "\$WANDB_ID_FILE"
    fi
    export WANDB_RESUME="allow"

    ebbf-train --config train_config.yaml --dataset ${dataset} --output_dir model_output \$RESUME_FLAG

    cat <<-END_VERSIONS > versions.yml
"${task.process}":
    ebbf: \$(python -c "import ebbf; print(ebbf.__version__)" 2>/dev/null || echo "1.0.0")
END_VERSIONS
    """
}
