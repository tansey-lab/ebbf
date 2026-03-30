"""Structured configuration for EBB&Flow training and inference.

This module defines typed dataclasses that serve as a schema for the YAML config files.
Using OmegaConf's structured config feature provides:
- Runtime type validation
- IDE autocomplete and type hints
- Clear documentation of available config options
"""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    patch_size: int = 100
    """Size of patches to extract from the data."""

    patch_padding: int = 10
    """Padding around patches."""

    max_tx: int = 32768
    """Maximum number of transcripts per patch."""

    dataset_size: float = 1e5
    """Number of samples in the dataset (can be float for scientific notation)."""

    min_nucs: int = 5
    """Minimum number of nuclei required per patch."""

    min_tx_factor: float = 0.75
    """Minimum transcript factor for sampling."""

    max_tx_factor: float = 1.25
    """Maximum transcript factor for sampling."""

    oversample_tx_factor: float = 1.5
    """Oversampling factor for transcripts."""

    max_nuc_dropout_prob: float = 0.2
    """Maximum probability of nucleus dropout during augmentation."""

    triangle_dirichlet_scale: float = 25.0
    """Scale parameter for triangle Dirichlet distribution."""

    max_cell_expansion_factor: float = 1.5
    """Maximum factor for cell expansion augmentation."""

    max_noise_prob: float = 0.15
    """Maximum probability of adding noise."""

    max_noise_scale: float = 3.0
    """Maximum scale of noise to add."""


@dataclass
class ModelConfig:
    """Configuration for the neural network model architecture."""

    conditioned: str | None = "nuc"
    """Conditioning type: 'nuc' for nuclear conditioning, or None."""

    img_resolution: int = 128
    """Resolution of the image representation."""

    in_channels: int = 64
    """Number of input channels to the UNet."""

    model_channels: int = 128
    """Base number of channels in the model."""

    out_channels: int = 128
    """Number of output channels from the UNet."""

    num_blocks: int = 2
    """Number of residual blocks per resolution level."""

    attn_resolutions: list[int] = field(default_factory=lambda: [32, 16])
    """Resolutions at which to apply attention."""

    dropout: float = 0.1
    """Dropout probability."""

    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    """Channel multipliers at each resolution level."""

    channel_mult_emb: int = 4
    """Channel multiplier for embeddings."""

    channels_per_head: int = 64
    """Number of channels per attention head."""

    nuc_channel_frac: float = 0.05
    """Fraction of channels allocated to nuclear features."""

    splat: bool = True
    """Whether to use splatting for grid encoding."""


@dataclass
class OptimConfig:
    """Configuration for the optimizer."""

    optimizer: str = "AdamW"
    """Optimizer type (currently only AdamW is supported)."""

    lr: float = 1e-4
    """Learning rate."""

    beta1: float = 0.9
    """Adam beta1 parameter."""

    beta2: float = 0.999
    """Adam beta2 parameter."""

    clip_grad: float = 10.0
    """Gradient clipping threshold."""

    eps: float = 1e-8
    """Adam epsilon for numerical stability."""

    use_scaler: bool = False
    """Whether to use gradient scaler for mixed precision training."""


@dataclass
class TrainConfig:
    """Configuration for training."""

    use_ema: bool = True
    """Whether to use Exponential Moving Average of model weights."""

    ema_warmup_epochs: int = 100
    """Number of epochs before EMA starts tracking (overwrites during warmup)."""

    save_freq: int = 1
    """Save checkpoint every N epochs."""

    batch_size: int = 64
    """Training batch size."""

    n_epochs: int = 350
    """Total number of training epochs."""

    grad_accum_steps: int = 2
    """Number of gradient accumulation steps."""

    time_skew: float = 2.0
    """Time skew parameter for Beta distribution (time ~ Beta(time_skew, 1))."""

    wandb_enabled: bool = True
    """Whether to enable Weights & Biases logging."""

    num_workers: int = 4
    """Number of DataLoader workers."""


@dataclass
class InferConfig:
    """Configuration for inference."""

    tx_fname: str | None = None
    """Path to transcript file for inference."""

    batch_size: int = 16
    """Inference batch size."""

    model_checkpoint: str | None = None
    """Path to model checkpoint for inference."""

    atol: float = 1e-5
    """Absolute tolerance for ODE solver."""

    rtol: float = 1e-5
    """Relative tolerance for ODE solver."""

    device: str = "auto"
    """Device for inference: 'auto', 'cuda', 'mps', or 'cpu'."""


@dataclass
class EBBFConfig:
    """Root configuration for EBB&Flow training and inference.

    Example YAML config:
        experiment_name: my_experiment
        seed: 100
        output_folder: ./output

        data:
          patch_size: 100
          ...

        model:
          img_resolution: 128
          ...

        optim:
          lr: 1e-4
          ...

        train:
          n_epochs: 350
          ...

        infer:
          device: auto
          ...
    """

    experiment_name: str = MISSING
    """Name of the experiment (used for logging)."""

    seed: int = 100
    """Random seed for reproducibility."""

    output_folder: str = MISSING
    """Output directory for checkpoints and logs."""

    data: DataConfig = field(default_factory=DataConfig)
    """Data loading and preprocessing configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model architecture configuration."""

    optim: OptimConfig = field(default_factory=OptimConfig)
    """Optimizer configuration."""

    train: TrainConfig = field(default_factory=TrainConfig)
    """Training configuration."""

    infer: InferConfig = field(default_factory=InferConfig)
    """Inference configuration."""

    # Dynamic fields set at runtime (not from YAML)
    n_genes: int = 0
    """Number of genes (set dynamically from data)."""

    n_ctypes: int = 1
    """Number of cell types (set dynamically from data)."""

    # Optional fields for resuming/continuing training
    model_initialization: str | None = None
    """Path to checkpoint file to initialize model from."""

    prev_model_folder: str | None = None
    """Path to previous model folder for continuing training."""
