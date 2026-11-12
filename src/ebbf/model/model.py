import torch

from ebbf.config import EBBFConfig
from ebbf.model.ema import EMA
from ebbf.model.flownet import TxFlowUNet

NUC_EMBEDDING_KEY = "nuc_ellipse_encoder.embed_table.weight"


def has_legacy_nuc_embedding(state_dict: dict) -> bool:
    """True if the checkpoint has one nucleus embedding per cell type (pre single-embedding)."""
    return any(k.endswith(NUC_EMBEDDING_KEY) and v.shape[0] > 1 for k, v in state_dict.items())


def load_model_state(model: torch.nn.Module, state_dict: dict, nuc_embedding_index: int = 1):
    """Load a checkpoint, mapping legacy per-cell-type nucleus embeddings to a single row.

    Legacy checkpoints store ``n_ctypes + 1`` nucleus embeddings; with cell types off, every
    nucleus used the last row (index 1 when n_ctypes = 1). Any tensor whose checkpoint shape is
    (K, C) with K > 1 where the model expects (1, C) -- the embedding itself and its EMA shadow
    copy -- is sliced to row ``nuc_embedding_index``.
    """
    model_state = model.state_dict()
    state_dict = dict(state_dict)
    for k, v in state_dict.items():
        target = model_state.get(k)
        if (
            target is not None
            and target.dim() == 2
            and target.shape[0] == 1
            and v.dim() == 2
            and v.shape[0] > 1
            and v.shape[1] == target.shape[1]
        ):
            state_dict[k] = v[nuc_embedding_index : nuc_embedding_index + 1]
    model.load_state_dict(state_dict)


def instantiate_model(
    config: EBBFConfig,
    device: torch.device | str = "cpu",
    checkpoint_path: str | None = None,
    nuc_embedding_index: int = 1,
):
    model = TxFlowUNet(config)

    if config.train.use_ema:
        model = EMA(model)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        load_model_state(model, state_dict["model"], nuc_embedding_index=nuc_embedding_index)
        model.float()

    return model
