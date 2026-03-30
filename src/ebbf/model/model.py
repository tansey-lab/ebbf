import torch

from ebbf.config import EBBFConfig
from ebbf.model.ema import EMA
from ebbf.model.flownet import TxFlowUNet


def instantiate_model(
    config: EBBFConfig, device: torch.device | str = "cpu", checkpoint_path: str | None = None
):
    model = TxFlowUNet(config)

    if config.train.use_ema:
        model = EMA(model)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict["model"])
        model.float()

    return model
