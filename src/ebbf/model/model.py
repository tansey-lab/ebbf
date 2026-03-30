import torch

from ebbf.config import EBBFConfig
from ebbf.model.ema import EMA
from ebbf.model.flownet import TxFlowUNet


def instantiate_model(config: EBBFConfig, device: torch.device | str = "cpu"):
    model = TxFlowUNet(config)

    if config.train.use_ema:
        model = EMA(model)

    if config.get("model_initialization"):
        state_dict = torch.load(config.model_initialization, map_location=device)
        model.load_state_dict(state_dict["model"])
        model.float()

    return model
