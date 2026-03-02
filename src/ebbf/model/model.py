import torch
from omegaconf import OmegaConf

from ebbf.model.ema import EMA
from ebbf.model.flownet import TxFlowUNet


def instantiate_model(config: OmegaConf):
    model = TxFlowUNet(config)

    if config.train.use_ema:
        model = EMA(model)

    if "model_initialization" in config:
        state_dict = torch.load(config.model_initialization, map_location="cpu")
        model.load_state_dict(state_dict["model"])

    return model
