import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.distributions.beta import Beta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ebbf.data.data import instantiate_train_dataloader, instantiate_transcript_sampler
from ebbf.data.patch import NuclearBatch
from ebbf.model.model import instantiate_model

logger = logging.getLogger(__name__)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    torch.cuda.set_device(rank)
    # Use file-based init for SLURM, or env-based for local
    if "SLURM_JOB_ID" in os.environ:
        file_path = f"/scratch/tanseyw/{os.environ['SLURM_JOB_ID']}_rdzv"
        init_method = f"file://{file_path}"
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        init_method = "env://"

    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        is_distributed: bool = False,
        is_main_process: bool = True,
    ) -> None:
        self.device = device
        self.config = config
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process

        model = model.to(device)
        if is_distributed:
            self.model = DDP(
                model, device_ids=[device.index], find_unused_parameters=True
            )
        else:
            self.model = model

        self.tx_sampler = instantiate_transcript_sampler(config, device=device)
        self.time_distribution = Beta(
            torch.tensor(1.0, device=device).float(),
            torch.tensor(config.train.time_skew, device=device).float(),
        )
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.patch_size: float = config.data.patch_size
        self.grad_accum_steps: int = config.train.grad_accum_steps
        self.n_epochs: int = config.train.n_epochs
        self.clip_grad: float = config.optim.clip_grad
        self.use_ema: bool = config.train.use_ema
        self.ema_warmup_epochs: int = config.train.ema_warmup_epochs
        self.save_freq: int = config.train.save_freq
        self.wandb_enabled: bool = config.train.get("wandb_enabled", True)

        if is_main_process:
            if self.wandb_enabled:
                wandb.login()
                self.run = wandb.init(
                    project="Transcript diffusion",
                    config=OmegaConf.to_container(config, resolve=True),
                    mode="offline" if not torch.cuda.is_available() else "online",
                )

            self.output_dir = config.output_folder
            os.makedirs(self.output_dir, exist_ok=True)

            ## Save out the configurations
            OmegaConf.save(config, os.path.join(self.output_dir, "config.yaml"))

    def _get_model_module(self):
        """Get the underlying model (unwrap DDP if needed)."""
        if self.is_distributed:
            return self.model.module
        return self.model

    def _generate_path(self, batch: NuclearBatch):
        batch.to(self.device)
        tx_batch = self.tx_sampler.sample_transcripts(batch)
        tx_pos_0 = tx_batch["tx_xy"]
        tx_pos_1 = tx_batch["tx_nuc_xy"]
        tx_seqlens = tx_batch["tx_seqlens"]

        batch_size = tx_seqlens.shape[0]

        ## sample time
        t = self.time_distribution.sample((batch_size,))
        t_rep = torch.repeat_interleave(t, repeats=tx_seqlens, dim=0).unsqueeze(-1)

        ## interpolate
        tx_pos_t = t_rep * tx_pos_1 + (1 - t_rep) * tx_pos_0

        ## target
        target = tx_pos_1 - tx_pos_0

        tx_batch["tx_pos_t"] = tx_pos_t
        tx_batch["t"] = t.squeeze()
        tx_batch["target"] = target

        return tx_batch

    def _calculate_loss(self, batch):
        tx_batch = self._generate_path(batch)
        pred = self.model(
            tx_pos=tx_batch["tx_pos_t"],
            gene_idx=tx_batch["tx_gene_idx"],
            t=tx_batch["t"],
            tx_seqlens=tx_batch["tx_seqlens"],
            nuc_pos=(tx_batch["nuc2xy"] if "nuc2xy" in tx_batch else None),
            nuc_seqlens=(
                tx_batch["nuc_seqlens"] if "nuc_seqlens" in tx_batch else None
            ),
            nuc_ellipse_pos=(
                tx_batch["nuc2ellipse_coordinates"]
                if "nuc2ellipse_coordinates" in tx_batch
                else None
            ),
            nuc_ellipse_seqlens=(
                tx_batch["nuc2ellipse_seqlens"]
                if "nuc2ellipse_seqlens" in tx_batch
                else None
            ),
            nuc_ctypes=(tx_batch["nuc2ctype"] if "nuc2ctype" in tx_batch else None),
        )
        loss = torch.mean(torch.square(pred - tx_batch["target"].squeeze()))

        ## scale loss to be on the absolute scale
        loss = (0.25 * self.patch_size**2) * loss

        ## gradient accumulation
        loss = loss / self.grad_accum_steps
        return loss

    def _run_batch(self, batch, step: int, epoch: int):
        ## Accumulate gradients
        if (step + 1) % self.grad_accum_steps == 0:
            loss = self._calculate_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.use_ema:
                self._get_model_module().update_ema(
                    overwrite=(epoch < self.ema_warmup_epochs)
                )
        else:
            if self.is_distributed:
                with self.model.no_sync():  ## No need to sync when we don't take a step
                    loss = self._calculate_loss(batch)
                    loss.backward()
            else:
                loss = self._calculate_loss(batch)
                loss.backward()
        return loss.item() * self.grad_accum_steps

    def _run_epoch(self, epoch):
        self.model.train(True)
        if self.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = len(self.train_dataloader)
        for step, batch in enumerate(self.train_dataloader):
            loss = self._run_batch(batch, step, epoch)
            if self.is_main_process:
                if self.wandb_enabled:
                    self.run.log({"Train loss (step)": loss})
                epoch_loss += loss / n_batches
        if self.is_main_process:
            if self.wandb_enabled:
                self.run.log({"Train loss (epoch)": epoch_loss, "Epoch": epoch})

    def _save_checkpoint(self, epoch):
        to_save = {
            "model": self._get_model_module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(to_save, os.path.join(self.output_dir, "chkpoint.pt"))

    def train(self):
        for epoch in range(self.n_epochs):
            self._run_epoch(epoch)
            if self.is_main_process and (epoch % self.save_freq == 0):
                self._save_checkpoint(epoch)
        # Final save
        if self.is_main_process:
            self._save_checkpoint(self.n_epochs - 1)


def main_distributed(rank: int, world_size: int, config: OmegaConf):
    """Distributed training entry point (multi-GPU)."""
    ddp_setup(rank, world_size)
    dataloader = instantiate_train_dataloader(
        config, distributed=True, num_workers=4, world_size=world_size, rank=rank
    )

    model = instantiate_model(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
    )
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device(f"cuda:{rank}"),
        is_distributed=True,
        is_main_process=(rank == 0),
    )
    trainer.train()
    destroy_process_group()


def main_single(config: OmegaConf, device: torch.device):
    """Single-process training entry point (CPU or single GPU)."""
    from ebbf.compat import get_num_cpus

    num_workers = min(4, get_num_cpus())

    dataloader = instantiate_train_dataloader(
        config, distributed=False, num_workers=num_workers
    )

    model = instantiate_model(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
    )
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        is_distributed=False,
        is_main_process=True,
    )
    trainer.train()


# Keep old name for backwards compatibility
main = main_distributed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config filename.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if "prev_model_folder" in config:
        prev_model_folder = config.prev_model_folder
        uconfig = OmegaConf.load(os.path.join(prev_model_folder, "config.yaml"))
        uconfig.model_initialization = os.path.join(prev_model_folder, "chkpoint.pt")
        uconfig.data.tma_fname = os.path.join(prev_model_folder, "tma.npz")
        config = OmegaConf.merge(uconfig, config)

    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(main_distributed, args=(world_size, config), nprocs=world_size)
    else:
        logger.info("No CUDA devices found, running on CPU...")
        main_single(config, torch.device("cpu"))
