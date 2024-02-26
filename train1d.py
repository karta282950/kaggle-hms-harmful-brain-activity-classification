import logging
from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
import hydra
import pandas as pd
import numpy as np 
import gc
import sys
#from model import CustomModel
#from sklearn.model_selection import KFold, GroupKFold
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from datamodule import SegDataModule1D
from resnet1d import EEGModel
import random
from kaggle_secrets import UserSecretsClient
import wandb

secret_label = "WANDB"
secret_value = UserSecretsClient().get_secret(secret_label)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

#import wandb
#wandb.login()
#wandb.init(settings=wandb.Settings(start_method="thread"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=1e-2)
    interval = "epoch"
    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer, T_0=20, T_mult=1, eta_min=1e-6, last_epoch=-1)

    return {
        "optimizer": model_optimizer, 
        "lr_scheduler": {
            "scheduler": lr_scheduler, "interval": interval, "monitor": "val_loss", "frequency": 1}}

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    seed_everything(cfg.SEED)
    # init lightning model
    datamodule = SegDataModule1D(cfg)
    datamodule.setup(stage=None)
    gc.collect()
    LOGGER.info("Set Up DataModule")
    model = EEGModel(cfg)
    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
        dirpath=cfg.OUTPUT_DIR,
        filename= f'eegnet_best_loss',
    )
    lr_monitor = LearningRateMonitor("epoch")
    model_summary = RichModelSummary(max_depth=2)
    wandb.login(key=secret_value)
    pl_logger = WandbLogger(name=cfg.EXP_NAME, project="Harmful Brain Activity Classification")
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    #early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
  
    trainer = pl.Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.EPOCHS,
        max_steps=cfg.EPOCHS * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, model_summary, progress_bar],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )
    trainer.fit(model, datamodule=datamodule)
    return

if __name__ == "__main__":
    main()