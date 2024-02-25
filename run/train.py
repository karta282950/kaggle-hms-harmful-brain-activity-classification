import logging
from pathlib import Path
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hydra
import pandas as pd
import numpy as np 
import sys
#from model import CustomModel
#from sklearn.model_selection import KFold, GroupKFold
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar,
    ProgressBar
)
import sys, os
sys.path.append('/kaggle/working/kaggle-hms-harmful-brain-activity-classification')

from src.datamodule import get_all_egg, get_all_spectrograms
from src.dataset.common import get_train_df
from src.dataset.seg import CustomDataset2D
from model import CustomModel
import random
from kaggle_secrets import UserSecretsClient
import wandb
from tqdm.auto import tqdm

secret_label = "WANDB"
secret_value = UserSecretsClient().get_secret(secret_label)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    seed_everything(cfg.SEED)
    LOGGER.info("Set Up DataModule")
    wandb.login(key=secret_value)
    wandb.init(name=cfg.EXP_NAME, project="Harmful Brain Activity Classification")

    specs = get_all_spectrograms(cfg)
    eeg = get_all_egg(cfg)
    train = get_train_df(cfg)
    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
        print('#'*25)
        print(f'### Fold {i}')
        #train_copy.loc[train_index, "fold"] = i
        ds_train = CustomDataset2D(train.iloc[train_index], cfg, specs=specs, eeg_specs=eeg)
        dl_train = DataLoader(ds_train, shuffle=True, batch_size=32, num_workers=3)
        ds_valid = CustomDataset2D(train.iloc[valid_index], cfg, specs=specs, eeg_specs=eeg)
        dl_valid = DataLoader(ds_valid, shuffle=False, batch_size=64, num_workers=3)
        print(f'### Train size: {len(train_index)}, Valid size: {len(valid_index)}')
        print('#'*25)
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        ckpt_callback = pl.callbacks.ModelCheckpoint(verbose=True, monitor='val_loss', save_top_k=1, mode='min', dirpath='/kaggle/working',filename=f'efficientnet_b0_fold_{i}')
        pl_logger = WandbLogger(name=cfg.EXP_NAME, project="Harmful Brain Activity Classification")
        lr_monitor = LearningRateMonitor("epoch")
        model_summary = RichModelSummary(max_depth=2)

        trainer = pl.Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.EPOCHS,
        max_steps=cfg.EPOCHS * len(dl_train),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[ckpt_callback, lr_monitor, model_summary, early_stopping],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(dl_train) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        )
        model = CustomModel(cfg)
        trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_valid)
        del trainer, model

    return


if __name__ == "__main__":
    main()