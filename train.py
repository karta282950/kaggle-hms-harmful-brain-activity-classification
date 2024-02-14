import logging
from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
import hydra
import pandas as pd
import numpy as np 
#from model import CustomModel
#from sklearn.model_selection import KFold, GroupKFold
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from datamodule import SegDataModule
from model import CustomModel
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
    torch.backends.cudnn.deterministic = True#将cuda加速的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = True#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
    torch.manual_seed(seed)#pytorch的随机种子
    np.random.seed(seed)#numpy的随机种子
    random.seed(seed)#python内置的随机种子

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    seed_everything(cfg.SEED)
    # init lightning model
    datamodule = SegDataModule(cfg)
    datamodule.setup(stage=None)

    LOGGER.info("Set Up DataModule")
    model = CustomModel(cfg)
    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
        dirpath=cfg.OUTPUT_DIR,
    )
    lr_monitor = LearningRateMonitor("epoch")
    model_summary = RichModelSummary(max_depth=2)
    wandb.login(key=secret_value)
    wandb.init(name=cfg.EXP_NAME, project="Harmful Brain Activity Classification")
    pl_logger = WandbLogger(name=cfg.EXP_NAME, project="Harmful Brain Activity Classification")
    #progress_bar = RichProgressBar()
    #progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
  
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
        callbacks=[checkpoint_cb, lr_monitor, model_summary, early_stopping],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)
    #trainer.save_checkpoint(f'EffNet_v{VER}_f{i}.ckpt')
    # load best weights
    '''
    model = model.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )
    
    weights_path = str("model_weights.pth")
      # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)
    '''

    return


if __name__ == "__main__":
    main()