import logging
from pathlib import Path
import torch
from torch import nn
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
from datamodule import SegDataModule, SegDataModule1D
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

#測試val進度條
class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar

from pytorch_lightning.callbacks import Callback
class FlexibleTqdm(Callback):
    def __init__(self, steps, column_width=10):
        super(FlexibleTqdm, self).__init__()
        self.steps = steps
        self.column_width = column_width
        self.info = "\rEpoch_%d %s%% [%s]"

    def on_train_start(self, trainer, module):
        history = module.history
        self.row = "-" * (self.column_width + 1) * (len(history) + 2) + "-"
        title = "|"
        title += "epoch".center(self.column_width) + "|"
        title += "time".center(self.column_width) + "|"
        for i in history.keys():
            title += i.center(self.column_width) + "|"
        print(self.row)
        print(title)
        print(self.row)
    
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        current_index = int((batch_idx + 1) * 100 / self.steps)
        tqdm = ["."] * 100
        for i in range(current_index - 1):
            tqdm[i] = "="
        if current_index:
            tqdm[current_index - 1] = ">"
        print(self.info % (module.current_epoch, str(current_index).rjust(3), "".join(tqdm)), end="")
    '''
    def on_train_epoch_start(self, trainer, module):
        print(self.info % (module.current_epoch, "  0", "." * 100), end="")
        self.begin = time.perf_counter()
    
    def on_train_epoch_end(self, trainer, module):
        self.end = time.perf_counter()
        history = module.history
        print(history)
        detail = "\r|"
        detail += str(module.current_epoch).center(self.column_width) + "|"
        detail += ("%d" % (self.end - self.begin)).center(self.column_width) + "|"
        print(history)
        for j in history.values():
            #value = j[-1] if j else 0
            detail += ("%.06f" % j[-1]).center(self.column_width) + "|"
        print("\r" + " " * 120, end="")
        print(detail)
        print(self.row)
    '''
class LitProgressBar(ProgressBar):

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,  # This two lines are only for pycharm
            ncols=100,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar

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
    #progress_bar = MyProgressBar() val會print lines
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
        callbacks=[checkpoint_cb, lr_monitor, model_summary, early_stopping, FlexibleTqdm(len(datamodule.train_ds) // 32, column_width=12)],
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