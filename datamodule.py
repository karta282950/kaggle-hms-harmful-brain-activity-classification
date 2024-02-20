import numpy as np
import pandas as pd 
from glob import glob
from tqdm import tqdm
import time 
import os 
import math
from contextlib import contextmanager
import psutil
import sys
from sklearn.model_selection import GroupShuffleSplit 

from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl 
import shutil
from omegaconf import DictConfig
from pathlib import Path
import albumentations as A
import torch_audiomentations as tA

import hydra
from utils import quantize_data, mu_law_encoding, mu_law_expansion, butter_lowpass_filter

import warnings
warnings.filterwarnings('ignore')
###################
# Dataset 
###################

from utils import get_train_df, get_train_df_pop2

def get_all_spectrograms(cfg, READ_SPEC_FILES=False):
    paths_spectograms = glob(cfg.TRAIN_SPECTOGRAMS + "*.parquet")
    print(f'There are {len(paths_spectograms)} spectrogram parquets')

    if READ_SPEC_FILES:
        all_spectrograms = {}
        for file_path in tqdm(paths_spectograms):
            aux = pd.read_parquet(file_path)
            name = int(file_path.split("/")[-1].split('.')[0])
            all_spectrograms[name] = aux.iloc[:,1:].values
            del aux
        return all_spectrograms
    else:
        all_spectrograms = np.load(cfg.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
        return all_spectrograms

def get_all_egg(cfg, READ_EEG_SPEC_FILES=False, READ_EEG_RAW_FILES=False):
    paths_eegs = glob(cfg.TRAIN_EEGS + "*.npy")
    print(f'There are {len(paths_eegs)} EEG spectograms')
    if READ_EEG_SPEC_FILES:
        all_eegs = {}
        for file_path in tqdm(paths_eegs):
            eeg_id = file_path.split("/")[-1].split(".")[0]
            eeg_spectogram = np.load(file_path)
            all_eegs[eeg_id] = eeg_spectogram
        return all_eegs
    if not READ_EEG_RAW_FILES:
        all_eegs = np.load(cfg.PRE_LOADED_EEGS, allow_pickle=True).item()
    else:
        all_eegs = np.load(cfg.TRAIN_RAW_EEGS, allow_pickle=True).item()
    return all_eegs
    

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)

###################
# Dataset - 2D
###################
class CustomDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        cfg: DictConfig,
        specs: dict[int, np.ndarray],
        eeg_specs: dict[int, np.ndarray],
        #augment: bool, 
        mode: str = 'train',
        ):
        self.df = df
        self.cfg = cfg
        self.batch_size = self.cfg.BATCH_SIZE_TRAIN
        self.augment = self.cfg.AUGMENT
        self.mode = mode
        self.label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        self.spectograms = specs
        self.eeg_spectograms = eeg_specs
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X) 
        if self.mode=='train':
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(X, dtype=torch.float32)
    
    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df.iloc[index]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)
        
        for region in range(4):
            img = self.spectograms[row.spectrogram_id][r:r+300, region*100:(region+1)*100].T
            
            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectograms[row.eeg_id]
            X[:, :, 4:] = img
                
            if self.mode != 'test':
                y = row[self.label_cols].values.astype(np.float32)
            
        return X, y
    
    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.CoarseDropout(max_holes=4),
                #A.AddColoredNoise(p=0.15,mode="per_channel",p_mode="per_channel", max_snr_in_db = 15, sample_rate=200),

            ], p=0.5),
        ])

        return transforms(image=img)['image']

###################
# DataModule - 2D
###################
class SegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_df, self.label_cols = get_train_df(cfg)
        self.specs = get_all_spectrograms(cfg)
        self.eeg_specs = get_all_egg(cfg)
    
    def setup(self, stage: str) -> None:
        splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
        split = splitter.split(self.train_df, groups=self.train_df['patient_id'])
        train_inds, test_inds = next(split)

        self.train_ds = CustomDataset(self.train_df.iloc[train_inds], cfg=self.cfg, specs=self.specs, eeg_specs=self.eeg_specs)
        self.valid_ds = CustomDataset(self.train_df.iloc[test_inds], cfg=self.cfg, specs=self.specs, eeg_specs=self.eeg_specs) 
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.BATCH_SIZE_TRAIN,
            shuffle=True,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    
    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.cfg.BATCH_SIZE_VALID,
            shuffle=False,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
        )
        return valid_loader

###################
# Dataset - 1D
###################
def get_transforms(*, data):
    
    if data == 'train':
        return tA.Compose(
                transforms=[
                     # tA.ShuffleChannels(p=0.25,mode="per_channel",p_mode="per_channel",),
                     tA.AddColoredNoise(p=0.15,mode="per_channel",p_mode="per_channel", max_snr_in_db = 15, sample_rate=200),
                ])

    elif data == 'valid':
        return tA.Compose([
        ])


class CustomDataset1D(Dataset):
    def __init__(self, cfg, df, eegs=None, augmentations = None, test = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.eegs = eegs
        self.augmentations = augmentations
        self.test = test
        self.label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        #if row.eeg_id==568657:
        data = self.eegs[row.eeg_id]
        data = np.clip(data,-1024,1024)
        data = np.nan_to_num(data, nan=0) / 32.0
        
        data = butter_lowpass_filter(data)
        data = quantize_data(data,1)

        samples = torch.from_numpy(data).float()
        
        samples = self.augmentations(samples.unsqueeze(0))
        samples = samples.squeeze()

        samples = samples.permute(1,0)
        if not self.test:
            label = row[self.label_cols]
            label = torch.tensor(label).float()
            print(samples, label)
            return samples, label
        else:
            return samples
    
    def __transform(self):
        if self.augmentations:
            return tA.Compose(
                transforms=[
                    # tA.ShuffleChannels(p=0.25,mode="per_channel",p_mode="per_channel",),
                    tA.AddColoredNoise(p=0.15,mode="per_channel",p_mode="per_channel", max_snr_in_db = 15, sample_rate=200),
                ])
        return tA.Compose([])

class SegDataModule1D(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_df, self.label_cols = get_train_df(cfg)
        self.eegs = get_all_egg(cfg, READ_EEG_RAW_FILES=True)
    
    def setup(self, stage: str) -> None:
        splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
        split = splitter.split(self.train_df, groups=self.train_df['patient_id'])
        train_inds, test_inds = next(split)

        self.train_ds = CustomDataset1D(
            df=self.train_df.iloc[train_inds], cfg=self.cfg, augmentations = get_transforms(data='valid'), eegs=self.eegs)
        self.valid_ds = CustomDataset1D(
            df=self.train_df.iloc[test_inds], cfg=self.cfg, augmentations = get_transforms(data='valid'), eegs=self.eegs)
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.BATCH_SIZE_TRAIN,
            shuffle=False,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    
    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.cfg.BATCH_SIZE_VALID,
            shuffle=False,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
        )
        return valid_loader

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):    
    datamodule = SegDataModule1D(cfg)
    datamodule.setup(stage=None)
    for inputs, outputs in datamodule.train_dataloader():
        print(inputs.shape)
        print(outputs.shape)
        break
    return

if __name__ == '__main__':
    main()