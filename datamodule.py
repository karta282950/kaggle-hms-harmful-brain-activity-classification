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
import hydra

import warnings
warnings.filterwarnings('ignore')
###################
# Dataset 
###################

def strong_aug(p=1):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=.1),
            A.Blur(blur_limit=3, p=.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomContrast(),
            A.RandomBrightness(),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)

def get_train_df(cfg):
    df = pd.read_csv(cfg.TRAIN_CSV)
    label_cols = df.columns[-6:]
    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_id':'first',
        'spectrogram_label_offset_seconds':'min'
        })
    train_df.columns = ['spectogram_id', 'min']
    aux = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_label_offset_seconds':'max'
        })
    train_df['max'] = aux
    aux = df.groupby('eeg_id')[['patient_id']].agg('first')
    train_df['patient_id'] = aux

    aux = df.groupby('eeg_id')[label_cols].agg('sum')
    for label in label_cols:
        train_df[label] = aux[label].values
        
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[label_cols] = y_data

    aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train_df['target'] = aux

    train_df = train_df.reset_index()
    return train_df, label_cols

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

def get_all_egg(cfg, READ_EEG_SPEC_FILES=False):
    paths_eegs = glob(cfg.TRAIN_EEGS + "*.npy")
    print(f'There are {len(paths_eegs)} EEG spectograms')
    if READ_EEG_SPEC_FILES:
        all_eegs = {}
        for file_path in tqdm(paths_eegs):
            eeg_id = file_path.split("/")[-1].split(".")[0]
            eeg_spectogram = np.load(file_path)
            all_eegs[eeg_id] = eeg_spectogram
        return all_eegs
    else:
        all_eegs = np.load(cfg.PRE_LOADED_EEGS, allow_pickle=True).item()
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
# Dataset
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
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                        
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
            img = self.spectograms[row.spectogram_id][r:r+300, region*100:(region+1)*100].T
            
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
        transforms = strong_aug()
        
        #    A.Compose([
        #    A.HorizontalFlip(p=0.5),
            
        #])
        return transforms(image=img)['image']

###################
# DataModule
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
    
@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):    
    datamodule = SegDataModule(cfg)
    datamodule.setup(stage=None)
    for inputs, outputs in datamodule.train_dataloader():
        print(inputs.shape)
        print(outputs.shape)
        break
    return 

if __name__ == '__main__':
    main()