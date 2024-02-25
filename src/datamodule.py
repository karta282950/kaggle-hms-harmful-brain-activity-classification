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
from src.dataset.seg import CustomDataset1D, CustomDataset2D
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


@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    return

if __name__ == '__main__':
    main()