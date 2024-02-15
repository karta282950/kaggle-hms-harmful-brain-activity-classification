import torch
from torch.utils.data import DataLoader
from model import CustomModel
import hydra
import pandas as pd
import pywt
import librosa
import numpy as np 
import matplotlib.pyplot as plt 
import os
from datamodule import CustomDataset
VER = 5
i = 0
LOAD_MODELS_FROM = None

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

def get_test_df(cfg):
    test = pd.read_csv(cfg.TEST_CSV)
    #test = test.rename({'spectrogram_id': 'spec_id'}, axis=1)
    print('Test shape',test.shape)
    return test

def get_all_spectrograms(cfg, READ_SPEC_FILES=True):
    files = os.listdir(cfg.TEST_SPECTOGRAMS)
    print(f'There are {len(files)} test spectrogram parquets'); print()
    spectrograms2 = {}
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(i, ', ',end='')
        tmp = pd.read_parquet(f'{cfg.TEST_SPECTOGRAMS}{f}')
        name = int(f.split('.')[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    
    return spectrograms2

def get_all_egg(cfg, test, READ_EEG_SPEC_FILES=False, DISPLAY = 1):
    EEG_IDS = test.eeg_id.unique()
    print('Converting Test EEG to Spectrograms...'); print()
    all_eegs={}
    for i, eeg_id in enumerate(EEG_IDS):
        # CREATE SPECTROGRAM FROM EEG PARQUET
        img = spectrogram_from_eeg(f'{cfg.TEST_EEGS}{eeg_id}.parquet', i < DISPLAY)
        all_eegs[eeg_id] = img
    
    return all_eegs

def spectrogram_from_eeg(parquet_path, display=False, USE_WAVELET=None, eeg_id=568657):
    NAMES = ['LL','LP','RP','RR']

    FEATS = [['Fp1','F7','T3','T5','O1'],
            ['Fp1','F3','C3','P3','O1'],
            ['Fp2','F8','T4','T6','O2'],
            ['Fp2','F4','C4','P4','O2']]
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')
    
    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')
            
    if display: 
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()
        
    return img

from dataclasses import dataclass
@dataclass
class InferenceConfig:
    TEST_CSV: str
    TEST_SPECTOGRAMS: str
    TEST_EEGS: str
    BATCH_SIZE_TRAIN: int
    AUGMENT: bool
    LOAD_MODELS_FROM: str
    FREEZE: bool
    model: str
    NUM_FROZEN_LAYERS: int
    EPOCHS: int

def load_model(cfg):
    model = CustomModel(cfg)
    model.load_from_checkpoint(torch.load(cfg.LOAD_MODELS_FROM))
    print('load weight from "{}"'.format(cfg.LOAD_MODELS_FROM))
    return model

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    test = get_test_df(cfg)
    spectrograms = get_all_spectrograms(cfg)
    all_eegs = get_all_egg(cfg, test)
    preds = []
    test_ds = CustomDataset(test, cfg=cfg, mode='test', specs=spectrograms, eeg_specs=all_eegs)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=3)
    ckpt_file = cfg.LOAD_MODELS_FROM
    model = CustomModel(cfg) 
    model.load_state_dict(torch.load(ckpt_file),False)
    model = model.to(device).eval()
    preds = []
    with torch.inference_mode():
        for test_batch in test_loader:
            print(test_batch)
            test_batch = test_batch.to(device)
            pred = torch.softmax(model(test_batch), dim=1).cpu().numpy()
            preds.append(pred)
    print()
    print('Test preds shape',preds.shape)
    sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
    sub[TARGETS] = preds
    sub.to_csv('submission.csv',index=False)
    print('Submissionn shape',sub.shape)
    sub.head()
    

if __name__ == '__main__':
    main()