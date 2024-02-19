import pandas as pd 
import numpy as np 
from scipy.signal import butter, lfilter

def get_train_df(cfg):
    df = pd.read_csv(cfg.TRAIN_CSV)
    label_cols = df.columns[-6:]
    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_id':'first',
        'spectrogram_label_offset_seconds':'min'
        })
    train_df.columns = ['spectrogram_id', 'min']
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

def get_train_df_pop2(cfg, pop=False):
    df = pd.read_csv(cfg.TRAIN_CSV)
    label_cols = df.columns[-6:]
    df['total_evaluators'] = df[label_cols].sum(axis=1)

    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_id':'first',
        'spectrogram_label_offset_seconds':'min'
        })
    train_df.columns = ['spectrogram_id', 'min']
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
    train_df = train_df.merge(\
        df[['eeg_id', 'spectrogram_id', 'total_evaluators']].drop_duplicates().rename(columns={'spectrogram_id':'spec_id'}),\
            on=['eeg_id', 'spec_id'], how='left')
    train_df = train_df.reset_index()
    if not pop:
        return train_df.drop(['total_evaluators'], axis=1)
    else:
        train_df_less10 = train_df.query('total_evaluators<10')
        train_df_large10 = train_df.query('total_evaluators>=10')
        return train_df_less10, train_df_large10, label_cols

###################
# 1D-Denoise 
###################
def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    # bins = np.linspace(-1, 1, classes)
    # quantized = np.digitize(mu_x, bins) - 1
    return mu_x#quantized

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data