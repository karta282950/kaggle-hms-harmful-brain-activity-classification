import pandas as pd 

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