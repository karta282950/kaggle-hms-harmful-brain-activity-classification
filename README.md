# Harmful-Brain-Activity-Classification
## 模型筆記

### 1. 現在2D的方法用renet34d跟effficient_b0-effficient_b2做

### 2. 1D方式用cnn1d

### 3. 標籤用投票數10當分界，分別訓練兩個模型


## Build Environment
### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

## Prepare Data

### 1. Download data

```bash
cd data
kaggle competitions download -c hms-harmful-brain-activity-classification
unzip hms-harmful-brain-activity-classification.zip
```

### 2. Preprocess data

```bash
rye run python -m prepare_data.py phase=train/test
```

## Train Model
The following commands are for training the model of LB0.4
```bash
rye run python train.py
```

## Upload Model
```bash
rye run python tools.py upload_dataset.py
```

## Inference
The following commands are for inference of LB0.4
```bash
rye run python inference.py dir=kaggle exp_name=exp001 weight.run_name=single
```