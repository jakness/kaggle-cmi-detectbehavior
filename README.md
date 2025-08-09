# kaggle-cmi-detectbehavior
Kaggle competition: CMI - Detect Behavior with Sensor Data, 2025

## Setup the environment

Create a conda virtual environment and activate it, e.g. with [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main):
```bash
conda create -y -n kaggle-cmi python=3.12.11
conda activate kaggle-cmi
```

Install `kaggle-cmi`:
```bash
  pip install -e ".[dev]"
```

Install `PyTorch`:
```bash
  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

For development, install the pre-commit hooks:
```bash
   pre-commit install
```

## Download the data
Download the data and unzip the contents into `./data/raw` directory:
```bash
  kaggle competitions download -c cmi-detect-behavior-with-sensor-data --path data/raw
  unzip cmi-detect-behavior-with-sensor-data.zip
```

## Do basic data exploration for understanding the data
```bash
    python -m kaggle_cmi.explore.basic_eda
```

## Train a model

### Models available
- `se_gru`
  - Recommended for IMU (Inertial Measurement Unit) features only
- `conv1d`
  - Works for IMU features only or IMU, temperature and TOF (time of flight) features
- `two_branch`
  - IMU, temperature and TOF (time of flight) features required

### Training parameters

The following parameters can be used when training a model:

- `--model`: Name of the model to train
- `--use_only_imu`: Use only IMU features (accelerometer and rotation)
- `--use_only_valid`: Use only validation split (no test split)
- `--use_early_stopping`: Use early stopping during training
- `--n_epochs`: Number of training epochs (default: 300)
- `--sequence_length`: Length of an input sequence for the model (default: 80)
- `--saved_model_dir_path`: Directory path to save the trained model
- `--saved_model_name`: Name for the saved model

### Examples
#### Train a model that uses only IMU features with train/valid/test split and without early stopping

```bash
  SAVE_PATH="save_path"
  NAME="name"
    python -m kaggle_cmi.models.train \
      --model se_gru \
      --use_only_imu \
      --saved_model_dir_path $SAVE_PATH \
      --saved_model_name $NAME
```

### Train a model with all features (IMU + temperature + TOF) with only train/valid split and with early stopping

```bash
  SAVE_PATH="save_path"
  NAME="name"
    python -m kaggle_cmi.models.train \
      --model two_branch \
      --use_only_valid \
      --use_early_stopping \
      --saved_model_dir_path $SAVE_PATH \
      --saved_model_name $NAME
```
