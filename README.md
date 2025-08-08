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
