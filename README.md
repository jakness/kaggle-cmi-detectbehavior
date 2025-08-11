# kaggle-cmi-detectbehavior
This is a repository for training models for the
[`Kaggle competition: CMI - Detect Behavior with Sensor Data, 2025`](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).
The goal of this competition is to develop a predictive model that distinguishes BFRB-like (body-focused repetitive behaviors,
like hair pulling) and non-BFRB-like (everyday gestures, like adjusting glasses) activity using data from a variety of sensors collected via a wrist-worn device.

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

#### Train a model with all features (IMU + temperature + TOF) with only train/valid split and with early stopping

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

## Notes
The competition had an interesting problem setting with real world benefits, which is why I wanted to test how good
the data is for learning a model that could detect the right behaviors. I had a limited time window (a couple of days)
to work on this, so I mostly focused on training a single model that could possibly be used in production. In production, we most
likely want to have real-time detection of the behaviors so we need to be mindful of the model size and inference time.
For example, ensembling multiple models might be too resource intensive for real-time inference but would likely improve
the competition score.

### Insights from the data exploration
- All the features are numerical
- No duplicates
- Distribution of gestures is pretty balanced, but there are some gestures with a bit fewer samples but nothing too extreme
- There are missing values that should be handled
- There are -1 values in TOF data which should be treated as missing values since -1 doesn't make sense for these features
- The scale of the features is pretty similar, so scaling might not be necessary, but it wouldn't hurt either
- Multiple different sequences
- Every sequence contains a transition phase and a gesture phase
  - The gesture phase is always at the end of the sequence
- The sequences and the phases in the sequences are of different lengths
- The longest gesture phase in a sequence is around 70 samples
  - `--sequence_length` parameter in training is by default set to 80 because of this. So at training we use
  the last 80 observations from a sequence to make a prediction. If the length of the original sequence is smaller
  than 80 then the sequence is padded with zeros so that the length would be 80
- Sampling rate is not known, but it might be around 20Hz
- A couple of pretty big correlations between the features, but nothing too extreme
- There is also demographic data available, but I didn't have time to explore and use it

### Models
- `conv1d`
  - Basic 1D convolutional neural network for a benchmark result.
  1D convolutions are a good starting point for modeling since we are working with time series data.
  1D convolution slides through the sequences trying to capture relations between current and past measurements.
  Additional improvements in the training process should get a better score.
- `se_gru`
  - Combines several different layers to capture the relationships in the data:
    - 1D convolution for local patterns from the original sequence
    - Squeeze-and-excitation block to capture interdependencies between channels
    - Residual shortcut connection allows information from earlier layers to be directly fed into later layers
    - Gated recurrent unit to capture dependencies in layer sequences
    - Linear layers to refine the previous layers' information into the final gesture predictions
- `two_branch`
  - Processes IMU and temperature/TOF features in two separate branches
    - IMU branch contains 1D convolutions and squeeze-and-excitation blocks
    - Temperature/TOF contains 1D convolutions
    - Merged IMU and temperature/TOF branches are fed into LSTM layer to capture dependencies in the sequences
    - Linear layers for final gesture predictions

### Feature engineering
We create features from IMU and TOF sensors.
- Features derived from IMU
  - Acceleration magnitude and jerk
  - Rotation angle and angle velocity
  - Linear acceleration, linear acceleration magnitude and jerk
  - Angular velocity for x, y and z axis; angular velocity jerk and snap for x, y and z axis; angular velocity magnitude, jerk and snap
  - Angular distance
- Features derived from TOF
  - Mean, min and max


### Results
The submission for the competition requires at least a model that uses only IMU features because the test data has sequences
where the temperature and TOF measurements are missing
- `conv1d` benchmark model with only IMU features achieved a competition score of 0.708
- `conv1d` model with only IMU features and `conv1d` model with IMU, temperature and TOF features achieved a competition score of 0.712
  - So adding more features for this basic architecture didn't relly make a difference
- `conv1d` model with only IMU features and `two_branch` model with IMU, temperature and TOF features achieved a competition score of 0.755
  - Better model architecture makes a difference
- `se_gru` benchmark model with only IMU features achieved a competition score of 0.736
  - For models with only IMU features the score went from 0.708 to 0.736. Again, better model architecture makes a difference
- `se_gru` model with only IMU features and `two_branch` model with IMU, temperature and TOF features achieved a competition score of 0.766
- `se_gru` model with only IMU features + engineered IMU features and `two_branch` model with IMU, temperature, TOF features +
engineered IMU and TOF features achieved a competition score of 0.789
  - New features are helpful
  - With standard scaler achieved a competition score of 0.79
    - Scaling the data didn't really make a difference
- At the time of latest submission the best competition score was 0.87

### Next steps
- Create more features. For example, frequency domain features might work pretty well since we are trying to classify repetitive behaviours
- Ensembling for better competition score. I would estimate the improvement to be around 2% with `se_gru` and `two_branch`
models trained with different folds.
- Tuning all the parameters of the models. There are a lot of parameters to tune with neural networks so there is
potential for good improvements
- Model that separates IMU, temperature and TOF features into their own branch
- Different models, for example Transformers or LightGBM
  - LightGBM isn't natively built for time series data, but you can make it work for time series data by creating
  features from the previous measurements. For example, shifting and differencing
- Seed everything for reproducibility
- Confusion matrix to check how well the gestures are predicted
- Interpret the importance of the features to gain insight what features work
  - Great resource: https://christophm.github.io/interpretable-ml-book/overview.html
