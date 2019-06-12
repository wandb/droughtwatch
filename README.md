# Weights & Biases Benchmark for Drought Prediction

This is a deep learning/computer vision project to predict drought severity from satellite imagery and expert-labeled ground-level photos.
It is instrumented with [Weights & Biases](https://www.wandb.com) to track and visualize model training and facilitate collaboration.

## How to participate

You can learn more and [join the benchmark here](https://app.wandb.ai/wandb/droughtwatch/benchmark).

## Dataset

The current dataset consists of 102,457 train and 5412 test satellite images, 65x65 pixels each, in 10 spectrum bands. Human experts (pastoralists) have labeled these with the number of cows that the corresponding geographic location could support (0, 1, 2, or 3+ cows). More data will become available and described here in the near future.

## Usage

Please refer to the [benchmark instructions](https://app.wandb.ai/wandb/droughtwatch/benchmark) to get started.

To train the model:

```sh
# Optional: create a virtualenv
pyenv virtualenv 3.6.4 droughtwatch-3.6
pyenv local droughtwatch-3.6
echo droughtwatch-3.6 > .python-version

# Install requirements.
pip install -U -r requirements.txt

# Install tensorflow GPU support if needed
pip install tensorflow-gpu

# Link to W&B -- this will track your training and save your run results.
wandb init

# Download the train and test data (~6.5GB) (default location: ``data`` in the repo)
bash download_data.sh

# Train your model in Tensorflow. Run with -h to see existing command line options
python train.py
```

## Next Steps

Here are some ideas to try next:
* Keras/PyTorch integration for faster experimentation
* different network architectures, loss functions, optimizers, and other hyperparameter settings
* reformulations of the problem as a classification instead of a regression
* explore correlations between the sparse expertly-labeled data (RGB ground-level photos) and the dense, easier-to-obtain data (lower resolution satellite imagery in 10 spectral bands)
* explore the data distribution and class balance: are different levels of drought intensity/forage quality evenly represented? what kind of new data would be most helpful to label?
