# Weights & Biases Benchmark for Drought Prediction

This is a deep learning/computer vision project to predict drought severity from satellite imagery and expert-labeled ground-level photos.
It is instrumented with [Weights & Biases](https://www.wandb.com) to track and visualize model training and facilitate collaboration.

## How to participate

You can learn more and [join the benchmark here](https://app.wandb.ai/wandb/droughtwatch/benchmark).

## Dataset

The current dataset consists of 86,295 train and 10,787 test satellite images, 65x65 pixels each, in 10 spectrum bands. Human experts (pastoralists) have labeled these with the number of cows that the corresponding geographic location could support (0, 1, 2, or 3+ cows). The data is in TFRecords format and takes up ~4.3GB. Permissions are pending on some of the data, and we will update this section as more data becomes available. You can [learn more about the format here](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_RT).

## Usage

Please refer to the [benchmark instructions](https://app.wandb.ai/wandb/droughtwatch/benchmark) to get started.

To train the model after you've cloned this repo:

```sh
# Install requirements.
pip install -U -r requirements.txt

# Install tensorflow GPU support if needed
pip install tensorflow-gpu

# Link to W&B -- this will track your training and save your run results.
wandb init

# Download the train and test data (~4.3GB) (default location: ``data`` in the repo)
bash download_data.sh

# Train a baseline model in Keras. Run with -h to see command line options
# To quickly verify that the model is training, set epochs=1
python train.py --epochs=1
```

## Next Steps

Here are some ideas to try next:

* different network architectures, loss functions, optimizers, and other hyperparameter settings
* explore subsets of bands and architectures that account for differences in the bands
* data augmentation (rotate, flip) and narrowing the focus (center crop)
* comparison between formulating this task as a regression vs a classification (predicting a continuous value vs a discrete label for drought severity)
* explore correlations between the sparse expertly-labeled data (RGB ground-level photos) and the dense, easier-to-obtain data (lower resolution satellite imagery in 10 spectral bands)
* better account for the class imbalance (actual distribution of ground truth labels: { 0: 63927, 1: 16135, 2: 16939, 3: 9993 })
