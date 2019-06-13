# Weights & Biases Benchmark for Drought Prediction

This project leverages deep learning and computer vision for drought
resilience, using satellite images and human expert labels to
detect drought conditions in Northern Kenya.

It is instrumented with [Weights & Biases](https://www.wandb.com) to track and visualize model training and facilitate collaboration.
The [W&B Benchmark](https://app.wandb.ai/wandb/droughtwatch/benchmark) is a public home for developing deep learning
models for drought prediction. The current challenge is to learn from ~100K expert labels of forage quality (concretely, how many cows from
0 to 3+ can the given geolocation support?) to make more accurate predictions from unlabeled satellite images. With better models,
index insurance companies can monitor drought conditions&mdash;and send resources to families in the area&mdash;more effectively.

## How to participate

You can learn more and [join the benchmark here](https://app.wandb.ai/wandb/droughtwatch/benchmark).

## Dataset

The current dataset consists of 97,082 train and 5,393 test satellite images, 65x65 pixels each, in 10 spectrum bands. Human experts (pastoralists) have labeled these with the number of cows that the corresponding geographic location could support (0, 1, 2, or 3+ cows). The data is in TFRecords format and takes up ~4.5GB. We are actively iterating on the dataset and will update this section as newer versions become available. You can [learn more about the format here](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_RT).

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
* explore subsets of spectral bands and architectures that account for differences in the spectral bands
* data augmentation (rotate, flip) and narrowing the focus (center crop)
* comparison between formulating this task as a regression (predicting a continuous value for drought severity or forage quality)
vs a classification (predicting a discrete label))
* explore correlations between the sparse expertly-labeled data (RGB ground-level photos) and the dense, easier-to-obtain data (lower resolution satellite imagery in 10 spectral bands)
* explore strategies to account for the class imbalance (roughly ~60% of the full data gathered is of class 0, classes 1 and 2 have ~15% each, and the remaining ~10% is class 3)
