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

The current dataset consists of 86,317 train and 10,778 validation satellite images, 65x65 pixels each, in 10 spectrum bands, with 10,774 images withheld to test long-term generalization (107,869 total). Human experts (pastoralists) have labeled these with the number of cows that the geographic location at the **center** of the image could support (0, 1, 2, or 3+ cows). Each pixel represents a 30 meter square, so the images at full size are 1.95 kilometers across. Pastoralists are asked to rate the quality of the area within 20 meters of where they are standing, which corresponds to an area slightly larger a single pixel. Since forage quality is correlated across space, the larger image may be useful for prediction. 

The data is in TFRecords format, split into ``train`` and ``val``, and takes up ~4.3GB (2.15GB zipped). 
You can [learn more about the format of the satellite images here](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_RT).

## Setup instructions

```sh
# Clone this repository
git clone https://github.com/wandb/droughtwatch.git

# Install requirements
cd droughtwatch
pip install -U -r requirements.txt

# Install tensorflow GPU support if needed--this will make your model train much faster.
pip install --user tensorflow-gpu

# Optional: if using Keras on GPU
# To enable Keras to run on GPU, you'll need to set CUDA_VISIBLE_DEVICES to the id of the GPU
# you want to use (typically 0 if your machine has one GPU)
export CUDA_VISIBLE_DEVICES=0

# Link to W&B -- this will track your training and save your run results.
# For cloud instances, you may need to update your PATH.
export PATH=~/.local/bin:$PATH
wandb init

# Download the train and validation data (~4.3GB) (default location: ``data`` in the repo)
bash download_data.sh

# Train a baseline model in Keras. Run with -h to see command line options
python train.py

# To quickly verify that the model is training, set epochs=1
python train.py --epochs=1
```

## Next Steps

Here are some ideas to try next:

* different network architectures, loss functions, optimizers, and other hyperparameter settings
* explore subsets of spectral bands and architectures that account for differences in the spectral bands
* data augmentation (rotate, flip) and narrowing the focus (center crop)
* comparison between formulating this task as a regression (predicting a continuous value for drought severity or forage quality) vs a classification (predicting a discrete label))
* explore correlations between the RGB ground-level photos and satellite images
* explore strategies to account for the class imbalance (roughly ~60% of the full data gathered is of class 0, classes 1 and 2 have ~15% each, and the remaining ~10% is class 3)
