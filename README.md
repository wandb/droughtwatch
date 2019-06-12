# Weights & Biases Benchmark for Drought Prediction

This is a deep learning/computer vision project to predict drought severity from satellite imagery and expert-labeled ground-level photos.
It is instrumented with [Weights & Biases](https://www.wandb.com) to track and visualize model training and facilitate collaboration.

## How to participate

You can learn more and [join the benchmark here](https://app.wandb.ai/wandb/droughtwatch/benchmark).

## Dataset

The current dataset consists of 102,457 train and 5412 test satellite images, 65x65 pixels, in 11 spectrum bands. Human experts (pastoralists) have labeled these with the number of cows that the corresponding geographic location could support (0, 1, 2, or 3+ cows). More data will become available and described here in the near future.

## Usage

Please refer to the [benchmark instructions](https://app.wandb.ai/wandb/droughtwatch/benchmark) to get started.
Run ``python train.py -h`` to see all the existing options&mdash;hyperparameters and config can be set at the top of each script or overriden via the command line for convenience.

## Next Steps

Here are some ideas to try next:
* Keras/PyTorch integration for faster experimentation
* reformulations of the problem as a classification instead of a regression
* explore the data distribution and class balance: are different levels of drought intensity/forage quality evenly represented? what kind of new data would be most helpful to label?
* explore correlations between the sparse expert data&mdash;RGB ground-level photos&mdash;and the dense, lower quality unlabeled data&mdash;lower resolution satellite imagery in 11 spectral bands
* different network architectures
