#!/bin/sh

echo "Downloading data"
curl -SL https://storage.googleapis.com/wandb_datasets/dw_train_86K_val_10K.zip > dw_data.zip
unzip dw_data.zip
rm dw_data.zip
mv droughtwatch_data/ data/
