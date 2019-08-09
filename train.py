#!/usr/bin/env python3

# keras_train.py
# --------------------
# Use Keras to train a simple CNN to predict a discrete
# indicator of forage quality (inversely related to drought severity) from satellite
# images in 10 frequency bands. The ground truth label is the number of
# cows that a human expert standing at the center of the satellite image at ground level
# thinks the surrounding land could support (0, 1, 2, or 3+)

import argparse
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers

import wandb
from wandb.keras import WandbCallback

tf.compat.v1.set_random_seed(1)

# for categorical classification, there are 4 classes: 0, 1, 2, or 3+ cows
NUM_CLASSES = 4
# fixed image counts from TFRecords
NUM_TRAIN = 86317
NUM_VAL = 10778

# default image side dimension (65 x 65 square)
IMG_DIM = 65
# use 7 out of 10 bands for now
NUM_BANDS = 7

# settings/hyperparams
# these defaults can be edited here or overwritten via command line
MODEL_NAME = ""
DATA_PATH = "../dw/droughtwatch_data"
BATCH_SIZE = 64
EPOCHS = 10
L1_SIZE = 32
L2_SIZE = 64
L3_SIZE = 128
FC_SIZE = 128
DROPOUT_1 = 0.2
DROPOUT_2 = 0.2

def class_weights_matrix():
  # define class weights to account for uneven distribution of classes
  # distribution of ground truth labels:
  # 0: ~60%
  # 1: ~15%
  # 2: ~15%
  # 3: ~10%
  class_weights = np.zeros((NUM_TRAIN, NUM_CLASSES))
  class_weights[:, 0] += 1.0
  class_weights[:, 1] += 4.0
  class_weights[:, 2] += 4.0
  class_weights[:, 3] += 6.0
  return class_weights

# data-loading and parsing utils
#----------------------------------
def load_data(data_path):
  train = file_list_from_folder("train", data_path)
  val = file_list_from_folder("val", data_path)
  return train, val

def file_list_from_folder(folder, data_path):
  folderpath = os.path.join(data_path, folder)
  filelist = []
  for filename in os.listdir(folderpath):
    if filename.startswith('part-') and not filename.endswith('gstmp'):
      filelist.append(os.path.join(folderpath, filename))
  return filelist

# data field specification for TFRecords
features = {
  'B1': tf.io.FixedLenFeature([], tf.string),
  'B2': tf.io.FixedLenFeature([], tf.string),
  'B3': tf.io.FixedLenFeature([], tf.string),
  'B4': tf.io.FixedLenFeature([], tf.string),
  'B5': tf.io.FixedLenFeature([], tf.string),
  'B6': tf.io.FixedLenFeature([], tf.string),
  'B7': tf.io.FixedLenFeature([], tf.string),
  'B8': tf.io.FixedLenFeature([], tf.string),
  'B9': tf.io.FixedLenFeature([], tf.string),
  'B10': tf.io.FixedLenFeature([], tf.string),
  'B11': tf.io.FixedLenFeature([], tf.string),
  'label': tf.io.FixedLenFeature([], tf.int64),
}        

def parse_tfrecords(filelist, batch_size, buffer_size):
  # try a subset of possible bands
  def _parse_(serialized_example, keylist=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']):
    example = tf.parse_single_example(serialized_example, features)
   
    def getband(example_key):
      img = tf.decode_raw(example_key, tf.uint8)
      return tf.reshape(img[:IMG_DIM**2], shape=(IMG_DIM, IMG_DIM, 1))
    
    bandlist = [getband(example[key]) for key in keylist]
    
    # combine bands into tensor
    image = tf.concat(bandlist, -1)
    # one-hot encode ground truth labels 
    label = tf.cast(example['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    return {'image': image}, label
    
  tfrecord_dataset = tf.data.TFRecordDataset(filelist)
  tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x)).shuffle(buffer_size).repeat(-1).batch(batch_size)
  tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
  image, label = tfrecord_iterator.get_next()
  return image, label

def build_regression_model(args):
  # initial regression model
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=[IMG_DIM, IMG_DIM, NUM_BANDS], name='image'))
  model.add(layers.Conv2D(filters=args.l1_size, kernel_size=(5, 5), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=args.l2_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Flatten())

  model.add(layers.Dense(units=args.fc_size, activation='relu'))
  model.add(layers.Dense(units=1, activation = 'sigmoid'))
  model.compile(loss=tf.keras.losses.mean_squared_error, 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['mse'])
  return model

def build_classification_model(args):
  # simple CNN for classifcation (default)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=[IMG_DIM, IMG_DIM, NUM_BANDS], name='image'))
  model.add(layers.Conv2D(filters=args.l1_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(filters=args.l2_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.Conv2D(filters=args.l2_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(args.dropout_1))
  
  model.add(layers.Conv2D(filters=args.l3_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.Conv2D(filters=args.l3_size, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(rate=args.dropout_2))
  model.add(layers.Flatten())

  model.add(layers.Dense(units=args.fc_size, activation='relu'))
  model.add(layers.Dense(units=50, activation='relu'))
  model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
  model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['accuracy'])
  return model

def train_cnn(args):
  # load training data in TFRecord format
  train_tfrecords, val_tfrecords = load_data(args.data_path)
  
  # initialize wandb logging for your project and save your settingsg
  wandb.init()
  config={
    "batch_size" : args.batch_size,
    "epochs": args.epochs,
    "l1_size" : args.l1_size,
    "l2_size" : args.l2_size,
    "l3_size" : args.l3_size,
    "fc_size" : args.fc_size,
    "dropout_1" : args.dropout_1,
    "dropout_2" : args.dropout_2,
    "n_train" : NUM_TRAIN,
    "n_val" : NUM_VAL
  }
  wandb.config.update(config)

  # load images and labels from TFRecords
  train_images, train_labels = parse_tfrecords(train_tfrecords, args.batch_size, NUM_TRAIN)
  val_images, val_labels = parse_tfrecords(val_tfrecords, args.batch_size, NUM_VAL)
  
  # number of steps per epoch is the total data size divided by the batch size
  train_steps_per_epoch = int(math.floor(float(NUM_TRAIN) /float(args.batch_size)))
  val_steps_per_epoch = int(math.floor(float(NUM_VAL)/float(args.batch_size)))
  
  model = build_classification_model(args)
  model.fit(train_images, train_labels, steps_per_epoch=train_steps_per_epoch, \
            epochs=args.epochs, class_weight=class_weights_matrix(), \
            validation_data=(val_images, val_labels), \
            validation_steps=val_steps_per_epoch, \
            callbacks=[WandbCallback()])
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-d",
    "--data_path",
    type=str,
    default=DATA_PATH,
    help="Path to data, containing train/ and val/")
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Number of images in training batch")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="Number of training epochs")
  parser.add_argument(
    "--l1_size",
    type=int,
    default=L1_SIZE,
    help="size of first conv layer")
  parser.add_argument(
    "--l2_size",
    type=int,
    default=L2_SIZE,
    help="size of second conv layer")
  parser.add_argument(
    "--l3_size",
    type=int,
    default=L3_SIZE,
    help="size of third conv layer")
  parser.add_argument(
    "--fc_size",
    type=int,
    default=FC_SIZE,
    help="size of first fully-connected layer")
  parser.add_argument(
    "--dropout_1",
    type=float,
    default=DROPOUT_1,
    help="first dropout rate")
  parser.add_argument(
    "--dropout_2",
    type=float,
    default=DROPOUT_2,
    help="second dropout rate") 
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "--quick_run",
    action="store_true",
    help="train quickly on a tenth of the data")   
  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name from command line
  if args.model_name:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_cnn(args) 
