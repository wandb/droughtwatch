#!/usr/bin/env python3
# train.py
# ----------
# Use Tensorflow Estimators to train a simple CNN to predict a continuous
# indicator of forage quality between 0 and 1, with lower values indicating
# lower forage quality (higher drought severity) and higher values indicating
# higher forage quality (lower drought severity) 

import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers
import wandb
from wandb.tensorflow import WandbHook

MODEL_NAME = ""
DATA_PATH = "data"
BATCH_SIZE = 32
EPOCHS = 10
L1_SIZE = 32
L2_SIZE = 64
FC_SIZE = 128

# DATA UTILS
#------------------
def load_data(data_path):
  train = file_list_from_folder("train", data_path)
  test = file_list_from_folder("test", data_path)
  return train, test

def file_list_from_folder(folder, data_path):
  folderpath = os.path.join(data_path, folder)
  filelist = []
  for filename in os.listdir(folderpath):
    if filename.startswith('part-') and not filename.endswith('gstmp'):
      filelist.append(os.path.join(folderpath, filename))
    else:
      print('Omitted', filename)
  return filelist

features = {
    'B1': tf.FixedLenFeature([], tf.string),
    'B2': tf.FixedLenFeature([], tf.string),
    'B3': tf.FixedLenFeature([], tf.string),
    'B4': tf.FixedLenFeature([], tf.string),
    'B5': tf.FixedLenFeature([], tf.string),
    'B6': tf.FixedLenFeature([], tf.string),
    'B7': tf.FixedLenFeature([], tf.string),
    'B8': tf.FixedLenFeature([], tf.string),
    'B9': tf.FixedLenFeature([], tf.string),
    'B10': tf.FixedLenFeature([], tf.string),
    'B11': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
}        

def parse_tfrecords(filelist, batch_size, num_epochs):
  def _parse_(serialized_example, keylist=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']):
    example = tf.parse_single_example(serialized_example, features)
    def getband(example_key):
      img = tf.decode_raw(example_key, tf.uint8)
      return tf.reshape(img[:4225], shape=(65, 65, 1))
    bandlist = [getband(example[key]) for key in keylist]
    # combine bands into tensor
    image = tf.concat(bandlist, -1)
    label = tf.cast(example['label'], tf.int32)
    # divide the label by 3 so it's between 0 and 1
    label = tf.truediv(label, 3)
    return {'image': image}, label
    
  tfrecord_dataset = tf.data.TFRecordDataset(filelist)
  tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x)).shuffle(True).batch(batch_size).repeat(num_epochs)
  tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
  return tfrecord_iterator.get_next()

def build_estimator_from_model_original(args):
  final_bias_init = initializers.Constant(value=0.249)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=[65,65,10], name='image'))
  model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
  model.add(layers.AveragePooling2D())
  model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
  model.add(layers.AveragePooling2D())
  model.add(layers.Flatten())

  model.add(layers.Dense(units=120, activation='relu'))
  model.add(layers.Dense(units=84, activation='relu'))
  model.add(layers.Dense(units=1, activation = 'sigmoid', bias_initializer=final_bias_init))
  model.compile(loss=tf.keras.losses.mean_squared_error, 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['mse'])
  estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
  return estimator

def build_estimator_from_model_test(args):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=[65,65,10], name='image'))
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
  estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
  return estimator


def train_cnn(args):
  # load training data
  train, test = load_data(args.data_path)
  N_TRAIN = len(train)
  N_TEST = len(test)
  print("Num train: ", N_TRAIN, " Num test: ", N_TEST) 
  
  # initialize wandb logging for your project
  wandb.init()
  config={
    "batch_size" : args.batch_size,
    "epochs": args.epochs,
    "n_train" : N_TRAIN,
    "n_test": N_TEST,
    "l1_size" : args.l1_size,
    "l2_size" : args.l2_size,
    "fc_size" : args.fc_size,
    "loss_type" : "mse"
  }
  wandb.config.update(config)
 
  estimator = build_estimator_from_model_test(args)
  max_steps = (float(N_TRAIN) / float (args.batch_size)) * args.epochs
  print("MAX STEPS: ", max_steps)
  train_spec = tf.estimator.TrainSpec(input_fn=lambda: parse_tfrecords(train, args.batch_size, args.epochs),
                                      max_steps=max_steps,
                                      hooks=[WandbHook()])
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: parse_tfrecords(test, args.batch_size, 1))
                                    #steps=int(float(N_TEST)/float(args.batch_size)),
                                    #hooks=[WandbHook()])
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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
    help="Path to data, containing train/ and test/")
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
    "--fc_size",
    type=int,
    default=FC_SIZE,
    help="size of first fully-connected layer")
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
