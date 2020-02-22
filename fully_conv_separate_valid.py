import sys
import os
import argparse

import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Conv1D, Flatten, Reshape, Dropout, Input, multiply, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils import multi_gpu_model

import ideal_meta
from data_store import LOCAL_DATA_DIR
from generate_s2s_dataset import all_s2s_single_appliance


model_name = 'fully_conv_separate_valid'


class MyModelCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        pass


parser = argparse.ArgumentParser(description='Generate s2s training data')
parser.add_argument('--appliance', default='kettle')
parser.add_argument('--predict_only', action='store_true', default=False)
args = parser.parse_args()

appliance = args.appliance

appliance_stats = pd.read_csv('appliance_stats.csv')
appliance_stats = appliance_stats[appliance_stats.appliancetype==appliance]
mean_on_power = float(appliance_stats.mean_on_power)
std_on_power = float(appliance_stats.std_on_power)

gpus = 1
num_epochs = 200
batch_size = 128 * gpus
receptive_field = 2053
sampling_freq = 8

IDEAL_DATA_DIR = os.environ['IDEAL_DATA_DIR']

appliance_model_name = model_name + '_{0}'.format(appliance)
model_dir = IDEAL_DATA_DIR + '/models/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + appliance_model_name + '.h5'
stats_dir = IDEAL_DATA_DIR + '/stats/'
if not os.path.exists(stats_dir):
    os.mkdir(stats_dir)
stats_path = stats_dir + appliance_model_name +'.csv'
predictions_dir = IDEAL_DATA_DIR + '/predictions/'
if not os.path.exists(predictions_dir):
    os.mkdir(predictions_dir)

# load training data
train_inputs, train_targets, train_targets_mask = [
  np.load(LOCAL_DATA_DIR + '/nilm/s2s_{0}_train_{1}.npy'.format(appliance, array_name))
             for array_name in ('inputs', 'targets', 'targets_mask')]

# normalise data
train_inputs = (train_inputs - train_inputs.mean(axis=1).reshape((-1, 1))) / ideal_meta.mains_std

train_targets = train_targets / mean_on_power

train_targets_mask = train_targets_mask.astype(np.bool)
train_inputs = np.nan_to_num(train_inputs)
train_targets = np.nan_to_num(train_targets)
np.putmask(train_targets, train_targets_mask, 0)
train_targets_mask = ~train_targets_mask  # masked targets are now zero

input_window_length = train_inputs.shape[1]
output_window_length = train_targets.shape[1]
offset = (input_window_length - output_window_length) // 2

# get validation data
valid_inputs, valid_targets, valid_targets_mask = [
  np.load(LOCAL_DATA_DIR + '/nilm/s2s_{0}_valid_{1}.npy'.format(appliance, array_name))
             for array_name in ('inputs', 'targets', 'targets_mask')]

# normalise data
valid_inputs = (valid_inputs - valid_inputs.mean(axis=1).reshape((-1,1))) / ideal_meta.mains_std

valid_targets = valid_targets / mean_on_power

valid_targets_mask = valid_targets_mask.astype(np.bool)
valid_inputs = np.nan_to_num(valid_inputs)
valid_targets = np.nan_to_num(valid_targets)
np.putmask(valid_targets, valid_targets_mask, 0)
valid_targets_mask = ~valid_targets_mask  # masked targets are now zero

# define model
main_input = Input(shape=(input_window_length,), name='main_input')
x = Reshape((input_window_length, 1),  input_shape=(input_window_length,))(main_input)

x = Conv1D(128, 9, padding='same', activation='relu', dilation_rate=1)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=2)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=8)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=16)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=32)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=64)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=128)(x)
x = Conv1D(128, 3, padding='same', activation='relu', dilation_rate=256)(x)
x = Conv1D(256, 1, padding='same', activation='relu')(x)
x = Conv1D(1, 1, padding='same', activation=None)(x)

x = Reshape((input_window_length,),  input_shape=(input_window_length,1))(x)

x = Lambda(lambda x: x[:, offset:-offset], output_shape=(output_window_length,))(x)

targets_mask = Input(shape=(output_window_length,), name='targets_mask')
main_output = multiply([x, targets_mask])
single_model = Model(inputs=[main_input, targets_mask], outputs=[main_output])

# summary
single_model.summary()

if gpus > 1:
    model = multi_gpu_model(single_model, gpus=gpus)
else:
    model = single_model

if args.predict_only:
        single_model.load_weights(model_path)
else:
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('Training samples: {0}'.format(train_inputs.shape))

    # train model

    model_checkpoint = MyModelCheckpoint(model_path, monitor='val_loss', save_best_only=False,
                                 save_weights_only=True)
    model_checkpoint.model = single_model
    callbacks = [model_checkpoint,
                 EarlyStopping(monitor='val_loss', patience=15),
                 CSVLogger(stats_path),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                  mode='auto', cooldown=0, min_lr=0.000001)]

    model.fit(x=[train_inputs, train_targets_mask],
              y=train_targets,
              epochs=num_epochs,
              batch_size=batch_size,
              verbose=1,
              callbacks=callbacks,
              validation_data=[[valid_inputs, valid_targets_mask], valid_targets],
              shuffle=True)

# predict
test_homes = np.genfromtxt('test_homes.csv').astype(int)
valid_homes = np.genfromtxt('valid_homes.csv').astype(int)
appliance_map = pd.read_csv('appliance_map.csv', dtype=np.int32, index_col=0)

for home in np.concatenate((test_homes, valid_homes)):
    if appliance_map.loc[home, appliance]:

        results_name = '/home_{0}/{1}'.format(home, appliance)
        print(results_name)

        test_inputs, test_targets, test_targets_mask, time = all_s2s_single_appliance(
            [home], appliance, sampling_freq, receptive_field, output_window_length)[:4]

        test_targets_mask = test_targets_mask.astype(np.bool)
        test_inputs_norm = (test_inputs - test_inputs.mean(axis=1).reshape((-1, 1))) / ideal_meta.mains_std
        test_inputs_norm = np.nan_to_num(test_inputs_norm)
        predictions = model.predict([test_inputs_norm, ~test_targets_mask], verbose=1)

        # process predictions
        predictions = predictions * mean_on_power
        predictions[test_targets_mask.astype(np.bool)] = np.nan
        predictions[predictions < 0] = 0

        # put results into dataframe
        results = pd.DataFrame(
                index=pd.to_datetime(time.flatten(), unit='s'),
                data={'mains': test_inputs[:, offset:-offset].flatten(),
                      'predictions': predictions.flatten(),
                      'ground_truth': test_targets.flatten()})

        results.dropna(inplace=True)

        # store results
        pred_store_name = IDEAL_DATA_DIR + '/predictions/{0}.h5'.format(model_name)

        with pd.HDFStore(pred_store_name) as s:
            s[results_name] = results
