import sys

import os
import argparse
import pandas as pd
import datetime as dt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

import ideal_meta
import ideal_loader
from data_provider import S2PDataProvider

def load_train_data():
    # load train data
    y_train, x_train, t_train, _ \
        = ideal_loader.load_appliances([appliance], train_homes, sample_rate=sample_rate)

    y_valid, x_valid, t_valid, _ \
        = ideal_loader.load_appliances([appliance], valid_homes, sample_rate=sample_rate)
        
    return x_train, y_train, t_train, x_valid, y_valid, t_valid

def load_test_data(home):
    # load data
    y_test, x_test, t_test, house_appliances \
        = ideal_loader.load_appliances([appliance], [home], sample_rate=sample_rate)
        
    return x_test, y_test, t_test

def normalise_inputs(inputs_batch):
    return (inputs_batch - inputs_batch.mean(axis=1).reshape((-1,1))) / ideal_meta.mains_std

def normalise_targets(targets_batch):
    return (targets_batch - mean_on_power) / std_on_power


def predictions_transform(predictions):
    return predictions * std_on_power + mean_on_power

def create_model(input_window_length):
    
    model = Sequential()
    model.add(Reshape([input_window_length, 1],  input_shape=[input_window_length]))
    model.add(Conv1D(30, 10, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(30, 8, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(40, 6, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 5, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 5, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))
    
    return model

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train pointnet model on IDEAL')
    parser.add_argument('--appliance', default='kettle')
    parser.add_argument('--predict_only', action='store_true')
    args = parser.parse_args()
    
    sample_rate = 8
    gpus = 1
    appliance = args.appliance
    #refit_appliance = ideal_loader.appliance_ideal2refit[appliance]
    train_homes = np.genfromtxt('train_homes.csv').astype(int)
    valid_homes = np.genfromtxt('valid_homes.csv').astype(int)
    test_homes = np.genfromtxt('test_homes.csv').astype(int)
    batch_size = 1024 * gpus
    input_window_length = 599#refit_meta.pointnet_window_length[refit_appliance]
    IDEAL_DATA_DIR = os.environ['IDEAL_DATA_DIR']
    model_name = 'pointnet_large_separate_valid'
    appliance_model_name = model_name + '_{0}'.format(appliance)
    model_path = IDEAL_DATA_DIR + '/models/' + appliance_model_name + '.h5'
    stats_path = IDEAL_DATA_DIR + '/stats/' + appliance_model_name +'.csv'
    
    appliance_stats = pd.read_csv('appliance_stats.csv')
    appliance_stats = appliance_stats[appliance_stats.appliancetype==appliance]

    mean_on_power = float(appliance_stats.mean_on_power)
    std_on_power = float(appliance_stats.std_on_power)
    
    # create keras model
    model = create_model(input_window_length)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    
    
    if args.predict_only:
        model.load_weights(model_path)
    else:
        # use maximum 8 weeks of data from each home
        history_length = int(dt.timedelta(weeks=8).total_seconds() / sample_rate)

        x_train, y_train, t_train, x_valid, y_valid, t_valid = load_train_data()

        train_provider = S2PDataProvider(x_train, y_train, t_train, batch_size=batch_size,
                        input_window_size=input_window_length,
                              shuffle_order=True, inputs_transform=normalise_inputs,
                                         targets_transform=normalise_targets)

        valid_provider = S2PDataProvider(x_valid, y_valid, t_valid, batch_size=batch_size,
                        input_window_size=input_window_length,
                              shuffle_order=True, inputs_transform=normalise_inputs,
                                         targets_transform=normalise_targets)

        print('Training samples: {0}'.format(train_provider.indices.shape))
        print('Validation examples: {0}'.format(valid_provider.indices.shape))

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        model.summary()

        callbacks = [ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                     save_weights_only=True),
                     EarlyStopping(monitor='val_loss', patience=10),
                     CSVLogger(stats_path)]

        model.fit_generator(train_provider.generator(), train_provider.num_batches, epochs=100,
                            verbose=1, validation_data=valid_provider.generator(),
                            validation_steps=valid_provider.num_batches, callbacks=callbacks) 
    
    # make predictions
    predictions_store = pd.HDFStore(IDEAL_DATA_DIR + '/predictions/{0}.h5'.format(model_name))

    for home in test_homes:
        x_test, y_test, t_test = load_test_data(home)
        
        if len(x_test) > 0:
            
            results_name = '/home_{0}/{1}'.format(home, appliance)
            print(results_name)
            
            test_provider = S2PDataProvider(x_test, y_test, t_test, batch_size=batch_size,
                                  input_window_size=input_window_length,
                                  shuffle_order=False, inputs_transform=normalise_inputs)

            predictions = model.predict_generator(test_provider.generator(), 
                                                  test_provider.num_batches, verbose=1)

            predictions = predictions.flatten()

            predictions = predictions_transform(predictions)
            
            # clip negative values
            predictions[predictions < 0] = 0
            
            t_test = t_test[0][test_provider.indices]
            x_test = x_test[0][test_provider.indices]
            y_test = y_test[0][test_provider.indices]
            
            # put results into dataframe and store
            results = pd.DataFrame(
                index=pd.to_datetime(t_test, unit='s'),
                data={'mains':x_test,
                     'predictions':predictions, 
                     'ground_truth':y_test})

            predictions_store[results_name] = results
