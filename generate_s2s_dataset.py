import sys
import os
import argparse

import numpy as np
import pandas as pd
import datetime as dt

from data_store import HomeReadingStore, LOCAL_DATA_DIR
from data_preprocessing import ActivationDetector

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def home_s2s_multi_appliance(readings, appliances, sampling_freq, receptive_field, 
                             output_length):
    # TODO fix appliances present (get from metadatastore?)
    
    # resample
    freq_string = '{0}S'.format(sampling_freq)
    readings = readings.resample(freq_string).mean()
        
    home_appliances = [key for key in readings.keys() if key in appliances]

    if readings.shape[1] == 1:
        readings['dummy'] = 0
        home_appliances.append('dummy')
    
    # find gaps
    readings.iloc[0] = np.nan
    readings.iloc[-1] = np.nan
    readings['gaps'] = readings.mains_apparent.isnull()

    # everything within offset of a gap is also labelled gap
    if readings.shape[0] <= receptive_field:
        readings['gaps'] = True
    else:
        readings['gaps'] = np.convolve(readings.gaps.astype(np.int8).values,
                                       np.ones(receptive_field), mode='same') > 0
    
    # drop gaps that were filled in by resampler
    readings.dropna(subset=['mains_apparent'], inplace=True)
    
    offset = receptive_field // 2
    num_windows = int(np.ceil((readings.shape[0] - 2 * offset) / output_length))
    
    # pad the end of the dataframe so that it has an exact multiple of windows
    pad_size = num_windows * output_length + 2 * offset - readings.shape[0]
    nan_pad = pd.DataFrame(index=pd.DatetimeIndex(
        freq=freq_string, start=readings.index[-1], periods=pad_size + 1, closed='right'),
                                columns=readings.keys())
    readings = readings.append(nan_pad)
    
    # reshape into windows
    targets = readings[home_appliances].iloc[offset:-offset].values
    targets = targets.reshape((num_windows, output_length, len(home_appliances)))
    
    # timestamps
    time = readings.index[offset:-offset].values
    time = time.reshape((num_windows, output_length))
    
    # inputs
    input_length = 2 * offset + output_length
    if num_windows > 0:
        inputs = np.vstack([readings.mains_apparent.values[
                    i*output_length: i*output_length + input_length]
                      for i in range(num_windows)])
    else:
        inputs = np.zeros((0, input_length))
    
    # usable indices for each appliance
    targets_mask = np.logical_or(readings[home_appliances].isnull().values,
                                 readings.gaps.values.reshape((-1,1)))
    targets_mask = targets_mask[offset:-offset].reshape(
                            (num_windows, output_length, len(home_appliances)))
    
    # TODO this is wrong, appliances present should take into account appliances that are not sensored - get from metadata
    appliances_present = np.array([appliance in home_appliances for appliance in appliances])
    appliances_present = np.tile(appliances_present, (num_windows, 1))
    
    return inputs, targets, targets_mask, appliances_present, time


def all_s2s_multi_appliance(homes, appliances, sampling_freq, receptive_field, output_length):
    data_lists = [[], [], [], []]
    
    for home in homes:
        with HomeReadingStore() as s:
            readings = s.get_readings(home)
        data_arrays = home_s2s_multi_appliance(readings, appliances, sampling_freq,
                                                  receptive_field, output_length)
        for data_list, data_array in zip(data_lists, data_arrays):
            data_list.append(data_array)
            
    # inputs, targets, targets_mask, appliances_present
    return [np.vstack(data_list) for data_list in data_lists]


def get_activations_list(readings, appliance, sample_rate):
    activation_detector = ActivationDetector(appliance, sample_rate=sample_rate)
    activations = activation_detector.get_activations(readings[appliance])
    activations['duration'] = activations.end - activations.start

    # round start and end
    # activations['start'] = activations.start.apply(readings.index.searchsorted)
    # activations['end'] = activations.end.apply(readings.index.searchsorted)

    # put activations into list
    readings = readings.sort_index()
    return [readings.loc[row.start:row.end, appliance].values 
            for index, row in activations.iterrows()]


def all_s2s_single_appliance(homes, appliance, sampling_freq, receptive_field, output_length,
                            history_length=None, split_wmtd=True, data_dir=LOCAL_DATA_DIR, get_activations=True):
    
    appliance_map = pd.read_csv('appliance_map.csv', dtype=np.int32, index_col=0)
    data_lists = [[], [], [], []]
    window_weights_list = []
    activations_list = []
    
    for home in homes:
        if appliance_map.loc[home, appliance]:
            with HomeReadingStore(data_dir) as s:
                readings = s.get_readings(home)
            readings = readings.sort_index()

            wmtd = 'washingmachinetumbledrier'
            if split_wmtd and appliance == 'washingmachine' and wmtd in readings.keys():
                activation_detector = ActivationDetector(wmtd)
                readings = activation_detector.split_wmtd_readings(readings)
                del readings[wmtd]
            
            data_arrays = home_s2s_multi_appliance(readings, [appliance], sampling_freq,
                                                      receptive_field, output_length)
            
            data_arrays = [data_arrays[i] for i in [0, 1, 2, 4]] # delete appliance present
            data_arrays = [data_array.squeeze() for data_array in data_arrays]
            
            # drop windows that are completely masked
            drop_windows = data_arrays[2].all(axis=1).astype(np.bool)
    
            data_arrays = [data[~drop_windows] for data in data_arrays]
            
            # limit the history length by removing older windows
            num_windows = data_arrays[0].shape[0]
            if history_length is not None and num_windows * output_length > history_length: 
                start_idx = num_windows - history_length // output_length
                data_arrays = [a[start_idx:] for a in data_arrays]
                num_windows = data_arrays[0].shape[0]

            # get activations
            if num_windows > 0:
                start_time = data_arrays[3].min()
                readings = readings[start_time:]
                if get_activations:
                    activations_list.extend(get_activations_list(readings, appliance, sampling_freq))

                window_weights = np.ones(num_windows) / num_windows
                window_weights_list.append(window_weights)

                for data_list, data_array in zip(data_lists, data_arrays):
                    data_list.append(data_array)

    data_arrays = [np.vstack(data_list) for data_list in data_lists]

    # stack and normalise weights
    window_weights = np.hstack(window_weights_list)
    window_weights = window_weights / np.mean(window_weights)
    
    # put activations in array
    if get_activations:
        activations_list = [a for a in activations_list if ~np.isnan(a).any()]
        activation_length = np.array([len(activation) for activation in activations_list])
        max_activation_len = activation_length.max() if len(activations_list) > 0 else 0
        activations_array = np.zeros((len(activations_list), max_activation_len))

        for i in range(len(activations_list)):
            activation = activations_list[i]
            activations_array[i, :len(activation)] = activation
    else:
        activations_array = None
        activation_length = None
    
    data_arrays.append(activations_array)
    data_arrays.append(activation_length)
    data_arrays.append(window_weights)
    
    # inputs, targets, targets_mask, time, activations, activation_length, window_weights
    return data_arrays
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate s2s training data')
    parser.add_argument('--appliances', nargs='+', type=str, default=[
                            'kettle', 'microwave', 'dishwasher', 'washingmachine',
                            'electricshower', 'electriccooker'])
    parser.add_argument('--history', type=int, default=None) # history in weeks
    parser.add_argument('--directory', type=str, default='nilm')
    parser.add_argument('--sample_rate', type=int, default=8)
    parser.add_argument('--receptive_field', type=int, default=2053)
    parser.add_argument('--get_activations', action='store_true')
    args = parser.parse_args()

    target_dir = LOCAL_DATA_DIR + '/' + args.directory
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    train_homes = np.genfromtxt('train_homes.csv').astype(int)
    test_homes = np.genfromtxt('test_homes.csv').astype(int)
    valid_homes = np.genfromtxt('valid_homes.csv').astype(int)
    
    appliances = args.appliances
    sample_rate = args.sample_rate
    
    receptive_field = args.receptive_field
    output_length = receptive_field
    if args.history is not None:
        history_length = int(dt.timedelta(weeks=args.history).total_seconds() / sample_rate)
    else:
        history_length = None

    print(history_length)
    for appliance in appliances:
        print(appliance)
        for set_name, set_homes in (('train', train_homes),
                                    ('test', test_homes),
                                    ('valid', valid_homes)):

            inputs, targets, targets_mask, time, activations, activation_length, weights \
                = all_s2s_single_appliance(set_homes, appliance, sample_rate, receptive_field,
                                       output_length, history_length, get_activations=args.get_activations)[:7]


            name_array_pairs = [('inputs', inputs),
                                ('targets', targets),
                                ('targets_mask', targets_mask),
                                ('time', time),
                                ('activations', activations)]

            if args.get_activations:
                name_array_pairs.append(('activation_length', activation_length))
                name_array_pairs.append(('weights', weights))

            for array_name, array_data in name_array_pairs:

                save_path = LOCAL_DATA_DIR + '/{3}/s2s_{0}_{1}_{2}.npy'.format(
                                              appliance, set_name, array_name, args.directory)

                np.save(save_path, array_data)
