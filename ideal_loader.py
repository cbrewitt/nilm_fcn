import sys
import numpy as np
import pandas as pd
from data_store import LOCAL_DATA_DIR, HomeReadingStore
from data_preprocessing import ActivationDetector

appliance_ideal2refit = {'kettle':'Kettle',
                         'microwave':'Microwave', 
                         'fridge':'Fridge', 
                         'dishwasher':'Dishwasher',
                         'washingmachine':'Washing Machine'}

def load_appliances(appliances, houses, path=LOCAL_DATA_DIR, history_length=None, merge=True,
                    treat_as_multi_appliance=False, sample_rate=8, split_wmtd=True):
    '''
    Load data for multiple appliances from multiple houses
    :param appliances: list of appliance names
    :param houses: list of house numbers
    :param path: path to  files
    :param history_length: maximum number of time steps to load from each house
    :param merge: merge data for appliance groups * not yet supported
    :return appliance_data_list: list of arrays of shape (time, appliance) for each house
    :return aggregate_data_list: list of array of aggregate data for each house
    :return timestamps_list: list of array of timestamps for each house
    :return house_appliances: boolean array of shape (house, appliance) showing appliances in each house
    '''
    appliance_data_list = []
    aggregate_data_list = []
    timestamps_list = []
    house_appliances = np.zeros((len(houses), len(appliances)))

    appliance_map = pd.read_csv('appliance_map.csv', dtype=np.int32, index_col=0)

    for house_idx, house in enumerate(houses):
        with HomeReadingStore() as s:
            readings = s.get_readings(house)
        columns = ['mains_apparent'] + [appliance for appliance in readings.keys()
                                                 if appliance in appliances]
        
        wmtd = 'washingmachinetumbledrier'
        if 'washingmachine' in appliances and wmtd in readings.keys():
            columns += [wmtd]
        
        readings = readings[columns]
        readings = readings.asfreq('{0}S'.format(sample_rate))
        readings.dropna(inplace=True)
        
        # split washingmachinetumbledrier
        # so far only stores washing machine
        if wmtd in readings.keys():
            activation_detector = ActivationDetector(wmtd,
                        rulesfile='rules.json')
            readings = activation_detector.split_wmtd_readings(readings)
            
        timestamps = readings.index.astype(np.int64).values // 10**9
        aggregate_data = readings.mains_apparent.values

        if history_length is None or history_length > timestamps.size:
            house_history_length = timestamps.size
        else:
            house_history_length = history_length

        start_idx = timestamps.size - house_history_length
        readings = readings.iloc[start_idx:]

        appliance_data = np.zeros((house_history_length, len(appliances)))
        for appliance_idx, appliance in enumerate(appliances):
            
            if (appliance in readings.keys()) and appliance_map.loc[house, appliance]:
                house_appliances[house_idx, appliance_idx] = 1
                
                appliance_data[:, appliance_idx] += readings[appliance].values
            
        if len(appliances) == 1 and not treat_as_multi_appliance:
            appliance_data = appliance_data.flatten()
        
        if house_appliances[house_idx].sum() > 0:
            appliance_data_list.append(appliance_data)
            aggregate_data_list.append(aggregate_data[start_idx:])
            timestamps_list.append(timestamps[start_idx:])
        
    house_appliances = house_appliances[house_appliances.sum(axis=1) > 0]

    if len(appliances) == 1 and not treat_as_multi_appliance:
        house_appliances = None

    return appliance_data_list, aggregate_data_list, timestamps_list, house_appliances

if __name__ == '__main__':
    
    sample_rate = 8
    appliance = 'kettle'
    refit_appliance = appliance_ideal2refit[appliance]
    train_homes = np.genfromtxt('train_homes.csv').astype(int)
    appliance_data_list, aggregate_data_list, timestamps_list, house_appliances \
        = load_appliances([appliance], train_homes, sample_rate=sample_rate)
    import pdb;pdb.set_trace()