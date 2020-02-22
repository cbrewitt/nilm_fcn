"""
Created on 19/03/18
@author Cillian Brewitt
"""
import numpy as np
import pandas as pd
import datetime as dt
import gc
import json
import re

from electric import submeter

from metadata import MetaData
from data_store import MetaDataStore, ReadingDataStore

class ElecPreprocessor(object):

    sensors_to_merge = {5316: [3099],
                        5317: [3100],
                        5311: [4076, 4650],
                        5314: [5276]}

    def __init__(self, mains_clamp_max_fill=60, oem_appliance_max_fill=60, oem_mains_max_fill=60,
                 zwave_max_fill=3600, power_limit_30A=4000, sample_rate=1,
                 mains_max_power=20000, zwave_max_power=4000, oem_appliance_max_power=15000, oem_min_flatline=300,
                 resample_method='instant'):
        """
        Creates object for cleaning electrical readings

        :param mains_clamp_max_fill:  int
            maximum gap length in seconds in current clamp readings to backfill
        :param oem_max_fill: int
            maximum gap length in seconds in OEM readings to forward fill
        :param zwave_max_fill: int
            maximum gap length in seconds in ZWave readings to forward fill
        :param power_limit_30A: int
            Threshold in Watts above which to use 100A sensor readings instead
            of 30A sensor readings
        :param sample_rate: int
            sample rate in seconds of cleaned data
        """

        self.mains_clamp_max_fill = mains_clamp_max_fill
        self.oem_appliance_max_fill=oem_appliance_max_fill
        self.oem_mains_max_fill = oem_mains_max_fill
        self.zwave_max_fill=zwave_max_fill
        self.power_limit_30A = power_limit_30A
        self.sample_rate = sample_rate
        self.mains_max_power = mains_max_power
        self.zwave_max_power = zwave_max_power
        self.oem_appliance_max_power = oem_appliance_max_power
        self.oem_min_flatline = oem_min_flatline

        assert resample_method in ['instant', 'mean']
        self.resample_method = resample_method
        
    def convert_mains_clamp(self, readings):
        """adjust columns and clip values of mains current clamp readings"""
        readings.drop_duplicates(subset='time', keep='last', inplace=True)
        if "tenths_seconds_since_last_reading" in readings.columns:
            readings = readings[readings.tenths_seconds_since_last_reading != 0]
        readings.set_index('time', inplace=True)
        if readings.shape[0] == 0:
            readings.index = pd.DatetimeIndex([])
        if "tenths_seconds_since_last_reading" in readings.columns:
            readings['value'] = readings.value / (
                    readings.tenths_seconds_since_last_reading / 10)
            del readings['tenths_seconds_since_last_reading']
        readings.rename(columns={'value':'mains'}, inplace=True)
        
        # clip values
        readings.loc[readings.mains < 0, 'mains'] = 0
        readings.loc[readings.mains > self.mains_max_power] = self.mains_max_power
        
        return readings

    def process_mains_clamp(self, mains_30A_readings, mains_100A_readings, merge=True):
        """ Cleans and merges readings from 30A and 100A current clamp sensors

        :param mains_30A_readings: DataFrame
            readings from the 30A mains sensor
        :param mains_100A_readings: DataFrame
        :return: DataFrame
            processed data
        """
        
        # set time as index, drop duplicates, and convert to watts
        columns = ['time', 'value']
        if 'tenths_seconds_since_last_reading' in mains_30A_readings.columns:
            columns.append('tenths_seconds_since_last_reading')
        mains_30A_readings, mains_100A_readings = [self.convert_mains_clamp(readings[columns])
                           for readings in [mains_30A_readings, mains_100A_readings]]
        
        if not merge:
            mains_30A_readings, mains_100A_readings = [self.fill_small_gaps(readings, 
                             self.mains_clamp_max_fill, method='bfill') 
                           for readings in [mains_30A_readings, mains_100A_readings]]

        # merge 30A and 100A sensor readings
        readings_merged = mains_30A_readings.join(
            mains_100A_readings, how='outer', lsuffix='_30A', rsuffix='_100A')

        del mains_30A_readings
        del mains_100A_readings
        
        if merge:
            # fill gaps in 30A with 100A readings
            readings_merged['mains_apparent'] = readings_merged.mains_30A.\
                fillna(readings_merged.mains_100A)

            # use 100A readings when power is greater than power_limit_30A
            indices = ~readings_merged.mains_100A.isnull() & (
                    readings_merged.mains_100A > self.power_limit_30A)

            readings_merged.loc[indices, 'mains_apparent'] = \
                readings_merged.loc[indices, 'mains_100A']

            del readings_merged['mains_30A']
            del readings_merged['mains_100A']
            
            # backfill small gaps
            readings_merged = self.fill_small_gaps(readings_merged,
                                               self.mains_clamp_max_fill,
                                               method='bfill')

        readings_merged = readings_merged.astype(np.float32)
        return readings_merged
    
    def fill_small_gaps(self, readings, max_fill, method='ffill'):
        if readings.shape[0] > 0:
            # find large gaps
            # offset = -1 if method == 'ffill' else 0
            # readings['gaps'] = readings.index.to_series().diff().shift(offset) \
            #                    > dt.timedelta(seconds=max_fill)
            readings = readings.astype(float)
            readings = readings.resample(
                '{0}S'.format(self.sample_rate))

            if self.resample_method == 'instant':
                readings = readings.asfreq()
            elif self.resample_method == 'mean':
                readings = readings.mean()
            else:
                raise ValueError
            # readings['gaps'].fillna(method=method, inplace=True)
            #
            # # fill small gaps
            # readings.loc[~readings.gaps] = readings.loc[~readings.gaps].fillna(method=method)
            #
            # del readings['gaps']

            limit = max_fill//self.sample_rate
            if limit == 0:
                limit = None
            readings.fillna(method=method, inplace=True, limit=limit)

            # drop readings for large gaps
            readings.dropna(inplace=True)
        return readings

    def process_power_readings(self, readings, max_fill, max_power):
        if 'tenths_seconds_since_last_reading' in readings.columns:
            del readings['tenths_seconds_since_last_reading']
        readings.drop_duplicates(subset='time', keep='last', inplace=True)
        readings.set_index('time', inplace=True)
        readings.rename(columns={'value': 'power'}, inplace=True)
        readings['power'] = readings['power'].astype(np.float32)

        # don't process empty DataFrame
        if readings.shape[0] == 0:
            return readings

        # clip values
        readings.loc[readings.power < 0, 'power'] = 0
        readings.loc[readings.power > max_power] = 0

        # fill small gaps
        readings = self.fill_small_gaps(readings, max_fill, method='ffill')
        return readings

    def process_oem_appliance_readings(self, readings):
        """ preprocess OEM readings

        :param readings: DataFrame
            readings to preprocess
        """
        return self.process_power_readings(readings, self.oem_appliance_max_fill,
                                           self.oem_appliance_max_power)

    def process_oem_mains_readings(self, readings):
        """ preprocess OEM readings

        :param readings: DataFrame
            readings to preprocess
        """
        return self.process_power_readings(readings, self.oem_mains_max_fill,
                                           self.mains_max_power)

    def process_zwave_readings(self, readings):
        """ preprocess OEM readings

        :param readings: DataFrame
            readings to preprocess
        """
        readings['value'] = readings['value'].astype(np.float32) / 10
        return self.process_power_readings(readings, self.zwave_max_fill, self.zwave_max_power)

    def get_sensor_readings(self, sensorid, readings_getter, anomalous_sensors=None):
        """ gets sensor readings while merging duplicated sensors

        :param readings_getter: function
            takes sensorid as argument and returns readings DataFrame
        :return: DataFrame
            merged readings
        """

        if sensorid in self.sensors_to_merge:
            readings = pd.concat([readings_getter(sid) for sid in
             self.sensors_to_merge[sensorid]])
        else:
            readings = readings_getter(sensorid)
        readings.drop_duplicates('time', inplace=True)

        if anomalous_sensors is not None and sensorid in anomalous_sensors.sensorid.values:
            for idx, anomalous_period in anomalous_sensors[anomalous_sensors.sensorid == sensorid].iterrows():
                readings = readings.loc[(readings.time < anomalous_period.starttime)
                                        | (readings.time > anomalous_period.endtime)]
        return readings

    def find_oem_flatline(self, readings):
        readings = readings.to_frame()

        readings['flat'] = readings.diff() == 0
        readings['flatness_change'] = readings.flat.astype(np.int8).diff()
        flatness_changes = readings[readings.flatness_change != 0]
        flatness_changes['duration'] = -flatness_changes.index.to_series().diff(-1)

        flat_periods = flatness_changes[(flatness_changes.duration
                                         >= dt.timedelta(seconds=self.oem_min_flatline))
                                        & (flatness_changes.flatness_change == 1)]

        return flat_periods

        # readings['flat'] = False
        # for start_time, period in flat_periods.iterrows():
        #     end_time = start_time + period.duration
        #     readings.loc[start_time:end_time, 'flat'] = True

        # return readings.flat

    def get_home_readings(self, homeid, merge_mains_clamps=True, oem_mains_readings=True,
                          unusable_sensors=None, appliance_readings=True, cutoff_date=None):
        """ get processed and merged readings from locally stored reading data.
        Must run store_gold_elec_data_locally.py before calling this method

        :param homeid: int
            homeid of the home for which to retrive readings
        :return: DataFrame
            processed readings for electrical mains and appliances
        """
        anomalous_sensors = None
        if unusable_sensors is None:
            anomalous_sensors = pd.read_csv('anomalous_sensors.csv', dtype={'homeid':np.int32,
                                       'sensorid':np.int32, 'notes':str}, parse_dates=['starttime', 'endtime'])
            unusable_sensors = anomalous_sensors[(anomalous_sensors.starttime == pd.NaT)
                                                 & (anomalous_sensors.endtime == pd.NaT)].sensorid.values

        # get metadata and readings store
        with MetaDataStore() as s:
            metadata = MetaData(s)
        
        reading_store = ReadingDataStore()

        duplicated_sensors = [
            u for v in self.sensors_to_merge.values() for u in v]

        sensors = metadata.sensor_merged()
        indices = sensors['sensorid'].isin(reading_store.get_sensorids())\
            & (sensors['homeid'] == homeid)\
            & ~sensors.sensorid.isin(duplicated_sensors)
        
        indices = indices & ~sensors.sensorid.isin(unusable_sensors)
        
        sensors = sensors.loc[indices]
        
        # get sensorids
        mains_30A_sensorid, mains_100A_sensorid = [
            sensors.sensorid[sensors.sensorid.isin(ids)] for ids in [
                metadata.mains_30A_rms_sensors(),
                metadata.mains_100A_rms_sensors()]]

        dummy_readings = pd.DataFrame(
            columns=['time','value','tenths_seconds_since_last_reading'])
        dummy_readings['time'] = dummy_readings['time'].astype('datetime64[ns]')

        # get apparent power readings
        mains_30A_readings, mains_100A_readings = [
            self.get_sensor_readings(
                int(sid), reading_store.get_sensor_readings, anomalous_sensors)
            if (sid.shape[0] == 1) else dummy_readings.copy() for sid in [
                mains_30A_sensorid, mains_100A_sensorid]]
        
        if cutoff_date is not None:
            mains_30A_readings, mains_100A_readings = [readings[readings.time > cutoff_date]
                                  for readings in [mains_30A_readings, mains_100A_readings]]
        
        readings_processed = self.process_mains_clamp(mains_30A_readings,
                                mains_100A_readings, merge=merge_mains_clamps)

        del mains_30A_readings, mains_100A_readings

        oem_sensors = []

        if appliance_readings:
            # get oem and zwave appliance readings
            oem_appliances = metadata.appliance_oem_sensors()
            indices = oem_appliances.sensorid.isin(sensors.sensorid)
            oem_appliances = oem_appliances[indices]

            oem_sensors.extend(list(oem_appliances.appliancetype.values))

            zwave_appliances = metadata.appliance_zwave_sensors()
            indices = zwave_appliances.sensorid.isin(sensors.sensorid)
            zwave_appliances = zwave_appliances[indices]

            for appliances, readings_processor in zip(
                    [oem_appliances, zwave_appliances],
                    [self.process_oem_appliance_readings, self.process_zwave_readings]):

                for index, row in appliances.iterrows():

                    readings = self.get_sensor_readings(int(row.sensorid),
                                            reading_store.get_sensor_readings, anomalous_sensors)

                    if cutoff_date is not None:
                        readings = readings[readings.time > cutoff_date]

                    readings = readings_processor(readings)
                    readings.rename(columns={'power': row.appliancetype},
                                    inplace=True)

                    # merge multiple appliances of same type
                    if row.appliancetype in readings_processed.keys():
                        readings_processed[row.appliancetype] = \
                            readings_processed[row.appliancetype] + \
                            readings[row.appliancetype]
                        readings_processed[row.appliancetype].fillna(
                            readings[row.appliancetype])
                    else:
                        readings_processed = readings_processed.join(
                            readings, how='left')
                    del readings

                    gc.collect()

        if oem_mains_readings:
            # get oem mains readings
            mains_oem_sensorid = sensors.sensorid[sensors.sensorid.isin(
                metadata.mains_oem_sensors())]

            if len(mains_oem_sensorid) == 1:

                mains_oem_readings = self.get_sensor_readings(int(mains_oem_sensorid),
                                            reading_store.get_sensor_readings, anomalous_sensors) \
                    if (mains_oem_sensorid.shape[0] == 1) else dummy_readings.copy()

                if cutoff_date is not None:
                    mains_oem_readings = mains_oem_readings[mains_oem_readings.time > cutoff_date]

                mains_oem_readings = self.process_oem_mains_readings(mains_oem_readings)
                readings_processed = readings_processed.join(mains_oem_readings, how='left')
                del mains_oem_readings
                readings_processed.rename(columns={'power': 'mains_real'},
                                          inplace=True)

                oem_sensors.append('mains_real')

                # replace OEM flatlines with NAN
                if readings_processed.shape[0] > 0:
                    oem_flat_periods = self.find_oem_flatline(readings_processed.mains_real)

                    for start_time, period in oem_flat_periods.iterrows():
                        end_time = start_time + period.duration
                        readings_processed.loc[start_time:end_time, oem_sensors] = np.nan

                readings_processed.loc[readings_processed.mains_real.isnull(), oem_sensors] = np.NaN

        # close files
        reading_store.close()

        return readings_processed


class ReadingConsistencyTester:
    
    def __init__(self):
        raise NotImplementedError


class ActivationDetector:

    def __init__(self, appliance, rulesfile='gt_rules.json', sample_rate=1):
        
        self.rulesfile=rulesfile
        
        with open(rulesfile) as data_file:
            rules = json.load(data_file)['rules']

        self.rule = None
        for rule in rules:
            pattern = re.compile(rule["appliance"])
            if not (pattern.match(appliance) is None):
                self.rule = rule
                break

        assert self.rule is not None
        self.appliance = appliance

        self.min_off_duration=int(self.rule["min_off_duration"])
        self.min_on_duration=int(self.rule["min_on_duration"])
        self.max_on_duration=int(self.rule["max_on_duration"])
        self.on_power_threshold=int(self.rule["on_power_threshold"])
        
        # minumum activation energy in joules
        
        if "min_energy" in self.rule:
            self.min_energy = int(self.rule["min_energy"])
        else:
            self.min_energy = 0

        self.sample_rate = sample_rate

    def get_activations(self, readings):
        """Get start, end and energy (joules) of appliance activations"""
        #resample and add buffer to start and end
        buffer = dt.timedelta(seconds=self.min_off_duration+self.sample_rate*2)
        start_time = readings.index[0]
        end_time = readings.index[-1]

        readings.loc[start_time - buffer] = 0
        readings.loc[end_time + buffer] = 0
        readings.sort_index(inplace=True)

        readings = (readings
                    .resample('{0}S'.format(self.sample_rate))
                    .fillna('nearest', 1)
                    .fillna(0))

        end = readings.index[-1]
        on_offs = submeter().get_ons_and_offs(readings.rename(columns=['values']), end, None,
                                            min_off_duration=self.min_off_duration,
                                            min_on_duration=self.min_on_duration,
                                            max_on_duration=self.max_on_duration,
                                            on_power_threshold=self.on_power_threshold)
        
        starts = on_offs.time[on_offs.state_change=='on'].values
        if on_offs.iloc[-1].state_change=='on':
            starts = starts[:-1]
        
        ends = on_offs.time[on_offs.state_change=='off'].values
        if on_offs.iloc[0].state_change=='off':
            ends = ends[1:]
        
        activations = pd.DataFrame({'start':starts, 'end':ends})
        if activations.shape[0] > 0:
            activations['energy'] = activations.apply(lambda r: readings[r.start:r.end].sum(),
                                                      axis=1) * self.sample_rate
        else:
            activations['energy'] = []
            
        activations = activations[activations.energy > self.min_energy]
        
        return activations
    
    def split_washingmachinetumbledrier(self, activation, tumbledrier_heating_min_off=1800,
                                   washingmachine_min_on=900):
        """split a washingmachinetumbledrier activation into washing machine and tumble drier"""
        
        assert self.appliance == 'tdheating'
        start = activation.index[0]
        end = activation.index[-1]

        heating_events = self.get_activations(activation)

        if heating_events.shape[0] > 0 and (end - heating_events.end.iloc[-1] <= 
                                dt.timedelta(seconds=tumbledrier_heating_min_off)):
            tumbledrier_start = heating_events.start.iloc[-1]
        else:
            tumbledrier_start = end

        if tumbledrier_start - start < dt.timedelta(seconds=washingmachine_min_on):
            tumbledrier_start = start

        return tumbledrier_start
    
    def split_wmtd_readings(self, readings):
        """ split readings into washing machine and tumble drier"""
        
        wmtd = 'washingmachinetumbledrier'
        assert self.appliance == wmtd
        
        if 'washingmachine' not in readings.keys():
            readings['washingmachine'] = 0
        if 'tumbledrier' not in readings.keys():
            readings['tumbledrier'] = 0
            
        activations = self.get_activations(readings[wmtd])
        activation_detector = ActivationDetector('tdheating', rulesfile=self.rulesfile)
        
        for index, row in activations.iterrows():
            activation = readings.loc[row.start:row.end, wmtd]
            td_start = activation_detector.split_washingmachinetumbledrier(activation)
            readings.loc[row.start:td_start, 'washingmachine'] += readings.loc[
                                                                       row.start:td_start, wmtd]
            readings.loc[td_start:row.end, 'tumbledrier'] += readings.loc[td_start:row.end, wmtd]
        
        return readings


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    preprocessor = ElecPreprocessor()
    readings = preprocessor.get_home_readings(228)
    readings.loc['2018-05-28':'2018-06-02'].plot()
    plt.show()