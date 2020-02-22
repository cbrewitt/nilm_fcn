"""
Created on 15/02/18
@author Cillian Brewitt
"""
import pandas as pd
import re

class MetaData(object):
    """ Class to store metadata with methods for selecting sets sensors, homes
    etc. from the following tables from the IDEAL database:

        sensor, sensorbox, room, combinedhome, appliance

    Methods in this class also clean up some data inconsistencies, such
    as removing missing/orphaned items. The aim is to clean as much as possible
    without looking at sensor readings.
    """

    appliancetypes = (
        'mainthermostat','gasfire','electricfire','electricheater','aircon',
        'fridge','freezer','fridgefreezer','icemaker','wastedisposal','gashob',
        'electrichob','gasoven','electricoven','grill','toaster',
        'woodburningstove','microwave','shower', 'electricshower','bath',
        'hydrotherapybath','washingmachine','tumbledrier',
        'washingmachinetumbledrier','dishwasher','sink','kettle','boiler',
        'television','dvd','recorder','settop','computer','hifi','light',
        'other','aquarium','dehumidifier','vacuumcleaner','winecooler','electriccooker')

    oem2appliancetype = {
        'oven': 'electricoven',
        'shower': 'electricshower',
        'cooker': 'electriccooker',
        'water heater': 'other'
    }

    name2appliancetype = {
        'microwavesimple': 'microwave',
        'washingmachinedryer': 'washingmachinetumbledrier',
        'combinedfridgefreezer': 'fridgefreezer',
        'microwavegrillandoven': 'microwave',
        'clothesdryer': 'tumbledrier',
        'portablevacuumcleaner': 'vacuumcleaner',
        'separatefridge': 'fridge',
        'pressurecooker': 'other',
        'microwaveoven': 'microwave',
        'vacuum': 'vacuumcleaner',
        'portableheater': 'electricheater',
        'fixedelectricheaters': 'electricheater',
        'dryer': 'tumbledrier',
        'computer': 'other',
        'microwavegrill': 'microwave',
        'separatefreezer': 'freezer'
    }

    def __init__(self, tables):
        """Constructs MetaData object

        :param tables: dict-like
            Contains pandas Dataframes for tables from the IDEAL database

            This should have the keys ['sensor', 'sensorbox', 'room',
            'combinedhome', 'appliance'], and pandas dataframes as the
            corresponding values
        """

        # setting a separate attribute asserts that each table exists in tables,
        # and loads from HDF5 file if used
        self.sensor = tables['sensor']
        self.sensorbox = tables['sensorbox']
        self.room = tables['room']
        self.combinedhome = tables['home']
        self.appliance = tables['appliance']

        self._sensor_merged_cache = None

    def sensor_merged(self):
        """Joins sensor with sensorbox, room, combinedhome, and appliance.

            Missing values are dropped. Column names that appear in multiple
            tables are suffixed with '_tablename', e.g. 'notes_sensorbox'.

        :return: pandas Dataframe
            Sensor joined with sensorbox, room, combinedhome, and appliance
        """
        if self._sensor_merged_cache is None:

            sensors = pd.merge(self.sensor, self.sensorbox, 'inner',
                               on='sensorboxid',
                               suffixes=('_sensor', '_sensorbox'))
            sensors.drop('roomid_sensorbox', axis=1, inplace=True)
            sensors.rename(columns={'roomid_sensor': 'roomid'}, inplace=True)

            sensors = pd.merge(sensors, self.room, 'inner', 'roomid',
                               suffixes=('_sensor', '_room'))

            sensors = pd.merge(sensors, self.combinedhome, 'inner', 'homeid',
                               suffixes=('_room', '_home'))

            sensors = pd.merge(sensors, self.appliance, 'left', 'applianceid')
            sensors.drop(['roomid_y', 'homeid_y'], axis=1, inplace=True)
            sensors.rename(columns={'roomid_x': 'roomid',
                                    'homeid_x': 'homeid'}, inplace=True)

            # indices = ((sensors['status'] == 'study') &
            #            sensors['stdcategory'].isin(
            #                ['installed', 'droppedout', 'uninstalled']))
            # sensors = sensors[indices]

            self._sensor_merged_cache = sensors

        return self._sensor_merged_cache

    @staticmethod
    def _oem_notes2name(notes):
        notes_split = re.findall('ct[0-9]:[A-z\s]*(?=;)', notes)
        names_dict = {}
        for note in notes_split:
            counter, name = note.split(':')
            names_dict[int(counter[2])] = name.lower().strip()
        return names_dict

    def appliance_oem_sensors(self):
        """Gets sensorid and appliancetype of all openenergy appliance sensors

        :return: pandas Dataframe
            columns sensorid and appliancetype
        """
        sensors = self.oem_sensors()
        sensors = sensors.loc[sensors['oem_name'].isin(
            MetaData.oem2appliancetype.keys())]

        sensors['appliancetype'] = sensors['oem_name'].map(
            MetaData.oem2appliancetype)

        return sensors[['sensorid', 'appliancetype']]

    def oem_sensors(self):
        """Gets sensorid and name of all openenergy sensors

        :return: pandas Dataframe
            columns sensorid and oem_name
        """
        indices = ((self.sensor_merged()['sensorbox_type'] == 'subcircuit_monitor') &
                   (self.sensor_merged()['notes'].notnull()))
        sensors = self.sensor_merged().loc[indices,
                            ['sensorid', 'notes', 'counter']]
        get_oem_name = lambda row: MetaData._oem_notes2name(
            row['notes']).get(row['counter'])
        sensors['oem_name'] = sensors.apply(get_oem_name, axis=1)
        sensors = sensors.loc[sensors['oem_name'].notnull(),
                              ['sensorid','oem_name']]
        return sensors

    def appliance_zwave_sensors(self):
        """Gets the sensorid and appliancetype of all zwave sensors

        :return: pandas Dataframe
            columns sensorid and appliancetype
        """
        indices = self.sensor_merged()['sensorbox_type'] == 'plug_monitor'
        sensors = self.sensor_merged().loc[indices]

        # convert name to appliancetype
        clean_string = lambda s: s.lower().replace(' ', '')
        sensors.loc[:, 'name_clean'] = sensors['name'].apply(
            clean_string)

        indices = sensors['name_clean'].isin(MetaData.name2appliancetype.keys())

        sensors.loc[indices, 'name_clean'] = \
            sensors.loc[indices, 'name_clean'].map(MetaData.name2appliancetype)

        # fill in missing appliancetypes from name
        indices = ((sensors['appliancetype'].isnull() |
                    (sensors['appliancetype'] == 'other')) &
                    sensors['name_clean'].isin(
                       MetaData.appliancetypes))

        sensors.loc[indices, 'appliancetype'] = \
            sensors.loc[indices, 'name_clean']

        sensors.loc[sensors['appliancetype'].isnull(), 'appliancetype'] = 'other'
        return sensors[['sensorid', 'appliancetype']]

    def elec_appliance_sensors(self):
        """Gets the sensorid and appliancetype of all electrical appliance
        sensors

        :return: pandas Dataframe
            Columns for sensorid and appliancetype
        """
        raise NotImplementedError

    def mains_30A_rms_sensors(self):
        """Gets the sensorid of all 30A mains rms sensors

        :return: pandas Series
            sensorid of all 30A mains rms sensors
        """
        return self._mains_rms_senors('30A')

    def mains_100A_rms_sensors(self):
        """Gets the sensorid of all 100A mains rms sensors

        :return: pandas Series
            sensorid of all 100A mains rms sensors
        """
        return self._mains_rms_senors('100A')

    def _mains_rms_senors(self, current_range):
        indices = ((self.sensor_merged()['type_sensor'] == 'electric') &
                   (self.sensor_merged()['counter'] == 2) &
                   (self.sensor_merged()['currentrange'] == current_range))
        return self.sensor_merged().loc[indices, 'sensorid']

    def mains_oem_sensors(self):
        """Gets the sensorid of all mains Openenergy sensors

        :return: pandas Series
            sensorids
        """
        sensors = self.oem_sensors()
        return sensors.loc[(sensors['oem_name'] == 'mains'), 'sensorid']

    def electric_sensors(self):
        """Get sensorid of all electrical sensors
        :return: pandas Series
            sensorids
        """
        return pd.concat([self.mains_30A_rms_sensors(),
                          self.mains_100A_rms_sensors(),
                          self.mains_oem_sensors(),
                          self.appliance_oem_sensors()['sensorid'],
                          self.appliance_zwave_sensors()['sensorid']])

    def gold_homes(self):
        """Gets the homeid of all gold homes which have the expected sensors

        :return: pandas Series
            homeid of all gold homes
        """
        indices = self.sensor_merged()['install_type_home'] == 'enhanced'

        gold_sensors = self.sensor_merged().loc[indices, ['homeid','sensorid']]

        # get only homes that have both a 30A and 100A mains rms sensors
        gold_30A_homes = pd.merge(gold_sensors,
                            self.mains_30A_rms_sensors().to_frame(),
                            'inner', 'sensorid')['homeid']
        gold_100A_homes = pd.merge(gold_sensors,
                             self.mains_100A_rms_sensors().to_frame(),
                            'inner', 'sensorid')['homeid']

        gold_currentclamp = set(gold_30A_homes) & set(gold_100A_homes)

        return pd.Series(list(gold_currentclamp)).rename('homeid')

    def battery_sensors(self):
        sensors = self.sensor_merged()
        return sensors.sensorid[sensors.type_sensor == 'battery']
    
    def homes(self):
        return self.sensor_merged()['homeid'].unique()


if __name__ == '__main__':

    from data_store import MetaDataStore

    metadata = MetaData(MetaDataStore())
    print(metadata.gold_homes())

