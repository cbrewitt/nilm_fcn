"""
This file contains a class for storing a local copy of data from the IDEAL
 csv files
"""

import os
import glob
import argparse

from data_store import MetaDataStore, ReadingDataStore, LOCAL_DATA_DIR
from metadata import MetaData
import pandas as pd
import logging


class IdealCSVReader(object):
    """This class contains methods which read the IDEAL csv files
    """

    def __init__(self, dataset_dir):
        """
        :param dataset_dir: directory of the IDEAL dataset
        """
        self.dataset_dir = dataset_dir

    def get_metadata_table(self, table_name):
        """Loads a metadata table from csv file into a pandas dataframe

        :param table_name: name of the table
        :type table_name: str
        :return: dataframe containing data from tabe
        :rtype: pandas.Dataframe
        """
        filepath = self.dataset_dir + '/metadata/' + table_name + '.csv'
        df = pd.read_csv(filepath)
        # fix types
        return df

    def get_sensor_readings(self, sensorid):
        """ Loads the readings for one sensor into a dataframe with columns:
                time, value, tenths_seconds_since_last_reading

        TODO: read in chunks to avoid large memory consumption

        :param sensorid: id of the sensor for which to get readings
        :type sensorid: int
        :return: dataframe containing sensor readings
        :rtype: pandas.Dataframe
        """

        readings_dir = self.dataset_dir + '/sensordata/'
        filenames = glob.glob(readings_dir + '*sensor{}_*.csv.gz'.format(sensorid))

        if len(filenames) > 0:
            readings = pd.read_csv(filenames[0], compression='gzip', parse_dates=[0],
                                   names=['time', 'value'])
        else:
            readings = pd.DataFrame(columns=['time', 'value'])

        # use smaller int types
        readings['value'] = readings['value'].astype(dtype='int32')
        return readings


class IdealCSV2Hdf5(object):
    """Copies data from the ideal csv files to local HDF5 files"""

    default_tables = ('sensor', 'sensorbox', 'appliance', 'room',
                      'home')

    def __init__(self, dataset_dir, data_dir=LOCAL_DATA_DIR):
        """ Initialise Idealdb2Hdf5Converter object

        :param dataset_dir: directory of the IDEAL dataset
        :param data_dir:
        """

        self.reader = IdealCSVReader(dataset_dir=dataset_dir)
        self.data_dir = data_dir
        self._make_data_dir()

    def _make_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def store_metadata(self, table_names=default_tables):
        """Stores tables from IDEAL database containing metadata.

        Default tables are :
            ('sensor', 'sensorbox', 'appliance', 'room', 'home')

        This methods uses the first database connection if there are multiple
        database conections.

        :param table_names: Names of the tables to store
        :type table_names: list(str)
        """
        store = MetaDataStore(self.data_dir)
        for table_name in table_names:
            store[table_name] = self.reader.get_metadata_table(table_name)
        store.close()

    def store_readings(self, sensorid):
        """Stores readings sensors.

        :param sensorid: int
            sensorid of sensors to store readings for
        """
        # store readings for each sensor
        readings = self.reader.get_sensor_readings(sensorid).sort_values('time')
        store = ReadingDataStore(self.data_dir)
        store.set_sensor_readings(sensorid, readings)
        store.close()


def main():
    # set up logging
    logger = logging.getLogger('store_data_locally')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('store_data_locally.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # get arguments
    parser = argparse.ArgumentParser(description='Query sensor readings from the IDEAL database'
                                                 'and store locally.')
    parser.add_argument('--dataset_path', help='directory of the original IDEAL dataset')
    parser.add_argument('--data_path', default=LOCAL_DATA_DIR, help='directory to store data')

    args = parser.parse_args()

    # store metadata locally
    converter = IdealCSV2Hdf5(args.dataset_path, data_dir=args.data_path)
    converter.store_metadata()

    with MetaDataStore(data_dir=args.data_path) as s:
        metadata = MetaData(s)

    # get relevant sensorids
    sensors = metadata.sensor_merged()
    indices = pd.Series([False] * sensors.shape[0], index=sensors.index.copy())
    indices = indices | sensors.sensorid.isin(metadata.electric_sensors())
    indices = indices & sensors.homeid.astype(int).isin(metadata.gold_homes())
    sensorids = sensors.sensorid[indices]
    sensorids_to_store = sensorids

    print('Query and store readings from {0} sensors'.format(
        len(sensorids_to_store)))

    for idx, sensorid in enumerate(sensorids_to_store):
            converter = IdealCSV2Hdf5(args.dataset_path, data_dir=args.data_path)

            logger.info('({0}/{1}) Sensorid: {2}'.format(
                idx + 1, len(sensorids_to_store), sensorid))

            converter.store_readings(sensorid)

    # try and read stored data
    readings_store = ReadingDataStore(data_dir=args.data_path)
    readings_count = 0

    for idx, sensorid in enumerate(sensorids):
        readings = readings_store.get_sensor_readings(sensorid)
        readings_count += len(readings)

    logger.info('Total readings : {0}'.format(readings_count))


if __name__ == '__main__':
    main()
