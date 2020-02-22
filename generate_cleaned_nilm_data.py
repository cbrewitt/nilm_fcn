"""
Created on 21/02/18
@author Cillian Brewitt

This script creates a cleaned dataset for electricity dissagration
store_gold_elec_data_locally.py should be run first
"""
import argparse
import datetime as dt

from metadata import MetaData
from data_store import MetaDataStore, HomeReadingStore
from data_preprocessing import ElecPreprocessor

def valid_date(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

# get arguments
parser = argparse.ArgumentParser(description='Clean electrical sensor readings and merge .')
parser.add_argument('--home',type=int, default=-1, help='home to process, default all')
parser.add_argument('--enddate', type=valid_date, default=None, help='use only readings before this date')
args = parser.parse_args()

end_date = args.enddate

with MetaDataStore() as s:
    metadata = MetaData(s)
    
home_reading_store = HomeReadingStore()
preprocessor = ElecPreprocessor()

if args.home == -1:
    homeids = metadata.gold_homes()
else:
    homeids = [args.home]

for homeid in homeids:
    print('homeid: {0}'.format(homeid))
    readings = preprocessor.get_home_readings(homeid)
    if end_date is not None:
        readings = readings[readings.index < end_date]
    home_reading_store.set_readings(homeid, readings)
