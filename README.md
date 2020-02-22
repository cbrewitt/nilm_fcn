# Non-instrusive Load Monitoring with Fully Convolution networks

Code for the method in [this paper](https://arxiv.org/abs/1812.03915).

## Running the code

Python 3 is used. First set the environment variable IDEAL_DATA_DIR to the directory where you wish to store intermediate data, models, and predictions.

Next, run the script `csv_to_hdf5_converter.py`, with an argument giving the path to the IDEAL dataset. This will parse the data from the CSV files are store in HDF5 format. 

`python csv_to_hdf5_converter.py --dataset_path <path to dataset>`

Next, preprocess the data, filling short gaps and merging sensors:

`python generate_cleaned_nilm_data.py`

Generate windows of data to use for training and testing:

`python generate_s2s_dataset.py`

Train and predict with the Fully Convolutional Network:

`python fully_conv_separate_valid.py`

Train and predict with the Sequence-to-Point baseline:

`python pointnet_large_separate_valid.py`

Calculate results with the Jupyter Notebook 'fcn-evaluation.ipynb`






