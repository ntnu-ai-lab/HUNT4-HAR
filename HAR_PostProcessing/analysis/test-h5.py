import h5py
import pandas as pd

filename = '/Volumes/LaCie/dataset/500_timestamped_predictions.hdf'
hf = h5py.File(filename, 'r')

# List all groups
#print(list(hf))

pd_file = pd.HDFStore( filename )

with h5py.File(filename) as f:
    print(f.keys())

hf.close()