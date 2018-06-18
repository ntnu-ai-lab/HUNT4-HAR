import numpy as np
import pandas as pd
import glob

folders = glob.glob("/PATH/TO/DATA/*")
all_files = []
all_files2 = []
for folder in folders:
    all_files.append(glob.glob(folder+"/*"))

for folders in all_files:
    for i in range(len(folders)):
        all_files2.append(folders[i])

for file in all_files2:
    if "GoPro" in file:
        labels = pd.read_csv(file)
        if np.max(labels.values) > 19:
            print(file)
            print(np.max(labels.values))




