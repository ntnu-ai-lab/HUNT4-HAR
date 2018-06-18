import pandas as pd
import os, re
import numpy as np
import hashlib

filename = "../output/validparticipants.csv"
root_dir = os.path.dirname(os.path.abspath(__file__))
subject_file = os.path.join(root_dir, filename)

validparticipants = pd.read_csv(subject_file)

validparticipants['H4ID'] = validparticipants['H4ID'].apply(lambda x: hashlib.md5(np.int64(x)).hexdigest())

validparticipants.to_csv("../output/validparticipants-hashedid.csv", index=False)