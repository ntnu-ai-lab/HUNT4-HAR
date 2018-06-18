import glob
import re

from acrechain import complete_end_to_end_prediction
from joblib import Parallel, delayed
import time

from HUNT4ParticipantPlot import create_barplot, create_summary

def create_predictions(filename, path, freq=100):
    print('filename starting: ', filename)
    start = time.time()
    subjectid = list(map(int, re.findall('\d+', filename))).pop().__str__()
    input = path + subjectid + "/" + subjectid
    backfile =''
    thighfile = ''

    for name in glob.glob(input + '*_B.cwa'):
        backfile = name

    for name in glob.glob(input + '*_T.cwa'):
        thighfile = name

    output = path + "tmp/" + subjectid + "_timestamped_predictions.csv"

    complete_end_to_end_prediction(backfile, thighfile, output, sampling_frequency=freq)
    # output >> sent to make summaries & pretty
    print('backfile: ', backfile)
    startrecording = re.search('%s(.*)%s' % ('_', '_'), backfile).group(1)
    print('startrecording', startrecording)
    summary_data_filename = create_summary(output, subjectid, path, startrecording)
    create_barplot(summary_data_filename, subjectid, path)

path = "huntdata/"
subjectpath = path + "9*"

for fname in glob.glob(subjectpath):
    create_predictions(fname, path, 100)

# parallel on 20 cores:
#Parallel(n_jobs=1)(delayed(create_predictions)(fname) for fname in glob.glob(path))