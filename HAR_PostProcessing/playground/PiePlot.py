
import pandas as pd
import os, re
import matplotlib.pyplot as plt

filename = "../output/1014_summary.csv"
root_dir = os.path.dirname(os.path.abspath(__file__))
subject_file = os.path.join(root_dir, filename)

subjectid = map(int,re.findall('\d+', filename)).pop().__str__()
data = pd.read_csv(subject_file)
data = data.drop('H4ID', axis=1)

# convert seconds to minutes
data = data.set_index('date')
data = data.divide(60).transpose()
data['Total'] = data.sum(axis=1)

explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

colors=['darkgreen', 'm', 'darkmagenta', 'orchid', 'orange', 'gold', 'lightgrey', 'c', 'green', 'forestgreen']
data['Total'].transpose().plot(kind='pie', figsize=(9, 7), colors=colors, explode=explode, title='Subject #' + subjectid)
plt.axis('equal')
#plt.legend(loc=9,ncol=6)
plt.show()