
import pandas as pd
import os, re
import matplotlib.pyplot as plt
import seaborn as sns

#filename = "../output/validparticipants-hashedid.csv"
filename = "../output/h4-sum-minutes.csv"
root_dir = os.path.dirname(os.path.abspath(__file__))
subject_file = os.path.join(root_dir, filename)

data = pd.read_csv(subject_file)

# find users that didn't wear the sensors and remove them from the data set
extract = data
removeids = extract[(extract.sitting == 0) & (extract.standing <= 0)]
removeids = removeids.H4ID
removeids = removeids.drop_duplicates()

removeids.to_csv("../output/notwornIDs.csv", index=False)

validparticipants = extract[~extract.H4ID.isin(removeids)]

# store data
validparticipants.to_csv("../output/validparticipants.csv", index=False)

validparticipants = data

#print(validparticipants.head())
#print(validparticipants.groupby('weekday'))
#print(validparticipants.groupby('weekday').agg(['min', 'max']))

##################################################################
# get the data
ax = validparticipants[['lying','sitting','standing','walking','running','cycling']].boxplot()

# Set the x-axis label
ax.set_xlabel("Activities")

# Set the y-axis label
ax.set_ylabel("Minutes")
fig = ax.get_figure()

ax.set_title('Activity times (' + validparticipants.H4ID.drop_duplicates().__len__().__str__() + ' participants)', fontsize=12, fontweight='bold')

fig.savefig('plots/boxplot.png')
##################################################################

##################################################################
# get the data
walkingbyday = validparticipants[['date','walking']].sort_values('date')
ax = walkingbyday.groupby(walkingbyday.date).mean().plot()

# Set the y-axis label
ax.set_ylabel("Minutes (mean)")
fig = ax.get_figure()
fig.set_size_inches(18.5, 10.5)
ax.set_title('Walking times (' + validparticipants.H4ID.drop_duplicates().__len__().__str__() + ' participants)', fontsize=12, fontweight='bold')

fig.savefig('plots/walkinglineplot.png')
##################################################################

##################################################################
# get the data
activitybyday = validparticipants[['date','running','cycling']]

ax = activitybyday.groupby(activitybyday.date).mean().plot()

# Set the y-axis label
ax.set_ylabel("Minutes (mean)")
fig = ax.get_figure()
fig.set_size_inches(18.5, 10.5)
ax.set_title('Vigorous activity times (' + validparticipants.H4ID.drop_duplicates().__len__().__str__() + ' participants)', fontsize=12, fontweight='bold')

fig.savefig('plots/vigactlineplot.png')
##################################################################
sns.set_style("whitegrid")
activitylst = ['lying','sitting','standing','walking','running','cycling']

for group in activitylst:
    plt.figure(figsize=(75,12))

    # create our boxplot which is drawn on an Axes object
    bplot = sns.boxplot(x='date', y=group, data=validparticipants.sort_values('date'), whis=[5,95])

    title = ('Distribution of '+group+' times (' + validparticipants.H4ID.drop_duplicates().__len__().__str__() + ' participants)')

    # We can call all the methods avaiable to Axes objects
    bplot.set_title(title, fontsize=20)
    bplot.set_xlabel('date', fontsize=16)
    bplot.set_ylabel('minutes', fontsize=16)
    bplot.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=90)
    sns.despine(left=True)

    plt.savefig('plots/bp-'+group+'.png')


##################################################################
sns.set_style("whitegrid")
activitylst = ['lying','sitting','standing','walking','running','cycling']

print(validparticipants.info())
validparticipants['date'] = pd.to_datetime(validparticipants['date'])
print(validparticipants.info())
print(validparticipants.head())

for group in activitylst:
    plt.figure(figsize=(20,15))

    # create our boxplot which is drawn on an Axes object
    bplot = sns.boxplot(x='weekday', y=group, data=validparticipants, order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    title = ('Distribution of '+group+' times (' + validparticipants.H4ID.drop_duplicates().__len__().__str__() + ' participants)')

    # We can call all the methods avaiable to Axes objects
    bplot.set_title(title, fontsize=20)
    bplot.set_xlabel('Weekday', fontsize=16)
    bplot.set_ylabel('Minutes', fontsize=16)
    bplot.tick_params(axis='both', labelsize=12)
    #plt.xticks(rotation=90)
    sns.despine(left=True)

    plt.savefig('plots/bp-byweekday-'+group+'.png')
