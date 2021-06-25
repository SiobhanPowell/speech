data_folder = 'Folder/'  # * fill in with location *
year = '2019'
subfolder = 'NewData/'
import os
import shutil
# if not os.path.isdir('../Data/NewData'):
#     os.mkdir('../Data/NewData')
num_clusters = 9

'''Process the data post clustering to be used by speech.'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load data
driver_subset = pd.read_csv(data_folder+'sessions'+year+'_driverdata_unscaled_withlabels.csv', index_col=0)

# P(G)
pg = pd.DataFrame(dict(driver_subset['Agglom Cluster Number'].value_counts(normalize=True)), index=['pg']).T.sort_index()
pg.to_csv(subfolder+'pg.csv', index=None)

# P(z|G)
relabel = {'Home': 'home_l2', 'Work': 'work_l2', 'MUD': 'mud_l2', 'Other Slow': 'public_l2', 'Other Fast': 'public_l3'}
col_mapping = {}
for key, val in relabel.items():
    for w in ['weekdays', 'weekenddays']:
        col_mapping[key+' - Fraction of '+w+' with session'] = key+' - Fraction of '+w+' with session'
        # col_mapping[key+' - Fraction of '+w+' with session'] = val+' - Fraction of '+w+' with session'
for i in range(num_clusters):
    inds = driver_subset[driver_subset['Agglom Cluster Number'] == i].index
    pz_subset = driver_subset.loc[inds, col_mapping.keys()].reset_index(drop=True)
    pz_subset = pz_subset.rename(columns=col_mapping)
    pz_subset['home_l1 - Fraction of weekdays with session'] = 0
    pz_subset['home_l1 - Fraction of weekenddays with session'] = 0
    pz_subset.to_csv(subfolder+'pz_weekday_g_'+str(i)+'.csv')
    pz_subset.to_csv(subfolder+'pz_weekend_g_'+str(i)+'.csv')

# P(s | z, G) are the GMMs already done.

shutil.copytree('NewData', '../Data/NewData')
# Final to-do:
# Copy the following as a method into the class `DataSetConfigurations` in `speech.py`


def new_data(self):
    """New Clustering. Also note new ng = """

    self.categories = ['Home', 'MUD', 'Work', 'Other Slow', 'Other Fast']
    self.labels = ['Residential L2', 'MUD L2', 'Workplace L2', 'Public L2', 'Public DCFC']
    self.colours = {'Residential L2': '#dfc27d', 'MUD L2': '#f6e8c3', 'Workplace L2': '#80cdc1', 'Public L2': '#01665e', 'Public DCFC': '#003c30'}
    self.num_categories = 5
    self.rates = [6.6, 6.6, 6.6, 6.6, 50]
    self.gmm_names = {'Home': 'home', 'Work': 'work', 'Other Slow': 'other_slow', 'MUD': 'mud', 'Other Fast': 'other_fast'}
    self.start_time_scaler = 1/60
    self.zkey_weekday = ' - Fraction of weekdays with session'
    self.zkey_weekend = ' - Fraction of weekenddays with session'
    self.start_mod = 24*3600  # since start time is in seconds
    self.timers_dict = {}
    # Optional: record shifts for removing timers from fitted model
    # self.timers_dict = {group number with timers in it:
    # {gmm segment number with timers in it: 0,
    # other gmm segment number to switch to: current weight + weight from timer segment}}
    self.shift_timers_dict = {}
    # Similarly: self.shift_timers_dict = {'Components': {group number with timers: [list of gmm segments with timers],
    # another group number with timers: [list of gmm segments with timers],},
    # 'Targets': {'PGE': 23, 'SMUD': 0, 'SCE': 21, 'SDGE': 0}}
    self.timer_cat = 'Home'

