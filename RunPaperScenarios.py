import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pomegranate

from speech import DataSetConfigurations
from speech import SPEECh
from speech import SPEEChGeneralConfiguration


def plot_together(data, ax, set_ymax, yticks, fonts=14, yax=True, xax=True, legendloc='upper right', nolegend=False):

    colours = ['#dfc27d', '#f6e8c3', '#80cdc1',  '#01665e', '#003c30']
    labels = ['Residential L2', 'Multi-Unit Dwelling L2', 'Workplace L2', 'Public L2', 'Public DCFC']

    xplot = (1/60)*np.arange(0, 1440)
    data = np.copy(data / 1000)  # MW
    base = np.zeros((1440, ))
    for i in range(np.shape(data)[1]):
        ax.plot(xplot, base+data[:, i], color=colours[i])
        ax.fill_between(xplot, base, base+data[:, i], color=colours[i], label=labels[i])
        base += data[:, i]
    ax.plot(xplot, base, 'k')

    ax.set_xlim([0, 24])
    ax.set_ylim([0, set_ymax])
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    if xax:
        ax.set_xlabel('Time of day [h]', fontsize=fonts+2)
        ax.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21], fontsize=fonts)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(yticks)
    if yax:
        ax.set_yticklabels(yticks.astype(int), fontsize=fonts)
        ax.set_ylabel('Load [MW]', fontsize=fonts+2)
    else:
        ax.set_yticklabels([])
    if not nolegend:
        ax.legend(loc=legendloc, ncol=2, fontsize=fonts-1)
    return ax


total_evs = 8e6
weekday_option = 'weekday'

# Prepare weights for scenarios
data = DataSetConfigurations('Original16')
original_pg = pd.read_csv('Data/Original16/pg.csv')

counts_df = pd.DataFrame({'Original Weight': original_pg['pg'].values, 'AC Cluster Number': original_pg.index.values})
counts_df['Dend Cluster Number'] = 0
for i, j in data.cluster_reorder_dendtoac.items():
    counts_df.loc[counts_df[counts_df['AC Cluster Number'] == j].index, 'Dend Cluster Number'] = i
counts_df = counts_df.sort_values(by='Dend Cluster Number').reset_index(drop=True)

## Scenario 123
rescales = {(0.67*0.9): [3, 4, 5], (0.67*0.1): [9], 0.33: [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15]}
counts_df['Scen1'] = counts_df['Original Weight'].copy()
for key, val in rescales.items():
    counts_df.loc[val, 'Scen1'] = counts_df.loc[val, 'Scen1'] * (key / sum(counts_df.loc[val, 'Scen1']))
## Scenario 4
rescales = {0.6: [3, 4, 5], 0.3: [9], 0.1: [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15]}
counts_df['Scen4'] = counts_df['Original Weight'].copy()
for key, val in rescales.items():
    counts_df.loc[val, 'Scen4'] = counts_df.loc[val, 'Scen4'] * (key / sum(counts_df.loc[val, 'Scen4']) )
## Scenario 5
rescales = {(0.5*0.9): [3, 4, 5], (0.5*0.1): [9], 0.5: [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15]}
counts_df['Scen5'] = counts_df['Original Weight'].copy()
for key, val in rescales.items():
    counts_df.loc[val, 'Scen5'] = counts_df.loc[val, 'Scen5'] * (key / sum(counts_df.loc[val, 'Scen5']))
## Scenario 6
counts_df['Scen6'] = counts_df['Scen5'].copy()
val = sum(counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([6, 7, 8, 12, 13, 14, 15])].index, 'Scen6'])
counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([12, 13, 14, 15])].index, 'Scen6'] = 0
counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([6, 7, 8])].index, 'Scen6'] = (val / sum(counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([6, 7, 8])].index, 'Scen6'])) * counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([6, 7, 8])].index, 'Scen6']
val = sum(counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([0, 1, 2])].index, 'Scen6'])
counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([0, 1])].index, 'Scen6'] = 0
counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([2])].index, 'Scen6'] = (val / sum(counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([2])].index, 'Scen6'])) * counts_df.loc[counts_df[counts_df['Dend Cluster Number'].isin([2])].index, 'Scen6']


goal_weight = 0.67  # What to increase timers to
key = 'Data/Original16/GMMs/weekday_home_'+str(data.cluster_reorder_dendtoac[3])+'.p'
joint_gmm = pickle.load(open(key, "rb"))
# Augmenting timers for cluster 3:
base_weights3 = {}
base_weights3[0] = goal_weight
# Removing timers for cluster 3:
new_weights3 = {}
new_weights3[0] = 0
new_weights3[2] = joint_gmm.weights_[2] + joint_gmm.weights_[0]
key = 'Data/Original16/GMMs/weekday_home_'+str(data.cluster_reorder_dendtoac[4])+'.p'
joint_gmm = pickle.load(open(key, "rb"))
# Augmenting timers for cluster 4:
base_weights4 = {}
base_weights4[4] = goal_weight*(joint_gmm.weights_[4] / (joint_gmm.weights_[4] + joint_gmm.weights_[6]))
base_weights4[6] = goal_weight*(joint_gmm.weights_[6] / (joint_gmm.weights_[4] + joint_gmm.weights_[6]))
# Removing timers for cluster 4:
new_weights4 = {}
new_weights4[4] = 0
new_weights4[6] = 0
w1 = joint_gmm.weights_[0]
w2 = joint_gmm.weights_[5]
new_weights4[0] = joint_gmm.weights_[0] + (w1 / (w1+w2))*(joint_gmm.weights_[4] + joint_gmm.weights_[6])
new_weights4[5] = joint_gmm.weights_[5] + (w2 / (w1+w2))*(joint_gmm.weights_[4] + joint_gmm.weights_[6])
key = 'Data/Original16/GMMs/weekday_home_'+str(data.cluster_reorder_dendtoac[5])+'.p'
joint_gmm = pickle.load(open(key, "rb"))
# Augmenting timers for cluster 5:
base_weights5 = {}
base_weights5[0] = goal_weight*(joint_gmm.weights_[0] / (joint_gmm.weights_[0] + joint_gmm.weights_[1]))
base_weights5[1] = goal_weight*(joint_gmm.weights_[1] / (joint_gmm.weights_[0] + joint_gmm.weights_[1]))
# Removing timers for cluster 5:
new_weights5 = {}
new_weights5[0] = 0
new_weights5[1] = 0
w1 = joint_gmm.weights_[7]
w2 = joint_gmm.weights_[4]
new_weights5[7] = joint_gmm.weights_[7] + (w1 / (w1+w2))*(joint_gmm.weights_[0] + joint_gmm.weights_[1])
new_weights5[4] = joint_gmm.weights_[4] + (w2 / (w1+w2))*(joint_gmm.weights_[0] + joint_gmm.weights_[1])

# Adjusting workplace behaviors:
key = 'Data/Original16/GMMs/weekday_work_'+str(data.cluster_reorder_dendtoac[3])+'.p'
joint_gmm = pickle.load(open(key, "rb"))
new_weights3_work = {}
new_weights3_work[1] = 0.5 * (joint_gmm.weights_[1] / (joint_gmm.weights_[1]+joint_gmm.weights_[3]))
new_weights3_work[3] = 0.5 * (joint_gmm.weights_[3] / (joint_gmm.weights_[1]+joint_gmm.weights_[3]))
key = 'Data/Original16/GMMs/weekday_work_'+str(data.cluster_reorder_dendtoac[5])+'.p'
joint_gmm = pickle.load(open(key, "rb"))
new_weights5_work = {}
new_weights5_work[3] = 0.5 * (joint_gmm.weights_[3] / (joint_gmm.weights_[3]+joint_gmm.weights_[5]))
new_weights5_work[5] = 0.5 * (joint_gmm.weights_[5] / (joint_gmm.weights_[3]+joint_gmm.weights_[5]))

# Run scenarios
data = DataSetConfigurations('Original16')

# Set 1
fig, axes = plt.subplots(1, 3, figsize=(24, 5.5))

# Scenario 1
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen1']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', base_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', base_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', base_weights5)
config.run_all(weekday=weekday_option)
print('Ran 1')
axes[0] = plot_together(config.total_load_segments, axes[0], fonts=20, yax=True, set_ymax=8200, yticks=np.arange(0, 8100, 1000), legendloc='upper left')
axes[0].set_yticks(np.arange(0, 8100, 1000))
axes[0].set_yticklabels(np.arange(0, 8100, 1000).astype(int), fontsize=20)

# Scenario 2
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen1']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', new_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', new_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', new_weights5)
config.run_all(weekday=weekday_option)
print('Ran 2')
axes[1] = plot_together(config.total_load_segments, axes[1], fonts=20, yax=False, set_ymax=8200, yticks=np.arange(0, 8100, 1000), nolegend=True)

# Scenario 3
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen1']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', new_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', new_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', new_weights5)
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Work', 'weekday', new_weights3_work)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Work', 'weekday', new_weights5_work)
config.run_all(weekday=weekday_option)
print('Ran 3')
axes[2] = plot_together(config.total_load_segments, axes[2], fonts=20, yax=False, set_ymax=8200, yticks=np.arange(0, 8100, 1000), nolegend=True)

plt.tight_layout()
plt.savefig('scenarios123.png', bbox_inches='tight')
plt.close()


# Set 2
fig, axes = plt.subplots(1, 3, figsize=(24, 5.5))

# Scenario 4
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen4']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', base_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', base_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', base_weights5)
config.run_all(weekday=weekday_option)
print('Ran 4')
axes[0] = plot_together(config.total_load_segments, axes[0], fonts=20, yax=True, set_ymax=8800, yticks=np.arange(0, 8100, 1000), legendloc='upper left')
axes[0].set_yticks(np.arange(0, 8100, 1000))
axes[0].set_yticklabels(np.arange(0, 8100, 1000).astype(int), fontsize=20)

# Scenario 5
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen5']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', base_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', base_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', base_weights5)
config.run_all(weekday=weekday_option)
print('Ran 5')
axes[1] = plot_together(config.total_load_segments, axes[1], fonts=20, yax=False, set_ymax=8800, yticks=np.arange(0, 8100, 1000), nolegend=True)

# Scenario 6
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
new_weights_pg = dict(zip(counts_df['AC Cluster Number'], counts_df['Scen6']))
config.change_pg(new_weights=new_weights_pg)  # Adjust distribution over driver groups
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
config.change_ps_zg(data.cluster_reorder_dendtoac[3], 'Home', 'weekday', base_weights3)
config.change_ps_zg(data.cluster_reorder_dendtoac[4], 'Home', 'weekday', base_weights4)
config.change_ps_zg(data.cluster_reorder_dendtoac[5], 'Home', 'weekday', base_weights5)
config.run_all(weekday=weekday_option)
print('Ran 6')
axes[2] = plot_together(config.total_load_segments, axes[2], fonts=20, yax=False, set_ymax=8800, yticks=np.arange(0, 8100, 1000), nolegend=True)

plt.tight_layout()
plt.savefig('scenarios456.png', bbox_inches='tight')
plt.close()
