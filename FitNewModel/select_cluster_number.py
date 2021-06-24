data_folder = 'Folder/'  # *fill with correction location*

"""Creates elbow plot to select optimal number of driver clusters. 

This script will return a plot, `elbow_plot_of_driver_clustering.png`. 
You should use that, looking for a kink or `elbow`, to select the best number of clusters to use.

The next step is the script `cluster_drivers.py`."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

year = '2019'
driver_subset = pd.read_csv(data_folder+'sessions'+year+'_driverdata.csv', index_col=0)

main_cols = ['Num Zip Codes', 'Battery Capacity', 'Num Workplace Sessions',
             'Num Single Family Residential Sessions', 'Num MUD Sessions',
             'Num Other Slow Sessions', 'Num Other Fast Sessions',
             'Work - Session energy - mean', 'Work - Session time - mean',
             'Work - Start hour - mean', 'Work - Weekend fraction',
             'Other Fast - Session energy - mean', 'Other Fast - Session time - mean',
             'Other Fast - Start hour - mean', 'Other Fast - Weekend fraction',
             'Other Slow - Session energy - mean', 'Other Slow - Session time - mean',
             'Other Slow - Start hour - mean', 'Other Slow - Weekend fraction',
             'MUD - Session energy - mean', 'MUD - Session time - mean',
             'MUD - Start hour - mean', 'MUD - Weekend fraction',
             'Home - Session energy - mean', 'Home - Session time - mean',
             'Home - Start hour - mean', 'Home - Weekend fraction']

# quick cleaning:
initial_num = driver_subset.shape[0]
driver_subset = driver_subset.dropna(axis=0, subset=main_cols)
if driver_subset.shape[0] != initial_num:
    print('Dropped ', initial_num-driver_subset.shape[0], 'drivers in cleaning nan values.')


# normalize vector:
def normalize_df(df, cols_keep):
    scaling_df = {'Col':[], 'Shift':[], 'Denom':[]}
    df_here = df.loc[:, cols_keep]
    for col in cols_keep:
        scaling_df['Col'].append(col)
        scaling_df['Shift'].append(df_here[col].min())
        scaling_df['Denom'].append(df_here[col].max()-df_here[col].min())
        if (df_here[col].max()-df_here[col].min()) > 0:
            df_here[col] = (df_here[col]-df_here[col].min())/(df_here[col].max()-df_here[col].min())
    scaling_df = pd.DataFrame(scaling_df)
    return df_here, scaling_df


X_df, scaling_df = normalize_df(driver_subset, main_cols)

linkage = shc.linkage(X_df, method='ward')
np.save('linkage_matrix_'+str(year)+'.npy', linkage)

# Plot some sample dendrograms
fig, axes = plt.subplots(1, 1, figsize=(12,5))
dend = shc.dendrogram(linkage, truncate_mode='lastp', p=16, show_leaf_counts=True, ax=axes)
plt.tight_layout()
plt.savefig('dend_'+str(16)+'.png', bbox_inches='tight')
plt.show()
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(12,5))
dend = shc.dendrogram(linkage, truncate_mode='lastp', p=32, show_leaf_counts=True, ax=axes)
plt.tight_layout()
plt.savefig('dend_'+str(32)+'.png', bbox_inches='tight')
plt.show()
plt.close()

heights1 = []
heights2 = []
oldmin = np.max(np.max(dend['dcoord'], axis=1))
nks = np.arange(3, 100)
for nk in nks:
    dend = shc.dendrogram(linkage, truncate_mode='lastp', p=int(nk), no_plot=True)
    heights1.append(oldmin - np.min(np.max(dend['dcoord'], axis=1)))
    heights2.append(np.max(np.max(dend['dcoord'], axis=1)) - np.min(np.max(dend['dcoord'], axis=1)))
    oldmin = np.min(np.max(dend['dcoord'], axis=1))
    
plt.figure()
plt.plot(nks, np.max(heights2)-heights2)
plt.xlabel('Num clusters')
plt.ylabel('Distance')
plt.title('Elbow Plot')
plt.tight_layout()
plt.savefig('elbow_plot_of_driver_clustering.png', bbox_inches='tight')
plt.show()


