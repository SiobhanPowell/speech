num_clusters = 16  # *fill in selected number of clusters*
data_folder = 'Folder/'  # *fill with correction location*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
from sklearn.cluster import AgglomerativeClustering

year = '2019'
driver_subset = pd.read_csv(folder+'sessions'+year+'_driverdata.csv', index_col=0)

# Process data:
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

initial_num = driver_subset.shape[0]
driver_subset = driver_subset.dropna(axis=0, subset=main_cols).reset_index(drop=True)
if driver_subset.shape[0] != initial_num:
    print('Dropped ', initial_num-driver_subset.shape[0], 'drivers in cleaning nan values.')

    
def normalize_df(df, cols_keep):
    
    scaling_df = {'Col': [], 'Shift': [], 'Denom': []}  # store information about the scaling in case its needed
    df_here = df.loc[:, cols_keep]
    for col in cols_keep:
        scaling_df['Col'].append(col)
        scaling_df['Shift'].append(df_here[col].min())
        scaling_df['Denom'].append(df_here[col].max()-df_here[col].min())
        df_here[col] = (df_here[col]-df_here[col].min())/(df_here[col].max()-df_here[col].min())
    scaling_df = pd.DataFrame(scaling_df)

    return df_here, scaling_df


# Cluster:
X_df, scaling_df = normalize_df(driver_subset, main_cols)
ac = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward').fit_predict(X_df.values)
X_df['Agglom Cluster Number'] = ac
X_df.to_csv(data_folder+'sessions'+year+'_driverdata_scaled_withlabels.csv')
driver_subset['Agglom Cluster Number'] = ac
driver_subset.to_csv(data_folder+'sessions'+year+'_driverdata_unscaled_withlabels.csv')
