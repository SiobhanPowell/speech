"""
SPEECh: Scalable Probabilistic Estimates of EV Charging
Code first published in October 2021.
Developed by Siobhan Powell (siobhan.powell@stanford.edu).

This script demonstrates how to extract individual sessions data from running the model.
"""

from speech import DataSetConfigurations
from speech import SPEECh
from speech import SPEEChGeneralConfiguration
from speech import Plotting
from speech import Scenarios
from speech import LoadProfile

import copy
import os
import pandas as pd
if not os.path.isdir('IndividualSessionsOutputData'):
    os.mkdir('IndividualSessionsOutputData')
    
# Set up the scenario
total_evs = 500
weekday_option = 'weekday'

data = DataSetConfigurations('Original16', ng=16)
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
scenario = Scenarios('BaseCase')
config.change_pg(new_weights = scenario.new_weights)
config.change_pg(new_weights = {7:0.2, 10:0.2}, dend=True) # If you are making your own weights, use "dend=True"


# Run a version of "config.run_all":
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()

# edited contents of config.run_all():

individual_session_parameters_all = {key:None for key in data.categories}
# Run through driver groups:
for g in range(data.ng): 
    print('Group '+str(g))
    model = LoadProfile(config, config.group_configs[g], weekday=weekday_option) # part of config.run_all()
    individual_session_parameters = model.calculate_load(return_individual_session_parameters=True) # extra flag returns the data we want
    for key in individual_session_parameters.keys():
        if individual_session_parameters_all[key] is not None:
            individual_session_parameters_all[key] = pd.concat((individual_session_parameters_all[key], individual_session_parameters[key]), axis=0)
        else:
            individual_session_parameters_all[key] = copy.deepcopy(individual_session_parameters[key])

# Save the results:
for key in individual_session_parameters_all.keys():
    if individual_session_parameters_all[key] is not None:
        individual_session_parameters_all[key].to_csv('IndividualSessionsOutputData/'+key+'.csv')
        
        
# Extract the individual load profiles, not just the session parameters
categories = ['Home', 'MUD', 'Work', 'Other Slow', 'Other Fast']
total_load_profiles = {}
for segment in categories:
    if individual_session_parameters_all[segment] is not None:
        individual_session_parameters_all[segment] = individual_session_parameters_all[segment].reset_index(drop=True)
        n_segment = len(individual_session_parameters_all[segment])

        individual_profiles = {}  # collect the individual load profiles
        for index_value in np.arange(0, n_segment):  # calculate for each session
            tmp1, tmp2 = model.end_times_and_load(individual_session_parameters_all[segment].loc[index_value, 'Start'].reshape(1,), individual_session_parameters_all[segment].loc[index_value, 'Energy'].reshape(1,), individual_session_parameters_all[segment].loc[index_value, 'Rate'].reshape(1,))
            individual_profiles[index_value] = tmp2
        individual_profiles = pd.DataFrame(individual_profiles)
        individual_profiles.to_csv('IndividualSessionsOutputData/'+segment+'_individual_load_profiles.csv', index=None)  # save the individual load profiles in the same folder
        total_load_profiles[segment] = individual_profiles.sum(axis=1)

total_load_profiles = pd.DataFrame(total_load_profiles)
total_load_profiles['Total'] = total_load_profiles.loc[:, categories].sum(axis=1)  # calculate the total load profile as the sum of all sessions in all segments
total_load_profiles.to_csv('IndividualSessionsOutputData/total_load_profiles.csv', index=None)  # save the total load profile
