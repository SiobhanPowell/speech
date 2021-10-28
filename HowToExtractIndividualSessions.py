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
config.change_pg(new_weights = {7:0.2, 10:0.2})#scenario.new_weights)


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
