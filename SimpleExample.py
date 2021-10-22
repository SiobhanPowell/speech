"""
SPEECh: Scalable Probabilistic Estimates of EV Charging
Code first published in October 2021.
Developed by Siobhan Powell (siobhan.powell@stanford.edu).

This script demonstrates running the model, simple changes to behavioural component weights, and the use of built-in plotting options.
Make sure to close each figure after it opens to continue the script.
"""

from speech import DataSetConfigurations
from speech import SPEECh
from speech import SPEEChGeneralConfiguration
from speech import Plotting


total_evs = 1000
weekday_option = 'weekday'

# data = DataSetConfigurations('NewData', ng=9)
data = DataSetConfigurations('Original16', ng=16)
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
plots = Plotting(model, n=total_evs)  # plots total_evs
plots.total(weekday='weekday', save_str='simple_example_plot.png')
plots.pg()
plots.sessions_components(g=1, cat='Work', weekday='weekday')
plots.groups(save_string='simple_example_groups.png', n=total_evs) # plots total_evs in each group

# Demonstration of changing group weights and behaviour weights:
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
config.change_pg(new_weights={0: 0.5, 1: 0.5})  # Adjust distribution over driver groups so that 0 and 1 have each 50%
# (^ down-weights the others to 0 accordingly)
config.num_evs(total_evs)  # Input number of EVs in simulation
config.groups()
# Give weights 40% and 60% to behaviors 0 and 1 in the group 1 workplace segment:
# (down-weights the others to 0 accordingly)
config.change_ps_zg(1, 'Work', 'weekday', {0: 0.4, 1: 0.6})
config.run_all(weekday=weekday_option)
plots = Plotting(model, config)
plots.total(save_str='simple_example_plot_adjusted.png')
