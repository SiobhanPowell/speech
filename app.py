"""
SPEECh: Scalable Probabilistic Estimates of EV Charging
Code first published in October 2021.
Developed by Lesley Ryan and Siobhan Powell.

This file is used to run the back-end of the interactive tool and website.
"""
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
from speech import SPEECh, SPEEChGeneralConfiguration, LoadProfile, Scenarios, DataSetConfigurations
import copy

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():

    data = DataSetConfigurations('Original16')
    speech = SPEECh(data)  # Begin the model
    speech.pg_()  # Load the initial distribution over driver groups
    
    if request.method == "GET":  # Called when first viewing the page, default values
        # set default values    
        total_evs = int(100000)
        weekday_option = "weekday"
        scenario_option = 'BaseCase'
        remove_timers = 'False'

    if request.method == "POST":  # Called when user presses 'Submit'
        total_evs = int(request.form['no_evs'])
        weekday_option = request.form['weekday']
        scenario_option = request.form['scenario']
        remove_timers = request.form['remove_timers']

    if remove_timers == 'False':
        config = SPEEChGeneralConfiguration(speech)  # Load the model configuration
    else:
        config = SPEEChGeneralConfiguration(speech, remove_timers=True)  # Load the model configuration, removing timers

    if scenario_option == 'Custom':  # Use user custom weights for driver groups
        new_weights = {}
        for i in np.arange(1, 17):
            weight_gx = request.form['weight_g'+str(i)]
            if weight_gx != '':
                new_weights[speech.data.cluster_reorder_dendtoac[i-1]] = float(weight_gx)
        total_weights = sum(new_weights.values())
        if total_weights > 1:
            new_weights2 = {key: val/total_weights for key, val in new_weights.items()}
            new_weights = copy.deepcopy(new_weights2)
    
    elif scenario_option == 'HighMUD':  # Use built-in scenario for high level of multi-unit dwelling charging
        scenario = Scenarios(scenario_name='HighMUD')
        new_weights = scenario.new_weights
    elif scenario_option == 'Workplace':  # Use built-in scenario for high level of workplace charging
        scenario = Scenarios(scenario_name='Workplace')
        new_weights = scenario.new_weights
    elif scenario_option == 'WorkplaceLargeBattery':  # Use built-in scenario: 'Workplace' with large battery vehicles
        scenario = Scenarios(scenario_name='WorkplaceLargeBattery')
        new_weights = scenario.new_weights
    elif scenario_option == 'BaseCase':  # Use built-in base case scenario
        scenario = Scenarios(scenario_name='BaseCase')
        new_weights = scenario.new_weights
    else:
        scenario = Scenarios(scenario_name='OriginalData')  # Use original distribution from the data set
        new_weights = scenario.new_weights

    config.change_pg(new_weights=new_weights)  # Adjust distribution over driver groups
    config.num_evs(total_evs)  # Input number of EVs in simulation
    config.groups()  # Configure driver group models

    if request.method == "POST":
        # second level of inputs allows the user to adjust the sessions model component distributions:
        num_inputs = pd.read_csv('Data/Original16/plot_num_inputs_weekday.csv', index_col=0)  # Each has a different number
        for cluster_number in np.arange(1, 17):
            for segment_number in np.arange(1, 6):
                n = num_inputs.loc[segment_number, str(cluster_number)]
                if n == n:  # num_inputs has NaN value where segment is not large for the given driver group
                    new_weights2 = {}
                    changed = False
                    for i in np.arange(1, n+1):
                        key = 'cluster'+str(cluster_number)+'_segment'+str(segment_number)+'_input'+str(int(i))
                        value = request.form[key]
                        if value != '':  # value was changed by the user
                            new_weights2[int(i-1)] = float(value)
                            changed = True
                    if changed:  # the user changed at least one value, so we update the model
                        print('Changing cluster '+str(cluster_number)+' segment '+str(segment_number))
                        segment_names = {1: 'Home', 2: 'MUD', 3: 'Work', 4: 'Other Slow', 5: 'Other Fast'}
                        config.change_ps_zg(speech.data.cluster_reorder_dendtoac[cluster_number-1],
                                            segment_names[segment_number], 'weekday', new_weights2)

    config.run_all(weekday=weekday_option)  # run all results

    output = config.total_load_segments  # manipulate into format for display and graphing
    x = (24 / np.shape(output)[0]) * np.arange(0, np.shape(output)[0])
    scaling = 1/1000  # Convert to MW
    
    y1 = np.zeros(np.shape(output[:, 0]))  # Res L1 is zero in data set 'Original16'
    y2 = output[:, 0]
    y3 = output[:, 1]
    y4 = output[:, 2]
    y5 = output[:, 3]
    y6 = output[:, 4]
    
    all_lines = [{'x': x[i], 'line1': scaling*y1[i], 'line2': scaling*y2[i], 'line3': scaling*y3[i],
                  'line4': scaling*y4[i], 'line5': scaling*y5[i], 'line6': scaling*y6[i]} for i in range(len(x))]

    return render_template('main.html', input={'no_evs': total_evs, 'weekday_option': weekday_option,
                                               'scenario': scenario_option, 'remove_timers': remove_timers},
                           output=all_lines,)


@app.route("/about")
def about():

    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
