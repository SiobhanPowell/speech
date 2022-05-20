"""
SPEECh: Scalable Probabilistic Estimates of EV Charging
Code first published in October 2021.
Developed by Siobhan Powell (siobhan.powell@stanford.edu).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import os


class DataSetConfigurations(object):
    """Store the information for each data set prepared.

    Attributes:
        ng: number of driver groups
        data_set: name of data set used
        folder: location of the data and model object files

        cluster_reorder_actodend: mapping from cluster labels in agglomerative clustering (ac) to dendrogram (dend)
        cluster_reorder_dendtoac: reverse mapping

        categories: charging segments
        labels: proper names for charging segments
        colours: plotting colours for charging segments
        num_categories: number of charging segments modeled
        rates: charging rate for a session in each segment, in kW

        gmm_names: part of file name for sessions model GMM object
        start_time_scaler: adjustment to start time produced by GMM model
        start_mod: limit when calculating modulo of generated start times

        zkey_weekday, zkey_weekend: naming convention within pz

        timers_dict: dictionary used to remove residential timers
            Includes one entry for each driver group which needs to change.
            For each driver group, a dictionary rebalancing weights among the sessions components.
        timer_cat: which charging segment contains the timers

    Methods:
        original16(): sets up data properties for original data set with 16 clusters. When other datasets are added,
        similar methods should be created for them.

        set_up_dendrogram_cluster_mapping(): where the labels assigned in clustering are ordered differently than how
        clusters are presented in the dendrogram, we reorder them so plotted orders match our expectation from the
        dendrogram.
    """

    def __init__(self, data_set, ng=None):

        self.categories = []
        self.labels = []
        self.colours = {}
        self.num_categories = 0
        self.rates = []
        self.zkey_weekday = ''
        self.zkey_weekend = ''

        self.cluster_reorder_dendtoac = {}
        self.cluster_reorder_actodend = {}

        self.gmm_names = {}
        self.start_time_scaler = 1
        self.start_mod = 1

        self.timers_dict = {}
        self.shift_timers_dict = {}
        self.timer_cat = ''

        if data_set == 'Original16':
            self.data_set = 'Original16'
            self.ng = 16
            self.original16()
        elif data_set == 'NewData':
            self.data_set = 'NewData'
            self.ng = ng
            self.new_data()
        else:
            self.ng = ng
            self.data_set = data_set
            raise Exception('Select set-up method for new data set.')

        self.set_up_dendrogram_cluster_mapping()
        self.folder = 'Data/'+self.data_set+'/'

    def original16(self):
        """Clustering used in first publication."""

        self.categories = ['Home', 'MUD', 'Work', 'Other Slow', 'Other Fast']
        self.labels = ['Residential L2', 'MUD L2', 'Workplace L2', 'Public L2', 'Public DCFC']
        self.colours = {'Residential L2': '#dfc27d', 'MUD L2': '#f6e8c3', 'Workplace L2': '#80cdc1', 'Public L2': '#01665e', 'Public DCFC': '#003c30'}
        self.num_categories = 5
        self.rates = [6.6, 6.6, 6.6, 6.6, 150]
        self.gmm_names = {'Home': 'home', 'Work': 'work', 'Other Slow': 'other_slow', 'MUD': 'mud', 'Other Fast': 'other_fast'}
        self.start_time_scaler = 1/60
        self.zkey_weekday = ' - Fraction of weekdays with session'
        self.zkey_weekend = ' - Fraction of weekenddays with session'
        self.start_mod = 24*3600
        self.timers_dict = {1: {0: 0, 2: 0.40824126 + 0.27438057}, 13: {4: 0, 6: 0, 0: 0.13107678 + (0.13 / (0.13 + 0.25))*(0.03736147 + 0.22645425), 5: 0.25021152 + (0.25 / (0.13 + 0.25))*(0.03736147 + 0.22645425)},
                            3: {0: 0, 1: 0, 4: 0.20294322 + (0.20 / (0.14 + 0.20))*(0.25518963 + 0.03967571), 7: 0.1398116 + (0.14 / (0.14 + 0.20))*(0.25518963 + 0.03967571)}}
        self.shift_timers_dict = {'Components': {1: [0], 13: [4, 6], 3: [0, 1]}, 'Targets': {'PGE': 23, 'SMUD': 0, 'SCE': 21, 'SDGE': 0}}
        self.timer_cat = 'Home'

    def new_data(self):
        """New Clustering. Also note new ng = 9 (replace with true value from implementation on new data."""

        self.categories = ['Home', 'MUD', 'Work', 'Other Slow', 'Other Fast']
        self.labels = ['Residential L2', 'MUD L2', 'Workplace L2', 'Public L2', 'Public DCFC']
        self.colours = {'Residential L2': '#dfc27d', 'MUD L2': '#f6e8c3', 'Workplace L2': '#80cdc1', 'Public L2': '#01665e', 'Public DCFC': '#003c30'}
        self.num_categories = 5
        self.rates = [6.6, 6.6, 6.6, 6.6, 150]
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



    def set_up_dendrogram_cluster_mapping(self):

        if self.data_set == 'Original16':
            self.cluster_reorder_actodend = {11: 0, 12: 1, 5: 2, 1: 3, 13: 4, 3: 5, 8: 6, 15: 7, 2: 8, 9: 9, 7: 10, 14: 11, 4: 12, 6: 13, 10: 14, 0: 15}
            self.cluster_reorder_dendtoac = {0: 11, 1: 12, 2: 5, 3: 1, 4: 13, 5: 3, 6: 8, 7: 15, 8: 2, 9: 9, 10: 7, 11: 14, 12: 4, 13: 6, 14: 10, 15: 0}
        else:
            self.cluster_reorder_dendtoac = {i: i for i in range(self.ng)}
            self.cluster_reorder_actodend = {i: i for i in range(self.ng)}


class SPEECh(object):
    """Top-level SPEECh class.

    Attributes:
        data: object from DataSetConfiguration class with set up of data set used.
        pz: segment probabilities,
            e.g. pz['weekday'][0]['Home'] gives the probability a driver in group 0 will charge at home on a weekday
        pg: driver group probabilities

    Methods:
        pg_(): loads the base distribution for pg
        pz_g(): loads the base values for pz
    """
    
    def __init__(self, data):

        self.data = data

        self.pz = {'weekday': {}, 'weekend': {}}
        self.pg_()
        self.pz_g()

    def pg_(self):
        """Loads the base distribution over driver groups."""
        self.pg = pd.read_csv(self.data.folder+'pg.csv')
        
    def pz_g(self):
        """Loads the segment probabilities."""
        for i in range(self.data.ng):
            self.pz['weekday'][i] = pd.read_csv(self.data.folder+'pz_weekday_g_'+str(i)+'.csv')
            # if self.weekend_exists:
            self.pz['weekend'][i] = pd.read_csv(self.data.folder+'pz_weekend_g_'+str(i)+'.csv')


class SPEEChGeneralConfiguration(object):
    """General configuration using the speech class to set up the driver groups and segments.

    Attributes:
        speech: object from the SPEECh class
        num_total_drivers: number of drivers in the simulation
        group_configs: sub-configuration for each driver group
        num_drivers: number of drivers assigned to each driver group

        time_step: time step for load profile simulated, in hours
        num_time_steps: number of time steps in the profile
        time_steps_per_hour: number of time steps per hour (inverse of time_step)
        energy_clip: upper bound to energy values generated by GMM

        total_load_dict: dictionary with the total load result
        all_load_dicts: dictionary with the load results for each driver group
        total_load_segments: array with the total load result
        all_load_segments: dictionary of arrays with the load results for each driver group

        remove_timers: option to remove timers from the segment specified in speech

    Methods:
        num_evs: number of electric vehicles (EVs) in simulation and driver groups.
        groups: load configurations for each of the driver groups
        change_ps_zg: can change the distribution over sessions model components, P(s|z, g)
        change_pg: can change the distribution over driver groups, P(g)
        run_all: calculates the load profiles
    """
    def __init__(self, speech, remove_timers=False):
        
        self.speech = speech
        self.group_configs = {}
        self.num_drivers = np.zeros((self.speech.data.ng, ))
        self.time_step = (1/60)
        self.num_time_steps = 1440
        self.time_steps_per_hour = 60
        self.energy_clip = 100
        self.num_total_drivers = None
        self.total_load_dict = {}
        self.total_load_segments = np.zeros((self.num_time_steps, len(self.speech.data.labels)))
        self.all_load_dicts = {}
        self.all_load_segments = {}

        self.remove_timers = remove_timers
        # self.shift_timers = False

    def num_evs(self, num_total_drivers, col='pg'):
        """Inputs the number of EVs in the simulation. Uses pg to calculate the number assigned to each driver group."""
        self.num_total_drivers = num_total_drivers
        for i in range(self.speech.data.ng):
            self.num_drivers[i] = int(self.num_total_drivers * self.speech.pg.loc[i, col])
        
    def groups(self):
        """For each driver group, create an object from the class SPEEChGroupConfiguration.
        This method also implements the timer removal option by calling change_ps_zg().
        """
        for i in range(self.speech.data.ng):
            self.group_configs[i] = SPEEChGroupConfiguration(self, i)
            if self.remove_timers:
                if i in self.speech.data.timers_dict.keys():
                    self.change_ps_zg(i, self.speech.data.timer_cat, 'weekday', self.speech.data.timers_dict[i])

    def change_ps_zg(self, g, cat, weekday, new_weights):
        """Change the distribution over sessions model components, P(s|z, g).

        Parameters:
            g: driver group
            cat: charging segment (category)
            weekday: weekday option
            new_weights: dictionary giving new weights for segment cat in driver group g

        The method alters the weights directly in the gmm without saving the original values.
        The input need only specify new weights for a subset of the components;
        the remaining weight will be distributed among the remaining components proportionately.
        """
        if cat in self.group_configs[g].segment_gmms[weekday].keys():
            
            gmm = self.group_configs[g].segment_gmms[weekday][cat]

            all_inds = np.arange(0, np.shape(gmm.weights_)[0])
            total_rem = 1
            for key, val in new_weights.items():
                gmm.weights_[key] = val
                total_rem -= val
            other_keys = np.delete(all_inds, list(new_weights.keys()))
            gmm.weights_[other_keys] = gmm.weights_[other_keys] * total_rem / sum(gmm.weights_[other_keys])

            self.group_configs[g].segment_gmms[weekday][cat] = gmm

    def change_pg(self, new_weights, new_col='pg', dend=False):
        """Change the distribution over driver groups, P(g).

        Parameters:
            new_weights: dictionary giving the new weights for the driver groups
            new_col: option to keep the original weights in column 'pg' and create a new column with the new weights

        The input need only specify new weights for a subset of the components;
        the remaining weight will be distributed among the remaining components proportionately.
        """
        
        if dend: # If inputs are given using cluster numbers from the dendrogram
            new_weights_old = copy.deepcopy(new_weights)
            new_weights = {}
            for key, val in new_weights_old.items():
                new_weights[self.speech.data.cluster_reorder_dendtoac[key]] = val

        self.speech.pg[new_col] = self.speech.pg['pg'].copy()
        all_inds = np.arange(0, self.speech.data.ng)
        total_rem = 1
        for key, val in new_weights.items():
            self.speech.pg.loc[key, new_col] = val
            total_rem -= val
        other_keys = np.delete(all_inds, list(new_weights.keys()))
        self.speech.pg.loc[other_keys, new_col] = self.speech.pg.loc[other_keys, new_col] * total_rem / sum(self.speech.pg.loc[other_keys, new_col])
        
    def run_all(self, verbose=False, weekday='weekday'):
        """Calculates the load profiles for each driver group using class LoadProfile.
        Records the results for the separate groups and for the aggregate.
        """
        self.total_load_dict = {x: np.zeros((self.num_time_steps,)) for x in self.speech.data.labels}
        self.total_load_segments = np.zeros((self.num_time_steps, len(self.speech.data.labels)))
        self.all_load_dicts = {}
        self.all_load_segments = {}
        for g in range(self.speech.data.ng):
            if verbose:
                print('Group '+str(g))
            model = LoadProfile(self, self.group_configs[g], weekday=weekday)
            model.calculate_load()
            self.all_load_dicts[g] = model.load_segments_dict
            self.all_load_segments[g] = model.load_segments_array
            for key, val in model.load_segments_dict.items():
                self.total_load_dict[key] += val
            self.total_load_segments += model.load_segments_array

        
class SPEEChGroupConfiguration(object):
    """Configuration, sessions counts, and gmms for each individual driver group.

    Attributes:
        speech_config: object from class SPEEChGeneralConfiguration
        g: driver group
        total_drivers: number of drivers in the driver group
        segment_session_numbers: dictionary of the number of sessions to generate from each charging segment
        segment_gmms: the GMM objects needed to generate sessions

    Methods:
        numbers: calculate the values in segment_session_numbers
        load_gmms: load the objects in segment_gmms
    """
    
    def __init__(self, speech_config, g):
        
        self.speech_config = speech_config
        self.g = g
        self.total_drivers = speech_config.num_drivers[g]
        self.segment_session_numbers = {'weekday': {}, 'weekend': {}}
        self.segment_gmms = {'weekday': {}, 'weekend': {}}
        
        self.numbers()
        self.load_gmms()

    def numbers(self, total_drivers=None):
        """Calculates the number of sessions for this driver group in each of the charging segments.

        Some driver groups have a very small number of charging sessions in a certain segment: too small for a sessions
        model to have been trained on. In this method, if the number of sessions in the segment is fewer than 0.1% of
        the number of drivers, we use that as a proxy to catch those cases where the model would throw an error.

        Parameters:
            total_drivers: gives the option to recalculate with a new value of total_drivers
        """
        
        if total_drivers is not None:
            self.total_drivers = total_drivers
        
        inds = np.random.choice(range(len(self.speech_config.speech.pz['weekday'][self.g])), int(self.total_drivers), replace=True)
        for cat in self.speech_config.speech.data.categories:
            if os.path.isfile(self.speech_config.speech.data.folder+'GMMs/'+'weekday'+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'):
                self.segment_session_numbers['weekday'][cat] = int(sum(self.speech_config.speech.pz['weekday'][self.g].loc[inds, cat+self.speech_config.speech.data.zkey_weekday]))
            else:
                self.segment_session_numbers['weekday'][cat] = 0
            if os.path.isfile(self.speech_config.speech.data.folder+'GMMs/'+'weekend'+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'):
                self.segment_session_numbers['weekend'][cat] = int(sum(self.speech_config.speech.pz['weekend'][self.g].loc[inds, cat+self.speech_config.speech.data.zkey_weekend]))
            else:
                self.segment_session_numbers['weekend'][cat] = 0

    def load_gmms(self):
        """Loads the GMM model object for each of the segments."""

        for cat in self.speech_config.speech.data.categories:
            weekdaykeys = ['weekday', 'weekend']
            for weekday in weekdaykeys:
                if self.segment_session_numbers[weekday][cat] > 0:
                    key = self.speech_config.speech.data.folder+'GMMs/'+weekday+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'
                    self.segment_gmms[weekday][cat] = pickle.load(open(key, "rb"))

                    
class LoadProfile(object):
    """Calculating the load profile for a given configuration.

    Attributes:
         config: configuration, object from the class SPEEChGeneralConfiguration
         group_config: group configuration, object from the class SPEEChGroupConfiguration
         weekday: weekday option
         load_segments_dict: results, dictionary of load segments
         load_segments_array: results, array of load segments

    Methods:
         calculate_load: calculate the total load for each segment
         end_times_and_load: calculate the uncontrolled load profiles for a set of sessions
    """
    
    def __init__(self, config, group_config, weekday='weekday'):
        """Other option for weekday: 'weekend'."""

        self.config = config  # speech general configuration
        self.group_config = group_config
        self.weekday = weekday
        self.load_segments_dict = {}
        self.load_segments_array = np.zeros((self.config.num_time_steps, self.config.speech.data.num_categories))

    def calculate_load(self, return_individual_session_parameters=False):
        """For each segment, calculate the total load profile.
        For each segment, the process is as follows:
            Step 1: generate sessions parameters using the segment GMM, stored in full_output
            Step 2: post-process parameters
            Step 3: calculate sessions load profiles, calling end_times_and_load
            Step 4: store the result
        """
        
        if return_individual_session_parameters:
            individual_session_parameters = {}

        for segment_number in range(self.config.speech.data.num_categories):
            cat = self.config.speech.data.categories[segment_number]
            num_vehicles = self.group_config.segment_session_numbers[self.weekday][cat]
            if num_vehicles > 0:
                gmm = self.group_config.segment_gmms[self.weekday][cat]
                full_output = gmm.sample(num_vehicles)
                output = full_output[0]
                output = output[np.random.choice(np.shape(output)[0], np.shape(output)[0], replace=False), :]
                if self.config.speech.data.start_mod == 1:
                    start_times = (self.config.speech.data.start_time_scaler * np.mod(24*3600*output[:, 0], 24*3600)).astype(int)
                else:
                    start_times = (self.config.speech.data.start_time_scaler * np.mod(output[:, 0], 24*3600)).astype(int)
                energies = np.clip(np.abs(output[:, 1]), 0, self.config.energy_clip)
                end_times, load = self.end_times_and_load(start_times, energies, self.config.speech.data.rates[segment_number])
                
                if return_individual_session_parameters:
                    individual_session_parameters[cat] = pd.DataFrame({'Start':start_times, 'Energy':np.round(energies,2), 'Duration': (self.config.speech.data.start_time_scaler * np.clip(np.abs(output[:, 2]), 0, 24*3600*7)).astype(int), 'Rate': self.config.speech.data.rates[segment_number] * np.ones(np.shape(start_times))})

            else:
                load = np.zeros((self.config.num_time_steps, ))
            self.load_segments_dict[self.config.speech.data.labels[segment_number]] = load
            self.load_segments_array[:, segment_number] = load
                
        if return_individual_session_parameters:
            return individual_session_parameters

    def end_times_and_load(self, start_times, energies, rate):
        """Calculate the load profile given data on individual sessions.

        Parameters:
             start_times: set of start time indices
             energies: energy delivered in each session, in kWh
             rate: uncontrolled max charging rate of the session, in kW

         Returns:
             end_times: set of end time indices
             load: total load from the set of sessions, a time series in kW
            """

        time_steps_per_hour = self.config.time_steps_per_hour
        num_time_steps = self.config.num_time_steps
        load = np.zeros((num_time_steps,))
        end_times = np.zeros(np.shape(start_times)).astype(int)

        lengths = (time_steps_per_hour * energies / rate).astype(int)
        extra_charges = energies - lengths * rate / time_steps_per_hour
        inds1 = np.where((start_times + lengths) > num_time_steps)[0]
        inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)

        end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+lengths[inds1]-num_time_steps, num_time_steps)).astype(int)
        end_times[inds2] = (start_times[inds2] + lengths[inds2]).astype(int)
        inds3 = np.where(end_times >= num_time_steps)[0]
        inds4 = np.delete(np.arange(0, np.shape(end_times)[0]), inds3)

        for i in range(len(inds1)):
            idx = int(inds1[i])
            load[np.arange(int(start_times[idx]), num_time_steps)] += rate * np.ones((num_time_steps - int(start_times[idx]),))
            load[np.arange(0, end_times[idx])] += rate * np.ones((end_times[idx],))
        for i in range(len(inds2)):
            idx = int(inds2[i])
            load[np.arange(int(start_times[idx]), end_times[idx])] += rate * np.ones((lengths[idx],))
        load[0] += np.sum(extra_charges[inds3] * time_steps_per_hour)
        for i in range(len(inds4)):
            load[end_times[int(inds4[i])]] += extra_charges[int(inds4[i])] * time_steps_per_hour

        return end_times, load


class Plotting(object):
    """Plotting class - including final results of load profile, intermediate profiles, and distributions."""

    def __init__(self, speech, config=None, n=5e6):

        self.speech = speech
        if config is None:
            self.config = SPEEChGeneralConfiguration(speech)
            self.config.num_evs(n)
            self.config.groups()
        else:
            self.config = config

    def pg(self, col='pg'):

        vals = np.zeros((self.speech.data.ng, ))
        for i in range(self.speech.data.ng):
            j = self.speech.data.cluster_reorder_dendtoac[i]
            vals[i] = self.speech.pg.loc[j, col]
        plt.figure()
        plt.bar(np.arange(1, self.speech.data.ng+1), vals)
        plt.ylabel('P(G)')
        plt.xlabel('G')
        plt.show()

    def total(self, verbose=False, weekday='weekday', save_str=None):

        self.config.run_all(verbose=verbose, weekday=weekday)
        self.plot_single(self.config.total_load_segments, self.config.total_load_dict, save_str=save_str)

    def groups(self, n=1e5, weekday='weekday', save_string=None):

        nrow = int(np.ceil(np.divide(self.speech.data.ng, 4)))
        fig, axes = plt.subplots(nrow, 4, sharex=True, sharey=True, figsize=(12, int(nrow*3)))
        config = copy.deepcopy(self.config)
        ymax = 0
        for i in range(self.speech.data.ng):
            row = int(np.divide(i, 4))
            col = np.mod(i, 4)
            j = self.speech.data.cluster_reorder_dendtoac[i]
            config.group_configs[j].numbers(total_drivers=n)
            config.group_configs[j].load_gmms()
            model = LoadProfile(config, config.group_configs[j], weekday=weekday)
            model.calculate_load()
            if np.max(np.sum(model.load_segments_array, axis=1)) > ymax:
                ymax = np.max(np.sum(model.load_segments_array, axis=1))
            ylab = False
            if col == 0:
                ylab = True
            axes[row, col] = self.plot(axes[row, col], model.load_segments_array, model.load_segments_dict, title='Group '+str(i), ylab=ylab)
        if (row == nrow) & (col < 3):
            for col_left in np.arange(col, 4):
                axes[row, col_left].set_axis_off()
        for i in range(nrow):
            for j in range(4):
                axes[row, col].set_ylim([0, (1/1000)*ymax])
        plt.tight_layout()
        if save_string is not None:
            plt.savefig(save_string, bbox_inches='tight')
        plt.show()

    def sessions_components(self, g, cat, weekday, n=1e5):

        gmm = self.config.group_configs[g].segment_gmms[weekday][cat]
        output = gmm.sample(n)
        output_values = output[0]
        output_labels = output[1]

        fig, ax = plt.subplots(1,1,figsize=(8,5))
        inds = np.arange(0, np.shape(output_values)[0])
        if self.speech.data.start_mod == 1:
            start_times = (self.speech.data.start_time_scaler * np.mod(24*3600*output_values[inds, 0], 24*3600)).astype(int)
        else:
            start_times = (self.speech.data.start_time_scaler * np.mod(output_values[inds, 0], 24*3600)).astype(int)
        energies = np.clip(np.abs(output_values[inds, 1]), 0, self.config.energy_clip)
        segment_number = np.where(np.array(self.speech.data.categories)==cat)[0][0]
        end_times, load = self.end_times_and_load(start_times, energies, self.speech.data.rates[segment_number])
        load_segments_array = np.zeros((self.config.num_time_steps, self.speech.data.num_categories))
        load_segments_array[:, segment_number] = load
        load_segments_dict = {self.speech.data.labels[segment_number]:load}
        ax = self.plot(ax, load_segments_array, load_segments_dict, 'Total')
        plt.tight_layout()
        plt.show()

        nc = gmm.n_components
        fig, axes = plt.subplots(1, nc, sharex=True, sharey=True, figsize=(3*nc, 3))
        ymax = 0
        for i in range(nc):
            inds = np.where(output_labels == i)[0]
            if self.speech.data.start_mod == 1:
                start_times = (self.speech.data.start_time_scaler * np.mod(24*3600*output_values[inds, 0], 24*3600)).astype(int)
            else:
                start_times = (self.speech.data.start_time_scaler * np.mod(output_values[inds, 0], 24*3600)).astype(int)
            energies = np.clip(np.abs(output_values[inds, 1]), 0, self.config.energy_clip)
            segment_number = np.where(np.array(self.speech.data.categories)==cat)[0][0]
            end_times, load = self.end_times_and_load(start_times, energies, self.speech.data.rates[segment_number])
            load_segments_array = np.zeros((self.config.num_time_steps, self.speech.data.num_categories))
            load_segments_array[:, segment_number] = load
            if np.max(load) > ymax:
                ymax = np.max(load)
            load_segments_dict = {self.speech.data.labels[segment_number]:load}
            axes[i] = self.plot(axes[i], load_segments_array, load_segments_dict, 'Weight: '+str(np.round(gmm.weights_[i], 2)))
        for i in range(nc):
            axes[i].set_ylim([0, (1/1000)*ymax])
        plt.tight_layout()
        plt.show()

    def plot(self, ax, load_segments_array, load_segments_dict, title, ylab=False):

        x = (1/60)*np.arange(0, 1440)
        mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * load_segments_array) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        for key, val in load_segments_dict.items():
            ax.plot(x, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            ax.fill_between(x, scaling * mark, scaling * (mark + val), color=self.speech.data.colours[key])
            mark += val
        ax.plot(x, scaling * mark, 'k')
        ax.set_xlim([0, np.max(x)])
        if ylab:
            ax.set_ylabel(unit, fontsize=14)
        ax.set_xlabel('Hour', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(bottom=0)

        return ax

    def end_times_and_load(self, start_times, energies, rate):

        time_steps_per_hour = self.config.time_steps_per_hour
        num_time_steps = self.config.num_time_steps
        load = np.zeros((num_time_steps,))
        end_times = np.zeros(np.shape(start_times)).astype(int)

        lengths = (time_steps_per_hour * energies / rate).astype(int)
        extra_charges = energies - lengths * rate / time_steps_per_hour
        inds1 = np.where((start_times + lengths) > num_time_steps)[0]
        inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)

        end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+lengths[inds1]-num_time_steps, num_time_steps)).astype(int)
        end_times[inds2] = (start_times[inds2] + lengths[inds2]).astype(int)
        inds3 = np.where(end_times >= num_time_steps)[0]
        inds4 = np.delete(np.arange(0, np.shape(end_times)[0]), inds3)

        for i in range(len(inds1)):
            idx = int(inds1[i])
            load[np.arange(int(start_times[idx]), num_time_steps)] += rate * np.ones((num_time_steps - int(start_times[idx]),))
            load[np.arange(0, end_times[idx])] += rate * np.ones((end_times[idx],))
        for i in range(len(inds2)):
            idx = int(inds2[i])
            load[np.arange(int(start_times[idx]), end_times[idx])] += rate * np.ones((lengths[idx],))
        load[0] += np.sum(extra_charges[inds3] * time_steps_per_hour)
        for i in range(len(inds4)):
            load[end_times[int(inds4[i])]] += extra_charges[int(inds4[i])] * time_steps_per_hour

        return end_times, load

    def plot_single(self, load_segments_array, load_segments_dict, legend_subset=None, set_ylim=None, save_str=None, title=None):

        x = (1/60)*np.arange(0, 1440)
        mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * np.sum(load_segments_array, axis=1)) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        plt.figure(figsize=(8, 5))
        for key, val in load_segments_dict.items():
            plt.plot(x, scaling * (mark + val), color=self.speech.data.colours[key])
            if legend_subset is not None:
                if key in legend_subset:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
                else:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), color=self.speech.data.colours[key])
            else:
                plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            mark += val
        plt.plot(x, scaling * mark, 'k')
        plt.legend(fontsize=12, loc='upper left')
        plt.xlim([0, np.max(x)])
        if set_ylim is None:
            plt.ylim([0, 1.1 * np.max(scaling * mark)])
        else:
            plt.ylim([0, set_ylim])
        plt.ylabel(unit, fontsize=14)
        plt.xlabel('Hour', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if title is not None:
            plt.set_title(title)
        if save_str is not None:
            plt.tight_layout()
            plt.savefig(save_str, bbox_inches='tight')
        plt.show()

    def plot_single_above_base(self, load_segments_array, load_segments_dict, base_load, legend_subset=None, set_ylim=None, save_str=None, title=None):

        x = (1/60)*np.arange(0, 1440)
        if np.shape(base_load)[0] == 24:
            xold = np.arange(0, 25)
            yold = np.zeros((25,))
            yold[np.arange(0, 24)] = np.copy(base_load)
            yold[-1] = yold[0]
            f2 = interp1d(xold, yold, kind='cubic')
            base_load = f2(x)
        elif np.shape(base_load)[0] == 288:
            xold = (1/12)*np.arange(0, 289)
            yold = np.zeros((289,))
            yold[np.arange(0, 288)] = np.copy(base_load)
            yold[-1] = yold[0]
            f2 = interp1d(xold, yold, kind='cubic')
            base_load = f2(x)

        # mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * (base_load + np.sum(load_segments_array, axis=1))) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        plt.figure(figsize=(8, 5))
        plt.plot(x, scaling*base_load, color='k', alpha=0.8)
        plt.fill_between(x, 0, scaling*base_load, color='k', alpha=0.4, label='Base Load')
        mark = base_load
        for key, val in load_segments_dict.items():
            plt.plot(x, scaling * (mark + val), color=self.speech.data.colours[key])
            if legend_subset is not None:
                if key in legend_subset:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
                else:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), color=self.speech.data.colours[key])
            else:
                plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            mark += val
        plt.plot(x, scaling * mark, 'k')
        plt.legend(fontsize=12, loc='lower left', ncol=2)
        plt.xlim([0, np.max(x)])
        if set_ylim is None:
            plt.ylim([0, 1.1 * np.max(scaling * mark)])
        else:
            plt.ylim([0, set_ylim])
        plt.ylabel(unit, fontsize=14)
        plt.xlabel('Hour', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if title is not None:
            plt.set_title(title)
        if save_str is not None:
            plt.tight_layout()
            plt.savefig(save_str, bbox_inches='tight')
        plt.show()


class Scenarios(object):
    """The scenarios for P(G) based on those in the publication. For use by the user interface tool."""

    def __init__(self, scenario_name):

        self.scenario_name = scenario_name
        self.new_weights = {}

        if scenario_name == 'BaseCase':
            self.base_case()
        elif scenario_name == 'Workplace':
            self.workplace()
        elif scenario_name == 'WorkplaceLargeBattery':
            self.workplace_large_batt()
        elif scenario_name == 'HighMUD':
            self.high_mud()
        else:
            self.original_data()

    def base_case(self):

        self.new_weights = {11: 0.05676, 12: 0.02521, 5: 0.04375, 1: 0.24707, 13: 0.14994, 3: 0.206, 8: 0.00727, 15: 0.0077,
                            2: 0.05344, 9: 0.067, 7: 0.01592, 14: 0.00891, 4: 0.0195, 6: 0.00762, 10: 0.0239, 0: 0.06004}

    def workplace(self):

        self.new_weights = {11: 0.086, 12: 0.03819, 5: 0.06628, 1: 0.18438, 13: 0.11189, 3: 0.15373, 8: 0.01102, 15: 0.01167,
                            2: 0.08096, 9: 0.05, 7: 0.02412, 14: 0.0135, 4: 0.02955, 6: 0.01154, 10: 0.03621, 0: 0.09096}

    def workplace_large_batt(self):

        self.new_weights = {11: 0.0, 12: 0.0, 5: 0.19047, 1: 0.18438, 13: 0.11189, 3: 0.15373, 8: 0.02891, 15: 0.03061,
                            2: 0.2124, 9: 0.05, 7: 0.02412, 14: 0.0135, 4: 0.0, 6: 0.0, 10: 0.0, 0: 0.0}

    def high_mud(self):

        self.new_weights = {11: 0.0172, 12: 0.00764, 5: 0.01326, 1: 0.24584, 13: 0.14919, 3: 0.20497, 8: 0.0022, 15: 0.00233,
                            2: 0.01619, 9: 0.3, 7: 0.00482, 14: 0.0027, 4: 0.00591, 6: 0.00231, 10: 0.00724, 0: 0.01819}

    def original_data(self):

        self.new_weights = {11: 0.16336, 12: 0.07255, 5: 0.12591, 1: 0.01981, 13: 0.01202, 3: 0.01652, 8: 0.02093, 15: 0.02216,
                            2: 0.1538, 9: 0.00186, 7: 0.04581, 14: 0.02564, 4: 0.05613, 6: 0.02193, 10: 0.06878, 0: 0.17279}
