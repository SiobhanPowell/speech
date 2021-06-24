data_folder = 'Folder/' # *fill with correct location*

''' Data should be preprocessed and located in the above folder. Description of data format:

Description of data: 
- One data file per year called, for example, <sessions2019.csv>. 
- Each row represents a unique charging session.
- Columns must include parameters about the driver: `Driver ID` and `Battery Capacity` in kWh
- Columns must include parameters about the session: `Energy (kWh)`, `Max Power`, 
duration `Session Time (secs)`, and start time
- Several columns describe the start time: `start_day` which gives the day of the year (1 to 365), 
`start_weekday` which gives the weekday of the session start time (0 is Monday, 6 is Sunday),
and `start_seconds` gives the second of the day when the session started.
- One column must denote the category of the charging session, `Category` (i.e. Workplace, Single family residential, 
Multifamily Home Service, or other)
- If no location or zipcode data for the session is available, can comment out that line and element in the feature vector
- The data should also be *cleaned*, e.g. energies and durations within expected bounds (nonnegative)
'''

import pandas as pd
import numpy as np

weekdays_ct = {'2015': 5*52+1, '2016': 5*52+1, '2017': 5*52, '2018': 5*52+1, '2019': 5*52+1, '2020': 5*52+2}
weekends_ct = {'2015': 2*52, '2016': 2*52+1, '2017': 2*52+1, '2018': 2*52, '2019': 2*52, '2020': 2*52}

# for year in ['2019', '2015', '2016', '2017', '2018', '2020']:
year = '2019'

print(year)
data = pd.read_csv(data_folder+'sessions'+year+'.csv', index_col=0)

print('Num sessions: ', len(data))
drivers = list(set(data['Driver ID']))
print('Num drivers: ', len(drivers))

keep_drivers = []
all_drivers = list(set(data['Driver ID']))
print('Total number of drivers: ', len(all_drivers))
for i in range(len(all_drivers)):
    if np.mod(i, 10000) == 0:
        print('On driver ', i)
    driver = all_drivers[i]
    subsub = data[data['Driver ID']==driver]
    if subsub['Energy (kWh)'].max() <= min(list(set(subsub['Battery Capacity']))):
        keep_drivers.append(driver)

print('Keeping fraction: ', len(keep_drivers) / len(drivers))

driver_stats = pd.DataFrame({'Unique Driver ID':[0]}, index=[0])

weekdays_2019 = weekdays_ct[year]
weekends_2019 = weekends_ct[year]

i = 0
for ct in range(len(keep_drivers)):
    driver = keep_drivers[ct]
    subset = data[data['Driver ID']==driver]
    subset = subset.sort_values(by='start_day')
    if len(subset) >= 25:

        driver_stats.loc[i, 'Unique Driver ID'] = driver

        driver_stats.loc[i, 'Num Sessions'] = len(subset)
        driver_stats.loc[i, 'Num Zip Codes'] = len(set(subset['Zip Code']))  # remove if no zipcode data available
        driver_stats.loc[i, 'Battery Capacity'] = min(list(set(subset['Battery Capacity'])))

        wp_set = subset[subset['Category'] == 'Workplace']
        res_set = subset[subset['Category'] == 'Single family residential']
        mud_set = subset[subset['Category'] == 'Multifamily Home Service']
        other_set = subset[~(subset['Category'].isin(['Workplace', 'Single family residential',
                                                      'Multifamily Home Service']))]
        other_slow_set = other_set[other_set['Max Power']<20]
        other_fast_set = other_set[other_set['Max Power']>=20]

        driver_stats.loc[i, 'Num Workplace Sessions'] = len(wp_set)
        driver_stats.loc[i, 'Num Single Family Residential Sessions'] = len(res_set)
        driver_stats.loc[i, 'Num MUD Sessions'] = len(mud_set)
        driver_stats.loc[i, 'Num Other Slow Sessions'] = len(other_slow_set)
        driver_stats.loc[i, 'Num Other Fast Sessions'] = len(other_fast_set)

        loc_dict = {'Home': res_set, 'Work': wp_set, 'Other Slow': other_slow_set, 'Other Fast': other_fast_set,
                    'MUD': mud_set}
        for location in ['Work', 'Home', 'Other Slow', 'Other Fast', 'MUD']:
            subsub = loc_dict[location]
            if len(subsub) > 0:
                driver_stats.loc[i, location+' - Session energy - mean'] = subsub['Energy (kWh)'].mean()
                driver_stats.loc[i, location+' - Session time - mean'] = ((1/3600)*(subsub['Session Time (secs)'])).mean()
                driver_stats.loc[i, location+' - Start hour - mean'] = ((1/3600)*(subsub['start_seconds'])).mean()

                weekday_subsub = subsub[subsub['start_weekday'].isin([0, 1, 2, 3, 4])]
                weekend_subsub = subsub[subsub['start_weekday'].isin([5, 6])]

                driver_stats.loc[i, location+' - Weekend fraction'] = len(weekend_subsub) / len(subset)

                driver_stats.loc[i, location+' - Average sessions per weekday'] = len(weekday_subsub) / weekdays_2019
                driver_stats.loc[i, location+' - Average sessions per weekendday'] = len(weekend_subsub) / weekends_2019
                driver_stats.loc[i, location+' - Fraction of weekdays with session'] = len(set(weekday_subsub['start_day']))/weekdays_2019
                driver_stats.loc[i, location+' - Fraction of weekenddays with session'] = len(set(weekend_subsub['start_day']))/weekends_2019

            else:
                driver_stats.loc[i, location+' - Session energy - mean'] = 0
                driver_stats.loc[i, location+' - Session time - mean'] = 0
                driver_stats.loc[i, location+' - Start hour - mean'] = 0
                driver_stats.loc[i, location+' - Weekend fraction'] = 0
                driver_stats.loc[i, location+' - Average sessions per weekday'] = 0
                driver_stats.loc[i, location+' - Average sessions per weekendday'] = 0
                driver_stats.loc[i, location+' - Fraction of weekdays with session'] = 0
                driver_stats.loc[i, location+' - Fraction of weekenddays with session'] = 0


        i += 1

driver_stats.to_csv(data_folder+'sessions'+year+'_driverdata.csv')
print('Number of drivers:', len(driver_stats))


