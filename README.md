# SPEECh
## Scalable Probabilistic Estimates of Electric Vehicle Charging

Primary Contact: Siobhan Powell.

Email: siobhan (dot) powell (at) stanford (dot) edu

Publications: 
S. Powell, G. V. Cezar, R. Rajagopal, "Scalable Probabilistic Estimates of Electric Vehicle Charging Given Observed Driver Behavior", submitted.

Read more about the project here: https://energy.stanford.edu/bitsandwatts/research/ev50-flagship-project/long-range-planning-ev50-what-future-demand-charging. 

Project Advisors: Ram Rajagopal, Ines Azevedo, Gustavo Cezar, Liang Min. 

Collaborators and Team: Charles Kolstad, Lesley Ryan, Noel Crisostomo, Matt Alexander, Eric Wood, Wahila Wilkie, the GISMo, S3L, and EV50 groups, and many more.

Thank you to the many collaborators who have made this project possible.

## Interactive Application

To run the interactive application locally:
0. Make sure you have a working version of Python 3 (this code was tested with Python 3.7.4).
1. Navigate to this folder in your terminal.
2. Download the data folder from: https://s3.us-west-1.amazonaws.com/speech.data/Original16.zip. Store it in a subfolder called `Data`.
3. In a new virtual environment, run `pip install -r requirements.txt`. You can also run with the packages in your main system, but the `requirements.txt` file lists versions which we know will work with together in this code. 
4. Run `flask run`.
5. Go to `localhost:5000` in your browser.
6. Play with the model and run your own scenarios!

This web application was developed by Lesley Ryan with help from Siobhan Powell. For more information on how it works, please refer to the 'About' page in the application. 


## Code
The main code is presented in `speech.py` and the file `SimpleExample.py` presents a simple example of how to use it. 

The more complex scenarios used in `RunPaperScenarios.py` generate the scenarios included in the publication, "Scalable Probabilistic Estimates of Electric Vehicle Charging Given Observed Driver Behavior". 

### Classes in SPEECh
- `DataSetConfigurations` stores information about the data set being used, such as the location and the particular segment labels/naming conventions
- `SPEECh` is the top level class storing the data and configurations. It is very general. It loads P(G) and P(z|G). 
- `SPEEChGeneralConfiguration` manages model configuration details not specific to just one of the data sets, such as the time step. It includes methods to alter the distributions over groups (`change_pg()`) and behaviours (`change_ps_zg()`). It stores an instance of `SPEEChGroupConfiguration` for each driver group.
- `SPEEChGroupConfiguration` is used for each individual driver group. It calculates the number of sessions that driver group will use in each segment, and stores the GMM session models for that driver group.
- `LoadProfile` uses the above to calculate the total load profile for a particular group
- `Plotting` includes plotting functions

### Mini Tutorial
Based on the `SimpleExample.py` file.

First, import classes from `speech.py`
```
from speech import DataSetConfigurations
from speech import SPEECh
from speech import SPEEChGeneralConfiguration
from speech import Plotting
```

Then, input the total number of EV drivers and whether you are modeling a weekday or weekend:
```
total_evs = 1000
weekday_option = 'weekday'
```

Create your main model objects. Optionally, use 'NewData' and a custom number of groups, ng, if you are working with another data set.
```
data = DataSetConfigurations('Original16', ng=16)
model = SPEECh(data)
config = SPEEChGeneralConfiguration(model)
```

At this stage you have the option to adjust the distribution over driver groups, P(G). As an example, this would adjust the weight of clusters 0 and 1 to cover 50% of drivers each. I.e. P(G=0)=0.5, P(G=1)=0.5. This step is optional.
```
config.change_pg(new_weights={0: 0.5, 1: 0.5})
```

Next, tell the configuration class how many drivers to simulate, and call `groups()` to load the group configurations. 
```
config.num_evs(total_evs)
config.groups()
```

At this stage you have the option to adjust the distribution over charging session behaviours, P(s|z, G). As an example, this would adjust the distribution for driver group 1 workplace weekday charging, giving 40% weight to the first behaviour component in the mixture model and 60% to the second. This step is optional.
```
config.change_ps_zg(1, 'Work', 'weekday', {0: 0.4, 1: 0.6})
```

Finally, calculate the load profile and plot:
```
config.run_all(weekday=weekday_option)
plots = Plotting(model, config)
plots.total(save_str='simple_example_plot_adjusted.png')
```

The plotting class also offers other methods to explore the model. `plots.pg()` plots the distribution P(G) as a bar chart. `plots.sessions_components()` plots the separate elements in the mixture model for a given segment and driver group; in this example, workplace weekday charging for cluster 1. `groups()` generates a grid of plots showing the load profile for drivers in each individual group. Use these functions to explore the driver groups and behaviours you want to change.
```
plots.pg()
plots.sessions_components(g=1, cat='Work', weekday='weekday')
plots.groups(save_string='simple_example_groups.png')
```

### Running with Custom Data
Code included in the directory `FitNewModel` will let you fit your own version of the model to your own data. The process for doing this is divided into several steps: 
1. Run `FitNewModel/process_driver_data.py` to process your data to calculate the feature vectors used for clustering the drivers. Make sure to follow the specific comments about data formatting and necessary columns. Edit the top of the file to direct to the folder containing your data. Example file name: `sessions2019.csv`. 
2. Run `FitNewModel/select_cluster_number.py` to produce an elbow plot of the driver clustering. Edit the top of the file to direct to the folder containing your data. Look for a kink or elbow in the elbow plot that is produced to select your number of clusters; this is an input to the following steps. In the paper this was 16. 
3. Run `FitNewModel/cluster_drivers.py` to cluster the drivers. Before running, edit the top of the file to direct to your data folder and to give your selected number of clusters. 
4. Use the `FitNewModel/fit_gmms.ipynb` to: 1) generate and save gaussian mixture models (GMMs) for all the driver clusters and charging segments; and 2) go through these results one by one, looking at an elbow plot of AIC, and select the optimal number of mixture components for each. These will be copied over to the final GMM folder.
5. Run `FitNewModel/postprocessing.py` to post-process the data, putting the results of the clustering into the P(G) and P(z|G) language needed by the main model. This file also has a sample method to include in the `DataSetConfigurations` class in `speech.py` where your new data set will be called `NewData`. You can edit this here and copy over the changes. 
6. Use `SimpleExample.py` to explore your resulting model, making sure to input the right number of driver groups, `ng`, and data set name, `NewData`, when you call the `DataSetConfigurations` class.


### Requirements and Set-up

This works with: 
```
    pandas == 0.25.1
    matplotlib == 3.1.1
    numpy == 1.18.1
    sklearn == 0.22.2.post1
    flask == 1.1.1
    json == 2.0.9
```    
Python 3.7.4.
