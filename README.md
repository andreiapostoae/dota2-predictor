# dota2-predictor

## Overview
dota2-predictor is a tool that uses Machine Learning over a dataset of over 500000 past matches in order to predict the outcome of a game. This project achieves roughly 0.63 ROC AUC score using both logistic regression and neural networks.

## Requirements
The project requires a handful of python packages. Install them using:
```bash
pip install -r requirements.txt
```

## Basic usage
dota2-predictor has two main use cases: one for simply predicting the outcome of the game knowing all the heroes and one for predicting what the best last pick is given a configuration of the other nine heroes.

You can customise your preferred hero names in the preprocessing/heroes.json file.

### Predicting the outcome of the game
```bash
python query.py 3520 Dire Luna SD WK TA PA AM Kunkka Tide Phoenix Zeus
```
The first argument is the average MMR of your game and is followed by a list of the 10 heroes: first 5 must be the radiant team and last 5 must be the dire team.
The program will find you the best pretrained model given the average MMR of the game and output the chance to win.
```
Using closest model available: 3500 MMR
Dire chance: 47.217%
```

### Predicting the best last pick
```bash
python query.py 3520 Radiant Luna SD WK TA PA AM Kunkka Tide Phoenix
```
The main difference from the previous example is that now you only input 9 heroes, in their respective order. You will get a list of possible picks and their corresponding chances in order to increase your chances of winning the game.
```
Wisp                      40.068%
Alchemist                 40.801%
Oracle                    41.269%
[...]
Sven                      60.3%
Ursa                      60.561%
Abyssal Underlord         63.929%
```

## Advanced usage

### Downloading new data
Patches are released almost monthly, so differences between data trained in different periods of time can get significant. Because the Steam API does not (easily) provide access to a player's MMR and the skill level query is broken, there are two steps in mining new data:
- download lists of games played starting with a sequence number directly from Steam and filter irrelevant games
```bash
cd mining
python steam_miner.py list.csv NUM_GAMES
```
- take each game from the list and find the hero configuration, the winner, and the MMR of the players who made it public using the opendota API (limited to 1 request per second)
```
python opendota_miner.py list.csv output.csv NUM_GAMES
```

### Training a model
The raw input CSV are filtered using a DataPreprocess object and the remaining games will be given as input for the Logistic Regression.
```python
# filter the games in the [mmr - offset, mmr + offset] interval
data_preprocess = DataPreprocess(full_list, mmr, offset)
filtered_list = data_preprocess.run()

# instantiate a LogReg object using the filtered games
log_reg = LogReg(filtered_list, mmr, offset, output_model="my_model")

# set evaluate to 1 to display information about the dataset and the training accuracy
logreg.run(evaluate=1)
```

This will save your model and synergy dictionaries in pickle format, which you can load and query later using the functions in evaluate.py.

### Plotting the learning curve
You can plot the learning curve of your model using the learning_curve flag.
```python
logreg.run(evaluate=1, learning_curve=1)
```
![alt text](http://i.imgur.com/YxOpVtk.png)

### Plotting the heatmap of synergies and counter synergies
While training, statistics about hero synergies and counter synergies are stored in dictionaries that are saved in the pickle format, similar to the model. You can visualize those graphs using the heat_map flag.

Keep in mind that on both axis, the number represent the heroes indices (e.g. 0 is Anti-Mage, 80 is Chaos Knight etc).
```python
logreg.run(evaluate=1, heat_map=1)
```

![alt text](http://i.imgur.com/eonS02J.png)

### Plotting hero winrates
Data about hero winrates during the training phase can be plotted using the winrates flag.
As there are 113 heroes currently, it is hard to fit the plot in this README, but you can find it [here](http://i.imgur.com/Sf2WRAx.png).
```python
logreg.run(evaluate=1, winrates=1)
```

### Pretraining models in a MMR interval
Alternatively, instead of training a single model at a time, you can train multiple models in an interval with your desired offset. The first argument should be the CSV containing the data, and the second should be the MMR offset
```bash
python pretrain.py 706d.csv 200
```
The MIN_MMR, MAX_MMR are set within the script, and the models and dictionaries will be saved in the [pretrained](https://github.com/andreiapostoae/dota2-predictor/tree/master/pretrained) folder, as well as a [CSV file](https://github.com/andreiapostoae/dota2-predictor/blob/master/pretrained/results.csv) containing the statistics for every model.


## Author's note
This is a hobby project started with the goal of achieving as high accuracy as possible given the picks from a game. 
Of course, one could argue that there are other statistics such as GPM, XPM or itemization that influence the outcome of a game, but this tool's usage is to suggest you the best possible pick before the game starts. Other statistics are dynamic throughout the game, so they do not help the prediction.

Even though there are papers where people claim to have achieved a much higher accuracy (>70% in some cases), it highly depends on the data they used. 
We need to keep in mind that Dota 2 is a game that is constantly evolving, getting more complex as new heroes are released and also balance changes are made all the time.

There is also the human factor that strongly influences the outcome of a game, so there is no way of predicting each game with close-to-perfect accuracy.
This tool, however, is up-to-date with the current patch and does a decent job predicting your best possible last pick given a situation in order to give you that little extra chance that turns the tides in your favor.

Good luck in your matches and game on!
