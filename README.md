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
Patches are released almost monthly, so differences between data trained in different periods of time can get significant. Because the Steam API does not (easily) provide access to a player's MMR and the skill level query is broken, so there are two steps in mining new data:
- download lists of games played starting with a sequence number directly from Steam and filter irrelevant games
```bash
cd mining
python steam_miner.py list.csv NUM_GAMES
```
- take each game from the list and find the hero configuration, the winner, and the MMR of the players who made it public by making a http request at opendota (limited to 1 request per second)
```
python opendota_miner.py list.csv output.csv NUM_GAMES
```

### Training a model

### Plotting the learning curve

### Plotting the heatmap of synergies and counter synergies

### Plotting hero winrates

### Pretraining models in a MMR interval

## Author's note
