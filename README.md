# dota2-predictor

## Table of Contents
1. [Overview](#overview)
2. [Basic usage](#basic)
3. [Downloading and running](#downloading)
  * [Linux](#linux)
  * [Windows](#windows)
4. [Advanced usage](#advanced)
  * [Downloading new data](#data)
  * [Training a model](#training)
  * [Plotting the learning curve](#learning)
  * [Plotting the heatmap of synergyies and counter synergies](#heatmap)
  * [Plotting hero winrates](#winrates)
5. [Author's note](#author)
6. [FAQ](#faq)

## Overview <a name="overview"></a>
dota2-predictor is a tool that uses Machine Learning over a dataset of over 500k past matches in order to predict the outcome of a game. This project achieves roughly 0.63 ROC AUC score using both logistic regression and neural networks.

## Basic usage <a name="basic"></a>

The tool has two main use cases: one for simply predicting the outcome of the game knowing all the heroes and one for predicting what the best last pick is given a configuration of the other nine heroes. It uses the closest model available given the average MMR of your game.

For the first case, you should select all the heroes, write the average MMR and press the "Predict winner" button.

For the second case, you should select all the other nine heroes in their corresponding team, write the average MMR and press the "Suggest hero" button. A list of top 10 hero suggestions will be displayed.

| [![Suggest Hero](http://i.imgur.com/xbh8903.png)](http://i.imgur.com/xbh8903.png)  | [![Predict winner](http://i.imgur.com/MXDhZt2.png)](http://i.imgur.com/MXDhZt2.png)  |
|:---:|:---:|
| Suggest Hero | Predict Winner | 







## Downloading and running <a name="downloading"></a>
### Linux <a name="linux"></a>

The project requires Python 2.7, pip, and a handful of Python packages. Install those packages using the following commands in a terminal:
```bash
git clone https://github.com/andreiapostoae/dota2-predictor.git
cd dota2-predictor
pip install -r requirements.txt
```
For starting the tool:
```bash
python basic_gui.py
```

Alternatively, you could install Anaconda which installs the packages by itself, then run the basic_gui.py script in a conda environment. 

### Windows <a name="windows"></a>

1. Download the [zip file](https://github.com/andreiapostoae/dota2-predictor/archive/master.zip) of this repository and unzip it.
2. Install [Anaconda with Python 2.7](https://www.continuum.io/downloads) in order to have the packages mentioned in the requirements.txt already installed, without you having to manually do it. 
3. Double-click basic_gui.py, open with *C:\Users\your_user\Anaconda2\python.exe*, check "Always use this app to open .py files". You will not get prompted when you run it again. If nothing happened, go to step 4.
4. Run Anaconda Prompt.
5. Navigate to the folder where you unzipped and run the GUI script.
```bash
cd C:\Users\Apo\Desktop\dota2-predictor
python basic_gui.py
```

## Advanced usage <a name="advanced"></a>

### Downloading new data <a name="data"></a>
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

### Training a model <a name="training"></a>
The raw input CSV are filtered using a DataPreprocess object and the remaining games will be given as input for the Logistic Regression.
```bash
python -m training.logistic_regression 706d.csv 3000 200 model
```
- 706d.csv = CSV file of your mined games
- 3000 = target MMR
- 200 = offset MMR (meaning the games will be in the [2800, 3200] interval)
- model = name of output model file, saved in pkl file


### Plotting the learning curve <a name="learning"></a>
You can plot the learning curve of your model by modifying training/logistic_regression.py script.

Add the learning_curve flag in the main function and train the model normally afterwards.
```python
logreg.run(learning_curve=1)
```
![alt text](http://i.imgur.com/YxOpVtk.png)

### Plotting the heatmap of synergies and counter synergies <a name="heatmap"></a>
While training, statistics about hero synergies and counter synergies are stored in dictionaries that are saved in the pickle format, similar to the model. You can visualize those graphs using the heat_map flag.

Keep in mind that on both axis, the number represent the heroes indices (e.g. 0 is Anti-Mage, 80 is Chaos Knight etc).
```python
logreg.run(heat_map=1)
```

![alt text](http://i.imgur.com/eonS02J.png)

### Plotting hero winrates <a name="winrates"></a>
Data about hero winrates during the training phase can be plotted using the winrates flag.
As there are 113 heroes currently, it is hard to fit the plot in this README, but you can find it [here](http://i.imgur.com/Sf2WRAx.png).
```python
logreg.run(winrates=1)
```

### Pretraining models in a MMR interval <a name="pretraining"></a>
Alternatively, instead of training a single model at a time, you can train multiple models in an interval with your desired offset. The first argument should be the CSV containing the data, and the second should be the MMR offset
```bash
python pretrain.py 706d.csv 200
```
The MIN_MMR, MAX_MMR are set within the script, and the models and dictionaries will be saved in the [pretrained](https://github.com/andreiapostoae/dota2-predictor/tree/master/pretrained) folder, as well as a [CSV file](https://github.com/andreiapostoae/dota2-predictor/blob/master/pretrained/results.csv) containing the statistics for every model.


## Author's note <a name="author"></a>
This is a hobby project started with the goal of achieving as high accuracy as possible given the picks from a game. 
Of course, one could argue that there are other statistics such as GPM, XPM or itemization that influence the outcome of a game, but this tool's usage is to suggest you the best possible pick before the game starts. Other statistics are dynamic throughout the game, so they do not help the prediction.

Even though there are papers where people claim to have achieved a much higher accuracy (>70% in some cases), it highly depends on the data they used. 
We need to keep in mind that Dota 2 is a game that is constantly evolving, getting more complex as new heroes are released and also balance changes are made all the time.

There is also the human factor that strongly influences the outcome of a game, so there is no way of predicting each game with close-to-perfect accuracy.
This tool, however, is up-to-date with the current patch and does a decent job predicting your best possible last pick given a situation in order to give you that little extra chance that turns the tides in your favor.

Good luck in your matches and game on!

## FAQ <a name="faq"></a>

1. Only 60% accuracy? That is not much better than predicting that radiant always wins.
  * Yes, using logistic regression and neural networks on the current data does not seem to provide better results. However, I have some ideas on slightly improving it.

2. I had a team of 4 carries already and your tool suggested me to pick Spectre. Why?
  * You need to take the results with a grain of salt. The algorithm does not know (at least at this point) how to evaluate a team composition. Simplified, it just looks statistically on past games and evaluates each possible hero that it thinks would work. However, synergy between heroes is not properly modelled.
  * The best solution is to always pick a 5th hero whose role fits in your team and you decently know how to play.

3. Why does this tool require Anaconda to be used on Windows? Can't you just give us simple .exe?
  * I tried different approaches regarding how can I make dota2-predictor available to people as fast as possible. Generating a .exe file using pyinstaller (because the code is written in python) results in a file around 200MB. I figured out that not many people would be downloading an executable file from the internet with such size.
  * Also, installing packages with pip on Windows is a headache, so Anaconda is the simplest solution I have at the moment.

4. Any plan for a web interface so we don't have to install stuff?
  * Yes! I do not have much web development knowledge at the moment, but I will do my best to make this available to everybody **easily**.

5. Why don't you use only 6k+ games to train your model, because people at that skill level are more versatile?
  * While this is true, applying 6k logic on a 3k game will not mean the algorithm can correctly guess the outcome. There is also the lack of data on high MMR problem.

6. Do you plan on switching the algorithm?
  * Yes, I did some k-NearestNeighbors experiments which failed, but options are still open.
