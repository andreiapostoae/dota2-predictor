""" Helper module for evaluating a trained model and plotting the learning curves """
import json
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def get_hero_names(path=''):
	""" Returns a map of (index_hero, hero_name)

	path -- relative path to heroes.json
	"""

	with open(path + 'preprocessing/heroes.json') as data_file:
		data = json.load(data_file)

	result = data["heroes"]
	hero_map = {}
	for hero in result:
		hero_map[hero["id"]] = hero["localized_name"]

	return hero_map


def evaluate_model(model, data_list):
	""" Evaluates accuracy, area under curve and f1-score for a trained model

	model -- trained data model to be analyzed
	data_list -- list of X and y split into train and test data
	"""

	[_, x_test, y_train, y_test] = data_list

	predicted = model.predict(x_test)
	probabilities = model.predict_proba(x_test)

	dataset_size = (len(y_train) + len(y_test))
	raw_accuracy = metrics.accuracy_score(y_test, predicted)
	roc_auc_score = metrics.roc_auc_score(y_test, probabilities[:, 1])
	f1_score = metrics.f1_score(y_test, predicted)

	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	logger.info("%-25s %d", "Data set size:", dataset_size)
	logger.info("%-25s %.3f", "Raw accuracy:", raw_accuracy)
	logger.info("%-25s %.3f", "ROC AUC score:", roc_auc_score)
	logger.info("%-25s %.3f", "F1 score:", f1_score)

	return [dataset_size, raw_accuracy, roc_auc_score, f1_score]


def plot_learning_curve(data_list, subsets=20):
	""" Plots the learning curve of a trained model; plots both in terms of raw accuracy
	and AUC score

	data_list -- list of X and y split into train and test data
	subset (optionals) -- number of points in which the data is evaluated
	"""

	[x_train, x_test, y_train, y_test] = data_list
	subset_sizes = np.exp(np.linspace(3, np.log(len(y_train)), subsets)).astype(int)

	results_list = [[], [], [], []]

	for subset_size in subset_sizes:
		model = LogisticRegression()
		model.fit(x_train[:subset_size], y_train[:subset_size])

		probabilities_train_accu = metrics.roc_auc_score(y_train[:subset_size], \
			model.predict_proba(x_train[:subset_size])[:, 1])
		probabilities_test_accu = metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

		result_train_accu = metrics.accuracy_score(y_train[:subset_size], \
			model.predict(x_train[:subset_size]))
		result_test_accu = metrics.accuracy_score(y_test, model.predict(x_test))

		results_list[0].append(probabilities_train_accu)
		results_list[1].append(probabilities_test_accu)
		results_list[2].append(result_train_accu)
		results_list[3].append(result_test_accu)

	plot_data(subset_sizes, results_list)

def plot_data(subset_sizes, data_list):
	""" Plots data using a list of training and test accuracies and probabilities

	subset_sizes -- array of sizes for each subspace (logarithmic space)
	data_list -- list of X and y split into train and test data
	"""

	plt.plot(subset_sizes, data_list[0], lw=2)
	plt.plot(subset_sizes, data_list[1], lw=2)
	plt.plot(subset_sizes, data_list[2], lw=2)
	plt.plot(subset_sizes, data_list[3], lw=2)

	plt.legend(['Training (AUC)', 'Test (AUC)', 'Training', 'Test'])
	plt.xscale('log')
	plt.xlabel('Dataset size')
	plt.ylabel('Accuracy')
	plt.title('Model response to dataset size')
	plt.show()


def heatmap(dicts, index=0, show_color=0, on_screen=1):
	""" Creates a file with 114x114 colored squares representing a heatmap of the
	dataset.

	index -- 0 for all (default)
			 1 for radiant synergy
			 2 for dire synergy
			 3 for counter
	show_color -- set to 1 for showing color bar (only works with on_screen = 1)
	on_screen -- set to 1 to print the heatmap on screen, 0 for saving as PNG
	"""

	if index == 0:
		heatmap(dicts, 1, show_color, on_screen)
		heatmap(dicts, 2, show_color, on_screen)
		heatmap(dicts, 3, show_color, on_screen)
		return
	elif index == 1:
		heatmap_data = dicts[0]['winrate']
		title = 'Radiant synergy heatmap'

	elif index == 2:
		heatmap_data = dicts[1]['winrate']
		title = 'Dire synergy heatmap'
	else:
		heatmap_data = dicts[2]['winrate']
		title = 'Counter heatmap'

	fig = plt.figure(figsize=(15, 15))

	axes = fig.add_subplot(111)
	axes.set_title(title + " @ 7.06d")
	plt.imshow(heatmap_data)
	axes.set_aspect('equal')

	if show_color == 1:
		color_axes = fig.add_axes([0.12, 0.1, 0.78, 0.8])
		color_axes.get_xaxis().set_visible(False)
		color_axes.get_yaxis().set_visible(False)
		color_axes.patch.set_alpha(0)
		color_axes.set_frame_on(False)

	if on_screen == 0:
		filename = "heatmap" + str(index) + ".png"
		plt.savefig(filename, bbox_inches='tight')
	else:
		plt.colorbar(orientation='vertical')
		plt.show()

def plot_hero_winrates(radiant_winrates, dire_winrates, target_mmr, offset_mmr, radiant=1):
	""" Calculates the hero winrates over the filtered games

	radiant (optional) -- 1 for radiant games (default)
						  0 for dire games
	"""

	hero_map = get_hero_names()
	heroes_dict = {}
	hero_list = []

	for i in range(114):
		if i <= 22:
			hero_list.append(i)

			if radiant == 1:
				heroes_dict[radiant_winrates['winrate'][i]] = hero_map[i + 1]
			else:
				heroes_dict[dire_winrates['winrate'][i]] = hero_map[i + 1]

		else:
			if i != 23:
				hero_list.append(i - 1)

				if radiant == 1:
					heroes_dict[radiant_winrates['winrate'][i]] = hero_map[i + 1]
				else:
					heroes_dict[dire_winrates['winrate'][i]] = hero_map[i + 1]

	keys = heroes_dict.keys()
	keys.sort(reverse=True)

	fig = plt.figure(figsize=(20, 20))
	axes = fig.add_subplot(111)

	axes.set_ylim([-1, 113])
	axes.set_xlim([0, 0.7])
	axes.invert_yaxis()

	sorted_values = []
	for key in keys:
		sorted_values.append(heroes_dict.get(key))


	plt.yticks(hero_list, sorted_values, size=5)

	rects = axes.barh(hero_list, keys, height=0.8)

	for rect in rects:
		width = rect.get_width()
		axes.text(width * 1.03, rect.get_y() + rect.get_height()/2. + 0.75, \
			'%.2f%%' % (width * 100), ha='center', va='bottom', size=6)

	plt.title('Radiant winrate at %d - %d MMR @ 7.06d' % (target_mmr - offset_mmr, \
		target_mmr + offset_mmr), size=20)
	plt.show()

