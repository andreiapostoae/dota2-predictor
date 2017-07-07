""" Module responsible for training the preprocessed data using logistic regression """
import csv
import sys
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import preprocessing.prepare_data as prep
from training.evaluate import evaluate_model, plot_learning_curve, heatmap, plot_hero_winrates
import training.meta as meta


NUMBER_OF_HEROES = prep.NUMBER_OF_HEROES
NUMBER_OF_FEATURES = 2 * NUMBER_OF_HEROES


def index_heroes(heroes):
	""" Converts a list of heroes into list of 0s and 1s

	heroes -- list of heroes to be converted
	"""

	heroes_indexed = []
	for i in range(2 * NUMBER_OF_HEROES):
		heroes_indexed.append(0)

	for i in range(10):
		if i < 5:
			heroes_indexed[int(heroes[i]) - 1] = 1
		else:
			heroes_indexed[int(heroes[i]) + 113] = 1

	return heroes_indexed


def split_data(matrix):
	""" Splits the data in the preprocessed matrix into test and train data

	matrix -- matrix to be split
	"""

	x_matrix = matrix[:, 0:10]
	y_matrix = np.ravel(matrix[:, -1])

	x_train, x_test, y_train, y_test = \
		train_test_split(x_matrix, y_matrix, test_size=prep.TEST_RATIO, random_state=0)

	return [x_train, x_test, y_train, y_test]

class LogReg(object):
	""" Class for using the logistic regression algorithm, given the preprocessed data

	filtered_list -- list of games that are valid for the query
	dictionaries -- preprocessed synergy and counter data
	output_model (optional) -- filename for the model to be outputted to (pkl format)
	"""


	def __init__(self, filtered_list, mmr, offset, output_model=None):
		self.games_list = filtered_list
		self.model_name = output_model
		self.target_mmr = mmr
		self.offset_mmr = offset

		self.dicts = [{}, {}, {}, {}, {}]
		self.initialize_dicts()

	def initialize_dicts(self):
		""" Initializes the dictionaries:
		radiant_synergy, dire_synergy, counter, radiant_winrate, dire_winrate
		"""
		meta.initialize_dict(self.dicts[0], 2)
		meta.initialize_dict(self.dicts[1], 2)
		meta.initialize_dict(self.dicts[2], 2)
		meta.initialize_dict(self.dicts[3], 1)
		meta.initialize_dict(self.dicts[4], 1)

	def construct_nparray(self):
		""" Takes the list of preprocessed games and turns it into an np.array """
		length = len(self.games_list)

		# 10 heroes + radiant_win
		matrix = np.zeros((length, 11))

		for i in range(length):
			heroes = self.games_list[i][1:11]
			radiant_win = self.games_list[i][0]
			matrix[i, :10] = heroes
			matrix[i, 10] = radiant_win

		return matrix

	def construct_dicts(self, split_list):
		"""
		adds new features using the dictionaries containing synergies
		"""

		[x_train, x_test, y_train, y_test] = split_list
		train_len = x_train.shape[0]
		test_len = x_test.shape[0]

		logging.basicConfig(level=logging.INFO)
		logger = logging.getLogger(__name__)

		logger.info("%-25s %d", "Training set size:", train_len)
		logger.info("%-25s %d", "Test set size:", test_len)

		x_train_new = np.zeros((x_train.shape[0], NUMBER_OF_FEATURES))
		x_test_new = np.zeros((x_test.shape[0], NUMBER_OF_FEATURES))

		for i in range(train_len):
			hero_list = x_train[i]
			radiant_win = y_train[i]
			meta.update_dicts(hero_list, radiant_win, self.dicts)

		meta.calculate_synergy_winrates(self.dicts)

		for i in range(train_len):
			hero_list = x_train[i]
			x_train_new[i, :NUMBER_OF_FEATURES] = index_heroes(hero_list)


		for i in range(test_len):
			hero_list = x_test[i]
			x_test_new[i, :NUMBER_OF_FEATURES] = index_heroes(hero_list)

		return [x_train_new, x_test_new, y_train, y_test]


	def train_model(self, data_list, options=(0, 0, 0, 0)):
		""" Trains the model given the data list

		data_list -- X and y matrices split into train and test
		evaluate_model -- show model metrics
		learning_curve (optional) -- set to 1 for plotting the learning curve
		"""

		[x_train, _, y_train, _] = data_list
		model = LogisticRegression()
		model.fit(x_train, y_train)

		if options[0] == 1:
			evaluate_model(model, data_list)

		if options[1] == 1:
			plot_learning_curve(data_list)

		if options[2] == 1:
			heatmap(self.dicts, index=0, show_color=0, on_screen=1)

		if options[3] == 1:
			plot_hero_winrates(self.dicts[3], self.dicts[4], self.target_mmr, self.offset_mmr)

		if self.model_name is not None:
			joblib.dump(model, self.model_name + ".pkl")
			dicts = [self.dicts[0]['winrate'], self.dicts[1]['winrate'], \
				self.dicts[2]['winrate']]
			meta.save_dictionaries(dicts, self.model_name + "_dicts.pkl")

		return [model, data_list]

	def run(self, learning_curve=0, heat_map=0, evaluate=0, winrates=0):
		""" Does the training """
		matrix = self.construct_nparray()
		[x_train, x_test, y_train, y_test] = split_data(matrix)

		aux_list = self.construct_dicts([x_train, x_test, y_train, y_test])
		results = self.train_model(aux_list, \
									(evaluate, learning_curve, heat_map, winrates))
		return results

def main():
	""" Main function """
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	if len(sys.argv) < 3:
		logger.critical("Usage: %s input_file MMR [offset]", sys.argv[0])
		sys.exit(1)

	try:
		in_file = open(sys.argv[1], "rt")
	except IOError:
		logger.critical("Invalid input file")
		sys.exit(1)

	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)

	try:
		mmr = int(sys.argv[2])
	except ValueError:
		logger.critical("Invalid MMR")
		sys.exit(1)

	if mmr < 0 or mmr > 5000:
		logger.critical("Invalid MMR")
		sys.exit(1)

	try:
		offset = int(sys.argv[3])
		data_preprocess = prep.DataPreprocess(full_list, mmr, offset)
	except ValueError:
		logger.error("The offset is invalid. Using the default offset (%d)", prep.DEFAULT_MMR_OFFSET)
		data_preprocess = prep.DataPreprocess(full_list, mmr)

	filtered_list = data_preprocess.run()

	logger.info("Finished data preprocessing")

	logreg = LogReg(filtered_list, mmr, offset)
	logreg.run(evaluate=1)

if __name__ == "__main__":
	main()
