""" Module responsible for training the preprocessed data using logistic regression """
import csv
import sys
import pickle
import random
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
from preprocessing.prepare_data import DataPreprocess, TEST_RATIO, DEFAULT_MMR_OFFSET, NUMBER_OF_HEROES
from training.evaluate import evaluate_model, plot_learning_curve, heatmap

NUMBER_OF_FEATURES = 2 * NUMBER_OF_HEROES

def get_hero_names(path=''):
	""" Returns a map of (index_hero, hero_name)

	path -- relative path to heroes.json
	"""

	with open(path + 'heroes.json') as data_file:
		data = json.load(data_file)

	result = data["heroes"]
	hero_map = {}
	for hero in result:
		hero_map[hero["id"]] = hero["localized_name"]

	return hero_map

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


def calculate_rating(heroes, synergy_radiant, synergy_dire, counter_ratio):
	""" Given a list of heroes use the dictionaries to return the synergy and counter
	features

	heroes -- list of hero_ids
	synergy_ratio_1 -- dictionary of radiant synergy winrates
	synergy_ratio_2 -- dictionary of dire synergy winrates
	counter_ration -- dictionary of radiant heroes that counter dire heroes
	"""

	counter = 0
	radiant = 0
	dire = 0

	for j in range(5):
		for k in range(5):
			hero_radiant_1 = int(heroes[j]) - 1
			hero_radiant_2 = int(heroes[k]) - 1
			hero_dire_1 = int(heroes[j + 5]) - 1
			hero_dire_2 = int(heroes[k + 5]) - 1
			if j != k:
				radiant += synergy_radiant[hero_radiant_1][hero_radiant_2]
				dire += synergy_dire[hero_dire_1][hero_dire_2]
			counter += counter_ratio[hero_radiant_1][hero_dire_2]

	return [radiant - dire, counter]

def initialize_dict(dictionary, dims=2):
	""" Initializes the dictionary with 3 pairs of 114x114 numpy matrices

	dictionary -- dictionary to be initialized
	dims -- number of dimensions of the numpy array
	"""

	if dims == 2:
		dictionary['apps'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))
		dictionary['wins'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))
		dictionary['winrate'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))
	else:
		dictionary['apps'] = np.zeros(NUMBER_OF_HEROES)
		dictionary['wins'] = np.zeros(NUMBER_OF_HEROES)
		dictionary['winrate'] = np.zeros(NUMBER_OF_HEROES)

def update_dicts(heroes, radiant_win, dicts):
	""" Given a list of heroes, update the dictionaries when specific heroes are present

	heroes -- list of hero_ids
	dicts -- list of dictionaries:
		synergy_radiant -- dictionary of radiant synergy winrates
		synergy_dire -- dictionary of dire synergy winrates
		counter_ratio -- dictionary of radiant heroes that counter dire heroes
		winrate_radiant -- dictionary of radiant winrate
		winrate_dire -- dictionary of dire winrate
	"""

	synergy_radiant = dicts[0]
	synergy_dire = dicts[1]
	counter = dicts[2]

	for j in range(5):
		hero_radiant_1 = int(heroes[j]) - 1
		hero_dire_1 = int(heroes[j + 5]) - 1

		for k in range(5):
			hero_radiant_2 = int(heroes[k]) - 1
			hero_dire_2 = int(heroes[k + 5]) - 1

			if j != k:
				synergy_radiant['apps'][hero_radiant_1][hero_radiant_2] += 1
				synergy_dire['apps'][hero_dire_1][hero_dire_2] += 1

				if radiant_win == 1:
					synergy_radiant['wins'][hero_radiant_1][hero_radiant_2] += 1
				else:
					synergy_dire['wins'][hero_dire_1][hero_dire_2] += 1

			counter['apps'][hero_radiant_1][hero_dire_2] += 1
			if radiant_win == 1:
				counter['wins'][hero_radiant_1][hero_dire_2] += 1



def calculate_synergy_winrates(dicts):
	""" Using the number of wins and number of total games for each pair of heroes,
	calculate their winrate

	dicts -- list of dictionaries:
		synergy_radiant -- dictionary of radiant synergy winrates
		synergy_dire -- dictionary of dire synergy winrates
		counter_ratio -- dictionary of radiant heroes that counter dire heroes
		winrate_radiant -- dictionary of radiant winrate
		winrate_dire -- dictionary of dire winrate
	"""

	#radiant_winrate = dicts[3]
	#dire_winrate = dicts[4]

	for i in range(NUMBER_OF_HEROES):
		for j in range(NUMBER_OF_HEROES):
			if i != j:
				for k in range(3):
					if dicts[k]['apps'][i][j] == 0.0:
						dicts[k]['winrate'][i][j] = 0.5
					else:
						dicts[k]['winrate'][i][j] = \
							dicts[k]['wins'][i][j] / float(dicts[k]['apps'][i][j])


def save_dictionaries(dicts, name):
	""" Saves the list of dictionaries to a file

	dicts -- list of dictionaries
	name -- file path
	"""

	with open(name, 'wb') as file_handle:
		pickle.dump(dicts, file_handle, pickle.HIGHEST_PROTOCOL)


def split_data(matrix):
	""" Splits the data in the preprocessed matrix into test and train data

	matrix -- matrix to be split
	"""

	x_matrix = matrix[:, 0:10]
	y_matrix = np.ravel(matrix[:, -1])

	x_train, x_test, y_train, y_test = \
		train_test_split(x_matrix, y_matrix, test_size=TEST_RATIO, random_state=0)

	return [x_train, x_test, y_train, y_test]

class LogReg(object):
	""" Class for using the logistic regression algorithm, given the preprocessed data

	filtered_list -- list of games that are valid for the query
	dictionaries -- preprocessed synergy and counter data
	output_model (optional) -- filename for the model to be outputted to (pkl format)
	"""


	def __init__(self, filtered_list, output_model=None):
		self.games_list = filtered_list
		self.model_name = output_model
		self.initialize_dicts()

	def initialize_dicts(self):
		""" Initializes the dictionaries:
		radiant_synergy, dire_synergy, counter, radiant_winrate, dire_winrate
		"""
		self.dicts = [{}, {}, {}, {}, {}]

		initialize_dict(self.dicts[0], 2)
		initialize_dict(self.dicts[1], 2)
		initialize_dict(self.dicts[2], 2)
		initialize_dict(self.dicts[3], 1)
		initialize_dict(self.dicts[4], 1)

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

	def add_extra_features(self, split_list):
		[x_train, x_test, y_train, y_test] = split_list
		train_len = x_train.shape[0]
		test_len = x_test.shape[0]

		print train_len
		print test_len

		x_train_new = np.zeros((x_train.shape[0], NUMBER_OF_FEATURES))
		x_test_new = np.zeros((x_test.shape[0], NUMBER_OF_FEATURES))
		
		for i in range(train_len):
			hero_list = x_train[i]
			radiant_win = y_train[i]
			update_dicts(hero_list, radiant_win, self.dicts)


		calculate_synergy_winrates(self.dicts)


		for i in range(train_len):
			hero_list = x_train[i]
			results = calculate_rating(hero_list, self.dicts[0]['winrate'], self.dicts[1]['winrate'], self.dicts[2]['winrate'])
			x_train_new[i, :NUMBER_OF_FEATURES] = index_heroes(hero_list)


		for i in range(test_len):
			hero_list = x_test[i]
			results = calculate_rating(hero_list, self.dicts[0]['winrate'], self.dicts[1]['winrate'], self.dicts[2]['winrate'])
			x_test_new[i, :NUMBER_OF_FEATURES] = index_heroes(hero_list)

		return [x_train_new, x_test_new, y_train, y_test]


	def train_model(self, data_list, evaluate=0, learning_curve=0, heat_map=0):
		""" Trains the model given the data list

		data_list -- X and y matrices split into train and test
		evaluate_model -- print model metrics
		learning_curve (optional) -- set to 1 for plotting the learning curve
		"""

		[x_train, x_test, y_train, y_test] = data_list
		model = LogisticRegression()
		model.fit(x_train, y_train)

		if evaluate == 1:
			evaluate_model(model, data_list)

		if learning_curve == 1:
			plot_learning_curve(data_list)

		if heat_map == 1:
			heatmap(self.dicts, index=0, show_color=0, on_screen=1)

		if self.model_name is not None:
			joblib.dump(model, self.model_name + ".pkl")
			dicts = [self.dicts[0]['winrate'], self.dicts[1]['winrate'], \
				self.dicts[2]['winrate']]
			save_dictionaries(dicts, self.model_name + "_dicts.pkl")

		return [model, data_list]

	def run(self, learning_curve=0, heatmap=0, evaluate=0):
		""" Does the training """
		matrix = self.construct_nparray()
		[x_train, x_test, y_train, y_test] = split_data(matrix)

		aux_list = self.add_extra_features([x_train, x_test, y_train, y_test])
		results = self.train_model(aux_list, learning_curve=learning_curve, heat_map=heatmap, evaluate=evaluate)
		return results

def main():
	""" Main function """

	if len(sys.argv) < 3:
		sys.exit("Usage: %s input_file MMR [offset]" % sys.argv[0])

	try:
		in_file = open(sys.argv[1], "rt")
	except IOError:
		sys.exit("Invalid input file")

	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)

	try:
		mmr = int(sys.argv[2])
	except ValueError:
		sys.exit("Invalid MMR")

	if mmr < 0 or mmr > 5000:
		sys.exit("Invalid MMR")

	try:
		offset = int(sys.argv[3])
		data_preprocess = DataPreprocess(full_list, mmr, offset)
	except ValueError:
		print "The offset is invalid. Using the default offset (%d)" % DEFAULT_MMR_OFFSET
		data_preprocess = DataPreprocess(full_list, mmr)


	filtered_list = data_preprocess.run()

	print "Finished data preprocessing\n"
	logreg = LogReg(filtered_list)
	logreg.run(heatmap=1)

if __name__ == "__main__":
	main()
