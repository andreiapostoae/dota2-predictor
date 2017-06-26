""" Module responsible for training the preprocessed data using logistic regression """
import csv
import sys
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from preprocessing.prepare_data import DataPreprocess, DEFAULT_MMR_OFFSET, NUMBER_OF_HEROES
from training.evaluate import evaluate_model, plot_learning_curve

TEST_RATIO = 0.25
NUMBER_OF_FEATURES = 2 * NUMBER_OF_HEROES + 2

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

	x_matrix = matrix[:, 0:NUMBER_OF_FEATURES]
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

	def __init__(self, filtered_list, dictionaries, output_model=None):
		self.games_list = filtered_list
		self.synergy_radiant = dictionaries[0]
		self.synergy_dire = dictionaries[1]
		self.counter = dictionaries[2]
		self.model_name = output_model

	def construct_nparray(self):
		""" Takes the list of preprocessed games and turns it into an np.array """
		length = len(self.games_list)
		matrix = np.zeros((length, NUMBER_OF_FEATURES + 2))

		for i in range(length):
			array = np.array(self.games_list[i])
			matrix[i] = array

		matrix = np.delete(matrix, 0, 1)
		return matrix

	def train_model(self, data_list, evaluate=0, learning_curve=0):
		""" Trains the model given the data list

		data_list -- X and y matrices split into train and test
		evaluate_model -- print model metrics
		learning_curve (optional) -- set to 1 for plotting the learning curve
		"""

		[x_train, _, y_train, _] = data_list
		model = LogisticRegression()
		model.fit(x_train, y_train)

		if evaluate == 1:
			evaluate_model(model, data_list)

		if learning_curve == 1:
			plot_learning_curve(data_list)

		if self.model_name is not None:
			joblib.dump(model, self.model_name + ".pkl")
			dicts = [self.synergy_radiant['winrate'], self.synergy_dire['winrate'], \
				self.counter['winrate']]
			save_dictionaries(dicts, self.model_name + "_dicts.pkl")

		return [model, data_list]

	def run(self, learning_curve=0):
		""" Does the training """
		matrix = self.construct_nparray()
		[x_train, x_test, y_train, y_test] = split_data(matrix)
		results = self.train_model([x_train, x_test, y_train, y_test], learning_curve)
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


	(filtered_list, dictionaries) = data_preprocess.run()

	print "Finished data preprocessing\n"
	logreg = LogReg(filtered_list, dictionaries)
	logreg.run()

if __name__ == "__main__":
	main()
