import pandas
import csv
import numpy as np
import sys

from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from preprocessing.prepare_data import DataPreprocess, calculate_rating

TEST_RATIO = 0.3
NUMBER_OF_FEATURES = 230

class LogReg(object):
	def __init__(self, filtered_list, dictionaries, output_model=None):
		self.games_list = filtered_list
		self.synergy_radiant = dictionaries[0]
		self.synergy_dire = dictionaries[1]
		self.counter = dictionaries[2]

	def construct_nparray(self):
		length = len(self.games_list)
		self.matrix = np.zeros((length, NUMBER_OF_FEATURES + 2))

		for i in range(length):
			array = np.array(self.games_list[i])
			self.matrix[i] = array

		self.matrix = np.delete(self.matrix, 0, 1)

	def split_data(self):
		X_matrix = self.matrix[:, 0:230]
		y_matrix = np.ravel(self.matrix[:, -1])

		self.X_train, self.X_test, self.y_train, self.y_test = \
			train_test_split(X_matrix, y_matrix, test_size=TEST_RATIO, random_state=0)

	def evaluate_model(self):
		self.model = LogisticRegression()
		self.model.fit(self.X_train, self.y_train)

		predicted = self.model.predict(self.X_test)
		probabilities = self.model.predict_proba(self.X_test)

		print "Data set size: %d" % len(self.games_list)
		print "Raw accuracy: %.3f" % metrics.accuracy_score(self.y_test, predicted)
		print "ROC AUC score: %.3f" % metrics.roc_auc_score(self.y_test, probabilities[:, 1])

	def run(self):
		self.construct_nparray()
		self.split_data()
		self.evaluate_model()


def main():
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