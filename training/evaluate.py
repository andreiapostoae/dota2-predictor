""" Helper module for evaluating a trained model and plotting the learning curves """
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def evaluate_model(model, data_list):
	""" Evaluates accuracy, area under curve and f1-score for a trained model

	model -- trained data model to be analyzed
	data_list -- list of X and y split into train and test data
	"""

	[_, x_test, y_train, y_test] = data_list

	predicted = model.predict(x_test)
	probabilities = model.predict_proba(x_test)

	print "Data set size: %d" % (len(y_train) + len(y_test))
	print "Raw accuracy: %.3f" % metrics.accuracy_score(y_test, predicted)
	print "ROC AUC score: %.3f" % metrics.roc_auc_score(y_test, probabilities[:, 1])
	print "F1 score: %.3f" % metrics.f1_score(y_test, predicted)

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
