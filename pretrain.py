""" Pretrains models in MIN_MMR, MAX_MMR range in iterations of 100 """
import csv
import sys
from preprocessing.prepare_data import DataPreprocess
from training.logistic_regression import LogReg
from training.evaluate import evaluate_model

FOLDER = "pretrained"
MIN_MMR = 2000
MAX_MMR = 4200

def process_row(row):
	""" Rounds accuracy, AUC, F1-score to 3 decimals

	row -- list of elements in a csv row
	"""

	for element in row:
		if element < 1:
			element = round(element, 3)

def filter_and_train(mmr, offset):
	""" Filters games within (mmr - offset, mmr + offset), trains the model
	and writes the results in FOLDER/results.csv

	mmr -- target mmr
	offset -- gives the range of search
	"""

	try:
		in_file = open(sys.argv[1], "rt")
	except IOError:
		sys.exit("Usage: %s input_file offset" % sys.argv[0])

	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)

	print "%d - %d range" % (mmr - offset, mmr + offset)

	data_preprocess = DataPreprocess(full_list, mmr, offset=offset)
	(filtered_list, dictionaries) = data_preprocess.run()

	model_name = FOLDER + "/" + str(mmr)
	logreg = LogReg(filtered_list, dictionaries, output_model=model_name)
	[model, data_list] = logreg.run()

	results = evaluate_model(model, data_list)
	process_row(results)

	results.insert(0, mmr)
	print ""
	in_file.close()

	return results


def main():
	""" Main function """

	try:
		offset = int(sys.argv[2])
	except ValueError:
		sys.exit("Usage: %s input_file MMR [offset]" % sys.argv[0])

	output_path = FOLDER + "/results.csv"
	try:
		output_file = open(output_path, "wt")
	except IOError:
		sys.exit("Could not open %s" % output_path)

	csv_writer = csv.writer(output_file, delimiter=",")
	csv_writer.writerow(['MMR', 'Data set size', 'Accuracy', 'AUC score', 'F1-score'])

	mmrs = []
	for i in range((MAX_MMR - MIN_MMR) / 100):
		mmrs.append(MIN_MMR + i * 100)

	for mmr in mmrs:
		results = filter_and_train(mmr, offset)
		csv_writer.writerow(results)

	output_file.close()

if __name__ == "__main__":
	main()
