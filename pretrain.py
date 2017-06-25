import csv
import sys
from preprocessing.prepare_data import DataPreprocess
from training.logistic_regression import LogReg
from training.evaluate import evaluate_model

FOLDER = "pretrained"
mmrs = []
for i in range(23):
	mmrs.append(2000 + i * 100)

offset = int(sys.argv[2])

output_file = open(FOLDER + "/results.csv", "wt")
csv_writer = csv.writer(output_file, delimiter=",")

csv_writer.writerow(['MMR', 'Data set size', 'Accuracy', 'AUC score', 'F1-score'])

for mmr in mmrs:
	in_file = open(sys.argv[1], "rt")
	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)
	print "%d - %d range" % (mmr - offset, mmr + offset)
	data_preprocess = DataPreprocess(full_list, mmr, offset=offset)
	(filtered_list, dictionaries) = data_preprocess.run()

	model_name = FOLDER + "/" + str(mmr)
	logreg = LogReg(filtered_list, dictionaries, output_model=model_name)
	[model, data_list]= logreg.run()
	
	in_file.close()

	results = evaluate_model(model, data_list)
	results = list(map(lambda x: round(x, 3) if x < 1 else x, results))
	results.insert(0, mmr)
	csv_writer.writerow(results)
	print ""

output_file.close()
