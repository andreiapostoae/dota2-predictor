import sys
import operator
import csv
import json
import numpy as np
from sklearn import metrics
from preprocessing.prepare_data import index_heroes, calculate_rating, NUMBER_OF_HEROES, DataPreprocess, get_hero_names
from training.logistic_regression import LogReg


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
model = logreg.run()



hero_list = [11, 93, 60, 83, 26, 50, 74, 28, 71]

dictionary = {}
for i in range(114):
	if (i + 1) not in hero_list:
		hero_list.append(i + 1)

		results = calculate_rating(hero_list, dictionaries[0]['winrate'], dictionaries[1]['winrate'], dictionaries[2]['winrate'])

		final_list = []
		final_list.extend(index_heroes(hero_list))
		final_list.extend(results)

		result = model.predict_proba(np.array(final_list).reshape(1, -1))

		dictionary[i + 1] = result[0][0]
		del hero_list[-1]

aux_dict = sorted(dictionary.items(), key=operator.itemgetter(1))

print "\n"
hero_map = get_hero_names("preprocessing/")
for (hero, chance) in aux_dict:
	if hero is not 24:
		print "%s: %.3f%%" % (hero_map[hero], chance * 100)

#print "Radiant: %.3f%%" % (result[0][1] * 100)
#print "Dire: %.3f%%" % (result[0][0] * 100)
