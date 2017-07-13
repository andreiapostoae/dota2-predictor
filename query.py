""" Module that uses the pretrained models and dictionaries from pickle files to
predict the result of the game """
import sys
import json
import pickle
import logging
import operator
import os
from os import listdir
from sklearn.externals import joblib
from training.logistic_regression import index_heroes

MAX_MMR = 9000
MMR_INIT = 10000

def open_dictionaries(filename):
	""" Loads winrate of synergies and counter dictionaries

	filename -- path to pkl representing the list of dictionaries
	"""

	with open(filename, 'rb') as file_handle:
		return pickle.load(file_handle)

def find_hero_id(name, hero_list, logger=logging.getLogger(__name__)):
	""" Returns the id of the hero corresponding to its popular name

	name -- popular name, including abbreviations
	hero_list -- list of heroes clipped from heroes.json
	logger -- Logger object to redirect output to
	"""

	for hero in hero_list:
		if hero["name"] == name:
			return hero["id"]

	logger.critical("Hero \"%s\" not found. Check heroes.json for the correct names.", name)
	sys.exit(1)

def input_error(logger):
	""" Prints an input error using a logger

	logger -- Logger object used for displaying messages
	"""

	logger.critical("Usage1: %s <MMR> <faction> <10_hero_list>", sys.argv[0])
	logger.critical("Usage2: %s <MMR> <faction> <9_hero_list>", sys.argv[0])


def give_result(query_list, faction, model, logger):
	""" Gives the prediction of a game using a 10-hero configuration

	query_list -- list of 9 or 10 hero_ids
	faction -- radiant or dire
	model -- trained model to apply queries to
	logger -- Logger object used for displaying messages
	"""

	indexed_heroes = index_heroes(query_list)
	result = model.predict_proba([indexed_heroes])

	if faction == 'Radiant':
		logger.info("Radiant chance: %.3f%%", (result[0][1] * 100))
		return result[0][1] * 100
	else:
		logger.info("Dire chance: %.3f%%", (result[0][0] * 100))
		return result[0][0] * 100


def process_query_list(query_list, heroes, faction, model, logger):
	""" This function will print a dictionary of each hero and its corresponding win chance

	query_list -- list of 9 or 10 hero_ids
	heroes -- list of current heroes extracted from the JSON
	faction -- radiant or dire
	model -- trained model to apply queries to
	logger -- Logger object used for displaying messages
	"""

	probabilities_dict = {}

	for i in range(114):
		if i + 1 not in query_list and i != 23:
			if faction == 'Radiant':
				query_list.insert(0, i+1)
			else:
				query_list.append(i+1)

			indexed_heroes = index_heroes(query_list)
			result = model.predict_proba([indexed_heroes])

			if faction == 'Radiant':
				probabilities_dict[i] = result[0][1] * 100
				query_list.pop(0)
			else:
				probabilities_dict[i] = result[0][0] * 100
				del query_list[-1]

	sorted_dict = sorted(probabilities_dict.items(), key=operator.itemgetter(1), reverse=True)

	for (hero_id, value) in sorted_dict:
		value = round(value, 3)

		for hero in heroes:
			if hero["id"] == hero_id + 1:
				logger.info("%-25s %s%%", hero["localized_name"], value)
				break

	return sorted_dict


def main():
	""" Main function """
	logging.basicConfig(level=logging.INFO, format='%(name)-10s %(levelname)-8s %(message)s')
	logger = logging.getLogger(__name__)

	try:
		mmr = int(sys.argv[1])
	except ValueError:
		input_error(logger)
		sys.exit(1)

	if len(sys.argv) != 12 and len(sys.argv) != 13:
		input_error(logger)
		sys.exit(1)

	json_data = json.load(open(os.path.join('preprocessing', 'heroes.json'), "rt"))
	heroes = json_data["heroes"]

	file_list = [int(valid_file[:-4]) for valid_file in listdir('pretrained') \
					if 'dicts' not in valid_file and 'results' not in valid_file]

	file_list.sort()

	min_distance = MMR_INIT
	final_mmr = MMR_INIT

	for model_mmr in file_list:
		if abs(mmr - model_mmr) < min_distance:
			min_distance = abs(mmr - model_mmr)
			final_mmr = model_mmr

	if final_mmr == MMR_INIT or mmr < 0 or mmr > 9000:
		logger.critical("Please use a MMR between 0 and %d.", MAX_MMR)
		sys.exit(1)

	logger.info("Using closest model available: %d MMR", final_mmr)

	model = joblib.load(os.path.join("pretrained", str(final_mmr) + ".pkl"))
	query_list = []

	for i in range(len(sys.argv) - 3):
		name = sys.argv[i + 3]
		hero_id = find_hero_id(name, heroes)
		query_list.append(hero_id)

	faction = sys.argv[2]

	if faction != 'Radiant' and faction != 'Dire':
		input_error(logger)
		sys.exit(1)


	if len(query_list) == 10:
		give_result(query_list, faction, model, logger)
	else:
		process_query_list(query_list, heroes, faction, model, logger)


if __name__ == "__main__":
	main()
