""" Module that uses the pretrained models and dictionaries from pickle files to
predict the result of the game """
import sys
import json
import pickle
from sklearn.externals import joblib
from preprocessing.prepare_data import index_heroes, calculate_rating

def open_dictionaries(filename):
	""" Loads winrate of synergies and counter dictionaries

	filename -- path to pkl representing the list of dictionaries
	"""

	with open(filename, 'rb') as file_handle:
		return pickle.load(file_handle)

def find_hero_id(name, hero_list):
	""" Returns the id of the hero corresponding to its popular name

	name -- popular name, including abbreviations
	hero_list -- list of heroes clipped from heroes.json
	"""

	for hero in hero_list:
		if hero["name"] == name:
			return hero["id"]

	sys.exit("hero %s not found" % name)

def main():
	""" Main function """

	try:
		mmr = sys.argv[1]
	except ValueError:
		sys.exit("Usage: %s MMR <hero_list>" % sys.argv[0])


	input_file = open("preprocessing/heroes.json", "rt")
	json_data = json.load(input_file)
	heroes = json_data["heroes"]

	query_list = []

	for i in range(10):
		name = sys.argv[i + 2]
		hero_id = find_hero_id(name, heroes)
		query_list.append(hero_id)

	dicts = open_dictionaries("pretrained/" + str(mmr) + "_dicts.pkl")
	model = joblib.load("pretrained/" + str(mmr) + ".pkl")

	indexed_heroes = index_heroes(query_list)
	[synergy, counter] = calculate_rating(query_list, dicts[0], dicts[1], dicts[2])

	indexed_heroes.append(synergy)
	indexed_heroes.append(counter)

	result = model.predict_proba([indexed_heroes])
	print "Radiant chance: %.3f%%" % (result[0][1] * 100)

if __name__ == "__main__":
	main()
