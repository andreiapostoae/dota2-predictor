""" Module responsible for meta game manipulation
There are five dictionaries: synergy_radiant, synergy_dire,
counter, radiant_winrate, dire_winrate that store statistics
"""

import pickle
import numpy as np
from preprocessing.prepare_data import NUMBER_OF_HEROES


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
	radiant_winrate = dicts[3]
	dire_winrate = dicts[4]

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

		radiant_winrate['apps'][hero_radiant_1] += 1
		dire_winrate['apps'][hero_dire_1] += 1

		if radiant_win == 1:
			radiant_winrate['wins'][hero_radiant_1] += 1
		else:
			dire_winrate['wins'][hero_dire_1] += 1


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

	radiant_winrate = dicts[3]
	dire_winrate = dicts[4]

	for i in range(NUMBER_OF_HEROES):
		for j in range(NUMBER_OF_HEROES):
			if i != j:
				for k in range(3):
					if dicts[k]['apps'][i][j] == 0.0:
						dicts[k]['winrate'][i][j] = 0.5
					else:
						dicts[k]['winrate'][i][j] = \
							dicts[k]['wins'][i][j] / float(dicts[k]['apps'][i][j])

		if radiant_winrate['apps'][i] == 0:
			radiant_winrate['winrate'][i] = 0.5
		else:
			radiant_winrate['winrate'][i] = radiant_winrate['wins'][i] / radiant_winrate['apps'][i]

		if dire_winrate['apps'][i] == 0:
			dire_winrate['winrate'][i] = 0.5
		else:
			dire_winrate['winrate'][i] = dire_winrate['wins'][i] / dire_winrate['apps'][i]


def save_dictionaries(dicts, name):
	""" Saves the list of dictionaries to a file

	dicts -- list of dictionaries
	name -- file path
	"""

	with open(name, 'wb') as file_handle:
		pickle.dump(dicts, file_handle, pickle.HIGHEST_PROTOCOL)
