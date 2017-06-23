"""This module mines from opendota public API given a list of games
Turns data as shown in mining/mining_headers.csv to data as shown in
preprocessing/ml_header.csv
"""

import csv
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab

NUMBER_OF_HEROES = 114
DEFAULT_MMR_OFFSET = 500

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
			if j > k:
				radiant += synergy_radiant[hero_radiant_1][hero_radiant_2]
				dire += synergy_dire[hero_dire_1][hero_dire_2]
			counter += counter_ratio[hero_radiant_1][hero_dire_2]
	return [radiant - dire, counter]

def initialize_dict(dictionary):
	""" Initializes the dictionary with 3 pairs of 114x114 numpy matrices

	dictionary -- dictionary to be initialized
	"""

	dictionary['apps'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))
	dictionary['wins'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))
	dictionary['winrate'] = np.zeros((NUMBER_OF_HEROES, NUMBER_OF_HEROES))

def update_dicts(heroes, radiant_win, synergy_radiant, synergy_dire, counter):
	""" Given a list of heroes, update the dictionaries when specific heroes are present

	heroes -- list of hero_ids
	synergy_radiant -- dictionary of radiant synergy winrates
	synergy_dire -- dictionary of dire synergy winrates
	counter_ratio -- dictionary of radiant heroes that counter dire heroes
	"""

	for j in range(5):
		for k in range(5):
			hero_radiant_1 = int(heroes[j]) - 1
			hero_radiant_2 = int(heroes[k]) - 1
			hero_dire_1 = int(heroes[j + 5]) - 1
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

def calculate_winrates(synergy_radiant, synergy_dire, counter):
	""" Using the number of wins and number of total games for each pair of heroes,
	calculate their winrate

	synergy_radiant -- dictionary of radiant synergy winrates
	synergy_dire -- dictionary of dire synergy winrates
	counter -- dictionary of radiant heroes that counter dire heroes
	"""

	for i in range(NUMBER_OF_HEROES):
		for j in range(NUMBER_OF_HEROES):
			if i != j:
				if synergy_radiant['apps'][i][j] == 0.0:
					synergy_radiant['winrate'][i][j] = 0
				else:
					synergy_radiant['winrate'][i][j] = \
						synergy_radiant['wins'][i][j] / synergy_radiant['apps'][i][j]

				if synergy_dire['apps'][i][j] == 0.0:
					synergy_dire['winrate'][i][j] = 0
				else:
					synergy_dire['winrate'][i][j] =  \
						synergy_dire['wins'][i][j] / synergy_dire['apps'][i][j]

				if counter['apps'][i][j] == 0.0:
					counter['winrate'][i][j] = 0
				else:
					counter['winrate'][i][j] = counter['wins'][i][j] / counter['apps'][i][j]


class DataPreprocess(object):
	""" Class that handles the preprocessing by filtering games that have MMR outside
	desired range and arranging data for the machine learning algorithm

	list_of_games -- games to be processed
	output_handle -- opened output file where the processed data is written
	mmr -- target MMR
	offset (optional) -- how far from the target MMR should the search be done
	"""

	def __init__(self, list_of_games, output_handle, mmr, offset=DEFAULT_MMR_OFFSET):
		self.games_list = list_of_games
		self.target_mmr = mmr
		self.output_file = output_handle
		self.offset_mmr = offset
		self.initialize_dicts()

	def initialize_dicts(self):
		""" Initializes the dictionaries """
		self.synergy_radiant = {}
		self.synergy_dire = {}
		self.counter = {}

		initialize_dict(self.synergy_radiant)
		initialize_dict(self.synergy_dire)
		initialize_dict(self.counter)

	def is_mmr_valid(self, mmr):
		""" Checks if a MMR is in the desired range """
		if mmr is not -1 and mmr < self.target_mmr + self.offset_mmr \
				and mmr >= self.target_mmr - self.offset_mmr:
			return True
		return False

	def run(self):
		""" Preprocessing logic """
		length = len(self.games_list)

		filtered_list = []

		for i in range(length):
			current_game = self.games_list[i]
			current_mmr = int(current_game[13])
			radiant_win = int(current_game[1])

			if self.is_mmr_valid(current_mmr):
				filtered_list.append(self.games_list[i])
				heroes = current_game[2:12]
				update_dicts(heroes, radiant_win, self.synergy_radiant, \
					self.synergy_dire, self.counter)


		calculate_winrates(self.synergy_radiant, self.synergy_dire, self.counter)

		for match in filtered_list:
			hero_list = match[2:12]
			radiant_win = match[1]

			del match[1] # delete radiant_win
			del match[12] # delete mmr
			del match[11] # delete number of shown mmrs
			del match[1:11] # remove raw hero indices
			

			match.extend(index_heroes(hero_list))

			# add synergy and counter features
			results = calculate_rating(hero_list, self.synergy_radiant['winrate'], \
						self.synergy_dire['winrate'], self.counter['winrate'])
			match.extend(results)

			# add result
			match.append(radiant_win)

		csv_writer = csv.writer(self.output_file, delimiter=",")

		for match in filtered_list:
			csv_writer.writerow(match)

	def heatmap(self, index=0, show_color=0, on_screen=1):
		""" Creates a file with 114x114 colored squares representing a heatmap of the
		dataset.

		index -- 0 for all (default)
				 1 for radiant synergy
				 2 for dire synergy
				 3 for counter
		show_color -- set to 1 for showing color bar (only works with on_screen = 1)
		on_screen -- set to 1 to print the heatmap on screen, 0 for saving as PNG
		"""

		if index == 0:
			self.heatmap(1, show_color, on_screen)
			self.heatmap(2, show_color, on_screen)
			self.heatmap(3, show_color, on_screen)
			return
		elif index == 1:
			heatmap_data = self.synergy_radiant['winrate']
			title = 'Radiant synergy heatmap'
			
		elif index == 2:
			heatmap_data = self.synergy_dire['winrate']
			title = 'Dire synergy heatmap'
		else:
			heatmap_data = self.counter['winrate']
			title = 'Counter heatmap'

		fig = plt.figure(figsize=(15, 15))

		axes = fig.add_subplot(111)
		axes.set_title(title)
		plt.imshow(heatmap_data)
		axes.set_aspect('equal')

		if(show_color == 1):
			color_axes = fig.add_axes([0.12, 0.1, 0.78, 0.8])
			color_axes.get_xaxis().set_visible(False)
			color_axes.get_yaxis().set_visible(False)
			color_axes.patch.set_alpha(0)
			color_axes.set_frame_on(False)

		if on_screen == 0:
			filename = "heatmap" + str(index) + ".png"
			pylab.savefig(filename, bbox_inches='tight')
		else:
			plt.colorbar(orientation='vertical')
			plt.show()

def main():
	""" Main method """
	if len(sys.argv) < 3:
		sys.exit("Usage: %s input_file output_file MMR [offset]" % sys.argv[0])

	try:
		in_file = open(sys.argv[1], "rt")
	except IOError:
		sys.exit("Invalid input file")

	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)

	try:
		out_file = open(sys.argv[2], "wt")
	except IOError:
		sys.exit("Invalid output file")

	try:
		mmr = int(sys.argv[3])
	except ValueError:
		sys.exit("Invalid MMR")

	if mmr < 0 or mmr > 5000:
		sys.exit("Invalid MMR")

	try:
		offset = int(sys.argv[4])
		data_preprocess = DataPreprocess(full_list, out_file, mmr, offset)
	except ValueError:
		print "The offset is invalid. Using the default offset (%d)" % DEFAULT_MMR_OFFSET
		data_preprocess = DataPreprocess(full_list, out_file, mmr)

	data_preprocess.run()
	data_preprocess.heatmap()

	in_file.close()
	out_file.close()

if __name__ == "__main__":
	main()
