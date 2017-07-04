"""This module mines from opendota public API given a list of games
Turns data as shown in mining/mining_headers.csv to data as shown in
preprocessing/ml_header.csv
"""

import csv
import sys
import json
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from matplotlib import pylab

NUMBER_OF_HEROES = 114
DEFAULT_MMR_OFFSET = 500
TEST_RATIO = 0.25




class DataPreprocess(object):
	""" Class that handles the preprocessing by filtering games that have MMR outside
	desired range and arranging data for the machine learning algorithm

	list_of_games -- games to be processed
	output_handle -- opened output file where the processed data is written
	mmr -- target MMR
	offset (optional) -- how far from the target MMR should the search be done
	"""

	def __init__(self, list_of_games, mmr, offset=DEFAULT_MMR_OFFSET, output_handle=None):
		self.games_list = list_of_games
		self.target_mmr = mmr
		self.output_file = output_handle
		self.offset_mmr = offset

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
				filtered_list.append(current_game[1:12])

		'''new_length = int(len(filtered_list) * 0.75 // 1) + 1

		for k in range(new_length):
			current_game = filtered_list[k]
			radiant_win = int(current_game[1])
			heroes = current_game[2:12]
			update_dicts(heroes, radiant_win, self.dicts)

		calculate_synergy_winrates(self.dicts)

		for match in filtered_list:
			hero_list = match[2:12]
			radiant_win = match[1]

			del match[1] # delete radiant_win
			del match[12] # delete mmr
			del match[11] # delete number of shown mmrs
			del match[1:11] # remove raw hero indices

			match.extend(index_heroes(hero_list))

			# add synergy and counter features
			results = calculate_rating(hero_list, self.dicts[0]['winrate'], \
						self.dicts[1]['winrate'], self.dicts[2]['winrate'])
			match.extend(results)

			# add result
			match.append(int(radiant_win))
		'''
		if self.output_file is not None:
			csv_writer = csv.writer(self.output_file, delimiter=",")

			for match in filtered_list:
				csv_writer.writerow(match)
		else:
			return filtered_list
	
	'''
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
			heatmap_data = self.dicts[0]['winrate']
			title = 'Radiant synergy heatmap'

		elif index == 2:
			heatmap_data = self.dicts[1]['winrate']
			title = 'Dire synergy heatmap'
		else:
			heatmap_data = self.dicts[2]['winrate']
			title = 'Counter heatmap'

		fig = plt.figure(figsize=(15, 15))

		axes = fig.add_subplot(111)
		axes.set_title(title)
		plt.imshow(heatmap_data)
		axes.set_aspect('equal')

		if show_color == 1:
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
	'''
	'''
	def hero_winrates(self, radiant=1):
		""" Calculates the hero winrates over the filtered games

		radiant (optional) -- 1 for radiant games (default)
							  0 for dire games
		"""

		hero_map = get_hero_names()
		heroes_dict = {}
		hero_list = []

		for i in range(114):
			if i != 23:
				hero_list.append(i + 1)

				if radiant == 1:
					heroes_dict[self.dicts[3]['winrate'][i]] = hero_map[i + 1]
				else:
					heroes_dict[self.dicts[4]['winrate'][i]] = hero_map[i + 1]

		keys = heroes_dict.keys()
		keys.sort(reverse=True)

		fig = plt.figure(figsize=(20, 20))
		axes = fig.add_subplot(111)

		axes.set_ylim([0, 115])
		axes.set_xlim([0, 1])
		axes.invert_yaxis()

		sorted_values = []
		for key in keys:
			sorted_values.append(heroes_dict.get(key))

		plt.yticks(hero_list, sorted_values, size=5)

		rects = axes.barh(hero_list, keys, height=0.8)

		for rect in rects:
			width = rect.get_width()
			axes.text(width * 1.03, rect.get_y() + rect.get_height()/2. + 0.75, \
				'%.2f%%' % (width * 100), ha='center', va='bottom', size=6)

		plt.title('Radiant winrate at %d - %d MMR' % (self.target_mmr - self.offset_mmr, \
			self.target_mmr + self.offset_mmr), size=20)
		plt.show()
	'''

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
		data_preprocess = DataPreprocess(full_list, mmr, offset, out_file)
	except IndexError:
		print "The offset is invalid. Using the default offset (%d)" % DEFAULT_MMR_OFFSET
		data_preprocess = DataPreprocess(full_list, mmr, out_file)

	data_preprocess.run()
	#data_preprocess.hero_winrates()

	in_file.close()
	out_file.close()

if __name__ == "__main__":
	main()
