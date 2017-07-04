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


		if self.output_file is not None:
			csv_writer = csv.writer(self.output_file, delimiter=",")

			for match in filtered_list:
				csv_writer.writerow(match)
		else:
			return filtered_list
	
	

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

	in_file.close()
	out_file.close()

if __name__ == "__main__":
	main()
