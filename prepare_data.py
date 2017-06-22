import csv
import numpy as np
import sys

def calculate_rating(heroes, synergy_winrate_radiant, synergy_winrate_dire, counter_winrate):
	counter = 0
	synergy_radiant = 0
	synergy_dire = 0

	for j in range(5):
		for k in range(5):
			hero_radiant_1 = int(heroes[j]) - 1
			hero_radiant_2 = int(heroes[k]) - 1
			hero_dire_1 = int(heroes[j + 5]) - 1
			hero_dire_2 = int(heroes[k + 5]) - 1
			if j > k:
				synergy_radiant += synergy_winrate_radiant[hero_radiant_1][hero_radiant_2]
				synergy_dire += synergy_winrate_dire[hero_dire_1][hero_dire_2]
			counter += counter_winrate[hero_radiant_1][hero_dire_2]
	return [synergy_radiant - synergy_dire, counter]


class DataPreprocess(object):
	def __init__(self, list_of_games, mmr):
		self.games_list = list_of_games
		self.target_mmr = mmr

	def run(self):
		length = len(self.games_list)

		synergy_appearances_radiant = np.zeros((114, 114))
		synergy_wins_radiant = np.zeros((114, 114))
		synergy_winrate_radiant = np.zeros((114, 114))

		synergy_appearances_dire = np.zeros((114, 114))
		synergy_wins_dire = np.zeros((114, 114))
		synergy_winrate_dire = np.zeros((114, 114))

		counter_appearances = np.zeros((114, 114))
		counter_wins = np.zeros((114, 114))
		counter_winrate = np.zeros((114, 114))

		filtered_list = []

		for i in range(length):
			current_game = self.games_list[i]
			current_mmr = int(current_game[13])
			radiant_win = int(current_game[1])

			if current_mmr is not -1 and current_mmr < self.target_mmr + 500 \
				and current_mmr >= self.target_mmr - 500:
				filtered_list.append(self.games_list[i])
				for j in range(5):
					for k in range(5):
						hero_radiant_1 = int(current_game[j + 2]) - 1
						hero_radiant_2 = int(current_game[k + 2]) - 1
						hero_dire_1 = int(current_game[j + 7]) - 1
						hero_dire_2 = int(current_game[k + 7]) - 1

						if j != k:
							synergy_appearances_radiant[hero_radiant_1][hero_radiant_2] += 1
							synergy_appearances_dire[hero_dire_1][hero_dire_2] += 1

							if radiant_win == 1:
								synergy_wins_radiant[hero_radiant_1][hero_radiant_2] += 1	
							else:
								synergy_wins_dire[hero_dire_1][hero_dire_2] += 1
						
						counter_appearances[hero_radiant_1][hero_dire_2] += 1
						if radiant_win == 1:
							counter_wins[hero_radiant_1][hero_dire_2] += 1

		for i in range(114):
			for j in range(114):
				if i != j:
					if synergy_appearances_radiant[i][j] == 0.0:
						synergy_winrate_radiant[i][j] = 0
					else:
						synergy_winrate_radiant[i][j] = synergy_wins_radiant[i][j] / synergy_appearances_radiant[i][j]

					if synergy_appearances_dire[i][j] == 0.0:
						synergy_winrate_dire[i][j] = 0
					else:
						synergy_winrate_dire[i][j] = synergy_wins_dire[i][j] / synergy_appearances_dire[i][j]

					if counter_appearances[i][j] == 0.0:
						counter_winrate[i][j] = 0
					else:
						counter_winrate[i][j] = counter_wins[i][j] / counter_appearances[i][j]

		for match in filtered_list:
			hero_list = match[1:11]
			results = calculate_rating(hero_list, synergy_winrate_radiant, synergy_winrate_dire, counter_winrate)
			match.extend(results)

		f = open("test.csv", "wt")
		csv_wr = csv.writer(f, delimiter=",")

		for match in filtered_list:
			csv_wr.writerow(match)







def main():
	if len(sys.argv) < 3:
		sys.exit("Usage: %s <input_file> <MMR>" % sys.argv[0])

	try:
		in_file = open(sys.argv[1], "rt")
	except IOError:
		sys.exit("Invalid input file")


	try:
		mmr = int(sys.argv[2])
	except ValueError:
		sys.exit("Invalid MMR")

	if mmr < 0 or mmr > 5000:
		sys.exit("Invalid MMR")

	csv_reader = csv.reader(in_file, delimiter=",")
	full_list = list(csv_reader)

	data_preprocess = DataPreprocess(full_list, mmr)
	data_preprocess.run()

if __name__ == "__main__":
	main()
