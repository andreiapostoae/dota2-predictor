""" This module mines relevant match IDs from the Steam API """
import json
import sys
import urllib
import os

STEAM_BASE_URL = \
	"https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v0001/?key="
SEQ_STRING = "&start_at_match_seq_num="
REQUESTS_STRING = "&matches_requested="
MIN_DURATION = 1200

# this should change every time a new patch is released
STARTING_MATCH_SEQ_NUM = 2833500000

def valid(match_json):
	""" checks if a game is valid: at least 20 mins long, all-pick ranked mode, no leavers

	match_json -- dictionary of a single match taken from the response json
	"""
	for player in match_json['players']:
		try:
			if player['leaver_status'] == 1:
				return False
		except KeyError:
			return False

	if match_json['duration'] < MIN_DURATION:
		return False

	if match_json['human_players'] != 10:
		return False

	if match_json['game_mode'] != 22:
		return False

	return True

class SteamMiner(object):
	""" Sends HTTP requests to Steam, parses the JSON that comes as a
	response and saves the relevant match IDs in a file

	Keyword arguments:
	number_of_games -- how many games should be processed
	out_file_handle -- handle of the file where the match IDs are written
	key -- Valve API key
	"""

	def __init__(self, number_of_games, out_file_handle, key):
		self.games_number = number_of_games
		self.out_file = out_file_handle
		self.api_key = key
		self.seq_num = STARTING_MATCH_SEQ_NUM

	def get_url(self, matches_requested):
		""" Concatenates the request into an URL

		matches_requested -- number of games requested (100 or lesser)
		"""
		return STEAM_BASE_URL + self.api_key + SEQ_STRING + str(self.seq_num) + \
				REQUESTS_STRING + str(matches_requested)

	def get_response(self, url_api):
		""" Recursive method that tries getting a response from Steam until the response
		is valid; The invalid response are probably caused by a rate limit that Steam has
		which is not specified, so the request are sent continuously

		url_api -- the URL where the HTTP request is sent
		games_json -- JSON with valid response
		"""

		response = urllib.urlopen(url_api)
		games_json = json.load(response)

		if 'result' not in games_json:
			return self.get_response(url_api)

		return games_json

	def run(self):
		""" Schedules the HTTP request, considering the fact that one request can handle
		up to 100 matches requested, so the process is done in chunks
		"""

		chunks = self.games_number / 100
		remainder = self.games_number - chunks * 100

		for i in range(chunks + 1):
			if i == chunks:
				url = self.get_url(remainder)
			else:
				url = self.get_url(100)

			response_json = self.get_response(url)

			for match in response_json['result']['matches']:
				match_id = match['match_id']
				if valid(match):
					self.out_file.write(str(match_id) + "\n")
				self.seq_num = match['match_seq_num'] + 1

def main():
	""" Main function """

	try:
		api_key = os.environ['API_KEY']
	except KeyError:
		sys.exit("Please set API_KEY environment variable.")

	if len(sys.argv) < 3:
		sys.exit("Usage: %s <output_file> <number_of_games>" % sys.argv[0])

	try:
		out_file = open(sys.argv[1], "wt")
	except IOError:
		sys.exit("Invalid output file")

	try:
		games_number = int(sys.argv[2])
	except ValueError:
		sys.exit("Invalid number of games")

	miner = SteamMiner(games_number, out_file, api_key)
	miner.run()

	out_file.close()


if __name__ == "__main__":
	main()
