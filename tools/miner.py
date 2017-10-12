""" Module responsible with mining games from Opendota """

import json
import logging
import pandas as pd
import ssl
import time
import urllib2

from metadata import get_last_patch
from pandas.io.json import json_normalize


OPENDOTA_URL = "https://api.opendota.com/api/publicMatches?less_than_match_id="
REQUEST_TIMEOUT = 0.3
COLUMNS = ['match_id', 'radiant_win', 'radiant_team', 'dire_team', 'avg_mmr', 'num_mmr',
           'game_mode', 'lobby_type']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

last_patch_dict = get_last_patch()
first_match = last_patch_dict['first_match_id']
last_match = last_patch_dict['last_match_id']


def mine_data(file_name=None,
              first_match_id=first_match,
              last_match_id=last_match,
              stop_at=None,
              timeout=15,
              save_every=1000):
    """ Mine data using the official Opendota API. Keep requests at a decent rate (3/s).
    For every request, a JSON containing 100 games is returned. The games are downloaded
    in descending order of the match IDs.

    Args:
        file_name: the name of the file where the dataframe will be stored
        first_match_id: lowest match ID to look at; currently set at the start of 7.06e
        last_match_id: highest match ID to look at; currently start at the end of 7.06e
        stop_at: when the dataframe contains stop_at games, the mining stops
        timeout: in case Opendota does not respond, wait timeout seconds before retrying
        save_every: save the dataframe every save_every entries

    Returns:
        dataframe with the mined games
    """
    global OPENDOTA_URL
    global REQUEST_TIMEOUT
    global COLUMNS
    global logger

    results_dataframe = pd.DataFrame()
    current_chunk = 1
    current_match_id = last_match_id
    games_remaining = stop_at

    while current_match_id > first_match_id:
        try:
            current_link = OPENDOTA_URL + str(current_match_id)
            logger.info("Mining chunk starting at match ID %d", current_match_id)
            response = urllib2.urlopen(current_link, timeout=timeout)
        except (urllib2.URLError, ssl.SSLError) as error:
            logger.error("Failed to make a request starting at match ID %d", current_match_id)
            logger.info("Waiting %d seconds before retrying", timeout)
            time.sleep(timeout)
            current_match_id -= 1
            continue

        try:
            response_json = json.load(response)
            last_match_id = response_json[-1]['match_id']
        except (ValueError, KeyError) as error:
            logger.error("Corrupt JSON starting at match ID %d, skipping it", current_match_id)
            current_match_id -= 1
            continue

        current_match_id = last_match_id

        if games_remaining:
            games_remaining -= len(response_json)

        current_dataframe = json_normalize(response_json)

        if len(current_dataframe) == 0:
            logger.info("Found an empty dataframe, skipping 10 games")
            current_match_id -= 10
            continue

        results_dataframe = results_dataframe.append(current_dataframe, ignore_index=True)

        if len(results_dataframe) >= current_chunk * save_every:
            current_chunk += 1

            if file_name:
                pd.DataFrame(results_dataframe, columns=COLUMNS).to_csv(file_name, index=False)
                logger.info("Saving to csv. Total of games mined: %d", len(results_dataframe))

                if stop_at:
                    if len(results_dataframe) >= stop_at:
                        return results_dataframe

        if stop_at:
            if len(results_dataframe) >= stop_at:
                break

        time.sleep(REQUEST_TIMEOUT)

    return results_dataframe
