import logging
import numpy as np
import pandas as pd

from advantages import compute_advantages
from augmenter import augment_with_advantages
from tools.metadata import get_last_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dataset_to_features(dataset_df, advantages=None):
    """ Transforms a mined pandas DataFrame into a feature matrix. This method assumes the following
    format of a DataFrame:
    columns: [match_id,radiant_win,radiant_team,dire_team,avg_mmr,num_mmr,game_mode,lobby_type]

    Args:
        dataset_df: pandas DataFrame to be transformed
        advantages: if given, the synergy and counters matrix are used to compute synergy and
            counter rating for every game which are appended to the feature matrix
    Returns:
        [X, y], where X is feature matrix, y are outputs
    """
    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']
    synergy_matrix, counter_matrix = None, None

    if advantages:
        x_matrix = np.zeros((dataset_df.shape[0], 2 * heroes_released + 3))
        [synergy_matrix, counter_matrix] = advantages
    else:
        x_matrix = np.zeros((dataset_df.shape[0], 2 * heroes_released))

    y_matrix = np.zeros(dataset_df.shape[0])

    dataset_np = dataset_df.values

    for i, row in enumerate(dataset_np):
        radiant_win = row[1]
        radiant_heroes = map(int, row[2].split(','))
        dire_heroes = map(int, row[3].split(','))

        for j in range(5):
            x_matrix[i, radiant_heroes[j] - 1] = 1
            x_matrix[i, dire_heroes[j] - 1 + heroes_released] = 1

            if advantages:
                x_matrix[i, -3:] = augment_with_advantages(synergy_matrix,
                                                           counter_matrix,
                                                           radiant_heroes,
                                                           dire_heroes)

        y_matrix[i] = 1 if radiant_win else 0

    return [x_matrix, y_matrix]


def read_dataset(csv_path,
                 low_mmr=None,
                 high_mmr=None,
                 advantages=False):
    """ Reads pandas DataFrame from csv_path, filters games between low_mmr and high_mmr if given
    and appends synergy and counter features

    Args:
        csv_path: path to read pandas DataFrame from
        low_mmr: lower MMR bound
        high_mmr: higher MMR bound
        advantages: if True, advantages are recalculated and saved to files, else it is read from
            they are read from files
    Returns:
        [feature_matrix, [synergy_matrix, counter_matrix]]
    """
    global logger
    dataset_df = pd.read_csv(csv_path)

    if low_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr > low_mmr]

    if high_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr < high_mmr]

    logger.info("The dataset contains %d games", len(dataset_df))

    if advantages:
        logger.info("Computing advantages...")
        advantages_list = compute_advantages(dataset_df)
    else:
        logger.info("Loading advantages from files...")
        synergies = np.loadtxt('pretrained/synergies_all.csv')
        counters = np.loadtxt('pretrained/counters_all.csv')
        advantages_list = [synergies, counters]

    logger.info("Transforming dataframe in feature map...")
    feature_map = _dataset_to_features(dataset_df, advantages=advantages_list)

    return [feature_map, advantages_list]
