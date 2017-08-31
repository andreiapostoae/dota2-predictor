import pandas as pd
import numpy as np
import logging
from advantages import compute_advantages
from tools.metadata import get_last_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dataset_to_features(dataset_df):
    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']

    feature_map = np.zeros((len(dataset_df), 2 * heroes_released + 1))

    dataset_np = dataset_df.values

    for i, row in enumerate(dataset_np):
        radiant_win = row[1]
        radiant_heroes = map(int, row[2].split(','))
        dire_heroes = map(int, row[3].split(','))

        for j in range(5):
            feature_map[i, radiant_heroes[j] - 1] = 1
            feature_map[i, dire_heroes[j] - 1 + heroes_released] = 1

        feature_map[i, -1] = 1 if radiant_win else 0

    return feature_map


def read_dataset(csv_path,
                 low_mmr=None,
                 high_mmr=None,
                 advantages=False):
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
        advantages_list = None

    logger.info("Transforming dataframe in feature map...")
    feature_map = _dataset_to_features(dataset_df)

    return [feature_map, advantages_list]
