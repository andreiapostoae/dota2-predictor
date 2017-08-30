import pandas as pd
import numpy as np
import logging
from advantages import compute_advantages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_dataset(csv_path,
                 low_mmr=None,
                 high_mmr=None,
                 advantages=False):
    dataset_df = pd.read_csv(csv_path)

    if low_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr > low_mmr]

    if high_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr < high_mmr]

    logger.info("The dataset contains %d games", len(dataset_df))

    if advantages:
        advantages_list = compute_advantages(dataset_df)
        return dataset_df, advantages_list

    return dataset_df
