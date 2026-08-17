""" Vectorised, memory-efficient replacement for the original dense feature builder.

The original _dataset_to_features allocates a dense (n_games, 2*H+3) float64 array
and fills it with a Python loop. At 1.8M games that is ~3.3 GB and several minutes.
Here we parse hero ids once with a vectorised split and emit a scipy CSR matrix.
"""
import numpy as np
import pandas as pd
from scipy import sparse


def parse_teams(dataset_df):
    """ Parse the comma-separated hero id strings into two (n, 5) int arrays.

    Returns:
        radiant (n,5) int16, dire (n,5) int16, y (n,) int8, match_id (n,) int64, mmr (n,) int32
    """
    radiant = dataset_df['radiant_team'].str.split(',', expand=True).to_numpy(dtype=np.int16)
    dire = dataset_df['dire_team'].str.split(',', expand=True).to_numpy(dtype=np.int16)
    y = dataset_df['radiant_win'].to_numpy()
    if y.dtype != np.int8:
        y = (y == True).astype(np.int8) if y.dtype == object or y.dtype == bool else y.astype(np.int8)
    match_id = dataset_df['match_id'].to_numpy(dtype=np.int64)
    mmr = dataset_df['avg_mmr'].to_numpy(dtype=np.int32)
    return radiant, dire, y, match_id, mmr


def build_onehot(radiant, dire, heroes_released):
    """ Build the sparse (n, 2*H) one-hot hero indicator matrix. """
    n = radiant.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int32), 10)
    cols = np.concatenate([
        (radiant - 1).astype(np.int32),
        (dire - 1).astype(np.int32) + heroes_released
    ], axis=1).ravel()
    data = np.ones(n * 10, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, 2 * heroes_released), dtype=np.float32)


def advantage_columns(radiant, dire, synergy, counter):
    """ Vectorised version of augment_with_advantages over the whole dataset.

    Returns (n, 3) float32: radiant synergy, dire synergy, radiant-vs-dire counter total.
    """
    r = (radiant - 1).astype(np.int32)
    d = (dire - 1).astype(np.int32)

    syn_r = np.zeros(r.shape[0], dtype=np.float32)
    syn_d = np.zeros(r.shape[0], dtype=np.float32)
    for i in range(5):
        for j in range(i):
            syn_r += synergy[r[:, i], r[:, j]]
            syn_d += synergy[d[:, i], d[:, j]]

    cnt = np.zeros(r.shape[0], dtype=np.float32)
    for i in range(5):
        for j in range(5):
            cnt += counter[r[:, i], d[:, j]]

    return np.column_stack([syn_r, syn_d, cnt]).astype(np.float32)


def build_features(dataset_df, synergy, counter, heroes_released, with_advantages=True):
    """ Full feature build: sparse one-hot horizontally stacked with dense advantage columns. """
    radiant, dire, y, match_id, mmr = parse_teams(dataset_df)
    X = build_onehot(radiant, dire, heroes_released)
    if with_advantages:
        adv = advantage_columns(radiant, dire, synergy, counter)
        X = sparse.hstack([X, sparse.csr_matrix(adv)], format='csr')
    return X, y, match_id, mmr, radiant, dire
