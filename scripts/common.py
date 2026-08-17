""" Shared data loading for the reproduction scripts.

Handles: extracting the bundled dataset, parsing hero ids, building the temporal
split, and caching intermediates so later scripts start instantly.
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data_raw')
CACHE_DIR = os.path.join(ROOT, 'artifacts', 'cache')
ZIP_PATH = os.path.join(ROOT, '706e.zip')
TRAIN_CSV = os.path.join(DATA_DIR, '706e_train_dataset.csv')
TEST_CSV = os.path.join(DATA_DIR, '706e_test_dataset.csv')
SPLIT_CACHE = os.path.join(CACHE_DIR, 'temporal.npz')


def ensure_extracted():
    """ Unzip the bundled dataset on first use. """
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        return
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            "706e.zip not found. It ships with the upstream repo; "
            "clone with the archive present or download it from upstream.")
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Extracting %s ..." % ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)


def hero_names():
    """ Map hero id (1-based, as in the data) to display name. """
    meta = json.load(open(os.path.join(ROOT, 'metadata.json')))
    return {h['id']: h['name'] for h in meta['heroes']}


def heroes_released():
    meta = json.load(open(os.path.join(ROOT, 'metadata.json')))
    return meta['patches'][0]['heroes_released']


def load_temporal(frac=0.85, rebuild=False):
    """ Load both CSVs, concatenate, sort by match_id, and split chronologically.

    match_id increases monotonically with time, so this gives an honest
    evaluation: train on older games, test on newer ones. The split shipped with
    the upstream repo is random, and 100% of its test match ids fall inside the
    training id range.

    Returns:
        radiant (n,5) int16 1-based, dire (n,5), y (n,) float64,
        match_id (n,) int64, train_mask (n,) bool, test_mask (n,) bool
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(SPLIT_CACHE) and not rebuild:
        z = np.load(SPLIT_CACHE)
        return (z['r'], z['d'], z['y'], z['mid'], z['tr_mask'], z['te_mask'])

    ensure_extracted()
    print("Reading CSVs ...")
    frames = [pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)]
    df = pd.concat(frames, ignore_index=True)

    r = df['radiant_team'].str.split(',', expand=True).to_numpy(dtype=np.int16)
    d = df['dire_team'].str.split(',', expand=True).to_numpy(dtype=np.int16)
    y = df['radiant_win'].to_numpy()
    y = (y == True).astype(np.float64) if y.dtype != np.float64 else y
    mid = df['match_id'].to_numpy(dtype=np.int64)

    order = np.argsort(mid)
    r, d, y, mid = r[order], d[order], y[order], mid[order]

    cutoff = np.quantile(mid, frac)
    tr_mask, te_mask = mid <= cutoff, mid > cutoff

    print("Temporal split: %d train / %d test (cutoff match_id %d)"
          % (tr_mask.sum(), te_mask.sum(), int(cutoff)))
    np.savez_compressed(SPLIT_CACHE, r=r, d=d, y=y, mid=mid,
                        tr_mask=tr_mask, te_mask=te_mask)
    return r, d, y, mid, tr_mask, te_mask


def train_only_advantages(r, d, y, tr_mask, H, prior_strength=0.0):
    """ Synergy and counter matrices computed on TRAINING GAMES ONLY.

    The shipped pretrained/synergies_all.csv was computed over the full dataset
    including test games, which inflates results by roughly 0.013 AUC.

    prior_strength enables empirical-Bayes shrinkage. It is 0 by default because
    at this data volume it makes no measurable difference -- see FINDINGS.md.
    """
    from preprocessing.improved import build_pair_counts, shrink
    c = build_pair_counts(r[tr_mask], d[tr_mask], y[tr_mask], H)
    if prior_strength > 0:
        syn = shrink(c['syn_w'], c['syn_g'], prior_strength)
        cnt = shrink(c['cnt_w'], c['cnt_g'], prior_strength)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            syn = np.nan_to_num(c['syn_w'] / c['syn_g'])
            cnt = np.nan_to_num(c['cnt_w'] / c['cnt_g'])
    syn = (syn - 0.5).astype(np.float32)
    cnt = (cnt - 0.5).astype(np.float32)
    np.fill_diagonal(syn, 0.0)
    np.fill_diagonal(cnt, 0.0)
    return syn, cnt, c
