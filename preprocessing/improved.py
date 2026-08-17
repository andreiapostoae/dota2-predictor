""" Improvements over the original preprocessing.

1. Empirical-Bayes shrunk synergy / counter matrices.
   The original divides wins by games with no regard for sample size, so a pair
   seen 3 times is trusted as much as a pair seen 30,000 times. We shrink each
   pair's winrate toward the global mean with a pseudo-count prior.

2. Temporal split helper. match_id increases monotonically with time, so a
   proper evaluation trains on older games and tests on newer ones.

3. Mirror augmentation. Swapping Radiant and Dire must flip the label, which
   both doubles the data and removes the model's ability to memorise side bias
   beyond the single global intercept.
"""
import numpy as np
from scipy import sparse


def build_pair_counts(radiant, dire, y, heroes_released):
    """ Tally co-occurrence and win counts for teammate pairs and opponent pairs.

    Returns dict with synergy/counter games and wins matrices (H, H).
    """
    H = heroes_released
    syn_g = np.zeros((H, H), dtype=np.float64)
    syn_w = np.zeros((H, H), dtype=np.float64)
    cnt_g = np.zeros((H, H), dtype=np.float64)
    cnt_w = np.zeros((H, H), dtype=np.float64)

    r = (radiant - 1).astype(np.int64)
    d = (dire - 1).astype(np.int64)
    rw = y.astype(np.float64)

    for i in range(5):
        for j in range(5):
            if i != j:
                np.add.at(syn_g, (r[:, i], r[:, j]), 1.0)
                np.add.at(syn_g, (d[:, i], d[:, j]), 1.0)
                np.add.at(syn_w, (r[:, i], r[:, j]), rw)
                np.add.at(syn_w, (d[:, i], d[:, j]), 1.0 - rw)
            np.add.at(cnt_g, (r[:, i], d[:, j]), 1.0)
            np.add.at(cnt_g, (d[:, i], r[:, j]), 1.0)
            np.add.at(cnt_w, (r[:, i], d[:, j]), rw)
            np.add.at(cnt_w, (d[:, i], r[:, j]), 1.0 - rw)

    return dict(syn_g=syn_g, syn_w=syn_w, cnt_g=cnt_g, cnt_w=cnt_w)


def shrink(wins, games, prior_strength=50.0):
    """ Empirical-Bayes shrinkage of a winrate matrix toward the global mean.

    posterior = (wins + k * global_mean) / (games + k)

    With k=50, a pair seen 3 times sits essentially at the global mean, while a
    pair seen 30,000 times is barely moved. This is the single highest-value
    statistical fix in the project.
    """
    total_w = wins.sum()
    total_g = games.sum()
    global_mean = total_w / max(total_g, 1.0)
    return (wins + prior_strength * global_mean) / (games + prior_strength)


def build_shrunk_advantages(radiant, dire, y, heroes_released, prior_strength=50.0):
    """ Full replacement for compute_advantages, with shrinkage and centring. """
    c = build_pair_counts(radiant, dire, y, heroes_released)
    syn = shrink(c['syn_w'], c['syn_g'], prior_strength)
    cnt = shrink(c['cnt_w'], c['cnt_g'], prior_strength)
    # centre on 0.5 so "no information" means zero contribution, not 0.5
    syn = (syn - 0.5).astype(np.float32)
    cnt = (cnt - 0.5).astype(np.float32)
    np.fill_diagonal(syn, 0.0)
    np.fill_diagonal(cnt, 0.0)
    return syn, cnt, c


def temporal_split(match_id, frac=0.85):
    """ Return boolean masks splitting on match_id chronology. """
    cutoff = np.quantile(match_id, frac)
    return match_id <= cutoff, match_id > cutoff


def mirror(X_onehot, adv, y, heroes_released):
    """ Swap Radiant and Dire halves, flip label, negate the counter column and
    swap the two synergy columns. Returns stacked (original + mirrored).
    """
    H = heroes_released
    left = X_onehot[:, :H]
    right = X_onehot[:, H:2 * H]
    X_m = sparse.hstack([right, left], format='csr')

    adv_m = np.column_stack([adv[:, 1], adv[:, 0], -adv[:, 2]]).astype(np.float32)
    y_m = 1 - y

    X_all = sparse.vstack([X_onehot, X_m], format='csr')
    adv_all = np.vstack([adv, adv_m])
    y_all = np.concatenate([y, y_m])
    return X_all, adv_all, y_all
