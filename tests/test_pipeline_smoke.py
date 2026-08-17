""" Slow end-to-end tests against the real dataset.

Skipped automatically when 706e.zip is absent (e.g. in CI without the data).
Run explicitly with:  pytest tests/ -m slow -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_DATA = os.path.exists(os.path.join(ROOT, '706e.zip'))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="706e.zip not present")


@pytest.fixture(scope='module')
def data():
    from scripts.common import load_temporal
    return load_temporal()


@needs_data
@pytest.mark.slow
def test_dataset_shape_and_balance(data):
    r, d, y, mid, tr, te = data
    assert r.shape[1] == 5 and d.shape[1] == 5
    assert r.shape[0] == d.shape[0] == y.shape[0]
    assert r.shape[0] > 1_500_000, "expected ~1.9M games"
    assert 0.50 < y.mean() < 0.56, "radiant winrate should be slightly above 0.5"


@needs_data
@pytest.mark.slow
def test_no_duplicate_heroes_within_a_match(data):
    r, d, _, _, _, _ = data
    both = np.hstack([r[:5000], d[:5000]])
    for row in both:
        assert len(set(row.tolist())) == 10, "a hero appears twice in one match"


@needs_data
@pytest.mark.slow
def test_hero_ids_are_in_range(data):
    r, d, _, _, _, _ = data
    from scripts.common import heroes_released
    H = heroes_released()
    assert r.min() >= 1 and d.min() >= 1
    assert r.max() <= H and d.max() <= H


@needs_data
@pytest.mark.slow
def test_temporal_split_is_leak_free(data):
    _, _, _, mid, tr, te = data
    assert mid[tr].max() <= mid[te].min()
    assert tr.sum() + te.sum() == mid.shape[0]


@needs_data
@pytest.mark.slow
def test_baseline_reaches_expected_auc(data):
    """ Regression guard: the documented baseline is 0.6504. """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scripts.common import train_only_advantages, heroes_released
    from preprocessing.fast_features import build_onehot, advantage_columns

    r, d, y, mid, tr, te = data
    H = heroes_released()
    syn, cnt = train_only_advantages(r, d, y, tr, H)[:2]
    adv = advantage_columns(r, d, syn, cnt)
    mu, sd = adv[tr].mean(0), adv[tr].std(0)
    X = sparse.hstack([build_onehot(r, d, H),
                       sparse.csr_matrix((adv - mu) / sd)], format='csr')
    m = LogisticRegression(C=1.0, random_state=42, max_iter=300).fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    assert auc == pytest.approx(0.6504, abs=0.004), "baseline AUC drifted: %.4f" % auc
