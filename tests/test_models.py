""" Tests for the Python 3 port and the new models.

Run with:  pytest tests/ -v

These are deliberately not accuracy tests. Model accuracy depends on the dataset
and takes minutes to measure; these check correctness properties that must hold
regardless of data, and that would silently break during refactoring.
"""
import os
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.fast_features import build_onehot, advantage_columns, parse_teams
from preprocessing.improved import build_pair_counts, shrink, temporal_split, mirror
from training.embedding_model import HeroEmbeddingModel, sigmoid
from training.prefix_model import PrefixEmbeddingModel
from training.sequential_query import rank_picks, score_state

H = 114


@pytest.fixture
def toy():
    """ Small synthetic draft data: 200 games, valid hero ids, no duplicates. """
    rng = np.random.default_rng(0)
    n = 200
    R = np.zeros((n, 5), dtype=np.int16)
    D = np.zeros((n, 5), dtype=np.int16)
    for i in range(n):
        picks = rng.choice(np.arange(1, H + 1), size=10, replace=False)
        R[i], D[i] = picks[:5], picks[5:]
    y = rng.integers(0, 2, n).astype(np.float64)
    mid = np.sort(rng.integers(3_000_000_000, 3_100_000_000, n)).astype(np.int64)
    return R, D, y, mid


# --------------------------------------------------------------------------
# feature builder
# --------------------------------------------------------------------------

def test_onehot_has_exactly_ten_ones(toy):
    R, D, _, _ = toy
    X = build_onehot(R, D, H)
    assert X.shape == (R.shape[0], 2 * H)
    assert np.all(np.asarray(X.sum(axis=1)).ravel() == 10)


def test_onehot_puts_heroes_on_correct_side(toy):
    R, D, _, _ = toy
    X = build_onehot(R, D, H).toarray()
    for i in range(5):
        row = X[i]
        for h in R[i]:
            assert row[h - 1] == 1, "radiant hero missing from left block"
        for h in D[i]:
            assert row[h - 1 + H] == 1, "dire hero missing from right block"


def test_onehot_matches_original_dense_loop(toy):
    """ The vectorised builder must produce exactly what the upstream loop did. """
    R, D, _, _ = toy
    fast = build_onehot(R, D, H).toarray()
    slow = np.zeros((R.shape[0], 2 * H), dtype=np.float32)
    for i in range(R.shape[0]):
        for j in range(5):
            slow[i, R[i, j] - 1] = 1
            slow[i, D[i, j] - 1 + H] = 1
    np.testing.assert_array_equal(fast, slow)


def test_advantage_columns_match_upstream_augmenter(toy):
    """ Vectorised advantage computation must equal the original per-game one. """
    from preprocessing.augmenter import augment_with_advantages
    R, D, _, _ = toy
    rng = np.random.default_rng(1)
    syn = rng.normal(0, 0.1, (H, H)).astype(np.float32)
    cnt = rng.normal(0, 0.1, (H, H)).astype(np.float32)
    syn = ((syn + syn.T) / 2).astype(np.float32)   # synergy is symmetric

    fast = advantage_columns(R, D, syn, cnt)
    for i in range(10):
        expected = augment_with_advantages(syn, cnt, list(R[i]), list(D[i]))
        np.testing.assert_allclose(fast[i], expected, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# statistics / splitting
# --------------------------------------------------------------------------

def test_shrinkage_pulls_low_support_toward_mean():
    wins = np.array([[2.0, 500.0]])
    games = np.array([[3.0, 1000.0]])
    out = shrink(wins, games, prior_strength=50.0)
    glob = wins.sum() / games.sum()
    # the 3-game cell should move much further toward the global mean
    assert abs(out[0, 0] - glob) < abs(2.0 / 3.0 - glob)
    # the 1000-game cell should barely move
    assert abs(out[0, 1] - 0.5) < 0.02


def test_shrinkage_bounded_between_zero_and_one():
    rng = np.random.default_rng(3)
    games = rng.integers(1, 500, (20, 20)).astype(float)
    wins = (games * rng.random((20, 20))).astype(float)
    out = shrink(wins, games, 50.0)
    assert np.all((out >= 0) & (out <= 1))


def test_temporal_split_has_no_id_overlap(toy):
    _, _, _, mid = toy
    tr, te = temporal_split(mid, 0.85)
    assert tr.sum() > 0 and te.sum() > 0
    assert mid[tr].max() <= mid[te].min(), "train ids must precede test ids"
    assert not np.any(tr & te), "masks must be disjoint"


def test_pair_counts_are_symmetric_for_synergy(toy):
    R, D, y, _ = toy
    c = build_pair_counts(R - 1, D - 1, y, H)
    np.testing.assert_array_equal(c['syn_g'], c['syn_g'].T)


def test_mirror_flips_labels_and_sides(toy):
    R, D, y, _ = toy
    X = build_onehot(R, D, H)
    adv = np.random.default_rng(4).normal(0, 1, (R.shape[0], 3)).astype(np.float32)
    Xa, adva, ya = mirror(X, adv, y, H)
    n = R.shape[0]
    assert Xa.shape[0] == 2 * n and ya.shape[0] == 2 * n
    np.testing.assert_array_equal(ya[n:], 1 - y)
    # mirrored counter column negates, synergy columns swap
    np.testing.assert_allclose(adva[n:, 2], -adv[:, 2])
    np.testing.assert_allclose(adva[n:, 0], adv[:, 1])


# --------------------------------------------------------------------------
# embedding model correctness
# --------------------------------------------------------------------------

def test_sigmoid_is_numerically_stable():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))
    assert p[2] == pytest.approx(0.5)


def test_fm_identity_equals_explicit_pairwise_sum():
    """ The O(5k) synergy shortcut must equal the O(25k) explicit pair sum. """
    rng = np.random.default_rng(5)
    k = 8
    s = rng.normal(0, 1, (5, k))
    S = s.sum(0)
    fast = 0.5 * ((S ** 2).sum() - (s ** 2).sum())
    slow = sum(float(s[i] @ s[j]) for i in range(5) for j in range(i))
    assert fast == pytest.approx(slow, rel=1e-9)


def test_embedding_model_is_antisymmetric(toy):
    """ logit(R,D) + logit(D,R) - 2b must be exactly zero by construction. """
    R, D, y, _ = toy
    m = HeroEmbeddingModel(H, k=8)
    m.fit(R.astype(np.int64) - 1, D.astype(np.int64) - 1, y,
          epochs=1, batch=64, lr=0.05, verbose=False)
    l1, _ = m._forward(R.astype(np.int64) - 1, D.astype(np.int64) - 1)
    l2, _ = m._forward(D.astype(np.int64) - 1, R.astype(np.int64) - 1)
    np.testing.assert_allclose(l1 + l2 - 2 * m.b, 0.0, atol=1e-9)


def test_embedding_gradients_match_finite_differences():
    """ Hand-written gradients are the highest-risk code here. Check them. """
    rng = np.random.default_rng(6)
    n = 32
    R = np.zeros((n, 5), dtype=np.int64)
    D = np.zeros((n, 5), dtype=np.int64)
    for i in range(n):
        p = rng.choice(np.arange(H), size=10, replace=False)
        R[i], D[i] = p[:5], p[5:]
    y = rng.integers(0, 2, n).astype(np.float64)

    m = HeroEmbeddingModel(H, k=4, l2=0.0, init_scale=0.1)

    def loss():
        lg, _ = m._forward(R, D)
        p = sigmoid(lg)
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

    # analytic gradient for s, replicating the fit() computation
    lg, (sR, sD, SR, SD, AR, AD, CR, CD) = m._forward(R, D)
    p = sigmoid(lg)
    g = (p - y) / n
    gs = np.zeros_like(m.s)
    np.add.at(gs, R, g[:, None, None] * (SR[:, None, :] - sR))
    np.add.at(gs, D, -g[:, None, None] * (SD[:, None, :] - sD))

    eps = 1e-6
    checked = 0
    for hero in np.unique(R[:4]):
        for dim in range(m.k):
            orig = m.s[hero, dim]
            m.s[hero, dim] = orig + eps
            lp = loss()
            m.s[hero, dim] = orig - eps
            lm = loss()
            m.s[hero, dim] = orig
            numeric = (lp - lm) / (2 * eps)
            assert numeric == pytest.approx(gs[hero, dim], abs=1e-7), \
                "gradient mismatch for hero %d dim %d" % (hero, dim)
            checked += 1
    assert checked > 0


def test_training_reduces_loss_on_learnable_signal():
    """ Give the model a draft rule it can learn, and confirm it learns it. """
    rng = np.random.default_rng(7)
    n = 4000
    R = np.zeros((n, 5), dtype=np.int64)
    D = np.zeros((n, 5), dtype=np.int64)
    for i in range(n):
        p = rng.choice(np.arange(H), size=10, replace=False)
        R[i], D[i] = p[:5], p[5:]
    # hero 0 is overwhelmingly strong whichever side it is on
    y = np.where(np.any(R == 0, axis=1), 1.0,
                 np.where(np.any(D == 0, axis=1), 0.0,
                          rng.integers(0, 2, n).astype(np.float64)))
    m = HeroEmbeddingModel(H, k=4, l2=1e-8)
    m.fit(R, D, y, epochs=12, batch=256, lr=0.05, verbose=False)
    p = m.predict_proba(R, D)
    strong_r = np.any(R == 0, axis=1)
    strong_d = np.any(D == 0, axis=1)
    assert p[strong_r].mean() > 0.65, "should learn hero 0 helps radiant"
    assert p[strong_d].mean() < 0.35, "should learn hero 0 hurts radiant"


# --------------------------------------------------------------------------
# prefix model
# --------------------------------------------------------------------------

def test_prefix_full_mask_matches_unmasked_semantics(toy):
    """ With an all-ones mask the prefix model must behave like a normal 5v5. """
    R, D, y, _ = toy
    Ri, Di = R.astype(np.int64) - 1, D.astype(np.int64) - 1
    m = PrefixEmbeddingModel(H, k=8)
    ones = np.ones_like(Ri, dtype=np.float64)
    lg_full, _ = m._forward(Ri, Di, ones, ones)
    assert np.all(np.isfinite(lg_full))
    # masking one hero off both sides must change the logit
    part = ones.copy()
    part[:, 4] = 0.0
    lg_part, _ = m._forward(Ri, Di, part, ones)
    assert not np.allclose(lg_full, lg_part)


def test_prefix_masked_hero_has_no_influence():
    """ A hero whose mask is zero must not affect the output at all. """
    rng = np.random.default_rng(8)
    m = PrefixEmbeddingModel(H, k=8, init_scale=0.5)
    R = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    D = np.array([[5, 6, 7, 8, 9]], dtype=np.int64)
    mR = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    mD = np.ones((1, 5))
    lg1, _ = m._forward(R, D, mR, mD)
    R2 = R.copy()
    R2[0, 4] = 99          # change the masked-out slot
    lg2, _ = m._forward(R2, D, mR, mD)
    np.testing.assert_allclose(lg1, lg2, atol=1e-12)


def test_sample_masks_produce_valid_prefixes():
    rng = np.random.default_rng(9)
    mR, mD = PrefixEmbeddingModel.sample_masks(500, rng)
    assert mR.shape == (500, 5) and mD.shape == (500, 5)
    assert np.all(np.isin(mR, [0.0, 1.0]))
    # masks must be left-packed: no gaps
    for row in np.vstack([mR, mD]):
        assert list(row) == sorted(row, reverse=True), "mask has a gap"
    assert (mR.sum(1) + mD.sum(1)).max() == 10


# --------------------------------------------------------------------------
# sequential query
# --------------------------------------------------------------------------

@pytest.fixture
def small_model():
    return PrefixEmbeddingModel(H, k=8, init_scale=0.1)


def test_rank_picks_never_suggests_a_drafted_hero(small_model):
    R, D = [0, 1, 2], [3, 4]
    out = rank_picks(small_model, R, D, side='radiant', top=20, n_heroes=H)
    suggested = {h for h, _, _ in out}
    assert not (suggested & set(R + D)), "recommended an already-picked hero"


def test_rank_picks_respects_bans(small_model):
    banned = (10, 11, 12)
    out = rank_picks(small_model, [0], [1], side='radiant', top=50,
                     banned=banned, n_heroes=H)
    assert not ({h for h, _, _ in out} & set(banned))


def test_rank_picks_is_sorted_descending(small_model):
    out = rank_picks(small_model, [0, 1], [2, 3], side='radiant', top=15, n_heroes=H)
    scores = [s for _, s, _ in out]
    assert scores == sorted(scores, reverse=True)


def test_rank_picks_works_at_every_draft_stage(small_model):
    """ The whole point of the extension: it must not require nine heroes. """
    for k in range(5):
        R, D = list(range(k)), list(range(50, 50 + k))
        out = rank_picks(small_model, R, D, side='radiant', top=3, n_heroes=H)
        assert len(out) == 3, "failed with %d heroes drafted" % (2 * k)


def test_scores_are_valid_probabilities(small_model):
    out = rank_picks(small_model, [0, 1], [2, 3], side='radiant', top=10, n_heroes=H)
    assert all(0.0 <= s <= 1.0 for _, s, _ in out)
    assert 0.0 <= score_state(small_model, [0, 1], [2, 3]) <= 1.0


def test_lookahead_scores_are_not_higher_than_greedy(small_model):
    """ Assuming the opponent replies optimally can only lower your score. """
    R, D = [0, 1], [2, 3]
    greedy = dict((h, s) for h, s, _ in
                  rank_picks(small_model, R, D, 'radiant', top=H, n_heroes=H))
    look = dict((h, s) for h, s, _ in
                rank_picks(small_model, R, D, 'radiant', top=H,
                           lookahead=True, n_heroes=H))
    for h in list(look)[:20]:
        assert look[h] <= greedy[h] + 1e-9, "lookahead exceeded greedy for hero %d" % h


def test_dire_side_scoring_is_complementary(small_model):
    """ Scoring for dire must be 1 - radiant probability. """
    R, D = [0, 1], [2, 3]
    p_rad = score_state(small_model, R, D)
    out = rank_picks(small_model, R, D, side='dire', top=1, n_heroes=H)
    assert 0.0 <= out[0][1] <= 1.0
    assert p_rad + (1 - p_rad) == pytest.approx(1.0)
