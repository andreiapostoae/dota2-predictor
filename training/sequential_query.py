""" Sequential pick recommendation.

The original query.py answers one question: given nine heroes, what is the best
tenth? It scores every candidate in the empty slot and sorts.

Two things are missing from that. First, it cannot answer the question at any
earlier point in the draft. Second, it is greedy: it treats your pick as the
last word, when in reality the opponent responds to it.

This module fixes both. Candidates are scored against a partial draft of any
size, and each candidate is optionally evaluated after assuming the opponent
plays their own best reply -- a one-ply minimax lookahead.
"""
import numpy as np
from training.embedding_model import sigmoid


def _pad(team, size=5):
    arr = np.full(size, 0, dtype=np.int64)
    mask = np.zeros(size, dtype=np.float64)
    for i, h in enumerate(team[:size]):
        arr[i] = h
        mask[i] = 1.0
    return arr, mask


def score_state(model, radiant, dire):
    """ Win probability for Radiant given two partial hero lists (zero-based ids). """
    R, mR = _pad(radiant)
    D, mD = _pad(dire)
    return float(model.predict_proba(R[None, :], D[None, :], mR[None, :], mD[None, :])[0])


def rank_picks(model, radiant, dire, side='radiant', banned=(), n_heroes=114,
               top=10, lookahead=False, hero_names=None):
    """ Rank every legal hero for the side that is about to pick.

    Args:
        radiant, dire: lists of zero-based hero ids already drafted
        side: which team is picking now
        banned: hero ids unavailable
        lookahead: if True, assume the opponent then plays their single best
            reply, and score the resulting position instead of the immediate one
    Returns:
        list of (hero_id, score, name) sorted best-first for the picking side
    """
    taken = set(radiant) | set(dire) | set(banned)
    legal = [h for h in range(n_heroes) if h not in taken]
    if not legal:
        return []

    cand = np.array(legal, dtype=np.int64)
    n = len(cand)

    if side == 'radiant':
        base_r = list(radiant)
        R = np.zeros((n, 5), dtype=np.int64); mR = np.zeros((n, 5))
        for i, h in enumerate(base_r):
            R[:, i] = h; mR[:, i] = 1.0
        slot = len(base_r)
        R[:, slot] = cand; mR[:, slot] = 1.0
        D = np.zeros((n, 5), dtype=np.int64); mD = np.zeros((n, 5))
        for i, h in enumerate(dire):
            D[:, i] = h; mD[:, i] = 1.0
    else:
        base_d = list(dire)
        D = np.zeros((n, 5), dtype=np.int64); mD = np.zeros((n, 5))
        for i, h in enumerate(base_d):
            D[:, i] = h; mD[:, i] = 1.0
        slot = len(base_d)
        D[:, slot] = cand; mD[:, slot] = 1.0
        R = np.zeros((n, 5), dtype=np.int64); mR = np.zeros((n, 5))
        for i, h in enumerate(radiant):
            R[:, i] = h; mR[:, i] = 1.0

    p = model.predict_proba(R, D, mR, mD)
    scores = p if side == 'radiant' else 1.0 - p

    if lookahead:
        adjusted = np.empty(n)
        for i, h in enumerate(cand):
            if side == 'radiant':
                new_r = list(radiant) + [int(h)]
                if len(dire) >= 5:
                    adjusted[i] = scores[i]
                    continue
                reply = rank_picks(model, new_r, list(dire), side='dire',
                                   banned=tuple(taken) + (int(h),),
                                   n_heroes=n_heroes, top=1, lookahead=False)
                adjusted[i] = 1.0 - reply[0][1] if reply else scores[i]
            else:
                new_d = list(dire) + [int(h)]
                if len(radiant) >= 5:
                    adjusted[i] = scores[i]
                    continue
                reply = rank_picks(model, list(radiant), new_d, side='radiant',
                                   banned=tuple(taken) + (int(h),),
                                   n_heroes=n_heroes, top=1, lookahead=False)
                adjusted[i] = 1.0 - reply[0][1] if reply else scores[i]
        scores = adjusted

    order = np.argsort(-scores)[:top]
    out = []
    for j in order:
        hid = int(cand[j])
        name = hero_names.get(hid + 1, str(hid + 1)) if hero_names else str(hid + 1)
        out.append((hid, float(scores[j]), name))
    return out
