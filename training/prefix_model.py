""" Draft-prefix embedding model.

The original project can only score a draft that is complete or one hero short:
query.py assumes nine heroes are locked and brute-forces the tenth. Real drafting
is a sequence of decisions made when the board is half empty, so a useful model
has to score partial states.

This variant accepts variable-size teams via a boolean mask, and is trained on
randomly truncated drafts so that it sees every prefix length during training
rather than only the complete 5v5.

Team representation is a padded (n, 5) index array plus an (n, 5) float mask.
Every sum over a team is a masked sum, so the factorization-machine identity
still holds for teams of any size:

    syn(T) = ( ||sum_{i in T} s_i||^2 - sum_{i in T} ||s_i||^2 ) / 2

with absent slots contributing zero to both terms.
"""
import numpy as np
from training.embedding_model import sigmoid


class PrefixEmbeddingModel:
    def __init__(self, n_heroes, k=16, seed=42, l2=1e-5, init_scale=0.01):
        rng = np.random.default_rng(seed)
        self.H, self.k, self.l2 = n_heroes, k, l2
        self.b = 0.0
        self.w = np.zeros(n_heroes)
        self.s = rng.normal(0, init_scale, (n_heroes, k))
        self.a = rng.normal(0, init_scale, (n_heroes, k))
        self.c = rng.normal(0, init_scale, (n_heroes, k))
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ('w', 's', 'a', 'c')}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ('w', 's', 'a', 'c')}
        self.mb = self.vb = 0.0
        self.t = 0

    def _forward(self, R, D, mR, mD):
        sR = self.s[R] * mR[:, :, None]
        sD = self.s[D] * mD[:, :, None]
        SR, SD = sR.sum(1), sD.sum(1)
        synR = 0.5 * ((SR ** 2).sum(1) - (sR ** 2).sum((1, 2)))
        synD = 0.5 * ((SD ** 2).sum(1) - (sD ** 2).sum((1, 2)))

        aR = self.a[R] * mR[:, :, None]
        aD = self.a[D] * mD[:, :, None]
        cR = self.c[R] * mR[:, :, None]
        cD = self.c[D] * mD[:, :, None]
        AR, AD, CR, CD = aR.sum(1), aD.sum(1), cR.sum(1), cD.sum(1)
        counter = (AR * CD).sum(1) - (AD * CR).sum(1)

        lin = (self.w[R] * mR).sum(1) - (self.w[D] * mD).sum(1)
        return self.b + lin + synR - synD + counter, (sR, sD, SR, SD, AR, AD, CR, CD)

    def predict_proba(self, R, D, mR, mD, batch=200000):
        out = np.empty(R.shape[0])
        for i in range(0, R.shape[0], batch):
            lg, _ = self._forward(R[i:i + batch], D[i:i + batch],
                                  mR[i:i + batch], mD[i:i + batch])
            out[i:i + batch] = sigmoid(lg)
        return out

    def _adam(self, name, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m[name] = b1 * self.m[name] + (1 - b1) * grad
        self.v[name] = b2 * self.v[name] + (1 - b2) * grad ** 2
        mh = self.m[name] / (1 - b1 ** self.t)
        vh = self.v[name] / (1 - b2 ** self.t)
        setattr(self, name, getattr(self, name) - lr * mh / (np.sqrt(vh) + eps))

    @staticmethod
    def sample_masks(n, rng, full_prob=0.25):
        """ Sample realistic draft prefixes.

        With probability full_prob emit a complete 5v5. Otherwise pick a total
        number of heroes drafted so far and split it alternately between the two
        sides, which mirrors how captains alternate picks.
        """
        mR = np.ones((n, 5), dtype=np.float64)
        mD = np.ones((n, 5), dtype=np.float64)
        partial = rng.random(n) >= full_prob
        idx = np.where(partial)[0]
        total = rng.integers(1, 10, size=idx.size)
        nR = (total + 1) // 2
        nD = total // 2
        swap = rng.random(idx.size) < 0.5
        nR2 = np.where(swap, nD, nR)
        nD2 = np.where(swap, nR, nD)
        ar = np.arange(5)
        mR[idx] = (ar[None, :] < nR2[:, None]).astype(np.float64)
        mD[idx] = (ar[None, :] < nD2[:, None]).astype(np.float64)
        return mR, mD

    def fit(self, R, D, y, epochs=6, batch=16384, lr=0.01, val=None, seed=0, verbose=True):
        n = R.shape[0]
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for st in range(0, n, batch):
                idx = perm[st:st + batch]
                Rb, Db, yb = R[idx], D[idx], y[idx]
                mb_, md_ = self.sample_masks(len(idx), rng)
                # shuffle hero order within each team so the mask truncates a
                # random subset rather than always the same slots
                for arr, msk in ((Rb, mb_), (Db, md_)):
                    pass
                perm5 = np.argsort(rng.random(Rb.shape), axis=1)
                Rb = np.take_along_axis(Rb, perm5, axis=1)
                perm5d = np.argsort(rng.random(Db.shape), axis=1)
                Db = np.take_along_axis(Db, perm5d, axis=1)

                m = len(idx)
                logit, (sR, sD, SR, SD, AR, AD, CR, CD) = self._forward(Rb, Db, mb_, md_)
                p = sigmoid(logit)
                g = (p - yb) / m
                tot += -np.mean(yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))
                nb += 1

                gw = np.zeros_like(self.w)
                gs = np.zeros_like(self.s)
                ga = np.zeros_like(self.a)
                gc = np.zeros_like(self.c)

                np.add.at(gw, Rb, g[:, None] * mb_)
                np.add.at(gw, Db, -g[:, None] * md_)

                gk = g[:, None, None]
                np.add.at(gs, Rb, gk * (SR[:, None, :] - sR) * mb_[:, :, None])
                np.add.at(gs, Db, -gk * (SD[:, None, :] - sD) * md_[:, :, None])
                np.add.at(ga, Rb, gk * CD[:, None, :] * mb_[:, :, None])
                np.add.at(ga, Db, -gk * CR[:, None, :] * md_[:, :, None])
                np.add.at(gc, Db, gk * AR[:, None, :] * md_[:, :, None])
                np.add.at(gc, Rb, -gk * AD[:, None, :] * mb_[:, :, None])

                gw += self.l2 * self.w
                gs += self.l2 * self.s
                ga += self.l2 * self.a
                gc += self.l2 * self.c

                self.t += 1
                gb = g.sum()
                self.mb = 0.9 * self.mb + 0.1 * gb
                self.vb = 0.999 * self.vb + 0.001 * gb ** 2
                self.b -= lr * (self.mb / (1 - 0.9 ** self.t)) / (
                    np.sqrt(self.vb / (1 - 0.999 ** self.t)) + 1e-8)
                self._adam('w', gw, lr)
                self._adam('s', gs, lr)
                self._adam('a', ga, lr)
                self._adam('c', gc, lr)

            if verbose:
                msg = "epoch %d  train logloss %.5f" % (ep + 1, tot / nb)
                if val is not None:
                    from sklearn.metrics import roc_auc_score
                    Rv, Dv, yv = val
                    ones = np.ones_like(Rv, dtype=np.float64)
                    msg += "  full-draft val AUC %.4f" % roc_auc_score(
                        yv, self.predict_proba(Rv, Dv, ones, ones))
                print(msg, flush=True)
