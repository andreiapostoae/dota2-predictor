""" Learned hero embeddings for draft win prediction.

The original model gives every hero an independent one-hot coefficient plus three
hand-counted summary scalars. Here each hero instead gets three learned vectors,
and synergy / counter effects emerge from their dot products.

Model
-----
For a draft with radiant set R and dire set D:

    logit = b
          + sum_{i in R} w[i]  -  sum_{j in D} w[j]              linear hero strength
          + syn(R)             -  syn(D)                          within-team synergy
          + <A_R, C_D>         -  <A_D, C_R>                      cross-team counter

where syn(T) = ( ||sum_{i in T} s_i||^2 - sum_{i in T} ||s_i||^2 ) / 2

That identity is the factorization-machine trick: it computes the sum over all
pairs in the team without ever enumerating pairs, so a 5-hero team costs O(5k)
instead of O(25k).

Side symmetry
-------------
Every term except the bias flips sign when R and D are exchanged, so
    logit(D, R) = -logit(R, D) + 2b
Mirror symmetry is therefore built into the architecture rather than being
something the model has to learn from mirrored training data. The bias b
absorbs the genuine Radiant-side advantage.

Parameters: w (H,), s (H,k), a (H,k), c (H,k).
"""
import numpy as np


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class HeroEmbeddingModel:
    def __init__(self, n_heroes, k=16, seed=42, l2=1e-6, init_scale=0.01):
        rng = np.random.default_rng(seed)
        self.H, self.k, self.l2 = n_heroes, k, l2
        self.b = 0.0
        self.w = np.zeros(n_heroes, dtype=np.float64)
        self.s = rng.normal(0, init_scale, (n_heroes, k))
        self.a = rng.normal(0, init_scale, (n_heroes, k))
        self.c = rng.normal(0, init_scale, (n_heroes, k))
        self._init_adam()

    def _init_adam(self):
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ('w', 's', 'a', 'c')}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ('w', 's', 'a', 'c')}
        self.mb = self.vb = 0.0
        self.t = 0

    def _forward(self, R, D):
        """ R, D are (n,5) zero-based hero index arrays. Returns logit and cached sums. """
        sR = self.s[R]                      # (n,5,k)
        sD = self.s[D]
        SR, SD = sR.sum(1), sD.sum(1)       # (n,k)
        synR = 0.5 * ((SR ** 2).sum(1) - (sR ** 2).sum((1, 2)))
        synD = 0.5 * ((SD ** 2).sum(1) - (sD ** 2).sum((1, 2)))

        AR, AD = self.a[R].sum(1), self.a[D].sum(1)
        CR, CD = self.c[R].sum(1), self.c[D].sum(1)
        counter = (AR * CD).sum(1) - (AD * CR).sum(1)

        lin = self.w[R].sum(1) - self.w[D].sum(1)
        logit = self.b + lin + synR - synD + counter
        return logit, (sR, sD, SR, SD, AR, AD, CR, CD)

    def predict_proba(self, R, D, batch=200000):
        out = np.empty(R.shape[0])
        for i in range(0, R.shape[0], batch):
            lg, _ = self._forward(R[i:i + batch], D[i:i + batch])
            out[i:i + batch] = sigmoid(lg)
        return out

    def _adam_step(self, name, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m[name] = b1 * self.m[name] + (1 - b1) * grad
        self.v[name] = b2 * self.v[name] + (1 - b2) * grad ** 2
        mhat = self.m[name] / (1 - b1 ** self.t)
        vhat = self.v[name] / (1 - b2 ** self.t)
        setattr(self, name, getattr(self, name) - lr * mhat / (np.sqrt(vhat) + eps))

    def fit(self, R, D, y, epochs=6, batch=8192, lr=0.02, val=None, verbose=True, seed=0):
        n = R.shape[0]
        rng = np.random.default_rng(seed)
        history = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot_loss, nb = 0.0, 0
            for st in range(0, n, batch):
                idx = perm[st:st + batch]
                Rb, Db, yb = R[idx], D[idx], y[idx]
                m = len(idx)

                logit, (sR, sD, SR, SD, AR, AD, CR, CD) = self._forward(Rb, Db)
                p = sigmoid(logit)
                g = (p - yb) / m                                  # (m,) dL/dlogit

                eps = 1e-12
                tot_loss += -np.mean(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps))
                nb += 1

                gw = np.zeros_like(self.w)
                gs = np.zeros_like(self.s)
                ga = np.zeros_like(self.a)
                gc = np.zeros_like(self.c)

                g5 = g[:, None]
                np.add.at(gw, Rb, g5)
                np.add.at(gw, Db, -g5)

                gk = g[:, None, None]
                np.add.at(gs, Rb, gk * (SR[:, None, :] - sR))
                np.add.at(gs, Db, -gk * (SD[:, None, :] - sD))

                np.add.at(ga, Rb, gk * CD[:, None, :])
                np.add.at(ga, Db, -gk * CR[:, None, :])
                np.add.at(gc, Db, gk * AR[:, None, :])
                np.add.at(gc, Rb, -gk * AD[:, None, :])

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

                self._adam_step('w', gw, lr)
                self._adam_step('s', gs, lr)
                self._adam_step('a', ga, lr)
                self._adam_step('c', gc, lr)

            msg = "epoch %d  train logloss %.5f" % (ep + 1, tot_loss / nb)
            if val is not None:
                from sklearn.metrics import roc_auc_score
                Rv, Dv, yv = val
                auc = roc_auc_score(yv, self.predict_proba(Rv, Dv))
                msg += "  val AUC %.4f" % auc
                history.append(auc)
            if verbose:
                print(msg, flush=True)
        return history
