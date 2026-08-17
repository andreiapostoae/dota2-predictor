""" Train the prefix model and measure how predictive signal grows through the draft.

    python scripts/train_prefix.py [--k 16] [--epochs 8]

This produces the draft-completeness curve, which the upstream code cannot: it
only scores complete drafts.
"""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.common import load_temporal, heroes_released
from training.prefix_model import PrefixEmbeddingModel


def main(k, epochs, lr, l2):
    H = heroes_released()
    r, d, y, mid, tr, te = load_temporal()
    R, D = r.astype(np.int64) - 1, d.astype(np.int64) - 1

    model = PrefixEmbeddingModel(H, k=k, l2=l2)
    t0 = time.time()
    model.fit(R[tr], D[tr], y[tr], epochs=epochs, batch=16384, lr=lr,
              val=(R[te], D[te], y[te]))
    print("trained in %.0fs" % (time.time() - t0))

    os.makedirs('artifacts', exist_ok=True)
    pickle.dump(model, open('artifacts/prefix_model.pkl', 'wb'))

    print("\n=== AUC by draft completeness ===")
    n = int(te.sum())
    ar = np.arange(5)
    for total in [2, 4, 6, 8, 10]:
        nR, nD = (total + 1) // 2, total // 2
        mR = np.tile((ar < nR).astype(np.float64), (n, 1))
        mD = np.tile((ar < nD).astype(np.float64), (n, 1))
        p = model.predict_proba(R[te], D[te], mR, mD)
        print("  %2d heroes known   AUC %.4f" % (total, roc_auc_score(y[te], p)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--l2', type=float, default=1e-5)
    a = ap.parse_args()
    main(a.k, a.epochs, a.lr, a.l2)
