""" Train the fixed-size hero embedding model and ensemble it with the baseline.

    python scripts/train_embeddings.py [--k 16] [--epochs 8]
"""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

from scripts.common import load_temporal, hero_names, heroes_released
from training.embedding_model import HeroEmbeddingModel


def main(k, epochs, lr, l2):
    H = heroes_released()
    r, d, y, mid, tr, te = load_temporal()
    R, D = r.astype(np.int64) - 1, d.astype(np.int64) - 1

    model = HeroEmbeddingModel(H, k=k, l2=l2)
    t0 = time.time()
    model.fit(R[tr], D[tr], y[tr], epochs=epochs, batch=16384, lr=lr,
              val=(R[te], D[te], y[te]))
    print("trained in %.0fs" % (time.time() - t0))

    p_emb = model.predict_proba(R[te], D[te])
    yte = y[te]
    print("\nembeddings k=%d   AUC %.4f  Brier %.4f"
          % (k, roc_auc_score(yte, p_emb), brier_score_loss(yte, p_emb)))

    # verify the architecture's antisymmetry guarantee
    l1, _ = model._forward(R[:5], D[:5])
    l2_, _ = model._forward(D[:5], R[:5])
    print("antisymmetry residual (should be ~0):", np.abs(l1 + l2_ - 2 * model.b).max())

    pb = 'artifacts/cache/p_base.npy'
    if os.path.exists(pb):
        p_base = np.load(pb)
        p_avg = (p_base + p_emb) / 2
        print("average ensemble  AUC %.4f  Brier %.4f"
              % (roc_auc_score(yte, p_avg), brier_score_loss(yte, p_avg)))
        print("logit correlation base vs emb: %.4f"
              % np.corrcoef(np.log(p_base / (1 - p_base)),
                            np.log(p_emb / (1 - p_emb)))[0, 1])
    else:
        print("(run scripts/run_baseline.py first to see the ensemble)")

    os.makedirs('artifacts', exist_ok=True)
    pickle.dump(model, open('artifacts/embedding_model.pkl', 'wb'))

    names = hero_names()
    S = model.s / (np.linalg.norm(model.s, axis=1, keepdims=True) + 1e-9)
    sim = S @ S.T
    np.fill_diagonal(sim, -9)
    print("\nnearest neighbours in synergy space (NOTE: not semantically")
    print("meaningful -- see negative result 4 in FINDINGS.md)")
    for hid in [1, 8, 14, 41, 74]:
        if hid in names:
            nb = np.argsort(-sim[hid - 1])[:3]
            print("  %-20s -> %s" % (names[hid],
                  ", ".join(names.get(int(j) + 1, "?") for j in nb)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--l2', type=float, default=1e-5)
    a = ap.parse_args()
    main(a.k, a.epochs, a.lr, a.l2)
