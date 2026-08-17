""" Reproduce the baseline and the model comparison table in FINDINGS.md.

    python scripts/run_baseline.py [--lightgbm]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from scripts.common import load_temporal, train_only_advantages, heroes_released
from preprocessing.fast_features import build_onehot, advantage_columns


def report(name, y, p):
    auc = roc_auc_score(y, p)
    print("%-46s AUC %.4f  Brier %.4f" % (name, auc, brier_score_loss(y, p)), flush=True)
    return auc


def main(use_lgbm=False):
    t0 = time.time()
    H = heroes_released()
    r, d, y, mid, tr, te = load_temporal()

    syn, cnt = train_only_advantages(r, d, y, tr, H)[:2]
    adv = advantage_columns(r, d, syn, cnt)
    mu, sd = adv[tr].mean(0), adv[tr].std(0)
    X = sparse.hstack([build_onehot(r, d, H),
                       sparse.csr_matrix((adv - mu) / sd)], format='csr')
    yte = y[te]

    print("\n=== model comparison (temporal split, train-only advantages) ===")
    base = LogisticRegression(C=1.0, random_state=42, max_iter=300).fit(X[tr], y[tr])
    p_base = base.predict_proba(X[te])[:, 1]
    report("one-hot + advantages (baseline)", yte, p_base)

    m2 = LogisticRegression(C=1.0, random_state=42, max_iter=300).fit(X[tr][:, :-3], y[tr])
    report("one-hot only (ablation)", yte, m2.predict_proba(X[te][:, :-3])[:, 1])

    if use_lgbm:
        import lightgbm as lgb
        params = dict(objective='binary', metric='auc', learning_rate=0.05,
                      num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, verbose=-1)
        gbm = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=400)
        report("LightGBM (same features)", yte, gbm.predict(X[te]))

    print("\n=== calibration (10 quantile bins) ===")
    frac, mean = calibration_curve(yte, p_base, n_bins=10, strategy='quantile')
    for pr, ob in zip(mean, frac):
        print("  predicted %.3f -> observed %.3f" % (pr, ob))
    print("  mean |predicted - observed| = %.4f" % np.mean(np.abs(frac - mean)))

    os.makedirs('artifacts/cache', exist_ok=True)
    np.save('artifacts/cache/p_base.npy', p_base)
    print("\ndone in %.0fs" % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lightgbm', action='store_true', help='also run the LightGBM comparison')
    main(ap.parse_args().lightgbm)
