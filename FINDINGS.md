# dota2-predictor — port and extensions

Fork of andreiapostoae/dota2-predictor. Original: Python 2.7, logistic regression on
one-hot hero indicators plus three hand-counted advantage scalars, ~0.63 ROC AUC.

Dataset: patch 7.06e, 1,928,183 public matches (1,793,211 train + 134,972 test as shipped).

## What changed

### Port
- Python 2.7 → 3.12 across `advantages.py`, `dataset.py`, `cross_validation.py`.
  (`metadata.py` and `augmenter.py` needed no changes.)
- `map()` iterator indexing, implicit relative imports, `sklearn.externals.joblib`.
- Fixed a real bug: `dataset.py` computed the advantage columns inside the
  5-iteration hero loop, recomputing identical values five times per game.

### Performance
`preprocessing/fast_features.py` replaces the dense row-by-row builder with a
vectorised sparse one.

| | original | new |
|---|---|---|
| memory (1.8M games) | 3,314 MB | 194 MB |
| build time | minutes (OOM-killed) | 14 s |

### Methodology
- **Temporal split.** `match_id` increases with time. The shipped split is random —
  100% of test match ids fall inside the train id range. `preprocessing/improved.py`
  splits chronologically instead.
- **Leakage removed.** `pretrained/synergies_all.csv` was computed over the whole
  dataset including test games. Recomputing on train only costs 0.013 AUC of
  phantom performance.

### Models
- `training/embedding_model.py` — factorization-machine style. Each hero gets three
  learned vectors (synergy `s`, counter `a`/`c`) instead of a one-hot column.
  Team synergy uses the FM identity so a 5-hero team costs O(5k), not O(25k).
  The counter term is antisymmetric by construction, so
  `logit(R,D) = -logit(D,R) + 2b` holds to machine precision — mirror symmetry is
  structural, not learned, which makes mirror data augmentation unnecessary.
  Hand-written NumPy with explicit gradients and Adam.
- `training/prefix_model.py` — same model with masking, trained on randomly
  truncated drafts so it can score any partial state.
- `training/sequential_query.py` — pick recommendation at any draft stage, with
  optional one-ply opponent-reply lookahead.

## Results

All temporal split, features computed on train only.

| model | AUC | Brier |
|---|---|---|
| original setup, random split (reproduces author) | 0.6470 | 0.2326 |
| one-hot only, no advantage columns | 0.6269 | — |
| leaky precomputed advantages | 0.6632 | 0.2288 |
| **honest baseline: one-hot + train-only advantages** | **0.6504** | 0.2319 |
| LightGBM, same features | 0.6427 | 0.2335 |
| learned embeddings, k=16 | 0.6484 | 0.2323 |
| average ensemble | 0.6528 | 0.2313 |
| stacked | **0.6536** | **0.2311** |
| prefix model, full draft | 0.6455 | — |

### Signal accumulates through the draft

| heroes drafted | AUC |
|---|---|
| 2 | 0.5572 |
| 4 | 0.5830 |
| 6 | 0.6049 |
| 8 | 0.6250 |
| 10 | 0.6455 |

The original code cannot produce this curve; it only scores complete drafts.

### Calibration
Mean absolute gap between predicted and observed win rate, 10 quantile bins:
baseline 0.0086, embeddings 0.0058, LightGBM 0.0037. All three are well calibrated
across the full 0.28–0.77 range, which matters more than AUC for a decision tool.

### Greedy scoring is optimistic
Scoring a candidate pick without the opponent's reply overstates it. Example state
(Radiant: Pudge, Juggernaut / Dire: Invoker, Lion): Spectre reads 0.684 greedily,
0.586 after one-ply lookahead. Top-5 overlap between the two rankings is 3/5 —
heroes that are strong but easily countered drop out.

## Negative results

Recorded because they were predicted to work and did not.

1. **Empirical Bayes shrinkage of the winrate matrices: no effect.** k=10 through
   k=1000 all land at ~0.650. Median teammate-pair support is 3,227 games and only
   1.8% of pairs have under 100, so the prior is swamped. This fix is right for a
   smaller dataset and pointless at 1.9M games.
2. **LightGBM underperforms logistic regression** (0.6427 vs 0.6504). Sparse binary
   indicators with weak signal are not favourable to tree splits.
3. **Embeddings alone lose to the one-hot baseline.** 231 freely-fit coefficients
   beat a low-rank factorization. They only pay off in ensemble — logit correlation
   is 0.907, so they are learning something partly different.
4. **The synergy embedding is not semantically interpretable.** Nearest neighbours
   are not similar heroes (Anti-Mage → Lion, Juggernaut → Timbersaw). The objective
   is win prediction, not similarity, and the signal is too weak for structure to
   emerge. Would need a co-occurrence objective instead.

## Not done
- `tools/miner.py` port (`urllib2`, `pandas.append`, `json_normalize` all need
  replacing; untestable here since the OpenDota API is unreachable from the sandbox)
- Bans, role/lane composition, MMR as a continuous conditioning input
- Deeper search than one ply
