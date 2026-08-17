# dota2-predictor — Python 3 port and modelling extensions

Fork of [andreiapostoae/dota2-predictor](https://github.com/andreiapostoae/dota2-predictor).

The upstream project predicts Dota 2 match outcomes from the draft alone: logistic
regression over one-hot hero indicators plus three hand-counted "advantage" scalars,
reaching roughly 0.63 ROC AUC. It targets Python 2.7 and no longer runs.

This fork ports it to Python 3, fixes two methodology problems that were inflating
the reported numbers, and extends it from a full-draft scorer into something that
can score and recommend at **any point in the draft**.

Full write-up with all results, including the ones that did not work:
**[FINDINGS.md](FINDINGS.md)**

---

## Headline results

Dataset: patch 7.06e, 1,928,183 public matches. Chronological 85/15 split on `match_id`.

| model | AUC | Brier |
|---|---|---|
| upstream setup, random split (reproduces author) | 0.6470 | 0.2326 |
| *leaky* precomputed advantages | *0.6632* | *0.2288* |
| **honest baseline: one-hot + train-only advantages** | **0.6504** | 0.2319 |
| LightGBM, same features | 0.6427 | 0.2335 |
| learned hero embeddings, k=16 | 0.6484 | 0.2323 |
| stacked (baseline + embeddings) | **0.6536** | **0.2311** |

Signal accumulates as the draft fills in — a curve the upstream code cannot produce:

| heroes drafted | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| AUC | 0.5572 | 0.5830 | 0.6049 | 0.6250 | 0.6455 |

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_baseline.py --lightgbm     # baseline + calibration  (~1 min)
python scripts/train_embeddings.py            # embedding model         (~3 min)
python scripts/train_prefix.py                # prefix model + curve    (~3 min)

python scripts/demo_draft.py \
    --radiant Pudge,Juggernaut --dire Invoker,Lion --lookahead
```

The dataset ships as `706e.zip` and is extracted automatically on first run.
Intermediates are cached in `artifacts/cache/`, so only the first script pays the
parsing cost.

---

## What changed

**Port.** Python 2.7 → 3.12 across `advantages.py`, `dataset.py`,
`cross_validation.py` (`metadata.py` and `augmenter.py` needed nothing): `map()`
iterator indexing, implicit relative imports, `sklearn.externals.joblib`. Also fixed
a genuine bug — `dataset.py` computed the advantage columns inside the 5-iteration
hero loop, recomputing identical values five times per game.

**Performance.** `preprocessing/fast_features.py` replaces the dense row-by-row
builder with a vectorised sparse one: **3,314 MB → 194 MB**, minutes → 14 s. The
original OOM-killed on the full dataset.

**Two methodology fixes.** The shipped split is random, and 100% of its test
`match_id` values fall inside the training range; `match_id` increases with time, so
the split should be chronological. Separately, `pretrained/synergies_all.csv` was
computed over the *whole* dataset including test games. Together these were worth
about 0.013 AUC of phantom performance.

**Learned embeddings** (`training/embedding_model.py`). Each hero gets three learned
vectors instead of a one-hot column, with synergy and counter emerging from dot
products. Uses the factorization-machine identity so a 5-hero team costs O(5k) rather
than O(25k). The counter term is antisymmetric by construction, so
`logit(R,D) = -logit(D,R) + 2b` holds to machine precision — side symmetry is
structural rather than learned, which makes mirror data augmentation unnecessary.
Hand-written NumPy with explicit gradients and Adam.

**Partial drafts** (`training/prefix_model.py`). The same model with masking, trained
on randomly truncated drafts so it can score a board that is half empty.

**Sequential recommendation** (`training/sequential_query.py`). Upstream `query.py`
answers one question — given nine heroes, what is the tenth? This ranks candidates at
any draft stage, with optional one-ply opponent-reply lookahead.

Greedy scoring turns out to be systematically optimistic. From
`Radiant: Pudge, Juggernaut / Dire: Invoker, Lion`, Spectre reads **0.684** greedily
but **0.586** once the opponent replies, and only 3 of the top 5 survive the change.
Heroes that are strong but easily punished drop out.

---

## Testing

```bash
pip install pytest
pytest                             # 24 fast unit tests, ~2s, no dataset needed
pytest -m slow                     # 5 integration tests against real data, ~60s
pytest -m "slow or not slow" -v    # everything
```

The fast suite checks correctness properties that hold regardless of data: the
vectorised feature builder produces identical output to the upstream loop, the
O(5k) factorization-machine shortcut equals the explicit O(25k) pair sum, the
hand-written gradients match finite differences, masked-out heroes have exactly
zero influence, and `rank_picks` never suggests a drafted or banned hero.

The slow suite guards the headline number: `test_baseline_reaches_expected_auc`
fails if the documented 0.6504 drifts by more than 0.004.

The suite was validated by mutation testing — breaking the FM constant, flipping
the counter term's sign, removing the one-hot side offset, and disabling the
drafted-hero filter each produce failures, so the tests are not vacuous.

CI runs the fast suite on Python 3.10 and 3.12 (`.github/workflows/tests.yml`).

---

## Layout

```
preprocessing/
  advantages.py        upstream pair-winrate counting (ported)
  augmenter.py         upstream per-game advantage summary
  dataset.py           upstream feature builder (ported, bug fixed)
  fast_features.py     NEW  vectorised sparse feature builder
  improved.py          NEW  shrinkage, temporal split, mirroring
training/
  cross_validation.py  upstream train/evaluate (ported)
  query.py             upstream last-pick query
  embedding_model.py   NEW  hero embedding model, NumPy + Adam
  prefix_model.py      NEW  masked variant for partial drafts
  sequential_query.py  NEW  any-stage recommendation + lookahead
scripts/               NEW  reproduction entry points
tests/                 NEW  pytest suite (24 fast + 5 slow)
artifacts/             trained models and results.json
docs/UPSTREAM_README.md  the original author's README
FINDINGS.md            full write-up including negative results
```

---

## Honest caveats

- The modelling gain is small: **0.6536 vs 0.6504** against the leakage-free baseline.
  The larger number against the upstream 0.6470 is partly split methodology, not model
  quality.
- The prefix model was still improving at 8 epochs — it is undertrained.
- Four things that were expected to help did not: shrinkage, LightGBM, embeddings on
  their own, and embedding interpretability. All four are documented with diagnosis in
  FINDINGS.md rather than quietly dropped.
- `tools/miner.py` is **not** ported. It needs `urllib2`, `pandas.append` and
  `json_normalize` replaced, and could not be tested in the environment this work was
  done in.

## Licence
Upstream licence retained — see [LICENSE](LICENSE).
