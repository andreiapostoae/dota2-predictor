# How to apply this and push

This archive is an **overlay** for your existing fork clone. It contains only files
added or changed — not `706e.zip`, `static/`, `pretrained/`, or anything you have.

```bash
cd /path/to/your/dota2-predictor    # your clone of sidharthabhi/dota2-predictor

git checkout -b py3-port-and-embeddings
unzip -o /path/to/dota2-predictor-py3-port.zip -d .
```

## Test before you commit

```bash
pip install -r requirements.txt
pip install pytest

pytest                    # 24 fast tests, ~2s, no dataset needed -> all should pass
pytest -m slow            # 5 integration tests, ~60s -> all should pass
```

If the fast suite fails, something did not unzip correctly. If only
`test_baseline_reaches_expected_auc` fails, the model results drifted — worth
investigating before committing.

Then check the scripts run:

```bash
python scripts/run_baseline.py                      # prints AUC 0.6504
python scripts/train_prefix.py                      # ~3 min, draft-completeness curve
python scripts/demo_draft.py --radiant Pudge,Juggernaut --dire Invoker,Lion --lookahead
```

## Commit and push

```bash
git status
git diff --stat
git add -A
git commit -m "Port to Python 3, fix evaluation leakage, add hero embeddings

- Port advantages.py, dataset.py, cross_validation.py from Python 2.7
- Fix bug: dataset.py recomputed advantage columns 5x per game
- Add vectorised sparse feature builder (3314MB -> 194MB, minutes -> 14s)
- Switch to chronological train/test split on match_id
- Recompute advantage matrices on training games only (removes ~0.013 AUC leak)
- Add hero embedding model with antisymmetric counter term (NumPy + Adam)
- Add prefix model for partial drafts and any-stage pick recommendation
- Add pytest suite: 24 unit + 5 integration tests, validated by mutation testing
- Document results including four negative findings"

git push -u origin py3-port-and-embeddings
```

Open a PR from the branch to your own `master`. Do **not** PR to upstream unless
you mean to — this is a fork.

## Notes

- `artifacts/*.pkl` are trained models (135 KB each), committed so the demo runs
  without retraining. Drop them if you prefer no binaries in git;
  `scripts/train_prefix.py` regenerates them.
- `.gitignore` now excludes `data_raw/`, `artifacts/cache/` and `*.npz`, so the
  extracted CSVs (1.5 GB) never get committed.
- GitHub Actions will run the fast suite automatically on push.
