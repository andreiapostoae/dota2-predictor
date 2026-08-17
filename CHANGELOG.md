# Changelog (fork)

## Port
- Python 2.7 -> 3.12: `preprocessing/advantages.py`, `preprocessing/dataset.py`,
  `training/cross_validation.py`
- `map()` iterator indexing, implicit relative imports, `sklearn.externals.joblib`
- Bug fix: `dataset.py` recomputed advantage columns 5x per game (loop nesting)
- `requirements.txt` rewritten for modern Python

## Added
- `preprocessing/fast_features.py` — vectorised sparse feature builder
- `preprocessing/improved.py` — empirical-Bayes shrinkage, temporal split, mirroring
- `training/embedding_model.py` — hero embedding model (NumPy, manual gradients, Adam)
- `training/prefix_model.py` — masked variant trained on partial drafts
- `training/sequential_query.py` — any-stage pick ranking with one-ply lookahead
- `scripts/` — reproduction entry points
- `README.md`, `FINDINGS.md`, `CHANGELOG.md`

## Moved
- Original README -> `docs/UPSTREAM_README.md`

## Testing
- `tests/test_models.py` — 24 unit tests (feature builder equivalence, FM identity,
  finite-difference gradient check, masking, query invariants)
- `tests/test_pipeline_smoke.py` — 5 integration tests incl. AUC regression guard
- `pytest.ini`, `.github/workflows/tests.yml`
- Suite validated by mutation testing
