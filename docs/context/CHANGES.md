> Historical project record. File paths and setup details describe the project at the time of writing. See the [current README](../../README.md) for current commands.

# Phase 1 cleanup record

Scope: Phase 1 repairs only; no transformer, DL training or Phase 2 implementation.

Preserved the entire original folder in the sibling phase1_backup archive before
editing. Report.pdf, the presentation and historical metrics are unchanged.

- Replaced four duplicated stage implementations and the orchestration script with
  thin entry points calling shared Phase 1 modules. One documented command runs all.
- Removed unused spaCy/tqdm dependencies and installed the pinned environment.
- Restored official train.csv, verified dev/test against the same fixed revision,
  and recorded SHA-256 hashes and actual CSV record counts.
- Added content-based IDs independent of split/index; audited shared premises and
  excluded shared development premises from selection. Original splits are intact.
- Replaced training CV mislabeled as dev evaluation and test-based winner selection
  with explicit development selection. Saved preprocessing with feature order.
- Bounded the ML parameter search and calibrated the selected graph model on training
  folds. Selected the probability threshold using development macro F1. Reused processed/feature caches by hash.
- Split graph construction by premise. Only direct claims produce direct edges.
  Other relationship types remain annotations. Removed constant 1.0 confidence.
- Removed silent ground-truth/synthetic fallbacks and placeholder successful results.
- Replaced assumed text-derived BN effects with a separate specified confounded
  example and three numerically verified conditioning/intervention answers.
- Added EDA/metric/graph figures, per-class reports and behavioral tests.
- Streamed legacy vocabulary counting to avoid one enormous list of words.

Documentation corrections: the supplied training CSV contains 205,734 records,
not 411,469. Feature schema has 28 numeric columns, not 29. Original test badges
reported 30/34 checks while the old script actually ran 27 in this copy. New test
results are recorded separately. Some one-hot columns are unused by the historical
feature design; they are retained for comparability, not represented as new signal.

Scientific limitations: a candidate DAG is not proof of causality; the current data
cannot supply empirical CPDs or complete gold DAG precision. Future graph evaluation
needs a separately documented gold set. No claims of stronger model performance are
made without measured corrected results.

## Removal of redundant files

Removed obsolete numbered stage wrappers, superseded models/features/processed
CSVs/graphs, placeholder context files, unused decorative SVGs, upstream placeholder
README, unused seaborn dependency and generated Python/OS caches. Retained original
EDA plots because they support Phase 1 evaluation. Shared data and feature code now
live in src/data.py and src/features.py. Only called functions/constants remain.
Report, presentation, original metrics and EDA figures are together under
docs/phase1_submission/. Literature, feature dictionary and operating notes are in docs/.

The pre-pruning state is also saved as
../phase1_backup/AMLDL_before_pruning.tar.gz (environment, raw CSVs and outputs excluded).
The original full backup remains available. No Phase 2 work was added.
