# Causal reasoning from text

We use Corr2Cause to ask whether a causal claim follows from statements about correlation and independence. Phase 1 uses classical machine learning. Phase 2 tests whether fine-tuning DistilBERT helps and whether averaging its scores with a Random Forest helps further.

| Phase | Start here | What to show the evaluator |
| --- | --- | --- |
| **1 · Classical ML** | [Workflow and commands](docs/phase1/README.md) | [Walkthrough](notebooks/01_phase1_classical_ml.ipynb) · [Results](results/phase1/RESULTS.md) · [Original report](docs/phase1/Report.pdf) · [Slides](docs/phase1/AML_Phase1_Project_Presentation.pdf) |
| **2 · Deep learning and hybrid** | [Phase 2 guide](docs/phase2/RUN_AND_EXPLAIN.md) | [Walkthrough](notebooks/02_phase2_deep_learning.ipynb) · [Results](results/phase2/RESULTS.md) · [Report](docs/phase2/Report.pdf) · [Slides](docs/phase2/Phase%202.pptx) |

The calibrated Phase 1 classifier scored **0.6357** test macro F1 in the corrected full-data run. In Phase 2, matched ML and DistilBERT trained on the same 2,100 examples. Matched ML scored **0.5927**, DistilBERT scored **0.6726**, and the hybrid scored **0.6718**. The hybrid led on development data. DistilBERT led on test data.

## Run the saved Phase 2 demo

Both walkthrough notebooks show their outputs on GitHub. To run the Phase 2 demo locally from the repository root on macOS or Linux with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/phase2.txt
python run_phase2.py demo
python scripts/replay_demo.py
python tests/run_tests.py
```

The demo reads the published results directly. The replay rebuilds the reviewed graph and recomputes the Bayesian queries using the supplied probability tables.

## Where things go

- `run_phase1.py` and `run_phase2.py` run the two workflows. `src/` contains their implementation, and `configs/` contains the settings.
- `dataset/` has the pinned Corr2Cause manifest. `python scripts/download_dataset.py` fetches and verifies the CSVs into this folder.
- `results/` contains the published evidence. Local training writes models, caches, and fresh results to `outputs/`.
- `requirements/phase1.txt` and `requirements/phase2.txt` are the install files. The lock files record the original macOS environments.

See the phase guides for full training. The [original Phase 1 GitHub version](https://github.com/Ritik-rao8/AML_DL_Project/tree/7363b7b834ac6c9814eca935ef152651b9d720b0) is preserved in the repository history.
