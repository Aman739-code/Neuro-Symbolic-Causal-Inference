# Phase 2 · Deep learning and hybrid

1. **Match the training data.** Train a Random Forest and DistilBERT on the same 2,100 examples from whole premise groups. Use the full-data Phase 1 model as a separate reference.
2. **Test the design.** Compare the fine-tuned model with a frozen encoder and with ordinary, unweighted loss. Average the matched ML and fine-tuned DL scores for the hybrid.
3. **Evaluate and explain.** Choose thresholds on development data, report test macro F1, and review direct-cause edges. Replay a Bayesian query with explicitly supplied probabilities.

Open the [executed walkthrough](../../notebooks/02_phase2_deep_learning.ipynb), [results](../../results/phase2/RESULTS.md), [report](Report.pdf), or [slides](Phase%202.pptx). The [original verification record](../../results/phase2/verification.json) summarizes checkpoint and repeatability checks for the published run.

## Run the saved demo

Use the [setup commands from the README](../../README.md#run-the-saved-phase-2-demo). Then run `python run_phase2.py demo` from the repository root. The demo reads `results/phase2/` directly, and `python scripts/replay_demo.py` rebuilds the reviewed graph and recomputes its query answers.

## Reproduce the experiments

Run the commands from the repository root using a Python 3.13 virtual environment. Phase 2 preparation uses the Phase 1 feature cache and the calibrated full-data baseline, so complete Phase 1 first.

```bash
python -m pip install -r requirements/phase2.txt
python scripts/download_dataset.py
python run_phase1.py
python scripts/download_model.py
python run_phase2.py prepare
python run_phase2.py pilot
python run_phase2.py train --experiment main
python run_phase2.py train --experiment unweighted
python run_phase2.py train --experiment frozen --device cpu
python run_phase2.py evaluate
python run_phase2.py demo --local
python tests/run_tests.py --phase 2
```

Use `--device cpu`, `--device mps`, or `--device cuda` to select a device. Completed compatible runs are reused unless you pass `--force`. Fresh metrics, predictions, and checkpoints are written under `outputs/phase2/`.

The [split manifest](../../results/phase2/data_manifest.json) records the training sample. The [per-example predictions](../../results/phase2/test_predictions.csv) contain the label, score, and prediction for all six reported variants on each test row.
