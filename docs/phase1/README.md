# Phase 1 · Classical machine learning

Each Corr2Cause example contains a premise, a causal hypothesis, and a label. A label of 1 means the hypothesis follows from the premise.

1. **Check the data.** Use the official train, development, and test splits. Inspect the [label counts](../../results/phase1/label_distribution.png), [template mix](eda_plots/templateDistribution.png), and [text lengths](eda_plots/textLengthDistribution.png). The last two charts are from the original submission.
2. **Build features.** Extract [28 features](featureDictionary.md) that describe the variables and patterns in the text.
3. **Train and evaluate.** Compare Logistic Regression, Linear SVM, and Random Forest with a rule-based baseline. Choose settings on development data and report [test macro F1](../../results/phase1/RESULTS.md).
4. **Build graphs and check queries.** Turn selected direct-cause claims into edges and remove cycles. Use a specified synthetic network to check the arithmetic of Bayesian intervention queries.

Open the [executed walkthrough](../../notebooks/01_phase1_classical_ml.ipynb) to see the corrected results. The [original report](Report.pdf), [slides](AML_Phase1_Project_Presentation.pptx), and [submitted scores](original_results.csv) document the earlier Phase 1 submission.

## Run Phase 1

From the repository root on macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/phase1.txt
python scripts/download_dataset.py
python run_phase1.py
python tests/run_tests.py --phase 1
```

If you have a local copy of the official CSVs, place them in `dataset/` before running the download command. It checks their hashes and reuses matching files.

The command reads model settings from [phase1.json](../../configs/phase1.json) and writes local artifacts under `outputs/phase1/`. Use `python run_phase1.py --evaluate-only` to reuse the models from a compatible local run.
