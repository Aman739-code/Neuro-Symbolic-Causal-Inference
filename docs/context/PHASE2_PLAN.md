> Historical project record. File paths and setup details describe the project at the time of writing. See the [current README](../../README.md) for current commands.

# Phase 2 plan: one transformer, one small hybrid experiment, one clear demo

Status: original approved plan; implementation completed on the documented local training budget. See phase2/RUN_AND_EXPLAIN.md, phase2/VERIFICATION.md and outputs/phase2/SUCCESS.json for the actual scope and evidence.

## 1. Outcome and scope

Build a small, reproducible extension of the existing Phase 1 project:

1. DistilBERT reads a premise and a hypothesis and predicts whether the claim follows.
2. Compare its predictions with the existing classical ML baseline.
3. Try one simple ML + DL combination, using existing ML scores.
4. Convert accepted direct-cause claims into separate candidate graphs for each premise.
5. Evaluate those graphs against labelled entailed direct relations.
6. Review the small demonstration cases before approving their graph edges.
7. Demonstrate Bayesian intervention inference with explicitly supplied CPDs and known numerical answers.

Presentation sentence: “The language model judges claims, the graph checks structure, and the Bayesian model answers intervention questions when its causal assumptions and probabilities are supplied.”

The main learned task remains binary premise–hypothesis classification, not unrestricted scientific-text extraction or discovery of a uniquely identifiable real-world DAG. Corr2Cause is the starter reasoning benchmark permitted by the brief. Free-form scientific text and PubMed are optional later extensions.

Keep the same AMLDL project. Add only a few files. Do not add GNNs, NOTEARS, RAG, LLM APIs, multiple transformer families, a web dashboard, or automatic real-world causal-effect claims to this scope.

## 2. Phase 1 recheck and starting evidence

Rechecked the current project before writing this plan:

- Evaluation-only execution completed successfully.
- All 10 behavioral/output tests passed, with no skips.
- Metrics, claims, six graphs and three query answers reproduced exactly across nine files.
- `pip check` reported no broken dependencies.
- Current calibrated ML baseline: test macro F1 0.6357; uncalibrated Random Forest: 0.6055.
- These are corrected results, distinct from the preserved submitted SVM result 0.5907.
- Train: 205,734 rows; dev: 1,076; test: 1,162. Existing leakage exclusions leave 984 development rows for selection.
- This local folder has no Git checkout or configured remote. Restore the intended existing repository before Git-based collaboration; do not create an unrelated remote.

Recheck logs: `outputs/phase1/logs/recheck_evaluate.log` and `recheck_tests.log`.

Two new read-only audits inform this plan:

| Split | Median words | 90th percentile | Maximum words | More than 512 words |
|---|---:|---:|---:|---:|
| Train | 320 | 785 | 2,112 | 50,589 |
| Dev | 216 | 519 | 694 | 130 |
| Test | 192 | 393 | 404 | 0 |

These are whitespace word counts, not tokenizer lengths. Even test examples below 512 words can exceed 512 subword tokens. Exact token lengths must be measured before training.

A second audit found 17 dev premise groups and 18 test premise groups containing labels for every ordered direct-cause variable pair. No contradictory duplicate pair labels were found. Of the 18 test groups, 11 have at least one positive direct edge and seven have none. Apply the existing development leakage mask before deciding which dev groups are eligible for selection. Do not use the test groups to choose settings.

## 3. The smallest architecture that preserves the input

Use one pretrained checkpoint: `distilbert/distilbert-base-uncased`, with its revision pinned when implementation starts. DistilBERT has a default 512-position input limit [2, 3]. The exact pair includes special tokens, premise tokens and hypothesis tokens.

Because long premises are common, use a simple document-level wrapper:

- If the premise–hypothesis pair fits, process it once.
- Otherwise divide the premise into consecutive token windows, reserving room for the complete hypothesis and special tokens in every window.
- Reuse the SAME DistilBERT on every window. This does not create multiple models.
- Take the mean of the valid window representations and pass it through one small classification head.
- Produce one two-class score and one training loss for the entire original example.
- Ignore padded windows in the mean. Do not assign the global label independently to every window.
- Retain all premise tokens. Do not silently discard late windows or remove difficult examples.

Simple explanation: “For a long example, the same reader reads several pages. We combine its page summaries and make one decision.”

This pooling is our proposed implementation, not a claim that DistilBERT natively handles long documents. It preserves coverage but cannot guarantee full logical reasoning between distant windows. Report this limitation. Use the same method for frozen, fine-tuned and hybrid experiments.

Do not introduce a long-context model as a second architecture merely to avoid explaining this averaging step. If the measured memory/time budget is impractical, resolve the budget at the pilot milestone rather than quietly switching protocols.

## 4. Data and split rules

Reuse `src/data.py`, the raw CSVs, content IDs and data manifest. Do not download another large dataset.

- Train model parameters using the official training split.
- Use the same 984 eligible development examples for settings, checkpoint and threshold selection.
- Keep each original document and every derived window in its existing split.
- Never split windows randomly across train/dev/test.
- Keep official test labels out of the training, model-selection and review-decision code paths.
- Save exact sample IDs for any pilot subset. Pilot results are debugging results, not full-data comparison results.
- Cache tokenizer outputs using dataset, tokenizer revision, windowing and code hashes.
- Report label prevalence and token/window count distributions by split.
- Restore original text, premise ID and hypothesis alongside every final prediction.

Phase 1 test results are already known, so call the new work a fixed-protocol evaluation, not a previously unseen benchmark. Freeze the Phase 2 design using dev evidence; do not change it in response to test scores.

## 5. Training: three required training configurations

Use a small default configuration rather than a large search:

| Setting | Starting choice |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 for encoder fine-tuning |
| Epoch limit | 3 |
| Early stopping | Stop after one epoch without improvement in dev macro F1 |
| Weight decay | 0.01 |
| Dropout | Keep checkpoint/classifier defaults; log the actual values |
| Loss | Class-weighted cross entropy, weights from training labels only |
| Gradient clipping | Norm 1.0 |
| Primary seed | 42 |
| Physical batch | Start with 2 documents; profile using actual window counts |
| Effective batch | Aim for 16 documents using accumulation if needed |
| Precision | Use mixed precision only if supported and checked on the chosen device |
| Saved checkpoints | Best checkpoint and final configuration; avoid saving every step |

The frozen-encoder ablation trains only its classification head; choose a documented head learning rate using development data or a fixed small candidate list. It is not meaningful to force its head to use the tiny encoder fine-tuning learning rate.

Required runs:

A. Frozen DistilBERT + trained head + weighted loss.
B. Fine-tuned DistilBERT + weighted loss: primary DL model.
C. Fine-tuned DistilBERT + ordinary loss: imbalance-handling ablation.

Keep all other preprocessing and training-budget rules fixed. Different epoch counts from early stopping are recorded, not hidden.

If run B is unstable or clearly fails to learn, inspect parsing, labels, windows and learning curves first. At most one additional learning rate (5e-5) is a bounded dev-only fallback; record it as an extra experiment. Do not launch a broad search.

If resources allow, repeat the chosen fine-tuned configuration with seeds 7 and 21. Report all three seeds and mean/std; never select the best test seed. If only seed 42 is feasible, report that limit explicitly. Additional seeds are a robustness extension, not a reason to leave the core deliverables unfinished.

## 6. A small actual ML + DL hybrid

The project is naturally neuro-symbolic. However, the rubric specifically discusses combining ML and DL for bonus marks, so symbolic rules alone should not be promised as qualifying.

Try one inexpensive late-fusion experiment:

`hybrid_score = (existing_calibrated_ML_score + DL_score) / 2`

- Reuse the existing calibrated Random Forest, unchanged.
- Use the fine-tuned transformer's positive-class score for the same row.
- Fix the weights at 0.5/0.5. Do not train a meta-model or add stacking.
- Choose a decision threshold using the eligible development set and the same macro-F1 rule used for DL alone.
- Treat this as a combined model score; averaging does not prove probability calibration.
- Cache predictions so the hybrid and downstream graph comparisons require no extra training.

Rationale: the ML branch measures explicit structural counts; the DL branch learns representations from the text. Their errors may complement each other. This is a hypothesis to test.

Plain explanation: “One model checks simple structural clues; one reads the text. We average their scores and check whether the combination helps.”

Compare ML only, DL only and the average. Decide which is the final demo classifier using dev results before test evaluation. If the average does not help, report the negative result and use DL alone. No guaranteed performance improvement or hybrid bonus is claimed. Acyclicity by itself is not evidence that ML + DL fusion helps.

## 7. Graph evaluation without inventing a new dataset

Use the existing full-coverage premise groups to create a small graph benchmark.

For each eligible premise:

1. Parse direct-cause hypotheses into ordered pairs.
2. Deduplicate repeated pair annotations, checking that their labels agree.
3. Require annotations for all `n*(n-1)` ordered non-self pairs.
4. Define reference edges as direct-cause pairs whose Corr2Cause label is 1.
5. Keep label-0 pairs as claims that are not entailed.
6. Build model candidate edges from predicted positive direct-cause pairs only.
7. Evaluate before and after the cycle-removal step using the SAME pair universe.

Critical interpretation: this reference is a graph of ENTAILED direct relations. It is not necessarily the full hidden generating DAG. Label 0 means “not entailed,” which need not mean “impossible in every compatible DAG.” Empty reference graphs are valid and must remain in the evaluation.

Measure:

- Pooled edge precision, recall and F1 across the 18 eligible test systems.
- Per-system TP/FP/FN and counts of proposed/retained edges.
- Number of cyclic candidate graphs and edges removed.
- Number of systems with no proposed edge.
- Coverage and precision together: abstaining everywhere is not a useful result.

For a model with no predicted edges, show precision as undefined/N/A and report its zero coverage; define the aggregate F1 convention explicitly. Keep zero-reference-edge systems in FP accounting.

Use the same graph conversion and cycle rule for ML, DL and hybrid. Reuse `src/relations.py`; preserve provenance. Use the selected classifier's dev threshold for the graph input rather than introducing another independent threshold search.

This extends the earlier Phase 1 statement that full underlying gold DAGs were unavailable: the CSVs still lack those full DAGs, but a labelled reference for entailed direct edges can be assembled from the complete pair groups. Report that narrower claim precisely.

## 8. Reviewed predictions and Bayesian reasoning

Separate automatic evaluation from human review:

- Automatic candidate graphs are scored against held-out reference labels without human edits.
- Choose three presentation cases by a documented rule on dev/demo data: a straightforward case, a case showing structural inconsistency, and an error/uncertain case.
- A reviewer records approved/rejected/uncertain for each proposed demonstration edge, the supporting text and a short reason.
- Reviewers should not consult the test reference labels to make approval decisions. Store review evidence separately from evaluation labels.
- Only approved predictions enter the final demonstration graph used for reasoning.
- Do not describe reviewed results as fully automatic model performance.
- A reviewer cannot insert missing edges and then credit them to the model.

Retain the existing simple relation rules: direct cause becomes an edge; indirect cause, collider and confounder remain annotations unless a named, supported structure is explicitly available. Keep systems separate and reject self-loops.

A directed graph alone does not supply probabilities. Provide the Bayesian stage with an explicit graph AND documented CPDs. For a Corr2Cause graph without observational data/CPDs, display “numerical causal effect unavailable” rather than generating arbitrary probabilities.

Reuse the specified confounded network in `src/bayes.py` as the small independent numeric validation example:

| Query | Expected |
|---|---:|
| P(Y=1 given X=1) | 0.89 |
| P(Y=1 given do(X=1)) | 0.85 |
| P(Y=1 given do(X=0)) | 0.25 |

For one small development/demo case, also connect the approved text-derived graph to pgmpy using a tiny, explicitly supplied CPD file. This is the end-to-end integration demonstration. Choose the case and document the CPDs before final test evaluation; keep the graph small enough to calculate an expected query by hand or independent enumeration. Do not add missing edges and attribute them to the model. If the proposed graph cannot be approved or the CPDs do not match its structure, report that failure and resolve it on development/demo cases before declaring the integration finished.

Explain the evidence tracks clearly in the notebook and slides: learned text/graph quality on Corr2Cause; integration on a reviewed graph with supplied probability assumptions; numerical inference correctness on the specified confounded network. These supplied CPDs are assumptions, not quantities learned from Corr2Cause.

Individual counterfactual estimation needs additional structural assumptions beyond these interventions. Keep it out of the claimed implemented scope unless specifically implemented and validated later.

## 9. Small experiment and results set

Training configurations A/B/C are the only required new training jobs. Reuse cached outputs for these comparisons:

| Comparison | Question |
|---|---|
| Current ML vs fine-tuned DL | Does learned text representation help? |
| Frozen vs fine-tuned encoder | Does adapting encoder weights help? |
| Weighted vs ordinary loss | Does imbalance handling help? |
| ML only vs DL only vs fixed average | Does the hybrid add measured value? |
| Candidate graph before vs after cycle handling | What changes in validity, edge precision and recall? |

The last comparison is a graph-stage ablation, not a classification-F1 improvement claim. Cycle removal can remove correct edges too; measure it.

Keep final presentation results to three tables/figures:

1. Classifier comparison: macro F1, positive precision/recall/F1, accuracy and training cost.
2. Ablation table, with optional seed variability noted.
3. Graph precision/recall/coverage and three query answers.

Save full per-template, variable-count, confusion-matrix and prediction details in the report/output files. Avoid showing a separate slide for every file or model.

## 10. Files and commands to add later

This is a proposed layout, not files already implemented:

```
AMLDL/
  phase2.py                         # prepare/train/evaluate/demo entry point
  configs/phase2.json                # one visible configuration
  src/dl.py                         # windowing, shared encoder, pooling, training
  src/phase2.py                     # experiment orchestration and evaluation
  src/relations.py                  # reused; shared graph logic
  src/bayes.py                      # reused; known query validation
  data/graph_reviews/               # three small documented demo reviews
  notebooks/project_demo.ipynb       # thin calls into src, no copied model code
  outputs/phase1/                   # preserved
  outputs/phase2/                   # predictions, metrics, graphs and best model
  docs/phase2/                      # literature, experiments, report and slides
```

Use one `dl.py` at first; split it only if it becomes hard to read. Do not create folders for each experiment. Keep Phase 2 dependencies in a separate requirements file extending the Phase 1 environment; pin tested versions only after an import/checkpoint smoke test.

Proposed interface:

```
python phase2.py prepare
python phase2.py train --experiment main
python phase2.py train --experiment frozen
python phase2.py train --experiment unweighted
python phase2.py evaluate
python phase2.py demo
```

`evaluate` must reuse saved predictions/model artifacts, not retrain models. `demo` must work from local saved artifacts without internet or GPU training. Save at least dataset/model revisions, config, seed, selected epoch/threshold, environment, prediction IDs, graph audit and expected/actual query answers. Evaluation refuses stale artifacts, as Phase 1 already does.

## 11. Implementation sequence and stop conditions

| Milestone | Work | Completion evidence |
|---|---|---|
| 1. Freeze setup | Restore intended Git checkout, preserve Phase 1, record hardware, pin compatible DL packages | Dependency check; checkpoint loads; Phase 1 tests still pass |
| 2. Prepare inputs | Token audit, complete-window coverage, cached features, graph-reference manifests | No missing hypothesis/premise tokens; no cross-split windows |
| 3. Pilot | One forward/backward pass, then a fixed small training pilot | Loss/gradients valid; weights update; dev predictions nonconstant; measured speed/memory |
| 4. Main DL | Required run B on full official train | Best dev checkpoint, config, curves and saved predictions |
| 5. Ablations/hybrid | Runs A/C; average cached ML/DL scores; freeze selection | Small dev experiment table and final selected settings |
| 6. Final evaluation | Classification, 18-system entailed-edge benchmark, graph ablation | Saved test metrics; all variants reported, no test-based redesign |
| 7. Review and demo | Three reviewed cases, graph visualization, known CPD queries | Notebook works locally; rejected/uncertain claims are visible |
| 8. Submission | IEEE/LaTeX report, slides, viva notes, contribution record | Fresh-clone rerun instructions, tests, rehearsed demo |

Plan for roughly 7–10 focused team sessions plus compute time; this is a planning allowance, not a promised deadline. Measure 100–200 training updates on the actual device and extrapolate from observed documents/windows per second. Do not promise a full run will take a fixed number of hours before profiling.

Prefer one available GPU machine for full training; local CPU is suitable for checks and saved-result demos. If training access is unavailable, arrange that before full runs. Do not buy compute, upload data, or start external jobs merely because this plan mentions a GPU.

If the full dataset is infeasible, document a fixed grouped/stratified training subset and train comparable ML/DL models on that same subset. Report those separately from the existing full-train Phase 1 baseline. A pilot subset must never be silently presented as full-data training.

## 12. Team understanding and presentation

Suggested five-member ownership, independent of names:

1. Data, IDs, token/window coverage and dataset explanation.
2. DistilBERT training and architecture explanation.
3. Evaluation, ablations, fusion experiment and reproducibility.
4. Typed graphs, edge benchmark, review records and Bayesian queries.
5. Notebook integration, IEEE report, slides and demo rehearsal.

Each person reviews another person's work. Everyone must explain the full pipeline; individual viva is explicitly required by the rubric. Track commits and a short contribution table, rather than artificially creating five unrelated components.

A 10-slide narrative:

1. Problem: causal claim validity differs from word matching.
2. Dataset and class imbalance.
3. Phase 1 baseline and the remaining gap.
4. The one-transformer architecture, including long-input averaging.
5. Training and validation rules.
6. ML vs DL results.
7. Ablations and whether fusion helped.
8. One reviewed graph, with a before/after structural example if present.
9. Observing vs intervening: the known numerical example and its assumptions.
10. Limitations, contributions and conclusion.

Prepare a 3–4 minute saved-artifact demo: choose an example, show premise/hypothesis and score, display typed candidate claims and reviewed edges, visualize the graph, then show the separately specified intervention example. Include one failure/abstention example. No live training, network-dependent downloads or additional UI application is needed.

Essential viva answers:

- What do labels 0 and 1 mean? “Not entailed” and “entailed,” not general truth/falsehood.
- Why macro F1? Both classes matter despite imbalance.
- What is fine-tuning? Updating pretrained encoder weights for this task.
- Why windows? The input exceeds the encoder's position limit; we retain it in parts.
- What does a transformer score mean? Model confidence for the classification task, not a causal effect.
- What makes the hybrid justified? Tested complementary structural/text signals, not simply two models connected together.
- Does a DAG prove causality? No; acyclicity is a structural constraint.
- Where do probabilities come from? Explicit CPDs in the known validation example; no empirical CPDs from Corr2Cause.
- What differs between conditioning and intervention? Observation updates belief; intervention changes the assumed mechanism by fixing a variable.

## 13. Rubric coverage and final acceptance

| Requirement | Evidence |
|---|---|
| Architecture logic | One encoder, document pooling, classifier and a clear rationale |
| Hybrid innovation | Fixed ML/DL score combination, ablations, measured benefit or honest negative result |
| Ablation studies | Frozen/fine-tuned, weighted/ordinary loss, fusion and graph-stage comparisons |
| DL literature | BERT, DistilBERT, Corr2Cause, causal graph/intervention foundations |
| Dataset and regularization | Split audit, token coverage, early stopping, dropout, weight decay |
| Technical validation | Per-class test metrics, entailed-edge reference, known numerical queries |
| Theoretical rigor | Attention, cross entropy, fine-tuning, d-separation, uncertainty, conditioning vs do |
| Repository and code | Same project, pinned setup, reusable modules, meaningful tests and contribution log |
| Presentation and viva | Ten-slide narrative, small saved demo, cross-review and individual preparation |
| IEEE/LaTeX report | Complete source and compiled PDF with measured results and limitations |
| Project notebook | Thin notebook comparing ML/DL, visualizing graphs and connecting a reviewed graph plus supplied CPDs to queries |

Before marking Phase 2 complete:

- All Phase 1 tests remain green.
- New tests cover window token coverage, masks/pooling, checkpoint reload parity, ID alignment, reference graph construction, no test-based selection, review gates and missing CPDs.
- No result is claimed for a model/configuration that was not actually run.
- Every variant has the same evaluation examples and documented training budget.
- Reference-edge evaluation includes empty-graph systems and abstention coverage.
- Review-assisted outputs are separated from automatic scores.
- Numerical queries match known answers and state their CPD assumptions.
- Report, plots, slides and notebook use the same saved result files.
- Fresh-setup instructions and a saved-artifact demo are verified.
- No promise of guaranteed DL improvement, hybrid bonus, real-world causal discovery or unsupported counterfactual inference appears in the submission.

## 14. Sources

Local requirements: `../../CSAI_Project_Instructions_Phase_1_and_Phase_2.md` and `../../DL_AML_Project_Ideas.pdf`.

1. [BERT paper](https://arxiv.org/abs/1810.04805).
2. [DistilBERT paper](https://arxiv.org/abs/1910.01108).
3. [Official DistilBERT documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert).
4. [Official padding/truncation documentation](https://huggingface.co/docs/transformers/main/pad_truncation).
5. [Corr2Cause paper](https://arxiv.org/abs/2306.05836).
6. [Official Trainer documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer).
7. [Official gradient accumulation guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/grad_accumulation.md).
8. [Official pgmpy causal inference examples](https://pgmpy.org/examples/Causal_Inference.html).

The window pooling, fixed score average, experiment budget and milestone sequence above are proposed design choices for this project; they are not claims that these sources mandate them.
