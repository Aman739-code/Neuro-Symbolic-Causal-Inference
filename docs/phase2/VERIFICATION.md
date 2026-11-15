# Phase 2 verification

- The published training sample contains 2,100 rows from 25 premise groups. The eligible development and official test sets contain 984 and 1,162 rows.
- The original run reloaded all three checkpoints on CPU. The largest score difference was below 0.00000012. [Checkpoint validation](../../results/phase2/checkpoint_validation.json)
- Repeated evaluation reproduced nine outputs, including classification, graph, selection, and prediction results. [Repeatability record](../../results/phase2/repeatability.json)
- The test suite recomputes all six classification reports from the published predictions CSV. It also checks score arrays, graph coverage, and the saved Bayesian answers. [Test predictions](../../results/phase2/test_predictions.csv)
- The executable [demo notebook](../../notebooks/02_phase2_deep_learning.ipynb) rebuilds the reviewed graph and computes the Bayesian answers from the supplied probability tables. [Current assumptions file](../../configs/cpd_assumptions.json)

The [original verification record](../../results/phase2/verification.json) preserves the original run signature and file checksums. Its notebook checksum belongs to the version before the folder reorganization.
