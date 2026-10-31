# Verified experiment results

The experiment ran with seed 42 and the original UCI AI4I 2020 CSV. Six integration tests passed. The saved model reproduces the held-out demonstration scores and its SHAP contributions reconstruct each prediction within 1e-6.

| Model | Test AP | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| Majority dummy | 0.034 | 0.000 | 0.000 | 0.000 |
| Weighted logistic regression | 0.393 | 0.294 | 0.588 | 0.490 |
| Raw random forest | 0.715 | 0.524 | 0.809 | 0.729 |
| Engineered random forest | 0.870 | 0.873 | 0.809 | 0.821 |

Primary test confusion counts: **TN 1,924, FP 8, FN 13, TP 55**. Threshold: **0.288**, selected only from validation F2. Train/validation/test: **6,000 / 2,000 / 2,000** rows. Primary test ROC-AUC: **0.963**.

An independent ordered sensitivity experiment with a separately fixed forest achieved AP **0.782**, precision **0.824**, and recall **0.718**. It changes both split and model configuration and uses CSV order as a proxy, so it cannot establish future-time performance or isolate the split effect.

AI4I is synthetic. These are current-reading classification results. There is no validated early-warning horizon or real-factory performance claim. The audit also records **27** disagreements between outcome flags and the overall target, which remains unchanged.

For exact unrounded values, confidence intervals, subgroup results, dependency versions and source checksum, see `results/metrics.json`. Every slide and report result derives from those executed artifacts. Run `python -m unittest -v test_project.py` on the evaluation computer before presenting.
