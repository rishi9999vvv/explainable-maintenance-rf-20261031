# AUCML presentation and individual viva preparation

Project: Explainable AI-Based Predictive Maintenance for Early Detection of Industrial Machine Failure

## Presentation rehearsal

Allow about 10–12 minutes plus questions. Adjust to your department's allotted duration once confirmed.

1. Problem and scope (1 minute): explain why rare failures and missed alerts matter. State that this implementation classifies the current reading on a synthetic benchmark. True advance warning remains future work.
2. Data and leakage controls (2 minutes): identify six input variables, the target, excluded IDs and outcome flags, and the class imbalance.
3. Method and algorithms (2 minutes): explain the split, five-fold cross-validation, the baseline models, engineering features, and validation threshold.
4. Results and limitations (2 minutes): show average precision, precision/recall, the confusion matrix, confidence intervals, and ordered holdout.
5. Explainability and demo (2–3 minutes): run one true positive and one false negative. Explain score reconstruction and why SHAP is not causal evidence.
6. Conclusion and contributions (1 minute): state the verified result and what would be needed before deployment. Each member explains work they actually completed.

## Technical viva: concise answers

**1. What problem does your implementation solve?**
Binary classification of the machine-failure label for a sensor snapshot. It supports screening and human review. The dataset does not establish a forecast horizon or remaining useful life.

**2. Is AI4I real factory data?**
No. It is a public, reproducible synthetic benchmark from UCI, designed to reflect industrial maintenance patterns. We do not describe it as measurements from an actual factory.

**3. Why keep “early detection” in the title?**
It is the approved research aim. The implemented evidence is current-reading classification. Demonstrating early warning needs timestamped trajectories, a future outcome window, and evaluation of lead time.

**4. What is leakage here?**
TWF, HDF, PWF, OSF and RNF are outcome-related flags. Feeding them to a failure classifier reveals the answer. IDs may encode ordering or shortcuts. We exclude all these fields and the target from the feature pipeline.

**5. Why have three partitions?**
Training fits preprocessing and models and supports cross-validation. Validation selects the threshold. Test estimates performance after those choices are fixed. Repeatedly choosing settings using test performance invalidates the holdout.

**6. Why use stratification?**
It preserves approximately the same failure prevalence in each random partition. It does not make adjacent or related readings independent, so an ordered sensitivity analysis is also included.

**7. Why not call the ordered split a future-time validation?**
The CSV order is available, but verified timestamps and machine identities are absent. The split tests sensitivity to order and changing conditions, without proving future-fleet performance.

**8. Why is accuracy misleading?**
An always-negative classifier can score very high accuracy because failures are rare, while detecting no failures. Recall, precision, average precision, F2 and the confusion matrix expose this failure.

**9. What is average precision (AP)?**
AP summarizes precision at successive recall increments over thresholds. It is not the trapezoidal area under a precision-recall curve. We report the scikit-learn AP implementation consistently.

**10. Why report ROC-AUC too?**
It summarizes ranking of positives versus negatives. With severe imbalance a small false-positive rate can still create many alerts, so it is a secondary metric here.

**11. What does F2 emphasize?**
F2 = 5PR / (4P + R). It places greater weight on recall than precision. Maximizing validation F2 is a declared classroom preference, not an established factory cost policy.

**12. Why change the 0.5 threshold?**
The best operational threshold depends on error costs, prevalence and model scoring behaviour. We select it using validation F2 and lock it before test evaluation. The results also record the standard 0.5 threshold for comparison.

**13. Why use class weighting?**
Balanced weights make failures contribute more to fitting. We do not oversample before splitting. Weighting can affect score calibration, so the demo calls its output a score.

**14. How do logistic regression and a random forest differ?**
Logistic regression learns a linear relationship in log-odds after scaling. A random forest averages randomized trees that can model nonlinear relationships and interactions. It is more complex to interpret directly.

**15. What engineered features did you add?**
Process minus air temperature in kelvin, torque times angular speed in watts, and wear times torque. They use only allowed sensor inputs. Their relevance is informed by domain knowledge and the published benchmark mechanism, which limits how novel or transferable the result is.

**16. Is using benchmark-informed engineering leakage?**
It is not direct target-column leakage, because no per-row outcomes enter the transform. It is prior knowledge of a synthetic generator, which can make the task unusually favourable. We disclose this and need external real-data validation.

**17. What model settings were searched?**
Only maximum depth (8 or unrestricted) and minimum leaf size (1 or 3) for the engineered forest, with 250 trees. Five-fold training AP chooses the configuration. The small search makes the experiment easy to reproduce and defend.

**18. What does SHAP explain?**
It allocates a model prediction relative to a reference output across features using Shapley-value principles. Positive contributions increase the predicted failure score and negative ones decrease it. Our code checks that baseline plus contributions reconstructs the prediction.

**19. Does SHAP identify the physical cause of failure?**
No. It explains model behaviour under specified feature-dependence assumptions. Correlated sensor readings and derived features can share attribution. A technician must verify a physical diagnosis.

**Why is the SHAP reference approximately 50% when failures are rare?**
The tree-path-dependent reference uses weighted tree path counts. Balanced class weights make positive and negative examples contribute comparably during fitting. This model reference is not the observed population prevalence, which is 3.39% in the dataset.

**20. Can the output be interpreted as a calibrated failure probability?**
No calibration claim is established. We report a Brier score as a diagnostic, but have not performed a dedicated calibration study. A random-forest score of 0.8 is not proof that exactly 80% of such machines fail.

**21. What do the confidence intervals mean?**
They are percentile intervals from 500 stratified bootstrap resamples of the fixed test predictions. They condition on the fitted model and observed class counts. They omit training variability, dependence among rows and uncertainty about new factories.

**22. What is a false negative in this project?**
A labelled failure with score below the chosen threshold. The demo includes an observed false negative if one exists. We show this limitation instead of demonstrating only successful cases.

**23. What about conflicting failure flags?**
The audit counts rows where the OR of the five flags disagrees with the overall target. We retain the published overall target and exclude all flags from learning. Corrections would require a separate, justified experiment.

**24. What does reproducibility mean here?**
The project preserves the CSV hash, split membership, seed, dependency versions, selected parameters, scores, model artifact and commands. Small numerical differences can occur across platforms or dependency versions.

**25. How would you extend this into actual early warning?**
Collect machine IDs and ordered sensor histories. Define an actionable horizon, such as failure within the next operating shift, before constructing labels. Use only past information, split by machine and time, and evaluate recall, alert burden and warning lead time on future machines.

**26. How would deployment differ from the demo?**
Validate sensor units and missing readings, assess calibration and drift, choose a threshold with maintenance staff, monitor alert workload, document responses and test on external machines. The current local demo has no machine-control authority.

**27. Why not use deep learning?**
There are only 10,000 tabular records and no validated sequence structure for the target. Tree models provide a strong, reproducible starting point with modest compute and useful explanations.

**28. What is your personal contribution?**
Describe only the tasks you actually completed or independently verified. Open the relevant file, explain an implementation choice, reproduce its result, and discuss a limitation. Do not claim authorship of automatically generated work without disclosure consistent with your institution's policy.

## Member contribution worksheet

Names, PRNs, group size and actual contributions have not been provided. The roles below are suggestions, not completed contribution claims. Combine or split them to match the real group.

| Workstream | Suggested evidence to demonstrate | Member / PRN | Actual work completed and date |
|---|---|---|---|
| Problem and dataset | Data source, units, source limitations, CSV hash | To complete | To complete |
| Leakage and experiment design | Excluded columns, split manifest, duplicate checks | To complete | To complete |
| Models and evaluation | Run training, explain CV and threshold, reconcile confusion matrix | To complete | To complete |
| Explainability and demo | Explain a SHAP reconstruction and run an error case | To complete | To complete |
| Integration and presentation | Reproduce setup, rehearse slides, assemble references | To complete | To complete |

## Faculty checklist

- Presentation window: 26–30 October 2026. The exact group slot is unconfirmed.
- Verify members, PRNs and approved title against the official group list when available.
- All group members must attend and arrive at least 10 minutes before the assigned session.
- Keep the slides, project folder, saved model and dataset locally available before reporting.
- Run the demo and rehearse one individual technical explanation per member.
- Keep a copy of the figures and results as a fallback if the demo computer fails.

## Demonstration sequence

1. Launch the local app and show the source/scope statement.
2. Select `true_positive`. Explain that the displayed target is for demonstration and is never sent to the model.
3. Select Analyse reading. Explain the threshold, score, top SHAP effects, and baseline reconstruction.
4. Select `false_negative`, if available. Discuss why a missed failure matters and why lower thresholds cause more alerts.
5. Change one valid input and rerun. Treat this as a model sensitivity experiment, not a causal intervention.
6. Show `results/metrics.csv` and match the test confusion counts to the presentation.

## Last-minute self-check

Every member should be able to explain the difference between train, validation and test; distinguish precision from recall; name the leakage columns; state the synthetic-data and early-warning limitations; and show their own verified contribution.
