"""Generate report and slide content from actual run artifacts, never hand-enter metrics."""
from pathlib import Path
import json, csv, html
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
r=json.loads((ROOT/'results/metrics.json').read_text())
m=next(x for x in r['models'] if x['model']==r['primary'])['test']
a=r['audit'];b=r['ordered_holdout'];ci=r['confidence_intervals']
fmt=lambda x:f'{x:.3f}'
pct=lambda x:f'{100*x:.1f}%'
table='| Model | AP | ROC-AUC | Precision | Recall | F1 | F2 | Threshold |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for row in r['models']:
    q=row['test'];table+='| '+row['model']+' | '+' | '.join(fmt(q[k]) for k in ['ap','roc_auc','precision','recall','f1','f2','threshold'])+' |\n'
ci_table='| Metric | Estimate | 95% bootstrap interval |\n|---|---:|---:|\n'+'\n'.join(f'| {k} | {fmt(m[k])} | {fmt(v[0])}–{fmt(v[1])} |' for k,v in ci.items())
subtable='| Type | Test n | Failures | Precision | Recall | AP |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {k} | {v['n']} | {v['tp']+v['fn']} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['ap'])} |" for k,v in r['subgroups'].items())
cvtable='| Model | Training CV mean AP | Fold SD |\n|---|---:|---:|\n'+'\n'.join(f"| {v['model']} | {fmt(v['mean_ap'])} | {fmt(v['std_ap'])} |" for v in r['cv'])
report=f'''# Explainable AI-Based Predictive Maintenance for Early Detection of Industrial Machine Failure

AUCML End Semester Evaluation · 26–30 October 2026

## Abstract

This project implements an explainable failure-screening prototype using UCI's AI4I 2020 synthetic predictive-maintenance benchmark. It compares a majority baseline, weighted logistic regression, a random forest on the original inputs, and a tuned forest with engineering features. A fixed, stratified 60/20/20 partition separates fitting, threshold selection and final testing. The primary engineered forest achieved test average precision {fmt(m['ap'])}, precision {pct(m['precision'])}, and recall {pct(m['recall'])}. At the validation-selected threshold {fmt(m['threshold'])}, it detected {m['tp']} of {m['tp']+m['fn']} labelled failures, missed {m['fn']}, and produced {m['fp']} false alerts among 2,000 test readings. SHAP explanations reconstruct individual model scores. The data supports current-reading classification; it does not establish real-factory performance or advance warning before a future event.

## 1. Problem statement and objectives

Maintenance teams need to identify unusual operating conditions while controlling missed failures and excessive alerts. The objective is to build a reproducible classifier and provide understandable explanations for its output. The prototype accepts product type, air temperature, process temperature, rotational speed, torque and tool wear. It returns a failure score, a threshold-based alert and SHAP contributions for human review.

The approved title expresses the long-term aim of early detection. The implemented target is the supplied failure indicator for the same record. There is no validated future horizon, machine trajectory, lead-time label or remaining-useful-life target. We therefore evaluate snapshot classification and treat genuine early warning as future work.

## 2. Dataset and provenance

The original CSV was downloaded from the UCI Machine Learning Repository, dataset 601, DOI 10.24432/C5HS5C, licensed CC BY 4.0. AI4I is synthetic and was designed to resemble industrial conditions. It is not a collection of actual factory measurements. The public source makes this classroom experiment independently reproducible.

The downloaded file has {a['rows']:,} rows and {a['failures']} positive labels ({pct(a['prevalence'])}). It contains {a['missing']} missing values and {a['feature_duplicates']} duplicate rows across the six allowed inputs. Its SHA-256 digest is `{a['sha256']}`.

| Input | Meaning | Unit |
|---|---|---|
| Type | Product quality category L, M or H | Category |
| Air temperature | Ambient air temperature | K |
| Process temperature | Process temperature | K |
| Rotational speed | Shaft rotational speed | rpm |
| Torque | Applied torque | Nm |
| Tool wear | Accumulated tool-use measure | min |

The target is `Machine failure`. UDI and Product ID are excluded. TWF, HDF, PWF, OSF and RNF are outcome-related flags and are also excluded. An audit found {a['flag_target_disagreements']} rows where the OR of those flags differs from the published target. We preserve the original target and record this inconsistency rather than silently relabeling records.

## 3. Experimental design and leakage controls

The fixed seed is {r['seed']}. Stratified random sampling produces 6,000 training rows ({a['splits']['train']['failures']} failures), 2,000 validation rows ({a['splits']['validation']['failures']} failures), and 2,000 test rows ({a['splits']['test']['failures']} failures). The split manifest preserves the membership of every UDI. All three partitions have disjoint row IDs and zero identical allowed-input rows across their boundaries.

Preprocessing lives inside each fitted pipeline, so it is fitted separately within every training fold. Numeric scaling is used for logistic regression. Product type uses one-hot encoding. No imputation is needed for this complete benchmark. No oversampling occurs. Outcome flags and IDs never enter any model. The same sensor-only transformation is shared by training and the demo.

Five stratified folds within the training partition compare AP. The engineered forest searches four combinations: maximum depth 8 or unrestricted and minimum leaf size 1 or 3, with 250 trees and balanced bootstrap class weights. The winning settings are `{r['best_params']}`. The engineered forest is the predeclared primary model, not a choice made by inspecting test performance. The training CV result for its winning configuration is selection-biased and is not treated as an independent generalization estimate.

{cvtable}

Each fitted non-dummy model uses validation predictions to select the threshold maximizing F2. Ties choose the highest threshold. F2 = 5PR / (4P + R), expressing an illustrative preference for failure recall. This is a classroom criterion, not a measured industrial cost function. The dummy uses an always-negative decision at threshold 1.0. We retain the models fitted on the 6,000 training rows after threshold selection, so the demo matches the evaluated model. We do not refit on validation or test data.

## 4. Algorithms and feature engineering

The dummy assigns the training prevalence as every score. Weighted logistic regression is an interpretable linear baseline in log-odds. The raw random forest averages 250 randomized trees, with minimum leaf size 2 and balanced bootstrap class weights. The engineered forest adds three deterministic sensor-only features:

- Temperature difference = process temperature − air temperature, in K.
- Mechanical power = torque × rotational speed × 2π/60, in W.
- Wear–torque interaction = tool wear × torque, in min Nm.

These transformations use the published benchmark mechanism and general engineering knowledge. They do not use outcome columns or per-row labels, but knowledge of the synthetic generator can make the benchmark easier. Performance on this dataset is not proof that those engineered relations transfer to real machines. The exact source and transformations are disclosed for this reason.

## 5. Held-out test results

AP (average precision) is the primary ranking metric because positive cases are rare. It is a recall-weighted summary of precision, not a trapezoidal PR area. We report ROC-AUC as a secondary ranking measure and precision, recall, F1 and F2 at the locked validation threshold. Accuracy alone would favour predicting the majority class.

{table}

Primary model confusion matrix, where rows are actual labels and columns are predicted labels:

| Actual / predicted | No alert | Alert |
|---|---:|---:|
| No failure | {m['tn']} | {m['fp']} |
| Failure | {m['fn']} | {m['tp']} |

The primary model's accuracy is {pct(m['accuracy'])}, balanced accuracy is {pct(m['balanced_accuracy'])}, and Brier score is {fmt(m['brier'])}. Lower Brier score is better, but a single value does not establish calibration. Its predicted scores are not presented as calibrated event probabilities. Fixed-threshold 0.5 comparisons are preserved in `results/metrics.json`.

![Held-out precision-recall curves](../results/figures/precision_recall.png)

## 6. Uncertainty and robustness

{ci_table}

These percentile intervals use 500 stratified bootstrap resamples of the fixed test predictions and preserve the observed class counts. They condition on one fitted model and do not include model-selection variability, training variability or unseen-factory uncertainty. Row dependence can also make them optimistic. There are only {m['tp']+m['fn']} failures in the main test set, so a small number of mistakes changes recall appreciably.

An independent ordered sensitivity experiment trains on the first 6,000 rows, selects an F2 threshold on the next 2,000, and tests on the final 2,000. It uses a separately predeclared forest with depth 8 and leaf size 3, rather than settings selected by random-test results. It has failure counts {b['split_failures']} in train, validation and test. Its AP is {fmt(b['ap'])}, precision {pct(b['precision'])}, recall {pct(b['recall'])}, F2 {fmt(b['f2'])}, and confusion counts TN={b['tn']}, FP={b['fp']}, FN={b['fn']}, TP={b['tp']}. It is a combined change of split and fixed configuration, not a controlled estimate of the split effect alone. Row order is a sensitivity proxy, not a verified timeline.

Subgroup results for the primary random test partition:

{subtable}

These small subgroup samples are descriptive and cannot establish fairness or reliable performance across unseen product categories. The target population here remains the synthetic benchmark.

## 7. Explainability

The implementation uses the SHAP library's exact TreeExplainer with the tree-path-dependent feature-dependence assumption. For this scikit-learn random forest, the explained class-1 output is the forest score in probability units. The baseline is {r['shap_base']:.6f}; each feature's signed contribution adds to or subtracts from this reference. Code asserts that baseline plus all contributions matches `predict_proba` within 1e-6.

The tree-path reference reflects the forest's weighted training path counts. Balanced class weighting makes this baseline approximately 50%, which differs from the observed 3.39% failure prevalence. It must not be read as population failure prevalence.

Global importance is mean absolute SHAP magnitude over 200 randomly selected held-out readings, without class balancing. It describes this sample and model, not causal importance. Local explanations include saved true-positive, true-negative, false-positive and false-negative examples when those categories occur. One-hot type indicators and engineered features are shown explicitly, so correlated original and derived features may divide attribution.

![Global SHAP importance](../results/figures/shap_global.png)

No claim is made that changing a feature by its SHAP contribution changes physical failure risk by that amount. The explanation is for model output. Maintenance decisions require domain verification.

## 8. Implementation and demo

`core.py` defines sensor-only engineering, preprocessing and SHAP inference. `train.py` runs the experiment and writes audit evidence, scores, figures and the saved model. `app.py` provides a local Flask application that loads that exact model. The browser form supplies product type and five numeric readings. The API rejects extra fields, invalid types, non-finite values and readings outside training ranges. Range checks do not prove that a combination is physically plausible.

The UI shows a fixed-threshold alert, model score and signed SHAP contributions. It labels synthetic data, the current-reading target, lack of calibration and explanation limits. A saved false-negative case helps demonstrate honest failure analysis. The app listens only on localhost and is an educational demonstration, not a production maintenance or control system.

`test_project.py` checks the saved model and API, SHAP reconstruction, leakage-column rejection, invalid inputs, partition accounting and page availability. The test transcript is included with the results. Run it again on the presentation computer.

## 9. Conclusions and future work

The project delivers a reproducible benchmark experiment and a working explanation interface. At the chosen threshold, the primary model detects {m['tp']} failures while missing {m['fn']} and generating {m['fp']} false alerts in the held-out test set. Its value is an auditable example of imbalanced classification and model explanation. Synthetic labels, limited positive counts, correlated inputs, selection uncertainty and unknown industrial transfer prevent operational reliability claims.

The next research step is real machine telemetry with machine IDs, timestamps and maintenance logs. Define the prediction horizon before constructing labels, ensure features precede the event, split by machine and forward time, and measure lead time and alert burden. Then assess calibration, drift and costs with maintenance staff. A second independent factory or machine family is needed to evaluate transfer.

## 10. Submission and member contributions

The faculty's evaluation window is 26–30 October 2026. The exact group session and membership are not yet known. Verify the official group list and PRNs before submission. All members must attend and report at least 10 minutes early, with code, demo and slides ready. The separate viva guide contains a contribution worksheet. Fill it with actual completed work and evidence, not suggested roles presented as facts.

## References

1. AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository (2020). https://doi.org/10.24432/C5HS5C. Dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset. CC BY 4.0.
2. SHAP documentation, TreeExplainer. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html.
3. Scikit-learn documentation, Metrics and scoring. https://scikit-learn.org/stable/modules/model_evaluation.html.

## Reproduction record

Python {r['python']}. Seed {r['seed']}. Dependency versions: {json.dumps(r['versions'])}. Source digest and row split membership are preserved. `results/metrics.json` is the machine-readable source for all reported numeric results. The files in this submission were generated using coding assistance and should be independently studied and verified by each presenting member.
'''
(OUT/'PROJECT_REPORT.md').write_text(report,encoding='utf-8')

slides=[]
def add(title,lines=None,note='',kind='text',**extra):
    slides.append(dict(title=title,lines=lines or [],notes=note,kind=kind,**extra))
add('Explainable AI-Based Predictive Maintenance', ['For Early Detection of Industrial Machine Failure','AUCML End Semester Evaluation · 26–30 October 2026','Implementation: current-reading classification on AI4I 2020'],kind='cover',note='Approved full title: Explainable AI-Based Predictive Maintenance for Early Detection of Industrial Machine Failure. Add the verified member names and PRNs before submission. The experiment does not validate a future warning horizon.')
add('Problem and scope',['Rare failures make missed detections easy to hide behind high accuracy.','The prototype turns six input fields into a failure score and a review alert.','AI4I supports current-reading classification. Early warning remains a research objective.'],note='Explain the practical reason for maintenance screening. State the scope before showing performance. No lead-time or remaining-useful-life claim.')
add('Dataset and target',[f"{a['rows']:,} readings · {a['failures']} failures · {pct(a['prevalence'])} positive",'Inputs: type, air and process temperature, speed, torque and wear.','UCI AI4I 2020 is a synthetic industrial benchmark.','Target: Machine failure for the same reading.'],note='Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset. DOI 10.24432/C5HS5C. CC BY 4.0. Explain the difference between a published dataset and real factory measurements.')
add('Leakage and data-quality controls',['Exclude UDI, Product ID and all five failure-type flags.','Fit preprocessing inside each training fold.','No overlapping rows or identical input rows across partitions.',f"{a['flag_target_disagreements']} flag/target disagreements documented. Published target retained."],note='Flags TWF/HDF/PWF/OSF/RNF encode outcomes. IDs may encode shortcuts. The full audit, excluded field list and source checksum are in results/metrics.json. Zero duplicate feature rows across the three partitions.')
add('Evaluation protocol',['Train: 6,000 rows. Five-fold AP selects forest settings.','Validation: 2,000 rows. F2 selects the alert threshold.','Test: 2,000 rows. All model and threshold choices are fixed.','No final refit: the saved demo model is the evaluated model.'],note=f"Seed 42. Positive counts train/validation/test: {a['splits']['train']['failures']}/{a['splits']['validation']['failures']}/{a['splits']['test']['failures']}. The primary engineered forest was predeclared. F2 emphasizes recall. Selection uses only train/validation.")
add('Models and engineering features',['Baselines: majority dummy, weighted logistic regression and raw random forest.','Primary: 250-tree forest with four depth/leaf configurations.','Added features: temperature difference, mechanical power and wear × torque.','Engineering uses sensor inputs and published benchmark knowledge.'],note=f"Selected parameters: {r['best_params']}. Mechanical power = torque × rpm × 2π/60. No generator failure thresholds are copied into the features. Benchmark-informed engineering limits novelty and transfer claims.")
add('Held-out model comparison',kind='table',headers=['Model','AP','Precision','Recall','F2'],rows=[[x['model']]+[fmt(x['test'][k]) for k in ['ap','precision','recall','f2']] for x in r['models']],lines=['Each non-dummy model uses its own validation-selected threshold.'],note='Source: results/metrics.json, models[*].test. AP is average precision, not trapezoidal PR-AUC. Dummy predicts no failures at threshold 1.0. ROC-AUC and F1 are in the report.')
add('Primary model: detected and missed failures',kind='table',headers=['Actual / predicted','No alert','Alert'],rows=[['No failure',str(m['tn']),str(m['fp'])],['Failure',str(m['fn']),str(m['tp'])]],lines=[f"Recall {pct(m['recall'])} · Precision {pct(m['precision'])} · Threshold {fmt(m['threshold'])}",f"{m['fn']} missed failures and {m['fp']} false alerts remain."],note='Source: results/metrics.json. Test n=2,000. Explain the cost of a false negative and the workload from a false positive. Do not describe the score as calibrated failure probability.')
add('Average precision across models',kind='chart',categories=[x['model'] for x in r['models']],values=[x['test']['ap'] for x in r['models']],lines=['Higher is better. Test prevalence provides the constant-score AP baseline.'],note='Source: results/metrics.json. All bars use the same held-out random test partition. Editable native chart. Threshold choice does not change AP.')
add('Robustness and uncertainty',[f"Primary AP: {fmt(m['ap'])}, bootstrap interval {fmt(ci['ap'][0])}–{fmt(ci['ap'][1])}.",f"Primary recall interval: {pct(ci['recall'][0])}–{pct(ci['recall'][1])}.",f"Ordered holdout: AP {fmt(b['ap'])}, recall {pct(b['recall'])}, precision {pct(b['precision'])}.",'Ordered rows are a sensitivity check, with a separate fixed forest.'],note='500 stratified bootstrap resamples of fixed test predictions. Intervals omit training variability and may be optimistic under dependence. Ordered experiment uses first 6000 / next 2000 / last 2000 rows, depth 8 and leaf size 3. This changes both split and model configuration. No verified timestamp or future-machine claim.')
imp=list(csv.DictReader((ROOT/'results/shap_importance.csv').open()))[:6]
add('Global SHAP importance',kind='shapchart',categories=[x['feature'] for x in imp],values=[float(x['mean_abs_shap']) for x in imp],lines=['Mean absolute contributions across 200 randomly selected test readings.'],note='Source: results/shap_importance.csv. TreeExplainer uses tree_path_dependent. Contributions refer to the forest class-1 score. Correlated and engineered features can share attribution. Not causal feature importance.')
ex=json.loads((ROOT/'results/demo_examples.json').read_text())['true_positive']
top=sorted(ex['shap'].items(),key=lambda x:abs(x[1]),reverse=True)[:3]
add('An explanation for one held-out failure',[f"UDI {ex['udi']}: actual failure, model score {pct(ex['score'])}.",f"Reference score {pct(ex['base'])} + all contributions = {pct(ex['score'])}."]+[f"{k}: {v*100:+.2f} percentage points" for k,v in top],note='Source: results/demo_examples.json, true_positive. Slide lists only the three largest effects; the UI shows all features. The reconstruction statement includes every contribution. SHAP explains prediction, not the physical cause. Additivity is checked to 1e-6.')
add('Live demonstration',['Load a saved true-positive case and analyse its sensor readings.','Explain the fixed threshold and signed SHAP contributions.','Load the false-negative case to show a missed failure.','Change a valid input and discuss model sensitivity.'],note='Run python app.py, open http://127.0.0.1:5000. The target displayed for sample cases is not submitted to the classifier. Input fields are checked against training ranges. If the computer fails, use saved result files and figures.')
add('Conclusions and next steps',[f"Verified primary test result: AP {fmt(m['ap'])}, recall {pct(m['recall'])}.",'The code, saved model, explanations and demo reproduce the experiment.','Real early warning needs timestamped histories and future-event labels.','Next: machine/time splits, external validation, calibration and alert-cost analysis.'],note='Do not claim reduced downtime, improved factory safety or real-time production readiness without operational trials. Current evidence is a synthetic benchmark with current-reading labels.')
add('Individual contribution evidence',['Each member explains work they actually completed and verified.','Evidence: dataset audit, split checks, model results, SHAP or demo tests.','Complete the member/PRN worksheet before submission.','Confirm the allotted slot and report at least 10 minutes early.'],note='Faculty instructions supplied by the user: evaluation October 26–30, 2026; all members present; individual viva; presentation, code and demo ready. No group list or exact slot has been supplied. Do not invent member names or contributions.')
add('References',['UCI AI4I 2020 dataset · DOI 10.24432/C5HS5C · CC BY 4.0','SHAP documentation · TreeExplainer','Scikit-learn documentation · Metrics and scoring','Project evidence · results/metrics.json and saved predictions'],note='https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\nhttps://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html\nhttps://scikit-learn.org/stable/modules/model_evaluation.html\nAll reported project metrics come from executed code.')
(OUT/'slides_content.json').write_text(json.dumps(slides,indent=2,ensure_ascii=False),encoding='utf-8')
print('Generated report and',len(slides),'slides from measured results')
