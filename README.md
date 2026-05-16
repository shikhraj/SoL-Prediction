# Personalized Sleepability Prediction from Wearable Data

## Research Question
Given a person's current/pre-bedtime state and recent history, can we estimate the probability that they will fall asleep quickly?

The model is framed as `P(SOL < 15 minutes)`, which is the sleepability score. Slow onset/failure is still evaluated because it is the practically important minority class.

## Dataset And Raw Loading
The analysis starts from raw participant folders under `../Raw DATA/ifh_affect/`. Participant folders are detected automatically. Oura sleep is used as the anchor table because it contains bedtime and onset latency. Modality availability is saved in `outputs/tables/modality_availability_by_participant.csv`.

## Sleep Episode Construction And Target
Rows with missing, negative, or implausibly extreme onset latency are removed. The primary target is `target_sleep_success_15 = 1` when onset latency is less than 900 seconds and 0 otherwise.

## Threshold Comparison
15 minutes was selected as a compromise between behavioral interpretability and learnability. 10 minutes may be too mild/common, while 20 minutes is stricter but severely imbalanced.

|   threshold_minutes | slow_onset_definition             |   slow_onset_count |   quick_onset_count |   total_count |   slow_onset_rate |   quick_onset_rate | interpretation                                                   |
|--------------------:|:----------------------------------|-------------------:|--------------------:|--------------:|------------------:|-------------------:|:-----------------------------------------------------------------|
|                  10 | slow_onset_10 = SOL >= 10 minutes |               1748 |                2249 |          3997 |          0.437328 |           0.562672 | Milder/common threshold; useful for balance but may be too easy. |
|                  15 | slow_onset_15 = SOL >= 15 minutes |               1076 |                2921 |          3997 |          0.269202 |           0.730798 | Primary compromise between behavioral meaning and learnability.  |
|                  20 | slow_onset_20 = SOL >= 20 minutes |                475 |                3522 |          3997 |          0.118839 |           0.881161 | Stricter delayed-SOL threshold; more imbalanced.                 |

## Leakage Prevention
No raw timestamps, participant identifiers, same-night sleep outcomes, same-night onset latency, hypnogram features, or post-bedtime records are used as model inputs. Previous and rolling sleep features are shifted within participant. EMA and sensor windows are restricted to records before bedtime. The feature audit is saved in `outputs/tables/feature_audit.csv`.

## Feature Groups
The fresh pipeline engineered temporal features, previous sleep history, lagged Oura activity/readiness, Samsung HRV windows, Samsung pedometer windows, EMA mood/self-report features, and optional Personicle context. Feature groups were tested through ablation before selecting the final feature set.

## Feature Group Ablation
The final feature set was selected based on participant-held-out performance, calibration, missingness, leakage safety, and interpretability.

| feature_set                  |   n_features |   missingness |   pr_auc_success |   pr_auc_slow_onset |   brier_score_success |   f1_slow_onset |   selection_score |
|:-----------------------------|-------------:|--------------:|-----------------:|--------------------:|----------------------:|----------------:|------------------:|
| temporal_sleep_history       |           18 |    0.00415589 |         0.823558 |            0.411278 |              0.201145 |        0.433673 |             10.5  |
| temporal_sleep_oura          |           34 |    0.0134586  |         0.812873 |            0.395126 |              0.203865 |        0.45728  |             13.75 |
| temporal_sleep_hrv           |           54 |    0.487625   |         0.816911 |            0.406604 |              0.203361 |        0.464615 |             15    |
| temporal_sleep_pedometer     |           60 |    0.16326    |         0.819938 |            0.39721  |              0.203319 |        0.442607 |             16.25 |
| temporal_sleep_hrv_pedometer |           96 |    0.375547   |         0.816358 |            0.393315 |              0.205517 |        0.45298  |             21.75 |
| temporal_only                |            6 |    0          |         0.753785 |            0.317128 |              0.212426 |        0.415    |             24.25 |
| temporal_sleep_ema           |           68 |    0.0163321  |         0.786537 |            0.370464 |              0.210956 |        0.417312 |             24.5  |
| full_clean_multimodal        |          172 |    0.244852   |         0.788309 |            0.350471 |              0.217638 |        0.44407  |             27    |

Selected feature set: `temporal_sleep_history`. Reason: Lowest composite rank across slow-onset PR-AUC, slow-onset F1, Brier score, missingness, and complexity.

## Standalone Modality Ablation
The standalone modality ablation asks whether each modality has predictive signal by itself. This is different from the incremental ablation above, which asks whether each modality improves the temporal + previous sleep-history baseline. In other words, standalone ablation is about independent signal; incremental ablation is about added value beyond the clean baseline.

| feature_set                     |   n_features |   missingness |   pr_auc_success |   pr_auc_slow_onset |   brier_score_success |   f1_slow_onset |   balanced_accuracy |
|:--------------------------------|-------------:|--------------:|-----------------:|--------------------:|----------------------:|----------------:|--------------------:|
| temporal_previous_sleep_history |           18 |    0.00415589 |         0.823558 |            0.411278 |              0.201145 |        0.433673 |            0.579765 |
| previous_sleep_history_only     |           12 |    0.00623384 |         0.822159 |            0.406728 |              0.20193  |        0.451977 |            0.59664  |
| full_clean_multimodal           |          172 |    0.244852   |         0.788309 |            0.350471 |              0.217638 |        0.44407  |            0.595169 |
| temporal_only                   |            6 |    0          |         0.753785 |            0.317128 |              0.212426 |        0.415    |            0.558636 |
| oura_activity_readiness_only    |           16 |    0.0239242  |         0.764549 |            0.300959 |              0.217844 |        0.425982 |            0.565877 |
| samsung_pedometer_only          |           42 |    0.231447   |         0.753452 |            0.290657 |              0.218415 |        0.369639 |            0.526545 |
| samsung_hrv_only                |           36 |    0.72936    |         0.750461 |            0.27805  |              0.215822 |        0.428571 |            0.53948  |
| ema_mood_only                   |           50 |    0.0207155  |         0.7232   |            0.277757 |              0.226396 |        0.250717 |            0.46092  |

## Class Imbalance And Validation
Oversampling was applied only inside training folds after participant-aware splitting. The held-out test folds preserve the natural class distribution. The main strategy oversamples slow-onset/failure to approximately 40% of each training fold. Validation uses `StratifiedGroupKFold`.

## Model Ladder
The final model ladder compares Logistic Regression, Random Forest, and CatBoost on the selected feature set.

| model_name          |   pr_auc_success |   pr_auc_slow_onset |   brier_score_success |   expected_calibration_error_success |   f1_success_tuned |   f1_slow_onset_tuned |
|:--------------------|-----------------:|--------------------:|----------------------:|-------------------------------------:|-------------------:|----------------------:|
| logistic_regression |         0.823558 |            0.411278 |              0.201145 |                            0.123419  |           0.634417 |              0.433673 |
| random_forest       |         0.826581 |            0.397972 |              0.196437 |                            0.102776  |           0.703963 |              0.464512 |
| catboost            |         0.794688 |            0.365874 |              0.203191 |                            0.0878276 |           0.698532 |              0.402239 |

## Recommendation
Recommended final model: `random_forest`. Random Forest improvement over Logistic Regression: True. CatBoost improvement over Logistic Regression: False.

## Calibration
Because the output is a probability, Brier score and expected calibration error are reported. If a nonlinear model improves PR-AUC or F1 but worsens calibration, that trade-off should be considered before product use.

## Top Features
Feature interpretation tables and plots are saved for Logistic Regression, Random Forest, and CatBoost. Logistic regression coefficients indicate features associated with higher or lower P(success); tree importances are descriptive, not causal.

## Limitations And App Readiness
This remains a prototype and is not clinically validated. The dataset is small and observational, device labels are noisy, and held-out participant results may not generalize to a real deployment. A real app would require prospective validation, better calibration, uncertainty reporting, and user-safety review.

## Rerun
From this folder:

```bash
python scripts/run_all.py
```

From the repository root:

```bash
python sleepability_15min_full_fresh_analysis/scripts/run_all.py
```

## Output Guide
Tables are in `outputs/tables/`, plots in `outputs/plots/`, out-of-fold predictions in `outputs/predictions/`, logs in `outputs/logs/`, and model artifacts if saved are in `outputs/models/`.

## Advanced Model Diagnostics

### Logistic Regression Odds-Ratio Interpretation
An additional statsmodels Logit model was fit using the selected temporal + previous sleep-history feature set. Predictors were median-imputed and standardized, so odds ratios describe the change in odds of sleep success for a one-standard-deviation increase in a feature. Cluster-robust standard errors by participant are saved because repeated sleep episodes are not fully independent.

Diagnostics saved under `outputs/advanced_model_diagnostics/` include likelihood-ratio testing, McFadden pseudo R² (0.072), AIC/BIC, Wald z-tests, Hosmer-Lemeshow calibration, and VIF. Statistical significance should not be oversold because this is repeated-measures wearable data.

### Random Forest Hyperparameter Tuning
The original Random Forest used `n_estimators=300`, `max_depth=5`, and `min_samples_leaf=5`, which were regularized defaults chosen to reduce overfitting. A nested participant-aware tuning analysis searched n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, and criterion using slow-onset PR-AUC as the primary objective.

Original RF slow-onset PR-AUC was 0.398; tuned RF slow-onset PR-AUC was 0.388. Original RF slow-onset F1 was 0.465; tuned RF slow-onset F1 was 0.435. Original RF Brier was 0.196; tuned RF Brier was 0.198. The tuning report explains whether the improvement is large enough to change the final model recommendation.
## HRV-Enhanced Final Model
The core model remains the behavioral baseline: temporal features plus previous sleep history. A new HRV-enhanced analysis was added under `outputs/final_hrv_enhanced_model/` to test whether selected pre-bedtime Samsung HRV features improve sleepability prediction.

HRV was included because autonomic recovery and pre-bedtime arousal are theoretically relevant to sleep onset. The enhanced model adds a compact 3-hour pre-bedtime HRV subset, including heart-rate summary statistics, RMSSD, SDNN, record count, and HRV availability/missingness indicators.

HRV missingness was handled with median imputation fit only inside training folds, plus missingness indicators. Oversampling still occurs only after participant-aware splitting, and held-out participants preserve the natural class distribution.

| feature_set              | model_name          |   pr_auc_slow_onset |   f1_slow_onset_tuned |   recall_slow_onset_tuned |   lift_at_top_20_percent_risk |   brier_score_success |   expected_calibration_error_success |
|:-------------------------|:--------------------|--------------------:|----------------------:|--------------------------:|------------------------------:|----------------------:|-------------------------------------:|
| core_behavioral_baseline | logistic_regression |            0.411278 |              0.433673 |                  0.63197  |                       1.66696 |              0.201145 |                            0.123419  |
| core_behavioral_baseline | random_forest       |            0.397972 |              0.464512 |                  0.614312 |                       1.53231 |              0.196437 |                            0.102776  |
| core_behavioral_baseline | catboost            |            0.365874 |              0.402239 |                  0.500929 |                       1.42087 |              0.203191 |                            0.0878276 |
| hrv_enhanced             | logistic_regression |            0.409633 |              0.442819 |                  0.618959 |                       1.68089 |              0.203177 |                            0.126473  |
| hrv_enhanced             | random_forest       |            0.404152 |              0.453807 |                  0.545539 |                       1.6066  |              0.19712  |                            0.109147  |
| hrv_enhanced             | catboost            |            0.385069 |              0.449825 |                  0.656134 |                       1.61124 |              0.199022 |                            0.0801759 |

Recommendation: Use both: core behavioral baseline as the clean reference, and HRV-enhanced model as a physiological sensitivity model.

## Extended Model Benchmark
An additional benchmark was added under `outputs/new_model_benchmark/` to test whether model families beyond Logistic Regression, Random Forest, and CatBoost improve the 15-minute sleepability task. The benchmark evaluates both the core behavioral feature set and the HRV-enhanced feature set using the same participant-aware validation, train-fold-only oversampling, and leakage controls.

Models tested include ExtraTrees, sklearn GradientBoosting, sklearn HistGradientBoosting, a shallow MLP, and an RBF SVM, plus the original reference models. Optional XGBoost, LightGBM, and EBM were checked and documented if unavailable.

| feature_set     | model_name          |   pr_auc_slow_onset |   f1_slow_onset_tuned |   brier_score_success |   expected_calibration_error_success |   lift_at_top_20_percent_risk |
|:----------------|:--------------------|--------------------:|----------------------:|----------------------:|-------------------------------------:|------------------------------:|
| core_behavioral | logistic_regression |            0.411278 |              0.433673 |              0.201145 |                            0.123419  |                       1.66696 |
| hrv_enhanced    | logistic_regression |            0.409633 |              0.442819 |              0.203177 |                            0.126473  |                       1.68089 |
| core_behavioral | ebm                 |            0.406441 |              0.429175 |              0.201572 |                            0.117717  |                       1.61589 |
| hrv_enhanced    | gradient_boosting   |            0.406411 |              0.431501 |              0.197642 |                            0.100795  |                       1.65768 |
| hrv_enhanced    | extra_trees         |            0.405309 |              0.460231 |              0.200726 |                            0.114929  |                       1.64839 |
| hrv_enhanced    | random_forest       |            0.404152 |              0.453807 |              0.19712  |                            0.109147  |                       1.6066  |
| hrv_enhanced    | rbf_svm             |            0.402311 |              0.418988 |              0.198768 |                            0.0985513 |                       1.66232 |
| hrv_enhanced    | ebm                 |            0.402161 |              0.416766 |              0.204239 |                            0.118615  |                       1.61589 |

Final benchmark note: Overall recommended benchmark model is random_forest with core_behavioral features.

