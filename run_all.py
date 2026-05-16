from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_features import build_all_features
from src.build_sleep_episodes import build_sleep_episodes, threshold_distribution
from src.evaluate import choose_threshold, probability_metrics, threshold_metrics
from src.feature_audit import META, run_feature_audit
from src.feature_groups import ABLATIONS
from src.imbalance import oversample_minority
from src.load_data import modality_availability, participant_dirs
from src.models import fit_predict, make_model
from src.plots import bar_plot, final_model_plots
from src.split_validation import participant_splitter
from src.utils import ensure_output_dirs, load_config, setup_logger


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("+", "plus").replace("/", "_")


def class_weights(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts()
    return {int(k): float(len(y) / (len(counts) * v)) for k, v in counts.items()}


def feature_list_for_groups(feature_groups: list[str], audit: pd.DataFrame, feature_df: pd.DataFrame) -> list[str]:
    rows = audit[audit["feature_group"].isin(feature_groups) & audit["is_allowed"]]
    return [f for f in rows["feature_name"].tolist() if f in feature_df.columns]


def inner_split(df: pd.DataFrame, train_idx: np.ndarray, seed: int):
    groups = df.iloc[train_idx]["participant_id"].to_numpy()
    y = df.iloc[train_idx]["target_sleep_success_15"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=40, test_size=0.25, random_state=seed)
    fallback = None
    for fit_rel, val_rel in splitter.split(train_idx, y, groups):
        fit_idx = train_idx[fit_rel]
        val_idx = train_idx[val_rel]
        if df.iloc[fit_idx]["target_sleep_success_15"].nunique() == 2 and df.iloc[val_idx]["target_sleep_success_15"].nunique() == 2:
            return fit_idx, val_idx
        fallback = (fit_idx, val_idx)
    return fallback if fallback else (train_idx, train_idx)


def train_fold_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, strategy: str, seed: int):
    cw = None
    if strategy == "class_weight_balanced":
        cw = class_weights(y_train)
        x_use, y_use = x_train.reset_index(drop=True), y_train.reset_index(drop=True)
    elif strategy == "oversample_50":
        x_use, y_use = oversample_minority(x_train, y_train, 1.0, seed)
    elif strategy == "oversample_40":
        x_use, y_use = oversample_minority(x_train, y_train, 0.667, seed)
    else:
        x_use, y_use = x_train.reset_index(drop=True), y_train.reset_index(drop=True)
    model, p_test, _ = fit_predict(model_name, x_use, y_use, x_test, seed, class_weight=cw)
    return model, p_test, x_use, y_use


def oof_train(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    strategy: str,
    cfg: dict,
    save_prefix: str | None = None,
    collect_fold_balance: bool = False,
):
    y = df["target_sleep_success_15"].astype(int)
    splitter, validation_name = participant_splitter(cfg["n_splits"])
    splits = list(splitter.split(df, y, df["participant_id"]))
    p_success = np.zeros(len(df))
    tuned_pred = np.zeros(len(df), dtype=int)
    default_pred = np.zeros(len(df), dtype=int)
    folds = np.zeros(len(df), dtype=int)
    thresholds = []
    curves = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        fit_idx, val_idx = inner_split(df, np.array(train_idx), cfg["random_seed"] + fold)
        _, p_val, _, _ = train_fold_model(
            model_name,
            df.iloc[fit_idx][features],
            y.iloc[fit_idx],
            df.iloc[val_idx][features],
            strategy,
            cfg["random_seed"] + 100 + fold,
        )
        threshold, curve = choose_threshold(y.iloc[val_idx].to_numpy(dtype=int), p_val)
        curve["fold"] = fold
        curve["model_name"] = model_name
        curve["feature_set"] = save_prefix or "feature_set"
        curves.append(curve)
        _, p_test, x_after, y_after = train_fold_model(
            model_name,
            df.iloc[train_idx][features],
            y.iloc[train_idx],
            df.iloc[test_idx][features],
            strategy,
            cfg["random_seed"] + 200 + fold,
        )
        p_success[test_idx] = p_test
        default_pred[test_idx] = (p_test >= 0.5).astype(int)
        tuned_pred[test_idx] = (p_test >= threshold).astype(int)
        folds[test_idx] = fold
        thresholds.append(threshold)
        if collect_fold_balance:
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            fold_rows.append({
                "fold": fold,
                "train_participants": train["participant_id"].nunique(),
                "test_participants": test["participant_id"].nunique(),
                "n_train_before": int(len(train)),
                "n_train_after": int(len(y_after)),
                "n_test": int(len(test)),
                "train_success_rate_before_oversampling": float(train["target_sleep_success_15"].mean()),
                "train_slow_onset_rate_before_oversampling": float(train["target_slow_onset_15"].mean()),
                "train_success_rate_after_oversampling": float(y_after.mean()),
                "train_slow_onset_rate_after_oversampling": float(1 - y_after.mean()),
                "test_success_rate": float(test["target_sleep_success_15"].mean()),
                "test_slow_onset_rate": float(test["target_slow_onset_15"].mean()),
            })
    pred = pd.DataFrame({
        "participant_id": df["participant_id"],
        "bedtime_start_timestamp": df["bedtime_start_timestamp"],
        "onset_latency_minutes": df["onset_latency_minutes"],
        "target_sleep_success_15": y,
        "target_slow_onset_15": 1 - y,
        "predicted_probability_success": p_success,
        "predicted_probability_slow_onset": 1 - p_success,
        "predicted_success_default": default_pred,
        "predicted_success_tuned": tuned_pred,
        "fold": folds,
    })
    if save_prefix:
        pred.to_csv(cfg["output_root"] / "predictions" / f"oof_predictions_{save_prefix}.csv", index=False)
    return pred, thresholds, pd.concat(curves, ignore_index=True), pd.DataFrame(fold_rows), validation_name


def summarize_predictions(pred: pd.DataFrame, thresholds: list[float], model_name: str, feature_set: str, strategy: str, n_features: int) -> dict[str, object]:
    y = pred["target_sleep_success_15"].to_numpy(dtype=int)
    p = pred["predicted_probability_success"].to_numpy(dtype=float)
    default = threshold_metrics(y, p, 0.5, pred["predicted_success_default"].to_numpy(dtype=int))
    tuned = threshold_metrics(y, p, float(np.mean(thresholds)), pred["predicted_success_tuned"].to_numpy(dtype=int))
    probs = probability_metrics(y, p)
    return {
        "model_name": model_name,
        "selected_feature_set": feature_set,
        "target_definition": "target_sleep_success_15 = 1 if SOL < 15 minutes",
        "imbalance_strategy": strategy,
        "n_rows": int(len(pred)),
        "n_participants": int(pred["participant_id"].nunique()),
        "success_count": int(y.sum()),
        "slow_onset_count": int((1 - y).sum()),
        "success_rate": float(y.mean()),
        "slow_onset_rate": float((1 - y).mean()),
        "n_features": int(n_features),
        **probs,
        **{f"{k}_default": v for k, v in default.items()},
        **{f"{k}_tuned": v for k, v in tuned.items()},
        "tuned_threshold_mean": float(np.mean(thresholds)),
        "tuned_threshold_std": float(np.std(thresholds)),
    }


def plot_threshold_distribution(thresholds: pd.DataFrame, output_root: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.bar(thresholds["threshold_minutes"].astype(str), thresholds["slow_onset_rate"], color=["#4c78a8", "#59a14f", "#e15759"])
    plt.ylabel("Slow-onset rate")
    plt.xlabel("Threshold minutes")
    plt.title("Slow-onset class prevalence by threshold")
    for i, v in enumerate(thresholds["slow_onset_rate"]):
        plt.text(i, v + 0.015, f"{v:.1%}", ha="center")
    plt.ylim(0, max(thresholds["slow_onset_rate"]) + 0.12)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "class_distribution_10_15_20.png", dpi=170)
    plt.close()


def plot_threshold_curve(curve: pd.DataFrame, model_name: str, output_root: Path) -> None:
    avg = curve.groupby("threshold", as_index=False)[["precision_success", "recall_success", "f1_success", "precision_slow_onset", "recall_slow_onset", "f1_slow_onset", "balanced_accuracy"]].mean()
    plt.figure(figsize=(8, 5))
    for col in ["precision_success", "recall_success", "f1_success", "precision_slow_onset", "recall_slow_onset", "f1_slow_onset", "balanced_accuracy"]:
        plt.plot(avg["threshold"], avg[col], label=col)
    plt.legend(fontsize=8, ncol=2)
    plt.xlabel("P(success) threshold")
    plt.ylabel("Metric")
    plt.title(f"Threshold curve: {model_name}")
    plt.tight_layout()
    plt.savefig(output_root / "plots" / f"threshold_curve_{model_name}.png", dpi=170)
    plt.close()


def run_ablation(df: pd.DataFrame, audit: pd.DataFrame, cfg: dict):
    rows = []
    prediction_dir = cfg["output_root"] / "predictions"
    all_curves = []
    for name, group_list in ABLATIONS.items():
        features = feature_list_for_groups(group_list, audit, df)
        pred, thresholds, curves, _, _ = oof_train(df, features, "logistic_regression", "oversample_40", cfg, save_prefix=f"ablation_{name}")
        y = pred["target_sleep_success_15"].to_numpy(dtype=int)
        p = pred["predicted_probability_success"].to_numpy(dtype=float)
        probs = probability_metrics(y, p)
        tuned = threshold_metrics(y, p, float(np.mean(thresholds)), pred["predicted_success_tuned"].to_numpy(dtype=int))
        rows.append({
            "feature_set": name,
            "feature_groups": "+".join(group_list),
            "n_features": len(features),
            "missingness": float(df[features].isna().mean().mean()) if features else 0,
            **probs,
            "f1_success": tuned["f1_success"],
            "f1_slow_onset": tuned["f1_slow_onset"],
            "balanced_accuracy": tuned["balanced_accuracy"],
        })
        curves["feature_set"] = name
        all_curves.append(curves)
    out = pd.DataFrame(rows)
    out["selection_score"] = (
        out["pr_auc_slow_onset"].rank(ascending=False)
        + out["f1_slow_onset"].rank(ascending=False)
        + out["brier_score_success"].rank(ascending=True)
        + out["missingness"].rank(ascending=True)
        + out["n_features"].rank(ascending=True) * 0.25
    )
    out = out.sort_values("selection_score")
    out.to_csv(cfg["output_root"] / "tables" / "feature_group_ablation_results.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(cfg["output_root"] / "tables" / "threshold_metrics_ablation.csv", index=False)
    plots = cfg["output_root"] / "plots"
    bar_plot(out.sort_values("feature_set"), "feature_set", "pr_auc_success", plots / "feature_group_ablation_pr_auc_success.png", "Feature ablation PR-AUC success")
    bar_plot(out.sort_values("feature_set"), "feature_set", "pr_auc_slow_onset", plots / "feature_group_ablation_pr_auc_slow_onset.png", "Feature ablation PR-AUC slow onset")
    bar_plot(out.sort_values("feature_set"), "feature_set", "f1_slow_onset", plots / "feature_group_ablation_f1_slow_onset.png", "Feature ablation F1 slow onset")
    bar_plot(out.sort_values("feature_set"), "feature_set", "brier_score_success", plots / "feature_group_ablation_brier.png", "Feature ablation Brier score")
    selected = out.iloc[0]
    selection = pd.DataFrame([{
        "selected_feature_set": selected["feature_set"],
        "selected_feature_groups": selected["feature_groups"],
        "why_selected": "Lowest composite rank across slow-onset PR-AUC, slow-onset F1, Brier score, missingness, and complexity.",
        "which_feature_groups_helped": "See feature_group_ablation_results.csv; helped groups are those improving held-out slow-onset PR-AUC/F1 without large calibration or missingness penalties.",
        "which_feature_groups_did_not_help": "Groups with weaker composite rank or added missingness/complexity were not selected.",
        "did_full_multimodal_improve_performance": bool(out.loc[out["feature_set"].eq("full_clean_multimodal"), "selection_score"].iloc[0] <= selected["selection_score"]),
        "was_added_complexity_missingness_worth_it": "Selected only if ablation ranking justified added complexity.",
    }])
    selection.to_csv(cfg["output_root"] / "tables" / "final_feature_set_selection.csv", index=False)
    return selected["feature_set"], selected["feature_groups"].split("+"), feature_list_for_groups(selected["feature_groups"].split("+"), audit, df), out


def run_standalone_modality_ablation(df: pd.DataFrame, audit: pd.DataFrame, cfg: dict, incremental_results: pd.DataFrame):
    experiments = {
        "temporal_only": ["temporal"],
        "previous_sleep_history_only": ["sleep_history"],
        "oura_activity_readiness_only": ["oura"],
        "samsung_hrv_only": ["samsung_hrv"],
        "samsung_pedometer_only": ["samsung_pedometer"],
        "ema_mood_only": ["ema"],
        "temporal_previous_sleep_history": ["temporal", "sleep_history"],
        "full_clean_multimodal": ["temporal", "sleep_history", "oura", "samsung_hrv", "samsung_pedometer", "ema", "personicle"],
    }
    rows = []
    for name, group_list in experiments.items():
        features = feature_list_for_groups(group_list, audit, df)
        pred, thresholds, _, _, _ = oof_train(
            df,
            features,
            "logistic_regression",
            "oversample_40",
            cfg,
            save_prefix=f"standalone_{name}",
        )
        y = pred["target_sleep_success_15"].to_numpy(dtype=int)
        p = pred["predicted_probability_success"].to_numpy(dtype=float)
        probs = probability_metrics(y, p)
        tuned = threshold_metrics(y, p, float(np.mean(thresholds)), pred["predicted_success_tuned"].to_numpy(dtype=int))
        rows.append({
            "feature_set": name,
            "feature_groups": "+".join(group_list),
            "n_features": len(features),
            "missingness": float(df[features].isna().mean().mean()) if features else 0.0,
            **probs,
            "f1_success": tuned["f1_success"],
            "f1_slow_onset": tuned["f1_slow_onset"],
            "balanced_accuracy": tuned["balanced_accuracy"],
        })
    out = pd.DataFrame(rows).sort_values(["pr_auc_slow_onset", "f1_slow_onset"], ascending=False)
    out.to_csv(cfg["output_root"] / "tables" / "standalone_modality_ablation_results.csv", index=False)
    plots = cfg["output_root"] / "plots"
    bar_plot(out.sort_values("feature_set"), "feature_set", "pr_auc_slow_onset", plots / "standalone_modality_ablation_pr_auc_slow_onset.png", "Standalone modality PR-AUC slow onset")
    bar_plot(out.sort_values("feature_set"), "feature_set", "f1_slow_onset", plots / "standalone_modality_ablation_f1_slow_onset.png", "Standalone modality F1 slow onset")

    inc = incremental_results[["feature_set", "pr_auc_slow_onset", "f1_slow_onset"]].copy()
    inc["analysis_type"] = "incremental"
    stand = out[["feature_set", "pr_auc_slow_onset", "f1_slow_onset"]].copy()
    stand["analysis_type"] = "standalone"
    combined = pd.concat([stand, inc], ignore_index=True)
    combined.to_csv(cfg["output_root"] / "tables" / "standalone_vs_incremental_feature_summary.csv", index=False)
    top = combined.sort_values("pr_auc_slow_onset", ascending=False).head(14).copy()
    labels = top["analysis_type"] + ": " + top["feature_set"]
    y_pos = np.arange(len(top))
    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, top["pr_auc_slow_onset"], color=np.where(top["analysis_type"].eq("standalone"), "#4c78a8", "#59a14f"))
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("PR-AUC slow onset")
    plt.title("Standalone vs incremental feature summary")
    plt.tight_layout()
    plt.savefig(plots / "standalone_vs_incremental_feature_summary.png", dpi=170)
    plt.close()
    return out


def run_imbalance_sensitivity(df: pd.DataFrame, features: list[str], cfg: dict):
    rows = []
    for model in ["logistic_regression", "random_forest", "catboost"]:
        for strategy in ["none", "oversample_40", "oversample_50", "class_weight_balanced"]:
            pred, thresholds, _, _, _ = oof_train(df, features, model, strategy, cfg)
            rows.append(summarize_predictions(pred, thresholds, model, "selected_feature_set", strategy, len(features)))
    out = pd.DataFrame(rows)
    out.to_csv(cfg["output_root"] / "tables" / "imbalance_strategy_comparison.csv", index=False)
    return out


def aggregate_importance(model_name: str, features: list[str], model, x: pd.DataFrame) -> pd.DataFrame:
    if model_name == "catboost":
        return pd.DataFrame({"feature_name": features, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
    names = model.named_steps["prep"].get_feature_names_out()
    raw = model.named_steps["model"].feature_importances_ if model_name == "random_forest" else np.abs(model.named_steps["model"].coef_[0])
    rows = []
    for name, importance in zip(names, raw):
        clean = name.split("__", 1)[-1]
        source = next((f for f in features if clean == f or clean.startswith(f + "_")), clean)
        rows.append({"feature_name": source, "importance": float(importance)})
    return pd.DataFrame(rows).groupby("feature_name", as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def feature_importance_outputs(df: pd.DataFrame, features: list[str], cfg: dict):
    x = df[features]
    y = df["target_sleep_success_15"].astype(int)
    x_over, y_over = oversample_minority(x, y, cfg["main_oversampling_strategy"], cfg["random_seed"])
    for model_name, table, plot_name, title in [
        ("logistic_regression", "logistic_regression_coefficients.csv", "logistic_regression_top_coefficients.png", "Logistic regression top coefficients"),
        ("random_forest", "random_forest_feature_importance.csv", "random_forest_top_features.png", "Random forest top features"),
        ("catboost", "catboost_feature_importance.csv", "catboost_top_features.png", "CatBoost top features"),
    ]:
        model, _, _ = fit_predict(model_name, x_over, y_over, x_over, cfg["random_seed"])
        imp = aggregate_importance(model_name, features, model, x_over)
        if model_name == "logistic_regression":
            imp = imp.rename(columns={"importance": "abs_scaled_coefficient"})
        imp.to_csv(cfg["output_root"] / "tables" / table, index=False)
        val_col = imp.columns[-1]
        top = imp.head(18).sort_values(val_col)
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature_name"], top[val_col])
        plt.xlabel(val_col)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(cfg["output_root"] / "plots" / plot_name, dpi=170)
        plt.close()


def final_ladder(df: pd.DataFrame, selected_name: str, features: list[str], cfg: dict):
    rows = []
    all_curves = []
    fold_balance_written = False
    for model_name in ["logistic_regression", "random_forest", "catboost"]:
        pred, thresholds, curves, fold_balance, validation_name = oof_train(
            df,
            features,
            model_name,
            "oversample_40",
            cfg,
            save_prefix=model_name,
            collect_fold_balance=not fold_balance_written,
        )
        if not fold_balance_written:
            fold_balance.to_csv(cfg["output_root"] / "tables" / "fold_class_balance.csv", index=False)
            fold_balance_written = True
        curves["model_name"] = model_name
        all_curves.append(curves)
        final_model_plots(pred, model_name, cfg["output_root"])
        plot_threshold_curve(curves, model_name, cfg["output_root"])
        rows.append(summarize_predictions(pred, thresholds, model_name, selected_name, "oversample_40_slow_onset", len(features)))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(cfg["output_root"] / "tables" / "final_model_comparison_15min_success.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(cfg["output_root"] / "tables" / "threshold_metrics_by_model.csv", index=False)
    plots = cfg["output_root"] / "plots"
    bar_plot(comparison, "model_name", "pr_auc_success", plots / "model_comparison_pr_auc_success.png", "Model comparison PR-AUC success")
    bar_plot(comparison, "model_name", "pr_auc_slow_onset", plots / "model_comparison_pr_auc_slow_onset.png", "Model comparison PR-AUC slow onset")
    bar_plot(comparison, "model_name", "f1_success_tuned", plots / "model_comparison_f1_success.png", "Model comparison tuned F1 success")
    bar_plot(comparison, "model_name", "f1_slow_onset_tuned", plots / "model_comparison_f1_slow_onset.png", "Model comparison tuned F1 slow onset")
    bar_plot(comparison, "model_name", "brier_score_success", plots / "model_comparison_brier.png", "Model comparison Brier score")
    bar_plot(comparison, "model_name", "expected_calibration_error_success", plots / "model_comparison_ece.png", "Model comparison ECE")
    return comparison, validation_name


def write_readme(cfg: dict, threshold_df: pd.DataFrame, ablation: pd.DataFrame, standalone: pd.DataFrame, selection: pd.DataFrame, comparison: pd.DataFrame, summary: dict):
    selected = selection.iloc[0]
    threshold_md = threshold_df.to_markdown(index=False)
    ablation_md = ablation[["feature_set", "n_features", "missingness", "pr_auc_success", "pr_auc_slow_onset", "brier_score_success", "f1_slow_onset", "selection_score"]].to_markdown(index=False)
    standalone_md = standalone[["feature_set", "n_features", "missingness", "pr_auc_success", "pr_auc_slow_onset", "brier_score_success", "f1_slow_onset", "balanced_accuracy"]].to_markdown(index=False)
    comparison_md = comparison[["model_name", "pr_auc_success", "pr_auc_slow_onset", "brier_score_success", "expected_calibration_error_success", "f1_success_tuned", "f1_slow_onset_tuned"]].to_markdown(index=False)
    text = f"""# Personalized Sleepability Prediction from Wearable Data

## Research Question
Given a person's current/pre-bedtime state and recent history, can we estimate the probability that they will fall asleep quickly?

The model is framed as `P(SOL < 15 minutes)`, which is the sleepability score. Slow onset/failure is still evaluated because it is the practically important minority class.

## Dataset And Raw Loading
The analysis starts from raw participant folders under `../Raw DATA/ifh_affect/`. Participant folders are detected automatically. Oura sleep is used as the anchor table because it contains bedtime and onset latency. Modality availability is saved in `outputs/tables/modality_availability_by_participant.csv`.

## Sleep Episode Construction And Target
Rows with missing, negative, or implausibly extreme onset latency are removed. The primary target is `target_sleep_success_15 = 1` when onset latency is less than 900 seconds and 0 otherwise.

## Threshold Comparison
15 minutes was selected as a compromise between behavioral interpretability and learnability. 10 minutes may be too mild/common, while 20 minutes is stricter but severely imbalanced.

{threshold_md}

## Leakage Prevention
No raw timestamps, participant identifiers, same-night sleep outcomes, same-night onset latency, hypnogram features, or post-bedtime records are used as model inputs. Previous and rolling sleep features are shifted within participant. EMA and sensor windows are restricted to records before bedtime. The feature audit is saved in `outputs/tables/feature_audit.csv`.

## Feature Groups
The fresh pipeline engineered temporal features, previous sleep history, lagged Oura activity/readiness, Samsung HRV windows, Samsung pedometer windows, EMA mood/self-report features, and optional Personicle context. Feature groups were tested through ablation before selecting the final feature set.

## Feature Group Ablation
The final feature set was selected based on participant-held-out performance, calibration, missingness, leakage safety, and interpretability.

{ablation_md}

Selected feature set: `{selected['selected_feature_set']}`. Reason: {selected['why_selected']}

## Standalone Modality Ablation
The standalone modality ablation asks whether each modality has predictive signal by itself. This is different from the incremental ablation above, which asks whether each modality improves the temporal + previous sleep-history baseline. In other words, standalone ablation is about independent signal; incremental ablation is about added value beyond the clean baseline.

{standalone_md}

## Class Imbalance And Validation
Oversampling was applied only inside training folds after participant-aware splitting. The held-out test folds preserve the natural class distribution. The main strategy oversamples slow-onset/failure to approximately 40% of each training fold. Validation uses `{summary['validation_method_used']}`.

## Model Ladder
The final model ladder compares Logistic Regression, Random Forest, and CatBoost on the selected feature set.

{comparison_md}

## Recommendation
Recommended final model: `{summary['recommended_final_model']}`. Random Forest improvement over Logistic Regression: {summary['whether_Random_Forest_improved_over_Logistic_Regression']}. CatBoost improvement over Logistic Regression: {summary['whether_CatBoost_improved_over_Logistic_Regression']}.

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
"""
    (cfg["output_root"].parent / "README.md").write_text(text, encoding="utf-8")


def main():
    cfg = load_config()
    ensure_output_dirs(cfg["output_root"])
    logger = setup_logger(cfg["output_root"])
    logger.info("Starting fresh full 15-minute sleepability analysis")
    participants = participant_dirs(cfg["dataset_root"])
    if not participants:
        raise FileNotFoundError(f"No par_* folders found under {cfg['dataset_root']}")
    availability = modality_availability(participants, cfg["output_root"])
    logger.info("Detected %d participant folders", len(participants))
    episodes = build_sleep_episodes(participants, cfg, logger)
    threshold_df = threshold_distribution(episodes, cfg["output_root"])
    plot_threshold_distribution(threshold_df, cfg["output_root"])
    feature_df, groups, sources = build_all_features(episodes, participants, cfg, logger)
    candidate_features = [c for c in feature_df.columns if c not in META]
    audit = run_feature_audit(candidate_features, groups, sources, cfg["output_root"])
    selected_name, selected_groups, selected_features, ablation = run_ablation(feature_df, audit, cfg)
    selection = pd.read_csv(cfg["output_root"] / "tables" / "final_feature_set_selection.csv")
    standalone = run_standalone_modality_ablation(feature_df, audit, cfg, ablation)
    imbalance = run_imbalance_sensitivity(feature_df, selected_features, cfg)
    comparison, validation_name = final_ladder(feature_df, selected_name, selected_features, cfg)
    feature_importance_outputs(feature_df, selected_features, cfg)
    rec_rows = []
    lr = comparison.set_index("model_name").loc["logistic_regression"]
    rf = comparison.set_index("model_name").loc["random_forest"]
    cb = comparison.set_index("model_name").loc["catboost"]
    rf_improved = bool((rf["pr_auc_success"] > lr["pr_auc_success"]) or (rf["pr_auc_slow_onset"] > lr["pr_auc_slow_onset"]) or (rf["f1_slow_onset_tuned"] > lr["f1_slow_onset_tuned"]))
    cb_improved = bool((cb["pr_auc_success"] > lr["pr_auc_success"]) or (cb["pr_auc_slow_onset"] > lr["pr_auc_slow_onset"]) or (cb["f1_slow_onset_tuned"] > lr["f1_slow_onset_tuned"]))
    recommended = comparison.assign(
        rank_sum=(
            comparison["pr_auc_slow_onset"].rank(ascending=False)
            + comparison["brier_score_success"].rank(ascending=True)
            + comparison["f1_slow_onset_tuned"].rank(ascending=False)
        )
    ).sort_values("rank_sum").iloc[0]["model_name"]
    answers = [
        ("Which feature set was selected and why?", selected_name, selection.iloc[0]["why_selected"]),
        ("Which feature groups helped?", "See ablation table", "Groups with stronger held-out PR-AUC/F1 and acceptable calibration/missingness helped."),
        ("Which feature groups did not help?", "See ablation table", "Groups not selected added less value or more missingness/complexity."),
        ("Did full multimodal improve performance?", str(selection.iloc[0]["did_full_multimodal_improve_performance"]), ""),
        ("Which model has the best PR-AUC for success?", comparison.sort_values("pr_auc_success", ascending=False).iloc[0]["model_name"], ""),
        ("Which model has the best PR-AUC for slow onset/failure?", comparison.sort_values("pr_auc_slow_onset", ascending=False).iloc[0]["model_name"], ""),
        ("Which model has the best F1 for success?", comparison.sort_values("f1_success_tuned", ascending=False).iloc[0]["model_name"], ""),
        ("Which model has the best F1 for slow onset/failure?", comparison.sort_values("f1_slow_onset_tuned", ascending=False).iloc[0]["model_name"], ""),
        ("Which model has the best Brier score?", comparison.sort_values("brier_score_success").iloc[0]["model_name"], ""),
        ("Which model has the best calibration?", comparison.sort_values(["expected_calibration_error_success", "brier_score_success"]).iloc[0]["model_name"], ""),
        ("Which model is most interpretable?", "logistic_regression", ""),
        ("Does Random Forest improve over Logistic Regression?", str(rf_improved), ""),
        ("Does CatBoost improve over Logistic Regression?", str(cb_improved), ""),
        ("Is added model complexity worth it?", str(recommended != "logistic_regression"), ""),
        ("Which model should be used as the final model?", recommended, ""),
        ("Which model should be emphasized in the presentation?", recommended, ""),
    ]
    pd.DataFrame([{"question": q, "answer": a, "notes": n} for q, a, n in answers]).to_csv(cfg["output_root"] / "tables" / "final_recommendation_summary.csv", index=False)
    rates = threshold_df.set_index("threshold_minutes")["slow_onset_rate"].to_dict()
    y = feature_df["target_sleep_success_15"].astype(int)
    summary = {
        "number_of_participants_processed": int(feature_df["participant_id"].nunique()),
        "number_of_valid_sleep_episodes": int(len(feature_df)),
        "ten_minute_slow_onset_rate": float(rates[10]),
        "fifteen_minute_slow_onset_rate": float(rates[15]),
        "twenty_minute_slow_onset_rate": float(rates[20]),
        "final_target_used": "target_sleep_success_15 = 1 if SOL < 15 minutes",
        "success_count": int(y.sum()),
        "slow_onset_count": int((1 - y).sum()),
        "success_rate": float(y.mean()),
        "slow_onset_rate": float((1 - y).mean()),
        "validation_method_used": validation_name,
        "imbalance_strategy_used": "train-fold-only random oversampling to ~40% slow-onset/failure",
        "feature_sets_tested": ", ".join(ABLATIONS.keys()),
        "selected_final_feature_set": selected_name,
        "models_trained": "logistic_regression, random_forest, catboost",
        "best_model_by_PR_AUC_success": comparison.sort_values("pr_auc_success", ascending=False).iloc[0]["model_name"],
        "best_model_by_PR_AUC_slow_onset_failure": comparison.sort_values("pr_auc_slow_onset", ascending=False).iloc[0]["model_name"],
        "best_model_by_F1_success": comparison.sort_values("f1_success_tuned", ascending=False).iloc[0]["model_name"],
        "best_model_by_F1_slow_onset_failure": comparison.sort_values("f1_slow_onset_tuned", ascending=False).iloc[0]["model_name"],
        "best_model_by_Brier_score": comparison.sort_values("brier_score_success").iloc[0]["model_name"],
        "best_calibrated_model": comparison.sort_values(["expected_calibration_error_success", "brier_score_success"]).iloc[0]["model_name"],
        "recommended_final_model": recommended,
        "whether_Random_Forest_improved_over_Logistic_Regression": rf_improved,
        "whether_CatBoost_improved_over_Logistic_Regression": cb_improved,
        "output_folder_location": str(cfg["output_root"].resolve()),
    }
    pd.DataFrame([summary]).to_csv(cfg["output_root"] / "tables" / "final_execution_summary.csv", index=False)
    (cfg["output_root"] / "tables" / "final_execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(cfg, threshold_df, ablation, standalone, selection, comparison, summary)
    print("\nFINAL EXECUTION SUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
