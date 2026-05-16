from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import choose_threshold, probability_metrics, threshold_metrics
from src.feature_groups import FEATURE_GROUPS
from src.imbalance import oversample_minority
from src.models import fit_predict
from src.split_validation import participant_splitter
from src.utils import load_config


ANALYSIS_NAME = "final_hrv_enhanced_model"
MODEL_NAMES = ["logistic_regression", "random_forest", "catboost"]
CORE_FEATURES = FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["sleep_history"]
PREFERRED_HRV_FEATURES = [
    "samsung_hrv_3h_count_records",
    "samsung_hrv_3h_hr_mean",
    "samsung_hrv_3h_hr_std",
    "samsung_hrv_3h_hr_min",
    "samsung_hrv_3h_hr_max",
    "samsung_hrv_3h_hrv_rmssd_mean",
    "samsung_hrv_3h_hrv_rmssd_std",
    "samsung_hrv_3h_hrv_sdnn_mean",
    "samsung_hrv_3h_hrv_sdnn_std",
]
FALLBACK_HRV_FEATURES = [
    "samsung_hrv_6h_count_records",
    "samsung_hrv_6h_hr_mean",
    "samsung_hrv_6h_hrv_rmssd_mean",
    "samsung_hrv_6h_hrv_sdnn_mean",
]


def make_output_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "tables": base / "tables",
        "plots": base / "plots",
        "predictions": base / "predictions",
        "logs": base / "logs",
        "models": base / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def safe_model_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def select_hrv_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    selected = [f for f in PREFERRED_HRV_FEATURES if f in df.columns]
    used_fallback = []
    count = "samsung_hrv_3h_count_records"
    enough_3h = count in df.columns and df[count].notna().mean() >= 0.20
    if len(selected) < 4 or not enough_3h:
        used_fallback = [f for f in FALLBACK_HRV_FEATURES if f in df.columns]
        selected = list(dict.fromkeys(selected + used_fallback))
    return selected, used_fallback


def add_hrv_missingness_indicators(df: pd.DataFrame, hrv_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    indicators = []
    if "samsung_hrv_3h_count_records" in out.columns:
        out["has_hrv_3h"] = out["samsung_hrv_3h_count_records"].fillna(0).gt(0).astype(int)
        indicators.append("has_hrv_3h")
    if "samsung_hrv_6h_count_records" in out.columns:
        out["has_hrv_6h"] = out["samsung_hrv_6h_count_records"].fillna(0).gt(0).astype(int)
        indicators.append("has_hrv_6h")
    for feature in hrv_features:
        if feature.endswith("_count_records"):
            continue
        indicator = f"{feature}_missing"
        out[indicator] = out[feature].isna().astype(int)
        indicators.append(indicator)
    return out, indicators


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


def train_fold_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, seed: int):
    x_use, y_use = oversample_minority(x_train, y_train, sampling_strategy=0.667, random_state=seed)
    model, p_test, _ = fit_predict(model_name, x_use, y_use, x_test, random_state=seed, class_weight=None)
    return model, p_test, x_use, y_use


def oof_train(df: pd.DataFrame, features: list[str], model_name: str, feature_set_name: str, out_dirs: dict[str, Path], seed: int, n_splits: int):
    y = df["target_sleep_success_15"].astype(int)
    splitter, validation_name = participant_splitter(n_splits)
    splits = list(splitter.split(df, y, df["participant_id"]))
    p_success = np.zeros(len(df))
    default_pred = np.zeros(len(df), dtype=int)
    tuned_pred = np.zeros(len(df), dtype=int)
    folds = np.zeros(len(df), dtype=int)
    thresholds = []
    curves = []
    balance_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        fit_idx, val_idx = inner_split(df, train_idx, seed + fold)
        _, p_val, _, _ = train_fold_model(
            model_name,
            df.iloc[fit_idx][features],
            y.iloc[fit_idx],
            df.iloc[val_idx][features],
            seed + 100 + fold,
        )
        threshold, curve = choose_threshold(y.iloc[val_idx].to_numpy(dtype=int), p_val)
        curve["fold"] = fold
        curve["model_name"] = model_name
        curve["feature_set"] = feature_set_name
        curves.append(curve)
        _, p_test, x_after, y_after = train_fold_model(
            model_name,
            df.iloc[train_idx][features],
            y.iloc[train_idx],
            df.iloc[test_idx][features],
            seed + 200 + fold,
        )
        p_success[test_idx] = p_test
        default_pred[test_idx] = (p_test >= 0.5).astype(int)
        tuned_pred[test_idx] = (p_test >= threshold).astype(int)
        folds[test_idx] = fold
        thresholds.append(float(threshold))
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        balance_rows.append({
            "feature_set": feature_set_name,
            "model_name": model_name,
            "fold": fold,
            "train_participants": int(train["participant_id"].nunique()),
            "test_participants": int(test["participant_id"].nunique()),
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
    pred.to_csv(out_dirs["predictions"] / f"oof_predictions_{feature_set_name}_{model_name}.csv", index=False)
    return pred, thresholds, pd.concat(curves, ignore_index=True), pd.DataFrame(balance_rows), validation_name


def top20_risk_metrics(y_success, p_success) -> dict[str, float]:
    y = np.asarray(y_success, dtype=int)
    risk = 1 - np.asarray(p_success, dtype=float)
    y_slow = 1 - y
    n_top = max(1, int(np.ceil(0.20 * len(y))))
    top_idx = np.argsort(-risk)[:n_top]
    slow_in_top = y_slow[top_idx].sum()
    precision = float(slow_in_top / n_top)
    recall = float(slow_in_top / max(1, y_slow.sum()))
    base_rate = float(y_slow.mean())
    lift = float(precision / base_rate) if base_rate > 0 else np.nan
    return {
        "slow_onset_recall_at_top_20_percent_risk": recall,
        "slow_onset_precision_at_top_20_percent_risk": precision,
        "lift_at_top_20_percent_risk": lift,
    }


def summarize_predictions(pred: pd.DataFrame, thresholds: list[float], model_name: str, feature_set_name: str, n_features: int) -> dict[str, object]:
    y = pred["target_sleep_success_15"].to_numpy(dtype=int)
    p = pred["predicted_probability_success"].to_numpy(dtype=float)
    default = threshold_metrics(y, p, 0.5, pred["predicted_success_default"].to_numpy(dtype=int))
    tuned = threshold_metrics(y, p, float(np.mean(thresholds)), pred["predicted_success_tuned"].to_numpy(dtype=int))
    return {
        "feature_set": feature_set_name,
        "model_name": model_name,
        "target_definition": "target_sleep_success_15 = 1 if SOL < 15 minutes",
        "imbalance_strategy": "train-fold-only random oversampling to ~40% slow-onset/failure",
        "n_rows": int(len(pred)),
        "n_participants": int(pred["participant_id"].nunique()),
        "success_count": int(y.sum()),
        "slow_onset_count": int((1 - y).sum()),
        "success_rate": float(y.mean()),
        "slow_onset_rate": float((1 - y).mean()),
        "n_features": int(n_features),
        **probability_metrics(y, p),
        **{f"{k}_default": v for k, v in default.items()},
        **{f"{k}_tuned": v for k, v in tuned.items()},
        **top20_risk_metrics(y, p),
        "tuned_threshold_mean": float(np.mean(thresholds)),
        "tuned_threshold_std": float(np.std(thresholds)),
    }


def plot_metric_bars(comparison: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    plot_specs = [
        ("pr_auc_slow_onset", "core_vs_hrv_pr_auc_slow_onset.png", "PR-AUC slow onset"),
        ("f1_slow_onset_tuned", "core_vs_hrv_f1_slow_onset.png", "Tuned F1 slow onset"),
        ("recall_slow_onset_tuned", "core_vs_hrv_recall_slow_onset.png", "Tuned recall slow onset"),
        ("brier_score_success", "core_vs_hrv_brier.png", "Brier score for P(success)"),
        ("expected_calibration_error_success", "core_vs_hrv_ece.png", "Expected calibration error"),
        ("lift_at_top_20_percent_risk", "top20_risk_lift_comparison.png", "Top 20% risk lift"),
    ]
    for metric, filename, title in plot_specs:
        pivot = comparison.pivot(index="model_name", columns="feature_set", values=metric)
        ax = pivot.plot(kind="bar", figsize=(8, 4.8))
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / filename, dpi=170)
        plt.close()


def plot_hrv_missingness(df: pd.DataFrame, features: list[str], out_dirs: dict[str, Path]) -> None:
    if not features:
        return
    sample = df[features].isna().astype(int)
    if len(sample) > 800:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, 800).astype(int)]
    plt.figure(figsize=(10, 5))
    plt.imshow(sample.T, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(np.arange(len(features)), features, fontsize=7)
    plt.xlabel("Sleep episodes, sampled in chronological table order")
    plt.title("Missingness heatmap for selected HRV features")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "hrv_missingness_heatmap.png", dpi=170)
    plt.close()


def final_prediction_plots(pred: pd.DataFrame, model_name: str, feature_set_name: str, out_dirs: dict[str, Path]) -> None:
    y = pred["target_sleep_success_15"].astype(int)
    p = pred["predicted_probability_success"].astype(float)
    stem = f"{feature_set_name}_{model_name}"
    RocCurveDisplay.from_predictions(y, p)
    plt.title(f"ROC: {feature_set_name} {model_name}")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / f"roc_curve_{stem}.png", dpi=170)
    plt.close()
    PrecisionRecallDisplay.from_predictions(1 - y, 1 - p)
    plt.title(f"PR curve slow onset: {feature_set_name} {model_name}")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / f"pr_curve_slow_onset_{stem}.png", dpi=170)
    plt.close()
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean, frac, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted P(success)")
    plt.ylabel("Observed success rate")
    plt.title(f"Calibration: {feature_set_name} {model_name}")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / f"calibration_curve_{stem}.png", dpi=170)
    plt.close()
    ConfusionMatrixDisplay.from_predictions(
        y,
        pred["predicted_success_tuned"].astype(int),
        labels=[0, 1],
        display_labels=["slow onset", "success"],
    )
    plt.title(f"Tuned confusion matrix: {feature_set_name} {model_name}")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / f"confusion_matrix_tuned_{stem}.png", dpi=170)
    plt.close()


def aggregate_importance(model_name: str, features: list[str], model) -> pd.DataFrame:
    if model_name == "catboost":
        return pd.DataFrame({"feature_name": features, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
    names = model.named_steps["prep"].get_feature_names_out()
    if model_name == "random_forest":
        raw = model.named_steps["model"].feature_importances_
        label = "importance"
    else:
        raw = model.named_steps["model"].coef_[0]
        label = "scaled_coefficient"
    rows = []
    for name, importance in zip(names, raw):
        clean = name.split("__", 1)[-1]
        source = next((f for f in features if clean == f or clean.startswith(f + "_")), clean)
        rows.append({"feature_name": source, label: float(importance)})
    return pd.DataFrame(rows).groupby("feature_name", as_index=False)[label].sum().sort_values(label, ascending=False)


def feature_group(feature: str) -> str:
    if feature in FEATURE_GROUPS["temporal"]:
        return "temporal"
    if feature in FEATURE_GROUPS["sleep_history"]:
        return "sleep_history"
    if feature.startswith("samsung_hrv") or feature.startswith("has_hrv"):
        return "HRV"
    return "other"


def fit_full_importance_models(df: pd.DataFrame, features: list[str], out_dirs: dict[str, Path], seed: int) -> pd.DataFrame:
    x_over, y_over = oversample_minority(df[features], df["target_sleep_success_15"].astype(int), 0.667, seed)
    grouped_rows = []
    for model_name in MODEL_NAMES:
        model, _, _ = fit_predict(model_name, x_over, y_over, x_over, random_state=seed)
        imp = aggregate_importance(model_name, features, model)
        if model_name == "logistic_regression":
            imp["abs_scaled_coefficient"] = imp["scaled_coefficient"].abs()
            imp.to_csv(out_dirs["tables"] / "hrv_enhanced_logistic_coefficients.csv", index=False)
        else:
            imp.to_csv(out_dirs["tables"] / f"hrv_enhanced_{model_name}_feature_importance.csv", index=False)
            top = imp.head(18).sort_values("importance")
            plt.figure(figsize=(8, 6))
            plt.barh(top["feature_name"], top["importance"])
            plt.xlabel("importance")
            plt.title(f"HRV-enhanced feature importance: {model_name}")
            plt.tight_layout()
            plt.savefig(out_dirs["plots"] / f"hrv_enhanced_feature_importance_{model_name}.png", dpi=170)
            plt.close()
        value_col = "importance" if "importance" in imp.columns else "abs_scaled_coefficient"
        tmp = imp.copy()
        tmp["feature_group"] = tmp["feature_name"].map(feature_group)
        grouped = tmp.groupby("feature_group", as_index=False)[value_col].sum()
        total = grouped[value_col].sum()
        grouped["group_importance_fraction"] = grouped[value_col] / total if total else np.nan
        grouped["model_name"] = model_name
        grouped = grouped.rename(columns={value_col: "group_importance"})
        grouped_rows.append(grouped)
    return pd.concat(grouped_rows, ignore_index=True)


def write_feature_tables(df: pd.DataFrame, hrv_features: list[str], hrv_indicators: list[str], used_fallback: list[str], out_dirs: dict[str, Path]) -> None:
    rows = []
    for feature in CORE_FEATURES:
        if feature in df.columns:
            rows.append({
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "missing_rate": float(df[feature].isna().mean()),
                "reason_included": "Core behavioral baseline feature retained for comparison.",
                "leakage_status": "Allowed: derived from bedtime timing or previous sleep episodes only.",
            })
    for feature in hrv_features + hrv_indicators:
        rows.append({
            "feature_name": feature,
            "feature_group": "HRV",
            "missing_rate": float(df[feature].isna().mean()),
            "reason_included": "Compact pre-bedtime HRV signal; 3-hour window preferred." + (" Six-hour fallback added because 3-hour coverage was sparse." if feature in used_fallback else ""),
            "leakage_status": "Allowed: computed only from Samsung HRV records before bedtime_start_timestamp.",
        })
    pd.DataFrame(rows).to_csv(out_dirs["tables"] / "hrv_enhanced_feature_list.csv", index=False)
    miss = []
    for feature in hrv_features:
        indicators = [i for i in hrv_indicators if i == f"{feature}_missing" or i in ["has_hrv_3h", "has_hrv_6h"]]
        miss.append({
            "feature_name": feature,
            "missing_rate": float(df[feature].isna().mean()),
            "imputation_strategy": "Median imputation fit inside each training fold through the model pipeline.",
            "missingness_indicator_used": "; ".join(indicators) if indicators else "No dedicated indicator.",
        })
    pd.DataFrame(miss).to_csv(out_dirs["tables"] / "hrv_missingness_report.csv", index=False)


def recommendation_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        core = comparison[(comparison["model_name"] == model) & (comparison["feature_set"] == "core_behavioral_baseline")].iloc[0]
        hrv = comparison[(comparison["model_name"] == model) & (comparison["feature_set"] == "hrv_enhanced")].iloc[0]
        rows.append({
            "model_name": model,
            "does_hrv_improve_over_core": bool(
                (hrv["pr_auc_slow_onset"] > core["pr_auc_slow_onset"])
                or (hrv["f1_slow_onset_tuned"] > core["f1_slow_onset_tuned"])
                or (hrv["lift_at_top_20_percent_risk"] > core["lift_at_top_20_percent_risk"])
            ),
            "does_hrv_improve_slow_onset_f1": bool(hrv["f1_slow_onset_tuned"] > core["f1_slow_onset_tuned"]),
            "does_hrv_improve_slow_onset_recall": bool(hrv["recall_slow_onset_tuned"] > core["recall_slow_onset_tuned"]),
            "does_hrv_improve_pr_auc_slow_onset": bool(hrv["pr_auc_slow_onset"] > core["pr_auc_slow_onset"]),
            "does_hrv_improve_top20_risk_lift": bool(hrv["lift_at_top_20_percent_risk"] > core["lift_at_top_20_percent_risk"]),
            "does_hrv_worsen_brier": bool(hrv["brier_score_success"] > core["brier_score_success"]),
            "does_hrv_worsen_ece": bool(hrv["expected_calibration_error_success"] > core["expected_calibration_error_success"]),
            "core_pr_auc_slow_onset": core["pr_auc_slow_onset"],
            "hrv_pr_auc_slow_onset": hrv["pr_auc_slow_onset"],
            "core_f1_slow_onset": core["f1_slow_onset_tuned"],
            "hrv_f1_slow_onset": hrv["f1_slow_onset_tuned"],
            "core_top20_lift": core["lift_at_top_20_percent_risk"],
            "hrv_top20_lift": hrv["lift_at_top_20_percent_risk"],
            "core_brier": core["brier_score_success"],
            "hrv_brier": hrv["brier_score_success"],
            "core_ece": core["expected_calibration_error_success"],
            "hrv_ece": hrv["expected_calibration_error_success"],
        })
    out = pd.DataFrame(rows)
    rf = out[out["model_name"] == "random_forest"].iloc[0]
    any_detection = bool(out[["does_hrv_improve_slow_onset_f1", "does_hrv_improve_slow_onset_recall", "does_hrv_improve_top20_risk_lift"]].any(axis=None))
    final = "Use both: core behavioral baseline as the clean reference, and HRV-enhanced model as a physiological sensitivity model."
    if any_detection and not bool(rf["does_hrv_worsen_brier"] and rf["does_hrv_worsen_ece"]):
        final = "Use the HRV-enhanced model as the final enhanced model, while reporting the core model as the behavioral baseline."
    out["is_hrv_enhanced_worth_using_as_final_model"] = final
    out["presentation_recommendation"] = final
    return out


def write_report(comparison: pd.DataFrame, rec: pd.DataFrame, hrv_features: list[str], out_dirs: dict[str, Path]) -> None:
    best_hrv = comparison[comparison["feature_set"] == "hrv_enhanced"].sort_values(
        ["f1_slow_onset_tuned", "pr_auc_slow_onset"], ascending=False
    ).iloc[0]
    rec_text = rec.iloc[0]["presentation_recommendation"]
    table = comparison[[
        "feature_set",
        "model_name",
        "pr_auc_slow_onset",
        "f1_slow_onset_tuned",
        "recall_slow_onset_tuned",
        "lift_at_top_20_percent_risk",
        "brier_score_success",
        "expected_calibration_error_success",
    ]].to_markdown(index=False)
    text = f"""# HRV-Enhanced Final Model Report

## Purpose
This analysis keeps the original core behavioral baseline intact and adds a separate HRV-enhanced final model. The goal is to test whether pre-bedtime autonomic physiology adds useful signal beyond bedtime timing and previous sleep history.

## Why HRV Is Relevant
Heart rate and heart-rate variability are plausible sleep-onset predictors because they reflect autonomic arousal and recovery before bed. A person with elevated heart rate or altered HRV before bedtime may be physiologically less ready to fall asleep quickly, even if their recent sleep schedule looks stable.

## Features Included
The HRV-enhanced model uses the core behavioral features plus a compact Samsung HRV subset:

{', '.join(hrv_features)}

The preferred window is the 3-hour pre-bedtime window. These features were engineered only from timestamped records before `bedtime_start_timestamp`, so they do not use same-night sleep outcomes.

## Missingness Handling
HRV is substantially missing for some sleep episodes. Numeric HRV features are median-imputed inside each training fold only through the model pipeline. Missingness indicators such as `has_hrv_3h` and per-feature missing flags are included so the model can distinguish absent HRV data from observed low/high values. Held-out participants are never used to fit imputers.

## Results
{table}

## Interpretation
Best HRV-enhanced model by tuned slow-onset F1: `{best_hrv['model_name']}`.

The comparison should be read as a detection-versus-reliability tradeoff. HRV may be useful if it improves slow-onset recall, F1, or top-risk lift, but it also introduces missingness and can worsen calibration depending on the model.

## Recommendation
{rec_text}

For presentation, the core model should remain the clean behavioral baseline. The HRV-enhanced model can be presented as the final enhanced model if its detection metrics improve enough to justify the added physiological modality and missingness handling.
"""
    (out_dirs["root"] / "HRV_ENHANCED_MODEL_REPORT.md").write_text(text, encoding="utf-8")


def update_main_readme(readme_path: Path, comparison: pd.DataFrame, rec: pd.DataFrame) -> None:
    marker = "## HRV-Enhanced Final Model"
    text = readme_path.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    compact = comparison[[
        "feature_set",
        "model_name",
        "pr_auc_slow_onset",
        "f1_slow_onset_tuned",
        "recall_slow_onset_tuned",
        "lift_at_top_20_percent_risk",
        "brier_score_success",
        "expected_calibration_error_success",
    ]].to_markdown(index=False)
    section = f"""## HRV-Enhanced Final Model
The core model remains the behavioral baseline: temporal features plus previous sleep history. A new HRV-enhanced analysis was added under `outputs/final_hrv_enhanced_model/` to test whether selected pre-bedtime Samsung HRV features improve sleepability prediction.

HRV was included because autonomic recovery and pre-bedtime arousal are theoretically relevant to sleep onset. The enhanced model adds a compact 3-hour pre-bedtime HRV subset, including heart-rate summary statistics, RMSSD, SDNN, record count, and HRV availability/missingness indicators.

HRV missingness was handled with median imputation fit only inside training folds, plus missingness indicators. Oversampling still occurs only after participant-aware splitting, and held-out participants preserve the natural class distribution.

{compact}

Recommendation: {rec.iloc[0]['presentation_recommendation']}
"""
    readme_path.write_text(text + section + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    output_root = cfg["output_root"] / ANALYSIS_NAME
    out_dirs = make_output_dirs(output_root)
    feature_path = cfg["output_root"] / "tables" / "feature_matrix_full.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}. Run scripts/run_all.py first.")
    df = pd.read_csv(feature_path)
    missing_core = [f for f in CORE_FEATURES if f not in df.columns]
    if missing_core:
        raise ValueError(f"Missing core features: {missing_core}")
    hrv_features, used_fallback = select_hrv_features(df)
    if not hrv_features:
        raise ValueError("No Samsung HRV features found in feature_matrix_full.csv")
    df, hrv_indicators = add_hrv_missingness_indicators(df, hrv_features)
    hrv_feature_set = list(dict.fromkeys(CORE_FEATURES + hrv_features + hrv_indicators))
    write_feature_tables(df, hrv_features, hrv_indicators, used_fallback, out_dirs)
    plot_hrv_missingness(df, hrv_features, out_dirs)
    feature_sets = {
        "core_behavioral_baseline": CORE_FEATURES,
        "hrv_enhanced": hrv_feature_set,
    }
    rows = []
    all_curves = []
    all_balance = []
    validation_name = None
    for feature_set_name, features in feature_sets.items():
        for model_name in MODEL_NAMES:
            pred, thresholds, curves, balance, validation_name = oof_train(
                df,
                features,
                model_name,
                feature_set_name,
                out_dirs,
                cfg["random_seed"],
                cfg["n_splits"],
            )
            rows.append(summarize_predictions(pred, thresholds, model_name, feature_set_name, len(features)))
            all_curves.append(curves)
            all_balance.append(balance)
            final_prediction_plots(pred, model_name, feature_set_name, out_dirs)
    comparison = pd.DataFrame(rows)
    comparison["validation_method"] = validation_name
    comparison.to_csv(out_dirs["tables"] / "core_vs_hrv_enhanced_model_comparison.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(out_dirs["tables"] / "threshold_curves_core_vs_hrv.csv", index=False)
    pd.concat(all_balance, ignore_index=True).to_csv(out_dirs["tables"] / "fold_class_balance_core_vs_hrv.csv", index=False)
    top20 = comparison[[
        "feature_set",
        "model_name",
        "slow_onset_recall_at_top_20_percent_risk",
        "slow_onset_precision_at_top_20_percent_risk",
        "lift_at_top_20_percent_risk",
    ]]
    top20.to_csv(out_dirs["tables"] / "top20_risk_lift_comparison.csv", index=False)
    plot_metric_bars(comparison, out_dirs)
    grouped = fit_full_importance_models(df, hrv_feature_set, out_dirs, cfg["random_seed"])
    grouped.to_csv(out_dirs["tables"] / "hrv_grouped_feature_importance.csv", index=False)
    rec = recommendation_summary(comparison)
    rec.to_csv(out_dirs["tables"] / "hrv_enhanced_recommendation_summary.csv", index=False)
    write_report(comparison, rec, hrv_features, out_dirs)
    update_main_readme(ROOT / "README.md", comparison, rec)
    summary = {
        "output_folder": str(output_root.resolve()),
        "validation_method": validation_name,
        "core_features": len(CORE_FEATURES),
        "hrv_enhanced_features": len(hrv_feature_set),
        "selected_hrv_features": hrv_features,
        "hrv_missingness_indicators": hrv_indicators,
        "best_core_model_by_slow_f1": comparison[comparison["feature_set"].eq("core_behavioral_baseline")].sort_values("f1_slow_onset_tuned", ascending=False).iloc[0]["model_name"],
        "best_hrv_model_by_slow_f1": comparison[comparison["feature_set"].eq("hrv_enhanced")].sort_values("f1_slow_onset_tuned", ascending=False).iloc[0]["model_name"],
        "recommendation": rec.iloc[0]["presentation_recommendation"],
    }
    (out_dirs["tables"] / "final_hrv_enhanced_execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dirs["tables"] / "final_hrv_enhanced_execution_summary.csv", index=False)
    print("\nHRV-ENHANCED MODEL SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
