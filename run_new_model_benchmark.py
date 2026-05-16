from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import choose_threshold, probability_metrics, threshold_metrics
from src.feature_groups import FEATURE_GROUPS
from src.imbalance import oversample_minority
from src.split_validation import participant_splitter
from src.utils import ece_score, load_config

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:
    ExplainableBoostingClassifier = None


ANALYSIS_NAME = "new_model_benchmark"
CORE_FEATURES = FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["sleep_history"]
MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "catboost",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "ebm",
    "gradient_boosting",
    "hist_gradient_boosting",
    "shallow_mlp",
    "rbf_svm",
]


def make_output_dirs(base: Path) -> dict[str, Path]:
    paths = {name: base / name for name in ["tables", "plots", "predictions", "logs", "models"]}
    paths["root"] = base
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def package_available(package: str | None) -> bool:
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None


def model_availability() -> pd.DataFrame:
    rows = [
        ("logistic_regression", "sklearn", True, "run", "Reference linear baseline."),
        ("random_forest", "sklearn", True, "run", "Current reference nonlinear model."),
        ("catboost", "catboost", CatBoostClassifier is not None, "run" if CatBoostClassifier is not None else "skipped", "Optional package."),
        ("extra_trees", "sklearn", True, "run", "Randomized tree ensemble benchmark."),
        ("xgboost", "xgboost", XGBClassifier is not None, "run" if XGBClassifier is not None else "skipped", "Optional package not installed or import failed." if XGBClassifier is None else "Optional package available."),
        ("lightgbm", "lightgbm", LGBMClassifier is not None, "run" if LGBMClassifier is not None else "skipped", "Optional package not installed or import failed." if LGBMClassifier is None else "Optional package available."),
        ("ebm", "interpret", ExplainableBoostingClassifier is not None, "run" if ExplainableBoostingClassifier is not None else "skipped", "Optional interpret package not installed or import failed." if ExplainableBoostingClassifier is None else "Optional package available."),
        ("gradient_boosting", "sklearn", True, "run", "Classic sklearn boosting benchmark."),
        ("hist_gradient_boosting", "sklearn", True, "run", "Histogram boosting benchmark."),
        ("shallow_mlp", "sklearn", True, "run", "Small neural network benchmark."),
        ("rbf_svm", "sklearn", True, "run", "Probability-calibrated RBF SVM benchmark; skipped only if runtime fails."),
    ]
    return pd.DataFrame(rows, columns=["model_name", "package_required", "package_available", "installed_or_skipped", "notes"])


def split_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in x.columns if x[c].dtype == "object"]
    return [c for c in x.columns if c not in cat], cat


def preprocess_linear(x: pd.DataFrame) -> ColumnTransformer:
    num, cat = split_columns(x)
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def preprocess_tree(x: pd.DataFrame) -> ColumnTransformer:
    num, cat = split_columns(x)
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def make_model(model_name: str, x_train: pd.DataFrame, random_state: int = 42):
    if model_name == "logistic_regression":
        return Pipeline([
            ("prep", preprocess_linear(x_train)),
            ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, class_weight=None)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, max_features="sqrt", random_state=random_state, n_jobs=-1)),
        ])
    if model_name == "extra_trees":
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, max_features="sqrt", random_state=random_state, n_jobs=-1)),
        ])
    if model_name == "gradient_boosting":
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=random_state)),
        ])
    if model_name == "hist_gradient_boosting":
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.1, random_state=random_state)),
        ])
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is not available.")
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=350,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                reg_alpha=0.1,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is not available.")
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", LGBMClassifier(
                objective="binary",
                n_estimators=350,
                num_leaves=15,
                max_depth=5,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                lambda_l1=0.1,
                lambda_l2=3.0,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )),
        ])
    if model_name == "ebm":
        if ExplainableBoostingClassifier is None:
            raise RuntimeError("ExplainableBoostingClassifier is not available.")
        return Pipeline([
            ("prep", preprocess_tree(x_train)),
            ("model", ExplainableBoostingClassifier(
                random_state=random_state,
                interactions=0,
                max_rounds=200,
                learning_rate=0.03,
                n_jobs=-1,
            )),
        ])
    if model_name == "shallow_mlp":
        return Pipeline([
            ("prep", preprocess_linear(x_train)),
            ("model", MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.001, learning_rate_init=0.001, early_stopping=True, max_iter=500, random_state=random_state)),
        ])
    if model_name == "rbf_svm":
        return Pipeline([
            ("prep", preprocess_linear(x_train)),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)),
        ])
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is not available.")
        return CatBoostClassifier(
            iterations=700,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            class_weights=None,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_predict_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, seed: int):
    model = make_model(model_name, x_train, seed)
    if model_name == "catboost":
        cat_cols = [c for c in x_train.columns if x_train[c].dtype == "object"]
        train = x_train.copy()
        test = x_test.copy()
        for c in cat_cols:
            train[c] = train[c].astype(str).fillna("__MISSING__")
            test[c] = test[c].astype(str).fillna("__MISSING__")
        model.fit(train, y_train, cat_features=cat_cols if cat_cols else None)
        return model, model.predict_proba(test)[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return model, model.predict_proba(x_test)[:, 1]


def add_hrv_missingness_indicators(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    indicators = []
    if "samsung_hrv_3h_count_records" in out.columns:
        out["has_hrv_3h"] = out["samsung_hrv_3h_count_records"].fillna(0).gt(0).astype(int)
        indicators.append("has_hrv_3h")
    if "samsung_hrv_6h_count_records" in out.columns:
        out["has_hrv_6h"] = out["samsung_hrv_6h_count_records"].fillna(0).gt(0).astype(int)
        indicators.append("has_hrv_6h")
    for feature in features:
        if feature.endswith("_count_records") or feature not in out.columns:
            continue
        indicator = f"{feature}_missing"
        out[indicator] = out[feature].isna().astype(int)
        indicators.append(indicator)
    return out, indicators


def load_feature_sets(df: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    hrv_list_path = output_root / "final_hrv_enhanced_model" / "tables" / "hrv_enhanced_feature_list.csv"
    hrv_source = pd.read_csv(hrv_list_path) if hrv_list_path.exists() else pd.DataFrame()
    hrv_base = []
    if not hrv_source.empty:
        hrv_base = hrv_source.loc[hrv_source["feature_group"].eq("HRV"), "feature_name"].tolist()
        hrv_base = [f for f in hrv_base if f in df.columns and not f.endswith("_missing") and not f.startswith("has_hrv")]
    if not hrv_base:
        hrv_base = [f for f in [
            "samsung_hrv_3h_count_records",
            "samsung_hrv_3h_hr_mean",
            "samsung_hrv_3h_hr_std",
            "samsung_hrv_3h_hr_min",
            "samsung_hrv_3h_hr_max",
            "samsung_hrv_3h_hrv_rmssd_mean",
            "samsung_hrv_3h_hrv_rmssd_std",
            "samsung_hrv_3h_hrv_sdnn_mean",
            "samsung_hrv_3h_hrv_sdnn_std",
        ] if f in df.columns]
    df2, indicators = add_hrv_missingness_indicators(df, hrv_base)
    core = [f for f in CORE_FEATURES if f in df2.columns]
    hrv = list(dict.fromkeys(core + hrv_base + indicators))
    return df2, {"core_behavioral": core, "hrv_enhanced": hrv}


def feature_group(feature: str) -> str:
    if feature in FEATURE_GROUPS["temporal"]:
        return "temporal"
    if feature in FEATURE_GROUPS["sleep_history"]:
        return "sleep_history"
    if feature.startswith("samsung_hrv") or feature.startswith("has_hrv"):
        return "HRV"
    return "other"


def write_feature_sets_used(df: pd.DataFrame, feature_sets: dict[str, list[str]], out_dirs: dict[str, Path]) -> None:
    rows = []
    for set_name, features in feature_sets.items():
        for feature in features:
            rows.append({
                "feature_set_name": set_name,
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "missing_rate": float(df[feature].isna().mean()),
                "included": True,
                "notes": "Core baseline feature." if set_name == "core_behavioral" else "Included in HRV-enhanced benchmark.",
            })
    pd.DataFrame(rows).to_csv(out_dirs["tables"] / "feature_sets_used.csv", index=False)


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


def train_fold_model(df: pd.DataFrame, indices: np.ndarray, test_indices: np.ndarray, features: list[str], model_name: str, seed: int):
    y = df["target_sleep_success_15"].astype(int)
    x_over, y_over = oversample_minority(df.iloc[indices][features], y.iloc[indices], 0.667, seed)
    model, p_test = fit_predict_model(model_name, x_over, y_over, df.iloc[test_indices][features], seed)
    return model, p_test, x_over, y_over


def oof_train(df: pd.DataFrame, features: list[str], model_name: str, feature_set: str, out_dirs: dict[str, Path], seed: int, n_splits: int):
    y = df["target_sleep_success_15"].astype(int)
    splitter, validation_name = participant_splitter(n_splits)
    p_success = np.zeros(len(df))
    default_pred = np.zeros(len(df), dtype=int)
    tuned_pred = np.zeros(len(df), dtype=int)
    folds = np.zeros(len(df), dtype=int)
    thresholds = []
    curves = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, df["participant_id"]), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        fit_idx, val_idx = inner_split(df, train_idx, seed + fold)
        _, p_val, _, _ = train_fold_model(df, fit_idx, val_idx, features, model_name, seed + 100 + fold)
        threshold, curve = choose_threshold(y.iloc[val_idx].to_numpy(dtype=int), p_val)
        curve["fold"] = fold
        curve["model_name"] = model_name
        curve["feature_set"] = feature_set
        curves.append(curve)
        _, p_test, _, _ = train_fold_model(df, train_idx, test_idx, features, model_name, seed + 200 + fold)
        p_success[test_idx] = p_test
        default_pred[test_idx] = (p_test >= 0.5).astype(int)
        tuned_pred[test_idx] = (p_test >= threshold).astype(int)
        folds[test_idx] = fold
        thresholds.append(float(threshold))
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
        "feature_set": feature_set,
        "model_name": model_name,
    })
    pred.to_csv(out_dirs["predictions"] / f"oof_predictions_{model_name}_{feature_set}.csv", index=False)
    return pred, thresholds, pd.concat(curves, ignore_index=True), validation_name


def top20_risk_metrics(y_success, p_success) -> dict[str, float]:
    y = np.asarray(y_success, dtype=int)
    risk = 1 - np.asarray(p_success, dtype=float)
    y_slow = 1 - y
    n_top = max(1, int(np.ceil(0.20 * len(y))))
    top_idx = np.argsort(-risk)[:n_top]
    precision = float(y_slow[top_idx].sum() / n_top)
    recall = float(y_slow[top_idx].sum() / max(1, y_slow.sum()))
    lift = float(precision / y_slow.mean()) if y_slow.mean() > 0 else np.nan
    return {
        "slow_onset_precision_at_top_20_percent_risk": precision,
        "slow_onset_recall_at_top_20_percent_risk": recall,
        "lift_at_top_20_percent_risk": lift,
    }


def summarize(pred: pd.DataFrame, thresholds: list[float], feature_set: str, model_name: str, model_family: str, package_available_flag: bool, n_features: int, notes: str) -> dict[str, object]:
    y = pred["target_sleep_success_15"].to_numpy(dtype=int)
    p = pred["predicted_probability_success"].to_numpy(dtype=float)
    default = threshold_metrics(y, p, 0.5, pred["predicted_success_default"].to_numpy(dtype=int))
    tuned = threshold_metrics(y, p, float(np.mean(thresholds)), pred["predicted_success_tuned"].to_numpy(dtype=int))
    return {
        "feature_set": feature_set,
        "model_name": model_name,
        "model_family": model_family,
        "package_available": package_available_flag,
        "hyperparameter_strategy": "fixed regularized settings; threshold tuned on inner participant split",
        "n_features": int(n_features),
        "n_rows": int(len(pred)),
        "n_participants": int(pred["participant_id"].nunique()),
        "success_rate": float(y.mean()),
        "slow_onset_rate": float((1 - y).mean()),
        **probability_metrics(y, p),
        **{f"{k}_default": v for k, v in default.items()},
        **{f"{k}_tuned": v for k, v in tuned.items()},
        "tuned_threshold_mean": float(np.mean(thresholds)),
        "tuned_threshold_std": float(np.std(thresholds)),
        **top20_risk_metrics(y, p),
        "notes": notes,
    }


def plot_comparison(comparison: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    specs = [
        ("pr_auc_slow_onset", "model_comparison_pr_auc_slow_onset.png", "PR-AUC slow onset"),
        ("f1_slow_onset_tuned", "model_comparison_f1_slow_onset.png", "Tuned F1 slow onset"),
        ("brier_score_success", "model_comparison_brier.png", "Brier score"),
        ("expected_calibration_error_success", "model_comparison_ece.png", "Expected calibration error"),
        ("lift_at_top_20_percent_risk", "model_comparison_top20_lift.png", "Top-20% slow-onset lift"),
        ("balanced_accuracy_tuned", "model_comparison_balanced_accuracy.png", "Tuned balanced accuracy"),
    ]
    for metric, filename, title in specs:
        top = comparison.sort_values(metric, ascending=metric in ["brier_score_success", "expected_calibration_error_success"]).copy()
        labels = top["model_name"] + "\n" + top["feature_set"]
        plt.figure(figsize=(10, 5.5))
        plt.bar(labels, top[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / filename, dpi=170)
        plt.close()
    for metric, filename, title in [
        ("pr_auc_slow_onset", "core_vs_hrv_by_model_pr_auc_slow_onset.png", "Core vs HRV by model: PR-AUC slow onset"),
        ("f1_slow_onset_tuned", "core_vs_hrv_by_model_f1_slow_onset.png", "Core vs HRV by model: F1 slow onset"),
        ("lift_at_top_20_percent_risk", "core_vs_hrv_by_model_top20_lift.png", "Core vs HRV by model: Top-20% lift"),
    ]:
        pivot = comparison.pivot(index="model_name", columns="feature_set", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_title(title)
        ax.set_ylabel(metric)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / filename, dpi=170)
        plt.close()


def best_model_plots(pred_path: Path, out_dirs: dict[str, Path]) -> None:
    pred = pd.read_csv(pred_path)
    y = pred["target_sleep_success_15"].astype(int)
    p = pred["predicted_probability_success"].astype(float)
    RocCurveDisplay.from_predictions(y, p)
    plt.title("Best benchmark model ROC")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_roc_curve.png", dpi=170)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y, p)
    plt.title("Best benchmark model PR curve: success")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_pr_curve_success.png", dpi=170)
    plt.close()
    PrecisionRecallDisplay.from_predictions(1 - y, 1 - p)
    plt.title("Best benchmark model PR curve: slow onset")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_pr_curve_slow_onset.png", dpi=170)
    plt.close()
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean, frac, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted P(success)")
    plt.ylabel("Observed success rate")
    plt.title("Best benchmark model calibration")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_calibration_curve.png", dpi=170)
    plt.close()
    plt.figure(figsize=(6, 4))
    for cls, label in [(0, "slow onset"), (1, "success")]:
        plt.hist(p[y == cls], bins=22, alpha=0.55, label=label)
    plt.legend()
    plt.xlabel("P(success)")
    plt.ylabel("Episodes")
    plt.title("Best benchmark model probability histogram")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_probability_histogram.png", dpi=170)
    plt.close()
    ConfusionMatrixDisplay.from_predictions(y, pred["predicted_success_tuned"].astype(int), labels=[0, 1], display_labels=["slow onset", "success"])
    plt.title("Best benchmark model tuned confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_confusion_matrix_tuned.png", dpi=170)
    plt.close()


def rank_and_recommend(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        ("best_pr_auc_slow_onset", "pr_auc_slow_onset", False),
        ("best_f1_slow_onset", "f1_slow_onset_tuned", False),
        ("best_brier_score", "brier_score_success", True),
        ("best_ece", "expected_calibration_error_success", True),
        ("best_top20_lift", "lift_at_top_20_percent_risk", False),
        ("best_balanced_accuracy", "balanced_accuracy_tuned", False),
    ]
    rows = []
    for label, metric, asc in metrics:
        best = comparison.sort_values(metric, ascending=asc).iloc[0]
        rows.append({"metric": label, "model_name": best["model_name"], "feature_set": best["feature_set"], "value": best[metric]})
    ranked = comparison.assign(
        overall_rank=(
            comparison["pr_auc_slow_onset"].rank(ascending=False)
            + comparison["f1_slow_onset_tuned"].rank(ascending=False)
            + comparison["brier_score_success"].rank(ascending=True)
            + comparison["expected_calibration_error_success"].rank(ascending=True)
            + comparison["lift_at_top_20_percent_risk"].rank(ascending=False)
        )
    ).sort_values("overall_rank")
    rf_core = comparison[(comparison["model_name"].eq("random_forest")) & (comparison["feature_set"].eq("core_behavioral"))].iloc[0]
    rf_hrv = comparison[(comparison["model_name"].eq("random_forest")) & (comparison["feature_set"].eq("hrv_enhanced"))].iloc[0]
    new_models = comparison[~comparison["model_name"].isin(["logistic_regression", "random_forest", "catboost"])]
    new_beats_rf = bool((new_models["pr_auc_slow_onset"].max() > rf_core["pr_auc_slow_onset"]) or (new_models["f1_slow_onset_tuned"].max() > rf_core["f1_slow_onset_tuned"]))
    new_beats_hrv_rf = bool((new_models["pr_auc_slow_onset"].max() > rf_hrv["pr_auc_slow_onset"]) or (new_models["f1_slow_onset_tuned"].max() > rf_hrv["f1_slow_onset_tuned"]))
    composite_best = ranked.iloc[0]
    replacement_candidates = new_models[
        (new_models["f1_slow_onset_tuned"] >= rf_core["f1_slow_onset_tuned"])
        & (new_models["brier_score_success"] <= rf_core["brier_score_success"] + 0.002)
        & (new_models["expected_calibration_error_success"] <= rf_core["expected_calibration_error_success"] + 0.01)
    ]
    if len(replacement_candidates):
        best = replacement_candidates.assign(
            conservative_rank=(
                replacement_candidates["f1_slow_onset_tuned"].rank(ascending=False)
                + replacement_candidates["brier_score_success"].rank(ascending=True)
                + replacement_candidates["pr_auc_slow_onset"].rank(ascending=False)
            )
        ).sort_values("conservative_rank").iloc[0]
        replacement = True
        replacement_note = "A new model matched or improved Random Forest slow-onset F1 without a meaningful Brier/ECE penalty."
    else:
        best = rf_core
        replacement = False
        replacement_note = (
            f"The composite rank favored {composite_best['model_name']} / {composite_best['feature_set']}, "
            "but it did not beat core Random Forest on the main practical detection metric plus calibration reliability."
        )
    best_rank = ranked[
        ranked["model_name"].eq(best["model_name"]) & ranked["feature_set"].eq(best["feature_set"])
    ]["overall_rank"]
    rows.append({
        "metric": "best_overall_recommended",
        "model_name": best["model_name"],
        "feature_set": best["feature_set"],
        "value": float(best_rank.iloc[0]) if len(best_rank) else np.nan,
    })
    best_by_metric = pd.DataFrame(rows)
    rec = pd.DataFrame([
        {"question": "Did any new model beat current Random Forest?", "answer": new_beats_rf, "notes": "Compared against core Random Forest on slow-onset PR-AUC and tuned F1."},
        {"question": "Did any new model beat HRV-enhanced extension?", "answer": new_beats_hrv_rf, "notes": "Compared against HRV-enhanced Random Forest on slow-onset PR-AUC and tuned F1."},
        {"question": "Which model is best for slow-onset detection?", "answer": best_by_metric.loc[best_by_metric["metric"].eq("best_f1_slow_onset"), "model_name"].iloc[0], "notes": "Based on tuned F1 slow-onset."},
        {"question": "Which model is best calibrated?", "answer": best_by_metric.loc[best_by_metric["metric"].eq("best_ece"), "model_name"].iloc[0], "notes": "Based on lowest ECE."},
        {"question": "Which model has best top-risk lift?", "answer": best_by_metric.loc[best_by_metric["metric"].eq("best_top20_lift"), "model_name"].iloc[0], "notes": "Based on highest slow-onset lift in top 20% risk."},
        {"question": "Is any new model worth replacing the current final model?", "answer": replacement, "notes": f"Overall recommended benchmark model is {best['model_name']} with {best['feature_set']} features."},
        {"question": "If not, why not? If yes, why?", "answer": f"{best['model_name']} / {best['feature_set']}", "notes": replacement_note},
    ])
    return best_by_metric, rec


def write_hyperparameter_summary(successful: list[dict[str, object]], out_dirs: dict[str, Path]) -> None:
    params = {
        "logistic_regression": {"penalty": "l2", "solver": "lbfgs", "max_iter": 5000},
        "random_forest": {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5, "max_features": "sqrt"},
        "catboost": {"iterations": 700, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 6},
        "extra_trees": {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
        "gradient_boosting": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.8},
        "hist_gradient_boosting": {"max_iter": 350, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        "xgboost": {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3, "reg_alpha": 0.1},
        "lightgbm": {"n_estimators": 350, "num_leaves": 15, "max_depth": 5, "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 3},
        "ebm": {"interactions": 0, "max_rounds": 200, "learning_rate": 0.03},
        "shallow_mlp": {"hidden_layer_sizes": [32], "alpha": 0.001, "early_stopping": True, "max_iter": 500},
        "rbf_svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True},
    }
    rows = []
    for item in successful:
        rows.append({
            "model_name": item["model_name"],
            "feature_set": item["feature_set"],
            "best_params": json.dumps(params.get(item["model_name"], {})),
            "tuning_metric": "threshold tuned by inner balanced accuracy; fixed regularized model hyperparameters",
            "tuning_notes": "This benchmark uses controlled fixed settings to compare model families without expensive nested optimization.",
        })
    pd.DataFrame(rows).to_csv(out_dirs["tables"] / "hyperparameter_summary.csv", index=False)


def fit_best_importance(df: pd.DataFrame, features: list[str], best_model: str, out_dirs: dict[str, Path], seed: int) -> None:
    if best_model in ["rbf_svm", "shallow_mlp"]:
        return
    x_over, y_over = oversample_minority(df[features], df["target_sleep_success_15"].astype(int), 0.667, seed)
    model, _ = fit_predict_model(best_model, x_over, y_over, x_over, seed)
    rows = []
    if best_model == "catboost":
        rows = [{"feature_name": f, "importance": v} for f, v in zip(features, model.get_feature_importance())]
    elif hasattr(model, "named_steps"):
        fitted = model.named_steps["model"]
        names = model.named_steps["prep"].get_feature_names_out()
        if hasattr(fitted, "feature_importances_"):
            vals = fitted.feature_importances_
        elif hasattr(fitted, "coef_"):
            vals = np.abs(fitted.coef_[0])
        else:
            return
        for name, val in zip(names, vals):
            clean = name.split("__", 1)[-1]
            source = next((f for f in features if clean == f or clean.startswith(f + "_")), clean)
            rows.append({"feature_name": source, "importance": float(val)})
    imp = pd.DataFrame(rows).groupby("feature_name", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    imp.to_csv(out_dirs["tables"] / "best_model_feature_importance.csv", index=False)
    top = imp.head(18).sort_values("importance")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature_name"], top["importance"])
    plt.xlabel("importance")
    plt.title(f"Best benchmark model feature importance: {best_model}")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "best_model_feature_importance.png", dpi=170)
    plt.close()


def write_report(comparison: pd.DataFrame, availability: pd.DataFrame, best_by_metric: pd.DataFrame, rec: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    skipped = availability[availability["installed_or_skipped"].eq("skipped")]
    current_rf = comparison[(comparison["model_name"].eq("random_forest")) & (comparison["feature_set"].eq("core_behavioral"))].iloc[0]
    table = comparison.copy()
    table["better_than_rf"] = np.where(
        (table["pr_auc_slow_onset"] > current_rf["pr_auc_slow_onset"]) | (table["f1_slow_onset_tuned"] > current_rf["f1_slow_onset_tuned"]),
        "Yes",
        "No",
    )
    table["reason"] = np.where(table["feature_set"].eq("hrv_enhanced"), "Tests added physiological signal; watch missingness/calibration.", "Clean behavioral baseline comparison.")
    md = table[["model_name", "feature_set", "pr_auc_slow_onset", "f1_slow_onset_tuned", "brier_score_success", "expected_calibration_error_success", "lift_at_top_20_percent_risk", "better_than_rf", "reason"]].to_markdown(index=False)
    best_md = best_by_metric.to_markdown(index=False)
    skip_md = skipped.to_markdown(index=False) if len(skipped) else "No models were skipped."
    rec_md = rec.to_markdown(index=False)
    text = f"""# New Model Benchmark Report

## Why Additional Models Were Tested
The original project compared Logistic Regression, Random Forest, and CatBoost. This benchmark tests whether additional nonlinear or flexible model families can improve sleepability prediction, especially slow-onset detection, without changing the participant-aware validation or leakage rules.

## Feature Sets
Two feature sets were evaluated: the core behavioral set (`temporal + previous sleep history`) and the HRV-enhanced set (`core + selected pre-bedtime Samsung HRV features`). Participant identifiers, raw timestamps, onset latency, and same-night sleep outcomes were not used as predictors.

## Validation And Imbalance Strategy
The benchmark uses participant-aware cross-validation with train-fold-only oversampling of slow-onset/failure cases to approximately 40% of the training fold. Test participants are untouched. Thresholds are tuned only on inner participant-held-out validation splits.

## Models Tested
Reference models: Logistic Regression, Random Forest, and CatBoost. New models: ExtraTrees, sklearn GradientBoosting, sklearn HistGradientBoosting, shallow MLP, and RBF SVM. Optional XGBoost, LightGBM, and EBM were checked and skipped if unavailable.

## Skipped Models
{skip_md}

## Result Summary
{md}

## Best Model By Metric
{best_md}

## Final Recommendation Questions
{rec_md}

## Interpretation
ExtraTrees tests whether stronger randomized tree ensembles help beyond Random Forest. Boosting models test whether more sequential nonlinear fitting can extract additional tabular signal. The shallow MLP and RBF SVM test flexible nonlinear boundaries, but these models can struggle on small participant-level tabular datasets and can be harder to calibrate.

If a model improves slow-onset F1 or top-risk lift but worsens Brier score or ECE, it should be presented as a detection-versus-reliability tradeoff rather than an unconditional win. If Random Forest remains competitive, the bottleneck is likely data signal, participant variability, and modality missingness rather than simply model choice.

## Limitations
This is still an observational, small-participant wearable dataset. Fixed regularized hyperparameters were used for practical runtime; the benchmark is fair in validation design, but not an exhaustive tuning competition. Probability calibration remains important before any app use.
"""
    (out_dirs["root"] / "NEW_MODEL_BENCHMARK_REPORT.md").write_text(text, encoding="utf-8")


def update_readme(readme_path: Path, comparison: pd.DataFrame, rec: pd.DataFrame) -> None:
    text = readme_path.read_text(encoding="utf-8")
    marker = "## Extended Model Benchmark"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    compact = comparison.sort_values("pr_auc_slow_onset", ascending=False).head(8)[[
        "feature_set", "model_name", "pr_auc_slow_onset", "f1_slow_onset_tuned", "brier_score_success", "expected_calibration_error_success", "lift_at_top_20_percent_risk"
    ]].to_markdown(index=False)
    final = rec.loc[rec["question"].eq("Is any new model worth replacing the current final model?"), "notes"].iloc[0]
    section = f"""## Extended Model Benchmark
An additional benchmark was added under `outputs/new_model_benchmark/` to test whether model families beyond Logistic Regression, Random Forest, and CatBoost improve the 15-minute sleepability task. The benchmark evaluates both the core behavioral feature set and the HRV-enhanced feature set using the same participant-aware validation, train-fold-only oversampling, and leakage controls.

Models tested include ExtraTrees, sklearn GradientBoosting, sklearn HistGradientBoosting, a shallow MLP, and an RBF SVM, plus the original reference models. Optional XGBoost, LightGBM, and EBM were checked and documented if unavailable.

{compact}

Final benchmark note: {final}
"""
    readme_path.write_text(text + section + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    out_dirs = make_output_dirs(cfg["output_root"] / ANALYSIS_NAME)
    availability = model_availability()
    availability["reason_if_skipped"] = np.where(availability["installed_or_skipped"].eq("skipped"), availability["notes"], "")
    availability.to_csv(out_dirs["tables"] / "model_availability_and_notes.csv", index=False)
    feature_path = cfg["output_root"] / "tables" / "feature_matrix_full.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feature_path}")
    df = pd.read_csv(feature_path)
    df, feature_sets = load_feature_sets(df, cfg["output_root"])
    write_feature_sets_used(df, feature_sets, out_dirs)
    model_meta = {
        "logistic_regression": ("linear", "Reference interpretable baseline."),
        "random_forest": ("bagged_trees", "Reference final model family."),
        "catboost": ("boosted_trees", "Reference boosted tabular model."),
        "extra_trees": ("randomized_trees", "New randomized tree ensemble."),
        "xgboost": ("boosted_trees", "Regularized gradient boosting benchmark."),
        "lightgbm": ("boosted_trees", "Efficient gradient boosting benchmark."),
        "ebm": ("interpretable_boosting", "Interpretable additive boosting benchmark."),
        "gradient_boosting": ("boosted_trees", "New sklearn boosting baseline."),
        "hist_gradient_boosting": ("boosted_trees", "New histogram boosting baseline."),
        "shallow_mlp": ("neural_network", "Small neural network; not expected to dominate small tabular data."),
        "rbf_svm": ("kernel_method", "Flexible nonlinear kernel model with probability estimates."),
    }
    rows = []
    successful = []
    curves = []
    for model_name in MODEL_ORDER:
        av = availability[availability["model_name"].eq(model_name)].iloc[0]
        if av["installed_or_skipped"] == "skipped":
            continue
        for feature_set, features in feature_sets.items():
            print(f"Running {model_name} on {feature_set} ({len(features)} features)")
            try:
                pred, thresholds, curve, validation_name = oof_train(df, features, model_name, feature_set, out_dirs, cfg["random_seed"], cfg["n_splits"])
                family, notes = model_meta[model_name]
                rows.append(summarize(pred, thresholds, feature_set, model_name, family, True, len(features), notes))
                successful.append({"model_name": model_name, "feature_set": feature_set})
                curves.append(curve)
            except Exception as exc:
                availability.loc[availability["model_name"].eq(model_name), "installed_or_skipped"] = "skipped"
                availability.loc[availability["model_name"].eq(model_name), "reason_if_skipped"] = str(exc)
                availability.loc[availability["model_name"].eq(model_name), "notes"] = f"Runtime failure on {feature_set}: {exc}"
                print(f"Skipped {model_name} on {feature_set}: {exc}")
                break
    if not rows:
        raise RuntimeError("No benchmark models ran successfully.")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dirs["tables"] / "new_model_benchmark_comparison.csv", index=False)
    availability.to_csv(out_dirs["tables"] / "model_availability_and_notes.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out_dirs["tables"] / "threshold_curves_new_model_benchmark.csv", index=False)
    write_hyperparameter_summary(successful, out_dirs)
    best_by_metric, rec = rank_and_recommend(comparison)
    best_by_metric.to_csv(out_dirs["tables"] / "best_model_by_metric.csv", index=False)
    rec.to_csv(out_dirs["tables"] / "new_model_final_recommendation.csv", index=False)
    plot_comparison(comparison, out_dirs)
    best = best_by_metric[best_by_metric["metric"].eq("best_overall_recommended")].iloc[0]
    best_pred_path = out_dirs["predictions"] / f"oof_predictions_{best['model_name']}_{best['feature_set']}.csv"
    best_model_plots(best_pred_path, out_dirs)
    fit_best_importance(df, feature_sets[best["feature_set"]], best["model_name"], out_dirs, cfg["random_seed"])
    write_report(comparison, availability, best_by_metric, rec, out_dirs)
    update_readme(ROOT / "README.md", comparison, rec)
    skipped = availability[availability["installed_or_skipped"].eq("skipped")]
    hrv_helped = bool(
        comparison.pivot(index="model_name", columns="feature_set", values="pr_auc_slow_onset")
        .dropna()
        .eval("hrv_enhanced > core_behavioral")
        .any()
    )
    summary = {
        "number_of_models_attempted": int(len(MODEL_ORDER)),
        "number_of_models_successfully_run": int(comparison["model_name"].nunique()),
        "models_skipped_and_why": "; ".join((skipped["model_name"] + ": " + skipped["reason_if_skipped"].fillna(skipped["notes"])).tolist()) if len(skipped) else "None",
        "best_model_by_pr_auc_slow_onset": best_by_metric.loc[best_by_metric["metric"].eq("best_pr_auc_slow_onset"), "model_name"].iloc[0],
        "best_model_by_f1_slow_onset": best_by_metric.loc[best_by_metric["metric"].eq("best_f1_slow_onset"), "model_name"].iloc[0],
        "best_model_by_brier_score": best_by_metric.loc[best_by_metric["metric"].eq("best_brier_score"), "model_name"].iloc[0],
        "best_model_by_ece": best_by_metric.loc[best_by_metric["metric"].eq("best_ece"), "model_name"].iloc[0],
        "best_model_by_top20_lift": best_by_metric.loc[best_by_metric["metric"].eq("best_top20_lift"), "model_name"].iloc[0],
        "best_overall_recommended_model": best_by_metric.loc[best_by_metric["metric"].eq("best_overall_recommended"), "model_name"].iloc[0],
        "best_overall_recommended_feature_set": best_by_metric.loc[best_by_metric["metric"].eq("best_overall_recommended"), "feature_set"].iloc[0],
        "whether_current_random_forest_recommendation_changes": bool(best_by_metric.loc[best_by_metric["metric"].eq("best_overall_recommended"), "model_name"].iloc[0] != "random_forest"),
        "whether_hrv_enhanced_features_helped_newer_models": hrv_helped,
        "output_folder_location": str(out_dirs["root"].resolve()),
    }
    pd.DataFrame([summary]).to_csv(out_dirs["tables"] / "new_model_benchmark_execution_summary.csv", index=False)
    (out_dirs["tables"] / "new_model_benchmark_execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nNEW MODEL BENCHMARK SUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
