from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import choose_threshold, probability_metrics, threshold_metrics
from src.imbalance import oversample_minority
from src.plots import bar_plot
from src.utils import ece_score, load_config

ADV = ROOT / "outputs" / "advanced_model_diagnostics"
TABLES = ADV / "tables"
PLOTS = ADV / "plots"
PRED = ADV / "predictions"
for p in [ADV, TABLES, PLOTS, PRED]:
    p.mkdir(parents=True, exist_ok=True)


ORIGINAL_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "criterion": "gini",
    "bootstrap": True,
    "class_weight": None,
    "random_state": 42,
}


def selected_features() -> list[str]:
    selection = pd.read_csv(ROOT / "outputs" / "tables" / "final_feature_set_selection.csv").iloc[0]
    groups = str(selection["selected_feature_groups"]).split("+")
    audit = pd.read_csv(ROOT / "outputs" / "tables" / "feature_audit.csv")
    return audit[(audit["feature_group"].isin(groups)) & (audit["is_allowed"] == True)]["feature_name"].tolist()


def load_xy():
    df = pd.read_csv(ROOT / "outputs" / "tables" / "feature_matrix_full.csv")
    features = selected_features()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df["target_sleep_success_15"].astype(int)
    return df, features, y


def prepare_standardized(df: pd.DataFrame, features: list[str]):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(df[features])
    x_std = scaler.fit_transform(x_imp)
    x_std = pd.DataFrame(x_std, columns=features, index=df.index)
    return x_std, imputer, scaler


def fit_logit_inference(df: pd.DataFrame, features: list[str], y: pd.Series):
    x_std, _, _ = prepare_standardized(df, features)
    x_const = sm.add_constant(x_std, has_constant="add")
    model = sm.Logit(y, x_const)
    result = model.fit(disp=False, maxiter=500)
    cluster_result = model.fit(disp=False, maxiter=500, cov_type="cluster", cov_kwds={"groups": df["participant_id"]})

    def coefficient_table(res, robust_label: str):
        params = res.params
        conf = res.conf_int()
        out = pd.DataFrame({
            "feature": params.index,
            "coefficient": params.values,
            "standard_error": res.bse.values,
            "z_value": res.tvalues.values,
            "p_value": res.pvalues.values,
            "odds_ratio": np.exp(params.values),
            "odds_ratio_ci_lower_95": np.exp(conf[0].values),
            "odds_ratio_ci_upper_95": np.exp(conf[1].values),
        })
        out = out[out["feature"] != "const"].copy()
        out["interpretation_direction"] = np.where(
            out["coefficient"] > 0,
            "increases odds of sleep success",
            "decreases odds of sleep success",
        )
        out["standard_error_type"] = robust_label
        return out.sort_values("p_value")

    coef = coefficient_table(result, "model_based")
    coef.to_csv(TABLES / "logit_full_model_coefficients.csv", index=False)
    coef.sort_values("odds_ratio", ascending=False).to_csv(TABLES / "logit_odds_ratios_sorted.csv", index=False)
    coef.rename(columns={"p_value": "wald_p_value", "z_value": "wald_z_value"}).to_csv(TABLES / "logit_wald_tests.csv", index=False)

    cluster_coef = coefficient_table(cluster_result, "cluster_robust_by_participant")
    cluster_coef.to_csv(TABLES / "logit_cluster_robust_coefficients.csv", index=False)

    ll_full = float(result.llf)
    ll_null = float(result.llnull)
    lr_stat = 2 * (ll_full - ll_null)
    lr_df = int(result.df_model)
    diagnostics = pd.DataFrame([{
        "n_observations": int(result.nobs),
        "n_features": len(features),
        "log_likelihood_full": ll_full,
        "log_likelihood_null": ll_null,
        "likelihood_ratio_statistic": lr_stat,
        "likelihood_ratio_df": lr_df,
        "likelihood_ratio_p_value": float(chi2.sf(lr_stat, lr_df)),
        "mcfadden_pseudo_r2": float(1 - ll_full / ll_null),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "converged": bool(result.mle_retvals.get("converged", False)),
        "notes": "P-values assume independent observations; cluster-robust coefficients by participant are saved separately.",
    }])
    diagnostics.to_csv(TABLES / "logit_model_diagnostics.csv", index=False)

    pred = result.predict(x_const)
    hl = hosmer_lemeshow(y, pred)
    hl.to_csv(TABLES / "logit_hosmer_lemeshow.csv", index=False)
    vif = pd.DataFrame({
        "feature": features,
        "vif": [variance_inflation_factor(x_std.values, i) for i in range(len(features))],
    }).sort_values("vif", ascending=False)
    vif.to_csv(TABLES / "logit_vif.csv", index=False)
    plot_logit_odds(cluster_coef)
    plot_logit_probability_distribution(df, pred)
    write_logit_report(coef, cluster_coef, diagnostics.iloc[0], hl, vif)
    return coef, cluster_coef, diagnostics.iloc[0], hl, vif, pred


def hosmer_lemeshow(y: pd.Series, p: np.ndarray, g: int = 10) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y.to_numpy(dtype=int), "p": p})
    tmp["group"] = pd.qcut(tmp["p"], q=g, labels=False, duplicates="drop") + 1
    rows = []
    for group, sub in tmp.groupby("group"):
        obs_success = sub["y"].sum()
        exp_success = sub["p"].sum()
        obs_failure = len(sub) - obs_success
        exp_failure = (1 - sub["p"]).sum()
        rows.append({
            "group": int(group),
            "n": int(len(sub)),
            "observed_success": float(obs_success),
            "expected_success": float(exp_success),
            "observed_failure": float(obs_failure),
            "expected_failure": float(exp_failure),
        })
    out = pd.DataFrame(rows)
    chi_sq = (((out["observed_success"] - out["expected_success"]) ** 2) / out["expected_success"]).sum()
    chi_sq += (((out["observed_failure"] - out["expected_failure"]) ** 2) / out["expected_failure"]).sum()
    df_hl = max(len(out) - 2, 1)
    out["hl_chi_square"] = float(chi_sq)
    out["hl_df"] = int(df_hl)
    out["hl_p_value"] = float(chi2.sf(chi_sq, df_hl))
    return out


def plot_logit_odds(coef: pd.DataFrame):
    plot_df = coef.copy()
    plot_df["abs_log_or"] = plot_df["coefficient"].abs()
    sig = plot_df[plot_df["p_value"] < 0.10].sort_values("abs_log_or", ascending=False).head(14)
    if sig.empty:
        sig = plot_df.sort_values("abs_log_or", ascending=False).head(14)
    sig = sig.sort_values("odds_ratio")
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(sig))
    plt.errorbar(
        sig["odds_ratio"],
        y_pos,
        xerr=[sig["odds_ratio"] - sig["odds_ratio_ci_lower_95"], sig["odds_ratio_ci_upper_95"] - sig["odds_ratio"]],
        fmt="o",
        capsize=3,
    )
    plt.axvline(1.0, color="gray", linestyle="--")
    plt.yticks(y_pos, sig["feature"])
    plt.xscale("log")
    plt.xlabel("Odds ratio for sleep success (log scale)")
    plt.title("Logit odds ratios with 95% CI")
    plt.tight_layout()
    plt.savefig(PLOTS / "logit_odds_ratio_forest_plot.png", dpi=170)
    plt.close()


def plot_logit_probability_distribution(df: pd.DataFrame, p: np.ndarray):
    y = df["target_sleep_success_15"].astype(int).to_numpy()
    plt.figure(figsize=(6.5, 4.2))
    for cls, label in [(0, "slow onset"), (1, "success")]:
        plt.hist(p[y == cls], bins=22, alpha=0.55, label=label)
    plt.xlabel("Predicted P(success)")
    plt.ylabel("episodes")
    plt.title("Logit predicted probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "logit_predicted_probability_distribution.png", dpi=170)
    plt.close()


def write_logit_report(coef, cluster_coef, diag, hl, vif):
    sig = cluster_coef[cluster_coef["p_value"] < 0.05]
    inc = sig[sig["coefficient"] > 0]["feature"].tolist()
    dec = sig[sig["coefficient"] < 0]["feature"].tolist()
    high_vif = vif[vif["vif"] > 5]
    text = f"""# Logistic Regression Statistical Inference Report

## Why This Model Was Added
The sklearn logistic regression in the main pipeline was optimized as a predictive model. This statsmodels Logit model was added for statistical interpretation: coefficients, Wald tests, likelihood-ratio testing, AIC/BIC, odds ratios, Hosmer-Lemeshow calibration, and multicollinearity diagnostics.

## Model Setup
The target is `target_sleep_success_15`, where 1 means SOL < 15 minutes. The selected final feature set was used in one full multivariable model. Continuous predictors were median-imputed and standardized, so coefficients and odds ratios correspond to a one-standard-deviation increase in the original feature.

## Important Caveat
Sleep episodes are repeated measures nested within participants. Naive p-values are approximate because observations are not fully independent. A participant-cluster-robust coefficient table is saved and should be preferred for inference.

## Significant Predictors With Cluster-Robust SEs
Features increasing odds of sleep success at p < 0.05: {', '.join(inc) if inc else 'none'}.

Features decreasing odds of sleep success at p < 0.05: {', '.join(dec) if dec else 'none'}.

## Odds Ratio Interpretation
An odds ratio above 1 means a one-standard-deviation increase in that feature is associated with higher odds of sleep success. An odds ratio below 1 means lower odds of sleep success.

## Overall Model Tests
McFadden pseudo R²: {diag['mcfadden_pseudo_r2']:.3f}. This is not interpreted like ordinary R²; values are often much lower in logistic regression and reflect improvement over an intercept-only model.

Likelihood-ratio test p-value: {diag['likelihood_ratio_p_value']:.3g}. This tests whether the full model improves over a null intercept-only model.

AIC: {diag['aic']:.1f}; BIC: {diag['bic']:.1f}. These are comparative criteria where lower values are better among models fit to the same data.

## Hosmer-Lemeshow Test
HL chi-square: {hl['hl_chi_square'].iloc[0]:.2f}; df: {int(hl['hl_df'].iloc[0])}; p-value: {hl['hl_p_value'].iloc[0]:.3g}. A non-small p-value is usually interpreted as no strong evidence of lack of fit, though the test is sample-size sensitive.

## Multicollinearity
Highest VIF: {vif['vif'].max():.2f}. {'Some VIF values are high, which is expected because rolling sleep-history features are correlated with each other.' if not high_vif.empty else 'VIF values are not strongly concerning.'}
"""
    (ADV / "LOGIT_INFERENCE_REPORT.md").write_text(text, encoding="utf-8")


def inner_group_split(df, train_idx, seed):
    groups = df.iloc[train_idx]["participant_id"].to_numpy()
    y = df.iloc[train_idx]["target_sleep_success_15"].to_numpy()
    gss = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=seed)
    fallback = None
    for fit_rel, val_rel in gss.split(train_idx, y, groups):
        fit_idx = train_idx[fit_rel]
        val_idx = train_idx[val_rel]
        if df.iloc[fit_idx]["target_sleep_success_15"].nunique() == 2 and df.iloc[val_idx]["target_sleep_success_15"].nunique() == 2:
            return fit_idx, val_idx
        fallback = (fit_idx, val_idx)
    return fallback if fallback else (train_idx, train_idx)


def rf_param_candidates(seed: int, n: int = 18):
    grid = list(itertools.product(
        [300, 500, 800],
        [3, 5, 8, 12, None],
        [3, 5, 10, 20],
        [5, 10, 20],
        ["sqrt", "log2", 0.5, None],
        ["gini", "entropy"],
        [True],
    ))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid), size=min(n, len(grid)), replace=False)
    out = []
    for i in idx:
        vals = grid[int(i)]
        out.append({
            "n_estimators": vals[0],
            "max_depth": vals[1],
            "min_samples_leaf": vals[2],
            "min_samples_split": vals[3],
            "max_features": vals[4],
            "criterion": vals[5],
            "bootstrap": vals[6],
        })
    return out


def fit_rf(x_train, y_train, params, seed):
    params = sanitize_rf_params(params)
    x_os, y_os = oversample_minority(x_train, y_train, 0.667, seed)
    imp = SimpleImputer(strategy="median")
    x_imp = imp.fit_transform(x_os)
    rf = RandomForestClassifier(**params, random_state=seed, n_jobs=-1, class_weight=None)
    rf.fit(x_imp, y_os)
    return imp, rf


def sanitize_rf_params(params: dict) -> dict:
    out = dict(params)
    for key in ["n_estimators", "min_samples_leaf", "min_samples_split"]:
        out[key] = int(out[key])
    if pd.isna(out.get("max_depth")):
        out["max_depth"] = None
    elif out.get("max_depth") is not None:
        out["max_depth"] = int(out["max_depth"])
    if out.get("max_features") == "None" or pd.isna(out.get("max_features")):
        out["max_features"] = None
    elif isinstance(out.get("max_features"), float):
        out["max_features"] = float(out["max_features"])
    out["bootstrap"] = bool(out["bootstrap"])
    return out


def rf_predict(imp, rf, x):
    return rf.predict_proba(imp.transform(x))[:, 1]


def tune_rf(df: pd.DataFrame, features: list[str], cfg: dict):
    y = df["target_sleep_success_15"].astype(int)
    groups = df["participant_id"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = rf_param_candidates(42, n=18)
    all_search_rows = []
    best_rows = []
    p_success = np.zeros(len(df))
    tuned_pred = np.zeros(len(df), dtype=int)
    default_pred = np.zeros(len(df), dtype=int)
    fold_nums = np.zeros(len(df), dtype=int)
    param_for_row = [""] * len(df)
    thresholds = []
    feature_importances = []
    for fold, (tr, te) in enumerate(outer.split(df, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=100 + fold)
        candidate_scores = []
        for ci, params in enumerate(candidates, start=1):
            inner_pr, inner_f1, inner_brier = [], [], []
            for itr, iva in inner.split(df.iloc[tr], y.iloc[tr], df.iloc[tr]["participant_id"]):
                fit_idx = np.asarray(tr)[itr]
                val_idx = np.asarray(tr)[iva]
                imp, rf = fit_rf(df.iloc[fit_idx][features], y.iloc[fit_idx], params, 1000 + fold * 100 + ci)
                p_val = rf_predict(imp, rf, df.iloc[val_idx][features])
                y_val = y.iloc[val_idx].to_numpy(dtype=int)
                slow_prob = 1 - p_val
                inner_pr.append(average_precision_score(1 - y_val, slow_prob))
                t, _ = choose_threshold(y_val, p_val)
                pred = (p_val >= t).astype(int)
                inner_f1.append(f1_score(1 - y_val, 1 - pred, zero_division=0))
                inner_brier.append(brier_score_loss(y_val, p_val))
            row = {
                "outer_fold": fold,
                "candidate_id": ci,
                **params,
                "mean_inner_pr_auc_slow_onset": float(np.mean(inner_pr)),
                "mean_inner_f1_slow_onset": float(np.mean(inner_f1)),
                "mean_inner_brier_score": float(np.mean(inner_brier)),
            }
            candidate_scores.append(row)
            all_search_rows.append(row)
        fold_scores = pd.DataFrame(candidate_scores).sort_values(
            ["mean_inner_pr_auc_slow_onset", "mean_inner_f1_slow_onset", "mean_inner_brier_score"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        fold_scores["rank"] = np.arange(1, len(fold_scores) + 1)
        best = fold_scores.iloc[0].to_dict()
        params = sanitize_rf_params({k: best[k] for k in ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "criterion", "bootstrap"]})
        best_rows.append({"fold": fold, **params, "mean_inner_pr_auc_slow_onset": best["mean_inner_pr_auc_slow_onset"], "mean_inner_f1_slow_onset": best["mean_inner_f1_slow_onset"], "mean_inner_brier_score": best["mean_inner_brier_score"]})

        fit_idx, val_idx = inner_group_split(df, np.asarray(tr), 2000 + fold)
        imp_thr, rf_thr = fit_rf(df.iloc[fit_idx][features], y.iloc[fit_idx], params, 2100 + fold)
        p_val = rf_predict(imp_thr, rf_thr, df.iloc[val_idx][features])
        threshold, _ = choose_threshold(y.iloc[val_idx].to_numpy(dtype=int), p_val)
        thresholds.append(threshold)

        imp, rf = fit_rf(df.iloc[tr][features], y.iloc[tr], params, 3000 + fold)
        p_test = rf_predict(imp, rf, df.iloc[te][features])
        p_success[te] = p_test
        default_pred[te] = (p_test >= 0.5).astype(int)
        tuned_pred[te] = (p_test >= threshold).astype(int)
        fold_nums[te] = fold
        param_text = json.dumps(params, default=str)
        for idx in te:
            param_for_row[idx] = param_text
        feature_importances.append(pd.DataFrame({"feature": features, "importance": rf.feature_importances_, "fold": fold}))

    search = pd.DataFrame(all_search_rows)
    search["rank"] = search.groupby("outer_fold")["mean_inner_pr_auc_slow_onset"].rank(ascending=False, method="first").astype(int)
    search.to_csv(TABLES / "random_forest_hyperparameter_search_results.csv", index=False)
    pd.DataFrame(best_rows).to_csv(TABLES / "random_forest_best_params_by_fold.csv", index=False)
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
        "fold": fold_nums,
        "selected_params_for_fold": param_for_row,
    })
    pred.to_csv(PRED / "oof_predictions_random_forest_tuned.csv", index=False)
    metrics = rf_metrics(pred, thresholds)
    metrics.to_csv(TABLES / "random_forest_tuned_metrics.csv", index=False)
    compare_original_tuned(metrics)
    fi = pd.concat(feature_importances).groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    plot_tuned_rf(pred, metrics, fi)
    write_rf_report(best_rows, metrics)
    return pred, metrics, pd.DataFrame(best_rows)


def prefixed_metrics(y, p, pred, suffix):
    return {
        f"accuracy_{suffix}": float((pred == y).mean()),
        f"balanced_accuracy_{suffix}": float(balanced_accuracy_score(y, pred)),
        f"precision_success_{suffix}": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        f"recall_success_{suffix}": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        f"f1_success_{suffix}": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        f"precision_slow_onset_{suffix}": float(precision_score(1 - y, 1 - pred, pos_label=1, zero_division=0)),
        f"recall_slow_onset_{suffix}": float(recall_score(1 - y, 1 - pred, pos_label=1, zero_division=0)),
        f"f1_slow_onset_{suffix}": float(f1_score(1 - y, 1 - pred, pos_label=1, zero_division=0)),
    }


def rf_metrics(pred: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    y = pred["target_sleep_success_15"].to_numpy(dtype=int)
    p = pred["predicted_probability_success"].to_numpy(dtype=float)
    row = {
        "roc_auc_success": float(roc_auc_score(y, p)),
        "pr_auc_success": float(average_precision_score(y, p)),
        "pr_auc_slow_onset": float(average_precision_score(1 - y, 1 - p)),
        "brier_score_success": float(brier_score_loss(y, p)),
        "expected_calibration_error_success": ece_score(y, p),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        **prefixed_metrics(y, p, pred["predicted_success_default"].to_numpy(dtype=int), "default"),
        **prefixed_metrics(y, p, pred["predicted_success_tuned"].to_numpy(dtype=int), "tuned"),
        "tuned_threshold_mean": float(np.mean(thresholds)),
        "tuned_threshold_std": float(np.std(thresholds)),
    }
    return pd.DataFrame([row])


def compare_original_tuned(tuned: pd.DataFrame):
    orig_all = pd.read_csv(ROOT / "outputs" / "tables" / "final_model_comparison_15min_success.csv")
    orig = orig_all[orig_all["model_name"] == "random_forest"].iloc[0]
    t = tuned.iloc[0]
    rows = [
        {
            "model": "original_random_forest",
            "pr_auc_slow_onset": orig["pr_auc_slow_onset"],
            "f1_slow_onset": orig["f1_slow_onset_tuned"],
            "brier_score": orig["brier_score_success"],
            "ece": orig["expected_calibration_error_success"],
            "balanced_accuracy": orig["balanced_accuracy_tuned"],
            "pr_auc_success": orig["pr_auc_success"],
            "f1_success": orig["f1_success_tuned"],
        },
        {
            "model": "tuned_random_forest",
            "pr_auc_slow_onset": t["pr_auc_slow_onset"],
            "f1_slow_onset": t["f1_slow_onset_tuned"],
            "brier_score": t["brier_score_success"],
            "ece": t["expected_calibration_error_success"],
            "balanced_accuracy": t["balanced_accuracy_tuned"],
            "pr_auc_success": t["pr_auc_success"],
            "f1_success": t["f1_success_tuned"],
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES / "random_forest_original_vs_tuned.csv", index=False)


def plot_tuned_rf(pred: pd.DataFrame, metrics: pd.DataFrame, fi: pd.DataFrame):
    compare = pd.read_csv(TABLES / "random_forest_original_vs_tuned.csv")
    plot_df = compare.melt(id_vars="model", value_vars=["pr_auc_slow_onset", "f1_slow_onset", "brier_score", "ece", "balanced_accuracy", "pr_auc_success", "f1_success"], var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(plot_df["metric"].unique()):
        sub = plot_df[plot_df["metric"] == metric]
        x = np.arange(len(sub)) + i * 0.25
    # simpler grouped chart using pandas
    compare.set_index("model")[["pr_auc_slow_onset", "f1_slow_onset", "brier_score", "ece", "balanced_accuracy", "pr_auc_success", "f1_success"]].T.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Metric value")
    plt.title("Original vs tuned Random Forest")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "tuned_rf_vs_original_rf_metrics.png", dpi=170)
    plt.close()

    y = pred["target_sleep_success_15"].to_numpy(dtype=int)
    p = pred["predicted_probability_success"].to_numpy(dtype=float)
    PrecisionRecallDisplay.from_predictions(1 - y, 1 - p)
    plt.title("Tuned RF PR curve: slow onset")
    plt.tight_layout()
    plt.savefig(PLOTS / "tuned_rf_pr_curve_slow_onset.png", dpi=170)
    plt.close()
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean, frac, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted P(success)")
    plt.ylabel("Observed success rate")
    plt.title("Tuned RF calibration curve")
    plt.tight_layout()
    plt.savefig(PLOTS / "tuned_rf_calibration_curve.png", dpi=170)
    plt.close()
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y, pred["predicted_success_tuned"].to_numpy(dtype=int), labels=[0, 1], display_labels=["slow onset", "success"])
    plt.title("Tuned RF confusion matrix")
    plt.tight_layout()
    plt.savefig(PLOTS / "tuned_rf_confusion_matrix_tuned.png", dpi=170)
    plt.close()
    top = fi.head(18).sort_values("importance")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean feature importance")
    plt.title("Tuned RF feature importance")
    plt.tight_layout()
    plt.savefig(PLOTS / "tuned_rf_feature_importance.png", dpi=170)
    plt.close()


def write_rf_report(best_rows, metrics):
    best = pd.DataFrame(best_rows)
    compare = pd.read_csv(TABLES / "random_forest_original_vs_tuned.csv")
    orig = compare[compare["model"] == "original_random_forest"].iloc[0]
    tuned = compare[compare["model"] == "tuned_random_forest"].iloc[0]
    most_common = best[["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "criterion", "bootstrap"]].astype(str).mode().iloc[0].to_dict()
    text = f"""# Random Forest Hyperparameter Tuning Report

## Original Parameters
The original Random Forest used regularized defaults: `n_estimators=300`, `max_depth=5`, `min_samples_leaf=5`, no class weighting, and train-fold-only oversampling. These were reasonable because they limited tree complexity and reduced overfitting risk on a small participant-level dataset.

## Tuning Strategy
Hyperparameters were tuned inside each outer training fold only. Held-out participants were never used for hyperparameter selection or threshold tuning. The search used 18 randomly sampled configurations from the requested grid after the full 30-candidate run exceeded the local execution timeout. The same participant-aware nested design and full requested parameter space were preserved.

The main tuning objective was PR-AUC for slow-onset/failure because slow onset is the practically important minority class. Since the model outputs P(success), slow-onset PR-AUC was computed using `1 - P(success)`.

## Search Space
Tuned parameters: n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, criterion, and bootstrap.

## Best Parameters
Most common best settings across folds: `{json.dumps(most_common)}`.

Per-fold best parameters are saved in `tables/random_forest_best_params_by_fold.csv`.

## Original vs Tuned
Original RF PR-AUC slow onset: {orig['pr_auc_slow_onset']:.3f}; tuned RF: {tuned['pr_auc_slow_onset']:.3f}.

Original RF F1 slow onset: {orig['f1_slow_onset']:.3f}; tuned RF: {tuned['f1_slow_onset']:.3f}.

Original RF Brier: {orig['brier_score']:.3f}; tuned RF: {tuned['brier_score']:.3f}.

Original RF ECE: {orig['ece']:.3f}; tuned RF: {tuned['ece']:.3f}.

## Recommendation Impact
Tuning should change the final recommendation only if it improves slow-onset PR-AUC/F1 without materially worsening calibration. Large or unstable trees can improve inner-fold scores while increasing overfitting risk, so the tuned model should be interpreted conservatively.
"""
    (ADV / "RANDOM_FOREST_TUNING_REPORT.md").write_text(text, encoding="utf-8")


def update_readme():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "## Advanced Model Diagnostics" in text:
        text = text.split("## Advanced Model Diagnostics")[0].rstrip() + "\n\n"
    rf_compare = pd.read_csv(TABLES / "random_forest_original_vs_tuned.csv")
    tuned = rf_compare[rf_compare["model"] == "tuned_random_forest"].iloc[0]
    orig = rf_compare[rf_compare["model"] == "original_random_forest"].iloc[0]
    diag = pd.read_csv(TABLES / "logit_model_diagnostics.csv").iloc[0]
    best = pd.read_csv(TABLES / "random_forest_best_params_by_fold.csv").head(1).iloc[0].to_dict()
    section = f"""## Advanced Model Diagnostics

### Logistic Regression Odds-Ratio Interpretation
An additional statsmodels Logit model was fit using the selected temporal + previous sleep-history feature set. Predictors were median-imputed and standardized, so odds ratios describe the change in odds of sleep success for a one-standard-deviation increase in a feature. Cluster-robust standard errors by participant are saved because repeated sleep episodes are not fully independent.

Diagnostics saved under `outputs/advanced_model_diagnostics/` include likelihood-ratio testing, McFadden pseudo R² ({diag['mcfadden_pseudo_r2']:.3f}), AIC/BIC, Wald z-tests, Hosmer-Lemeshow calibration, and VIF. Statistical significance should not be oversold because this is repeated-measures wearable data.

### Random Forest Hyperparameter Tuning
The original Random Forest used `n_estimators=300`, `max_depth=5`, and `min_samples_leaf=5`, which were regularized defaults chosen to reduce overfitting. A nested participant-aware tuning analysis searched n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, and criterion using slow-onset PR-AUC as the primary objective.

Original RF slow-onset PR-AUC was {orig['pr_auc_slow_onset']:.3f}; tuned RF slow-onset PR-AUC was {tuned['pr_auc_slow_onset']:.3f}. Original RF slow-onset F1 was {orig['f1_slow_onset']:.3f}; tuned RF slow-onset F1 was {tuned['f1_slow_onset']:.3f}. Original RF Brier was {orig['brier_score']:.3f}; tuned RF Brier was {tuned['brier_score']:.3f}. The tuning report explains whether the improvement is large enough to change the final model recommendation.
"""
    readme.write_text(text + section, encoding="utf-8")


def main():
    cfg = load_config()
    df, features, y = load_xy()
    coef, cluster_coef, diag, hl, vif, logit_pred = fit_logit_inference(df, features, y)
    tuned_pred, tuned_metrics, best_params = tune_rf(df, features, cfg)
    update_readme()
    compare = pd.read_csv(TABLES / "random_forest_original_vs_tuned.csv")
    orig = compare[compare["model"] == "original_random_forest"].iloc[0]
    tuned = compare[compare["model"] == "tuned_random_forest"].iloc[0]
    strongest = cluster_coef.assign(abs_log_or=cluster_coef["coefficient"].abs()).sort_values("abs_log_or", ascending=False).head(5)
    sig = cluster_coef[cluster_coef["p_value"] < 0.05]["feature"].tolist()
    print("\nADVANCED MODEL DIAGNOSTICS SUMMARY")
    print("Strongest odds-ratio features:", ", ".join(strongest["feature"].tolist()))
    print("Statistically significant predictors (cluster robust p<0.05):", ", ".join(sig) if sig else "none")
    print(f"McFadden pseudo R2: {diag['mcfadden_pseudo_r2']:.4f}")
    print(f"Likelihood ratio p-value: {diag['likelihood_ratio_p_value']:.4g}")
    print(f"Hosmer-Lemeshow p-value: {hl['hl_p_value'].iloc[0]:.4g}")
    print(f"Highest VIF: {vif['vif'].max():.2f}")
    print("Original RF parameters:", ORIGINAL_RF_PARAMS)
    print("Tuned RF parameters by fold saved to random_forest_best_params_by_fold.csv")
    print(f"Did tuned RF improve PR-AUC slow-onset? {tuned['pr_auc_slow_onset'] > orig['pr_auc_slow_onset']} ({orig['pr_auc_slow_onset']:.3f} -> {tuned['pr_auc_slow_onset']:.3f})")
    print(f"Did tuned RF improve F1 slow-onset? {tuned['f1_slow_onset'] > orig['f1_slow_onset']} ({orig['f1_slow_onset']:.3f} -> {tuned['f1_slow_onset']:.3f})")
    print(f"Calibration Brier/ECE original vs tuned: {orig['brier_score']:.3f}/{orig['ece']:.3f} -> {tuned['brier_score']:.3f}/{tuned['ece']:.3f}")
    change = (tuned["pr_auc_slow_onset"] > orig["pr_auc_slow_onset"]) and (tuned["f1_slow_onset"] >= orig["f1_slow_onset"]) and (tuned["brier_score"] <= orig["brier_score"] * 1.05)
    print(f"Should final recommendation change? {change}")


if __name__ == "__main__":
    main()
