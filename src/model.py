"""
src/model.py — Modelling, evaluation, explainability and business layer.

Run the full training pipeline with:

    python -m src.model

Pipeline (in order)
-------------------
1. Stratified 80/20 split; the 20% test set is locked away until the end.
2. Three candidate models, each a Pipeline(preprocessor -> classifier):
       Logistic Regression (elastic-net) | Random Forest | XGBoost
   (sklearn HistGradientBoosting is used automatically if xgboost is absent.)
3. NESTED cross-validation on the training set:
       inner 5-fold RandomizedSearchCV  -> hyperparameter tuning
       outer 5-fold                     -> unbiased performance estimate
4. Champion selection with the "one-standard-error rule": pick the simplest
   model whose CV ROC-AUC is within 1 SE of the best (simpler = easier to
   explain to a regulator).
5. Re-tune the champion on the full training set; produce out-of-fold
   probabilities; choose the approve/decline cut-off that MAXIMISES EXPECTED
   PROFIT under the bank's loan economics (not accuracy).
6. Single evaluation on the untouched test set: AUC, Gini, KS, PR-AUC,
   Brier, confusion matrix, profit vs. "approve everyone" baseline.
7. Explainability (SHAP), error analysis by segment, fairness audit.
8. Persist the model bundle + metrics + figures to artifacts/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline

from src import data as D

try:                                    # optional heavy dependencies
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                     # pragma: no cover
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:                     # pragma: no cover
    HAS_SHAP = False

ARTIFACTS = D.ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
RS = D.RANDOM_STATE


# ---------------------------------------------------------------------------
# Business economics — every number an executive may want to change
# ---------------------------------------------------------------------------
@dataclass
class Economics:
    avg_ticket_inr: float = 500_000      # average loan size (Rs 5 lakh)
    margin_good: float = 0.10            # lifetime net profit on a good loan, % of principal
    lgd: float = 0.50                    # loss given default, % of principal
    annual_applications: int = 10_000    # yearly application volume

    @property
    def gain_good(self) -> float:        # Rs earned by approving a good loan
        return self.avg_ticket_inr * self.margin_good

    @property
    def loss_bad(self) -> float:         # Rs lost by approving a bad loan
        return self.avg_ticket_inr * self.lgd

    @property
    def breakeven_pd(self) -> float:
        """PD at which approving has zero expected value:
        (1-p)*gain = p*loss  ->  p* = gain / (gain + loss)."""
        return self.gain_good / (self.gain_good + self.loss_bad)


# Note: default 10% margin vs 50% LGD gives a 1:5 cost ratio, identical to
# the official Statlog German Credit cost matrix (false accept costs 5x a
# false reject), so our economics are consistent with the dataset's design.


def expected_profit(y_true, p_default, threshold, econ: Economics) -> float:
    """Total Rs profit if we approve every applicant with PD < threshold."""
    y_true = np.asarray(y_true)
    approve = np.asarray(p_default) < threshold
    good = approve & (y_true == 0)
    bad = approve & (y_true == 1)
    return good.sum() * econ.gain_good - bad.sum() * econ.loss_bad


def rulebook_decline(X: pd.DataFrame) -> np.ndarray:
    """A typical manual-underwriting rulebook used as the business baseline:
    decline if the checking balance is in the lowest band, OR the applicant
    has past repayment delays, OR the tenure exceeds 36 months."""
    return ((X["status"] == "... < 100 DM")
            | (X["credit_history"] == "delay in paying off in the past")
            | (X["duration"] > 36)).to_numpy()


def profit_curve(y_true, p_default, econ: Economics, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    n = len(y_true)
    rows = []
    for t in grid:
        approve = np.asarray(p_default) < t
        rows.append({
            "threshold": t,
            "approval_rate": approve.mean(),
            "bad_rate_in_approved": (np.asarray(y_true)[approve].mean()
                                     if approve.any() else np.nan),
            "profit_per_application": expected_profit(y_true, p_default, t, econ) / n,
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, p_default, econ: Economics) -> float:
    curve = profit_curve(y_true, p_default, econ)
    return float(curve.loc[curve["profit_per_application"].idxmax(), "threshold"])


# ---------------------------------------------------------------------------
# Candidate models and search spaces
# ---------------------------------------------------------------------------
# Class imbalance (30% bad) is handled at the DECISION layer by the
# profit-optimal threshold, not by re-weighting classes. Re-weighting would
# distort the predicted probabilities, and a lender needs honest PDs for
# expected-loss and pricing.
def candidates() -> dict[str, tuple[object, dict, int]]:
    """name -> (classifier, param_distributions, n_iter). Ordered simplest first."""
    models = {
        "Logistic Regression": (
            LogisticRegression(solver="saga", max_iter=5000, random_state=RS),
            {"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": [0.0, 0.5, 1.0]},
            20,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS),
            {"clf__max_depth": [3, 5, 8, 12, None],
             "clf__min_samples_leaf": [1, 3, 5, 10, 20],
             "clf__max_features": ["sqrt", 0.3, 0.5]},
            15,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", n_jobs=-1, random_state=RS, verbosity=0),
            {"clf__n_estimators": randint(100, 600),
             "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__max_depth": [2, 3, 4, 5],
             "clf__min_child_weight": [1, 3, 5, 10],
             "clf__subsample": uniform(0.6, 0.4),
             "clf__colsample_bytree": uniform(0.5, 0.5),
             "clf__reg_lambda": loguniform(0.1, 10)},
            25,
        )
    else:
        warnings.warn("xgboost not installed - using sklearn HistGradientBoosting instead.")
        models["Gradient Boosting (sklearn)"] = (
            HistGradientBoostingClassifier(random_state=RS),
            {"clf__learning_rate": loguniform(0.01, 0.2),
             "clf__max_iter": randint(100, 500),
             "clf__max_depth": [2, 3, 4, 5],
             "clf__min_samples_leaf": [5, 10, 20, 40],
             "clf__l2_regularization": loguniform(1e-3, 10)},
            25,
        )
    return models


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", D.build_preprocessor()), ("clf", clf)])


def make_search(clf, params, n_iter, seed=RS) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        make_pipeline(clone(clf)), params, n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=-1, refit=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ks_statistic(y_true, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative distributions of
    goods and bads — the classic credit-scoring separation metric."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def score_set(y_true, p) -> dict:
    auc = roc_auc_score(y_true, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y_true, p),
            "pr_auc": average_precision_score(y_true, p),
            "brier": brier_score_loss(y_true, p), "log_loss": log_loss(y_true, p)}


def nested_cv(X, y, outer_folds=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer loop estimates generalisation; inner loop (inside the
    RandomizedSearchCV) tunes hyperparameters. Every fold re-fits the
    preprocessor, so scaling/encoding never sees validation data."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=RS + 1)
    fold_rows = []
    for name, (clf, params, n_iter) in candidates().items():
        t0 = time.time()
        for k, (tr, va) in enumerate(outer.split(X, y)):
            search = make_search(clf, params, n_iter, seed=RS + k)
            search.fit(X.iloc[tr], y.iloc[tr])
            p = search.predict_proba(X.iloc[va])[:, 1]
            fold_rows.append({"model": name, "fold": k + 1, **score_set(y.iloc[va], p)})
        print(f"  nested CV  {name:<30s} {time.time() - t0:5.1f}s")
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("model", sort=False).agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.drop(columns=["fold_mean", "fold_std"])
    summary["roc_auc_se"] = summary["roc_auc_std"] / np.sqrt(outer_folds)
    return folds, summary.reset_index()


def pick_champion(summary: pd.DataFrame) -> str:
    """One-standard-error rule. `summary` rows are ordered simplest first."""
    best = summary.loc[summary["roc_auc_mean"].idxmax()]
    cutoff = best["roc_auc_mean"] - best["roc_auc_se"]
    eligible = summary[summary["roc_auc_mean"] >= cutoff]
    return str(eligible.iloc[0]["model"])


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class RiskExplainer:
    """Per-applicant attribution of the default-risk score to each ORIGINAL
    business feature, in log-odds units (positive = pushes risk up).

    * Logistic Regression -> exact linear SHAP: coef_j * (x_j - mean_j),
      summed over each feature's one-hot columns. Needs no extra library.
    * Tree models + `shap` installed -> shap.TreeExplainer (exact TreeSHAP,
      interventional w.r.t. a background sample), raw log-odds output.
    * Tree models without `shap` -> occlusion fallback: replace one feature
      at a time with background values, measure the change in log-odds,
      then rescale so contributions add up exactly like SHAP.
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.pipe = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.groups = D.feature_groups(self.prep)
        self.background = background.reset_index(drop=True)
        self.bg_t = self.prep.transform(self.background)
        if isinstance(self.clf, LogisticRegression):
            self.method = "exact linear SHAP"
        elif HAS_SHAP and isinstance(self.clf, (RandomForestClassifier,)):
            self.method = "TreeSHAP (shap)"
            self._tree = shap.TreeExplainer(self.clf, data=self.bg_t,
                                            feature_perturbation="interventional",
                                            model_output="raw")
        elif HAS_SHAP:
            self.method = "TreeSHAP (shap)"
            self._tree = shap.TreeExplainer(self.clf, data=self.bg_t,
                                            feature_perturbation="interventional")
        else:
            self.method = "occlusion (SHAP-style, additive-rescaled)"
        self.base_value = float(np.mean(_logit(self.pipe.predict_proba(self.background)[:, 1])))

    # -- helpers ------------------------------------------------------------
    def _roll_up(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({f: values[:, idx].sum(axis=1) for f, idx in self.groups.items()})

    def _tree_shap(self, Xt: np.ndarray) -> np.ndarray:
        sv = self._tree.shap_values(Xt)
        if isinstance(sv, list):                 # older shap: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                         # (n, features, classes)
            sv = sv[:, :, 1]
        if isinstance(self.clf, RandomForestClassifier):
            # RF trees output probabilities; map onto the log-odds scale
            # proportionally so every model is reported in the same units.
            p = self.pipe.named_steps["clf"].predict_proba(Xt)[:, 1]
            target = _logit(p) - self.base_value
            tot = sv.sum(axis=1)
            scale = np.divide(target, tot, out=np.ones_like(tot), where=np.abs(tot) > 1e-9)
            sv = sv * scale[:, None]
        return sv

    def _occlusion(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        full = _logit(self.pipe.predict_proba(X)[:, 1])
        bg = self.background.sample(min(50, len(self.background)), random_state=RS)
        out = {}
        for f in D.FEATURES:
            reps = X.loc[X.index.repeat(len(bg))].reset_index(drop=True)
            reps[f] = np.tile(bg[f].to_numpy(), len(X))
            occl = _logit(self.pipe.predict_proba(reps)[:, 1]).reshape(len(X), len(bg)).mean(1)
            out[f] = full - occl
        contrib = pd.DataFrame(out)
        target = full - self.base_value
        tot = contrib.sum(axis=1).to_numpy()
        scale = np.divide(target, tot, out=np.ones_like(tot), where=np.abs(tot) > 1e-9)
        return contrib.mul(scale, axis=0)

    # -- public -------------------------------------------------------------
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows = applicants, columns = original features, values = log-odds
        contribution. base_value + row sum == logit(predicted PD)."""
        X = X[D.FEATURES]
        if self.method == "exact linear SHAP":
            Xt = self.prep.transform(X)
            coef = self.clf.coef_.ravel()
            vals = (Xt - self.bg_t.mean(axis=0)) * coef
            out = self._roll_up(vals)
            # base for the linear model is the logit at the mean input
            self.base_value = float(self.clf.intercept_[0] + self.bg_t.mean(axis=0) @ coef)
            return out
        if self.method.startswith("TreeSHAP"):
            Xt = self.prep.transform(X)
            out = self._roll_up(self._tree_shap(Xt))
            if not isinstance(self.clf, RandomForestClassifier):
                ev = np.atleast_1d(self._tree.expected_value)
                self.base_value = float(ev[-1])
            return out
        return self._occlusion(X)


def reason_codes(contrib_row: pd.Series, applicant: pd.Series, top_n: int = 4):
    """Plain-English adverse-action style reasons for one applicant."""
    def fmt(f):
        v = applicant[f]
        if isinstance(v, (float, np.floating)):
            v = f"{v:,.0f}"
        return f"{D.FEATURE_LABELS[f]}: {v}"
    s = contrib_row.sort_values()
    risk_up = [(fmt(f), float(c)) for f, c in s[::-1].items() if c > 0][:top_n]
    risk_down = [(fmt(f), float(c)) for f, c in s.items() if c < 0][:top_n]
    return risk_up, risk_down


# ---------------------------------------------------------------------------
# Error analysis & fairness
# ---------------------------------------------------------------------------
def error_analysis(X, y, p, threshold) -> pd.DataFrame:
    df = X.copy()
    df["y"] = np.asarray(y)
    df["approve"] = np.asarray(p) < threshold
    df["amount_band"] = pd.qcut(df["amount"], 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"]).astype(str)
    df["age_band"] = pd.cut(df["age"], [0, 25, 35, 50, 120], labels=["<=25", "26-35", "36-50", "50+"]).astype(str)
    rows = []
    for seg in ["status", "purpose", "amount_band", "age_band", "credit_history"]:
        for lvl, g in df.groupby(seg):
            if len(g) < 8:
                continue
            bads, goods = g[g.y == 1], g[g.y == 0]
            rows.append({
                "segment": seg, "level": lvl, "n": len(g),
                "actual_bad_rate": g.y.mean(),
                "missed_defaults_rate": (bads.approve.mean() if len(bads) else np.nan),   # FN rate
                "wrongly_rejected_rate": ((~goods.approve).mean() if len(goods) else np.nan),  # FP rate
            })
    return pd.DataFrame(rows)


def fairness_audit(audit: pd.DataFrame, y, p, threshold) -> pd.DataFrame:
    """Approval-rate parity on attributes the model never saw.
    Disparate-impact ratio < 0.8 (the 'four-fifths rule') is flagged."""
    df = audit.copy()
    df["approve"] = np.asarray(p) < threshold
    df["y"] = np.asarray(y)
    rows = []
    for attr in ["sex", "age_band", "foreign_worker"]:
        rates = df.groupby(attr)["approve"].mean()
        ref = rates.max()
        for lvl, g in df.groupby(attr):
            appr = g[g.approve]
            rows.append({"attribute": attr, "group": lvl, "n": len(g),
                         "approval_rate": g.approve.mean(),
                         "disparate_impact": g.approve.mean() / ref if ref else np.nan,
                         "bad_rate_if_approved": appr.y.mean() if len(appr) else np.nan,
                         "flag": "REVIEW" if g.approve.mean() / ref < 0.8 else "ok"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference API (used by app.py)
# ---------------------------------------------------------------------------
def load_bundle(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def score_applicants(bundle: dict, applicants: pd.DataFrame, threshold: float | None = None,
                     refer_band: float = 0.05) -> pd.DataFrame:
    """Return PD, decision and risk grade for each applicant.
    Applicants within +/- refer_band of the cut-off go to a human underwriter."""
    X = D.engineer_features(applicants)[D.FEATURES]
    t = bundle["threshold"] if threshold is None else threshold
    pd_ = bundle["pipeline"].predict_proba(X)[:, 1]
    decision = np.where(pd_ < t - refer_band, "APPROVE",
                        np.where(pd_ >= t + refer_band, "DECLINE", "REFER TO UNDERWRITER"))
    grade = pd.cut(pd_, [0, .05, .10, .20, .35, .50, 1.0],
                   labels=["A", "B", "C", "D", "E", "F"], include_lowest=True).astype(str)
    return pd.DataFrame({"pd": pd_, "risk_grade": grade, "decision": decision}, index=applicants.index)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
INK, MUTED, ACCENT, BAD = "#1f2937", "#9ca3af", "#2563eb", "#dc2626"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(alpha=.25)


def save_figures(folds, y_test, p_test, curve, threshold, importance, champion):
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = folds["model"].unique()
    ax.boxplot([folds.loc[folds.model == m, "roc_auc"] for m in order], tick_labels=order)
    _style(ax, "Nested 5-fold CV — ROC-AUC by model")
    ax.set_ylabel("ROC-AUC (outer folds)")
    fig.tight_layout(); fig.savefig(FIGURES / "cv_comparison.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_test, p_test)
    axes[0].plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {roc_auc_score(y_test, p_test):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False); _style(axes[0], f"ROC — {champion} (test)")
    pr, rc, _ = precision_recall_curve(y_test, p_test)
    axes[1].plot(rc, pr, color=ACCENT, lw=2, label=f"AP = {average_precision_score(y_test, p_test):.3f}")
    axes[1].axhline(np.mean(y_test), ls="--", color=MUTED, label="base rate")
    axes[1].set_xlabel("Recall (defaults caught)"); axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False); _style(axes[1], "Precision-Recall (test)")
    fig.tight_layout(); fig.savefig(FIGURES / "roc_pr.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    frac, mean_p = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    ax.plot(mean_p, frac, "o-", color=ACCENT, label="model"); ax.plot([0, 1], [0, 1], "--", color=MUTED, label="perfect")
    ax.set_xlabel("Predicted PD"); ax.set_ylabel("Observed default rate"); ax.legend(frameon=False)
    _style(ax, "Calibration (test)")
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.threshold, curve.profit_per_application, color=ACCENT, lw=2)
    ax.axvline(threshold, color=BAD, ls="--", label=f"chosen cut-off = {threshold:.2f}")
    ax.axhline(curve.profit_per_application.iloc[-1], color=MUTED, ls=":", label="approve everyone")
    ax.set_xlabel("PD cut-off (approve if PD below)"); ax.set_ylabel("Profit per application (Rs)")
    ax.legend(frameon=False); _style(ax, "Profit curve (out-of-fold, training set)")
    fig.tight_layout(); fig.savefig(FIGURES / "profit_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_test, (np.asarray(p_test) >= threshold).astype(int))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color=INK)
    ax.set_xticks([0, 1], ["Approve", "Decline"]); ax.set_yticks([0, 1], ["Good", "Bad"])
    ax.set_xlabel("Model decision"); ax.set_ylabel("Actual outcome")
    ax.set_title("Confusion matrix (test)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    imp = importance.sort_values()
    ax.barh([D.FEATURE_LABELS[f] for f in imp.index], imp.values, color=ACCENT)
    ax.set_xlabel("Mean |contribution| (log-odds)"); _style(ax, "Global feature importance (test)")
    fig.tight_layout(); fig.savefig(FIGURES / "global_importance.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    return o


def train(econ: Economics | None = None) -> dict:
    econ = econ or Economics()
    t_start = time.time()
    df = D.load_raw()
    X_train, X_test, y_train, y_test = D.split(df)
    print(f"Data: {len(df)} loans, {df[D.TARGET].mean():.0%} bad | train {len(X_train)} / test {len(X_test)}")
    print(f"xgboost available: {HAS_XGB} | shap available: {HAS_SHAP}")

    # 1-2. nested CV model comparison
    folds, summary = nested_cv(X_train, y_train)
    champion = pick_champion(summary)
    print(summary[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "brier_mean"]].round(3).to_string(index=False))
    print(f"Champion (1-SE rule): {champion}")

    # 3. final tuning of the champion on the full training set
    clf, params, n_iter = candidates()[champion]
    search = make_search(clf, params, n_iter)
    search.fit(X_train, y_train)
    pipe = search.best_estimator_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

    # 4. profit-optimal threshold from out-of-fold predictions (no test data)
    fixed = make_pipeline(clone(pipe.named_steps["clf"]))
    oof = cross_val_predict(fixed, X_train, y_train, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=RS))[:, 1]
    threshold = best_threshold(y_train, oof, econ)
    curve = profit_curve(y_train, oof, econ)

    # 5. ONE look at the test set
    p_test = pipe.predict_proba(X_test)[:, 1]
    test_scores = score_set(y_test, p_test)
    tn, fp, fn, tp = confusion_matrix(y_test, (p_test >= threshold).astype(int)).ravel()
    n_test = len(y_test)
    profit_model = expected_profit(y_test, p_test, threshold, econ) / n_test
    profit_all = expected_profit(y_test, p_test, 1.01, econ) / n_test
    profit_perfect = (y_test == 0).sum() * econ.gain_good / n_test
    rule_p = rulebook_decline(X_test).astype(float)          # 1 = decline
    profit_rule = expected_profit(y_test, rule_p, 0.5, econ) / n_test
    business = {
        "approval_rate_rulebook": float((rule_p < 0.5).mean()),
        "bad_rate_rulebook_book": float(y_test[rule_p < 0.5].mean()),
        "profit_per_app_rulebook": profit_rule,
        "annual_uplift_vs_rulebook_inr": (profit_model - profit_rule) * econ.annual_applications,
        "threshold": threshold,
        "breakeven_pd_theory": econ.breakeven_pd,
        "approval_rate_model": float((p_test < threshold).mean()),
        "bad_rate_approve_all": float(y_test.mean()),
        "bad_rate_model_book": float(y_test[p_test < threshold].mean()),
        "defaults_caught_pct": float(tp / (tp + fn)),
        "good_loans_lost_pct": float(fp / (fp + tn)),
        "profit_per_app_model": profit_model,
        "profit_per_app_approve_all": profit_all,
        "profit_per_app_perfect": profit_perfect,
        "annual_uplift_inr": (profit_model - profit_all) * econ.annual_applications,
        "value_captured_pct": (profit_model - profit_all) / (profit_perfect - profit_all),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # 6. explainability
    bg = X_train.sample(100, random_state=RS)
    explainer = RiskExplainer(pipe, bg)
    contrib_test = explainer.explain(X_test)
    importance = contrib_test.abs().mean().sort_values(ascending=False)

    # 7. error analysis + fairness (out-of-sample predictions for all 1,000)
    errors = error_analysis(X_test, y_test, p_test, threshold)
    all_p = pd.Series(np.nan, index=df.index)
    all_p.loc[X_train.index] = oof
    all_p.loc[X_test.index] = p_test
    fairness = fairness_audit(D.audit_attributes(df), df[D.TARGET], all_p.values, threshold)

    # 8. persist
    ARTIFACTS.mkdir(exist_ok=True)
    save_figures(folds, y_test, p_test, curve, threshold, importance, champion)
    folds.to_csv(ARTIFACTS / "cv_folds.csv", index=False)
    summary.to_csv(ARTIFACTS / "cv_summary.csv", index=False)
    curve.to_csv(ARTIFACTS / "profit_curve.csv", index=False)
    errors.to_csv(ARTIFACTS / "error_analysis.csv", index=False)
    fairness.to_csv(ARTIFACTS / "fairness_audit.csv", index=False)
    importance.rename("mean_abs_contribution").to_csv(ARTIFACTS / "global_importance.csv")

    version = hashlib.sha1(f"{champion}{best_params}{threshold}".encode()).hexdigest()[:8]
    metrics = {
        "model_version": version,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "champion": champion,
        "best_params": best_params,
        "inner_cv_best_auc": float(search.best_score_),
        "explainer": explainer.method,
        "libraries": {"xgboost": HAS_XGB, "shap": HAS_SHAP},
        "cv_summary": summary.round(4).to_dict(orient="records"),
        "test": test_scores,
        "business": business,
        "economics": asdict(econ),
        "excluded_features": D.EXCLUDED_PROTECTED,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), indent=2, default=str))
    bundle = {"pipeline": pipe, "threshold": threshold, "model_name": champion,
              "version": version, "background": bg, "metrics": _jsonable(metrics),
              "categories": D.category_options(df), "ranges": D.numeric_ranges(df),
              "test_set": (X_test, y_test), "oof": (y_train, oof)}
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nTEST  AUC {test_scores['roc_auc']:.3f} | Gini {test_scores['gini']:.3f} | "
          f"KS {test_scores['ks']:.3f} | Brier {test_scores['brier']:.3f}")
    print(f"Cut-off PD < {threshold:.2f} | approve {business['approval_rate_model']:.0%} | "
          f"book bad-rate {business['bad_rate_model_book']:.1%} vs {business['bad_rate_approve_all']:.1%}")
    print(f"Profit/app Rs {profit_model:,.0f} vs approve-all Rs {profit_all:,.0f} -> "
          f"annual uplift Rs {business['annual_uplift_inr'] / 1e7:,.2f} crore")
    print(f"Rulebook: approve {business['approval_rate_rulebook']:.0%}, book bad-rate "
          f"{business['bad_rate_rulebook_book']:.1%}, profit/app Rs {profit_rule:,.0f} -> "
          f"model uplift vs rulebook Rs {business['annual_uplift_vs_rulebook_inr'] / 1e7:,.2f} crore/yr")
    print(f"Saved to {ARTIFACTS} in {metrics['runtime_sec']}s")
    return metrics


if __name__ == "__main__":
    train()
