"""
app.py — CreditRisk AI: executive-facing loan underwriting console.

    streamlit run app.py

Tabs
----
1. Score an applicant   - live inference, PD, decision, reason codes, explanation chart
2. Batch scoring        - upload a CSV of applications, download decisions
3. Portfolio & ROI      - change loan economics / cut-off and see the profit impact
4. Model card           - validation results, fairness audit, error analysis
Every single-applicant decision is written to logs/decision_log.csv (audit trail).
"""
from __future__ import annotations

import datetime as dt
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import data as D
from src import model as M

st.set_page_config(page_title="CreditRisk AI", page_icon="🏦", layout="wide")
LOG_PATH = D.ROOT / "logs" / "decision_log.csv"
DECISION_COLOR = {"APPROVE": "#15803d", "REFER TO UNDERWRITER": "#b45309", "DECLINE": "#b91c1c"}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load():
    bundle = M.load_bundle()
    explainer = M.RiskExplainer(bundle["pipeline"], bundle["background"])
    return bundle, explainer


def inr(x: float) -> str:
    """Indian-style rupee formatting with lakh / crore."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:,.2f} L"
    return f"{sign}₹{x:,.0f}"


def contribution_chart(contrib: pd.Series, top_n: int = 10):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1))
    colors = ["#b91c1c" if v > 0 else "#15803d" for v in s.values]
    ax.barh([D.FEATURE_LABELS[f] for f in s.index], s.values, color=colors)
    ax.axvline(0, color="#6b7280", lw=1)
    ax.set_xlabel("← lowers default risk      contribution (log-odds)      raises default risk →")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def log_decision(row: dict):
    LOG_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if not M.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.model` first, then reload this page.")
    st.stop()

bundle, explainer = load()
metrics = bundle["metrics"]
X_test, y_test = bundle["test_set"]
y_oof, p_oof = bundle["oof"]

# ---------------------------------------------------------------------------
# Sidebar — business levers
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 CreditRisk AI")
st.sidebar.caption(f"Model **{bundle['model_name']}** · version `{bundle['version']}`")
st.sidebar.subheader("Loan economics")
ticket = st.sidebar.number_input("Average loan size (₹)", 50_000, 5_000_000, 500_000, step=50_000)
margin = st.sidebar.slider("Lifetime profit on a good loan (% of loan)", 2, 30, 10) / 100
lgd = st.sidebar.slider("Loss given default (% of loan)", 10, 90, 50) / 100
volume = st.sidebar.number_input("Applications per year", 1_000, 1_000_000, 10_000, step=1_000)
econ = M.Economics(ticket, margin, lgd, int(volume))

opt_t = M.best_threshold(y_oof, p_oof, econ)
st.sidebar.subheader("Decision policy")
threshold = st.sidebar.slider("Approve if PD is below", 0.05, 0.60, float(round(opt_t, 2)), 0.01,
                              help=f"Profit-optimal cut-off for these economics: {opt_t:.2f}")
refer = st.sidebar.slider("Refer-to-underwriter band (±)", 0.0, 0.10, 0.03, 0.01)
st.sidebar.caption(f"Break-even PD in theory: **{econ.breakeven_pd:.2f}** · "
                   f"data-optimal cut-off: **{opt_t:.2f}**")

tab_score, tab_batch, tab_roi, tab_card = st.tabs(
    ["🔍 Score an applicant", "📂 Batch scoring", "💰 Portfolio & ROI", "📋 Model card"])

# ---------------------------------------------------------------------------
# Tab 1 — single applicant
# ---------------------------------------------------------------------------
with tab_score:
    st.header("Score a loan application")
    cats, ranges = bundle["categories"], bundle["ranges"]

    # Preset applicants from the hold-out set so the live demo uses real data
    p_t = bundle["pipeline"].predict_proba(X_test)[:, 1]
    order = np.argsort(p_t)
    presets = {"— enter manually —": None,
               "Sample: low-risk applicant": X_test.iloc[order[5]],
               "Sample: borderline applicant": X_test.iloc[order[min(np.searchsorted(p_t[order], threshold), len(order) - 1)]],
               "Sample: high-risk applicant": X_test.iloc[order[-5]]}
    choice = st.selectbox("Start from", list(presets))
    base = presets[choice]

    def default(col):
        if base is not None:
            return base[col]
        return ranges[col][2] if col in ranges else cats[col][0]

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Loan request**")
            amount = st.number_input(D.FEATURE_LABELS["amount"] + " (DM, dataset units)",
                                     int(ranges["amount"][0]), int(ranges["amount"][1]), int(default("amount")), step=250)
            duration = st.slider(D.FEATURE_LABELS["duration"], 4, 72, int(default("duration")))
            purpose = st.selectbox(D.FEATURE_LABELS["purpose"], cats["purpose"], index=cats["purpose"].index(default("purpose")))
            installment_rate = st.slider(D.FEATURE_LABELS["installment_rate"], 1, 4, int(default("installment_rate")))
            other_debtors = st.selectbox(D.FEATURE_LABELS["other_debtors"], cats["other_debtors"], index=cats["other_debtors"].index(default("other_debtors")))
            prop = st.selectbox(D.FEATURE_LABELS["property"], cats["property"], index=cats["property"].index(default("property")))
        with c2:
            st.markdown("**Banking relationship**")
            status = st.selectbox(D.FEATURE_LABELS["status"], cats["status"], index=cats["status"].index(default("status")))
            savings = st.selectbox(D.FEATURE_LABELS["savings"], cats["savings"], index=cats["savings"].index(default("savings")))
            credit_history = st.selectbox(D.FEATURE_LABELS["credit_history"], cats["credit_history"], index=cats["credit_history"].index(default("credit_history")))
            number_credits = st.slider(D.FEATURE_LABELS["number_credits"], 1, 4, int(default("number_credits")))
            other_plans = st.selectbox(D.FEATURE_LABELS["other_installment_plans"], cats["other_installment_plans"], index=cats["other_installment_plans"].index(default("other_installment_plans")))
        with c3:
            st.markdown("**Applicant profile**")
            age = st.slider(D.FEATURE_LABELS["age"], 18, 80, int(default("age")))
            employment = st.selectbox(D.FEATURE_LABELS["employment_duration"], cats["employment_duration"], index=cats["employment_duration"].index(default("employment_duration")))
            job = st.selectbox(D.FEATURE_LABELS["job"], cats["job"], index=cats["job"].index(default("job")))
            housing = st.selectbox(D.FEATURE_LABELS["housing"], cats["housing"], index=cats["housing"].index(default("housing")))
            residence = st.slider(D.FEATURE_LABELS["present_residence"], 1, 4, int(default("present_residence")))
            people = st.slider(D.FEATURE_LABELS["people_liable"], 1, 2, int(default("people_liable")))
            telephone = st.selectbox(D.FEATURE_LABELS["telephone"], cats["telephone"], index=cats["telephone"].index(default("telephone")))
        submitted = st.form_submit_button("Assess credit risk", type="primary", width="stretch")

    if submitted:
        applicant = pd.DataFrame([{
            "duration": duration, "amount": amount, "installment_rate": installment_rate,
            "present_residence": residence, "age": age, "number_credits": number_credits,
            "people_liable": people, "status": status, "credit_history": credit_history,
            "purpose": purpose, "savings": savings, "employment_duration": employment,
            "other_debtors": other_debtors, "property": prop, "other_installment_plans": other_plans,
            "housing": housing, "job": job, "telephone": telephone}])
        res = M.score_applicants(bundle, applicant, threshold, refer).iloc[0]
        X_app = D.engineer_features(applicant)[D.FEATURES]
        contrib = explainer.explain(X_app).iloc[0]
        up, down = M.reason_codes(contrib, X_app.iloc[0])
        ev = (1 - res.pd) * econ.gain_good - res.pd * econ.loss_bad

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Probability of default", f"{res.pd:.1%}")
        k2.metric("Risk grade", res.risk_grade)
        k3.metric("Expected value if approved", inr(ev))
        k4.metric("Cut-off", f"PD < {threshold:.2f}")
        color = DECISION_COLOR[res.decision]
        st.markdown(f"<div style='padding:14px;border-radius:10px;background:{color};color:white;"
                    f"font-size:22px;font-weight:700;text-align:center'>{res.decision}</div>",
                    unsafe_allow_html=True)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Why — key reasons")
            st.markdown("**Factors raising risk**")
            for txt, v in up:
                st.markdown(f"- 🔺 {txt}  `+{v:.2f}`")
            st.markdown("**Factors lowering risk**")
            for txt, v in down:
                st.markdown(f"- 🟢 {txt}  `{v:.2f}`")
            st.caption(f"Explanation method: {explainer.method}. Values are log-odds contributions "
                       f"relative to an average applicant; they add up to this applicant's score.")
        with right:
            st.pyplot(contribution_chart(contrib))

        log_decision({"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                      "model_version": bundle["version"], "threshold": threshold,
                      **applicant.iloc[0].to_dict(), "pd": round(float(res.pd), 4),
                      "risk_grade": res.risk_grade, "decision": res.decision,
                      "top_reasons": "; ".join(t for t, _ in up[:3])})
        st.success("Decision recorded in the audit trail.")

    if LOG_PATH.exists():
        with st.expander("Audit trail — recent decisions"):
            log = pd.read_csv(LOG_PATH)
            st.dataframe(log.tail(20).iloc[::-1], width="stretch")
            st.download_button("Download full audit log", log.to_csv(index=False), "decision_log.csv")

# ---------------------------------------------------------------------------
# Tab 2 — batch scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.header("Batch scoring")
    st.write("Upload a CSV with one application per row (same columns as the template).")
    raw_cols = [c for c in D.FEATURES if c != "monthly_burden"]
    template = X_test[raw_cols].head(25)
    st.download_button("Download template (25 sample applications)", template.to_csv(index=False),
                       "applications_template.csv")
    up_file = st.file_uploader("Applications CSV", type="csv")
    batch = pd.read_csv(up_file) if up_file is not None else None
    if batch is None and st.checkbox("Use the sample applications instead"):
        batch = template.copy()
    if batch is not None:
        missing = set(raw_cols) - set(batch.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            scored = pd.concat([batch, M.score_applicants(bundle, batch, threshold, refer)], axis=1)
            counts = scored["decision"].value_counts()
            b1, b2, b3 = st.columns(3)
            b1.metric("Approve", int(counts.get("APPROVE", 0)))
            b2.metric("Refer", int(counts.get("REFER TO UNDERWRITER", 0)))
            b3.metric("Decline", int(counts.get("DECLINE", 0)))
            st.dataframe(scored.sort_values("pd"), width="stretch")
            st.download_button("Download decisions", scored.to_csv(index=False), "scored_applications.csv")

# ---------------------------------------------------------------------------
# Tab 3 — portfolio & ROI
# ---------------------------------------------------------------------------
with tab_roi:
    st.header("Portfolio impact & ROI")
    st.caption("Measured on the 200-loan hold-out test set the model never saw during training; "
               "scaled to your yearly application volume.")
    p_test = bundle["pipeline"].predict_proba(X_test)[:, 1]
    n = len(y_test)
    rule_decline = M.rulebook_decline(X_test).astype(float)
    policies = {
        "Approve everyone": np.zeros(n),
        "Manual rulebook": rule_decline,
        "CreditRisk AI": (p_test >= threshold).astype(float),
    }
    rows = []
    for name, dec in policies.items():
        appr = dec < 0.5
        prof = M.expected_profit(y_test, dec, 0.5, econ) / n
        rows.append({"Policy": name, "Approval rate": appr.mean(),
                     "Bad-loan rate in book": y_test[appr].mean() if appr.any() else np.nan,
                     "Profit per application": prof, "Annual profit": prof * econ.annual_applications})
    table = pd.DataFrame(rows)
    ai, rb = table.iloc[2], table.iloc[1]
    r1, r2, r3 = st.columns(3)
    r1.metric("Annual profit with CreditRisk AI", inr(ai["Annual profit"]))
    r2.metric("Uplift vs manual rulebook", inr(ai["Annual profit"] - rb["Annual profit"]))
    r3.metric("Bad-loan rate in book", f"{ai['Bad-loan rate in book']:.1%}",
              f"{(ai['Bad-loan rate in book'] - rb['Bad-loan rate in book']) * 100:+.1f} pp vs rulebook",
              delta_color="inverse")
    show = table.copy()
    show["Approval rate"] = show["Approval rate"].map("{:.0%}".format)
    show["Bad-loan rate in book"] = show["Bad-loan rate in book"].map("{:.1%}".format)
    show["Profit per application"] = show["Profit per application"].map(inr)
    show["Annual profit"] = show["Annual profit"].map(inr)
    st.table(show.set_index("Policy"))

    curve = M.profit_curve(y_oof, p_oof, econ)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(curve.threshold, curve.profit_per_application * econ.annual_applications / 1e7, color="#2563eb", lw=2)
    ax.axvline(threshold, color="#b91c1c", ls="--", label=f"your cut-off {threshold:.2f}")
    ax.axvline(opt_t, color="#15803d", ls=":", label=f"profit-optimal {opt_t:.2f}")
    ax.set_xlabel("Approve if PD below …"); ax.set_ylabel("Annual profit (₹ crore)")
    ax.legend(frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Profit curve uses out-of-fold predictions on the training set, so choosing the "
               "cut-off never touches the test data.")

# ---------------------------------------------------------------------------
# Tab 4 — model card
# ---------------------------------------------------------------------------
with tab_card:
    st.header("Model card")
    t = metrics["test"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
    m2.metric("Gini", f"{t['gini']:.3f}")
    m3.metric("KS", f"{t['ks']:.3f}")
    m4.metric("PR-AUC", f"{t['pr_auc']:.3f}")
    m5.metric("Brier", f"{t['brier']:.3f}")
    st.markdown(f"**Champion:** {metrics['champion']} (chosen by the one-standard-error rule "
                f"from nested 5-fold CV) · **Explainer:** {metrics['explainer']}")
    st.markdown("**Tuned hyperparameters:**")
    st.code(json.dumps(metrics["best_params"], indent=2, default=str), language="json")

    st.subheader("Nested cross-validation (training set, 5 outer folds)")
    cv = pd.DataFrame(metrics["cv_summary"])
    st.dataframe(cv[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "ks_mean", "brier_mean"]],
                 width="stretch")
    fig_dir = M.FIGURES
    g1, g2 = st.columns(2)
    g1.image(str(fig_dir / "cv_comparison.png"))
    g2.image(str(fig_dir / "calibration.png"))
    st.image(str(fig_dir / "roc_pr.png"))
    g3, g4 = st.columns(2)
    g3.image(str(fig_dir / "global_importance.png"))
    g4.image(str(fig_dir / "confusion_matrix.png"))

    st.subheader("Fairness audit")
    st.caption("Sex, marital status and nationality are excluded from the model. Approval rates are "
               "still checked across groups (four-fifths rule: disparate impact < 0.8 is flagged).")
    st.dataframe(pd.read_csv(M.ARTIFACTS / "fairness_audit.csv").round(3), width="stretch")

    st.subheader("Error analysis by segment (test set)")
    st.caption("missed_defaults_rate = share of bad loans the model approved; "
               "wrongly_rejected_rate = share of good loans it declined.")
    st.dataframe(pd.read_csv(M.ARTIFACTS / "error_analysis.csv").round(3), width="stretch")

    st.subheader("Limitations")
    st.markdown(
        "- Trained on 1,000 historical loans (UCI Statlog German Credit); amounts are in Deutsche Mark. "
        "A production deployment would retrain on the lender's own INR portfolio.\n"
        "- Only approved loans have observed outcomes in any real lender's data (reject-inference bias).\n"
        "- Monitor score drift (PSI) monthly and re-validate the model at least once a year.")
