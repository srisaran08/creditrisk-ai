# CreditRisk AI — Real-time Loan Underwriting Engine

Capstone project · Data Science for Managers · Track 2: Decision Automation

CreditRisk AI predicts a loan applicant's **probability of default (PD)**. It turns that score into an
**Approve / Refer / Decline** decision at the cut-off that maximises the lender's expected profit.
It gives every decision **plain-English reason codes** and writes it to an **audit trail**, so the
decision can be explained to a compliance team.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.model        # trains, validates and saves to artifacts/ (about 5 minutes)
streamlit run app.py       # opens the web app at http://localhost:8501
pytest -q                  # sanity tests
```

## Repository layout

```
├── app.py                 Streamlit app: score, batch, ROI, model card
├── src/
│   ├── data.py            loading, schema, feature engineering, preprocessing, split
│   └── model.py           nested CV, tuning, threshold optimisation, SHAP, fairness, artifacts
├── data/german_credit.csv UCI Statlog German Credit (1,000 loans)
├── artifacts/             model.joblib, metrics.json, CV results, figures (created by training)
├── tests/test_pipeline.py data, profit-maths and explanation-additivity tests
├── docs/VIBE_CODING_LOG.md
└── requirements.txt
```

## Pipeline

| Step | What happens | Why |
|---|---|---|
| Data | 1,000 loans, 30% defaulted. Stratified 80/20 split; the test set is used once, at the end | Honest final estimate |
| Compliance | `personal_status_sex` and `foreign_worker` are **never** model inputs, but they are kept for the fairness audit | Fair-lending practice |
| Features | 7 raw numeric fields + engineered `monthly_burden = amount / duration`; 11 categorical fields one-hot encoded | Repayment pressure is a known risk driver |
| Preprocessing | `log1p` on skewed money fields → `StandardScaler`; `OneHotEncoder(handle_unknown="ignore")`. Lives *inside* the Pipeline | Re-fitted on every CV fold, so there is no leakage |
| Models | Elastic-net Logistic Regression · Random Forest · XGBoost | Interpretable baseline vs. bagging vs. boosting |
| Tuning | `RandomizedSearchCV`, 5-fold stratified, scored on ROC-AUC | Efficient search of continuous spaces |
| Validation | **Nested CV**: inner 5-fold tunes, outer 5-fold estimates | Tuning on the same folds you report would inflate the score |
| Selection | **One-standard-error rule**: pick the simplest model within 1 SE of the best AUC | Parsimony; easiest to defend to a regulator |
| Decision | Cut-off chosen to maximise expected ₹ profit on out-of-fold predictions | Lenders optimise profit, not accuracy |
| Explainability | Exact linear SHAP (LR) / TreeSHAP (trees), one-hot values summed back to business features | Additive reason codes for each applicant |
| Audit | Error analysis by segment, disparate-impact (four-fifths rule) check, decision log | Model risk management |

Class imbalance (30% bad) is handled by the **profit-optimal threshold**, not by class weights, so
the PDs stay calibrated (Brier 0.160).

## Results (run in this repo: Logistic Regression champion)

| Nested CV (train, 5 outer folds) | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic Regression | 0.773 ± 0.066 | 0.597 | 0.171 |
| Random Forest | 0.792 ± 0.054 | 0.635 | 0.166 |
| Gradient Boosting* | 0.770 ± 0.052 | 0.583 | 0.172 |

Random Forest's AUC was 0.019 higher, which is less than one standard error (0.024), so the
simpler Logistic Regression wins.
**Test set:** AUC 0.791 · Gini 0.582 · KS 0.560 · Brier 0.160.

**Business impact.** Assumptions: average ticket ₹5 L, 10% lifetime margin, 50% LGD, 10,000 applications a year.

| Policy | Approval rate | Bad-loan rate in book | Profit / application |
|---|---|---|---|
| Approve everyone | 100% | 30.0% | −₹40,000 |
| Manual rulebook | 58% | 15.5% | ₹2,000 |
| **CreditRisk AI (PD < 0.21)** | 40% | **11.2%** | **₹6,500** |

That is **≈ ₹4.5 crore a year more profit than the rulebook**. The model catches 85% of defaults.

\*The run above used scikit-learn's HistGradientBoosting because XGBoost could not be installed in the
build sandbox. Running `python -m src.model` with `requirements.txt` installed swaps in XGBoost and
TreeSHAP automatically. Expect small changes in the numbers.

## Data & limitations

* Source: UCI Statlog (German Credit Data), Hofmann (1994), using the descriptive labels of the R
  distribution. Amounts are in Deutsche Mark. Rupee figures come from the economics assumptions, not
  from the data.
* 1,000 rows means wide CV error bars (±0.05–0.07 AUC per fold). This is why we report nested CV and not one split.
* Reject-inference bias: real lenders only observe outcomes for loans they approved.
* The fairness audit flags applicants aged ≤25 (disparate impact 0.46). Age is a legitimate but
  sensitive feature, so these applicants should go to manual review or the age feature should be
  constrained before deployment.
