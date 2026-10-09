# Vibe Coding Log — CreditRisk AI

**Platforms:** Claude Cowork (agentic build in a cloud sandbox) · VS Code + GitHub Copilot (local runs and edits) · GitHub (version control)

## 1. Agentic workflow

| Phase | What the agent (Claude Cowork) did | Human decision / review |
|---|---|---|
| Brief | Read the full assignment spec and rubric; proposed 4 tracks | Team chose Track 2 (Credit Risk) for finance-domain fit and a public dataset |
| Data | Found and profiled the UCI German Credit CSV (1,000 × 21) | Accept the exclusion of sex/marital status and nationality from model inputs |
| Pipeline | Generated `src/data.py` (schema, leakage-safe preprocessing) and `src/model.py` (nested CV, tuning, threshold, SHAP, fairness) | Review: nested CV, the 1-SE selection rule and the profit-based cut-off |
| Debugging | Fixed the scikit-learn 1.8 `penalty` deprecation (moved to `l1_ratio` + `saga`) and Streamlit's `width` API change | Confirm the code runs on our laptops |
| Constraint | The sandbox blocked PyPI, so the agent built automatic fallbacks (sklearn boosting, in-house SHAP) | *(to do)* Re-run locally with real XGBoost and SHAP |
| App | Generated the 4-tab Streamlit app; smoke-tested every tab with a stub `streamlit` module | *(to do)* Usability test with a non-technical classmate |
| Verification | Wrote tests: schema, stratified split, profit maths, **SHAP additivity** | *(to do)* Re-check deck numbers against `metrics.json` after the local run |

## 2. Prompt and agent strategies used

1. **Give the full brief and rubric up front.** The agent saw the grading criteria first, so it built for the graded
   items (CV, tuning, SHAP, ROI, usable app) instead of a generic demo.
2. **Clarify before building.** The agent asked three multiple-choice questions (track, deliverables, dataset)
   before writing code, which avoided rework.
3. **Constraints before code.** Preprocessing lives inside the sklearn `Pipeline`, so every CV fold re-fits the scaler and encoder.
   This rules out data leakage by design.
4. **"Why" lives in docstrings.** Every design choice (no class weights, 1-SE rule, log1p, protected-attribute
   exclusion) is explained in code comments. The comments double as Q&A preparation notes.
5. **Tests that prove properties.** The SHAP contributions must sum to the model's log-odds, and protected attributes must
   be absent from the features. These tests check correctness, not just that the code runs.
6. **Business-number sanity check.** The first ROI figure compared against "approve everyone" (₹46.5 Cr a year), which is a straw man.
   We added a realistic manual-rulebook baseline, which gives a defensible ₹4.5 Cr a year.

## 3. Design decisions the team must be able to defend

- No `class_weight="balanced"`, because it distorts PDs. The profit-optimal threshold handles the 30% imbalance instead.
- Random Forest had the top CV AUC (0.792). Logistic Regression won under the 1-SE rule (gap 0.019 < SE 0.024).
- The fairness audit flags applicants aged ≤25 (disparate impact 0.46). Our proposal: send them to manual review.

## 4. Team prompt log (add your own entries)

| Date | Member | Tool | Prompt (short) | Outcome / what we changed |
|---|---|---|---|---|
| 09-Oct-2026 | Dhanush | Claude Cowork | Pasted capstone brief; chose Credit Risk + public data + repo/deck/log | Repo v1, deck v1 |
| | | | | |
| | | | | |
