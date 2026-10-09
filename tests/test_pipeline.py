"""Fast sanity tests: `pytest -q` (uses the trained bundle if present)."""
import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import model as M


@pytest.fixture(scope="module")
def df():
    return D.load_raw()


def test_data_schema(df):
    assert len(df) == 1000
    assert df[D.TARGET].mean() == pytest.approx(0.30)
    for col in D.EXCLUDED_PROTECTED:
        assert col not in D.FEATURES, "protected attribute leaked into model features"


def test_split_is_stratified(df):
    X_tr, X_te, y_tr, y_te = D.split(df)
    assert len(X_te) == 200
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_feature_groups_cover_every_column(df):
    prep = D.build_preprocessor().fit(df[D.FEATURES])
    groups = D.feature_groups(prep)
    used = sorted(i for idx in groups.values() for i in idx)
    assert used == list(range(len(prep.get_feature_names_out())))
    assert all(groups[f] for f in D.FEATURES)


def test_breakeven_matches_cost_matrix():
    # 10% margin vs 50% LGD -> 1:5 cost ratio -> break-even PD = 1/6
    assert M.Economics().breakeven_pd == pytest.approx(1 / 6)


def test_profit_function():
    y = np.array([0, 0, 1])
    p = np.array([0.1, 0.2, 0.9])
    e = M.Economics()
    assert M.expected_profit(y, p, 0.5, e) == 2 * e.gain_good          # bad loan declined
    assert M.expected_profit(y, p, 1.01, e) == 2 * e.gain_good - e.loss_bad


@pytest.mark.skipif(not M.MODEL_PATH.exists(), reason="run `python -m src.model` first")
def test_bundle_scores_and_explains():
    bundle = M.load_bundle()
    X_te, _ = bundle["test_set"]
    raw = X_te[[c for c in D.FEATURES if c != "monthly_burden"]].head(5)
    out = M.score_applicants(bundle, raw)
    assert out["pd"].between(0, 1).all()
    assert set(out["decision"]) <= {"APPROVE", "DECLINE", "REFER TO UNDERWRITER"}
    expl = M.RiskExplainer(bundle["pipeline"], bundle["background"])
    contrib = expl.explain(D.engineer_features(raw))
    logit = np.log(out["pd"] / (1 - out["pd"])).to_numpy()
    # additivity: base value + contributions == model log-odds
    assert np.allclose(expl.base_value + contrib.sum(axis=1).to_numpy(), logit, atol=1e-3)
