"""Unit tests for the statistics module against known values and simulated data."""
import numpy as np
import pytest
import pandas as pd
from scipy import stats as sps

from vlamech.metrics.stats import (
    auroc,
    hierarchical_bootstrap,
    holm,
    mcnemar_exact,
    paired_sign_flip_test,
    task_group_permutation_test,
    wilson_ci,
)


def test_task_group_test_is_exact_over_all_relabellings():
    v = {t: float(t) for t in range(10)}
    r = task_group_permutation_test(v, [0, 1, 2])  # the three lowest tasks: the most extreme of all 120 choices
    assert r["n_relabellings"] == 120 and r["p"] == pytest.approx(1 / 120)
    assert r["T"] == pytest.approx(1.0 - 6.0)
    assert task_group_permutation_test(v, [7, 8, 9])["p"] == pytest.approx(1.0)  # one-sided: highest is not lower
    with pytest.raises(ValueError):
        task_group_permutation_test(v, [3, 11])


def test_wilson_matches_statsmodels_reference():
    lo, hi = wilson_ci(97, 100)
    assert abs(lo - 0.9154) < 1e-3 and abs(hi - 0.9897) < 1e-3


def test_auroc_matches_sklearn_formula():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    y = (s + rng.normal(size=200) > 0).astype(int)
    from sklearn.metrics import roc_auc_score

    assert abs(auroc(s, y) - roc_auc_score(y, s)) < 1e-12
    assert np.isnan(auroc(s, np.zeros_like(y)))


def test_holm():
    p = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(p, [0.03, 0.06, 0.06, 0.02])


def test_mcnemar():
    a = np.array([1] * 40 + [0] * 10, bool)
    b = np.array([1] * 30 + [0] * 10 + [0] * 10, bool)
    r = mcnemar_exact(a, b)
    assert r["n10"] == 10 and r["n01"] == 0
    assert abs(r["p"] - sps.binomtest(0, 10, 0.5).pvalue) < 1e-12
    assert abs(r["risk_diff"] + 0.2) < 1e-12


def test_sign_flip_null_and_alt():
    rng = np.random.default_rng(1)
    assert paired_sign_flip_test(rng.normal(0, 1, 50), seed=1) > 0.01
    assert paired_sign_flip_test(rng.normal(1, 1, 50), seed=1) < 0.001


def _clustered_frame(seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(10):
        task_eff = rng.normal(0, 1.0)
        for e in range(20):
            ep_eff = rng.normal(0, 0.5)
            for s in range(5):
                rows.append({"task": t, "init_state": e, "y": task_eff + ep_eff + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def test_bootstrap_task_resampling_widens_ci():
    """Sensitivity mode (tasks random) must be much wider than stratified mode when task effects dominate,
    and both must be wider than a naive row bootstrap that ignores clustering."""
    df = _clustered_frame()
    mean = lambda d: d["y"].mean()
    rand = hierarchical_bootstrap(df, mean, levels=("task", "init_state"), strata=(), n_boot=2000)
    strat = hierarchical_bootstrap(df, mean, levels=("init_state",), strata=("task",), n_boot=2000)
    naive = hierarchical_bootstrap(df.assign(row=np.arange(len(df))), mean, levels=("row",), strata=(), n_boot=2000)
    w = lambda r: r["ci_high"] - r["ci_low"]
    assert w(rand) > 3 * w(strat)
    assert w(strat) > w(naive)
    for r in (rand, strat):
        assert r["ci_low"] < df["y"].mean() < r["ci_high"]


def test_bootstrap_strata_fixed_and_pairs_kept_together():
    """Every task keeps its row count in every replicate; rows sharing (task, init_state) are drawn together."""
    df = _clustered_frame().assign(cond=lambda d: np.where(np.arange(len(d)) % 2 == 0, "a", "b"))
    seen = []

    def stat(d):
        seen.append((d.groupby("task").size().to_dict(),
                     d.groupby(["task", "init_state"]).size().unique().tolist()))
        return d["y"].mean()

    hierarchical_bootstrap(df, stat, n_boot=50)
    for per_task, cluster_sizes in seen:
        assert per_task == {t: 100 for t in range(10)}
        assert all(c % 5 == 0 for c in cluster_sizes)  # whole init-state clusters only


def test_bootstrap_refuses_undefined_statistic():
    df = _clustered_frame()
    with pytest.raises(FloatingPointError):
        hierarchical_bootstrap(df, lambda d: np.nan, n_boot=10)
    with pytest.raises(KeyError):
        hierarchical_bootstrap(df, lambda d: 0.0, levels=("episode",), n_boot=10)


def test_invalid_inputs_raise_instead_of_fabricating_results():
    """NaN / non-binary / out-of-range inputs must never become plausible statistics."""
    with pytest.raises(ValueError):
        holm([0.01, np.nan])
    with pytest.raises(ValueError):
        holm([0.01, 1.2])
    with pytest.raises(ValueError):
        mcnemar_exact([1, np.nan], [1, 0])
    with pytest.raises(ValueError):
        mcnemar_exact([1, 2], [1, 0])
    with pytest.raises(ValueError):
        paired_sign_flip_test([1.0, np.nan])
    with pytest.raises(ValueError):
        auroc([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError):
        wilson_ci(3, 0)
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))  # one class: explicitly undefined
