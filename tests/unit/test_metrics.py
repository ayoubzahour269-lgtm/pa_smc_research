"""Metriques et temoin aleatoire apparie."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import flat_bars, make_frame, make_market

from alphalab.backtest.engine import Order, run
from alphalab.backtest.metrics import (
    bootstrap_ci,
    describe,
    equity_curve,
    matched_random_orders,
    max_drawdown,
)

T0 = pd.Timestamp("2020-01-06 00:00", tz="UTC")


def test_ic_bootstrap_encadre_la_moyenne() -> None:
    r = np.random.default_rng(0).normal(0.2, 1.0, 2000)
    lo, hi = bootstrap_ci(r, n_boot=2000, seed=7)
    assert lo < r.mean() < hi


def test_ic_bootstrap_est_reproductible() -> None:
    """Une mesure publiee doit se rejouer a l'identique."""
    r = np.random.default_rng(1).normal(0.0, 1.0, 500)
    assert bootstrap_ci(r, n_boot=1000, seed=3) == bootstrap_ci(r, n_boot=1000, seed=3)


def test_ic_bootstrap_exclut_zero_sur_signal_franc() -> None:
    r = np.full(500, 0.5)
    lo, hi = bootstrap_ci(r, n_boot=1000, seed=0)
    assert lo > 0 and hi > 0


def test_ic_bootstrap_sur_echantillon_vide() -> None:
    lo, hi = bootstrap_ci(np.empty(0))
    assert np.isnan(lo) and np.isnan(hi)


def test_courbe_d_equite_et_drawdown() -> None:
    trades = pd.DataFrame(
        {
            "exit_ts": pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC"),
            "r_weighted": [1.0, -2.0, 0.5, 1.0],
        }
    )
    eq = equity_curve(trades)
    assert eq.tolist() == [1.0, -1.0, -0.5, 0.5]
    assert max_drawdown(eq) == pytest.approx(-2.0)


def test_drawdown_nul_si_croissance_monotone() -> None:
    eq = pd.Series([0.5, 1.0, 2.0], index=pd.date_range("2021-01-01", periods=3, tz="UTC"))
    assert max_drawdown(eq) == pytest.approx(0.0)


def test_temoin_apparie_conserve_symbole_sens_heure_et_geometrie() -> None:
    """Le temoin ne doit differer de la strategie que par le CHOIX DU MOMENT."""
    frames = {"A": make_frame(flat_bars(200), freq="h")}
    market = make_market(frames)
    reference = [
        Order("A", market["A"].index[i], 1 if i % 2 else -1, 10.0, 20.0, 5, tag="reel")
        for i in range(3, 100, 7)
    ]
    rng = np.random.default_rng(0)
    control = matched_random_orders(reference, market, rng)

    assert len(control) == len(reference)
    for real, fake in zip(reference, control, strict=True):
        assert fake.symbol == real.symbol
        assert fake.direction == real.direction
        assert fake.stop_distance == real.stop_distance
        assert fake.target_distance == real.target_distance
        assert fake.max_hold == real.max_hold
        assert pd.Timestamp(fake.signal_ts).hour == pd.Timestamp(real.signal_ts).hour


def test_temoin_apparie_change_bien_de_dates() -> None:
    frames = {"A": make_frame(flat_bars(500), freq="h")}
    market = make_market(frames)
    reference = [Order("A", market["A"].index[i], 1, 10.0, 20.0, 3) for i in range(0, 200, 5)]
    rng = np.random.default_rng(42)
    control = matched_random_orders(reference, market, rng)
    pairs = zip(reference, control, strict=True)
    identiques = sum(1 for a, b in pairs if a.signal_ts == b.signal_ts)
    assert identiques < len(reference) // 2


def test_temoin_respecte_une_geometrie_fournie() -> None:
    market = make_market({"A": make_frame(flat_bars(100), freq="h")})
    reference = [Order("A", market["A"].index[5], 1, 10.0, 20.0, 3)]
    control = matched_random_orders(
        reference, market, np.random.default_rng(0), geometry=lambda _s, _i: (2.0, 8.0)
    )
    assert control[0].stop_distance == 2.0
    assert control[0].target_distance == 8.0


def test_description_d_une_execution() -> None:
    bars = [*flat_bars(1), (100.0, 121.0, 99.9, 120.0), *flat_bars(8)]
    market = make_market({"A": make_frame(bars)})
    res = run([Order("A", market["A"].index[0], 1, 10.0, 20.0, 5)], market)
    stats = describe(res, n_boot=500, seed=0)
    assert stats["trades"] == 1
    assert stats["R_moyen"] == pytest.approx(2.0)
    assert stats["taux_reussite"] == pytest.approx(1.0)


def test_description_sur_execution_vide() -> None:
    market = make_market({"A": make_frame(flat_bars(5))})
    res = run([], market)
    stats = describe(res)
    assert stats["trades"] == 0
    assert not stats["exclut_0"]
