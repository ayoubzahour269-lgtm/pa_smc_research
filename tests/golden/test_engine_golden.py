"""Cas de reference du moteur : chaque resultat attendu se calcule a la main.

Ces tests valident le moteur contre la verite arithmetique, et non contre une
implementation anterieure. C'est ce qui autorise a jeter l'ancien moteur sans perdre
la garantie que le nouveau calcule juste.

Convention commune a tous les cas : signal sur la barre 0, entree a l'ouverture de la
barre 1 au prix 100, stop_distance = 10 (donc 1R = 10 points), target_distance = 20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import flat_bars, make_frame, make_market

from alphalab.backtest.engine import (
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    ExecConfig,
    Order,
    SymbolData,
    run,
)

ENTRY = 100.0
RISK = 10.0
TARGET = 20.0
T0 = pd.Timestamp("2020-01-06 00:00", tz="UTC")


def order(direction: int = 1, *, max_hold: int = 5, risk_frac: float = 1.0) -> Order:
    return Order(
        symbol="TEST",
        signal_ts=T0,
        direction=direction,
        stop_distance=RISK,
        target_distance=TARGET,
        max_hold=max_hold,
        risk_frac=risk_frac,
        tag="golden",
    )


def test_entree_a_ouverture_de_la_barre_suivante() -> None:
    """Le prix de la barre du signal ne doit jamais servir d'entree."""
    bars = [(999.0, 999.0, 999.0, 999.0), *flat_bars(4)]
    market = make_market({"TEST": make_frame(bars)})
    res = run([order()], market)
    assert res.n_trades == 1
    trade = res.trades.iloc[0]
    assert trade["entry"] == pytest.approx(ENTRY)
    assert trade["entry_ts"] == T0 + pd.Timedelta(hours=1)


def test_objectif_atteint_donne_R_egal_au_ratio() -> None:
    """Objectif a +20 pour 1R = 10 -> R = +2.0 exactement, cout nul."""
    bars = [*flat_bars(1), (100.0, 121.0, 99.9, 120.0), *flat_bars(3)]
    market = make_market({"TEST": make_frame(bars)})
    res = run([order()], market)
    trade = res.trades.iloc[0]
    assert trade["outcome"] == EXIT_TARGET
    assert trade["exit"] == pytest.approx(120.0)
    assert trade["r"] == pytest.approx(2.0)
    assert not trade["ambiguous"]


def test_stop_atteint_donne_moins_un_R() -> None:
    """Stop a -10 pour 1R = 10 -> R = -1.0 exactement, cout nul."""
    bars = [*flat_bars(1), (100.0, 100.1, 89.0, 90.0), *flat_bars(3)]
    market = make_market({"TEST": make_frame(bars)})
    res = run([order()], market)
    trade = res.trades.iloc[0]
    assert trade["outcome"] == EXIT_STOP
    assert trade["exit"] == pytest.approx(90.0)
    assert trade["r"] == pytest.approx(-1.0)


def test_echeance_a_prix_plat_donne_R_nul() -> None:
    """Ni stop ni objectif, cloture = entree, cout nul -> R = 0."""
    market = make_market({"TEST": make_frame(flat_bars(6))})
    res = run([order(max_hold=3)], market)
    trade = res.trades.iloc[0]
    assert trade["outcome"] == EXIT_TIME
    assert trade["r"] == pytest.approx(0.0)
    assert trade["bars_held"] == 3


def test_le_cout_est_preleve_en_fraction_de_R() -> None:
    """Cout de 1.0 pour 1R = 10 -> R = -0.1 quand le prix ne bouge pas."""
    market = make_market({"TEST": make_frame(flat_bars(6))}, cost=1.0)
    res = run([order(max_hold=3)], market)
    assert res.trades.iloc[0]["r"] == pytest.approx(-0.1)


def test_vente_symetrique_de_l_achat() -> None:
    """A la vente, un objectif a -20 rend +2R ; un stop a +10 rend -1R."""
    down = [*flat_bars(1), (100.0, 100.1, 79.0, 80.0), *flat_bars(3)]
    res_tp = run([order(-1)], make_market({"TEST": make_frame(down)}))
    assert res_tp.trades.iloc[0]["outcome"] == EXIT_TARGET
    assert res_tp.trades.iloc[0]["r"] == pytest.approx(2.0)

    up = [*flat_bars(1), (100.0, 111.0, 99.9, 110.0), *flat_bars(3)]
    res_sl = run([order(-1)], make_market({"TEST": make_frame(up)}))
    assert res_sl.trades.iloc[0]["outcome"] == EXIT_STOP
    assert res_sl.trades.iloc[0]["r"] == pytest.approx(-1.0)


def test_barre_ambigue_retient_le_stop_et_est_signalee() -> None:
    """Une barre touchant stop ET objectif doit compter comme un stop, et etre marquee."""
    both = [*flat_bars(1), (100.0, 125.0, 85.0, 100.0), *flat_bars(3)]
    market = make_market({"TEST": make_frame(both)})
    res = run([order()], market)
    trade = res.trades.iloc[0]
    assert trade["outcome"] == EXIT_STOP
    assert trade["r"] == pytest.approx(-1.0)
    assert bool(trade["ambiguous"])
    assert res.ambiguity_rate == pytest.approx(1.0)


def test_regle_intrabarre_alternative_change_l_issue() -> None:
    """La regle optimiste existe pour borner l'incertitude, pas pour etre utilisee."""
    both = [*flat_bars(1), (100.0, 125.0, 85.0, 100.0), *flat_bars(3)]
    market = make_market({"TEST": make_frame(both)})
    res = run([order()], market, ExecConfig(intrabar="target_first"))
    assert res.trades.iloc[0]["outcome"] == EXIT_TARGET
    assert res.trades.iloc[0]["r"] == pytest.approx(2.0)


def test_max_hold_borne_la_duree() -> None:
    """max_hold = n signifie n barres detenues, entree comprise."""
    market = make_market({"TEST": make_frame(flat_bars(10))})
    res = run([order(max_hold=2)], market)
    trade = res.trades.iloc[0]
    assert trade["bars_held"] == 2
    assert trade["exit_ts"] == T0 + pd.Timedelta(hours=2)


def test_le_stop_prime_meme_si_l_objectif_vient_plus_tard() -> None:
    """Stop touche en barre 1, objectif en barre 2 : c'est le stop qui compte."""
    bars = [*flat_bars(1), (100.0, 100.1, 89.0, 95.0), (95.0, 130.0, 94.0, 129.0), *flat_bars(2)]
    market = make_market({"TEST": make_frame(bars)})
    res = run([order()], market)
    assert res.trades.iloc[0]["outcome"] == EXIT_STOP
    assert res.trades.iloc[0]["bars_held"] == 1


def test_plafond_par_symbole_ignore_le_second_ordre() -> None:
    """Deux signaux chevauchants sur le meme symbole : le second est refuse et compte."""
    market = make_market({"TEST": make_frame(flat_bars(10))})
    orders = [
        order(max_hold=5),
        Order("TEST", T0 + pd.Timedelta(hours=1), 1, RISK, TARGET, 5, tag="golden"),
    ]
    res = run(orders, market, ExecConfig(max_per_symbol=1))
    assert res.n_orders == 2
    assert res.n_trades == 1
    assert res.skipped == {"plafond de positions par symbole": 1}


def test_plafond_global_arbitre_entre_symboles() -> None:
    """max_concurrent=1 : un second symbole chevauchant est refuse."""
    frames = {"A": make_frame(flat_bars(10)), "B": make_frame(flat_bars(10))}
    market = make_market(frames)
    orders = [
        Order("A", T0, 1, RISK, TARGET, 5, tag="g"),
        Order("B", T0 + pd.Timedelta(hours=1), 1, RISK, TARGET, 5, tag="g"),
    ]
    res = run(orders, market, ExecConfig(max_concurrent=1, max_per_symbol=1))
    assert res.n_trades == 1
    assert res.trades.iloc[0]["symbol"] == "A"
    assert res.skipped == {"plafond de positions simultanees": 1}

    res2 = run(orders, market, ExecConfig(max_concurrent=2, max_per_symbol=1))
    assert res2.n_trades == 2
    assert res2.skipped == {}


def test_ponderation_de_risque_affecte_r_weighted_pas_r() -> None:
    """Les paliers A/B/C changent la taille, pas la qualite du signal."""
    bars = [*flat_bars(1), (100.0, 121.0, 99.9, 120.0), *flat_bars(3)]
    market = make_market({"TEST": make_frame(bars)})
    res = run([order(risk_frac=0.25)], market)
    trade = res.trades.iloc[0]
    assert trade["r"] == pytest.approx(2.0)
    assert trade["r_weighted"] == pytest.approx(0.5)


def test_cout_manquant_declenche_le_repli_penalisant() -> None:
    """Un cout NaN doit couter le repli configure, jamais zero."""
    df = make_frame(flat_bars(6))
    market = {"TEST": SymbolData.from_frame("TEST", df, np.full(len(df), np.nan))}
    res = run([order(max_hold=3)], market, ExecConfig(fallback_cost_frac_of_risk=0.05))
    assert res.trades.iloc[0]["cost"] == pytest.approx(0.05 * RISK)
    assert res.trades.iloc[0]["r"] == pytest.approx(-0.05)


def test_signal_sur_la_derniere_barre_est_inexploitable() -> None:
    """Sans barre suivante, il n'y a pas d'entree possible : refus explicite."""
    market = make_market({"TEST": make_frame(flat_bars(3))})
    last = market["TEST"].index[-1]
    res = run([Order("TEST", last, 1, RISK, TARGET, 5)], market)
    assert res.n_trades == 0
    assert res.skipped == {"horodatage inexploitable": 1}


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_geometrie_invalide_est_refusee_a_la_construction(bad: float) -> None:
    """Un stop nul ou non fini rendrait R infini : on refuse en amont."""
    with pytest.raises(ValueError):
        Order("TEST", T0, 1, bad, TARGET, 5)


def test_direction_invalide_est_refusee() -> None:
    with pytest.raises(ValueError):
        Order("TEST", T0, 0, RISK, TARGET, 5)
