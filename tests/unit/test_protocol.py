"""Le gate d'acceptation et le journal des essais."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import flat_bars, make_frame, make_market

from alphalab.backtest import protocol
from alphalab.backtest.engine import Order

FLAT = (100.0, 100.1, 99.9, 100.0)
WINNER = (100.0, 121.0, 99.9, 120.0)  # touche un objectif place a +20


def _winning_market(n_signals: int = 150, n_bars: int = 4000) -> tuple[dict, list[Order]]:
    """Marche truque ou chaque signal precede une barre gagnante.

    Les barres gagnantes sont dispersees au hasard : l'edge n'est donc PAS reductible
    a une heure de la journee, et le temoin apparie ne peut pas le reproduire.
    """
    rng = np.random.default_rng(4242)
    winners = np.sort(rng.choice(np.arange(2, n_bars - 2, 4), size=n_signals, replace=False))
    bars = [WINNER if i in set(winners.tolist()) else FLAT for i in range(n_bars)]
    market = make_market({"A": make_frame(bars)})
    idx = market["A"].index
    orders = [Order("A", idx[int(w) - 1], 1, 10.0, 20.0, 1, tag="truque") for w in winners]
    return market, orders


def _hourly_artifact_market() -> tuple[dict, list[Order]]:
    """Marche ou le gain depend UNIQUEMENT de l'heure de la journee.

    Motif de periode 3 sur une grille horaire : chaque heure UTC tombe toujours sur la
    meme phase du motif. Une "strategie" qui vise ces barres n'a decouvert qu'un
    creneau horaire, pas une information sur le prix.
    """
    n_cycles = 200
    bars = [FLAT, WINNER, FLAT] * n_cycles
    market = make_market({"A": make_frame(bars)})
    idx = market["A"].index
    orders = [
        Order("A", idx[i], 1, 10.0, 20.0, 1, tag="horaire") for i in range(0, len(idx) - 2, 3)
    ]
    return market, orders


def test_gate_rejette_une_strategie_neutre() -> None:
    """Prix plats : esperance nulle ou negative, aucune porte ne doit s'ouvrir."""
    market = make_market({"A": make_frame(flat_bars(600))}, cost=0.5)
    idx = market["A"].index
    orders = [Order("A", idx[i], 1, 10.0, 20.0, 3) for i in range(0, 500, 4)]
    gate, _ = protocol.evaluate("plat", orders, market, n_boot=500, n_control=5)
    assert not gate.passed
    assert "R_positif" in gate.failed_checks


def test_gate_rejette_un_echantillon_trop_petit() -> None:
    market, orders = _winning_market(n_signals=5)
    gate, _ = protocol.evaluate("mini", orders, market, n_boot=200, n_control=3)
    assert not gate.passed
    assert "echantillon_suffisant" in gate.failed_checks


def test_gate_accepte_un_signal_truque_evident() -> None:
    """Controle de sensibilite : un edge enorme et reel DOIT franchir toutes les portes.

    Sans ce test, un gate casse qui rejette tout passerait pour de la rigueur.
    """
    market, orders = _winning_market()
    gate, res = protocol.evaluate("truque", orders, market, n_boot=500, n_control=10)
    assert res.n_trades >= 140
    assert gate.passed, gate.failed_checks
    assert gate.details["stats"]["R_moyen"] == pytest.approx(2.0)


def test_le_temoin_horaire_demasque_un_artefact_de_creneau() -> None:
    """Un "edge" qui n'est qu'une heure de la journee doit etre rejete.

    C'est la raison d'etre de l'appariement horaire du temoin : l'esperance-R est
    excellente et l'IC exclut zero, mais le temoin apparie fait aussi bien, donc la
    strategie n'apporte aucune information. Sans cet appariement, elle serait publiee
    comme une decouverte.
    """
    market, orders = _hourly_artifact_market()
    gate, res = protocol.evaluate("horaire", orders, market, n_boot=500, n_control=10)
    assert res.n_trades >= 150
    assert gate.checks["R_positif"] and gate.checks["IC_exclut_0"]
    assert not gate.checks["bat_hasard"]
    assert not gate.passed


def test_majoration_du_cout_reduit_l_esperance() -> None:
    market = make_market({"A": make_frame(flat_bars(100))}, cost=1.0)
    stressed = protocol.with_cost_multiplier(market, 2.0)
    assert np.nanmean(stressed["A"].cost) == pytest.approx(2.0)
    assert np.nanmean(market["A"].cost) == pytest.approx(1.0)


def test_journal_des_essais_est_append_only(tmp_path: Path) -> None:
    """Le nombre d'essais est le `m` de la correction pour tests multiples."""
    journal = tmp_path / "journal.jsonl"
    for k in range(3):
        protocol.record_trial(
            f"essai-{k}",
            family="test",
            params={"seuil": k},
            universe=["A"],
            timeframe="H1",
            window="in-sample",
            path=journal,
        )
    entries = protocol.read_trials(journal)
    assert len(entries) == 3
    assert protocol.n_trials(journal) == 3
    assert entries[1]["params"] == {"seuil": 1}
    assert entries[0]["famille"] == "test"


def test_journal_absent_donne_zero(tmp_path: Path) -> None:
    assert protocol.n_trials(tmp_path / "rien.jsonl") == 0
