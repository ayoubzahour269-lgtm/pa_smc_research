"""Plan quotidien de bout en bout, sur donnees reelles.

Marque `slow` : le plan reconstruit tout l'historique de candidats pour calibrer sur
les seuls trades clos avant la date demandee.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphalab.config import HOLDOUT_START
from alphalab.report import daily

pytestmark = pytest.mark.slow

DATE = "2022-06-15"


@pytest.fixture(scope="module")
def plan() -> daily.DailyPlan:
    return daily.build(DATE, "H1")


def test_le_plan_est_produit(plan: daily.DailyPlan) -> None:
    assert plan.date == pd.Timestamp(DATE, tz="UTC")
    assert plan.timeframe == "H1"


def test_chaque_ligne_a_une_geometrie_coherente(plan: daily.DailyPlan) -> None:
    for line in plan.lines:
        if line.direction == 1:
            assert line.stop < line.reference_price < line.target
        else:
            assert line.target < line.reference_price < line.stop
        assert 0.0 <= line.probability <= 1.0
        assert line.grade in {"A", "B", "C"}
        assert line.risk_fraction > 0, "un plan emis engage toujours quelque chose"


def test_le_plan_ne_lit_pas_le_futur(plan: daily.DailyPlan) -> None:
    """Aucun signal ne peut etre posterieur a la journee planifiee."""
    limit = pd.Timestamp(DATE, tz="UTC") + pd.Timedelta(days=1)
    for line in plan.lines:
        assert pd.Timestamp(line.signal_ts) < limit


def test_le_plafond_par_symbole_est_respecte(plan: daily.DailyPlan) -> None:
    symbols = [line.symbol for line in plan.lines]
    assert len(symbols) == len(set(symbols))


def test_la_qualite_du_modele_est_annoncee(plan: daily.DailyPlan) -> None:
    """Un modele sans apport doit etre signale, pas masque derriere une probabilite."""
    assert plan.diagnostics
    for diag in plan.diagnostics.values():
        assert "skill" in diag
        assert "n_evaluation" in diag
        assert diag["n_evaluation"] < diag["n_hors_pli"], "l'ECE ne doit pas etre auto-evalue"


def test_les_fenetres_cheres_sont_signalees(plan: daily.DailyPlan) -> None:
    """Une entree a 22h UTC sur l'or paie ~3x le spread median : le plan doit le dire."""
    for line in plan.lines:
        if line.symbol == "XAUUSD" and pd.Timestamp(line.signal_ts).hour == 22:
            assert plan.cost_warnings, "le pic de rollover doit declencher un avertissement"


def test_le_rendu_console_est_lisible(plan: daily.DailyPlan) -> None:
    text = plan.to_console()
    assert "PLAN DU" in text
    assert "Risque de portefeuille" in text


def test_un_plan_dans_le_holdout_est_signale() -> None:
    """Le hold-out ne doit jamais etre consulte sans avertissement explicite."""
    plan = daily.build(HOLDOUT_START + pd.Timedelta(days=200), "H1", symbols=["XAUUSD"])
    assert any("hold-out" in note for note in plan.notes)


def test_comparatif_grade_vs_force(plan: daily.DailyPlan) -> None:
    """Chiffre ce que coute la contrainte 'taille pleine tous les jours'."""
    table = daily.compare_graded_vs_forced([plan])
    if plan.lines:
        assert set(table.columns) == {"date", "grade", "E[R]_gradee", "E[R]_forcee", "ecart"}
        assert len(table) == 1
