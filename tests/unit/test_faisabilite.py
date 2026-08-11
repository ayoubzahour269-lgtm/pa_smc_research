"""Porte de faisabilite : les grandeurs doivent etre calculables a la main."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.backtest import feasibility
from alphalab.config import SCALPING_MAX_COST_FRAC_OF_MFE


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def test_excursion_a_une_barre_est_la_barre_suivante() -> None:
    # Entree a l'ouverture de la barre 1 (= 10). Elle monte a 12, descend a 9.
    # Achat : +2. Vente : +1. Moyenne des deux sens : 1,5.
    df = _frame([(5, 5, 5, 5), (10, 12, 9, 11), (11, 11, 11, 11)])
    mfe = feasibility.forward_excursion(df, 1)
    assert mfe.iloc[0] == pytest.approx(1.5)


def test_excursion_agrege_bien_plusieurs_barres() -> None:
    # Entree a 10 (barre 1). Sur les barres 1 et 2 : plus haut 14, plus bas 8.
    # Achat +4, vente +2, moyenne 3.
    df = _frame([(5, 5, 5, 5), (10, 12, 9, 11), (11, 14, 8, 9), (9, 9, 9, 9)])
    assert feasibility.forward_excursion(df, 2).iloc[0] == pytest.approx(3.0)


def test_excursion_ne_regarde_jamais_la_barre_du_signal() -> None:
    # Un pic enorme SUR la barre du signal ne doit rien changer : l'entree a lieu apres.
    calme = _frame([(5, 5, 5, 5), (10, 12, 9, 11), (11, 11, 11, 11)])
    pic = _frame([(5, 500, 0, 5), (10, 12, 9, 11), (11, 11, 11, 11)])
    assert (
        feasibility.forward_excursion(calme, 1).iloc[0]
        == feasibility.forward_excursion(pic, 1).iloc[0]
    )


def test_excursion_incomplete_est_nan() -> None:
    # Faute de barres futures suffisantes, la mesure n'existe pas : elle ne doit pas
    # etre silencieusement tronquee a ce qui est disponible.
    df = _frame([(5, 5, 5, 5), (10, 12, 9, 11)])
    assert np.isnan(feasibility.forward_excursion(df, 2).iloc[0])


def test_horizon_nul_est_refuse() -> None:
    with pytest.raises(ValueError):
        feasibility.forward_excursion(_frame([(1, 1, 1, 1)]), 0)


def _report(timeframe: str, horizons: dict[int, float]) -> feasibility.FeasibilityReport:
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    return feasibility.FeasibilityReport(
        symbol="TEST",
        timeframe=timeframe,
        n_bars=1000,
        start=ts,
        end=ts,
        spread_median=1.0,
        true_range_median=4.0,
        horizons=tuple(
            feasibility.HorizonMeasure(h, 1.0 / f, f, 0.5 + f / 2.0) for h, f in horizons.items()
        ),
    )


def test_seuil_de_viabilite_est_celui_du_preenregistrement() -> None:
    juste_dessous = feasibility.HorizonMeasure(1, 1.0, SCALPING_MAX_COST_FRAC_OF_MFE, 0.0)
    juste_dessus = feasibility.HorizonMeasure(1, 1.0, SCALPING_MAX_COST_FRAC_OF_MFE + 1e-9, 0.0)
    assert juste_dessous.viable
    assert not juste_dessus.viable


def test_taux_de_reussite_minimal_suit_la_formule() -> None:
    # p*x - (1-p)*x - c = 0  =>  p = 0.5 + c/(2x). Un cout valant 25 % de l'objectif
    # impose donc 62,5 % de reussite pour ne RIEN gagner.
    report = _report("H1", {1: 0.25})
    assert report.horizons[0].breakeven_winrate == pytest.approx(0.625)


def test_loi_de_duree_est_exactement_en_racine_du_temps() -> None:
    # k = 2 : fraction(60 min) = 2/sqrt(60), fraction(240 min) = 2/sqrt(240).
    k = 2.0
    reports = {
        "H1": _report("H1", {1: k / np.sqrt(60), 4: k / np.sqrt(240)}),
        "M15": _report("M15", {4: k / np.sqrt(60), 16: k / np.sqrt(240)}),
    }
    assert feasibility.duration_law(reports) == pytest.approx(k)
    # T* = (k / seuil)^2 = (2 / 0.25)^2 = 64 minutes.
    assert feasibility.min_viable_duration(reports) == pytest.approx(64.0)


def test_les_grilles_sont_confrontees_a_duree_egale() -> None:
    reports = {
        "H1": _report("H1", {1: 0.20}),
        "M15": _report("M15", {4: 0.22}),  # meme duree : 4 x 15 min = 60 min
    }
    table = feasibility.grid_agreement(reports)
    assert list(table["duree_min"]) == [60]
    assert table["ecart_relatif"].iloc[0] == pytest.approx(0.10)


def test_la_table_de_duree_distingue_mesure_et_extrapolation() -> None:
    reports = {"H1": _report("H1", {1: 0.20, 4: 0.10})}
    table = feasibility.duration_table(reports, durations=(1, 60, 240))
    nature = dict(zip(table["duree_min"], table["nature"], strict=True))
    assert nature == {1: "extrapolation", 60: "mesure", 240: "mesure"}
    # La minute extrapolee doit couter bien plus cher que l'heure mesuree.
    fractions = dict(zip(table["duree_min"], table["fraction_cout"], strict=True))
    assert fractions[1] > fractions[60] > fractions[240]


def test_loi_de_duree_sans_point_mesure_echoue_franchement() -> None:
    with pytest.raises(ValueError):
        feasibility.duration_law({})
