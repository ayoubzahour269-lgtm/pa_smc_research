"""Paliers de conviction et risque de portefeuille ajuste de la correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.config import GRADE_RISK
from alphalab.risk import portfolio, sizing

# --------------------------------------------------------------------------------------
# Paliers
# --------------------------------------------------------------------------------------


def test_esperance_negative_donne_un_grade_C() -> None:
    verdict = sizing.grade_candidate(0.20, 2.0, percentile=1.0)
    assert verdict.grade == "C"
    assert verdict.expected_r < 0
    assert verdict.is_paper
    assert verdict.risk_fraction == GRADE_RISK["C"]


def test_esperance_forte_et_haut_de_classement_donne_un_grade_A() -> None:
    verdict = sizing.grade_candidate(0.50, 2.0, percentile=1.0)
    assert verdict.grade == "A"
    assert verdict.risk_fraction == GRADE_RISK["A"]


def test_esperance_forte_mais_bas_de_classement_reste_en_B() -> None:
    """Le grade A exige aussi d'etre la meilleure occasion du jour.

    Sans cette condition, plusieurs candidats mediocres du meme jour recevraient tous
    le risque maximal.
    """
    verdict = sizing.grade_candidate(0.50, 2.0, percentile=0.2)
    assert verdict.grade == "B"


def test_esperance_faiblement_positive_donne_un_grade_B() -> None:
    verdict = sizing.grade_candidate(0.35, 2.0, percentile=1.0)
    assert verdict.grade == "B"
    assert 0 <= verdict.expected_r < 0.15


def test_le_risque_decroit_avec_le_grade() -> None:
    assert GRADE_RISK["A"] > GRADE_RISK["B"] > GRADE_RISK["C"] > 0


def test_un_plan_existe_meme_quand_tout_est_mauvais() -> None:
    """L'exigence 'une position chaque jour' n'autorise pas une taille nulle par defaut."""
    verdict = sizing.grade_candidate(0.05, 2.0, percentile=1.0)
    assert verdict.grade == "C"
    assert verdict.risk_fraction > 0


def test_percentiles_classent_de_zero_a_un() -> None:
    ranks = sizing.percentiles(np.array([0.1, 0.5, 0.3]))
    assert ranks[0] == pytest.approx(0.0)
    assert ranks[1] == pytest.approx(1.0)
    assert ranks[2] == pytest.approx(0.5)
    assert sizing.percentiles(np.array([2.0]))[0] == pytest.approx(1.0)


def test_taille_de_position_part_du_risque_accepte() -> None:
    """On part du montant qu'on accepte de perdre. Jamais l'inverse."""
    assert sizing.position_size(10_000, 0.01, stop_distance=20.0) == pytest.approx(5.0)
    assert sizing.position_size(10_000, 0.01, stop_distance=40.0) == pytest.approx(2.5)
    assert sizing.position_size(10_000, 0.01, stop_distance=0.0) == 0.0


# --------------------------------------------------------------------------------------
# Portefeuille et correlation
# --------------------------------------------------------------------------------------


def _correlations(rho: float, a: str = "XAUUSD", b: str = "EURUSD") -> pd.DataFrame:
    return pd.DataFrame([[1.0, rho], [rho, 1.0]], index=[a, b], columns=[a, b])


def test_deux_paris_identiques_ne_diversifient_pas() -> None:
    """Le cas du cahier des charges : long or + short EUR/USD contre le dollar.

    Les deux positions sont le meme pari. La somme naive donnerait 2 % ; la mesure
    ajustee de la correlation doit s'en approcher, pas donner la racine de la somme
    des carres.
    """
    expositions = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("EURUSD", 1, 0.01),
    ]
    risque = portfolio.portfolio_risk(expositions, _correlations(0.95))
    assert risque == pytest.approx(0.0198, abs=0.001)
    # Une addition naive ferait croire a 2 % ; une hypothese d'independance a 1,41 %.
    assert risque > np.sqrt(0.01**2 + 0.01**2)


def test_deux_paris_independants_diversifient() -> None:
    expositions = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("EURUSD", 1, 0.01),
    ]
    risque = portfolio.portfolio_risk(expositions, _correlations(0.0))
    assert risque == pytest.approx(0.01414, abs=0.001)


def test_des_sens_opposes_sur_actifs_correles_se_compensent() -> None:
    expositions = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("EURUSD", -1, 0.01),
    ]
    assert portfolio.portfolio_risk(expositions, _correlations(0.95)) < 0.005


def test_paris_redondants_detectes() -> None:
    expositions = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("EURUSD", 1, 0.01),
    ]
    pairs = portfolio.duplicate_pairs(expositions, _correlations(0.9))
    assert pairs and pairs[0][2] == pytest.approx(0.9)

    # Sens opposes sur actifs ANTI-correles : egalement redondant.
    inverse = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("EURUSD", -1, 0.01),
    ]
    assert portfolio.duplicate_pairs(inverse, _correlations(-0.9))


def test_le_plafond_de_risque_ecarte_les_paris_redondants() -> None:
    candidats = [portfolio.Exposure(f"S{i}", 1, 0.01) for i in range(4)]
    correlations = pd.DataFrame(
        np.ones((4, 4)) * 0.95 + np.eye(4) * 0.05,
        index=[f"S{i}" for i in range(4)],
        columns=[f"S{i}" for i in range(4)],
    )
    check = portfolio.apply_caps(candidats, correlations, max_concurrent=4, max_risk=0.02)
    assert len(check.accepted) < 4
    assert any("correlation" in reason for _, reason in check.rejected)


def test_le_plafond_par_symbole_est_applique() -> None:
    candidats = [
        portfolio.Exposure("XAUUSD", 1, 0.01),
        portfolio.Exposure("XAUUSD", -1, 0.01),
    ]
    check = portfolio.apply_caps(candidats, pd.DataFrame(), max_per_symbol=1)
    assert len(check.accepted) == 1
    assert check.rejected[0][1] == "plafond par symbole"


def test_matrice_de_correlation_sur_les_rendements() -> None:
    index = pd.date_range("2021-01-01", periods=300, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=300), index=index)
    correlations = portfolio.correlation_matrix({"A": base, "B": base * 1.0, "C": -base})
    assert correlations.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)
    assert correlations.loc["A", "C"] == pytest.approx(-1.0, abs=1e-6)


def test_limite_de_perte_quotidienne() -> None:
    """Ce qui empeche une mauvaise serie de devenir un mauvais mois."""
    assert portfolio.daily_loss_breached(-2.0, limit=2.0)
    assert portfolio.daily_loss_breached(-3.0, limit=2.0)
    assert not portfolio.daily_loss_breached(-1.5, limit=2.0)
