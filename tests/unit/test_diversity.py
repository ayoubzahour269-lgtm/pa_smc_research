"""Mesure de diversite du catalogue.

Ce module protege contre une erreur subtile : croire qu'on a explore large parce qu'on
a beaucoup de noms de familles. Ces tests verifient qu'il distingue bien la vraie
redondance (memes moments, meme sens) de la vraie diversite (memes moments, sens
opposes — qui est la diversite maximale, pas la redondance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.alpha import diversity


def _signals(**columns: list[int]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(columns, index=index)


# --------------------------------------------------------------------------------------
# Chevauchement et accord
# --------------------------------------------------------------------------------------


def test_deux_familles_identiques_se_chevauchent_totalement() -> None:
    signals = _signals(a=[1, 0, -1, 0, 1], b=[1, 0, -1, 0, 1])
    overlap, agreement = diversity.pairwise_overlap(signals)
    assert overlap.at["a", "b"] == pytest.approx(1.0)
    assert agreement.at["a", "b"] == pytest.approx(1.0)


def test_deux_familles_disjointes_ne_se_chevauchent_pas() -> None:
    signals = _signals(a=[1, 0, 1, 0], b=[0, 1, 0, 1])
    overlap, _ = diversity.pairwise_overlap(signals)
    assert overlap.at["a", "b"] == pytest.approx(0.0)


def test_paris_opposes_se_chevauchent_mais_ne_sont_pas_redondants() -> None:
    """Le point central : memes instants, sens inverse = diversite maximale.

    Sans cette distinction, on classerait comme redondantes deux familles qui parient
    exactement l'inverse l'une de l'autre — c'est-a-dire le contraire de la verite.
    """
    signals = _signals(a=[1, -1, 1, -1], b=[-1, 1, -1, 1])
    overlap, agreement = diversity.pairwise_overlap(signals)
    assert overlap.at["a", "b"] == pytest.approx(1.0)
    assert agreement.at["a", "b"] == pytest.approx(0.0)

    clusters = diversity.cluster(overlap, agreement)
    assert sorted(len(c) for c in clusters) == [1, 1], "des paris opposes ne forment pas un groupe"


def test_le_regroupement_isole_les_familles_redondantes() -> None:
    signals = _signals(
        a=[1, 0, -1, 0, 1, 0],
        b=[1, 0, -1, 0, 1, 0],  # identique a `a`
        c=[0, 1, 0, 1, 0, -1],  # disjointe
    )
    overlap, agreement = diversity.pairwise_overlap(signals)
    clusters = diversity.cluster(overlap, agreement)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]
    groupe = next(c for c in clusters if len(c) == 2)
    assert set(groupe) == {"a", "b"}


# --------------------------------------------------------------------------------------
# Nombre effectif de tests
# --------------------------------------------------------------------------------------


def test_des_tests_independants_comptent_tous() -> None:
    identity = pd.DataFrame(np.eye(5), index=list("abcde"), columns=list("abcde"))
    assert diversity.effective_n_tests(identity) == pytest.approx(5.0, abs=1e-6)


def test_des_tests_parfaitement_correles_n_en_valent_qu_un() -> None:
    """Cinq copies du meme test ne donnent pas cinq chances de faux positif."""
    ones = pd.DataFrame(np.ones((5, 5)), index=list("abcde"), columns=list("abcde"))
    assert diversity.effective_n_tests(ones) == pytest.approx(1.0, abs=1e-6)


def test_une_correlation_partielle_donne_un_compte_intermediaire() -> None:
    matrix = np.full((4, 4), 0.6)
    np.fill_diagonal(matrix, 1.0)
    corr = pd.DataFrame(matrix, index=list("abcd"), columns=list("abcd"))
    effective = diversity.effective_n_tests(corr)
    assert 1.0 < effective < 4.0


def test_le_compte_effectif_est_borne_par_le_nombre_de_tests() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(200, 6))
    corr = pd.DataFrame(np.corrcoef(noise.T), index=list("abcdef"), columns=list("abcdef"))
    assert 1.0 <= diversity.effective_n_tests(corr) <= 6.0


def test_matrice_vide_donne_zero() -> None:
    assert diversity.effective_n_tests(pd.DataFrame()) == 0.0


# --------------------------------------------------------------------------------------
# Rapport
# --------------------------------------------------------------------------------------


def test_le_rapport_chiffre_la_redondance() -> None:
    report = diversity.DiversityReport(
        overlap=pd.DataFrame(np.eye(4), index=list("abcd"), columns=list("abcd")),
        agreement=pd.DataFrame(np.ones((4, 4)), index=list("abcd"), columns=list("abcd")),
        correlation=pd.DataFrame(np.eye(4), index=list("abcd"), columns=list("abcd")),
        clusters=[["a"], ["b"], ["c"], ["d"]],
        n_families=4,
        n_effective=4.0,
    )
    assert report.redundancy_ratio == pytest.approx(0.0)
    assert "4 familles declarees" in report.summary()
    assert "Aucun groupe redondant" in report.summary()


def test_un_catalogue_redondant_est_signale() -> None:
    report = diversity.DiversityReport(
        overlap=pd.DataFrame(np.ones((4, 4)), index=list("abcd"), columns=list("abcd")),
        agreement=pd.DataFrame(np.ones((4, 4)), index=list("abcd"), columns=list("abcd")),
        correlation=pd.DataFrame(np.ones((4, 4)), index=list("abcd"), columns=list("abcd")),
        clusters=[["a", "b", "c", "d"]],
        n_families=4,
        n_effective=1.0,
    )
    assert report.redundancy_ratio == pytest.approx(1.0)
    assert "Groupes de familles redondantes" in report.summary()


def test_les_pires_paires_sont_triees() -> None:
    signals = _signals(a=[1, 0, 1, 0], b=[1, 0, 1, 0], c=[0, 1, 0, 0])
    overlap, agreement = diversity.pairwise_overlap(signals)
    report = diversity.DiversityReport(
        overlap=overlap,
        agreement=agreement,
        correlation=signals.corr().fillna(0.0),
        clusters=diversity.cluster(overlap, agreement),
        n_families=3,
        n_effective=diversity.effective_n_tests(signals.corr().fillna(0.0)),
    )
    pires = report.worst_pairs(3)
    assert list(pires.iloc[0][["A", "B"]]) == ["a", "b"]
    assert pires["chevauchement"].is_monotonic_decreasing
