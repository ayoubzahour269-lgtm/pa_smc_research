"""Correction pour tests multiples.

Ces tests protegent la conclusion du projet. Sans correction, explorer largement
GARANTIT de trouver des "strategies gagnantes" qui ne sont que du bruit — et plus
l'exploration est large, plus elles sont convaincantes.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.model.multipletesting import (
    benjamini_hochberg,
    bonferroni,
    permutation_pvalue,
)

# --------------------------------------------------------------------------------------
# p-value de permutation
# --------------------------------------------------------------------------------------


def test_pvalue_faible_quand_l_observe_domine_les_temoins() -> None:
    assert permutation_pvalue(10.0, np.zeros(99)) == pytest.approx(1 / 100)


def test_pvalue_elevee_quand_les_temoins_font_mieux() -> None:
    assert permutation_pvalue(-1.0, np.zeros(99)) == pytest.approx(1.0)


def test_pvalue_jamais_nulle() -> None:
    """Une p-value nulle pretendrait une certitude que n tirages ne fournissent pas."""
    assert permutation_pvalue(1e9, np.zeros(10)) > 0.0


def test_pvalue_sans_temoin_est_indefinie() -> None:
    assert np.isnan(permutation_pvalue(1.0, []))


def test_pvalue_est_environ_uniforme_sous_l_hypothese_nulle() -> None:
    """Sous H0, la p-value doit se comporter comme une loi uniforme.

    C'est la propriete sur laquelle repose toute la correction : si elle est fausse,
    le controle du taux de fausses decouvertes ne vaut rien.
    """
    rng = np.random.default_rng(0)
    pvals = [
        permutation_pvalue(float(rng.normal()), rng.normal(size=199)) for _ in range(400)
    ]
    arr = np.asarray(pvals)
    assert 0.03 <= float((arr <= 0.05).mean()) <= 0.09
    assert 0.4 <= float(arr.mean()) <= 0.6


# --------------------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------------------


def test_bh_retient_un_signal_franc_isole() -> None:
    result = benjamini_hochberg(["vrai", "a", "b", "c"], [0.0001, 0.6, 0.7, 0.9])
    assert result.rejected[0]
    assert not any(result.rejected[1:])
    assert result.n_survivors == 1


def test_bh_ne_retient_rien_sous_l_hypothese_nulle() -> None:
    """20 essais, p-values uniformes : aucune decouverte ne doit survivre."""
    rng = np.random.default_rng(3)
    p = list(rng.uniform(size=20))
    result = benjamini_hochberg([f"h{i}" for i in range(20)], p)
    assert result.n_survivors == 0
    assert "Aucune hypothese ne survit" in result.summary()


def test_le_nombre_d_essais_reels_durcit_la_correction() -> None:
    """Le point central : les essais non presentes doivent quand meme compter.

    Une p-value de 0,01 sur 3 hypotheses presentees survit. La meme p-value, quand on
    declare avoir en realite teste 300 configurations, ne survit plus — et c'est le
    comportement correct.
    """
    p = [0.01, 0.5, 0.9]
    sans = benjamini_hochberg(["a", "b", "c"], p)
    avec = benjamini_hochberg(["a", "b", "c"], p, n_trials=300)
    assert sans.n_survivors == 1
    assert avec.n_survivors == 0
    assert avec.qvalues[0] > sans.qvalues[0]


def test_les_qvalues_sont_monotones_et_bornees() -> None:
    rng = np.random.default_rng(5)
    p = np.sort(rng.uniform(size=30))
    result = benjamini_hochberg([f"h{i}" for i in range(30)], list(p))
    q = np.asarray(result.qvalues)
    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.all(np.diff(q) >= -1e-12), "les q-values doivent croitre avec les p-values"


def test_bh_tolere_les_pvalues_indefinies() -> None:
    """Une famille sans trade n'a pas de p-value : elle ne doit ni planter ni survivre."""
    result = benjamini_hochberg(["a", "b"], [0.001, float("nan")])
    assert result.rejected[0]
    assert not result.rejected[1]


def test_bonferroni_est_plus_severe_que_bh() -> None:
    p = [0.004, 0.02, 0.03, 0.2]
    bh = benjamini_hochberg(list("abcd"), p)
    bf = bonferroni(list("abcd"), p)
    assert bf.n_survivors <= bh.n_survivors


def test_le_resume_nomme_les_survivants() -> None:
    result = benjamini_hochberg(["gagnante", "autre"], [0.00001, 0.9])
    assert "gagnante" in result.summary()
