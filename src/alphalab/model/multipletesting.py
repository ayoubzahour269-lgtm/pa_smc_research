"""Correction pour tests multiples — la contrepartie obligatoire d'une exploration large.

Le probleme, chiffre : a 5 % de seuil, tester 100 configurations produit en moyenne 5
"strategies gagnantes" alors qu'aucune ne contient d'information. Plus on explore, plus
on trouve — et ce qu'on trouve est de plus en plus souvent du bruit. Une exploration
large SANS cette correction n'est pas plus informative qu'une exploration etroite : elle
est plus dangereuse, parce qu'elle produit des faux positifs plus convaincants.

Deux quantites font tout le travail :

  - la **p-value de permutation**, issue du temoin aleatoire apparie. Elle repond a :
    "quelle est la probabilite qu'un tirage au hasard, de meme densite, meme biais,
    meme heure et meme geometrie, fasse aussi bien ?" ;
  - le **nombre total d'essais**, lu dans le journal append-only. Il ne peut pas etre
    reconstitue apres coup : on oublie systematiquement les essais rates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alphalab.types import FloatArray


def permutation_pvalue(observed: float, controls: Sequence[float] | FloatArray) -> float:
    """p-value unilaterale : proportion de temoins qui egalent ou depassent l'observe.

    La correction `(1 + k) / (1 + n)` est standard : elle interdit une p-value nulle,
    qui pretendrait une certitude que `n` tirages ne peuvent pas fournir.
    """
    arr = np.asarray(controls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + int((arr >= observed).sum())) / (1 + arr.size))


@dataclass(frozen=True, slots=True)
class MultipleTestResult:
    """Verdict corrige d'un ensemble d'hypotheses testees ensemble."""

    labels: tuple[str, ...]
    pvalues: tuple[float, ...]
    qvalues: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    n_trials: int

    @property
    def n_survivors(self) -> int:
        return int(sum(self.rejected))

    def summary(self) -> str:
        if self.n_survivors == 0:
            return (
                f"Aucune hypothese ne survit a la correction (m = {self.n_trials} essais, "
                f"FDR = {self.alpha}). Les esperances positives observees sont compatibles "
                "avec le hasard."
            )
        survivors = [lbl for lbl, ok in zip(self.labels, self.rejected, strict=True) if ok]
        return (
            f"{self.n_survivors} hypothese(s) survivante(s) sur {self.n_trials} essais "
            f"(FDR = {self.alpha}) : {', '.join(survivors)}"
        )


def benjamini_hochberg(
    labels: Sequence[str],
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
    n_trials: int | None = None,
) -> MultipleTestResult:
    """Controle du taux de fausses decouvertes (FDR) par la procedure de Benjamini-Hochberg.

    `n_trials` permet de corriger sur le nombre REEL d'essais menes, qui depasse le
    nombre d'hypotheses presentees ici des lors qu'on a teste des variantes ecartees en
    cours de route. Ne corriger que sur les hypotheses retenues reviendrait a
    dissimuler les essais rates — et donc a annuler l'effet de la correction.

    On choisit le FDR plutot que Bonferroni : Bonferroni controle la probabilite de
    la moindre fausse decouverte, ce qui est trop severe pour un travail exploratoire
    et ferait rejeter des pistes reelles. Le FDR borne la PROPORTION de fausses
    decouvertes parmi celles retenues, ce qui correspond a la question posee ici.
    """
    p = np.asarray(pvalues, dtype=float)
    m = int(n_trials) if n_trials is not None else int(p.size)
    m = max(m, int(p.size), 1)

    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))
    ranks = np.arange(1, p.size + 1, dtype=float)

    sorted_p = p[order]
    # q_i = min sur j >= i de (m/j) * p_j, en parcourant de la fin vers le debut.
    scaled = np.where(np.isfinite(sorted_p), sorted_p * m / ranks, np.inf)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    rejected = np.isfinite(p) & (q <= alpha)

    return MultipleTestResult(
        labels=tuple(labels),
        pvalues=tuple(float(x) for x in p),
        qvalues=tuple(float(x) for x in q),
        rejected=tuple(bool(x) for x in rejected),
        alpha=float(alpha),
        n_trials=m,
    )


def bonferroni(
    labels: Sequence[str],
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
    n_trials: int | None = None,
) -> MultipleTestResult:
    """Correction de Bonferroni, fournie comme borne severe de reference."""
    p = np.asarray(pvalues, dtype=float)
    m = max(int(n_trials) if n_trials is not None else int(p.size), int(p.size), 1)
    q = np.clip(p * m, 0.0, 1.0)
    rejected = np.isfinite(p) & (q <= alpha)
    return MultipleTestResult(
        labels=tuple(labels),
        pvalues=tuple(float(x) for x in p),
        qvalues=tuple(float(x) for x in q),
        rejected=tuple(bool(x) for x in rejected),
        alpha=float(alpha),
        n_trials=m,
    )
