"""Paliers de conviction et dimensionnement.

Le point de conception de ce module repond a une exigence explicite : **avoir une
position chaque jour**. C'est tenable — les prix bougent tous les jours — a condition de
ne pas confondre "avoir un plan" et "engager la meme somme". Un jour sans candidat de
qualite produit donc une ligne de grade C, a taille minimale, avec sa probabilite
calibree affichee en clair.

La taille ne vient pas d'une opinion mais de l'esperance mesuree :

    E[R] = p x R_objectif - (1 - p) x 1 - cout

ou `p` est la probabilite CALIBREE hors echantillon. Un candidat dont l'esperance est
negative reste affiche — il est le meilleur du jour — mais son grade et sa taille le
disent sans ambiguite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from alphalab.config import (
    GRADE_A_MIN_PERCENTILE,
    GRADE_A_MIN_R,
    GRADE_B_MIN_R,
    GRADE_RISK,
)
from alphalab.types import FloatArray

Grade = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class Sizing:
    """Verdict de dimensionnement d'un candidat."""

    grade: Grade
    risk_fraction: float
    expected_r: float
    probability: float
    rationale: str

    @property
    def is_paper(self) -> bool:
        """Un grade C d'esperance negative n'est pas une recommandation d'engagement."""
        return self.grade == "C" and self.expected_r < 0


def expected_r(probability: float, reward_ratio: float, cost_r: float = 0.0) -> float:
    """Esperance en R d'un candidat, cout compris."""
    return float(probability * reward_ratio - (1.0 - probability) - cost_r)


def grade_candidate(
    probability: float,
    reward_ratio: float,
    *,
    cost_r: float = 0.0,
    percentile: float = 0.0,
) -> Sizing:
    """Attribue un palier et une fraction de risque.

    `percentile` est le rang du candidat parmi ceux du jour (0 = le plus faible, 1 = le
    meilleur). Il sert au grade A : une esperance elevee ne suffit pas, il faut aussi que
    ce soit la meilleure opportunite disponible, sans quoi on engagerait le risque
    maximal sur plusieurs candidats mediocres le meme jour.
    """
    e = expected_r(probability, reward_ratio, cost_r)

    if e >= GRADE_A_MIN_R and percentile >= GRADE_A_MIN_PERCENTILE:
        grade: Grade = "A"
        rationale = f"esperance {e:+.3f}R et candidat du haut de classement"
    elif e >= GRADE_B_MIN_R:
        grade = "B"
        rationale = (
            f"esperance {e:+.3f}R positive mais insuffisante pour le palier A"
            if e < GRADE_A_MIN_R
            else f"esperance {e:+.3f}R, hors du haut de classement"
        )
    else:
        grade = "C"
        rationale = (
            f"esperance {e:+.3f}R NEGATIVE — meilleur candidat du jour malgre tout, "
            "taille minimale ou papier"
        )

    return Sizing(
        grade=grade,
        risk_fraction=GRADE_RISK[grade],
        expected_r=e,
        probability=float(probability),
        rationale=rationale,
    )


def percentiles(values: FloatArray) -> FloatArray:
    """Rang relatif de chaque valeur dans [0, 1]. Une seule valeur -> 1.0."""
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = values.argsort().argsort().astype(np.float64)
    result: FloatArray = order / (n - 1)
    return result


def position_size(
    capital: float, risk_fraction: float, stop_distance: float, point_value: float = 1.0
) -> float:
    """Taille de position en unites de l'instrument.

    Le raisonnement est le seul qui protege reellement : on part du montant qu'on accepte
    de perdre, et la distance au stop determine la taille. Jamais l'inverse.
    """
    if stop_distance <= 0 or point_value <= 0:
        return 0.0
    return float(capital * risk_fraction / (stop_distance * point_value))
