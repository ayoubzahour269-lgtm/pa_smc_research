"""Risque de portefeuille ajuste de la correlation.

Le probleme que ce module resout est exactement celui souleve dans le cahier des
charges : « l'or est lie a l'EUR/USD, quand le dollar monte l'or descend ». Une
consequence directe, et couteuse si on l'ignore :

    long or + short EUR/USD  ==  deux fois le meme pari contre le dollar

Additionner les risques ligne a ligne donnerait 2 %. Le risque reel est proche de 4 %,
parce que les deux positions gagnent et perdent ensemble. Un compte peut mourir en
croyant etre diversifie.

La mesure correcte est la norme quadratique ponderee par la matrice de correlation :

    risque = sqrt( w' Σ w )

qui vaut la somme quand les positions sont parfaitement correlees, et nettement moins
quand elles sont independantes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.config import (
    DUPLICATE_CORRELATION,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_R,
    MAX_PER_SYMBOL,
    MAX_PORTFOLIO_RISK,
)


@dataclass(frozen=True, slots=True)
class Exposure:
    """Une exposition candidate : symbole, sens, fraction de capital risquee."""

    symbol: str
    direction: int
    risk_fraction: float

    @property
    def signed_risk(self) -> float:
        return self.direction * self.risk_fraction


def correlation_matrix(
    returns: Mapping[str, pd.Series], *, window: int = 240, min_periods: int = 60
) -> pd.DataFrame:
    """Matrice de correlation glissante des rendements, au dernier point disponible.

    Calculee sur les RENDEMENTS, jamais sur les prix : deux series de prix non
    stationnaires affichent des correlations proches de 1 sans aucun contenu.
    """
    if not returns:
        return pd.DataFrame()
    frame = pd.DataFrame(returns).dropna(how="all")
    if frame.empty:
        return pd.DataFrame()
    recent = frame.tail(window)
    matrix = recent.corr(min_periods=min_periods)
    return matrix.fillna(0.0).astype(float)


def portfolio_risk(exposures: Sequence[Exposure], correlations: pd.DataFrame) -> float:
    """Risque total ajuste de la correlation, en fraction du capital.

    Les positions inverses sur actifs positivement correles s'annulent partiellement, et
    les positions de meme sens s'additionnent — ce qui est le comportement recherche.
    """
    if not exposures:
        return 0.0
    weights = np.array([e.signed_risk for e in exposures], dtype=float)
    symbols = [e.symbol for e in exposures]

    n = len(exposures)
    sigma = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = 0.0
            if (
                not correlations.empty
                and symbols[i] in correlations.index
                and symbols[j] in correlations.columns
            ):
                value = correlations.at[symbols[i], symbols[j]]
                rho = 0.0 if pd.isna(value) else float(value)  # type: ignore[arg-type]
            sigma[i, j] = sigma[j, i] = rho

    variance = float(weights @ sigma @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def duplicate_pairs(
    exposures: Sequence[Exposure],
    correlations: pd.DataFrame,
    *,
    threshold: float = DUPLICATE_CORRELATION,
) -> list[tuple[str, str, float]]:
    """Couples de positions qui constituent en realite le meme pari.

    Deux positions de meme sens sur actifs fortement correles, ou de sens opposes sur
    actifs fortement ANTI-correles, sont redondantes. C'est le cas or/EUR/USD.
    """
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(exposures):
        for b in exposures[i + 1 :]:
            if correlations.empty or a.symbol not in correlations.index:
                continue
            if b.symbol not in correlations.columns:
                continue
            value = correlations.at[a.symbol, b.symbol]
            if pd.isna(value):
                continue
            effective = float(value) * a.direction * b.direction  # type: ignore[arg-type]
            if effective >= threshold:
                pairs.append((a.symbol, b.symbol, round(effective, 3)))
    return pairs


@dataclass(frozen=True, slots=True)
class PortfolioCheck:
    """Verdict des plafonds de portefeuille."""

    accepted: list[Exposure]
    rejected: list[tuple[Exposure, str]]
    risk: float
    duplicates: list[tuple[str, str, float]]

    @property
    def ok(self) -> bool:
        return not self.rejected


def apply_caps(
    candidates: Sequence[Exposure],
    correlations: pd.DataFrame,
    *,
    max_concurrent: int = MAX_CONCURRENT_POSITIONS,
    max_per_symbol: int = MAX_PER_SYMBOL,
    max_risk: float = MAX_PORTFOLIO_RISK,
) -> PortfolioCheck:
    """Retient les candidats par ordre de presentation, sous contrainte de plafonds.

    Les candidats sont supposes deja tries du meilleur au moins bon : c'est donc le
    meilleur qui passe en premier, et les plafonds ecartent la queue de liste.
    """
    accepted: list[Exposure] = []
    rejected: list[tuple[Exposure, str]] = []
    per_symbol: dict[str, int] = {}

    for candidate in candidates:
        if len(accepted) >= max_concurrent:
            rejected.append((candidate, "plafond de positions simultanees"))
            continue
        if per_symbol.get(candidate.symbol, 0) >= max_per_symbol:
            rejected.append((candidate, "plafond par symbole"))
            continue
        trial = [*accepted, candidate]
        risk = portfolio_risk(trial, correlations)
        if risk > max_risk:
            rejected.append(
                (
                    candidate,
                    f"plafond de risque ajuste de la correlation "
                    f"({risk:.3%} > {max_risk:.1%})",
                )
            )
            continue
        accepted.append(candidate)
        per_symbol[candidate.symbol] = per_symbol.get(candidate.symbol, 0) + 1

    return PortfolioCheck(
        accepted=accepted,
        rejected=rejected,
        risk=portfolio_risk(accepted, correlations),
        duplicates=duplicate_pairs(accepted, correlations),
    )


def daily_loss_breached(realized_r: float, *, limit: float = MAX_DAILY_LOSS_R) -> bool:
    """Vrai si la perte du jour atteint la limite : on arrete pour la journee.

    La limite journaliere est ce qui empeche une mauvaise serie de devenir un mauvais
    mois. Elle prime sur toute qualite de signal.
    """
    return realized_r <= -abs(limit)
