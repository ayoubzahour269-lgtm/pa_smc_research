"""Mesure de la diversite reelle d'un catalogue de familles.

Annoncer "18 strategies testees" ne veut rien dire si plusieurs d'entre elles se
declenchent aux memes moments et dans le meme sens. Ce module mesure ce que le
catalogue explore VRAIMENT, au lieu de le supposer.

Deux consequences pratiques :

  1. **Honnetete de la revendication.** Si quatre familles partagent le meme
     declencheur sous des filtres differents, l'espace explore est plus etroit que le
     compte ne le laisse croire. Le rapport doit le dire.

  2. **Justesse de la correction pour tests multiples.** Benjamini-Hochberg corrige sur
     le NOMBRE d'essais. Si ces essais sont fortement correles, ils ne constituent pas
     autant de chances independantes de tomber sur un faux positif : corriger sur le
     compte brut devient inutilement severe, et peut faire rejeter un resultat reel.
     On estime donc un nombre EFFECTIF de tests independants.

Le point d'equilibre compte : on ne cherche ni a gonfler le compte (ce qui rendrait la
correction trop laxiste), ni a le laisser brut quand les tests sont redondants (ce qui
la rendrait trop severe). L'estimation par valeurs propres repond exactement a ca.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.alpha.registry import family_label


def signal_matrix(ctx: Context, families: Sequence[AlphaFamily]) -> pd.DataFrame:
    """Signaux de toutes les familles, alignes sur le meme index."""
    columns = {
        family_label(f): f.signal(ctx).reindex(ctx.df.index).fillna(0).astype(int)
        for f in families
    }
    return pd.DataFrame(columns, index=ctx.df.index)


@dataclass(frozen=True, slots=True)
class DiversityReport:
    """Ce que le catalogue explore reellement."""

    overlap: pd.DataFrame
    agreement: pd.DataFrame
    correlation: pd.DataFrame
    clusters: list[list[str]]
    n_families: int
    n_effective: float

    @property
    def redundancy_ratio(self) -> float:
        """1 = catalogue parfaitement redondant, 0 = familles independantes."""
        if self.n_families <= 1:
            return 0.0
        return 1.0 - (self.n_effective - 1) / (self.n_families - 1)

    def worst_pairs(self, n: int = 10) -> pd.DataFrame:
        """Paires les plus redondantes, triees."""
        rows = []
        names = list(self.overlap.columns)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                accord = float(self.agreement.at[a, b])  # type: ignore[arg-type]
                rows.append(
                    {
                        "A": a,
                        "B": b,
                        "chevauchement": round(float(self.overlap.at[a, b]), 3),  # type: ignore[arg-type]
                        "accord_de_sens": round(accord, 2) if np.isfinite(accord) else None,
                    }
                )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        return table.sort_values("chevauchement", ascending=False).head(n)

    def summary(self) -> str:
        lines = [
            f"{self.n_families} familles declarees, "
            f"{self.n_effective:.1f} tests effectivement independants "
            f"(redondance {self.redundancy_ratio:.0%}).",
        ]
        grouped = [c for c in self.clusters if len(c) > 1]
        if grouped:
            lines.append("Groupes de familles redondantes :")
            for cluster in grouped:
                lines.append("  - " + ", ".join(cluster))
        else:
            lines.append("Aucun groupe redondant detecte au seuil retenu.")
        return "\n".join(lines)


def pairwise_overlap(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chevauchement (Jaccard) et accord de direction entre familles.

    Le chevauchement seul ne suffit pas : deux familles peuvent se declencher aux memes
    instants en pariant l'INVERSE, ce qui est de la diversite maximale, pas de la
    redondance. D'ou la seconde matrice.
    """
    names = list(signals.columns)
    fired = signals != 0
    overlap = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    agreement = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = fired[a] & fired[b]
            union = int((fired[a] | fired[b]).sum())
            inter = int(both.sum())
            jaccard = inter / union if union else 0.0
            same = float((signals.loc[both, a] == signals.loc[both, b]).mean()) if inter else np.nan
            overlap.at[a, b] = overlap.at[b, a] = jaccard
            agreement.at[a, b] = agreement.at[b, a] = same
    return overlap, agreement


def effective_n_tests(correlation: pd.DataFrame) -> float:
    """Nombre de tests independants equivalents, par decomposition en valeurs propres.

    Methode de Li et Ji : chaque valeur propre contribue sa partie entiere (1 si elle
    depasse 1) plus sa partie fractionnaire. Une famille de tests parfaitement
    correles donne une seule grande valeur propre, donc M_eff proche de 1 ; des tests
    independants donnent des valeurs propres toutes egales a 1, donc M_eff = M.

    Le resultat est borne entre 1 et le nombre de tests.
    """
    if correlation.empty:
        return 0.0
    raw = np.nan_to_num(correlation.to_numpy(dtype=float), nan=0.0)
    np.fill_diagonal(raw, 1.0)
    # Symetrisation : les arrondis peuvent rendre la matrice legerement asymetrique.
    matrix = (raw + raw.T) / 2.0
    eigenvalues = np.abs(np.linalg.eigvalsh(matrix))
    contributions = (eigenvalues >= 1.0).astype(float) + (eigenvalues - np.floor(eigenvalues))
    return float(np.clip(contributions.sum(), 1.0, matrix.shape[0]))


def cluster(
    overlap: pd.DataFrame, agreement: pd.DataFrame, *, threshold: float = 0.10
) -> list[list[str]]:
    """Regroupe les familles redondantes.

    Deux familles sont dans le meme groupe si elles se declenchent souvent ensemble ET
    parient dans le meme sens. Regroupement transitif par simple parcours de graphe.
    """
    names = list(overlap.columns)
    adjacency: dict[str, set[str]] = {n: set() for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            accord = float(agreement.at[a, b])  # type: ignore[arg-type]
            same_bet = bool(np.isfinite(accord) and accord >= 0.8)
            if float(overlap.at[a, b]) >= threshold and same_bet:  # type: ignore[arg-type]
                adjacency[a].add(b)
                adjacency[b].add(a)

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for name in names:
        if name in seen:
            continue
        stack, group = [name], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            stack.extend(adjacency[current] - seen)
        clusters.append(sorted(group))
    return clusters


def analyse(
    ctx: Context, families: Sequence[AlphaFamily], *, threshold: float = 0.10
) -> DiversityReport:
    """Rapport complet de diversite pour un catalogue et un contexte donnes."""
    signals = signal_matrix(ctx, families)
    overlap, agreement = pairwise_overlap(signals)
    correlation = signals.corr().fillna(0.0)
    return DiversityReport(
        overlap=overlap,
        agreement=agreement,
        correlation=correlation,
        clusters=cluster(overlap, agreement, threshold=threshold),
        n_families=len(signals.columns),
        n_effective=effective_n_tests(correlation),
    )
