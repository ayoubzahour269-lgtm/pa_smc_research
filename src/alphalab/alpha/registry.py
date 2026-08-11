"""Catalogue des familles explorees.

Les familles sont regroupees par theme dans `families/`, une classe par piste. Ajouter
une piste d'exploration coute donc une classe et une ligne ici — c'est l'inverse de la
version precedente du projet, ou chaque hypothese vivait dans un script jetable, ce qui
avait borne l'exploration a huit essais.

Aucun ordre de preference : l'ordre ci-dessous est celui de l'ecriture. En particulier,
les familles SMC ne beneficient d'aucun statut particulier — elles passent le meme
protocole que les autres.
"""

from __future__ import annotations

from alphalab.alpha.base import AlphaFamily
from alphalab.alpha.families import (
    crossasset,
    microstructure,
    reversion,
    structure,
    timing,
    trend,
)


def core_families() -> list[AlphaFamily]:
    """Familles ne dependant que de l'instrument lui-meme."""
    return [
        # Le temps
        timing.SessionDrift(),
        timing.OpeningRangeBreakout(),
        timing.TurnOfMonth(),
        # La tendance
        trend.ConditionedMomentum(),
        trend.VolatilityExpansion(),
        trend.TrendPullback(),
        # Le retour a la moyenne
        reversion.VwapReversion(),
        reversion.OvernightGap(),
        reversion.RangeFade(),
        # La structure de prix (SMC)
        structure.ChochRetracement(),
        structure.BosContinuation(),
        structure.LiquiditySweepReversal(),
        structure.FvgRetest(),
        # La microstructure
        microstructure.LiquidityRegimeBreakout(),
        microstructure.SpreadShockFade(),
    ]


def crossasset_families(peers: list[str]) -> list[AlphaFamily]:
    """Familles inter-actifs, instanciees une fois par actif pair disponible.

    Sans pair disponible la liste est vide — et le rapport doit le signaler, plutot
    que de laisser croire que ces pistes ont ete testees et rejetees.
    """
    families: list[AlphaFamily] = []
    for peer in peers:
        # Les deux sens sont testes separement : rien ne garantit a priori que les deux
        # actifs se suivent ou s'opposent, et le supposer serait deja un resultat non
        # mesure. Chaque sens compte comme un essai distinct.
        families.append(crossasset.LeadLag(peer, sign=1))
        families.append(crossasset.LeadLag(peer, sign=-1))
        families.append(crossasset.CorrelationDivergence(peer))
    return families


def all_families(peers: list[str] | None = None) -> list[AlphaFamily]:
    """Catalogue complet pour un symbole donne."""
    return [*core_families(), *crossasset_families(peers or [])]


def family_label(family: AlphaFamily) -> str:
    """Identifiant lisible, incluant l'actif pair et le sens quand ils existent."""
    with_peer = getattr(family, "name_with_peer", None)
    if with_peer is None:
        return family.name
    sign = getattr(family, "sign", None)
    if sign is None:
        return str(with_peer)
    return f"{with_peer}{'+' if sign == 1 else '-'}"
