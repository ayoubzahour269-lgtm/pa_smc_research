"""Catalogue des familles explorees.

Deux principes de composition, tous deux issus d'une mesure et non d'une preference :

  1. **Aucun statut privilegie.** Les familles SMC passent le meme protocole que les
     autres. C'est une difference de fond avec l'orientation initiale du depot, qui
     designait la branche SMC comme la suite naturelle du travail.

  2. **Pas de fausse largeur.** La mesure de diversite (`alpha/diversity.py`) a montre
     que trois familles de la premiere campagne — cassure filtree par l'ADX, cassure
     filtree par le spread, et cassure nue — partageaient le meme declencheur et
     s'accordaient sur la direction dans 100 % des cas. Elles sont desormais une seule
     famille parametree. Trois entrees au catalogue pour une idee, cela donne
     l'illusion d'explorer large et durcit la correction pour tests multiples sans
     raison.

`opening_range` et `smc_bos` cassent aussi un extreme, mais un extreme defini
autrement — range de seance pour l'une, pivot de structure confirme pour l'autre — et
leur chevauchement mesure reste sous 13 %. Elles restent donc distinctes.
"""

from __future__ import annotations

from alphalab.alpha.base import AlphaFamily
from alphalab.alpha.families import (
    breakout,
    crossasset,
    ensemble,
    microstructure,
    regime,
    reversion,
    structure,
    timing,
)


def core_families() -> list[AlphaFamily]:
    """Familles ne dependant que de l'instrument lui-meme."""
    return [
        # Le temps
        timing.SessionDrift(),
        timing.OpeningRangeBreakout(),
        timing.TurnOfMonth(),
        # La cassure, unifiee : un declencheur, trois filtres declares
        breakout.Breakout(condition="aucun"),
        breakout.Breakout(condition="adx"),
        breakout.Breakout(condition="compression"),
        breakout.Breakout(condition="spread_bas"),
        # La tendance
        trend_family(),
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
        microstructure.SpreadShockFade(),
        # Les regimes et les horizons — mecanismes absents de la premiere campagne
        regime.VolatilityRegimeSwitch(),
        regime.MultiDayMomentum(),
        regime.IntradaySeasonality(),
        # Le consensus entre mecanismes distincts
        ensemble.Consensus(),
    ]


def trend_family() -> AlphaFamily:
    """Repli de tendance : conserve car son declencheur (croisement du RSI) n'a rien
    de commun avec une cassure d'extreme."""
    from alphalab.alpha.families import trend

    return trend.TrendPullback()


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
        families.append(crossasset.Cointegration(peer))
    return families


def all_families(peers: list[str] | None = None) -> list[AlphaFamily]:
    """Catalogue complet pour un symbole donne."""
    return [*core_families(), *crossasset_families(peers or [])]


def family_label(family: AlphaFamily) -> str:
    """Identifiant lisible, incluant la variante, l'actif pair et le sens s'ils existent."""
    with_peer = getattr(family, "name_with_peer", None)
    if with_peer is None:
        return family.name
    sign = getattr(family, "sign", None)
    if sign is None:
        return str(with_peer)
    return f"{with_peer}{'+' if sign == 1 else '-'}"
