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
    documented,
    ensemble,
    microstructure,
    regime,
    reversion,
    structure,
    timing,
)
from alphalab.backtest.engine import ExitPolicy


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
        # Anomalies documentees dans la litterature. Leurs parametres viennent de
        # l'exterieur : ils n'ont pas ete choisis en regardant nos donnees, ce qui
        # elimine une source de sur-ajustement propre aux familles maison.
        documented.IntradayMomentum(),
        documented.OvernightDrift(direction=1),
        documented.OvernightDrift(direction=-1),
        documented.FailedBreakout(),
        documented.RoundNumber(fade=True),
        documented.RoundNumber(fade=False),
        documented.TimeSeriesMomentum(),
        # Variantes de SORTIE sur une entree constante.
        #
        # Les campagnes S6 a S9 ont evalue 48 configurations avec une seule et meme
        # regle de sortie : stop fixe, objectif fixe, echeance. C'est un angle mort
        # majeur — la litterature de suivi de tendance tient la sortie pour au moins
        # aussi determinante que l'entree. Ces trois variantes isolent son effet en
        # gardant l'entree strictement identique.
        *_exit_variants(),
    ]


def _exit_variants() -> list[AlphaFamily]:
    """Meme entree, trois sorties differentes. Chaque variante est un essai distinct."""
    policies = [
        # Laisser courir : stop suiveur a 1R derriere l'extreme favorable.
        ExitPolicy(trailing_r=1.0),
        # Proteger tot : stop a l'entree des +1R atteint.
        ExitPolicy(breakeven_at_r=1.0),
        # Ne rien garder hors seance liquide : cloture d'office a 21h UTC.
        ExitPolicy(close_at_hour=21),
    ]
    variants: list[AlphaFamily] = []
    for policy in policies:
        family = breakout.Breakout(condition="aucun")
        family.exit_policy = policy
        family.exit_label = policy.label
        variants.append(family)
    return variants


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
    """Identifiant lisible, incluant la variante, l'actif pair, le sens et la sortie."""
    with_peer = getattr(family, "name_with_peer", None)
    base = family.name if with_peer is None else str(with_peer)
    sign = getattr(family, "sign", None)
    if with_peer is not None and sign is not None:
        base = f"{base}{'+' if sign == 1 else '-'}"
    exit_label = getattr(family, "exit_label", None)
    return base if exit_label is None else f"{base}/{exit_label}"
