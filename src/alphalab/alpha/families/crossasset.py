"""Familles inter-actifs : lead-lag et divergence de correlation.

C'est ici que se teste l'intuition a l'origine du projet : "l'or est lie a l'EUR/USD,
quand le dollar monte l'or descend". Cette observation est traitee comme une
**hypothese mesurable**, pas comme un postulat. Deux formes distinctes en decoulent :

  - **lead-lag** : un actif mene-t-il l'autre dans le temps ? Si oui, le mouvement du
    premier informe sur le suivant ;
  - **divergence de correlation** : quand deux actifs habituellement lies s'ecartent,
    l'ecart se resorbe-t-il ?

Les deux peuvent etre fausses. Une correlation forte n'implique NI que l'un mene
l'autre, NI que leurs ecarts se resorbent : deux series peuvent bouger ensemble sans
qu'aucune ne soit exploitable a partir de l'autre. C'est precisement ce que ces
familles mesurent.

Etat des donnees : avec seulement l'or et le Nasdaq, ce qui se teste est un axe
risk-on / risk-off. Le veritable facteur dollar exige EUR/USD (et idealement GBP/USD et
USD/JPY) — tant qu'il est absent, le module le declare indisponible plutot que de
bricoler un substitut trompeur.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators


def _peer_close(ctx: Context, peer: str) -> pd.Series | None:
    """Cloture d'un actif pair, alignee sans anticipation sur l'index de travail.

    Le `ffill` ne remonte que vers le passe : si le pair n'a pas cote sur cette barre,
    on utilise sa derniere cotation connue. Jamais la suivante.
    """
    frame = ctx.peers.get(peer)
    if frame is None or frame.empty:
        return None
    return frame["close"].reindex(ctx.df.index).ffill()


class LeadLag(AlphaFamily):
    """Un actif mene-t-il l'autre ?"""

    name = "lead_lag"
    question = (
        "Le rendement recent d'un actif correle predit-il le rendement suivant de "
        "celui qu'on trade ?"
    )

    def __init__(
        self, peer: str, lookback: int = 4, z_threshold: float = 1.5, sign: int = 1
    ) -> None:
        self.peer = peer
        self.lookback = lookback
        self.z_threshold = z_threshold
        # +1 : on suit le pair. -1 : on prend le sens inverse (actifs anticorreles).
        self.sign = sign

    @property
    def name_with_peer(self) -> str:
        return f"{self.name}[{self.peer}]"

    def parameters(self) -> dict[str, Any]:
        return {
            "pair": self.peer,
            "lookback": self.lookback,
            "seuil_z": self.z_threshold,
            "sens": self.sign,
        }

    def signal(self, ctx: Context) -> pd.Series:
        peer_close = _peer_close(ctx, self.peer)
        if peer_close is None:
            return pd.Series(0, index=ctx.df.index, name="signal")

        peer_return = np.log(peer_close).diff(self.lookback)
        z = indicators.zscore(peer_return, 100)
        raw = np.where(
            z >= self.z_threshold, self.sign, np.where(z <= -self.z_threshold, -self.sign, 0)
        )
        signal = pd.Series(raw, index=ctx.df.index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class CorrelationDivergence(AlphaFamily):
    """Quand deux actifs lies s'ecartent, l'ecart se resorbe-t-il ?"""

    name = "divergence_correlation"
    question = (
        "Lorsque l'ecart normalise entre deux actifs habituellement correles atteint "
        "un extreme, converge-t-il ensuite ?"
    )

    def __init__(
        self,
        peer: str,
        corr_window: int = 240,
        spread_window: int = 96,
        z_threshold: float = 2.0,
        min_abs_corr: float = 0.3,
    ) -> None:
        self.peer = peer
        self.corr_window = corr_window
        self.spread_window = spread_window
        self.z_threshold = z_threshold
        self.min_abs_corr = min_abs_corr

    @property
    def name_with_peer(self) -> str:
        return f"{self.name}[{self.peer}]"

    def parameters(self) -> dict[str, Any]:
        return {
            "pair": self.peer,
            "fenetre_correlation": self.corr_window,
            "fenetre_ecart": self.spread_window,
            "seuil_z": self.z_threshold,
            "correlation_min": self.min_abs_corr,
        }

    def signal(self, ctx: Context) -> pd.Series:
        peer_close = _peer_close(ctx, self.peer)
        if peer_close is None:
            return pd.Series(0, index=ctx.df.index, name="signal")

        own = np.log(ctx.df["close"])
        other = np.log(peer_close)
        own_ret = own.diff()
        other_ret = other.diff()

        # Correlation glissante des RENDEMENTS. La faire sur les prix produirait des
        # correlations proches de 1 entre deux series non stationnaires, sans contenu.
        corr = own_ret.rolling(self.corr_window).corr(other_ret)

        # Ecart normalise : on retire la relation moyenne recente entre les deux
        # niveaux, puis on mesure la deviation restante en ecarts-types.
        relation = (own - other).rolling(self.spread_window).mean()
        spread = (own - other) - relation
        z = indicators.zscore(spread, self.spread_window)

        # La convergence ne se parie que si le lien existe encore. Sans ce filtre, on
        # parierait sur un retour a une relation qui n'a plus cours.
        linked = corr.abs() >= self.min_abs_corr
        raw = np.where(
            (z >= self.z_threshold) & linked,
            -1,
            np.where((z <= -self.z_threshold) & linked, 1, 0),
        )
        signal = pd.Series(raw, index=ctx.df.index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class Cointegration(AlphaFamily):
    """Deux actifs partagent-ils un equilibre de long terme vers lequel ils reviennent ?

    Difference essentielle avec `CorrelationDivergence`, qui utilise une simple
    difference de log-prix : ici le ratio de couverture est ESTIME sur une fenetre
    glissante, et la convergence n'est pariee que si l'ecart revient effectivement vers
    sa moyenne a une vitesse mesurable (demi-vie).

    Deux actifs peuvent etre fortement correles sans etre cointegres : ils bougent
    ensemble mais leur ecart derive sans jamais revenir. Parier la convergence dans ce
    cas est une facon classique de perdre lentement, puis brutalement.
    """

    name = "cointegration"
    question = (
        "Une fois le ratio de couverture estime sur le passe, l'ecart entre deux actifs "
        "revient-il vers sa moyenne assez vite pour etre exploitable ?"
    )

    def __init__(
        self,
        peer: str,
        beta_window: int = 480,
        z_window: int = 120,
        z_threshold: float = 2.0,
        max_half_life: float = 48.0,
    ) -> None:
        self.peer = peer
        self.beta_window = beta_window
        self.z_window = z_window
        self.z_threshold = z_threshold
        self.max_half_life = max_half_life

    @property
    def name_with_peer(self) -> str:
        return f"{self.name}[{self.peer}]"

    def parameters(self) -> dict[str, Any]:
        return {
            "pair": self.peer,
            "fenetre_beta": self.beta_window,
            "fenetre_z": self.z_window,
            "seuil_z": self.z_threshold,
            "demi_vie_max": self.max_half_life,
        }

    def signal(self, ctx: Context) -> pd.Series:
        peer_close = _peer_close(ctx, self.peer)
        if peer_close is None:
            return pd.Series(0, index=ctx.df.index, name="signal")

        own = np.log(ctx.df["close"])
        other = np.log(peer_close)

        # Ratio de couverture par regression glissante : cov / var, sur le passe seul.
        covariance = own.rolling(self.beta_window).cov(other)
        variance = other.rolling(self.beta_window).var(ddof=1).replace(0.0, np.nan)
        beta = covariance / variance

        spread = own - beta * other
        z = indicators.zscore(spread, self.z_window)

        # Demi-vie du retour a la moyenne, estimee sur un AR(1) glissant de l'ecart.
        # Une demi-vie infinie ou negative signale un ecart qui derive au lieu de
        # revenir : dans ce cas on ne parie pas.
        lagged = spread.shift(1)
        delta = spread - lagged
        slope = lagged.rolling(self.z_window).cov(delta) / lagged.rolling(self.z_window).var(
            ddof=1
        ).replace(0.0, np.nan)
        half_life = -np.log(2.0) / slope.where(slope < 0)
        reverting = half_life.between(1.0, self.max_half_life)

        raw = np.where(
            (z >= self.z_threshold) & reverting,
            -1,
            np.where((z <= -self.z_threshold) & reverting, 1, 0),
        )
        signal = pd.Series(raw, index=ctx.df.index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)
