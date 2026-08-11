"""Famille de cassure, unifiee et parametree.

Cette famille remplace trois familles precedentes — `momentum_conditionne`,
`microstructure_liquidite` et la cassure nue — dont la mesure de diversite a montre
qu'elles partageaient le MEME declencheur (une cloture depasse un extreme recent) et
s'accordaient sur la direction dans 100 % des cas quand elles se declenchaient
ensemble. Trois entrees au catalogue pour une seule idee, c'est une exploration qui
parait plus large qu'elle ne l'est.

Ici, le declencheur est unique et les filtres sont des PARAMETRES declares. Le
catalogue expose une variante par filtre, mais elles sont regroupees explicitement, et
le nombre effectif de tests independants en tient compte.

Note : `opening_range` et `smc_bos` restent des familles distinctes. Elles cassent
aussi un extreme, mais un extreme defini autrement — le range d'une seance pour l'une,
un pivot de structure confirme pour l'autre — et leur chevauchement mesure reste sous
13 %.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators, sessions

BreakoutFilter = Literal["aucun", "adx", "compression", "spread_bas"]


class Breakout(AlphaFamily):
    """La cassure d'un extreme recent se poursuit-elle, et sous quelle condition ?"""

    name = "cassure"
    question = (
        "Une cloture qui depasse l'extreme des N barres precedentes annonce-t-elle une "
        "poursuite du mouvement, et un filtre de contexte change-t-il la reponse ?"
    )

    def __init__(
        self,
        lookback: int = 24,
        condition: BreakoutFilter = "aucun",
        session: str | None = "newyork",
        adx_min: float = 25.0,
        ratio_max: float = 0.8,
        spread_quantile: float = 0.4,
        spread_window: int = 480,
    ) -> None:
        self.lookback = lookback
        self.condition = condition
        self.session = session
        self.adx_min = adx_min
        self.ratio_max = ratio_max
        self.spread_quantile = spread_quantile
        self.spread_window = spread_window

    @property
    def name_with_peer(self) -> str:
        """Reutilise le mecanisme d'etiquetage : la variante apparait dans le nom."""
        return f"{self.name}[{self.condition}]"

    def parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "lookback": self.lookback,
            "filtre": self.condition,
            "seance": self.session,
        }
        if self.condition == "adx":
            params["adx_min"] = self.adx_min
        elif self.condition == "compression":
            params["ratio_max"] = self.ratio_max
        elif self.condition == "spread_bas":
            params["quantile_spread"] = self.spread_quantile
            params["fenetre_quantile"] = self.spread_window
        return params

    def _condition_mask(self, ctx: Context, index: pd.DatetimeIndex) -> pd.Series:
        """Filtre de contexte, evalue sur des informations deja connues."""
        if self.condition == "adx":
            return ctx.features["adx"] >= self.adx_min
        if self.condition == "compression":
            # Decalage d'une barre : la compression doit PRECEDER la sortie, sinon on
            # ne fait que decrire la sortie elle-meme.
            return ctx.features["atr_ratio"].shift(1) <= self.ratio_max
        if self.condition == "spread_bas":
            spread = ctx.cost.spread.reindex(index)
            # Quantile GLISSANT et decale : un quantile calcule sur tout l'historique
            # ferait entrer le futur dans la decision.
            threshold = spread.shift(1).rolling(self.spread_window).quantile(self.spread_quantile)
            return spread <= threshold
        return pd.Series(True, index=index)

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]
        highest = indicators.rolling_high(ctx.df, self.lookback)
        lowest = indicators.rolling_low(ctx.df, self.lookback)

        above = close > highest
        below = close < lowest
        # Premiere barre de la cassure uniquement : sans cela, chaque barre au-dessus
        # de l'extreme redeclencherait un signal.
        first_above = above & ~above.shift(1, fill_value=False)
        first_below = below & ~below.shift(1, fill_value=False)

        allowed = self._condition_mask(ctx, index)
        if self.session is not None:
            window = next(w for w in sessions.SESSIONS if w.name == self.session)
            allowed = allowed & window.mask(index)

        raw = np.where(first_above & allowed, 1, np.where(first_below & allowed, -1, 0))
        return pd.Series(raw, index=index, name="signal")
