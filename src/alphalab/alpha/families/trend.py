"""Familles de tendance : momentum conditionne, expansion apres compression.

La cassure nue a deja ete testee et rejetee dans l'historique de ce projet, sur deux
instruments. On ne la retente donc PAS a l'identique. La question ouverte, et
differente, est celle du conditionnement : une cassure se comporte-t-elle autrement
selon le regime de tendance et la seance ? Un resultat negatif ici serait une reponse
franche, pas une repetition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators, sessions


class ConditionedMomentum(AlphaFamily):
    """Une cassure filtree par la force de tendance et la seance vaut-elle mieux ?"""

    name = "momentum_conditionne"
    question = (
        "La cassure d'un extreme recent devient-elle exploitable lorsqu'elle est "
        "filtree par un ADX eleve et restreinte aux seances liquides ?"
    )

    def __init__(
        self, lookback: int = 24, adx_min: float = 25.0, session: str = "newyork"
    ) -> None:
        self.lookback = lookback
        self.adx_min = adx_min
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"lookback": self.lookback, "adx_min": self.adx_min, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]
        hh = indicators.rolling_high(ctx.df, self.lookback)
        ll = indicators.rolling_low(ctx.df, self.lookback)

        above = close > hh
        below = close < ll
        first_above = above & ~above.shift(1, fill_value=False)
        first_below = below & ~below.shift(1, fill_value=False)

        strong = ctx.features["adx"] >= self.adx_min
        window = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        allowed = strong & window

        raw = np.where(first_above & allowed, 1, np.where(first_below & allowed, -1, 0))
        return pd.Series(raw, index=index, name="signal")


class VolatilityExpansion(AlphaFamily):
    """Une compression de volatilite annonce-t-elle une expansion directionnelle ?"""

    name = "compression_expansion"
    question = (
        "Apres une contraction de l'ATR court par rapport a l'ATR long, la premiere "
        "sortie de la plage recente se poursuit-elle ?"
    )

    def __init__(
        self, ratio_max: float = 0.8, lookback: int = 12, session: str = "newyork"
    ) -> None:
        self.ratio_max = ratio_max
        self.lookback = lookback
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"ratio_max": self.ratio_max, "lookback": self.lookback, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]
        # Compression mesuree sur la barre PRECEDENTE : la compression doit exister
        # avant la sortie, sinon on decrit la sortie elle-meme.
        compressed = ctx.features["atr_ratio"].shift(1) <= self.ratio_max
        hh = indicators.rolling_high(ctx.df, self.lookback)
        ll = indicators.rolling_low(ctx.df, self.lookback)

        window = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        allowed = compressed & window

        raw = np.where((close > hh) & allowed, 1, np.where((close < ll) & allowed, -1, 0))
        signal = pd.Series(raw, index=index, name="signal")
        # Une seule entree par episode de compression.
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class TrendPullback(AlphaFamily):
    """Un repli dans une tendance etablie offre-t-il une entree meilleure que la cassure ?"""

    name = "repli_de_tendance"
    question = (
        "Dans une tendance definie par les moyennes mobiles, un repli momentane du RSI "
        "vers la zone neutre offre-t-il une entree exploitable ?"
    )

    def __init__(
        self, rsi_long: float = 45.0, rsi_short: float = 55.0, session: str = "newyork"
    ) -> None:
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"rsi_long": self.rsi_long, "rsi_short": self.rsi_short, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        uptrend = ctx.features["ema50"] > ctx.features["ema200"]
        downtrend = ctx.features["ema50"] < ctx.features["ema200"]
        rsi = ctx.features["rsi14"]

        # Traversee ascendante du seuil : le repli doit etre TERMINE, pas en cours.
        cross_up = (rsi > self.rsi_long) & (rsi.shift(1) <= self.rsi_long)
        cross_down = (rsi < self.rsi_short) & (rsi.shift(1) >= self.rsi_short)

        window = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        raw = np.where(
            uptrend & cross_up & window, 1, np.where(downtrend & cross_down & window, -1, 0)
        )
        return pd.Series(raw, index=index, name="signal")
