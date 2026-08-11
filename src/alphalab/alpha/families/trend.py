"""Repli de tendance.

Ce module ne contient plus qu'une famille. Les deux autres — momentum conditionne et
compression suivie d'expansion — reposaient sur le meme declencheur que la cassure
(une cloture depassant un extreme recent) et ont ete fusionnees dans
`families/breakout.py`, ou le filtre est devenu un parametre declare.

Le repli de tendance reste ici parce que son declencheur est different : il attend une
traversee de seuil du RSI dans une tendance etablie, et ne regarde aucun extreme.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import sessions


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
